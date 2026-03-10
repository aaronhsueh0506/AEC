#!/usr/bin/env python3
"""
AEC 評估腳本

使用 AEC Challenge synthetic dataset 評估 AEC 各模式效果。

指標：
- ERLE (Echo Return Loss Enhancement)
- Echo Suppression (殘餘回音抑制)
- PESQ (語音品質)
- STOI (語音可懂度)
- segSNR improvement (分段信噪比改善)
- Near-end Speech Distortion (近端語音失真)

Dataset 結構（flat directory，用 fileid_N 配對）：
  farend_speech_fileid_0.wav
  nearend_speech_fileid_0.wav
  nearend_mic_fileid_0.wav
  echo_signal_fileid_0.wav

Usage:
  python evaluate_aec.py <dataset_dir> [--mode time|freq|subband|lms|all]
  python evaluate_aec.py <dataset_dir> --mode all --output results.csv
"""

import numpy as np
import os
import sys
import re
import csv
import argparse
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import soundfile as sf

# Add parent directory for aec import
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from aec import AEC, AecConfig, AecMode

# Try importing NR metrics
NR_UTILS_PATH = os.path.join(os.path.dirname(__file__), '..', '..', 'NR')
sys.path.insert(0, NR_UTILS_PATH)

try:
    from utils.metrics import calculate_pesq, calculate_stoi
    EXTERNAL_METRICS = True
except ImportError:
    EXTERNAL_METRICS = False
    # Fallback: optional deps
    try:
        from pesq import pesq as pesq_score
        PESQ_AVAILABLE = True
    except ImportError:
        PESQ_AVAILABLE = False
    try:
        from pystoi import stoi as stoi_score
        STOI_AVAILABLE = True
    except ImportError:
        STOI_AVAILABLE = False


# ============================================================
# AEC-specific metrics
# ============================================================

def calculate_erle(mic_signal: np.ndarray, output_signal: np.ndarray,
                   frame_size: int = 256, hop_size: int = 128) -> Dict[str, float]:
    """
    Calculate ERLE (Echo Return Loss Enhancement) frame-by-frame.

    ERLE = 10 * log10(P_mic / P_output)

    Args:
        mic_signal: Microphone input (near-end + echo)
        output_signal: AEC output
        frame_size: Frame size in samples
        hop_size: Hop size in samples

    Returns:
        Dict with 'mean', 'max', 'median' ERLE in dB
    """
    min_len = min(len(mic_signal), len(output_signal))
    mic_signal = mic_signal[:min_len]
    output_signal = output_signal[:min_len]

    num_frames = (min_len - frame_size) // hop_size + 1
    erle_values = []

    for i in range(num_frames):
        start = i * hop_size
        end = start + frame_size

        mic_power = np.mean(mic_signal[start:end] ** 2)
        out_power = np.mean(output_signal[start:end] ** 2)

        # Skip silent frames
        if mic_power < 1e-10:
            continue

        erle = 10 * np.log10(mic_power / (out_power + 1e-10))
        erle = np.clip(erle, -10.0, 50.0)
        erle_values.append(erle)

    if len(erle_values) == 0:
        return {'mean': 0.0, 'max': 0.0, 'median': 0.0}

    return {
        'mean': float(np.mean(erle_values)),
        'max': float(np.max(erle_values)),
        'median': float(np.median(erle_values)),
    }


def calculate_echo_suppression(echo_signal: np.ndarray,
                                output_signal: np.ndarray,
                                clean_nearend: np.ndarray) -> float:
    """
    Calculate echo suppression level.

    Residual echo = output - clean_nearend
    Echo suppression = 10 * log10(P_echo / P_residual_echo)

    Higher is better (more echo removed).

    Args:
        echo_signal: Pure echo signal
        output_signal: AEC output
        clean_nearend: Clean near-end speech

    Returns:
        Echo suppression in dB
    """
    min_len = min(len(echo_signal), len(output_signal), len(clean_nearend))
    echo_signal = echo_signal[:min_len]
    output_signal = output_signal[:min_len]
    clean_nearend = clean_nearend[:min_len]

    # Residual echo = what remains after subtracting clean near-end from output
    residual_echo = output_signal - clean_nearend

    echo_power = np.mean(echo_signal ** 2)
    residual_power = np.mean(residual_echo ** 2)

    if echo_power < 1e-10:
        return 0.0

    suppression = 10 * np.log10(echo_power / (residual_power + 1e-10))
    return float(suppression)


def calculate_nearend_distortion(clean_nearend: np.ndarray,
                                  output_signal: np.ndarray,
                                  frame_size: int = 256,
                                  hop_size: int = 128) -> float:
    """
    Calculate near-end speech distortion (SDR-like metric).

    SDR = 10 * log10(P_clean / P_distortion)

    Only considers frames where near-end speech is active.
    Higher is better (less distortion).

    Args:
        clean_nearend: Clean near-end speech
        output_signal: AEC output
        frame_size: Frame size in samples
        hop_size: Hop size in samples

    Returns:
        Near-end speech distortion in dB (higher = less distortion)
    """
    min_len = min(len(clean_nearend), len(output_signal))
    clean_nearend = clean_nearend[:min_len]
    output_signal = output_signal[:min_len]

    num_frames = (min_len - frame_size) // hop_size + 1
    sdr_values = []

    for i in range(num_frames):
        start = i * hop_size
        end = start + frame_size

        clean_frame = clean_nearend[start:end]
        output_frame = output_signal[start:end]

        clean_power = np.mean(clean_frame ** 2)

        # Only consider frames where near-end is active
        if clean_power < 1e-8:
            continue

        distortion = output_frame - clean_frame
        dist_power = np.mean(distortion ** 2)

        if dist_power < 1e-10:
            sdr = 35.0
        else:
            sdr = 10 * np.log10(clean_power / dist_power)

        sdr = np.clip(sdr, -10.0, 35.0)
        sdr_values.append(sdr)

    if len(sdr_values) == 0:
        return 0.0

    return float(np.mean(sdr_values))


def calculate_segmental_snr_improvement(
    noisy: np.ndarray,
    clean: np.ndarray,
    enhanced: np.ndarray,
    frame_size: int = 256,
    hop_size: int = 128
) -> Tuple[float, float, float]:
    """
    Calculate segmental SNR improvement.

    Returns:
        (input_segsnr, output_segsnr, improvement) in dB
    """
    def _segsnr(ref, test):
        min_len = min(len(ref), len(test))
        ref = ref[:min_len]
        test = test[:min_len]
        num_frames = (min_len - frame_size) // hop_size + 1
        snrs = []
        for i in range(num_frames):
            s = i * hop_size
            e = s + frame_size
            sig_power = np.mean(ref[s:e] ** 2)
            if sig_power < 1e-10:
                continue
            noise_power = np.mean((test[s:e] - ref[s:e]) ** 2)
            if noise_power < 1e-10:
                snr = 35.0
            else:
                snr = 10 * np.log10(sig_power / noise_power)
            snrs.append(np.clip(snr, -10.0, 35.0))
        return float(np.mean(snrs)) if snrs else 0.0

    input_snr = _segsnr(clean, noisy)
    output_snr = _segsnr(clean, enhanced)
    return input_snr, output_snr, output_snr - input_snr


def _calculate_pesq_local(clean: np.ndarray, enhanced: np.ndarray,
                           fs: int = 16000) -> Optional[float]:
    """Calculate PESQ (fallback if NR metrics not available)."""
    if EXTERNAL_METRICS:
        return calculate_pesq(clean, enhanced, fs)

    if not PESQ_AVAILABLE:
        return None
    min_len = min(len(clean), len(enhanced))
    try:
        return float(pesq_score(fs, clean[:min_len], enhanced[:min_len], 'wb'))
    except Exception:
        return None


def _calculate_stoi_local(clean: np.ndarray, enhanced: np.ndarray,
                           fs: int = 16000) -> Optional[float]:
    """Calculate STOI (fallback if NR metrics not available)."""
    if EXTERNAL_METRICS:
        return calculate_stoi(clean, enhanced, fs)

    if not STOI_AVAILABLE:
        return None
    min_len = min(len(clean), len(enhanced))
    try:
        return float(stoi_score(clean[:min_len], enhanced[:min_len], fs))
    except Exception:
        return None


# ============================================================
# Dataset scanning
# ============================================================

def scan_dataset(dataset_dir: str) -> List[Dict[str, str]]:
    """
    Scan dataset directory for AEC Challenge file groups.

    Looks for files matching: {type}_fileid_{N}.wav
    Groups them by fileid.

    Returns:
        List of dicts, each with keys: 'fileid', 'nearend_mic', 'farend_speech',
        'nearend_speech', 'echo_signal'
    """
    dataset_path = Path(dataset_dir)

    # Find all nearend_mic files to get file IDs
    mic_files = sorted(dataset_path.glob('nearend_mic_fileid_*.wav'))

    if not mic_files:
        # Try alternative: nearend_mic_signal pattern
        mic_files = sorted(dataset_path.glob('nearend_mic_signal_fileid_*.wav'))

    groups = []
    for mic_file in mic_files:
        # Extract fileid
        name = mic_file.stem
        match = re.search(r'fileid_(\d+)', name)
        if not match:
            continue
        fileid = match.group(1)

        # Build expected paths
        nearend_speech = dataset_path / f'nearend_speech_fileid_{fileid}.wav'
        farend_speech = dataset_path / f'farend_speech_fileid_{fileid}.wav'
        echo_signal = dataset_path / f'echo_signal_fileid_{fileid}.wav'

        # Check all exist
        if not nearend_speech.exists():
            print(f"  Warning: missing nearend_speech for fileid_{fileid}")
            continue
        if not farend_speech.exists():
            print(f"  Warning: missing farend_speech for fileid_{fileid}")
            continue
        if not echo_signal.exists():
            print(f"  Warning: missing echo_signal for fileid_{fileid}")
            continue

        groups.append({
            'fileid': fileid,
            'nearend_mic': str(mic_file),
            'farend_speech': str(farend_speech),
            'nearend_speech': str(nearend_speech),
            'echo_signal': str(echo_signal),
        })

    return groups


# ============================================================
# AEC processing + evaluation
# ============================================================

def process_and_evaluate(group: Dict[str, str], config: AecConfig) -> Dict:
    """
    Run AEC on one file group and compute all metrics.

    Args:
        group: Dict with file paths
        config: AEC configuration

    Returns:
        Dict with all metric results
    """
    # Load audio
    mic_data, mic_sr = sf.read(group['nearend_mic'])
    ref_data, ref_sr = sf.read(group['farend_speech'])
    clean_data, clean_sr = sf.read(group['nearend_speech'])
    echo_data, echo_sr = sf.read(group['echo_signal'])

    # Ensure mono
    if mic_data.ndim > 1: mic_data = mic_data[:, 0]
    if ref_data.ndim > 1: ref_data = ref_data[:, 0]
    if clean_data.ndim > 1: clean_data = clean_data[:, 0]
    if echo_data.ndim > 1: echo_data = echo_data[:, 0]

    # Align lengths
    min_len = min(len(mic_data), len(ref_data), len(clean_data), len(echo_data))
    mic_data = mic_data[:min_len].astype(np.float32)
    ref_data = ref_data[:min_len].astype(np.float32)
    clean_data = clean_data[:min_len].astype(np.float32)
    echo_data = echo_data[:min_len].astype(np.float32)

    # Update config sample rate
    config.sample_rate = mic_sr

    # Create AEC and process
    aec = AEC(config)
    hop_size = aec.hop_size

    output = np.zeros(min_len, dtype=np.float32)
    processed = 0

    while processed + hop_size <= min_len:
        mic_block = mic_data[processed:processed + hop_size]
        ref_block = ref_data[processed:processed + hop_size]
        out_block = aec.process(mic_block, ref_block)
        output[processed:processed + hop_size] = out_block
        processed += hop_size

    # Trim to processed length
    output = output[:processed]
    mic_data = mic_data[:processed]
    ref_data = ref_data[:processed]
    clean_data = clean_data[:processed]
    echo_data = echo_data[:processed]

    # Calculate metrics
    results = {}
    results['fileid'] = group['fileid']

    # 1. ERLE
    erle = calculate_erle(mic_data, output)
    results['erle_mean'] = erle['mean']
    results['erle_max'] = erle['max']
    results['erle_median'] = erle['median']

    # 2. Echo Suppression
    results['echo_suppression'] = calculate_echo_suppression(echo_data, output, clean_data)

    # 3. Near-end Speech Distortion
    results['nearend_sdr'] = calculate_nearend_distortion(clean_data, output)

    # 4. segSNR improvement
    input_segsnr, output_segsnr, segsnr_imp = calculate_segmental_snr_improvement(
        mic_data, clean_data, output
    )
    results['input_segsnr'] = input_segsnr
    results['output_segsnr'] = output_segsnr
    results['segsnr_improvement'] = segsnr_imp

    # 5. PESQ
    results['pesq'] = _calculate_pesq_local(clean_data, output, mic_sr)

    # 6. STOI
    results['stoi'] = _calculate_stoi_local(clean_data, output, mic_sr)

    return results


def print_results(results: List[Dict], mode_name: str):
    """Print results table to console."""
    if not results:
        print("No results.")
        return

    print(f"\n{'='*80}")
    print(f"AEC Evaluation Results — Mode: {mode_name}")
    print(f"{'='*80}")

    # Header
    header = (f"{'ID':>6} | {'ERLE':>7} | {'EchoSup':>7} | {'NE-SDR':>7} | "
              f"{'segSNR':>7} | {'Improv':>7} | {'PESQ':>6} | {'STOI':>6}")
    units = (f"{'':>6} | {'(dB)':>7} | {'(dB)':>7} | {'(dB)':>7} | "
             f"{'(dB)':>7} | {'(dB)':>7} | {'':>6} | {'':>6}")
    print(header)
    print(units)
    print('-' * 80)

    # Rows
    for r in results:
        pesq_str = f"{r['pesq']:.2f}" if r['pesq'] is not None else 'N/A'
        stoi_str = f"{r['stoi']:.3f}" if r['stoi'] is not None else 'N/A'
        print(f"{r['fileid']:>6} | {r['erle_mean']:7.1f} | {r['echo_suppression']:7.1f} | "
              f"{r['nearend_sdr']:7.1f} | {r['output_segsnr']:7.1f} | "
              f"{r['segsnr_improvement']:7.1f} | {pesq_str:>6} | {stoi_str:>6}")

    # Average
    print('-' * 80)
    n = len(results)
    avg_erle = np.mean([r['erle_mean'] for r in results])
    avg_echo_sup = np.mean([r['echo_suppression'] for r in results])
    avg_sdr = np.mean([r['nearend_sdr'] for r in results])
    avg_segsnr = np.mean([r['output_segsnr'] for r in results])
    avg_imp = np.mean([r['segsnr_improvement'] for r in results])

    pesq_vals = [r['pesq'] for r in results if r['pesq'] is not None]
    stoi_vals = [r['stoi'] for r in results if r['stoi'] is not None]
    avg_pesq_str = f"{np.mean(pesq_vals):.2f}" if pesq_vals else 'N/A'
    avg_stoi_str = f"{np.mean(stoi_vals):.3f}" if stoi_vals else 'N/A'

    print(f"{'AVG':>6} | {avg_erle:7.1f} | {avg_echo_sup:7.1f} | "
          f"{avg_sdr:7.1f} | {avg_segsnr:7.1f} | "
          f"{avg_imp:7.1f} | {avg_pesq_str:>6} | {avg_stoi_str:>6}")
    print(f"{'='*80}\n")


def save_csv(results: List[Dict], output_path: str, mode_name: str):
    """Save results to CSV."""
    if not results:
        return

    fieldnames = ['mode', 'fileid', 'erle_mean', 'erle_max', 'erle_median',
                  'echo_suppression', 'nearend_sdr',
                  'input_segsnr', 'output_segsnr', 'segsnr_improvement',
                  'pesq', 'stoi']

    file_exists = os.path.exists(output_path)

    with open(output_path, 'a', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        for r in results:
            row = {k: r.get(k, '') for k in fieldnames}
            row['mode'] = mode_name
            writer.writerow(row)

    print(f"Results appended to {output_path}")


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description='AEC Evaluation using AEC Challenge synthetic dataset',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python evaluate_aec.py ./synthetic_data --mode time
    python evaluate_aec.py ./synthetic_data --mode all --output results.csv
    python evaluate_aec.py ./synthetic_data --mode subband --filter 1024 --mu 0.3
        """
    )
    parser.add_argument('dataset_dir', help='Directory containing AEC Challenge wav files')
    parser.add_argument('--mode', choices=['lms', 'time', 'freq', 'subband', 'all'],
                        default='time', help='Filter mode (default: time)')
    parser.add_argument('--mu', type=float, default=None, help='Step size override')
    parser.add_argument('--filter', type=int, default=None, help='Filter length in samples')
    parser.add_argument('--no-dtd', action='store_true', help='Disable DTD')
    parser.add_argument('--clear-history', action='store_true', help='Clear TIME/LMS history each block')
    parser.add_argument('--output', type=str, default=None, help='Output CSV file path')
    parser.add_argument('--limit', type=int, default=0, help='Limit number of test cases (0=all)')

    args = parser.parse_args()

    # Scan dataset
    print(f"Scanning dataset: {args.dataset_dir}")
    groups = scan_dataset(args.dataset_dir)

    if not groups:
        print("Error: No valid file groups found.")
        print("Expected files: nearend_mic_fileid_N.wav, farend_speech_fileid_N.wav, "
              "nearend_speech_fileid_N.wav, echo_signal_fileid_N.wav")
        sys.exit(1)

    print(f"Found {len(groups)} test cases")

    if args.limit > 0:
        groups = groups[:args.limit]
        print(f"Limited to {args.limit} cases")

    # Determine modes to run
    mode_map = {
        'lms': AecMode.LMS,
        'time': AecMode.TIME,
        'freq': AecMode.FREQ,
        'subband': AecMode.SUBBAND,
    }

    if args.mode == 'all':
        modes_to_run = ['time', 'freq', 'subband', 'lms']
    else:
        modes_to_run = [args.mode]

    # Clear CSV if exists and running fresh
    if args.output and os.path.exists(args.output):
        os.remove(args.output)

    # Run evaluation for each mode
    for mode_name in modes_to_run:
        aec_mode = mode_map[mode_name]

        # Build config
        mu = args.mu
        if mu is None:
            mu = 0.01 if mode_name == 'lms' else 0.3

        filter_length = args.filter
        if filter_length is None:
            if mode_name == 'subband':
                filter_length = 1024
            else:
                filter_length = 512  # frame_size default

        config = AecConfig(
            mode=aec_mode,
            mu=mu,
            filter_length=filter_length,
            enable_dtd=not args.no_dtd,
            clear_filter_history=args.clear_history,
        )

        print(f"\nProcessing mode: {mode_name} (mu={mu}, filter={filter_length})")

        results = []
        for i, group in enumerate(groups):
            print(f"  [{i+1}/{len(groups)}] fileid_{group['fileid']}...", end='', flush=True)
            try:
                r = process_and_evaluate(group, config)
                results.append(r)
                print(f" ERLE={r['erle_mean']:.1f}dB, PESQ={r['pesq']}" if r['pesq'] else
                      f" ERLE={r['erle_mean']:.1f}dB")
            except Exception as e:
                print(f" Error: {e}")

        # Print results
        print_results(results, mode_name)

        # Save CSV
        if args.output:
            save_csv(results, args.output, mode_name)


if __name__ == '__main__':
    main()
