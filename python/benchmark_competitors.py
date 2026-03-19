"""
Benchmark AEC competitors on AEC Challenge dataset.

Compares:
  - Our AEC (subband DTD+RES)
  - SpeexDSP echo canceller
  - WebRTC AEC3 (via compiled CLI)

Usage:
    python3 benchmark_competitors.py ../wav/ [--filter 1024]
"""

import numpy as np
import soundfile as sf
import argparse
import os
import re
import sys
import subprocess
import tempfile
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from aec import AEC, AecConfig, AecMode

# Try SpeexDSP
try:
    from speexdsp import EchoCanceller
    HAS_SPEEX = True
except ImportError:
    HAS_SPEEX = False
    print("Warning: speexdsp not installed. pip3 install speexdsp")

# Try WebRTC AEC3 CLI
AEC3_CLI = '/tmp/webrtc-ap/aec3_cli'
HAS_AEC3 = os.path.isfile(AEC3_CLI) and os.access(AEC3_CLI, os.X_OK)
if not HAS_AEC3:
    print("Warning: WebRTC AEC3 CLI not found at", AEC3_CLI)


def scan_fileids(dataset_dir):
    p = Path(dataset_dir)
    mic_files = sorted(p.glob('nearend_mic_fileid_*.wav'))
    groups = []
    for mic_file in mic_files:
        match = re.search(r'fileid_(\d+)', mic_file.stem)
        if not match:
            continue
        fid = match.group(1)
        farend = p / f'farend_speech_fileid_{fid}.wav'
        if not farend.exists():
            continue
        groups.append({
            'fileid': fid,
            'mic': str(mic_file),
            'ref': str(farend),
        })
    return groups


def compute_erle(mic, output):
    """ERLE = 10*log10(mean(mic²) / mean(output²))"""
    mic_pwr = np.mean(mic ** 2)
    out_pwr = np.mean(output ** 2)
    if out_pwr < 1e-20:
        return 60.0
    return 10.0 * np.log10(mic_pwr / (out_pwr + 1e-20))


def run_ours(mic, ref, sr, filter_length=1024, enable_res=True):
    """Run our AEC (subband DTD+RES)."""
    config = AecConfig(
        sample_rate=sr,
        mode=AecMode.SUBBAND,
        enable_dtd=True,
        enable_res=enable_res,
        filter_length=filter_length,
    )
    aec = AEC(config)
    hop = aec.hop_size
    n = min(len(mic), len(ref))
    output = np.zeros(n, dtype=np.float32)
    processed = 0
    while processed + hop <= n:
        out_block = aec.process(mic[processed:processed+hop],
                                ref[processed:processed+hop])
        output[processed:processed+hop] = out_block
        processed += hop
    return output[:n]


def run_speexdsp(mic, ref, sr, filter_length=2048):
    """Run SpeexDSP echo canceller."""
    frame_size = 256
    ec = EchoCanceller.create(frame_size, filter_length, sr)

    n = min(len(mic), len(ref))
    output = np.zeros(n, dtype=np.float32)

    processed = 0
    while processed + frame_size <= n:
        # Convert float32 [-1,1] to int16 bytes
        mic_frame = mic[processed:processed+frame_size]
        ref_frame = ref[processed:processed+frame_size]
        mic_i16 = (mic_frame * 32767).clip(-32768, 32767).astype(np.int16)
        ref_i16 = (ref_frame * 32767).clip(-32768, 32767).astype(np.int16)

        out_bytes = ec.process(mic_i16.tobytes(), ref_i16.tobytes())
        out_i16 = np.frombuffer(out_bytes, dtype=np.int16)
        output[processed:processed+frame_size] = out_i16.astype(np.float32) / 32767.0
        processed += frame_size

    return output[:n]


def run_webrtc_aec3(mic_path, ref_path, sr):
    """Run WebRTC AEC3 via CLI tool."""
    with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp:
        out_path = tmp.name
    try:
        result = subprocess.run(
            [AEC3_CLI, mic_path, ref_path, out_path],
            capture_output=True, text=True, timeout=30)
        if result.returncode != 0:
            return None
        output, _ = sf.read(out_path)
        return output.astype(np.float32)
    except Exception:
        return None
    finally:
        if os.path.exists(out_path):
            os.unlink(out_path)


def main():
    parser = argparse.ArgumentParser(description='Benchmark AEC competitors')
    parser.add_argument('dataset_dir', nargs='?', default=None)
    parser.add_argument('--filter', type=int, default=1024,
                        help='Our filter length (default: 1024)')
    parser.add_argument('--speex-filter', type=int, default=2048,
                        help='SpeexDSP filter length (default: 2048)')
    parser.add_argument('--files', type=str, default=None,
                        help='Comma-separated fileid list')
    args = parser.parse_args()

    base = os.path.dirname(os.path.abspath(__file__))
    dataset_dir = args.dataset_dir or os.path.join(base, '..', 'wav')
    dataset_dir = os.path.abspath(dataset_dir)

    groups = scan_fileids(dataset_dir)
    if args.files:
        selected = set(args.files.split(','))
        groups = [g for g in groups if g['fileid'] in selected]
    if not groups:
        print(f"No files found in {dataset_dir}")
        sys.exit(1)

    print(f"Benchmarking {len(groups)} files from {dataset_dir}")
    print(f"Our config: subband DTD+RES, FL={args.filter}")
    if HAS_SPEEX:
        print(f"SpeexDSP: FL={args.speex_filter}")
    if HAS_AEC3:
        print(f"WebRTC AEC3: {AEC3_CLI}")
    print()

    # Header
    cols = f"{'fileid':>8} {'Ours':>8} {'Ours-NR':>8}"
    if HAS_SPEEX:
        cols += f" {'Speex':>8}"
    if HAS_AEC3:
        cols += f" {'AEC3':>8}"
    cols += f" {'Winner':>8}"
    print(cols)
    print("-" * len(cols))

    results = []
    for group in sorted(groups, key=lambda g: int(g['fileid'])):
        fid = group['fileid']
        mic, sr = sf.read(group['mic'])
        ref, _ = sf.read(group['ref'])
        mic = mic.astype(np.float32)
        ref = ref.astype(np.float32)
        n = min(len(mic), len(ref))
        mic = mic[:n]
        ref = ref[:n]

        # Our AEC with RES
        out_ours = run_ours(mic, ref, sr, filter_length=args.filter, enable_res=True)
        erle_ours = compute_erle(mic, out_ours)

        # Our AEC without RES
        out_ours_nr = run_ours(mic, ref, sr, filter_length=args.filter, enable_res=False)
        erle_ours_nr = compute_erle(mic, out_ours_nr)

        row = {'fileid': fid, 'ours': erle_ours, 'ours_nores': erle_ours_nr}

        erle_speex = None
        if HAS_SPEEX:
            out_speex = run_speexdsp(mic, ref, sr, filter_length=args.speex_filter)
            erle_speex = compute_erle(mic, out_speex)
            row['speex'] = erle_speex

        erle_aec3 = None
        if HAS_AEC3:
            out_aec3 = run_webrtc_aec3(group['mic'], group['ref'], sr)
            if out_aec3 is not None:
                out_aec3 = out_aec3[:n]
                erle_aec3 = compute_erle(mic, out_aec3)
                row['aec3'] = erle_aec3

        results.append(row)

        # Determine winner
        candidates = {'Ours': erle_ours}
        if erle_speex is not None:
            candidates['Speex'] = erle_speex
        if erle_aec3 is not None:
            candidates['AEC3'] = erle_aec3
        winner = max(candidates, key=candidates.get)

        line = f"{fid:>8} {erle_ours:>7.1f}  {erle_ours_nr:>7.1f} "
        if HAS_SPEEX:
            line += f" {erle_speex:>7.1f} " if erle_speex is not None else f" {'N/A':>7} "
        if HAS_AEC3:
            line += f" {erle_aec3:>7.1f} " if erle_aec3 is not None else f" {'N/A':>7} "
        line += f" {winner:>8}"
        print(line)

    # Summary
    print("-" * len(cols))
    ours_mean = np.mean([r['ours'] for r in results])
    ours_nr_mean = np.mean([r['ours_nores'] for r in results])
    summary = f"{'MEAN':>8} {ours_mean:>7.1f}  {ours_nr_mean:>7.1f} "
    if HAS_SPEEX:
        speex_vals = [r['speex'] for r in results if 'speex' in r]
        summary += f" {np.mean(speex_vals):>7.1f} " if speex_vals else f" {'N/A':>7} "
    if HAS_AEC3:
        aec3_vals = [r['aec3'] for r in results if 'aec3' in r]
        summary += f" {np.mean(aec3_vals):>7.1f} " if aec3_vals else f" {'N/A':>7} "
    print(summary)

    ours_med = np.median([r['ours'] for r in results])
    ours_nr_med = np.median([r['ours_nores'] for r in results])
    summary2 = f"{'MEDIAN':>8} {ours_med:>7.1f}  {ours_nr_med:>7.1f} "
    if HAS_SPEEX:
        summary2 += f" {np.median(speex_vals):>7.1f} " if speex_vals else f" {'N/A':>7} "
    if HAS_AEC3:
        summary2 += f" {np.median(aec3_vals):>7.1f} " if aec3_vals else f" {'N/A':>7} "
    print(summary2)

    # Win/Loss summary
    print()
    for comp_name, comp_key in [('Speex', 'speex'), ('AEC3', 'aec3')]:
        comp_results = [r for r in results if comp_key in r]
        if not comp_results:
            continue
        wins_ours = sum(1 for r in comp_results if r['ours'] > r[comp_key])
        wins_comp = sum(1 for r in comp_results if r[comp_key] > r['ours'])
        ties = len(comp_results) - wins_ours - wins_comp
        print(f"vs {comp_name}: Ours {wins_ours} wins, {comp_name} {wins_comp} wins, {ties} ties")


if __name__ == '__main__':
    main()
