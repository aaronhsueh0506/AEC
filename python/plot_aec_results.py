"""
Run AEC on AEC Challenge format dataset and plot:
  - Waveforms (mic, output)
  - ERLE over time
  - Estimated impulse response (vs true IR if available)

Reads files from gen_sim_data.py output (AEC Challenge naming):
  farend_speech_fileid_N.wav
  nearend_mic_fileid_N.wav
  nearend_speech_fileid_N.wav  (optional, for reference)
  echo_fileid_N.wav            (optional)

Usage:
    python3 plot_aec_results.py ../wav/ [--mode nlms|lms|freq|subband] [--no-dtd]
"""

import numpy as np
import soundfile as sf
import matplotlib.pyplot as plt
import argparse
import os
import re
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from aec import AEC, AecConfig, AecMode, NlmsFilter

# Try importing true RIR from gen_sim_data
try:
    from gen_sim_data import make_rir as make_true_rir
    HAS_TRUE_RIR = True
except ImportError:
    HAS_TRUE_RIR = False


def scan_fileids(dataset_dir):
    """Scan for AEC Challenge file groups, return list of (fileid, paths dict)."""
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
        nearend_speech = p / f'nearend_speech_fileid_{fid}.wav'
        echo = p / f'echo_fileid_{fid}.wav'
        groups.append({
            'fileid': fid,
            'mic': str(mic_file),
            'ref': str(farend),
            'nearend_speech': str(nearend_speech) if nearend_speech.exists() else None,
            'echo': str(echo) if echo.exists() else None,
        })
    return groups


def run_aec(mic_path, ref_path, mode, enable_dtd=True, mu=0.3, filter_length=512):
    """Run AEC and return output + filter object."""
    mic, sr = sf.read(mic_path)
    ref, _ = sf.read(ref_path)
    mic = mic.astype(np.float32)
    ref = ref.astype(np.float32)

    if mode == AecMode.LMS and mu == 0.3:
        mu = 0.01

    config = AecConfig(
        sample_rate=sr,
        mode=mode,
        enable_dtd=enable_dtd,
        mu=mu,
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

    return mic[:n], ref[:n], output, aec, sr


def get_estimated_ir(aec):
    """Extract estimated impulse response from AEC filter."""
    if isinstance(aec.filter, NlmsFilter):
        return aec.filter.weights.copy()
    else:
        W = aec.filter.W
        block_size = aec.filter.block_size
        hop = aec.filter.hop_size
        ir = np.zeros(W.shape[0] * hop, dtype=np.float32)
        for p in range(W.shape[0]):
            w_time = np.fft.irfft(W[p], block_size)
            ir[p * hop:(p + 1) * hop] = w_time[:hop]
        return ir


def main():
    parser = argparse.ArgumentParser(
        description='Plot AEC results with estimated impulse response')
    parser.add_argument('dataset_dir', nargs='?', default=None,
                        help='Directory with AEC Challenge wav files (default: ../wav/)')
    parser.add_argument('--mode', choices=['lms', 'nlms', 'freq', 'subband'],
                        default='nlms')
    parser.add_argument('--no-dtd', action='store_true', help='Disable DTD')
    parser.add_argument('--mu', type=float, default=0.3)
    parser.add_argument('--filter', type=int, default=512,
                        help='Filter length in samples')
    args = parser.parse_args()

    mode_map = {'lms': AecMode.LMS, 'nlms': AecMode.NLMS,
                'freq': AecMode.FREQ, 'subband': AecMode.SUBBAND}
    mode = mode_map[args.mode]

    # Default dataset dir
    base = os.path.dirname(os.path.abspath(__file__))
    dataset_dir = args.dataset_dir or os.path.join(base, '..', 'wav')
    dataset_dir = os.path.abspath(dataset_dir)

    # Scan for files
    groups = scan_fileids(dataset_dir)
    if not groups:
        print(f"No AEC Challenge files found in {dataset_dir}")
        print("Run: python3 gen_sim_data.py first")
        sys.exit(1)

    print(f"Found {len(groups)} file(s) in {dataset_dir}")

    # True IR (from gen_sim_data defaults)
    true_ir = None
    if HAS_TRUE_RIR:
        true_ir = make_true_rir(delay=200, gain=0.8, n_taps=512)

    # Plot
    n_sets = len(groups)
    fig, axes = plt.subplots(n_sets, 3, figsize=(16, 4 * n_sets))
    if n_sets == 1:
        axes = axes[np.newaxis, :]

    for row, group in enumerate(groups):
        fid = group['fileid']
        print(f"  Processing fileid_{fid} ...", end='', flush=True)

        mic, ref, out, aec, sr = run_aec(
            group['mic'], group['ref'], mode,
            enable_dtd=not args.no_dtd,
            mu=args.mu, filter_length=args.filter)
        est_ir = get_estimated_ir(aec)
        print(f" done (ERLE={aec.get_erle():.1f} dB)")

        t = np.arange(len(mic)) / sr

        # ── Col 0: waveforms ────────────────────────────────────────
        ax = axes[row, 0]
        ax.plot(t, mic, alpha=0.6, label='mic')
        ax.plot(t, out, alpha=0.8, label='AEC output')
        ax.set_title(f'fileid_{fid} — Waveforms')
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Amplitude')
        ax.legend(loc='upper right', fontsize=8)
        ax.set_xlim(0, t[-1])

        # ── Col 1: ERLE ────────────────────────────────────────────
        ax = axes[row, 1]
        frame_len = 256
        n_frames = len(mic) // frame_len
        mic_energy = np.array([np.mean(mic[i*frame_len:(i+1)*frame_len]**2)
                               for i in range(n_frames)])
        out_energy = np.array([np.mean(out[i*frame_len:(i+1)*frame_len]**2)
                               for i in range(n_frames)])
        t_frames = np.arange(n_frames) * frame_len / sr
        eps = 1e-10
        # Only compute ERLE where mic has energy
        erle = np.where(mic_energy > eps,
                        10 * np.log10(mic_energy / (out_energy + eps)),
                        0.0)
        ax.plot(t_frames, erle)
        ax.set_title(f'fileid_{fid} — ERLE (dB)')
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('ERLE (dB)')
        ax.set_xlim(0, t_frames[-1])
        ax.axhline(0, color='gray', ls='--', lw=0.5)

        # ── Col 2: estimated IR (+ true IR if available) ───────────
        ax = axes[row, 2]
        t_est = np.arange(len(est_ir)) / sr * 1000  # ms
        ax.plot(t_est, est_ir, label='Estimated IR', lw=1.0, alpha=0.8)
        if true_ir is not None:
            t_true = np.arange(len(true_ir)) / sr * 1000
            ax.plot(t_true, true_ir, label='True IR', lw=1.5, alpha=0.6)
        ax.set_title(f'fileid_{fid} — Impulse Response')
        ax.set_xlabel('Time (ms)')
        ax.set_ylabel('Amplitude')
        ax.legend(fontsize=8)
        xlim = len(est_ir) / sr * 1000 * 0.6
        if true_ir is not None:
            xlim = max(xlim, len(true_ir) / sr * 1000)
        ax.set_xlim(0, xlim)

    dtd_str = 'DTD off' if args.no_dtd else 'DTD on'
    fig.suptitle(f'AEC Results (mode={args.mode}, {dtd_str})', fontsize=14, y=1.01)
    plt.tight_layout()

    dtd_tag = '_no_dtd' if args.no_dtd else ''
    out_path = os.path.join(base, f'aec_results_{args.mode}{dtd_tag}.png')
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    print(f"Saved: {out_path}")
    plt.show()


if __name__ == '__main__':
    main()
