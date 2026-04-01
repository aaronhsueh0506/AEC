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
    python3 plot_aec_results.py ../wav/ [--mode nlms|lms|fdaf|pbfdaf|pbfdkf] [--no-dtd]
    python3 plot_aec_results.py --mic mic.wav --ref ref.wav [--rir rir.wav] [--mode nlms] [--no-dtd]
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


def run_aec(mic_path, ref_path, mode, enable_dtd=True, enable_res=False,
            enable_shadow=False, mu=0.3, filter_length=0):
    """Run AEC and return output + filter object + confidence history."""
    mic, sr = sf.read(mic_path)
    ref, _ = sf.read(ref_path)
    mic = mic.astype(np.float32)
    ref = ref.astype(np.float32)

    if mode == AecMode.LMS and mu == 0.3:
        mu = 0.01

    if filter_length == 0:
        filter_length = 512

    config = AecConfig(
        sample_rate=sr,
        mode=mode,
        enable_dtd=enable_dtd,
        enable_res=enable_res,
        enable_shadow=enable_shadow,
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


def draw_dtd_spans(ax, confidence_history, hop_size, sr):
    """Draw red background spans where DTD confidence > 0 (not updating)."""
    if not confidence_history:
        return
    conf = np.array(confidence_history)
    # Find contiguous regions where confidence > 0
    active = conf > 0.0
    if not np.any(active):
        return
    # Find edges
    diff = np.diff(active.astype(int))
    starts = np.where(diff == 1)[0] + 1
    ends = np.where(diff == -1)[0] + 1
    # Handle edge cases
    if active[0]:
        starts = np.concatenate(([0], starts))
    if active[-1]:
        ends = np.concatenate((ends, [len(conf)]))
    # Draw spans with alpha proportional to max confidence in region
    for s, e in zip(starts, ends):
        t_start = s * hop_size / sr
        t_end = e * hop_size / sr
        max_conf = np.max(conf[s:e])
        alpha = 0.1 + 0.3 * max_conf  # 0.1 ~ 0.4
        ax.axvspan(t_start, t_end, color='red', alpha=alpha, zorder=0)


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
    parser.add_argument('--mic', type=str, default=None,
                        help='Path to mic (near-end) wav file')
    parser.add_argument('--ref', type=str, default=None,
                        help='Path to ref (far-end) wav file')
    parser.add_argument('--mode', choices=['lms', 'nlms', 'fdaf', 'pbfdaf', 'pbfdkf', 'subband'],
                        default='nlms')
    parser.add_argument('--no-dtd', action='store_true', help='Disable DTD')
    parser.add_argument('--enable-res', action='store_true', help='Enable RES post-filter')
    parser.add_argument('--enable-shadow', action='store_true', help='Enable shadow filter (dual-filter)')
    parser.add_argument('--mu', type=float, default=0.3)
    parser.add_argument('--filter', type=int, default=0,
                        help='Filter length in samples (0=mode default)')
    parser.add_argument('--rir', type=str, default=None,
                        help='Path to target RIR wav file (for synthetic data)')
    parser.add_argument('--save-wav', action='store_true',
                        help='Save AEC output wav files (no-RES and RES)')
    parser.add_argument('--files', type=str, default=None,
                        help='Comma-separated fileid list (e.g. "450,50")')
    args = parser.parse_args()

    if (args.mic is None) != (args.ref is None):
        parser.error('--mic and --ref must be specified together')


    mode_map = {'lms': AecMode.LMS, 'nlms': AecMode.NLMS,
                'fdaf': AecMode.FDAF, 'pbfdaf': AecMode.PBFDAF,
                'pbfdkf': AecMode.PBFDKF, 'subband': AecMode.PBFDKF}
    mode = mode_map[args.mode]

    base = os.path.dirname(os.path.abspath(__file__))

    if args.mic and args.ref:
        # Direct mic/ref mode
        mic_path = os.path.abspath(args.mic)
        ref_path = os.path.abspath(args.ref)
        if not os.path.isfile(mic_path):
            print(f"Mic file not found: {mic_path}")
            sys.exit(1)
        if not os.path.isfile(ref_path):
            print(f"Ref file not found: {ref_path}")
            sys.exit(1)
        label = Path(mic_path).stem
        groups = [{
            'fileid': label,
            'mic': mic_path,
            'ref': ref_path,
            'nearend_speech': None,
            'echo': None,
        }]
        print(f"Using mic={mic_path}, ref={ref_path}")
    else:
        # Dataset directory mode
        dataset_dir = args.dataset_dir or os.path.join(base, '..', 'wav')
        dataset_dir = os.path.abspath(dataset_dir)

        groups = scan_fileids(dataset_dir)
        if args.files:
            selected = set(args.files.split(','))
            groups = [g for g in groups if g['fileid'] in selected]
        if not groups:
            print(f"No AEC Challenge files found in {dataset_dir}")
            print("Run: python3 gen_sim_data.py first")
            sys.exit(1)

        print(f"Found {len(groups)} file(s) in {dataset_dir}")

    # Target RIR: from --rir file, or from gen_sim_data defaults
    true_ir = None
    if args.rir:
        rir_path = os.path.abspath(args.rir)
        if not os.path.isfile(rir_path):
            print(f"RIR file not found: {rir_path}")
            sys.exit(1)
        true_ir, _ = sf.read(rir_path)
        true_ir = true_ir.astype(np.float32)
        print(f"Using target RIR: {rir_path} ({len(true_ir)} taps)")
    elif HAS_TRUE_RIR:
        true_ir = make_true_rir(delay=200, gain=0.8, n_taps=512)

    # Each fileid → separate figure with 4 rows (far, mic, output, RIR)
    rows_per_file = 4

    for gi, group in enumerate(groups):
        fid = group['fileid']
        print(f"  Processing fileid_{fid} ...", end='', flush=True)

        # Run without RES
        mic, ref, out_no_res, aec_no_res, sr = run_aec(
            group['mic'], group['ref'], mode,
            enable_dtd=not args.no_dtd,
            enable_res=False,
            enable_shadow=args.enable_shadow,
            mu=args.mu, filter_length=args.filter)
        erle_no_res = aec_no_res.get_erle()

        # Run with RES
        _, _, out_res, aec_res, _ = run_aec(
            group['mic'], group['ref'], mode,
            enable_dtd=not args.no_dtd,
            enable_res=True,
            enable_shadow=args.enable_shadow,
            mu=args.mu, filter_length=args.filter)
        erle_res = aec_res.get_erle()
        print(f" done (no-RES={erle_no_res:.1f}, RES={erle_res:.1f} dB)")

        t = np.arange(len(mic)) / sr
        ymax = max(np.max(np.abs(mic)), np.max(np.abs(ref)))
        ylim = (-ymax * 1.05, ymax * 1.05)

        fig, axes = plt.subplots(rows_per_file, 1, figsize=(16, 9),
                                 gridspec_kw={'height_ratios': [1, 1, 1, 0.8]})
        # Share x-axis for waveform rows (0-2)
        axes[1].sharex(axes[0])
        axes[2].sharex(axes[0])

        # Row 0: Far-end (reference)
        ax = axes[0]
        ax.plot(t, ref, color='green', linewidth=0.4, alpha=0.8)
        ax.set_ylabel('Far-end', fontsize=9)
        dtd_str = 'DTD off' if args.no_dtd else 'DTD on'
        shadow_str = ', Shadow on' if args.enable_shadow else ''
        ax.set_title(f'fileid_{fid} (mode={args.mode}, FL={args.filter}, {dtd_str}{shadow_str})',
                     fontsize=12, fontweight='bold', loc='left')
        ax.set_ylim(ylim)
        ax.grid(True, alpha=0.2)

        # Row 1: Mic (near-end + echo)
        ax = axes[1]
        ax.plot(t, mic, color='royalblue', linewidth=0.4, alpha=0.8)
        ax.set_ylabel('Mic', fontsize=9)
        ax.set_ylim(ylim)
        ax.grid(True, alpha=0.2)

        # Row 2: AEC output — overlay no-RES vs RES
        ax = axes[2]
        if not args.no_dtd:
            draw_dtd_spans(ax, aec_res.confidence_history, aec_res.hop_size, sr)
        ax.plot(t, out_no_res, color='orange', linewidth=0.4, alpha=0.7,
                label=f'No RES (ERLE={erle_no_res:.1f} dB)')
        ax.plot(t, out_res, color='darkgreen', linewidth=0.4, alpha=0.8,
                label=f'RES on (ERLE={erle_res:.1f} dB)')
        ax.set_ylabel('AEC Output', fontsize=9)
        ax.set_ylim(ylim)
        ax.legend(loc='upper right', fontsize=8)
        ax.grid(True, alpha=0.2)

        # Row 3: Estimated RIR (vs target if available)
        ax = axes[3]
        est_ir = get_estimated_ir(aec_res)
        t_ir = np.arange(len(est_ir)) / sr * 1000  # ms
        if true_ir is not None:
            t_true = np.arange(len(true_ir)) / sr * 1000
            ax.plot(t_true, true_ir, color='royalblue', linewidth=0.8, alpha=0.6,
                    label='Target RIR')
        ax.plot(t_ir, est_ir, color='red', linewidth=0.6, alpha=0.8,
                label='Estimated RIR')
        ax.set_ylabel('RIR', fontsize=9)
        ax.set_xlabel('Time (ms)')
        ax.legend(loc='upper right', fontsize=8)
        ax.grid(True, alpha=0.2)

        plt.tight_layout()

        dtd_tag = '_no_dtd' if args.no_dtd else ''
        shadow_tag = '_shadow' if args.enable_shadow else ''
        out_path = os.path.join(base,
            f'aec_results_{args.mode}{dtd_tag}_res_compare{shadow_tag}_fileid_{fid}.png')
        plt.savefig(out_path, dpi=150, bbox_inches='tight')
        print(f"  Saved: {out_path}")
        plt.close(fig)

        if args.save_wav:
            wav_no_res = os.path.join(base,
                f'aec_out_{args.mode}{dtd_tag}{shadow_tag}_nores_{fid}.wav')
            wav_res = os.path.join(base,
                f'aec_out_{args.mode}{dtd_tag}{shadow_tag}_res_{fid}.wav')
            sf.write(wav_no_res, out_no_res, sr)
            sf.write(wav_res, out_res, sr)
            print(f"  Saved: {wav_no_res}")
            print(f"  Saved: {wav_res}")


if __name__ == '__main__':
    main()
