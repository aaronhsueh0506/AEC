#!/usr/bin/env python3
"""Run AEC on one mic/ref pair, save output WAV + diagnostic plot.

Usage:
    python3 python/run_one_case.py mic.wav ref.wav out.wav [--preset balanced]
                                   [--cng] [--filter 832] [--no-res]
                                   [--plot out.png]

The plot panels show:
    1. Mic (near-end input) waveform
    2. Reference (far-end / loopback) waveform
    3. AEC output waveform
    4. Mic vs output spectrograms (side-by-side)
    5. Per-frame ERLE (dB) over time

If --plot is omitted, the PNG is written next to the output WAV with a
.png suffix (e.g. out.wav -> out.png). matplotlib is required for the
plot; if unavailable, the WAV is still written and a warning is logged.
"""
import argparse
import os
import sys

import numpy as np
import soundfile as sf

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from aec import AEC, AecConfig, AecMode, AecPreset


PRESET_MAP = {
    'mild':       AecPreset.MILD,
    'balanced':   AecPreset.BALANCED,
    'aggressive': AecPreset.AGGRESSIVE,
    'maximum':    AecPreset.MAXIMUM,
}


def run_aec(mic_path, ref_path, out_path, *, preset='balanced',
            filter_length=832, enable_cng=True, enable_res=True,
            sample_rate=16000):
    """Process one case; return (mic, ref, out, erle_per_frame, sample_rate)."""
    cfg = AecConfig.from_preset(
        PRESET_MAP[preset],
        sample_rate=sample_rate,
        filter_length=filter_length,
        mode=AecMode.PBFDKF,
        enable_res=enable_res,
        enable_cng=enable_cng,
        enable_shadow=True,
    )
    aec = AEC(cfg)

    mic, sr_mic = sf.read(mic_path)
    ref, sr_ref = sf.read(ref_path)
    if sr_mic != sample_rate or sr_ref != sample_rate:
        raise ValueError(
            f"sample rate mismatch: mic={sr_mic} ref={sr_ref} expected={sample_rate}"
        )
    n = min(len(mic), len(ref))
    mic = mic[:n].astype(np.float32)
    ref = ref[:n].astype(np.float32)

    hop = aec.hop_size
    out = np.zeros(n, dtype=np.float32)
    erle_log = []
    pos = 0
    while pos + hop <= n:
        block = aec.process(mic[pos:pos + hop], ref[pos:pos + hop])
        out[pos:pos + hop] = block
        erle_log.append(aec.get_erle_instant())
        pos += hop

    sf.write(out_path, out, sample_rate)
    return mic[:pos], ref[:pos], out[:pos], np.asarray(erle_log), sample_rate


def make_plot(mic, ref, out, erle, sample_rate, png_path, title=''):
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError:
        print('warning: matplotlib not available — skipping plot', file=sys.stderr)
        return False

    n = len(mic)
    t = np.arange(n) / sample_rate

    fig, axes = plt.subplots(5, 1, figsize=(12, 11), constrained_layout=True)
    fig.suptitle(title or 'AEC single-case diagnostic', fontsize=12)

    axes[0].plot(t, mic, lw=0.5, color='#1f77b4')
    axes[0].set_ylabel('mic')
    axes[0].set_xlim(0, t[-1] if n else 1)

    axes[1].plot(t, ref, lw=0.5, color='#ff7f0e')
    axes[1].set_ylabel('ref')
    axes[1].set_xlim(0, t[-1] if n else 1)

    axes[2].plot(t, out, lw=0.5, color='#2ca02c')
    axes[2].set_ylabel('AEC out')
    axes[2].set_xlim(0, t[-1] if n else 1)
    axes[2].set_xlabel('time (s)')

    nfft = 512
    hop_spec = 256
    def _spec(x):
        from numpy.fft import rfft
        n_blocks = max(0, (len(x) - nfft) // hop_spec + 1)
        if n_blocks <= 0:
            return np.zeros((nfft // 2 + 1, 1)), np.array([0.0])
        S = np.zeros((nfft // 2 + 1, n_blocks), dtype=np.float32)
        win = np.hanning(nfft).astype(np.float32)
        for k in range(n_blocks):
            seg = x[k * hop_spec:k * hop_spec + nfft] * win
            S[:, k] = np.abs(rfft(seg)) + 1e-10
        times = (np.arange(n_blocks) * hop_spec + nfft / 2) / sample_rate
        return 20 * np.log10(S), times
    Smic, t_spec = _spec(mic)
    Sout, _ = _spec(out)
    freqs = np.linspace(0, sample_rate / 2, Smic.shape[0])
    vmin, vmax = -80, 0
    ax_l = axes[3]
    ax_l.imshow(Smic, aspect='auto', origin='lower',
                extent=[t_spec[0], t_spec[-1] if len(t_spec) else 1,
                         freqs[0], freqs[-1]],
                vmin=vmin, vmax=vmax, cmap='magma')
    ax_l.set_ylabel('mic spec\n(Hz)')
    ax_l.set_yticks([0, 2000, 4000, 6000, 8000])

    fig2_inset = axes[3].inset_axes([1.02, 0, 0.5, 1])
    fig2_inset.imshow(Sout, aspect='auto', origin='lower',
                       extent=[t_spec[0], t_spec[-1] if len(t_spec) else 1,
                               freqs[0], freqs[-1]],
                       vmin=vmin, vmax=vmax, cmap='magma')
    fig2_inset.set_title('out spec', fontsize=9)
    fig2_inset.set_yticks([])

    if len(erle):
        t_erle = np.arange(len(erle)) * (n / max(len(erle), 1)) / sample_rate
        axes[4].plot(t_erle, erle, lw=0.7, color='#d62728')
    axes[4].set_ylabel('ERLE (dB)')
    axes[4].set_xlabel('time (s)')
    axes[4].axhline(0, color='#888', lw=0.5)
    axes[4].set_xlim(0, t[-1] if n else 1)

    fig.savefig(png_path, dpi=120)
    plt.close(fig)
    return True


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('mic')
    p.add_argument('ref')
    p.add_argument('out', help='output WAV path')
    p.add_argument('--preset', choices=PRESET_MAP.keys(), default='balanced')
    p.add_argument('--cng', action='store_true', default=True,
                    help='enable comfort noise (default: on)')
    p.add_argument('--no-cng', dest='cng', action='store_false')
    p.add_argument('--no-res', dest='res', action='store_false', default=True)
    p.add_argument('--filter', type=int, default=832,
                    help='filter length in samples (default 832 = 52 ms @16k)')
    p.add_argument('--sample-rate', type=int, default=16000)
    p.add_argument('--plot', help='diagnostic PNG path (default: <out>.png)')
    p.add_argument('--no-plot', action='store_true', help='skip the diagnostic plot')
    args = p.parse_args()

    print(f'preset={args.preset} cng={args.cng} res={args.res} '
          f'filter={args.filter} sr={args.sample_rate}', file=sys.stderr)

    mic, ref, out, erle, sr = run_aec(
        args.mic, args.ref, args.out,
        preset=args.preset, filter_length=args.filter,
        enable_cng=args.cng, enable_res=args.res,
        sample_rate=args.sample_rate,
    )
    print(f'wrote {args.out} ({len(out)} samples, {len(out) / sr:.2f}s)',
          file=sys.stderr)

    if args.no_plot:
        return
    png_path = args.plot or os.path.splitext(args.out)[0] + '.png'
    title = (f'{os.path.basename(args.mic)} | preset={args.preset} '
             f'cng={"on" if args.cng else "off"} '
             f'res={"on" if args.res else "off"}')
    if make_plot(mic, ref, out, erle, sr, png_path, title=title):
        print(f'wrote {png_path}', file=sys.stderr)


if __name__ == '__main__':
    main()
