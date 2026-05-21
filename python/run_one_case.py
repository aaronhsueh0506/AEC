#!/usr/bin/env python3
"""Run AEC on one mic/ref pair, save output WAV + diagnostic plot.

Usage:
    python3 python/run_one_case.py mic.wav ref.wav out.wav [--preset balanced]
                                   [--cng] [--filter 832] [--no-res]
                                   [--plot out.png]
    python3 python/run_one_case.py mic.wav ref.wav out.wav --demo

Standard plot panels:
    1. Mic waveform with AEC output overlaid
    2. Reference (far-end / loopback) waveform
    3. Mic spectrogram
    4. AEC output spectrogram
    5. Long-term magnitude spectrum (frequency response): mic vs out
    6. Per-frame ERLE (dB) over time

Demo mode (--demo) runs four configurations on the same input and
stacks them for A/B comparison:
    - bypass     (no linear filter, no RES, no CNG — mic passthrough)
    - linear     (linear AEC only; RES + CNG disabled)
    - +res       (linear + residual suppressor)
    - +res +cng  (full pipeline; the recommended default)
"""
import argparse
import os
import sys

import numpy as np
import soundfile as sf

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from aec import AEC, AecConfig, AecMode, AecPreset


PRESET_MAP = {
    'balanced': AecPreset.BALANCED,
}


def run_aec(mic_path, ref_path, out_path, *, preset='balanced',
            filter_length=832, enable_cng=True, enable_res=True,
            sample_rate=16000, write_wav=True,
            mic_pad=0, ref_pad=0,
            diag_csv_path=None,
            trace_aec_state=False,
            diverged_reset=False,
            diverged_reset_streak_frames=50,
            diverged_reset_cooldown_frames=400,
            plan_b_dt_per_bin_gamma=False,
            trace_hf_chain_path=None):
    """Process one case; return (mic, ref, out, erle_per_frame, sample_rate).

    mic_pad / ref_pad: prepend N zero samples to mic / ref before processing
    (use to absorb a static delay larger than max_delay_ms). Output WAV is
    stripped of the leading max(mic_pad, ref_pad) samples.

    diag_csv_path: write per-frame AecStats rows for filter / DTD trajectory
    analysis.
    """
    cfg = AecConfig.from_preset(
        PRESET_MAP[preset],
        sample_rate=sample_rate,
        filter_length=filter_length,
        mode=AecMode.PBFDKF,
        enable_res=enable_res,
        enable_cng=enable_cng,
        enable_shadow=True,
        diverged_reset_enabled=diverged_reset,
        diverged_reset_streak_frames=diverged_reset_streak_frames,
        diverged_reset_cooldown_frames=diverged_reset_cooldown_frames,
        plan_b_dt_per_bin_gamma=plan_b_dt_per_bin_gamma,
        trace_hf_chain=bool(trace_hf_chain_path),
    )
    # v3.21.5 Phase 1 Sprint A — AEC3 echo_remover.cc:495-501 clamp opt-in
    # via env var (mirrors eval_aec_challenge.py convention).
    if 'AEC_E2_Y2_CLAMP' in os.environ:
        cfg.e2_y2_clamp_enabled = (
            os.environ['AEC_E2_Y2_CLAMP'].lower() not in ('0', 'false', 'off', 'no'))
    # v3.15 §B3: seed CNG for byte-equal sanity across CLI invocations.
    # Matches eval_aec_challenge.py:325 convention (seed=0 per-case);
    # without this, CNG (np.random.randn at aec.py:2009-2010) was
    # non-deterministic and masked code-induced Δ in run-to-run compare.
    np.random.seed(0)
    aec = AEC(cfg)

    mic, sr_mic = sf.read(mic_path)
    ref, sr_ref = sf.read(ref_path)
    if sr_mic != sample_rate or sr_ref != sample_rate:
        raise ValueError(
            f"sample rate mismatch: mic={sr_mic} ref={sr_ref} expected={sample_rate}"
        )
    mic = np.asarray(mic, dtype=np.float32)
    ref = np.asarray(ref, dtype=np.float32)
    if mic_pad > 0:
        mic = np.concatenate([np.zeros(mic_pad, dtype=np.float32), mic])
    if ref_pad > 0:
        ref = np.concatenate([np.zeros(ref_pad, dtype=np.float32), ref])
    n = min(len(mic), len(ref))
    mic = mic[:n]
    ref = ref[:n]

    hop = aec.hop_size
    out = np.zeros(n, dtype=np.float32)
    erle_log = []
    diag_rows = []
    pos = 0
    while pos + hop <= n:
        block = aec.process(mic[pos:pos + hop], ref[pos:pos + hop])
        out[pos:pos + hop] = block
        erle_log.append(aec.get_erle_instant())
        if diag_csv_path is not None:
            s = aec.get_stats()
            row = [
                s.frame_count, s.time_s,
                int(s.filter_converged), int(s.filter_once_converged),
                s.erle_inst_db, s.erle_windowed_db,
                s.dt_confidence, s.dt_from_energy, s.dt_from_shadow,
                s.dt_from_coherence, int(s.dt_active),
                s.far_power_db, s.mic_power_db, s.error_power_db,
                s.far_activity, s.divergence, int(s.epc_active),
                s.res_gain_mean_db, s.echo_psd_mean_db, s.error_psd_mean_db,
                s.delay_samples, s.delay_ms,
                int(s.res_using_render),
            ]
            if trace_high_band_metrics:
                d = aec._diag
                row.extend([
                    d.get('m_excess_ratio_a05', 0.0),
                    d.get('m_excess_ratio_a10', 0.0),
                    d.get('m_excess_ratio_a20', 0.0),
                    d.get('m_modulation', 0.0),
                    d.get('m_spectral_flatness', 0.0),
                ])
            # P3e advisory + mu_scale always-on diag (cheap)
            d = aec._diag
            row.extend([
                int(d.get('dt_advisory_active', False)),
                int(d.get('dt_advisory_hit', False)),
                float(d.get('mu_scale', 1.0)),
            ])
            if trace_aec_state:
                row.extend([
                    float(d.get('main_err_ratio', 0.0)),
                    float(d.get('shadow_err_ratio', 0.0)),
                    float(d.get('p3f_shadow_advantage', 1.0)),
                    float(d.get('erle_slope_db_per_s', 0.0)),
                    float(d.get('post_reset_age_ms', 0.0)),
                    str(d.get('filter_state', 'startup')),
                    int(d.get('usable_linear', False)),
                    float(d.get('residual_psd_linear', 0.0)),
                    float(d.get('residual_psd_render', 0.0)),
                    float(d.get('residual_render_blend', 0.0)),
                    int(d.get('p3h_reset_fired', False)),
                    int(d.get('p3h_reset_count', 0)),
                    float(d.get('p4b_dt_per_bin_mean', 0.0)),
                    float(d.get('p4b_dt_per_bin_hf_mean', 0.0)),
                    float(d.get('p4b_coh2_hf_mean', 0.0)),
                    float(d.get('p4b_effective_dt', 0.0)),
                    int(d.get('p4b_is_stationary_dt', 0)),
                    float(d.get('p4b_gain_hf_mean', 1.0)),
                    float(d.get('p4b_res_echo_hf_mean_db', -120.0)),
                ])
            diag_rows.append(tuple(row))
        pos += hop

    pad_strip = max(mic_pad, ref_pad)
    out_trim = out[pad_strip:pos]
    mic_trim = mic[pad_strip:pos]
    ref_trim = ref[pad_strip:pos]

    if write_wav and out_path:
        sf.write(out_path, out_trim, sample_rate)

    if diag_csv_path is not None and diag_rows:
        import csv
        header = ['frame', 'time_s', 'conv', 'once_conv',
                  'erle_inst_db', 'erle_win_db',
                  'dt_conf', 'dt_energy', 'dt_shadow', 'dt_coh', 'dt_active',
                  'far_db', 'mic_db', 'err_db',
                  'far_act', 'divergence', 'epc',
                  'res_gain_db', 'echo_psd_db', 'err_psd_db',
                  'delay_samp', 'delay_ms',
                  'using_render']
        if trace_high_band_metrics:
            header.extend(['m_excess_a05', 'm_excess_a10', 'm_excess_a20',
                           'm_modulation', 'm_spectral_flatness'])
        header.extend(['dt_adv_active', 'dt_adv_hit', 'mu_scale'])
        if trace_aec_state:
            header.extend(['main_err_ratio', 'shadow_err_ratio',
                           'p3f_shadow_advantage', 'erle_slope_db_per_s',
                           'post_reset_age_ms', 'filter_state', 'usable_linear',
                           'residual_psd_linear', 'residual_psd_render',
                           'residual_render_blend',
                           'p3h_reset_fired', 'p3h_reset_count',
                           'p4b_dt_per_bin_mean', 'p4b_dt_per_bin_hf_mean',
                           'p4b_coh2_hf_mean', 'p4b_effective_dt',
                           'p4b_is_stationary_dt', 'p4b_gain_hf_mean',
                           'p4b_res_echo_hf_mean_db'])
        with open(diag_csv_path, 'w', newline='') as fp:
            w = csv.writer(fp)
            w.writerow(header)
            w.writerows(diag_rows)

    # v3.21.2 S1: HF causal chain trace CSV dump
    if trace_hf_chain_path and getattr(aec, '_hf_chain_trace', None):
        import csv as _csv
        rows = aec._hf_chain_trace
        keys = list(rows[0].keys())
        with open(trace_hf_chain_path, 'w', newline='') as fp:
            w = _csv.DictWriter(fp, fieldnames=keys)
            w.writeheader()
            w.writerows(rows)
        print(f"  hf_chain trace -> {trace_hf_chain_path} ({len(rows)} frames)")

    return mic_trim, ref_trim, out_trim, np.asarray(erle_log), sample_rate


def _spectrogram(x, sample_rate, nfft=512, hop=256):
    from numpy.fft import rfft
    n_blocks = max(0, (len(x) - nfft) // hop + 1)
    if n_blocks <= 0:
        return (np.zeros((nfft // 2 + 1, 1)), np.array([0.0]),
                np.linspace(0, sample_rate / 2, nfft // 2 + 1))
    win = np.hanning(nfft).astype(np.float32)
    S = np.zeros((nfft // 2 + 1, n_blocks), dtype=np.float32)
    for k in range(n_blocks):
        seg = x[k * hop:k * hop + nfft] * win
        S[:, k] = np.abs(rfft(seg)) + 1e-10
    times = (np.arange(n_blocks) * hop + nfft / 2) / sample_rate
    freqs = np.linspace(0, sample_rate / 2, nfft // 2 + 1)
    return 20 * np.log10(S), times, freqs


def _avg_magnitude_db(x, sample_rate, nfft=1024, hop=512):
    from numpy.fft import rfft
    n_blocks = max(0, (len(x) - nfft) // hop + 1)
    freqs = np.linspace(0, sample_rate / 2, nfft // 2 + 1)
    if n_blocks <= 0:
        return np.full(nfft // 2 + 1, -120.0), freqs
    win = np.hanning(nfft).astype(np.float32)
    psum = np.zeros(nfft // 2 + 1, dtype=np.float64)
    for k in range(n_blocks):
        seg = x[k * hop:k * hop + nfft] * win
        psum += np.abs(rfft(seg)) ** 2
    avg = np.sqrt(psum / n_blocks) + 1e-12
    return 20 * np.log10(avg), freqs


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
    t_end = t[-1] if n else 1.0

    fig, axes = plt.subplots(6, 1, figsize=(12, 14), constrained_layout=True)
    fig.suptitle(title or 'AEC single-case diagnostic', fontsize=12)

    axes[0].plot(t, mic, lw=0.5, color='#1f77b4', label='mic', alpha=0.7)
    axes[0].plot(t, out, lw=0.5, color='#2ca02c', label='AEC out', alpha=0.85)
    axes[0].set_ylabel('mic + out')
    axes[0].set_xlim(0, t_end)
    axes[0].legend(loc='upper right', fontsize=8)

    axes[1].plot(t, ref, lw=0.5, color='#ff7f0e')
    axes[1].set_ylabel('ref')
    axes[1].set_xlim(0, t_end)

    Smic, t_spec, freqs = _spectrogram(mic, sample_rate)
    Sout, _, _ = _spectrogram(out, sample_rate)
    vmin, vmax = -80, 0
    extent = [t_spec[0], t_spec[-1] if len(t_spec) else 1, freqs[0], freqs[-1]]
    axes[2].imshow(Smic, aspect='auto', origin='lower', extent=extent,
                   vmin=vmin, vmax=vmax, cmap='magma')
    axes[2].set_ylabel('mic spec\n(Hz)')
    axes[2].set_yticks([0, 2000, 4000, 6000, 8000])

    axes[3].imshow(Sout, aspect='auto', origin='lower', extent=extent,
                   vmin=vmin, vmax=vmax, cmap='magma')
    axes[3].set_ylabel('out spec\n(Hz)')
    axes[3].set_yticks([0, 2000, 4000, 6000, 8000])

    mic_db, fr = _avg_magnitude_db(mic, sample_rate)
    out_db, _ = _avg_magnitude_db(out, sample_rate)
    axes[4].semilogx(fr[1:], mic_db[1:], color='#1f77b4', lw=1.0, label='mic')
    axes[4].semilogx(fr[1:], out_db[1:], color='#2ca02c', lw=1.0, label='AEC out')
    axes[4].set_ylabel('avg mag (dB)')
    axes[4].set_xlabel('frequency (Hz)')
    axes[4].set_xlim(50, sample_rate / 2)
    axes[4].grid(True, which='both', alpha=0.3)
    axes[4].legend(loc='lower left', fontsize=8)

    if len(erle):
        t_erle = np.arange(len(erle)) * (n / max(len(erle), 1)) / sample_rate
        axes[5].plot(t_erle, erle, lw=0.7, color='#d62728')
    axes[5].set_ylabel('ERLE (dB)')
    axes[5].set_xlabel('time (s)')
    axes[5].axhline(0, color='#888', lw=0.5)
    axes[5].set_xlim(0, t_end)

    fig.savefig(png_path, dpi=120)
    plt.close(fig)
    return True


def make_demo_pair_plot(mic, ref, left, right, sample_rate, png_path, title=''):
    """left/right: (label, out, erle). 2-column side-by-side comparison."""
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        from matplotlib.gridspec import GridSpec
    except ImportError:
        print('warning: matplotlib not available — skipping plot', file=sys.stderr)
        return False

    n = len(mic)
    t = np.arange(n) / sample_rate
    t_end = t[-1] if n else 1.0

    fig = plt.figure(figsize=(14, 11), constrained_layout=True)
    gs = GridSpec(4, 2, figure=fig)
    fig.suptitle(title or 'AEC demo: feature comparison', fontsize=12)

    Smic, t_spec, freqs = _spectrogram(mic, sample_rate)
    extent = [t_spec[0], t_spec[-1] if len(t_spec) else 1, freqs[0], freqs[-1]]
    vmin, vmax = -80, 0

    ax_ref = fig.add_subplot(gs[0, :])
    ax_ref.plot(t, ref, lw=0.5, color='#ff7f0e')
    ax_ref.set_ylabel('ref')
    ax_ref.set_xlim(0, t_end)
    ax_ref.set_title('reference (far-end loopback)', fontsize=10)

    for col, (label, out_i, erle_i) in enumerate([left, right]):
        ax_w = fig.add_subplot(gs[1, col])
        ax_w.plot(t, mic, lw=0.5, color='#1f77b4', alpha=0.5, label='mic')
        ax_w.plot(t, out_i, lw=0.5, color='#2ca02c', alpha=0.85, label='out')
        ax_w.set_xlim(0, t_end)
        ax_w.set_title(label, fontsize=11, fontweight='bold')
        ax_w.legend(loc='upper right', fontsize=8)
        if col == 0:
            ax_w.set_ylabel('mic + out')

        ax_s = fig.add_subplot(gs[2, col])
        S, _, _ = _spectrogram(out_i, sample_rate)
        ax_s.imshow(S, aspect='auto', origin='lower', extent=extent,
                    vmin=vmin, vmax=vmax, cmap='magma')
        ax_s.set_yticks([0, 2000, 4000, 6000, 8000])
        if col == 0:
            ax_s.set_ylabel('out spec (Hz)')

        ax_e = fig.add_subplot(gs[3, col])
        if len(erle_i):
            t_erle = np.arange(len(erle_i)) * (n / max(len(erle_i), 1)) / sample_rate
            ax_e.plot(t_erle, erle_i, lw=0.7, color='#d62728')
        ax_e.axhline(0, color='#888', lw=0.5)
        ax_e.set_xlim(0, t_end)
        ax_e.set_xlabel('time (s)')
        if col == 0:
            ax_e.set_ylabel('ERLE (dB)')

    fig.savefig(png_path, dpi=120)
    plt.close(fig)
    return True


def run_demo(args):
    """Run several feature configurations and produce a comparison plot."""
    base = os.path.splitext(args.out)[0]
    configs = [
        ('linear',     dict(enable_res=False, enable_cng=False)),
        ('+res',       dict(enable_res=True,  enable_cng=False)),
        ('+res +cng',  dict(enable_res=True,  enable_cng=True)),
    ]

    mic_ref = None
    results = {}
    for label, flags in configs:
        wav_path = f'{base}__{label.replace(" ", "").replace("+", "p")}.wav'
        mic, ref, out, erle, sr = run_aec(
            args.mic, args.ref, wav_path,
            preset=args.preset, filter_length=args.filter,
            sample_rate=args.sample_rate, **flags,
        )
        print(f'wrote {wav_path}', file=sys.stderr)
        if mic_ref is None:
            mic_ref = (mic, ref, sr)
        results[label] = (out, erle)

    mic, ref, sr = mic_ref
    base_title = (f'{os.path.basename(args.mic)} | preset={args.preset} '
                  f'fl={args.filter}')

    pairs = [
        ('linear_vs_res', 'linear', '+res',
         'demo: linear vs linear+res'),
        ('res_vs_rescng', '+res', '+res +cng',
         'demo: +res vs +res+cng'),
    ]
    for tag, l_label, r_label, subtitle in pairs:
        png_path = f'{base}_demo_{tag}.png'
        l = (l_label, *results[l_label])
        r = (r_label, *results[r_label])
        if make_demo_pair_plot(mic, ref, l, r, sr, png_path,
                               title=f'{base_title} | {subtitle}'):
            print(f'wrote {png_path}', file=sys.stderr)


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('mic')
    p.add_argument('ref')
    p.add_argument('out', help='output WAV path (in --demo, used as base name)')
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
    p.add_argument('--demo', action='store_true',
                   help='run linear / +res / +res+cng and produce a comparison plot')
    p.add_argument('--mic-pad', type=int, default=0,
                   help='prepend N zero samples to mic before processing')
    p.add_argument('--ref-pad', type=int, default=0,
                   help='prepend N zero samples to ref before processing')
    p.add_argument('--diag-csv',
                   help='write per-frame AecStats trajectory CSV here')
    p.add_argument('--trace-aec-state', action='store_true',
                   help='P3f Phase 1: include Mini AecState fields '
                        '(main_err_ratio, shadow_err_ratio, shadow_advantage, '
                        'erle_slope, post_reset_age, filter_state, usable_linear)')
    p.add_argument('--diverged-reset', action='store_true',
                   help='P3h: reset filter on sustained diverged (off by default)')
    p.add_argument('--diverged-reset-streak', type=int, default=50,
                   help='P3h: frames of sustained diverged before reset (default 50)')
    p.add_argument('--diverged-reset-cooldown', type=int, default=400,
                   help='P3h: cooldown frames after a reset (default 400)')
    p.add_argument('--plan-b', action='store_true',
                   help='P4B: γ²(k)-primary dt_per_bin (γ=1-coh2 with soft '
                        'floor lift only when effective_dt > 0.5; off by default)')
    p.add_argument('--trace-hf-chain',
                   help='v3.21.2 S1: write per-frame HF damage causal chain '
                        'trace CSV here (convergence -> ERLE -> R² -> '
                        'DominantNearendDetector -> HF cap gate -> gain)')
    args = p.parse_args()

    if args.demo:
        print(f'demo mode: preset={args.preset} filter={args.filter} '
              f'sr={args.sample_rate}', file=sys.stderr)
        run_demo(args)
        return

    print(f'preset={args.preset} cng={args.cng} res={args.res} '
          f'filter={args.filter} sr={args.sample_rate}', file=sys.stderr)
    mic, ref, out, erle, sr = run_aec(
        args.mic, args.ref, args.out,
        preset=args.preset, filter_length=args.filter,
        enable_cng=args.cng, enable_res=args.res,
        sample_rate=args.sample_rate,
        mic_pad=args.mic_pad, ref_pad=args.ref_pad,
        diag_csv_path=args.diag_csv,
        trace_aec_state=args.trace_aec_state,
        diverged_reset=args.diverged_reset,
        diverged_reset_streak_frames=args.diverged_reset_streak,
        diverged_reset_cooldown_frames=args.diverged_reset_cooldown,
        plan_b_dt_per_bin_gamma=args.plan_b,
        trace_hf_chain_path=args.trace_hf_chain,
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
