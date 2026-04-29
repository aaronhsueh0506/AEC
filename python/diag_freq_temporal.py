"""Per-bin gain + spectrogram dump for worst/winning DT cases.

For each case, traces frame-by-frame:
  - gain_smooth (per-bin, full vector)
  - residual_echo_psd, error_psd, nearend_est (per-bin)
  - ERL estimate, render_ceil mean
  - mic, lpb, our_output spectrograms (rfft per frame)

Then for each case, computes:
  - per-band gain stats (low <500Hz, mid 500-2kHz, high 2k-8k) — mean, p10, p50, p90
  - per-frame gain_min trajectory (find transient leaks)
  - leak time-frequency map: where output_psd > nearend_est × 2
  - vs AEC2 output spectrogram (load aec2 wav for comparison)

Saves NPYs for offline inspection + prints summary.

Usage:
  python3 diag_freq_temporal.py STEM1 STEM2 ...
"""
import os, sys, json
from pathlib import Path
import numpy as np
import soundfile as sf

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from aec import AEC, AecConfig, AecMode

REPO = Path(__file__).parent.parent
WAV_BASE = REPO / 'wav/aec_challenge_blind'
AEC2_DIR = REPO / 'python/output_ref'
OUT = Path(__file__).parent / 'output_freq_temporal'


def _resolve(stem):
    for sub in ('doubletalk', 'farend_singletalk', 'nearend_singletalk'):
        p = WAV_BASE / sub / f'{stem}_mic.wav'
        if p.is_file():
            return sub, p, p.parent / f'{stem}_lpb.wav'
    raise FileNotFoundError(stem)


def _trace(stem):
    np.random.seed(20260428)
    sub, mic_p, lpb_p = _resolve(stem)
    mic, sr = sf.read(str(mic_p), dtype='float32')
    lpb, _  = sf.read(str(lpb_p), dtype='float32')
    if mic.ndim > 1: mic = mic[:, 0]
    if lpb.ndim > 1: lpb = lpb[:, 0]
    n = min(len(mic), len(lpb))

    is_mv = '_with_movement' in stem
    delay_kw = (dict(enable_delay_est=True, delay_est_period_s=0.25, delay_est_init_s=0.2)
                if is_mv else dict(enable_delay_est=False))
    cfg = AecConfig.from_preset('balanced', sample_rate=sr, mode=AecMode.PBFDKF,
                                enable_dtd=False, enable_shadow=True, enable_res=True,
                                use_kalman=True, **delay_kw)
    aec = AEC(cfg)
    if aec.res: aec.res.enable_stats()
    hop = aec.hop_size
    n_freqs = aec.res.n_freqs if aec.res else 257

    pos = 0; idx = 0
    out = np.zeros(n, dtype=np.float32)
    gains = []
    erls = []; render_ceils = []
    res_psds = []; err_psds = []; ne_psds = []
    mic_psds = []; lpb_psds = []; out_psds = []
    dt_combs = []; conv_flags = []; once_flags = []; render_flags = []
    far_pwrs = []
    while pos + hop <= n:
        mic_f = mic[pos:pos+hop]; lpb_f = lpb[pos:pos+hop]
        mic_spec = np.fft.rfft(mic_f, n=2*(n_freqs-1))
        lpb_spec = np.fft.rfft(lpb_f, n=2*(n_freqs-1))
        o = aec.process(mic_f, lpb_f); out[pos:pos+hop] = o
        out_spec = np.fft.rfft(o, n=2*(n_freqs-1))
        # length-mismatched; just take first n_freqs of input fft
        mic_psds.append(np.abs(mic_spec[:n_freqs])**2)
        lpb_psds.append(np.abs(lpb_spec[:n_freqs])**2)
        out_psds.append(np.abs(out_spec[:n_freqs])**2)
        gains.append(aec.res.gain_smooth.copy())
        erls.append(getattr(aec.res, '_stats_last_erl_estimate', 0))
        render_ceils.append(getattr(aec.res, '_stats_last_render_ceil_mean', 0))
        res_psds.append(getattr(aec.res, '_stats_last_res_psd', 0))
        err_psds.append(float(np.mean(aec.res.error_psd)))
        ne_psds.append(getattr(aec.res, '_stats_last_nearend', 0))
        far_pwrs.append(float(np.mean(lpb_f**2)))
        s = aec._aec_state
        dt_combs.append(float(s.dt_combined))
        conv_flags.append(bool(s.filter_converged))
        once_flags.append(bool(s.filter_once_converged))
        render_flags.append(bool(aec.res._using_render_based))
        pos += hop; idx += 1

    return {
        'stem': stem, 'sub': sub, 'sr': sr, 'hop': hop, 'n_freqs': n_freqs,
        'gains': np.stack(gains),         # (T, F)
        'mic_psd': np.stack(mic_psds),
        'lpb_psd': np.stack(lpb_psds),
        'out_psd': np.stack(out_psds),
        'res_psd': np.array(res_psds),
        'err_psd': np.array(err_psds),
        'ne_psd': np.array(ne_psds),
        'erl': np.array(erls),
        'render_ceil': np.array(render_ceils),
        'far_pwr': np.array(far_pwrs),
        'dt_combined': np.array(dt_combs),
        'converged': np.array(conv_flags),
        'once_converged': np.array(once_flags),
        'using_render': np.array(render_flags),
        'output_signal': out[:pos],
    }


def _analyze(t, aec2_signal):
    """Print per-band gain stats + leak map summary + vs AEC2 diff."""
    sr = t['sr']; hop = t['hop']; n_freqs = t['n_freqs']
    freqs = np.linspace(0, sr/2, n_freqs)
    bands = {
        'lo (<500Hz)':  (freqs < 500),
        'mid (.5-2k)':  (freqs >= 500) & (freqs < 2000),
        'hi (2k-8k)':   (freqs >= 2000) & (freqs < 8000),
    }
    gains = t['gains']     # (T, F)
    mask = (t['far_pwr'] > 1e-4) & (t['dt_combined'] > 0.3)  # DT × far frames
    n_dtfar = int(mask.sum())
    print(f'\n=== {t["stem"]} ({t["sub"]}, {len(gains)} frames, {n_dtfar} DT×far) ===')
    if n_dtfar < 5:
        print('  too few DT×far frames'); return
    g_dtfar = gains[mask]
    for name, fmask in bands.items():
        gband = g_dtfar[:, fmask]            # (n_dtfar, n_band)
        flat = gband.flatten()
        print(f'  gain {name:<12s}: mean={flat.mean():.3f}  p10={np.percentile(flat,10):.3f}  '
              f'p50={np.percentile(flat,50):.3f}  p90={np.percentile(flat,90):.3f}')

    # Per-frame transient: gain max in DT×far (high gain when echo loud)
    g_max_per_frame = gains[mask].max(axis=1)
    print(f'  gain_max trajectory: mean={g_max_per_frame.mean():.3f}  p90={np.percentile(g_max_per_frame,90):.3f}  '
          f'p99={np.percentile(g_max_per_frame,99):.3f}  max={g_max_per_frame.max():.3f}')

    # ERL trajectory
    erl = t['erl'][t['erl'] > 0]
    if len(erl):
        print(f'  ERL  : mean={erl.mean():.3f}  median={np.median(erl):.3f}  min={erl.min():.3f}  max={erl.max():.3f}')
        print(f'         clamped@0.3 frames: {int((erl >= 0.299).sum())}/{len(erl)} '
              f'({100*(erl>=0.299).sum()/len(erl):.0f}%)')

    # vs AEC2 leak comparison
    if aec2_signal is not None:
        # frame our output vs aec2 output, compare per-band echo residual
        a2 = aec2_signal[:len(t['output_signal'])]
        n_frames = len(gains)
        a2_psds = []; ours_psds = []
        for i in range(n_frames):
            s, e = i*hop, (i+1)*hop
            if e > len(a2): break
            a2_psds.append(np.abs(np.fft.rfft(a2[s:e], n=2*(n_freqs-1))[:n_freqs])**2)
            ours_psds.append(np.abs(np.fft.rfft(t['output_signal'][s:e], n=2*(n_freqs-1))[:n_freqs])**2)
        a2_psds = np.stack(a2_psds); ours_psds = np.stack(ours_psds)
        m2 = mask[:len(a2_psds)]
        if m2.sum() > 5:
            for name, fmask in bands.items():
                ours_band = ours_psds[m2][:, fmask].mean()
                aec2_band = a2_psds[m2][:, fmask].mean()
                ratio_db = 10*np.log10((ours_band+1e-12)/(aec2_band+1e-12))
                print(f'  out_psd {name:<12s}: ours={ours_band:.2e} aec2={aec2_band:.2e} '
                      f'ours/aec2 = {ratio_db:+.1f} dB')


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('stems', nargs='+')
    args = ap.parse_args()
    OUT.mkdir(exist_ok=True)
    for stem in args.stems:
        t = _trace(stem)
        # Load AEC2 reference
        a2 = AEC2_DIR / f'{stem}_aec2.wav'
        a2_sig = None
        if a2.is_file():
            sig, _ = sf.read(str(a2), dtype='float32')
            a2_sig = sig[:, 0] if sig.ndim > 1 else sig
        _analyze(t, a2_sig)
        # Save NPYs for offline plot
        np.savez(OUT / f'{stem}.npz',
                 gains=t['gains'], mic_psd=t['mic_psd'], lpb_psd=t['lpb_psd'],
                 out_psd=t['out_psd'], erl=t['erl'], render_ceil=t['render_ceil'],
                 dt_combined=t['dt_combined'], far_pwr=t['far_pwr'],
                 using_render=t['using_render'], converged=t['converged'])


if __name__ == '__main__':
    main()
