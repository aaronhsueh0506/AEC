"""Linear-filter stability deep trace per case.

Per-frame collects:
  - inst_erle (linear filter cancellation in dB)
  - filter_converged / once_converged / divergence
  - shadow_advantage (main_err / shadow_err)
  - mu_scale (effective adaptation rate)
  - erl_estimate
  - using_render_based
  - PR-A trigger (error_max > 0.05 in render mode)
  - epc_active
  - far_activity / dt_from_energy / dt_from_shadow

Classifies each case into Pattern A (filter never converged), Pattern B
(partial convergence), Pattern C (converged but PR-A misfires).

Usage:
  python3 diag_linear_stability.py STEM [STEM ...]
"""
import os, sys
from pathlib import Path
import numpy as np
import soundfile as sf

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from aec import AEC, AecConfig, AecMode

REPO = Path(__file__).parent.parent
WAV_BASE = REPO / 'wav/aec_challenge_blind'


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
    hop = aec.hop_size

    pos = 0
    metrics = {
        'erle_inst': [], 'erle_factor': [], 'converged': [], 'once_converged': [],
        'divergence': [], 'mu_scale': [], 'shadow_advantage': [],
        'erl_estimate': [], 'using_render_based': [],
        'far_activity': [], 'dt_from_energy': [], 'dt_from_shadow': [],
        'epc_active': [], 'saturation_level': [],
        'echo_psd_mean': [], 'error_psd_mean': [],
        'pra_trig': [], 'pra_fired': [],
    }
    pra_trig_total = 0
    pra_fired_total = 0
    while pos + hop <= n:
        mic_f = mic[pos:pos+hop]; lpb_f = lpb[pos:pos+hop]
        aec.process(mic_f, lpb_f)
        # PR-A would fire when far_active AND using_render_based AND
        # max(|error|) > 0.05. We don't have access to error_hop post-hoc here,
        # so use using_render_based × far_active as proxy.
        d = aec._diag
        for k in ('erle_inst', 'erle_factor', 'mu_scale', 'shadow_advantage',
                  'erl_estimate', 'using_render_based', 'far_activity',
                  'dt_from_energy', 'dt_from_shadow', 'epc_active',
                  'saturation_level', 'echo_psd_mean', 'error_psd_mean',
                  'divergence'):
            metrics[k].append(d.get(k, 0.0))
        metrics['converged'].append(bool(d.get('converged', False)))
        metrics['once_converged'].append(bool(aec._filter_once_converged))
        metrics['pra_trig'].append(0)
        metrics['pra_fired'].append(int(d.get('using_render_based', False)
                                       and d.get('far_activity', 0) > 0.3))
        pos += hop

    out = {k: np.array(v) for k, v in metrics.items()}
    out['stem'] = stem
    out['sub'] = sub
    out['n_frames'] = pos // hop
    return out


def _erle_db(t):
    e = t['erle_inst']
    return 10 * np.log10(np.maximum(e, 1e-10))


def _classify(t):
    """Return Pattern A/B/C label + supporting numbers."""
    erle = _erle_db(t)
    far_active = t['far_activity'] > 0.3
    if far_active.sum() < 10:
        return 'NoFar', {}
    erle_far = erle[far_active]
    converged = t['converged']
    once_conv = t['once_converged']

    erle_p50 = float(np.percentile(erle_far, 50))
    erle_p90 = float(np.percentile(erle_far, 90))
    conv_pct_far = float(converged[far_active].mean())
    once_conv_final = bool(once_conv[-1]) if len(once_conv) else False

    # Pattern A: filter never converged (median ERLE < 0 AND once_converged never set)
    if erle_p90 < 5.0 and not once_conv_final:
        label = 'A_never_conv'
    elif erle_p50 < 5.0 and conv_pct_far < 0.4:
        label = 'A_mostly_unconv'
    # Pattern C: converged most of the time but ERL drifts high
    elif conv_pct_far > 0.7 and float(np.percentile(t['erl_estimate'][far_active], 90)) > 0.8:
        label = 'C_erl_drift'
    # Pattern B: partial convergence
    else:
        label = 'B_partial_conv'
    return label, {
        'erle_p50_dB': erle_p50,
        'erle_p90_dB': erle_p90,
        'conv_pct_far': conv_pct_far,
        'once_conv': once_conv_final,
        'erl_p90': float(np.percentile(t['erl_estimate'][far_active], 90)),
    }


def _bar(values, cells=32, mapper=None):
    """ASCII bar over `cells` bins from `values`."""
    n = len(values)
    if n < cells: return '?' * cells
    step = max(1, n // cells)
    line = ''
    for i in range(cells):
        seg = values[i*step:(i+1)*step]
        if len(seg) == 0:
            line += '?'; continue
        v = float(np.mean(seg))
        line += mapper(v) if mapper else f'{v:.1f}'
    return line


def _summarize(t):
    n = t['n_frames']
    label, info = _classify(t)
    far_active = t['far_activity'] > 0.3
    fa_idx = far_active
    pra_fired_pct = 100 * t['pra_fired'].sum() / max(1, fa_idx.sum())
    using_render_pct = 100 * t['using_render_based'][fa_idx].mean() if fa_idx.sum() else 0
    mu_far = t['mu_scale'][fa_idx]
    print(f"\n=== {t['stem'][:50]} ({t['sub']}, {n} frames) ===")
    print(f"  PATTERN: {label}   (ERLE p50={info.get('erle_p50_dB',0):+.1f}dB  "
          f"p90={info.get('erle_p90_dB',0):+.1f}dB)")
    print(f"  conv_far={info.get('conv_pct_far',0)*100:.0f}%  once_conv={info.get('once_conv',False)}  "
          f"erl_p90={info.get('erl_p90',0):.2f}")
    print(f"  using_render={using_render_pct:.0f}% (of far-active)  PR-A fired={pra_fired_pct:.0f}%")
    print(f"  mu_scale (far-active): mean={mu_far.mean():.3f}  median={np.median(mu_far):.3f}  "
          f"p25={np.percentile(mu_far,25):.3f}  p75={np.percentile(mu_far,75):.3f}")
    # Timeline mini-bars
    erle = _erle_db(t)
    print(f"  ERLE :       |{_bar(erle, mapper=lambda v: '.' if v<0 else '-' if v<5 else '+' if v<10 else '#')}|  (.<0  -<5  +<10  #>=10 dB)")
    print(f"  mu_scale:    |{_bar(t['mu_scale'], mapper=lambda v: '0' if v<0.1 else '_' if v<0.3 else '-' if v<0.6 else '+' if v<0.85 else '#')}|  (0:<.1  _:<.3  -:<.6  +:<.85  #:>=.85)")
    print(f"  far_active:  |{_bar(t['far_activity'], mapper=lambda v: '.' if v<0.1 else '_' if v<0.3 else '-' if v<0.6 else '#')}|")
    print(f"  dt_energy:   |{_bar(t['dt_from_energy'], mapper=lambda v: '.' if v<0.1 else '_' if v<0.3 else '-' if v<0.5 else '#')}|")
    print(f"  shadow_adv:  |{_bar(t['shadow_advantage'], mapper=lambda v: '.' if v<0.5 else '_' if v<0.9 else '-' if v<1.1 else '+' if v<1.5 else '#')}|  (.<.5  _<.9  -<1.1  +<1.5  #>=1.5)")


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('stems', nargs='+')
    args = ap.parse_args()
    for stem in args.stems:
        t = _trace(stem)
        _summarize(t)


if __name__ == '__main__':
    main()
