"""Filter weight + Kalman P + shadow trajectory trace.

Reads main filter W and shadow filter W per frame, computes:
- W_norm (Frobenius over partitions)
- W_main_vs_shadow ratio
- W "stability" (frame-to-frame change)

Goal: distinguish filter "never started" vs "started but learning wrong"
vs "learning correctly but slowly".

Usage:
  python3 diag_filter_dynamics.py STEM
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
    n_part = aec.filter.n_partitions

    pos = 0
    log = {'w_main': [], 'w_shadow': [], 'p_main': [], 'erle_db': [],
           'far_active': [], 'dt_e': [], 'erl': [], 'mu_scale': [],
           'shadow_adv': [], 'using_render': [], 'main_err': [], 'shadow_err': [],
           'erl_inst': []}
    while pos + hop <= n:
        mic_f = mic[pos:pos+hop]; lpb_f = lpb[pos:pos+hop]
        aec.process(mic_f, lpb_f)
        d = aec._diag
        # Filter weight norms (Frobenius across partitions × bins)
        log['w_main'].append(float(np.linalg.norm(aec.filter.W)))
        if aec.shadow_filter is not None:
            log['w_shadow'].append(float(np.linalg.norm(aec.shadow_filter.W)))
        else:
            log['w_shadow'].append(0.0)
        # Kalman P diagonal sum (proxy for "uncertainty")
        if hasattr(aec.filter, 'P'):
            log['p_main'].append(float(np.mean(aec.filter.P)))
        else:
            log['p_main'].append(0.0)
        log['erle_db'].append(10 * np.log10(max(d.get('erle_inst', 1e-10), 1e-10)))
        log['far_active'].append(float(d.get('far_activity', 0)) > 0.3)
        log['dt_e'].append(d.get('dt_from_energy', 0))
        log['erl'].append(d.get('erl_estimate', 0))
        log['mu_scale'].append(d.get('mu_scale', 0))
        log['shadow_adv'].append(d.get('shadow_advantage', 1.0))
        log['using_render'].append(d.get('using_render_based', False))
        log['main_err'].append(float(aec.main_err_smooth))
        log['shadow_err'].append(float(aec.shadow_err_smooth))
        log['erl_inst'].append(d.get('erle_inst', 0))
        pos += hop

    out = {k: np.array(v) for k, v in log.items()}
    out['stem'] = stem; out['sub'] = sub
    out['n_partitions'] = n_part
    out['filter_length'] = cfg.filter_length
    out['hop'] = hop
    out['sr'] = sr
    return out


def _seg_summary(t, label, idx_lo, idx_hi):
    """Stats over frame range [idx_lo, idx_hi)."""
    fa = t['far_active']
    sl = slice(idx_lo, idx_hi)
    fa_s = fa[sl]
    if fa_s.sum() < 5: return None
    return {
        'label': label,
        'frames': idx_hi - idx_lo,
        'far_active_pct': 100 * fa_s.mean(),
        'erle_dB_mean': float(t['erle_db'][sl][fa_s].mean()),
        'erle_dB_p90':  float(np.percentile(t['erle_db'][sl][fa_s], 90)),
        'w_main_mean':  float(t['w_main'][sl].mean()),
        'w_shadow_mean':float(t['w_shadow'][sl].mean()),
        'main_err_mean':float(t['main_err'][sl].mean()),
        'shadow_err_mean':float(t['shadow_err'][sl].mean()),
        'shadow_adv_mean':float(t['shadow_adv'][sl].mean()),
        'erl_mean':     float(t['erl'][sl].mean()),
        'mu_mean':      float(t['mu_scale'][sl].mean()),
    }


def _summarize(t):
    n = len(t['w_main'])
    print(f"\n=== {t['stem'][:50]} ({t['sub']}) ===")
    print(f"  filter_length={t['filter_length']} samples ({t['filter_length']/t['sr']*1000:.0f}ms),  "
          f"n_partitions={t['n_partitions']},  hop={t['hop']}")
    # 5 segments: 0-20%, 20-40%, ..., 80-100% of frames
    edges = [int(n*x) for x in (0, 0.2, 0.4, 0.6, 0.8, 1.0)]
    print(f"\n  segment-wise (over far-active frames):")
    print(f"  {'segment':<12} {'far%':>5} {'ERLE_dB':>9} {'p90':>6} {'W_main':>7} {'W_shdw':>7}"
          f" {'shdw_adv':>9} {'ERL':>6} {'mu':>5}")
    for i in range(5):
        s = _seg_summary(t, f'{i*20}-{(i+1)*20}%', edges[i], edges[i+1])
        if s is None:
            print(f"  {f'{i*20}-{(i+1)*20}%':<12}  (no far-active)")
        else:
            print(f"  {s['label']:<12} {s['far_active_pct']:>5.0f} {s['erle_dB_mean']:>+9.1f}"
                  f" {s['erle_dB_p90']:>+6.1f} {s['w_main_mean']:>7.3f} {s['w_shadow_mean']:>7.3f}"
                  f" {s['shadow_adv_mean']:>9.2f} {s['erl_mean']:>6.2f} {s['mu_mean']:>5.2f}")
    # W timeline
    cells = 32
    if n >= cells:
        step = n // cells
        line_w = ''
        line_e = ''
        line_a = ''  # shadow_adv
        for i in range(cells):
            seg = slice(i*step, (i+1)*step)
            wmean = float(t['w_main'][seg].mean())
            line_w += '0' if wmean<0.05 else '_' if wmean<0.2 else '-' if wmean<0.5 else '+' if wmean<1.0 else '#'
            erle = float(t['erle_db'][seg].mean())
            line_e += '.' if erle<0 else '-' if erle<5 else '+' if erle<10 else '#'
            sa = float(t['shadow_adv'][seg].mean())
            line_a += '.' if sa<0.5 else '_' if sa<0.9 else '-' if sa<1.1 else '+' if sa<1.5 else '#'
        print(f"\n  W_main:    |{line_w}|   (0:<.05  _:<.2  -:<.5  +:<1.0  #:>=1.0)")
        print(f"  ERLE:      |{line_e}|   (.<0  -<5  +<10  #>=10 dB)")
        print(f"  shadow_adv:|{line_a}|   (shadow vs main; >1 = shadow worse)")


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
