"""Deep per-frame trace of one DT case to identify exactly where echo gets killed.

Picks worst static-DT case (#1 ZtGitIxr, Δecho=+1.832 vs AEC2). Runs AEC with
ResFilter stats enabled, captures every per-frame intermediate from RES pipeline.

Per frame fields (all means over n_freqs unless noted):
  ── filter side ──
    echo_psd_mean, error_psd_mean, far_psd_mean, mic_pwr, far_pwr, raw_err_pwr
    erle_inst, erle_factor, erl_estimate, filter_converged, once_converged
  ── DT detection ──
    dt_energy, dt_shadow, dt_coh, dt_combined, effective_dt
  ── EPC ──
    epc_active, epc_hangover, render_forced_remaining
  ── ResidualEchoEstimator (Phase B) ──
    using_render_based, residual_after_attribute (= linear+render mix BEFORE caps)
  ── 4 cap stages ──
    res_after_echo_cap, res_after_error_cap, res_after_dt_cap, res_after_render_ceil
  ── post-residual ──
    nearend_est, min_ne_from_dt, enr
    ne_g_floor, spectral_g_min, noise_floor_gain
    gain_before_floor, gain_after_floor, gain_after_smoothing

Outputs JSONL (one line per frame) + flags problem frames.

Usage:
  python3 diag_deep_trace.py [stem] [--talk dt|st|nst]
"""
import os
import sys
import json
import argparse
from pathlib import Path
import numpy as np
import soundfile as sf

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from aec import AEC, AecConfig, AecMode

REPO = Path(__file__).parent.parent
WAV_BASE = REPO / 'wav/aec_challenge_blind'
OUT = Path(__file__).parent / 'output_trace'

# Default: worst static-DT (Δecho +1.832 vs AEC2, never converged, render% 98%)
DEFAULT_STEM = 'ZtGitIxrzU0ILwu0HACaaw_doubletalk'


def _resolve(stem):
    for sub in ('doubletalk', 'farend_singletalk', 'nearend_singletalk'):
        p = WAV_BASE / sub / f'{stem}_mic.wav'
        if p.is_file():
            return sub, p, p.parent / f'{stem}_lpb.wav'
    raise FileNotFoundError(stem)


def _frame_record(idx: int, aec: AEC, far_pwr, mic_pwr, raw_err_pwr, raw_output, far_end):
    res = aec.res
    s = aec._aec_state
    d = aec._diag
    rec = {
        'idx': idx,
        # filter
        'far_pwr': float(far_pwr),
        'mic_pwr': float(mic_pwr),
        'raw_err_pwr': float(raw_err_pwr),
        'echo_psd_mean': float(np.mean(res.echo_psd)),
        'error_psd_mean': float(np.mean(res.error_psd)),
        'far_psd_mean': float(np.mean(np.abs(np.fft.rfft(far_end))**2)),
        'erle_inst': float(d.get('erle_inst', 0)),
        'erle_factor': float(d.get('erle_factor', 0)),
        'erl_estimate': float(s._epc.config.epc_hangover * 0 + aec._erl_estimate),  # raw
        'filter_converged': bool(s.filter_converged),
        'once_converged': bool(s.filter_once_converged),
        # DT
        'dt_energy': float(s.dt_from_energy),
        'dt_shadow': float(s.dt_from_shadow),
        'dt_coh': float(s.dt_from_coherence),
        'dt_combined': float(s.dt_combined),
        # EPC
        'epc_active': bool(s.epc_active),
        'epc_hangover': int(s.epc_hangover_count),
        'render_forced_remaining': int(getattr(aec, '_epc_render_forced_remaining', 0)),
        # ResidualEchoEstimator
        'using_render_based': bool(res._using_render_based),
        'residual_after_attribute': float(getattr(res, '_stats_last_res_after_attribute', 0)),
        'res_after_echo_cap':      float(getattr(res, '_stats_last_res_after_echo_cap', 0)),
        'res_after_error_cap':     float(getattr(res, '_stats_last_res_after_error_cap', 0)),
        'res_after_dt_cap':        float(getattr(res, '_stats_last_res_after_dt_cap', 0)),
        'res_after_render_ceil':   float(getattr(res, '_stats_last_res_after_render_ceil', 0)),
        # post-residual
        'res_psd_final':           float(getattr(res, '_stats_last_res_psd', 0)),
        'nearend_est_mean':        float(getattr(res, '_stats_last_nearend', 0)),
        'min_ne_from_dt_mean':     float(getattr(res, '_stats_last_min_ne', 0)),
        'enr_mean':                float(getattr(res, '_stats_last_enr', 0)),
        'ne_g_floor':              float(getattr(res, '_stats_last_ne_g_floor', 0)),
        'spectral_g_min':          float(getattr(res, '_stats_last_spectral_g_min', 0)),
        'noise_floor_gain':        float(getattr(res, '_stats_last_noise_floor_gain', 0)),
        'gain_before_floor':       float(getattr(res, '_stats_last_gain_before_floor', 0)),
        'gain_after_floor':        float(getattr(res, '_stats_last_gain_after_floor', 0)),
        'gain_after_smoothing':    float(getattr(res, '_stats_last_gain_after_smoothing', 0)),
        'gain_smooth_mean':        float(np.mean(res.gain_smooth)),
        # output
        'output_pwr':              float(np.mean(raw_output ** 2)),
    }
    return rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('stem', nargs='?', default=DEFAULT_STEM)
    ap.add_argument('--talk', default=None, help='override talk type (dt/st/nst)')
    ap.add_argument('--out', default=None)
    args = ap.parse_args()

    OUT.mkdir(exist_ok=True)
    sub, mic_p, lpb_p = _resolve(args.stem)
    out_path = Path(args.out) if args.out else OUT / f'{args.stem}.deep.jsonl'

    mic, sr = sf.read(str(mic_p), dtype='float32')
    lpb, _  = sf.read(str(lpb_p), dtype='float32')
    if mic.ndim > 1: mic = mic[:, 0]
    if lpb.ndim > 1: lpb = lpb[:, 0]
    n = min(len(mic), len(lpb))

    is_mv = '_with_movement' in args.stem
    delay_kw = (dict(enable_delay_est=True, delay_est_period_s=0.25, delay_est_init_s=0.2)
                if is_mv else dict(enable_delay_est=False))
    cfg = AecConfig.from_preset('balanced', sample_rate=sr, mode=AecMode.PBFDKF,
                                enable_dtd=False, enable_shadow=True, enable_res=True,
                                use_kalman=True, enable_cng=False, **delay_kw)
    aec = AEC(cfg)
    if aec.res:
        aec.res.enable_stats()

    hop = aec.hop_size
    pos = 0
    idx = 0
    out = np.zeros(n, dtype=np.float32)
    with open(out_path, 'w') as f:
        while pos + hop <= n:
            mic_frame = mic[pos:pos+hop]
            lpb_frame = lpb[pos:pos+hop]
            far_pwr = float(np.mean(lpb_frame ** 2))
            mic_pwr = float(np.mean(mic_frame ** 2))
            o = aec.process(mic_frame, lpb_frame)
            out[pos:pos+hop] = o
            raw_err_pwr = float(np.mean(o ** 2))
            rec = _frame_record(idx, aec, far_pwr, mic_pwr, raw_err_pwr, o, lpb_frame)
            f.write(json.dumps(rec) + '\n')
            pos += hop
            idx += 1

    print(f'wrote {idx} frames -> {out_path}')

    # Quick analysis: identify problem frames + which cap killed residual
    rows = [json.loads(l) for l in open(out_path)]
    print(f'\n=== quick analysis ===')
    n_total = len(rows)
    n_render = sum(1 for r in rows if r['using_render_based'])
    n_far_active = sum(1 for r in rows if r['far_pwr'] > 1e-4)
    n_dt = sum(1 for r in rows if r['dt_combined'] > 0.3)
    print(f'frames: {n_total}, far_active: {n_far_active} ({100*n_far_active/n_total:.0f}%), '
          f'DT: {n_dt} ({100*n_dt/n_total:.0f}%), render-based: {n_render} ({100*n_render/n_total:.0f}%)')

    # Frames in DT + render-based + far_active
    dt_render_far = [r for r in rows if r['using_render_based']
                                       and r['far_pwr'] > 1e-4
                                       and r['dt_combined'] > 0.3]
    print(f'DT × render-based × far-active: {len(dt_render_far)} frames')
    if dt_render_far:
        # Show 3 representative frames
        n = len(dt_render_far)
        for label, r in [('first', dt_render_far[0]),
                         ('mid',   dt_render_far[n//2]),
                         ('last',  dt_render_far[-1])]:
            print(f'\n--- {label} (frame {r["idx"]}) ---')
            print(f'  far_pwr={r["far_pwr"]:.2e}  mic_pwr={r["mic_pwr"]:.2e}  raw_err={r["raw_err_pwr"]:.2e}')
            print(f'  echo_psd={r["echo_psd_mean"]:.2e}  error_psd={r["error_psd_mean"]:.2e}')
            print(f'  far_psd={r["far_psd_mean"]:.2e}  erl_estimate={r["erl_estimate"]:.3f}')
            print(f'  dt_combined={r["dt_combined"]:.2f}  effective_dt(dt_for_fs proxy)≈{r["dt_combined"]:.2f}')
            print(f'  render_based={r["using_render_based"]}  epc={r["epc_active"]}  hangover={r["epc_hangover"]}')
            print(f'  ─ residual through 4 caps ─')
            print(f'    after_attribute  : {r["residual_after_attribute"]:.4e}')
            print(f'    after_echo_cap   : {r["res_after_echo_cap"]:.4e}  (echo_psd*2 = {r["echo_psd_mean"]*2:.2e})')
            print(f'    after_error_cap  : {r["res_after_error_cap"]:.4e}  (error_psd = {r["error_psd_mean"]:.2e})')
            print(f'    after_dt_cap     : {r["res_after_dt_cap"]:.4e}')
            print(f'    after_render_ceil: {r["res_after_render_ceil"]:.4e}')
            print(f'    final res_psd    : {r["res_psd_final"]:.4e}')
            print(f'  nearend_est={r["nearend_est_mean"]:.4e}  min_ne_from_dt={r["min_ne_from_dt_mean"]:.4e}')
            print(f'  enr={r["enr_mean"]:.4f}  spectral_g_min={r["spectral_g_min"]:.3f}')
            print(f'  ne_g_floor={r["ne_g_floor"]:.3f}  noise_floor_gain={r["noise_floor_gain"]:.3f}')
            print(f'  gain: before_floor={r["gain_before_floor"]:.3f}  after_floor={r["gain_after_floor"]:.3f}'
                  f'  after_smooth={r["gain_after_smoothing"]:.3f}  smooth_mean={r["gain_smooth_mean"]:.3f}')
            # Identify which cap was active
            stages = [
                ('attribute', r['residual_after_attribute']),
                ('echo_cap', r['res_after_echo_cap']),
                ('error_cap', r['res_after_error_cap']),
                ('dt_cap', r['res_after_dt_cap']),
                ('render_ceil', r['res_after_render_ceil']),
            ]
            for i in range(1, len(stages)):
                prev_v = stages[i-1][1]
                cur_v = stages[i][1]
                if prev_v > 0 and cur_v < prev_v * 0.5:
                    print(f'    >>> {stages[i][0]} reduced residual by {100*(1-cur_v/prev_v):.0f}% (active cap)')


if __name__ == '__main__':
    main()
