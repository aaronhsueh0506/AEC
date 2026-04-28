"""Per-frame AecState trace dump for Phase B/C diagnostics.

Runs AEC on a wav pair, dumps each frame's AecState + key auxiliary signals
to JSONL. Used to inspect why path-change / shadow-copy / usable-linear didn't
trigger on movement-DT failure cases — replaces the inline-print approach.

Usage:
    python3 diag_state_trace.py <stem>                 # auto-locate under wav/aec_challenge_blind
    python3 diag_state_trace.py <mic.wav> <lpb.wav>    # explicit paths
    python3 diag_state_trace.py <stem> --out trace.jsonl
"""
import json
import os
import sys
import argparse
from pathlib import Path
import numpy as np
import soundfile as sf

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from aec import AEC, AecConfig, AecMode

REPO = Path(__file__).parent.parent
WAV_BASE = REPO / 'wav/aec_challenge_blind'


def _resolve(arg: str):
    """If arg is a stem, search the three scenario subdirs; else treat as path."""
    p = Path(arg)
    if p.is_file():
        return p
    for sub in ('farend_singletalk', 'doubletalk', 'nearend_singletalk'):
        cand = WAV_BASE / sub / f'{arg}_mic.wav'
        if cand.is_file():
            return cand
        cand = WAV_BASE / sub / arg
        if cand.is_file():
            return cand
    raise FileNotFoundError(arg)


def _frame_record(idx: int, aec: AEC, far_pwr: float, mic_pwr: float, raw_err_pwr: float):
    s = aec._aec_state
    d = aec._diag
    return {
        'idx': idx,
        # ── Render activity ──
        'far_pwr': far_pwr,
        'mic_pwr': mic_pwr,
        'render_active': s.render_active,
        'render_stationary': s.render_stationary,
        # ── Filter convergence ──
        'filter_converged': s.filter_converged,
        'filter_once_converged': s.filter_once_converged,
        'divergence': float(s.divergence),
        # ── DT signals ──
        'dt_energy': float(s.dt_from_energy),
        'dt_shadow': float(s.dt_from_shadow),
        'dt_coh': float(s.dt_from_coherence),
        'dt_combined': float(s.dt_combined),
        # ── EPC ──
        'epc_active': s.epc_active,
        'epc_hangover': s.epc_hangover_count,
        'epv_ratio': float(d.get('epv_gain_ratio', 1.0)),
        # ── Shadow ──
        'main_paused': s.main_paused,
        'shadow_adv': float(s.shadow_advantage),
        'main_err_smooth': float(getattr(aec, 'main_err_smooth', 0.0)),
        'shadow_err_smooth': float(getattr(aec, 'shadow_err_smooth', 0.0)),
        'copy_err_baseline': float(d.get('copy_err_baseline', 1e-6)),
        # ── Aggregate (Phase B gate) ──
        'usable_linear': s.usable_linear_estimate,
        # ── Auxiliary metrics ──
        'raw_err_pwr': raw_err_pwr,
        'erle_inst': float(d.get('erle_inst', 0.0)),
        'erle_factor': float(d.get('erle_factor', 0.0)),
        'erl_estimate': float(d.get('erl_estimate', 0.01)),
        'mu_scale': float(d.get('mu_scale', 1.0)),
        'using_render': bool(d.get('using_render_based', False)),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('arg', help='Stem under wav/aec_challenge_blind, OR mic.wav path')
    ap.add_argument('lpb', nargs='?', help='lpb.wav path (only if arg is mic.wav)')
    ap.add_argument('--out', default=None, help='Output JSONL path (default: <stem>.trace.jsonl)')
    ap.add_argument('--no-movement', action='store_true',
                    help='Force enable_delay_est=False (default: True if filename has _with_movement)')
    args = ap.parse_args()

    if args.lpb:
        mic_path, lpb_path = Path(args.arg), Path(args.lpb)
        stem = mic_path.stem
    else:
        mic_path = _resolve(args.arg)
        stem = mic_path.name[:-len('_mic.wav')] if mic_path.name.endswith('_mic.wav') else mic_path.stem
        lpb_path = mic_path.parent / f'{stem}_lpb.wav'

    if not lpb_path.is_file():
        print(f'ERROR: lpb not found at {lpb_path}', file=sys.stderr)
        return 2

    out_path = Path(args.out) if args.out else Path(f'{stem}.trace.jsonl')

    mic, sr = sf.read(str(mic_path), dtype='float32')
    lpb, _ = sf.read(str(lpb_path), dtype='float32')
    if mic.ndim > 1: mic = mic[:, 0]
    if lpb.ndim > 1: lpb = lpb[:, 0]
    n = min(len(mic), len(lpb))

    is_movement = (not args.no_movement) and ('_with_movement' in mic_path.name)
    delay_kw = (dict(enable_delay_est=True, delay_est_period_s=0.25, delay_est_init_s=0.2)
                if is_movement else dict(enable_delay_est=False))
    cfg = AecConfig.from_preset('balanced', sample_rate=sr, mode=AecMode.PBFDKF,
                                enable_dtd=False, enable_shadow=True, enable_res=True,
                                use_kalman=True, enable_cng=False, **delay_kw)
    aec = AEC(cfg)

    hop = aec.hop_size
    pos = 0
    idx = 0
    with open(out_path, 'w') as f:
        while pos + hop <= n:
            mic_frame = mic[pos:pos+hop]
            lpb_frame = lpb[pos:pos+hop]
            far_pwr = float(np.mean(lpb_frame ** 2))
            mic_pwr = float(np.mean(mic_frame ** 2))
            out = aec.process(mic_frame, lpb_frame)
            raw_err_pwr = float(np.mean(out ** 2))
            f.write(json.dumps(_frame_record(idx, aec, far_pwr, mic_pwr, raw_err_pwr)) + '\n')
            pos += hop
            idx += 1

    print(f'wrote {idx} frames -> {out_path}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
