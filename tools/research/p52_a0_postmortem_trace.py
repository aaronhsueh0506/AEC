"""P52 A.0 post-mortem forensic trace.

Run on the single regressing case `qNvSMyUSXUyrDGpOw7s6qg_farend_singletalk`.
Read-only — no aec.py logic changes. Captures per-frame:
  - Controller fire timeline (boost_q / reverse_copy / pause_main / counters)
  - Main filter state proxies (W norm, Q max, _alpha_r EMA effective on R)
  - ERLE_main from get_stats
  - DT signals (dt_from_energy / dt_from_shadow / dt_from_coherence)
  - Saturation / far_active / epc_active

Two modes:
  capture  --tag {pre,post}  emit per-frame CSV
  diff     compare two tagged CSVs on fire-frames; report what changed

Usage:
  # Step 1 — controller active (run after `git checkout 3236f6c` or current
  # Phase A branch HEAD which has the revert applied)
  python tools/research/p52_a0_postmortem_trace.py capture \\
      --tag pre --out /tmp/p52_a0_pm/pre.csv

  # Step 3 — controller retired (run after `git checkout eac5325`)
  python tools/research/p52_a0_postmortem_trace.py capture \\
      --tag post --out /tmp/p52_a0_pm/post.csv

  python tools/research/p52_a0_postmortem_trace.py diff \\
      --pre /tmp/p52_a0_pm/pre.csv --post /tmp/p52_a0_pm/post.csv \\
      --out /tmp/p52_a0_pm/diff.json
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from pathlib import Path

import numpy as np
import soundfile as sf

_REPO = Path('/Users/mingyu/Desktop/novatek/SE/AEC')
sys.path.insert(0, str(_REPO / 'python'))

from aec import AEC, AecConfig, AecMode  # noqa: E402
from run_one_case import PRESET_MAP  # noqa: E402

CASE_STEM = 'qNvSMyUSXUyrDGpOw7s6qg_farend_singletalk'
MIC = _REPO / 'wav/aec_challenge_blind/farend_singletalk' / f'{CASE_STEM}_mic.wav'
LPB = _REPO / 'wav/aec_challenge_blind/farend_singletalk' / f'{CASE_STEM}_lpb.wav'


def _instrument_controller(aec):
    """Wrap _regime_handler.update to record per-frame decision + state.

    Returns a list that gets appended to once per process() call.
    """
    log = []
    ctrl = aec._regime_handler
    orig = ctrl.update

    def wrapped(**kw):
        decision = orig(**kw)
        log.append({
            'main_err_smooth': kw['main_err_smooth'],
            'shadow_err_smooth': kw['shadow_err_smooth'],
            'epc_active': bool(kw['epc_active']),
            'saturation_level': kw['saturation_level'],
            'dt_from_energy': kw['dt_from_energy'],
            'dt_from_coherence': kw['dt_from_coherence'],
            'far_pwr': kw['far_pwr'],
            'boost_q': bool(decision.boost_q),
            'reverse_copy': bool(decision.reverse_copy),
            'pause_main': bool(decision.pause_main),
            'main_paused': bool(ctrl.main_paused),
            'copy_counter': int(ctrl.copy_counter),
            'copy_err_baseline': float(ctrl.copy_err_baseline),
        })
        return decision
    ctrl.update = wrapped
    return log


def cmd_capture(args):
    out_dir = Path(args.out).parent
    out_dir.mkdir(parents=True, exist_ok=True)
    mic, sr = sf.read(str(MIC), dtype='float32')
    lpb, _ = sf.read(str(LPB), dtype='float32')
    if mic.ndim > 1: mic = mic[:, 0]
    if lpb.ndim > 1: lpb = lpb[:, 0]
    n = min(len(mic), len(lpb))
    mic = mic[:n].astype(np.float32)
    lpb = lpb[:n].astype(np.float32)

    cfg = AecConfig.from_preset(
        PRESET_MAP['balanced'],
        sample_rate=sr,
        filter_length=832,
        mode=AecMode.PBFDKF,
        enable_cng=True, enable_res=True, enable_shadow=True,
    )
    aec = AEC(cfg)
    hop = aec.hop_size
    ctrl_log = _instrument_controller(aec)

    rows = []
    pos = 0
    fi = 0
    while pos + hop <= n:
        mic_pwr_block = float(np.mean(mic[pos:pos + hop] ** 2))
        aec.process(mic[pos:pos + hop], lpb[pos:pos + hop])
        s = aec.get_stats()
        # Pull controller log entry for THIS frame (controller may skip
        # very-early frames; align by index)
        ctrl = ctrl_log[fi] if fi < len(ctrl_log) else None
        # Main filter weight L2 + max Q proxy
        try:
            w_norm = float(np.linalg.norm(aec.filter.W))
        except Exception:
            w_norm = 0.0
        try:
            q_max = float(np.max(aec.filter.Q)) if hasattr(aec.filter, 'Q') else 0.0
        except Exception:
            q_max = 0.0
        try:
            r_mean = float(np.mean(aec.filter.R)) if hasattr(aec.filter, 'R') else 0.0
        except Exception:
            r_mean = 0.0
        try:
            shadow_w_norm = float(np.linalg.norm(aec.shadow_filter.W)) if aec.shadow_filter is not None else 0.0
        except Exception:
            shadow_w_norm = 0.0
        try:
            p_max_override = int(getattr(aec.filter, '_p_max_override_frames', 0))
        except Exception:
            p_max_override = 0

        rows.append({
            'frame': fi,
            'mic_pwr': mic_pwr_block,
            'erle_inst_db': float(s.erle_inst_db),
            'erle_win_db': float(s.erle_windowed_db),
            'filter_converged': int(s.filter_converged),
            'dt_from_energy': float(s.dt_from_energy),
            'dt_from_shadow': float(s.dt_from_shadow),
            'dt_from_coherence': float(s.dt_from_coherence),
            'epc_active': int(s.epc_active),
            'far_activity': float(s.far_activity),
            'saturation_level': float(s.saturation_level),
            'w_norm': w_norm,
            'shadow_w_norm': shadow_w_norm,
            'q_max': q_max,
            'r_mean': r_mean,
            'p_max_override_frames': p_max_override,
            'boost_q': 0 if ctrl is None else int(ctrl['boost_q']),
            'reverse_copy': 0 if ctrl is None else int(ctrl['reverse_copy']),
            'main_paused': 0 if ctrl is None else int(ctrl['main_paused']),
            'copy_counter': 0 if ctrl is None else ctrl['copy_counter'],
            'copy_err_baseline': 0.0 if ctrl is None else ctrl['copy_err_baseline'],
            'main_err_smooth': 0.0 if ctrl is None else ctrl['main_err_smooth'],
            'shadow_err_smooth': 0.0 if ctrl is None else ctrl['shadow_err_smooth'],
        })
        pos += hop
        fi += 1

    with open(args.out, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f'[{args.tag}] wrote {args.out} frames={len(rows)}', flush=True)

    # Summary
    boost_q_frames = [r['frame'] for r in rows if r['boost_q']]
    reverse_copy_frames = [r['frame'] for r in rows if r['reverse_copy']]
    paused_frames = [r['frame'] for r in rows if r['main_paused']]
    print(f'  boost_q fires: {len(boost_q_frames)} -> frames {boost_q_frames[:20]}')
    print(f'  reverse_copy fires: {len(reverse_copy_frames)} -> frames {reverse_copy_frames[:20]}')
    print(f'  main_paused frames: {len(paused_frames)} ({100*len(paused_frames)/len(rows):.2f}%)')


def cmd_diff(args):
    import csv as _csv
    def load(p):
        with open(p) as f:
            return list(_csv.DictReader(f))
    pre = load(args.pre)
    post = load(args.post)
    n = min(len(pre), len(post))
    boost_frames = [int(r['frame']) for r in pre if int(r['boost_q'])]
    reverse_frames = [int(r['frame']) for r in pre if int(r['reverse_copy'])]
    pause_frames = [int(r['frame']) for r in pre if int(r['main_paused'])]
    fire_frames = sorted(set(boost_frames + pause_frames))

    erle_pre = np.array([float(r['erle_inst_db']) for r in pre[:n]])
    erle_post = np.array([float(r['erle_inst_db']) for r in post[:n]])
    delta = erle_post - erle_pre

    fire_mask = np.zeros(n, dtype=bool)
    for f in fire_frames:
        if f < n:
            fire_mask[f] = True
    # Also include 50-frame window after each fire (effects propagate)
    fire_window = np.zeros(n, dtype=bool)
    for f in fire_frames:
        s, e = max(0, f), min(n, f + 50)
        fire_window[s:e] = True

    summary = {
        'case': CASE_STEM,
        'frames': n,
        'pre_fire_frames': {
            'boost_q_count': len(boost_frames),
            'boost_q_frames': boost_frames[:30],
            'reverse_copy_count': len(reverse_frames),
            'reverse_copy_frames': reverse_frames[:30],
            'pause_main_count': len(pause_frames),
            'pause_main_first_last': [pause_frames[0], pause_frames[-1]] if pause_frames else None,
        },
        'erle_main_delta_summary': {
            'mean_all': float(np.mean(delta)),
            'mean_fire_only': float(np.mean(delta[fire_mask])) if fire_mask.any() else None,
            'mean_fire_window50': float(np.mean(delta[fire_window])) if fire_window.any() else None,
            'mean_outside_fire_window': float(np.mean(delta[~fire_window])) if (~fire_window).any() else None,
        },
        'pre_post_w_norm_at_first_fire': None,
        'pre_post_w_norm_at_last_fire': None,
    }
    if fire_frames:
        f0 = fire_frames[0]
        flast = fire_frames[-1]
        if f0 < n:
            summary['pre_post_w_norm_at_first_fire'] = {
                'frame': f0,
                'pre_w_norm': float(pre[f0]['w_norm']),
                'post_w_norm': float(post[f0]['w_norm']),
                'pre_q_max': float(pre[f0]['q_max']),
                'post_q_max': float(post[f0]['q_max']),
                'pre_erle_db': float(pre[f0]['erle_inst_db']),
                'post_erle_db': float(post[f0]['erle_inst_db']),
            }
        if flast < n:
            summary['pre_post_w_norm_at_last_fire'] = {
                'frame': flast,
                'pre_w_norm': float(pre[flast]['w_norm']),
                'post_w_norm': float(post[flast]['w_norm']),
                'pre_q_max': float(pre[flast]['q_max']),
                'post_q_max': float(post[flast]['q_max']),
                'pre_erle_db': float(pre[flast]['erle_inst_db']),
                'post_erle_db': float(post[flast]['erle_inst_db']),
            }
    # ERLE delta segmented across recording
    seg = max(1, n // 10)
    seg_means = []
    for i in range(0, n, seg):
        seg_means.append(float(np.mean(delta[i:i+seg])))
    summary['erle_delta_per_decile_db'] = seg_means

    # Find frames with biggest regressions
    regressing_idx = np.argsort(delta)[:20].tolist()
    summary['top20_regressing_frames'] = [
        {'frame': int(i),
         'delta_db': float(delta[i]),
         'pre_erle_db': float(erle_pre[i]),
         'post_erle_db': float(erle_post[i]),
         'pre_w_norm': float(pre[i]['w_norm']),
         'post_w_norm': float(post[i]['w_norm']),
         'within_50_of_fire': bool(fire_window[i]),
         'dt_energy_pre': float(pre[i]['dt_from_energy']),
         'main_paused_pre': int(pre[i]['main_paused']),
         } for i in regressing_idx
    ]
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, 'w') as f:
        json.dump(summary, f, indent=2, sort_keys=False)
    print(json.dumps(summary, indent=2, sort_keys=False))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest='cmd', required=True)
    a = sub.add_parser('capture'); a.add_argument('--tag', required=True); a.add_argument('--out', required=True); a.set_defaults(func=cmd_capture)
    b = sub.add_parser('diff'); b.add_argument('--pre', required=True); b.add_argument('--post', required=True); b.add_argument('--out', required=True); b.set_defaults(func=cmd_diff)
    args = ap.parse_args()
    args.func(args)


if __name__ == '__main__':
    main()
