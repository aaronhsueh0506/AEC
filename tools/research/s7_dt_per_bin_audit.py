#!/usr/bin/env python3
"""S7 fire-rate audit for Phase 3B v3 Option alpha (dt_per_bin unification).

Renders the 800-case AEC Challenge corpus in BALANCED preset with the
durable ResFilter audit counter substrate enabled. Aggregates per-bucket
counters into a JSON report under --out.

Design reference: docs/v3_12_phase3b_v3_design.md sec 6.1.

Counters are read-only (verified byte-equal pre-commit on case
0KjzXA3g). Output WAVs are NOT written by this harness (audit-only;
saves I/O time).

Usage:
    python3 tools/research/s7_dt_per_bin_audit.py \\
        --out results/v3_12_s7_audit -j 4
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import soundfile as sf

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(os.path.dirname(_HERE))
sys.path.insert(0, os.path.join(_REPO, 'python'))


def _estimate_delay(mic, ref, sr, max_delay_ms=250.0):
    """Bit-for-bit port of eval_aec_challenge.estimate_delay."""
    max_d = int(sr * max_delay_ms / 1000)
    n = min(len(mic), len(ref))
    mic = mic[:n]
    ref = ref[:n]
    nfft = 1 << int(np.ceil(np.log2(n + max_d)))
    M = np.fft.rfft(mic, n=nfft)
    R = np.fft.rfft(ref, n=nfft)
    xc = np.fft.irfft(M * np.conj(R), n=nfft)
    candidates = xc[: max_d + 1]
    return int(np.argmax(candidates))


def _run_one_audit(mic_path: str, lpb_path: str, stem: str, scenario: str,
                   movement: bool, fl: int, enable_cng: bool) -> dict:
    """Worker: process one case, return audit counters + identity fields.

    Mirrors eval_aec_challenge.py:run_ours processing chain; no WAV output.
    """
    from aec import AEC, AecConfig, AecMode, AecPreset

    t0 = time.time()
    try:
        mic, sr_m = sf.read(mic_path)
        lpb, sr_l = sf.read(lpb_path)
        if sr_m != sr_l:
            return {'stem': stem, 'status': f'sr_mismatch:{sr_m},{sr_l}'}
        sr = int(sr_m)
        mic = mic.astype(np.float32)
        lpb = lpb.astype(np.float32)

        delay = _estimate_delay(mic, lpb, sr)
        n = min(len(mic), len(lpb))
        if delay > 0 and delay < n:
            lpb_aligned = np.zeros(n, dtype=np.float32)
            lpb_aligned[delay:] = lpb[: n - delay]
        else:
            lpb_aligned = lpb[:n]
        mic = mic[:n]

        cfg = AecConfig.from_preset(
            AecPreset.BALANCED,
            sample_rate=sr, mode=AecMode.PBFDKF,
            filter_length=fl,
            enable_dtd=False, enable_shadow=True, enable_res=True,
            enable_cng=enable_cng,
            use_kalman=True,
            enable_delay_est=False,
        )
        np.random.seed(0)
        aec = AEC(cfg)
        aec.enable_res_audit()

        hop = aec.hop_size
        pos = 0
        while pos + hop <= n:
            aec.process(mic[pos: pos + hop], lpb_aligned[pos: pos + hop])
            pos += hop

        ac = aec.get_res_audit()
        return {
            'stem': stem,
            'scenario': scenario,
            'movement': movement,
            'status': 'ok',
            'elapsed_s': time.time() - t0,
            'counters': ac,
        }
    except Exception as e:
        return {'stem': stem, 'status': f'err:{e}'}


def _collect_cases(dataset_dir: str):
    cases = []
    for scenario in ('doubletalk', 'farend_singletalk', 'nearend_singletalk'):
        scen_dir = Path(dataset_dir) / scenario
        if not scen_dir.is_dir():
            continue
        for mic_f in sorted(scen_dir.glob('*_mic.wav')):
            stem = mic_f.name[: -len('_mic.wav')]
            lpb_f = scen_dir / f'{stem}_lpb.wav'
            if not lpb_f.is_file():
                continue
            movement = '_with_movement' in stem
            cases.append((stem, scenario, movement, str(mic_f), str(lpb_f)))
    return cases


def _bucket_of(scenario: str, movement: bool) -> str:
    if scenario == 'farend_singletalk':
        return 'FS_movement' if movement else 'FS_static'
    if scenario == 'doubletalk':
        return 'DT_movement' if movement else 'DT_static'
    return 'NE'


def _aggregate(per_case: list[dict]) -> dict:
    """Sum counters per bucket + global; compute reduction percentages."""
    buckets = ('FS_static', 'FS_movement', 'DT_static', 'DT_movement', 'NE')
    counter_keys = (
        'total_frames',
        's7_legacy_path_frames',
        's7_f31v3_path_frames',
        's7_planb_path_frames',
        's7_legacy_epc_active_frames',
        's7_legacy_not_converged_frames',
        's7_legacy_not_lw_ready_frames',
        's7_legacy_target_other_frames',
        's7_target_fs_bin_count',
        's7_target_fs_bin_legacy_sum',
        's7_target_fs_bin_unified_sum',
        's7_alt_fs_bin_count',
        's7_alt_fs_bin_legacy_sum',
        's7_alt_fs_bin_unified_sum',
    )
    agg = {b: {k: 0 if not k.endswith('_sum') else 0.0
               for k in counter_keys} | {'cases': 0} for b in buckets}
    agg['GLOBAL'] = {k: 0 if not k.endswith('_sum') else 0.0
                     for k in counter_keys} | {'cases': 0}

    for rec in per_case:
        if rec.get('status') != 'ok':
            continue
        b = _bucket_of(rec['scenario'], rec['movement'])
        c = rec['counters']
        for k in counter_keys:
            agg[b][k] += c[k]
            agg['GLOBAL'][k] += c[k]
        agg[b]['cases'] += 1
        agg['GLOBAL']['cases'] += 1

    # Derive percentages and reductions
    for b in list(agg.keys()):
        d = agg[b]
        tot = d['total_frames']
        d['frames'] = tot
        if tot > 0:
            d['s7_legacy_pct'] = d['s7_legacy_path_frames'] / tot * 100
            d['s7_f31v3_pct'] = d['s7_f31v3_path_frames'] / tot * 100
            d['s7_legacy_epc_active_pct'] = (
                d['s7_legacy_epc_active_frames'] / tot * 100)
        else:
            d['s7_legacy_pct'] = 0.0
            d['s7_f31v3_pct'] = 0.0
            d['s7_legacy_epc_active_pct'] = 0.0
        if d['s7_target_fs_bin_count'] > 0:
            lm = d['s7_target_fs_bin_legacy_sum'] / d['s7_target_fs_bin_count']
            um = d['s7_target_fs_bin_unified_sum'] / d['s7_target_fs_bin_count']
            d['s7_target_legacy_mean'] = lm
            d['s7_target_unified_mean'] = um
            d['s7_target_reduction_pct'] = (lm - um) / lm * 100 if lm > 0 else 0.0
        else:
            d['s7_target_legacy_mean'] = 0.0
            d['s7_target_unified_mean'] = 0.0
            d['s7_target_reduction_pct'] = 0.0
        if d['s7_alt_fs_bin_count'] > 0:
            lm = d['s7_alt_fs_bin_legacy_sum'] / d['s7_alt_fs_bin_count']
            um = d['s7_alt_fs_bin_unified_sum'] / d['s7_alt_fs_bin_count']
            d['s7_alt_legacy_mean'] = lm
            d['s7_alt_unified_mean'] = um
            d['s7_alt_reduction_pct'] = (lm - um) / lm * 100 if lm > 0 else 0.0
        else:
            d['s7_alt_legacy_mean'] = 0.0
            d['s7_alt_unified_mean'] = 0.0
            d['s7_alt_reduction_pct'] = 0.0
    return agg


def main() -> int:
    ap = argparse.ArgumentParser(
        description='S7 dt_per_bin pre-implementation fire-rate audit')
    ap.add_argument('--dataset', default='wav/aec_challenge_blind')
    ap.add_argument('--out', required=True, help='Output dir for audit JSON')
    ap.add_argument('--filter', type=int, default=832)
    ap.add_argument('-j', '--jobs', type=int, default=4)
    ap.add_argument('--limit', type=int, default=None)
    ap.add_argument('--no-cng', action='store_true')
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    cases = _collect_cases(args.dataset)
    if args.limit is not None:
        cases = cases[: args.limit]
    enable_cng = not args.no_cng

    print(f'[s7-audit] cases={len(cases)} jobs={args.jobs} '
          f'cng={enable_cng} out={args.out}', flush=True)

    t_start = time.time()
    per_case = []
    n_done = 0
    n_err = 0
    with ProcessPoolExecutor(max_workers=args.jobs) as pool:
        futures = {}
        for stem, scen, mvt, mic_p, lpb_p in cases:
            futures[pool.submit(
                _run_one_audit, mic_p, lpb_p, stem, scen, mvt,
                args.filter, enable_cng,
            )] = stem
        for fut in as_completed(futures):
            rec = fut.result()
            per_case.append(rec)
            n_done += 1
            if rec.get('status') != 'ok':
                n_err += 1
                print(f'[s7-audit] ERR {rec.get("stem")}: '
                      f'{rec.get("status")}', flush=True)
            if n_done % 100 == 0 or n_done == len(cases):
                elapsed = time.time() - t_start
                eta = elapsed / n_done * (len(cases) - n_done)
                print(f'[s7-audit] {n_done}/{len(cases)} '
                      f'errors={n_err} elapsed={elapsed:.0f}s '
                      f'eta={eta:.0f}s', flush=True)

    agg = _aggregate(per_case)
    total = time.time() - t_start

    out_json = {
        'design_ref': 'docs/v3_12_phase3b_v3_design.md §6.1',
        'preset': 'BALANCED',
        'filter_length': args.filter,
        'enable_cng': enable_cng,
        'seed': 0,
        'cases_run': n_done,
        'errors': n_err,
        'elapsed_s': total,
        'per_bucket': agg,
        'per_case': per_case,
    }
    out_path = os.path.join(args.out, 'audit.json')
    with open(out_path, 'w') as f:
        json.dump(out_json, f, indent=2)

    # Print summary table
    print()
    print('=' * 88)
    print(f'{"bucket":<14} {"cases":>5} {"frames":>9} {"legacy%":>8} '
          f'{"F3.1v3%":>8} {"legacy+EPC%":>11} {"target_bins":>11} '
          f'{"reduce%":>8}')
    print('-' * 88)
    for b in ('FS_static', 'FS_movement', 'DT_static', 'DT_movement', 'NE',
              'GLOBAL'):
        d = agg[b]
        print(f'{b:<14} {d["cases"]:>5} {d["frames"]:>9} '
              f'{d["s7_legacy_pct"]:>7.2f}% {d["s7_f31v3_pct"]:>7.2f}% '
              f'{d["s7_legacy_epc_active_pct"]:>10.2f}% '
              f'{d["s7_target_fs_bin_count"]:>11} '
              f'{d["s7_target_reduction_pct"]:>7.1f}%')
    print('=' * 88)
    print(f'[s7-audit] DONE in {total:.0f}s ({total/60:.1f}min) '
          f'errors={n_err}/{n_done}')
    print(f'[s7-audit] Wrote {out_path}')
    return 0 if n_err == 0 else 1


if __name__ == '__main__':
    raise SystemExit(main())
