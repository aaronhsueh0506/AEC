#!/usr/bin/env python3
"""v3.15 §1.7 — RES gain-floor / cap fire-rate audit on post-§1.2/§1.3 substrate.

Re-runs the v3.13 Phase 3 audit pattern (docs/v3_13_phase3_res_audit_verdict.md)
on the v3.15 substrate (Arc P + R + S-orth.A landed on main; §1.2 closed
CANNOT SHIP, §1.3 deferred). Goal is to determine whether the 5 RES
gain-floor / cap paths in `ResFilter` have shifted in fire-rate vs. the
v3.13 baseline, and to seed candidate refactor entries for v3.16.

The 5 paths audited (per `_stage_gain_compute` + `_stage_gain_postprocess`
in python/aec.py):

  1. spectral_floor   — `g = max(g, spectral_g_min)` after softgate_emr
                         (line ~3156). v3.13 baseline: load-bearing on
                         cohort tail (97% on qNvSMyU).
  2. ne_g_floor       — `spectral_g_min = max(spectral_g_min, ne_g_floor)`
                         (line ~3696). v3.13 baseline: universal floor
                         88-99% across all buckets, low skew (0.13).
  3. epc_dt_cap       — `g = min(g, 0.85)` when `epc_dt` (line ~3201).
                         v3.13 baseline: 0/800 fires (DEAD CODE).
  4. quiet_mask       — `g[quiet_mask] = 1.0` (line ~3206). v3.13
                         baseline: physical noise gate, FS-skewed (51%
                         FS / 30% DT / 6% NE).
  5. hf_cap           — `g[hf_cap_bin+1:] = min(g[hf_cap_bin+1:], cap)`
                         (line ~3275/3287/3295). Three sub-modes:
                         conditional / plan_a_2k / v3.8.3-strict. v3.13
                         baseline: not measured separately (rolled into
                         general post-process).

Detection method (audit-only, ZERO behaviour change):
  Use existing `AecConfig.capture_stages=True` substrate to grab the
  per-bin gain vector AFTER each of the 5 floor sites. Compare consecutive
  stages to detect "any bin modified by floor-i". Fire rate per stage =
  fraction of frames where consecutive vectors differ (≥1 bin modified).
  Per-bucket aggregation matches s9_noise_floor_audit.py.

This script MIRRORS the v3.13 Phase 3 instrumentation (no `aec.py` diff
required) — the per-bin diff is computed in user-space from
`res.get_stage_gains()` output. The v3.13 verdict instrumented this via
a parallel `_diag_floor_fires` dict; that branch was preserved in
worktree-agent-abbbb8ba75683ce4d but is not on this worktree, so we
re-derive fire-rate from the public `_stage_gains` API.

Bench config: preset=BALANCED, fl=832, cng=True, seed=0, j=4. Standard
800-case AEC Challenge corpus.

Usage:
  python3 tools/research/v3_15_res_audit.py --out /tmp/v3_15_res_audit/ -j 4

Outputs:
  /tmp/v3_15_res_audit/per_case/<stem>.json   — per-case fire counts
  /tmp/v3_15_res_audit/audit.json             — aggregated per-bucket
  /tmp/v3_15_res_audit/summary.csv            — flat summary table

Reads: docs/v3_13_phase3_res_audit_verdict.md (baseline for diff column).

Hard-bar (consumed by docs/v3_15_res_audit_and_refactor_plan.md §6):
  - delete-only candidates (fire-rate = 0/800 on this substrate) ship
    in §1.7 as standalone byte-equal-verified commits;
  - refactor candidates with predicted AECMOS Δ ≥ +0.005 → flagged for
    v3.16 arc;
  - if < 3 such candidates → declare RES architecture stable, no v3.16.
"""
from __future__ import annotations

import argparse
import csv
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


# ---------------------------------------------------------------------------
# v3.13 Phase 3 baseline numbers (from docs/v3_13_phase3_res_audit_verdict.md)
# Used for the diff column in the summary CSV. Values = mean fire rate
# (frames where stage modified ≥ 1 bin / total frames) per bucket.
# ---------------------------------------------------------------------------
V3_13_BASELINE = {
    'spectral_floor': {'FS_static': 0.894, 'FS_movement': 0.881,
                       'DT_static': 0.524, 'DT_movement': 0.529,
                       'NE': 0.097, 'cohort_tail': 0.974},
    'ne_g_floor':     {'FS_static': 0.880, 'FS_movement': 0.867,
                       'DT_static': 0.934, 'DT_movement': 0.933,
                       'NE': 0.999, 'cohort_tail': 0.750},
    'epc_dt_cap':     {'FS_static': 0.000, 'FS_movement': 0.000,
                       'DT_static': 0.000, 'DT_movement': 0.000,
                       'NE': 0.000, 'cohort_tail': 0.000},
    'quiet_mask':     {'FS_static': 0.509, 'FS_movement': 0.495,
                       'DT_static': 0.298, 'DT_movement': 0.307,
                       'NE': 0.060, 'cohort_tail': 0.679},
    'hf_cap':         {'FS_static': None,  'FS_movement': None,  # not in v3.13
                       'DT_static': None,  'DT_movement': None,
                       'NE': None,         'cohort_tail': None},
}

# Cohort-tail anchor stem (qNvSMyU... per v3.13 verdict + p52 docs).
COHORT_TAIL_STEMS = {
    'qNvSMyUSXUyrDGpOw7s6qg_farend_singletalk',
    'qNvSMyUSXUyrDGpOw7s6qg_farend_singletalk_with_movement',
    'qNvSMyUSXUyrDGpOw7s6qg_doubletalk',
    'qNvSMyUSXUyrDGpOw7s6qg_doubletalk_with_movement',
}

# Stage keys exposed by ResFilter.get_stage_gains() (capture_stages=True).
# Order matches _stage_gain_compute → _stage_gain_postprocess emission.
STAGE_KEYS = (
    '01_softgate_emr',     # before any floor — reference
    '02_spectral_floor',   # after spectral_g_min floor
    '03_epc_dt_cap',       # after epc_dt 0.85 cap
    '04_quiet_mask',       # after quiet bin pass-through
    '05_3bin_smooth',      # after 3-bin convolve (not a floor — separator)
    '06_hf_cap',           # after HF tail cap
    '07_pre_temporal',     # after divergence override (not a floor — endpoint)
)

# Floor-path → (pre-stage key, post-stage key, semantic label).
# A floor "fires" on a frame if pre[k] != post[k] for ≥1 bin.
# ne_g_floor is computed *into* spectral_g_min before stage 02, so it's
# folded into spectral_floor here; we infer ne_g_floor fire-rate via a
# second pass that also captures the raw spectral_g_min vector.
# (See v3.13 verdict §Method "spectral_floor" row vs "ne_g_floor" row.)
FLOOR_PATHS = (
    ('spectral_floor', '01_softgate_emr', '02_spectral_floor',
     'g = max(g, spectral_g_min)'),
    # ne_g_floor: inferred via a side-channel — see _ne_g_floor_fired() below.
    # Listed here for completeness; aggregator handles it specially.
    ('ne_g_floor',     None,              None,
     'spectral_g_min = max(spectral_g_min, ne_g_floor)'),
    ('epc_dt_cap',     '02_spectral_floor', '03_epc_dt_cap',
     'g = min(g, 0.85) when epc_dt'),
    ('quiet_mask',     '03_epc_dt_cap',    '04_quiet_mask',
     'g[quiet_mask] = 1.0'),
    ('hf_cap',         '05_3bin_smooth',   '06_hf_cap',
     'g[hf_cap_bin+1:] = min(g[hf_cap_bin+1:], cap)'),
)


def _estimate_delay(mic, ref, sr, max_delay_ms=1024.0):
    """GCC-PHAT delay estimate (matches E2 Path 3 default 1024 ms)."""
    max_d = int(max_delay_ms * sr / 1000)
    n = min(len(mic), len(ref))
    m = mic[:n].astype(np.float64)
    r = ref[:n].astype(np.float64)
    fft_size = 1
    while fft_size < 2 * n:
        fft_size *= 2
    mic_spec = np.fft.rfft(m, n=fft_size)
    ref_spec = np.fft.rfft(r, n=fft_size)
    cross = mic_spec * np.conj(ref_spec)
    cross_phat = cross / (np.abs(cross) + 1e-10)
    xcorr_phat = np.fft.irfft(cross_phat, n=fft_size)
    max_search = min(max_d, fft_size // 2)
    peak_val = np.max(np.abs(xcorr_phat[:max_search + 1]))
    peak_idx = int(np.argmax(np.abs(xcorr_phat[:max_search + 1])))
    rms = np.sqrt(np.mean(xcorr_phat[:max_search + 1] ** 2))
    confidence = peak_val / (rms + 1e-10)
    if confidence < 5.0:
        xcorr_plain = np.fft.irfft(cross, n=fft_size)
        delay = int(np.argmax(np.abs(xcorr_plain[:max_search + 1])))
    else:
        delay = peak_idx
    return delay


def _bucket_of(stem, scenario):
    """Map case → bucket. Mirrors s9_noise_floor_audit._bucket_of + cohort tail."""
    if stem in COHORT_TAIL_STEMS:
        return 'cohort_tail'
    movement = '_with_movement' in stem
    if scenario == 'farend_singletalk':
        return 'FS_movement' if movement else 'FS_static'
    if scenario == 'doubletalk':
        return 'DT_movement' if movement else 'DT_static'
    return 'NE'


def _stage_modified(pre_vec, post_vec, atol=1e-7):
    """Return True if any bin differs between pre/post (within atol)."""
    if pre_vec is None or post_vec is None:
        return False
    if pre_vec.shape != post_vec.shape:
        return False
    return bool(np.any(np.abs(post_vec - pre_vec) > atol))


def _stage_gain_delta(pre_vec, post_vec):
    """Mean abs gain delta on bins that were modified. Returns 0.0 if no change."""
    if pre_vec is None or post_vec is None:
        return 0.0
    if pre_vec.shape != post_vec.shape:
        return 0.0
    diff = np.abs(post_vec - pre_vec)
    mask = diff > 1e-7
    if not np.any(mask):
        return 0.0
    return float(np.mean(diff[mask]))


def _ne_g_floor_fired(res):
    """Detect ne_g_floor fire on this frame.

    ne_g_floor is folded INTO spectral_g_min before _stage_gain_compute (line
    ~3696: `spectral_g_min = max(spectral_g_min, ne_g_floor)`), so it does NOT
    show up as a separate stage in get_stage_gains(). We derive its fire
    indicator by reading the cached scalars `_stats_last_ne_g_floor` and
    `_stats_last_spectral_g_min` (set during stage 02). Fired iff
    ne_g_floor > spectral_g_min_pre_floor (i.e. ne_g_floor was the binding
    raise). NOTE: this is a frame-level binary (any bin), not per-bin —
    matches v3.13 verdict's definition.

    Returns (fired:bool, gain_delta:float).
    """
    ne_g  = float(getattr(res, '_stats_last_ne_g_floor', 0.0))
    sp_g  = float(getattr(res, '_stats_last_spectral_g_min', 0.0))
    if ne_g <= 0.0 and sp_g <= 0.0:
        return (False, 0.0)
    fired = ne_g > sp_g + 1e-7
    delta = max(ne_g - sp_g, 0.0) if fired else 0.0
    return (fired, float(delta))


def _run_one(mic_path, lpb_path, stem, scenario, fl, enable_cng):
    """Run one case with capture_stages=True; return per-stage fire counts."""
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

        delay = _estimate_delay(mic, lpb, sr, max_delay_ms=1024.0)
        n = min(len(mic), len(lpb))
        if 0 < delay < n:
            lpb_aligned = np.zeros(n, dtype=np.float32)
            lpb_aligned[delay:] = lpb[: n - delay]
        else:
            lpb_aligned = lpb[:n]
        mic = mic[:n]

        movement = '_with_movement' in stem
        if movement:
            delay_est_kw = dict(enable_delay_est=True,
                                delay_est_period_s=0.25,
                                delay_est_init_s=0.2)
        else:
            delay_est_kw = dict(enable_delay_est=False)

        cfg = AecConfig.from_preset(
            AecPreset.BALANCED,
            sample_rate=sr,
            mode=AecMode.PBFDKF,
            filter_length=fl,
            enable_dtd=False,
            enable_shadow=True,
            enable_res=True,
            enable_cng=enable_cng,
            use_kalman=True,
            capture_stages=True,   # the audit hook
            **delay_est_kw,
        )
        np.random.seed(0)
        aec = AEC(cfg)
        hop = aec.hop_size

        # Per-stage counters: name → frames_fired, sum gain delta, frames_total.
        counters = {p: {'fired': 0, 'delta_sum': 0.0} for p, *_ in FLOOR_PATHS}
        n_frames = 0

        pos = 0
        while pos + hop <= n:
            aec.process(mic[pos: pos + hop], lpb_aligned[pos: pos + hop])
            stage_gains = aec.res.get_stage_gains() if aec.res is not None else {}
            if not stage_gains:
                pos += hop
                continue
            n_frames += 1

            for path_name, pre_key, post_key, _ in FLOOR_PATHS:
                if path_name == 'ne_g_floor':
                    fired, delta = _ne_g_floor_fired(aec.res)
                else:
                    pre = stage_gains.get(pre_key)
                    post = stage_gains.get(post_key)
                    fired = _stage_modified(pre, post)
                    delta = _stage_gain_delta(pre, post) if fired else 0.0
                if fired:
                    counters[path_name]['fired'] += 1
                    counters[path_name]['delta_sum'] += delta
            pos += hop

        return {
            'stem': stem, 'scenario': scenario, 'movement': movement,
            'status': 'ok',
            'elapsed_s': time.time() - t0,
            'n_frames': n_frames,
            'counters': counters,
        }
    except Exception as e:
        return {'stem': stem, 'status': f'err:{e}'}


def _collect_cases(dataset_dir):
    cases = []
    for scenario in ('doubletalk', 'farend_singletalk', 'nearend_singletalk'):
        sd = Path(dataset_dir) / scenario
        if not sd.is_dir():
            continue
        for mic_f in sorted(sd.glob('*_mic.wav')):
            stem = mic_f.name[: -len('_mic.wav')]
            lpb_f = sd / f'{stem}_lpb.wav'
            if not lpb_f.is_file():
                continue
            cases.append((stem, scenario, str(mic_f), str(lpb_f)))
    return cases


def _aggregate(per_case):
    """Aggregate per-case fire counts into per-bucket fire rate + gain delta.

    Output schema:
      agg[bucket][path_name] = {
          'cases': N, 'frames': M,
          'frames_fired': F, 'fire_rate': F/M,
          'mean_delta_when_fired': sum_delta / F,
      }
    """
    buckets = ('FS_static', 'FS_movement', 'DT_static', 'DT_movement',
               'NE', 'cohort_tail', 'GLOBAL')
    paths = [p for p, *_ in FLOOR_PATHS]

    agg = {b: {p: {'cases': 0, 'frames': 0, 'frames_fired': 0,
                   'delta_sum': 0.0}
               for p in paths}
           for b in buckets}

    for rec in per_case:
        if rec.get('status') != 'ok':
            continue
        b = _bucket_of(rec['stem'], rec['scenario'])
        n_frames = rec['n_frames']
        for path_name in paths:
            c = rec['counters'][path_name]
            agg[b][path_name]['cases'] += 1
            agg[b][path_name]['frames'] += n_frames
            agg[b][path_name]['frames_fired'] += c['fired']
            agg[b][path_name]['delta_sum'] += c['delta_sum']
            if b != 'GLOBAL':
                agg['GLOBAL'][path_name]['cases'] += 1
                agg['GLOBAL'][path_name]['frames'] += n_frames
                agg['GLOBAL'][path_name]['frames_fired'] += c['fired']
                agg['GLOBAL'][path_name]['delta_sum'] += c['delta_sum']

    for b in buckets:
        for p in paths:
            d = agg[b][p]
            d['fire_rate'] = (d['frames_fired'] / d['frames']
                              if d['frames'] > 0 else 0.0)
            d['mean_delta_when_fired'] = (d['delta_sum'] / d['frames_fired']
                                          if d['frames_fired'] > 0 else 0.0)
    return agg


def _write_summary_csv(agg, out_path):
    """Flat summary CSV: rows = (bucket, path); cols = fire_rate, delta, baseline, diff."""
    buckets = ('FS_static', 'FS_movement', 'DT_static', 'DT_movement',
               'NE', 'cohort_tail', 'GLOBAL')
    paths = [p for p, *_ in FLOOR_PATHS]
    with open(out_path, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['bucket', 'path', 'cases', 'frames', 'frames_fired',
                    'fire_rate', 'mean_delta_when_fired',
                    'v3_13_baseline_rate', 'diff_vs_baseline'])
        for b in buckets:
            for p in paths:
                d = agg[b][p]
                base = V3_13_BASELINE.get(p, {}).get(b)
                diff = (d['fire_rate'] - base) if base is not None else None
                w.writerow([
                    b, p, d['cases'], d['frames'], d['frames_fired'],
                    f"{d['fire_rate']:.4f}",
                    f"{d['mean_delta_when_fired']:.4f}",
                    f"{base:.4f}" if base is not None else '',
                    f"{diff:+.4f}" if diff is not None else '',
                ])


def _print_summary(agg):
    """Console summary table — fire rate per bucket per path + diff vs v3.13."""
    print()
    print('=' * 110)
    print('v3.15 §1.7 — RES gain-floor / cap fire-rate audit '
          '(post-§1.2/§1.3 substrate)')
    print('-' * 110)
    print(f"{'path':<16} {'bucket':<14} {'fire_rate':>10} "
          f"{'v3.13 base':>11} {'diff':>9} "
          f"{'mean Δgain':>11} {'frames':>10}")
    print('-' * 110)
    for path_name, *_ in FLOOR_PATHS:
        for b in ('FS_static', 'FS_movement', 'DT_static', 'DT_movement',
                  'NE', 'cohort_tail', 'GLOBAL'):
            d = agg[b][path_name]
            base = V3_13_BASELINE.get(path_name, {}).get(b)
            base_s = f"{base:.3f}" if base is not None else '   n/a'
            diff_s = (f"{d['fire_rate'] - base:+.3f}"
                      if base is not None else '    -')
            print(f"{path_name:<16} {b:<14} "
                  f"{d['fire_rate']:>9.3f}  {base_s:>10}  {diff_s:>8}  "
                  f"{d['mean_delta_when_fired']:>10.4f}  "
                  f"{d['frames']:>10}")
        print('-' * 110)
    print('=' * 110)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--dataset', default=os.path.join(_REPO, 'wav', 'aec_challenge_blind'),
                    help='AEC Challenge dataset root.')
    ap.add_argument('--out', default='/tmp/v3_15_res_audit/',
                    help='Output directory (per-case JSONs + audit.json + summary.csv).')
    ap.add_argument('--filter', type=int, default=832, help='filter_length in samples (default 832 = 52 ms).')
    ap.add_argument('-j', '--jobs', type=int, default=4)
    ap.add_argument('--n-cases', type=int, default=800,
                    help='Cap total cases (for smoke tests; default 800 = full).')
    ap.add_argument('--no-cng', action='store_true')
    ap.add_argument('--no-percase-json', action='store_true',
                    help='Skip per-case JSON dump (saves disk on full bench).')
    args = ap.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    percase_dir = out_dir / 'per_case'
    percase_dir.mkdir(parents=True, exist_ok=True)

    cases = _collect_cases(args.dataset)
    if args.n_cases is not None and args.n_cases < len(cases):
        cases = cases[: args.n_cases]
    enable_cng = not args.no_cng

    print(f'[v3.15-res-audit] cases={len(cases)} jobs={args.jobs} '
          f'fl={args.filter} cng={enable_cng} out={args.out}', flush=True)

    t_start = time.time()
    per_case = []
    n_done = 0
    n_err = 0

    with ProcessPoolExecutor(max_workers=args.jobs) as pool:
        futures = {pool.submit(_run_one, mic_p, lpb_p, stem, scen,
                               args.filter, enable_cng): stem
                   for stem, scen, mic_p, lpb_p in cases}
        for fut in as_completed(futures):
            rec = fut.result()
            per_case.append(rec)
            n_done += 1
            if rec.get('status') != 'ok':
                n_err += 1
                print(f"[v3.15-res-audit] ERR {rec.get('stem')}: "
                      f"{rec.get('status')}", flush=True)
            elif not args.no_percase_json:
                with open(percase_dir / f"{rec['stem']}.json", 'w') as f:
                    json.dump(rec, f, indent=2)
            if n_done % 100 == 0 or n_done == len(cases):
                elapsed = time.time() - t_start
                eta = elapsed / n_done * (len(cases) - n_done) if n_done else 0
                print(f'[v3.15-res-audit] {n_done}/{len(cases)} '
                      f'errors={n_err} elapsed={elapsed:.0f}s '
                      f'eta={eta:.0f}s', flush=True)

    agg = _aggregate(per_case)
    total = time.time() - t_start

    out_json = {
        'version': 'v3.15.§1.7',
        'baseline_ref': 'docs/v3_13_phase3_res_audit_verdict.md',
        'preset': 'BALANCED',
        'filter_length': args.filter,
        'enable_cng': enable_cng,
        'seed': 0,
        'cases_run': n_done,
        'errors': n_err,
        'elapsed_s': total,
        'paths_audited': [
            {'name': p, 'pre_key': pre, 'post_key': post, 'desc': desc}
            for p, pre, post, desc in FLOOR_PATHS
        ],
        'per_bucket': agg,
    }
    audit_path = out_dir / 'audit.json'
    with open(audit_path, 'w') as f:
        json.dump(out_json, f, indent=2)

    csv_path = out_dir / 'summary.csv'
    _write_summary_csv(agg, csv_path)

    _print_summary(agg)

    print(f'[v3.15-res-audit] DONE in {total:.0f}s ({total / 60:.1f}min) '
          f'errors={n_err}/{n_done}')
    print(f'[v3.15-res-audit] audit.json:  {audit_path}')
    print(f'[v3.15-res-audit] summary.csv: {csv_path}')
    print(f'[v3.15-res-audit] per-case:    {percase_dir}/')
    print()
    print('Next step: fill findings + refactor candidates into '
          'docs/v3_15_res_audit_and_refactor_plan.md.')

    return 0 if n_err == 0 else 1


if __name__ == '__main__':
    raise SystemExit(main())
