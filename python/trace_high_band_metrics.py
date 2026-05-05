#!/usr/bin/env python3
"""P1 Phase 1: cohort sweep for high-band NE evidence metrics.

Runs AEC with trace_high_band_metrics=True over sampled cases from each
cohort (FS_static, FS_movement, DT_static, DT_movement, NE) plus
DT-positive / DT-flat sub-cohorts derived from per-case Δ DT deg between
v3.10.4 BALANCED and B2 (minus HF cap).

For each case computes per-frame metrics, takes the median over
"speech-active" frames, then aggregates per-cohort distribution + AUROC.

Output:
  /tmp/p1ph1/case_medians.csv  — per-case median row per metric
  /tmp/p1ph1/cohort_summary.json — cohort tables, AUROC, verdict
  docs/research_log_high_band_evidence.md — final write-up

Run:
  python3 python/trace_high_band_metrics.py
"""
from __future__ import annotations
import argparse
import json
import os
import random
import sys
from glob import glob
from pathlib import Path

import numpy as np
import soundfile as sf

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / 'python'))
from aec import AEC, AecConfig, AecMode, AecPreset  # noqa: E402

WAV_ROOT = REPO / 'wav' / 'aec_challenge_blind'
OUT_DIR = Path('/tmp/p1ph1')
OUT_DIR.mkdir(parents=True, exist_ok=True)

V3104_SCORES = '/tmp/bench_v3104_bal_scores/scores.json'
B2_SCORES = '/tmp/bench_p10_b2_scores/scores.json'

METRICS = ['m_excess_ratio_a05', 'm_excess_ratio_a10', 'm_excess_ratio_a20',
           'm_modulation', 'm_spectral_flatness']

# ---------------------------------------------------------------- cohorts
def list_pairs(subdir, predicate):
    """Return [(stem, mic_path, ref_path), ...] for *_mic.wav under subdir
    where predicate(stem) is True."""
    pairs = []
    for mic in sorted(glob(str(WAV_ROOT / subdir / '*_mic.wav'))):
        stem = os.path.basename(mic).replace('_mic.wav', '')
        if not predicate(stem):
            continue
        ref = mic.replace('_mic.wav', '_lpb.wav')
        if os.path.exists(ref):
            pairs.append((stem, mic, ref))
    return pairs


def build_cohorts():
    fs_static = list_pairs('farend_singletalk',
                           lambda s: 'with_movement' not in s and 'farend_singletalk' in s)
    fs_movement = list_pairs('farend_singletalk',
                             lambda s: 'with_movement' in s)
    dt_static = list_pairs('doubletalk',
                           lambda s: 'with_movement' not in s and 'doubletalk' in s)
    dt_movement = list_pairs('doubletalk',
                             lambda s: 'with_movement' in s)
    ne = list_pairs('nearend_singletalk', lambda s: True)
    return {
        'FS_static': fs_static,
        'FS_movement': fs_movement,
        'DT_static': dt_static,
        'DT_movement': dt_movement,
        'NE': ne,
    }


def build_dt_subcohorts(all_cohorts):
    """Per-DT-case Δ = v3104_deg - B2_deg → DT_positive / DT_flat lists."""
    a = json.load(open(V3104_SCORES))['scores']
    b = json.load(open(B2_SCORES))['scores']
    dt_keys = set(s for s, _, _ in all_cohorts['DT_static']) | \
              set(s for s, _, _ in all_cohorts['DT_movement'])
    pos, flat = [], []
    for k, va in a.items():
        if k not in dt_keys or k not in b:
            continue
        delta = va['deg'] - b[k]['deg']
        for stem, mic, ref in all_cohorts['DT_static'] + all_cohorts['DT_movement']:
            if stem == k:
                if delta >= 0.10:
                    pos.append((stem, mic, ref, delta))
                elif abs(delta) <= 0.02:
                    flat.append((stem, mic, ref, delta))
                break
    pos.sort(key=lambda x: -x[3])
    flat.sort(key=lambda x: abs(x[3]))
    return pos, flat


# -------------------------------------------------------- per-case driver
def run_case(mic_path, ref_path, *, only_second_half=False):
    """Run AEC on a case, return per-frame metric dict (np arrays)."""
    cfg = AecConfig.from_preset(
        AecPreset.BALANCED,
        sample_rate=16000,
        filter_length=832,
        mode=AecMode.PBFDKF,
        enable_res=True,
        enable_cng=True,
        enable_shadow=True,
        trace_high_band_metrics=True,
    )
    aec = AEC(cfg)
    mic, sr_m = sf.read(mic_path)
    ref, sr_r = sf.read(ref_path)
    assert sr_m == 16000 and sr_r == 16000, f"sr mismatch: {sr_m} {sr_r}"
    mic = np.asarray(mic, dtype=np.float32)
    ref = np.asarray(ref, dtype=np.float32)
    n = min(len(mic), len(ref))
    mic, ref = mic[:n], ref[:n]
    hop = aec.hop_size
    rows = {k: [] for k in METRICS}
    rows['mic_pwr'] = []
    pos = 0
    while pos + hop <= n:
        aec.process(mic[pos:pos + hop], ref[pos:pos + hop])
        d = aec._diag
        for m in METRICS:
            rows[m].append(d.get(m, 0.0))
        rows['mic_pwr'].append(float(np.mean(mic[pos:pos + hop] ** 2)))
        pos += hop
    out = {k: np.asarray(v, dtype=np.float64) for k, v in rows.items()}
    n_frames = len(out['mic_pwr'])
    if only_second_half:
        # NE bucket: speech is usually toward end. Restrict to last 50%.
        out = {k: v[n_frames // 2:] for k, v in out.items()}
    return out


def case_median(rows, *, mic_active_only=True):
    """Median per metric over (optionally) active frames."""
    pwr = rows['mic_pwr']
    if mic_active_only and len(pwr) > 4:
        # active = mic_pwr above 20th percentile (drop silence)
        thr = np.percentile(pwr, 20)
        mask = pwr > thr
        if mask.sum() < 4:
            mask = np.ones_like(pwr, dtype=bool)
    else:
        mask = np.ones_like(pwr, dtype=bool)
    return {m: float(np.median(rows[m][mask])) for m in METRICS}


# --------------------------------------------------------------- AUROC
def auroc(pos_vals, neg_vals):
    """AUROC: P(pos > neg). Mann-Whitney-U based."""
    pos = np.asarray(pos_vals, dtype=np.float64)
    neg = np.asarray(neg_vals, dtype=np.float64)
    if len(pos) == 0 or len(neg) == 0:
        return float('nan')
    all_v = np.concatenate([pos, neg])
    ranks = all_v.argsort().argsort().astype(np.float64) + 1
    # ties → average rank (good enough; small N)
    rp = ranks[:len(pos)].sum()
    u = rp - len(pos) * (len(pos) + 1) / 2
    return u / (len(pos) * len(neg))


# -------------------------------------------------------- main sweep
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--n', type=int, default=30,
                    help='cases per cohort (default 30)')
    ap.add_argument('--seed', type=int, default=42)
    args = ap.parse_args()

    rnd = random.Random(args.seed)
    cohorts = build_cohorts()
    pos_sub, flat_sub = build_dt_subcohorts(cohorts)
    print(f'[cohorts] sizes: ' + ', '.join(
        f'{k}={len(v)}' for k, v in cohorts.items())
        + f' | DT_pos={len(pos_sub)} DT_flat={len(flat_sub)}',
        file=sys.stderr)

    plan = []
    for name, lst in cohorts.items():
        sample = rnd.sample(lst, min(args.n, len(lst)))
        for stem, mic, ref in sample:
            plan.append((name, stem, mic, ref, name == 'NE'))
    for stem, mic, ref, _ in pos_sub[:args.n]:
        plan.append(('DT_positive', stem, mic, ref, False))
    for stem, mic, ref, _ in flat_sub[:args.n]:
        plan.append(('DT_flat', stem, mic, ref, False))

    print(f'[plan] {len(plan)} runs', file=sys.stderr)

    medians = []  # (cohort, stem, m_*)
    for i, (cohort, stem, mic, ref, ne_half) in enumerate(plan):
        try:
            rows = run_case(mic, ref, only_second_half=ne_half)
            med = case_median(rows, mic_active_only=True)
            medians.append((cohort, stem, med))
            print(f'[{i+1}/{len(plan)}] {cohort:<12} {stem[:22]:<22} '
                  + ' '.join(f'{m[2:14]}={med[m]:.3f}' for m in METRICS),
                  file=sys.stderr)
        except Exception as e:
            print(f'[FAIL] {cohort} {stem}: {e}', file=sys.stderr)

    # Per-case CSV
    csv_path = OUT_DIR / 'case_medians.csv'
    with open(csv_path, 'w') as fp:
        fp.write('cohort,stem,' + ','.join(METRICS) + '\n')
        for cohort, stem, med in medians:
            fp.write(f'{cohort},{stem},'
                     + ','.join(f'{med[m]:.6f}' for m in METRICS) + '\n')

    # Aggregate per cohort
    by_cohort = {}
    for cohort, stem, med in medians:
        by_cohort.setdefault(cohort, []).append(med)

    summary = {'cohorts': {}, 'auroc': {}, 'verdict': {}}
    for cohort, lst in by_cohort.items():
        per_metric = {}
        for m in METRICS:
            vals = np.asarray([d[m] for d in lst], dtype=np.float64)
            per_metric[m] = {
                'n': len(vals),
                'median': float(np.median(vals)),
                'p25': float(np.percentile(vals, 25)),
                'p75': float(np.percentile(vals, 75)),
            }
        summary['cohorts'][cohort] = per_metric

    # AUROC: positives = NE ∪ DT_positive, negatives = FS_static ∪ FS_movement
    pos_set = by_cohort.get('NE', []) + by_cohort.get('DT_positive', [])
    neg_set = by_cohort.get('FS_static', []) + by_cohort.get('FS_movement', [])
    for m in METRICS:
        pv = [d[m] for d in pos_set]
        nv = [d[m] for d in neg_set]
        summary['auroc'][m] = auroc(pv, nv)

    # Verdict per metric
    cohort_gate = lambda c, m: by_cohort.get(c, [])
    verdict_lines = []
    for m in METRICS:
        s = summary['cohorts']
        fs_s = s.get('FS_static', {}).get(m, {}).get('median', float('inf'))
        fs_m = s.get('FS_movement', {}).get(m, {}).get('median', float('inf'))
        ne_v = s.get('NE', {}).get(m, {}).get('median', -float('inf'))
        dtp = s.get('DT_positive', {}).get(m, {}).get('median', -float('inf'))
        au = summary['auroc'].get(m, float('nan'))
        passed = (fs_s <= 0.15 and fs_m <= 0.15
                  and ne_v >= 0.40 and dtp >= 0.40 and au >= 0.85)
        verdict_lines.append(
            f'{m}: fs_s={fs_s:.3f} fs_m={fs_m:.3f} NE={ne_v:.3f} '
            f'DTpos={dtp:.3f} AUROC={au:.3f} -> '
            + ('PASS' if passed else 'FAIL'))
        summary['verdict'][m] = {
            'passed': passed,
            'fs_static_med': fs_s,
            'fs_movement_med': fs_m,
            'ne_med': ne_v,
            'dt_positive_med': dtp,
            'auroc': au,
        }

    # α-sensitivity
    aurs = [summary['auroc'][f'm_excess_ratio_a{a}']
            for a in ('05', '10', '20')]
    aurs = [a for a in aurs if not (a != a)]  # drop NaN
    a_range = (max(aurs) - min(aurs)) if aurs else float('nan')
    summary['excess_alpha_auroc_range'] = a_range
    summary['excess_alpha_sensitive'] = bool(a_range >= 0.05)

    # DT-positive cohort stems (for repro)
    summary['dt_positive_stems'] = [stem for _, stem, *_ in
                                     [(c, s) for c, s, _ in medians
                                      if c == 'DT_positive']]
    summary['dt_flat_stems'] = [stem for c, s, _ in medians if c == 'DT_flat']

    json_path = OUT_DIR / 'cohort_summary.json'
    with open(json_path, 'w') as fp:
        json.dump(summary, fp, indent=2)

    print('\n=== VERDICT ===', file=sys.stderr)
    for line in verdict_lines:
        print(line, file=sys.stderr)
    print(f'α-AUROC range = {a_range:.3f} '
          f'({"α-sensitive" if a_range >= 0.05 else "α-stable"})',
          file=sys.stderr)
    print(f'\nWrote {csv_path}', file=sys.stderr)
    print(f'Wrote {json_path}', file=sys.stderr)
    return summary


if __name__ == '__main__':
    main()
