#!/usr/bin/env python3
"""v3.15 §1.4.S4 Arc G — fire-count audit on 80-case representative sample.

Purpose: identify the top 5 gain-change cases (per-band ERL discontinuity)
to use as nores listen cohort for §1.4.S4 / §1.4.S5 verification. The
Arc G detector fires when fast/slow per-band ERL EMA ratio >= drift_ratio
on a band that is NOT in PathChangeRegimeHandler EPC window. Sudden
mic-gain shifts produce per-band ERL jumps; this audit ranks cases by
total per-band fire count.

Strategy:
  - Run AEC with arc_g_per_band_w_reset=True + arc_g_drift_ratio=2.0 (low
    threshold = high recall, will over-detect on speech-spectrum
    variation; we filter-by-rank rather than threshold-tune here)
  - Process each case end-to-end; capture aec._arc_g_fire_count[3] after
  - Sort cases by total fires (sum across 3 bands)
  - Pick top 5 for nores listen
  - Top 5 control cases (zero or minimal fires) for false-positive
    audit baseline

Bench config: preset=BALANCED, filter_length=832, cng=True, seed=0
(matches eval_aec_challenge.py bench convention).

Usage:
    python3 tools/research/v3_15_arc_g_fire_audit.py \\
        --dataset-dir wav/aec_challenge_blind \\
        --out results/v3_15_arc_g_fire_audit

Output:
    <out>/per_case_fires.json    # all cases with fire counts
    <out>/top_candidates.md      # top-5 + control-5 ranked
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import soundfile as sf

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(os.path.dirname(_HERE))
sys.path.insert(0, os.path.join(_REPO, 'python'))

# Cases to audit. Mix: movement-heavy (likely candidates) + random
# non-movement (control). Empirical case selection avoids cherry-picking.
# We pull all FS_movement + DT_movement + 20 random non-movement cases.

def collect_cases(dataset_dir: Path, max_movement: int = 40, n_control: int = 20):
    """Return list of (case_stem, bucket, mic_path, lpb_path)."""
    cases = []
    for bucket in ('farend_singletalk', 'doubletalk', 'nearend_singletalk'):
        bucket_dir = dataset_dir / bucket
        if not bucket_dir.exists():
            continue
        mic_files = sorted([f for f in bucket_dir.iterdir()
                            if f.name.endswith('_mic.wav')])
        for mf in mic_files:
            stem = mf.name.replace('_mic.wav', '')
            lpb = bucket_dir / f"{stem}_lpb.wav"
            if not lpb.exists():
                continue
            is_movement = '_with_movement' in stem
            cases.append((stem, bucket, str(mf), str(lpb), is_movement))
    movement = [c for c in cases if c[4]]
    non_movement = [c for c in cases if not c[4]]
    rng = np.random.default_rng(42)
    movement_sample = movement[:max_movement]
    if len(non_movement) > n_control:
        non_movement_idx = rng.choice(len(non_movement), n_control, replace=False)
        non_movement_sample = [non_movement[i] for i in sorted(non_movement_idx.tolist())]
    else:
        non_movement_sample = non_movement
    return movement_sample + non_movement_sample


def run_one(stem: str, bucket: str, mic_path: str, lpb_path: str,
            drift_ratio: float = 2.0, fl: int = 832):
    """Process one case with Arc G ON + low threshold; return fire counts."""
    from aec import AEC, AecConfig, AecPreset

    mic, sr = sf.read(mic_path)
    lpb, _ = sf.read(lpb_path)
    n = min(len(mic), len(lpb))
    mic = mic[:n].astype(np.float32)
    lpb = lpb[:n].astype(np.float32)

    cfg = AecConfig.from_preset(
        AecPreset.BALANCED,
        sample_rate=sr,
        filter_length=fl,
        enable_res=True,
        enable_cng=True,
    )
    cfg.arc_g_per_band_w_reset = True
    cfg.arc_g_drift_ratio = drift_ratio

    np.random.seed(0)  # match eval_aec_challenge.py per-case CNG seed
    aec = AEC(cfg)
    hop = aec.hop_size
    pos = 0
    while pos + hop <= n:
        aec.process(mic[pos:pos+hop], lpb[pos:pos+hop])
        pos += hop

    fires = aec._arc_g_fire_count.copy()
    return {
        'stem': stem,
        'bucket': bucket,
        'is_movement': '_with_movement_' in stem,
        'fires_lf': int(fires[0]),
        'fires_mf': int(fires[1]),
        'fires_hf': int(fires[2]),
        'fires_total': int(fires.sum()),
        'duration_sec': n / sr,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset-dir', required=True, type=Path)
    ap.add_argument('--out', required=True, type=Path)
    ap.add_argument('--drift-ratio', type=float, default=2.0,
                    help='Arc G detector threshold (low = high recall)')
    ap.add_argument('--max-movement', type=int, default=40)
    ap.add_argument('--n-control', type=int, default=20)
    ap.add_argument('--limit', type=int, default=None,
                    help='Limit total cases (for debugging)')
    args = ap.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)

    cases = collect_cases(args.dataset_dir, args.max_movement, args.n_control)
    if args.limit:
        cases = cases[:args.limit]
    print(f'Auditing {len(cases)} cases (drift_ratio={args.drift_ratio}) ...')

    results = []
    t0 = time.time()
    for i, (stem, bucket, mic_path, lpb_path, _is_mvmt) in enumerate(cases):
        try:
            r = run_one(stem, bucket, mic_path, lpb_path, args.drift_ratio)
            results.append(r)
            if r['fires_total'] > 0:
                print(f"  [{i+1}/{len(cases)}] {bucket[:2]:>2} {stem[:40]:<40} "
                      f"fires=L{r['fires_lf']}/M{r['fires_mf']}/H{r['fires_hf']}={r['fires_total']}")
        except Exception as e:
            print(f"  FAIL {stem}: {e}")
    elapsed = time.time() - t0
    print(f'Done in {elapsed:.1f}s.')

    # Save raw results
    (args.out / 'per_case_fires.json').write_text(json.dumps(results, indent=2))

    # Sort by total fires, pick top 5 candidates + 5 control (zero fires)
    sorted_by_fires = sorted(results, key=lambda r: -r['fires_total'])
    top5 = sorted_by_fires[:5]
    zero_fire = [r for r in results if r['fires_total'] == 0]
    control5 = zero_fire[:5] if len(zero_fire) >= 5 else zero_fire

    md = ['# Arc G fire-count audit — top candidates\n',
          f'Drift ratio: {args.drift_ratio} (low threshold = high recall)\n',
          f'Cases audited: {len(results)}\n',
          f'Cases with non-zero fires: {sum(1 for r in results if r["fires_total"] > 0)}\n',
          '\n## Top 5 candidates (by total fires)\n\n',
          '| Rank | Bucket | Stem | LF | MF | HF | Total | Movement |\n',
          '|---|---|---|---:|---:|---:|---:|---|\n']
    for i, r in enumerate(top5, 1):
        md.append(f"| {i} | {r['bucket']} | `{r['stem']}` | {r['fires_lf']} | "
                  f"{r['fires_mf']} | {r['fires_hf']} | {r['fires_total']} | "
                  f"{'Y' if r['is_movement'] else 'N'} |\n")
    md.append('\n## Control 5 (zero fires — false-positive baseline)\n\n')
    md.append('| Rank | Bucket | Stem | Movement |\n|---|---|---|---|\n')
    for i, r in enumerate(control5, 1):
        md.append(f"| {i} | {r['bucket']} | `{r['stem']}` | "
                  f"{'Y' if r['is_movement'] else 'N'} |\n")
    md.append(f'\n## Distribution\n\n')
    fires_dist = sorted([r['fires_total'] for r in results], reverse=True)
    md.append(f'p99: {fires_dist[len(fires_dist)//100] if len(fires_dist) > 100 else fires_dist[0]}\n')
    md.append(f'p95: {fires_dist[len(fires_dist)//20]}\n')
    md.append(f'p50: {fires_dist[len(fires_dist)//2]}\n')
    md.append(f'mean: {sum(fires_dist) / len(fires_dist):.2f}\n')
    md.append(f'max: {fires_dist[0]}\n')
    md.append(f'zero-fire fraction: {sum(1 for f in fires_dist if f == 0)/len(fires_dist):.1%}\n')

    (args.out / 'top_candidates.md').write_text(''.join(md))
    print(f'Top 5 + control 5 saved to {args.out / "top_candidates.md"}')

    # Print summary to stdout
    print('\n--- TOP 5 CANDIDATES ---')
    for i, r in enumerate(top5, 1):
        print(f"  {i}. {r['stem'][:50]:<50} ({r['bucket'][:2]}) fires={r['fires_total']}")


if __name__ == '__main__':
    main()
