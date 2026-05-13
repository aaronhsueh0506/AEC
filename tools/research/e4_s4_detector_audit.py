#!/usr/bin/env python3
"""E4.S4 — 800-case detector fire-rate audit.

Per E4 plan §3.2 S4 acceptance, after E4.S3.1 gate refinement
(commit 798c11b), validate detector fire-rate distribution across
all 800 cases × 5 buckets.

Acceptance (from design lock):
  - NL cohort (5 listen-validated) ≥ 30% voice-active frames per case
  - NE bucket fires < 1% voice-active frames per case (random sample)
  - FS / DT buckets: distributional check (no specific bar)

This renders all 800 cases with `e4_nlp_enabled=True`. Detector is
audit-only (byte-equal output vs baseline) so the render output is
identical to v3_11_candidate; this audit only measures the detector
fire-rate (which is recorded into self._diag per hop).

Output:
  - per-case JSON: fire_count, active_count, max_conf, mean_conf,
    duration_s, bucket
  - summary: bucket distribution + NL cohort hit table + NE
    bucket worst-20 (highest fire rate)

Usage:
  python3 tools/research/e4_s4_detector_audit.py \\
      --dataset wav/aec_challenge_blind \\
      --out results/v3_13_e4_s4_audit \\
      --workers 4
"""
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import sys
import time
from pathlib import Path

import numpy as np
import soundfile as sf

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(os.path.dirname(_HERE))
sys.path.insert(0, os.path.join(_REPO, 'python'))


# 5 NL-validated listen cases (from E4.S1)
NL_COHORT = {
    'Gsy0lC5QSUi540hiax9XtA_farend_singletalk',
    '9xjhiFbGo06hdQIsHTS6qA_farend_singletalk',
    'IrQvqOTCmEWMXn9k2ICtRQ_farend_singletalk_with_movement',
    'WTdBhXa080WJEeGDde9BGA_farend_singletalk',
    'm4789fdio0q92zjf9gvh1Q_farend_singletalk',
}


def bucket_of(stem: str) -> str:
    if '_doubletalk' in stem:
        return 'DT_movement' if '_with_movement' in stem else 'DT_static'
    if '_farend_singletalk' in stem:
        return 'FS_movement' if '_with_movement' in stem else 'FS_static'
    if '_nearend_singletalk' in stem:
        return 'NE'
    return 'unknown'


def subdir_of(stem: str) -> str:
    if '_doubletalk' in stem:
        return 'doubletalk'
    if '_farend_singletalk' in stem:
        return 'farend_singletalk'
    if '_nearend_singletalk' in stem:
        return 'nearend_singletalk'
    return ''


def audit_one(args):
    """Worker function — renders one case with detector enabled."""
    stem, dataset = args
    from aec import AEC, AecConfig, AecMode, AecPreset
    from eval_aec_challenge import estimate_delay

    t0 = time.time()
    sub = subdir_of(stem)
    bucket = bucket_of(stem)
    mic_p = os.path.join(dataset, sub, f'{stem}_mic.wav')
    lpb_p = os.path.join(dataset, sub, f'{stem}_lpb.wav')
    if not (os.path.isfile(mic_p) and os.path.isfile(lpb_p)):
        return {'stem': stem, 'error': 'missing_input'}

    mic, sr = sf.read(mic_p)
    lpb, _ = sf.read(lpb_p)
    sr = int(sr)
    mic = mic.astype(np.float32)
    lpb = lpb.astype(np.float32)
    # Use production GCC pre-align (now max=1024 per E2.S5 default)
    d = estimate_delay(mic, lpb, sr)
    n = min(len(mic), len(lpb))
    if 0 < d < n:
        lpb_a = np.zeros(n, dtype=np.float32)
        lpb_a[d:] = lpb[: n - d]
        lpb = lpb_a
    mic = mic[:n]
    lpb = lpb[:n]

    cfg = AecConfig.from_preset(
        AecPreset.BALANCED, sample_rate=sr, mode=AecMode.PBFDKF,
        filter_length=832,
        enable_dtd=False, enable_shadow=True, enable_res=True,
        enable_cng=True, use_kalman=True,
        e4_nlp_enabled=True,
    )
    np.random.seed(0)
    aec = AEC(cfg)
    hop = aec.hop_size

    fire_count = 0
    active_count = 0
    max_conf = 0.0
    sum_conf = 0.0
    pos = 0
    while pos + hop <= n:
        aec.process(mic[pos: pos + hop], lpb[pos: pos + hop])
        det = aec.nl_detector
        if det is None:
            break
        if det._buf_filled >= det._win_samples:
            active_count += 1
            if det._nl_confidence_last > 0:
                fire_count += 1
                sum_conf += det._nl_confidence_last
                if det._nl_confidence_last > max_conf:
                    max_conf = det._nl_confidence_last
        pos += hop

    return {
        'stem': stem,
        'bucket': bucket,
        'is_nl_cohort': stem in NL_COHORT,
        'duration_s': float(n / sr),
        'gcc_delay_samp': int(d),
        'fire_count': int(fire_count),
        'active_count': int(active_count),
        'fire_rate': float(fire_count / active_count) if active_count else 0.0,
        'max_conf': float(max_conf),
        'mean_conf': float(sum_conf / fire_count) if fire_count else 0.0,
        'elapsed_s': float(time.time() - t0),
    }


def main():
    ap = argparse.ArgumentParser(description='E4.S4 800-case detector audit')
    ap.add_argument('--dataset', default='wav/aec_challenge_blind')
    ap.add_argument('--out', required=True)
    ap.add_argument('--workers', type=int, default=4)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    # Discover all 800 cases
    stems = []
    for sub in ('doubletalk', 'farend_singletalk', 'nearend_singletalk'):
        sub_p = os.path.join(args.dataset, sub)
        if not os.path.isdir(sub_p):
            continue
        for f in sorted(os.listdir(sub_p)):
            if f.endswith('_mic.wav'):
                stems.append(f[:-len('_mic.wav')])
    print(f'[e4-s4] {len(stems)} cases, workers={args.workers}', flush=True)

    work = [(s, args.dataset) for s in stems]
    rows = []
    t_start = time.time()

    if args.workers <= 1:
        for i, w in enumerate(work):
            r = audit_one(w)
            rows.append(r)
            if (i + 1) % 50 == 0:
                print(f'[e4-s4] {i+1}/{len(work)} '
                      f'elapsed={time.time()-t_start:.0f}s', flush=True)
    else:
        with mp.Pool(args.workers) as pool:
            for i, r in enumerate(pool.imap_unordered(audit_one, work)):
                rows.append(r)
                if (i + 1) % 50 == 0:
                    print(f'[e4-s4] {i+1}/{len(work)} '
                          f'elapsed={time.time()-t_start:.0f}s', flush=True)
    print(f'[e4-s4] done in {time.time()-t_start:.0f}s', flush=True)

    with open(os.path.join(args.out, 'rows.json'), 'w') as f:
        json.dump(rows, f, indent=2)

    # Aggregate by bucket
    buckets = ('FS_static', 'FS_movement', 'DT_static', 'DT_movement', 'NE')
    md = [
        '# E4.S4 800-case detector fire-rate audit',
        '',
        'Detector: SubtractiveNLP pitch tracker (E4.S3 + S3.1 RMS gate).',
        'Audit-only — detector is pure observer; AEC output byte-equal to baseline.',
        '',
        '## Bucket aggregate',
        '',
        '| Bucket | n | mean fire rate | median | p25 | p75 | p95 | max |',
        '|---|---:|---:|---:|---:|---:|---:|---:|',
    ]
    bucket_stats = {}
    for b in buckets:
        vals = [r['fire_rate'] for r in rows if r.get('bucket') == b
                and 'error' not in r]
        if not vals:
            continue
        arr = np.array(vals)
        bucket_stats[b] = {
            'n': len(vals),
            'mean': float(np.mean(arr)),
            'median': float(np.median(arr)),
            'p25': float(np.percentile(arr, 25)),
            'p75': float(np.percentile(arr, 75)),
            'p95': float(np.percentile(arr, 95)),
            'max': float(np.max(arr)),
            'n_above_5pct': int(np.sum(arr > 0.05)),
            'n_above_30pct': int(np.sum(arr > 0.30)),
        }
        s = bucket_stats[b]
        md.append(f"| {b} | {s['n']} | {s['mean']*100:.2f}% | "
                  f"{s['median']*100:.2f}% | {s['p25']*100:.2f}% | "
                  f"{s['p75']*100:.2f}% | {s['p95']*100:.2f}% | "
                  f"{s['max']*100:.2f}% |")
    md.append('')

    md.append('## NL cohort (5 listen-validated) — required ≥30% per case')
    md.append('')
    md.append('| Stem | Bucket | Fire rate | Max conf | Mean conf | PASS? |')
    md.append('|---|---|---:|---:|---:|:---:|')
    nl_pass = 0
    for r in rows:
        if r.get('is_nl_cohort'):
            ok = r['fire_rate'] >= 0.30
            if ok:
                nl_pass += 1
            md.append(f"| `{r['stem'][:48]}` | {r['bucket']} | "
                      f"{r['fire_rate']*100:.1f}% | {r['max_conf']:.2f} | "
                      f"{r['mean_conf']:.2f} | {'YES' if ok else 'NO'} |")
    md.append('')
    md.append(f'NL cohort acceptance: {nl_pass}/5 cases at ≥30% fire rate')
    md.append('')

    md.append('## NE bucket — required <1% per case')
    md.append('')
    ne_rows = [r for r in rows if r.get('bucket') == 'NE' and 'error' not in r]
    ne_over_1pct = sum(1 for r in ne_rows if r['fire_rate'] > 0.01)
    ne_over_5pct = sum(1 for r in ne_rows if r['fire_rate'] > 0.05)
    md.append(f'NE total: {len(ne_rows)}, '
              f'cases >1% fire: {ne_over_1pct}, '
              f'cases >5% fire: {ne_over_5pct}')
    md.append('')
    md.append('Worst-20 NE cases by fire rate:')
    md.append('')
    md.append('| Stem | Fire rate | Max conf |')
    md.append('|---|---:|---:|')
    for r in sorted(ne_rows, key=lambda x: -x['fire_rate'])[:20]:
        md.append(f"| `{r['stem'][:48]}` | {r['fire_rate']*100:.2f}% | "
                  f"{r['max_conf']:.2f} |")
    md.append('')

    md.append('## FS bucket — top 20 fire rates (NL detection candidates)')
    md.append('')
    md.append('| Stem | Bucket | Fire rate | Max conf | NL cohort? |')
    md.append('|---|---|---:|---:|:---:|')
    fs_rows = [r for r in rows if r.get('bucket', '').startswith('FS_')
               and 'error' not in r]
    for r in sorted(fs_rows, key=lambda x: -x['fire_rate'])[:20]:
        in_nl = 'YES' if r.get('is_nl_cohort') else ''
        md.append(f"| `{r['stem'][:48]}` | {r['bucket']} | "
                  f"{r['fire_rate']*100:.2f}% | {r['max_conf']:.2f} | "
                  f"{in_nl} |")
    md.append('')

    md.append('## DT bucket — top 20 fire rates (potential NE false positive)')
    md.append('')
    md.append('| Stem | Bucket | Fire rate | Max conf |')
    md.append('|---|---|---:|---:|')
    dt_rows = [r for r in rows if r.get('bucket', '').startswith('DT_')
               and 'error' not in r]
    for r in sorted(dt_rows, key=lambda x: -x['fire_rate'])[:20]:
        md.append(f"| `{r['stem'][:48]}` | {r['bucket']} | "
                  f"{r['fire_rate']*100:.2f}% | {r['max_conf']:.2f} |")
    md.append('')

    out_md = os.path.join(args.out, 'summary.md')
    with open(out_md, 'w') as f:
        f.write('\n'.join(md))
    with open(os.path.join(args.out, 'bucket_stats.json'), 'w') as f:
        json.dump(bucket_stats, f, indent=2)
    print(f'[e4-s4] wrote {out_md}', flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
