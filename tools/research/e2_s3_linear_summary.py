#!/usr/bin/env python3
"""E2.S3 linear summary — aggregate linear ERLE per `delay_est_period_s`.

Per Phase 0 discipline: use `_ours_nores.wav` (RES-disabled) so we measure
front-end linear cancellation quality without RES masking.

Metric per case:
  linear_erle_db = 10*log10( mean(mic²) / mean(nores²) ) on signal-active frames

Aggregates over FS_movement cases × 4 periods (50/100/250/500 ms).

Output: results/v3_13_e2_s3_linear_summary/summary.md
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import soundfile as sf


def _signal_active_mask(x, win_size, hop_size, energy_pct=0.10):
    n = len(x)
    n_frames = (n - win_size) // hop_size + 1
    if n_frames <= 0:
        return np.ones(n, dtype=bool)
    energies = np.empty(n_frames, dtype=np.float64)
    for i in range(n_frames):
        seg = x[i * hop_size: i * hop_size + win_size].astype(np.float64)
        energies[i] = np.mean(seg ** 2)
    threshold = np.percentile(energies, energy_pct * 100)
    mask = np.zeros(n, dtype=bool)
    for i in range(n_frames):
        if energies[i] > threshold:
            mask[i * hop_size: i * hop_size + win_size] = True
    return mask


def linear_erle(mic_path, nores_path, sr=16000):
    mic, _ = sf.read(mic_path)
    nores, _ = sf.read(nores_path)
    n = min(len(mic), len(nores))
    mic = mic[:n].astype(np.float64)
    nores = nores[:n].astype(np.float64)
    win = int(0.05 * sr); hop = int(0.01 * sr)
    mask = _signal_active_mask(mic, win, hop)
    if mask.sum() < 100:
        return float('nan')
    mic_pwr = float(np.mean(mic[mask] ** 2))
    nores_pwr = float(np.mean(nores[mask] ** 2))
    if nores_pwr < 1e-12 or mic_pwr < 1e-12:
        return float('nan')
    return 10.0 * np.log10(mic_pwr / nores_pwr)


def main():
    ap = argparse.ArgumentParser(description='E2.S3 linear summary')
    ap.add_argument('--dataset', default='wav/aec_challenge_blind')
    ap.add_argument('--results-root', default='results')
    ap.add_argument('--periods', nargs='+', type=int,
                    default=[50, 100, 250, 500])
    ap.add_argument('--out', default='results/v3_13_e2_s3_linear_summary')
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)

    # FS_movement subdir
    fs_dir = os.path.join(args.dataset, 'farend_singletalk')
    # gather case stems
    stems = []
    for f in sorted(os.listdir(fs_dir)):
        if f.endswith('_mic.wav') and '_with_movement_' in f:
            stems.append(f[:-len('_mic.wav')])
    print(f'[e2-s3-lin] {len(stems)} FS_movement cases', flush=True)

    # period_dir -> {stem: erle_db}
    period_erle = {p: {} for p in args.periods}
    for p in args.periods:
        d = os.path.join(args.results_root, f'v3_13_e2_s3_period_{p}ms')
        for i, stem in enumerate(stems):
            mic = os.path.join(fs_dir, f'{stem}_mic.wav')
            nores = os.path.join(d, f'{stem}_ours_nores.wav')
            if not os.path.isfile(nores):
                period_erle[p][stem] = float('nan')
                continue
            period_erle[p][stem] = linear_erle(mic, nores)
            if (i + 1) % 50 == 0:
                print(f'  p={p}ms {i+1}/{len(stems)}', flush=True)

    # Aggregate
    rows = []
    for stem in stems:
        row = {'stem': stem}
        for p in args.periods:
            row[f'p{p}'] = period_erle[p].get(stem, float('nan'))
        rows.append(row)

    # Summary stats per period
    stats = {}
    for p in args.periods:
        vals = [v for v in period_erle[p].values() if not np.isnan(v)]
        stats[p] = {
            'n': len(vals),
            'mean': float(np.mean(vals)),
            'median': float(np.median(vals)),
            'p10': float(np.percentile(vals, 10)),
            'p90': float(np.percentile(vals, 90)),
        }

    # Baseline = 250ms (production)
    baseline = 250
    delta_stats = {}
    for p in args.periods:
        if p == baseline:
            continue
        deltas = []
        for stem in stems:
            a = period_erle[p].get(stem, float('nan'))
            b = period_erle[baseline].get(stem, float('nan'))
            if not (np.isnan(a) or np.isnan(b)):
                deltas.append(a - b)
        delta_stats[p] = {
            'n': len(deltas),
            'mean_d': float(np.mean(deltas)) if deltas else float('nan'),
            'median_d': float(np.median(deltas)) if deltas else float('nan'),
            'gt_1db': sum(1 for d in deltas if d > 1),
            'gt_3db': sum(1 for d in deltas if d > 3),
            'lt_neg1db': sum(1 for d in deltas if d < -1),
            'lt_neg3db': sum(1 for d in deltas if d < -3),
        }

    md = [
        '# E2.S3 linear summary — `delay_est_period_s` scan on FS_movement',
        '',
        'Linear ERLE = 10·log10(mean(mic²) / mean(nores²)) on signal-active frames.',
        'Path 3 extended pre-align (max_delay_ms=1024) is ON for all variants;',
        'the only knob is the online F-DelayTrack tracker period.',
        '',
        f'Baseline = {baseline}ms (production). n={len(stems)} FS_movement cases.',
        '',
        '## Per-period linear ERLE distribution (dB)',
        '',
        '| Period (ms) | n | mean | median | p10 | p90 |',
        '|---:|---:|---:|---:|---:|---:|',
    ]
    for p in args.periods:
        s = stats[p]
        md.append(f"| {p} | {s['n']} | {s['mean']:.3f} | {s['median']:.3f} | "
                  f"{s['p10']:.3f} | {s['p90']:.3f} |")
    md += ['', f'## Δ vs {baseline}ms baseline', '',
           '| Period (ms) | n | mean Δ | median Δ | >+1 dB | >+3 dB | <−1 dB | <−3 dB |',
           '|---:|---:|---:|---:|---:|---:|---:|---:|']
    for p in args.periods:
        if p == baseline:
            continue
        d = delta_stats[p]
        md.append(f"| {p} | {d['n']} | {d['mean_d']:+.3f} | {d['median_d']:+.3f} | "
                  f"{d['gt_1db']} | {d['gt_3db']} | {d['lt_neg1db']} | {d['lt_neg3db']} |")
    md.append('')

    # Top movers per non-baseline period
    for p in args.periods:
        if p == baseline:
            continue
        deltas = []
        for stem in stems:
            a = period_erle[p].get(stem, float('nan'))
            b = period_erle[baseline].get(stem, float('nan'))
            if not (np.isnan(a) or np.isnan(b)):
                deltas.append((stem, a, b, a - b))
        deltas.sort(key=lambda x: -x[3])
        md.append(f'## {p}ms — top 10 gains vs {baseline}ms')
        md.append('')
        md.append(f'| Stem | {p}ms dB | {baseline}ms dB | Δ dB |')
        md.append('|---|---:|---:|---:|')
        for stem, a, b, d in deltas[:10]:
            md.append(f"| `{stem[:48]}` | {a:.2f} | {b:.2f} | {d:+.2f} |")
        md.append('')
        md.append(f'## {p}ms — top 10 regressions vs {baseline}ms')
        md.append('')
        md.append(f'| Stem | {p}ms dB | {baseline}ms dB | Δ dB |')
        md.append('|---|---:|---:|---:|')
        for stem, a, b, d in sorted(deltas, key=lambda x: x[3])[:10]:
            md.append(f"| `{stem[:48]}` | {a:.2f} | {b:.2f} | {d:+.2f} |")
        md.append('')

    summary = os.path.join(args.out, 'summary.md')
    with open(summary, 'w') as f:
        f.write('\n'.join(md))
    with open(os.path.join(args.out, 'rows.json'), 'w') as f:
        json.dump(rows, f, indent=2)
    with open(os.path.join(args.out, 'stats.json'), 'w') as f:
        json.dump({'stats': stats, 'delta_stats': delta_stats}, f, indent=2)
    print(f'[e2-s3-lin] wrote {summary}', flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
