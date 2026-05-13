#!/usr/bin/env python3
"""E2.S2 — linear-only comparison: baseline (max=250) vs Path 3 (max=1024).

User insight (2026-05-13): the 800-case AECMOS scores compare
post-RES output. The DT regression (Δdeg -0.050 on DT_static) might
be RES over-suppression, NOT linear filter degradation. To answer
"is the linear filter actually doing better under Path 3?" we need
to compare linear-only output (`*_ours_nores.wav`).

Both baseline (`results/v3_11_candidate/`) and Path 3
(`results/v3_13_e2_s2_path3_800case/`) bench runs already produce
`*_ours_nores.wav` (linear residual, RES disabled).

Metric per case:
  linear_pwr_ratio = 10*log10( mean(mic²) / mean(nores²) )
                   = linear ERLE in FS (mic = echo only)
                   = degree of cancellation in DT (mic = NE + echo)

For FS cases:
  Δlinear_pwr_ratio > 0 → Path 3 cancellation truly better
  Δlinear_pwr_ratio ≈ 0 → no linear-stage gain

For DT cases:
  Δlinear_pwr_ratio > 0 → MORE cancellation under Path 3.
    If small (≤ 1 dB): probably good (only echo reduced more)
    If large (> 3 dB): probably bad (NE also reduced — over-cancel)

If Path 3 has clean linear gains on FS AND minimal DT linear delta,
the AECMOS DT_static regression is purely RES over-suppression
(unmasking of existing behavior, not new). If Path 3 also has big
DT linear over-cancellation, the linear filter itself is broken on
those cases.

Output: results/v3_13_e2_s2_linear_compare/summary.md
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import soundfile as sf


def _signal_active_mask(x, win_size, hop_size, energy_pct=0.10):
    """Return boolean array over samples: True for frames where mic energy
    is above percentile threshold (signal-active). Suppress silence to
    avoid skewing the power ratio."""
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


def linear_pwr_ratio(mic_path, nores_path, sr=16000):
    mic, _ = sf.read(mic_path)
    nores, _ = sf.read(nores_path)
    n = min(len(mic), len(nores))
    mic = mic[:n].astype(np.float64)
    nores = nores[:n].astype(np.float64)
    # Signal-active mask using mic energy (~50ms window, 10ms hop)
    win = int(0.05 * sr); hop = int(0.01 * sr)
    mask = _signal_active_mask(mic, win, hop)
    if mask.sum() < 100:
        return float('nan'), float('nan'), float('nan')
    mic_pwr = float(np.mean(mic[mask] ** 2))
    nores_pwr = float(np.mean(nores[mask] ** 2))
    if nores_pwr < 1e-12 or mic_pwr < 1e-12:
        return float('nan'), 10*np.log10(mic_pwr+1e-12), 10*np.log10(nores_pwr+1e-12)
    ratio_db = 10.0 * np.log10(mic_pwr / nores_pwr)
    return ratio_db, 10*np.log10(mic_pwr), 10*np.log10(nores_pwr)


def classify(filename):
    """Return (bucket, talk_type, stem) from <stem>_<talk>_<variant>_ours_nores.wav."""
    if not filename.endswith('_ours_nores.wav'):
        return None
    base = filename[:-len('_ours_nores.wav')]
    if base.endswith('_doubletalk'):
        return ('DT_static', 'doubletalk', base[:-len('_doubletalk')])
    if base.endswith('_doubletalk_with_movement'):
        return ('DT_movement', 'doubletalk', base[:-len('_doubletalk_with_movement')])
    if base.endswith('_farend_singletalk'):
        return ('FS_static', 'farend_singletalk', base[:-len('_farend_singletalk')])
    if base.endswith('_farend_singletalk_with_movement'):
        return ('FS_movement', 'farend_singletalk', base[:-len('_farend_singletalk_with_movement')])
    if base.endswith('_nearend_singletalk'):
        return ('NE', 'nearend_singletalk', base[:-len('_nearend_singletalk')])
    return None


def find_mic_path(stem, scenario, dataset):
    """Subdir mapping: doubletalk → doubletalk/<stem>_doubletalk[_with_movement]_mic.wav
                     farend_singletalk → farend_singletalk/<stem>_farend_singletalk[_with_movement]_mic.wav
                     nearend_singletalk → nearend_singletalk/<stem>_nearend_singletalk_mic.wav"""
    # The full file name on disk is <stem>_<scenario>_mic.wav with the variant suffix preserved
    # in stem-naming during bench classify is awkward; just glob within scenario subdir
    sub = os.path.join(dataset, scenario)
    if not os.path.isdir(sub):
        return None
    # The file name is <stem>_<scenario>[_with_movement?]_mic.wav. We don't know movement
    # at this call-site so match by file's stem prefix.
    target = f'{stem}_{scenario}'
    cand = sorted(p for p in os.listdir(sub) if p.startswith(target) and p.endswith('_mic.wav'))
    if cand:
        return os.path.join(sub, cand[0])
    return None


def main():
    ap = argparse.ArgumentParser(description='E2.S2 linear-only comparison')
    ap.add_argument('--baseline-dir', default='results/v3_11_candidate')
    ap.add_argument('--path3-dir', default='results/v3_13_e2_s2_path3_800case')
    ap.add_argument('--dataset', default='wav/aec_challenge_blind')
    ap.add_argument('--out', default='results/v3_13_e2_s2_linear_compare')
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    base_dir = args.baseline_dir
    path3_dir = args.path3_dir

    # Iterate intersection of both dirs' *_ours_nores.wav files
    base_files = {f for f in os.listdir(base_dir) if f.endswith('_ours_nores.wav')}
    path3_files = {f for f in os.listdir(path3_dir) if f.endswith('_ours_nores.wav')}
    common = sorted(base_files & path3_files)
    print(f'[linear] {len(common)} cases in common', flush=True)

    rows = []
    for i, f in enumerate(common):
        c = classify(f)
        if not c:
            continue
        bucket, talk_type, stem = c
        mic = find_mic_path(stem, talk_type, args.dataset)
        if not mic:
            continue
        base_p = os.path.join(base_dir, f)
        path3_p = os.path.join(path3_dir, f)
        r_base, mp_b, np_b = linear_pwr_ratio(mic, base_p)
        r_path3, mp_p, np_p = linear_pwr_ratio(mic, path3_p)
        if np.isnan(r_base) or np.isnan(r_path3):
            continue
        rows.append({
            'stem': f[:-len('_ours_nores.wav')],
            'bucket': bucket,
            'base_lin_db': r_base,
            'path3_lin_db': r_path3,
            'delta_db': r_path3 - r_base,
        })
        if (i+1) % 100 == 0:
            print(f'  {i+1}/{len(common)}', flush=True)

    # Aggregate by bucket
    bucket_stats = {}
    for b in ('FS_static','FS_movement','DT_static','DT_movement','NE'):
        bvals = [r['delta_db'] for r in rows if r['bucket']==b]
        if not bvals:
            continue
        bucket_stats[b] = {
            'n': len(bvals),
            'mean_delta_db': float(np.mean(bvals)),
            'median_delta_db': float(np.median(bvals)),
            'gt_3db': sum(1 for d in bvals if d > 3),
            'gt_1db': sum(1 for d in bvals if d > 1),
            'lt_neg1db': sum(1 for d in bvals if d < -1),
            'lt_neg3db': sum(1 for d in bvals if d < -3),
        }

    out_md = [
        '# E2.S2 linear-only comparison: Path 3 vs baseline',
        '',
        '`linear_pwr_ratio_db = 10*log10(mean(mic²) / mean(nores²))` on signal-active frames.',
        '',
        'For FS: higher = better echo cancellation in linear stage.',
        'For DT: higher = more aggressive cancellation. If Path 3 large positive on DT → linear over-cancel (NE leaked into cancellation).',
        '',
        '## Bucket aggregate (Δ = Path 3 − baseline, dB)',
        '',
        '| Bucket | n | mean Δ | median Δ | n cases >+1 dB | >+3 dB | <−1 dB | <−3 dB |',
        '|---|---:|---:|---:|---:|---:|---:|---:|',
    ]
    for b, s in bucket_stats.items():
        out_md.append(f"| {b} | {s['n']} | {s['mean_delta_db']:+.3f} | {s['median_delta_db']:+.3f} | "
                      f"{s['gt_1db']} | {s['gt_3db']} | {s['lt_neg1db']} | {s['lt_neg3db']} |")
    out_md.append('')

    # Top regressing/improving per bucket
    for b in ('FS_static','FS_movement','DT_static','DT_movement','NE'):
        brows = [r for r in rows if r['bucket']==b]
        if not brows:
            continue
        out_md.append(f'## {b} — top 10 Path 3 gains (linear cancellation more aggressive)')
        out_md.append('')
        out_md.append('| Stem | base dB | path3 dB | Δ dB |')
        out_md.append('|---|---:|---:|---:|')
        for r in sorted(brows, key=lambda x: -x['delta_db'])[:10]:
            out_md.append(f"| `{r['stem']}` | {r['base_lin_db']:.2f} | {r['path3_lin_db']:.2f} | {r['delta_db']:+.2f} |")
        out_md.append('')

    # Cross-reference: 7 DT regressors from AECMOS
    dt_regressors = ['S22FCqKDWUyymN1YbpItIw_doubletalk',
                     'JtodX3Ug6Eu5TYu0HN5IOw_doubletalk',
                     'khqZY41lNEyIvMf2ZNJuVA_doubletalk',
                     '7GTxyTksSUqCnP5y0ILG4A_doubletalk',
                     'zzCIhneJ8UKTWZ48U0kRXw_doubletalk',
                     'V0JqgjlrB0Ke9y91r0rxNw_doubletalk',
                     'XTqo1aOXDEiqyWTFK99I5Q_doubletalk']
    out_md.append('## Cross-ref: 7 AECMOS DT_static deg regressors')
    out_md.append('')
    out_md.append('Goal: is the LINEAR stage cancelling more aggressively on these cases? '
                  'If yes → linear over-cancel (likely NE leakage). '
                  'If no → linear stage fine; RES is the over-suppressor.')
    out_md.append('')
    out_md.append('| Stem | base lin dB | path3 lin dB | Δ lin dB | AECMOS Δdeg |')
    out_md.append('|---|---:|---:|---:|---:|')
    deg_map = {'S22FCqKDWUyymN1YbpItIw_doubletalk': -2.045,
               'JtodX3Ug6Eu5TYu0HN5IOw_doubletalk': -1.913,
               'khqZY41lNEyIvMf2ZNJuVA_doubletalk': -1.585,
               '7GTxyTksSUqCnP5y0ILG4A_doubletalk': -1.504,
               'zzCIhneJ8UKTWZ48U0kRXw_doubletalk': -0.876,
               'V0JqgjlrB0Ke9y91r0rxNw_doubletalk': -0.758,
               'XTqo1aOXDEiqyWTFK99I5Q_doubletalk': -0.576}
    for stem in dt_regressors:
        match = [r for r in rows if r['stem'] == stem]
        if match:
            r = match[0]
            out_md.append(f"| `{stem}` | {r['base_lin_db']:.2f} | {r['path3_lin_db']:.2f} | "
                          f"{r['delta_db']:+.2f} | {deg_map[stem]:+.3f} |")
        else:
            out_md.append(f"| `{stem}` | (missing) | | | {deg_map[stem]:+.3f} |")
    out_md.append('')

    summary = os.path.join(args.out, 'summary.md')
    with open(summary, 'w') as f:
        f.write('\n'.join(out_md))

    with open(os.path.join(args.out, 'rows.json'), 'w') as f:
        json.dump(rows, f, indent=2)
    print(f'[linear] wrote {summary}', flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
