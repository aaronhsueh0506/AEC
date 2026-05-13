#!/usr/bin/env python3
"""E2.S4 — movement persistence classifier prototype (audit-only).

Builds on E2.S1 substrate: dumps `DelayEstimator._trace_rows` per case
and classifies each case by the time-evolution of the delay estimate
into one of four movement classes:

  STATIC                 — delay variance low; one stable cluster
  INTERMITTENT_STEP      — 1-2 big jumps but mostly clustered (e.g.
                           one room change, then settles)
  PERSISTENT_MOVEMENT    — continuous variance / many micro-jumps
                           (mic walking, continuous environment change)
  UNCERTAIN              — solid_fraction too low or PAR too noisy
                           (estimator can't trust its own outputs)

Per the v3.13 plan E2.S4: this is audit-only (no AEC behaviour change).
The goal is to characterise the FS_movement / DT_movement buckets and
identify cases for downstream conditional deployment (E2.S5).

Why we need this:
  - E2.S3 closed NEGATIVE (period scan on FS_movement flat). The
    online F-DelayTrack output does not currently feed back into ref
    alignment; the bench has static-within-session delays.
  - But the *naming* "with_movement" in FS_movement implies these
    cases were generated with simulated movement. The within-session
    delay trace should still show movement structure even if the
    bench pre-align already absorbed the average.
  - This audit answers: do FS_movement cases actually show movement
    in their online delay trace? And do FS_static cases stay STATIC?
    (Sanity check on the naming convention.)

Output per case: trace_rows JSON + classification.
Output overall: bucket histogram, list per class, summary.md.

Usage:
  python3 tools/research/e2_s4_movement_classifier.py \\
      --buckets FS_static FS_movement \\
      --out results/v3_13_e2_s4_classifier
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


# Same pre-align as eval_aec_challenge (max=250ms — production baseline).
# Path 3 (max=1024) is orthogonal: we want to characterise movement
# under the production setting first; conditional Path 3 gating is the
# E2.S5 outcome.
def _estimate_delay(mic, ref, sr, max_delay_ms=250.0):
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
    peak_val_phat = np.max(np.abs(xcorr_phat[:max_search + 1]))
    peak_idx_phat = int(np.argmax(np.abs(xcorr_phat[:max_search + 1])))
    rms = np.sqrt(np.mean(xcorr_phat[:max_search + 1] ** 2))
    confidence = peak_val_phat / (rms + 1e-10)
    if confidence < 5.0:
        xcorr_plain = np.fft.irfft(cross, n=fft_size)
        delay = int(np.argmax(np.abs(xcorr_plain[:max_search + 1])))
    else:
        delay = peak_idx_phat
    return delay


# Classification thresholds (v0 — tunable based on bucket distribution).
# Computed on the *post-GCC-pre-align* online delay trace, in samples
# at sr=16kHz (1 ms ≈ 16 samples).
THRESH = {
    'static_std_samp': 30.0,   # < 30 samp std → STATIC (≈ 2 ms)
    'persistent_std_samp': 100.0,   # ≥ 100 samp std → PERSISTENT
    'min_solid_frac': 0.20,    # < 20% solid → UNCERTAIN
    'min_mean_par': 3.0,       # < 3 mean PAR → UNCERTAIN
    'big_jump_samp': 100.0,    # |Δdelay| > 100 samp = "big jump"
    'cluster_window_samp': 50.0,    # within ±50 samp = same cluster
}


def classify_case(trace_rows, threshold=THRESH):
    """Return (class, metrics) from DelayEstimator trace_rows."""
    if not trace_rows:
        return 'UNCERTAIN', {'reason': 'no_trace'}

    # Filter to rows where the estimator produced a valid value.
    valid = [r for r in trace_rows if r.get('estimated_delay', -1) >= 0]
    n_total = len(trace_rows)
    n_valid = len(valid)
    solid_frac = (sum(1 for r in trace_rows if r.get('is_solid')) / n_total
                  if n_total > 0 else 0.0)
    pars = [r.get('last_par', 0.0) for r in valid if r.get('last_par', 0) > 0]
    mean_par = float(np.mean(pars)) if pars else 0.0

    metrics = {
        'n_trace_rows': n_total,
        'n_valid': n_valid,
        'solid_fraction': float(solid_frac),
        'mean_par': mean_par,
    }

    # UNCERTAIN gate: not enough reliable info.
    if (solid_frac < threshold['min_solid_frac']
            or mean_par < threshold['min_mean_par']
            or n_valid < 5):
        metrics['reason'] = 'low_reliability'
        return 'UNCERTAIN', metrics

    delays = np.array([r['estimated_delay'] for r in valid], dtype=np.float64)
    delay_std = float(np.std(delays))
    delay_range = float(np.max(delays) - np.min(delays))
    delay_mean = float(np.mean(delays))
    diffs = np.diff(delays)
    n_big_jumps = int(np.sum(np.abs(diffs) > threshold['big_jump_samp']))

    # Coarse clustering: count points within ±cluster_window of the median.
    median_d = float(np.median(delays))
    near_median = int(np.sum(np.abs(delays - median_d)
                              <= threshold['cluster_window_samp']))
    near_median_frac = float(near_median / n_valid)

    metrics.update({
        'delay_mean_samp': delay_mean,
        'delay_std_samp': delay_std,
        'delay_range_samp': delay_range,
        'n_big_jumps': n_big_jumps,
        'near_median_frac': near_median_frac,
    })

    # Classification (priority order):
    if delay_std < threshold['static_std_samp']:
        return 'STATIC', metrics
    if delay_std >= threshold['persistent_std_samp']:
        # High variance — distinguish persistent vs intermittent.
        # If ≥80% of samples cluster around median → intermittent (a
        # few outlier jumps); else persistent (variance spread out).
        if near_median_frac >= 0.80 and n_big_jumps <= 3:
            return 'INTERMITTENT_STEP', metrics
        return 'PERSISTENT_MOVEMENT', metrics
    # std in [30, 100): mild movement; n_big_jumps decides.
    if n_big_jumps == 0:
        return 'STATIC', metrics
    if n_big_jumps <= 2 and near_median_frac >= 0.80:
        return 'INTERMITTENT_STEP', metrics
    return 'PERSISTENT_MOVEMENT', metrics


def list_cases(dataset, buckets):
    """Return list of (bucket, stem, mic_path, lpb_path)."""
    cases = []
    for bucket in buckets:
        if bucket == 'FS_static':
            sub = 'farend_singletalk'
            filt = lambda f: f.endswith('_mic.wav') and '_with_movement_' not in f
        elif bucket == 'FS_movement':
            sub = 'farend_singletalk'
            filt = lambda f: f.endswith('_mic.wav') and '_with_movement_' in f
        elif bucket == 'DT_static':
            sub = 'doubletalk'
            filt = lambda f: f.endswith('_mic.wav') and '_with_movement_' not in f
        elif bucket == 'DT_movement':
            sub = 'doubletalk'
            filt = lambda f: f.endswith('_mic.wav') and '_with_movement_' in f
        elif bucket == 'NE':
            sub = 'nearend_singletalk'
            filt = lambda f: f.endswith('_mic.wav')
        else:
            print(f'[e2-s4] unknown bucket {bucket}', flush=True)
            continue
        sub_path = os.path.join(dataset, sub)
        if not os.path.isdir(sub_path):
            continue
        for f in sorted(os.listdir(sub_path)):
            if filt(f):
                stem = f[:-len('_mic.wav')]
                mic_p = os.path.join(sub_path, f)
                lpb_p = os.path.join(sub_path, f.replace('_mic.wav', '_lpb.wav'))
                if os.path.isfile(lpb_p):
                    cases.append((bucket, stem, mic_p, lpb_p))
    return cases


def audit_one(bucket, stem, mic_p, lpb_p):
    from aec import AEC, AecConfig, AecMode, AecPreset
    t0 = time.time()
    mic, sr = sf.read(mic_p)
    lpb, _ = sf.read(lpb_p)
    sr = int(sr)
    mic = mic.astype(np.float32)
    lpb = lpb.astype(np.float32)

    gcc_delay = _estimate_delay(mic, lpb, sr)
    n = min(len(mic), len(lpb))
    if 0 < gcc_delay < n:
        lpb_aligned = np.zeros(n, dtype=np.float32)
        lpb_aligned[gcc_delay:] = lpb[: n - gcc_delay]
    else:
        lpb_aligned = lpb[:n]
    mic = mic[:n]

    cfg = AecConfig.from_preset(
        AecPreset.BALANCED,
        sample_rate=sr, mode=AecMode.PBFDKF,
        filter_length=832,
        enable_dtd=False, enable_shadow=True, enable_res=True,
        enable_cng=True, use_kalman=True,
        enable_delay_est=True,
        delay_est_period_s=0.25,
        delay_est_init_s=0.2,
        trace_delay_est=True,
    )
    np.random.seed(0)
    aec = AEC(cfg)
    hop = aec.hop_size
    pos = 0
    while pos + hop <= n:
        aec.process(mic[pos: pos + hop], lpb_aligned[pos: pos + hop])
        pos += hop
    trace_rows = (aec.delay_est._trace_rows
                  if aec.delay_est is not None else [])
    cls, metrics = classify_case(trace_rows)
    return {
        'bucket': bucket,
        'stem': stem,
        'gcc_phat_initial_delay_samples': int(gcc_delay),
        'gcc_phat_initial_delay_ms': float(gcc_delay * 1000.0 / sr),
        'class': cls,
        'metrics': metrics,
        'duration_s': float(n / sr),
        'elapsed_s': time.time() - t0,
    }


def main():
    ap = argparse.ArgumentParser(description='E2.S4 movement persistence classifier')
    ap.add_argument('--dataset', default='wav/aec_challenge_blind')
    ap.add_argument('--buckets', nargs='+',
                    default=['FS_static', 'FS_movement'],
                    choices=['FS_static', 'FS_movement',
                             'DT_static', 'DT_movement', 'NE'])
    ap.add_argument('--out', required=True)
    ap.add_argument('--limit', type=int, default=0,
                    help='process only first N cases (for smoke testing)')
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    cases = list_cases(args.dataset, args.buckets)
    if args.limit > 0:
        cases = cases[:args.limit]
    print(f'[e2-s4] {len(cases)} cases across {args.buckets}', flush=True)

    results = []
    t_start = time.time()
    for i, (bucket, stem, mic_p, lpb_p) in enumerate(cases):
        try:
            r = audit_one(bucket, stem, mic_p, lpb_p)
        except Exception as e:
            print(f'[e2-s4] {bucket} {stem[:24]} ERR: {e}', flush=True)
            continue
        results.append(r)
        if (i + 1) % 25 == 0:
            print(f'[e2-s4] {i+1}/{len(cases)} '
                  f'last={r["elapsed_s"]:.1f}s elapsed={time.time()-t_start:.0f}s',
                  flush=True)
    print(f'[e2-s4] done in {time.time()-t_start:.0f}s', flush=True)

    # Aggregate by bucket × class
    classes = ('STATIC', 'INTERMITTENT_STEP', 'PERSISTENT_MOVEMENT', 'UNCERTAIN')
    bucket_class_count = {}
    for r in results:
        bucket_class_count.setdefault(r['bucket'], {c: 0 for c in classes})
        bucket_class_count[r['bucket']][r['class']] += 1

    md = [
        '# E2.S4 movement persistence classifier (audit-only)',
        '',
        'Per-case classification from `DelayEstimator._trace_rows`. '
        'Pre-alignment uses production max=250ms GCC-PHAT (same as bench).',
        '',
        'Thresholds (v0): std<30 samp = STATIC; std≥100 samp + spread = '
        'PERSISTENT; ≥80% within ±50 samp of median + few big jumps = '
        'INTERMITTENT_STEP; low solid_frac or low PAR = UNCERTAIN.',
        '',
        '## Bucket × class histogram',
        '',
        '| Bucket | n | STATIC | INTERMITTENT_STEP | PERSISTENT_MOVEMENT | UNCERTAIN |',
        '|---|---:|---:|---:|---:|---:|',
    ]
    for bucket in args.buckets:
        if bucket not in bucket_class_count:
            continue
        cnts = bucket_class_count[bucket]
        n = sum(cnts.values())
        md.append(f"| {bucket} | {n} | {cnts['STATIC']} | "
                  f"{cnts['INTERMITTENT_STEP']} | "
                  f"{cnts['PERSISTENT_MOVEMENT']} | {cnts['UNCERTAIN']} |")
    md.append('')

    # Per-class case lists (limited per bucket for readability)
    for bucket in args.buckets:
        for cls in classes:
            rows = [r for r in results
                    if r['bucket'] == bucket and r['class'] == cls]
            if not rows:
                continue
            rows.sort(key=lambda x: -x['metrics'].get('delay_std_samp', 0))
            md.append(f'## {bucket} / {cls} ({len(rows)})')
            md.append('')
            md.append('| Stem | GCC delay (samp) | std (samp) | range (samp) '
                      '| n_jumps | near_median% | solid% | mean PAR |')
            md.append('|---|---:|---:|---:|---:|---:|---:|---:|')
            for r in rows[:15]:
                m = r['metrics']
                md.append(
                    f"| `{r['stem'][:48]}` | "
                    f"{r['gcc_phat_initial_delay_samples']} | "
                    f"{m.get('delay_std_samp', 0):.1f} | "
                    f"{m.get('delay_range_samp', 0):.1f} | "
                    f"{m.get('n_big_jumps', 0)} | "
                    f"{m.get('near_median_frac', 0)*100:.0f}% | "
                    f"{m.get('solid_fraction', 0)*100:.0f}% | "
                    f"{m.get('mean_par', 0):.2f} |"
                )
            if len(rows) > 15:
                md.append(f'... ({len(rows) - 15} more)')
            md.append('')

    summary = os.path.join(args.out, 'summary.md')
    with open(summary, 'w') as f:
        f.write('\n'.join(md))
    with open(os.path.join(args.out, 'results.json'), 'w') as f:
        json.dump(results, f, indent=2)
    print(f'[e2-s4] wrote {summary}', flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
