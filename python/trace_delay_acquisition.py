#!/usr/bin/env python3
"""P3c Phase B — 800-case delay-acquisition latency summary.

For every (mic, lpb) pair under wav/aec_challenge_blind, run the AEC
through frame-by-frame and record one summary row per case:

    stem, scenario, total_s, time_to_first_solid_s, delay_at_first_solid_ms,
    max_par_observed, frames_processed, ever_solid (bool)

No waveforms written. No behaviour change vs v3.10.4 (only
trace_delay_est=True flips on the per-call _trace_rows capture, which is
zero-cost when off).

Output: a single CSV at the path given on the CLI.

Usage:
    python3 python/trace_delay_acquisition.py wav/aec_challenge_blind \\
        /tmp/p3c_800case_summary.csv --parallel
"""
import argparse
import csv
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import soundfile as sf

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)

from aec import AEC, AecConfig, AecPreset, AecMode  # noqa: E402

PRESET_MAP = {
    'mild': AecPreset.MILD,
    'soft': AecPreset.SOFT,
    'balanced': AecPreset.BALANCED,
    'aggressive': AecPreset.AGGRESSIVE,
    'maximum': AecPreset.MAXIMUM,
}


def classify_scenario(stem):
    if 'farend_singletalk_with_movement' in stem:
        return 'FS_movement'
    if 'farend_singletalk' in stem:
        return 'FS_static'
    if 'doubletalk_with_movement' in stem:
        return 'DT_movement'
    if 'doubletalk' in stem:
        return 'DT_static'
    if 'nearend_singletalk' in stem:
        return 'NE'
    return 'unknown'


def run_case(args_tuple):
    mic_path, lpb_path, stem = args_tuple
    fp_on = os.environ.get('AEC_DELAY_FAST_PATH', '0').lower() not in ('0', 'false', 'off', 'no')
    fp_par = float(os.environ.get('AEC_DELAY_FAST_PAR', '40.0'))
    cfg = AecConfig.from_preset(
        PRESET_MAP['balanced'],
        sample_rate=16000, filter_length=832, mode=AecMode.PBFDKF,
        enable_res=True, enable_cng=True, enable_shadow=True,
        trace_delay_est=True,
        delay_fast_path_enabled=fp_on,
        delay_fast_par_threshold=fp_par,
    )
    aec = AEC(cfg)
    mic, _ = sf.read(mic_path)
    lpb, _ = sf.read(lpb_path)
    mic = np.asarray(mic, dtype=np.float32)
    lpb = np.asarray(lpb, dtype=np.float32)
    n = min(len(mic), len(lpb))
    mic = mic[:n]
    lpb = lpb[:n]
    hop = aec.hop_size
    pos = 0
    while pos + hop <= n:
        aec.process(mic[pos:pos + hop], lpb[pos:pos + hop])
        pos += hop

    rows = aec.delay_est._trace_rows or []
    total_s = pos / 16000.0

    first_solid_idx = None
    delay_at_first_solid = -1
    max_par = 0.0
    for i, r in enumerate(rows):
        if r['last_par'] > max_par:
            max_par = float(r['last_par'])
        if first_solid_idx is None and r['is_solid']:
            first_solid_idx = i
            delay_at_first_solid = int(r['estimated_delay'])

    if first_solid_idx is not None:
        first_solid_s = float(rows[first_solid_idx]['time_s'])
        ever_solid = True
    else:
        first_solid_s = -1.0
        ever_solid = False

    return {
        'stem': stem,
        'scenario': classify_scenario(stem),
        'total_s': round(total_s, 3),
        'time_to_first_solid_s': round(first_solid_s, 3),
        'delay_at_first_solid_ms': round(delay_at_first_solid * 1000.0 / 16000.0, 1)
            if delay_at_first_solid >= 0 else -1,
        'max_par_observed': round(max_par, 3),
        'frames_processed': len(rows),
        'ever_solid': ever_solid,
    }


def find_pairs(dataset_root):
    pairs = []
    for sub in ('farend_singletalk', 'doubletalk', 'nearend_singletalk'):
        d = os.path.join(dataset_root, sub)
        if not os.path.isdir(d):
            continue
        for f in sorted(os.listdir(d)):
            if not f.endswith('_mic.wav'):
                continue
            stem = f[:-len('_mic.wav')]
            mic_path = os.path.join(d, f)
            lpb_path = os.path.join(d, stem + '_lpb.wav')
            if not os.path.exists(lpb_path):
                continue
            pairs.append((mic_path, lpb_path, stem))
    return pairs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('dataset_dir')
    ap.add_argument('output_csv')
    ap.add_argument('--parallel', action='store_true')
    args = ap.parse_args()

    pairs = find_pairs(args.dataset_dir)
    print(f'Found {len(pairs)} cases under {args.dataset_dir}', file=sys.stderr)

    results = []
    if args.parallel:
        # Cap workers to 3 so total python process count (parent + workers)
        # stays within the project-wide budget of 4.
        with ProcessPoolExecutor(max_workers=3) as ex:
            futs = {ex.submit(run_case, p): p for p in pairs}
            for i, fut in enumerate(as_completed(futs), 1):
                results.append(fut.result())
                if i % 50 == 0:
                    print(f'  {i}/{len(pairs)}', file=sys.stderr)
    else:
        for i, p in enumerate(pairs, 1):
            results.append(run_case(p))
            if i % 50 == 0:
                print(f'  {i}/{len(pairs)}', file=sys.stderr)

    results.sort(key=lambda r: r['stem'])
    with open(args.output_csv, 'w', newline='') as fp:
        w = csv.DictWriter(fp, fieldnames=list(results[0].keys()))
        w.writeheader()
        w.writerows(results)
    print(f'wrote {args.output_csv} ({len(results)} rows)', file=sys.stderr)


if __name__ == '__main__':
    main()
