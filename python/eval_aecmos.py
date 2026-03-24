#!/usr/bin/env python3
"""Evaluate AEC outputs using Microsoft AECMOS (speechmos).

Usage:
    python3 eval_aecmos.py <aec_challenge_dir>

Requires: pip install speechmos librosa onnxruntime
"""

import os
import sys
import numpy as np

METHODS = ['ours', 'ours_nores', 'aec3', 'speex']
METHOD_LABELS = ['Ours', 'NoRES', 'AEC3', 'Speex']

def find_fs_cases(base_dir, out_dir=None):
    """Find farend singletalk cases and map to output files."""
    fs_dir = os.path.join(base_dir, 'farend_singletalk')
    if out_dir is None:
        out_dir = os.path.join(base_dir, 'output')
    mic_files = sorted([f for f in os.listdir(fs_dir) if '_farend_singletalk_mic.wav' in f])

    cases = []
    for i, mic_f in enumerate(mic_files):
        prefix = mic_f.replace('_farend_singletalk_mic.wav', '')
        lpb_f = f"{prefix}_farend_singletalk_lpb.wav"
        cases.append({
            'idx': i,
            'type': 'fs',
            'mic': os.path.join(fs_dir, mic_f),
            'lpb': os.path.join(fs_dir, lpb_f),
            'ours': os.path.join(out_dir, f'fs_{i}_ours.wav'),
            'ours_nores': os.path.join(out_dir, f'fs_{i}_ours_nores.wav'),
            'aec3': os.path.join(out_dir, f'fs_{i}_aec3.wav'),
            'speex': os.path.join(out_dir, f'fs_{i}_speex.wav'),
        })
    return cases

def find_dt_cases(base_dir, out_dir=None):
    """Find doubletalk cases and map to output files."""
    dt_dir = os.path.join(base_dir, 'doubletalk')
    if out_dir is None:
        out_dir = os.path.join(base_dir, 'output')

    mic_files = sorted([f for f in os.listdir(dt_dir) if '_doubletalk_mic.wav' in f])

    cases = []
    for i, mic_f in enumerate(mic_files):
        prefix = mic_f.replace('_doubletalk_mic.wav', '')
        lpb_f = f"{prefix}_doubletalk_lpb.wav"
        cases.append({
            'idx': i,
            'type': 'dt',
            'mic': os.path.join(dt_dir, mic_f),
            'lpb': os.path.join(dt_dir, lpb_f),
            'ours': os.path.join(out_dir, f'dt_{i}_ours.wav'),
            'ours_nores': os.path.join(out_dir, f'dt_{i}_ours_nores.wav'),
            'aec3': os.path.join(out_dir, f'dt_{i}_aec3.wav'),
            'speex': os.path.join(out_dir, f'dt_{i}_speex.wav'),
        })
    return cases

def find_ne_cases(base_dir, out_dir=None):
    """Find nearend singletalk cases and map to output files."""
    ne_dir = os.path.join(base_dir, 'nearend_singletalk')
    if out_dir is None:
        out_dir = os.path.join(base_dir, 'output')
    if not os.path.isdir(ne_dir):
        return []

    mic_files = sorted([f for f in os.listdir(ne_dir) if '_nearend_singletalk_mic.wav' in f])

    cases = []
    for i, mic_f in enumerate(mic_files):
        prefix = mic_f.replace('_nearend_singletalk_mic.wav', '')
        lpb_f = f"{prefix}_nearend_singletalk_lpb.wav"
        cases.append({
            'idx': i,
            'type': 'ne',
            'mic': os.path.join(ne_dir, mic_f),
            'lpb': os.path.join(ne_dir, lpb_f),
            'ours': os.path.join(out_dir, f'ne_{i}_ours.wav'),
            'ours_nores': os.path.join(out_dir, f'ne_{i}_ours_nores.wav'),
            'aec3': os.path.join(out_dir, f'ne_{i}_aec3.wav'),
            'speex': os.path.join(out_dir, f'ne_{i}_speex.wav'),
        })
    return cases


def eval_aecmos(cases, talk_type=None):
    """Run AECMOS on all cases for all methods."""
    from speechmos.aecmos import run

    results = []
    for case in cases:
        label = case.get('fid', str(case.get('idx', '?')))
        row = {'label': label}

        for method in METHODS:
            enh_path = case[method]
            if not os.path.isfile(enh_path):
                row[method] = None
                continue

            sample = {
                'lpb': case['lpb'],
                'mic': case['mic'],
                'enh': enh_path,
            }
            try:
                res = run(sample, sr=16000, talk_type=talk_type)
                row[method] = res
            except Exception as e:
                print(f"  Error on {label}/{method}: {e}")
                row[method] = None

        results.append(row)
    return results

def print_results(title, results):
    """Print AECMOS results in a table."""
    print(f"\n{'='*80}")
    print(f"{title}")
    print(f"{'='*80}")

    # Detect available score keys from first valid result (only numeric values)
    score_keys = []
    for r in results:
        for method in METHODS:
            if r.get(method) and isinstance(r[method], dict):
                score_keys = [k for k, v in r[method].items()
                              if isinstance(v, (int, float))]
                break
        if score_keys:
            break

    if not score_keys:
        print("No valid results found.")
        return

    for key in score_keys:
        print(f"\n--- {key} ---")
        header = f"{'Case':>6}"
        for label in METHOD_LABELS:
            header += f"  {label:>8}"
        print(header)
        print("-" * len(header))

        vals = {m: [] for m in METHODS}
        for r in results:
            line = f"{r['label']:>6}"
            for method in METHODS:
                if r.get(method) and isinstance(r[method], dict) and key in r[method]:
                    v = r[method][key]
                    line += f"  {v:8.3f}"
                    vals[method].append(v)
                else:
                    line += f"  {'N/A':>8}"
            print(line)

        # Print mean
        line = f"{'MEAN':>6}"
        for method in METHODS:
            if vals[method]:
                line += f"  {np.mean(vals[method]):8.3f}"
            else:
                line += f"  {'N/A':>8}"
        print(line)

def main():
    import argparse
    parser = argparse.ArgumentParser(description='Evaluate AEC outputs using AECMOS')
    parser.add_argument('dataset_dir', help='aec_challenge/ directory')
    parser.add_argument('-o', '--output-dir', default=None, help='Output directory (default: <dataset_dir>/output)')
    parser.add_argument('--all-presets', action='store_true',
                        help='Evaluate all preset subdirs (balanced/aggressive/maximum)')
    args = parser.parse_args()

    base_dir = args.dataset_dir
    out_dir = args.output_dir or os.path.join(base_dir, 'output')

    if args.all_presets:
        for preset in ['balanced', 'aggressive', 'maximum']:
            preset_dir = os.path.join(out_dir, preset)
            if not os.path.isdir(preset_dir):
                print(f"Skipping {preset}: {preset_dir} not found")
                continue
            print(f"\n{'#'*60}")
            print(f"  PRESET: {preset.upper()}")
            print(f"{'#'*60}")
            fs_cases = find_fs_cases(base_dir, preset_dir)
            print(f"Found {len(fs_cases)} farend singletalk cases")
            fs_results = eval_aecmos(fs_cases, talk_type=None)
            print_results(f"FAREND SINGLETALK — AECMOS [{preset.upper()}]", fs_results)

            ne_cases = find_ne_cases(base_dir, preset_dir)
            if ne_cases:
                print(f"\nFound {len(ne_cases)} nearend singletalk cases")
                ne_results = eval_aecmos(ne_cases, talk_type=None)
                print_results(f"NEAREND SINGLETALK — AECMOS [{preset.upper()}]", ne_results)

            dt_cases = find_dt_cases(base_dir, preset_dir)
            print(f"\nFound {len(dt_cases)} doubletalk cases")
            dt_results = eval_aecmos(dt_cases, talk_type=None)
            print_results(f"DOUBLETALK — AECMOS [{preset.upper()}]", dt_results)
    else:
        fs_cases = find_fs_cases(base_dir, out_dir)
        print(f"Found {len(fs_cases)} farend singletalk cases")
        fs_results = eval_aecmos(fs_cases, talk_type=None)
        print_results("FAREND SINGLETALK — AECMOS", fs_results)

        ne_cases = find_ne_cases(base_dir, out_dir)
        if ne_cases:
            print(f"\nFound {len(ne_cases)} nearend singletalk cases")
            ne_results = eval_aecmos(ne_cases, talk_type=None)
            print_results("NEAREND SINGLETALK — AECMOS", ne_results)

        dt_cases = find_dt_cases(base_dir, out_dir)
        print(f"\nFound {len(dt_cases)} doubletalk cases")
        dt_results = eval_aecmos(dt_cases, talk_type=None)
        print_results("DOUBLETALK — AECMOS", dt_results)

if __name__ == '__main__':
    main()
