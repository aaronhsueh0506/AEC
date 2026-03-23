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

def find_fs_cases(base_dir):
    """Find farend singletalk cases and map to output files."""
    fs_dir = os.path.join(base_dir, 'farend_singletalk')
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

def find_dt_cases(base_dir):
    """Find doubletalk cases and map to output files."""
    dt_dir = os.path.join(base_dir, 'doubletalk')
    out_dir = os.path.join(base_dir, 'output')

    fids = sorted(set(
        f.replace('nearend_mic_fileid_', '').replace('.wav', '')
        for f in os.listdir(dt_dir) if f.startswith('nearend_mic_fileid_')
    ), key=int)

    cases = []
    for fid in fids:
        cases.append({
            'fid': fid,
            'type': 'dt',
            'mic': os.path.join(dt_dir, f'nearend_mic_fileid_{fid}.wav'),
            'lpb': os.path.join(dt_dir, f'farend_speech_fileid_{fid}.wav'),
            'ours': os.path.join(out_dir, f'dt_{fid}_ours.wav'),
            'ours_nores': os.path.join(out_dir, f'dt_{fid}_ours_nores.wav'),
            'aec3': os.path.join(out_dir, f'dt_{fid}_aec3.wav'),
            'speex': os.path.join(out_dir, f'dt_{fid}_speex.wav'),
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
    if len(sys.argv) < 2:
        print(f"Usage: {sys.argv[0]} <aec_challenge_dir>")
        sys.exit(1)

    base_dir = sys.argv[1]

    # Far-end singletalk
    fs_cases = find_fs_cases(base_dir)
    print(f"Found {len(fs_cases)} farend singletalk cases")
    fs_results = eval_aecmos(fs_cases, talk_type=None)
    print_results("FAREND SINGLETALK — AECMOS", fs_results)

    # Doubletalk
    dt_cases = find_dt_cases(base_dir)
    print(f"\nFound {len(dt_cases)} doubletalk cases")
    dt_results = eval_aecmos(dt_cases, talk_type=None)
    print_results("DOUBLETALK — AECMOS", dt_results)

if __name__ == '__main__':
    main()
