#!/usr/bin/env python3
"""Evaluate AEC outputs using Microsoft AECMOS (local ONNX model from GitHub).

Uses AEC-Challenge repo: https://github.com/microsoft/AEC-Challenge
Model: Run_1663915512_Stage_0.onnx (16kHz, scenario-based)

Usage:
    python3 eval_aecmos_local.py <aec_challenge_dir>
    python3 eval_aecmos_local.py <aec_challenge_dir> --all-presets

Requires: pip install numpy onnxruntime librosa torch soundfile
          git clone https://github.com/microsoft/AEC-Challenge /tmp/AEC-Challenge
"""

import os
import sys
import re
import numpy as np
import argparse

# Add AECMOS local path
AECMOS_DIR = '/tmp/AEC-Challenge/AECMOS/AECMOS_local'
AECMOS_MODEL = os.path.join(AECMOS_DIR, 'Run_1663915512_Stage_0.onnx')

sys.path.insert(0, AECMOS_DIR)
from aecmos import AECMOSEstimator

METHODS = ['ours', 'ours_nores', 'speex', 'aec3', 'aec3_linear', 'old_aec']
METHOD_LABELS = ['Ours', 'Ours(Linear)', 'Speex', 'AEC3', 'AEC3(Linear)', 'OldAEC']

# Map dataset talk type to AECMOS talk type
TALK_TYPE_MAP = {
    'fs': 'st',   # farend singletalk
    'dt': 'dt',   # doubletalk
    'ne': 'nst',  # nearend singletalk
}


def find_cases(base_dir, subdir_name, out_dir=None):
    """Find test cases in a subdirectory.

    Handles blind test naming: {id}_{scenario}_mic.wav
    where scenario can be e.g. farend_singletalk, farend_singletalk_with_movement,
    doubletalk, doubletalk_with_movement, nearend_singletalk, etc.
    """
    sub_dir = os.path.join(base_dir, subdir_name)
    if not os.path.isdir(sub_dir):
        return []
    if out_dir is None:
        out_dir = os.path.join(base_dir, 'output')

    mic_files = sorted([f for f in os.listdir(sub_dir) if f.endswith('_mic.wav')])

    cases = []
    for i, mic_f in enumerate(mic_files):
        # Strip _mic.wav to get the full prefix (includes scenario name)
        stem = mic_f[:-len('_mic.wav')]
        lpb_f = f"{stem}_lpb.wav"
        lpb_path = os.path.join(sub_dir, lpb_f)
        if not os.path.isfile(lpb_path):
            continue

        case = {
            'idx': i,
            'stem': stem,
            'mic': os.path.join(sub_dir, mic_f),
            'lpb': lpb_path,
        }
        for method in METHODS:
            case[method] = os.path.join(out_dir, f'{stem}_{method}.wav')
        cases.append(case)
    return cases


def eval_aecmos(cases, talk_type, estimator):
    """Run AECMOS on all cases for all methods."""
    aecmos_talk = TALK_TYPE_MAP.get(talk_type, talk_type)

    # Also score the mic (unprocessed) as baseline
    all_methods = ['mic'] + METHODS

    results = []
    n = len(cases)
    for ci, case in enumerate(cases):
        label = str(case.get('idx', '?'))
        row = {'label': label}

        for method in all_methods:
            if method == 'mic':
                enh_path = case['mic']
            else:
                enh_path = case[method]

            if not os.path.isfile(enh_path):
                row[method] = None
                continue

            try:
                lpb_sig, mic_sig, enh_sig = estimator.read_and_process_audio_files(
                    case['lpb'], case['mic'], enh_path)
                echo_mos, deg_mos = estimator.run(aecmos_talk, lpb_sig, mic_sig, enh_sig)
                row[method] = {'echo_mos': echo_mos, 'deg_mos': deg_mos}
            except Exception as e:
                print(f"  Error on {label}/{method}: {e}")
                row[method] = None

        if (ci + 1) % 50 == 0 or ci == n - 1:
            print(f"  [{ci+1}/{n}]", flush=True)

        results.append(row)
    return results


def print_results(title, results):
    """Print AECMOS results in a table."""
    all_methods = ['mic'] + METHODS
    all_labels = ['Mic'] + METHOD_LABELS

    # Find active methods
    active = []
    active_labels = []
    for method, label in zip(all_methods, all_labels):
        if any(r.get(method) is not None for r in results):
            active.append(method)
            active_labels.append(label)

    if not active:
        print(f"\n{title}: No valid results found.")
        return

    print(f"\n{'='*80}")
    print(f"{title} ({len(results)} cases)")
    print(f"{'='*80}")

    for key, desc in [('echo_mos', 'Echo MOS (higher=less echo, 1-5)'),
                      ('deg_mos', 'Degradation MOS (higher=less degradation, 1-5)')]:
        print(f"\n--- {desc} ---")
        header = f"{'Case':>6}"
        for label in active_labels:
            header += f"  {label:>15}"
        print(header)
        print("-" * len(header))

        vals = {m: [] for m in active}
        for r in results:
            line = f"{r['label']:>6}"
            for method in active:
                if r.get(method) and isinstance(r[method], dict) and key in r[method]:
                    v = r[method][key]
                    line += f"  {v:15.9f}"
                    vals[method].append(v)
                else:
                    line += f"  {'N/A':>15}"
            print(line)

        # Mean
        print("-" * len(header))
        line = f"{'MEAN':>6}"
        for method in active:
            if vals[method]:
                line += f"  {np.mean(vals[method]):15.9f}"
            else:
                line += f"  {'N/A':>15}"
        print(line)


def main():
    parser = argparse.ArgumentParser(description='Evaluate AEC outputs using AECMOS (local)')
    parser.add_argument('dataset_dir', help='aec_challenge/ directory')
    parser.add_argument('-o', '--output-dir', default=None,
                        help='Output directory (default: <dataset_dir>/output)')
    parser.add_argument('--model', default=AECMOS_MODEL,
                        help=f'ONNX model path (default: {AECMOS_MODEL})')
    args = parser.parse_args()

    if not os.path.isfile(args.model):
        print(f"Error: AECMOS model not found at {args.model}")
        print("Clone the repo: git clone https://github.com/microsoft/AEC-Challenge /tmp/AEC-Challenge")
        sys.exit(1)

    estimator = AECMOSEstimator(args.model)
    base_dir = args.dataset_dir
    out_dir = args.output_dir or os.path.join(base_dir, 'output')

    scenarios = [
        ('farend_singletalk', 'fs'),
        ('nearend_singletalk', 'ne'),
        ('doubletalk', 'dt'),
    ]

    for subdir, talk_type in scenarios:
        cases = find_cases(base_dir, subdir, out_dir)
        if not cases:
            print(f"No cases found for {subdir}")
            continue
        print(f"\nScoring {len(cases)} {subdir} cases with AECMOS...")
        results = eval_aecmos(cases, talk_type, estimator)
        print_results(f"{subdir.upper()} — AECMOS", results)


if __name__ == '__main__':
    main()
