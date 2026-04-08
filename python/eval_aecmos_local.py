#!/usr/bin/env python3
"""Evaluate AEC outputs using local AECMOS ONNX model.

Usage:
    python3 eval_aecmos_local.py <aec_challenge_dir>

Requires: pip install librosa onnxruntime torch numpy
ONNX model: /tmp/AEC-Challenge/AECMOS/AECMOS_local/Run_1663915512_Stage_0.onnx
"""

import os
import sys
import numpy as np

sys.path.insert(0, '/tmp/AEC-Challenge/AECMOS/AECMOS_local')
from aecmos import AECMOSEstimator

ONNX_PATH = '/tmp/AEC-Challenge/AECMOS/AECMOS_local/Run_1663915512_Stage_0.onnx'
METHODS = ['ours', 'ours_nores']
METHOD_LABELS = ['Ours', 'NoRES']


def find_cases(base_dir, scenario_dir_name, out_dir, talk_type):
    """Find cases and map to output files."""
    src_dir = os.path.join(base_dir, scenario_dir_name)
    if not os.path.isdir(src_dir):
        return []
    mic_files = sorted([f for f in os.listdir(src_dir) if f.endswith('_mic.wav')])
    cases = []
    for i, mic_f in enumerate(mic_files):
        stem = mic_f[:-len('_mic.wav')]
        cases.append({
            'idx': i,
            'stem': stem,
            'talk_type': talk_type,
            'mic': os.path.join(src_dir, mic_f),
            'lpb': os.path.join(src_dir, f"{stem}_lpb.wav"),
            'ours': os.path.join(out_dir, f'{stem}_ours.wav'),
            'ours_nores': os.path.join(out_dir, f'{stem}_ours_nores.wav'),
        })
    return cases


def eval_aecmos_local(cases, estimator):
    """Run local AECMOS on all cases."""
    results = []
    for case in cases:
        row = {'label': str(case['idx'])}
        for method in METHODS:
            enh_path = case[method]
            if not os.path.isfile(enh_path):
                row[method] = None
                continue
            try:
                lpb_sig, mic_sig, enh_sig = estimator.read_and_process_audio_files(
                    case['lpb'], case['mic'], enh_path)
                echo_mos, deg_mos = estimator.run(case['talk_type'], lpb_sig, mic_sig, enh_sig)
                row[method] = {'echo_mos': echo_mos, 'deg_mos': deg_mos}
            except Exception as e:
                print(f"  Error on {case['idx']}/{method}: {e}")
                row[method] = None
        results.append(row)
    return results


def print_results(title, results):
    """Print AECMOS results in a table."""
    print(f"\n{'='*80}")
    print(f"{title}")
    print(f"{'='*80}")

    for key in ['echo_mos', 'deg_mos']:
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

        line = f"{'MEAN':>6}"
        for method in METHODS:
            if vals[method]:
                line += f"  {np.mean(vals[method]):8.3f}"
            else:
                line += f"  {'N/A':>8}"
        print(line)


def main():
    import argparse
    parser = argparse.ArgumentParser(description='Evaluate AEC outputs using local AECMOS ONNX')
    parser.add_argument('dataset_dir', help='aec_challenge/ directory')
    parser.add_argument('-o', '--output-dir', default=None)
    parser.add_argument('--all-presets', action='store_true')
    parser.add_argument('--model', default=ONNX_PATH, help='Path to ONNX model')
    args = parser.parse_args()

    if not os.path.isfile(args.model):
        print(f"ONNX model not found: {args.model}")
        print("Clone: git clone https://github.com/microsoft/AEC-Challenge /tmp/AEC-Challenge")
        sys.exit(1)

    estimator = AECMOSEstimator(args.model)
    base_dir = args.dataset_dir
    out_dir = args.output_dir or os.path.join(base_dir, 'output')

    scenarios = [
        ('farend_singletalk', 'st', 'FAREND SINGLETALK'),
        ('nearend_singletalk', 'nst', 'NEAREND SINGLETALK'),
        ('doubletalk', 'dt', 'DOUBLETALK'),
    ]

    if args.all_presets:
        for preset in ['mild', 'balanced', 'aggressive', 'maximum']:
            preset_dir = os.path.join(out_dir, preset)
            if not os.path.isdir(preset_dir):
                print(f"Skipping {preset}: {preset_dir} not found")
                continue
            print(f"\n{'#'*60}")
            print(f"  PRESET: {preset.upper()}")
            print(f"{'#'*60}")
            for dir_name, talk_type, label in scenarios:
                cases = find_cases(base_dir, dir_name, preset_dir, talk_type)
                if not cases:
                    continue
                print(f"\nFound {len(cases)} {label.lower()} cases")
                results = eval_aecmos_local(cases, estimator)
                print_results(f"{label} — AECMOS [{preset.upper()}]", results)
    else:
        for dir_name, talk_type, label in scenarios:
            cases = find_cases(base_dir, dir_name, out_dir, talk_type)
            if not cases:
                continue
            print(f"Found {len(cases)} {label.lower()} cases")
            results = eval_aecmos_local(cases, estimator)
            print_results(f"{label} — AECMOS", results)


if __name__ == '__main__':
    main()
