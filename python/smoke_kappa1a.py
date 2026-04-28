#!/usr/bin/env python3
"""Smoke test: κ-1-A shadow_dt_suppress_k sweep (50-case).

Outputs _ours.wav files to <out_dir>/ in the same flat structure
that eval_aecmos.py expects (out_dir/{stem}_ours.wav).

Usage:
    python3 smoke_kappa1a.py <dataset_dir> --k 0.0 -o /tmp/smoke_k0
    python3 smoke_kappa1a.py <dataset_dir> --k 0.7 -o /tmp/smoke_k07
    # then score with:
    python3 eval_aecmos.py <dataset_dir> -o /tmp/smoke_k0
"""
import os, sys, argparse
import numpy as np
import soundfile as sf

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from eval_aec_challenge import run_ours


def process_scenario(scene_dir, out_dir, k_val, n_cases, label):
    mic_files = sorted([f for f in os.listdir(scene_dir) if f.endswith('_mic.wav')])[:n_cases]
    print(f"[{label}] K={k_val}  {len(mic_files)} cases")
    ok = 0
    for mic_f in mic_files:
        stem = mic_f[:-len('_mic.wav')]
        mic_path = os.path.join(scene_dir, mic_f)
        lpb_path = os.path.join(scene_dir, f"{stem}_lpb.wav")
        if not os.path.isfile(lpb_path):
            continue
        mic, sr = sf.read(mic_path, dtype='float32')
        lpb, _  = sf.read(lpb_path, dtype='float32')
        if mic.ndim > 1: mic = mic[:, 0]
        if lpb.ndim > 1: lpb = lpb[:, 0]

        out = run_ours(mic, lpb, sr, fl=2048, enable_res=True,
                       is_movement='movement' in stem,
                       shadow_dt_suppress_k=k_val)
        sf.write(os.path.join(out_dir, f"{stem}_ours.wav"), out.astype(np.float32), sr)
        ok += 1
    print(f"  wrote {ok} files")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('dataset_dir')
    parser.add_argument('--k', type=float, default=0.0)
    parser.add_argument('-o', '--output-dir', required=True)
    parser.add_argument('-n', '--n-cases', type=int, default=50)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    base = args.dataset_dir

    for sub, lbl in [('farend_singletalk', 'FS'), ('doubletalk', 'DT'), ('nearend_singletalk', 'NE')]:
        scene_dir = os.path.join(base, sub)
        if os.path.isdir(scene_dir):
            process_scenario(scene_dir, args.output_dir, args.k, args.n_cases, lbl)


if __name__ == '__main__':
    main()
