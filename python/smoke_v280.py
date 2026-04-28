#!/usr/bin/env python3
"""Smoke test compatible with v2.8.0 AecConfig (no shadow_dt_suppress_k).

Usage:
    python3 smoke_v280.py <dataset_dir> -o /tmp/smoke_v280 -n 50
"""
import os, sys, argparse
import numpy as np
import soundfile as sf

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from eval_aec_challenge import run_ours


def process_scenario(scene_dir, out_dir, n_cases, label, filter_length=512, preset='balanced'):
    mic_files = sorted([f for f in os.listdir(scene_dir) if f.endswith('_mic.wav')])[:n_cases]
    print(f"[{label}]  {len(mic_files)} cases")
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

        # No shadow_dt_suppress_k — not present in v2.8.0 AecConfig
        out = run_ours(mic, lpb, sr, fl=filter_length, enable_res=True,
                       preset=preset, is_movement='movement' in stem)
        sf.write(os.path.join(out_dir, f"{stem}_ours.wav"), out.astype(np.float32), sr)
        ok += 1
    print(f"  wrote {ok} files")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('dataset_dir')
    parser.add_argument('-o', '--output-dir', required=True)
    parser.add_argument('-n', '--n-cases', type=int, default=50)
    parser.add_argument('--filter-length', type=int, default=512)
    parser.add_argument('--preset', default='balanced')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    base = args.dataset_dir

    for sub, lbl in [('farend_singletalk', 'FS'), ('doubletalk', 'DT'), ('nearend_singletalk', 'NE')]:
        scene_dir = os.path.join(base, sub)
        if os.path.isdir(scene_dir):
            process_scenario(scene_dir, args.output_dir, args.n_cases, lbl,
                             filter_length=args.filter_length, preset=args.preset)


if __name__ == '__main__':
    main()
