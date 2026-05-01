#!/usr/bin/env python3
"""Batch-run C AEC on AEC Challenge blind set and produce output wavs
compatible with eval_aecmos_local.py (same file naming convention)."""

import os
import sys
import subprocess
import multiprocessing as mp
from pathlib import Path

C_BIN = str(Path(__file__).resolve().parents[1] / 'c_impl/bin/aec_wav')


def process_one(args):
    mic, lpb, out_main, out_nores, preset = args
    try:
        # Match Python eval defaults: delay_est ON (BALANCED preset has
        # enable_delay_est=True). Without it, filter never gets aligned and
        # FS bucket scores collapse.
        subprocess.run([C_BIN, mic, lpb, out_main, '--preset', preset],
                       capture_output=True, check=True)
        subprocess.run([C_BIN, mic, lpb, out_nores, '--preset', preset, '--no-res'],
                       capture_output=True, check=True)
        return True, mic
    except subprocess.CalledProcessError as e:
        return False, f"{mic}: {e.stderr.decode()[:200]}"


def main():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument('dataset_dir')
    p.add_argument('-o', '--output-dir', required=True)
    p.add_argument('--preset', default='balanced')
    p.add_argument('--jobs', type=int, default=mp.cpu_count())
    args = p.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    tasks = []
    for scenario, suffix in [
        ('farend_singletalk', 'farend_singletalk'),
        ('nearend_singletalk', 'nearend_singletalk'),
        ('doubletalk', 'doubletalk'),
    ]:
        src = os.path.join(args.dataset_dir, scenario)
        if not os.path.isdir(src):
            continue
        for f in sorted(os.listdir(src)):
            if not f.endswith('_mic.wav'):
                continue
            base = f.replace('_mic.wav', '')
            mic = os.path.join(src, f)
            lpb = os.path.join(src, base + '_lpb.wav')
            if not os.path.isfile(lpb):
                continue
            # Handle "with_movement" in filename — keep as part of base for scenario label
            out_main = os.path.join(args.output_dir, f"{base}_ours.wav")
            out_nores = os.path.join(args.output_dir, f"{base}_ours_nores.wav")
            tasks.append((mic, lpb, out_main, out_nores, args.preset))

    print(f"Running {len(tasks)} cases on {args.jobs} workers...")
    ok = fail = 0
    with mp.Pool(args.jobs) as pool:
        for i, (status, info) in enumerate(pool.imap_unordered(process_one, tasks), 1):
            if status:
                ok += 1
            else:
                fail += 1
                print(f"  FAIL: {info}")
            if i % 50 == 0 or i == len(tasks):
                print(f"  {i}/{len(tasks)} ({ok} ok, {fail} fail)")
    print(f"\nDone: {ok} ok, {fail} fail. Output in {args.output_dir}")


if __name__ == '__main__':
    main()
