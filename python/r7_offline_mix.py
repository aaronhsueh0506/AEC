#!/usr/bin/env python3
"""R7.2 offline mix upper bound — fixed-ratio blend ours.wav + ours_nores.wav.

mixed = (1-r) * ours + r * nores

Source eval_output is R7.1a's (functionally baseline since R7.1a delta vs
baseline_v381_seeded was within +/-0.001 noise floor across all buckets).

Usage:
    python3 python/r7_offline_mix.py SRC_EVAL_DIR DST_DIR --ratio 0.10
"""
import argparse
import glob
import os

import numpy as np
import soundfile as sf


def _mix_one(args_tuple):
    of, dst_dir, ratio = args_tuple
    nf = of.replace('_ours.wav', '_ours_nores.wav')
    if not os.path.isfile(nf):
        return of, 'missing nores'
    ours, sr_o = sf.read(of)
    nores, sr_n = sf.read(nf)
    if sr_o != sr_n:
        return of, f'sr mismatch {sr_o}!={sr_n}'
    n = min(len(ours), len(nores))
    mixed = (1.0 - ratio) * ours[:n] + ratio * nores[:n]
    mixed = np.clip(mixed, -1.0, 1.0).astype(np.float32)
    out_o = os.path.join(dst_dir, os.path.basename(of))
    out_n = os.path.join(dst_dir, os.path.basename(nf))
    sf.write(out_o, mixed, sr_o)
    sf.write(out_n, nores[:n].astype(np.float32), sr_n)
    return of, 'ok'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('src_dir', help='Source eval_output with *_ours.wav and *_ours_nores.wav')
    ap.add_argument('dst_dir', help='Destination eval_output')
    ap.add_argument('--ratio', type=float, required=True, help='nores mix ratio in [0,1]')
    ap.add_argument('--jobs', '-j', type=int, default=4)
    args = ap.parse_args()

    os.makedirs(args.dst_dir, exist_ok=True)
    ours_files = sorted(glob.glob(os.path.join(args.src_dir, '*_ours.wav')))
    print(f'src={args.src_dir} dst={args.dst_dir} ratio={args.ratio} cases={len(ours_files)}')

    tasks = [(of, args.dst_dir, args.ratio) for of in ours_files]

    if args.jobs > 1:
        from concurrent.futures import ProcessPoolExecutor, as_completed
        with ProcessPoolExecutor(max_workers=args.jobs) as ex:
            futs = [ex.submit(_mix_one, t) for t in tasks]
            done = 0
            for f in as_completed(futs):
                _, status = f.result()
                done += 1
                if status != 'ok':
                    print(f'  {status}')
                if done % 100 == 0:
                    print(f'  {done}/{len(ours_files)}', flush=True)
    else:
        for i, t in enumerate(tasks):
            _mix_one(t)
            if (i + 1) % 100 == 0:
                print(f'  {i+1}/{len(ours_files)}', flush=True)
    print('done')


if __name__ == '__main__':
    main()
