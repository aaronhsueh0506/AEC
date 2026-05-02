#!/usr/bin/env python3
"""Export saturation_{ref,mic}.npz to flat .bin for parity_saturation.c.

Layout (LE):
  header:     i32 n_frames, i32 hop, f64 threshold
  per frame:  f64 level_in, f64 level_out, f32 input[hop]
"""
import sys, struct, argparse
from pathlib import Path
import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('dump_dir')
    ap.add_argument('out_bin')
    ap.add_argument('--side', choices=['ref', 'mic'], default='ref')
    ap.add_argument('--threshold', type=float, default=0.95)
    args = ap.parse_args()

    dump_dir = Path(args.dump_dir)
    meta = np.load(str(dump_dir / 'meta.npz'))
    hop = int(meta['hop'])

    d = np.load(str(dump_dir / f'sat_{args.side}.npz'))
    n_frames = d['input'].shape[0]
    assert d['input'].shape == (n_frames, hop)

    with open(args.out_bin, 'wb') as f:
        f.write(struct.pack('<iid', n_frames, hop, args.threshold))
        for i in range(n_frames):
            f.write(struct.pack('<dd',
                                float(d['level_in'][i]),
                                float(d['level_out'][i])))
            f.write(d['input'][i].astype('<f4').tobytes())

    print(f'Wrote {n_frames} frames × hop={hop} → {args.out_bin}')


if __name__ == '__main__':
    main()
