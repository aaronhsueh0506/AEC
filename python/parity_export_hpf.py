#!/usr/bin/env python3
"""
Convert dump_module_state.py's hpf_{mic,ref}.npz into a flat .bin readable
by c_impl/test/parity/parity_hpf.c.

Layout (little-endian, struct.pack):
  header:     i32 n_frames, i32 hop, i32 sample_rate, f64 cutoff_hz
  per frame:  f64 z1_in, z2_in, z1_out, z2_out
              f32 input[hop], f32 expected_output[hop]
"""
import sys
import struct
import argparse
from pathlib import Path
import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('dump_dir')
    ap.add_argument('out_bin')
    ap.add_argument('--side', choices=['mic', 'ref'], default='mic')
    ap.add_argument('--cutoff-hz', type=float, default=80.0,
                    help="HPF cutoff (must match what AEC config uses)")
    args = ap.parse_args()

    dump_dir = Path(args.dump_dir)
    meta = np.load(str(dump_dir / 'meta.npz'))
    hop = int(meta['hop'])
    sr  = int(meta['sample_rate'])

    side_path = dump_dir / f'hpf_{args.side}.npz'
    d = np.load(str(side_path))
    n_frames = d['input'].shape[0]
    assert d['input'].shape == (n_frames, hop), d['input'].shape
    assert d['output'].shape == (n_frames, hop)

    with open(args.out_bin, 'wb') as f:
        f.write(struct.pack('<iiid', n_frames, hop, sr, args.cutoff_hz))
        for i in range(n_frames):
            f.write(struct.pack('<dddd',
                                float(d['z1_in'][i]),  float(d['z2_in'][i]),
                                float(d['z1_out'][i]), float(d['z2_out'][i])))
            f.write(d['input'][i].astype('<f4').tobytes())
            f.write(d['output'][i].astype('<f4').tobytes())

    print(f'Wrote {n_frames} frames × hop={hop} → {args.out_bin}')


if __name__ == '__main__':
    main()
