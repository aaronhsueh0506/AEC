#!/usr/bin/env python3
"""Export delay_est.npz to flat .bin for parity_delay.c.

Layout (LE):
  header:  i32 n_frames, i32 hop, i32 sample_rate,
           f64 max_delay_ms, f64 init_seconds, f64 period_seconds
  per frame: f32 mic[hop], f32 ref[hop],
             i32 estimated_delay, f64 par, i32 n_updates
"""
import struct, argparse
from pathlib import Path
import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('dump_dir')
    ap.add_argument('out_bin')
    ap.add_argument('--max-delay-ms', type=float, default=250.0)
    ap.add_argument('--init-seconds', type=float, default=0.5)
    ap.add_argument('--period-seconds', type=float, default=2.0)
    args = ap.parse_args()

    dump_dir = Path(args.dump_dir)
    meta = np.load(str(dump_dir / 'meta.npz'))
    hop = int(meta['hop']); sr = int(meta['sample_rate'])

    d = np.load(str(dump_dir / 'delay_est.npz'))
    n_frames = d['mic'].shape[0]
    assert d['mic'].shape == (n_frames, hop)

    with open(args.out_bin, 'wb') as f:
        f.write(struct.pack('<iiiddd',
                            n_frames, hop, sr,
                            args.max_delay_ms, args.init_seconds,
                            args.period_seconds))
        for i in range(n_frames):
            f.write(d['mic'][i].astype('<f4').tobytes())
            f.write(d['ref'][i].astype('<f4').tobytes())
            f.write(struct.pack('<idi',
                                int(d['estimated_delay'][i]),
                                float(d['par'][i]),
                                int(d['n_updates'][i])))
    print(f'Wrote {n_frames} frames → {args.out_bin}')


if __name__ == '__main__':
    main()
