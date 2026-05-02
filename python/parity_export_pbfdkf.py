#!/usr/bin/env python3
"""Export pbfdkf_main.npz to flat .bin for parity_pbfdkf.c.

Layout (LE):
  header:  i32 hop, i32 n_freqs, i32 n_partitions, i32 n_frames,
           f32 mu, f32 delta
  per frame:
    f32 near[hop], far[hop]
    i32 mu_scale_is_array (0=scalar, 1=array[n_freqs])
    f32 mu_scale_scalar     (if scalar)
        OR f32 mu_scale_arr[n_freqs] (if array)
    --- expected outputs after this frame ---
    f32 output[hop]
    f32 error_spec[2*n_freqs]    (re, im interleaved)
    f32 echo_spec[2*n_freqs]
    f32 P[n_partitions*n_freqs]  (real)
    f32 R[n_freqs]
"""
import sys, struct, argparse
from pathlib import Path
import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('dump_dir')
    ap.add_argument('out_bin')
    ap.add_argument('--side', choices=['main', 'shadow'], default='main')
    ap.add_argument('--max-frames', type=int, default=200)
    args = ap.parse_args()

    dump_dir = Path(args.dump_dir)
    meta = np.load(str(dump_dir / 'meta.npz'))
    hop = int(meta['hop'])

    d = np.load(str(dump_dir / f'pbfdkf_{args.side}.npz'), allow_pickle=True)
    n_frames = min(d['near'].shape[0], args.max_frames)
    n_freqs  = d['error_spec'].shape[1]
    n_parts  = d['P'].shape[1]

    # Try to read mu / delta from Python AEC config defaults
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from aec import AecConfig, AecPreset
    cfg = AecConfig.from_preset(AecPreset.BALANCED, sample_rate=16000)
    mu, delta = float(cfg.mu), float(cfg.delta)

    with open(args.out_bin, 'wb') as f:
        f.write(struct.pack('<iiiiff', hop, n_freqs, n_parts, n_frames, mu, delta))
        for i in range(n_frames):
            f.write(d['near'][i].astype('<f4').tobytes())
            f.write(d['far'][i].astype('<f4').tobytes())

            mu_scale = d['mu_scale'][i]
            if np.isscalar(mu_scale) or (hasattr(mu_scale, 'shape') and mu_scale.shape == ()):
                f.write(struct.pack('<i', 0))
                f.write(struct.pack('<f', float(mu_scale)))
            else:
                arr = np.asarray(mu_scale, dtype=np.float32)
                if arr.shape == (n_freqs,):
                    f.write(struct.pack('<i', 1))
                    f.write(arr.tobytes())
                else:
                    # Scalar masquerading as 0-d
                    f.write(struct.pack('<i', 0))
                    f.write(struct.pack('<f', float(arr)))

            # Expected outputs (Python state AFTER process() returned)
            err_spec  = d['error_spec'][i]
            echo_spec = d['echo_spec'][i]
            err_inter  = np.empty(2 * n_freqs, dtype=np.float32)
            err_inter[0::2] = err_spec.real;  err_inter[1::2] = err_spec.imag
            echo_inter = np.empty(2 * n_freqs, dtype=np.float32)
            echo_inter[0::2] = echo_spec.real; echo_inter[1::2] = echo_spec.imag

            # 'output' isn't directly captured by hook — derive via near - irfft(echo).
            # For parity simplicity, we use 'error' field from the hook
            # which captured the time-domain output of process().
            out_arr = d['error'][i].astype(np.float32)
            f.write(out_arr.tobytes())
            f.write(err_inter.tobytes())
            f.write(echo_inter.tobytes())
            f.write(d['P'][i].astype(np.float32).tobytes())
            f.write(d['R'][i].astype(np.float32).tobytes())

    print(f'Wrote {n_frames} frames × n_freqs={n_freqs} × n_parts={n_parts} → {args.out_bin}')


if __name__ == '__main__':
    main()
