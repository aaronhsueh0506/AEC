#!/usr/bin/env python3
"""Generate synthetic test vectors for FilterErle/FullbandErle/erle_confidence
parity checks. Runs the Python classes on the vectors and writes inputs +
expected outputs to a flat .bin.

Layout (LE):
  header:   i32 n_freqs, i32 n_frames
  per frame:
    f32 echo_re[K], echo_im[K], err_re[K], err_im[K]
    i32 far_active, f64 dt_indicator,
    f64 near_power, f64 error_power
    --- expected outputs after this frame ---
    f32 erle_l1[K]
    f64 fb_erle
    f64 confidence
"""
import sys, struct, os
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np
from aec import (FilterErleEstimator, FullbandErleEstimator,
                 compute_erle_confidence)


def main():
    if len(sys.argv) < 2:
        print('usage: parity_export_erle.py <out.bin> [n_freqs] [n_frames]')
        return 1
    out_path = sys.argv[1]
    n_freqs  = int(sys.argv[2]) if len(sys.argv) > 2 else 257
    n_frames = int(sys.argv[3]) if len(sys.argv) > 3 else 200

    rng = np.random.default_rng(0xAEC0)
    fer = FilterErleEstimator(n_freqs)
    fbe = FullbandErleEstimator()

    with open(out_path, 'wb') as f:
        f.write(struct.pack('<ii', n_freqs, n_frames))
        for fi in range(n_frames):
            # Echo amplitude scales over time to simulate convergence
            echo_amp = 0.1 + 2.0 * (fi / n_frames)
            err_amp  = 0.05 * (1.0 + 0.5 * np.sin(fi * 0.05))
            echo_spec = (echo_amp * rng.standard_normal(n_freqs).astype(np.complex64)
                         + 1j * echo_amp * rng.standard_normal(n_freqs).astype(np.complex64)
                        ).astype(np.complex64)
            error_spec = (err_amp * rng.standard_normal(n_freqs).astype(np.complex64)
                          + 1j * err_amp * rng.standard_normal(n_freqs).astype(np.complex64)
                         ).astype(np.complex64)
            far_active   = bool(fi > 5)
            dt_indicator = float(0.5 * (1.0 + np.sin(fi * 0.1)))
            near_power   = float(0.5 * np.mean(np.abs(echo_spec) ** 2))
            error_power  = float(np.mean(np.abs(error_spec) ** 2))

            # Pack inputs (must come before update for harness order)
            f.write(echo_spec.real.astype('<f4').tobytes())
            f.write(echo_spec.imag.astype('<f4').tobytes())
            f.write(error_spec.real.astype('<f4').tobytes())
            f.write(error_spec.imag.astype('<f4').tobytes())
            f.write(struct.pack('<iddd', int(far_active), dt_indicator,
                                near_power, error_power))

            # Run Python oracle in same order as C will
            fer.update(echo_spec, error_spec, far_active, dt_indicator)
            fbe.update(near_power, error_power, far_active, dt_indicator)
            conf = compute_erle_confidence(fer.erle, fbe.fb_erle)

            f.write(fer.erle.astype('<f4').tobytes())
            f.write(struct.pack('<dd', float(fbe.fb_erle), float(conf)))

    print(f'Wrote {n_frames} frames × n_freqs={n_freqs} → {out_path}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
