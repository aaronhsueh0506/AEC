#!/usr/bin/env python3
"""Phase 2a parity baseline: FilterErleEstimator + FullbandErleEstimator +
compute_erle_confidence.

Drives all three with deterministic synthetic input (same RNG seed as Phase 1)
and saves per-frame state for cross-language compare.
"""
import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from aec import FilterErleEstimator, FullbandErleEstimator, compute_erle_confidence

N_FREQS = 257
N_FRAMES = 200
SEED = 42

OUT_FILE   = Path(__file__).parent / 'parity_multi_erle_python.npz'
INPUT_FILE = Path(__file__).parent / 'parity_multi_erle_input.bin'


def main():
    rng = np.random.default_rng(SEED)

    # Synthetic complex echo/error specs (~ FFT bin scale)
    echo_re = rng.standard_normal((N_FRAMES, N_FREQS)).astype(np.float32) * 0.3
    echo_im = rng.standard_normal((N_FRAMES, N_FREQS)).astype(np.float32) * 0.3
    err_re  = rng.standard_normal((N_FRAMES, N_FREQS)).astype(np.float32) * 0.1
    err_im  = rng.standard_normal((N_FRAMES, N_FREQS)).astype(np.float32) * 0.1

    # FullbandErle inputs: synthetic near/error broadband powers
    near_pwr = (rng.uniform(1e-7, 1e-2, N_FRAMES)).astype(np.float32)
    err_pwr  = (rng.uniform(1e-7, 1e-2, N_FRAMES)).astype(np.float32)

    # Schedule: alternating far_active / dt levels
    far_active = np.array([1, 1, 1, 0, 1, 1, 1, 1, 1, 1] * (N_FRAMES // 10),
                          dtype=np.int32)
    dt_indicator = np.linspace(0.0, 0.6, N_FRAMES).astype(np.float32)

    # Run Python estimators
    fe = FilterErleEstimator(N_FREQS)
    fbe = FullbandErleEstimator()

    fe_erle_states = np.zeros((N_FRAMES, N_FREQS), dtype=np.float32)
    fbe_states     = np.zeros(N_FRAMES, dtype=np.float32)
    confidence     = np.zeros(N_FRAMES, dtype=np.float32)

    for f in range(N_FRAMES):
        echo_spec = (echo_re[f] + 1j * echo_im[f]).astype(np.complex64)
        err_spec  = (err_re[f]  + 1j * err_im[f]).astype(np.complex64)

        fe.update(echo_spec, err_spec, bool(far_active[f]), float(dt_indicator[f]))
        fbe.update(float(near_pwr[f]), float(err_pwr[f]),
                   bool(far_active[f]), float(dt_indicator[f]))

        fe_erle_states[f] = fe.erle.copy()
        fbe_states[f] = fbe.fb_erle
        confidence[f] = compute_erle_confidence(fe.erle, fbe.fb_erle)

    np.savez(OUT_FILE,
             fe_erle=fe_erle_states, fbe=fbe_states, confidence=confidence,
             config=np.array([N_FREQS, N_FRAMES], dtype=np.int32))

    # Raw input bin for C consumer
    with open(INPUT_FILE, 'wb') as fp:
        fp.write(echo_re.tobytes())
        fp.write(echo_im.tobytes())
        fp.write(err_re.tobytes())
        fp.write(err_im.tobytes())
        fp.write(near_pwr.tobytes())
        fp.write(err_pwr.tobytes())
        fp.write(far_active.tobytes())
        fp.write(dt_indicator.tobytes())

    print(f"Saved Python baseline → {OUT_FILE}")
    print(f"Saved C input bin     → {INPUT_FILE}")
    print(f"  fe_erle[final] mean: {np.mean(fe_erle_states[-1]):.4e}")
    print(f"  fbe[final]:          {fbe_states[-1]:.4e}")
    print(f"  confidence[final]:   {confidence[-1]:.4e}")


if __name__ == '__main__':
    main()
