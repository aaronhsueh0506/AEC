#!/usr/bin/env python3
"""Compare Python vs C PBFDKF parity baseline.

Hard requirement (per docs/c_rewrite_plan.md): drift attributable to
float32-vs-float64 reduction order only, not algorithmic mismatch.

Tolerance calibrated for 200-frame cumulative drift through EMA + FFT
chains. Float32 epsilon = 1.19e-7; after ~200 cumulative ops, observed
drift on significant elements (mag > 1e-4) ranges:
    mean_rel  ~ 1e-6   (epsilon-level)
    p99_rel   ~ 2e-5   (10× epsilon)
    max_rel   ~ 1e-3   (rare outliers on small mag values)

Tolerance: rtol=1e-4 atol=1e-5 — passes float32 noise, fails algorithmic
mismatch (which would produce ≥1% drift).

Usage:
    python3 parity_pbfdkf_gen.py                            # python baseline
    cd c_impl && ./bin/parity_pbfdkf_test \\
        ../python/parity_pbfdkf_input.bin \\
        ../python/parity_pbfdkf_c.bin
    python3 parity_pbfdkf_load_c.py                         # bin → npz
    python3 parity_pbfdkf_check.py                          # diff

Exit 0 = parity OK; non-zero = phase fails, halt.
"""
import numpy as np
import sys
from pathlib import Path

PY_FILE = Path(__file__).parent / 'parity_pbfdkf_python.npz'
C_FILE  = Path(__file__).parent / 'parity_pbfdkf_c.npz'

# Realistic float32-vs-float64 reduction-order tolerance over 200 frames
RTOL = 1e-4
ATOL = 1e-5


def diff_report(name, py, c):
    if py.shape != c.shape:
        print(f"  ✗ {name}: shape mismatch py={py.shape} c={c.shape}")
        return False
    abs_diff = np.abs(py - c)
    max_abs = float(np.max(abs_diff))
    rel_denom = np.abs(py) + ATOL
    max_rel = float(np.max(abs_diff / rel_denom))
    ok = np.allclose(py, c, rtol=RTOL, atol=ATOL)
    flag = "✓" if ok else "✗"
    print(f"  {flag} {name:20} shape={str(py.shape):20} max_abs={max_abs:.3e} max_rel={max_rel:.3e}")
    return ok


def main():
    if not PY_FILE.exists():
        print(f"ERROR: {PY_FILE} not found. Run parity_pbfdkf_gen.py first.")
        sys.exit(2)
    if not C_FILE.exists():
        print(f"ERROR: {C_FILE} not found. Build + run c_impl/bin/parity_pbfdkf_test first.")
        sys.exit(2)

    py = np.load(PY_FILE)
    c  = np.load(C_FILE)

    print(f"Comparing PBFDKF parity (rtol={RTOL}, atol={ATOL}):")
    all_ok = True
    for key in ('output', 'err_specs', 'echo_specs', 'P_states',
                'W_states', 'power_states'):
        ok = diff_report(key, py[key], c[key])
        all_ok = all_ok and ok

    print()
    if all_ok:
        print("PARITY OK — Phase 1 PBFDKF G1 KX blended verified.")
        sys.exit(0)
    else:
        print("PARITY FAIL — root cause before next phase.")
        print("Common drift sources:")
        print("  - np.float32 vs float in scalar arithmetic (mu_mean blend factor)")
        print("  - kiss_fft IFFT scaling: numpy divides by N, kiss does not")
        print("  - reduction order in Python `+=` vs C accumulator loop")
        print("  - delta cast: Python uses np.float32(self.delta) explicitly")
        sys.exit(1)


if __name__ == '__main__':
    main()
