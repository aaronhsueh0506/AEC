#!/usr/bin/env python3
"""Phase 3 ResFilter parity check (Python vs C)."""
import numpy as np
import sys
from pathlib import Path

PY_FILE = Path(__file__).parent / 'parity_resfilter_python.npz'
C_BIN   = Path(__file__).parent / 'parity_resfilter_c.bin'

# Tolerance for ResFilter: includes WOLA, FFT, multiple EMAs, complex
# spectral chain — float32 cumulative drift can reach ~1e-3 over 200 frames.
RTOL = 5e-3
ATOL = 1e-4


def main():
    py = np.load(PY_FILE)
    BLOCK, FRAME, HOP, NF, N, SR = py['config']
    with open(C_BIN, 'rb') as fp:
        data = fp.read()
    hdr = np.frombuffer(data[:24], dtype=np.int32)
    if hdr[0] != BLOCK or hdr[3] != NF or hdr[4] != N:
        print(f"ERROR: header mismatch C={hdr.tolist()}"); sys.exit(2)

    body = np.frombuffer(data[24:], dtype=np.float32)
    per_frame = HOP + 4 * NF
    body = body.reshape(N, per_frame)
    c_output    = body[:, 0:HOP].reshape(-1)
    c_gain      = body[:, HOP:HOP+NF]
    c_echo_psd  = body[:, HOP+NF:HOP+2*NF]
    c_error_psd = body[:, HOP+2*NF:HOP+3*NF]
    c_noise_psd = body[:, HOP+3*NF:HOP+4*NF]

    print(f"Phase 3 parity (rtol={RTOL}, atol={ATOL}):")
    all_ok = True
    for name, p, c in [('output',     py['output'],   c_output),
                       ('gain',       py['gain'],     c_gain),
                       ('echo_psd',   py['echo_psd'], c_echo_psd),
                       ('error_psd',  py['error_psd'],c_error_psd),
                       ('noise_psd',  py['noise_psd'],c_noise_psd)]:
        ok = np.allclose(p, c, rtol=RTOL, atol=ATOL)
        max_abs = float(np.max(np.abs(p - c)))
        sig = np.abs(p) > 1e-4
        if sig.sum() > 0:
            mean_rel = float(np.mean(np.abs(p[sig] - c[sig]) / (np.abs(p[sig]) + ATOL)))
            p99_rel = float(np.percentile(np.abs(p[sig] - c[sig]) / (np.abs(p[sig]) + ATOL), 99))
        else:
            mean_rel = p99_rel = 0.0
        flag = "✓" if ok else "✗"
        print(f"  {flag} {name:12} max_abs={max_abs:.3e} mean_rel={mean_rel:.3e} p99_rel={p99_rel:.3e}")
        all_ok = all_ok and ok

    if all_ok:
        print("PARITY OK — Phase 3 ResFilter verified.")
        sys.exit(0)
    else:
        print("PARITY FAIL — root cause before Phase 5.")
        sys.exit(1)


if __name__ == '__main__':
    main()
