#!/usr/bin/env python3
"""Compare Python vs C multi_erle parity (Phase 2a)."""
import numpy as np
import sys
from pathlib import Path

PY_FILE = Path(__file__).parent / 'parity_multi_erle_python.npz'
C_BIN   = Path(__file__).parent / 'parity_multi_erle_c.bin'

RTOL = 1e-4
ATOL = 1e-5


def main():
    py = np.load(PY_FILE)
    n_freqs, n_frames = py['config']

    with open(C_BIN, 'rb') as fp:
        data = fp.read()
    hdr = np.frombuffer(data[:8], dtype=np.int32)
    if hdr[0] != n_freqs or hdr[1] != n_frames:
        print(f"ERROR: header mismatch C={hdr.tolist()} Py={[n_freqs, n_frames]}")
        sys.exit(2)

    # Per frame: erle[n_freqs], fb_erle[1], confidence[1]
    per_frame = n_freqs + 2
    body = np.frombuffer(data[8:], dtype=np.float32).reshape(n_frames, per_frame)
    c_erle = body[:, 0:n_freqs]
    c_fbe  = body[:, n_freqs]
    c_conf = body[:, n_freqs + 1]

    print(f"Phase 2a parity (rtol={RTOL}, atol={ATOL}):")
    all_ok = True
    for name, p, c in [('fe_erle', py['fe_erle'], c_erle),
                       ('fbe',     py['fbe'],     c_fbe),
                       ('confidence', py['confidence'], c_conf)]:
        ok = np.allclose(p, c, rtol=RTOL, atol=ATOL)
        max_abs = float(np.max(np.abs(p - c)))
        max_rel = float(np.max(np.abs(p - c) / (np.abs(p) + ATOL)))
        flag = "✓" if ok else "✗"
        print(f"  {flag} {name:12} shape={str(p.shape):16} max_abs={max_abs:.3e} max_rel={max_rel:.3e}")
        all_ok = all_ok and ok

    if all_ok:
        print("PARITY OK — Phase 2a multi_erle verified.")
        sys.exit(0)
    else:
        print("PARITY FAIL")
        sys.exit(1)


if __name__ == '__main__':
    main()
