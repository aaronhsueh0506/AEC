#!/usr/bin/env python3
"""Phase 2bcd parity check (RenderActivity + DTAnalyzer + FilterConvergence)."""
import numpy as np
import sys
from pathlib import Path

PY_FILE = Path(__file__).parent / 'parity_aux_python.npz'
C_BIN   = Path(__file__).parent / 'parity_aux_c.bin'

RTOL = 1e-4
ATOL = 1e-5


def main():
    py = np.load(PY_FILE)
    HOP, N = py['config']
    with open(C_BIN, 'rb') as fp:
        data = fp.read()
    hdr = np.frombuffer(data[:8], dtype=np.int32)
    if hdr[0] != HOP or hdr[1] != N:
        print(f"ERROR: header mismatch {hdr.tolist()}"); sys.exit(2)

    body = data[8:]
    pos = 0

    def take_int32(n):
        nonlocal pos
        a = np.frombuffer(body[pos:pos + n*4], dtype=np.int32)
        pos += n*4
        return a

    def take_float32(n):
        nonlocal pos
        a = np.frombuffer(body[pos:pos + n*4], dtype=np.float32)
        pos += n*4
        return a

    c = {
        'ra_active':     take_int32(N),
        'ra_stationary': take_int32(N),
        'ra_far_pwr':    take_float32(N),
        'dt_energy':     take_float32(N),
        'dt_shadow':     take_float32(N),
        'dt_advantage':  take_float32(N),
        'fc_converged':  take_int32(N),
        'fc_once':       take_int32(N),
        'fc_div':        take_float32(N),
    }

    print(f"Phase 2bcd parity (rtol={RTOL}, atol={ATOL}):")
    all_ok = True
    for name in c:
        p = py[name]
        cv = c[name]
        if p.dtype.kind == 'i':
            ok = np.array_equal(p, cv)
            mismatches = int(np.sum(p != cv))
            print(f"  {'✓' if ok else '✗'} {name:14} (int) shape={str(p.shape):8} mismatches={mismatches}")
        else:
            ok = np.allclose(p, cv, rtol=RTOL, atol=ATOL)
            max_abs = float(np.max(np.abs(p - cv)))
            max_rel = float(np.max(np.abs(p - cv) / (np.abs(p) + ATOL)))
            print(f"  {'✓' if ok else '✗'} {name:14} (flt) shape={str(p.shape):8} max_abs={max_abs:.3e} max_rel={max_rel:.3e}")
        all_ok = all_ok and ok

    if all_ok:
        print("PARITY OK — Phase 2bcd verified.")
        sys.exit(0)
    else:
        print("PARITY FAIL")
        sys.exit(1)


if __name__ == '__main__':
    main()
