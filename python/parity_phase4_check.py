#!/usr/bin/env python3
"""Phase 4 parity check (EPC + ShadowCopy)."""
import numpy as np
import sys
from pathlib import Path

PY_FILE = Path(__file__).parent / 'parity_phase4_python.npz'
C_BIN   = Path(__file__).parent / 'parity_phase4_c.bin'

RTOL = 1e-4
ATOL = 1e-5


def main():
    py = np.load(PY_FILE)
    N = int(py['config'][0])
    with open(C_BIN, 'rb') as fp:
        data = fp.read()
    hdr = np.frombuffer(data[:4], dtype=np.int32)
    if hdr[0] != N:
        print(f"ERROR: header mismatch {hdr[0]} vs {N}"); sys.exit(2)

    body = data[4:]
    pos = 0
    def take_int(n):
        nonlocal pos
        a = np.frombuffer(body[pos:pos+n*4], dtype=np.int32); pos += n*4; return a
    def take_flt(n):
        nonlocal pos
        a = np.frombuffer(body[pos:pos+n*4], dtype=np.float32); pos += n*4; return a

    c = {
        'epc_active':       take_int(N),
        'epc_hangover':     take_int(N),
        'epc_gain_fast':    take_flt(N),
        'epc_gain_slow':    take_flt(N),
        'epc_event_source': take_int(N),
        'sc_pause':         take_int(N),
        'sc_boost_q':       take_int(N),
        'sc_reverse':       take_int(N),
        'sc_baseline':      take_flt(N),
    }

    print(f"Phase 4 parity (rtol={RTOL}, atol={ATOL}):")
    all_ok = True
    for name, cv in c.items():
        p = py[name]
        if p.dtype.kind == 'i':
            ok = np.array_equal(p, cv)
            mismatches = int(np.sum(p != cv))
            print(f"  {'✓' if ok else '✗'} {name:20} (int) mismatches={mismatches}")
        else:
            ok = np.allclose(p, cv, rtol=RTOL, atol=ATOL)
            max_abs = float(np.max(np.abs(p - cv)))
            max_rel = float(np.max(np.abs(p - cv) / (np.abs(p) + ATOL)))
            print(f"  {'✓' if ok else '✗'} {name:20} (flt) max_abs={max_abs:.3e} max_rel={max_rel:.3e}")
        all_ok = all_ok and ok

    if all_ok:
        print("PARITY OK — Phase 4 verified.")
        sys.exit(0)
    else:
        print("PARITY FAIL")
        sys.exit(1)


if __name__ == '__main__':
    main()
