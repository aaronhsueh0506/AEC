"""Generate a binary golden for the C reverb_model port (WS5 Phase 5.1).

Runs the Python ReverbModel over a deterministic float32 input sequence (both
update flavours + a decay<=0 no-op case) and writes inputs + expected outputs to
a raw little-endian file that c_impl/test/parity_reverb_model.c replays.

Layout (LE):
  int32 n_bins, int32 n_nfs, int32 n_u
  n_nfs × [ ps[n_bins] f32 | scaling f32 | decay f32 | expected[n_bins] f32 ]
  n_u   × [ ps[n_bins] f32 | scaling[n_bins] f32 | decay f32 | expected[n_bins] f32 ]

Run: python3 python/diag/gen_reverb_golden.py /tmp/reverb_golden.bin
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from modules.residual.reverb_model import ReverbModel  # noqa: E402

N_BINS = 257
N_NFS = 24
N_U = 24


def main():
    out = sys.argv[1] if len(sys.argv) > 1 else '/tmp/reverb_golden.bin'
    rng = np.random.RandomState(7)
    with open(out, 'wb') as f:
        np.array([N_BINS, N_NFS, N_U], dtype=np.int32).tofile(f)

        # update_no_freq_shaping sequence (scalar scaling); call 5 uses decay=0.
        m = ReverbModel(n_bins=N_BINS)
        for i in range(N_NFS):
            ps = (rng.rand(N_BINS).astype(np.float32) * 1.0e6).astype(np.float32)
            scaling = np.float32(rng.rand() * 2.0)
            decay = np.float32(0.0 if i == 5 else 0.5 + rng.rand() * 0.45)
            m.update_no_freq_shaping(ps, float(scaling), float(decay))
            ps.tofile(f)
            np.float32(scaling).tofile(f)
            np.float32(decay).tofile(f)
            m.reverb.astype(np.float32).tofile(f)

        # update sequence (per-bin scaling array); call 9 uses decay=0.
        m = ReverbModel(n_bins=N_BINS)
        for i in range(N_U):
            ps = (rng.rand(N_BINS).astype(np.float32) * 1.0e6).astype(np.float32)
            scaling = (rng.rand(N_BINS).astype(np.float32) * 2.0).astype(np.float32)
            decay = np.float32(0.0 if i == 9 else 0.5 + rng.rand() * 0.45)
            m.update(ps, scaling, float(decay))
            ps.tofile(f)
            scaling.tofile(f)
            np.float32(decay).tofile(f)
            m.reverb.astype(np.float32).tofile(f)
    print(f"wrote {out}  ({N_BINS} bins, {N_NFS} nfs + {N_U} u calls)")


if __name__ == '__main__':
    main()
