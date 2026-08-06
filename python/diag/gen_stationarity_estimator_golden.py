"""Generate a binary golden for the C stationarity_estimator port (WS5 Phase 5.1).

Runs the Python StationarityEstimator over a deterministic float32 render-PSD
sequence (warmup avg-init phase → EMA rising/falling → past initial_phase so the
mask10 (10×pn < pb) branch fires; the history ring fills + the flag/hangover/
3-bin-smooth/is_block_stationary chain all evolve) and writes inputs + expected
outputs to a raw little-endian file that
c_impl/test/historical/parity_stationarity_estimator.c replays.

Per-frame order matches the real pipeline (orchestrator.py:1809-1810):
  update_noise_estimator(psd)  then  update_stationarity_flags(psd)

Real captured dtypes: render_psd float32, average_reverb None, noise EMA float32,
flags bool, hangovers int32, history float32, n_freqs=257, window=hangover=5 hops.

Layout (LE):
  int32 n_freqs, int32 n_frames, int32 window_hops, int32 hangover_hops
  n_frames × [ psd[n_freqs] f32
             | noise[n_freqs] f32          (after update_noise_estimator)
             | flags[n_freqs] u8           (after update_stationarity_flags)
             | hangovers[n_freqs] i32
             | band_mask[n_freqs] u8
             | is_block_stationary i32 ]

Run: python3 python/diag/gen_stationarity_estimator_golden.py /tmp/stat_golden.bin
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from modules.state.stationarity_estimator import StationarityEstimator  # noqa: E402

N_FREQS = 257
N_FRAMES = 280          # > initial_phase_hops (200) so the mask10 branch fires
HOP_SAMPLES = 160
SAMPLE_RATE = 16000


def make_psd(rng, frame_idx):
    """Deterministic float32 render-PSD frame.

    Phase A (0..7, the avg_init warmup): small near-constant 'stationary' floor
    so the noise estimate seeds. Then a mix of constant-ish (stationary) and
    spiky (non-stationary) frames, plus occasional 10×-jump spikes to exercise
    the rising/mask10 branch and the falling branch. Values stay positive and
    in a float-PSD-plausible range (~1e-8 .. ~1e-2).
    """
    base = (1.0e-6 + 5.0e-7 * rng.rand(N_FREQS).astype(np.float32)).astype(np.float32)
    if frame_idx < 8:
        # warmup: tiny, near-constant — seeds the noise floor
        psd = (1.0e-8 + 2.0e-9 * rng.rand(N_FREQS).astype(np.float32)).astype(np.float32)
    elif frame_idx % 7 == 0:
        # big broadband spike on a random subset → rising branch, some 10×
        psd = base.copy()
        idx = rng.rand(N_FREQS) < 0.4
        psd[idx] = (psd[idx] * (50.0 + 100.0 * rng.rand(int(idx.sum())).astype(np.float32))
                    ).astype(np.float32)
    elif frame_idx % 5 == 0:
        # quiet frame → falling branch toward the floor
        psd = (3.0e-9 + 1.0e-9 * rng.rand(N_FREQS).astype(np.float32)).astype(np.float32)
    else:
        # near-stationary low-level frame
        psd = base.copy()
    return psd.astype(np.float32)


def main():
    out = sys.argv[1] if len(sys.argv) > 1 else '/tmp/stat_golden.bin'
    rng = np.random.RandomState(11)

    s = StationarityEstimator(n_freqs=N_FREQS, hop_samples=HOP_SAMPLES,
                              sample_rate=SAMPLE_RATE)

    with open(out, 'wb') as f:
        np.array([N_FREQS, N_FRAMES, s._window_hops, s._hangover_hops],
                 dtype=np.int32).tofile(f)
        for i in range(N_FRAMES):
            psd = make_psd(rng, i)
            # Real pipeline order: noise estimator first, then flags.
            s.update_noise_estimator(psd)
            s.update_stationarity_flags(psd)

            mask = s.band_stationary_mask()
            blk = 1 if s.is_block_stationary() else 0

            psd.astype(np.float32).tofile(f)
            s.noise.noise.astype(np.float32).tofile(f)
            s.stationarity_flags.astype(np.uint8).tofile(f)
            s.hangovers.astype(np.int32).tofile(f)
            mask.astype(np.uint8).tofile(f)
            np.array([blk], dtype=np.int32).tofile(f)

    print(f"wrote {out}  ({N_FREQS} bins, {N_FRAMES} frames, "
          f"window={s._window_hops}, hangover={s._hangover_hops})")


if __name__ == '__main__':
    main()
