"""Generate a binary golden for the C subband_erle port (WS5 Phase 5.2).

Runs the Python SubbandErleEstimator over a deterministic multi-frame
x2/y2/e2 sequence (incl. converged_filter True/False, a coherence gate mask,
the kPointsToAccumulate(=6) accumulation rollover, the low-render branch, the
onset-detection hold/release chain) and writes inputs + the full per-frame
ERLE state to a raw little-endian file that
c_impl/test/parity_subband_erle.c replays.

We emit MULTIPLE config variants back-to-back in one file so the C test can
exercise:
  - the production config (use_onset_detection=True,
    use_min_erle_during_onsets=True, e2y2_gate disabled, coh gate active);
  - use_min_erle_during_onsets=False (exercises the _erle_during_onsets
    smoothing branch inside the onset transition);
  - e2y2_gate_enabled=True (the E2/Y2 freeze gate);
  - coh_gate_mask=None (no coherence gate).

Real captured dtypes (doubletalk case 0I0XMl3M0...):
  x2 / y2 / e2 float32 (257,); converged_filter bool; coh_gate_mask bool (257,);
  _erle / _erle_onset_compensated / _erle_unbounded / _erle_during_onsets /
  _max_erle / _y2_acc / _e2_acc float32 (257,); _coming_onset / _low_render
  bool (257,); _hold_counters int32 (257,); _x2_band_energy_threshold f64.

Per-frame order matches the orchestrator: a single update(...) call per hop.

Layout (LE):
  int32 n_variants
  per variant:
    int32 n_bins, int32 n_frames
    int32 use_onset, int32 use_min_during, int32 e2y2_gate, int32 coh_active
    float32 max_erle_l, float32 max_erle_h, float32 min_erle, float32 e2y2_thr
    float64 x2_band_thr
    n_frames x [ x2[n_bins] f32 | y2[n_bins] f32 | e2[n_bins] f32
               | converged i32 | coh_mask[n_bins] u8 (only if coh_active)
               | erle[n_bins] f32 | erle_oc[n_bins] f32 | erle_unb[n_bins] f32
               | erle_during[n_bins] f32 | coming_onset[n_bins] u8
               | hold[n_bins] i32 | low_render[n_bins] u8
               | y2_acc[n_bins] f32 | e2_acc[n_bins] f32 | num_points i32 ]

Run: python3 python/diag/gen_subband_erle_golden.py /tmp/se_golden.bin
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from modules.state.subband_erle import SubbandErleEstimator  # noqa: E402

N_BINS = 257
N_FRAMES = 220          # > a few accumulate cycles + onset hold window (100)
HOP_SIZE = 160


def make_frame(rng, idx):
    """Deterministic float32 (x2, y2, e2) + converged flag + coh mask.

    Designs in: converged True/False phases, low-render frames, doubletalk-ish
    frames where e2 ~ y2 (high ratio -> e2y2 gate fires), clean-cancellation
    frames where e2 << y2 (large ERLE), and a coherence mask that drops out on
    a random subset.
    """
    # render power x2: mostly active, occasionally below the band threshold.
    x2 = (1.0e-3 + 5.0e-3 * rng.rand(N_BINS).astype(np.float32)).astype(np.float32)
    if idx % 11 == 0:
        # low-render frame: many bins below x2_band_energy_threshold (≈4.4e7
        # >> any float-scale psd, so essentially ALL bins are "low render").
        x2 = (1.0e-9 + 1.0e-9 * rng.rand(N_BINS).astype(np.float32)).astype(np.float32)

    # capture power y2 and error power e2.
    y2 = (1.0e-2 + 5.0e-2 * rng.rand(N_BINS).astype(np.float32)).astype(np.float32)
    if idx % 4 == 0:
        # doubletalk-ish: error ~ capture (high e2/y2 ratio).
        e2 = (y2 * (0.6 + 0.5 * rng.rand(N_BINS).astype(np.float32))).astype(np.float32)
    elif idx % 4 == 1:
        # clean cancellation: tiny error (large ERLE, hits caps).
        e2 = (y2 * (1.0e-4 + 1.0e-4 * rng.rand(N_BINS).astype(np.float32))).astype(np.float32)
    else:
        # moderate.
        e2 = (y2 * (0.01 + 0.2 * rng.rand(N_BINS).astype(np.float32))).astype(np.float32)
    # occasionally zero-out e2 on a subset (e2_acc could stay 0 -> mask False).
    if idx % 9 == 0:
        z = rng.rand(N_BINS) < 0.1
        e2 = e2.copy()
        e2[z] = 0.0
    e2 = e2.astype(np.float32)

    # converged_filter: warm up un-converged for the first few frames, then a
    # mostly-converged stretch with intermittent drop-outs.
    if idx < 5:
        converged = False
    elif idx % 13 == 0:
        converged = False
    else:
        converged = True

    # coherence gate: True (update allowed) for most bins, drop a random subset.
    coh = np.ones(N_BINS, dtype=bool)
    drop = rng.rand(N_BINS) < 0.15
    coh[drop] = False
    return x2, y2, e2, converged, coh


def run_variant(f, *, use_onset, use_min_during, e2y2_gate, coh_active, seed):
    rng = np.random.RandomState(seed)
    s = SubbandErleEstimator(
        n_bins=N_BINS, min_erle=1.0, max_erle_l=4.0, max_erle_h=1.5,
        use_onset_detection=use_onset,
        use_min_erle_during_onsets=use_min_during,
        hop_size=HOP_SIZE,
        e2y2_gate_enabled=e2y2_gate, e2y2_gate_threshold=0.5,
    )

    np.array([N_BINS, N_FRAMES], dtype=np.int32).tofile(f)
    np.array([1 if use_onset else 0, 1 if use_min_during else 0,
              1 if e2y2_gate else 0, 1 if coh_active else 0],
             dtype=np.int32).tofile(f)
    np.array([4.0, 1.5, 1.0, 0.5], dtype=np.float32).tofile(f)
    np.array([float(s._x2_band_energy_threshold)], dtype=np.float64).tofile(f)

    for i in range(N_FRAMES):
        x2, y2, e2, converged, coh = make_frame(rng, i)
        coh_arg = coh if coh_active else None
        s.update(x2=x2, y2=y2, e2=e2, converged_filter=converged,
                 coh_gate_mask=coh_arg)

        x2.astype(np.float32).tofile(f)
        y2.astype(np.float32).tofile(f)
        e2.astype(np.float32).tofile(f)
        np.array([1 if converged else 0], dtype=np.int32).tofile(f)
        if coh_active:
            coh.astype(np.uint8).tofile(f)
        s._erle.astype(np.float32).tofile(f)
        s._erle_onset_compensated.astype(np.float32).tofile(f)
        s._erle_unbounded.astype(np.float32).tofile(f)
        s._erle_during_onsets.astype(np.float32).tofile(f)
        s._coming_onset.astype(np.uint8).tofile(f)
        s._hold_counters.astype(np.int32).tofile(f)
        s._low_render_energy.astype(np.uint8).tofile(f)
        s._y2_acc.astype(np.float32).tofile(f)
        s._e2_acc.astype(np.float32).tofile(f)
        np.array([s._num_points], dtype=np.int32).tofile(f)


def main():
    out = sys.argv[1] if len(sys.argv) > 1 else '/tmp/se_golden.bin'
    variants = [
        dict(use_onset=True,  use_min_during=True,  e2y2_gate=False, coh_active=True,  seed=7),
        dict(use_onset=True,  use_min_during=False, e2y2_gate=False, coh_active=True,  seed=23),
        dict(use_onset=True,  use_min_during=True,  e2y2_gate=True,  coh_active=True,  seed=41),
        dict(use_onset=True,  use_min_during=True,  e2y2_gate=False, coh_active=False, seed=59),
        dict(use_onset=False, use_min_during=True,  e2y2_gate=False, coh_active=True,  seed=83),
    ]
    with open(out, 'wb') as f:
        np.array([len(variants)], dtype=np.int32).tofile(f)
        for v in variants:
            run_variant(f, **v)
    print(f"wrote {out}  ({len(variants)} variants, {N_BINS} bins, "
          f"{N_FRAMES} frames each)")


if __name__ == '__main__':
    main()
