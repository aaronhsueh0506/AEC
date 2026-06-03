"""Generate a binary golden for the C erl_estimator port (WS5 Phase 5.2).

Runs the Python ErlEstimator over a deterministic float32 render/capture-PSD
sequence and writes inputs + full per-frame state to a raw little-endian file
that c_impl/test/parity_erl_estimator.c replays.

The sequence is engineered to exercise every branch:
  * startup gate (blocks_since_reset < startup_hops → early return)
  * converged_filter gate (toggled off for a stretch → early return)
  * per-bin downward jump (new_erl < erl[k]) → 10% nudge + hold-counter arm,
    including the max(.,_MIN_ERL=0.01) clamp on deep drops
  * x2[k] <= x2_min bins (no update) interleaved with active bins
  * the hold-counter decrement + doubling branch (HOLD_HOPS=400 so we run
    >400 frames with bins that stop re-arming, forcing the 2× recovery)
  * endpoint mirroring erl[0]=erl[1], erl[-1]=erl[-2]
  * the fullband path: x2_sum/y2_sum pairwise f32 sums, the threshold gate,
    the downward-nudge, max-clamp, hold decrement + 2× recovery.

Real captured dtypes (single DT case, preset=balanced, hop=160):
  render_psd / capture_psd float32 (257,); _erl float32; _hold_counters int32;
  _x2_min = 44015068.0 (py float); _erl_time_domain py float; n_bins=257,
  startup_hops=200, HOLD_HOPS=400.

Layout (LE):
  int32 n_bins, int32 n_frames, int32 startup_hops, int32 hold_hops
  float64 x2_min
  n_frames × [ x2[n_bins] f32 | y2[n_bins] f32 | converged i32
             | erl[n_bins] f32 | hold_counters[n_bins-2] i32
             | erl_time_domain f64 | hold_counter_time_domain i32 ]

Run: python3 python/diag/gen_erl_estimator_golden.py /tmp/erl_golden.bin
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from modules.state.erl_estimator import ErlEstimator  # noqa: E402

N_BINS = 257
STARTUP_HOPS = 200
# Run well past startup + more than a full HOLD_HOPS window so the hold-counter
# expiry / doubling branch fires for both the per-bin and the FULLBAND estimate
# (the fullband counter only expires after >400 active frames with NO new
# fullband minimum, so the tail region below holds the fullband ratio high+flat).
N_FRAMES = 1100
HOP_SIZE = 160
SAMPLE_RATE = 16000

_X2_MIN = 44015068.0   # ErlEstimator._AEC3_X2_MIN at hop=160 (no-op scaling)


def make_frame(rng, i):
    """Deterministic float32 (x2, y2, converged) frame.

    x2 (render PSD) is built well above _x2_min on most bins so the per-bin
    branch is active; a deterministic subset is pushed BELOW _x2_min so the
    skip path is also covered. y2 (capture PSD) is varied so new_erl wanders
    up and down relative to the running estimate, arming hold counters on the
    down-swings and letting them expire (→ doubling) on the long flat stretches.
    """
    # Base render power comfortably above the X2_MIN threshold.
    scale = _X2_MIN * (4.0 + 8.0 * rng.rand(N_BINS).astype(np.float32))
    x2 = scale.astype(np.float32)

    # Push a deterministic ~15% of bins below threshold (skip path).
    below = (rng.rand(N_BINS) < 0.15)
    x2[below] = (_X2_MIN * (0.05 + 0.5 * rng.rand(int(below.sum())).astype(np.float32))
                 ).astype(np.float32)

    # Capture power: a slowly drifting ERL target with occasional deep dips
    # (force max(.,0.01) clamp) and occasional spikes (no-update path).
    erl_target = 0.02 + 0.3 * (0.5 + 0.5 * np.sin(0.05 * i + np.linspace(0, 6.0, N_BINS)))
    erl_target = erl_target.astype(np.float32)
    if i >= 600:
        # Tail region: hold the ERL target HIGH and flat (above any prior
        # estimate) so neither the per-bin nor the fullband estimate finds a new
        # minimum → hold counters decrement uninterrupted past 400 → the 2×
        # recovery (per-bin 'double' and fullband 'fb_double') branches fire.
        erl_target = (8.0 + 0.0 * erl_target).astype(np.float32)
        y2 = (x2 * erl_target).astype(np.float32)
        return x2.astype(np.float32), y2.astype(np.float32), True
    if 220 <= i < 320:
        # Sustained deep dip on a FIXED bin band for many consecutive active
        # frames so the 10%-per-frame nudge actually crosses below _MIN_ERL=0.01
        # and the max(.,0.01) clamp branch fires.
        erl_target[40:60] = np.float32(1.0e-4)
    elif i % 11 == 0:
        # deep dip on a random subset → drive erl down (clamp may fire)
        dip = (rng.rand(N_BINS) < 0.25)
        erl_target[dip] = (1.0e-4 * rng.rand(int(dip.sum())).astype(np.float32)
                           ).astype(np.float32)
    if i % 9 == 0:
        # spike on a subset → new_erl >= erl[k] (no per-bin update)
        spike = (rng.rand(N_BINS) < 0.2)
        erl_target[spike] = (5.0 + 50.0 * rng.rand(int(spike.sum())).astype(np.float32)
                             ).astype(np.float32)
    y2 = (x2 * erl_target).astype(np.float32)

    # converged_filter: False during startup-ish warmup and a mid-run OFF stretch
    # to cover the gate, True elsewhere.
    if i < 205:
        converged = (i >= 150)            # exercise startup gate AND converged gate
    elif 320 <= i < 360:
        converged = False                 # mid-run de-converge stretch
    else:
        converged = True
    return x2.astype(np.float32), y2.astype(np.float32), bool(converged)


def main():
    out = sys.argv[1] if len(sys.argv) > 1 else '/tmp/erl_golden.bin'
    rng = np.random.RandomState(23)

    e = ErlEstimator(startup_phase_length_hops=STARTUP_HOPS, n_bins=N_BINS,
                     hop_size=HOP_SIZE)
    hold_hops = int(4.0 * 100)  # _HOLD_HOPS

    with open(out, 'wb') as f:
        np.array([N_BINS, N_FRAMES, STARTUP_HOPS, hold_hops],
                 dtype=np.int32).tofile(f)
        np.array([e._x2_min], dtype=np.float64).tofile(f)
        for i in range(N_FRAMES):
            x2, y2, converged = make_frame(rng, i)
            e.update(render_psd=x2, capture_psd=y2, converged_filter=converged)

            x2.astype(np.float32).tofile(f)
            y2.astype(np.float32).tofile(f)
            np.array([1 if converged else 0], dtype=np.int32).tofile(f)
            e._erl.astype(np.float32).tofile(f)
            e._hold_counters.astype(np.int32).tofile(f)
            np.array([e._erl_time_domain], dtype=np.float64).tofile(f)
            np.array([e._hold_counter_time_domain], dtype=np.int32).tofile(f)

    print(f"wrote {out}  ({N_BINS} bins, {N_FRAMES} frames, "
          f"startup={STARTUP_HOPS}, hold={hold_hops}, x2_min={e._x2_min})")


if __name__ == '__main__':
    main()
