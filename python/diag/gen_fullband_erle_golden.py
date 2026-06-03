"""Generate a binary golden for the C fullband_erle port (WS5 Phase 5.2).

Runs the Python FullBandErleEstimator over a deterministic multi-frame
float32 (x2, y2, e2, converged_filter) sequence and snapshots the FULL state
each frame so the C parity replay verifies the EMA / quality / max-min / hold
evolution bit-for-bit (not just a final scalar).

The sequence is designed to exercise every branch:
  - converged True / False frames (gate the accumulator on/off),
  - x2_sum below AND above the per-bin energy threshold (the X2 gate),
  - the 6-point accumulation firing repeatedly (-> _erle_log2 set, EMA steps),
  - quiet (e2_acum==0 -> no update) frames,
  - the hold_counter_inst_erle countdown reaching 0 (reset_accumulators),
  - rising and falling instantaneous ERLE (max/min envelope + quality EMA).

Real captured dtypes (BALANCED doubletalk case): x2/y2/e2 float32 length 257,
converged_filter bool; np.sum -> float32 (pairwise), then float() -> f64; every
downstream scalar (acum/ratio/log2/EMA/quality) is f64. thr=44015068.0 (hop=160
no-op), min_erle_log2=log2(1.001), max_erle_lf_log2=log2(4.001), td_alpha=0.05.

Layout (LE):
  int32 n_freqs, int32 n_frames
  float64 x2_band_energy_threshold, float64 min_erle_log2, float64 max_erle_lf_log2
  n_frames x [ x2[n_freqs] f32 | y2[n_freqs] f32 | e2[n_freqs] f32 | conv u8
             | erle_time_domain_log2 f64
             | linear_quality_valid u8 | linear_quality f64
             | inst_has_erle_log2 u8 | inst_erle_log2 f64
             | inst_inst_quality_estimate f64
             | inst_max_erle_log2 f64 | inst_min_erle_log2 f64
             | inst_y2_acum f64 | inst_e2_acum f64 | inst_num_points i32
             | hold_counter_inst_erle i32 ]

Run: python3 python/diag/gen_fullband_erle_golden.py /tmp/fberle_golden.bin
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from modules.state.fullband_erle import FullBandErleEstimator  # noqa: E402

N_FREQS = 257
# 270 active frames (gate passes -> 6-pt accumulator fires repeatedly, EMA +
# max/min envelope + quality evolve) then a 50-frame non-converged QUIET TAIL so
# the 40-hop hold counter drains through exactly 0 and fires reset_accumulators
# (the `if hold_counter == 0` branch).
N_ACTIVE = 270
N_QUIET_TAIL = 50
N_FRAMES = N_ACTIVE + N_QUIET_TAIL
HOP_SIZE = 160


def make_inputs(rng, frame_idx):
    """Deterministic float32 (x2, y2, e2, converged) frame.

    x2 magnitude is chosen so x2_sum straddles the per-bin energy threshold
    (44015068 * 257 ~ 1.13e10) across the sequence; y2/e2 chosen so the y2/e2
    ratio sweeps high->low (rising/falling instantaneous ERLE), with occasional
    zero-e2 frames (no estimate update) and non-converged frames. The X2 gate
    must PASS most frames so the 6-point accumulator fires, _erle_log2 updates,
    the max/min envelope + quality EMA evolve, and the 40-hop hold counter
    eventually counts down to 0 (reset_accumulators).
    """
    # Per-bin x2 ~ render-PSD scale (1e8..1e10). The gate needs the per-bin
    # AVERAGE > 4.4e7, so a base of ~3e8 passes comfortably; a few frames drop
    # below to exercise the gate-fail branch.
    if frame_idx >= N_ACTIVE:
        # quiet tail: non-converged so no accumulation -> hold counter drains
        x2 = (1.0e6 + 1.0e5 * rng.rand(N_FREQS).astype(np.float32)).astype(np.float32)
        y2 = (1.0e5 + 5.0e5 * rng.rand(N_FREQS).astype(np.float32)).astype(np.float32)
        e2 = (1.0e4 + 1.0e4 * rng.rand(N_FREQS).astype(np.float32)).astype(np.float32)
        return (x2.astype(np.float32), y2.astype(np.float32),
                e2.astype(np.float32), False)

    if frame_idx % 11 == 0:
        # below-threshold render: tiny x2 -> X2 gate fails (no accumulation)
        x2 = (1.0e6 + 1.0e5 * rng.rand(N_FREQS).astype(np.float32)).astype(np.float32)
    else:
        scale = np.float32(3.0e8 + 1.0e9 * (frame_idx % 7) / 6.0)
        x2 = (scale * (0.5 + rng.rand(N_FREQS).astype(np.float32))).astype(np.float32)

    # y2 capture PSD ~ int16 scale.
    y2 = (1.0e5 + 5.0e5 * rng.rand(N_FREQS).astype(np.float32)).astype(np.float32)

    # e2 error PSD: sweep the ratio so ERLE rises then falls; some zero frames.
    if frame_idx % 13 == 0:
        e2 = np.zeros(N_FREQS, dtype=np.float32)            # e2_acum stays 0
    else:
        # ratio y2/e2 high (good cancellation) early -> low (poor) later, with a
        # periodic dip so the inst ERLE both rises and falls across the window.
        atten = np.float32(2.0 + 18.0 * ((frame_idx * 7) % 23) / 22.0)
        e2 = ((y2 / atten) +
              1.0e2 * rng.rand(N_FREQS).astype(np.float32)).astype(np.float32)

    converged = (frame_idx % 9) != 0  # mostly converged, periodic non-converged
    return x2.astype(np.float32), y2.astype(np.float32), e2.astype(np.float32), converged


def main():
    out = sys.argv[1] if len(sys.argv) > 1 else '/tmp/fberle_golden.bin'
    rng = np.random.RandomState(23)

    s = FullBandErleEstimator(min_erle=1.0, max_erle_l=4.0, hop_size=HOP_SIZE)

    with open(out, 'wb') as f:
        np.array([N_FREQS, N_FRAMES], dtype=np.int32).tofile(f)
        np.array([s._x2_band_energy_threshold, s._min_erle_log2,
                  s._max_erle_lf_log2], dtype=np.float64).tofile(f)
        for i in range(N_FRAMES):
            x2, y2, e2, conv = make_inputs(rng, i)
            s.update(x2=x2, y2=y2, e2=e2, converged_filter=conv)

            inst = s._instantaneous_erle
            q = s.get_inst_linear_quality_estimate()
            iel = inst.get_inst_erle_log2()

            x2.astype(np.float32).tofile(f)
            y2.astype(np.float32).tofile(f)
            e2.astype(np.float32).tofile(f)
            np.array([1 if conv else 0], dtype=np.uint8).tofile(f)

            np.array([s._erle_time_domain_log2], dtype=np.float64).tofile(f)
            np.array([0 if q is None else 1], dtype=np.uint8).tofile(f)
            np.array([0.0 if q is None else q], dtype=np.float64).tofile(f)

            np.array([0 if iel is None else 1], dtype=np.uint8).tofile(f)
            np.array([0.0 if iel is None else iel], dtype=np.float64).tofile(f)
            np.array([inst._inst_quality_estimate], dtype=np.float64).tofile(f)
            np.array([inst._max_erle_log2], dtype=np.float64).tofile(f)
            np.array([inst._min_erle_log2], dtype=np.float64).tofile(f)
            np.array([inst._y2_acum], dtype=np.float64).tofile(f)
            np.array([inst._e2_acum], dtype=np.float64).tofile(f)
            np.array([inst._num_points], dtype=np.int32).tofile(f)
            np.array([s._hold_counter_inst_erle], dtype=np.int32).tofile(f)

    print(f"wrote {out}  ({N_FREQS} bins, {N_FRAMES} frames, "
          f"thr={s._x2_band_energy_threshold}, "
          f"min_log2={s._min_erle_log2}, max_lf_log2={s._max_erle_lf_log2})")


if __name__ == '__main__':
    main()
