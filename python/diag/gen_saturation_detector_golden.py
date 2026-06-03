"""Generate a binary golden for the C saturation_detector port (WS5 Phase 5.2).

Runs the Python SaturationDetector (AecState variant,
modules/state/saturation_detector.py) over a deterministic sequence that
exercises all three branches and their boundaries, then writes inputs + the
expected per-frame _saturated_echo flag to a raw little-endian file that
c_impl/test/parity_saturation_detector.c replays.

Real captured dtypes (aec_state.py:333 call site, monkeypatch on one
doubletalk case): render_block float32 1-D (hop=160); saturated_capture +
usable_linear_estimate Python bool; subtractor_s_refined/coarse_max_abs +
echo_path_gain Python float (f64); output _saturated_echo Python bool.

Branches covered across frames:
  - saturated_capture False (early return → flag stays False).
  - usable_linear_estimate True: refined>THR / coarse>THR / both<=THR /
    exactly == THR (strict > so == is NOT saturated).
  - usable_linear_estimate False (render path): peak above / below /
    exactly == _INT16_SATURATION; empty render_block (size 0 → max_sample 0);
    varied echo_path_gain incl. the threshold-straddling value.

Layout (LE):
  int32 n_frames
  per frame:
    int32 render_block_len
    float32 render_block[render_block_len]
    int32  saturated_capture        (0/1)
    int32  usable_linear_estimate   (0/1)
    float64 subtractor_s_refined_max_abs
    float64 subtractor_s_coarse_max_abs
    float64 echo_path_gain
    uint8  expected_saturated_echo

Run: python3 python/diag/gen_saturation_detector_golden.py /tmp/satdet_golden.bin
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from modules.state.saturation_detector import SaturationDetector  # noqa: E402

HOP = 160
_THR = 20000.0
_MARGIN = 10.0
_INT16 = 32000.0


def build_frames(rng):
    """List of dicts: render_block(f32 1-D), saturated_capture, usable, refined,
    coarse, echo_path_gain. Hand-built to cover every branch + boundary."""
    frames = []

    def f32(n, lo, hi):
        return (lo + (hi - lo) * rng.rand(n).astype(np.float32)).astype(np.float32)

    # --- early-return branch: saturated_capture False (flag must stay False) ---
    frames.append(dict(rb=f32(HOP, -1.0, 1.0), sat=False, usable=False,
                       refined=999999.0, coarse=999999.0, epg=10.0))
    frames.append(dict(rb=f32(HOP, -30000.0, 30000.0), sat=False, usable=True,
                       refined=999999.0, coarse=0.0, epg=1.0))

    # --- usable-linear branch ---
    # refined > THR, coarse small  → True
    frames.append(dict(rb=f32(HOP, -1.0, 1.0), sat=True, usable=True,
                       refined=20000.0001, coarse=100.0, epg=1.0))
    # coarse > THR, refined small  → True
    frames.append(dict(rb=f32(HOP, -1.0, 1.0), sat=True, usable=True,
                       refined=100.0, coarse=25000.0, epg=1.0))
    # both <= THR → False
    frames.append(dict(rb=f32(HOP, -1.0, 1.0), sat=True, usable=True,
                       refined=19999.0, coarse=15000.0, epg=1.0))
    # exactly == THR (strict > → NOT saturated) on both
    frames.append(dict(rb=f32(HOP, -1.0, 1.0), sat=True, usable=True,
                       refined=20000.0, coarse=20000.0, epg=1.0))
    # both huge → True
    frames.append(dict(rb=f32(HOP, -1.0, 1.0), sat=True, usable=True,
                       refined=1.0e9, coarse=1.0e9, epg=1.0))
    # zeros → False
    frames.append(dict(rb=f32(HOP, -1.0, 1.0), sat=True, usable=True,
                       refined=0.0, coarse=0.0, epg=1.0))

    # --- render-path branch (usable False) ---
    # peak well above _INT16 → True   (max≈30000 * 1.0 * 10 = 300000)
    rb = f32(HOP, -30000.0, 30000.0)
    rb[7] = np.float32(30000.0)
    frames.append(dict(rb=rb, sat=True, usable=False,
                       refined=0.0, coarse=0.0, epg=1.0))
    # peak well below _INT16 → False  (max≈100 * 1.0 * 10 = 1000)
    frames.append(dict(rb=f32(HOP, -100.0, 100.0), sat=True, usable=False,
                       refined=0.0, coarse=0.0, epg=1.0))
    # echo_path_gain straddles: max=3200, gain=1.0 → peak=32000 == _INT16 (NOT >)
    rb = np.zeros(HOP, dtype=np.float32)
    rb[3] = np.float32(3200.0)
    frames.append(dict(rb=rb, sat=True, usable=False,
                       refined=0.0, coarse=0.0, epg=1.0))
    # same but gain slightly >1 → peak just above _INT16 → True
    frames.append(dict(rb=rb.copy(), sat=True, usable=False,
                       refined=0.0, coarse=0.0, epg=1.0000001))
    # negative-peak abs handling: large negative sample dominates
    rb = f32(HOP, -50.0, 50.0)
    rb[100] = np.float32(-40000.0)
    frames.append(dict(rb=rb, sat=True, usable=False,
                       refined=0.0, coarse=0.0, epg=1.0))
    # tiny gain keeps it below → False
    frames.append(dict(rb=f32(HOP, -30000.0, 30000.0), sat=True, usable=False,
                       refined=0.0, coarse=0.0, epg=1.0e-6))
    # empty render_block (size 0) → max_sample 0 → peak 0 → False
    frames.append(dict(rb=np.zeros(0, dtype=np.float32), sat=True, usable=False,
                       refined=0.0, coarse=0.0, epg=1.0e9))
    # fractional float32 values, mid-range gain, to stress f32→f64 promotion
    for _ in range(40):
        rb = f32(rng.randint(1, HOP + 1), -5000.0, 5000.0)
        frames.append(dict(rb=rb, sat=True, usable=False, refined=0.0,
                           coarse=0.0, epg=float(rng.rand() * 8.0)))
    # more usable-branch random frames near the threshold
    for _ in range(40):
        frames.append(dict(rb=f32(HOP, -1.0, 1.0), sat=True, usable=True,
                           refined=float(rng.rand() * 40000.0),
                           coarse=float(rng.rand() * 40000.0),
                           epg=1.0))
    return frames


def main():
    out = sys.argv[1] if len(sys.argv) > 1 else '/tmp/satdet_golden.bin'
    rng = np.random.RandomState(23)
    frames = build_frames(rng)

    d = SaturationDetector()
    with open(out, 'wb') as f:
        np.array([len(frames)], dtype=np.int32).tofile(f)
        n_true = 0
        for fr in frames:
            rb = fr['rb'].astype(np.float32)
            d.update(
                render_block=rb,
                saturated_capture=bool(fr['sat']),
                usable_linear_estimate=bool(fr['usable']),
                subtractor_s_refined_max_abs=float(fr['refined']),
                subtractor_s_coarse_max_abs=float(fr['coarse']),
                echo_path_gain=float(fr['epg']),
            )
            flag = 1 if d.saturated_echo() else 0
            n_true += flag

            np.array([rb.size], dtype=np.int32).tofile(f)
            rb.tofile(f)
            np.array([1 if fr['sat'] else 0,
                      1 if fr['usable'] else 0], dtype=np.int32).tofile(f)
            np.array([float(fr['refined']), float(fr['coarse']),
                      float(fr['epg'])], dtype=np.float64).tofile(f)
            np.array([flag], dtype=np.uint8).tofile(f)

    print(f"wrote {out}  ({len(frames)} frames, {n_true} saturated_echo=True)")


if __name__ == '__main__':
    main()
