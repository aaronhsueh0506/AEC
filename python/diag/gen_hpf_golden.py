"""Generate a binary golden for the C hpf port (mic-path high-pass biquad).

Runs the Python HighPassFilter (modules/preprocessing.py) over a deterministic
multi-frame stream, capturing per-frame input, expected float32 output, and the
fp64 delay-state (z1/z2) before and after each frame. Writes a raw little-endian
file that c_impl/test/parity_hpf.c replays through the C Hpf port and asserts
bit-exact float32 match (plus per-frame state-drift checks).

Why bit-exact is achievable: Python __init__ computes coefficients in fp64 and
process() runs the sample loop in Python `float` (fp64), then stores into
`np.empty_like(x)`. Production feeds a float32 mic block (orchestrator.py:1340,
near_end is float32), so the output array is float32 and `out[i] = yi` truncates
fp64 -> f32 — which the C port mirrors with `(float)yi`. The C Hpf struct also
uses double internally (hpf.h), so init coefficients + the recurrence are
identical to the bit.

Production config (config.py): highpass_cutoff_hz=80.0, sample_rate=16000,
hop=160. We replay that operating point.

Layout (LE):
  int32   n_frames
  int32   hop
  int32   sample_rate
  float64 cutoff_hz
  per frame:
    float64 z1_in, z2_in        (state BEFORE this frame's process)
    float64 z1_out, z2_out      (state AFTER this frame's process)
    float32 input[hop]
    float32 expected_output[hop]

Run: python3 python/diag/gen_hpf_golden.py /tmp/hpf_golden.bin
"""
import os
import struct
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from modules.preprocessing import HighPassFilter  # noqa: E402

CUTOFF_HZ = 80.0
SAMPLE_RATE = 16000
HOP = 160


def build_blocks(rng):
    """List of float32 1-D hop-length blocks exercising transients + steady
    state.  Mirrors a mic stream: silence, an impulse, a step (DC the HPF must
    reject), a tone, broadband noise, and large/small amplitudes."""
    blocks = []

    # frame 0: pure silence (output must stay 0; state stays 0)
    blocks.append(np.zeros(HOP, dtype=np.float32))

    # frame 1: unit impulse at sample 0 (full impulse response feeds state)
    b = np.zeros(HOP, dtype=np.float32)
    b[0] = 1.0
    blocks.append(b)

    # frame 2: DC step (HPF must drive output back toward 0; state non-trivial)
    blocks.append(np.full(HOP, 0.5, dtype=np.float32))

    # frame 3: low-frequency tone near/below cutoff (heavily attenuated)
    t = np.arange(HOP, dtype=np.float64)
    blocks.append((0.3 * np.sin(2.0 * np.pi * 50.0 * t / SAMPLE_RATE)).astype(np.float32))

    # frame 4: high-frequency tone (passes through)
    blocks.append((0.7 * np.sin(2.0 * np.pi * 2000.0 * t / SAMPLE_RATE)).astype(np.float32))

    # frame 5: broadband noise, moderate amplitude
    blocks.append((0.4 * rng.standard_normal(HOP)).astype(np.float32))

    # frame 6: large amplitude (near int16-normalised peaks)
    blocks.append((0.95 * rng.standard_normal(HOP)).astype(np.float32))

    # frame 7: tiny amplitude (denormal-adjacent values exercise f32 rounding)
    blocks.append((1e-6 * rng.standard_normal(HOP)).astype(np.float32))

    # frame 8: negative DC step (sign coverage)
    blocks.append(np.full(HOP, -0.5, dtype=np.float32))

    # frames 9..12: continued noise so state carries across many frames
    for _ in range(4):
        blocks.append((0.4 * rng.standard_normal(HOP)).astype(np.float32))

    return blocks


def main():
    out_path = sys.argv[1] if len(sys.argv) > 1 else '/tmp/hpf_golden.bin'

    rng = np.random.RandomState(0)
    blocks = build_blocks(rng)

    hpf = HighPassFilter(CUTOFF_HZ, SAMPLE_RATE)

    records = []
    for blk in blocks:
        assert blk.dtype == np.float32 and blk.shape == (HOP,)
        z1_in, z2_in = float(hpf.z1), float(hpf.z2)
        out = hpf.process(blk.copy())
        assert out.dtype == np.float32, out.dtype
        z1_out, z2_out = float(hpf.z1), float(hpf.z2)
        records.append((z1_in, z2_in, z1_out, z2_out, blk, out))

    with open(out_path, 'wb') as f:
        f.write(struct.pack('<iiid', len(records), HOP, SAMPLE_RATE, CUTOFF_HZ))
        for z1_in, z2_in, z1_out, z2_out, blk, out in records:
            f.write(struct.pack('<dddd', z1_in, z2_in, z1_out, z2_out))
            f.write(blk.astype('<f4').tobytes())
            f.write(out.astype('<f4').tobytes())

    print('wrote %s: %d frames, hop=%d sr=%d cutoff=%.2f' % (
        out_path, len(records), HOP, SAMPLE_RATE, CUTOFF_HZ))


if __name__ == '__main__':
    main()
