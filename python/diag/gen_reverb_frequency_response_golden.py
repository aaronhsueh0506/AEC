"""Generate a binary golden for the C reverb_frequency_response port (WS5 Phase 5.1).

Drives the Python ReverbFrequencyResponse over a deterministic float32 input
sequence (float32 (n_partitions, n_freqs) frequency_response, int
filter_delay_blocks, float linear_filter_quality, bool stationary) and writes
inputs + expected tail_response + average_decay to a raw little-endian file that
c_impl/test/historical/parity_reverb_frequency_response.c replays.

Input dtypes captured from a real balanced/DT case
(0I0XMl3M0ECO0U1N0cJvpg_doubletalk): frequency_response is float32 with shape
(n_partitions, n_freqs); filter_delay_blocks is a python int;
linear_filter_quality is a python float in [0,1] (or None).

Two configs are exercised: (A) use_conservative=True, smoothing_base=0.4275...
(wall-clock, the balanced production value), (B) use_conservative=False,
smoothing_base=0.2 (strict AEC3). Each config runs a sequence that includes
ordinary updates, a stationary skip, a None-quality skip, an out-of-range
filter_delay_blocks, a direct_energy==0 row, and filter_delay_blocks == last.

Layout (LE):
  int32 n_freqs, int32 n_partitions, int32 n_configs
  per config:
    int32 use_conservative, float64 smoothing_base, int32 n_calls
    per call:
      float32 frequency_response[n_partitions * n_freqs]   (row-major)
      int32   filter_delay_blocks
      int32   quality_is_none          (1 -> None, 0 -> use quality below)
      float64 linear_filter_quality
      int32   stationary_block
      float64 expected_average_decay   (post-update state)
      float32 expected_tail_response[n_freqs]

Run: python3 python/diag/gen_reverb_frequency_response_golden.py /tmp/rfr_golden.bin
"""
import os
import struct
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from modules.residual.reverb_frequency_response import ReverbFrequencyResponse  # noqa: E402

N_FREQS = 257
N_PART = 6


def make_freq_response(rng, n_part, n_freqs, *, zero_direct_row=None):
    """float32 (n_part, n_freqs) of non-negative |W|^2-like magnitudes."""
    fr = (rng.rand(n_part, n_freqs).astype(np.float32) * 1.0e5).astype(np.float32)
    if zero_direct_row is not None:
        fr[zero_direct_row, :] = np.float32(0.0)
    return fr


def build_call_list(rng):
    """Return list of (fr, fdb, quality_or_None, stationary)."""
    calls = []
    # ordinary updates, quality sweeping [0,1]
    for q in (0.0, 0.137, 0.5, 0.831, 1.0):
        calls.append((make_freq_response(rng, N_PART, N_FREQS), 2, q, False))
    # filter_delay_blocks == last partition (direct == tail row)
    calls.append((make_freq_response(rng, N_PART, N_FREQS), N_PART - 1, 0.6, False))
    # stationary skip (state must not change)
    calls.append((make_freq_response(rng, N_PART, N_FREQS), 2, 0.7, True))
    # None quality skip
    calls.append((make_freq_response(rng, N_PART, N_FREQS), 2, None, False))
    # out-of-range fdb (negative) -> skip
    calls.append((make_freq_response(rng, N_PART, N_FREQS), -1, 0.7, False))
    # out-of-range fdb (>= n_part) -> skip
    calls.append((make_freq_response(rng, N_PART, N_FREQS), N_PART, 0.7, False))
    # direct_energy == 0 (zeroed direct row) -> average_decay decays toward 0
    calls.append((make_freq_response(rng, N_PART, N_FREQS, zero_direct_row=2),
                  2, 0.9, False))
    # a few more ordinary updates to exercise EMA after the zero
    for q in (0.25, 0.88, 0.4):
        calls.append((make_freq_response(rng, N_PART, N_FREQS), 2, q, False))
    return calls


def main():
    out = sys.argv[1] if len(sys.argv) > 1 else '/tmp/rfr_golden.bin'
    rng = np.random.RandomState(11)

    # Wall-clock smoothing base for the balanced production config (hop=160).
    from modules import aec3_scale as _aec3_scale
    wallclock_base = float(
        _aec3_scale.per_block_ema_alpha_to_per_hop(0.2, 160, 16000)
    )

    configs = [
        # (use_conservative, use_wallclock_smoothing)
        (True, True),    # balanced production: conservative + wall-clock 0.4275
        (False, False),  # strict AEC3: plain max off + 0.2
    ]

    with open(out, 'wb') as f:
        f.write(struct.pack('<iii', N_FREQS, N_PART, len(configs)))

        for use_cons, use_wc in configs:
            r = ReverbFrequencyResponse(
                n_freqs=N_FREQS,
                use_conservative_tail_frequency_response=use_cons,
                sr=16000, hop_size=160,
                use_wallclock_smoothing=use_wc,
            )
            sb = float(r._smoothing_base)
            # sanity: wall-clock base must match the precomputed constant
            if use_wc:
                assert abs(sb - wallclock_base) < 1e-15, (sb, wallclock_base)

            calls = build_call_list(rng)
            f.write(struct.pack('<idi', 1 if use_cons else 0, sb, len(calls)))

            for fr, fdb, q, stationary in calls:
                # Inputs (match real dtypes: fr float32, fdb int, q float/None)
                fr.astype(np.float32).tofile(f)
                f.write(struct.pack('<i', int(fdb)))
                if q is None:
                    f.write(struct.pack('<i', 1))
                    f.write(struct.pack('<d', 0.0))
                else:
                    f.write(struct.pack('<i', 0))
                    f.write(struct.pack('<d', float(q)))
                f.write(struct.pack('<i', 1 if stationary else 0))

                r.update(
                    frequency_response=fr,
                    filter_delay_blocks=int(fdb),
                    linear_filter_quality=(None if q is None else float(q)),
                    stationary_block=bool(stationary),
                )

                # Expected outputs / state
                f.write(struct.pack('<d', float(r.average_decay)))
                np.asarray(r.tail_response, dtype=np.float32).tofile(f)

    print(f"wrote {out}  ({N_FREQS} freqs, {N_PART} partitions, "
          f"{len(configs)} configs)")


if __name__ == '__main__':
    main()
