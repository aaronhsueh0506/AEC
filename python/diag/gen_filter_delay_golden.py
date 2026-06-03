"""Generate a binary golden for the C filter_delay port (WS5 Phase 5.2).

Drives the Python FilterDelay (AecState::FilterDelay) over a deterministic
multi-frame sequence and snapshots the FULL state after every call:
``filter_delays_blocks`` (per channel), ``min_direct_path_filter_delay``, and
``external_delay_reported``. The C parity test replays the identical input
sequence and asserts exact equality at every frame, verifying the integer
state machine (branch selection + floor-division + external-delay caching +
the analyzer length-mismatch error path).

This module is PURELY INTEGER — there are no numpy dtypes to match. Inputs are
block counts and array indices. Dtypes were captured from the real
balanced/DT case: ``delay_headroom_samples`` is a python int (default 32),
``num_capture_channels`` int, ``analyzer_filter_delay_estimates_blocks`` is an
Optional[list[int]], ``external_delay`` an Optional[DelayEstimate(quality:enum,
delay:int)], ``blocks_with_proper_filter_adaptation`` int.

Layout (LE):
  int32 n_configs
  per config:
    int32 delay_headroom_samples
    int32 num_capture_channels
    int32 n_calls
    --- init snapshot (state right after __init__) ---
    int32 init_min
    int32 init_ext_reported
    int32 init_delays[num_capture_channels]
    per call:
      int32 has_analyzer            (1 -> analyzer list present, 0 -> None)
      int32 analyzer_len            (only meaningful when has_analyzer)
      int32 analyzer[analyzer_len]  (only when has_analyzer; uses analyzer_len)
      int32 ext_reported            (1 -> DelayEstimate present, 0 -> None)
      int32 ext_quality             (DelayQuality value; 0 when ext absent)
      int32 ext_delay               (samples; 0 when ext absent)
      int32 blocks_with_proper_filter_adaptation
      --- expected post-call state ---
      int32 expect_error            (1 -> Python raised ValueError, else 0)
      int32 expect_min              (min_direct_path_filter_delay; pre-call on err)
      int32 expect_ext_reported
      int32 expect_delays[num_capture_channels]

Run: python3 python/diag/gen_filter_delay_golden.py /tmp/fd_golden.bin
"""
import os
import struct
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from modules.state.filter_delay import FilterDelay  # noqa: E402
from modules.delay.delay_types import DelayEstimate, DelayQuality  # noqa: E402


def _w32(f, v):
    f.write(struct.pack('<i', int(v)))


def build_call_list(num_channels):
    """Return list of (analyzer_or_None, external_or_None, blocks).

    Exercises every branch + the analyzer length-mismatch error path +
    external-delay caching/persistence across calls. analyzer lists are sized
    to num_channels except for the deliberate mismatch entries.
    """
    de_r = DelayEstimate(quality=DelayQuality.REFINED, delay=320)
    de_c = DelayEstimate(quality=DelayQuality.COARSE, delay=160)
    full = [10 + c for c in range(num_channels)]        # valid-length analyzer
    full2 = [20 + 2 * c for c in range(num_channels)]   # second valid analyzer
    short = list(range(max(0, num_channels - 1)))       # wrong length (-1)
    long_ = list(range(num_channels + 2))               # wrong length (+2)

    calls = [
        # 1. converged + analyzer present -> copy analyzer (branch elif)
        (full, None, 300),
        # 2. unconverged but NO external cached yet -> falls to elif -> analyzer
        (full2, None, 5),
        # 3. report an external delay while converged -> caches it, analyzer copy
        (full, de_r, 250),
        # 4. unconverged + external cached -> reset all to headroom (branch if)
        (full, None, 5),
        # 5. threshold boundary: exactly 200 -> NOT unconverged -> analyzer copy
        (full2, None, FILTER_THRESHOLD),
        # 6. 199 (unconverged) + external cached -> reset to headroom
        (full, None, FILTER_THRESHOLD - 1),
        # 7. analyzer None + converged -> untouched (else)
        (None, None, 300),
        # 8. analyzer None + unconverged + external cached -> reset (branch if)
        (None, None, 0),
        # 9. replace external with a different DelayEstimate (caching update)
        (full2, de_c, 300),
        # 10. analyzer length too SHORT + converged -> ValueError (expect_error)
        (short, None, 300),
        # 11. analyzer length too LONG + converged -> ValueError (expect_error)
        (long_, None, 300),
        # 12. after the errors, a normal converged analyzer copy still works
        (full, None, 300),
    ]
    return calls


# bound the threshold reference at import (mirror the module constant)
from modules.state.filter_delay import _FILTER_ADAPTATION_THRESHOLD_HOPS as FILTER_THRESHOLD  # noqa: E402


def run_config(f, delay_headroom_samples, num_channels):
    fd = FilterDelay(
        delay_headroom_samples=delay_headroom_samples,
        num_capture_channels=num_channels,
    )

    _w32(f, delay_headroom_samples)
    _w32(f, num_channels)

    calls = build_call_list(num_channels)
    _w32(f, len(calls))

    # init snapshot
    _w32(f, fd.min_direct_path_filter_delay())
    _w32(f, 1 if fd.external_delay_reported() else 0)
    for d in fd.direct_path_filter_delays():
        _w32(f, d)

    for analyzer, external, blocks in calls:
        # serialise inputs
        if analyzer is None:
            _w32(f, 0)            # has_analyzer
            _w32(f, 0)            # analyzer_len
        else:
            _w32(f, 1)            # has_analyzer
            _w32(f, len(analyzer))
            for v in analyzer:
                _w32(f, v)
        if external is None:
            _w32(f, 0)            # ext_reported
            _w32(f, 0)            # ext_quality
            _w32(f, 0)            # ext_delay
        else:
            _w32(f, 1)
            _w32(f, external.quality.value)
            _w32(f, external.delay)
        _w32(f, blocks)

        # run + capture state (catch the length-mismatch ValueError)
        expect_error = 0
        try:
            fd.update(
                analyzer_filter_delay_estimates_blocks=analyzer,
                external_delay=external,
                blocks_with_proper_filter_adaptation=blocks,
            )
        except ValueError:
            expect_error = 1

        # expected post-call state (on error, Python aborted before any mutation
        # of filter_delays / min, but external_delay caching at the top of update
        # already ran; we snapshot whatever the live object holds)
        _w32(f, expect_error)
        _w32(f, fd.min_direct_path_filter_delay())
        _w32(f, 1 if fd.external_delay_reported() else 0)
        for d in fd.direct_path_filter_delays():
            _w32(f, d)


def main():
    out = sys.argv[1] if len(sys.argv) > 1 else '/tmp/fd_golden.bin'

    configs = [
        # (delay_headroom_samples, num_capture_channels)
        (32, 1),    # default headroom (32 // 160 = 0 blocks), single channel
        (480, 3),   # 480 // 160 = 3 blocks, multi-channel (min over 3)
        (200, 2),   # 200 // 160 = 1 block (non-exact floor div), 2 channels
    ]

    with open(out, 'wb') as f:
        f.write(struct.pack('<i', len(configs)))
        for headroom, nch in configs:
            run_config(f, headroom, nch)

    print(f"wrote {out}  ({len(configs)} configs)")


if __name__ == '__main__':
    main()
