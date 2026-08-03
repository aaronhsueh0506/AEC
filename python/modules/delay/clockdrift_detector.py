"""Clockdrift detector — recognises sub-sample drift patterns in the lag history.

Mirrors docs/aec3_extracts/src/aec3/clockdrift_detector.{cc,h}.

Rate note: ``update()`` is called once per INNER 64-sample block drained by
``EchoPathDelayEstimator._process_inner_block`` (echo_path_delay_estimator.py)
-- NOT once per outer hop. The inner block is a FIXED sample count
(``_AEC3_BLOCK_SIZE=64``, mirrored below), so its wall-clock duration is
``64 / sample_rate`` seconds regardless of the outer hop_size. AEC3's
7500-update stability gate spans 30 s at its native 4 ms/block cadence
(64 samples @ 16 kHz); the previous port used ``_rates.ms_to_hops(30_000)``,
which assumes an unrelated 10 ms *outer-hop* cadence -- wrong even at
sr=16000 (3000 x 4 ms = 12 s, not 30 s) and further off at other sample
rates (the inner block's duration scales with sample_rate, not hop_size).
"""
from enum import Enum

# AEC3 kBlockSize -- mirrors echo_path_delay_estimator.py's _AEC3_BLOCK_SIZE
# (duplicated rather than imported to avoid a circular import; this is an
# AEC3-verbatim constant, not expected to change independently).
_AEC3_BLOCK_SIZE = 64


class ClockdriftLevel(Enum):
    NONE = 0
    PROBABLE = 1
    VERIFIED = 2


class ClockdriftDetector:
    """Detects positive / negative clockdrift from monotonic lag drift patterns.

    State machine:
      - kNone     : no drift detected (initial; restored after stability window)
      - kProbable : two-of-three monotonic step pattern observed (d1,d2 in {±1,±2})
      - kVerified : three-of-three monotonic step pattern observed (d1,d2,d3 cover ±1,±2,±3)
    """

    def __init__(self, *, sample_rate: int = 16000) -> None:
        self._delay_history = [0, 0, 0]  # newest -> oldest
        self._level = ClockdriftLevel.NONE
        self._stability_counter = 0
        # AEC3 7500 blocks (~30 s at the native 4 ms/block cadence). Ticks
        # are inner 64-sample blocks, so the tick period is 64/sample_rate
        # seconds -- rescale by real sample_rate, not by outer hop_size.
        self._stability_reset_ticks = round(
            30.0 * sample_rate / _AEC3_BLOCK_SIZE
        )

    def level(self) -> ClockdriftLevel:
        return self._level

    def update(self, delay_estimate: int) -> None:
        if delay_estimate == self._delay_history[0]:
            self._stability_counter += 1
            if self._stability_counter > self._stability_reset_ticks:
                self._level = ClockdriftLevel.NONE
            return

        self._stability_counter = 0
        d1 = self._delay_history[0] - delay_estimate
        d2 = self._delay_history[1] - delay_estimate
        d3 = self._delay_history[2] - delay_estimate

        # AEC3 clockdrift_detector.cc:36-46 — patterns x-1,x-2,x and x-2,x-1,x
        # are "probable up"; same triplets with x-3 land on d3 too -> "verified".
        probable_up = (d1 == -1 and d2 == -2) or (d1 == -2 and d2 == -1)
        verified_up = probable_up and d3 == -3
        probable_down = (d1 == 1 and d2 == 2) or (d1 == 2 and d2 == 1)
        verified_down = probable_down and d3 == 3

        if verified_up or verified_down:
            self._level = ClockdriftLevel.VERIFIED
        elif (probable_up or probable_down) and self._level is ClockdriftLevel.NONE:
            self._level = ClockdriftLevel.PROBABLE

        self._delay_history[2] = self._delay_history[1]
        self._delay_history[1] = self._delay_history[0]
        self._delay_history[0] = delay_estimate
