"""Clockdrift detector — recognises sub-sample drift patterns in the lag history.

Mirrors docs/aec3_extracts/src/aec3/clockdrift_detector.{cc,h}.

Rate note (AEC3 N blocks (~Xms) -> our M hops):
  AEC3 calls ``Update`` per matched-filter aggregation call. In AEC3 the
  aggregator runs per-block (4 ms), so the 7500-update stability gate
  spans 30 s. In our port the aggregator runs per outer hop (10 ms), so
  7500 updates would be 75 s. Rescale: 30 s -> 3000 hops via
  ``ms_to_hops(30000)``.
"""
from enum import Enum

from .._rates import ms_to_hops


class ClockdriftLevel(Enum):
    NONE = 0
    PROBABLE = 1
    VERIFIED = 2


_STABILITY_RESET_HOPS = ms_to_hops(30_000)  # AEC3 7500 blocks (~30 s) -> our 3000 hops


class ClockdriftDetector:
    """Detects positive / negative clockdrift from monotonic lag drift patterns.

    State machine:
      - kNone     : no drift detected (initial; restored after stability window)
      - kProbable : two-of-three monotonic step pattern observed (d1,d2 in {±1,±2})
      - kVerified : three-of-three monotonic step pattern observed (d1,d2,d3 cover ±1,±2,±3)
    """

    def __init__(self) -> None:
        self._delay_history = [0, 0, 0]  # newest -> oldest
        self._level = ClockdriftLevel.NONE
        self._stability_counter = 0

    def level(self) -> ClockdriftLevel:
        return self._level

    def update(self, delay_estimate: int) -> None:
        if delay_estimate == self._delay_history[0]:
            self._stability_counter += 1
            if self._stability_counter > _STABILITY_RESET_HOPS:
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
