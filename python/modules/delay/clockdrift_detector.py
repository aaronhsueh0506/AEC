"""Clock-drift detector: recognises monotonic sub-sample drift patterns in the lag history.

``update()`` is called once per inner 64-sample block (64/sample_rate seconds), not once per outer hop.
"""
from enum import Enum

# Inner block size in samples (duplicated from echo_path_delay_estimator to avoid a circular import).
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
        # 30 s stability window in inner-block ticks (64/sample_rate seconds each).
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

        # x-1,x-2,x or x-2,x-1,x is "probable up"; the same with x-3 on d3 is "verified".
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
