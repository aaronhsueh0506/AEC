"""Render delay controller: converts the sample-domain matched-filter delay to a
hop-quantised buffer delay and emits ``EchoPathVariability`` events.

Reference implementation, not used by the orchestrator (hop quantisation loses
sub-hop precision). Rates come from the fixed constants in ``_rates.py``
(16 kHz, 10 ms hop), not from ``AecConfig``.
"""
from typing import Optional

from .._rates import HOP_SAMPLES
from .delay_types import DelayAdjustment, DelayEstimate, DelayQuality, EchoPathVariability
from .echo_path_delay_estimator import EchoPathDelayEstimator


import numpy as np


def _compute_buffer_delay_hops(
    current: Optional[DelayEstimate],
    hysteresis_limit_hops: int,
    estimated: DelayEstimate,
) -> DelayEstimate:
    """Quantise the sample delay to hops; hysteresis applies to small increases only."""
    new_delay_hops = int(estimated.delay) // HOP_SAMPLES
    if current is not None:
        current_hops = int(current.delay)
        if (
            new_delay_hops > current_hops
            and new_delay_hops <= current_hops + hysteresis_limit_hops
        ):
            new_delay_hops = current_hops
    return DelayEstimate(quality=estimated.quality, delay=new_delay_hops)


class RenderDelayController:
    """Wraps ``EchoPathDelayEstimator``; ``get_delay()`` returns ``(delay_in_hops, echo_path_variability)``.

    ``delay_in_hops`` is None until a COARSE estimate exists, then carries the buffer
    delay in hops with the aggregator's quality. ``delay_change`` is BUFFER_FLUSH on
    the first call after ``reset()``, NEW_DETECTED_DELAY when the quantised delay
    changes (after hysteresis), else NONE.
    """

    def __init__(
        self,
        *,
        hysteresis_limit_hops: int = 1,
        delay_estimator: Optional[EchoPathDelayEstimator] = None,
    ) -> None:
        self._hysteresis_limit = int(hysteresis_limit_hops)
        self._estimator = delay_estimator or EchoPathDelayEstimator()
        self._last_quality = DelayQuality.COARSE
        self._delay_in_hops: Optional[DelayEstimate] = None
        self._sample_delay: Optional[DelayEstimate] = None
        self._pending_flush = False

    @property
    def consistent_estimate_counter(self) -> int:
        return self._estimator.consistent_estimate_counter

    def reliable_delay_found(self) -> bool:
        return self._estimator.reliable_delay_found()

    def has_clockdrift(self) -> bool:
        from .clockdrift_detector import ClockdriftLevel
        return self._estimator.clockdrift_level() is not ClockdriftLevel.NONE

    def reset(self, reset_delay_confidence: bool) -> None:
        self._estimator.reset(reset_delay_confidence)
        self._delay_in_hops = None
        self._sample_delay = None
        if reset_delay_confidence:
            self._last_quality = DelayQuality.COARSE
        self._pending_flush = True

    def get_delay(
        self, render_hop: np.ndarray, capture_hop: np.ndarray
    ) -> tuple[Optional[DelayEstimate], EchoPathVariability]:
        sample_delay = self._estimator.estimate_delay(render_hop, capture_hop)
        if sample_delay is not None:
            self._sample_delay = sample_delay

        prev_hops = self._delay_in_hops.delay if self._delay_in_hops is not None else -1
        if self._sample_delay is not None:
            use_hysteresis = (
                self._last_quality is DelayQuality.REFINED
                and self._sample_delay.quality is DelayQuality.REFINED
            )
            self._delay_in_hops = _compute_buffer_delay_hops(
                current=self._delay_in_hops,
                hysteresis_limit_hops=self._hysteresis_limit if use_hysteresis else 0,
                estimated=self._sample_delay,
            )
            self._last_quality = self._delay_in_hops.quality

        new_hops = self._delay_in_hops.delay if self._delay_in_hops is not None else -1
        if self._pending_flush:
            adjustment = DelayAdjustment.BUFFER_FLUSH
            self._pending_flush = False
        elif new_hops != prev_hops and self._delay_in_hops is not None:
            adjustment = DelayAdjustment.NEW_DETECTED_DELAY
        else:
            adjustment = DelayAdjustment.NONE
        variability = EchoPathVariability(
            gain_change=False,
            delay_change=adjustment,
            clock_drift=self.has_clockdrift(),
        )
        return self._delay_in_hops, variability
