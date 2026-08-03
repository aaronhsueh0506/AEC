"""Render delay controller — converts sample-domain matched-filter delay into
hop-aligned buffer delay and emits ``EchoPathVariability`` events.

Mirrors docs/aec3_extracts/src/aec3/render_delay_controller.{cc,h} +
collapses the ``EchoPathVariability`` emission that AEC3 does inside
``render_delay_buffer::AlignFromDelay``.

NOT the production delay path (LEGACY/reference port, no live callers).
``orchestrator.py`` uses ``LegacyDelayShim``/``EchoPathDelayEstimator``
instead -- see ``legacy_compat.py``'s module docstring: hop-quantising the
delay here loses the sub-hop sample precision the orchestrator's ring buffer
needs. This module is kept as an AEC3-structural reference/for future
migrations, has no dedicated test coverage, and its rate constants
(``.._rates.HOP_SAMPLES`` etc.) are a fixed legacy 16 kHz/10 ms reference,
NOT the live per-instance grid -- they do not track ``AecConfig``'s current
hop size (which defaults to 8 ms at 16 kHz as of the 2026-08-01 grid change).
Do not wire this into a live per-rate/per-grid path without first making
``_rates.py`` take the actual sample_rate/hop_size instead of module-level
constants, and adding real test coverage.

Rate notes (against the fixed legacy reference above, not the live grid):
  - AEC3 reports buffer delay in ``kBlockSize=64``-sample blocks
    (``>> kBlockSizeLog2``). This port reports buffer delay in the legacy
    10 ms hops (``>> log2(HOP_SAMPLES)`` analogue, via ``// HOP_SAMPLES``).
  - ``hysteresis_limit_blocks`` (AEC3 default 1 block = 4 ms) maps to 1 of
    those legacy hops (10 ms) — the smallest hysteresis expressible there.
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
    """Mirrors ComputeBufferDelay (render_delay_controller.cc:64-81) — quantises
    sample delay to OUR hop units and applies hysteresis when both qualities
    are REFINED."""
    new_delay_hops = int(estimated.delay) // HOP_SAMPLES
    if current is not None:
        current_hops = int(current.delay)
        # Only shrink hysteresis upward (avoid thrash on small increases).
        if (
            new_delay_hops > current_hops
            and new_delay_hops <= current_hops + hysteresis_limit_hops
        ):
            new_delay_hops = current_hops
    return DelayEstimate(quality=estimated.quality, delay=new_delay_hops)


class RenderDelayController:
    """Wraps EchoPathDelayEstimator. Per ``get_delay()`` call returns
    ``(delay_estimate_in_hops, echo_path_variability)``.

    ``delay_estimate_in_hops`` is None until enough estimates have crossed
    the COARSE threshold. After that, it carries the buffer delay in OUR
    hop units. The quality field tracks whether the underlying matched-
    filter aggregator is in COARSE or REFINED state.

    ``echo_path_variability.delay_change`` is:
      - ``BUFFER_FLUSH``  on the call immediately after ``reset()``;
      - ``NEW_DETECTED_DELAY`` when the hop-quantised buffer delay
        crosses an integer boundary (after hysteresis);
      - ``NONE`` otherwise.
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
