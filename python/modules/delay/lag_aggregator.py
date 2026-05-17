"""Matched-filter lag aggregator — 250-estimate histogram voter.

Mirrors docs/aec3_extracts/src/aec3/matched_filter_lag_aggregator.{cc,h}.

Histogram window is 250 ESTIMATES, NOT a time constant. In AEC3 the
aggregator is called per-block (4 ms) → 250 estimates ≈ 1 s of memory.
In our port the aggregator is called per outer hop (10 ms) → 250
estimates ≈ 2.5 s of memory. Acceptable: this is a sliding peak
histogram, not a time-constant gate. Keep 250 verbatim (do NOT rescale).

PreEchoLagAggregator is NOT ported (matched_filter defaults
``detect_pre_echo=False`` so the pre-echo path never fires).
"""
from dataclasses import dataclass
from typing import Optional

import numpy as np

from .delay_types import DelayEstimate, DelayQuality
from .matched_filter import LagEstimate


_HISTOGRAM_WINDOW = 250  # AEC3 verbatim — see file docstring


@dataclass(frozen=True)
class DelaySelectionThresholds:
    """Mirrors EchoCanceller3Config::Delay::DelaySelectionThresholds.

    AEC3 defaults (echo_canceller3_config.h DelaySelectionThresholds):
        initial    = 5
        converged  = 20
    """

    initial: int = 5
    converged: int = 20


class _HighestPeakAggregator:
    """Ring of recent winner lags + per-lag histogram. ``candidate`` is the
    argmax bin index. Mirrors HighestPeakAggregator in the .cc file."""

    def __init__(self, max_filter_lag: int) -> None:
        self._histogram = np.zeros(max_filter_lag + 1, dtype=np.int32)
        self._ring = np.zeros(_HISTOGRAM_WINDOW, dtype=np.int32)
        self._ring_index = 0
        self._candidate = -1

    def reset(self) -> None:
        self._histogram.fill(0)
        self._ring.fill(0)
        self._ring_index = 0

    def aggregate(self, lag: int) -> None:
        # Evict oldest, insert newest, advance ring pointer.
        old_lag = int(self._ring[self._ring_index])
        self._histogram[old_lag] -= 1
        if lag < 0:
            lag = 0
        elif lag >= self._histogram.size:
            lag = self._histogram.size - 1
        self._ring[self._ring_index] = lag
        self._histogram[lag] += 1
        self._ring_index = (self._ring_index + 1) % _HISTOGRAM_WINDOW
        self._candidate = int(np.argmax(self._histogram))

    def candidate(self) -> int:
        return self._candidate

    def histogram(self) -> np.ndarray:
        return self._histogram


class MatchedFilterLagAggregator:
    """Aggregates per-call matched-filter lag estimates into a stable
    ``DelayEstimate{quality, delay}`` once the histogram crosses the
    thresholds. ``ReliableDelayFound()`` latches True once the converged
    threshold has ever been crossed; cleared only by hard reset."""

    def __init__(
        self,
        *,
        max_filter_lag: int,
        thresholds: Optional[DelaySelectionThresholds] = None,
        delay_headroom_samples: int = 32,
        down_sampling_factor: int = 4,
    ) -> None:
        self._thresholds = thresholds or DelaySelectionThresholds()
        if self._thresholds.initial > self._thresholds.converged:
            raise ValueError("thresholds.initial must be <= thresholds.converged")
        self._headroom = int(delay_headroom_samples // down_sampling_factor)
        self._highest_peak = _HighestPeakAggregator(max_filter_lag=max_filter_lag)
        self._significant_candidate_found = False

    def reset(self, hard_reset: bool) -> None:
        self._highest_peak.reset()
        if hard_reset:
            self._significant_candidate_found = False

    def reliable_delay_found(self) -> bool:
        return self._significant_candidate_found

    def delay_at_highest_peak(self) -> int:
        return self._highest_peak.candidate()

    def aggregate(
        self, lag_estimate: Optional[LagEstimate]
    ) -> Optional[DelayEstimate]:
        if lag_estimate is None:
            return None
        lag = max(0, int(lag_estimate.lag) - self._headroom)
        self._highest_peak.aggregate(lag)
        hist = self._highest_peak.histogram()
        candidate = self._highest_peak.candidate()
        hist_val = int(hist[candidate])
        # Latch ``significant_candidate_found_`` the moment the converged bar
        # is crossed (AEC3 cc:86-87).
        self._significant_candidate_found = (
            self._significant_candidate_found or hist_val > self._thresholds.converged
        )
        if hist_val > self._thresholds.converged or (
            hist_val > self._thresholds.initial and not self._significant_candidate_found
        ):
            quality = (
                DelayQuality.REFINED
                if self._significant_candidate_found
                else DelayQuality.COARSE
            )
            return DelayEstimate(quality=quality, delay=int(candidate))
        return None
