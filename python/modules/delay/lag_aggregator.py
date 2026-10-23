"""Matched-filter lag aggregator: histogram voter over the last second of lag estimates.

The window is ``sample_rate / 64`` estimates (one per inner 64-sample block).
With ``detect_pre_echo`` a second histogram tracks an earlier-reflection candidate;
it is computed for diagnostics only. The reported delay is always the dominant
(highest-peak) candidate, so a long-ahead onset cannot place the main echo outside
the short filter.
"""
from dataclasses import dataclass
from typing import Optional

import numpy as np

from .delay_types import DelayEstimate, DelayQuality
from .matched_filter import LagEstimate


_K_BLOCK_SIZE_LOG2 = 6  # log2 of the 64-sample block
_K_MATCHED_FILTER_WINDOW_SUB_BLOCKS = 32
_PRE_ECHO_HISTOGRAM_DATA_NOT_UPDATED = -1


def _num_blocks_per_second(sample_rate: int) -> int:
    """Inner 64-sample blocks per second at ``sample_rate``."""
    return max(1, round(sample_rate / 64.0))


def _get_down_sampling_block_size_log2(down_sampling_factor: int) -> int:
    """block_size_log2 - log2(down_sampling_factor), clamped at 0 (factor 4 gives 4)."""
    ds_log2 = 0
    f = down_sampling_factor >> 1
    while f > 0:
        ds_log2 += 1
        f >>= 1
    return max(0, _K_BLOCK_SIZE_LOG2 - ds_log2)


@dataclass(frozen=True)
class DelaySelectionThresholds:
    """Histogram-count thresholds for COARSE (``initial``) and REFINED (``converged``) estimates."""

    initial: int = 5
    converged: int = 20


class _HighestPeakAggregator:
    """Ring of recent winner lags with a per-lag histogram; ``candidate`` is the argmax bin."""

    def __init__(self, max_filter_lag: int, sample_rate: int = 16000) -> None:
        self._histogram = np.zeros(max_filter_lag + 1, dtype=np.int32)
        self._window = _num_blocks_per_second(sample_rate)
        self._ring = np.zeros(self._window, dtype=np.int32)
        self._ring_index = 0
        self._candidate = -1

    def reset(self) -> None:
        self._histogram.fill(0)
        self._ring.fill(0)
        self._ring_index = 0

    def aggregate(self, lag: int) -> None:
        old_lag = int(self._ring[self._ring_index])
        self._histogram[old_lag] -= 1
        if lag < 0:
            lag = 0
        elif lag >= self._histogram.size:
            lag = self._histogram.size - 1
        self._ring[self._ring_index] = lag
        self._histogram[lag] += 1
        self._ring_index = (self._ring_index + 1) % self._window
        self._candidate = int(np.argmax(self._histogram))

    def candidate(self) -> int:
        return self._candidate

    def histogram(self) -> np.ndarray:
        return self._histogram


class _PreEchoLagAggregator:
    """Pre-echo histogram keyed by ``pre_echo_lag >> block_size_log2``.

    The candidate is in downsampled samples. For the first two seconds of updates the
    histogram is scanned in 32-bin windows with a 0.7x penalty per later window,
    favouring the earliest strong lag; afterwards it is a plain argmax.
    """

    def __init__(self, max_filter_lag: int, down_sampling_factor: int,
                 sample_rate: int = 16000) -> None:
        self._block_size_log2 = _get_down_sampling_block_size_log2(down_sampling_factor)
        size = ((max_filter_lag + 1) * down_sampling_factor) >> _K_BLOCK_SIZE_LOG2
        if size <= 0:
            size = 1
        self._histogram = np.zeros(size, dtype=np.int32)
        self._num_blocks_per_second = _num_blocks_per_second(sample_rate)
        self._window = self._num_blocks_per_second
        self._ring = np.full(self._window, _PRE_ECHO_HISTOGRAM_DATA_NOT_UPDATED,
                             dtype=np.int32)
        self._ring_index = 0
        self._number_updates = 0
        self._pre_echo_candidate = 0

    def reset(self) -> None:
        self._histogram.fill(0)
        self._ring.fill(_PRE_ECHO_HISTOGRAM_DATA_NOT_UPDATED)
        self._ring_index = 0
        self._number_updates = 0
        self._pre_echo_candidate = 0

    def aggregate(self, pre_echo_lag: int) -> None:
        pre_echo_block_size = int(pre_echo_lag) >> self._block_size_log2
        pre_echo_block_size = max(0, min(pre_echo_block_size, self._histogram.size - 1))
        old = int(self._ring[self._ring_index])
        if old != _PRE_ECHO_HISTOGRAM_DATA_NOT_UPDATED:
            self._histogram[old] -= 1
        self._ring[self._ring_index] = pre_echo_block_size
        self._histogram[pre_echo_block_size] += 1
        self._ring_index = (self._ring_index + 1) % self._window

        if self._number_updates < self._num_blocks_per_second * 2:
            self._number_updates += 1
            window = _K_MATCHED_FILTER_WINDOW_SUB_BLOCKS
            penalty = 1.0
            best_value = -1.0
            best_idx = 0
            n = self._histogram.size
            i = 0
            while n - i >= window:
                end = i + window
                seg = self._histogram[i:end]
                local_max_offset = int(np.argmax(seg))
                local_max_value = float(seg[local_max_offset]) * penalty
                if local_max_value > best_value:
                    best_value = local_max_value
                    best_idx = i + local_max_offset
                penalty *= 0.7
                i = end
            self._pre_echo_candidate = best_idx << self._block_size_log2
        else:
            best_idx = int(np.argmax(self._histogram))
            self._pre_echo_candidate = best_idx << self._block_size_log2

    def candidate(self) -> int:
        return self._pre_echo_candidate


class MatchedFilterLagAggregator:
    """Aggregates matched-filter lag estimates into a ``DelayEstimate{quality, delay}``
    once the histogram crosses the thresholds. ``reliable_delay_found()`` latches True
    when the converged threshold is crossed and is cleared only by a hard reset.
    ``detect_pre_echo`` enables the diagnostic pre-echo histogram; it never changes
    the reported delay.
    """

    def __init__(
        self,
        *,
        max_filter_lag: int,
        thresholds: Optional[DelaySelectionThresholds] = None,
        delay_headroom_samples: int = 32,
        down_sampling_factor: int = 4,
        detect_pre_echo: bool = False,
        sample_rate: int = 16000,
    ) -> None:
        self._thresholds = thresholds or DelaySelectionThresholds()
        if self._thresholds.initial > self._thresholds.converged:
            raise ValueError("thresholds.initial must be <= thresholds.converged")
        self._headroom = int(delay_headroom_samples // down_sampling_factor)
        self._highest_peak = _HighestPeakAggregator(
            max_filter_lag=max_filter_lag, sample_rate=sample_rate
        )
        self._pre_echo: Optional[_PreEchoLagAggregator] = (
            _PreEchoLagAggregator(max_filter_lag, down_sampling_factor, sample_rate=sample_rate)
            if detect_pre_echo else None
        )
        self._significant_candidate_found = False

    def reset(self, hard_reset: bool) -> None:
        self._highest_peak.reset()
        if self._pre_echo is not None:
            self._pre_echo.reset()
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
        if self._pre_echo is not None:
            pre_lag = max(0, int(lag_estimate.pre_echo_lag) - self._headroom)
            self._pre_echo.aggregate(pre_lag)
        lag = max(0, int(lag_estimate.lag) - self._headroom)
        self._highest_peak.aggregate(lag)
        hist = self._highest_peak.histogram()
        candidate = self._highest_peak.candidate()
        hist_val = int(hist[candidate])
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
            # The dominant peak is always reported: ``quality`` derives from its histogram,
            # and the pre-echo candidate could lie outside the filter span.
            return DelayEstimate(quality=quality, delay=int(candidate))
        return None
