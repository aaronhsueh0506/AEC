"""Matched-filter lag aggregator — 250-estimate histogram voter.

Mirrors docs/aec3_extracts/src/aec3/matched_filter_lag_aggregator.{cc,h}.

Histogram window is kNumBlocksPerSecond ESTIMATES (~1 s of memory), NOT a
fixed time constant -- previously the bare literal 250 (correct only at
sr=16000, since kNumBlocksPerSecond = sample_rate/64). The aggregator is
called per inner 64-raw-sample block via the EchoPathDelayEstimator
outer→inner adapter, so the window is now computed from the real
sample_rate (see _num_blocks_per_second) instead of assumed to be 16 kHz.

PreEchoLagAggregator ported from AEC3
matched_filter_lag_aggregator.cc:130-189. When enabled (``detect_pre_echo=True``
on the matched filter AND constructor flag ``detect_pre_echo=True``), the
aggregator runs a SECOND histogram on the pre-echo lag (earlier reflection
candidate). For the first 2 seconds it scans the histogram in
``kMatchedFilterWindowSizeSubBlocks`` (=32) -sized windows applying a
``0.7×`` penalty per later window — biasing the candidate toward the
EARLIEST high-confidence lag. After 2 s, the candidate is the simple argmax.
The reported delay in ``Aggregate`` switches from highest-peak to pre-echo
candidate when this aggregator is active.
"""
from dataclasses import dataclass
from typing import Optional

import numpy as np

from .delay_types import DelayEstimate, DelayQuality
from .matched_filter import LagEstimate


_K_BLOCK_SIZE_LOG2 = 6   # AEC3 kBlockSizeLog2 (64 = 1<<6)
_K_MATCHED_FILTER_WINDOW_SUB_BLOCKS = 32  # AEC3 kMatchedFilterWindowSizeSubBlocks
_PRE_ECHO_HISTOGRAM_DATA_NOT_UPDATED = -1


def _num_blocks_per_second(sample_rate: int) -> int:
    """AEC3 kNumBlocksPerSecond = sample_rate / kBlockSize(=64), i.e. how many
    64-raw-sample inner blocks this class is updated on per wall-clock
    second. Was the bare literal 250 (== 16000/64), correct only at
    sr=16000 -- e.g. 750 at 48000, not 250. aggregate() is called once per
    EchoPathDelayEstimator inner block (_AEC3_BLOCK_SIZE=64 raw samples),
    so this -- not hop_size -- is the right basis, matching how
    ClockdriftDetector's own stability counter was fixed."""
    return max(1, round(sample_rate / 64.0))


def _get_down_sampling_block_size_log2(down_sampling_factor: int) -> int:
    """Mirrors GetDownSamplingBlockSizeLog2 (matched_filter_lag_aggregator.cc:31-41).
    Returns kBlockSizeLog2 - log2(down_sampling_factor), clamped at 0.
    For down_sampling_factor=4 → returns 6-2 = 4 (so each histogram bin spans
    16 downsampled samples = 64 raw samples = 4 ms @ 16 kHz)."""
    ds_log2 = 0
    f = down_sampling_factor >> 1
    while f > 0:
        ds_log2 += 1
        f >>= 1
    return max(0, _K_BLOCK_SIZE_LOG2 - ds_log2)


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
        # Evict oldest, insert newest, advance ring pointer.
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
    """Mirrors PreEchoLagAggregator in
    docs/aec3_extracts/src/aec3/matched_filter_lag_aggregator.cc:130-189.

    A separate histogram keyed by ``pre_echo_lag >> block_size_log2``.
    Output ``pre_echo_candidate`` is in DOWNSAMPLED samples (i.e. same units
    as the highest-peak candidate), produced by ``candidate_block << block_size_log2``.

    For the first ``kNumBlocksPerSecond * 2`` (=500) updates, scan the
    histogram in ``kMatchedFilterWindowSizeSubBlocks`` (=32) -sized windows
    and apply a 0.7× penalty per later window, biasing the candidate toward
    the earliest high-confidence lag. After that, simple argmax.
    """

    def __init__(self, max_filter_lag: int, down_sampling_factor: int,
                 sample_rate: int = 16000) -> None:
        self._block_size_log2 = _get_down_sampling_block_size_log2(down_sampling_factor)
        # histogram_size = ((max_filter_lag+1) * ds_factor) >> kBlockSizeLog2
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
        # Evict oldest entry (skip if still sentinel-init).
        old = int(self._ring[self._ring_index])
        if old != _PRE_ECHO_HISTOGRAM_DATA_NOT_UPDATED:
            self._histogram[old] -= 1
        self._ring[self._ring_index] = pre_echo_block_size
        self._histogram[pre_echo_block_size] += 1
        self._ring_index = (self._ring_index + 1) % self._window

        # First 2 seconds (2 * kNumBlocksPerSecond) → earliest-lag bias.
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
    """Aggregates per-call matched-filter lag estimates into a stable
    ``DelayEstimate{quality, delay}`` once the histogram crosses the
    thresholds. ``ReliableDelayFound()`` latches True once the converged
    threshold has ever been crossed; cleared only by hard reset.

    ``detect_pre_echo`` enables the PreEchoLagAggregator
    (matched_filter_lag_aggregator.cc:73-99). When active, the reported
    delay switches from highest-peak candidate to pre-echo candidate.
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
        # Feed pre-echo aggregator BEFORE primary (AEC3 cc:75-79).
        if self._pre_echo is not None:
            pre_lag = max(0, int(lag_estimate.pre_echo_lag) - self._headroom)
            self._pre_echo.aggregate(pre_lag)
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
            reported_delay = (
                self._pre_echo.candidate() if self._pre_echo is not None else candidate
            )
            return DelayEstimate(quality=quality, delay=int(reported_delay))
        return None
