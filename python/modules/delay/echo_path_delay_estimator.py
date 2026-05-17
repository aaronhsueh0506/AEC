"""Top-level echo-path delay estimator. Wraps decimator + matched_filter +
lag aggregator + clockdrift detector.

Mirrors docs/aec3_extracts/src/aec3/echo_path_delay_estimator.{cc,h}.

OUTER→INNER ADAPTER (the rate-bridge): AEC3 processes capture in 64-sample
``kBlockSize`` blocks (4 ms @ 16 kHz). Our pipeline calls in hops of 160
samples (10 ms). 160 / 64 = 2.5 — not integer. We buffer raw capture and
render here and DRAIN in integer 64-sample chunks; per outer hop we run
matched_filter ~2–3 times. The "edge-chunker" lives in
``estimate_delay()``. AEC3 sub_block / window / alignment / smoothing
constants stay verbatim — see file header comments inside
``matched_filter.py``.

Consistent-estimate counter ticks per INNER block call (AEC3-native), so
``kNumBlocksPerSecondBy2 = 125`` (~500 ms) stays at 125 verbatim. Same
for the aggregator histogram window (250 estimates ≈ 1 s).
"""
from typing import Optional

import numpy as np

from .clockdrift_detector import ClockdriftDetector
from .delay_types import DelayEstimate, DelayQuality
from .downsampled_ring import DownsampledRenderBuffer, Decimator, required_capacity
from .lag_aggregator import DelaySelectionThresholds, MatchedFilterLagAggregator
from .matched_filter import MatchedFilter


_AEC3_BLOCK_SIZE = 64       # AEC3 kBlockSize @ 16 kHz (4 ms)
_DOWN_SAMPLING_FACTOR = 4   # only 4x ds-path ported; 8x would need bandpass biquads
_SUB_BLOCK_SIZE = _AEC3_BLOCK_SIZE // _DOWN_SAMPLING_FACTOR  # 16

# AEC3 kNumBlocksPerSecondBy2 = 125 blocks (~500 ms). Counter ticks per
# inner-block call (AEC3 rate), so no rate-rescale needed.
_CONSISTENT_ESTIMATE_THRESHOLD = 125


class EchoPathDelayEstimator:
    """Estimate the echo-path delay from per-hop capture + render frames.

    ``estimate_delay()`` returns the latest aggregated ``DelayEstimate`` if
    one was produced during this hop's inner-block drain, else None. Delay
    is reported in RAW (non-downsampled) samples.
    """

    def __init__(
        self,
        *,
        num_filters: int = 5,
        window_size_sub_blocks: int = 32,
        alignment_shift_sub_blocks: int = 24,
        excitation_limit: float = 150.0,
        smoothing_fast: float = 0.7,
        smoothing_slow: float = 0.1,
        matching_filter_threshold: float = 0.3,
        delay_headroom_samples: int = 32,
        thresholds: Optional[DelaySelectionThresholds] = None,
    ) -> None:
        self._capture_decimator = Decimator(_DOWN_SAMPLING_FACTOR)
        self._render_decimator = Decimator(_DOWN_SAMPLING_FACTOR)
        ring_capacity = required_capacity(
            num_filters=num_filters,
            window_size_sub_blocks=window_size_sub_blocks,
            alignment_shift_sub_blocks=alignment_shift_sub_blocks,
            sub_block_size=_SUB_BLOCK_SIZE,
        )
        self._render_ring = DownsampledRenderBuffer(capacity=ring_capacity)
        self._matched_filter = MatchedFilter(
            sub_block_size=_SUB_BLOCK_SIZE,
            window_size_sub_blocks=window_size_sub_blocks,
            num_filters=num_filters,
            alignment_shift_sub_blocks=alignment_shift_sub_blocks,
            excitation_limit=excitation_limit,
            smoothing_fast=smoothing_fast,
            smoothing_slow=smoothing_slow,
            matching_filter_threshold=matching_filter_threshold,
            detect_pre_echo=False,
        )
        self._aggregator = MatchedFilterLagAggregator(
            max_filter_lag=self._matched_filter.get_max_filter_lag(),
            thresholds=thresholds,
            delay_headroom_samples=delay_headroom_samples,
            down_sampling_factor=_DOWN_SAMPLING_FACTOR,
        )
        self._clockdrift = ClockdriftDetector()
        self._old_aggregated_lag: Optional[DelayEstimate] = None
        self._consistent_estimate_counter = 0
        # Outer→inner edge buffers — accumulate raw 16 kHz samples until we
        # have ≥ 64 to feed an AEC3-rate matched_filter update.
        self._capture_pending = np.zeros(0, dtype=np.float32)
        self._render_pending = np.zeros(0, dtype=np.float32)

    # --------------------------------------------------------------- public API

    @property
    def consistent_estimate_counter(self) -> int:
        return self._consistent_estimate_counter

    def clockdrift_level(self):
        return self._clockdrift.level()

    def reliable_delay_found(self) -> bool:
        return self._aggregator.reliable_delay_found()

    def reset(self, reset_delay_confidence: bool) -> None:
        self._reset(reset_lag_aggregator=True, reset_delay_confidence=reset_delay_confidence)

    def estimate_delay(
        self, render_hop: np.ndarray, capture_hop: np.ndarray
    ) -> Optional[DelayEstimate]:
        if render_hop.size != capture_hop.size:
            raise ValueError("render_hop and capture_hop must be same length")
        self._render_pending = np.concatenate(
            (self._render_pending, render_hop.astype(np.float32, copy=False))
        )
        self._capture_pending = np.concatenate(
            (self._capture_pending, capture_hop.astype(np.float32, copy=False))
        )
        latest: Optional[DelayEstimate] = None
        while self._render_pending.size >= _AEC3_BLOCK_SIZE:
            r_block = self._render_pending[:_AEC3_BLOCK_SIZE]
            c_block = self._capture_pending[:_AEC3_BLOCK_SIZE]
            self._render_pending = self._render_pending[_AEC3_BLOCK_SIZE:]
            self._capture_pending = self._capture_pending[_AEC3_BLOCK_SIZE:]
            latest = self._process_inner_block(r_block, c_block) or latest
        return latest

    # ----------------------------------------------------------------- helpers

    def _process_inner_block(
        self, render_block: np.ndarray, capture_block: np.ndarray
    ) -> Optional[DelayEstimate]:
        render_sub = self._render_decimator.decimate(render_block)
        capture_sub = self._capture_decimator.decimate(capture_block)
        self._render_ring.push_sub_block(render_sub)
        self._matched_filter.update(
            self._render_ring, capture_sub, self._aggregator.reliable_delay_found()
        )
        aggregated = self._aggregator.aggregate(
            self._matched_filter.get_best_lag_estimate()
        )
        if aggregated is not None and aggregated.quality is DelayQuality.REFINED:
            self._clockdrift.update(self._aggregator.delay_at_highest_peak())
        # Multiply downsampled-domain delay back to raw samples.
        if aggregated is not None:
            aggregated = DelayEstimate(
                quality=aggregated.quality,
                delay=int(aggregated.delay) * _DOWN_SAMPLING_FACTOR,
            )
        # Stability counter (AEC3-native per-block tick).
        if (
            self._old_aggregated_lag is not None
            and aggregated is not None
            and self._old_aggregated_lag.delay == aggregated.delay
        ):
            self._consistent_estimate_counter += 1
        else:
            self._consistent_estimate_counter = 0
        self._old_aggregated_lag = aggregated
        if self._consistent_estimate_counter > _CONSISTENT_ESTIMATE_THRESHOLD:
            # AEC3 echo_path_delay_estimator.cc:117 — partial reset
            # (filters + aggregator soft) without losing confidence.
            self._reset(reset_lag_aggregator=False, reset_delay_confidence=False)
        return aggregated

    def _reset(self, reset_lag_aggregator: bool, reset_delay_confidence: bool) -> None:
        if reset_lag_aggregator:
            self._aggregator.reset(reset_delay_confidence)
        self._matched_filter.reset(full_reset=reset_lag_aggregator)
        self._old_aggregated_lag = None
        self._consistent_estimate_counter = 0
