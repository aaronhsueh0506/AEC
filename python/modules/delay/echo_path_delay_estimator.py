"""Echo-path delay estimator: decimator, matched filter, lag aggregator and clock-drift detector.

Capture and render are buffered and drained in 64-sample inner blocks, so the
matched filter runs a variable number of times per outer hop. The consistent-
estimate counter ticks per inner block; its threshold (about 500 ms) and the
aggregator window derive from the effective inner-block rate.
"""
from typing import Optional

import numpy as np
from scipy.signal import sosfilt

from .clockdrift_detector import ClockdriftDetector
from .delay_types import DelayEstimate, DelayQuality
from .downsampled_ring import DownsampledRenderBuffer, Decimator, required_capacity
from .lag_aggregator import (
    DelaySelectionThresholds, MatchedFilterLagAggregator, _num_blocks_per_second,
)
from .matched_filter import MatchedFilter


_AEC3_BLOCK_SIZE = 64  # inner block size in samples
_DOWN_SAMPLING_FACTOR = 4  # only the 4x decimation path is implemented
_SUB_BLOCK_SIZE = _AEC3_BLOCK_SIZE // _DOWN_SAMPLING_FACTOR  # 16


class _Resample48:
    """48 kHz -> 16 kHz anti-alias filter and decimate-by-3.

    7th-order elliptic low-pass (4 SOS sections; passband edge 6800 Hz at 1 dB
    ripple, stopband edge 8000 Hz at 60 dB), float64, direct form II transposed.
    """

    FACTOR = 3
    _SOS = np.array([
        [5.11914823e-03, 4.05806316e-04, 5.11914823e-03, 1.0, -7.66104626e-01, 0.0],
        [1.0, 1.0, 0.0, 1.0, -1.43464531e+00, 6.95652982e-01],
        [1.0, -8.01570761e-01, 1.0, 1.0, -1.29338870e+00, 8.61962453e-01],
        [1.0, -1.01138476e+00, 1.0, 1.0, -1.23733825e+00, 9.63974476e-01],
    ], dtype=np.float64)

    def __init__(self) -> None:
        self._zi = np.zeros((4, 2), dtype=np.float64)
        self._phase = 0

    def reset(self) -> None:
        self._zi[:] = 0.0
        self._phase = 0

    def process_decimate(self, x: np.ndarray) -> np.ndarray:
        filtered, self._zi = sosfilt(self._SOS, x.astype(np.float64, copy=False), zi=self._zi)
        start = (-self._phase) % self.FACTOR
        kept = filtered[start::self.FACTOR]
        self._phase = (self._phase + x.shape[0]) % self.FACTOR
        return kept.astype(np.float32)


class EchoPathDelayEstimator:
    """Estimate the echo-path delay from per-hop capture and render frames.

    ``estimate_delay()`` returns the latest aggregated ``DelayEstimate`` produced during the hop,
    else None; delay is in raw (input-rate) samples. At 48 kHz both streams pass through
    ``_Resample48`` so the inner chain runs at 16 kHz; clock-drift and aggregator time constants
    use that effective rate.
    """

    def __init__(
        self,
        *,
        sample_rate: int = 16000,
        num_filters: int = 5,
        window_size_sub_blocks: int = 32,
        alignment_shift_sub_blocks: int = 24,
        # float-scale AEC3 default of 150 (int16 amplitude)
        excitation_limit: float = 150.0 / 32768.0,
        smoothing_fast: float = 0.7,
        smoothing_slow: float = 0.1,
        matching_filter_threshold: float = 0.3,
        delay_headroom_samples: int = 32,
        thresholds: Optional[DelaySelectionThresholds] = None,
        detect_pre_echo: bool = True,
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
            detect_pre_echo=detect_pre_echo,
        )
        # 48 kHz: anti-alias + decimate-by-3 sidechain, so the effective inner-block rate is 16000.
        self._rate_factor = 3 if sample_rate == 48000 else 1
        effective_rate = 16000 if self._rate_factor > 1 else sample_rate
        if self._rate_factor > 1:
            self._resample48_render = _Resample48()
            self._resample48_capture = _Resample48()
        else:
            self._resample48_render = None
            self._resample48_capture = None
        self._aggregator = MatchedFilterLagAggregator(
            max_filter_lag=self._matched_filter.get_max_filter_lag(),
            thresholds=thresholds,
            delay_headroom_samples=delay_headroom_samples,
            down_sampling_factor=_DOWN_SAMPLING_FACTOR,
            detect_pre_echo=detect_pre_echo,
            sample_rate=effective_rate,
        )
        self._clockdrift = ClockdriftDetector(sample_rate=effective_rate)
        self._old_aggregated_lag: Optional[DelayEstimate] = None
        self._consistent_estimate_counter = 0
        # About 500 ms of inner blocks at the effective rate.
        self._consistent_estimate_threshold = _num_blocks_per_second(effective_rate) // 2
        # Outer->inner edge buffers at the effective rate; drained in 64-sample blocks.
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
        # Full cold-start reset: matched filter, aggregator, 48 kHz sidechain, decimators,
        # ring and pending buffers. The clock-drift detector is kept (30 s horizon).
        self._reset(reset_lag_aggregator=True, reset_delay_confidence=reset_delay_confidence)
        if self._resample48_render is not None:
            self._resample48_render.reset()
            self._resample48_capture.reset()
        self._capture_decimator.reset()
        self._render_decimator.reset()
        self._render_ring.reset()
        self._capture_pending = np.zeros(0, dtype=np.float32)
        self._render_pending = np.zeros(0, dtype=np.float32)

    def estimate_delay(
        self, render_hop: np.ndarray, capture_hop: np.ndarray
    ) -> Optional[DelayEstimate]:
        if render_hop.size != capture_hop.size:
            raise ValueError("render_hop and capture_hop must be same length")
        if self._rate_factor > 1:
            render_hop = self._resample48_render.process_decimate(
                render_hop.astype(np.float32, copy=False)
            )
            capture_hop = self._resample48_capture.process_decimate(
                capture_hop.astype(np.float32, copy=False)
            )
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
        # Scale the decimated-domain delay back to input-rate samples.
        if aggregated is not None:
            aggregated = DelayEstimate(
                quality=aggregated.quality,
                delay=int(aggregated.delay) * _DOWN_SAMPLING_FACTOR * self._rate_factor,
            )
        if (
            self._old_aggregated_lag is not None
            and aggregated is not None
            and self._old_aggregated_lag.delay == aggregated.delay
        ):
            self._consistent_estimate_counter += 1
        else:
            self._consistent_estimate_counter = 0
        self._old_aggregated_lag = aggregated
        if self._consistent_estimate_counter > self._consistent_estimate_threshold:
            # Soft reset: matched filters only; aggregator and confidence are kept.
            self._reset(reset_lag_aggregator=False, reset_delay_confidence=False)
        return aggregated

    def _reset(self, reset_lag_aggregator: bool, reset_delay_confidence: bool) -> None:
        if reset_lag_aggregator:
            self._aggregator.reset(reset_delay_confidence)
        self._matched_filter.reset(full_reset=reset_lag_aggregator)
        self._old_aggregated_lag = None
        self._consistent_estimate_counter = 0
