"""Top-level echo-path delay estimator. Wraps decimator + matched_filter +
lag aggregator + clockdrift detector.

Mirrors docs/aec3_extracts/src/aec3/echo_path_delay_estimator.{cc,h}.

OUTER→INNER ADAPTER (the rate-bridge): AEC3 processes capture in 64-sample
``kBlockSize`` blocks (4 ms @ an effective 16 kHz). The outer hop size
generally does not divide 64 evenly. We buffer raw capture and render here
and DRAIN in integer 64-sample chunks; per outer hop we run matched_filter
a variable number of times. The "edge-chunker" lives in
``estimate_delay()``. AEC3 sub_block / window / alignment / smoothing
constants stay verbatim — see file header comments inside
``matched_filter.py``.

Consistent-estimate counter ticks per INNER block call (AEC3-native), so
``kNumBlocksPerSecondBy2`` is derived from the effective inner-block rate
(``_num_blocks_per_second(effective_rate) // 2`` — 125 at the effective
16 kHz native/48kHz-sidechain case, rate-relative at native 8 kHz), not a
frozen 125. Same basis for the aggregator histogram window.
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


_AEC3_BLOCK_SIZE = 64       # AEC3 kBlockSize @ 16 kHz (4 ms)
_DOWN_SAMPLING_FACTOR = 4   # only 4x ds-path ported; 8x would need bandpass biquads
_SUB_BLOCK_SIZE = _AEC3_BLOCK_SIZE // _DOWN_SAMPLING_FACTOR  # 16


class _Resample48:
    """48kHz->16kHz pre-decimation anti-alias filter, decimate by FACTOR=3.

    Exact port of delay_aec3.c's DaResample48/da_biquad_process1: elliptic
    lowpass, order 7 (4 SOS sections), scipy.signal.ellipord/ellip design for
    48kHz input (passband edge 6800 Hz/1.0 dB ripple, stopband edge 8000 Hz/
    60 dB attenuation). Coefficients below are the SAME values baked into
    delay_aec3.c's DA_RESAMPLE48_B/A (there stored as float32 for the C
    port's static tables; here float64, matching this repo's fp64-spec
    convention -- the delay chain's Python<->C bit-exact parity is already
    retired by design, see the C file's own banner). scipy.signal.sosfilt's
    default per-section recurrence IS Direct-Form-II-Transposed:
        y[n] = b0*x[n] + z0
        z0'  = b1*x[n] - a1*y[n] + z1
        z1'  = b2*x[n] - a2*y[n]
    the identical recurrence da_biquad_process1() implements -- so this is a
    numerically-verifiable port, not a from-scratch reimplementation.
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
    """Estimate the echo-path delay from per-hop capture + render frames.

    ``estimate_delay()`` returns the latest aggregated ``DelayEstimate`` if
    one was produced during this hop's inner-block drain, else None. Delay
    is reported in RAW (non-downsampled) samples.

    48kHz / multi-rate status: at sample_rate==48000, render/capture are
    first passed through ``_Resample48`` (a real anti-alias filter, not a
    bare stride-pick) before reaching the inner-block matched-filter chain,
    matching delay_aec3.c's DaResample48 sidechain (2026-08-02); the inner
    chain then always runs on a genuine 16kHz-equivalent stream, exactly
    like at native 16kHz. Native 8kHz has no sidechain (rate_factor=1,
    matching the C side) -- the effective inner-block rate is native
    sample_rate there, so ClockdriftDetector/MatchedFilterLagAggregator's
    real-time constants are parametrized on ``_effective_rate`` (== 16000
    when the sidechain is active, else the true sample_rate), not the raw
    sample_rate directly. 2026-08-03: ported to match the C side after a
    parity_aec_e2e investigation found this asymmetry caused the two
    languages' delay estimators to lock onto different delays at 48kHz
    (Python: no anti-alias -> aliased matched-filter input).
    """

    def __init__(
        self,
        *,
        sample_rate: int = 16000,
        num_filters: int = 5,
        window_size_sub_blocks: int = 32,
        alignment_shift_sub_blocks: int = 24,
        # AEC3 default 150 is int16 amplitude. Float-scale equivalent
        # is 150/32768 = 4.578e-3.
        excitation_limit: float = 150.0 / 32768.0,
        smoothing_fast: float = 0.7,
        smoothing_slow: float = 0.1,
        matching_filter_threshold: float = 0.3,
        delay_headroom_samples: int = 32,
        thresholds: Optional[DelaySelectionThresholds] = None,
        detect_pre_echo: bool = True,   # AEC3 strict default
        # AEC3 default: true (echo_canceller3_config.h:73).
        # PreEchoLagAggregator may override highest-peak with
        # pre_echo_candidate when accumulated_error shows a leading peak.
        # On cases WITHOUT real pre-echo this can produce noisy delay
        # estimates; add a config toggle if it becomes load-bearing for
        # our cohort rather than diverging from AEC3.
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
        # 48kHz feeds the inner-block chain a real anti-alias-filtered,
        # decimate-by-3 stream (matches delay_aec3.c); the effective
        # inner-block rate is then always 16000, same as native 16kHz.
        # Native 8kHz has no sidechain (rate_factor=1) -- matches the C side.
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
        # AEC3 kNumBlocksPerSecondBy2 = 125 blocks (~500 ms). Counter ticks
        # per inner-block call, i.e. per effective_rate/64 blocks/sec -- NOT
        # the native sample_rate once the 48kHz sidechain changes the actual
        # inner-block feed rate to 16000.
        self._consistent_estimate_threshold = _num_blocks_per_second(effective_rate) // 2
        # Outer→inner edge buffers — accumulate raw effective-rate samples
        # (post-resample48 at 48kHz, native otherwise) until we have >= 64
        # to feed an AEC3-rate matched_filter update.
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
        # Full reset. The only caller of this public reset() is
        # LegacyDelayShim.reset()/RenderDelayController.reset() <-
        # AEC.reset()/aec_reset() (orchestrator.py / aec.c), a top-level
        # cold-start-style reset (recreates AecState/ResidualEchoEstimator/
        # SuppressionGain, clears CNG state, zeroes the OLA buffer, resets
        # every other subsystem to a pristine state). It is NOT the
        # lightweight per-echo-path-change nudge AEC3 does internally --
        # that path is the SEPARATE `_reset(reset_lag_aggregator=False, ...)`
        # call below in `_process_inner_block` (fires on the consistent-
        # estimate stability counter, every ~500 ms in normal steady state),
        # which correctly leaves the whole signal chain (sidechain,
        # decimators, ring, pending) untouched.
        #
        # Given that, this reset must clear the ENTIRE delay-estimator
        # signal chain consistently, not just the matched filter +
        # aggregator: the outer 48kHz anti-alias sidechain (_resample48_*)
        # AND the inner decimators / ring buffer / pending edge-chunker
        # buffers. Previously only the sidechain was reset here (added
        # alongside the sidechain itself) while the inner decimator biquad
        # memory, ring-buffer audio history and pending partial-block
        # samples were left stale -- a freshly-reset sidechain feeding a
        # stale inner estimator, which could let pre-reset echo-path audio
        # in the ring contaminate the matched filter's first correlation
        # windows after reset. clockdrift is deliberately left untouched
        # (pre-existing behaviour predating the sidechain, out of scope
        # here -- it tracks a 30s-horizon hardware clock-drift estimate,
        # not reset-scoped audio history).
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
        # Multiply downsampled-domain delay back to raw samples: first to
        # effective-rate (16000) samples via the inner 4x decimator, then
        # (at 48kHz) to true native-rate samples via the resample48 sidechain.
        if aggregated is not None:
            aggregated = DelayEstimate(
                quality=aggregated.quality,
                delay=int(aggregated.delay) * _DOWN_SAMPLING_FACTOR * self._rate_factor,
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
        if self._consistent_estimate_counter > self._consistent_estimate_threshold:
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
