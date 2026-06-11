"""Matched filter — multi-hypothesis NLMS cross-correlator over a downsampled ring.

Mirrors docs/aec3_extracts/src/aec3/matched_filter.{cc,h} (scalar core only;
NEON / SSE2 / AVX2 variants are not ported — numpy vectorises the inner
tap loop).

Rate / coverage notes:
  - All sizes are in DOWNSAMPLED sub-block units. ``sub_block_size`` (16
    samples = 1 ms at 16 kHz / down-sample 4) is AEC3-native; do NOT
    rescale.
  - ``window_size_sub_blocks`` (32), ``alignment_shift_sub_blocks`` (24)
    and ``num_filters`` (5) are AEC3 defaults — total coverage =
    5 * 24 ms + 32 ms = 152 ms. Match coverage in MS, not in
    "N blocks at our 10 ms hop" (user lock 2026-05-17).
  - ``smoothing_fast`` / ``smoothing_slow`` are per-update constants
    inside the sub-block loop. The faithful sub-block adapter (see
    ``echo_path_delay_estimator.py``) drives the update at AEC3-native
    1 ms rate, so these values DO NOT rescale.
"""
from dataclasses import dataclass
from typing import Optional

import numpy as np

from .downsampled_ring import DownsampledRenderBuffer


@dataclass(frozen=True)
class LagEstimate:
    lag: int          # downsampled samples; multiply by down_sampling_factor to get raw samples
    pre_echo_lag: int


# AEC3 matched_filter.cc:43 — subsample rate for accumulated error.
_ACCUMULATED_ERROR_SUBSAMPLE_RATE = 4
_PRE_ECHO_UPDATES_TO_REPORT = 50
# AEC3 saturation guard is 32000 in int16 amplitude scale. We operate
# on float[-1, 1], so the equivalent threshold is 32000/32768.
_SATURATION_LIMIT = 32000.0 / 32768.0   # 0.9766


def max_square_peak_index(h: np.ndarray) -> int:
    """Index of the strongest squared tap. Mirrors MaxSquarePeakIndex
    (matched_filter.cc:557-590). Returns 0 on |h| < 2."""
    if h.size < 2:
        return 0
    sq = h * h
    return int(np.argmax(sq))


def matched_filter_core(
    *,
    render_buffer: DownsampledRenderBuffer,
    alignment_shift_back: int,
    x2_sum_threshold: float,
    smoothing: float,
    y: np.ndarray,
    h: np.ndarray,
    instantaneous_error_out: Optional[np.ndarray] = None,
) -> tuple[bool, float]:
    """NLMS matched-filter core operating on a forward-stored ring.

    Semantically equivalent to MatchedFilterCore (matched_filter.cc:496-555),
    rewritten to use ``DownsampledRenderBuffer.gather_back`` instead of
    AEC3's reverse-stored circular access.

    For each sub-block sample ``i`` (oldest→newest in ``y``):
      - Gather ``h.size`` render samples in time-reverse order starting
        ``alignment_shift_back + (sub_block_size - 1 - i)`` samples
        before the newest render sample.
      - Compute ``e = y[i] - h·x_window``, accumulate ``error_sum``.
      - NLMS update ``h += smoothing * e / x2_sum * x_window`` when
        ``x2_sum > threshold`` and capture isn't saturated.

    If ``instantaneous_error_out`` is provided (length filter_size /
    _ACCUMULATED_ERROR_SUBSAMPLE_RATE), per-sample e² are accumulated
    into subsampled bins for pre-echo detection.
    """
    h_size = h.size
    error_sum = 0.0
    filters_updated = False
    sub_block_size = int(y.size)
    if instantaneous_error_out is not None:
        instantaneous_error_out.fill(0.0)
    for i in range(sub_block_size):
        start_back = alignment_shift_back + (sub_block_size - 1 - i)
        x_window = render_buffer.gather_back(start_back, h_size)
        x2_sum = float(np.dot(x_window, x_window))
        s = float(np.dot(h, x_window))
        e = float(y[i]) - s
        saturation = (
            float(y[i]) >= _SATURATION_LIMIT or float(y[i]) <= -_SATURATION_LIMIT
        )
        e2 = e * e
        error_sum += e2
        if instantaneous_error_out is not None:
            bin_idx = i // _ACCUMULATED_ERROR_SUBSAMPLE_RATE
            if bin_idx < instantaneous_error_out.size:
                instantaneous_error_out[bin_idx] += e2
        if x2_sum > x2_sum_threshold and not saturation:
            alpha = smoothing * e / x2_sum
            h += alpha * x_window
            filters_updated = True
    return filters_updated, error_sum


def _update_accumulated_error(
    instantaneous: np.ndarray, accumulated: np.ndarray, one_over_anchor: float
) -> None:
    """Mirrors UpdateAccumulatedError (matched_filter.cc:45-60)."""
    SMOOTH_INC = 0.015
    norm = instantaneous * one_over_anchor
    mask = norm < accumulated
    accumulated[mask] = norm[mask]
    accumulated[~mask] += SMOOTH_INC * (norm[~mask] - accumulated[~mask])


def _compute_pre_echo_lag(
    accumulated_error: np.ndarray, lag: int, alignment_shift_winner: int
) -> int:
    """Mirrors ComputePreEchoLag (matched_filter.cc:62-78)."""
    PRE_ECHO_THRESHOLD = 0.5
    if lag < alignment_shift_winner:
        return lag
    pre_echo_lag = lag - alignment_shift_winner
    maximum = min(
        pre_echo_lag // _ACCUMULATED_ERROR_SUBSAMPLE_RATE, accumulated_error.size
    )
    for k in range(int(maximum) - 1, -1, -1):
        if accumulated_error[k] > PRE_ECHO_THRESHOLD:
            break
        pre_echo_lag = (k + 1) * _ACCUMULATED_ERROR_SUBSAMPLE_RATE - 1
    return pre_echo_lag + alignment_shift_winner


class MatchedFilter:
    """Bank of ``num_filters`` cross-correlation filters covering staggered lag
    windows. ``update()`` is called once per sub-block (default 16 samples at
    AEC3 1 ms). ``get_best_lag_estimate()`` returns the winning lag (in
    downsampled samples) or None.

    Lag computed as ``MaxSquarePeakIndex(h_n) + n * filter_intra_lag_shift``.
    The winner is the filter with the smallest ``error_sum`` that is
    ``reliable`` (peak in [3, size-10], error_sum < threshold * anchor).
    Ties between adjacent filters on the same lag are broken toward the
    earlier filter (preserves pre-echo discovery path).
    """

    def __init__(
        self,
        *,
        sub_block_size: int = 16,
        window_size_sub_blocks: int = 32,
        num_filters: int = 5,
        alignment_shift_sub_blocks: int = 24,
        excitation_limit: float = 150.0 / 32768.0,  # float[-1,1] amplitude scale
        smoothing_fast: float = 0.7,
        smoothing_slow: float = 0.1,
        matching_filter_threshold: float = 0.3,
        detect_pre_echo: bool = False,
    ) -> None:
        if sub_block_size % 4 != 0:
            raise ValueError("sub_block_size must be a multiple of 4")
        self.sub_block_size = sub_block_size
        self._filter_intra_lag_shift = alignment_shift_sub_blocks * sub_block_size
        self._filter_size = window_size_sub_blocks * sub_block_size
        self._num_filters = num_filters
        self._filters = [
            np.zeros(self._filter_size, dtype=np.float32) for _ in range(num_filters)
        ]
        self._excitation_limit = float(excitation_limit)
        self._smoothing_fast = float(smoothing_fast)
        self._smoothing_slow = float(smoothing_slow)
        self._matching_filter_threshold = float(matching_filter_threshold)
        self._detect_pre_echo = bool(detect_pre_echo)
        self._reported_lag: Optional[LagEstimate] = None
        self._winner_lag: Optional[int] = None
        self._last_detected_best_lag_filter = -1
        self._number_pre_echo_updates = 0
        if self._detect_pre_echo:
            acc_size = self._filter_size // _ACCUMULATED_ERROR_SUBSAMPLE_RATE
            self._accumulated_error = [
                np.ones(acc_size, dtype=np.float32) for _ in range(num_filters)
            ]
            self._instantaneous_error = np.zeros(acc_size, dtype=np.float32)

    # ------------------------------------------------------------------ public

    def get_best_lag_estimate(self) -> Optional[LagEstimate]:
        return self._reported_lag

    def get_max_filter_lag(self) -> int:
        """Deepest reachable lag in DOWNSAMPLED samples."""
        return self._num_filters * self._filter_intra_lag_shift + self._filter_size

    def filter_intra_lag_shift(self) -> int:
        return self._filter_intra_lag_shift

    def filter_size(self) -> int:
        return self._filter_size

    def reset(self, full_reset: bool) -> None:
        for f in self._filters:
            f.fill(0.0)
        self._winner_lag = None
        self._reported_lag = None
        if full_reset and self._detect_pre_echo:
            for a in self._accumulated_error:
                a.fill(1.0)
            self._number_pre_echo_updates = 0

    def update(
        self,
        render_buffer: DownsampledRenderBuffer,
        capture: np.ndarray,
        use_slow_smoothing: bool,
    ) -> None:
        if capture.size != self.sub_block_size:
            raise ValueError(
                f"capture length {capture.size} != sub_block_size {self.sub_block_size}"
            )
        smoothing = self._smoothing_slow if use_slow_smoothing else self._smoothing_fast
        x2_sum_threshold = self._filter_size * self._excitation_limit * self._excitation_limit
        error_sum_anchor = float(np.dot(capture, capture))

        winner_error_sum = error_sum_anchor
        self._winner_lag = None
        self._reported_lag = None
        winner_index = -1
        alignment_shift = 0
        previous_lag_estimate: Optional[int] = None

        for n in range(self._num_filters):
            # Pass instantaneous_error_out for pre-echo tracking when enabled.
            _ie_out = (self._instantaneous_error
                       if self._detect_pre_echo else None)
            filters_updated, error_sum = matched_filter_core(
                render_buffer=render_buffer,
                alignment_shift_back=alignment_shift,
                x2_sum_threshold=x2_sum_threshold,
                smoothing=smoothing,
                y=capture,
                h=self._filters[n],
                instantaneous_error_out=_ie_out,
            )

            lag_estimate = max_square_peak_index(self._filters[n])
            reliable = (
                lag_estimate > 2
                and lag_estimate < (self._filter_size - 10)
                and error_sum < self._matching_filter_threshold * error_sum_anchor
            )

            lag = lag_estimate + alignment_shift
            if filters_updated and reliable and error_sum < winner_error_sum:
                winner_error_sum = error_sum
                winner_index = n
                if previous_lag_estimate is not None and previous_lag_estimate == lag:
                    self._winner_lag = previous_lag_estimate
                    winner_index = n - 1
                else:
                    self._winner_lag = lag
            previous_lag_estimate = lag
            alignment_shift += self._filter_intra_lag_shift

        if winner_index != -1 and self._winner_lag is not None:
            self._reported_lag = LagEstimate(
                lag=int(self._winner_lag), pre_echo_lag=int(self._winner_lag)
            )
            # Pre-echo path — only fires when detect_pre_echo=True and the
            # same filter remains winner across calls.
            if self._detect_pre_echo and self._last_detected_best_lag_filter == winner_index:
                # AEC3 gate is in int16-scale (error_sum > 1.0 in Q15²).
                # float[-1,1] equivalent: 1.0 / 32768² ≈ 9.3e-10.
                if error_sum_anchor > 1.0 / (32768.0 * 32768.0):
                    _update_accumulated_error(
                        self._instantaneous_error,
                        self._accumulated_error[winner_index],
                        1.0 / error_sum_anchor,
                    )
                    self._number_pre_echo_updates += 1
                if self._number_pre_echo_updates >= _PRE_ECHO_UPDATES_TO_REPORT:
                    pre_echo = _compute_pre_echo_lag(
                        self._accumulated_error[winner_index],
                        int(self._winner_lag),
                        winner_index * self._filter_intra_lag_shift,
                    )
                    self._reported_lag = LagEstimate(
                        lag=int(self._winner_lag), pre_echo_lag=int(pre_echo)
                    )
            self._last_detected_best_lag_filter = winner_index
