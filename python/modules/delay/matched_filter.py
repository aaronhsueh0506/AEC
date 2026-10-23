"""Matched filter: bank of NLMS cross-correlators over a downsampled render ring.

Sizes are in downsampled units (4 kHz grid, 0.25 ms per sample, 16-sample sub-block = 4 ms).
Defaults: window 32 sub-blocks (128 ms), alignment shift 24 sub-blocks (96 ms), 5 filters,
so the bank spans 512 ms. Smoothing constants are per sub-block update and do not rescale.
"""
from dataclasses import dataclass
from typing import Optional

import numpy as np

from .downsampled_ring import DownsampledRenderBuffer


@dataclass(frozen=True)
class LagEstimate:
    lag: int  # downsampled samples (multiply by the decimation factor for raw samples)
    pre_echo_lag: int


# Subsample rate for accumulated error.
_ACCUMULATED_ERROR_SUBSAMPLE_RATE = 4
_PRE_ECHO_UPDATES_TO_REPORT = 50
# Saturation guard (int16 32000 on the float [-1, 1] scale).
_SATURATION_LIMIT = 32000.0 / 32768.0   # 0.9766


def max_square_peak_index(h: np.ndarray) -> int:
    """Index of the strongest squared tap (0 when fewer than 2 taps)."""
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
    """NLMS matched-filter core over a forward-stored ring; returns (filters_updated, error_sum).

    For each sub-block sample ``i`` it gathers ``h.size`` render samples ending
    ``alignment_shift_back + (sub_block_size - 1 - i)`` before the newest, forms
    ``e = y[i] - h.x`` and updates ``h`` when ``x2_sum > x2_sum_threshold`` and the
    capture is not saturated. If ``instantaneous_error_out`` is given
    (length filter_size / 4), the squared prefix error per 4-tap group is accumulated into it.
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
            # Per 4-tap group: accumulate the squared error of the filter prefix up to that group.
            g = _ACCUMULATED_ERROR_SUBSAMPLE_RATE
            prefix = np.cumsum((h * x_window).reshape(-1, g).sum(axis=1))
            diff = prefix - float(y[i])
            instantaneous_error_out[:diff.size] += diff * diff
        if x2_sum > x2_sum_threshold and not saturation:
            alpha = smoothing * e / x2_sum
            h += alpha * x_window
            filters_updated = True
    return filters_updated, error_sum


def _update_accumulated_error(
    instantaneous: np.ndarray, accumulated: np.ndarray, one_over_anchor: float
) -> None:
    """Update the accumulated per-group error: take the minimum, otherwise smooth upward."""
    SMOOTH_INC = 0.015
    norm = instantaneous * one_over_anchor
    mask = norm < accumulated
    accumulated[mask] = norm[mask]
    accumulated[~mask] += SMOOTH_INC * (norm[~mask] - accumulated[~mask])


def _compute_pre_echo_lag(
    accumulated_error: np.ndarray, lag: int, alignment_shift_winner: int
) -> int:
    """Walk back from the winner for the earliest tap group whose prefix explains the echo."""
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
    """Bank of ``num_filters`` cross-correlation filters covering staggered lag windows.

    ``update()`` is called once per sub-block. The winner is the reliable filter (peak
    in [3, size-10], error_sum < threshold * anchor) with the smallest error_sum;
    equal lags on adjacent filters resolve to the earlier filter.
    ``get_best_lag_estimate()`` returns the lag in downsampled samples, or None.
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
        """Histogram-sizing bound in downsampled samples; not a delay ceiling.

        It exceeds the geometric span (2048 ds = 512 ms at the defaults) by one
        alignment shift; the deepest accepted lag is 2037 ds = 509.25 ms.
        """
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
            # Pre-echo tracking: only when enabled and the same filter wins on consecutive calls.
            if self._detect_pre_echo and self._last_detected_best_lag_filter == winner_index:
                # Gate at int16 error_sum > 1.0, i.e. 1 / 32768^2 on the float scale.
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
