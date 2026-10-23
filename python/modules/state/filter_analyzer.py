"""FilterAnalyzer (single channel): peak delay, echo-path gain and consistency of the linear filter.

``update(filter_taps, render_block)`` keeps a 3-tap high-passed copy of the taps and
scans ``hop_size`` taps per call, cycling over the filter. Outputs:
``filter_delays_blocks()`` = ``[peak_index // hop_size]`` (hop units),
``any_filter_consistent()``, ``max_echo_path_gain()`` (|h[peak]| of the high-passed taps)
and ``get_adjusted_filters()``. The active-render threshold is in int16 amplitude units.
"""
from typing import Optional

import numpy as np

from .. import aec3_scale as _aec3_scale

# Minimum-phase 3-tap high-pass.
_HPF_COEFFS = np.array([0.7929742, -0.36072128, -0.47047766], dtype=np.float32)

# ConsistentFilterDetector thresholds.
_FLOOR_RATIO = 10.0
_SECONDARY_RATIO = 2.0
_FLOOR_LOW_SAMPLES_16K = 64
_FLOOR_HIGH_SAMPLES_16K = 128

# Active-render limit in int16 amplitude; threshold per block = limit^2 * block length.
_DEFAULT_ACTIVE_RENDER_LIMIT = 100.0


class _ConsistentFilterDetector:
    """Decides whether the filter peak is significant and stable over time.

    State resets at the start of each full-filter sweep (region_start == 0).
    """

    def __init__(
        self,
        active_render_threshold: float,
        consistent_hold_hops: int,
        sample_rate: int,
    ) -> None:
        self._active_render_threshold = active_render_threshold
        self._consistent_hold_hops = consistent_hold_hops
        # Floor-exclusion window: 64/128 taps at 16 kHz (4/8 ms), 3x at 48 kHz.
        scale = 3 if int(sample_rate) == 48000 else 1
        self._floor_low_offset = _FLOOR_LOW_SAMPLES_16K * scale
        self._floor_high_offset = _FLOOR_HIGH_SAMPLES_16K * scale
        self.reset()

    def reset(self) -> None:
        self._significant_peak = False
        self._floor_accum = 0.0
        self._secondary_peak = 0.0
        self._floor_low_limit = 0
        self._floor_high_limit = 0
        self._counter = 0
        self._delay_ref = -10

    def detect(
        self,
        filter_taps: np.ndarray,
        region_start: int,
        region_end: int,
        render_block: np.ndarray,
        peak_index: int,
        delay_blocks: int,
    ) -> bool:
        size = filter_taps.size
        if region_start == 0:
            self._floor_accum = 0.0
            self._secondary_peak = 0.0
            self._floor_low_limit = (
                peak_index - self._floor_low_offset
                if peak_index >= self._floor_low_offset else 0
            )
            self._floor_high_limit = (
                0 if peak_index > size - (self._floor_high_offset + 1)
                else peak_index + self._floor_high_offset
            )

        hi_lower = min(region_end + 1, self._floor_low_limit)
        if region_start < hi_lower:
            seg = np.abs(filter_taps[region_start:hi_lower])
            self._floor_accum += float(seg.sum())
            self._secondary_peak = max(self._secondary_peak, float(seg.max(initial=0.0)))

        lo_upper = max(self._floor_high_limit, region_start)
        if lo_upper <= region_end:
            seg = np.abs(filter_taps[lo_upper:region_end + 1])
            self._floor_accum += float(seg.sum())
            self._secondary_peak = max(self._secondary_peak, float(seg.max(initial=0.0)))

        # End of sweep: significance check.
        if region_end == size - 1:
            denom = max(1, self._floor_low_limit + size - self._floor_high_limit)
            floor = self._floor_accum / denom
            abs_peak = float(abs(filter_taps[peak_index]))
            self._significant_peak = (
                abs_peak > _FLOOR_RATIO * floor
                and abs_peak > _SECONDARY_RATIO * self._secondary_peak
            )

        if self._significant_peak:
            active = bool(float((render_block.astype(np.float64) ** 2).sum())
                          > self._active_render_threshold)
            if self._delay_ref == delay_blocks:
                if active:
                    self._counter += 1
            else:
                self._counter = 0
                self._delay_ref = delay_blocks

        return self._counter > self._consistent_hold_hops


class FilterAnalyzer:
    """Single-capture-channel filter analyzer; call ``update()`` before ``FilterDelay.update``."""

    def __init__(
        self,
        *,
        active_render_limit: float = _DEFAULT_ACTIVE_RENDER_LIMIT,
        bounded_erl: bool = False,
        default_gain: float = 1.0,
        hop_size: int = 160,
        sample_rate: int = 16000,
    ) -> None:
        self._hop_size = int(hop_size)
        self._active_render_threshold = float((active_render_limit ** 2) * self._hop_size)
        self._bounded_erl = bool(bounded_erl)
        self._default_gain = float(default_gain)
        # 5 s convergence threshold, 1.5 s consistency hold.
        self._convergence_threshold_hops = _aec3_scale.ms_to_hops(
            5000.0, hop_size, sample_rate
        )
        self._consistent_hold_hops = _aec3_scale.ms_to_hops(
            1500.0, hop_size, sample_rate
        )
        self._consistent = _ConsistentFilterDetector(
            self._active_render_threshold, self._consistent_hold_hops,
            sample_rate,
        )
        self._region_start = 0
        self._region_end = 0
        self._blocks_since_reset = 0
        self._peak_index = 0
        self._gain = float(default_gain)
        self._consistent_estimate = False
        self._delay_blocks = 0
        self._h_highpass: Optional[np.ndarray] = None

    def reset(self) -> None:
        self._region_start = 0
        self._region_end = 0
        self._blocks_since_reset = 0
        self._peak_index = 0
        self._gain = self._default_gain
        self._consistent_estimate = False
        self._delay_blocks = 0
        self._consistent.reset()
        if self._h_highpass is not None:
            self._h_highpass[:] = 0.0

    def _set_region(self, size: int) -> None:
        block = self._hop_size
        if self._region_end >= size - 1:
            self._region_start = 0
        else:
            self._region_start = self._region_end + 1
        self._region_end = min(self._region_start + block - 1, size - 1)

    def _preprocess(self, filter_taps: np.ndarray) -> None:
        # Zero the region, then causal 3-tap FIR.
        size = filter_taps.size
        if self._h_highpass is None or self._h_highpass.size != size:
            self._h_highpass = np.zeros(size, dtype=np.float32)
        self._h_highpass[self._region_start:self._region_end + 1] = 0.0
        start = max(_HPF_COEFFS.size - 1, self._region_start)
        end = self._region_end + 1
        if start >= end:
            return
        out = (filter_taps[start:end] * _HPF_COEFFS[0]
               + filter_taps[start - 1:end - 1] * _HPF_COEFFS[1]
               + filter_taps[start - 2:end - 2] * _HPF_COEFFS[2])
        self._h_highpass[start:end] = out.astype(np.float32)

    def update(self, filter_taps: np.ndarray, render_block: np.ndarray) -> None:
        size = int(filter_taps.size)
        if size == 0:
            return
        self._blocks_since_reset += 1
        self._set_region(size)
        self._preprocess(filter_taps)
        h = self._h_highpass
        if self._peak_index >= size:
            self._peak_index = size - 1
        cur_sq = float(h[self._peak_index] * h[self._peak_index])
        seg = h[self._region_start:self._region_end + 1]
        seg_sq = seg * seg
        seg_amax = int(np.argmax(seg_sq))
        if float(seg_sq[seg_amax]) > cur_sq:
            self._peak_index = self._region_start + seg_amax
        self._delay_blocks = self._peak_index // self._hop_size
        # Gain is |h_hpf[peak]| of the high-passed taps.
        sufficient = self._blocks_since_reset > self._convergence_threshold_hops
        peak_abs = float(abs(h[self._peak_index]))
        if sufficient and self._consistent_estimate:
            self._gain = peak_abs
        elif self._gain:
            self._gain = max(self._gain, peak_abs)
        if self._bounded_erl and self._gain:
            self._gain = max(self._gain, 0.01)
        self._consistent_estimate = self._consistent.detect(
            h, self._region_start, self._region_end, render_block,
            self._peak_index, self._delay_blocks,
        )

    def filter_delays_blocks(self) -> list[int]:
        return [self._delay_blocks]

    def min_filter_delay_blocks(self) -> int:
        return self._delay_blocks

    def any_filter_consistent(self) -> bool:
        return self._consistent_estimate

    def max_echo_path_gain(self) -> float:
        return self._gain

    def get_adjusted_filters(self) -> Optional[np.ndarray]:
        return self._h_highpass

    def peak_index(self) -> int:
        return self._peak_index
