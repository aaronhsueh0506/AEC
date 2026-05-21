"""AEC3 FilterAnalyzer single-channel port.

Mirrors docs/aec3_extracts/src/aec3/filter_analyzer.{cc,h}.

Per-frame ``update(filter_taps, render_block)`` produces:
  - ``filter_delays_blocks()`` -> ``[peak_index // HOP_SAMPLES]``
  - ``min_filter_delay_blocks()`` -> the same scalar
  - ``any_filter_consistent()`` -> ConsistentFilterDetector verdict
  - ``max_echo_path_gain()`` -> |h[peak]| (post 5 s convergence)
  - ``get_adjusted_filters()`` -> the high-pass-filtered TD taps

The peak-finding region cycles incrementally over the full filter
length, processing ``HOP_SAMPLES`` samples per call (AEC3 verbatim
``kNumberBlocksToUpdate=1`` * ``kBlockSize``). The HPF is the AEC3
verbatim 3-tap minimum-phase ~600 Hz cutoff filter.

Block unit translation:
  AEC3 kBlockSize=64 samples (4 ms @16k) -> our HOP_SAMPLES=160 (10 ms).
  delay_blocks = peak_index // HOP_SAMPLES. With filter_taps ~ 960
  samples (6 partitions * 160) the delay range is 0..5 blocks (vs
  AEC3's 0..12); resolution is 2.5x coarser but the downstream
  consumer (FilterDelay.min_direct_path_filter_delay) only needs a
  non-zero scalar in OUR block units.

Active-render threshold uses ``render_block_scaled`` (int16 amplitude),
matching the units AEC3 uses (``active_render_limit`` defaults to 100
in int16 amplitude).
"""
from typing import Optional

import numpy as np

from .._rates import HOP_SAMPLES, SR_HZ


_HOPS_PER_SECOND = SR_HZ // HOP_SAMPLES  # 16000/160 = 100

# AEC3 verbatim ints translated to OUR hop unit.
_CONVERGENCE_THRESHOLD_HOPS = 5 * _HOPS_PER_SECOND     # AEC3 5 * 250 -> 500
_CONSISTENT_HOLD_HOPS = int(1.5 * _HOPS_PER_SECOND)    # AEC3 1.5 * 250 -> 150

# Minimum-phase HPF cutoff ~600 Hz (filter_analyzer.cc:170-172 verbatim).
_HPF_COEFFS = np.array([0.7929742, -0.36072128, -0.47047766], dtype=np.float32)

# ConsistentFilterDetector thresholds (filter_analyzer.cc:264-265 verbatim).
_FLOOR_RATIO = 10.0
_SECONDARY_RATIO = 2.0
_FLOOR_LOW_OFFSET = 64
_FLOOR_HIGH_OFFSET = 128

# AEC3 EchoCanceller3Config::render_levels.active_render_limit default = 100.f
# (int16 amplitude). Threshold per block = (limit^2) * block_length.
_DEFAULT_ACTIVE_RENDER_LIMIT = 100.0


class _ConsistentFilterDetector:
    """Mirrors FilterAnalyzer::ConsistentFilterDetector.

    State is reset at the start of each full-filter sweep (when
    region_start == 0) — see filter_analyzer.cc:232-238.
    """

    def __init__(self, active_render_threshold: float) -> None:
        self._active_render_threshold = active_render_threshold
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
            self._floor_low_limit = peak_index - _FLOOR_LOW_OFFSET if peak_index >= _FLOOR_LOW_OFFSET else 0
            self._floor_high_limit = (
                0 if peak_index > size - (_FLOOR_HIGH_OFFSET + 1) else peak_index + _FLOOR_HIGH_OFFSET
            )

        # Lower sweep: region_start .. min(region_end+1, floor_low_limit)
        hi_lower = min(region_end + 1, self._floor_low_limit)
        if region_start < hi_lower:
            seg = np.abs(filter_taps[region_start:hi_lower])
            self._floor_accum += float(seg.sum())
            self._secondary_peak = max(self._secondary_peak, float(seg.max(initial=0.0)))

        # Upper sweep: max(floor_high_limit, region_start) .. region_end
        lo_upper = max(self._floor_high_limit, region_start)
        if lo_upper <= region_end:
            seg = np.abs(filter_taps[lo_upper:region_end + 1])
            self._floor_accum += float(seg.sum())
            self._secondary_peak = max(self._secondary_peak, float(seg.max(initial=0.0)))

        # End-of-sweep significance check (filter_analyzer.cc:258-266).
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

        return self._counter > _CONSISTENT_HOLD_HOPS


class FilterAnalyzer:
    """Single-capture-channel AEC3 FilterAnalyzer.

    Construct once at AecState init (when the flag is on). Per frame,
    call ``update(filter_taps_full, render_block_scaled)`` BEFORE
    ``FilterDelay.update``.
    """

    def __init__(
        self,
        *,
        active_render_limit: float = _DEFAULT_ACTIVE_RENDER_LIMIT,
        bounded_erl: bool = False,
        default_gain: float = 1.0,
    ) -> None:
        self._active_render_threshold = float((active_render_limit ** 2) * HOP_SAMPLES)
        self._bounded_erl = bool(bounded_erl)
        self._default_gain = float(default_gain)
        self._consistent = _ConsistentFilterDetector(self._active_render_threshold)
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
        # filter_analyzer.cc:194-206. Cycles one block forward; wraps at end.
        block = HOP_SAMPLES
        if self._region_end >= size - 1:
            self._region_start = 0
        else:
            self._region_start = self._region_end + 1
        self._region_end = min(self._region_start + block - 1, size - 1)

    def _preprocess(self, filter_taps: np.ndarray) -> None:
        # filter_analyzer.cc:161-187: zero region, then causal 3-tap convolution.
        size = filter_taps.size
        if self._h_highpass is None or self._h_highpass.size != size:
            self._h_highpass = np.zeros(size, dtype=np.float32)
        self._h_highpass[self._region_start:self._region_end + 1] = 0.0
        start = max(_HPF_COEFFS.size - 1, self._region_start)
        end = self._region_end + 1
        if start >= end:
            return
        # Causal FIR with 3 taps, vectorised.
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
        # Find peak (filter_analyzer.cc:127-130). Compare squared magnitudes.
        if self._peak_index >= size:
            self._peak_index = size - 1
        cur_sq = float(h[self._peak_index] * h[self._peak_index])
        seg = h[self._region_start:self._region_end + 1]
        seg_sq = seg * seg
        seg_amax = int(np.argmax(seg_sq))
        if float(seg_sq[seg_amax]) > cur_sq:
            self._peak_index = self._region_start + seg_amax
        self._delay_blocks = self._peak_index // HOP_SAMPLES
        # UpdateFilterGain (filter_analyzer.cc:142-159). Uses the RAW filter,
        # not the HPF'd version (matches AEC3 — only peak-finding sees HPF).
        sufficient = self._blocks_since_reset > _CONVERGENCE_THRESHOLD_HOPS
        peak_abs = float(abs(filter_taps[self._peak_index]))
        if sufficient and self._consistent_estimate:
            self._gain = peak_abs
        elif self._gain:
            self._gain = max(self._gain, peak_abs)
        if self._bounded_erl and self._gain:
            self._gain = max(self._gain, 0.01)
        # ConsistentFilterDetector (filter_analyzer.cc:135-138).
        self._consistent_estimate = self._consistent.detect(
            h, self._region_start, self._region_end, render_block,
            self._peak_index, self._delay_blocks,
        )

    # ------------------------ public queries -------------------------
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
