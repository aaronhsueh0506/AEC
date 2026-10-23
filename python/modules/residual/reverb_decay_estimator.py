"""Adaptive reverb decay estimator.

Two paths, chosen by ``use_aec3_block_energy``: a partition-energy regression over
``log2(sum_k |W_p[k]|^2)`` (cheap), and a time-domain path on 64-sample blocks of the impulse
response (early-reverb length, late-reverb log2(h^2) regression); the decay is accepted when
the first reverb block exceeds 4x the tail gain, peak energy is below 100 and the late region has >= 5 blocks.
"""
from __future__ import annotations

import math
from typing import Optional

import numpy as np

from ..aec3_scale import AEC3_BLOCK_SAMPLES_16K


_K_EARLY_REVERB_MIN_SIZE_BLOCKS = 3
_K_BLOCKS_PER_SECTION = 6
_K_MIN_DECAY = 0.02
_K_MAX_DECAY = 0.95
_K_FFT_LENGTH_BY_2 = 64
_K_FFT_LENGTH_BY_2_LOG2 = 6

_LOG2_1_1 = 0.13750352374993502
_LOG2_0_8 = -0.32192809488736229


def _symmetric_arithmetic_sum(N: int) -> float:
    """Sum of squares ``2 * sum_{i=0.5}^{(N-1)/2} i^2 = N (N^2 - 1) / 12``."""
    return N * (N * N - 1.0) / 12.0


def _block_energy_average(h: np.ndarray, block_index: int) -> float:
    """Mean of squared taps in a 64-sample block."""
    s = block_index * _K_FFT_LENGTH_BY_2
    e = s + _K_FFT_LENGTH_BY_2
    seg = h[s:e]
    return float(np.sum(seg.astype(np.float64) ** 2) / _K_FFT_LENGTH_BY_2)


def _block_energy_peak(h: np.ndarray, peak_block: int) -> float:
    """Maximum squared tap in a 64-sample block."""
    s = peak_block * _K_FFT_LENGTH_BY_2
    e = s + _K_FFT_LENGTH_BY_2
    seg = h[s:e].astype(np.float64)
    peak_idx = int(np.argmax(seg * seg))
    val = float(seg[peak_idx])
    return val * val


class _LateReverbLinearRegressor:
    """Linear regression of log2(h^2) per sample; slope = sum(z * (i - mid)) / sum((i - mid)^2)
    with the denominator in closed form."""

    def __init__(self) -> None:
        self.reset(0)

    def reset(self, num_data_points: int) -> None:
        N = int(num_data_points)
        self._N = N
        self._n = 0
        self._nz = 0.0
        self._nn = _symmetric_arithmetic_sum(N)
        self._count = (-0.5 * N + 0.5) if N > 0 else 0.0

    def accumulate(self, z: float) -> None:
        self._nz += self._count * float(z)
        self._count += 1.0
        self._n += 1

    def estimate_available(self) -> bool:
        return self._n == self._N and self._N > 0

    def estimate(self) -> float:
        if self._nn == 0.0:
            return 0.0
        return self._nz / self._nn


class _EarlyReverbLengthEstimator:
    """Overlapping 6-block sections, each with its own regression numerator
    (section k covers blocks k..k+5). ``estimate()`` compares the first 9 smoothed
    numerators with the minimum of the rest to find the early-reverb extent."""

    def __init__(self, max_blocks: int) -> None:
        nsec = max(0, max_blocks - _K_BLOCKS_PER_SECTION)
        self._numerators = [0.0] * nsec
        self._numerators_smooth = [0.0] * nsec
        self._n_sections = 0
        self._coefficients_counter = 0
        self._block_counter = 0
        self._first_point = (-0.5 * _K_BLOCKS_PER_SECTION * _K_FFT_LENGTH_BY_2
                             + 0.5)

    def reset(self) -> None:
        for i in range(len(self._numerators)):
            self._numerators[i] = 0.0
        self._coefficients_counter = 0
        self._block_counter = 0

    def accumulate(self, value: float, smoothing: float) -> None:
        first_section = max(self._block_counter - _K_BLOCKS_PER_SECTION + 1, 0)
        last_section = min(self._block_counter, len(self._numerators) - 1)
        if last_section < 0:
            # Block beyond storage: still advance the counter so block boundaries trip.
            self._coefficients_counter += 1
            if self._coefficients_counter == _K_FFT_LENGTH_BY_2:
                self._block_counter += 1
                self._coefficients_counter = 0
            return

        x_value = self._coefficients_counter + self._first_point
        value_to_inc = _K_FFT_LENGTH_BY_2 * value
        value_to_add = (x_value * value
                        + (self._block_counter - last_section) * value_to_inc)
        for section in range(last_section, first_section - 1, -1):
            self._numerators[section] += value_to_add
            value_to_add += value_to_inc

        self._coefficients_counter += 1
        if self._coefficients_counter == _K_FFT_LENGTH_BY_2:
            if self._block_counter >= (_K_BLOCKS_PER_SECTION - 1):
                section = self._block_counter - (_K_BLOCKS_PER_SECTION - 1)
                if section < len(self._numerators_smooth):
                    self._numerators_smooth[section] += smoothing * (
                        self._numerators[section]
                        - self._numerators_smooth[section]
                    )
                    self._n_sections = section + 1
            self._block_counter += 1
            self._coefficients_counter = 0

    def estimate(self) -> int:
        """Early-reverb size in blocks."""
        kNumSectionsToAnalyze = 9
        if self._n_sections < kNumSectionsToAnalyze:
            return 0
        N = _K_BLOCKS_PER_SECTION * _K_FFT_LENGTH_BY_2
        nn = _symmetric_arithmetic_sum(N)
        numerator_11 = _LOG2_1_1 * nn / _K_FFT_LENGTH_BY_2
        numerator_08 = _LOG2_0_8 * nn / _K_FFT_LENGTH_BY_2

        tail = self._numerators_smooth[kNumSectionsToAnalyze:self._n_sections]
        if not tail:
            return 0
        min_numerator_tail = min(tail)

        early_size_minus_1 = 0
        for k in range(kNumSectionsToAnalyze):
            val = self._numerators_smooth[k]
            if (val > numerator_11
                    or (val < numerator_08
                        and val < 0.9 * min_numerator_tail)):
                early_size_minus_1 = k

        return 0 if early_size_minus_1 == 0 else early_size_minus_1 + 1


class ReverbDecayEstimator:
    """Adaptive per-hop reverb decay scalar; ``update()`` takes either partition energies
    or the time-domain filter, matching the mode chosen at construction."""

    def __init__(self, n_partitions: int, hop_size: int,
                 default_decay: float = 0.85,
                 mild_decay: float = 0.5,
                 use_adaptive: bool = True,
                 use_aec3_block_energy: bool = False,
                 sample_rate: int = 16000) -> None:
        self._n_partitions = int(n_partitions)
        self._hop_size = int(hop_size)
        self._sample_rate = int(sample_rate)
        self._default_decay = float(default_decay)
        self._mild_decay = float(mild_decay)
        self._use_adaptive = bool(use_adaptive)
        self._use_aec3_block_energy = bool(use_aec3_block_energy)
        self._decay: float = float(default_decay)
        self._smoothing_constant: float = 0.0

        # Time-domain path state; the filter length in blocks is set on the first update.
        self._filter_length_blocks: int = 0
        self._early_reverb_estimator: Optional[_EarlyReverbLengthEstimator] = None
        self._late_reverb_decay_estimator = _LateReverbLinearRegressor()
        self._late_reverb_start = 0
        self._late_reverb_end = 0
        self._block_to_analyze = 0
        self._estimation_region_identified = False
        self._estimation_region_candidate_size = 0
        self._previous_gains: list[float] = []
        self._tail_gain = 0.0

    @property
    def decay_value(self) -> float:
        return self._decay

    def decay(self, mild: bool) -> float:
        if self._use_adaptive:
            return self._decay
        return self._mild_decay if mild else self._default_decay

    def reset(self) -> None:
        self._decay = self._default_decay
        self._smoothing_constant = 0.0
        self._reset_decay_estimation()

    def _reset_decay_estimation(self) -> None:
        """Reset the estimation state."""
        if self._early_reverb_estimator is not None:
            self._early_reverb_estimator.reset()
        self._late_reverb_decay_estimator.reset(0)
        self._block_to_analyze = 0
        self._estimation_region_candidate_size = 0
        self._estimation_region_identified = False
        self._smoothing_constant = 0.0
        self._late_reverb_start = 0
        self._late_reverb_end = 0

    def _lazy_init_aec3(self, td_filter_size: int) -> None:
        new_blocks = td_filter_size // _K_FFT_LENGTH_BY_2
        if (self._early_reverb_estimator is None
                or self._filter_length_blocks != new_blocks):
            self._filter_length_blocks = new_blocks
            self._previous_gains = [0.0] * new_blocks
            self._early_reverb_estimator = _EarlyReverbLengthEstimator(
                max_blocks=new_blocks - _K_EARLY_REVERB_MIN_SIZE_BLOCKS
            )

    def update(self, partition_energies: Optional[np.ndarray] = None,
               *,
               time_domain_filter: Optional[np.ndarray] = None,
               filter_quality: Optional[float] = None,
               filter_delay_blocks: int = 0,
               usable_linear_filter: bool = False,
               stationary_signal: bool = False) -> None:
        """Refresh the decay scalar from ``partition_energies`` or ``time_domain_filter``
        (per the mode); a missing input for the active mode is a no-op."""
        if self._use_aec3_block_energy:
            if time_domain_filter is None:
                return
            self._update_aec3_strict(
                td_filter=time_domain_filter,
                filter_quality=filter_quality,
                filter_delay_blocks=filter_delay_blocks,
                usable_linear_filter=usable_linear_filter,
                stationary_signal=stationary_signal,
            )
        else:
            if partition_energies is None:
                return
            self._update_legacy(
                partition_energies=partition_energies,
                filter_quality=filter_quality,
                filter_delay_blocks=filter_delay_blocks,
                usable_linear_filter=usable_linear_filter,
                stationary_signal=stationary_signal,
            )


    def _update_legacy(self, partition_energies: np.ndarray,
                       filter_quality: Optional[float],
                       filter_delay_blocks: int,
                       usable_linear_filter: bool,
                       stationary_signal: bool) -> None:
        if stationary_signal:
            return
        if not self._use_adaptive:
            return
        if not usable_linear_filter:
            return
        if filter_delay_blocks < 0:
            return
        if filter_delay_blocks > (
            self._n_partitions - _K_EARLY_REVERB_MIN_SIZE_BLOCKS - 1
        ):
            return

        new_smoothing = (filter_quality or 0.0) * 0.2
        self._smoothing_constant = max(new_smoothing, self._smoothing_constant)
        if self._smoothing_constant == 0.0:
            return

        late_start = filter_delay_blocks + _K_EARLY_REVERB_MIN_SIZE_BLOCKS
        late_end = self._n_partitions
        if (late_end - late_start) < 2:
            return

        e = partition_energies[late_start:late_end].astype(np.float64)
        e = np.maximum(e, 1e-30)
        log2_e = np.log2(e)
        x = np.arange(log2_e.size, dtype=np.float64)
        x_mean = x.mean()
        y_mean = log2_e.mean()
        num = float(np.sum((x - x_mean) * (log2_e - y_mean)))
        den = float(np.sum((x - x_mean) ** 2))
        if den == 0.0:
            return
        slope_per_partition = num / den
        # Slope stored per 4 ms block; the consumer retimes it to the live hop.
        block_samples = (AEC3_BLOCK_SAMPLES_16K / 16000.0) * max(1, self._sample_rate)
        decay_log2_per_block = (
            slope_per_partition * block_samples / max(1, self._hop_size)
        )
        decay_new = float(math.pow(2.0, decay_log2_per_block))
        decay_new = max(0.97 * self._decay, decay_new)
        decay_new = min(decay_new, _K_MAX_DECAY)
        decay_new = max(decay_new, _K_MIN_DECAY)
        self._decay += self._smoothing_constant * (decay_new - self._decay)
        self._smoothing_constant = 0.0


    def _update_aec3_strict(self, td_filter: np.ndarray,
                            filter_quality: Optional[float],
                            filter_delay_blocks: int,
                            usable_linear_filter: bool,
                            stationary_signal: bool) -> None:
        """Time-domain update: gate feasibility, then analyse blocks sequentially and estimate."""
        if stationary_signal:
            return
        self._lazy_init_aec3(td_filter.size)

        filter_size = int(td_filter.size)
        feasible = (filter_delay_blocks
                    <= (self._filter_length_blocks
                        - _K_EARLY_REVERB_MIN_SIZE_BLOCKS - 1))
        feasible = feasible and (filter_delay_blocks > 0)
        feasible = feasible and usable_linear_filter
        feasible = feasible and (
            (filter_size % _K_FFT_LENGTH_BY_2) == 0
        )

        if not feasible:
            self._reset_decay_estimation()
            return

        if not self._use_adaptive:
            return

        new_smoothing = (filter_quality or 0.0) * 0.2
        self._smoothing_constant = max(new_smoothing, self._smoothing_constant)
        if self._smoothing_constant == 0.0:
            return

        if self._block_to_analyze < self._filter_length_blocks:
            self._analyze_filter(td_filter)
            self._block_to_analyze += 1
        else:
            self._estimate_decay(td_filter, filter_delay_blocks)

    def _analyze_block_gain(self, h2_block: np.ndarray,
                            previous_gain_idx: int) -> tuple[bool, bool]:
        """Returns (adapting, decaying_gain)."""
        gain = max(float(np.mean(h2_block)), 1e-32)
        prev = self._previous_gains[previous_gain_idx]
        adapting = (prev > 1.1 * gain) or (prev < 0.9 * gain)
        decaying = gain > self._tail_gain
        self._previous_gains[previous_gain_idx] = gain
        return adapting, decaying

    def _analyze_filter(self, td_filter: np.ndarray) -> None:
        """Analyse one sub-block of the filter."""
        idx = self._block_to_analyze
        s = idx * _K_FFT_LENGTH_BY_2
        e = s + _K_FFT_LENGTH_BY_2
        h = td_filter[s:e].astype(np.float64)
        h2 = h * h

        adapting, above_noise_floor = self._analyze_block_gain(h2, idx)

        self._estimation_region_identified = (
            self._estimation_region_identified
            or adapting or not above_noise_floor
        )
        if not self._estimation_region_identified:
            self._estimation_region_candidate_size += 1

        # In the late region accumulate both estimators, before it only the early one;
        # the late regressor is reset to exactly the number of points accumulated.
        if idx <= self._late_reverb_end:
            if idx >= self._late_reverb_start:
                for h2_k in h2:
                    h2_log2 = math.log2(float(h2_k) + 1e-10)
                    self._late_reverb_decay_estimator.accumulate(h2_log2)
                    if self._early_reverb_estimator is not None:
                        self._early_reverb_estimator.accumulate(
                            h2_log2, self._smoothing_constant
                        )
            else:
                for h2_k in h2:
                    h2_log2 = math.log2(float(h2_k) + 1e-10)
                    if self._early_reverb_estimator is not None:
                        self._early_reverb_estimator.accumulate(
                            h2_log2, self._smoothing_constant
                        )

    def _estimate_decay(self, td_filter: np.ndarray, peak_block: int) -> None:
        """Estimate the decay from the analysed region."""
        self._block_to_analyze = min(
            peak_block + _K_EARLY_REVERB_MIN_SIZE_BLOCKS,
            self._filter_length_blocks,
        )

        first_reverb_gain = _block_energy_average(td_filter,
                                                  self._block_to_analyze)
        h_size_blocks = td_filter.size >> _K_FFT_LENGTH_BY_2_LOG2
        self._tail_gain = _block_energy_average(td_filter, h_size_blocks - 1)
        peak_energy = _block_energy_peak(td_filter, peak_block)
        sufficient_reverb_decay = first_reverb_gain > 4.0 * self._tail_gain
        valid_filter = (first_reverb_gain > 2.0 * self._tail_gain
                        and peak_energy < 100.0)

        if self._early_reverb_estimator is not None:
            size_early_reverb = self._early_reverb_estimator.estimate()
        else:
            size_early_reverb = 0
        size_late_reverb = max(
            self._estimation_region_candidate_size - size_early_reverb, 0
        )

        if size_late_reverb >= 5:
            if (valid_filter
                    and self._late_reverb_decay_estimator.estimate_available()):
                slope = self._late_reverb_decay_estimator.estimate()
                decay = math.pow(2.0, slope * _K_FFT_LENGTH_BY_2)
                decay = max(0.97 * self._decay, decay)
                decay = min(decay, _K_MAX_DECAY)
                decay = max(decay, _K_MIN_DECAY)
                self._decay += self._smoothing_constant * (decay - self._decay)

            self._late_reverb_decay_estimator.reset(
                size_late_reverb * _K_FFT_LENGTH_BY_2
            )
            self._late_reverb_start = (
                peak_block + _K_EARLY_REVERB_MIN_SIZE_BLOCKS + size_early_reverb
            )
            self._late_reverb_end = (
                self._block_to_analyze
                + self._estimation_region_candidate_size - 1
            )
        else:
            self._late_reverb_decay_estimator.reset(0)
            self._late_reverb_start = 0
            self._late_reverb_end = 0

        self._estimation_region_identified = not (
            valid_filter and sufficient_reverb_decay
        )
        self._estimation_region_candidate_size = 0
        self._smoothing_constant = 0.0
        if self._early_reverb_estimator is not None:
            self._early_reverb_estimator.reset()


__all__ = ["ReverbDecayEstimator"]
