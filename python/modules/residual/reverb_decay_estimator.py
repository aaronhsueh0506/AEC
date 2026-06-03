"""Adaptive reverb decay estimator.

Two paths:

  * **Legacy (default)** — partition-energy linear regression over
    ``log2(Σ_k |W_p[k]|²)`` per partition. Coarse but cheap.

  * **AEC3-strict** (`use_aec3_block_energy=True`) — full port of
    `docs/aec3_extracts/src/aec3/reverb_decay_estimator.{cc,h}`. Operates
    on the time-domain impulse response sliced into ``kFftLengthBy2 = 64``
    sample sub-blocks. Includes:
      - ``BlockEnergyAverage`` / ``BlockEnergyPeak``
      - ``AnalyzeBlockGain`` (adapt + above-noise-floor gates)
      - ``EarlyReverbLengthEstimator`` (overlapping kBlocksPerSection=6
        windows + per-section linear regressor; identifies early
        reverberation region)
      - ``LateReverbLinearRegressor`` (per-sample log2(h²) accumulation,
        symmetric-arithmetic-sum closed form)
      - ``EstimateDecay`` gating (``first_reverb_gain > 4·tail_gain``,
        ``peak_energy < 100``, ``size_late_reverb >= 5``)

The two paths are mutually exclusive per-instance: the orchestrator
selects which input to provide. Default-OFF preserves byte-equal.
"""
from __future__ import annotations

import math
from typing import Optional

import numpy as np


_K_EARLY_REVERB_MIN_SIZE_BLOCKS = 3
_K_BLOCKS_PER_SECTION = 6
_K_MIN_DECAY = 0.02
_K_MAX_DECAY = 0.95
_K_FFT_LENGTH_BY_2 = 64           # AEC3 kFftLengthBy2 = 64 samples
_K_FFT_LENGTH_BY_2_LOG2 = 6       # log2(64)

# log2(1.1) and log2(0.8) — see AEC3 cc:372-374.
_LOG2_1_1 = 0.13750352374993502
_LOG2_0_8 = -0.32192809488736229


def _symmetric_arithmetic_sum(N: int) -> float:
    """AEC3 cc:60-62 — Arithmetic sum of 2·Σ_{i=0.5}^{(N-1)/2} i² = N(N²-1)/12."""
    return N * (N * N - 1.0) / 12.0


def _block_energy_average(h: np.ndarray, block_index: int) -> float:
    """AEC3 cc:76-85 — mean of squared TD taps in a 64-sample block."""
    s = block_index * _K_FFT_LENGTH_BY_2
    e = s + _K_FFT_LENGTH_BY_2
    seg = h[s:e]
    return float(np.sum(seg.astype(np.float64) ** 2) / _K_FFT_LENGTH_BY_2)


def _block_energy_peak(h: np.ndarray, peak_block: int) -> float:
    """AEC3 cc:65-73 — max squared TD tap in a 64-sample block."""
    s = peak_block * _K_FFT_LENGTH_BY_2
    e = s + _K_FFT_LENGTH_BY_2
    seg = h[s:e].astype(np.float64)
    peak_idx = int(np.argmax(seg * seg))
    val = float(seg[peak_idx])
    return val * val


class _LateReverbLinearRegressor:
    """Port of `ReverbDecayEstimator::LateReverbLinearRegressor` (cc:277-304).

    Linear regression on log2(h²) per sample, accumulating ``z[i] *
    (i - (N-1)/2)`` for ``i = 0..N-1``. The denominator
    ``Σ(i - (N-1)/2)²`` is a closed-form symmetric sum (``N(N²-1)/12``).
    Slope estimate is ``nz / nn`` where ``nz = Σ z·(i - mid)`` and
    ``nn = Σ (i - mid)²``.
    """

    def __init__(self) -> None:
        self.reset(0)

    def reset(self, num_data_points: int) -> None:
        N = int(num_data_points)
        # AEC3 RTC_DCHECK_EQ(0, N % 2) — must be even (we don't crash on odd).
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
    """Port of `ReverbDecayEstimator::EarlyReverbLengthEstimator` (cc:306-404).

    Overlapping ``kBlocksPerSection=6`` block windows, each with its own
    LinearRegressor numerator. Section ``k`` covers blocks ``[k, k+5]``.
    Each TD coefficient appears in up to 6 sections; the per-section
    numerator is updated using a closed-form increment formula.

    ``Estimate()`` compares first 9 sections' numerator to the minimum
    of the tail (sections ≥ 9) to identify early-reverb extent.
    """

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
        # Note: AEC3 doesn't reset numerators_smooth in Reset().

    def accumulate(self, value: float, smoothing: float) -> None:
        first_section = max(self._block_counter - _K_BLOCKS_PER_SECTION + 1, 0)
        last_section = min(self._block_counter, len(self._numerators) - 1)
        if last_section < 0:
            # Block beyond storage — still advance the coefficient counter
            # so the block boundary trips correctly.
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
        """Returns size in blocks of early reverb. AEC3 cc:366-404."""
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
    """Adaptive per-hop reverb decay scalar.

    Two operating modes selectable per-update:
      - Legacy partition path: provide ``partition_energies`` only.
      - AEC3-strict TD path: provide ``time_domain_filter`` only.
    """

    def __init__(self, n_partitions: int, hop_size: int,
                 default_decay: float = 0.85,
                 mild_decay: float = 0.5,
                 use_adaptive: bool = True,
                 use_aec3_block_energy: bool = False) -> None:
        self._n_partitions = int(n_partitions)
        self._hop_size = int(hop_size)
        self._default_decay = float(default_decay)
        self._mild_decay = float(mild_decay)
        self._use_adaptive = bool(use_adaptive)
        self._use_aec3_block_energy = bool(use_aec3_block_energy)
        self._decay: float = float(default_decay)
        self._smoothing_constant: float = 0.0

        # AEC3-strict state (cc:89-102).
        # filter_length_blocks_ = number of kFftLengthBy2 sub-blocks in TD
        # filter. Determined lazily on first update from TD length.
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
        """AEC3 cc:151-160."""
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
        """Refresh the adaptive decay scalar.

        Pass exactly one of ``partition_energies`` (legacy) or
        ``time_domain_filter`` (AEC3-strict). If AEC3-strict mode is
        enabled on the instance but ``time_domain_filter`` is not
        provided, the call is a no-op.
        """
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

    # ------------------------------------------------------------ legacy

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
        decay_log2_per_sample = slope_per_partition / max(1, self._hop_size)
        decay_new = float(math.pow(2.0, decay_log2_per_sample))
        decay_new = max(0.97 * self._decay, decay_new)
        decay_new = min(decay_new, _K_MAX_DECAY)
        decay_new = max(decay_new, _K_MIN_DECAY)
        self._decay += self._smoothing_constant * (decay_new - self._decay)
        self._smoothing_constant = 0.0

    # -------------------------------------------------------- AEC3-strict

    def _update_aec3_strict(self, td_filter: np.ndarray,
                            filter_quality: Optional[float],
                            filter_delay_blocks: int,
                            usable_linear_filter: bool,
                            stationary_signal: bool) -> None:
        """AEC3 reverb_decay_estimator.cc:106-149 verbatim port."""
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
        """AEC3 cc:47-57 — returns (adapting, decaying_gain)."""
        gain = max(float(np.mean(h2_block)), 1e-32)
        prev = self._previous_gains[previous_gain_idx]
        adapting = (prev > 1.1 * gain) or (prev < 0.9 * gain)
        decaying = gain > self._tail_gain
        self._previous_gains[previous_gain_idx] = gain
        return adapting, decaying

    def _analyze_filter(self, td_filter: np.ndarray) -> None:
        """AEC3 cc:226-264."""
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

        # AEC3 cc:250-263: inside the late region accumulate BOTH the
        # late-decay regressor and the early-length estimator; before it,
        # early only. The late regressor is Reset() in _estimate_decay to
        # exactly ``size_late_reverb * kFftLengthBy2`` points and the late
        # region spans exactly that many blocks*samples, so accumulation is
        # called exactly N times by construction — AEC3 has NO
        # ``estimate_available()`` gate here, so neither do we.
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
        """AEC3 cc:162-224."""
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
