"""FullBandErleEstimator — fullband ERLE (single-channel) with quality estimate.

Mirrors docs/aec3_extracts/src/aec3/fullband_erle_estimator.{cc,h}.

Single-channel port: AEC3's per-channel vector is collapsed to scalars
since our pipeline is always num_capture_channels=1.

Rate notes:
  - ``kBlocksToHoldErle = 100`` AEC3 blocks (~400 ms) -> our 40 hops.
  - ``kPointsToAccumulate = 6`` is a count, not a time -> stays 6.
  - The 0.0004 forget factor on max/min ERLE is per-update; keeps the
    same per-update rate inside our hop-rate caller so DO NOT rescale.
"""
from typing import Optional

import numpy as np

from ._constants import HOPS_PER_SECOND


_EPSILON = 1e-3
_X2_BAND_ENERGY_THRESHOLD = 44015068.0
_POINTS_TO_ACCUMULATE = 6
_BLOCKS_TO_HOLD_ERLE = int(0.4 * HOPS_PER_SECOND)  # AEC3 100 blocks (~400 ms) -> 40 hops


def _fast_log2(x: float) -> float:
    return float(np.log2(x))


class _ErleInstantaneous:
    """Mirrors FullBandErleEstimator::ErleInstantaneous."""

    def __init__(
        self,
        *,
        clamp_quality_to_zero: bool = True,
        clamp_quality_to_one: bool = True,
        quality_alpha: float = 0.07,
    ) -> None:
        self._clamp_zero = bool(clamp_quality_to_zero)
        self._clamp_one = bool(clamp_quality_to_one)
        self._quality_alpha = float(quality_alpha)
        self._erle_log2: Optional[float] = None
        self._inst_quality_estimate = 0.0
        self._max_erle_log2 = -10.0
        self._min_erle_log2 = 33.0
        self._y2_acum = 0.0
        self._e2_acum = 0.0
        self._num_points = 0

    def reset(self) -> None:
        self.reset_accumulators()
        self._max_erle_log2 = -10.0
        self._min_erle_log2 = 33.0
        self._inst_quality_estimate = 0.0

    def reset_accumulators(self) -> None:
        self._erle_log2 = None
        self._inst_quality_estimate = 0.0
        self._num_points = 0
        self._y2_acum = 0.0
        self._e2_acum = 0.0

    def update(self, y2_sum: float, e2_sum: float) -> bool:
        update_estimates = False
        self._e2_acum += e2_sum
        self._y2_acum += y2_sum
        self._num_points += 1
        if self._num_points == _POINTS_TO_ACCUMULATE:
            if self._e2_acum > 0.0:
                update_estimates = True
                self._erle_log2 = _fast_log2(self._y2_acum / self._e2_acum + _EPSILON)
            self._num_points = 0
            self._e2_acum = 0.0
            self._y2_acum = 0.0
        if update_estimates:
            self._update_max_min()
            self._update_quality_estimate()
        return update_estimates

    def get_inst_erle_log2(self) -> Optional[float]:
        return self._erle_log2

    def get_quality_estimate(self) -> Optional[float]:
        if self._erle_log2 is None:
            return None
        value = self._inst_quality_estimate
        if self._clamp_zero:
            value = max(0.0, value)
        if self._clamp_one:
            value = min(1.0, value)
        return value

    def _update_max_min(self) -> None:
        # 0.0004 per update ~ 1 dB / 3 s — keep verbatim.
        self._max_erle_log2 -= 0.0004
        self._max_erle_log2 = max(self._max_erle_log2, self._erle_log2)
        self._min_erle_log2 += 0.0004
        self._min_erle_log2 = min(self._min_erle_log2, self._erle_log2)

    def _update_quality_estimate(self) -> None:
        alpha = self._quality_alpha
        quality = 0.0
        if self._max_erle_log2 > self._min_erle_log2:
            quality = (self._erle_log2 - self._min_erle_log2) / (
                self._max_erle_log2 - self._min_erle_log2
            )
        if quality > self._inst_quality_estimate:
            self._inst_quality_estimate = quality
        else:
            self._inst_quality_estimate += alpha * (quality - self._inst_quality_estimate)


class FullBandErleEstimator:
    """Single-channel fullband ERLE in log2 units (FullbandErleLog2)."""

    def __init__(self, *, min_erle: float = 1.0, max_erle_l: float = 4.0,
                 wallclock_smoothing: bool = False, hop_size: int = 160,
                 sample_rate: int = 16000) -> None:
        # AEC3 fullband EMAs (quality 0.07, erle_time_domain 0.05) are
        # per-4ms-block; both fire on the 6-point accumulation here (per hop),
        # 2.5× slower in wall-clock. Convert when ON (config
        # use_aec3_wallclock_fullband_erle_smoothing). 0.0004 envelope left verbatim.
        if wallclock_smoothing:
            from ..aec3_scale import per_block_ema_alpha_to_per_hop as _ema
            _q_alpha = float(_ema(0.07, hop_size, sample_rate))
            self._td_alpha = float(_ema(0.05, hop_size, sample_rate))
        else:
            _q_alpha = 0.07
            self._td_alpha = 0.05
        self._min_erle_log2 = _fast_log2(min_erle + _EPSILON)
        self._max_erle_lf_log2 = _fast_log2(max_erle_l + _EPSILON)
        self._hold_counter_inst_erle = 0
        self._erle_time_domain_log2 = self._min_erle_log2
        self._instantaneous_erle = _ErleInstantaneous(quality_alpha=_q_alpha)
        self._linear_filter_quality: Optional[float] = None
        self.reset()

    def reset(self) -> None:
        self._instantaneous_erle.reset()
        self._update_quality_estimate()
        self._erle_time_domain_log2 = self._min_erle_log2
        self._hold_counter_inst_erle = 0

    def update(
        self,
        *,
        x2: np.ndarray,
        y2: np.ndarray,
        e2: np.ndarray,
        converged_filter: bool,
    ) -> None:
        if converged_filter:
            x2_sum = float(np.sum(x2))
            if x2_sum > _X2_BAND_ENERGY_THRESHOLD * x2.size:
                y2_sum = float(np.sum(y2))
                e2_sum = float(np.sum(e2))
                if self._instantaneous_erle.update(y2_sum, e2_sum):
                    self._hold_counter_inst_erle = _BLOCKS_TO_HOLD_ERLE
                    self._erle_time_domain_log2 += self._td_alpha * (
                        self._instantaneous_erle.get_inst_erle_log2()
                        - self._erle_time_domain_log2
                    )
                    self._erle_time_domain_log2 = max(
                        self._erle_time_domain_log2, self._min_erle_log2
                    )
        self._hold_counter_inst_erle -= 1
        if self._hold_counter_inst_erle == 0:
            self._instantaneous_erle.reset_accumulators()
        self._update_quality_estimate()

    def fullband_erle_log2(self) -> float:
        return self._erle_time_domain_log2

    def get_inst_linear_quality_estimate(self) -> Optional[float]:
        return self._linear_filter_quality

    def _update_quality_estimate(self) -> None:
        self._linear_filter_quality = self._instantaneous_erle.get_quality_estimate()
