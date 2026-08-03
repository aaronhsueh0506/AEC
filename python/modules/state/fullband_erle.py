"""FullBandErleEstimator — fullband ERLE (single-channel) with quality estimate.

Mirrors docs/aec3_extracts/src/aec3/fullband_erle_estimator.{cc,h}.

Single-channel port: AEC3's per-channel vector is collapsed to scalars
since our pipeline is always num_capture_channels=1.

Rate notes:
  - ``kBlocksToHoldErle = 100`` AEC3 blocks (~400 ms), computed live from the
    real hop_size/sample_rate at construction time.
  - ``kPointsToAccumulate = 6`` is a count, not a time -> stays 6.
  - ``quality_alpha``/``td_alpha`` are AEC3-native per-4ms-block EMA
    constants that fire once per 6-hop accumulation window. Both our
    accumulation window (6 hops) and AEC3's reference update period (6
    native 4ms blocks = 24ms) scale by the same factor of 6, so they cancel:
    rescaling is done by passing the *single-hop* period against
    per_block_ema_alpha_to_per_hop's fixed single-native-block (4ms)
    denominator, NOT ``6*hop_size`` -- passing 6*hop_size would compare 6
    real hops against only 1 native block, inflating the exponent 6x. (A
    2026-08 review caught this: the fix on 2026-08-01 initially passed
    ``_POINTS_TO_ACCUMULATE * hop_size``, which was wrong for exactly this
    reason.)
  - The 0.0004 forget factor on max/min ERLE IS an AEC3-native per-block
    constant after all (verified against docs/aec3_extracts: verbatim
    literal + comment "Forget factor, approx 1dB every 3 sec." in
    fullband_erle_estimator.cc's UpdateMaxMin(), gated on the identical
    6-point accumulation cadence as quality_alpha/td_alpha). An earlier
    in-file claim that this was already wall-clock-calibrated and should be
    left verbatim was a provenance error; it is now rescaled the same way
    as its two siblings.
"""
from typing import Optional

import numpy as np

from .. import aec3_scale as _aec3_scale

_EPSILON = 1e-3
_AEC3_X2_BAND_ENERGY_THRESHOLD = 44015068.0  # AEC3 source value — scaled per hop in __init__
_POINTS_TO_ACCUMULATE = 6


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
        maxmin_forget: float = 0.0004,
    ) -> None:
        self._clamp_zero = bool(clamp_quality_to_zero)
        self._clamp_one = bool(clamp_quality_to_one)
        self._quality_alpha = float(quality_alpha)
        self._maxmin_forget = float(maxmin_forget)
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
        # AEC3-native per-block forget factor, rescaled to wall-clock at
        # construction time (see FullBandErleEstimator.__init__).
        self._max_erle_log2 -= self._maxmin_forget
        self._max_erle_log2 = max(self._max_erle_log2, self._erle_log2)
        self._min_erle_log2 += self._maxmin_forget
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
                 hop_size: int = 160, sample_rate: int = 16000) -> None:
        # AEC3 fullband EMAs (quality 0.07, erle_time_domain 0.05) are
        # per-4ms-block, and fire once per _POINTS_TO_ACCUMULATE(=6)-hop
        # accumulation window (see update() below), not every hop. Both our
        # update period (6*hop_size samples) and AEC3's reference period
        # (6 native 4ms blocks = 24ms) carry the same factor of 6, so it
        # cancels: rescale by comparing a *single* hop against
        # per_block_ema_alpha_to_per_hop's fixed single-native-block (4ms)
        # denominator -- pass plain hop_size, not 6*hop_size (passing
        # 6*hop_size compares 6 real hops against only 1 native block,
        # inflating the exponent 6x too far).
        _q_alpha = _aec3_scale.per_block_ema_alpha_to_per_hop(
            0.07, hop_size, sample_rate
        )
        self._td_alpha = _aec3_scale.per_block_ema_alpha_to_per_hop(
            0.05, hop_size, sample_rate
        )
        # 0.0004 is AEC3's own literal ERLE max/min envelope forget *step*
        # (fullband_erle_estimator.cc UpdateMaxMin(), same 6-point-gated
        # cadence as quality_alpha/td_alpha) -- it is a fixed additive
        # decrement per event, not a retention alpha, so the wall-clock-
        # preserving rescale is the linear per_block_rate_to_per_hop (same
        # hop_size-cancels-the-6x reasoning as above), not the EMA formula.
        _maxmin_forget = _aec3_scale.per_block_rate_to_per_hop(
            0.0004, hop_size, sample_rate
        )
        self._x2_band_energy_threshold = _aec3_scale.per_bin_psd_threshold(
            _AEC3_X2_BAND_ENERGY_THRESHOLD, hop_size
        )
        # AEC3 kBlocksToHoldErle=100 blocks (~400 ms). Was the module-level
        # `_BLOCKS_TO_HOLD_ERLE = int(0.4 * HOPS_PER_SECOND)`, baked at the
        # stale hop=160/sr=16000 assumption; computed live here instead.
        self._blocks_to_hold_erle = _aec3_scale.ms_to_hops(400.0, hop_size, sample_rate)
        self._min_erle_log2 = _fast_log2(min_erle + _EPSILON)
        self._max_erle_lf_log2 = _fast_log2(max_erle_l + _EPSILON)
        self._hold_counter_inst_erle = 0
        self._erle_time_domain_log2 = self._min_erle_log2
        self._instantaneous_erle = _ErleInstantaneous(
            quality_alpha=_q_alpha, maxmin_forget=_maxmin_forget
        )
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
            if x2_sum > self._x2_band_energy_threshold * x2.size:
                y2_sum = float(np.sum(y2))
                e2_sum = float(np.sum(e2))
                if self._instantaneous_erle.update(y2_sum, e2_sum):
                    self._hold_counter_inst_erle = self._blocks_to_hold_erle
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
