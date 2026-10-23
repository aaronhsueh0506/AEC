"""ErlEstimator: per-bin echo return loss with minimum-statistics tracking (single channel).

A downward jump holds the estimate for 4 s (converted to hops from the live grid)
before it recovers upward by doubling per update.
"""
import numpy as np

from .. import aec3_scale as _aec3_scale

_MIN_ERL = 0.01
_MAX_ERL = 1000.0
_AEC3_X2_MIN = 44015068.0  # scaled per hop in __init__


class ErlEstimator:
    def __init__(self, *, startup_phase_length_hops: int = 200, n_bins: int = 257,
                 hop_size: int = 160, sample_rate: int = 16000) -> None:
        self._startup_hops = int(startup_phase_length_hops)
        self._n_bins = int(n_bins)
        self._x2_min = _aec3_scale.per_bin_psd_threshold(_AEC3_X2_MIN, hop_size)
        # 4 s hold-after-drop at the live grid.
        self._hold_hops = _aec3_scale.ms_to_hops(4000.0, hop_size, sample_rate)
        self._erl = np.full(self._n_bins, _MAX_ERL, dtype=np.float32)
        self._hold_counters = np.zeros(self._n_bins - 2, dtype=np.int32)
        self._erl_time_domain = _MAX_ERL
        self._hold_counter_time_domain = 0
        self._blocks_since_reset = 0

    def reset(self) -> None:
        self._blocks_since_reset = 0

    def update(
        self,
        *,
        render_psd: np.ndarray,    # X2 (per-bin)
        capture_psd: np.ndarray,   # Y2 (per-bin)
        converged_filter: bool,
    ) -> None:
        self._blocks_since_reset += 1
        if self._blocks_since_reset < self._startup_hops or not converged_filter:
            return
        x2 = render_psd
        y2 = capture_psd
        for k in range(1, self._n_bins - 1):
            if x2[k] > self._x2_min:
                new_erl = y2[k] / x2[k]
                if new_erl < self._erl[k]:
                    self._hold_counters[k - 1] = self._hold_hops
                    self._erl[k] += 0.1 * (new_erl - self._erl[k])
                    self._erl[k] = max(self._erl[k], _MIN_ERL)
        # Counters at or below zero double the estimate per update (recovery toward max).
        self._hold_counters -= 1
        for k in range(1, self._n_bins - 1):
            if self._hold_counters[k - 1] <= 0:
                self._erl[k] = min(_MAX_ERL, 2.0 * self._erl[k])
        self._erl[0] = self._erl[1]
        self._erl[-1] = self._erl[-2]
        x2_sum = float(np.sum(x2))
        if x2_sum > self._x2_min * x2.size:
            y2_sum = float(np.sum(y2))
            new_erl = y2_sum / x2_sum
            if new_erl < self._erl_time_domain:
                self._hold_counter_time_domain = self._hold_hops
                self._erl_time_domain += 0.1 * (new_erl - self._erl_time_domain)
                self._erl_time_domain = max(self._erl_time_domain, _MIN_ERL)
        self._hold_counter_time_domain -= 1
        if self._hold_counter_time_domain <= 0:
            self._erl_time_domain = min(_MAX_ERL, 2.0 * self._erl_time_domain)

    def erl(self) -> np.ndarray:
        return self._erl

    def erl_time_domain(self) -> float:
        return self._erl_time_domain
