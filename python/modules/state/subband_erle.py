"""SubbandErleEstimator: per-bin ERLE with onset compensation (single channel).

Accumulates 6 hops, holds ERLE for 400 ms and watches for onsets for 1 s (at the live grid).
"""
import numpy as np

from .. import aec3_scale as _aec3_scale

_AEC3_X2_BAND_ENERGY_THRESHOLD = 44015068.0  # scaled per hop in __init__
_POINTS_TO_ACCUMULATE = 6
_UNBOUNDED_ERLE_MAX = 100000.0

# The lower half of the bins take ``max_erle_l``, the rest ``max_erle_h``.


def _set_max_erle_bands(n_bins: int, max_erle_l: float, max_erle_h: float) -> np.ndarray:
    half = (n_bins - 1) // 2  # kFftLengthBy2 // 2 analogue
    arr = np.empty(n_bins, dtype=np.float32)
    arr[:half] = max_erle_l
    arr[half:] = max_erle_h
    return arr


class SubbandErleEstimator:
    def __init__(
        self,
        *,
        n_bins: int = 257,
        min_erle: float = 1.0,
        max_erle_l: float = 4.0,
        max_erle_h: float = 1.5,
        use_onset_detection: bool = True,
        use_min_erle_during_onsets: bool = True,
        hop_size: int = 160,
        sample_rate: int = 16000,
    ) -> None:
        self._n_bins = int(n_bins)
        self._min_erle = float(min_erle)
        self._max_erle = _set_max_erle_bands(self._n_bins, max_erle_l, max_erle_h)
        self._use_onset_detection = bool(use_onset_detection)
        self._use_min_erle_during_onsets = bool(use_min_erle_during_onsets)
        # Hold 400 ms, onset detection 1 s, at the live grid.
        self._blocks_to_hold_erle = _aec3_scale.ms_to_hops(400.0, hop_size, sample_rate)
        self._blocks_for_onset_detection = _aec3_scale.ms_to_hops(1000.0, hop_size, sample_rate)
        # alpha_up/alpha_down fire once per 6-hop window and the factor 6 cancels, so the
        # rescale uses a single hop; onset_release_decay fires every hop.
        self._alpha_up = _aec3_scale.per_block_ema_alpha_to_per_hop(
            0.05, hop_size, sample_rate
        )
        self._alpha_down = _aec3_scale.per_block_ema_alpha_to_per_hop(
            0.1, hop_size, sample_rate
        )
        self._onset_release_decay = 1.0 - _aec3_scale.per_block_ema_alpha_to_per_hop(
            1.0 - 0.97, hop_size, sample_rate
        )
        self._erle = np.full(self._n_bins, self._min_erle, dtype=np.float32)
        self._erle_onset_compensated = np.full(self._n_bins, self._min_erle, dtype=np.float32)
        self._erle_unbounded = np.full(self._n_bins, self._min_erle, dtype=np.float32)
        self._erle_during_onsets = np.full(self._n_bins, self._min_erle, dtype=np.float32)
        self._coming_onset = np.ones(self._n_bins, dtype=bool)
        self._hold_counters = np.zeros(self._n_bins, dtype=np.int32)
        self._y2_acc = np.zeros(self._n_bins, dtype=np.float32)
        self._e2_acc = np.zeros(self._n_bins, dtype=np.float32)
        self._low_render_energy = np.zeros(self._n_bins, dtype=bool)
        self._num_points = 0
        # Per-bin render energy floor scaled to the frame size.
        self._x2_band_energy_threshold = _aec3_scale.per_bin_psd_threshold(
            _AEC3_X2_BAND_ENERGY_THRESHOLD, hop_size
        )
        self.reset()

    def reset(self) -> None:
        self._erle.fill(self._min_erle)
        self._erle_onset_compensated.fill(self._min_erle)
        self._erle_unbounded.fill(self._min_erle)
        self._erle_during_onsets.fill(self._min_erle)
        self._coming_onset.fill(True)
        self._hold_counters.fill(0)
        self._reset_accumulated_spectra()

    def update(
        self,
        *,
        x2: np.ndarray,
        y2: np.ndarray,
        e2: np.ndarray,
        converged_filter: bool,
        coh_gate_mask: np.ndarray = None,  # coherence gate (bool, len=n_bins)
    ) -> None:
        self._update_accumulated_spectra(x2, y2, e2, converged_filter)
        self._update_bands(converged_filter, coh_gate_mask=coh_gate_mask)
        if self._use_onset_detection:
            self._decrease_erle_per_band_for_low_render_signals()
        self._erle[0] = self._erle[1]
        self._erle[-1] = self._erle[-2]
        self._erle_onset_compensated[0] = self._erle_onset_compensated[1]
        self._erle_onset_compensated[-1] = self._erle_onset_compensated[-2]
        self._erle_unbounded[0] = self._erle_unbounded[1]
        self._erle_unbounded[-1] = self._erle_unbounded[-2]

    def erle(self, onset_compensated: bool) -> np.ndarray:
        if onset_compensated and self._use_onset_detection:
            return self._erle_onset_compensated
        return self._erle

    def erle_unbounded(self) -> np.ndarray:
        return self._erle_unbounded

    def erle_during_onsets(self) -> np.ndarray:
        return self._erle_during_onsets

    # ----------------------------------------------------------- internals

    def _reset_accumulated_spectra(self) -> None:
        self._y2_acc.fill(0.0)
        self._e2_acc.fill(0.0)
        self._low_render_energy.fill(False)
        self._num_points = 0

    def _update_accumulated_spectra(
        self,
        x2: np.ndarray,
        y2: np.ndarray,
        e2: np.ndarray,
        converged_filter: bool,
    ) -> None:
        if not converged_filter:
            return
        if self._num_points == _POINTS_TO_ACCUMULATE:
            self._num_points = 0
            self._y2_acc.fill(0.0)
            self._e2_acc.fill(0.0)
            self._low_render_energy.fill(False)
        self._y2_acc += y2
        self._e2_acc += e2
        self._low_render_energy |= x2 < self._x2_band_energy_threshold
        self._num_points += 1

    def _update_bands(self, converged_filter: bool,
                      coh_gate_mask: np.ndarray = None) -> None:
        if not converged_filter:
            return
        if self._num_points != _POINTS_TO_ACCUMULATE:
            return
        new_erle = np.empty(self._n_bins, dtype=np.float32)
        is_erle_updated = np.zeros(self._n_bins, dtype=bool)
        mid = slice(1, self._n_bins - 1)
        mask = self._e2_acc[mid] > 0.0
        new_erle[mid] = np.where(
            mask, self._y2_acc[mid] / np.maximum(self._e2_acc[mid], 1e-30), 0.0
        )
        is_erle_updated[mid] = mask
        # Coherence gate: freeze ERLE for bins where the mask is False (low coherence, e.g. double-talk).
        if coh_gate_mask is not None:
            is_erle_updated[mid] &= coh_gate_mask[mid]
        # Onset detection: a bin with enough render energy leaves the coming-onset state and holds.
        if self._use_onset_detection:
            for k in range(1, self._n_bins - 1):
                if not is_erle_updated[k] or self._low_render_energy[k]:
                    continue
                if self._coming_onset[k]:
                    self._coming_onset[k] = False
                    if not self._use_min_erle_during_onsets:
                        alpha = 0.3 if new_erle[k] < self._erle_during_onsets[k] else 0.15
                        v = self._erle_during_onsets[k] + alpha * (
                            new_erle[k] - self._erle_during_onsets[k]
                        )
                        self._erle_during_onsets[k] = float(
                            np.clip(v, self._min_erle, self._max_erle[k])
                        )
                self._hold_counters[k] = self._blocks_for_onset_detection
        for k in range(1, self._n_bins - 1):
            if not is_erle_updated[k]:
                continue
            low = bool(self._low_render_energy[k])
            self._erle[k] = self._smoothed_update(
                self._erle[k], new_erle[k], low, self._min_erle, float(self._max_erle[k])
            )
            if self._use_onset_detection:
                self._erle_onset_compensated[k] = self._smoothed_update(
                    self._erle_onset_compensated[k],
                    new_erle[k],
                    low,
                    self._min_erle,
                    float(self._max_erle[k]),
                )
            self._erle_unbounded[k] = self._smoothed_update(
                self._erle_unbounded[k],
                new_erle[k],
                low,
                self._min_erle,
                _UNBOUNDED_ERLE_MAX,
            )

    def _smoothed_update(
        self, prev: float, new_val: float, low_render: bool, min_v: float, max_v: float
    ) -> float:
        alpha = self._alpha_up
        if new_val < prev:
            alpha = 0.0 if low_render else self._alpha_down
        v = prev + alpha * (new_val - prev)
        return float(np.clip(v, min_v, max_v))

    def _decrease_erle_per_band_for_low_render_signals(self) -> None:
        for k in range(1, self._n_bins - 1):
            self._hold_counters[k] -= 1
            if self._hold_counters[k] <= (
                self._blocks_for_onset_detection - self._blocks_to_hold_erle
            ):
                if self._erle_onset_compensated[k] > self._erle_during_onsets[k]:
                    self._erle_onset_compensated[k] = max(
                        self._erle_during_onsets[k],
                        self._onset_release_decay * self._erle_onset_compensated[k],
                    )
                if self._hold_counters[k] <= 0:
                    self._coming_onset[k] = True
                    self._hold_counters[k] = 0
