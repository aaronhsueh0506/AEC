"""SubbandErleEstimator — per-bin ERLE with onset compensation (single-channel).

Mirrors docs/aec3_extracts/src/aec3/subband_erle_estimator.{cc,h}.

Single-channel port (num_capture_channels=1 always for us).

Rate notes:
  - ``kBlocksToHoldErle = 100`` AEC3 blocks (~400 ms) -> 40 hops.
  - ``kBlocksForOnsetDetection = 250`` AEC3 blocks (~1 s) -> 100 hops.
  - ``kPointsToAccumulate = 6`` is a count, not a time -> stays 6.
"""
import numpy as np

from ._constants import HOPS_PER_SECOND
from .. import aec3_scale as _aec3_scale

_AEC3_X2_BAND_ENERGY_THRESHOLD = 44015068.0  # AEC3 source value — scaled per hop in __init__
_BLOCKS_TO_HOLD_ERLE = int(0.4 * HOPS_PER_SECOND)            # AEC3 100 (~400 ms) -> 40 hops
_BLOCKS_FOR_ONSET_DETECTION = _BLOCKS_TO_HOLD_ERLE + int(0.6 * HOPS_PER_SECOND)  # AEC3 250 (~1 s) -> 100 hops
_POINTS_TO_ACCUMULATE = 6
_UNBOUNDED_ERLE_MAX = 100000.0

# AEC3 kFftLengthBy2Plus1 for the 16 kHz pipeline = 65 bins (FFT=128).
# Our pipeline has FFT=512 -> 257 bins. The first kFftLengthBy2 // 2 bins
# are the "LF half" that takes ``max_erle_l`` (AEC3 default 4.0); the rest
# take ``max_erle_h`` (default 1.5).


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
        e2y2_gate_enabled: bool = False,
        e2y2_gate_threshold: float = 0.5,
    ) -> None:
        self._n_bins = int(n_bins)
        self._min_erle = float(min_erle)
        self._max_erle = _set_max_erle_bands(self._n_bins, max_erle_l, max_erle_h)
        self._use_onset_detection = bool(use_onset_detection)
        self._use_min_erle_during_onsets = bool(use_min_erle_during_onsets)
        # AEC3 subband EMA constants are per-4ms-block; our update fires per hop.
        self._alpha_up = 0.05
        self._alpha_down = 0.1
        self._onset_release_decay = 0.97
        self._erle = np.full(self._n_bins, self._min_erle, dtype=np.float32)
        self._erle_onset_compensated = np.full(self._n_bins, self._min_erle, dtype=np.float32)
        self._erle_unbounded = np.full(self._n_bins, self._min_erle, dtype=np.float32)
        self._erle_during_onsets = np.full(self._n_bins, self._min_erle, dtype=np.float32)
        self._coming_onset = np.ones(self._n_bins, dtype=bool)
        self._hold_counters = np.zeros(self._n_bins, dtype=np.int32)
        # Accumulator state.
        self._y2_acc = np.zeros(self._n_bins, dtype=np.float32)
        self._e2_acc = np.zeros(self._n_bins, dtype=np.float32)
        self._low_render_energy = np.zeros(self._n_bins, dtype=bool)
        self._num_points = 0
        # C: E²/Y² per-bin ERLE gate (default OFF).
        # Skips ERLE update for bins where accumulated e2/y2 > threshold,
        # preventing nearend contamination in doubletalk from pulling ERLE down.
        self._e2y2_gate_enabled = bool(e2y2_gate_enabled)
        self._e2y2_gate_threshold = float(e2y2_gate_threshold)
        # Per-bin render energy floor — AEC3 kX2BandEnergyThreshold scaled to
        # our frame (2×hop_size actual samples, 50% OLA).
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
        coh_gate_mask: np.ndarray = None,  # C': coherence gate (bool, len=n_bins)
    ) -> None:
        self._update_accumulated_spectra(x2, y2, e2, converged_filter)
        self._update_bands(converged_filter, coh_gate_mask=coh_gate_mask)
        if self._use_onset_detection:
            self._decrease_erle_per_band_for_low_render_signals()
        # Mirror first / last bin (AEC3 cc:100-109).
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
        # Per-bin compute only for k = 1 .. n_bins-2 (excludes endpoints).
        mid = slice(1, self._n_bins - 1)
        mask = self._e2_acc[mid] > 0.0
        new_erle[mid] = np.where(
            mask, self._y2_acc[mid] / np.maximum(self._e2_acc[mid], 1e-30), 0.0
        )
        is_erle_updated[mid] = mask
        # C: E²/Y² gate — freeze ERLE for bins where accumulated error energy
        # exceeds threshold × capture energy (nearend contaminating error in DT).
        # In FS (echo cancellation working): e2 << y2 → ratio low → not gated.
        # In DT: nearend inflates e2 toward y2 → ratio high → freeze ERLE update.
        if self._e2y2_gate_enabled:
            with np.errstate(divide='ignore', invalid='ignore'):
                e2y2 = self._e2_acc[mid] / np.maximum(self._y2_acc[mid], 1e-30)
            is_erle_updated[mid] &= (e2y2 <= self._e2y2_gate_threshold)
        # C': Coherence gate — freeze ERLE for bins where Γ²(Ŷ, Y) < threshold.
        # coh_gate_mask[k]=True means ERLE update allowed (coherence high enough).
        # In DT: nearend decorrelates Y from Ŷ → Γ² drops → False → freeze.
        # Unlike E²/Y² gate, this does NOT use E (error) → no circular dependency.
        if coh_gate_mask is not None:
            is_erle_updated[mid] &= coh_gate_mask[mid]
        # Onset detection: when bin sees "high enough" render AND was in
        # coming_onset state, transition out and hold for a window.
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
                self._hold_counters[k] = _BLOCKS_FOR_ONSET_DETECTION
        # Per-bin smoothed ERLE update.
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
                _BLOCKS_FOR_ONSET_DETECTION - _BLOCKS_TO_HOLD_ERLE
            ):
                if self._erle_onset_compensated[k] > self._erle_during_onsets[k]:
                    self._erle_onset_compensated[k] = max(
                        self._erle_during_onsets[k],
                        self._onset_release_decay * self._erle_onset_compensated[k],
                    )
                if self._hold_counters[k] <= 0:
                    self._coming_onset[k] = True
                    self._hold_counters[k] = 0
