"""ResidualEchoEstimator — port of AEC3 residual_echo_estimator.cc.

Mirrors docs/aec3_extracts/src/aec3/residual_echo_estimator.{cc,h}.

Single-channel port. Computes R² (residual echo power²) per-bin from:
  - Linear path  (UsableLinearEstimate=True):  R² = S²_linear / ERLE
  - NonLinear   (UsableLinearEstimate=False): R² = X² * echo_path_gain²
                with noise gate + stationary subtraction.
  - Saturated echo override: R² = Y² (capture power) in either path.

Reverb addition NOT YET WIRED — Phase 3.4 ports ReverbModelEstimator
which produces ``aec_state.get_reverb_frequency_response()`` (currently
returns zeros, so AddReverb is a no-op). Once reverb lands the chain is
unchanged here; AecState just starts returning non-zero reverb.

NeuralResidualEchoEstimator NOT ported (off by default in AEC3 config).

UseStationarityProperties NOT yet wired (echo_audibility port deferred);
default config.echo_audibility.use_stationarity_properties = False so
this branch is inactive in default configs.
"""
from dataclasses import dataclass

import numpy as np

from ..state.aec_state import AecState


@dataclass(frozen=True)
class EchoModelConfig:
    """Subset of AEC3 ``EchoCanceller3Config::EchoModel`` we use."""

    noise_floor_hold: int = 50
    min_noise_floor_power: float = 1638400.0  # AEC3 default (= 16²·6400)
    noise_gate_power: float = 27509562.0       # AEC3 default
    noise_gate_slope: float = 0.3
    stationary_gate_slope: float = 10.0
    model_reverb_in_nonlinear_mode: bool = True


@dataclass(frozen=True)
class EpStrengthConfig:
    """Subset of AEC3 ``EchoCanceller3Config::EpStrength``."""

    default_gain: float = 0.014  # gain_amplitude; squared inside GetEchoPathGain
    bounded_erl: bool = False
    erle_onset_compensation_in_dominant_nearend: bool = False


_TRANSPARENT_MODE_GAIN = 0.01  # AEC3 kDefaultTransparentModeGain verbatim


class ResidualEchoEstimator:
    def __init__(
        self,
        *,
        n_bins: int = 257,
        echo_model: EchoModelConfig = EchoModelConfig(),
        ep_strength: EpStrengthConfig = EpStrengthConfig(),
    ) -> None:
        self._n_bins = int(n_bins)
        self._echo_model = echo_model
        self._ep_strength = ep_strength
        self._tm_gain_early = _TRANSPARENT_MODE_GAIN
        self._tm_gain_late = _TRANSPARENT_MODE_GAIN
        self._default_gain_early = float(ep_strength.default_gain)
        self._default_gain_late = float(ep_strength.default_gain)
        self._erle_onset_compensation_in_dominant = bool(
            ep_strength.erle_onset_compensation_in_dominant_nearend
        )
        self._x2_noise_floor = np.full(
            self._n_bins, echo_model.min_noise_floor_power, dtype=np.float32
        )
        self._x2_noise_floor_counter = np.full(
            self._n_bins, echo_model.noise_floor_hold, dtype=np.int32
        )

    def reset(self) -> None:
        self._x2_noise_floor.fill(self._echo_model.min_noise_floor_power)
        self._x2_noise_floor_counter.fill(self._echo_model.noise_floor_hold)

    def estimate(
        self,
        *,
        aec_state: AecState,
        render_psd: np.ndarray,    # X² current frame (single-channel)
        capture_psd: np.ndarray,   # Y²
        s2_linear: np.ndarray,     # |H·X|² from PBFDKF
        dominant_nearend: bool,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Returns ``(R2, R2_unbounded)``."""
        # Step 1: update stationary render noise floor.
        self._update_render_noise_power(render_psd)

        r2 = np.empty(self._n_bins, dtype=np.float32)
        r2_unbounded = np.empty(self._n_bins, dtype=np.float32)

        usable = aec_state.usable_linear_estimate()
        saturated = aec_state.saturated_echo()

        if usable:
            if saturated:
                r2[:] = capture_psd
                r2_unbounded[:] = capture_psd
            else:
                onset_compensated = (
                    self._erle_onset_compensation_in_dominant or not dominant_nearend
                )
                erle = aec_state.erle(onset_compensated)
                erle_unb = aec_state.erle_unbounded()
                # AEC3 cc:91-105 — R² = S²_linear / ERLE per-bin.
                r2[:] = s2_linear / np.maximum(erle, 1e-30)
                r2_unbounded[:] = s2_linear / np.maximum(erle_unb, 1e-30)
            # Reverb add (stub until Phase 3.4 — get_reverb_frequency_response
            # returns zeros so this is a no-op).
            reverb = aec_state.get_reverb_frequency_response()
            r2 += reverb
            r2_unbounded += reverb
        else:
            echo_path_gain = self._get_echo_path_gain(
                aec_state, gain_for_early_reflections=True
            )
            if saturated:
                r2[:] = capture_psd
                r2_unbounded[:] = capture_psd
            else:
                # Take current-frame render PSD as X². AEC3 walks a window of
                # render history at the filter-delay alignment; we use current
                # frame since our orchestrator hands us the already
                # delay-aligned render. EchoGeneratingPower window port can
                # come later if needed.
                x2 = np.asarray(render_psd, dtype=np.float32).copy()
                if not aec_state.transparent_mode_active():
                    # AEC3 cc:121-129 noise gate.
                    mask = self._echo_model.noise_gate_power > x2
                    x2[mask] = np.maximum(
                        0.0,
                        x2[mask]
                        - self._echo_model.noise_gate_slope
                        * (self._echo_model.noise_gate_power - x2[mask]),
                    )
                # Subtract stationary noise (AEC3 cc:284-288).
                x2 -= self._echo_model.stationary_gate_slope * self._x2_noise_floor
                np.maximum(x2, 0.0, out=x2)
                r2[:] = x2 * echo_path_gain
                r2_unbounded[:] = x2 * echo_path_gain
            # Reverb add in nonlinear mode (stub).
            if (
                self._echo_model.model_reverb_in_nonlinear_mode
                and not aec_state.transparent_mode_active()
            ):
                reverb = aec_state.get_reverb_frequency_response()
                r2 += reverb
                r2_unbounded += reverb

        return r2, r2_unbounded

    # ------------------------------------------------------------- helpers

    def _update_render_noise_power(self, render_psd: np.ndarray) -> None:
        """Mirrors UpdateRenderNoisePower (cc:325-359). Per-bin
        min-statistics with rapid decrease + delayed leaky increase."""
        # Decrease rapidly: where X² < current noise floor, snap down.
        mask_down = render_psd < self._x2_noise_floor
        self._x2_noise_floor[mask_down] = render_psd[mask_down].astype(np.float32)
        self._x2_noise_floor_counter[mask_down] = 0
        # Increase: bins past the hold counter ramp up 10% / step (clamped to min).
        not_down = ~mask_down
        hold = self._echo_model.noise_floor_hold
        ramp_mask = not_down & (self._x2_noise_floor_counter >= hold)
        if ramp_mask.any():
            self._x2_noise_floor[ramp_mask] = np.maximum(
                self._x2_noise_floor[ramp_mask] * 1.1,
                self._echo_model.min_noise_floor_power,
            )
        # Counter increment for non-down bins not yet past hold.
        not_down_hold = not_down & (self._x2_noise_floor_counter < hold)
        self._x2_noise_floor_counter[not_down_hold] += 1

    def _get_echo_path_gain(
        self, aec_state: AecState, *, gain_for_early_reflections: bool
    ) -> float:
        if aec_state.transparent_mode_active():
            g = self._tm_gain_early if gain_for_early_reflections else self._tm_gain_late
        else:
            g = (
                self._default_gain_early
                if gain_for_early_reflections
                else self._default_gain_late
            )
        return float(g * g)
