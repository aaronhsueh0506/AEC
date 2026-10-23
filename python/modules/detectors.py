"""Render-activity, filter-convergence, double-talk and plateau detectors.

``RenderActivityDetector`` (far-end activity and stationarity), ``FilterConvergenceAnalyzer``
(convergence state, divergence EMA), ``DoubleTalkAnalyzer`` (energy and shadow-differential
double-talk signals), ``FilterPlateauDetector`` (ERLE plateau, one-shot recovery).
Per-hop EMA coefficients and duration gates are authored at hop 160 / 16 kHz (``*_REF``) and
retimed to the live grid in ``__init__``; dimensionless gates and evidence counts are not.
"""
import numpy as np

from . import aec3_scale
from .dataclasses import RenderActivityState, FilterConvergenceState

# Grid the wall-clock constants were authored against.
_REF_HOP = 160
_REF_SR = 16000


def _rehop(retention_at_ref: float, hop_size: int, sample_rate: int) -> float:
    """Retime a per-hop retention-convention EMA coefficient to the live grid."""
    return aec3_scale.growth_rehop(retention_at_ref, _REF_HOP, _REF_SR,
                                   hop_size, sample_rate)


def _rehop_ms(authored_ms: float, hop_size: int, sample_rate: int) -> int:
    """Retime an authored duration to an integer hop count on the live grid."""
    return aec3_scale.ms_to_hops(authored_ms, hop_size, sample_rate)


class RenderActivityDetector:
    """Far-end activity and stationarity detector.

    Tracks the far-end power envelope EMA and its variance; low coefficient of variation
    (CV^2 < ``STATIONARY_CV2``) marks a stationary far-end (noise, fans), used to gate
    path-change false positives and DT branches.
    """
    ALPHA_CV_REF = 0.99  # envelope smoothing, about 1 s at the 10 ms ref hop
    # Dimensionless CV^2 gate (grid-invariant).
    STATIONARY_CV2 = 0.02

    def __init__(self, hop_size: int = _REF_HOP, sample_rate: int = _REF_SR):
        self.ALPHA_CV = _rehop(self.ALPHA_CV_REF, hop_size, sample_rate)
        self._env_mean = 1e-10
        self._env_var = 0.0
        self._active_prev = False
        self._is_stationary = False

    def reset(self) -> None:
        self._env_mean = 1e-10
        self._env_var = 0.0
        self._active_prev = False
        self._is_stationary = False

    def update(self, far_end: np.ndarray) -> RenderActivityState:
        far_pwr_raw = float(np.mean(far_end ** 2))
        far_pwr = far_pwr_raw + 1e-10
        warmup_active = far_pwr_raw > 1e-6
        if far_pwr > 1e-6:
            if not self._active_prev:
                self._env_mean = far_pwr
                self._env_var = 0.0
                self._active_prev = True
                # No variance data yet.
                self._is_stationary = False
            else:
                old_mean = self._env_mean
                self._env_mean = (self.ALPHA_CV * self._env_mean
                                  + (1 - self.ALPHA_CV) * far_pwr)
                self._env_var = (self.ALPHA_CV * self._env_var
                                 + (1 - self.ALPHA_CV) * (far_pwr - old_mean) ** 2)
                far_cv2 = self._env_var / (self._env_mean ** 2 + 1e-10)
                self._is_stationary = far_cv2 < self.STATIONARY_CV2
        else:
            self._active_prev = False
            self._is_stationary = False
        return RenderActivityState(
            far_pwr=far_pwr,
            is_active=self._active_prev,
            is_stationary=self._is_stationary,
            warmup_active=warmup_active,
        )

    @property
    def is_active(self) -> bool: return self._active_prev
    @property
    def is_stationary(self) -> bool: return self._is_stationary


class FilterConvergenceAnalyzer:
    """Filter-convergence state machine and divergence-indicator EMA.

    Converged after ``CONV_FRAMES`` consecutive far-active frames with inst-ERLE above
    ``CONV_ERLE_DB`` once warmup is done. Divergence is a post-convergence EMA of
    "inst-ERLE < -2 dB". ``mark_diverged()`` is the only way external events invalidate
    convergence.
    """
    # dB gate, grid-invariant
    CONV_ERLE_DB = 5.0
    # Consecutive-evidence count (not retimed): non-qualifying hops (far inactive, warmup
    # unfinished, negligible near power) are skipped without resetting the counter.
    CONV_FRAMES = 10
    # linear power ratio (-2 dB), grid-invariant
    DIV_ERLE_LIN = 0.63
    DIV_ALPHA_REF = 0.9  # per-hop EMA over a boolean indicator, at the 10 ms ref hop
    DIV_DECAY_REF = 0.95  # per-hop decay, at the 10 ms ref hop

    def __init__(self, hop_size: int = _REF_HOP, sample_rate: int = _REF_SR):
        self.DIV_ALPHA = _rehop(self.DIV_ALPHA_REF, hop_size, sample_rate)
        self.DIV_DECAY = _rehop(self.DIV_DECAY_REF, hop_size, sample_rate)
        self._converged = False
        self._once_converged = False
        self._conv_counter = 0
        self._divergence = 0.0

    def reset(self) -> None:
        self._converged = False
        self._once_converged = False
        self._conv_counter = 0
        self._divergence = 0.0

    def mark_diverged(self) -> None:
        """EPC / delay shift / echo-path change: drop convergence, restart counter."""
        self._converged = False
        self._conv_counter = 0

    def update_divergence(self, near_power: float, raw_error_power: float) -> None:
        """Mid-frame divergence indicator EMA (only meaningful post-convergence)."""
        if self._converged and near_power > 1e-8:
            inst_erle_lin = near_power / (raw_error_power + 1e-10)
            is_diverged = float(inst_erle_lin < self.DIV_ERLE_LIN)
            self._divergence = (self.DIV_ALPHA * self._divergence
                                + (1 - self.DIV_ALPHA) * is_diverged)
        else:
            self._divergence *= self.DIV_DECAY

    def update_convergence(self, *, near_power: float, raw_error_power: float,
                           far_active: bool, warmup_done: bool) -> bool:
        """End-of-frame convergence detection. Returns True on the transition frame."""
        if self._converged or near_power <= 1e-8 or not warmup_done or not far_active:
            return False
        inst_erle_db = 10.0 * np.log10(near_power / (raw_error_power + 1e-10))
        if inst_erle_db > self.CONV_ERLE_DB:
            self._conv_counter += 1
        else:
            self._conv_counter = 0
        if self._conv_counter >= self.CONV_FRAMES:
            self._converged = True
            self._once_converged = True
            return True
        return False

    @property
    def converged(self) -> bool: return self._converged
    @property
    def once_converged(self) -> bool: return self._once_converged
    @property
    def divergence(self) -> float: return self._divergence


class DoubleTalkAnalyzer:
    """Aggregates the pre-filter and cross-filter double-talk signals.

    ``_dt_from_energy``: mic-energy excess over (far x ERL ceiling x margin), fast rise and
    slow decay EMA. ``_dt_from_shadow``: (shadow_advantage - offset) / scale, smoothed.
    ``_shadow_advantage``: main_err / shadow_err, raw.
    """

    # Shadow-filter DT blind period; retimed from the 10 ms ref hop.
    SHADOW_FRAME_GATE_MS = 200.0
    ERL_CEILING_FLOOR = 0.01
    SAFETY_MARGIN = 2.0
    # Retention EMAs at the 10 ms ref hop; only the old term is retimed, new = 1 - old.
    DTE_RISE_OLD_REF = 0.3
    DTE_DECAY_OLD_REF = 0.9
    DTS_OLD_REF = 0.7
    DTS_INACTIVE_DECAY_REF = 0.95

    def __init__(self, config: 'AecConfig'):
        self.config = config
        hop, sr = config.hop_size, config.sample_rate
        self.SHADOW_FRAME_GATE = _rehop_ms(self.SHADOW_FRAME_GATE_MS, hop, sr)
        self.DTE_RISE_OLD = _rehop(self.DTE_RISE_OLD_REF, hop, sr)
        self.DTE_RISE_NEW = 1.0 - self.DTE_RISE_OLD
        self.DTE_DECAY_OLD = _rehop(self.DTE_DECAY_OLD_REF, hop, sr)
        self.DTE_DECAY_NEW = 1.0 - self.DTE_DECAY_OLD
        self.DTS_OLD = _rehop(self.DTS_OLD_REF, hop, sr)
        self.DTS_NEW = 1.0 - self.DTS_OLD
        self.DTS_INACTIVE_DECAY = _rehop(self.DTS_INACTIVE_DECAY_REF, hop, sr)
        self._dt_from_energy = 0.0
        self._dt_from_shadow = 0.0
        self._shadow_advantage = 1.0

    def reset(self) -> None:
        self._dt_from_energy = 0.0
        self._dt_from_shadow = 0.0
        self._shadow_advantage = 1.0

    def update_shadow_dt(self, *, shadow_frame_count: int, far_excited: bool,
                         main_err_smooth: float, shadow_err_smooth: float) -> None:
        """Shadow-advantage DT signal; once per frame in the shadow block."""
        if shadow_frame_count >= self.SHADOW_FRAME_GATE and far_excited:
            self._shadow_advantage = main_err_smooth / (shadow_err_smooth + 1e-10)
            raw = float(np.clip(
                (self._shadow_advantage - self.config.shadow_dtd_offset)
                / self.config.shadow_dtd_advantage_scale,
                0.0, 1.0))
            self._dt_from_shadow = self.DTS_OLD * self._dt_from_shadow + self.DTS_NEW * raw
        else:
            self._dt_from_shadow *= self.DTS_INACTIVE_DECAY

    def update_energy_dt(self, *, far_active: bool, far_pwr: float,
                         mic_pwr: float, erl_estimate: float) -> None:
        """Pre-filter mic-energy DT signal; once per frame in the RES block."""
        if far_active and far_pwr > 1e-4:
            erl_ceiling = 1.0 / max(erl_estimate, self.ERL_CEILING_FLOOR)
            max_echo_expected = far_pwr * erl_ceiling * self.SAFETY_MARGIN
            inst = max(0.0, (mic_pwr - max_echo_expected) / mic_pwr)
        else:
            inst = 0.0
        if inst > self._dt_from_energy:
            self._dt_from_energy = (self.DTE_RISE_OLD * self._dt_from_energy
                                    + self.DTE_RISE_NEW * inst)
        else:
            self._dt_from_energy = (self.DTE_DECAY_OLD * self._dt_from_energy
                                    + self.DTE_DECAY_NEW * inst)

    @property
    def dt_from_energy(self) -> float: return self._dt_from_energy
    @property
    def dt_from_shadow(self) -> float: return self._dt_from_shadow
    @property
    def shadow_advantage(self) -> float: return self._shadow_advantage


class FilterPlateauDetector:
    """Detects a filter stuck at a bad-convergence plateau and triggers one-shot recovery.

    Trigger (all hold for ``CONSECUTIVE_REQUIRED`` frames): past the grace period, never
    converged, far-end active more than ``far_active_ratio`` of the time, windowed ERLE
    below ``erle_max_db``, a double-talk signal ratio above ``dt_signal_ratio`` and
    double-talk present in the current frame. Recovery (the caller resets main/shadow
    taps, convergence state and RES, then re-warms) fires at most ``max_attempts`` times.
    """

    # Not wired into the pipeline.
    GRACE_FRAMES_MS = 4000.0
    POST_RESET_GRACE_MS = 2000.0  # no re-arm within this window after a reset
    ERLE_MAX_DB_DEFAULT = 6.0
    FAR_ACTIVE_RATIO_DEFAULT = 0.5
    DT_SIGNAL_RATIO_DEFAULT = 0.10
    MAX_ATTEMPTS_DEFAULT = 2
    # Consecutive-evidence count kept at the authored value (not retimed).
    CONSECUTIVE_REQUIRED = 50

    def __init__(self,
                 grace_frames: int = None,
                 erle_max_db: float = ERLE_MAX_DB_DEFAULT,
                 far_active_ratio: float = FAR_ACTIVE_RATIO_DEFAULT,
                 dt_signal_ratio: float = DT_SIGNAL_RATIO_DEFAULT,
                 max_attempts: int = MAX_ATTEMPTS_DEFAULT,
                 hop_size: int = _REF_HOP,
                 sample_rate: int = _REF_SR):
        if grace_frames is None:
            grace_frames = _rehop_ms(self.GRACE_FRAMES_MS, hop_size, sample_rate)
        self.post_reset_grace_frames = _rehop_ms(
            self.POST_RESET_GRACE_MS, hop_size, sample_rate)
        self.grace_frames = grace_frames
        self.erle_max_db = erle_max_db
        self.far_active_ratio = far_active_ratio
        self.dt_signal_ratio = dt_signal_ratio
        self.max_attempts = max_attempts
        self._frame_count = 0
        self._far_active_count = 0
        self._dt_signal_count = 0
        self._consecutive_match = 0
        self._attempts = 0
        self._cooldown_remaining = 0
        self._last_reset_frame = -1

    def reset(self) -> None:
        """Full reset."""
        self._frame_count = 0
        self._far_active_count = 0
        self._dt_signal_count = 0
        self._consecutive_match = 0
        self._attempts = 0
        self._cooldown_remaining = 0
        self._last_reset_frame = -1

    def update(self, *, far_active: bool, dt_signal_present: bool,
               erle_windowed_db: float, once_converged: bool) -> bool:
        """Tick one frame. Returns True iff a recovery action should fire."""
        self._frame_count += 1
        if far_active:
            self._far_active_count += 1
        if dt_signal_present:
            self._dt_signal_count += 1

        if self._cooldown_remaining > 0:
            self._cooldown_remaining -= 1
            return False

        if once_converged:
            # Filter recovered: clear the consecutive counter; attempts persist.
            self._consecutive_match = 0
            return False

        if self._attempts >= self.max_attempts:
            return False

        if self._frame_count <= self.grace_frames:
            return False

        far_ratio = self._far_active_count / max(1, self._frame_count)
        dt_ratio = self._dt_signal_count / max(1, self._frame_count)

        # Also require DT in the current frame, so cumulative ratios cannot trigger recovery in far-end-only segments.
        criteria_met = (
            far_ratio > self.far_active_ratio
            and erle_windowed_db < self.erle_max_db
            and dt_ratio > self.dt_signal_ratio
            and dt_signal_present
        )

        if not criteria_met:
            self._consecutive_match = 0
            return False

        self._consecutive_match += 1
        if self._consecutive_match < self.CONSECUTIVE_REQUIRED:
            return False

        # Fire: restart the cumulative counters; attempts and last_reset_frame persist.
        self._consecutive_match = 0
        self._attempts += 1
        self._cooldown_remaining = self.post_reset_grace_frames
        self._last_reset_frame = self._frame_count
        self._frame_count = 0
        self._far_active_count = 0
        self._dt_signal_count = 0
        return True

    @property
    def attempts(self) -> int: return self._attempts
    @property
    def last_reset_frame(self) -> int: return self._last_reset_frame
