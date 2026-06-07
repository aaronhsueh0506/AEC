"""Detector cluster (render-activity / filter-convergence / DT / plateau).

Extracted from ``aec.py`` during refactor R.6. Four detectors:

* ``RenderActivityDetector`` — far-end activity + stationarity
* ``FilterConvergenceAnalyzer`` — convergence state + divergence EMA
* ``DoubleTalkAnalyzer`` — DT classifier (energy + shadow differential)
* ``FilterPlateauDetector`` — ERLE plateau detector for adaptive tuning

Consumes value-type dataclasses from ``modules.dataclasses`` for the
read-only state snapshots they emit.

Self-contained: numpy + modules.dataclasses.
"""
import numpy as np

from .dataclasses import RenderActivityState, FilterConvergenceState


class RenderActivityDetector:
    """Far-end activity + stationarity detector.

    Tracks far-end power envelope EMA and its variance to detect
    stationary far-end (white noise / fans), which downstream blocks
    use to gate EPC false-positives and to switch DT-detection branches.

    State (private):
        _env_mean       far-power EMA
        _env_var        far-power variance EMA
        _active_prev    True after first audible far-end frame; resets only when far drops to silence
        _is_stationary  CV² < threshold this frame
    """
    ALPHA_CV = 0.99           # TC ≈ 1 s envelope smoothing
    STATIONARY_CV2 = 0.02     # CV² gate for stationary far-end

    def __init__(self):
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
    """Owns filter-convergence state machine + divergence-indicator EMA.

    Convergence rule: 10 consecutive far-active frames with inst-ERLE > 5 dB
    after warmup is exhausted.
    Divergence rule: post-convergence EMA of (inst-ERLE_linear < 0.63 ↔ ERLE < -2 dB).

    External signals (EPC, delay shift, echo-path change) call mark_diverged()
    to invalidate convergence and reset the counter — the analyzer never
    self-resets on those events.
    """
    CONV_ERLE_DB = 5.0
    CONV_FRAMES = 10
    DIV_ERLE_LIN = 0.63
    DIV_ALPHA = 0.9
    DIV_DECAY = 0.95

    def __init__(self):
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
    """Aggregates the three pre-filter / cross-filter DT signals.

    Owns:
        _dt_from_energy   : pre-filter mic-energy excess over (far × ERL_ceiling × 2)
                            EMA-smoothed (fast-rise 0.3/0.7, slow-decay 0.9/0.1).
        _dt_from_shadow   : (shadow_advantage − offset) / scale, smoothed 70/30.
        _shadow_advantage : main_err / shadow_err (raw, no EMA).

    Coherence-based double-talk detection was retired; the
    dt_from_coherence diagnostic is now permanently 0.0.
    """

    SHADOW_FRAME_GATE = 50  # match shadow filter warmup
    ERL_CEILING_FLOOR = 0.01
    SAFETY_MARGIN = 2.0
    DTE_RISE_OLD, DTE_RISE_NEW = 0.3, 0.7
    DTE_DECAY_OLD, DTE_DECAY_NEW = 0.9, 0.1
    DTS_OLD, DTS_NEW = 0.7, 0.3
    DTS_INACTIVE_DECAY = 0.95

    def __init__(self, config: 'AecConfig'):
        self.config = config
        self._dt_from_energy = 0.0
        self._dt_from_shadow = 0.0
        self._shadow_advantage = 1.0

    def reset(self) -> None:
        self._dt_from_energy = 0.0
        self._dt_from_shadow = 0.0
        self._shadow_advantage = 1.0

    def update_shadow_dt(self, *, shadow_frame_count: int, far_excited: bool,
                         main_err_smooth: float, shadow_err_smooth: float) -> None:
        """Shadow-advantage based DT signal. Runs once per frame in the shadow block."""
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
        """Pre-filter mic-energy DT signal. Runs once per frame in the RES block."""
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
    """Detect filter stuck at bad-convergence plateau and trigger a
    one-shot recovery (taps reset + re-warmup).

    Motivation
    ----------
    DT-from-frame-0 cases (NE talks from sample 0) cause the main filter
    to learn NE leak in the first ~100 frames. Once these poisoned taps
    are encoded, even a perfect DTD downstream only prevents *further*
    learning — the existing taps stay wrong, ERLE stays low (~4 dB),
    coherence DTD never recovers (the positive-feedback trap that
    upstream DTD fixes mu_scale, but not at the tap level).

    Trigger conditions (all must hold for `consecutive_required` frames)
    ------------------------------------------------------------------
      • frame_count > grace_frames    — give natural convergence time
      • not once_converged            — never declared converged
      • far_active_ratio > 0.5        — far-end has been active enough
      • erle_windowed_db < erle_max   — filter not learning effectively
      • dt_signal_ratio > 0.10        — DT pattern (NE present, suspect
                                         DT-from-frame-0)

    Recovery action (one-shot per session, capped by `max_attempts=2`)
    ------------------------------------------------------------------
      1. main filter taps reset
      2. shadow filter taps reset
      3. convergence state machine reset
      4. RES reset (clears any garbage echo PSD estimate)
      5. force a brief warmup-equivalent window with high mu

    After 2nd attempt fails to converge, give up gracefully.
    """

    GRACE_FRAMES_DEFAULT = 400        # ~4 s at 100 fps
    ERLE_MAX_DB_DEFAULT = 6.0
    FAR_ACTIVE_RATIO_DEFAULT = 0.5
    DT_SIGNAL_RATIO_DEFAULT = 0.10
    CONSECUTIVE_REQUIRED = 50          # hold criteria for 50 frames before firing
    POST_RESET_GRACE_FRAMES = 200     # don't re-arm within this many frames after a reset
    MAX_ATTEMPTS_DEFAULT = 2

    def __init__(self,
                 grace_frames: int = GRACE_FRAMES_DEFAULT,
                 erle_max_db: float = ERLE_MAX_DB_DEFAULT,
                 far_active_ratio: float = FAR_ACTIVE_RATIO_DEFAULT,
                 dt_signal_ratio: float = DT_SIGNAL_RATIO_DEFAULT,
                 max_attempts: int = MAX_ATTEMPTS_DEFAULT):
        self.grace_frames = grace_frames
        self.erle_max_db = erle_max_db
        self.far_active_ratio = far_active_ratio
        self.dt_signal_ratio = dt_signal_ratio
        self.max_attempts = max_attempts
        # Session counters
        self._frame_count = 0
        self._far_active_count = 0
        self._dt_signal_count = 0
        self._consecutive_match = 0
        self._attempts = 0
        self._cooldown_remaining = 0
        # Last reset frame (audio-passive trace)
        self._last_reset_frame = -1

    def reset(self) -> None:
        """Full reset (call from AEC.reset)."""
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
            # Filter recovered naturally — clear consecutive counter so we
            # don't fire later if temporary divergence happens. Recovery
            # attempts are still tracked across the session.
            self._consecutive_match = 0
            return False

        if self._attempts >= self.max_attempts:
            return False

        if self._frame_count <= self.grace_frames:
            return False

        far_ratio = self._far_active_count / max(1, self._frame_count)
        dt_ratio = self._dt_signal_count / max(1, self._frame_count)

        # Require current frame to also show DT presence, otherwise
        # session-level cumulative dt_ratio could remain >threshold long after
        # the DT-from-zero stretch ended and trigger plateau recovery during
        # pure FS segments (where ERLE-low + far-active alone are not pathological).
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

        # Fire — recovery action. Reset cumulative far/dt counters so the
        # next evaluation window judges post-reset behaviour, not the
        # bad-startup history that caused this fire. _attempts and
        # _last_reset_frame remain session-level (MAX_ATTEMPTS bounds retries).
        self._consecutive_match = 0
        self._attempts += 1
        self._cooldown_remaining = self.POST_RESET_GRACE_FRAMES
        self._last_reset_frame = self._frame_count
        self._frame_count = 0
        self._far_active_count = 0
        self._dt_signal_count = 0
        return True

    @property
    def attempts(self) -> int: return self._attempts
    @property
    def last_reset_frame(self) -> int: return self._last_reset_frame
