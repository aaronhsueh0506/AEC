"""Render-activity, filter-convergence, double-talk, and plateau detectors.

The module provides four stateful detectors:

* ``RenderActivityDetector`` — far-end activity + stationarity
* ``FilterConvergenceAnalyzer`` — convergence state + divergence EMA
* ``DoubleTalkAnalyzer`` — DT classifier (energy + shadow differential)
* ``FilterPlateauDetector`` — ERLE plateau detector for adaptive tuning

Read-only state snapshots use value types from ``modules.dataclasses``.

Timing convention
-----------------
Per-hop EMA coefficients and explicitly duration-authored gates in this module
were calibrated at hop=160 @ 16000 Hz (10 ms) and are retimed to the live grid
in ``__init__`` via
``aec3_scale.growth_rehop`` / ``aec3_scale.ms_to_hops``. The authored values are
kept as the ``*_REF`` class attributes so the intent stays readable; the
instance attributes are what the update methods use.

Authoring-grid evidence, reproduced numerically rather than assumed:
``ALPHA_CV_REF`` 0.99 -> TC 994.99 ms against its own "TC ~ 1 s" comment, and
``GRACE_FRAMES_REF`` 400 -> 4000 ms against its "~4 s at 100 fps" comment
(100 fps IS a 10 ms hop). Neither reproduces at 8.000 / 16.000 / 10.667 ms.

NOT retimed, deliberately: ``CONV_FRAMES``, ``CONV_ERLE_DB``, ``DIV_ERLE_LIN``,
``STATIONARY_CV2``, ``ERL_CEILING_FLOOR``, ``SAFETY_MARGIN`` and the
``FilterPlateauDetector`` ratio/threshold defaults, retry budget and unresolved
consecutive-evidence gate. These are evidence counts or dimensionless gates,
not established durations; see each comment.
"""
import numpy as np

from . import aec3_scale
from .dataclasses import RenderActivityState, FilterConvergenceState

# The grid every wall-clock constant in this module was authored against.
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
    ALPHA_CV_REF = 0.99       # TC ≈ 1 s envelope smoothing, at the 10 ms ref hop
    # Dimensionless CV² level gate: env_var / (env_mean² + eps) carries no time
    # content (all of the timescale lives in ALPHA_CV), so it is grid-invariant.
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
                # No variance data yet; don't declare stationarity on first frame.
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
    """Owns filter-convergence state machine + divergence-indicator EMA.

    Convergence rule: 10 consecutive far-active frames with inst-ERLE > 5 dB
    after warmup is exhausted.
    Divergence rule: post-convergence EMA of (inst-ERLE_linear < 0.63 ↔ ERLE < -2 dB).

    External signals (EPC, delay shift, echo-path change) call mark_diverged()
    to invalidate convergence and reset the counter — the analyzer never
    self-resets on those events.
    """
    # dB level gate on a power ratio -- no time content, grid-invariant.
    CONV_ERLE_DB = 5.0
    # NOT a duration, so NOT retimed: update_convergence() early-returns without
    # touching _conv_counter when far is inactive / warmup is unfinished /
    # near_power is negligible, so non-qualifying hops are SKIPPED rather than
    # resetting it. Only a qualifying-but-failing hop resets. The realised span
    # is therefore far-end-duty dependent, which makes this a consecutive-
    # evidence count, matching config.py's ne_recent_sustain=3 precedent.
    CONV_FRAMES = 10
    # Dimensionless linear power ratio (-2 dB) -- grid-invariant.
    DIV_ERLE_LIN = 0.63
    DIV_ALPHA_REF = 0.9       # per-hop EMA over a boolean indicator, 10 ms ref
    DIV_DECAY_REF = 0.95      # per-hop else-branch decay, 10 ms ref

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
    """Aggregates the three pre-filter / cross-filter DT signals.

    Owns:
        _dt_from_energy   : pre-filter mic-energy excess over (far × ERL_ceiling × 2)
                            EMA-smoothed (fast-rise 0.3/0.7, slow-decay 0.9/0.1).
        _dt_from_shadow   : (shadow_advantage − offset) / scale, smoothed 70/30.
        _shadow_advantage : main_err / shadow_err (raw, no EMA).

    Coherence-based double-talk detection was retired; the
    dt_from_coherence diagnostic is now permanently 0.0.
    """

    # Shadow-filter DT blind period, 200 ms (commit 6cd995e shortened it from
    # 500 ms). 20 hops only AT the 10 ms reference hop, hence the retime.
    SHADOW_FRAME_GATE_MS = 200.0
    # Dimensionless level gates -- grid-invariant.
    ERL_CEILING_FLOOR = 0.01
    SAFETY_MARGIN = 2.0
    # Per-hop retention EMAs at the 10 ms reference hop. Only the OLD term is
    # retimed; NEW is 1-OLD by construction so the pair keeps summing to 1.
    DTE_RISE_OLD_REF = 0.3
    DTE_DECAY_OLD_REF = 0.9      # the "TC~90ms" hangover named at orchestrator.py
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

    # NOTE: this class is currently instantiated NOWHERE in the repo (the C
    # trio filter_plateau_{init,reset,update} is called only from
    # test_counter_saturation.c). The durations below are retimed for
    # consistency with the rest of the module, but that retiming is
    # unexercised by any production path -- treat it as unverified until the
    # detector is actually wired in.
    GRACE_FRAMES_MS = 4000.0           # "~4 s at 100 fps"; 100 fps IS a 10 ms hop
    POST_RESET_GRACE_MS = 2000.0       # don't re-arm within this window after a reset
    # Dimensionless ratio/level gates and a retry budget -- all grid-invariant.
    ERLE_MAX_DB_DEFAULT = 6.0
    FAR_ACTIVE_RATIO_DEFAULT = 0.5
    DT_SIGNAL_RATIO_DEFAULT = 0.10
    MAX_ATTEMPTS_DEFAULT = 2
    # Classification UNRESOLVED, left at the authored value. It skips without
    # resetting at three sites, which matches the evidence-count shape, but
    # none of those skips is signal-duty-dependent, so the realised span is a
    # duty-INDEPENDENT 500 ms debounce -- which is the duration shape. Settling
    # it needs a measurement no production path can currently supply, because
    # the detector is not wired in. Do not retime on a guess.
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
