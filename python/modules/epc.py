"""Echo-path-change family: ``classify_epc_event``, ``EchoPathChangeDetector`` and ``PathChangeRegimeHandler``."""
import numpy as np

from . import aec3_scale as _aec3_scale
from .dataclasses import AecEvent, AecEventType, EpcEvent, RegimeHandlerDecision


def classify_epc_event(epc_event: 'EpcEvent') -> AecEvent:
    """Map an ``EpcEvent`` to an ``AecEvent``; pure, returns an empty event when not fired."""
    if not epc_event.fired:
        return AecEvent()
    if epc_event.source == 'delay':
        return AecEvent(delay_change=AecEventType.DELAY_NEW_DETECTED)
    if epc_event.source in ('epv', 'shadow_rise'):
        return AecEvent(gain_change=True)
    return AecEvent()


class EchoPathChangeDetector:
    """Unified state for the three echo-path-change trigger sources.

    Triggers, invoked individually by the caller: ``force_delay()`` (delay realignment),
    ``update_epv(...)`` (fast/slow far-power EMA ratio outside [0.25, 4]) and
    ``update_shadow_rise(...)`` (both filter errors rising together after convergence).
    The caller applies the side effects from ``EpcEvent.source``; ``tick_hangover()`` runs once
    per frame inside the shadow/converged gate, only when shadow_rise did not fire.
    ``EPV_FAST_TC`` / ``EPV_SLOW_TC`` are per-hop retention constants authored at the 10 ms grid.
    """
    EPV_FAST_TC = 0.98
    EPV_SLOW_TC = 0.999
    EPV_LOW = 0.25
    EPV_HIGH = 4.0

    def __init__(self, config: 'AecConfig', *, hop_size: int = None,
                 sample_rate: int = None):
        self.config = config
        self._active = False
        self._hangover = 0
        self._epv_gain_fast = 0.0
        self._epv_gain_slow = 0.0
        self._prev_total_err = 0.0
        _hop = config.hop_size if hop_size is None else hop_size
        _sr = config.sample_rate if sample_rate is None else sample_rate
        self._epv_fast_tc = _aec3_scale.growth_rehop(
            self.EPV_FAST_TC, 160, 16000, _hop, _sr)
        self._epv_slow_tc = _aec3_scale.growth_rehop(
            self.EPV_SLOW_TC, 160, 16000, _hop, _sr)

    def reset(self) -> None:
        self._active = False
        self._hangover = 0
        self._epv_gain_fast = 0.0
        self._epv_gain_slow = 0.0
        self._prev_total_err = 0.0

    def force_delay(self) -> EpcEvent:
        """Delay-shift re-alignment (caller has validated consistency)."""
        self._active = True
        self._hangover = self.config.epc_hangover
        return EpcEvent(fired=True, source='delay')

    def update_epv(self, *, far_pwr_global: float, filter_converged: bool,
                   main_paused: bool) -> EpcEvent:
        """Far-power EPV gain ratio: update the fast/slow EMAs and fire on outliers."""
        if far_pwr_global <= 1e-6:
            return EpcEvent()
        if self._epv_gain_fast < 1e-12:
            self._epv_gain_fast = self._epv_gain_slow = far_pwr_global
        else:
            self._epv_gain_fast = (self._epv_fast_tc * self._epv_gain_fast
                                   + (1 - self._epv_fast_tc) * far_pwr_global)
            self._epv_gain_slow = (self._epv_slow_tc * self._epv_gain_slow
                                   + (1 - self._epv_slow_tc) * far_pwr_global)
        if (filter_converged and not self._active and not main_paused
                and self._epv_gain_slow > 1e-10):
            ratio = self._epv_gain_fast / (self._epv_gain_slow + 1e-10)
            if ratio < self.EPV_LOW or ratio > self.EPV_HIGH:
                self._active = True
                self._hangover = self.config.epc_hangover
                return EpcEvent(fired=True, source='epv')
        return EpcEvent()

    def update_shadow_rise(self, *, main_err_smooth: float, shadow_err_smooth: float,
                           is_stationary: bool) -> EpcEvent:
        """Shadow-based error rise; always updates ``_prev_total_err``."""
        total_err = main_err_smooth + shadow_err_smooth
        if total_err > 1e-10:
            delta_ratio = abs(main_err_smooth - shadow_err_smooth) / total_err
        else:
            delta_ratio = 0.0
        errors_rising = (total_err > self._prev_total_err * self.config.epc_total_rise
                         and self._prev_total_err > 1e-10)
        is_echo_change = errors_rising and delta_ratio < self.config.epc_delta_threshold
        # Stationary far-end plus an error rise is double-talk, not a path change.
        if is_echo_change and is_stationary:
            is_echo_change = False
        self._prev_total_err = total_err
        if is_echo_change:
            self._active = True
            self._hangover = self.config.epc_hangover
            return EpcEvent(fired=True, source='shadow_rise')
        return EpcEvent()

    def tick_hangover(self) -> None:
        """Count down the hangover and clear ``active`` when expired."""
        if self._hangover > 0:
            self._hangover -= 1
            self._active = True
        else:
            self._active = False

    @property
    def active(self) -> bool: return self._active
    @property
    def hangover_count(self) -> int: return self._hangover
    @property
    def epv_gain_fast(self) -> float: return self._epv_gain_fast
    @property
    def epv_gain_slow(self) -> float: return self._epv_gain_slow


class PathChangeRegimeHandler:
    """Shadow-copy gate state machine: protects against wildly nonstationary echo paths.

    Where the echo path has no stable post-change state, the boost_q / pause_main /
    reverse_copy actions drive the main filter toward zero (pass-through) instead of letting
    it diverge. Inputs are read-only measurements; outputs are ``RegimeHandlerDecision``
    flags that the caller applies (it never mutates filter state itself).
    """

    BASELINE_INIT = 1e-6
    # Consecutive-hop count, not a duration; not retimed.
    HYS_STREAK_MIN = 10
    # Authored on the 10 ms grid; live instances use a wall-clock-preserving retention.
    COPY_ERR_BASELINE_RETENTION = 0.995
    AEC3_STREAK_FRAMES = 5  # gate_mode='streak_only': consecutive-block rule
    SHADOW_COPY_WARMUP_MS = 500.0

    GATE_ENERGY = 'energy'  # dt_from_energy < 0.3
    GATE_COHERENCE = 'coherence'  # dt_from_coherence < 0.4
    GATE_COH_DELAY = 'coherence_delay'  # coherence gate and delay_reliable
    GATE_STREAK = 'streak_only'  # 5-block streak, no DT gate

    def __init__(self, config: 'AecConfig', gate_mode: str = 'energy'):
        self.config = config
        self.gate_mode = gate_mode
        self._hys_streak_min = self.HYS_STREAK_MIN
        self._copy_err_baseline_retention = _aec3_scale.growth_rehop(
            self.COPY_ERR_BASELINE_RETENTION, 160, 16000,
            config.hop_size, config.sample_rate)
        self._shadow_copy_warmup_hops = _aec3_scale.ms_to_hops_f32(
            self.SHADOW_COPY_WARMUP_MS, config.hop_size, config.sample_rate)
        self._copy_err_baseline = self.BASELINE_INIT
        self._copy_counter = 0
        self._streak = 0
        self._main_paused = False
        self._pause_resume = 0

    def reset(self) -> None:
        self._copy_err_baseline = self.BASELINE_INIT
        self._copy_counter = 0
        self._streak = 0
        self._main_paused = False
        self._pause_resume = 0

    @property
    def main_paused(self) -> bool:
        return self._main_paused

    @property
    def copy_err_baseline(self) -> float:
        return self._copy_err_baseline

    @property
    def copy_counter(self) -> int:
        return self._copy_counter

    @property
    def hys_streak_min(self) -> int:
        return self._hys_streak_min

    @property
    def copy_err_baseline_retention(self) -> float:
        return self._copy_err_baseline_retention

    def _dt_safe(self, dt_from_energy: float, dt_from_coherence: float,
                 delay_reliable: bool) -> bool:
        """True when copy is allowed from the double-talk perspective (per gate mode)."""
        m = self.gate_mode
        if m == self.GATE_ENERGY:
            return dt_from_energy < 0.3
        if m == self.GATE_COHERENCE:
            return dt_from_coherence < 0.4
        if m == self.GATE_COH_DELAY:
            return dt_from_coherence < 0.4 and delay_reliable
        if m == self.GATE_STREAK:
            return True
        return dt_from_energy < 0.3  # fallback: energy gate

    def update(self, *, shadow_frame_count: int, far_pwr: float,
               main_err_smooth: float, shadow_err_smooth: float,
               epc_active: bool, saturation_level: float,
               dt_from_energy: float,
               dt_from_coherence: float = 0.0,
               delay_reliable: bool = False) -> RegimeHandlerDecision:
        decision = RegimeHandlerDecision()
        if shadow_frame_count < self._shadow_copy_warmup_hops:
            return decision

        threshold = self.config.shadow_copy_threshold
        # Far-active threshold for the shadow/EPC gate.
        _far_thr = 1e-4
        far_active = far_pwr > _far_thr

        err_sum = main_err_smooth + shadow_err_smooth + 1e-10
        err_balance = abs(main_err_smooth - shadow_err_smooth) / err_sum
        is_stable_fs = far_active and err_balance < 0.3 and not epc_active
        if is_stable_fs:
            best_err = min(main_err_smooth, shadow_err_smooth)
            retention = self._copy_err_baseline_retention
            self._copy_err_baseline = (retention * self._copy_err_baseline
                                        + (1.0 - retention) * best_err)

        error_is_normal = main_err_smooth < self._copy_err_baseline * 4.0 + 1e-10
        not_saturating = saturation_level < 0.3
        # DT guard: during double-talk the shadow can chase near-end speech and look artificially good.
        dt_safe = self._dt_safe(dt_from_energy, dt_from_coherence, delay_reliable)
        copy_allowed = (far_active and error_is_normal
                        and not epc_active and not_saturating and dt_safe)

        # Streak-only mode: consecutive shadow-better frames, no hysteresis pair.
        if self.gate_mode == self.GATE_STREAK:
            if copy_allowed and shadow_err_smooth < main_err_smooth * threshold:
                self._streak += 1
                if self._streak >= self.AEC3_STREAK_FRAMES:
                    self._streak = 0
                    self._main_paused = True
                    self._pause_resume = self.config.epc_hangover
                    decision.boost_q = True
            else:
                self._streak = 0
            if self._main_paused:
                if self._pause_resume > 0:
                    self._pause_resume -= 1
                else:
                    self._main_paused = False
            if (copy_allowed
                    and main_err_smooth < shadow_err_smooth * threshold
                    and error_is_normal):
                decision.reverse_copy = True
            decision.pause_main = self._main_paused
            return decision

        # Energy / coherence gates: two-counter plus streak logic.
        if copy_allowed:
            if shadow_err_smooth < main_err_smooth * threshold:
                self._copy_counter += 1
                self._streak += 1
            else:
                self._copy_counter = 0
                self._streak = 0

            if (self._copy_counter >= self.config.shadow_copy_hysteresis
                    and self._streak >= self._hys_streak_min):
                self._copy_counter = 0
                self._streak = 0
                self._main_paused = True
                self._pause_resume = self.config.epc_hangover
                decision.boost_q = True

            if self._main_paused:
                if self._pause_resume > 0:
                    self._pause_resume -= 1
                else:
                    self._main_paused = False

            if (main_err_smooth < shadow_err_smooth * threshold
                    and error_is_normal):
                decision.reverse_copy = True
        else:
            self._copy_counter = 0
            self._streak = 0
            self._main_paused = False

        decision.pause_main = self._main_paused
        return decision
