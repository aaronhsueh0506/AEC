"""Echo-Path-Change family.

Extracted from ``aec.py`` during refactor R.7:

* ``classify_epc_event`` — utility mapping EpcEvent → AecEvent
* ``EchoPathChangeDetector`` — coherence-drop EPC detector
* ``PathChangeRegimeHandler`` — post-EPC adaptation regime selector
  (formerly ShadowCopyController, renamed in P52 Phase A)

Consumes value-type dataclasses from ``modules.dataclasses``.

Self-contained: numpy + modules.dataclasses.
"""
import numpy as np

from .dataclasses import AecEvent, AecEventType, EpcEvent, RegimeHandlerDecision


def classify_epc_event(epc_event: 'EpcEvent') -> AecEvent:
    """Phase F.1 classifier — map our 3-source EpcEvent to AEC3 AecEvent.

    Pure function, no state. Returns empty AecEvent (no-op) when the
    EpcEvent didn't fire. Callers should pass every EpcEvent regardless
    of `fired`, so the classification result follows trigger ordering.
    """
    if not epc_event.fired:
        return AecEvent()
    if epc_event.source == 'delay':
        return AecEvent(delay_change=AecEventType.DELAY_NEW_DETECTED)
    if epc_event.source in ('epv', 'shadow_rise'):
        return AecEvent(gain_change=True)
    return AecEvent()


class EchoPathChangeDetector:
    """Unified state for the three echo-path-change trigger sources.

    Triggers (caller invokes individually so call-site ordering matches v2.8.1):
        force_delay()        → delay-realignment trigger (caller-driven)
        update_epv(...)      → fast/slow far-power EMA divergence (±6 dB)
        update_shadow_rise(...)  → both filters' errors rising in tandem (post-converged)

    Caller applies side effects (Q-boost, P-override, ERL cap, render-forced,
    mark_diverged, dtd_coherence dampening) by inspecting EpcEvent.source.

    Hangover countdown is unified into a single counter; tick_hangover() is
    called once per frame inside the same `(shadow_filter and filter_converged)`
    gate the original code used, **and only when shadow_rise did not fire this
    frame** — preserving the original `if/elif/else` semantics bit-exactly.

    State (private):
        _active            currently in EPC (firing or hangover)
        _hangover          remaining hangover frames
        _epv_gain_fast     fast far-power EMA (TC ≈ 50 frames)
        _epv_gain_slow     slow far-power EMA (TC ≈ 1000 frames)
        _prev_total_err    last frame's main_err+shadow_err sum (for shadow_rise)
    """
    EPV_FAST_TC = 0.98
    EPV_SLOW_TC = 0.999
    EPV_LOW = 0.25
    EPV_HIGH = 4.0

    def __init__(self, config: 'AecConfig'):
        self.config = config
        self._active = False
        self._hangover = 0
        self._epv_gain_fast = 0.0
        self._epv_gain_slow = 0.0
        self._prev_total_err = 0.0

    def reset(self) -> None:
        self._active = False
        self._hangover = 0
        self._epv_gain_fast = 0.0
        self._epv_gain_slow = 0.0
        self._prev_total_err = 0.0

    def force_delay(self) -> EpcEvent:
        """Trigger 1: delay-shift re-alignment (caller already validated consistency)."""
        self._active = True
        self._hangover = self.config.epc_hangover
        return EpcEvent(fired=True, source='delay')

    def update_epv(self, *, far_pwr_global: float, filter_converged: bool,
                   main_paused: bool) -> EpcEvent:
        """Trigger 2: EPV gain-ratio. Updates fast/slow EMAs, fires on outliers."""
        if far_pwr_global <= 1e-6:
            return EpcEvent()
        if self._epv_gain_fast < 1e-12:
            self._epv_gain_fast = self._epv_gain_slow = far_pwr_global
        else:
            self._epv_gain_fast = (self.EPV_FAST_TC * self._epv_gain_fast
                                   + (1 - self.EPV_FAST_TC) * far_pwr_global)
            self._epv_gain_slow = (self.EPV_SLOW_TC * self._epv_gain_slow
                                   + (1 - self.EPV_SLOW_TC) * far_pwr_global)
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
        """Trigger 3: shadow-based error rise. Always updates _prev_total_err."""
        total_err = main_err_smooth + shadow_err_smooth
        if total_err > 1e-10:
            delta_ratio = abs(main_err_smooth - shadow_err_smooth) / total_err
        else:
            delta_ratio = 0.0
        errors_rising = (total_err > self._prev_total_err * self.config.epc_total_rise
                         and self._prev_total_err > 1e-10)
        is_echo_change = errors_rising and delta_ratio < self.config.epc_delta_threshold
        # White-noise guard: stationary far + error rise = DT, not echo path change
        if is_echo_change and is_stationary:
            is_echo_change = False
        self._prev_total_err = total_err
        if is_echo_change:
            self._active = True
            self._hangover = self.config.epc_hangover
            return EpcEvent(fired=True, source='shadow_rise')
        return EpcEvent()

    def tick_hangover(self) -> None:
        """Count down hangover and clear active when expired.

        Caller MUST gate this on the original (shadow_filter and filter_converged)
        guard, AND only call it when shadow_rise did not fire this frame.
        """
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
    """Catastrophe defence for cohort echo-path-nonstationarity outliers.

    Post-A.0 post-mortem reframe (2026-05-11, see docs/p52_a0_postmortem.md):
    this handler is **correct by design**, not a legacy artifact. P52 v1.1
    Task A.0 attempted to retire it on the premise that "shadow is just an
    information source." Empirically that premise is invalid on at least one
    cohort case (qNvSMyUSXUyrDGpOw7s6qg_farend_singletalk, p99.2 on
    ERL_decile_std): with the handler retired, the main filter diverges into
    anti-correlated output (W L2 grows large but wrong → ERLE_main −27 dB).
    The handler's boost_q / pause_main / reverse_copy actions force main W →
    0 on wildly-nonstationary echo paths, degrading the AEC to pass-through
    rather than catastrophe.

    The Yang-2017 / Jung-2011 R-reset-only mechanism does NOT cover this
    failure mode (their fast-recovery assumes a stable post-change path; the
    cohort tail has none — only W → 0 protects).

    Owns shadow-copy gate state machine. Per-frame inputs are read-only
    measurements; outputs are explicit RegimeHandlerDecision flags. Caller
    (AEC.process) applies the decisions: gate main_mu, apply filter Q-boost,
    perform shadow.copy_weights_from(main). The handler never mutates filter
    state itself.

    State (private):
        _copy_err_baseline : EMA of best(main, shadow) error during stable FS
        _copy_counter      : consecutive frames shadow < main * threshold
        _streak            : same as above (kept for parity with original two-counter logic)
        _main_paused       : main filter weight update currently frozen
        _pause_resume      : countdown to un-pause
    """

    BASELINE_INIT = 1e-6
    HYS_STREAK_MIN = 10  # additional streak gate beyond shadow_copy_hysteresis
    AEC3_STREAK_FRAMES = 5  # gate_mode='streak_only' uses pure 5-block AEC3 rule

    # Gate-mode choices for Phase C1 ablation. S0=energy is the v2.8.1 baseline.
    GATE_ENERGY = 'energy'                # S0: dt_from_energy < 0.3
    GATE_COHERENCE = 'coherence'          # S1: dt_from_coherence < 0.4
    GATE_COH_DELAY = 'coherence_delay'    # S2: S1 AND delay_reliable
    GATE_STREAK = 'streak_only'           # S3: AEC3 5-block, no DT gate

    def __init__(self, config: 'AecConfig', gate_mode: str = 'energy'):
        self.config = config
        self.gate_mode = gate_mode
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

    def _dt_safe(self, dt_from_energy: float, dt_from_coherence: float,
                 delay_reliable: bool) -> bool:
        """Resolve the per-mode DT-safety gate. Returns True when copy is allowed
        from a DT-perspective (NOT from far-active / saturation / EPC perspective)."""
        m = self.gate_mode
        if m == self.GATE_ENERGY:
            return dt_from_energy < 0.3
        if m == self.GATE_COHERENCE:
            return dt_from_coherence < 0.4
        if m == self.GATE_COH_DELAY:
            return dt_from_coherence < 0.4 and delay_reliable
        if m == self.GATE_STREAK:
            return True   # AEC3 mode: skip DT gate, rely on streak alone
        return dt_from_energy < 0.3  # fallback: legacy

    def update(self, *, shadow_frame_count: int, far_pwr: float,
               main_err_smooth: float, shadow_err_smooth: float,
               epc_active: bool, saturation_level: float,
               dt_from_energy: float,
               dt_from_coherence: float = 0.0,
               delay_reliable: bool = False) -> RegimeHandlerDecision:
        decision = RegimeHandlerDecision()
        if shadow_frame_count < 50:
            return decision

        threshold = self.config.shadow_copy_threshold
        far_active = far_pwr > 1e-4

        err_sum = main_err_smooth + shadow_err_smooth + 1e-10
        err_balance = abs(main_err_smooth - shadow_err_smooth) / err_sum
        is_stable_fs = far_active and err_balance < 0.3 and not epc_active
        if is_stable_fs:
            best_err = min(main_err_smooth, shadow_err_smooth)
            self._copy_err_baseline = (0.995 * self._copy_err_baseline
                                        + 0.005 * best_err)

        error_is_normal = main_err_smooth < self._copy_err_baseline * 4.0 + 1e-10
        not_saturating = saturation_level < 0.3
        # DT guard: shadow may chase near-end speech during DT, making shadow_err
        # artificially low. Gate selectable for Phase C1 ablation.
        dt_safe = self._dt_safe(dt_from_energy, dt_from_coherence, delay_reliable)
        copy_allowed = (far_active and error_is_normal
                        and not epc_active and not_saturating and dt_safe)

        # S3 streak-only mode: AEC3 5-block consecutive shadow-better rule, no
        # hysteresis pair. Skip the legacy two-counter logic and go directly
        # to a simple streak.
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

        # Legacy / coherence gates: original two-counter + streak logic.
        if copy_allowed:
            if shadow_err_smooth < main_err_smooth * threshold:
                self._copy_counter += 1
                self._streak += 1
            else:
                self._copy_counter = 0
                self._streak = 0

            if (self._copy_counter >= self.config.shadow_copy_hysteresis
                    and self._streak >= self.HYS_STREAK_MIN):
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
