"""AEC value-type dataclasses + lightweight event taxonomy.

Extracted from ``aec.py`` during refactor R.3. All types here are
plain dataclasses or class-level constant containers; no runtime
behaviour. Depends only on ``modules.enums`` for ``AecFilterState``.
"""
from dataclasses import dataclass
from typing import Optional

import numpy as np

from .enums import AecFilterState


@dataclass
class AecStats:
    """Comprehensive per-frame statistics for debugging and algorithmic decisions.

    Returned by AEC.get_stats() / AEC.GetStats().
    All power quantities are in dB (20·log10 for amplitudes, 10·log10 for power).
    Confidence scores are in [0, 1].
    """
    # ── Identity ──────────────────────────────────────────────────────────────
    frame_count: int        # Total frames processed since last reset()
    time_s: float           # Elapsed audio time (seconds)

    # ── Filter state ──────────────────────────────────────────────────────────
    # B2 note: filter_state is the *public* priority-ranked Kalman state
    # returned by get_filter_state() (AecFilterState enum).  This is distinct
    # from the internal P3f diagnostic string stored in _diag['filter_state']
    # / _prev_filter_state (values: 'idle', 'startup', 'diverged',
    # 'suspicious_dt', 'refined_usable', 'coarse_learning') which is a
    # separate finer-grained state machine used only inside the filter-state
    # computation block.  The two must never be confused.
    filter_state: AecFilterState  # public enum; use .value for string form
    filter_converged: bool
    filter_once_converged: bool  # True if converged at least once since reset
    warmup_remaining: int        # Frames remaining in warmup (0 when complete)

    # ── ERLE — Echo Return Loss Enhancement ───────────────────────────────────
    erle_inst_db: float       # EMA-smoothed instantaneous ERLE
    erle_windowed_db: float   # ~10 s decaying-window ERLE (more stable)
    erle_cumulative_db: float # Full-segment cumulative average ERLE

    # ── Echo path ─────────────────────────────────────────────────────────────
    erl_db: float        # Echo Return Loss estimate (dB); lower = harder (more coupling)
    divergence: float    # [0, 1] filter divergence indicator
    epc_active: bool     # Echo Path Change detector active
    epv_ratio: float     # Echo Path Variability: fast/slow gain ratio (≠1 → path changing)

    # ── Adaptation rate ───────────────────────────────────────────────────────
    mu_scale: float       # [mu_min, 1.0] current adaptation multiplier
    filter_w_norm: float  # Main filter weight L2 norm
    shadow_w_norm: float  # Shadow filter weight L2 norm

    # ── Double-talk detection ─────────────────────────────────────────────────
    dt_confidence: float    # Combined DT confidence [0, 1]
    dt_from_energy: float   # Pre-filter energy signal (immune to ERLE correction)
    dt_from_shadow: float   # Shadow filter DT signal
    dt_from_coherence: float# Coherence-based DT signal (fullband C²)
    dt_active: bool         # True when dt_confidence > 0.5

    # ── Energy levels ─────────────────────────────────────────────────────────
    far_power_db: float    # EMA far-end power (dB)
    mic_power_db: float    # EMA mic power (dB)
    error_power_db: float  # EMA raw filter output power (dB)
    far_activity: float    # [0, 1] far-end activity (from RES)
    saturation_level: float# [0, 1] speaker saturation estimate

    # ── Delay ─────────────────────────────────────────────────────────────────
    delay_samples: int  # Current estimated/fixed delay (−1 = not yet estimated)
    delay_ms: float     # Delay in milliseconds

    # ── Shadow filter ─────────────────────────────────────────────────────────
    shadow_advantage: float  # main_err / shadow_err; >1 means shadow is better
    shadow_copy_count: int   # Total shadow→main copies since reset()
    main_paused: bool        # True = main filter weight update frozen this frame

    # ── RES post-filter ───────────────────────────────────────────────────────
    res_gain_mean_db: float  # Mean spectral gain applied by RES (dB)
    res_using_render: bool   # True = render-based echo estimate active
    echo_psd_mean_db: float  # Mean residual echo PSD estimate (dB)
    error_psd_mean_db: float # Mean error PSD (dB)

    # Arc T cohort tail real-time detector signal. Placed at end of
    # dataclass to satisfy "default-args-after-non-default" rule.
    cohort_tail_T: bool = False


@dataclass
class AecResContext:
    """Per-frame context for external RES processing."""
    raw_output: np.ndarray       # (hop_size,) linear AEC output
    formed_output: np.ndarray    # (hop_size,) selected output underlying error_spec
    echo_spec: np.ndarray        # (n_freqs,) matching WOLA echo estimate
    far_power: float             # mean(far_end²)
    far_spec: np.ndarray         # (n_freqs,) complex far-end spectrum
    near_spec: np.ndarray        # (n_freqs,) matching WOLA mic spectrum (E + echo)
    filter_converged: bool
    erle_factor: float           # [0, 1] convergence metric
    dt_indicator: float          # [0, 0.8] double-talk confidence
    divergence: float            # [0, 1] divergence indicator
    over_sub: float              # dynamic over_sub value
    saturation_level: float
    erl_estimate: float = 0.01    # E2: dynamic ERL for external RES render-based
    # Freq-domain seam for an EXTERNAL residual-echo suppressor applied AFTER NR
    # (Audio_ALG AEC-linear → NR → RES). These carry the linear stage's own
    # windowed error spectrum + the AEC3 SuppressionGain it computed this frame,
    # so the post-NR stage is a pure freq multiply S(f)=error_spec·G_nr·res_gain
    # (+CNG) — reusing the tuned gain instead of re-deriving it. Populated only
    # when return_res_context=True; None on the default production path.
    error_spec: Optional[np.ndarray] = None    # (n_freqs,) reconstructing WOLA linear E(f)
    res_gain: Optional[np.ndarray] = None       # (n_freqs,) real AEC3 suppression gain G_res
    comfort_noise: Optional[np.ndarray] = None  # (n_freqs,) real CNG power N² (int16²-scaled)
    r2: Optional[np.ndarray] = None             # (n_freqs,) real residual-echo PSD R² (int16²-scaled)


@dataclass
class RenderActivityState:
    """One-frame summary of far-end activity for downstream consumers.

    far_pwr        mean(far²) + 1e-10 (always positive; safe denominator)
    is_active      latched: True once far has been audible (>1e-6) since last silence
    is_stationary  CV² of far envelope < 0.02 → stationary (white-noise-like)
    warmup_active  raw mean(far²) > 1e-6 (used to gate warmup-frame consumption)
    """
    far_pwr: float
    is_active: bool
    is_stationary: bool
    warmup_active: bool


@dataclass
class FilterConvergenceState:
    """One-frame snapshot of filter convergence health."""
    converged: bool
    once_converged: bool
    just_converged: bool   # True for the single frame the transition fires
    divergence: float      # [0, 1] EMA: rate of post-convergence inst-ERLE < -2 dB


@dataclass
class RegimeHandlerDecision:
    """One-frame decision emitted by PathChangeRegimeHandler.

    pause_main:    main filter weight update is gated off this frame
    boost_q:       this frame triggered the pause; caller should boost Q on main filter
    reverse_copy:  this frame, shadow should be re-synced from main (main-winning case)
    """
    pause_main: bool = False
    boost_q: bool = False
    reverse_copy: bool = False


class AecEventType:
    """AEC3-aligned echo-path event constants.

    Mirrors `webrtc::EchoPathVariability::DelayAdjustment` taxonomy
    (echo_path_variability.h):
      DELAY_NONE          — no delay adjustment
      DELAY_BUFFER_FLUSH  — render-buffer flush; alignment recovery in-flight
      DELAY_NEW_DETECTED  — new alignment found by delay estimator

    F.1 maps our existing source strings:
      EpcEvent.source='delay'        → DELAY_NEW_DETECTED
      EpcEvent.source='epv'          → gain_change=True
      EpcEvent.source='shadow_rise'  → gain_change=True

    Buffer-flush and clock_drift have no current detector.
    """
    DELAY_NONE = 'none'
    DELAY_BUFFER_FLUSH = 'buffer_flush'
    DELAY_NEW_DETECTED = 'new_detected'


@dataclass
class AecEvent:
    """One-frame classified echo-path event (AEC3-aligned 3-tuple).

    Mirrors `webrtc::EchoPathVariability` (echo_path_variability.h):
      gain_change    — amplitude/level change. AEC3 response: ERLE-only soft
                       reset (subtractor.HandleEchoPathChange gain-only path
                       + aec_state ERLE reset(false)).
      delay_change   — temporal alignment change (3-way enum). AEC3 response:
                       full cascade reset (filter weights, gains, partition,
                       filter_analyzer, transparent, ERLE, ERL, FQA).
      clock_drift    — render/capture clock skew. AEC3 response: no direct
                       reset; downstream consumers gated.

    Phase F.1 wires classification only; consumers in F.2+ implement the
    asymmetric reset cascade. Until then, `_classified_event` is trace-only
    when the gating flag is ON, no-op when OFF.
    """
    gain_change: bool = False
    delay_change: str = AecEventType.DELAY_NONE
    clock_drift: bool = False

    @property
    def audio_path_changed(self) -> bool:
        return self.gain_change or self.delay_change != AecEventType.DELAY_NONE


@dataclass
class EpcEvent:
    """One-frame EPC trigger result.

    fired:  True if this frame fired one of the three triggers (delay/epv/shadow_rise)
    source: which trigger fired ('delay' | 'epv' | 'shadow_rise')
    """
    fired: bool = False
    source: str = ''
