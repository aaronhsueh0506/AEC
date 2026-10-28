"""AEC value types: statistics, RES context, per-frame state records and the event taxonomy."""
from dataclasses import dataclass
from typing import Optional

import numpy as np

from .enums import AecFilterState


@dataclass
class AecStats:
    """Per-frame statistics returned by ``AEC.get_stats()``; powers in dB, confidences in [0, 1]."""
    frame_count: int        # Total frames processed since last reset()
    time_s: float           # Elapsed audio time (seconds)

    # Public priority-ranked filter state (distinct from the finer-grained diagnostic string).
    filter_state: AecFilterState  # public enum; use .value for string form
    filter_converged: bool
    filter_once_converged: bool  # True if converged at least once since reset
    warmup_remaining: int        # Frames remaining in warmup (0 when complete)

    erle_inst_db: float       # EMA-smoothed instantaneous ERLE
    erle_windowed_db: float   # decaying-window ERLE, time constant 1000 hops (more stable)
    erle_cumulative_db: float # Full-segment cumulative average ERLE

    erl_db: float        # Echo Return Loss estimate (dB); lower = harder (more coupling)
    divergence: float    # [0, 1] filter divergence indicator
    epc_active: bool     # Echo Path Change detector active
    epv_ratio: float     # Echo Path Variability: fast/slow gain ratio (≠1 → path changing)

    mu_scale: float       # [mu_min, 1.0] current adaptation multiplier
    filter_w_norm: float  # Main filter weight L2 norm
    shadow_w_norm: float  # Shadow filter weight L2 norm

    dt_confidence: float    # Always 0.0; see dt_from_energy / dt_from_shadow
    dt_from_energy: float   # Pre-filter energy signal (immune to ERLE correction)
    dt_from_shadow: float   # Shadow filter DT signal
    dt_from_coherence: float# Coherence-based DT signal (fullband C²)
    dt_active: bool         # Always False

    far_power_db: float    # EMA far-end power (dB)
    mic_power_db: float    # EMA mic power (dB)
    error_power_db: float  # EMA raw filter output power (dB)
    far_activity: float    # [0, 1] far-end activity (from RES)
    saturation_level: float# [0, 1] speaker saturation estimate

    delay_samples: int  # Current estimated/fixed delay (0 until estimated)
    delay_ms: float     # Delay in milliseconds

    shadow_advantage: float  # main_err / shadow_err; >1 means shadow is better
    shadow_copy_count: int   # Total shadow→main copies since reset()
    main_paused: bool        # True = main filter weight update frozen this frame

    res_gain_mean_db: float  # Mean spectral gain applied by RES (dB)
    res_using_render: bool   # True = render-based echo estimate active
    echo_psd_mean_db: float  # Mean residual echo PSD estimate (dB)
    error_psd_mean_db: float # Mean error PSD (dB)

    # Tail-detector signal (always False).
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
    over_sub: float              # equals erle_factor
    saturation_level: float
    erl_estimate: float = 0.01    # dynamic ERL for external render-based RES
    # Frequency-domain seam for an external residual-echo suppressor applied after NR: the
    # linear stage's windowed error spectrum and the suppression gain computed this frame.
    # Populated only when return_res_context=True.
    error_spec: Optional[np.ndarray] = None    # (n_freqs,) reconstructing WOLA linear E(f)
    res_gain: Optional[np.ndarray] = None       # (n_freqs,) real AEC3 suppression gain G_res
    comfort_noise: Optional[np.ndarray] = None  # (n_freqs,) real CNG power N² (int16²-scaled)
    r2: Optional[np.ndarray] = None             # (n_freqs,) real residual-echo PSD R² (int16²-scaled)
    # This frame's near-end-latch double-talk floor decision.
    res_floor_protect: bool = False
    # This frame's usable-linear-estimate verdict; None when there is none (e.g. a context fused from several lanes).
    usable_linear: Optional[bool] = None
    # Rendering base (copy of the output-capture guard's choice): error_spec, or error_spec + echo_spec (= near_spec)
    # on a hop where the guard fires. Render S(f) = output_base_spec * gain (+ CNG). It is NOT paired with
    # formed_output / error_spec (STFT / unity-WOLA pairing holds for the analysis seam only), and the guard never
    # replaces error_spec, echo_spec, near_spec or formed_output.
    output_base_spec: Optional[np.ndarray] = None   # (n_freqs,) complex64
    # True when this hop's rendering base is the capture Y (guard fired: usable verdict false and sum|E|^2 > sum|Y|^2).
    guard_active: bool = False
    # Hard post-chain reset counter (uint32, wraps) as of this hop; see AEC.get_post_reset_serial().
    post_reset_serial: int = 0


@dataclass
class RenderActivityState:
    """One-frame far-end activity summary: far_pwr = mean(far^2) + 1e-10; is_active latches
    once far is audible (> 1e-6) until silence; is_stationary = envelope CV^2 < 0.02;
    warmup_active = raw mean(far^2) > 1e-6."""
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
    """One-frame decision from PathChangeRegimeHandler: pause_main gates the main update;
    boost_q (the frame that triggered the pause) asks the caller to boost Q; reverse_copy
    asks for the shadow to be re-synced from main."""
    pause_main: bool = False
    boost_q: bool = False
    reverse_copy: bool = False


class AecEventType:
    """Echo-path delay-adjustment constants: DELAY_NONE, DELAY_BUFFER_FLUSH, DELAY_NEW_DETECTED.

    EpcEvent source 'delay' maps to DELAY_NEW_DETECTED; 'epv' and 'shadow_rise' map to gain_change.
    """
    DELAY_NONE = 'none'
    DELAY_BUFFER_FLUSH = 'buffer_flush'
    DELAY_NEW_DETECTED = 'new_detected'


@dataclass
class AecEvent:
    """One-frame classified echo-path event: ``gain_change`` (level change: ERLE-only soft
    reset), ``delay_change`` (alignment change: full reset) and ``clock_drift`` (consumers gated)."""
    gain_change: bool = False
    delay_change: str = AecEventType.DELAY_NONE
    clock_drift: bool = False

    @property
    def audio_path_changed(self) -> bool:
        return self.gain_change or self.delay_change != AecEventType.DELAY_NONE


@dataclass
class EpcEvent:
    """One-frame EPC trigger result: ``fired`` and ``source`` ('delay' | 'epv' | 'shadow_rise')."""
    fired: bool = False
    source: str = ''
