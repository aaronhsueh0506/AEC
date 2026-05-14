"""
Acoustic Echo Cancellation (AEC) - Python Reference Implementation

Supports five filter modes:
- Time-domain NLMS (--mode nlms): sample-by-sample, lowest latency
- Frequency-domain Adaptive Filter (--mode fdaf): single FFT block, no partitions
- Partitioned Block FDAF (--mode pbfdaf): multiple partitions, NLMS adaptation
- Partitioned Block FDKF (--mode pbfdkf): multiple partitions, Kalman adaptation (recommended)

Additional features:
- Double-Talk Detection (DTD)
- Residual Echo Suppressor (RES)

Usage:
    python aec.py mic.wav ref.wav output.wav [--mode nlms|fdaf|pbfdaf|pbfdkf] [--enable-res]
"""

__version__ = "3.14.0-s-orth-a"

import os
import numpy as np
from collections import deque
from dataclasses import dataclass, field
from typing import List, Optional, Tuple
from enum import Enum
import argparse
import soundfile as sf


class AecMode(Enum):
    LMS = "lms"         # Time-domain LMS (no normalization, simplest)
    NLMS = "nlms"       # Time-domain NLMS (sample-by-sample)
    FDAF = "fdaf"       # Frequency-domain Adaptive Filter (single block, n_partitions=1)
    PBFDAF = "pbfdaf"   # Partitioned Block FDAF (NLMS adaptation)
    PBFDKF = "pbfdkf"   # Partitioned Block FDKF (Kalman adaptation, recommended)
    SUBBAND = "subband" # Backward compat alias (= PBFDKF)


# Frequency-domain modes (partitioned or single block)
_FREQ_MODES = (AecMode.FDAF, AecMode.PBFDAF, AecMode.PBFDKF, AecMode.SUBBAND)
# Partitioned block modes
_PB_MODES = (AecMode.PBFDAF, AecMode.PBFDKF, AecMode.SUBBAND)

# F3.1-v3: blend weight between mic-excess-ratio and legacy (1-coh²) in dt_per_bin.
# 0.7 keeps excess as dominant signal while capping swing under ERL underestimation.
_BLEND_F31_MIC_EXCESS = 0.7


class AecPreset(Enum):
    MILD = "mild"               # Lightest echo suppression, best near-end preservation
    SOFT = "soft"               # Between MILD and BALANCED — gentler RES for music / sensitive NE
    BALANCED = "balanced"       # Balanced echo suppression and near-end quality (default)
    AGGRESSIVE = "aggressive"   # Stronger echo suppression, moderate near-end impact
    MAXIMUM = "maximum"         # Maximum echo suppression, significant near-end impact


class AecFilterState(Enum):
    """Algorithmic state of the adaptive filter, in priority order."""
    WARMUP         = "warmup"          # Pre-convergence: warmup frames not yet exhausted
    DIVERGED       = "diverged"        # Filter diverged (output >> mic), adaptation unreliable
    EPC_RECOVERY   = "epc_recovery"    # Echo path changed, filter re-converging (P_MAX raised)
    DT_ACTIVE      = "dt_active"       # Double-talk detected, adaptation frozen / slowed
    STATIONARY_FAR = "stationary_far"  # Stationary far-end (white noise / fan), special gating
    CONVERGED      = "converged"       # Stable convergence, good echo cancellation
    CONVERGING     = "converging"      # Adapting, not yet converged


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


@dataclass
class AecResContext:
    """Per-frame context for external RES processing."""
    raw_output: np.ndarray       # (hop_size,) linear AEC output
    echo_spec: np.ndarray        # (n_freqs,) complex echo estimate
    far_power: float             # mean(far_end²)
    far_spec: np.ndarray         # (n_freqs,) complex far-end spectrum
    near_spec: np.ndarray        # (n_freqs,) complex mic spectrum
    filter_converged: bool
    erle_factor: float           # [0, 1] convergence metric
    dt_indicator: float          # [0, 0.8] double-talk confidence
    divergence: float            # [0, 1] divergence indicator
    over_sub: float              # dynamic over_sub value
    saturation_level: float
    erl_estimate: float = 0.01    # E2: dynamic ERL for external RES render-based


@dataclass
class AecConfig:
    """AEC Configuration (all sizes in samples)"""
    sample_rate: int = 16000      # 8000 / 16000 / 48000
    frame_size: int = -1          # Auto: sample_rate * 20ms (160@8k, 320@16k, 960@48k)
    hop_size: int = -1            # Auto: frame_size / 2 (80@8k, 160@16k, 480@48k)
    filter_length: int = -1      # Auto: 32ms (8k/16k) or 64ms (48k)
    mu: float = 0.3              # Step size
    delta: float = 1e-8          # Regularization
    enable_dtd: bool = False
    dtd_hangover_frames: int = 15

    # Geigel DTD parameters (LMS/NLMS)
    dtd_geigel_threshold: float = 0.5     # |mic| > thresh × max(|ref|) → double-talk
    dtd_mu_min_ratio: float = 0.05        # During double-talk, mu drops to 5%
    dtd_confidence_attack: float = 0.3    # Confidence ramp-up rate per block
    dtd_confidence_release: float = 0.05  # Confidence ramp-down rate per block

    # Divergence detection parameters (frequency-domain modes, output-vs-input)
    dtd_divergence_factor: float = 1.5    # output > input × factor → diverged

    # Coherence-based DTD parameters (frequency-domain modes, complements divergence)
    dtd_coh_alpha: float = 0.85           # PSD smoothing factor (~6 block time constant)
    dtd_coh_high: float = 0.6            # Coherence above → no DT (correlated error)
    dtd_coh_low: float = 0.3             # Coherence below → DT (uncorrelated error)
    dtd_coh_energy_floor: float = 0.1    # Min error/far energy ratio to trigger DT
    dtd_coh_hangover: int = 3            # Coherence DTD hangover blocks (shorter than Geigel)
    dtd_coh_release: float = 0.1         # Coherence confidence release rate (faster recovery)

    # RES parameters
    enable_res: bool = True
    res_g_min_db: float = -25.0
    res_over_sub: float = 3.0
    res_alpha: float = 0.8
    enable_cng: bool = False           # Comfort noise generation in RES (off by default)
    enable_td_constraint: bool = True  # Time-domain constraint on filter weights
    # P52 Phase B: route RES through `ResFilterRefactored` (subclass that
    # delegates each `_stage_*` to a free function in `python/res_refactored/`).
    # 800-case byte-equal verified (Task B.4). Default-OFF until a future
    # phase retires the legacy methods.
    use_res_refactored: bool = False

    # Shadow filter (dual-filter divergence control, frequency-domain modes only)
    enable_shadow: bool = True
    shadow_mu_ratio: float = 1.0
    shadow_copy_threshold: float = 0.65
    shadow_err_alpha: float = 0.80      # D3: 0.85→0.80, faster shadow EMA tracking
    shadow_mu_min: float = 0.5           # Shadow-only mode: DT mu floor (50%)
    shadow_copy_hysteresis: int = 3     # Consecutive frames needed for copy
    shadow_q_ratio: float = 3.0        # Shadow Q = main Q × ratio (FDKF mode)
    shadow_dtd_advantage_scale: float = 3.0  # Shadow DTD: (advantage-offset)/scale → DT confidence
    shadow_dtd_offset: float = 1.5           # Shadow DTD: advantage must exceed this to signal DT

    # Coherence DTD absolute energy floor
    dtd_coh_abs_floor: float = 1e-6     # #8: Absolute error energy floor

    # PBFDKF (Partitioned Block Frequency Domain Kalman Filter) — faster convergence than NLMS
    use_kalman: bool = True           # True=PBFDKF, False=PBFDAF (NLMS)
    kalman_q_high: float = 1e-3     # PBFDKF Q_high convergence speed
    kalman_q_low: float = 1e-6        # D6: 1e-5→1e-6, lower misadjustment (P_floor=1e-4 protects)
    warmup_frames: int = 80          # Frames with forced high mu at startup

    # Echo path change detection (requires shadow filter)
    epc_delta_threshold: float = 0.3    # |ΔE/total_E| < threshold → echo change
    epc_total_rise: float = 1.5         # total_err > prev × rise → errors increasing
    epc_hangover: int = 20              # keep EPC active for N frames after detection
    epc_mu_floor: float = 0.5           # mu_scale floor during EPC

    # Delay estimation (GCC-PHAT). v3.10.4: max_delay_ms 512 → 1024 to
    # match WebRTC's older AEC (modules/audio_processing/aec/) which keeps a
    # ~1 s far-end history and lets the binary delay estimator search the
    # full window. AEC3 uses 512 ms because typical desktop/laptop skew is
    # well under 200 ms; AEC challenge mobile/BT cases (e.g. ~788 ms skew)
    # need the wider Old-AEC range. Render ring buffer sized to 2048 ms to
    # leave headroom for max_delay_ms = 1024 + drift + tracking jitter.
    enable_delay_est: bool = True       # Enable automatic delay estimation + ref alignment
    max_delay_ms: float = 1024.0        # Maximum delay to search (ms)
    delay_buffer_ms: float = 2048.0     # Render ring buffer total capacity (ms)
    delay_est_period_s: float = 0.5     # Re-estimate delay every N seconds
    delay_est_init_s: float = 0.3       # Accumulate this much data before first estimate
    fixed_delay_samples: int = -1       # If >= 0, use this fixed delay instead of estimation
    # Confidence threshold below which mu_scale is held at a higher floor
    # (don't learn aggressively while delay alignment is uncertain).
    delay_par_low_threshold: float = 5.0     # PAR < this → low confidence
    delay_par_solid_threshold: float = 8.0   # PAR > this → high confidence

    # High-pass filter (DC blocker + low-freq removal)
    enable_highpass: bool = True
    highpass_cutoff_hz: float = 80.0    # Cutoff freq: removes DC, 50/60Hz hum, rumble

    # Saturation / non-linear echo handling
    enable_saturation_detect: bool = True
    saturation_threshold: float = 0.95       # |sample| > threshold → clipping
    saturation_over_sub_boost: float = 3.0   # Extra over_sub during saturation
    saturation_softclip_ref: bool = True     # Soft-clip reference for better filter modeling

    # RES dynamic over_sub formula: base + scale × erle_factor - dt_reduction × dt_indicator
    res_over_sub_base: float = 2.5           # Unconverged over_sub base
    res_over_sub_scale: float = 4.0          # Scale with erle_factor (converged adds this)
    res_dt_reduction: float = 3.5            # DT reduction coefficient

    # RES anti-blackout
    res_max_drop_db_per_frame: float = 6.0   # Max gain drop per frame (dB)
    res_max_rise_db_per_frame: float = 6.0   # Max gain rise per frame (dB)
    res_spectral_floor: bool = True          # Spectral-shape-preserving gain floor
    res_spectral_floor_db: float = -25.0     # Floor relative to spectral envelope
    res_ne_protect_db: float = -10.0         # Per-bin near-end protection ceiling (dB)

    # RES v2: direct echo estimation + Wiener gain + reverb tail
    res_echo_method: str = "direct"           # "coherence" (legacy) or "direct" (use filter echo est)
    res_gain_type: str = "enr"               # "spectral_sub" (legacy) / "wiener" / "enr"
    res_enable_reverb: bool = False          # Reverb tail model
    res_reverb_decay: float = 0.5            # Exponential decay rate
    res_reverb_gain: float = 1.0             # Reverb contribution scale
    res_alpha_echo_psd: float = 0.7          # Echo PSD smoothing (overridable per preset)
    res_alpha_error_psd: float = 0.8         # Error PSD smoothing (overridable per preset)
    res_enr_scale: float = 1.0              # ENR threshold scale (1.0=AEC3 defaults, <1=more aggressive)
    startup_dt_min_ne_scale: float = 1.0   # Scale min_ne_from_dt when startup_dt (not once_converged); 0.0=disable floor
    startup_dt_gain_floor: float = 1.0    # Cap spectral_g_min during startup_dt (not filter_converged); 1.0=no effect
    startup_dt_noise_floor_scale: float = 1.0  # Scale noise_floor_gain during startup_dt; 0.0=bypass, 1.0=normal
    startup_dt_mu_min: float = 0.0            # Floor mu_scale during startup_dt; 0.0=no override

    # Diagnostic: when True, ResFilter populates per-bin gain vectors at each
    # post-processing stage (read via res.get_stage_gains()). Hot-path cost is
    # one numpy copy per stage per frame; off by default.
    capture_stages: bool = False

    # P1 Phase 1: trace high-band NE evidence metrics into self._diag.
    # Off by default — purely instrumentation, no behaviour change.
    trace_high_band_metrics: bool = False

    # P3b: per-call DelayEstimator instrumentation. When True, every call
    # to DelayEstimator.accumulate() appends one diagnostic row (input
    # power, top-3 peaks of the running EMA cross-spectrum, state flags)
    # to a list accessible at AEC.delay_est._trace_rows. Off by default —
    # zero overhead and no behaviour change in the released v3.10.4 path.
    trace_delay_est: bool = False
    trace_delay_est_path: str = ""  # if non-empty, CSV is flushed here at AEC destruction or via dump method

    # P52 A.0R.2: per-frame regime handler observability. When True, AEC
    # records one row per frame into AEC._regime_trace_rows with the handler
    # decision flags (boost_q_fired / reverse_copy_fired / main_paused_fired)
    # plus main filter W L2 norm before/after the per-frame update, and
    # ERLE_main before/after. Default False — zero overhead and no behaviour
    # change. Audio-passive: trace reads existing state, never mutates.
    trace_p52_regime_handler: bool = False
    trace_p52_regime_handler_path: str = ""  # if non-empty, dump to .npz on demand

    # P53 Step 0: per-frame innovation-sequence statistics on PBFDKF main
    # filter, for offline Kalman orthogonality audit (`r = C_obs / C_exp`).
    # Default False — zero overhead. Audit-only: never reads in production.
    # See docs/p53_design_lock.md §2.
    trace_p53_innovation: bool = False

    # P1.0 Plan A internal attribution toggles. Default True keeps v3.10.4
    # release behaviour. Setting False reverts the corresponding sub-change
    # to the v3.8.3 baseline behaviour, used for isolating each Plan A
    # sub-change's contribution to the FS regression.
    plan_a_kernel_tight: bool = True       # False → smoothing kernel [0.25,0.5,0.25]
    plan_a_hf_cap_2k: bool = True          # False → HF cap anchor 500 Hz with v3.8.3 gate
    plan_a_stat_mask_7k: bool = True       # False → stat_dt_mask cuts off at 4 kHz

    # P4B — γ²(k)-primary dt_per_bin (default OFF; A/B candidate).
    # Replaces dt_per_bin = max(effective_dt, 1-coh2) with γ²-only base
    # plus a soft floor that lifts only when effective_dt > 0.5. Aim:
    # remove uniform frame-scalar lift in the ambiguous DTD regime
    # (effective_dt 0.2–0.5) so per-bin γ²(k) discrimination survives.
    plan_b_dt_per_bin_gamma: bool = False

    # F3.1 — per-bin mic-energy excess evidence (default OFF; A/B candidate).
    # Replaces the `(1 - coh2)` term in dt_per_bin with the validated
    # `max(error_psd - far_lw·ERL_est, 0) / error_psd` per-bin metric.
    # `(1 - coh2)` saturates to 1 in FS post-cancellation (echo cancelled →
    # residual decorrelated → low coh2 → "NE-like"), an acknowledged
    # dead-code symptom flagged in the HF-cap comment at ~line 2060. The
    # excess-ratio metric is mic-energy-based, immune to that saturation,
    # and was validated in P1 Phase 1 with AUROC 0.871 (FS-vs-NE+DT_positive)
    # for the HF-cap gate; F3.1 extends the same metric per-bin to the
    # primary dt_per_bin axis. Gated on `filter_converged AND
    # _long_window_n_updates > 0` so the metric is only used when
    # `erl_estimate` and `far_lw` are reliable; falls back to the legacy
    # path (γ² primary OR coh2 primary) otherwise. Requires
    # `res_echo_method="direct"` (the balanced default) since
    # `_long_window_far_psd` is only maintained on that path.
    use_mic_excess_evidence: bool = False

    # F2.1 — reset stale upstream state on EPC fire (default OFF).
    # The shipped EPC path (EPV at aec.py:~5107 and shadow_rise at
    # ~5128) already caps `_erl_estimate` to min(0.3) and boosts Q/P,
    # but does NOT reset:
    #   • `_erl_estimate` — EMA α=0.999 post-conv, TC ≈ 10s; can stay
    #     pinned to the previous-room value for a full sec after a path
    #     change, breaking any consumer that multiplies it against
    #     far_psd (RES residual model, F3.1 metric, render-ceil cap).
    #   • `_erle_window_near` / `_erle_window_err` — α=0.999 windowed
    #     accumulators (TC ≈ 10s) feeding `erle_factor`; persist old
    #     room's ERLE shape across the change.
    #   • `_wn_err_baseline` — speech-frozen with α=0.999 inside DT
    #     hangover; if the freeze caught the previous room, post-change
    #     "jump" detection mis-fires as DT.
    # F2.1 resets all three to init values on the rising edge of
    # `_epc_det.active`. Default OFF keeps v3.10.4 byte-identical.
    # Companion to F3.1: F3.1's excess metric relies on
    # `_erl_estimate` accuracy; the FS_movement worst-case -0.301 in
    # the F3.1 Phase 1 verdict (docs/f3_1_phase1_verdict.md) traced to
    # erl staleness during movement-induced path change.
    use_epc_state_reset: bool = False

    # P1 Phase 2 — conditional HF cap based on m_excess_ratio (validated
    # in P1 Phase 1 with AUROC 0.871 on FS-vs-NE+DT_positive). When
    # hf_cap_conditional is True, the stage-6 cap behaviour is gated by
    # the per-frame metric: metric < threshold → apply v3.8.3 strict cap
    # (anchor 500 Hz, gate effective_dt < 0.5); metric >= threshold →
    # skip the cap entirely (preserve high-band NE). Default off keeps
    # v3.10.4 release behaviour. plan_a_hf_cap_2k still controls the
    # baseline cap mode used when the conditional path is disabled.
    hf_cap_conditional: bool = False
    hf_cap_metric_threshold: float = 0.30

    # P3e — DT advisory gate for adaptation protection (post-release work).
    # Independent of `enable_dtd`. When True, shadow-DTD and energy-DTD
    # signals (already smoothed by DoubleTalkAnalyzer) are routed to:
    #   • reduce mu_scale by `dt_advisory_mu_factor` during DT
    #   • inhibit shadow→main copies during DT
    # Hysteresis: hold for `dt_advisory_hold_ms` after the last hit so
    # per-frame flicker doesn't pop in and out of protection. Does NOT
    # touch RES suppression, residual echo estimate, or HF cap; the
    # 7GT P3d trace showed that the underlying problem is filter taps
    # being learnt against NE-contaminated error during DT, not the
    # post-filter. Default off keeps v3.10.4 release behaviour.
    dt_advisory_enabled: bool = False
    dt_advisory_shadow_th: float = 0.5
    dt_advisory_energy_th: float = 0.4
    dt_advisory_hold_ms: float = 400.0
    dt_advisory_mu_factor: float = 0.3
    # P3f Phase 3: when True, gate fires on
    # `filter_state == 'suspicious_dt'` instead of the raw dt_shadow /
    # dt_energy thresholds above. Phase 2a invariants showed the
    # state-based gate is FS-safe by construction (suspicious_dt requires
    # refined_latched + NE evidence + main_err jump + shadow lead, AND'd).
    dt_advisory_use_p3f_state: bool = False

    # P3h — filter reset on sustained diverged (high-impact action,
    # 7GT-only dry-run only; default off, never enabled by preset).
    # Triggered when ALL hold:
    #   • once_converged is set (we never reset a filter that has never
    #     been good — startup / coarse_learning is not the target)
    #   • _p3f_diverged_streak >= diverged_reset_streak_frames (sustained)
    #   • cooldown timer has expired
    # On fire: _reset_filter_derived_state(reason='p3h_diverged') and
    # cooldown timer set so we don't loop-reset on a flaky classifier.
    diverged_reset_enabled: bool = False
    diverged_reset_streak_frames: int = 50          # ~500 ms at hop=160 / 16 kHz
    diverged_reset_cooldown_frames: int = 400        # ~4 s minimum gap

    # F2.2 — EMA-based diverged-streak gate (plan ~/.claude/plans/se-aec-aec-main-hazy-lynx.md).
    # Legacy `_p3f_diverged_streak` is a hard counter that resets to 0 on any
    # frame below the 1.3 ratio threshold; when main_err_ratio oscillates
    # around 1.3 it never accumulates to the 50-frame `diverged_reset_streak_frames`
    # bar, so the P3h reset action becomes dead code. F2.2 replaces the
    # reset-gate streak check with an EMA tracker (α=0.95 ⇒ TC ≈ 20 frames
    # ≈ 200 ms at hop=160) so noisy frames don't fully reset the evidence.
    # Default OFF — opt-in flag, paired with `diverged_reset_enabled=True`.
    # Classifier-state hard-counter at `_diverged_min_streak=5` is untouched
    # (the noisy 1-frame fast path still tags 'diverged' for diagnostics).
    use_diverged_streak_ema: bool = False
    diverged_streak_ema_alpha: float = 0.95
    diverged_streak_ema_threshold: float = 0.7

    # F1.2 ablation: PathChangeRegimeHandler gate mode selector.
    # 'energy' = legacy dt_from_energy < 0.3 gate (production default).
    # 'streak_only' = AEC3-style 5-block consecutive streak, no DT gate.
    regime_gate_mode: str = 'energy'

    # F2.3 — Yang 2017 R-reset on EPC (plan ~/.claude/plans/se-aec-aec-main-hazy-lynx.md).
    # When boost_q fires (echo path change detected), also reset PBFDKF._error_psd
    # and R to their initial value (1e-2). Over-estimated R from a prior DT period
    # suppresses Kalman gain K, causing slow reconvergence after an EPC event.
    # Default OFF — opt-in ablation flag.
    epc_r_reset_enabled: bool = False

    # B5 — shadow R-reset symmetric extension (v3.11 Phase 1 Sprint 1-2).
    # F2.3 resets only the main filter's R/_error_psd; shadow retains its stale
    # R from the same DT period. When the regime handler subsequently fires
    # reverse_copy (shadow→main), the K-handicapped shadow's W is transplanted
    # into main, undoing the F2.3 fast-recovery benefit. Symmetric reset (both
    # filters) closes this asymmetry. Effective only when shadow exists.
    # Default OFF — opt-in ablation flag.
    shadow_r_reset_enabled: bool = False

    # B6 — state-aware shadow µ schedule (v3.11 Phase 1 Sprint 3-4).
    # Current shadow_mu_scale is binary (1.0 or 0.1) based on far_excited +
    # saturation_safe only; it ignores main_paused and the filter_state
    # classification. When main is paused (regime handler trying to let W
    # decay), shadow keeps adapting at full speed; when filter_state is
    # diverged, shadow trains on a broken main reference. Both cases poison
    # the next reverse_copy.
    # B6 adds 4-band schedule: 0 when main_paused or filter_state=='diverged';
    # 0.1 when far weak or saturating; 0.5 when filter_state=='suspicious_dt'
    # (DT caution); 1.0 normal. Uses previous frame's filter_state since the
    # state classifier runs at end of process().
    # Default OFF — opt-in ablation flag.
    shadow_mu_state_aware: bool = False

    # F-E1 — ERL clip range + far_active hysteresis (v3.11 Phase 1 Sprint 5-6).
    # Edge case E1: ref/mic energy difference too large.
    #   E1-1: current `np.clip(inst_erl_raw, 0.001, 1.0)` clamps real ERL=0.0005
    #         (extreme high coupling) to 0.001, biasing mic_excess metric
    #         downstream. Extend lower bound to 1e-5 to admit extreme cases.
    #   E1-3: `far_pwr > 1e-4` is a single-threshold gate; far power near 1e-4
    #         bounces in/out of "active", causing ERL updates to stall in
    #         marginal-reference scenarios. Add fast-attack / slow-release
    #         hysteresis (attack at 1e-4, release after 5 frames < 3e-5).
    # Default OFF — opt-in ablation flag.
    f_e1_enabled: bool = False

    # F-DelayTrack — continuous delay variance tracking (v3.11 Phase 1 Sprint 7-8).
    # Edge case E2: echo path change / movement (gradual delay creep).
    # Current `delay_reliable = self.delay_est.confidence >= 0.5` is a hard
    # cut on PAR-derived confidence. PAR is noisy during movement (multiple
    # peaks compete) → confidence oscillates → reliability gate bounces.
    # F-DelayTrack replaces with variance-based stability: track the last 8
    # delay estimates, compute std; gate true when std < 4 samples (0.25 ms
    # at 16 kHz) AND confidence >= 0.3 (relaxed minimum since variance is
    # the primary stability signal). Switchboard AEC3 continuous-tracking
    # pattern; helps movement / gradual-creep scenarios where instantaneous
    # PAR is misleading.
    # Default OFF — opt-in ablation flag.
    f_delaytrack_enabled: bool = False

    # F-E3 — consecutive-EPC hangover extension + W partial reset
    # (v3.11 Phase 1 Sprint 9-10). Edge case E3: echo gain change (blind
    # test pattern — mid-clip gain ±N dB then revert to original).
    # First gain change fires EPC (epv or shadow_rise); state reset works.
    # Second gain change (within ~1s) fires EPC again BUT W still tracks
    # the first-gain regime → re-learn against mismatched taps → slow
    # convergence. F-E3 detects "consecutive" fires (within window) and:
    #   1. Extends hangover to ≥1s so RES knows we're still in transition
    #   2. Scales W by w_reset_factor (default 0.5) to soften stale taps
    # Cohort-tail guard: min_gap prevents W reset from spamming in
    # always-non-stationary paths (qNvSMyU-class cases fire shadow_rise
    # every few frames; without a gap-guard F-E3 would erase all converged
    # state on those cases).
    # Default OFF — opt-in ablation flag.
    f_e3_enabled: bool = False
    f_e3_consecutive_window_frames: int = 100   # within 1s at hop=160 / sr=16k
    f_e3_w_reset_min_gap_frames: int = 1000     # ≥10s between W resets
    f_e3_w_reset_factor: float = 0.5            # W *= this when consecutive fires

    # diverged_reset triple-AND gate (v3.11 Phase 1 Sprint 13-14).
    # Existing diverged_reset_enabled has gates (streak / cooldown /
    # filter_state=='diverged' / once_converged) but is default-OFF
    # because F2.2 EMA variant FAILed: EMA confuses shadow-tracking-during-
    # movement with true divergence. The triple-AND adds shadow_advantage
    # > 2.0 — requires shadow to also be ahead, which eliminates the
    # false-positive movement case.
    # Use together with diverged_reset_enabled=True (master enable).
    # Default OFF — opt-in ablation flag.
    diverged_reset_triple_and: bool = False
    diverged_reset_triple_and_shadow_adv_min: float = 2.0

    # v3.12 Phase 2 (C2 wiring) — filter_state → RES consumer hookup.
    # Phase 2 wires _prev_filter_state through ResFilter.process() and
    # _stage_gain_compute as a kwarg. When this flag is False, the kwarg
    # exists but is ignored (logic guard inside RES). Default OFF preserves
    # byte-equality with v3.11.x for the steady-state path. Phase 3 flips
    # this on and adds the actual consumption (gain_floor unification +
    # per-state ENR tuple).
    res_consume_filter_state: bool = False

    # v3.12 Phase 3B (S6-S7) — unified gain floor.
    # Replaces `(1 - coh²)` evidence in `ne_g_floor` with the F3.1 v3
    # mic-excess blend (0.7 mic + 0.3 legacy), matching the evidence used
    # by `dt_per_bin` in NE estimation. This addresses the Q7 V3 verdict:
    # `(1 - coh²)` saturates in FS post-cancellation, so the legacy
    # ne_g_floor erroneously raises the floor in FS (echo leakage).
    # Gate matches the existing F3.1 v3 path: filter_converged AND
    # long-window-ready AND not epc_active. Requires `use_mic_excess_evidence`
    # to be True (no-op when False since there's no blend available).
    # Falls back to `(1 - coh²)` when the gate fails (same as legacy path).
    # Default OFF preserves byte-equality with v3.11.2. Phase 3 verification:
    # flag OFF must be byte-equal, flag ON must pass 800-case AECMOS with
    # FS Δecho ≥ -0.02 / DT Δdeg ≥ -0.005 / cohort tail Δecho ≥ -0.05.
    res_unified_gain_floor: bool = False

    # v3.12 Phase 3B (S6b) — state-driven `epc_dt_cap` gate. Default OFF
    # research substrate. The S6b 800-case fire-rate audit (2026-05-13)
    # found the legacy gate `epc_active AND effective_dt > 0.35` fires
    # 0/2,032,022 frames at production thresholds — the cap action is
    # dead code in BALANCED. Q7 V3 mis-identified `epc_dt_cap` as the FS
    # leak carrier; real carriers are elsewhere (other caps / dt_per_bin
    # ENR — re-investigation pending). Flag retained for a future Phase 3C
    # state-set revision; current `{diverged, suspicious_dt}` set is
    # additive (not retargeting). See docs/v3_12_s6b_verdict.md.
    res_state_driven_epc_dt_cap: bool = False

    # v3.12 S7 (Phase 3B v3 Option α): drop the `NOT epc_active` gate on the
    # F3.1 v3 mic-excess path so it also fires during EPC-active frames.
    # Pre-implementation 800-case audit (2026-05-13, docs/v3_12_s7_*) found:
    #   target slice (filter_converged AND lw_ready AND epc_active) =
    #     33.97% global, 32-43% per bucket — well above 0.5% floor;
    #   unified dt_per_bin reduction in FS bins (coh²<0.1) = 67.7% global,
    #     67-68% per bucket — well above 20% bar.
    # Mechanism: legacy fallback path uses `(1-coh²)`-based dt_per_bin which
    # saturates to 1 in FS post-cancellation (coh²→0 → false NE-like
    # evidence) → over-shapes `nearend_est` → inflates Wiener / softgate
    # gain → echo leakage. F3.1 v3 mic-excess blend evidence (validated
    # via P1-Phase-1 AUROC 0.871) replaces the (1-coh²) saturation source.
    # NE bucket has 0% target frames (filter rarely converges in NE) so
    # this flag is NE-risk-free by audit. See
    # docs/v3_12_phase3b_v3_design.md §3.1 / §6.1.
    res_dt_per_bin_unified: bool = False

    # v3.12 S11 — Cap2 (residual_echo ≤ error_psd × mult) loosening in
    # FS-confident bins. S8 audit (800-case) shows Cap2 binding in
    # 18.28% / 17.11% of FS_static / FS_movement bins with mean
    # 12.34 / 12.95 dB downward magnitude shift. S6/S6b/S7/S10
    # (all NEUTRAL) targeted the ENR denominator (nearend_est /
    # gain-floor side); S11 targets the ENR numerator (residual_echo)
    # in the opposite direction — disables Cap2 in FS bins so
    # residual_echo passes through uncapped, raising ENR and lowering
    # gain. Inverse mechanism may not be neutralised by the same
    # downstream caps that absorbed S6-S10.
    res_cap2_fs_loosen: bool = False

    # v3.12 S10 — `noise_floor_psd` refinement (Phase 3B v5 follow-up).
    # S9-A.2 pre-audit data on 800-case (BALANCED, fl=832, seed=0):
    # baseline `noise_floor_psd = mean(error_psd) * 0.01` wins argmax in
    # 44% of FS bins (S8 finding). Replacing it with per-bin
    # `error_psd * 0.005` in FS-confident bins (coh² < 0.1) yields:
    #   FS_static  release-to-raw   11.53%, mean dB reduction 11.71
    #   FS_movement release-to-raw  14.25%, mean dB reduction 11.78
    #   intrusion outside floor baseline = 0% in all buckets
    # H1 hypothesis: the 23 dB nearend_est magnitude reduction in FS
    # (from S9-D upper bound) translates to ENR rise → gain drop →
    # FS leak reduction. This flag tests H1 via 800-case A/B AECMOS.
    # DT / NE bins (coh² ≥ 0.1) unchanged — keeps NE-side protection
    # intact. See docs/v3_12_s9_verdict.md §S10 plan.
    res_noise_floor_refined: bool = False

    # F-E5 — saturation handling extensions (v3.11 Phase 1 Sprint 11-12).
    # Edge case E5: clip & saturation. Current handling has 4 known gaps:
    #   E5-1: ref soft-clipped (saturation_softclip_ref), mic NOT — asymmetric
    #   E5-2: main filter mu only freezes at extreme mic clip (>0.8); shadow
    #         freezes at sat_level >= 0.5. Asymmetric: main keeps learning
    #         on clipped reference at sat_level 0.5-0.8.
    #   E5-3: filter._error_psd EMA not reset post-saturation; clipped samples
    #         pollute the EMA for the α=0.95 TC (~20 frames after sat ends).
    #   E5-4: sustained mid-saturation rises both filter errors → false
    #         shadow_rise EPC fire on what is really nonlinear distortion.
    # All four gated under one flag.
    # Default OFF — opt-in ablation flag.
    f_e5_enabled: bool = False
    f_e5_main_mu_sat_threshold: float = 0.5     # freeze main mu when sat above
    f_e5_mic_softclip_threshold: float = 0.3    # symmetric to ref soft-clip

    # F2.4 — mu holdoff no-reset (plan ~/.claude/plans/se-aec-aec-main-hazy-lynx.md).
    # _update_simple_mu_ratio resets holdoff=20 every DT frame; in marginal DT
    # (ratio oscillates around _simple_mu_ratio) holdoff never counts down → mu
    # is stuck low indefinitely. Fix: only set holdoff when it is at 0 (fresh
    # attack transition). During holdoff, DT frames apply fast-attack alpha but
    # do not extend the counter.
    # Default OFF — opt-in ablation flag.
    mu_holdoff_no_reset: bool = False

    # F2.5 — prev_dtd_conf two-stage hangover (plan ~/.claude/plans/se-aec-aec-main-hazy-lynx.md).
    # Current ×0.9 per-frame decay gives TC ≈ 10 frames (100ms), shorter than R EMA
    # recovery (α=0.95, TC ≈ 20 frames). Fix: attack fast (1 frame), then hold
    # for 10 frames before the ×0.9 decay begins → total release ~300ms.
    # Default OFF — opt-in ablation flag.
    dtd_conf_two_stage: bool = False

    # F1.1 — reverse_copy P reset (plan ~/.claude/plans/se-aec-aec-main-hazy-lynx.md).
    # When reverse_copy fires (shadow←main, main-winning case), shadow gets main's W
    # but retains its old high-Q P. Stale P → wrong K for the copied W for several
    # frames. Fix: re-arm shadow's P_max_override + P_floor_beta so K recalibrates
    # quickly. Also re-arm main with a mini-EPC P-boost for faster re-adaptation.
    # Default OFF — opt-in ablation flag.
    reverse_copy_p_reset: bool = False

    # P3c Phase 1a — DelayEstimator high-PAR fast-path. The shipped
    # `confidence` property forces n_updates >= 3 before any non-zero
    # output even when PAR is overwhelmingly above the solid threshold;
    # P3c trace showed this costs ~1 s of unnecessary blind window on
    # 80% of bench cases. The fast path lets us promote `is_solid` at
    # n_updates >= 2 IF (a) PAR >= delay_fast_par_threshold and
    # (b) the same lag is reported for two consecutive estimates.
    # Both guards together rule out single-frame spurious peaks.
    # Default ON — 800-case bench bit-identical to baseline n_updates>=3 path
    # across all buckets, 0 wrong locks across 800, and median TTFS drops
    # 4.09→3.57 s on 59 % of cases. Set False to revert to the n_updates>=3
    # gate.
    delay_fast_path_enabled: bool = True
    delay_fast_par_threshold: float = 40.0           # ~5x normal solid threshold

    # v3.13 E4 — SubtractiveNLP detector (audit-only in S3 per
    # docs/v3_13_e4_s2_design_lock.md). Voice-band autocorrelation
    # pitch tracker on post-RES output; outputs nl_confidence per hop
    # to self._diag, does NOT modify the AEC output. S5+ adds the
    # suppressor that actually consumes nl_confidence.
    # Default OFF — must be byte-equal output to baseline.
    e4_nlp_enabled: bool = False
    e4_nlp_pitch_threshold: float = 0.45      # listen-evidence: NL min 0.50, CTL max 0.38
    e4_nlp_continuity_frames: int = 3         # of 4-frame history
    e4_nlp_window_ms: float = 32.0            # analysis window for autocorr
    # S3.1 secondary gate: minimum linear-residual RMS in the analysis
    # window. Filters out pitched-but-low-residual frames in clean cases
    # (CTL1 dropped 8.7%→3.7% under 0.05; 5/5 NL still hit).
    e4_nlp_min_residual_rms: float = 0.05
    # S4.1 tertiary gate: mic_rms_ema / raw_output_rms_ema > threshold.
    # NE bucket has ratio ≈ 1.00 (filter doesn't cancel because no echo);
    # NL cohort has ratio 1.05-1.7. Threshold 1.05 cleanly separates:
    # 5/5 NL still fire, NE aggregate drops 14.31%→0.00%. Heavy-NL cases
    # (07/B) have low cancel_ratio because NL prevents convergence —
    # threshold ≥1.10 starts cutting into NL detection.
    e4_nlp_cancel_ratio_threshold: float = 1.05
    e4_nlp_cancel_ema_alpha: float = 0.99    # TC ~100 hops = 1 sec @ 10ms hop

    # S-orth.A — shadow state decoupling (v3.14 Arc S-orth Sprint A).
    # Motivation: Riccati equation forces dual PBFDKF shadows to track each
    # other (Switchboard AEC3 2026 deep dive). Q×3.5 damps but does not
    # break the coupling — shadow _error_psd and R follow the same signal
    # path as main's, making shadow a "damped echo of main" rather than
    # orthogonal evidence.
    #
    # Shipped decoupling (v3.14 Arc S-orth.A):
    #   _shadow_error_psd       per-bin EMA (shadow's own observation of error PSD)
    #   _shadow_R               shadow's Kalman R (written into shadow_filter.R each frame)
    #
    # RESERVED (declared but not wired — future arc scope, not v3.15):
    #   _shadow_copy_err_baseline shadow's view of "best error in stable FS"
    #                           — would require PathChangeRegimeHandler redesign
    #                           to track main and shadow baselines separately
    #                           (currently min(main_err, shadow_err) feeds a
    #                           single _copy_err_baseline).
    #   _shadow_mu_holdoff      shadow's own mu-holdoff counter — would require
    #                           a parallel _shadow_simple_mu_ratio mechanism
    #                           gating shadow's mu independently from main.
    # B5 (v3.15 §1.0.S2, 2026-05-14): doc aligned with actual implementation;
    # init/reset for _shadow_mu_holdoff kept harmless for future wiring.
    #
    # Safety regularization: Option B (quiescent re-sync). When the filter is
    # in steady-state FS (far_power > 1e-4 AND filter_state=='refined_usable'
    # AND shadow_error_psd > 3× main_error_psd OR < 0.33× main_error_psd),
    # shadow _error_psd is gently nudged toward main by a 10% blend per frame.
    # This caps divergence without breaking orthogonality on dynamic paths.
    # Rationale for Option B over Option A (hard ±50% cap):
    #   - Hard cap can mask true signal divergence during rapid echo-path change
    #   - Quiescent re-sync fires only when the filter IS converged, so it
    #     cannot corrupt the scenario where orthogonal evidence most matters
    #     (the non-stationary path precisely when filter is NOT converged).
    #
    # When OFF (default): behaviour is byte-equal to baseline (all F2.3/B5/F2.4
    # partial-decoupling paths remain active as before). When ON: full decoupling
    # + quiescent safety regularization. S-orth.B (L1 regularization) is a
    # separate gated sprint — do not enable until S-orth.A 800-case is run.
    shadow_state_decoupled: bool = False

    # v3.14 Arc-P Sprint 02: adaptive per-band ERL EMA driven by
    # echo_psd[k] / far_psd[k] (PBFDKF filter output, per-bin).
    #
    # Motivation (P.S1 audit, 2026-05-14): scalar erl_estimate=0.3 is a
    # 7× OVER-estimate in low-coupling rooms (case 04 LF=0.043, MF=0.191,
    # HF=0.111). This inflates the echo mask in the F3.1-v3 mic-excess
    # formula → excess_ratio ≈ 0 → dt_per_bin ≈ 0 → echo leaks into
    # dt_per_bin when it should be identified as FS. Inter-room variance
    # (11×) >> inter-band variance → a fixed per-band table cannot
    # generalise; adaptive EMA per band is the canonical solution.
    #
    # Design (P.S2):
    #   3 bands: LF 0–1k Hz, MF 1–4k Hz, HF 4–8k Hz
    #   Source: per-bin |echo_spec|² / |far_spec|² (PBFDKF output)
    #   EMA α = 0.99 (TC ≈ 100 frames ≈ 1 s at hop=160/16 kHz, research-
    #           backed: Jung 2011 §IV.A uses similar TC for echo-path EMA)
    #   Update gate: filter_state in ('refined_usable', 'coarse_learning')
    #           AND far_pwr > 1e-4 (same gate as scalar erl_estimate)
    #           AND raw_dt_ratio < 2.0 AND inst_erl < 1.5 (NE corruption guard)
    #   Post-EPC cap: per-band min(current, per_band_erl_cap[b]) where
    #           cap[LF]=0.6, cap[MF]=0.8, cap[HF]=1.0 (conservative, wide)
    #           (replaces scalar min(current, 0.3) — only active when flag ON)
    #   Byte-equal proof: when flag=False, per-band arrays are not updated
    #           and erl_e in F3.1 formula uses float(self._erl_estimate)
    #           — identical to pre-P.S2 path. No production code path touched.
    #
    # Default OFF: byte-equal to baseline when False. P.S3 tune sprint will
    # optimise α and the per-band clip bounds after P.S2 wiring is verified.
    f3_1_per_band_erl_adaptive: bool = False
    # Per-band ERL EMA time constant (α). 0.99 = TC ≈ 100 hops at hop=160.
    # Slow tracking is intentional: ERL changes only on room/position change.
    per_band_erl_alpha: float = 0.99
    # Post-EPC per-band cap values [LF, MF, HF]. Replaces scalar 0.3 cap
    # (line 6265 / 6304) when flag is ON. Wide conservative values from P.S1
    # clip_hi recommendations; tighten in P.S3 after seeing 800-case distribution.
    per_band_erl_cap_lf: float = 0.6
    per_band_erl_cap_mf: float = 0.8
    per_band_erl_cap_hf: float = 1.0
    # Per-band ERL clip bounds (safety floor/ceil on the EMA value itself).
    per_band_erl_clip_lo: float = 0.005
    per_band_erl_clip_hi: float = 1.5

    # v3.14 Arc-R Sprint S1: per-band ENR threshold wire.
    #
    # Motivation: ENR thresholds `enr_t_ne` / `enr_s_ne` in
    # `ResFilter._stage_gain_compute` (aec.py:2792-2793) currently use a
    # `_enr_blend`-driven interpolation between two scalar endpoints
    # (2.0/1.5 for `t_ne`, 3.0/2.5 for `s_ne`). The blend ramps over
    # bins 5-10 (~150-300 Hz @ 16kHz/256-fft); bins 11+ asymptote to
    # the high-band scalar. After P.S3's adaptive per-band ERL (which
    # already varies LF/MF/HF by ~4× per-room), the downstream ENR
    # thresholds remain band-flat — this clips P.S3's signal before it
    # reaches the gain decision.
    #
    # Per-bin audit (`project_per_bin_audit_findings.md`) ranks
    # `enr_t_ne / enr_s_ne` as the #2 per-band candidate (right after
    # `erl_estimate` which P-revised P.S3 addresses).
    #
    # Wire-only sprint (R.S1): default tuple values are uniform =
    # current high-band scalar (1.5 / 2.5), and the flag is default
    # OFF. When OFF: legacy `_enr_blend` path is untouched (byte-equal).
    # When ON: a per-bin array is built once at ResFilter init from the
    # 3-band tuple (using the same LF 0-1k / MF 1-4k / HF 4-8k split as
    # P.S3) and substituted at the use site. Tuning happens in R.S2.
    res_per_band_enr: bool = False
    # Per-band tuple values [LF, MF, HF] for the NE-side ENR thresholds.
    # Defaults match the R.S2 sweep winner "block_lf" (DT bucket +0.007 dB
    # mean Δdeg on 800-case AECMOS; FS regression within -0.02 bar). The
    # band tilt blocks LF echo (raise threshold) while admitting HF speech
    # (lower threshold). Mirror-scaled by ~1.67× for the SD-side tuple to
    # preserve the legacy t:s ratio.
    enr_t_ne_per_band: tuple = (2.0, 1.5, 1.0)
    enr_s_ne_per_band: tuple = (3.33, 2.5, 1.67)

    # v3.15 §1.2 — DT-NE compression fix (default OFF, byte-equal flag-OFF).
    #
    # Motivation: §1.1 audit on 25 worst-DT cases (by Δdeg vs v3.10.5
    # pre-E2) found that worst-DT cases spend 40-99% of frames in
    # `coarse_learning` state (filter never converges), so the linear-
    # filter `residual_echo_psd` is INFLATED throughout the case. The
    # ENR gate `enr = residual_echo_psd / nearend_est` therefore over-fires
    # and softgate stage S0 outputs voice-band mean gain of 0.254
    # (≈ -11.9 dB) on NE-evidence frames. F3.1-v3 mic-excess
    # (`dt_per_bin`) fires correctly at 1.0 on these frames but is
    # additive evidence that cannot override the inflated-residual ENR.
    #
    # Two complementary mechanisms (both gated by `dt_ne_compression_fix`):
    #
    # (a) Per-state ENR scalar:
    #     Multiply enr_t_ne / enr_s_ne by `_state_scale` based on the
    #     internal P3f filter_state. `refined_usable` is FORCED to 1.0
    #     (byte-equal to v3.13.x BALANCED steady path). Pre-convergence
    #     states (idle/startup/coarse_learning/diverged) get a
    #     larger scale to relax the gate when residual_echo_psd is known
    #     to be inflated. `suspicious_dt` stays at 1.0 (already protected
    #     by upstream DTD).
    #
    # (b) Per-bin dt_per_bin override:
    #     On bins where `dt_per_bin > dt_ne_per_bin_thresh` (per-bin
    #     mic-excess evidence says "strong NE"), additionally multiply
    #     enr_t_ne / enr_s_ne by `dt_ne_per_bin_scale`. This surgical
    #     override preserves NE syllables on bins where mic-excess is
    #     decisive, regardless of state.
    #
    # Flag-OFF default: byte-equal to v3.14 BALANCED. ON in §1.2.S4
    # 800-case A/B + listen verify; ship target = DT bucket Δdeg ≥
    # +0.020 dB (40% of E2 Path 3 debt), cohort tail Δecho ≥ -0.05,
    # FS Δecho ≥ -0.02.
    dt_ne_compression_fix: bool = False
    # Multiplicative scale on enr_t_ne / enr_s_ne per state. `refined_usable`
    # MUST be 1.0 to preserve steady-state byte-equal with v3.13 BALANCED.
    dt_ne_state_scale: dict = field(default_factory=lambda: {
        'idle':            2.0,
        'startup':         2.0,
        'coarse_learning': 2.0,
        'refined_usable':  1.0,
        'diverged':        2.0,
        'suspicious_dt':   1.0,
    })
    # Per-bin threshold for the dt_per_bin override gate.
    dt_ne_per_bin_thresh: float = 0.5
    # Multiplicative scale applied to enr_t_ne / enr_s_ne on bins where
    # dt_per_bin > dt_ne_per_bin_thresh.  Stacks with state_scale.
    dt_ne_per_bin_scale: float = 2.0

    # Mode
    mode: AecMode = AecMode.PBFDKF

    # External RES context
    return_res_context: bool = False   # True → process() returns (output, AecResContext)

    # TIME/LMS history control
    clear_filter_history: bool = False  # Clear ref_buffer each block (default: keep 1 hop history)

    def __post_init__(self):
        if self.frame_size == -1:
            self.frame_size = self.sample_rate * 20 // 1000  # 20ms
        if self.hop_size == -1:
            self.hop_size = self.frame_size // 2             # 10ms
        if self.filter_length == -1:
            # D5: 48kHz needs longer filter (room reverb more prominent)
            # PR-D1 (v3.6.0): bumped 16/8kHz default 32→52ms to match AEC3
            # default (13 blocks × 4ms) — captures more RT60 tail.
            if self.sample_rate >= 44100:
                self.filter_length = self.sample_rate * 64 // 1000  # 64ms
            else:
                self.filter_length = self.sample_rate * 52 // 1000  # 52ms (was 32ms)

    @property
    def fft_size(self) -> int:
        # Next power of 2 >= frame_size (= frame_size when frame_size is power of 2)
        n = self.frame_size
        return 1 << (n - 1).bit_length()

    @classmethod
    def from_preset(cls, preset: 'AecPreset', **kwargs) -> 'AecConfig':
        """Create config from preset with optional overrides.

        Presets (echo suppression strength):
          MILD:       Best near-end preservation, lightest echo suppression
          BALANCED:   Balanced echo suppression and near-end quality (default)
          AGGRESSIVE: Stronger echo suppression, moderate near-end degradation
          MAXIMUM:    Maximum echo suppression, significant near-end impact
        """
        if isinstance(preset, str):
            preset = AecPreset(preset)
        if preset == AecPreset.MILD:
            # v3.8.3: shifted one slot lighter — minimum-touch RES for
            # quiet/light-echo cases where NE intelligibility trumps echo
            # cleanup. Former v3.8.2 MILD values now live in SOFT.
            defaults = dict(
                # RES v2
                res_echo_method="direct",
                res_gain_type="enr",
                res_enable_reverb=True,
                res_reverb_decay=0.45,
                res_reverb_gain=0.4,
                res_alpha_echo_psd=0.6,
                res_alpha_error_psd=0.6,
                res_enr_scale=1.15,
                # RES suppression (ultra-light)
                res_g_min_db=-25.0,
                res_over_sub_base=1.5,
                res_over_sub_scale=2.5,
                res_dt_reduction=4.5,
                res_spectral_floor_db=-18.0,
                res_ne_protect_db=-7.0,
                enable_cng=True,
                shadow_q_ratio=3.0,
                # Adaptive filter
                shadow_mu_min=0.5,
                warmup_frames=80,
                kalman_q_high=1.5e-3,
            )
        elif preset == AecPreset.SOFT:
            # = former v3.8.2 MILD. Shifted one slot to make room for an
            # even lighter MILD; preserved for users who liked the v3.8.2
            # MILD positioning (light RES with audible echo cleanup).
            defaults = dict(
                # RES v2
                res_echo_method="direct",
                res_gain_type="enr",
                res_enable_reverb=True,
                res_reverb_decay=0.6,
                res_reverb_gain=0.8,
                res_alpha_echo_psd=0.5,
                res_alpha_error_psd=0.6,
                res_enr_scale=1.0,
                # RES suppression
                res_g_min_db=-35.0,
                res_over_sub_base=2.5,
                res_over_sub_scale=4.0,
                res_dt_reduction=3.5,
                res_spectral_floor_db=-25.0,
                res_ne_protect_db=-10.0,
                enable_cng=True,
                shadow_q_ratio=3.0,
                # Adaptive filter
                shadow_mu_min=0.5,
                warmup_frames=80,
                kalman_q_high=1.5e-3,
            )
        elif preset == AecPreset.BALANCED:
            defaults = dict(
                # RES v2
                res_echo_method="direct",
                res_gain_type="enr",
                res_enable_reverb=True,
                res_reverb_decay=0.85,    # v3.3: TC ~130ms (was 50ms); RT60-typical
                res_reverb_gain=1.6,      # v3.3: bump (was 1.4); DT gate weakened separately
                res_alpha_echo_psd=0.4,
                res_alpha_error_psd=0.5,
                res_enr_scale=0.85,
                # RES suppression (balanced echo/speech trade-off)
                res_g_min_db=-55.0,
                res_over_sub_base=5.0,
                res_over_sub_scale=9.0,
                res_dt_reduction=2.5,
                res_spectral_floor_db=-38.0,
                res_ne_protect_db=-16.0,
                # v2.7 E6: min-stat noise floor + CNG fill suppression gap
                enable_cng=True,
                shadow_q_ratio=3.5,
                # Adaptive filter
                shadow_mu_min=0.6,
                warmup_frames=80,
                kalman_q_high=1e-3,
                # F3.1-v3: per-bin mic-energy excess NE evidence (GREEN-PASS, 800-case)
                use_mic_excess_evidence=True,
                # F2.3: Yang 2017 R-reset on EPC — net 0.3× improvement ratio (800-case PASS)
                epc_r_reset_enabled=True,
                # F2.4: mu holdoff only armed on fresh onset — net 0.9× (800-case PASS)
                mu_holdoff_no_reset=True,
                # v3.11 B5: symmetric Yang 2017 R-reset on shadow filter (Phase 1 Sprint 1-2 PASS)
                shadow_r_reset_enabled=True,
                # v3.14 S-orth.A: decouple shadow's Kalman _error_psd + R
                # from main's (B5/§1.0.S2: only these two are actually wired
                # in v3.14; _copy_err_baseline / mu_holdoff remain coupled
                # and are reserved for a future shadow-decoupling arc).
                # 800-case GREEN PASS (commit 8089974): all 5 buckets within bar,
                # cohort tail qNvSMyU Δecho +0.0036, state correlation drops
                # main vs shadow 0.99 → 0.47 on DT_static (target 0.5-0.7 hit).
                # First mechanism across 5+ shadow-retirement attempts that
                # produces genuinely independent shadow Kalman state.
                shadow_state_decoupled=True,
                # v3.14 Arc-P P.S3: adaptive per-band ERL EMA driven by
                # error_psd / far_lw (Option B source signal). Replaces
                # scalar erl_estimate=0.3 (7× over-estimate in low-coupling
                # rooms) with 3-band LF/MF/HF EMA (α=0.99).
                f3_1_per_band_erl_adaptive=True,
                # v3.14 Arc-R R.S2: per-band ENR thresholds with block_lf
                # tilt (raise LF, lower HF). DT bucket +0.007 dB mean Δdeg
                # on 800-case; FS regression within -0.02 bar. 7-case
                # xrtntuju listen verification 2026-05-14: NE not damaged,
                # FS not audibly leaking. Paired with f3_1_per_band_erl_adaptive
                # for end-to-end per-band gate.
                res_per_band_enr=True,
                # v3.11 F-E5: saturation handling extensions (Phase 1 Sprint 11-12 PASS)
                f_e5_enabled=True,
                # v3.11 diverged_reset: triple-AND gate unblocks dead code safely
                # (Phase 1 Sprint 13-14 PASS, F2.2 trap avoided via shadow_advantage > 2.0)
                diverged_reset_enabled=True,
                diverged_reset_triple_and=True,
                # v3.12 S1: B6 shadow_mu state-aware (PASS bucket mean +0.007,
                # wlAXM0i listen verdict: no audible diff, B6 has more NE detail in spectrogram)
                shadow_mu_state_aware=True,
                # v3.12 S2: F-E1 + F-DelayTrack default-ON (NEUTRAL on 800-case bench,
                # correctness-only for extreme ERL / long-session delay drift).
                # Cross-coupling audited disjoint from B5/F-E5/diverged_reset.
                f_e1_enabled=True,
                f_delaytrack_enabled=True,
            )
        elif preset == AecPreset.AGGRESSIVE:
            defaults = dict(
                # RES v2
                res_echo_method="direct",
                res_gain_type="enr",
                res_enable_reverb=True,
                res_reverb_decay=0.7,
                res_reverb_gain=2.0,
                res_alpha_echo_psd=0.3,
                res_alpha_error_psd=0.4,
                res_enr_scale=0.7,
                # RES suppression (stronger echo suppression)
                res_g_min_db=-65.0,
                res_over_sub_base=7.0,
                res_over_sub_scale=12.0,
                res_dt_reduction=1.5,
                res_spectral_floor_db=-45.0,
                res_ne_protect_db=-22.0,
                enable_cng=True,
                shadow_q_ratio=4.0,
                # Adaptive filter
                shadow_mu_min=0.7,
                warmup_frames=80,
                kalman_q_high=7e-4,
            )
        elif preset == AecPreset.MAXIMUM:
            defaults = dict(
                # RES v2
                res_echo_method="direct",
                res_gain_type="enr",
                res_enable_reverb=True,
                res_reverb_decay=0.8,
                res_reverb_gain=3.0,
                res_alpha_echo_psd=0.2,
                res_alpha_error_psd=0.3,
                res_enr_scale=0.5,
                # RES suppression (maximum, significant near-end impact)
                res_g_min_db=-72.0,
                res_over_sub_base=10.0,
                res_over_sub_scale=15.0,
                res_dt_reduction=0.5,
                res_spectral_floor_db=-55.0,
                res_ne_protect_db=-30.0,
                enable_cng=True,
                shadow_q_ratio=5.0,
                # Adaptive filter
                shadow_mu_min=0.9,
                warmup_frames=100,
                kalman_q_high=7e-4,
            )
        else:
            defaults = {}
        defaults.update(kwargs)
        return cls(**defaults)


class DelayEstimator:
    """GCC-PHAT delay estimator for AEC reference alignment.

    Uses short overlapping segments for fast initial estimation.
    Cross-spectrum is accumulated over segments and smoothed with EMA.
    """

    def __init__(self, sample_rate: int, max_delay_ms: float = 1024.0,
                 init_seconds: float = 0.5, period_seconds: float = 2.0,
                 par_low_threshold: float = 5.0,
                 par_solid_threshold: float = 8.0,
                 trace: bool = False,
                 fast_path_enabled: bool = False,
                 fast_par_threshold: float = 40.0):
        """v3.10.4: max_delay_ms default 250 → 512 → 1024 (matches WebRTC's
        Old AEC ~1 s far-end history; AEC3's 512 ms misses BT/mobile skew
        cases). seg_size auto-scales to 2× max_delay.

        Confidence reporting:
          peak_to_avg_ratio (PAR) is the GCC-PHAT peak height divided by
          the average magnitude over the search window. PAR > solid → trust
          the estimate (drive aggressive RES + filter mu_scale). PAR < low
          → don't trust (hold mu_scale at higher floor, RES stays
          conservative). Between: progressive blend.
        """
        self.sample_rate = sample_rate
        self.max_delay_samples = int(max_delay_ms * sample_rate / 1000)
        self.init_seconds = init_seconds
        self.period_seconds = period_seconds
        self.par_low_threshold = par_low_threshold
        self.par_solid_threshold = par_solid_threshold
        # P3c Phase 1a: high-PAR fast-path knobs.
        self.fast_path_enabled = bool(fast_path_enabled)
        self.fast_par_threshold = float(fast_par_threshold)
        self._prev_estimated_delay = -1  # set in _estimate before overwrite

        # Analysis window: 2x max_delay, but at least 2048
        self.seg_size = 1
        min_seg = max(2048, 2 * self.max_delay_samples)
        while self.seg_size < min_seg:
            self.seg_size *= 2
        self.seg_hop = self.seg_size // 2  # 50% overlap
        self.n_freqs = self.seg_size // 2 + 1

        # Smoothed cross-spectrum
        self._cross_spec = np.zeros(self.n_freqs, dtype=np.complex128)
        self._alpha = 0.6
        self._n_updates = 0

        # Sliding buffers (accumulate hop-by-hop)
        self._mic_buf = np.zeros(self.seg_size, dtype=np.float32)
        self._ref_buf = np.zeros(self.seg_size, dtype=np.float32)
        self._buf_pos = 0  # how many samples in current segment

        # State
        self.estimated_delay = -1
        self._samples_accumulated = 0
        self._samples_since_est = 0
        self._init_done = False
        self._init_samples = int(init_seconds * sample_rate)
        self._period_samples = int(period_seconds * sample_rate)
        self._n_estimates = 0
        self._last_par = 0.0  # A5: Peak-to-Average Ratio confidence

        # P3b instrumentation (opt-in; default off → zero overhead).
        self._trace = bool(trace)
        self._trace_rows: List[dict] = [] if self._trace else None
        self._trace_call_idx = 0
        self._trace_total_samples = 0

    def reset(self):
        self._cross_spec.fill(0)
        self._mic_buf.fill(0)
        self._ref_buf.fill(0)
        self._buf_pos = 0
        self._n_updates = 0
        self.estimated_delay = -1
        self._samples_accumulated = 0
        self._samples_since_est = 0
        self._init_done = False
        self._n_estimates = 0

    def accumulate(self, mic: np.ndarray, ref: np.ndarray) -> bool:
        """Feed mic/ref samples. Returns True if a new delay estimate was made."""
        n = len(mic)
        # P3b: capture pre-call state so we can emit a trace row even on
        # branches that don't run _estimate.
        if self._trace:
            mic_pwr_pre = float(np.mean(np.asarray(mic, dtype=np.float64) ** 2))
            ref_pwr_pre = float(np.mean(np.asarray(ref, dtype=np.float64) ** 2))
            n_updates_pre = self._n_updates
            samples_pre = self._samples_accumulated
        self._samples_accumulated += n
        self._samples_since_est += n

        # Accumulate into segment buffer
        remaining = n
        src_pos = 0
        while remaining > 0:
            space = self.seg_size - self._buf_pos
            chunk = min(remaining, space)
            self._mic_buf[self._buf_pos:self._buf_pos + chunk] = mic[src_pos:src_pos + chunk]
            self._ref_buf[self._buf_pos:self._buf_pos + chunk] = ref[src_pos:src_pos + chunk]
            self._buf_pos += chunk
            src_pos += chunk
            remaining -= chunk

            if self._buf_pos >= self.seg_size:
                # Segment full — update cross-spectrum
                self._update_cross_spectrum()
                # Shift by seg_hop (50% overlap)
                self._mic_buf[:self.seg_hop] = self._mic_buf[self.seg_hop:]
                self._ref_buf[:self.seg_hop] = self._ref_buf[self.seg_hop:]
                self._buf_pos = self.seg_hop

        # Estimate when enough data accumulated
        did_estimate = False
        if self._n_updates < 2:
            ret = False
        elif not self._init_done:
            if self._samples_accumulated >= self._init_samples:
                self._estimate()
                self._init_done = True
                did_estimate = True
                ret = True
            else:
                ret = False
        else:
            if self._samples_since_est >= self._period_samples:
                self._estimate()
                did_estimate = True
                ret = True
            else:
                ret = False

        # P3b: emit one trace row per accumulate() call.
        if self._trace:
            sr = self.sample_rate
            top3 = []
            par_top1 = 0.0
            if self._n_updates >= 1:
                # Recompute GCC-PHAT over the running EMA cross-spectrum
                # (same expression as _estimate; uses fp64 throughout).
                mag = np.abs(self._cross_spec) + 1e-12
                phat = self._cross_spec / mag
                gcc = np.fft.irfft(phat, n=self.seg_size).astype(np.float64)
                max_d = min(self.max_delay_samples, self.seg_size // 2)
                region = np.abs(gcc[: max_d + 1])
                # Top-3 with greedy 16-sample lobe suppression.
                tmp = region.copy()
                for _ in range(3):
                    idx = int(np.argmax(tmp))
                    height = float(tmp[idx])
                    if height <= 0.0:
                        break
                    # PAR vs full-region mean excluding this sample.
                    peak_v = float(region[idx])
                    mean_excl = (float(region.sum()) - peak_v) / (
                        len(region) - 1 + 1e-10
                    )
                    par = float(peak_v / (mean_excl + 1e-10))
                    top3.append((idx, height, par))
                    lo = max(0, idx - 16)
                    hi = min(len(tmp), idx + 17)
                    tmp[lo:hi] = -1.0
                if top3:
                    par_top1 = top3[0][2]
            row = {
                "call_idx": self._trace_call_idx,
                "frame_samples": n,
                "time_s": self._trace_total_samples / sr,
                "mic_pwr_pre": mic_pwr_pre,
                "ref_pwr_pre": ref_pwr_pre,
                "n_updates_pre": n_updates_pre,
                "n_updates_post": self._n_updates,
                "samples_accumulated": self._samples_accumulated,
                "in_init_window": self._samples_accumulated < self._init_samples,
                "init_done": self._init_done,
                "did_estimate": did_estimate,
                "estimated_delay": int(self.estimated_delay),
                "last_par": float(self._last_par),
                "confidence": float(self.confidence),
                "is_solid": bool(self.is_solid),
                "top1_lag": top3[0][0] if len(top3) >= 1 else -1,
                "top1_height": top3[0][1] if len(top3) >= 1 else 0.0,
                "top1_par": top3[0][2] if len(top3) >= 1 else 0.0,
                "top2_lag": top3[1][0] if len(top3) >= 2 else -1,
                "top2_height": top3[1][1] if len(top3) >= 2 else 0.0,
                "top2_par": top3[1][2] if len(top3) >= 2 else 0.0,
                "top3_lag": top3[2][0] if len(top3) >= 3 else -1,
                "top3_height": top3[2][1] if len(top3) >= 3 else 0.0,
                "top3_par": top3[2][2] if len(top3) >= 3 else 0.0,
            }
            self._trace_rows.append(row)
            self._trace_call_idx += 1
            self._trace_total_samples += n

        return ret

    def _update_cross_spectrum(self):
        """Update smoothed cross-spectrum from current segment."""
        mic_spec = np.fft.rfft(self._mic_buf)
        ref_spec = np.fft.rfft(self._ref_buf)
        cross = mic_spec * np.conj(ref_spec)
        self._n_updates += 1
        if self._n_updates == 1:
            self._cross_spec = cross.copy()
        else:
            self._cross_spec = self._alpha * self._cross_spec + (1 - self._alpha) * cross

    def _estimate(self):
        """Estimate delay from accumulated cross-spectrum using GCC-PHAT."""
        magnitude = np.abs(self._cross_spec) + 1e-10
        phat = self._cross_spec / magnitude
        gcc = np.fft.irfft(phat, n=self.seg_size)

        max_d = min(self.max_delay_samples, self.seg_size // 2)

        # Search positive delays (mic lags ref — normal case)
        pos_range = gcc[:max_d + 1]
        best_pos = np.argmax(np.abs(pos_range))

        # A5: PAR (Peak-to-Average Ratio) confidence
        peak = float(np.abs(gcc[best_pos]))
        mean_excl = (np.sum(np.abs(pos_range)) - peak) / (len(pos_range) - 1 + 1e-10)
        self._last_par = float(peak / (mean_excl + 1e-10))

        # P3c Phase 1a: remember the previous estimate's lag before
        # overwriting, so the fast-path can require lag stability across
        # two consecutive estimates.
        self._prev_estimated_delay = self.estimated_delay
        self.estimated_delay = best_pos
        self._samples_since_est = 0
        self._n_estimates += 1

    @property
    def confidence(self) -> float:
        """[0, 1] — 0 = no evidence, 1 = solid PAR ≥ par_solid_threshold.

        Used by AEC mu_scale floor and RES conservative-mode gating. v3.10.0:
        promoted from internal _last_par to first-class API.

        P3c Phase 1a: when `fast_path_enabled`, allow the gate to clear at
        n_updates >= 2 if the PAR is overwhelmingly above the solid
        threshold AND the same lag was reported by the previous estimate.
        Both guards are required; a single high-PAR sample alone does not
        promote (rules out spurious peaks). Default off — opt-in.
        """
        if self.estimated_delay < 0:
            return 0.0
        par = self._last_par
        if (self.fast_path_enabled
                and self._n_updates >= 2
                and par >= self.fast_par_threshold
                and self._prev_estimated_delay == self.estimated_delay
                and self._prev_estimated_delay >= 0):
            return 1.0
        if self._n_updates < 3:
            return 0.0
        lo = self.par_low_threshold
        hi = self.par_solid_threshold
        if par <= lo:
            return 0.0
        if par >= hi:
            return 1.0
        return float((par - lo) / (hi - lo))

    @property
    def is_solid(self) -> bool:
        """Convenience: True when confidence is at the strong threshold."""
        return self.confidence >= 1.0


class NlmsFilter:
    """Time-domain NLMS Adaptive Filter"""

    def __init__(self, filter_length: int, mu: float = 0.3,
                 delta: float = 1e-8,
                 normalize: bool = True):
        self.filter_length = filter_length
        self.mu = mu
        self.delta = delta
        self.normalize = normalize
        self.weights = np.zeros(filter_length, dtype=np.float32)
        self.ref_buffer = np.zeros(filter_length, dtype=np.float32)
        self.power_sum = 0.0
        self.clear_history = False
        self.max_w_norm = 1.5  # Weight norm constraint (prevents explosion during double-talk)

    def reset(self):
        self.weights.fill(0)
        self.ref_buffer.fill(0)
        self.power_sum = 0.0

    def process_sample(self, near_end: float, far_end: float,
                       mu_scale: float = 1.0) -> Tuple[float, float]:
        oldest = self.ref_buffer[-1]
        self.power_sum = max(0, self.power_sum - oldest * oldest + far_end * far_end)
        self.ref_buffer[1:] = self.ref_buffer[:-1]
        self.ref_buffer[0] = far_end
        echo_est = np.dot(self.weights, self.ref_buffer)
        error = near_end - echo_est

        if mu_scale > 0 and self.power_sum > self.delta * self.filter_length:
            if self.normalize:
                mu_eff = (self.mu * mu_scale) / (self.power_sum + self.delta)
            else:
                mu_eff = self.mu * mu_scale
            self.weights += mu_eff * error * self.ref_buffer

        return error, echo_est

    def process_block(self, near_end: np.ndarray, far_end: np.ndarray,
                      mu_scale: float = 1.0) -> Tuple[np.ndarray, np.ndarray]:
        # Optionally clear history (no carry-over between blocks)
        if self.clear_history:
            self.ref_buffer.fill(0)
            self.power_sum = 0.0

        n = len(near_end)
        output = np.zeros(n, dtype=np.float32)
        echo_est = np.zeros(n, dtype=np.float32)
        for i in range(n):
            output[i], echo_est[i] = self.process_sample(
                near_end[i], far_end[i], mu_scale)

        # Weight norm constraint: prevent explosion during double-talk
        w_norm = np.linalg.norm(self.weights)
        if w_norm > self.max_w_norm:
            self.weights *= self.max_w_norm / w_norm

        return output, echo_est


class PBFDAF:
    """
    Partitioned Block Frequency-Domain Adaptive Filter (PBFDAF)

    NLMS-based adaptive filter using overlap-save for linear convolution.
    For Kalman-based adaptation, use PBFDKF subclass.
    """

    def __init__(self, block_size: int, n_partitions: int,
                 mu: float = 0.3, delta: float = 1e-8,
                 hop_size: int = 0):
        # Overlap-save: block_size = 2 × hop (proper 50% ratio for TD constraint)
        # FFT zero-pads to next power of 2 if block_size isn't one
        self.hop_size = hop_size if hop_size > 0 else block_size // 2
        self.block_size = 2 * self.hop_size  # overlap-save buffer (exactly 2× hop)
        self.fft_size = 1 << (self.block_size - 1).bit_length()  # next pow2
        self.n_partitions = n_partitions
        self.n_freqs = self.fft_size // 2 + 1
        self.mu = mu
        self.delta = delta
        self.alpha_power = 0.9
        self.enable_td_constraint = True  # can be disabled for diagnosis

        # Time-domain constraint window: clean 50% truncation
        # block_size = 2×hop → truncation at 50%, minimal Gibbs ringing
        self._td_window = np.ones(self.fft_size, dtype=np.float32)
        fade_len = self.hop_size // 4  # 40 samples for hop=160
        fade = 0.5 * (1.0 - np.cos(np.pi * np.arange(fade_len) / fade_len))
        self._td_window[self.hop_size - fade_len:self.hop_size] = fade[::-1].astype(np.float32)
        self._td_window[self.hop_size:] = 0.0

        # Filter weights [n_partitions, n_freqs]
        self.W = np.zeros((n_partitions, self.n_freqs), dtype=np.complex64)

        # Reference spectrum history [n_partitions, n_freqs]
        self.X_buf = np.zeros((n_partitions, self.n_freqs), dtype=np.complex64)
        self.partition_idx = 0

        # Input buffers (block_size = 2 × hop for overlap-save)
        self.near_buffer = np.zeros(self.block_size, dtype=np.float32)
        self.far_buffer = np.zeros(self.block_size, dtype=np.float32)

        # Power estimation
        self.power = np.zeros(self.n_freqs, dtype=np.float32)

        # Output spectra (for RES / coherence DTD)
        self.near_spec = np.zeros(self.n_freqs, dtype=np.complex64)
        self.echo_spec = np.zeros(self.n_freqs, dtype=np.complex64)
        self.error_spec = np.zeros(self.n_freqs, dtype=np.complex64)
        self.far_spec = np.zeros(self.n_freqs, dtype=np.complex64)

        # Windowed error spectrum for RES analysis (sqrt-Hann, same variance
        # as OLA spec but time-aligned with far_spec/echo_spec for coherence)
        self._sqrt_hann_analysis = np.sqrt(
            np.hanning(self.block_size)).astype(np.float32)
        self.error_spec_windowed = np.zeros(self.n_freqs, dtype=np.complex64)

    def reset(self):
        self.W.fill(0)
        self.X_buf.fill(0)
        self.near_buffer.fill(0)
        self.far_buffer.fill(0)
        self.power.fill(0)
        self.partition_idx = 0
        self.error_spec_windowed.fill(0)

    def process(self, near_end: np.ndarray, far_end: np.ndarray,
                mu_scale=1.0) -> np.ndarray:
        """Process hop_size samples. mu_scale: scalar or per-bin array [n_freqs]."""
        hop = self.hop_size

        # Shift buffers (overlap-save)
        self.near_buffer[:-hop] = self.near_buffer[hop:]
        self.near_buffer[-hop:] = near_end

        self.far_buffer[:-hop] = self.far_buffer[hop:]
        self.far_buffer[-hop:] = far_end

        # FFT (zero-pad block_size buffer to fft_size, cast to float32 precision)
        near_spec = np.fft.rfft(self.near_buffer, self.fft_size).astype(np.complex64)
        far_spec = np.fft.rfft(self.far_buffer, self.fft_size).astype(np.complex64)
        self.near_spec = near_spec  # expose for RES overlap-save
        self.far_spec = far_spec  # expose for coherence DTD

        # Store far-end spectrum
        curr_p = self.partition_idx
        self.X_buf[curr_p] = far_spec

        # Update power estimate (cold start: initialize directly on first active frame)
        far_psd = np.abs(far_spec) ** 2
        if np.sum(self.power) < 1e-10 and np.sum(far_psd) > 1e-10:
            self.power = far_psd.astype(np.float32)
        else:
            self.power = (self.alpha_power * self.power +
                         (1 - self.alpha_power) * far_psd)

        # Compute echo estimate
        self.echo_spec.fill(0)
        for p in range(self.n_partitions):
            p_idx = (curr_p - p) % self.n_partitions
            self.echo_spec += self.W[p] * self.X_buf[p_idx]

        # IFFT (fft_size → take block_size valid samples)
        echo_time = np.fft.irfft(self.echo_spec, self.fft_size).astype(np.float32)

        # Error (take valid region for output)
        output = self.near_buffer[-hop:] - echo_time[self.hop_size:self.block_size]

        # Error spectrum (zero-pad to fft_size, valid region at [hop, block_size))
        error_time = np.zeros(self.fft_size, dtype=np.float32)
        error_time[self.hop_size:self.block_size] = output
        self.error_spec = np.fft.rfft(error_time).astype(np.complex64)

        # Windowed error spec for RES analysis: near_buffer × sqrt-Hann − echo_spec.
        # Same time alignment as far_spec/echo_spec, but with sqrt-Hann variance
        # (low inter-frame noise). Used for coherence/PSD/ENR in ResFilter.
        near_win = self.near_buffer[:self.block_size] * self._sqrt_hann_analysis
        near_spec_win = np.fft.rfft(near_win, self.fft_size).astype(np.complex64)
        self.error_spec_windowed = near_spec_win - self.echo_spec

        # Update weights — gate on far-end activity
        far_hop_energy = np.sum(far_end ** 2) / hop
        if far_hop_energy > 1e-4:  # ~ -40 dBFS, unified with far_active threshold
            self._update_weights(curr_p, mu_scale)

        self.partition_idx = (self.partition_idx + 1) % self.n_partitions
        return output.astype(np.float32)

    def _update_weights(self, curr_p: int, mu_scale):
        """NLMS weight update."""
        mu_scale_arr = np.asarray(mu_scale, dtype=np.float32)
        if mu_scale_arr.ndim == 0:
            mu_scale_arr = np.full(self.n_freqs, float(mu_scale_arr), dtype=np.float32)
        if not np.any(mu_scale_arr > 0):
            return
        # Per-bin local floor: allows low-energy mid-freq bins higher effective mu
        local_floor = self.power * 0.01 + self.delta        # per-bin 1% floor
        global_floor = np.mean(self.power) * 0.001 + self.delta  # global extreme floor
        power_floor = np.maximum(self.power, np.maximum(local_floor, global_floor))
        mu_eff = (self.mu * mu_scale_arr) / (power_floor * self.n_partitions + self.delta)
        for p in range(self.n_partitions):
            p_idx = (curr_p - p) % self.n_partitions
            grad = self.error_spec * np.conj(self.X_buf[p_idx])
            self.W[p] += mu_eff * grad
            # Time-domain constraint: fade out non-causal part (raised cosine)
            if self.enable_td_constraint:
                w_time = np.fft.irfft(self.W[p], self.fft_size).astype(np.float32)
                w_time *= self._td_window
                self.W[p] = np.fft.rfft(w_time).astype(np.complex64)

    def get_error_energy(self) -> float:
        return float(np.sum(np.abs(self.error_spec) ** 2))

    def copy_weights_from(self, src: 'PBFDAF'):
        self.W[:] = src.W


class PBFDKF(PBFDAF):
    """
    Partitioned Block Frequency-Domain Kalman Filter (PBFDKF)

    Extends PBFDAF with per-bin Kalman gain for faster convergence
    and automatic step-size adaptation.
    """

    def __init__(self, block_size: int, n_partitions: int,
                 mu: float = 0.3, delta: float = 1e-8,
                 hop_size: int = 0):
        super().__init__(block_size, n_partitions, mu, delta, hop_size)

        # P: error covariance (real, per-partition per-bin)
        self.P = np.ones((n_partitions, self.n_freqs), dtype=np.float32) * 0.01
        # Q: process noise — controls adaptation speed (two-stage)
        self.Q_high = np.ones(self.n_freqs, dtype=np.float32) * 1e-4
        self.Q_low  = np.ones(self.n_freqs, dtype=np.float32) * 1e-5
        self.Q = self.Q_high.copy()
        # R: measurement noise PSD (estimated from error)
        self.R = np.ones(self.n_freqs, dtype=np.float32) * 1e-2
        self._error_psd = np.ones(self.n_freqs, dtype=np.float32) * 1e-2
        self._alpha_r = 0.95   # faster R tracking for DT protection

        # GPT Phase 1 debug trace (off by default, zero overhead).
        # When enabled, accumulates per-frame stats to verify hypothesis:
        # "DT 期間 mu_scale 壓低但 P 仍因 K_optimal 快下降 → DT 結束後 P 偏低 → recovery 慢"
        self._enable_kx_trace = False
        self._kx_trace = []  # list of dicts, one per call to _update_weights

        # P53 Step 0 innovation-audit hook (default OFF, zero overhead).
        # When enabled, _update_weights appends per-frame per-bin arrays
        # capturing the Kalman innovation orthogonality components.
        self._enable_p53_trace = False
        self._p53_innovation_trace = []  # list of dicts of np.ndarray

    def reset(self):
        super().reset()
        self.P.fill(0.01)
        self.R.fill(1e-2)
        self._error_psd.fill(1e-2)
        self.Q[:] = self.Q_high
        # B1 fix: unconditional cleanup of dynamic P-override attrs.
        # Using try/except is safer than hasattr+delattr: if reset() is called
        # mid-countdown the _frames attr exists and is deleted; if called when
        # only the base attr remains (countdown just expired) it is also deleted;
        # if called on a freshly-constructed filter (no attr) it silently skips.
        # This prevents any case where a second reset() re-inherits a stale
        # countdown that the first reset() partially cleared.
        for attr in ('_p_max_override', '_p_max_override_frames',
                     '_p_floor_beta', '_p_floor_beta_frames'):
            try:
                delattr(self, attr)
            except AttributeError:
                pass

    def _update_weights(self, curr_p: int, mu_scale):
        """Frequency-Domain Kalman Filter weight update."""
        mu_scale_arr = np.asarray(mu_scale, dtype=np.float32)
        if mu_scale_arr.ndim == 0:
            mu_scale_arr = np.full(self.n_freqs, float(mu_scale_arr), dtype=np.float32)

        # Update measurement noise estimate from error PSD
        error_psd = np.abs(self.error_spec) ** 2
        self._error_psd = self._alpha_r * self._error_psd + (1 - self._alpha_r) * error_psd
        self.R = np.maximum(self._error_psd, self.delta)

        # Adaptive R: scale by mu_scale to break R-deadlock
        mu_mean = float(np.mean(mu_scale_arr))
        R_scale = 0.1 + 0.9 * (1.0 - mu_mean)
        self.R = self.R * R_scale

        # Q modulation: reduce Q during DT (Kalman-specific, not in NLMS)
        q_scale = 0.1 + 0.9 * mu_mean  # FS: 1.0, strong DT: 0.1
        Q_modulated = self.Q * q_scale

        # Per-bin far-end activity mask
        far_power_smooth = self.power
        far_activity_mask = far_power_smooth > (np.mean(far_power_smooth) * 0.01 + 1e-6)
        Q_floor = Q_modulated * 0.05
        Q_gated = np.where(far_activity_mask, Q_modulated, Q_floor)

        # P_MAX: overridable during EPC for faster re-convergence
        p_max = getattr(self, '_p_max_override', 0.5)
        if hasattr(self, '_p_max_override_frames'):
            self._p_max_override_frames -= 1
            if self._p_max_override_frames <= 0:
                self._p_max_override = 0.5
                del self._p_max_override_frames

        # P-floor: prevent P from collapsing to delta after long convergence
        # (which kills Kalman gain on movement). EPC sets beta=1.0 transiently
        # to force full re-tracking; steady state uses beta=0.1.
        beta = getattr(self, '_p_floor_beta', 0.1)
        if hasattr(self, '_p_floor_beta_frames'):
            self._p_floor_beta_frames -= 1
            if self._p_floor_beta_frames <= 0:
                self._p_floor_beta = 0.1
                del self._p_floor_beta_frames
        P_floor = self.Q_high * beta

        # Global denominator: sum over ALL partitions (correct Kalman theory)
        # C-parity fix: cast delta to float32 to prevent denominator from
        # promoting to float64, which cascades K_optimal to complex128 and
        # causes divergence from C's float32-only arithmetic.
        total_echo_var = np.zeros(self.n_freqs, dtype=np.float32)
        for p in range(self.n_partitions):
            p_idx = (curr_p - p) % self.n_partitions
            X = self.X_buf[p_idx]
            total_echo_var += self.P[p] * (np.abs(X) ** 2)
        denominator = total_echo_var + self.R + np.float32(self.delta)

        # P53 Step 0: per-frame innovation-audit capture (default OFF).
        # Stores per-bin arrays needed to compute r = C_obs / C_exp offline.
        if self._enable_p53_trace:
            P_diag = np.zeros(self.n_freqs, dtype=np.float32)
            far_psd_sum = np.zeros(self.n_freqs, dtype=np.float32)
            for _p in range(self.n_partitions):
                _pi = (curr_p - _p) % self.n_partitions
                P_diag += self.P[_p]
                far_psd_sum += np.abs(self.X_buf[_pi]) ** 2
            self._p53_innovation_trace.append({
                'innovation_power': error_psd.astype(np.float32).copy(),
                'R': self.R.astype(np.float32).copy(),
                'total_echo_var': total_echo_var.astype(np.float32).copy(),
                'denominator': denominator.astype(np.float32).copy(),
                'P_diag': P_diag,
                'far_psd': far_psd_sum,
            })

        # GPT Phase 1 trace: per-partition KX_optimal vs KX_scaled accumulator.
        if self._enable_kx_trace:
            kx_opt_acc = []
            kx_scaled_acc = []
            p_before_acc = []
            p_after_acc = []

        for p in range(self.n_partitions):
            p_idx = (curr_p - p) % self.n_partitions
            X = self.X_buf[p_idx]

            K_optimal = (self.P[p] * np.conj(X)) / denominator

            # Bug 2 fix: separate K for weights (scaled) and P update (unscaled)
            K_scaled = K_optimal * mu_scale_arr

            self.W[p] += K_scaled * self.error_spec

            # PR-G1 (v3.7 candidate, GPT Phase 1): blended KX for P update.
            # KX trace 2026-04-30 confirmed hypothesis on DT_st bucket: when
            # mu_scale=0 (full DT freeze), W stays put but P drops 72% per
            # cycle via K_optimal — P 與 W 不一致，DT 結束後 K 偏低、recovery 慢.
            # Blend KX_optimal (current) with KX_scaled (W-consistent) by
            # mu_mean: FS (mu=1) → 100% optimal, DT (mu=0) → 100% scaled,
            # smooth transition between. Avoids both extremes' regression
            # risk (full KX_optimal: P over-confident in DT; full KX_scaled:
            # P over-cautious in steady state with weak far / per-bin gating).
            KX_optimal = np.real(K_optimal * X).astype(np.float32)
            KX_scaled = np.real(K_scaled * X).astype(np.float32)
            mu_mean_f32 = np.float32(mu_mean)
            KX = mu_mean_f32 * KX_optimal + (np.float32(1.0) - mu_mean_f32) * KX_scaled
            if self._enable_kx_trace:
                kx_opt_acc.append(float(np.mean(KX_optimal)))
                kx_scaled_acc.append(float(np.mean(KX_scaled)))
                p_before_acc.append(float(np.mean(self.P[p])))
            self.P[p] = np.minimum(
                np.maximum((np.float32(1.0) - KX) * self.P[p] + Q_gated, P_floor),
                p_max
            )
            if self._enable_kx_trace:
                p_after_acc.append(float(np.mean(self.P[p])))

            # Time-domain constraint (raised cosine fade)
            if self.enable_td_constraint:
                w_time = np.fft.irfft(self.W[p], self.fft_size).astype(np.float32)
                w_time *= self._td_window
                self.W[p] = np.fft.rfft(w_time).astype(np.complex64)

        if self._enable_kx_trace:
            self._kx_trace.append({
                'mu_mean': float(np.mean(mu_scale_arr)),
                'kx_opt_mean': float(np.mean(kx_opt_acc)),
                'kx_scaled_mean': float(np.mean(kx_scaled_acc)),
                'p_before_mean': float(np.mean(p_before_acc)),
                'p_after_mean': float(np.mean(p_after_acc)),
                'p_p10': float(np.percentile([np.percentile(self.P[p], 10) for p in range(self.n_partitions)], 50)),
                'p_p50': float(np.median([np.median(self.P[p]) for p in range(self.n_partitions)])),
                'p_p90': float(np.percentile([np.percentile(self.P[p], 90) for p in range(self.n_partitions)], 50)),
                'q_gated_mean': float(np.mean(Q_gated)),
                'far_power_mean': float(np.mean(self.power)),
                'error_power_mean': float(np.mean(self._error_psd)),
            })

    def copy_weights_from(self, src: 'PBFDAF'):
        self.W[:] = src.W
        # Only copy filter coefficients W, not Kalman internal state (P/Q/R).
        # P/Q/R are confidence states accumulated from each filter's own
        # learning history — copying them across filters contaminates the
        # destination's Kalman gain. AEC3's shadow uses NLMS (no state to
        # copy); our PBFDKF shadow needs this protection.
        # After W copy, P may temporarily mismatch W but Kalman self-corrects
        # within a few frames.


# Backward compatibility alias
SubbandNlms = PBFDKF


class FilterErleEstimator:
    """Per-bin ERLE from adaptive filter echo estimate vs error.
    erle[k] = |echo_spec[k]|² / |error_spec[k]|²

    Key difference from erle_per_bin:
    - Does NOT use near_psd → breaks circular dependency
    - DT: erle naturally drops (speech increases error, echo_spec stays)
    """
    def __init__(self, n_freqs: int):
        self.n_freqs = n_freqs
        self.erle = np.ones(n_freqs, dtype=np.float32)
        self._alpha_rise = 0.95   # slow rise (stable convergence)
        self._alpha_drop = 0.7    # fast drop (DT protection)

    def update(self, echo_spec: np.ndarray, error_spec: np.ndarray,
               far_active: bool, dt_indicator: float) -> None:
        if not far_active:
            return
        echo_pwr = np.abs(echo_spec) ** 2
        error_pwr = np.abs(error_spec) ** 2 + 1e-10
        inst_erle = np.clip(echo_pwr / error_pwr, 0.1, 1000.0)

        # Asymmetric EMA: fast drop (DT protection), slow rise (FS convergence).
        # Bug 2 fix: dt_indicator now freezes rise (alpha_rise→1), instead of
        # inverted dt_weight that accelerated rise during DT (echo leak).
        dt_factor = float(np.clip(dt_indicator, 0.0, 1.0))
        alpha_rise_eff = self._alpha_rise + (1.0 - self._alpha_rise) * dt_factor
        alpha = np.where(
            inst_erle < self.erle,
            self._alpha_drop,
            alpha_rise_eff,
        )
        self.erle = alpha * self.erle + (1.0 - alpha) * inst_erle

        # 3-bin smoothing + cap
        kernel = np.array([0.25, 0.5, 0.25], dtype=np.float32)
        self.erle = np.convolve(self.erle, kernel, mode='same').astype(np.float32)
        self.erle = np.clip(self.erle, 0.5, 200.0)

    def reset(self):
        self.erle.fill(1.0)


class FullbandErleEstimator:
    """Broadband ERLE for cross-validation confidence.
    Uses near_psd/error_psd broadband mean — stable but slow, FS-only update.
    """
    def __init__(self):
        self.fb_erle = 1.0
        self._alpha = 0.97

    def update(self, near_power: float, error_power: float,
               far_active: bool, dt_indicator: float) -> None:
        if not far_active or dt_indicator > 0.3 or near_power < 1e-8:
            return
        inst = np.clip(near_power / (error_power + 1e-10), 0.5, 100.0)
        self.fb_erle = self._alpha * self.fb_erle + (1.0 - self._alpha) * inst

    def reset(self):
        self.fb_erle = 1.0


def compute_erle_confidence(erle_l1: np.ndarray, fb_erle: float) -> float:
    """Compare FilterErle mean (L1) with FullbandErle (L2).
    Returns confidence in [0, 1]: 1 = consistent, 0 = divergent.
    """
    l1_mean = float(np.mean(erle_l1))
    if l1_mean < 0.5 or fb_erle < 0.5:
        return 0.0
    log_diff = abs(np.log(l1_mean + 1e-10) - np.log(fb_erle + 1e-10))
    return float(np.exp(-log_diff / 2.0))


class HighPassFilter:
    """2nd-order Butterworth IIR high-pass filter (bilinear transform).

    Removes DC offset, 50/60Hz hum, and low-frequency rumble.
    12 dB/octave rolloff. Processes sample-by-sample with two delay states.
    """

    def __init__(self, cutoff_hz: float, sample_rate: int):
        # Bilinear transform: pre-warp analog frequency
        wc = 2.0 * np.pi * cutoff_hz / sample_rate
        wc_w = np.tan(wc / 2.0)
        k = wc_w * wc_w
        sqrt2 = np.sqrt(2.0)
        norm = 1.0 / (1.0 + sqrt2 * wc_w + k)

        # Transfer function coefficients (Direct Form II)
        self.b0 = norm
        self.b1 = -2.0 * norm
        self.b2 = norm
        self.a1 = 2.0 * (k - 1.0) * norm
        self.a2 = (1.0 - sqrt2 * wc_w + k) * norm

        # Delay states
        self.z1 = 0.0
        self.z2 = 0.0

    def process(self, x: np.ndarray) -> np.ndarray:
        """Process a block of samples through the HP filter."""
        out = np.empty_like(x)
        b0, b1, b2, a1, a2 = self.b0, self.b1, self.b2, self.a1, self.a2
        z1, z2 = self.z1, self.z2
        for i in range(len(x)):
            xi = float(x[i])
            yi = b0 * xi + z1
            z1 = b1 * xi - a1 * yi + z2
            z2 = b2 * xi - a2 * yi
            out[i] = yi
        self.z1, self.z2 = z1, z2
        return out

    def reset(self):
        self.z1 = 0.0
        self.z2 = 0.0


class SaturationDetector:
    """Detects speaker clipping/saturation in audio signals.

    Returns a smoothed saturation_level in [0, 1] indicating how much
    non-linear distortion is present. Also provides soft-clipping to
    model the speaker's saturation behavior for the adaptive filter.
    """

    def __init__(self, threshold: float = 0.95):
        self.threshold = threshold
        self.saturation_level = 0.0
        self.alpha_attack = 0.3    # Fast attack when saturation detected
        self.alpha_release = 0.98  # Slow release (echo path retains saturation effects)

    def detect(self, signal: np.ndarray) -> float:
        """Detect saturation level in signal. Returns smoothed level in [0, 1]."""
        n = len(signal)
        if n == 0:
            return self.saturation_level

        abs_sig = np.abs(signal)
        # Count clipped samples
        clip_count = np.sum(abs_sig > self.threshold)

        # Count consecutive identical peak samples (digital clipping signature)
        consec_count = 0
        high_mask = abs_sig > self.threshold * 0.8
        for i in range(1, n):
            if high_mask[i] and high_mask[i - 1] and abs(signal[i] - signal[i - 1]) < 1e-6:
                consec_count += 1

        raw_sat = min((clip_count + 2 * consec_count) / n, 1.0)

        # Asymmetric EMA
        if raw_sat > self.saturation_level:
            alpha = self.alpha_attack
        else:
            alpha = self.alpha_release
        self.saturation_level = alpha * self.saturation_level + (1.0 - alpha) * raw_sat
        return self.saturation_level

    @staticmethod
    def soft_clip(signal: np.ndarray, knee: float = 0.8) -> np.ndarray:
        """Soft-clip signal to model speaker saturation behavior.

        Below knee: pass through. Above knee: tanh compression.
        """
        out = signal.copy()
        abs_sig = np.abs(signal)
        mask = abs_sig >= knee
        if np.any(mask):
            sign = np.sign(signal[mask])
            excess = abs_sig[mask] - knee
            scale = 1.0 - knee
            compressed = knee + np.tanh(excess / max(scale, 1e-6)) * scale
            out[mask] = sign * compressed
        return out

    def reset(self):
        self.saturation_level = 0.0


class ResFilter:
    """
    Residual Echo Suppressor (Post-Filter)

    Uses EER-based spectral suppression with OLA + sqrt-Hann windowing
    to avoid frame-boundary artifacts and musical noise.
    """

    # Backward-compat: _using_render_based / _render_based_hold now live on self._residual_est
    @property
    def _using_render_based(self) -> bool:
        return self._residual_est.using_render_based

    @_using_render_based.setter
    def _using_render_based(self, val: bool) -> None:
        # Legacy code path (e.g. AEC.process EPC render-forced) sets this directly;
        # forward to estimator state to keep one source of truth.
        self._residual_est._using_render_based = bool(val)

    def __init__(self, block_size: int, n_freqs: int, g_min_db: float = -20.0,
                 over_sub: float = 1.5, alpha: float = 0.8,
                 enable_cng: bool = False,
                 max_drop_db_per_frame: float = 6.0,
                 max_rise_db_per_frame: float = 3.0,
                 enable_spectral_floor: bool = True,
                 spectral_floor_db: float = -25.0,
                 ne_protect_db: float = -10.0,
                 frame_size: int = 0,
                 hop_size: int = 0,
                 echo_method: str = "coherence",
                 gain_type: str = "spectral_sub",
                 enable_reverb: bool = False,
                 reverb_decay: float = 0.5,
                 reverb_gain: float = 1.0,
                 alpha_echo_psd: float = 0.7,
                 alpha_error_psd: float = 0.8,
                 enr_scale: float = 1.0,
                 startup_dt_min_ne_scale: float = 1.0,
                 startup_dt_gain_floor: float = 1.0,
                 startup_dt_noise_floor_scale: float = 1.0,
                 sample_rate: int = 16000,
                 capture_stages: bool = False,
                 plan_a_kernel_tight: bool = True,
                 plan_a_hf_cap_2k: bool = True,
                 plan_a_stat_mask_7k: bool = True,
                 hf_cap_conditional: bool = False,
                 hf_cap_metric_threshold: float = 0.30,
                 plan_b_dt_per_bin_gamma: bool = False,
                 use_mic_excess_evidence: bool = False,
                 consume_filter_state: bool = False,
                 unified_gain_floor: bool = False,
                 state_driven_epc_dt_cap: bool = False,
                 dt_per_bin_unified: bool = False,
                 noise_floor_refined: bool = False,
                 cap2_fs_loosen: bool = False,
                 per_band_enr: bool = False,
                 enr_t_ne_per_band: tuple = (1.5, 1.5, 1.5),
                 enr_s_ne_per_band: tuple = (2.5, 2.5, 2.5),
                 dt_ne_compression_fix: bool = False,
                 dt_ne_state_scale: dict = None,
                 dt_ne_per_bin_thresh: float = 0.5,
                 dt_ne_per_bin_scale: float = 2.0):
        self._plan_a_kernel_tight = plan_a_kernel_tight
        self._plan_b_dt_per_bin_gamma = plan_b_dt_per_bin_gamma
        self._plan_a_hf_cap_2k = plan_a_hf_cap_2k
        self._plan_a_stat_mask_7k = plan_a_stat_mask_7k
        self._hf_cap_conditional = hf_cap_conditional
        self._hf_cap_metric_threshold = hf_cap_metric_threshold
        self._use_mic_excess_evidence = use_mic_excess_evidence
        self._consume_filter_state = consume_filter_state
        self._unified_gain_floor = unified_gain_floor
        self._state_driven_epc_dt_cap = state_driven_epc_dt_cap
        self._dt_per_bin_unified = dt_per_bin_unified
        self._noise_floor_refined = noise_floor_refined
        self._cap2_fs_loosen = cap2_fs_loosen
        self._per_band_enr = per_band_enr
        self._enr_t_ne_per_band_cfg = tuple(enr_t_ne_per_band)
        self._enr_s_ne_per_band_cfg = tuple(enr_s_ne_per_band)
        # v3.15 §1.2 — DT-NE compression fix flags (default OFF byte-equal).
        self._dt_ne_compression_fix = dt_ne_compression_fix
        self._dt_ne_state_scale = dt_ne_state_scale if dt_ne_state_scale is not None else {
            'idle':            2.0,
            'startup':         2.0,
            'coarse_learning': 2.0,
            'refined_usable':  1.0,
            'diverged':        2.0,
            'suspicious_dt':   1.0,
        }
        self._dt_ne_per_bin_thresh = float(dt_ne_per_bin_thresh)
        self._dt_ne_per_bin_scale = float(dt_ne_per_bin_scale)
        self.block_size = block_size          # FFT size (power of 2)
        self.sample_rate = sample_rate        # Hz, used for freq → bin conversion
        self.frame_size = frame_size if frame_size > 0 else block_size  # WOLA frame
        self.hop_size = hop_size if hop_size > 0 else self.frame_size // 2
        self.n_freqs = n_freqs
        self.g_min = 10 ** (g_min_db / 20)
        self.over_sub = over_sub
        self.alpha = alpha
        self.ne_protect_db = ne_protect_db
        self.alpha_echo_psd = alpha_echo_psd
        self.alpha_error_psd = alpha_error_psd
        self.enr_scale = enr_scale           # ENR threshold scale (1.0=AEC3 defaults)
        self.startup_dt_min_ne_scale = startup_dt_min_ne_scale
        self.startup_dt_gain_floor = startup_dt_gain_floor
        self.startup_dt_noise_floor_scale = startup_dt_noise_floor_scale

        # C1-C4: precomputed per-bin constants (invariant after init)
        freq_res = sample_rate / block_size
        f_bins = np.arange(n_freqs, dtype=np.float32) * freq_res
        # C1: stationary DT mask. v3.8.4: upper edge extended 4 kHz → 7 kHz
        # so fricatives / sibilants (4–7 kHz) get DT protection during
        # stationary DT (was: silently zero above 4 kHz).
        self._stat_dt_mask = np.zeros(n_freqs, dtype=np.float32)
        self._stat_dt_mask[(f_bins >= 300) & (f_bins <= 3000)] = 0.8
        low = (f_bins > 100) & (f_bins < 300)
        self._stat_dt_mask[low] = 0.8 * ((f_bins[low] - 100.0) / 200.0)
        # P1.0 toggle: plan_a_stat_mask_7k=False reverts to v3.8.3 behaviour
        # (no high-band coverage; mask was silently zero above 3 kHz).
        if self._plan_a_stat_mask_7k:
            high = (f_bins > 3000) & (f_bins < 7000)
            self._stat_dt_mask[high] = 0.8 * np.maximum(
                0.0, 1.0 - (f_bins[high] - 3000.0) / 4000.0)
        # C2: ENR blend array
        self._enr_blend = np.clip((np.arange(n_freqs, dtype=np.float32) - 5) / 5, 0, 1)
        # v3.14 Arc-R Sprint S1: per-band ENR threshold per-bin arrays.
        # Pre-built once from the LF/MF/HF tuple values using the same band
        # boundaries (1 kHz / 4 kHz) as P.S3 adaptive per-band ERL EMA so
        # the two arcs operate on consistent frequency partitions. When
        # `_per_band_enr=False` (default) these arrays are unused → no
        # behavioural change → byte-equal to baseline.
        _b1k = max(1, min(int(round(1000.0 / freq_res)), n_freqs - 2))
        _b4k = max(_b1k + 1, min(int(round(4000.0 / freq_res)), n_freqs - 1))
        self._enr_per_band_b1k = _b1k
        self._enr_per_band_b4k = _b4k
        _t_lf, _t_mf, _t_hf = self._enr_t_ne_per_band_cfg
        _s_lf, _s_mf, _s_hf = self._enr_s_ne_per_band_cfg
        self._enr_t_ne_pb = np.empty(n_freqs, dtype=np.float32)
        self._enr_t_ne_pb[:_b1k] = float(_t_lf)
        self._enr_t_ne_pb[_b1k:_b4k] = float(_t_mf)
        self._enr_t_ne_pb[_b4k:] = float(_t_hf)
        self._enr_s_ne_pb = np.empty(n_freqs, dtype=np.float32)
        self._enr_s_ne_pb[:_b1k] = float(_s_lf)
        self._enr_s_ne_pb[_b1k:_b4k] = float(_s_mf)
        self._enr_s_ne_pb[_b4k:] = float(_s_hf)
        # C3: frequency bin indices
        self._hf_cap_bin = min(int(500.0 / freq_res), n_freqs - 1)
        # v3.8.4: secondary cap anchor at 2 kHz so vowel formants (1–3 kHz)
        # are not dragged down by the 500 Hz bin's gain.
        self._hf_cap_bin_2k = min(int(2000.0 / freq_res), n_freqs - 1)
        # C4: harmonic distortion bin bounds
        self._harm_lf_start = max(1, int(100.0 / freq_res))
        self._harm_lf_end = min(int(500.0 / freq_res), n_freqs - 1)
        self._harm_hf_start = int(1000.0 / freq_res)
        self._harm_hf_end = min(int(4000.0 / freq_res), n_freqs - 1)
        # Round 4: voice-band mask (300–3000 Hz) for per-bin RES diagnostics + R2.
        self._voice_band_mask = ((f_bins >= 300.0) & (f_bins <= 3000.0))
        self._voice_band_idx = np.where(self._voice_band_mask)[0]
        # Round 4: per-bin trace caches (audio-passive)
        self._diag_coh2_last = np.zeros(n_freqs, dtype=np.float32)
        self._diag_nearend_est_last = np.zeros(n_freqs, dtype=np.float32)
        self._diag_residual_echo_psd_last = np.zeros(n_freqs, dtype=np.float32)
        self._diag_round4 = {}
        # Round 5: per-stage gain means (voice-band), audio-passive.
        # Indices: 0=softgate_emr, 1=spectral_floor, 2=epc_dt_cap, 3=quiet_mask,
        #          4=3bin_smooth, 5=hf_cap, 6=pre_temporal, 7=post_temporal,
        #          8=after_noise_lift (final gain_smooth)
        self._diag_round5_stages = np.zeros(9, dtype=np.float32)

        # Per-bin gain capture (full vectors, opt-in via capture_stages)
        self._capture_stages = capture_stages
        self._stage_gains = {}
        # v3.8.4: cached per-bin DT confidence from compute stage, read by
        # postprocess HF-cap "high-band NE present?" gate.
        self._dt_per_bin_last = np.zeros(n_freqs, dtype=np.float32)

        # P4B diag (set by gain_compute / gain_postprocess each frame).
        self._p4b_dt_per_bin_mean = 0.0
        self._p4b_dt_per_bin_hf_mean = 0.0
        self._p4b_coh2_hf_mean = 0.0
        self._p4b_effective_dt = 0.0
        self._p4b_is_stationary_dt = 0
        self._p4b_gain_hf_mean = 1.0
        self._p4b_res_echo_hf_mean_db = -120.0

        # RES v2: direct echo estimation + Wiener gain + reverb
        self.echo_method = echo_method       # "coherence" or "direct"
        self.gain_type = gain_type           # "spectral_sub" or "wiener"
        self.enable_reverb = enable_reverb
        self.reverb_decay = reverb_decay
        self.reverb_gain = reverb_gain

        # --- Immutable config (set once, not cleared by reset) ---
        self.alpha_coh = 0.65           # Cross-PSD smoothing (TC≈50ms)
        self.enable_cng = enable_cng
        self.alpha_noise = 0.98
        self.max_drop_ratio = 10 ** (max_drop_db_per_frame / 20)
        self.max_rise_ratio = 10 ** (max_rise_db_per_frame / 20)
        self.enable_spectral_floor = enable_spectral_floor
        self.spectral_floor_ratio = 10 ** (spectral_floor_db / 20)
        self.alpha_envelope = 0.95
        self.window = np.sqrt(np.hanning(self.frame_size)).astype(np.float32)

        # --- Runtime state arrays (allocated once, cleared by reset) ---
        self.gain_smooth = np.full(n_freqs, self.g_min, dtype=np.float32)
        self.echo_psd = np.zeros(n_freqs, dtype=np.float32)
        self.error_psd = np.zeros(n_freqs, dtype=np.float32)
        self.S_fe = np.zeros(n_freqs, dtype=np.complex64)
        self.S_ff = np.zeros(n_freqs, dtype=np.float32)
        self.S_ee = np.zeros(n_freqs, dtype=np.float32)
        self.near_psd = np.zeros(n_freqs, dtype=np.float32)
        self.near_psd_buf = np.zeros((4, n_freqs), dtype=np.float32)
        self.reverb_psd = np.zeros(n_freqs, dtype=np.float32)
        self.noise_psd = np.zeros(n_freqs, dtype=np.float32)
        self._coh2_smooth = np.zeros(n_freqs, dtype=np.float32)
        self.error_envelope = np.ones(n_freqs, dtype=np.float32)
        self.input_buf = np.zeros(self.frame_size, dtype=np.float32)
        self.ola_buf = np.zeros(self.frame_size, dtype=np.float32)

        # Multi-ERLE estimators
        self._filter_erle_est = FilterErleEstimator(n_freqs)
        self._fb_erle_est = FullbandErleEstimator()

        # Per-frame diagnostic stats (disabled by default; call enable_stats() to activate)
        self._stats = None
        # Durable audit counter substrate (Phase 3B v3 S7+, default-OFF zero-cost).
        # Use enable_audit_counters() to activate; get_audit_counters() to retrieve.
        self._audit_counters = None
        self._stats_last_enr = 0.0
        self._stats_last_nearend = 0.0
        self._stats_last_res_psd = 0.0
        self._stats_last_min_ne = 0.0
        # _stats_last_using_render wired from self._residual_est.using_render_based
        # in get_stats accumulator. Removed in v3.8.x (never-wired after residual
        # estimator refactor): render_echo, linear_res, want_render, should_render_v1.
        self._stats_last_using_render = False
        self._stats_last_ne_g_floor = 0.0
        self._stats_last_spectral_g_min = 0.0
        self._stats_last_gain_before_floor = 1.0
        self._stats_last_gain_after_floor = 1.0
        self._stats_last_gain_after_smoothing = 1.0
        self._stats_last_noise_floor_gain = 0.0
        self._stats_last_noise_psd = 0.0
        self._stats_last_spec_pwr = 0.0
        self._stats_last_nfl_lifted = False
        self._last_effective_dt = 0.0   # exported for external startup_dt detection

        # Residual echo attribution (stage 1 + stage 2 of original inline logic).
        # Default mode='legacy' → bit-exact parity. Phase B2 ablation flips to 'split'.
        self._residual_est = ResidualEchoEstimator(n_freqs, mode='legacy')

        # E1: call reset() to initialize all runtime scalar state
        # from a single source of truth (avoids init/reset divergence)
        self.reset()

    def reset(self, preserve_long_window_ema: bool = False):
        """Reset all runtime state. Arrays are .fill(0), scalars are set.

        preserve_long_window_ema (v3.10.2): forwarded to ResidualEchoEstimator
        so the long-window far-PSD EMA survives recovery resets that are
        triggered by bad-filter-state plateaus / delay_first acquisition —
        the EMA is input-side and should not be discarded.
        """
        self.gain_smooth.fill(self.g_min)
        self.echo_psd.fill(0)
        self.error_psd.fill(0)
        self.S_fe.fill(0)
        self.S_ff.fill(0)
        self.S_ee.fill(0)
        self.near_psd.fill(0)
        self.near_psd_buf.fill(0)
        self.near_psd_idx = 0
        self.reverb_psd.fill(0)
        self.noise_psd.fill(0)
        self._noise_initialized = False
        self._coh2_smooth.fill(0)
        self.error_envelope.fill(1.0)
        self.input_buf.fill(0)
        self.ola_buf.fill(0)
        self.far_activity = 0.0
        self._nonlinear_frames = 0
        # _render_based_hold + _using_render_based moved to self._residual_est
        if hasattr(self, '_residual_est'):
            self._residual_est.reset(preserve_long_window_ema=preserve_long_window_ema)
        self._diag_gain_mean = 1.0
        self._diag_gain_min = 1.0
        self._diag_effective_g_min = 1.0
        self._diag_far_activity = 0.0
        self._diag_echo_psd_mean = 0.0
        self._diag_error_psd_mean = 0.0
        self._filter_erle_est.reset()
        self._fb_erle_est.reset()

    def enable_stats(self):
        """Enable per-frame DT statistics collection (zero cost when disabled)."""
        self._stats = {
            # existing fields
            'total_frames': 0,
            'dt_active': 0, 'epc_dt': 0,
            'low_erle_dt': 0, 'startup_dt': 0, 'any_special_dt': 0,
            'dt_erle_sum': 0.0, 'dt_gain_sum': 0.0,
            'dt_enr_sum': 0.0, 'dt_res_sum': 0.0, 'dt_ne_sum': 0.0,
            # (2) filter convergence/divergence
            'filter_once_converged_dt': 0,
            'filter_diverged_dt': 0,
            'shadow_better_dt': 0,
            'e2_main_y2_sum': 0.0, 'e2_shadow_y2_sum': 0.0, 'e2_shadow_e2_main_sum': 0.0,
            # (3) usability candidates
            'usable_v1': 0, 'usable_v2': 0, 'usable_v3': 0, 'usable_v4': 0,
            'unusable_dt': 0,
            # (4) residual echo model
            'using_render_based_dt': 0,
            'min_ne_sum': 0.0,
            # (5) startup_dt gain stage diagnostics (accumulated for not filter_converged frames)
            'startup_dt_once_conv': 0,   # DT frames where not filter_once_converged
            'st_ne_g_floor_sum': 0.0,
            'st_spectral_g_min_sum': 0.0,
            'st_gain_before_floor_sum': 0.0,
            'st_gain_after_floor_sum': 0.0,
            'st_gain_after_smoothing_sum': 0.0,
            # (6) noise_floor_gain diagnostics (accumulated for not filter_once_converged frames)
            'st_nfl_noise_floor_gain_sum': 0.0,
            'st_nfl_noise_psd_sum': 0.0,
            'st_nfl_spec_pwr_sum': 0.0,
            'st_nfl_lifted_count': 0,
            'st_nfl_final_gain_sum': 0.0,
        }
        self._stats_last_min_ne = 0.0
        self._stats_last_using_render = False
        self._stats_last_ne_g_floor = 0.0
        self._stats_last_spectral_g_min = 0.0
        self._stats_last_gain_before_floor = 1.0
        self._stats_last_gain_after_floor = 1.0
        self._stats_last_gain_after_smoothing = 1.0
        self._stats_last_noise_floor_gain = 0.0
        self._stats_last_noise_psd = 0.0
        self._stats_last_spec_pwr = 0.0
        self._stats_last_nfl_lifted = False

    def enable_audit_counters(self):
        """Enable durable RES audit counter substrate (Phase 3B v3 S7+).

        Counters accumulate across frames within a stream; consumers (eval
        harness) read via get_audit_counters() at end-of-stream. Zero cost
        when not enabled.

        S7 (Option α) target: measure mean reduction of `dt_per_bin` legacy
        vs unified-hypothetical across FS bins (coh² < 0.1) in the
        `filter_converged AND _lw_ready AND epc_active` slice. See design
        [docs/v3_12_phase3b_v3_design.md §6.1](../docs/v3_12_phase3b_v3_design.md).
        """
        self._audit_counters = {
            'total_frames': 0,
            # S7 Option α — dt_per_bin path classification
            's7_legacy_path_frames': 0,
            's7_f31v3_path_frames': 0,
            's7_planb_path_frames': 0,
            # Legacy-path sub-reasons (sum may exceed s7_legacy_path_frames
            # because a frame can hit multiple reasons, e.g. not_converged AND
            # not_lw_ready)
            's7_legacy_epc_active_frames': 0,
            's7_legacy_not_converged_frames': 0,
            's7_legacy_not_lw_ready_frames': 0,
            's7_legacy_target_other_frames': 0,
            # Primary S7 target slice: filter_converged AND _lw_ready AND epc_active
            's7_target_fs_bin_count': 0,
            's7_target_fs_bin_legacy_sum': 0.0,
            's7_target_fs_bin_unified_sum': 0.0,
            # Cross-validation slice: filter_converged AND _lw_ready AND NOT epc_active
            # (legacy path only reachable when use_mic_excess_evidence=False)
            's7_alt_fs_bin_count': 0,
            's7_alt_fs_bin_legacy_sum': 0.0,
            's7_alt_fs_bin_unified_sum': 0.0,
            # S8 (Phase 3B v4) — downstream-clamp audit. Each counter
            # aggregates over FS bins (coh² < 0.1) across all frames.
            # Stage 1 4-cap chain (line ~2138/2144/2151/2163): per-cap
            # binding counts (how often the cap clamped a FS bin) +
            # reduction sums (sum of log10(pre/post) on those bins).
            's8_stage1_fs_bin_total': 0,        # total FS bin opportunities
            's8_cap1_echo_x2_binding': 0,
            's8_cap1_echo_x2_reduction_sum': 0.0,
            's8_cap2_err_mult_binding': 0,
            's8_cap2_err_mult_reduction_sum': 0.0,
            's8_cap3_dt_suppress_binding': 0,
            's8_cap3_dt_suppress_reduction_sum': 0.0,
            's8_cap4_render_ceil_binding': 0,
            's8_cap4_render_ceil_reduction_sum': 0.0,
            # Nearend_est floor 4-way binding (line ~2363/2372/2377):
            # of the four candidate floor sources, which is the MAX
            # (the actual binding) for each FS bin.
            's8_nef_raw_count': 0,         # raw_nearend_est * dt_shaped wins
            's8_nef_noise_floor_count': 0, # noise_floor_psd wins
            's8_nef_min_ne_count': 0,      # min_ne_from_dt wins
            's8_nef_ne_physical_count': 0, # ne_physical_floor wins
            # S9 Phase 3B v5 pre-implementation audit — noise_floor_psd
            # refinement. For FS bins (coh² < 0.1) where the current
            # baseline winner is noise_floor_psd (s8_nef_noise_floor_count
            # ≈ 43% global), compute hypothetical winners under two
            # candidate refinements:
            #   A.1: scalar × 0.001 (10× lower than baseline 0.01)
            #   A.2: per-bin error_psd × 0.005 (scalar→per-bin)
            # `release_to_raw`: bin previously bound by noise_floor now
            #   bound by raw_nearend_est (good — FS-honest evidence wins).
            # `shift_to_min_ne`: bin shifts to min_ne_from_dt (neutral —
            #   carrier shifts from #1 to #2, doesn't reduce overall
            #   suppression in FS).
            # `stays_floor`: still bound by noise_floor (new value still
            #   highest).
            # `reduction_db_sum`: 10·log10(old_nearend_est /
            #   new_nearend_est) summed over bins where any candidate
            #   wins changed — magnitude of the nearend_est drop.
            's9_a1_release_to_raw': 0,
            's9_a1_shift_to_min_ne': 0,
            's9_a1_shift_to_phys': 0,
            's9_a1_stays_floor': 0,
            's9_a1_reduction_db_sum': 0.0,
            's9_a1_reduction_count': 0,
            's9_a2_release_to_raw': 0,
            's9_a2_shift_to_min_ne': 0,
            's9_a2_shift_to_phys': 0,
            's9_a2_stays_floor': 0,
            's9_a2_reduction_db_sum': 0.0,
            's9_a2_reduction_count': 0,
            # Out-of-FS sanity: bins currently NOT bound by noise_floor
            # but where A.1 / A.2 would dethrone the current winner
            # (only possible if candidate noise_floor RISES above winner
            # — A.2 in high-error_psd bins can do this).
            's9_a1_intrudes_outside_floor_baseline': 0,
            's9_a2_intrudes_outside_floor_baseline': 0,
            # S9-C pre-audit — joint noise_floor + min_ne_from_dt attack.
            # S9-A finding: A.2 only releases 11.5% of FS floor bins to
            # raw_NE; 88% shifts to min_ne_from_dt (F3.1 v3 floor =
            # error_psd * dt_shaped). Joint candidates attack both
            # floors simultaneously in FS bins:
            #   C.1: noise_floor → error_psd × 0.005 (=A.2)
            #         + min_ne_from_dt × 0.1 (10× reduction)
            #   C.2: noise_floor → error_psd × 0.005 (=A.2)
            #         + min_ne_from_dt → 0   (eliminated in FS)
            # `release_to_raw`: FS bin previously bound by ANY floor
            #   (noise_floor / min_ne_from_dt / ne_physical) now bound
            #   by raw_nearend_est. This is the real outcome we want —
            #   FS bins should not be floored.
            # `still_floor` / `still_min_ne` / `still_phys`: residual
            #   floor binding under candidate (shows which floors still
            #   block release).
            's9c_c1_release_to_raw': 0,
            's9c_c1_still_floor': 0,
            's9c_c1_still_min_ne': 0,
            's9c_c1_still_phys': 0,
            's9c_c1_reduction_db_sum': 0.0,
            's9c_c1_reduction_count': 0,
            's9c_c2_release_to_raw': 0,
            's9c_c2_still_floor': 0,
            's9c_c2_still_min_ne': 0,
            's9c_c2_still_phys': 0,
            's9c_c2_reduction_db_sum': 0.0,
            's9c_c2_reduction_count': 0,
            # FS bins baseline-bound by ANY floor (denominator for
            # release_pct on the C candidates).
            's9c_baseline_any_floor_count': 0,
            # S9-D sanity — attack ALL three nearend_est floors:
            #   noise_floor → error_psd × 0.005 (A.2)
            #   min_ne_from_dt → 0
            #   ne_physical_floor → 0
            # Stack reduces to [raw_NE * dt_shaped, noise_floor_A2, 0, 0].
            # Winner can only be 0 (raw) or 1 (tiny noise_floor at
            # 0.005×error_psd). If raw_NE * dt_shaped >
            # 0.005×error_psd → release_to_raw; else still_floor.
            # Goal: prove that the nearend_est stack fully controls FS
            # release rate (FS release ≥90% confirms no hidden 4th
            # carrier). Read-only; informs Phase 4 RES canonical
            # refactor design lock.
            's9d_release_to_raw': 0,
            's9d_still_floor': 0,
            's9d_reduction_db_sum': 0.0,
            's9d_reduction_count': 0,
        }

    def get_audit_counters(self):
        """Return audit counter dict (or None if not enabled)."""
        return self._audit_counters

    def get_stage_gains(self):
        """Return dict of per-bin gain vectors captured this frame.

        Empty dict unless `capture_stages=True` was passed to __init__.
        Keys: '01_softgate_emr', '02_spectral_floor', '03_epc_dt_cap',
        '04_quiet_mask', '05_3bin_smooth', '06_hf_cap', '07_pre_temporal',
        '08_post_temporal'. Vectors are np.float32 length n_freqs.
        """
        return self._stage_gains

    def get_stats(self):
        """Return aggregated DT stats dict, or None if not enabled / no DT frames."""
        s = self._stats
        if s is None or s['dt_active'] == 0:
            return None
        n, t = s['dt_active'], s['total_frames']
        return {
            'total_frames': t, 'dt_active': n, 'dt_pct': n / t,
            'epc_dt': s['epc_dt'], 'epc_dt_pct': s['epc_dt'] / n,
            'low_erle_dt': s['low_erle_dt'], 'low_erle_dt_pct': s['low_erle_dt'] / n,
            'startup_dt': s['startup_dt'], 'startup_dt_pct': s['startup_dt'] / n,
            'any_special_dt': s['any_special_dt'], 'any_special_dt_pct': s['any_special_dt'] / n,
            'mean_erle_factor': s['dt_erle_sum'] / n,
            'mean_gain': s['dt_gain_sum'] / n,
            'mean_enr': s['dt_enr_sum'] / n,
            'mean_residual_echo_psd': s['dt_res_sum'] / n,
            'mean_nearend_est': s['dt_ne_sum'] / n,
            # (2) filter convergence/divergence
            'filter_once_converged_pct': s['filter_once_converged_dt'] / n,
            'filter_diverged_pct': s['filter_diverged_dt'] / n,
            'shadow_better_pct': s['shadow_better_dt'] / n,
            'mean_e2_main_y2': s['e2_main_y2_sum'] / n,
            'mean_e2_shadow_y2': s['e2_shadow_y2_sum'] / n,
            'mean_e2_shadow_e2_main': s['e2_shadow_e2_main_sum'] / n,
            # (3) usability
            'usable_v1_pct': s['usable_v1'] / n,
            'usable_v2_pct': s['usable_v2'] / n,
            'usable_v3_pct': s['usable_v3'] / n,
            'usable_v4_pct': s['usable_v4'] / n,
            'unusable_dt_pct': s['unusable_dt'] / n,
            # (4) residual echo model
            'using_render_based_pct': s['using_render_based_dt'] / n,
            'mean_min_ne_from_dt': s['min_ne_sum'] / n,
            # (5) startup_dt gain stage diagnostics
            'startup_dt_once_conv_pct': s['startup_dt_once_conv'] / n,
            'mean_ne_g_floor': s['st_ne_g_floor_sum'] / s['startup_dt'] if s['startup_dt'] > 0 else 0.0,
            'mean_spectral_g_min': s['st_spectral_g_min_sum'] / s['startup_dt'] if s['startup_dt'] > 0 else 0.0,
            'mean_gain_before_floor': s['st_gain_before_floor_sum'] / s['startup_dt'] if s['startup_dt'] > 0 else 0.0,
            'mean_gain_after_floor': s['st_gain_after_floor_sum'] / s['startup_dt'] if s['startup_dt'] > 0 else 0.0,
            'mean_gain_after_smoothing': s['st_gain_after_smoothing_sum'] / s['startup_dt'] if s['startup_dt'] > 0 else 0.0,
            # (6) noise_floor_gain diagnostics (denominator: startup_dt_once_conv frames)
            'mean_noise_floor_gain': s['st_nfl_noise_floor_gain_sum'] / s['startup_dt_once_conv'] if s['startup_dt_once_conv'] > 0 else 0.0,
            'mean_noise_psd': s['st_nfl_noise_psd_sum'] / s['startup_dt_once_conv'] if s['startup_dt_once_conv'] > 0 else 0.0,
            'mean_spec_pwr': s['st_nfl_spec_pwr_sum'] / s['startup_dt_once_conv'] if s['startup_dt_once_conv'] > 0 else 0.0,
            'nfl_lifted_pct': s['st_nfl_lifted_count'] / s['startup_dt_once_conv'] if s['startup_dt_once_conv'] > 0 else 0.0,
            'mean_final_gain_nfl': s['st_nfl_final_gain_sum'] / s['startup_dt_once_conv'] if s['startup_dt_once_conv'] > 0 else 0.0,
        }

    # ── Stage methods (Round 6 refactor) ─────────────────────────────────
    # Each _stage_* method handles a logical block of the suppressor pipeline.
    # All share self.* state; arguments are local-only values that flow
    # between stages. Behavior identical to the pre-refactor inline code.

    def _stage_residual_model(self, *, coh2, far_spec, far_power, erle_factor,
                                dt_for_fs, epc_active, saturation_level,
                                filter_converged, erl_estimate, near_spec,
                                is_stationary_dt, aec_state, echo_pwr_linear):
        """Stage 1: residual_echo_psd attribution + 4 caps + reverb tail + echo boost.

        Returns residual_echo_psd (np.ndarray or None). Mutates near_psd buffers,
        reverb_psd, _nonlinear_frames, and stats fields.
        """
        residual_echo_psd = None
        if self.echo_method == "direct" and near_spec is not None:
            near_pwr = np.abs(near_spec) ** 2
            self.near_psd_buf[self.near_psd_idx] = near_pwr
            self.near_psd_idx = (self.near_psd_idx + 1) % 4
            self.near_psd = np.mean(self.near_psd_buf, axis=0)

            # Stage 1+2 of residual echo attribution: delegated to ResidualEchoEstimator.
            residual_echo_psd = self._residual_est.attribute(
                echo_psd=self.echo_psd, error_psd=self.error_psd,
                coh2=coh2, far_spec=far_spec, far_power=far_power,
                erle_factor=erle_factor, dt_for_fs=dt_for_fs,
                far_activity=self.far_activity,
                epc_active=epc_active, saturation_level=saturation_level,
                filter_converged=filter_converged, erl_estimate=erl_estimate,
                filter_erle=self._filter_erle_est, fb_erle=self._fb_erle_est,
                aec_state=aec_state,
            )

            # Nonlinear echo mode: harmonics from speaker distortion
            if saturation_level > 0.3:
                self._nonlinear_frames += 1
            else:
                self._nonlinear_frames = max(0, self._nonlinear_frames - 1)
            is_nonlinear = self._nonlinear_frames > 5
            if is_nonlinear and far_power > 1e-4:
                nonlinear_boost = 1.0 + 1.0 * saturation_level
                residual_echo_psd = residual_echo_psd * nonlinear_boost

            # Harmonic distortion: HF floor from LF echo
            if saturation_level > 0.05 and far_power > 1e-4:
                lf_start, lf_end = self._harm_lf_start, self._harm_lf_end
                hf_start, hf_end = self._harm_hf_start, self._harm_hf_end
                if lf_end > lf_start and hf_end > hf_start:
                    lf_echo_mean = float(np.mean(residual_echo_psd[lf_start:lf_end]))
                    distortion_factor = 0.1 + 0.4 * saturation_level
                    harmonic_floor = lf_echo_mean * distortion_factor
                    residual_echo_psd[hf_start:hf_end] = np.maximum(
                        residual_echo_psd[hf_start:hf_end], harmonic_floor)

            # === DEEP-TRACE HOOKS: capture residual_echo_psd at each cap stage ===
            if self._stats is not None:
                self._stats_last_res_after_attribute = float(np.mean(residual_echo_psd))

            # S8 audit: FS bin total opportunity count (per-frame, increment once).
            # Read-only; FS-bin classification (coh² < 0.1) consistent with S7.
            if self._audit_counters is not None:
                _ac_s8 = self._audit_counters
                _fs_mask_s8 = coh2 < 0.1
                _ac_s8['s8_stage1_fs_bin_total'] += int(np.sum(_fs_mask_s8))

            # Cap 1: echo_psd × 2.0 (skipped in render-mode)
            if not self._residual_est.using_render_based:
                if self._audit_counters is not None:
                    _cap1_arr = self.echo_psd * 2.0
                    _bind = _fs_mask_s8 & (residual_echo_psd > _cap1_arr)
                    _n_bind = int(np.sum(_bind))
                    if _n_bind > 0:
                        _ac_s8['s8_cap1_echo_x2_binding'] += _n_bind
                        _ac_s8['s8_cap1_echo_x2_reduction_sum'] += 10.0 * float(
                            np.sum(np.log10((residual_echo_psd[_bind] + 1e-30)
                                            / (_cap1_arr[_bind] + 1e-30))))
                residual_echo_psd = np.minimum(residual_echo_psd, self.echo_psd * 2.0)
            if self._stats is not None:
                self._stats_last_res_after_echo_cap = float(np.mean(residual_echo_psd))

            # Cap 2: error_psd × (1.5 if render else 1.0)
            err_cap_mult = 1.5 if self._residual_est.using_render_based else 1.0
            if self._audit_counters is not None:
                _cap2_arr = self.error_psd * err_cap_mult
                _bind = _fs_mask_s8 & (residual_echo_psd > _cap2_arr)
                _n_bind = int(np.sum(_bind))
                if _n_bind > 0:
                    _ac_s8['s8_cap2_err_mult_binding'] += _n_bind
                    _ac_s8['s8_cap2_err_mult_reduction_sum'] += 10.0 * float(
                        np.sum(np.log10((residual_echo_psd[_bind] + 1e-30)
                                        / (_cap2_arr[_bind] + 1e-30))))
            if self._cap2_fs_loosen:
                # S11: skip Cap2 in FS-confident bins (coh² < 0.1).
                # Raises residual_echo_psd numerator in ENR → drops gain
                # → expected more FS suppression. Inverse mechanism vs
                # S6-S10 (which lowered ENR denominator).
                _cap2_val = self.error_psd * err_cap_mult
                _capped = np.minimum(residual_echo_psd, _cap2_val)
                residual_echo_psd = np.where(coh2 < 0.1, residual_echo_psd, _capped)
            else:
                residual_echo_psd = np.minimum(residual_echo_psd, self.error_psd * err_cap_mult)
            if self._stats is not None:
                self._stats_last_res_after_error_cap = float(np.mean(residual_echo_psd))

            # Cap 3: dt_suppress (skipped in render-mode)
            if not self._residual_est.using_render_based:
                dt_suppress = np.clip(1.0 - dt_for_fs**2, 0.1, 1.0)
                if self._audit_counters is not None:
                    _cap3_arr = self.error_psd * dt_suppress
                    _bind = _fs_mask_s8 & (residual_echo_psd > _cap3_arr)
                    _n_bind = int(np.sum(_bind))
                    if _n_bind > 0:
                        _ac_s8['s8_cap3_dt_suppress_binding'] += _n_bind
                        _ac_s8['s8_cap3_dt_suppress_reduction_sum'] += 10.0 * float(
                            np.sum(np.log10((residual_echo_psd[_bind] + 1e-30)
                                            / (_cap3_arr[_bind] + 1e-30))))
                residual_echo_psd = np.minimum(residual_echo_psd, self.error_psd * dt_suppress)
            if self._stats is not None:
                self._stats_last_res_after_dt_cap = float(np.mean(residual_echo_psd))

            # Cap 4: render_ceil (skipped in render-mode)
            # v3.14 Arc-P P.S2: when erl_estimate is a per-bin array
            # (flag=ON path), use its mean as a scalar for the Cap 4 ceiling
            # factor. The render_ceil is a conservative upper bound on the
            # residual echo PSD; using mean(per_band) is reasonable since this
            # cap is broadband. The per-bin ERL is consumed by the F3.1-v3
            # excess formula in _stage_gain_compute (the primary P.S2 target).
            _erl_scalar = (float(np.mean(erl_estimate))
                           if isinstance(erl_estimate, np.ndarray)
                           else float(erl_estimate))
            if far_spec is not None and far_power > 1e-4 and _erl_scalar > 0.0:
                far_psd_k = np.abs(far_spec) ** 2
                render_ceil = far_psd_k * min(_erl_scalar * 2.0, 1.0)
                if self._stats is not None:
                    self._stats_last_render_ceil_mean = float(np.mean(render_ceil))
                    self._stats_last_erl_estimate = _erl_scalar
                if not self._residual_est.using_render_based:
                    if self._audit_counters is not None:
                        _bind = _fs_mask_s8 & (residual_echo_psd > render_ceil)
                        _n_bind = int(np.sum(_bind))
                        if _n_bind > 0:
                            _ac_s8['s8_cap4_render_ceil_binding'] += _n_bind
                            _ac_s8['s8_cap4_render_ceil_reduction_sum'] += 10.0 * float(
                                np.sum(np.log10((residual_echo_psd[_bind] + 1e-30)
                                                / (render_ceil[_bind] + 1e-30))))
                    residual_echo_psd = np.minimum(residual_echo_psd, render_ceil)
            if self._stats is not None:
                self._stats_last_res_after_render_ceil = float(np.mean(residual_echo_psd))

            # Reverb tail (WebRTC-AEC3-style IIR on far_psd)
            if self.enable_reverb:
                far_psd = np.abs(far_spec) ** 2 if far_spec is not None else echo_pwr_linear
                self.reverb_psd = (self.reverb_decay * self.reverb_psd
                                   + (1 - self.reverb_decay) * far_psd)
                if not is_stationary_dt:
                    ne_reverb_factor = 0.7 + 0.3 * self.far_activity * (1.0 - dt_for_fs)
                    reverb_gate = self.far_activity * ne_reverb_factor
                    if self.far_activity < 0.1:
                        reverb_gate = 0.0
                    residual_echo_psd = (residual_echo_psd
                                         + self.reverb_gain * self.reverb_psd * reverb_gate)

        # Per-bin echo boost: high-coh2 bins → boost residual estimate
        if far_power > 1e-4 and erle_factor > 0.3 and residual_echo_psd is not None:
            echo_boost = 1.0 + 0.5 * coh2 if dt_for_fs < 0.2 else np.ones_like(coh2)
            residual_echo_psd = residual_echo_psd * echo_boost
        return residual_echo_psd

    def _stage_gain_compute(self, *, residual_echo_psd, eer, coh2, effective_dt,
                              is_stationary_dt, far_power, filter_once_converged,
                              spectral_g_min, eps,
                              erl_estimate: float = 0.01,
                              filter_converged: bool = False,
                              epc_active: bool = False,
                              filter_state: str = 'idle'):
        """Stage 2: ENR / Wiener / spectral_sub gain compute + EMR + spectral floor lift.

        Returns g (post-spectral-floor). Mutates dominant_ne / Round 4 / Round 5
        diag caches and stats.

        `erl_estimate`, `filter_converged`, and `epc_active` are consumed
        only by the F3.1 mic-excess-evidence branch (default-OFF flag).
        Legacy and P4B paths are byte-identical to the pre-F3.1
        implementation.

        `filter_state` is the v3.12 Phase 2 wiring hook (C2). It receives
        the previous frame's enum state from FilterConvergenceAnalyzer
        (idle / startup / diverged / suspicious_dt / refined_usable /
        coarse_learning). Phase 2 plumbs the parameter only; Phase 3
        consumes it (gated by self._consume_filter_state) to drive the
        per-state ENR tuple and gain_floor unification.
        """
        if self.gain_type == "enr" and residual_echo_psd is not None:
            raw_nearend_est = np.maximum(self.error_psd - residual_echo_psd, 0.0)
            if self._noise_floor_refined:
                # S10: per-bin `error_psd × 0.005` in FS-confident bins
                # (coh² < 0.1); baseline scalar `mean(error_psd) × 0.01`
                # in DT/NE. Lowers nearend_est in FS only; DT/NE bins
                # unchanged. Audit-validated: 0% intrusion in S9-A.2.
                noise_floor_psd = np.where(
                    coh2 < 0.1,
                    self.error_psd * 0.005,
                    np.mean(self.error_psd) * 0.01,
                ).astype(self.error_psd.dtype) + 1e-10
            else:
                noise_floor_psd = np.mean(self.error_psd) * 0.01 + 1e-10

            # Per-bin DT indicator: base from coh2 (works for speech far-end).
            # P4B: γ²(k)-primary form removes the frame-scalar floor in the
            # ambiguous DTD regime (effective_dt 0.2–0.5) so per-bin γ²
            # discrimination survives instead of being clamped uniformly to
            # the frame value. effective_dt only contributes a soft floor
            # when it crosses 0.5 (DTD strongly evidences NE).
            # F3.1: per-bin mic-energy excess metric. Reuses the P1-Phase-1
            # validated `max(error_psd − far_lw·ERL_est, 0) / error_psd`
            # ratio (AUROC 0.871 in HF-cap gate) as the *primary* per-bin
            # NE evidence — replaces `(1 - coh2)` which saturates to 1 in
            # FS post-cancellation. Gated on `filter_converged` AND
            # long-window initialised so `erl_estimate` and `far_lw` are
            # reliable; falls through to the P4B / legacy path otherwise.
            _lw_ready = (
                self._residual_est is not None
                and self._residual_est._long_window_n_updates > 0
            )
            # F3.1 v3 (2026-05-12): blend with legacy `(1 - coh2)`
            # (weight=0.7 F3.1, 0.3 legacy) to soften HF over-
            # suppression in high-coupling rooms where erl_estimate
            # underestimates true ERL (Lsa5Wpw / wr54weK pattern:
            # mic/far 0.83/0.55, true ERL ~0.68/0.30 but erl_estimate
            # capped to 0.3 after EPC). Pure F3.1 over-attributes NE
            # → over-suppresses → spectral imbalance hurts AECMOS
            # even though total echo drops. Blend leaves F3.1 as the
            # dominant signal but caps its swing.
            #
            # The earlier v3 attempt also tried a `mic_pwr <= 2·far·erl`
            # envelope gate to block Regime-2 (pG9Bikvr non-echo
            # content). It correctly blocked the FS-noise case but
            # also blocked legitimate DT cases where mic naturally
            # exceeds expected echo (i2BU43nm). Adding `OR effective_dt
            # >= 0.2` softened the FS protection back out. Conclusion:
            # binary gating on mic/far ratio can't be made FS-only
            # without a label we don't have; the blend alone is the
            # honest cap.
            # v3.12 S7 (Phase 3B v3 Option α): when `_dt_per_bin_unified` is
            # True, drop the `NOT epc_active` constraint so the F3.1 v3
            # mic-excess blend also fires during EPC-active frames. Legacy
            # fallback at `else:` below remains the path for `not
            # filter_converged` or `not _lw_ready` regardless of flag.
            if (self._use_mic_excess_evidence
                    and filter_converged
                    and _lw_ready
                    and (self._dt_per_bin_unified or not epc_active)):
                far_lw = self._residual_est._long_window_far_psd
                # v3.14 Arc-P P.S2: when erl_estimate is a per-bin ndarray
                # (passed by AEC when f3_1_per_band_erl_adaptive=True), use it
                # directly (numpy broadcasting handles scalar and array equally).
                # When erl_estimate is scalar float (default-OFF path), float()
                # preserves existing behaviour — byte-equal guaranteed.
                if isinstance(erl_estimate, np.ndarray):
                    erl_e = erl_estimate  # per-bin array, shape (n_freqs,)
                else:
                    erl_e = float(erl_estimate)
                excess = np.maximum(self.error_psd - far_lw * erl_e, 0.0)
                excess_ratio = np.clip(
                    excess / (self.error_psd + 1e-10), 0.0, 1.0,
                ).astype(np.float32)
                # Blend with legacy `(1 - coh2)` to soften the over-attribution
                # under erl_estimate underestimation (Regime-3 mitigation).
                legacy = 1.0 - coh2
                dt_per_bin = (_BLEND_F31_MIC_EXCESS * excess_ratio
                              + (1.0 - _BLEND_F31_MIC_EXCESS) * legacy)
                if effective_dt > 0.5:
                    floor_lift = float((effective_dt - 0.5) * 2.0)
                    dt_per_bin = np.maximum(dt_per_bin, floor_lift)
            elif self._plan_b_dt_per_bin_gamma:
                dt_per_bin = (1.0 - coh2).astype(np.float32)
                if effective_dt > 0.5:
                    floor_lift = float((effective_dt - 0.5) * 2.0)
                    dt_per_bin = np.maximum(dt_per_bin, floor_lift)
            else:
                dt_per_bin = np.maximum(
                    np.full(self.n_freqs, effective_dt, dtype=np.float32),
                    1.0 - coh2
                )
            # S7 (Phase 3B v3 Option α) fire-rate audit hook. Zero-cost when
            # `_audit_counters` is None. Read-only; does not mutate dt_per_bin
            # or any other state. Computes unified-hypothetical dt_per_bin
            # alongside legacy and aggregates FS-bin (coh²<0.1) sums for
            # post-stream reduction-percentage analysis.
            if self._audit_counters is not None:
                _ac = self._audit_counters
                _ac['total_frames'] += 1
                _f31_active = (self._use_mic_excess_evidence
                               and filter_converged
                               and _lw_ready
                               and not epc_active)
                if _f31_active:
                    _ac['s7_f31v3_path_frames'] += 1
                elif self._plan_b_dt_per_bin_gamma:
                    _ac['s7_planb_path_frames'] += 1
                else:
                    _ac['s7_legacy_path_frames'] += 1
                    if epc_active:
                        _ac['s7_legacy_epc_active_frames'] += 1
                    if not filter_converged:
                        _ac['s7_legacy_not_converged_frames'] += 1
                    if not _lw_ready:
                        _ac['s7_legacy_not_lw_ready_frames'] += 1
                    _target_slice = filter_converged and _lw_ready and epc_active
                    _alt_slice = (filter_converged and _lw_ready
                                  and not epc_active)
                    if _target_slice or _alt_slice:
                        _far_lw_au = self._residual_est._long_window_far_psd
                        _erl_e_au = float(erl_estimate)
                        _excess_au = np.maximum(
                            self.error_psd - _far_lw_au * _erl_e_au, 0.0,
                        )
                        _excess_ratio_au = np.clip(
                            _excess_au / (self.error_psd + 1e-10), 0.0, 1.0,
                        ).astype(np.float32)
                        _legacy_au = (1.0 - coh2).astype(np.float32)
                        _unified_au = (
                            _BLEND_F31_MIC_EXCESS * _excess_ratio_au
                            + (1.0 - _BLEND_F31_MIC_EXCESS) * _legacy_au
                        )
                        if effective_dt > 0.5:
                            _floor_lift_au = float((effective_dt - 0.5) * 2.0)
                            _unified_au = np.maximum(_unified_au, _floor_lift_au)
                        _fs_mask_au = coh2 < 0.1
                        _n_fs_au = int(np.sum(_fs_mask_au))
                        if _n_fs_au > 0:
                            _legacy_sum_au = float(np.sum(dt_per_bin[_fs_mask_au]))
                            _unified_sum_au = float(np.sum(_unified_au[_fs_mask_au]))
                            if _target_slice:
                                _ac['s7_target_fs_bin_count'] += _n_fs_au
                                _ac['s7_target_fs_bin_legacy_sum'] += _legacy_sum_au
                                _ac['s7_target_fs_bin_unified_sum'] += _unified_sum_au
                            else:
                                _ac['s7_alt_fs_bin_count'] += _n_fs_au
                                _ac['s7_alt_fs_bin_legacy_sum'] += _legacy_sum_au
                                _ac['s7_alt_fs_bin_unified_sum'] += _unified_sum_au
                    else:
                        _ac['s7_legacy_target_other_frames'] += 1
            if is_stationary_dt:
                dt_per_bin = np.maximum(dt_per_bin, self._stat_dt_mask)
            # v3.8.4: stash for postprocess HF-cap "high bins NE confidence" gate
            self._dt_per_bin_last = dt_per_bin

            # P4B diag (zero-cost; means over small slices). Captures the
            # symptom this plan diagnoses: dt_per_bin and 1-coh2 both
            # saturate ~1.0 in DT and FS post-cancel.
            _hf_2k = self._hf_cap_bin_2k
            self._p4b_dt_per_bin_mean = float(np.mean(dt_per_bin))
            self._p4b_dt_per_bin_hf_mean = (
                float(np.mean(dt_per_bin[_hf_2k:]))
                if dt_per_bin.shape[0] > _hf_2k else 0.0
            )
            self._p4b_coh2_hf_mean = (
                float(np.mean(coh2[_hf_2k:])) if coh2.shape[0] > _hf_2k else 0.0
            )
            self._p4b_effective_dt = float(effective_dt)
            self._p4b_is_stationary_dt = int(is_stationary_dt)

            dt_shaped_per_bin = dt_per_bin ** 1.1
            nearend_est = np.maximum(raw_nearend_est * dt_shaped_per_bin, noise_floor_psd)

            min_ne_from_dt = self.error_psd * dt_shaped_per_bin
            _startup_dt_cond = (effective_dt > 0.35 and far_power > 1e-4
                                and not filter_once_converged)
            if _startup_dt_cond and self.startup_dt_min_ne_scale != 1.0:
                min_ne_from_dt = min_ne_from_dt * self.startup_dt_min_ne_scale
            if self._residual_est.using_render_based:
                min_ne_from_dt = min_ne_from_dt * getattr(self, 'render_min_ne_factor', 0.5)
            nearend_est = np.maximum(nearend_est, min_ne_from_dt)
            if self._stats is not None:
                self._stats_last_min_ne = float(np.mean(min_ne_from_dt))

            ne_physical_floor = self.error_psd * 0.05
            nearend_est = np.maximum(nearend_est, ne_physical_floor)

            # S8 audit: nearend_est 4-way binding identification on FS bins.
            # Of (raw * dt_shaped, noise_floor_psd, min_ne_from_dt,
            # ne_physical_floor), record which value is the MAX (=binding).
            if self._audit_counters is not None:
                _fs_mask_nef = coh2 < 0.1
                if np.any(_fs_mask_nef):
                    _v1 = raw_nearend_est * dt_shaped_per_bin
                    _v2 = np.full_like(_v1, noise_floor_psd)
                    _v3 = min_ne_from_dt
                    _v4 = ne_physical_floor
                    _stack = np.stack([_v1, _v2, _v3, _v4], axis=0)
                    _winner = np.argmax(_stack, axis=0)
                    _ac_nef = self._audit_counters
                    _ac_nef['s8_nef_raw_count'] += int(
                        np.sum((_winner == 0) & _fs_mask_nef))
                    _ac_nef['s8_nef_noise_floor_count'] += int(
                        np.sum((_winner == 1) & _fs_mask_nef))
                    _ac_nef['s8_nef_min_ne_count'] += int(
                        np.sum((_winner == 2) & _fs_mask_nef))
                    _ac_nef['s8_nef_ne_physical_count'] += int(
                        np.sum((_winner == 3) & _fs_mask_nef))

                    # S9 pre-audit: hypothetical noise_floor_psd
                    # refinements. Reuses _v1/_v3/_v4 and _winner from S8;
                    # only swaps v2 for each candidate and re-argmaxes.
                    _v2_a1 = np.full_like(_v1, float(np.mean(self.error_psd)) * 0.001 + 1e-10)
                    _v2_a2 = self.error_psd.astype(_v1.dtype) * 0.005 + 1e-10
                    _stack_a1 = np.stack([_v1, _v2_a1, _v3, _v4], axis=0)
                    _stack_a2 = np.stack([_v1, _v2_a2, _v3, _v4], axis=0)
                    _winner_a1 = np.argmax(_stack_a1, axis=0)
                    _winner_a2 = np.argmax(_stack_a2, axis=0)
                    _baseline_floor = (_winner == 1) & _fs_mask_nef
                    _baseline_max = _stack.max(axis=0)
                    for _label, _wA in (('a1', _winner_a1), ('a2', _winner_a2)):
                        _ac_nef[f's9_{_label}_release_to_raw'] += int(
                            np.sum(_baseline_floor & (_wA == 0)))
                        _ac_nef[f's9_{_label}_stays_floor'] += int(
                            np.sum(_baseline_floor & (_wA == 1)))
                        _ac_nef[f's9_{_label}_shift_to_min_ne'] += int(
                            np.sum(_baseline_floor & (_wA == 2)))
                        _ac_nef[f's9_{_label}_shift_to_phys'] += int(
                            np.sum(_baseline_floor & (_wA == 3)))
                        # Bins NOT bound by floor under baseline but whose
                        # winner CHANGES under the candidate (A.2 can do
                        # this when error_psd[i] is high enough that
                        # error_psd[i]*0.005 surpasses the current winner).
                        _not_floor = _fs_mask_nef & (_winner != 1)
                        _ac_nef[f's9_{_label}_intrudes_outside_floor_baseline'] += int(
                            np.sum(_not_floor & (_wA != _winner)))
                        # Magnitude: 10*log10(baseline_max / candidate_max)
                        # over bins where any change happened (only positive
                        # values count — candidate strictly lowers nearend_est).
                        _stack_X = _stack_a1 if _label == 'a1' else _stack_a2
                        _cand_max = _stack_X.max(axis=0)
                        _changed = _fs_mask_nef & (_cand_max < _baseline_max)
                        _n_changed = int(np.sum(_changed))
                        if _n_changed > 0:
                            _ratio = (_baseline_max[_changed] + 1e-30) / (
                                _cand_max[_changed] + 1e-30)
                            _ac_nef[f's9_{_label}_reduction_db_sum'] += 10.0 * float(
                                np.sum(np.log10(_ratio)))
                            _ac_nef[f's9_{_label}_reduction_count'] += _n_changed

                    # S9-C joint floor attack. Uses A.2 for noise_floor
                    # (per-bin error_psd * 0.005) and additionally
                    # scales min_ne_from_dt down in FS bins. Tracks fate
                    # of ALL baseline-floor-bound FS bins (not just
                    # noise_floor winners).
                    _baseline_any_floor = _fs_mask_nef & (_winner != 0)
                    _n_any_floor = int(np.sum(_baseline_any_floor))
                    _ac_nef['s9c_baseline_any_floor_count'] += _n_any_floor
                    if _n_any_floor > 0:
                        _v3_c1 = _v3 * 0.1                # min_ne × 0.1
                        _v3_c2 = np.zeros_like(_v3)        # min_ne → 0
                        _stack_c1 = np.stack(
                            [_v1, _v2_a2, _v3_c1, _v4], axis=0)
                        _stack_c2 = np.stack(
                            [_v1, _v2_a2, _v3_c2, _v4], axis=0)
                        _winner_c1 = np.argmax(_stack_c1, axis=0)
                        _winner_c2 = np.argmax(_stack_c2, axis=0)
                        for _lab, _wC, _stk in (
                                ('c1', _winner_c1, _stack_c1),
                                ('c2', _winner_c2, _stack_c2)):
                            _ac_nef[f's9c_{_lab}_release_to_raw'] += int(
                                np.sum(_baseline_any_floor & (_wC == 0)))
                            _ac_nef[f's9c_{_lab}_still_floor'] += int(
                                np.sum(_baseline_any_floor & (_wC == 1)))
                            _ac_nef[f's9c_{_lab}_still_min_ne'] += int(
                                np.sum(_baseline_any_floor & (_wC == 2)))
                            _ac_nef[f's9c_{_lab}_still_phys'] += int(
                                np.sum(_baseline_any_floor & (_wC == 3)))
                            _cand_max_c = _stk.max(axis=0)
                            _changed_c = _fs_mask_nef & (
                                _cand_max_c < _baseline_max)
                            _n_c = int(np.sum(_changed_c))
                            if _n_c > 0:
                                _r_c = (_baseline_max[_changed_c] + 1e-30) / (
                                    _cand_max_c[_changed_c] + 1e-30)
                                _ac_nef[
                                    f's9c_{_lab}_reduction_db_sum'
                                ] += 10.0 * float(np.sum(np.log10(_r_c)))
                                _ac_nef[
                                    f's9c_{_lab}_reduction_count'
                                ] += _n_c

                        # S9-D sanity: attack all three floors. Stack
                        # = [raw*dt_shaped, error_psd*0.005, 0, 0].
                        _stack_d = np.stack(
                            [_v1, _v2_a2, np.zeros_like(_v3),
                             np.zeros_like(_v4)], axis=0)
                        _winner_d = np.argmax(_stack_d, axis=0)
                        _ac_nef['s9d_release_to_raw'] += int(
                            np.sum(_baseline_any_floor & (_winner_d == 0)))
                        _ac_nef['s9d_still_floor'] += int(
                            np.sum(_baseline_any_floor & (_winner_d == 1)))
                        _cand_max_d = _stack_d.max(axis=0)
                        _changed_d = _fs_mask_nef & (_cand_max_d < _baseline_max)
                        _n_d = int(np.sum(_changed_d))
                        if _n_d > 0:
                            _r_d = (_baseline_max[_changed_d] + 1e-30) / (
                                _cand_max_d[_changed_d] + 1e-30)
                            _ac_nef['s9d_reduction_db_sum'] += 10.0 * float(
                                np.sum(np.log10(_r_d)))
                            _ac_nef['s9d_reduction_count'] += _n_d

            # Round 4 trace cache (audio-passive)
            self._diag_nearend_est_last = nearend_est
            self._diag_residual_echo_psd_last = residual_echo_psd

            enr = residual_echo_psd / (nearend_est + 1e-10)

            blend = self._enr_blend
            scale = self.enr_scale
            ne_confidence = dt_per_bin
            effective_scale = scale
            # v3.14 Arc-R Sprint S1: per-band ENR threshold (default OFF).
            # When `_per_band_enr=True`, substitute the precomputed per-bin
            # arrays built from `enr_t_ne_per_band`/`enr_s_ne_per_band` tuples.
            # Default OFF keeps the legacy `_enr_blend` formula → byte-equal.
            if self._per_band_enr:
                enr_t_ne = self._enr_t_ne_pb
                enr_s_ne = self._enr_s_ne_pb
            else:
                enr_t_ne = (1 - blend) * 2.0 + blend * 1.5
                enr_s_ne = (1 - blend) * 3.0 + blend * 2.5
            enr_t_fs = (1 - blend) * (0.3 * effective_scale) + blend * (0.07 * effective_scale)
            enr_s_fs = (1 - blend) * (0.4 * effective_scale) + blend * (0.1 * effective_scale)
            if effective_dt > 0.4:
                dt_enr_relax = 1.0 + (effective_dt - 0.4) / 0.6 * 0.5
                enr_t_ne = enr_t_ne * dt_enr_relax
                enr_s_ne = enr_s_ne * dt_enr_relax
            # v3.15 §1.2 — DT-NE compression fix (default OFF, byte-equal).
            # Apply per-state scale then per-bin dt_per_bin override to
            # enr_t_ne / enr_s_ne.  refined_usable state-scale must be 1.0
            # to preserve byte-equal with v3.13 steady BALANCED path.
            if self._dt_ne_compression_fix:
                _state_scale = self._dt_ne_state_scale.get(
                    filter_state, 1.0)
                if _state_scale != 1.0:
                    enr_t_ne = enr_t_ne * _state_scale
                    enr_s_ne = enr_s_ne * _state_scale
                _bin_scale = self._dt_ne_per_bin_scale
                if _bin_scale != 1.0:
                    _mask = (dt_per_bin > self._dt_ne_per_bin_thresh)
                    if np.any(_mask):
                        # Broadcast scalar enr_t_ne / enr_s_ne to per-bin if needed.
                        if np.ndim(enr_t_ne) == 0:
                            enr_t_ne = np.full_like(dt_per_bin, float(enr_t_ne))
                        else:
                            enr_t_ne = np.asarray(enr_t_ne, dtype=np.float32).copy()
                        if np.ndim(enr_s_ne) == 0:
                            enr_s_ne = np.full_like(dt_per_bin, float(enr_s_ne))
                        else:
                            enr_s_ne = np.asarray(enr_s_ne, dtype=np.float32).copy()
                        enr_t_ne[_mask] = enr_t_ne[_mask] * _bin_scale
                        enr_s_ne[_mask] = enr_s_ne[_mask] * _bin_scale
            enr_t = ne_confidence * enr_t_ne + (1 - ne_confidence) * enr_t_fs
            enr_s = ne_confidence * enr_s_ne + (1 - ne_confidence) * enr_s_fs
            min_gate_width = 0.2
            enr_s_safe = np.maximum(enr_s, enr_t + min_gate_width)

            g = np.where(enr > enr_t,
                         np.clip((enr_s_safe - enr) / (enr_s_safe - enr_t + eps), 0.0, 1.0),
                         1.0)

            # EMR: AEC3-style noise masking
            if np.sum(self.noise_psd) > 0:
                emr = residual_echo_psd / (self.noise_psd + 1e-10)
                emr_transparent = 0.3
                g_emr = np.clip(emr_transparent / (emr + 1e-10), 0.0, 1.0)
                g = np.maximum(g, g_emr)

            if self._stats is not None:
                self._stats_last_gain_before_floor = float(np.mean(g))
            if getattr(self, '_capture_stages', False):
                self._stage_gains = {'01_softgate_emr': g.copy()}
            self._diag_round5_stages[0] = float(np.mean(g[self._voice_band_idx])) if self._voice_band_idx.size > 0 else 0.0
            g = np.maximum(g, spectral_g_min)
            if self._stats is not None:
                self._stats_last_gain_after_floor = float(np.mean(g))
            if getattr(self, '_capture_stages', False):
                self._stage_gains['02_spectral_floor'] = g.copy()
            self._diag_round5_stages[1] = float(np.mean(g[self._voice_band_idx])) if self._voice_band_idx.size > 0 else 0.0

            if self._stats is not None:
                self._stats_last_enr = float(np.mean(enr))
                self._stats_last_nearend = float(np.mean(nearend_est))
                self._stats_last_res_psd = float(np.mean(residual_echo_psd))

            # Phase 0 trace: dominant_nearend_like raw signal (frame-level)
            _ne_mean = float(np.mean(nearend_est))
            _res_mean = float(np.mean(residual_echo_psd))
            _noise_mean = float(np.mean(self.noise_psd)) + 1e-10
            self._diag_dominant_nearend_raw = bool(
                _ne_mean > 3.0 * _res_mean
                and _ne_mean > 5.0 * _noise_mean
            )

        elif self.gain_type == "wiener" and residual_echo_psd is not None:
            noise_floor_psd = np.mean(self.error_psd) * 0.01 + eps
            nearend_est = np.maximum(self.error_psd - residual_echo_psd, noise_floor_psd)
            beta = self.over_sub
            g = nearend_est / (nearend_est + beta * residual_echo_psd + eps)
            g = np.maximum(g, spectral_g_min)
        elif residual_echo_psd is not None:
            eer_direct = residual_echo_psd / (self.error_psd + eps)
            g = np.maximum(1.0 - self.over_sub * eer_direct, spectral_g_min)
        else:
            g = np.maximum(1.0 - self.over_sub * eer, spectral_g_min)
        return g

    def _stage_gain_postprocess(self, *, g_in, epc_dt, quiet_mask, far_power,
                                  effective_dt, is_stationary_dt, divergence,
                                  erl_estimate=0.01):
        """Stage 3: EPC_DT cap, quiet mask, 3-bin smooth, HF cap, divergence override.

        Returns updated g (post-divergence-override). Mutates _diag_round5_stages[2..6].
        """
        g = g_in
        # EPC_DT gain cap: echo path changed + DT → cap gain to force minimum echo suppression.
        if epc_dt:
            EPC_DT_GAIN_CAP = 0.85
            g = np.minimum(g, EPC_DT_GAIN_CAP)
        if getattr(self, '_capture_stages', False):
            self._stage_gains['03_epc_dt_cap'] = g.copy()
        self._diag_round5_stages[2] = float(np.mean(g[self._voice_band_idx])) if self._voice_band_idx.size > 0 else 0.0

        g[quiet_mask] = 1.0  # Noise gate: pass through quiet bins
        if getattr(self, '_capture_stages', False):
            self._stage_gains['04_quiet_mask'] = g.copy()
        self._diag_round5_stages[3] = float(np.mean(g[self._voice_band_idx])) if self._voice_band_idx.size > 0 else 0.0

        # --- Frequency-domain postprocessing (cf. AEC3 PostprocessGains) ---
        if far_power > 1e-4:
            # 3-bin cross-frequency smoothing.
            # v3.8.4: kernel tightened from [0.25, 0.5, 0.25] to
            # [0.1, 0.8, 0.1]. The original kernel let low-band echo gain
            # leak into high bins via 25% sidelobes — measured 10 dB extra
            # cut in the 4–8 kHz band on case 7GTxyT (DT, BALANCED). The
            # tighter kernel still suppresses single-bin spurious spikes
            # (its original purpose) while preserving cross-band gain
            # independence, which keeps fricative / formant content audible.
            # P1.0 toggle: plan_a_kernel_tight=False reverts to v3.8.3 kernel
            if self._plan_a_kernel_tight:
                kernel = np.array([0.1, 0.8, 0.1], dtype=np.float32)
            else:
                kernel = np.array([0.25, 0.5, 0.25], dtype=np.float32)
            g = np.convolve(g, kernel, mode='same').astype(np.float32)
            if getattr(self, '_capture_stages', False):
                self._stage_gains['05_3bin_smooth'] = g.copy()
            self._diag_round5_stages[4] = float(np.mean(g[self._voice_band_idx])) if self._voice_band_idx.size > 0 else 0.0
            # DC consistency: bins 0-1 follow bin 2
            if self.n_freqs > 2:
                g[:2] = np.minimum(g[1], g[2])
            # v3.8.4: HF cap reworked. Two changes:
            #   (a) only fire when DTD is confident NE absent (was < 0.5,
            #       too permissive — fired throughout DT). Now < 0.3.
            #   (b) anchor the cap at 2 kHz instead of 500 Hz so vowel
            #       formants (1–3 kHz) are not dragged down by the 500 Hz
            #       bin's gain.
            #   (c) skip the cap entirely when high bins themselves show NE
            #       energy (per-bin DT confidence > 0.3 in 2 kHz+).
            # NOTE (v3.10.5 investigation): dt_per_bin = max(effective_dt,
            # 1-coh2) saturates ~1 in FS post-cancellation (echo cancelled →
            # low coh2 → "NE-like"), so the high_ne_conf < 0.3 gate rarely
            # fires in FS — cap is largely dead code there. Plan A's actual
            # FS cost lives in the smoothing kernel change, not here. Left
            # as-is pending a redesigned evidence metric that distinguishes
            # DT-NE from FS-decoupling.
            # P1 Phase 2 — conditional HF cap based on m_excess_ratio.
            # m_excess_ratio = mean(max(error_psd[2k:] - far_lw[2k:]·erl_est, 0))
            #                 / (mean(error_psd[2k:]) + eps)
            # Validated in P1 Phase 1 (AUROC 0.871 FS-vs-NE+DT_positive,
            # α=1.0 stable across {0.5, 1.0, 2.0}). Replaces the broken
            # `1 - coh2`-based high_ne_conf gate.
            if self._hf_cap_conditional:
                try:
                    far_lw = self._residual_est._long_window_far_psd
                    err_hb = self.error_psd[self._hf_cap_bin_2k:]
                    far_hb = far_lw[self._hf_cap_bin_2k:]
                    # v3.14 Arc-P P.S2: HF cap uses scalar ERL (broadband).
                    erl_e = (float(np.mean(erl_estimate[self._hf_cap_bin_2k:]))
                             if isinstance(erl_estimate, np.ndarray)
                             else float(erl_estimate))
                    excess = np.maximum(err_hb - 1.0 * far_hb * erl_e, 0.0)
                    err_hb_mean = float(np.mean(err_hb)) + 1e-10
                    metric = float(np.mean(excess)) / err_hb_mean
                except Exception:
                    metric = 0.0  # fail open: skip cap when metric unavailable
                if metric < self._hf_cap_metric_threshold:
                    # FS-like: apply v3.8.3 strict cap (anchor 500 Hz, gate < 0.5)
                    hf_cap_bin = self._hf_cap_bin
                    if (self.n_freqs > hf_cap_bin + 1
                            and effective_dt < 0.5
                            and not is_stationary_dt):
                        hf_cap = g[hf_cap_bin]
                        g[hf_cap_bin + 1:] = np.minimum(g[hf_cap_bin + 1:], hf_cap)
                # else: DT-like → skip cap entirely (preserve high-band NE)
            elif self._plan_a_hf_cap_2k:
                # Default v3.10.4 behaviour: Plan A cap anchor 2 kHz, gate < 0.3,
                # skip on (broken) high_ne_conf < 0.3
                hf_cap_bin = self._hf_cap_bin_2k
                if (self.n_freqs > hf_cap_bin + 1
                        and effective_dt < 0.3
                        and not is_stationary_dt):
                    high_ne_conf = float(np.mean(self._dt_per_bin_last[hf_cap_bin:]))
                    if high_ne_conf < 0.3:
                        hf_cap = g[hf_cap_bin]
                        g[hf_cap_bin + 1:] = np.minimum(g[hf_cap_bin + 1:], hf_cap)
            else:
                # P1.0 toggle: plan_a_hf_cap_2k=False reverts to v3.8.3 cap
                hf_cap_bin = self._hf_cap_bin
                if (self.n_freqs > hf_cap_bin + 1
                        and effective_dt < 0.5
                        and not is_stationary_dt):
                    hf_cap = g[hf_cap_bin]
                    g[hf_cap_bin + 1:] = np.minimum(g[hf_cap_bin + 1:], hf_cap)
            if getattr(self, '_capture_stages', False):
                self._stage_gains['06_hf_cap'] = g.copy()
            self._diag_round5_stages[5] = float(np.mean(g[self._voice_band_idx])) if self._voice_band_idx.size > 0 else 0.0

        # Divergence override: when filter diverges, cap gain severely
        if divergence > 0.3:
            divergence_gain = 0.01 + (1.0 - 0.01) * (1.0 - divergence)
            g = np.minimum(g, divergence_gain)
        if getattr(self, '_capture_stages', False):
            self._stage_gains['07_pre_temporal'] = g.copy()
        self._diag_round5_stages[6] = float(np.mean(g[self._voice_band_idx])) if self._voice_band_idx.size > 0 else 0.0

        # P4B diag: HF gain after pre-temporal postprocess + HF residual echo.
        _hf_2k = self._hf_cap_bin_2k
        if g.shape[0] > _hf_2k:
            self._p4b_gain_hf_mean = float(np.mean(g[_hf_2k:]))
        else:
            self._p4b_gain_hf_mean = 0.0
        _re = getattr(self, '_diag_residual_echo_psd_last', None)
        if _re is not None and _re.shape[0] > _hf_2k:
            self._p4b_res_echo_hf_mean_db = (
                10.0 * float(np.log10(float(np.mean(_re[_hf_2k:])) + 1e-12))
            )
        else:
            self._p4b_res_echo_hf_mean_db = -120.0
        return g

    def _stage_temporal_smoothing(self, *, g_in, dt_indicator, effective_dt,
                                   is_stationary_dt, erle_factor, fs_confidence,
                                   spectral_g_min, effective_g_min):
        """Stage 4: split attack/release EMA + rate limiting + LF protect + render ceil.

        Mutates: self.gain_smooth (= smoothed final gain),
                 self._diag_round5_stages[7],
                 self._stats_last_gain_after_smoothing.
        """
        g = g_in
        # Temporal DT: when Stationary DT confirmed, treat as dt=0.8 for smoothing/rate
        dt_temporal = 0.8 if is_stationary_dt else max(dt_indicator, effective_dt * 0.5)

        alpha_fast = 0.3 + 0.2 * (1.0 - erle_factor)   # 0.3-0.5
        alpha_slow = 0.85 + 0.1 * (1.0 - erle_factor)  # 0.85-0.95
        alpha_attack = alpha_slow + (alpha_fast - alpha_slow) * fs_confidence
        if is_stationary_dt:
            alpha_attack = alpha_attack * np.clip(1.0 - dt_temporal**2, 0.1, 1.0)
        alpha_release_light = 0.5 - 0.2 * dt_temporal
        smoothed = np.where(g < self.gain_smooth,
                            alpha_attack * self.gain_smooth + (1 - alpha_attack) * g,
                            alpha_release_light * self.gain_smooth + (1 - alpha_release_light) * g)

        # Rate limiting
        activity_scale = 0.5 + 0.5 * self.far_activity
        eff_drop = self.max_drop_ratio ** activity_scale
        rise_exp = 0.5 + 0.5 * (1.0 - self.far_activity)
        if dt_temporal > 0.3:
            dt_rise_boost = 1.0 + dt_temporal
            rise_exp = rise_exp / dt_rise_boost
        eff_rise = self.max_rise_ratio ** rise_exp
        gain_floor = self.gain_smooth / eff_drop
        gain_ceil = self.gain_smooth * eff_rise
        if fs_confidence < 0.9:
            lf_limit = min(8, self.n_freqs)
            lf_factor = 0.25 * max(1.0 - fs_confidence * 2.0, 0.0)
            if lf_factor > 0.01:
                gain_floor[:lf_limit] = np.maximum(
                    gain_floor[:lf_limit], self.gain_smooth[:lf_limit] * lf_factor)
        smoothed = np.maximum(smoothed, gain_floor)
        smoothed = np.minimum(smoothed, gain_ceil)
        if isinstance(spectral_g_min, np.ndarray):
            smoothed = np.maximum(smoothed, spectral_g_min)
        else:
            smoothed = np.maximum(smoothed, effective_g_min)
        smoothed = np.minimum(smoothed, 1.0)
        # v3.2 Axis 2: hard ceiling in render-mode whenever far is active.
        if self._residual_est.using_render_based and self.far_activity > 0.3:
            ceil = getattr(self, 'render_dt_gain_ceil', 0.6)
            smoothed = np.minimum(smoothed, ceil)
        if getattr(self, '_capture_stages', False):
            self._stage_gains['08_post_temporal'] = smoothed.copy()
        self._diag_round5_stages[7] = float(np.mean(smoothed[self._voice_band_idx])) if self._voice_band_idx.size > 0 else 0.0
        self.gain_smooth = smoothed
        if self._stats is not None:
            self._stats_last_gain_after_smoothing = float(np.mean(self.gain_smooth))

    def _stage_noise_floor_and_cng(self, *, spec_synth, far_power, effective_dt,
                                    dt_indicator, filter_once_converged,
                                    effective_g_min, hop):
        """Stage 5: noise_psd tracker + noise-floor lift + CNG + IFFT/OLA.

        Mutates: self.noise_psd, self.gain_smooth, self._smooth_cn_gain,
                 self.ola_buf, self._diag_round5_stages[8],
                 self._stats_last_* (when stats enabled).
        Returns: output[hop_size] (float32).
        """
        # E6: min-statistics noise tracker (always on, used for dynamic floor + CNG)
        # Tracks quiet-floor of residual over time. In DT with quiet near-end,
        # this floor includes near-end energy — so gain >= noise_floor_gain
        # preserves quiet speech above the learned floor (Speex/AEC3 style).
        if not self._noise_initialized:
            self.noise_psd = self.error_psd.copy() + 1e-8
            self._noise_initialized = True
            self._smooth_cn_gain = np.zeros(self.n_freqs, dtype=np.float32)
        is_learning_safe = (self.far_activity < 0.01) and (dt_indicator < 0.1)
        alpha_down = 0.98
        alpha_up = 0.998 if is_learning_safe else 1.0
        alpha_n = np.where(self.error_psd > self.noise_psd, alpha_up, alpha_down)
        self.noise_psd = alpha_n * self.noise_psd + (1 - alpha_n) * self.error_psd

        # Dynamic floor: per-bin minimum gain so output |g*spec| >= sqrt(noise_psd).
        spec_pwr_synth = np.abs(spec_synth) ** 2 + 1e-10
        noise_floor_gain = np.sqrt(self.noise_psd / spec_pwr_synth)
        noise_floor_gain = np.clip(noise_floor_gain, effective_g_min, 1.0)

        _startup_dt_nfl = effective_dt > 0.35 and far_power > 1e-4 and not filter_once_converged
        if self._stats is not None:
            _nfl_mean = float(np.mean(noise_floor_gain))
            self._stats_last_noise_floor_gain = _nfl_mean
            self._stats_last_noise_psd = float(np.mean(self.noise_psd))
            self._stats_last_spec_pwr = float(np.mean(spec_pwr_synth))
            self._stats_last_nfl_lifted = _nfl_mean > float(np.mean(self.gain_smooth))

        if _startup_dt_nfl and self.startup_dt_noise_floor_scale < 1.0:
            if self.startup_dt_noise_floor_scale > 0.0:
                self.gain_smooth = np.maximum(self.gain_smooth,
                                              noise_floor_gain * self.startup_dt_noise_floor_scale)
            # else scale=0.0: bypass — noise_floor_gain not applied
        else:
            self.gain_smooth = np.maximum(self.gain_smooth, noise_floor_gain)
        # Round 5 stage 8: final gain after noise-floor lift (= what reaches IFFT)
        self._diag_round5_stages[8] = float(np.mean(self.gain_smooth[self._voice_band_idx])) if self._voice_band_idx.size > 0 else 0.0

        # Apply gain + synthesis sqrt-Hann window + IFFT
        enhanced_spec = self.gain_smooth * spec_synth

        # --- CNG: Comfort Noise Generation (fill remaining suppression gap) ---
        if self.enable_cng and np.sum(self.noise_psd) > 1e-7:
            target_cn_gain = np.sqrt(np.maximum(1.0 - self.gain_smooth ** 2, 0.0)) * 0.4
            self._smooth_cn_gain = 0.8 * self._smooth_cn_gain + 0.2 * target_cn_gain
            noise_std = np.sqrt(self.noise_psd / 2.0).astype(np.float32)
            cng_real = np.random.randn(self.n_freqs).astype(np.float32) * noise_std
            cng_imag = np.random.randn(self.n_freqs).astype(np.float32) * noise_std
            cng_spec = (self._smooth_cn_gain * (cng_real + 1j * cng_imag)).astype(np.complex64)
            enhanced_spec = enhanced_spec + cng_spec

        enhanced_time = np.fft.irfft(enhanced_spec, self.block_size)[:self.frame_size]
        enhanced_time *= self.window

        # Overlap-add (frame_size buffer)
        self.ola_buf += enhanced_time
        output = self.ola_buf[:hop].copy()
        self.ola_buf[:-hop] = self.ola_buf[hop:]
        self.ola_buf[-hop:] = 0.0
        return output

    def process(self, error_hop: np.ndarray, echo_spec: np.ndarray,
                far_power: float, far_spec: np.ndarray = None,
                filter_converged: bool = False,
                erle_factor: float = 0.0,
                dt_indicator: float = 0.0,
                near_spec: np.ndarray = None,
                divergence: float = 0.0,
                is_stationary_dt: bool = False,
                saturation_level: float = 0.0,
                epc_active: bool = False,
                error_spec_from_filter: np.ndarray = None,
                shadow_dt: float = 0.0,
                erl_estimate: float = 0.01,
                e2_main: float = 0.0,
                e2_shadow: float = 0.0,
                y2: float = 0.0,
                filter_once_converged: bool = False,
                aec_state=None,
                filter_state: str = 'idle') -> np.ndarray:
        """Process hop-size error signal, return enhanced hop via OLA.

        far_spec: far-end frequency spectrum (complex), used for coherence-
                  based nonlinear echo PSD estimation.
        near_spec: mic signal spectrum (complex), used for direct echo method.
        error_spec_from_filter: error spectrum from PBFDAF (rectangular window,
                  aligned with far_spec/near_spec/echo_spec). When provided,
                  used for coherence/ENR calculation instead of the OLA spec.
        shadow_dt: double-talk confidence from shadow-filter-based DTD
                  (main/shadow error ratio). Unlike energy-based dt_indicator
                  which is suppressed by inst_erle correction in high-coupling
                  DT, shadow_dt is reliable in exactly the crush cases. Used
                  to drive effective_dt for ne_floor and ENR relaxation.
        """
        hop = self.hop_size

        # Slide in new error samples (frame_size buffer)
        self.input_buf[:-hop] = self.input_buf[hop:]
        self.input_buf[-hop:] = error_hop

        # Analysis: sqrt-Hann window + zero-pad to FFT size
        # This spec is used ONLY for synthesis (gain application + IFFT).
        windowed = self.input_buf * self.window
        spec_synth = np.fft.rfft(windowed, n=self.block_size)

        # Analysis spec for coherence/PSD/ENR: prefer filter's error_spec
        # (rectangular window, aligned with far_spec/near_spec/echo_spec).
        # Fallback to OLA spec if not provided (backward compat).
        if error_spec_from_filter is not None:
            spec = error_spec_from_filter
        else:
            spec = spec_synth

        # Compute power spectra
        echo_pwr_linear = np.abs(echo_spec) ** 2
        error_pwr = np.abs(spec) ** 2

        # --- Coherence-based echo PSD estimation ---
        # Coherence² between far-end and error IS the echo-to-error ratio:
        #   coh²[k] = |S_fe[k]|² / (S_ff[k] × S_ee[k])
        # This captures both linear and nonlinear echo (both correlate with
        # far-end). During DT, near-end is uncorrelated → coh² drops → less
        # suppression → near-end preserved.
        coh2 = np.zeros(self.n_freqs, dtype=np.float32)
        if far_spec is not None and far_power > 1e-4:
            a = self.alpha_coh
            self.S_fe = a * self.S_fe + (1 - a) * spec * np.conj(far_spec)
            self.S_ff = a * self.S_ff + (1 - a) * np.abs(far_spec) ** 2
            self.S_ee = a * self.S_ee + (1 - a) * error_pwr
            coh2_raw = np.abs(self.S_fe) ** 2 / (self.S_ff * self.S_ee + 1e-10)
            coh2_raw = np.minimum(coh2_raw, 1.0).astype(np.float32)
            # Asymmetric EMA: fast drop (DT protection) / slow rise (stable tracking)
            # After convergence: rise slower → coh2 more stable near 1.0 in echo-only
            # _coh2_smooth initialized in __init__ and cleared in reset()
            if filter_converged:
                a_coh_rise = 0.90   # TC≈160ms, stable echo-only tracking
                a_coh_drop = 0.50   # TC≈25ms, fast DT protection
            else:
                a_coh_rise = 0.80
                a_coh_drop = 0.50
            a_coh = np.where(coh2_raw < self._coh2_smooth, a_coh_drop, a_coh_rise)
            self._coh2_smooth = a_coh * self._coh2_smooth + (1.0 - a_coh) * coh2_raw
            coh2 = self._coh2_smooth
        else:
            if far_power <= 1e-4:
                self.S_fe *= 0.5
                self.S_ff *= 0.5
                self.S_ee *= 0.5  # A1: sync decay, prevent coh2 bias on far restart
        # Round 4 trace cache (audio-passive)
        self._diag_coh2_last = coh2

        # Cold start: skip EMA warmup, initialize PSD directly on first far-end frame
        if far_power > 1e-4 and np.sum(self.echo_psd) < 1e-10:
            self.echo_psd[:] = echo_pwr_linear
            self.error_psd[:] = error_pwr

        # Linear EER from adaptive filter echo estimate
        self.echo_psd = self.alpha_echo_psd * self.echo_psd + (1 - self.alpha_echo_psd) * echo_pwr_linear
        self.error_psd = self.alpha_error_psd * self.error_psd + (1 - self.alpha_error_psd) * error_pwr

        # Multi-ERLE update (Phase 2)
        far_active = far_power > 1e-4
        self._filter_erle_est.update(echo_spec, spec, far_active, dt_indicator)
        near_power_broad = float(np.mean(self.near_psd)) if near_spec is not None else 0.0
        error_power_broad = float(np.mean(error_pwr))
        self._fb_erle_est.update(near_power_broad, error_power_broad, far_active, dt_indicator)

        if far_power < 1e-4:
            self.echo_psd *= 0.3  # fast decay during far-end silence

        # --- Dynamic g_min: track far-end activity ---
        is_far_active = float(far_power > 1e-4)
        if is_far_active > self.far_activity:
            # Far-end resumes: fast attack (TC≈30ms, ~2 frames)
            self.far_activity = 0.7 * self.far_activity + 0.3 * is_far_active
        else:
            # Far-end stops: slow decay (TC≈800ms, wait for echo_psd to decay first)
            self.far_activity = 0.98 * self.far_activity + 0.02 * is_far_active
        # far_activity=1.0 → g_min normal; far_activity=0.0 → g_min→1.0 (no suppression)
        # Fixed g_min: gain floor is constant regardless of far_activity
        # (AEC3 style — gain is purely ENR-driven, not activity-gated)
        effective_g_min = self.g_min

        # --- Noise gate: don't suppress quiet segments ---
        signal_floor = np.mean(self.error_psd) * 0.001 + 1e-8
        quiet_mask = ((self.echo_psd < signal_floor)
                      & (self.error_psd < signal_floor))

        # --- Stationary DT virtual DT indicator ---
        dt_for_fs = 0.8 if is_stationary_dt else dt_indicator

        # Effective DT: includes pre-filter energy-based signal (shadow_dt
        # parameter, reused as transport for energy DT signal). This bypasses
        # the inst_erle correction that suppresses dt_indicator in high-
        # coupling DT. Take max so FS paths still rely on dt_for_fs ≈ 0
        # while DT paths get the energy signal.
        effective_dt = max(float(dt_for_fs), float(shadow_dt))
        self._last_effective_dt = effective_dt

        # EPC_DT: echo path change detected AND double-talk active.
        # Gain cap bypasses ENR path (locked ~1.0 by DT nearend protection).
        # v3.12 S6b research substrate (default OFF): state-driven gate
        # option. See docs/v3_12_s6b_verdict.md — legacy gate was found
        # to fire 0% at production thresholds, so the cap is dead code
        # in BALANCED. Flag retained for future Phase 3C state-set work.
        if self._state_driven_epc_dt_cap and self._consume_filter_state:
            epc_dt = filter_state in ('diverged', 'suspicious_dt')
        else:
            epc_dt = epc_active and effective_dt > 0.35

        eps = 1e-10
        residual_echo_psd = self._stage_residual_model(
            coh2=coh2,
            far_spec=far_spec,
            far_power=far_power,
            erle_factor=erle_factor,
            dt_for_fs=dt_for_fs,
            epc_active=epc_active,
            saturation_level=saturation_level,
            filter_converged=filter_converged,
            erl_estimate=erl_estimate,
            near_spec=near_spec,
            is_stationary_dt=is_stationary_dt,
            aec_state=aec_state,
            echo_pwr_linear=echo_pwr_linear,
        )

        # v3.8.x ABL-4 (ablate ERL-based linear_failed branch): trace-verified
        # `self._erl_estimate` clipped to [0.001, 1.0] in update path (~line
        # 3974), so `erl_estimate > 1.2` never triggers (R6 trace 2026-04-30:
        # 0.00% fire rate across all 5 buckets). Branch was dead code retained
        # as v3.7.1 PR-B "physical mic/far ratio" defense, but value is
        # structurally bounded → defense is impossible to engage.
        # Removed alongside ABL-1+2: completes the family of e2-floor /
        # mic-as-echo-proxy / error-as-echo-proxy structural cleanups.

        # Compute coherence-based EER (only used by legacy spectral_sub path)
        if self.gain_type not in ("enr", "wiener"):
            eer_linear = self.echo_psd / (self.error_psd + eps)
            eer_converged = eer_linear * (0.5 + 0.5 * coh2)
            if far_power > 1e-4:
                eer = (1.0 - erle_factor) * coh2 + erle_factor * eer_converged
            else:
                eer = eer_converged
        else:
            eer = None  # B3: not used in ENR/Wiener path


        # --- Spectral-shape-preserving floor ---
        if self.enable_spectral_floor and far_power > 1e-4:
            error_mag = np.sqrt(error_pwr + 1e-10)
            self.error_envelope = (self.alpha_envelope * self.error_envelope
                                   + (1 - self.alpha_envelope) * error_mag)
            env_max = np.max(self.error_envelope) + 1e-10
            env_normalized = self.error_envelope / env_max
            # Bins with more energy get higher floor → preserves spectral shape
            spectral_g_min = effective_g_min + (1.0 - effective_g_min) * env_normalized * self.spectral_floor_ratio
            spectral_g_min = np.maximum(spectral_g_min, effective_g_min)
        else:
            spectral_g_min = effective_g_min

        # fs_confidence: continuous FS/DT/NE indicator (single definition)
        # Used by ne_g_floor, ENR two-tuning, attack speed, LF rate limit
        fs_confidence = self.far_activity * (1.0 - effective_dt) ** 2.0

        # --- Per-bin near-end gate ---

        # v3.12 Phase 3B (S6-S7) — unified gain floor.
        # `ne_g_floor` uses `(1 - coh²)` as NE evidence by default. Q7 V3
        # verdict: this evidence saturates in FS post-cancellation
        # (echo cancelled → low coh² → "NE-like") and so falsely raises
        # the floor in FS, leaking echo. The unified path swaps in the
        # F3.1 v3 mic-excess blend (already validated AUROC 0.871, in
        # production use for `dt_per_bin` since v3.10.6). Gate matches
        # the F3.1 v3 path: filter_converged AND long-window-ready AND
        # not epc_active; fallback to `(1 - coh²)` when gate fails.
        if self._unified_gain_floor and self._use_mic_excess_evidence:
            _lw_ready_floor = (
                self._residual_est is not None
                and self._residual_est._long_window_n_updates > 0
            )
            if filter_converged and _lw_ready_floor and not epc_active:
                far_lw = self._residual_est._long_window_far_psd
                # v3.14 Arc-P P.S2: unified_gain_floor uses per-bin ERL when array.
                if isinstance(erl_estimate, np.ndarray):
                    erl_e = erl_estimate  # per-bin: broadcast against far_lw
                else:
                    erl_e = float(erl_estimate)
                excess = np.maximum(self.error_psd - far_lw * erl_e, 0.0)
                excess_ratio = np.clip(
                    excess / (self.error_psd + 1e-10), 0.0, 1.0,
                ).astype(np.float32)
                legacy_evidence = 1.0 - coh2
                ne_evidence = (_BLEND_F31_MIC_EXCESS * excess_ratio
                               + (1.0 - _BLEND_F31_MIC_EXCESS) * legacy_evidence)
            else:
                ne_evidence = 1.0 - coh2
        else:
            ne_evidence = 1.0 - coh2

        # --- Per-bin near-end gate with fs_confidence ---
        ne_erle_gate = max(erle_factor, 0.3)  # B4: simplified (0.2 floor never triggered)
        # Scale ne_protection by (1-fs_confidence): FS→no protection, DT/NE→full protection
        ne_protection = ne_evidence * ne_erle_gate * (1.0 - fs_confidence)
        ne_g_min_ceil = 10 ** (self.ne_protect_db / 20)
        ne_g_floor = effective_g_min + (ne_g_min_ceil - effective_g_min) * ne_protection
        ne_g_floor = np.maximum(ne_g_floor, effective_g_min)
        spectral_g_min = np.maximum(spectral_g_min, ne_g_floor)

        # startup_dt conditions: trigger when DT active + filter not converged
        _startup_dt_curr = effective_dt > 0.35 and far_power > 1e-4 and not filter_converged
        _startup_dt_once = effective_dt > 0.35 and far_power > 1e-4 and not filter_once_converged

        if self._stats is not None:
            self._stats_last_ne_g_floor = float(np.mean(ne_g_floor))
            self._stats_last_spectral_g_min = float(np.mean(spectral_g_min))

        # startup_dt gain floor cap: lower spectral_g_min ceiling for ablation
        if _startup_dt_curr and self.startup_dt_gain_floor < 1.0:
            spectral_g_min = np.minimum(spectral_g_min, self.startup_dt_gain_floor)

        g = self._stage_gain_compute(
            residual_echo_psd=residual_echo_psd,
            eer=eer,
            coh2=coh2,
            effective_dt=effective_dt,
            is_stationary_dt=is_stationary_dt,
            far_power=far_power,
            filter_once_converged=filter_once_converged,
            spectral_g_min=spectral_g_min,
            eps=eps,
            erl_estimate=erl_estimate,
            filter_converged=filter_converged,
            epc_active=epc_active,
            filter_state=filter_state,
        )
        g = self._stage_gain_postprocess(
            g_in=g,
            epc_dt=epc_dt,
            quiet_mask=quiet_mask,
            far_power=far_power,
            effective_dt=effective_dt,
            is_stationary_dt=is_stationary_dt,
            divergence=divergence,
            erl_estimate=erl_estimate,
        )

        self._stage_temporal_smoothing(
            g_in=g,
            dt_indicator=dt_indicator,
            effective_dt=effective_dt,
            is_stationary_dt=is_stationary_dt,
            erle_factor=erle_factor,
            fs_confidence=fs_confidence,
            spectral_g_min=spectral_g_min,
            effective_g_min=effective_g_min,
        )

        output = self._stage_noise_floor_and_cng(
            spec_synth=spec_synth,
            far_power=far_power,
            effective_dt=effective_dt,
            dt_indicator=dt_indicator,
            filter_once_converged=filter_once_converged,
            effective_g_min=effective_g_min,
            hop=hop,
        )

        # Per-frame stats accumulation (no-op when _stats is None)
        if self._stats is not None:
            s = self._stats
            dt_k = effective_dt > 0.35 and far_power > 1e-4
            s['total_frames'] += 1
            if dt_k:
                s['dt_active'] += 1
                s['dt_erle_sum'] += float(erle_factor)
                s['dt_gain_sum'] += float(np.mean(self.gain_smooth))
                s['dt_enr_sum'] += self._stats_last_enr
                s['dt_res_sum'] += self._stats_last_res_psd
                s['dt_ne_sum'] += self._stats_last_nearend
                if epc_active:               s['epc_dt'] += 1
                if erle_factor < 0.4:        s['low_erle_dt'] += 1
                if not filter_converged:     s['startup_dt'] += 1
                if epc_active or erle_factor < 0.4 or not filter_converged:
                    s['any_special_dt'] += 1
                # (2) filter convergence/divergence
                if filter_once_converged:    s['filter_once_converged_dt'] += 1
                if not filter_once_converged: s['startup_dt_once_conv'] += 1
                if divergence > 0.3:         s['filter_diverged_dt'] += 1
                _e2m_y2 = e2_main / (y2 + 1e-10)
                _e2s_y2 = e2_shadow / (y2 + 1e-10)
                _e2s_e2m = e2_shadow / (e2_main + 1e-10)
                s['e2_main_y2_sum'] += _e2m_y2
                s['e2_shadow_y2_sum'] += _e2s_y2
                s['e2_shadow_e2_main_sum'] += _e2s_e2m
                if e2_shadow < e2_main:      s['shadow_better_dt'] += 1
                # (3) usability candidates (DT frame)
                _uv1 = filter_converged
                _uv2 = _uv1 and erle_factor > 0.4
                _uv3 = _uv2 and divergence <= 0.3
                _uv4 = _uv3 and _e2s_e2m > 0.8       # shadow not better than main (same units)
                if _uv1: s['usable_v1'] += 1
                if _uv2: s['usable_v2'] += 1
                if _uv3: s['usable_v3'] += 1
                if _uv4: s['usable_v4'] += 1
                if not _uv1: s['unusable_dt'] += 1
                # (4) residual echo model — wire using_render directly from estimator
                if self._residual_est.using_render_based:
                    s['using_render_based_dt'] += 1
                s['min_ne_sum'] += self._stats_last_min_ne
                # (5) startup_dt gain stage diagnostics (only when not filter_converged)
                if not filter_converged:
                    s['st_ne_g_floor_sum'] += self._stats_last_ne_g_floor
                    s['st_spectral_g_min_sum'] += self._stats_last_spectral_g_min
                    s['st_gain_before_floor_sum'] += self._stats_last_gain_before_floor
                    s['st_gain_after_floor_sum'] += self._stats_last_gain_after_floor
                    s['st_gain_after_smoothing_sum'] += self._stats_last_gain_after_smoothing
                # (6) noise_floor_gain diagnostics (only when not filter_once_converged)
                if not filter_once_converged:
                    s['st_nfl_noise_floor_gain_sum'] += self._stats_last_noise_floor_gain
                    s['st_nfl_noise_psd_sum'] += self._stats_last_noise_psd
                    s['st_nfl_spec_pwr_sum'] += self._stats_last_spec_pwr
                    if self._stats_last_nfl_lifted: s['st_nfl_lifted_count'] += 1
                    s['st_nfl_final_gain_sum'] += float(np.mean(self.gain_smooth))

        # Diagnostic: store latest gains for external access
        self._diag_gain_mean = float(np.mean(self.gain_smooth))
        self._diag_gain_min = float(np.min(self.gain_smooth))
        self._diag_effective_g_min = float(effective_g_min)
        self._diag_far_activity = float(self.far_activity)
        self._diag_echo_psd_mean = float(np.mean(self.echo_psd))
        self._diag_error_psd_mean = float(np.mean(self.error_psd))

        # Round 4 per-bin RES diagnostics (audio-passive). Built only when ENR
        # path actually ran (residual_echo_psd not None). For non-ENR or no-far
        # frames, leave previous values stale (still float-valued, won't break
        # downstream).
        _coh2 = self._diag_coh2_last
        _ne = self._diag_nearend_est_last
        _res = self._diag_residual_echo_psd_last
        _err = self.error_psd
        _noise = self.noise_psd
        _vidx = self._voice_band_idx
        _eps = 1e-10
        _err_eps = _err + _eps
        # Clip ratios at 10: residual_echo_psd / nearend_est can be ≫1 in
        # quiet bins where error_psd ≈ 0; raw mean would be dominated by tails.
        _res_over_err = np.clip(_res / _err_eps, 0.0, 10.0)
        _ne_over_err = np.clip(_ne / _err_eps, 0.0, 10.0)
        _noise_over_err = np.clip(_noise / _err_eps, 0.0, 10.0)
        _g_final = self.gain_smooth
        _g_voice = _g_final[_vidx] if _vidx.size > 0 else _g_final
        _echo_dom = ((_coh2 > 0.5) & (_res_over_err > 0.5) & (_ne_over_err < 0.3))
        self._diag_round4 = {
            'coh2_mean_full': float(np.mean(_coh2)),
            'coh2_mean_voice': float(np.mean(_coh2[_vidx])) if _vidx.size > 0 else 0.0,
            'res_over_err_mean_full': float(np.mean(_res_over_err)),
            'res_over_err_mean_voice': float(np.mean(_res_over_err[_vidx])) if _vidx.size > 0 else 0.0,
            'ne_over_err_mean_full': float(np.mean(_ne_over_err)),
            'ne_over_err_mean_voice': float(np.mean(_ne_over_err[_vidx])) if _vidx.size > 0 else 0.0,
            'noise_over_err_mean_full': float(np.mean(_noise_over_err)),
            'g_voice_mean': float(np.mean(_g_voice)),
            'g_voice_min': float(np.min(_g_voice)),
            'g_voice_p10': float(np.percentile(_g_voice, 10)),
            'echo_dominant_bin_pct': float(np.mean(_echo_dom)),
        }

        return output.astype(np.float32)


class DtdEstimator:
    """Double-Talk Detector with per-mode strategy.

    - 'geigel' mode (LMS/NLMS): Geigel DTD with hangover + confidence
    - 'divergence' mode (frequency-domain modes): Output-vs-input divergence detection
    - 'coherence' mode (frequency-domain modes): Error-reference coherence DT detection
    """

    def __init__(self, mode: str = 'geigel', *,
                 window_blocks: int = 4,
                 geigel_threshold: float = 0.5,
                 hangover_max: int = 15,
                 divergence_factor: float = 1.5,
                 attack: float = 0.3,
                 release: float = 0.05,
                 warmup_frames: int = 50,
                 # Coherence mode params
                 n_freqs: int = 0,
                 coh_alpha: float = 0.85,
                 coh_high: float = 0.6,
                 coh_low: float = 0.3,
                 coh_energy_floor: float = 0.01,
                 coh_abs_floor: float = 1e-6,
                 sample_rate: int = 16000,
                 block_size: int = 512):
        self.mode = mode  # 'geigel', 'divergence', or 'coherence'
        self.confidence = 0.0
        self.attack = attack
        self.release = release
        self.warmup_frames = warmup_frames
        self.frame_count = 0

        # Geigel state
        self.far_abs_buffer = np.zeros(max(1, window_blocks))
        self.buf_idx = 0
        self.geigel_threshold = geigel_threshold
        self.hangover_max = hangover_max
        self.hangover_count = 0

        # Divergence state
        self.divergence_factor = divergence_factor

        # Coherence state
        self.coh_alpha = coh_alpha
        self.coh_high = coh_high
        self.coh_low = coh_low
        self.coh_energy_floor = coh_energy_floor
        self.coh_abs_floor = coh_abs_floor  # #8: Absolute energy floor
        if mode == 'coherence' and n_freqs > 0:
            self.S_ex = np.zeros(n_freqs, dtype=np.complex64)
            self.S_ee = np.zeros(n_freqs, dtype=np.float32)
            self.S_xx = np.zeros(n_freqs, dtype=np.float32)
            # #7: Voice-band weighting (300Hz-4kHz emphasized)
            self.voice_weight = np.ones(n_freqs, dtype=np.float32)
            freq_per_bin = sample_rate / block_size
            for k in range(n_freqs):
                f = k * freq_per_bin
                if 300.0 <= f <= 4000.0:
                    self.voice_weight[k] = 3.0  # 3× weight for speech band
                elif f < 100.0 or f > 6000.0:
                    self.voice_weight[k] = 0.3  # De-weight extremes
        else:
            self.S_ex = None
            self.S_ee = None
            self.S_xx = None
            self.voice_weight = None

    def reset(self):
        self.confidence = 0.0
        self.frame_count = 0
        self.far_abs_buffer.fill(0)
        self.buf_idx = 0
        self.hangover_count = 0
        if self.S_ex is not None:
            self.S_ex.fill(0)
            self.S_ee.fill(0)
            self.S_xx.fill(0)

    def _update_confidence(self, detected: bool):
        """Update confidence with attack/release + hangover."""
        if detected:
            self.hangover_count = self.hangover_max
            self.confidence = min(self.confidence + self.attack, 1.0)
        elif self.hangover_count > 0:
            self.hangover_count -= 1
            self.confidence = max(self.confidence - self.release * 0.5, 0.0)
        else:
            self.confidence = max(self.confidence - self.release, 0.0)

    def _detect_geigel(self, near_end: np.ndarray, far_end: np.ndarray):
        """Geigel DTD: |mic| > threshold × max(|ref|) over window."""
        # Update far-end max circular buffer
        self.far_abs_buffer[self.buf_idx] = np.max(np.abs(far_end))
        self.buf_idx = (self.buf_idx + 1) % len(self.far_abs_buffer)
        far_max = np.max(self.far_abs_buffer)

        # Geigel test
        near_max = np.max(np.abs(near_end))
        detected = (far_max > 1e-6) and (near_max > self.geigel_threshold * far_max)

        self._update_confidence(detected)

    def _detect_divergence(self, near_end: np.ndarray, output: np.ndarray):
        """Output-vs-input divergence detection (output > input).

        Uses both energy-based and peak-based detection. Peak-based catches
        localized spikes that energy-based misses (e.g., transition transients).
        """
        output_energy = np.mean(output ** 2)
        near_energy = np.mean(near_end ** 2)
        output_peak = np.max(np.abs(output))
        near_peak = np.max(np.abs(near_end))

        if near_energy < 1e-10 and near_peak < 1e-6:
            # Silence → release
            self.confidence = max(self.confidence - self.release, 0.0)
            return

        # Check both energy and peak divergence
        energy_ratio = output_energy / (near_energy + 1e-10) if near_energy > 1e-10 else 0.0
        peak_ratio = output_peak / (near_peak + 1e-10) if near_peak > 1e-6 else 0.0
        ratio = max(energy_ratio, peak_ratio)

        mild_threshold = 1.2  # ratio < 1.2 is normal (unconverged, not diverging)
        if ratio > self.divergence_factor:
            # Severe divergence
            self.confidence = min(self.confidence + self.attack, 1.0)
        elif ratio > mild_threshold:
            # Mild divergence — proportional attack
            self.confidence = min(
                self.confidence + self.attack * (ratio - mild_threshold), 1.0)
        else:
            # Normal — faster release when ratio is well below 1.0
            release_scale = max(1.0 - ratio, 0.2)  # 0.2x ~ 1.0x
            self.confidence = max(
                self.confidence - self.release * (1.0 + 4.0 * release_scale), 0.0)

    def _detect_coherence(self, error_spec: np.ndarray, far_spec: np.ndarray):
        """Coherence-based double-talk detection.

        Uses smoothed magnitude-squared coherence between error and far-end.
        Low coherence + high error energy → near-end speech present → DT.
        High coherence → residual echo (unconverged) → keep updating.
        """
        alpha = self.coh_alpha

        # Update smoothed PSDs
        cross = error_spec * np.conj(far_spec)
        self.S_ex = alpha * self.S_ex + (1 - alpha) * cross
        self.S_ee = alpha * self.S_ee + (1 - alpha) * np.abs(error_spec) ** 2
        self.S_xx = alpha * self.S_xx + (1 - alpha) * np.abs(far_spec) ** 2

        # #7: Voice-band weighted coherence (ratio-of-sums)
        w = self.voice_weight
        num = np.sum(w * np.abs(self.S_ex) ** 2)
        den = np.sum(w * self.S_ee * self.S_xx)
        coherence = num / (den + 1e-10)

        # Energy check: only declare DT if error has meaningful energy
        sum_ee = np.sum(self.S_ee)
        sum_xx = np.sum(self.S_xx)
        # #8: Absolute energy floor prevents false triggers on quiet far-end
        has_energy = (sum_ee > self.coh_energy_floor * sum_xx and
                      sum_xx > 1e-10 and
                      sum_ee > self.coh_abs_floor)

        if coherence > self.coh_high:
            # Correlated → residual echo, not DT → release
            self._update_confidence(False)
        elif coherence < self.coh_low and has_energy:
            # Uncorrelated + energy → near-end speech → DT
            self._update_confidence(True)
        else:
            # Ambiguous → slow release
            self.confidence = max(self.confidence - self.release * 0.5, 0.0)

    def detect_block(self, near_end: np.ndarray, far_end: np.ndarray,
                     output: np.ndarray = None,
                     error_spec: np.ndarray = None,
                     far_spec: np.ndarray = None) -> float:
        """Update DTD state and return confidence [0.0, 1.0].

        For geigel mode: uses near_end and far_end.
        For divergence mode: uses near_end and output.
        For coherence mode: uses error_spec and far_spec.
        """
        self.frame_count += 1
        # Warmup: all detectors share the same warmup period.
        # Coherence also needs warmup because unconverged filter → error ≈ echo
        # → coherence estimate is unreliable (false DT triggers).
        if self.frame_count < self.warmup_frames:
            return 0.0

        if self.mode == 'geigel':
            self._detect_geigel(near_end, far_end)
        elif self.mode == 'coherence':
            if error_spec is not None and far_spec is not None:
                self._detect_coherence(error_spec, far_spec)
        else:
            self._detect_divergence(near_end, output)

        return self.confidence


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


@dataclass
class FilterConvergenceState:
    """One-frame snapshot of filter convergence health."""
    converged: bool
    once_converged: bool
    just_converged: bool   # True for the single frame the transition fires
    divergence: float      # [0, 1] EMA: rate of post-convergence inst-ERLE < -2 dB


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

    Coherence-based DT lives in DtdEstimator (self.dtd_coherence on AEC) and
    is read by name from there; this analyzer does not own it but exposes a
    combined() helper for AecState assembly.
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
    """v3.10.0 — detect filter stuck at bad-convergence plateau and trigger
    a one-shot recovery (taps reset + re-warmup).

    Motivation
    ----------
    DT-from-frame-0 cases (NE talks from sample 0) cause the main filter
    to learn NE leak in the first ~100 frames. Once these poisoned taps
    are encoded, even a perfect DTD downstream only prevents *further*
    learning — the existing taps stay wrong, ERLE stays low (~4 dB),
    coherence DTD never recovers (the positive-feedback trap that the
    v3.9.x DTD fix addresses upstream of mu_scale, but not at the tap
    level).

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

        # v3.10.3 — require current frame to also show DT presence, otherwise
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

        # Fire — recovery action.
        # v3.10.3 (H3) — also reset cumulative far/dt counters so the next
        # evaluation window judges post-reset behaviour, not the bad-startup
        # history that caused this fire. _attempts and _last_reset_frame
        # remain session-level (we still want MAX_ATTEMPTS to bound retries).
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


class ResidualEchoEstimator:
    """Two-path residual echo PSD attribution (stage 1 + stage 2 of ResFilter).

    Replaces ResFilter's inline ~100 lines of residual_echo_psd computation with
    an explicit class that owns the render-based switching state. Provides two
    modes:

      'legacy': bit-exact reproduction of v2.8.1 inline logic — ERLE-blended
                linear estimate with optional render-based blend driven by
                an ENR-adaptive switching threshold + hysteresis + min hold.
                Used as default for parity validation.

      'split' : explicit AEC3-style branch on `aec_state.usable_linear_estimate`.
                Linear path: `S2_linear / ERLE`. Nonlinear path: `X2 * echo_path_gain`.
                Used by Phase B2 ablation (R1 variant). Disabled by default.

    Owns: _using_render_based, _render_based_hold (legacy state machine).
    Caller (ResFilter) keeps echo_psd / error_psd / near_psd / coh2 / far_activity
    on itself; we pass them in by reference per call.
    """
    LEGACY = 'legacy'
    SPLIT = 'split'

    def __init__(self, n_freqs: int, mode: str = 'legacy',
                 long_window_alpha: float = 0.993):
        """v3.10.0: long-window render PSD EMA mirrors WebRTC AEC3's
        kRenderTransferQueueSizeFrames=100 (= 1000 ms at 10ms/frame). When
        delay alignment is unreliable, RES uses this long-time-averaged
        far PSD as the echo PSD basis instead of the instantaneous far_psd
        (which is sensitive to misalignment). Speech long-term PSD is
        roughly stationary over 1 s, so the smoothed estimate gives a
        usable echo template even when the filter taps are wrong.

        TC ≈ 100 frames at alpha=0.993 (alpha^100 ≈ 0.5).
        """
        self.n_freqs = n_freqs
        self.mode = mode
        self._using_render_based = False
        self._render_based_hold = 0
        self._long_window_alpha = long_window_alpha
        self._long_window_far_psd = np.zeros(n_freqs, dtype=np.float32)
        self._long_window_n_updates = 0
        # P3g Phase 0 dry-run scalars
        self._last_linear_residual_psd_mean = 0.0
        self._last_render_residual_psd_mean = 0.0
        self._last_render_blend = 0.0

    def reset(self, preserve_long_window_ema: bool = False) -> None:
        """Clear residual-echo estimator state.

        preserve_long_window_ema (v3.10.2): when True, keep the long-window
        far-PSD EMA (`_long_window_far_psd` + `_long_window_n_updates`).
        Used by filter-derived-state resets (plateau recovery, delay_first):
        the EMA is input-side context — its accumulated long-term render
        spectrum is independent of the bad filter taps and should survive
        the reset, otherwise the freshly reset filter spends 100 frames
        in pre-warmup-fallback mode again.
        """
        self._using_render_based = False
        self._render_based_hold = 0
        if not preserve_long_window_ema:
            self._long_window_far_psd.fill(0.0)
            self._long_window_n_updates = 0

    @property
    def using_render_based(self) -> bool: return self._using_render_based

    def attribute(self, *, aec_state=None, **kw) -> np.ndarray:
        """Mode-dispatch entry. Caller passes the union of legacy+split kwargs;
        method picks the relevant subset."""
        if self.mode == self.SPLIT and aec_state is not None:
            return self.attribute_split(
                echo_psd=kw['echo_psd'], error_psd=kw['error_psd'],
                far_spec=kw['far_spec'], far_power=kw['far_power'],
                erle_factor=kw['erle_factor'], erl_estimate=kw['erl_estimate'],
                filter_erle=kw['filter_erle'], fb_erle=kw['fb_erle'],
                aec_state=aec_state,
            )
        return self.attribute_legacy(**kw)

    def attribute_legacy(self, *, echo_psd: np.ndarray, error_psd: np.ndarray,
                         coh2: np.ndarray, far_spec, far_power: float,
                         erle_factor: float, dt_for_fs: float, far_activity: float,
                         epc_active: bool, saturation_level: float,
                         filter_converged: bool, erl_estimate: float,
                         filter_erle, fb_erle) -> np.ndarray:
        """Stage 1 (ERLE-blended linear) + Stage 2 (render-based switch).

        Bit-exact port of ResFilter.process() residual-echo block from v2.8.1
        (lines ~1456-1555). Mutates self._using_render_based / _render_based_hold.
        """
        # Multi-ERLE residual estimation (Phase 2)
        confidence = compute_erle_confidence(filter_erle.erle, fb_erle.fb_erle)
        erle_corrected = (confidence * filter_erle.erle
                          + (1.0 - confidence) * 1.0)
        erle_corrected = np.maximum(erle_corrected, 0.5)

        erle_est = echo_psd / erle_corrected
        direct_est = echo_psd

        if far_power > 1e-4:
            dt_weight = 1.0 - dt_for_fs
            nonlinear_floor = error_psd * coh2 * far_activity * dt_weight
            direct_est = np.maximum(direct_est, nonlinear_floor)
            erle_est = np.maximum(erle_est, nonlinear_floor)

        residual_echo_psd = (1.0 - erle_factor) * direct_est + erle_factor * erle_est
        # P3g Phase 0: stash the linear-only residual (Stage-1 ERLE-blended)
        # for dry-run audit before any Stage-2 render-based override.
        # Read by AEC.process() into _diag — purely diagnostic.
        self._last_linear_residual_psd_mean = float(np.mean(residual_echo_psd))

        # v3.10.0: maintain long-window far-PSD EMA every frame regardless of
        # render mode — so it's ready immediately when delay drops out and
        # we need a delay-agnostic echo template. Skip on silence to avoid
        # poisoning the EMA with the noise floor.
        if far_spec is not None and far_power > 1e-6:
            inst_far_psd = (np.abs(far_spec) ** 2).astype(np.float32)
            if self._long_window_n_updates == 0:
                self._long_window_far_psd[:] = inst_far_psd
            else:
                a = self._long_window_alpha
                self._long_window_far_psd = (a * self._long_window_far_psd
                                              + (1.0 - a) * inst_far_psd)
            self._long_window_n_updates += 1

        if far_power > 1e-4:
            error_power_mean = float(np.mean(error_psd)) + 1e-10
            enr = far_power / error_power_mean
            switching_threshold = 0.5 * np.clip(enr / (enr + 1.0), 0.3, 0.7)
            hysteresis = 0.05
            if self._using_render_based:
                effective_threshold = switching_threshold + hysteresis
            else:
                effective_threshold = switching_threshold
            force_render = (
                epc_active
                or saturation_level > 0.5
                or not filter_converged
            )
            want_render = (erle_factor < effective_threshold) or force_render
            if want_render and not self._using_render_based:
                self._render_based_hold = 5
            if self._using_render_based:
                self._render_based_hold = max(self._render_based_hold - 1, 0)
            can_exit = (not want_render and self._render_based_hold == 0)
            self._using_render_based = want_render or (self._using_render_based and not can_exit)

            if self._using_render_based:
                # v3.10.1: long-window EMA far-PSD blended with instantaneous.
                # The EMA is updated EVERY far-active frame (regardless of
                # mode — see block above) so its time constant is constant,
                # not dependent on how long fallback has been engaged.
                # We READ it only here (fallback-only). Two refinements
                # vs v3.10.0's hard-replace:
                #   (a) Warmup gate: 100 updates (= 1 EMA TC at alpha=0.993)
                #       so the EMA has formed a real long-term estimate
                #       before being trusted. Below 100 updates: instantaneous
                #       only.
                #   (b) Blend, not hard replace: even after warmup, mix 70%
                #       long-window + 30% instantaneous. The instantaneous
                #       component preserves response to fast far-end onsets;
                #       the long-window component carries delay-agnostic
                #       structure. This avoids the FS smearing risk of
                #       hard-replacing inst with stale EMA when far is
                #       changing fast (Codex finding 3).
                if far_spec is not None:
                    inst_far_psd = (np.abs(far_spec) ** 2).astype(np.float32)
                    warmup_weight = float(min(self._long_window_n_updates / 100.0, 1.0))
                    lw_weight = 0.7 * warmup_weight
                    far_psd = ((1.0 - lw_weight) * inst_far_psd
                               + lw_weight * self._long_window_far_psd)
                else:
                    far_psd = np.zeros(self.n_freqs, dtype=np.float32)
                # v3.8 ABL-1 (ablate v3.3 error_based_floor): error_psd contains
                # NE during DT, so using it as residual_echo floor structurally
                # over-suppresses near-end (same lesson as v3.7.1 PR-B). Use only
                # render_based_echo = far × ERL — AEC3-aligned.
                render_based_echo = far_psd * erl_estimate
                blend = 1.0 - erle_factor / effective_threshold
                blend = np.clip(blend, 0.0, 1.0)
                residual_echo_psd = ((1.0 - blend) * residual_echo_psd
                                     + blend * render_based_echo)
                # P3g Phase 0: stash the render-based component and the
                # blend used, for dry-run audit. (linear residual mean is
                # already stashed above before any override.)
                self._last_render_residual_psd_mean = float(np.mean(render_based_echo))
                self._last_render_blend = float(np.mean(blend))

        # If render-based path didn't run this frame, leave previous
        # render-residual stale; consumers should gate on using_render_based.
        return residual_echo_psd

    def attribute_split(self, *, echo_psd: np.ndarray, error_psd: np.ndarray,
                        far_spec, far_power: float,
                        erle_factor: float, erl_estimate: float,
                        filter_erle, fb_erle, aec_state) -> np.ndarray:
        """AEC3-style two-path R2: linear if `aec_state.usable_linear_estimate`,
        else render-based. Used by Phase B2 R1 ablation."""
        # Always update _using_render_based to reflect current decision
        self._using_render_based = not aec_state.usable_linear_estimate
        if aec_state.usable_linear_estimate:
            confidence = compute_erle_confidence(filter_erle.erle, fb_erle.fb_erle)
            erle_corrected = (confidence * filter_erle.erle + (1.0 - confidence) * 1.0)
            erle_corrected = np.maximum(erle_corrected, 0.5)
            return echo_psd / erle_corrected
        # Render-based path: X2 * echo_path_gain
        far_psd = np.abs(far_spec) ** 2 if far_spec is not None else np.zeros(self.n_freqs)
        return far_psd * erl_estimate


class AecState:
    """WebRTC AEC3-style read-only aggregator over the per-frame detector outputs.

    Holds references to the 5 detectors (RenderActivityDetector, DoubleTalkAnalyzer,
    FilterConvergenceAnalyzer, EchoPathChangeDetector, PathChangeRegimeHandler) and
    DtdEstimator-coherence; exposes derived flags via @property so consumers don't
    rebuild aggregation logic in multiple places.

    Phase B consumes this as the gate for two-path residual echo attribution:
    `usable_linear_estimate` decides linear vs nonlinear R2 model.

    All properties are read-only and reflect the current frame's detector state
    (no explicit per-frame update needed — properties delegate live).
    """

    def __init__(self, *, render_activity, convergence, dt_analyzer, epc_det,
                 regime_handler, dtd_coherence_getter):
        self._render = render_activity
        self._conv = convergence
        self._dt = dt_analyzer
        self._epc = epc_det
        self._regime = regime_handler
        self._dtd_coh = dtd_coherence_getter

    # ── Render activity ──────────────────────────────────────────────────────
    @property
    def render_active(self) -> bool: return self._render.is_active
    @property
    def render_stationary(self) -> bool: return self._render.is_stationary

    # ── Filter convergence ───────────────────────────────────────────────────
    @property
    def filter_converged(self) -> bool: return self._conv.converged
    @property
    def filter_once_converged(self) -> bool: return self._conv.once_converged
    @property
    def divergence(self) -> float: return self._conv.divergence

    # ── Double-talk signals ──────────────────────────────────────────────────
    @property
    def dt_from_energy(self) -> float: return self._dt.dt_from_energy
    @property
    def dt_from_shadow(self) -> float: return self._dt.dt_from_shadow
    @property
    def dt_from_coherence(self) -> float: return self._dtd_coh()
    @property
    def dt_combined(self) -> float:
        """Max of the three DT signals — same aggregation as inline code today."""
        return max(self._dt.dt_from_energy, self._dt.dt_from_shadow, self._dtd_coh())

    # ── Echo path change ─────────────────────────────────────────────────────
    @property
    def epc_active(self) -> bool: return self._epc.active
    @property
    def epc_hangover_count(self) -> int: return self._epc.hangover_count

    # ── Shadow filter management ─────────────────────────────────────────────
    @property
    def main_paused(self) -> bool: return self._regime.main_paused
    @property
    def shadow_advantage(self) -> float: return self._dt.shadow_advantage

    # ── Aggregated decisions (Phase B consumers) ─────────────────────────────
    @property
    def usable_linear_estimate(self) -> bool:
        """Can the main filter's echo estimate be trusted as residual-echo source?

        AEC3-style: requires once-converged + currently-converged + not in EPC recovery.
        When False, Phase B's residual echo estimator should fall back to
        render-window-based attribution.
        """
        return (self._conv.once_converged
                and self._conv.converged
                and not self._epc.active)


@dataclass
class EpcEvent:
    """One-frame EPC trigger result.

    fired:  True if this frame fired one of the three triggers (delay/epv/shadow_rise)
    source: which trigger fired ('delay' | 'epv' | 'shadow_rise')
    """
    fired: bool = False
    source: str = ''


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


class SubtractiveNLP:
    """E4 NLP detector — voice-band autocorrelation pitch tracker.

    Audit-only in v3.13 E4.S3 (no output mutation). Per
    docs/v3_13_e4_s2_design_lock.md: detects mic-side non-linear
    residual (loudspeaker NL + transmission codec NL) via FFT-based
    autocorrelation peak in 80-500 Hz fundamental band, gated on
    filter_state == 'refined_usable' and far_active.

    Outputs per-hop scalar `nl_confidence` ∈ [0, 1]; downstream
    suppressor (S5+) will consume.
    """

    def __init__(self, sample_rate: int, hop_size: int,
                 window_ms: float = 32.0,
                 pitch_threshold: float = 0.45,
                 continuity_frames: int = 3,
                 history_len: int = 4,
                 min_residual_rms: float = 0.05,
                 cancellation_ratio_threshold: float = 2.0,
                 cancellation_ema_alpha: float = 0.99):
        self._sr = int(sample_rate)
        self._hop = int(hop_size)
        win = int(window_ms * sample_rate / 1000)
        n = 1
        while n < win:
            n *= 2
        self._win_samples = n
        self._qmin = int(0.002 * sample_rate)
        self._qmax = int(0.0125 * sample_rate)
        self._pitch_threshold = float(pitch_threshold)
        self._continuity_frames = int(continuity_frames)
        self._history_len = int(history_len)
        self._min_residual_rms = float(min_residual_rms)
        # S4.1 cancellation gate — "filter is canceling something" signal.
        # NE bucket: ratio ≈ 1 (filter doesn't cancel because no echo).
        # FS converged: ratio > threshold (mic energy reduced by filter).
        self._cancel_ratio_thr = float(cancellation_ratio_threshold)
        self._cancel_ema_alpha = float(cancellation_ema_alpha)
        self._mic_rms_ema = 0.0
        self._raw_rms_ema = 0.0
        self._cancel_ratio_last = 0.0
        self._window = np.hanning(self._win_samples).astype(np.float32)
        self._buf = np.zeros(self._win_samples, dtype=np.float32)
        self._buf_pos = 0
        self._buf_filled = 0
        self._pitch_lag_hist = []
        self._nl_confidence_last = 0.0
        self._pitch_strength_last = 0.0
        self._pitch_lag_last = 0
        self._fire_count = 0
        self._call_count = 0

    def _append_hop(self, hop_samples: np.ndarray) -> None:
        n = len(hop_samples)
        end = self._buf_pos + n
        if end <= self._win_samples:
            self._buf[self._buf_pos:end] = hop_samples
        else:
            tail = self._win_samples - self._buf_pos
            self._buf[self._buf_pos:] = hop_samples[:tail]
            self._buf[:n - tail] = hop_samples[tail:]
        self._buf_pos = (self._buf_pos + n) % self._win_samples
        self._buf_filled = min(self._buf_filled + n, self._win_samples)

    def _compute_pitch_strength(self) -> tuple:
        if self._buf_pos == 0:
            frame = self._buf
        else:
            frame = np.concatenate(
                (self._buf[self._buf_pos:], self._buf[:self._buf_pos]))
        x = frame * self._window
        n2 = 2 * self._win_samples
        spec = np.fft.rfft(x, n=n2)
        ac = np.fft.irfft(np.abs(spec) ** 2, n=n2)[:self._win_samples]
        if ac[0] < 1e-15:
            return 0.0, 0
        ac = ac / ac[0]
        slab = ac[self._qmin: self._qmax + 1]
        peak_idx = int(np.argmax(slab))
        return float(slab[peak_idx]), self._qmin + peak_idx

    def process(self, hop_samples: np.ndarray,
                filter_state: str, far_active: bool,
                mic_hop_samples: 'Optional[np.ndarray]' = None) -> float:
        self._call_count += 1
        if len(hop_samples) != self._hop:
            self._nl_confidence_last = 0.0
            return 0.0
        self._append_hop(hop_samples.astype(np.float32))
        # S4.1 cancellation-ratio EMA: track mic_rms / raw_output_rms.
        # Updated every hop regardless of gating so EMA reflects sustained
        # filter activity. When NE bucket (no echo to cancel), filter
        # doesn't reduce mic energy → ratio ≈ 1. When FS converged,
        # ratio ≫ 1.
        hop_rms = float(np.sqrt(
            np.mean(hop_samples.astype(np.float64) ** 2) + 1e-15))
        if mic_hop_samples is not None:
            mic_rms = float(np.sqrt(
                np.mean(mic_hop_samples.astype(np.float64) ** 2) + 1e-15))
            a = self._cancel_ema_alpha
            self._mic_rms_ema = a * self._mic_rms_ema + (1.0 - a) * mic_rms
            self._raw_rms_ema = a * self._raw_rms_ema + (1.0 - a) * hop_rms
            if self._raw_rms_ema > 1e-9:
                self._cancel_ratio_last = self._mic_rms_ema / self._raw_rms_ema
            else:
                self._cancel_ratio_last = 0.0
        # Gate: allow refined_usable OR coarse_learning — heavy-NL cases
        # never reach refined_usable because the NL itself prevents linear
        # filter convergence. Excluded: idle / startup / diverged /
        # suspicious_dt (DT protection — NE voice could mimic pitched NL).
        if (filter_state not in ('refined_usable', 'coarse_learning')
                or not far_active
                or self._buf_filled < self._win_samples):
            self._nl_confidence_last = 0.0
            self._pitch_lag_hist.clear()
            return 0.0
        # S3.1 secondary gate: minimum residual RMS on the current hop.
        # Filters out pitched-but-low-residual frames in clean cases.
        if hop_rms < self._min_residual_rms:
            self._nl_confidence_last = 0.0
            return 0.0
        # S4.1 cancellation-ratio gate: filter must be canceling something
        # (mic_rms_ema / raw_rms_ema > threshold). NE bucket fails this
        # because filter has nothing to cancel.
        if (mic_hop_samples is not None
                and self._cancel_ratio_last < self._cancel_ratio_thr):
            self._nl_confidence_last = 0.0
            return 0.0
        pitch_strength, pitch_lag = self._compute_pitch_strength()
        self._pitch_strength_last = pitch_strength
        self._pitch_lag_last = pitch_lag
        is_pitched = pitch_strength >= self._pitch_threshold
        continuity_passed = False
        if is_pitched and self._pitch_lag_hist:
            tol = 0.05 * pitch_lag
            in_tol = sum(1 for prev in self._pitch_lag_hist
                         if abs(prev - pitch_lag) <= tol)
            required = min(self._continuity_frames, len(self._pitch_lag_hist))
            continuity_passed = in_tol >= required
        if is_pitched and continuity_passed:
            x_log = (pitch_strength - self._pitch_threshold) * 4.0
            nl_confidence = float(1.0 / (1.0 + np.exp(-x_log)))
            self._fire_count += 1
        else:
            nl_confidence = 0.0
        self._pitch_lag_hist.append(pitch_lag)
        if len(self._pitch_lag_hist) > self._history_len:
            self._pitch_lag_hist.pop(0)
        self._nl_confidence_last = nl_confidence
        return nl_confidence


class AEC:
    """
    Acoustic Echo Cancellation

    Supports five filter modes:
    - NLMS:    Time-domain NLMS (sample-by-sample processing)
    - FDAF:    Frequency-domain Adaptive Filter (single FFT block, n_partitions=1)
    - PBFDAF:  Partitioned Block FDAF (NLMS adaptation)
    - PBFDKF:  Partitioned Block FDKF (Kalman adaptation, recommended)
    """

    # ── Convergence-state delegations (state lives in self._convergence) ─────
    @property
    def _filter_converged(self) -> bool: return self._convergence.converged
    @property
    def _filter_once_converged(self) -> bool: return self._convergence.once_converged
    @property
    def _divergence_indicator(self) -> float: return self._convergence.divergence

    # ── DT-signal delegations (state lives in self._dt_analyzer) ─────────────
    @property
    def _dt_from_energy(self) -> float: return self._dt_analyzer.dt_from_energy
    @property
    def _dt_from_shadow(self) -> float: return self._dt_analyzer.dt_from_shadow
    @property
    def _shadow_advantage(self) -> float: return self._dt_analyzer.shadow_advantage

    # ── Phase C2 ablation knob: disable convergence reset on EPC sources ─────
    # Set is one of: {'delay_first', 'delay_shift', 'epv', 'shadow_rise'} or empty.
    # Used by _maybe_mark_diverged() to skip mark_diverged calls per-source.
    _epc_no_reset_sources: frozenset = frozenset()

    def _maybe_mark_diverged(self, source: str) -> None:
        if source not in self._epc_no_reset_sources:
            self._convergence.mark_diverged()
        # Round 3 trace: per-source counter (audio-passive)
        if not hasattr(self, '_round3_div_counts'):
            self._round3_div_counts = {'delay_first': 0, 'delay_shift': 0,
                                       'epv': 0, 'shadow_rise': 0}
        self._round3_div_counts[source] = self._round3_div_counts.get(source, 0) + 1
        self._round3_last_div_source = source

    def _f_e3_handle_epc_fire(self, source: str) -> None:
        """F-E3 consecutive-EPC handler. Called from EPC fire sites; checks
        whether this fire is within the consecutive window of a prior fire,
        and if so applies (a) hangover extension to ≥1s and (b) gated W
        partial reset (gap-guarded against cohort-tail spam).

        Always resets `_frames_since_last_epc` to 0; the caller must invoke
        this regardless of whether f_e3_enabled to keep the counter live.
        """
        if (self.config.f_e3_enabled
                and self._frames_since_last_epc
                    < self.config.f_e3_consecutive_window_frames):
            # E3-1: extend hangover to ≥1s
            min_hangover = self.config.f_e3_consecutive_window_frames
            if self._epc_det._hangover < min_hangover:
                self._epc_det._hangover = min_hangover
            # E3-3: gap-guarded W partial reset (cohort-tail defence)
            if (self._frames_since_last_f_e3_w_reset
                    >= self.config.f_e3_w_reset_min_gap_frames):
                factor = float(self.config.f_e3_w_reset_factor)
                for filt in [self.filter, self.shadow_filter]:
                    if filt is not None and hasattr(filt, 'W'):
                        filt.W *= factor
                self._frames_since_last_f_e3_w_reset = 0
        self._frames_since_last_epc = 0

    def _apply_epc_state_reset(self, source: str) -> None:
        """F2.1: reset stale upstream state on EPC rising edge.

        Restores `_erl_estimate`, the windowed ERLE accumulators, and the
        DT-jump baseline to their post-`__init__` values. The shipped EPC
        code already caps `_erl_estimate` to min(0.3) and boosts Q/P
        floors; this method completes the picture for state that has
        decay constants on the order of 10 s and otherwise persists
        across a path change. Gated by `config.use_epc_state_reset`;
        callers must check the flag before invoking.

        `source` is one of {'epv', 'shadow_rise'} for tracing; the reset
        body is identical because both trigger types invalidate the same
        upstream state by the same mechanism (room/path discontinuity).
        """
        self._erl_estimate = 0.1
        self._erle_window_near = 1e-10
        self._erle_window_err = 1e-10
        self._wn_err_baseline = 1e-8
        # Telemetry: count per-source firings so audit can correlate with
        # bench deltas. Audio-passive; never read by hot path.
        if not hasattr(self, '_f2_1_reset_counts'):
            self._f2_1_reset_counts = {'epv': 0, 'shadow_rise': 0}
        self._f2_1_reset_counts[source] = self._f2_1_reset_counts.get(source, 0) + 1

    # ── EPC-state delegations (state lives in self._epc_det) ─────────────────
    @property
    def epc_active(self) -> bool: return self._epc_det.active
    @property
    def epc_hangover_count(self) -> int: return self._epc_det.hangover_count
    @property
    def _epv_gain_fast(self) -> float: return self._epc_det.epv_gain_fast
    @property
    def _epv_gain_slow(self) -> float: return self._epc_det.epv_gain_slow

    # Per-mode optimal mu defaults (tuned on fileid_0/1/2)
    _MODE_DEFAULT_MU = {
        AecMode.LMS: 0.02,
        AecMode.NLMS: 0.4,
        AecMode.FDAF: 0.3,
        AecMode.PBFDAF: 0.5,
        AecMode.PBFDKF: 0.5,
        AecMode.SUBBAND: 0.5,
    }

    def __init__(self, config: Optional[AecConfig] = None):
        self.config = config or AecConfig()

        # Apply per-mode default mu if user didn't override
        if self.config.mu == AecConfig.mu:  # still at dataclass default
            self.config.mu = self._MODE_DEFAULT_MU.get(self.config.mode, 0.3)

        # Delay estimation + reference alignment
        if self.config.enable_delay_est or self.config.fixed_delay_samples >= 0:
            max_delay_samp = int(self.config.max_delay_ms * self.config.sample_rate / 1000)
            # v3.10.0: ring buffer sized to delay_buffer_ms (default 1024 ms,
            # matching WebRTC AEC3 kRenderTransferQueueSizeFrames=1000 ms).
            # The +4096 below is legacy headroom; we keep it on top of the
            # configured buffer to absorb hop-boundary alignment.
            buffer_samp = int(self.config.delay_buffer_ms * self.config.sample_rate / 1000)
            buffer_samp = max(buffer_samp, max_delay_samp + 4096)
            if self.config.fixed_delay_samples >= 0:
                buffer_samp = max(buffer_samp, self.config.fixed_delay_samples + 4096)
                self.delay_est = None
                self._current_delay = self.config.fixed_delay_samples
            else:
                self.delay_est = DelayEstimator(
                    sample_rate=self.config.sample_rate,
                    max_delay_ms=self.config.max_delay_ms,
                    init_seconds=self.config.delay_est_init_s,
                    period_seconds=self.config.delay_est_period_s,
                    par_low_threshold=self.config.delay_par_low_threshold,
                    par_solid_threshold=self.config.delay_par_solid_threshold,
                    trace=getattr(self.config, "trace_delay_est", False),
                    fast_path_enabled=getattr(self.config,
                                              "delay_fast_path_enabled", False),
                    fast_par_threshold=getattr(self.config,
                                               "delay_fast_par_threshold", 40.0),
                )
                self._current_delay = -1  # -1 = not yet estimated
            # Reference ring buffer for delay compensation
            self._ref_ring = np.zeros(buffer_samp, dtype=np.float32)
            self._ref_ring_write = 0
            self._ref_ring_size = len(self._ref_ring)
            self._ref_ring_filled = 0  # Total samples written (for warmup)
            self._delay_active = True
        else:
            self.delay_est = None
            self._delay_active = False

        # Create adaptive filter based on mode
        if self.config.mode in _FREQ_MODES:
            if self.config.mode == AecMode.FDAF:
                # Classic FDAF: single FFT block, n_partitions=1
                # block_size = 2 × filter_length, hop = filter_length
                # Faster than time-domain LMS (O(N log N) vs O(N²)) with same result.
                # Limitation: large hop → slow update rate → motivates partitioned approach.
                desired = 2 * self.config.filter_length
                block_size = 256
                while block_size < desired:
                    block_size *= 2
                n_partitions = 1
                self._internal_hop = block_size // 2
            else:
                # Partitioned block (PBFDAF/PBFDKF/SUBBAND)
                # block_size = 2 × hop (proper overlap-save, 50% TD constraint)
                # FFT size determined inside PBFDAF as next_pow2(block_size)
                hop_size = self.config.hop_size
                block_size = 2 * hop_size
                n_partitions = max(1, (self.config.filter_length + hop_size - 1) // hop_size)
                self._internal_hop = hop_size

            # FilterClass: mode determines adaptation algorithm
            if self.config.mode in (AecMode.PBFDKF, AecMode.SUBBAND):
                FilterClass = PBFDKF
            elif self.config.mode == AecMode.PBFDAF:
                FilterClass = PBFDAF
            else:  # FDAF — use_kalman flag for flexibility
                FilterClass = PBFDKF if self.config.use_kalman else PBFDAF
            self.filter = FilterClass(
                block_size=block_size,
                n_partitions=n_partitions,
                mu=self.config.mu,
                delta=self.config.delta,
                hop_size=self._internal_hop
            )
            self.filter.enable_td_constraint = self.config.enable_td_constraint
            self._hop_size = self.config.hop_size
            self._n_partitions = n_partitions

            # PBFDKF: apply config Q_high/Q_low
            if isinstance(self.filter, PBFDKF):
                self.filter.Q_high[:] = self.config.kalman_q_high
                self.filter.Q_low[:]  = self.config.kalman_q_low
                self.filter.Q[:] = self.config.kalman_q_high
                # P53 Step 0: enable innovation-audit hook from config.
                self.filter._enable_p53_trace = bool(
                    getattr(self.config, 'trace_p53_innovation', False))

            # FDAF buffering (when internal_hop > external hop)
            if self.config.mode == AecMode.FDAF and self._internal_hop > self._hop_size:
                # Buffer large enough for accumulation (internal_hop + one extra external hop)
                buf_size = self._internal_hop + self._hop_size
                self._freq_near_queue = np.zeros(buf_size, dtype=np.float32)
                self._freq_far_queue = np.zeros(buf_size, dtype=np.float32)
                self._freq_out_buf = np.zeros(buf_size, dtype=np.float32)
                self._freq_out_valid = 0  # valid output samples remaining
                self._freq_out_read = 0
                self._freq_queue_write = 0
                # DTD independent buffer: FL-point FFT with hop=FL/2
                # Decouples coherence DTD from FDAF's larger block_size
                fl = self.config.filter_length
                self._dtd_fft_size = fl
                self._dtd_hop = fl // 2
                self._dtd_err_buf = np.zeros(fl, dtype=np.float32)
                self._dtd_far_buf = np.zeros(fl, dtype=np.float32)
                self._dtd_acc_err = np.zeros(fl // 2, dtype=np.float32)
                self._dtd_acc_far = np.zeros(fl // 2, dtype=np.float32)
                self._dtd_acc_pos = 0
            else:
                self._freq_near_queue = None
                self._dtd_fft_size = 0
        elif self.config.mode == AecMode.LMS:
            # LMS: Time-domain, no normalization
            self.filter = NlmsFilter(
                filter_length=self.config.filter_length,
                mu=self.config.mu,
                delta=self.config.delta,
                normalize=False
            )
            self.filter.clear_history = self.config.clear_filter_history
            self._hop_size = self.config.hop_size
            self._internal_hop = self.config.hop_size
            self._n_partitions = 0
            self._freq_near_queue = None
        else:
            # TIME: Time-domain NLMS
            self.filter = NlmsFilter(
                filter_length=self.config.filter_length,
                mu=self.config.mu,
                delta=self.config.delta,
                normalize=True
            )
            self.filter.clear_history = self.config.clear_filter_history
            self._hop_size = self.config.hop_size
            self._internal_hop = self.config.hop_size
            self._n_partitions = 0
            self._freq_near_queue = None

        # DTD: frequency-domain modes only (divergence + coherence dual detector)
        # LMS/NLMS have no effective DTD — all methods (Geigel, NCC, coherence,
        # VSS-NLMS) either don't work for AEC or cause vicious cycles with slow
        # convergence. Output Limiter provides the safety net instead.
        if self.config.enable_dtd and self.config.mode in _FREQ_MODES:
            # Warmup: 50 DTD invocations before coherence starts.
            # FDAFDTD runs every dtd_hop/hop_size external frames,
            # so 50 DTD invocations = 50 * dtd_hop/hop_size external frames.
            warmup = 50
            self.dtd_divergence = DtdEstimator(
                mode='divergence',
                divergence_factor=self.config.dtd_divergence_factor,
                attack=self.config.dtd_confidence_attack,
                release=self.config.dtd_confidence_release,
                warmup_frames=warmup,
            )
            # FDAF: FL-point FFT (matches filter length, hop=FL/2)
            # PBFDAF/PBFDKF: use filter's own spectra (block_size from filter)
            if self._dtd_fft_size > 0:
                dtd_block_size = self._dtd_fft_size
            else:
                dtd_block_size = self.filter.block_size
            coh_n_freqs = dtd_block_size // 2 + 1
            self.dtd_coherence = DtdEstimator(
                mode='coherence',
                n_freqs=coh_n_freqs,
                coh_alpha=self.config.dtd_coh_alpha,
                coh_high=self.config.dtd_coh_high,
                coh_low=self.config.dtd_coh_low,
                coh_energy_floor=self.config.dtd_coh_energy_floor,
                coh_abs_floor=self.config.dtd_coh_abs_floor,
                hangover_max=self.config.dtd_coh_hangover,
                attack=self.config.dtd_confidence_attack,
                release=self.config.dtd_coh_release,
                warmup_frames=warmup,
                sample_rate=self.config.sample_rate,
                block_size=dtd_block_size,
            )
        else:
            self.dtd_divergence = None
            self.dtd_coherence = None

        # RES (only for frequency-domain modes)
        if self.config.enable_res and self.config.mode in _FREQ_MODES:
            if self.config.use_res_refactored:
                from res_refactored.res_filter_refactored import ResFilterRefactored
                _ResCls = ResFilterRefactored
            else:
                _ResCls = ResFilter
            self.res = _ResCls(
                block_size=self.filter.fft_size,
                n_freqs=self.filter.n_freqs,
                g_min_db=self.config.res_g_min_db,
                over_sub=self.config.res_over_sub,
                alpha=self.config.res_alpha,
                enable_cng=self.config.enable_cng,
                max_drop_db_per_frame=self.config.res_max_drop_db_per_frame,
                max_rise_db_per_frame=self.config.res_max_rise_db_per_frame,
                enable_spectral_floor=self.config.res_spectral_floor,
                spectral_floor_db=self.config.res_spectral_floor_db,
                ne_protect_db=self.config.res_ne_protect_db,
                frame_size=self.config.frame_size,
                hop_size=self.config.hop_size,
                echo_method=self.config.res_echo_method,
                gain_type=self.config.res_gain_type,
                enable_reverb=self.config.res_enable_reverb,
                reverb_decay=self.config.res_reverb_decay,
                reverb_gain=self.config.res_reverb_gain,
                alpha_echo_psd=self.config.res_alpha_echo_psd,
                alpha_error_psd=self.config.res_alpha_error_psd,
                enr_scale=self.config.res_enr_scale,
                startup_dt_min_ne_scale=self.config.startup_dt_min_ne_scale,
                startup_dt_gain_floor=self.config.startup_dt_gain_floor,
                startup_dt_noise_floor_scale=self.config.startup_dt_noise_floor_scale,
                sample_rate=self.config.sample_rate,
                capture_stages=self.config.capture_stages,
                plan_a_kernel_tight=self.config.plan_a_kernel_tight,
                plan_a_hf_cap_2k=self.config.plan_a_hf_cap_2k,
                plan_a_stat_mask_7k=self.config.plan_a_stat_mask_7k,
                hf_cap_conditional=self.config.hf_cap_conditional,
                hf_cap_metric_threshold=self.config.hf_cap_metric_threshold,
                plan_b_dt_per_bin_gamma=self.config.plan_b_dt_per_bin_gamma,
                use_mic_excess_evidence=self.config.use_mic_excess_evidence,
                consume_filter_state=self.config.res_consume_filter_state,
                unified_gain_floor=self.config.res_unified_gain_floor,
                state_driven_epc_dt_cap=self.config.res_state_driven_epc_dt_cap,
                dt_per_bin_unified=self.config.res_dt_per_bin_unified,
                noise_floor_refined=self.config.res_noise_floor_refined,
                cap2_fs_loosen=self.config.res_cap2_fs_loosen,
                per_band_enr=self.config.res_per_band_enr,
                enr_t_ne_per_band=self.config.enr_t_ne_per_band,
                enr_s_ne_per_band=self.config.enr_s_ne_per_band,
                dt_ne_compression_fix=self.config.dt_ne_compression_fix,
                dt_ne_state_scale=self.config.dt_ne_state_scale,
                dt_ne_per_bin_thresh=self.config.dt_ne_per_bin_thresh,
                dt_ne_per_bin_scale=self.config.dt_ne_per_bin_scale,
            )
        else:
            self.res = None

        # Shadow filter (dual-filter, frequency-domain modes only)
        # Can be used alone (≈ WebRTC/SpeexDSP) or with DTD (dual protection)
        self.shadow_filter = None
        self.main_err_smooth = 0.0
        self.shadow_err_smooth = 0.0
        if (self.config.enable_shadow and
                self.config.mode in _FREQ_MODES
                and hasattr(self.filter, 'W')):
            shadow_mu = self.config.mu * self.config.shadow_mu_ratio
            self.shadow_filter = FilterClass(
                block_size=self.filter.block_size,
                n_partitions=self.filter.n_partitions,
                mu=shadow_mu,
                delta=self.config.delta,
                hop_size=self.filter.hop_size
            )
            self.shadow_filter.enable_td_constraint = self.config.enable_td_constraint
            # PBFDKF shadow: higher Q via ratio for faster tracking
            if isinstance(self.shadow_filter, PBFDKF):
                self.shadow_filter.Q_high = self.filter.Q_high * self.config.shadow_q_ratio
                self.shadow_filter.Q_low  = self.filter.Q_low  * self.config.shadow_q_ratio
                self.shadow_filter.Q      = self.shadow_filter.Q_high.copy()

        # Echo path change detector (owns active/hangover/EPV-EMAs/prev_total_err)
        self._epc_det = EchoPathChangeDetector(self.config)

        # #4: Confidence memory decay
        self.prev_dtd_conf = 0.0
        self._dtd_conf_holdoff = 0  # F2.5: frames remaining in hold phase

        # Filter convergence + divergence-indicator (extracted to FilterConvergenceAnalyzer).
        # Backward-compat reads via @property below.
        self._convergence = FilterConvergenceAnalyzer()
        # EPC render-forced countdown (Change D)
        self._epc_render_forced_remaining = 0
        # Dynamic ERL estimate for render-based echo (B4)
        self._erl_estimate = 0.1  # initial -20dB, conservative
        # v3.14 Arc-P P.S2: adaptive per-band ERL EMA [LF, MF, HF].
        # Only updated and consumed when f3_1_per_band_erl_adaptive=True.
        # Initialised to the same 0.1 as scalar _erl_estimate so the first
        # few frames before any update gate fires produce reasonable values.
        self._per_band_erl = np.array([0.1, 0.1, 0.1], dtype=np.float64)
        # Double-talk analyzer (owns _dt_from_energy / _dt_from_shadow / _shadow_advantage)
        self._dt_analyzer = DoubleTalkAnalyzer(self.config)

        # v3.10.0 — filter plateau detector (one-shot recovery for
        # DT-from-frame-0 cases where main filter learned NE leak in the
        # first ~100 frames and is now stuck below convergence threshold).
        self._plateau_detector = FilterPlateauDetector()

        # P3e — DT advisory gate state. Hold counter is in samples so we
        # can convert dt_advisory_hold_ms once. Diag fields exposed via _diag.
        self._dt_advisory_hold_remaining = 0  # frames; decremented per process()
        _hop = self._hop_size if hasattr(self, '_hop_size') else (
            self.config.hop_size if self.config.hop_size > 0 else self.config.frame_size // 2)
        self._dt_advisory_hold_frames = max(
            1, int(self.config.dt_advisory_hold_ms * self.config.sample_rate / 1000.0 / max(1, _hop)))

        # P3f — Mini AecState trace (no behaviour change). post_reset_age_frames
        # is incremented per process() and zeroed by _reset_filter_derived_state.
        # erle_inst ring buffer kept for slope estimation (~500 ms window at
        # hop=160 / 16 kHz = 50 frames).
        self._post_reset_age_frames = 0
        _slope_n = max(2, int(0.5 * self.config.sample_rate / max(1, _hop)))
        self._erle_slope_buf = deque(maxlen=_slope_n)
        self._p3f_refined_latched = False  # latches true once refined_usable seen
        self._p3f_main_err_baseline = 0.0  # EMA baseline for jump detection
        self._p3f_diverged_streak = 0      # consecutive frames over diverged TH
        # B6 — previous-frame filter_state cache for state-aware shadow_mu.
        # Filter state classifier runs at end of process(); shadow µ schedule
        # at start needs the previous frame's value.
        self._prev_filter_state = 'idle'
        # v3.13 E4.S3 — SubtractiveNLP detector (audit-only).
        # Per docs/v3_13_e4_s2_design_lock.md. Outputs nl_confidence per
        # hop into self._diag; does NOT modify output. Pure observer.
        self.nl_detector: Optional[SubtractiveNLP] = None
        if getattr(self.config, 'e4_nlp_enabled', False):
            self.nl_detector = SubtractiveNLP(
                sample_rate=self.config.sample_rate,
                hop_size=self.hop_size,
                window_ms=getattr(self.config, 'e4_nlp_window_ms', 32.0),
                pitch_threshold=getattr(self.config, 'e4_nlp_pitch_threshold', 0.45),
                continuity_frames=getattr(self.config, 'e4_nlp_continuity_frames', 3),
                min_residual_rms=getattr(self.config, 'e4_nlp_min_residual_rms', 0.05),
                cancellation_ratio_threshold=getattr(
                    self.config, 'e4_nlp_cancel_ratio_threshold', 2.0),
                cancellation_ema_alpha=getattr(
                    self.config, 'e4_nlp_cancel_ema_alpha', 0.99),
            )
        # F-E1 — far_active hysteresis state (fast attack / slow release).
        # Once far crosses 1e-4 it stays "active" until 5 consecutive frames
        # below 3e-5. Stabilises ERL update gating across brief power dips.
        self._f_e1_far_active = False
        self._f_e1_far_release_count = 0
        # F-DelayTrack — recent delay-estimate history for variance check.
        # Bounded deque; appended only when delay_est emits a valid estimate.
        self._delay_history = deque(maxlen=8)
        # F-E3 — consecutive-EPC tracking. Counters reset on respective fires;
        # large initial value means "never fired".
        self._frames_since_last_epc = 10**9
        self._frames_since_last_f_e3_w_reset = 10**9
        # F-E5 — saturation hysteresis: track previous frame's sat level so
        # we can fast-attack reset _error_psd on the sat → clean transition.
        self._f_e5_prev_sat_level = 0.0
        # F2.2 EMA tracker — always maintained (cheap), only consumed by P3h
        # reset gate when `use_diverged_streak_ema` is True.
        self._p3f_diverged_streak_ema = 0.0
        # P3h — sustained-diverged reset cooldown. Decremented per frame;
        # reset action gated on cooldown == 0.
        self._p3h_reset_cooldown_remaining = 0
        self._p3h_reset_count = 0           # diagnostic: reset fires this run

        # S-orth.A — shadow decoupled state (initialised here regardless of flag;
        # only *used* when shadow_state_decoupled=True so flag-OFF is byte-equal).
        # These are initialised to the same values as PBFDKF.__init__ so that
        # after enabling the flag the first frame starts from the same state.
        _n_freqs = (self.filter.n_freqs
                    if hasattr(self.filter, 'n_freqs') else 1)
        self._shadow_error_psd = np.ones(_n_freqs, dtype=np.float32) * 1e-2
        self._shadow_R = np.ones(_n_freqs, dtype=np.float32) * 1e-2
        self._shadow_mu_holdoff = 0   # independent of main's _simple_mu_holdoff

        # Windowed decaying ERLE accumulator for erle_factor (TC ≈ 10s)
        self._erle_window_near = 1e-10
        self._erle_window_err = 1e-10
        self._erle_factor_prev = 0.0  # Previous frame's erle_factor for shadow DTD weight

        # Smoothed inst ERLE for dt_indicator correction (~3 frame / 30ms)
        self._inst_erle_smooth = 1.0

        # Per-bin mu_scale (updated from RES echo_psd/error_psd each frame)
        self._per_bin_mu_scale = None  # None = use scalar fallback

        # P1 Phase 1 trace state: ring buffer of high-band mic power for
        # modulation CV^2 metric. 32 frames @ hop=160 @ 16kHz ≈ 320 ms.
        self._hb_mic_pwr_ring = np.zeros(32, dtype=np.float32)
        self._hb_mic_pwr_idx = 0
        self._hb_mic_pwr_n = 0  # filled-count (≤ 32)

        # Diagnostic tracking (per-frame, latest values)
        self._diag = {
            'erle_inst': 0.0, 'mu_scale': 1.0, 'far_activity': 0.0,
            'res_gain_mean': 1.0, 'res_gain_min': 1.0, 'effective_g_min': 1.0,
            'converged': False, 'erle_factor': 0.0,
            'echo_psd_mean': 0.0, 'error_psd_mean': 0.0, 'divergence': 0.0,
            'using_render_based': False,
            'shadow_advantage': 1.0,
            'dt_from_energy': 0.0,
            'dt_from_shadow': 0.0,
            'erl_estimate': 0.1,
            'epc_active': False,
            'saturation_level': 0.0,
            'erle_windowed': 0.0,
            # DT / filter debug fields
            'dt_indicator': 0.0,        # final DT confidence fed to RES
            'main_err_smooth': 0.0,     # main filter error EMA
            'shadow_err_smooth': 0.0,   # shadow filter error EMA
            'main_paused': False,       # True = main filter weights frozen this frame
            'epv_gain_ratio': 1.0,      # EPV fast/slow gain ratio (!=1 → gain change)
            'dt_residual_scale': 1.0,   # scaling on echo_spec passed to RES
            'filter_w_norm': 0.0,       # main filter L2 weight norm
            'shadow_w_norm': 0.0,       # shadow filter L2 weight norm
            'copy_err_baseline': 1e-6,  # copy gate error baseline
        }

        # Output limiter: smoothed gain to avoid frame-boundary clicking
        self._limiter_gain = 1.0

        # High-pass filter (DC blocker + low-freq removal)
        if self.config.enable_highpass:
            self._hp_mic = HighPassFilter(self.config.highpass_cutoff_hz, self.config.sample_rate)
            self._hp_ref = HighPassFilter(self.config.highpass_cutoff_hz, self.config.sample_rate)
        else:
            self._hp_mic = None
            self._hp_ref = None

        # Saturation detector (non-linear echo handling)
        if self.config.enable_saturation_detect:
            self._sat_detector_ref = SaturationDetector(self.config.saturation_threshold)
            self._sat_detector_mic = SaturationDetector(self.config.saturation_threshold)
        else:
            self._sat_detector_ref = None
            self._sat_detector_mic = None
        self._saturation_level = 0.0

        # Far-end activity + stationarity detector (extracted from inline EMA logic)
        self._render_activity = RenderActivityDetector()
        self._stat_far_hangover = 0
        self._inst_erle_smooth = 1.0
        self._wn_err_baseline = 1e-8
        self._stat_dt_hangover = 0  # Stationary DT hold-off counter (frames)

        # Simple variable mu (for non-DTD modes, inspired by Valin 2007 RER)
        self._simple_mu_ratio = 1.0
        self._simple_mu_holdoff = 0  # holdoff counter: blocks release for N frames
        self._warmup_frames = self.config.warmup_frames
        self._warmup_far_active = False  # only consume warmup when far-end is active

        # Shadow divergence detection state (WebRTC-style: pause + Q boost, no output switch)
        self.shadow_frame_count = 0
        self._regime_handler = PathChangeRegimeHandler(
            self.config,
            gate_mode=getattr(self.config, 'regime_gate_mode', 'energy'),
        )
        self._last_raw_output: Optional[np.ndarray] = None   # raw filter output before RES (diagnostic)
        # EchoPathVariability EMAs moved into EchoPathChangeDetector (self._epc_det)
        # AecState aggregator: WebRTC-style read-only seam over the 5 detectors.
        # Phase B consumes this to decide linear vs nonlinear residual-echo path.
        self._aec_state = AecState(
            render_activity=self._render_activity,
            convergence=self._convergence,
            dt_analyzer=self._dt_analyzer,
            epc_det=self._epc_det,
            regime_handler=self._regime_handler,
            dtd_coherence_getter=lambda: (
                self.dtd_coherence.confidence if self.dtd_coherence else 0.0),
        )
        self._far_power_ema = 0.0           # TC≈50ms for GetStats()
        self._mic_power_ema = 0.0
        self._frame_count = 0               # frames since reset()

        # P52 A.0R.2: per-frame regime handler trace rows. Only appended to
        # when self.config.trace_p52_regime_handler is True (default False
        # → list stays empty → zero memory overhead).
        self._regime_trace_rows = []

        # ERLE (raw = filter-only, final = post-RES)
        self.near_power = 0.0
        self.error_power = 0.0  # backward compat alias for raw
        self.raw_error_power = 0.0
        self.final_error_power = 0.0
        self.alpha = 0.95
        # Cumulative ERLE (full-segment average)
        self.near_power_sum = 0.0
        self.error_power_sum = 0.0  # backward compat alias for raw
        self.raw_error_power_sum = 0.0
        self.final_error_power_sum = 0.0
        # _conv_counter moved to FilterConvergenceAnalyzer (self._convergence)

        # DTD confidence history (one entry per process() call)
        self.confidence_history = deque(maxlen=1000)

    def reset(self):
        self.filter.reset()
        if self.shadow_filter:
            self.shadow_filter.reset()
            self.main_err_smooth = 0.0
            self.shadow_err_smooth = 0.0
        if self._delay_active:
            if self.delay_est is not None:
                self.delay_est.reset()
                self._current_delay = -1
            else:
                self._current_delay = self.config.fixed_delay_samples
            self._ref_ring.fill(0)
            self._ref_ring_write = 0
            self._ref_ring_filled = 0
        self._epc_det.reset()
        self.prev_dtd_conf = 0.0
        self._dtd_conf_holdoff = 0
        self._convergence.reset()
        # v3.10.0: clear plateau-detector counters on AEC reset
        if hasattr(self, '_plateau_detector'):
            self._plateau_detector.reset()
        # P3e: clear DT advisory hold
        self._dt_advisory_hold_remaining = 0
        self._erle_window_near = 1e-10
        self._erle_window_err = 1e-10
        self._erle_factor_prev = 0.0
        self._inst_erle_smooth = 1.0
        self._simple_mu_ratio = 1.0
        self._simple_mu_holdoff = 0
        # S-orth.A: reset shadow decoupled state on full AEC.reset()
        if hasattr(self, '_shadow_error_psd'):
            self._shadow_error_psd.fill(1e-2)
        if hasattr(self, '_shadow_R'):
            self._shadow_R.fill(1e-2)
        if hasattr(self, '_shadow_mu_holdoff'):
            self._shadow_mu_holdoff = 0
        self._warmup_frames = self.config.warmup_frames
        self._warmup_far_active = False
        # v3.8.1: clear lazy-getattr diagnostic counters so cross-case batch
        # eval doesn't leak prior-case state into next-case stats interpretation.
        # Diagnostics-only — does not affect audio output.
        self._far_active_blocks = 0
        self._dt_from_zero_count = 0
        self._diag = {
            'erle_inst': 0.0, 'mu_scale': 1.0, 'far_activity': 0.0,
            'res_gain_mean': 1.0, 'res_gain_min': 1.0, 'effective_g_min': 1.0,
            'converged': False, 'erle_factor': 0.0,
            'echo_psd_mean': 0.0, 'error_psd_mean': 0.0, 'divergence': 0.0,
            'using_render_based': False,
            'shadow_advantage': 1.0,
            'dt_from_energy': 0.0,
            'dt_from_shadow': 0.0,
            'dt_from_coherence': 0.0,
            'far_power': 0.0,
            'mic_power': 0.0,
            'erl_estimate': 0.1,
            'epc_active': False,
            'saturation_level': 0.0,
            'erle_windowed': 0.0,
            'dt_indicator': 0.0,
            'main_err_smooth': 0.0,
            'shadow_err_smooth': 0.0,
            'main_paused': False,
            'epv_gain_ratio': 1.0,
            'dt_residual_scale': 1.0,
            'filter_w_norm': 0.0,
            'shadow_w_norm': 0.0,
            'copy_err_baseline': 1e-6,
        }
        self.shadow_frame_count = 0
        self._regime_handler.reset()
        # _epc_det reset above already cleared its EPV EMAs and prev_total_err
        self._far_power_ema = 0.0
        self._mic_power_ema = 0.0
        self._frame_count = 0
        if self.dtd_divergence:
            self.dtd_divergence.reset()
        if self.dtd_coherence:
            self.dtd_coherence.reset()
        if self.res:
            self.res.reset()
        if self._freq_near_queue is not None:
            self._freq_near_queue.fill(0)
            self._freq_far_queue.fill(0)
            self._freq_out_buf.fill(0)
            self._freq_queue_write = 0
            self._freq_out_valid = 0
            self._freq_out_read = 0
        self.near_power = 0.0
        self.error_power = 0.0
        self.raw_error_power = 0.0
        self.final_error_power = 0.0
        self.near_power_sum = 0.0
        self.error_power_sum = 0.0
        self.raw_error_power_sum = 0.0
        self.final_error_power_sum = 0.0
        # v3.10.3 — clear cross-case lazy state
        if hasattr(self, '_pending_delay'):
            del self._pending_delay
        for _attr in ('_round3_div_counts', '_round3_last_div_source',
                      '_r7_prev_delay', '_r7_prev_div_counts',
                      '_dominant_nearend_hold'):
            if hasattr(self, _attr):
                delattr(self, _attr)
        # _conv_counter is owned by self._convergence (reset above)
        if self._hp_mic is not None:
            self._hp_mic.reset()
            self._hp_ref.reset()
        if self._sat_detector_ref is not None:
            self._sat_detector_ref.reset()
            self._sat_detector_mic.reset()
        self._saturation_level = 0.0
        self._render_activity.reset()
        self._stat_far_hangover = 0
        self._inst_erle_smooth = 1.0
        self._wn_err_baseline = 1e-8
        self._stat_dt_hangover = 0  # Stationary DT hold-off counter (frames)
        self._limiter_gain = 1.0
        self._per_bin_mu_scale = None
        if self._dtd_fft_size > 0:
            self._dtd_acc_pos = 0
            self._dtd_acc_err.fill(0)
            self._dtd_acc_far.fill(0)
            self._dtd_err_buf.fill(0)
            self._dtd_far_buf.fill(0)
        # Reset DT signals (now owned by DoubleTalkAnalyzer)
        self._dt_analyzer.reset()
        self._epc_render_forced_remaining = 0
        self._erl_estimate = 0.1
        # v3.14 Arc-P P.S2: reset per-band ERL EMA to initial conservative value.
        self._per_band_erl[:] = 0.1

    def _reset_filter_derived_state(self, reason: str = 'plateau',
                                     preserve_render_ema: bool = True) -> None:
        """v3.10.2 — clear all filter-output-derived state. Generic helper
        called by both plateau recovery (`reason='plateau'`) and first delay
        acquisition (`reason='delay_first'`). Earlier versions of these
        recovery paths reset only filter/shadow/res, leaving the freshly
        reset filter to learn against poisoned downstream state
        (main_err_smooth / DTD / EPC hangover / ERLE windows all still
        carrying values from the bad taps).

        Both recovery paths share the same shape: filter taps were trained
        against incorrect input (misaligned ref for delay_first; NE leak
        for plateau), and all derived state from those taps is now wrong.

        Differs from full AEC.reset() in that this preserves time-axis +
        input-side context:

          PRESERVED (input-side / temporal context):
            • _frame_count                 — elapsed time
            • delay_est + _current_delay   — delay alignment lives upstream
            • _ref_ring (delay buffer)     — far history is valid
            • _plateau_detector            — its own attempts counter
            • _far_power_ema / _mic_power_ema  — input-side
            • _hp_mic / _hp_ref            — input-side HPF
            • _sat_detector_*              — input-side
            • _render_activity             — input-side
            • RES long-window far-PSD EMA — input-side (when
              preserve_render_ema=True; default). The EMA is updated every
              far-active frame regardless of mode, so its accumulated
              long-term render spectrum is independent of the bad taps.
              Discarding it forces the freshly reset filter through 100
              frames of pre-warmup-fallback all over again.

          CLEARED (filter-output-derived; would otherwise re-poison the
          freshly reset filter):
            • filter / shadow_filter taps via .reset()
            • _convergence + _epc_det
            • main_err_smooth / shadow_err_smooth
            • _dt_analyzer (energy / shadow DT histories)
            • _erle_window_* / _inst_erle_smooth / _erle_factor_prev
            • _simple_mu_ratio / _simple_mu_holdoff / _per_bin_mu_scale
            • _epc_render_forced_remaining / _erl_estimate
            • prev_dtd_conf
            • DTD divergence + coherence smoothed PSDs
            • RES post-filter state (gain_smooth / echo_psd / noise_psd /
              gates). Long-window far-PSD EMA optionally preserved.
            • shadow_frame_count + _regime_handler
            • Diagnostic _diag dict (would otherwise show stale stats)
        """
        # Filter taps + shadow
        self.filter.reset()
        if self.shadow_filter is not None:
            self.shadow_filter.reset()

        # Filter convergence + EPC + divergence DTD + coherence DTD
        self._convergence.reset()
        self._epc_det.reset()
        if self.dtd_divergence is not None:
            self.dtd_divergence.reset()
        if self.dtd_coherence is not None:
            self.dtd_coherence.reset()

        # Filter-output power / smoother / err quantities
        self.main_err_smooth = 0.0
        self.shadow_err_smooth = 0.0
        self.error_power = 0.0
        self.raw_error_power = 0.0
        self.final_error_power = 0.0
        self.error_power_sum = 0.0
        self.raw_error_power_sum = 0.0
        self.final_error_power_sum = 0.0
        # v3.10.3 — near_power EMA must be reset alongside error_power, otherwise
        # get_erle_inst() = near_power / error_power transiently spikes (stale
        # mic EMA / fresh tiny error) and could mis-trigger early convergence.
        # near_power is sample-loop EMA (alpha=0.999) so it recovers in ~10 frames.
        self.near_power = 0.0
        self.near_power_sum = 0.0

        # ERLE accumulators
        self._erle_window_near = 1e-10
        self._erle_window_err = 1e-10
        self._erle_factor_prev = 0.0
        self._inst_erle_smooth = 1.0

        # Mu-scale state (depends on DTD which depends on filter)
        self._simple_mu_ratio = 1.0
        self._simple_mu_holdoff = 0
        self._per_bin_mu_scale = None
        self.prev_dtd_conf = 0.0
        self._dtd_conf_holdoff = 0

        # ERL + EPC-render forced + DT analyzer (all filter-output-derived)
        self._erl_estimate = 0.1
        # v3.14 Arc-P P.S2: per-band ERL is filter-output-derived (echo_spec /
        # far_spec from PBFDKF), so reset it together with scalar _erl_estimate.
        self._per_band_erl[:] = 0.1
        self._epc_render_forced_remaining = 0
        self._dt_analyzer.reset()
        self._stat_dt_hangover = 0
        self._stat_far_hangover = 0

        # Coherence DTD's accumulated err/far PSDs (live in _dtd_acc_*)
        if self._dtd_fft_size > 0:
            self._dtd_acc_pos = 0
            self._dtd_acc_err.fill(0)
            self._dtd_acc_far.fill(0)
            self._dtd_err_buf.fill(0)
            self._dtd_far_buf.fill(0)

        # Shadow copy controller
        self.shadow_frame_count = 0
        self._regime_handler.reset()

        # P3f trace state — zero so post_reset_age_ms restarts from 0 and
        # refined latch / err baseline don't carry pre-reset values forward.
        self._post_reset_age_frames = 0
        self._erle_slope_buf.clear()
        self._p3f_refined_latched = False
        self._p3f_main_err_baseline = 0.0
        self._p3f_diverged_streak = 0
        self._p3f_diverged_streak_ema = 0.0
        self._prev_filter_state = 'idle'
        self._f_e1_far_active = False
        self._f_e1_far_release_count = 0
        self._delay_history.clear()
        self._frames_since_last_epc = 10**9
        self._frames_since_last_f_e3_w_reset = 10**9
        self._f_e5_prev_sat_level = 0.0

        # RES — clears its echo_psd / error_psd / noise_psd / gain_smooth /
        # render state. Long-window far-PSD EMA is preserved when
        # preserve_render_ema=True (input-side context, see docstring).
        if self.res is not None:
            self.res.reset(preserve_long_window_ema=preserve_render_ema)

        # Diagnostic dict — would otherwise show stale ERLE / DT signals
        for k, v in (('erle_inst', 0.0), ('mu_scale', 1.0), ('far_activity', 0.0),
                     ('res_gain_mean', 1.0), ('res_gain_min', 1.0),
                     ('effective_g_min', 1.0), ('converged', False),
                     ('erle_factor', 0.0), ('echo_psd_mean', 0.0),
                     ('error_psd_mean', 0.0), ('divergence', 0.0),
                     ('using_render_based', False), ('shadow_advantage', 1.0),
                     ('dt_from_energy', 0.0), ('dt_from_shadow', 0.0),
                     ('dt_from_coherence', 0.0), ('erl_estimate', 0.1),
                     ('epc_active', False), ('erle_windowed', 0.0),
                     ('dt_indicator', 0.0), ('main_err_smooth', 0.0),
                     ('shadow_err_smooth', 0.0), ('main_paused', False),
                     ('epv_gain_ratio', 1.0), ('dt_residual_scale', 1.0),
                     ('filter_w_norm', 0.0), ('shadow_w_norm', 0.0),
                     ('copy_err_baseline', 1e-6)):
            self._diag[k] = v

        # S-orth.A: reset shadow decoupled state on derived-state reset
        # (same as filter taps reset — shadow's observation history restarts).
        if hasattr(self, '_shadow_error_psd'):
            self._shadow_error_psd.fill(1e-2)
        if hasattr(self, '_shadow_R'):
            self._shadow_R.fill(1e-2)
        if hasattr(self, '_shadow_mu_holdoff'):
            self._shadow_mu_holdoff = 0

        # Re-arm warmup so the second-pass training starts with high mu.
        # Boost Q on both filters (high-Q convergence mode).
        for filt in [self.filter, self.shadow_filter]:
            if filt is not None and hasattr(filt, 'Q'):
                if hasattr(filt, 'Q_high'):
                    filt.Q = filt.Q_high.copy()
                if hasattr(filt, '_p_max_override'):
                    filt._p_max_override = 1.0
                    filt._p_max_override_frames = 30
        self._warmup_frames = max(self._warmup_frames,
                                   self.config.warmup_frames // 2)
        self._warmup_far_active = False

        # v3.10.3 — clear cross-recovery state that would otherwise mis-fire.
        # _pending_delay: stale pending shift could pair with a later rogue
        #   estimate and trigger a spurious force_delay (audio bug).
        # NOTE: _round3_div_counts / _round3_last_div_source /
        # _dominant_nearend_hold are session-cumulative diagnostic counters
        # and MUST survive recovery — they are only cleared in full AEC.reset().
        if hasattr(self, '_pending_delay'):
            del self._pending_delay
        if hasattr(self, '_pending_delay_ttl'):
            del self._pending_delay_ttl

    @property
    def hop_size(self) -> int:
        return self._hop_size

    def _compute_mu_scale(self) -> float:
        """Convert combined DTD confidence to mu_scale [mu_min_ratio, 1.0].

        v3.10.4 — when fallback is active, alignment is unreliable so we must
        not adapt against absent/bad ref. Return 0 to freeze taps; RES rides
        its existing render-based path because filter never converges.

        #3: Coherence is primary; divergence is fallback only when coherence inactive.
        #4: Confidence has memory decay to avoid sudden drops.
        EPC: mu_scale floor during echo path change.

        v3.10.0 — delay-confidence ceiling:
          When delay-est confidence is low (PAR < par_low_threshold), the
          filter is learning against a misaligned reference — driving mu
          high will encode garbage. Cap mu_scale at a delay-confidence-
          dependent ceiling: low confidence → mu_scale ≤ 0.5; full
          confidence → no cap (1.0). This mirrors WebRTC AEC3's behavior:
          AEC3 stays conservative until matched-filter delay is solid.
        """
        conf_div = self.dtd_divergence.confidence if self.dtd_divergence else 0.0
        conf_coh = self.dtd_coherence.confidence if self.dtd_coherence else 0.0

        # #3: Coherence primary, divergence fallback
        if conf_coh > 0.1:
            raw_conf = conf_coh
        else:
            raw_conf = max(conf_div, conf_coh)

        # #4: Confidence memory decay (avoid sudden drops)
        # F2.5: two-stage hangover — attack fast (1 frame), hold 10 frames, then ×0.9 decay.
        if self.config.dtd_conf_two_stage:
            if raw_conf > self.prev_dtd_conf:
                conf = raw_conf
                self._dtd_conf_holdoff = 10
            elif self._dtd_conf_holdoff > 0:
                conf = self.prev_dtd_conf
                self._dtd_conf_holdoff -= 1
            else:
                conf = max(raw_conf, self.prev_dtd_conf * 0.9)
        else:
            conf = max(raw_conf, self.prev_dtd_conf * 0.9)
        self.prev_dtd_conf = conf

        if conf == 0.0:
            mu_scale = 1.0
        else:
            min_r = self.config.dtd_mu_min_ratio
            # Before convergence, allow higher mu_min so filter can still learn during DT
            if not self._filter_converged:
                min_r = max(min_r, 0.3)
            mu_scale = 1.0 - conf * (1.0 - min_r)

        # Echo path change: keep mu high so filter can adapt to new path
        if self.epc_active:
            mu_scale = max(mu_scale, self.config.epc_mu_floor)

        # v3.10.0: delay-confidence ceiling. Cap mu when delay alignment
        # is uncertain (avoid learning garbage against misaligned ref).
        # v3.10.3 (H2): skip the ceiling during a post-reset warmup window so
        # the high-Q boost armed by _reset_filter_derived_state can actually
        # take effect. Otherwise PAR fluctuating between low/solid thresholds
        # right after delay acquisition caps mu at ~0.5–0.7 and defeats the
        # warmup re-arm, slowing ERLE rebuild and risking a wasted second
        # plateau attempt.
        in_post_reset_warmup = (
            self._warmup_frames > 0
            or (self.filter is not None
                and getattr(self.filter, '_p_max_override_frames', 0) > 0)
        )
        if self.delay_est is not None and not in_post_reset_warmup:
            delay_conf = self.delay_est.confidence
            if delay_conf < 1.0:
                # 0.5 ceiling at delay_conf=0, linear interpolate to 1.0
                delay_ceiling = 0.5 + 0.5 * delay_conf
                mu_scale = min(mu_scale, delay_ceiling)

        return mu_scale

    def _get_simple_mu_scale(self, mu_min: float = None):
        """Get mu_scale from smoothed EER (per-bin array or scalar fallback)."""
        if mu_min is None:
            mu_min = self.config.shadow_mu_min
        # Warmup: fast convergence, but respect DT signals to protect near-end
        if self._warmup_frames > 0:
            # Only consume warmup when far-end is active (don't waste on silence)
            if self._warmup_far_active:
                self._warmup_frames -= 1
            if self._simple_mu_ratio < 0.2:
                # Strong DT detected: don't force high mu, protect filter
                return max(0.2, self._simple_mu_ratio)
            return min(1.0, max(0.5, self._simple_mu_ratio + 0.2))
        if not self._filter_converged:
            mu_min = max(mu_min, 0.3)   # Pre-convergence: floor 0.3, better DT protection
        else:
            mu_min = max(mu_min, 0.2)
        # Per-bin mu_scale from RES echo_psd/error_psd (set previous frame, post-RES)
        if self._per_bin_mu_scale is not None:
            return np.maximum(self._per_bin_mu_scale, mu_min)
        return mu_min + (1.0 - mu_min) * self._simple_mu_ratio

    def _update_simple_mu_ratio(self, output: np.ndarray,
                                 far: np.ndarray) -> None:
        """Update simple variable mu ratio after process (Valin 2007 RER-inspired).

        far/error ratio: DT → error >> far → ratio low → next frame mu drops.
        Asymmetric EMA: fast attack (ratio drops), slow release (ratio recovers).
        """
        error_power = np.mean(output ** 2) + 1e-10
        far_power = np.mean(far ** 2) + 1e-10

        # Don't update ratio during silence — preserve current value for warmup
        if far_power < 1e-6 and error_power < 1e-6:
            return

        # Quick reset when far-end transitions from silence to active
        if far_power > 1e-4 and self._simple_mu_ratio < 0.1:
            self._simple_mu_ratio = 0.8
            self._simple_mu_holdoff = 0
            return

        ratio = min(far_power / error_power, 1.0)
        # Echo estimate ratio: if filter is learning echo, don't suppress mu too much
        if hasattr(self.filter, 'echo_spec') and self.filter.echo_spec is not None:
            echo_est_pwr = np.sum(np.abs(self.filter.echo_spec) ** 2) + 1e-10
            near_pwr = np.sum(np.abs(getattr(self.filter, 'near_spec', np.zeros(1))) ** 2) + 1e-10
            if near_pwr > 1e-8:
                ratio_echo = np.clip(echo_est_pwr / near_pwr, 0.0, 1.0)
                ratio = max(ratio, ratio_echo * 0.5)

        # Asymmetric EMA + holdoff: fast attack, slow release with holdoff
        if ratio < self._simple_mu_ratio:
            # Attack: fast drop.
            # F2.4: only arm holdoff on fresh DT onset (holdoff==0); do not
            # reset during the holdoff window — marginal DT oscillation keeps
            # resetting holdoff to 20 so mu never releases.
            alpha = 0.3
            if not self.config.mu_holdoff_no_reset or self._simple_mu_holdoff == 0:
                self._simple_mu_holdoff = 20  # hold low for ~20 frames (~320ms)
        elif self._simple_mu_holdoff > 0:
            # Holdoff active: keep ratio low, don't release yet
            self._simple_mu_holdoff -= 1
            alpha = 0.99  # nearly frozen
        else:
            # Release: slow recovery
            alpha = 0.95
        self._simple_mu_ratio = alpha * self._simple_mu_ratio + (1 - alpha) * ratio

    def process(self, near_end: np.ndarray, far_end: np.ndarray) -> np.ndarray:
        # High-pass filter: remove DC + low-freq noise
        if self._hp_mic is not None:
            near_end = self._hp_mic.process(near_end.copy())
            far_end = self._hp_ref.process(far_end.copy())

        # Saturation detection + soft-clip reference
        if self._sat_detector_ref is not None:
            sat_ref = self._sat_detector_ref.detect(far_end)
            sat_mic = self._sat_detector_mic.detect(near_end)
            self._saturation_level = max(sat_ref, sat_mic * 0.5)
            if self.config.saturation_softclip_ref and sat_ref > 0.1:
                far_end = SaturationDetector.soft_clip(far_end.copy())
            # F-E5 / E5-1: symmetric mic soft-clip on sat_mic threshold.
            if (self.config.f_e5_enabled
                    and sat_mic > self.config.f_e5_mic_softclip_threshold):
                near_end = SaturationDetector.soft_clip(near_end.copy())
            # F-E5 / E5-3: fast-attack _error_psd reset on sat → clean
            # transition so the α=0.95 EMA does not propagate clipped
            # samples into R for ~20 frames after sat ends.
            if (self.config.f_e5_enabled
                    and self._f_e5_prev_sat_level > 0.5
                    and self._saturation_level < 0.2
                    and hasattr(self.filter, '_error_psd')):
                self.filter._error_psd.fill(1e-2)
                if (self.shadow_filter is not None
                        and hasattr(self.shadow_filter, '_error_psd')):
                    self.shadow_filter._error_psd.fill(1e-2)
            self._f_e5_prev_sat_level = float(self._saturation_level)

        # Delay estimation + reference alignment
        if self._delay_active:
            hop = len(far_end)

            # Online delay estimation (if not using fixed delay).
            # v3.10.2: tier the gate logic into TWO INDEPENDENT paths so a
            # solid-confidence shift (current_delay >= 0, is_solid, large
            # delta) is not swallowed by the first-acquisition outer gate.
            # Codex fix: previous v3.10.1 wrapped both paths under a single
            # outer `if is_solid`, so the second `elif` was unreachable when
            # is_solid was True but current_delay was already set.
            if (self.delay_est is not None
                    and self._delay_active):
                self.delay_est.accumulate(near_end, far_end)
                new_delay = self.delay_est.estimated_delay
                _delay_eligible = (new_delay >= 0
                                    and self.delay_est._n_updates >= 3)

                # Path A: first delay acquisition. Heavy action (resets filter
                # taps + downstream derived state). Demands solid PAR.
                if (_delay_eligible
                        and self._current_delay < 0
                        and self.delay_est.is_solid):
                    self._current_delay = new_delay
                    # Reset filter taps + ALL filter-output-derived state
                    # (main_err_smooth / DTD / EPC / ERLE / mu / RES). The
                    # first ~300 ms was learned against a misaligned ref,
                    # so its derived state is poisoned the same way as the
                    # plateau case. Use the shared helper to keep behavior
                    # consistent across recovery paths.
                    self._reset_filter_derived_state(reason='delay_first',
                                                     preserve_render_ema=True)
                    self._maybe_mark_diverged('delay_first')

                # v3.10.3 (H1) — age out _pending_delay so a stale pending
                # value cannot pair with a later rogue estimate hours after
                # it was set. Decrements once per estimation cycle.
                if hasattr(self, '_pending_delay_ttl'):
                    self._pending_delay_ttl -= 1
                    if self._pending_delay_ttl <= 0:
                        if hasattr(self, '_pending_delay'):
                            del self._pending_delay
                        del self._pending_delay_ttl

                # Path B: delay shift. Independent of Path A — fires only
                # when current_delay is already set and a meaningful shift
                # is detected. Medium confidence (≥ 0.5) + consecutive-
                # consistent gate prevents single-frame noise from triggering
                # a force_delay() EPC chain.
                if (_delay_eligible
                        and self._current_delay >= 0
                        and self.delay_est.confidence >= 0.5
                        and abs(new_delay - self._current_delay) > 32):
                    if (hasattr(self, '_pending_delay')
                            and abs(new_delay - self._pending_delay) < 16):
                        self._current_delay = new_delay
                        del self._pending_delay
                        if hasattr(self, '_pending_delay_ttl'):
                            del self._pending_delay_ttl
                        # v3.10.3 (M4) — filter taps were trained against the
                        # old delay alignment; treating the shift like Path A
                        # (clear filter-output-derived state) avoids ~50–100
                        # frames of poor cancellation while taps re-converge
                        # against state that no longer matches.
                        self._reset_filter_derived_state(reason='delay_shift',
                                                         preserve_render_ema=True)
                        self._epc_det.force_delay()
                        for filt in [self.filter, self.shadow_filter]:
                            if filt is not None and hasattr(filt, 'Q'):
                                filt.Q = filt.Q_high.copy()
                                filt._p_max_override = 1.0
                                filt._p_max_override_frames = 30
                        self._maybe_mark_diverged('delay_shift')
                    else:
                        self._pending_delay = new_delay
                        self._pending_delay_ttl = 3

            # Write far_end into ring buffer
            w = self._ref_ring_write
            ring_sz = self._ref_ring_size
            if w + hop <= ring_sz:
                self._ref_ring[w:w + hop] = far_end
            else:
                part1 = ring_sz - w
                self._ref_ring[w:ring_sz] = far_end[:part1]
                self._ref_ring[:hop - part1] = far_end[part1:]
            self._ref_ring_write = (w + hop) % ring_sz
            self._ref_ring_filled += hop

            # Apply delay compensation (only after enough data in ring buffer)
            if self._current_delay > 0 and self._ref_ring_filled >= self._current_delay + hop:
                d = self._current_delay
                read_pos = (self._ref_ring_write - hop - d) % ring_sz
                if read_pos + hop <= ring_sz:
                    far_end = self._ref_ring[read_pos:read_pos + hop].copy()
                else:
                    part1 = ring_sz - read_pos
                    far_end = np.concatenate([
                        self._ref_ring[read_pos:ring_sz],
                        self._ref_ring[:hop - part1]
                    ])

        # DTD: dual detector (divergence + coherence) for frequency-domain modes
        # Combined confidence = max(divergence, coherence) → mu_scale
        # Non-DTD: simple variable mu (Valin 2007 RER-inspired)

        # Far-end activity + stationarity (single source of truth for warmup gate,
        # stationary-DT detector, and EPC stationary guard).
        _render = self._render_activity.update(far_end)
        self._warmup_far_active = _render.warmup_active

        if self.config.enable_dtd:
            mu_scale = self._compute_mu_scale()
        else:
            mu_scale = self._get_simple_mu_scale()

        # P3e — DT advisory gate. Routes shadow/energy DTD evidence into
        # mu reduction even when enable_dtd=False. The 7GT P3d trace
        # showed dt_shadow median 0.51 / dt_energy median 0.24 in the
        # post-alignment back half but the composite gate driving mu
        # never fired, so the filter learnt against NE-contaminated error.
        # Hit-then-hold hysteresis avoids per-frame flicker.
        if self.config.dt_advisory_enabled:
            if self.config.dt_advisory_use_p3f_state:
                # P3f Phase 3: gate fires on filter_state == 'suspicious_dt'
                # using the previous frame's classification (computed late
                # in process() — 1-frame lag, ~10 ms at hop=160 / 16 kHz).
                # The state already encodes refined_latched + NE evidence
                # + main_err jump + shadow lead, so no additional guards
                # are needed at this site.
                _adv_hit = bool(self._diag.get('filter_state', 'idle')
                                == 'suspicious_dt')
            else:
                # V3 (legacy) convergence guard: only honour shadow/energy
                # DT evidence after the filter has converged at least once
                # and is past the post-reset warmup window. Retained for
                # A/B comparison; default off.
                _in_post_reset_warmup = (
                    self._warmup_frames > 0
                    or (self.filter is not None
                        and getattr(self.filter, '_p_max_override_frames', 0) > 0)
                )
                _adv_hit = (
                    bool(_render.is_active)
                    and bool(self._filter_once_converged)
                    and (not _in_post_reset_warmup)
                    and (float(self._dt_from_shadow) > self.config.dt_advisory_shadow_th
                         or float(self._dt_from_energy) > self.config.dt_advisory_energy_th)
                )
            if _adv_hit:
                self._dt_advisory_hold_remaining = self._dt_advisory_hold_frames
            elif self._dt_advisory_hold_remaining > 0:
                self._dt_advisory_hold_remaining -= 1
            if self._dt_advisory_hold_remaining > 0:
                _f = float(self.config.dt_advisory_mu_factor)
                if isinstance(mu_scale, np.ndarray):
                    mu_scale = mu_scale * _f
                else:
                    mu_scale = mu_scale * _f
            self._diag['dt_advisory_active'] = bool(self._dt_advisory_hold_remaining > 0)
            self._diag['dt_advisory_hit'] = bool(_adv_hit)

        # startup_dt mu floor: raise mu_scale during startup_dt so filter can converge despite DT
        # Uses previous frame's effective_dt from ResFilter (1-frame lag, acceptable)
        if self.config.startup_dt_mu_min > 0.0 and self.res is not None:
            _far_now = float(np.mean(far_end ** 2))
            _eff_dt = getattr(self.res, '_last_effective_dt', 0.0)
            if _eff_dt > 0.35 and _far_now > 1e-4 and not self._filter_once_converged:
                if isinstance(mu_scale, np.ndarray):
                    mu_scale = np.maximum(mu_scale, self.config.startup_dt_mu_min)
                else:
                    mu_scale = max(float(mu_scale), self.config.startup_dt_mu_min)

        # Mic clipping emergency: freeze filter and clamp RES output to floor.
        # Hard clipping turns mic into square waves; filter would learn garbage.
        if self._sat_detector_mic is not None:
            mic_clip = self._sat_detector_mic.saturation_level
            if mic_clip > 0.8:
                mu_scale = 0.0
                if self.res:
                    self.res.gain_smooth[:] = self.res.g_min
        # F-E5 / E5-2: extended main mu sat-gate. Match shadow's threshold
        # (saturation_safe = sat < 0.5) so main filter does not keep learning
        # on a clipped reference signal while shadow is already paused.
        if (self.config.f_e5_enabled
                and self._saturation_level > self.config.f_e5_main_mu_sat_threshold):
            if isinstance(mu_scale, np.ndarray):
                mu_scale = np.zeros_like(mu_scale)
            else:
                mu_scale = 0.0

        _res_context = None  # populated when return_res_context=True and no internal RES

        # Stationary feature consumed by EPC / RES baseline / stationary-DT detector.
        # _render was populated above; far_pwr_global, _far_active_prev, _is_stationary_far
        # are kept as locals for downstream call-sites that still read them by name.
        far_pwr_global = _render.far_pwr

        if self.config.mode in _FREQ_MODES:
            if self._freq_near_queue is not None:
                # Buffered FDAF: accumulate into queue, process when enough
                hop = self._hop_size
                ihop = self._internal_hop
                w = self._freq_queue_write
                self._freq_near_queue[w:w+hop] = near_end
                self._freq_far_queue[w:w+hop] = far_end
                self._freq_queue_write = w + hop

                if self._freq_queue_write >= ihop:
                    # Process one internal block
                    big_out = self.filter.process(
                        self._freq_near_queue[:ihop],
                        self._freq_far_queue[:ihop], mu_scale)
                    # Store output and shift leftover input
                    leftover = self._freq_queue_write - ihop
                    self._freq_out_buf[:ihop] = big_out
                    self._freq_out_valid = ihop
                    self._freq_out_read = 0
                    if leftover > 0:
                        self._freq_near_queue[:leftover] = self._freq_near_queue[ihop:ihop+leftover]
                        self._freq_far_queue[:leftover] = self._freq_far_queue[ihop:ihop+leftover]
                    self._freq_queue_write = leftover

                r = self._freq_out_read
                raw_output = self._freq_out_buf[r:r+hop].copy()
                self._freq_out_read = r + hop
            else:
                # WebRTC-style: freeze main filter weights when shadow detected divergence
                main_mu = 0.0 if self._regime_handler.main_paused else mu_scale
                raw_output = self.filter.process(near_end, far_end, main_mu)

            # Shadow filter with DTD protection (#1) and bidirectional copy (#6)
            if self.shadow_filter is not None and self._freq_near_queue is None:
                self.shadow_frame_count += 1
                # Shadow is a true background filter: always adapts at full
                # speed. Kalman's Q_high × shadow_q_ratio keeps P alive even
                # when shadow learns DT speech, so it re-converges within
                # ~100-200 frames after DT ends. The copy gate (FS baseline
                # tracking) is the sole defense against poisoning main.
                # Gate shadow update: skip adaptation when far-end is too
                # weak (poor excitation) or speaker is saturating (nonlinear
                # distortion makes error unreliable). Cf. AEC3's
                # PoorSignalExcitation() gate on main_filter_update_gain.
                far_excited = np.mean(far_end ** 2) > 1e-4
                saturation_safe = self._saturation_level < 0.5
                if self.config.shadow_mu_state_aware:
                    # B6: 4-band state-aware schedule. Precedence:
                    #   pause (main_paused/diverged) > safety (sat/weak-far)
                    #   > caution (suspicious_dt) > default.
                    if (self._regime_handler.main_paused
                            or self._prev_filter_state == 'diverged'):
                        shadow_mu_scale = 0.0
                    elif not (far_excited and saturation_safe):
                        shadow_mu_scale = 0.1
                    elif self._prev_filter_state == 'suspicious_dt':
                        shadow_mu_scale = 0.5
                    else:
                        shadow_mu_scale = 1.0
                else:
                    shadow_mu_scale = 1.0 if (far_excited and saturation_safe) else 0.1
                self.shadow_filter.process(near_end, far_end, shadow_mu_scale)

                # S-orth.A: after shadow processes, overwrite shadow's _error_psd
                # and R with the independently-tracked decoupled state when the
                # flag is ON.  This breaks the Riccati coupling: each filter now
                # accumulates its own observation-noise estimate from its own
                # residual stream rather than sharing the same EMA accumulator.
                #
                # When flag OFF: shadow_filter._error_psd / .R are set by the
                # shadow's own _update_weights call (existing behaviour).  We do
                # NOT touch them, so the path is byte-equal to baseline.
                if (self.config.shadow_state_decoupled
                        and isinstance(self.shadow_filter, PBFDKF)):
                    # --- Decoupled _error_psd update ---
                    # Use the same EMA formula as PBFDKF._update_weights but from
                    # shadow's own error_spec (already computed by shadow's process()).
                    shadow_err_spec = getattr(self.shadow_filter, 'error_spec', None)
                    if shadow_err_spec is not None:
                        _alpha_r = self.shadow_filter._alpha_r  # 0.95
                        _shadow_err_psd_inst = np.abs(shadow_err_spec) ** 2
                        self._shadow_error_psd = (
                            _alpha_r * self._shadow_error_psd
                            + (1.0 - _alpha_r) * _shadow_err_psd_inst
                        )
                        self._shadow_R = np.maximum(
                            self._shadow_error_psd, self.shadow_filter.delta)
                        # Quiescent safety regularization (Option B):
                        # When filter is converged and shadow _error_psd has drifted
                        # more than 3× away from main's in either direction, nudge
                        # shadow back by 10% blend per frame.  Fires only in steady
                        # FS (refined_usable + far_excited) so it cannot corrupt the
                        # non-stationary path where orthogonality matters most.
                        # B4 fix (2026-05-14): drop dead 'converged' branch — that
                        # string belongs to AecFilterState enum, not the internal
                        # P3f state machine (lines 7180-7199), which only sets
                        # 'refined_usable' for steady FS.
                        _is_quiescent = (
                            far_excited
                            and hasattr(self.filter, '_error_psd')
                            and getattr(self, '_prev_filter_state', 'idle')
                                == 'refined_usable'
                        )
                        if _is_quiescent:
                            _main_psd = self.filter._error_psd  # current main
                            _ratio = self._shadow_error_psd / (
                                _main_psd + np.float32(1e-10))
                            _needs_nudge = np.any(_ratio > 3.0) or np.any(
                                _ratio < 0.333)
                            if _needs_nudge:
                                # 10% blend toward main per quiescent frame
                                self._shadow_error_psd = (
                                    np.float32(0.9) * self._shadow_error_psd
                                    + np.float32(0.1) * _main_psd)
                                self._shadow_R = np.maximum(
                                    self._shadow_error_psd,
                                    self.shadow_filter.delta)
                        # Write back into shadow_filter so subsequent Kalman
                        # gain K uses the decoupled R.
                        self.shadow_filter._error_psd = self._shadow_error_psd
                        self.shadow_filter.R = self._shadow_R

                main_err = self.filter.get_error_energy()
                shadow_err = self.shadow_filter.get_error_energy()

                alpha_s = self.config.shadow_err_alpha
                self.main_err_smooth = alpha_s * self.main_err_smooth + (1 - alpha_s) * main_err
                self.shadow_err_smooth = alpha_s * self.shadow_err_smooth + (1 - alpha_s) * shadow_err

                self._dt_analyzer.update_shadow_dt(
                    shadow_frame_count=self.shadow_frame_count,
                    far_excited=far_excited,
                    main_err_smooth=self.main_err_smooth,
                    shadow_err_smooth=self.shadow_err_smooth,
                )

                # Copy gate: delegated to PathChangeRegimeHandler. The handler
                # owns _copy_err_baseline / streak counters / pause hangover and
                # returns a RegimeHandlerDecision. Filter mutations (Q-boost, reverse
                # copy) are applied here so the handler stays decision-only.
                # Phase C1: optional coherence+delay gate inputs (default gate_mode='energy'
                # ignores them, so legacy behavior parity-preserved).
                _dt_coh = self.dtd_coherence.confidence if self.dtd_coherence else 0.0
                # v3.10.1: shadow-copy gate raised from any-confidence to ≥0.5.
                # Shadow→main copy permanently overwrites filter taps; should
                # only allow it when delay alignment is at least mid-confidence
                # (PAR halfway between par_low and par_solid).
                # F-DelayTrack: track delay estimate stability via variance.
                # Append valid estimates to bounded history; when enabled,
                # gate reliability on variance < 4 samples AND confidence >=
                # 0.3 (relaxed minimum since variance is the primary signal).
                if (self.delay_est is not None
                        and self.delay_est.estimated_delay >= 0):
                    self._delay_history.append(int(self.delay_est.estimated_delay))
                if self.config.f_delaytrack_enabled:
                    if (self.delay_est is not None
                            and len(self._delay_history) >= 3):
                        delay_std = float(np.std(np.asarray(self._delay_history,
                                                            dtype=np.float32)))
                        _delay_reliable = (
                            delay_std < 4.0
                            and self.delay_est.confidence >= 0.3
                        )
                    else:
                        _delay_reliable = False
                else:
                    _delay_reliable = (
                        self.delay_est is not None
                        and self.delay_est.confidence >= 0.5
                    )
                # P52 A.0R.2 trace: snapshot regime-relevant state *before*
                # the handler decision + filter mutations. Audio-passive.
                _trace_p52 = getattr(self.config, 'trace_p52_regime_handler', False)
                if _trace_p52:
                    _t_main_w_before = float(np.linalg.norm(self.filter.W)) \
                        if hasattr(self.filter, 'W') else 0.0
                    _t_main_q_before = float(np.max(self.filter.Q)) \
                        if hasattr(self.filter, 'Q') else 0.0
                    _t_shadow_w_before = float(np.linalg.norm(self.shadow_filter.W)) \
                        if (self.shadow_filter is not None and
                            hasattr(self.shadow_filter, 'W')) else 0.0
                    _t_erle_before = float(self.get_erle_instant())

                shadow_decision = self._regime_handler.update(
                    shadow_frame_count=self.shadow_frame_count,
                    far_pwr=float(np.mean(far_end ** 2)),
                    main_err_smooth=float(self.main_err_smooth),
                    shadow_err_smooth=float(self.shadow_err_smooth),
                    epc_active=self.epc_active,
                    saturation_level=float(self._saturation_level),
                    dt_from_energy=float(self._dt_from_energy),
                    dt_from_coherence=float(_dt_coh),
                    delay_reliable=bool(_delay_reliable),
                )
                if shadow_decision.boost_q:
                    if hasattr(self.filter, 'Q') and hasattr(self.filter, 'Q_high'):
                        self.filter.Q = self.filter.Q_high.copy()
                        self.filter._p_max_override = 1.0
                        self.filter._p_max_override_frames = 20
                    # F2.3: Yang 2017 R-reset — over-estimated R from a prior DT
                    # period suppresses Kalman gain K post-EPC, causing slow
                    # reconvergence. Reset to R_init (1e-2) so K recovers fast.
                    if self.config.epc_r_reset_enabled:
                        self.filter._error_psd.fill(1e-2)
                        self.filter.R.fill(1e-2)
                    # B5: symmetric R-reset on shadow filter — without it, the
                    # K-handicapped shadow (stale R from same DT period) feeds
                    # a wrong-K W into main on the next reverse_copy event,
                    # undoing F2.3's fast-recovery benefit.
                    if self.config.shadow_r_reset_enabled and self.shadow_filter is not None:
                        self.shadow_filter._error_psd.fill(1e-2)
                        self.shadow_filter.R.fill(1e-2)
                if shadow_decision.reverse_copy:
                    # Sync shadow back to main when main is clearly better.
                    self.shadow_filter.copy_weights_from(self.filter)
                    self.shadow_err_smooth = self.main_err_smooth
                    # F1.1: re-arm shadow P so K recalibrates for copied W;
                    # also boost main P for faster re-adaptation post-copy.
                    if self.config.reverse_copy_p_reset:
                        for filt in (self.shadow_filter, self.filter):
                            # Use hasattr(filt, 'P') as PBFDKF guard — P is always
                            # set as an instance var; _p_max_override is dynamic
                            # (deleted when expired), so hasattr on it gives False
                            # outside active EPC overrides.
                            if filt is not None and hasattr(filt, 'P'):
                                filt._p_max_override = 1.0
                                filt._p_max_override_frames = 15
                                filt._p_floor_beta = 1.0
                                filt._p_floor_beta_frames = 15

                if _trace_p52:
                    self._regime_trace_rows.append({
                        'frame': int(self._frame_count),
                        'boost_q_fired': bool(shadow_decision.boost_q),
                        'reverse_copy_fired': bool(shadow_decision.reverse_copy),
                        'main_paused_fired': bool(shadow_decision.pause_main),
                        'w_l2_before': _t_main_w_before,
                        'w_l2_after': float(np.linalg.norm(self.filter.W))
                            if hasattr(self.filter, 'W') else 0.0,
                        'q_max_before': _t_main_q_before,
                        'q_max_after': float(np.max(self.filter.Q))
                            if hasattr(self.filter, 'Q') else 0.0,
                        'shadow_w_l2_before': _t_shadow_w_before,
                        'shadow_w_l2_after': float(np.linalg.norm(self.shadow_filter.W))
                            if (self.shadow_filter is not None and
                                hasattr(self.shadow_filter, 'W')) else 0.0,
                        'erle_main_before': _t_erle_before,
                        'erle_main_after': float(self.get_erle_instant()),
                        'copy_counter': int(self._regime_handler.copy_counter),
                        'copy_err_baseline': float(
                            self._regime_handler.copy_err_baseline),
                    })

            # F-E3: increment "frames since last EPC" counters once per frame,
            # before EPC detection so that fire-then-reset-to-0 logic in the
            # helper observes the correct count when consecutive EPC fires.
            if self._frames_since_last_epc < 10**9:
                self._frames_since_last_epc += 1
            if self._frames_since_last_f_e3_w_reset < 10**9:
                self._frames_since_last_f_e3_w_reset += 1

            # EchoPathVariability: gain-change detection (delegated to EchoPathChangeDetector)
            epv_event = self._epc_det.update_epv(
                far_pwr_global=far_pwr_global,
                filter_converged=self._filter_converged,
                main_paused=self._regime_handler.main_paused,
            )
            # ── Round 7.1a: EPV weak-filter false-positive damping ──
            # Worst-DT_mv fires EPV 2.75x more often than best (P0 trace) but
            # its filter_w_norm is 40% of best (~7 vs ~18). Re-arming Kalman
            # state on a filter that hasn't yet grown destabilises it without
            # benefit. Skip the EPV response when main filter W norm is below
            # threshold; let it warm up. Gated by env so default = baseline.
            _epv_raw = bool(epv_event.fired)
            _epv_suppressed = False
            if _epv_raw:
                _w_thr = float(os.environ.get('R7_EPV_WEAK_THR', '0.0') or 0.0)
                if _w_thr > 0.0 and self.filter is not None:
                    _w_norm_now = float(np.linalg.norm(self.filter.W))
                    if _w_norm_now < _w_thr:
                        _epv_suppressed = True
            self._diag['epv_event_raw'] = _epv_raw
            self._diag['epv_event_suppressed'] = _epv_suppressed
            if epv_event.fired and not _epv_suppressed:
                for filt in [self.filter, self.shadow_filter]:
                    if filt and hasattr(filt, 'Q'):
                        filt.Q = filt.Q_high.copy()
                        filt._p_max_override = 1.0
                        filt._p_max_override_frames = 30
                        filt._p_floor_beta = 1.0
                        filt._p_floor_beta_frames = 30
                self._maybe_mark_diverged('epv')
                self._epc_render_forced_remaining = self.config.epc_hangover
                self._erl_estimate = min(self._erl_estimate, 0.3)
                # v3.14 Arc-P P.S2: when per-band ERL is active, also cap each
                # per-band EMA to its per-band post-EPC ceiling so a stale
                # high-coupling EMA doesn't persist across an echo path change.
                # When flag is OFF this block is skipped → byte-equal preserved.
                if self.config.f3_1_per_band_erl_adaptive:
                    _pb_caps = (self.config.per_band_erl_cap_lf,
                                self.config.per_band_erl_cap_mf,
                                self.config.per_band_erl_cap_hf)
                    for _bi, _cap in enumerate(_pb_caps):
                        self._per_band_erl[_bi] = min(self._per_band_erl[_bi], _cap)
                if self.config.use_epc_state_reset:
                    self._apply_epc_state_reset('epv')
                self._f_e3_handle_epc_fire('epv')

            # Echo path change: shadow-error rise (delegated to EchoPathChangeDetector).
            # Update + hangover tick are inside the original (shadow_filter, filter_converged)
            # gate to preserve bit-exact countdown semantics from v2.8.1.
            if self.shadow_filter is not None and self._filter_converged:
                rise_event = self._epc_det.update_shadow_rise(
                    main_err_smooth=self.main_err_smooth,
                    shadow_err_smooth=self.shadow_err_smooth,
                    is_stationary=self._render_activity.is_stationary,
                )
                # F-E5 / E5-4: mask shadow_rise during sustained saturation.
                # Clipped input causes both filter errors to rise in tandem;
                # the detector reads that as path change but it is really
                # nonlinear distortion. Avoid false EPC triggering filter
                # re-initialisation during a sat event.
                if (self.config.f_e5_enabled
                        and self._saturation_level > self.config.f_e5_main_mu_sat_threshold
                        and rise_event.fired):
                    rise_event = type(rise_event)(fired=False, source=rise_event.source)
                if rise_event.fired:
                    if self.dtd_coherence:
                        self.dtd_coherence.confidence *= 0.3
                    for filt in [self.filter, self.shadow_filter]:
                        if filt and hasattr(filt, 'Q'):
                            filt.Q = filt.Q_high.copy()
                    self._maybe_mark_diverged('shadow_rise')
                    # P_MAX relax + P_floor raise: force filter to abandon stale path estimate
                    for filt in [self.filter, self.shadow_filter]:
                        if filt:
                            filt._p_max_override = 1.0
                            filt._p_max_override_frames = 30
                            filt._p_floor_beta = 1.0
                            filt._p_floor_beta_frames = 30
                    # Change D: arm RES render-forced + cap stale ERL
                    self._epc_render_forced_remaining = self.config.epc_hangover
                    self._erl_estimate = min(self._erl_estimate, 0.3)
                    # v3.14 Arc-P P.S2: per-band EPC cap (symmetric with EPV path above).
                    # Byte-equal when flag is OFF.
                    if self.config.f3_1_per_band_erl_adaptive:
                        _pb_caps = (self.config.per_band_erl_cap_lf,
                                    self.config.per_band_erl_cap_mf,
                                    self.config.per_band_erl_cap_hf)
                        for _bi, _cap in enumerate(_pb_caps):
                            self._per_band_erl[_bi] = min(self._per_band_erl[_bi], _cap)
                    if self.config.use_epc_state_reset:
                        self._apply_epc_state_reset('shadow_rise')
                    self._f_e3_handle_epc_fire('shadow_rise')
                else:
                    # Hangover tick — only when shadow_rise did NOT fire (preserves
                    # original if/elif/else structure exactly).
                    self._epc_det.tick_hangover()

            # WebRTC-style: no output switching. Main filter output is always used.
            # (Shadow filter drives divergence detection + Q boost + pause, not output selection.)

            # final_output starts from raw_output; RES modifies final_output only
            self._last_raw_output = raw_output  # save for diagnostic (time-domain echo power)
            final_output = raw_output.copy()

            # RES post-filter using OLA + sqrt-Hann (skip for buffered FDAF)
            if (self.res or self.config.return_res_context) and self._freq_near_queue is None:
                far_power = np.mean(far_end ** 2)
                # Dynamic over_sub: moderate base, scale with convergence.
                # Windowed decaying ERLE (TC ≈ 10s) replaces irreversible
                # cumulative get_erle(): in DT, instant ERLE drops because near
                # speech raises raw_error_power, and the windowed accumulator
                # follows it down within hundreds of frames, so erle_factor
                # actually backs off and base_over_sub drops below the
                # converged-state ceiling. Pure instant is too noisy in
                # onsets, so we still take max() with instant ERLE.
                _erle_decay = 0.999  # TC ≈ 1000 frames ≈ 10s
                self._erle_window_near = (_erle_decay * self._erle_window_near
                                           + self.near_power)
                self._erle_window_err = (_erle_decay * self._erle_window_err
                                          + self.raw_error_power)
                erle_windowed = 10.0 * np.log10(
                    (self._erle_window_near + 1e-10)
                    / (self._erle_window_err + 1e-10))
                erle_for_factor = max(self.get_erle_instant(), erle_windowed)
                # D2: ramp from 0 dB (was 2 dB). With B4 dynamic ERL,
                # render-based is now useful at low ERLE → smoother blend.
                erle_factor = np.clip(erle_for_factor / 10.0, 0.0, 1.0)
                self._erle_factor_prev = float(erle_factor)
                base_over_sub = self.config.res_over_sub_base + self.config.res_over_sub_scale * erle_factor
                # Saturation boost: non-linear echo needs more suppression
                base_over_sub += self._saturation_level * self.config.saturation_over_sub_boost
                # DT protection
                far_pwr = np.mean(far_end ** 2) + 1e-10
                mic_pwr = np.mean(near_end ** 2) + 1e-10
                raw_err_pwr = np.mean(raw_output ** 2) + 1e-10
                # B4: track ERL for render-based echo estimate.
                # Gate: only update when residual is not dominated by near-end
                # speech (raw_dt < 2.0 allows high-coupling echo-only through).
                # Pre-convergence only: after convergence, filter-based echo
                # estimate is reliable and render-based mode is off.
                # F-E1: hysteresis far_active gate (attack 1e-4, release
                # after 5 frames below 3e-5). Stabilises ERL update during
                # marginal-reference dips. Falls back to simple threshold
                # when flag disabled.
                if self.config.f_e1_enabled:
                    if far_pwr > 1e-4:
                        self._f_e1_far_active = True
                        self._f_e1_far_release_count = 0
                    elif self._f_e1_far_active:
                        if far_pwr < 3e-5:
                            self._f_e1_far_release_count += 1
                            if self._f_e1_far_release_count >= 5:
                                self._f_e1_far_active = False
                                self._f_e1_far_release_count = 0
                        else:
                            self._f_e1_far_release_count = 0
                    erl_update_gate = self._f_e1_far_active
                    # F-E1: extend ERL clip lower bound to 1e-5 (was 0.001)
                    # so extreme high coupling cases pass through cleanly.
                    erl_clip_lo = 1e-5
                else:
                    erl_update_gate = (far_pwr > 1e-4)
                    erl_clip_lo = 0.001
                if erl_update_gate:
                    raw_dt_ratio = raw_err_pwr / (far_pwr + 1e-10)
                    inst_erl_raw = mic_pwr / far_pwr
                    # v3.2 Axis 1: NE-corruption protection. ERL > 1.5 physically
                    # implausible (mic louder than far → NE dominates), so skip update.
                    if raw_dt_ratio < 2.0 and inst_erl_raw < 1.5:
                        inst_erl = np.clip(inst_erl_raw, erl_clip_lo, 1.0)
                        alpha_erl = 0.99 if not self._filter_converged else 0.999
                        self._erl_estimate = float(alpha_erl * self._erl_estimate + (1 - alpha_erl) * inst_erl)
                    # v3.14 Arc-P P.S3: adaptive per-band ERL EMA update.
                    # SOURCE SIGNAL CORRECTION (P.S3):
                    #   P.S2 used |echo_spec|²/|far_spec|² (PBFDKF predicted
                    #   echo divided by far, single-frame complex ratio).
                    #   Discrepancy: P.S2 gave LF=0.57 vs P.S1 oracle=0.043.
                    #   Root cause: |W·X|²/|X|² = |Ĥ(k)|² (estimated filter
                    #   response power), not room ERL.  In FS-converged state
                    #   PBFDKF W overmodels: ||W||² reflects cumulative energy
                    #   in the W taps, not the ratio of cancelled-to-total echo.
                    #   CORRECT source: res.error_psd / far_lw — exactly what
                    #   P.S1 validated.  In converged FS: error ≈ residual echo
                    #   ≈ far × ERL_room, so error_psd/far_lw ≈ ERL_room per
                    #   band.  res.error_psd is an EMA-smoothed per-bin PSD
                    #   (alpha=0.8 BALANCED) — robust to single-frame noise.
                    #   far_lw is the long-window far PSD used in F3.1 excess
                    #   formula — same denominator for consistency.
                    # GATE (sibling of scalar ERL update, not nested inside):
                    #   Uses _filter_converged + lw_ready + far_active (the
                    #   outer erl_update_gate already enforces far_pwr > 1e-4).
                    #   Does NOT require raw_dt_ratio < 2.0 or inst_erl_raw < 1.5
                    #   because res.error_psd is EMA-smoothed (robust to single-
                    #   frame NE contamination) and _filter_converged is the
                    #   primary reliability gate.  In high-coupling rooms (case
                    #   08), inst_erl_raw=28 because mic >>> far at the reference
                    #   level, but error_psd/far_lw correctly tracks the room.
                    if (self.config.f3_1_per_band_erl_adaptive
                            and self._filter_converged
                            and self.res is not None
                            and self.res._residual_est is not None
                            and self.res._residual_est._long_window_n_updates > 0):
                        _err_psd_pb = self.res.error_psd      # EMA per-bin, shape (n_freqs,)
                        _far_lw_pb = self.res._residual_est._long_window_far_psd
                        # Band boundaries in bins (16 kHz, fft_size=640 → 257 bins)
                        # freq_per_bin = sr / fft_size
                        _fpb = self.config.sample_rate / float(self.config.fft_size)
                        _bin_1k = max(1, int(round(1000.0 / _fpb)))
                        _bin_4k = max(_bin_1k + 1, int(round(4000.0 / _fpb)))
                        _n = _err_psd_pb.shape[0]
                        _bin_1k = min(_bin_1k, _n - 2)
                        _bin_4k = min(_bin_4k, _n - 1)
                        # Per-band mean(error_psd) / mean(far_lw); skip band
                        # if far energy is negligible (avoids 0-div artefacts).
                        _alpha_pb = self.config.per_band_erl_alpha
                        _clip_lo = self.config.per_band_erl_clip_lo
                        _clip_hi = self.config.per_band_erl_clip_hi
                        for _bi, (_bs, _be) in enumerate(
                                ((0, _bin_1k), (_bin_1k, _bin_4k), (_bin_4k, _n))):
                            if _be <= _bs:
                                continue
                            _far_band = float(np.mean(_far_lw_pb[_bs:_be]))
                            if _far_band < 1e-10:
                                continue
                            _inst_pb = float(np.mean(_err_psd_pb[_bs:_be])) / _far_band
                            _inst_pb = float(np.clip(_inst_pb, _clip_lo, _clip_hi))
                            self._per_band_erl[_bi] = (
                                _alpha_pb * self._per_band_erl[_bi]
                                + (1.0 - _alpha_pb) * _inst_pb
                            )

                # Pre-filter DT signal (Stage B): mic energy excess over
                # far × max_ERL. Realistic rooms have ERL ≤ +6 dB (coupling
                # factor 4.0). When mic_pwr > 4 × far_pwr, the mic has
                # energy that can't be explained by echo alone → near-end
                # speech or NE segment. This signal is PRE-FILTER so it's
                # immune to the inst_erle correction that kills raw_dt in
                # high-coupling DT crush cases.
                # Gate on far_active: NE-only segments (far≈0) would produce
                # dt_from_energy≈1.0, and the slow EMA decay (TC≈90ms) would
                # hang over into the following FS segment, relaxing ENR
                # thresholds while echo is present → FS echo leakage.
                self._dt_analyzer.update_energy_dt(
                    far_active=self._render_activity.is_active,
                    far_pwr=far_pwr,
                    mic_pwr=mic_pwr,
                    erl_estimate=self._erl_estimate,
                )

                # Step 1: base DT confidence
                if self.config.enable_dtd:
                    raw_dt = self.get_dtd_confidence()
                else:
                    simple_dt = 1.0 - far_pwr / (mic_pwr + far_pwr)
                    # ERL-corrected blend: simple ratio misfires in high-ERL FS
                    # (mic ≈ echo > far → simple says "DT", ERL-corrected says "FS").
                    # Take max of ERL-corrected estimate and half the simple ratio
                    # so true DT (dt_from_energy high) is preserved, false DT is halved.
                    raw_dt = max(float(self._dt_from_energy), simple_dt * 0.5)

                # Stationary DT macro detection (sets flag only, does NOT override raw_dt)
                is_stationary_dt = False
                if self._render_activity.is_stationary and self._filter_converged:
                    if hasattr(self.filter, 'error_spec'):
                        freq_per_bin = self.config.sample_rate / self.filter.fft_size
                        vb_start = max(1, int(100.0 / freq_per_bin))
                        vb_limit = min(int(3000.0 / freq_per_bin), len(self.filter.error_spec))
                        track_err_pwr = (float(np.sum(
                            np.abs(self.filter.error_spec[vb_start:vb_limit]) ** 2)) + 1e-10)
                    else:
                        track_err_pwr = raw_err_pwr

                    if self._wn_err_baseline < 1e-6:
                        self._wn_err_baseline = track_err_pwr

                    jump_ratio = track_err_pwr / (self._wn_err_baseline + 1e-10)

                    if jump_ratio > 1.5:
                        self._stat_dt_hangover = 80  # 800ms protection window (covers syllable gaps)

                    if self._stat_dt_hangover > 0:
                        is_stationary_dt = True
                        self._stat_dt_hangover -= 1
                        # Speech active: nearly freeze baseline (TC ≈ 1000 frames)
                        self._wn_err_baseline = (0.999 * self._wn_err_baseline
                                                  + 0.001 * track_err_pwr)
                    else:
                        is_stationary_dt = False
                        # Silence: normal EMA tracking WN baseline
                        self._wn_err_baseline = (0.95 * self._wn_err_baseline
                                                  + 0.05 * track_err_pwr)

                # D4: slowly track baseline during non-stationary far-end
                # (converged only). Prevents stale 1e-8 baseline when clip
                # starts with speech far-end → first stationary transition
                # sees huge jump_ratio → false stationary DT trigger.
                if (self._filter_converged and not self._render_activity.is_stationary
                        and far_pwr > 1e-4 and self._wn_err_baseline > 1e-6):
                    self._wn_err_baseline = (0.995 * self._wn_err_baseline
                                              + 0.005 * raw_err_pwr)

                # inst_erle correction (only no-DTD)
                if not self.config.enable_dtd:
                    inst_erle_fast_raw = mic_pwr / raw_err_pwr
                    self._inst_erle_smooth = (0.7 * self._inst_erle_smooth
                                              + 0.3 * inst_erle_fast_raw)
                    if self._inst_erle_smooth > 2.0:
                        # Cap correction divisor: inst_erle=15 would make
                        # raw_dt=0.6→0.04, killing DT protection entirely.
                        # Cap at 4.0 keeps raw_dt=0.6→0.15, preserving
                        # some DT awareness while still reducing false DT
                        # in high-coupling FS (original purpose).
                        erle_for_dt = min(self._inst_erle_smooth, 4.0)
                        raw_dt /= erle_for_dt

                # EPC physical gate
                # Round 3 D3 trace: record raw_dt BEFORE the EPC zero so we can
                # see what the suppressor would have seen if the gate were
                # split (adaptation vs RES). audio-passive.
                self._round3_raw_dt_pre_epc = float(raw_dt)
                if self.epc_active:
                    raw_dt = 0.0
                    is_stationary_dt = False  # EPC error spike is from filter divergence, not speech

                dt_indicator = np.clip(raw_dt, 0.0, 0.8)
                # NOTE: dt_reduction → effective_over_sub is dead code when
                # gain_type="enr" (all presets). over_sub is only read by
                # wiener/spectral_sub paths. Kept for backward compat if
                # gain_type is ever changed.
                dt_reduction = self.config.res_dt_reduction * dt_indicator
                effective_over_sub = max(base_over_sub - dt_reduction, 0.5)

                # Divergence indicator EMA (delegated to FilterConvergenceAnalyzer)
                self._convergence.update_divergence(self.near_power, self.raw_error_power)

                if self.res:
                    # Change D: during EPC render-forced window, force RES
                    # into render-based echo estimate (unreliable filter W).
                    if getattr(self, '_epc_render_forced_remaining', 0) > 0:
                        self._epc_render_forced_remaining -= 1
                        self.res._using_render_based = True
                    self.res.over_sub = effective_over_sub

                    # DT conservative residual scaling: 1.0→0.5 as dt goes 0→0.8
                    dt_residual_scale = 1.0 - 0.5 * float(np.clip(dt_indicator, 0.0, 0.8) / 0.8)
                    eff_echo_spec = self.filter.echo_spec * dt_residual_scale

                    _shadow_dt = max(float(self._dt_from_energy),
                                     float(getattr(self, '_dt_from_shadow', 0.0)))
                    shadow_dt = 0.08 * _shadow_dt if self.epc_active else _shadow_dt

                    # v3.14 Arc-P P.S2: when f3_1_per_band_erl_adaptive=True,
                    # pass a per-bin ERL array to ResFilter so the F3.1-v3
                    # mic-excess formula uses per-band adaptive estimates
                    # instead of the scalar _erl_estimate.  The per-bin array
                    # is built from _per_band_erl[LF, MF, HF] by broadcasting
                    # each band's value to the corresponding bin range.  This
                    # is identical to erl_estimate=scalar when the flag is OFF
                    # (scalar float path) → byte-equal guaranteed.
                    if self.config.f3_1_per_band_erl_adaptive and hasattr(self, '_per_band_erl'):
                        _fpb2 = self.config.sample_rate / float(self.config.fft_size)
                        _nf2 = self.res.n_freqs
                        _b1k2 = max(1, min(int(round(1000.0 / _fpb2)), _nf2 - 2))
                        _b4k2 = max(_b1k2 + 1, min(int(round(4000.0 / _fpb2)), _nf2 - 1))
                        _erl_pb = np.empty(_nf2, dtype=np.float32)
                        _erl_pb[:_b1k2] = float(self._per_band_erl[0])
                        _erl_pb[_b1k2:_b4k2] = float(self._per_band_erl[1])
                        _erl_pb[_b4k2:] = float(self._per_band_erl[2])
                        _erl_arg = _erl_pb
                    else:
                        _erl_arg = self._erl_estimate
                    final_output = self.res.process(raw_output, eff_echo_spec,
                                                    far_power, self.filter.far_spec,
                                                    filter_converged=self._filter_converged,
                                                    erle_factor=erle_factor,
                                                    dt_indicator=float(dt_indicator),
                                                    near_spec=self.filter.near_spec,
                                                    divergence=self._divergence_indicator,
                                                    is_stationary_dt=is_stationary_dt,
                                                    saturation_level=self._saturation_level,
                                                    epc_active=self.epc_active,
                                                    shadow_dt=shadow_dt,
                                                    erl_estimate=_erl_arg,
                                                    e2_main=float(self.main_err_smooth),
                                                    e2_shadow=float(self.shadow_err_smooth),
                                                    y2=float(far_power),
                                                    filter_once_converged=self._filter_once_converged,
                                                    aec_state=self._aec_state,
                                                    filter_state=self._prev_filter_state)

                    # v3.13 E4.S3 — SubtractiveNLP detector (audit-only).
                    # Pure observer: reads the LINEAR residual (raw_output =
                    # mic − linear_echo_estimate, RES input) so the NL
                    # harmonic signature is not masked by RES suppression.
                    # Also reads mic_hop (near_end) for the S4.1
                    # cancellation-ratio gate (NE bucket discrimination).
                    # See E4.S1 Pass A finding: production output (post-RES)
                    # hides NL evidence.
                    if self.nl_detector is not None:
                        _nl_conf = self.nl_detector.process(
                            raw_output,
                            filter_state=self._prev_filter_state,
                            far_active=(far_power > 1e-4),
                            mic_hop_samples=near_end)
                        self._diag['nl_confidence'] = _nl_conf
                        self._diag['nl_pitch_strength'] = (
                            self.nl_detector._pitch_strength_last)
                        self._diag['nl_pitch_lag'] = (
                            self.nl_detector._pitch_lag_last)

                    # Update per-bin mu_scale AFTER RES (echo_psd is now current frame)
                    if not self.config.enable_dtd:
                        if self._filter_converged:
                            per_bin_eer = self.res.echo_psd / (self.res.error_psd + 1e-10)
                            per_bin_eer = np.clip(per_bin_eer, 0.0, 1.0)
                            mu_min = self.config.shadow_mu_min
                            self._per_bin_mu_scale = (mu_min + (1.0 - mu_min) * per_bin_eer).astype(np.float32)
                            self._simple_mu_ratio = float(np.mean(per_bin_eer))
                            # Stationary DT: only freeze when speech actually
                            # detected (jump_ratio + hangover). _is_stationary_far
                            # alone fires on 32% of normal speech far-end frames,
                            # which would crush filter convergence on plain FS.
                            if is_stationary_dt:
                                self._per_bin_mu_scale[:] = mu_min
                                self._simple_mu_ratio = mu_min
                        else:
                            # Pre-convergence: no per_bin, let ratio track DT naturally
                            self._per_bin_mu_scale = None
                            self._update_simple_mu_ratio(raw_output, far_end)

                if self.config.return_res_context and not self.res:
                    _res_context = AecResContext(
                        raw_output=raw_output.copy(),
                        echo_spec=self.filter.echo_spec.copy(),
                        far_power=far_power,
                        far_spec=self.filter.far_spec.copy(),
                        near_spec=self.filter.near_spec.copy(),
                        filter_converged=self._filter_converged,
                        erle_factor=float(erle_factor),
                        dt_indicator=float(dt_indicator),
                        divergence=float(self._divergence_indicator),
                        over_sub=float(effective_over_sub),
                        saturation_level=float(self._saturation_level),
                        erl_estimate=float(self._erl_estimate),
                    )

            # C-parity fix: when RES is disabled, _update_simple_mu_ratio is never
            # called in PBFDKF path. C always calls update_simple_mu_ratio regardless.
            # Add call here when RES is not active (avoids double-update when RES is on).
            if not self.res and not self.config.enable_dtd and not self._filter_converged:
                self._update_simple_mu_ratio(raw_output, far_end)

            # Update diagnostics
            if self.res and hasattr(self.res, '_diag_gain_mean'):
                self._diag['res_gain_mean'] = self.res._diag_gain_mean
                self._diag['res_gain_min'] = self.res._diag_gain_min
                self._diag['effective_g_min'] = self.res._diag_effective_g_min
                self._diag['far_activity'] = self.res._diag_far_activity
                self._diag['echo_psd_mean'] = self.res._diag_echo_psd_mean
                self._diag['error_psd_mean'] = self.res._diag_error_psd_mean
            if self.res is not None and hasattr(self.res, '_p4b_dt_per_bin_mean'):
                self._diag['p4b_dt_per_bin_mean'] = float(self.res._p4b_dt_per_bin_mean)
                self._diag['p4b_dt_per_bin_hf_mean'] = float(self.res._p4b_dt_per_bin_hf_mean)
                self._diag['p4b_coh2_hf_mean'] = float(self.res._p4b_coh2_hf_mean)
                self._diag['p4b_effective_dt'] = float(self.res._p4b_effective_dt)
                self._diag['p4b_is_stationary_dt'] = int(self.res._p4b_is_stationary_dt)
                self._diag['p4b_gain_hf_mean'] = float(self.res._p4b_gain_hf_mean)
                self._diag['p4b_res_echo_hf_mean_db'] = float(self.res._p4b_res_echo_hf_mean_db)
            self._diag['erle_inst'] = self.get_erle_instant()

            # P1 Phase 1: high-band NE evidence metrics (trace-only).
            # Computes 3 candidate metrics from post-attribution residual
            # error_psd[2k:], far_lw, and mic near_psd. NO behaviour change.
            if self.config.trace_high_band_metrics and self.res is not None:
                try:
                    err_psd = self.res.error_psd
                    near_psd = self.res.near_psd
                    far_lw = self.res._residual_est._long_window_far_psd
                    n_freqs = err_psd.shape[0]
                    sr = self.config.sample_rate
                    fft_sz = self.config.fft_size
                    bin_2k = int(round(2000.0 * fft_sz / sr))
                    bin_2k = max(1, min(bin_2k, n_freqs - 2))
                    err_hb = err_psd[bin_2k:]
                    far_hb = far_lw[bin_2k:]
                    near_hb = near_psd[bin_2k:]
                    erl_e = float(self._erl_estimate)
                    err_hb_mean = float(np.mean(err_hb)) + 1e-10
                    # m_excess_ratio for α ∈ {0.5, 1.0, 2.0}
                    for alpha, key in ((0.5, 'm_excess_ratio_a05'),
                                        (1.0, 'm_excess_ratio_a10'),
                                        (2.0, 'm_excess_ratio_a20')):
                        excess = np.maximum(err_hb - alpha * far_hb * erl_e, 0.0)
                        self._diag[key] = float(np.mean(excess)) / err_hb_mean
                    # m_modulation: high-band mic envelope CV^2 over 32-frame window
                    cur_pwr = float(np.mean(near_hb))
                    self._hb_mic_pwr_ring[self._hb_mic_pwr_idx] = cur_pwr
                    self._hb_mic_pwr_idx = (self._hb_mic_pwr_idx + 1) % 32
                    if self._hb_mic_pwr_n < 32:
                        self._hb_mic_pwr_n += 1
                    win = self._hb_mic_pwr_ring[:self._hb_mic_pwr_n]
                    win_mean = float(np.mean(win))
                    win_var = float(np.var(win))
                    self._diag['m_modulation'] = win_var / (win_mean ** 2 + 1e-10)
                    # m_spectral_flatness on err_hb (Wiener entropy)
                    err_hb_safe = err_hb + 1e-10
                    log_geo = float(np.mean(np.log(err_hb_safe)))
                    arith = float(np.mean(err_hb_safe)) + 1e-10
                    self._diag['m_spectral_flatness'] = float(np.exp(log_geo)) / arith
                    # Aux: bin index used (for sanity / debug)
                    self._diag['m_bin_2k'] = int(bin_2k)
                except Exception as _e:
                    # Never break release path — trace is best-effort.
                    self._diag['m_excess_ratio_a05'] = 0.0
                    self._diag['m_excess_ratio_a10'] = 0.0
                    self._diag['m_excess_ratio_a20'] = 0.0
                    self._diag['m_modulation'] = 0.0
                    self._diag['m_spectral_flatness'] = 0.0

            mu_val = mu_scale
            self._diag['mu_scale'] = float(np.mean(mu_val)) if isinstance(mu_val, np.ndarray) else float(mu_val)
            self._diag['converged'] = self._filter_converged
            self._diag['erle_factor'] = float(erle_factor) if 'erle_factor' in locals() else 0.0
            self._diag['divergence'] = self._divergence_indicator
            # G4: expanded diagnostics
            self._diag['using_render_based'] = bool(getattr(self.res, '_using_render_based', False)) if self.res else False
            self._diag['shadow_advantage'] = getattr(self, '_shadow_advantage', 1.0)
            self._diag['dt_from_energy'] = self._dt_from_energy
            self._diag['dt_from_shadow'] = getattr(self, '_dt_from_shadow', 0.0)
            self._diag['erl_estimate'] = self._erl_estimate
            # v3.14 Arc-P P.S2: expose per-band ERL EMA values for auditing.
            # Zero-cost when flag OFF (array always exists but values stay 0.1).
            self._diag['per_band_erl_lf'] = float(self._per_band_erl[0])
            self._diag['per_band_erl_mf'] = float(self._per_band_erl[1])
            self._diag['per_band_erl_hf'] = float(self._per_band_erl[2])
            self._diag['epc_active'] = self.epc_active
            self._diag['saturation_level'] = self._saturation_level
            self._diag['erle_windowed'] = float(erle_windowed) if 'erle_windowed' in locals() else 0.0
            # DT / filter debug fields
            self._diag['dt_indicator'] = float(dt_indicator) if 'dt_indicator' in locals() else 0.0
            self._diag['main_err_smooth'] = float(getattr(self, 'main_err_smooth', 0.0))
            self._diag['shadow_err_smooth'] = float(getattr(self, 'shadow_err_smooth', 0.0))
            self._diag['main_paused'] = bool(self._regime_handler.main_paused)
            _epv_ratio = (self._epv_gain_fast / (self._epv_gain_slow + 1e-10)
                          if self._epv_gain_slow > 1e-12 else 1.0)
            self._diag['epv_gain_ratio'] = float(_epv_ratio)
            self._diag['dt_residual_scale'] = float(dt_residual_scale) if 'dt_residual_scale' in locals() else 1.0
            self._diag['filter_w_norm'] = float(np.linalg.norm(self.filter.W)) if hasattr(self.filter, 'W') else 0.0
            self._diag['shadow_w_norm'] = (float(np.linalg.norm(self.shadow_filter.W))
                                            if self.shadow_filter and hasattr(self.shadow_filter, 'W') else 0.0)
            self._diag['copy_err_baseline'] = float(self._regime_handler.copy_err_baseline)

            # ---- P3f Mini AecState trace (no behaviour change) ----
            # Full-band WebRTC-style ratios e2/y2. Numerator/denominator must
            # share units: use the sample-loop EMAs of time-domain power
            # (self.raw_error_power, self.near_power, alpha=0.999), which are
            # commensurate. main_err_smooth / shadow_err_smooth are sums over
            # frequency bins (different scale from mean(near²)) so they
            # cannot be used directly as e2 numerators. We still use them
            # to derive shadow_err_power by scaling: same get_error_energy()
            # scaling factor for both main and shadow, so the ratio is
            # preserved.
            _mic_pwr_p3f = max(float(self.near_power), 1e-12)
            _main_err_pwr = max(float(self.raw_error_power), 1e-12)
            _main_err_smooth_p3f = max(float(self.main_err_smooth), 1e-12)
            _shadow_err_smooth_p3f = max(float(self.shadow_err_smooth), 1e-12)
            # shadow_err in the same time-domain scale as raw_error_power
            _shadow_err_pwr = _shadow_err_smooth_p3f * (
                _main_err_pwr / _main_err_smooth_p3f)
            _main_err_ratio = _main_err_pwr / _mic_pwr_p3f
            _shadow_err_ratio = _shadow_err_pwr / _mic_pwr_p3f
            _shadow_advantage_p3f = (
                _main_err_smooth_p3f / _shadow_err_smooth_p3f)

            self._post_reset_age_frames += 1
            _hop_ms_p3f = float(self.config.hop_size) * 1000.0 / float(self.config.sample_rate)
            _post_reset_age_ms = self._post_reset_age_frames * _hop_ms_p3f

            # erle_slope: dB/s over the trailing ~0.5s window
            _erle_inst_db = float(self._diag.get('erle_inst', 0.0))
            self._erle_slope_buf.append(_erle_inst_db)
            if len(self._erle_slope_buf) >= 2:
                _slope_dt = (len(self._erle_slope_buf) - 1) * _hop_ms_p3f / 1000.0
                _erle_slope_db_per_s = (
                    self._erle_slope_buf[-1] - self._erle_slope_buf[0]
                ) / max(_slope_dt, 1e-6)
            else:
                _erle_slope_db_per_s = 0.0

            # Filter-state classifier v1 (Phase 2a). State priority order:
            #   idle < startup < diverged < refined_usable < suspicious_dt
            #                           < coarse_learning (default)
            # Thresholds picked from Phase 1 smoke and Phase 2a iteration on
            # 5 trace cases (FS_static, FS_movement, DT_static, DT_movement,
            # 7GT). Re-tune in Phase 2b if invariants still fail.
            # Startup is the *unstable* post-reset window only. The
            # _p_max_override (shadow Q-boost) re-arms repeatedly during
            # healthy adaptation and would otherwise mask refined_usable.
            _post_reset_warmup_p3f = (self._warmup_frames > 0)
            _far_active_p3f = bool(self._render_activity.is_active)
            # Idle: insufficient signal to classify. Either far is silent
            # (no excitation to evaluate filter) or near is silent (no echo
            # / NE — main_err and mic both clamp to floor giving ratio 1).
            # Recognise by the floor signature: main_err_smooth at clamp
            # (1e-9 or smaller) means filter has not produced a meaningful
            # error sample yet on the current input.
            _is_idle = (
                (not _far_active_p3f)
                or _main_err_smooth_p3f <= 1e-9
                or _mic_pwr_p3f <= 1e-8
            )
            # Diverged: filter actively making it worse. Require ratio
            # comfortably above unity (1.3) for at least 5 consecutive
            # far-active frames (~50 ms hysteresis at hop=160) to filter
            # out single-frame noise. Reset streak when condition fails.
            _diverged_th = 1.3
            _diverged_min_streak = 5
            _diverged_hit_this_frame = (not _is_idle) and _main_err_ratio > _diverged_th
            if _diverged_hit_this_frame:
                self._p3f_diverged_streak += 1
            else:
                self._p3f_diverged_streak = 0
            # F2.2 — EMA-smoothed streak. Single-frame dips don't fully reset
            # the evidence (legacy hard counter does). Always tracked; only
            # consumed by P3h reset gate when `use_diverged_streak_ema` flag
            # is True. α=0.95 by default → TC ≈ 20 frames ≈ 200 ms at hop=160.
            _alpha_dse = float(self.config.diverged_streak_ema_alpha)
            self._p3f_diverged_streak_ema = (
                _alpha_dse * self._p3f_diverged_streak_ema
                + (1.0 - _alpha_dse) * (1.0 if _diverged_hit_this_frame else 0.0)
            )

            # Pre-compute suspicious_dt criteria (used both in the flag
            # below and to gate refined_usable): NE evidence AND
            # main_err jump above the refined baseline AND shadow lead.
            # All three must be present — a single shadow_advantage spike
            # (FS-early shadow lead) does NOT qualify.
            _ne_evidence = (
                float(getattr(self, '_dt_from_energy', 0.0)) > 0.3
                or float(getattr(self, '_dt_from_shadow', 0.0)) > 0.5
            )
            _main_err_jump = (
                self._p3f_main_err_baseline > 1e-6
                and _main_err_ratio > 2.0 * self._p3f_main_err_baseline
            )
            _shadow_lead = _shadow_advantage_p3f > 1.5
            _suspicious_dt_hit = (
                self._p3f_refined_latched
                and _ne_evidence
                and _main_err_jump
                and _shadow_lead
            )

            if _is_idle:
                _filter_state = 'idle'
            elif _post_reset_warmup_p3f or _post_reset_age_ms < 200.0:
                _filter_state = 'startup'
            elif self._p3f_diverged_streak >= _diverged_min_streak:
                _filter_state = 'diverged'
            elif _suspicious_dt_hit:
                # Suspicious_dt takes precedence over refined_usable so a
                # frame with NE evidence + main_err jump + shadow lead is
                # surfaced even when the absolute ratio still looks
                # converged.
                _filter_state = 'suspicious_dt'
            elif (self._filter_once_converged
                  and _main_err_ratio < 0.7
                  and _erle_slope_db_per_s > -5.0):
                # Latch refined once seen so suspicious_dt can subsequently
                # detect departures from this baseline.
                self._p3f_refined_latched = True
                _filter_state = 'refined_usable'
            else:
                _filter_state = 'coarse_learning'

            # Baseline tracks the *best* converged main_err_ratio: it
            # follows downward (so a better-converged filter raises the bar
            # for suspicious_dt) but freezes when the ratio rises (so an
            # NE-contaminated rising ratio doesn't drag the baseline along
            # with it and defeat the 2× jump trigger). Only updated while
            # in refined_usable.
            if _filter_state == 'refined_usable':
                if self._p3f_main_err_baseline <= 1e-6:
                    self._p3f_main_err_baseline = _main_err_ratio
                elif _main_err_ratio < self._p3f_main_err_baseline:
                    # downward EMA: capture better convergence
                    self._p3f_main_err_baseline = (
                        0.9 * self._p3f_main_err_baseline + 0.1 * _main_err_ratio)
                # else: ratio is rising — freeze baseline so jump can fire

            _delay_solid_p3f = bool(
                self.delay_est is not None
                and getattr(self.delay_est, 'is_solid', False))
            _usable_linear = bool(
                _delay_solid_p3f
                and _filter_state == 'refined_usable')

            self._diag['main_err_ratio'] = float(_main_err_ratio)
            self._diag['shadow_err_ratio'] = float(_shadow_err_ratio)
            self._diag['p3f_shadow_advantage'] = float(_shadow_advantage_p3f)
            self._diag['erle_slope_db_per_s'] = float(_erle_slope_db_per_s)
            self._diag['post_reset_age_ms'] = float(_post_reset_age_ms)
            self._diag['filter_state'] = str(_filter_state)
            self._diag['usable_linear'] = bool(_usable_linear)
            self._diag['p3f_main_err_baseline'] = float(self._p3f_main_err_baseline)
            # B6 — cache for next-frame shadow_mu state-aware schedule.
            self._prev_filter_state = _filter_state

            # P3g Phase 0 — dry-run residual source audit. Linear residual
            # (Stage-1 ERLE-blended) is computed every frame; render-based
            # residual is computed only when the legacy switch hits Stage-2
            # (`using_render_based` is True). Comparing the two tells us
            # how much the render override is changing the residual the
            # post-filter sees, per state class. No behaviour change.
            if self.res is not None and hasattr(self.res, '_residual_est'):
                _est = self.res._residual_est
                _lin = float(getattr(_est, '_last_linear_residual_psd_mean', 0.0))
                _ren = float(getattr(_est, '_last_render_residual_psd_mean', 0.0))
                self._diag['residual_psd_linear'] = _lin
                self._diag['residual_psd_render'] = _ren
                self._diag['residual_render_blend'] = float(
                    getattr(_est, '_last_render_blend', 0.0))
            # ---- end P3f trace ----

            # ---- P3h sustained-diverged filter reset (default off) ----
            # Decrement cooldown every frame. Fire reset only when
            # filter has been good before (`_filter_once_converged`),
            # the classifier reports diverged, the streak meets the
            # configured threshold, and cooldown has elapsed. Sets
            # cooldown so we never loop-reset on a flaky classifier.
            if self._p3h_reset_cooldown_remaining > 0:
                self._p3h_reset_cooldown_remaining -= 1
            _p3h_fired = False
            # F2.2 — streak-evidence selector. Default (flag OFF): legacy
            # hard counter `>= diverged_reset_streak_frames` (≥50 by default).
            # Flag ON: EMA gate `streak_ema > threshold` (default 0.7 over
            # α=0.95 ⇒ ~80%-of-frames-diverged-over-200 ms window). EMA
            # variant survives single-frame ratio dips so the gate actually
            # fires on cohort-tail cases where ratio oscillates around 1.3.
            if self.config.use_diverged_streak_ema:
                _streak_evidence_ok = (
                    self._p3f_diverged_streak_ema
                    > float(self.config.diverged_streak_ema_threshold)
                )
            else:
                _streak_evidence_ok = (
                    self._p3f_diverged_streak
                    >= int(self.config.diverged_reset_streak_frames)
                )
            # Sprint 13-14: triple-AND gate adds shadow_advantage > 2.0 to
            # eliminate the F2.2 EMA false-positive pattern (shadow tracking
            # during movement looked like divergence). Requires shadow to
            # also be ahead — only fires on true main-filter divergence
            # signature, not on path-change events.
            _triple_and_ok = (
                not self.config.diverged_reset_triple_and
                or _shadow_advantage_p3f
                    > float(self.config.diverged_reset_triple_and_shadow_adv_min)
            )
            if (self.config.diverged_reset_enabled
                    and self._filter_once_converged
                    and self._p3h_reset_cooldown_remaining == 0
                    and _filter_state == 'diverged'
                    and _streak_evidence_ok
                    and _triple_and_ok):
                self._reset_filter_derived_state(reason='p3h_diverged')
                self._p3h_reset_cooldown_remaining = int(
                    self.config.diverged_reset_cooldown_frames)
                self._p3h_reset_count += 1
                _p3h_fired = True
            self._diag['p3h_reset_fired'] = bool(_p3h_fired)
            self._diag['p3h_reset_cooldown'] = int(self._p3h_reset_cooldown_remaining)
            self._diag['p3h_reset_count'] = int(self._p3h_reset_count)
            self._diag['p3f_diverged_streak_ema'] = float(self._p3f_diverged_streak_ema)
            # ---- end P3h ----

            self._far_power_ema = 0.95 * self._far_power_ema + 0.05 * far_pwr_global
            self._mic_power_ema = 0.95 * self._mic_power_ema + 0.05 * (np.mean(near_end ** 2) + 1e-10)
            self._frame_count += 1
            # PR-D4: stats-only DT-from-frame-0 detector. Counts frames where
            # filter has had ≥2s of far-active without converging AND ERL has
            # drifted upward — signature of NE-corrupted filter learning.
            # See docs/aec_methods.md appendix E for full rationale.
            if self._render_activity.is_active:
                self._far_active_blocks = getattr(self, '_far_active_blocks', 0) + 1
            if (getattr(self, '_far_active_blocks', 0) > 200
                    and not self._filter_converged
                    and self._erl_estimate > 0.4):
                self._dt_from_zero_count = getattr(self, '_dt_from_zero_count', 0) + 1
            self._diag['dt_from_zero_count'] = getattr(self, '_dt_from_zero_count', 0)
            self._diag['far_power'] = self._far_power_ema
            self._diag['mic_power'] = self._mic_power_ema
            self._diag['dt_from_coherence'] = (
                self.dtd_coherence.confidence if self.dtd_coherence else 0.0)

            # Update DTD detectors for NEXT block
            # Skip divergence detector before convergence (output>mic is normal
            # when filter hasn't learned echo path yet, not actual divergence)
            if self.dtd_divergence and self._filter_converged:
                self.dtd_divergence.detect_block(near_end, far_end, output=raw_output)
            if self.dtd_coherence and self._filter_converged:
                if self._dtd_fft_size > 0:
                    # FDAFbuffered: accumulate into DTD buffer, run at hop=FL/2
                    hop = self._hop_size
                    pos = self._dtd_acc_pos
                    self._dtd_acc_err[pos:pos+hop] = raw_output
                    self._dtd_acc_far[pos:pos+hop] = far_end
                    self._dtd_acc_pos = pos + hop
                    if self._dtd_acc_pos >= self._dtd_hop:
                        # Shift main buffer and run DTD
                        dh = self._dtd_hop
                        self._dtd_err_buf[:dh] = self._dtd_err_buf[dh:]
                        self._dtd_err_buf[dh:] = self._dtd_acc_err
                        self._dtd_far_buf[:dh] = self._dtd_far_buf[dh:]
                        self._dtd_far_buf[dh:] = self._dtd_acc_far
                        self._dtd_acc_pos = 0
                        error_spec = np.fft.rfft(self._dtd_err_buf)
                        far_spec = np.fft.rfft(self._dtd_far_buf)
                        self.dtd_coherence.detect_block(
                            near_end, far_end,
                            error_spec=error_spec, far_spec=far_spec)
                else:
                    # PBFDAF/PBFDKF: use filter's spectra directly (every frame)
                    self.dtd_coherence.detect_block(
                        near_end, far_end,
                        error_spec=self.filter.error_spec,
                        far_spec=self.filter.far_spec)
        else:
            # LMS/NLMS: use mu_scale from DTD or simple variable mu
            raw_output, echo_est = self.filter.process_block(near_end, far_end,
                                                              mu_scale=mu_scale)
            final_output = raw_output.copy()
            if not self.config.enable_dtd:
                self._update_simple_mu_ratio(raw_output, far_end)

        # Output limiter: final_output should never exceed mic amplitude.
        # Uses smoothed gain to avoid frame-boundary clicking artifacts.
        near_peak = np.max(np.abs(near_end))
        out_peak = np.max(np.abs(final_output))
        if out_peak > near_peak > 1e-6:
            target_gain = near_peak / out_peak
        else:
            target_gain = 1.0
        if target_gain < self._limiter_gain:
            alpha_lim = 0.3   # attack: compress quickly
        else:
            alpha_lim = 0.8   # release: recover moderately
        self._limiter_gain = alpha_lim * self._limiter_gain + (1 - alpha_lim) * target_gain
        final_output *= self._limiter_gain

        # ERLE: track raw (filter-only) and final (post-RES) separately
        for i in range(len(near_end)):
            self.near_power = self.alpha * self.near_power + (1 - self.alpha) * near_end[i] ** 2
            self.raw_error_power = self.alpha * self.raw_error_power + (1 - self.alpha) * raw_output[i] ** 2
            self.final_error_power = self.alpha * self.final_error_power + (1 - self.alpha) * final_output[i] ** 2
        self.near_power_sum += np.sum(near_end ** 2)
        self.raw_error_power_sum += np.sum(raw_output ** 2)
        self.final_error_power_sum += np.sum(final_output ** 2)
        # Backward compat: error_power = raw (for convergence detection / inst ERLE)
        self.error_power = self.raw_error_power
        self.error_power_sum = self.raw_error_power_sum

        # Convergence detection: 10 consecutive far-active frames with ERLE > 5 dB.
        # Gate on far_active (not _simple_mu_ratio) to avoid deadlock in
        # high-coupling FS where mic ≈ far × strong_coupling pulls
        # _simple_mu_ratio < 0.5 forever, blocking convergence.
        # ERLE > 5 dB sustained 10 frames is essentially impossible during
        # real DT, so we don't need an extra DT exclusion gate.
        just_converged = self._convergence.update_convergence(
            near_power=self.near_power,
            raw_error_power=self.raw_error_power,
            far_active=float(np.mean(far_end ** 2)) > 1e-4,
            warmup_done=self._warmup_frames <= 0,
        )
        if just_converged:
            # Switch to low Q: stable tracking mode
            for filt in [self.filter, self.shadow_filter]:
                if filt and hasattr(filt, 'Q_low'):
                    filt.Q = filt.Q_low.copy()

        # v3.10.0 — filter plateau detection + one-shot recovery.
        # Dispatched here (not in DT analyzer or convergence analyzer)
        # because it needs both signals: convergence flag + DT pattern
        # signature. Recovery action below is intentionally heavy (resets
        # filter taps + shadow + RES + EPC mark_diverged) — it's only
        # triggered when the filter has been stuck below ERLE convergence
        # threshold for a sustained time despite far-end activity.
        _far_now = float(np.mean(far_end ** 2))
        _far_active_now = _far_now > 1e-4
        # Use the same windowed-ERLE expression as the diag pipeline above.
        _erle_win_db = float(10.0 * np.log10(
            (self._erle_window_near + 1e-10)
            / (self._erle_window_err + 1e-10)))
        _dt_signal_now = (float(self._dt_from_shadow) > 0.3
                          or float(self._dt_from_energy) > 0.3)
        if self._plateau_detector.update(
            far_active=_far_active_now,
            dt_signal_present=_dt_signal_now,
            erle_windowed_db=_erle_win_db,
            once_converged=self._filter_once_converged,
        ):
            # Plateau confirmed — full derived-state reset. Shared helper
            # with delay_first acquisition (Codex finding: both paths reset
            # filter taps, both should clear downstream state the same way).
            # preserve_render_ema=True keeps the long-window far-PSD EMA
            # alive across recovery — it's input-side context and the
            # freshly reset filter wants it ready immediately for fallback.
            self._reset_filter_derived_state(reason='plateau',
                                              preserve_render_ema=True)

        # ── Phase 0 trace-only AEC3 state diagnostics (read-only; do not gate audio) ──
        _initial_state_active = (
            (not self._filter_once_converged)
            or self._frame_count < (self.config.warmup_frames + 50)
        )
        _initial_transition_triggered = bool(just_converged)
        _epc_hangover = self._epc_det.hangover_count if hasattr(self, '_epc_det') else 0
        _usable_v1 = self._aec_state.usable_linear_estimate
        _usable_v2 = (
            _usable_v1
            and self._frame_count > self.config.warmup_frames + 30
            and self._convergence.divergence < 0.3
            and _epc_hangover < 1
        )
        # dominant_nearend with hold counter + initial-state gate
        _ne_raw = bool(getattr(self.res, '_diag_dominant_nearend_raw', False)) if self.res else False
        _hold = getattr(self, '_dominant_nearend_hold', 0)
        if _ne_raw and not _initial_state_active:
            _hold = 5
        else:
            _hold = max(_hold - 1, 0)
        self._dominant_nearend_hold = _hold
        _dominant_ne_state = _hold > 0
        # ERLE reset taxonomy (label only, do not actually reset)
        if _initial_transition_triggered:
            _erle_reset_signal = 1   # startup-tail
        elif self._aec_state.epc_active:
            _erle_reset_signal = 2   # EPC
        elif self._convergence.divergence > 0.5:
            _erle_reset_signal = 3   # divergence-spike
        else:
            _erle_reset_signal = 0
        self._diag['initial_state_active'] = bool(_initial_state_active)
        self._diag['initial_transition_triggered'] = _initial_transition_triggered
        self._diag['usable_linear_estimate_v1'] = bool(_usable_v1)
        self._diag['usable_linear_estimate_v2'] = bool(_usable_v2)
        self._diag['dominant_nearend_like_state'] = bool(_dominant_ne_state)
        self._diag['erle_reset_signal'] = int(_erle_reset_signal)
        # ── end Phase 0 trace ──

        # ── Round 3 trace-only: delay / EPC / P-override / scale ──
        # These are audio-passive (read-only inspection of state set by other code).
        self._diag['epc_active_now'] = bool(self._epc_det.active)
        self._diag['epc_hangover_count'] = int(self._epc_det.hangover_count)
        # P-override sticky-flag: True if either filter currently in transient boost.
        _pmax_active = False
        _pfloor_active = False
        for _filt in [self.filter, self.shadow_filter]:
            if _filt is None:
                continue
            if getattr(_filt, '_p_max_override', 0.5) > 0.6:
                _pmax_active = True
            if getattr(_filt, '_p_floor_beta', 0.1) > 0.15:
                _pfloor_active = True
        self._diag['p_max_override_active'] = _pmax_active
        self._diag['p_floor_beta_active'] = _pfloor_active
        self._diag['div_source_last'] = getattr(self, '_round3_last_div_source', '')
        self._diag['div_counts'] = dict(getattr(self, '_round3_div_counts',
                                                {'delay_first': 0, 'delay_shift': 0,
                                                 'epv': 0, 'shadow_rise': 0}))
        # Filter scale ratio: filter's echo_psd magnitude vs render-based estimate
        # (far * erl). If <<1 consistently → filter underestimates → misadjustment.
        try:
            _fpw = float(np.mean(np.abs(self.filter.echo_spec) ** 2))
            _fp_far = float(np.mean(np.abs(self.filter.far_spec) ** 2))
            _scale_ratio = _fpw / (_fp_far * float(self._erl_estimate) + 1e-12)
            self._diag['filter_scale_ratio'] = _scale_ratio
        except Exception:
            self._diag['filter_scale_ratio'] = 1.0
        # Inst ERLE smooth (read from ResFilter; lives there, not on AEC).
        self._diag['inst_erle_smooth'] = float(getattr(self, '_inst_erle_smooth', 0.0))
        # Pre-EPC DT for D3 design: if we ZEROED raw_dt due to EPC, this records the
        # value we WOULD have had. Set in process() before the EPC zero (search marker).
        self._diag['raw_dt_pre_epc'] = float(getattr(self, '_round3_raw_dt_pre_epc', 0.0))
        # ── end Round 3 trace ──

        # ── Round 4 trace: per-bin RES diagnostics (audio-passive) ──
        if self.res is not None:
            for _k, _v in getattr(self.res, '_diag_round4', {}).items():
                self._diag[_k] = _v
        # ── end Round 4 trace ──

        # ── Round 5 trace: per-stage gain means (voice-band, audio-passive) ──
        if self.res is not None:
            _r5_stages = getattr(self.res, '_diag_round5_stages', None)
            if _r5_stages is not None:
                _R5_NAMES = ('softgate_emr', 'spectral_floor', 'epc_dt_cap',
                             'quiet_mask', '3bin_smooth', 'hf_cap',
                             'pre_temporal', 'post_temporal', 'after_noise_lift')
                for _i, _n in enumerate(_R5_NAMES):
                    self._diag[f'g_stage_{_n}_voice'] = float(_r5_stages[_i])
        # ── end Round 5 trace ──

        # ── Round 7 trace: filter trajectory + transition events (audio-passive) ──
        if not hasattr(self, '_r7_prev_delay'):
            self._r7_prev_delay = -1
            self._r7_prev_div_counts = {'delay_first': 0, 'delay_shift': 0,
                                         'epv': 0, 'shadow_rise': 0}

        cur_delay = int(getattr(self, '_current_delay', -1))
        self._diag['delay_samples'] = cur_delay
        if cur_delay >= 0 and self._r7_prev_delay >= 0:
            self._diag['delay_delta'] = cur_delay - self._r7_prev_delay
        else:
            self._diag['delay_delta'] = 0
        self._r7_prev_delay = cur_delay

        cur_div = getattr(self, '_round3_div_counts',
                          {'delay_first': 0, 'delay_shift': 0, 'epv': 0, 'shadow_rise': 0})
        for _src in ('delay_first', 'delay_shift', 'epv', 'shadow_rise'):
            self._diag[f'event_{_src}'] = bool(cur_div.get(_src, 0) > self._r7_prev_div_counts.get(_src, 0))
        self._r7_prev_div_counts = dict(cur_div)

        _filt = getattr(self, 'filter', None)
        self._diag['p_max_override_remaining'] = int(getattr(_filt, '_p_max_override_frames', 0)) if _filt is not None else 0
        self._diag['p_floor_beta_remaining'] = int(getattr(_filt, '_p_floor_beta_frames', 0)) if _filt is not None else 0

        self._diag['filter_once_converged'] = bool(self._filter_once_converged)
        self._diag['filter_converged_now'] = bool(self._convergence.converged)
        self._diag['epc_render_forced_remaining'] = int(getattr(self, '_epc_render_forced_remaining', 0))

        try:
            _np_eps = 1e-12
            _mic_pwr = float(np.mean(near_end ** 2))
            _nores_pwr = float(np.mean(raw_output ** 2))
            _final_pwr = float(np.mean(final_output ** 2))
            self._diag['mic_power_frame'] = _mic_pwr
            self._diag['nores_output_power'] = _nores_pwr
            self._diag['final_output_power'] = _final_pwr
            self._diag['res_required_gain'] = float(np.sqrt(_final_pwr / max(_nores_pwr, _np_eps)))
            self._diag['nores_echo_proxy'] = float(np.sqrt(_nores_pwr / max(_mic_pwr, _np_eps)))
        except (NameError, AttributeError):
            pass
        # ── end Round 7 trace ──

        # Record DTD confidence for plotting
        self.confidence_history.append(self.get_dtd_confidence())

        result = final_output.astype(np.float32)
        if _res_context is not None:
            return (result, _res_context)
        return result

    def get_diagnostics(self) -> dict:
        """Return per-frame diagnostic dict (latest values)."""
        return self._diag.copy()

    def get_erle(self) -> float:
        """Return cumulative ERLE (full-segment average)."""
        eps = 1e-10
        if self.near_power_sum < eps and self.error_power_sum < eps:
            return 0.0
        return 10 * np.log10((self.near_power_sum + eps) / (self.error_power_sum + eps))

    def get_erle_instant(self) -> float:
        """Return instantaneous ERLE (EMA-smoothed)."""
        eps = 1e-10
        if self.near_power < eps and self.error_power < eps:
            return 0.0
        return 10 * np.log10((self.near_power + eps) / (self.error_power + eps))

    def dump_p53_trace(self, path: str) -> int:
        """P53 Step 0: dump captured per-frame innovation-audit rows to .npz.

        Returns the number of frames written. No-op (returns 0) if the flag
        was off or the filter is not PBFDKF. Audit-only; never reads in
        production. See docs/p53_design_lock.md §2.
        """
        if not isinstance(self.filter, PBFDKF):
            return 0
        rows = getattr(self.filter, '_p53_innovation_trace', [])
        if not rows:
            return 0
        n_frames = len(rows)
        n_freqs = self.filter.n_freqs
        cols = {}
        for k in rows[0].keys():
            arr = np.empty((n_frames, n_freqs), dtype=np.float32)
            for i, r in enumerate(rows):
                arr[i] = r[k]
            cols[k] = arr
        os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
        np.savez(path, **cols)
        return n_frames

    def dump_regime_trace(self, path: str) -> int:
        """P52 A.0R.2: dump captured per-frame regime trace rows to .npz.

        Returns the number of rows written. No-op (returns 0) if the flag was
        off and the buffer is empty. Audio-passive: dump only reads the
        already-populated `_regime_trace_rows` list.

        Columns (one array per key, shape (n_frames,) with dtype matching the
        first row's value type):
          frame, boost_q_fired, reverse_copy_fired, main_paused_fired,
          w_l2_before, w_l2_after, q_max_before, q_max_after,
          shadow_w_l2_before, shadow_w_l2_after,
          erle_main_before, erle_main_after,
          copy_counter, copy_err_baseline
        """
        rows = self._regime_trace_rows
        if not rows:
            return 0
        cols = {}
        for k in rows[0].keys():
            cols[k] = np.array([r[k] for r in rows])
        os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
        np.savez(path, **cols)
        return len(rows)

    def is_dtd_active(self) -> bool:
        return self.get_dtd_confidence() > 0.5

    def get_dtd_confidence(self) -> float:
        conf_div = self.dtd_divergence.confidence if self.dtd_divergence else 0.0
        conf_coh = self.dtd_coherence.confidence if self.dtd_coherence else 0.0
        # #3: Same logic as _compute_mu_scale — coherence primary
        if conf_coh > 0.1:
            return conf_coh
        return max(conf_div, conf_coh)

    def get_filter_state(self) -> AecFilterState:
        """Return the highest-priority active filter state as AecFilterState enum.

        B2 note: This is the *public* API state (7 values: WARMUP, DIVERGED,
        EPC_RECOVERY, DT_ACTIVE, STATIONARY_FAR, CONVERGED, CONVERGING).
        It is distinct from the internal P3f diagnostic string stored in
        _diag['filter_state'] / _prev_filter_state which uses a different
        vocabulary ('idle', 'startup', 'diverged', 'suspicious_dt',
        'refined_usable', 'coarse_learning') and is only used inside the
        filter-state computation block and shadow-mu scheduling.
        """
        if self._warmup_frames > 0:
            return AecFilterState.WARMUP
        if self._divergence_indicator > 0.6:
            return AecFilterState.DIVERGED
        if self.epc_active:
            return AecFilterState.EPC_RECOVERY
        if self._diag.get('dt_indicator', 0.0) > 0.5:
            return AecFilterState.DT_ACTIVE
        if self._render_activity.is_stationary:
            return AecFilterState.STATIONARY_FAR
        if self._filter_converged:
            return AecFilterState.CONVERGED
        return AecFilterState.CONVERGING

    def get_stats(self) -> AecStats:
        d = self._diag
        _db = lambda x, floor=1e-10: 10.0 * np.log10(max(x, floor))
        hop_s = self._hop_size / self.config.sample_rate
        delay_samples = int(self._current_delay) if self._current_delay >= 0 else 0
        delay_ms = delay_samples / self.config.sample_rate * 1000.0
        return AecStats(
            frame_count=self._frame_count,
            time_s=self._frame_count * hop_s,
            filter_state=self.get_filter_state(),
            filter_converged=self._filter_converged,
            filter_once_converged=self._filter_once_converged,
            warmup_remaining=self._warmup_frames,
            erle_inst_db=self.get_erle_instant(),
            erle_windowed_db=d.get('erle_windowed', 0.0),
            erle_cumulative_db=self.get_erle(),
            erl_db=_db(self._erl_estimate, 1e-6),
            divergence=self._divergence_indicator,
            epc_active=self.epc_active,
            epv_ratio=d.get('epv_gain_ratio', 1.0),
            mu_scale=d.get('mu_scale', 1.0),
            filter_w_norm=d.get('filter_w_norm', 0.0),
            shadow_w_norm=d.get('shadow_w_norm', 0.0),
            dt_confidence=self.get_dtd_confidence(),
            dt_from_energy=self._dt_from_energy,
            dt_from_shadow=self._dt_from_shadow,
            dt_from_coherence=d.get('dt_from_coherence', 0.0),
            dt_active=self.is_dtd_active(),
            far_power_db=_db(self._far_power_ema),
            mic_power_db=_db(self._mic_power_ema),
            error_power_db=_db(self.error_power),
            far_activity=d.get('far_activity', 0.0),
            saturation_level=self._saturation_level,
            delay_samples=delay_samples,
            delay_ms=delay_ms,
            shadow_advantage=self._shadow_advantage,
            shadow_copy_count=self._regime_handler.copy_counter,
            main_paused=self._regime_handler.main_paused,
            res_gain_mean_db=_db(d.get('res_gain_mean', 1.0)),
            res_using_render=d.get('using_render_based', False),
            echo_psd_mean_db=_db(d.get('echo_psd_mean', 1e-10)),
            error_psd_mean_db=_db(d.get('error_psd_mean', 1e-10)),
        )

    def GetStats(self) -> AecStats:
        return self.get_stats()

    def enable_res_audit(self) -> None:
        """Enable durable RES audit counter substrate (Phase 3B v3 S7+).

        No-op when RES is disabled (self.res is None).
        """
        if self.res is not None:
            self.res.enable_audit_counters()

    def get_res_audit(self):
        """Return RES audit counter dict (or None if not enabled / RES disabled)."""
        if self.res is None:
            return None
        return self.res.get_audit_counters()


class AecDebugLogger:
    """Periodically logs AEC internal state to console.

    Usage::

        aec = AEC(config)
        logger = AecDebugLogger(aec, log_interval_s=1.0)
        # inside processing loop:
        output = aec.process(mic, ref)
        logger.update()
    """

    _HEADER = (
        "  t(s)  | state          |fltr| epc |dt_conf|ERLE_i|ERLE_w|  ERL |"
        "far_dB|mic_dB|mu_scl|shd_adv|sat | delay"
    )
    _SEP = "-" * len(_HEADER)

    def __init__(self, aec: 'AEC', log_interval_s: float = 1.0):
        self._aec = aec
        self._interval_frames = max(1, int(log_interval_s / (aec._hop_size / aec.config.sample_rate)))
        self._tick = 0
        self._header_interval = 20

    def update(self) -> None:
        self._tick += 1
        if self._tick % self._interval_frames != 0:
            return
        s = self._aec.get_stats()
        line_no = self._tick // self._interval_frames
        if line_no % self._header_interval == 1:
            print(self._SEP)
            print(self._HEADER)
            print(self._SEP)
        conv = 'Y' if s.filter_converged else 'N'
        epc  = 'Y' if s.epc_active else 'N'
        print(
            f"{s.time_s:7.2f}  | {s.filter_state.value:<14s} | {conv}  | {epc}  |"
            f" {s.dt_confidence:5.3f} |{s.erle_inst_db:6.1f}|{s.erle_windowed_db:6.1f}|"
            f"{s.erl_db:6.1f}|{s.far_power_db:6.1f}|{s.mic_power_db:6.1f}|"
            f" {s.mu_scale:5.3f}| {s.shadow_advantage:5.3f} |{s.saturation_level:4.2f}|"
            f" {s.delay_ms:.1f}ms"
        )


def process_wav_files(mic_path: str, ref_path: str, out_path: str,
                      config: Optional[AecConfig] = None, diag: bool = False):
    """Process WAV files through AEC"""
    mic_data, mic_sr = sf.read(mic_path)
    ref_data, ref_sr = sf.read(ref_path)

    if mic_sr != ref_sr:
        raise ValueError(f"Sample rate mismatch: mic={mic_sr}, ref={ref_sr}")

    if mic_data.ndim > 1:
        mic_data = mic_data[:, 0]
    if ref_data.ndim > 1:
        ref_data = ref_data[:, 0]

    num_samples = min(len(mic_data), len(ref_data))
    mic_data = mic_data[:num_samples].astype(np.float32)
    ref_data = ref_data[:num_samples].astype(np.float32)

    print(f"AEC Processing:")
    print(f"  Microphone: {mic_path} ({num_samples} samples)")
    print(f"  Reference:  {ref_path}")
    print(f"  Sample rate: {mic_sr} Hz")
    print(f"  Duration: {num_samples / mic_sr:.2f} seconds")

    if config is None:
        config = AecConfig(sample_rate=mic_sr)
    else:
        # v3.8.x: when external config has sample_rate-dependent auto fields
        # already resolved (frame_size/hop_size/filter_length computed in
        # __post_init__ at construction time), updating sample_rate alone
        # leaves stale sizes. Re-resolve auto fields by reverting them to
        # sentinel and re-running __post_init__.
        if config.sample_rate != mic_sr:
            from dataclasses import replace as _dc_replace
            config = _dc_replace(config,
                                  sample_rate=mic_sr,
                                  frame_size=-1,
                                  hop_size=-1,
                                  filter_length=-1)

    print(f"  Mode: {config.mode.value}")
    print(f"  Step size (mu): {config.mu}")
    print(f"  Filter length: {config.filter_length} samples ({1000 * config.filter_length / config.sample_rate:.1f} ms)")
    print(f"  DTD: {'enabled' if config.enable_dtd else 'disabled'}")
    print(f"  RES: {'enabled' if config.enable_res else 'disabled'}")
    print()

    aec = AEC(config)
    hop_size = aec.hop_size

    output = np.zeros(num_samples, dtype=np.float32)
    processed = 0
    max_erle = 0.0
    dtd_frames = 0

    while processed + hop_size <= num_samples:
        mic_block = mic_data[processed:processed + hop_size]
        ref_block = ref_data[processed:processed + hop_size]

        out_block = aec.process(mic_block, ref_block)
        output[processed:processed + hop_size] = out_block

        if aec.is_dtd_active():
            dtd_frames += 1

        erle = aec.get_erle()
        max_erle = max(max_erle, erle)
        processed += hop_size

        if processed % (mic_sr // 2) == 0:
            if diag:
                d = aec.get_diagnostics()
                t = processed / mic_sr
                g_mean_db = 20 * np.log10(max(d['res_gain_mean'], 1e-10))
                g_min_db = 20 * np.log10(max(d['res_gain_min'], 1e-10))
                eff_gmin_db = 20 * np.log10(max(d['effective_g_min'], 1e-10))
                print(f"[{t:5.1f}s] ERLE={d['erle_inst']:6.1f}dB mu={d['mu_scale']:.2f} "
                      f"far_act={d['far_activity']:.2f} "
                      f"g_mean={g_mean_db:5.1f}dB g_min={g_min_db:5.1f}dB "
                      f"eff_gmin={eff_gmin_db:5.1f}dB "
                      f"conv={'Y' if d['converged'] else 'N'} "
                      f"div={d['divergence']:.2f}")
            else:
                print(f"  Processed: {processed / mic_sr:.1f} s, ERLE: {erle:.1f} dB\r",
                      end='', flush=True)

    print(f"\n\nResults:")
    print(f"  Processed samples: {processed}")
    print(f"  Max ERLE: {max_erle:.1f} dB")
    print(f"  DTD active frames: {dtd_frames} ({100 * dtd_frames * hop_size / max(processed, 1):.1f}%)")

    sf.write(out_path, output, mic_sr)
    print(f"\nOutput written to: {out_path}")


def main():
    parser = argparse.ArgumentParser(
        description='Acoustic Echo Cancellation (AEC)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Filter modes:
    lms     - Time-domain LMS (simplest, fixed step size, mu~0.01)
    nlms    - Time-domain NLMS (normalized, default)
    fdaf    - Frequency-domain Adaptive Filter (single FFT block, no partitions)
    pbfdaf  - Partitioned Block FDAF (NLMS adaptation, multiple partitions)
    pbfdkf  - Partitioned Block FDKF (Kalman adaptation, recommended)
    subband - Alias for pbfdkf (backward compatibility)

Examples:
    python aec.py mic.wav ref.wav output.wav
    python aec.py mic.wav ref.wav output.wav --mode pbfdkf --enable-res
    python aec.py mic.wav ref.wav output.wav --mode pbfdkf --enable-res --preset balanced
    python aec.py mic.wav ref.wav output.wav --mu 0.5 --filter 1024
        """
    )
    parser.add_argument('mic', help='Microphone input WAV file')
    parser.add_argument('ref', help='Reference/loudspeaker WAV file')
    parser.add_argument('output', help='Output WAV file')
    parser.add_argument('--mu', type=float, default=0.3, help='Step size (default: 0.3)')
    parser.add_argument('--filter', type=int, default=0,
                        help='Filter length in samples (default: mode-dependent)')
    parser.add_argument('--mode', choices=['lms', 'nlms', 'fdaf', 'pbfdaf', 'pbfdkf', 'subband'],
                        default='nlms', help='Filter mode (default: nlms)')
    parser.add_argument('--enable-dtd', action='store_true',
                        help='Enable DTD (default: off, shadow filter provides DT protection)')
    # v3.10.4 (F6): use BooleanOptionalAction with default=None so the CLI
    # only overrides preset values when the user explicitly passes the flag.
    # Previously --enable-res / --cng defaulted to False and unconditionally
    # overrode the preset's True values, making `aec.py mic ref out --preset
    # balanced` silently disable RES + CNG.
    parser.add_argument('--enable-res', default=None,
                        action=argparse.BooleanOptionalAction,
                        help='Enable RES post-filter (default: from preset, else off)')
    parser.add_argument('--res-g-min', type=float, default=-20.0, help='RES min gain (dB)')
    parser.add_argument('--cng', default=None,
                        action=argparse.BooleanOptionalAction,
                        help='Enable comfort noise generation in RES (default: from preset, else off)')
    parser.add_argument('--no-td-constraint', action='store_true',
                        help='Disable time-domain constraint on filter weights (diagnostic)')
    parser.add_argument('--preset', choices=['mild', 'soft', 'balanced', 'aggressive', 'maximum'],
                        help='Use preset config (overrides RES/adaptive params)')
    parser.add_argument('--no-shadow', action='store_true', help='Disable shadow filter')
    parser.add_argument('--no-highpass', action='store_true', help='Disable high-pass filter')
    parser.add_argument('--highpass-cutoff', type=float, default=80.0,
                        help='High-pass filter cutoff frequency in Hz (default: 80)')
    parser.add_argument('--no-saturation-detect', action='store_true',
                        help='Disable saturation/clipping detection')
    parser.add_argument('--clear-history', action='store_true',
                        help='Clear TIME/LMS buffer each block (no carry-over)')
    parser.add_argument('--diag', action='store_true',
                        help='Print per-second diagnostic output (ERLE, gains, etc.)')

    args = parser.parse_args()

    # Map mode string to enum
    mode_map = {
        'lms': AecMode.LMS,
        'nlms': AecMode.NLMS,
        'fdaf': AecMode.FDAF,
        'pbfdaf': AecMode.PBFDAF,
        'pbfdkf': AecMode.PBFDKF,
        'subband': AecMode.PBFDKF,  # backward compat
    }

    aec_mode = mode_map[args.mode]

    # Mode-dependent default step size
    mu = args.mu
    if args.mode == 'lms' and args.mu == 0.3:
        mu = 0.01  # LMS needs much smaller step size
    elif args.mode == 'fdaf' and args.mu == 0.3:
        mu = 0.1   # FDAFsingle-block: smaller mu to avoid overshoot

    # filter_length default: auto (-1) or user-specified
    filter_length = args.filter
    if filter_length == 0:
        filter_length = -1  # Auto: 32ms (resolved in __post_init__)

    common_kw = dict(
        mu=mu,
        filter_length=filter_length,
        mode=aec_mode,
        enable_dtd=args.enable_dtd,
        enable_td_constraint=not args.no_td_constraint,
        enable_shadow=not args.no_shadow,
        enable_highpass=not args.no_highpass,
        highpass_cutoff_hz=args.highpass_cutoff,
        enable_saturation_detect=not args.no_saturation_detect,
        clear_filter_history=args.clear_history,
    )
    # Only override preset RES params if user explicitly specified them
    # (don't let CLI default -20dB override preset values like -35dB)
    if not args.preset or args.res_g_min != -20.0:
        common_kw['res_g_min_db'] = args.res_g_min
    # v3.10.4 (F6): only forward enable_res / enable_cng to AecConfig when
    # the user explicitly set them on the CLI. With BooleanOptionalAction +
    # default=None, args.enable_res / args.cng are None unless --[no-]enable-res
    # or --[no-]cng was passed; in that case the preset value (or AecConfig
    # default) is preserved.
    if args.enable_res is not None:
        common_kw['enable_res'] = args.enable_res
    if args.cng is not None:
        common_kw['enable_cng'] = args.cng
    if args.preset:
        config = AecConfig.from_preset(args.preset, **common_kw)
    else:
        config = AecConfig(**common_kw)

    process_wav_files(args.mic, args.ref, args.output, config, diag=args.diag)


if __name__ == '__main__':
    main()
