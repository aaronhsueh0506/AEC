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

__version__ = "3.15.0"

import os
import numpy as np
from collections import deque
from dataclasses import dataclass, field
from typing import List, Optional, Tuple
from enum import Enum
import argparse
import soundfile as sf
# v3.18 Phase C.A — FilterAnalyzer audit-only port. Lazy-imported inside
# AEC.__init__ so flag-OFF AEC instances don't pay scipy.signal import cost.
_FilterAnalyzer = None  # populated on first flag-ON construction
_FilteringQualityAnalyzer = None  # populated on first C.B flag-ON construction


# v3.19 refactor R.3 — enums + value-type dataclasses live under
# python/modules/. Re-imported here so existing callers continue to
# do `from aec import AecMode, AecPreset, AecFilterState, AecStats,
# AecResContext, RenderActivityState, FilterConvergenceState,
# RegimeHandlerDecision, AecEventType, AecEvent, EpcEvent` unchanged.
from modules.enums import (  # noqa: E402
    AecMode, AecPreset, AecFilterState, _FREQ_MODES, _PB_MODES,
)
from modules.dataclasses import (  # noqa: E402
    AecStats, AecResContext, RenderActivityState, FilterConvergenceState,
    RegimeHandlerDecision, AecEventType, AecEvent, EpcEvent,
)

# F3.1-v3: blend weight between mic-excess-ratio and legacy (1-coh²) in dt_per_bin.
# 0.7 keeps excess as dominant signal while capping swing under ERL underestimation.
_BLEND_F31_MIC_EXCESS = 0.7

# v3.19 refactor R.4 — leaf utilities live under python/modules/.
from modules.delay import DelayEstimator  # noqa: E402
from modules.erle import (  # noqa: E402
    FilterErleEstimator, FullbandErleEstimator, compute_erle_confidence,
)
from modules.preprocessing import HighPassFilter, SaturationDetector  # noqa: E402
from modules.filters import NlmsFilter, PBFDAF, PBFDKF  # noqa: E402  # R.5
from modules.dtd import DtdEstimator  # noqa: E402  # R.6
from modules.detectors import (  # noqa: E402  # R.6
    RenderActivityDetector, FilterConvergenceAnalyzer,
    DoubleTalkAnalyzer, FilterPlateauDetector,
)
from modules.epc import (  # noqa: E402  # R.7
    classify_epc_event, EchoPathChangeDetector, PathChangeRegimeHandler,
)
from modules.residual_estimator import ResidualEchoEstimator  # noqa: E402  # R.8
from modules.state import AecState  # noqa: E402  # R.8
from modules.res_filter import ResFilter  # noqa: E402  # R.9


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

    # v3.18 Phase A.2 — shadow filter adaptation class (AEC3 alignment).
    # False (default): shadow uses same class as main (PBFDKF in BALANCED).
    # True: shadow uses PBFDAF (NLMS). Restores structural orthogonality
    #       between shadow and main update direction in high-signal regime
    #       where Kalman gain saturates (X·P·X* >> R → K → 1/X*).
    #       Design lock: docs/v3_18_a1_shadow_nlms_design.md.
    shadow_class_nlms: bool = False
    # NLMS step-size for shadow. Only consumed when shadow_class_nlms=True.
    # AEC3 coarse-filter μ default is 0.5. A.5 tunes via {0.3, 0.5, 0.7} grid.
    shadow_mu_nlms: float = 0.5

    # v3.18 Phase B.2 — Filter misadjustment estimator + ScaleFilter
    # (AEC3 Gap #5/#13). Design lock: docs/v3_18_b1_misadjustment_design.md.
    # Tracks long-term echo/error ratio drift; rescales main filter W when
    # systematic under-modelling is detected. Flag-OFF: byte-equal (no
    # estimator updates, no scale action). Flag-ON: estimator + trigger
    # + scale_filter all active.
    filter_misadjustment_enabled: bool = False
    # Asymmetric EMA for misadjustment smoothing (slow up, fast down).
    filter_misadjustment_alpha_up: float = 0.99
    filter_misadjustment_alpha_dn: float = 0.95
    # Threshold: smoothed ratio < this → under-modelling → trigger.
    filter_misadjustment_threshold: float = 0.5
    # Hangover frames between consecutive ScaleFilter fires.
    filter_misadjustment_hangover_frames: int = 100
    # Stability gate: refined_usable + no EPC must hold this many frames.
    filter_misadjustment_stable_frames: int = 30
    # Scale clamp (conservative — AEC3 doesn't clamp; we do for safety).
    filter_misadjustment_scale_min: float = 0.5
    filter_misadjustment_scale_max: float = 2.0
    # If True: also scale Kalman P by scale² (Option A); else P untouched
    # (Option B, AEC3-aligned default).
    filter_misadjustment_scale_p: bool = False

    # v3.19 Phase 3 — B FilterMisadjustment retry per Phase 3.1 design.
    # When True: gate switches from `_prev_filter_state == 'refined_usable'`
    # (~0-38% coverage in v3.18) to `fq_usable AND reset_done` (52-86%
    # coverage from C.B substrate); threshold switches from
    # `< filter_misadjustment_threshold` (legacy 0.5) to
    # `< filter_misadjustment_threshold_phase3` (default 2.0, calibrated
    # for our `_erl_estimate` budget semantics where smoothed clusters
    # at 10-42× vs AEC3's clustering near 1.0). Requires C.A+C.B+C.C
    # all ON for AecState back-ref. Default-OFF: byte-equal.
    # Design: docs/v3_19_phase3_1_b_retry_design.md.
    filter_misadjustment_use_fq_usable: bool = False
    filter_misadjustment_reset_done_frames: int = 20
    filter_misadjustment_threshold_phase3: float = 2.0

    # v3.18 Phase C.A — FilterAnalyzer audit-only port (AEC3-aligned).
    # When True: instantiates FilterAnalyzer on init; per frame, IFFTs
    # main filter W → time domain → HP 600 Hz → peak detection →
    # `consistent_estimate` boolean. Output exposed via _diag['filter_
    # analyzer_*'] only; no consumer changes behaviour. Pre-bench gate
    # at C.A.3 decides whether to advance to C.B.
    filter_analyzer_enabled: bool = False

    # v3.18 Phase C.B — FilteringQualityAnalyzer audit-only port.
    # Multi-gate usable_linear_estimate combining startup_timer +
    # reset_timer + convergence_seen + not_transparent (far_active_recent).
    # Output exposed via _diag['fq_*'] only; no consumer changes
    # behaviour. Pre-bench gate at C.B.3 decides whether to advance to C.C.
    filter_quality_enabled: bool = False

    # v3.18 Phase C.C — AecState ADT facade (AEC3-aligned read-only).
    # Wraps C.A + C.B + legacy state behind a single AecState class.
    # Initially read-only; consumer migration deferred to C.E.
    # No behaviour change; verification gate at C.C.2.
    aec_state_enabled: bool = False

    # v3.18 Phase C.D-α — leakage_diverged Q-bifurcation trigger
    # (AEC3-aligned). Fires when fq_usable says "filter is trustworthy"
    # but shadow contradicts (shadow_advantage > threshold). On fire:
    # _arc_m_q_boost on refined filter + arms hangover. Requires C.A +
    # C.B + C.C all ON for AecState back-ref to read fq_usable.
    # Design lock: docs/v3_18_c_d1_leakage_diverged_design.md.
    leakage_diverged_enabled: bool = False
    leakage_diverged_threshold: float = 2.0   # shadow_advantage ratio
    leakage_diverged_hangover_frames: int = 100

    # v3.18 Phase C.E — RES filter_converged → fq_usable migration.
    # Surgical first migration: AEC passes fq_usable to RES.process()
    # instead of _filter_converged when flag is ON. Tests Phase C.D-α
    # closeout hypothesis 3 (RES gating bottleneck). Requires C.A+C.B+C.C
    # all ON for the back-ref to fq_usable substrate.
    # Design: docs/v3_18_c_e1_consumer_migration_design.md.
    c_e_res_use_fq_usable: bool = False

    # v3.19 Phase 1 — per-RES-branch C.E migration. Per-branch ablation
    # of the v3.18 C.E whole-arg switch; identifies which RES branches
    # benefit from fq_usable (AEC3-aligned multi-gate) vs hurt from it.
    # All default OFF; byte-equal to v3.18 production. Each flag toggles
    # ONE branch from `_filter_converged` semantics to `fq_usable`.
    # Branches inventoried in docs/v3_19_phase1_1a_resfilter_branch_inventory.md.
    # Design lock: docs/v3_19_phase1_2_per_branch_flag_design.md.
    c_e_branch_force_render_use_fq_usable: bool = False
    c_e_branch_dt_per_bin_use_fq_usable: bool = False
    c_e_branch_coh2_ema_use_fq_usable: bool = False

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

    # v3.17 B.1 — Movement-rate DelayEst (default OFF). When EPC is active or
    # in hangover, override DelayEstimator's `_period_samples` to a faster
    # cadence (`delay_est_period_s_fast`, default 0.25 s = 2× production rate)
    # AND reduce the cross-spectrum EMA alpha to `delay_est_alpha_fast`
    # (default 0.2 vs production 0.6) so the EMA tracks the new echo path
    # faster. Polling alone is insufficient — the EMA itself must converge
    # faster for delay estimates to actually move during movement events.
    # Restore baseline period + alpha when EPC quiet.
    #
    # Origin: v3.16 C6 audit case `0I0XMl3M` showed estimated_delay jumps
    # 1230 → 4132 → 4369 → 2 during fast movement, ERLE p5_bad −49 dB. Filter
    # trained on stale alignment generates 50 dB of artefact. EPC is the
    # cleanest in-production motion proxy (already fires on echo path
    # change events that trigger force_delay() chain).
    #
    # §0.6 metric channel: LINEAR-FILTER PRIMARY (nores listen on 0I0XMl3M
    # + cohort tail regression guard). HARD: cohort tail Δecho ≥ -0.05;
    # SOFT: 60-case AECMOS bucket means stable.
    mov_rate_delay_est_enabled: bool = False
    delay_est_period_s_fast: float = 0.25
    delay_est_alpha_fast: float = 0.2

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

    # v3.15 §1.6 — Arc F: per-band Kalman Q schedule (default OFF, byte-equal).
    #
    # Motivation: §1.1 audit found worst-DT cases stuck in `coarse_learning`
    # 40-99% — Kalman filter does not converge fast enough.  Echo paths
    # change MUCH faster in HF (room reverb, mic placement, gain shifts)
    # than LF (acoustic coupling stable).  Current uniform Q (1.5e-3 across
    # all bins on BALANCED) treats LF/HF identically, leaving HF
    # convergence under-driven.
    #
    # Mechanism: scale `kalman_q_high` and `kalman_q_low` per band:
    #   LF  (0-1 kHz):   slower (e.g. ×0.5)  → keep stable
    #   MF  (1-4 kHz):   baseline (×1.0)
    #   HF  (4-8 kHz):   faster (e.g. ×2.0)  → faster re-lock
    # Band boundaries match Arc P / Arc R (1k / 4k splits).
    #
    # Flag-OFF default: byte-equal to v3.14.  When ON: Q_high / Q_low arrays
    # tilted at AEC.__init__ time; subsequent _update_weights paths
    # (Q modulation, far-activity gating, P-floor coupled to Q_high) all
    # benefit from the tilted profile.
    #
    # §0.6 asymmetric metric rule (linear-filter arc):
    #   nores Δdeg < 0 → automatic FAIL (NE damage permanent)
    #   nores Δecho < 0 → SOFT, allowed if 800-case full-pipeline
    #     AECMOS Δecho ≥ -0.020 (RES recovers)
    #
    # Hard bar: nores listen on cohort tail + 5 movement + 5 transient-
    # onset cases shows audible HF re-lock improvement on ≥ 2 channels;
    # 800-case full-pipeline AECMOS DT Δdeg ≥ -0.005, FS Δecho ≥ -0.020.
    kalman_q_per_band: bool = False
    kalman_q_band_scales: tuple = (0.5, 1.0, 2.0)  # (LF, MF, HF) on Q_high/Q_low

    # v3.15 §1.4 Arc M — movement-aware adaptive Q boost (default OFF).
    #
    # Motivation: Arc F (uniform-time per-band Q tilt) closed CANNOT SHIP
    # because steady-state HF Q boost violates cohort tail (-0.067 Δecho
    # on qNvSMyU vs -0.05 bar) — cohort tail is HF-stability-sensitive.
    # The canonical fix is to apply per-band Q tilt ONLY during transient
    # adaptation events (EPC fires, movement detector triggers), reverting
    # to uniform baseline Q at steady state.
    #
    # Mechanism (Arc M v1, EPC-gated): when EPC detector raises Q via
    # `Q = Q_high.copy()` (existing infrastructure at line ~6035 / 6305 /
    # 6631 / 6659), apply Arc F's per-band scale on top — only for the
    # duration of `_p_max_override_frames` (30 frame countdown).  After
    # countdown, Q decays back via the per-frame `q_scale` modulation that
    # uses `mu_mean`.  Steady-state Q stays uniform (baseline) — cohort
    # tail safe.
    #
    # Requires: `kalman_q_per_band=True` and `kalman_q_band_scales` set.
    # When `arc_m_epc_gated=True`, the per-band tilt at AEC.__init__ time
    # is REPLACED with this gated variant — Q_high is left uniform (cohort
    # tail safe), but the per-band tilt is applied transiently by the EPC
    # rising-edge handlers and decays back via q_scale.
    #
    # §0.6 metric: linear-filter primary (nores listen).  HARD: nores Δdeg
    # ≥ 0; SOFT: nores Δecho ≥ -0.020 (RES rescue).  HARD: full-pipeline
    # AECMOS DT Δdeg ≥ -0.005, FS Δecho ≥ -0.020, cohort tail ≥ -0.05.
    arc_m_epc_gated: bool = False

    # v3.15 §1.5b Arc M.v3 — T-gated rescue retry (default OFF, additive on
    # top of arc_m_epc_gated). When BOTH this flag AND arc_t_cohort_detector
    # are ON, every _arc_m_q_boost(filt) invocation is wrapped with a gate:
    #   if not (arc_m_t_gated_enabled AND _arc_t_cohort_tail_signal):
    #       _arc_m_q_boost(filt)
    # so the per-band Q tilt is suppressed during cohort-tail-signal-asserted
    # windows. Hypothesis: V1's +0.023 DT_movement Δdeg win came from
    # non-cohort-tail EPC windows; V1's FS_movement / cohort tail damage
    # came from cohort-tail EPC windows. T-gating excludes the catastrophe
    # windows while preserving the convergence-recovery wins.
    #
    # Reads `self._arc_t_cohort_tail_signal` (Arc T S1 detector signal).
    # Default-OFF (Arc T flag OFF) holds the field at False every frame,
    # so this gate is byte-equal no-op until BOTH flags ON.
    arc_m_t_gated_enabled: bool = False

    # v3.15 §1.4 Arc G — per-band W reset on detected gain-change drift
    # (default OFF). Mechanism orthogonal to Arc F/M Q-modification trade-off:
    # Arc G targets sudden mic-gain or path discontinuities by zeroing only
    # the affected band's filter weights, leaving steady-state stability of
    # the unaffected bands intact.
    #
    # Detector: maintain a fast per-band ERL EMA (alpha = arc_g_fast_alpha)
    # alongside Arc P's slow per-band ERL.  When ratio max/min > arc_g_drift_ratio
    # on band b AND PathChangeRegimeHandler is NOT currently asserting EPC
    # (avoids double-handling cohort tail catastrophe; PathChangeRegimeHandler
    # is load-bearing on cohort tail per `feedback_aec_code_review_accuracy`),
    # zero W[:, bin_range_for_band_b] and cooldown the band for
    # arc_g_cooldown_frames frames.
    #
    # Requires `f3_1_per_band_erl_adaptive=True` (consumes Arc P infrastructure).
    #
    # §0.6 metric: linear-filter primary (nores listen on 5 gain-change cases).
    # HARD: cohort tail Δecho ≥ -0.05 (must NOT damage catastrophe defence);
    # full-pipeline DT Δdeg ≥ -0.005, FS Δecho ≥ -0.020.
    arc_g_per_band_w_reset: bool = False
    arc_g_fast_alpha: float = 0.85
    arc_g_drift_ratio: float = 4.0
    arc_g_cooldown_frames: int = 100

    # v3.15 §1.5 Arc T — cohort tail real-time detector (default OFF).
    # Computes a per-frame ERL_decile_std proxy = max-over-bands of
    # 10·log10(rolling_max / rolling_min) on EMA-smoothed per-band proxy
    # ERL = mean(res.error_psd[band]) / mean(_long_window_far_psd[band]).
    # Asserts cohort_tail_T when proxy >= arc_t_threshold_hi_db; releases
    # via hysteresis at arc_t_threshold_lo_db over arc_t_hysteresis_frames.
    # Source signal is UN-GATED on _filter_converged because the canonical
    # cohort tail case (qNvSMyU) NEVER reaches refined_usable, so the
    # converged-only per-band ERL block is silent there.
    #
    # Dual role:
    #   1. RES preempt — when arc_t_res_preempt_mode=True AND cohort_tail_T
    #      asserts, force render-based mode + boost over_sub by
    #      arc_t_over_sub_boost (H1+H2 stack per design doc §2.3).
    #   2. §1.5b Arc M.v3 upstream gate — _arc_m_q_boost reads
    #      self._arc_t_cohort_tail_signal and skips per-band Q boost during
    #      the asserted window (rescues Arc M V1 from cohort tail wall).
    #
    # §0.6 metric channel: HYBRID (AECMOS primary, nores secondary).
    # Hard bar: cohort tail Δecho ≥ +0.030, FP rate ≤ 5%.
    arc_t_cohort_detector: bool = False
    arc_t_res_preempt_mode: bool = False
    arc_t_inst_alpha: float = 0.85
    arc_t_window_frames: int = 64
    # T_HI calibrated 2026-05-15 from 8-case S1 validation: TAIL min max_db
    # = 19.15 (Hp5g1asacUCt5rJVLO1FuQ_doubletalk_with_movement); CTRL DT
    # mis-adaptation max = 18.01 (NN7yhG2XTEqq46X8X0yLfA_doubletalk).
    # 18.5 dB is the optimal separator (~0.5 dB margin both sides).
    arc_t_threshold_hi_db: float = 18.5
    # T_LO sets hysteresis hold band; widened from 10 → 13 to reduce long-tail
    # holds on legitimate post-catastrophe-recovery frames.
    arc_t_threshold_lo_db: float = 13.0
    arc_t_hysteresis_frames: int = 200
    arc_t_over_sub_boost: float = 1.3

    # v3.16-A — Arc T S2 H2 fix (`force_render` OR-in). Original Arc T
    # S2 wired `self.res._using_render_based = True` from AEC level when
    # `_arc_t_cohort_tail_signal` asserted, but `compute_residual_echo`
    # state machine (`ResidualEchoEstimator.attribute_legacy`)
    # overwrites that field 1 line later via the local `want_render`
    # decision (closure: docs/v3_15_arc_t_s2_wiring_closure.md).
    # Fix: OR cohort_tail_T INTO the `force_render` decision INSIDE
    # `attribute_legacy` so the render-based path engages from within
    # the state machine. Default OFF → byte-equal flag-OFF; predicted
    # +0.030 cohort tail Δecho when ON. See plan §10 v3.16-A.
    arc_t_force_render_or_in: bool = False

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

    # v3.18 Phase D.1 — Subband NE detector substrate (R4, default OFF).
    #
    # Ports AEC3 `SubbandNearendDetector` (88-line module). Detects
    # nearend speech by structural cue:
    #   sub1 (HF region) quiet  AND  sub2 (voice band) loud
    # i.e. `ne_pow_sub1 < threshold * ne_pow_sub2`  AND  `ne_pow_sub1 >
    # snr_threshold * noise_pow_sub1`.
    #
    # Unlike `DominantNearendDetector` it does NOT consume residual echo;
    # it is purely a structural NE detection (resilient when residual
    # estimate is unreliable, e.g. cohort tail).
    #
    # Default-OFF: byte-equal flag-OFF; state remains constant False.
    # Phase D.1 wires audit-only state `self.res._subband_ne_state`;
    # Phase D.3 consumes it for 2-way mask profile swap.
    #
    # Default bin indices for fs=16k, n_freqs=513 (frame=832 → fft=1024):
    #   sub1 = bins 192-320 (~3-5 kHz; speech HF "breath" region)
    #   sub2 = bins  32-128 (~500 Hz-2 kHz; voice band)
    # Final tuning in Phase D.5.
    subband_ne_detect_enabled: bool = False
    subband_ne_sub1_low: int = 192
    subband_ne_sub1_high: int = 320
    subband_ne_sub2_low: int = 32
    subband_ne_sub2_high: int = 128
    subband_ne_threshold: float = 0.5
    subband_ne_snr_threshold: float = 30.0

    # v3.18 Phase D.2 — Mask profile substrate (R2 — per-bin AEC3-style
    # 2-profile masking threshold tables, default OFF).
    #
    # Builds two per-bin lookup tables `_normal_mask_profile[k]` and
    # `_nearend_mask_profile[k]` at ResFilter init, each holding
    # (enr_transparent[k], enr_suppress[k], emr_transparent[k]).
    # Interpolation is linear LF→HF across `[res_mask_last_lf_band ..
    # res_mask_first_hf_band]` mirroring AEC3 `GainParameters::SetConfig`
    # ([suppression_gain.cc:487](../docs/aec3_extracts/src/aec3/suppression_gain.cc)).
    #
    # Anchor defaults ported verbatim from
    # `api/audio/echo_canceller3_config.h` @ commit 9310b29acd:
    #   normal.mask_lf:  (enr_t=0.3,  enr_s=0.4, emr_t=0.3)
    #   normal.mask_hf:  (enr_t=0.07, enr_s=0.1, emr_t=0.3)
    #   nearend.mask_lf: (enr_t=1.09, enr_s=1.1, emr_t=0.3)
    #   nearend.mask_hf: (enr_t=0.1,  enr_s=0.3, emr_t=0.3)
    # Band defaults scaled from AEC3 65-bin (last_lf=5, first_hf=8 =
    # 625 Hz / 1000 Hz @ fs=16k fft=128) to our 257-bin (20 / 32 @
    # fs=16k fft=512, 31.25 Hz/bin). RES `block_size` = `filter.fft_size`
    # which is 512 for the standard BALANCED + fl=832 config (per
    # python/aec.py:5639 wiring).
    #
    # Phase D.2 substrate: tables built when flag ON; not consumed yet.
    # Phase D.3 wires `_stage_gain_compute` to use these tables via
    # AEC3 `GainToNoAudibleEcho` per-bin formula.
    #
    # The NE↔normal swap key is `_subband_ne_state` (D.1 substrate) AND
    # `effective_dt > 0.3` (echo-aware gate; matches AEC3
    # `DominantNearendDetector` AND-combine). Wired in D.3.
    res_mask_profile_swap_enabled: bool = False
    res_mask_last_lf_band: int = 20
    res_mask_first_hf_band: int = 32
    res_mask_normal_lf: tuple = (0.3, 0.4, 0.3)
    res_mask_normal_hf: tuple = (0.07, 0.1, 0.3)
    res_mask_nearend_lf: tuple = (1.09, 1.1, 0.3)
    res_mask_nearend_hf: tuple = (0.1, 0.3, 0.3)
    # NE-profile activation gate threshold. Profile swaps to `nearend`
    # only when `_subband_ne_state AND effective_dt > res_mask_ne_gate_dt`.
    # 0.3 default (initial D.3 wire) gave −0.140 DT Δdeg / 0 NE Δdeg on
    # 60-case → too loose. D.5 sweeps stricter values.
    res_mask_ne_gate_dt: float = 0.3
    # v3.18 D-Path-D — Asymmetric per-bin overlay mode.
    # 'binary': D.3 atomic swap (entire profile swaps on subband NE state).
    # 'asymmetric': legacy `ne_confidence × interp` runs first, then
    # `normal_profile` overlays on FS-confident bins (coh² high AND
    # effective_dt low AND NOT subband_ne_state). Preserves the D.3
    # FS_static Δecho +0.258 win on confident-FS bins while keeping
    # the legacy ne_confidence floor lift on DT/NE bins.
    res_mask_swap_mode: str = 'binary'
    res_mask_fs_overlay_coh2_min: float = 0.85
    res_mask_fs_overlay_dt_max: float = 0.2

    # v3.18 Phase B1 — Dominant NE detector port (AEC3 default detector).
    # Direct port of `DominantNearendDetector` from
    # docs/aec3_extracts/src/aec3/dominant_nearend_detector.cc.
    # Echo-aware (uses residual_echo_psd) + hysteresis (trigger counter +
    # hold duration + fast-exit on strong echo). Default OFF.
    #
    # LF band default for fs=16k fft=512 (31.25 Hz/bin): bins 4-60 maps
    # to 125-1875 Hz, matching AEC3 65-bin bins 1-15 (125-1875 Hz @
    # fft=128).
    #
    # When `dominant_ne_detect_enabled=True` AND `subband_ne_detect_enabled=True`,
    # the per-frame combined NE state is OR-aggregated:
    # `ne_combined = subband_ne_state OR dominant_ne_state`. Path-D
    # asymmetric mode and D.3 binary mode both read `_ne_combined`.
    dominant_ne_detect_enabled: bool = False
    dominant_ne_lf_low: int = 4
    dominant_ne_lf_high: int = 60
    # Defaults from AEC3 echo_canceller3_config.h. Field trials override
    # to 0.5 (Sensitive) or 0.75 (VerySensitive).
    dominant_ne_enr_threshold: float = 0.25
    dominant_ne_enr_exit_threshold: float = 10.0
    dominant_ne_snr_threshold: float = 30.0
    dominant_ne_trigger_threshold: int = 12
    dominant_ne_hold_duration: int = 50

    # v3.18 Phase F.1 — AEC3-aligned echo-path event classification.
    # When True, every EpcEvent fired by EchoPathChangeDetector is classified
    # into the AEC3 3-tuple `(gain_change, delay_change, clock_drift)` and
    # stored in `_classified_event`. Trace-only in F.1 — no consumer logic
    # changes; asymmetric cascade reset wired in F.2+. Default OFF preserves
    # byte-equal output.
    aec_event_classification_enabled: bool = False

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
                # v3.15 §10.S0b: cohort tail real-time detector promoted to
                # default ON. Signal-only substrate — RES preempt path
                # (arc_t_res_preempt_mode) and arc_m gate (arc_m_t_gated_enabled)
                # both stay default OFF, so detector ON is byte-equal on audio
                # output (only AecStats.cohort_tail_T becomes informative).
                # Enables §1.7 RES audit and v3.16 Phase 3-4 candidates to
                # consume the signal without per-bench env-flag flipping.
                arc_t_cohort_detector=True,
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

    def _arc_m_q_boost(self, filt) -> None:
        """v3.15 §1.4 Arc M: set Q = Q_high.copy(), optionally per-band-tilted.

        When Arc M is enabled (`arc_m_epc_gated=True` in cfg + `kalman_q_per_band`
        also True so the band scale was stored), the EPC rising-edge Q
        boost gets a per-band scale applied to the freshly-copied Q array.
        After `_p_max_override_frames` countdown, q_scale modulation in
        `_update_weights` decays Q back toward baseline behavior — but Q
        itself stays at the tilted value until the next time something
        explicitly resets it.

        When the flag is OFF or `_arc_m_band_scale` was never stored,
        behaviour is identical to legacy `filt.Q = filt.Q_high.copy()`
        (byte-equal).
        """
        if not hasattr(filt, 'Q_high'):
            return
        filt.Q = filt.Q_high.copy()
        _scale = getattr(filt, '_arc_m_band_scale', None)
        if _scale is not None:
            filt.Q = filt.Q * _scale

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

    def _handle_gain_change_soft(self, source: str) -> None:
        """v3.18 Phase F.2 — AEC3-aligned soft reset for gain_change events.

        Mirrors `webrtc::Subtractor::HandleEchoPathChange` gain-only branch
        (subtractor.cc:170-174): only `refined_gains_->HandleEchoPathChange()`
        runs, i.e. the adaptive step-size gain receives a boost so the
        filter re-tracks faster — filter weights / partition / coarse
        filter / aec_state ERLE all preserved.

        Our equivalent: Q-boost on main + shadow (preserves weights);
        no Kalman P relax, no filter-derived-state reset, no ERL cap.

        Wired by F.3 behind `aec_event_classification_enabled` for
        EpcEvent.source ∈ {'epv', 'shadow_rise'}.
        """
        for filt in [self.filter, self.shadow_filter]:
            if filt is not None and hasattr(filt, 'Q'):
                if not (self.config.arc_m_t_gated_enabled
                        and getattr(self, '_arc_t_cohort_tail_signal', False)):
                    self._arc_m_q_boost(filt)
        self._f_e3_handle_epc_fire(source)

    def _handle_delay_change_full(self, source: str) -> None:
        """v3.18 Phase F.2 — AEC3-aligned full reset for delay_change events.

        Mirrors `webrtc::Subtractor::HandleEchoPathChange` delay branch
        (subtractor.cc:148-168 `full_reset`): filter weights zeroed
        (refined + coarse), refined/coarse gains reset to initial,
        partition size reset to initial. Plus aec_state full cascade
        in aec_state.cc.

        Our equivalent: Q-boost + Kalman P relax (30 frames) + Kalman
        P-floor lift (30 frames) + ERL cap + filter-derived-state reset
        + EPC render forced. Matches today's `delay_shift` site exactly.

        Wired by F.3 behind `aec_event_classification_enabled` for
        EpcEvent.source == 'delay'.
        """
        for filt in [self.filter, self.shadow_filter]:
            if filt is not None and hasattr(filt, 'Q'):
                if not (self.config.arc_m_t_gated_enabled
                        and getattr(self, '_arc_t_cohort_tail_signal', False)):
                    self._arc_m_q_boost(filt)
                # Kalman P-override attrs are PBFDKF-only. When
                # shadow_class_nlms=True, the shadow filter is PBFDAF and
                # has no Kalman state — skip the P-override block on it.
                # Guard is filter-type, not flag: keeps PBFDKF byte-equal.
                if isinstance(filt, PBFDKF):
                    filt._p_max_override = 1.0
                    filt._p_max_override_frames = 30
                    filt._p_floor_beta = 1.0
                    filt._p_floor_beta_frames = 30
        self._maybe_mark_diverged(source)
        self._epc_render_forced_remaining = self.config.epc_hangover
        self._erl_estimate = min(self._erl_estimate, 0.3)
        if self.config.f3_1_per_band_erl_adaptive:
            _pb_caps = (self.config.per_band_erl_cap_lf,
                        self.config.per_band_erl_cap_mf,
                        self.config.per_band_erl_cap_hf)
            for _bi, _cap in enumerate(_pb_caps):
                self._per_band_erl[_bi] = min(self._per_band_erl[_bi], _cap)
        if self.config.use_epc_state_reset:
            self._apply_epc_state_reset(source)
        self._f_e3_handle_epc_fire(source)

    # v3.18 Phase B.2/B.3 — Filter misadjustment estimator + ScaleFilter wiring.
    def _update_misadjustment_estimator(self) -> None:
        """Asymmetric EMA on filter_scale_ratio = echo_psd / (far_psd × ERL).

        Tracks long-term W magnitude drift. Slow-up / fast-down EMA biases
        the smoothed value toward sustained under-modelling, not transients.
        Skipped when far is silent (no observation signal).
        """
        if not self.config.filter_misadjustment_enabled:
            return
        if self.filter is None or not hasattr(self.filter, 'echo_spec'):
            return
        _fp_echo = float(np.mean(np.abs(self.filter.echo_spec) ** 2))
        _fp_far  = float(np.mean(np.abs(self.filter.far_spec) ** 2))
        if _fp_far < 1e-10:
            return
        raw_ratio = _fp_echo / (_fp_far * float(self._erl_estimate) + 1e-12)
        if raw_ratio < self._misadjustment_smoothed:
            alpha = self.config.filter_misadjustment_alpha_up
        else:
            alpha = self.config.filter_misadjustment_alpha_dn
        self._misadjustment_smoothed = (
            alpha * self._misadjustment_smoothed
            + (1.0 - alpha) * raw_ratio
        )

    def _check_and_apply_misadjustment_scale(self) -> None:
        """Trigger gate + ScaleFilter fire per AEC3 IsAdjustmentNeeded.

        Gates: filter converged AND refined_usable AND not epc_active AND
        not main_paused AND stable_count ≥ threshold AND hangover == 0.
        On fire: scale = clamp(1.0 / smoothed, scale_min, scale_max);
        filter.scale_filter(scale, scale_p); smoothed→1.0; arm hangover.
        """
        if not self.config.filter_misadjustment_enabled:
            return
        if self._misadjustment_hangover_remaining > 0:
            self._misadjustment_hangover_remaining -= 1
            return
        stable = (
            self._filter_converged
            and getattr(self, '_prev_filter_state', '') == 'refined_usable'
            and not self.epc_active
            and not self._regime_handler.main_paused
        )
        if not stable:
            self._misadjustment_stable_count = 0
            return
        self._misadjustment_stable_count += 1
        if self._misadjustment_stable_count < self.config.filter_misadjustment_stable_frames:
            return
        if self._misadjustment_smoothed >= self.config.filter_misadjustment_threshold:
            return
        # Fire: rescale W (and optionally P).
        proposed_scale = 1.0 / max(self._misadjustment_smoothed, 1e-6)
        scale = max(self.config.filter_misadjustment_scale_min,
                    min(self.config.filter_misadjustment_scale_max,
                        proposed_scale))
        if isinstance(self.filter, PBFDKF):
            self.filter.scale_filter(scale,
                scale_p=self.config.filter_misadjustment_scale_p)
        else:
            self.filter.scale_filter(scale)
        self._misadjustment_smoothed = 1.0
        self._misadjustment_hangover_remaining = (
            self.config.filter_misadjustment_hangover_frames)
        self._misadjustment_fire_count += 1

    # v3.18 Phase C.D-α — leakage_diverged Q-bifurcation trigger.
    def _check_leakage_diverged(self) -> bool:
        """AEC3-aligned leakage_diverged: refined claims usable but
        coarse contradicts → re-learn refined via Q-boost.
        """
        if not self.config.leakage_diverged_enabled:
            return False
        if self._leakage_diverged_hangover > 0:
            self._leakage_diverged_hangover -= 1
            return False
        if self.epc_active or self._regime_handler.main_paused:
            return False
        # Need AecState back-ref AND C.B fq_usable substrate to gate.
        if (self._aec_state is None
                or getattr(self._aec_state, '_aec_ref', None) is None):
            return False
        if not self._aec_state.fq_usable():
            return False
        sh_adv = float(getattr(self._dt_analyzer, 'shadow_advantage', 1.0))
        return sh_adv >= self.config.leakage_diverged_threshold

    def _apply_leakage_diverged(self) -> None:
        """Q-boost refined filter on leakage_diverged fire; arm hangover."""
        self._arc_m_q_boost(self.filter)
        self._leakage_diverged_hangover = (
            self.config.leakage_diverged_hangover_frames)
        self._leakage_diverged_fire_count += 1

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
                # v3.15 §1.6 Arc F: per-band Q schedule (default OFF).
                # Tilt Q_high/Q_low across LF/MF/HF bands using same band
                # boundaries (1k / 4k Hz) as Arc P / Arc R.  When OFF, the
                # uniform fill above stands → byte-equal to v3.14.
                if self.config.kalman_q_per_band:
                    _freq_res = self.config.sample_rate / (2 * (self.filter.n_freqs - 1))
                    _b1k = max(1, min(int(round(1000.0 / _freq_res)),
                                       self.filter.n_freqs - 2))
                    _b4k = max(_b1k + 1, min(int(round(4000.0 / _freq_res)),
                                              self.filter.n_freqs - 1))
                    _lf, _mf, _hf = self.config.kalman_q_band_scales
                    _scale = np.ones(self.filter.n_freqs, dtype=np.float32)
                    _scale[:_b1k] = float(_lf)
                    _scale[_b1k:_b4k] = float(_mf)
                    _scale[_b4k:] = float(_hf)
                    if self.config.arc_m_epc_gated:
                        # Arc M: keep baseline Q uniform (cohort tail safe);
                        # store the per-band scale for transient application
                        # at EPC rising edges. Default Q_high stays uniform.
                        self.filter._arc_m_band_scale = _scale
                    else:
                        # Arc F (standalone, time-invariant): apply tilt now.
                        self.filter.Q_high *= _scale
                        self.filter.Q_low  *= _scale
                        self.filter.Q[:] = self.filter.Q_high
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
                from modules.res_refactored.res_filter_refactored import ResFilterRefactored
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
                subband_ne_detect_enabled=self.config.subband_ne_detect_enabled,
                subband_ne_sub1_low=self.config.subband_ne_sub1_low,
                subband_ne_sub1_high=self.config.subband_ne_sub1_high,
                subband_ne_sub2_low=self.config.subband_ne_sub2_low,
                subband_ne_sub2_high=self.config.subband_ne_sub2_high,
                subband_ne_threshold=self.config.subband_ne_threshold,
                subband_ne_snr_threshold=self.config.subband_ne_snr_threshold,
                res_mask_profile_swap_enabled=self.config.res_mask_profile_swap_enabled,
                res_mask_last_lf_band=self.config.res_mask_last_lf_band,
                res_mask_first_hf_band=self.config.res_mask_first_hf_band,
                res_mask_normal_lf=self.config.res_mask_normal_lf,
                res_mask_normal_hf=self.config.res_mask_normal_hf,
                res_mask_nearend_lf=self.config.res_mask_nearend_lf,
                res_mask_nearend_hf=self.config.res_mask_nearend_hf,
                res_mask_ne_gate_dt=self.config.res_mask_ne_gate_dt,
                res_mask_swap_mode=self.config.res_mask_swap_mode,
                res_mask_fs_overlay_coh2_min=self.config.res_mask_fs_overlay_coh2_min,
                res_mask_fs_overlay_dt_max=self.config.res_mask_fs_overlay_dt_max,
                dominant_ne_detect_enabled=self.config.dominant_ne_detect_enabled,
                dominant_ne_lf_low=self.config.dominant_ne_lf_low,
                dominant_ne_lf_high=self.config.dominant_ne_lf_high,
                dominant_ne_enr_threshold=self.config.dominant_ne_enr_threshold,
                dominant_ne_enr_exit_threshold=self.config.dominant_ne_enr_exit_threshold,
                dominant_ne_snr_threshold=self.config.dominant_ne_snr_threshold,
                dominant_ne_trigger_threshold=self.config.dominant_ne_trigger_threshold,
                dominant_ne_hold_duration=self.config.dominant_ne_hold_duration,
                c_e_branch_dt_per_bin_use_fq_usable=self.config.c_e_branch_dt_per_bin_use_fq_usable,
                c_e_branch_coh2_ema_use_fq_usable=self.config.c_e_branch_coh2_ema_use_fq_usable,
            )
            # v3.16-A — wire force_render OR-in enable flag onto the
            # ResidualEchoEstimator. Reading happens inside
            # `attribute_legacy`; default OFF preserves byte-equal.
            if self.res._residual_est is not None:
                self.res._residual_est._arc_t_force_render_or_in_enabled = bool(
                    self.config.arc_t_force_render_or_in)
                # v3.19 Phase 1 Branch R1 — wire flag onto ResidualEchoEstimator
                # (R1 lives inside attribute_legacy, not ResFilter). Default-OFF.
                self.res._residual_est._c_e_branch_force_render_use_fq_usable = bool(
                    self.config.c_e_branch_force_render_use_fq_usable)
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
            # v3.18 Phase A.2 — shadow class selection.
            # Flag-OFF (default): shadow uses same class as main (PBFDKF in
            # BALANCED). Flag-ON: shadow uses PBFDAF (NLMS), AEC3-aligned.
            if self.config.shadow_class_nlms:
                ShadowClass = PBFDAF
                shadow_mu = self.config.shadow_mu_nlms
            else:
                ShadowClass = FilterClass
                shadow_mu = self.config.mu * self.config.shadow_mu_ratio
            self.shadow_filter = ShadowClass(
                block_size=self.filter.block_size,
                n_partitions=self.filter.n_partitions,
                mu=shadow_mu,
                delta=self.config.delta,
                hop_size=self.filter.hop_size
            )
            self.shadow_filter.enable_td_constraint = self.config.enable_td_constraint
            # PBFDKF shadow: higher Q via ratio for faster tracking.
            # When shadow_class_nlms=True the shadow is PBFDAF (no Q state),
            # so this scaling is skipped — guard is filter-type, not flag.
            if isinstance(self.shadow_filter, PBFDKF):
                self.shadow_filter.Q_high = self.filter.Q_high * self.config.shadow_q_ratio
                self.shadow_filter.Q_low  = self.filter.Q_low  * self.config.shadow_q_ratio
                self.shadow_filter.Q      = self.shadow_filter.Q_high.copy()

        # Echo path change detector (owns active/hangover/EPV-EMAs/prev_total_err)
        self._epc_det = EchoPathChangeDetector(self.config)
        # v3.18 Phase F.1 — latest classified AEC3-aligned event (trace-only
        # in F.1; consumers wired in F.2+). Empty AecEvent when classification
        # flag is OFF or no event fired this frame.
        self._classified_event = AecEvent()

        # v3.18 Phase B.2 — Filter misadjustment estimator state.
        # Lazy-init only when flag is enabled so flag-OFF stays byte-equal
        # (these attrs simply don't exist on baseline-config AEC instances).
        if self.config.filter_misadjustment_enabled:
            self._misadjustment_smoothed = 1.0
            self._misadjustment_stable_count = 0
            self._misadjustment_hangover_remaining = 0
            self._misadjustment_fire_count = 0
            # v3.19 Phase 3 — reset_done counter for fq_usable gate.
            # Increments per frame when no recent reset event; reset to
            # 0 on epc_active / main_paused / leakage_diverged_fired.
            # Used only when filter_misadjustment_use_fq_usable=True.
            self._misadjustment_reset_done_count = 0

        # v3.18 Phase C.A — FilterAnalyzer (audit-only). Lazy-init guards
        # both module import and instantiation so flag-OFF stays byte-equal.
        self._filter_analyzer = None
        if self.config.filter_analyzer_enabled:
            global _FilterAnalyzer
            if _FilterAnalyzer is None:
                from modules.filter_analyzer import FilterAnalyzer as _FA
                _FilterAnalyzer = _FA
            self._filter_analyzer = _FilterAnalyzer(
                sample_rate=self.config.sample_rate)

        # v3.18 Phase C.B — FilteringQualityAnalyzer (audit-only). Same
        # lazy-init pattern as C.A.
        self._filter_quality = None
        if self.config.filter_quality_enabled:
            global _FilteringQualityAnalyzer
            if _FilteringQualityAnalyzer is None:
                from modules.filter_quality import FilteringQualityAnalyzer as _FQA
                _FilteringQualityAnalyzer = _FQA
            self._filter_quality = _FilteringQualityAnalyzer()
        # Per-frame helper: set True inside any EPC fire site this frame.
        # Read by FilteringQualityAnalyzer; ignored when flag is OFF.
        self._epc_reset_fired_this_frame = False

        # v3.18 Phase C.C — AecState extension. The existing AecState class
        # (defined above) is constructed later in __init__ with detector
        # references. When aec_state_enabled=True, we set a back-ref to
        # this AEC on the already-constructed AecState so AEC3-aligned
        # methods (consistent_estimate, fq_usable, etc.) can read from
        # C.A/C.B substrate. The actual wiring happens after AecState
        # construction below (search marker 'AecState back-ref').

        # v3.18 Phase C.D-α — leakage_diverged state.
        if self.config.leakage_diverged_enabled:
            self._leakage_diverged_hangover = 0
            self._leakage_diverged_fire_count = 0

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
        # v3.15 §1.4 Arc G — fast per-band ERL EMA for drift detection
        # (default-OFF; only updated/consumed when arc_g_per_band_w_reset=True).
        self._per_band_erl_fast = np.array([0.1, 0.1, 0.1], dtype=np.float64)
        # Per-band cooldown counter (frames remaining where reset is suppressed
        # after a fire on that band).
        self._arc_g_cooldown = np.zeros(3, dtype=np.int32)
        # Diagnostic counter — total Arc G fires per band over the stream.
        self._arc_g_fire_count = np.zeros(3, dtype=np.int64)
        # v3.15 §1.5 Arc T — cohort tail real-time detector state.
        # All fields stay at init until arc_t_cohort_detector=True; default
        # OFF byte-equal sanity preserved by the outer flag gate at the
        # proxy compute block (after the per-band ERL update loop).
        # _W is sized at init time from arc_t_window_frames so 3 ring buffers
        # can be allocated lazily — we size them here using a default to keep
        # __init__ simple; size matches config.arc_t_window_frames.
        _arc_t_W = int(self.config.arc_t_window_frames)
        self._arc_t_inst_pb_smooth = np.array([0.1, 0.1, 0.1], dtype=np.float64)
        self._arc_t_window_max = np.array([1e-10, 1e-10, 1e-10], dtype=np.float64)
        self._arc_t_window_min = np.array([1e10, 1e10, 1e10], dtype=np.float64)
        self._arc_t_window_buf = [
            deque(maxlen=_arc_t_W),
            deque(maxlen=_arc_t_W),
            deque(maxlen=_arc_t_W),
        ]
        self._arc_t_cohort_tail_signal = False
        self._arc_t_hys_remaining = 0
        self._arc_t_proxy_db_last = 0.0
        # Cumulative diagnostic; only cleared by full AEC.reset(). Mirrors
        # Arc G's _arc_g_fire_count diagnostic surface.
        self._arc_t_fire_count = 0
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
        #
        # NOTE (v3.15 §B6, 2026-05-15): `_prev_filter_state` is the
        # INTERNAL P3f-string state machine (values: 'idle', 'startup',
        # 'diverged', 'suspicious_dt', 'refined_usable', 'coarse_learning').
        # It is distinct from the PUBLIC `AecFilterState` enum (which
        # has values like CONVERGED / WARMUP / EPC_RECOVERY) returned by
        # `get_filter_state()`.  The two state systems serve different
        # consumers — see B2 docblock at AecStats.filter_state and the
        # B4 fix at aec.py:6361 for the load-bearing distinction.
        self._prev_filter_state: str = 'idle'
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
        # v3.18 Phase C.C — AecState back-ref. Marker: AecState back-ref.
        # When aec_state_enabled=True, AecState's AEC3-aligned methods
        # (consistent_estimate, fq_usable, active_render, etc.) read
        # from C.A/C.B substrate via this back-ref. When False, methods
        # fall back to legacy semantics — no behaviour change.
        if self.config.aec_state_enabled:
            self._aec_state._aec_ref = self
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
        # v3.18 Phase C.A — FilterAnalyzer state reset on full AEC reset.
        if getattr(self, '_filter_analyzer', None) is not None:
            self._filter_analyzer.reset()
        # v3.18 Phase C.B — FilteringQualityAnalyzer state reset.
        if getattr(self, '_filter_quality', None) is not None:
            self._filter_quality.reset()
        self._epc_reset_fired_this_frame = False
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
        # v3.15 §1.5 Arc T — clear detector state (full reset; cumulative
        # _arc_t_fire_count IS cleared here per the AEC.reset() contract).
        self._arc_t_inst_pb_smooth[:] = 0.1
        self._arc_t_window_max[:] = 1e-10
        self._arc_t_window_min[:] = 1e10
        for _q in self._arc_t_window_buf:
            _q.clear()
        self._arc_t_cohort_tail_signal = False
        self._arc_t_hys_remaining = 0
        self._arc_t_proxy_db_last = 0.0
        self._arc_t_fire_count = 0

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
        # v3.15 §1.5 Arc T — proxy state is filter-output-derived (reads
        # res.error_psd which is filter-output-derived); reset alongside
        # per-band ERL.  Cumulative fire counter is PRESERVED on partial
        # reset (only AEC.reset() clears it).
        self._arc_t_inst_pb_smooth[:] = 0.1
        self._arc_t_window_max[:] = 1e-10
        self._arc_t_window_min[:] = 1e10
        for _q in self._arc_t_window_buf:
            _q.clear()
        self._arc_t_cohort_tail_signal = False
        self._arc_t_hys_remaining = 0
        self._arc_t_proxy_db_last = 0.0
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
                    if not (self.config.arc_m_t_gated_enabled
                            and getattr(self, '_arc_t_cohort_tail_signal', False)):
                        self._arc_m_q_boost(filt)
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
        # v3.18 Phase C.B — reset per-frame EPC-fired helper. Set True by
        # any EPC fire site (delay_shift / EPV / shadow_rise) below.
        # Read by FilteringQualityAnalyzer; safe no-op when flag is OFF.
        self._epc_reset_fired_this_frame = False

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
                # v3.17 B.1: dynamically override DelayEst period + EMA alpha
                # under EPC (motion proxy). When EPC active or in hangover,
                # switch to fast cadence + faster EMA so the cross-spectrum
                # tracks the new echo path; restore baseline otherwise.
                # Read-modify-write the fields — DelayEstimator reads on
                # every accumulate() call.
                if self.config.mov_rate_delay_est_enabled:
                    _epc_motion = (self.epc_active
                                   or self._epc_det.hangover_count > 0)
                    if _epc_motion:
                        self.delay_est._period_samples = int(
                            self.config.delay_est_period_s_fast
                            * self.config.sample_rate
                        )
                        self.delay_est._alpha = float(
                            self.config.delay_est_alpha_fast
                        )
                    else:
                        self.delay_est._period_samples = int(
                            self.config.delay_est_period_s
                            * self.config.sample_rate
                        )
                        self.delay_est._alpha = 0.6
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
                        _delay_event = self._epc_det.force_delay()
                        # v3.18 Phase F.1 — classify delay-shift event.
                        if self.config.aec_event_classification_enabled:
                            self._classified_event = classify_epc_event(_delay_event)
                        for filt in [self.filter, self.shadow_filter]:
                            if filt is not None and hasattr(filt, 'Q'):
                                if not (self.config.arc_m_t_gated_enabled
                                        and getattr(self, '_arc_t_cohort_tail_signal', False)):
                                    self._arc_m_q_boost(filt)
                                # PBFDKF-only Kalman P-override (NLMS shadow
                                # has no P state — skip cleanly).
                                if isinstance(filt, PBFDKF):
                                    filt._p_max_override = 1.0
                                    filt._p_max_override_frames = 30
                        self._maybe_mark_diverged('delay_shift')
                        self._epc_reset_fired_this_frame = True   # C.B FQA signal
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
                        if not (self.config.arc_m_t_gated_enabled
                                and getattr(self, '_arc_t_cohort_tail_signal', False)):
                            self._arc_m_q_boost(self.filter)
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
                    # undoing F2.3's fast-recovery benefit. PBFDKF-only state
                    # (NLMS shadow under shadow_class_nlms=True has no R / no
                    # _error_psd → skip cleanly).
                    if (self.config.shadow_r_reset_enabled
                            and isinstance(self.shadow_filter, PBFDKF)):
                        self.shadow_filter._error_psd.fill(1e-2)
                        self.shadow_filter.R.fill(1e-2)
                # v3.18 Phase A.3 (corrected 2026-05-16) — AEC3 has no W
                # copy between refined/coarse filters. The reverse_copy
                # mechanism exists today because PBFDKF shadow has P-memory
                # and can get stuck in a wrong basin (sending misleading
                # `shadow_advantage`). NLMS shadow has no P-memory and
                # re-adapts from its own residual; W copy becomes a no-op
                # at best and a perturbation at worst. Skip under flag-ON.
                if (shadow_decision.reverse_copy
                        and not self.config.shadow_class_nlms):
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
                self._epc_reset_fired_this_frame = True   # C.B FQA signal
                if self.config.aec_event_classification_enabled:
                    # v3.18 Phase F.3 — AEC3-aligned: EPV is gain_change → soft
                    # reset only (Q-boost on refined/coarse step-size). Skips
                    # Kalman P relax, ERL cap, state reset; mirrors AEC3
                    # subtractor.cc:170-174 (only refined_gains->HEPC runs).
                    self._handle_gain_change_soft('epv')
                else:
                    for filt in [self.filter, self.shadow_filter]:
                        if filt and hasattr(filt, 'Q'):
                            if not (self.config.arc_m_t_gated_enabled
                                    and getattr(self, '_arc_t_cohort_tail_signal', False)):
                                self._arc_m_q_boost(filt)
                            # PBFDKF-only Kalman P-override (NLMS shadow skip).
                            if isinstance(filt, PBFDKF):
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

            # v3.18 Phase F.1 — AEC3 event classification (trace-only; no
            # consumer logic in F.1, so byte-equal preserved when flag OFF).
            if self.config.aec_event_classification_enabled:
                _evt = epv_event if epv_event.fired and not _epv_suppressed else EpcEvent()
                self._classified_event = classify_epc_event(_evt)

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
                    self._epc_reset_fired_this_frame = True   # C.B FQA signal
                    if self.config.aec_event_classification_enabled:
                        # v3.18 Phase F.3 — AEC3-aligned: shadow_rise is a
                        # gain_change proxy (both errors rising signals filter
                        # mistracking, not delay mis-alignment) → soft reset
                        # only. DTD confidence dampening retained because it
                        # protects the per-frame DT signal regardless of
                        # which reset path runs.
                        if self.dtd_coherence:
                            self.dtd_coherence.confidence *= 0.3
                        self._handle_gain_change_soft('shadow_rise')
                    else:
                        if self.dtd_coherence:
                            self.dtd_coherence.confidence *= 0.3
                        for filt in [self.filter, self.shadow_filter]:
                            if filt and hasattr(filt, 'Q'):
                                if not (self.config.arc_m_t_gated_enabled
                                        and getattr(self, '_arc_t_cohort_tail_signal', False)):
                                    self._arc_m_q_boost(filt)
                        self._maybe_mark_diverged('shadow_rise')
                        # P_MAX relax + P_floor raise: force filter to abandon
                        # stale path estimate. PBFDKF-only (NLMS shadow skip).
                        for filt in [self.filter, self.shadow_filter]:
                            if filt and isinstance(filt, PBFDKF):
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
                # v3.18 Phase F.1 — classify shadow_rise event (post-mask).
                if self.config.aec_event_classification_enabled and rise_event.fired:
                    self._classified_event = classify_epc_event(rise_event)

            # WebRTC-style: no output switching. Main filter output is always used.
            # (Shadow filter drives divergence detection + Q boost + pause, not output selection.)

            # final_output starts from raw_output; RES modifies final_output only
            self._last_raw_output = raw_output  # save for diagnostic (time-domain echo power)
            final_output = raw_output.copy()

            # v3.18 Phase B.2/B.3 — FilterMisadjustmentEstimator + ScaleFilter
            # update. Estimator tracks long-term echo/render ratio; trigger
            # fires scale_filter when stable convergence + persistent under-
            # modelling. Both methods return immediately when flag is OFF
            # (byte-equal preserved). Scale action affects subsequent frames
            # only; current frame's raw_output already computed.
            self._update_misadjustment_estimator()
            self._check_and_apply_misadjustment_scale()

            # v3.18 Phase C.D-α — leakage_diverged check. 5th independent
            # Q-boost trigger; fires when fq_usable says refined is good
            # but shadow_advantage says otherwise. Skipped flag-OFF.
            if self.config.leakage_diverged_enabled:
                if self._check_leakage_diverged():
                    self._apply_leakage_diverged()
                    self._epc_reset_fired_this_frame = True
                    self._diag['leakage_diverged_fired'] = True
                else:
                    self._diag['leakage_diverged_fired'] = False

            # v3.18 Phase C.A — FilterAnalyzer audit-only update. Reads main
            # filter W → time-domain impulse → HP-filter → peak detection +
            # consistency check. Outputs exposed via _diag only; no consumer
            # changes behaviour. Skipped when flag OFF (byte-equal preserved).
            if (self._filter_analyzer is not None
                    and self.filter is not None
                    and hasattr(self.filter, 'W')):
                _W_sum = self.filter.W.sum(axis=0)
                _w_time = np.fft.irfft(_W_sum, self.filter.fft_size).astype(np.float32)
                self._filter_analyzer.update(_w_time)
                self._diag['filter_analyzer_consistent'] = bool(
                    self._filter_analyzer.consistent_estimate)
                self._diag['filter_analyzer_peak_index'] = int(
                    self._filter_analyzer.peak_index)
                self._diag['filter_analyzer_max_gain'] = float(
                    self._filter_analyzer.max_echo_path_gain)

            # v3.18 Phase C.B — FilteringQualityAnalyzer audit-only update.
            # Multi-gate usable_linear_estimate; outputs to _diag['fq_*'].
            # convergence_signal prefers FilterAnalyzer.consistent_estimate
            # (AEC3-aligned semantic) when C.A is available; else falls
            # back to legacy _filter_converged.
            if self._filter_quality is not None:
                _fq_far_active = bool(np.mean(far_end ** 2) > 1e-4)
                if self._filter_analyzer is not None:
                    _fq_conv_signal = bool(self._filter_analyzer.consistent_estimate)
                else:
                    _fq_conv_signal = bool(self._filter_converged)
                self._filter_quality.update(
                    far_active=_fq_far_active,
                    epc_reset_fired=bool(self._epc_reset_fired_this_frame),
                    convergence_signal=_fq_conv_signal)
                self._diag['fq_usable'] = bool(self._filter_quality.usable)
                self._diag['fq_startup_done'] = bool(self._filter_quality.startup_done)
                self._diag['fq_reset_done'] = bool(self._filter_quality.reset_done)
                self._diag['fq_convergence_seen'] = bool(
                    self._filter_quality.convergence_seen)
                self._diag['fq_far_active_recent'] = bool(
                    self._filter_quality.far_active_recent)

            # v3.18 Phase C.C — AecState AEC3-aligned snapshot (audit-only).
            # Only emit when aec_state_enabled (back-ref set) — legacy AEC
            # config skips trace to keep diag dict structure unchanged.
            if (self.config.aec_state_enabled
                    and getattr(self._aec_state, '_aec_ref', None) is not None):
                self._diag['aec_state_snapshot'] = self._aec_state.aec3_snapshot()

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
                        # Arc G — fast EMA + drift detector + per-band W reset.
                        _arc_g_on = self.config.arc_g_per_band_w_reset
                        _arc_g_alpha = self.config.arc_g_fast_alpha
                        _arc_g_ratio = self.config.arc_g_drift_ratio
                        _arc_g_cool = self.config.arc_g_cooldown_frames
                        _arc_g_epc_quiet = (self._epc_render_forced_remaining <= 0)
                        # Decrement cooldown counters each per-band-update frame.
                        if _arc_g_on:
                            for _bi in range(3):
                                if self._arc_g_cooldown[_bi] > 0:
                                    self._arc_g_cooldown[_bi] -= 1
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
                            if _arc_g_on:
                                self._per_band_erl_fast[_bi] = (
                                    _arc_g_alpha * self._per_band_erl_fast[_bi]
                                    + (1.0 - _arc_g_alpha) * _inst_pb
                                )
                                _slow = self._per_band_erl[_bi]
                                _fast = self._per_band_erl_fast[_bi]
                                if (_arc_g_epc_quiet
                                        and self._arc_g_cooldown[_bi] == 0
                                        and _slow > 1e-6 and _fast > 1e-6):
                                    _ratio = max(_fast, _slow) / min(_fast, _slow)
                                    if _ratio >= _arc_g_ratio:
                                        # Drift detected — zero band's W weights
                                        # in main filter, cooldown the band.
                                        if hasattr(self.filter, 'W'):
                                            self.filter.W[:, _bs:_be] = 0.0
                                        self._arc_g_cooldown[_bi] = _arc_g_cool
                                        self._arc_g_fire_count[_bi] += 1
                                        # Snap fast EMA to slow so the next
                                        # frame doesn't immediately re-fire.
                                        self._per_band_erl_fast[_bi] = _slow

                # v3.15 §1.5 Arc T — cohort tail real-time detector. Computes
                # a per-frame ERL_decile_std proxy = max-over-bands of
                # 10·log10(rolling_max / rolling_min) on EMA-smoothed per-band
                # proxy ERL = mean(res.error_psd[band]) / mean(_long_window
                # _far_psd[band]).  UN-GATED on _filter_converged so it fires
                # on cohort tail (qNvSMyU class) where the converged-only
                # per-band ERL block above is silent.  Default OFF byte-equal.
                # See docs/v3_15_arc_t_s1_design.md for derivation.
                #
                # NE-corruption gate (S1 calibration 2026-05-15): on DT cases
                # NE speech inflates error_psd → proxy false-fires. Skip when
                # inst_erl_raw = mic_pwr/far_pwr >= 1.5 ("mic ≥ 1.5× far" =
                # NE-dominant frame, same rule scalar ERL update at line
                # ~6953 uses). Cohort tail (qNvSMyU class) has mic ≈
                # small_ERL × far → inst_erl_raw ≈ 0.3-0.7 < 1.5 ✓, so this
                # gate does NOT block cohort tail.
                _arc_t_inst_erl_raw = mic_pwr / max(far_pwr, 1e-10)
                if (self.config.arc_t_cohort_detector
                        and erl_update_gate
                        and _arc_t_inst_erl_raw < 1.5
                        and self.res is not None
                        and self.res._residual_est is not None
                        and self.res._residual_est._long_window_n_updates >= 100):
                    _err_psd_pb_t = self.res.error_psd
                    _far_lw_pb_t = self.res._residual_est._long_window_far_psd
                    _fpb_t = self.config.sample_rate / float(self.config.fft_size)
                    _b1k_t = max(1, int(round(1000.0 / _fpb_t)))
                    _b4k_t = max(_b1k_t + 1, int(round(4000.0 / _fpb_t)))
                    _n_t = _err_psd_pb_t.shape[0]
                    _b1k_t = min(_b1k_t, _n_t - 2)
                    _b4k_t = min(_b4k_t, _n_t - 1)
                    _alpha_inst_t = self.config.arc_t_inst_alpha
                    _clip_lo_t = self.config.per_band_erl_clip_lo
                    _clip_hi_t = self.config.per_band_erl_clip_hi
                    _ratio_db_max = -1e6
                    _min_window_fill = 32  # half a window
                    for _bi_t, (_bs_t, _be_t) in enumerate(
                            ((0, _b1k_t), (_b1k_t, _b4k_t), (_b4k_t, _n_t))):
                        if _be_t <= _bs_t:
                            continue
                        _far_band_t = float(np.mean(_far_lw_pb_t[_bs_t:_be_t]))
                        if _far_band_t < 1e-10:
                            continue
                        _inst_pb_t = float(np.mean(_err_psd_pb_t[_bs_t:_be_t])) / _far_band_t
                        _inst_pb_t = float(np.clip(_inst_pb_t, _clip_lo_t, _clip_hi_t))
                        # Smooth.
                        self._arc_t_inst_pb_smooth[_bi_t] = (
                            _alpha_inst_t * self._arc_t_inst_pb_smooth[_bi_t]
                            + (1.0 - _alpha_inst_t) * _inst_pb_t
                        )
                        _v_t = self._arc_t_inst_pb_smooth[_bi_t]
                        # Rolling window: ring buffer + np.max/min (W = 64).
                        _ring_t = self._arc_t_window_buf[_bi_t]
                        _ring_t.append(_v_t)
                        if len(_ring_t) >= _min_window_fill:
                            _arr_t = np.asarray(_ring_t, dtype=np.float64)
                            _wmax_t = float(np.max(_arr_t))
                            _wmin_t = max(float(np.min(_arr_t)), 1e-10)
                            self._arc_t_window_max[_bi_t] = _wmax_t
                            self._arc_t_window_min[_bi_t] = _wmin_t
                            _ratio_db_t = 10.0 * np.log10(_wmax_t / _wmin_t)
                            if _ratio_db_t > _ratio_db_max:
                                _ratio_db_max = _ratio_db_t
                    # Hysteresis state machine.
                    if _ratio_db_max >= self.config.arc_t_threshold_hi_db:
                        self._arc_t_cohort_tail_signal = True
                        self._arc_t_hys_remaining = self.config.arc_t_hysteresis_frames
                    elif (self._arc_t_hys_remaining > 0
                            and _ratio_db_max >= self.config.arc_t_threshold_lo_db):
                        self._arc_t_cohort_tail_signal = True
                        self._arc_t_hys_remaining -= 1
                    else:
                        self._arc_t_cohort_tail_signal = False
                        if self._arc_t_hys_remaining > 0:
                            self._arc_t_hys_remaining -= 1
                    self._arc_t_proxy_db_last = (
                        float(_ratio_db_max) if _ratio_db_max > -1e5 else 0.0
                    )
                    if self._arc_t_cohort_tail_signal:
                        self._arc_t_fire_count += 1

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
                # v3.17 A.1.1: over_sub chain (dt_reduction → effective_over_sub
                # → self.res.over_sub) is dead in ENR mode. `self.res.over_sub`
                # only read by gain_type ∈ {'wiener', 'spectral_sub'} branches in
                # ResFilter._stage_gain_compute (lines 3244, 3249, 3251); all 5
                # presets use gain_type='enr'. Skip per-frame computation +
                # assignment in ENR mode to eliminate wasted ops; preserve
                # wiener/spectral_sub behaviour if gain_type ever changes.
                _over_sub_live = self.config.res_gain_type != 'enr'
                if _over_sub_live:
                    dt_reduction = self.config.res_dt_reduction * dt_indicator
                    effective_over_sub = max(base_over_sub - dt_reduction, 0.5)

                # Divergence indicator EMA (delegated to FilterConvergenceAnalyzer)
                self._convergence.update_divergence(self.near_power, self.raw_error_power)

                if self.res:
                    # v3.16-A — propagate cohort_tail_T signal to RES
                    # residual estimator. Read by `force_render` OR-in
                    # inside `ResidualEchoEstimator.attribute_legacy`;
                    # byte-equal when `arc_t_force_render_or_in=False`.
                    if self.res._residual_est is not None:
                        self.res._residual_est._arc_t_cohort_tail_signal = bool(
                            getattr(self, '_arc_t_cohort_tail_signal', False))
                    # Change D: during EPC render-forced window, force RES
                    # into render-based echo estimate (unreliable filter W).
                    if getattr(self, '_epc_render_forced_remaining', 0) > 0:
                        self._epc_render_forced_remaining -= 1
                        self.res._using_render_based = True
                    # v3.15 §1.5.S2 Arc T — RES preempt mode (H1+H2 stack):
                    # When cohort_tail_T asserts AND arc_t_res_preempt_mode
                    # enabled, force RES into render-based echo estimate
                    # (H2: same defence the EPC render-forced path uses) AND
                    # boost over_sub by arc_t_over_sub_boost (H1: stronger
                    # spectral attenuation across all bins). Default OFF;
                    # byte-equal flag-OFF preserved by the gate.
                    if (self.config.arc_t_res_preempt_mode
                            and getattr(self, '_arc_t_cohort_tail_signal', False)):
                        self.res._using_render_based = True
                        # arc_t_over_sub_boost is part of the dead over_sub chain
                        # in ENR mode (Arc T S2 H1 closure); v3.16-A `force_render`
                        # OR-in is the alive path for ENR.
                        if _over_sub_live:
                            effective_over_sub = effective_over_sub * float(
                                self.config.arc_t_over_sub_boost)
                    if _over_sub_live:
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
                    # v3.18 Phase C.E — RES filter_converged migration.
                    # Flag-OFF: pass legacy _filter_converged (byte-equal).
                    # Flag-ON: pass fq_usable (multi-gate, 52-86% FS/DT
                    # coverage vs ~5% legacy). Substitution only when
                    # AecState back-ref + C.B substrate both available.
                    if (self.config.c_e_res_use_fq_usable
                            and self._aec_state is not None
                            and getattr(self._aec_state, '_aec_ref', None) is not None):
                        _ce_fc_arg = self._aec_state.fq_usable()
                    else:
                        _ce_fc_arg = self._filter_converged
                    final_output = self.res.process(raw_output, eff_echo_spec,
                                                    far_power, self.filter.far_spec,
                                                    filter_converged=_ce_fc_arg,
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
            cohort_tail_T=bool(getattr(self, '_arc_t_cohort_tail_signal', False)),
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
