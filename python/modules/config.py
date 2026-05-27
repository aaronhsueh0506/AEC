"""AecConfig dataclass + 5 preset factories.

Extracted from ``aec.py`` during refactor R.10. ~1,300 lines including
``from_preset()`` classmethod that materialises MILD / SOFT / BALANCED /
AGGRESSIVE / MAXIMUM presets.

Self-contained: numpy + dataclasses + modules.enums.
"""
from dataclasses import dataclass, field

from .enums import AecMode, AecPreset


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

    # RES parameters (v3.21: legacy ResFilter retired; only enable_res +
    # enable_cng + enable_td_constraint remain as gates).
    enable_res: bool = True
    enable_cng: bool = False           # Comfort noise generation in AEC3 post (off by default)
    enable_td_constraint: bool = True  # Time-domain constraint on filter weights
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
    # v3.21 Phase B.1: DEFAULT TRUE. The shadow now plays the AEC3 coarse-
    # filter role (fast NLMS adapter parallel to refined Kalman). Phase B.6
    # adds the poor_coarse_filter_counter + coarse-reset hangover that
    # consumes this dual-filter structure. Set False only for legacy A/B.
    # Design lock: docs/v3_18_a1_shadow_nlms_design.md +
    # ~/.claude/plans/se-aec-aec-main-hazy-lynx.md §Phase B.
    shadow_class_nlms: bool = True
    # NLMS step-size for shadow. Only consumed when shadow_class_nlms=True.
    # AEC3 coarse-filter μ default is 0.5.
    shadow_mu_nlms: float = 0.5

    # v3.18 Phase B.2 — Filter misadjustment estimator + ScaleFilter
    # (AEC3 Gap #5/#13). Design lock: docs/v3_18_b1_misadjustment_design.md.
    # Tracks long-term echo/error ratio drift; rescales main filter W when
    # systematic under-modelling is detected.
    # v3.21 Phase B.5: DEFAULT TRUE. Gate also loosened in orchestrator to
    # drop the `refined_usable` requirement → matches AEC3 IsAdjustmentNeeded.
    filter_misadjustment_enabled: bool = True
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

    # v3.21.10 nores+stress audit 2026-05-22 — AEC3
    # `Subtractor::FilterMisadjustmentEstimator` direction parity (#3a).
    # The legacy v3.18 estimator above measures `echo_psd / (far_psd × ERL)`
    # and fires when it stays < 0.5 (echo estimate UNDER-modelling vs ERL
    # prediction → GROW W). AEC3 measures `e2_refined / y2` (post-filter
    # error vs mic energy, both TIME-domain block sum-of-squares over
    # n_blocks_=4 AEC3 blocks ≈ 16 ms) and fires when inv_misadjustment_ > 10
    # (post-filter error LARGER than mic → SHRINK W). These detect opposite
    # failure modes; v3.23 G.0 A.1 INFORMATIVE_NULL (legacy fired 0/20188) is
    # silence on the under-modelling failure mode — says nothing about
    # AEC3-style over-adaptation/divergence. v3.21.7 partition_summed_x2
    # Cat C stress cluster shows refined PBFDKF locally diverging
    # (UseRefinedOutput cond2 firing 37 % on XRTnTUjU) → this is exactly
    # what AEC3 estimator targets. See [docs/v3_21_10_misadjustment_direction_audit.md](docs/v3_21_10_misadjustment_direction_audit.md).
    #
    # Two-step rollout:
    #   1. `aec3_misadj_trace_enabled` (default OFF) — when ON, computes
    #      AEC3 inv_misadjustment in parallel with legacy. NO behaviour
    #      change. Populates `self._aec3_misadj_trace` counter dict so
    #      offline scripts can inspect frames where AEC3 would fire vs
    #      where legacy did. Byte-equal preserved by gating inside the
    #      legacy methods (legacy path runs first, AEC3 trace logged after
    #      with no side-effect).
    #   2. `use_aec3_filter_misadjustment_parity` (default ON, AEC3 parity)
    #      — legacy path is BYPASSED and replaced by AEC3-parity update +
    #      fire (e² / y² accumulator, > 10 trigger, 2 / sqrt scale, SHRINK
    #      W). The outer state gates (`epc_active` / `main_paused` /
    #      `_filter_converged` / `hangover`) are retained as transient
    #      guards (AEC3 doesn't have them; we keep to match v3.18 B.5
    #      framing). `scale_p` config still respected (default False,
    #      AEC3-aligned — W only, no P).
    #      byte-equal broken — aligned to AEC3 production (2026-05-27).
    #
    # `aec3_misadj_parity_n_hops`: how many hops to accumulate before
    # checking the trigger. AEC3 uses n_blocks_=4 of kBlockSize=64 samples
    # (256 samples = 16 ms at 16 kHz). We use n_hops × hop_size; with the
    # default hop=160, n_hops=2 covers 320 samples = 20 ms wall-clock —
    # closest integer match to AEC3's 16 ms.
    aec3_misadj_trace_enabled: bool = False
    use_aec3_filter_misadjustment_parity: bool = True
    aec3_misadj_parity_n_hops: int = 2

    # v3.21.11 — AEC3 SubtractorOutputAnalyzer `coarse_filter_converged_relaxed`
    # downstream signal parity (#1d). AEC3
    # `subtractor_output_analyzer.cc:50-51` publishes a second coarse-convergence
    # signal with looser thresholds (`e2_coarse < 0.3·y2 AND y2 > 20²·kBlockSize`,
    # vs strict `< 0.05·y2 AND > 50²·kBlockSize`). Only consumer in AEC3 is the
    # HMM `TransparentModeImpl` (transparent_mode.cc:53-100) which uses it as
    # the `any_coarse_filter_converged` observation in its 2-state HMM.
    # Our port uses `LegacyTransparentModeImpl` (Legacy variant ignores the
    # relaxed signal, per transparent_mode.cc:150 `bool /*any_coarse_filter_converged*/`)
    # AND `transparent_mode_enabled` defaults False — so the relaxed signal
    # has NO consumer in our current pipeline. This flag exposes it as
    # audit-only trace + diagnostic field; behaviour-impact requires both
    # porting the HMM variant AND enabling transparent_mode_enabled, neither
    # of which is in v3.21.x scope. Default OFF preserves byte-equal.
    # See [docs/v3_21_11_coarse_relaxed_signal_audit.md].
    coarse_filter_converged_relaxed_enabled: bool = False

    # v3.21.6 Sprint P1 — FilterAnalyzer port (AEC3 filter_analyzer.cc).
    # When True: AecState owns single-channel FilterAnalyzer; per frame
    # IFFTs PBFDAF partitions → 3-tap 600 Hz HPF → region-sweep peak
    # detection → ConsistentFilterDetector. Output routes through
    # FilterDelay.update (analyzer_filter_delay_estimates_blocks) and
    # TransparentMode.update (any_filter_consistent); orchestrator
    # consumes aec_state.min_direct_path_filter_delay() for reverb
    # update _delay_blocks. 800-case bench: FS_static Δecho +0.059 /
    # FS_movement +0.036 vs v3.21.5; DT buckets within ±0.01; NE flat.
    # Indirectly closes v3.21.5 Sprint C reverb-tail blocker.
    filter_analyzer_enabled: bool = True

    # v3.22 Sprint E.1 — NE-presence augmentation using stationary mask
    # (PBFDKF-port intentional divergence; NOT AEC3 parity). When True,
    # SuppressionGain's 4 gain-policy consumer sites (HF cap gate,
    # max-inc selector, LF-smoothing dec selector, ENR-EMR tuning selector)
    # read an augmented NE-presence proxy:
    #   is_nearend_state() OR (stat_mask_frac > stat_aware_ne_proxy_threshold)
    # where stat_mask_frac is the fraction of bins flagged stationary by
    # _aec3_stationarity.band_stationary_mask(). Designed to cover the
    # DominantNearendDetector failure region on stationary-far conditions
    # (v3.21.6 P4 root cause). Default-OFF preserves v3.21.6 byte-equal.
    # Detector hysteresis untouched; public is_dominant_nearend() unchanged.
    # See docs/v3_22_e0_hf_cap_ne_cross_tab_verdict.md.
    e_stat_aware_ne_proxy_enabled: bool = False
    e_stat_aware_ne_proxy_threshold: float = 0.10  # 10% bins masked

    # v3.22 Sprint F — Reverb tail dead-fallback (intentional divergence
    # from AEC3; AEC3 has no equivalent safety net). When the residual-echo
    # estimator's reverb frequency response can't update for an extended
    # streak (FilterAnalyzer preconditions fail on some FS cases — e.g.
    # LN18k5r8 100% tail-dead through entire case, s90M7MOT 73.5%), inject
    # a conservative late-reflection mass into R²_unb so SuppressionGain
    # still sees the late-tail echo presence. Fires only when the dead
    # streak >= reverb_tail_dead_threshold_frames. Strength is multiplied
    # into render_psd as `render_psd * strength`, mimicking ~strength
    # fraction of unsuppressed reverb tail. Conservative initial defaults;
    # 800-case bench drives ship/reject. Default OFF preserves byte-equal.
    # See docs/v3_22_g0_triage_verdict.md (F PROCEED 2/8 cohort signal).
    reverb_tail_dead_fallback_enabled: bool = False
    reverb_tail_dead_threshold_frames: int = 50      # 500 ms at 10 ms hop
    reverb_tail_dead_fallback_strength: float = 0.25  # conservative

    # v3.21.6 Sprint P2 — Re-enable AEC3 Legacy TransparentMode (was
    # hard-disabled in orchestrator with stale rationale citing legacy
    # 10-frame ERLE latch; that latch was retired in v3.21 by the
    # per-frame e²<0.5·y² gate in _aec3_post). When True: AecState
    # instantiates LegacyTransparentMode + threads any_filter_consistent
    # from P1's FilterAnalyzer. P2 also corrects 3 block-unit constants
    # in transparent_mode.py that were left at AEC3's 4ms-block values
    # (now 2.5x'd to match our 10ms-hop wall-clock semantics). Default
    # OFF preserves v3.21.5 byte-equal anchor until cohort verify.
    transparent_mode_enabled: bool = False

    # v3.18 Phase C.B — FilteringQualityAnalyzer audit-only port.
    # Multi-gate usable_linear_estimate combining startup_timer +
    # reset_timer + convergence_seen + not_transparent (far_active_recent).
    # Output exposed via _diag['fq_*'] only; no consumer changes
    # behaviour. Pre-bench gate at C.B.3 decides whether to advance to C.C.
    filter_quality_enabled: bool = False

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

    # v3.21.1 — AEC3 cc:128-138 per-bin H_error refresh.
    # When False: legacy scalar compare
    # `np.sum(_error_psd) ≤ _e2_coarse_for_refresh` selects a single
    # leakage_converged/diverged factor for all 513 bins.
    # When True (default ON, AEC3 parity): per-bin instantaneous compare
    # selects per-bin leakage.
    # v3.21.4 U4.A retest on canonical state: bucket means neutral
    # (±0.007 dB) BUT cohort tail bad (82 echo + 54 deg per-case
    # regressions > 0.05; worst Δdeg −0.437 on LHsrJBRGn). Same Pareto
    # damage as v3.21.1 (HARD-FAIL on cohort). Historical substrate
    # retained as comment context; byte-equal broken — aligned to AEC3
    # production (2026-05-27).
    use_per_bin_h_error_refresh: bool = True

    # v3.21 sub-ladder audit — AEC3 H_error ceiling parity.
    # AEC3 `refined_filter_update_gain.cc` uses `kHErrorCeiling = 2.0` in
    # float[-1,1] audio. Python's `H_ERROR_CEIL_FLOAT = 1e2` traces back to
    # the int16-scaled derivation and is 50× higher. xFk7igec and jtYTdZm3
    # show 40+ % of steady-state frames saturating at 100.0 (excl50 metric),
    # meaning the Python ceiling is actually live. When True (default ON,
    # AEC3 parity): swap the per-hop clip in `_h_error_refresh` from
    # 100.0 → 2.0 (AEC3 target). byte-equal broken — aligned to AEC3
    # production (2026-05-27).
    use_aec3_h_error_ceil: bool = True

    # v3.21.6 nores LF artifact debug 2026-05-22 — AEC3
    # `Subtractor::Process → render_buffer.SpectralSum →
    # RefinedFilterUpdateGain::Compute` parity. AEC3 passes the
    # partition-summed render power `X²[k] = Σ_p |X_buf[p][k]|²`
    # (`render_buffer.cc::SpectralSum`) into `RefinedFilterUpdateGain`,
    # where it is consumed in THREE places:
    #   1. denominator `0.5·H_error·X² + n_part·E²_refined`
    #      (refined_filter_update_gain.cc:113-115),
    #   2. per-bin noise gate `X² >= noise_gate` (cc:104-111),
    #   3. H_error decay `H_error -= 0.5·mu·X²·H_error` (cc:116-119).
    # Our port currently uses `X²_latest = |X_buf[curr_p]|²` (latest hop
    # only), which under-states the true render energy per bin → denom
    # can be too small on LF bins where energy is spread across history
    # → mu blows up → W_lf grows wrong-phase. Switching to summed X²
    # tracks the canonical AEC3 update equation.
    # W update direction stays per-partition `conj(X_buf[p])` — only
    # the per-bin `mu` and decay use the summed X², matching AEC3.
    # Default ON (AEC3 parity); byte-equal broken — aligned to AEC3
    # production (2026-05-27). Independent of
    # `use_per_bin_h_error_refresh` and raw-E² ablations.
    use_partition_summed_x2_for_h_error_gain: bool = True

    # R0.1 — PBFDKF refined filter weight-update noise gate constant correction.
    # AEC3 `refined_filter_update_gain.cc` and `echo_canceller3_config.h` use
    # noise_gate = 20075344 (int16²) for BOTH refined and coarse filter weight-update
    # gates → FILTER_NOISE_GATE_POWER_FLOAT = psd_int16_to_float(20075344) = 0.01870.
    # Our prior default used NOISE_GATE_POWER_FLOAT = psd_int16_to_float(27509562) = 0.02562,
    # ported from AEC3's echo_model suppression path (not the filter weight-update path).
    # This makes the legacy gate 1.37× too tight: bins where AEC3 would update W are hard-zeroed.
    # OFF: NOISE_GATE_POWER_FLOAT (0.02562) — legacy v3.21.6 behaviour.
    # ON (default, AEC3 parity): FILTER_NOISE_GATE_POWER_FLOAT (0.01870) — AEC3 refined
    # filter gate parity. byte-equal broken — aligned to AEC3 production (2026-05-27).
    # Scope: PBFDKF refined filter _update_weights_aec3 only. Does NOT affect
    # residual echo model noise gate path (R0.2, separate audit — different constant).
    # Classification: Class B (wrong constant from suppression path used in filter gate).
    use_aec3_filter_noise_gate_power: bool = True

    # R0.2 — Residual echo model noise gate constant correction.
    # AEC3 echo_model.noise_gate_power = 27509.42f (int16² scale). Python default
    # 27509562.0 is 1000× too large: non-linear path R2 direct-path echo always
    # zeroed (gate fires for amplitude < 5245 int16 vs AEC3's 166 int16). Affects
    # xFk7 DT_mvmt and any case where usable_linear_estimate() = False.
    # Classification: Class B (wrong constant — Python copied suppression-path value
    # that happened to be 1000× larger than the correct echo-model threshold).
    # Default flipped True 2026-05-27 for AEC3 alignment.
    use_aec3_residual_noise_gate: bool = True

    # R0.3 — EchoGeneratingPower render pre-window size correction.
    # AEC3 echo_model.render_pre_window_size = 1, render_post_window_size = 1
    # (residual_echo_estimator.cc:80-86). Python default: pre=0, post=1 (2 blocks).
    # AEC3 uses 3 blocks (delay-1 to delay+1) for max over render history.
    # Classification: Class A (AEC3 parity gap — window size mismatch).
    # Default flipped True 2026-05-27 for AEC3 alignment.
    use_aec3_echo_gen_power_window: bool = True

    # R0.4 — ERLE inst linear quality for reverb model update (continuous 0→1).
    # AEC3 fullband_erle_estimator.cc::UpdateQualityEstimate computes a continuous
    # quality = (erle_log2 - min) / (max - min) with EMA alpha=0.07. This drives
    # reverb_model_estimator update step: new_smoothing = quality * 0.2.
    # Python default: binary 1.0 (converged) or None (skip update).
    # Classification: Class A (AEC3 parity gap — binary vs continuous quality).
    use_aec3_erle_reverb_quality: bool = False

    # v3.21.20 Phase D — AEC3 RefinedFilterUpdateGain::HandleEchoPathChange port
    # (refined_filter_update_gain.cc:53-68). On EPC fire, reset filter H_error
    # to kHErrorInitial (10000) + call_counter to 0 + poor_excitation_counter
    # to 1000-blocks. Re-triggers Fix B startup window so post-EPC re-adaptation
    # uses conservative single-partition X² (avoiding wVYS-style movement-W-
    # shift accumulation under partition-sum X²). Default True since this is
    # AEC3-verbatim alignment.
    use_aec3_handle_echo_path_change: bool = True

    # v3.21.20 Phase E — AEC3 Subtractor::HandleEchoPathChange dispatcher parity
    # (subtractor.cc:HandleEchoPathChange). When set, the orchestrator EPC fire
    # sites call `handle_echo_path_change()` on the shadow PBFDAF in addition
    # to the refined PBFDKF, AND both call paths set W = 0 on delay_change
    # (port of AdaptiveFirFilter::ZeroFilter, adaptive_fir_filter.cc:ZeroFilter,
    # invoked by AdaptiveFirFilter::HandleEchoPathChange when variability is a
    # delay change). Without this flag, Phase D resets only refined-side
    # H_error / counters but leaves W intact (filter retains a partially-stale
    # impulse-response estimate) and the shadow filter receives no EPC signal.
    # Default OFF because W.fill(0) is destructive and may interact with the
    # load-bearing PathChangeRegimeHandler (main_paused / reverse_copy / boost_q)
    # on cohort tail (~7/800 cases). Needs 12-case + 800-case validation before
    # any default flip. M2 (filter W=0) + M3 (shadow handler wiring) atomic.
    use_aec3_zero_filter_on_epc: bool = False

    # v3.21.20 Phase F — AEC3 EchoPathVariability classification parity
    # (subtractor.cc:HandleEchoPathChange + RefinedFilterUpdateGain). AEC3
    # `EchoPathVariability` has two orthogonal fields:
    #   - delay_change != kNone  → calls AdaptiveFirFilter::ZeroFilter (W=0)
    #                              AND RefinedFilterUpdateGain H_error reset
    #   - !gain_change           → RefinedFilterUpdateGain counter reset
    # Real delay change events (delay_first / delay_shift in our pipeline)
    # set delay_change != kNone. Gain change events (EPV — gain ratio detector)
    # set gain_change=True only. Shadow_rise is OUR synthetic "filter
    # mistracking" detector with no AEC3 equivalent; treated as gain_change
    # under AEC3 semantics.
    # Phase D's wiring passes `delay_change=True` at EPV+shadow_rise sites,
    # which is a classification bug — those events are NOT delay changes per
    # AEC3 semantics. Combined with use_aec3_zero_filter_on_epc=True, this
    # bug fires W=0 on gain_change/mistracking events → 60ms blank window +
    # explosive recovery cascade (root cause of Phase E 12-case regression,
    # nVUnxqHLr -0.510, jtYTdZm -0.300; trace evidence in v3.21.20 Phase E
    # Round 2).
    # When this flag is ON:
    # - EPV site:           delay_change=False, gain_change=True (no-op handler)
    # - shadow_rise site:   delay_change=False, gain_change=True (no-op handler)
    # - delay_first site:   ADD handle_echo_path_change(delay_change=True, gain_change=False)
    # - delay_shift site:   ADD same
    # M2 (W=0) only fires on real delay change events under correct
    # classification. Default OFF preserves Phase D wiring (since removing
    # H_error reset at EPV+shadow_rise loses the +0.155 jtYTdZm gain Phase D
    # was providing). Must validate on 12-case before any default flip.
    use_aec3_epc_classification: bool = False

    # v3.21.20 Phase G — isolation experiment. Decouple AecState `gain_change`
    # dispatch from EPV + shadow_rise sites. Architecture audit 2026-05-24
    # found Phase D's load-bearing benefit is filter-side H_error reset only;
    # `_aec3_pending_gain_change = True` at EPV + shadow_rise queues
    # `_aec3_state.handle_echo_path_change(gain_change=True)` which calls
    # `erle_estimator.reset(False)`. The 3-case trace shows `erle_reset_signal`
    # fires identically in A_off and F_off → AecState side is decorative on
    # these synthetic PBFDKF events. When this flag is ON, the orchestrator
    # SKIPS setting `_aec3_pending_gain_change` at EPV and shadow_rise sites
    # ONLY. Filter-side `filter.handle_echo_path_change` and the legacy
    # companion machinery (boost_q / p_override / ERL cap / mark_diverged /
    # f_e3 / _apply_epc_state_reset) are UNCHANGED. delay_first / delay_shift
    # sites are UNCHANGED (they still queue `_aec3_pending_delay_change`).
    # This is an isolation experiment, NOT a ship candidate. Default OFF.
    skip_aec3_gain_change_dispatch_at_epv_shadow_rise: bool = False

    # v3.21 Q1 parity — true-delay transient leakage response (2026-05-25).
    # AEC3 applies SetConfig(refined_initial) on ALL four EPC trigger classes,
    # giving a 2.5 s transient leakage burst (5e-3/block converged, 5e-1/block
    # diverged) after ANY EPC event. Our production only resets H_error on EPV /
    # shadow_rise (Phase D wiring); delay_first / delay_shift get NO leakage
    # response. This flag adds an AEC3-like transient leakage override on REAL
    # delay events (delay_first / delay_shift) only — NOT on EPV / shadow_rise.
    # No H_error hard-reset (that remains EPV/SR-only). Default OFF: byte-equal.
    # Design: docs/v3_21_q1_true_delay_transient_design.md
    use_q1_true_delay_transient_leakage: bool = False
    q1_tdt_transient_hops: int = 250   # 2.5 s at hop=160/sr=16000 (AEC3 initial_state_seconds)
    q1_tdt_smoothing_hops: int = 100   # 1 s smooth back to elevated steady (AEC3 config_change_duration)
    q1_tdt_lc_factor: float = 5.0      # multiplier on lc_steady; 5.0 = AEC3 ported default (Phase 2.1 R1 candidate)
    q1_tdt_ld_factor: float = 5.0      # multiplier on ld_steady; 5.0 = AEC3 ported default
    # Phase 3.1 re-entry guard: terminate Q1 when a non-delay EPC event (EPV or
    # shadow_rise) fires while Q1 is active. Restores cached steady leakage
    # same-hop so the SR/EPV post-hangover Kalman explosion sees the same lc
    # in both ON and OFF variants. Default OFF: byte-equal to Variant C baseline.
    use_q1_terminate_on_non_delay_epc: bool = False

    # v3.21 A.1 — full delay-change chain port (2026-05-25).
    # Implements AEC3's complete echo-path-change cascade for real delay events
    # (delay_first / delay_shift) only. EPV / shadow_rise / gain_change excluded.
    # Under flag ON (default ON, AEC3 parity):
    #   - Steps 3-4: H_error=kHErrorInitial (10000) + poor_excitation/call_counter
    #     reset on refined (PBFDKF) and shadow (PBFDAF) filters
    #   - AecState._full_reset queued for delay_first (fixes production gap;
    #     delay_shift already wired via _aec3_pending_delay_change)
    #   - Step 9: suppression_gain.set_initial_state(True) on delay_change;
    #             set_initial_state(False) on TransitionTriggered
    # byte-equal broken — aligned to AEC3 production (2026-05-27).
    # Design: docs/v3_21_a1_delay_change_chain_design.md §4
    # Gate 0 verdict: docs/v3_21_a1_gate0_trace_verdict.md
    use_full_delay_change_chain: bool = True

    # Gate 0 Variant C — plateau suppression within post-delay initial chain window.
    # When ON (and use_full_delay_change_chain=True): FilterPlateauDetector is
    # suppressed from firing while aec3_state.initial_state_active() is True AND
    # once_converged is False. Plateau fires normally outside that window.
    # Gate fails if initial_state exits before plateau grace window completes (wVYSGVTT:
    # initial_state exits frame 454, plateau would fire frame 462 — 8 frames gap).
    # Classification: Python functional adaptation — NOT a strict AEC3 port.
    # AEC3 has no FilterPlateauDetector; this prevents the Python-only plateau
    # from interfering with the initial convergence window after delay_first.
    # Default OFF: byte-equal; zero effect unless use_full_delay_change_chain=True.
    use_full_delay_plateau_suppression: bool = False

    # Gate 0 Variant C2 — plateau suppression with fixed-hop counter from delay event.
    # When ON (and use_full_delay_change_chain=True): FilterPlateauDetector is
    # suppressed for plateau_suppression_fixed_hops after delay_first / delay_shift,
    # regardless of initial_state_active() window. Once once_converged latches, suppression
    # is released early. Addresses the 8-frame gap failure of Variant C.
    # Classification: Python functional adaptation — NOT a strict AEC3 port.
    # Default OFF: byte-equal; zero effect unless use_full_delay_change_chain=True.
    use_full_delay_plateau_suppression_fixed_hops: bool = False
    plateau_suppression_fixed_hops: int = 500   # 500 hops = 5 s @ 16 kHz/hop=160

    # Gate 0 Variant D1_corrected — SetConfig(refined_initial) leakage-only port.
    # When ON (and use_full_delay_change_chain=True): switches PBFDKF H_error leakage
    # rates to the AEC3 'initial' profile on delay_first, then restores normal rates
    # on ExitInitialState (TransitionTriggered). No W.fill(0) — consistent with AEC3
    # ZeroFilter being a no-op in steady state (current=max=13).
    # Classification: Partial AEC3 alignment — leakage switch only (Category A partial).
    # D2 (full AEC3 initial chain) is documented in v3_21_m_full_delay_d2_design.md
    # but NOT yet implemented; D1_corrected FAIL does NOT close D2.
    # Correct AEC3 source values (echo_canceller3_config.h lines 102-107):
    #   refined_initial.leakage_converged = 0.005f per block → 0.0125 per 10ms hop (5× default)
    #   refined_initial.leakage_diverged  = 0.5f  per block → 1.25  per 10ms hop (5× default)
    #   refined_initial.error_floor       = 0.001f (SAME as normal refined — no change)
    # Prior comment showed 0.01/0.1 per block — those were WRONG source values (WITHDRAWN).
    # Constants aliased to LEAKAGE_CONVERGED/DIVERGED_SETCONFIG_INITIAL in aec3_scale.py.
    # Default OFF: byte-equal; zero effect unless use_full_delay_change_chain=True.
    use_full_delay_setconfig_initial: bool = False

    # v3.21.12 RefinedFilterUpdateGain input-parity audit 2026-05-22 — AEC3
    # `RefinedFilterUpdateGain::Compute` (refined_filter_update_gain.cc:103-107)
    # `mu[k] = H_error[k] / (0.5·H_error[k]·X²[k] + size_partitions · E²_refined[k])`.
    # `E²_refined` is the CURRENT-block `SubtractorOutput.E2_refined` = per-bin
    # `|FFT(e_refined_time_domain)|²` for THIS hop — instantaneous, no smoothing
    # (subtractor.cc:252-257 ZeroPaddedFft + Spectrum). Our default substitutes
    # `self._error_psd` (0.95 EMA, 95-hop time constant ≈ 200 ms). The smoothing
    # lags refined-filter divergence transients, making the denominator track
    # stale residual amplitude during fast-changing regimes (echo path change,
    # double-talk onset, partition_summed_x2 cohort tail).
    # OFF: smoothed `_error_psd` in the `n_part · E²` term — legacy
    # v3.21.6 behaviour.
    # ON (default, AEC3 parity): current-block per-bin `|self.error_spec|²`
    # in the same term — matches AEC3 SubtractorOutput.E2_refined
    # semantically. byte-equal broken — aligned to AEC3 production
    # (2026-05-27).
    # W direction unchanged. Companion to
    # `use_partition_summed_x2_for_h_error_gain`; A/B/C/D variant matrix in
    # docs/v3_21_12_refined_filter_update_gain_input_parity_plan.md.
    use_current_e2_refined_in_h_error_denominator: bool = True

    # v3.21.13 — AEC3 echo_remover.cc:475 `UseLinearFilterOutput` Y-vs-E
    # final-output selection parity. AEC3 selects between linear residual `E`
    # and capture spectrum `Y` as input to `SuppressionFilter.ApplyGain`:
    #
    #     const auto& Y_fft = aec_state_.UseLinearFilterOutput() ? E : Y;
    #     suppression_filter_.ApplyGain(..., Y_fft, ...);
    #
    # `UseLinearFilterOutput()` returns `filter_quality_state_.LinearFilterUsable()
    # && config_.filter.use_linear_filter` — semantically equal to our
    # `_aec3_state.usable_linear_estimate()` when `use_linear_filter` is True
    # (default in AEC3 and our pipeline).
    #
    # Y is `WindowedPaddedFft(y_post_hpf, y_old, sqrt-Hann)` — same windowing
    # / OLA buffer scheme as E. Our equivalent is
    # `filter.error_spec_windowed + filter.echo_spec` (recomposes
    # `rfft(near_buffer[:block_size] × sqrt-Hann, fft_size)` by reversing
    # `error_spec_windowed = near_spec_win - echo_spec` from PBFDKF.process).
    #
    # Our default ALWAYS applies gain to the linear residual `error_spec`,
    # even when `usable_linear_estimate() == False`. This is the v3.21.x
    # production parity gap — when the linear filter is unusable (e.g.
    # pre-convergence, transient, transparent mode), AEC3 falls back to
    # gain-suppressed mic instead of gain-suppressed residual.
    #
    # OFF (default): byte-equal v3.21.6 — always use `error_spec`.
    # ON: per-hop branch on `usable_linear_estimate()`:
    #   True  → `out_base = error_spec` (current behaviour)
    #   False → `out_base = filter.error_spec_windowed + filter.echo_spec` (= Y)
    # Same `gain` applied to either base. Same OLA / synthesis-window path.
    # Does NOT touch v3.21.8 UseRefinedOutput (per-frame coarse-vs-refined
    # within the linear branch); v3.21.13 selects the entire linear vs
    # capture base. The two flags address different AEC3 mechanisms.
    use_linear_filter_output_selection_for_final_output: bool = False

    # v3.21.8 AEC3 UseRefinedOutput + FormLinearFilterOutput parity
    # (2026-05-22). AEC3 `echo_remover.cc:112-147` selects per-frame
    # between the refined-filter linear residual (`e_refined`) and the
    # coarse-filter linear residual (`e_coarse`) so that a diverged
    # refined filter (or moments where the coarse filter is cleaner)
    # does not cascade into AecState / FilterQuality / SuppressionGain.
    # Our prior pipeline propagated the refined `raw_output`
    # unconditionally, which Gate 2 of the v3.21.7 partition_summed_x2
    # 800-case verdict identified as the primary root-cause candidate
    # for the DT no-clean-convergence stress cluster.
    #
    # Predicate (echo_remover.cc:112):
    #   (1) use coarse when e2_coarse < 0.9·e2_refined AND y2 > 30²·N
    #       AND (s2_refined > 60²·N OR s2_coarse > 60²·N)
    #   (2) use coarse when refined diverged: e2_coarse < e2_refined
    #       AND y2 < e2_refined
    #   default: use refined.
    # Transition (echo_remover.cc:134): sample-by-sample linear
    # crossfade between previous-selected and current-selected output.
    # Default ON (AEC3 parity); byte-equal broken — aligned to AEC3
    # production (2026-05-27). See
    # [[project-aec3-use-refined-output-missing]] memory and
    # `docs/v3_21_7_gate2_verdict.md` for full trace evidence.
    use_refined_output_selection_for_linear_path: bool = True

    # v3.21 alignment — FormLinearFilterOutput 30-sample SignalTransition
    # parity (2026-05-26). AEC3 echo_remover.cc:134 / signal_transition.cc
    # crossfades between the previously-selected and current-selected
    # time-domain error over the first kTransitionBlock=30 samples using
    # a (k+1)/(kT+1) ramp, then copies the new selection for the remaining
    # kBlockSize-30 samples. When False: existing hop-level binary
    # spectral switch (legacy v3.21.6 behaviour). Default ON (AEC3 parity);
    # byte-equal broken — aligned to AEC3 production (2026-05-27). Requires
    # `use_refined_output_selection_for_linear_path=True` to have any
    # effect (crossfade is a no-op when selection never changes).
    form_linear_filter_crossfade_enabled: bool = True

    # v3.21.9 AEC3 SubtractorOutput::ComputeMetrics parity for coarse e²
    # block-window consistency (2026-05-22). AEC3
    # `subtractor_output.cc:38-48` computes
    #   e2_coarse = sum(e_coarse[n]² for n in range(kBlockSize))
    # i.e. a time-domain block sum-of-squares over kBlockSize samples,
    # matching the same window as y2 (also block-summed). Our
    # `_aec3_post` convergence check previously reconstructed e2_coarse
    # via Parseval over fft_size = 512 samples while y2 was summed
    # over hop = 160 — a 3.2× window mismatch. The 5% AEC3-intended
    # coarse_conv threshold became an effective 1.6% threshold,
    # making `coarse_conv` fire ~3× less than AEC3 intends.
    #
    # When True, `_aec3_post` reads `self._last_shadow_output_time`
    # (captured at the shadow.process() call in `process()`) and
    # computes e2_coarse = sum(shadow_output_time²) over hop, which
    # is dimensionally consistent with y2 = sum(near_end²) over hop.
    # Default False keeps v3.21.6 baseline byte-equal. Pure formula
    # fix; independent from `use_refined_output_selection_for_linear_path`
    # (no shadow convergence precondition needed; the block-window
    # consistency is mathematically deterministic). See
    # `docs/v3_21_7_gate2_verdict.md` finding #1c.
    use_coarse_e2_time_domain_parity: bool = False

    # v3.21.14 — PBFDAF shadow NLMS AEC3 protection alignment (2026-05-23).
    # AEC3 `CoarseFilterUpdateGain::Compute` (coarse_filter_update_gain.cc:34-82)
    # defines 5 protection mechanisms for the coarse NLMS filter that PBFDKF
    # main has all ported but PBFDAF shadow has NONE. Shadow update directly
    # affects `any_filter_converged` → `convergence_seen` latch → Cat C trap on
    # no-clean-convergence stress cases (XRTnTUjU / MYrVxVEM / nVUnxqHLr).
    # Each flag is an independent AEC3-verbatim port; all five default ON
    # (AEC3 parity); byte-equal broken — aligned to AEC3 production
    # (2026-05-27). Full plan in `~/.claude/plans/se-aec-aec-main-hazy-lynx.md`.
    #
    # A.1: mu denominator uses partition-summed X² = Σ_p |X_buf[p]|² (current
    #      frame, no smoothing). Matches AEC3 `mu[k] = rate / X²[k]` where X²
    #      comes from `render_buffer.SpectralSum` (coarse_filter_update_gain.cc:68).
    #      Legacy uses EMA-smoothed power × n_partitions (α=0.9, ~10-hop lag);
    #      ON gives shadow the same transient response as AEC3 coarse.
    use_partition_summed_x2_for_shadow_mu: bool = True
    # A.2: noise_gate hard-zero. AEC3 sets `mu[k] = 0` when `X²[k] < noise_gate`
    #      (coarse_filter_update_gain.cc:67-71) using the AEC3 coarse filter
    #      constant (aec3_scale.FILTER_NOISE_GATE_POWER_FLOAT = 0.01870,
    #      from 20075344 int16²). T1.2 fix: prior code used NOISE_GATE_POWER_FLOAT
    #      (27509562 = 0.02562) from suppression path — now corrected.
    #      Legacy uses local_floor + global_floor as denominator lower bounds
    #      — never sets mu to zero. ON matches AEC3 hard-gate semantics.
    use_aec3_noise_gate_for_shadow: bool = True
    # A.3: poor_signal_excitation + call_counter startup gate. AEC3 returns
    #      early when `++poor_signal_excitation_counter_ < size_partitions ||
    #      call_counter_ <= size_partitions` (cc:51-61). Our PBFDAF shadow
    #      previously had NO such counters; PBFDKF main has both. ON adds the
    #      counters to shadow + orchestrator wires them from RenderSignalAnalyzer.
    use_poor_excitation_gate_for_shadow: bool = True
    # A.4: narrowband mask. AEC3 calls
    #      `render_signal_analyzer.MaskRegionsAroundNarrowBands(&mu)` (cc:75)
    #      to zero mu in tonal regions. Our PBFDAF shadow previously had no
    #      `_render_signal_analyzer` reference; PBFDKF main has it
    #      (filters.py:307, applied filters.py:440-443). ON wires the same
    #      single-instance RSA to shadow and applies the mask.
    use_narrowband_mask_for_shadow: bool = True
    # A.5: saturation gate. AEC3 returns early when `saturated_capture_signal`
    #      (cc:56-57). Our PBFDAF reads `self._saturated_capture` set by
    #      orchestrator line 2014 but previously did not gate on it; PBFDKF main
    #      does (filters.py:414-418). ON adds the gate to PBFDAF._update_weights.
    use_saturation_gate_for_shadow: bool = True
    # B.R — AEC3-exact poor_coarse_filter rescue copy. AEC3 subtractor.cc:288-307:
    #   condition:  e2_refined < e2_coarse (strict <, no safety margin)
    #   threshold:  5 AEC3 blocks = ceil(5×64/hop_size) = 2 Python hops
    #   on fire:    copy refined W → coarse W (SetFilter); same-hop coarse
    #               update uses E_refined (not E_coarse) for fast catchup
    #   hangover:   coarse_reset_hangover_blocks=25 (config.h:114)
    #               = ceil(25×64/hop_size) = 10 Python hops; coarse/shadow
    #               continues adapting; only refined gets disallow_leakage_diverged
    # Default OFF = existing behaviour: 0.5× safety margin on condition,
    #   5-hop threshold, 16-hop hangover, shadow frozen during hangover.
    # ON = AEC3-exact for condition + threshold + hangover count + hangover
    #   behaviour (no shadow freeze), PLUS Gap C (same-hop E_refined override):
    #   shadow.process(defer_update=True) skips the inline W update so the
    #   orchestrator can drive shadow.complete_update(
    #   error_override=self.filter.error_spec) on copy_fired hops. Off-fire
    #   hops call complete_update(None) which is identical to the inline path.
    #   Wired in orchestrator.py rescue block; PBFDAF / PBFDKF carry matching
    #   error_override surface in _update_weights / _update_weights_aec3.
    use_aec3_poor_coarse_rescue_copy: bool = False

    # v3.21.17 — SignalDependentErleEstimator port (2026-05-23). AEC3
    # `signal_dependent_erle_estimator.{cc,h}` divides the linear filter
    # into sections (non-linear partitioning where lower sections have
    # finer resolution) and tracks per-section per-subband ERLE
    # refinement. 6 subbands (~125-1000 / 1000-2000 / 2000-3000 /
    # 3000-4000 / 4000-6000 / 6000-8000 Hz).
    #
    # `signal_dependent_erle_sections` (default 0 = OFF — SDE not instantiated,
    # byte-equal preserved). When 1: SDE runs but degenerate (single section
    # = no refinement, clamps to per-subband max_erle). When >= 2: per-section
    # ERLE refinement active. AEC3 ships with `num_sections=1` in production
    # (degenerate); higher values are field-trial / multi-channel configs.
    signal_dependent_erle_sections: int = 0
    # SDE per-filter-section block budget. Defaults match our PBFDKF main
    # filter at preset BALANCED (n_partitions=13 for filter_length=832).
    sde_num_blocks: int = 13
    sde_delay_headroom_blocks: int = 0

    # v3.21.6 nores LF artifact debug 2026-05-22 — usable_linear gate-3
    # ablation knobs. See `python/modules/state/filter_quality.py`
    # docstring + [[project-usable-linear-gate3-latch-bug]] memory.
    # Default-OFF preserves AEC3 legacy 4-gate AND verbatim.
    #
    # Bug: gate 3 = `external_delay OR convergence_seen` is too
    # permissive on no-clean-convergence stress cases (XRTnTUjU_DT_static).
    # `convergence_seen` is a binary latch that fires after a single hop
    # of `any_filter_converged` and never resets. Under partition_summed_x2
    # ON, brief 20% refined-convergence on the stress case latches gate 3
    # → usable_linear True 96.6% of utterance → over-suppression.
    #
    # Three independent knobs (combinations allowed):
    #   - `usable_linear_convergence_hops_required`: int (default 0 =
    #     legacy binary latch). When > 0, gate 3 requires the counter
    #     `_convergence_hops_counter ≥ N` instead of "ever fired once".
    #   - `usable_linear_require_filter_analyzer_consistent`: bool. When
    #     True, gate 3 is AND-ed with `filter_analyzer_consistent()`. No
    #     effect when FilterAnalyzer is disabled (passes False AND).
    #   - `usable_linear_disable_external_delay_shortcut`: bool. When
    #     True, the `external_delay is not None` branch of gate 3 is
    #     dropped — convergence is the ONLY way to satisfy gate 3.
    usable_linear_convergence_hops_required: int = 0
    usable_linear_require_filter_analyzer_consistent: bool = False
    usable_linear_disable_external_delay_shortcut: bool = False

    # v3.21.6 nores LF artifact debug 2026-05-22 — trusted external_delay
    # ablation ([[project-usable-linear-gate3-latch-bug]] follow-up). The
    # orchestrator currently constructs the AEC3 `external_delay`
    # DelayEstimate from legacy `_current_delay` whenever
    # `_delay_active=True`, hardcoding quality=REFINED. On no-clean-
    # convergence stress cases (XRTnTUjU_DT_static) the legacy delay
    # tracker latches a delay and never reports None, so the AEC3 gate-3
    # `external_delay is not None` branch is permanently satisfied — even
    # when the actual xcorr confidence has dropped. This semantic
    # mismatch (our "external_delay always present" vs AEC3's "external
    # delay only when an authoritative source reports one") is what makes
    # usable_linear permissive on stress cases.
    #
    # When True: orchestrator emits ext_delay only when
    #   - fixed_delay_samples >= 0 (explicit configured delay), OR
    #   - delay_est.is_solid (xcorr PAR currently >= solid threshold).
    # Otherwise ext_delay = None — let FilterAnalyzer / convergence path
    # decide. Independent of the other gate-3 knobs above.
    usable_linear_trusted_external_delay_only: bool = False

    # High-pass filter (DC blocker + low-freq removal)
    enable_highpass: bool = True
    highpass_cutoff_hz: float = 80.0    # Cutoff freq: removes DC, 50/60Hz hum, rumble
    # Reference-path HPF — kept default ON. v3.19 attempted AEC3-aligned
    # default OFF (commit ab44842) but 800-case AECMOS verdict (CANNOT SHIP,
    # see docs/v3_19_hpf_ref_flip_verdict.md): FS echo gain (+0.017 to +0.022)
    # paired with DT_movement Δdeg −0.034 (~7× worst-case −0.005 hard bar).
    # Pareto trade-off as the user predicted — AEC3's HPF-on-ref protects DT
    # NE preservation by limiting low-freq ref energy reaching the filter.
    # Flag retained for ablation; revisit after Item 2 RES re-audit if RES
    # gains the ability to handle low-freq ref energy without DT damage.
    #
    # 2026-05-16 user-locked: default = False (ref HPF OFF). See
    # `project_aec_hpf_lock.md` memory. Mic-path HPF remains ON. Bench
    # `AEC_HPF_REF=0` env override is now no-op (matches default); any
    # research that needs ref HPF ON must opt-in explicitly via the env or
    # config kwarg.
    enable_highpass_ref: bool = False

    # Saturation / non-linear echo handling
    enable_saturation_detect: bool = True
    saturation_threshold: float = 0.95       # |sample| > threshold → clipping
    saturation_softclip_ref: bool = True     # Soft-clip reference for better filter modeling

    # v3.21 alignment — SaturationDetector subtractor output inputs
    # (2026-05-26). AEC3 SaturationDetector::Update receives
    # subtractor_output.s_refined_max_abs and s_coarse_max_abs (echo
    # estimates: near - e). The orchestrator always passes 0.0 (default
    # kwarg). When True: compute s_refined = near_end − e_refined and
    # s_coarse = near_end − e_coarse time-domain max-abs and feed them to
    # AecState.update(). Trace-only audit until behavioral impact confirmed.
    # Default False: byte-equal to v3.21.6 baseline.
    saturation_subtractor_inputs_enabled: bool = False

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

    # v3.21.2 S1 — HF damage causal chain trace. Captures per-frame the
    # 5-link chain that the HF damage hypothesis runs through:
    # convergence signal -> ERLE -> R^2 -> DominantNearendDetector ->
    # HF cap gate -> per-bin gain output. Recorded at the end of _aec3_post
    # into AEC._hf_chain_trace (list of dicts). Default False — zero
    # overhead and byte-equal preserving. Read-only observer.
    trace_hf_chain: bool = False

    # v3.21.5 Phase 1 Sprint A — AEC3 echo_remover.cc:495-501 port-fidelity
    # fix. Restores the canonical E2 = min(E2, Y2) clamp before passing
    # nearend_spectrum into SuppressionGain. When usable_linear_estimate()
    # is True and PBFDKF residual (error_psd) exceeds mic spectrum
    # (near_psd) on some bins, the unclamped error_psd inflates
    # nearend_pwr → DominantNearendDetector ENR (= echo/nearend) is biased
    # low → detector mis-triggers nearend → SuppressionGain enters
    # nearend_tuning (conservative) → echo leaks through.
    #
    # v3.21.5 ship default = True (cumulative bench PASS:
    # FS_static +0.033 / FS_movement +0.035 dB; DT deg AECMOS-sensitive
    # but not audible per user spectrogram check; see
    # docs/v3_21_5_phase1_a_e2_y2_clamp_verdict.md). Set False to opt back
    # into pre-v3.21.5 behavior (byte-equal vs v3.21.4) for A/B work.
    e2_y2_clamp_enabled: bool = True

    # v3.21.5 Phase 1 Sprint B — AEC3 echo_audibility.h:40-51 +
    # residual_echo_estimator.cc:303-313 port-fidelity flag. AEC3's
    # stationarity-driven R² scaling (0/1 per bin) is gated by
    # aec_state.UseStationarityProperties(), which reads
    # `EchoCanceller3Config::EchoAudibility.use_stationarity_properties`
    # (AEC3 default = false). Our pre-v3.21.5 orchestrator unconditionally
    # zeroed R² on stationary bins whenever the filter had converged,
    # ignoring the config.
    #
    # v3.21.5 Sprint B verdict (CLOSED 2026-05-21, rejected for v3.21.5):
    # Setting default=False (= AEC3 default) gave bucket means
    # FS_static +0.032 / FS_movement +0.048 dB BUT 60+ DT cohort-tail cases
    # with 1-2 dB formant attenuation (xQEUtY2 worst: 6 catastrophic
    # segments, 2.8s of full NE-speech suppression in a 40s case, deg
    # -0.602; full evidence in
    # docs/v3_21_5_phase1_b_stationarity_gate_verdict.md).
    #
    # Root cause: the stationarity zeroing is a load-bearing safety net
    # compensating for our incomplete AEC3 port (missing companion
    # ScaleFilter / FilterMisadjustment that AEC3 uses to keep
    # `is_nearend_state` correctly firing under stationary-far conditions).
    # Without the zeroing, detector mis-fires NE → far-tuning gain → NE
    # speech destroyed on cohort tail.
    #
    # Default TRUE KEEPS the legacy zeroing (load-bearing for cohort tail).
    # Set False to opt into AEC3-faithful behavior (research / A-B only).
    # Re-test scheduled for v3.21.6 P4 AFTER companion mechanisms ported
    # (P1 FilterAnalyzer + P2 transparent_mode audit + P3 EchoAudibility
    # config wiring); see ~/.claude/plans/se-aec-aec-main-hazy-lynx.md.
    #
    # v3.21.6 Sprint P3 — DEPRECATED ALIAS. Canonical control point now
    # lives at ``SuppressorConfig.echo_audibility.use_stationarity_properties``
    # (mirrors AEC3 ``EchoCanceller3Config::EchoAudibility``). The value
    # here is propagated into the nested field at orchestrator init; the
    # top-level alias will be removed entirely in v3.22 Sprint I cleanup
    # after P4 verdict ships. Env hook ``AEC_STATIONARITY_ZERO`` still
    # works via this alias.
    aec3_post_stationarity_zero_enabled: bool = True

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
        # 50% overlap invariant: frame_size = 2 * hop_size. Enforced because the
        # whole STFT analysis/synthesis chain (sqrt-Hann ana × Hann syn) relies
        # on it. hop_size and fft_size can vary; this ratio cannot.
        if self.frame_size != 2 * self.hop_size:
            raise ValueError(
                f"50% overlap invariant violated: frame_size={self.frame_size} "
                f"must equal 2 * hop_size={self.hop_size}"
            )

    @property
    def fft_size(self) -> int:
        # Next power of 2 >= frame_size (= frame_size when frame_size is power of 2)
        n = self.frame_size
        return 1 << (n - 1).bit_length()

    @property
    def n_partitions(self) -> int:
        # Filter is partitioned in hop_size-sized blocks. AEC3 default at 16 kHz
        # is 13 partitions × 64 samples = 832-sample filter; ours at hop=160 is
        # 6 partitions × 160 = 960-sample filter (filter_length 832 rounded up).
        return (self.filter_length + self.hop_size - 1) // self.hop_size

    @property
    def psd_scale(self) -> float:
        # Float[-1,1] → int16² conversion factor. Used to lift our float PSDs to
        # match the AEC3-domain thresholds in `aec3_scale.py`.
        return 32768.0 ** 2

    @classmethod
    def from_preset(cls, preset: 'AecPreset', **kwargs) -> 'AecConfig':
        """Create config from preset with optional overrides.

        Single preset: BALANCED — routes final_output through `_aec3_post`
        (AecState + ResidualEchoEstimator + SuppressionGain + CNG).
        """
        if isinstance(preset, str):
            preset = AecPreset(preset)
        if preset == AecPreset.BALANCED:
            defaults = dict(
                enable_cng=True,
                shadow_q_ratio=3.5,
                shadow_mu_min=0.5,
                warmup_frames=100,
                kalman_q_high=1e-3,
            )
        else:
            defaults = {}
        defaults.update(kwargs)
        return cls(**defaults)
