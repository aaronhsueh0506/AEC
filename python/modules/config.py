"""AecConfig dataclass + BALANCED preset.

Customer-facing / system-parameter surface only. AEC3-alignment flags
that were default-True at release are hard-coded into the call sites
under ``modules.orchestrator`` / ``modules.filters``; closed-substrate
NOSHIP flags are deleted entirely.
"""
from dataclasses import dataclass, field

from .enums import AecMode, AecPreset


@dataclass
class AecConfig:
    """AEC configuration (release surface).

    System sizes are samples; rates Hz. Fields marked INTERNAL are
    preserved for backward compatibility with the BALANCED preset and
    internal call sites — do not document as customer knobs.
    """

    # ── System / framing ────────────────────────────────────────────────
    sample_rate: int = 16000        # 8000 / 16000 / 48000
    frame_size: int = -1            # Auto: sample_rate * 20ms
    hop_size: int = -1              # Auto: frame_size / 2
    filter_length: int = -1         # Auto: 32ms (8k/16k) or 64ms (48k)
    mu: float = 0.3                 # Step size
    delta: float = 1e-8             # Regularization

    # ── DTD subsystem (interface, default disabled) ─────────────────────
    enable_dtd: bool = False
    dtd_hangover_frames: int = 15
    dtd_geigel_threshold: float = 0.5
    dtd_mu_min_ratio: float = 0.05
    dtd_confidence_attack: float = 0.3
    dtd_confidence_release: float = 0.05
    dtd_divergence_factor: float = 1.5
    dtd_coh_alpha: float = 0.85
    dtd_coh_high: float = 0.6
    dtd_coh_low: float = 0.3
    dtd_coh_energy_floor: float = 0.1
    dtd_coh_hangover: int = 3
    dtd_coh_release: float = 0.1
    dtd_coh_abs_floor: float = 1e-6

    # ── Residual / comfort noise ────────────────────────────────────────
    enable_res: bool = True
    enable_cng: bool = False
    # AEC3 comfort-noise floor (dBFS). Float[-1,1] PSD scale.
    comfort_noise_floor_dbfs: float = -96.03406
    enable_td_constraint: bool = True

    # ── v3.22 future work: HF minimum-gain floor during NE-dominant ─────
    # NOT AEC3-strict (AEC3 SG's min_gain = render_limit / R² has the same
    # structural weakness — when R² is huge at HF bins where filter spuriously
    # outputs S²_linear during NE-only periods, min_gain → 0 and HF is
    # painted-black). v3.21 AEC3 surface is exhausted for this symptom.
    #
    # SPEC (when flag ON):
    #   When the SG dominant-nearend detector says NE-state is active, force
    #   ``min_gain[k] >= 10^(hf_min_gain_floor_during_dne_db / 10)`` for all
    #   HF bins (k >= first_hf_band ≈ 1000 Hz @ fft=512). This caps total
    #   HF suppression at the configured dB-floor regardless of the linear
    #   R² estimate. Final amplitude floor = sqrt(power floor).
    #
    # WHY -15 dB:
    #   Current symptom (568_EVB DT→FS→DT2 at 6.5-7.0 s) has gain_HF ≈ -35 dB
    #   median during NE-active periods → audible "painted-black"/"大魔王"
    #   robot artefact on Mandarin "/sì/" fricative HF + formant valleys.
    #   −15 dB floor lifts gain ≥ 0.178 amplitude (= 0.0316 power) which keeps
    #   the NE voice fully audible while still allowing ~5× HF attenuation
    #   when light residual echo bleeds through NE periods.
    #
    # GATING:
    #   Only fires when ``_ne_state_for_gain_rules()`` is True (i.e., DNE
    #   detector currently in NE-state). When DNE=False (echo-dominant
    #   moments), SG's full dynamic range is preserved — no impact on
    #   echo-cancellation aggressiveness during FS or DT-echo-loud periods.
    #
    # Default OFF for v3.21.x byte-equal; intended to flip ON in v3.22 after
    # multi-case verification (incl. confirming no FS-echo regression).
    hf_min_gain_floor_during_dne_enabled: bool = False
    hf_min_gain_floor_during_dne_db: float = -15.0

    # ── v3.22 C: E²/Y² per-bin ERLE gate (doubletalk contamination protection) ──
    # Prevents nearend speech from pulling ERLE down during doubletalk by freezing
    # per-bin ERLE updates when accumulated E²(k)/Y²(k) exceeds the threshold.
    #
    # MECHANISM:
    #   In FS (echo being cancelled): E² << Y² → ratio low → update proceeds.
    #   In DT (nearend inflates error): E² → Y² → ratio rises → freeze ERLE for that bin.
    #   Frozen ERLE holds at pre-DT value → R² = S²/ERLE stays bounded → Wiener gain stable.
    #
    # THRESHOLD=0.5: E² > 0.5×Y² → nearend contributes ≥ 50% of error energy → unreliable.
    # Per-bin: only contaminated bins are frozen; unaffected bins update normally.
    #
    # Advantage over all gain-floor approaches (W4/D1/D5): operates at R² source,
    # not at gain output → no FS false positive from nearend detection signals.
    #
    # Default OFF for byte-equal; bench with AEC_ERLE_E2Y2_GATE=1.
    erle_e2y2_gate_enabled: bool = False
    erle_e2y2_gate_threshold: float = 0.5

    # ── ERLE update gate = AEC3 per-frame SubtractorOutputAnalyzer rule ──
    # The ERLE estimator's update is gated by `converged_filter`. We have been
    # sourcing that from the legacy FilterConvergenceAnalyzer latch (10
    # consecutive far-active frames with inst-ERLE>5 dB) which is near-
    # contaminated and almost NEVER latches in double-talk (~5% of frames) →
    # ERLE frozen at min=1.0 → R²=S²/ERLE=S² inflated → ENR ramp over-suppresses
    # near-end. AEC3 instead gates on a PER-FRAME rule (subtractor_output_analyzer.cc:46):
    #   converged = (e2_refined < 0.5·y2) OR (e2_coarse < 0.05·y2),  y2 > floor
    # which fires on every echo-dominant frame so ERLE earns credit (then holds
    # via the accumulator's converged-gating through near-loud frames). When ON,
    # source the ERLE update gate from this per-frame e2<thr·y2 rule (refined-only;
    # coarse term omitted — main filter error is `error_psd`). Default OFF for
    # byte-equal; bench with AEC_ERLE_SUBCONV=<thr>.
    erle_gate_subtractor_converged: bool = False
    erle_gate_subtractor_threshold: float = 0.5
    # y2 floor (our capture_psd-sum scale) below which a frame can't count as
    # converged (avoids crediting silence). Calibrated empirically.
    erle_gate_subtractor_y2_floor: float = 1.0e6

    # ── v3.22 C': Coherence-based ERLE gate (Γ²_ŶY) ──
    # Gate SubbandErle updates per-bin when Γ²(Ŷ, Y) < threshold.
    # Ŷ = echo estimate (echo_spec), Y = capture (near_spec); both complex.
    # In DT: nearend decorrelates Y from Ŷ → Γ² drops → freeze ERLE.
    # Avoids circular dependency of E²/Y² gate (E is contaminated by nearend).
    # In FS converged: Y ≈ Ŷ → Γ² ≈ 1 → ERLE updates normally.
    # erle_coh_gate_alpha: EMA rate for cross-PSD tracking (0.05 ≈ 200ms).
    # Bench with AEC_ERLE_COH_GATE=<threshold>.
    erle_coh_gate_enabled: bool = True   # v3.22 C' shipped: +0.014 FS_static echo
    erle_coh_gate_threshold: float = 0.5
    erle_coh_gate_alpha: float = 0.05

    # ── v3.22 Layer1: Coherence gain floor (AEC2 NLP-inspired) ──
    # Per-bin gain floor from Γ²(Ŷ, Y): G_floor = sqrt(max(0, 1-Γ²))·strength.
    # 1-Γ² = nearend power fraction at mic = near/(echo+near) (X⊥nearend).
    # FS (converged OR not): Ŷ=H̃X, Y=HX both ∝ X → Coh=1 → 1-Γ²→0 → floor→0
    #   → NO echo suppression impact, NO FS-unconverged false positive.
    # DT: nearend N adds X-uncorrelated component → Γ² drops → floor rises
    #   → nearend preserved. This is AEC2's cohxd-gated NLP relaxation,
    #   delay-compensated for free (Ŷ already time-aligned to Y).
    # Reuses the C' Γ²_ŶY EMA (erle_coh_gate_alpha). Default OFF.
    # Bench with AEC_COH_FLOOR=<strength>.
    coh_gain_floor_enabled: bool = False
    coh_gain_floor_strength: float = 0.5

    # ── v3.22 split min-gain floor (DEFAULT ON) ─────────────────────────────
    # The AEC3 min-gain floor (min_echo_power / R²) collapses to ~0 in DT when
    # ERLE contamination inflates R² — the structural cause of nearend over-
    # suppression (RES cuts nearend up to -8dB; audio-localised). A flat power-
    # domain floor caps the deepest suppression. We split it by far-end activity:
    #   • far-active (FS/DT): -28 dB  — deeper far-active suppression for more echo
    #     cancellation; the per-bin near-blend (soft_nearend_blend_per_bin, ON)
    #     protects near-end frequency-selectively so the deeper floor does not
    #     over-cut near in DT.
    #   • far-silent (pure NE): -12 dB — lifts NE nearend at ZERO echo cost
    #     (no echo present to leak).
    # Routing uses a per-recording LATCH on instantaneous far energy
    # (mean(render_block²) > latch_power, render_block int16-scaled ×32768; the
    # threshold ≈ the orchestrator's own far-active criterion). Latches from the
    # first far frame so FS/DT use the gentler floor throughout (no cold-start
    # leak); only recordings where far is never active stay on the strong floor.
    # 800-case (v3.22.2, far_active -28 + per-bin blend, movement-agnostic):
    #   FS echo 3.549 / DT echo 4.155 / DT deg 2.183 / NE deg 4.047 (all 4 ship
    #   bars met; DT echo +0.113 vs the -22 baseline at -0.044 DT deg). The
    #   far_active floor is the documented preset knob (weak -18 / strong -28+);
    #   floors are amplitude-domain dB.
    min_gain_split_floor_enabled: bool = True
    min_gain_floor_far_active_db: float = -28.0
    min_gain_floor_far_silent_db: float = -12.0
    min_gain_far_latch_power: float = 1.0e6

    # ── v3.22 cohxd selective floor release (delay-aligned reference Γ²(X,Y)) ──
    # The constant split floor (above) masks the AEC3 R²-adaptive min_gain on
    # echo-dominant DT bins (proven: floor caps 0.3–2.9 dB of suppression the
    # AEC3 formula wants). This RELEASES the floor per-bin where the residual is
    # confidently echo by REFERENCE coherence: Γ²(X,Y), X=far_spec (delay-aligned
    # reference, bulk delay removed upstream), Y=near_spec (mic). High Γ² ⟹ echo
    # correlates with reference ⟹ suppress like AEC3 (echo↑). Low Γ² ⟹ nearend
    # (uncorrelated with X) ⟹ keep protective floor (deg held). Convergence-
    # robust — uses X not Ŷ/ERLE, so it works on the 100%-nonlinear hard DT
    # cases where the linear filter never converges (unlike the closed Γ²(Ŷ,Y)
    # Layer1, which measured filter quality). ASYMMETRIC: only LOWERS the floor
    # on high-Γ² bins, never raises it → worst case = current behaviour (no
    # regression). Only active in the far-active latched state.
    #   release_db : floor target for fully-confident-echo bins (Γ²≥gamma_hi)
    #   gamma_lo/hi: Γ² interpolation band (log-domain lerp between the split
    #                floor and release_db)
    #   alpha      : EMA rate for the cross/auto-PSD coherence tracking
    # Conservative defaults (high gamma_hi): downstream MMSE-OMLSA amplifies any
    # nearend over-suppression, so release only on strong echo confidence.
    # Bench with AEC_COHXD_RELEASE=<release_db>.
    cohxd_floor_release_enabled: bool = False
    cohxd_floor_release_db: float = -45.0
    cohxd_gamma_lo: float = 0.5
    cohxd_gamma_hi: float = 0.85
    cohxd_alpha: float = 0.05

    # ── per-bin near-end speech-presence probability (NearendSpp) ──
    # DT both-axes frontier-mover. Tracks, per bin, the minimum over a window of
    # the residual-to-reference power ratio |E|²/|X|² (the residual echo-path
    # transfer); a near-end onset spikes that ratio above the tracked floor while
    # a reverb-tail far-end stays at the floor (it is still explained by X). Maps
    # the dB spike through a sigmoid to p_ne[k] ∈ [0,1]. Uses the reliable
    # reference X (not Ŷ/ERLE) and multi-frame minima, so it separates near-end
    # from reverb-tail — the wall that closed every single-lag coherence route —
    # and works on under-converged hard cases. Default OFF research substrate;
    # consumers (near-gated cohxd, general floor modulation) read p_ne.
    #   spike_thr_db/soft_db : sigmoid centre/softness, dB of ratio above floor
    #   minima_subwindow     : MCRA sub-window D in frames (floor spans [D,2D))
    #   alpha                : EMA rate for |E|² / |X|² smoothing
    # Slow time constants are load-bearing: near-end talk-spurts are sustained
    # (~seconds) while far-only residual-echo fluctuations + R² lag are fast, so
    # heavy smoothing (alpha≈0.02 ≈500 ms) + a long minima window (≈2 s) averages
    # out the echo transients but keeps the near-end. Fast constants (0.2/60)
    # fail the Step-0 audio-proof — far-only echo spikes read as near-end.
    nearend_spp_enabled: bool = False
    nearend_spp_alpha: float = 0.02
    nearend_spp_minima_subwindow: int = 200
    nearend_spp_spike_thr_db: float = 5.0
    nearend_spp_spike_soft_db: float = 2.0
    # cohxd near-gate: scale the cohxd floor RELEASE by (1 - p_ne) so the floor
    # is only released where near-end is absent. Requires both cohxd and SPP on.
    cohxd_nearend_spp_gate_enabled: bool = False

    # ── linear-filter cold-start DEADLOCK breaker (PBFDKF Kalman gain) ──
    # mu = H_error/(0.5·H_error·X² + n·E²); H_error refreshed by
    # `H_error += leakage × Σ|W|²`. Σ|W|² is ~0 before the filter adapts, so on
    # hard echo paths the refresh can't bootstrap: W≈0 → refresh≈0 → H_error
    # decays to floor → mu dies → W stays ≈0 (self-reinforcing deadlock). Survey:
    # 67% of FS (no-DT!) cases never converge, mean ERLE 0.14 dB — the linear
    # filter is broadly idle and the nonlinear RES+floor carries echo, which is
    # the root of the DT-echo Pareto wall.
    #   h_error_refresh_erl_floor: floor on the erl(=Σ|W|²) used in the refresh
    #     so a minimum refresh survives cold-start; self-fades once Σ|W|² grows
    #     past it (no effect on already-adapted bins). 0 = OFF.
    #   h_error_floor_override: raise the H_error clamp floor (default 1e-3) to
    #     guarantee a minimum Kalman gain. 0 = use default. Higher = more
    #     adaptation but less steady-state stability.
    # Both default OFF (byte-equal). Bench with AEC_ERL_REFRESH_FLOOR /
    # AEC_HERROR_FLOOR.
    h_error_refresh_erl_floor: float = 0.0
    h_error_floor_override: float = 0.0

    # ── v3.22 D2: SER-based gain floor (nearend preservation without DT detector) ──
    # SER (Signal-to-Echo Ratio) floor in power domain, inserted after Wiener-gain
    # clip in SuppressionGain._lower_band_gain.
    #
    # MECHANISM:
    #   G_ser_floor[k] = nearend[k] / (nearend[k] + weighted_residual[k] + 1.0)
    #   G[k]           = max(G_wiener[k], G_ser_floor[k] * ser_floor_strength)
    #
    # SELF-CONSISTENT (no external DT detector needed):
    #   FS (nearend ≈ 0):  G_ser_floor → 0   → floor → 0   → echo suppression unaffected
    #   DT (nearend >> echo): G_ser_floor → 1 → floor high  → nearend preserved
    #   NE (nearend >> 0, echo ≈ 0): G_ser_floor → 1        → pass-through preserved
    #
    # WHY ser_floor_strength=0.5 default:
    #   Full floor (strength=1.0) = Speex MMSE-STSA SPP behaviour. 0.5 blends
    #   halfway between Wiener and SPP-like — empirical starting point for A/B.
    #
    # Default OFF for byte-equal; intended to bench with AEC_SER_FLOOR=1.
    ser_floor_enabled: bool = False
    ser_floor_strength: float = 0.5

    # ── v3.22 D3: Soft nearend tuning blend (sigmoid ENR interpolation) ──
    # Replaces the binary DNE is_ne switch in _gain_to_no_audible_echo with a
    # continuous sigmoid weight derived from the LF-band ENR sum (same 1-2kHz
    # band as DominantNearendDetector). ne_weight smoothly transitions from
    # nearend_tuning (ne_weight→1) to normal_tuning (ne_weight→0) as ENR rises.
    #
    # ne_weight = sigmoid((enr_threshold - enr_lf) / softness)
    # enr_tr = ne_weight * nearend_enr_tr + (1-ne_weight) * normal_enr_tr
    #
    # enr_threshold=0.25: DNE-canonical pivot (0.25 = echo < 0.25×ne → nearend dominant).
    # softness=0.25: ENR range over which the transition occurs (±1 std = 0-0.5 ENR).
    #
    # DEFAULT ON (v3.22, 2026-05-30) — 800-case verdict: FS echo +0.042 avg
    # (縮小 FS_movement −0.080 gap vs AEC2 → −0.036); DT deg −0.040 but
    # +0.107/+0.230 buffer over AEC3 ship target. All ship targets met.
    soft_nearend_blend_enabled: bool = True
    soft_nearend_blend_enr_threshold: float = 0.25
    soft_nearend_blend_softness: float = 0.25
    # Per-bin near-end blend (v3.22 P5): replace the scalar broadband-LF ne_w with
    # a PER-BIN ne_w from per-bin ENR (echo[k]/nearend[k]). Bins where near-end
    # dominates blend toward nearend_tuning (gentle, preserve near), echo-dominant
    # bins toward normal_tuning (aggressive, suppress echo) — frequency-selective
    # near-end protection that lets the far-active floor drop without the broadband
    # DT near-end cost. DEFAULT ON (v3.22.2): paired with far_active -28 it
    # recovers ~+0.058 DT deg vs the floor-alone -28 point (frequency-selective
    # near protection), so the deeper floor's echo gain costs only -0.044 DT deg
    # net vs the -22 baseline (all 4 ship bars met). AEC_SOFT_NE_PER_BIN=0 forces
    # OFF (reproduces the scalar-blend baseline).
    soft_nearend_blend_per_bin: bool = True

    # ── v3.22 D5: ne_weight gain floor (Speex SPP-proxy for DT nearend protection) ──
    # Direct Speex MMSE-STSA analogue: uses D3's ne_weight as SPP proxy to apply
    # a gain lower bound that prevents near-end over-suppression in doubletalk.
    #
    # G_floor = ne_weight × floor_strength; G = max(G_wiener, G_floor)
    # ne_weight = sigmoid((enr_threshold - enr_lf) / softness) [shared with D3]
    #
    # FS (ENR high → ne_weight→0): floor→0, full echo suppression preserved.
    # DT (ENR low → ne_weight→1): floor=floor_strength, nearend protected.
    # Unlike D2 (failed): no dependence on R² accuracy — uses ENR directly.
    #
    # floor_strength=0.3: power-domain floor → amplitude = sqrt(0.3) ≈ 0.55 (−5.2dB).
    # Maximum suppression in pure NE-dominant frame = 14.7 dB (preserves voice audibility).
    #
    # Default OFF for byte-equal; bench with AEC_D5_FLOOR=<strength>.
    d5_ne_floor_enabled: bool = False
    d5_ne_floor_strength: float = 0.3

    # ── v3.22 L1: Kuech-Kellermann second-order nonlinear residual ──
    # Adds quadratic render PSD term to the nonlinear R² path to capture
    # loudspeaker harmonic / intermodulation distortion echo.
    # Only fires when usable_linear=False (filter not converged) — the
    # primary failure mode for FS worst cases (pcb1Nh0Z gap −1.22, 9xjhiFb −1.30 vs AEC3).
    #
    # FORMULA: R²_nl = nl_alpha × x2² / 1.07e7
    #   1.07e7 = -20dBFS per-bin in int16² units (calibration reference).
    #   nl_alpha=0.1 → 10% addition at -20dBFS, grows quadratically louder.
    #
    # DEFAULT ON (v3.22, 2026-05-31) — 800-case vs d3 baseline:
    # FS echo +0.079/+0.106 (FS_movement 3.483→3.589, beats AEC2 3.519 by +0.070).
    # DT deg −0.027/−0.031 (still +0.08/+0.20 above AEC3 ship target).
    # NE deg −0.029 (still +0.45 above AEC3). All ship targets met.
    nl_r2_enabled: bool = True
    nl_r2_alpha: float = 0.1

    # ── v3.22 BUG FIX: ERLE render-activity x2 PSD-scale (revives reverb) ──
    # The render power fed to the ERLE estimators (ComputeAvgRenderReverb's
    # `_x2_reverb_for_erle` = |X_buf|² + avg_reverb, orchestrator ~3091) is in
    # the RAW |far_spec|² scale, but FullBandErle/SubbandErle's render-activity
    # gate `_X2_BAND_ENERGY_THRESHOLD = 4.4e7` is an AEC3 int16²-convention
    # constant. far_psd / capture_psd / error_psd are all scaled ×_PSD_SCALE
    # (32768²) for that convention; `_x2_reverb_for_erle` was NOT — so the gate
    # sits ~6-7 orders of magnitude too high, x2_sum never crosses it
    # (measured 0% on aec_record AND every cohort FS case), FullBandErle never
    # accumulates → `linear_filter_quality` stays None forever →
    # ReverbFrequencyResponse skips its update → reverb tail = 0 → the
    # late-reverb / far-offset echo tail is never suppressed (far-offset
    # residual leaks 15-22 dB vs AEC3; reverb model is DEAD across all cases).
    # When ON, scale `_x2_reverb_for_erle` by _PSD_SCALE to match far_psd and
    # the threshold. Validated revival on aec_record: quality 0→32%, reverb
    # tail 0→73%, far-offset −18→−30 dB. DEFAULT ON (v3.22, 2026-05-30) —
    # reverb-revival bug fix. 800-case on the E1-ON baseline: FS echo +0.07
    # (deg held), DT echo +0.03 / deg −0.02 (affordable — post-fix DT deg 1.99
    # still beats AEC3's 1.86, and A1 near-protection recovers it). The shelved
    # buggy-baseline verdict (DT −0.05) was the re-audit's first payoff: cost
    # halved on the corrected baseline. AEC_ERLE_X2_SCALE=0 forces OFF.
    erle_render_x2_psd_scale: bool = True

    # v3.22: reverb-tail conservativeness (only meaningful when the reverb model
    # is revived via erle_render_x2_psd_scale). Scales the additive reverb-tail
    # R² contribution. 1.0 = full AEC3 reverb output; <1.0 = more conservative.
    # The revived reverb's 800-case DT-deg cost (−0.04/−0.05) was traced to this
    # additive reverb-tail R² over-suppressing near in double-talk; FS far-only
    # gain is pure (no near). This knob tunes the FS-gain ↔ DT-cost Pareto.
    reverb_tail_strength: float = 1.0

    # ── v3.22 E1: ERLE Y2/E2 coordinate consistency (windowed capture PSD) ──
    # AEC3 forms BOTH Y2 (capture power) and E2 (residual power) from the same
    # WindowedPaddedFft (sqrt-Hann; echo_remover.cc:445). Our E2 (error_psd) is
    # windowed (_sel_esw) but Y2 (capture_psd = |filter.near_spec|²) is the RAW
    # rectangular overlap-save FFT (filters.py:192) — a mixed coordinate. The
    # sqrt-Hann window drops E2's energy ~½ vs a rectangular E2, so ERLE = Y2/E2
    # (rect / windowed) is inflated ~3 dB → the linear-path R² = s2_linear / ERLE
    # is UNDER-estimated → far-only echo leaks. When ON, feed the ERLE estimator
    # a windowed Y2 (near_spec_win = error_spec_windowed + echo_spec, the same
    # windowed-capture reconstruction the selection path uses at
    # orchestrator:2889) so Y2/E2 share the sqrt-Hann coordinate. ERLE-estimator
    # ONLY (ERL / saturation keep the rectangular capture_psd). SOFT-nores
    # (post-filter). DEFAULT ON (v3.22, 2026-05-30) — confirmed correctness fix.
    # The clamps this ERLE feeds (1.0/4.0/1.5) are AEC3-canonical
    # (echo_canceller3_config.h:123-125), built for the CONSISTENT coordinate, so
    # the mixed rect/Hann ERLE was the bug (no orphaned clamp-tune to re-derive).
    # 800-case e1-vs-e0: FS echo +0.08 (deg held), DT echo +0.04, NE deg +0.00;
    # DT deg −0.05 = removal of the accidental near-protection the inflated ERLE
    # provided → to be recovered by EXPLICIT DT near-protection (Track A), not by
    # reverting this fix. AEC_ERLE_Y2_WIN=0 forces OFF (reproduces old baseline).
    erle_windowed_capture_psd: bool = True

    # ── v3.22 E2: output base = raw capture Y when the linear filter is unusable ─
    # AEC3 (echo_remover.cc:475) forms the output from `UseLinearFilterOutput()
    # ? E : Y` — when the linear filter is NOT usable it applies the suppression
    # gain to the RAW windowed capture Y, not the (diverged/unreliable) linear
    # residual E. `UseLinearFilterOutput() == UsableLinearEstimate()` (aec_state.h
    # :54-63, identical bodies). We always emit `error_spec(E) × gain` (orchestrator
    # ~3467); on diverged frames (E > Y, e.g. the negative-ERLE cases) that
    # suppresses a residual louder than the mic. When ON and usable_linear is
    # False, switch the output base to Y = near_spec_win (= error_spec +
    # _sel_echo_spec, the windowed capture) — but ONLY when frame |E|² > |Y|²
    # (genuine over-output). The blanket switch (no |E|>|Y| guard) re-leaked echo
    # on unusable-but-cancelling frames → 800-case FS echo −0.023; the guard aims
    # to keep the DT/NE deg gains (+0.013/+0.014) without that regress. SOFT-nores
    # (gain unchanged). VERDICT (2026-05-30): guarded version ≈ 0.000 on 800-case
    # AECMOS — NOT because the gate rarely fires, but because perceptual effect is
    # neutral. Gate/fire trace: usable_linear=False opens 30.8% of frames overall;
    # |E|>|Y| fires 34.5% of those (10.6% of all frames); FS_movement highest
    # (41% fire rate). 800-case ≈0 because during movement/divergence suppressor
    # gain is already near-zero → Y×gain ≈ E×gain ≈ 0 either way. But E2 is a
    # CORRECT AEC3 port (Y-reconstruction verified, gate signal correct, not a
    # threshold/input mismatch) and protects against genuine over-output on
    # diverged frames. DEFAULT ON per user directive: correct port → ship ON.
    # AEC_OUT_CAPTURE_UNUSABLE=0 forces OFF.
    output_capture_when_linear_unusable: bool = True

    # P4 fix [DEFAULT ON, 2026-06-02]: protect a working alignment from a spurious
    # late first-acquisition. When the linear filter is already cancelling
    # (>2.5 dB windowed ERLE) at the current alignment, reject Path-A delay
    # acquisition (which would reset the filter + apply a spurious large shift).
    # Recovers FS-echo on ~4% of cases the bench pre-align + in-pipeline
    # matched-filter conflict was destroying (3–7.5 dB). 800-case (vs splitcfg):
    # 26 cases changed, net Σecho +2.85 / Σdeg +0.28; biggest wins kZogUfYc +2.06,
    # Xv7jH2 +1.51; the FS/DT "casualties" are AECMOS movement-quirk echo-score
    # redistribution (audio-verified: removes echo, not near-end). Harmless in
    # production (no GCC pre-align → cold filter ERLE≈0 → guard never blocks a
    # legitimate cold acquisition); the gain is a bench-measurement artifact fix.
    delay_acquire_protect_converged: bool = True

    # ── v3.22 W4: DNE ENR-relax when near-end is loud vs noise (DT NE protect) ─
    # The dominant-nearend (DNE) detector only declares NE-state when the
    # residual-echo estimate is small vs the near-end (echo_sum <
    # enr_threshold·ne_sum, enr_threshold=0.25). During doubletalk the near-end
    # leaks into the filter error, so ERLE reads low (1.0–2.1) and R² =
    # s2_linear/ERLE stays large → echo_sum ≈ ne_sum (measured enr 0.64–1.39) →
    # the ENR test fails 76% → NE-state fires only ~45% of NE-present collapse
    # hops → the suppressor uses its aggressive `_normal_*` tuning and wipes the
    # near-end (gain → 0.00–0.016, −36 dB) even though the near-end is
    # 240×–100000× above the noise floor (the SNR trigger passes overwhelmingly).
    #
    # SPEC (when flag ON): when the near-end overwhelmingly dominates the noise
    # floor (ne_sum > snr_factor · snr_threshold · noise_sum, i.e. far past the
    # SNR-trigger bar), relax the ENR trigger threshold from enr_threshold (0.25)
    # to `enr_threshold` below (0.75). A clearly-present loud near-end is not
    # abandoned just because the NE-inflated echo estimate looks large. AEC3's
    # `early_exit` (echo > enr_exit_threshold·ne, =10×) is preserved as the
    # genuine-echo-dominance guard. FS is self-guarded: in far-end-only the
    # "near-end" estimate IS residual echo so enr ≈ 1 > 0.75 → no false NE-state.
    #
    # Beyond-AEC3 (AEC3's DNE is deliberately echo-protective during DT); aligns
    # with the PRIMARY=near-end / SECONDARY=echo priority. Cost: some DT echo
    # leak — matched-magnitude 800-case Pareto is the arbiter. SOFT-nores
    # (post-filter detector → gain tuning only). Default OFF for byte-equal.
    dne_loud_nearend_enr_relax_enabled: bool = False
    dne_loud_nearend_snr_factor: float = 3.0
    dne_loud_nearend_enr_threshold: float = 0.75

    # ── Shadow filter (dual-filter divergence control) ──────────────────
    enable_shadow: bool = True
    shadow_mu_ratio: float = 1.0
    shadow_copy_threshold: float = 0.65
    shadow_err_alpha: float = 0.80
    shadow_mu_min: float = 0.5
    shadow_copy_hysteresis: int = 3
    shadow_dtd_advantage_scale: float = 3.0
    shadow_dtd_offset: float = 1.5
    # INTERNAL: shadow is always PBFDAF NLMS; mu kept for preset wiring.
    shadow_mu_nlms: float = 0.5

    # ── Filter misadjustment estimator (AEC3 parity, INTERNAL tuning) ──
    # AEC3 inv_misadjustment over n-hop window; shrinks W on divergence.
    filter_misadjustment_alpha_up: float = 0.99
    filter_misadjustment_alpha_dn: float = 0.95
    filter_misadjustment_threshold: float = 0.5
    filter_misadjustment_hangover_frames: int = 100
    filter_misadjustment_stable_frames: int = 30
    filter_misadjustment_scale_min: float = 0.5
    filter_misadjustment_scale_max: float = 2.0

    # ── PBFDKF (Kalman) ─────────────────────────────────────────────────
    use_kalman: bool = True
    kalman_q_high: float = 1e-3
    kalman_q_low: float = 1e-6
    warmup_frames: int = 80

    # ── Echo-path-change detection (requires shadow) ────────────────────
    epc_delta_threshold: float = 0.3
    epc_total_rise: float = 1.5
    epc_hangover: int = 20
    epc_mu_floor: float = 0.5

    # ── Delay estimation (matched-filter + ring buffer) ─────────────────
    enable_delay_est: bool = True
    max_delay_ms: float = 1024.0
    delay_buffer_ms: float = 2048.0
    delay_est_period_s: float = 0.5
    delay_est_init_s: float = 0.3
    fixed_delay_samples: int = -1
    delay_par_low_threshold: float = 5.0
    delay_par_solid_threshold: float = 8.0

    # ── High-pass filtering ─────────────────────────────────────────────
    enable_highpass: bool = True
    highpass_cutoff_hz: float = 80.0
    # Reference-path HPF was retired (v3.19); mic-path HPF remains ON.

    # ── Saturation / non-linear echo handling ───────────────────────────
    enable_saturation_detect: bool = True
    saturation_threshold: float = 0.95
    saturation_softclip_ref: bool = True

    # ── Mode / output ───────────────────────────────────────────────────
    mode: AecMode = AecMode.PBFDKF
    return_res_context: bool = False
    clear_filter_history: bool = False

    def __post_init__(self):
        if self.frame_size == -1:
            self.frame_size = self.sample_rate * 20 // 1000  # 20ms
        if self.hop_size == -1:
            self.hop_size = self.frame_size // 2             # 10ms
        if self.filter_length == -1:
            if self.sample_rate >= 44100:
                self.filter_length = self.sample_rate * 64 // 1000
            else:
                self.filter_length = self.sample_rate * 52 // 1000
        if self.frame_size != 2 * self.hop_size:
            raise ValueError(
                f"50% overlap invariant violated: frame_size={self.frame_size} "
                f"must equal 2 * hop_size={self.hop_size}"
            )

    @property
    def fft_size(self) -> int:
        n = self.frame_size
        return 1 << (n - 1).bit_length()

    @property
    def n_partitions(self) -> int:
        return (self.filter_length + self.hop_size - 1) // self.hop_size

    @property
    def psd_scale(self) -> float:
        return 32768.0 ** 2

    @classmethod
    def from_preset(cls, preset: 'AecPreset', **kwargs) -> 'AecConfig':
        """BALANCED preset (the only shipping preset)."""
        if isinstance(preset, str):
            preset = AecPreset(preset)
        if preset == AecPreset.BALANCED:
            defaults = dict(
                enable_cng=True,
                shadow_mu_min=0.5,
                warmup_frames=100,
                kalman_q_high=1e-3,
            )
        else:
            defaults = {}
        defaults.update(kwargs)
        return cls(**defaults)
