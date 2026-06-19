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
    filter_length: int = -1         # Auto: 52ms (<44.1 kHz) or 64ms (≥44.1 kHz)
    mu: float = 0.3                 # Step size
    delta: float = 1e-8             # Regularization

    # ── Residual / comfort noise ────────────────────────────────────────
    enable_res: bool = True
    enable_cng: bool = False
    # AEC3 comfort-noise floor (dBFS). Float[-1,1] PSD scale.
    comfort_noise_floor_dbfs: float = -96.03406
    enable_td_constraint: bool = True

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
    # DT-gated min-gain floor lift (default ON). During double-talk
    # (far-active latched AND near recently present, _ne_recent_frames>0) use
    # this higher floor instead of far_active, selectively protecting near-end
    # in DT while keeping the aggressive far_active floor during FS (no near).
    # 800-case no-PA validated on top of dt_aware_recovery_soft: DT_static deg
    # 2.016->2.074, DT_movement 2.088->2.140 (both EXCEED pre-align), at a
    # bounded echo cost (DT echo 4.28->4.22, FS echo 3.59->3.54, all bars hold).
    # Energy gate false-arms a little on loud FS echo (hence the FS cost); -20dB
    # is the operating point that maximises DT deg while holding FS>3.5.
    dt_aware_res_floor_enabled: bool = True
    min_gain_floor_dt_db: float = -20.0

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

    # ── Shadow filter (dual-filter divergence control) ──────────────────
    enable_shadow: bool = True
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

    # ── Delay estimation (matched-filter + ring buffer) ─────────────────
    enable_delay_est: bool = True
    max_delay_ms: float = 1024.0
    delay_buffer_ms: float = 2048.0
    delay_est_period_s: float = 0.5
    delay_est_init_s: float = 0.3
    fixed_delay_samples: int = -1
    delay_par_low_threshold: float = 5.0
    delay_par_solid_threshold: float = 8.0
    # DT-aware soft-recovery gate. The delay/EPC recovery triggers
    # (Path A/B, EPV, shadow_rise) fire at far-only moments but their
    # aggressive re-convergence tail overfits near-end speech that arrives
    # shortly after. A held "near-end seen recently" flag lets the soft
    # (non-destructive) recovery path cover that tail without touching pure
    # far-end single-talk (which never raises the near-end indicator, so its
    # recoveries stay aggressive and FS echo depth is preserved).
    ne_recent_threshold: float = 0.3
    ne_recent_hold: int = 150
    ne_recent_sustain: int = 3
    # Master switch for DT-aware soft recovery. When True, the delay/EPC
    # recovery triggers (Path A first-acquisition, Path B re-lock, EPV,
    # shadow_rise) drop their destructive full-reset + Kalman P=1.0 +
    # mark_diverged and instead realign softly (keep the converged filter,
    # re-adapt at normal step size) — but ONLY while near-end was seen
    # recently (held gate). Far-end single-talk never arms the gate, so its
    # recoveries stay aggressive and FS echo depth is preserved. Default ON:
    # 800-case no-PA validated — DT_static deg 1.993->2.016, DT_movement
    # 2.071->2.088, FS echo holds >3.5, all ship bars pass. Recovers the
    # online-acquisition filter-reset penalty that pre-align never pays.
    dt_aware_recovery_soft: bool = True

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
        """Three Pareto operating points on the residual-echo strength axis.

        All presets share the same AEC3 post-filter chain + the 800-case-tuned
        base (CNG, shadow/Kalman warmup). They differ ONLY on
        ``min_gain_floor_far_active_db`` — the far-active residual-gain floor
        that trades echo suppression against near-end preservation:

          gentle      near-priority  (higher floor → more near kept, more echo leak)
          balanced    −28 dB         the shipped all-four-bars-met operating point
          aggressive  echo-priority  (deeper floor → more echo killed, more near loss)

        gentle/aggressive deliberately sit off the balanced ship-bar set
        (Pareto picks), not within it — the DT-deg vs echo trade is a proven
        single-channel DSP wall, so a strength axis is the honest way to expose it.
        """
        if isinstance(preset, str):
            preset = AecPreset(preset)
        base = dict(
            enable_cng=True,
            shadow_mu_min=0.5,
            warmup_frames=100,
            kalman_q_high=1e-3,
        )
        # Strength axis: far-active min-gain floor (dataclass default −28 = balanced).
        strength = {
            AecPreset.GENTLE:     dict(min_gain_floor_far_active_db=-20.0),
            AecPreset.BALANCED:   dict(),
            AecPreset.AGGRESSIVE: dict(min_gain_floor_far_active_db=-38.0),
        }
        defaults = {**base, **strength.get(preset, {})}
        defaults.update(kwargs)
        return cls(**defaults)
