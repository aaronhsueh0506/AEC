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

    # ── v3.22 W1': ERLE startup gate follows convergence (NE over-suppression) ─
    # The ERLE estimator is frozen at min_erle for a fixed 200-hop session
    # startup window (and re-armed on every delay-change), while usable_linear
    # opens earlier (~40 active-render hops + convergence_seen). In that desync
    # window the linear residual path runs R² = s2_linear / min_erle = s2_linear
    # (undivided) → min_gain collapses → near-end over-suppressed in the first
    # ~2 s and after each path change. When ON, ErleEstimator drops the fixed
    # session-startup gate and updates as soon as `converged_filter` is True
    # (Subband/FullBand already self-gate on convergence), so ERLE-readiness
    # tracks the same convergence signal usable_linear uses. SOFT-nores
    # (post-filter; no linear-adaptation change). Default OFF for byte-equal.
    erle_startup_follows_convergence: bool = False

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

    # ── v3.22 future work: LF filter-failure R² injection (NOT AEC3-strict) ─
    # AEC3 itself has no per-bin mechanism to detect "linear filter is
    # clearly failing at this bin AND ref signal has strong content here";
    # AEC3 accepts some LF echo bleed during DT to protect NE F0.
    #
    # This v3.22 candidate adds per-bin R² injection when the linear
    # cancellation is ~0 dB (filter useless) AND the DNE detector says
    # NE-dominant. Without injection, R²_direct = S²_linear / ERLE
    # underestimates → SG sees low ENR → gate doesn't fire → ref bleeds.
    # With injection: R² is lifted to a configured fraction of near_psd,
    # which makes ENR cross the nearend-tuning gate threshold (enr_tr_lf
    # = 1.09 by AEC3 spec) so SG suppresses naturally.
    #
    # SPEC (when flag ON, per-bin in [lf_low_hz, lf_high_hz] band):
    #   trigger: DNE is in NE-state
    #            AND error_psd[k] >= cancel_ratio × near_psd[k]
    #   action : R²[k] = max(R²[k], inject_factor × near_psd[k])
    #
    # OBSERVED CASE (568_EVB 6.5-7.0 s seg1):
    #   200-500 Hz: ref content +32 dB vs mic +30 dB (ref louder than mic,
    #   continuous DT speech overlap). Linear filter cancellation = -1.12
    #   dB (filter does nothing). User describes audible "two-pitch" /
    #   chorus effect from ref F0 + NE F0 overlapping at 200-500 Hz.
    #
    # TRADE-OFFS:
    #   - During NE-only (no ref): near_psd modest, but error_psd is
    #     basically the NE itself = near_psd → trigger fires → R² inject
    #     creates false suppression. Mitigation: gate also requires
    #     far_psd > silence_floor (configurable).
    #   - During FS-only (no NE): DNE=False → gate inactive.
    #   - During DT with strong echo + light NE: SG already suppresses
    #     via the standard path → trigger usually doesn't fire (error
    #     much smaller than near after good cancellation).
    enable_lf_filter_failure_r2_injection: bool = False
    lf_filter_failure_lf_low_hz: float = 200.0
    lf_filter_failure_lf_high_hz: float = 500.0
    lf_filter_failure_cancel_ratio: float = 0.9
    lf_filter_failure_r2_inject_factor: float = 1.2
    lf_filter_failure_min_far_psd_db: float = -40.0

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
    filter_misadjustment_scale_p: bool = False

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
    # Reference-path HPF is locked OFF (mic-path HPF remains ON).
    enable_highpass_ref: bool = False

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
