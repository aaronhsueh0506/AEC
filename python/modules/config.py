"""AecConfig dataclass + BALANCED preset.

Customer-facing / system-parameter surface only. AEC3-alignment flags
that were default-True at release are hard-coded into the call sites
under ``modules.orchestrator`` / ``modules.filters``; closed-substrate
NOSHIP flags are deleted entirely.
"""
from dataclasses import dataclass, field
from typing import NamedTuple, Optional, Tuple

from . import aec3_scale as _aec3_scale
from .enums import AecDelayMode, AecMode, AecPreset


# ── Signal grid: ONE resolver, shared by every derivation ────────────────
# (sample_rate, fft_size) is the complete init-time description of the frame
# geometry; everything else is derived. This project fixes
# frame_size == fft_size and hop_size == fft_size // 2, but that derivation
# lives in exactly ONE place -- ``resolve_signal_grid()`` -- which
# ``AecConfig.__post_init__`` is the sole caller of. Mirrors the C
# ``aec_resolve_signal_grid()`` / ``AEC_GRID_TABLE`` (c_impl/src/aec.c) row
# for row, including the is_legacy flag.
#
# ``fft_size`` must be explicit at the core: 16 kHz has TWO production grids
# (256 and 512), so a resolver that guessed would silently pick one.
# ``default_fft_size()`` is the CONVENIENCE default a factory may apply
# before resolving (what the ``frame_size = -1`` sentinel triggers below,
# and what C's ``aec_config_defaults()`` fills in) -- it is the only place a
# grid is ever chosen for a caller.
class SignalGrid(NamedTuple):
    """A fully resolved, self-consistent frame geometry."""

    sample_rate: int
    fft_size: int          # == frame_size in this project
    frame_size: int
    hop_size: int          # == fft_size // 2
    n_freqs: int           # == fft_size // 2 + 1
    is_legacy: bool        # True for 8 kHz: supported, not a product grid


# 8 kHz is a LEGACY grid (productization plan §11.2, decided): supported by
# this library and by the Audio_ALG MONO pipeline, but NOT a product grid
# and NOT supported by the Audio_ALG 4-CHANNEL pipeline, whose public API
# contracts to exactly the three product grids. Kept because real tests
# depend on it; flagged rather than left sitting anonymously in a guessed
# path.
_GRID_TABLE: Tuple[SignalGrid, ...] = (
    SignalGrid(16000,  256,  256, 128, 129, False),   # product, 16k default
    SignalGrid(16000,  512,  512, 256, 257, False),   # product, 16k alternate
    SignalGrid(48000, 1024, 1024, 512, 513, False),   # product
    SignalGrid( 8000,  256,  256, 128, 129, True),    # LEGACY
)


def default_fft_size(sample_rate: int) -> int:
    """The product-default fft_size for a rate, or 0 if unsupported.

    The first table row for the rate (16 kHz -> 256, the low-latency /
    low-compute grid; 48 kHz -> 1024; 8 kHz -> 256). A caller wanting the
    16 kHz alternate must ask for 512 explicitly -- which is the whole
    reason ``resolve_signal_grid`` refuses an unspecified fft_size.
    Mirrors C's ``aec_default_fft_size()``.
    """
    for g in _GRID_TABLE:
        if g.sample_rate == sample_rate:
            return g.fft_size
    return 0


def resolve_signal_grid(sample_rate: int, fft_size: int,
                        hop_size: Optional[int] = None) -> SignalGrid:
    """Resolve (sample_rate, fft_size) into a full grid, or raise.

    ``hop_size``, when given, is CHECKED against the resolved hop rather
    than used to derive anything -- the 50%-overlap invariant is a property
    of the grid, not a caller input. Pass None (or the matching value) for
    the normal path.
    """
    if fft_size is None or fft_size <= 0:
        raise ValueError(
            f"signal grid must be explicit: got frame_size/fft_size="
            f"{fft_size} at sample_rate={sample_rate}. 16 kHz has two "
            f"production grids (256 and 512), so nothing guesses one for "
            f"you here; use default_fft_size() if you want the product "
            f"default.")
    if fft_size & (fft_size - 1):
        raise ValueError(
            f"no-padding invariant violated: frame_size={fft_size} "
            "must be a positive power of two")
    for g in _GRID_TABLE:
        if g.sample_rate == sample_rate and g.fft_size == fft_size:
            if hop_size is not None and hop_size != g.hop_size:
                raise ValueError(
                    f"50% overlap invariant violated: frame_size={fft_size} "
                    f"must equal 2 * hop_size={hop_size}")
            return g
    raise ValueError(
        f"unsupported signal grid: sample_rate={sample_rate}, "
        f"frame_size/fft_size={fft_size}")


@dataclass
class AecConfig:
    """AEC configuration (release surface).

    System sizes are samples; rates Hz. Fields marked INTERNAL are
    preserved for backward compatibility with the BALANCED preset and
    internal call sites — do not document as customer knobs.
    """

    # ── System / framing ────────────────────────────────────────────────
    sample_rate: int = 16000        # 8000 / 16000 / 48000
    # No-padding signal grid. Auto: 256 @ 8 kHz, 256 @ 16 kHz (8 ms hop,
    # the low-latency/low-compute default), 1024 @ 48 kHz. 16 kHz also
    # accepts the higher-compute 512/256 grid (16 ms hop) explicitly.
    frame_size: int = -1            # analysis frame == FFT size
    hop_size: int = -1              # frame_size / 2
    filter_length: int = -1         # Auto: 52ms (<44.1 kHz) or 64ms (≥44.1 kHz)
    mu: float = 0.3                 # Step size
    delta: float = 1e-8             # Regularization

    # ── Residual / comfort noise ────────────────────────────────────────
    enable_res: bool = True
    enable_cng: bool = False
    # AEC3 comfort-noise floor (dBFS). Float[-1,1] PSD scale.
    comfort_noise_floor_dbfs: float = -96.03406
    enable_td_constraint: bool = True
    # AEC3 round-robin TD constraint (DEFAULT ON) — constrain ONE filter
    # partition per hop (cycling) instead of all partitions every hop
    # (adaptive_fir_filter.cc:686-689). Cuts the per-hop constraint FFTs
    # ~n_partitions× (≈58% of the integrated-path per-hop FFTs). Our old
    # all-partitions-every-hop was OVER-constraining (the float32 irfft/rfft
    # round-trip + boundary window repeatedly nibbled freshly-adapted taps),
    # so this also DEEPENS linear convergence → +echo. 800-case no-PA validated
    # (vs full-constraint): echo up every far-active bucket (FS +0.055/+0.073,
    # DT +0.043/+0.055) at −0.031 DT deg; the DT cost is neutralised by the
    # min_gain_floor_dt_db −20→−16 re-tune below (round-robin freed FS-echo
    # headroom to spend on DT near-protection). Propagated to main + shadow.
    constraint_round_robin: bool = True

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
    # Energy gate false-arms a little on loud FS echo (hence the FS cost).
    # Re-tuned -20->-16 alongside constraint_round_robin: round-robin's deeper
    # linear convergence lifts FS echo, freeing headroom to raise this DT-only
    # floor and neutralise round-robin's −0.031 DT-deg cost. 800-case no-PA
    # validated (round-robin + this −16): vs full-constraint baseline, DT deg
    # back to baseline (DT_static +0.002, DT_movement −0.004) while echo stays
    # up everywhere (FS +0.029/+0.022, DT +0.014/+0.027), all four bars hold.
    dt_aware_res_floor_enabled: bool = True
    min_gain_floor_dt_db: float = -16.0

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
    # dominates blend toward nearend_tuning (mild, preserve near), echo-dominant
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

    # Recent INST-ERLE peak (dB) over the last ~15 frames of the 500 ms slope
    # ring. erle_windowed lags ~0 in the ~140 ms between far-end onset and first
    # delay lock at no-PA cold start, so it cannot tell whether the cold filter is
    # already cancelling via tap-reach. The inst-ERLE peak can. LOAD-BEARING: it is
    # the gate threshold for delay_acquire_warm_transfer below (warm only fires
    # when the filter was genuinely cancelling, peak > this).
    delay_acquire_inst_erle_db: float = 4.0
    # REJECTED option-A substrate (default-OFF): use the inst-ERLE peak to BLOCK the
    # first realign entirely. A/B'd on 800-case — echo cost not cleanly gateable
    # (165 cases fire, 15 >1 dB, corr(delay,cost)≈0). Superseded by warm transfer
    # (which keeps the realign but preserves cancellation). Kept for reference.
    delay_acquire_protect_inst_erle: bool = False

    # Warm tap-transfer on first delay acquisition (default-ON, production). When
    # the online matched-filter locks the delay and the ring buffer realigns,
    # instead of ZEROING the filter (which discards the cancellation it built at
    # the cold alignment and exposes the echo — the 1.14 s "vertical line" in
    # cold-start FS/DT), shift the learned IR LEFT by the acquired delay so the
    # cancellation survives the realign (gated on: filter already cancelling
    # [inst-ERLE peak > delay_acquire_inst_erle_db] AND delay within tap reach).
    # Keeps our strong linear AEC (~12 dB) through the lock, unlike AEC3 which
    # dodges the line by barely cancelling in linear. Validated no-PA on the AEC
    # 800-case (enable_res) AND the Audio_ALG NR pipeline: every bucket echo+deg
    # neutral-to-up (DT deg +0.089 firing-subset / +0.019 full-corpus, FS echo
    # +0.009), NE byte-equal, zero ship-bar risk. Fires on ~117/600 FS+DT cases;
    # non-firing cases are byte-equal to the zero-reset path.
    delay_acquire_warm_transfer: bool = True

    # ── Shadow filter (dual-filter divergence control) ──────────────────
    enable_shadow: bool = True
    shadow_copy_threshold: float = 0.65
    # Per-hop EMA alpha (EMA lag ~44.8 ms @ this literal's legacy hop=160/
    # sample_rate=16000 = 10 ms grid) smoothing main/shadow error energies.
    # Wall-clock-authored with zero rate conversion; retimed live in
    # __post_init__ below (2026-08 gap-fix) -- see that block's comment.
    shadow_err_alpha: float = 0.80
    shadow_mu_min: float = 0.5
    shadow_copy_hysteresis: int = 3
    shadow_dtd_advantage_scale: float = 3.0
    shadow_dtd_offset: float = 1.5
    # INTERNAL: shadow is always PBFDAF NLMS; mu kept for preset wiring.
    shadow_mu_nlms: float = 0.5

    # ── Filter misadjustment estimator (AEC3 parity, INTERNAL tuning) ──
    # AEC3 inv_misadjustment over n-hop window; shrinks W on divergence.
    # Both hop counts below are wall-clock-authored @ legacy hop=160/
    # sample_rate=16000 (10 ms) grid (100 hops = 1000 ms hold-off,
    # 30 hops = 300 ms required stability), zero rate conversion; retimed
    # live in __post_init__ below (2026-08 gap-fix).
    filter_misadjustment_hangover_frames: int = 100
    filter_misadjustment_stable_frames: int = 30
    filter_misadjustment_scale_min: float = 0.5
    filter_misadjustment_scale_max: float = 2.0

    # ── PBFDKF (Kalman) ─────────────────────────────────────────────────
    use_kalman: bool = True
    kalman_q_high: float = 1e-3
    kalman_q_low: float = 1e-6
    # Warm-up hop count (80 hops = 800 ms @ legacy hop=160/sample_rate=16000
    # = 10 ms grid); ``from_preset`` overrides this to 100 (=1000 ms) for
    # every preset -- 100 is the actual production/live value, this 80 is
    # only reachable via a bare ``AecConfig()``. Both are wall-clock-
    # authored with zero rate conversion; retimed live in __post_init__
    # below (2026-08 gap-fix).
    warmup_frames: int = 80

    # ── Echo-path-change detection (requires shadow) ────────────────────
    epc_delta_threshold: float = 0.3
    epc_total_rise: float = 1.5
    # 20 hops = 200 ms @ legacy hop=160/sample_rate=16000 (10 ms) grid,
    # zero rate conversion; retimed live in __post_init__ below (2026-08
    # gap-fix).
    epc_hangover: int = 20

    # ── Delay estimation (matched-filter + ring buffer) ─────────────────
    # THE source of truth for far/mic alignment (see AecDelayMode). Accepts
    # the enum, its int value, or its name as a string; normalised to the
    # enum in __post_init__. init-time immutable in the C port (the pool
    # layout differs per mode), so treat it as immutable here too.
    delay_mode: AecDelayMode = AecDelayMode.MATCHED
    # DEPRECATED translation-layer mirror of ``delay_mode``. Kept so
    # pre-delay_mode callers keep working unchanged; it is NOT a second
    # source of truth -- __post_init__ folds it into ``delay_mode`` once and
    # then rewrites it to match, and every downstream read (orchestrator,
    # 4ch wrappers, AIAEC) is of ``delay_mode`` alone. Mirrors the C
    # ``AecConfig.enable_delay_est`` / ``aec_config_resolve_delay()``.
    # Scheduled for removal once all callers have moved to delay_mode.
    enable_delay_est: bool = True
    max_delay_ms: float = 1024.0
    delay_buffer_ms: float = 2048.0
    delay_est_period_s: float = 0.5
    delay_est_init_s: float = 0.3
    # FIXED only: the measured system delay in native-rate samples, >= 0.
    # Must stay at the -1 unset sentinel in MATCHED / EXTERNAL_ALIGNED --
    # a fixed delay those modes cannot honour is REJECTED, not ignored.
    # (Before delay_mode existed, ``fixed_delay_samples >= 0`` IMPLICITLY
    # overrode enable_delay_est. That implicit override is gone: say
    # ``delay_mode=AecDelayMode.FIXED`` -- or, legacy-style,
    # ``enable_delay_est=False`` -- alongside it.)
    fixed_delay_samples: int = -1
    delay_par_low_threshold: float = 5.0
    delay_par_solid_threshold: float = 8.0
    # Size of the matched-filter bank (number of staggered NLMS hypotheses).
    # 5 is the AEC3 default and the ONLY value bench / dataset generation may
    # use -- it is what the published scores were measured with. Lowering it
    # is a COMPUTE knob for the embedded target, where the (bring-up
    # measured) system delay is already compensated via
    # ``fixed_delay_samples`` and the matched filter only has to track the
    # residual; each filter dropped removes ~4.2 MMAC/s of full-rate search
    # cost (n=1 is -73% vs n=5). Window / alignment shift / thresholds are
    # NOT touched, so every surviving filter behaves exactly as it does at
    # n=5 -- only the bank's reach shrinks. Reliable coverage is
    # ``(n-1)*384 + 501`` downsampled samples at 0.25 ms each:
    #   n=1 -> 125 ms | n=2 -> 221 ms | n=3 -> 317 ms | n=4 -> 413 ms
    #   n=5 -> 509 ms (see docs/delay_estimator_design_zh_TW.md §4/§5).
    # Range 1..5: the ring buffer and the aggregator histograms are sized
    # from this value, and 5 is the geometry this port was validated at.
    delay_num_filters: int = 5
    # DT-aware soft-recovery gate. The delay/EPC recovery triggers
    # (Path A/B, EPV, shadow_rise) fire at far-only moments but their
    # aggressive re-convergence tail overfits near-end speech that arrives
    # shortly after. A held "near-end seen recently" flag lets the soft
    # (non-destructive) recovery path cover that tail without touching pure
    # far-end single-talk (which never raises the near-end indicator, so its
    # recoveries stay aggressive and FS echo depth is preserved).
    ne_recent_threshold: float = 0.3
    # 150 hops = 1500 ms @ legacy hop=160/sample_rate=16000 (10 ms) grid,
    # zero rate conversion; retimed live in __post_init__ below (2026-08
    # gap-fix).
    ne_recent_hold: int = 150
    # Genuine consecutive-event count (near-end-active hops required to ARM
    # the hold above), NOT a duration -- intentionally left unretimed.
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
    # Expose the shadow/main-selected, crossfaded, WOLA-formed linear hop via
    # get_formed_output(). This is not the raw main-filter process() output.
    # If AEC3 post-processing already runs, this is a free readback; otherwise
    # it runs only the selector/WOLA work required to form this signal.
    # It does not change filter adaptation or enable RES/CNG/suppression.
    # Default off so no existing caller pays for it.
    return_formed_output: bool = False
    clear_filter_history: bool = False

    # ── Wall-clock timing provenance (INTERNAL, do not document as a
    # customer knob) ─────────────────────────────────────────────────────
    # Canonical (grid-independent) authored values for shadow_err_alpha /
    # warmup_frames / epc_hangover / ne_recent_hold / filter_misadjustment_
    # {stable,hangover}_frames, captured ONCE against the legacy hop=160/
    # sample_rate=16000 (10 ms) grid these fields' literals were originally
    # authored at, and NEVER updated afterward. -1.0 sentinel means "not yet
    # captured" (a genuinely fresh construction). Regular dataclass fields
    # (not ``field(init=False)``), so ``dataclasses.replace()``/
    # ``**asdict()`` roundtrip construction carries an already-captured
    # value forward unchanged -- see __post_init__ below for why rebasing
    # from this PINNED canonical value (rather than from the previous
    # grid's already-rounded hop-count/alpha) is what avoids quantization
    # drift compounding through a chain of grid changes.
    _canonical_ms_warmup_frames: float = field(default=-1.0, repr=False)
    _canonical_ms_epc_hangover: float = field(default=-1.0, repr=False)
    _canonical_ms_ne_recent_hold: float = field(default=-1.0, repr=False)
    _canonical_ms_filter_misadjustment_stable_frames: float = field(default=-1.0, repr=False)
    _canonical_ms_filter_misadjustment_hangover_frames: float = field(default=-1.0, repr=False)
    _canonical_retention_shadow_err_alpha: float = field(default=-1.0, repr=False)

    def __post_init__(self):
        # ── signal grid: ONE resolver call, no inline table ──────────────
        # ``frame_size = -1`` is this dataclass's CONVENIENCE-default
        # sentinel (the Python analogue of C's aec_config_defaults() filling
        # a concrete fft_size): it is expanded to the rate's product default
        # HERE, and then the strict resolver has the final word. Nothing
        # downstream re-derives hop / n_freqs / validity -- that used to be
        # spread across three separate blocks in this method plus an inline
        # ``valid_grids`` dict, and the C side had its own third copy.
        if self.frame_size == -1:
            self.frame_size = default_fft_size(self.sample_rate)
        grid = resolve_signal_grid(
            self.sample_rate, self.frame_size,
            hop_size=None if self.hop_size == -1 else self.hop_size)
        self.frame_size = grid.frame_size
        self.hop_size = grid.hop_size
        if self.filter_length == -1:
            if self.sample_rate >= 44100:
                self.filter_length = self.sample_rate * 64 // 1000
            else:
                self.filter_length = self.sample_rate * 52 // 1000

        # Matched-filter bank size. Fail fast (same style as the grid check
        # above) rather than clamping: a caller that asks for 0 or 7 filters
        # has a wrong mental model of the delay geometry, and silently
        # substituting 5 would hide it. Upper bound 5 mirrors the C side's
        # DA_NUM_FILTERS compile-time array bound (delay_aec3.h).
        if not 1 <= self.delay_num_filters <= 5:
            raise ValueError(
                f"delay_num_filters must be in [1, 5], got "
                f"{self.delay_num_filters}"
            )

        self._resolve_delay_mode()

        # Wall-clock-authored top-level (non-AEC3) constants (2026-08
        # gap-fix, follow-up to the AEC3-internal per-block/hop-count
        # constant audit). shadow_err_alpha / warmup_frames / epc_hangover /
        # ne_recent_hold / filter_misadjustment_{stable,hangover}_frames
        # were literal hop counts / per-hop EMA alphas frozen at the legacy
        # hop=160/sample_rate=16000 (10 ms) grid that predates this repo's
        # own multi-rate history, with zero rate conversion when the
        # 16 kHz default flipped to 8 ms hop (2026-08-01) or at 8/48 kHz.
        # Retimed here via aec3_scale.ms_to_hops()/growth_rehop() so they
        # cover approximately the same wall-clock duration at every grid.
        #
        # IDEMPOTENCY + CROSS-GRID PRECISION (2026-08-04 gap-fix, revised
        # same day): an earlier version of this fix tracked "which grid is
        # the CURRENT field value calibrated for" and rebased from that --
        # correct for repeated same-grid reconstruction, but still lossy
        # across a CHAIN of different grids, because ``ms_to_hops()`` rounds
        # to the nearest integer hop and each step in the chain rebased from
        # the PREVIOUS step's already-rounded hop count instead of the
        # original canonical millisecond value (e.g. 16k/512 -> 48k/1024 ->
        # 16k/512 landed one hop off the true original -- confirmed by
        # direct measurement, not just theory). Fixed by capturing the
        # canonical ms/retention values ONCE (into the ``_canonical_*``
        # fields above, pinned for the lifetime of the value chain since
        # they are regular fields that survive replace()/asdict()
        # roundtrips unchanged) and ALWAYS re-deriving the live fields
        # fresh from that pinned canonical value + whatever grid was just
        # resolved above -- a pure function of (canonical, grid) with no
        # dependency on the field's own previous value, so however many
        # reconstructions or however many intermediate grids a config has
        # been through, every retime rebases from the exact same pristine
        # source. Must match aec.c's aec_config_defaults()/aec_carve() (C
        # side) -- there this is a non-issue because retiming reads the
        # caller's untouched *cfg into a->cfg exactly once, at construction,
        # with no re-entrant path.
        if self._canonical_ms_warmup_frames < 0:
            # First time this value chain has run __post_init__: capture
            # canonical values by interpreting whatever currently sits in
            # each field as authored at the legacy 10 ms grid -- true for
            # the dataclass default AND an explicit constructor kwarg like
            # ``from_preset``'s ``warmup_frames=100`` base override, since
            # both were authored under the same frozen 10 ms assumption.
            self._canonical_ms_warmup_frames = self.warmup_frames * 10.0
            self._canonical_ms_epc_hangover = self.epc_hangover * 10.0
            self._canonical_ms_ne_recent_hold = self.ne_recent_hold * 10.0
            self._canonical_ms_filter_misadjustment_stable_frames = (
                self.filter_misadjustment_stable_frames * 10.0)
            self._canonical_ms_filter_misadjustment_hangover_frames = (
                self.filter_misadjustment_hangover_frames * 10.0)
            self._canonical_retention_shadow_err_alpha = self.shadow_err_alpha

        # shadow_err_alpha uses the RETENTION convention (``alpha*old +
        # (1-alpha)*new`` -- see orchestrator.py's actual smoothing
        # formula), so ``growth_rehop`` (a direct power law), NOT
        # ``per_block_ema_alpha_to_per_hop``'s AEC3-new-sample-weight form,
        # is the correct helper here. Always rebased from the FIXED legacy
        # 160/16000 grid (never from the previous grid), so this too is
        # immune to chained-conversion drift.
        self.shadow_err_alpha = _aec3_scale.growth_rehop(
            self._canonical_retention_shadow_err_alpha, 160, 16000,
            self.hop_size, self.sample_rate)
        self.warmup_frames = _aec3_scale.ms_to_hops(
            self._canonical_ms_warmup_frames, self.hop_size, self.sample_rate)
        self.epc_hangover = _aec3_scale.ms_to_hops(
            self._canonical_ms_epc_hangover, self.hop_size, self.sample_rate)
        self.ne_recent_hold = _aec3_scale.ms_to_hops(
            self._canonical_ms_ne_recent_hold, self.hop_size, self.sample_rate)
        # ne_recent_sustain is a genuine consecutive-event count (NOT a
        # duration -- see orchestrator.py's ``_ne_recent_frames`` gate) and
        # is intentionally left unretimed; also out of scope for this pass
        # (not part of the audited constant batch).
        self.filter_misadjustment_stable_frames = _aec3_scale.ms_to_hops(
            self._canonical_ms_filter_misadjustment_stable_frames,
            self.hop_size, self.sample_rate)
        self.filter_misadjustment_hangover_frames = _aec3_scale.ms_to_hops(
            self._canonical_ms_filter_misadjustment_hangover_frames,
            self.hop_size, self.sample_rate)

    # ── delay-mode translation + validation ─────────────────────────────
    # Byte-for-byte the same contract as C's aec_config_resolve_delay() +
    # aec_validate_config() (c_impl/src/aec.c). Kept as one method so the
    # two halves -- "fold the deprecated mirror away" and "reject illegal
    # mode/field combinations" -- cannot drift apart or be run out of order.
    #
    # Mapping (the ONLY three legacy shapes that ever existed):
    #
    #   enable_delay_est | fixed_delay_samples | delay_mode in | resolved
    #   -----------------+---------------------+---------------+----------
    #    True (default)  | any                 | any           | delay_mode (unchanged)
    #    False           | >= 0                | MATCHED (dflt)| FIXED
    #    False           | <  0                | MATCHED (dflt)| EXTERNAL_ALIGNED
    #    False           | any                 | FIXED         | FIXED
    #    False           | any                 | EXTERNAL_ALIGN| EXTERNAL_ALIGNED
    #
    # ``enable_delay_est is True`` (its default) carries no information and
    # leaves delay_mode alone; clearing it is the legacy way of saying "not
    # MATCHED" and only translates while delay_mode is still at ITS default.
    # No conflict case exists, because the two non-MATCHED delay_mode values
    # are exactly the two outcomes clearing the legacy flag can select.
    # Idempotent: the mirror is rewritten from the resolved mode, so
    # re-running __post_init__ (dataclasses.replace / **asdict round-trips)
    # is a no-op.
    def _resolve_delay_mode(self) -> None:
        self.delay_mode = AecDelayMode.coerce(self.delay_mode)
        if (not self.enable_delay_est
                and self.delay_mode is AecDelayMode.MATCHED):
            self.delay_mode = (
                AecDelayMode.FIXED if self.fixed_delay_samples >= 0
                else AecDelayMode.EXTERNAL_ALIGNED)
        self.enable_delay_est = (self.delay_mode is AecDelayMode.MATCHED)

        fixed = int(self.fixed_delay_samples)
        if self.delay_mode is AecDelayMode.FIXED:
            if fixed < 0:
                raise ValueError(
                    "delay_mode=FIXED requires fixed_delay_samples >= 0, got "
                    f"{fixed}")
            # Generous but finite, and rate-relative: the reference ring is
            # grown to fixed + 4096, so an unbounded value here would drive
            # an unbounded allocation. 120 s matches delay_buffer_ms's own
            # 120000 ms upper bound on the C side.
            if fixed > 120 * self.sample_rate:
                raise ValueError(
                    f"fixed_delay_samples={fixed} exceeds the 120 s bound "
                    f"({120 * self.sample_rate} samples at "
                    f"{self.sample_rate} Hz)")
        else:
            if fixed != -1:
                raise ValueError(
                    f"fixed_delay_samples={fixed} is only meaningful with "
                    f"delay_mode=FIXED (got {self.delay_mode.name}); leave it "
                    "at the -1 unset sentinel")
        if (self.delay_mode is not AecDelayMode.MATCHED
                and self.delay_num_filters != 5):
            # No matched filter exists to size, so a non-default n would be
            # a silently ignored input -- and would let a caller believe
            # they had bought a compute saving this mode already gives them
            # in full.
            raise ValueError(
                f"delay_num_filters={self.delay_num_filters} is only "
                f"meaningful with delay_mode=MATCHED (got "
                f"{self.delay_mode.name}); leave it at the default 5")

    @property
    def fft_size(self) -> int:
        return self.frame_size

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

          mild      near-priority  (higher floor → more near kept, more echo leak)
          balanced    −28 dB         the shipped all-four-bars-met operating point
          aggressive  echo-priority  (deeper floor → more echo killed, more near loss)

        mild/aggressive deliberately sit off the balanced ship-bar set
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
            AecPreset.MILD:     dict(min_gain_floor_far_active_db=-20.0),
            AecPreset.BALANCED:   dict(),
            AecPreset.AGGRESSIVE: dict(min_gain_floor_far_active_db=-38.0),
        }
        defaults = {**base, **strength.get(preset, {})}
        defaults.update(kwargs)
        return cls(**defaults)
