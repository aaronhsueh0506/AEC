"""AecConfig dataclass + BALANCED preset.

Customer-facing / system-parameter surface only. AEC3-alignment flags
that were default-True at release are hard-coded into the call sites
under ``modules.orchestrator`` / ``modules.filters``; closed-substrate
NOSHIP flags are deleted entirely.
"""
import math
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


# 8 kHz is a LEGACY grid: supported by
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


# Legacy hop-boundary headroom carried by the MATCHED search ring. NOT applied
# to FIXED (see ref_ring_samples): under MATCHED it is load-bearing rather than
# slack, because the ring size also feeds the orchestrator's
# ``new_delay <= _ref_ring_size - hop`` eligibility gate -- changing it would
# change the default path's audio. Mirrors C's
# ``AEC_REF_RING_MATCHED_HEADROOM``.
REF_RING_MATCHED_HEADROOM = 4096


def ref_ring_samples(config, hop_size: int) -> int:
    """Reference alignment ring capacity in samples for a resolved config.

    The single source of truth for that number on the Python side, mirroring
    C's ``aec_ref_ring_samples()`` in ``aec.c`` term for term:

      * ``MATCHED``  -- ``max(delay_buffer_ms, max_delay_ms + headroom)``, a
        SEARCH ring: the applied delay is unknown at init and can move at any
        hop, so it is sized for the whole configured search budget.
      * ``FIXED``    -- ``fixed_delay_samples + hop_size`` exactly.
      * ``EXTERNAL_ALIGNED`` -- 0 (nothing to buffer).

    FIXED derivation. With ``T`` the total samples written so far, the ring
    holds absolute samples ``[T-rs, T)``. Each hop the orchestrator first
    writes the newest ``hop`` samples (advancing ``T``), then the
    delay-compensating read takes ``[T-d-hop, T-d)``. Validity is therefore
    exactly ``rs >= d + hop``, and under FIXED ``d`` is immutable (no
    estimator exists, and ``reset()`` re-seeds the same value) -- a tight
    bound on a constant, not a worst case over a moving quantity, so no
    headroom term is meaningful. Equality is safe: at ``rs == d + hop`` the
    read starts exactly at the write cursor, i.e. on the oldest hop in the
    ring, which is precisely the one this hop's write did not overwrite.

    Byte-exactness against the old (oversized) ring follows from the fill
    gate ``_ref_ring_filled >= d + hop``: for ANY ``rs >= d + hop`` the gate
    first passes at hop index ``ceil((d+hop)/hop)`` and serves the same
    absolute samples. ``d == 0`` degenerates to a hop-sized, write-only ring
    (the read is skipped by the ``_current_delay > 0`` guard), kept rather
    than special-cased away so FIXED has exactly one ring code path.
    """
    from .enums import AecDelayMode          # local: avoids an import cycle
    if hop_size <= 0:
        raise ValueError(f"ref_ring_samples: hop_size={hop_size} must be > 0")
    mode = config.delay_mode
    if mode is AecDelayMode.EXTERNAL_ALIGNED:
        return 0
    if mode is AecDelayMode.FIXED:
        return int(config.fixed_delay_samples) + int(hop_size)
    max_delay_samp = int(config.max_delay_ms * config.sample_rate / 1000)
    buffer_samp = int(config.delay_buffer_ms * config.sample_rate / 1000)
    return max(buffer_samp, max_delay_samp + REF_RING_MATCHED_HEADROOM)


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
    # The historical −20→−16 re-tune spent round-robin's FS-echo headroom
    # on DT near protection.  The current −20 operating point deliberately
    # returns that small amount of headroom to residual-echo suppression; see
    # min_gain_floor_dt_db below for the full-corpus trade-off.
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
    # The former -16 dB setting spent round-robin's extra FS-echo headroom on
    # near-end protection. The selected -20 dB point returns part of it to
    # echo suppression; the paired-corpus trade-off is recorded below.
    dt_aware_res_floor_enabled: bool = True
    # 832-file paired A/B versus -16 dB: FS echo MOS +0.053/+0.064 and ERLE
    # +0.46/+0.53 dB (movement/static), for DT degradation MOS
    # -0.027/-0.030; NE-only is neutral.  This is a selected Pareto point,
    # not a claim that the near/far detector is now correct.
    min_gain_floor_dt_db: float = -20.0
    # ── Stage 3 (A21 plan section 六, P4): per-bin min-gain floor blend ──
    # Default OFF, byte-equal when off. The far-active branch of the split
    # floor above picks ONE scalar per hop (min_gain_floor_dt_db vs
    # min_gain_floor_far_active_db) via dt_protect_active, a single hop-level
    # flag -- so a bin that is actually near-dominant this hop gets the same
    # deep far-active floor as a bin that is actually echo-dominant, unless
    # the WHOLE hop happens to be flagged DT. When enabled, bins that are NOT
    # under the scalar DT flag instead blend toward the DT floor in
    # proportion to their OWN per-bin ENR (reusing the exact sigmoid already
    # computed for soft_nearend_blend_per_bin's gain-rule blend, same enr_thr/
    # softness, independent of whether that feature is itself enabled) --
    # i.e. a per-bin signal already shipped and validated for a different
    # consumer, not a new double-talk detector. Variant i only (this flag):
    # echo-dominant bins keep today's min_gain_floor_far_active_db unchanged
    # (no deepening); only near-dominant bins' floor can rise toward
    # min_gain_floor_dt_db. See suppression_gain.py's _get_min_gain.
    per_bin_floor_enabled: bool = False
    # ── Stage 4c (A21 plan section 七, P7): RSA persistence in wall-clock ms ──
    # Default OFF, byte-equal when off. RenderSignalAnalyzer's narrow-band
    # mask and poor-excitation freeze fire after a FIXED HOP COUNT
    # (_COUNTER_THRESHOLD=5, _POOR_EXCITATION_THRESHOLD=10), not a wall-clock
    # duration -- so the same hop count is a different duration on different
    # grids (grid A's 8ms hop vs grid B's 16ms hop at this project's two
    # standard operating points: ~48/88ms on A, ~96/176ms on B for the same
    # counts). Stage 0.4 single-flag measurement (fid0, S1 linear-only):
    # masking off alone was worth +1.2dB on grid A's LF ERLE but only
    # +0.25dB on grid B, consistent with grid A masking more persistently in
    # wall-clock terms for no corresponding benefit. When enabled, both
    # thresholds are computed from the ms fields below via ms_to_hops
    # (floor 1 hop) instead of the hard-coded hop counts, so the SAME
    # wall-clock persistence applies on every grid. Defaults below are the
    # "B-equivalent" point (grid B's own current hop-count behaviour
    # expressed in ms, i.e. a no-op for B and a loosening for A) -- the
    # plan's other pre-registered point ("AEC3-strict", 20/40ms, i.e. the
    # literal AEC3 4ms-block counts) tightens grid A further and was flagged
    # as likely wrong-directioned by the same measurement; not the default.
    rsa_wallclock_persistence: bool = False
    rsa_mask_hold_ms: float = 96.0
    rsa_poor_excitation_hold_ms: float = 176.0
    # dt-protect ARMING source (which signal flips _dt_protect_active, i.e.
    # which of min_gain_floor_far_active_db / min_gain_floor_dt_db applies).
    # 'energy' (default, byte-equal): today's pre-filter mic-energy-excess
    # rule (DoubleTalkAnalyzer.update_energy_dt via _ne_recent_frames).
    # 'dne': arm from SuppressionGain's own DominantNearendDetector
    # (is_dominant_nearend(), unbounded-R^2/ENR-based, already AEC3-aligned)
    # instead. Read ONE HOP LATE (the detector is updated inside the same
    # get_gain() call that reads this flag -- see orchestrator.py's wiring),
    # matching the C side's equally-one-hop-late dominant_ne read; this is a
    # deliberate match, not a bug to "fix" on either side. Not yet validated
    # for production use on every clip class tested; keep 'energy' unless
    # you have re-run the validation battery yourself.
    dt_protect_source: str = 'energy'
    # Extra hold (ms) added on top of the DNE source's own 500 ms hold
    # (DominantNearendConfig.hold_duration_ms, untouched -- it also gates
    # onset-ERLE selection and the HF limiter) before dt-protect releases.
    # 0 = no extra hold. Only consulted when dt_protect_source == 'dne'.
    dt_protect_extra_hold_ms: float = 0.0
    # Confirmation delay (ms) before a 'dne'-source DNE verdict is trusted:
    # is_dominant_nearend() must read True for MORE than this many
    # consecutive hops before dt-protect actually arms (0 = arm on the
    # first True hop, no delay). This does NOT change DNE's own state or
    # its other two consumers (onset-ERLE selection, the HF limiter) --
    # only how long THIS ONE consumer waits before trusting a verdict that
    # is already there. Only consulted when dt_protect_source == 'dne'.
    # A confirmation delay only filters DNE excursions SHORTER than the
    # delay; it does not separate a sustained (multi-second) false arm from
    # a sustained true one, since both look the same by duration alone.
    dt_protect_confirm_ms: float = 0.0
    # ── live LF-only early-shifted observer filter (residual/tail_observer.py) ──
    # A second, independent NLMS filter (same partition count as the main
    # filter) that reads the far-end ring `live_observer_shift_ms` EARLIER
    # than the main filter's own delay-compensated read, so it covers the
    # span just ahead of today's window instead of the span today's window
    # already covers. A scalar per-bin echo-evidence accumulator was found
    # structurally insufficient for a multi-tap echo path -- this observer
    # models a real span of taps instead. It never subtracts from the
    # output, never touches the delay estimator, ERLE, or the main filter's
    # own window -- only `max(r2, evidence)`. The "shift earlier" is the
    # same window
    # relocation the rejected main-filter "headroom" change made, applied to
    # this separate, output-less filter instead -- see
    # SE/.research/2026-10-16/a21/DIAG_PLAN_AMEND_6.md and the fable/opus
    # design discussion the same day. Default OFF, byte-equal when off.
    live_observer_enabled: bool = False
    live_observer_shift_ms: float = 24.0
    live_observer_max_hz: float = 1000.0
    live_observer_mu: float = 0.3
    # Injection guard, added 2026-10-17 (AMEND_12/13) after a two-path
    # (Ochiai/MDF-style) foreground/background protection was tried and
    # found to be the wrong tool: that scheme guards against a filter
    # DIVERGING relative to its own past state, not against it learning
    # near-end content the live near-end gate (DominantNearendDetector-
    # based, structurally imperfect -- see the plan's Stage 1 section) fails
    # to catch. Comparing the observer against ITS OWN past state cannot
    # tell "correctly cancelling echo" from "also cancelling near-end" --
    # both lower the observer's own residual. Comparing against the MAIN
    # filter's residual on the SAME hop can: if the observer's estimate is
    # real extra echo, subtracting it lowers the residual below what the
    # main filter alone already achieves; near-end appears in both
    # residuals and cancels to first order. Evidence is trusted only while
    # the observer's own LF residual energy, EMA'd over
    # `live_observer_corrob_ms`, is no worse than the main filter's (margin
    # `live_observer_corrob_margin_db`, negative = stricter); otherwise the
    # hop's evidence is zero. See DIAG_PLAN_AMEND_12.md.
    live_observer_corrob_ms: float = 100.0
    live_observer_corrob_margin_db: float = 0.0
    # Injection-only near-end discriminator, added after the corroboration
    # guard above was found insufficient on
    # IrQvqOTCmEWMXn9k2ICtRQ_doubletalk_with_movement (AMEND_29/32's worst
    # DT_movement_deg case, -0.117 MOS). A per-hop trace showed the
    # corroboration margin itself (`live_observer_corrob_margin_db`) cannot
    # separate this file's false-positive injections from legitimate ones
    # elsewhere (xFk7.../fid0): the EMA error-ratio distributions overlap
    # almost completely (IrQv false-positive median ~0.82 vs xFk7 legitimate
    # median ~0.84; fid0 legitimate ratios are even lower/stricter at
    # ~0.73/~0.21). A minimum-consecutive-corroboration streak was tried
    # next and also failed: fid0's own legitimate injection is itself bursty
    # (many 1-35-hop runs interleaved with a few 50-100-hop runs), so any
    # streak long enough to zero out IrQv's longest false-positive run
    # (38 hops) also zeroes fid0 grid A down to 1.1% of its baseline
    # injected hops.
    # What DOES separate them cleanly: `self._dt_from_energy`, the SAME
    # continuous NE-evidence signal already used at the 0.3 threshold
    # elsewhere in this codebase (epc.py's
    # `PathChangeRegimeHandler.GATE_ENERGY`; orchestrator.py's own
    # `_ne_evidence = dt_from_energy > 0.3` for refined_usable, ~line 3231)
    # -- NOT the hysteresis-latched `_dt_protect_active` flag the tail-evidence
    # learning gate above deliberately avoids (that latch is armed on
    # ~100% of far-only hops on fid0's own W2 window; the raw continuous
    # value is not). On IrQv's false-positive hops dt_from_energy's median
    # is ~0.84-0.91 (near-end clearly active); on xFk7's and fid0's
    # legitimate hops it is ~0.00-0.002 (near-silent).
    # Threshold tuning: the 0.3 value reused elsewhere in this codebase
    # (same constant) was tried FIRST and does separate the cases, but costs
    # more of fid0's own ERLE benefit than the evidence justifies: at 0.3,
    # fid0 grid B's full-file ERLE delta drops from +7.12 dB to +5.70 dB
    # (−1.42 dB, ~20% of the feature's headline benefit) while grid A is
    # untouched (+0.64 dB both ways). NOT traced per-hop whether the lost
    # grid-B hops (dt_from_energy in (0.3, 0.5]) are genuine double-talk or
    # proxy noise on far-only content -- avoid over-claiming here. At 0.5
    # the SAME IrQv case is fixed to the same practical degree (AECMOS
    # DT_movement_deg delta −0.117 -> +0.0008, i.e. the measured cost is
    # gone) while fid0 grid B only drops
    # to +6.91 dB (−0.22 dB, ~3%) and grid A is unaffected; xFk7 (the
    # legitimate-detection control file) is BYTE-IDENTICAL at both 0.3 and
    # 0.5 since its dt_from_energy is ~0 throughout. 0.5 is the shipped
    # default on that trade-off, not 0.3 -- this is evidence-driven, not a
    # re-use of the existing 0.3 constant for its own sake (feedback_
    # registered_guards_and_claim_scope: don't recommend past what was
    # measured; 0.3 was measured to overcorrect here).
    # Deliberately scoped to INJECTION only (not the observer's own
    # learning/mu_scale above): a learning mistake costs at most a missed
    # suppression opportunity (see the tail-evidence comment a few hundred
    # lines up), so the existing lagged-DNE gate on learning is left as-is;
    # this is the narrower, injection-only lever the plan called for when a
    # DNE-based check alone did not separate the cases (DNE itself reads
    # False on 100% of IrQv's false-positive hops here, even with the
    # existing one-hop-lag design matching the learning gate's own timing --
    # the gap is DNE's sensitivity on this file, not the lag).
    live_observer_inject_dt_energy_max: float = 0.5
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
    # DEFAULT ON — 800-case verdict: FS echo +0.042 avg
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
    # DEFAULT ON — 800-case vs d3 baseline:
    # FS echo +0.079/+0.106 (FS_movement 3.483→3.589, beats AEC2 3.519 by +0.070).
    # DT deg −0.027/−0.031 (still +0.08/+0.20 above AEC3 ship target).
    # NE deg −0.029 (still +0.45 above AEC3). All ship targets met.
    nl_r2_enabled: bool = True
    nl_r2_alpha: float = 0.1

    # Caps r2_nl at `nl_r2_cap_multiple` x the direct term (x2*echo_path_gain,
    # computed just above it) instead of the fixed -20dBFS reference above.
    # The direct term is signal-derived, so this makes the cap scale with
    # the actual render level instead of being quadratic-unbounded.
    # 300x: zero output delta on the 5-file FS Rule-A prescreen (bit-exact
    # vs uncapped on all 10 file/grid pairs) while bounding the observed
    # worst-case r2_nl ~420x (2.76e18 -> 6.52e15). 0.0 disables the cap.
    nl_r2_cap_multiple: float = 300.0

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
    # tail 0→73%, far-offset −18→−30 dB. DEFAULT ON —
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
    # (post-filter). DEFAULT ON — confirmed correctness fix.
    # The clamps this ERLE feeds (1.0/4.0/1.5) are AEC3-canonical
    # (echo_canceller3_config.h:123-125), built for the CONSISTENT coordinate, so
    # the mixed rect/Hann ERLE was the bug (no orphaned clamp-tune to re-derive).
    # 800-case e1-vs-e0: FS echo +0.08 (deg held), DT echo +0.04, NE deg +0.00;
    # DT deg −0.05 = removal of the accidental near-protection the inflated ERLE
    # provided → to be recovered by EXPLICIT DT near-protection (Track A), not by
    # reverting this fix. AEC_ERLE_Y2_WIN=0 forces OFF (reproduces old baseline).
    erle_windowed_capture_psd: bool = True

    # ── CNG N2 tracker coordinate fix (windowed capture PSD input) ──
    # A distinct bug from the one above, in the SAME family: AEC3's
    # ComfortNoiseGenerator (comfort_noise_generator.cc) estimates its noise
    # target from Y2 formed via WindowedPaddedFft(kSqrtHanning) — the SAME
    # windowed FFT the subtractor forms E2 from, not a rectangular one. This
    # port's N2/N2_initial tracker (below, in _aec3_post) is fed from
    # near_psd, the RECTANGULAR |near_spec|² — so the tracker's own target
    # coordinate runs ~3 dB too high relative to AEC3's windowed reference
    # (confirmed by an isolated seeded-N2 measurement against the windowed
    # Y2_smoothed quantity erle_windowed_capture_psd already forms above).
    # The CNG synthesis/injection path's window-loss compensation (the
    # kSqrt2Sin-style scaling) already assumes its N2 input sits in the
    # windowed coordinate, so an unwindowed-fed N2 overshoots there too — the
    # fix belongs upstream, at the tracker's input, not as an output-side
    # compensation factor. When ON, re-point the tracker (both the
    # first-hop y2_smoothed init and the per-hop EMA) at the SAME windowed
    # capture PSD shared with erle_windowed_capture_psd above (no second
    # FFT — reuses error_spec_windowed + echo_spec unconditionally formed
    # per hop). DEFAULT OFF: validated only via an isolated seeded-N2 A/B
    # harness (scratch, not committed), NOT yet run through the 800-case
    # corpus gate this project requires before a default-on promotion — do
    # not flip this default without that gate.
    cng_n2_windowed_psd: bool = False

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
    # (gain unchanged). VERDICT: guarded version ≈ 0.000 on 800-case
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

    # P4 fix [DEFAULT ON]: protect a working alignment from a spurious
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

    # DEFAULT OFF. Backward-jump quarantine on a delay CHANGE (Path B), the
    # sibling of delay_acquire_protect_converged above (which guards only the
    # FIRST acquisition and is untouched by these two fields).
    #
    # QUARANTINED, and nothing else: a candidate strictly EARLIER than the
    # delay already in force — the backward / pre-echo direction — offered
    # while the linear filter is demonstrably cancelling at the alignment
    # currently applied. A FORWARD jump (a larger delay), which pre-echo
    # mis-attribution cannot produce, is never held; neither is the first
    # acquisition.
    #
    # Why a direction test and not a longer confirmation window: the failure
    # this targets is a sustained wrong candidate, not a one-hop glitch. With
    # a white-noise far at a true bulk delay of 6400 samples the estimator
    # locks correctly at 6336 (hop 32) and then re-locks to 4800 at hop 49 —
    # exactly 1600 samples (100 ms) early, the pre-echo mis-attribution
    # signature — and keeps reporting 4800 for the rest of the run. Any
    # K-consecutive-confirmation gate confirms it. The re-lock is EARLIER
    # than the truth, which is what makes direction a usable filter.
    #
    # Evidence for "still cancelling": the SAME two ERLE readings Path A's
    # guard already uses — windowed ERLE against its 2.5 dB threshold, and
    # the recent inst-ERLE peak against delay_acquire_inst_erle_db. Both arms
    # are needed here, unlike Path A, because a pre-echo re-lock arrives
    # within a few hundred ms of the first lock: measured at the 6400→4800
    # acceptance hop, windowed ERLE is only 1.660 dB (still inside the lag
    # documented on delay_acquire_inst_erle_db above) while the inst peak is
    # 7.269 dB. A windowed-only predicate cannot see this defect at all.
    #
    # Both readings are 0.0 until the machinery that fills them has run, so
    # an unavailable metric leaves the predicate false and the quarantine
    # inert (fail-open to current behaviour) rather than blocking on an
    # absent one.
    #
    # BOUNDED: the countdown is armed on the first refusal and spends one
    # tick per estimation cycle; at expiry the candidate is ACCEPTED. So a
    # mis-lock is DELAYED by at most the window, never cured — curing it is
    # estimator work, not this guard's. There is no early release on
    # dominance because the estimator publishes none: DelayEstimator's
    # confidence is histogram consistency quantised to 0 / 0.5 / 1.0 and
    # carries no quality information (the pre-echo candidate reaches REFINED
    # too), and Path B already gates on it, so persistence — which the window
    # measures — is the only signal available.
    #
    # The unbounded, direction-blind predicate this replaced — and the
    # multipath measurement that killed it — are written up once, on
    # delay_backward_quarantine_enabled in c_impl/include/aec.h.
    delay_backward_quarantine_enabled: bool = False
    # Quarantine window in SECONDS (converted to estimation cycles once, at
    # construction, against the resolved hop/sample rate; minimum 1 cycle).
    # Inert while delay_backward_quarantine_enabled is False.
    delay_backward_quarantine_s: float = 1.0

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
    #
    # TWO MECHANISMS, never both live on one instance. Besides the MATCHED
    # first acquisition described above, this flag also enables the warm
    # tap-transfer inside AEC.apply_external_realign(), the caller-driven
    # realign that only EXTERNAL_ALIGNED has. Clearing it to study acquisition
    # therefore also turns every external realign into a filter reset, i.e.
    # reinstates the "vertical line" in a mode that has no acquisition at all.
    delay_acquire_warm_transfer: bool = True

    # ── Shadow filter (dual-filter divergence control) ──────────────────
    enable_shadow: bool = True
    shadow_copy_threshold: float = 0.65
    # Per-hop EMA alpha (EMA lag ~44.8 ms @ this literal's legacy hop=160/
    # sample_rate=16000 = 10 ms grid) smoothing main/shadow error energies.
    # Wall-clock-authored with zero rate conversion; retimed live in
    # __post_init__ below -- see that block's comment.
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
    # live in __post_init__ below.
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
    # below.
    warmup_frames: int = 80

    # ── Echo-path-change detection (requires shadow) ────────────────────
    epc_delta_threshold: float = 0.3
    epc_total_rise: float = 1.5
    # 20 hops = 200 ms @ legacy hop=160/sample_rate=16000 (10 ms) grid,
    # zero rate conversion; retimed live in __post_init__ below.
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
    # (non-destructive) recovery path cover that tail. The indicator is
    # normally low in far-end single-talk but can false-arm on loud residual
    # echo; that measured trade-off is why its RES floor is not set shallower.
    ne_recent_threshold: float = 0.3
    # 150 hops = 1500 ms @ legacy hop=160/sample_rate=16000 (10 ms) grid,
    # zero rate conversion; retimed live in __post_init__ below.
    ne_recent_hold: int = 150
    # Consecutive-above-threshold hop count required to ARM the hold above.
    # Unlike a genuine event count (e.g. FilterConvergenceAnalyzer's
    # _conv_counter, which SKIPS non-qualifying hops instead of resetting),
    # the counter this drives (_ne_above, in process() below) increments or
    # resets to 0 on EVERY hop with no skip condition, so N consecutive hops
    # IS unconditionally N hops of wall-clock time -- the "genuine event
    # count" framing this field used to carry is wrong.
    #
    # Even so, left as the literal 3 at every grid, deliberately: the three
    # shipped grids' CURRENT "3 hops" already cover three different
    # wall-clock spans -- 24 ms @ 16 kHz/128-hop (default), 48 ms @
    # 16 kHz/256-hop, 32 ms @ 48 kHz/512-hop -- so there is no single
    # canonical-ms value this field was ever actually validated at; unlike
    # ne_recent_hold, "retime it to a canonical ms" is the wrong model here.
    # Routing it through that generic retime anyway (canonical = 3 hops *
    # 10 ms = 30 ms) tightens the default grid to 4 hops, which the
    # duration argument above says it should, but it also LOOSENS the
    # 16 kHz/256-hop grid from 3 hops (48 ms) to 2 (32 ms) -- a 300-case
    # farend-singletalk Rule-A battery measured that this loosening produces
    # real new echo-leak regressions there (confirmed directly, not
    # inferred). A grid-A-only tightening to 4 hops, leaving the other two
    # grids' already-validated "3 hops" untouched, is the better-targeted
    # fix, but it is a different, narrower change than this generic retime
    # path and needs its own dedicated battery before shipping -- not
    # applied here.
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
        # The formed output is produced by the AEC3 post chain's FORM step,
        # and its capture fallback is gated on the filtering-quality verdict
        # that AecState publishes in the same chain. Asking for the formed
        # output while that chain is off leaves the verdict at its
        # constructed False for the whole stream, so the fallback degrades to
        # a bare energy comparison and this seam stops agreeing with the one
        # the board ships. Require the chain rather than silently serving a
        # different signal.
        if self.return_formed_output and not (
                self.enable_res or self.return_res_context):
            raise ValueError(
                "return_formed_output requires enable_res or "
                "return_res_context: the formed seam is produced by the "
                "AEC3 post chain"
            )

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

        # Wall-clock-authored top-level (non-AEC3) constants.
        # shadow_err_alpha / warmup_frames / epc_hangover /
        # ne_recent_hold / filter_misadjustment_{stable,hangover}_frames
        # are authored as literal hop counts / per-hop EMA alphas at the
        # hop=160/sample_rate=16000 (10 ms) grid, with no rate conversion
        # of their own.
        # Retimed here via aec3_scale.ms_to_hops()/growth_rehop() so they
        # cover approximately the same wall-clock duration at every grid.
        #
        # IDEMPOTENCY + CROSS-GRID PRECISION: rebasing from the field's current value
        # (tracking "which grid is the CURRENT field value calibrated for") is
        # correct for repeated same-grid reconstruction, but lossy
        # across a CHAIN of different grids, because ``ms_to_hops()`` rounds
        # to the nearest integer hop and each step in the chain would rebase from
        # the PREVIOUS step's already-rounded hop count instead of the
        # original canonical millisecond value (e.g. 16k/512 -> 48k/1024 ->
        # 16k/512 would land one hop off the true original). So the
        # canonical ms/retention values are captured ONCE (into the ``_canonical_*``
        # fields above, pinned for the lifetime of the value chain since
        # they are regular fields that survive replace()/asdict()
        # roundtrips unchanged) and the live fields are ALWAYS re-derived
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
        # ne_recent_sustain: left as the literal 3 at every grid -- see its
        # field comment above for why (duration in disguise, but a generic
        # canonical-ms retime regresses the already-validated 16 kHz/256-hop
        # grid; a grid-A-only fix is recommended, not applied here).
        self.filter_misadjustment_stable_frames = _aec3_scale.ms_to_hops(
            self._canonical_ms_filter_misadjustment_stable_frames,
            self.hop_size, self.sample_rate)
        self.filter_misadjustment_hangover_frames = _aec3_scale.ms_to_hops(
            self._canonical_ms_filter_misadjustment_hangover_frames,
            self.hop_size, self.sample_rate)

        if self.dt_protect_source not in ('energy', 'dne'):
            raise ValueError(
                f"dt_protect_source must be 'energy' or 'dne', got "
                f"{self.dt_protect_source!r}"
            )
        # One >= 0 loop for every non-negative-only duration/count/strength
        # field across dt_protect/live_observer -- two standalone checks
        # plus two separate per-group loops, all the same shape, were added
        # across two review passes; folded into one.
        for _name in ('dt_protect_extra_hold_ms', 'dt_protect_confirm_ms',
                      'live_observer_shift_ms', 'live_observer_max_hz',
                      'live_observer_corrob_ms', 'live_observer_inject_dt_energy_max'):
            if getattr(self, _name) < 0:
                raise ValueError(f"{_name} must be >= 0, got {getattr(self, _name)}")
        if not 0.0 < self.live_observer_mu <= 2.0:
            raise ValueError(
                f"live_observer_mu must be in (0, 2], got {self.live_observer_mu}"
            )
        # aec.c's aec_validate_config() rejects a non-finite
        # live_observer_corrob_margin_db; this side had no equivalent check,
        # so a NaN was silently accepted here (comparisons against NaN are
        # always False, so `_corroborated` would just always read False --
        # a quiet divergence from C's hard rejection of the same input).
        if not math.isfinite(self.live_observer_corrob_margin_db):
            raise ValueError(
                f"live_observer_corrob_margin_db must be finite, got "
                f"{self.live_observer_corrob_margin_db}"
            )
        for _name in ('rsa_mask_hold_ms', 'rsa_poor_excitation_hold_ms'):
            if getattr(self, _name) <= 0:
                raise ValueError(f"{_name} must be > 0, got {getattr(self, _name)}")

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
            # Generous but finite, and rate-relative: in FIXED the reference
            # ring is sized fixed + hop (the 4096-sample headroom belongs to
            # the MATCHED search ring, which this mode does not build), so an
            # unbounded value here would drive an unbounded allocation. 120 s
            # matches delay_buffer_ms's own 120000 ms upper bound on the C
            # side.
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
        """Four Pareto operating points on the residual-echo strength axis.

        All presets share the same AEC3 post-filter chain + the 800-case-tuned
        base (CNG, shadow/Kalman warmup). mild/balanced differ from each
        other ONLY on ``min_gain_floor_far_active_db`` — the far-active
        residual-gain floor that trades echo suppression against near-end
        preservation. aggressive ALSO deepens ``min_gain_floor_dt_db`` (see
        below); the fourth tier removes the floor mechanism entirely instead
        of deepening it further:

          mild       near-priority  −20 dB / −20 dB  (higher floor → more near kept, more echo leak)
          balanced   shipped         −28 dB / −20 dB  the shipped all-four-bars-met operating point
          aggressive echo-priority  −38 dB / −30 dB  (deeper floors → more echo killed, more near loss)
          no_floor   floor OFF      (n/a)            ``min_gain_split_floor_enabled=False`` — gain is
                                                       purely audibility/R²-derived (AEC3-like), with no
                                                       fixed-dB clamp at all; the deepest point on the same
                                                       axis, and the most near-end loss.

        (far_active dB / dt dB above.) mild/balanced leave ``min_gain_floor_
        dt_db`` at its shared dataclass default (−20) because their own
        far_active values are already at or above it — widening it would not
        change their behavior, only aggressive's deepened far_active (−38)
        makes a distinct, shallower dt floor (−30) actually visible: without
        this, the DT-lifted floor (``dt_protect_active``, see below) hard-
        overrides far_active with the SAME shared −20 dB for all three
        presets whenever it fires — which includes a measured ~30% false-arm
        rate on pure far-end-singletalk hops (see dt_aware_res_floor_enabled
        below) — so mild/balanced/aggressive used to collapse to
        indistinguishable behavior on exactly the hops where a user would
        most expect to hear the strength difference. 47-case dev-pool
        measurement (both grids, vs today's shared-−20 aggressive): DT_static
        echo MOS +0.17/+0.14, deg −0.16/−0.14; near-end 200ms clean-anchored
        guard (44-case mix battery) drops from 42/48→39/48 (grid A) and
        44/48→43/48 (grid B) — a real, bounded near-end cost, consistent
        with aggressive's own definition (echo-priority, more near-end loss),
        not a free lunch.

        ``min_gain_split_floor_enabled`` is the single gate for ALL THREE floor
        variants in ``suppression_gain.py``'s ``_get_min_gain`` — far-silent
        (``min_gain_floor_far_silent_db``, applied when the per-recording
        far-active latch never fires — NOT simply "pure NE": some NE-scenario
        recordings have non-trivial far-end energy and DO latch, landing on
        far-active/DT instead), far-active (``min_gain_floor_far_active_db``),
        AND the DT-lifted floor (``min_gain_floor_dt_db``). no_floor removes
        all three in one switch, not just the far-active one mild/aggressive
        vary. Unlike mild/balanced/aggressive (NE-neutral by construction on
        unlatched NE: see commit ec47063's 800-case NE deg 4.052/4.047/4.046),
        no_floor is expected to cost near-end-only quality too — via
        far_silent on unlatched recordings, or via far_active/DT (same as a
        DT case) on latched ones. ``dt_aware_res_floor_enabled`` also becomes
        behaviorally inert under no_floor (its only consumer is inside this
        same gate).

        mild/aggressive/no_floor deliberately sit off the balanced ship-bar set
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
        # Strength axis: far-active min-gain floor (dataclass default −28 = balanced),
        # or — for no_floor — remove the floor mechanism itself. min_gain_floor_
        # far_active_db is inert when min_gain_split_floor_enabled=False, so no_floor
        # deliberately does not also set it. aggressive additionally deepens
        # min_gain_floor_dt_db (dataclass default −20, shared by mild/balanced
        # unchanged) -- see the docstring above for why only aggressive needs it.
        strength = {
            AecPreset.MILD:     dict(min_gain_floor_far_active_db=-20.0),
            AecPreset.BALANCED:   dict(),
            AecPreset.AGGRESSIVE: dict(min_gain_floor_far_active_db=-38.0,
                                       min_gain_floor_dt_db=-30.0),
            AecPreset.NO_FLOOR:   dict(min_gain_split_floor_enabled=False),
        }
        defaults = {**base, **strength.get(preset, {})}
        defaults.update(kwargs)
        return cls(**defaults)
