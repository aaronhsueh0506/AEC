"""AecConfig dataclass and the four strength presets (release surface; fields marked INTERNAL are not customer knobs)."""
import math
from dataclasses import dataclass, field
from typing import NamedTuple, Optional, Tuple

from . import aec3_scale as _aec3_scale
from .enums import AecDelayMode, AecMode, AecPreset


# Signal grid: (sample_rate, fft_size) fully determines the frame geometry (frame_size == fft_size,
# hop_size == fft_size // 2); ``resolve_signal_grid()`` is the only derivation. fft_size must be explicit
# because 16 kHz has two production grids; ``default_fft_size()`` supplies the product default.
class SignalGrid(NamedTuple):
    """A fully resolved, self-consistent frame geometry."""

    sample_rate: int
    fft_size: int          # == frame_size in this project
    frame_size: int
    hop_size: int          # == fft_size // 2
    n_freqs: int           # == fft_size // 2 + 1
    is_legacy: bool        # True for 8 kHz: supported, not a product grid


# 8 kHz is supported by this library and the mono pipeline, but is not a product grid and is not
# supported by the 4-channel pipeline.
_GRID_TABLE: Tuple[SignalGrid, ...] = (
    SignalGrid(16000,  256,  256, 128, 129, False),   # product, 16k default
    SignalGrid(16000,  512,  512, 256, 257, False),   # product, 16k alternate
    SignalGrid(48000, 1024, 1024, 512, 513, False),   # product
    SignalGrid( 8000,  256,  256, 128, 129, True),  # 8 kHz: supported, not a product grid
)


def default_fft_size(sample_rate: int) -> int:
    """Product-default fft_size for a rate (first table row: 16 kHz -> 256, 48 kHz -> 1024, 8 kHz -> 256), or 0 if unsupported."""
    for g in _GRID_TABLE:
        if g.sample_rate == sample_rate:
            return g.fft_size
    return 0


def resolve_signal_grid(sample_rate: int, fft_size: int,
                        hop_size: Optional[int] = None) -> SignalGrid:
    """Resolve (sample_rate, fft_size) into a full grid, or raise; ``hop_size``, if given, is only checked against it."""
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


# Headroom added to the MATCHED search ring (not FIXED); it also feeds the new-delay eligibility gate.
REF_RING_MATCHED_HEADROOM = 4096


def ref_ring_samples(config, hop_size: int) -> int:
    """Reference alignment ring capacity in samples: MATCHED ``max(delay_buffer_ms, max_delay_ms + headroom)``,
    FIXED ``fixed_delay_samples + hop_size``, EXTERNAL_ALIGNED 0."""
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
    """AEC configuration (release surface). Sizes are samples, rates Hz; fields marked INTERNAL are not customer knobs."""

    # ── System / framing ────────────────────────────────────────────────
    sample_rate: int = 16000        # 8000 / 16000 / 48000
    # No-padding grid: auto 256 at 8 and 16 kHz, 1024 at 48 kHz; 16 kHz also accepts 512 (16 ms hop).
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
    # Round-robin constraint: constrain one filter partition per hop (cycling) instead of all of them.
    constraint_round_robin: bool = True

    # Coherence-based ERLE gate: freeze per-bin SubbandErle updates where the coherence between the echo
    # estimate and the capture falls below the threshold (double-talk decorrelates them).
    # erle_coh_gate_alpha is the cross-PSD EMA rate (0.05 is about 200 ms).
    erle_coh_gate_enabled: bool = True
    erle_coh_gate_threshold: float = 0.5
    erle_coh_gate_alpha: float = 0.05

    # Split min-gain floor (amplitude dB): caps the deepest suppression so the AEC3 floor cannot collapse in
    # double-talk. Far-active (echo present) uses min_gain_floor_far_active_db, far-silent (pure near-end)
    # min_gain_floor_far_silent_db. Routing latches on instantaneous far energy
    # (mean(render_block^2) > min_gain_far_latch_power, int16-scaled) and stays latched for the stream.
    min_gain_split_floor_enabled: bool = True
    min_gain_floor_far_active_db: float = -28.0
    min_gain_floor_far_silent_db: float = -12.0
    min_gain_far_latch_power: float = 1.0e6
    # DT-gated floor lift: with the far-active latch set and near-end recently present, min_gain_floor_dt_db
    # replaces the far-active floor.
    dt_aware_res_floor_enabled: bool = True
    min_gain_floor_dt_db: float = -20.0
    # Per-bin floor blend (off): bins blend toward min_gain_floor_dt_db in proportion to their own ENR
    # (the soft_nearend_blend_per_bin sigmoid); echo-dominant bins keep the far-active floor.
    per_bin_floor_enabled: bool = False
    # RSA persistence in wall-clock ms (off): the narrow-band mask and poor-excitation hold use the ms fields
    # below (at least 1 hop) instead of fixed hop counts. Defaults match the 16 kHz / 256-hop grid.
    rsa_wallclock_persistence: bool = False
    rsa_mask_hold_ms: float = 96.0
    rsa_poor_excitation_hold_ms: float = 176.0
    # dt-protect arming source: 'energy' (pre-filter mic-energy excess) or 'dne' (the dominant-near-end
    # detector, read one hop late).
    dt_protect_source: str = 'energy'
    # Extra hold (ms) after the 'dne' source releases; 0 = none. Only used when dt_protect_source == 'dne'.
    dt_protect_extra_hold_ms: float = 0.0
    # Confirmation delay (ms): 'dne' must read True longer than this before dt-protect arms; 0 = immediate.
    # Only used when dt_protect_source == 'dne'.
    dt_protect_confirm_ms: float = 0.0
    # Live low-frequency observer: a second NLMS filter reading the far-end ring live_observer_shift_ms earlier
    # than the main filter, covering the span just ahead of the main window. It never subtracts from the
    # output; it only raises r2 (max(r2, evidence)).
    live_observer_enabled: bool = False
    live_observer_shift_ms: float = 24.0
    live_observer_max_hz: float = 1000.0
    live_observer_mu: float = 0.3
    # Injection guard: evidence is trusted only while the observer's LF residual energy, EMA'd over
    # live_observer_corrob_ms, is no worse than the main filter's (margin in dB; negative = stricter).
    live_observer_corrob_ms: float = 100.0
    live_observer_corrob_margin_db: float = 0.0
    # Injection-only near-end discriminator: evidence is zeroed while dt_from_energy exceeds this value
    # (lower is stricter).
    live_observer_inject_dt_energy_max: float = 0.5
    # Cold-start deadlock breaker (PBFDKF): on hard echo paths W ~ 0 gives no H_error refresh, so mu dies.
    # h_error_refresh_erl_floor floors the ERL used in the refresh (0 = off); h_error_floor_override raises
    # the H_error clamp floor (0 = default 1e-3; higher = more adaptation, less steady-state stability).
    h_error_refresh_erl_floor: float = 0.0
    h_error_floor_override: float = 0.0

    # Soft near-end tuning blend: sigmoid((enr_threshold - enr_lf) / softness) weights nearend_tuning against
    # normal_tuning on the LF-band ENR. enr_threshold 0.25 is the dominant-near-end pivot; softness sets the
    # transition width.
    soft_nearend_blend_enabled: bool = True
    soft_nearend_blend_enr_threshold: float = 0.25
    soft_nearend_blend_softness: float = 0.25
    # Per-bin near-end blend: per-bin ENR weights replace the scalar LF weight (near-dominant bins toward
    # nearend_tuning, echo-dominant bins toward normal_tuning).
    soft_nearend_blend_per_bin: bool = True

    # Second-order nonlinear residual: R2_nl = nl_r2_alpha * x2^2 / 1.07e7 is added to the nonlinear R2 path
    # only (usable_linear False); 1.07e7 is the -20 dBFS per-bin reference in int16^2 units.
    nl_r2_enabled: bool = True
    nl_r2_alpha: float = 0.1

    # Cap on r2_nl as a multiple of the direct term (x2 * echo_path_gain); 0.0 disables.
    nl_r2_cap_multiple: float = 300.0

    # ERLE render-activity PSD scale: scale the ERLE estimators' render power to the int16^2 scale of the
    # activity threshold; without it the gate never opens and the reverb model stays idle.
    erle_render_x2_psd_scale: bool = True

    # Scale of the additive reverb-tail R2 (1.0 = full, < 1.0 more conservative).
    reverb_tail_strength: float = 1.0

    # ERLE Y2/E2 coordinate consistency: feed the ERLE estimator a sqrt-Hann windowed Y2 so Y2 and E2 share one window.
    erle_windowed_capture_psd: bool = True

    # CNG N2 tracker input: use the windowed capture PSD (shared with erle_windowed_capture_psd) instead of the
    # rectangular one.
    cng_n2_windowed_psd: bool = False

    # Output-capture guard (rendering base only, never the analysis seam): when the linear filter is unusable this
    # hop and frame |E|^2 > |Y|^2, apply the gain to the windowed capture Y instead of the residual E. Decided once
    # per hop for every API mode; the context exposes it as output_base_spec / guard_active.
    output_capture_when_linear_unusable: bool = True

    # Reject a late first delay acquisition while the linear filter already cancels (> 2.5 dB windowed ERLE)
    # at the current alignment.
    delay_acquire_protect_converged: bool = True

    # Recent inst-ERLE peak (dB) over the last ~15 frames; threshold for delay_acquire_warm_transfer.
    delay_acquire_inst_erle_db: float = 4.0

    # Warm tap-transfer on delay acquisition: instead of zeroing the filter, shift the learned impulse response
    # by the acquired delay (when the filter was already cancelling and the delay is within tap reach).
    # It also enables the warm transfer in AEC.apply_external_realign(); clearing it turns every external
    # realign into a filter reset.
    delay_acquire_warm_transfer: bool = True

    # ── Shadow filter (dual-filter divergence control) ──────────────────
    enable_shadow: bool = True
    shadow_copy_threshold: float = 0.65
    # Per-hop EMA of the main/shadow error energies (retention form, retimed in __post_init__).
    shadow_err_alpha: float = 0.80
    shadow_mu_min: float = 0.5
    shadow_copy_hysteresis: int = 3
    shadow_dtd_advantage_scale: float = 3.0
    shadow_dtd_offset: float = 1.5
    # INTERNAL: shadow is always PBFDAF NLMS; mu kept for preset wiring.
    shadow_mu_nlms: float = 0.5

    # Filter misadjustment estimator (INTERNAL): hold-off and required-stability hop counts, retimed in __post_init__.
    filter_misadjustment_hangover_frames: int = 100
    filter_misadjustment_stable_frames: int = 30
    filter_misadjustment_scale_min: float = 0.5
    filter_misadjustment_scale_max: float = 2.0

    # ── PBFDKF (Kalman) ─────────────────────────────────────────────────
    use_kalman: bool = True
    kalman_q_high: float = 1e-3
    kalman_q_low: float = 1e-6
    # Warm-up hops (retimed in __post_init__); from_preset sets 100 for every preset.
    warmup_frames: int = 80

    # ── Echo-path-change detection (requires shadow) ────────────────────
    epc_delta_threshold: float = 0.3
    epc_total_rise: float = 1.5
    # Hangover hops (retimed in __post_init__).
    epc_hangover: int = 20

    # ── Delay estimation (matched-filter + ring buffer) ─────────────────
    # Far/mic alignment mode (see AecDelayMode): enum, int or name string, normalised in __post_init__; fixed at construction.
    delay_mode: AecDelayMode = AecDelayMode.MATCHED
    # Deprecated mirror of delay_mode, folded into it in __post_init__; read delay_mode instead.
    enable_delay_est: bool = True
    max_delay_ms: float = 1024.0
    delay_buffer_ms: float = 2048.0
    delay_est_period_s: float = 0.5
    delay_est_init_s: float = 0.3
    # FIXED only: measured system delay in native-rate samples (>= 0); must stay -1 in the other modes.
    fixed_delay_samples: int = -1
    delay_par_low_threshold: float = 5.0
    delay_par_solid_threshold: float = 8.0
    # Matched-filter bank size (1..5). Fewer filters cut compute (about 4.2 MMAC/s each)
    # and shrink reach to (n - 1) * 384 + 501 downsampled samples of 0.25 ms:
    # n = 1..5 -> 125 / 221 / 317 / 413 / 509 ms.
    delay_num_filters: int = 5
    # DT-aware soft-recovery gate: a held "near-end seen recently" flag lets the non-destructive recovery path
    # cover the re-convergence tail that would otherwise overfit near-end speech. It can false-arm on loud residual echo.
    ne_recent_threshold: float = 0.3
    # Hold hops (retimed in __post_init__).
    ne_recent_hold: int = 150
    # Consecutive above-threshold hops needed to arm the hold; a hop count on every grid, not retimed.
    ne_recent_sustain: int = 3
    # DT-aware soft recovery: while near-end was seen recently, delay/EPC recovery triggers realign softly
    # (keep the converged filter) instead of a full reset with P = 1.0 and mark_diverged.
    dt_aware_recovery_soft: bool = True

    # ── High-pass filtering ─────────────────────────────────────────────
    enable_highpass: bool = True
    highpass_cutoff_hz: float = 80.0

    # ── Saturation / non-linear echo handling ───────────────────────────
    enable_saturation_detect: bool = True
    saturation_threshold: float = 0.95
    saturation_softclip_ref: bool = True

    # ── Mode / output ───────────────────────────────────────────────────
    mode: AecMode = AecMode.PBFDKF
    return_res_context: bool = False
    # Expose the shadow/main-selected, crossfaded, WOLA-formed linear hop via get_formed_output();
    # it adds work only when requested and does not change adaptation.
    return_formed_output: bool = False
    clear_filter_history: bool = False

    # INTERNAL: pinned canonical values (wall-clock authored at the 10 ms grid) of the retimed fields, captured once;
    # -1.0 = not yet captured. __post_init__ re-derives the live fields from them, so chained grid changes do not drift.
    _canonical_ms_warmup_frames: float = field(default=-1.0, repr=False)
    _canonical_ms_epc_hangover: float = field(default=-1.0, repr=False)
    _canonical_ms_ne_recent_hold: float = field(default=-1.0, repr=False)
    _canonical_ms_filter_misadjustment_stable_frames: float = field(default=-1.0, repr=False)
    _canonical_ms_filter_misadjustment_hangover_frames: float = field(default=-1.0, repr=False)
    _canonical_retention_shadow_err_alpha: float = field(default=-1.0, repr=False)

    def __post_init__(self):
        # The formed output comes from the AEC3 post chain, so it requires enable_res or return_res_context.
        if self.return_formed_output and not (
                self.enable_res or self.return_res_context):
            raise ValueError(
                "return_formed_output requires enable_res or "
                "return_res_context: the formed seam is produced by the "
                "AEC3 post chain"
            )

        # Signal grid: frame_size = -1 expands to the rate's default, then the strict resolver validates.
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

        # Matched-filter bank size: fail fast instead of clamping (the upper bound is the C array bound).
        if not 1 <= self.delay_num_filters <= 5:
            raise ValueError(
                f"delay_num_filters must be in [1, 5], got "
                f"{self.delay_num_filters}"
            )

        self._resolve_delay_mode()

        # Wall-clock-authored fields (hop counts and per-hop alphas authored at the 10 ms grid) are re-derived
        # from the pinned canonical values for the resolved grid, so repeated or chained reconstruction never drifts.
        if self._canonical_ms_warmup_frames < 0:
            # First run of this value chain: capture the canonical values, reading the current field values as authored at the 10 ms grid.
            self._canonical_ms_warmup_frames = self.warmup_frames * 10.0
            self._canonical_ms_epc_hangover = self.epc_hangover * 10.0
            self._canonical_ms_ne_recent_hold = self.ne_recent_hold * 10.0
            self._canonical_ms_filter_misadjustment_stable_frames = (
                self.filter_misadjustment_stable_frames * 10.0)
            self._canonical_ms_filter_misadjustment_hangover_frames = (
                self.filter_misadjustment_hangover_frames * 10.0)
            self._canonical_retention_shadow_err_alpha = self.shadow_err_alpha

        # shadow_err_alpha is a retention-form EMA constant, so growth_rehop applies.
        self.shadow_err_alpha = _aec3_scale.growth_rehop(
            self._canonical_retention_shadow_err_alpha, 160, 16000,
            self.hop_size, self.sample_rate)
        self.warmup_frames = _aec3_scale.ms_to_hops(
            self._canonical_ms_warmup_frames, self.hop_size, self.sample_rate)
        self.epc_hangover = _aec3_scale.ms_to_hops(
            self._canonical_ms_epc_hangover, self.hop_size, self.sample_rate)
        self.ne_recent_hold = _aec3_scale.ms_to_hops(
            self._canonical_ms_ne_recent_hold, self.hop_size, self.sample_rate)
        # ne_recent_sustain stays at 3 hops on every grid (see its field comment).
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
        for _name in ('dt_protect_extra_hold_ms', 'dt_protect_confirm_ms',
                      'live_observer_shift_ms', 'live_observer_max_hz',
                      'live_observer_corrob_ms', 'live_observer_inject_dt_energy_max'):
            if getattr(self, _name) < 0:
                raise ValueError(f"{_name} must be >= 0, got {getattr(self, _name)}")
        if not 0.0 < self.live_observer_mu <= 2.0:
            raise ValueError(
                f"live_observer_mu must be in (0, 2], got {self.live_observer_mu}"
            )
        if not math.isfinite(self.live_observer_corrob_margin_db):
            raise ValueError(
                f"live_observer_corrob_margin_db must be finite, got "
                f"{self.live_observer_corrob_margin_db}"
            )
        for _name in ('rsa_mask_hold_ms', 'rsa_poor_excitation_hold_ms'):
            if getattr(self, _name) <= 0:
                raise ValueError(f"{_name} must be > 0, got {getattr(self, _name)}")

    # Delay-mode translation and validation (same contract as the C resolver). enable_delay_est True leaves
    # delay_mode alone; False with delay_mode at MATCHED resolves to FIXED when fixed_delay_samples >= 0, else
    # EXTERNAL_ALIGNED; False with an explicit non-MATCHED mode keeps it. The mirror is rewritten from the
    # resolved mode, so re-running __post_init__ is idempotent.
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
            # Bounded and rate-relative (120 s): the FIXED ring is sized fixed + hop.
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
            # A non-default bank size has no effect outside MATCHED.
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
        """Four operating points on the residual-echo strength axis (far-active / DT min-gain floors, amplitude dB).

          mild        -20 / -20  more near-end kept, more echo leakage
          balanced    -28 / -20  default
          aggressive  -38 / -30  deeper floors, more near-end loss
          no_floor    ``min_gain_split_floor_enabled=False``: deepest suppression; ``dt_aware_res_floor_enabled`` becomes inert.
        """
        if isinstance(preset, str):
            preset = AecPreset(preset)
        base = dict(
            enable_cng=True,
            shadow_mu_min=0.5,
            warmup_frames=100,
            kalman_q_high=1e-3,
        )
        # Strength axis: far-active floor (and DT floor for aggressive); no_floor disables the floor mechanism.
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
