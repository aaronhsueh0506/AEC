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
    # AEC3-strict wall-clock alignment of DominantNearendDetector
    # trigger_threshold (48 ms = 5 hops at hop=160/sr=16k, vs legacy
    # 12 hops = 120 ms = 2.5× AEC3). Default OFF for byte-equal.
    use_aec3_wallclock_dne_trigger_threshold: bool = False
    # AEC3-strict wall-clock alignment of ReverbFrequencyResponse EMA
    # smoothing α (0.2·quality per AEC3 4 ms block → per-hop equivalent
    # 0.428·quality at hop=160). Verbatim 0.2 per hop is 2.5× too slow,
    # making average_decay stick to stale FS values when entering DT2 →
    # tail_response stays inflated → SG wipes HF ("painted-black" after
    # far-end-single-talk). Default OFF for byte-equal.
    use_aec3_wallclock_reverb_smoothing: bool = False
    # AEC3-strict ``JustResetEchoPath`` linear-path gate. While the
    # poor-coarse rescue hangover counter is non-zero (= our analogue of
    # AEC3's recent-reset event), force the RES + SG path to behave as if
    # ``usable_linear_estimate == False``: nearend reference reverts to
    # raw Y² and R² goes through the nonlinear path (R² = X² · default_gain²).
    # Without this, after a rescue copy the linear residual + stale
    # ERLE/reverb continue flowing into SG for ~100 ms, inflating R² and
    # crushing HF gain.
    use_aec3_just_reset_gate_on_linear_path: bool = False
    # AEC3-strict ResidualEchoEstimator reset on the rising edge of a
    # rescue-arming event. Mirrors AEC3 ``ResidualEchoEstimator::Reset()``
    # being called via ``EchoRemoverImpl::HandleEchoPathChange``: clears
    # ReverbModel state, ReverbFrequencyResponse, x2_noise_floor counter.
    # Without this, stale FS-period reverb tail persists into DT2.
    use_aec3_reset_res_on_rescue_edge: bool = False
    # AEC3-strict fft-density rescale of per-bin PSD floor constants.
    # AEC3 hardcodes 64 = kFftLengthBy2 in:
    #   * ComfortNoiseGenerator::GetNoiseFloorFactor (CN noise floor base)
    #   * EchoAudibilityConfig.floor_power (= 2 × 64)
    #   * EchoAudibilityConfig.low_render_limit (= 4 × 64)
    #   * EchoAudibilityConfig.normal_render_limit (= 64)
    #   * EchoModelConfig.min_noise_floor_power (1638400 = AEC3 fft scale)
    # Plus _LowNoiseRenderDetector's `50² × kBlockSize = 160000`
    # time-domain energy threshold (scales with hop, not fft).
    # Verbatim ports keep AEC3 numerics, which at our fft=512 (vs 128)
    # leave per-bin floors 4× too low — WeightEchoForAudibility never
    # downweights weak HF echo, GetMinGain's protection floor is too
    # low, EMR-bypass never fires → formant valleys / broadband HF
    # fricatives get painted-black even when filter+linear-path output
    # is clean. Flag ON applies fft_density_scale(..., fft_size) (4×)
    # to the five fft-density constants and block_energy_scale(...,
    # hop_size) (2.5×) to LowNoiseRender threshold. Default OFF for
    # byte-equal; toggled per-case via tracer during validation.
    use_aec3_fft_density_scaled_psd_floors: bool = False

    # ── Shadow filter (dual-filter divergence control) ──────────────────
    enable_shadow: bool = True
    shadow_mu_ratio: float = 1.0
    shadow_copy_threshold: float = 0.65
    shadow_err_alpha: float = 0.80
    shadow_mu_min: float = 0.5
    shadow_copy_hysteresis: int = 3
    shadow_q_ratio: float = 3.0
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
                shadow_q_ratio=3.5,
                shadow_mu_min=0.5,
                warmup_frames=100,
                kalman_q_high=1e-3,
            )
        else:
            defaults = {}
        defaults.update(kwargs)
        return cls(**defaults)
