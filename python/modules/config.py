"""AecConfig dataclass + BALANCED preset.

Release-cleaned (v3.21.6 release). All dev-time substrate / ablation /
NOSHIP flags have been removed; AEC3-alignment flags that were
default-True are now hard-coded into the call sites. The remaining
fields are customer-facing or system parameters only.
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

    # ── Backwards-compat shim for legacy substrate flags ───────────────
    #
    # The cleanup removed many dev-time ablation flags from the public
    # surface. Some internal call sites still reference them; this shim
    # returns the hard-coded production value (True for AEC3-alignment
    # flags that shipped on, False/0 for dev-time NOSHIP flags) so those
    # sites continue to behave as if the flag were locked. Customer-
    # facing presets cannot set these — they are not dataclass fields.
    _LEGACY_HARDCODE_TRUE = frozenset({
        'use_aec3_filter_misadjustment_parity',
        'use_per_bin_h_error_refresh',
        'use_aec3_h_error_ceil',
        'use_partition_summed_x2_for_h_error_gain',
        'use_aec3_filter_noise_gate_power',
        'use_aec3_residual_noise_gate',
        'use_aec3_echo_gen_power_window',
        'use_aec3_handle_echo_path_change',
        'use_full_delay_change_chain',
        'use_current_e2_refined_in_h_error_denominator',
        'use_refined_output_selection_for_linear_path',
        'form_linear_filter_crossfade_enabled',
        'use_partition_summed_x2_for_shadow_mu',
        'use_aec3_noise_gate_for_shadow',
        'use_poor_excitation_gate_for_shadow',
        'use_narrowband_mask_for_shadow',
        'use_saturation_gate_for_shadow',
        'filter_misadjustment_enabled',
        'shadow_class_nlms',
        'filter_analyzer_enabled',
        'e2_y2_clamp_enabled',
        'aec3_post_stationarity_zero_enabled',
        'delay_fast_path_enabled',
    })
    _LEGACY_HARDCODE_FALSE = frozenset({
        'aec3_misadj_trace_enabled',
        'coarse_filter_converged_relaxed_enabled',
        'e_stat_aware_ne_proxy_enabled',
        'reverb_tail_dead_fallback_enabled',
        'transparent_mode_enabled',
        'filter_quality_enabled',
        'use_aec3_erle_reverb_quality',
        'use_aec3_zero_filter_on_epc',
        'use_aec3_epc_classification',
        'skip_aec3_gain_change_dispatch_at_epv_shadow_rise',
        'use_q1_true_delay_transient_leakage',
        'use_q1_terminate_on_non_delay_epc',
        'enable_plateau_detector',
        'use_full_delay_plateau_suppression',
        'use_full_delay_plateau_suppression_fixed_hops',
        'use_full_delay_setconfig_initial',
        'use_coarse_e2_time_domain_parity',
        'use_linear_filter_output_selection_for_final_output',
        'use_aec3_poor_coarse_rescue_copy',
        'usable_linear_require_filter_analyzer_consistent',
        'usable_linear_disable_external_delay_shortcut',
        'usable_linear_trusted_external_delay_only',
        'saturation_subtractor_inputs_enabled',
        'trace_p52_regime_handler',
        'trace_p53_innovation',
        'trace_hf_chain',
        'plan_a_kernel_tight',
        'plan_a_hf_cap_2k',
        'plan_a_stat_mask_7k',
        'plan_b_dt_per_bin_gamma',
        'use_epc_state_reset',
        'hf_cap_conditional',
        'dt_advisory_enabled',
        'dt_advisory_use_p3f_state',
        'diverged_reset_enabled',
        'diverged_reset_triple_and',
        'use_diverged_streak_ema',
        'epc_r_reset_enabled',
        'shadow_r_reset_enabled',
        'shadow_mu_state_aware',
        'f_e1_enabled',
        'f_delaytrack_enabled',
        'f_e3_enabled',
        'f_e5_enabled',
        'mu_holdoff_no_reset',
        'dtd_conf_two_stage',
        'reverse_copy_p_reset',
        'shadow_state_decoupled',
        'res_consume_filter_state',
        'res_unified_gain_floor',
        'res_dt_per_bin_unified',
        'res_cap2_fs_loosen',
        'res_noise_floor_refined',
        'res_per_band_enr',
        'kalman_q_per_band',
        'arc_m_epc_gated',
        'dt_ne_compression_fix',
        'subband_ne_detect_enabled',
        'res_mask_profile_swap_enabled',
        'dominant_ne_detect_enabled',
        'aec_event_classification_enabled',
    })
    _LEGACY_HARDCODE_DEFAULTS = {
        # int / numeric constants for dev-time flags whose call sites use
        # them as bounds, thresholds, or counts. Values chosen to keep
        # the dependent branch dormant (the gating bool above is False).
        'aec3_misadj_parity_n_hops': 2,
        'usable_linear_convergence_hops_required': 0,
        'signal_dependent_erle_sections': 0,
        'sde_num_blocks': 13,
        'sde_delay_headroom_blocks': 0,
        'q1_tdt_transient_hops': 250,
        'q1_tdt_smoothing_hops': 100,
        'q1_tdt_lc_factor': 5.0,
        'q1_tdt_ld_factor': 5.0,
        'diverged_reset_streak_frames': 50,
        'diverged_reset_cooldown_frames': 400,
        'diverged_reset_triple_and_shadow_adv_min': 2.0,
        'diverged_streak_ema_alpha': 0.95,
        'diverged_streak_ema_threshold': 0.7,
        'f_e3_consecutive_window_frames': 100,
        'f_e3_w_reset_min_gap_frames': 1000,
        'f_e3_w_reset_factor': 0.5,
        'f_e5_main_mu_sat_threshold': 0.5,
        'f_e5_mic_softclip_threshold': 0.3,
        'plateau_suppression_fixed_hops': 500,
        'reverb_tail_dead_threshold_frames': 50,
        'reverb_tail_dead_fallback_strength': 0.25,
        'e_stat_aware_ne_proxy_threshold': 0.10,
        'hf_cap_metric_threshold': 0.30,
        'dt_advisory_shadow_th': 0.5,
        'dt_advisory_energy_th': 0.4,
        'dt_advisory_hold_ms': 400.0,
        'dt_advisory_mu_factor': 0.3,
        'subband_ne_sub1_low': 192,
        'subband_ne_sub1_high': 320,
        'subband_ne_sub2_low': 32,
        'subband_ne_sub2_high': 128,
        'subband_ne_threshold': 0.5,
        'subband_ne_snr_threshold': 30.0,
        'dominant_ne_lf_low': 4,
        'dominant_ne_lf_high': 60,
        'dominant_ne_enr_threshold': 0.25,
        'dominant_ne_enr_exit_threshold': 10.0,
        'dominant_ne_snr_threshold': 30.0,
        'dominant_ne_trigger_threshold': 12,
        'dominant_ne_hold_duration': 50,
        'res_mask_last_lf_band': 20,
        'res_mask_first_hf_band': 32,
        'res_mask_normal_lf': (0.3, 0.4, 0.3),
        'res_mask_normal_hf': (0.07, 0.1, 0.3),
        'res_mask_nearend_lf': (1.09, 1.1, 0.3),
        'res_mask_nearend_hf': (0.1, 0.3, 0.3),
        'res_mask_ne_gate_dt': 0.3,
        'res_mask_swap_mode': 'binary',
        'res_mask_fs_overlay_coh2_min': 0.85,
        'res_mask_fs_overlay_dt_max': 0.2,
        'enr_t_ne_per_band': (2.0, 1.5, 1.0),
        'enr_s_ne_per_band': (3.33, 2.5, 1.67),
        'kalman_q_band_scales': (0.5, 1.0, 2.0),
        'dt_ne_state_scale': None,
        'dt_ne_per_bin_thresh': 0.5,
        'dt_ne_per_bin_scale': 2.0,
        'delay_fast_par_threshold': 40.0,
        'regime_gate_mode': 'energy',
        'trace_p52_regime_handler_path': '',
    }

    def __getattr__(self, name):
        # Only invoked when the normal lookup fails (i.e. the attribute is
        # not on the instance/class). Provides legacy-flag fallbacks.
        if name in type(self)._LEGACY_HARDCODE_TRUE:
            return True
        if name in type(self)._LEGACY_HARDCODE_FALSE:
            return False
        if name in type(self)._LEGACY_HARDCODE_DEFAULTS:
            return type(self)._LEGACY_HARDCODE_DEFAULTS[name]
        raise AttributeError(name)

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
