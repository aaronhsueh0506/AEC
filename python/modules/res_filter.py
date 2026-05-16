"""ResFilter — 9-stage residual echo suppressor.

Extracted from ``aec.py`` during refactor R.9 (verbatim, byte-equal).
Largest module in the AEC stack at ~2,100 lines.

R.9.1 (next commit) will split the `gain_type` branch in
`_stage_gain_compute` into `ResFilterEnr` (production default,
all `over_sub` logic) and `ResFilterWiener` (minimal Wiener stub)
subclasses inside this same file.

Depends on numpy + several `modules.*` siblings; imports declared
explicitly at import time, plus a deferred import of
``ResidualEchoEstimator`` from this package to avoid the
``aec.py`` re-export chain.
"""
import numpy as np

from .residual_estimator import ResidualEchoEstimator
from .erle import FilterErleEstimator, FullbandErleEstimator


class ResFilter:
    """
    Residual Echo Suppressor (Post-Filter)

    Uses EER-based spectral suppression with OLA + sqrt-Hann windowing
    to avoid frame-boundary artifacts and musical noise.
    """

    # Backward-compat: _using_render_based / _render_based_hold now live on self._residual_est
    @property
    def _using_render_based(self) -> bool:
        return self._residual_est.using_render_based

    @_using_render_based.setter
    def _using_render_based(self, val: bool) -> None:
        # Legacy code path (e.g. AEC.process EPC render-forced) sets this directly;
        # forward to estimator state to keep one source of truth.
        self._residual_est._using_render_based = bool(val)

    def __init__(self, block_size: int, n_freqs: int, g_min_db: float = -20.0,
                 over_sub: float = 1.5, alpha: float = 0.8,
                 enable_cng: bool = False,
                 max_drop_db_per_frame: float = 6.0,
                 max_rise_db_per_frame: float = 3.0,
                 enable_spectral_floor: bool = True,
                 spectral_floor_db: float = -25.0,
                 ne_protect_db: float = -10.0,
                 frame_size: int = 0,
                 hop_size: int = 0,
                 echo_method: str = "coherence",
                 gain_type: str = "spectral_sub",
                 enable_reverb: bool = False,
                 reverb_decay: float = 0.5,
                 reverb_gain: float = 1.0,
                 alpha_echo_psd: float = 0.7,
                 alpha_error_psd: float = 0.8,
                 enr_scale: float = 1.0,
                 startup_dt_min_ne_scale: float = 1.0,
                 startup_dt_gain_floor: float = 1.0,
                 startup_dt_noise_floor_scale: float = 1.0,
                 sample_rate: int = 16000,
                 capture_stages: bool = False,
                 plan_a_kernel_tight: bool = True,
                 plan_a_hf_cap_2k: bool = True,
                 plan_a_stat_mask_7k: bool = True,
                 hf_cap_conditional: bool = False,
                 hf_cap_metric_threshold: float = 0.30,
                 plan_b_dt_per_bin_gamma: bool = False,
                 use_mic_excess_evidence: bool = False,
                 consume_filter_state: bool = False,
                 unified_gain_floor: bool = False,
                 dt_per_bin_unified: bool = False,
                 noise_floor_refined: bool = False,
                 cap2_fs_loosen: bool = False,
                 per_band_enr: bool = False,
                 enr_t_ne_per_band: tuple = (1.5, 1.5, 1.5),
                 enr_s_ne_per_band: tuple = (2.5, 2.5, 2.5),
                 dt_ne_compression_fix: bool = False,
                 dt_ne_state_scale: dict = None,
                 dt_ne_per_bin_thresh: float = 0.5,
                 dt_ne_per_bin_scale: float = 2.0,
                 subband_ne_detect_enabled: bool = False,
                 subband_ne_sub1_low: int = 192,
                 subband_ne_sub1_high: int = 320,
                 subband_ne_sub2_low: int = 32,
                 subband_ne_sub2_high: int = 128,
                 subband_ne_threshold: float = 0.5,
                 subband_ne_snr_threshold: float = 30.0,
                 res_mask_profile_swap_enabled: bool = False,
                 res_mask_last_lf_band: int = 20,
                 res_mask_first_hf_band: int = 32,
                 res_mask_normal_lf: tuple = (0.3, 0.4, 0.3),
                 res_mask_normal_hf: tuple = (0.07, 0.1, 0.3),
                 res_mask_nearend_lf: tuple = (1.09, 1.1, 0.3),
                 res_mask_nearend_hf: tuple = (0.1, 0.3, 0.3),
                 res_mask_ne_gate_dt: float = 0.3,
                 res_mask_swap_mode: str = 'binary',
                 res_mask_fs_overlay_coh2_min: float = 0.85,
                 res_mask_fs_overlay_dt_max: float = 0.2,
                 dominant_ne_detect_enabled: bool = False,
                 dominant_ne_lf_low: int = 4,
                 dominant_ne_lf_high: int = 60,
                 dominant_ne_enr_threshold: float = 0.25,
                 dominant_ne_enr_exit_threshold: float = 10.0,
                 dominant_ne_snr_threshold: float = 30.0,
                 dominant_ne_trigger_threshold: int = 12,
                 dominant_ne_hold_duration: int = 50,
                 c_e_branch_dt_per_bin_use_fq_usable: bool = False,
                 c_e_branch_coh2_ema_use_fq_usable: bool = False):
        self._plan_a_kernel_tight = plan_a_kernel_tight
        self._plan_b_dt_per_bin_gamma = plan_b_dt_per_bin_gamma
        self._plan_a_hf_cap_2k = plan_a_hf_cap_2k
        self._plan_a_stat_mask_7k = plan_a_stat_mask_7k
        self._hf_cap_conditional = hf_cap_conditional
        self._hf_cap_metric_threshold = hf_cap_metric_threshold
        self._use_mic_excess_evidence = use_mic_excess_evidence
        self._consume_filter_state = consume_filter_state
        self._unified_gain_floor = unified_gain_floor
        self._dt_per_bin_unified = dt_per_bin_unified
        # v3.19 Phase 1 — per-RES-branch C.E migration flags. G1 + P1
        # live in ResFilter; R1 lives in ResidualEchoEstimator (wired
        # separately via AEC.__init__).
        self._c_e_branch_dt_per_bin_use_fq_usable = c_e_branch_dt_per_bin_use_fq_usable
        self._c_e_branch_coh2_ema_use_fq_usable = c_e_branch_coh2_ema_use_fq_usable
        self._noise_floor_refined = noise_floor_refined
        self._cap2_fs_loosen = cap2_fs_loosen
        self._per_band_enr = per_band_enr
        self._enr_t_ne_per_band_cfg = tuple(enr_t_ne_per_band)
        self._enr_s_ne_per_band_cfg = tuple(enr_s_ne_per_band)
        # v3.15 §1.2 — DT-NE compression fix flags (default OFF byte-equal).
        self._dt_ne_compression_fix = dt_ne_compression_fix
        self._dt_ne_state_scale = dt_ne_state_scale if dt_ne_state_scale is not None else {
            'idle':            2.0,
            'startup':         2.0,
            'coarse_learning': 2.0,
            'refined_usable':  1.0,
            'diverged':        2.0,
            'suspicious_dt':   1.0,
        }
        self._dt_ne_per_bin_thresh = float(dt_ne_per_bin_thresh)
        self._dt_ne_per_bin_scale = float(dt_ne_per_bin_scale)
        # v3.18 Phase D.1 — Subband NE detector substrate
        self._subband_ne_detect_enabled = bool(subband_ne_detect_enabled)
        self._subband_ne_sub1_low = int(subband_ne_sub1_low)
        self._subband_ne_sub1_high = int(subband_ne_sub1_high)
        self._subband_ne_sub2_low = int(subband_ne_sub2_low)
        self._subband_ne_sub2_high = int(subband_ne_sub2_high)
        self._subband_ne_threshold = float(subband_ne_threshold)
        self._subband_ne_snr_threshold = float(subband_ne_snr_threshold)
        self._subband_ne_state = False
        # v3.18 Phase D.2 — Mask profile substrate
        self._mask_profile_swap_enabled = bool(res_mask_profile_swap_enabled)
        self._mask_last_lf_band = int(res_mask_last_lf_band)
        self._mask_first_hf_band = int(res_mask_first_hf_band)
        self._mask_anchors = {
            'normal_lf':  tuple(res_mask_normal_lf),
            'normal_hf':  tuple(res_mask_normal_hf),
            'nearend_lf': tuple(res_mask_nearend_lf),
            'nearend_hf': tuple(res_mask_nearend_hf),
        }
        # `_normal_mask_profile` / `_nearend_mask_profile` built lazily in
        # `_build_mask_profiles()`. Each holds 3 per-bin arrays:
        #   (enr_transparent[k], enr_suppress[k], emr_transparent[k])
        # Built once after `self.n_freqs` is known; cached as np.ndarray.
        self._normal_mask_profile = None
        self._nearend_mask_profile = None
        self._mask_ne_gate_dt = float(res_mask_ne_gate_dt)
        self._mask_swap_mode = str(res_mask_swap_mode)
        self._mask_fs_overlay_coh2_min = float(res_mask_fs_overlay_coh2_min)
        self._mask_fs_overlay_dt_max = float(res_mask_fs_overlay_dt_max)
        self._diag_mask_profile_nearend = False  # last-frame audit cache
        self._diag_mask_fs_overlay_fraction = 0.0  # last-frame audit: % of bins overlaid
        # v3.18 Phase B1 — Dominant NE detector (AEC3 default detector port).
        self._dominant_ne_detect_enabled = bool(dominant_ne_detect_enabled)
        self._dominant_ne_lf_low = int(dominant_ne_lf_low)
        self._dominant_ne_lf_high = int(dominant_ne_lf_high)
        self._dominant_ne_enr_threshold = float(dominant_ne_enr_threshold)
        self._dominant_ne_enr_exit_threshold = float(dominant_ne_enr_exit_threshold)
        self._dominant_ne_snr_threshold = float(dominant_ne_snr_threshold)
        self._dominant_ne_trigger_threshold = int(dominant_ne_trigger_threshold)
        self._dominant_ne_hold_duration = int(dominant_ne_hold_duration)
        self._dominant_ne_state = False
        self._dominant_ne_trigger_counter = 0
        self._dominant_ne_hold_counter = 0
        # Combined NE state (subband OR dominant). Updated each frame.
        self._ne_combined_state = False
        self.block_size = block_size          # FFT size (power of 2)
        self.sample_rate = sample_rate        # Hz, used for freq → bin conversion
        self.frame_size = frame_size if frame_size > 0 else block_size  # WOLA frame
        self.hop_size = hop_size if hop_size > 0 else self.frame_size // 2
        self.n_freqs = n_freqs
        self.g_min = 10 ** (g_min_db / 20)
        self.over_sub = over_sub
        self.alpha = alpha
        self.ne_protect_db = ne_protect_db
        self.alpha_echo_psd = alpha_echo_psd
        self.alpha_error_psd = alpha_error_psd
        self.enr_scale = enr_scale           # ENR threshold scale (1.0=AEC3 defaults)
        self.startup_dt_min_ne_scale = startup_dt_min_ne_scale
        self.startup_dt_gain_floor = startup_dt_gain_floor
        self.startup_dt_noise_floor_scale = startup_dt_noise_floor_scale

        # C1-C4: precomputed per-bin constants (invariant after init)
        freq_res = sample_rate / block_size
        f_bins = np.arange(n_freqs, dtype=np.float32) * freq_res
        # C1: stationary DT mask. v3.8.4: upper edge extended 4 kHz → 7 kHz
        # so fricatives / sibilants (4–7 kHz) get DT protection during
        # stationary DT (was: silently zero above 4 kHz).
        self._stat_dt_mask = np.zeros(n_freqs, dtype=np.float32)
        self._stat_dt_mask[(f_bins >= 300) & (f_bins <= 3000)] = 0.8
        low = (f_bins > 100) & (f_bins < 300)
        self._stat_dt_mask[low] = 0.8 * ((f_bins[low] - 100.0) / 200.0)
        # P1.0 toggle: plan_a_stat_mask_7k=False reverts to v3.8.3 behaviour
        # (no high-band coverage; mask was silently zero above 3 kHz).
        if self._plan_a_stat_mask_7k:
            high = (f_bins > 3000) & (f_bins < 7000)
            self._stat_dt_mask[high] = 0.8 * np.maximum(
                0.0, 1.0 - (f_bins[high] - 3000.0) / 4000.0)
        # C2: ENR blend array
        self._enr_blend = np.clip((np.arange(n_freqs, dtype=np.float32) - 5) / 5, 0, 1)
        # v3.14 Arc-R Sprint S1: per-band ENR threshold per-bin arrays.
        # Pre-built once from the LF/MF/HF tuple values using the same band
        # boundaries (1 kHz / 4 kHz) as P.S3 adaptive per-band ERL EMA so
        # the two arcs operate on consistent frequency partitions. When
        # `_per_band_enr=False` (default) these arrays are unused → no
        # behavioural change → byte-equal to baseline.
        _b1k = max(1, min(int(round(1000.0 / freq_res)), n_freqs - 2))
        _b4k = max(_b1k + 1, min(int(round(4000.0 / freq_res)), n_freqs - 1))
        self._enr_per_band_b1k = _b1k
        self._enr_per_band_b4k = _b4k
        _t_lf, _t_mf, _t_hf = self._enr_t_ne_per_band_cfg
        _s_lf, _s_mf, _s_hf = self._enr_s_ne_per_band_cfg
        self._enr_t_ne_pb = np.empty(n_freqs, dtype=np.float32)
        self._enr_t_ne_pb[:_b1k] = float(_t_lf)
        self._enr_t_ne_pb[_b1k:_b4k] = float(_t_mf)
        self._enr_t_ne_pb[_b4k:] = float(_t_hf)
        self._enr_s_ne_pb = np.empty(n_freqs, dtype=np.float32)
        self._enr_s_ne_pb[:_b1k] = float(_s_lf)
        self._enr_s_ne_pb[_b1k:_b4k] = float(_s_mf)
        self._enr_s_ne_pb[_b4k:] = float(_s_hf)
        # C3: frequency bin indices
        self._hf_cap_bin = min(int(500.0 / freq_res), n_freqs - 1)
        # v3.8.4: secondary cap anchor at 2 kHz so vowel formants (1–3 kHz)
        # are not dragged down by the 500 Hz bin's gain.
        self._hf_cap_bin_2k = min(int(2000.0 / freq_res), n_freqs - 1)
        # C4: harmonic distortion bin bounds
        self._harm_lf_start = max(1, int(100.0 / freq_res))
        self._harm_lf_end = min(int(500.0 / freq_res), n_freqs - 1)
        self._harm_hf_start = int(1000.0 / freq_res)
        self._harm_hf_end = min(int(4000.0 / freq_res), n_freqs - 1)
        # Round 4: voice-band mask (300–3000 Hz) for per-bin RES diagnostics + R2.
        self._voice_band_mask = ((f_bins >= 300.0) & (f_bins <= 3000.0))
        self._voice_band_idx = np.where(self._voice_band_mask)[0]
        # Round 4: per-bin trace caches (audio-passive)
        self._diag_coh2_last = np.zeros(n_freqs, dtype=np.float32)
        self._diag_nearend_est_last = np.zeros(n_freqs, dtype=np.float32)
        self._diag_residual_echo_psd_last = np.zeros(n_freqs, dtype=np.float32)
        self._diag_round4 = {}
        # Round 5: per-stage gain means (voice-band), audio-passive.
        # Indices: 0=softgate_emr, 1=spectral_floor, 2=epc_dt_cap_legacy,
        #          3=quiet_mask, 4=3bin_smooth, 5=hf_cap, 6=pre_temporal,
        #          7=post_temporal, 8=after_noise_lift (final gain_smooth)
        # Slot 2 is preserved post v3.16 C1 epc_dt_cap removal as an alias
        # of slot 1 (cap action removed because fire-rate was 0/800 in
        # v3.13 + v3.14 audits). 9-slot layout retained for diag-consumer
        # backward compat (P52 Phase B refactor expects fixed shape).
        self._diag_round5_stages = np.zeros(9, dtype=np.float32)

        # Per-bin gain capture (full vectors, opt-in via capture_stages)
        self._capture_stages = capture_stages
        self._stage_gains = {}
        # v3.8.4: cached per-bin DT confidence from compute stage, read by
        # postprocess HF-cap "high-band NE present?" gate.
        self._dt_per_bin_last = np.zeros(n_freqs, dtype=np.float32)

        # P4B diag (set by gain_compute / gain_postprocess each frame).
        self._p4b_dt_per_bin_mean = 0.0
        self._p4b_dt_per_bin_hf_mean = 0.0
        self._p4b_coh2_hf_mean = 0.0
        self._p4b_effective_dt = 0.0
        self._p4b_is_stationary_dt = 0
        self._p4b_gain_hf_mean = 1.0
        self._p4b_res_echo_hf_mean_db = -120.0

        # RES v2: direct echo estimation + Wiener gain + reverb
        self.echo_method = echo_method       # "coherence" or "direct"
        self.gain_type = gain_type           # "spectral_sub" or "wiener"
        self.enable_reverb = enable_reverb
        self.reverb_decay = reverb_decay
        self.reverb_gain = reverb_gain

        # --- Immutable config (set once, not cleared by reset) ---
        self.alpha_coh = 0.65           # Cross-PSD smoothing (TC≈50ms)
        self.enable_cng = enable_cng
        self.alpha_noise = 0.98
        self.max_drop_ratio = 10 ** (max_drop_db_per_frame / 20)
        self.max_rise_ratio = 10 ** (max_rise_db_per_frame / 20)
        self.enable_spectral_floor = enable_spectral_floor
        self.spectral_floor_ratio = 10 ** (spectral_floor_db / 20)
        self.alpha_envelope = 0.95
        self.window = np.sqrt(np.hanning(self.frame_size)).astype(np.float32)

        # --- Runtime state arrays (allocated once, cleared by reset) ---
        self.gain_smooth = np.full(n_freqs, self.g_min, dtype=np.float32)
        self.echo_psd = np.zeros(n_freqs, dtype=np.float32)
        self.error_psd = np.zeros(n_freqs, dtype=np.float32)
        self.S_fe = np.zeros(n_freqs, dtype=np.complex64)
        self.S_ff = np.zeros(n_freqs, dtype=np.float32)
        self.S_ee = np.zeros(n_freqs, dtype=np.float32)
        self.near_psd = np.zeros(n_freqs, dtype=np.float32)
        self.near_psd_buf = np.zeros((4, n_freqs), dtype=np.float32)
        self.reverb_psd = np.zeros(n_freqs, dtype=np.float32)
        self.noise_psd = np.zeros(n_freqs, dtype=np.float32)
        self._coh2_smooth = np.zeros(n_freqs, dtype=np.float32)
        self.error_envelope = np.ones(n_freqs, dtype=np.float32)
        self.input_buf = np.zeros(self.frame_size, dtype=np.float32)
        self.ola_buf = np.zeros(self.frame_size, dtype=np.float32)

        # Multi-ERLE estimators
        self._filter_erle_est = FilterErleEstimator(n_freqs)
        self._fb_erle_est = FullbandErleEstimator()

        # Per-frame diagnostic stats (disabled by default; call enable_stats() to activate)
        self._stats = None
        # Durable audit counter substrate (Phase 3B v3 S7+, default-OFF zero-cost).
        # Use enable_audit_counters() to activate; get_audit_counters() to retrieve.
        self._audit_counters = None
        self._stats_last_enr = 0.0
        self._stats_last_nearend = 0.0
        self._stats_last_res_psd = 0.0
        self._stats_last_min_ne = 0.0
        # _stats_last_using_render wired from self._residual_est.using_render_based
        # in get_stats accumulator. Removed in v3.8.x (never-wired after residual
        # estimator refactor): render_echo, linear_res, want_render, should_render_v1.
        self._stats_last_using_render = False
        self._stats_last_ne_g_floor = 0.0
        self._stats_last_spectral_g_min = 0.0
        # v3.16 C1c: pre-max captures so audits read pre-floor surface.
        self._stats_pre_max_spectral_g_min = 0.0
        self._stats_pre_max_spectral_g_min_max = 0.0
        self._stats_ne_g_floor_max = 0.0
        self._stats_ne_g_floor_any_bin_fired = False
        self._stats_last_gain_before_floor = 1.0
        self._stats_last_gain_after_floor = 1.0
        self._stats_last_gain_after_smoothing = 1.0
        self._stats_last_noise_floor_gain = 0.0
        self._stats_last_noise_psd = 0.0
        self._stats_last_spec_pwr = 0.0
        self._stats_last_nfl_lifted = False
        self._last_effective_dt = 0.0   # exported for external startup_dt detection

        # Residual echo attribution (stage 1 + stage 2 of original inline logic).
        # Default mode='legacy' → bit-exact parity. Phase B2 ablation flips to 'split'.
        self._residual_est = ResidualEchoEstimator(n_freqs, mode='legacy')

        # v3.18 Phase D.2 — build per-bin mask profiles once after n_freqs known.
        self._build_mask_profiles()

        # E1: call reset() to initialize all runtime scalar state
        # from a single source of truth (avoids init/reset divergence)
        self.reset()

    def _build_mask_profiles(self):
        """Build per-bin `enr_transparent / enr_suppress / emr_transparent`
        tables for `normal` and `nearend` mask profiles.

        Linear LF→HF interpolation across `[last_lf_band .. first_hf_band]`
        mirroring AEC3 `GainParameters::SetConfig`
        (suppression_gain.cc:487). When flag-OFF the tables are still
        built (cheap, ~6 KB total) so D.3 can flip the consumer on at
        runtime without re-init.
        """
        n_freqs = self.n_freqs
        last_lf = max(0, min(self._mask_last_lf_band, n_freqs - 2))
        first_hf = max(last_lf + 1, min(self._mask_first_hf_band, n_freqs - 1))

        # Per-bin interpolation weight: 0 in LF, 1 in HF, linear in between.
        a = np.empty(n_freqs, dtype=np.float32)
        a[:last_lf + 1] = 0.0
        a[first_hf:] = 1.0
        if first_hf > last_lf + 1:
            tr = first_hf - last_lf  # transition span
            for k in range(last_lf + 1, first_hf):
                a[k] = float(k - last_lf) / float(tr)

        def _profile(lf_tuple, hf_tuple):
            lf_t, lf_s, lf_emr = lf_tuple
            hf_t, hf_s, hf_emr = hf_tuple
            enr_t = (1 - a) * float(lf_t) + a * float(hf_t)
            enr_s = (1 - a) * float(lf_s) + a * float(hf_s)
            emr_t = (1 - a) * float(lf_emr) + a * float(hf_emr)
            return (enr_t.astype(np.float32),
                    enr_s.astype(np.float32),
                    emr_t.astype(np.float32))

        self._normal_mask_profile = _profile(
            self._mask_anchors['normal_lf'],
            self._mask_anchors['normal_hf'])
        self._nearend_mask_profile = _profile(
            self._mask_anchors['nearend_lf'],
            self._mask_anchors['nearend_hf'])

    def reset(self, preserve_long_window_ema: bool = False):
        """Reset all runtime state. Arrays are .fill(0), scalars are set.

        preserve_long_window_ema (v3.10.2): forwarded to ResidualEchoEstimator
        so the long-window far-PSD EMA survives recovery resets that are
        triggered by bad-filter-state plateaus / delay_first acquisition —
        the EMA is input-side and should not be discarded.
        """
        self.gain_smooth.fill(self.g_min)
        self.echo_psd.fill(0)
        self.error_psd.fill(0)
        self.S_fe.fill(0)
        self.S_ff.fill(0)
        self.S_ee.fill(0)
        self.near_psd.fill(0)
        self.near_psd_buf.fill(0)
        self.near_psd_idx = 0
        self.reverb_psd.fill(0)
        self.noise_psd.fill(0)
        self._noise_initialized = False
        self._coh2_smooth.fill(0)
        self.error_envelope.fill(1.0)
        self.input_buf.fill(0)
        self.ola_buf.fill(0)
        self.far_activity = 0.0
        self._nonlinear_frames = 0
        # _render_based_hold + _using_render_based moved to self._residual_est
        if hasattr(self, '_residual_est'):
            self._residual_est.reset(preserve_long_window_ema=preserve_long_window_ema)
        self._diag_gain_mean = 1.0
        self._diag_gain_min = 1.0
        self._diag_effective_g_min = 1.0
        self._diag_far_activity = 0.0
        self._diag_echo_psd_mean = 0.0
        self._diag_error_psd_mean = 0.0
        self._filter_erle_est.reset()
        self._fb_erle_est.reset()
        # v3.18 Phase D.1 — Subband NE detector state
        self._subband_ne_state = False
        # v3.18 Phase B1 — Dominant NE detector state + hysteresis counters
        self._dominant_ne_state = False
        self._dominant_ne_trigger_counter = 0
        self._dominant_ne_hold_counter = 0
        self._ne_combined_state = False

    def enable_stats(self):
        """Enable per-frame DT statistics collection (zero cost when disabled)."""
        self._stats = {
            # existing fields
            'total_frames': 0,
            'dt_active': 0, 'epc_dt': 0,
            'low_erle_dt': 0, 'startup_dt': 0, 'any_special_dt': 0,
            'dt_erle_sum': 0.0, 'dt_gain_sum': 0.0,
            'dt_enr_sum': 0.0, 'dt_res_sum': 0.0, 'dt_ne_sum': 0.0,
            # (2) filter convergence/divergence
            'filter_once_converged_dt': 0,
            'filter_diverged_dt': 0,
            'shadow_better_dt': 0,
            'e2_main_y2_sum': 0.0, 'e2_shadow_y2_sum': 0.0, 'e2_shadow_e2_main_sum': 0.0,
            # (3) usability candidates
            'usable_v1': 0, 'usable_v2': 0, 'usable_v3': 0, 'usable_v4': 0,
            'unusable_dt': 0,
            # (4) residual echo model
            'using_render_based_dt': 0,
            'min_ne_sum': 0.0,
            # (5) startup_dt gain stage diagnostics (accumulated for not filter_converged frames)
            'startup_dt_once_conv': 0,   # DT frames where not filter_once_converged
            'st_ne_g_floor_sum': 0.0,
            'st_spectral_g_min_sum': 0.0,
            'st_gain_before_floor_sum': 0.0,
            'st_gain_after_floor_sum': 0.0,
            'st_gain_after_smoothing_sum': 0.0,
            # (6) noise_floor_gain diagnostics (accumulated for not filter_once_converged frames)
            'st_nfl_noise_floor_gain_sum': 0.0,
            'st_nfl_noise_psd_sum': 0.0,
            'st_nfl_spec_pwr_sum': 0.0,
            'st_nfl_lifted_count': 0,
            'st_nfl_final_gain_sum': 0.0,
        }
        self._stats_last_min_ne = 0.0
        self._stats_last_using_render = False
        self._stats_last_ne_g_floor = 0.0
        self._stats_last_spectral_g_min = 0.0
        # v3.16 C1c: pre-max captures so audits read pre-floor surface.
        self._stats_pre_max_spectral_g_min = 0.0
        self._stats_pre_max_spectral_g_min_max = 0.0
        self._stats_ne_g_floor_max = 0.0
        self._stats_ne_g_floor_any_bin_fired = False
        self._stats_last_gain_before_floor = 1.0
        self._stats_last_gain_after_floor = 1.0
        self._stats_last_gain_after_smoothing = 1.0
        self._stats_last_noise_floor_gain = 0.0
        self._stats_last_noise_psd = 0.0
        self._stats_last_spec_pwr = 0.0
        self._stats_last_nfl_lifted = False

    def enable_audit_counters(self):
        """Enable durable RES audit counter substrate (Phase 3B v3 S7+).

        Counters accumulate across frames within a stream; consumers (eval
        harness) read via get_audit_counters() at end-of-stream. Zero cost
        when not enabled.

        S7 (Option α) target: measure mean reduction of `dt_per_bin` legacy
        vs unified-hypothetical across FS bins (coh² < 0.1) in the
        `filter_converged AND _lw_ready AND epc_active` slice. See design
        [docs/v3_12_phase3b_v3_design.md §6.1](../docs/v3_12_phase3b_v3_design.md).
        """
        self._audit_counters = {
            'total_frames': 0,
            # S7 Option α — dt_per_bin path classification
            's7_legacy_path_frames': 0,
            's7_f31v3_path_frames': 0,
            's7_planb_path_frames': 0,
            # Legacy-path sub-reasons (sum may exceed s7_legacy_path_frames
            # because a frame can hit multiple reasons, e.g. not_converged AND
            # not_lw_ready)
            's7_legacy_epc_active_frames': 0,
            's7_legacy_not_converged_frames': 0,
            's7_legacy_not_lw_ready_frames': 0,
            's7_legacy_target_other_frames': 0,
            # Primary S7 target slice: filter_converged AND _lw_ready AND epc_active
            's7_target_fs_bin_count': 0,
            's7_target_fs_bin_legacy_sum': 0.0,
            's7_target_fs_bin_unified_sum': 0.0,
            # Cross-validation slice: filter_converged AND _lw_ready AND NOT epc_active
            # (legacy path only reachable when use_mic_excess_evidence=False)
            's7_alt_fs_bin_count': 0,
            's7_alt_fs_bin_legacy_sum': 0.0,
            's7_alt_fs_bin_unified_sum': 0.0,
            # S8 (Phase 3B v4) — downstream-clamp audit. Each counter
            # aggregates over FS bins (coh² < 0.1) across all frames.
            # Stage 1 4-cap chain (line ~2138/2144/2151/2163): per-cap
            # binding counts (how often the cap clamped a FS bin) +
            # reduction sums (sum of log10(pre/post) on those bins).
            's8_stage1_fs_bin_total': 0,        # total FS bin opportunities
            's8_cap1_echo_x2_binding': 0,
            's8_cap1_echo_x2_reduction_sum': 0.0,
            's8_cap2_err_mult_binding': 0,
            's8_cap2_err_mult_reduction_sum': 0.0,
            's8_cap3_dt_suppress_binding': 0,
            's8_cap3_dt_suppress_reduction_sum': 0.0,
            's8_cap4_render_ceil_binding': 0,
            's8_cap4_render_ceil_reduction_sum': 0.0,
            # Nearend_est floor 4-way binding (line ~2363/2372/2377):
            # of the four candidate floor sources, which is the MAX
            # (the actual binding) for each FS bin.
            's8_nef_raw_count': 0,         # raw_nearend_est * dt_shaped wins
            's8_nef_noise_floor_count': 0, # noise_floor_psd wins
            's8_nef_min_ne_count': 0,      # min_ne_from_dt wins
            's8_nef_ne_physical_count': 0, # ne_physical_floor wins
            # S9 Phase 3B v5 pre-implementation audit — noise_floor_psd
            # refinement. For FS bins (coh² < 0.1) where the current
            # baseline winner is noise_floor_psd (s8_nef_noise_floor_count
            # ≈ 43% global), compute hypothetical winners under two
            # candidate refinements:
            #   A.1: scalar × 0.001 (10× lower than baseline 0.01)
            #   A.2: per-bin error_psd × 0.005 (scalar→per-bin)
            # `release_to_raw`: bin previously bound by noise_floor now
            #   bound by raw_nearend_est (good — FS-honest evidence wins).
            # `shift_to_min_ne`: bin shifts to min_ne_from_dt (neutral —
            #   carrier shifts from #1 to #2, doesn't reduce overall
            #   suppression in FS).
            # `stays_floor`: still bound by noise_floor (new value still
            #   highest).
            # `reduction_db_sum`: 10·log10(old_nearend_est /
            #   new_nearend_est) summed over bins where any candidate
            #   wins changed — magnitude of the nearend_est drop.
            's9_a1_release_to_raw': 0,
            's9_a1_shift_to_min_ne': 0,
            's9_a1_shift_to_phys': 0,
            's9_a1_stays_floor': 0,
            's9_a1_reduction_db_sum': 0.0,
            's9_a1_reduction_count': 0,
            's9_a2_release_to_raw': 0,
            's9_a2_shift_to_min_ne': 0,
            's9_a2_shift_to_phys': 0,
            's9_a2_stays_floor': 0,
            's9_a2_reduction_db_sum': 0.0,
            's9_a2_reduction_count': 0,
            # Out-of-FS sanity: bins currently NOT bound by noise_floor
            # but where A.1 / A.2 would dethrone the current winner
            # (only possible if candidate noise_floor RISES above winner
            # — A.2 in high-error_psd bins can do this).
            's9_a1_intrudes_outside_floor_baseline': 0,
            's9_a2_intrudes_outside_floor_baseline': 0,
            # S9-C pre-audit — joint noise_floor + min_ne_from_dt attack.
            # S9-A finding: A.2 only releases 11.5% of FS floor bins to
            # raw_NE; 88% shifts to min_ne_from_dt (F3.1 v3 floor =
            # error_psd * dt_shaped). Joint candidates attack both
            # floors simultaneously in FS bins:
            #   C.1: noise_floor → error_psd × 0.005 (=A.2)
            #         + min_ne_from_dt × 0.1 (10× reduction)
            #   C.2: noise_floor → error_psd × 0.005 (=A.2)
            #         + min_ne_from_dt → 0   (eliminated in FS)
            # `release_to_raw`: FS bin previously bound by ANY floor
            #   (noise_floor / min_ne_from_dt / ne_physical) now bound
            #   by raw_nearend_est. This is the real outcome we want —
            #   FS bins should not be floored.
            # `still_floor` / `still_min_ne` / `still_phys`: residual
            #   floor binding under candidate (shows which floors still
            #   block release).
            's9c_c1_release_to_raw': 0,
            's9c_c1_still_floor': 0,
            's9c_c1_still_min_ne': 0,
            's9c_c1_still_phys': 0,
            's9c_c1_reduction_db_sum': 0.0,
            's9c_c1_reduction_count': 0,
            's9c_c2_release_to_raw': 0,
            's9c_c2_still_floor': 0,
            's9c_c2_still_min_ne': 0,
            's9c_c2_still_phys': 0,
            's9c_c2_reduction_db_sum': 0.0,
            's9c_c2_reduction_count': 0,
            # FS bins baseline-bound by ANY floor (denominator for
            # release_pct on the C candidates).
            's9c_baseline_any_floor_count': 0,
            # S9-D sanity — attack ALL three nearend_est floors:
            #   noise_floor → error_psd × 0.005 (A.2)
            #   min_ne_from_dt → 0
            #   ne_physical_floor → 0
            # Stack reduces to [raw_NE * dt_shaped, noise_floor_A2, 0, 0].
            # Winner can only be 0 (raw) or 1 (tiny noise_floor at
            # 0.005×error_psd). If raw_NE * dt_shaped >
            # 0.005×error_psd → release_to_raw; else still_floor.
            # Goal: prove that the nearend_est stack fully controls FS
            # release rate (FS release ≥90% confirms no hidden 4th
            # carrier). Read-only; informs Phase 4 RES canonical
            # refactor design lock.
            's9d_release_to_raw': 0,
            's9d_still_floor': 0,
            's9d_reduction_db_sum': 0.0,
            's9d_reduction_count': 0,
        }

    def get_audit_counters(self):
        """Return audit counter dict (or None if not enabled)."""
        return self._audit_counters

    def get_stage_gains(self):
        """Return dict of per-bin gain vectors captured this frame.

        Empty dict unless `capture_stages=True` was passed to __init__.
        Keys: '01_softgate_emr', '02_spectral_floor', '03_epc_dt_cap',
        '04_quiet_mask', '05_3bin_smooth', '06_hf_cap', '07_pre_temporal',
        '08_post_temporal'. Vectors are np.float32 length n_freqs.
        Note: '03_epc_dt_cap' is post v3.16 C1 an alias of
        '02_spectral_floor' (cap action removed; key retained for audit
        script backward compat).
        """
        return self._stage_gains

    def get_stats(self):
        """Return aggregated DT stats dict, or None if not enabled / no DT frames."""
        s = self._stats
        if s is None or s['dt_active'] == 0:
            return None
        n, t = s['dt_active'], s['total_frames']
        return {
            'total_frames': t, 'dt_active': n, 'dt_pct': n / t,
            'epc_dt': s['epc_dt'], 'epc_dt_pct': s['epc_dt'] / n,
            'low_erle_dt': s['low_erle_dt'], 'low_erle_dt_pct': s['low_erle_dt'] / n,
            'startup_dt': s['startup_dt'], 'startup_dt_pct': s['startup_dt'] / n,
            'any_special_dt': s['any_special_dt'], 'any_special_dt_pct': s['any_special_dt'] / n,
            'mean_erle_factor': s['dt_erle_sum'] / n,
            'mean_gain': s['dt_gain_sum'] / n,
            'mean_enr': s['dt_enr_sum'] / n,
            'mean_residual_echo_psd': s['dt_res_sum'] / n,
            'mean_nearend_est': s['dt_ne_sum'] / n,
            # (2) filter convergence/divergence
            'filter_once_converged_pct': s['filter_once_converged_dt'] / n,
            'filter_diverged_pct': s['filter_diverged_dt'] / n,
            'shadow_better_pct': s['shadow_better_dt'] / n,
            'mean_e2_main_y2': s['e2_main_y2_sum'] / n,
            'mean_e2_shadow_y2': s['e2_shadow_y2_sum'] / n,
            'mean_e2_shadow_e2_main': s['e2_shadow_e2_main_sum'] / n,
            # (3) usability
            'usable_v1_pct': s['usable_v1'] / n,
            'usable_v2_pct': s['usable_v2'] / n,
            'usable_v3_pct': s['usable_v3'] / n,
            'usable_v4_pct': s['usable_v4'] / n,
            'unusable_dt_pct': s['unusable_dt'] / n,
            # (4) residual echo model
            'using_render_based_pct': s['using_render_based_dt'] / n,
            'mean_min_ne_from_dt': s['min_ne_sum'] / n,
            # (5) startup_dt gain stage diagnostics
            'startup_dt_once_conv_pct': s['startup_dt_once_conv'] / n,
            'mean_ne_g_floor': s['st_ne_g_floor_sum'] / s['startup_dt'] if s['startup_dt'] > 0 else 0.0,
            'mean_spectral_g_min': s['st_spectral_g_min_sum'] / s['startup_dt'] if s['startup_dt'] > 0 else 0.0,
            'mean_gain_before_floor': s['st_gain_before_floor_sum'] / s['startup_dt'] if s['startup_dt'] > 0 else 0.0,
            'mean_gain_after_floor': s['st_gain_after_floor_sum'] / s['startup_dt'] if s['startup_dt'] > 0 else 0.0,
            'mean_gain_after_smoothing': s['st_gain_after_smoothing_sum'] / s['startup_dt'] if s['startup_dt'] > 0 else 0.0,
            # (6) noise_floor_gain diagnostics (denominator: startup_dt_once_conv frames)
            'mean_noise_floor_gain': s['st_nfl_noise_floor_gain_sum'] / s['startup_dt_once_conv'] if s['startup_dt_once_conv'] > 0 else 0.0,
            'mean_noise_psd': s['st_nfl_noise_psd_sum'] / s['startup_dt_once_conv'] if s['startup_dt_once_conv'] > 0 else 0.0,
            'mean_spec_pwr': s['st_nfl_spec_pwr_sum'] / s['startup_dt_once_conv'] if s['startup_dt_once_conv'] > 0 else 0.0,
            'nfl_lifted_pct': s['st_nfl_lifted_count'] / s['startup_dt_once_conv'] if s['startup_dt_once_conv'] > 0 else 0.0,
            'mean_final_gain_nfl': s['st_nfl_final_gain_sum'] / s['startup_dt_once_conv'] if s['startup_dt_once_conv'] > 0 else 0.0,
        }

    # ── Stage methods (Round 6 refactor) ─────────────────────────────────
    # Each _stage_* method handles a logical block of the suppressor pipeline.
    # All share self.* state; arguments are local-only values that flow
    # between stages. Behavior identical to the pre-refactor inline code.

    def _stage_residual_model(self, *, coh2, far_spec, far_power, erle_factor,
                                dt_for_fs, epc_active, saturation_level,
                                filter_converged, erl_estimate, near_spec,
                                is_stationary_dt, aec_state, echo_pwr_linear):
        """Stage 1: residual_echo_psd attribution + 4 caps + reverb tail + echo boost.

        Returns residual_echo_psd (np.ndarray or None). Mutates near_psd buffers,
        reverb_psd, _nonlinear_frames, and stats fields.
        """
        residual_echo_psd = None
        if self.echo_method == "direct" and near_spec is not None:
            near_pwr = np.abs(near_spec) ** 2
            self.near_psd_buf[self.near_psd_idx] = near_pwr
            self.near_psd_idx = (self.near_psd_idx + 1) % 4
            self.near_psd = np.mean(self.near_psd_buf, axis=0)

            # v3.18 Phase D.1 — Subband NE detector (audit-only when default-OFF).
            # Mirror of AEC3 `SubbandNearendDetector::Update`. Uses smoothed
            # near_psd (4-frame avg above) + prev-frame noise_psd (1-frame lag
            # acceptable for audit substrate). Default-OFF: state stays False.
            if self._subband_ne_detect_enabled:
                _s1l = self._subband_ne_sub1_low
                _s1h = min(self._subband_ne_sub1_high, self.n_freqs - 1)
                _s2l = self._subband_ne_sub2_low
                _s2h = min(self._subband_ne_sub2_high, self.n_freqs - 1)
                _noise_pow_sub1 = float(np.mean(self.noise_psd[_s1l:_s1h + 1])) + 1e-10
                _ne_pow_sub1 = float(np.mean(self.near_psd[_s1l:_s1h + 1]))
                _ne_pow_sub2 = float(np.mean(self.near_psd[_s2l:_s2h + 1])) + 1e-10
                self._subband_ne_state = bool(
                    _ne_pow_sub1 < self._subband_ne_threshold * _ne_pow_sub2
                    and _ne_pow_sub1 > self._subband_ne_snr_threshold * _noise_pow_sub1
                )

            # Stage 1+2 of residual echo attribution: delegated to ResidualEchoEstimator.
            residual_echo_psd = self._residual_est.attribute(
                echo_psd=self.echo_psd, error_psd=self.error_psd,
                coh2=coh2, far_spec=far_spec, far_power=far_power,
                erle_factor=erle_factor, dt_for_fs=dt_for_fs,
                far_activity=self.far_activity,
                epc_active=epc_active, saturation_level=saturation_level,
                filter_converged=filter_converged, erl_estimate=erl_estimate,
                filter_erle=self._filter_erle_est, fb_erle=self._fb_erle_est,
                aec_state=aec_state,
            )

            # v3.18 Phase B1 — Dominant NE detector (AEC3 default port).
            # Echo-aware (uses residual_echo_psd just computed) +
            # hysteresis (trigger_counter + hold_counter + fast-exit).
            # Direct port of dominant_nearend_detector.cc Update().
            # Default-OFF: state and counters frozen.
            if self._dominant_ne_detect_enabled:
                _lf_lo = self._dominant_ne_lf_low
                _lf_hi = min(self._dominant_ne_lf_high, self.n_freqs - 1)
                _ne_sum = float(np.sum(self.near_psd[_lf_lo:_lf_hi + 1]))
                _echo_sum = float(np.sum(residual_echo_psd[_lf_lo:_lf_hi + 1]))
                _noise_sum = float(np.sum(self.noise_psd[_lf_lo:_lf_hi + 1])) + 1e-10
                _trig_cond = (
                    _echo_sum < self._dominant_ne_enr_threshold * _ne_sum
                    and _ne_sum > self._dominant_ne_snr_threshold * _noise_sum
                )
                if _trig_cond:
                    self._dominant_ne_trigger_counter += 1
                    if self._dominant_ne_trigger_counter >= self._dominant_ne_trigger_threshold:
                        self._dominant_ne_hold_counter = self._dominant_ne_hold_duration
                        self._dominant_ne_trigger_counter = self._dominant_ne_trigger_threshold
                else:
                    self._dominant_ne_trigger_counter = max(
                        0, self._dominant_ne_trigger_counter - 1)
                # Fast-exit on strong echo
                if (_echo_sum > self._dominant_ne_enr_exit_threshold * _ne_sum
                        and _echo_sum > self._dominant_ne_snr_threshold * _noise_sum):
                    self._dominant_ne_hold_counter = 0
                self._dominant_ne_hold_counter = max(
                    0, self._dominant_ne_hold_counter - 1)
                self._dominant_ne_state = self._dominant_ne_hold_counter > 0
            # Combined NE state: OR-aggregate active detectors. When both
            # disabled, stays False (substrate inert).
            self._ne_combined_state = bool(
                self._subband_ne_state or self._dominant_ne_state)

            # Nonlinear echo mode: harmonics from speaker distortion
            if saturation_level > 0.3:
                self._nonlinear_frames += 1
            else:
                self._nonlinear_frames = max(0, self._nonlinear_frames - 1)
            is_nonlinear = self._nonlinear_frames > 5
            if is_nonlinear and far_power > 1e-4:
                nonlinear_boost = 1.0 + 1.0 * saturation_level
                residual_echo_psd = residual_echo_psd * nonlinear_boost

            # Harmonic distortion: HF floor from LF echo
            if saturation_level > 0.05 and far_power > 1e-4:
                lf_start, lf_end = self._harm_lf_start, self._harm_lf_end
                hf_start, hf_end = self._harm_hf_start, self._harm_hf_end
                if lf_end > lf_start and hf_end > hf_start:
                    lf_echo_mean = float(np.mean(residual_echo_psd[lf_start:lf_end]))
                    distortion_factor = 0.1 + 0.4 * saturation_level
                    harmonic_floor = lf_echo_mean * distortion_factor
                    residual_echo_psd[hf_start:hf_end] = np.maximum(
                        residual_echo_psd[hf_start:hf_end], harmonic_floor)

            # === DEEP-TRACE HOOKS: capture residual_echo_psd at each cap stage ===
            if self._stats is not None:
                self._stats_last_res_after_attribute = float(np.mean(residual_echo_psd))

            # S8 audit: FS bin total opportunity count (per-frame, increment once).
            # Read-only; FS-bin classification (coh² < 0.1) consistent with S7.
            if self._audit_counters is not None:
                _ac_s8 = self._audit_counters
                _fs_mask_s8 = coh2 < 0.1
                _ac_s8['s8_stage1_fs_bin_total'] += int(np.sum(_fs_mask_s8))

            # Cap 1: echo_psd × 2.0 (skipped in render-mode)
            if not self._residual_est.using_render_based:
                if self._audit_counters is not None:
                    _cap1_arr = self.echo_psd * 2.0
                    _bind = _fs_mask_s8 & (residual_echo_psd > _cap1_arr)
                    _n_bind = int(np.sum(_bind))
                    if _n_bind > 0:
                        _ac_s8['s8_cap1_echo_x2_binding'] += _n_bind
                        _ac_s8['s8_cap1_echo_x2_reduction_sum'] += 10.0 * float(
                            np.sum(np.log10((residual_echo_psd[_bind] + 1e-30)
                                            / (_cap1_arr[_bind] + 1e-30))))
                residual_echo_psd = np.minimum(residual_echo_psd, self.echo_psd * 2.0)
            if self._stats is not None:
                self._stats_last_res_after_echo_cap = float(np.mean(residual_echo_psd))

            # Cap 2: error_psd × (1.5 if render else 1.0)
            err_cap_mult = 1.5 if self._residual_est.using_render_based else 1.0
            if self._audit_counters is not None:
                _cap2_arr = self.error_psd * err_cap_mult
                _bind = _fs_mask_s8 & (residual_echo_psd > _cap2_arr)
                _n_bind = int(np.sum(_bind))
                if _n_bind > 0:
                    _ac_s8['s8_cap2_err_mult_binding'] += _n_bind
                    _ac_s8['s8_cap2_err_mult_reduction_sum'] += 10.0 * float(
                        np.sum(np.log10((residual_echo_psd[_bind] + 1e-30)
                                        / (_cap2_arr[_bind] + 1e-30))))
            if self._cap2_fs_loosen:
                # S11: skip Cap2 in FS-confident bins (coh² < 0.1).
                # Raises residual_echo_psd numerator in ENR → drops gain
                # → expected more FS suppression. Inverse mechanism vs
                # S6-S10 (which lowered ENR denominator).
                _cap2_val = self.error_psd * err_cap_mult
                _capped = np.minimum(residual_echo_psd, _cap2_val)
                residual_echo_psd = np.where(coh2 < 0.1, residual_echo_psd, _capped)
            else:
                residual_echo_psd = np.minimum(residual_echo_psd, self.error_psd * err_cap_mult)
            if self._stats is not None:
                self._stats_last_res_after_error_cap = float(np.mean(residual_echo_psd))

            # Cap 3: dt_suppress (skipped in render-mode)
            if not self._residual_est.using_render_based:
                dt_suppress = np.clip(1.0 - dt_for_fs**2, 0.1, 1.0)
                if self._audit_counters is not None:
                    _cap3_arr = self.error_psd * dt_suppress
                    _bind = _fs_mask_s8 & (residual_echo_psd > _cap3_arr)
                    _n_bind = int(np.sum(_bind))
                    if _n_bind > 0:
                        _ac_s8['s8_cap3_dt_suppress_binding'] += _n_bind
                        _ac_s8['s8_cap3_dt_suppress_reduction_sum'] += 10.0 * float(
                            np.sum(np.log10((residual_echo_psd[_bind] + 1e-30)
                                            / (_cap3_arr[_bind] + 1e-30))))
                residual_echo_psd = np.minimum(residual_echo_psd, self.error_psd * dt_suppress)
            if self._stats is not None:
                self._stats_last_res_after_dt_cap = float(np.mean(residual_echo_psd))

            # Cap 4: render_ceil (skipped in render-mode)
            # v3.14 Arc-P P.S2: when erl_estimate is a per-bin array
            # (flag=ON path), use its mean as a scalar for the Cap 4 ceiling
            # factor. The render_ceil is a conservative upper bound on the
            # residual echo PSD; using mean(per_band) is reasonable since this
            # cap is broadband. The per-bin ERL is consumed by the F3.1-v3
            # excess formula in _stage_gain_compute (the primary P.S2 target).
            _erl_scalar = (float(np.mean(erl_estimate))
                           if isinstance(erl_estimate, np.ndarray)
                           else float(erl_estimate))
            if far_spec is not None and far_power > 1e-4 and _erl_scalar > 0.0:
                far_psd_k = np.abs(far_spec) ** 2
                render_ceil = far_psd_k * min(_erl_scalar * 2.0, 1.0)
                if self._stats is not None:
                    self._stats_last_render_ceil_mean = float(np.mean(render_ceil))
                    self._stats_last_erl_estimate = _erl_scalar
                if not self._residual_est.using_render_based:
                    if self._audit_counters is not None:
                        _bind = _fs_mask_s8 & (residual_echo_psd > render_ceil)
                        _n_bind = int(np.sum(_bind))
                        if _n_bind > 0:
                            _ac_s8['s8_cap4_render_ceil_binding'] += _n_bind
                            _ac_s8['s8_cap4_render_ceil_reduction_sum'] += 10.0 * float(
                                np.sum(np.log10((residual_echo_psd[_bind] + 1e-30)
                                                / (render_ceil[_bind] + 1e-30))))
                    residual_echo_psd = np.minimum(residual_echo_psd, render_ceil)
            if self._stats is not None:
                self._stats_last_res_after_render_ceil = float(np.mean(residual_echo_psd))

            # Reverb tail (WebRTC-AEC3-style IIR on far_psd)
            if self.enable_reverb:
                far_psd = np.abs(far_spec) ** 2 if far_spec is not None else echo_pwr_linear
                self.reverb_psd = (self.reverb_decay * self.reverb_psd
                                   + (1 - self.reverb_decay) * far_psd)
                if not is_stationary_dt:
                    ne_reverb_factor = 0.7 + 0.3 * self.far_activity * (1.0 - dt_for_fs)
                    reverb_gate = self.far_activity * ne_reverb_factor
                    if self.far_activity < 0.1:
                        reverb_gate = 0.0
                    residual_echo_psd = (residual_echo_psd
                                         + self.reverb_gain * self.reverb_psd * reverb_gate)

        # Per-bin echo boost: high-coh2 bins → boost residual estimate
        if far_power > 1e-4 and erle_factor > 0.3 and residual_echo_psd is not None:
            echo_boost = 1.0 + 0.5 * coh2 if dt_for_fs < 0.2 else np.ones_like(coh2)
            residual_echo_psd = residual_echo_psd * echo_boost
        return residual_echo_psd

    def _stage_gain_compute(self, *, residual_echo_psd, eer, coh2, effective_dt,
                              is_stationary_dt, far_power, filter_once_converged,
                              spectral_g_min, eps,
                              erl_estimate: float = 0.01,
                              filter_converged: bool = False,
                              epc_active: bool = False,
                              filter_state: str = 'idle',
                              aec_state=None):
        """Stage 2: ENR / Wiener / spectral_sub gain compute + EMR + spectral floor lift.

        Returns g (post-spectral-floor). Mutates dominant_ne / Round 4 / Round 5
        diag caches and stats.

        `erl_estimate`, `filter_converged`, and `epc_active` are consumed
        only by the F3.1 mic-excess-evidence branch (default-OFF flag).
        Legacy and P4B paths are byte-identical to the pre-F3.1
        implementation.

        `filter_state` is the v3.12 Phase 2 wiring hook (C2). It receives
        the previous frame's enum state from FilterConvergenceAnalyzer
        (idle / startup / diverged / suspicious_dt / refined_usable /
        coarse_learning). Phase 2 plumbs the parameter only; Phase 3
        consumes it (gated by self._consume_filter_state) to drive the
        per-state ENR tuple and gain_floor unification.
        """
        if self.gain_type == "enr" and residual_echo_psd is not None:
            raw_nearend_est = np.maximum(self.error_psd - residual_echo_psd, 0.0)
            if self._noise_floor_refined:
                # S10: per-bin `error_psd × 0.005` in FS-confident bins
                # (coh² < 0.1); baseline scalar `mean(error_psd) × 0.01`
                # in DT/NE. Lowers nearend_est in FS only; DT/NE bins
                # unchanged. Audit-validated: 0% intrusion in S9-A.2.
                noise_floor_psd = np.where(
                    coh2 < 0.1,
                    self.error_psd * 0.005,
                    np.mean(self.error_psd) * 0.01,
                ).astype(self.error_psd.dtype) + 1e-10
            else:
                noise_floor_psd = np.mean(self.error_psd) * 0.01 + 1e-10

            # Per-bin DT indicator: base from coh2 (works for speech far-end).
            # P4B: γ²(k)-primary form removes the frame-scalar floor in the
            # ambiguous DTD regime (effective_dt 0.2–0.5) so per-bin γ²
            # discrimination survives instead of being clamped uniformly to
            # the frame value. effective_dt only contributes a soft floor
            # when it crosses 0.5 (DTD strongly evidences NE).
            # F3.1: per-bin mic-energy excess metric. Reuses the P1-Phase-1
            # validated `max(error_psd − far_lw·ERL_est, 0) / error_psd`
            # ratio (AUROC 0.871 in HF-cap gate) as the *primary* per-bin
            # NE evidence — replaces `(1 - coh2)` which saturates to 1 in
            # FS post-cancellation. Gated on `filter_converged` AND
            # long-window initialised so `erl_estimate` and `far_lw` are
            # reliable; falls through to the P4B / legacy path otherwise.
            _lw_ready = (
                self._residual_est is not None
                and self._residual_est._long_window_n_updates > 0
            )
            # F3.1 v3 (2026-05-12): blend with legacy `(1 - coh2)`
            # (weight=0.7 F3.1, 0.3 legacy) to soften HF over-
            # suppression in high-coupling rooms where erl_estimate
            # underestimates true ERL (Lsa5Wpw / wr54weK pattern:
            # mic/far 0.83/0.55, true ERL ~0.68/0.30 but erl_estimate
            # capped to 0.3 after EPC). Pure F3.1 over-attributes NE
            # → over-suppresses → spectral imbalance hurts AECMOS
            # even though total echo drops. Blend leaves F3.1 as the
            # dominant signal but caps its swing.
            #
            # The earlier v3 attempt also tried a `mic_pwr <= 2·far·erl`
            # envelope gate to block Regime-2 (pG9Bikvr non-echo
            # content). It correctly blocked the FS-noise case but
            # also blocked legitimate DT cases where mic naturally
            # exceeds expected echo (i2BU43nm). Adding `OR effective_dt
            # >= 0.2` softened the FS protection back out. Conclusion:
            # binary gating on mic/far ratio can't be made FS-only
            # without a label we don't have; the blend alone is the
            # honest cap.
            # v3.12 S7 (Phase 3B v3 Option α): when `_dt_per_bin_unified` is
            # True, drop the `NOT epc_active` constraint so the F3.1 v3
            # mic-excess blend also fires during EPC-active frames. Legacy
            # fallback at `else:` below remains the path for `not
            # filter_converged` or `not _lw_ready` regardless of flag.
            # v3.19 Phase 1 Branch G1 — when c_e_branch_dt_per_bin_use_fq_usable
            # is ON, the F3.1 v3 mic-excess gate uses fq_usable instead of
            # filter_converged. Default-OFF: byte-equal to legacy.
            _fc_g1 = filter_converged
            if (self._c_e_branch_dt_per_bin_use_fq_usable
                    and aec_state is not None
                    and getattr(aec_state, '_aec_ref', None) is not None):
                _fc_g1 = aec_state.fq_usable()
            if (self._use_mic_excess_evidence
                    and _fc_g1
                    and _lw_ready
                    and (self._dt_per_bin_unified or not epc_active)):
                far_lw = self._residual_est._long_window_far_psd
                # v3.14 Arc-P P.S2: when erl_estimate is a per-bin ndarray
                # (passed by AEC when f3_1_per_band_erl_adaptive=True), use it
                # directly (numpy broadcasting handles scalar and array equally).
                # When erl_estimate is scalar float (default-OFF path), float()
                # preserves existing behaviour — byte-equal guaranteed.
                if isinstance(erl_estimate, np.ndarray):
                    erl_e = erl_estimate  # per-bin array, shape (n_freqs,)
                else:
                    erl_e = float(erl_estimate)
                excess = np.maximum(self.error_psd - far_lw * erl_e, 0.0)
                excess_ratio = np.clip(
                    excess / (self.error_psd + 1e-10), 0.0, 1.0,
                ).astype(np.float32)
                # Blend with legacy `(1 - coh2)` to soften the over-attribution
                # under erl_estimate underestimation (Regime-3 mitigation).
                legacy = 1.0 - coh2
                dt_per_bin = (_BLEND_F31_MIC_EXCESS * excess_ratio
                              + (1.0 - _BLEND_F31_MIC_EXCESS) * legacy)
                if effective_dt > 0.5:
                    floor_lift = float((effective_dt - 0.5) * 2.0)
                    dt_per_bin = np.maximum(dt_per_bin, floor_lift)
            elif self._plan_b_dt_per_bin_gamma:
                dt_per_bin = (1.0 - coh2).astype(np.float32)
                if effective_dt > 0.5:
                    floor_lift = float((effective_dt - 0.5) * 2.0)
                    dt_per_bin = np.maximum(dt_per_bin, floor_lift)
            else:
                dt_per_bin = np.maximum(
                    np.full(self.n_freqs, effective_dt, dtype=np.float32),
                    1.0 - coh2
                )
            # S7 (Phase 3B v3 Option α) fire-rate audit hook. Zero-cost when
            # `_audit_counters` is None. Read-only; does not mutate dt_per_bin
            # or any other state. Computes unified-hypothetical dt_per_bin
            # alongside legacy and aggregates FS-bin (coh²<0.1) sums for
            # post-stream reduction-percentage analysis.
            if self._audit_counters is not None:
                _ac = self._audit_counters
                _ac['total_frames'] += 1
                _f31_active = (self._use_mic_excess_evidence
                               and filter_converged
                               and _lw_ready
                               and not epc_active)
                if _f31_active:
                    _ac['s7_f31v3_path_frames'] += 1
                elif self._plan_b_dt_per_bin_gamma:
                    _ac['s7_planb_path_frames'] += 1
                else:
                    _ac['s7_legacy_path_frames'] += 1
                    if epc_active:
                        _ac['s7_legacy_epc_active_frames'] += 1
                    if not filter_converged:
                        _ac['s7_legacy_not_converged_frames'] += 1
                    if not _lw_ready:
                        _ac['s7_legacy_not_lw_ready_frames'] += 1
                    _target_slice = filter_converged and _lw_ready and epc_active
                    _alt_slice = (filter_converged and _lw_ready
                                  and not epc_active)
                    if _target_slice or _alt_slice:
                        _far_lw_au = self._residual_est._long_window_far_psd
                        _erl_e_au = float(erl_estimate)
                        _excess_au = np.maximum(
                            self.error_psd - _far_lw_au * _erl_e_au, 0.0,
                        )
                        _excess_ratio_au = np.clip(
                            _excess_au / (self.error_psd + 1e-10), 0.0, 1.0,
                        ).astype(np.float32)
                        _legacy_au = (1.0 - coh2).astype(np.float32)
                        _unified_au = (
                            _BLEND_F31_MIC_EXCESS * _excess_ratio_au
                            + (1.0 - _BLEND_F31_MIC_EXCESS) * _legacy_au
                        )
                        if effective_dt > 0.5:
                            _floor_lift_au = float((effective_dt - 0.5) * 2.0)
                            _unified_au = np.maximum(_unified_au, _floor_lift_au)
                        _fs_mask_au = coh2 < 0.1
                        _n_fs_au = int(np.sum(_fs_mask_au))
                        if _n_fs_au > 0:
                            _legacy_sum_au = float(np.sum(dt_per_bin[_fs_mask_au]))
                            _unified_sum_au = float(np.sum(_unified_au[_fs_mask_au]))
                            if _target_slice:
                                _ac['s7_target_fs_bin_count'] += _n_fs_au
                                _ac['s7_target_fs_bin_legacy_sum'] += _legacy_sum_au
                                _ac['s7_target_fs_bin_unified_sum'] += _unified_sum_au
                            else:
                                _ac['s7_alt_fs_bin_count'] += _n_fs_au
                                _ac['s7_alt_fs_bin_legacy_sum'] += _legacy_sum_au
                                _ac['s7_alt_fs_bin_unified_sum'] += _unified_sum_au
                    else:
                        _ac['s7_legacy_target_other_frames'] += 1
            if is_stationary_dt:
                dt_per_bin = np.maximum(dt_per_bin, self._stat_dt_mask)
            # v3.8.4: stash for postprocess HF-cap "high bins NE confidence" gate
            self._dt_per_bin_last = dt_per_bin

            # P4B diag (zero-cost; means over small slices). Captures the
            # symptom this plan diagnoses: dt_per_bin and 1-coh2 both
            # saturate ~1.0 in DT and FS post-cancel.
            _hf_2k = self._hf_cap_bin_2k
            self._p4b_dt_per_bin_mean = float(np.mean(dt_per_bin))
            self._p4b_dt_per_bin_hf_mean = (
                float(np.mean(dt_per_bin[_hf_2k:]))
                if dt_per_bin.shape[0] > _hf_2k else 0.0
            )
            self._p4b_coh2_hf_mean = (
                float(np.mean(coh2[_hf_2k:])) if coh2.shape[0] > _hf_2k else 0.0
            )
            self._p4b_effective_dt = float(effective_dt)
            self._p4b_is_stationary_dt = int(is_stationary_dt)

            dt_shaped_per_bin = dt_per_bin ** 1.1
            nearend_est = np.maximum(raw_nearend_est * dt_shaped_per_bin, noise_floor_psd)

            min_ne_from_dt = self.error_psd * dt_shaped_per_bin
            _startup_dt_cond = (effective_dt > 0.35 and far_power > 1e-4
                                and not filter_once_converged)
            if _startup_dt_cond and self.startup_dt_min_ne_scale != 1.0:
                min_ne_from_dt = min_ne_from_dt * self.startup_dt_min_ne_scale
            if self._residual_est.using_render_based:
                min_ne_from_dt = min_ne_from_dt * getattr(self, 'render_min_ne_factor', 0.5)
            nearend_est = np.maximum(nearend_est, min_ne_from_dt)
            if self._stats is not None:
                self._stats_last_min_ne = float(np.mean(min_ne_from_dt))

            ne_physical_floor = self.error_psd * 0.05
            nearend_est = np.maximum(nearend_est, ne_physical_floor)

            # S8 audit: nearend_est 4-way binding identification on FS bins.
            # Of (raw * dt_shaped, noise_floor_psd, min_ne_from_dt,
            # ne_physical_floor), record which value is the MAX (=binding).
            if self._audit_counters is not None:
                _fs_mask_nef = coh2 < 0.1
                if np.any(_fs_mask_nef):
                    _v1 = raw_nearend_est * dt_shaped_per_bin
                    _v2 = np.full_like(_v1, noise_floor_psd)
                    _v3 = min_ne_from_dt
                    _v4 = ne_physical_floor
                    _stack = np.stack([_v1, _v2, _v3, _v4], axis=0)
                    _winner = np.argmax(_stack, axis=0)
                    _ac_nef = self._audit_counters
                    _ac_nef['s8_nef_raw_count'] += int(
                        np.sum((_winner == 0) & _fs_mask_nef))
                    _ac_nef['s8_nef_noise_floor_count'] += int(
                        np.sum((_winner == 1) & _fs_mask_nef))
                    _ac_nef['s8_nef_min_ne_count'] += int(
                        np.sum((_winner == 2) & _fs_mask_nef))
                    _ac_nef['s8_nef_ne_physical_count'] += int(
                        np.sum((_winner == 3) & _fs_mask_nef))

                    # S9 pre-audit: hypothetical noise_floor_psd
                    # refinements. Reuses _v1/_v3/_v4 and _winner from S8;
                    # only swaps v2 for each candidate and re-argmaxes.
                    _v2_a1 = np.full_like(_v1, float(np.mean(self.error_psd)) * 0.001 + 1e-10)
                    _v2_a2 = self.error_psd.astype(_v1.dtype) * 0.005 + 1e-10
                    _stack_a1 = np.stack([_v1, _v2_a1, _v3, _v4], axis=0)
                    _stack_a2 = np.stack([_v1, _v2_a2, _v3, _v4], axis=0)
                    _winner_a1 = np.argmax(_stack_a1, axis=0)
                    _winner_a2 = np.argmax(_stack_a2, axis=0)
                    _baseline_floor = (_winner == 1) & _fs_mask_nef
                    _baseline_max = _stack.max(axis=0)
                    for _label, _wA in (('a1', _winner_a1), ('a2', _winner_a2)):
                        _ac_nef[f's9_{_label}_release_to_raw'] += int(
                            np.sum(_baseline_floor & (_wA == 0)))
                        _ac_nef[f's9_{_label}_stays_floor'] += int(
                            np.sum(_baseline_floor & (_wA == 1)))
                        _ac_nef[f's9_{_label}_shift_to_min_ne'] += int(
                            np.sum(_baseline_floor & (_wA == 2)))
                        _ac_nef[f's9_{_label}_shift_to_phys'] += int(
                            np.sum(_baseline_floor & (_wA == 3)))
                        # Bins NOT bound by floor under baseline but whose
                        # winner CHANGES under the candidate (A.2 can do
                        # this when error_psd[i] is high enough that
                        # error_psd[i]*0.005 surpasses the current winner).
                        _not_floor = _fs_mask_nef & (_winner != 1)
                        _ac_nef[f's9_{_label}_intrudes_outside_floor_baseline'] += int(
                            np.sum(_not_floor & (_wA != _winner)))
                        # Magnitude: 10*log10(baseline_max / candidate_max)
                        # over bins where any change happened (only positive
                        # values count — candidate strictly lowers nearend_est).
                        _stack_X = _stack_a1 if _label == 'a1' else _stack_a2
                        _cand_max = _stack_X.max(axis=0)
                        _changed = _fs_mask_nef & (_cand_max < _baseline_max)
                        _n_changed = int(np.sum(_changed))
                        if _n_changed > 0:
                            _ratio = (_baseline_max[_changed] + 1e-30) / (
                                _cand_max[_changed] + 1e-30)
                            _ac_nef[f's9_{_label}_reduction_db_sum'] += 10.0 * float(
                                np.sum(np.log10(_ratio)))
                            _ac_nef[f's9_{_label}_reduction_count'] += _n_changed

                    # S9-C joint floor attack. Uses A.2 for noise_floor
                    # (per-bin error_psd * 0.005) and additionally
                    # scales min_ne_from_dt down in FS bins. Tracks fate
                    # of ALL baseline-floor-bound FS bins (not just
                    # noise_floor winners).
                    _baseline_any_floor = _fs_mask_nef & (_winner != 0)
                    _n_any_floor = int(np.sum(_baseline_any_floor))
                    _ac_nef['s9c_baseline_any_floor_count'] += _n_any_floor
                    if _n_any_floor > 0:
                        _v3_c1 = _v3 * 0.1                # min_ne × 0.1
                        _v3_c2 = np.zeros_like(_v3)        # min_ne → 0
                        _stack_c1 = np.stack(
                            [_v1, _v2_a2, _v3_c1, _v4], axis=0)
                        _stack_c2 = np.stack(
                            [_v1, _v2_a2, _v3_c2, _v4], axis=0)
                        _winner_c1 = np.argmax(_stack_c1, axis=0)
                        _winner_c2 = np.argmax(_stack_c2, axis=0)
                        for _lab, _wC, _stk in (
                                ('c1', _winner_c1, _stack_c1),
                                ('c2', _winner_c2, _stack_c2)):
                            _ac_nef[f's9c_{_lab}_release_to_raw'] += int(
                                np.sum(_baseline_any_floor & (_wC == 0)))
                            _ac_nef[f's9c_{_lab}_still_floor'] += int(
                                np.sum(_baseline_any_floor & (_wC == 1)))
                            _ac_nef[f's9c_{_lab}_still_min_ne'] += int(
                                np.sum(_baseline_any_floor & (_wC == 2)))
                            _ac_nef[f's9c_{_lab}_still_phys'] += int(
                                np.sum(_baseline_any_floor & (_wC == 3)))
                            _cand_max_c = _stk.max(axis=0)
                            _changed_c = _fs_mask_nef & (
                                _cand_max_c < _baseline_max)
                            _n_c = int(np.sum(_changed_c))
                            if _n_c > 0:
                                _r_c = (_baseline_max[_changed_c] + 1e-30) / (
                                    _cand_max_c[_changed_c] + 1e-30)
                                _ac_nef[
                                    f's9c_{_lab}_reduction_db_sum'
                                ] += 10.0 * float(np.sum(np.log10(_r_c)))
                                _ac_nef[
                                    f's9c_{_lab}_reduction_count'
                                ] += _n_c

                        # S9-D sanity: attack all three floors. Stack
                        # = [raw*dt_shaped, error_psd*0.005, 0, 0].
                        _stack_d = np.stack(
                            [_v1, _v2_a2, np.zeros_like(_v3),
                             np.zeros_like(_v4)], axis=0)
                        _winner_d = np.argmax(_stack_d, axis=0)
                        _ac_nef['s9d_release_to_raw'] += int(
                            np.sum(_baseline_any_floor & (_winner_d == 0)))
                        _ac_nef['s9d_still_floor'] += int(
                            np.sum(_baseline_any_floor & (_winner_d == 1)))
                        _cand_max_d = _stack_d.max(axis=0)
                        _changed_d = _fs_mask_nef & (_cand_max_d < _baseline_max)
                        _n_d = int(np.sum(_changed_d))
                        if _n_d > 0:
                            _r_d = (_baseline_max[_changed_d] + 1e-30) / (
                                _cand_max_d[_changed_d] + 1e-30)
                            _ac_nef['s9d_reduction_db_sum'] += 10.0 * float(
                                np.sum(np.log10(_r_d)))
                            _ac_nef['s9d_reduction_count'] += _n_d

            # Round 4 trace cache (audio-passive)
            self._diag_nearend_est_last = nearend_est
            self._diag_residual_echo_psd_last = residual_echo_psd

            enr = residual_echo_psd / (nearend_est + 1e-10)

            # v3.18 Phase D.3 / D-Path-D — Mask profile pathway.
            # Step 1 (always): compute legacy `enr_t / enr_s` via the
            # `ne_confidence × ne_anchor + (1-ne_confidence) × fs_anchor`
            # continuous interpolation. Includes per_band_enr (v3.14 Arc R)
            # and dt_ne_compression_fix (v3.15 §1.2). Byte-equal preserved.
            # Step 2 (D.3 / D-Path-D, gated): override / overlay with AEC3
            # `normal` / `nearend` profile tables based on swap mode.
            _emr_transparent_pb = None  # set when AEC3 path overlays (per-bin); legacy uses scalar 0.3

            # --- Step 1: legacy compute (always runs) ---
            blend = self._enr_blend
            scale = self.enr_scale
            ne_confidence = dt_per_bin
            effective_scale = scale
            # v3.14 Arc-R Sprint S1: per-band ENR threshold (default OFF).
            # When `_per_band_enr=True`, substitute the precomputed per-bin
            # arrays built from `enr_t_ne_per_band`/`enr_s_ne_per_band` tuples.
            # Default OFF keeps the legacy `_enr_blend` formula → byte-equal.
            if self._per_band_enr:
                enr_t_ne = self._enr_t_ne_pb
                enr_s_ne = self._enr_s_ne_pb
            else:
                enr_t_ne = (1 - blend) * 2.0 + blend * 1.5
                enr_s_ne = (1 - blend) * 3.0 + blend * 2.5
            enr_t_fs = (1 - blend) * (0.3 * effective_scale) + blend * (0.07 * effective_scale)
            enr_s_fs = (1 - blend) * (0.4 * effective_scale) + blend * (0.1 * effective_scale)
            if effective_dt > 0.4:
                dt_enr_relax = 1.0 + (effective_dt - 0.4) / 0.6 * 0.5
                enr_t_ne = enr_t_ne * dt_enr_relax
                enr_s_ne = enr_s_ne * dt_enr_relax
            # v3.15 §1.2 — DT-NE compression fix (default OFF, byte-equal).
            # Apply per-state scale then per-bin dt_per_bin override to
            # enr_t_ne / enr_s_ne.  refined_usable state-scale must be 1.0
            # to preserve byte-equal with v3.13 steady BALANCED path.
            if self._dt_ne_compression_fix:
                _state_scale = self._dt_ne_state_scale.get(
                    filter_state, 1.0)
                if _state_scale != 1.0:
                    enr_t_ne = enr_t_ne * _state_scale
                    enr_s_ne = enr_s_ne * _state_scale
                _bin_scale = self._dt_ne_per_bin_scale
                if _bin_scale != 1.0:
                    _mask = (dt_per_bin > self._dt_ne_per_bin_thresh)
                    if np.any(_mask):
                        # Broadcast scalar enr_t_ne / enr_s_ne to per-bin if needed.
                        if np.ndim(enr_t_ne) == 0:
                            enr_t_ne = np.full_like(dt_per_bin, float(enr_t_ne))
                        else:
                            enr_t_ne = np.asarray(enr_t_ne, dtype=np.float32).copy()
                        if np.ndim(enr_s_ne) == 0:
                            enr_s_ne = np.full_like(dt_per_bin, float(enr_s_ne))
                        else:
                            enr_s_ne = np.asarray(enr_s_ne, dtype=np.float32).copy()
                        enr_t_ne[_mask] = enr_t_ne[_mask] * _bin_scale
                        enr_s_ne[_mask] = enr_s_ne[_mask] * _bin_scale
            enr_t = ne_confidence * enr_t_ne + (1 - ne_confidence) * enr_t_fs
            enr_s = ne_confidence * enr_s_ne + (1 - ne_confidence) * enr_s_fs

            # --- Step 2 (gated): mask profile override / overlay ---
            if self._mask_profile_swap_enabled:
                if self._mask_swap_mode == 'asymmetric':
                    # D-Path-D (per-frame 3-state): FS-confident → use
                    # AEC3 `normal_profile` (all bins; recovers D.3 FS_static
                    # +0.258 Δecho); NE-confident → AEC3 `nearend_profile`;
                    # uncertain/DT → keep legacy `ne_confidence` interp.
                    # Avoids D.3 DT regression by NOT applying normal_profile
                    # when NE detector(s) failed to identify DT-NE segments.
                    # B1 (combined): uses _ne_combined_state (subband OR
                    # dominant). Dominant detector is echo-aware + hysteresis,
                    # complements subband's structural cue.
                    _is_fs = (
                        float(effective_dt) < self._mask_fs_overlay_dt_max
                        and not self._ne_combined_state
                    )
                    _is_ne = bool(
                        self._ne_combined_state
                        and float(effective_dt) > self._mask_ne_gate_dt)
                    if _is_fs:
                        _profile = self._normal_mask_profile
                        enr_t = _profile[0]
                        enr_s = _profile[1]
                        _emr_transparent_pb = _profile[2]
                        self._diag_mask_profile_nearend = False
                        self._diag_mask_fs_overlay_fraction = 1.0
                    elif _is_ne:
                        _profile = self._nearend_mask_profile
                        enr_t = _profile[0]
                        enr_s = _profile[1]
                        _emr_transparent_pb = _profile[2]
                        self._diag_mask_profile_nearend = True
                        self._diag_mask_fs_overlay_fraction = 0.0
                    else:
                        # Uncertain — keep legacy enr_t / enr_s from Step 1
                        self._diag_mask_profile_nearend = False
                        self._diag_mask_fs_overlay_fraction = 0.0
                else:
                    # D.3 binary: atomic swap between nearend/normal profile
                    # based on combined NE state AND echo-aware gate.
                    _use_nearend = bool(
                        self._ne_combined_state
                        and float(effective_dt) > self._mask_ne_gate_dt)
                    _profile = (self._nearend_mask_profile if _use_nearend
                                else self._normal_mask_profile)
                    enr_t = _profile[0]
                    enr_s = _profile[1]
                    _emr_transparent_pb = _profile[2]
                    self._diag_mask_profile_nearend = _use_nearend
                    self._diag_mask_fs_overlay_fraction = 0.0
            min_gate_width = 0.2
            enr_s_safe = np.maximum(enr_s, enr_t + min_gate_width)

            g = np.where(enr > enr_t,
                         np.clip((enr_s_safe - enr) / (enr_s_safe - enr_t + eps), 0.0, 1.0),
                         1.0)

            # EMR: AEC3-style noise masking. AEC3 uses per-bin emr_transparent
            # from the swapped profile; legacy path uses scalar 0.3.
            if np.sum(self.noise_psd) > 0:
                emr = residual_echo_psd / (self.noise_psd + 1e-10)
                if _emr_transparent_pb is not None:
                    g_emr = np.clip(_emr_transparent_pb / (emr + 1e-10), 0.0, 1.0)
                else:
                    g_emr = np.clip(0.3 / (emr + 1e-10), 0.0, 1.0)
                g = np.maximum(g, g_emr)

            if self._stats is not None:
                self._stats_last_gain_before_floor = float(np.mean(g))
            if getattr(self, '_capture_stages', False):
                self._stage_gains = {'01_softgate_emr': g.copy()}
            self._diag_round5_stages[0] = float(np.mean(g[self._voice_band_idx])) if self._voice_band_idx.size > 0 else 0.0
            g = np.maximum(g, spectral_g_min)
            if self._stats is not None:
                self._stats_last_gain_after_floor = float(np.mean(g))
            if getattr(self, '_capture_stages', False):
                self._stage_gains['02_spectral_floor'] = g.copy()
            self._diag_round5_stages[1] = float(np.mean(g[self._voice_band_idx])) if self._voice_band_idx.size > 0 else 0.0

            if self._stats is not None:
                self._stats_last_enr = float(np.mean(enr))
                self._stats_last_nearend = float(np.mean(nearend_est))
                self._stats_last_res_psd = float(np.mean(residual_echo_psd))

            # Phase 0 trace: dominant_nearend_like raw signal (frame-level)
            _ne_mean = float(np.mean(nearend_est))
            _res_mean = float(np.mean(residual_echo_psd))
            _noise_mean = float(np.mean(self.noise_psd)) + 1e-10
            self._diag_dominant_nearend_raw = bool(
                _ne_mean > 3.0 * _res_mean
                and _ne_mean > 5.0 * _noise_mean
            )

        elif self.gain_type == "wiener" and residual_echo_psd is not None:
            noise_floor_psd = np.mean(self.error_psd) * 0.01 + eps
            nearend_est = np.maximum(self.error_psd - residual_echo_psd, noise_floor_psd)
            beta = self.over_sub
            g = nearend_est / (nearend_est + beta * residual_echo_psd + eps)
            g = np.maximum(g, spectral_g_min)
        elif residual_echo_psd is not None:
            eer_direct = residual_echo_psd / (self.error_psd + eps)
            g = np.maximum(1.0 - self.over_sub * eer_direct, spectral_g_min)
        else:
            g = np.maximum(1.0 - self.over_sub * eer, spectral_g_min)
        return g

    def _stage_gain_postprocess(self, *, g_in, quiet_mask, far_power,
                                  effective_dt, is_stationary_dt, divergence,
                                  erl_estimate=0.01):
        """Stage 3: quiet mask, 3-bin smooth, HF cap, divergence override.

        Returns updated g (post-divergence-override). Mutates
        _diag_round5_stages[2..6]. Slot 2 is preserved as alias of slot 1
        post v3.16 C1 (epc_dt_cap removed; cap action fired 0/800 in
        v3.13 + v3.14 audits).
        """
        g = g_in
        # v3.16 C1 — epc_dt_cap removed (zero fire-rate). Slot 03 / stage 2
        # are preserved as aliases of stage 02 for diagnostic backward compat
        # (audit scripts + P52 res_refactored expect 9-slot _diag_round5_stages).
        if getattr(self, '_capture_stages', False):
            self._stage_gains['03_epc_dt_cap'] = g.copy()
        self._diag_round5_stages[2] = float(np.mean(g[self._voice_band_idx])) if self._voice_band_idx.size > 0 else 0.0

        g[quiet_mask] = 1.0  # Noise gate: pass through quiet bins
        if getattr(self, '_capture_stages', False):
            self._stage_gains['04_quiet_mask'] = g.copy()
        self._diag_round5_stages[3] = float(np.mean(g[self._voice_band_idx])) if self._voice_band_idx.size > 0 else 0.0

        # --- Frequency-domain postprocessing (cf. AEC3 PostprocessGains) ---
        if far_power > 1e-4:
            # 3-bin cross-frequency smoothing.
            # v3.8.4: kernel tightened from [0.25, 0.5, 0.25] to
            # [0.1, 0.8, 0.1]. The original kernel let low-band echo gain
            # leak into high bins via 25% sidelobes — measured 10 dB extra
            # cut in the 4–8 kHz band on case 7GTxyT (DT, BALANCED). The
            # tighter kernel still suppresses single-bin spurious spikes
            # (its original purpose) while preserving cross-band gain
            # independence, which keeps fricative / formant content audible.
            # P1.0 toggle: plan_a_kernel_tight=False reverts to v3.8.3 kernel
            if self._plan_a_kernel_tight:
                kernel = np.array([0.1, 0.8, 0.1], dtype=np.float32)
            else:
                kernel = np.array([0.25, 0.5, 0.25], dtype=np.float32)
            g = np.convolve(g, kernel, mode='same').astype(np.float32)
            if getattr(self, '_capture_stages', False):
                self._stage_gains['05_3bin_smooth'] = g.copy()
            self._diag_round5_stages[4] = float(np.mean(g[self._voice_band_idx])) if self._voice_band_idx.size > 0 else 0.0
            # DC consistency: bins 0-1 follow bin 2
            if self.n_freqs > 2:
                g[:2] = np.minimum(g[1], g[2])
            # v3.8.4: HF cap reworked. Two changes:
            #   (a) only fire when DTD is confident NE absent (was < 0.5,
            #       too permissive — fired throughout DT). Now < 0.3.
            #   (b) anchor the cap at 2 kHz instead of 500 Hz so vowel
            #       formants (1–3 kHz) are not dragged down by the 500 Hz
            #       bin's gain.
            #   (c) skip the cap entirely when high bins themselves show NE
            #       energy (per-bin DT confidence > 0.3 in 2 kHz+).
            # NOTE (v3.10.5 investigation): dt_per_bin = max(effective_dt,
            # 1-coh2) saturates ~1 in FS post-cancellation (echo cancelled →
            # low coh2 → "NE-like"), so the high_ne_conf < 0.3 gate rarely
            # fires in FS — cap is largely dead code there. Plan A's actual
            # FS cost lives in the smoothing kernel change, not here. Left
            # as-is pending a redesigned evidence metric that distinguishes
            # DT-NE from FS-decoupling.
            # P1 Phase 2 — conditional HF cap based on m_excess_ratio.
            # m_excess_ratio = mean(max(error_psd[2k:] - far_lw[2k:]·erl_est, 0))
            #                 / (mean(error_psd[2k:]) + eps)
            # Validated in P1 Phase 1 (AUROC 0.871 FS-vs-NE+DT_positive,
            # α=1.0 stable across {0.5, 1.0, 2.0}). Replaces the broken
            # `1 - coh2`-based high_ne_conf gate.
            if self._hf_cap_conditional:
                try:
                    far_lw = self._residual_est._long_window_far_psd
                    err_hb = self.error_psd[self._hf_cap_bin_2k:]
                    far_hb = far_lw[self._hf_cap_bin_2k:]
                    # v3.14 Arc-P P.S2: HF cap uses scalar ERL (broadband).
                    erl_e = (float(np.mean(erl_estimate[self._hf_cap_bin_2k:]))
                             if isinstance(erl_estimate, np.ndarray)
                             else float(erl_estimate))
                    excess = np.maximum(err_hb - 1.0 * far_hb * erl_e, 0.0)
                    err_hb_mean = float(np.mean(err_hb)) + 1e-10
                    metric = float(np.mean(excess)) / err_hb_mean
                except Exception:
                    metric = 0.0  # fail open: skip cap when metric unavailable
                if metric < self._hf_cap_metric_threshold:
                    # FS-like: apply v3.8.3 strict cap (anchor 500 Hz, gate < 0.5)
                    hf_cap_bin = self._hf_cap_bin
                    if (self.n_freqs > hf_cap_bin + 1
                            and effective_dt < 0.5
                            and not is_stationary_dt):
                        hf_cap = g[hf_cap_bin]
                        g[hf_cap_bin + 1:] = np.minimum(g[hf_cap_bin + 1:], hf_cap)
                # else: DT-like → skip cap entirely (preserve high-band NE)
            elif self._plan_a_hf_cap_2k:
                # Default v3.10.4 behaviour: Plan A cap anchor 2 kHz, gate < 0.3,
                # skip on (broken) high_ne_conf < 0.3
                hf_cap_bin = self._hf_cap_bin_2k
                if (self.n_freqs > hf_cap_bin + 1
                        and effective_dt < 0.3
                        and not is_stationary_dt):
                    high_ne_conf = float(np.mean(self._dt_per_bin_last[hf_cap_bin:]))
                    if high_ne_conf < 0.3:
                        hf_cap = g[hf_cap_bin]
                        g[hf_cap_bin + 1:] = np.minimum(g[hf_cap_bin + 1:], hf_cap)
            else:
                # P1.0 toggle: plan_a_hf_cap_2k=False reverts to v3.8.3 cap
                hf_cap_bin = self._hf_cap_bin
                if (self.n_freqs > hf_cap_bin + 1
                        and effective_dt < 0.5
                        and not is_stationary_dt):
                    hf_cap = g[hf_cap_bin]
                    g[hf_cap_bin + 1:] = np.minimum(g[hf_cap_bin + 1:], hf_cap)
            if getattr(self, '_capture_stages', False):
                self._stage_gains['06_hf_cap'] = g.copy()
            self._diag_round5_stages[5] = float(np.mean(g[self._voice_band_idx])) if self._voice_band_idx.size > 0 else 0.0

        # Divergence override: when filter diverges, cap gain severely
        if divergence > 0.3:
            divergence_gain = 0.01 + (1.0 - 0.01) * (1.0 - divergence)
            g = np.minimum(g, divergence_gain)
        if getattr(self, '_capture_stages', False):
            self._stage_gains['07_pre_temporal'] = g.copy()
        self._diag_round5_stages[6] = float(np.mean(g[self._voice_band_idx])) if self._voice_band_idx.size > 0 else 0.0

        # P4B diag: HF gain after pre-temporal postprocess + HF residual echo.
        _hf_2k = self._hf_cap_bin_2k
        if g.shape[0] > _hf_2k:
            self._p4b_gain_hf_mean = float(np.mean(g[_hf_2k:]))
        else:
            self._p4b_gain_hf_mean = 0.0
        _re = getattr(self, '_diag_residual_echo_psd_last', None)
        if _re is not None and _re.shape[0] > _hf_2k:
            self._p4b_res_echo_hf_mean_db = (
                10.0 * float(np.log10(float(np.mean(_re[_hf_2k:])) + 1e-12))
            )
        else:
            self._p4b_res_echo_hf_mean_db = -120.0
        return g

    def _stage_temporal_smoothing(self, *, g_in, dt_indicator, effective_dt,
                                   is_stationary_dt, erle_factor, fs_confidence,
                                   spectral_g_min, effective_g_min):
        """Stage 4: split attack/release EMA + rate limiting + LF protect + render ceil.

        Mutates: self.gain_smooth (= smoothed final gain),
                 self._diag_round5_stages[7],
                 self._stats_last_gain_after_smoothing.
        """
        g = g_in
        # Temporal DT: when Stationary DT confirmed, treat as dt=0.8 for smoothing/rate
        dt_temporal = 0.8 if is_stationary_dt else max(dt_indicator, effective_dt * 0.5)

        alpha_fast = 0.3 + 0.2 * (1.0 - erle_factor)   # 0.3-0.5
        alpha_slow = 0.85 + 0.1 * (1.0 - erle_factor)  # 0.85-0.95
        alpha_attack = alpha_slow + (alpha_fast - alpha_slow) * fs_confidence
        if is_stationary_dt:
            alpha_attack = alpha_attack * np.clip(1.0 - dt_temporal**2, 0.1, 1.0)
        alpha_release_light = 0.5 - 0.2 * dt_temporal
        smoothed = np.where(g < self.gain_smooth,
                            alpha_attack * self.gain_smooth + (1 - alpha_attack) * g,
                            alpha_release_light * self.gain_smooth + (1 - alpha_release_light) * g)

        # Rate limiting
        activity_scale = 0.5 + 0.5 * self.far_activity
        eff_drop = self.max_drop_ratio ** activity_scale
        rise_exp = 0.5 + 0.5 * (1.0 - self.far_activity)
        if dt_temporal > 0.3:
            dt_rise_boost = 1.0 + dt_temporal
            rise_exp = rise_exp / dt_rise_boost
        eff_rise = self.max_rise_ratio ** rise_exp
        gain_floor = self.gain_smooth / eff_drop
        gain_ceil = self.gain_smooth * eff_rise
        if fs_confidence < 0.9:
            lf_limit = min(8, self.n_freqs)
            lf_factor = 0.25 * max(1.0 - fs_confidence * 2.0, 0.0)
            if lf_factor > 0.01:
                gain_floor[:lf_limit] = np.maximum(
                    gain_floor[:lf_limit], self.gain_smooth[:lf_limit] * lf_factor)
        smoothed = np.maximum(smoothed, gain_floor)
        smoothed = np.minimum(smoothed, gain_ceil)
        if isinstance(spectral_g_min, np.ndarray):
            smoothed = np.maximum(smoothed, spectral_g_min)
        else:
            smoothed = np.maximum(smoothed, effective_g_min)
        smoothed = np.minimum(smoothed, 1.0)
        # v3.2 Axis 2: hard ceiling in render-mode whenever far is active.
        if self._residual_est.using_render_based and self.far_activity > 0.3:
            ceil = getattr(self, 'render_dt_gain_ceil', 0.6)
            smoothed = np.minimum(smoothed, ceil)
        if getattr(self, '_capture_stages', False):
            self._stage_gains['08_post_temporal'] = smoothed.copy()
        self._diag_round5_stages[7] = float(np.mean(smoothed[self._voice_band_idx])) if self._voice_band_idx.size > 0 else 0.0
        self.gain_smooth = smoothed
        if self._stats is not None:
            self._stats_last_gain_after_smoothing = float(np.mean(self.gain_smooth))

    def _stage_noise_floor_and_cng(self, *, spec_synth, far_power, effective_dt,
                                    dt_indicator, filter_once_converged,
                                    effective_g_min, hop):
        """Stage 5: noise_psd tracker + noise-floor lift + CNG + IFFT/OLA.

        Mutates: self.noise_psd, self.gain_smooth, self._smooth_cn_gain,
                 self.ola_buf, self._diag_round5_stages[8],
                 self._stats_last_* (when stats enabled).
        Returns: output[hop_size] (float32).
        """
        # E6: min-statistics noise tracker (always on, used for dynamic floor + CNG)
        # Tracks quiet-floor of residual over time. In DT with quiet near-end,
        # this floor includes near-end energy — so gain >= noise_floor_gain
        # preserves quiet speech above the learned floor (Speex/AEC3 style).
        if not self._noise_initialized:
            self.noise_psd = self.error_psd.copy() + 1e-8
            self._noise_initialized = True
            self._smooth_cn_gain = np.zeros(self.n_freqs, dtype=np.float32)
        is_learning_safe = (self.far_activity < 0.01) and (dt_indicator < 0.1)
        alpha_down = 0.98
        alpha_up = 0.998 if is_learning_safe else 1.0
        alpha_n = np.where(self.error_psd > self.noise_psd, alpha_up, alpha_down)
        self.noise_psd = alpha_n * self.noise_psd + (1 - alpha_n) * self.error_psd

        # Dynamic floor: per-bin minimum gain so output |g*spec| >= sqrt(noise_psd).
        spec_pwr_synth = np.abs(spec_synth) ** 2 + 1e-10
        noise_floor_gain = np.sqrt(self.noise_psd / spec_pwr_synth)
        noise_floor_gain = np.clip(noise_floor_gain, effective_g_min, 1.0)

        _startup_dt_nfl = effective_dt > 0.35 and far_power > 1e-4 and not filter_once_converged
        if self._stats is not None:
            _nfl_mean = float(np.mean(noise_floor_gain))
            self._stats_last_noise_floor_gain = _nfl_mean
            self._stats_last_noise_psd = float(np.mean(self.noise_psd))
            self._stats_last_spec_pwr = float(np.mean(spec_pwr_synth))
            self._stats_last_nfl_lifted = _nfl_mean > float(np.mean(self.gain_smooth))

        if _startup_dt_nfl and self.startup_dt_noise_floor_scale < 1.0:
            if self.startup_dt_noise_floor_scale > 0.0:
                self.gain_smooth = np.maximum(self.gain_smooth,
                                              noise_floor_gain * self.startup_dt_noise_floor_scale)
            # else scale=0.0: bypass — noise_floor_gain not applied
        else:
            self.gain_smooth = np.maximum(self.gain_smooth, noise_floor_gain)
        # Round 5 stage 8: final gain after noise-floor lift (= what reaches IFFT)
        self._diag_round5_stages[8] = float(np.mean(self.gain_smooth[self._voice_band_idx])) if self._voice_band_idx.size > 0 else 0.0

        # Apply gain + synthesis sqrt-Hann window + IFFT
        enhanced_spec = self.gain_smooth * spec_synth

        # --- CNG: Comfort Noise Generation (fill remaining suppression gap) ---
        if self.enable_cng and np.sum(self.noise_psd) > 1e-7:
            target_cn_gain = np.sqrt(np.maximum(1.0 - self.gain_smooth ** 2, 0.0)) * 0.4
            self._smooth_cn_gain = 0.8 * self._smooth_cn_gain + 0.2 * target_cn_gain
            noise_std = np.sqrt(self.noise_psd / 2.0).astype(np.float32)
            cng_real = np.random.randn(self.n_freqs).astype(np.float32) * noise_std
            cng_imag = np.random.randn(self.n_freqs).astype(np.float32) * noise_std
            cng_spec = (self._smooth_cn_gain * (cng_real + 1j * cng_imag)).astype(np.complex64)
            enhanced_spec = enhanced_spec + cng_spec

        enhanced_time = np.fft.irfft(enhanced_spec, self.block_size)[:self.frame_size]
        enhanced_time *= self.window

        # Overlap-add (frame_size buffer)
        self.ola_buf += enhanced_time
        output = self.ola_buf[:hop].copy()
        self.ola_buf[:-hop] = self.ola_buf[hop:]
        self.ola_buf[-hop:] = 0.0
        return output

    def process(self, error_hop: np.ndarray, echo_spec: np.ndarray,
                far_power: float, far_spec: np.ndarray = None,
                filter_converged: bool = False,
                erle_factor: float = 0.0,
                dt_indicator: float = 0.0,
                near_spec: np.ndarray = None,
                divergence: float = 0.0,
                is_stationary_dt: bool = False,
                saturation_level: float = 0.0,
                epc_active: bool = False,
                error_spec_from_filter: np.ndarray = None,
                shadow_dt: float = 0.0,
                erl_estimate: float = 0.01,
                e2_main: float = 0.0,
                e2_shadow: float = 0.0,
                y2: float = 0.0,
                filter_once_converged: bool = False,
                aec_state=None,
                filter_state: str = 'idle') -> np.ndarray:
        """Process hop-size error signal, return enhanced hop via OLA.

        far_spec: far-end frequency spectrum (complex), used for coherence-
                  based nonlinear echo PSD estimation.
        near_spec: mic signal spectrum (complex), used for direct echo method.
        error_spec_from_filter: error spectrum from PBFDAF (rectangular window,
                  aligned with far_spec/near_spec/echo_spec). When provided,
                  used for coherence/ENR calculation instead of the OLA spec.
        shadow_dt: double-talk confidence from shadow-filter-based DTD
                  (main/shadow error ratio). Unlike energy-based dt_indicator
                  which is suppressed by inst_erle correction in high-coupling
                  DT, shadow_dt is reliable in exactly the crush cases. Used
                  to drive effective_dt for ne_floor and ENR relaxation.
        """
        hop = self.hop_size

        # Slide in new error samples (frame_size buffer)
        self.input_buf[:-hop] = self.input_buf[hop:]
        self.input_buf[-hop:] = error_hop

        # Analysis: sqrt-Hann window + zero-pad to FFT size
        # This spec is used ONLY for synthesis (gain application + IFFT).
        windowed = self.input_buf * self.window
        spec_synth = np.fft.rfft(windowed, n=self.block_size)

        # Analysis spec for coherence/PSD/ENR: prefer filter's error_spec
        # (rectangular window, aligned with far_spec/near_spec/echo_spec).
        # Fallback to OLA spec if not provided (backward compat).
        if error_spec_from_filter is not None:
            spec = error_spec_from_filter
        else:
            spec = spec_synth

        # Compute power spectra
        echo_pwr_linear = np.abs(echo_spec) ** 2
        error_pwr = np.abs(spec) ** 2

        # --- Coherence-based echo PSD estimation ---
        # Coherence² between far-end and error IS the echo-to-error ratio:
        #   coh²[k] = |S_fe[k]|² / (S_ff[k] × S_ee[k])
        # This captures both linear and nonlinear echo (both correlate with
        # far-end). During DT, near-end is uncorrelated → coh² drops → less
        # suppression → near-end preserved.
        coh2 = np.zeros(self.n_freqs, dtype=np.float32)
        if far_spec is not None and far_power > 1e-4:
            a = self.alpha_coh
            self.S_fe = a * self.S_fe + (1 - a) * spec * np.conj(far_spec)
            self.S_ff = a * self.S_ff + (1 - a) * np.abs(far_spec) ** 2
            self.S_ee = a * self.S_ee + (1 - a) * error_pwr
            coh2_raw = np.abs(self.S_fe) ** 2 / (self.S_ff * self.S_ee + 1e-10)
            coh2_raw = np.minimum(coh2_raw, 1.0).astype(np.float32)
            # Asymmetric EMA: fast drop (DT protection) / slow rise (stable tracking)
            # After convergence: rise slower → coh2 more stable near 1.0 in echo-only
            # _coh2_smooth initialized in __init__ and cleared in reset()
            #
            # v3.19 Phase 1 Branch P1 — when c_e_branch_coh2_ema_use_fq_usable
            # is ON, the asymmetric tuning gate uses fq_usable instead of
            # filter_converged. Default-OFF: byte-equal to legacy.
            _fc_p1 = filter_converged
            if (self._c_e_branch_coh2_ema_use_fq_usable
                    and aec_state is not None
                    and getattr(aec_state, '_aec_ref', None) is not None):
                _fc_p1 = aec_state.fq_usable()
            if _fc_p1:
                a_coh_rise = 0.90   # TC≈160ms, stable echo-only tracking
                a_coh_drop = 0.50   # TC≈25ms, fast DT protection
            else:
                a_coh_rise = 0.80
                a_coh_drop = 0.50
            a_coh = np.where(coh2_raw < self._coh2_smooth, a_coh_drop, a_coh_rise)
            self._coh2_smooth = a_coh * self._coh2_smooth + (1.0 - a_coh) * coh2_raw
            coh2 = self._coh2_smooth
        else:
            if far_power <= 1e-4:
                self.S_fe *= 0.5
                self.S_ff *= 0.5
                self.S_ee *= 0.5  # A1: sync decay, prevent coh2 bias on far restart
        # Round 4 trace cache (audio-passive)
        self._diag_coh2_last = coh2

        # Cold start: skip EMA warmup, initialize PSD directly on first far-end frame
        if far_power > 1e-4 and np.sum(self.echo_psd) < 1e-10:
            self.echo_psd[:] = echo_pwr_linear
            self.error_psd[:] = error_pwr

        # Linear EER from adaptive filter echo estimate
        self.echo_psd = self.alpha_echo_psd * self.echo_psd + (1 - self.alpha_echo_psd) * echo_pwr_linear
        self.error_psd = self.alpha_error_psd * self.error_psd + (1 - self.alpha_error_psd) * error_pwr

        # Multi-ERLE update (Phase 2)
        far_active = far_power > 1e-4
        self._filter_erle_est.update(echo_spec, spec, far_active, dt_indicator)
        near_power_broad = float(np.mean(self.near_psd)) if near_spec is not None else 0.0
        error_power_broad = float(np.mean(error_pwr))
        self._fb_erle_est.update(near_power_broad, error_power_broad, far_active, dt_indicator)

        if far_power < 1e-4:
            self.echo_psd *= 0.3  # fast decay during far-end silence

        # --- Dynamic g_min: track far-end activity ---
        is_far_active = float(far_power > 1e-4)
        if is_far_active > self.far_activity:
            # Far-end resumes: fast attack (TC≈30ms, ~2 frames)
            self.far_activity = 0.7 * self.far_activity + 0.3 * is_far_active
        else:
            # Far-end stops: slow decay (TC≈800ms, wait for echo_psd to decay first)
            self.far_activity = 0.98 * self.far_activity + 0.02 * is_far_active
        # far_activity=1.0 → g_min normal; far_activity=0.0 → g_min→1.0 (no suppression)
        # Fixed g_min: gain floor is constant regardless of far_activity
        # (AEC3 style — gain is purely ENR-driven, not activity-gated)
        effective_g_min = self.g_min

        # --- Noise gate: don't suppress quiet segments ---
        signal_floor = np.mean(self.error_psd) * 0.001 + 1e-8
        quiet_mask = ((self.echo_psd < signal_floor)
                      & (self.error_psd < signal_floor))

        # --- Stationary DT virtual DT indicator ---
        dt_for_fs = 0.8 if is_stationary_dt else dt_indicator

        # Effective DT: includes pre-filter energy-based signal (shadow_dt
        # parameter, reused as transport for energy DT signal). This bypasses
        # the inst_erle correction that suppresses dt_indicator in high-
        # coupling DT. Take max so FS paths still rely on dt_for_fs ≈ 0
        # while DT paths get the energy signal.
        effective_dt = max(float(dt_for_fs), float(shadow_dt))
        self._last_effective_dt = effective_dt

        # v3.16 C1 — epc_dt gate computation removed. The legacy gate
        # `epc_active AND effective_dt > 0.35` fired 0/2,032,022 frames
        # in v3.13 + v3.14 audits; the corresponding 0.85 gain cap in
        # _stage_gain_postprocess was confirmed dead code. State-driven
        # variant retired with the cap action.

        eps = 1e-10
        residual_echo_psd = self._stage_residual_model(
            coh2=coh2,
            far_spec=far_spec,
            far_power=far_power,
            erle_factor=erle_factor,
            dt_for_fs=dt_for_fs,
            epc_active=epc_active,
            saturation_level=saturation_level,
            filter_converged=filter_converged,
            erl_estimate=erl_estimate,
            near_spec=near_spec,
            is_stationary_dt=is_stationary_dt,
            aec_state=aec_state,
            echo_pwr_linear=echo_pwr_linear,
        )

        # v3.8.x ABL-4 (ablate ERL-based linear_failed branch): trace-verified
        # `self._erl_estimate` clipped to [0.001, 1.0] in update path (~line
        # 3974), so `erl_estimate > 1.2` never triggers (R6 trace 2026-04-30:
        # 0.00% fire rate across all 5 buckets). Branch was dead code retained
        # as v3.7.1 PR-B "physical mic/far ratio" defense, but value is
        # structurally bounded → defense is impossible to engage.
        # Removed alongside ABL-1+2: completes the family of e2-floor /
        # mic-as-echo-proxy / error-as-echo-proxy structural cleanups.

        # Compute coherence-based EER (only used by legacy spectral_sub path)
        if self.gain_type not in ("enr", "wiener"):
            eer_linear = self.echo_psd / (self.error_psd + eps)
            eer_converged = eer_linear * (0.5 + 0.5 * coh2)
            if far_power > 1e-4:
                eer = (1.0 - erle_factor) * coh2 + erle_factor * eer_converged
            else:
                eer = eer_converged
        else:
            eer = None  # B3: not used in ENR/Wiener path


        # --- Spectral-shape-preserving floor ---
        if self.enable_spectral_floor and far_power > 1e-4:
            error_mag = np.sqrt(error_pwr + 1e-10)
            self.error_envelope = (self.alpha_envelope * self.error_envelope
                                   + (1 - self.alpha_envelope) * error_mag)
            env_max = np.max(self.error_envelope) + 1e-10
            env_normalized = self.error_envelope / env_max
            # Bins with more energy get higher floor → preserves spectral shape
            spectral_g_min = effective_g_min + (1.0 - effective_g_min) * env_normalized * self.spectral_floor_ratio
            spectral_g_min = np.maximum(spectral_g_min, effective_g_min)
        else:
            spectral_g_min = effective_g_min

        # fs_confidence: continuous FS/DT/NE indicator (single definition)
        # Used by ne_g_floor, ENR two-tuning, attack speed, LF rate limit
        fs_confidence = self.far_activity * (1.0 - effective_dt) ** 2.0

        # --- Per-bin near-end gate ---

        # v3.12 Phase 3B (S6-S7) — unified gain floor.
        # `ne_g_floor` uses `(1 - coh²)` as NE evidence by default. Q7 V3
        # verdict: this evidence saturates in FS post-cancellation
        # (echo cancelled → low coh² → "NE-like") and so falsely raises
        # the floor in FS, leaking echo. The unified path swaps in the
        # F3.1 v3 mic-excess blend (already validated AUROC 0.871, in
        # production use for `dt_per_bin` since v3.10.6). Gate matches
        # the F3.1 v3 path: filter_converged AND long-window-ready AND
        # not epc_active; fallback to `(1 - coh²)` when gate fails.
        if self._unified_gain_floor and self._use_mic_excess_evidence:
            _lw_ready_floor = (
                self._residual_est is not None
                and self._residual_est._long_window_n_updates > 0
            )
            if filter_converged and _lw_ready_floor and not epc_active:
                far_lw = self._residual_est._long_window_far_psd
                # v3.14 Arc-P P.S2: unified_gain_floor uses per-bin ERL when array.
                if isinstance(erl_estimate, np.ndarray):
                    erl_e = erl_estimate  # per-bin: broadcast against far_lw
                else:
                    erl_e = float(erl_estimate)
                excess = np.maximum(self.error_psd - far_lw * erl_e, 0.0)
                excess_ratio = np.clip(
                    excess / (self.error_psd + 1e-10), 0.0, 1.0,
                ).astype(np.float32)
                legacy_evidence = 1.0 - coh2
                ne_evidence = (_BLEND_F31_MIC_EXCESS * excess_ratio
                               + (1.0 - _BLEND_F31_MIC_EXCESS) * legacy_evidence)
            else:
                ne_evidence = 1.0 - coh2
        else:
            ne_evidence = 1.0 - coh2

        # --- Per-bin near-end gate with fs_confidence ---
        ne_erle_gate = max(erle_factor, 0.3)  # B4: simplified (0.2 floor never triggered)
        # Scale ne_protection by (1-fs_confidence): FS→no protection, DT/NE→full protection
        ne_protection = ne_evidence * ne_erle_gate * (1.0 - fs_confidence)
        ne_g_min_ceil = 10 ** (self.ne_protect_db / 20)
        ne_g_floor = effective_g_min + (ne_g_min_ceil - effective_g_min) * ne_protection
        ne_g_floor = np.maximum(ne_g_floor, effective_g_min)
        # v3.16 C1c — capture pre-floor spectral_g_min so audits can detect
        # ne_g_floor binding fires (any-bin) by comparing pre-floor vs ne_g_floor.
        # The legacy `_stats_last_spectral_g_min` writes the post-max value
        # below; reading it against ne_g_floor (as v3.15 audit did) is
        # mathematically False because post-max >= ne_g_floor element-wise.
        # Computed unconditionally (cheap; needed by audit hook even when
        # `self._stats is None`).
        self._stats_pre_max_spectral_g_min = float(np.mean(spectral_g_min))
        self._stats_pre_max_spectral_g_min_max = float(np.max(spectral_g_min))
        self._stats_ne_g_floor_max = float(np.max(ne_g_floor))
        self._stats_ne_g_floor_any_bin_fired = bool(
            np.any(ne_g_floor > spectral_g_min + 1e-7)
        )
        spectral_g_min = np.maximum(spectral_g_min, ne_g_floor)

        # startup_dt conditions: trigger when DT active + filter not converged
        _startup_dt_curr = effective_dt > 0.35 and far_power > 1e-4 and not filter_converged
        _startup_dt_once = effective_dt > 0.35 and far_power > 1e-4 and not filter_once_converged

        if self._stats is not None:
            self._stats_last_ne_g_floor = float(np.mean(ne_g_floor))
            self._stats_last_spectral_g_min = float(np.mean(spectral_g_min))

        # startup_dt gain floor cap: lower spectral_g_min ceiling for ablation
        if _startup_dt_curr and self.startup_dt_gain_floor < 1.0:
            spectral_g_min = np.minimum(spectral_g_min, self.startup_dt_gain_floor)

        g = self._stage_gain_compute(
            residual_echo_psd=residual_echo_psd,
            eer=eer,
            coh2=coh2,
            effective_dt=effective_dt,
            is_stationary_dt=is_stationary_dt,
            far_power=far_power,
            filter_once_converged=filter_once_converged,
            spectral_g_min=spectral_g_min,
            eps=eps,
            erl_estimate=erl_estimate,
            filter_converged=filter_converged,
            epc_active=epc_active,
            filter_state=filter_state,
            aec_state=aec_state,
        )
        g = self._stage_gain_postprocess(
            g_in=g,
            quiet_mask=quiet_mask,
            far_power=far_power,
            effective_dt=effective_dt,
            is_stationary_dt=is_stationary_dt,
            divergence=divergence,
            erl_estimate=erl_estimate,
        )

        self._stage_temporal_smoothing(
            g_in=g,
            dt_indicator=dt_indicator,
            effective_dt=effective_dt,
            is_stationary_dt=is_stationary_dt,
            erle_factor=erle_factor,
            fs_confidence=fs_confidence,
            spectral_g_min=spectral_g_min,
            effective_g_min=effective_g_min,
        )

        output = self._stage_noise_floor_and_cng(
            spec_synth=spec_synth,
            far_power=far_power,
            effective_dt=effective_dt,
            dt_indicator=dt_indicator,
            filter_once_converged=filter_once_converged,
            effective_g_min=effective_g_min,
            hop=hop,
        )

        # Per-frame stats accumulation (no-op when _stats is None)
        if self._stats is not None:
            s = self._stats
            dt_k = effective_dt > 0.35 and far_power > 1e-4
            s['total_frames'] += 1
            if dt_k:
                s['dt_active'] += 1
                s['dt_erle_sum'] += float(erle_factor)
                s['dt_gain_sum'] += float(np.mean(self.gain_smooth))
                s['dt_enr_sum'] += self._stats_last_enr
                s['dt_res_sum'] += self._stats_last_res_psd
                s['dt_ne_sum'] += self._stats_last_nearend
                if epc_active:               s['epc_dt'] += 1
                if erle_factor < 0.4:        s['low_erle_dt'] += 1
                if not filter_converged:     s['startup_dt'] += 1
                if epc_active or erle_factor < 0.4 or not filter_converged:
                    s['any_special_dt'] += 1
                # (2) filter convergence/divergence
                if filter_once_converged:    s['filter_once_converged_dt'] += 1
                if not filter_once_converged: s['startup_dt_once_conv'] += 1
                if divergence > 0.3:         s['filter_diverged_dt'] += 1
                _e2m_y2 = e2_main / (y2 + 1e-10)
                _e2s_y2 = e2_shadow / (y2 + 1e-10)
                _e2s_e2m = e2_shadow / (e2_main + 1e-10)
                s['e2_main_y2_sum'] += _e2m_y2
                s['e2_shadow_y2_sum'] += _e2s_y2
                s['e2_shadow_e2_main_sum'] += _e2s_e2m
                if e2_shadow < e2_main:      s['shadow_better_dt'] += 1
                # (3) usability candidates (DT frame)
                _uv1 = filter_converged
                _uv2 = _uv1 and erle_factor > 0.4
                _uv3 = _uv2 and divergence <= 0.3
                _uv4 = _uv3 and _e2s_e2m > 0.8       # shadow not better than main (same units)
                if _uv1: s['usable_v1'] += 1
                if _uv2: s['usable_v2'] += 1
                if _uv3: s['usable_v3'] += 1
                if _uv4: s['usable_v4'] += 1
                if not _uv1: s['unusable_dt'] += 1
                # (4) residual echo model — wire using_render directly from estimator
                if self._residual_est.using_render_based:
                    s['using_render_based_dt'] += 1
                s['min_ne_sum'] += self._stats_last_min_ne
                # (5) startup_dt gain stage diagnostics (only when not filter_converged)
                if not filter_converged:
                    s['st_ne_g_floor_sum'] += self._stats_last_ne_g_floor
                    s['st_spectral_g_min_sum'] += self._stats_last_spectral_g_min
                    s['st_gain_before_floor_sum'] += self._stats_last_gain_before_floor
                    s['st_gain_after_floor_sum'] += self._stats_last_gain_after_floor
                    s['st_gain_after_smoothing_sum'] += self._stats_last_gain_after_smoothing
                # (6) noise_floor_gain diagnostics (only when not filter_once_converged)
                if not filter_once_converged:
                    s['st_nfl_noise_floor_gain_sum'] += self._stats_last_noise_floor_gain
                    s['st_nfl_noise_psd_sum'] += self._stats_last_noise_psd
                    s['st_nfl_spec_pwr_sum'] += self._stats_last_spec_pwr
                    if self._stats_last_nfl_lifted: s['st_nfl_lifted_count'] += 1
                    s['st_nfl_final_gain_sum'] += float(np.mean(self.gain_smooth))

        # Diagnostic: store latest gains for external access
        self._diag_gain_mean = float(np.mean(self.gain_smooth))
        self._diag_gain_min = float(np.min(self.gain_smooth))
        self._diag_effective_g_min = float(effective_g_min)
        self._diag_far_activity = float(self.far_activity)
        self._diag_echo_psd_mean = float(np.mean(self.echo_psd))
        self._diag_error_psd_mean = float(np.mean(self.error_psd))

        # Round 4 per-bin RES diagnostics (audio-passive). Built only when ENR
        # path actually ran (residual_echo_psd not None). For non-ENR or no-far
        # frames, leave previous values stale (still float-valued, won't break
        # downstream).
        _coh2 = self._diag_coh2_last
        _ne = self._diag_nearend_est_last
        _res = self._diag_residual_echo_psd_last
        _err = self.error_psd
        _noise = self.noise_psd
        _vidx = self._voice_band_idx
        _eps = 1e-10
        _err_eps = _err + _eps
        # Clip ratios at 10: residual_echo_psd / nearend_est can be ≫1 in
        # quiet bins where error_psd ≈ 0; raw mean would be dominated by tails.
        _res_over_err = np.clip(_res / _err_eps, 0.0, 10.0)
        _ne_over_err = np.clip(_ne / _err_eps, 0.0, 10.0)
        _noise_over_err = np.clip(_noise / _err_eps, 0.0, 10.0)
        _g_final = self.gain_smooth
        _g_voice = _g_final[_vidx] if _vidx.size > 0 else _g_final
        _echo_dom = ((_coh2 > 0.5) & (_res_over_err > 0.5) & (_ne_over_err < 0.3))
        self._diag_round4 = {
            'coh2_mean_full': float(np.mean(_coh2)),
            'coh2_mean_voice': float(np.mean(_coh2[_vidx])) if _vidx.size > 0 else 0.0,
            'res_over_err_mean_full': float(np.mean(_res_over_err)),
            'res_over_err_mean_voice': float(np.mean(_res_over_err[_vidx])) if _vidx.size > 0 else 0.0,
            'ne_over_err_mean_full': float(np.mean(_ne_over_err)),
            'ne_over_err_mean_voice': float(np.mean(_ne_over_err[_vidx])) if _vidx.size > 0 else 0.0,
            'noise_over_err_mean_full': float(np.mean(_noise_over_err)),
            'g_voice_mean': float(np.mean(_g_voice)),
            'g_voice_min': float(np.min(_g_voice)),
            'g_voice_p10': float(np.percentile(_g_voice, 10)),
            'echo_dominant_bin_pct': float(np.mean(_echo_dom)),
        }

        return output.astype(np.float32)
