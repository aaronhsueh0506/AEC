"""AEC engine class, WAV pipeline and CLI ``main()``; ``aec.py`` re-exports these symbols."""
import os
import argparse
import numpy as np
from collections import deque
from typing import Optional
import soundfile as sf

from .enums import (
    AecDelayMode, AecMode, AecPreset, AecFilterState, _FREQ_MODES,
)
from .dataclasses import AecStats, AecResContext
from .delay.legacy_compat import LegacyDelayShim as DelayEstimator
from .preprocessing import HighPassFilter, SaturationDetector
from .filters import PBFDAF, PBFDKF
from .detectors import (
    RenderActivityDetector, FilterConvergenceAnalyzer,
    DoubleTalkAnalyzer,
)
from .epc import EchoPathChangeDetector, PathChangeRegimeHandler
from .config import AecConfig, ref_ring_samples, resolve_signal_grid
from .debug_logger import AecDebugLogger

# The four simple-mu values form one state machine and are not retimed per grid.
_SIMPLE_MU_HOLDOFF_HOPS = 20
_SIMPLE_MU_ALPHA_ATTACK = 0.3
_SIMPLE_MU_ALPHA_HOLD = 0.99
_SIMPLE_MU_ALPHA_RELEASE = 0.95

# Hops a Path-B delay shift has to prove itself before the bad-relock judgment compares windowed ERLE with the pre-shift value.
_DELAY_SHIFT_JUDGMENT_HOPS = 100


# Linear-filter output candidates, in the order the crossfade indexes them.
_SEL_COARSE, _SEL_REFINED, _SEL_CAPTURE = 0, 1, 2


class AEC:
    """Acoustic echo canceller: frequency-domain adaptive filtering (FDAF / PBFDAF / PBFDKF, default PBFDKF),
    shadow filter, double-talk handling and a residual-echo suppression post-filter."""

    # ── Convergence-state delegations (state lives in self._convergence) ─────
    @property
    def _filter_converged(self) -> bool: return self._convergence.converged
    @property
    def _filter_once_converged(self) -> bool: return self._convergence.once_converged
    @property
    def _divergence_indicator(self) -> float: return self._convergence.divergence

    # ── DT-signal delegations (state lives in self._dt_analyzer) ─────────────
    @property
    def _dt_from_energy(self) -> float: return self._dt_analyzer.dt_from_energy
    @property
    def _dt_from_shadow(self) -> float: return self._dt_analyzer.dt_from_shadow
    @property
    def _shadow_advantage(self) -> float: return self._dt_analyzer.shadow_advantage

    # Ablation: EPC sources ('delay_first', 'delay_shift', 'epv', 'shadow_rise') for which mark_diverged() is skipped.
    _epc_no_reset_sources: frozenset = frozenset()

    def _arc_m_q_boost(self, filt) -> None:
        """Set Q = Q_high, optionally scaled per band by ``_arc_m_band_scale`` stored on the filter."""
        if not hasattr(filt, 'Q_high'):
            return
        filt.Q = filt.Q_high.copy()
        _scale = getattr(filt, '_arc_m_band_scale', None)
        if _scale is not None:
            filt.Q = filt.Q * _scale

    def _maybe_mark_diverged(self, source: str) -> None:
        if source not in self._epc_no_reset_sources:
            self._convergence.mark_diverged()
        if not hasattr(self, '_divergence_source_counts'):
            self._divergence_source_counts = {
                'delay_first': 0,
                'delay_shift': 0,
                'epv': 0,
                'shadow_rise': 0,
            }
        self._divergence_source_counts[source] = (
            self._divergence_source_counts.get(source, 0) + 1
        )
        self._last_divergence_source = source


    def _update_misadjustment_estimator(
        self,
        near_end_hpf: Optional[np.ndarray] = None,
        raw_output: Optional[np.ndarray] = None,
    ) -> None:
        """Misadjustment estimator update: accumulate e^2 / y^2 over a 2-hop window and EMA e2/y2 into
        ``_aec3_misadj_inv`` (see ``_fire_aec3_misadj_scale``)."""
        if near_end_hpf is None or raw_output is None:
            return
        # Per-hop sums of squares over a 2-hop window; thresholds scale with the accumulated sample count.
        e2_block = float(np.sum(raw_output.astype(np.float64) ** 2))
        y2_block = float(np.sum(near_end_hpf.astype(np.float64) ** 2))
        self._aec3_misadj_e2_acum += e2_block
        self._aec3_misadj_y2_acum += y2_block
        self._aec3_misadj_n_acum += 1
        n_hops_target = 2  # window length in hops
        if self._aec3_misadj_n_acum < n_hops_target:
            return
        hop = int(raw_output.shape[0])
        total_samples = n_hops_target * hop
        # Thresholds: y2 floor 200 and e2 overhang trigger 7500 (int16 amplitude), converted to float.
        int16_sq = 32768.0 ** 2
        y2_threshold = (200.0 ** 2) * total_samples / int16_sq
        e2_overhang_threshold = (7500.0 ** 2) * total_samples / int16_sq
        if self._aec3_misadj_y2_acum > y2_threshold:
            update = self._aec3_misadj_e2_acum / max(self._aec3_misadj_y2_acum, 1e-20)
            if self._aec3_misadj_e2_acum > e2_overhang_threshold:
                self._aec3_misadj_overhang = 4
            else:
                self._aec3_misadj_overhang = max(self._aec3_misadj_overhang - 1, 0)
            if (update < self._aec3_misadj_inv) or (self._aec3_misadj_overhang > 0):
                self._aec3_misadj_inv += 0.1 * (update - self._aec3_misadj_inv)
        self._aec3_misadj_e2_acum = 0.0
        self._aec3_misadj_y2_acum = 0.0
        self._aec3_misadj_n_acum = 0

    def _fire_aec3_misadj_scale(self) -> None:
        """Shrink W by 2 / sqrt(inv_misadjustment) once it exceeds 10, after the hangover and a stable
        converged period (no EPC, main filter not paused)."""
        if self._misadjustment_hangover_remaining > 0:
            self._misadjustment_hangover_remaining -= 1
            return
        stable = (
            self._filter_converged
            and not self.epc_active
            and not self._regime_handler.main_paused
        )
        if not stable:
            self._misadjustment_stable_count = 0
            return
        self._misadjustment_stable_count += 1
        if self._misadjustment_stable_count < self.config.filter_misadjustment_stable_frames:
            return
        if self._aec3_misadj_inv <= 10.0:
            return
        # scale = 2 / sqrt(inv): 0.632 at inv = 10, 0.2 at inv = 100.
        scale_raw = 2.0 / max(self._aec3_misadj_inv, 1e-6) ** 0.5
        scale = max(self.config.filter_misadjustment_scale_min,
                    min(self.config.filter_misadjustment_scale_max,
                        scale_raw))
        self.filter.scale_filter(scale)
        self._aec3_misadj_inv = 0.0
        self._aec3_misadj_overhang = 0
        self._misadjustment_hangover_remaining = (
            self.config.filter_misadjustment_hangover_frames)

    # ── EPC-state delegations (state lives in self._epc_det) ─────────────────
    @property
    def epc_active(self) -> bool: return self._epc_det.active
    @property
    def epc_hangover_count(self) -> int: return self._epc_det.hangover_count
    @property
    def _epv_gain_fast(self) -> float: return self._epc_det.epv_gain_fast
    @property
    def _epv_gain_slow(self) -> float: return self._epc_det.epv_gain_slow

    _MODE_DEFAULT_MU = {
        AecMode.FDAF: 0.3,
        AecMode.PBFDAF: 0.5,
        AecMode.PBFDKF: 0.5,
        AecMode.SUBBAND: 0.5,
    }

    def __init__(self, config: Optional[AecConfig] = None):
        self.config = config or AecConfig()

        self._init_config()

        self._init_delay()

        block_size, n_partitions = self._init_linear_filter()

        _hop = self._init_post_state()

        _hop = self._init_render_analysis(_hop)

        self._init_shadow_filter()

        _aec3_scale_dtp = self._init_path_tracking()

        self._init_observer(block_size, n_partitions, _aec3_scale_dtp)

        self._init_trace_state(_hop)

        self._init_preprocessing()
        self._init_output_state()

    def _seed_poor_excitation_counters(self):
        """Construction-time excitation-counter seed for both filters, shared by ``__init__`` and
        ``reset()``; a no-op without a RenderSignalAnalyzer."""
        if self._render_signal_analyzer is None:
            return
        from . import aec3_scale as _aec3_scale_seed
        hops = _aec3_scale_seed.blocks_to_hops(
            1000, self.config.hop_size, self.config.sample_rate)
        self.filter._poor_excitation_counter = hops
        if self.shadow_filter:
            self.shadow_filter._poor_excitation_counter = hops
        # The live observer, once constructed, needs the same seed or its excitation guard holds its taps at zero.
        if getattr(self, '_observer_filter', None) is not None:
            self._observer_filter._poor_excitation_counter = hops

    def _rearm_filters_for_delay_change(self) -> None:
        """Re-arm every live filter's excitation counters after a delay change (soft acquisition,
        the full delay-change chain and the Path-B delay shift)."""
        for _filt in (self.filter, self.shadow_filter):
            if _filt is not None and hasattr(_filt, 'handle_echo_path_change'):
                _filt.handle_echo_path_change(
                    delay_change=True, gain_change=False, zero_filter=False)
        self._observer_filter.handle_echo_path_change(
            delay_change=True, gain_change=False, zero_filter=False)

    def _apply_delay_shift_hard_reset(self, target_delay: int) -> None:
        """Commit ``target_delay`` as the new ring-read alignment and run the full delay-shift recovery
        chain (filter-derived-state reset, EPC force_delay, Kalman Q/P boost, rearm), for a new,
        unverified delay. A revert to a known-good delay uses ``_apply_delay_shift_soft_revert``."""
        self._current_delay = target_delay
        self._reset_filter_derived_state(reason='delay_shift',
                                         preserve_render_ema=True)
        self._epc_det.force_delay()
        for filt in [self.filter, self.shadow_filter]:
            if filt is not None and hasattr(filt, 'Q'):
                self._arc_m_q_boost(filt)
                if isinstance(filt, PBFDKF):
                    filt._p_max_override = 1.0
                    filt._p_max_override_frames = 30
        self._maybe_mark_diverged('delay_shift')
        self._epc_reset_fired_this_frame = True
        from .delay.delay_types import DelayAdjustment as _DA
        self._aec3_pending_delay_change = _DA.NEW_DETECTED_DELAY
        self._rearm_filters_for_delay_change()

    def _apply_delay_shift_soft_revert(
            self, target_delay: int, bad_delay: int,
            pre_W: np.ndarray, pre_shadow_W: Optional[np.ndarray],
            pre_H_error: Optional[np.ndarray],
            pre_shadow_H_error: Optional[np.ndarray]) -> None:
        """Revert to the pre-shift delay and its tap / H_error snapshot without the hard-reset chain,
        since confidence in those taps was never lost. It re-points ``_current_delay``, restores W
        (and H_error), clears X_buf when the revert retards the alignment, reinitialises the
        windowed-ERLE accumulators and the main / shadow error EMAs, and drops any pending delay."""
        self._current_delay = target_delay
        self.filter.W[:] = pre_W
        if pre_H_error is not None and hasattr(self.filter, 'H_error_per_bin'):
            self.filter.H_error_per_bin[:] = pre_H_error
        if self.shadow_filter is not None:
            if pre_shadow_W is not None:
                self.shadow_filter.W[:] = pre_shadow_W
            if (pre_shadow_H_error is not None
                    and hasattr(self.shadow_filter, 'H_error_per_bin')):
                self.shadow_filter.H_error_per_bin[:] = pre_shadow_H_error
        # Sign rule as in warm_shift_ir: a retard invalidates the held far history, an advance leaves a sub-hop gap.
        _revert_delta = int(target_delay) - int(bad_delay)
        if _revert_delta < 0:
            self.filter.X_buf.fill(0)
            if self.shadow_filter is not None:
                self.shadow_filter.X_buf.fill(0)
        self._erle_window_near = 1e-10
        self._erle_window_err = 1e-10
        self.main_err_smooth = 0.0
        self.shadow_err_smooth = 0.0
        if hasattr(self, '_pending_delay'):
            del self._pending_delay
        if hasattr(self, '_pending_delay_ttl'):
            del self._pending_delay_ttl

    def reset(self):
        self.filter.reset()
        if self.shadow_filter:
            self.shadow_filter.reset()
            self.main_err_smooth = 0.0
            self.shadow_err_smooth = 0.0
        # reset() is the mid-stream relock wipe and leaves the startup gates standing; this method
        # restores a fresh instance, including the construction-time excitation-counter seed.
        self.filter.reset_carried_state()
        if self.shadow_filter:
            self.shadow_filter.reset_carried_state()
        self._observer_filter.reset_carried_state()
        self._seed_poor_excitation_counters()
        self._erle_slope_buf.clear()
        self._last_shadow_output_time = None
        self._refined_filter_output_last_selected = True
        self._aec3_misadj_e2_acum = 0.0
        self._aec3_misadj_y2_acum = 0.0
        self._aec3_misadj_n_acum = 0
        self._aec3_misadj_inv = 0.0
        self._aec3_misadj_overhang = 0
        if self._delay_active:
            if self.delay_est is not None:
                self.delay_est.reset()
                self._current_delay = -1
            else:
                self._current_delay = self.config.fixed_delay_samples
            self._ref_ring.fill(0)
            self._ref_ring_write = 0
            self._ref_ring_filled = 0
        self._epc_det.reset()
        self._convergence.reset()
        self._erle_window_near = 1e-10
        self._erle_window_err = 1e-10
        self._erle_factor_prev = 0.0
        self._inst_erle_smooth = 1.0
        self._simple_mu_ratio = 1.0
        self._simple_mu_holdoff = 0
        # Reset the shadow decoupled state; FilterAnalyzer reset is owned by AecState._full_reset.
        self._epc_reset_fired_this_frame = False
        self._warmup_frames = self.config.warmup_frames
        self._warmup_far_active = False
        self._far_active_blocks = 0
        self._dt_from_zero_count = 0
        self._diag = {
            'erle_inst': 0.0, 'mu_scale': 1.0, 'far_activity': 0.0,
            'res_gain_mean': 1.0, 'res_gain_min': 1.0, 'effective_g_min': 1.0,
            'converged': False, 'erle_factor': 0.0,
            'echo_psd_mean': 0.0, 'error_psd_mean': 0.0, 'divergence': 0.0,
            'using_render_based': False,
            'shadow_advantage': 1.0,
            'dt_from_energy': 0.0,
            'dt_from_shadow': 0.0,
            'dt_from_coherence': 0.0,
            'far_power': 0.0,
            'mic_power': 0.0,
            'erl_estimate': 0.1,
            'epc_active': False,
            'saturation_level': 0.0,
            'erle_windowed': 0.0,
            'dt_indicator': 0.0,
            'main_err_smooth': 0.0,
            'shadow_err_smooth': 0.0,
            'main_paused': False,
            'epv_gain_ratio': 1.0,
            'dt_residual_scale': 1.0,
            'filter_w_norm': 0.0,
            'shadow_w_norm': 0.0,
            'copy_err_baseline': 1e-6,
        }
        self.shadow_frame_count = 0
        self._regime_handler.reset()
        self._far_power_ema = 0.0
        self._mic_power_ema = 0.0
        self._frame_count = 0
        if self._freq_near_queue is not None:
            self._freq_near_queue.fill(0)
            self._freq_far_queue.fill(0)
            self._freq_out_buf.fill(0)
            self._freq_queue_write = 0
            self._freq_out_valid = 0
            self._freq_out_read = 0
        self.near_power = 0.0
        self.error_power = 0.0
        self.raw_error_power = 0.0
        self.near_power_sum = 0.0
        self.error_power_sum = 0.0
        self.raw_error_power_sum = 0.0
        if hasattr(self, '_pending_delay'):
            del self._pending_delay
        # A fresh instance has no basis for a prior rejection.
        if hasattr(self, '_delay_reject_values'):
            del self._delay_reject_values
        if hasattr(self, '_delay_shift_judgment'):
            del self._delay_shift_judgment
        for _attr in ('_divergence_source_counts', '_last_divergence_source',
                      '_diag_prev_delay', '_diag_prev_divergence_counts',
                      '_raw_dt_pre_epc',
                      '_dominant_nearend_hold',
                      '_poor_coarse_counter', '_coarse_reset_hangover',
                      '_block_stationary_for_next_hop',
                      # Near-end recency pair and the dt_protect 'dne' counters: state armed by the run this reset ends must not steer the next one.
                      '_ne_above', '_ne_recent_frames',
                      '_dt_protect_frames', '_dne_confirm_counter',
                      '_leakage_div_sustained_counter'):
            if hasattr(self, _attr):
                delattr(self, _attr)
        if getattr(self, '_observer_filter', None) is not None:
            self._observer_filter.reset()
            self._observer_far_end = None
            self._observer_obs_err_ema = 0.0
            self._observer_main_err_ema = 0.0
        if self._hp_mic is not None:
            self._hp_mic.reset()
        if self._sat_detector_ref is not None:
            self._sat_detector_ref.reset()
            self._sat_detector_mic.reset()
        self._saturation_level = 0.0
        self._render_activity.reset()
        self._inst_erle_smooth = 1.0
        self._wn_err_baseline = 1e-8
        self._stat_dt_hangover = 0  # stationary double-talk hold-off counter (frames)
        self._formed_output = None
        self._per_bin_mu_scale = None
        self._dt_analyzer.reset()
        self._epc_render_forced_remaining = 0
        self._erl_estimate = 0.1
        # Clear the post-filter state so a reused instance carries nothing across streams.
        self._reset_aec3_post()

    def _build_sg_kwargs(self, n_bins, sg_config):
        """SuppressionGain constructor kwargs shared by ``__init__`` and ``_reset_aec3_post``."""
        cfg = self.config
        return dict(
            n_bins=n_bins,
            config=sg_config,
            sr=cfg.sample_rate,
            hop_size=cfg.hop_size,
            use_wallclock_block_energy_threshold=True,
            # use_wallclock_ema_alpha is intentionally not passed (class default False).
            use_wallclock_gain_ratchet=True,
            soft_nearend_blend_enabled=bool(
                getattr(cfg, "soft_nearend_blend_enabled", False)),
            soft_nearend_blend_enr_threshold=float(
                getattr(cfg, "soft_nearend_blend_enr_threshold", 0.25)),
            soft_nearend_blend_softness=float(
                getattr(cfg, "soft_nearend_blend_softness", 0.25)),
            soft_nearend_blend_per_bin=bool(
                getattr(cfg, "soft_nearend_blend_per_bin", False)),
            split_floor_enabled=bool(
                getattr(cfg, "min_gain_split_floor_enabled", True)),
            split_floor_far_active_db=float(
                getattr(cfg, "min_gain_floor_far_active_db", -22.0)),
            split_floor_far_silent_db=float(
                getattr(cfg, "min_gain_floor_far_silent_db", -12.0)),
            split_floor_dt_db=float(
                getattr(cfg, "min_gain_floor_dt_db", -20.0)),
            split_floor_latch_power=float(
                getattr(cfg, "min_gain_far_latch_power", 1.0e6)),
            per_bin_floor_enabled=bool(
                getattr(cfg, "per_bin_floor_enabled", False)),
        )

    def _reset_coherence_state(self) -> None:
        """Initialise or clear the per-bin coherence-gate EMA accumulators.

        The ERLE-gate coherence tracker has no other reset path, so a path change, delay reset or
        instance reuse must clear it or the stale value keeps gating ERLE against the old path.
        """
        n = int(self.filter.n_freqs) if hasattr(self.filter, 'n_freqs') else 1
        self._coh_erle_sye = np.zeros(n, dtype=np.complex64)
        self._coh_erle_syy = np.full(n, 1e-30, dtype=np.float64)
        self._coh_erle_see = np.full(n, 1e-30, dtype=np.float64)

    def _reset_aec3_post(self, *, preserve_render_side: bool = False) -> None:
        """Clear the post-filter chain state. ``preserve_render_side=True`` (filter-derived recovery)
        keeps the render-stationarity context; ``reset()`` clears everything. AecState and SuppressionGain
        are recreated; the other trackers are reset in place."""
        if self._aec3_state is None:
            return  # AEC3 chain wasn't initialised (no filter)
        from .state import AecState as _Aec3State, AecStateConfig as _Aec3StateConfig
        from .residual import ResidualEchoEstimator, SuppressionGain
        n_bins = self._aec3_n_bins
        self._aec3_state = _Aec3State(_Aec3StateConfig(
            n_bins=n_bins,
            enable_transparent_mode=False,
            enable_filter_analyzer=True,
            hop_size=int(self.config.hop_size),
            sample_rate=int(self.config.sample_rate),
        ))
        # Re-derive the FFT-density-scaled echo model so a reset keeps the same per-bin PSD floors as init.
        from .residual.residual_echo_estimator import EchoModelConfig
        from . import aec3_scale as _aec3_scale_floors
        _ff_for_floors_reset = float(self.filter.fft_size) if hasattr(self.filter, "fft_size") else 512.0
        _echo_model_cfg_reset = EchoModelConfig(
            min_noise_floor_power=_aec3_scale_floors.fft_density_scale(
                EchoModelConfig().min_noise_floor_power, int(_ff_for_floors_reset)
            ),
        )
        self._aec3_ree = ResidualEchoEstimator(
            n_bins=n_bins,
            echo_model=_echo_model_cfg_reset,
            sr=self.config.sample_rate,
            hop_size=self.config.hop_size,
            use_aec3_residual_noise_gate=True,
            use_aec3_echo_gen_window=True,
            use_aec3_wallclock_reverb_smoothing=True,
            nl_r2_enabled=bool(getattr(self.config, "nl_r2_enabled", False)),
            nl_r2_alpha=float(getattr(self.config, "nl_r2_alpha", 0.1)),
            nl_r2_cap_multiple=float(
                getattr(self.config, "nl_r2_cap_multiple", 0.0)
            ),
            use_stationarity_properties=True,
        )
        self._aec3_ree._reverb_tail_strength = float(
            getattr(self.config, "reverb_tail_strength", 1.0))
        self._aec3_sg = SuppressionGain(
            **self._build_sg_kwargs(n_bins, self._aec3_sg_config))
        # The coherence-gate EMAs derive from the filter output, so they are always cleared.
        self._reset_coherence_state()
        self._aec3_ola_buf.fill(0)
        self._aec3_pending_gain_change = False
        self._aec3_pending_delay_change = None
        self._form_prev_output_time = None
        self._form_last_selection = _SEL_REFINED
        if hasattr(self, "_aec3_avg_render_reverb"):
            self._aec3_avg_render_reverb.reset()
        self._aec3_y2_smoothed.fill(0.0)
        self._aec3_n2.fill(1.0e6)
        self._aec3_n2_initial.fill(0.0)
        self._aec3_n2_counter = 0
        self._aec3_cng_seed = 42
        self._aec3_noise_initialized = False
        if not preserve_render_side:
            # Render-side context: cleared on full reset, kept on filter-derived recovery.
            if self._aec3_stationarity is not None:
                self._aec3_stationarity.reset()
            self._aec3_non_zero_render_seen = False
            self._aec3_stationarity_active_hops = 0

    def _reset_filter_derived_state(self, reason: str = 'plateau',
                                     preserve_render_ema: bool = True) -> None:
        """Clear all filter-output-derived state after a plateau (``reason='plateau'``) or a first delay
        acquisition (``'delay_first'``), when the taps were trained on bad input.

        Preserved (input side): frame count, delay estimator and ``_current_delay``, the reference ring,
        power EMAs, high-pass filters, saturation detectors, render activity and the render-side stationarity trackers.
        Cleared: taps, convergence and EPC state, error smoothers, DT analyzer, ERLE / mu / ERL state, the post chain and diagnostics.
        """
        self.filter.reset()
        if self.shadow_filter is not None:
            self.shadow_filter.reset()
        # The live observer's taps were trained at the previous alignment; _ref_ring is kept, so it re-learns from valid history.
        if getattr(self, '_observer_filter', None) is not None:
            self._observer_filter.reset()
            self._observer_obs_err_ema = 0.0
            self._observer_main_err_ema = 0.0

        self._convergence.reset()

        self._reset_filter_latches()

        self._reset_aec3_post(preserve_render_side=True)

        # Clear the pending delay so a stale value cannot pair with a later estimate and force a spurious delay.
        # The session-cumulative diagnostic counters survive; only AEC.reset() clears them.
        if hasattr(self, '_pending_delay'):
            del self._pending_delay
        if hasattr(self, '_pending_delay_ttl'):
            del self._pending_delay_ttl

    def _reset_filter_latches(self) -> None:
        """Filter-derived latches, EMAs and detectors that a restarted filter must not carry forward;
        shared by ``_reset_filter_derived_state`` and ``_reset_filter_for_realign``."""
        self._epc_det.reset()

        self.main_err_smooth = 0.0
        self.shadow_err_smooth = 0.0
        self.error_power = 0.0
        self.raw_error_power = 0.0
        self.error_power_sum = 0.0
        self.raw_error_power_sum = 0.0
        # near_power is reset with error_power so get_erle_instant() does not spike on a stale mic EMA.
        self.near_power = 0.0
        self.near_power_sum = 0.0

        self._erle_window_near = 1e-10
        self._erle_window_err = 1e-10
        self._erle_factor_prev = 0.0
        self._inst_erle_smooth = 1.0

        self._simple_mu_ratio = 1.0
        self._simple_mu_holdoff = 0
        self._per_bin_mu_scale = None

        self._erl_estimate = 0.1
        self._epc_render_forced_remaining = 0
        self._dt_analyzer.reset()
        self._stat_dt_hangover = 0

        self.shadow_frame_count = 0
        self._regime_handler.reset()

        self._post_reset_age_frames = 0
        self._erle_slope_buf.clear()
        self._p3f_refined_latched = False
        self._p3f_main_err_baseline = 0.0
        self._p3f_diverged_streak = 0

        for k, v in (('erle_inst', 0.0), ('mu_scale', 1.0), ('far_activity', 0.0),
                     ('res_gain_mean', 1.0), ('res_gain_min', 1.0),
                     ('effective_g_min', 1.0), ('converged', False),
                     ('erle_factor', 0.0), ('echo_psd_mean', 0.0),
                     ('error_psd_mean', 0.0), ('divergence', 0.0),
                     ('using_render_based', False), ('shadow_advantage', 1.0),
                     ('dt_from_energy', 0.0), ('dt_from_shadow', 0.0),
                     ('dt_from_coherence', 0.0), ('erl_estimate', 0.1),
                     ('epc_active', False), ('erle_windowed', 0.0),
                     ('dt_indicator', 0.0), ('main_err_smooth', 0.0),
                     ('shadow_err_smooth', 0.0), ('main_paused', False),
                     ('epv_gain_ratio', 1.0), ('dt_residual_scale', 1.0),
                     ('filter_w_norm', 0.0), ('shadow_w_norm', 0.0),
                     ('copy_err_baseline', 1e-6)):
            self._diag[k] = v

        # Coarse-filter quality counters (created lazily in the process loop).
        self._poor_coarse_counter = 0
        self._coarse_reset_hangover = 0
        self._leakage_div_sustained_counter = 0
        self._block_stationary_for_next_hop = False

        for filt in [self.filter, self.shadow_filter]:
            if filt is not None and hasattr(filt, 'Q'):
                if hasattr(filt, 'Q_high'):
                    self._arc_m_q_boost(filt)
                if hasattr(filt, '_p_max_override'):
                    filt._p_max_override = 1.0
                    filt._p_max_override_frames = 30
        self._warmup_frames = max(self._warmup_frames,
                                   self.config.warmup_frames // 2)
        self._warmup_far_active = False

    def _reset_filter_for_realign(self) -> None:
        """Filter-only restart for a caller-side far realign (see ``apply_external_realign``).

        Preserved: the AEC3 post chain (including the overlap-add tail), the mic half of the analysis frame, the delay-side
        state and ``_frame_count``. Cleared: taps, per-partition far spectra, far-power normalizer and ``far_buffer``
        (captured at the abandoned alignment). It also re-arms the stationarity counter so a wiped filter is not held frozen.
        """
        self.filter.reset_taps()
        self.filter.far_buffer.fill(0)
        if self.shadow_filter is not None:
            self.shadow_filter.reset_taps()
            self.shadow_filter.far_buffer.fill(0)
        if getattr(self, '_observer_filter', None) is not None:
            self._observer_filter.reset_taps()
            self._observer_filter.far_buffer.fill(0)
            self._observer_obs_err_ema = 0.0
            self._observer_main_err_ema = 0.0

        # mark_diverged directly: a caller-driven realign is not one of the per-source recovery paths.
        self._convergence.mark_diverged()
        self._reset_filter_latches()

        self._aec3_stationarity_active_hops = 0

    def apply_external_realign(self, delta_samples: int) -> int:
        """Realign an ``EXTERNAL_ALIGNED`` instance after a caller-side change of the far alignment, without ``reset()``.

        ``delta_samples`` = new_alignment - old_alignment (positive: the aligned far stream advanced). If
        ``linear_is_cancelling()`` and the shift fits the tap span, the learned IR is shifted; otherwise the filter
        is reset (overlap-add and mic-side frame untouched). A retard clears the far history, exposing echo until it refills.

        Returns 1 for the warm tap-transfer, 0 for a filter reset (also for ``delta_samples == 0``), -1 when not ``EXTERNAL_ALIGNED``.
        """
        if self.config.delay_mode is not AecDelayMode.EXTERNAL_ALIGNED:
            return -1
        delta = int(delta_samples)
        if delta == 0:
            return 0
        # The shift must fit the tap span in both directions; the windowed evidence exists only when the post chain runs.
        warm_ok = (self.config.delay_acquire_warm_transfer
                   and self.linear_is_cancelling()
                   and abs(delta) < self.filter.tap_span)
        if warm_ok:
            for filt in (self.filter, self.shadow_filter):
                if filt is not None:
                    filt.warm_shift_ir(delta)
            return 1
        self._reset_filter_for_realign()
        return 0

    def set_preset(self, preset, ramp_ms: float = 0.0) -> None:
        """Retarget the residual-echo strength axis on a running instance without disturbing filter state.

        ``ramp_ms`` governs the far-active floor only (a linear walk in dB); the DT floor and the split-floor gate
        step immediately. ``ramp_ms = 0`` lands on the value a fresh instance built with ``preset`` would use.
        Call between hops only. Raises ``ValueError`` on an unknown preset or an out-of-range ``ramp_ms``,
        leaving the instance untouched (the C API returns -1).
        """
        # Coerce first: AecConfig.from_preset() falls back to balanced for unknown values, which a setter must reject.
        try:
            preset = AecPreset(preset)
        except (TypeError, ValueError):
            raise ValueError(f"unknown AEC preset: {preset!r}")
        target_cfg = AecConfig.from_preset(preset)
        db = target_cfg.min_gain_floor_far_active_db
        dt_db = target_cfg.min_gain_floor_dt_db
        enabled = target_cfg.min_gain_split_floor_enabled
        # Validate through the suppressor first so a rejected call cannot leave self.config disagreeing with the floor in force.
        self._aec3_sg.set_split_floor_far_active_db(db, ramp_ms)
        self._aec3_sg.set_split_floor_dt_db(dt_db)
        self._aec3_sg.set_split_floor_enabled(enabled)
        self.config.min_gain_floor_far_active_db = db
        self.config.min_gain_floor_dt_db = dt_db
        self.config.min_gain_split_floor_enabled = enabled

    @property
    def hop_size(self) -> int:
        return self._hop_size

    def _get_simple_mu_scale(self, mu_min: float = None):
        """Get mu_scale from smoothed EER (per-bin array or scalar fallback)."""
        if mu_min is None:
            mu_min = self.config.shadow_mu_min
        if self._warmup_frames > 0:
            if self._warmup_far_active:
                self._warmup_frames -= 1
            if self._simple_mu_ratio < 0.2:
                return max(0.2, self._simple_mu_ratio)
            return min(1.0, max(0.5, self._simple_mu_ratio + 0.2))
        if not self._filter_converged:
            mu_min = max(mu_min, 0.3)  # pre-convergence floor
        else:
            mu_min = max(mu_min, 0.2)
        # Per-bin mu_scale from the previous frame's post-RES echo/error PSD.
        if self._per_bin_mu_scale is not None:
            return np.maximum(self._per_bin_mu_scale, mu_min)
        return mu_min + (1.0 - mu_min) * self._simple_mu_ratio

    def _update_simple_mu_ratio(self, output: np.ndarray,
                                 far: np.ndarray) -> None:
        """Update the variable mu ratio from the far/error power ratio with an asymmetric EMA
        (fast attack when DT drops the ratio, slow release after a holdoff)."""
        error_power = np.mean(output ** 2) + 1e-10
        far_power = np.mean(far ** 2) + 1e-10

        if far_power < 1e-6 and error_power < 1e-6:
            return

        if far_power > 1e-4 and self._simple_mu_ratio < 0.1:
            self._simple_mu_ratio = 0.8
            self._simple_mu_holdoff = 0
            return

        ratio = min(far_power / error_power, 1.0)
        if hasattr(self.filter, 'echo_spec') and self.filter.echo_spec is not None:
            echo_est_pwr = np.sum(np.abs(self.filter.echo_spec) ** 2) + 1e-10
            near_pwr = np.sum(np.abs(getattr(self.filter, 'near_spec', np.zeros(1))) ** 2) + 1e-10
            if near_pwr > 1e-8:
                ratio_echo = np.clip(echo_est_pwr / near_pwr, 0.0, 1.0)
                ratio = max(ratio, ratio_echo * 0.5)

        if ratio < self._simple_mu_ratio:
            alpha = _SIMPLE_MU_ALPHA_ATTACK
            # Arm the holdoff only on a fresh attack, so marginal double-talk cannot keep re-arming it.
            if self._simple_mu_holdoff == 0:
                self._simple_mu_holdoff = _SIMPLE_MU_HOLDOFF_HOPS
        elif self._simple_mu_holdoff > 0:
            self._simple_mu_holdoff -= 1
            alpha = _SIMPLE_MU_ALPHA_HOLD  # nearly frozen
        else:
            alpha = _SIMPLE_MU_ALPHA_RELEASE
        self._simple_mu_ratio = alpha * self._simple_mu_ratio + (1 - alpha) * ratio

    def process(self, near_end: np.ndarray, far_end: np.ndarray) -> np.ndarray:
        near_end, far_end = self._prepare_capture(near_end, far_end)

        far_end = self._align_reference(near_end, far_end)
        _render, mu_scale = self._update_render_gates(far_end)
        # Main mu saturation gate matches the shadow's threshold, so the main filter does not learn on a clipped reference.

        _res_context = None  # populated when return_res_context=True and no internal RES

        # Stationary far-end feature for EPC, the RES baseline and the stationary-DT detector.
        far_pwr_global = _render.far_pwr

        if self.config.mode in _FREQ_MODES:
            raw_output, final_output, _res_context = self._process_frequency(near_end, far_end, mu_scale, far_pwr_global)
        else:
            raw_output, echo_est = self.filter.process_block(near_end, far_end,
                                                              mu_scale=mu_scale)
            final_output = raw_output.copy()
            self._update_simple_mu_ratio(raw_output, far_end)

        just_converged = self._track_erle_convergence(near_end, far_end, raw_output, _res_context)

        self._update_state_diagnostics(just_converged)

        self._update_trajectory_diagnostics(near_end, raw_output, final_output)

        result = final_output.astype(np.float32)
        if _res_context is not None:
            return (result, _res_context)
        return result

    def get_diagnostics(self) -> dict:
        """Return the latest per-frame diagnostic dict."""
        return self._diag.copy()

    def get_formed_output(self) -> np.ndarray:
        """Return the selected, crossfaded, WOLA-formed linear output; requires ``return_formed_output=True`` and one processed hop."""
        if not self.config.return_formed_output:
            raise ValueError(
                "get_formed_output() requires config.return_formed_output=True"
            )
        if self._formed_output is None:
            raise ValueError("get_formed_output() called before process()")
        return self._formed_output

    def get_erle(self) -> float:
        """Return cumulative ERLE (full-segment average)."""
        eps = 1e-10
        if self.near_power_sum < eps and self.error_power_sum < eps:
            return 0.0
        return 10 * np.log10((self.near_power_sum + eps) / (self.error_power_sum + eps))

    def get_erle_instant(self) -> float:
        """Return instantaneous ERLE (EMA-smoothed)."""
        eps = 1e-10
        if self.near_power < eps and self.error_power < eps:
            return 0.0
        return 10 * np.log10((self.near_power + eps) / (self.error_power + eps))

    def linear_is_cancelling(self) -> bool:
        """True when the linear filter is demonstrably cancelling: windowed ERLE above 2.5 dB, or the
        recent inst-ERLE peak above ``delay_acquire_inst_erle_db``. It is the warm-transfer gate behind
        ``apply_external_realign`` and is public so wrappers can ask the same question of their lanes.
        False means nothing proves the filter works, not that it is broken (an unrun or reset instance answers False)."""
        _recent = list(self._erle_slope_buf)[-15:]
        return bool(
            float(self._diag.get('erle_windowed', 0.0)) > 2.5  # dB
            or (max(_recent) if _recent else 0.0)
            > float(getattr(self.config, 'delay_acquire_inst_erle_db', 4.0)))

    def dump_p53_trace(self, path: str) -> int:
        """Dump the captured per-frame innovation-audit rows to .npz; returns the frame count (0 if none or not PBFDKF)."""
        if not isinstance(self.filter, PBFDKF):
            return 0
        rows = getattr(self.filter, '_p53_innovation_trace', [])
        if not rows:
            return 0
        n_frames = len(rows)
        n_freqs = self.filter.n_freqs
        cols = {}
        for k in rows[0].keys():
            arr = np.empty((n_frames, n_freqs), dtype=np.float32)
            for i, r in enumerate(rows):
                arr[i] = r[k]
            cols[k] = arr
        os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
        np.savez(path, **cols)
        return n_frames

    def get_filter_state(self) -> AecFilterState:
        """Return the highest-priority active filter state (WARMUP, DIVERGED, EPC_RECOVERY, DT_ACTIVE, STATIONARY_FAR, CONVERGED, CONVERGING)."""
        if self._warmup_frames > 0:
            return AecFilterState.WARMUP
        if self._divergence_indicator > 0.6:
            return AecFilterState.DIVERGED
        if self.epc_active:
            return AecFilterState.EPC_RECOVERY
        if self._diag.get('dt_indicator', 0.0) > 0.5:
            return AecFilterState.DT_ACTIVE
        if self._render_activity.is_stationary:
            return AecFilterState.STATIONARY_FAR
        if self._filter_converged:
            return AecFilterState.CONVERGED
        return AecFilterState.CONVERGING

    def get_stats(self) -> AecStats:
        d = self._diag
        _db = lambda x, floor=1e-10: 10.0 * np.log10(max(x, floor))
        hop_s = self._hop_size / self.config.sample_rate
        delay_samples = int(self._current_delay) if self._current_delay >= 0 else 0
        delay_ms = delay_samples / self.config.sample_rate * 1000.0
        return AecStats(
            frame_count=self._frame_count,
            time_s=self._frame_count * hop_s,
            filter_state=self.get_filter_state(),
            filter_converged=self._filter_converged,
            filter_once_converged=self._filter_once_converged,
            warmup_remaining=self._warmup_frames,
            erle_inst_db=self.get_erle_instant(),
            erle_windowed_db=d.get('erle_windowed', 0.0),
            erle_cumulative_db=self.get_erle(),
            erl_db=_db(self._erl_estimate, 1e-6),
            divergence=self._divergence_indicator,
            epc_active=self.epc_active,
            epv_ratio=d.get('epv_gain_ratio', 1.0),
            cohort_tail_T=False,
            mu_scale=d.get('mu_scale', 1.0),
            filter_w_norm=d.get('filter_w_norm', 0.0),
            shadow_w_norm=d.get('shadow_w_norm', 0.0),
            dt_confidence=0.0,
            dt_from_energy=self._dt_from_energy,
            dt_from_shadow=self._dt_from_shadow,
            dt_from_coherence=d.get('dt_from_coherence', 0.0),
            dt_active=False,
            far_power_db=_db(self._far_power_ema),
            mic_power_db=_db(self._mic_power_ema),
            error_power_db=_db(self.error_power),
            far_activity=d.get('far_activity', 0.0),
            saturation_level=self._saturation_level,
            delay_samples=delay_samples,
            delay_ms=delay_ms,
            shadow_advantage=self._shadow_advantage,
            shadow_copy_count=self._regime_handler.copy_counter,
            main_paused=self._regime_handler.main_paused,
            res_gain_mean_db=_db(d.get('res_gain_mean', 1.0)),
            res_using_render=d.get('using_render_based', False),
            echo_psd_mean_db=_db(d.get('echo_psd_mean', 1e-10)),
            error_psd_mean_db=_db(d.get('error_psd_mean', 1e-10)),
        )

    def GetStats(self) -> AecStats:
        return self.get_stats()

    def _aec3_select_linear_filter_output(
        self, *, e_refined_time: np.ndarray, near_end_block: np.ndarray,
        e_coarse_time: Optional[np.ndarray] = None,
    ) -> tuple:
        """Select the linear-filter output (refined, coarse or the raw capture) and form the WOLA spectra for RES and SuppressionGain.

        Coarse is chosen when cleaner (e2_coarse < 0.9 e2_refined with enough capture and echo energy) or when the refined
        filter has diverged; the formed output crossfades over 30 samples on a change. Without a shadow filter
        ``e_coarse_time`` is omitted. Returns ``(selected_error_spec_windowed, selected_echo_spec)`` (complex64, n_freqs),
        which share one periodic sqrt-Hann frame, so their sum is the windowed capture spectrum.
        """
        if e_coarse_time is None:
            e_coarse_time = e_refined_time
        hop = int(near_end_block.shape[0])
        # Time-domain block energies; s_* are the echo estimates (capture minus residual).
        s_refined_time = near_end_block - e_refined_time
        s_coarse_time = near_end_block - e_coarse_time
        # float32 sums, so a near-threshold decision is deterministic.
        e2_refined = np.sum(e_refined_time * e_refined_time, dtype=np.float32)
        e2_coarse = np.sum(e_coarse_time * e_coarse_time, dtype=np.float32)
        y2 = np.sum(near_end_block * near_end_block, dtype=np.float32)
        s2_refined = np.sum(s_refined_time * s_refined_time, dtype=np.float32)
        s2_coarse = np.sum(s_coarse_time * s_coarse_time, dtype=np.float32)
        # Thresholds 30^2 and 60^2 per block sample, scaled to the hop on the float [-1, 1] scale.
        int16_scale_sq = np.float32(32768.0 ** 2)
        thr_30 = np.float32(30.0 ** 2 * hop) / int16_scale_sq
        thr_60 = np.float32(60.0 ** 2 * hop) / int16_scale_sq
        cond_coarse_cleaner = (
            e2_coarse < np.float32(0.9) * e2_refined
            and y2 > thr_30
            and (s2_refined > thr_60 or s2_coarse > thr_60)
        )
        cond_refined_diverged = e2_coarse < e2_refined and y2 < e2_refined
        use_refined = not (cond_coarse_cleaner or cond_refined_diverged)
        self._refined_filter_output_last_selected = bool(use_refined)
        # Form the output with a 30-sample transition between the previous and current selection.
        # Capture is the third candidate: when the quality analyzer has not declared the linear estimate
        # usable and its residual exceeds the capture, the capture is used (the verdict is as of the previous hop).
        e2_selected = e2_refined if use_refined else e2_coarse
        _selection = _SEL_REFINED if use_refined else _SEL_COARSE
        # Only where the post-stage over-output guard cannot reach (with enable_res that guard owns the rule).
        _capture_gate = (self.config.output_capture_when_linear_unusable
                         and not self.config.enable_res
                         and self.config.return_res_context)
        if (_capture_gate
                and not self._aec3_state.usable_linear_estimate()
                and e2_selected > y2):
            _selection = _SEL_CAPTURE
        _candidate = (e_coarse_time, e_refined_time, near_end_block)
        from_time = _candidate[self._form_last_selection]
        to_time = _candidate[_selection]
        _form_transition_active = (self._form_last_selection != _selection)
        if _form_transition_active:
            _kT = 30  # transition length
            _k = np.arange(_kT, dtype=np.float32) + 1.0
            _s = _k / (_kT + 1.0)
            e_form = np.empty(hop, dtype=np.float32)
            e_form[:_kT] = ((1.0 - _s) * from_time[:_kT]
                            + _s * to_time[:_kT])
            e_form[_kT:] = to_time[_kT:]
        else:
            e_form = to_time.copy()
        # Windowed FFT of [previous formed output | formed output]; zeros on the first frame.
        _e_old = (self._form_prev_output_time
                  if self._form_prev_output_time is not None
                  else np.zeros(hop, dtype=np.float32))
        _e_block = np.concatenate([_e_old, e_form])
        _e_block_win = _e_block * self.filter._sqrt_hann_analysis
        selected_esw = np.fft.rfft(
            _e_block_win, self.filter.fft_size
        ).astype(np.complex64)
        _near_spec_win = (
            self.filter.error_spec_windowed + self.filter.echo_spec
        ).astype(np.complex64)
        selected_echo_spec = (_near_spec_win - selected_esw).astype(np.complex64)
        self._form_prev_output_time = e_form
        self._form_last_selection = _selection
        return selected_esw, selected_echo_spec

    def _aec3_post(self, raw_output: np.ndarray, near_end: np.ndarray,
                   far_end: np.ndarray) -> np.ndarray:
        """Post-stage: AecState, ResidualEchoEstimator and SuppressionGain.

        Takes the linear-filter time-domain residual (hop_size samples) and returns the suppressed hop.
        The suppression gain is applied to the formed WOLA spectrum and synthesised with sqrt-Hann overlap-add.
        """
        from .filter import build_filter_state_bridge

        n_bins = self.filter.n_freqs
        # PSDs are scaled up to the int16 magnitude convention of the AEC3 thresholds; the gain is a ratio, so the scale cancels.
        _PSD_SCALE = (32768.0) ** 2  # int16 max^2
        _sel_echo_spec, near_psd, far_psd, echo_psd, error_spec, error_psd, _near_psd_win, _capture_psd_erle, far_pwr, render_block_scaled = self._post_prepare_spectra(raw_output, near_end, far_end, _PSD_SCALE)

        bridge = self._post_filter_bridge(raw_output, near_end, build_filter_state_bridge)
        ext_delay = self._post_external_delay()
        self._post_pending_changes()

        _filter_taps_full = self._post_filter_taps()
        _x2_reverb_for_erle = self._post_render_reverb(_PSD_SCALE)
        self._post_update_state(bridge, ext_delay, far_psd, near_psd, error_psd, echo_psd, far_pwr, render_block_scaled, _filter_taps_full, _x2_reverb_for_erle, _capture_psd_erle)

        r2, r2_unb, _filter_converged_enough, _effective_usable_linear = self._post_estimate_residual(far_psd, near_psd, echo_psd)
        r2, r2_unb, _stationary_mask = self._post_stationary_residual(r2, r2_unb, _filter_converged_enough)

        r2 = self._post_observer_residual(near_end, error_spec, _sel_echo_spec, _PSD_SCALE, _stationary_mask, r2)

        nearend_pwr = self._post_nearend_power(error_psd, near_psd)
        comfort_noise = self._post_comfort_noise(near_psd, _near_psd_win)
        gain = self._post_suppression_gain(nearend_pwr, r2, r2_unb, comfort_noise, render_block_scaled, _stationary_mask)

        # Frequency-domain seam for an external post-NR residual suppressor: stash the WOLA error / echo
        # spectra, gain and comfort noise so the caller can apply S(f) = error_spec * G_nr * res_gain (+ CNG).
        if self.config.return_res_context:
            self._post_stash_context(error_spec, _sel_echo_spec, gain, comfort_noise, r2)
            # With internal RES off, emit the linear residual so the external stage owns suppression.
            if not self.config.enable_res:
                return raw_output.astype(np.float32)

        e_out_spec = self._post_output_spectrum(error_spec, _sel_echo_spec, _effective_usable_linear, gain)

        e_out_spec = self._post_add_comfort_noise(e_out_spec, n_bins, _PSD_SCALE, comfort_noise, gain)

        return self._post_synthesize(e_out_spec)

    def enable_res_audit(self) -> None:
        """No-op."""

    def get_res_audit(self):
        """No-op; returns None."""
        return None


    def _init_config(self):
        # The signal grid must already be resolved by AecConfig.__post_init__ (as C requires).
        _grid = resolve_signal_grid(self.config.sample_rate,
                                    self.config.frame_size,
                                    hop_size=self.config.hop_size)
        if self.config.hop_size != _grid.hop_size:   # defensive, unreachable
            raise ValueError(
                f"AEC config is not resolved: hop_size={self.config.hop_size} "
                f"does not match the resolved grid {_grid}")

        if self.config.mu == AecConfig.mu:  # still at dataclass default
            self.config.mu = self._MODE_DEFAULT_MU.get(self.config.mode, 0.3)

        # Re-run the delay-mode translation at construction so a caller that sets ``enable_delay_est`` after
        # building the config is honoured (and an illegal combination fails here).
        self.config._resolve_delay_mode()

    def _init_delay(self):
        # ``delay_mode`` is the single source of truth (AecConfig.__post_init__ has folded in the deprecated
        # mirror and rejected illegal combinations).
        _delay_mode = self.config.delay_mode
        if _delay_mode is not AecDelayMode.EXTERNAL_ALIGNED:
            # Ring capacity from the mode-aware helper shared with the C port: MATCHED keeps the full search ring,
            # FIXED shrinks to ``fixed_delay_samples + hop``.
            buffer_samp = ref_ring_samples(self.config, self.config.hop_size)
            if _delay_mode is AecDelayMode.FIXED:
                self.delay_est = None
                self._current_delay = self.config.fixed_delay_samples
            else:
                # Matched-filter bank, delay estimator and clock-drift detector behind the DelayEstimator interface.
                self.delay_est = DelayEstimator(
                    sample_rate=self.config.sample_rate,
                    hop_size=self.config.hop_size,
                    # Matched-filter bank size (forwarded to EchoPathDelayEstimator, which sizes the ring and histograms from it).
                    num_filters=self.config.delay_num_filters,
                    max_delay_ms=self.config.max_delay_ms,
                    init_seconds=self.config.delay_est_init_s,
                    period_seconds=self.config.delay_est_period_s,
                    par_low_threshold=self.config.delay_par_low_threshold,
                    par_solid_threshold=self.config.delay_par_solid_threshold,
                    fast_path_enabled=True,
                    fast_par_threshold=40.0,
                )
                self._current_delay = -1  # -1 = not yet estimated
            self._ref_ring = np.zeros(buffer_samp, dtype=np.float32)
            self._ref_ring_write = 0
            self._ref_ring_size = len(self._ref_ring)
            self._ref_ring_filled = 0  # Total samples written (for warmup)
            self._delay_active = True
        else:
            # EXTERNAL_ALIGNED: no estimator or ring; ``ref`` is aligned to ``mic`` by the caller.
            self.delay_est = None
            self._delay_active = False
            # The applied delay is 0 by contract; get_stats() reads _current_delay unconditionally.
            self._current_delay = 0

    def _init_linear_filter(self):
        if self.config.mode in _FREQ_MODES:
            if self.config.mode == AecMode.FDAF:
                # Single FFT block: block_size = 2 x filter_length, hop = filter_length.
                desired = 2 * self.config.filter_length
                block_size = 256
                while block_size < desired:
                    block_size *= 2
                n_partitions = 1
                self._internal_hop = block_size // 2
            else:
                # Partitioned block (PBFDAF/PBFDKF/SUBBAND)
                # block_size = 2 x hop; the FFT size is next_pow2(block_size) inside PBFDAF.
                hop_size = self.config.hop_size
                block_size = 2 * hop_size
                n_partitions = max(1, (self.config.filter_length + hop_size - 1) // hop_size)
                self._internal_hop = hop_size

            if self.config.mode in (AecMode.PBFDKF, AecMode.SUBBAND):
                FilterClass = PBFDKF
            elif self.config.mode == AecMode.PBFDAF:
                FilterClass = PBFDAF
            else:  # FDAF — use_kalman flag for flexibility
                FilterClass = PBFDKF if self.config.use_kalman else PBFDAF
            self.filter = FilterClass(
                block_size=block_size,
                n_partitions=n_partitions,
                mu=self.config.mu,
                delta=self.config.delta,
                hop_size=self._internal_hop,
                sample_rate=self.config.sample_rate
            )
            self.filter.enable_td_constraint = self.config.enable_td_constraint
            self.filter._constraint_round_robin = self.config.constraint_round_robin
            self._hop_size = self.config.hop_size
            self._n_partitions = n_partitions

            if isinstance(self.filter, PBFDKF):
                self.filter.Q_high[:] = self.config.kalman_q_high
                self.filter.Q_low[:]  = self.config.kalman_q_low
                self.filter.Q[:] = self.config.kalman_q_high
                # Cold-start deadlock breaker (0 = off).
                self.filter._h_error_refresh_erl_floor = np.float32(
                    getattr(self.config, "h_error_refresh_erl_floor", 0.0)
                )
                _hef_ovr = float(getattr(self.config, "h_error_floor_override", 0.0))
                if _hef_ovr > 0.0:
                    self.filter._h_error_floor = np.float32(_hef_ovr)

            if self.config.mode == AecMode.FDAF and self._internal_hop > self._hop_size:
                if self._internal_hop % self._hop_size != 0:
                    raise ValueError(
                        f"FDAF internal_hop ({self._internal_hop}) must be an integer multiple "
                        f"of hop_size ({self._hop_size}); adjust filter_length so that "
                        f"next_pow2(2*filter_length)//2 is divisible by hop_size"
                    )
                # Buffer large enough for accumulation (internal_hop + one extra external hop)
                buf_size = self._internal_hop + self._hop_size
                self._freq_near_queue = np.zeros(buf_size, dtype=np.float32)
                self._freq_far_queue = np.zeros(buf_size, dtype=np.float32)
                self._freq_out_buf = np.zeros(buf_size, dtype=np.float32)
                self._freq_out_valid = 0  # valid output samples remaining
                self._freq_out_read = 0
                self._freq_queue_write = 0
            else:
                self._freq_near_queue = None
        else:
            raise ValueError(f"unsupported AEC mode: {self.config.mode!r}")
        return block_size, n_partitions

    def _init_suppression_chain(self):
        from .state import AecState as _Aec3State, AecStateConfig as _Aec3StateConfig
        from .residual import ResidualEchoEstimator, SuppressionGain
        from .residual.suppression_gain import SuppressorConfig
        n_bins = int(self.filter.n_freqs)
        self._aec3_state = _Aec3State(_Aec3StateConfig(
            n_bins=n_bins,
            enable_transparent_mode=False,
            enable_filter_analyzer=True,
            hop_size=int(self.config.hop_size),
            sample_rate=int(self.config.sample_rate),
        ))
        # FFT-density-scaled PSD floors: AEC3's per-bin floors assume kFftLengthBy2 = 64 and are rescaled
        # by fft_size / 2 / 64 (a single-constant approximation); the low-noise render block-energy
        # threshold is scaled by hop / 64.
        import dataclasses as _dc
        from .residual.residual_echo_estimator import EchoModelConfig
        from .residual.suppression_gain import EchoAudibilityConfig
        _ff_for_floors = float(self.filter.fft_size) if hasattr(self.filter, "fft_size") else 512.0
        from . import aec3_scale as _aec3_scale_floors
        _echo_model_cfg = EchoModelConfig(
            min_noise_floor_power=_aec3_scale_floors.fft_density_scale(
                EchoModelConfig().min_noise_floor_power, int(_ff_for_floors)
            ),
        )
        self._aec3_ree = ResidualEchoEstimator(
            n_bins=n_bins,
            echo_model=_echo_model_cfg,
            sr=self.config.sample_rate,
            hop_size=self.config.hop_size,
            use_aec3_residual_noise_gate=True,
            use_aec3_echo_gen_window=True,
            use_aec3_wallclock_reverb_smoothing=True,
            nl_r2_enabled=bool(getattr(self.config, "nl_r2_enabled", False)),
            nl_r2_alpha=float(getattr(self.config, "nl_r2_alpha", 0.1)),
            nl_r2_cap_multiple=float(
                getattr(self.config, "nl_r2_cap_multiple", 0.0)
            ),
            use_stationarity_properties=True,
        )
        self._aec3_ree._reverb_tail_strength = float(
            getattr(self.config, "reverb_tail_strength", 1.0))
        _sg_config = SuppressorConfig()
        # Stationarity zeroing is on in production: override the default EchoAudibilityConfig so it fires;
        # the audibility floors are FFT-density-scaled like the echo-model floors.
        _ea_default = EchoAudibilityConfig()
        _sg_config.echo_audibility = _dc.replace(
            _sg_config.echo_audibility,
            use_stationarity_properties=True,
            floor_power=_aec3_scale_floors.fft_density_scale(
                _ea_default.floor_power, int(_ff_for_floors)
            ),
            low_render_limit=_aec3_scale_floors.fft_density_scale(
                _ea_default.low_render_limit, int(_ff_for_floors)
            ),
            normal_render_limit=_aec3_scale_floors.fft_density_scale(
                _ea_default.normal_render_limit, int(_ff_for_floors)
            ),
        )
        _sg_config.dominant_nearend_detection = _dc.replace(
            _sg_config.dominant_nearend_detection,
            use_wallclock_trigger_threshold=True,
        )
        self._aec3_sg = SuppressionGain(**self._build_sg_kwargs(n_bins, _sg_config))
        self._aec3_n_bins = n_bins
        self._aec3_sg_config = _sg_config
        return n_bins

    def _init_post_synthesis(self, n_bins):
        # Dedicated ReverbModel for the ERLE estimators: steady (non-mild) decay, fed from X^2[delay + 1];
        # output = X^2[delay] + tail.
        from .residual.reverb_model import ReverbModel as _AvgReverbModel
        self._aec3_avg_render_reverb = _AvgReverbModel(n_bins=n_bins)
        # Synthesis window: periodic sqrt-Hann (denominator N); analysis x synthesis = Hann sums to 1 across 50 % overlap.
        bs = int(self.filter.block_size)
        _idx = np.arange(bs, dtype=np.float64)
        self._aec3_synth_window = np.sqrt(
            0.5 * (1.0 - np.cos(2.0 * np.pi * _idx / float(bs)))
        ).astype(np.float32)
        self._aec3_ola_buf = np.zeros(bs, dtype=np.float32)
        # Comfort-noise generator state, all in int16^2 PSD scale: Y2_smoothed (per-bin Y^2 EMA), N2 (background
        # noise estimate, init 1e6), N2_initial (transient estimate over the first 1000 blocks), N2_counter,
        # the noise-floor factor and the LCG seed (42). Per-frame constants are rescaled from the 4 ms block
        # reference to the hop via the aec3_scale helpers.
        from .aec3_scale import (
            blocks_to_hops as _blocks_to_hops,
            per_block_ema_alpha_to_per_hop as _ema_to_hop,
            per_block_growth_to_per_hop as _growth_to_hop,
        )
        _hop = int(self.config.hop_size)
        _sr = int(self.config.sample_rate)
        self._aec3_y2_smoothed = np.zeros(n_bins, dtype=np.float32)
        self._aec3_n2 = np.full(n_bins, 1.0e6, dtype=np.float32)
        self._aec3_n2_initial = np.zeros(n_bins, dtype=np.float32)
        self._aec3_n2_counter = 0
        _dbfs = float(self.config.comfort_noise_floor_dbfs)
        # The noise-floor factor is fft_size / 2 (256 at fft 512).
        _ff = float(self.filter.fft_size) if hasattr(self.filter, "fft_size") else 512.0
        _cn_floor_factor = _ff / 2.0
        self._aec3_noise_floor_int16sq = float(
            _cn_floor_factor * (10.0 ** ((90.30899869919436 + _dbfs) * 0.1))
        )
        self._aec3_cng_seed = 42
        self._aec3_noise_initialized = False
        # Rescaled constants (at hop 160 / 16 kHz): Y2 alpha 0.1 -> 0.232, N2 track-down freshness 0.9 -> 0.997,
        # N2 slow-up 1.0002 -> 1.0005, N2_initial alpha 0.001 -> 0.0025, N2 update onset 50 blocks -> 20 hops,
        # N2_initial transient 1000 blocks -> 400 hops.
        self._aec3_cng_y2_alpha = float(_ema_to_hop(0.1, _hop, _sr))
        self._aec3_cng_n2_track_freshness = float(_ema_to_hop(0.9, _hop, _sr))
        self._aec3_cng_n2_track_retention = 1.0 - self._aec3_cng_n2_track_freshness
        self._aec3_cng_n2_slow_up = float(_growth_to_hop(1.0002, _hop, _sr))
        self._aec3_cng_n2_initial_alpha = float(_ema_to_hop(0.001, _hop, _sr))
        self._aec3_cng_n2_update_onset_hops = int(_blocks_to_hops(50, _hop, _sr))
        self._aec3_cng_n2_initial_duration_hops = int(_blocks_to_hops(1000, _hop, _sr))
        # sqrt(2) * sin(2 pi i / 32) table: the sqrt(2) compensates the overlap-add power loss of uncorrelated CN frames.
        _lut_idx = np.arange(32, dtype=np.float64)
        self._aec3_sqrt2_sin_lut = (
            np.sqrt(2.0) * np.sin(2.0 * np.pi * _lut_idx / 32.0)
        ).astype(np.float32)
        pass
        return _hop

    def _init_post_state(self):
        # ``res`` stays None (the post-filter chain is the AEC3-style state / residual / suppression stack).
        self.res = None

        # Post-stage chain: AecState, ResidualEchoEstimator, SuppressionGain and CNG consume the filter's error spectrum.
        self._aec3_state = None
        self._aec3_ree = None
        self._aec3_sg = None
        self._aec3_ola_buf = None
        self._aec3_synth_window = None
        # Pending EchoPathVariability from the event detectors, consumed at the next _aec3_post call before aec_state.update().
        self._aec3_pending_gain_change = False
        self._aec3_pending_delay_change = None  # None = no event; else DelayAdjustment
        # True-delay transient leakage: counter and cached steady values.
        self._q1_tdt_rem = 0
        self._q1_tdt_lc_steady = float(self.filter._leakage_converged) if self.filter is not None else 2.5e-3
        self._q1_tdt_ld_steady = float(self.filter._leakage_diverged) if self.filter is not None else 2.5e-1
        if self.filter is not None:
            n_bins = self._init_suppression_chain()
            _hop = self._init_post_synthesis(n_bins)
        return _hop

    def _init_render_analysis(self, _hop):
        # StationarityEstimator: per-bin stationary render (hum / fan / line noise). It zeros R^2 on stationary
        # bands in _aec3_post and makes the filter skip the W update on stationary blocks.
        from .state.stationarity_estimator import StationarityEstimator as _StatEst
        if self.filter is not None and hasattr(self.filter, 'n_freqs'):
            self._aec3_stationarity = _StatEst(
                n_freqs=int(self.filter.n_freqs),
                hop_samples=int(self.config.hop_size),
                sample_rate=int(self.config.sample_rate),
            )
        else:
            self._aec3_stationarity = None
        # Render peak floor: 10 in int16 amplitude.
        self._aec3_non_zero_render_seen = False
        self._AEC3_RENDER_PEAK_FLOOR = 10.0 / 32768.0
        # Filter-has-had-time-to-converge: 0.8 s of strong, unsaturated render.
        self._aec3_stationarity_active_hops = 0
        _hop_st = int(self.config.hop_size)
        _sr_st = int(self.config.sample_rate)
        self._aec3_stationarity_converge_hops = int(round(0.8 * _sr_st / _hop_st))

        # RenderSignalAnalyzer: per-bin narrow-band mask for the filter gain and a poor-excitation flag that freezes the W update.
        from .render.render_signal_analyzer import RenderSignalAnalyzer as _RSA
        if hasattr(self.filter, 'n_freqs') and hasattr(self.filter, 'n_partitions'):
            _rsa_kwargs = {}
            if getattr(self.config, 'rsa_wallclock_persistence', False):
                from . import aec3_scale as _aec3_scale_rsa
                _hop, _sr = self.config.hop_size, self.config.sample_rate
                # threshold = hops-to-fire - 1 (fires on the hop after the hold elapses).
                _rsa_kwargs['counter_threshold_hops'] = max(0, _aec3_scale_rsa.ms_to_hops(
                    self.config.rsa_mask_hold_ms, _hop, _sr) - 1)
                _rsa_kwargs['poor_excitation_threshold_hops'] = max(0, _aec3_scale_rsa.ms_to_hops(
                    self.config.rsa_poor_excitation_hold_ms, _hop, _sr) - 1)
            self._render_signal_analyzer = _RSA(
                n_freqs=self.filter.n_freqs,
                strong_peak_freeze_duration=self.filter.n_partitions,
                **_rsa_kwargs,
            )
            # Wire the analyzer into the refined filter; the startup counter is seeded by _seed_poor_excitation_counters().
            self.filter._render_signal_analyzer = self._render_signal_analyzer
        else:
            self._render_signal_analyzer = None
        return _hop

    def _init_shadow_filter(self):
        # Shadow filter (dual-filter, frequency-domain modes only).
        self.shadow_filter = None
        self.main_err_smooth = 0.0
        self.shadow_err_smooth = 0.0
        if (self.config.enable_shadow and
                self.config.mode in _FREQ_MODES
                and hasattr(self.filter, 'W')):
            # Shadow is always PBFDAF NLMS (the coarse-filter role).
            ShadowClass = PBFDAF
            shadow_mu = self.config.shadow_mu_nlms
            self.shadow_filter = ShadowClass(
                block_size=self.filter.block_size,
                n_partitions=self.filter.n_partitions,
                mu=shadow_mu,
                delta=self.config.delta,
                hop_size=self.filter.hop_size,
                sample_rate=self.config.sample_rate
            )
            self.shadow_filter.enable_td_constraint = self.config.enable_td_constraint
            self.shadow_filter._constraint_round_robin = self.config.constraint_round_robin
            # The shadow's near_spec / error_spec_windowed are never read, so skip those two FFTs per hop.
            self.shadow_filter._lightweight = True
            # The analyzer is shared with the refined filter; its startup counter is seeded below.
            if self._render_signal_analyzer is not None:
                self.shadow_filter._render_signal_analyzer = self._render_signal_analyzer
            self.shadow_filter._saturated_capture = False
        self._seed_poor_excitation_counters()

    def _init_path_tracking(self):
        # Echo-path-change detector; hop_size / sample_rate retime its EPV constants.
        self._epc_det = EchoPathChangeDetector(
            self.config, hop_size=self.config.hop_size,
            sample_rate=self.config.sample_rate)

        self._misadjustment_stable_count = 0
        self._misadjustment_hangover_remaining = 0

        # FilterAnalyzer ownership lives on AecState (``self._aec3_state``).
        self._epc_reset_fired_this_frame = False

        self._convergence = FilterConvergenceAnalyzer(
            self.config.hop_size, self.config.sample_rate)
        self._epc_render_forced_remaining = 0
        self._erl_estimate = 0.1  # initial -20 dB, conservative
        # ERL tracker retention EMAs, authored at the 10 ms grid and retimed.
        from . import aec3_scale as _aec3s
        self._alpha_erl_tracking = _aec3s.growth_rehop(
            0.99, 160, 16000, self.config.hop_size, self.config.sample_rate)
        self._alpha_erl_converged = _aec3s.growth_rehop(
            0.999, 160, 16000, self.config.hop_size, self.config.sample_rate)
        # Per-band ERL broadcast to filter._erl_per_bin; stays uniform at 0.1.
        self._per_band_erl = np.array([0.1, 0.1, 0.1], dtype=np.float64)
        self._dt_analyzer = DoubleTalkAnalyzer(self.config)
        from . import aec3_scale as _aec3_scale_dtp
        self._dt_protect_extra_hold_hops = _aec3_scale_dtp.ms_to_hops(
            getattr(self.config, 'dt_protect_extra_hold_ms', 0.0),
            self.config.hop_size, self.config.sample_rate)
        self._dt_protect_confirm_hops = _aec3_scale_dtp.ms_to_hops(
            getattr(self.config, 'dt_protect_confirm_ms', 0.0),
            self.config.hop_size, self.config.sample_rate)
        return _aec3_scale_dtp

    def _init_observer(self, block_size, n_partitions, _aec3_scale_dtp):
        # Live low-frequency observer: a second plain-NLMS PBFDAF with the main filter's partition count, fed the
        # far-end ring live_observer_shift_ms earlier than the main read. Only echo_spec is read, so the two
        # unused FFTs are skipped. Constructed unconditionally; only the per-hop call sites are gated on live_observer_enabled.
        self._observer_filter = PBFDAF(
            block_size=block_size,
            n_partitions=n_partitions,
            mu=float(getattr(self.config, 'live_observer_mu', 0.3)),
            delta=self.config.delta,
            hop_size=self._internal_hop,
            sample_rate=self.config.sample_rate,
        )
        self._observer_filter._lightweight = True
        self._observer_filter._constraint_round_robin = (
            self.config.constraint_round_robin)
        self._observer_shift_hops = _aec3_scale_dtp.ms_to_hops(
            getattr(self.config, 'live_observer_shift_ms', 24.0),
            self.config.hop_size, self.config.sample_rate)
        _lo_bin_hz = self.config.sample_rate / float(self.config.frame_size)
        self._observer_max_bin = int(np.ceil(
            getattr(self.config, 'live_observer_max_hz', 1000.0) / _lo_bin_hz))
        # Injection-guard EMA state: the observer is corroborated against the main filter's residual.
        # Its excitation counter is seeded right after construction.
        _lo_hop_ms = 1000.0 * self.config.hop_size / self.config.sample_rate
        _lo_corrob_ms = max(1e-6, getattr(self.config, 'live_observer_corrob_ms', 100.0))
        self._observer_err_ema_alpha = _aec3_scale_dtp.ema_alpha_from_tau_ms(
            _lo_corrob_ms, _lo_hop_ms)
        self._observer_corrob_margin = _aec3_scale_dtp.db_to_linear(
            getattr(self.config, 'live_observer_corrob_margin_db', 0.0))
        self._observer_obs_err_ema = 0.0
        self._observer_main_err_ema = 0.0
        # Injection-only near-end discriminator.
        self._observer_inject_dt_energy_max = float(
            getattr(self.config, 'live_observer_inject_dt_energy_max', 0.5))
        # Defined for every delay mode: the per-hop assignment only runs when ``_delay_active`` (not in EXTERNAL_ALIGNED).
        self._observer_far_end = None
        self._seed_poor_excitation_counters()

    def _init_trace_state(self, _hop):
        # Mini trace state: post-reset age and an inst-ERLE ring for slope estimation (~500 ms).
        self._post_reset_age_frames = 0
        _slope_n = max(2, int(0.5 * self.config.sample_rate / max(1, _hop)))
        self._erle_slope_buf = deque(maxlen=_slope_n)
        self._p3f_refined_latched = False  # latches once refined_usable is seen
        self._p3f_main_err_baseline = 0.0  # EMA baseline for jump detection
        self._p3f_diverged_streak = 0  # consecutive frames over the diverged threshold
        # Windowed decaying ERLE accumulator for erle_factor (time constant 1000 hops).
        self._erle_window_near = 1e-10
        self._erle_window_err = 1e-10
        self._erle_factor_prev = 0.0  # previous erle_factor, for the shadow DTD weight

        self._reset_coherence_state()

        # Smoothed inst ERLE for the dt_indicator correction (about 3 frames).
        self._inst_erle_smooth = 1.0

        # Per-bin mu_scale from the post-RES echo / error PSD of the previous frame.
        self._per_bin_mu_scale = None  # None = scalar fallback

        self._diag = {
            'erle_inst': 0.0, 'mu_scale': 1.0, 'far_activity': 0.0,
            'res_gain_mean': 1.0, 'res_gain_min': 1.0, 'effective_g_min': 1.0,
            'converged': False, 'erle_factor': 0.0,
            'echo_psd_mean': 0.0, 'error_psd_mean': 0.0, 'divergence': 0.0,
            'using_render_based': False,
            'shadow_advantage': 1.0,
            'dt_from_energy': 0.0,
            'dt_from_shadow': 0.0,
            'erl_estimate': 0.1,
            'epc_active': False,
            'saturation_level': 0.0,
            'erle_windowed': 0.0,
            # DT / filter debug fields (dt_indicator is telemetry; the RES floor decision is AecResContext.res_floor_protect).
            'dt_indicator': 0.0,
            'main_err_smooth': 0.0,     # main filter error EMA
            'shadow_err_smooth': 0.0,   # shadow filter error EMA
            'main_paused': False,       # True = main filter weights frozen this frame
            'epv_gain_ratio': 1.0,      # EPV fast/slow gain ratio (!=1 → gain change)
            'dt_residual_scale': 1.0,   # constant 1.0
            'filter_w_norm': 0.0,       # main filter L2 weight norm
            'shadow_w_norm': 0.0,       # shadow filter L2 weight norm
            'copy_err_baseline': 1e-6,  # copy gate error baseline
        }

        # Shadow/main-selected, crossfaded, WOLA-formed linear output (distinct from the raw output of process()).
        self._formed_output = None

    def _init_preprocessing(self):
        if self.config.enable_highpass:
            self._hp_mic = HighPassFilter(self.config.highpass_cutoff_hz, self.config.sample_rate)
        else:
            self._hp_mic = None
        self._hp_ref = None

        if self.config.enable_saturation_detect:
            self._sat_detector_ref = SaturationDetector(
                self.config.saturation_threshold,
                self.config.hop_size, self.config.sample_rate)
            self._sat_detector_mic = SaturationDetector(
                self.config.saturation_threshold,
                self.config.hop_size, self.config.sample_rate)
        else:
            self._sat_detector_ref = None
            self._sat_detector_mic = None
        self._saturation_level = 0.0

        self._render_activity = RenderActivityDetector(
            self.config.hop_size, self.config.sample_rate)
        self._wn_err_baseline = 1e-8
        self._stat_dt_hangover = 0  # stationary double-talk hold-off counter (frames)

        # Simple variable mu: the holdoff and three alphas form one state machine and stay hop-authored.
        self._simple_mu_ratio = 1.0

    def _init_output_state(self):
        self._simple_mu_holdoff = 0  # blocks release for N frames
        self._warmup_frames = self.config.warmup_frames
        self._warmup_far_active = False  # consume warmup only while the far-end is active

        # Shadow divergence detection state (pause and Q boost, no output switch).
        self.shadow_frame_count = 0
        self._regime_handler = PathChangeRegimeHandler(
            self.config,
            gate_mode='energy',
        )
        self._last_raw_output: Optional[np.ndarray] = None  # raw filter output before RES (diagnostic)

        # Coarse-filter output cache for ``_aec3_post``, and the refined-output selection latch.
        self._last_shadow_output_time: Optional[np.ndarray] = None
        self._refined_filter_output_last_selected: bool = True
        # Formed-output crossfade state: previous formed block (zeros on first use) and previous selection.
        self._form_prev_output_time: Optional[np.ndarray] = None
        # Which candidate the previous hop published (see _SEL_*).
        self._form_last_selection: int = _SEL_REFINED

        # Misadjustment estimator accumulators.
        self._aec3_misadj_e2_acum: float = 0.0
        self._aec3_misadj_y2_acum: float = 0.0
        self._aec3_misadj_n_acum: int = 0
        self._aec3_misadj_inv: float = 0.0
        self._aec3_misadj_overhang: int = 0

        # EPV EMAs live in EchoPathChangeDetector (``self._epc_det``); convergence and EPC state are read through properties.
        self._far_power_ema = 0.0  # for GetStats()
        self._mic_power_ema = 0.0
        self._frame_count = 0  # frames since reset()

        self.near_power = 0.0
        self.error_power = 0.0
        self.raw_error_power = 0.0
        # Per-sample ERLE power EMA, applied per sample and hop-invariant, so retimed off the sample rate: 0.95^(16000/sr).
        self.alpha = 0.95 ** (16000.0 / self.config.sample_rate)
        self.near_power_sum = 0.0
        self.error_power_sum = 0.0
        self.raw_error_power_sum = 0.0

    def _prepare_capture(self, near_end, far_end):
        self._epc_reset_fired_this_frame = False

        # Held "near-end seen recently" gate for DT-aware soft recovery. dt_from_energy alone is used (low in far-end
        # single-talk; dt_from_shadow false-fires on far-end path changes). The indicator must stay above the threshold
        # for ``sustain`` frames before arming, so 1-2 frame energy transients do not arm it.
        _ne_thr = float(getattr(self.config, 'ne_recent_threshold', 0.3))
        _ne_hold = int(getattr(self.config, 'ne_recent_hold', 150))
        _ne_sustain = int(getattr(self.config, 'ne_recent_sustain', 3))
        _ne_ind = float(getattr(self, '_dt_from_energy', 0.0))
        if _ne_ind > _ne_thr:
            self._ne_above = getattr(self, '_ne_above', 0) + 1
        else:
            self._ne_above = 0
        if self._ne_above >= _ne_sustain:
            self._ne_recent_frames = _ne_hold
        elif getattr(self, '_ne_recent_frames', 0) > 0:
            self._ne_recent_frames -= 1
        else:
            self._ne_recent_frames = 0

        if self._hp_mic is not None:
            near_end = self._hp_mic.process(near_end.copy())

        if self._sat_detector_ref is not None:
            sat_ref = self._sat_detector_ref.detect(far_end)
            sat_mic = self._sat_detector_mic.detect(near_end)
            self._saturation_level = max(sat_ref, sat_mic * 0.5)
            if self.config.saturation_softclip_ref and sat_ref > 0.1:
                far_end = SaturationDetector.soft_clip(far_end.copy())
        return near_end, far_end

    def _judge_delay_shift(self):
        # Judge any pending post-shift window once per hop, before Path A/B run. A confirmed-bad relock's candidate
        # is remembered in _delay_reject_values until reset(), since the aggregator histogram cannot un-learn a
        # favoured candidate.
        if getattr(self, '_delay_shift_judgment', None) is not None:
            (_pre_delay, _pre_erle, _hops_left,
             _pre_W, _pre_shadow_W,
             _pre_H_error, _pre_shadow_H_error) = self._delay_shift_judgment
            _hops_left -= 1
            if _hops_left <= 0:
                _cur_erle = float(self._diag.get('erle_windowed', 0.0))
                # Windowed ERLE alone cannot distinguish "worse" from "no echo estimate yet" (without RES the echo
                # estimate is exactly zero, so raw_error_power == near_power): require a measurably non-zero estimate
                # before trusting the dB reading; otherwise keep the new delay.
                _has_estimate = (self.raw_error_power
                                 < self.near_power * (1.0 - 1e-9))
                if _has_estimate and _cur_erle < _pre_erle - 1.0:
                    _bad_delay = self._current_delay
                    # Soft revert to the trusted pre-shift state, not the hard-reset chain (see _apply_delay_shift_soft_revert).
                    self._apply_delay_shift_soft_revert(
                        _pre_delay, _bad_delay,
                        _pre_W, _pre_shadow_W,
                        _pre_H_error, _pre_shadow_H_error)
                    if not hasattr(self, '_delay_reject_values'):
                        self._delay_reject_values = []
                    self._delay_reject_values.append(_bad_delay)
                self._delay_shift_judgment = None
            else:
                self._delay_shift_judgment = (
                    _pre_delay, _pre_erle, _hops_left, _pre_W, _pre_shadow_W,
                    _pre_H_error, _pre_shadow_H_error)

    def _acquire_delay(self, new_delay, _delay_eligible):
        # Path A: first delay acquisition (heavy action: resets the filter taps and derived state; needs a solid PAR).
        # Guard: if the filter already cancels at the current alignment, the "solid" estimate is a spurious large lag, so
        # the reset is skipped. Windowed ERLE is the signal because it decays when a real path change stops cancellation.
        _already_cancelling = (
            bool(getattr(self.config, 'delay_acquire_protect_converged', False))
            and float(self._diag.get('erle_windowed', 0.0)) > 2.5)  # dB
        if (_delay_eligible
                and self._current_delay < 0
                and self.delay_est.is_solid
                and not _already_cancelling):
            self._current_delay = new_delay
            # Warm transfer only when the filter was cancelling (inst-ERLE ring peak) and the delay fits the tap reach;
            # outside that, shifting by more than the IR length just zeros the taps without the derived-state reset.
            # The ring alone is used (not linear_is_cancelling()): this site fires on the estimator's own first acquisition,
            # which is far-active by construction.
            _warm_ok = False
            if getattr(self.config, 'delay_acquire_warm_transfer', False) \
                    and self.filter is not None:
                _wpk = max(list(self._erle_slope_buf)[-15:] or [0.0])
                _reach = self.filter.tap_span
                _warm_ok = (_wpk > float(getattr(
                    self.config, 'delay_acquire_inst_erle_db', 4.0))
                    and 0 < int(new_delay) < _reach)
            if _warm_ok:
                # Shift the learned IR by the acquired delay instead of zeroing, so the cancellation survives the realign.
                for _filt in (self.filter, self.shadow_filter):
                    if _filt is not None and hasattr(_filt, 'warm_shift_ir'):
                        _filt.warm_shift_ir(int(new_delay))
            elif self.config.dt_aware_recovery_soft and self._ne_recent_frames > 0:
                # Soft acquisition: apply the alignment and re-arm the excitation counters only; keep the taps and
                # convergence, and re-adapt at the normal step size (avoids an aggressive recovery that overfits near-end).
                self._rearm_filters_for_delay_change()
            else:
                # Reset the filter taps and all filter-output-derived state, which was learned against a misaligned reference.
                self._reset_filter_derived_state(reason='delay_first',
                                                 preserve_render_ema=True)
                self._maybe_mark_diverged('delay_first')
                # Full delay-change chain: H_error reset and counter reset on refined and shadow, AecState._full_reset.
                self._rearm_filters_for_delay_change()
                from .delay.delay_types import DelayAdjustment as _DA
                self._aec3_pending_delay_change = _DA.NEW_DETECTED_DELAY

    def _accept_delay_shift(self, new_delay):
        old_delay_before_shift = self._current_delay
        _pre_shift_erle = float(self._diag.get('erle_windowed', 0.0))
        # Snapshot the taps so a confirmed-bad relock can restore them exactly.
        _pre_shift_W = self.filter.W.copy()
        _pre_shift_shadow_W = (self.shadow_filter.W.copy()
                               if self.shadow_filter is not None
                               else None)
        # H_error_per_bin (the Kalman step-size driver) is a statistic of the taps, so a revert restores it with W.
        _pre_shift_H_error = (self.filter.H_error_per_bin.copy()
                              if hasattr(self.filter, 'H_error_per_bin')
                              else None)
        _pre_shift_shadow_H_error = (
            self.shadow_filter.H_error_per_bin.copy()
            if (self.shadow_filter is not None
                and hasattr(self.shadow_filter, 'H_error_per_bin'))
            else None)
        self._current_delay = new_delay
        del self._pending_delay
        if hasattr(self, '_pending_delay_ttl'):
            del self._pending_delay_ttl
        if (self.config.dt_aware_recovery_soft
                and self._ne_recent_frames > 0):
            # Soft realign: keep the converged filter and shift its IR by the delta the ring read just applied
            # (instead of letting the mismatch close at the unboosted rate, which can overshoot on loud far-end bursts);
            # no tap wipe, Kalman P-override, q-boost or mark_diverged.
            _soft_delta = int(new_delay) - int(old_delay_before_shift)
            if abs(_soft_delta) < self.filter.tap_span:
                for _filt in (self.filter, self.shadow_filter):
                    if _filt is not None and hasattr(_filt, 'warm_shift_ir'):
                        _filt.warm_shift_ir(_soft_delta)
            self._epc_reset_fired_this_frame = True
        else:
            # The taps were trained at the old alignment: treat the shift like Path A.
            self._apply_delay_shift_hard_reset(new_delay)
        # Bad-relock judgment (both branches): the matched-filter aggregator accepts the best-correlating lag
        # without checking that it cancels better. After _DELAY_SHIFT_JUDGMENT_HOPS, if windowed ERLE is more
        # than 1 dB worse than before the shift, the delay and tap / H_error snapshot are soft-reverted and the
        # same candidate is refused for the rest of the instance's life (one correction attempt, no retry loop).
        # A new shift inside a pending judgment window overwrites the pending judgment.
        _judgment_hops = _DELAY_SHIFT_JUDGMENT_HOPS
        self._delay_shift_judgment = (
            old_delay_before_shift, _pre_shift_erle,
            _judgment_hops, _pre_shift_W, _pre_shift_shadow_W,
            _pre_shift_H_error, _pre_shift_shadow_H_error)

    def _track_delay_shift(self, new_delay, _delay_eligible):
        # Age out _pending_delay so a stale value cannot pair with a later estimate (decremented per estimation cycle).
        if hasattr(self, '_pending_delay_ttl'):
            self._pending_delay_ttl -= 1
            if self._pending_delay_ttl <= 0:
                if hasattr(self, '_pending_delay'):
                    del self._pending_delay
                del self._pending_delay_ttl

        # Path B: delay shift, independent of Path A. Needs confidence >= 0.5 and two consistent estimates, so
        # single-frame noise cannot trigger a force_delay() EPC chain.
        if (_delay_eligible
                and self._current_delay >= 0
                and self.delay_est.confidence >= 0.5
                and abs(new_delay - self._current_delay) > 32
                and not any(abs(new_delay - _rv) < 16 for _rv in
                            getattr(self, '_delay_reject_values', ()))):
            if (hasattr(self, '_pending_delay')
                    and abs(new_delay - self._pending_delay) < 16):
                self._accept_delay_shift(new_delay)
            else:
                self._pending_delay = new_delay
                self._pending_delay_ttl = 3

    def _update_delay_estimate(self, near_end, far_end):
        # Online delay estimation in two independent paths, so a solid shift is not swallowed by the first-acquisition gate.
        if (self.delay_est is not None
                and self._delay_active):
            self.delay_est.accumulate(near_end, far_end)
            new_delay = self.delay_est.estimated_delay
            _delay_eligible = (new_delay >= 0
                                and self.delay_est._n_updates >= 3)

            self._judge_delay_shift()

            self._acquire_delay(new_delay, _delay_eligible)

            self._track_delay_shift(new_delay, _delay_eligible)

    def _align_reference(self, near_end, far_end):
        if self._delay_active:
            hop = len(far_end)

            self._update_delay_estimate(near_end, far_end)

            # Reset every hop; only the success path below (live observer enabled, delay > 0, ring filled) sets a real array.
            self._observer_far_end = None
            w = self._ref_ring_write
            ring_sz = self._ref_ring_size
            if w + hop <= ring_sz:
                self._ref_ring[w:w + hop] = far_end
            else:
                part1 = ring_sz - w
                self._ref_ring[w:ring_sz] = far_end[:part1]
                self._ref_ring[:hop - part1] = far_end[part1:]
            self._ref_ring_write = (w + hop) % ring_sz
            self._ref_ring_filled += hop

            if self._current_delay > 0 and self._ref_ring_filled >= self._current_delay + hop:
                d = self._current_delay
                read_pos = (self._ref_ring_write - hop - d) % ring_sz
                if read_pos + hop <= ring_sz:
                    far_end = self._ref_ring[read_pos:read_pos + hop].copy()
                else:
                    part1 = ring_sz - read_pos
                    far_end = np.concatenate([
                        self._ref_ring[read_pos:ring_sz],
                        self._ref_ring[:hop - part1]
                    ])

                # Live observer: the same ring read ``_observer_shift_hops`` hops earlier than the main read, clamped
                # at 0 (a clip whose delay is shorter than the shift gets as much shift as the delay allows).
                # Read only when enabled.
                if getattr(self.config, 'live_observer_enabled', False):
                    d_obs = max(0, d - self._observer_shift_hops * hop)
                    read_pos_obs = (self._ref_ring_write - hop - d_obs) % ring_sz
                    if read_pos_obs + hop <= ring_sz:
                        self._observer_far_end = self._ref_ring[
                            read_pos_obs:read_pos_obs + hop].copy()
                    else:
                        part1o = ring_sz - read_pos_obs
                        self._observer_far_end = np.concatenate([
                            self._ref_ring[read_pos_obs:ring_sz],
                            self._ref_ring[:hop - part1o]
                        ])
        return far_end

    def _update_render_gates(self, far_end):
        # Far-end activity and stationarity: the single source for the warmup gate, the stationary-DT detector and the EPC stationary guard.
        _render = self._render_activity.update(far_end)
        self._warmup_far_active = _render.warmup_active

        mu_scale = self._get_simple_mu_scale()

        if self._sat_detector_mic is not None:
            mic_clip = self._sat_detector_mic.saturation_level
            if mic_clip > 0.8:
                mu_scale = 0.0
        return _render, mu_scale

    def _prepare_linear_gates(self, far_end):
        # |far_spec|^2 as the delay-aligned render proxy (far_end is already aligned upstream).
        if self._render_signal_analyzer is not None:
            if hasattr(self.filter, 'far_spec') and self.filter.far_spec is not None:
                _rsa_psd = (np.abs(self.filter.far_spec) ** 2).astype(np.float32)
            else:
                _rsa_psd = None
            self._render_signal_analyzer.update(_rsa_psd, far_end)
            # Poor-excitation counter: reset to 0 when poor_signal_excitation fires, else incremented; the filter resumes
            # the W update once it reaches n_partitions.
            if self._render_signal_analyzer.poor_signal_excitation():
                self.filter._poor_excitation_counter = 0
            else:
                self.filter._poor_excitation_counter += 1
            self.filter._saturated_capture = (self._saturation_level > 0.5)
            if self.shadow_filter is not None:
                if self._render_signal_analyzer.poor_signal_excitation():
                    self.shadow_filter._poor_excitation_counter = 0
                else:
                    self.shadow_filter._poor_excitation_counter += 1
                self.shadow_filter._saturated_capture = (self._saturation_level > 0.5)
            if getattr(self.config, 'live_observer_enabled', False):
                if self._render_signal_analyzer.poor_signal_excitation():
                    self._observer_filter._poor_excitation_counter = 0
                else:
                    self._observer_filter._poor_excitation_counter += 1
                self._observer_filter._saturated_capture = (self._saturation_level > 0.5)

        # Push the stationary-block flag so PBFDKF skips the W update on broadband stationary render. The flag is the
        # previous hop's (stationary signals do not flip between hops); the analyzer covers tonal peaks.
        _stat_flag = bool(getattr(self, '_block_stationary_for_next_hop', False))
        self.filter._block_stationary = _stat_flag
        if self.shadow_filter is not None:
            self.shadow_filter._block_stationary = _stat_flag
        if getattr(self.config, 'live_observer_enabled', False):
            self._observer_filter._block_stationary = _stat_flag

    def _update_render_stationarity(self, far_end):
        # Refresh the StationarityEstimator from the render PSD; the flag gates the next hop's W update
        # (the same update is mirrored inside _aec3_post for the residual chain).
        if (self._aec3_stationarity is not None
                and hasattr(self.filter, 'far_spec')
                and self.filter.far_spec is not None):
            if not self._aec3_non_zero_render_seen:
                if float(np.max(np.abs(far_end))) >= self._AEC3_RENDER_PEAK_FLOOR:
                    self._aec3_non_zero_render_seen = True
            if self._aec3_non_zero_render_seen:
                _far_psd_st = (np.abs(self.filter.far_spec) ** 2).astype(np.float32)
                self._aec3_stationarity.update_noise_estimator(_far_psd_st)
                self._aec3_stationarity.update_stationarity_flags(
                    _far_psd_st,
                    average_reverb=self._aec3_avg_render_reverb.reverb,
                )
                self._aec3_stationarity_active_hops += 1
            # Latch for the next hop's filter gate, trusted only after 800 ms of active render.
            _converged_enough = (
                self._aec3_stationarity_active_hops
                >= self._aec3_stationarity_converge_hops
            )
            self._block_stationary_for_next_hop = bool(
                _converged_enough
                and self._aec3_stationarity.is_block_stationary()
            )

    def _process_linear(self, near_end, far_end, mu_scale):
        _shadow_ran_pre_main = False
        if self._freq_near_queue is not None:
            # Buffered FDAF: accumulate into the queue and process when enough is available.
            hop = self._hop_size
            ihop = self._internal_hop
            w = self._freq_queue_write
            self._freq_near_queue[w:w+hop] = near_end
            self._freq_far_queue[w:w+hop] = far_end
            self._freq_queue_write = w + hop

            if self._freq_queue_write >= ihop:
                big_out = self.filter.process(
                    self._freq_near_queue[:ihop],
                    self._freq_far_queue[:ihop], mu_scale)
                leftover = self._freq_queue_write - ihop
                self._freq_out_buf[:ihop] = big_out
                self._freq_out_valid = ihop
                self._freq_out_read = 0
                if leftover > 0:
                    self._freq_near_queue[:leftover] = self._freq_near_queue[ihop:ihop+leftover]
                    self._freq_far_queue[:leftover] = self._freq_far_queue[ihop:ihop+leftover]
                self._freq_queue_write = leftover

            r = self._freq_out_read
            if r + hop > self._freq_out_valid:
                # internal_hop is not a multiple of hop: avoid an overread.
                raw_output = np.zeros(hop, dtype=np.float32)
            else:
                raw_output = self._freq_out_buf[r:r+hop].copy()
                self._freq_out_read = r + hop
        else:
            # Update the RenderSignalAnalyzer and push the gates onto the refined filter before the W update.
            self._prepare_linear_gates(far_end)

            # Run the shadow first and publish _e2_coarse_per_bin, so the main filter's H_error refresh sees the
            # same-hop coarse error (as in AEC3) instead of the previous hop's.
            _shadow_ran_pre_main = False
            if self.shadow_filter is not None and self._freq_near_queue is None:
                _far_thr_pre = 1e-4
                _fe_pre = np.mean(far_end ** 2) > _far_thr_pre
                _ss_pre = self._saturation_level < 0.5
                _smu_pre = 1.0 if (_fe_pre and _ss_pre) else 0.1
                self._last_shadow_output_time = self.shadow_filter.process(
                    near_end, far_end, _smu_pre)
                _shadow_ran_pre_main = True
                if hasattr(self.shadow_filter, 'error_spec'):
                    _e2_coa_pre = (
                        np.abs(self.shadow_filter.error_spec) ** 2
                    ).astype(np.float32)
                    self.filter._e2_coarse_for_refresh = float(np.sum(_e2_coa_pre))
                    self.filter._e2_coarse_per_bin = _e2_coa_pre

            main_mu = 0.0 if self._regime_handler.main_paused else mu_scale
            # The shadow ran first on the same far_end (identical far_buffer), so reuse its far_spec.
            raw_output = self.filter.process(
                near_end, far_end, main_mu,
                precomputed_far_spec=(
                    self.shadow_filter.far_spec if _shadow_ran_pre_main else None))

            self._update_render_stationarity(far_end)
        return raw_output, _shadow_ran_pre_main

    def _update_shadow_leakage(self):
        if (hasattr(self.filter, 'error_spec')
                and hasattr(self.shadow_filter, 'error_spec')):
            _e2_ref = float(np.sum(np.abs(self.filter.error_spec) ** 2))
            _e2_coa_per_bin = (
                np.abs(self.shadow_filter.error_spec) ** 2
            ).astype(np.float32)
            _e2_coa = float(np.sum(_e2_coa_per_bin))
            # Feed E^2_coarse (scalar and per-bin) to the refined filter's H_error refresh.
            self.filter._e2_coarse_for_refresh = _e2_coa
            self.filter._e2_coarse_per_bin = _e2_coa_per_bin
            # Per-bin echo return loss erl[k] = sum_p |W_p[k]|^2, for the H_error leakage refresh.
            self.filter._erl_per_bin = (
                np.abs(self.filter.W) ** 2
            ).sum(axis=0).astype(np.float32)
            # Poor-coarse rescue copy: trigger after blocks_to_hops(5) hops (about 20 ms) of e2_refined < 0.5 * e2_coarse;
            # hangover blocks_to_hops(25) hops (about 100 ms).
            from . import aec3_scale as _aec3_scale
            _cond_fire = bool(_e2_ref < 0.5 * _e2_coa)
            _threshold_hops = _aec3_scale.blocks_to_hops(
                5, self.config.hop_size, self.config.sample_rate)
            if _cond_fire:
                self._poor_coarse_counter = getattr(
                    self, '_poor_coarse_counter', 0) + 1
            else:
                self._poor_coarse_counter = 0
            if self._poor_coarse_counter >= _threshold_hops:
                try:
                    self.shadow_filter.copy_weights_from(self.filter)
                except (AttributeError, TypeError):
                    self.shadow_filter.W[:] = self.filter.W
                self._coarse_reset_hangover = _aec3_scale.blocks_to_hops(
                    25, self.config.hop_size, self.config.sample_rate)
                self._poor_coarse_counter = 0
                # Reset the residual echo estimator on the rescue rising edge so a stale far-end-single-talk tail does not bleed into double-talk.
                if self._aec3_ree is not None:
                    self._aec3_ree.reset()
                    self._diag['aec3_reset_res_on_rescue_edge_fired'] = (
                        self._diag.get(
                            'aec3_reset_res_on_rescue_edge_fired', 0
                        ) + 1
                    )
            # Sustained leakage_div gate: 50 ms of consecutive hops with more than 50 % of bins on the diverged leakage,
            # covering the double-talk onset before the coarse reset hangover; the anti-windup clamp is twice the threshold.
            _ld_frac = float(getattr(
                self.filter, '_last_leakage_div_frac', 0.0))
            _ld_sustain_hops = _aec3_scale.ms_to_hops(
                50.0, self.config.hop_size, self.config.sample_rate)
            _ld_ctr = getattr(
                self, '_leakage_div_sustained_counter', 0)
            _ld_ctr = (min(_ld_ctr + 1, 2 * _ld_sustain_hops) if _ld_frac > 0.5
                       else max(_ld_ctr - 1, 0))
            self._leakage_div_sustained_counter = _ld_ctr
            _dt_leakage_gate = (_ld_ctr >= _ld_sustain_hops)
            if getattr(self, '_coarse_reset_hangover', 0) > 0:
                self._coarse_reset_hangover -= 1
                self.filter._disallow_leakage_diverged = True
            elif _dt_leakage_gate:
                self.filter._disallow_leakage_diverged = True
            else:
                self.filter._disallow_leakage_diverged = False

    def _update_shadow(self, near_end, far_end, _shadow_ran_pre_main):
        if self.shadow_filter is not None and self._freq_near_queue is None:
            self.shadow_frame_count += 1
            # The shadow is a background NLMS filter: it always adapts at full speed and re-converges from its own
            # residual after double-talk; the copy gate is the sole defence against poisoning the main filter.
            # Its update is gated when the far-end is too weak or the speaker saturates (error unreliable).
            _far_thr_shadow = 1e-4
            far_excited = np.mean(far_end ** 2) > _far_thr_shadow
            saturation_safe = self._saturation_level < 0.5
            shadow_mu_scale = 1.0 if (far_excited and saturation_safe) else 0.1
            # During the coarse reset hangover the coarse filter keeps updating (E_refined as gradient source);
            # its output feeds the refined-output selection in ``_aec3_post``.
            # The shadow already ran before the main filter to publish the current-hop coarse error.
            if not _shadow_ran_pre_main:
                self._last_shadow_output_time = self.shadow_filter.process(
                    near_end, far_end, shadow_mu_scale)

            # poor_coarse counter and coarse reset: 5 consecutive hops of E^2_refined < E^2_coarse copy refined to coarse.
            self._update_shadow_leakage()


            main_err = self.filter.get_error_energy()
            shadow_err = self.shadow_filter.get_error_energy()

            alpha_s = self.config.shadow_err_alpha
            self.main_err_smooth = alpha_s * self.main_err_smooth + (1 - alpha_s) * main_err
            self.shadow_err_smooth = alpha_s * self.shadow_err_smooth + (1 - alpha_s) * shadow_err

            self._dt_analyzer.update_shadow_dt(
                shadow_frame_count=self.shadow_frame_count,
                far_excited=far_excited,
                main_err_smooth=self.main_err_smooth,
                shadow_err_smooth=self.shadow_err_smooth,
            )

            # Copy gate delegated to PathChangeRegimeHandler (decision-only); filter mutations are applied here.
            # Shadow-to-main copy requires at least mid-confidence delay, since it overwrites taps.
            _delay_reliable = (
                self.delay_est is not None
                and self.delay_est.confidence >= 0.5
            )

            shadow_decision = self._regime_handler.update(
                shadow_frame_count=self.shadow_frame_count,
                far_pwr=float(np.mean(far_end ** 2)),
                main_err_smooth=float(self.main_err_smooth),
                shadow_err_smooth=float(self.shadow_err_smooth),
                epc_active=self.epc_active,
                saturation_level=float(self._saturation_level),
                dt_from_energy=float(self._dt_from_energy),
                dt_from_coherence=0.0,
                delay_reliable=bool(_delay_reliable),
            )
            if shadow_decision.boost_q:
                if hasattr(self.filter, 'Q') and hasattr(self.filter, 'Q_high'):
                    self._arc_m_q_boost(self.filter)
                    self.filter._p_max_override = 1.0
                    self.filter._p_max_override_frames = 20

    def _update_path_changes(self, far_pwr_global):

        epv_event = self._epc_det.update_epv(
            far_pwr_global=far_pwr_global,
            filter_converged=self._filter_converged,
            main_paused=self._regime_handler.main_paused,
        )
        # EPV weak-filter damping: EPV fires far more often on weak filters (small W norm); re-arming Kalman state
        # on an unwarmed filter destabilises it. The R7_EPV_WEAK_THR environment variable (default 0) skips the response below that W norm.
        _epv_raw = bool(epv_event.fired)
        _epv_suppressed = False
        if _epv_raw:
            _w_thr = float(os.environ.get('R7_EPV_WEAK_THR', '0.0') or 0.0)
            if _w_thr > 0.0 and self.filter is not None:
                _w_norm_now = float(np.linalg.norm(self.filter.W))
                if _w_norm_now < _w_thr:
                    _epv_suppressed = True
        self._diag['epv_event_raw'] = _epv_raw
        self._diag['epv_event_suppressed'] = _epv_suppressed
        if epv_event.fired and not _epv_suppressed and self.config.dt_aware_recovery_soft and self._ne_recent_frames > 0:
            # EPV recovery is double-talk-blind (far-power EMA swing only): skip the aggressive re-adapt, mark_diverged and ERLE reset.
            self._epc_reset_fired_this_frame = True
        elif epv_event.fired and not _epv_suppressed:
            self._epc_reset_fired_this_frame = True  # FQA signal
            # EPV is a gain change: queued for the next _aec3_post to reset ERLE.
            self._aec3_pending_gain_change = True
            # EPV is treated as delay_change=True so H_error and the counters reset.
            if (self.filter is not None
                    and hasattr(self.filter, 'handle_echo_path_change')):
                self.filter.handle_echo_path_change(
                    delay_change=True, gain_change=False, zero_filter=False)
            for filt in [self.filter, self.shadow_filter]:
                if filt and hasattr(filt, 'Q'):
                    self._arc_m_q_boost(filt)
                    if isinstance(filt, PBFDKF):
                        filt._p_max_override = 1.0
                        filt._p_max_override_frames = 30
                        filt._p_floor_beta = 1.0
                        filt._p_floor_beta_frames = 30
            self._maybe_mark_diverged('epv')
            self._epc_render_forced_remaining = self.config.epc_hangover
            self._erl_estimate = min(self._erl_estimate, 0.3)

        # Shadow-error rise (EchoPathChangeDetector); update and hangover tick sit inside the (shadow_filter, filter_converged) gate.
        if self.shadow_filter is not None and self._filter_converged:
            rise_event = self._epc_det.update_shadow_rise(
                main_err_smooth=self.main_err_smooth,
                shadow_err_smooth=self.shadow_err_smooth,
                is_stationary=self._render_activity.is_stationary,
            )
            if rise_event.fired and self.config.dt_aware_recovery_soft and self._ne_recent_frames > 0:
                # shadow_rise recovery is double-talk-blind: skip the aggressive re-adapt, mark_diverged and ERLE reset.
                self._epc_reset_fired_this_frame = True
            elif rise_event.fired:
                self._epc_reset_fired_this_frame = True  # FQA signal
                self._aec3_pending_gain_change = True
                # shadow_rise is a synthetic mistracking detector: treated as delay_change for the H_error and counter reset.
                if (self.filter is not None
                        and hasattr(self.filter, 'handle_echo_path_change')):
                    self.filter.handle_echo_path_change(
                        delay_change=True, gain_change=False, zero_filter=False)
                for filt in [self.filter, self.shadow_filter]:
                    if filt and hasattr(filt, 'Q'):
                        self._arc_m_q_boost(filt)
                self._maybe_mark_diverged('shadow_rise')
                # Relax P_MAX and raise P_floor (PBFDKF only) to abandon the stale path estimate.
                for filt in [self.filter, self.shadow_filter]:
                    if filt and isinstance(filt, PBFDKF):
                        filt._p_max_override = 1.0
                        filt._p_max_override_frames = 30
                        filt._p_floor_beta = 1.0
                        filt._p_floor_beta_frames = 30
                self._epc_render_forced_remaining = self.config.epc_hangover
                self._erl_estimate = min(self._erl_estimate, 0.3)
            else:
                self._epc_det.tick_hangover()

        # Render-forced countdown, decremented every frame (set to epc_hangover on EPV / shadow_rise).
        if self._epc_render_forced_remaining > 0:
            self._epc_render_forced_remaining -= 1

    def _track_linear_error(self, near_end, raw_output):
        erle_windowed = 0.0
        # No output switching: the main filter output is always used.
        # final_output starts from raw_output; RES modifies final_output only
        self._last_raw_output = raw_output  # diagnostic copy
        final_output = raw_output.copy()

        # Misadjustment estimator and ScaleFilter update; a scale action affects subsequent frames only.
        self._update_misadjustment_estimator(
            near_end_hpf=near_end, raw_output=raw_output)
        self._fire_aec3_misadj_scale()

        self._diag['filter_analyzer_consistent'] = bool(
            self._aec3_state.filter_analyzer_consistent())
        self._diag['filter_analyzer_peak_index'] = int(
            self._aec3_state.filter_analyzer_peak_index())
        self._diag['filter_analyzer_max_gain'] = float(
            self._aec3_state.filter_analyzer_max_echo_path_gain())


        # Windowed ERLE is general-purpose (Path B judgment, linear_is_cancelling(), the duty-cycle watchdog,
        # delay_acquire_protect_converged), so it updates regardless of enable_res.
        if self._freq_near_queue is None:
            # Windowed decaying ERLE (time constant 1000 hops) replaces cumulative ERLE; the max with instant ERLE
            # is taken downstream because the windowed value alone is slow at onsets.
            _erle_decay = 0.999  # about 1000 frames
            self._erle_window_near = (_erle_decay * self._erle_window_near
                                       + self.near_power)
            self._erle_window_err = (_erle_decay * self._erle_window_err
                                      + self.raw_error_power)
            erle_windowed = 10.0 * np.log10(
                (self._erle_window_near + 1e-10)
                / (self._erle_window_err + 1e-10))
            # The divergence EMA is likewise independent of enable_res (diagnostic API only).
            self._convergence.update_divergence(self.near_power, self.raw_error_power)
        return final_output, erle_windowed

    def _compute_residual_steering(self, near_end, far_end, raw_output, erle_windowed):
        far_power = np.mean(far_end ** 2)
        # Dynamic over-suppression: erle_factor ramps from 0 dB, scaling with convergence.
        erle_for_factor = max(self.get_erle_instant(), erle_windowed)
        erle_factor = np.clip(erle_for_factor / 10.0, 0.0, 1.0)
        self._erle_factor_prev = float(erle_factor)
        far_pwr = np.mean(far_end ** 2) + 1e-10
        mic_pwr = np.mean(near_end ** 2) + 1e-10
        raw_err_pwr = np.mean(raw_output ** 2) + 1e-10
        # Track ERL for render-based echo estimate; skipped when the residual is dominated by near-end speech.
        erl_update_gate = (far_pwr > 1e-4)
        erl_clip_lo = 0.001
        if erl_update_gate:
            raw_dt_ratio = raw_err_pwr / (far_pwr + 1e-10)
            inst_erl_raw = mic_pwr / far_pwr
            # ERL > 1.5 is physically implausible (near-end dominates), so skip the update.
            if raw_dt_ratio < 2.0 and inst_erl_raw < 1.5:
                inst_erl = np.clip(inst_erl_raw, erl_clip_lo, 1.0)
                alpha_erl = (self._alpha_erl_converged
                             if self._filter_converged
                             else self._alpha_erl_tracking)
                self._erl_estimate = float(alpha_erl * self._erl_estimate + (1 - alpha_erl) * inst_erl)

        # Pre-filter DT signal: mic energy excess over far x max ERL. Realistic rooms have ERL <= +6 dB, so extra mic
        # energy is near-end. Gated on far_active, since near-end-only segments would hang over into the next far-end
        # single-talk segment through the slow decay.
        self._dt_analyzer.update_energy_dt(
            far_active=self._render_activity.is_active,
            far_pwr=far_pwr,
            mic_pwr=mic_pwr,
            erl_estimate=self._erl_estimate,
        )

        simple_dt = 1.0 - far_pwr / (mic_pwr + far_pwr)
        # ERL-corrected blend: the simple ratio misfires in high-ERL far-end single-talk, so take the max of the
        # ERL-corrected estimate and half the simple ratio.
        raw_dt = max(float(self._dt_from_energy), simple_dt * 0.5)

        is_stationary_dt = False
        if self._render_activity.is_stationary and self._filter_converged:
            if hasattr(self.filter, 'error_spec'):
                freq_per_bin = self.config.sample_rate / self.filter.fft_size
                vb_start = max(1, int(100.0 / freq_per_bin))
                vb_limit = min(int(3000.0 / freq_per_bin), len(self.filter.error_spec))
                track_err_pwr = (float(np.sum(
                    np.abs(self.filter.error_spec[vb_start:vb_limit]) ** 2)) + 1e-10)
            else:
                track_err_pwr = raw_err_pwr

            if self._wn_err_baseline < 1e-6:
                self._wn_err_baseline = track_err_pwr

            jump_ratio = track_err_pwr / (self._wn_err_baseline + 1e-10)

            if jump_ratio > 1.5:
                # 800 ms protection window (covers syllable gaps), retimed to the live grid.
                from . import aec3_scale as _aec3_scale_dt
                self._stat_dt_hangover = _aec3_scale_dt.ms_to_hops(
                    800.0, self.config.hop_size, self.config.sample_rate)

            if self._stat_dt_hangover > 0:
                is_stationary_dt = True
                self._stat_dt_hangover -= 1
                self._wn_err_baseline = (0.999 * self._wn_err_baseline
                                          + 0.001 * track_err_pwr)
            else:
                is_stationary_dt = False
                self._wn_err_baseline = (0.95 * self._wn_err_baseline
                                          + 0.05 * track_err_pwr)

        # Track the baseline slowly during non-stationary far-end (converged only), so the first stationary
        # transition does not see a huge jump_ratio.
        if (self._filter_converged and not self._render_activity.is_stationary
                and far_pwr > 1e-4 and self._wn_err_baseline > 1e-6):
            self._wn_err_baseline = (0.995 * self._wn_err_baseline
                                      + 0.005 * raw_err_pwr)

        inst_erle_fast_raw = mic_pwr / raw_err_pwr
        self._inst_erle_smooth = (0.7 * self._inst_erle_smooth
                                  + 0.3 * inst_erle_fast_raw)
        if self._inst_erle_smooth > 2.0:
            # Cap the correction divisor: an inst_erle of 15 would cut raw_dt 0.6 to 0.04 and kill DT protection.
            erle_for_dt = min(self._inst_erle_smooth, 4.0)
            raw_dt /= erle_for_dt

        self._raw_dt_pre_epc = float(raw_dt)
        if self.epc_active:
            raw_dt = 0.0
            is_stationary_dt = False  # the EPC error spike comes from filter divergence, not speech

        dt_indicator = np.clip(raw_dt, 0.0, 0.8)
        return far_power, erle_factor, dt_indicator, is_stationary_dt

    def _build_res_context(self, raw_output, far_power, erle_factor, dt_indicator):
        _res_context = AecResContext(
            raw_output=raw_output.astype(np.float32, copy=True),
            formed_output=np.asarray(
                getattr(self, '_res_formed_output', raw_output),
                dtype=np.float32,
            ).copy(),
            echo_spec=np.asarray(
                getattr(self, '_res_echo_spec',
                        getattr(self.filter, 'echo_spec', np.zeros(1))),
                dtype=np.complex64,
            ).copy(),
            far_power=float(far_power),
            far_spec=np.asarray(
                getattr(self.filter, 'far_spec', np.zeros(1)),
                dtype=np.complex64,
            ).copy(),
            near_spec=np.asarray(
                getattr(self, '_res_near_spec',
                        getattr(self.filter, 'near_spec', np.zeros(1))),
                dtype=np.complex64,
            ).copy(),
            filter_converged=bool(self._convergence.converged),
            # Same-frame local values (the diag entries are written after this build).
            erle_factor=float(
                erle_factor),
            dt_indicator=float(dt_indicator),
            divergence=float(self._divergence_indicator),
            over_sub=float(
                erle_factor),
            saturation_level=float(self._saturation_level),
            erl_estimate=float(self._erl_estimate),
            # Frequency-domain seam for an external post-NR RES (set in _aec3_post).
            error_spec=getattr(self, '_res_error_spec', None),
            res_gain=getattr(self, '_res_gain', None),
            comfort_noise=getattr(self, '_res_comfort_noise', None),
            r2=getattr(self, '_res_r2', None),
            res_floor_protect=bool(
                getattr(self._aec3_sg, '_dt_protect_active', False)),
            usable_linear=bool(
                self._aec3_state is not None
                and self._aec3_state.usable_linear_estimate()),
        )
        return _res_context

    def _update_residual_mu(self, is_stationary_dt, raw_output, far_end):
        if self._filter_converged:
            mu_min = self.config.shadow_mu_min
            self._per_bin_mu_scale = np.full(
                self.filter.n_freqs, mu_min, dtype=np.float32)
            self._simple_mu_ratio = 0.0
            # Stationary DT freezes only when speech is actually detected; the stationary-far flag alone also fires on normal speech.
            if is_stationary_dt:
                self._per_bin_mu_scale[:] = mu_min
                self._simple_mu_ratio = mu_min
        else:
            # Pre-convergence: no per-bin scale; the ratio tracks DT naturally.
            self._per_bin_mu_scale = None
            # Only the RES path updates the ratio here; the non-RES path updates it elsewhere (avoids a double update).
            if self.config.enable_res:
                self._update_simple_mu_ratio(raw_output, far_end)

    def _process_residual(self, near_end, far_end, raw_output, erle_windowed):
        _res_context = None
        far_power, erle_factor, dt_indicator, is_stationary_dt = self._compute_residual_steering(near_end, far_end, raw_output, erle_windowed)

        final_output = self._aec3_post(raw_output, near_end, far_end)

        # _aec3_post already ran the FORM step for this hop, so copy the WOLA memory out instead of calling the selector again.
        if self.config.return_formed_output:
            self._formed_output = self._form_prev_output_time.copy()

        if self.config.return_res_context:
            _res_context = self._build_res_context(raw_output, far_power, erle_factor, dt_indicator)

        self._update_residual_mu(is_stationary_dt, raw_output, far_end)
        return final_output, _res_context, erle_factor, dt_indicator

    def _update_linear_only_steering(self, near_end, far_end, raw_output):
        # No-RES steering: in context-only mode this is the in-gate update; with the post chain and context seam both off
        # it is the sole update site.
        if (not self.config.enable_res
                and not self._filter_converged):
            self._update_simple_mu_ratio(raw_output, far_end)
        elif (not self.config.enable_res
                and not self.config.return_res_context
                and self._filter_converged):
            # The no-RES path skips the post block, so mirror the converged mu clamp to keep the filter trajectory the same.
            mu_min = self.config.shadow_mu_min
            self._per_bin_mu_scale = np.full(
                self.filter.n_freqs, mu_min, dtype=np.float32)
            self._simple_mu_ratio = 0.0

        # The no-RES path also skips the ERL / DT updates; mirror them for the same reason.
        if (not self.config.enable_res
                and not self.config.return_res_context
                and self._freq_near_queue is None):
            _c2_far_pwr = float(np.mean(far_end ** 2)) + 1e-10
            _c2_mic_pwr = float(np.mean(near_end ** 2)) + 1e-10
            _c2_err_pwr = float(np.mean(raw_output ** 2)) + 1e-10
            if _c2_far_pwr > 1e-4:
                _c2_dt_ratio = _c2_err_pwr / _c2_far_pwr
                _c2_erl_raw = _c2_mic_pwr / _c2_far_pwr
                if _c2_dt_ratio < 2.0 and _c2_erl_raw < 1.5:
                    _c2_inst_erl = float(np.clip(_c2_erl_raw, 0.001, 1.0))
                    # Same alpha pair as the primary ERL update.
                    _c2_alpha = (self._alpha_erl_converged
                                 if self._filter_converged
                                 else self._alpha_erl_tracking)
                    self._erl_estimate = (_c2_alpha * self._erl_estimate
                                           + (1.0 - _c2_alpha) * _c2_inst_erl)
            self._dt_analyzer.update_energy_dt(
                far_active=self._render_activity.is_active,
                far_pwr=_c2_far_pwr,
                mic_pwr=_c2_mic_pwr,
                erl_estimate=self._erl_estimate,
            )

    def _update_frequency_diagnostics(self, mu_scale, erle_factor, erle_windowed, dt_indicator):
        self._diag['res_gain_mean'] = 1.0
        self._diag['res_gain_min'] = 1.0
        self._diag['effective_g_min'] = 1.0
        # far_activity is written in _aec3_post; do not overwrite it here.
        self._diag['echo_psd_mean'] = 0.0
        self._diag['error_psd_mean'] = 0.0
        self._diag['p4b_dt_per_bin_mean'] = 0.0
        self._diag['p4b_dt_per_bin_hf_mean'] = 0.0
        self._diag['p4b_coh2_hf_mean'] = 0.0
        self._diag['p4b_effective_dt'] = 0.0
        self._diag['p4b_is_stationary_dt'] = 0
        self._diag['p4b_gain_hf_mean'] = 1.0
        self._diag['p4b_res_echo_hf_mean_db'] = -120.0
        self._diag['erle_inst'] = self.get_erle_instant()

        mu_val = mu_scale
        self._diag['mu_scale'] = float(np.mean(mu_val)) if isinstance(mu_val, np.ndarray) else float(mu_val)
        self._diag['converged'] = self._filter_converged
        self._diag['erle_factor'] = float(erle_factor)
        self._diag['divergence'] = self._divergence_indicator
        self._diag['using_render_based'] = False
        self._diag['shadow_advantage'] = self._shadow_advantage
        self._diag['dt_from_energy'] = self._dt_from_energy
        self._diag['dt_from_shadow'] = self._dt_from_shadow
        self._diag['erl_estimate'] = self._erl_estimate
        self._diag['per_band_erl_lf'] = float(self._per_band_erl[0])
        self._diag['per_band_erl_mf'] = float(self._per_band_erl[1])
        self._diag['per_band_erl_hf'] = float(self._per_band_erl[2])
        self._diag['epc_active'] = self.epc_active
        self._diag['saturation_level'] = self._saturation_level
        self._diag['erle_windowed'] = float(erle_windowed)
        self._diag['dt_indicator'] = float(dt_indicator)
        self._diag['main_err_smooth'] = float(self.main_err_smooth)
        self._diag['shadow_err_smooth'] = float(getattr(self, 'shadow_err_smooth', 0.0))
        self._diag['main_paused'] = bool(self._regime_handler.main_paused)
        _epv_ratio = (self._epv_gain_fast / (self._epv_gain_slow + 1e-10)
                      if self._epv_gain_slow > 1e-12 else 1.0)
        self._diag['epv_gain_ratio'] = float(_epv_ratio)
        self._diag['dt_residual_scale'] = 1.0
        self._diag['filter_w_norm'] = float(np.linalg.norm(self.filter.W)) if hasattr(self.filter, 'W') else 0.0
        self._diag['shadow_w_norm'] = (float(np.linalg.norm(self.shadow_filter.W))
                                        if self.shadow_filter and hasattr(self.shadow_filter, 'W') else 0.0)
        self._diag['copy_err_baseline'] = float(self._regime_handler.copy_err_baseline)

    def _update_filter_state_diagnostics(self):
        # Full-band ratios e2/y2 from the sample-loop power EMAs (commensurate units); the shadow error power is scaled
        # from main/shadow_err_smooth with the same factor, so the ratio is preserved.
        _mic_pwr_p3f = max(float(self.near_power), 1e-12)
        _main_err_pwr = max(float(self.raw_error_power), 1e-12)
        _main_err_smooth_p3f = max(float(self.main_err_smooth), 1e-12)
        _shadow_err_smooth_p3f = max(float(self.shadow_err_smooth), 1e-12)
        _shadow_err_pwr = _shadow_err_smooth_p3f * (
            _main_err_pwr / _main_err_smooth_p3f)
        _main_err_ratio = _main_err_pwr / _mic_pwr_p3f
        _shadow_err_ratio = _shadow_err_pwr / _mic_pwr_p3f
        _shadow_advantage_p3f = (
            _main_err_smooth_p3f / _shadow_err_smooth_p3f)

        self._post_reset_age_frames += 1
        _hop_ms_p3f = float(self.config.hop_size) * 1000.0 / float(self.config.sample_rate)
        _post_reset_age_ms = self._post_reset_age_frames * _hop_ms_p3f

        _erle_inst_db = float(self._diag.get('erle_inst', 0.0))
        self._erle_slope_buf.append(_erle_inst_db)
        if len(self._erle_slope_buf) >= 2:
            _slope_dt = (len(self._erle_slope_buf) - 1) * _hop_ms_p3f / 1000.0
            _erle_slope_db_per_s = (
                self._erle_slope_buf[-1] - self._erle_slope_buf[0]
            ) / max(_slope_dt, 1e-6)
        else:
            _erle_slope_db_per_s = 0.0

        # Filter-state classifier. Priority: idle < startup < diverged < refined_usable < suspicious_dt < coarse_learning (default).
        # Startup is the unstable post-reset window only (the Q-boost re-arms repeatedly during healthy adaptation).
        _post_reset_warmup_p3f = (self._warmup_frames > 0)
        _far_active_p3f = bool(self._render_activity.is_active)
        # Idle: far silent or near silent (error clamped at its floor), so there is nothing to classify.
        _is_idle = (
            (not _far_active_p3f)
            or _main_err_smooth_p3f <= 1e-9
            or _mic_pwr_p3f <= 1e-8
        )
        # Diverged: err/mic ratio above 1.3 for 5 consecutive far-active frames.
        _diverged_th = 1.3
        _diverged_min_streak = 5
        _diverged_hit_this_frame = (not _is_idle) and _main_err_ratio > _diverged_th
        if _diverged_hit_this_frame:
            self._p3f_diverged_streak += 1
        else:
            self._p3f_diverged_streak = 0

        # Suspicious DT: near-end evidence, a main_err jump above the refined baseline and a shadow lead, all together.
        _ne_evidence = (
            float(self._dt_from_energy) > 0.3
            or float(self._dt_from_shadow) > 0.5
        )
        _main_err_jump = (
            self._p3f_main_err_baseline > 1e-6
            and _main_err_ratio > 2.0 * self._p3f_main_err_baseline
        )
        _shadow_lead = _shadow_advantage_p3f > 1.5
        _suspicious_dt_hit = (
            self._p3f_refined_latched
            and _ne_evidence
            and _main_err_jump
            and _shadow_lead
        )

        if _is_idle:
            _filter_state = 'idle'
        elif _post_reset_warmup_p3f or _post_reset_age_ms < 200.0:
            _filter_state = 'startup'
        elif self._p3f_diverged_streak >= _diverged_min_streak:
            _filter_state = 'diverged'
        elif _suspicious_dt_hit:
            # Takes precedence over refined_usable.
            _filter_state = 'suspicious_dt'
        elif (self._filter_once_converged
              and _main_err_ratio < 0.7
              and _erle_slope_db_per_s > -5.0):
            # Latch refined once seen so suspicious_dt can detect departures from it.
            self._p3f_refined_latched = True
            _filter_state = 'refined_usable'
        else:
            _filter_state = 'coarse_learning'

        # The baseline follows the best converged ratio downward and freezes when it rises, so a contaminated ratio
        # does not drag it along; updated only in refined_usable.
        if _filter_state == 'refined_usable':
            if self._p3f_main_err_baseline <= 1e-6:
                self._p3f_main_err_baseline = _main_err_ratio
            elif _main_err_ratio < self._p3f_main_err_baseline:
                self._p3f_main_err_baseline = (
                    0.9 * self._p3f_main_err_baseline + 0.1 * _main_err_ratio)

        _delay_solid_p3f = bool(
            self.delay_est is not None
            and getattr(self.delay_est, 'is_solid', False))
        _usable_linear = bool(
            _delay_solid_p3f
            and _filter_state == 'refined_usable')

        self._diag['main_err_ratio'] = float(_main_err_ratio)
        self._diag['shadow_err_ratio'] = float(_shadow_err_ratio)
        self._diag['p3f_shadow_advantage'] = float(_shadow_advantage_p3f)
        self._diag['erle_slope_db_per_s'] = float(_erle_slope_db_per_s)
        self._diag['post_reset_age_ms'] = float(_post_reset_age_ms)
        self._diag['filter_state'] = str(_filter_state)
        self._diag['usable_linear'] = bool(_usable_linear)
        self._diag['p3f_main_err_baseline'] = float(self._p3f_main_err_baseline)

    def _update_energy_diagnostics(self, near_end, far_pwr_global):

        self._far_power_ema = 0.95 * self._far_power_ema + 0.05 * far_pwr_global
        self._mic_power_ema = 0.95 * self._mic_power_ema + 0.05 * (np.mean(near_end ** 2) + 1e-10)
        self._frame_count += 1
        # Statistics-only DT-from-frame-0 detector: over 200 far-active hops without convergence and a drifted ERL (a sign of near-end-corrupted learning).
        if self._render_activity.is_active:
            self._far_active_blocks = getattr(self, '_far_active_blocks', 0) + 1
        if (getattr(self, '_far_active_blocks', 0) > 200
                and not self._filter_converged
                and self._erl_estimate > 0.4):
            self._dt_from_zero_count = getattr(self, '_dt_from_zero_count', 0) + 1
        self._diag['dt_from_zero_count'] = getattr(self, '_dt_from_zero_count', 0)
        self._diag['far_power'] = self._far_power_ema
        self._diag['mic_power'] = self._mic_power_ema
        self._diag['dt_from_coherence'] = 0.0

    def _process_frequency(self, near_end, far_end, mu_scale, far_pwr_global):
        _res_context = None
        erle_factor = 0.0
        dt_indicator = 0.0
        raw_output, _shadow_ran_pre_main = self._process_linear(near_end, far_end, mu_scale)

        self._update_shadow(near_end, far_end, _shadow_ran_pre_main)
        self._update_path_changes(far_pwr_global)

        final_output, erle_windowed = self._track_linear_error(near_end, raw_output)

        # Post-filter (overlap-add, sqrt-Hann), skipped for buffered FDAF. enable_res gates suppression;
        # erle_windowed and the divergence EMA are computed above unconditionally.
        if (self.config.enable_res or self.config.return_res_context) and self._freq_near_queue is None:
            final_output, _res_context, erle_factor, dt_indicator = self._process_residual(near_end, far_end, raw_output, erle_windowed)

        self._update_linear_only_steering(near_end, far_end, raw_output)

        self._update_frequency_diagnostics(mu_scale, erle_factor, erle_windowed, dt_indicator)

        self._update_filter_state_diagnostics()

        self._update_energy_diagnostics(near_end, far_pwr_global)
        return raw_output, final_output, _res_context

    def _track_erle_convergence(self, near_end, far_end, raw_output, _res_context):
        # No output limiter here; peak control belongs in a later AGC / DRC / output stage.

        for i in range(len(near_end)):
            self.near_power = self.alpha * self.near_power + (1 - self.alpha) * near_end[i] ** 2
            self.raw_error_power = self.alpha * self.raw_error_power + (1 - self.alpha) * raw_output[i] ** 2
        self.near_power_sum += np.sum(near_end ** 2)
        self.raw_error_power_sum += np.sum(raw_output ** 2)
        self.error_power = self.raw_error_power
        self.error_power_sum = self.raw_error_power_sum

        # Converged after 10 consecutive far-active frames with ERLE > 5 dB. Gated on far_active (not
        # _simple_mu_ratio, which can deadlock in high-coupling single-talk); no DT gate is needed.
        just_converged = self._convergence.update_convergence(
            near_power=self.near_power,
            raw_error_power=self.raw_error_power,
            far_active=float(np.mean(far_end ** 2)) > 1e-4,
            warmup_done=self._warmup_frames <= 0,
        )
        if just_converged:
            for filt in [self.filter, self.shadow_filter]:
                if filt and hasattr(filt, 'Q_low'):
                    filt.Q = filt.Q_low.copy()

        # AecResContext.filter_converged was built before update_convergence ran this hop; patch it so the
        # single-call return reports the post-update state.
        if _res_context is not None:
            _res_context.filter_converged = bool(self._convergence.converged)
        return just_converged

    def _update_state_diagnostics(self, just_converged):
        _initial_state_active = (
            (not self._filter_once_converged)
            or self._frame_count < (self.config.warmup_frames + 50)
        )
        _initial_transition_triggered = bool(just_converged)
        _epc_hangover = self._epc_det.hangover_count
        _usable_v1 = (self._filter_once_converged
                       and self._filter_converged
                       and not self.epc_active)
        _usable_v2 = (
            _usable_v1
            and self._frame_count > self.config.warmup_frames + 30
            and self._convergence.divergence < 0.3
            and _epc_hangover < 1
        )
        _ne_raw = False
        _hold = getattr(self, '_dominant_nearend_hold', 0)
        if _ne_raw and not _initial_state_active:
            _hold = 5
        else:
            _hold = max(_hold - 1, 0)
        self._dominant_nearend_hold = _hold
        _dominant_ne_state = _hold > 0
        if _initial_transition_triggered:
            _erle_reset_signal = 1   # startup-tail
        elif self.epc_active:
            _erle_reset_signal = 2   # EPC
        elif self._convergence.divergence > 0.5:
            _erle_reset_signal = 3   # divergence-spike
        else:
            _erle_reset_signal = 0
        self._diag['initial_state_active'] = bool(_initial_state_active)
        self._diag['initial_transition_triggered'] = _initial_transition_triggered
        self._diag['usable_linear_estimate_v1'] = bool(_usable_v1)
        self._diag['usable_linear_estimate_v2'] = bool(_usable_v2)
        self._diag['dominant_nearend_like_state'] = bool(_dominant_ne_state)
        self._diag['erle_reset_signal'] = int(_erle_reset_signal)

        # Trace-only diagnostics: delay / EPC / P-override / scale (read-only).
        self._diag['epc_active_now'] = bool(self._epc_det.active)
        self._diag['epc_hangover_count'] = int(self._epc_det.hangover_count)
        # P-override flags: either filter in a transient boost.
        _pmax_active = False
        _pfloor_active = False
        for _filt in [self.filter, self.shadow_filter]:
            if _filt is None:
                continue
            if getattr(_filt, '_p_max_override', 0.5) > 0.6:
                _pmax_active = True
            if getattr(_filt, '_p_floor_beta', 0.1) > 0.15:
                _pfloor_active = True
        self._diag['p_max_override_active'] = _pmax_active
        self._diag['p_floor_beta_active'] = _pfloor_active
        self._diag['div_source_last'] = getattr(self, '_last_divergence_source', '')
        self._diag['div_counts'] = dict(getattr(self, '_divergence_source_counts',
                                                {'delay_first': 0, 'delay_shift': 0,
                                                 'epv': 0, 'shadow_rise': 0}))
        # Filter scale ratio: filter echo power vs the render-based estimate (far * erl); persistently << 1 means underestimation.
        try:
            _fpw = float(np.mean(np.abs(self.filter.echo_spec) ** 2))
            _fp_far = float(np.mean(np.abs(self.filter.far_spec) ** 2))
            _scale_ratio = _fpw / (_fp_far * float(self._erl_estimate) + 1e-12)
            self._diag['filter_scale_ratio'] = _scale_ratio
        except Exception:
            self._diag['filter_scale_ratio'] = 1.0
        self._diag['inst_erle_smooth'] = float(self._inst_erle_smooth)
        # raw_dt before the EPC zero (the value it would have had).
        self._diag['raw_dt_pre_epc'] = float(getattr(self, '_raw_dt_pre_epc', 0.0))

    def _update_trajectory_diagnostics(self, near_end, raw_output, final_output):
        # Per-stage gain trace slots are constant zero (diag dict structure kept for external consumers).
        for _n in ('softgate_emr', 'spectral_floor', 'epc_dt_cap',
                   'quiet_mask', '3bin_smooth', 'hf_cap',
                   'pre_temporal', 'post_temporal', 'after_noise_lift'):
            self._diag[f'g_stage_{_n}_voice'] = 0.0

        if not hasattr(self, '_diag_prev_delay'):
            self._diag_prev_delay = -1
            self._diag_prev_divergence_counts = {
                'delay_first': 0,
                'delay_shift': 0,
                'epv': 0,
                'shadow_rise': 0,
            }

        cur_delay = int(getattr(self, '_current_delay', -1))
        self._diag['delay_samples'] = cur_delay
        if cur_delay >= 0 and self._diag_prev_delay >= 0:
            self._diag['delay_delta'] = cur_delay - self._diag_prev_delay
        else:
            self._diag['delay_delta'] = 0
        self._diag_prev_delay = cur_delay

        cur_div = getattr(self, '_divergence_source_counts',
                          {'delay_first': 0, 'delay_shift': 0, 'epv': 0, 'shadow_rise': 0})
        for _src in ('delay_first', 'delay_shift', 'epv', 'shadow_rise'):
            self._diag[f'event_{_src}'] = bool(
                cur_div.get(_src, 0)
                > self._diag_prev_divergence_counts.get(_src, 0)
            )
        self._diag_prev_divergence_counts = dict(cur_div)

        _filt = getattr(self, 'filter', None)
        self._diag['p_max_override_remaining'] = int(getattr(_filt, '_p_max_override_frames', 0)) if _filt is not None else 0
        self._diag['p_floor_beta_remaining'] = int(getattr(_filt, '_p_floor_beta_frames', 0)) if _filt is not None else 0

        self._diag['filter_once_converged'] = bool(self._filter_once_converged)
        self._diag['filter_converged_now'] = bool(self._convergence.converged)
        self._diag['epc_render_forced_remaining'] = int(getattr(self, '_epc_render_forced_remaining', 0))

        try:
            _np_eps = 1e-12
            _mic_pwr = float(np.mean(near_end ** 2))
            _nores_pwr = float(np.mean(raw_output ** 2))
            _final_pwr = float(np.mean(final_output ** 2))
            self._diag['mic_power_frame'] = _mic_pwr
            self._diag['nores_output_power'] = _nores_pwr
            self._diag['final_output_power'] = _final_pwr
            self._diag['res_required_gain'] = float(np.sqrt(_final_pwr / max(_nores_pwr, _np_eps)))
            self._diag['nores_echo_proxy'] = float(np.sqrt(_nores_pwr / max(_mic_pwr, _np_eps)))
        except (NameError, AttributeError):
            pass

    def _post_prepare_spectra(self, raw_output, near_end, far_end, _PSD_SCALE):
        # Form a reconstructing 50 %-overlap WOLA frame from the time-domain linear output (error_spec_windowed is an
        # estimator quantity, not the STFT of the residual). With a shadow output the helper also selects / crossfades
        # refined vs coarse; without one it formats the refined output only.
        _coarse_time = (
            self._last_shadow_output_time
            if self.shadow_filter is not None
            and self._last_shadow_output_time is not None
            else None
        )
        _sel_esw, _sel_echo_spec = self._aec3_select_linear_filter_output(
            e_refined_time=raw_output,
            near_end_block=near_end,
            e_coarse_time=_coarse_time,
        )
        near_psd = (np.abs(self.filter.near_spec) ** 2 * _PSD_SCALE).astype(np.float32)
        far_psd = (np.abs(self.filter.far_spec) ** 2 * _PSD_SCALE).astype(np.float32)
        echo_psd = (np.abs(_sel_echo_spec) ** 2 * _PSD_SCALE).astype(np.float32)
        error_spec = _sel_esw
        error_psd = (np.abs(error_spec) ** 2 * _PSD_SCALE).astype(np.float32)
        # Windowed Y2 for the ERLE Y2/E2 coordinate (E2 is windowed too). _near_psd_win is formed once when either
        # erle_windowed_capture_psd or cng_n2_windowed_psd needs it.
        _near_psd_win = None
        if (getattr(self.config, "erle_windowed_capture_psd", True)
                or getattr(self.config, "cng_n2_windowed_psd", False)):
            _near_spec_win = (self.filter.error_spec_windowed
                              + self.filter.echo_spec).astype(np.complex64)
            _near_psd_win = (np.abs(_near_spec_win) ** 2
                             * _PSD_SCALE).astype(np.float32)
        # The ERLE consumer is gated on erle_windowed_capture_psd alone; None keeps near_psd.
        _capture_psd_erle = (
            _near_psd_win
            if getattr(self.config, "erle_windowed_capture_psd", True)
            else None
        )
        far_pwr = float(np.mean(far_end ** 2))
        # Render block scaled to int16 amplitude for the AEC3 absolute thresholds.
        render_block_scaled = (far_end * 32768.0).astype(np.float32)
        return _sel_echo_spec, near_psd, far_psd, echo_psd, error_spec, error_psd, _near_psd_win, _capture_psd_erle, far_pwr, render_block_scaled

    def _post_filter_bridge(self, raw_output, near_end, build_filter_state_bridge):
        # Per-frame convergence as in the subtractor output analyzer: refined converged when e2_refined < 0.5 y2, coarse
        # when e2_coarse < 0.05 y2, both with y2 above the convergence threshold (50^2 * hop on the float scale);
        # either counts. The coarse energy comes from the shadow error spectrum by Parseval.
        from . import aec3_scale as _aec3_scale_y2
        _y2_time = float(np.sum(near_end.astype(np.float64) ** 2))
        _e2_refined = float(np.sum(raw_output.astype(np.float64) ** 2))
        # Threshold scaled to the live hop via block_energy_scale.
        _y2_threshold = _aec3_scale_y2.block_energy_scale(
            50.0 * 50.0 * 64.0, self.config.hop_size
        ) / (32768.0 ** 2)
        # Relaxed gate threshold: 20^2 * hop.
        _y2_threshold_low = _y2_threshold * (20.0 / 50.0) ** 2
        # Strict diverged gate: y2 above 30^2 * hop.
        _y2_threshold_diverged = _y2_threshold * (30.0 / 50.0) ** 2
        _refined_conv = _e2_refined < 0.5 * _y2_time and _y2_time > _y2_threshold
        # Shadow coarse convergence: error-spectrum energy converted to time-domain energy by Parseval.
        _coarse_conv = False
        _coarse_conv_relaxed = False
        _e2_coarse = 0.0
        if self.shadow_filter is not None:
            # Parseval-mapped energy over the fft_size window.
            if hasattr(self.shadow_filter, 'error_spec'):
                _e_spec = self.shadow_filter.error_spec
                _e2_coarse = float(
                    (2 * np.sum(np.abs(_e_spec[1:-1]) ** 2) + np.abs(_e_spec[0]) ** 2 + np.abs(_e_spec[-1]) ** 2)
                    / self.filter.fft_size
                )
            _coarse_conv = _e2_coarse < 0.05 * _y2_time and _y2_time > _y2_threshold
            _coarse_conv_relaxed = _e2_coarse < 0.3 * _y2_time and _y2_time > _y2_threshold_low
        _aec3_converged = _refined_conv or _coarse_conv
        # Strict all_filters_diverged flag.
        _min_e2 = min(_e2_refined, _e2_coarse) if self.shadow_filter is not None else _e2_refined
        _all_diverged = _min_e2 > 1.5 * _y2_time and _y2_time > _y2_threshold_diverged

        bridge = build_filter_state_bridge(
            filter_converged=_aec3_converged,
            pbfdkf=self.filter,
            regime_handler=self._regime_handler,
            mu_final=1.0,
            external_delay_samples=int(self._current_delay) if self._delay_active else -1,
            shadow_filter=self.shadow_filter,
            any_coarse_filter_converged=_coarse_conv_relaxed,
            all_filters_diverged=_all_diverged,
        )
        return bridge

    def _post_external_delay(self):
        # The filter-quality gate needs an external delay or seen convergence before the linear estimate becomes usable;
        # without it the linear branch never engages.
        from .delay.delay_types import DelayEstimate, DelayQuality
        if self._delay_active and self._current_delay >= 0:
            ext_delay = DelayEstimate(
                quality=DelayQuality.REFINED, delay=int(self._current_delay)
            )
        else:
            ext_delay = None
        # Gate diagnostics for the external-delay source.
        _ext_delay_present = (ext_delay is not None)
        _delay_is_solid = (
            self.delay_est is not None
            and getattr(self.delay_est, 'is_solid', False)
        )
        _fixed_delay_active = (self.config.delay_mode is AecDelayMode.FIXED)
        if not _ext_delay_present:
            _ext_delay_source = 'none'
        elif _fixed_delay_active:
            _ext_delay_source = 'fixed'
        elif _delay_is_solid:
            _ext_delay_source = 'is_solid'
        else:
            _ext_delay_source = 'always'
        return ext_delay

    def _post_pending_changes(self):
        # Consume the pending variability queued by the event detectors; handle_echo_path_change must run before update().
        from .delay.delay_types import DelayAdjustment, EchoPathVariability
        if (self._aec3_pending_gain_change
                or self._aec3_pending_delay_change is not None):
            variability = EchoPathVariability(
                gain_change=bool(self._aec3_pending_gain_change),
                delay_change=(self._aec3_pending_delay_change
                              if self._aec3_pending_delay_change is not None
                              else DelayAdjustment.NONE),
                clock_drift=False,
            )
            _a1_was_delay = (self._aec3_pending_delay_change is not None)
            self._aec3_state.handle_echo_path_change(variability)
            # A delay change sets the suppressor's initial state.
            if _a1_was_delay and self._aec3_sg is not None:
                self._aec3_sg.set_initial_state(True)
                self._diag['a1_set_initial_state'] = 'on'
            self._aec3_pending_gain_change = False
            self._aec3_pending_delay_change = None

    def _post_filter_taps(self):
        self._aec3_state.update_capture_saturation(self._saturation_level > 0.5)
        # Feed the FilterAnalyzer the full time-domain impulse response.
        _filter_taps_full = (
            self.filter.get_time_domain_filter()
            if (self.filter is not None
                and hasattr(self.filter, 'get_time_domain_filter'))
            else None
        )
        return _filter_taps_full

    def _post_render_reverb(self, _PSD_SCALE):
        # Average render reverb for the ERLE estimators only: X^2[delay] plus a separate steady-decay ReverbModel
        # fed with X^2[delay + 1]; other consumers use the bare render PSD.
        _x2_reverb_for_erle = None
        _ree = getattr(self, "_aec3_ree", None)
        _avg = getattr(self, "_aec3_avg_render_reverb", None)
        if (_ree is not None and _avg is not None
                and hasattr(self.filter, "X_buf")
                and hasattr(self.filter, "partition_idx")):
            _n_part = self.filter.n_partitions
            # After filter.process() partition_idx is the next write slot, so the newest X^2 is at partition_idx - 1.
            _curr_p = (self.filter.partition_idx - 1) % _n_part
            _delay = int(self._aec3_state.min_direct_path_filter_delay())
            _delay = max(0, min(_delay, _n_part - 2))  # n_part - 2 keeps _past_idx from wrapping to the current slot
            _delay_idx = (_curr_p - _delay) % _n_part
            _past_idx = (_curr_p - _delay - 1) % _n_part
            _x2_at_delay = (np.abs(self.filter.X_buf[_delay_idx])
                            ** 2).astype(np.float32)
            _x2_past = (np.abs(self.filter.X_buf[_past_idx])
                        ** 2).astype(np.float32)
            _decay_steady = _ree._reverb_decay(dominant_nearend=False)
            _avg.update_no_freq_shaping(
                _x2_past, scaling=1.0, decay=_decay_steady,
            )
            _x2_reverb_for_erle = (
                _x2_at_delay + _avg.reverb
            ).astype(np.float32)
            # Scale the ERLE render-activity X^2 to the int16^2 PSD convention so the band-energy threshold can fire
            # (see AecConfig.erle_render_x2_psd_scale).
            if getattr(self.config, "erle_render_x2_psd_scale", False):
                _x2_reverb_for_erle = (
                    _x2_reverb_for_erle * _PSD_SCALE
                ).astype(np.float32)
        return _x2_reverb_for_erle

    def _post_update_state(self, bridge, ext_delay, far_psd, near_psd, error_psd, echo_psd, far_pwr, render_block_scaled, _filter_taps_full, _x2_reverb_for_erle, _capture_psd_erle):
        # Saturation inputs: echo-predictor time-domain peaks of the refined and shadow filters, scaled to int16 amplitude.
        _s_ref_max = float(getattr(self.filter, "_last_s_max_abs", 0.0)) * 32768.0
        _s_coa_max = ((float(getattr(self.shadow_filter, "_last_s_max_abs", 0.0)) * 32768.0)
                      if self.shadow_filter is not None else 0.0)
        # active_render threshold 100^2 * 64 / 32768^2 (about 5.96e-4).
        _ar_thr = 5.96e-4
        _active_render_this_frame = bool(far_pwr > _ar_thr)
        self._diag['far_activity'] = 1.0 if _active_render_this_frame else 0.0
        # Coherence-based ERLE gate: gamma^2(echo estimate, capture) per bin; double-talk lowers it and closes the gate.
        _coh_gate_mask: Optional[np.ndarray] = None
        # The accumulators feed only the ERLE gate, so skip them when it is disabled.
        if self.config.erle_coh_gate_enabled:
            _echo_c = self.filter.echo_spec.astype(np.complex64)
            _near_c = self.filter.near_spec.astype(np.complex64)
            _a = float(self.config.erle_coh_gate_alpha)
            self._coh_erle_sye = (
                (1.0 - _a) * self._coh_erle_sye + _a * (_echo_c * np.conj(_near_c))
            )
            self._coh_erle_syy = (
                (1.0 - _a) * self._coh_erle_syy + _a * np.abs(_echo_c) ** 2
            )
            self._coh_erle_see = (
                (1.0 - _a) * self._coh_erle_see + _a * np.abs(_near_c) ** 2
            )
            _gamma2 = (np.abs(self._coh_erle_sye) ** 2
                       / np.maximum(self._coh_erle_syy * self._coh_erle_see,
                                    1e-30)).astype(np.float32)
            _coh_gate_mask = (_gamma2 >= self.config.erle_coh_gate_threshold)
        self._aec3_state.update(
            bridge=bridge,
            external_delay=ext_delay,
            render_psd=far_psd,
            capture_psd=near_psd,
            error_psd=error_psd,
            echo_psd=echo_psd,
            active_render=_active_render_this_frame,
            render_block=render_block_scaled,
            filter_taps_full=_filter_taps_full,
            sde_filter_freq_response=None,
            sde_x2_history=None,
            subtractor_s_refined_max_abs=_s_ref_max,
            subtractor_s_coarse_max_abs=_s_coa_max,
            x2_reverb_for_erle=_x2_reverb_for_erle,
            capture_psd_erle=_capture_psd_erle,
            erle_coh_gate_mask=_coh_gate_mask,
        )
        _a1_aec3_initial_state = self._aec3_state.initial_state_active()
        _a1_transition_triggered = self._aec3_state.transition_triggered()
        self._diag['aec3_initial_state_active'] = bool(_a1_aec3_initial_state)
        self._diag['aec3_transition_triggered'] = bool(_a1_transition_triggered)
        self._diag['usable_linear_estimate'] = bool(
            self._aec3_state.usable_linear_estimate())
        # A transition clears the suppressor's initial state.
        if _a1_transition_triggered and self._aec3_sg is not None:
            self._aec3_sg.set_initial_state(False)
            self._diag['a1_set_initial_state'] = 'off'
        if (self.filter is not None
                and hasattr(self.filter, 'H_error_per_bin')):
            self._diag['h_error_mean'] = float(
                np.mean(self.filter.H_error_per_bin))
        else:
            self._diag['h_error_mean'] = 0.0

    def _post_estimate_residual(self, far_psd, near_psd, echo_psd):
        # Stationarity is updated once per hop upstream; read the converge flag from the shared counter.
        _filter_converged_enough = (
            self._aec3_stationarity_active_hops
            >= self._aec3_stationarity_converge_hops
        )

        dominant_ne = self._aec3_sg.is_dominant_nearend()

        # Lazy-bind the decay estimator (needs n_partitions and hop_size from the filter).
        self._aec3_ree.attach_reverb_decay_estimator(
            n_partitions=int(self.filter.n_partitions),
            hop_size=int(self.filter.hop_size),
            use_aec3_block_energy=False,
        )
        # Per-partition |W|^2 spectra (n_partitions, n_freqs).
        _w_mag2 = (np.abs(self.filter.W) ** 2).astype(np.float32)
        # Filter delay in partitions from AecState.min_direct_path_filter_delay() (0 is the pre-convergence headroom guess).
        _delay_blocks = int(self._aec3_state.min_direct_path_filter_delay())
        # linear_filter_quality from the fullband ERLE estimator: continuous in [0, 1] with a ~400 ms hold, so the reverb tail refreshes past per-frame convergence loss.
        _filter_q = self._aec3_state.get_inst_linear_quality_estimate()
        _stationary_block = self._aec3_stationarity.is_block_stationary()
        _just_reset_active = False
        self._aec3_just_reset_active = False
        _effective_usable_linear = bool(self._aec3_state.usable_linear_estimate())
        self._diag['aec3_just_reset_active'] = False
        self._diag['aec3_effective_usable_linear'] = _effective_usable_linear
        # ReverbDecayEstimator runs on partition-summed |W|^2; no time-domain filter is passed.
        _td_filter_for_decay = None
        self._aec3_ree.update_reverb_models(
            frequency_response=_w_mag2,
            filter_delay_blocks=_delay_blocks,
            filter_quality=_filter_q,
            usable_linear_filter=_effective_usable_linear,
            stationary_block=_stationary_block,
            time_domain_filter=_td_filter_for_decay,
        )

        r2, r2_unb = self._aec3_ree.estimate(
            aec_state=self._aec3_state,
            render_psd=far_psd,
            capture_psd=near_psd,
            s2_linear=echo_psd,
            dominant_nearend=dominant_ne,
            filter_delay_blocks=_delay_blocks,
            filter_length_blocks=int(getattr(self.filter, 'n_partitions', 0)),
            force_nonlinear_path=_just_reset_active,
        )

        # Reverb tail dead-streak tracking: tail-dead means max(tail_response) <= 0.
        _reverb_fr_now = getattr(self._aec3_ree, '_reverb_freq_resp', None)
        _reverb_tail_max_now = 0.0
        if (_reverb_fr_now is not None
                and hasattr(_reverb_fr_now, 'tail_response')):
            _reverb_tail_max_now = float(
                np.max(_reverb_fr_now.tail_response))
        return r2, r2_unb, _filter_converged_enough, _effective_usable_linear

    def _post_stationary_residual(self, r2, r2_unb, _filter_converged_enough):
        # Stationarity-driven R^2 scaling: zero R^2 / R^2_unb on stationary bands once the filter has had time to converge,
        # so a constant far-end hum does not damp near-end speech. Controlled by
        # SuppressorConfig.echo_audibility.use_stationarity_properties.
        _use_stationarity = bool(
            self._aec3_sg_config.echo_audibility.use_stationarity_properties)
        _need_stationary_mask = (_use_stationarity and _filter_converged_enough)
        _stationary_mask = (
            self._aec3_stationarity.band_stationary_mask()
            if _need_stationary_mask else None
        )
        if (_use_stationarity
                and _filter_converged_enough
                and _stationary_mask is not None
                and np.any(_stationary_mask)):
            r2 = np.where(_stationary_mask, 0.0, r2).astype(np.float32)
            r2_unb = np.where(
                _stationary_mask, 0.0, r2_unb
            ).astype(np.float32)
        return r2, r2_unb, _stationary_mask

    def _post_observer_residual(self, near_end, _sel_esw, _sel_echo_spec, _PSD_SCALE, _stationary_mask, r2):
        # Live low-frequency observer (AecConfig.live_observer_*). Its learning is gated read-only: a live-adapting
        # echo estimator must not learn on double-talk hops, and is not gated on the live dt-protect flag.
        if getattr(self.config, 'live_observer_enabled', False):
            _lo_trustworthy = bool(
                self._render_activity.is_active
                and self._aec3_state.usable_linear_estimate()
                and not self._aec3_sg.is_dominant_nearend()
            )
            _lo_mu_scale = 1.0 if _lo_trustworthy else 0.0
            # Diagnostics: the lagged DNE verdict folded into _lo_trustworthy.
            self._diag['lo_trustworthy'] = bool(_lo_trustworthy)
            self._diag['lo_dne_lagged'] = bool(self._aec3_sg.is_dominant_nearend())
            if self._observer_far_end is not None:
                # Injection guard: trust the observer's evidence only while its LF residual is no worse than the main
                # filter's on the same hop (both use the selected linear output that R^2 is built from), and only
                # while _lo_trustworthy, so a sharp onset where is_dominant_nearend() just flipped cannot inject.
                self._observer_filter.process(
                    near_end, self._observer_far_end, mu_scale=_lo_mu_scale)
                _kmax = self._observer_max_bin
                _e_obs = float(np.sum(
                    np.abs(self._observer_filter.error_spec[:_kmax]) ** 2))
                _e_main = float(np.sum(np.abs(_sel_esw[:_kmax]) ** 2))
                _a = self._observer_err_ema_alpha
                self._observer_obs_err_ema = (
                    _a * self._observer_obs_err_ema + (1.0 - _a) * _e_obs)
                self._observer_main_err_ema = (
                    _a * self._observer_main_err_ema + (1.0 - _a) * _e_main)
                _corroborated = (self._observer_obs_err_ema
                                  <= self._observer_corrob_margin
                                  * self._observer_main_err_ema)
                self._diag['lo_err_ratio'] = (
                    self._observer_obs_err_ema / self._observer_main_err_ema
                    if self._observer_main_err_ema > 0.0 else float('inf'))
                self._diag['lo_corroborated'] = bool(_corroborated)
                # Injection-only near-end discriminator: the continuous dt_from_energy (not the hysteresis-latched
                # dt-protect flag) separates false-positive injections that the corroboration margin cannot; the
                # observer's own learning stays on the lagged-DNE gate.
                _lo_ne_evidence = (float(self._dt_from_energy)
                                    > self._observer_inject_dt_energy_max)
                _lo_injected = bool(_corroborated and _lo_trustworthy
                                     and not _lo_ne_evidence)
                self._diag['lo_injected'] = _lo_injected
                if _lo_injected:
                    _lo_diff = (np.abs(self._observer_filter.echo_spec
                                        - _sel_echo_spec) ** 2).astype(np.float32)
                    _lo_cap = (np.abs(_sel_esw) ** 2).astype(np.float32)
                    _lo_ev = (np.minimum(_lo_diff, _lo_cap)
                              * _PSD_SCALE).astype(np.float32)
                    if _kmax < _lo_ev.shape[0]:
                        _lo_ev[_kmax:] = 0.0
                    if _stationary_mask is not None:
                        _lo_ev = np.where(_stationary_mask, 0.0, _lo_ev).astype(np.float32)
                    r2 = np.maximum(r2, _lo_ev).astype(np.float32)
        return r2

    def _post_nearend_power(self, error_psd, near_psd):
        # nearend spectrum = E^2 (clamped to Y^2) when the linear estimate is usable, else Y^2; a just-reset gate
        # treats it as unusable.
        if (self._aec3_state.usable_linear_estimate()
                and not getattr(self, "_aec3_just_reset_active", False)):
            nearend_pwr = np.minimum(error_psd, near_psd).astype(np.float32)
        else:
            nearend_pwr = near_psd
        return nearend_pwr

    def _post_comfort_noise(self, near_psd, _near_psd_win):
        # Comfort noise: N2 tracked from the raw capture PSD (near_psd = |Y|^2 * _PSD_SCALE in int16^2), so the
        # suppressor's near-end / SNR ratios share the noise reference with the CN injection. With
        # cng_n2_windowed_psd (default off) the tracker input is the windowed capture PSD instead.
        _cng_input_psd = (
            _near_psd_win
            if getattr(self.config, "cng_n2_windowed_psd", False)
            else near_psd
        )
        _saturated_capture = (self._saturation_level > 0.5)
        if not self._aec3_noise_initialized:
            self._aec3_y2_smoothed = _cng_input_psd.copy().astype(np.float32)
            self._aec3_noise_initialized = True

        if not _saturated_capture:
            _y2_alpha = self._aec3_cng_y2_alpha
            self._aec3_y2_smoothed = (
                self._aec3_y2_smoothed
                + _y2_alpha * (_cng_input_psd - self._aec3_y2_smoothed)
            ).astype(np.float32)

            # N2 update after warm-up: below Y2_smoothed, track down fast and slow up; otherwise slow up only
            # (wall-clock rescales of the per-block 0.9 / 0.1 and 1.0002 constants).
            if self._aec3_n2_counter > self._aec3_cng_n2_update_onset_hops:
                _below = self._aec3_y2_smoothed < self._aec3_n2
                _fresh = self._aec3_cng_n2_track_freshness
                _retain = self._aec3_cng_n2_track_retention
                _g = self._aec3_cng_n2_slow_up
                _track = (
                    _fresh * self._aec3_y2_smoothed + _retain * self._aec3_n2
                ) * _g
                _up = self._aec3_n2 * _g
                self._aec3_n2 = np.where(_below, _track, _up).astype(np.float32)

            # N2_initial transient over the rescaled 1000-block window: it moves slowly toward N2 from above and
            # follows N2 from below; N2 takes over on the release frame.
            _dur = self._aec3_cng_n2_initial_duration_hops
            if self._aec3_n2_counter < _dur:
                self._aec3_n2_counter += 1
                if self._aec3_n2_counter < _dur:
                    _above = self._aec3_n2 > self._aec3_n2_initial
                    _ia = self._aec3_cng_n2_initial_alpha
                    _slow = self._aec3_n2_initial + _ia * (
                        self._aec3_n2 - self._aec3_n2_initial
                    )
                    self._aec3_n2_initial = np.where(
                        _above, _slow, self._aec3_n2
                    ).astype(np.float32)

            np.maximum(
                self._aec3_n2, self._aec3_noise_floor_int16sq,
                out=self._aec3_n2,
            )
            if self._aec3_n2_counter < _dur:
                np.maximum(
                    self._aec3_n2_initial,
                    self._aec3_noise_floor_int16sq,
                    out=self._aec3_n2_initial,
                )

        # N2_initial during the transient window, then N2; used for the suppressor's ENR / SNR ratios and the CN injection.
        comfort_noise = (
            self._aec3_n2_initial
            if self._aec3_n2_counter < self._aec3_cng_n2_initial_duration_hops
            else self._aec3_n2
        )
        return comfort_noise

    def _post_suppression_gain(self, nearend_pwr, r2, r2_unb, comfort_noise, render_block_scaled, _stationary_mask):
        # DT-gated min-gain floor lift. The near-recent latch is the product policy. With dt_protect_source == 'dne'
        # the suppressor's own dominant-near-end detector is read instead; its verdict is the previous hop's
        # (get_gain() below advances it), a deliberate one-hop delay. A separate counter keeps the energy rule's
        # state for dt_aware_recovery_soft.
        if getattr(self.config, 'dt_protect_source', 'energy') == 'dne':
            _dne_prev = bool(self._aec3_sg.is_dominant_nearend())
            # Confirmation delay: _dne_prev must read True for more than _dt_protect_confirm_hops hops (0 = the first).
            if _dne_prev:
                self._dne_confirm_counter = getattr(self, '_dne_confirm_counter', 0) + 1
            else:
                self._dne_confirm_counter = 0
            _dne_confirmed = _dne_prev and self._dne_confirm_counter > self._dt_protect_confirm_hops
            if _dne_confirmed:
                self._dt_protect_frames = self._dt_protect_extra_hold_hops
            elif getattr(self, '_dt_protect_frames', 0) > 0:
                self._dt_protect_frames -= 1
            else:
                self._dt_protect_frames = 0
            _protect = _dne_confirmed or self._dt_protect_frames > 0
        else:
            _protect = getattr(self, '_ne_recent_frames', 0) > 0
        self._aec3_sg._dt_protect_active = bool(
            getattr(self.config, 'dt_aware_res_floor_enabled', False)
            and _protect)
        gain = self._aec3_sg.get_gain(
            aec_state=self._aec3_state,
            nearend_spectrum=nearend_pwr,
            residual_echo_spectrum=r2,
            residual_echo_spectrum_unbounded=r2_unb,
            comfort_noise_spectrum=comfort_noise,
            render_block=render_block_scaled,
            clock_drift=False,
            stationary_mask=_stationary_mask,
        )
        return gain

    def _post_stash_context(self, error_spec, _sel_echo_spec, gain, comfort_noise, r2):
        self._res_error_spec = error_spec.astype(np.complex64, copy=True)
        self._res_echo_spec = _sel_echo_spec.astype(np.complex64, copy=True)
        self._res_near_spec = (
            error_spec + _sel_echo_spec
        ).astype(np.complex64, copy=True)
        self._res_formed_output = np.asarray(
            self._form_prev_output_time, dtype=np.float32
        ).copy()
        self._res_gain = gain.astype(np.float32, copy=True)
        self._res_comfort_noise = comfort_noise.astype(np.float32, copy=True)
        # Residual-echo PSD R^2 (int16^2-scaled, like comfort_noise), so an external NR can fold echo into its noise floor.
        self._res_r2 = np.asarray(r2, dtype=np.float32).copy()

    def _post_output_spectrum(self, error_spec, _sel_echo_spec, _effective_usable_linear, gain):
        # When the linear filter is unusable the base can switch to the windowed capture Y = error_spec + _sel_echo_spec.
        _out_base = error_spec
        if (getattr(self.config, "output_capture_when_linear_unusable", False)
                and not _effective_usable_linear):
            _y_base = (error_spec + _sel_echo_spec).astype(error_spec.dtype,
                                                           copy=False)
            # Fall back to the capture Y only when the residual is worse than the mic (diverged over-output, |E| > |Y|),
            # per frame; this bounds a rare diverged frame.
            if (float(np.sum(np.abs(error_spec) ** 2))
                    > float(np.sum(np.abs(_y_base) ** 2))):
                _out_base = _y_base
        e_out_spec = _out_base * gain.astype(error_spec.dtype, copy=False)
        return e_out_spec

    def _post_add_comfort_noise(self, e_out_spec, n_bins, _PSD_SCALE, comfort_noise, gain):
        # Comfort-noise injection: noise_gain[k] = sqrt(1 - G[k]^2); CN_re/im[k] = sqrt(N2[k]) * sqrt(2) sin(idx);
        # E[k] += noise_gain[k] * CN[k]; DC and Nyquist bins are zero. At 16 kHz there is only the lowest band, so the
        # upper-band 0.4 scaling does not apply.
        if self.config.enable_cng:
            _n_random = n_bins - 2
            _seed = int(self._aec3_cng_seed)
            _re_idx = np.empty(_n_random, dtype=np.int32)
            _im_idx = np.empty(_n_random, dtype=np.int32)
            for _k in range(_n_random):
                _seed = (_seed * 69069 + 1) & 0x7FFFFFFF
                _ix = _seed >> 26   # top 5 bits, 0..31
                _re_idx[_k] = _ix
                _im_idx[_k] = (_ix + 8) & 31
            self._aec3_cng_seed = _seed

            # N2 is int16^2; divide by _PSD_SCALE before the sqrt to match e_out_spec.
            _N_float = np.sqrt(
                np.maximum(comfort_noise / _PSD_SCALE, 0.0)
            ).astype(np.float32)
            _cn_re = np.zeros(n_bins, dtype=np.float32)
            _cn_im = np.zeros(n_bins, dtype=np.float32)
            _cn_re[1:-1] = _N_float[1:-1] * self._aec3_sqrt2_sin_lut[_re_idx]
            _cn_im[1:-1] = _N_float[1:-1] * self._aec3_sqrt2_sin_lut[_im_idx]

            _noise_gain = np.sqrt(
                np.maximum(1.0 - gain.astype(np.float32) ** 2, 0.0)
            ).astype(np.float32)

            _cng_spec = (_noise_gain * (_cn_re + 1j * _cn_im)).astype(np.complex64)
            e_out_spec = e_out_spec + _cng_spec
        return e_out_spec

    def _post_synthesize(self, e_out_spec):
        e_out_full = np.fft.irfft(e_out_spec, n=self.filter.fft_size).astype(np.float32)
        bs = self.filter.block_size
        hop = self.filter.hop_size
        windowed = e_out_full[:bs] * self._aec3_synth_window
        self._aec3_ola_buf += windowed
        out = self._aec3_ola_buf[:hop].copy()
        self._aec3_ola_buf[:-hop] = self._aec3_ola_buf[hop:]
        self._aec3_ola_buf[-hop:] = 0.0
        return out.astype(np.float32)


def process_wav_files(mic_path: str, ref_path: str, out_path: str,
                      config: Optional[AecConfig] = None, diag: bool = False):
    """Process WAV files through AEC"""
    mic_data, mic_sr = sf.read(mic_path)
    ref_data, ref_sr = sf.read(ref_path)

    if mic_sr != ref_sr:
        raise ValueError(f"Sample rate mismatch: mic={mic_sr}, ref={ref_sr}")

    if mic_data.ndim > 1:
        mic_data = mic_data[:, 0]
    if ref_data.ndim > 1:
        ref_data = ref_data[:, 0]

    num_samples = min(len(mic_data), len(ref_data))
    mic_data = mic_data[:num_samples].astype(np.float32)
    ref_data = ref_data[:num_samples].astype(np.float32)

    print(f"AEC Processing:")
    print(f"  Microphone: {mic_path} ({num_samples} samples)")
    print(f"  Reference:  {ref_path}")
    print(f"  Sample rate: {mic_sr} Hz")
    print(f"  Duration: {num_samples / mic_sr:.2f} seconds")

    if config is None:
        config = AecConfig(sample_rate=mic_sr)
    else:
        # Auto fields resolved for another sample rate would stay stale: revert them to the sentinel
        # and re-run __post_init__.
        if config.sample_rate != mic_sr:
            from dataclasses import replace as _dc_replace
            config = _dc_replace(config,
                                  sample_rate=mic_sr,
                                  frame_size=-1,
                                  hop_size=-1,
                                  filter_length=-1)

    print(f"  Mode: {config.mode.value}")
    print(f"  Step size (mu): {config.mu}")
    print(f"  Filter length: {config.filter_length} samples ({1000 * config.filter_length / config.sample_rate:.1f} ms)")
    print(f"  RES: {'enabled' if config.enable_res else 'disabled'}")
    print()

    aec = AEC(config)
    hop_size = aec.hop_size

    output = np.zeros(num_samples, dtype=np.float32)
    processed = 0
    max_erle = 0.0

    while processed + hop_size <= num_samples:
        mic_block = mic_data[processed:processed + hop_size]
        ref_block = ref_data[processed:processed + hop_size]

        out_block = aec.process(mic_block, ref_block)
        output[processed:processed + hop_size] = out_block

        erle = aec.get_erle()
        max_erle = max(max_erle, erle)
        processed += hop_size

        if processed % (mic_sr // 2) == 0:
            if diag:
                d = aec.get_diagnostics()
                t = processed / mic_sr
                g_mean_db = 20 * np.log10(max(d['res_gain_mean'], 1e-10))
                g_min_db = 20 * np.log10(max(d['res_gain_min'], 1e-10))
                eff_gmin_db = 20 * np.log10(max(d['effective_g_min'], 1e-10))
                print(f"[{t:5.1f}s] ERLE={d['erle_inst']:6.1f}dB mu={d['mu_scale']:.2f} "
                      f"far_act={d['far_activity']:.2f} "
                      f"g_mean={g_mean_db:5.1f}dB g_min={g_min_db:5.1f}dB "
                      f"eff_gmin={eff_gmin_db:5.1f}dB "
                      f"conv={'Y' if d['converged'] else 'N'} "
                      f"div={d['divergence']:.2f}")
            else:
                print(f"  Processed: {processed / mic_sr:.1f} s, ERLE: {erle:.1f} dB\r",
                      end='', flush=True)

    print(f"\n\nResults:")
    print(f"  Processed samples: {processed}")
    print(f"  Max ERLE: {max_erle:.1f} dB")

    # fp32 PCM output (lossless for float input).
    sf.write(out_path, output, mic_sr, subtype='FLOAT')
    print(f"\nOutput written to: {out_path}")


def main():
    parser = argparse.ArgumentParser(
        description='Acoustic Echo Cancellation (AEC)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Filter modes:
    lms     - Time-domain LMS (simplest, fixed step size, mu~0.01)
    nlms    - Time-domain NLMS (normalized, default)
    fdaf    - Frequency-domain Adaptive Filter (single FFT block, no partitions)
    pbfdaf  - Partitioned Block FDAF (NLMS adaptation, multiple partitions)
    pbfdkf  - Partitioned Block FDKF (Kalman adaptation, recommended)
    subband - Alias for pbfdkf (backward compatibility)

Examples:
    python aec.py mic.wav ref.wav output.wav
    python aec.py mic.wav ref.wav output.wav --mode pbfdkf --enable-res
    python aec.py mic.wav ref.wav output.wav --mode pbfdkf --enable-res --preset balanced
    python aec.py mic.wav ref.wav output.wav --mu 0.5 --filter 1024
        """
    )
    parser.add_argument('mic', help='Microphone input WAV file')
    parser.add_argument('ref', help='Reference/loudspeaker WAV file')
    parser.add_argument('output', help='Output WAV file')
    parser.add_argument('--mu', type=float, default=0.3, help='Step size (default: 0.3)')
    parser.add_argument('--filter', type=int, default=0,
                        help='Filter length in samples (default: mode-dependent)')
    parser.add_argument('--mode', choices=['fdaf', 'pbfdaf', 'pbfdkf', 'subband'],
                        default='pbfdkf', help='Filter mode (default: pbfdkf)')
    # BooleanOptionalAction with default=None: the CLI overrides preset values only when the flag is passed.
    parser.add_argument('--enable-res', default=None,
                        action=argparse.BooleanOptionalAction,
                        help='Enable RES post-filter (default: from preset, else off)')
    parser.add_argument('--res-g-min', type=float, default=-20.0, help='RES min gain (dB)')
    parser.add_argument('--cng', default=None,
                        action=argparse.BooleanOptionalAction,
                        help='Enable comfort noise generation in RES (default: from preset, else off)')
    parser.add_argument('--no-td-constraint', action='store_true',
                        help='Disable time-domain constraint on filter weights (diagnostic)')
    from preset_arg import PRESET_METAVAR, parse_preset_arg
    parser.add_argument('--preset', type=parse_preset_arg, metavar=PRESET_METAVAR,
                        help='Preset: 0 mild / 1 balanced / 2 aggressive / 3 no_floor '
                             '(names also accepted).')
    parser.add_argument('--no-shadow', action='store_true', help='Disable shadow filter')
    parser.add_argument('--no-delay-est', action='store_true',
                        help='Disable online delay estimation (mirrors C aec_wav --no-delay-est)')
    parser.add_argument('--no-highpass', action='store_true', help='Disable high-pass filter')
    parser.add_argument('--highpass-cutoff', type=float, default=80.0,
                        help='High-pass filter cutoff frequency in Hz (default: 80)')
    parser.add_argument('--no-saturation-detect', action='store_true',
                        help='Disable saturation/clipping detection')
    parser.add_argument('--clear-history', action='store_true',
                        help='Clear TIME/LMS buffer each block (no carry-over)')
    parser.add_argument('--diag', action='store_true',
                        help='Print per-second diagnostic output (ERLE, gains, etc.)')

    args = parser.parse_args()

    # Map the mode string to the enum.
    mode_map = {
        'fdaf': AecMode.FDAF,
        'pbfdaf': AecMode.PBFDAF,
        'pbfdkf': AecMode.PBFDKF,
        'subband': AecMode.PBFDKF,  # alias of pbfdkf
    }

    aec_mode = mode_map[args.mode]

    # Mode-dependent default step size.
    mu = args.mu
    if args.mode == 'fdaf' and args.mu == 0.3:
        mu = 0.1  # FDAF single-block: smaller mu avoids overshoot

    # filter_length: auto (-1) or user-specified.
    filter_length = args.filter
    if filter_length == 0:
        filter_length = -1  # auto, resolved in __post_init__

    common_kw = dict(
        mu=mu,
        filter_length=filter_length,
        mode=aec_mode,
        enable_td_constraint=not args.no_td_constraint,
        enable_shadow=not args.no_shadow,
        enable_highpass=not args.no_highpass,
        highpass_cutoff_hz=args.highpass_cutoff,
        enable_saturation_detect=not args.no_saturation_detect,
        clear_filter_history=args.clear_history,
    )
    # Keep the preset floor unless --res-g-min differs from -20 dB; without a preset use the CLI value.
    if not args.preset or args.res_g_min != -20.0:
        common_kw['min_gain_floor_far_active_db'] = args.res_g_min
    # Forward enable_res / enable_cng only when set on the CLI; otherwise the preset (or AecConfig) value stays.
    if args.enable_res is not None:
        common_kw['enable_res'] = args.enable_res
    if args.cng is not None:
        common_kw['enable_cng'] = args.cng
    if args.no_delay_est:
        common_kw['enable_delay_est'] = False
    if args.preset:
        config = AecConfig.from_preset(args.preset, **common_kw)
    else:
        config = AecConfig(**common_kw)

    process_wav_files(args.mic, args.ref, args.output, config, diag=args.diag)
