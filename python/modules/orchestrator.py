"""AEC orchestrator — top-level engine class + WAV pipeline + CLI main.

Extracted from ``aec.py`` during refactor R.11. ~3,500 lines for the
``AEC`` class itself, plus ``process_wav_files`` and ``main()`` for the
file-based CLI.

aec.py becomes a thin shim that re-imports everything from this module
plus the prior R.3-R.10 modules so existing callers
(``from aec import AEC, AecConfig, ...``) keep working unchanged.
"""
import os
import argparse
import numpy as np
from collections import deque
from typing import List, Optional, Tuple
import soundfile as sf

# Lazy holders for v3.18 Phase C.A / C.B audit modules; populated on
# first flag-ON construction so flag-OFF AEC instances skip the scipy
# import cost.
_FilterAnalyzer = None
_FilteringQualityAnalyzer = None

# F3.1-v3 blend weight (mic-excess-ratio vs legacy 1-coh²) — kept here
# because the dt_per_bin path lives inside AEC.
_BLEND_F31_MIC_EXCESS = 0.7

from .enums import (
    AecMode, AecPreset, AecFilterState, _FREQ_MODES, _PB_MODES,
)
from .dataclasses import (
    AecStats, AecResContext, RenderActivityState, FilterConvergenceState,
    RegimeHandlerDecision, AecEventType, AecEvent, EpcEvent,
)
from .delay.legacy_compat import LegacyDelayShim as DelayEstimator
from .erle import (
    FilterErleEstimator, FullbandErleEstimator, compute_erle_confidence,
)
from .preprocessing import HighPassFilter, SaturationDetector
from .filters import NlmsFilter, PBFDAF, PBFDKF
from .dtd import DtdEstimator
from .detectors import (
    RenderActivityDetector, FilterConvergenceAnalyzer,
    DoubleTalkAnalyzer, FilterPlateauDetector,
)
from .epc import (
    classify_epc_event, EchoPathChangeDetector, PathChangeRegimeHandler,
)
from .residual_estimator import ResidualEchoEstimator
from .config import AecConfig
from .nlp import SubtractiveNLP
from .debug_logger import AecDebugLogger


class AEC:
    """
    Acoustic Echo Cancellation

    Supports five filter modes:
    - NLMS:    Time-domain NLMS (sample-by-sample processing)
    - FDAF:    Frequency-domain Adaptive Filter (single FFT block, n_partitions=1)
    - PBFDAF:  Partitioned Block FDAF (NLMS adaptation)
    - PBFDKF:  Partitioned Block FDKF (Kalman adaptation, recommended)
    """

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

    # ── Phase C2 ablation knob: disable convergence reset on EPC sources ─────
    # Set is one of: {'delay_first', 'delay_shift', 'epv', 'shadow_rise'} or empty.
    # Used by _maybe_mark_diverged() to skip mark_diverged calls per-source.
    _epc_no_reset_sources: frozenset = frozenset()

    def _arc_m_q_boost(self, filt) -> None:
        """v3.15 §1.4 Arc M: set Q = Q_high.copy(), optionally per-band-tilted.

        When Arc M is enabled (`arc_m_epc_gated=True` in cfg + `kalman_q_per_band`
        also True so the band scale was stored), the EPC rising-edge Q
        boost gets a per-band scale applied to the freshly-copied Q array.
        After `_p_max_override_frames` countdown, q_scale modulation in
        `_update_weights` decays Q back toward baseline behavior — but Q
        itself stays at the tilted value until the next time something
        explicitly resets it.

        When the flag is OFF or `_arc_m_band_scale` was never stored,
        behaviour is identical to legacy `filt.Q = filt.Q_high.copy()`
        (byte-equal).
        """
        if not hasattr(filt, 'Q_high'):
            return
        filt.Q = filt.Q_high.copy()
        _scale = getattr(filt, '_arc_m_band_scale', None)
        if _scale is not None:
            filt.Q = filt.Q * _scale

    def _maybe_mark_diverged(self, source: str) -> None:
        if source not in self._epc_no_reset_sources:
            self._convergence.mark_diverged()
        # Round 3 trace: per-source counter (audio-passive)
        if not hasattr(self, '_round3_div_counts'):
            self._round3_div_counts = {'delay_first': 0, 'delay_shift': 0,
                                       'epv': 0, 'shadow_rise': 0}
        self._round3_div_counts[source] = self._round3_div_counts.get(source, 0) + 1
        self._round3_last_div_source = source

    def _f_e3_handle_epc_fire(self, source: str) -> None:
        """F-E3 consecutive-EPC handler. Called from EPC fire sites; checks
        whether this fire is within the consecutive window of a prior fire,
        and if so applies (a) hangover extension to ≥1s and (b) gated W
        partial reset (gap-guarded against cohort-tail spam).

        Always resets `_frames_since_last_epc` to 0; the caller must invoke
        this regardless of whether f_e3_enabled to keep the counter live.
        """
        if (self.config.f_e3_enabled
                and self._frames_since_last_epc
                    < self.config.f_e3_consecutive_window_frames):
            # E3-1: extend hangover to ≥1s
            min_hangover = self.config.f_e3_consecutive_window_frames
            if self._epc_det._hangover < min_hangover:
                self._epc_det._hangover = min_hangover
            # E3-3: gap-guarded W partial reset (cohort-tail defence)
            if (self._frames_since_last_f_e3_w_reset
                    >= self.config.f_e3_w_reset_min_gap_frames):
                factor = float(self.config.f_e3_w_reset_factor)
                for filt in [self.filter, self.shadow_filter]:
                    if filt is not None and hasattr(filt, 'W'):
                        filt.W *= factor
                self._frames_since_last_f_e3_w_reset = 0
        self._frames_since_last_epc = 0

    def _apply_epc_state_reset(self, source: str) -> None:
        """F2.1: reset stale upstream state on EPC rising edge.

        Restores `_erl_estimate`, the windowed ERLE accumulators, and the
        DT-jump baseline to their post-`__init__` values. The shipped EPC
        code already caps `_erl_estimate` to min(0.3) and boosts Q/P
        floors; this method completes the picture for state that has
        decay constants on the order of 10 s and otherwise persists
        across a path change. Gated by `config.use_epc_state_reset`;
        callers must check the flag before invoking.

        `source` is one of {'epv', 'shadow_rise'} for tracing; the reset
        body is identical because both trigger types invalidate the same
        upstream state by the same mechanism (room/path discontinuity).
        """
        self._erl_estimate = 0.1
        self._erle_window_near = 1e-10
        self._erle_window_err = 1e-10
        self._wn_err_baseline = 1e-8
        # Telemetry: count per-source firings so audit can correlate with
        # bench deltas. Audio-passive; never read by hot path.
        if not hasattr(self, '_f2_1_reset_counts'):
            self._f2_1_reset_counts = {'epv': 0, 'shadow_rise': 0}
        self._f2_1_reset_counts[source] = self._f2_1_reset_counts.get(source, 0) + 1

    def _handle_gain_change_soft(self, source: str) -> None:
        """v3.18 Phase F.2 — AEC3-aligned soft reset for gain_change events.

        Mirrors `webrtc::Subtractor::HandleEchoPathChange` gain-only branch
        (subtractor.cc:170-174): only `refined_gains_->HandleEchoPathChange()`
        runs, i.e. the adaptive step-size gain receives a boost so the
        filter re-tracks faster — filter weights / partition / coarse
        filter / aec_state ERLE all preserved.

        Our equivalent: Q-boost on main + shadow (preserves weights);
        no Kalman P relax, no filter-derived-state reset, no ERL cap.

        Wired by F.3 behind `aec_event_classification_enabled` for
        EpcEvent.source ∈ {'epv', 'shadow_rise'}.
        """
        for filt in [self.filter, self.shadow_filter]:
            if filt is not None and hasattr(filt, 'Q'):
                self._arc_m_q_boost(filt)
        self._f_e3_handle_epc_fire(source)

    def _handle_delay_change_full(self, source: str) -> None:
        """v3.18 Phase F.2 — AEC3-aligned full reset for delay_change events.

        Mirrors `webrtc::Subtractor::HandleEchoPathChange` delay branch
        (subtractor.cc:148-168 `full_reset`): filter weights zeroed
        (refined + coarse), refined/coarse gains reset to initial,
        partition size reset to initial. Plus aec_state full cascade
        in aec_state.cc.

        Our equivalent: Q-boost + Kalman P relax (30 frames) + Kalman
        P-floor lift (30 frames) + ERL cap + filter-derived-state reset
        + EPC render forced. Matches today's `delay_shift` site exactly.

        Wired by F.3 behind `aec_event_classification_enabled` for
        EpcEvent.source == 'delay'.
        """
        for filt in [self.filter, self.shadow_filter]:
            if filt is not None and hasattr(filt, 'Q'):
                self._arc_m_q_boost(filt)
                # Kalman P-override attrs are PBFDKF-only. When
                # shadow_class_nlms=True, the shadow filter is PBFDAF and
                # has no Kalman state — skip the P-override block on it.
                # Guard is filter-type, not flag: keeps PBFDKF byte-equal.
                if isinstance(filt, PBFDKF):
                    filt._p_max_override = 1.0
                    filt._p_max_override_frames = 30
                    filt._p_floor_beta = 1.0
                    filt._p_floor_beta_frames = 30
        self._maybe_mark_diverged(source)
        self._epc_render_forced_remaining = self.config.epc_hangover
        self._erl_estimate = min(self._erl_estimate, 0.3)
        if self.config.use_epc_state_reset:
            self._apply_epc_state_reset(source)
        self._f_e3_handle_epc_fire(source)

    # v3.18 Phase B.2/B.3 — Filter misadjustment estimator + ScaleFilter wiring.
    def _update_misadjustment_estimator(self) -> None:
        """Asymmetric EMA on filter_scale_ratio = echo_psd / (far_psd × ERL).

        Tracks long-term W magnitude drift. Slow-up / fast-down EMA biases
        the smoothed value toward sustained under-modelling, not transients.
        Skipped when far is silent (no observation signal).
        """
        if not self.config.filter_misadjustment_enabled:
            return
        if self.filter is None or not hasattr(self.filter, 'echo_spec'):
            return
        _fp_echo = float(np.mean(np.abs(self.filter.echo_spec) ** 2))
        _fp_far  = float(np.mean(np.abs(self.filter.far_spec) ** 2))
        if _fp_far < 1e-10:
            return
        raw_ratio = _fp_echo / (_fp_far * float(self._erl_estimate) + 1e-12)
        if raw_ratio < self._misadjustment_smoothed:
            alpha = self.config.filter_misadjustment_alpha_up
        else:
            alpha = self.config.filter_misadjustment_alpha_dn
        self._misadjustment_smoothed = (
            alpha * self._misadjustment_smoothed
            + (1.0 - alpha) * raw_ratio
        )

    def _check_and_apply_misadjustment_scale(self) -> None:
        """Trigger gate + ScaleFilter fire per AEC3 IsAdjustmentNeeded.

        Gates: filter converged AND refined_usable AND not epc_active AND
        not main_paused AND stable_count ≥ threshold AND hangover == 0.
        On fire: scale = clamp(1.0 / smoothed, scale_min, scale_max);
        filter.scale_filter(scale, scale_p); smoothed→1.0; arm hangover.
        """
        if not self.config.filter_misadjustment_enabled:
            return
        if self._misadjustment_hangover_remaining > 0:
            self._misadjustment_hangover_remaining -= 1
            return
        # v3.21 Phase B.5: AEC3-aligned always-on gate. Drop the
        # `refined_usable` requirement that v3.18 B.3 added — AEC3's
        # IsAdjustmentNeeded fires whenever the ratio is below threshold
        # regardless of AecState (subtractor.cc:239-249). The remaining
        # guards (epc_active / main_paused / converged) prevent firing
        # during transients where the ratio is dominated by misalignment
        # rather than steady-state under-modelling.
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
        if self._misadjustment_smoothed >= self.config.filter_misadjustment_threshold:
            return
        # Fire: rescale W (and optionally P).
        proposed_scale = 1.0 / max(self._misadjustment_smoothed, 1e-6)
        scale = max(self.config.filter_misadjustment_scale_min,
                    min(self.config.filter_misadjustment_scale_max,
                        proposed_scale))
        if isinstance(self.filter, PBFDKF):
            self.filter.scale_filter(scale,
                scale_p=self.config.filter_misadjustment_scale_p)
        else:
            self.filter.scale_filter(scale)
        self._misadjustment_smoothed = 1.0
        self._misadjustment_hangover_remaining = (
            self.config.filter_misadjustment_hangover_frames)
        self._misadjustment_fire_count += 1

    # ── EPC-state delegations (state lives in self._epc_det) ─────────────────
    @property
    def epc_active(self) -> bool: return self._epc_det.active
    @property
    def epc_hangover_count(self) -> int: return self._epc_det.hangover_count
    @property
    def _epv_gain_fast(self) -> float: return self._epc_det.epv_gain_fast
    @property
    def _epv_gain_slow(self) -> float: return self._epc_det.epv_gain_slow

    # Per-mode optimal mu defaults (tuned on fileid_0/1/2)
    _MODE_DEFAULT_MU = {
        AecMode.LMS: 0.02,
        AecMode.NLMS: 0.4,
        AecMode.FDAF: 0.3,
        AecMode.PBFDAF: 0.5,
        AecMode.PBFDKF: 0.5,
        AecMode.SUBBAND: 0.5,
    }

    def __init__(self, config: Optional[AecConfig] = None):
        self.config = config or AecConfig()

        # Apply per-mode default mu if user didn't override
        if self.config.mu == AecConfig.mu:  # still at dataclass default
            self.config.mu = self._MODE_DEFAULT_MU.get(self.config.mode, 0.3)

        # Delay estimation + reference alignment
        if self.config.enable_delay_est or self.config.fixed_delay_samples >= 0:
            max_delay_samp = int(self.config.max_delay_ms * self.config.sample_rate / 1000)
            # v3.10.0: ring buffer sized to delay_buffer_ms (default 1024 ms,
            # matching WebRTC AEC3 kRenderTransferQueueSizeFrames=1000 ms).
            # The +4096 below is legacy headroom; we keep it on top of the
            # configured buffer to absorb hop-boundary alignment.
            buffer_samp = int(self.config.delay_buffer_ms * self.config.sample_rate / 1000)
            buffer_samp = max(buffer_samp, max_delay_samp + 4096)
            if self.config.fixed_delay_samples >= 0:
                buffer_samp = max(buffer_samp, self.config.fixed_delay_samples + 4096)
                self.delay_est = None
                self._current_delay = self.config.fixed_delay_samples
            else:
                # v3.21 Phase A.1: swap legacy GCC-PHAT for AEC3-aligned
                # RenderDelayController (matched filter bank + histogram
                # aggregator + clockdrift). LegacyDelayShim exposes the
                # legacy attribute surface so existing call sites continue
                # working unchanged.
                self.delay_est = DelayEstimator(
                    sample_rate=self.config.sample_rate,
                    hop_size=self.config.hop_size,
                    # Legacy kwargs accepted as no-op for call-site compat:
                    max_delay_ms=self.config.max_delay_ms,
                    init_seconds=self.config.delay_est_init_s,
                    period_seconds=self.config.delay_est_period_s,
                    par_low_threshold=self.config.delay_par_low_threshold,
                    par_solid_threshold=self.config.delay_par_solid_threshold,
                    trace=getattr(self.config, "trace_delay_est", False),
                    fast_path_enabled=getattr(self.config,
                                              "delay_fast_path_enabled", False),
                    fast_par_threshold=getattr(self.config,
                                               "delay_fast_par_threshold", 40.0),
                )
                self._current_delay = -1  # -1 = not yet estimated
            # Reference ring buffer for delay compensation
            self._ref_ring = np.zeros(buffer_samp, dtype=np.float32)
            self._ref_ring_write = 0
            self._ref_ring_size = len(self._ref_ring)
            self._ref_ring_filled = 0  # Total samples written (for warmup)
            self._delay_active = True
        else:
            self.delay_est = None
            self._delay_active = False

        # Default-init mode-divergent attributes so every code path that
        # reads `self._dtd_fft_size` (set inside the FDAF-buffering block)
        # also works for PBFDKF/PBFDAF/SUBBAND/LMS/TIME. Was a latent
        # AttributeError surfaced by CLI smoke on BALANCED.
        self._dtd_fft_size = 0

        # Create adaptive filter based on mode
        if self.config.mode in _FREQ_MODES:
            if self.config.mode == AecMode.FDAF:
                # Classic FDAF: single FFT block, n_partitions=1
                # block_size = 2 × filter_length, hop = filter_length
                # Faster than time-domain LMS (O(N log N) vs O(N²)) with same result.
                # Limitation: large hop → slow update rate → motivates partitioned approach.
                desired = 2 * self.config.filter_length
                block_size = 256
                while block_size < desired:
                    block_size *= 2
                n_partitions = 1
                self._internal_hop = block_size // 2
            else:
                # Partitioned block (PBFDAF/PBFDKF/SUBBAND)
                # block_size = 2 × hop (proper overlap-save, 50% TD constraint)
                # FFT size determined inside PBFDAF as next_pow2(block_size)
                hop_size = self.config.hop_size
                block_size = 2 * hop_size
                n_partitions = max(1, (self.config.filter_length + hop_size - 1) // hop_size)
                self._internal_hop = hop_size

            # FilterClass: mode determines adaptation algorithm
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
                hop_size=self._internal_hop
            )
            self.filter.enable_td_constraint = self.config.enable_td_constraint
            self._hop_size = self.config.hop_size
            self._n_partitions = n_partitions

            # PBFDKF: apply config Q_high/Q_low
            if isinstance(self.filter, PBFDKF):
                self.filter.Q_high[:] = self.config.kalman_q_high
                self.filter.Q_low[:]  = self.config.kalman_q_low
                self.filter.Q[:] = self.config.kalman_q_high
                # v3.15 §1.6 Arc F: per-band Q schedule (default OFF).
                # Tilt Q_high/Q_low across LF/MF/HF bands using same band
                # boundaries (1k / 4k Hz) as Arc P / Arc R.  When OFF, the
                # uniform fill above stands → byte-equal to v3.14.
                if self.config.kalman_q_per_band:
                    _freq_res = self.config.sample_rate / (2 * (self.filter.n_freqs - 1))
                    _b1k = max(1, min(int(round(1000.0 / _freq_res)),
                                       self.filter.n_freqs - 2))
                    _b4k = max(_b1k + 1, min(int(round(4000.0 / _freq_res)),
                                              self.filter.n_freqs - 1))
                    _lf, _mf, _hf = self.config.kalman_q_band_scales
                    _scale = np.ones(self.filter.n_freqs, dtype=np.float32)
                    _scale[:_b1k] = float(_lf)
                    _scale[_b1k:_b4k] = float(_mf)
                    _scale[_b4k:] = float(_hf)
                    if self.config.arc_m_epc_gated:
                        # Arc M: keep baseline Q uniform (cohort tail safe);
                        # store the per-band scale for transient application
                        # at EPC rising edges. Default Q_high stays uniform.
                        self.filter._arc_m_band_scale = _scale
                    else:
                        # Arc F (standalone, time-invariant): apply tilt now.
                        self.filter.Q_high *= _scale
                        self.filter.Q_low  *= _scale
                        self.filter.Q[:] = self.filter.Q_high
                # P53 Step 0: enable innovation-audit hook from config.
                self.filter._enable_p53_trace = bool(
                    getattr(self.config, 'trace_p53_innovation', False))

            # FDAF buffering (when internal_hop > external hop)
            if self.config.mode == AecMode.FDAF and self._internal_hop > self._hop_size:
                # Buffer large enough for accumulation (internal_hop + one extra external hop)
                buf_size = self._internal_hop + self._hop_size
                self._freq_near_queue = np.zeros(buf_size, dtype=np.float32)
                self._freq_far_queue = np.zeros(buf_size, dtype=np.float32)
                self._freq_out_buf = np.zeros(buf_size, dtype=np.float32)
                self._freq_out_valid = 0  # valid output samples remaining
                self._freq_out_read = 0
                self._freq_queue_write = 0
                # DTD independent buffer: FL-point FFT with hop=FL/2
                # Decouples coherence DTD from FDAF's larger block_size
                fl = self.config.filter_length
                self._dtd_fft_size = fl
                self._dtd_hop = fl // 2
                self._dtd_err_buf = np.zeros(fl, dtype=np.float32)
                self._dtd_far_buf = np.zeros(fl, dtype=np.float32)
                self._dtd_acc_err = np.zeros(fl // 2, dtype=np.float32)
                self._dtd_acc_far = np.zeros(fl // 2, dtype=np.float32)
                self._dtd_acc_pos = 0
            else:
                self._freq_near_queue = None
                self._dtd_fft_size = 0
        elif self.config.mode == AecMode.LMS:
            # LMS: Time-domain, no normalization
            self.filter = NlmsFilter(
                filter_length=self.config.filter_length,
                mu=self.config.mu,
                delta=self.config.delta,
                normalize=False
            )
            self.filter.clear_history = self.config.clear_filter_history
            self._hop_size = self.config.hop_size
            self._internal_hop = self.config.hop_size
            self._n_partitions = 0
            self._freq_near_queue = None
        else:
            # TIME: Time-domain NLMS
            self.filter = NlmsFilter(
                filter_length=self.config.filter_length,
                mu=self.config.mu,
                delta=self.config.delta,
                normalize=True
            )
            self.filter.clear_history = self.config.clear_filter_history
            self._hop_size = self.config.hop_size
            self._internal_hop = self.config.hop_size
            self._n_partitions = 0
            self._freq_near_queue = None

        # DTD: frequency-domain modes only (divergence + coherence dual detector)
        # LMS/NLMS have no effective DTD — all methods (Geigel, NCC, coherence,
        # VSS-NLMS) either don't work for AEC or cause vicious cycles with slow
        # convergence. Output Limiter provides the safety net instead.
        if self.config.enable_dtd and self.config.mode in _FREQ_MODES:
            # Warmup: 50 DTD invocations before coherence starts.
            # FDAFDTD runs every dtd_hop/hop_size external frames,
            # so 50 DTD invocations = 50 * dtd_hop/hop_size external frames.
            warmup = 50
            self.dtd_divergence = DtdEstimator(
                mode='divergence',
                divergence_factor=self.config.dtd_divergence_factor,
                attack=self.config.dtd_confidence_attack,
                release=self.config.dtd_confidence_release,
                warmup_frames=warmup,
            )
            # FDAF: FL-point FFT (matches filter length, hop=FL/2)
            # PBFDAF/PBFDKF: use filter's own spectra (block_size from filter)
            if self._dtd_fft_size > 0:
                dtd_block_size = self._dtd_fft_size
            else:
                dtd_block_size = self.filter.block_size
            coh_n_freqs = dtd_block_size // 2 + 1
            self.dtd_coherence = DtdEstimator(
                mode='coherence',
                n_freqs=coh_n_freqs,
                coh_alpha=self.config.dtd_coh_alpha,
                coh_high=self.config.dtd_coh_high,
                coh_low=self.config.dtd_coh_low,
                coh_energy_floor=self.config.dtd_coh_energy_floor,
                coh_abs_floor=self.config.dtd_coh_abs_floor,
                hangover_max=self.config.dtd_coh_hangover,
                attack=self.config.dtd_confidence_attack,
                release=self.config.dtd_coh_release,
                warmup_frames=warmup,
                sample_rate=self.config.sample_rate,
                block_size=dtd_block_size,
            )
        else:
            self.dtd_divergence = None
            self.dtd_coherence = None

        # Legacy ResFilter retired in v3.21. The AEC3 chain (modules/state +
        # modules/residual + modules/filter + modules/render) replaces the
        # 9-stage ResFilter pipeline. self.res preserved as None so any
        # external caller still introspecting AEC.res sees a stable None.
        self.res = None

        # v3.21 AEC3-aligned post-stage chain. The linear filter (PBFDKF)
        # produces an error spectrum which the AEC3 chain (AecState +
        # ResidualEchoEstimator + SuppressionGain + CNG) consumes; the
        # legacy self.res.process() call is bypassed in process().
        self._aec3_state = None
        self._aec3_ree = None
        self._aec3_sg = None
        self._aec3_ola_buf = None
        self._aec3_synth_window = None
        # Pending EchoPathVariability accumulated by legacy event detectors
        # (EPV / shadow_rise / delay subsystem) and consumed at next
        # _aec3_post call (BEFORE aec_state.update() per AEC3 contract).
        # AEC3 dispatch pattern mirrored from echo_remover.cc; detection
        # source is our legacy stack because AEC3 has no internal
        # gain_change detector (its gain_change comes from external
        # level_change input on EchoCanceller3::ProcessCapture).
        self._aec3_pending_gain_change = False
        self._aec3_pending_delay_change = None  # None = no event; else DelayAdjustment
        if self.filter is not None:
            from .state import AecState as _Aec3State, AecStateConfig as _Aec3StateConfig
            from .residual import ResidualEchoEstimator, SuppressionGain
            n_bins = int(self.filter.n_freqs)
            # TransparentMode requires AEC3's SubtractorOutputAnalyzer (not yet
            # ported) to feed a per-frame "any_filter_converged" pulse. Our
            # legacy FilterConvergenceAnalyzer is a hard 10-frame >5 dB ERLE
            # latch which permanently sits at False on hard cases like 9xjhi;
            # that makes TM falsely activate after 6s of strong render -> kills
            # usable_linear -> R^2 collapses to nonlinear path forever. Disable
            # TM until the proper analyzer ports.
            self._aec3_state = _Aec3State(_Aec3StateConfig(
                n_bins=n_bins,
                enable_transparent_mode=False,
            ))
            self._aec3_ree = ResidualEchoEstimator(n_bins=n_bins)
            self._aec3_sg = SuppressionGain(n_bins=n_bins)
            # Synthesis OLA: sqrt-Hann analysis * sqrt-Hann synthesis = Hann,
            # which sums to 1 across 50%-overlap hops (perfect reconstruction).
            bs = int(self.filter.block_size)
            self._aec3_synth_window = np.sqrt(np.hanning(bs)).astype(np.float32)
            self._aec3_ola_buf = np.zeros(bs, dtype=np.float32)
            # v3.21 Phase C.1 — CNG state for `_aec3_post`. Mirrors the
            # production legacy CNG (res_refactored/noise_floor_cng.py:55-62):
            # per-bin noise PSD EMA + per-bin target CN gain smoothing.
            self._aec3_noise_psd = np.zeros(n_bins, dtype=np.float32)
            self._aec3_smooth_cn_gain = np.zeros(n_bins, dtype=np.float32)
            self._aec3_noise_initialized = False
            pass  # AEC3 chain init scope

        # v3.21 Phase C.3 — StationarityEstimator.
        # Detects per-bin stationary render (constant hum / fan / line noise).
        # Two consumers:
        #   1. _aec3_post: zeros R² on stationary bands.
        #   2. filter.process: skips W update when block-stationary so
        #      PBFDKF doesn't learn mic-as-echo coupling against
        #      uncorrelated stationary noise (E0l0 / wJVP NE outliers).
        from .state.stationarity_estimator import StationarityEstimator as _StatEst
        if self.filter is not None and hasattr(self.filter, 'n_freqs'):
            self._aec3_stationarity = _StatEst(
                n_freqs=int(self.filter.n_freqs),
                hop_samples=int(getattr(self.config, 'hop_size', 160)),
                sample_rate=int(getattr(self.config, 'sample_rate', 16000)),
            )
        else:
            self._aec3_stationarity = None
        # AEC3 IsRenderTooLow threshold (echo_audibility.cc:112) — peak
        # amplitude of 10 in int16 = 10/32768 = 3.05e-4 in float[-1,1].
        self._aec3_non_zero_render_seen = False
        self._AEC3_RENDER_PEAK_FLOOR = 10.0 / 32768.0
        # AEC3 filter_has_had_time_to_converge = strong_not_saturated_render
        # blocks >= 0.8 × kNumBlocksPerSecond (aec_state.cc:111-113).
        self._aec3_stationarity_active_hops = 0
        _hop_st = int(getattr(self.config, 'hop_size', 160))
        _sr_st = int(getattr(self.config, 'sample_rate', 16000))
        self._aec3_stationarity_converge_hops = int(round(0.8 * _sr_st / _hop_st))

        # v3.21 Phase B.3 + B.4 — RenderSignalAnalyzer + startup gates.
        # RSA tracks per-bin narrow-band tonal regions in the render history
        # and produces (a) a per-bin mask for refined-filter gain compute
        # and (b) a poor_signal_excitation flag that freezes the W update
        # entirely. Mirrors AEC3 render_signal_analyzer.cc. The matched
        # filter delay (Phase A.1) gives us delay-aligned X² access for the
        # detector; we use the current-frame |far|² as a first-order proxy
        # while the delay is locking in.
        from .render.render_signal_analyzer import RenderSignalAnalyzer as _RSA
        from . import aec3_scale as _aec3_scale
        if hasattr(self.filter, 'n_freqs') and hasattr(self.filter, 'n_partitions'):
            self._render_signal_analyzer = _RSA(
                n_freqs=self.filter.n_freqs,
                strong_peak_freeze_duration=self.filter.n_partitions,
            )
            # Wire RSA into refined filter for the per-bin mask + initialise
            # its startup counter from the hop-scaled default.
            self.filter._render_signal_analyzer = self._render_signal_analyzer
            self.filter._poor_excitation_counter = _aec3_scale.blocks_to_hops(
                1000, self.config.hop_size, self.config.sample_rate
            )
        else:
            self._render_signal_analyzer = None

        # Shadow filter (dual-filter, frequency-domain modes only)
        # Can be used alone (≈ WebRTC/SpeexDSP) or with DTD (dual protection)
        self.shadow_filter = None
        self.main_err_smooth = 0.0
        self.shadow_err_smooth = 0.0
        if (self.config.enable_shadow and
                self.config.mode in _FREQ_MODES
                and hasattr(self.filter, 'W')):
            # v3.18 Phase A.2 — shadow class selection.
            # Flag-OFF (default): shadow uses same class as main (PBFDKF in
            # BALANCED). Flag-ON: shadow uses PBFDAF (NLMS), AEC3-aligned.
            if self.config.shadow_class_nlms:
                ShadowClass = PBFDAF
                shadow_mu = self.config.shadow_mu_nlms
            else:
                ShadowClass = FilterClass
                shadow_mu = self.config.mu * self.config.shadow_mu_ratio
            self.shadow_filter = ShadowClass(
                block_size=self.filter.block_size,
                n_partitions=self.filter.n_partitions,
                mu=shadow_mu,
                delta=self.config.delta,
                hop_size=self.filter.hop_size
            )
            self.shadow_filter.enable_td_constraint = self.config.enable_td_constraint
            # PBFDKF shadow: higher Q via ratio for faster tracking.
            # When shadow_class_nlms=True the shadow is PBFDAF (no Q state),
            # so this scaling is skipped — guard is filter-type, not flag.
            if isinstance(self.shadow_filter, PBFDKF):
                self.shadow_filter.Q_high = self.filter.Q_high * self.config.shadow_q_ratio
                self.shadow_filter.Q_low  = self.filter.Q_low  * self.config.shadow_q_ratio
                self.shadow_filter.Q      = self.shadow_filter.Q_high.copy()

        # Echo path change detector (owns active/hangover/EPV-EMAs/prev_total_err)
        self._epc_det = EchoPathChangeDetector(self.config)
        # v3.18 Phase F.1 — latest classified AEC3-aligned event (trace-only
        # in F.1; consumers wired in F.2+). Empty AecEvent when classification
        # flag is OFF or no event fired this frame.
        self._classified_event = AecEvent()

        # v3.18 Phase B.2 — Filter misadjustment estimator state.
        # Lazy-init only when flag is enabled so flag-OFF stays byte-equal
        # (these attrs simply don't exist on baseline-config AEC instances).
        if self.config.filter_misadjustment_enabled:
            self._misadjustment_smoothed = 1.0
            self._misadjustment_stable_count = 0
            self._misadjustment_hangover_remaining = 0
            self._misadjustment_fire_count = 0
            # v3.19 Phase 3 — reset_done counter for fq_usable gate.
            # Increments per frame when no recent reset event; reset to
            # 0 on epc_active / main_paused / leakage_diverged_fired.
            # Used only when filter_misadjustment_use_fq_usable=True.
            self._misadjustment_reset_done_count = 0

        # v3.18 Phase C.A — FilterAnalyzer (audit-only). Lazy-init guards
        # both module import and instantiation so flag-OFF stays byte-equal.
        self._filter_analyzer = None
        if self.config.filter_analyzer_enabled:
            global _FilterAnalyzer
            if _FilterAnalyzer is None:
                from modules.filter_analyzer import FilterAnalyzer as _FA
                _FilterAnalyzer = _FA
            self._filter_analyzer = _FilterAnalyzer(
                sample_rate=self.config.sample_rate)

        # v3.18 Phase C.B — FilteringQualityAnalyzer (audit-only). Same
        # lazy-init pattern as C.A.
        self._filter_quality = None
        if self.config.filter_quality_enabled:
            global _FilteringQualityAnalyzer
            if _FilteringQualityAnalyzer is None:
                from modules.filter_quality import FilteringQualityAnalyzer as _FQA
                _FilteringQualityAnalyzer = _FQA
            self._filter_quality = _FilteringQualityAnalyzer()
        # Per-frame helper: set True inside any EPC fire site this frame.
        # Read by FilteringQualityAnalyzer; ignored when flag is OFF.
        self._epc_reset_fired_this_frame = False

        # #4: Confidence memory decay
        self.prev_dtd_conf = 0.0
        self._dtd_conf_holdoff = 0  # F2.5: frames remaining in hold phase

        # Filter convergence + divergence-indicator (extracted to FilterConvergenceAnalyzer).
        # Backward-compat reads via @property below.
        self._convergence = FilterConvergenceAnalyzer()
        # EPC render-forced countdown (Change D)
        self._epc_render_forced_remaining = 0
        # Dynamic ERL estimate for render-based echo (B4)
        self._erl_estimate = 0.1  # initial -20dB, conservative
        # Per-band ERL EMA broadcast to filter._erl_per_bin. Stays at uniform
        # 0.1 in v3.21 (the Arc G EMA update that would mutate this lived in
        # the post-filter loop that was retired with ResFilter).
        self._per_band_erl = np.array([0.1, 0.1, 0.1], dtype=np.float64)
        # Double-talk analyzer (owns _dt_from_energy / _dt_from_shadow / _shadow_advantage)
        self._dt_analyzer = DoubleTalkAnalyzer(self.config)

        # v3.10.0 — filter plateau detector (one-shot recovery for
        # DT-from-frame-0 cases where main filter learned NE leak in the
        # first ~100 frames and is now stuck below convergence threshold).
        self._plateau_detector = FilterPlateauDetector()

        # P3e — DT advisory gate state. Hold counter is in samples so we
        # can convert dt_advisory_hold_ms once. Diag fields exposed via _diag.
        self._dt_advisory_hold_remaining = 0  # frames; decremented per process()
        _hop = self._hop_size if hasattr(self, '_hop_size') else (
            self.config.hop_size if self.config.hop_size > 0 else self.config.frame_size // 2)
        self._dt_advisory_hold_frames = max(
            1, int(self.config.dt_advisory_hold_ms * self.config.sample_rate / 1000.0 / max(1, _hop)))

        # P3f — Mini AecState trace (no behaviour change). post_reset_age_frames
        # is incremented per process() and zeroed by _reset_filter_derived_state.
        # erle_inst ring buffer kept for slope estimation (~500 ms window at
        # hop=160 / 16 kHz = 50 frames).
        self._post_reset_age_frames = 0
        _slope_n = max(2, int(0.5 * self.config.sample_rate / max(1, _hop)))
        self._erle_slope_buf = deque(maxlen=_slope_n)
        self._p3f_refined_latched = False  # latches true once refined_usable seen
        self._p3f_main_err_baseline = 0.0  # EMA baseline for jump detection
        self._p3f_diverged_streak = 0      # consecutive frames over diverged TH
        # B6 — previous-frame filter_state cache for state-aware shadow_mu.
        # Filter state classifier runs at end of process(); shadow µ schedule
        # at start needs the previous frame's value.
        #
        # NOTE (v3.15 §B6, 2026-05-15): `_prev_filter_state` is the
        # INTERNAL P3f-string state machine (values: 'idle', 'startup',
        # 'diverged', 'suspicious_dt', 'refined_usable', 'coarse_learning').
        # It is distinct from the PUBLIC `AecFilterState` enum (which
        # has values like CONVERGED / WARMUP / EPC_RECOVERY) returned by
        # `get_filter_state()`.  The two state systems serve different
        # consumers — see B2 docblock at AecStats.filter_state and the
        # B4 fix at aec.py:6361 for the load-bearing distinction.
        self._prev_filter_state: str = 'idle'
        # v3.13 E4.S3 — SubtractiveNLP detector (audit-only).
        # Per docs/v3_13_e4_s2_design_lock.md. Outputs nl_confidence per
        # hop into self._diag; does NOT modify output. Pure observer.
        self.nl_detector: Optional[SubtractiveNLP] = None
        if getattr(self.config, 'e4_nlp_enabled', False):
            self.nl_detector = SubtractiveNLP(
                sample_rate=self.config.sample_rate,
                hop_size=self.hop_size,
                window_ms=getattr(self.config, 'e4_nlp_window_ms', 32.0),
                pitch_threshold=getattr(self.config, 'e4_nlp_pitch_threshold', 0.45),
                continuity_frames=getattr(self.config, 'e4_nlp_continuity_frames', 3),
                min_residual_rms=getattr(self.config, 'e4_nlp_min_residual_rms', 0.05),
                cancellation_ratio_threshold=getattr(
                    self.config, 'e4_nlp_cancel_ratio_threshold', 2.0),
                cancellation_ema_alpha=getattr(
                    self.config, 'e4_nlp_cancel_ema_alpha', 0.99),
            )
        # F-E1 — far_active hysteresis state (fast attack / slow release).
        # Once far crosses 1e-4 it stays "active" until 5 consecutive frames
        # below 3e-5. Stabilises ERL update gating across brief power dips.
        self._f_e1_far_active = False
        self._f_e1_far_release_count = 0
        # F-DelayTrack — recent delay-estimate history for variance check.
        # Bounded deque; appended only when delay_est emits a valid estimate.
        self._delay_history = deque(maxlen=8)
        # F-E3 — consecutive-EPC tracking. Counters reset on respective fires;
        # large initial value means "never fired".
        self._frames_since_last_epc = 10**9
        self._frames_since_last_f_e3_w_reset = 10**9
        # F-E5 — saturation hysteresis: track previous frame's sat level so
        # we can fast-attack reset _error_psd on the sat → clean transition.
        self._f_e5_prev_sat_level = 0.0
        # F2.2 EMA tracker — always maintained (cheap), only consumed by P3h
        # reset gate when `use_diverged_streak_ema` is True.
        self._p3f_diverged_streak_ema = 0.0
        # P3h — sustained-diverged reset cooldown. Decremented per frame;
        # reset action gated on cooldown == 0.
        self._p3h_reset_cooldown_remaining = 0
        self._p3h_reset_count = 0           # diagnostic: reset fires this run

        # S-orth.A — shadow decoupled state (initialised here regardless of flag;
        # only *used* when shadow_state_decoupled=True so flag-OFF is byte-equal).
        # These are initialised to the same values as PBFDKF.__init__ so that
        # after enabling the flag the first frame starts from the same state.
        _n_freqs = (self.filter.n_freqs
                    if hasattr(self.filter, 'n_freqs') else 1)
        self._shadow_error_psd = np.ones(_n_freqs, dtype=np.float32) * 1e-2
        self._shadow_R = np.ones(_n_freqs, dtype=np.float32) * 1e-2
        self._shadow_mu_holdoff = 0   # independent of main's _simple_mu_holdoff

        # Windowed decaying ERLE accumulator for erle_factor (TC ≈ 10s)
        self._erle_window_near = 1e-10
        self._erle_window_err = 1e-10
        self._erle_factor_prev = 0.0  # Previous frame's erle_factor for shadow DTD weight

        # Smoothed inst ERLE for dt_indicator correction (~3 frame / 30ms)
        self._inst_erle_smooth = 1.0

        # Per-bin mu_scale (updated from RES echo_psd/error_psd each frame)
        self._per_bin_mu_scale = None  # None = use scalar fallback

        # P1 Phase 1 trace state: ring buffer of high-band mic power for
        # modulation CV^2 metric. 32 frames @ hop=160 @ 16kHz ≈ 320 ms.
        self._hb_mic_pwr_ring = np.zeros(32, dtype=np.float32)
        self._hb_mic_pwr_idx = 0
        self._hb_mic_pwr_n = 0  # filled-count (≤ 32)

        # Diagnostic tracking (per-frame, latest values)
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
            # DT / filter debug fields
            'dt_indicator': 0.0,        # final DT confidence fed to RES
            'main_err_smooth': 0.0,     # main filter error EMA
            'shadow_err_smooth': 0.0,   # shadow filter error EMA
            'main_paused': False,       # True = main filter weights frozen this frame
            'epv_gain_ratio': 1.0,      # EPV fast/slow gain ratio (!=1 → gain change)
            'dt_residual_scale': 1.0,   # scaling on echo_spec passed to RES
            'filter_w_norm': 0.0,       # main filter L2 weight norm
            'shadow_w_norm': 0.0,       # shadow filter L2 weight norm
            'copy_err_baseline': 1e-6,  # copy gate error baseline
        }

        # Output limiter: smoothed gain to avoid frame-boundary clicking
        self._limiter_gain = 1.0
        # v3.21 NE fix — when routing through the AEC3 chain, `_aec3_post`
        # output is OLA-lagged by 1 hop, so comparing same-frame near_peak
        # to out_peak miscalibrates the limiter (it fires on speech-silence
        # transitions where loud OLA reconstruction lands in a quiet hop).
        # Buffer one hop of mic so the limiter compares against the SOURCE
        # frame for `final_output`, not the current frame. Sized lazily on
        # first use.
        self._limiter_near_lag = None

        # High-pass filter (DC blocker + low-freq removal)
        if self.config.enable_highpass:
            self._hp_mic = HighPassFilter(self.config.highpass_cutoff_hz, self.config.sample_rate)
            if self.config.enable_highpass_ref:
                self._hp_ref = HighPassFilter(self.config.highpass_cutoff_hz, self.config.sample_rate)
            else:
                self._hp_ref = None
        else:
            self._hp_mic = None
            self._hp_ref = None

        # Saturation detector (non-linear echo handling)
        if self.config.enable_saturation_detect:
            self._sat_detector_ref = SaturationDetector(self.config.saturation_threshold)
            self._sat_detector_mic = SaturationDetector(self.config.saturation_threshold)
        else:
            self._sat_detector_ref = None
            self._sat_detector_mic = None
        self._saturation_level = 0.0

        # Far-end activity + stationarity detector (extracted from inline EMA logic)
        self._render_activity = RenderActivityDetector()
        self._stat_far_hangover = 0
        self._inst_erle_smooth = 1.0
        self._wn_err_baseline = 1e-8
        self._stat_dt_hangover = 0  # Stationary DT hold-off counter (frames)

        # Simple variable mu (for non-DTD modes, inspired by Valin 2007 RER)
        self._simple_mu_ratio = 1.0
        self._simple_mu_holdoff = 0  # holdoff counter: blocks release for N frames
        self._warmup_frames = self.config.warmup_frames
        self._warmup_far_active = False  # only consume warmup when far-end is active

        # Shadow divergence detection state (WebRTC-style: pause + Q boost, no output switch)
        self.shadow_frame_count = 0
        self._regime_handler = PathChangeRegimeHandler(
            self.config,
            gate_mode=getattr(self.config, 'regime_gate_mode', 'energy'),
        )
        self._last_raw_output: Optional[np.ndarray] = None   # raw filter output before RES (diagnostic)
        # EchoPathVariability EMAs moved into EchoPathChangeDetector (self._epc_det)
        # Legacy AecState aggregator (modules.legacy_state.AecState) retired in
        # v3.21 — its 5-detector facade is fully replaced by direct property
        # access on AEC (self._filter_converged / self.epc_active / etc.) and
        # by the AEC3 chain's own AecState (modules.state.aec_state.AecState,
        # bound to self._aec3_state below).
        self._far_power_ema = 0.0           # TC≈50ms for GetStats()
        self._mic_power_ema = 0.0
        self._frame_count = 0               # frames since reset()

        # P52 A.0R.2: per-frame regime handler trace rows. Only appended to
        # when self.config.trace_p52_regime_handler is True (default False
        # → list stays empty → zero memory overhead).
        self._regime_trace_rows = []

        # ERLE (raw = filter-only, final = post-RES)
        self.near_power = 0.0
        self.error_power = 0.0  # backward compat alias for raw
        self.raw_error_power = 0.0
        self.final_error_power = 0.0
        self.alpha = 0.95
        # Cumulative ERLE (full-segment average)
        self.near_power_sum = 0.0
        self.error_power_sum = 0.0  # backward compat alias for raw
        self.raw_error_power_sum = 0.0
        self.final_error_power_sum = 0.0
        # _conv_counter moved to FilterConvergenceAnalyzer (self._convergence)

        # DTD confidence history (one entry per process() call)
        self.confidence_history = deque(maxlen=1000)

    def reset(self):
        self.filter.reset()
        if self.shadow_filter:
            self.shadow_filter.reset()
            self.main_err_smooth = 0.0
            self.shadow_err_smooth = 0.0
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
        self.prev_dtd_conf = 0.0
        self._dtd_conf_holdoff = 0
        self._convergence.reset()
        # v3.10.0: clear plateau-detector counters on AEC reset
        if hasattr(self, '_plateau_detector'):
            self._plateau_detector.reset()
        # P3e: clear DT advisory hold
        self._dt_advisory_hold_remaining = 0
        self._erle_window_near = 1e-10
        self._erle_window_err = 1e-10
        self._erle_factor_prev = 0.0
        self._inst_erle_smooth = 1.0
        self._simple_mu_ratio = 1.0
        self._simple_mu_holdoff = 0
        # S-orth.A: reset shadow decoupled state on full AEC.reset()
        if hasattr(self, '_shadow_error_psd'):
            self._shadow_error_psd.fill(1e-2)
        if hasattr(self, '_shadow_R'):
            self._shadow_R.fill(1e-2)
        # v3.18 Phase C.A — FilterAnalyzer state reset on full AEC reset.
        if getattr(self, '_filter_analyzer', None) is not None:
            self._filter_analyzer.reset()
        # v3.18 Phase C.B — FilteringQualityAnalyzer state reset.
        if getattr(self, '_filter_quality', None) is not None:
            self._filter_quality.reset()
        self._epc_reset_fired_this_frame = False
        if hasattr(self, '_shadow_mu_holdoff'):
            self._shadow_mu_holdoff = 0
        self._warmup_frames = self.config.warmup_frames
        self._warmup_far_active = False
        # v3.8.1: clear lazy-getattr diagnostic counters so cross-case batch
        # eval doesn't leak prior-case state into next-case stats interpretation.
        # Diagnostics-only — does not affect audio output.
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
        # _epc_det reset above already cleared its EPV EMAs and prev_total_err
        self._far_power_ema = 0.0
        self._mic_power_ema = 0.0
        self._frame_count = 0
        if self.dtd_divergence:
            self.dtd_divergence.reset()
        if self.dtd_coherence:
            self.dtd_coherence.reset()
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
        self.final_error_power = 0.0
        self.near_power_sum = 0.0
        self.error_power_sum = 0.0
        self.raw_error_power_sum = 0.0
        self.final_error_power_sum = 0.0
        # v3.10.3 — clear cross-case lazy state
        if hasattr(self, '_pending_delay'):
            del self._pending_delay
        for _attr in ('_round3_div_counts', '_round3_last_div_source',
                      '_r7_prev_delay', '_r7_prev_div_counts',
                      '_dominant_nearend_hold'):
            if hasattr(self, _attr):
                delattr(self, _attr)
        # _conv_counter is owned by self._convergence (reset above)
        if self._hp_mic is not None:
            self._hp_mic.reset()
        if self._hp_ref is not None:
            self._hp_ref.reset()
        if self._sat_detector_ref is not None:
            self._sat_detector_ref.reset()
            self._sat_detector_mic.reset()
        self._saturation_level = 0.0
        self._render_activity.reset()
        self._stat_far_hangover = 0
        self._inst_erle_smooth = 1.0
        self._wn_err_baseline = 1e-8
        self._stat_dt_hangover = 0  # Stationary DT hold-off counter (frames)
        self._limiter_gain = 1.0
        self._per_bin_mu_scale = None
        if self._dtd_fft_size > 0:
            self._dtd_acc_pos = 0
            self._dtd_acc_err.fill(0)
            self._dtd_acc_far.fill(0)
            self._dtd_err_buf.fill(0)
            self._dtd_far_buf.fill(0)
        # Reset DT signals (now owned by DoubleTalkAnalyzer)
        self._dt_analyzer.reset()
        self._epc_render_forced_remaining = 0
        self._erl_estimate = 0.1
        # Reset per-band ERL EMA to initial conservative value.
        self._per_band_erl[:] = 0.1

    def _reset_filter_derived_state(self, reason: str = 'plateau',
                                     preserve_render_ema: bool = True) -> None:
        """v3.10.2 — clear all filter-output-derived state. Generic helper
        called by both plateau recovery (`reason='plateau'`) and first delay
        acquisition (`reason='delay_first'`). Earlier versions of these
        recovery paths reset only filter/shadow/res, leaving the freshly
        reset filter to learn against poisoned downstream state
        (main_err_smooth / DTD / EPC hangover / ERLE windows all still
        carrying values from the bad taps).

        Both recovery paths share the same shape: filter taps were trained
        against incorrect input (misaligned ref for delay_first; NE leak
        for plateau), and all derived state from those taps is now wrong.

        Differs from full AEC.reset() in that this preserves time-axis +
        input-side context:

          PRESERVED (input-side / temporal context):
            • _frame_count                 — elapsed time
            • delay_est + _current_delay   — delay alignment lives upstream
            • _ref_ring (delay buffer)     — far history is valid
            • _plateau_detector            — its own attempts counter
            • _far_power_ema / _mic_power_ema  — input-side
            • _hp_mic / _hp_ref            — input-side HPF
            • _sat_detector_*              — input-side
            • _render_activity             — input-side
            • RES long-window far-PSD EMA — input-side (when
              preserve_render_ema=True; default). The EMA is updated every
              far-active frame regardless of mode, so its accumulated
              long-term render spectrum is independent of the bad taps.
              Discarding it forces the freshly reset filter through 100
              frames of pre-warmup-fallback all over again.

          CLEARED (filter-output-derived; would otherwise re-poison the
          freshly reset filter):
            • filter / shadow_filter taps via .reset()
            • _convergence + _epc_det
            • main_err_smooth / shadow_err_smooth
            • _dt_analyzer (energy / shadow DT histories)
            • _erle_window_* / _inst_erle_smooth / _erle_factor_prev
            • _simple_mu_ratio / _simple_mu_holdoff / _per_bin_mu_scale
            • _epc_render_forced_remaining / _erl_estimate
            • prev_dtd_conf
            • DTD divergence + coherence smoothed PSDs
            • RES post-filter state (gain_smooth / echo_psd / noise_psd /
              gates). Long-window far-PSD EMA optionally preserved.
            • shadow_frame_count + _regime_handler
            • Diagnostic _diag dict (would otherwise show stale stats)
        """
        # Filter taps + shadow
        self.filter.reset()
        if self.shadow_filter is not None:
            self.shadow_filter.reset()

        # Filter convergence + EPC + divergence DTD + coherence DTD
        self._convergence.reset()
        self._epc_det.reset()
        if self.dtd_divergence is not None:
            self.dtd_divergence.reset()
        if self.dtd_coherence is not None:
            self.dtd_coherence.reset()

        # Filter-output power / smoother / err quantities
        self.main_err_smooth = 0.0
        self.shadow_err_smooth = 0.0
        self.error_power = 0.0
        self.raw_error_power = 0.0
        self.final_error_power = 0.0
        self.error_power_sum = 0.0
        self.raw_error_power_sum = 0.0
        self.final_error_power_sum = 0.0
        # v3.10.3 — near_power EMA must be reset alongside error_power, otherwise
        # get_erle_inst() = near_power / error_power transiently spikes (stale
        # mic EMA / fresh tiny error) and could mis-trigger early convergence.
        # near_power is sample-loop EMA (alpha=0.999) so it recovers in ~10 frames.
        self.near_power = 0.0
        self.near_power_sum = 0.0

        # ERLE accumulators
        self._erle_window_near = 1e-10
        self._erle_window_err = 1e-10
        self._erle_factor_prev = 0.0
        self._inst_erle_smooth = 1.0

        # Mu-scale state (depends on DTD which depends on filter)
        self._simple_mu_ratio = 1.0
        self._simple_mu_holdoff = 0
        self._per_bin_mu_scale = None
        self.prev_dtd_conf = 0.0
        self._dtd_conf_holdoff = 0

        # ERL + EPC-render forced + DT analyzer (all filter-output-derived)
        self._erl_estimate = 0.1
        # v3.14 Arc-P P.S2: per-band ERL is filter-output-derived (echo_spec /
        # far_spec from PBFDKF), so reset it together with scalar _erl_estimate.
        self._per_band_erl[:] = 0.1
        self._epc_render_forced_remaining = 0
        self._dt_analyzer.reset()
        self._stat_dt_hangover = 0
        self._stat_far_hangover = 0

        # Coherence DTD's accumulated err/far PSDs (live in _dtd_acc_*)
        if self._dtd_fft_size > 0:
            self._dtd_acc_pos = 0
            self._dtd_acc_err.fill(0)
            self._dtd_acc_far.fill(0)
            self._dtd_err_buf.fill(0)
            self._dtd_far_buf.fill(0)

        # Shadow copy controller
        self.shadow_frame_count = 0
        self._regime_handler.reset()

        # P3f trace state — zero so post_reset_age_ms restarts from 0 and
        # refined latch / err baseline don't carry pre-reset values forward.
        self._post_reset_age_frames = 0
        self._erle_slope_buf.clear()
        self._p3f_refined_latched = False
        self._p3f_main_err_baseline = 0.0
        self._p3f_diverged_streak = 0
        self._p3f_diverged_streak_ema = 0.0
        self._prev_filter_state = 'idle'
        self._f_e1_far_active = False
        self._f_e1_far_release_count = 0
        self._delay_history.clear()
        self._frames_since_last_epc = 10**9
        self._frames_since_last_f_e3_w_reset = 10**9
        self._f_e5_prev_sat_level = 0.0

        # Diagnostic dict — would otherwise show stale ERLE / DT signals
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

        # S-orth.A: reset shadow decoupled state on derived-state reset
        # (same as filter taps reset — shadow's observation history restarts).
        if hasattr(self, '_shadow_error_psd'):
            self._shadow_error_psd.fill(1e-2)
        if hasattr(self, '_shadow_R'):
            self._shadow_R.fill(1e-2)
        if hasattr(self, '_shadow_mu_holdoff'):
            self._shadow_mu_holdoff = 0

        # Re-arm warmup so the second-pass training starts with high mu.
        # Boost Q on both filters (high-Q convergence mode).
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

        # v3.10.3 — clear cross-recovery state that would otherwise mis-fire.
        # _pending_delay: stale pending shift could pair with a later rogue
        #   estimate and trigger a spurious force_delay (audio bug).
        # NOTE: _round3_div_counts / _round3_last_div_source /
        # _dominant_nearend_hold are session-cumulative diagnostic counters
        # and MUST survive recovery — they are only cleared in full AEC.reset().
        if hasattr(self, '_pending_delay'):
            del self._pending_delay
        if hasattr(self, '_pending_delay_ttl'):
            del self._pending_delay_ttl

    @property
    def hop_size(self) -> int:
        return self._hop_size

    def _compute_mu_scale(self) -> float:
        """Convert combined DTD confidence to mu_scale [mu_min_ratio, 1.0].

        v3.10.4 — when fallback is active, alignment is unreliable so we must
        not adapt against absent/bad ref. Return 0 to freeze taps; RES rides
        its existing render-based path because filter never converges.

        #3: Coherence is primary; divergence is fallback only when coherence inactive.
        #4: Confidence has memory decay to avoid sudden drops.
        EPC: mu_scale floor during echo path change.

        v3.10.0 — delay-confidence ceiling:
          When delay-est confidence is low (PAR < par_low_threshold), the
          filter is learning against a misaligned reference — driving mu
          high will encode garbage. Cap mu_scale at a delay-confidence-
          dependent ceiling: low confidence → mu_scale ≤ 0.5; full
          confidence → no cap (1.0). This mirrors WebRTC AEC3's behavior:
          AEC3 stays conservative until matched-filter delay is solid.
        """
        conf_div = self.dtd_divergence.confidence if self.dtd_divergence else 0.0
        conf_coh = self.dtd_coherence.confidence if self.dtd_coherence else 0.0

        # #3: Coherence primary, divergence fallback
        if conf_coh > 0.1:
            raw_conf = conf_coh
        else:
            raw_conf = max(conf_div, conf_coh)

        # #4: Confidence memory decay (avoid sudden drops)
        # F2.5: two-stage hangover — attack fast (1 frame), hold 10 frames, then ×0.9 decay.
        if self.config.dtd_conf_two_stage:
            if raw_conf > self.prev_dtd_conf:
                conf = raw_conf
                self._dtd_conf_holdoff = 10
            elif self._dtd_conf_holdoff > 0:
                conf = self.prev_dtd_conf
                self._dtd_conf_holdoff -= 1
            else:
                conf = max(raw_conf, self.prev_dtd_conf * 0.9)
        else:
            conf = max(raw_conf, self.prev_dtd_conf * 0.9)
        self.prev_dtd_conf = conf

        if conf == 0.0:
            mu_scale = 1.0
        else:
            min_r = self.config.dtd_mu_min_ratio
            # Before convergence, allow higher mu_min so filter can still learn during DT
            if not self._filter_converged:
                min_r = max(min_r, 0.3)
            mu_scale = 1.0 - conf * (1.0 - min_r)

        # Echo path change: keep mu high so filter can adapt to new path
        if self.epc_active:
            mu_scale = max(mu_scale, self.config.epc_mu_floor)

        # v3.10.0: delay-confidence ceiling. Cap mu when delay alignment
        # is uncertain (avoid learning garbage against misaligned ref).
        # v3.10.3 (H2): skip the ceiling during a post-reset warmup window so
        # the high-Q boost armed by _reset_filter_derived_state can actually
        # take effect. Otherwise PAR fluctuating between low/solid thresholds
        # right after delay acquisition caps mu at ~0.5–0.7 and defeats the
        # warmup re-arm, slowing ERLE rebuild and risking a wasted second
        # plateau attempt.
        in_post_reset_warmup = (
            self._warmup_frames > 0
            or (self.filter is not None
                and getattr(self.filter, '_p_max_override_frames', 0) > 0)
        )
        if self.delay_est is not None and not in_post_reset_warmup:
            delay_conf = self.delay_est.confidence
            if delay_conf < 1.0:
                # 0.5 ceiling at delay_conf=0, linear interpolate to 1.0
                delay_ceiling = 0.5 + 0.5 * delay_conf
                mu_scale = min(mu_scale, delay_ceiling)

        return mu_scale

    def _get_simple_mu_scale(self, mu_min: float = None):
        """Get mu_scale from smoothed EER (per-bin array or scalar fallback)."""
        if mu_min is None:
            mu_min = self.config.shadow_mu_min
        # Warmup: fast convergence, but respect DT signals to protect near-end
        if self._warmup_frames > 0:
            # Only consume warmup when far-end is active (don't waste on silence)
            if self._warmup_far_active:
                self._warmup_frames -= 1
            if self._simple_mu_ratio < 0.2:
                # Strong DT detected: don't force high mu, protect filter
                return max(0.2, self._simple_mu_ratio)
            return min(1.0, max(0.5, self._simple_mu_ratio + 0.2))
        if not self._filter_converged:
            mu_min = max(mu_min, 0.3)   # Pre-convergence: floor 0.3, better DT protection
        else:
            mu_min = max(mu_min, 0.2)
        # Per-bin mu_scale from RES echo_psd/error_psd (set previous frame, post-RES)
        if self._per_bin_mu_scale is not None:
            return np.maximum(self._per_bin_mu_scale, mu_min)
        return mu_min + (1.0 - mu_min) * self._simple_mu_ratio

    def _update_simple_mu_ratio(self, output: np.ndarray,
                                 far: np.ndarray) -> None:
        """Update simple variable mu ratio after process (Valin 2007 RER-inspired).

        far/error ratio: DT → error >> far → ratio low → next frame mu drops.
        Asymmetric EMA: fast attack (ratio drops), slow release (ratio recovers).
        """
        error_power = np.mean(output ** 2) + 1e-10
        far_power = np.mean(far ** 2) + 1e-10

        # Don't update ratio during silence — preserve current value for warmup
        if far_power < 1e-6 and error_power < 1e-6:
            return

        # Quick reset when far-end transitions from silence to active
        if far_power > 1e-4 and self._simple_mu_ratio < 0.1:
            self._simple_mu_ratio = 0.8
            self._simple_mu_holdoff = 0
            return

        ratio = min(far_power / error_power, 1.0)
        # Echo estimate ratio: if filter is learning echo, don't suppress mu too much
        if hasattr(self.filter, 'echo_spec') and self.filter.echo_spec is not None:
            echo_est_pwr = np.sum(np.abs(self.filter.echo_spec) ** 2) + 1e-10
            near_pwr = np.sum(np.abs(getattr(self.filter, 'near_spec', np.zeros(1))) ** 2) + 1e-10
            if near_pwr > 1e-8:
                ratio_echo = np.clip(echo_est_pwr / near_pwr, 0.0, 1.0)
                ratio = max(ratio, ratio_echo * 0.5)

        # Asymmetric EMA + holdoff: fast attack, slow release with holdoff
        if ratio < self._simple_mu_ratio:
            # Attack: fast drop.
            # F2.4: only arm holdoff on fresh DT onset (holdoff==0); do not
            # reset during the holdoff window — marginal DT oscillation keeps
            # resetting holdoff to 20 so mu never releases.
            alpha = 0.3
            if not self.config.mu_holdoff_no_reset or self._simple_mu_holdoff == 0:
                self._simple_mu_holdoff = 20  # hold low for ~20 frames (~320ms)
        elif self._simple_mu_holdoff > 0:
            # Holdoff active: keep ratio low, don't release yet
            self._simple_mu_holdoff -= 1
            alpha = 0.99  # nearly frozen
        else:
            # Release: slow recovery
            alpha = 0.95
        self._simple_mu_ratio = alpha * self._simple_mu_ratio + (1 - alpha) * ratio

    def process(self, near_end: np.ndarray, far_end: np.ndarray) -> np.ndarray:
        # v3.18 Phase C.B — reset per-frame EPC-fired helper. Set True by
        # any EPC fire site (delay_shift / EPV / shadow_rise) below.
        # Read by FilteringQualityAnalyzer; safe no-op when flag is OFF.
        self._epc_reset_fired_this_frame = False

        # High-pass filter: remove DC + low-freq noise
        if self._hp_mic is not None:
            near_end = self._hp_mic.process(near_end.copy())
        if self._hp_ref is not None:
            far_end = self._hp_ref.process(far_end.copy())

        # Saturation detection + soft-clip reference
        if self._sat_detector_ref is not None:
            sat_ref = self._sat_detector_ref.detect(far_end)
            sat_mic = self._sat_detector_mic.detect(near_end)
            self._saturation_level = max(sat_ref, sat_mic * 0.5)
            if self.config.saturation_softclip_ref and sat_ref > 0.1:
                far_end = SaturationDetector.soft_clip(far_end.copy())
            # F-E5 / E5-1: symmetric mic soft-clip on sat_mic threshold.
            if (self.config.f_e5_enabled
                    and sat_mic > self.config.f_e5_mic_softclip_threshold):
                near_end = SaturationDetector.soft_clip(near_end.copy())
            # F-E5 / E5-3: fast-attack _error_psd reset on sat → clean
            # transition so the α=0.95 EMA does not propagate clipped
            # samples into R for ~20 frames after sat ends.
            if (self.config.f_e5_enabled
                    and self._f_e5_prev_sat_level > 0.5
                    and self._saturation_level < 0.2
                    and hasattr(self.filter, '_error_psd')):
                self.filter._error_psd.fill(1e-2)
                if (self.shadow_filter is not None
                        and hasattr(self.shadow_filter, '_error_psd')):
                    self.shadow_filter._error_psd.fill(1e-2)
            self._f_e5_prev_sat_level = float(self._saturation_level)

        # Delay estimation + reference alignment
        if self._delay_active:
            hop = len(far_end)

            # Online delay estimation (if not using fixed delay).
            # v3.10.2: tier the gate logic into TWO INDEPENDENT paths so a
            # solid-confidence shift (current_delay >= 0, is_solid, large
            # delta) is not swallowed by the first-acquisition outer gate.
            # Codex fix: previous v3.10.1 wrapped both paths under a single
            # outer `if is_solid`, so the second `elif` was unreachable when
            # is_solid was True but current_delay was already set.
            if (self.delay_est is not None
                    and self._delay_active):
                # v3.17 B.1: dynamically override DelayEst period + EMA alpha
                # under EPC (motion proxy). When EPC active or in hangover,
                # switch to fast cadence + faster EMA so the cross-spectrum
                # tracks the new echo path; restore baseline otherwise.
                # Read-modify-write the fields — DelayEstimator reads on
                # every accumulate() call.
                if self.config.mov_rate_delay_est_enabled:
                    _epc_motion = (self.epc_active
                                   or self._epc_det.hangover_count > 0)
                    if _epc_motion:
                        self.delay_est._period_samples = int(
                            self.config.delay_est_period_s_fast
                            * self.config.sample_rate
                        )
                        self.delay_est._alpha = float(
                            self.config.delay_est_alpha_fast
                        )
                    else:
                        self.delay_est._period_samples = int(
                            self.config.delay_est_period_s
                            * self.config.sample_rate
                        )
                        self.delay_est._alpha = 0.6
                self.delay_est.accumulate(near_end, far_end)
                new_delay = self.delay_est.estimated_delay
                _delay_eligible = (new_delay >= 0
                                    and self.delay_est._n_updates >= 3)

                # Path A: first delay acquisition. Heavy action (resets filter
                # taps + downstream derived state). Demands solid PAR.
                if (_delay_eligible
                        and self._current_delay < 0
                        and self.delay_est.is_solid):
                    self._current_delay = new_delay
                    # Reset filter taps + ALL filter-output-derived state
                    # (main_err_smooth / DTD / EPC / ERLE / mu / RES). The
                    # first ~300 ms was learned against a misaligned ref,
                    # so its derived state is poisoned the same way as the
                    # plateau case. Use the shared helper to keep behavior
                    # consistent across recovery paths.
                    self._reset_filter_derived_state(reason='delay_first',
                                                     preserve_render_ema=True)
                    self._maybe_mark_diverged('delay_first')

                # v3.10.3 (H1) — age out _pending_delay so a stale pending
                # value cannot pair with a later rogue estimate hours after
                # it was set. Decrements once per estimation cycle.
                if hasattr(self, '_pending_delay_ttl'):
                    self._pending_delay_ttl -= 1
                    if self._pending_delay_ttl <= 0:
                        if hasattr(self, '_pending_delay'):
                            del self._pending_delay
                        del self._pending_delay_ttl

                # Path B: delay shift. Independent of Path A — fires only
                # when current_delay is already set and a meaningful shift
                # is detected. Medium confidence (≥ 0.5) + consecutive-
                # consistent gate prevents single-frame noise from triggering
                # a force_delay() EPC chain.
                if (_delay_eligible
                        and self._current_delay >= 0
                        and self.delay_est.confidence >= 0.5
                        and abs(new_delay - self._current_delay) > 32):
                    if (hasattr(self, '_pending_delay')
                            and abs(new_delay - self._pending_delay) < 16):
                        self._current_delay = new_delay
                        del self._pending_delay
                        if hasattr(self, '_pending_delay_ttl'):
                            del self._pending_delay_ttl
                        # v3.10.3 (M4) — filter taps were trained against the
                        # old delay alignment; treating the shift like Path A
                        # (clear filter-output-derived state) avoids ~50–100
                        # frames of poor cancellation while taps re-converge
                        # against state that no longer matches.
                        self._reset_filter_derived_state(reason='delay_shift',
                                                         preserve_render_ema=True)
                        _delay_event = self._epc_det.force_delay()
                        # v3.18 Phase F.1 — classify delay-shift event.
                        if self.config.aec_event_classification_enabled:
                            self._classified_event = classify_epc_event(_delay_event)
                        for filt in [self.filter, self.shadow_filter]:
                            if filt is not None and hasattr(filt, 'Q'):
                                self._arc_m_q_boost(filt)
                                # PBFDKF-only Kalman P-override (NLMS shadow
                                # has no P state — skip cleanly).
                                if isinstance(filt, PBFDKF):
                                    filt._p_max_override = 1.0
                                    filt._p_max_override_frames = 30
                        self._maybe_mark_diverged('delay_shift')
                        self._epc_reset_fired_this_frame = True   # C.B FQA signal
                        # AEC3-pattern dispatch (echo_remover.cc): queue
                        # delay_change for next _aec3_post -> AecState runs
                        # full reset cascade.
                        from .delay.delay_types import DelayAdjustment as _DA
                        self._aec3_pending_delay_change = _DA.NEW_DETECTED_DELAY
                    else:
                        self._pending_delay = new_delay
                        self._pending_delay_ttl = 3

            # Write far_end into ring buffer
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

            # Apply delay compensation (only after enough data in ring buffer)
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

        # DTD: dual detector (divergence + coherence) for frequency-domain modes
        # Combined confidence = max(divergence, coherence) → mu_scale
        # Non-DTD: simple variable mu (Valin 2007 RER-inspired)

        # Far-end activity + stationarity (single source of truth for warmup gate,
        # stationary-DT detector, and EPC stationary guard).
        _render = self._render_activity.update(far_end)
        self._warmup_far_active = _render.warmup_active

        if self.config.enable_dtd:
            mu_scale = self._compute_mu_scale()
        else:
            mu_scale = self._get_simple_mu_scale()

        # P3e — DT advisory gate. Routes shadow/energy DTD evidence into
        # mu reduction even when enable_dtd=False. The 7GT P3d trace
        # showed dt_shadow median 0.51 / dt_energy median 0.24 in the
        # post-alignment back half but the composite gate driving mu
        # never fired, so the filter learnt against NE-contaminated error.
        # Hit-then-hold hysteresis avoids per-frame flicker.
        if self.config.dt_advisory_enabled:
            if self.config.dt_advisory_use_p3f_state:
                # P3f Phase 3: gate fires on filter_state == 'suspicious_dt'
                # using the previous frame's classification (computed late
                # in process() — 1-frame lag, ~10 ms at hop=160 / 16 kHz).
                # The state already encodes refined_latched + NE evidence
                # + main_err jump + shadow lead, so no additional guards
                # are needed at this site.
                _adv_hit = bool(self._diag.get('filter_state', 'idle')
                                == 'suspicious_dt')
            else:
                # V3 (legacy) convergence guard: only honour shadow/energy
                # DT evidence after the filter has converged at least once
                # and is past the post-reset warmup window. Retained for
                # A/B comparison; default off.
                _in_post_reset_warmup = (
                    self._warmup_frames > 0
                    or (self.filter is not None
                        and getattr(self.filter, '_p_max_override_frames', 0) > 0)
                )
                _adv_hit = (
                    bool(_render.is_active)
                    and bool(self._filter_once_converged)
                    and (not _in_post_reset_warmup)
                    and (float(self._dt_from_shadow) > self.config.dt_advisory_shadow_th
                         or float(self._dt_from_energy) > self.config.dt_advisory_energy_th)
                )
            if _adv_hit:
                self._dt_advisory_hold_remaining = self._dt_advisory_hold_frames
            elif self._dt_advisory_hold_remaining > 0:
                self._dt_advisory_hold_remaining -= 1
            if self._dt_advisory_hold_remaining > 0:
                _f = float(self.config.dt_advisory_mu_factor)
                if isinstance(mu_scale, np.ndarray):
                    mu_scale = mu_scale * _f
                else:
                    mu_scale = mu_scale * _f
            self._diag['dt_advisory_active'] = bool(self._dt_advisory_hold_remaining > 0)
            self._diag['dt_advisory_hit'] = bool(_adv_hit)

        # Mic clipping emergency: freeze filter and clamp RES output to floor.
        # Hard clipping turns mic into square waves; filter would learn garbage.
        if self._sat_detector_mic is not None:
            mic_clip = self._sat_detector_mic.saturation_level
            if mic_clip > 0.8:
                mu_scale = 0.0
        # F-E5 / E5-2: extended main mu sat-gate. Match shadow's threshold
        # (saturation_safe = sat < 0.5) so main filter does not keep learning
        # on a clipped reference signal while shadow is already paused.
        if (self.config.f_e5_enabled
                and self._saturation_level > self.config.f_e5_main_mu_sat_threshold):
            if isinstance(mu_scale, np.ndarray):
                mu_scale = np.zeros_like(mu_scale)
            else:
                mu_scale = 0.0

        _res_context = None  # populated when return_res_context=True and no internal RES

        # Stationary feature consumed by EPC / RES baseline / stationary-DT detector.
        # _render was populated above; far_pwr_global, _far_active_prev, _is_stationary_far
        # are kept as locals for downstream call-sites that still read them by name.
        far_pwr_global = _render.far_pwr

        if self.config.mode in _FREQ_MODES:
            if self._freq_near_queue is not None:
                # Buffered FDAF: accumulate into queue, process when enough
                hop = self._hop_size
                ihop = self._internal_hop
                w = self._freq_queue_write
                self._freq_near_queue[w:w+hop] = near_end
                self._freq_far_queue[w:w+hop] = far_end
                self._freq_queue_write = w + hop

                if self._freq_queue_write >= ihop:
                    # Process one internal block
                    big_out = self.filter.process(
                        self._freq_near_queue[:ihop],
                        self._freq_far_queue[:ihop], mu_scale)
                    # Store output and shift leftover input
                    leftover = self._freq_queue_write - ihop
                    self._freq_out_buf[:ihop] = big_out
                    self._freq_out_valid = ihop
                    self._freq_out_read = 0
                    if leftover > 0:
                        self._freq_near_queue[:leftover] = self._freq_near_queue[ihop:ihop+leftover]
                        self._freq_far_queue[:leftover] = self._freq_far_queue[ihop:ihop+leftover]
                    self._freq_queue_write = leftover

                r = self._freq_out_read
                raw_output = self._freq_out_buf[r:r+hop].copy()
                self._freq_out_read = r + hop
            else:
                # v3.21 Phase B.3 + B.4 — update RenderSignalAnalyzer + push
                # gates onto the refined filter BEFORE the W update fires.
                # Use the current far_end + |far_spec|² as the delay-aligned
                # render proxy (matched-filter delay drives ring-buffer
                # alignment upstream, so far_end here is already aligned).
                if self._render_signal_analyzer is not None:
                    if hasattr(self.filter, 'far_spec') and self.filter.far_spec is not None:
                        _rsa_psd = (np.abs(self.filter.far_spec) ** 2).astype(np.float32)
                    else:
                        _rsa_psd = None
                    self._render_signal_analyzer.update(_rsa_psd, far_end)
                    # Update PoorSignalExcitation counter: counter resets to 0
                    # when poor_signal_excitation fires, else increments. Once
                    # counter ≥ n_partitions the filter resumes W update.
                    if self._render_signal_analyzer.poor_signal_excitation():
                        self.filter._poor_excitation_counter = 0
                    else:
                        self.filter._poor_excitation_counter += 1
                    self.filter._saturated_capture = (self._saturation_level > 0.5)

                # v3.21 Phase C.3 / B+D — push stationary-block flag onto
                # the filter so PBFDKF skips W update on broadband stationary
                # render. Uses STALE flag (computed at end of previous hop)
                # — for stationary signals by definition the flag doesn't
                # flip between hops, so 1-hop latency is harmless. RSA
                # (B.3 / B.4) covers tonal peaks; this gate covers broadband
                # stationary noise (RSA poor_signal_excitation 0% on E0l0
                # hum-only case but stationarity flag 100%).
                _stat_flag = bool(getattr(self, '_block_stationary_for_next_hop', False))
                self.filter._block_stationary = _stat_flag
                if self.shadow_filter is not None:
                    self.shadow_filter._block_stationary = _stat_flag

                # WebRTC-style: freeze main filter weights when shadow detected divergence
                main_mu = 0.0 if self._regime_handler.main_paused else mu_scale
                raw_output = self.filter.process(near_end, far_end, main_mu)

                # v3.21 Phase C.3 — refresh StationarityEstimator using the
                # post-filter render PSD. Computes the flag used by the next
                # hop's filter.process gate (see `_block_stationary_for_next_hop`
                # push above). Same update is mirrored inside _aec3_post for
                # the residual chain; this one is for the W-gate path.
                if (self._aec3_stationarity is not None
                        and hasattr(self.filter, 'far_spec')
                        and self.filter.far_spec is not None):
                    if not self._aec3_non_zero_render_seen:
                        if float(np.max(np.abs(far_end))) >= self._AEC3_RENDER_PEAK_FLOOR:
                            self._aec3_non_zero_render_seen = True
                    if self._aec3_non_zero_render_seen:
                        _far_psd_st = (np.abs(self.filter.far_spec) ** 2).astype(np.float32)
                        self._aec3_stationarity.update_noise_estimator(_far_psd_st)
                        self._aec3_stationarity.update_stationarity_flags(_far_psd_st)
                        self._aec3_stationarity_active_hops += 1
                    # Latch flag for next hop's filter.process gate. Require
                    # post-converge (≥800 ms active render) so the noise floor
                    # estimate is reliable before we trust the stationary flag.
                    _converged_enough = (
                        self._aec3_stationarity_active_hops
                        >= self._aec3_stationarity_converge_hops
                    )
                    self._block_stationary_for_next_hop = bool(
                        _converged_enough
                        and self._aec3_stationarity.is_block_stationary()
                    )

            # Shadow filter with DTD protection (#1) and bidirectional copy (#6)
            if self.shadow_filter is not None and self._freq_near_queue is None:
                self.shadow_frame_count += 1
                # Shadow is a true background filter: always adapts at full
                # speed. Kalman's Q_high × shadow_q_ratio keeps P alive even
                # when shadow learns DT speech, so it re-converges within
                # ~100-200 frames after DT ends. The copy gate (FS baseline
                # tracking) is the sole defense against poisoning main.
                # Gate shadow update: skip adaptation when far-end is too
                # weak (poor excitation) or speaker is saturating (nonlinear
                # distortion makes error unreliable). Cf. AEC3's
                # PoorSignalExcitation() gate on main_filter_update_gain.
                far_excited = np.mean(far_end ** 2) > 1e-4
                saturation_safe = self._saturation_level < 0.5
                if self.config.shadow_mu_state_aware:
                    # B6: 4-band state-aware schedule. Precedence:
                    #   pause (main_paused/diverged) > safety (sat/weak-far)
                    #   > caution (suspicious_dt) > default.
                    if (self._regime_handler.main_paused
                            or self._prev_filter_state == 'diverged'):
                        shadow_mu_scale = 0.0
                    elif not (far_excited and saturation_safe):
                        shadow_mu_scale = 0.1
                    elif self._prev_filter_state == 'suspicious_dt':
                        shadow_mu_scale = 0.5
                    else:
                        shadow_mu_scale = 1.0
                else:
                    shadow_mu_scale = 1.0 if (far_excited and saturation_safe) else 0.1
                # v3.21 Phase B.6 — coarse-reset hangover. When the
                # poor_coarse_filter_counter has fired (set by the post-
                # process step below), freeze the coarse update for
                # COARSE_RESET_HANGOVER_HOPS so it can't immediately re-
                # drift after being reset to refined's weights.
                if getattr(self, '_coarse_reset_hangover', 0) > 0:
                    shadow_mu_scale = 0.0
                self.shadow_filter.process(near_end, far_end, shadow_mu_scale)

                # v3.21 Phase B.6 — poor_coarse_filter_counter + coarse-reset.
                # Mirrors AEC3 subtractor.cc:264-307. When the refined filter
                # has been beating the coarse filter for ≥5 consecutive hops
                # (E²_refined < E²_coarse), reset coarse ← copy(refined) and
                # arm a 16-hop hangover during which coarse cannot adapt and
                # the refined filter's H_error refresh stays on the
                # leakage_converged branch (set via filter._disallow_leakage_diverged
                # which Phase B.2 consumes).
                if (hasattr(self.filter, 'error_spec')
                        and hasattr(self.shadow_filter, 'error_spec')):
                    _e2_ref = float(np.sum(np.abs(self.filter.error_spec) ** 2))
                    _e2_coa = float(np.sum(np.abs(self.shadow_filter.error_spec) ** 2))
                    # v3.21 Phase B.2 — feed E²_coarse + per-bin ERL to the
                    # refined filter's H_error refresh formula.
                    self.filter._e2_coarse_for_refresh = _e2_coa
                    if hasattr(self, '_per_band_erl') and self._per_band_erl is not None:
                        # Broadcast 3-band ERL to per-bin: low / mid / high
                        # third partitioning across n_freqs.
                        _per_band = np.asarray(self._per_band_erl, dtype=np.float32)
                        _nf = self.filter.n_freqs
                        _b = _nf // 3
                        _erl_pb = np.empty(_nf, dtype=np.float32)
                        _erl_pb[:_b] = _per_band[0]
                        _erl_pb[_b:2 * _b] = _per_band[1]
                        _erl_pb[2 * _b:] = _per_band[2]
                        self.filter._erl_per_bin = _erl_pb
                    elif hasattr(self, '_erl_estimate'):
                        # Fallback: scalar ERL broadcast to all bins.
                        self.filter._erl_per_bin = np.full(
                            self.filter.n_freqs, float(self._erl_estimate),
                            dtype=np.float32,
                        )
                    # AEC3 cc:264-307 uses per-bin E²_refined < E²_coarse;
                    # we use scalar with a 0.5× safety margin so the
                    # trigger requires refined to be DECISIVELY better
                    # (not just slightly better in aggregate). This
                    # prevents thrashing on cases where refined and coarse
                    # alternate winning by small amounts (e.g. WqEY DTm).
                    if _e2_ref < 0.5 * _e2_coa:
                        self._poor_coarse_counter = getattr(
                            self, '_poor_coarse_counter', 0) + 1
                    else:
                        self._poor_coarse_counter = 0
                    if self._poor_coarse_counter >= 5:
                        # Copy refined → coarse + arm hangover.
                        try:
                            self.shadow_filter.copy_weights_from(self.filter)
                        except (AttributeError, TypeError):
                            # PBFDKF / PBFDAF compat: copy via W slice.
                            self.shadow_filter.W[:] = self.filter.W
                        from . import aec3_scale as _aec3_scale
                        self._coarse_reset_hangover = _aec3_scale.blocks_to_hops(
                            40, self.config.hop_size, self.config.sample_rate
                        )
                        self._poor_coarse_counter = 0
                    if getattr(self, '_coarse_reset_hangover', 0) > 0:
                        self._coarse_reset_hangover -= 1
                        self.filter._disallow_leakage_diverged = True
                    else:
                        self.filter._disallow_leakage_diverged = False

                # S-orth.A: after shadow processes, overwrite shadow's _error_psd
                # and R with the independently-tracked decoupled state when the
                # flag is ON.  This breaks the Riccati coupling: each filter now
                # accumulates its own observation-noise estimate from its own
                # residual stream rather than sharing the same EMA accumulator.
                #
                # When flag OFF: shadow_filter._error_psd / .R are set by the
                # shadow's own _update_weights call (existing behaviour).  We do
                # NOT touch them, so the path is byte-equal to baseline.
                if (self.config.shadow_state_decoupled
                        and isinstance(self.shadow_filter, PBFDKF)):
                    # --- Decoupled _error_psd update ---
                    # Use the same EMA formula as PBFDKF._update_weights but from
                    # shadow's own error_spec (already computed by shadow's process()).
                    shadow_err_spec = getattr(self.shadow_filter, 'error_spec', None)
                    if shadow_err_spec is not None:
                        _alpha_r = self.shadow_filter._alpha_r  # 0.95
                        _shadow_err_psd_inst = np.abs(shadow_err_spec) ** 2
                        self._shadow_error_psd = (
                            _alpha_r * self._shadow_error_psd
                            + (1.0 - _alpha_r) * _shadow_err_psd_inst
                        )
                        self._shadow_R = np.maximum(
                            self._shadow_error_psd, self.shadow_filter.delta)
                        # Quiescent safety regularization (Option B):
                        # When filter is converged and shadow _error_psd has drifted
                        # more than 3× away from main's in either direction, nudge
                        # shadow back by 10% blend per frame.  Fires only in steady
                        # FS (refined_usable + far_excited) so it cannot corrupt the
                        # non-stationary path where orthogonality matters most.
                        # B4 fix (2026-05-14): drop dead 'converged' branch — that
                        # string belongs to AecFilterState enum, not the internal
                        # P3f state machine (lines 7180-7199), which only sets
                        # 'refined_usable' for steady FS.
                        _is_quiescent = (
                            far_excited
                            and hasattr(self.filter, '_error_psd')
                            and getattr(self, '_prev_filter_state', 'idle')
                                == 'refined_usable'
                        )
                        if _is_quiescent:
                            _main_psd = self.filter._error_psd  # current main
                            _ratio = self._shadow_error_psd / (
                                _main_psd + np.float32(1e-10))
                            _needs_nudge = np.any(_ratio > 3.0) or np.any(
                                _ratio < 0.333)
                            if _needs_nudge:
                                # 10% blend toward main per quiescent frame
                                self._shadow_error_psd = (
                                    np.float32(0.9) * self._shadow_error_psd
                                    + np.float32(0.1) * _main_psd)
                                self._shadow_R = np.maximum(
                                    self._shadow_error_psd,
                                    self.shadow_filter.delta)
                        # Write back into shadow_filter so subsequent Kalman
                        # gain K uses the decoupled R.
                        self.shadow_filter._error_psd = self._shadow_error_psd
                        self.shadow_filter.R = self._shadow_R

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

                # Copy gate: delegated to PathChangeRegimeHandler. The handler
                # owns _copy_err_baseline / streak counters / pause hangover and
                # returns a RegimeHandlerDecision. Filter mutations (Q-boost, reverse
                # copy) are applied here so the handler stays decision-only.
                # Phase C1: optional coherence+delay gate inputs (default gate_mode='energy'
                # ignores them, so legacy behavior parity-preserved).
                _dt_coh = self.dtd_coherence.confidence if self.dtd_coherence else 0.0
                # v3.10.1: shadow-copy gate raised from any-confidence to ≥0.5.
                # Shadow→main copy permanently overwrites filter taps; should
                # only allow it when delay alignment is at least mid-confidence
                # (PAR halfway between par_low and par_solid).
                # F-DelayTrack: track delay estimate stability via variance.
                # Append valid estimates to bounded history; when enabled,
                # gate reliability on variance < 4 samples AND confidence >=
                # 0.3 (relaxed minimum since variance is the primary signal).
                if (self.delay_est is not None
                        and self.delay_est.estimated_delay >= 0):
                    self._delay_history.append(int(self.delay_est.estimated_delay))
                if self.config.f_delaytrack_enabled:
                    if (self.delay_est is not None
                            and len(self._delay_history) >= 3):
                        delay_std = float(np.std(np.asarray(self._delay_history,
                                                            dtype=np.float32)))
                        _delay_reliable = (
                            delay_std < 4.0
                            and self.delay_est.confidence >= 0.3
                        )
                    else:
                        _delay_reliable = False
                else:
                    _delay_reliable = (
                        self.delay_est is not None
                        and self.delay_est.confidence >= 0.5
                    )
                # P52 A.0R.2 trace: snapshot regime-relevant state *before*
                # the handler decision + filter mutations. Audio-passive.
                _trace_p52 = getattr(self.config, 'trace_p52_regime_handler', False)
                if _trace_p52:
                    _t_main_w_before = float(np.linalg.norm(self.filter.W)) \
                        if hasattr(self.filter, 'W') else 0.0
                    _t_main_q_before = float(np.max(self.filter.Q)) \
                        if hasattr(self.filter, 'Q') else 0.0
                    _t_shadow_w_before = float(np.linalg.norm(self.shadow_filter.W)) \
                        if (self.shadow_filter is not None and
                            hasattr(self.shadow_filter, 'W')) else 0.0
                    _t_erle_before = float(self.get_erle_instant())

                shadow_decision = self._regime_handler.update(
                    shadow_frame_count=self.shadow_frame_count,
                    far_pwr=float(np.mean(far_end ** 2)),
                    main_err_smooth=float(self.main_err_smooth),
                    shadow_err_smooth=float(self.shadow_err_smooth),
                    epc_active=self.epc_active,
                    saturation_level=float(self._saturation_level),
                    dt_from_energy=float(self._dt_from_energy),
                    dt_from_coherence=float(_dt_coh),
                    delay_reliable=bool(_delay_reliable),
                )
                if shadow_decision.boost_q:
                    if hasattr(self.filter, 'Q') and hasattr(self.filter, 'Q_high'):
                        self._arc_m_q_boost(self.filter)
                        self.filter._p_max_override = 1.0
                        self.filter._p_max_override_frames = 20
                    # F2.3: Yang 2017 R-reset — over-estimated R from a prior DT
                    # period suppresses Kalman gain K post-EPC, causing slow
                    # reconvergence. Reset to R_init (1e-2) so K recovers fast.
                    if self.config.epc_r_reset_enabled:
                        self.filter._error_psd.fill(1e-2)
                        self.filter.R.fill(1e-2)
                    # B5: symmetric R-reset on shadow filter — without it, the
                    # K-handicapped shadow (stale R from same DT period) feeds
                    # a wrong-K W into main on the next reverse_copy event,
                    # undoing F2.3's fast-recovery benefit. PBFDKF-only state
                    # (NLMS shadow under shadow_class_nlms=True has no R / no
                    # _error_psd → skip cleanly).
                    if (self.config.shadow_r_reset_enabled
                            and isinstance(self.shadow_filter, PBFDKF)):
                        self.shadow_filter._error_psd.fill(1e-2)
                        self.shadow_filter.R.fill(1e-2)
                # v3.18 Phase A.3 (corrected 2026-05-16) — AEC3 has no W
                # copy between refined/coarse filters. The reverse_copy
                # mechanism exists today because PBFDKF shadow has P-memory
                # and can get stuck in a wrong basin (sending misleading
                # `shadow_advantage`). NLMS shadow has no P-memory and
                # re-adapts from its own residual; W copy becomes a no-op
                # at best and a perturbation at worst. Skip under flag-ON.
                if (shadow_decision.reverse_copy
                        and not self.config.shadow_class_nlms):
                    # Sync shadow back to main when main is clearly better.
                    self.shadow_filter.copy_weights_from(self.filter)
                    self.shadow_err_smooth = self.main_err_smooth
                    # F1.1: re-arm shadow P so K recalibrates for copied W;
                    # also boost main P for faster re-adaptation post-copy.
                    if self.config.reverse_copy_p_reset:
                        for filt in (self.shadow_filter, self.filter):
                            # Use hasattr(filt, 'P') as PBFDKF guard — P is always
                            # set as an instance var; _p_max_override is dynamic
                            # (deleted when expired), so hasattr on it gives False
                            # outside active EPC overrides.
                            if filt is not None and hasattr(filt, 'P'):
                                filt._p_max_override = 1.0
                                filt._p_max_override_frames = 15
                                filt._p_floor_beta = 1.0
                                filt._p_floor_beta_frames = 15

                if _trace_p52:
                    self._regime_trace_rows.append({
                        'frame': int(self._frame_count),
                        'boost_q_fired': bool(shadow_decision.boost_q),
                        'reverse_copy_fired': bool(shadow_decision.reverse_copy),
                        'main_paused_fired': bool(shadow_decision.pause_main),
                        'w_l2_before': _t_main_w_before,
                        'w_l2_after': float(np.linalg.norm(self.filter.W))
                            if hasattr(self.filter, 'W') else 0.0,
                        'q_max_before': _t_main_q_before,
                        'q_max_after': float(np.max(self.filter.Q))
                            if hasattr(self.filter, 'Q') else 0.0,
                        'shadow_w_l2_before': _t_shadow_w_before,
                        'shadow_w_l2_after': float(np.linalg.norm(self.shadow_filter.W))
                            if (self.shadow_filter is not None and
                                hasattr(self.shadow_filter, 'W')) else 0.0,
                        'erle_main_before': _t_erle_before,
                        'erle_main_after': float(self.get_erle_instant()),
                        'copy_counter': int(self._regime_handler.copy_counter),
                        'copy_err_baseline': float(
                            self._regime_handler.copy_err_baseline),
                    })

            # F-E3: increment "frames since last EPC" counters once per frame,
            # before EPC detection so that fire-then-reset-to-0 logic in the
            # helper observes the correct count when consecutive EPC fires.
            if self._frames_since_last_epc < 10**9:
                self._frames_since_last_epc += 1
            if self._frames_since_last_f_e3_w_reset < 10**9:
                self._frames_since_last_f_e3_w_reset += 1

            # EchoPathVariability: gain-change detection (delegated to EchoPathChangeDetector)
            epv_event = self._epc_det.update_epv(
                far_pwr_global=far_pwr_global,
                filter_converged=self._filter_converged,
                main_paused=self._regime_handler.main_paused,
            )
            # ── Round 7.1a: EPV weak-filter false-positive damping ──
            # Worst-DT_mv fires EPV 2.75x more often than best (P0 trace) but
            # its filter_w_norm is 40% of best (~7 vs ~18). Re-arming Kalman
            # state on a filter that hasn't yet grown destabilises it without
            # benefit. Skip the EPV response when main filter W norm is below
            # threshold; let it warm up. Gated by env so default = baseline.
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
            if epv_event.fired and not _epv_suppressed:
                self._epc_reset_fired_this_frame = True   # C.B FQA signal
                # AEC3-pattern dispatch (echo_remover.cc): EPV = gain_change;
                # queue for next _aec3_post call -> AecState.handle_echo_path_change
                # resets ERLE. Detection source is legacy EPV; AEC3 has no
                # internal gain_change detector. Independent of legacy
                # aec_event_classification_enabled flag (which only gates the
                # legacy soft/full reset branch dispatch, not AEC3 chain).
                self._aec3_pending_gain_change = True
                if self.config.aec_event_classification_enabled:
                    # v3.18 Phase F.3 — AEC3-aligned: EPV is gain_change → soft
                    # reset only (Q-boost on refined/coarse step-size). Skips
                    # Kalman P relax, ERL cap, state reset; mirrors AEC3
                    # subtractor.cc:170-174 (only refined_gains->HEPC runs).
                    self._handle_gain_change_soft('epv')
                else:
                    for filt in [self.filter, self.shadow_filter]:
                        if filt and hasattr(filt, 'Q'):
                            self._arc_m_q_boost(filt)
                            # PBFDKF-only Kalman P-override (NLMS shadow skip).
                            if isinstance(filt, PBFDKF):
                                filt._p_max_override = 1.0
                                filt._p_max_override_frames = 30
                                filt._p_floor_beta = 1.0
                                filt._p_floor_beta_frames = 30
                    self._maybe_mark_diverged('epv')
                    self._epc_render_forced_remaining = self.config.epc_hangover
                    self._erl_estimate = min(self._erl_estimate, 0.3)
                    if self.config.use_epc_state_reset:
                        self._apply_epc_state_reset('epv')
                    self._f_e3_handle_epc_fire('epv')

            # v3.18 Phase F.1 — AEC3 event classification (trace-only; no
            # consumer logic in F.1, so byte-equal preserved when flag OFF).
            if self.config.aec_event_classification_enabled:
                _evt = epv_event if epv_event.fired and not _epv_suppressed else EpcEvent()
                self._classified_event = classify_epc_event(_evt)

            # Echo path change: shadow-error rise (delegated to EchoPathChangeDetector).
            # Update + hangover tick are inside the original (shadow_filter, filter_converged)
            # gate to preserve bit-exact countdown semantics from v2.8.1.
            if self.shadow_filter is not None and self._filter_converged:
                rise_event = self._epc_det.update_shadow_rise(
                    main_err_smooth=self.main_err_smooth,
                    shadow_err_smooth=self.shadow_err_smooth,
                    is_stationary=self._render_activity.is_stationary,
                )
                # F-E5 / E5-4: mask shadow_rise during sustained saturation.
                # Clipped input causes both filter errors to rise in tandem;
                # the detector reads that as path change but it is really
                # nonlinear distortion. Avoid false EPC triggering filter
                # re-initialisation during a sat event.
                if (self.config.f_e5_enabled
                        and self._saturation_level > self.config.f_e5_main_mu_sat_threshold
                        and rise_event.fired):
                    rise_event = type(rise_event)(fired=False, source=rise_event.source)
                if rise_event.fired:
                    self._epc_reset_fired_this_frame = True   # C.B FQA signal
                    # AEC3-pattern dispatch: shadow_rise = gain_change proxy.
                    # Queue independent of legacy flag.
                    self._aec3_pending_gain_change = True
                    if self.config.aec_event_classification_enabled:
                        # v3.18 Phase F.3 — AEC3-aligned: shadow_rise is a
                        # gain_change proxy (both errors rising signals filter
                        # mistracking, not delay mis-alignment) → soft reset
                        # only. DTD confidence dampening retained because it
                        # protects the per-frame DT signal regardless of
                        # which reset path runs.
                        if self.dtd_coherence:
                            self.dtd_coherence.confidence *= 0.3
                        self._handle_gain_change_soft('shadow_rise')
                    else:
                        if self.dtd_coherence:
                            self.dtd_coherence.confidence *= 0.3
                        for filt in [self.filter, self.shadow_filter]:
                            if filt and hasattr(filt, 'Q'):
                                self._arc_m_q_boost(filt)
                        self._maybe_mark_diverged('shadow_rise')
                        # P_MAX relax + P_floor raise: force filter to abandon
                        # stale path estimate. PBFDKF-only (NLMS shadow skip).
                        for filt in [self.filter, self.shadow_filter]:
                            if filt and isinstance(filt, PBFDKF):
                                filt._p_max_override = 1.0
                                filt._p_max_override_frames = 30
                                filt._p_floor_beta = 1.0
                                filt._p_floor_beta_frames = 30
                        # Change D: arm RES render-forced + cap stale ERL
                        self._epc_render_forced_remaining = self.config.epc_hangover
                        self._erl_estimate = min(self._erl_estimate, 0.3)
                        if self.config.use_epc_state_reset:
                            self._apply_epc_state_reset('shadow_rise')
                        self._f_e3_handle_epc_fire('shadow_rise')
                else:
                    # Hangover tick — only when shadow_rise did NOT fire (preserves
                    # original if/elif/else structure exactly).
                    self._epc_det.tick_hangover()
                # v3.18 Phase F.1 — classify shadow_rise event (post-mask).
                if self.config.aec_event_classification_enabled and rise_event.fired:
                    self._classified_event = classify_epc_event(rise_event)

            # WebRTC-style: no output switching. Main filter output is always used.
            # (Shadow filter drives divergence detection + Q boost + pause, not output selection.)

            # final_output starts from raw_output; RES modifies final_output only
            self._last_raw_output = raw_output  # save for diagnostic (time-domain echo power)
            final_output = raw_output.copy()

            # v3.18 Phase B.2/B.3 — FilterMisadjustmentEstimator + ScaleFilter
            # update. Estimator tracks long-term echo/render ratio; trigger
            # fires scale_filter when stable convergence + persistent under-
            # modelling. Both methods return immediately when flag is OFF
            # (byte-equal preserved). Scale action affects subsequent frames
            # only; current frame's raw_output already computed.
            self._update_misadjustment_estimator()
            self._check_and_apply_misadjustment_scale()

            # v3.18 Phase C.A — FilterAnalyzer audit-only update. Reads main
            # filter W → time-domain impulse → HP-filter → peak detection +
            # consistency check. Outputs exposed via _diag only; no consumer
            # changes behaviour. Skipped when flag OFF (byte-equal preserved).
            if (self._filter_analyzer is not None
                    and self.filter is not None
                    and hasattr(self.filter, 'W')):
                _W_sum = self.filter.W.sum(axis=0)
                _w_time = np.fft.irfft(_W_sum, self.filter.fft_size).astype(np.float32)
                self._filter_analyzer.update(_w_time)
                self._diag['filter_analyzer_consistent'] = bool(
                    self._filter_analyzer.consistent_estimate)
                self._diag['filter_analyzer_peak_index'] = int(
                    self._filter_analyzer.peak_index)
                self._diag['filter_analyzer_max_gain'] = float(
                    self._filter_analyzer.max_echo_path_gain)

            # v3.18 Phase C.B — FilteringQualityAnalyzer audit-only update.
            # Multi-gate usable_linear_estimate; outputs to _diag['fq_*'].
            # convergence_signal prefers FilterAnalyzer.consistent_estimate
            # (AEC3-aligned semantic) when C.A is available; else falls
            # back to legacy _filter_converged.
            if self._filter_quality is not None:
                _fq_far_active = bool(np.mean(far_end ** 2) > 1e-4)
                if self._filter_analyzer is not None:
                    _fq_conv_signal = bool(self._filter_analyzer.consistent_estimate)
                else:
                    _fq_conv_signal = bool(self._filter_converged)
                self._filter_quality.update(
                    far_active=_fq_far_active,
                    epc_reset_fired=bool(self._epc_reset_fired_this_frame),
                    convergence_signal=_fq_conv_signal)
                self._diag['fq_usable'] = bool(self._filter_quality.usable)
                self._diag['fq_startup_done'] = bool(self._filter_quality.startup_done)
                self._diag['fq_reset_done'] = bool(self._filter_quality.reset_done)
                self._diag['fq_convergence_seen'] = bool(
                    self._filter_quality.convergence_seen)
                self._diag['fq_far_active_recent'] = bool(
                    self._filter_quality.far_active_recent)

            # AEC3 post-filter using OLA + sqrt-Hann (skip for buffered FDAF).
            # enable_res gates suppression so eval_aec_challenge's no-RES
            # comparison run still emits the linear residual.
            if (self.config.enable_res or self.config.return_res_context) and self._freq_near_queue is None:
                far_power = np.mean(far_end ** 2)
                # Dynamic over_sub: moderate base, scale with convergence.
                # Windowed decaying ERLE (TC ≈ 10s) replaces irreversible
                # cumulative get_erle(): in DT, instant ERLE drops because near
                # speech raises raw_error_power, and the windowed accumulator
                # follows it down within hundreds of frames, so erle_factor
                # actually backs off and base_over_sub drops below the
                # converged-state ceiling. Pure instant is too noisy in
                # onsets, so we still take max() with instant ERLE.
                _erle_decay = 0.999  # TC ≈ 1000 frames ≈ 10s
                self._erle_window_near = (_erle_decay * self._erle_window_near
                                           + self.near_power)
                self._erle_window_err = (_erle_decay * self._erle_window_err
                                          + self.raw_error_power)
                erle_windowed = 10.0 * np.log10(
                    (self._erle_window_near + 1e-10)
                    / (self._erle_window_err + 1e-10))
                erle_for_factor = max(self.get_erle_instant(), erle_windowed)
                # D2: ramp from 0 dB (was 2 dB). With B4 dynamic ERL,
                # render-based is now useful at low ERLE → smoother blend.
                erle_factor = np.clip(erle_for_factor / 10.0, 0.0, 1.0)
                self._erle_factor_prev = float(erle_factor)
                # DT protection
                far_pwr = np.mean(far_end ** 2) + 1e-10
                mic_pwr = np.mean(near_end ** 2) + 1e-10
                raw_err_pwr = np.mean(raw_output ** 2) + 1e-10
                # B4: track ERL for render-based echo estimate.
                # Gate: only update when residual is not dominated by near-end
                # speech (raw_dt < 2.0 allows high-coupling echo-only through).
                # Pre-convergence only: after convergence, filter-based echo
                # estimate is reliable and render-based mode is off.
                # F-E1: hysteresis far_active gate (attack 1e-4, release
                # after 5 frames below 3e-5). Stabilises ERL update during
                # marginal-reference dips. Falls back to simple threshold
                # when flag disabled.
                if self.config.f_e1_enabled:
                    if far_pwr > 1e-4:
                        self._f_e1_far_active = True
                        self._f_e1_far_release_count = 0
                    elif self._f_e1_far_active:
                        if far_pwr < 3e-5:
                            self._f_e1_far_release_count += 1
                            if self._f_e1_far_release_count >= 5:
                                self._f_e1_far_active = False
                                self._f_e1_far_release_count = 0
                        else:
                            self._f_e1_far_release_count = 0
                    erl_update_gate = self._f_e1_far_active
                    # F-E1: extend ERL clip lower bound to 1e-5 (was 0.001)
                    # so extreme high coupling cases pass through cleanly.
                    erl_clip_lo = 1e-5
                else:
                    erl_update_gate = (far_pwr > 1e-4)
                    erl_clip_lo = 0.001
                if erl_update_gate:
                    raw_dt_ratio = raw_err_pwr / (far_pwr + 1e-10)
                    inst_erl_raw = mic_pwr / far_pwr
                    # v3.2 Axis 1: NE-corruption protection. ERL > 1.5 physically
                    # implausible (mic louder than far → NE dominates), so skip update.
                    if raw_dt_ratio < 2.0 and inst_erl_raw < 1.5:
                        inst_erl = np.clip(inst_erl_raw, erl_clip_lo, 1.0)
                        alpha_erl = 0.99 if not self._filter_converged else 0.999
                        self._erl_estimate = float(alpha_erl * self._erl_estimate + (1 - alpha_erl) * inst_erl)

                # Pre-filter DT signal (Stage B): mic energy excess over
                # far × max_ERL. Realistic rooms have ERL ≤ +6 dB (coupling
                # factor 4.0). When mic_pwr > 4 × far_pwr, the mic has
                # energy that can't be explained by echo alone → near-end
                # speech or NE segment. This signal is PRE-FILTER so it's
                # immune to the inst_erle correction that kills raw_dt in
                # high-coupling DT crush cases.
                # Gate on far_active: NE-only segments (far≈0) would produce
                # dt_from_energy≈1.0, and the slow EMA decay (TC≈90ms) would
                # hang over into the following FS segment, relaxing ENR
                # thresholds while echo is present → FS echo leakage.
                self._dt_analyzer.update_energy_dt(
                    far_active=self._render_activity.is_active,
                    far_pwr=far_pwr,
                    mic_pwr=mic_pwr,
                    erl_estimate=self._erl_estimate,
                )

                # Step 1: base DT confidence
                if self.config.enable_dtd:
                    raw_dt = self.get_dtd_confidence()
                else:
                    simple_dt = 1.0 - far_pwr / (mic_pwr + far_pwr)
                    # ERL-corrected blend: simple ratio misfires in high-ERL FS
                    # (mic ≈ echo > far → simple says "DT", ERL-corrected says "FS").
                    # Take max of ERL-corrected estimate and half the simple ratio
                    # so true DT (dt_from_energy high) is preserved, false DT is halved.
                    raw_dt = max(float(self._dt_from_energy), simple_dt * 0.5)

                # Stationary DT macro detection (sets flag only, does NOT override raw_dt)
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
                        self._stat_dt_hangover = 80  # 800ms protection window (covers syllable gaps)

                    if self._stat_dt_hangover > 0:
                        is_stationary_dt = True
                        self._stat_dt_hangover -= 1
                        # Speech active: nearly freeze baseline (TC ≈ 1000 frames)
                        self._wn_err_baseline = (0.999 * self._wn_err_baseline
                                                  + 0.001 * track_err_pwr)
                    else:
                        is_stationary_dt = False
                        # Silence: normal EMA tracking WN baseline
                        self._wn_err_baseline = (0.95 * self._wn_err_baseline
                                                  + 0.05 * track_err_pwr)

                # D4: slowly track baseline during non-stationary far-end
                # (converged only). Prevents stale 1e-8 baseline when clip
                # starts with speech far-end → first stationary transition
                # sees huge jump_ratio → false stationary DT trigger.
                if (self._filter_converged and not self._render_activity.is_stationary
                        and far_pwr > 1e-4 and self._wn_err_baseline > 1e-6):
                    self._wn_err_baseline = (0.995 * self._wn_err_baseline
                                              + 0.005 * raw_err_pwr)

                # inst_erle correction (only no-DTD)
                if not self.config.enable_dtd:
                    inst_erle_fast_raw = mic_pwr / raw_err_pwr
                    self._inst_erle_smooth = (0.7 * self._inst_erle_smooth
                                              + 0.3 * inst_erle_fast_raw)
                    if self._inst_erle_smooth > 2.0:
                        # Cap correction divisor: inst_erle=15 would make
                        # raw_dt=0.6→0.04, killing DT protection entirely.
                        # Cap at 4.0 keeps raw_dt=0.6→0.15, preserving
                        # some DT awareness while still reducing false DT
                        # in high-coupling FS (original purpose).
                        erle_for_dt = min(self._inst_erle_smooth, 4.0)
                        raw_dt /= erle_for_dt

                # EPC physical gate
                # Round 3 D3 trace: record raw_dt BEFORE the EPC zero so we can
                # see what the suppressor would have seen if the gate were
                # split (adaptation vs RES). audio-passive.
                self._round3_raw_dt_pre_epc = float(raw_dt)
                if self.epc_active:
                    raw_dt = 0.0
                    is_stationary_dt = False  # EPC error spike is from filter divergence, not speech

                dt_indicator = np.clip(raw_dt, 0.0, 0.8)

                # Divergence indicator EMA (delegated to FilterConvergenceAnalyzer)
                self._convergence.update_divergence(self.near_power, self.raw_error_power)

                final_output = self._aec3_post(raw_output, near_end, far_end)

                # v3.13 E4.S3 — SubtractiveNLP detector (audit-only).
                # Pure observer: reads the LINEAR residual (raw_output =
                # mic − linear_echo_estimate, RES input) so the NL
                # harmonic signature is not masked by RES suppression.
                # Also reads mic_hop (near_end) for the S4.1
                # cancellation-ratio gate (NE bucket discrimination).
                # See E4.S1 Pass A finding: production output (post-RES)
                # hides NL evidence.
                if self.nl_detector is not None:
                    _nl_conf = self.nl_detector.process(
                        raw_output,
                        filter_state=self._prev_filter_state,
                        far_active=(far_power > 1e-4),
                        mic_hop_samples=near_end)
                    self._diag['nl_confidence'] = _nl_conf
                    self._diag['nl_pitch_strength'] = (
                        self.nl_detector._pitch_strength_last)
                    self._diag['nl_pitch_lag'] = (
                        self.nl_detector._pitch_lag_last)

                # Update per-bin mu_scale AFTER RES. echo_psd/error_psd were
                # only written from ResFilter.process() (dead since R4) → both
                # were permanently zero, so per_bin_eer was always np.zeros and
                # _per_bin_mu_scale was a fresh mu_min*ones array per frame.
                if not self.config.enable_dtd:
                    if self._filter_converged:
                        mu_min = self.config.shadow_mu_min
                        self._per_bin_mu_scale = np.full(
                            self.filter.n_freqs, mu_min, dtype=np.float32)
                        self._simple_mu_ratio = 0.0
                        # Stationary DT: only freeze when speech actually
                        # detected (jump_ratio + hangover). _is_stationary_far
                        # alone fires on 32% of normal speech far-end frames,
                        # which would crush filter convergence on plain FS.
                        if is_stationary_dt:
                            self._per_bin_mu_scale[:] = mu_min
                            self._simple_mu_ratio = mu_min
                    else:
                        # Pre-convergence: no per_bin, let ratio track DT naturally
                        self._per_bin_mu_scale = None
                        self._update_simple_mu_ratio(raw_output, far_end)

            # C-parity fix: when RES is disabled, _update_simple_mu_ratio is
            # never called in PBFDKF path. C always calls update_simple_mu_ratio
            # regardless. Add call here when RES is off (avoids double-update
            # when RES is on).
            if (not self.config.enable_res
                    and not self.config.enable_dtd
                    and not self._filter_converged):
                self._update_simple_mu_ratio(raw_output, far_end)

            # Update diagnostics. ResFilter._diag_* fields were only written
            # from process() (dead since R4) → all consumers saw __init__
            # defaults forever. Hardcoded here so the diag dict structure is
            # preserved without depending on self.res existing.
            self._diag['res_gain_mean'] = 1.0
            self._diag['res_gain_min'] = 1.0
            self._diag['effective_g_min'] = 1.0
            self._diag['far_activity'] = 0.0
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
            self._diag['erle_factor'] = float(erle_factor) if 'erle_factor' in locals() else 0.0
            self._diag['divergence'] = self._divergence_indicator
            self._diag['using_render_based'] = False
            self._diag['shadow_advantage'] = getattr(self, '_shadow_advantage', 1.0)
            self._diag['dt_from_energy'] = self._dt_from_energy
            self._diag['dt_from_shadow'] = getattr(self, '_dt_from_shadow', 0.0)
            self._diag['erl_estimate'] = self._erl_estimate
            # v3.14 Arc-P P.S2: expose per-band ERL EMA values for auditing.
            # Zero-cost when flag OFF (array always exists but values stay 0.1).
            self._diag['per_band_erl_lf'] = float(self._per_band_erl[0])
            self._diag['per_band_erl_mf'] = float(self._per_band_erl[1])
            self._diag['per_band_erl_hf'] = float(self._per_band_erl[2])
            self._diag['epc_active'] = self.epc_active
            self._diag['saturation_level'] = self._saturation_level
            self._diag['erle_windowed'] = float(erle_windowed) if 'erle_windowed' in locals() else 0.0
            # DT / filter debug fields
            self._diag['dt_indicator'] = float(dt_indicator) if 'dt_indicator' in locals() else 0.0
            self._diag['main_err_smooth'] = float(getattr(self, 'main_err_smooth', 0.0))
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

            # ---- P3f Mini AecState trace (no behaviour change) ----
            # Full-band WebRTC-style ratios e2/y2. Numerator/denominator must
            # share units: use the sample-loop EMAs of time-domain power
            # (self.raw_error_power, self.near_power, alpha=0.999), which are
            # commensurate. main_err_smooth / shadow_err_smooth are sums over
            # frequency bins (different scale from mean(near²)) so they
            # cannot be used directly as e2 numerators. We still use them
            # to derive shadow_err_power by scaling: same get_error_energy()
            # scaling factor for both main and shadow, so the ratio is
            # preserved.
            _mic_pwr_p3f = max(float(self.near_power), 1e-12)
            _main_err_pwr = max(float(self.raw_error_power), 1e-12)
            _main_err_smooth_p3f = max(float(self.main_err_smooth), 1e-12)
            _shadow_err_smooth_p3f = max(float(self.shadow_err_smooth), 1e-12)
            # shadow_err in the same time-domain scale as raw_error_power
            _shadow_err_pwr = _shadow_err_smooth_p3f * (
                _main_err_pwr / _main_err_smooth_p3f)
            _main_err_ratio = _main_err_pwr / _mic_pwr_p3f
            _shadow_err_ratio = _shadow_err_pwr / _mic_pwr_p3f
            _shadow_advantage_p3f = (
                _main_err_smooth_p3f / _shadow_err_smooth_p3f)

            self._post_reset_age_frames += 1
            _hop_ms_p3f = float(self.config.hop_size) * 1000.0 / float(self.config.sample_rate)
            _post_reset_age_ms = self._post_reset_age_frames * _hop_ms_p3f

            # erle_slope: dB/s over the trailing ~0.5s window
            _erle_inst_db = float(self._diag.get('erle_inst', 0.0))
            self._erle_slope_buf.append(_erle_inst_db)
            if len(self._erle_slope_buf) >= 2:
                _slope_dt = (len(self._erle_slope_buf) - 1) * _hop_ms_p3f / 1000.0
                _erle_slope_db_per_s = (
                    self._erle_slope_buf[-1] - self._erle_slope_buf[0]
                ) / max(_slope_dt, 1e-6)
            else:
                _erle_slope_db_per_s = 0.0

            # Filter-state classifier v1 (Phase 2a). State priority order:
            #   idle < startup < diverged < refined_usable < suspicious_dt
            #                           < coarse_learning (default)
            # Thresholds picked from Phase 1 smoke and Phase 2a iteration on
            # 5 trace cases (FS_static, FS_movement, DT_static, DT_movement,
            # 7GT). Re-tune in Phase 2b if invariants still fail.
            # Startup is the *unstable* post-reset window only. The
            # _p_max_override (shadow Q-boost) re-arms repeatedly during
            # healthy adaptation and would otherwise mask refined_usable.
            _post_reset_warmup_p3f = (self._warmup_frames > 0)
            _far_active_p3f = bool(self._render_activity.is_active)
            # Idle: insufficient signal to classify. Either far is silent
            # (no excitation to evaluate filter) or near is silent (no echo
            # / NE — main_err and mic both clamp to floor giving ratio 1).
            # Recognise by the floor signature: main_err_smooth at clamp
            # (1e-9 or smaller) means filter has not produced a meaningful
            # error sample yet on the current input.
            _is_idle = (
                (not _far_active_p3f)
                or _main_err_smooth_p3f <= 1e-9
                or _mic_pwr_p3f <= 1e-8
            )
            # Diverged: filter actively making it worse. Require ratio
            # comfortably above unity (1.3) for at least 5 consecutive
            # far-active frames (~50 ms hysteresis at hop=160) to filter
            # out single-frame noise. Reset streak when condition fails.
            _diverged_th = 1.3
            _diverged_min_streak = 5
            _diverged_hit_this_frame = (not _is_idle) and _main_err_ratio > _diverged_th
            if _diverged_hit_this_frame:
                self._p3f_diverged_streak += 1
            else:
                self._p3f_diverged_streak = 0
            # F2.2 — EMA-smoothed streak. Single-frame dips don't fully reset
            # the evidence (legacy hard counter does). Always tracked; only
            # consumed by P3h reset gate when `use_diverged_streak_ema` flag
            # is True. α=0.95 by default → TC ≈ 20 frames ≈ 200 ms at hop=160.
            _alpha_dse = float(self.config.diverged_streak_ema_alpha)
            self._p3f_diverged_streak_ema = (
                _alpha_dse * self._p3f_diverged_streak_ema
                + (1.0 - _alpha_dse) * (1.0 if _diverged_hit_this_frame else 0.0)
            )

            # Pre-compute suspicious_dt criteria (used both in the flag
            # below and to gate refined_usable): NE evidence AND
            # main_err jump above the refined baseline AND shadow lead.
            # All three must be present — a single shadow_advantage spike
            # (FS-early shadow lead) does NOT qualify.
            _ne_evidence = (
                float(getattr(self, '_dt_from_energy', 0.0)) > 0.3
                or float(getattr(self, '_dt_from_shadow', 0.0)) > 0.5
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
                # Suspicious_dt takes precedence over refined_usable so a
                # frame with NE evidence + main_err jump + shadow lead is
                # surfaced even when the absolute ratio still looks
                # converged.
                _filter_state = 'suspicious_dt'
            elif (self._filter_once_converged
                  and _main_err_ratio < 0.7
                  and _erle_slope_db_per_s > -5.0):
                # Latch refined once seen so suspicious_dt can subsequently
                # detect departures from this baseline.
                self._p3f_refined_latched = True
                _filter_state = 'refined_usable'
            else:
                _filter_state = 'coarse_learning'

            # Baseline tracks the *best* converged main_err_ratio: it
            # follows downward (so a better-converged filter raises the bar
            # for suspicious_dt) but freezes when the ratio rises (so an
            # NE-contaminated rising ratio doesn't drag the baseline along
            # with it and defeat the 2× jump trigger). Only updated while
            # in refined_usable.
            if _filter_state == 'refined_usable':
                if self._p3f_main_err_baseline <= 1e-6:
                    self._p3f_main_err_baseline = _main_err_ratio
                elif _main_err_ratio < self._p3f_main_err_baseline:
                    # downward EMA: capture better convergence
                    self._p3f_main_err_baseline = (
                        0.9 * self._p3f_main_err_baseline + 0.1 * _main_err_ratio)
                # else: ratio is rising — freeze baseline so jump can fire

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
            # B6 — cache for next-frame shadow_mu state-aware schedule.
            self._prev_filter_state = _filter_state

            # ---- P3h sustained-diverged filter reset (default off) ----
            # Decrement cooldown every frame. Fire reset only when
            # filter has been good before (`_filter_once_converged`),
            # the classifier reports diverged, the streak meets the
            # configured threshold, and cooldown has elapsed. Sets
            # cooldown so we never loop-reset on a flaky classifier.
            if self._p3h_reset_cooldown_remaining > 0:
                self._p3h_reset_cooldown_remaining -= 1
            _p3h_fired = False
            # F2.2 — streak-evidence selector. Default (flag OFF): legacy
            # hard counter `>= diverged_reset_streak_frames` (≥50 by default).
            # Flag ON: EMA gate `streak_ema > threshold` (default 0.7 over
            # α=0.95 ⇒ ~80%-of-frames-diverged-over-200 ms window). EMA
            # variant survives single-frame ratio dips so the gate actually
            # fires on cohort-tail cases where ratio oscillates around 1.3.
            if self.config.use_diverged_streak_ema:
                _streak_evidence_ok = (
                    self._p3f_diverged_streak_ema
                    > float(self.config.diverged_streak_ema_threshold)
                )
            else:
                _streak_evidence_ok = (
                    self._p3f_diverged_streak
                    >= int(self.config.diverged_reset_streak_frames)
                )
            # Sprint 13-14: triple-AND gate adds shadow_advantage > 2.0 to
            # eliminate the F2.2 EMA false-positive pattern (shadow tracking
            # during movement looked like divergence). Requires shadow to
            # also be ahead — only fires on true main-filter divergence
            # signature, not on path-change events.
            _triple_and_ok = (
                not self.config.diverged_reset_triple_and
                or _shadow_advantage_p3f
                    > float(self.config.diverged_reset_triple_and_shadow_adv_min)
            )
            if (self.config.diverged_reset_enabled
                    and self._filter_once_converged
                    and self._p3h_reset_cooldown_remaining == 0
                    and _filter_state == 'diverged'
                    and _streak_evidence_ok
                    and _triple_and_ok):
                self._reset_filter_derived_state(reason='p3h_diverged')
                self._p3h_reset_cooldown_remaining = int(
                    self.config.diverged_reset_cooldown_frames)
                self._p3h_reset_count += 1
                _p3h_fired = True
            self._diag['p3h_reset_fired'] = bool(_p3h_fired)
            self._diag['p3h_reset_cooldown'] = int(self._p3h_reset_cooldown_remaining)
            self._diag['p3h_reset_count'] = int(self._p3h_reset_count)
            self._diag['p3f_diverged_streak_ema'] = float(self._p3f_diverged_streak_ema)
            # ---- end P3h ----

            self._far_power_ema = 0.95 * self._far_power_ema + 0.05 * far_pwr_global
            self._mic_power_ema = 0.95 * self._mic_power_ema + 0.05 * (np.mean(near_end ** 2) + 1e-10)
            self._frame_count += 1
            # PR-D4: stats-only DT-from-frame-0 detector. Counts frames where
            # filter has had ≥2s of far-active without converging AND ERL has
            # drifted upward — signature of NE-corrupted filter learning.
            # See docs/aec_methods.md appendix E for full rationale.
            if self._render_activity.is_active:
                self._far_active_blocks = getattr(self, '_far_active_blocks', 0) + 1
            if (getattr(self, '_far_active_blocks', 0) > 200
                    and not self._filter_converged
                    and self._erl_estimate > 0.4):
                self._dt_from_zero_count = getattr(self, '_dt_from_zero_count', 0) + 1
            self._diag['dt_from_zero_count'] = getattr(self, '_dt_from_zero_count', 0)
            self._diag['far_power'] = self._far_power_ema
            self._diag['mic_power'] = self._mic_power_ema
            self._diag['dt_from_coherence'] = (
                self.dtd_coherence.confidence if self.dtd_coherence else 0.0)

            # Update DTD detectors for NEXT block
            # Skip divergence detector before convergence (output>mic is normal
            # when filter hasn't learned echo path yet, not actual divergence)
            if self.dtd_divergence and self._filter_converged:
                self.dtd_divergence.detect_block(near_end, far_end, output=raw_output)
            if self.dtd_coherence and self._filter_converged:
                if self._dtd_fft_size > 0:
                    # FDAFbuffered: accumulate into DTD buffer, run at hop=FL/2
                    hop = self._hop_size
                    pos = self._dtd_acc_pos
                    self._dtd_acc_err[pos:pos+hop] = raw_output
                    self._dtd_acc_far[pos:pos+hop] = far_end
                    self._dtd_acc_pos = pos + hop
                    if self._dtd_acc_pos >= self._dtd_hop:
                        # Shift main buffer and run DTD
                        dh = self._dtd_hop
                        self._dtd_err_buf[:dh] = self._dtd_err_buf[dh:]
                        self._dtd_err_buf[dh:] = self._dtd_acc_err
                        self._dtd_far_buf[:dh] = self._dtd_far_buf[dh:]
                        self._dtd_far_buf[dh:] = self._dtd_acc_far
                        self._dtd_acc_pos = 0
                        error_spec = np.fft.rfft(self._dtd_err_buf)
                        far_spec = np.fft.rfft(self._dtd_far_buf)
                        self.dtd_coherence.detect_block(
                            near_end, far_end,
                            error_spec=error_spec, far_spec=far_spec)
                else:
                    # PBFDAF/PBFDKF: use filter's spectra directly (every frame)
                    self.dtd_coherence.detect_block(
                        near_end, far_end,
                        error_spec=self.filter.error_spec,
                        far_spec=self.filter.far_spec)
        else:
            # LMS/NLMS: use mu_scale from DTD or simple variable mu
            raw_output, echo_est = self.filter.process_block(near_end, far_end,
                                                              mu_scale=mu_scale)
            final_output = raw_output.copy()
            if not self.config.enable_dtd:
                self._update_simple_mu_ratio(raw_output, far_end)

        # Output limiter: final_output should never exceed mic amplitude.
        # Uses smoothed gain to avoid frame-boundary clicking artifacts.
        # v3.21 fix — the AEC3 chain's `_aec3_post` output has a 1-hop OLA
        # delay relative to the live `near_end` we compare against. To keep
        # the limiter useful (it suppresses real echo overshoots on DT)
        # without falsely attenuating NE (the OLA-lag misaligned compare
        # fired on every speech-silence transition, mean limiter gain 0.79
        # on NE-only wJVPo), use the PREVIOUS hop's mic as the comparison
        # source when routing through the AEC3 chain. That mic was the
        # actual source of the OLA reconstruction now in `final_output`.
        if self._limiter_near_lag is None:
            self._limiter_near_lag = np.zeros_like(near_end)
        near_for_limiter = self._limiter_near_lag
        self._limiter_near_lag = near_end.copy()
        near_peak = np.max(np.abs(near_for_limiter))
        out_peak = np.max(np.abs(final_output))
        if out_peak > near_peak > 1e-6:
            target_gain = near_peak / out_peak
        else:
            target_gain = 1.0
        if target_gain < self._limiter_gain:
            alpha_lim = 0.3   # attack: compress quickly
        else:
            alpha_lim = 0.8   # release: recover moderately
        self._limiter_gain = alpha_lim * self._limiter_gain + (1 - alpha_lim) * target_gain
        final_output *= self._limiter_gain

        # ERLE: track raw (filter-only) and final (post-RES) separately
        for i in range(len(near_end)):
            self.near_power = self.alpha * self.near_power + (1 - self.alpha) * near_end[i] ** 2
            self.raw_error_power = self.alpha * self.raw_error_power + (1 - self.alpha) * raw_output[i] ** 2
            self.final_error_power = self.alpha * self.final_error_power + (1 - self.alpha) * final_output[i] ** 2
        self.near_power_sum += np.sum(near_end ** 2)
        self.raw_error_power_sum += np.sum(raw_output ** 2)
        self.final_error_power_sum += np.sum(final_output ** 2)
        # Backward compat: error_power = raw (for convergence detection / inst ERLE)
        self.error_power = self.raw_error_power
        self.error_power_sum = self.raw_error_power_sum

        # Convergence detection: 10 consecutive far-active frames with ERLE > 5 dB.
        # Gate on far_active (not _simple_mu_ratio) to avoid deadlock in
        # high-coupling FS where mic ≈ far × strong_coupling pulls
        # _simple_mu_ratio < 0.5 forever, blocking convergence.
        # ERLE > 5 dB sustained 10 frames is essentially impossible during
        # real DT, so we don't need an extra DT exclusion gate.
        just_converged = self._convergence.update_convergence(
            near_power=self.near_power,
            raw_error_power=self.raw_error_power,
            far_active=float(np.mean(far_end ** 2)) > 1e-4,
            warmup_done=self._warmup_frames <= 0,
        )
        if just_converged:
            # Switch to low Q: stable tracking mode
            for filt in [self.filter, self.shadow_filter]:
                if filt and hasattr(filt, 'Q_low'):
                    filt.Q = filt.Q_low.copy()

        # v3.10.0 — filter plateau detection + one-shot recovery.
        # Dispatched here (not in DT analyzer or convergence analyzer)
        # because it needs both signals: convergence flag + DT pattern
        # signature. Recovery action below is intentionally heavy (resets
        # filter taps + shadow + RES + EPC mark_diverged) — it's only
        # triggered when the filter has been stuck below ERLE convergence
        # threshold for a sustained time despite far-end activity.
        _far_now = float(np.mean(far_end ** 2))
        _far_active_now = _far_now > 1e-4
        # Use the same windowed-ERLE expression as the diag pipeline above.
        _erle_win_db = float(10.0 * np.log10(
            (self._erle_window_near + 1e-10)
            / (self._erle_window_err + 1e-10)))
        _dt_signal_now = (float(self._dt_from_shadow) > 0.3
                          or float(self._dt_from_energy) > 0.3)
        if self._plateau_detector.update(
            far_active=_far_active_now,
            dt_signal_present=_dt_signal_now,
            erle_windowed_db=_erle_win_db,
            once_converged=self._filter_once_converged,
        ):
            # Plateau confirmed — full derived-state reset. Shared helper
            # with delay_first acquisition (Codex finding: both paths reset
            # filter taps, both should clear downstream state the same way).
            # preserve_render_ema=True keeps the long-window far-PSD EMA
            # alive across recovery — it's input-side context and the
            # freshly reset filter wants it ready immediately for fallback.
            self._reset_filter_derived_state(reason='plateau',
                                              preserve_render_ema=True)

        # ── Phase 0 trace-only AEC3 state diagnostics (read-only; do not gate audio) ──
        _initial_state_active = (
            (not self._filter_once_converged)
            or self._frame_count < (self.config.warmup_frames + 50)
        )
        _initial_transition_triggered = bool(just_converged)
        _epc_hangover = self._epc_det.hangover_count if hasattr(self, '_epc_det') else 0
        _usable_v1 = (self._filter_once_converged
                       and self._filter_converged
                       and not self.epc_active)
        _usable_v2 = (
            _usable_v1
            and self._frame_count > self.config.warmup_frames + 30
            and self._convergence.divergence < 0.3
            and _epc_hangover < 1
        )
        # dominant_nearend with hold counter + initial-state gate
        _ne_raw = False
        _hold = getattr(self, '_dominant_nearend_hold', 0)
        if _ne_raw and not _initial_state_active:
            _hold = 5
        else:
            _hold = max(_hold - 1, 0)
        self._dominant_nearend_hold = _hold
        _dominant_ne_state = _hold > 0
        # ERLE reset taxonomy (label only, do not actually reset)
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
        # ── end Phase 0 trace ──

        # ── Round 3 trace-only: delay / EPC / P-override / scale ──
        # These are audio-passive (read-only inspection of state set by other code).
        self._diag['epc_active_now'] = bool(self._epc_det.active)
        self._diag['epc_hangover_count'] = int(self._epc_det.hangover_count)
        # P-override sticky-flag: True if either filter currently in transient boost.
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
        self._diag['div_source_last'] = getattr(self, '_round3_last_div_source', '')
        self._diag['div_counts'] = dict(getattr(self, '_round3_div_counts',
                                                {'delay_first': 0, 'delay_shift': 0,
                                                 'epv': 0, 'shadow_rise': 0}))
        # Filter scale ratio: filter's echo_psd magnitude vs render-based estimate
        # (far * erl). If <<1 consistently → filter underestimates → misadjustment.
        try:
            _fpw = float(np.mean(np.abs(self.filter.echo_spec) ** 2))
            _fp_far = float(np.mean(np.abs(self.filter.far_spec) ** 2))
            _scale_ratio = _fpw / (_fp_far * float(self._erl_estimate) + 1e-12)
            self._diag['filter_scale_ratio'] = _scale_ratio
        except Exception:
            self._diag['filter_scale_ratio'] = 1.0
        # Inst ERLE smooth (read from ResFilter; lives there, not on AEC).
        self._diag['inst_erle_smooth'] = float(getattr(self, '_inst_erle_smooth', 0.0))
        # Pre-EPC DT for D3 design: if we ZEROED raw_dt due to EPC, this records the
        # value we WOULD have had. Set in process() before the EPC zero (search marker).
        self._diag['raw_dt_pre_epc'] = float(getattr(self, '_round3_raw_dt_pre_epc', 0.0))
        # ── end Round 3 trace ──

        # Round 5 trace: 9 per-stage gain slots. Legacy ResFilter._stage_*
        # wrote per-frame voice-band means; AEC3 chain doesn't use this trace,
        # so the slots are constant zero (preserved here to keep the diag dict
        # structure stable for external consumers).
        for _n in ('softgate_emr', 'spectral_floor', 'epc_dt_cap',
                   'quiet_mask', '3bin_smooth', 'hf_cap',
                   'pre_temporal', 'post_temporal', 'after_noise_lift'):
            self._diag[f'g_stage_{_n}_voice'] = 0.0

        # ── Round 7 trace: filter trajectory + transition events (audio-passive) ──
        if not hasattr(self, '_r7_prev_delay'):
            self._r7_prev_delay = -1
            self._r7_prev_div_counts = {'delay_first': 0, 'delay_shift': 0,
                                         'epv': 0, 'shadow_rise': 0}

        cur_delay = int(getattr(self, '_current_delay', -1))
        self._diag['delay_samples'] = cur_delay
        if cur_delay >= 0 and self._r7_prev_delay >= 0:
            self._diag['delay_delta'] = cur_delay - self._r7_prev_delay
        else:
            self._diag['delay_delta'] = 0
        self._r7_prev_delay = cur_delay

        cur_div = getattr(self, '_round3_div_counts',
                          {'delay_first': 0, 'delay_shift': 0, 'epv': 0, 'shadow_rise': 0})
        for _src in ('delay_first', 'delay_shift', 'epv', 'shadow_rise'):
            self._diag[f'event_{_src}'] = bool(cur_div.get(_src, 0) > self._r7_prev_div_counts.get(_src, 0))
        self._r7_prev_div_counts = dict(cur_div)

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
        # ── end Round 7 trace ──

        # Record DTD confidence for plotting
        self.confidence_history.append(self.get_dtd_confidence())

        result = final_output.astype(np.float32)
        if _res_context is not None:
            return (result, _res_context)
        return result

    def get_diagnostics(self) -> dict:
        """Return per-frame diagnostic dict (latest values)."""
        return self._diag.copy()

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

    def dump_p53_trace(self, path: str) -> int:
        """P53 Step 0: dump captured per-frame innovation-audit rows to .npz.

        Returns the number of frames written. No-op (returns 0) if the flag
        was off or the filter is not PBFDKF. Audit-only; never reads in
        production. See docs/p53_design_lock.md §2.
        """
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

    def dump_regime_trace(self, path: str) -> int:
        """P52 A.0R.2: dump captured per-frame regime trace rows to .npz.

        Returns the number of rows written. No-op (returns 0) if the flag was
        off and the buffer is empty. Audio-passive: dump only reads the
        already-populated `_regime_trace_rows` list.

        Columns (one array per key, shape (n_frames,) with dtype matching the
        first row's value type):
          frame, boost_q_fired, reverse_copy_fired, main_paused_fired,
          w_l2_before, w_l2_after, q_max_before, q_max_after,
          shadow_w_l2_before, shadow_w_l2_after,
          erle_main_before, erle_main_after,
          copy_counter, copy_err_baseline
        """
        rows = self._regime_trace_rows
        if not rows:
            return 0
        cols = {}
        for k in rows[0].keys():
            cols[k] = np.array([r[k] for r in rows])
        os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
        np.savez(path, **cols)
        return len(rows)

    def is_dtd_active(self) -> bool:
        return self.get_dtd_confidence() > 0.5

    def get_dtd_confidence(self) -> float:
        conf_div = self.dtd_divergence.confidence if self.dtd_divergence else 0.0
        conf_coh = self.dtd_coherence.confidence if self.dtd_coherence else 0.0
        # #3: Same logic as _compute_mu_scale — coherence primary
        if conf_coh > 0.1:
            return conf_coh
        return max(conf_div, conf_coh)

    def get_filter_state(self) -> AecFilterState:
        """Return the highest-priority active filter state as AecFilterState enum.

        B2 note: This is the *public* API state (7 values: WARMUP, DIVERGED,
        EPC_RECOVERY, DT_ACTIVE, STATIONARY_FAR, CONVERGED, CONVERGING).
        It is distinct from the internal P3f diagnostic string stored in
        _diag['filter_state'] / _prev_filter_state which uses a different
        vocabulary ('idle', 'startup', 'diverged', 'suspicious_dt',
        'refined_usable', 'coarse_learning') and is only used inside the
        filter-state computation block and shadow-mu scheduling.
        """
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
            dt_confidence=self.get_dtd_confidence(),
            dt_from_energy=self._dt_from_energy,
            dt_from_shadow=self._dt_from_shadow,
            dt_from_coherence=d.get('dt_from_coherence', 0.0),
            dt_active=self.is_dtd_active(),
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

    def _aec3_post(self, raw_output: np.ndarray, near_end: np.ndarray,
                   far_end: np.ndarray) -> np.ndarray:
        """v3.21 AEC3-aligned post-stage: bypass legacy ResFilter and run
        AecState + ResidualEchoEstimator + SuppressionGain.

        Takes the linear-filter time-domain residual ``raw_output`` (length =
        hop_size) and returns the suppressed time-domain hop. Per-bin
        suppression gain comes from the AEC3 chain operating on PBFDKF
        spectra; the gain is applied to ``filter.error_spec_windowed`` and
        the result IFFT'd back to time domain over [hop_size:block_size].
        """
        from .filter import build_filter_state_bridge

        n_bins = self.filter.n_freqs
        # AEC3 reference is tuned for int16-magnitude PSDs (samples in
        # ~[-32768, 32767]); soundfile gives us float in [-1, 1] so all
        # absolute thresholds (noise_gate_power, min_noise_floor_power,
        # min_echo_power, audibility floor_power) are 32768^2 ~= 1.07e9
        # too large relative to our PSDs. Worst symptom: `_get_min_gain`
        # computes `min_echo_power / R^2` ~= 64/11 -> clamped to 1.0 ->
        # no suppression. Scale PSDs UP to the AEC3 magnitude convention
        # before they enter ResidualEchoEstimator + SuppressionGain.
        # Gain is a ratio so the scale cancels at apply time.
        _PSD_SCALE = (32768.0) ** 2  # int16 max^2
        near_psd = (np.abs(self.filter.near_spec) ** 2 * _PSD_SCALE).astype(np.float32)
        far_psd = (np.abs(self.filter.far_spec) ** 2 * _PSD_SCALE).astype(np.float32)
        echo_psd = (np.abs(self.filter.echo_spec) ** 2 * _PSD_SCALE).astype(np.float32)
        error_spec = self.filter.error_spec_windowed
        error_psd = (np.abs(error_spec) ** 2 * _PSD_SCALE).astype(np.float32)
        far_pwr = float(np.mean(far_end ** 2))
        # Render block is read by LowNoiseRenderDetector + SaturationDetector
        # using AEC3 absolute thresholds; rescale to int16 amplitude.
        render_block_scaled = (far_end * 32768.0).astype(np.float32)

        # Compute AEC3-style per-frame convergence (subtractor_output_analyzer.cc):
        #   refined_converged = e²_refined < 0.5 * y² AND y² > kConvergenceThreshold
        #   coarse_converged  = e²_coarse  < 0.05 * y² AND y² > kConvergenceThreshold
        #   any_filter_converged = refined OR coarse
        # AEC3 uses int16 power thresholds (50²·64=160000); in our float[-1,1]
        # space that's (50/32768)²·64 ≈ 1.49e-4. This permissive per-frame rule
        # replaces the legacy 10-frame >5 dB ERLE latch (which never fires on
        # hard cases like 9xjhi, leaving SubbandErleEstimator stuck at min_erle
        # = 1.0 -> R²=S²/1 -> SuppressionGain doesn't see correct echo strength).
        # Time-domain energy compare (mirrors AEC3 subtractor_output.cc which
        # uses sum(y[i]² over kBlockSize=64 samples). We sum over hop=160).
        # Threshold = 50²·64 / 32768² × (160/64) ≈ 3.73e-4 to scale-equivalent
        # in float[-1,1] over our hop. Refined = main filter raw_output. Coarse
        # filter's time-domain residual is shadow's near_buffer[-hop:] minus
        # shadow's echo_time — we approximate via shadow.error_spec by inverse
        # FFT, but cheaper to use spectrum energies (Parseval-equivalent for
        # the ratio, only threshold needs adjustment).
        _y2_time = float(np.sum(near_end.astype(np.float64) ** 2))
        _e2_refined = float(np.sum(raw_output.astype(np.float64) ** 2))
        _y2_threshold = 3.73e-4  # 50²·64 / 32768² · (160/64)
        _refined_conv = _e2_refined < 0.5 * _y2_time and _y2_time > _y2_threshold
        # Shadow filter coarse convergence: convert shadow's error_spec to
        # time-domain energy via Parseval. For rfft of length-fft signal:
        # sum(|X[k]|² for k=0..N/2) / N ≈ sum(x[n]² for n=0..N-1) / 2.
        _coarse_conv = False
        if self.shadow_filter is not None and hasattr(self.shadow_filter, 'error_spec'):
            # Parseval-mapped: full-spectrum sum (mirror+reflect 257 bins to 512)
            # ÷ fft_size gives time-domain energy over the fft_size window.
            _e_spec = self.shadow_filter.error_spec
            _e2_coarse = float(
                (2 * np.sum(np.abs(_e_spec[1:-1]) ** 2) + np.abs(_e_spec[0]) ** 2 + np.abs(_e_spec[-1]) ** 2)
                / self.filter.fft_size
            )
            _coarse_conv = _e2_coarse < 0.05 * _y2_time and _y2_time > _y2_threshold
        _aec3_converged = _refined_conv or _coarse_conv

        # Build per-hop filter-state snapshot for AecState.
        bridge = build_filter_state_bridge(
            filter_converged=_aec3_converged,
            pbfdkf=self.filter,
            regime_handler=self._regime_handler,
            mu_final=float(getattr(self, '_last_mu_scale_diag', 1.0)),
            external_delay_samples=int(self._current_delay) if self._delay_active else -1,
            shadow_filter=self.shadow_filter,
        )
        # Build external_delay estimate from legacy delay tracker. AecState's
        # FilterQuality 4-gate AND requires external_delay OR convergence_seen
        # before usable_linear flips True (aec_state.cc:filter_quality.py:58).
        # Without this, the linear branch never engages and we permanently sit
        # in the conservative nonlinear path (R^2 = X^2 * 0.014^2).
        from .delay.delay_types import DelayEstimate, DelayQuality
        if self._delay_active and self._current_delay >= 0:
            ext_delay = DelayEstimate(
                quality=DelayQuality.REFINED, delay=int(self._current_delay)
            )
        else:
            ext_delay = None
        # AEC3 contract: HandleEchoPathChange MUST be called BEFORE Update()
        # (aec_state.cc:148). Consume pending variability accumulated by the
        # legacy event detectors (EPV / shadow_rise / delay) since the last
        # _aec3_post call.
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
            self._aec3_state.handle_echo_path_change(variability)
            self._aec3_pending_gain_change = False
            self._aec3_pending_delay_change = None

        self._aec3_state.update_capture_saturation(self._saturation_level > 0.5)
        self._aec3_state.update(
            bridge=bridge,
            external_delay=ext_delay,
            render_psd=far_psd,
            capture_psd=near_psd,
            error_psd=error_psd,
            echo_psd=echo_psd,
            active_render=(far_pwr > 1e-4),
            render_block=render_block_scaled,
        )

        # v3.21 Phase C.3 — stationarity is updated ONCE per hop in the
        # process loop (right after filter.process) so both the W-gate (see
        # `_block_stationary_for_next_hop`) and the residual scaling here
        # see consistent state. Read the converge flag from the shared
        # counter; the per-hop update lives upstream.
        _filter_converged_enough = (
            self._aec3_stationarity_active_hops
            >= self._aec3_stationarity_converge_hops
        )

        dominant_ne = self._aec3_sg.is_dominant_nearend()

        # v3.21 Phase C.4 — adaptive reverb decay + tail freq response.
        # Lazy-bind the decay estimator on first use (needs n_partitions and
        # hop_size from the filter, only available post __init__).
        self._aec3_ree.attach_reverb_decay_estimator(
            n_partitions=int(self.filter.n_partitions),
            hop_size=int(self.filter.hop_size),
        )
        # Per-partition |W|² spectra. self.filter.W shape (n_partitions, n_freqs)
        # complex; |W|² is the AEC3-equivalent "frequency_response" matrix.
        _w_mag2 = (np.abs(self.filter.W) ** 2).astype(np.float32)
        # Delay in partitions: integer floor of sample-delay / hop_size. -1
        # signals "no usable delay" (estimator skips update).
        _delay_blocks = (int(self._current_delay) // int(self.filter.hop_size)
                         if (self._delay_active and self._current_delay >= 0)
                         else -1)
        # Filter quality proxy: AEC3 uses FilterAnalyzer.consistent_estimate
        # (a stable per-bin convergence score). We don't have that yet; use
        # the per-frame convergence signal we already compute in this scope
        # as a 0/1 indicator, gated by the stationarity-aware "filter has had
        # time to converge" so we don't feed false-positive quality before
        # warmup completes.
        _filter_q = (1.0 if (_aec3_converged and _filter_converged_enough)
                     else None)
        _stationary_block = self._aec3_stationarity.is_block_stationary()
        self._aec3_ree.update_reverb_models(
            frequency_response=_w_mag2,
            filter_delay_blocks=_delay_blocks,
            filter_quality=_filter_q,
            usable_linear_filter=self._aec3_state.usable_linear_estimate(),
            stationary_block=_stationary_block,
        )

        r2, r2_unb = self._aec3_ree.estimate(
            aec_state=self._aec3_state,
            render_psd=far_psd,
            capture_psd=near_psd,
            s2_linear=echo_psd,
            dominant_nearend=dominant_ne,
        )

        # v3.21 Phase C.3 (back half) — Stationarity-driven R² scaling
        # (residual_echo_estimator.cc:303-313). Zero R²/R²_unbounded on
        # stationary bands once the filter has had time to converge so the
        # suppression gain doesn't damp nearend speech on cases with a
        # constant background hum on the far-end (E0l0 / wJVP outliers).
        if _filter_converged_enough:
            _stationary_mask = self._aec3_stationarity.band_stationary_mask()
            if np.any(_stationary_mask):
                r2 = np.where(_stationary_mask, 0.0, r2).astype(np.float32)
                r2_unb = np.where(
                    _stationary_mask, 0.0, r2_unb
                ).astype(np.float32)

        # AEC3 contract (echo_remover.cc:452):
        #   nearend_spectrum = UsableLinearEstimate() ? E² : Y²
        nearend_pwr = error_psd if self._aec3_state.usable_linear_estimate() else near_psd
        # v3.21 Phase C.1 — comfort-noise spectrum derived from per-bin noise
        # floor EMA (mirrors res_refactored/noise_floor_cng.py:23-31). Lazy
        # init at first frame; update only when DT is quiet (legacy
        # is_learning_safe condition is approximated here by `dominant_ne is
        # False AND error magnitude rises slower than noise estimate`).
        if not self._aec3_noise_initialized:
            self._aec3_noise_psd = error_psd.copy() + 1e-8
            self._aec3_noise_initialized = True
        _is_learning_safe = (not dominant_ne) and (far_pwr < 1e-4)
        _alpha_n = np.where(
            error_psd > self._aec3_noise_psd,
            0.998 if _is_learning_safe else 1.0,  # slow up (or freeze on DT)
            0.98,                                 # fast down to track silence
        )
        self._aec3_noise_psd = (
            _alpha_n * self._aec3_noise_psd
            + (1.0 - _alpha_n) * error_psd
        ).astype(np.float32)
        comfort_noise = self._aec3_noise_psd
        gain = self._aec3_sg.get_gain(
            aec_state=self._aec3_state,
            nearend_spectrum=nearend_pwr,
            residual_echo_spectrum=r2,
            residual_echo_spectrum_unbounded=r2_unb,
            comfort_noise_spectrum=comfort_noise,
            render_block=render_block_scaled,
            clock_drift=False,
        )

        # Apply gain in spectrum domain, IFFT to fft_size=512, take the
        # block_size=320 region that holds the analysis window, then
        # synth-window + OLA. error_spec_windowed was built from
        # near_buffer[:block_size] * sqrt-Hann analysis (zero-padded to
        # fft_size). Multiplying it by sqrt-Hann synthesis and accumulating
        # at 50% overlap gives Hann-summed perfect reconstruction.
        e_out_spec = error_spec * gain.astype(error_spec.dtype, copy=False)

        # v3.21 Phase C.1 — CNG injection. Add Gaussian noise shaped by the
        # tracked noise_psd, scaled by smooth_cn_gain = 0.8 EMA of
        # sqrt(max(1 - G², 0)) × 0.4. Mirrors legacy CNG formula. Skip when
        # CNG is disabled in config or noise_psd has not been observed yet
        # (lazy-init guard, same as legacy).
        if (getattr(self.config, 'enable_cng', False)
                and float(np.sum(self._aec3_noise_psd)) > 1e-7):
            _target_cn_gain = (
                np.sqrt(np.maximum(1.0 - gain.astype(np.float32) ** 2, 0.0)) * 0.4
            )
            self._aec3_smooth_cn_gain = (
                0.8 * self._aec3_smooth_cn_gain + 0.2 * _target_cn_gain
            ).astype(np.float32)
            # noise_psd is in float-scale × _PSD_SCALE (we lifted it earlier);
            # convert back to amplitude std by dividing by sqrt(_PSD_SCALE).
            _noise_std = np.sqrt(
                self._aec3_noise_psd / 2.0 / _PSD_SCALE
            ).astype(np.float32)
            _cng_re = np.random.randn(n_bins).astype(np.float32) * _noise_std
            _cng_im = np.random.randn(n_bins).astype(np.float32) * _noise_std
            _cng_spec = (
                self._aec3_smooth_cn_gain * (_cng_re + 1j * _cng_im)
            ).astype(np.complex64)
            e_out_spec = e_out_spec + _cng_spec

        e_out_full = np.fft.irfft(e_out_spec, n=self.filter.fft_size).astype(np.float32)
        bs = self.filter.block_size
        hop = self.filter.hop_size
        windowed = e_out_full[:bs] * self._aec3_synth_window
        self._aec3_ola_buf += windowed
        out = self._aec3_ola_buf[:hop].copy()
        self._aec3_ola_buf[:-hop] = self._aec3_ola_buf[hop:]
        self._aec3_ola_buf[-hop:] = 0.0
        return out.astype(np.float32)

    def enable_res_audit(self) -> None:
        """No-op shim (legacy RES audit infrastructure retired with ResFilter)."""

    def get_res_audit(self):
        """No-op shim (legacy RES audit infrastructure retired with ResFilter)."""
        return None


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
        # v3.8.x: when external config has sample_rate-dependent auto fields
        # already resolved (frame_size/hop_size/filter_length computed in
        # __post_init__ at construction time), updating sample_rate alone
        # leaves stale sizes. Re-resolve auto fields by reverting them to
        # sentinel and re-running __post_init__.
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
    print(f"  DTD: {'enabled' if config.enable_dtd else 'disabled'}")
    print(f"  RES: {'enabled' if config.enable_res else 'disabled'}")
    print()

    aec = AEC(config)
    hop_size = aec.hop_size

    output = np.zeros(num_samples, dtype=np.float32)
    processed = 0
    max_erle = 0.0
    dtd_frames = 0

    while processed + hop_size <= num_samples:
        mic_block = mic_data[processed:processed + hop_size]
        ref_block = ref_data[processed:processed + hop_size]

        out_block = aec.process(mic_block, ref_block)
        output[processed:processed + hop_size] = out_block

        if aec.is_dtd_active():
            dtd_frames += 1

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
    print(f"  DTD active frames: {dtd_frames} ({100 * dtd_frames * hop_size / max(processed, 1):.1f}%)")

    sf.write(out_path, output, mic_sr)
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
    parser.add_argument('--mode', choices=['lms', 'nlms', 'fdaf', 'pbfdaf', 'pbfdkf', 'subband'],
                        default='nlms', help='Filter mode (default: nlms)')
    parser.add_argument('--enable-dtd', action='store_true',
                        help='Enable DTD (default: off, shadow filter provides DT protection)')
    # v3.10.4 (F6): use BooleanOptionalAction with default=None so the CLI
    # only overrides preset values when the user explicitly passes the flag.
    # Previously --enable-res / --cng defaulted to False and unconditionally
    # overrode the preset's True values, making `aec.py mic ref out --preset
    # balanced` silently disable RES + CNG.
    parser.add_argument('--enable-res', default=None,
                        action=argparse.BooleanOptionalAction,
                        help='Enable RES post-filter (default: from preset, else off)')
    parser.add_argument('--res-g-min', type=float, default=-20.0, help='RES min gain (dB)')
    parser.add_argument('--cng', default=None,
                        action=argparse.BooleanOptionalAction,
                        help='Enable comfort noise generation in RES (default: from preset, else off)')
    parser.add_argument('--no-td-constraint', action='store_true',
                        help='Disable time-domain constraint on filter weights (diagnostic)')
    parser.add_argument('--preset', choices=['balanced'],
                        help='Use preset config (overrides RES/adaptive params)')
    parser.add_argument('--no-shadow', action='store_true', help='Disable shadow filter')
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

    # Map mode string to enum
    mode_map = {
        'lms': AecMode.LMS,
        'nlms': AecMode.NLMS,
        'fdaf': AecMode.FDAF,
        'pbfdaf': AecMode.PBFDAF,
        'pbfdkf': AecMode.PBFDKF,
        'subband': AecMode.PBFDKF,  # backward compat
    }

    aec_mode = mode_map[args.mode]

    # Mode-dependent default step size
    mu = args.mu
    if args.mode == 'lms' and args.mu == 0.3:
        mu = 0.01  # LMS needs much smaller step size
    elif args.mode == 'fdaf' and args.mu == 0.3:
        mu = 0.1   # FDAFsingle-block: smaller mu to avoid overshoot

    # filter_length default: auto (-1) or user-specified
    filter_length = args.filter
    if filter_length == 0:
        filter_length = -1  # Auto: 32ms (resolved in __post_init__)

    common_kw = dict(
        mu=mu,
        filter_length=filter_length,
        mode=aec_mode,
        enable_dtd=args.enable_dtd,
        enable_td_constraint=not args.no_td_constraint,
        enable_shadow=not args.no_shadow,
        enable_highpass=not args.no_highpass,
        highpass_cutoff_hz=args.highpass_cutoff,
        enable_saturation_detect=not args.no_saturation_detect,
        clear_filter_history=args.clear_history,
    )
    # Only override preset RES params if user explicitly specified them
    # (don't let CLI default -20dB override preset values like -35dB)
    if not args.preset or args.res_g_min != -20.0:
        common_kw['res_g_min_db'] = args.res_g_min
    # v3.10.4 (F6): only forward enable_res / enable_cng to AecConfig when
    # the user explicitly set them on the CLI. With BooleanOptionalAction +
    # default=None, args.enable_res / args.cng are None unless --[no-]enable-res
    # or --[no-]cng was passed; in that case the preset value (or AecConfig
    # default) is preserved.
    if args.enable_res is not None:
        common_kw['enable_res'] = args.enable_res
    if args.cng is not None:
        common_kw['enable_cng'] = args.cng
    if args.preset:
        config = AecConfig.from_preset(args.preset, **common_kw)
    else:
        config = AecConfig(**common_kw)

    process_wav_files(args.mic, args.ref, args.output, config, diag=args.diag)
