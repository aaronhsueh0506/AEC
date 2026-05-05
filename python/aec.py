"""
Acoustic Echo Cancellation (AEC) - Python Reference Implementation

Supports five filter modes:
- Time-domain NLMS (--mode nlms): sample-by-sample, lowest latency
- Frequency-domain Adaptive Filter (--mode fdaf): single FFT block, no partitions
- Partitioned Block FDAF (--mode pbfdaf): multiple partitions, NLMS adaptation
- Partitioned Block FDKF (--mode pbfdkf): multiple partitions, Kalman adaptation (recommended)

Additional features:
- Double-Talk Detection (DTD)
- Residual Echo Suppressor (RES)

Usage:
    python aec.py mic.wav ref.wav output.wav [--mode nlms|fdaf|pbfdaf|pbfdkf] [--enable-res]
"""

__version__ = "3.8.1"

import os
import numpy as np
from collections import deque
from dataclasses import dataclass
from typing import Optional, Tuple
from enum import Enum
import argparse
import soundfile as sf


class AecMode(Enum):
    LMS = "lms"         # Time-domain LMS (no normalization, simplest)
    NLMS = "nlms"       # Time-domain NLMS (sample-by-sample)
    FDAF = "fdaf"       # Frequency-domain Adaptive Filter (single block, n_partitions=1)
    PBFDAF = "pbfdaf"   # Partitioned Block FDAF (NLMS adaptation)
    PBFDKF = "pbfdkf"   # Partitioned Block FDKF (Kalman adaptation, recommended)
    SUBBAND = "subband" # Backward compat alias (= PBFDKF)


# Frequency-domain modes (partitioned or single block)
_FREQ_MODES = (AecMode.FDAF, AecMode.PBFDAF, AecMode.PBFDKF, AecMode.SUBBAND)
# Partitioned block modes
_PB_MODES = (AecMode.PBFDAF, AecMode.PBFDKF, AecMode.SUBBAND)


class AecPreset(Enum):
    MILD = "mild"               # Lightest echo suppression, best near-end preservation
    SOFT = "soft"               # Between MILD and BALANCED — gentler RES for music / sensitive NE
    BALANCED = "balanced"       # Balanced echo suppression and near-end quality (default)
    AGGRESSIVE = "aggressive"   # Stronger echo suppression, moderate near-end impact
    MAXIMUM = "maximum"         # Maximum echo suppression, significant near-end impact


class AecFilterState(Enum):
    """Algorithmic state of the adaptive filter, in priority order."""
    WARMUP         = "warmup"          # Pre-convergence: warmup frames not yet exhausted
    DIVERGED       = "diverged"        # Filter diverged (output >> mic), adaptation unreliable
    EPC_RECOVERY   = "epc_recovery"    # Echo path changed, filter re-converging (P_MAX raised)
    DT_ACTIVE      = "dt_active"       # Double-talk detected, adaptation frozen / slowed
    STATIONARY_FAR = "stationary_far"  # Stationary far-end (white noise / fan), special gating
    CONVERGED      = "converged"       # Stable convergence, good echo cancellation
    CONVERGING     = "converging"      # Adapting, not yet converged


@dataclass
class AecStats:
    """Comprehensive per-frame statistics for debugging and algorithmic decisions.

    Returned by AEC.get_stats() / AEC.GetStats().
    All power quantities are in dB (20·log10 for amplitudes, 10·log10 for power).
    Confidence scores are in [0, 1].
    """
    # ── Identity ──────────────────────────────────────────────────────────────
    frame_count: int        # Total frames processed since last reset()
    time_s: float           # Elapsed audio time (seconds)

    # ── Filter state ──────────────────────────────────────────────────────────
    filter_state: AecFilterState
    filter_converged: bool
    filter_once_converged: bool  # True if converged at least once since reset
    warmup_remaining: int        # Frames remaining in warmup (0 when complete)

    # ── ERLE — Echo Return Loss Enhancement ───────────────────────────────────
    erle_inst_db: float       # EMA-smoothed instantaneous ERLE
    erle_windowed_db: float   # ~10 s decaying-window ERLE (more stable)
    erle_cumulative_db: float # Full-segment cumulative average ERLE

    # ── Echo path ─────────────────────────────────────────────────────────────
    erl_db: float        # Echo Return Loss estimate (dB); lower = harder (more coupling)
    divergence: float    # [0, 1] filter divergence indicator
    epc_active: bool     # Echo Path Change detector active
    epv_ratio: float     # Echo Path Variability: fast/slow gain ratio (≠1 → path changing)

    # ── Adaptation rate ───────────────────────────────────────────────────────
    mu_scale: float       # [mu_min, 1.0] current adaptation multiplier
    filter_w_norm: float  # Main filter weight L2 norm
    shadow_w_norm: float  # Shadow filter weight L2 norm

    # ── Double-talk detection ─────────────────────────────────────────────────
    dt_confidence: float    # Combined DT confidence [0, 1]
    dt_from_energy: float   # Pre-filter energy signal (immune to ERLE correction)
    dt_from_shadow: float   # Shadow filter DT signal
    dt_from_coherence: float# Coherence-based DT signal (fullband C²)
    dt_active: bool         # True when dt_confidence > 0.5

    # ── Energy levels ─────────────────────────────────────────────────────────
    far_power_db: float    # EMA far-end power (dB)
    mic_power_db: float    # EMA mic power (dB)
    error_power_db: float  # EMA raw filter output power (dB)
    far_activity: float    # [0, 1] far-end activity (from RES)
    saturation_level: float# [0, 1] speaker saturation estimate

    # ── Delay ─────────────────────────────────────────────────────────────────
    delay_samples: int  # Current estimated/fixed delay (−1 = not yet estimated)
    delay_ms: float     # Delay in milliseconds

    # ── Shadow filter ─────────────────────────────────────────────────────────
    shadow_advantage: float  # main_err / shadow_err; >1 means shadow is better
    shadow_copy_count: int   # Total shadow→main copies since reset()
    main_paused: bool        # True = main filter weight update frozen this frame

    # ── RES post-filter ───────────────────────────────────────────────────────
    res_gain_mean_db: float  # Mean spectral gain applied by RES (dB)
    res_using_render: bool   # True = render-based echo estimate active
    echo_psd_mean_db: float  # Mean residual echo PSD estimate (dB)
    error_psd_mean_db: float # Mean error PSD (dB)


@dataclass
class AecResContext:
    """Per-frame context for external RES processing."""
    raw_output: np.ndarray       # (hop_size,) linear AEC output
    echo_spec: np.ndarray        # (n_freqs,) complex echo estimate
    far_power: float             # mean(far_end²)
    far_spec: np.ndarray         # (n_freqs,) complex far-end spectrum
    near_spec: np.ndarray        # (n_freqs,) complex mic spectrum
    filter_converged: bool
    erle_factor: float           # [0, 1] convergence metric
    dt_indicator: float          # [0, 0.8] double-talk confidence
    divergence: float            # [0, 1] divergence indicator
    over_sub: float              # dynamic over_sub value
    saturation_level: float
    erl_estimate: float = 0.01    # E2: dynamic ERL for external RES render-based


@dataclass
class AecConfig:
    """AEC Configuration (all sizes in samples)"""
    sample_rate: int = 16000      # 8000 / 16000 / 48000
    frame_size: int = -1          # Auto: sample_rate * 20ms (160@8k, 320@16k, 960@48k)
    hop_size: int = -1            # Auto: frame_size / 2 (80@8k, 160@16k, 480@48k)
    filter_length: int = -1      # Auto: 32ms (8k/16k) or 64ms (48k)
    mu: float = 0.3              # Step size
    delta: float = 1e-8          # Regularization
    enable_dtd: bool = False
    dtd_hangover_frames: int = 15

    # Geigel DTD parameters (LMS/NLMS)
    dtd_geigel_threshold: float = 0.5     # |mic| > thresh × max(|ref|) → double-talk
    dtd_mu_min_ratio: float = 0.05        # During double-talk, mu drops to 5%
    dtd_confidence_attack: float = 0.3    # Confidence ramp-up rate per block
    dtd_confidence_release: float = 0.05  # Confidence ramp-down rate per block

    # Divergence detection parameters (frequency-domain modes, output-vs-input)
    dtd_divergence_factor: float = 1.5    # output > input × factor → diverged

    # Coherence-based DTD parameters (frequency-domain modes, complements divergence)
    dtd_coh_alpha: float = 0.85           # PSD smoothing factor (~6 block time constant)
    dtd_coh_high: float = 0.6            # Coherence above → no DT (correlated error)
    dtd_coh_low: float = 0.3             # Coherence below → DT (uncorrelated error)
    dtd_coh_energy_floor: float = 0.1    # Min error/far energy ratio to trigger DT
    dtd_coh_hangover: int = 3            # Coherence DTD hangover blocks (shorter than Geigel)
    dtd_coh_release: float = 0.1         # Coherence confidence release rate (faster recovery)

    # RES parameters
    enable_res: bool = True
    res_g_min_db: float = -25.0
    res_over_sub: float = 3.0
    res_alpha: float = 0.8
    enable_cng: bool = False           # Comfort noise generation in RES (off by default)
    enable_td_constraint: bool = True  # Time-domain constraint on filter weights

    # Shadow filter (dual-filter divergence control, frequency-domain modes only)
    enable_shadow: bool = True
    shadow_mu_ratio: float = 1.0
    shadow_copy_threshold: float = 0.65
    shadow_err_alpha: float = 0.80      # D3: 0.85→0.80, faster shadow EMA tracking
    shadow_mu_min: float = 0.5           # Shadow-only mode: DT mu floor (50%)
    shadow_copy_hysteresis: int = 3     # Consecutive frames needed for copy
    shadow_q_ratio: float = 3.0        # Shadow Q = main Q × ratio (FDKF mode)
    shadow_dtd_advantage_scale: float = 3.0  # Shadow DTD: (advantage-offset)/scale → DT confidence
    shadow_dtd_offset: float = 1.5           # Shadow DTD: advantage must exceed this to signal DT

    # Coherence DTD absolute energy floor
    dtd_coh_abs_floor: float = 1e-6     # #8: Absolute error energy floor

    # PBFDKF (Partitioned Block Frequency Domain Kalman Filter) — faster convergence than NLMS
    use_kalman: bool = True           # True=PBFDKF, False=PBFDAF (NLMS)
    kalman_q_high: float = 1e-3     # PBFDKF Q_high convergence speed
    kalman_q_low: float = 1e-6        # D6: 1e-5→1e-6, lower misadjustment (P_floor=1e-4 protects)
    warmup_frames: int = 80          # Frames with forced high mu at startup

    # Echo path change detection (requires shadow filter)
    epc_delta_threshold: float = 0.3    # |ΔE/total_E| < threshold → echo change
    epc_total_rise: float = 1.5         # total_err > prev × rise → errors increasing
    epc_hangover: int = 20              # keep EPC active for N frames after detection
    epc_mu_floor: float = 0.5           # mu_scale floor during EPC

    # Delay estimation (GCC-PHAT)
    enable_delay_est: bool = True       # Enable automatic delay estimation + ref alignment
    max_delay_ms: float = 250.0         # Maximum delay to search (ms)
    delay_est_period_s: float = 0.5     # Re-estimate delay every N seconds
    delay_est_init_s: float = 0.3       # Accumulate this much data before first estimate
    fixed_delay_samples: int = -1       # If >= 0, use this fixed delay instead of estimation

    # High-pass filter (DC blocker + low-freq removal)
    enable_highpass: bool = True
    highpass_cutoff_hz: float = 80.0    # Cutoff freq: removes DC, 50/60Hz hum, rumble

    # Saturation / non-linear echo handling
    enable_saturation_detect: bool = True
    saturation_threshold: float = 0.95       # |sample| > threshold → clipping
    saturation_over_sub_boost: float = 3.0   # Extra over_sub during saturation
    saturation_softclip_ref: bool = True     # Soft-clip reference for better filter modeling

    # RES dynamic over_sub formula: base + scale × erle_factor - dt_reduction × dt_indicator
    res_over_sub_base: float = 2.5           # Unconverged over_sub base
    res_over_sub_scale: float = 4.0          # Scale with erle_factor (converged adds this)
    res_dt_reduction: float = 3.5            # DT reduction coefficient

    # RES anti-blackout
    res_max_drop_db_per_frame: float = 6.0   # Max gain drop per frame (dB)
    res_max_rise_db_per_frame: float = 6.0   # Max gain rise per frame (dB)
    res_spectral_floor: bool = True          # Spectral-shape-preserving gain floor
    res_spectral_floor_db: float = -25.0     # Floor relative to spectral envelope
    res_ne_protect_db: float = -10.0         # Per-bin near-end protection ceiling (dB)

    # RES v2: direct echo estimation + Wiener gain + reverb tail
    res_echo_method: str = "direct"           # "coherence" (legacy) or "direct" (use filter echo est)
    res_gain_type: str = "enr"               # "spectral_sub" (legacy) / "wiener" / "enr"
    res_enable_reverb: bool = False          # Reverb tail model
    res_reverb_decay: float = 0.5            # Exponential decay rate
    res_reverb_gain: float = 1.0             # Reverb contribution scale
    res_alpha_echo_psd: float = 0.7          # Echo PSD smoothing (overridable per preset)
    res_alpha_error_psd: float = 0.8         # Error PSD smoothing (overridable per preset)
    res_enr_scale: float = 1.0              # ENR threshold scale (1.0=AEC3 defaults, <1=more aggressive)
    startup_dt_min_ne_scale: float = 1.0   # Scale min_ne_from_dt when startup_dt (not once_converged); 0.0=disable floor
    startup_dt_gain_floor: float = 1.0    # Cap spectral_g_min during startup_dt (not filter_converged); 1.0=no effect
    startup_dt_noise_floor_scale: float = 1.0  # Scale noise_floor_gain during startup_dt; 0.0=bypass, 1.0=normal
    startup_dt_mu_min: float = 0.0            # Floor mu_scale during startup_dt; 0.0=no override

    # Diagnostic: when True, ResFilter populates per-bin gain vectors at each
    # post-processing stage (read via res.get_stage_gains()). Hot-path cost is
    # one numpy copy per stage per frame; off by default.
    capture_stages: bool = False

    # Mode
    mode: AecMode = AecMode.PBFDKF

    # External RES context
    return_res_context: bool = False   # True → process() returns (output, AecResContext)

    # TIME/LMS history control
    clear_filter_history: bool = False  # Clear ref_buffer each block (default: keep 1 hop history)

    def __post_init__(self):
        if self.frame_size == -1:
            self.frame_size = self.sample_rate * 20 // 1000  # 20ms
        if self.hop_size == -1:
            self.hop_size = self.frame_size // 2             # 10ms
        if self.filter_length == -1:
            # D5: 48kHz needs longer filter (room reverb more prominent)
            # PR-D1 (v3.6.0): bumped 16/8kHz default 32→52ms to match AEC3
            # default (13 blocks × 4ms) — captures more RT60 tail.
            if self.sample_rate >= 44100:
                self.filter_length = self.sample_rate * 64 // 1000  # 64ms
            else:
                self.filter_length = self.sample_rate * 52 // 1000  # 52ms (was 32ms)

    @property
    def fft_size(self) -> int:
        # Next power of 2 >= frame_size (= frame_size when frame_size is power of 2)
        n = self.frame_size
        return 1 << (n - 1).bit_length()

    @classmethod
    def from_preset(cls, preset: 'AecPreset', **kwargs) -> 'AecConfig':
        """Create config from preset with optional overrides.

        Presets (echo suppression strength):
          MILD:       Best near-end preservation, lightest echo suppression
          BALANCED:   Balanced echo suppression and near-end quality (default)
          AGGRESSIVE: Stronger echo suppression, moderate near-end degradation
          MAXIMUM:    Maximum echo suppression, significant near-end impact
        """
        if isinstance(preset, str):
            preset = AecPreset(preset)
        if preset == AecPreset.MILD:
            # v3.8.3: shifted one slot lighter — minimum-touch RES for
            # quiet/light-echo cases where NE intelligibility trumps echo
            # cleanup. Former v3.8.2 MILD values now live in SOFT.
            defaults = dict(
                # RES v2
                res_echo_method="direct",
                res_gain_type="enr",
                res_enable_reverb=True,
                res_reverb_decay=0.45,
                res_reverb_gain=0.4,
                res_alpha_echo_psd=0.6,
                res_alpha_error_psd=0.6,
                res_enr_scale=1.15,
                # RES suppression (ultra-light)
                res_g_min_db=-25.0,
                res_over_sub_base=1.5,
                res_over_sub_scale=2.5,
                res_dt_reduction=4.5,
                res_spectral_floor_db=-18.0,
                res_ne_protect_db=-7.0,
                enable_cng=True,
                shadow_q_ratio=3.0,
                # Adaptive filter
                shadow_mu_min=0.5,
                warmup_frames=80,
                kalman_q_high=1.5e-3,
            )
        elif preset == AecPreset.SOFT:
            # = former v3.8.2 MILD. Shifted one slot to make room for an
            # even lighter MILD; preserved for users who liked the v3.8.2
            # MILD positioning (light RES with audible echo cleanup).
            defaults = dict(
                # RES v2
                res_echo_method="direct",
                res_gain_type="enr",
                res_enable_reverb=True,
                res_reverb_decay=0.6,
                res_reverb_gain=0.8,
                res_alpha_echo_psd=0.5,
                res_alpha_error_psd=0.6,
                res_enr_scale=1.0,
                # RES suppression
                res_g_min_db=-35.0,
                res_over_sub_base=2.5,
                res_over_sub_scale=4.0,
                res_dt_reduction=3.5,
                res_spectral_floor_db=-25.0,
                res_ne_protect_db=-10.0,
                enable_cng=True,
                shadow_q_ratio=3.0,
                # Adaptive filter
                shadow_mu_min=0.5,
                warmup_frames=80,
                kalman_q_high=1.5e-3,
            )
        elif preset == AecPreset.BALANCED:
            defaults = dict(
                # RES v2
                res_echo_method="direct",
                res_gain_type="enr",
                res_enable_reverb=True,
                res_reverb_decay=0.85,    # v3.3: TC ~130ms (was 50ms); RT60-typical
                res_reverb_gain=1.6,      # v3.3: bump (was 1.4); DT gate weakened separately
                res_alpha_echo_psd=0.4,
                res_alpha_error_psd=0.5,
                res_enr_scale=0.85,
                # RES suppression (balanced echo/speech trade-off)
                res_g_min_db=-55.0,
                res_over_sub_base=5.0,
                res_over_sub_scale=9.0,
                res_dt_reduction=2.5,
                res_spectral_floor_db=-38.0,
                res_ne_protect_db=-16.0,
                # v2.7 E6: min-stat noise floor + CNG fill suppression gap
                enable_cng=True,
                shadow_q_ratio=3.5,
                # Adaptive filter
                shadow_mu_min=0.6,
                warmup_frames=80,
                kalman_q_high=1e-3,
            )
        elif preset == AecPreset.AGGRESSIVE:
            defaults = dict(
                # RES v2
                res_echo_method="direct",
                res_gain_type="enr",
                res_enable_reverb=True,
                res_reverb_decay=0.7,
                res_reverb_gain=2.0,
                res_alpha_echo_psd=0.3,
                res_alpha_error_psd=0.4,
                res_enr_scale=0.7,
                # RES suppression (stronger echo suppression)
                res_g_min_db=-65.0,
                res_over_sub_base=7.0,
                res_over_sub_scale=12.0,
                res_dt_reduction=1.5,
                res_spectral_floor_db=-45.0,
                res_ne_protect_db=-22.0,
                enable_cng=True,
                shadow_q_ratio=4.0,
                # Adaptive filter
                shadow_mu_min=0.7,
                warmup_frames=80,
                kalman_q_high=7e-4,
            )
        elif preset == AecPreset.MAXIMUM:
            defaults = dict(
                # RES v2
                res_echo_method="direct",
                res_gain_type="enr",
                res_enable_reverb=True,
                res_reverb_decay=0.8,
                res_reverb_gain=3.0,
                res_alpha_echo_psd=0.2,
                res_alpha_error_psd=0.3,
                res_enr_scale=0.5,
                # RES suppression (maximum, significant near-end impact)
                res_g_min_db=-72.0,
                res_over_sub_base=10.0,
                res_over_sub_scale=15.0,
                res_dt_reduction=0.5,
                res_spectral_floor_db=-55.0,
                res_ne_protect_db=-30.0,
                enable_cng=True,
                shadow_q_ratio=5.0,
                # Adaptive filter
                shadow_mu_min=0.9,
                warmup_frames=100,
                kalman_q_high=7e-4,
            )
        else:
            defaults = {}
        defaults.update(kwargs)
        return cls(**defaults)


class DelayEstimator:
    """GCC-PHAT delay estimator for AEC reference alignment.

    Uses short overlapping segments for fast initial estimation.
    Cross-spectrum is accumulated over segments and smoothed with EMA.
    """

    def __init__(self, sample_rate: int, max_delay_ms: float = 250.0,
                 init_seconds: float = 0.5, period_seconds: float = 2.0):
        self.sample_rate = sample_rate
        self.max_delay_samples = int(max_delay_ms * sample_rate / 1000)
        self.init_seconds = init_seconds
        self.period_seconds = period_seconds

        # Analysis window: 2x max_delay, but at least 2048
        self.seg_size = 1
        min_seg = max(2048, 2 * self.max_delay_samples)
        while self.seg_size < min_seg:
            self.seg_size *= 2
        self.seg_hop = self.seg_size // 2  # 50% overlap
        self.n_freqs = self.seg_size // 2 + 1

        # Smoothed cross-spectrum
        self._cross_spec = np.zeros(self.n_freqs, dtype=np.complex128)
        self._alpha = 0.6
        self._n_updates = 0

        # Sliding buffers (accumulate hop-by-hop)
        self._mic_buf = np.zeros(self.seg_size, dtype=np.float32)
        self._ref_buf = np.zeros(self.seg_size, dtype=np.float32)
        self._buf_pos = 0  # how many samples in current segment

        # State
        self.estimated_delay = -1
        self._samples_accumulated = 0
        self._samples_since_est = 0
        self._init_done = False
        self._init_samples = int(init_seconds * sample_rate)
        self._period_samples = int(period_seconds * sample_rate)
        self._n_estimates = 0
        self._last_par = 0.0  # A5: Peak-to-Average Ratio confidence

    def reset(self):
        self._cross_spec.fill(0)
        self._mic_buf.fill(0)
        self._ref_buf.fill(0)
        self._buf_pos = 0
        self._n_updates = 0
        self.estimated_delay = -1
        self._samples_accumulated = 0
        self._samples_since_est = 0
        self._init_done = False
        self._n_estimates = 0

    def accumulate(self, mic: np.ndarray, ref: np.ndarray) -> bool:
        """Feed mic/ref samples. Returns True if a new delay estimate was made."""
        n = len(mic)
        self._samples_accumulated += n
        self._samples_since_est += n

        # Accumulate into segment buffer
        remaining = n
        src_pos = 0
        while remaining > 0:
            space = self.seg_size - self._buf_pos
            chunk = min(remaining, space)
            self._mic_buf[self._buf_pos:self._buf_pos + chunk] = mic[src_pos:src_pos + chunk]
            self._ref_buf[self._buf_pos:self._buf_pos + chunk] = ref[src_pos:src_pos + chunk]
            self._buf_pos += chunk
            src_pos += chunk
            remaining -= chunk

            if self._buf_pos >= self.seg_size:
                # Segment full — update cross-spectrum
                self._update_cross_spectrum()
                # Shift by seg_hop (50% overlap)
                self._mic_buf[:self.seg_hop] = self._mic_buf[self.seg_hop:]
                self._ref_buf[:self.seg_hop] = self._ref_buf[self.seg_hop:]
                self._buf_pos = self.seg_hop

        # Estimate when enough data accumulated
        if self._n_updates < 2:
            return False

        if not self._init_done:
            if self._samples_accumulated >= self._init_samples:
                self._estimate()
                self._init_done = True
                return True
        else:
            if self._samples_since_est >= self._period_samples:
                self._estimate()
                return True

        return False

    def _update_cross_spectrum(self):
        """Update smoothed cross-spectrum from current segment."""
        mic_spec = np.fft.rfft(self._mic_buf)
        ref_spec = np.fft.rfft(self._ref_buf)
        cross = mic_spec * np.conj(ref_spec)
        self._n_updates += 1
        if self._n_updates == 1:
            self._cross_spec = cross.copy()
        else:
            self._cross_spec = self._alpha * self._cross_spec + (1 - self._alpha) * cross

    def _estimate(self):
        """Estimate delay from accumulated cross-spectrum using GCC-PHAT."""
        magnitude = np.abs(self._cross_spec) + 1e-10
        phat = self._cross_spec / magnitude
        gcc = np.fft.irfft(phat, n=self.seg_size)

        max_d = min(self.max_delay_samples, self.seg_size // 2)

        # Search positive delays (mic lags ref — normal case)
        pos_range = gcc[:max_d + 1]
        best_pos = np.argmax(np.abs(pos_range))

        # A5: PAR (Peak-to-Average Ratio) confidence
        peak = float(np.abs(gcc[best_pos]))
        mean_excl = (np.sum(np.abs(pos_range)) - peak) / (len(pos_range) - 1 + 1e-10)
        self._last_par = float(peak / (mean_excl + 1e-10))

        self.estimated_delay = best_pos
        self._samples_since_est = 0
        self._n_estimates += 1


class NlmsFilter:
    """Time-domain NLMS Adaptive Filter"""

    def __init__(self, filter_length: int, mu: float = 0.3,
                 delta: float = 1e-8,
                 normalize: bool = True):
        self.filter_length = filter_length
        self.mu = mu
        self.delta = delta
        self.normalize = normalize
        self.weights = np.zeros(filter_length, dtype=np.float32)
        self.ref_buffer = np.zeros(filter_length, dtype=np.float32)
        self.power_sum = 0.0
        self.clear_history = False
        self.max_w_norm = 1.5  # Weight norm constraint (prevents explosion during double-talk)

    def reset(self):
        self.weights.fill(0)
        self.ref_buffer.fill(0)
        self.power_sum = 0.0

    def process_sample(self, near_end: float, far_end: float,
                       mu_scale: float = 1.0) -> Tuple[float, float]:
        oldest = self.ref_buffer[-1]
        self.power_sum = max(0, self.power_sum - oldest * oldest + far_end * far_end)
        self.ref_buffer[1:] = self.ref_buffer[:-1]
        self.ref_buffer[0] = far_end
        echo_est = np.dot(self.weights, self.ref_buffer)
        error = near_end - echo_est

        if mu_scale > 0 and self.power_sum > self.delta * self.filter_length:
            if self.normalize:
                mu_eff = (self.mu * mu_scale) / (self.power_sum + self.delta)
            else:
                mu_eff = self.mu * mu_scale
            self.weights += mu_eff * error * self.ref_buffer

        return error, echo_est

    def process_block(self, near_end: np.ndarray, far_end: np.ndarray,
                      mu_scale: float = 1.0) -> Tuple[np.ndarray, np.ndarray]:
        # Optionally clear history (no carry-over between blocks)
        if self.clear_history:
            self.ref_buffer.fill(0)
            self.power_sum = 0.0

        n = len(near_end)
        output = np.zeros(n, dtype=np.float32)
        echo_est = np.zeros(n, dtype=np.float32)
        for i in range(n):
            output[i], echo_est[i] = self.process_sample(
                near_end[i], far_end[i], mu_scale)

        # Weight norm constraint: prevent explosion during double-talk
        w_norm = np.linalg.norm(self.weights)
        if w_norm > self.max_w_norm:
            self.weights *= self.max_w_norm / w_norm

        return output, echo_est


class PBFDAF:
    """
    Partitioned Block Frequency-Domain Adaptive Filter (PBFDAF)

    NLMS-based adaptive filter using overlap-save for linear convolution.
    For Kalman-based adaptation, use PBFDKF subclass.
    """

    def __init__(self, block_size: int, n_partitions: int,
                 mu: float = 0.3, delta: float = 1e-8,
                 hop_size: int = 0):
        # Overlap-save: block_size = 2 × hop (proper 50% ratio for TD constraint)
        # FFT zero-pads to next power of 2 if block_size isn't one
        self.hop_size = hop_size if hop_size > 0 else block_size // 2
        self.block_size = 2 * self.hop_size  # overlap-save buffer (exactly 2× hop)
        self.fft_size = 1 << (self.block_size - 1).bit_length()  # next pow2
        self.n_partitions = n_partitions
        self.n_freqs = self.fft_size // 2 + 1
        self.mu = mu
        self.delta = delta
        self.alpha_power = 0.9
        self.enable_td_constraint = True  # can be disabled for diagnosis

        # Time-domain constraint window: clean 50% truncation
        # block_size = 2×hop → truncation at 50%, minimal Gibbs ringing
        self._td_window = np.ones(self.fft_size, dtype=np.float32)
        fade_len = self.hop_size // 4  # 40 samples for hop=160
        fade = 0.5 * (1.0 - np.cos(np.pi * np.arange(fade_len) / fade_len))
        self._td_window[self.hop_size - fade_len:self.hop_size] = fade[::-1].astype(np.float32)
        self._td_window[self.hop_size:] = 0.0

        # Filter weights [n_partitions, n_freqs]
        self.W = np.zeros((n_partitions, self.n_freqs), dtype=np.complex64)

        # Reference spectrum history [n_partitions, n_freqs]
        self.X_buf = np.zeros((n_partitions, self.n_freqs), dtype=np.complex64)
        self.partition_idx = 0

        # Input buffers (block_size = 2 × hop for overlap-save)
        self.near_buffer = np.zeros(self.block_size, dtype=np.float32)
        self.far_buffer = np.zeros(self.block_size, dtype=np.float32)

        # Power estimation
        self.power = np.zeros(self.n_freqs, dtype=np.float32)

        # Output spectra (for RES / coherence DTD)
        self.near_spec = np.zeros(self.n_freqs, dtype=np.complex64)
        self.echo_spec = np.zeros(self.n_freqs, dtype=np.complex64)
        self.error_spec = np.zeros(self.n_freqs, dtype=np.complex64)
        self.far_spec = np.zeros(self.n_freqs, dtype=np.complex64)

        # Windowed error spectrum for RES analysis (sqrt-Hann, same variance
        # as OLA spec but time-aligned with far_spec/echo_spec for coherence)
        self._sqrt_hann_analysis = np.sqrt(
            np.hanning(self.block_size)).astype(np.float32)
        self.error_spec_windowed = np.zeros(self.n_freqs, dtype=np.complex64)

    def reset(self):
        self.W.fill(0)
        self.X_buf.fill(0)
        self.near_buffer.fill(0)
        self.far_buffer.fill(0)
        self.power.fill(0)
        self.partition_idx = 0
        self.error_spec_windowed.fill(0)

    def process(self, near_end: np.ndarray, far_end: np.ndarray,
                mu_scale=1.0) -> np.ndarray:
        """Process hop_size samples. mu_scale: scalar or per-bin array [n_freqs]."""
        hop = self.hop_size

        # Shift buffers (overlap-save)
        self.near_buffer[:-hop] = self.near_buffer[hop:]
        self.near_buffer[-hop:] = near_end

        self.far_buffer[:-hop] = self.far_buffer[hop:]
        self.far_buffer[-hop:] = far_end

        # FFT (zero-pad block_size buffer to fft_size, cast to float32 precision)
        near_spec = np.fft.rfft(self.near_buffer, self.fft_size).astype(np.complex64)
        far_spec = np.fft.rfft(self.far_buffer, self.fft_size).astype(np.complex64)
        self.near_spec = near_spec  # expose for RES overlap-save
        self.far_spec = far_spec  # expose for coherence DTD

        # Store far-end spectrum
        curr_p = self.partition_idx
        self.X_buf[curr_p] = far_spec

        # Update power estimate (cold start: initialize directly on first active frame)
        far_psd = np.abs(far_spec) ** 2
        if np.sum(self.power) < 1e-10 and np.sum(far_psd) > 1e-10:
            self.power = far_psd.astype(np.float32)
        else:
            self.power = (self.alpha_power * self.power +
                         (1 - self.alpha_power) * far_psd)

        # Compute echo estimate
        self.echo_spec.fill(0)
        for p in range(self.n_partitions):
            p_idx = (curr_p - p) % self.n_partitions
            self.echo_spec += self.W[p] * self.X_buf[p_idx]

        # IFFT (fft_size → take block_size valid samples)
        echo_time = np.fft.irfft(self.echo_spec, self.fft_size).astype(np.float32)

        # Error (take valid region for output)
        output = self.near_buffer[-hop:] - echo_time[self.hop_size:self.block_size]

        # Error spectrum (zero-pad to fft_size, valid region at [hop, block_size))
        error_time = np.zeros(self.fft_size, dtype=np.float32)
        error_time[self.hop_size:self.block_size] = output
        self.error_spec = np.fft.rfft(error_time).astype(np.complex64)

        # Windowed error spec for RES analysis: near_buffer × sqrt-Hann − echo_spec.
        # Same time alignment as far_spec/echo_spec, but with sqrt-Hann variance
        # (low inter-frame noise). Used for coherence/PSD/ENR in ResFilter.
        near_win = self.near_buffer[:self.block_size] * self._sqrt_hann_analysis
        near_spec_win = np.fft.rfft(near_win, self.fft_size).astype(np.complex64)
        self.error_spec_windowed = near_spec_win - self.echo_spec

        # Update weights — gate on far-end activity
        far_hop_energy = np.sum(far_end ** 2) / hop
        if far_hop_energy > 1e-4:  # ~ -40 dBFS, unified with far_active threshold
            self._update_weights(curr_p, mu_scale)

        self.partition_idx = (self.partition_idx + 1) % self.n_partitions
        return output.astype(np.float32)

    def _update_weights(self, curr_p: int, mu_scale):
        """NLMS weight update."""
        mu_scale_arr = np.asarray(mu_scale, dtype=np.float32)
        if mu_scale_arr.ndim == 0:
            mu_scale_arr = np.full(self.n_freqs, float(mu_scale_arr), dtype=np.float32)
        if not np.any(mu_scale_arr > 0):
            return
        # Per-bin local floor: allows low-energy mid-freq bins higher effective mu
        local_floor = self.power * 0.01 + self.delta        # per-bin 1% floor
        global_floor = np.mean(self.power) * 0.001 + self.delta  # global extreme floor
        power_floor = np.maximum(self.power, np.maximum(local_floor, global_floor))
        mu_eff = (self.mu * mu_scale_arr) / (power_floor * self.n_partitions + self.delta)
        for p in range(self.n_partitions):
            p_idx = (curr_p - p) % self.n_partitions
            grad = self.error_spec * np.conj(self.X_buf[p_idx])
            self.W[p] += mu_eff * grad
            # Time-domain constraint: fade out non-causal part (raised cosine)
            if self.enable_td_constraint:
                w_time = np.fft.irfft(self.W[p], self.fft_size).astype(np.float32)
                w_time *= self._td_window
                self.W[p] = np.fft.rfft(w_time).astype(np.complex64)

    def get_error_energy(self) -> float:
        return float(np.sum(np.abs(self.error_spec) ** 2))

    def copy_weights_from(self, src: 'PBFDAF'):
        self.W[:] = src.W


class PBFDKF(PBFDAF):
    """
    Partitioned Block Frequency-Domain Kalman Filter (PBFDKF)

    Extends PBFDAF with per-bin Kalman gain for faster convergence
    and automatic step-size adaptation.
    """

    def __init__(self, block_size: int, n_partitions: int,
                 mu: float = 0.3, delta: float = 1e-8,
                 hop_size: int = 0):
        super().__init__(block_size, n_partitions, mu, delta, hop_size)

        # P: error covariance (real, per-partition per-bin)
        self.P = np.ones((n_partitions, self.n_freqs), dtype=np.float32) * 0.01
        # Q: process noise — controls adaptation speed (two-stage)
        self.Q_high = np.ones(self.n_freqs, dtype=np.float32) * 1e-4
        self.Q_low  = np.ones(self.n_freqs, dtype=np.float32) * 1e-5
        self.Q = self.Q_high.copy()
        # R: measurement noise PSD (estimated from error)
        self.R = np.ones(self.n_freqs, dtype=np.float32) * 1e-2
        self._error_psd = np.ones(self.n_freqs, dtype=np.float32) * 1e-2
        self._alpha_r = 0.95   # faster R tracking for DT protection

        # GPT Phase 1 debug trace (off by default, zero overhead).
        # When enabled, accumulates per-frame stats to verify hypothesis:
        # "DT 期間 mu_scale 壓低但 P 仍因 K_optimal 快下降 → DT 結束後 P 偏低 → recovery 慢"
        self._enable_kx_trace = False
        self._kx_trace = []  # list of dicts, one per call to _update_weights

    def reset(self):
        super().reset()
        self.P.fill(0.01)
        self.R.fill(1e-2)
        self._error_psd.fill(1e-2)
        self.Q[:] = self.Q_high
        for attr in ('_p_max_override', '_p_max_override_frames',
                     '_p_floor_beta', '_p_floor_beta_frames'):
            if hasattr(self, attr):
                delattr(self, attr)

    def _update_weights(self, curr_p: int, mu_scale):
        """Frequency-Domain Kalman Filter weight update."""
        mu_scale_arr = np.asarray(mu_scale, dtype=np.float32)
        if mu_scale_arr.ndim == 0:
            mu_scale_arr = np.full(self.n_freqs, float(mu_scale_arr), dtype=np.float32)

        # Update measurement noise estimate from error PSD
        error_psd = np.abs(self.error_spec) ** 2
        self._error_psd = self._alpha_r * self._error_psd + (1 - self._alpha_r) * error_psd
        self.R = np.maximum(self._error_psd, self.delta)

        # Adaptive R: scale by mu_scale to break R-deadlock
        mu_mean = float(np.mean(mu_scale_arr))
        R_scale = 0.1 + 0.9 * (1.0 - mu_mean)
        self.R = self.R * R_scale

        # Q modulation: reduce Q during DT (Kalman-specific, not in NLMS)
        q_scale = 0.1 + 0.9 * mu_mean  # FS: 1.0, strong DT: 0.1
        Q_modulated = self.Q * q_scale

        # Per-bin far-end activity mask
        far_power_smooth = self.power
        far_activity_mask = far_power_smooth > (np.mean(far_power_smooth) * 0.01 + 1e-6)
        Q_floor = Q_modulated * 0.05
        Q_gated = np.where(far_activity_mask, Q_modulated, Q_floor)

        # P_MAX: overridable during EPC for faster re-convergence
        p_max = getattr(self, '_p_max_override', 0.5)
        if hasattr(self, '_p_max_override_frames'):
            self._p_max_override_frames -= 1
            if self._p_max_override_frames <= 0:
                self._p_max_override = 0.5
                del self._p_max_override_frames

        # P-floor: prevent P from collapsing to delta after long convergence
        # (which kills Kalman gain on movement). EPC sets beta=1.0 transiently
        # to force full re-tracking; steady state uses beta=0.1.
        beta = getattr(self, '_p_floor_beta', 0.1)
        if hasattr(self, '_p_floor_beta_frames'):
            self._p_floor_beta_frames -= 1
            if self._p_floor_beta_frames <= 0:
                self._p_floor_beta = 0.1
                del self._p_floor_beta_frames
        P_floor = self.Q_high * beta

        # Global denominator: sum over ALL partitions (correct Kalman theory)
        # C-parity fix: cast delta to float32 to prevent denominator from
        # promoting to float64, which cascades K_optimal to complex128 and
        # causes divergence from C's float32-only arithmetic.
        total_echo_var = np.zeros(self.n_freqs, dtype=np.float32)
        for p in range(self.n_partitions):
            p_idx = (curr_p - p) % self.n_partitions
            X = self.X_buf[p_idx]
            total_echo_var += self.P[p] * (np.abs(X) ** 2)
        denominator = total_echo_var + self.R + np.float32(self.delta)

        # GPT Phase 1 trace: per-partition KX_optimal vs KX_scaled accumulator.
        if self._enable_kx_trace:
            kx_opt_acc = []
            kx_scaled_acc = []
            p_before_acc = []
            p_after_acc = []

        for p in range(self.n_partitions):
            p_idx = (curr_p - p) % self.n_partitions
            X = self.X_buf[p_idx]

            K_optimal = (self.P[p] * np.conj(X)) / denominator

            # Bug 2 fix: separate K for weights (scaled) and P update (unscaled)
            K_scaled = K_optimal * mu_scale_arr

            self.W[p] += K_scaled * self.error_spec

            # PR-G1 (v3.7 candidate, GPT Phase 1): blended KX for P update.
            # KX trace 2026-04-30 confirmed hypothesis on DT_st bucket: when
            # mu_scale=0 (full DT freeze), W stays put but P drops 72% per
            # cycle via K_optimal — P 與 W 不一致，DT 結束後 K 偏低、recovery 慢.
            # Blend KX_optimal (current) with KX_scaled (W-consistent) by
            # mu_mean: FS (mu=1) → 100% optimal, DT (mu=0) → 100% scaled,
            # smooth transition between. Avoids both extremes' regression
            # risk (full KX_optimal: P over-confident in DT; full KX_scaled:
            # P over-cautious in steady state with weak far / per-bin gating).
            KX_optimal = np.real(K_optimal * X).astype(np.float32)
            KX_scaled = np.real(K_scaled * X).astype(np.float32)
            mu_mean_f32 = np.float32(mu_mean)
            KX = mu_mean_f32 * KX_optimal + (np.float32(1.0) - mu_mean_f32) * KX_scaled
            if self._enable_kx_trace:
                kx_opt_acc.append(float(np.mean(KX_optimal)))
                kx_scaled_acc.append(float(np.mean(KX_scaled)))
                p_before_acc.append(float(np.mean(self.P[p])))
            self.P[p] = np.minimum(
                np.maximum((np.float32(1.0) - KX) * self.P[p] + Q_gated, P_floor),
                p_max
            )
            if self._enable_kx_trace:
                p_after_acc.append(float(np.mean(self.P[p])))

            # Time-domain constraint (raised cosine fade)
            if self.enable_td_constraint:
                w_time = np.fft.irfft(self.W[p], self.fft_size).astype(np.float32)
                w_time *= self._td_window
                self.W[p] = np.fft.rfft(w_time).astype(np.complex64)

        if self._enable_kx_trace:
            self._kx_trace.append({
                'mu_mean': float(np.mean(mu_scale_arr)),
                'kx_opt_mean': float(np.mean(kx_opt_acc)),
                'kx_scaled_mean': float(np.mean(kx_scaled_acc)),
                'p_before_mean': float(np.mean(p_before_acc)),
                'p_after_mean': float(np.mean(p_after_acc)),
                'p_p10': float(np.percentile([np.percentile(self.P[p], 10) for p in range(self.n_partitions)], 50)),
                'p_p50': float(np.median([np.median(self.P[p]) for p in range(self.n_partitions)])),
                'p_p90': float(np.percentile([np.percentile(self.P[p], 90) for p in range(self.n_partitions)], 50)),
                'q_gated_mean': float(np.mean(Q_gated)),
                'far_power_mean': float(np.mean(self.power)),
                'error_power_mean': float(np.mean(self._error_psd)),
            })

    def copy_weights_from(self, src: 'PBFDAF'):
        self.W[:] = src.W
        # Only copy filter coefficients W, not Kalman internal state (P/Q/R).
        # P/Q/R are confidence states accumulated from each filter's own
        # learning history — copying them across filters contaminates the
        # destination's Kalman gain. AEC3's shadow uses NLMS (no state to
        # copy); our PBFDKF shadow needs this protection.
        # After W copy, P may temporarily mismatch W but Kalman self-corrects
        # within a few frames.


# Backward compatibility alias
SubbandNlms = PBFDKF


class FilterErleEstimator:
    """Per-bin ERLE from adaptive filter echo estimate vs error.
    erle[k] = |echo_spec[k]|² / |error_spec[k]|²

    Key difference from erle_per_bin:
    - Does NOT use near_psd → breaks circular dependency
    - DT: erle naturally drops (speech increases error, echo_spec stays)
    """
    def __init__(self, n_freqs: int):
        self.n_freqs = n_freqs
        self.erle = np.ones(n_freqs, dtype=np.float32)
        self._alpha_rise = 0.95   # slow rise (stable convergence)
        self._alpha_drop = 0.7    # fast drop (DT protection)

    def update(self, echo_spec: np.ndarray, error_spec: np.ndarray,
               far_active: bool, dt_indicator: float) -> None:
        if not far_active:
            return
        echo_pwr = np.abs(echo_spec) ** 2
        error_pwr = np.abs(error_spec) ** 2 + 1e-10
        inst_erle = np.clip(echo_pwr / error_pwr, 0.1, 1000.0)

        # Asymmetric EMA: fast drop (DT protection), slow rise (FS convergence).
        # Bug 2 fix: dt_indicator now freezes rise (alpha_rise→1), instead of
        # inverted dt_weight that accelerated rise during DT (echo leak).
        dt_factor = float(np.clip(dt_indicator, 0.0, 1.0))
        alpha_rise_eff = self._alpha_rise + (1.0 - self._alpha_rise) * dt_factor
        alpha = np.where(
            inst_erle < self.erle,
            self._alpha_drop,
            alpha_rise_eff,
        )
        self.erle = alpha * self.erle + (1.0 - alpha) * inst_erle

        # 3-bin smoothing + cap
        kernel = np.array([0.25, 0.5, 0.25], dtype=np.float32)
        self.erle = np.convolve(self.erle, kernel, mode='same').astype(np.float32)
        self.erle = np.clip(self.erle, 0.5, 200.0)

    def reset(self):
        self.erle.fill(1.0)


class FullbandErleEstimator:
    """Broadband ERLE for cross-validation confidence.
    Uses near_psd/error_psd broadband mean — stable but slow, FS-only update.
    """
    def __init__(self):
        self.fb_erle = 1.0
        self._alpha = 0.97

    def update(self, near_power: float, error_power: float,
               far_active: bool, dt_indicator: float) -> None:
        if not far_active or dt_indicator > 0.3 or near_power < 1e-8:
            return
        inst = np.clip(near_power / (error_power + 1e-10), 0.5, 100.0)
        self.fb_erle = self._alpha * self.fb_erle + (1.0 - self._alpha) * inst

    def reset(self):
        self.fb_erle = 1.0


def compute_erle_confidence(erle_l1: np.ndarray, fb_erle: float) -> float:
    """Compare FilterErle mean (L1) with FullbandErle (L2).
    Returns confidence in [0, 1]: 1 = consistent, 0 = divergent.
    """
    l1_mean = float(np.mean(erle_l1))
    if l1_mean < 0.5 or fb_erle < 0.5:
        return 0.0
    log_diff = abs(np.log(l1_mean + 1e-10) - np.log(fb_erle + 1e-10))
    return float(np.exp(-log_diff / 2.0))


class HighPassFilter:
    """2nd-order Butterworth IIR high-pass filter (bilinear transform).

    Removes DC offset, 50/60Hz hum, and low-frequency rumble.
    12 dB/octave rolloff. Processes sample-by-sample with two delay states.
    """

    def __init__(self, cutoff_hz: float, sample_rate: int):
        # Bilinear transform: pre-warp analog frequency
        wc = 2.0 * np.pi * cutoff_hz / sample_rate
        wc_w = np.tan(wc / 2.0)
        k = wc_w * wc_w
        sqrt2 = np.sqrt(2.0)
        norm = 1.0 / (1.0 + sqrt2 * wc_w + k)

        # Transfer function coefficients (Direct Form II)
        self.b0 = norm
        self.b1 = -2.0 * norm
        self.b2 = norm
        self.a1 = 2.0 * (k - 1.0) * norm
        self.a2 = (1.0 - sqrt2 * wc_w + k) * norm

        # Delay states
        self.z1 = 0.0
        self.z2 = 0.0

    def process(self, x: np.ndarray) -> np.ndarray:
        """Process a block of samples through the HP filter."""
        out = np.empty_like(x)
        b0, b1, b2, a1, a2 = self.b0, self.b1, self.b2, self.a1, self.a2
        z1, z2 = self.z1, self.z2
        for i in range(len(x)):
            xi = float(x[i])
            yi = b0 * xi + z1
            z1 = b1 * xi - a1 * yi + z2
            z2 = b2 * xi - a2 * yi
            out[i] = yi
        self.z1, self.z2 = z1, z2
        return out

    def reset(self):
        self.z1 = 0.0
        self.z2 = 0.0


class SaturationDetector:
    """Detects speaker clipping/saturation in audio signals.

    Returns a smoothed saturation_level in [0, 1] indicating how much
    non-linear distortion is present. Also provides soft-clipping to
    model the speaker's saturation behavior for the adaptive filter.
    """

    def __init__(self, threshold: float = 0.95):
        self.threshold = threshold
        self.saturation_level = 0.0
        self.alpha_attack = 0.3    # Fast attack when saturation detected
        self.alpha_release = 0.98  # Slow release (echo path retains saturation effects)

    def detect(self, signal: np.ndarray) -> float:
        """Detect saturation level in signal. Returns smoothed level in [0, 1]."""
        n = len(signal)
        if n == 0:
            return self.saturation_level

        abs_sig = np.abs(signal)
        # Count clipped samples
        clip_count = np.sum(abs_sig > self.threshold)

        # Count consecutive identical peak samples (digital clipping signature)
        consec_count = 0
        high_mask = abs_sig > self.threshold * 0.8
        for i in range(1, n):
            if high_mask[i] and high_mask[i - 1] and abs(signal[i] - signal[i - 1]) < 1e-6:
                consec_count += 1

        raw_sat = min((clip_count + 2 * consec_count) / n, 1.0)

        # Asymmetric EMA
        if raw_sat > self.saturation_level:
            alpha = self.alpha_attack
        else:
            alpha = self.alpha_release
        self.saturation_level = alpha * self.saturation_level + (1.0 - alpha) * raw_sat
        return self.saturation_level

    @staticmethod
    def soft_clip(signal: np.ndarray, knee: float = 0.8) -> np.ndarray:
        """Soft-clip signal to model speaker saturation behavior.

        Below knee: pass through. Above knee: tanh compression.
        """
        out = signal.copy()
        abs_sig = np.abs(signal)
        mask = abs_sig >= knee
        if np.any(mask):
            sign = np.sign(signal[mask])
            excess = abs_sig[mask] - knee
            scale = 1.0 - knee
            compressed = knee + np.tanh(excess / max(scale, 1e-6)) * scale
            out[mask] = sign * compressed
        return out

    def reset(self):
        self.saturation_level = 0.0


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
                 capture_stages: bool = False):
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
        # C1: stationary DT mask
        self._stat_dt_mask = np.zeros(n_freqs, dtype=np.float32)
        self._stat_dt_mask[(f_bins >= 300) & (f_bins <= 3000)] = 0.8
        low = (f_bins > 100) & (f_bins < 300)
        self._stat_dt_mask[low] = 0.8 * ((f_bins[low] - 100.0) / 200.0)
        high = (f_bins > 3000) & (f_bins < 4000)
        self._stat_dt_mask[high] = 0.8 * ((4000.0 - f_bins[high]) / 1000.0)
        # C2: ENR blend array
        self._enr_blend = np.clip((np.arange(n_freqs, dtype=np.float32) - 5) / 5, 0, 1)
        # C3: frequency bin indices
        self._hf_cap_bin = min(int(500.0 / freq_res), n_freqs - 1)
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
        # Indices: 0=softgate_emr, 1=spectral_floor, 2=epc_dt_cap, 3=quiet_mask,
        #          4=3bin_smooth, 5=hf_cap, 6=pre_temporal, 7=post_temporal,
        #          8=after_noise_lift (final gain_smooth)
        self._diag_round5_stages = np.zeros(9, dtype=np.float32)

        # Per-bin gain capture (full vectors, opt-in via capture_stages)
        self._capture_stages = capture_stages
        self._stage_gains = {}

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

        # E1: call reset() to initialize all runtime scalar state
        # from a single source of truth (avoids init/reset divergence)
        self.reset()

    def reset(self):
        """Reset all runtime state. Arrays are .fill(0), scalars are set."""
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
            self._residual_est.reset()
        self._diag_gain_mean = 1.0
        self._diag_gain_min = 1.0
        self._diag_effective_g_min = 1.0
        self._diag_far_activity = 0.0
        self._diag_echo_psd_mean = 0.0
        self._diag_error_psd_mean = 0.0
        self._filter_erle_est.reset()
        self._fb_erle_est.reset()

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
        self._stats_last_gain_before_floor = 1.0
        self._stats_last_gain_after_floor = 1.0
        self._stats_last_gain_after_smoothing = 1.0
        self._stats_last_noise_floor_gain = 0.0
        self._stats_last_noise_psd = 0.0
        self._stats_last_spec_pwr = 0.0
        self._stats_last_nfl_lifted = False

    def get_stage_gains(self):
        """Return dict of per-bin gain vectors captured this frame.

        Empty dict unless `capture_stages=True` was passed to __init__.
        Keys: '01_softgate_emr', '02_spectral_floor', '03_epc_dt_cap',
        '04_quiet_mask', '05_3bin_smooth', '06_hf_cap', '07_pre_temporal',
        '08_post_temporal'. Vectors are np.float32 length n_freqs.
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

            # Cap 1: echo_psd × 2.0 (skipped in render-mode)
            if not self._residual_est.using_render_based:
                residual_echo_psd = np.minimum(residual_echo_psd, self.echo_psd * 2.0)
            if self._stats is not None:
                self._stats_last_res_after_echo_cap = float(np.mean(residual_echo_psd))

            # Cap 2: error_psd × (1.5 if render else 1.0)
            err_cap_mult = 1.5 if self._residual_est.using_render_based else 1.0
            residual_echo_psd = np.minimum(residual_echo_psd, self.error_psd * err_cap_mult)
            if self._stats is not None:
                self._stats_last_res_after_error_cap = float(np.mean(residual_echo_psd))

            # Cap 3: dt_suppress (skipped in render-mode)
            if not self._residual_est.using_render_based:
                dt_suppress = np.clip(1.0 - dt_for_fs**2, 0.1, 1.0)
                residual_echo_psd = np.minimum(residual_echo_psd, self.error_psd * dt_suppress)
            if self._stats is not None:
                self._stats_last_res_after_dt_cap = float(np.mean(residual_echo_psd))

            # Cap 4: render_ceil (skipped in render-mode)
            if far_spec is not None and far_power > 1e-4 and erl_estimate > 0.0:
                far_psd_k = np.abs(far_spec) ** 2
                render_ceil = far_psd_k * min(erl_estimate * 2.0, 1.0)
                if self._stats is not None:
                    self._stats_last_render_ceil_mean = float(np.mean(render_ceil))
                    self._stats_last_erl_estimate = float(erl_estimate)
                if not self._residual_est.using_render_based:
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
                              spectral_g_min, eps):
        """Stage 2: ENR / Wiener / spectral_sub gain compute + EMR + spectral floor lift.

        Returns g (post-spectral-floor). Mutates dominant_ne / Round 4 / Round 5
        diag caches and stats.
        """
        if self.gain_type == "enr" and residual_echo_psd is not None:
            raw_nearend_est = np.maximum(self.error_psd - residual_echo_psd, 0.0)
            noise_floor_psd = np.mean(self.error_psd) * 0.01 + 1e-10

            # Per-bin DT indicator: base from coh2 (works for speech far-end)
            dt_per_bin = np.maximum(
                np.full(self.n_freqs, effective_dt, dtype=np.float32),
                1.0 - coh2
            )
            if is_stationary_dt:
                dt_per_bin = np.maximum(dt_per_bin, self._stat_dt_mask)

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

            # Round 4 trace cache (audio-passive)
            self._diag_nearend_est_last = nearend_est
            self._diag_residual_echo_psd_last = residual_echo_psd

            enr = residual_echo_psd / (nearend_est + 1e-10)

            blend = self._enr_blend
            scale = self.enr_scale
            ne_confidence = dt_per_bin
            effective_scale = scale
            enr_t_ne = (1 - blend) * 2.0 + blend * 1.5
            enr_s_ne = (1 - blend) * 3.0 + blend * 2.5
            enr_t_fs = (1 - blend) * (0.3 * effective_scale) + blend * (0.07 * effective_scale)
            enr_s_fs = (1 - blend) * (0.4 * effective_scale) + blend * (0.1 * effective_scale)
            if effective_dt > 0.4:
                dt_enr_relax = 1.0 + (effective_dt - 0.4) / 0.6 * 0.5
                enr_t_ne = enr_t_ne * dt_enr_relax
                enr_s_ne = enr_s_ne * dt_enr_relax
            enr_t = ne_confidence * enr_t_ne + (1 - ne_confidence) * enr_t_fs
            enr_s = ne_confidence * enr_s_ne + (1 - ne_confidence) * enr_s_fs
            min_gate_width = 0.2
            enr_s_safe = np.maximum(enr_s, enr_t + min_gate_width)

            g = np.where(enr > enr_t,
                         np.clip((enr_s_safe - enr) / (enr_s_safe - enr_t + eps), 0.0, 1.0),
                         1.0)

            # EMR: AEC3-style noise masking
            if np.sum(self.noise_psd) > 0:
                emr = residual_echo_psd / (self.noise_psd + 1e-10)
                emr_transparent = 0.3
                g_emr = np.clip(emr_transparent / (emr + 1e-10), 0.0, 1.0)
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

    def _stage_gain_postprocess(self, *, g_in, epc_dt, quiet_mask, far_power,
                                  effective_dt, is_stationary_dt, divergence):
        """Stage 3: EPC_DT cap, quiet mask, 3-bin smooth, HF cap, divergence override.

        Returns updated g (post-divergence-override). Mutates _diag_round5_stages[2..6].
        """
        g = g_in
        # EPC_DT gain cap: echo path changed + DT → cap gain to force minimum echo suppression.
        if epc_dt:
            EPC_DT_GAIN_CAP = 0.85
            g = np.minimum(g, EPC_DT_GAIN_CAP)
        if getattr(self, '_capture_stages', False):
            self._stage_gains['03_epc_dt_cap'] = g.copy()
        self._diag_round5_stages[2] = float(np.mean(g[self._voice_band_idx])) if self._voice_band_idx.size > 0 else 0.0

        g[quiet_mask] = 1.0  # Noise gate: pass through quiet bins
        if getattr(self, '_capture_stages', False):
            self._stage_gains['04_quiet_mask'] = g.copy()
        self._diag_round5_stages[3] = float(np.mean(g[self._voice_band_idx])) if self._voice_band_idx.size > 0 else 0.0

        # --- Frequency-domain postprocessing (cf. AEC3 PostprocessGains) ---
        if far_power > 1e-4:
            # 3-bin cross-frequency smoothing
            kernel = np.array([0.25, 0.5, 0.25], dtype=np.float32)
            g = np.convolve(g, kernel, mode='same').astype(np.float32)
            if getattr(self, '_capture_stages', False):
                self._stage_gains['05_3bin_smooth'] = g.copy()
            self._diag_round5_stages[4] = float(np.mean(g[self._voice_band_idx])) if self._voice_band_idx.size > 0 else 0.0
            # DC consistency: bins 0-1 follow bin 2
            if self.n_freqs > 2:
                g[:2] = np.minimum(g[1], g[2])
            # HF cap: upper bins capped at gain of bin near ~500Hz
            hf_cap_bin = self._hf_cap_bin
            if self.n_freqs > hf_cap_bin + 1 and effective_dt < 0.5 and not is_stationary_dt:
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
                aec_state=None) -> np.ndarray:
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
            if filter_converged:
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

        # EPC_DT: echo path change detected AND double-talk active.
        # Gain cap bypasses ENR path (locked ~1.0 by DT nearend protection).
        epc_dt = epc_active and effective_dt > 0.35

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

        # --- Per-bin near-end gate with fs_confidence ---
        ne_erle_gate = max(erle_factor, 0.3)  # B4: simplified (0.2 floor never triggered)
        # Scale ne_protection by (1-fs_confidence): FS→no protection, DT/NE→full protection
        ne_protection = (1.0 - coh2) * ne_erle_gate * (1.0 - fs_confidence)
        ne_g_min_ceil = 10 ** (self.ne_protect_db / 20)
        ne_g_floor = effective_g_min + (ne_g_min_ceil - effective_g_min) * ne_protection
        ne_g_floor = np.maximum(ne_g_floor, effective_g_min)
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
        )
        g = self._stage_gain_postprocess(
            g_in=g,
            epc_dt=epc_dt,
            quiet_mask=quiet_mask,
            far_power=far_power,
            effective_dt=effective_dt,
            is_stationary_dt=is_stationary_dt,
            divergence=divergence,
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


class DtdEstimator:
    """Double-Talk Detector with per-mode strategy.

    - 'geigel' mode (LMS/NLMS): Geigel DTD with hangover + confidence
    - 'divergence' mode (frequency-domain modes): Output-vs-input divergence detection
    - 'coherence' mode (frequency-domain modes): Error-reference coherence DT detection
    """

    def __init__(self, mode: str = 'geigel', *,
                 window_blocks: int = 4,
                 geigel_threshold: float = 0.5,
                 hangover_max: int = 15,
                 divergence_factor: float = 1.5,
                 attack: float = 0.3,
                 release: float = 0.05,
                 warmup_frames: int = 50,
                 # Coherence mode params
                 n_freqs: int = 0,
                 coh_alpha: float = 0.85,
                 coh_high: float = 0.6,
                 coh_low: float = 0.3,
                 coh_energy_floor: float = 0.01,
                 coh_abs_floor: float = 1e-6,
                 sample_rate: int = 16000,
                 block_size: int = 512):
        self.mode = mode  # 'geigel', 'divergence', or 'coherence'
        self.confidence = 0.0
        self.attack = attack
        self.release = release
        self.warmup_frames = warmup_frames
        self.frame_count = 0

        # Geigel state
        self.far_abs_buffer = np.zeros(max(1, window_blocks))
        self.buf_idx = 0
        self.geigel_threshold = geigel_threshold
        self.hangover_max = hangover_max
        self.hangover_count = 0

        # Divergence state
        self.divergence_factor = divergence_factor

        # Coherence state
        self.coh_alpha = coh_alpha
        self.coh_high = coh_high
        self.coh_low = coh_low
        self.coh_energy_floor = coh_energy_floor
        self.coh_abs_floor = coh_abs_floor  # #8: Absolute energy floor
        if mode == 'coherence' and n_freqs > 0:
            self.S_ex = np.zeros(n_freqs, dtype=np.complex64)
            self.S_ee = np.zeros(n_freqs, dtype=np.float32)
            self.S_xx = np.zeros(n_freqs, dtype=np.float32)
            # #7: Voice-band weighting (300Hz-4kHz emphasized)
            self.voice_weight = np.ones(n_freqs, dtype=np.float32)
            freq_per_bin = sample_rate / block_size
            for k in range(n_freqs):
                f = k * freq_per_bin
                if 300.0 <= f <= 4000.0:
                    self.voice_weight[k] = 3.0  # 3× weight for speech band
                elif f < 100.0 or f > 6000.0:
                    self.voice_weight[k] = 0.3  # De-weight extremes
        else:
            self.S_ex = None
            self.S_ee = None
            self.S_xx = None
            self.voice_weight = None

    def reset(self):
        self.confidence = 0.0
        self.frame_count = 0
        self.far_abs_buffer.fill(0)
        self.buf_idx = 0
        self.hangover_count = 0
        if self.S_ex is not None:
            self.S_ex.fill(0)
            self.S_ee.fill(0)
            self.S_xx.fill(0)

    def _update_confidence(self, detected: bool):
        """Update confidence with attack/release + hangover."""
        if detected:
            self.hangover_count = self.hangover_max
            self.confidence = min(self.confidence + self.attack, 1.0)
        elif self.hangover_count > 0:
            self.hangover_count -= 1
            self.confidence = max(self.confidence - self.release * 0.5, 0.0)
        else:
            self.confidence = max(self.confidence - self.release, 0.0)

    def _detect_geigel(self, near_end: np.ndarray, far_end: np.ndarray):
        """Geigel DTD: |mic| > threshold × max(|ref|) over window."""
        # Update far-end max circular buffer
        self.far_abs_buffer[self.buf_idx] = np.max(np.abs(far_end))
        self.buf_idx = (self.buf_idx + 1) % len(self.far_abs_buffer)
        far_max = np.max(self.far_abs_buffer)

        # Geigel test
        near_max = np.max(np.abs(near_end))
        detected = (far_max > 1e-6) and (near_max > self.geigel_threshold * far_max)

        self._update_confidence(detected)

    def _detect_divergence(self, near_end: np.ndarray, output: np.ndarray):
        """Output-vs-input divergence detection (output > input).

        Uses both energy-based and peak-based detection. Peak-based catches
        localized spikes that energy-based misses (e.g., transition transients).
        """
        output_energy = np.mean(output ** 2)
        near_energy = np.mean(near_end ** 2)
        output_peak = np.max(np.abs(output))
        near_peak = np.max(np.abs(near_end))

        if near_energy < 1e-10 and near_peak < 1e-6:
            # Silence → release
            self.confidence = max(self.confidence - self.release, 0.0)
            return

        # Check both energy and peak divergence
        energy_ratio = output_energy / (near_energy + 1e-10) if near_energy > 1e-10 else 0.0
        peak_ratio = output_peak / (near_peak + 1e-10) if near_peak > 1e-6 else 0.0
        ratio = max(energy_ratio, peak_ratio)

        mild_threshold = 1.2  # ratio < 1.2 is normal (unconverged, not diverging)
        if ratio > self.divergence_factor:
            # Severe divergence
            self.confidence = min(self.confidence + self.attack, 1.0)
        elif ratio > mild_threshold:
            # Mild divergence — proportional attack
            self.confidence = min(
                self.confidence + self.attack * (ratio - mild_threshold), 1.0)
        else:
            # Normal — faster release when ratio is well below 1.0
            release_scale = max(1.0 - ratio, 0.2)  # 0.2x ~ 1.0x
            self.confidence = max(
                self.confidence - self.release * (1.0 + 4.0 * release_scale), 0.0)

    def _detect_coherence(self, error_spec: np.ndarray, far_spec: np.ndarray):
        """Coherence-based double-talk detection.

        Uses smoothed magnitude-squared coherence between error and far-end.
        Low coherence + high error energy → near-end speech present → DT.
        High coherence → residual echo (unconverged) → keep updating.
        """
        alpha = self.coh_alpha

        # Update smoothed PSDs
        cross = error_spec * np.conj(far_spec)
        self.S_ex = alpha * self.S_ex + (1 - alpha) * cross
        self.S_ee = alpha * self.S_ee + (1 - alpha) * np.abs(error_spec) ** 2
        self.S_xx = alpha * self.S_xx + (1 - alpha) * np.abs(far_spec) ** 2

        # #7: Voice-band weighted coherence (ratio-of-sums)
        w = self.voice_weight
        num = np.sum(w * np.abs(self.S_ex) ** 2)
        den = np.sum(w * self.S_ee * self.S_xx)
        coherence = num / (den + 1e-10)

        # Energy check: only declare DT if error has meaningful energy
        sum_ee = np.sum(self.S_ee)
        sum_xx = np.sum(self.S_xx)
        # #8: Absolute energy floor prevents false triggers on quiet far-end
        has_energy = (sum_ee > self.coh_energy_floor * sum_xx and
                      sum_xx > 1e-10 and
                      sum_ee > self.coh_abs_floor)

        if coherence > self.coh_high:
            # Correlated → residual echo, not DT → release
            self._update_confidence(False)
        elif coherence < self.coh_low and has_energy:
            # Uncorrelated + energy → near-end speech → DT
            self._update_confidence(True)
        else:
            # Ambiguous → slow release
            self.confidence = max(self.confidence - self.release * 0.5, 0.0)

    def detect_block(self, near_end: np.ndarray, far_end: np.ndarray,
                     output: np.ndarray = None,
                     error_spec: np.ndarray = None,
                     far_spec: np.ndarray = None) -> float:
        """Update DTD state and return confidence [0.0, 1.0].

        For geigel mode: uses near_end and far_end.
        For divergence mode: uses near_end and output.
        For coherence mode: uses error_spec and far_spec.
        """
        self.frame_count += 1
        # Warmup: all detectors share the same warmup period.
        # Coherence also needs warmup because unconverged filter → error ≈ echo
        # → coherence estimate is unreliable (false DT triggers).
        if self.frame_count < self.warmup_frames:
            return 0.0

        if self.mode == 'geigel':
            self._detect_geigel(near_end, far_end)
        elif self.mode == 'coherence':
            if error_spec is not None and far_spec is not None:
                self._detect_coherence(error_spec, far_spec)
        else:
            self._detect_divergence(near_end, output)

        return self.confidence


@dataclass
class RenderActivityState:
    """One-frame summary of far-end activity for downstream consumers.

    far_pwr        mean(far²) + 1e-10 (always positive; safe denominator)
    is_active      latched: True once far has been audible (>1e-6) since last silence
    is_stationary  CV² of far envelope < 0.02 → stationary (white-noise-like)
    warmup_active  raw mean(far²) > 1e-6 (used to gate warmup-frame consumption)
    """
    far_pwr: float
    is_active: bool
    is_stationary: bool
    warmup_active: bool


class RenderActivityDetector:
    """Far-end activity + stationarity detector.

    Tracks far-end power envelope EMA and its variance to detect
    stationary far-end (white noise / fans), which downstream blocks
    use to gate EPC false-positives and to switch DT-detection branches.

    State (private):
        _env_mean       far-power EMA
        _env_var        far-power variance EMA
        _active_prev    True after first audible far-end frame; resets only when far drops to silence
        _is_stationary  CV² < threshold this frame
    """
    ALPHA_CV = 0.99           # TC ≈ 1 s envelope smoothing
    STATIONARY_CV2 = 0.02     # CV² gate for stationary far-end

    def __init__(self):
        self._env_mean = 1e-10
        self._env_var = 0.0
        self._active_prev = False
        self._is_stationary = False

    def reset(self) -> None:
        self._env_mean = 1e-10
        self._env_var = 0.0
        self._active_prev = False
        self._is_stationary = False

    def update(self, far_end: np.ndarray) -> RenderActivityState:
        far_pwr_raw = float(np.mean(far_end ** 2))
        far_pwr = far_pwr_raw + 1e-10
        warmup_active = far_pwr_raw > 1e-6
        if far_pwr > 1e-6:
            if not self._active_prev:
                self._env_mean = far_pwr
                self._env_var = 0.0
                self._active_prev = True
            else:
                old_mean = self._env_mean
                self._env_mean = (self.ALPHA_CV * self._env_mean
                                  + (1 - self.ALPHA_CV) * far_pwr)
                self._env_var = (self.ALPHA_CV * self._env_var
                                 + (1 - self.ALPHA_CV) * (far_pwr - old_mean) ** 2)
            far_cv2 = self._env_var / (self._env_mean ** 2 + 1e-10)
            self._is_stationary = far_cv2 < self.STATIONARY_CV2
        else:
            self._active_prev = False
            self._is_stationary = False
        return RenderActivityState(
            far_pwr=far_pwr,
            is_active=self._active_prev,
            is_stationary=self._is_stationary,
            warmup_active=warmup_active,
        )

    @property
    def is_active(self) -> bool: return self._active_prev
    @property
    def is_stationary(self) -> bool: return self._is_stationary


@dataclass
class FilterConvergenceState:
    """One-frame snapshot of filter convergence health."""
    converged: bool
    once_converged: bool
    just_converged: bool   # True for the single frame the transition fires
    divergence: float      # [0, 1] EMA: rate of post-convergence inst-ERLE < -2 dB


class FilterConvergenceAnalyzer:
    """Owns filter-convergence state machine + divergence-indicator EMA.

    Convergence rule: 10 consecutive far-active frames with inst-ERLE > 5 dB
    after warmup is exhausted.
    Divergence rule: post-convergence EMA of (inst-ERLE_linear < 0.63 ↔ ERLE < -2 dB).

    External signals (EPC, delay shift, echo-path change) call mark_diverged()
    to invalidate convergence and reset the counter — the analyzer never
    self-resets on those events.
    """
    CONV_ERLE_DB = 5.0
    CONV_FRAMES = 10
    DIV_ERLE_LIN = 0.63
    DIV_ALPHA = 0.9
    DIV_DECAY = 0.95

    def __init__(self):
        self._converged = False
        self._once_converged = False
        self._conv_counter = 0
        self._divergence = 0.0

    def reset(self) -> None:
        self._converged = False
        self._once_converged = False
        self._conv_counter = 0
        self._divergence = 0.0

    def mark_diverged(self) -> None:
        """EPC / delay shift / echo-path change: drop convergence, restart counter."""
        self._converged = False
        self._conv_counter = 0

    def update_divergence(self, near_power: float, raw_error_power: float) -> None:
        """Mid-frame divergence indicator EMA (only meaningful post-convergence)."""
        if self._converged and near_power > 1e-8:
            inst_erle_lin = near_power / (raw_error_power + 1e-10)
            is_diverged = float(inst_erle_lin < self.DIV_ERLE_LIN)
            self._divergence = (self.DIV_ALPHA * self._divergence
                                + (1 - self.DIV_ALPHA) * is_diverged)
        else:
            self._divergence *= self.DIV_DECAY

    def update_convergence(self, *, near_power: float, raw_error_power: float,
                           far_active: bool, warmup_done: bool) -> bool:
        """End-of-frame convergence detection. Returns True on the transition frame."""
        if self._converged or near_power <= 1e-8 or not warmup_done or not far_active:
            return False
        inst_erle_db = 10.0 * np.log10(near_power / (raw_error_power + 1e-10))
        if inst_erle_db > self.CONV_ERLE_DB:
            self._conv_counter += 1
        else:
            self._conv_counter = 0
        if self._conv_counter >= self.CONV_FRAMES:
            self._converged = True
            self._once_converged = True
            return True
        return False

    @property
    def converged(self) -> bool: return self._converged
    @property
    def once_converged(self) -> bool: return self._once_converged
    @property
    def divergence(self) -> float: return self._divergence


class DoubleTalkAnalyzer:
    """Aggregates the three pre-filter / cross-filter DT signals.

    Owns:
        _dt_from_energy   : pre-filter mic-energy excess over (far × ERL_ceiling × 2)
                            EMA-smoothed (fast-rise 0.3/0.7, slow-decay 0.9/0.1).
        _dt_from_shadow   : (shadow_advantage − offset) / scale, smoothed 70/30.
        _shadow_advantage : main_err / shadow_err (raw, no EMA).

    Coherence-based DT lives in DtdEstimator (self.dtd_coherence on AEC) and
    is read by name from there; this analyzer does not own it but exposes a
    combined() helper for AecState assembly.
    """

    SHADOW_FRAME_GATE = 50  # match shadow filter warmup
    ERL_CEILING_FLOOR = 0.01
    SAFETY_MARGIN = 2.0
    DTE_RISE_OLD, DTE_RISE_NEW = 0.3, 0.7
    DTE_DECAY_OLD, DTE_DECAY_NEW = 0.9, 0.1
    DTS_OLD, DTS_NEW = 0.7, 0.3
    DTS_INACTIVE_DECAY = 0.95

    def __init__(self, config: 'AecConfig'):
        self.config = config
        self._dt_from_energy = 0.0
        self._dt_from_shadow = 0.0
        self._shadow_advantage = 1.0

    def reset(self) -> None:
        self._dt_from_energy = 0.0
        self._dt_from_shadow = 0.0
        self._shadow_advantage = 1.0

    def update_shadow_dt(self, *, shadow_frame_count: int, far_excited: bool,
                         main_err_smooth: float, shadow_err_smooth: float) -> None:
        """Shadow-advantage based DT signal. Runs once per frame in the shadow block."""
        if shadow_frame_count >= self.SHADOW_FRAME_GATE and far_excited:
            self._shadow_advantage = main_err_smooth / (shadow_err_smooth + 1e-10)
            raw = float(np.clip(
                (self._shadow_advantage - self.config.shadow_dtd_offset)
                / self.config.shadow_dtd_advantage_scale,
                0.0, 1.0))
            self._dt_from_shadow = self.DTS_OLD * self._dt_from_shadow + self.DTS_NEW * raw
        else:
            self._dt_from_shadow *= self.DTS_INACTIVE_DECAY

    def update_energy_dt(self, *, far_active: bool, far_pwr: float,
                         mic_pwr: float, erl_estimate: float) -> None:
        """Pre-filter mic-energy DT signal. Runs once per frame in the RES block."""
        if far_active and far_pwr > 1e-4:
            erl_ceiling = 1.0 / max(erl_estimate, self.ERL_CEILING_FLOOR)
            max_echo_expected = far_pwr * erl_ceiling * self.SAFETY_MARGIN
            inst = max(0.0, (mic_pwr - max_echo_expected) / mic_pwr)
        else:
            inst = 0.0
        if inst > self._dt_from_energy:
            self._dt_from_energy = (self.DTE_RISE_OLD * self._dt_from_energy
                                    + self.DTE_RISE_NEW * inst)
        else:
            self._dt_from_energy = (self.DTE_DECAY_OLD * self._dt_from_energy
                                    + self.DTE_DECAY_NEW * inst)

    @property
    def dt_from_energy(self) -> float: return self._dt_from_energy
    @property
    def dt_from_shadow(self) -> float: return self._dt_from_shadow
    @property
    def shadow_advantage(self) -> float: return self._shadow_advantage


class ResidualEchoEstimator:
    """Two-path residual echo PSD attribution (stage 1 + stage 2 of ResFilter).

    Replaces ResFilter's inline ~100 lines of residual_echo_psd computation with
    an explicit class that owns the render-based switching state. Provides two
    modes:

      'legacy': bit-exact reproduction of v2.8.1 inline logic — ERLE-blended
                linear estimate with optional render-based blend driven by
                an ENR-adaptive switching threshold + hysteresis + min hold.
                Used as default for parity validation.

      'split' : explicit AEC3-style branch on `aec_state.usable_linear_estimate`.
                Linear path: `S2_linear / ERLE`. Nonlinear path: `X2 * echo_path_gain`.
                Used by Phase B2 ablation (R1 variant). Disabled by default.

    Owns: _using_render_based, _render_based_hold (legacy state machine).
    Caller (ResFilter) keeps echo_psd / error_psd / near_psd / coh2 / far_activity
    on itself; we pass them in by reference per call.
    """
    LEGACY = 'legacy'
    SPLIT = 'split'

    def __init__(self, n_freqs: int, mode: str = 'legacy'):
        self.n_freqs = n_freqs
        self.mode = mode
        self._using_render_based = False
        self._render_based_hold = 0

    def reset(self) -> None:
        self._using_render_based = False
        self._render_based_hold = 0

    @property
    def using_render_based(self) -> bool: return self._using_render_based

    def attribute(self, *, aec_state=None, **kw) -> np.ndarray:
        """Mode-dispatch entry. Caller passes the union of legacy+split kwargs;
        method picks the relevant subset."""
        if self.mode == self.SPLIT and aec_state is not None:
            return self.attribute_split(
                echo_psd=kw['echo_psd'], error_psd=kw['error_psd'],
                far_spec=kw['far_spec'], far_power=kw['far_power'],
                erle_factor=kw['erle_factor'], erl_estimate=kw['erl_estimate'],
                filter_erle=kw['filter_erle'], fb_erle=kw['fb_erle'],
                aec_state=aec_state,
            )
        return self.attribute_legacy(**kw)

    def attribute_legacy(self, *, echo_psd: np.ndarray, error_psd: np.ndarray,
                         coh2: np.ndarray, far_spec, far_power: float,
                         erle_factor: float, dt_for_fs: float, far_activity: float,
                         epc_active: bool, saturation_level: float,
                         filter_converged: bool, erl_estimate: float,
                         filter_erle, fb_erle) -> np.ndarray:
        """Stage 1 (ERLE-blended linear) + Stage 2 (render-based switch).

        Bit-exact port of ResFilter.process() residual-echo block from v2.8.1
        (lines ~1456-1555). Mutates self._using_render_based / _render_based_hold.
        """
        # Multi-ERLE residual estimation (Phase 2)
        confidence = compute_erle_confidence(filter_erle.erle, fb_erle.fb_erle)
        erle_corrected = (confidence * filter_erle.erle
                          + (1.0 - confidence) * 1.0)
        erle_corrected = np.maximum(erle_corrected, 0.5)

        erle_est = echo_psd / erle_corrected
        direct_est = echo_psd

        if far_power > 1e-4:
            dt_weight = 1.0 - dt_for_fs
            nonlinear_floor = error_psd * coh2 * far_activity * dt_weight
            direct_est = np.maximum(direct_est, nonlinear_floor)
            erle_est = np.maximum(erle_est, nonlinear_floor)

        residual_echo_psd = (1.0 - erle_factor) * direct_est + erle_factor * erle_est

        if far_power > 1e-4:
            error_power_mean = float(np.mean(error_psd)) + 1e-10
            enr = far_power / error_power_mean
            switching_threshold = 0.5 * np.clip(enr / (enr + 1.0), 0.3, 0.7)
            hysteresis = 0.05
            if self._using_render_based:
                effective_threshold = switching_threshold + hysteresis
            else:
                effective_threshold = switching_threshold
            force_render = (
                epc_active
                or saturation_level > 0.5
                or not filter_converged
            )
            want_render = (erle_factor < effective_threshold) or force_render
            if want_render and not self._using_render_based:
                self._render_based_hold = 5
            if self._using_render_based:
                self._render_based_hold = max(self._render_based_hold - 1, 0)
            can_exit = (not want_render and self._render_based_hold == 0)
            self._using_render_based = want_render or (self._using_render_based and not can_exit)

            if self._using_render_based:
                far_psd = (np.abs(far_spec) ** 2 if far_spec is not None
                           else np.zeros(self.n_freqs, dtype=np.float32))
                # v3.8 ABL-1 (ablate v3.3 error_based_floor): error_psd contains
                # NE during DT, so using it as residual_echo floor structurally
                # over-suppresses near-end (same lesson as v3.7.1 PR-B). Use only
                # render_based_echo = far × ERL — AEC3-aligned.
                render_based_echo = far_psd * erl_estimate
                blend = 1.0 - erle_factor / effective_threshold
                blend = np.clip(blend, 0.0, 1.0)
                residual_echo_psd = ((1.0 - blend) * residual_echo_psd
                                     + blend * render_based_echo)

        return residual_echo_psd

    def attribute_split(self, *, echo_psd: np.ndarray, error_psd: np.ndarray,
                        far_spec, far_power: float,
                        erle_factor: float, erl_estimate: float,
                        filter_erle, fb_erle, aec_state) -> np.ndarray:
        """AEC3-style two-path R2: linear if `aec_state.usable_linear_estimate`,
        else render-based. Used by Phase B2 R1 ablation."""
        # Always update _using_render_based to reflect current decision
        self._using_render_based = not aec_state.usable_linear_estimate
        if aec_state.usable_linear_estimate:
            confidence = compute_erle_confidence(filter_erle.erle, fb_erle.fb_erle)
            erle_corrected = (confidence * filter_erle.erle + (1.0 - confidence) * 1.0)
            erle_corrected = np.maximum(erle_corrected, 0.5)
            return echo_psd / erle_corrected
        # Render-based path: X2 * echo_path_gain
        far_psd = np.abs(far_spec) ** 2 if far_spec is not None else np.zeros(self.n_freqs)
        return far_psd * erl_estimate


class AecState:
    """WebRTC AEC3-style read-only aggregator over the per-frame detector outputs.

    Holds references to the 5 detectors (RenderActivityDetector, DoubleTalkAnalyzer,
    FilterConvergenceAnalyzer, EchoPathChangeDetector, ShadowCopyController) and
    DtdEstimator-coherence; exposes derived flags via @property so consumers don't
    rebuild aggregation logic in multiple places.

    Phase B consumes this as the gate for two-path residual echo attribution:
    `usable_linear_estimate` decides linear vs nonlinear R2 model.

    All properties are read-only and reflect the current frame's detector state
    (no explicit per-frame update needed — properties delegate live).
    """

    def __init__(self, *, render_activity, convergence, dt_analyzer, epc_det,
                 shadow_copy_ctrl, dtd_coherence_getter):
        self._render = render_activity
        self._conv = convergence
        self._dt = dt_analyzer
        self._epc = epc_det
        self._shadow = shadow_copy_ctrl
        self._dtd_coh = dtd_coherence_getter

    # ── Render activity ──────────────────────────────────────────────────────
    @property
    def render_active(self) -> bool: return self._render.is_active
    @property
    def render_stationary(self) -> bool: return self._render.is_stationary

    # ── Filter convergence ───────────────────────────────────────────────────
    @property
    def filter_converged(self) -> bool: return self._conv.converged
    @property
    def filter_once_converged(self) -> bool: return self._conv.once_converged
    @property
    def divergence(self) -> float: return self._conv.divergence

    # ── Double-talk signals ──────────────────────────────────────────────────
    @property
    def dt_from_energy(self) -> float: return self._dt.dt_from_energy
    @property
    def dt_from_shadow(self) -> float: return self._dt.dt_from_shadow
    @property
    def dt_from_coherence(self) -> float: return self._dtd_coh()
    @property
    def dt_combined(self) -> float:
        """Max of the three DT signals — same aggregation as inline code today."""
        return max(self._dt.dt_from_energy, self._dt.dt_from_shadow, self._dtd_coh())

    # ── Echo path change ─────────────────────────────────────────────────────
    @property
    def epc_active(self) -> bool: return self._epc.active
    @property
    def epc_hangover_count(self) -> int: return self._epc.hangover_count

    # ── Shadow filter management ─────────────────────────────────────────────
    @property
    def main_paused(self) -> bool: return self._shadow.main_paused
    @property
    def shadow_advantage(self) -> float: return self._dt.shadow_advantage

    # ── Aggregated decisions (Phase B consumers) ─────────────────────────────
    @property
    def usable_linear_estimate(self) -> bool:
        """Can the main filter's echo estimate be trusted as residual-echo source?

        AEC3-style: requires once-converged + currently-converged + not in EPC recovery.
        When False, Phase B's residual echo estimator should fall back to
        render-window-based attribution.
        """
        return (self._conv.once_converged
                and self._conv.converged
                and not self._epc.active)


@dataclass
class EpcEvent:
    """One-frame EPC trigger result.

    fired:  True if this frame fired one of the three triggers (delay/epv/shadow_rise)
    source: which trigger fired ('delay' | 'epv' | 'shadow_rise')
    """
    fired: bool = False
    source: str = ''


class EchoPathChangeDetector:
    """Unified state for the three echo-path-change trigger sources.

    Triggers (caller invokes individually so call-site ordering matches v2.8.1):
        force_delay()        → delay-realignment trigger (caller-driven)
        update_epv(...)      → fast/slow far-power EMA divergence (±6 dB)
        update_shadow_rise(...)  → both filters' errors rising in tandem (post-converged)

    Caller applies side effects (Q-boost, P-override, ERL cap, render-forced,
    mark_diverged, dtd_coherence dampening) by inspecting EpcEvent.source.

    Hangover countdown is unified into a single counter; tick_hangover() is
    called once per frame inside the same `(shadow_filter and filter_converged)`
    gate the original code used, **and only when shadow_rise did not fire this
    frame** — preserving the original `if/elif/else` semantics bit-exactly.

    State (private):
        _active            currently in EPC (firing or hangover)
        _hangover          remaining hangover frames
        _epv_gain_fast     fast far-power EMA (TC ≈ 50 frames)
        _epv_gain_slow     slow far-power EMA (TC ≈ 1000 frames)
        _prev_total_err    last frame's main_err+shadow_err sum (for shadow_rise)
    """
    EPV_FAST_TC = 0.98
    EPV_SLOW_TC = 0.999
    EPV_LOW = 0.25
    EPV_HIGH = 4.0

    def __init__(self, config: 'AecConfig'):
        self.config = config
        self._active = False
        self._hangover = 0
        self._epv_gain_fast = 0.0
        self._epv_gain_slow = 0.0
        self._prev_total_err = 0.0

    def reset(self) -> None:
        self._active = False
        self._hangover = 0
        self._epv_gain_fast = 0.0
        self._epv_gain_slow = 0.0
        self._prev_total_err = 0.0

    def force_delay(self) -> EpcEvent:
        """Trigger 1: delay-shift re-alignment (caller already validated consistency)."""
        self._active = True
        self._hangover = self.config.epc_hangover
        return EpcEvent(fired=True, source='delay')

    def update_epv(self, *, far_pwr_global: float, filter_converged: bool,
                   main_paused: bool) -> EpcEvent:
        """Trigger 2: EPV gain-ratio. Updates fast/slow EMAs, fires on outliers."""
        if far_pwr_global <= 1e-6:
            return EpcEvent()
        if self._epv_gain_fast < 1e-12:
            self._epv_gain_fast = self._epv_gain_slow = far_pwr_global
        else:
            self._epv_gain_fast = (self.EPV_FAST_TC * self._epv_gain_fast
                                   + (1 - self.EPV_FAST_TC) * far_pwr_global)
            self._epv_gain_slow = (self.EPV_SLOW_TC * self._epv_gain_slow
                                   + (1 - self.EPV_SLOW_TC) * far_pwr_global)
        if (filter_converged and not self._active and not main_paused
                and self._epv_gain_slow > 1e-10):
            ratio = self._epv_gain_fast / (self._epv_gain_slow + 1e-10)
            if ratio < self.EPV_LOW or ratio > self.EPV_HIGH:
                self._active = True
                self._hangover = self.config.epc_hangover
                return EpcEvent(fired=True, source='epv')
        return EpcEvent()

    def update_shadow_rise(self, *, main_err_smooth: float, shadow_err_smooth: float,
                           is_stationary: bool) -> EpcEvent:
        """Trigger 3: shadow-based error rise. Always updates _prev_total_err."""
        total_err = main_err_smooth + shadow_err_smooth
        if total_err > 1e-10:
            delta_ratio = abs(main_err_smooth - shadow_err_smooth) / total_err
        else:
            delta_ratio = 0.0
        errors_rising = (total_err > self._prev_total_err * self.config.epc_total_rise
                         and self._prev_total_err > 1e-10)
        is_echo_change = errors_rising and delta_ratio < self.config.epc_delta_threshold
        # White-noise guard: stationary far + error rise = DT, not echo path change
        if is_echo_change and is_stationary:
            is_echo_change = False
        self._prev_total_err = total_err
        if is_echo_change:
            self._active = True
            self._hangover = self.config.epc_hangover
            return EpcEvent(fired=True, source='shadow_rise')
        return EpcEvent()

    def tick_hangover(self) -> None:
        """Count down hangover and clear active when expired.

        Caller MUST gate this on the original (shadow_filter and filter_converged)
        guard, AND only call it when shadow_rise did not fire this frame.
        """
        if self._hangover > 0:
            self._hangover -= 1
            self._active = True
        else:
            self._active = False

    @property
    def active(self) -> bool: return self._active
    @property
    def hangover_count(self) -> int: return self._hangover
    @property
    def epv_gain_fast(self) -> float: return self._epv_gain_fast
    @property
    def epv_gain_slow(self) -> float: return self._epv_gain_slow


@dataclass
class ShadowCopyDecision:
    """One-frame decision emitted by ShadowCopyController.

    pause_main:    main filter weight update is gated off this frame
    boost_q:       this frame triggered the pause; caller should boost Q on main filter
    reverse_copy:  this frame, shadow should be re-synced from main (main-winning case)
    """
    pause_main: bool = False
    boost_q: bool = False
    reverse_copy: bool = False


class ShadowCopyController:
    """Owns shadow-copy gate state machine.

    Inputs are read-only per-frame measurements; outputs are explicit decisions.
    Caller (AEC.process) applies the decisions: gate main_mu, apply filter Q-boost,
    perform shadow.copy_weights_from(main). The controller never mutates filter state.

    State (private):
        _copy_err_baseline : EMA of best(main, shadow) error during stable FS
        _copy_counter      : consecutive frames shadow < main * threshold
        _streak            : same as above (kept for parity with original two-counter logic)
        _main_paused       : main filter weight update currently frozen
        _pause_resume      : countdown to un-pause
    """

    BASELINE_INIT = 1e-6
    HYS_STREAK_MIN = 10  # additional streak gate beyond shadow_copy_hysteresis
    AEC3_STREAK_FRAMES = 5  # gate_mode='streak_only' uses pure 5-block AEC3 rule

    # Gate-mode choices for Phase C1 ablation. S0=energy is the v2.8.1 baseline.
    GATE_ENERGY = 'energy'                # S0: dt_from_energy < 0.3
    GATE_COHERENCE = 'coherence'          # S1: dt_from_coherence < 0.4
    GATE_COH_DELAY = 'coherence_delay'    # S2: S1 AND delay_reliable
    GATE_STREAK = 'streak_only'           # S3: AEC3 5-block, no DT gate

    def __init__(self, config: 'AecConfig', gate_mode: str = 'energy'):
        self.config = config
        self.gate_mode = gate_mode
        self._copy_err_baseline = self.BASELINE_INIT
        self._copy_counter = 0
        self._streak = 0
        self._main_paused = False
        self._pause_resume = 0

    def reset(self) -> None:
        self._copy_err_baseline = self.BASELINE_INIT
        self._copy_counter = 0
        self._streak = 0
        self._main_paused = False
        self._pause_resume = 0

    @property
    def main_paused(self) -> bool:
        return self._main_paused

    @property
    def copy_err_baseline(self) -> float:
        return self._copy_err_baseline

    @property
    def copy_counter(self) -> int:
        return self._copy_counter

    def _dt_safe(self, dt_from_energy: float, dt_from_coherence: float,
                 delay_reliable: bool) -> bool:
        """Resolve the per-mode DT-safety gate. Returns True when copy is allowed
        from a DT-perspective (NOT from far-active / saturation / EPC perspective)."""
        m = self.gate_mode
        if m == self.GATE_ENERGY:
            return dt_from_energy < 0.3
        if m == self.GATE_COHERENCE:
            return dt_from_coherence < 0.4
        if m == self.GATE_COH_DELAY:
            return dt_from_coherence < 0.4 and delay_reliable
        if m == self.GATE_STREAK:
            return True   # AEC3 mode: skip DT gate, rely on streak alone
        return dt_from_energy < 0.3  # fallback: legacy

    def update(self, *, shadow_frame_count: int, far_pwr: float,
               main_err_smooth: float, shadow_err_smooth: float,
               epc_active: bool, saturation_level: float,
               dt_from_energy: float,
               dt_from_coherence: float = 0.0,
               delay_reliable: bool = False) -> ShadowCopyDecision:
        decision = ShadowCopyDecision()
        if shadow_frame_count < 50:
            return decision

        threshold = self.config.shadow_copy_threshold
        far_active = far_pwr > 1e-4

        err_sum = main_err_smooth + shadow_err_smooth + 1e-10
        err_balance = abs(main_err_smooth - shadow_err_smooth) / err_sum
        is_stable_fs = far_active and err_balance < 0.3 and not epc_active
        if is_stable_fs:
            best_err = min(main_err_smooth, shadow_err_smooth)
            self._copy_err_baseline = (0.995 * self._copy_err_baseline
                                        + 0.005 * best_err)

        error_is_normal = main_err_smooth < self._copy_err_baseline * 4.0 + 1e-10
        not_saturating = saturation_level < 0.3
        # DT guard: shadow may chase near-end speech during DT, making shadow_err
        # artificially low. Gate selectable for Phase C1 ablation.
        dt_safe = self._dt_safe(dt_from_energy, dt_from_coherence, delay_reliable)
        copy_allowed = (far_active and error_is_normal
                        and not epc_active and not_saturating and dt_safe)

        # S3 streak-only mode: AEC3 5-block consecutive shadow-better rule, no
        # hysteresis pair. Skip the legacy two-counter logic and go directly
        # to a simple streak.
        if self.gate_mode == self.GATE_STREAK:
            if copy_allowed and shadow_err_smooth < main_err_smooth * threshold:
                self._streak += 1
                if self._streak >= self.AEC3_STREAK_FRAMES:
                    self._streak = 0
                    self._main_paused = True
                    self._pause_resume = self.config.epc_hangover
                    decision.boost_q = True
            else:
                self._streak = 0
            if self._main_paused:
                if self._pause_resume > 0:
                    self._pause_resume -= 1
                else:
                    self._main_paused = False
            if (copy_allowed
                    and main_err_smooth < shadow_err_smooth * threshold
                    and error_is_normal):
                decision.reverse_copy = True
            decision.pause_main = self._main_paused
            return decision

        # Legacy / coherence gates: original two-counter + streak logic.
        if copy_allowed:
            if shadow_err_smooth < main_err_smooth * threshold:
                self._copy_counter += 1
                self._streak += 1
            else:
                self._copy_counter = 0
                self._streak = 0

            if (self._copy_counter >= self.config.shadow_copy_hysteresis
                    and self._streak >= self.HYS_STREAK_MIN):
                self._copy_counter = 0
                self._streak = 0
                self._main_paused = True
                self._pause_resume = self.config.epc_hangover
                decision.boost_q = True

            if self._main_paused:
                if self._pause_resume > 0:
                    self._pause_resume -= 1
                else:
                    self._main_paused = False

            if (main_err_smooth < shadow_err_smooth * threshold
                    and error_is_normal):
                decision.reverse_copy = True
        else:
            self._copy_counter = 0
            self._streak = 0
            self._main_paused = False

        decision.pause_main = self._main_paused
        return decision


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

    def _maybe_mark_diverged(self, source: str) -> None:
        if source not in self._epc_no_reset_sources:
            self._convergence.mark_diverged()
        # Round 3 trace: per-source counter (audio-passive)
        if not hasattr(self, '_round3_div_counts'):
            self._round3_div_counts = {'delay_first': 0, 'delay_shift': 0,
                                       'epv': 0, 'shadow_rise': 0}
        self._round3_div_counts[source] = self._round3_div_counts.get(source, 0) + 1
        self._round3_last_div_source = source

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
            if self.config.fixed_delay_samples >= 0:
                max_delay_samp = max(max_delay_samp, self.config.fixed_delay_samples + 256)
                self.delay_est = None
                self._current_delay = self.config.fixed_delay_samples
            else:
                self.delay_est = DelayEstimator(
                    sample_rate=self.config.sample_rate,
                    max_delay_ms=self.config.max_delay_ms,
                    init_seconds=self.config.delay_est_init_s,
                    period_seconds=self.config.delay_est_period_s,
                )
                self._current_delay = -1  # -1 = not yet estimated
            # Reference ring buffer for delay compensation
            self._ref_ring = np.zeros(max_delay_samp + 4096, dtype=np.float32)
            self._ref_ring_write = 0
            self._ref_ring_size = len(self._ref_ring)
            self._ref_ring_filled = 0  # Total samples written (for warmup)
            self._delay_active = True
        else:
            self.delay_est = None
            self._delay_active = False

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

        # RES (only for frequency-domain modes)
        if self.config.enable_res and self.config.mode in _FREQ_MODES:
            self.res = ResFilter(
                block_size=self.filter.fft_size,
                n_freqs=self.filter.n_freqs,
                g_min_db=self.config.res_g_min_db,
                over_sub=self.config.res_over_sub,
                alpha=self.config.res_alpha,
                enable_cng=self.config.enable_cng,
                max_drop_db_per_frame=self.config.res_max_drop_db_per_frame,
                max_rise_db_per_frame=self.config.res_max_rise_db_per_frame,
                enable_spectral_floor=self.config.res_spectral_floor,
                spectral_floor_db=self.config.res_spectral_floor_db,
                ne_protect_db=self.config.res_ne_protect_db,
                frame_size=self.config.frame_size,
                hop_size=self.config.hop_size,
                echo_method=self.config.res_echo_method,
                gain_type=self.config.res_gain_type,
                enable_reverb=self.config.res_enable_reverb,
                reverb_decay=self.config.res_reverb_decay,
                reverb_gain=self.config.res_reverb_gain,
                alpha_echo_psd=self.config.res_alpha_echo_psd,
                alpha_error_psd=self.config.res_alpha_error_psd,
                enr_scale=self.config.res_enr_scale,
                startup_dt_min_ne_scale=self.config.startup_dt_min_ne_scale,
                startup_dt_gain_floor=self.config.startup_dt_gain_floor,
                startup_dt_noise_floor_scale=self.config.startup_dt_noise_floor_scale,
                sample_rate=self.config.sample_rate,
                capture_stages=self.config.capture_stages,
            )
        else:
            self.res = None

        # Shadow filter (dual-filter, frequency-domain modes only)
        # Can be used alone (≈ WebRTC/SpeexDSP) or with DTD (dual protection)
        self.shadow_filter = None
        self.main_err_smooth = 0.0
        self.shadow_err_smooth = 0.0
        if (self.config.enable_shadow and
                self.config.mode in _FREQ_MODES
                and hasattr(self.filter, 'W')):
            shadow_mu = self.config.mu * self.config.shadow_mu_ratio
            self.shadow_filter = FilterClass(
                block_size=self.filter.block_size,
                n_partitions=self.filter.n_partitions,
                mu=shadow_mu,
                delta=self.config.delta,
                hop_size=self.filter.hop_size
            )
            self.shadow_filter.enable_td_constraint = self.config.enable_td_constraint
            # PBFDKF shadow: higher Q via ratio for faster tracking
            if isinstance(self.shadow_filter, PBFDKF):
                self.shadow_filter.Q_high = self.filter.Q_high * self.config.shadow_q_ratio
                self.shadow_filter.Q_low  = self.filter.Q_low  * self.config.shadow_q_ratio
                self.shadow_filter.Q      = self.shadow_filter.Q_high.copy()

        # Echo path change detector (owns active/hangover/EPV-EMAs/prev_total_err)
        self._epc_det = EchoPathChangeDetector(self.config)

        # #4: Confidence memory decay
        self.prev_dtd_conf = 0.0

        # Filter convergence + divergence-indicator (extracted to FilterConvergenceAnalyzer).
        # Backward-compat reads via @property below.
        self._convergence = FilterConvergenceAnalyzer()
        # EPC render-forced countdown (Change D)
        self._epc_render_forced_remaining = 0
        # Dynamic ERL estimate for render-based echo (B4)
        self._erl_estimate = 0.1  # initial -20dB, conservative
        # Double-talk analyzer (owns _dt_from_energy / _dt_from_shadow / _shadow_advantage)
        self._dt_analyzer = DoubleTalkAnalyzer(self.config)

        # Windowed decaying ERLE accumulator for erle_factor (TC ≈ 10s)
        self._erle_window_near = 1e-10
        self._erle_window_err = 1e-10
        self._erle_factor_prev = 0.0  # Previous frame's erle_factor for shadow DTD weight

        # Smoothed inst ERLE for dt_indicator correction (~3 frame / 30ms)
        self._inst_erle_smooth = 1.0

        # Per-bin mu_scale (updated from RES echo_psd/error_psd each frame)
        self._per_bin_mu_scale = None  # None = use scalar fallback

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

        # High-pass filter (DC blocker + low-freq removal)
        if self.config.enable_highpass:
            self._hp_mic = HighPassFilter(self.config.highpass_cutoff_hz, self.config.sample_rate)
            self._hp_ref = HighPassFilter(self.config.highpass_cutoff_hz, self.config.sample_rate)
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
        self._shadow_copy_ctrl = ShadowCopyController(self.config)
        self._last_raw_output: Optional[np.ndarray] = None   # raw filter output before RES (diagnostic)
        # EchoPathVariability EMAs moved into EchoPathChangeDetector (self._epc_det)
        # AecState aggregator: WebRTC-style read-only seam over the 5 detectors.
        # Phase B consumes this to decide linear vs nonlinear residual-echo path.
        self._aec_state = AecState(
            render_activity=self._render_activity,
            convergence=self._convergence,
            dt_analyzer=self._dt_analyzer,
            epc_det=self._epc_det,
            shadow_copy_ctrl=self._shadow_copy_ctrl,
            dtd_coherence_getter=lambda: (
                self.dtd_coherence.confidence if self.dtd_coherence else 0.0),
        )
        self._far_power_ema = 0.0           # TC≈50ms for GetStats()
        self._mic_power_ema = 0.0
        self._frame_count = 0               # frames since reset()

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
        self._convergence.reset()
        self._erle_window_near = 1e-10
        self._erle_window_err = 1e-10
        self._erle_factor_prev = 0.0
        self._inst_erle_smooth = 1.0
        self._simple_mu_ratio = 1.0
        self._simple_mu_holdoff = 0
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
        self._shadow_copy_ctrl.reset()
        # _epc_det reset above already cleared its EPV EMAs and prev_total_err
        self._far_power_ema = 0.0
        self._mic_power_ema = 0.0
        self._frame_count = 0
        if self.dtd_divergence:
            self.dtd_divergence.reset()
        if self.dtd_coherence:
            self.dtd_coherence.reset()
        if self.res:
            self.res.reset()
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
        # _conv_counter is owned by self._convergence (reset above)
        if self._hp_mic is not None:
            self._hp_mic.reset()
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

    @property
    def hop_size(self) -> int:
        return self._hop_size

    def _compute_mu_scale(self) -> float:
        """Convert combined DTD confidence to mu_scale [mu_min_ratio, 1.0].

        #3: Coherence is primary; divergence is fallback only when coherence inactive.
        #4: Confidence has memory decay to avoid sudden drops.
        EPC: mu_scale floor during echo path change.
        """
        conf_div = self.dtd_divergence.confidence if self.dtd_divergence else 0.0
        conf_coh = self.dtd_coherence.confidence if self.dtd_coherence else 0.0

        # #3: Coherence primary, divergence fallback
        if conf_coh > 0.1:
            raw_conf = conf_coh
        else:
            raw_conf = max(conf_div, conf_coh)

        # #4: Confidence memory decay (avoid sudden drops)
        conf = max(raw_conf, self.prev_dtd_conf * 0.9)
        self.prev_dtd_conf = conf

        if conf == 0.0:
            return 1.0
        min_r = self.config.dtd_mu_min_ratio
        # Before convergence, allow higher mu_min so filter can still learn during DT
        if not self._filter_converged:
            min_r = max(min_r, 0.3)
        mu_scale = 1.0 - conf * (1.0 - min_r)

        # Echo path change: keep mu high so filter can adapt to new path
        if self.epc_active:
            mu_scale = max(mu_scale, self.config.epc_mu_floor)

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
            # Attack: fast drop + start holdoff
            alpha = 0.3
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
        # High-pass filter: remove DC + low-freq noise
        if self._hp_mic is not None:
            near_end = self._hp_mic.process(near_end.copy())
            far_end = self._hp_ref.process(far_end.copy())

        # Saturation detection + soft-clip reference
        if self._sat_detector_ref is not None:
            sat_ref = self._sat_detector_ref.detect(far_end)
            sat_mic = self._sat_detector_mic.detect(near_end)
            self._saturation_level = max(sat_ref, sat_mic * 0.5)
            if self.config.saturation_softclip_ref and sat_ref > 0.1:
                far_end = SaturationDetector.soft_clip(far_end.copy())

        # Delay estimation + reference alignment
        if self._delay_active:
            hop = len(far_end)

            # Online delay estimation (if not using fixed delay)
            if self.delay_est is not None:
                self.delay_est.accumulate(near_end, far_end)
                new_delay = self.delay_est.estimated_delay
                par_ok = getattr(self.delay_est, '_last_par', 0) > 5.0
                if new_delay >= 0 and self.delay_est._n_updates >= 3 and par_ok:
                    if self._current_delay < 0:
                        self._current_delay = new_delay
                        # First delay acquisition: W learned on wrong
                        # alignment (~300ms of garbage) → reset filters
                        self.filter.reset()
                        if self.shadow_filter is not None:
                            self.shadow_filter.reset()
                        if self.res is not None:
                            self.res.reset()
                        self._maybe_mark_diverged('delay_first')
                        for filt in [self.filter, self.shadow_filter]:
                            if filt is not None and isinstance(filt, PBFDKF):
                                filt.Q = filt.Q_high.copy()
                    elif abs(new_delay - self._current_delay) > 32:
                        # Require two consecutive consistent estimates before updating
                        if hasattr(self, '_pending_delay') and abs(new_delay - self._pending_delay) < 16:
                            self._current_delay = new_delay
                            del self._pending_delay
                            # Bug fix: trigger EPC on delay shift — filter W/P are
                            # for old delay, need fast re-convergence
                            self._epc_det.force_delay()
                            for filt in [self.filter, self.shadow_filter]:
                                if filt is not None and hasattr(filt, 'Q'):
                                    filt.Q = filt.Q_high.copy()
                                    filt._p_max_override = 1.0
                                    filt._p_max_override_frames = 30
                            self._maybe_mark_diverged('delay_shift')
                        else:
                            self._pending_delay = new_delay

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

        # startup_dt mu floor: raise mu_scale during startup_dt so filter can converge despite DT
        # Uses previous frame's effective_dt from ResFilter (1-frame lag, acceptable)
        if self.config.startup_dt_mu_min > 0.0 and self.res is not None:
            _far_now = float(np.mean(far_end ** 2))
            _eff_dt = getattr(self.res, '_last_effective_dt', 0.0)
            if _eff_dt > 0.35 and _far_now > 1e-4 and not self._filter_once_converged:
                if isinstance(mu_scale, np.ndarray):
                    mu_scale = np.maximum(mu_scale, self.config.startup_dt_mu_min)
                else:
                    mu_scale = max(float(mu_scale), self.config.startup_dt_mu_min)

        # Mic clipping emergency: freeze filter and clamp RES output to floor.
        # Hard clipping turns mic into square waves; filter would learn garbage.
        if self._sat_detector_mic is not None:
            mic_clip = self._sat_detector_mic.saturation_level
            if mic_clip > 0.8:
                mu_scale = 0.0
                if self.res:
                    self.res.gain_smooth[:] = self.res.g_min

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
                # WebRTC-style: freeze main filter weights when shadow detected divergence
                main_mu = 0.0 if self._shadow_copy_ctrl.main_paused else mu_scale
                raw_output = self.filter.process(near_end, far_end, main_mu)

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
                shadow_mu_scale = 1.0 if (far_excited and saturation_safe) else 0.1
                self.shadow_filter.process(near_end, far_end, shadow_mu_scale)

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

                # Copy gate: delegated to ShadowCopyController. The controller
                # owns _copy_err_baseline / streak counters / pause hangover and
                # returns a ShadowCopyDecision. Filter mutations (Q-boost, reverse
                # copy) are applied here so the controller stays decision-only.
                # Phase C1: optional coherence+delay gate inputs (default gate_mode='energy'
                # ignores them, so legacy behavior parity-preserved).
                _dt_coh = self.dtd_coherence.confidence if self.dtd_coherence else 0.0
                _delay_reliable = (
                    self.delay_est is not None
                    and getattr(self.delay_est, '_n_updates', 0) >= 3
                    and getattr(self.delay_est, '_last_par', 0.0) > 5.0
                )
                shadow_decision = self._shadow_copy_ctrl.update(
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
                        self.filter.Q = self.filter.Q_high.copy()
                        self.filter._p_max_override = 1.0
                        self.filter._p_max_override_frames = 20
                if shadow_decision.reverse_copy:
                    # Sync shadow back to main when main is clearly better.
                    # (PBFDAF←PBFDKF copy has no Kalman state to corrupt.)
                    self.shadow_filter.copy_weights_from(self.filter)
                    self.shadow_err_smooth = self.main_err_smooth

            # EchoPathVariability: gain-change detection (delegated to EchoPathChangeDetector)
            epv_event = self._epc_det.update_epv(
                far_pwr_global=far_pwr_global,
                filter_converged=self._filter_converged,
                main_paused=self._shadow_copy_ctrl.main_paused,
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
                for filt in [self.filter, self.shadow_filter]:
                    if filt and hasattr(filt, 'Q'):
                        filt.Q = filt.Q_high.copy()
                        filt._p_max_override = 1.0
                        filt._p_max_override_frames = 30
                        filt._p_floor_beta = 1.0
                        filt._p_floor_beta_frames = 30
                self._maybe_mark_diverged('epv')
                self._epc_render_forced_remaining = self.config.epc_hangover
                self._erl_estimate = min(self._erl_estimate, 0.3)

            # Echo path change: shadow-error rise (delegated to EchoPathChangeDetector).
            # Update + hangover tick are inside the original (shadow_filter, filter_converged)
            # gate to preserve bit-exact countdown semantics from v2.8.1.
            if self.shadow_filter is not None and self._filter_converged:
                rise_event = self._epc_det.update_shadow_rise(
                    main_err_smooth=self.main_err_smooth,
                    shadow_err_smooth=self.shadow_err_smooth,
                    is_stationary=self._render_activity.is_stationary,
                )
                if rise_event.fired:
                    if self.dtd_coherence:
                        self.dtd_coherence.confidence *= 0.3
                    for filt in [self.filter, self.shadow_filter]:
                        if filt and hasattr(filt, 'Q'):
                            filt.Q = filt.Q_high.copy()
                    self._maybe_mark_diverged('shadow_rise')
                    # P_MAX relax + P_floor raise: force filter to abandon stale path estimate
                    for filt in [self.filter, self.shadow_filter]:
                        if filt:
                            filt._p_max_override = 1.0
                            filt._p_max_override_frames = 30
                            filt._p_floor_beta = 1.0
                            filt._p_floor_beta_frames = 30
                    # Change D: arm RES render-forced + cap stale ERL
                    self._epc_render_forced_remaining = self.config.epc_hangover
                    self._erl_estimate = min(self._erl_estimate, 0.3)
                else:
                    # Hangover tick — only when shadow_rise did NOT fire (preserves
                    # original if/elif/else structure exactly).
                    self._epc_det.tick_hangover()

            # WebRTC-style: no output switching. Main filter output is always used.
            # (Shadow filter drives divergence detection + Q boost + pause, not output selection.)

            # final_output starts from raw_output; RES modifies final_output only
            self._last_raw_output = raw_output  # save for diagnostic (time-domain echo power)
            final_output = raw_output.copy()

            # RES post-filter using OLA + sqrt-Hann (skip for buffered FDAF)
            if (self.res or self.config.return_res_context) and self._freq_near_queue is None:
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
                base_over_sub = self.config.res_over_sub_base + self.config.res_over_sub_scale * erle_factor
                # Saturation boost: non-linear echo needs more suppression
                base_over_sub += self._saturation_level * self.config.saturation_over_sub_boost
                # DT protection
                far_pwr = np.mean(far_end ** 2) + 1e-10
                mic_pwr = np.mean(near_end ** 2) + 1e-10
                raw_err_pwr = np.mean(raw_output ** 2) + 1e-10
                # B4: track ERL for render-based echo estimate.
                # Gate: only update when residual is not dominated by near-end
                # speech (raw_dt < 2.0 allows high-coupling echo-only through).
                # Pre-convergence only: after convergence, filter-based echo
                # estimate is reliable and render-based mode is off.
                if far_pwr > 1e-4:
                    raw_dt_ratio = raw_err_pwr / (far_pwr + 1e-10)
                    inst_erl_raw = mic_pwr / far_pwr
                    # v3.2 Axis 1: NE-corruption protection. ERL > 1.5 physically
                    # implausible (mic louder than far → NE dominates), so skip update.
                    if raw_dt_ratio < 2.0 and inst_erl_raw < 1.5:
                        inst_erl = np.clip(inst_erl_raw, 0.001, 1.0)
                        alpha_erl = 0.99 if not self._filter_converged else 0.999
                        self._erl_estimate = alpha_erl * self._erl_estimate + (1 - alpha_erl) * inst_erl

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
                # NOTE: dt_reduction → effective_over_sub is dead code when
                # gain_type="enr" (all presets). over_sub is only read by
                # wiener/spectral_sub paths. Kept for backward compat if
                # gain_type is ever changed.
                dt_reduction = self.config.res_dt_reduction * dt_indicator
                effective_over_sub = max(base_over_sub - dt_reduction, 0.5)

                # Divergence indicator EMA (delegated to FilterConvergenceAnalyzer)
                self._convergence.update_divergence(self.near_power, self.raw_error_power)

                if self.res:
                    # Change D: during EPC render-forced window, force RES
                    # into render-based echo estimate (unreliable filter W).
                    if getattr(self, '_epc_render_forced_remaining', 0) > 0:
                        self._epc_render_forced_remaining -= 1
                        self.res._using_render_based = True
                    self.res.over_sub = effective_over_sub

                    # DT conservative residual scaling: 1.0→0.5 as dt goes 0→0.8
                    dt_residual_scale = 1.0 - 0.5 * float(np.clip(dt_indicator, 0.0, 0.8) / 0.8)
                    eff_echo_spec = self.filter.echo_spec * dt_residual_scale

                    _shadow_dt = max(float(self._dt_from_energy),
                                     float(getattr(self, '_dt_from_shadow', 0.0)))
                    shadow_dt = 0.08 * _shadow_dt if self.epc_active else _shadow_dt

                    final_output = self.res.process(raw_output, eff_echo_spec,
                                                    far_power, self.filter.far_spec,
                                                    filter_converged=self._filter_converged,
                                                    erle_factor=erle_factor,
                                                    dt_indicator=float(dt_indicator),
                                                    near_spec=self.filter.near_spec,
                                                    divergence=self._divergence_indicator,
                                                    is_stationary_dt=is_stationary_dt,
                                                    saturation_level=self._saturation_level,
                                                    epc_active=self.epc_active,
                                                    shadow_dt=shadow_dt,
                                                    erl_estimate=self._erl_estimate,
                                                    e2_main=float(self.main_err_smooth),
                                                    e2_shadow=float(self.shadow_err_smooth),
                                                    y2=float(far_power),
                                                    filter_once_converged=self._filter_once_converged,
                                                    aec_state=self._aec_state)

                    # Update per-bin mu_scale AFTER RES (echo_psd is now current frame)
                    if not self.config.enable_dtd:
                        if self._filter_converged:
                            per_bin_eer = self.res.echo_psd / (self.res.error_psd + 1e-10)
                            per_bin_eer = np.clip(per_bin_eer, 0.0, 1.0)
                            mu_min = self.config.shadow_mu_min
                            self._per_bin_mu_scale = (mu_min + (1.0 - mu_min) * per_bin_eer).astype(np.float32)
                            self._simple_mu_ratio = float(np.mean(per_bin_eer))
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

                if self.config.return_res_context and not self.res:
                    _res_context = AecResContext(
                        raw_output=raw_output.copy(),
                        echo_spec=self.filter.echo_spec.copy(),
                        far_power=far_power,
                        far_spec=self.filter.far_spec.copy(),
                        near_spec=self.filter.near_spec.copy(),
                        filter_converged=self._filter_converged,
                        erle_factor=float(erle_factor),
                        dt_indicator=float(dt_indicator),
                        divergence=float(self._divergence_indicator),
                        over_sub=float(effective_over_sub),
                        saturation_level=float(self._saturation_level),
                        erl_estimate=float(self._erl_estimate),
                    )

            # C-parity fix: when RES is disabled, _update_simple_mu_ratio is never
            # called in PBFDKF path. C always calls update_simple_mu_ratio regardless.
            # Add call here when RES is not active (avoids double-update when RES is on).
            if not self.res and not self.config.enable_dtd and not self._filter_converged:
                self._update_simple_mu_ratio(raw_output, far_end)

            # Update diagnostics
            if self.res and hasattr(self.res, '_diag_gain_mean'):
                self._diag['res_gain_mean'] = self.res._diag_gain_mean
                self._diag['res_gain_min'] = self.res._diag_gain_min
                self._diag['effective_g_min'] = self.res._diag_effective_g_min
                self._diag['far_activity'] = self.res._diag_far_activity
                self._diag['echo_psd_mean'] = self.res._diag_echo_psd_mean
                self._diag['error_psd_mean'] = self.res._diag_error_psd_mean
            self._diag['erle_inst'] = self.get_erle_instant()
            mu_val = mu_scale
            self._diag['mu_scale'] = float(np.mean(mu_val)) if isinstance(mu_val, np.ndarray) else float(mu_val)
            self._diag['converged'] = self._filter_converged
            self._diag['erle_factor'] = float(erle_factor) if 'erle_factor' in locals() else 0.0
            self._diag['divergence'] = self._divergence_indicator
            # G4: expanded diagnostics
            self._diag['using_render_based'] = bool(getattr(self.res, '_using_render_based', False)) if self.res else False
            self._diag['shadow_advantage'] = getattr(self, '_shadow_advantage', 1.0)
            self._diag['dt_from_energy'] = self._dt_from_energy
            self._diag['dt_from_shadow'] = getattr(self, '_dt_from_shadow', 0.0)
            self._diag['erl_estimate'] = self._erl_estimate
            self._diag['epc_active'] = self.epc_active
            self._diag['saturation_level'] = self._saturation_level
            self._diag['erle_windowed'] = float(erle_windowed) if 'erle_windowed' in locals() else 0.0
            # DT / filter debug fields
            self._diag['dt_indicator'] = float(dt_indicator) if 'dt_indicator' in locals() else 0.0
            self._diag['main_err_smooth'] = float(getattr(self, 'main_err_smooth', 0.0))
            self._diag['shadow_err_smooth'] = float(getattr(self, 'shadow_err_smooth', 0.0))
            self._diag['main_paused'] = bool(self._shadow_copy_ctrl.main_paused)
            _epv_ratio = (self._epv_gain_fast / (self._epv_gain_slow + 1e-10)
                          if self._epv_gain_slow > 1e-12 else 1.0)
            self._diag['epv_gain_ratio'] = float(_epv_ratio)
            self._diag['dt_residual_scale'] = float(dt_residual_scale) if 'dt_residual_scale' in locals() else 1.0
            self._diag['filter_w_norm'] = float(np.linalg.norm(self.filter.W)) if hasattr(self.filter, 'W') else 0.0
            self._diag['shadow_w_norm'] = (float(np.linalg.norm(self.shadow_filter.W))
                                            if self.shadow_filter and hasattr(self.shadow_filter, 'W') else 0.0)
            self._diag['copy_err_baseline'] = float(self._shadow_copy_ctrl.copy_err_baseline)
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
        near_peak = np.max(np.abs(near_end))
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

        # ── Phase 0 trace-only AEC3 state diagnostics (read-only; do not gate audio) ──
        _initial_state_active = (
            (not self._filter_once_converged)
            or self._frame_count < (self.config.warmup_frames + 50)
        )
        _initial_transition_triggered = bool(just_converged)
        _epc_hangover = self._epc_det.hangover_count if hasattr(self, '_epc_det') else 0
        _usable_v1 = self._aec_state.usable_linear_estimate
        _usable_v2 = (
            _usable_v1
            and self._frame_count > self.config.warmup_frames + 30
            and self._convergence.divergence < 0.3
            and _epc_hangover < 1
        )
        # dominant_nearend with hold counter + initial-state gate
        _ne_raw = bool(getattr(self.res, '_diag_dominant_nearend_raw', False)) if self.res else False
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
        elif self._aec_state.epc_active:
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

        # ── Round 4 trace: per-bin RES diagnostics (audio-passive) ──
        if self.res is not None:
            for _k, _v in getattr(self.res, '_diag_round4', {}).items():
                self._diag[_k] = _v
        # ── end Round 4 trace ──

        # ── Round 5 trace: per-stage gain means (voice-band, audio-passive) ──
        if self.res is not None:
            _r5_stages = getattr(self.res, '_diag_round5_stages', None)
            if _r5_stages is not None:
                _R5_NAMES = ('softgate_emr', 'spectral_floor', 'epc_dt_cap',
                             'quiet_mask', '3bin_smooth', 'hf_cap',
                             'pre_temporal', 'post_temporal', 'after_noise_lift')
                for _i, _n in enumerate(_R5_NAMES):
                    self._diag[f'g_stage_{_n}_voice'] = float(_r5_stages[_i])
        # ── end Round 5 trace ──

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
        """Return the highest-priority active filter state."""
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
            shadow_copy_count=self._shadow_copy_ctrl.copy_counter,
            main_paused=self._shadow_copy_ctrl.main_paused,
            res_gain_mean_db=_db(d.get('res_gain_mean', 1.0)),
            res_using_render=d.get('using_render_based', False),
            echo_psd_mean_db=_db(d.get('echo_psd_mean', 1e-10)),
            error_psd_mean_db=_db(d.get('error_psd_mean', 1e-10)),
        )

    def GetStats(self) -> AecStats:
        return self.get_stats()


class AecDebugLogger:
    """Periodically logs AEC internal state to console.

    Usage::

        aec = AEC(config)
        logger = AecDebugLogger(aec, log_interval_s=1.0)
        # inside processing loop:
        output = aec.process(mic, ref)
        logger.update()
    """

    _HEADER = (
        "  t(s)  | state          |fltr| epc |dt_conf|ERLE_i|ERLE_w|  ERL |"
        "far_dB|mic_dB|mu_scl|shd_adv|sat | delay"
    )
    _SEP = "-" * len(_HEADER)

    def __init__(self, aec: 'AEC', log_interval_s: float = 1.0):
        self._aec = aec
        self._interval_frames = max(1, int(log_interval_s / (aec._hop_size / aec.config.sample_rate)))
        self._tick = 0
        self._header_interval = 20

    def update(self) -> None:
        self._tick += 1
        if self._tick % self._interval_frames != 0:
            return
        s = self._aec.get_stats()
        line_no = self._tick // self._interval_frames
        if line_no % self._header_interval == 1:
            print(self._SEP)
            print(self._HEADER)
            print(self._SEP)
        conv = 'Y' if s.filter_converged else 'N'
        epc  = 'Y' if s.epc_active else 'N'
        print(
            f"{s.time_s:7.2f}  | {s.filter_state.value:<14s} | {conv}  | {epc}  |"
            f" {s.dt_confidence:5.3f} |{s.erle_inst_db:6.1f}|{s.erle_windowed_db:6.1f}|"
            f"{s.erl_db:6.1f}|{s.far_power_db:6.1f}|{s.mic_power_db:6.1f}|"
            f" {s.mu_scale:5.3f}| {s.shadow_advantage:5.3f} |{s.saturation_level:4.2f}|"
            f" {s.delay_ms:.1f}ms"
        )


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
    parser.add_argument('--enable-res', action='store_true', help='Enable RES post-filter')
    parser.add_argument('--res-g-min', type=float, default=-20.0, help='RES min gain (dB)')
    parser.add_argument('--cng', action='store_true', help='Enable comfort noise generation in RES (default: off)')
    parser.add_argument('--no-cng', action='store_true', help='(deprecated, CNG off by default)')
    parser.add_argument('--no-td-constraint', action='store_true',
                        help='Disable time-domain constraint on filter weights (diagnostic)')
    parser.add_argument('--preset', choices=['mild', 'soft', 'balanced', 'aggressive', 'maximum'],
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
        enable_res=args.enable_res,
        enable_cng=args.cng,
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
    if args.preset:
        config = AecConfig.from_preset(args.preset, **common_kw)
    else:
        config = AecConfig(**common_kw)

    process_wav_files(args.mic, args.ref, args.output, config, diag=args.diag)


if __name__ == '__main__':
    main()
