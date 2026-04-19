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
    MILD = "mild"               # Best near-end preservation, lightest echo suppression
    BALANCED = "balanced"       # Balanced echo suppression and near-end quality (default)
    AGGRESSIVE = "aggressive"   # Stronger echo suppression, moderate near-end impact
    MAXIMUM = "maximum"         # Maximum echo suppression, significant near-end impact


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
            if self.sample_rate >= 44100:
                self.filter_length = self.sample_rate * 64 // 1000  # 64ms
            else:
                self.filter_length = self.sample_rate * 32 // 1000  # 32ms

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
                # RES suppression (lightest)
                res_g_min_db=-35.0,
                res_over_sub_base=2.5,
                res_over_sub_scale=4.0,
                res_dt_reduction=3.5,
                res_spectral_floor_db=-25.0,
                res_ne_protect_db=-10.0,
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
                res_reverb_decay=0.65,
                res_reverb_gain=1.4,
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

        for p in range(self.n_partitions):
            p_idx = (curr_p - p) % self.n_partitions
            X = self.X_buf[p_idx]

            K_optimal = (self.P[p] * np.conj(X)) / denominator

            # Bug 2 fix: separate K for weights (scaled) and P update (unscaled)
            K_scaled = K_optimal * mu_scale_arr

            self.W[p] += K_scaled * self.error_spec

            # Covariance update with UNSCALED K_optimal
            # C-parity fix: use np.float32(1.0) to avoid float64 promotion
            # in (1.0 - KX) where Python literal 1.0 is float64. C uses
            # float32 throughout so this intermediate stays float32.
            KX = np.real(K_optimal * X).astype(np.float32)
            self.P[p] = np.minimum(
                np.maximum((np.float32(1.0) - KX) * self.P[p] + Q_gated, P_floor),
                p_max
            )

            # Time-domain constraint (raised cosine fade)
            if self.enable_td_constraint:
                w_time = np.fft.irfft(self.W[p], self.fft_size).astype(np.float32)
                w_time *= self._td_window
                self.W[p] = np.fft.rfft(w_time).astype(np.complex64)

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
                 sample_rate: int = 16000):
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
        self._render_based_hold = 0
        self._using_render_based = False
        self._diag_gain_mean = 1.0
        self._diag_gain_min = 1.0
        self._diag_effective_g_min = 1.0
        self._diag_far_activity = 0.0
        self._diag_echo_psd_mean = 0.0
        self._diag_error_psd_mean = 0.0
        self._filter_erle_est.reset()
        self._fb_erle_est.reset()

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
                erl_estimate: float = 0.01) -> np.ndarray:
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

        # --- Residual echo PSD estimation ---
        eps = 1e-10
        residual_echo_psd = None

        if self.echo_method == "direct" and near_spec is not None:
            # Direct method: use echo estimate + per-bin ERLE to estimate residual
            near_pwr = np.abs(near_spec) ** 2
            # 4-block moving average for stable near-end PSD (cf. AEC3)
            self.near_psd_buf[self.near_psd_idx] = near_pwr
            self.near_psd_idx = (self.near_psd_idx + 1) % 4
            self.near_psd = np.mean(self.near_psd_buf, axis=0)

            # Multi-ERLE residual estimation (Phase 2)
            # FilterErleEstimator breaks circular dependency: echo_spec²/error_spec²
            confidence = compute_erle_confidence(
                self._filter_erle_est.erle, self._fb_erle_est.fb_erle
            )

            # Corrected ERLE: blend L1 (filter-based) with conservative fallback (1.0)
            erle_corrected = (confidence * self._filter_erle_est.erle
                              + (1.0 - confidence) * 1.0)
            erle_corrected = np.maximum(erle_corrected, 0.5)

            # Residual from multi-ERLE (can be << error_psd when ERLE is high)
            erle_est = self.echo_psd / erle_corrected
            direct_est = self.echo_psd

            # Nonlinear echo floor
            if far_power > 1e-4:
                dt_weight = 1.0 - dt_for_fs
                nonlinear_floor = (self.error_psd * coh2
                                   * self.far_activity * dt_weight)
                direct_est = np.maximum(direct_est, nonlinear_floor)
                erle_est = np.maximum(erle_est, nonlinear_floor)

            residual_echo_psd = (1.0 - erle_factor) * direct_est + erle_factor * erle_est

            # AEC3-style echo estimate switching (residual_echo_estimator.cc):
            # When filter is converged (erle_factor high → UsableLinearEstimate),
            # use filter-based echo_psd (accurate). When unconverged, switch to
            # render_power × echo_path_gain (conservative, avoids garbage).
            # AEC3 transparent mode uses gain² = 0.01² = 1e-4; we use 0.01
            # as default echo_path_gain (≈ -20 dB ERL assumption).
            if far_power > 1e-4:
                # ENR-adaptive switching threshold: high ambient noise (low ENR)
                # makes erle_factor unreliable — use tighter threshold to avoid
                # false render-based switching that over-suppresses echo.
                # Clean environment (high ENR) keeps standard 0.5 threshold.
                error_power_mean = float(np.mean(self.error_psd)) + 1e-10
                enr = far_power / error_power_mean
                # enr high (clean) → threshold ≈ 0.35 (standard behavior)
                # enr low (noisy)  → threshold ≈ 0.15 (less switching)
                switching_threshold = 0.5 * np.clip(enr / (enr + 1.0), 0.3, 0.7)

                # Hysteresis: once in render-based mode, require higher erle_factor
                # to exit (prevents oscillation at boundary → DT onset artifacts)
                hysteresis = 0.05
                if getattr(self, '_using_render_based', False):
                    effective_threshold = switching_threshold + hysteresis
                else:
                    effective_threshold = switching_threshold
                # G2: explicit overrides for known unreliable states
                # Note: divergence > 0.5 removed — DT triggers false divergence
                # (mic = echo + speech → output > input). Divergence is already
                # handled by gain cap (L1514), no need for render-based override.
                force_render = (
                    epc_active
                    or saturation_level > 0.5
                )
                want_render = (erle_factor < effective_threshold) or force_render
                # G5: minimum hold time — once in render-based, stay ≥5 frames
                if want_render and not self._using_render_based:
                    self._render_based_hold = 5  # 50ms minimum hold
                if self._using_render_based:
                    self._render_based_hold = max(self._render_based_hold - 1, 0)
                can_exit = (not want_render and self._render_based_hold == 0)
                self._using_render_based = want_render or (self._using_render_based and not can_exit)

                if self._using_render_based:
                    # Filter unreliable → render-based conservative estimate
                    far_psd = np.abs(far_spec) ** 2 if far_spec is not None else np.zeros(self.n_freqs)
                    echo_path_gain = erl_estimate  # B4: dynamic ERL from AEC
                    render_based_echo = far_psd * echo_path_gain
                    blend = 1.0 - erle_factor / effective_threshold
                    blend = np.clip(blend, 0.0, 1.0)
                    residual_echo_psd = ((1.0 - blend) * residual_echo_psd
                                         + blend * render_based_echo)

            # Nonlinear echo mode: when speaker distortion is sustained,
            # the linear filter can't model harmonics. Boost residual_echo_psd
            # globally (not just HF) and increase over_sub to compensate.
            # saturation_level > 0.3 sustained → nonlinear mode with hangover.
            if saturation_level > 0.3:
                self._nonlinear_frames += 1
            else:
                self._nonlinear_frames = max(0, self._nonlinear_frames - 1)

            is_nonlinear = self._nonlinear_frames > 5  # 50ms hangover

            if is_nonlinear and far_power > 1e-4:
                # Boost echo PSD: harmonics spread across full spectrum
                nonlinear_boost = 1.0 + 1.0 * saturation_level  # 1.3-2.0×
                residual_echo_psd = residual_echo_psd * nonlinear_boost

            # Harmonic distortion mapping: HF floor from LF echo
            # (always active, complementary to nonlinear mode above)
            if saturation_level > 0.05 and far_power > 1e-4:
                lf_start, lf_end = self._harm_lf_start, self._harm_lf_end  # C4: precomputed
                hf_start, hf_end = self._harm_hf_start, self._harm_hf_end
                if lf_end > lf_start and hf_end > hf_start:
                    lf_echo_mean = float(np.mean(residual_echo_psd[lf_start:lf_end]))
                    distortion_factor = 0.1 + 0.4 * saturation_level
                    harmonic_floor = lf_echo_mean * distortion_factor
                    residual_echo_psd[hf_start:hf_end] = np.maximum(
                        residual_echo_psd[hf_start:hf_end], harmonic_floor)

            residual_echo_psd = np.minimum(residual_echo_psd, self.echo_psd * 2.0)

            # Physical limit: residual echo cannot exceed total error energy.
            residual_echo_psd = np.minimum(residual_echo_psd, self.error_psd)

            # DT physical limit: dt_indicator=0.8 means 80% confidence it's speech,
            # so residual echo can be at most 20% of error energy.
            # Floor at 0.1 prevents residual from vanishing completely at dt≈1.0.
            dt_suppress = np.clip(1.0 - dt_for_fs**2, 0.1, 1.0)
            residual_echo_psd = np.minimum(residual_echo_psd, self.error_psd * dt_suppress)

            # P2: render-based physical ceiling — echo cannot exceed ERL × far_psd × 2.0.
            # Prevents ENR blow-up when filter echo_spec overestimates residual.
            # Conservative (factor=2.0): only caps when residual greatly exceeds expected echo.
            if far_spec is not None and far_power > 1e-4 and erl_estimate > 0.0:
                far_psd_k = np.abs(far_spec) ** 2
                render_ceil = far_psd_k * min(erl_estimate * 2.0, 1.0)
                residual_echo_psd = np.minimum(residual_echo_psd, render_ceil)

            # Add reverb tail if enabled
            # Use render signal (far_spec) power instead of filter echo estimate
            # → doesn't depend on filter modeling quality for reverb tail
            if self.enable_reverb:
                far_psd = np.abs(far_spec) ** 2 if far_spec is not None else echo_pwr_linear
                self.reverb_psd = (self.reverb_decay * self.reverb_psd
                                   + (1 - self.reverb_decay) * far_psd)
                # Gate by far_activity; continuous NE/FS blend (not binary nearend_state)
                # Stationary DT: far_psd is huge WN energy, reverb accumulates and drowns speech
                if not is_stationary_dt:
                    ne_reverb_factor = 0.3 + 0.7 * self.far_activity * (1.0 - dt_for_fs)
                    reverb_gate = self.far_activity * ne_reverb_factor
                    residual_echo_psd = (residual_echo_psd
                                         + self.reverb_gain * self.reverb_psd * reverb_gate)

        # Per-bin echo boost: high-coh2 bins are echo-dominant → boost residual estimate
        # Only when filter converged (erle_factor > 0.3) to avoid over-suppression early
        if far_power > 1e-4 and erle_factor > 0.3 and residual_echo_psd is not None:
            echo_boost = 1.0 + 0.5 * coh2 if dt_for_fs < 0.2 else np.ones_like(coh2)
            residual_echo_psd = residual_echo_psd * echo_boost

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
        # Use effective_dt so energy DT also drives ne_protection & attack speed
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

        # --- Gain computation ---

        if self.gain_type == "enr" and residual_echo_psd is not None:
            raw_nearend_est = np.maximum(self.error_psd - residual_echo_psd, 0.0)
            noise_floor_psd = np.mean(self.error_psd) * 0.01 + 1e-10

            # Per-bin DT indicator: base from coh2 (works for speech far-end)
            # Use effective_dt (includes energy DT) so nearend_est denominator
            # rises in high-coupling DT where dt_for_fs ≈ 0.
            dt_per_bin = np.maximum(
                np.full(self.n_freqs, effective_dt, dtype=np.float32),
                1.0 - coh2
            )

            # Stationary far-end DT: coh2 fails (all bins high), use precomputed mask (C1)
            if is_stationary_dt:
                dt_per_bin = np.maximum(dt_per_bin, self._stat_dt_mask)

            # dt=0.5 → 0.41 (was 0.25), dt=0.3 → 0.22 (was 0.09), dt=0.8 → 0.75 (was 0.64).
            # Squared factor over-suppressed mid-range DT (>1 AECMOS damage on 51% of cases).
            dt_shaped_per_bin = dt_per_bin ** 1.1
            nearend_est = np.maximum(raw_nearend_est * dt_shaped_per_bin, noise_floor_psd)

            min_ne_from_dt = self.error_psd * dt_shaped_per_bin
            nearend_est = np.maximum(nearend_est, min_ne_from_dt)

            ne_physical_floor = self.error_psd * 0.05
            nearend_est = np.maximum(nearend_est, ne_physical_floor)


            enr = residual_echo_psd / (nearend_est + 1e-10)

            # --- ENR two-tuning (per-bin ne_confidence) ---
            blend = self._enr_blend  # C2: precomputed
            scale = self.enr_scale
            ne_confidence = dt_per_bin

            effective_scale = scale

            # Standard thresholds (ENR can now >> 1.0 with dt_indicator × nearend_est)
            enr_t_ne = (1 - blend) * 2.0 + blend * 1.5
            enr_s_ne = (1 - blend) * 3.0 + blend * 2.5
            enr_t_fs = (1 - blend) * (0.3 * effective_scale) + blend * (0.07 * effective_scale)
            enr_s_fs = (1 - blend) * (0.4 * effective_scale) + blend * (0.1 * effective_scale)

            # Improvement A: DT-aware ENR threshold relaxation.
            # Uses effective_dt = max(dt_for_fs, shadow_dt) where shadow_dt
            # carries the pre-filter energy-based DT signal. This fires in
            # high-coupling DT (where dt_indicator is crushed by inst_erle
            # correction) while staying ≈0 in pure FS.
            if effective_dt > 0.4:
                dt_enr_relax = 1.0 + (effective_dt - 0.4) / 0.6 * 0.5  # max 1.5×
                enr_t_ne = enr_t_ne * dt_enr_relax
                enr_s_ne = enr_s_ne * dt_enr_relax

            enr_t = ne_confidence * enr_t_ne + (1 - ne_confidence) * enr_t_fs
            enr_s = ne_confidence * enr_s_ne + (1 - ne_confidence) * enr_s_fs

            # BUG-3 fix: ensure minimum gate width to prevent hard cutoff at
            # FS end. FS thresholds (enr_t_fs≈0.07, enr_s_fs≈0.1) have width
            # 0.03 → slope 33× → near-vertical cutoff → gain jumps between
            # frames as ENR hovers near the threshold. min_gate_width=0.2
            # gives slope ≤5× at FS end, making the soft gate actually soft.
            min_gate_width = 0.2
            enr_s_safe = np.maximum(enr_s, enr_t + min_gate_width)

            # Soft gate: linear interpolation between transparent/suppress
            g = np.where(enr > enr_t,
                         np.clip((enr_s_safe - enr) / (enr_s_safe - enr_t + eps), 0.0, 1.0),
                         1.0)

            # EMR: echo-to-masker ratio (AEC3-style noise masking)
            # If echo is below noise floor at a bin, don't suppress (echo is inaudible)
            if np.sum(self.noise_psd) > 0:
                emr = residual_echo_psd / (self.noise_psd + 1e-10)
                emr_transparent = 0.3
                # Bins where echo is masked by noise → raise gain toward 1.0
                g_emr = np.clip(emr_transparent / (emr + 1e-10), 0.0, 1.0)
                g = np.maximum(g, g_emr)  # Don't suppress below noise-masking level

            g = np.maximum(g, spectral_g_min)
        elif self.gain_type == "wiener" and residual_echo_psd is not None:
            # Wiener gain (fixed: higher noise floor for FS stability)
            noise_floor_psd = np.mean(self.error_psd) * 0.01 + eps  # 1% floor
            nearend_est = np.maximum(self.error_psd - residual_echo_psd, noise_floor_psd)
            beta = self.over_sub
            g = nearend_est / (nearend_est + beta * residual_echo_psd + eps)
            g = np.maximum(g, spectral_g_min)
        elif residual_echo_psd is not None:
            # Spectral subtraction with direct residual echo PSD
            eer_direct = residual_echo_psd / (self.error_psd + eps)
            g = np.maximum(1.0 - self.over_sub * eer_direct, spectral_g_min)
        else:
            # Legacy coherence-based spectral subtraction
            g = np.maximum(1.0 - self.over_sub * eer, spectral_g_min)
        g[quiet_mask] = 1.0  # Noise gate: pass through quiet bins

        # --- Frequency-domain postprocessing (cf. AEC3 PostprocessGains) ---
        if far_power > 1e-4:
            # 3-bin cross-frequency smoothing: reduce isolated gain peaks/valleys
            # that cause musical noise / electrical noise artifacts
            kernel = np.array([0.25, 0.5, 0.25], dtype=np.float32)
            g = np.convolve(g, kernel, mode='same').astype(np.float32)
            # DC consistency: bins 0-1 follow bin 2
            if self.n_freqs > 2:
                g[:2] = np.minimum(g[1], g[2])
            # HF cap: upper bins capped at gain of bin near ~500Hz
            # Bypass during DT: speech harmonics above 500Hz must not be crushed
            hf_cap_bin = self._hf_cap_bin  # C3: precomputed
            if self.n_freqs > hf_cap_bin + 1 and effective_dt < 0.5 and not is_stationary_dt:
                hf_cap = g[hf_cap_bin]
                g[hf_cap_bin + 1:] = np.minimum(g[hf_cap_bin + 1:], hf_cap)

        # Divergence override: when filter diverges, cap gain severely
        if divergence > 0.3:
            divergence_gain = 0.01 + (1.0 - 0.01) * (1.0 - divergence)
            g = np.minimum(g, divergence_gain)

        # Temporal smoothing: far_activity-driven release (no feedback loop)
        # far_activity high (far-end speaking) → slow release (TC≈200ms)
        # far_activity low (far-end silent) → fast release (TC≈25ms)
        # B1 cleanup: alpha_release_base was computed but never used (legacy dual EMA).
        # Actual release uses alpha_release_light only (rate-clamp approach).

        # Temporal DT: when Stationary DT confirmed, treat as dt=0.8 for smoothing/rate
        # Include effective_dt (shadow+energy) at 0.5× to help gain release/rise in
        # high-coupling DT where dt_indicator≈0, without being too aggressive
        dt_temporal = 0.8 if is_stationary_dt else max(dt_indicator, effective_dt * 0.5)

        # Temporal smoothing: AEC3-style split attack/release.
        # ATTACK (g < gain_smooth): EMA for smooth echo suppression onset.
        #   Fast attack when FS-confident, slow when DT/NE.
        # RELEASE (g >= gain_smooth): rate-clamp only, NO EMA.
        #   AEC3 uses max_inc_factor=2.0 without EMA. Our previous dual
        #   EMA+clamp made recovery 3-5× slower than AEC3, causing speech
        #   crush in DT_bal segments (51% of DT cases damaged >1.0 AECMOS).
        #   Rate-clamp-only lets gain recover in ~7 frames (70ms) from FS→DT
        #   transition, vs ~20 frames (200ms) with EMA.
        alpha_fast = 0.3 + 0.2 * (1.0 - erle_factor)   # 0.3-0.5
        alpha_slow = 0.85 + 0.1 * (1.0 - erle_factor)  # 0.85-0.95
        alpha_attack = alpha_slow + (alpha_fast - alpha_slow) * fs_confidence

        # Stationary DT: also speed up attack (let gain rise to protect speech)
        # dt_temporal is already 0.8 (set above when is_stationary_dt)
        if is_stationary_dt:
            alpha_attack = alpha_attack * np.clip(1.0 - dt_temporal**2, 0.1, 1.0)

        # Split attack/release: attack uses heavy EMA (smooth onset), release
        # uses light EMA (α=0.5, TC≈20ms) for fast recovery while preventing
        # per-bin echo bounce that full bypass (α=0) caused in FS cases.
        # Improvement B: DT-aware release acceleration.
        # During DT, reduce release EMA inertia so gain recovers faster
        # (speech not truncated). dt_indicator=0.8 → alpha=0.34.
        alpha_release_light = 0.5 - 0.2 * dt_temporal
        smoothed = np.where(g < self.gain_smooth,
                            alpha_attack * self.gain_smooth + (1 - alpha_attack) * g,
                            alpha_release_light * self.gain_smooth + (1 - alpha_release_light) * g)

        # --- Gain rate limiting: prevent sudden blackout / pop ---
        # Relax rate limiting when far-end is silent (near-end needs to pass through)
        activity_scale = 0.5 + 0.5 * self.far_activity  # [0.5, 1.0]
        eff_drop = self.max_drop_ratio ** activity_scale  # Less limiting when silent
        # Tighter rise when far-end active (cf. AEC3 max_inc_factor=2.0)
        # Improvement B: DT-aware rise boost. During DT, far_activity is
        # still high but near-end speech needs faster gain recovery.
        rise_exp = 0.5 + 0.5 * (1.0 - self.far_activity)
        if dt_temporal > 0.3:
            dt_rise_boost = 1.0 + dt_temporal  # dt=0.8 → 1.8× faster
            rise_exp = rise_exp / dt_rise_boost
        eff_rise = self.max_rise_ratio ** rise_exp

        gain_floor = self.gain_smooth / eff_drop
        gain_ceil = self.gain_smooth * eff_rise
        # LF gain decrease limiting: scale by (1-fs_confidence)
        # FS (fs_conf≈1): no LF protection → gain drops freely
        # DT/NE (fs_conf≈0): full LF protection (0.25× floor)
        if fs_confidence < 0.9:  # skip entirely when clearly FS
            lf_limit = min(8, self.n_freqs)
            lf_factor = 0.25 * max(1.0 - fs_confidence * 2.0, 0.0)
            if lf_factor > 0.01:
                gain_floor[:lf_limit] = np.maximum(
                    gain_floor[:lf_limit], self.gain_smooth[:lf_limit] * lf_factor)
        smoothed = np.maximum(smoothed, gain_floor)
        smoothed = np.minimum(smoothed, gain_ceil)
        # Clamp to valid range
        if isinstance(spectral_g_min, np.ndarray):
            smoothed = np.maximum(smoothed, spectral_g_min)
        else:
            smoothed = np.maximum(smoothed, effective_g_min)
        smoothed = np.minimum(smoothed, 1.0)
        self.gain_smooth = smoothed

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
        # Only raises floor where noise tracker shows persistent content (incl.
        # quiet speech). Above FS echo (loud), spec_pwr >> noise_psd → floor
        # is small → no effect on FS. Below quiet speech → floor kicks in.
        spec_pwr_synth = np.abs(spec_synth) ** 2 + 1e-10
        noise_floor_gain = np.sqrt(self.noise_psd / spec_pwr_synth)
        noise_floor_gain = np.clip(noise_floor_gain, effective_g_min, 1.0)
        self.gain_smooth = np.maximum(self.gain_smooth, noise_floor_gain)

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

        # Diagnostic: store latest gains for external access
        self._diag_gain_mean = float(np.mean(self.gain_smooth))
        self._diag_gain_min = float(np.min(self.gain_smooth))
        self._diag_effective_g_min = float(effective_g_min)
        self._diag_far_activity = float(self.far_activity)
        self._diag_echo_psd_mean = float(np.mean(self.echo_psd))
        self._diag_error_psd_mean = float(np.mean(self.error_psd))

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


class AEC:
    """
    Acoustic Echo Cancellation

    Supports five filter modes:
    - NLMS:    Time-domain NLMS (sample-by-sample processing)
    - FDAF:    Frequency-domain Adaptive Filter (single FFT block, n_partitions=1)
    - PBFDAF:  Partitioned Block FDAF (NLMS adaptation)
    - PBFDKF:  Partitioned Block FDKF (Kalman adaptation, recommended)
    """

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
                sample_rate=self.config.sample_rate,
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

        # Echo path change detection state
        self.prev_total_err = 0.0
        self.epc_active = False
        self.epc_hangover_count = 0

        # #4: Confidence memory decay
        self.prev_dtd_conf = 0.0

        # Convergence state: prevent divergence DTD and allow higher mu_min
        # until filter has demonstrated basic echo cancellation (ERLE > 3 dB)
        self._filter_converged = False

        # Divergence indicator: smoothed signal [0,1] for suppressor override
        self._divergence_indicator = 0.0
        # EPC render-forced countdown (Change D)
        self._epc_render_forced_remaining = 0
        # Dynamic ERL estimate for render-based echo (B4)
        self._erl_estimate = 0.1  # initial -20dB, conservative
        # Pre-filter energy-based DT signal (Stage B)
        self._dt_from_energy = 0.0

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
            # G4: expanded diagnostics
            'using_render_based': False,
            'shadow_advantage': 1.0,
            'dt_from_energy': 0.0,
            'dt_from_shadow': 0.0,
            'erl_estimate': 0.1,
            'epc_active': False,
            'saturation_level': 0.0,
            'erle_windowed': 0.0,
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

        # White noise stationarity gate state
        self._far_env_mean = 1e-10
        self._far_env_var = 0.0
        self._far_active_prev = False
        self._is_stationary_far = False
        self._stat_far_hangover = 0
        self._inst_erle_smooth = 1.0
        self._wn_err_baseline = 1e-8
        self._stat_dt_hangover = 0  # Stationary DT hold-off counter (frames)

        # Simple variable mu (for non-DTD modes, inspired by Valin 2007 RER)
        self._simple_mu_ratio = 1.0
        self._simple_mu_holdoff = 0  # holdoff counter: blocks release for N frames
        self._warmup_frames = self.config.warmup_frames
        self._warmup_far_active = False  # only consume warmup when far-end is active

        # #5: Copy hysteresis counter
        self.shadow_copy_counter = 0
        self._shadow_advantage_streak = 0  # G3: consecutive advantage frames
        self.shadow_frame_count = 0  # warm-up counter for shadow copy
        self._copy_err_baseline = 1e-6  # FS error baseline for copy gate

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
        self._conv_counter = 0  # convergence consecutive frame counter

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
        self.prev_total_err = 0.0
        self.epc_active = False
        self.epc_hangover_count = 0
        self.prev_dtd_conf = 0.0
        self._filter_converged = False
        self._divergence_indicator = 0.0
        self._erle_window_near = 1e-10
        self._erle_window_err = 1e-10
        self._erle_factor_prev = 0.0
        self._inst_erle_smooth = 1.0
        self._simple_mu_ratio = 1.0
        self._simple_mu_holdoff = 0
        self._warmup_frames = self.config.warmup_frames
        self._warmup_far_active = False
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
        }
        self.shadow_copy_counter = 0
        self._shadow_advantage_streak = 0
        self.shadow_frame_count = 0
        self._copy_err_baseline = 1e-6
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
        self._conv_counter = 0
        if self._hp_mic is not None:
            self._hp_mic.reset()
            self._hp_ref.reset()
        if self._sat_detector_ref is not None:
            self._sat_detector_ref.reset()
            self._sat_detector_mic.reset()
        self._saturation_level = 0.0
        self._far_env_mean = 1e-10
        self._far_env_var = 0.0
        self._far_active_prev = False
        self._is_stationary_far = False
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
        # Reset pre-filter DT signal states (persist-across-calls hazard)
        self._dt_from_energy = 0.0
        self._dt_from_shadow = 0.0
        self._shadow_advantage = 1.0
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
                            self.res.reset()  # A2: clear stale PSD/coh2 from wrong alignment
                        self._filter_converged = False
                        self._conv_counter = 0
                        for filt in [self.filter, self.shadow_filter]:
                            if filt is not None and isinstance(filt, PBFDKF):
                                filt.Q = filt.Q_high.copy()
                    elif abs(new_delay - self._current_delay) > 32:
                        # Require two consecutive consistent estimates before updating
                        if hasattr(self, '_pending_delay') and abs(new_delay - self._pending_delay) < 16:
                            old_delay = self._current_delay
                            self._current_delay = new_delay
                            del self._pending_delay
                            # Bug fix: trigger EPC on delay shift — filter W/P are
                            # for old delay, need fast re-convergence
                            self.epc_active = True
                            self.epc_hangover_count = getattr(self.config, 'epc_hangover', 50)
                            for filt in [self.filter, self.shadow_filter]:
                                if filt is not None and hasattr(filt, 'Q'):
                                    filt.Q = filt.Q_high.copy()
                                    filt._p_max_override = 1.0
                                    filt._p_max_override_frames = 30
                            self._filter_converged = False
                            self._conv_counter = 0
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

        # Track far-end activity for warmup gating
        self._warmup_far_active = np.mean(far_end ** 2) > 1e-6

        if self.config.enable_dtd:
            mu_scale = self._compute_mu_scale()
        else:
            mu_scale = self._get_simple_mu_scale()

        # Mic clipping emergency: freeze filter and clamp RES output to floor.
        # Hard clipping turns mic into square waves; filter would learn garbage.
        if self._sat_detector_mic is not None:
            mic_clip = self._sat_detector_mic.saturation_level
            if mic_clip > 0.8:
                mu_scale = 0.0
                if self.res:
                    self.res.gain_smooth[:] = self.res.g_min

        _res_context = None  # populated when return_res_context=True and no internal RES

        # === Global stationary feature extraction (before EPC, before all modules) ===
        far_pwr_global = np.mean(far_end ** 2) + 1e-10
        if far_pwr_global > 1e-6:
            if not self._far_active_prev:
                self._far_env_mean = far_pwr_global
                self._far_env_var = 0.0
                self._far_active_prev = True
            else:
                # α=0.99 (TC≈1s): long enough to average out hop-level WN variance
                # (single hop std ≈ 11% mean for WN; α=0.95 gave CV2 flicker)
                alpha_cv = 0.99
                old_mean = self._far_env_mean
                self._far_env_mean = (alpha_cv * self._far_env_mean
                                      + (1 - alpha_cv) * far_pwr_global)
                self._far_env_var = (alpha_cv * self._far_env_var
                                     + (1 - alpha_cv) * (far_pwr_global - old_mean) ** 2)
            far_cv2 = self._far_env_var / (self._far_env_mean ** 2 + 1e-10)
            self._is_stationary_far = (far_cv2 < 0.02)
        else:
            self._far_active_prev = False
            self._is_stationary_far = False

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
                raw_output = self.filter.process(near_end, far_end, mu_scale)

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

                if self.shadow_frame_count >= 50 and far_excited:
                    shadow_advantage = self.main_err_smooth / (self.shadow_err_smooth + 1e-10)
                    self._shadow_advantage = shadow_advantage
                    dt_from_shadow = float(np.clip((shadow_advantage - self.config.shadow_dtd_offset) / self.config.shadow_dtd_advantage_scale, 0.0, 1.0))
                    self._dt_from_shadow = (0.7 * getattr(self, '_dt_from_shadow', 0.0)
                                            + 0.3 * dt_from_shadow)
                else:
                    self._dt_from_shadow = getattr(self, '_dt_from_shadow', 0.0) * 0.95

                # Copy gate: FS error baseline tracking. Update baseline only
                # when both filters are stable (similar errors → confidently
                # not in DT/EPC). DT raises both errors above baseline so
                # `error_is_normal` flips false. High-coupling FS naturally has
                # high baseline → no false trigger.
                if self.shadow_frame_count >= 50:
                    threshold = self.config.shadow_copy_threshold
                    far_active = np.mean(far_end ** 2) > 1e-4

                    err_sum = self.main_err_smooth + self.shadow_err_smooth + 1e-10
                    err_balance = abs(self.main_err_smooth - self.shadow_err_smooth) / err_sum
                    is_stable_fs = far_active and err_balance < 0.3 and not self.epc_active
                    if is_stable_fs:
                        best_err = min(self.main_err_smooth, self.shadow_err_smooth)
                        self._copy_err_baseline = (0.995 * self._copy_err_baseline
                                                    + 0.005 * best_err)

                    error_is_normal = (self.main_err_smooth
                                       < self._copy_err_baseline * 4.0 + 1e-10)
                    not_saturating = self._saturation_level < 0.3
                    copy_allowed = (far_active and error_is_normal
                                    and not self.epc_active and not_saturating)

                    if copy_allowed:
                        if self.shadow_err_smooth < self.main_err_smooth * threshold:
                            self.shadow_copy_counter += 1
                            self._shadow_advantage_streak += 1  # G3: track duration
                        else:
                            self.shadow_copy_counter = 0
                            self._shadow_advantage_streak = 0

                        # G3: require both hysteresis AND minimum streak (100ms)
                        # to prevent short echo bursts from triggering copy
                        min_streak = 10
                        if (self.shadow_copy_counter >= self.config.shadow_copy_hysteresis
                                and self._shadow_advantage_streak >= min_streak):
                            self.filter.copy_weights_from(self.shadow_filter)
                            self.main_err_smooth = self.shadow_err_smooth
                            self.shadow_copy_counter = 0
                            self._shadow_advantage_streak = 0
                        elif (self.main_err_smooth < self.shadow_err_smooth * threshold
                              and error_is_normal):
                            self.shadow_filter.copy_weights_from(self.filter)
                            self.shadow_err_smooth = self.main_err_smooth
                    else:
                        self.shadow_copy_counter = 0
                        self._shadow_advantage_streak = 0  # Fix 1: reset on copy_allowed=False

            # Echo path change detection (shadow-based, independent of DTD)
            # DT: one filter's error ↑, other stable → ΔE/total large
            # Echo change: both errors ↑ → ΔE/total small
            # When detected: reset Q to Q_high for fast re-convergence
            # Only after convergence: before that, errors naturally rise (filter learning)
            if self.shadow_filter is not None and self._filter_converged:
                total_err = self.main_err_smooth + self.shadow_err_smooth
                if total_err > 1e-10:
                    delta_ratio = abs(self.main_err_smooth - self.shadow_err_smooth) / total_err
                else:
                    delta_ratio = 0.0

                errors_rising = (total_err > self.prev_total_err * self.config.epc_total_rise
                                 and self.prev_total_err > 1e-10)
                is_echo_change = errors_rising and delta_ratio < self.config.epc_delta_threshold
                # White noise guard: stationary far-end + error rise = DT, not echo path change
                if is_echo_change and self._is_stationary_far:
                    is_echo_change = False
                self.prev_total_err = total_err

                if is_echo_change:
                    if self.dtd_coherence:
                        self.dtd_coherence.confidence *= 0.3
                    self.epc_hangover_count = self.config.epc_hangover
                    self.epc_active = True
                    # Reset Q to Q_high for fast re-convergence
                    for filt in [self.filter, self.shadow_filter]:
                        if filt and hasattr(filt, 'Q'):
                            filt.Q = filt.Q_high.copy()
                    self._filter_converged = False
                    self._conv_counter = 0
                    # Temporarily relax P_MAX for faster re-convergence,
                    # plus raise P-floor to force the filter to abandon the
                    # stale echo path estimate.
                    for filt in [self.filter, self.shadow_filter]:
                        if filt:
                            filt._p_max_override = 1.0
                            filt._p_max_override_frames = 30
                            filt._p_floor_beta = 1.0
                            filt._p_floor_beta_frames = 30
                    # Change D: arm EPC render-forced counter. For N frames
                    # after EPC, force RES into render-based echo estimate
                    # because filter W is being heavily updated and echo_spec
                    # is unreliable. Limited duration (not hysteresis) avoids
                    # getting stuck after re-convergence.
                    self._epc_render_forced_remaining = self.config.epc_hangover  # G1: sync with EPC duration (20 frames=200ms)
                    # A4: ERL may be stale after echo path change (e.g. device moved closer)
                    self._erl_estimate = min(self._erl_estimate, 0.3)
                elif self.epc_hangover_count > 0:
                    self.epc_hangover_count -= 1
                    self.epc_active = True
                else:
                    self.epc_active = False

            # final_output starts from raw_output; RES modifies final_output only
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
                    if raw_dt_ratio < 2.0:
                        inst_erl = np.clip(mic_pwr / far_pwr, 0.001, 10.0)
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
                if self._far_active_prev and far_pwr > 1e-4:
                    erl_ceiling = 1.0 / max(self._erl_estimate, 0.01)  # learned, max 100×
                    max_echo_expected = far_pwr * erl_ceiling * 2.0    # 2× safety margin
                    dt_from_energy = max(0.0, (mic_pwr - max_echo_expected) / mic_pwr)
                else:
                    dt_from_energy = 0.0  # far silent → no DT evidence available
                # EMA smooth: fast rise (protect onset) / slow decay (hangover)
                if dt_from_energy > self._dt_from_energy:
                    self._dt_from_energy = 0.3 * self._dt_from_energy + 0.7 * dt_from_energy
                else:
                    self._dt_from_energy = 0.9 * self._dt_from_energy + 0.1 * dt_from_energy

                # Step 1: base DT confidence
                if self.config.enable_dtd:
                    raw_dt = self.get_dtd_confidence()
                else:
                    raw_dt = 1.0 - far_pwr / (mic_pwr + far_pwr)

                # Stationary DT macro detection (sets flag only, does NOT override raw_dt)
                is_stationary_dt = False
                if self._is_stationary_far and self._filter_converged:
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
                if (self._filter_converged and not self._is_stationary_far
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

                # Divergence detection: monitor after convergence
                if self._filter_converged and self.near_power > 1e-8:
                    inst_erle_linear = self.near_power / (self.raw_error_power + 1e-10)
                    is_diverged = float(inst_erle_linear < 0.63)  # ERLE < -2dB
                    self._divergence_indicator = (0.9 * self._divergence_indicator
                                                  + 0.1 * is_diverged)
                else:
                    self._divergence_indicator *= 0.95

                if self.res:
                    # Change D: during EPC render-forced window, force RES
                    # into render-based echo estimate (unreliable filter W).
                    if getattr(self, '_epc_render_forced_remaining', 0) > 0:
                        self._epc_render_forced_remaining -= 1
                        self.res._using_render_based = True
                    self.res.over_sub = effective_over_sub
                    final_output = self.res.process(raw_output, self.filter.echo_spec,
                                                    far_power, self.filter.far_spec,
                                                    filter_converged=self._filter_converged,
                                                    erle_factor=erle_factor,
                                                    dt_indicator=dt_indicator,
                                                    near_spec=self.filter.near_spec,
                                                    divergence=self._divergence_indicator,
                                                    is_stationary_dt=is_stationary_dt,
                                                    saturation_level=self._saturation_level,
                                                    epc_active=self.epc_active,
                                                    shadow_dt=(0.08 * max(float(self._dt_from_energy),
                                                                        float(getattr(self, '_dt_from_shadow', 0.0)))
                                                               if self.epc_active
                                                               else max(float(self._dt_from_energy),
                                                                        float(getattr(self, '_dt_from_shadow', 0.0)))),
                                                    erl_estimate=self._erl_estimate)

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
        if not self._filter_converged and self.near_power > 1e-8 and self._warmup_frames <= 0:
            inst_erle = 10 * np.log10(self.near_power / (self.raw_error_power + 1e-10))
            far_active = float(np.mean(far_end ** 2)) > 1e-4
            if far_active:
                if inst_erle > 5.0:
                    self._conv_counter += 1
                else:
                    self._conv_counter = 0
                if self._conv_counter >= 10:
                    self._filter_converged = True
                    # Switch to low Q: stable tracking mode
                    for filt in [self.filter, self.shadow_filter]:
                        if filt and hasattr(filt, 'Q_low'):
                            filt.Q = filt.Q_low.copy()
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
        config.sample_rate = mic_sr

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
    parser.add_argument('--preset', choices=['mild', 'balanced', 'aggressive', 'maximum'],
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
