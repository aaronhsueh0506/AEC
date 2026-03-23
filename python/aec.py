"""
Acoustic Echo Cancellation (AEC) - Python Reference Implementation

Supports three filter modes:
- Time-domain NLMS (--mode nlms): sample-by-sample, lowest latency
- Frequency-domain NLMS (--mode freq): single FFT block, no partitions
- Partitioned FDAF (--mode subband): multiple partitions, for long echo paths

Additional features:
- Double-Talk Detection (DTD)
- Residual Echo Suppressor (RES)

Usage:
    python aec.py mic.wav ref.wav output.wav [--mode nlms|freq|subband] [--enable-res]
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Optional, Tuple
from enum import Enum
import argparse
import soundfile as sf


class AecMode(Enum):
    LMS = "lms"         # Time-domain LMS (no normalization, simplest)
    NLMS = "nlms"       # Time-domain NLMS (sample-by-sample)
    FREQ = "freq"       # Frequency-domain NLMS (single block, n_partitions=1)
    SUBBAND = "subband" # Partitioned block FDAF (multiple partitions)


@dataclass
class AecConfig:
    """AEC Configuration (all sizes in samples)"""
    sample_rate: int = 16000
    frame_size: int = 512         # Frame length in samples (512 @ 16kHz)
    hop_size: int = 256           # Hop size in samples (256 @ 16kHz)
    filter_length: int = 512     # Filter length in samples (mode-dependent)
    mu: float = 0.3              # Step size
    delta: float = 1e-8          # Regularization
    leak: float = 0.99999        # Weight leakage (slight leak for double-talk stability)
    use_leakage: bool = False    # Time-domain truncation (leakage causes circular convolution artifacts)
    freq_leakage: float = 0.9999 # Frequency-domain weight leakage coefficient
    enable_dtd: bool = False
    dtd_threshold: float = 2.0   # (legacy, kept for compat) Error-based DTD ratio
    dtd_hangover_frames: int = 15

    # Geigel DTD parameters (LMS/NLMS)
    dtd_geigel_threshold: float = 0.5     # |mic| > thresh × max(|ref|) → double-talk
    dtd_mu_min_ratio: float = 0.05        # During double-talk, mu drops to 5%
    dtd_confidence_attack: float = 0.3    # Confidence ramp-up rate per block
    dtd_confidence_release: float = 0.05  # Confidence ramp-down rate per block

    # Divergence detection parameters (FREQ/SUBBAND, output-vs-input)
    dtd_divergence_factor: float = 1.5    # output > input × factor → diverged

    # Coherence-based DTD parameters (FREQ/SUBBAND, complements divergence)
    dtd_coh_alpha: float = 0.85           # PSD smoothing factor (~6 block time constant)
    dtd_coh_high: float = 0.6            # Coherence above → no DT (correlated error)
    dtd_coh_low: float = 0.3             # Coherence below → DT (uncorrelated error)
    dtd_coh_energy_floor: float = 0.1    # Min error/far energy ratio to trigger DT
    dtd_coh_hangover: int = 3            # Coherence DTD hangover blocks (shorter than Geigel)
    dtd_coh_release: float = 0.1         # Coherence confidence release rate (faster recovery)

    # RES parameters
    enable_res: bool = False
    res_g_min_db: float = -40.0
    res_over_sub: float = 6.0
    res_alpha: float = 0.8
    enable_cng: bool = True            # Comfort noise generation in RES

    # Shadow filter (dual-filter divergence control, FREQ/SUBBAND only)
    enable_shadow: bool = True
    shadow_mu_ratio: float = 1.0
    shadow_copy_threshold: float = 0.7
    shadow_err_alpha: float = 0.85
    shadow_dtd_mu_min: float = 0.2      # #1: Shadow DTD floor (20% vs main's 5%)
    shadow_mu_min: float = 0.5           # Shadow-only mode: DT mu floor (50%)
    shadow_copy_hysteresis: int = 5     # #5: Consecutive frames needed for copy

    # Coherence DTD absolute energy floor
    dtd_coh_abs_floor: float = 1e-6     # #8: Absolute error energy floor

    # FDKF (Frequency Domain Kalman Filter) — faster convergence than NLMS
    use_kalman: bool = False

    # Echo path change detection (requires shadow filter)
    epc_delta_threshold: float = 0.3    # |ΔE/total_E| < threshold → echo change
    epc_total_rise: float = 1.5         # total_err > prev × rise → errors increasing
    epc_hangover: int = 20              # keep EPC active for N frames after detection
    epc_mu_floor: float = 0.5           # mu_scale floor during EPC

    # Delay estimation (GCC-PHAT)
    enable_delay_est: bool = True       # Enable automatic delay estimation + ref alignment
    max_delay_ms: float = 250.0         # Maximum delay to search (ms)
    delay_est_period_s: float = 2.0     # Re-estimate delay every N seconds
    delay_est_init_s: float = 0.5       # Accumulate this much data before first estimate
    fixed_delay_samples: int = -1       # If >= 0, use this fixed delay instead of estimation

    # High-pass filter (DC blocker + low-freq removal)
    enable_highpass: bool = True
    highpass_cutoff_hz: float = 80.0    # Cutoff freq: removes DC, 50/60Hz hum, rumble

    # Saturation / non-linear echo handling
    enable_saturation_detect: bool = True
    saturation_threshold: float = 0.95       # |sample| > threshold → clipping
    saturation_over_sub_boost: float = 3.0   # Extra over_sub during saturation
    saturation_softclip_ref: bool = True     # Soft-clip reference for better filter modeling

    # RES anti-blackout
    res_max_drop_db_per_frame: float = 6.0   # Max gain drop per frame (dB)
    res_max_rise_db_per_frame: float = 6.0   # Max gain rise per frame (dB)
    res_spectral_floor: bool = True          # Spectral-shape-preserving gain floor
    res_spectral_floor_db: float = -25.0     # Floor relative to spectral envelope

    # Mode
    mode: AecMode = AecMode.NLMS

    # TIME/LMS history control
    clear_filter_history: bool = False  # Clear ref_buffer each block (default: keep 1 hop history)

    @property
    def fft_size(self) -> int:
        # Next power of 2 >= frame_size (= frame_size when frame_size is power of 2)
        n = self.frame_size
        return 1 << (n - 1).bit_length()


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

        self.estimated_delay = best_pos
        self._samples_since_est = 0
        self._n_estimates += 1


class NlmsFilter:
    """Time-domain NLMS Adaptive Filter"""

    def __init__(self, filter_length: int, mu: float = 0.3,
                 delta: float = 1e-8, leak: float = 0.9999,
                 normalize: bool = True):
        self.filter_length = filter_length
        self.mu = mu
        self.delta = delta
        self.leak = leak
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
            self.weights = self.leak * self.weights + mu_eff * error * self.ref_buffer

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


class SubbandNlms:
    """
    Frequency-domain Partitioned Block Adaptive Filter

    Supports two adaptation modes:
    - NLMS: classic normalized LMS (mu / power normalization)
    - FDKF: Frequency-Domain Kalman Filter (per-bin Kalman gain, faster convergence)

    Uses overlap-save method for linear convolution.
    """

    def __init__(self, block_size: int, n_partitions: int,
                 mu: float = 0.3, delta: float = 1e-8,
                 use_leakage: bool = False, leakage: float = 0.9999,
                 use_kalman: bool = False):
        self.block_size = block_size
        self.hop_size = block_size // 2
        self.n_partitions = n_partitions
        self.n_freqs = block_size // 2 + 1
        self.mu = mu
        self.delta = delta
        self.alpha_power = 0.9
        self.use_leakage = use_leakage
        self.leakage = leakage
        self.use_kalman = use_kalman

        # Filter weights [n_partitions, n_freqs]
        self.W = np.zeros((n_partitions, self.n_freqs), dtype=np.complex64)

        # Reference spectrum history [n_partitions, n_freqs]
        self.X_buf = np.zeros((n_partitions, self.n_freqs), dtype=np.complex64)
        self.partition_idx = 0

        # Input buffers
        self.near_buffer = np.zeros(block_size, dtype=np.float32)
        self.far_buffer = np.zeros(block_size, dtype=np.float32)

        # Power estimation
        self.power = np.zeros(self.n_freqs, dtype=np.float32)

        # Output spectra (for RES / coherence DTD)
        self.near_spec = np.zeros(self.n_freqs, dtype=np.complex64)
        self.echo_spec = np.zeros(self.n_freqs, dtype=np.complex64)
        self.error_spec = np.zeros(self.n_freqs, dtype=np.complex64)
        self.far_spec = np.zeros(self.n_freqs, dtype=np.complex64)

        # FDKF state: per-partition, per-bin error covariance
        if self.use_kalman:
            # P: error covariance (real, per-partition per-bin)
            self.P = np.ones((n_partitions, self.n_freqs), dtype=np.float32) * 0.5
            # Q: process noise — controls adaptation speed
            self.Q = np.ones(self.n_freqs, dtype=np.float32) * 1e-5
            # R: measurement noise PSD (estimated from error)
            self.R = np.ones(self.n_freqs, dtype=np.float32) * 1e-2
            self._error_psd = np.ones(self.n_freqs, dtype=np.float32) * 1e-2
            self._alpha_r = 0.95  # smoothing for R estimation

    def reset(self):
        self.W.fill(0)
        self.X_buf.fill(0)
        self.near_buffer.fill(0)
        self.far_buffer.fill(0)
        self.power.fill(0)
        self.partition_idx = 0
        if self.use_kalman:
            self.P.fill(0.1)
            self.R.fill(1e-2)
            self._error_psd.fill(1e-2)

    def process(self, near_end: np.ndarray, far_end: np.ndarray,
                mu_scale=1.0) -> np.ndarray:
        """Process hop_size samples. mu_scale: scalar or per-bin array [n_freqs]."""
        hop = self.hop_size

        # Shift buffers (overlap-save)
        self.near_buffer[:hop] = self.near_buffer[hop:]
        self.near_buffer[hop:] = near_end

        self.far_buffer[:hop] = self.far_buffer[hop:]
        self.far_buffer[hop:] = far_end

        # FFT
        near_spec = np.fft.rfft(self.near_buffer)
        far_spec = np.fft.rfft(self.far_buffer)
        self.near_spec = near_spec  # expose for RES overlap-save
        self.far_spec = far_spec  # expose for coherence DTD

        # Store far-end spectrum
        curr_p = self.partition_idx
        self.X_buf[curr_p] = far_spec

        # Update power estimate
        self.power = (self.alpha_power * self.power +
                     (1 - self.alpha_power) * np.abs(far_spec) ** 2)

        # Compute echo estimate
        self.echo_spec.fill(0)
        for p in range(self.n_partitions):
            p_idx = (curr_p - p) % self.n_partitions
            self.echo_spec += self.W[p] * self.X_buf[p_idx]

        # IFFT
        echo_time = np.fft.irfft(self.echo_spec, self.block_size)

        # Error (take last hop_size samples)
        output = self.near_buffer[hop:] - echo_time[hop:]

        # Error spectrum (zero-pad first half)
        error_time = np.zeros(self.block_size, dtype=np.float32)
        error_time[hop:] = output
        self.error_spec = np.fft.rfft(error_time)

        # Update weights
        total_power = np.sum(self.power)
        if total_power > self.delta * self.n_freqs:
            if self.use_kalman:
                self._update_kalman(curr_p, mu_scale)
            else:
                self._update_nlms(curr_p, mu_scale)

        self.partition_idx = (self.partition_idx + 1) % self.n_partitions
        return output.astype(np.float32)

    def _update_nlms(self, curr_p: int, mu_scale):
        """NLMS weight update."""
        mu_scale_arr = np.asarray(mu_scale, dtype=np.float32)
        if mu_scale_arr.ndim == 0:
            mu_scale_arr = np.full(self.n_freqs, float(mu_scale_arr), dtype=np.float32)
        if not np.any(mu_scale_arr > 0):
            return
        global_floor = np.mean(self.power) * 0.01 + self.delta
        power_floor = np.maximum(self.power, global_floor)
        mu_eff = (self.mu * mu_scale_arr) / (power_floor * self.n_partitions + self.delta)
        for p in range(self.n_partitions):
            p_idx = (curr_p - p) % self.n_partitions
            grad = self.error_spec * np.conj(self.X_buf[p_idx])
            self.W[p] += mu_eff * grad
            if self.use_leakage:
                self.W[p] *= self.leakage
            else:
                w_time = np.fft.irfft(self.W[p], self.block_size)
                w_time[self.hop_size:] = 0
                self.W[p] = np.fft.rfft(w_time)

    def _update_kalman(self, curr_p: int, mu_scale):
        """Frequency-Domain Kalman Filter weight update.

        Per-bin Kalman gain replaces fixed step size:
        K = P * X / (X^H * P * X + R)
        W += K * error
        P = P - K * X^H * P + Q

        Advantages over NLMS:
        - Per-bin adaptation rate (bins with strong ref get faster updates)
        - Automatic step size: converges fast initially, slows at steady-state
        - Better handling of colored signals
        """
        mu_scale_arr = np.asarray(mu_scale, dtype=np.float32)
        if mu_scale_arr.ndim == 0:
            mu_scale_arr = np.full(self.n_freqs, float(mu_scale_arr), dtype=np.float32)

        # Update measurement noise estimate from error PSD
        error_psd = np.abs(self.error_spec) ** 2
        self._error_psd = self._alpha_r * self._error_psd + (1 - self._alpha_r) * error_psd
        # R = smoothed error PSD (represents noise + residual echo)
        self.R = np.maximum(self._error_psd, self.delta)

        for p in range(self.n_partitions):
            p_idx = (curr_p - p) % self.n_partitions
            X = self.X_buf[p_idx]
            X_power = np.abs(X) ** 2 + self.delta

            # Kalman gain: K = P * X^* / (|X|^2 * P + R)
            denominator = X_power * self.P[p] + self.R
            K = (self.P[p] * np.conj(X)) / (denominator + self.delta)

            # Apply mu_scale as DT protection
            K *= mu_scale_arr

            # Weight update
            self.W[p] += K * self.error_spec

            # Covariance update: P = (1 - K*X) * P + Q
            KX = np.real(K * X)
            self.P[p] = np.maximum((1.0 - KX) * self.P[p] + self.Q, self.delta)

            # Constraint: time-domain truncation
            if not self.use_leakage:
                w_time = np.fft.irfft(self.W[p], self.block_size)
                w_time[self.hop_size:] = 0
                self.W[p] = np.fft.rfft(w_time)

    def get_error_energy(self) -> float:
        return float(np.sum(np.abs(self.error_spec) ** 2))

    def copy_weights_from(self, src: 'SubbandNlms'):
        self.W[:] = src.W


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
                 spectral_floor_db: float = -25.0):
        self.block_size = block_size
        self.hop_size = block_size // 2
        self.n_freqs = n_freqs
        self.g_min = 10 ** (g_min_db / 20)
        self.over_sub = over_sub
        self.alpha = alpha
        self.alpha_echo_psd = 0.7    # echo PSD: moderate tracking (TC≈53ms), was 0.5
        self.alpha_error_psd = 0.8   # error PSD: moderate TC≈80ms

        self.gain_smooth = np.full(n_freqs, self.g_min, dtype=np.float32)
        self.echo_psd = np.zeros(n_freqs, dtype=np.float32)
        self.error_psd = np.zeros(n_freqs, dtype=np.float32)

        # Coherence-based nonlinear echo PSD estimation
        # Coherence between far-end and error captures both linear and
        # nonlinear echo components (both correlate with far-end).
        self.alpha_coh = 0.3            # Cross-PSD smoothing (TC≈23ms, fast tracking)
        self.S_fe = np.zeros(n_freqs, dtype=np.complex64)  # Cross-PSD far×error
        self.S_ff = np.zeros(n_freqs, dtype=np.float32)    # Far-end PSD
        self.S_ee = np.zeros(n_freqs, dtype=np.float32)    # Error PSD

        # CNG (comfort noise generation)
        self.enable_cng = enable_cng
        self.noise_psd = np.zeros(n_freqs, dtype=np.float32)
        self.alpha_noise = 0.98

        # Far-end activity tracking for dynamic g_min
        self.far_activity = 0.0

        # Anti-blackout: gain rate limiting
        self.max_drop_ratio = 10 ** (max_drop_db_per_frame / 20)  # e.g., 6dB → 1.995
        self.max_rise_ratio = 10 ** (max_rise_db_per_frame / 20)  # e.g., 3dB → 1.413

        # Spectral-shape-preserving floor
        self.enable_spectral_floor = enable_spectral_floor
        self.spectral_floor_ratio = 10 ** (spectral_floor_db / 20)  # e.g., -25dB → 0.056
        self.error_envelope = np.ones(n_freqs, dtype=np.float32)
        self.alpha_envelope = 0.95  # Slow-tracking spectral envelope

        # OLA: sqrt-Hann window + sliding input buffer + overlap buffer
        self.window = np.sqrt(np.hanning(block_size)).astype(np.float32)
        self.input_buf = np.zeros(block_size, dtype=np.float32)
        self.ola_buf = np.zeros(block_size, dtype=np.float32)

    def reset(self):
        self.gain_smooth.fill(self.g_min)
        self.echo_psd.fill(0)
        self.error_psd.fill(0)
        self.S_fe.fill(0)
        self.S_ff.fill(0)
        self.S_ee.fill(0)
        self.noise_psd.fill(0)
        self.far_activity = 0.0
        self.input_buf.fill(0)
        self.ola_buf.fill(0)

    def process(self, error_hop: np.ndarray, echo_spec: np.ndarray,
                far_power: float, far_spec: np.ndarray = None,
                filter_converged: bool = False,
                erle_factor: float = 0.0) -> np.ndarray:
        """Process hop-size error signal, return enhanced hop via OLA.

        far_spec: far-end frequency spectrum (complex), used for coherence-
                  based nonlinear echo PSD estimation.
        """
        hop = self.hop_size

        # Slide in new error samples
        self.input_buf[:hop] = self.input_buf[hop:]
        self.input_buf[hop:] = error_hop

        # Analysis: sqrt-Hann window + FFT
        windowed = self.input_buf * self.window
        spec = np.fft.rfft(windowed)

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
            if not hasattr(self, '_coh2_smooth'):
                self._coh2_smooth = np.zeros(self.n_freqs, dtype=np.float32)
            a_coh = np.where(coh2_raw < self._coh2_smooth, 0.50, 0.80)
            self._coh2_smooth = a_coh * self._coh2_smooth + (1.0 - a_coh) * coh2_raw
            coh2 = self._coh2_smooth
        else:
            if far_power <= 1e-4:
                self.S_fe *= 0.5
                self.S_ff *= 0.5

        # Cold start: skip EMA warmup, initialize PSD directly on first far-end frame
        if far_power > 1e-4 and np.sum(self.echo_psd) < 1e-10:
            self.echo_psd[:] = echo_pwr_linear
            self.error_psd[:] = error_pwr

        # Linear EER from adaptive filter echo estimate
        self.echo_psd = self.alpha_echo_psd * self.echo_psd + (1 - self.alpha_echo_psd) * echo_pwr_linear
        self.error_psd = self.alpha_error_psd * self.error_psd + (1 - self.alpha_error_psd) * error_pwr

        if far_power < 1e-4:
            self.echo_psd *= 0.3  # fast decay during far-end silence
            # Track noise floor for CNG during far-end silence
            if self.enable_cng:
                self.noise_psd = np.minimum(
                    self.alpha_noise * self.noise_psd
                    + (1 - self.alpha_noise) * error_pwr,
                    error_pwr * 2)

        # --- Dynamic g_min: track far-end activity ---
        is_far_active = float(far_power > 1e-4)
        if is_far_active > self.far_activity:
            # Far-end resumes: fast attack (TC≈30ms, ~2 frames)
            self.far_activity = 0.7 * self.far_activity + 0.3 * is_far_active
        else:
            # Far-end stops: slow decay (TC≈800ms, wait for echo_psd to decay first)
            self.far_activity = 0.98 * self.far_activity + 0.02 * is_far_active
        # far_activity=1.0 → g_min normal; far_activity=0.0 → g_min→1.0 (no suppression)
        effective_g_min = self.g_min + (1.0 - self.g_min) * (1.0 - self.far_activity)
        effective_g_min = max(effective_g_min, 10 ** (-60.0 / 20))  # floor at -60dB

        # --- Noise gate: don't suppress quiet segments ---
        signal_floor = np.mean(self.error_psd) * 0.001 + 1e-8
        quiet_mask = ((self.echo_psd < signal_floor)
                      & (self.error_psd < signal_floor))

        # Compute EER with soft convergence switch (no hard transition click)
        eps = 1e-10
        eer_linear = self.echo_psd / (self.error_psd + eps)
        # erle_factor=0 → pre-convergence (use coh2); erle_factor=1 → converged (use linear EER)
        eer_converged = eer_linear * (0.5 + 0.5 * coh2)
        if far_power > 1e-4:
            eer = (1.0 - erle_factor) * coh2 + erle_factor * eer_converged
        else:
            eer = eer_converged

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

        g = np.maximum(1.0 - self.over_sub * eer, spectral_g_min)
        g[quiet_mask] = 1.0  # Noise gate: pass through quiet bins

        # Temporal smoothing: far_activity-driven release (no feedback loop)
        # far_activity high (far-end speaking) → slow release (TC≈200ms)
        # far_activity low (far-end silent) → fast release (TC≈25ms)
        alpha_release = 0.4 + 0.5 * self.far_activity

        # Attack alpha: slow when unconverged (suppress oscillation), fast when converged
        alpha_attack = 0.60 + 0.25 * (1.0 - erle_factor)
        alpha_g = np.where(g < self.gain_smooth, alpha_attack, alpha_release)
        smoothed = alpha_g * self.gain_smooth + (1 - alpha_g) * g

        # --- Gain rate limiting: prevent sudden blackout / pop ---
        # Relax rate limiting when far-end is silent (near-end needs to pass through)
        activity_scale = 0.5 + 0.5 * self.far_activity  # [0.5, 1.0]
        eff_drop = self.max_drop_ratio ** activity_scale  # Less limiting when silent
        eff_rise = self.max_rise_ratio ** (1.0 / activity_scale)  # More permissive when silent
        gain_floor = self.gain_smooth / eff_drop
        gain_ceil = self.gain_smooth * eff_rise
        smoothed = np.maximum(smoothed, gain_floor)
        smoothed = np.minimum(smoothed, gain_ceil)
        # Clamp to valid range
        if isinstance(spectral_g_min, np.ndarray):
            smoothed = np.maximum(smoothed, spectral_g_min)
        else:
            smoothed = np.maximum(smoothed, effective_g_min)
        smoothed = np.minimum(smoothed, 1.0)
        self.gain_smooth = smoothed

        # Apply gain + synthesis sqrt-Hann window + IFFT
        enhanced_spec = self.gain_smooth * spec

        # --- CNG: inject comfort noise to avoid unnatural silence ---
        if self.enable_cng and np.sum(self.noise_psd) > 0 and far_power <= 1e-4:
            suppressed_pwr = (self.gain_smooth ** 2) * error_pwr
            target_pwr = self.noise_psd * 0.5  # Half noise floor level
            deficit = np.maximum(0, target_pwr - suppressed_pwr)
            if np.any(deficit > 0):
                cng_mag = np.sqrt(deficit).astype(np.float32)
                cng_phase = np.random.uniform(
                    -np.pi, np.pi, self.n_freqs).astype(np.float32)
                cng_spec = cng_mag * np.exp(1j * cng_phase)
                enhanced_spec = enhanced_spec + cng_spec.astype(np.complex64)

        enhanced_time = np.fft.irfft(enhanced_spec, self.block_size)
        enhanced_time *= self.window

        # Overlap-add
        self.ola_buf += enhanced_time
        output = self.ola_buf[:hop].copy()
        self.ola_buf[:hop] = self.ola_buf[hop:]
        self.ola_buf[hop:] = 0.0

        return output.astype(np.float32)


class DtdEstimator:
    """Double-Talk Detector with per-mode strategy.

    - 'geigel' mode (LMS/NLMS): Geigel DTD with hangover + confidence
    - 'divergence' mode (FREQ/SUBBAND): Output-vs-input divergence detection
    - 'coherence' mode (FREQ/SUBBAND): Error-reference coherence DT detection
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

    Supports three filter modes:
    - TIME:    Time-domain NLMS (sample-by-sample processing)
    - FREQ:    Frequency-domain NLMS (single FFT block, n_partitions=1)
    - SUBBAND: Partitioned block FDAF (multiple partitions for long echo paths)
    """

    # Per-mode optimal mu defaults (tuned on fileid_0/1/2)
    _MODE_DEFAULT_MU = {
        AecMode.LMS: 0.02,
        AecMode.NLMS: 0.4,
        AecMode.FREQ: 0.3,
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
        if self.config.mode in (AecMode.FREQ, AecMode.SUBBAND):
            if self.config.mode == AecMode.FREQ:
                # True FDAF: single big FFT block, n_partitions=1
                # block_size = next power of 2 >= 2 * filter_length
                desired = 2 * self.config.filter_length
                block_size = 256
                while block_size < desired:
                    block_size *= 2
                n_partitions = 1
                self._internal_hop = block_size // 2
            else:
                # PBFDAF: partitioned block, configurable filter_length
                block_size = self.config.fft_size
                hop_size = block_size // 2
                n_partitions = max(1, (self.config.filter_length + hop_size - 1) // hop_size)
                self._internal_hop = hop_size

            self.filter = SubbandNlms(
                block_size=block_size,
                n_partitions=n_partitions,
                mu=self.config.mu,
                delta=self.config.delta,
                use_leakage=self.config.use_leakage,
                leakage=self.config.freq_leakage,
                use_kalman=self.config.use_kalman
            )
            self._hop_size = self.config.hop_size  # External hop (always 256)
            self._n_partitions = n_partitions

            # FREQ buffering (when internal_hop > external hop)
            if self.config.mode == AecMode.FREQ and self._internal_hop > self._hop_size:
                self._freq_near_queue = np.zeros(self._internal_hop, dtype=np.float32)
                self._freq_far_queue = np.zeros(self._internal_hop, dtype=np.float32)
                self._freq_out_queue = np.zeros(self._internal_hop, dtype=np.float32)
                self._freq_queue_write = 0
                self._freq_out_read = 0
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
                leak=1.0,
                normalize=False
            )
            self.filter.clear_history = self.config.clear_filter_history
            self._hop_size = self.config.hop_size
            self._internal_hop = self.config.hop_size
            self._n_partitions = 0
            self._freq_near_queue = None
        else:
            # TIME: Time-domain NLMS
            # leak=1.0: NLMS has DTD + weight norm constraint for stability,
            # so leak is unnecessary and hurts convergence (75dB → 31dB with 0.99999)
            self.filter = NlmsFilter(
                filter_length=self.config.filter_length,
                mu=self.config.mu,
                delta=self.config.delta,
                leak=1.0,
                normalize=True
            )
            self.filter.clear_history = self.config.clear_filter_history
            self._hop_size = self.config.hop_size
            self._internal_hop = self.config.hop_size
            self._n_partitions = 0
            self._freq_near_queue = None

        # DTD: FREQ/SUBBAND only (divergence + coherence dual detector)
        # LMS/NLMS have no effective DTD — all methods (Geigel, NCC, coherence,
        # VSS-NLMS) either don't work for AEC or cause vicious cycles with slow
        # convergence. Output Limiter provides the safety net instead.
        if self.config.enable_dtd and self.config.mode in (AecMode.FREQ, AecMode.SUBBAND):
            # Warmup: 50 DTD invocations before coherence starts.
            # FREQ DTD runs every dtd_hop/hop_size external frames,
            # so 50 DTD invocations = 50 * dtd_hop/hop_size external frames.
            warmup = 50
            self.dtd_divergence = DtdEstimator(
                mode='divergence',
                divergence_factor=self.config.dtd_divergence_factor,
                attack=self.config.dtd_confidence_attack,
                release=self.config.dtd_confidence_release,
                warmup_frames=warmup,
            )
            # FREQ: FL-point FFT (matches filter length, hop=FL/2)
            # SUBBAND: use FDAF's own spectra (block_size from filter)
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
        if self.config.enable_res and self.config.mode in (AecMode.FREQ, AecMode.SUBBAND):
            self.res = ResFilter(
                block_size=self.filter.block_size,
                n_freqs=self.filter.n_freqs,
                g_min_db=self.config.res_g_min_db,
                over_sub=self.config.res_over_sub,
                alpha=self.config.res_alpha,
                enable_cng=self.config.enable_cng,
                max_drop_db_per_frame=self.config.res_max_drop_db_per_frame,
                max_rise_db_per_frame=self.config.res_max_rise_db_per_frame,
                enable_spectral_floor=self.config.res_spectral_floor,
                spectral_floor_db=self.config.res_spectral_floor_db
            )
        else:
            self.res = None

        # Shadow filter (dual-filter, FREQ/SUBBAND only)
        # Can be used alone (≈ WebRTC/SpeexDSP) or with DTD (dual protection)
        self.shadow_filter = None
        self.shadow_output = None
        self.main_err_smooth = 0.0
        self.shadow_err_smooth = 0.0
        if (self.config.enable_shadow and
                self.config.mode in (AecMode.FREQ, AecMode.SUBBAND)
                and hasattr(self.filter, 'W')):
            shadow_mu = self.config.mu * self.config.shadow_mu_ratio
            self.shadow_filter = SubbandNlms(
                block_size=self.filter.block_size,
                n_partitions=self.filter.n_partitions,
                mu=shadow_mu,
                delta=self.config.delta
            )

        # Echo path change detection state
        self.prev_total_err = 0.0
        self.epc_active = False
        self.epc_hangover_count = 0

        # #4: Confidence memory decay
        self.prev_dtd_conf = 0.0

        # Convergence state: prevent divergence DTD and allow higher mu_min
        # until filter has demonstrated basic echo cancellation (ERLE > 3 dB)
        self._filter_converged = False

        # Per-bin mu_scale (updated from RES echo_psd/error_psd each frame)
        self._per_bin_mu_scale = None  # None = use scalar fallback

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

        # Simple variable mu (for non-DTD modes, inspired by Valin 2007 RER)
        self._simple_mu_ratio = 1.0
        self._simple_mu_holdoff = 0  # holdoff counter: blocks release for N frames

        # #5: Copy hysteresis counter
        self.shadow_copy_counter = 0
        self.shadow_frame_count = 0  # warm-up counter for shadow copy

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
        self.confidence_history = []

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
        self._simple_mu_ratio = 1.0
        self._simple_mu_holdoff = 0
        self.shadow_copy_counter = 0
        self.shadow_frame_count = 0
        if self.dtd_divergence:
            self.dtd_divergence.reset()
        if self.dtd_coherence:
            self.dtd_coherence.reset()
        if self.res:
            self.res.reset()
        if self._freq_near_queue is not None:
            self._freq_near_queue.fill(0)
            self._freq_far_queue.fill(0)
            self._freq_out_queue.fill(0)
            self._freq_queue_write = 0
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
        if not self._filter_converged:
            mu_min = max(mu_min, 0.5)   # Pre-convergence: floor 0.5, ratio modulates
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
        ratio = min(far_power / error_power, 1.0)

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
                if new_delay >= 0:
                    if self._current_delay < 0:
                        self._current_delay = new_delay
                    elif abs(new_delay - self._current_delay) > 32:
                        self._current_delay = new_delay

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

        # DTD: dual detector (divergence + coherence) for FREQ/SUBBAND
        # Combined confidence = max(divergence, coherence) → mu_scale
        # Non-DTD: simple variable mu (Valin 2007 RER-inspired)

        if self.config.enable_dtd:
            mu_scale = self._compute_mu_scale()
        else:
            mu_scale = self._get_simple_mu_scale()

        if self.config.mode in (AecMode.FREQ, AecMode.SUBBAND):
            if self._freq_near_queue is not None:
                # Buffered FDAF: accumulate into queue
                hop = self._hop_size
                ihop = self._internal_hop
                w = self._freq_queue_write
                self._freq_near_queue[w:w+hop] = near_end
                self._freq_far_queue[w:w+hop] = far_end
                self._freq_queue_write = w + hop

                if self._freq_queue_write >= ihop:
                    # Buffer full — run one big FDAF
                    big_out = self.filter.process(
                        self._freq_near_queue, self._freq_far_queue, mu_scale)
                    self._freq_out_queue[:] = big_out
                    self._freq_queue_write = 0
                    self._freq_out_read = 0

                r = self._freq_out_read
                raw_output = self._freq_out_queue[r:r+hop].copy()
                self._freq_out_read = r + hop
            else:
                raw_output = self.filter.process(near_end, far_end, mu_scale)

            # Shadow filter with DTD protection (#1) and bidirectional copy (#6)
            if self.shadow_filter is not None and self._freq_near_queue is None:
                self.shadow_frame_count += 1
                # Shadow mu: always full mu (AEC3 background filter style)
                shadow_mu_scale = 1.0
                self.shadow_filter.process(near_end, far_end, shadow_mu_scale)

                main_err = self.filter.get_error_energy()
                shadow_err = self.shadow_filter.get_error_energy()

                alpha_s = self.config.shadow_err_alpha
                self.main_err_smooth = alpha_s * self.main_err_smooth + (1 - alpha_s) * main_err
                self.shadow_err_smooth = alpha_s * self.shadow_err_smooth + (1 - alpha_s) * shadow_err

                # Copy gate: only during far-end active + non-DT + no echo path change
                if self.shadow_frame_count >= 50:
                    threshold = self.config.shadow_copy_threshold
                    far_active = np.mean(far_end ** 2) > 1e-4
                    if self.config.enable_dtd:
                        not_dt = self.get_dtd_confidence() < 0.3
                    else:
                        # Use raw far/error ratio as DT indicator
                        raw_err_pwr = np.mean(raw_output ** 2) + 1e-10
                        far_pwr_now = np.mean(far_end ** 2) + 1e-10
                        not_dt = far_pwr_now / raw_err_pwr > 0.3

                    copy_allowed = far_active and not_dt and not self.epc_active

                    if copy_allowed:
                        if self.shadow_err_smooth < self.main_err_smooth * threshold:
                            self.shadow_copy_counter += 1
                        else:
                            self.shadow_copy_counter = 0

                        if self.shadow_copy_counter >= self.config.shadow_copy_hysteresis:
                            # Shadow → Main copy (don't copy echo_spec to avoid output discontinuity)
                            self.filter.copy_weights_from(self.shadow_filter)
                            self.main_err_smooth = self.shadow_err_smooth
                            self.shadow_copy_counter = 0
                        elif self.main_err_smooth < self.shadow_err_smooth * threshold and not_dt:
                            # Bidirectional: Main → Shadow
                            self.shadow_filter.copy_weights_from(self.filter)
                            self.shadow_err_smooth = self.main_err_smooth
                    else:
                        self.shadow_copy_counter = 0

            # Echo path change detection (shadow-based)
            # DT: refined error ↑, shadow stable → ΔE/total large
            # Echo change: both errors ↑ → ΔE/total small
            # When echo change detected, suppress coherence confidence
            # so filter can adapt to new echo path quickly.
            if self.shadow_filter is not None and self.dtd_coherence:
                total_err = self.main_err_smooth + self.shadow_err_smooth
                if total_err > 1e-10:
                    delta_ratio = abs(self.main_err_smooth - self.shadow_err_smooth) / total_err
                else:
                    delta_ratio = 0.0

                errors_rising = (total_err > self.prev_total_err * self.config.epc_total_rise
                                 and self.prev_total_err > 1e-10)
                is_echo_change = errors_rising and delta_ratio < self.config.epc_delta_threshold
                self.prev_total_err = total_err

                if is_echo_change:
                    self.dtd_coherence.confidence *= 0.3
                    self.epc_hangover_count = self.config.epc_hangover
                    self.epc_active = True
                elif self.epc_hangover_count > 0:
                    self.epc_hangover_count -= 1
                    self.epc_active = True
                else:
                    self.epc_active = False

            # final_output starts from raw_output; RES modifies final_output only
            final_output = raw_output.copy()

            # RES post-filter using OLA + sqrt-Hann (skip for buffered FDAF)
            if self.res and self._freq_near_queue is None:
                far_power = np.mean(far_end ** 2)
                # Dynamic over_sub: always aggressive, scale with config
                inst_erle = self.get_erle_instant()
                erle_factor = np.clip((inst_erle - 3.0) / 15.0, 0.0, 1.0)
                # Base: use config over_sub as minimum, scale up with convergence
                base_over_sub = self.config.res_over_sub + 2.0 * erle_factor
                # Saturation boost: non-linear echo needs more suppression
                base_over_sub += self._saturation_level * self.config.saturation_over_sub_boost
                # DT protection: reduce suppression when near-end is active
                dtd_conf = self.get_dtd_confidence()  # 0 if DTD disabled
                dt_reduction = 1.5 * dtd_conf
                self.res.over_sub = max(base_over_sub - dt_reduction, 1.0)
                final_output = self.res.process(raw_output, self.filter.echo_spec,
                                                far_power, self.filter.far_spec,
                                                filter_converged=self._filter_converged,
                                                erle_factor=erle_factor)

                # Update per-bin mu_scale AFTER RES (echo_psd is now current frame)
                if not self.config.enable_dtd:
                    if self._filter_converged:
                        per_bin_eer = self.res.echo_psd / (self.res.error_psd + 1e-10)
                        per_bin_eer = np.clip(per_bin_eer, 0.0, 1.0)
                        mu_min = self.config.shadow_mu_min
                        self._per_bin_mu_scale = (mu_min + (1.0 - mu_min) * per_bin_eer).astype(np.float32)
                        self._simple_mu_ratio = float(np.mean(per_bin_eer))
                    else:
                        # Pre-convergence: no per_bin, let ratio track DT naturally
                        self._per_bin_mu_scale = None
                        self._update_simple_mu_ratio(raw_output, far_end)

            # Update DTD detectors for NEXT block
            # Skip divergence detector before convergence (output>mic is normal
            # when filter hasn't learned echo path yet, not actual divergence)
            if self.dtd_divergence and self._filter_converged:
                self.dtd_divergence.detect_block(near_end, far_end, output=raw_output)
            if self.dtd_coherence and self._filter_converged:
                if self._dtd_fft_size > 0:
                    # FREQ buffered: accumulate into DTD buffer, run at hop=FL/2
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
                    # SUBBAND: use FDAF's spectra directly (every frame)
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

        # Output noise gate: suppress very low-level residual (far-end only)
        out_power = np.mean(final_output ** 2)
        near_power_inst = np.mean(near_end ** 2)
        far_power_inst = np.mean(far_end ** 2)
        if far_power_inst > 1e-4 and near_power_inst > 1e-8:
            snr = out_power / near_power_inst
            if snr < 0.01:  # final_output < -20dB of mic
                final_output *= snr / 0.01  # soft fade

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

        # Convergence detection: 10 consecutive frames with ERLE > 6 dB
        if not self._filter_converged and self.near_power > 1e-8:
            inst_erle = 10 * np.log10(self.near_power / (self.raw_error_power + 1e-10))
            if inst_erle > 6.0:
                self._conv_counter += 1
            else:
                self._conv_counter = 0
            if self._conv_counter >= 10:
                self._filter_converged = True

        # Record DTD confidence for plotting
        self.confidence_history.append(self.get_dtd_confidence())

        return final_output.astype(np.float32)

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
                      config: Optional[AecConfig] = None):
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

        if processed % mic_sr == 0:
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
    freq    - Frequency-domain NLMS (single FFT block, no partitions)
    subband - Partitioned block FDAF (for long echo paths, fastest convergence)

Examples:
    python aec.py mic.wav ref.wav output.wav
    python aec.py mic.wav ref.wav output.wav --mode freq
    python aec.py mic.wav ref.wav output.wav --mode subband --enable-res
    python aec.py mic.wav ref.wav output.wav --mu 0.5 --filter 1024
        """
    )
    parser.add_argument('mic', help='Microphone input WAV file')
    parser.add_argument('ref', help='Reference/loudspeaker WAV file')
    parser.add_argument('output', help='Output WAV file')
    parser.add_argument('--mu', type=float, default=0.3, help='Step size (default: 0.3)')
    parser.add_argument('--filter', type=int, default=0,
                        help='Filter length in samples (default: mode-dependent)')
    parser.add_argument('--mode', choices=['lms', 'nlms', 'freq', 'subband'], default='nlms',
                        help='Filter mode (default: nlms)')
    parser.add_argument('--enable-dtd', action='store_true',
                        help='Enable DTD (default: off, shadow filter provides DT protection)')
    parser.add_argument('--enable-res', action='store_true', help='Enable RES post-filter')
    parser.add_argument('--res-g-min', type=float, default=-20.0, help='RES min gain (dB)')
    parser.add_argument('--no-cng', action='store_true', help='Disable comfort noise generation in RES')
    parser.add_argument('--no-shadow', action='store_true', help='Disable shadow filter')
    parser.add_argument('--no-highpass', action='store_true', help='Disable high-pass filter')
    parser.add_argument('--highpass-cutoff', type=float, default=80.0,
                        help='High-pass filter cutoff frequency in Hz (default: 80)')
    parser.add_argument('--no-saturation-detect', action='store_true',
                        help='Disable saturation/clipping detection')
    parser.add_argument('--clear-history', action='store_true',
                        help='Clear TIME/LMS buffer each block (no carry-over)')

    args = parser.parse_args()

    # Map mode string to enum
    mode_map = {
        'lms': AecMode.LMS,
        'nlms': AecMode.NLMS,
        'freq': AecMode.FREQ,
        'subband': AecMode.SUBBAND
    }

    aec_mode = mode_map[args.mode]

    # Mode-dependent default step size
    mu = args.mu
    if args.mode == 'lms' and args.mu == 0.3:
        mu = 0.01  # LMS needs much smaller step size
    elif args.mode == 'freq' and args.mu == 0.3:
        mu = 0.1   # FREQ single-block: smaller mu to avoid overshoot

    # Mode-dependent filter_length default
    filter_length = args.filter
    if filter_length == 0:
        if aec_mode == AecMode.SUBBAND:
            filter_length = 1024  # 4 partitions
        else:
            filter_length = 512   # frame_size

    config = AecConfig(
        mu=mu,
        filter_length=filter_length,
        mode=aec_mode,
        enable_dtd=args.enable_dtd,
        enable_res=args.enable_res,
        res_g_min_db=args.res_g_min,
        enable_cng=not args.no_cng,
        enable_shadow=not args.no_shadow,
        enable_highpass=not args.no_highpass,
        highpass_cutoff_hz=args.highpass_cutoff,
        enable_saturation_detect=not args.no_saturation_detect,
        clear_filter_history=args.clear_history
    )

    process_wav_files(args.mic, args.ref, args.output, config)


if __name__ == '__main__':
    main()
