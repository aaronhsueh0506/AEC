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
    enable_dtd: bool = True
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
    res_g_min_db: float = -20.0
    res_over_sub: float = 1.5
    res_alpha: float = 0.8

    # Shadow filter (dual-filter divergence control, FREQ/SUBBAND only)
    enable_shadow: bool = False
    shadow_mu_ratio: float = 0.5
    shadow_copy_threshold: float = 0.8
    shadow_err_alpha: float = 0.95
    shadow_dtd_mu_min: float = 0.2      # #1: Shadow DTD floor (20% vs main's 5%)
    shadow_copy_hysteresis: int = 3     # #5: Consecutive frames needed for copy

    # Coherence DTD absolute energy floor
    dtd_coh_abs_floor: float = 1e-6     # #8: Absolute error energy floor

    # Echo path change detection (requires shadow filter)
    epc_delta_threshold: float = 0.3    # |ΔE/total_E| < threshold → echo change
    epc_total_rise: float = 1.5         # total_err > prev × rise → errors increasing
    epc_hangover: int = 20              # keep EPC active for N frames after detection
    epc_mu_floor: float = 0.5           # mu_scale floor during EPC

    # Mode
    mode: AecMode = AecMode.NLMS

    # TIME/LMS history control
    clear_filter_history: bool = False  # Clear ref_buffer each block (default: keep 1 hop history)

    @property
    def fft_size(self) -> int:
        # Next power of 2 >= frame_size (= frame_size when frame_size is power of 2)
        n = self.frame_size
        return 1 << (n - 1).bit_length()


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
    Frequency-domain NLMS (Partitioned Block FDAF)

    More efficient than time-domain for long echo paths.
    Uses overlap-save method for linear convolution.
    """

    def __init__(self, block_size: int, n_partitions: int,
                 mu: float = 0.3, delta: float = 1e-8):
        self.block_size = block_size
        self.hop_size = block_size // 2
        self.n_partitions = n_partitions
        self.n_freqs = block_size // 2 + 1
        self.mu = mu
        self.delta = delta
        self.alpha_power = 0.9

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

    def reset(self):
        self.W.fill(0)
        self.X_buf.fill(0)
        self.near_buffer.fill(0)
        self.far_buffer.fill(0)
        self.power.fill(0)
        self.partition_idx = 0

    def process(self, near_end: np.ndarray, far_end: np.ndarray,
                mu_scale: float = 1.0) -> np.ndarray:
        """Process hop_size samples. mu_scale in [0,1] controls adaptation rate."""
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

        # Update weights (skip when reference power too low)
        total_power = np.sum(self.power)
        if mu_scale > 0 and total_power > self.delta * self.n_freqs:
            # Per-bin normalization with power floor to prevent divergence
            power_floor = np.maximum(self.power, np.max(self.power) * 1e-4)
            mu_eff = (self.mu * mu_scale) / (power_floor * self.n_partitions + self.delta)
            for p in range(self.n_partitions):
                p_idx = (curr_p - p) % self.n_partitions
                grad = self.error_spec * np.conj(self.X_buf[p_idx])
                self.W[p] += mu_eff * grad

                # Constraint: time-domain truncation
                w_time = np.fft.irfft(self.W[p], self.block_size)
                w_time[hop:] = 0
                self.W[p] = np.fft.rfft(w_time)

        self.partition_idx = (self.partition_idx + 1) % self.n_partitions
        return output.astype(np.float32)

    def get_error_energy(self) -> float:
        return float(np.sum(np.abs(self.error_spec) ** 2))

    def copy_weights_from(self, src: 'SubbandNlms'):
        self.W[:] = src.W


class ResFilter:
    """
    Residual Echo Suppressor (Post-Filter)

    Uses Echo-to-Error Ratio (EER) based spectral suppression.
    """

    def __init__(self, n_freqs: int, g_min_db: float = -20.0,
                 over_sub: float = 1.5, alpha: float = 0.8):
        self.n_freqs = n_freqs
        self.g_min = 10 ** (g_min_db / 20)
        self.over_sub = over_sub
        self.alpha = alpha
        self.alpha_psd = 0.9

        self.gain = np.ones(n_freqs, dtype=np.float32)
        self.gain_smooth = np.ones(n_freqs, dtype=np.float32)
        self.echo_psd = np.zeros(n_freqs, dtype=np.float32)
        self.error_psd = np.zeros(n_freqs, dtype=np.float32)

    def reset(self):
        self.gain.fill(1)
        self.gain_smooth.fill(1)
        self.echo_psd.fill(0)
        self.error_psd.fill(0)

    def process(self, error_spec: np.ndarray, echo_spec: np.ndarray,
                far_power: float) -> np.ndarray:
        """Process spectrum and return enhanced output spectrum"""
        # Compute power spectra
        echo_pwr = np.abs(echo_spec) ** 2
        error_pwr = np.abs(error_spec) ** 2

        # Smooth PSD
        self.echo_psd = self.alpha_psd * self.echo_psd + (1 - self.alpha_psd) * echo_pwr
        self.error_psd = self.alpha_psd * self.error_psd + (1 - self.alpha_psd) * error_pwr

        # EER
        eps = 1e-10
        eer = self.echo_psd / (self.error_psd + eps)

        # Gain
        g = 1.0 / (1.0 + self.over_sub * eer)
        g = np.maximum(g, self.g_min)

        # Release when far-end inactive
        if far_power < 1e-6:
            g = np.ones_like(g)

        self.gain = g

        # Smooth (asymmetric)
        alpha_g = np.where(g < self.gain_smooth, 0.3, self.alpha)
        self.gain_smooth = alpha_g * self.gain_smooth + (1 - alpha_g) * g

        return self.gain_smooth * error_spec


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
        AecMode.FREQ: 0.2,
        AecMode.SUBBAND: 0.5,
    }

    def __init__(self, config: Optional[AecConfig] = None):
        self.config = config or AecConfig()

        # Apply per-mode default mu if user didn't override
        if self.config.mu == AecConfig.mu:  # still at dataclass default
            self.config.mu = self._MODE_DEFAULT_MU.get(self.config.mode, 0.3)

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
                delta=self.config.delta
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
                n_freqs=self.filter.n_freqs,
                g_min_db=self.config.res_g_min_db,
                over_sub=self.config.res_over_sub,
                alpha=self.config.res_alpha
            )
        else:
            self.res = None

        # Shadow filter (dual-filter, FREQ/SUBBAND only, requires DTD)
        self.shadow_filter = None
        self.shadow_output = None
        self.main_err_smooth = 0.0
        self.shadow_err_smooth = 0.0
        if (self.config.enable_shadow and
                self.config.mode in (AecMode.FREQ, AecMode.SUBBAND)):
            if not self.config.enable_dtd:
                import warnings
                warnings.warn("Shadow filter disabled: requires DTD for safe operation "
                              "(--no-dtd disables DTD, shadow loses its safety reference)")
                self.config.enable_shadow = False
            elif hasattr(self.filter, 'W'):
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

        # #5: Copy hysteresis counter
        self.shadow_copy_counter = 0

        # ERLE
        self.near_power = 0.0
        self.error_power = 0.0
        self.alpha = 0.95

        # DTD confidence history (one entry per process() call)
        self.confidence_history = []

    def reset(self):
        self.filter.reset()
        if self.shadow_filter:
            self.shadow_filter.reset()
            self.main_err_smooth = 0.0
            self.shadow_err_smooth = 0.0
        self.prev_total_err = 0.0
        self.epc_active = False
        self.epc_hangover_count = 0
        self.prev_dtd_conf = 0.0
        self.shadow_copy_counter = 0
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
        mu_scale = 1.0 - conf * (1.0 - min_r)

        # Echo path change: keep mu high so filter can adapt to new path
        if self.epc_active:
            mu_scale = max(mu_scale, self.config.epc_mu_floor)

        return mu_scale

    def process(self, near_end: np.ndarray, far_end: np.ndarray) -> np.ndarray:
        # DTD: dual detector (divergence + coherence) for FREQ/SUBBAND
        # Combined confidence = max(divergence, coherence) → mu_scale

        mu_scale = self._compute_mu_scale()

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
                output = self._freq_out_queue[r:r+hop].copy()
                self._freq_out_read = r + hop
            else:
                output = self.filter.process(near_end, far_end, mu_scale)

            # Shadow filter with DTD protection (#1) and bidirectional copy (#6)
            if self.shadow_filter is not None and self._freq_near_queue is None:
                # #1: Shadow also receives DTD protection, but more lenient
                conf = self.prev_dtd_conf  # use same conf from _compute_mu_scale
                shadow_mu_scale = 1.0 - conf * (1.0 - self.config.shadow_dtd_mu_min)
                shadow_out = self.shadow_filter.process(near_end, far_end, shadow_mu_scale)

                main_err = self.filter.get_error_energy()
                shadow_err = self.shadow_filter.get_error_energy()

                alpha_s = self.config.shadow_err_alpha
                self.main_err_smooth = alpha_s * self.main_err_smooth + (1 - alpha_s) * main_err
                self.shadow_err_smooth = alpha_s * self.shadow_err_smooth + (1 - alpha_s) * shadow_err

                threshold = self.config.shadow_copy_threshold
                # #5: Copy hysteresis — require N consecutive frames
                if self.shadow_err_smooth < self.main_err_smooth * threshold:
                    self.shadow_copy_counter += 1
                else:
                    self.shadow_copy_counter = 0

                if self.shadow_copy_counter >= self.config.shadow_copy_hysteresis:
                    # Shadow → Main copy
                    self.filter.copy_weights_from(self.shadow_filter)
                    self.filter.echo_spec[:] = self.shadow_filter.echo_spec
                    output = shadow_out
                    self.main_err_smooth = self.shadow_err_smooth
                    self.shadow_copy_counter = 0
                # #6: Bidirectional — Main → Shadow when main is clearly better
                elif self.main_err_smooth < self.shadow_err_smooth * threshold:
                    self.shadow_filter.copy_weights_from(self.filter)
                    self.shadow_err_smooth = self.main_err_smooth

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

            # RES post-filter using overlap-save (skip for buffered FDAF)
            # Apply RES gain to full-block residual spectrum (near_spec - echo_spec),
            # then IFFT and take last hop_size samples. This inherits the overlap-save
            # framework from PBFDAF, avoiding block boundary artifacts.
            if self.res and self._freq_near_queue is None:
                far_power = np.mean(far_end ** 2)
                residual_spec = self.filter.near_spec - self.filter.echo_spec
                enhanced_spec = self.res.process(residual_spec,
                                                  self.filter.echo_spec, far_power)
                enhanced_time = np.fft.irfft(enhanced_spec, self.filter.block_size)
                output = enhanced_time[self.filter.hop_size:].astype(np.float32)

            # Update DTD detectors for NEXT block
            if self.dtd_divergence:
                self.dtd_divergence.detect_block(near_end, far_end, output=output)
            if self.dtd_coherence:
                if self._dtd_fft_size > 0:
                    # FREQ buffered: accumulate into DTD buffer, run at hop=FL/2
                    hop = self._hop_size
                    pos = self._dtd_acc_pos
                    self._dtd_acc_err[pos:pos+hop] = output
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
            # LMS/NLMS: no DTD (Output Limiter provides safety net)
            output, echo_est = self.filter.process_block(near_end, far_end)

        # Output limiter: output should never exceed mic amplitude.
        # If echo_est is correct, output = mic - echo ≤ mic. Exceeding
        # means filter weights are wrong — scale down to prevent artifacts.
        near_peak = np.max(np.abs(near_end))
        out_peak = np.max(np.abs(output))
        if out_peak > near_peak > 1e-6:
            output *= near_peak / out_peak

        # ERLE
        for i in range(len(near_end)):
            self.near_power = self.alpha * self.near_power + (1 - self.alpha) * near_end[i] ** 2
            self.error_power = self.alpha * self.error_power + (1 - self.alpha) * output[i] ** 2

        # Record DTD confidence for plotting
        self.confidence_history.append(self.get_dtd_confidence())

        return output.astype(np.float32)

    def get_erle(self) -> float:
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
    parser.add_argument('--no-dtd', action='store_true', help='Disable DTD')
    parser.add_argument('--enable-res', action='store_true', help='Enable RES post-filter')
    parser.add_argument('--res-g-min', type=float, default=-20.0, help='RES min gain (dB)')
    parser.add_argument('--enable-shadow', action='store_true', help='Enable shadow filter (dual-filter, FREQ/SUBBAND, requires DTD)')
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
        enable_dtd=not args.no_dtd,
        enable_res=args.enable_res,
        res_g_min_db=args.res_g_min,
        enable_shadow=args.enable_shadow,
        clear_filter_history=args.clear_history
    )

    process_wav_files(args.mic, args.ref, args.output, config)


if __name__ == '__main__':
    main()
