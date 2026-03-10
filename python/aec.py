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
    leak: float = 0.9999         # Weight leakage
    enable_dtd: bool = True
    dtd_threshold: float = 2.0   # Error-based DTD: error_energy/echo_energy ratio
    dtd_hangover_frames: int = 15

    # RES parameters
    enable_res: bool = False
    res_g_min_db: float = -20.0
    res_over_sub: float = 1.5
    res_alpha: float = 0.8

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
        self.max_w_norm = 4.0  # Weight norm constraint (prevents explosion during double-talk)

    def reset(self):
        self.weights.fill(0)
        self.ref_buffer.fill(0)
        self.power_sum = 0.0

    def process_sample(self, near_end: float, far_end: float,
                       update_weights: bool = True) -> Tuple[float, float]:
        oldest = self.ref_buffer[-1]
        self.power_sum = max(0, self.power_sum - oldest * oldest + far_end * far_end)
        self.ref_buffer[1:] = self.ref_buffer[:-1]
        self.ref_buffer[0] = far_end
        echo_est = np.dot(self.weights, self.ref_buffer)
        error = near_end - echo_est

        if update_weights and self.power_sum > self.delta * self.filter_length:
            if self.normalize:
                mu_eff = self.mu / (self.power_sum + self.delta)
            else:
                mu_eff = self.mu
            self.weights = self.leak * self.weights + mu_eff * error * self.ref_buffer

        return error, echo_est

    def process_block(self, near_end: np.ndarray, far_end: np.ndarray,
                      update_weights: bool = True) -> Tuple[np.ndarray, np.ndarray]:
        # Optionally clear history (no carry-over between blocks)
        if self.clear_history:
            self.ref_buffer.fill(0)
            self.power_sum = 0.0

        n = len(near_end)
        output = np.zeros(n, dtype=np.float32)
        echo_est = np.zeros(n, dtype=np.float32)
        for i in range(n):
            output[i], echo_est[i] = self.process_sample(
                near_end[i], far_end[i], update_weights)

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

        # Output spectra (for RES)
        self.echo_spec = np.zeros(self.n_freqs, dtype=np.complex64)
        self.error_spec = np.zeros(self.n_freqs, dtype=np.complex64)

    def reset(self):
        self.W.fill(0)
        self.X_buf.fill(0)
        self.near_buffer.fill(0)
        self.far_buffer.fill(0)
        self.power.fill(0)
        self.partition_idx = 0

    def process(self, near_end: np.ndarray, far_end: np.ndarray,
                update_weights: bool = True) -> np.ndarray:
        """Process hop_size samples"""
        hop = self.hop_size

        # Shift buffers (overlap-save)
        self.near_buffer[:hop] = self.near_buffer[hop:]
        self.near_buffer[hop:] = near_end

        self.far_buffer[:hop] = self.far_buffer[hop:]
        self.far_buffer[hop:] = far_end

        # FFT
        near_spec = np.fft.rfft(self.near_buffer)
        far_spec = np.fft.rfft(self.far_buffer)

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
        if update_weights and total_power > self.delta * self.n_freqs:
            # Per-bin normalization with power floor to prevent divergence
            power_floor = np.maximum(self.power, np.max(self.power) * 1e-4)
            mu_eff = self.mu / (power_floor * self.n_partitions + self.delta)
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
    """Double-Talk Detector"""

    def __init__(self, window_length: int, threshold: float = 0.6,
                 hangover_frames: int = 15):
        self.threshold = threshold
        self.hangover_max = hangover_frames
        self.current_far_max = 0.0
        self.dtd_active = False
        self.hangover_count = 0

    def reset(self):
        self.current_far_max = 0.0
        self.dtd_active = False
        self.hangover_count = 0

    def detect_block(self, near_end: np.ndarray, far_end: np.ndarray) -> bool:
        self.current_far_max = max(self.current_far_max * 0.99,
                                   np.max(np.abs(far_end)))
        near_max = np.max(np.abs(near_end))
        detected = False

        if self.current_far_max > 1e-10:
            ratio = near_max / self.current_far_max
            detected = ratio > self.threshold

        if detected:
            self.hangover_count = self.hangover_max
            self.dtd_active = True
        elif self.hangover_count > 0:
            self.hangover_count -= 1
            self.dtd_active = True
        else:
            self.dtd_active = False

        return self.dtd_active


class AEC:
    """
    Acoustic Echo Cancellation

    Supports three filter modes:
    - TIME:    Time-domain NLMS (sample-by-sample processing)
    - FREQ:    Frequency-domain NLMS (single FFT block, n_partitions=1)
    - SUBBAND: Partitioned block FDAF (multiple partitions for long echo paths)
    """

    def __init__(self, config: Optional[AecConfig] = None):
        self.config = config or AecConfig()

        # Create adaptive filter based on mode
        if self.config.mode in (AecMode.FREQ, AecMode.SUBBAND):
            hop_size = self.config.fft_size // 2
            if self.config.mode == AecMode.FREQ:
                # FDAF: single-block frequency-domain filter
                n_partitions = 1
            else:
                # PBFDAF: partitioned block, configurable filter_length
                n_partitions = max(1, (self.config.filter_length + hop_size - 1) // hop_size)

            self.filter = SubbandNlms(
                block_size=self.config.fft_size,
                n_partitions=n_partitions,
                mu=self.config.mu,
                delta=self.config.delta
            )
            self._hop_size = self.filter.hop_size
            self._n_partitions = n_partitions
        elif self.config.mode == AecMode.LMS:
            # LMS: Time-domain, no normalization
            # filter_length = user config (default: frame_size)
            self.filter = NlmsFilter(
                filter_length=self.config.filter_length,
                mu=self.config.mu,
                delta=self.config.delta,
                leak=1.0,
                normalize=False
            )
            self.filter.clear_history = self.config.clear_filter_history
            self._hop_size = self.config.hop_size
            self._n_partitions = 0
        else:
            # TIME: Time-domain NLMS
            # filter_length = user config (default: frame_size)
            self.filter = NlmsFilter(
                filter_length=self.config.filter_length,
                mu=self.config.mu,
                delta=self.config.delta,
                leak=self.config.leak,
                normalize=True
            )
            self.filter.clear_history = self.config.clear_filter_history
            self._hop_size = self.config.hop_size
            self._n_partitions = 0

        # Error-based DTD state
        self._dtd_active = False
        self._dtd_ratio_smooth = 0.0  # Smoothed error/echo ratio

        # RES (only for frequency-domain modes)
        if self.config.enable_res and self.config.mode in (AecMode.FREQ, AecMode.SUBBAND):
            self.res = ResFilter(
                n_freqs=self.config.fft_size // 2 + 1,
                g_min_db=self.config.res_g_min_db,
                over_sub=self.config.res_over_sub,
                alpha=self.config.res_alpha
            )
        else:
            self.res = None

        # ERLE
        self.near_power = 0.0
        self.error_power = 0.0
        self.alpha = 0.95

    def reset(self):
        self.filter.reset()
        self._dtd_active = False
        self._dtd_ratio_smooth = 0.0
        if self.res:
            self.res.reset()
        self.near_power = 0.0
        self.error_power = 0.0

    @property
    def hop_size(self) -> int:
        return self._hop_size

    def process(self, near_end: np.ndarray, far_end: np.ndarray) -> np.ndarray:
        # Error-based DTD: compare error energy vs echo estimate energy.
        # Old Geigel DTD compared mic vs ref, but mic includes echo, causing
        # false triggers when echo_gain > threshold (e.g., gain=1.28 > 0.6).
        #
        # Use PREVIOUS block's smoothed ratio to decide current block's update.
        # This avoids the bootstrap problem where the filter can't converge
        # because DTD blocks updates before echo estimates are valid.

        # DTD strategy per mode:
        # - FREQ/SUBBAND: error-based DTD, freeze weights during double-talk
        # - NLMS/LMS: NO DTD — leak factor handles double-talk naturally;
        #   DTD causes weight explosion (soft DTD) or stale-weight drift (hard DTD)
        if self.config.mode in (AecMode.FREQ, AecMode.SUBBAND):
            update_weights = True
            if self.config.enable_dtd and self._dtd_active:
                update_weights = False

            output = self.filter.process(near_end, far_end, update_weights)

            # RES post-filter
            if self.res:
                far_power = np.mean(far_end ** 2)
                output_spec = np.fft.rfft(np.pad(output, (0, len(output))))
                echo_spec = self.filter.echo_spec
                enhanced_spec = self.res.process(output_spec[:len(echo_spec)],
                                                  echo_spec, far_power)
                output = np.fft.irfft(enhanced_spec, len(output) * 2)[:len(output)]

            # Update DTD state for NEXT block
            if self.config.enable_dtd:
                error_energy = np.mean(output ** 2)
                echo_energy = np.mean(np.abs(self.filter.echo_spec) ** 2)
                near_energy = np.mean(near_end ** 2)
                self._update_dtd(error_energy, echo_energy, near_energy)
        else:
            # NLMS/LMS: always update weights (no DTD)
            output, _ = self.filter.process_block(near_end, far_end, update_weights=True)

        # ERLE
        for i in range(len(near_end)):
            self.near_power = self.alpha * self.near_power + (1 - self.alpha) * near_end[i] ** 2
            self.error_power = self.alpha * self.error_power + (1 - self.alpha) * output[i] ** 2

        return output.astype(np.float32)

    def get_erle(self) -> float:
        if self.error_power < 1e-10:
            return 0.0
        return 10 * np.log10(self.near_power / (self.error_power + 1e-10) + 1e-10)

    def _update_dtd(self, error_energy: float, echo_energy: float,
                     near_energy: float):
        """Update error-based DTD state for next block.

        Only activate DTD when the filter is producing meaningful echo
        estimates (echo_energy > 1% of near_energy). Otherwise, the filter
        hasn't converged yet and DTD should remain inactive.
        """
        if echo_energy > near_energy * 0.01 and echo_energy > 1e-10:
            ratio = error_energy / (echo_energy + 1e-10)
            self._dtd_ratio_smooth = 0.8 * self._dtd_ratio_smooth + 0.2 * ratio
            self._dtd_active = self._dtd_ratio_smooth > self.config.dtd_threshold
        else:
            # Filter not converged yet — keep DTD inactive, reset smoothing
            self._dtd_ratio_smooth *= 0.9  # Slow decay
            self._dtd_active = False

    def is_dtd_active(self) -> bool:
        return self._dtd_active


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

    # LMS needs much smaller step size
    mu = args.mu
    if args.mode == 'lms' and args.mu == 0.3:
        mu = 0.01  # Default mu for LMS

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
        clear_filter_history=args.clear_history
    )

    process_wav_files(args.mic, args.ref, args.output, config)


if __name__ == '__main__':
    main()
