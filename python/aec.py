"""
Acoustic Echo Cancellation (AEC) - Python Reference Implementation

Supports:
- Time-domain NLMS adaptive filter
- Frequency-domain (Subband) NLMS
- Double-Talk Detection (DTD)
- Residual Echo Suppressor (RES)

Usage:
    python aec.py mic.wav ref.wav output.wav [--mode subband] [--enable-res]
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Optional, Tuple
from enum import Enum
import argparse
import soundfile as sf


class AecMode(Enum):
    TIME_DOMAIN = "time"
    SUBBAND = "subband"


@dataclass
class AecConfig:
    """AEC Configuration"""
    sample_rate: int = 16000
    frame_size_ms: int = 20
    frame_shift_ms: int = 10
    filter_length_ms: int = 250
    mu: float = 0.3              # Step size
    delta: float = 1e-8          # Regularization
    leak: float = 0.9999         # Weight leakage
    enable_dtd: bool = True
    dtd_threshold: float = 0.6   # Geigel threshold
    dtd_hangover_frames: int = 15

    # RES parameters
    enable_res: bool = False
    res_g_min_db: float = -20.0
    res_over_sub: float = 1.5
    res_alpha: float = 0.8

    # Mode
    mode: AecMode = AecMode.TIME_DOMAIN

    @property
    def frame_size(self) -> int:
        return self.sample_rate * self.frame_size_ms // 1000

    @property
    def hop_size(self) -> int:
        return self.sample_rate * self.frame_shift_ms // 1000

    @property
    def filter_length(self) -> int:
        return self.sample_rate * self.filter_length_ms // 1000

    @property
    def fft_size(self) -> int:
        # Next power of 2 >= frame_size * 2 (for overlap-save)
        n = self.frame_size * 2
        return 1 << (n - 1).bit_length()


class NlmsFilter:
    """Time-domain NLMS Adaptive Filter"""

    def __init__(self, filter_length: int, mu: float = 0.3,
                 delta: float = 1e-8, leak: float = 0.9999):
        self.filter_length = filter_length
        self.mu = mu
        self.delta = delta
        self.leak = leak
        self.weights = np.zeros(filter_length, dtype=np.float32)
        self.ref_buffer = np.zeros(filter_length, dtype=np.float32)
        self.power_sum = 0.0

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

        if update_weights:
            mu_eff = self.mu / (self.power_sum + self.delta)
            self.weights = self.leak * self.weights + mu_eff * error * self.ref_buffer

        return error, echo_est

    def process_block(self, near_end: np.ndarray, far_end: np.ndarray,
                      update_weights: bool = True) -> Tuple[np.ndarray, np.ndarray]:
        n = len(near_end)
        output = np.zeros(n, dtype=np.float32)
        echo_est = np.zeros(n, dtype=np.float32)
        for i in range(n):
            output[i], echo_est[i] = self.process_sample(
                near_end[i], far_end[i], update_weights)
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

        # Update weights
        if update_weights:
            for p in range(self.n_partitions):
                p_idx = (curr_p - p) % self.n_partitions
                mu_eff = self.mu / (self.power + self.delta)
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

    Supports time-domain and frequency-domain modes with optional RES.
    """

    def __init__(self, config: Optional[AecConfig] = None):
        self.config = config or AecConfig()

        # Create adaptive filter
        if self.config.mode == AecMode.SUBBAND:
            n_partitions = max(1, self.config.filter_length // (self.config.fft_size // 2))
            self.filter = SubbandNlms(
                block_size=self.config.fft_size,
                n_partitions=n_partitions,
                mu=self.config.mu,
                delta=self.config.delta
            )
            self._hop_size = self.filter.hop_size
        else:
            self.filter = NlmsFilter(
                filter_length=self.config.filter_length,
                mu=self.config.mu,
                delta=self.config.delta,
                leak=self.config.leak
            )
            self._hop_size = self.config.hop_size

        # DTD
        self.dtd = DtdEstimator(
            window_length=self.config.filter_length,
            threshold=self.config.dtd_threshold,
            hangover_frames=self.config.dtd_hangover_frames
        ) if self.config.enable_dtd else None

        # RES
        if self.config.enable_res and self.config.mode == AecMode.SUBBAND:
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
        if self.dtd:
            self.dtd.reset()
        if self.res:
            self.res.reset()
        self.near_power = 0.0
        self.error_power = 0.0

    @property
    def hop_size(self) -> int:
        return self._hop_size

    def process(self, near_end: np.ndarray, far_end: np.ndarray) -> np.ndarray:
        # DTD
        update_weights = True
        if self.dtd and self.config.enable_dtd:
            update_weights = not self.dtd.detect_block(near_end, far_end)

        # Process
        if self.config.mode == AecMode.SUBBAND:
            output = self.filter.process(near_end, far_end, update_weights)

            # RES post-filter
            if self.res:
                far_power = np.mean(far_end ** 2)
                output_spec = np.fft.rfft(np.pad(output, (0, len(output))))
                echo_spec = self.filter.echo_spec
                enhanced_spec = self.res.process(output_spec[:len(echo_spec)],
                                                  echo_spec, far_power)
                output = np.fft.irfft(enhanced_spec, len(output) * 2)[:len(output)]
        else:
            output, _ = self.filter.process_block(near_end, far_end, update_weights)

        # ERLE
        for i in range(len(near_end)):
            self.near_power = self.alpha * self.near_power + (1 - self.alpha) * near_end[i] ** 2
            self.error_power = self.alpha * self.error_power + (1 - self.alpha) * output[i] ** 2

        return output.astype(np.float32)

    def get_erle(self) -> float:
        if self.error_power < 1e-10:
            return 0.0
        return 10 * np.log10(self.near_power / (self.error_power + 1e-10) + 1e-10)

    def is_dtd_active(self) -> bool:
        return self.dtd.dtd_active if self.dtd else False


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
    print(f"  Filter length: {config.filter_length_ms} ms")
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
Examples:
    python aec.py mic.wav ref.wav output.wav
    python aec.py mic.wav ref.wav output.wav --mode subband --enable-res
    python aec.py mic.wav ref.wav output.wav --mu 0.5 --filter-ms 300
        """
    )
    parser.add_argument('mic', help='Microphone input WAV file')
    parser.add_argument('ref', help='Reference/loudspeaker WAV file')
    parser.add_argument('output', help='Output WAV file')
    parser.add_argument('--mu', type=float, default=0.3, help='Step size (default: 0.3)')
    parser.add_argument('--filter-ms', type=int, default=250, help='Filter length in ms')
    parser.add_argument('--mode', choices=['time', 'subband'], default='time',
                        help='Filter mode (default: time)')
    parser.add_argument('--no-dtd', action='store_true', help='Disable DTD')
    parser.add_argument('--enable-res', action='store_true', help='Enable RES post-filter')
    parser.add_argument('--res-g-min', type=float, default=-20.0, help='RES min gain (dB)')

    args = parser.parse_args()

    config = AecConfig(
        mu=args.mu,
        filter_length_ms=args.filter_ms,
        mode=AecMode.SUBBAND if args.mode == 'subband' else AecMode.TIME_DOMAIN,
        enable_dtd=not args.no_dtd,
        enable_res=args.enable_res,
        res_g_min_db=args.res_g_min
    )

    process_wav_files(args.mic, args.ref, args.output, config)


if __name__ == '__main__':
    main()
