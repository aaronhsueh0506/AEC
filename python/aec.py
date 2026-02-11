"""
Acoustic Echo Cancellation (AEC) - Python Reference Implementation

NLMS (Normalized Least Mean Squares) adaptive filter with DTD (Double-Talk Detection)

Usage:
    python aec.py mic.wav ref.wav output.wav [--mu 0.3] [--filter-ms 250]
"""

import numpy as np
from dataclasses import dataclass
from typing import Optional, Tuple
import argparse
import soundfile as sf


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

    @property
    def frame_size(self) -> int:
        return self.sample_rate * self.frame_size_ms // 1000

    @property
    def hop_size(self) -> int:
        return self.sample_rate * self.frame_shift_ms // 1000

    @property
    def filter_length(self) -> int:
        return self.sample_rate * self.filter_length_ms // 1000


class NlmsFilter:
    """
    Normalized Least Mean Squares Adaptive Filter

    Algorithm:
        y_hat[n] = w^T * x[n]           (echo estimate)
        e[n] = d[n] - y_hat[n]          (error/output)
        mu_eff = mu / (||x[n]||^2 + delta)
        w = leak * w + mu_eff * e[n] * x[n]  (weight update)
    """

    def __init__(self, filter_length: int, mu: float = 0.3,
                 delta: float = 1e-8, leak: float = 0.9999):
        self.filter_length = filter_length
        self.mu = mu
        self.delta = delta
        self.leak = leak

        # Filter weights and reference buffer
        self.weights = np.zeros(filter_length, dtype=np.float32)
        self.ref_buffer = np.zeros(filter_length, dtype=np.float32)
        self.power_sum = 0.0

    def reset(self):
        """Reset filter state"""
        self.weights.fill(0)
        self.ref_buffer.fill(0)
        self.power_sum = 0.0

    def process_sample(self, near_end: float, far_end: float,
                       update_weights: bool = True) -> Tuple[float, float]:
        """
        Process single sample

        Args:
            near_end: Microphone input (d[n])
            far_end: Reference signal (x[n])
            update_weights: If False, only compute output

        Returns:
            (error, echo_estimate)
        """
        # Update power sum (remove oldest, add newest)
        oldest = self.ref_buffer[-1]
        self.power_sum = max(0, self.power_sum - oldest * oldest + far_end * far_end)

        # Shift reference buffer and insert new sample
        self.ref_buffer[1:] = self.ref_buffer[:-1]
        self.ref_buffer[0] = far_end

        # Compute echo estimate: y_hat = w^T * x
        echo_est = np.dot(self.weights, self.ref_buffer)

        # Error signal
        error = near_end - echo_est

        # Update weights if enabled
        if update_weights:
            mu_eff = self.mu / (self.power_sum + self.delta)
            self.weights = (self.leak * self.weights +
                           mu_eff * error * self.ref_buffer)

        return error, echo_est

    def process_block(self, near_end: np.ndarray, far_end: np.ndarray,
                      update_weights: bool = True) -> Tuple[np.ndarray, np.ndarray]:
        """
        Process block of samples

        Args:
            near_end: Microphone input array
            far_end: Reference signal array
            update_weights: If False, only compute output

        Returns:
            (output, echo_estimate)
        """
        n = len(near_end)
        output = np.zeros(n, dtype=np.float32)
        echo_est = np.zeros(n, dtype=np.float32)

        for i in range(n):
            output[i], echo_est[i] = self.process_sample(
                near_end[i], far_end[i], update_weights
            )

        return output, echo_est


class DtdEstimator:
    """
    Double-Talk Detector

    Uses Geigel method: DTD = |d[n]| > threshold * max(|x[n-k]|)
    Plus energy-based criterion
    """

    def __init__(self, window_length: int, threshold: float = 0.6,
                 hangover_frames: int = 15):
        self.window_length = window_length
        self.threshold = threshold
        self.hangover_max = hangover_frames

        self.far_max_buffer = np.zeros(window_length, dtype=np.float32)
        self.buf_idx = 0
        self.current_far_max = 0.0

        self.dtd_active = False
        self.hangover_count = 0

    def reset(self):
        """Reset DTD state"""
        self.far_max_buffer.fill(0)
        self.buf_idx = 0
        self.current_far_max = 0.0
        self.dtd_active = False
        self.hangover_count = 0

    def detect_block(self, near_end: np.ndarray, far_end: np.ndarray) -> bool:
        """
        Detect double-talk for a block

        Args:
            near_end: Microphone input
            far_end: Reference signal

        Returns:
            True if double-talk detected
        """
        # Block-level energy comparison
        near_energy = np.mean(near_end ** 2)
        far_energy = np.mean(far_end ** 2)

        # Update far-end max tracking
        self.current_far_max = max(self.current_far_max * 0.99,
                                   np.max(np.abs(far_end)))

        # Geigel criterion
        near_max = np.max(np.abs(near_end))
        detected = False

        if self.current_far_max > 1e-10:
            ratio = near_max / self.current_far_max
            detected = ratio > self.threshold

        # Hangover logic
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

    Main class that orchestrates NLMS filter with DTD protection
    """

    def __init__(self, config: Optional[AecConfig] = None):
        self.config = config or AecConfig()

        # Create sub-modules
        self.nlms = NlmsFilter(
            filter_length=self.config.filter_length,
            mu=self.config.mu,
            delta=self.config.delta,
            leak=self.config.leak
        )

        self.dtd = DtdEstimator(
            window_length=self.config.filter_length,
            threshold=self.config.dtd_threshold,
            hangover_frames=self.config.dtd_hangover_frames
        ) if self.config.enable_dtd else None

        # ERLE estimation
        self.near_power = 0.0
        self.error_power = 0.0
        self.alpha = 0.95

    def reset(self):
        """Reset AEC state"""
        self.nlms.reset()
        if self.dtd:
            self.dtd.reset()
        self.near_power = 0.0
        self.error_power = 0.0

    @property
    def hop_size(self) -> int:
        return self.config.hop_size

    def process(self, near_end: np.ndarray, far_end: np.ndarray) -> np.ndarray:
        """
        Process audio block

        Args:
            near_end: Microphone input [hop_size]
            far_end: Reference signal [hop_size]

        Returns:
            Echo-cancelled output [hop_size]
        """
        # DTD detection
        update_weights = True
        if self.dtd and self.config.enable_dtd:
            dtd_active = self.dtd.detect_block(near_end, far_end)
            update_weights = not dtd_active

        # Process through NLMS
        output, echo_est = self.nlms.process_block(near_end, far_end, update_weights)

        # Update ERLE estimation
        for i in range(len(near_end)):
            self.near_power = (self.alpha * self.near_power +
                              (1 - self.alpha) * near_end[i] ** 2)
            self.error_power = (self.alpha * self.error_power +
                               (1 - self.alpha) * output[i] ** 2)

        return output

    def get_erle(self) -> float:
        """Get current ERLE in dB"""
        if self.error_power < 1e-10:
            return 0.0
        erle_linear = self.near_power / (self.error_power + 1e-10)
        return 10 * np.log10(erle_linear + 1e-10)

    def is_dtd_active(self) -> bool:
        """Check if DTD is currently active"""
        return self.dtd.dtd_active if self.dtd else False


def process_wav_files(mic_path: str, ref_path: str, out_path: str,
                      config: Optional[AecConfig] = None):
    """
    Process WAV files through AEC

    Args:
        mic_path: Path to microphone WAV file
        ref_path: Path to reference WAV file
        out_path: Path to output WAV file
        config: AEC configuration
    """
    # Read input files
    mic_data, mic_sr = sf.read(mic_path)
    ref_data, ref_sr = sf.read(ref_path)

    if mic_sr != ref_sr:
        raise ValueError(f"Sample rate mismatch: mic={mic_sr}, ref={ref_sr}")

    # Ensure mono
    if mic_data.ndim > 1:
        mic_data = mic_data[:, 0]
    if ref_data.ndim > 1:
        ref_data = ref_data[:, 0]

    # Match lengths
    num_samples = min(len(mic_data), len(ref_data))
    mic_data = mic_data[:num_samples].astype(np.float32)
    ref_data = ref_data[:num_samples].astype(np.float32)

    print(f"AEC Processing:")
    print(f"  Microphone: {mic_path} ({num_samples} samples)")
    print(f"  Reference:  {ref_path}")
    print(f"  Sample rate: {mic_sr} Hz")
    print(f"  Duration: {num_samples / mic_sr:.2f} seconds")

    # Create AEC
    if config is None:
        config = AecConfig(sample_rate=mic_sr)
    else:
        config.sample_rate = mic_sr

    print(f"  Step size (mu): {config.mu}")
    print(f"  Filter length: {config.filter_length_ms} ms ({config.filter_length} taps)")
    print(f"  DTD: {'enabled' if config.enable_dtd else 'disabled'}")
    print()

    aec = AEC(config)
    hop_size = aec.hop_size

    # Process
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

        # Progress
        if processed % mic_sr == 0:
            print(f"  Processed: {processed / mic_sr:.1f} s, ERLE: {erle:.1f} dB\r",
                  end='', flush=True)

    print(f"\n\nResults:")
    print(f"  Processed samples: {processed}")
    print(f"  Max ERLE: {max_erle:.1f} dB")
    print(f"  DTD active frames: {dtd_frames} ({100 * dtd_frames * hop_size / processed:.1f}%)")

    # Write output
    sf.write(out_path, output, mic_sr)
    print(f"\nOutput written to: {out_path}")


def main():
    parser = argparse.ArgumentParser(
        description='Acoustic Echo Cancellation (AEC)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python aec.py mic.wav ref.wav output.wav
    python aec.py mic.wav ref.wav output.wav --mu 0.5 --filter-ms 300
    python aec.py mic.wav ref.wav output.wav --no-dtd
        """
    )
    parser.add_argument('mic', help='Microphone input WAV file')
    parser.add_argument('ref', help='Reference/loudspeaker WAV file')
    parser.add_argument('output', help='Output WAV file')
    parser.add_argument('--mu', type=float, default=0.3,
                        help='Step size (default: 0.3)')
    parser.add_argument('--filter-ms', type=int, default=250,
                        help='Filter length in ms (default: 250)')
    parser.add_argument('--no-dtd', action='store_true',
                        help='Disable double-talk detection')

    args = parser.parse_args()

    config = AecConfig(
        mu=args.mu,
        filter_length_ms=args.filter_ms,
        enable_dtd=not args.no_dtd
    )

    process_wav_files(args.mic, args.ref, args.output, config)


if __name__ == '__main__':
    main()
