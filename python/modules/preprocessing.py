"""Audio pre-processing: high-pass filter + saturation detector.

Extracted from ``aec.py`` during refactor R.4. Self-contained: depends
only on numpy.
"""
import numpy as np


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
