"""SuppressionFilter — applies suppression gain + optional comfort noise to a
complex spectrum and returns the suppressed spectrum.

Mirrors the contract of docs/aec3_extracts/src/aec3/suppression_filter.cc but
deliberately stops short of the full IFFT / sqrt-Hann window / OLA management.
Our orchestrator already owns FFT framing (R.X refactors verified byte-equal),
so this module operates in the spectral domain only: input ``E`` (complex
spectrum), output ``E_out = g * E + sqrt(1-g²) * CN``.

  - ``suppression_gain`` is the per-bin amplitude gain (already sqrt'd by
    SuppressionGain.get_gain()).
  - ``comfort_noise`` is a per-bin complex spectrum; pass zeros to skip CNG.
"""
import numpy as np


def apply_suppression(
    *,
    spectrum: np.ndarray,           # complex (n_bins,)
    suppression_gain: np.ndarray,   # real (n_bins,) in [0, 1]
    comfort_noise: np.ndarray,      # complex (n_bins,); pass zeros for no CNG
) -> np.ndarray:
    """Returns ``g*spectrum + sqrt(1-g²)*comfort_noise``."""
    g = suppression_gain.astype(np.float32, copy=False)
    cn_gain = np.sqrt(np.maximum(1.0 - g * g, 0.0)).astype(np.float32)
    return (spectrum * g + comfort_noise * cn_gain).astype(spectrum.dtype, copy=False)
