"""Spectral-domain suppression filter: ``E_out = g * E + sqrt(1 - g^2) * CN`` (framing is owned by the orchestrator)."""
import numpy as np


def apply_suppression(
    *,
    spectrum: np.ndarray,           # complex (n_bins,)
    suppression_gain: np.ndarray,   # real (n_bins,) in [0, 1]
    comfort_noise: np.ndarray,      # complex (n_bins,); pass zeros for no CNG
) -> np.ndarray:
    """Apply the per-bin amplitude gain ``g`` and mix in comfort noise (zeros to skip)."""
    g = suppression_gain.astype(np.float32, copy=False)
    cn_gain = np.sqrt(np.maximum(1.0 - g * g, 0.0)).astype(np.float32)
    return (spectrum * g + comfort_noise * cn_gain).astype(spectrum.dtype, copy=False)
