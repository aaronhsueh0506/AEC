"""ReverbModel: per-bin exponential accumulator ``reverb[k] = (reverb[k] + power[k] * scaling) * decay``.

``update_no_freq_shaping`` takes a scalar scaling (nonlinear mode, echo-path gain);
``update`` takes a per-bin scaling (linear mode, filter frequency response squared).
"""
from __future__ import annotations

import numpy as np


class ReverbModel:
    def __init__(self, n_bins: int = 257) -> None:
        self._n_bins = int(n_bins)
        self._reverb = np.zeros(self._n_bins, dtype=np.float32)

    def reset(self) -> None:
        self._reverb.fill(0.0)

    @property
    def reverb(self) -> np.ndarray:
        return self._reverb

    def update_no_freq_shaping(
        self, power_spectrum: np.ndarray, scaling: float, decay: float
    ) -> None:
        """Scalar-scaling update; no-op when ``decay <= 0``."""
        if decay <= 0.0:
            return
        np.add(self._reverb, power_spectrum * float(scaling), out=self._reverb)
        self._reverb *= float(decay)

    def update(
        self,
        power_spectrum: np.ndarray,
        scaling: np.ndarray,
        decay: float,
    ) -> None:
        """Per-bin-scaling update; no-op when ``decay <= 0``."""
        if decay <= 0.0:
            return
        np.add(self._reverb, power_spectrum * scaling, out=self._reverb)
        self._reverb *= float(decay)
