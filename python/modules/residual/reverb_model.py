"""ReverbModel — port of AEC3 reverb_model.cc.

Mirrors docs/aec3_extracts/src/aec3/reverb_model.{cc,h}. Tiny exponential
IIR accumulator over a per-bin power spectrum:

    reverb[k] = (reverb[k] + power_spectrum[k] * scaling) * decay

Two update flavours match AEC3 source:
  * ``update_no_freq_shaping`` — scalar ``scaling`` (used in nonlinear
    mode, where scaling = echo_path_gain).
  * ``update`` — per-bin ``scaling`` (linear mode, scaling = filter
    frequency response squared).

The accumulator is shared across modes; only the update rule differs.
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
        """Mirrors UpdateReverbNoFreqShaping (cc:28-39)."""
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
        """Mirrors UpdateReverb (cc:41-52). Per-bin scaling."""
        if decay <= 0.0:
            return
        np.add(self._reverb, power_spectrum * scaling, out=self._reverb)
        self._reverb *= float(decay)
