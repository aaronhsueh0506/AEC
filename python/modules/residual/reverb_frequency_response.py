"""Reverb tail frequency response (AEC3 port).

Float-scale port of `docs/aec3_extracts/src/aec3/reverb_frequency_response.{cc,h}`.

Replaces our previous current-frame ``S²/X²`` coupling approximation with the
canonical AEC3 mechanism: take the last partition of the linear filter as the
"tail" response, the direct-path partition as the "early" response, compute
their per-bin energy ratio (the ``average_decay``), then synthesize a
smoothed tail spectrum that scales with the direct path. Used as
``power_spectrum_scaling`` for the linear-mode reverb model.
"""
from __future__ import annotations

import numpy as np


class ReverbFrequencyResponse:
    """Per-bin tail frequency response synthesizer.

    Maintains a single EMA scalar ``_average_decay`` (tail/direct path energy
    ratio) and produces ``tail_response[k] = direct_path[k] * average_decay``
    when ``Update`` is called with a fresh per-partition |W|² matrix.

    The conservative variant additionally clamps tail_response[k] to be at
    least the raw tail-partition energy at that bin (matches AEC3
    ``use_conservative_tail_frequency_response``).
    """

    def __init__(self, n_freqs: int,
                 use_conservative_tail_frequency_response: bool = False) -> None:
        self._n_freqs = int(n_freqs)
        self._use_conservative = bool(use_conservative_tail_frequency_response)
        self._average_decay: float = 0.0
        self._tail_response = np.zeros(self._n_freqs, dtype=np.float32)

    @property
    def tail_response(self) -> np.ndarray:
        return self._tail_response

    @property
    def average_decay(self) -> float:
        return float(self._average_decay)

    def reset(self) -> None:
        self._average_decay = 0.0
        self._tail_response.fill(0.0)

    def update(self, frequency_response: np.ndarray,
               filter_delay_blocks: int,
               linear_filter_quality: float | None,
               stationary_block: bool) -> None:
        """Refresh tail response using per-partition |W|² spectra.

        ``frequency_response`` shape: ``(n_partitions, n_freqs)`` float32. Each
        row is the magnitude-squared partition spectrum.
        ``filter_delay_blocks``: the partition index of the direct path.
        ``linear_filter_quality``: scalar in [0, 1] (or None) — convergence
        confidence used to smooth the decay estimate.
        ``stationary_block``: if True, skip the update (mirrors AEC3 cc:67).
        """
        if stationary_block or linear_filter_quality is None:
            return
        n_partitions = int(frequency_response.shape[0])
        if filter_delay_blocks < 0 or filter_delay_blocks >= n_partitions:
            return
        freq_resp_direct = frequency_response[filter_delay_blocks].astype(np.float32)
        freq_resp_tail = frequency_response[-1].astype(np.float32)

        # Discard DC bin in energy ratio (matches AEC3 cc:34 kSkipBins = 1).
        direct_energy = float(np.sum(freq_resp_direct[1:]))
        if direct_energy == 0.0:
            average_decay = 0.0
        else:
            tail_energy = float(np.sum(freq_resp_tail[1:]))
            average_decay = tail_energy / direct_energy

        smoothing = 0.2 * float(linear_filter_quality)
        self._average_decay += smoothing * (average_decay - self._average_decay)

        tail = freq_resp_direct * self._average_decay
        if self._use_conservative:
            np.maximum(tail, freq_resp_tail, out=tail)
        # Neighbour-max smoothing (AEC3 cc:101-105).
        if self._n_freqs >= 3:
            avg_neighbour = 0.5 * (tail[:-2] + tail[2:])
            tail[1:-1] = np.maximum(tail[1:-1], avg_neighbour)
        self._tail_response = tail.astype(np.float32)


__all__ = ["ReverbFrequencyResponse"]
