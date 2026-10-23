"""Reverb tail frequency response synthesizer.

From per-partition |W|^2 spectra: ``tail[k] = direct_path[k] * average_decay``, where
``average_decay`` is the EMA (rate ``0.2 * linear_filter_quality``) of the last-partition to
direct-path energy ratio over bins >= 1. The conservative variant takes the point-wise max with
the raw last partition, then ``tail[k] = max(tail[k], 0.5 * (tail[k-1] + tail[k+1]))`` in place, ascending.
"""
from __future__ import annotations

import numpy as np


class ReverbFrequencyResponse:
    """Per-bin tail frequency response synthesizer; ``update()`` refreshes it from a per-partition |W|^2 matrix."""

    def __init__(self, n_freqs: int,
                 use_conservative_tail_frequency_response: bool = False,
                 *,
                 sr: int = 16000,
                 hop_size: int = 160,
                 use_wallclock_smoothing: bool = False) -> None:
        self._n_freqs = int(n_freqs)
        self._use_conservative = bool(use_conservative_tail_frequency_response)
        self._average_decay: float = 0.0
        self._tail_response = np.zeros(self._n_freqs, dtype=np.float32)
        # The 0.2 smoothing is defined per 4 ms block; with ``use_wallclock_smoothing`` it is
        # converted to the per-hop equivalent, otherwise 0.2 is applied per hop.
        self._sr = int(sr)
        self._hop_size = int(hop_size)
        self._use_wallclock_smoothing = bool(use_wallclock_smoothing)
        if self._use_wallclock_smoothing:
            from .. import aec3_scale as _aec3_scale
            self._smoothing_base = _aec3_scale.per_block_ema_alpha_to_per_hop(
                0.2, self._hop_size, self._sr
            )
        else:
            self._smoothing_base = 0.2

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
        """Refresh the tail response.

        ``frequency_response``: (n_partitions, n_freqs) float32 |W|^2 per partition;
        ``filter_delay_blocks``: direct-path partition index; ``linear_filter_quality``:
        scalar in [0, 1] or None; ``stationary_block`` True skips the update.
        """
        if stationary_block or linear_filter_quality is None:
            return
        n_partitions = int(frequency_response.shape[0])
        if filter_delay_blocks < 0 or filter_delay_blocks >= n_partitions:
            return
        freq_resp_direct = frequency_response[filter_delay_blocks].astype(np.float32)
        freq_resp_tail = frequency_response[-1].astype(np.float32)

        # Average-decay scalar, skipping DC.
        direct_energy = float(np.sum(freq_resp_direct[1:]))
        if direct_energy == 0.0:
            average_decay = 0.0
        else:
            tail_energy = float(np.sum(freq_resp_tail[1:]))
            average_decay = tail_energy / direct_energy

        # EMA smoothing with rate base * linear_filter_quality.
        smoothing = self._smoothing_base * float(linear_filter_quality)
        self._average_decay += smoothing * (average_decay - self._average_decay)

        tail = freq_resp_direct * self._average_decay

        # Conservative variant: point-wise max with the raw last partition.
        if self._use_conservative:
            np.maximum(tail, freq_resp_tail, out=tail)

        # Neighbour-max pass; in place and ascending so a strong peak spreads rightward.
        if self._n_freqs >= 3:
            n = len(tail)
            for k in range(1, n - 1):
                tail[k] = max(tail[k], 0.5 * (tail[k - 1] + tail[k + 1]))

        self._tail_response = tail.astype(np.float32)


__all__ = ["ReverbFrequencyResponse"]
