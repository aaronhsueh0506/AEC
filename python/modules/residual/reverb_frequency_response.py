"""Reverb tail frequency response (AEC3 port).

Float-scale port of `docs/aec3_extracts/src/aec3/reverb_frequency_response.{cc,h}`.

Replaces our previous current-frame ``S²/X²`` coupling approximation with the
canonical AEC3 mechanism: take the last partition of the linear filter as the
"tail" response, the direct-path partition as the "early" response, compute
their per-bin energy ratio (the ``average_decay``), then synthesize a
smoothed tail spectrum that scales with the direct path. Used as
``power_spectrum_scaling`` for the linear-mode reverb model.

AEC3-equivalent freq-domain windows (2026-05-27): AEC3's per-bin operations
(conservative max + neighbour-max smoothing) at fft=128 cover 125 Hz per bin
(0.45% of nyquist). At our fft=512 (31.25 Hz/bin) the same ±1-bin operation
only covers 31.25 Hz — too narrow to smooth out voice-harmonic-induced
spectral valleys in the filter taps, which causes the tail_response to develop
spurious peaks in valley bins → reverb inflated → gain drop visible as
"black valley". We compensate by scaling the smoothing window so it covers
the same physical freq range as AEC3 (125 Hz neighbor width).
"""
from __future__ import annotations

import numpy as np


# AEC3 reference: kFftLength = 128 → 65 bins → 125 Hz/bin at sr=16000.
# All AEC3 freq-domain smoothing/conservative operations assume this resolution.
_AEC3_REFERENCE_BIN_HZ = 125.0


def _bins_per_side_for_aec3_neighbour_width(n_freqs: int, sr: int) -> int:
    """Return the per-side bin count so that ±N bins cover the same freq
    width as AEC3's ±1 bin (= 125 Hz). At our fft=512 this gives 4.
    """
    if n_freqs <= 1:
        return 1
    bin_width = float(sr) / float(2 * (n_freqs - 1))  # half-spectrum: bin = sr/(2*(N-1)) ≈ sr/fft
    return max(1, int(round(_AEC3_REFERENCE_BIN_HZ / bin_width)))


class ReverbFrequencyResponse:
    """Per-bin tail frequency response synthesizer.

    Maintains a single EMA scalar ``_average_decay`` (tail/direct path energy
    ratio) and produces ``tail_response[k] = direct_path[k] * average_decay``
    when ``Update`` is called with a fresh per-partition |W|² matrix.

    The conservative variant additionally clamps tail_response[k] to be at
    least the raw tail-partition energy at that bin (matches AEC3
    ``use_conservative_tail_frequency_response``).

    fft-resolution-aware smoothing windows preserve AEC3 freq-domain
    semantics regardless of fft size — see ``_bins_per_side_for_aec3_neighbour_width``.
    """

    def __init__(self, n_freqs: int,
                 use_conservative_tail_frequency_response: bool = False,
                 sr: int = 16000) -> None:
        self._n_freqs = int(n_freqs)
        self._use_conservative = bool(use_conservative_tail_frequency_response)
        self._average_decay: float = 0.0
        self._tail_response = np.zeros(self._n_freqs, dtype=np.float32)
        # Bins per side for the AEC3-equivalent ±125 Hz neighbour window.
        self._n_side = _bins_per_side_for_aec3_neighbour_width(self._n_freqs, sr)

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
        n = self._n_side
        if self._use_conservative:
            # AEC3 cc:88-91 (point-wise max with raw tail). At our finer fft
            # resolution, raw tail has voice-harmonic-driven spurious peaks in
            # spectral valleys; smooth raw_tail over an AEC3-equivalent freq
            # width (±n bins ≈ ±125 Hz, centered, includes self) before the
            # max so we don't over-inflate tail in valley bins.
            raw_tail_smoothed = _neighbour_window_mean(
                freq_resp_tail, n, include_self=True
            )
            np.maximum(tail, raw_tail_smoothed, out=tail)
        # Neighbour-max smoothing (AEC3 cc:101-105). At our fft, expand the
        # ±1-bin window to ±n bins so it covers AEC3's ±125 Hz freq width —
        # preserves the "smooth out narrow tail valleys" intent at our higher
        # freq resolution. Excludes self (matches AEC3 avg of k-1 + k+1 only).
        if self._n_freqs >= 2 * n + 1:
            avg_neighbour = _neighbour_window_mean(tail, n, include_self=False)
            tail[n:-n] = np.maximum(tail[n:-n], avg_neighbour[n:-n])
        self._tail_response = tail.astype(np.float32)


def _neighbour_window_mean(x: np.ndarray, half_width: int,
                            include_self: bool) -> np.ndarray:
    """Per-bin mean over [k-h, k+h]. Edges fall through with np.roll wraparound.

    Caller should only consume bins [h : n-h] where the result is valid.
    """
    if half_width <= 0:
        return x.copy() if include_self else np.zeros_like(x)
    out = np.zeros_like(x)
    count = 0
    for shift in range(-half_width, half_width + 1):
        if shift == 0 and not include_self:
            continue
        out += np.roll(x, -shift)
        count += 1
    out /= count
    return out


__all__ = ["ReverbFrequencyResponse"]
