"""Reverb tail frequency response (AEC3 strict port).

Strict line-by-line port of ``docs/aec3_extracts/src/aec3/reverb_frequency_response.{cc,h}``.

Mechanism (cc:74-106):
1. ``direct_path`` = ``frequency_response[filter_delay_blocks]`` (per-partition
   |W|² at the direct-path partition).
2. ``raw_tail`` = ``frequency_response[-1]`` (last partition's |W|²).
3. ``average_decay`` (scalar) = ``Σ_k≥1 raw_tail[k] / Σ_k≥1 direct_path[k]``,
   smoothed by EMA with ``0.2 · linear_filter_quality`` (cc:88-89).
4. ``tail_response[k] = direct_path[k] · average_decay`` (cc:91-93).
5. If ``use_conservative_tail_frequency_response``: point-wise
   ``max(tail_response[k], raw_tail[k])`` (cc:95-99).
6. Neighbour-max smoothing (cc:101-105): for k in [1, kFftLengthBy2):
   ``tail_response[k] = max(tail_response[k], 0.5·(tail[k-1] + tail[k+1]))``.

NOTE on fft-resolution: AEC3 runs at fft=128 (125 Hz/bin); we run at fft=512
(31.25 Hz/bin). At our finer resolution, raw_tail has voice-harmonic-induced
spurious peaks/valleys that AEC3's coarser grid averages out naturally. A
prior change (commit 8aafe61, reverted 2026-05-27) expanded both the
conservative max and the neighbour-max windows to ±125 Hz physical width
(= ±4 bins at our fft) as "physical equivalence". That change INFLATED the
HF tail response (sparse HF energy gets spread across the wider window),
which the residual_echo_estimator then accumulates into ``reverb_model``,
producing rev/dir ratios up to 1e6 and crushing G_hf during DT — the
"painted black HF" artifact. Uses AEC3 strict literals here; the
fft-aware expansion would be a beyond-AEC3 optimisation candidate.
"""
from __future__ import annotations

import numpy as np


class ReverbFrequencyResponse:
    """Per-bin tail frequency response synthesizer (AEC3 strict).

    Maintains a single EMA scalar ``_average_decay`` (tail/direct path energy
    ratio) and produces ``tail_response[k] = direct_path[k] * average_decay``
    when ``update`` is called with a fresh per-partition |W|² matrix.

    The conservative variant additionally point-wise-maxes tail_response[k]
    against the raw last-partition response (AEC3 cc:95-99).
    """

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
        # AEC3 reverb_frequency_response.cc:88 applies α=0.2·quality per
        # kBlockSize=64 sample block (= 4 ms @ 16 kHz). Our pipeline calls
        # update() per hop_size (= 10 ms @ hop=160), so a verbatim 0.2 is
        # 2.5× slower in wall-clock — average_decay sticks to stale FS
        # values when entering DT2, inflating tail_response and crushing
        # HF gain ("painted-black" symptom after far-end-single-talk).
        # Wall-clock-equivalent per-hop α derived via
        # per_block_ema_alpha_to_per_hop(0.2, hop, sr) ≈ 0.428 at hop=160.
        # Default-OFF preserves byte-equal; flag-ON enables the strict
        # AEC3 wall-clock envelope.
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
        """Refresh tail response using per-partition |W|² spectra.

        ``frequency_response`` shape: ``(n_partitions, n_freqs)`` float32. Each
        row is the magnitude-squared partition spectrum.
        ``filter_delay_blocks``: partition index of the direct path.
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

        # Average-decay scalar, skipping DC (AEC3 cc:34 kSkipBins = 1).
        direct_energy = float(np.sum(freq_resp_direct[1:]))
        if direct_energy == 0.0:
            average_decay = 0.0
        else:
            tail_energy = float(np.sum(freq_resp_tail[1:]))
            average_decay = tail_energy / direct_energy

        # EMA smoothing (cc:88-89): α = 0.2 · linear_filter_quality (per
        # AEC3 4 ms block). Wall-clock variant rescales the base alpha so
        # the per-hop envelope matches AEC3's per-block envelope.
        smoothing = self._smoothing_base * float(linear_filter_quality)
        self._average_decay += smoothing * (average_decay - self._average_decay)

        # Per-bin tail = direct × scalar decay (cc:91-93).
        tail = freq_resp_direct * self._average_decay

        # Conservative variant (cc:95-99): point-wise max with raw last
        # partition (NO smoothing — strict AEC3).
        if self._use_conservative:
            np.maximum(tail, freq_resp_tail, out=tail)

        # Neighbour-max smoothing (cc:101-105): ±1 bin avg, indices k=1..N-2.
        # Strict AEC3 literal — bin-index based, not freq-width based.
        if self._n_freqs >= 3:
            avg_neighbour = 0.5 * (tail[:-2] + tail[2:])
            tail[1:-1] = np.maximum(tail[1:-1], avg_neighbour)

        self._tail_response = tail.astype(np.float32)


__all__ = ["ReverbFrequencyResponse"]
