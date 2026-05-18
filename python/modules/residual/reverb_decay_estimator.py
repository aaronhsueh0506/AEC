"""Adaptive reverb decay estimator (AEC3-derived, partition-level).

Port of `docs/aec3_extracts/src/aec3/reverb_decay_estimator.{cc,h}` adapted to
our partition layout. AEC3 splits the time-domain impulse response into
``kFftLengthBy2 = 64`` sample sub-blocks (one per AEC3 partition slot); we
keep the analysis at the partition granularity (each PBFDKF partition spans
``hop_size`` samples, e.g. 160 @ 16 kHz), which (a) avoids costly per-sample
log/regress work and (b) maps cleanly to the per-partition |W|² output the
``ReverbFrequencyResponse`` port already requires.

The algorithm:

  1. Per hop, identify the "direct path" partition (= ``filter_delay_blocks``)
     and the "tail" region (partitions ``[direct + kEarlyReverbMinSizeBlocks,
     n_partitions-1]``).
  2. Compute the per-partition energy ``E[p] = sum(|W[p]|²)`` (Parseval-
     equivalent to the time-domain block energy AEC3 sums).
  3. Linear-regress ``log2(E[p])`` over the tail partitions ``p``. The slope
     ``s`` (in log2 per partition) maps to a per-sample decay
     ``decay = 2 ^ (s / hop_size_samples)`` and is clipped to
     ``[kMinDecay, kMaxDecay] = [0.02, 0.95]``.
  4. EMA-smooth the new estimate with the existing ``decay_`` using
     ``smoothing_constant = filter_quality * 0.2`` (matches AEC3).
  5. Returns the smoothed scalar via ``decay()``.

When the tail region is shorter than ``min_tail_partitions = 2`` partitions
the estimator returns the configured ``default_decay`` and does not update.
"""
from __future__ import annotations

import math

import numpy as np

_K_EARLY_REVERB_MIN_SIZE_BLOCKS = 3
_K_MIN_DECAY = 0.02
_K_MAX_DECAY = 0.95


class ReverbDecayEstimator:
    """Adaptive per-hop reverb decay scalar."""

    def __init__(self, n_partitions: int, hop_size: int,
                 default_decay: float = 0.85,
                 mild_decay: float = 0.5,
                 use_adaptive: bool = True) -> None:
        self._n_partitions = int(n_partitions)
        self._hop_size = int(hop_size)
        self._default_decay = float(default_decay)
        self._mild_decay = float(mild_decay)
        self._use_adaptive = bool(use_adaptive)
        self._decay: float = float(default_decay)
        self._smoothing_constant: float = 0.0

    @property
    def decay_value(self) -> float:
        return self._decay

    def decay(self, mild: bool) -> float:
        """Mirror ``ReverbDecayEstimator::Decay(bool mild)``.

        AEC3: adaptive mode ignores ``mild``; static mode toggles between two
        constants. Our port keeps the AEC3 semantic for diagnostic parity.
        """
        if self._use_adaptive:
            return self._decay
        return self._mild_decay if mild else self._default_decay

    def reset(self) -> None:
        self._decay = self._default_decay
        self._smoothing_constant = 0.0

    def update(self, partition_energies: np.ndarray,
               filter_quality: float | None,
               filter_delay_blocks: int,
               usable_linear_filter: bool,
               stationary_signal: bool) -> None:
        """Refresh the adaptive decay scalar.

        ``partition_energies`` shape: ``(n_partitions,)`` float32 — per-
        partition energy (= ``sum(|W[p]|²)``).
        ``filter_quality``: scalar in [0, 1] (or None) — convergence
        confidence used to gate + smooth the estimate.
        ``filter_delay_blocks``: partition index of the direct path peak.
        ``usable_linear_filter`` / ``stationary_signal``: AEC3 gates from
        ``aec_state``; both must allow update before adaptation runs.
        """
        if stationary_signal:
            return
        if not self._use_adaptive:
            return
        if not usable_linear_filter:
            return
        if filter_delay_blocks < 0:
            return
        if filter_delay_blocks > (
            self._n_partitions - _K_EARLY_REVERB_MIN_SIZE_BLOCKS - 1
        ):
            return

        new_smoothing = (filter_quality or 0.0) * 0.2
        self._smoothing_constant = max(new_smoothing, self._smoothing_constant)
        if self._smoothing_constant == 0.0:
            return

        late_start = filter_delay_blocks + _K_EARLY_REVERB_MIN_SIZE_BLOCKS
        late_end = self._n_partitions  # exclusive
        if (late_end - late_start) < 2:
            return  # too few partitions for a stable linear fit

        e = partition_energies[late_start:late_end].astype(np.float64)
        e = np.maximum(e, 1e-30)
        log2_e = np.log2(e)
        # Linear regression: y = a + b * x → b = cov(x,y) / var(x).
        x = np.arange(log2_e.size, dtype=np.float64)
        x_mean = x.mean()
        y_mean = log2_e.mean()
        num = float(np.sum((x - x_mean) * (log2_e - y_mean)))
        den = float(np.sum((x - x_mean) ** 2))
        if den == 0.0:
            return
        slope_per_partition = num / den

        # Convert per-partition log2 slope to per-sample decay multiplier:
        # E[p+1] = E[p] * decay^hop_size  =>  log2(E[p+1]/E[p]) = hop * log2(decay)
        # slope_per_partition = log2(decay) * hop_size
        decay_log2_per_sample = slope_per_partition / max(1, self._hop_size)
        decay_new = float(math.pow(2.0, decay_log2_per_sample))
        # AEC3 cc:195-197 — clip and prevent over-fast collapse.
        decay_new = max(0.97 * self._decay, decay_new)
        decay_new = min(decay_new, _K_MAX_DECAY)
        decay_new = max(decay_new, _K_MIN_DECAY)

        self._decay += self._smoothing_constant * (decay_new - self._decay)
        # Stop estimation until another "good filter" pulse arrives.
        self._smoothing_constant = 0.0


__all__ = ["ReverbDecayEstimator"]
