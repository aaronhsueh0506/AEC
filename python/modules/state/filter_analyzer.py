"""FilterAnalyzer + ConsistentFilterDetector — canonical AEC3 port.

Mirrors docs/aec3_extracts/src/aec3/filter_analyzer.{cc,h} for the
single-channel case (our pipeline has 1 mic + 1 ref).

Replaces the loose proxy `any_filter_converged && external_delay is not
None` that AecState previously used for `any_filter_consistent`. The proxy
fired in 1 hop; the canonical detector requires the impulse-response
shape to be "consistent" (significant peak above 10× floor and 2×
secondary peak) AND the delay-block reference to stay stable for
1.5 seconds under active render. That sustained-shape gate is what makes
TransparentMode's `sane_filter_recently_seen` early-exit gate behave
correctly — a too-loose proxy collapses TM's FS protection.

The port is identical to canonical with the following adaptations:
  * Single channel only (no per-channel loop).
  * `kBlockSize=64` kept as the delay-block unit so AecState's
    `MinDirectPathFilterDelay()` matches canonical thresholds like
    TM's `filter_delay_blocks < 5`.
  * `1.5 * kNumBlocksPerSecond` (canonical 375 blocks) rescaled to our
    hop rate as `1.5 * HOPS_PER_SECOND = 150 hops` because we tick
    Update() once per 10 ms hop, not per 4 ms canonical block.
  * Region paging dropped: we analyze the full filter each call
    (negligible CPU; simpler, no per-region state to maintain).
  * Caller passes a single time-domain impulse-response vector and a
    per-hop render energy float; orchestrator does the PBFDKF→time
    conversion and render-energy reduction.
"""
from dataclasses import dataclass, field

import numpy as np

from ._constants import HOPS_PER_SECOND


_HP_FIR = np.array([0.7929742, -0.36072128, -0.47047766], dtype=np.float32)
_K_BLOCK_SIZE = 64
_K_BLOCK_SIZE_LOG2 = 6
_CONSISTENT_THRESHOLD_HOPS = int(1.5 * HOPS_PER_SECOND)   # 150 hops ≈ 1.5 s
_SUFFICIENT_TIME_HOPS = int(5.0 * HOPS_PER_SECOND)        # 500 hops ≈ 5 s
_DEFAULT_GAIN = 0.01
_ACTIVE_RENDER_LIMIT = 200.0  # canonical EchoCanceller3Config.render_levels.active_render_limit
_ACTIVE_RENDER_THRESHOLD = _ACTIVE_RENDER_LIMIT * _ACTIVE_RENDER_LIMIT * _K_BLOCK_SIZE


def _preprocess_filter(filter_time_domain: np.ndarray) -> np.ndarray:
    """Apply canonical 3-tap FIR HP (~600 Hz cutoff) to filter impulse
    response. Direct port of FilterAnalyzer::PreProcessFilters."""
    h = filter_time_domain.astype(np.float32, copy=False)
    n = h.size
    if n < _HP_FIR.size:
        return np.zeros(n, dtype=np.float32)
    out = np.zeros(n, dtype=np.float32)
    out[_HP_FIR.size - 1:] = np.convolve(h, _HP_FIR, mode='valid')
    return out


@dataclass
class ConsistentFilterDetector:
    """Per-channel filter-shape consistency tracker. Direct port of
    FilterAnalyzer::ConsistentFilterDetector."""

    _significant_peak: bool = field(init=False, default=False)
    _consistent_estimate_counter: int = field(init=False, default=0)
    _consistent_delay_reference: int = field(init=False, default=-10)

    def reset(self) -> None:
        self._significant_peak = False
        self._consistent_estimate_counter = 0
        self._consistent_delay_reference = -10

    def detect(
        self,
        filter_to_analyze: np.ndarray,
        peak_index: int,
        delay_blocks: int,
        render_energy: float,
    ) -> bool:
        """One Update() pass on the full filter (we don't page regions).

        Returns True once `consistent_estimate_counter > 1.5 *
        HOPS_PER_SECOND` of consecutive active-render hops with the same
        delay-block reference.
        """
        n = filter_to_analyze.size
        floor_low = 0 if peak_index < 64 else peak_index - 64
        floor_high = 0 if peak_index > n - 129 else peak_index + 128

        abs_h = np.abs(filter_to_analyze)
        floor_accum = 0.0
        secondary_peak = 0.0
        if floor_low > 0:
            floor_accum += float(abs_h[:floor_low].sum())
            secondary_peak = max(secondary_peak, float(abs_h[:floor_low].max(initial=0.0)))
        if floor_high > 0 and floor_high < n:
            floor_accum += float(abs_h[floor_high:].sum())
            secondary_peak = max(secondary_peak, float(abs_h[floor_high:].max(initial=0.0)))

        denom = floor_low + (n - floor_high)
        if denom <= 0:
            return False
        filter_floor = floor_accum / denom
        abs_peak = float(abs_h[peak_index])
        self._significant_peak = (
            abs_peak > 10.0 * filter_floor and abs_peak > 2.0 * secondary_peak
        )

        if self._significant_peak:
            active_render = render_energy > _ACTIVE_RENDER_THRESHOLD
            if self._consistent_delay_reference == delay_blocks:
                if active_render:
                    self._consistent_estimate_counter += 1
            else:
                self._consistent_estimate_counter = 0
                self._consistent_delay_reference = delay_blocks
        return self._consistent_estimate_counter > _CONSISTENT_THRESHOLD_HOPS


@dataclass
class FilterAnalyzer:
    """Top-level FilterAnalyzer. Direct port (single-channel)."""

    bounded_erl: bool = False
    default_gain: float = _DEFAULT_GAIN
    consistent_estimate: bool = field(init=False, default=False)
    peak_index: int = field(init=False, default=0)
    gain: float = field(init=False, default=_DEFAULT_GAIN)
    filter_delay_blocks: int = field(init=False, default=0)
    _blocks_since_reset: int = field(init=False, default=0)
    _detector: ConsistentFilterDetector = field(
        init=False, default_factory=ConsistentFilterDetector
    )

    def __post_init__(self) -> None:
        self.gain = self.default_gain

    def reset(self) -> None:
        self._blocks_since_reset = 0
        self.peak_index = 0
        self.gain = self.default_gain
        self.consistent_estimate = False
        self.filter_delay_blocks = 0
        self._detector.reset()

    def update(
        self,
        *,
        filter_time_domain: np.ndarray,
        render_energy: float,
    ) -> None:
        """Per-hop update. Caller passes the main filter's effective
        time-domain impulse response and the current render hop energy
        (sum of squares over the hop).
        """
        self._blocks_since_reset += 1
        if filter_time_domain.size == 0:
            return
        h_hp = _preprocess_filter(filter_time_domain)
        abs_h2 = h_hp * h_hp
        self.peak_index = int(np.argmax(abs_h2))
        self.filter_delay_blocks = self.peak_index >> _K_BLOCK_SIZE_LOG2

        sufficient_time = self._blocks_since_reset > _SUFFICIENT_TIME_HOPS
        peak_mag = float(np.abs(h_hp[self.peak_index]))
        if sufficient_time and self.consistent_estimate:
            self.gain = peak_mag
        else:
            if self.gain:
                self.gain = max(self.gain, peak_mag)
        if self.bounded_erl and self.gain:
            self.gain = max(self.gain, 0.01)

        self.consistent_estimate = self._detector.detect(
            filter_to_analyze=h_hp,
            peak_index=self.peak_index,
            delay_blocks=self.filter_delay_blocks,
            render_energy=render_energy,
        )
