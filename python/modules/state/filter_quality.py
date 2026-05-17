"""AecState::FilteringQualityAnalyzer — the 4-gate AND that drives
``UsableLinearEstimate()``.

Mirrors docs/aec3_extracts/src/aec3/aec_state.cc:390-444 verbatim with
hop-rescaled thresholds.

The 4 gates (all must pass for the linear estimate to be usable):
  1. Sufficient data since startup: filter_update_blocks_since_start >
     0.4 * kNumBlocksPerSecond  (AEC3 ~100 blocks ~400 ms -> our 40 hops).
  2. Sufficient data since last reset: filter_update_blocks_since_reset >
     0.2 * kNumBlocksPerSecond  (AEC3 ~50 blocks ~200 ms -> our 20 hops).
  3. Convergence ever seen OR external delay present.
  4. NOT in transparent mode.
"""
from typing import Optional

from ._constants import HOPS_PER_SECOND
from ..delay.delay_types import DelayEstimate


_STARTUP_HOPS = int(0.4 * HOPS_PER_SECOND)  # AEC3 0.4*250=100 blocks (~400 ms) -> 40 hops
_RESET_HOPS = int(0.2 * HOPS_PER_SECOND)    # AEC3 0.2*250=50 blocks (~200 ms) -> 20 hops


class FilteringQualityAnalyzer:
    def __init__(self, *, use_linear_filter: bool = True) -> None:
        self._use_linear_filter = bool(use_linear_filter)
        self._overall_usable = False
        self._filter_update_blocks_since_reset = 0
        self._filter_update_blocks_since_start = 0
        self._convergence_seen = False

    def reset(self) -> None:
        self._overall_usable = False
        self._filter_update_blocks_since_reset = 0
        # NOTE: filter_update_blocks_since_start is NOT reset (matches AEC3).
        # NOTE: convergence_seen latches, NOT reset (matches AEC3).

    def update(
        self,
        *,
        active_render: bool,
        transparent_mode: bool,
        saturated_capture: bool,
        external_delay: Optional[DelayEstimate],
        any_filter_converged: bool,
    ) -> None:
        filter_update = active_render and not saturated_capture
        if filter_update:
            self._filter_update_blocks_since_reset += 1
            self._filter_update_blocks_since_start += 1
        if any_filter_converged:
            self._convergence_seen = True

        startup_ok = self._filter_update_blocks_since_start > _STARTUP_HOPS
        reset_ok = startup_ok and self._filter_update_blocks_since_reset > _RESET_HOPS
        usable = startup_ok and reset_ok
        usable = usable and (external_delay is not None or self._convergence_seen)
        usable = usable and not transparent_mode
        self._overall_usable = usable

    def linear_filter_usable(self) -> bool:
        return self._overall_usable and self._use_linear_filter
