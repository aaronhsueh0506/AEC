"""AecState::FilteringQualityAnalyzer — the 4-gate AND that drives
``UsableLinearEstimate()``.

Mirrors docs/aec3_extracts/src/aec3/aec_state.cc:390-444 verbatim with
hop-rescaled thresholds.

The 4 gates (all must pass for the linear estimate to be usable):
  1. Sufficient data since startup: filter_update_blocks_since_start >
     0.4 * kNumBlocksPerSecond  (AEC3 ~100 blocks, ~400 ms wall-clock).
  2. Sufficient data since last reset: filter_update_blocks_since_reset >
     0.2 * kNumBlocksPerSecond  (AEC3 ~50 blocks, ~200 ms wall-clock).
  3. Convergence ever seen OR external delay present.
  4. NOT in transparent mode.

Both hop counts are computed at construction time from the live hop_size/
sample_rate (see __init__), not a fixed hop assumption.
"""
from typing import Optional

from .. import aec3_scale as _aec3_scale
from ..delay.delay_types import DelayEstimate


class FilteringQualityAnalyzer:
    def __init__(
        self,
        *,
        use_linear_filter: bool = True,
        hop_size: int = 160,
        sample_rate: int = 16000,
    ) -> None:
        self._use_linear_filter = bool(use_linear_filter)
        self._overall_usable = False
        self._filter_update_blocks_since_reset = 0
        self._filter_update_blocks_since_start = 0
        self._convergence_seen = False
        # Counter retained for diagnostic readers (always increments with
        # convergence_seen; no behaviour gate consumes it).
        self._convergence_hops_counter = 0
        # AEC3 ~100/50 blocks (~400/200 ms). Was module-level constants
        # (`HOPS_PER_SECOND` from state/_constants.py, baked at the stale
        # hop=160/sr=16000 assumption); computed live here instead.
        self._startup_hops = _aec3_scale.ms_to_hops(400.0, hop_size, sample_rate)
        self._reset_hops = _aec3_scale.ms_to_hops(200.0, hop_size, sample_rate)

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
        filter_analyzer_consistent: bool = False,  # accepted as no-op
    ) -> None:
        filter_update = active_render and not saturated_capture
        if filter_update:
            self._filter_update_blocks_since_reset += 1
            self._filter_update_blocks_since_start += 1
        if any_filter_converged:
            self._convergence_seen = True
            self._convergence_hops_counter += 1

        startup_ok = self._filter_update_blocks_since_start > self._startup_hops
        reset_ok = startup_ok and self._filter_update_blocks_since_reset > self._reset_hops
        usable = startup_ok and reset_ok
        # Gate 3: convergence ever seen OR external delay present (AEC3 default).
        gate3 = (external_delay is not None) or self._convergence_seen
        usable = usable and gate3 and not transparent_mode
        self._overall_usable = usable

    def linear_filter_usable(self) -> bool:
        return self._overall_usable and self._use_linear_filter

    # --- diagnostic accessors (trace_hf_chain reads; no audio path) ----------

    def startup_blocks(self) -> int:
        """Blocks since AEC session start (gate-1 numerator)."""
        return self._filter_update_blocks_since_start

    def reset_blocks(self) -> int:
        """Blocks since last reset (gate-2 numerator)."""
        return self._filter_update_blocks_since_reset

    def gate1_pass(self) -> bool:
        return self._filter_update_blocks_since_start > self._startup_hops

    def gate2_pass(self) -> bool:
        return (self._filter_update_blocks_since_start > self._startup_hops
                and self._filter_update_blocks_since_reset > self._reset_hops)
