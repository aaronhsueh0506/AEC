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

v3.21.6 nores LF artifact debug 2026-05-22 — Gate-3 ablation knobs
([[project-usable-linear-gate3-latch-bug]]). The legacy gate 3 is too
permissive on no-clean-convergence stress cases (XRTnTUjU_DT_static):
  - `convergence_seen` is a binary latch (no reset)
  - `external_delay present` is a shortcut that ignores actual filter
    convergence

Three variants ship as default-OFF knobs so the legacy byte-equal path
stays unchanged:
  - `convergence_hops_required` (default 0 = legacy latch behaviour).
    When > 0, gate 3 needs `convergence_hops_counter ≥ N` instead of
    "ever fired once". Counter increments on `any_filter_converged`,
    does NOT decrement.
  - `require_filter_analyzer_consistent` (default False). When True,
    gate 3 is AND-gated with `filter_analyzer_consistent` (the AEC3
    FilterAnalyzer port signal). Requires the orchestrator to pass
    `filter_analyzer_consistent` into `update()`; when the analyzer
    is disabled the flag has no effect (no AND on True).
  - `disable_external_delay_shortcut` (default False). When True, the
    `external_delay is not None` branch of gate 3 is dropped — only
    the `convergence_*` branch can satisfy gate 3. Most aggressive
    variant; intended for stress-case audit.
"""
from typing import Optional

from ._constants import HOPS_PER_SECOND
from ..delay.delay_types import DelayEstimate


_STARTUP_HOPS = int(0.4 * HOPS_PER_SECOND)  # AEC3 0.4*250=100 blocks (~400 ms) -> 40 hops
_RESET_HOPS = int(0.2 * HOPS_PER_SECOND)    # AEC3 0.2*250=50 blocks (~200 ms) -> 20 hops


class FilteringQualityAnalyzer:
    def __init__(
        self,
        *,
        use_linear_filter: bool = True,
        # v3.21.6 gate-3 ablation knobs (default-OFF = legacy AEC3 behaviour)
        convergence_hops_required: int = 0,
        require_filter_analyzer_consistent: bool = False,
        disable_external_delay_shortcut: bool = False,
    ) -> None:
        self._use_linear_filter = bool(use_linear_filter)
        self._overall_usable = False
        self._filter_update_blocks_since_reset = 0
        self._filter_update_blocks_since_start = 0
        self._convergence_seen = False
        # v3.21.6 — convergence hops counter (used when
        # convergence_hops_required > 0; otherwise dormant).
        self._convergence_hops_counter = 0
        # Knobs
        self._convergence_hops_required = int(convergence_hops_required)
        self._require_filter_analyzer_consistent = bool(
            require_filter_analyzer_consistent)
        self._disable_external_delay_shortcut = bool(
            disable_external_delay_shortcut)

    def reset(self) -> None:
        self._overall_usable = False
        self._filter_update_blocks_since_reset = 0
        # NOTE: filter_update_blocks_since_start is NOT reset (matches AEC3).
        # NOTE: convergence_seen latches, NOT reset (matches AEC3).
        # NOTE: convergence_hops_counter latches the same way — it's the
        # graduated version of convergence_seen; resetting it on every
        # path-change would defeat the purpose of the counter.

    def update(
        self,
        *,
        active_render: bool,
        transparent_mode: bool,
        saturated_capture: bool,
        external_delay: Optional[DelayEstimate],
        any_filter_converged: bool,
        filter_analyzer_consistent: bool = False,
    ) -> None:
        filter_update = active_render and not saturated_capture
        if filter_update:
            self._filter_update_blocks_since_reset += 1
            self._filter_update_blocks_since_start += 1
        if any_filter_converged:
            self._convergence_seen = True
            self._convergence_hops_counter += 1

        startup_ok = self._filter_update_blocks_since_start > _STARTUP_HOPS
        reset_ok = startup_ok and self._filter_update_blocks_since_reset > _RESET_HOPS
        usable = startup_ok and reset_ok

        # Gate 3 — legacy by default, knobs override:
        # convergence branch (legacy = binary latch; knob = counter ≥ N)
        if self._convergence_hops_required > 0:
            conv_ok = (self._convergence_hops_counter
                       >= self._convergence_hops_required)
        else:
            conv_ok = self._convergence_seen
        # external_delay branch (legacy = shortcut; knob = drop)
        if self._disable_external_delay_shortcut:
            gate3 = conv_ok
        else:
            gate3 = (external_delay is not None) or conv_ok
        # Optional AND with filter_analyzer_consistent
        if self._require_filter_analyzer_consistent:
            gate3 = gate3 and bool(filter_analyzer_consistent)

        usable = usable and gate3
        usable = usable and not transparent_mode
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
        return self._filter_update_blocks_since_start > _STARTUP_HOPS

    def gate2_pass(self) -> bool:
        return (self._filter_update_blocks_since_start > _STARTUP_HOPS
                and self._filter_update_blocks_since_reset > _RESET_HOPS)
