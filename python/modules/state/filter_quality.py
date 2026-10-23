"""FilteringQualityAnalyzer: the gate set behind ``linear_filter_usable()``.

All must pass: (1) more than 400 ms of filter updates since start; (2) more than
200 ms since the last reset; (3) convergence seen or an external delay present;
(4) not in transparent mode. Thresholds are converted to hops from the live grid.
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
        # Diagnostic counter; no gate reads it.
        self._convergence_hops_counter = 0
        self._startup_hops = _aec3_scale.ms_to_hops(400.0, hop_size, sample_rate)
        self._reset_hops = _aec3_scale.ms_to_hops(200.0, hop_size, sample_rate)

    def reset(self) -> None:
        self._overall_usable = False
        self._filter_update_blocks_since_reset = 0
        # since_start and convergence_seen are not reset.

    def update(
        self,
        *,
        active_render: bool,
        transparent_mode: bool,
        saturated_capture: bool,
        external_delay: Optional[DelayEstimate],
        any_filter_converged: bool,
        filter_analyzer_consistent: bool = False,  # unused
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
        gate3 = (external_delay is not None) or self._convergence_seen
        usable = usable and gate3 and not transparent_mode
        self._overall_usable = usable

    def linear_filter_usable(self) -> bool:
        return self._overall_usable and self._use_linear_filter


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
