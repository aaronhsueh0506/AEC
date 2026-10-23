"""FilterDelay: caches the external delay and the filter analyzer's direct-path delays, in hops."""
from typing import Optional

from .. import aec3_scale as _aec3_scale
from ..delay.delay_types import DelayEstimate


class FilterDelay:
    """Direct-path filter delay state.

    During the first 2 s of adaptation, or without analyzer delays, the delay
    defaults to the delay headroom. A ``None`` analyzer argument keeps the cached estimate.
    """

    def __init__(
        self,
        *,
        delay_headroom_samples: int = 32,
        num_capture_channels: int = 1,
        hop_size: int = 160,
        sample_rate: int = 16000,
    ) -> None:
        self._delay_headroom_blocks = int(delay_headroom_samples) // hop_size
        self._filter_delays_blocks = [self._delay_headroom_blocks] * num_capture_channels
        self._min_filter_delay = self._delay_headroom_blocks
        self._external_delay: Optional[DelayEstimate] = None
        # 2 s filter-adaptation threshold at the live grid.
        self._filter_adaptation_threshold_hops = _aec3_scale.ms_to_hops(
            2000.0, hop_size, sample_rate
        )

    def update(
        self,
        *,
        analyzer_filter_delay_estimates_blocks: Optional[list[int]],
        external_delay: Optional[DelayEstimate],
        blocks_with_proper_filter_adaptation: int,
    ) -> None:
        if external_delay is not None:
            self._external_delay = external_delay

        delay_estimator_unconverged = (
            blocks_with_proper_filter_adaptation < self._filter_adaptation_threshold_hops
        )
        if delay_estimator_unconverged and self._external_delay is not None:
            self._filter_delays_blocks = [self._delay_headroom_blocks] * len(
                self._filter_delays_blocks
            )
        elif analyzer_filter_delay_estimates_blocks is not None:
            if len(analyzer_filter_delay_estimates_blocks) != len(self._filter_delays_blocks):
                raise ValueError("analyzer delays length mismatch")
            self._filter_delays_blocks = list(analyzer_filter_delay_estimates_blocks)

        self._min_filter_delay = min(self._filter_delays_blocks)

    def external_delay_reported(self) -> bool:
        return self._external_delay is not None

    def external_delay_samples(self) -> Optional[DelayEstimate]:
        return self._external_delay

    def direct_path_filter_delays(self) -> list[int]:
        return list(self._filter_delays_blocks)

    def min_direct_path_filter_delay(self) -> int:
        return self._min_filter_delay
