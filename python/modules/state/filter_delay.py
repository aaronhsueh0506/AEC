"""AecState::FilterDelay — caches external delay + filter-analyzer per-channel
direct-path delays.

Mirrors docs/aec3_extracts/src/aec3/aec_state.cc:355-388 verbatim with
hop-rescaled threshold.

Reports delay in OUR hop units (matches what RenderDelayController emits).
``delay_headroom_blocks`` translates AEC3's ``delay_headroom_samples`` /
``kBlockSize`` to OUR ``delay_headroom_samples`` // ``HOP_SAMPLES``.
"""
from typing import Optional

from ._constants import HOPS_PER_SECOND
from .._rates import HOP_SAMPLES
from ..delay.delay_types import DelayEstimate


_FILTER_ADAPTATION_THRESHOLD_HOPS = 2 * HOPS_PER_SECOND  # AEC3 2 s (=500 blocks) -> 200 hops


class FilterDelay:
    """Direct-path filter delay state.

    During the first ~2 s of adaptation OR when no filter-analyzer delay
    is available, falls back to the AEC3 "delay guess" = delay_headroom.

    For our port the per-channel filter analyzer is not yet implemented;
    we expose ``update`` with a None analyzer_delays argument which leaves
    the cached estimate untouched (effectively meaning: trust the external
    delay).
    """

    def __init__(
        self,
        *,
        delay_headroom_samples: int = 32,
        num_capture_channels: int = 1,
    ) -> None:
        self._delay_headroom_blocks = int(delay_headroom_samples) // HOP_SAMPLES
        self._filter_delays_blocks = [self._delay_headroom_blocks] * num_capture_channels
        self._min_filter_delay = self._delay_headroom_blocks
        self._external_delay: Optional[DelayEstimate] = None

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
            blocks_with_proper_filter_adaptation < _FILTER_ADAPTATION_THRESHOLD_HOPS
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

    def external_delay_blocks(self) -> Optional[DelayEstimate]:
        return self._external_delay

    def direct_path_filter_delays(self) -> list[int]:
        return list(self._filter_delays_blocks)

    def min_direct_path_filter_delay(self) -> int:
        return self._min_filter_delay
