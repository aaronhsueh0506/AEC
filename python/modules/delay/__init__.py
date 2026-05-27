"""AEC3-aligned delay subsystem (matched_filter + lag_aggregator + clockdrift
+ echo_path_delay_estimator + render_delay_controller).

Replaces the legacy GCC-PHAT ``python/modules/delay.py`` shim.

Entry point: ``RenderDelayController(...)``. Per-hop call:

    controller = RenderDelayController()
    delay_est, variability = controller.get_delay(render_hop, capture_hop)
    # delay_est: Optional[DelayEstimate]; .delay is in OUR 10 ms hop units.
    # variability.delay_change: NONE | BUFFER_FLUSH | NEW_DETECTED_DELAY.
"""
from .clockdrift_detector import ClockdriftDetector, ClockdriftLevel
from .delay_types import DelayAdjustment, DelayEstimate, DelayQuality, EchoPathVariability
from .downsampled_ring import Decimator, DownsampledRenderBuffer
from .echo_path_delay_estimator import EchoPathDelayEstimator
from .lag_aggregator import DelaySelectionThresholds, MatchedFilterLagAggregator
from .matched_filter import LagEstimate, MatchedFilter, matched_filter_core, max_square_peak_index
from .render_delay_controller import RenderDelayController

__all__ = [
    "ClockdriftDetector",
    "ClockdriftLevel",
    "Decimator",
    "DelayAdjustment",
    "DelayEstimate",
    "DelayQuality",
    "DelaySelectionThresholds",
    "DownsampledRenderBuffer",
    "EchoPathDelayEstimator",
    "EchoPathVariability",
    "LagEstimate",
    "MatchedFilter",
    "MatchedFilterLagAggregator",
    "RenderDelayController",
    "matched_filter_core",
    "max_square_peak_index",
]
