"""AEC3-style delay subsystem: matched filter, lag aggregator, clock drift, echo-path delay estimator.

``orchestrator.py`` uses ``LegacyDelayShim`` over ``EchoPathDelayEstimator``.
``RenderDelayController`` is a block-quantised reference implementation with
fixed 16 kHz / 10 ms constants.
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
