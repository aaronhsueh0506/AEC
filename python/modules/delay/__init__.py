"""AEC3-aligned delay subsystem (matched_filter + lag_aggregator + clockdrift
+ echo_path_delay_estimator + render_delay_controller).

Replaces the legacy GCC-PHAT ``python/modules/delay.py`` shim.

Production entry point: ``orchestrator.py`` uses ``LegacyDelayShim`` (see
``legacy_compat.py``) directly over ``EchoPathDelayEstimator``, NOT
``RenderDelayController`` -- the latter quantises delay to hop boundaries,
losing the sub-hop sample precision the orchestrator's ring buffer needs.
``RenderDelayController`` is exported below as an AEC3-structural
reference/for future migrations; it has no live caller, no dedicated test
coverage, and its rate constants are a fixed legacy 16 kHz/10 ms reference
that does not track the current per-instance grid (see its own docstring
before using it in anything new).
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
