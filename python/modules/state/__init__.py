"""AEC3-aligned per-frame state machine (Phase 3 of v3.21 cycle).

Phase 3.1 (in this commit): AecState skeleton + 4 helpers
(InitialState / FilterDelay / FilteringQualityAnalyzer /
SaturationDetector). ERLE / ERL / TransparentMode / Reverb stubbed.

Phase 3.2-3.4: fill the stubs.
Phase 3.5: orchestrator wiring + standalone smoke.
"""
from .aec_state import AecState, AecStateConfig
from .erl_estimator import ErlEstimator
from .erle_estimator import ErleEstimator
from .filter_delay import FilterDelay
from .filter_quality import FilteringQualityAnalyzer
from .initial_state import InitialState
from .saturation_detector import SaturationDetector
from .transparent_mode import TransparentMode

__all__ = [
    "AecState",
    "AecStateConfig",
    "ErlEstimator",
    "ErleEstimator",
    "FilterDelay",
    "FilteringQualityAnalyzer",
    "InitialState",
    "SaturationDetector",
    "TransparentMode",
]
