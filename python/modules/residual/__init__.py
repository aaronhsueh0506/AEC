"""AEC3-aligned residual + suppression chain.

Phase 4.1 in this commit: ResidualEchoEstimator (R² = S²/ERLE | X²*gain²).
Phase 4.2 in this commit: SuppressionGain (the AEC3 mask gate + smoother).

Phase 4.3-4.5 follow: suppression_filter, comfort_noise, echo_audibility,
orchestrator trunk reshape.
"""
from .residual_echo_estimator import (
    EchoModelConfig,
    EpStrengthConfig,
    ResidualEchoEstimator,
    ReverbConfig,
)
from .reverb_model import ReverbModel
from .suppression_filter import apply_suppression
from .suppression_gain import (
    DominantNearendConfig,
    EchoAudibilityConfig,
    HighFrequencySuppressionConfig,
    MaskTuning,
    SuppressionGain,
    SuppressorConfig,
    SuppressorTuning,
)

__all__ = [
    "DominantNearendConfig",
    "EchoAudibilityConfig",
    "EchoModelConfig",
    "EpStrengthConfig",
    "HighFrequencySuppressionConfig",
    "MaskTuning",
    "ResidualEchoEstimator",
    "ReverbConfig",
    "ReverbModel",
    "SuppressionGain",
    "SuppressorConfig",
    "SuppressorTuning",
    "apply_suppression",
]
