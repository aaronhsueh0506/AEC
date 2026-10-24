"""AEC top-level module: re-exports the public symbols of ``modules/``.

CLI: ``python3 python/aec.py mic.wav ref.wav out.wav --preset balanced``
"""
__version__ = "5.0.0rc1"

from modules.enums import (  # noqa: F401
    AecMode, AecPreset, AecFilterState,
)
from modules.dataclasses import (  # noqa: F401
    AecStats, AecResContext, RenderActivityState, FilterConvergenceState,
    RegimeHandlerDecision, AecEventType, AecEvent, EpcEvent,
)
from modules.config import AecConfig  # noqa: F401
from modules.preprocessing import HighPassFilter, SaturationDetector  # noqa: F401
from modules.delay.legacy_compat import LegacyDelayShim as DelayEstimator  # noqa: F401
from modules.filters import PBFDAF, PBFDKF  # noqa: F401
from modules.detectors import (  # noqa: F401
    RenderActivityDetector, FilterConvergenceAnalyzer,
    DoubleTalkAnalyzer, FilterPlateauDetector,
)
from modules.epc import (  # noqa: F401
    classify_epc_event, EchoPathChangeDetector, PathChangeRegimeHandler,
)
from modules.residual import ResidualEchoEstimator  # noqa: F401
from modules.debug_logger import AecDebugLogger  # noqa: F401
from modules.orchestrator import AEC, process_wav_files, main  # noqa: F401


if __name__ == "__main__":
    main()
