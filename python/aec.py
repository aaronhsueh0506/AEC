"""AEC algorithm — top-level shim.

The algorithm itself lives under ``python/modules/``:

* ``modules.config`` — AecConfig + single BALANCED preset
* ``modules.orchestrator`` — AEC engine class + process_wav_files + main
* ``modules.{enums, dataclasses, delay, erle, preprocessing, filters,
  detectors, dtd, epc, state, residual, residual_estimator, render, filter,
  debug_logger}`` — leaf modules

This file re-exports every public symbol so existing callers can keep
``from aec import AEC, AecConfig, AecMode, AecPreset, PBFDKF,
PathChangeRegimeHandler, RegimeHandlerDecision, ...`` unchanged.

CLI entry point: ``python3 python/aec.py mic.wav ref.wav out.wav --preset balanced``
"""
__version__ = "3.21.6.4"

# Algorithm classes live under python/modules/.
from modules.enums import (  # noqa: F401
    AecMode, AecPreset, AecFilterState,
)
from modules.dataclasses import (  # noqa: F401
    AecStats, AecResContext, RenderActivityState, FilterConvergenceState,
    RegimeHandlerDecision, AecEventType, AecEvent, EpcEvent,
)
from modules.config import AecConfig  # noqa: F401
from modules.preprocessing import HighPassFilter, SaturationDetector  # noqa: F401
from modules.erle import (  # noqa: F401
    FilterErleEstimator, FullbandErleEstimator, compute_erle_confidence,
)
from modules.delay.legacy_compat import LegacyDelayShim as DelayEstimator  # noqa: F401
from modules.filters import NlmsFilter, PBFDAF, PBFDKF  # noqa: F401
from modules.detectors import (  # noqa: F401
    RenderActivityDetector, FilterConvergenceAnalyzer,
    DoubleTalkAnalyzer, FilterPlateauDetector,
)
from modules.dtd import DtdEstimator  # noqa: F401
from modules.epc import (  # noqa: F401
    classify_epc_event, EchoPathChangeDetector, PathChangeRegimeHandler,
)
from modules.residual import ResidualEchoEstimator  # noqa: F401
from modules.debug_logger import AecDebugLogger  # noqa: F401
from modules.orchestrator import AEC, process_wav_files, main  # noqa: F401


if __name__ == "__main__":
    main()
