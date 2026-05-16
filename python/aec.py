"""AEC algorithm — top-level shim.

The algorithm itself lives under ``python/modules/``:

* ``modules.config`` — AecConfig + 5 presets
* ``modules.orchestrator`` — AEC engine class + process_wav_files + main
* ``modules.res_filter`` — ResFilter + ResFilterEnr (production) + ResFilterWiener
* ``modules.{enums, dataclasses, delay, erle, preprocessing, filters,
  detectors, dtd, epc, state, residual_estimator, nlp, debug_logger}``
  — leaf modules

This file re-exports every public symbol so existing callers can keep
``from aec import AEC, AecConfig, AecMode, AecPreset, ResFilter,
PBFDKF, PathChangeRegimeHandler, RegimeHandlerDecision, ...`` unchanged.

CLI entry point: ``python3 python/aec.py mic.wav ref.wav out.wav --preset balanced``
"""
__version__ = "3.15.0"

# v3.19 refactor R.3-R.11 — algorithm classes live under python/modules/.
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
from modules.delay import DelayEstimator  # noqa: F401
from modules.filters import NlmsFilter, PBFDAF, PBFDKF  # noqa: F401
from modules.detectors import (  # noqa: F401
    RenderActivityDetector, FilterConvergenceAnalyzer,
    DoubleTalkAnalyzer, FilterPlateauDetector,
)
from modules.dtd import DtdEstimator  # noqa: F401
from modules.epc import (  # noqa: F401
    classify_epc_event, EchoPathChangeDetector, PathChangeRegimeHandler,
)
from modules.state import AecState  # noqa: F401
from modules.residual_estimator import ResidualEchoEstimator  # noqa: F401
from modules.res_filter import ResFilter, ResFilterEnr, ResFilterWiener  # noqa: F401
from modules.nlp import SubtractiveNLP  # noqa: F401
from modules.debug_logger import AecDebugLogger  # noqa: F401
from modules.orchestrator import AEC, process_wav_files, main  # noqa: F401


if __name__ == "__main__":
    main()
