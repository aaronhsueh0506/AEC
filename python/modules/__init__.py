"""AEC algorithm modules.

Layout:

* ``enums``            — `AecMode`, `AecPreset`, `AecFilterState`, `AecEventType`
* ``dataclasses``      — `AecStats`, `AecResContext`, `RenderActivityState`,
                         `FilterConvergenceState`, `RegimeHandlerDecision`,
                         `AecEvent`, `EpcEvent`
* ``config``           — `AecConfig` and ``from_preset()`` classmethod
* ``preprocessing``    — `HighPassFilter`, `SaturationDetector`
* ``delay/``           — AEC3-aligned `EchoPathDelayEstimator` +
                         `LegacyDelayShim` (`DelayEstimator` re-export)
* ``filters``          — `PBFDAF`, `PBFDKF`
* ``detectors``        — `RenderActivityDetector`,
                         `FilterConvergenceAnalyzer`, `DoubleTalkAnalyzer`,
                         `FilterPlateauDetector`
* ``epc``              — `EchoPathChangeDetector`, `PathChangeRegimeHandler`,
                         `classify_epc_event`
* ``state``            — `AecState` (+ AEC3-aligned analyzers)
* ``residual``         — `ResidualEchoEstimator`, `SuppressionGain`
* ``filter``           — refined-filter substrate / bridges
* ``render``           — `RenderSignalAnalyzer`
* ``debug_logger``     — `AecDebugLogger`
* ``orchestrator``     — `AEC`, `process_wav_files`, `main`
* ``p52_regime_classifier`` — `AcousticRegimeClassifier` (analysis-only)

The shim at ``python/aec.py`` re-exports every public symbol from this
package for backward compat with existing callers.
"""
