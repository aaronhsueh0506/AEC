"""AEC algorithm modules.

This package holds the AEC algorithm implementation split out of the
monolithic ``aec.py`` orchestrator shell.

Layout (target, populated incrementally during refactor R.1 → R.11):

* ``constants``        — module-level constants (`_BLEND_F31_MIC_EXCESS` etc.)
* ``enums``            — `AecMode`, `AecPreset`, `AecFilterState`, `AecEventType`
* ``dataclasses``      — `AecStats`, `AecResContext`, `RenderActivityState`,
                         `FilterConvergenceState`, `RegimeHandlerDecision`,
                         `AecEvent`, `EpcEvent`
* ``config``           — `AecConfig` and 5-preset ``from_preset()`` classmethod
* ``preprocessing``    — `HighPassFilter`, `SaturationDetector`
* ``erle``             — `FilterErleEstimator`, `FullbandErleEstimator`,
                         `compute_erle_confidence`
* ``delay/``           — AEC3-aligned `EchoPathDelayEstimator` +
                         `LegacyDelayShim` (`DelayEstimator` re-export)
* ``filters``          — `NlmsFilter`, `PBFDAF`, `PBFDKF`
* ``detectors``        — `RenderActivityDetector`,
                         `FilterConvergenceAnalyzer`, `DoubleTalkAnalyzer`,
                         `FilterPlateauDetector`
* ``dtd``              — `DtdEstimator`
* ``epc``              — `EchoPathChangeDetector`, `PathChangeRegimeHandler`,
                         `classify_epc_event`
* ``state``            — `AecState`
* ``residual_estimator`` — `ResidualEchoEstimator`
* ``nlp``              — `SubtractiveNLP`
* ``debug_logger``     — `AecDebugLogger`
* ``orchestrator``     — `AEC`, `process_wav_files`, `main`
* ``filter_analyzer``  — `FilterAnalyzer` (audit-only, v3.18 Phase C.A)
* ``filter_quality``   — `FilteringQualityAnalyzer` (v3.18 Phase C.B)
* ``p52_regime_classifier`` — `AcousticRegimeClassifier` (analysis-only)

The shim at ``python/aec.py`` re-exports every public symbol from this
package for backward compat with existing callers
(``eval_aec_challenge.py``, ``run_one_case.py``, tests, etc.).
"""
