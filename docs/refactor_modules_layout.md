# `python/modules/` layout (v3.21.0)

**Refactor lineage**: the v3.19 R.0–R.12 cycle (2026-05-16) split the
monolithic 9 660-line `python/aec.py` into a 52-line shim plus 17
algorithm modules under `python/modules/`. The v3.21 cleanup cycle
(2026-05-19) then retired the legacy `ResFilter` chain and reorganised
the post-filter into AEC3-aligned subpackages.

## Layout

```
python/
├── aec.py                       # top-level shim re-exporting public symbols
├── eval_aec_challenge.py        # 800-case AEC Challenge render driver
├── bench_aecmos.py              # AECMOS scoring driver
├── check_byte_equal.py          # 25-case byte-equal regression harness
├── run_one_case.py              # single-case dev tool (5-panel diagnostic PNG)
├── run_e2e_parity.py            # Python ↔ C parity driver
├── batch_c_eval.py              # C binary batch driver
├── test_p52_regime.py           # pytest for the regime classifier anti-loophole contract
└── modules/
    ├── __init__.py
    ├── _rates.py                # block / FFT / hop / ms helpers (AEC3 rescale)
    ├── aec3_scale.py            # AEC3 ↔ our-hop conversion helpers
    ├── config.py                # AecConfig dataclass + from_preset (BALANCED)
    ├── dataclasses.py           # AecStats / AecResContext / per-frame state tuples
    ├── debug_logger.py          # --diag console logger
    ├── enums.py                 # AecMode / AecPreset / AecFilterState
    ├── orchestrator.py          # AEC engine class + process_wav_files + main()
    ├── preprocessing.py         # HighPassFilter + SaturationDetector
    ├── erle.py                  # FilterErleEstimator + FullbandErleEstimator
    ├── filters.py               # NlmsFilter + PBFDAF + PBFDKF
    ├── dtd.py                   # DtdEstimator
    ├── detectors.py             # RenderActivityDetector + FilterConvergenceAnalyzer + DoubleTalkAnalyzer + FilterPlateauDetector
    ├── epc.py                   # EchoPathChangeDetector + PathChangeRegimeHandler + classify_epc_event
    ├── nlp.py                   # SubtractiveNLP (v3.13 E4 substrate, default-OFF)
    ├── filter_analyzer.py       # FilterAnalyzer (impulse-response shape, audit-only)
    ├── p52_regime_classifier.py # AcousticRegimeClassifier (analysis-only)
    ├── residual_estimator.py    # ResidualEchoEstimator (legacy compatibility seam)
    ├── delay/                   # AEC3-aligned delay estimation subpackage
    │   ├── echo_path_delay_estimator.py
    │   ├── matched_filter.py
    │   ├── lag_aggregator.py
    │   ├── clockdrift_detector.py
    │   ├── downsampled_ring.py
    │   ├── render_delay_controller.py
    │   ├── delay_types.py
    │   └── legacy_compat.py     # LegacyDelayShim — exposes legacy DelayEstimator API
    ├── filter/                  # filter subpackage (state bridge)
    │   └── filter_state_bridge.py
    ├── render/                  # render-side analysis subpackage
    │   └── render_signal_analyzer.py
    ├── residual/                # AEC3-aligned residual + suppression subpackage
    │   ├── residual_echo_estimator.py
    │   ├── reverb_model.py
    │   ├── reverb_decay_estimator.py
    │   ├── reverb_frequency_response.py
    │   ├── suppression_gain.py
    │   └── suppression_filter.py
    └── state/                   # AEC3-aligned AecState ADT + 12 sub-analyzers
        ├── aec_state.py
        ├── _constants.py
        ├── erl_estimator.py
        ├── erle_estimator.py
        ├── filter_analyzer.py
        ├── filter_delay.py
        ├── filter_quality.py
        ├── fullband_erle.py
        ├── initial_state.py
        ├── saturation_detector.py
        ├── stationarity_estimator.py
        ├── subband_erle.py
        └── transparent_mode.py
```

## Module purpose (one line each)

| Module | Purpose |
|---|---|
| `aec.py` | Top-level shim re-exporting public symbols. `__version__ = "3.21.0"`. |
| `modules/config.py` | `AecConfig` dataclass + `from_preset(BALANCED)`. |
| `modules/orchestrator.py` | `AEC` engine class + `_aec3_post` + `process_wav_files` + `main`. |
| `modules/enums.py` | `AecMode` / `AecPreset` / `AecFilterState`. |
| `modules/dataclasses.py` | `AecStats` / `AecResContext` / per-frame state tuples. |
| `modules/preprocessing.py` | `HighPassFilter` (80 Hz IIR) + `SaturationDetector`. |
| `modules/filters.py` | `NlmsFilter` + `PBFDAF` (NLMS shadow) + `PBFDKF` (Kalman refined). |
| `modules/dtd.py` | `DtdEstimator` (DTD coherence + energy detector). |
| `modules/detectors.py` | `RenderActivityDetector` + `FilterConvergenceAnalyzer` + `DoubleTalkAnalyzer` + `FilterPlateauDetector`. |
| `modules/epc.py` | `EchoPathChangeDetector` + `PathChangeRegimeHandler` (formerly `ShadowCopyController`) + `classify_epc_event`. |
| `modules/erle.py` | `FilterErleEstimator` + `FullbandErleEstimator`. |
| `modules/nlp.py` | `SubtractiveNLP` (v3.13 E4 substrate). |
| `modules/filter_analyzer.py` | `FilterAnalyzer` (audit-only filter impulse-response port). |
| `modules/p52_regime_classifier.py` | `AcousticRegimeClassifier` (analysis-only). |
| `modules/residual_estimator.py` | Legacy `ResidualEchoEstimator` (compatibility seam). |
| `modules/delay/` | AEC3-aligned `EchoPathDelayEstimator` + matched filter / lag aggregator / clockdrift detector / render-delay controller + `LegacyDelayShim`. |
| `modules/filter/filter_state_bridge.py` | Read-only seam exposing refined-filter spectra / convergence state to the AEC3 post-filter. |
| `modules/render/render_signal_analyzer.py` | Per-bin narrowband-tonal mask + `poor_signal_excitation` gate. |
| `modules/residual/` | AEC3-aligned `ResidualEchoEstimator` (new) + `SuppressionGain` + `SuppressionFilter` + `ReverbModel` + `ReverbDecayEstimator` + `ReverbFrequencyResponse`. |
| `modules/state/` | `AecState` ADT + 12 sub-analyzers (FilterAnalyzer, FilteringQualityAnalyzer, SubbandErleEstimator, ErleEstimator, ErlEstimator, FullbandErleEstimator, SaturationDetector, InitialState, TransparentMode, StationarityEstimator, FilterDelay). |
| `modules/_rates.py` + `modules/aec3_scale.py` | Block / FFT / hop / ms helpers + AEC3 ↔ our-hop conversion. |

## v3.21 deletions

The v3.21 cleanup cycle deleted the following modules:

| Path | Reason |
|---|---|
| `python/modules/res_filter.py` (2 221 LOC) | Legacy 9-stage `ResFilter` chain, retired in R7. |
| `python/modules/res_refactored/` (8 files, 697 LOC) | P52 Phase B subclass-and-delegate scaffold; never promoted. |
| `python/modules/legacy_state.py` (156 LOC) | Legacy `AecState` aggregator; superseded by `modules/state/aec_state.py`. |
| `python/modules/legacy_delay.py` (280 LOC) | Legacy GCC-PHAT `DelayEstimator`; superseded by `modules/delay/`. |
| `python/modules/filter_quality.py` (top-level orphan) | Superseded by `modules/state/filter_quality.py`. |
| `python/test_f3_1_mic_excess.py` | Tested `ResFilter._stage_gain_compute` mic-excess branch; gone with ResFilter. |
| `python/diagnose_gcc_phat.py` | Research-only GCC-PHAT diagnostic tool, not on production path. |

Net Python LOC delta v3.10.5 → v3.21.0: **~−5 500** (mostly ResFilter
retirement; dead substrate / config flag sweep accounts for the rest).

## Backward compat

`python/aec.py` re-exports every public symbol so existing callers can
keep using `from aec import AEC, AecConfig, AecMode, AecPreset, PBFDKF,
PathChangeRegimeHandler, DelayEstimator, ResidualEchoEstimator, ...`
unchanged. The `DelayEstimator` re-export now points at
`modules.delay.legacy_compat.LegacyDelayShim`, which wraps the AEC3
`EchoPathDelayEstimator` with the historical `accumulate()` API.

CLI entry point preserved via `if __name__ == "__main__": main()` in the
shim, forwarding to `modules.orchestrator.main`.

## Mirrored C-side

The C port under `c_impl/` mirrors the Python class structure
(`PBFDKF`, `ShadowFilter`, `HighPassFilter`, etc.). v3.21 retired the
`res_filter.{h,c}` files on the Python side; the C port retains the
legacy ResFilter for the in-flight C cycle and will be re-aligned to
the AEC3 chain in a separate v3.21.x C-port arc. See
[`c_user_and_integration_guide.md`](c_user_and_integration_guide.md)
for the C integration contract.
