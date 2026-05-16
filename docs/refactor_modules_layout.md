# `python/modules/` layout — v3.19 R.0–R.12 refactor

**Cycle**: 2026-05-16, 13 commits on `feature/v3.18-aec3-fetch`.
**Outcome**: `python/aec.py` shrank from **9,660 lines → 53-line shim**.
17 algorithm modules now live under `python/modules/`.

## Module map

| Module                            | Source rows | Purpose                                                              | Mirrors C-side       |
| --------------------------------- | -----------:| -------------------------------------------------------------------- | -------------------- |
| `enums.py`                        |          92 | `AecMode` / `AecPreset` / `AecFilterState` / `_FREQ_MODES` / `_PB_MODES` | —                |
| `dataclasses.py`                  |         186 | `AecStats` / `AecResContext` / `RenderActivityState` / `FilterConvergenceState` / `RegimeHandlerDecision` / `AecEventType` / `AecEvent` / `EpcEvent` | — |
| `preprocessing.py`                |         124 | `HighPassFilter` (80 Hz IIR) + `SaturationDetector`                  | preprocessing.{h,c}  |
| `erle.py`                         |         103 | `FilterErleEstimator` + `FullbandErleEstimator` + `compute_erle_confidence` | erle.{h,c}    |
| `delay.py`                        |         283 | `DelayEstimator` (GCC-PHAT)                                          | delay.{h,c}          |
| `filters.py`                      |         461 | `NlmsFilter` + `PBFDAF` + `PBFDKF`                                   | filters.{h,c}        |
| `dtd.py`                          |         211 | `DtdEstimator` (own file, mirrors c_impl/dtd.{h,c})                  | dtd.{h,c}            |
| `detectors.py`                    |         342 | `RenderActivityDetector` + `FilterConvergenceAnalyzer` + `DoubleTalkAnalyzer` + `FilterPlateauDetector` | detectors.{h,c} |
| `epc.py`                          |         311 | `classify_epc_event` + `EchoPathChangeDetector` + `PathChangeRegimeHandler` (formerly ShadowCopyController, P52 Phase A) | epc.{h,c} |
| `state.py`                        |         151 | `AecState` (WebRTC AEC3 ADT facade; back-ref via runtime injection)  | state.{h,c}          |
| `residual_estimator.py`           |         247 | `ResidualEchoEstimator` (per-bin residual PSD, 2 modes)              | residual_estimator.{h,c} |
| `res_filter.py`                   |       2,170 | `ResFilter` base + `ResFilterEnr` (production) + `ResFilterWiener` (R.9.1 split) | res_filter.{h,c} |
| `nlp.py`                          |         164 | `SubtractiveNLP` (default-OFF v3.13 E4 substrate)                    | (research-only)      |
| `debug_logger.py`                 |          54 | `AecDebugLogger` (`--diag` console)                                  | (research-only)      |
| `config.py`                       |       1,294 | `AecConfig` + 5-preset `from_preset()`                               | aec_config.{h,c}     |
| `orchestrator.py`                 |       3,747 | `AEC` engine + `process_wav_files` + `main()` + `_BLEND_F31_MIC_EXCESS` | aec.{h,c}        |
| `__init__.py`                     |          16 | Package docstring + intended layout doc                              | —                    |

Plus 4 modules **migrated `git mv`** (R.2):

| Module                            | Origin                                   | Purpose                          |
| --------------------------------- | ---------------------------------------- | -------------------------------- |
| `filter_analyzer.py`              | `python/aec_filter_analyzer.py`          | v3.18 Phase C.A (audit-only)     |
| `filter_quality.py`               | `python/aec_filter_quality.py`           | v3.18 Phase C.B (audit-only)     |
| `p52_regime_classifier.py`        | `python/aec_p52_regime_classifier.py`    | P52 acoustic-regime classifier   |
| `res_refactored/`                 | `python/res_refactored/`                 | P52 Phase B subclass-and-delegate |

## Class hierarchy: ResFilter family (R.9.1)

```
ResFilter                  ← base; hosts 5 stages + dispatcher
├── ResFilterEnr           ← production default (BALANCED+ presets);
│                            overrides _stage_gain_compute → _gain_compute_enr
└── ResFilterWiener        ← legacy fallback; owns self.over_sub scalar;
                              overrides _stage_gain_compute → _gain_compute_wiener_legacy
```

Selection in `orchestrator.AEC.__init__` line 2019:
```python
elif self.config.res_gain_type == "enr":
    _ResCls = ResFilterEnr
else:
    _ResCls = ResFilterWiener
```

Rationale (per user 2026-05-16): `over_sub` config field is consumed
only by wiener / spectral_sub branches; ENR path uses dynamic
over-subtraction computed in `AEC._compute_mu_scale` instead. Class
split makes ownership explicit so future `over_sub` tweaks can't
appear to apply to ENR when they don't.

## Backward compat

`python/aec.py` is now a 53-line shim that re-exports every public
symbol. All 6 caller sites continue to work unchanged:

* `python/eval_aec_challenge.py`
* `python/run_one_case.py`
* `python/run_e2e_parity.py`
* `python/test_p52_regime.py`
* `python/test_f3_1_mic_excess.py`
* `python/modules/res_refactored/res_filter_refactored.py`

CLI entry point preserved via `if __name__ == "__main__": main()` in
the shim, which forwards to `modules.orchestrator.main`.

## Algorithm change bundled in cycle

Beyond pure refactor, **one algorithm-affecting commit** (`ab44842`)
ships in this cycle:

* **`feat(hpf) — align ref-path HPF default with AEC3`** —
  `enable_highpass_ref` config field added, defaulting to **False**.
  Mirrors WebRTC AEC3 behaviour where `high_pass_filter_echo_reference`
  is field-trial-gated (default OFF). Mic-path HPF (`enable_highpass`)
  unchanged. Breaks byte-equal vs the R.0 baseline by intent — the
  R.0c baseline (rendered at commit `ab44842`) becomes the new
  reference for R.10–R.12 byte-equal verification. AECMOS impact
  pending 800-case audit.

## Verification per sprint

| Sprint  | Commit    | aec.py rows | Validation                                  |
| ------- | --------- | -----------:| ------------------------------------------- |
| R.0     | (HEAD~13) |       9,660 | Baseline; 5-case md5 captured at `/tmp/r0_post` |
| R.1     | `2569140` |       9,660 | 5/5 byte-equal; `python/modules/__init__.py` only |
| R.2     | `51e6664` |       9,660 | 5/5 byte-equal; 4 already-separated modules `git mv`'d |
| R.3     | `4c54107` |       9,365 | 5/5 byte-equal; enums + dataclasses extracted |
| R.4     | `ebcd073` |       8,907 | 5/5 byte-equal; preprocessing + erle + delay |
| R.5     | `2fb5785` |       8,557 | 5/5 byte-equal; filters (NLMS / PBFDAF / PBFDKF) |
| R.6     | `b54af5d` |       8,018 | 5/5 byte-equal; detectors + dtd             |
| R.7     | `d27a30a` |       7,718 | 5/5 byte-equal; epc family                  |
| R.8     | `ac51d32` |       7,340 | 5/5 byte-equal; state + residual_estimator  |
| R.9     | `61dc2f3` |       5,235 | 5/5 byte-equal; ResFilter (largest)         |
| R.9.1   | `ed95a0a` |       5,249 | 5/5 byte-equal; ResFilterEnr / ResFilterWiener split |
| HPF     | `ab44842` |       5,249 | **NOT byte-equal**; new R.0c baseline at `/tmp/r0c_post` |
| R.10    | `983b710` |       3,770 | 5/5 byte-equal vs R.0c; AecConfig + NLP + DebugLogger |
| R.11    | `b516231` |          53 | 5/5 byte-equal vs R.0c; AEC orchestrator + shim |
| R.12 fix| `025af8a` |          53 | 5/5 byte-equal vs R.0c; `_BLEND_F31_MIC_EXCESS` constant |

## Known pre-existing bugs surfaced (NOT refactor regressions)

* **CLI crash on PBFDKF**: `python3 python/aec.py mic.wav ref.wav out.wav --preset balanced --enable-res --cng` raises `AttributeError: '_dtd_fft_size'` from `_reset_filter_derived_state` (orchestrator.py:1285). Reproduces against the original aec.py at commit `2d90721` (R.0 baseline) — the `_dtd_fft_size` attribute is initialised only inside the `if mode == AecMode.FDAF` branch (orchestrator.py:485 / :494); PBFDKF / LMS / TIME paths never set it. The standard `eval_aec_challenge.py` path (used for 800-case bench) does NOT trigger this — chunked processing avoids the `delay_first` reset. File as v3.20 housekeeping.

## Post-refactor next steps

* Resume Phase 3 (B FilterMisadjustment retry) — wire trigger to
  `fq_usable + reset_done`, threshold `<2.0`, fire-rate gate, 60-case
  A/B, 800-case ship gate.
* AECMOS audit of HPF ref-default flip on the in-progress 800-case
  R.0c baseline render.
* Optional: file `_dtd_fft_size` PBFDKF init bug as v3.20 cleanup.
