# P53 Phase R closure evidence

**Status**: closed with negative binary verdict, branch deleted
**Date**: 2026-05-12
**Trace chain context**: P30–P53, ~9 months investigation

## 1. Summary

R1 (Habets joint dereverb + RES) implementation completed through R1.6 TR2 hard test. Architectural limit measured. Phase R closed within classical scope.

3 substantive wins on trace chain regression anchors (PESQ improvement):
- `XRTnTUjU_doubletalk`: +0.230
- `pU21kfoo_doubletalk`: +0.231
- `qNvSMyU_farend_singletalk`: +0.115

But TR2 aggregate verdict failed binding criterion (4/9 vs required ≥6/9). Single-axis gate cannot separate functional NE from spurious NE evidence on the production DT detector signal.

## 2. Implementation summary

R1 architecture (implemented but not shipped):

| Sub-task | Component | Key file (deleted with branch) |
|---|---|---|
| R1.1 | LateReverbPsdEstimator (Lebart 2001) | `python/res_replacements/late_reverb_psd.py` |
| R1.2 | JointMlPsdEstimator (Ephraim-Malah 1984 decision-directed) | `python/res_replacements/joint_ml_psd.py` |
| R1.3 | Module 1 (residual_echo_psd) Habets path wiring | `aec.py _habets_residual_model` |
| R1.3.2 | FS/NE frame-level gate (`dt_for_fs` based) | same |
| R1.4 | Module 2 (gain compute) Wiener formula | `aec.py _habets_gain_compute` |
| R1.4.2 | FS-only frames bypass to legacy pipeline | same |
| R1.5 | Module 3–5 (postprocess / temporal / CNG) preservation | `aec.py` |

Flag: `use_res_habets` (default `False`, never shipped).

Total code added: ~750 lines production, ~600 lines test.

## 3. Hard test verdict (R1.6 TR2)

9 listener-anchored cases, A/B Habets vs legacy, locked threshold `_habets_ne_threshold = 0.15`:

| Case | ΔERLE (dB) | ΔPESQ | PESQ improve ≥ +0.05? |
|---|---|---|---|
| `XRTnTUjU_doubletalk` | -0.09 | **+0.230** | YES (trace anchor) |
| `LHsrJBRG_doubletalk` | -0.03 | -0.030 | no |
| `SgKY30fj_doubletalk_with_movement` | -0.01 | +0.060 | YES |
| `afHuFvfl_doubletalk_with_movement` | -0.17 | +0.040 | no (marginal) |
| `XqvGR01t_doubletalk_with_movement` | -0.11 | +0.017 | no |
| `qNvSMyU_farend_singletalk` | -0.78 | **+0.115** | YES (wildly anchor) |
| `MkSLte0F_farend_singletalk_with_movement` | -1.42 | +0.011 | no |
| `pU21kfoo_doubletalk` | -0.02 | **+0.231** | YES (anchor) |
| `SUYzW4QT_farend_singletalk` | **-6.09** | -0.001 | no (catastrophic ERLE leak) |

**Verdict**: 4/9 PESQ improve ≥ +0.05 vs required ≥ 6/9 → **FAIL**.

## 4. Branch A diagnostic (threshold 0.5)

Tightening `_habets_ne_threshold` 0.15 → 0.5 attempted as outlier fix:

| Metric | Threshold 0.15 | Threshold 0.5 |
|---|---|---|
| PESQ improve ≥ +0.05 | 4/9 | 0/9 |
| ERLE within ±1 dB | 7/9 | 9/9 |
| SUYzW4QT ERLE | -6.09 dB | 0.00 dB (resolved) |
| 3 anchor wins | +0.23, +0.23, +0.115 | 0, 0, 0 (collapsed) |

**Architectural limit confirmed**: threshold 0.5 eliminates the echo leak AND all PESQ wins simultaneously. `dt_for_fs` distribution overlaps between "functional NE evidence" (drives wins) and "spurious NE evidence" (drives SUYzW4QT leak). Single-axis gate insufficient.

## 5. Architectural limit description

Production DT detector signal `dt_for_fs` has the following distribution characteristics:

- Functional NE frames (DT cohort, listener anchors): `dt_for_fs` typically 0.15–0.7 range
- Spurious NE frames (FS-labelled audio with intra-utterance lip noise / breath / DTD false positives): `dt_for_fs` typically 0.15–0.5 range

The two populations overlap in [0.15, 0.5]. Any single-axis threshold either:
- Fires on both → echo leak on FS-labelled audio
- Suppresses both → all PESQ wins disappear

R1.3.3 threshold lock 0.15 was calibrated on cohort-aggregate metric (invoke fraction separation). Aggregate optimum does not align with per-case worst behaviour.

Per-case forensic diagnostic (SUYzW4QT outlier vs other FS-only cases, threshold 0.15):

| stem | invoke% | dt_med | dt_p90 | ERLE_dB |
|---|---|---|---|---|
| `SUYzW4QT…_farend_singletalk` (outlier) | 22.4% | 0.001 | 0.443 | 15.10 |
| `0KjzXA3g…_farend_singletalk` | 0.2% | 0.000 | 0.004 | 24.50 |
| `1fvt8ajG…_farend_singletalk` | 12.0% | 0.000 | 0.175 | 11.81 |
| `qNvSMyU…_farend_singletalk` (winner) | 5.9% | 0.000 | 0.052 | 7.77 |

Note: winner and outlier do not separate on `dt_for_fs` median; only on p90 tail (0.052 vs 0.443) but the latter is *within* the gate-fire region for any threshold that admits the winners.

## 6. Methodology contributions (carried forward)

7 production-reality / spec-defect meta-pattern findings during Phase R:

1. **R1.3 pre-flight**: `AcousticRegimeClassifier` API granularity (whole-recording vs per-frame). Resolution: `regime_class=None` default T60.
2. **R1.3 Detail 4**: Ephraim-Malah decision-directed equilibrium under production STFT (continuous transients vs synthetic constant unit tests). Resolution: cold-start grace + warmup gate.
3. **R1.3.2**: coh2 docstring vs production EMA asymmetric fast-drop characteristic. Production coh2 runs low on FS audio, opposite of docstring claim. Resolution: gate signal re-sourced from coh2 to `dt_for_fs`.
4. **R1.3.4 cohort labelling**: utterance-level cohort labels vs frame-level test assertions. Resolution: frame-level evidence sub-cohort metric.
5. **R1.4 sign error**: Wiener gain formula numerator mistake (`n_total/p_total` instead of `p_speech/p_total`) in spec lock. R1.5 T5 voice-band measurement caught it; R1.4 Test 6 all-band measurement missed it. Resolution: 1-line fix + voice-band test requirement.
6. **R1.4.2 fallback calibration**: custom spectral_sub fallback used `over_sub=1.5` calibrated for legacy residual magnitude, mismatched against Habets-scale residual (~18× larger median). Resolution: FS-only frames bypass entire Habets path to legacy pipeline.
7. **R1.6 TR4 absolute bound**: TR4 PASS criterion ≥0.95 assumed FS-only pass-through behaviour never present in legacy production (legacy ENR-based gain on FS-only is 0.5–0.7 by design). Resolution: TR4 re-specced as A/B no-regression.

Actionable methodology rules accumulated:
- Measure production signal distribution before locking gate/threshold values
- Use A/B no-regression rather than absolute bound for hard test criteria
- Per-case worst-behaviour cross-validation when calibrating on cohort aggregates
- Voice-band-specific test assertions for RES gain formulas (not all-band averages)
- Spec lock pre-flight should verify cross-phase API contract scope and granularity
- Unit tests for statistical estimators must exercise production-realistic input variability (transients, non-stationarity), not just synthetic uniform inputs
- Single-axis gate calibration insufficient when production detector signal has overlapping distributions for functional vs spurious evidence

## 7. Future paths (out of P53 scope)

To re-engage Phase R direction with better verdict, future investigation would explore:

1. **Two-axis gate**: `dt_for_fs + erle_factor` combined. Targets the SUYzW4QT outlier (high dt + high erle = FS false-positive DT). Risk: `erle_factor` may also have overlapping distributions.
2. **Per-case adaptive threshold via `AcousticRegimeClassifier`**: but classifier is P52 analysis-only by contract. Would require classifier per-frame API + P52 contract amendment. Architectural investment.
3. **NN postfilter**: replace single-axis gate with learned multi-signal decision boundary. Out of classical scope. Requires training data, inference compute, and product-level decision.
4. **Oracle dataset**: ground-truth per-frame NE/FS/DT labelled data. Out of P53 scope.

## 8. Trace chain status (P30 through P53)

Classical-architecture-internal solutions explored to architectural limit:

- **P30–P51**: 16+ hypotheses on filter/RES tuning, no binary verdict reached
- **P52**: ShadowCopyController retire fail (1/800 catastrophe), Path 3 `PathChangeRegimeHandler` rename + `AcousticRegimeClassifier` (analysis-only)
- **P53 Step 0**: T0.E outcome, forensic review β confirmed PBFDKF + handler joint dynamics ≠ textbook Kalman. Phase L (Directions 1–4) permanently scoped out
- **P53 Phase R**: Habets architecture binary verdict measured. TR2 4/9 < 6/9

Total: ~9 months investigation. Classical scope exhausted.

Further improvement on the 5 listener-anchored regression cases requires either:
- Non-classical approach (NN / oracle, items 3–4 above)
- New evidence enabling regime-aware redesign (items 1–2)

## 9. Branch state at closure

- `feature/p53-phase-r-habets`: deleted
- `main` HEAD: contains this closure document
- No R1 code on main (`use_res_habets` default-OFF flag, never merged)
- All Phase R code archived in this document, sections 2–6

## 10. Re-engagement instructions

If future product / research wishes to re-engage R1:

1. Read this entire document
2. R1 reference implementations (Lebart, Ephraim-Malah) were in commit history of deleted branch — they can be re-implemented from the references cited (Lebart 2001, Ephraim & Malah 1984)
3. Architectural limit (section 5) is the key blocker — any re-engagement must first propose a mechanism to resolve the single-axis gate limitation
4. R1.6 TR2 9-case methodology is the binding evaluation: future re-engagement must demonstrate ≥6/9 PESQ improvement before shipping consideration
