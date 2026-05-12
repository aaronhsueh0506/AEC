# P52 Phase B Task B.3 — Module 5 NoiseFloorAndCng verdict

**Date**: 2026-05-12
**Branch**: `feature/p52-phase-b-refactor`
**Status**: **PASS** (hard bar AND internal 100% target both met)

## Verdict at a glance

| Bar | Result |
|---|---|
| Hard bar §3.6 (≥99.99% within `atol=1e-6, rtol=1e-5`) | **PASS** (100% exact) |
| Internal advisory target (100% exact match) | **PASS** |
| Sample size | 100 cases (seed=42) |
| Total samples | 40,814,400 |
| Cases byte-identical | **100 / 100** |
| Top-10 max abs delta | 0.0 |

## Extraction methodology

`ResFilter._stage_noise_floor_and_cng` (aec.py:2160-2228) extracted into
`python/res_refactored/noise_floor_cng.py::noise_floor_cng(rf, **kwargs)`.
Verbatim transcription (`self.X` → `rf.X`); returns `output[hop_size]`
(float32). Subclass-and-delegate via
`ResFilterRefactored._stage_noise_floor_and_cng` override.

CNG path uses `np.random.randn` with the global numpy RNG state seeded per
case via `np.random.seed(42)` in the validation harness — same convention as
A.0R.6 / A.0R.8. Byte-equal is preserved because the legacy and refactored
runs execute identical sequences of `randn` calls in identical order.

## B.3 closure summary

All five modules now migrated:

| Module | File | Stage method overridden |
|---|---|---|
| 1 ResidualEstimator | `residual_estimator.py` | `_stage_residual_model` |
| 2 GainComputer | `gain_computer.py` | `_stage_gain_compute` |
| 3 SpectralShaper | `spectral_shaper.py` | `_stage_gain_postprocess` |
| 4 TemporalSmoother | `temporal_smoother.py` | `_stage_temporal_smoothing` |
| 5 NoiseFloorAndCng | `noise_floor_cng.py` | `_stage_noise_floor_and_cng` |

100-case byte-equal: **100/100 cases identical, 40.8M samples, zero drift**
across all five modules combined.

Ready to proceed to **B.4** (full 800-case byte-equal sweep) per §3.5 + §3.6.

## Anti-loophole compliance

- §5.5: verbatim copy; zero numerical change across all five modules.
- §5.4: no shadow / classifier reference anywhere in `res_refactored/`.
- §6.4: only `python/res_refactored/*` + tooling + docs touched.

## Cross-references

- Module impl: [python/res_refactored/noise_floor_cng.py](../python/res_refactored/noise_floor_cng.py)
- Orchestrator: [python/res_refactored/res_filter_refactored.py](../python/res_refactored/res_filter_refactored.py)
- Module 1-4 verdicts: in same docs directory
- Artefacts: `/tmp/p52_b3/refactored_all5.npz`, `/tmp/p52_b3/verdict_all5.json`
