# U5 FS recovery sweep verdict — v3.21.2 cycle

**Date**: 2026-05-20
**Branch**: v3.21.2 HEAD (commit 6a071c1 baseline; U5.3 ship candidate on top)
**Predecessor**: T1 (commit f1ea92c — frequency-canonical bin-index alignment minus B3)

## Question

T1 frequency-canonical fix recovered HF formant energy (U1 verdict) at the
cost of FS_static Δecho −0.152 / FS_movement Δecho −0.121. Can we recover
the FS regression *without* undoing the DT gain by re-tuning a non-bin-index
knob in the residual chain?

## Method

Phase 1 (Explore agent ranking, 2026-05-20): inventory all "residual echo
strength" knobs in [python/modules/residual/](../python/modules/residual/);
rank top-5 by expected FS ROI vs DT risk. Result:

| Rank | Knob | Suggest | FS ROI | DT Risk |
|---|---|---|---|---|
| 1 | `normal_tuning.mask_hf.enr_transparent` | 0.07 → ↓ | +0.03–0.05 dB | LOW |
| 2 | `EchoAudibilityConfig.normal_render_limit` | 64 → ↓ | +0.02–0.05 | LOW |
| 3 | `EpStrengthConfig.default_gain` | 0.014 → 0.020 | +0.03–0.05 | LOW |

(Ranks 4-5 omitted; lower expected ROI.)

Sweep protocol: one-knob-at-a-time on top of T1 state; 800-case AECMOS
bench (`preset=balanced --filter 832 --cng --parallel`); accept if every
bucket Δ vs T1 is non-negative (= asymmetric Pareto move).

## Results

### U5.1 — `mask_hf.enr_transparent` 0.07 → 0.05

(Original Explore-agent suggestion was 0.07→0.15. AEC3 source + own-code
re-read showed sign was inverted: higher transparent = more permissive
mask = worse FS suppression. Corrected direction = lower.)

**Bench delta vs T1**: ±0.002 dB all buckets. No effective leverage on
800-case cohort.

**Mechanism (post-mortem)**: `mask_hf` only fires in a narrow ENR slice
(`enr_transparent < enr < enr_suppress`) inside `GainToNoAudibleEcho`.
Most production frames are outside this slice. Knob lacks population.

**Verdict**: NO EFFECT, reverted.

### U5.2 — `normal_render_limit` 64 → 32

EchoAudibilityConfig floor that disables the audibility shortcut when
render energy is "low" (`X_max < normal_render_limit`).

**Bench delta vs T1**: identical to T1 in all reported buckets.

**Mechanism (post-mortem)**: the audibility shortcut rarely binds at
production volumes; lowering the floor never actually disables it on
the cohort.

**Verdict**: NO EFFECT, reverted.

### U5.3 — `EpStrengthConfig.default_gain` 0.014 → 0.020

Scales nonlinear-mode R² (residual echo estimate) by
`(0.020 / 0.014)² ≈ 2.04×` via `R² = X² × default_gain²` in
[python/modules/residual/residual_echo_estimator.py](../python/modules/residual/residual_echo_estimator.py)
`GetEchoPathGain`. Reverb-tail power scaled proportionally.

S1 trace shows our cohort spends 66–92% of frames in nonlinear mode
(linear ERLE not yet converged → ERLE-divide path unreliable → fall
through to default-gain path), so this knob has high frame-population
leverage on our 800-case bench.

**Bench delta vs T1** (= asymmetric Pareto-positive):

| Bucket | Δecho vs T1 | Δdeg vs T1 |
|---|---:|---:|
| FS_static | **+0.005** | +0.000 |
| FS_movement | **+0.004** | +0.000 |
| DT_static | +0.000 | **+0.002** |
| DT_movement | +0.000 | **+0.001** |
| NE | +0.000 | +0.001 |

Every bucket non-negative; FS recovered without DT damage.

**Cumulative vs v3.21.0 baseline (a537b65)** with U5.3:

| Bucket | Δecho | Δdeg |
|---|---:|---:|
| FS_static | −0.147 | +0.000 |
| FS_movement | −0.117 | +0.000 |
| DT_static | −0.049 | **+0.094** |
| DT_movement | −0.048 | **+0.115** |
| NE | +0.000 | +0.003 |

vs T1 alone (FS_static −0.152 / FS_movement −0.121): +0.005 / +0.004 recovered.

**AEC3 precedent**: `WebRTC-Aec3EchoPathGain` field-trial Aggressive profile
uses `0.02`. Our 0.020 = that field-trial value. Within AEC3-documented
range; not an invention.

**Verdict**: SHIP CANDIDATE.

## Mechanism (why U5.3 is asymmetric)

R² is the residual echo estimate consumed by `SuppressionGain`. Two paths:

1. **Linear** (ERLE-divide): R² = Ŷ² / ERLE. Used when filter has
   converged ENR-wise.
2. **Nonlinear** (default-gain): R² = X² × default_gain². Used when
   filter linear cancellation is unreliable or the path is highly
   nonlinear.

Our 800-case cohort spends most frames in the nonlinear path (S1 trace
66-92%). Bumping `default_gain` 0.014→0.020 raises R² by 2.04× **only in
nonlinear-path frames**. Two consequences:

- **FS frames**: nonlinear path is dominant; SuppressionGain sees more
  estimated echo → fires harder on borderline bins → FS_echo +0.004-0.005.
- **DT frames**: NE detector triggers nearend_tuning when real nearend
  is present, which uses a *different* gain rule less sensitive to R²
  magnitude → DT_deg essentially unchanged.

This asymmetry is what makes the knob a Pareto-positive move on top of T1.

## Halt-condition check

- **HALT-U5**: every FS-recovery knob loses DT > 0.02 → would ship T1 as-is.
  Not triggered — U5.3 lost DT 0.000 (no regression).

## Files

| Artifact | Path |
|---|---|
| U5.3 code change | [python/modules/residual/residual_echo_estimator.py](../python/modules/residual/residual_echo_estimator.py) (`EpStrengthConfig.default_gain = 0.020`) |
| U5.3 bench result | `results_v3_21_2_u53/result.md` |
| U5.1/U5.2 reverted | (no code artifact; commits not made) |

## Recommendation

Commit U5.3 as the final v3.21.2 ship candidate. Proceed to U6 release
prep (CHANGELOG finalization, `__version__` bump, baseline snapshot,
merge-to-main + tag — all gated on explicit user authorization).

## Non-shipping carry-overs surfaced during U5

Documented separately in the v3.21.2 plan and tracked for v3.22:

- **Time-domain unit-conversion bugs** (3 HIGH): `trigger_threshold=12`,
  `hold_duration=50`, `noise_floor_hold=50` — ported as bare ints from
  AEC3 4 ms blocks into our 10 ms hops, giving 2.5× longer time-equivalents
  than AEC3 intended. Scaling DOWN to correct values opposes FS recovery
  direction (more NE triggers → less suppression → FS_echo worse) — so
  this is a CORRECTNESS fix not a Pareto fix; ship in a dedicated v3.22
  audit.
- **ReverbDecayEstimator partial port** (1/3 of AEC3 size; missing
  `AnalyzeFilter` + `EarlyReverbLengthEstimator` + validation gates).
  Functional but coarser than AEC3.
- **Codex hygiene findings** (4 items): AEC.reset() doesn't clear AEC3
  post-state; `_reset_filter_derived_state()` docblock stale;
  `return_res_context=True` dead contract; legacy delay knobs no-op.
