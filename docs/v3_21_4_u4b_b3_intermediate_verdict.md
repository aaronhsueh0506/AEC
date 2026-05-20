# v3.21.4 U4.B B3 intermediate values verdict — CLOSED FAIL

**Date**: 2026-05-21
**Branch**: v3.21.4 HEAD
**Trigger**: v3.21.2 plan U4.B — try intermediate values for
`DominantNearendConfig.lf_endpoint_hz` between baseline (500 Hz) and
the failed B3 ship (2000 Hz, regressed cohort in v3.21.2 plan).

## Test plan

- U4.B1: 500 → 1000 Hz
- U4.B2: 500 → 1500 Hz

Both on top of v3.21.4 canonical state (post-Codex hygiene + ms refactor).

## Results

### U4.B1 — `lf_endpoint_hz` 500 → 1000 Hz

| Bucket | Δecho | Δdeg |
|---|---:|---:|
| FS_static | −0.007 | −0.000 |
| FS_movement | −0.004 | +0.000 |
| DT_static | +0.005 | **−0.012** |
| DT_movement | −0.002 | −0.001 |
| NE | +0.000 | +0.001 |

Cohort tail: 21 echo + 19 deg cases > 0.05 absolute regression.
Worst Δdeg −0.791 (`oEyXuSCCw0qdJ0J16FGfcQ` DT_static).

### U4.B2 — `lf_endpoint_hz` 500 → 1500 Hz

| Bucket | Δecho | Δdeg |
|---|---:|---:|
| FS_static | −0.008 | −0.000 |
| FS_movement | −0.004 | +0.000 |
| DT_static | +0.005 | **−0.015** |
| DT_movement | −0.002 | −0.000 |
| NE | +0.000 | +0.001 |

Cohort tail: 26 echo + 22 deg cases > 0.05 absolute regression.
Worst Δdeg −0.791 (same case as U4.B1: `oEyXuSCCw...`).

## Trend across all tested values

| Variant | DT_st Δdeg | FS_st Δecho | Cohort tail count |
|---|---:|---:|---:|
| 500 Hz (baseline) | 0.000 | 0.000 | 0 |
| **U4.B1 1000 Hz** | **−0.012** | **−0.007** | **40** |
| **U4.B2 1500 Hz** | **−0.015** | **−0.008** | **48** |
| 2000 Hz (B3 v3.21.2) | −0.016 | −0.012 | (similar) |

Monotonic: every value above 500 regresses both DT_static deg and
FS_static echo. No intermediate is a Pareto win.

## Mechanism (consistent with v3.21.2 B3 verdict)

On the AEC Challenge 800-case cohort, the 500–2000 Hz band carries
**more echo than voice energy on average**. The `DominantNearendDetector`
sums LF energy across `[0, lf_endpoint_hz]` to compute the ENR ratio.
Wider sum:

1. Pulls more echo into `echo_sum` → ENR rises
2. ENR > `enr_threshold` more often → fewer NE state triggers
3. NE state engagement drops → DT speech sees fewer NE-frames → less
   nearend_tuning → harder per-bin suppression → DT speech damaged

The same case (`oEyXuSCCw...` DT_static, Δdeg −0.791) is the worst
regression at both 1000 Hz and 1500 Hz — this is a single-case outlier
where the cohort-tail mechanism manifests sharply.

## Verdict

**CLOSED FAIL** — `lf_endpoint_hz = 500.0` is the cohort-empirical
sweet spot for this corpus. The original 500 Hz default (= AEC3 bin 16
@ fft_size=512, which was the v3.21.0 hardcoded value) remains optimal.
B3 dead at all tested values from 1000 to 2000 Hz.

## Action

- `DominantNearendConfig.lf_endpoint_hz: float = 500.0` (unchanged).
- B3 fully closed. No follow-up in v3.21.x AEC3-alignment arc.

## Files

| Artifact | Path |
|---|---|
| U4.B1 bench | `results_v3_21_4_u4b1/result.md` |
| U4.B2 bench | `results_v3_21_4_u4b2/result.md` |
| Mechanism comment | [python/modules/residual/suppression_gain.py:98-105](../python/modules/residual/suppression_gain.py#L98-L105) (existing v3.21.2 comment, validated by this retest) |
