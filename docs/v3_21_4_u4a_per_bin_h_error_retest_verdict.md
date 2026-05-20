# v3.21.4 U4.A per-bin H_error refresh retest verdict — CLOSED FAIL (cohort tail)

**Date**: 2026-05-20
**Branch**: v3.21.4 HEAD
**Trigger**: v3.21.2 plan U4.A — re-test v3.21.1's per-bin H_error
refresh substrate (commit d4e266e) on canonical state (v3.21.2
bin-index fix + v3.21.3 Codex hygiene fixes). Original v3.21.1 test
was on buggy SuppressionGain → maybe canonical state changes the
verdict.

## Substrate

Cherry-picked v3.21.1 `d4e266e` onto v3.21.4 → commit `12297ed`.
Conflict in config.py resolved (kept HEAD's Codex #4 removal of
mov_rate fields; added the new `use_per_bin_h_error_refresh: bool`
field). Test suite 25/25 PASS; byte-equal preserved at default flag=OFF.

## Test

Flipped `use_per_bin_h_error_refresh: bool = True` and ran 800-case
AECMOS bench against v3.21.3 baseline (`docs/bench/v3_21_3_baseline/balanced_scores.json`).

### Bucket means

| Bucket | Δecho | Δdeg |
|---|---:|---:|
| FS_static | +0.005 | −0.000 |
| FS_movement | −0.006 | +0.000 |
| DT_static | −0.004 | **+0.007** |
| DT_movement | +0.004 | −0.004 |
| NE | −0.000 | +0.001 |

All within ±0.007 dB — bucket means essentially flat.

### Cohort tail (the actual signal)

| Tail measure | v3.21.4 U4.A |
|---|---|
| Cases with Δecho < −0.05 | **82 / 800 (10%)** |
| Cases with Δdeg < −0.05 | **54 / 800 (7%)** |
| Worst Δecho | −0.366 (`Xv7jH2KcBEWqdpbT0...` FS_movement) |
| Worst Δdeg | **−0.437** (`LHsrJBRGnUKiMC2m...` DT_static) |

Top 5 Δecho regressions ≥ −0.20:
- `Xv7jH2KcBEWqdpbT0...` FS_movement −0.366
- `iyuYIcszXku7BWYOO...` FS_movement −0.339
- `sKXucFp4FUCJKo5d...` FS_static −0.333
- `nV9v63E5CUKtKTjha...` FS_movement −0.277
- `N2rQLbnp2UOg2QFR...` FS_movement −0.242

Top 5 Δdeg regressions ≥ −0.20:
- `LHsrJBRGnUKiMC2m...` DT_static −0.437
- `uLl640xveUuHp2k...` DT_static −0.247
- `x1pZr2uWBkulb77zc...` DT_static −0.239
- `WH0jN3PY40es2S0L...` DT_movement −0.221
- `zOiK6oSHp0ib3nHv...` DT_movement −0.205

## Verdict

**CLOSED FAIL** — canonical state retest does NOT rescue the v3.21.1
verdict.

Bucket-means presentation hides the per-case Pareto damage: gains and
losses cancel in averages, but 136 individual cases (17%) show > 0.05
dB regression on at least one metric. This is the SAME failure mode
v3.21.1 identified: "AEC3's per-bin leakage formula needs companion
ScaleFilter + FilterMisadjustment stabilisers we don't have aligned.
Without them, per-bin compare over-protects individually-converged
bins during echo path movement → tracking lag + DT NE damage."

The v3.21.2 / v3.21.3 changes (bin-index frequency canonical + Codex
hygiene fixes) do not address the missing ScaleFilter / FilterMisadjustment
alignment that the per-bin path depends on. So the failure mode
persists.

## Action

- `use_per_bin_h_error_refresh: bool = False` (default OFF, restored).
- Substrate code in [filters.py](../python/modules/filters.py) +
  [orchestrator.py](../python/modules/orchestrator.py) KEPT as dormant
  research code for future v3.22+ work, when AEC3 ScaleFilter +
  FilterMisadjustment full alignment lands and the per-bin path can be
  re-evaluated against a stabilised filter.

## Difference vs v3.21.1 verdict

| Measure | v3.21.1 verdict | v3.21.4 U4.A retest | Δ |
|---|---:|---:|---|
| DT_movement Δdeg (bucket mean) | −0.008 (HARD-FAIL) | −0.004 | small improvement |
| Cases regressed Δ > 0.05 (cohort tail) | 22+22+42+21 ≈ 107 | 82+54 = 136 | comparable scale |
| Worst single-case Δecho | −0.478 (Fi80...) | −0.366 (Xv7jH2...) | minor improvement |

Bucket-mean DT_movement deg moved from HARD-FAIL (−0.008) to within-noise
(−0.004); cohort tail still substantial. Net: not significantly different
from the v3.21.1 verdict.

## Files

| Artifact | Path |
|---|---|
| Substrate commit | `12297ed` (cherry-picked from `d4e266e`) |
| Retest bench output | `results_v3_21_4_u4a/result.md` |
| Per-bin code path | [python/modules/filters.py:625-670](../python/modules/filters.py#L625-L670) |
