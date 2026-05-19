# v3.21.1 A1 (per-bin H_error refresh) verdict — FAIL, defer to v3.22

**Date**: 2026-05-20
**Branch**: `v3.21.1` (local-only, post-revert state)
**Anchor commits**:
- `4a68604` v3.21.1 S0: add check_v3_21_1_gates.py
- `d4e266e` v3.21.1 S2 Commit-A: per-bin H_error refresh substrate (default OFF) ← **kept as dormant substrate**
- `Commit-B` (flag flip to True) → **reverted, not committed**

## Plan attempt

Per `~/.claude/plans/se-aec-aec-main-hazy-lynx.md` (v2, post-Codex review):
align PBFDKF's `_h_error_refresh()` with AEC3 `refined_filter_update_gain.cc:128-138`
by promoting the scalar `np.sum()` comparison to per-bin compare. Per-bin path
uses instantaneous `|self.error_spec|²` for E²_refined (addresses Codex F2
staleness on early-return paths).

## Bench result (800-case AECMOS, balanced preset, filter=832, cng=True, j4)

Anchor: `docs/bench/v3_21_3aadd2d_baseline/balanced_aec3_scores.json` (v3.21.0).
Snapshot: `docs/bench/v3_21_1_a1_attempt/`.

| Bucket | n | Δecho | Δdeg | new_deg | HARD |
|---|---:|---:|---:|---:|---|
| FS_static    | 169 | **+0.007** | −0.000 | 4.999 | PASS |
| FS_movement  | 131 | −0.010 | +0.000 | 4.999 | PASS |
| DT_static    | 186 | −0.005 | −0.001 | 2.386 | PASS |
| DT_movement  | 114 | +0.002 | **−0.008** | 2.363 | **FAIL (Δdeg<-0.005)** |
| NE           | 200 | −0.000 | +0.001 | 4.053 | PASS |

Cohort tail (per-case Δ > 0.05 regressions):

| Bucket | regressions | worst |
|---|---:|---|
| FS_static    | 22 | `JtodX3Ug6Eu5TYu0HN5IOw_farend_singletalk` Δecho **−0.145** |
| FS_movement  | 22 | `Fi80N5kW9U6nwaoS04O3vQ_farend_singletalk_with_movement` Δecho **−0.478** |
| DT_static    | 42 | `IUp2c0A4yEyttmLfJ3t0Xw_doubletalk` Δdeg **−0.088** |
| DT_movement  | 21 | `LN18k5r8t00C9DulUd809A_doubletalk_with_movement` Δdeg **−0.108** |
| NE           |  0 | clean |

HALT triggered: HALT-C (DT_movement Δdeg = −0.008 < −0.005 hard bar) + HALT-D
(107 cases regress > 0.05; way above 5-case limit).

## Failure analysis

**Pattern**: STATIC buckets borderline-OK, MOVEMENT buckets regress.

**Root cause (hypothesis)**: AEC3 reference at `refined_filter_update_gain.cc:128`
uses per-bin instantaneous E² compare correctly because the AEC3 design has
companion stabilisers we lack — `FilterMisadjustmentEstimator` + `ScaleFilter`
(per-block scale ratios in `subtractor.cc:240-249`) re-anchor the filter when
systematic under-modelling occurs. Without that re-anchoring, per-bin leakage
selection in our PBFDKF over-protects bins that are *individually* converged
but need to follow a moving echo path:

- Static cases: most bins truly converged → per-bin = scalar (small lift on
  `FS_static` echo).
- Movement cases: SOME bins (typically low-freq) re-converge faster than others;
  per-bin keeps them on `leakage_converged` (slow H_error growth) when those
  bins still need to track path change → under-correction → echo tracking lag.
- DT_movement: per-bin compare flips erratically frame-to-frame on bins where
  E²_refined and E²_coarse alternate winners; H_error trajectory becomes
  unstable → erratic Kalman gain → NE preservation damaged.

Scalar SUM averages across all bins, so the leakage choice tracks "bulk filter
health" which is the operative quantity for our adaptation regime. The scalar
path is therefore not an approximation of the AEC3 design but a different
stabiliser that happens to be required when ScaleFilter / FilterMisadjustment
are not co-aligned with AEC3.

## Disposition

- **Commit-A kept as dormant substrate** (`d4e266e` on `v3.21.1` branch).
  Byte-equal preserved at default OFF; code path exists for future v3.22
  re-attempt with companion ScaleFilter / FilterMisadjustment changes.
- **Commit-B (flag flip) discarded** without commit. `config.py` reverted to
  flag default `False`.
- **v3.21.1 NOT tagged**. v3.21.1 branch retains S0 (gate script) + Commit-A
  (dormant substrate) + this verdict. No functional change vs v3.21.0.
- Production `main` (= `v3.21.0` = `a537b65`) **unchanged**.

## v3.22+ re-attempt path

1. Port AEC3 `ScaleFilter` per-block scale ratios exactly (currently we have
   `FilterMisadjustmentEstimator` with asymmetric EMA — not the same
   mathematical object). This adds a global re-anchor that A1's per-bin
   compare can rely on.
2. With ScaleFilter live, re-attempt A1 (flip `use_per_bin_h_error_refresh =
   True`). Expectation: per-bin selectivity now safe because filter is
   re-anchored when bulk drift exceeds tolerance.
3. If still regressing, investigate `_leakage_converged` / `_leakage_diverged`
   values — they may be tuned for the scalar regime and need re-calibration
   for per-bin (e.g., smaller `leakage_diverged` to avoid over-correction on
   per-bin diverged calls).
4. Alternative: introduce smoothed-compare variant — apply EMA to the per-bin
   compare result rather than the E² values — to reduce frame-to-frame
   leakage flipping. Risk: smooths out the per-bin advantage too.

## Related references

- AEC3 `refined_filter_update_gain.cc:128-138` — per-bin leakage formula
- AEC3 `subtractor.cc:240-257` — ScaleFilter + per-frame E² computation
- v3.21.0 bench anchor: `docs/bench/v3_21_3aadd2d_baseline/`
- Failed attempt snapshot: `docs/bench/v3_21_1_a1_attempt/`
- Plan: `~/.claude/plans/se-aec-aec-main-hazy-lynx.md`
