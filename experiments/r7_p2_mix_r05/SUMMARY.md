# R7.2 — Offline upper-bound mix (fixed ratios, blanket)

**Date**: 2026-05-04
**Branch**: `algo/round7-signal-split-trajectory`
**Method**: `mixed = (1-r) * ours + r * ours_nores` per-case, written as `_ours.wav`,
re-scored by AECMOS. Source `ours` / `ours_nores` from R7.1a eval_output (functionally
baseline; R7.1a Δ vs `baseline_v381_seeded` ≤ 0.001 across all buckets).
3 ratios: 5%, 10%, 20%.

## Bucket-level Δ vs baseline_v381_seeded

| ratio | DT_mv Δdeg | DT_mv Δecho | FS_st Δecho | FS_mv Δecho | NE Δdeg |
|---|---:|---:|---:|---:|---:|
| 5%  | **+0.063** | -0.108 | **-0.065** | -0.012 | -0.010 |
| 10% | +0.158 | -0.240 | -0.228 | -0.142 | -0.018 |
| 20% | +0.431 | -0.483 | -0.507 | -0.473 | -0.022 |

## Acceptance gates per user spec

| gate | 5% | 10% | 20% |
|---|:-:|:-:|:-:|
| DT_mv deg ≥ +0.02 | ✓ | ✓ | ✓ |
| FS_st echo regress < 0.02 | **✗ (-0.065)** | ✗ (-0.228) | ✗ (-0.507) |
| FS_mv echo regress < 0.02 | ✓ (-0.012) | ✗ (-0.142) | ✗ (-0.473) |
| NE deg drop ≤ 0.01 | ✓ (-0.010) | ✗ (-0.018) | ✗ (-0.022) |
| worst20 DT_mv ≥10/20 impr | ✓ (11/20) | ✓ (14/20) | ✓ (20/20) |

**5% passes 4/5 gates; only FS_st fails by ~3.25× the limit.**
10% / 20% fail 3-4 gates each. Plan default acceptance also requires
`DT_mv echo regress < 0.02` — all 3 fail that (5%: -0.108, 5×).

## Worst-20 DT_mv per-case (5% mix)

| metric | result |
|---|---|
| deg improvement ≥ +0.02 | 11/20 cases |
| any deg improvement | 15/20 |
| deg regression | 5/20 |
| echo regression > 0.02 | 15/20 |

Top-5 worst (by baseline_deg):

| stem | base_deg | new_deg | Δdeg | Δecho |
|---|---:|---:|---:|---:|
| XV5L2dn3...doublet | 1.16 | 1.16 | +0.006 | -0.037 |
| wHmBm7VH...doublet | 1.26 | 1.22 | -0.035 | -0.023 |
| WqEYNwal...doublet | 1.30 | 1.29 | -0.011 | -0.049 |
| xofDX004...doublet | 1.31 | 1.33 | +0.027 | -0.018 |
| XuiheB7E...doublet | 1.32 | 1.32 | +0.007 | -0.021 |

The top-1 worst case (XV5L2, base 1.16) barely moves. Several top-5 cases
have nores audio that's no better than ours on these stems — the per-case
upper bound is highly variable; bucket gain is dominated by middle-of-list
cases that get +0.05 to +0.20 deg.

## Mechanism reading

Mix-mode lever works exactly as suspected: nores has substantially more
echo than ours (FS_st: nores echo 2.62 vs ours 3.77 from R5 Phase 0). At
5% blanket mix, that delta drags FS down by 1/20 = 0.05 echo points,
which matches the observed FS_st regress (-0.065) closely.

DT_mv deg gain is real (+0.063 = ~3× the +0.02 gate), but:
- comes paired with -0.108 echo regress (DT_mv echo)
- worst-20 case-level distribution is wide (5/20 regress on deg, 15/20
  regress on echo)
- top-1 worst case unmoved

This is consistent with R5 Phase 0 binding audit: nores wins ~+1.0 DT deg
but gives back ~+1.0 DT echo. Linear-only AEC alone is at the deg
ceiling for this dataset; everything else trades echo for deg or vice
versa, on the same Pareto curve.

## Decision

**Blanket fixed-ratio mix cannot satisfy R7.2 acceptance.** The gap is
not "close enough to tighten threshold and ship" — it's ~3-25× over
gate on echo metrics depending on ratio.

The 5% result IS the headroom upper bound for any case-level mix:
- maximum DT deg gain ≤ +0.063 (and that's at the cost of -0.065 FS echo)
- per-case the gain is variable; not all worst-20 cases benefit

A **conditional mix** (only blend on frames with `effective_dt > THR
AND nores_echo_proxy < echo_risk_thr`) could in principle preserve FS
(no mixing on FS-only frames) while keeping DT gain. But:
1. The FS_st loss (-0.065) suggests blanket-mix bleeds into FS even at
   5%. A conditional gate must be very tight to undo that — and tight
   gates undo the DT gain too.
2. The top-1 worst case is unmoved by ANY blanket mix. Conditional mix
   doesn't add new headroom there.
3. Implementation cost: in-process per-frame mix needs dual OLA buffers
   in ResFilter. Per plan §2.3, this is the most expensive R7 step.

**Recommend STOP R7.2 here.** The Pareto trade is bound; conditional
mix would at best recover the 5% blanket result with FS preserved, which
gives DT_mv deg ≈ +0.06 / -0.05 echo. Marginally above ship gates only
if echo gates are loosened — and that's not what the user asked for.

Net for R7: P0 GO + R7.1a NULL + R7.2 fixed-blanket exhausts the
classical-suppressor lever space. Filter trajectory ROOT is the only
remaining mechanism that hasn't been touched, but plan keeps that
out of scope.
