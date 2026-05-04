# R7.1a — EPV weak-filter false-positive damping (THR=8.0)

**Date**: 2026-05-04
**Branch**: `algo/round7-signal-split-trajectory`
**Variant**: env `R7_EPV_WEAK_THR=8.0` — at EPV trigger site, skip the entire
EPV response (Q_high, P_max=1, P_floor=1, mark_diverged, render_forced,
erl_clamp) when `|main_filter.W|_2 < 8.0`. Default-off keeps bit-exact parity
vs `baseline_v381_seeded` (verified via `parity_round6`).

## Verdict: NULL — non-ship diagnostic

| Bucket | echo | deg | Δecho | Δdeg | gate |
|---|---:|---:|---:|---:|---|
| FS_static   | 3.769 | 4.999 | +0.001 | +0.000 | n/a |
| FS_movement | 3.827 | 4.999 | -0.000 | +0.000 | n/a |
| DT_static   | 4.251 | 2.259 | -0.001 | +0.002 | n/a |
| **DT_movement** | 4.119 | 2.287 | -0.001 | **+0.001** | **FAIL (need +0.02)** |
| NE          | 4.998 | 4.007 | +0.000 | +0.000 | ok |

All bucket deltas inside ±0.005 noise floor. DT_movement deg gain is
sub-acceptance by 20×. Echo unchanged → mechanism didn't break anything,
but didn't help either.

## Mechanism reading: EPV is symptom, not cause

P0 found worst-DT_mv fires EPV 2.75× more than best (1.4 vs 0.4 mean count;
11/20 vs 5/20 hit-rate). R7.1a tested whether suppressing those fires lets
the filter grow. Result: no. Two reasons confirmed:

1. **Sparse fires**: 11/20 worst hit but mean only 1.4 → only ~28 EPV
   events across all 20 worst cases. Suppressing them is too small a
   per-case intervention to move the deg score by ≥0.02.
2. **Weak filter is independent of EPV**: smoke trace showed wHmBm7VHfk
   went from 3 baseline fires to 5 raw-fires-all-suppressed under THR=8,
   but filter_w_norm only changed 3.89 → 3.88. The filter doesn't grow
   when given the chance — EPV resets are not what's holding it small.

The P0 strong separators (`filter_w_norm` worst 7.16 vs best 17.83,
`main_err_smooth` 176 vs 647) describe a **signal-energy regime difference**
between worst and best DT_mv cases. EPV firing is one downstream symptom of
that regime, not its cause. The cause is likely upstream:

- worst cases sit in a low-far-power × high-near-power regime where the
  filter's ratio-based Kalman update never produces large W
- the filter's "small W = low confidence echo cancellation" is actually
  _correct_ on these cases (echo really is hard to identify when it's
  buried in near-end). RES then over-suppresses near-end to compensate.

This is the structural Pareto we already mapped in R5/R6: filter under-fits
→ RES over-suppresses to compensate → DT deg drops. EPV gating doesn't
move that lever.

## What we learned that R7 didn't have before

- Confirmed P0 EPV separator is correlation, not causation
- Established mechanism trace plumbing for raw-vs-suppressed event counts
- Default-off env-toggle pattern works for R7 sub-variants (no parity risk)

## Decision options

R7.1a closed as NULL. Per plan don't-do list, this is `non-ship diagnostic`,
no merge candidate.

Options for next variant:
1. **R7.1b shadow_rise dry-run** (plan default): closer to filter trajectory
   directly via shadow→main copy gate. Higher risk (plan flagged for
   dry-mechanism check first).
2. **Skip to R7.2 offline mix upper bound** (plan §2.1.1): cheap test of
   whether case-level nores/ours blend even has headroom. If null too,
   R7 has fully exhausted.
3. **Out-of-scope investigation**: filter trajectory ROOT (why W doesn't
   grow on worst cases) likely requires PBFDKF kernel changes, plan
   explicitly out of scope.

Recommend (2) before (1) — offline mix is faster to disprove and answers
"is there any signal-level headroom left at all" before investing in
another invasive ablation.
