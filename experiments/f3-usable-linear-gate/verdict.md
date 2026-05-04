# F3 verdict — null bucket-level, REJECT

**Branch**: `algo/f3-usable-linear-gate`  •  **Date**: 2026-05-02
**Patch**: `ResidualEchoEstimator.attribute_legacy`'s `force_render` clause
swaps `not filter_converged` for `not aec_state.usable_linear_estimate`
when `aec_state` is provided. Once a filter has converged for the first
time, transient `_filter_converged` dips no longer force render-mode —
hysteresis on already-trustworthy filters.

## Bench Δ vs baseline_v381 (800-case BALANCED)

| Bucket | Δecho | Δdeg | verdict |
|---|---:|---:|---|
| FS_static | -0.000 | +0.000 | ok |
| FS_movement | -0.002 | -0.000 | ok |
| DT_static | +0.002 | -0.001 | ok |
| DT_movement | -0.002 | +0.002 | ok (directionally right) |
| NE | -0.000 | +0.000 | ok |

DT_movement deg +0.002 is in the right direction but 10× below the
acceptance threshold (≥0.02 to be a structural fix).

## Per-case analysis (DT_movement, n=114)

- mean Δdeg over baseline-worst-20: +0.002 (7/20 helped > +0.005)
- distribution: min -0.226, p25 -0.025, med -0.003, p75 +0.017, max +0.194
- 98/114 cases change > 0.005 (wide bimodal — positive lean but symmetric)

## Why minimal effect

`usable_linear_estimate` differs from `filter_converged` only when:
`once_converged AND not currently_converged AND not epc_active`
i.e. only during transient convergence dips after first-converged. Such
dips are typically short-lived (few frames) — `_render_based_hold = 5`
hysteresis already covers most of them in the legacy logic. The new gate
just extends hysteresis slightly. Limited population → limited effect.

## Decision

**Reject** — bucket-level Δ within ±0.002 (10× below acceptance), per-case
distribution bimodal even though baseline-worst-20 shows slight positive
lean. Plan #3 / #4 not met.

## Cumulative finding

F1 + F3 + F4 all produce null bucket-level Δ, with per-case distributions
that are wide and bimodal. The three mechanisms target three different
populations (cold-start DT, transient-dip post-converged, ERLE-correction
overreach) but each population is a minority in this 800-case blind set.

The reviewer's overall hypothesis — that v3.8.1 lacks AEC3-style cold-start
+ usable-linear protection — describes a real architectural gap, but on
this dataset the gap is not the binding constraint. The DT_movement worst
cases are dominated by something else (likely movement-induced delay
re-acquisition, gain ramps, or echo-path nonlinearity beyond the filter's
representation), not the cold-start DT-from-frame-0 family.

## Implications for F5 (dominant-nearend)

Per plan: F5 conditional on F1/F4 being Pareto-bound. They are not
Pareto-bound — they are null. The same trigger style would produce the
same null. **Skip F5** unless a fundamentally different selector
(per-bin spectral NE confidence, or learned dominant-NE mask) is
designed first. Out of scope for this investigation branch.
