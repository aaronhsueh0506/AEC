# F4 verdict — null / minor regression, REJECT

**Branch**: `algo/f4-erle-correction-guarded`  •  **Date**: 2026-05-02
**Patch**: tighten inst-ERLE DT-correction cap from 4.0 → 2.0 when cold-start
DT-from-frame-0 signature is detected (`not once_converged AND
far_active_blocks > 200 AND erl_estimate > 0.4`).

## Bench Δ vs baseline_v381 (800-case BALANCED)

| Bucket | Δecho | Δdeg | verdict |
|---|---:|---:|---|
| FS_static | -0.001 | +0.000 | ok |
| FS_movement | -0.001 | -0.000 | ok |
| DT_static | +0.001 | -0.003 | ok |
| DT_movement | +0.002 | -0.004 | ok |
| NE | -0.000 | +0.000 | ok |

All buckets move within ±0.004 — well inside the ±0.02 acceptance window.

## Why null

Guard-fire frequency on a 30-case DT sample:
- 2 / 30 cases had the guard fire at all
- 62 / 111934 frames (0.1 %) total
- 19 / 30 cases reached `_filter_once_converged = True` (guard auto-disables)

The cold-start trigger is gated on `_erl_estimate > 0.4`. Even on DT cases
where the filter never converges, `_erl_estimate` stays low (echo path is
weak in this dataset, ERL drift signature is rare). The guard is too narrow
to alter behaviour on the populations that drive the DT_movement worst case.

## Decision

**Reject** — null effect on all 5 buckets, minor regression on DT (-0.003 to
-0.004 deg). Branch dropped. Per plan acceptance criterion #3
("DT_movement deg improvement < 0.02 → Pareto slide / no value"), there is no
directional benefit to take.

## Implications for F1

F1 (InitialDtGuard) uses the same trigger (`not once_converged AND
far_active_blocks > 200 AND erl_estimate > 0.4`). The fire-frequency data
above predicts F1 will also be a low-coverage intervention unless the
trigger is widened. Two paths:

1. Keep narrow trigger (per plan, NE-selectivity preserved) — accept that
   F1 may also produce a null bench. Proceed as-planned.
2. Widen the trigger before F1 sub-branch — drop `erl_estimate > 0.4` gate,
   replace with `using_render_based AND mic_pwr > far_pwr` NE-selectivity.

Path (1) follows the plan literally. Path (2) is a deviation. Lock to path
(1) for first F1 attempt; if null, then re-evaluate trigger conditions
before declaring F1 dead.
