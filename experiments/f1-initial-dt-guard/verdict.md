# F1 verdict — bimodal effect, not structural, REJECT

**Branch**: `algo/f1-initial-dt-guard`  •  **Date**: 2026-05-02
**Patch**: when InitialDtGuard fires (cold-start DT-from-frame-0 +
NE-likely), multiply `dt_residual_scale` by an extra 0.7× before passing
`eff_echo_spec` to ResFilter. Suppressor-only — no filter mu_scale / Q /
shadow copy changes.

## Bench Δ vs baseline_v381 (800-case BALANCED)

| Bucket | Δecho | Δdeg | verdict |
|---|---:|---:|---|
| FS_static | +0.002 | -0.000 | ok |
| FS_movement | -0.001 | -0.000 | ok |
| DT_static | +0.000 | +0.001 | ok |
| DT_movement | +0.003 | -0.004 | ok |
| NE | +0.000 | +0.000 | ok |

## Per-case analysis (120-case DT scan)

- Guard fired in 24 / 120 DT cases (20% — broader than F4's 2/30)
- Effect on fired cases is **bimodal**, not directional:
  - 7 cases improve > +0.02 deg (best +0.049, +0.044)
  - 6 cases regress < -0.02 deg (worst -0.108, -0.075, -0.053)
  - Mean over fired cases ≈ 0
- F1 effect on baseline-worst-20 DT_movement cases: mean Δdeg -0.002,
  only 3 / 20 helped >+0.005. The intervention does not concentrate on
  the cases that need help.

## Why the action is wrong

`dt_residual_scale × 0.7` reduces echo_spec uniformly across all bins
when the guard fires. On cases where NE happens to dominate, this
recovers NE — net win. On cases where echo dominates but the filter
just hasn't converged yet (still on a normal convergence trajectory),
this lets echo through — net loss. The trigger correlates with both
populations equally; the action helps one and hurts the other.

## Decision

**Reject** — null bucket-level Δ, bimodal per-case effect, no structural
fix on baseline-worst cases. Per plan acceptance criteria:
- #3: DT_movement Δdeg -0.004 < 0.02 with no FS gain → Pareto slide
- #4: improvements not concentrated on prior-worst cases → not structural

The trigger may be salvageable (it does identify a real population), but
the action (uniform echo_spec scaling) is too blunt — needs per-bin
selectivity or a different mechanism. Given F4 + F1 both null, the
next attempt should be F3 (different mechanism — residual attribution
gate, not echo_spec amplitude), which doesn't share this failure mode.

## Implications for F3 / F5

F1 + F4 both gated on the same DT-from-frame-0 signature and produced
null. Suggests this dataset's DT_movement worst cases are NOT
predominantly DT-from-frame-0 (per `docs/spec_dt_from_frame_zero.md`,
the spec's worst-12 cases included only 6 Pattern A — the population
isn't representative).

For F5 (dominant-nearend selector): if the trigger uses the same gate,
expect similar null. F5 should use independent triggers (per-bin NE
energy mask, not far_active_blocks + erl drift).
