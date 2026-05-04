# Phase 3 verdict — null with seeded CNG, REJECT

**Branch**: `algo/phase3-erle-reset-transitions` (will be dropped)
**Date**: 2026-05-02
**Patch**: ERLE / RES trust reset (`_filter_erle_est`, `_fb_erle_est`,
`_residual_est`) on filter convergence transitions:
- `_maybe_mark_diverged()` — when `_converged was True` (lose-conv).
- After `just_converged=True` (gain-conv).

Filter weights W untouched. State-routing only.

## Bench Δ vs `baseline_v381_seeded` (seeded CNG, 800-case BALANCED)

| Bucket | Δecho | Δdeg | verdict |
|---|---:|---:|---|
| FS_static | +0.001 | -0.000 | ok |
| FS_movement | -0.000 | +0.000 | ok |
| DT_static | -0.000 | +0.000 | ok |
| DT_movement | +0.000 | +0.000 | ok |
| NE | +0.000 | +0.000 | ok |

## Per-case analysis (DT_movement, n=114)

| metric | value |
|---|---|
| worst-20 mean Δdeg | +0.000 |
| cases helped > +0.005 | 1 / 20 worst, 1 / 114 total |
| cases hurt < -0.005 | 0 / 20 worst |
| min Δdeg | -0.001 |
| max Δdeg | +0.005 |
| changed > 0.020 | 0 / 114 |

113 / 114 DT_movement cases unchanged within ±0.005. With CNG seeded
the patch produces near-zero output divergence — confirming this is a
genuine no-op, not noise-cancellation.

## Why null

Three plausible mechanisms drown out the ERLE / RES reset:

1. **ERLE EMA recovers in 2-3 frames**: `FilterErleEstimator` alpha
   ≈ 0.9; resetting to 1.0 returns to within 1% of natural value within
   ~30 frames (300 ms). Most transitions happen during far-active speech
   so the EMA re-learns immediately.
2. **`ResidualEchoEstimator._using_render_based` is re-evaluated every
   frame**: legacy switch (aec.py:2598) sets it back to True on the
   next frame whenever `not filter_converged`. Reset → True → True
   transition is invisible downstream.
3. **Downstream caps dominate**: ResFilter applies `echo_psd × 2.0` cap,
   `error_psd × err_cap_mult`, `dt_suppress_cap` (aec.py:1576-1610).
   These bound `residual_echo_psd` regardless of which ERLE the
   estimator started from.

The transitions count *correlates* with worst-DT (Phase 0 finding) but
is not the *causal lever*. Fixing the trust state at transitions does
not address the underlying filter behavior that produces the wobble.

## Implications for the architectural gap

The reviewer's AEC3-state-routing thesis assumed that *if we route
state events properly*, the suppressor would benefit. Phase 0 + Phase 3
data say otherwise on this dataset:

- `dominant_nearend_like_state` doesn't separate populations (Phase 0)
- `usable_linear_estimate_v2` ≡ v1 (Phase 0; F3 already null)
- ERLE/RES reset on transitions doesn't move the bucket (Phase 3)

Implication: the worst-DT_mv damage is not in the *routing* layer.
It's in:
- Filter weight (W) trajectory during transitions — out of scope
  per user constraint ("don't reset filter weights")
- Movement-induced delay re-acquisition — orthogonal mechanism
- Echo path nonlinearity beyond filter representation — needs NN
  postfilter

State-routing patches don't move the binding constraint. To make
further progress on this dataset, the lever has to be either:
1. Filter mu_scale gating during NE-suspected windows (W-touching,
   excluded by user)
2. Per-bin NE confidence mask in suppressor (different mechanism than
   frame-level state machine)
3. Movement-specific delay tracking improvement
4. NN postfilter (out of this branch's scope)

## Decision

**Reject Phase 3.** Drop sub-branch. Patch reverted on
`algo/f1-f4-investigation`.

## Cumulative summary across this round

CNG-noise calibration (±0.004 DT_mv deg run-to-run) showed F1/F3/F4's
prior "null" results were probably also genuine null — the ±0.004 Δ
seen there was at the noise floor. With proper seeding, all four
findings (F1, F2-audit, F3, F4) plus Phase 3 produce no measurable
bench-mean change.

The reviewer's architectural-gap hypothesis described real architectural
deficiencies but the binding constraint on this dataset is elsewhere.
