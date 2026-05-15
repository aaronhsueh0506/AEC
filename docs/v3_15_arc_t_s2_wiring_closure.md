# v3.15 §1.5.S2 Arc T — RES preempt wiring CLOSED, signal-only retained

**Date**: 2026-05-15
**Branch**: `feature/v3.15-arc-g` (Arc G + Arc T consolidated; renamed
from arc-g sub-branch after Arc T cherry-pick of `2f7d1e9` + `4d8ae1c`)
**Substrate retained**:
- `arc_t_cohort_detector` flag (default OFF) — detector signal exposure
- `_arc_t_cohort_tail_signal` field — consumed by §1.5b Arc M.v3 upstream gate
- `arc_t_res_preempt_mode` flag KEPT for code symmetry but **NO-OP** at
  runtime; documented as such

## Disposition summary

Arc T detector S1 PASSED (5/5 TAIL fire + 3/3 CTRL no-fire, 8-case
validation). However S2 RES preempt wiring is **silent (zero output
effect)** in BALANCED preset. Two independent root causes confirmed
by single-case smoke test on `qNvSMyU` (cohort tail canonical):

| Layer | Expected effect | Actual at runtime |
|---|---|---|
| H1: `over_sub *= 1.3` boost | RES `g = 1 - over_sub × eer` lifts attenuation | **DEAD CODE** in BALANCED — `over_sub` is only read by `gain_type='wiener'` ([aec.py:3204-3211](../python/aec.py#L3204-L3211)); all 5 presets use `gain_type='enr'` ([aec.py:7308-7311](../python/aec.py#L7308-L7311) explicit comment) |
| H2: `_using_render_based = True` | Force RES into render-based echo template | **OVERWRITTEN 1 line later** — `_residual_est.compute_residual_echo()` at [aec.py:4597](../python/aec.py#L4597) recomputes `_using_render_based = want_render or (...)` from internal state machine, discarding Arc T's external assignment |

Smoke test on qNvSMyU (BALANCED, fl=832):
```
Rendering OFF...  fire_count=0
Rendering ON ...  fire_count=350      ← detector correctly fires
MD5 OFF = 9595b84fcec72f8f63ead538310ca11d
MD5 ON  = 9595b84fcec72f8f63ead538310ca11d
IDENTICAL? True                       ← but output bit-equal
max abs diff = 0.000000e+00
samples differing > 1e-9: 0/429760 (0.00%)
```

The detector **correctly fires 350 frames** (matching S1 verification),
but RES preempt's two write actions **both fail to reach the output path**.

## Why C (signal-only) over A (force_render fix) or B (ENR-path fix)

Per user 2026-05-15 decision after root-cause analysis:

1. **Detector signal is the primary v3.15 deliverable**.
   `_arc_t_cohort_tail_signal` is consumed by §1.5b Arc M.v3 as upstream
   gate (`(EPC_active AND NOT cohort_tail_T)`). §1.5b is the only
   remaining live arc with non-marginal expected gain (rescue Arc M V1's
   +0.023 DT_movement Δdeg). Signal correctness already PASSED via S1.
2. **Re-fixing RES preempt + re-running S3 800-case** would cost ~1 hr
   bench time + ~50% risk of FAIL anyway (force_render OR-in could
   damage FS bucket via over-aggressive preemption on FP cases). Even
   on PASS, expected gain is +0.030 dB cohort tail Δecho (marginal).
3. **§1.5b is independent of RES preempt** — it reads
   `self._arc_t_cohort_tail_signal` directly off `AEC` instance, never
   touches `self.res.over_sub` or `self.res._using_render_based`.

Decision: ship Arc T as **signal-only substrate**; defer A/B fix paths
to v3.16 RES refactor (see §v3.16 candidates below).

## v3.13 E2 Path 3 / cohort tail debt status

Arc T does NOT directly close the cohort tail debt (Δecho was the S3
target, never measured because S3 skipped). PathChangeRegimeHandler
remains the sole reactive defence on cohort tail (~7/800 cases).
§1.5b Arc M.v3 is the next attempt to use Arc T's signal preventively
(via Q-boost gating, not RES preemption).

## Files retained (substrate)

- [python/aec.py](../python/aec.py): Arc T config (8 fields), state init
  (8 fields), proxy compute block + hysteresis state machine + NE-corruption
  gate at the per-band ERL update site, dual-reset paths
  (`AEC.reset()` + `_reset_filter_derived_state()`), cohort_tail_T
  AecStats field, RES preempt block at line 7331-7335 (KEPT but documented
  as no-op).
- [python/eval_aec_challenge.py](../python/eval_aec_challenge.py):
  8 env overrides (`AEC_ARC_T_*`).
- [docs/v3_15_arc_t_s1_design_and_verdict.md](v3_15_arc_t_s1_design_and_verdict.md):
  S1 detector design + 8-case validation verdict (PASS).
- [docs/v3_15_arc_t_s2_wiring_closure.md](v3_15_arc_t_s2_wiring_closure.md):
  this doc (S2 wiring closure).

## v3.16 candidates (deferred from v3.15)

Documented for §1.7 RES audit doc consumption. Each is a candidate
v3.16 RES refactor arc (NOT auto-authorized; user reviews per §0.7).

### v3.16-A: Force-render OR-in `_arc_t_cohort_tail_signal`

**Mechanism**: change [aec.py:4586-4590](../python/aec.py#L4586-L4590)
`force_render` expression from
```python
force_render = (epc_active or saturation_level > 0.5 or not filter_converged)
```
to
```python
force_render = (epc_active or saturation_level > 0.5 or not filter_converged
                or getattr(self.res._residual_est, '_arc_t_cohort_tail_T', False))
```
(plus signal propagation from `AEC` to `_residual_est`).

**Expected effect**: real H2 — render-based mode FORCED during cohort
tail signal assert, no longer overridden by state machine.

**Risk**: FP rate 5% on non-tail cohort means 5% of frames in non-tail
cases will be force-rendered → potential FS bucket regression. Need
800-case A/B to validate.

**Hard bar (when authorized)**:
- cohort tail Δecho ≥ +0.030 (preemption helps)
- FS bucket no regression (FS_static + FS_movement Δecho ≥ -0.020)
- DT/NE bucket no regression (Δdeg ≥ -0.005)

### v3.16-B: ENR-path attenuation boost via `enr_t_ne` lift

**Mechanism**: when `_arc_t_cohort_tail_signal` asserts, scale
`enr_t_ne` (or per-band `enr_t_ne_pb`) by a configurable factor (e.g.
×0.7 to lift effective attenuation on cohort tail frames).

**Risk**: state-mutation conflict with Arc D (per-state ENR) and Arc R
(per-band ENR). Per §0.2 must do conflict-resolution design before
wiring (multiplicative blend vs precedence vs axis-expansion).

**Hard bar**: same as v3.16-A.

### v3.16-C: Different RES action — temporal smoothing extension

Speculative: instead of altering frequency-domain attenuation, extend
RES temporal smoothing window during cohort tail signal assert (so
post-disturbance gain stays conservative for longer). No precedent;
risk unquantified.

## v3.15 plan impact

Plan §1.5 reduced to **detector + signal exposure only**. §1.5b Arc M.v3
proceeds unblocked (consumes signal directly). No cohort tail Δecho
benefit in v3.15 production from Arc T itself.

§1.5 Hard bar (REVISED 2026-05-15):
- ~~Cohort tail bucket Δecho ≥ +0.030~~ (RES preempt killed)
- Detector FP rate ≤ 5% on non-tail cohort (S1 PASSED)
- `cohort_tail_T` AecStats field exposed for §1.5b consumption (S2 PASSED)
- `_arc_t_cohort_tail_signal` correctly fires on cohort tail cases
  (qNvSMyU 350 fires, S1+smoke verified)

## Closure protocol per §0.4

> If lever moves bucket-mean < 0.002 dB AND fires < 5% of frames after
> 3 A/B sweeps, ship as substrate for dependent arc (DO NOT rework).
> Document fire rate in verdict doc.

Arc T S2 wiring moves bucket-mean = 0 (mathematically — both H1 and
H2 are no-op as proven by smoke test MD5-identical output). Closure
trigger fires immediately, no A/B sweep needed. Substrate retained
(signal exposure) for §1.5b Arc M.v3 dependent arc.

> If S4 cannot identify 5 cases that produce audible improvement →
> Arc T CLOSED CANNOT SHIP.

Arc T S3 SKIPPED (would have been zero-delta A/B); closure documented
without 800-case bench since output mathematically cannot differ.
