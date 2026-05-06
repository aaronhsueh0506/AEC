# Research log — P3f Mini AecState gate (negative result)

Date: 2026-05-06
Code line: v3.10.4 (`dt_advisory_use_p3f_state` default off; toggle retained
for archive / future A/B). 800-case AEC Challenge AECMOS, balanced /
fl=52ms / cng=on.

## Question

P3e (V1/V2/V3) showed that single-threshold per-frame DT advisory gates
either regress FS (V1, V2, no guard) or eliminate the DT action they were
meant to deliver (V3, with `once_converged` guard). User-proposed P3f:
build a Mini AecState (filter / nearend / usable_linear), validate with
trace-only invariants on 5 fixed cases first, then wire mu reduction to
`suspicious_dt` only.

Two questions:

1. Does a state-classifier-gated mu reduction beat raw threshold gates?
2. Does it move 7GT (the case that motivated all of P3e/P3f)?

## Method

Phase 1 — added per-frame trace fields (`main_err_ratio`,
`shadow_err_ratio`, `p3f_shadow_advantage`, `erle_slope_db_per_s`,
`post_reset_age_ms`, `filter_state`, `usable_linear`) to `_diag` and
diag CSV via `--trace-aec-state`. Zero behaviour change.

Phase 2a/b — iterated classifier on 5 fixed trace cases until
invariants pass:

| Invariant | Result |
|---|---|
| 7GT 4.6–8 s post-delay = `coarse_learning` | 87 % PASS |
| 7GT 8–12 s = `refined_usable` ∪ `suspicious_dt` (matured filter) | 59 % PASS |
| 7GT 24–36 s NE-evidence frames = `suspicious_dt` ∪ `diverged` | 94 % PASS |
| FS_static 0 % `suspicious_dt` | 0 / 1191 PASS |
| FS_movement 0 % `suspicious_dt` | 0 / 2284 PASS |
| DT_static / DT_movement NE-evidence flagged | 0 % (`shadow_advantage`≈1.0 — honest negative) |

Classifier (v1, accepted at end of Phase 2):

```
state priority: idle < startup < diverged < suspicious_dt
                              < refined_usable < coarse_learning

idle           : not far_active OR main_err_smooth < 1e-9 OR mic < 1e-8
startup        : _warmup_frames > 0 OR post_reset_age_ms < 200
diverged       : main_err_ratio > 1.3 for ≥ 5 consecutive far-active frames
suspicious_dt  : refined_latched AND ne_evidence AND main_err_jump
                 AND shadow_lead   (all four required)
refined_usable : once_converged AND main_err_ratio < 0.7 AND erle_slope > -5
coarse_learning: default

ne_evidence    : dt_energy > 0.3 OR dt_shadow > 0.5
main_err_jump  : ratio > 2 × baseline (baseline = downward-only EMA of
                 main_err_ratio while in refined_usable)
shadow_lead    : main_err / shadow_err > 1.5
```

Phase 3 — wired `dt_advisory_enabled=True, dt_advisory_use_p3f_state=True`
with the existing P3e hit-and-hold plumbing: `mu *= 0.3` for 400 ms
post-hit. Direct config smoke confirmed gate fires 827 times on 7GT
across all 6 states.

## Results

| Bucket | v3.10.4 | V3 (P3e last) | P3f V1 | Δ vs baseline |
|---|---:|---:|---:|---:|
| FS_static echo  | 3.641 | 3.630 | **3.619** | **−0.022** ← fails ±0.02 |
| FS_movement echo| 3.704 | 3.688 |   3.693 | −0.011 ✓ |
| DT_static deg   | 2.328 | 2.335 |   2.341 | +0.013 |
| DT_movement deg | 2.370 | 2.388 |   2.384 | +0.014 |
| NE deg          | 4.013 | 4.010 |   4.010 | flat |
| 7GT doubletalk echo / deg | 3.366 / 3.895 | 3.366 / 3.895 | **3.366 / 3.895 (identical)** | — |

## Findings

### 1. The intervention is the wrong tool, not the gate

**The state classifier fires 827 times on 7GT, with all six states
observed across the run, and the AECMOS score does not move at all.**
This is the strongest single result of the P3e/P3f sweep: gating
`mu *= 0.3` on the right frames was the user's hypothesis-implied fix
for the 7GT 24–36 s ERLE collapse, and it produces zero bench movement.

Possible mechanisms (need follow-up before any new gate is tried):

- The taps are *already* polluted by the time `suspicious_dt` first
  fires (latching requires `refined_usable` to have been seen first;
  the latch may simply lag the contamination onset).
- `mu *= 0.3` is too gentle to prevent further pollution once the
  filter is already off the echo path; an actual freeze
  (`mu = 0` or shadow→main copy) may be required.
- AECMOS on this case is dominated by something downstream of the
  filter (RES render-based mode at 99 %, per P3d trace); reducing
  mu does not change the RES path the score sees.
- The post-filter (RES) may be dominating the audible artifacts
  AECMOS scores against, making filter-tap quality second-order.

### 2. The trace 5-case set under-represented FS divergence

FS_static and FS_movement smoke showed `0 / 1191` and `0 / 2284`
suspicious_dt frames — perfect on the 5-case audit. The full 800-case
bench delivers FS_static −0.022 anyway, meaning the classifier *does*
fire on FS in some real cases the 5-case set didn't cover. The
`refined_latched + main_err_jump + shadow_lead + ne_evidence`
conjunction is FS-safe on the cases we hand-picked but not on the
bench distribution. This is a sampling lesson for future P3 work: a
5-case fixed set is enough to falsify a classifier, not enough to
verify FS-safety.

### 3. State model is sound; the data is honest

DT_static and DT_movement smoke showed 0 % `suspicious_dt` on
NE-evidence frames — `shadow_advantage` stays ≈ 1.0 in those cases,
i.e. the shadow filter does *not* outperform main during NE there.
This corrects a hypothesis carried over from P3e V1/V2: those V1/V2
DT-bucket gains (+0.07, +0.05 deg) had to be coming from FS-side
false fires, not from genuine DT detection. The state model exposed
this by being explicit about what it can and cannot see; the
classifier is honest about negative information.

### 4. The single thread that *does* support a future direction

The P3d trace identified `using_render = 99 %` post-alignment on 7GT.
Coupled with the present finding that mu reduction does not move the
score, the implied next step is **a state-gated RES path switch**, not
another adaptation gate. Specifically: when the filter enters
`refined_usable` and the delay is solid (= `usable_linear == True`),
prefer linear residual (`echo_spec`-driven) over render-based for
several seconds; let the score speak. This is a behaviour change in
RES, deferred until separately scoped.

## Decision

Per user rule "if FS regresses → off": the toggle stays default off.
`dt_advisory_use_p3f_state` is retained in `AecConfig` for future
A/B comparison but does nothing in the shipped build.

## What to keep / what to discard

**Keep:**

- All Phase 1 trace fields (`main_err_ratio`, `shadow_err_ratio`,
  `p3f_shadow_advantage`, `erle_slope_db_per_s`, `post_reset_age_ms`,
  `filter_state`, `usable_linear`) and the `--trace-aec-state` flag.
  Zero runtime cost (only computed when diag is read), and the next
  RES experiment (state-gated path switch) needs `usable_linear`
  anyway.
- The state classifier and its priority order. Use it as the
  selector for any future state-gated action (P3g RES switch).

**Don't repeat:**

- Single-threshold per-frame DT advisories (P3e V1/V2/V3).
- mu reduction gated on per-frame DT signals (P3e/P3f V1 here).
- 5-case fixed audit alone for FS-safety verification — combine with
  bench of a non-target slice (e.g. all FS_static) before claiming
  FS-safety.

## Files

- Bench: `/tmp/bench_p3f_v1` (wavs), `/tmp/bench_p3f_v1_scores/`
- Trace driver: `python/run_one_case.py --trace-aec-state`
- Audit: `/tmp/p3f_audit.py`
- Trace cases: `/tmp/p3f_*.csv` (5 cases)
- Code: `python/aec.py` `AecConfig.dt_advisory_use_p3f_state`,
  classifier ≈ line 5215, advisory dispatch ≈ line 4622.
