# v3.18 Phase B Closeout — Filter misadjustment + ScaleFilter (2026-05-16)

**Verdict**: CLOSED — CANNOT SHIP. Substrate retained default-OFF.
B.4 fire-rate gate FAIL on 61-case trace (0 fires across all cases).

## Scope tested

B.1 design + B.2 estimator state + B.3 ScaleFilter primitive all
delivered as documented in [docs/v3_18_b1_misadjustment_design.md](v3_18_b1_misadjustment_design.md).

- B.2 added `AecConfig.filter_misadjustment_*` × 9 fields + lazy
  AEC state init (`_misadjustment_smoothed`, `_stable_count`,
  `_hangover_remaining`, `_fire_count`); byte-equal flag-OFF 5/5 PASS
- B.3 added `PBFDAF.scale_filter(scale)` + `PBFDKF.scale_filter(scale,
  scale_p=False)` methods; estimator + trigger methods wired into
  `AEC.process()` just before RES; byte-equal flag-OFF preserved
- B.4 fire-rate trace on 61-case (60 D-γ cohort + qNvSMyU)

## B.4 fire-rate gate (must pass ≥ 10 total fires)

```
n_cases=61  total_fires=0  mean=0.00  max=0  min=0
cases_with_fire=0/61
```

**0 fires across all 5 buckets** — gate FAIL.

## Mechanism failure analysis (3-case deep trace)

| Case | bucket | smoothed_mean | smoothed_min | refined_usable% | stable_max | fires |
|---|---|---:|---:|---:|---:|---:|
| N2rQLbnp_FS_static | FS_static | 10.91 | 0.105 | 38.0% | 15 | 0 |
| qNvSMyU_FS_static (cohort tail) | FS_static | 42.70 | 0.067 | **0.0%** | 0 | 0 |
| sUQrHEPA_DT_static | DT_static | 38.89 | 0.281 | **0.0%** | 0 | 0 |

Two structural mismatches with AEC3 design:

### 1. Smoothed ratio range is inverted vs AEC3 expectation

AEC3's `FilterMisadjustmentEstimator` assumes `echo_psd / (far_psd × ERL)`
clusters near 1.0 with under-modelling pulling the smoothed value
**below** 0.5. Our pipeline's smoothed value clusters at **10-42×**,
with brief excursions to 0.07-0.3.

Root cause: our `_erl_estimate` is conservatively low (starts at 0.1,
capped to 0.3 after EPC events, slow upward adaptation). The actual
filter-derived `echo_psd / far_psd` ratio is much higher than 0.1×0.3
= 0.03 predicts. So the AEC3-style under-model trigger (`smoothed <
0.5`) becomes structurally rare.

This is the same pattern as the v3.14 Arc P (per-band ERL EMA) audit
finding — `_erl_estimate` is a **conservative gain budget**, not a
"true ERL" measurement. AEC3's estimator depends on the latter.

### 2. `refined_usable` state coverage is low

AEC3 design assumes the filter spends majority time in `refined_usable`
(equivalent of their `filter_analyzer.consistent_estimate` AND no recent
reset). Our `_prev_filter_state == 'refined_usable'` was hit:
- 38% on the best FS_static case (N2rQLbnp)
- **0%** on qNvSMyU cohort tail (never converges to refined_usable)
- **0%** on the DT case (DT state machine doesn't enter refined_usable
  during near-end speech)

Even on the 38% case, `stable_count_max = 15` — never reached the
required 30 consecutive frames. State transitions between `refined_
usable` / `coarse_learning` / `idle` happen too often for the stability
gate to fire.

Phase C (FilteringQualityAnalyzer + AecState centralisation) is the
gating arc that would make `refined_usable` coverage match AEC3 design.
Without C, B's trigger gate is structurally silent.

## Risk validation (against B.1 §4)

| Risk # | Prediction | B.4 evidence |
|---|---|---|
| R1 | False-positive scale corrupts W during DT | Moot — trigger never fires |
| R2 | EPC event triggers spurious scale | Moot — trigger never fires |
| R3 | `_erl_estimate` stale → bad ratio | **Confirmed as root cause** — `_erl_estimate` is structurally low; ratio is structurally high. The stability gate doesn't save us because the gate ALSO never passes |
| R4 | Scale-P + P-override interaction | Moot |
| R5 | qNvSMyU never enters stable regime | **Confirmed** — 0% refined_usable on qNvSMyU |
| R6 | Cumulative interaction with Arc P | **Partially confirmed** — Arc P's ERL feeds our `_erl_estimate`; both are conservative-by-design. AEC3's metric needs a different ERL semantics |

## §6 kill criterion adjudication

> B.4 fire-rate gate fail → close, substrate retained.

B.4 produced 0 fires across 61 cases — far below the ≥10-fire pass
threshold. Per B.1 §6 + §0.4 negative-result protocol, **close B**.

No B.5 60-case AECMOS bench needed (mechanism is silent → flag-ON
output ≈ flag-OFF output anyway). Wall-clock saved by the upfront
fire-rate gate per B.1 §3.7 design lesson.

## Substrate retained (default OFF)

- `AecConfig.filter_misadjustment_*` × 9 fields
- `PBFDAF.scale_filter(scale)` + `PBFDKF.scale_filter(scale, scale_p)`
  methods (filter-class API additions; harmless when never called)
- `AEC._update_misadjustment_estimator()` + `_check_and_apply_misadjustment_scale()`
  methods (return early when flag is OFF)
- AEC state slots (`_misadjustment_smoothed` et al.) lazy-init only
  when flag is True
- `AEC_FILTER_MISADJUSTMENT*` env-var overrides in eval bench

Re-enable preconditions:
1. Phase C FilteringQualityAnalyzer must raise `refined_usable`
   coverage from ~0-38% to AEC3-level (~60-80%). Without that the
   stability gate is structurally silent
2. ERL semantics revision — either a "true ERL" measurement
   (distinct from `_erl_estimate` budget) OR re-derive the
   misadjustment metric to match our `_erl_estimate` scale (e.g.
   threshold > 2.0 for over-modelling, but then meaning inverts and
   the action direction reverses too)
3. The threshold needs co-design with our pipeline rather than
   import from AEC3

## Implications for v3.18 cycle

- **4-of-4 AEC3-port arcs in v3.18 have closed CANNOT SHIP**
  (D-γ + F + A + B). Pattern: every AEC3 mechanism we import depends
  on AEC3's surrounding design assumptions (per-band NE detector
  + AecState lifecycle + filter quality analyzer + delay-change
  semantics + ERL semantics) that we lack.
- **Phase C is the unblock** — FilteringQualityAnalyzer + AecState
  centralised ADT supplies the missing co-design context. Once
  `refined_usable` semantics align, Phase B / Phase F substrate
  can be revisited as Phase C follow-ups.
- v3.19+ matched-filter delay detector remains the v3.20 gating arc
  for shadow-side and event-classification work.
- Phase E (preset promotion) becomes the only remaining shippable
  v3.18 arc, but it depends on something to promote. Without B
  shipping, E's scope shrinks.

## Recommendation for next sprint

Per the 4-arc pattern, **stop AEC3-port arc trials** until Phase C
supplies the co-design context. Two paths from here:

**Path 1 — Phase C kickoff now (recommended)**: skip Phase B's
remaining sprints (B.5 unnecessary, B.6 blocked by B.4) and move
straight to Phase C design sprint (C.1 — FQA + AecState ADT). C is
larger LOE (9-12 sprints) but unlocks the remaining D-γ / F / A / B
substrate retroactively.

**Path 2 — Phase E audit-only sprint**: re-audit which BALANCED-only
substrate flags could promote to other presets without depending on
new mechanism. Lower-LOE; doesn't advance C but produces some ship
value while user decides on C.

**Default**: Path 1 unless user redirects.

## Cross-references

- [docs/v3_18_b1_misadjustment_design.md](v3_18_b1_misadjustment_design.md) — B.1 design lock
- [docs/v3_18_a_closeout.md](v3_18_a_closeout.md) — Phase A closeout (preceding arc)
- [docs/v3_18_f_closeout.md](v3_18_f_closeout.md) — Phase F closeout
- [docs/v3_18_d_gamma_closeout.md](v3_18_d_gamma_closeout.md) — Phase D-γ closeout
- [docs/v3_18_plan_revision_2026_05_15.md §11](v3_18_plan_revision_2026_05_15.md#L244) — §11 fallback chain
- [docs/aec3_reference.md §6.2](aec3_reference.md#L811) — AEC3 FilterMisadjustmentEstimator reference
