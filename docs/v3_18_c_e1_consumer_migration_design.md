# v3.18 Phase C.E.1 — RES filter_converged → fq_usable migration (design, doc-only) — 2026-05-16

**Status**: C.E first surgical migration. Tests Phase C.D-α closeout
hypothesis 3 ("RES still gates on `_filter_converged` so doesn't react
to upstream improvements"). If wrong, the migration framework itself
needs different scope.

## 0. Why surgical (1 consumer) before broad (~120 consumers)

The 4-of-4 closure pattern + C.D-α's silent downstream are convergent
evidence: substrate alone doesn't move AECMOS; consumer-side reaction
does. **One surgical migration with maximum AECMOS leverage** tests
this directly. If even RES migration produces no AECMOS movement, then
the broader 120-consumer C.E becomes unproductive — pivot earlier
rather than burn 4-6 sprints on a doomed framework.

Hypothesis: RES is the AECMOS-dominant consumer (per memory
`feedback_aec_code_review_accuracy.md` + multiple v3.13/v3.14 verdicts:
RES dominates final output magnitude vs linear filter).

## 1. Scope: AEC → RES `filter_converged` argument

### 1.1 Today's call (aec.py:8460-8467)

```python
final_output = self.res.process(
    raw_output, eff_echo_spec, far_power, self.filter.far_spec,
    filter_converged=self._filter_converged,
    erle_factor=erle_factor,
    dt_indicator=float(dt_indicator),
    ...
)
```

Inside `ResFilter.process()`, this `filter_converged` flag controls
multiple internal branches (per grep: 24 occurrences of
`filter_converged` reads in `aec.py` — many inside RES stage methods).

### 1.2 Migrated call (flag-ON)

```python
if (self.config.c_e_res_use_fq_usable
        and self._aec_state is not None
        and getattr(self._aec_state, '_aec_ref', None) is not None):
    _fc_arg = self._aec_state.fq_usable()
else:
    _fc_arg = self._filter_converged
final_output = self.res.process(
    raw_output, eff_echo_spec, far_power, self.filter.far_spec,
    filter_converged=_fc_arg,
    ...
)
```

When flag-ON AND C.A+C.B+C.C all wired, RES sees `fq_usable` (typically
52-86% True on FS/DT vs ~5% for legacy `_filter_converged`).

### 1.3 Why this might help OR hurt

**Could help**: RES has gates like "if filter_converged: trust linear
estimate more; less aggressive Wiener gain." Today these almost never
trigger (5% coverage). With `fq_usable` (60-80%), they trigger far
more often → potentially better suppression on FS, better NE preserve
on DT.

**Could hurt**: RES's per-state branches were CALIBRATED for the
~5% coverage. Suddenly triggering 60-80% might over-suppress (FS gain
> nominal) or under-protect NE (gating on fq_usable changes when NE
preserve fires).

### 1.4 Bench-direct verdict design

There's no cheap pre-bench gate for this — the value test IS the bench.
Run 60-case A/B directly; if PRIMARY moves positively, generalise the
migration framework. If neutral/negative, close C.E with conclusion
"hypothesis 3 falsified."

## 2. Sprint sequence

| Sprint | Action | Output |
|---|---|---|
| C.E.1 | This design doc | doc only |
| C.E.2 | `AecConfig.c_e_res_use_fq_usable` flag + 1-line migration at aec.py:8462 + 5-case byte-equal flag-OFF | byte-equal flag-OFF md5 PASS |
| C.E.3 | 60-case AECMOS A/B (no cheap gate; bench IS the test) | verdict |
| C.E.4 | If PASS: scope broader migration plan (D.D revisit, other consumers); else close C.E entire | doc or closeout |

## 3. Hard bar (C.E.3 60-case)

Anchored on prior closures:
- **PRIMARY**: DT_static Δdeg ≥ +0.010 (same as C.D-α — the consumer-
  migration hypothesis is exactly that RES NEEDS to know fq_usable to
  improve DT NE preserve)
- Guards: DT_movement Δdeg ≥ -0.005, FS_static Δecho ≥ -0.010,
  FS_movement Δecho ≥ -0.010, NE Δdeg ≥ -0.005
- Worst-case: no sample Δecho < -0.05

**Kill criterion** (per §0.4 + 5-of-5 close precedent if this fails):
- PRIMARY FAIL OR worst-case breach → close C.E entire
- This is the **decision point for Phase C overall**: if RES migration
  doesn't help, C.E broader scope is moot, and C closes with substrate
  retained (C.A/C.B/C.C ship as default-OFF audit infrastructure for
  v3.19+)

## 4. Confidence read

**60%** — same as C.D-α.

Components:
- (+) Direct test of the clearest hypothesis (downstream consumer)
- (+) Minimal code surface (1-line argument switch)
- (+) RES is the AECMOS-dominant consumer
- (−) RES branches are co-tuned for ~5% coverage; flipping to 60-80%
  could destabilise
- (−) No way to validate cheap; must go straight to bench

## 5. Cross-references

- [docs/v3_18_c_d_alpha_closeout.md](v3_18_c_d_alpha_closeout.md) — motivates
- [docs/v3_18_c_b_verdict.md](v3_18_c_b_verdict.md) — fq_usable substrate
