# v3.18 Phase C.B Verdict — FilteringQualityAnalyzer audit-only port (2026-05-16)

**Verdict**: PASS — pre-bench + 60-case gates cleared; multi-gate
`fq_usable` substrate ships as default-OFF; Phase C.C authorised.

## What landed

- New `python/aec_filter_quality.py` (FilteringQualityAnalyzer class)
- `AecConfig.filter_quality_enabled: bool = False`
- Lazy-init in `AEC.__init__`
- Per-frame update in `AEC.process()` right after C.A's FilterAnalyzer
  update; reset hook in `AEC.reset()`
- 5 diagnostic surfaces: `_diag['fq_usable' / 'fq_startup_done' /
  'fq_reset_done' / 'fq_convergence_seen' / 'fq_far_active_recent']`
- `_epc_reset_fired_this_frame` per-frame helper (set True at all 3 EPC
  fire sites: delay_shift / EPV / shadow_rise)
- Env-var bridge `AEC_FILTER_QUALITY` in eval bench

## Design refinement during C.B.3 (2026-05-16)

C.B.3 v1 (initial implementation) tied `convergence_seen` to
`_filter_converged`, then **3 of 5 trace cases NEVER tripped** (qNvSMyU,
sUQrHEPA, 0I0XMl3M_FS_mv) — because `_filter_converged` was the
bottleneck we were trying to broaden. Gate still passed 2/5 ≥20pp but
was structurally weak.

**Fix**: refactored FQA API to take `convergence_signal` as caller-
supplied; AEC wires it from `FilterAnalyzer.consistent_estimate` (C.A
substrate) with `_filter_converged` as fallback when C.A is OFF.
AEC3-aligned: AEC3's `convergence_seen` reads from filter_analyzer's
consistent_estimate, not from a separate convergence flag.

After fix (C.B.3 v2): 5/5 cases ≥20pp, max +96.3pp.

## C.B.2 byte-equal verification

- 5-case flag-OFF md5 vs A.2 baseline: **5/5 PASS**
- 5-case flag-ON output md5 vs flag-OFF: **5/5 SAME** (audit-only)

## C.B.3 v2 5-case pre-bench gate (C.A + C.B both ON)

| bucket | stem | _filter_converged% | fq_usable% | Δpp |
|---|---|---:|---:|---:|
| FS_mv | 0I0XMl3M_FS_mv | 0.0% | 89.7% | +89.7 |
| FS_tail | qNvSMyU | 0.0% | **96.3%** | +96.3 |
| FS_st | 0KjzXA3g | 1.9% | 56.8% | +54.9 |
| DT_st | 0I0XMl3M_DT | 14.9% | 61.4% | +46.5 |
| DT_st | sUQrHEPA | 0.0% | 61.0% | +61.0 |

**Gate verdict: PASS (5/5, max +96.3pp).**

## C.B.4 60-case bucket aggregate

| bucket | cases | total_frames | _filter_converged% | fq_usable% | consistent_estimate% | Δusable−conv |
|---|---:|---:|---:|---:|---:|---:|
| FS_static | 12 | 29357 | 8.5% | **85.3%** | 70.7% | +76.8 |
| FS_movement | 9 | 21248 | 5.4% | **85.8%** | 64.4% | +80.4 |
| DT_static | 13 | 48328 | 5.4% | **52.3%** | 86.2% | +47.0 |
| DT_movement | 7 | 26284 | 4.1% | **53.1%** | 82.6% | +49.1 |
| NE | 20 | 22569 | 0.0% | 14.6% | 93.2% | +14.6 |

Two key observations:

1. **FS buckets**: `usable` (85%) > `consistent` (65-70%) because
   `convergence_seen` is sticky — once `consistent_estimate` flips True
   anywhere in the stream, the gate stays True. AEC3-aligned semantic:
   "filter HAS BEEN stable at least once + far currently active +
   startup done + no recent reset".

2. **NE bucket**: usable correctly stays at 14.6% because
   `far_active_recent` excludes pure NE (no far activity → linear filter
   estimate isn't needed). AEC3 gates the same way.

## Implications for Phase B / D / A retry

Phase B's failure mode was "0% refined_usable for cohort tail; 38% on
best case." With `fq_usable`:
- Cohort tail (qNvSMyU): 0% → **96.3%** (single-case)
- DT_static aggregate: 5.4% → **52.3%** (~10× coverage)
- FS_static aggregate: 8.5% → **85.3%** (~10× coverage)

Phase B's misadjustment trigger (was: `_prev_filter_state == 'refined_
usable'` + `_stable_count ≥ 30`) becomes structurally activatable when
migrated to use `fq_usable + fq_reset_done` instead. **This is the
unblock Phase C exists to provide.** Phase B retry is now viable as a
Phase C.D follow-up.

## Risk validation (against C.B.1 §3 expected ordering)

Expected: `_filter_converged ≤ fq_usable ≤ consistent_estimate`.

Observed: `_filter_converged ≤ fq_usable` ✓ (always)
Observed: `fq_usable` vs `consistent_estimate` — depends on stickyness
- FS buckets: `usable > consistent` because sticky convergence_seen
- DT buckets: `usable < consistent` because reset_timer + startup
  damp the count
- NE: `usable << consistent` because far_active_recent excludes NE

This is **correct** behaviour — multi-gate enforces both
"convergence has been seen" AND "current state is suitable for using
the linear estimate." Single peak-stability misses both nuances.

## Phase C status

- C.A: **COMPLETE** ([docs/v3_18_c_a_verdict.md](v3_18_c_a_verdict.md))
- C.B: **COMPLETE** (this verdict)
- C.C: **AUTHORISED** — AecState ADT wrapping C.A + C.B + scattered
  legacy state. Verification gate: AecState public-method outputs match
  legacy scattered fields on byte-equal trace
- C.D-C.F: gated as per C.1 §2

## Cross-references

- [docs/v3_18_c1_aec_state_design.md](v3_18_c1_aec_state_design.md) — parent design
- [docs/v3_18_c_b1_filtering_quality_design.md](v3_18_c_b1_filtering_quality_design.md) — C.B.1 sub-design (note: updated 2026-05-16 to convergence_signal param)
- [docs/v3_18_c_a_verdict.md](v3_18_c_a_verdict.md) — C.A verdict (provides convergence_signal source)
- [docs/v3_18_b_closeout.md](v3_18_b_closeout.md) — Phase B closeout (now retry-viable)
- [python/aec_filter_quality.py](../python/aec_filter_quality.py) — implementation
- [docs/aec3_extracts/src/aec3/aec_state.h](aec3_extracts/src/aec3/aec_state.h) — `UsableLinearEstimate` reference
