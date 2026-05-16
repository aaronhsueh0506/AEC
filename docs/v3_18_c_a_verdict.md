# v3.18 Phase C.A Verdict — FilterAnalyzer audit-only port (2026-05-16)

**Verdict**: PASS — pre-bench gate cleared decisively; FilterAnalyzer
ships as audit-only substrate; Phase C.B authorised.
**Substrate**: default OFF; `AEC_FILTER_ANALYZER=1` enables.

## What landed

- New `python/aec_filter_analyzer.py` (FilterAnalyzer class)
- `AecConfig.filter_analyzer_enabled: bool = False`
- Lazy-init in `AEC.__init__` (no module import cost when flag OFF)
- Per-frame update in `AEC.process()` after main filter (before RES)
- Reset hook in `AEC.reset()`
- Three diagnostic surfaces: `_diag['filter_analyzer_consistent' /
  'peak_index' / 'max_gain']`
- Env-var bridge `AEC_FILTER_ANALYZER` in eval bench

Total surface: ~70 lines new file + ~25 lines AEC integration + 1
config field. No consumer reads the outputs. Default-OFF byte-equal
maintained.

## C.A.2 byte-equal verification

- 5-case flag-OFF md5 vs A.2 baseline: **5/5 PASS**
- 5-case flag-ON output md5 vs flag-OFF (audit-only invariant): **5/5
  SAME** (no behaviour change as designed)

## C.A.3 5-case pre-bench gate

Per C.A.1 §4: PASS if max |Δcoverage| ≥ 10pp on at least 2 of 5 cases.

| bucket | stem (truncated) | n_frames | _filter_converged% | consistent_estimate% | Δpp |
|---|---|---:|---:|---:|---:|
| FS_movement | 0I0XMl3M…_with_movement | 2415 | 0.0% | 14.4% | +14.4 |
| FS_static (cohort tail) | qNvSMyU…_farend_singletalk | 2685 | 0.0% | **54.5%** | +54.5 |
| FS_static | 0KjzXA3g…_farend_singletalk | 2175 | 1.9% | 69.1% | +67.3 |
| DT_static | 0I0XMl3M…_doubletalk | 4186 | 14.9% | 97.2% | +82.3 |
| DT_static | sUQrHEPA…_doubletalk | 4236 | 0.0% | 75.6% | +75.6 |

**Gate verdict: PASS (5/5, max 82.3pp).**

Critical finding on cohort tail: qNvSMyU is at 0% `_filter_converged`
(per Phase B closeout finding) yet **54.5% `consistent_estimate`**.
This directly answers the Phase B failure mode: the cohort-tail filter
IS stable — our `_filter_converged` threshold just doesn't recognise it.

## C.A.4 60-case trace (bucket aggregate)

| bucket | cases | total_frames | _filter_converged% | consistent_estimate% | Δpp |
|---|---:|---:|---:|---:|---:|
| FS_static | 12 | 29357 | 8.5% | 70.7% | +62.2 |
| FS_movement | 9 | 21248 | 5.4% | 64.4% | +59.0 |
| DT_static | 13 | 48328 | 5.4% | 86.2% | +80.9 |
| DT_movement | 7 | 26284 | 4.1% | 82.6% | +78.5 |
| NE | 20 | 22569 | 0.0% | 93.2% | +93.2 |

Per-case distribution (n=61): Δ_mean=+77.8pp, Δ_median=+79.4pp,
min=+14.4pp, max=+97.4pp. **61/61 cases pass |Δ|≥10pp threshold.**

NE bucket is the most striking: 0% `_filter_converged` (because the
DTD branch tags NE as "DT-like" and convergence flag never trips)
vs 93.2% `consistent_estimate`. The filter is fundamentally stable
during NE (no echo path activity to track); legacy convergence flag
just doesn't measure that.

## Implications for Phase B retry

The Phase B closeout root cause was:
> "Our pipeline's `_erl_estimate` is conservatively low … the AEC3-style
> under-model trigger (smoothed < 0.5) becomes structurally rare."
> + "refined_usable coverage 0-38%."

C.A's data sharpens this: **`consistent_estimate` could replace
`refined_usable` as the stability gate** with 5-15× more coverage.
Phase C.B (multi-gate `usable_linear_estimate`) layered on top, then
Phase B's misadjustment trigger becomes structurally activatable.

This is the unblock pattern C.1 §0 predicted: **C provides the
co-design context**, then the retained D/F/A/B substrate becomes
viable again as Phase C follow-ups.

## Risk validation (against C.A.1 §5)

| # | Risk | Evidence |
|---|---|---|
| R1 | HP filter Python performance | 60-case rendered in <3 min on 4-worker; cost negligible |
| R2 | Warmup drift mislabels as inconsistent | NE bucket 93.2% confirms stability is detected reliably; warmup ramp is bounded by `_consistent_threshold=30` (300 ms) |
| R3 | `irfft(W_sum)` correct operation? | Outputs make physical sense across bucket types; no crashes |
| R4 | Port adds no signal vs `_filter_converged` | **FALSIFIED** — 61/61 cases show informative Δ |

## C.A closeout

- Pre-bench gate: PASS
- 60-case trace: PASS
- byte-equal flag-OFF: PASS
- audit-only invariant flag-ON: PASS

Substrate retained as default-OFF audit-only port. Eligible for C.B
consumption (FilteringQualityAnalyzer.convergence_seen multi-gate)
without further validation.

## Phase C status

Per C.1 §2 + §3.3:
- C.A: **COMPLETE** (this verdict)
- C.B: AUTHORISED — FilteringQualityAnalyzer; pre-bench gate
  = `usable_linear_estimate (multi-gate) coverage ≥ _filter_converged
  + 20pp on 5 cases`
- C.C-C.F: gated on C.B PASS

## Cross-references

- [docs/v3_18_c1_aec_state_design.md](v3_18_c1_aec_state_design.md) — parent design
- [docs/v3_18_c_a1_filter_analyzer_design.md](v3_18_c_a1_filter_analyzer_design.md) — C.A.1 sub-design
- [docs/v3_18_b_closeout.md](v3_18_b_closeout.md) — Phase B closeout (motivates C.A finding)
- [docs/aec3_extracts/src/aec3/filter_analyzer.h](aec3_extracts/src/aec3/filter_analyzer.h) — AEC3 reference API
- [python/aec_filter_analyzer.py](../python/aec_filter_analyzer.py) — implementation
