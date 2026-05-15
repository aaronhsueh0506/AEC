# v3.16 C6 — DelayEst audit verdict (Phase 1, 2026-05-15)

**Status**: AUDIT COMPLETE — verdict **H2 (CLOSE C6 audit per §0.4)**.
**Branch**: `feature/v3.16` (commit pending).
**Sprint**: v3.16 Phase 1.0 (per `~/.claude/plans/se-aec-aec-main-hazy-lynx.md` §10).
**Predecessor**: [`v3_16_c6_delay_est_audit.md`](v3_16_c6_delay_est_audit.md) (design).

---

## 1. Headline

**Verdict: H2** — DelayEst is NOT the upstream root cause for the v3.15
movement / cohort tail / DT-NE closures. v3.16 Phase 3-4 ROI estimates
stand; do NOT open delay-aware mechanism arc.

**Caveats** (noted for v3.16 closeout backlog):
1. One movement case (`0I0XMl3M`) shows extreme `estimated_delay` jumps
   (1230 → 4132 → 20 samples) during movement window → ERLE p5_bad
   −49.49 dB. Tactical "movement-rate DelayEst" candidate noted for
   v3.17 backlog (NOT v3.16, single-case audible target).
2. Cohort tail (`qNvSMyU`) multi-peak ambiguity is **sample-level**
   (top1_lag 1607 / top2_lag 1587, Δ = 20 samples / 1.25 ms; equal
   magnitudes). Not DelayEst-fixable — reflects true room acoustic
   (early reflection within fl coverage).
3. HK-2 `pcb1N` confirmed: **DelayEst correctly identifies delay
   ~127 ms** (top1_lag 2030-2523, PAR mean 87.9, top2_par/top1_par
   = 0.36 well-separated). The issue is `fl=832` (52 ms) < delay span.
   **C9 reverb-aware scope CONFIRMED** (fl coverage / multi-path, not
   delay tracking).

---

## 2. Numerical results

### 2.1 Aggregate (10 Tier A cases)

| Metric | Value | H1 bar (≥) | H2 bar (<) | Determination |
|---|---:|---:|---:|---|
| Mean lead-lag attribution rate (filter-issue bursts with preceding DelayEst issue ≤ 250 ms) | **1.5 %** | 60 % | 30 % | **H2** |
| Mean delay-issue-in-bad-window rate (concurrent, not lead) | 43.7 % | 60 % | 30 % | MIXED-leaning-H1 *if interpreted alone* |
| H1_STRONG cases | 0/10 | — | — | — |
| H2 cases | 5/10 | — | — | — |
| MIXED cases | 5/10 | — | — | — |

**Interpretation**: lead-lag is essentially zero — DelayEst is NEVER
observed to drift / mis-lock / lose confidence FIRST before the filter
struggles. The 43.7 % concurrent rate is driven by **persistent
multi-peak ambiguity** in difficult cases (high `top1/top2 par ratio`),
which is not a tracking failure but a structural property of the
audio (multi-path / reverb). Per design doc §3.5 hard bar, lead-lag is
the primary indicator; concurrent is secondary.

### 2.2 Per-case table

| Stem | Verdict | Lead-lag attrib | Concurrent DE-in-FI | ERLE p5 bad (dB) | PAR p10 in bad | top1/2 p95 in bad |
|---|---|---:|---:|---:|---:|---:|
| qNvSMyU FS (cohort tail) | MIXED | 0.0% | 94.7% | **−33.88** | 14.62 | **0.980** |
| XqvGR01t DT_movement | H2 | 0.0% | 7.1% | −8.98 | 0.00 | 0.758 |
| pcb1N FS (HK-2) | H2 | 8.3% | 23.6% | −4.26 | 26.09 | 0.796 |
| XRTnTUjU DT (xrtntuju) | MIXED | 6.2% | 73.7% | −18.15 | 10.10 | **0.989** |
| 0I0XMl3M FS_movement | MIXED | 0.0% | 88.8% | **−49.49** | 6.44 | **0.986** |
| Hp5g1asac DT_movement | H2 | 0.0% | 0.0% | −0.95 | 47.44 | 0.277 |
| WH0jN3PY DT_movement | MIXED | 0.0% | 87.9% | −10.96 | 15.91 | **0.996** |
| jtYTdZm3 DT (F2.4) | H2 | 0.0% | 8.9% | −0.68 | 37.58 | 0.610 |
| OX2l6zV7 FS_movement | MIXED | 0.0% | 47.0% | −3.07 | 194.66 | 0.574 |
| Y91uE2t FS CTRL | H2 | 0.0% | 5.7% | +0.12 | 0.00 | 0.544 |

**Pattern**: MIXED cases share `top1/top2 par ratio p95 ≈ 0.98–1.00` →
multi-peak ambiguity dominates. But lead-lag = 0 in all MIXED cases —
DelayEst is concurrently ambiguous, not leading.

### 2.3 Critical observation: sample-level multi-peak in qNvSMyU

DelayEst trace dump for cohort-tail catastrophe case:

```
t=3.07s   L1=1607  L2=1587  L3=1626  P1=42.9  P2=42.4  P3=17.1
t=3.57s   L1=1607  L2=1587  L3=1626  P1=42.9  P2=42.4  P3=17.1
...
t=21.07s  L1=1587  L2=1621  L3=1604  P1=18.5  P2=11.7  P3=11.6
t=26.07s  L1=1626  L2=1607  L3=1587  P1=15.5  P2=14.3  P3=14.3
```

**Top1 and top2 lags differ by 20 samples (1.25 ms) and have nearly
identical PAR**. This is an **early-reflection signature**, not a
DelayEst failure. The filter trains on whichever peak DelayEst locks
to; the OTHER peak's echo bleeds through. ERLE −33 dB reflects this
sub-sample misalignment within `fl=832` coverage.

**Fix space**: NOT delay tracking. Possible candidates:
- Fractional-delay sub-sample interpolation (research-track, NOT
  v3.16).
- Increase filter `fl` beyond 832 (memory-bound; **violates
  v3.16 §8 invariant**).
- Wider Q in Kalman state during cohort tail (Arc M family
  — CLOSED v3.15).

These are all already-explored / out-of-scope, confirming the v3.15
cohort tail closure family was correct (no delay-aware path can fix
this).

### 2.4 0I0XMl3M extreme movement — narrow case worth noting

```
t=3.07s   L1=1230  L2=   1  L3=1248  P1=17.1  P2=13.5  P3= 7.7
t=8.07s   L1=4132  L2=4369  L3=4058  P1=13.6  P2=12.8  P3=12.8
t=13.57s  L1=4369  L2=4058  L3=4132  P1=14.1  P2=13.9  P3=13.6
t=19.07s  L1=4369  L2=4132  L3=4157  P1=11.1  P2= 9.6  P3= 9.0
t=23.57s  L1=   2  L2=  20  L3= 151  P1=11.1  P2= 8.0  P3= 5.9
```

**Estimated delay jumps**: 1230 → 4132 → 4369 → 2. PAR drops from 17
to 11. The 0.5-sec re-estimation interval (`period_seconds = 2.0`,
but trace fires at re-estimation events) clearly LAGS the actual delay
during fast movement.

**Filter impact**: ERLE p5_bad = −49.49 dB — extreme negative ERLE
implying the filter is adding 50 dB of artefact, consistent with
training on a stale or wrong delay.

**Not opening v3.16 arc** because:
- 1/9 bad cases (11 %) — narrow audible debt.
- No clean tactical fix without major DelayEst rewrite (faster update
  rate may help but introduces noise at low-confidence frames).
- 0I0XMl3M did NOT appear as a primary regression in any v3.15
  closure pack — it's a rare extreme-movement case.

**Backlog item**: "movement-rate DelayEst" — re-estimate every 0.25 s
instead of 2 s when motion detector indicates fast change. v3.17
candidate, LOW priority.

---

## 3. v3.16 plan implications

Per design doc §3.5 hard bar:

| Plan element | Status | Action |
|---|---|---|
| Phase 1.5 v3.16-A force_render OR-in | Gated on C6 PASS | **PROCEED** — v3.16-A's cohort-tail predicted Δecho +0.030 is NOT delay-driven (qNvSMyU multi-peak is sample-level, not coarse delay); Arc T S1 detector remains valid trigger |
| Phase 1.6 C5 per-state RES interface | Gated on Phase 1 progress | **PROCEED** — architectural foundation independent of C6 outcome |
| Phase 2 C2/C3/C4 | Gated on C5 | **PROCEED** — ROI estimates stand |
| Phase 3 v3.16-B ENR-path lift | Gated on C6 + C2 | **PROCEED** — cohort tail trigger unchanged by C6 verdict |
| Phase 4 C7 (Arc M retry) | Gated on C6 + Phase 3 | **PROCEED** — Arc M family's cohort tail damage is mechanism wall (Q steady-state), not delay-related |
| Phase 4 C8 (Arc G partial decay) | Gated on Phase 3 | **PROCEED** — Arc G gain-change detector unchanged |
| Phase 4 C9 (reverb-aware RES override) | Gated on C6 | **PROCEED, SCOPE CONFIRMED** — pcb1N case verifies the C9 mechanism (fl coverage / multi-path / NL), NOT delay tracking. C9 trigger = `top_ratio_p95 > 0.9 in DelayEst` is candidate (rather than `r < 0.15` only) — see §4 below |

**No phase ordering re-ordering required.** All Phase 1-4 candidate
ROI estimates and gates stand. The aggressive ordering (C6 → v3.16-A →
C5) per the 2026-05-15 user decision proceeds.

---

## 4. C9 scope refinement (Phase 4)

Original C9 design (per plan §1.4 / audit doc §4 Phase 4):
> "Detect persistently low mic-lpb cross-correlation (`r < 0.15`) over
> a sliding window with `far_power > threshold` → switch RES to
> FS-aggressive mode."

**C6 trace data refinement**: pcb1N audit reveals the trigger condition
can be CO-INDEXED by DelayEst trace fields already exposed:
- pcb1N `top1/top2 par ratio` mean 0.36 — well-separated (NOT trigger).
- qNvSMyU `top1/top2 par ratio` mean 0.87 — multi-peak (potential
  trigger for C9 if extended).
- 0I0XMl3M ratio mean 0.88 — extreme movement (likely NOT C9 target).

**Recommendation**: keep C9 trigger on mic-lpb cross-correlation
(orthogonal indicator), but ADD `DelayEst.top_ratio > 0.7 AND
estimated_delay > 0.8 × fl_samples` as auxiliary "fl undercoverage"
detector. Both signals already exist in trace surface, no new code.
Defer specific wiring to C9 sprint.

---

## 5. Substrate (committed)

- Audit script:
  [`tools/research/v3_16_c6_delay_est_audit.py`](../tools/research/v3_16_c6_delay_est_audit.py)
- Analyzer:
  [`tools/research/v3_16_c6_analyze.py`](../tools/research/v3_16_c6_analyze.py)
- Tier A case list:
  [`tools/research/v3_16_c6_tier_a_cases.txt`](../tools/research/v3_16_c6_tier_a_cases.txt)
- Trace JSONs (gitignored): `/tmp/v3_16_c6_audit/<stem>.json`,
  `summary.csv`, `attribution.json`.

No `python/aec.py` changes — audit consumed the existing
`trace_delay_est` flag + `AecStats` per-frame surface (zero behaviour
change vs production).

---

## 6. v3.17 backlog item (for v3.16 closeout)

**Movement-rate DelayEst** (LOW priority, single-case audible debt).
Origin: C6 audit case `0I0XMl3M0ECO0U1N0cJvpg_farend_singletalk_with_movement`
shows extreme `estimated_delay` jumps + ERLE p5_bad −49 dB during fast
movement. Mechanism: shorten `period_seconds` from 2.0 → 0.25 s when
motion detector (variance EMA proxy from Arc M closure) indicates
fast change. Acceptance: case `0I0XMl3M` ERLE p5_bad ≥ −20 dB AND no
800-case regression. Carry into v3.17 closeout queue.

---

## 7. Verdict signed-off

**H2** (audit verdict, per §0.4 negative-result acceptance protocol).
v3.16 Phase 1.5 v3.16-A and Phase 1.6 C5 can kick off immediately.
v3.16-A is the next sprint per aggressive ordering plan.
