# v3.18 Phase C.E Closeout — RES filter_converged → fq_usable migration (2026-05-16)

**Verdict**: CLOSED — CANNOT SHIP. Substrate retained default-OFF.
**Significance**: largest AECMOS movement of v3.18 cycle — but in Pareto
trade-off shape (massive DT NE preserve win, massive FS echo loss).
Hypothesis 3 from C.D-α closeout **strongly confirmed** but unshippable
at this scope. Selective-migration follow-up substantive.

## What landed

- `AecConfig.c_e_res_use_fq_usable: bool = False`
- 1-line argument switch at `aec.py:8462` — when flag-ON, RES.process()
  receives `fq_usable` instead of `_filter_converged`
- Env-var `AEC_C_E_RES_FQ_USABLE`

## C.E.2 byte-equal: 5/5 PASS flag-OFF

## C.E.3 60-case AECMOS A/B (C.A+C.B+C.C+C.E all ON, C.D-α OFF)

### Bucket means

| Bucket | Δecho | Δdeg | hard bar | verdict |
|---|---:|---:|---|---|
| FS_static | **-0.151** | +0.000 | Δecho ≥ -0.010 | **FAIL** (15× over) |
| FS_movement | **-0.049** | -0.000 | Δecho ≥ -0.010 | **FAIL** (5× over) |
| **DT_static** | -0.096 | **+0.214** | PRIMARY ≥+0.010 | **PASS** (21× over) |
| DT_movement | +0.033 | **+0.234** | Δdeg ≥ -0.005 | PASS |
| NE | +0.000 | +0.000 | Δdeg ≥ -0.005 | PASS |

### Per-case extremes (unprecedented magnitudes)

| stem | bucket | Δecho | Δdeg | comment |
|---|---|---:|---:|---|
| sUQrHEPA_DT_static | DT_static | -0.209 | **+1.2000** | largest single-case Δdeg of any v3.18 arc |
| oQK3bV_DT_movement | DT_movement | +0.170 | +0.7501 | DT win + echo improvement (win-win) |
| wZvBJ5R4_DT_movement | DT_movement | +0.224 | +0.5763 | win-win |
| uS9t2QYD_DT_static | DT_static | -0.337 | +0.4841 | Pareto: big DT win + big FS loss same case |
| x7VuXKV4_DT_static | DT_static | -0.335 | +0.3790 | Pareto |
| qNvSMyU (cohort tail) | FS_static | **-0.244** | -0.0000 | regression |
| N2rQLbnp_FS_static | FS_static | **-0.490** | +0.001 | worst |
| WTdBhXa0_FS_static | FS_static | -0.320 | +0.000 | bad |

Cases with Δdeg ≥ +0.1: **10/61**. Cases with Δecho < -0.05: **18/61**.

## Mechanism analysis — hypothesis 3 confirmed

C.D-α closeout's hypothesis 3:
> "RES still gates on `_filter_converged` (not `fq_usable`). Even if
> main filter improves from Q-boost, RES suppression decision doesn't
> change → no AECMOS movement."

**Strongly confirmed**. With RES receiving `fq_usable` instead:
- DT NE preserve dramatically improves (+0.2-1.2 Δdeg per case)
- FS echo dramatically regresses (-0.15-0.49 Δecho per case)
- Single-case wins of +1.2 Δdeg are unprecedented in this codebase

The mechanism: RES branches gated on `filter_converged` (~5% coverage)
were doing AGGRESSIVE echo suppression unconditionally because they
rarely fired the "trust linear estimate" path. With `fq_usable`
(52-86% coverage), those branches now activate frequently → RES
relaxes echo suppression in confident-filter regions. Effect:
- DT (NE present): relaxed suppression preserves NE speech → big Δdeg
- FS (NE absent): relaxed suppression lets through residual echo → big Δecho regression

## §6 kill criterion adjudication

Per C.E.1 §3:
- PRIMARY DT_static Δdeg ≥ +0.010: **PASS spectacularly** (+0.214)
- FS_static Δecho ≥ -0.010: **FAIL** (-0.151, 15× over)
- FS_movement Δecho ≥ -0.010: **FAIL** (-0.049, 5× over)
- worst-case ≥ -0.05: **FAIL** (-0.49 max, 18 breaches)

Worst-case + 2 guards fail → close per §0.4.

## Substrate retained (default OFF)

- `AecConfig.c_e_res_use_fq_usable: bool = False`
- AecState back-ref pattern (from C.C)
- 1-line argument switch at aec.py:8462
- Env-var `AEC_C_E_RES_FQ_USABLE`

## Re-enable preconditions / follow-up arcs

This finding is the **most actionable substrate** retained in the
entire v3.18 cycle. Three follow-up arcs that could ship from here:

### Follow-up 1 (medium LOE): RES branch-level migration
Don't migrate the whole `filter_converged` arg. Instead, identify which
specific RES branches inside `ResFilter._stage_gain_compute` /
`_stage_residual_estimate` benefit from `fq_usable` (DT NE preserve
branches) vs hurt from it (FS echo suppression branches). Migrate
only the helpful ones. Pareto-front-walking exercise — would likely
require 6-10 sprints (per-branch ablation).

### Follow-up 2 (small LOE): consume fq_usable as additional gate
Don't replace `_filter_converged`; ADD `fq_usable` as a second gate.
RES branches fire on `_filter_converged AND fq_usable` (intersection,
more conservative) OR `_filter_converged OR fq_usable` (union, more
permissive). Per-branch decision. Smaller risk, smaller win.

### Follow-up 3 (large LOE): per-bucket adaptive
Use `fq_usable` for DT-detected frames (`effective_dt > 0.3`); use
`_filter_converged` for FS-detected frames. Requires DT detector
co-reliability. Highest theoretical ceiling; most risk.

## Implications for v3.18 cycle

This closes the entire **executable** Phase C scope for v3.18:
- C.A / C.B / C.C: PASS audit-only, ship as default-OFF substrate
- C.D-α: CLOSED (silent)
- C.E surgical: CLOSED (Pareto trade-off identified)
- C.D-β / C.E broader / C.F all hit the same Pareto wall — re-bench
  rather than re-implement

**v3.18 closeout shape now**:
- Algorithm changes shipped: **0** (per all 6 phase closures)
- Default-OFF substrate shipped:
  - Phase 0 AEC3 reference docs (Phase 0)
  - F.1 event taxonomy + F.2 helpers (Phase F substrate)
  - D-γ Subband + Dominant NE detector substrate
  - Shadow NLMS conversion substrate (Phase A)
  - Filter misadjustment + ScaleFilter substrate (Phase B)
  - **C.A FilterAnalyzer + C.B FilteringQualityAnalyzer + C.C AecState
    extension (Phase C audit infrastructure)**
  - **C.D-α leakage_diverged trigger + C.E RES migration substrate**
- v3.19+ priority backlog enriched with all 6 follow-up paths

The substrate-retained pattern is now substantive: **8 distinct
default-OFF arcs** ready for v3.19+ refinement with proper consumer-
side co-design.

## Cross-references

- [docs/v3_18_c_e1_consumer_migration_design.md](v3_18_c_e1_consumer_migration_design.md) — design
- [docs/v3_18_c_d_alpha_closeout.md](v3_18_c_d_alpha_closeout.md) — hypothesis 3 source
- [docs/v3_18_c1_aec_state_design.md](v3_18_c1_aec_state_design.md) — Phase C parent
- All other v3.18_*.md docs — full cycle reference set
