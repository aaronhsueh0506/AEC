# v3.19 Phase 1.5 — G1+P1 combined verdict (2026-05-16)

**Status**: CLOSE — kill criterion triggered. **G1+P1 combined FAIL
PRIMARY (DT_static Δdeg ≥ +0.020) AND FAIL FS guards (4-11× over).**
**Disposition**: Phase 1 fully CLOSED CANNOT SHIP. Pareto wall confirmed
at combined level. Substrate retained per §0.4. Pivot to Phase 3.

## 1. Why we ran 1.5 despite 1.4 kill

Phase 1.4 single-branch sweep closed per kill criterion (no single
branch with positive (Δdeg − |Δecho|) score). User raised: deg gains
historically scarce — DT_movement +0.071 (G1) and +0.076 (R1) are
the largest deg gains seen in 6 cycles. Worth verifying combined
config doesn't surface PRIMARY DT_static via interaction effect.

Tested combo: **G1 + P1** (least-FS-cost pair; R1 excluded as worst
FS-regression driver per 1.4 hypothesis confirmation).

Cost: 1 × 60-case bench (~2 min wall-clock). Cheap insurance against
prematurely closing.

## 2. 60-case AECMOS bench result

Setup: BALANCED preset, fl=832, cng=True, j=4 parallel, 60-case
subset. Baseline = Phase 1.4 baseline (C.A+C.B+C.C audit-only ON,
no per-branch flags).

`AEC_C_E_BRANCH_DT_PER_BIN_FQ_USABLE=1 AEC_C_E_BRANCH_COH2_EMA_FQ_USABLE=1`

### Bucket means (G1+P1 combined)

| Bucket | n | echo | deg |
|---|---:|---:|---:|
| FS_static | 11 | 3.433 | 4.999 |
| FS_movement | 11 | 3.604 | 4.999 |
| DT_static | 14 | 4.416 | 2.377 |
| DT_movement | 13 | 3.999 | 2.501 |
| NE | 11 | 4.999 | 3.739 |

### Δ vs baseline

| Bucket | Δecho | Δdeg | hard bar | verdict |
|---|---:|---:|---|---|
| FS_static | **−0.074** | −0.000 | ≥ −0.010 | **FAIL** (7.4× over) |
| FS_movement | **−0.115** | +0.000 | ≥ −0.010 | **FAIL** (11.5× over) |
| DT_static | −0.022 | **−0.000** | PRIMARY ≥ +0.020 | **FAIL** (no movement) |
| DT_movement | −0.004 | **+0.079** | ≥ −0.005 | PASS (significant) |
| NE | +0.000 | −0.008 | ≥ −0.005 | borderline (3× over) |

Score `(Δdeg_DT_total − |Δecho_FS_total|)` =
(−0.000 + 0.079) − (0.074 + 0.115) = **−0.110** (worse than G1 alone
score −0.038).

## 3. Interaction analysis

| Config | DT_static Δdeg | DT_movement Δdeg | FS_static Δecho | FS_movement Δecho |
|---|---:|---:|---:|---:|
| G1 alone | −0.001 | +0.071 | −0.046 | −0.062 |
| P1 alone | −0.005 | +0.018 | −0.024 | −0.052 |
| **Naive sum predicted** | **−0.006** | **+0.089** | **−0.070** | **−0.114** |
| **G1+P1 actual** | **−0.000** | **+0.079** | **−0.074** | **−0.115** |

**Combined ≈ additive**. No interaction surprise. DT_movement +0.079
≈ predicted +0.089 (slight saturation). FS regression −0.074 /
−0.115 essentially = predicted sum.

**Critical finding**: DT_static is **flat at −0.000** despite combining
two branches that each had near-zero DT_static effect. The v3.18 C.E
whole-arg DT_static +0.214 source is NOT in {G1, P1, R1, G1+P1}
single-or-pair space. Most likely sources:
- **R1×G1 interaction** (R1 changes residual_echo_psd magnitude,
  G1 changes per-bin NE evidence at gain stage; multiplicative
  through Wiener formula `g = nearend / (nearend + residual)`)
- Cohort drift (v3.18 C.E used different 60-case sampling per
  docs/v3_18_c_e_closeout.md §3)
- Audit-instrumentation paths (lines 3343-3392) that we excluded
  from per-branch enumeration but DO mutate dt_per_bin via
  `_audit_counters` side-effects

Testing G1+R1 combined was offered as Option 2 in user dialog but
dropped — would match v3.18 whole-arg shape (additive DT gain +
additive massive FS regression), no new information.

## 4. Kill criterion adjudication

Per [docs/v3_19_phase1_2_per_branch_flag_design.md](v3_19_phase1_2_per_branch_flag_design.md) §7
+ plan §Phase 1 kill:

> 1.5 combined config fails PRIMARY or any guard → close

**TRIGGERED on multiple guards**:
- PRIMARY DT_static Δdeg ≥ +0.020: FAIL (−0.000)
- FS_static Δecho ≥ −0.010: FAIL (−0.074)
- FS_movement Δecho ≥ −0.010: FAIL (−0.115)
- NE Δdeg ≥ −0.005: borderline FAIL (−0.008)

Phase 1 CLOSES.

## 5. deg-gain optionality verdict

User raised: deg gains historically scarce, worth preserving the
optionality. Phase 1.5 G1+P1 combined demonstrates:

**The deg gain IS real and consistent** (DT_movement +0.079 across
2 branches, ~6 cycles' worth of deg-gain in one substrate package).
**But cannot be shipped as-is** under current FS guards.

Three paths to revisit the deg substrate:

### Path A — v3.20+ Pareto-walking with broader branch space (RECOMMENDED)

Re-enumerate ResFilter branches WITH the audit-instrumentation paths
(lines 3343-3392) and check whether any of those mutate dt_per_bin
via fq_usable substitution. If new branches surface, run their sweep
on the same hypothesis: identify the source of v3.18 C.E +0.214
DT_static. The G1+P1 substrate stays default-OFF as substrate ready
for re-combination.

### Path B — Volterra arc gates on G1 conditional

When v3.20 Volterra ships (per Phase 0.1 Option B fold-in), Volterra
detector (NL frame fire) could become the gate that conditionally
enables G1+P1 only on NL frames. NL frames are FS-NL by definition,
so the FS regression that G1+P1 introduces would be a non-issue
(NL frame already needs special FS handling). DT NE preserve gain
remains.

### Path C — RES re-tune to absorb FS regression

Tune RES BALANCED preset's FS suppression knobs (e.g., `res_dt_reduction`,
`res_over_sub_base`) to be more aggressive when G1+P1 active, balancing
DT gain against FS regression. Out of v3.19 scope (Phase E preset
promotion would be the natural site).

All three paths preserve substrate. None ships in v3.19.

## 6. Disposition

### Phase 1 CLOSED — substrate retained

Three per-branch flags + 3 env-var bridges stay in tree, default-OFF.
G1+P1 combination tested + verdict committed for v3.20+ Path A/B/C
reference.

### Phase 1.6 / 1.7 SKIPPED

No shipping config. Listen verification + 800-case ship gate not
applicable.

### Phase 2 SKIPPED

Depended on Phase 1 winning combo. None exists.

### Phase 3 PROCEEDS

Per plan §Phase 3 — independent of Phase 1 outcome. B FilterMisadjustment
retry uses C.B's `fq_usable` directly to fix v3.18 Phase B silent
failure. **Next sprint focus.**

### Phase 0.1 Volterra Option (B) auto-trigger fired

Per v3.19 plan §Phase 0 binding semantics. Background agent already
spawned (V.1 literature survey + V.2 design lock + V.3 cohort sizing
in parallel with Phase 3 execution).

### Sprint count update

| Sprint | Status |
|---|---|
| 1.1a / 1.1b | DONE (committed 0f57dac) |
| 1.2 / 1.3 | DONE (committed a7e7885) |
| 1.4 | DONE (committed b401411; single-branch FAIL) |
| 1.5 | DONE (this verdict; combined FAIL) |
| 1.6 / 1.7 | SKIPPED |

Phase 1 actual: **6 sprints** (vs 9-13 planned). 3-7 sprints freed
for Phase 3 + Volterra design lock parallel work.

## 7. Cross-references

- [docs/v3_19_phase1_4_single_branch_sweep_verdict.md](v3_19_phase1_4_single_branch_sweep_verdict.md) — sprint 1.4 (single-branch sweep)
- [docs/v3_19_phase1_2_per_branch_flag_design.md](v3_19_phase1_2_per_branch_flag_design.md) — design lock + kill criterion
- [docs/v3_18_c_e_closeout.md](v3_18_c_e_closeout.md) — v3.18 whole-arg evidence (DT_static +0.214 source still unidentified post-1.5)
- [docs/v3_19_phase0_1_volterra_eval.md](v3_19_phase0_1_volterra_eval.md) — Volterra Option B auto-trigger conditions
- Bench results: `/tmp/v3_19_phase1_5_results_g1_p1/result.md`
