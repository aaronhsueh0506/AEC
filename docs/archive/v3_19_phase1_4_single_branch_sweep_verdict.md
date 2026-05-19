# v3.19 Phase 1.4 — Single-branch sweep verdict (2026-05-16)

**Status**: CLOSE — kill criterion triggered. **No single branch met
PRIMARY (DT_static Δdeg ≥ +0.020).** No combined config (1.5)
required per kill criterion.
**Disposition**: Phase 1 CLOSED CANNOT SHIP. 3 per-branch substrate
flags retained default-OFF per §0.4. **Pivot to Phase 3** (B
FilterMisadjustment retry — independent of Phase 1) per plan §Phase 1
kill criterion.

## 1. 60-case AECMOS sweep — 4 configs, all rendered + scored

Setup: BALANCED preset, fl=832, cng=True, j=4 parallel,
`tools/research/v3_15_subset_cases.txt` (60 cases). All 4 configs
have C.A + C.B + C.C audit-only ports ON
(`AEC_FILTER_ANALYZER=1 AEC_FILTER_QUALITY=1 AEC_AEC_STATE=1`); only
the per-branch flag differs.

### Baseline bucket means (no per-branch flags)

| Bucket | n | echo | deg |
|---|---:|---:|---:|
| FS_static | 11 | 3.507 | 4.999 |
| FS_movement | 11 | 3.719 | 4.999 |
| DT_static | 14 | 4.439 | 2.377 |
| DT_movement | 13 | 4.003 | 2.421 |
| NE | 11 | 4.999 | 3.747 |

### Per-branch Δ vs baseline

#### G1 — `c_e_branch_dt_per_bin_use_fq_usable=True`

| Bucket | Δecho | Δdeg | hard bar | verdict |
|---|---:|---:|---|---|
| FS_static | **−0.046** | −0.000 | ≥ −0.010 | **FAIL** (4.6× over) |
| FS_movement | **−0.062** | +0.000 | ≥ −0.010 | **FAIL** (6.2× over) |
| DT_static | −0.017 | **−0.001** | PRIMARY ≥ +0.020 | **FAIL** (no movement) |
| DT_movement | +0.000 | **+0.071** | ≥ −0.005 | PASS (significant gain) |
| NE | +0.000 | −0.008 | ≥ −0.005 | borderline |

Score `(Δdeg_DT_static + Δdeg_DT_movement) − (|Δecho_FS_static| +
|Δecho_FS_movement|)` = (−0.001 + 0.071) − (0.046 + 0.062) =
**−0.038** (negative).

#### P1 — `c_e_branch_coh2_ema_use_fq_usable=True`

| Bucket | Δecho | Δdeg | hard bar | verdict |
|---|---:|---:|---|---|
| FS_static | **−0.024** | −0.000 | ≥ −0.010 | **FAIL** (2.4× over) |
| FS_movement | **−0.052** | +0.000 | ≥ −0.010 | **FAIL** (5.2× over) |
| DT_static | −0.008 | **−0.005** | PRIMARY ≥ +0.020 | **FAIL** (no gain) |
| DT_movement | −0.008 | **+0.018** | ≥ −0.005 | PASS (modest) |
| NE | +0.000 | −0.001 | ≥ −0.005 | PASS |

Score = (−0.005 + 0.018) − (0.024 + 0.052) = **−0.063** (negative).

#### R1 — `c_e_branch_force_render_use_fq_usable=True`

| Bucket | Δecho | Δdeg | hard bar | verdict |
|---|---:|---:|---|---|
| FS_static | **−0.056** | +0.000 | ≥ −0.010 | **FAIL** (5.6× over) |
| FS_movement | **−0.153** | +0.000 | ≥ −0.010 | **FAIL** (15.3× over) |
| DT_static | −0.027 | **−0.006** | PRIMARY ≥ +0.020 | **FAIL** (no gain) |
| DT_movement | −0.016 | **+0.076** | ≥ −0.005 | PASS (significant gain) |
| NE | +0.000 | +0.000 | ≥ −0.005 | PASS |

Score = (−0.006 + 0.076) − (0.056 + 0.153) = **−0.139** (worst negative).

### Ranking (least-bad first)

| Rank | Branch | Score | DT_static Δdeg | FS_movement Δecho |
|---:|---|---:|---:|---:|
| 1 | G1 | −0.038 | −0.001 | −0.062 |
| 2 | P1 | −0.063 | −0.005 | −0.052 |
| 3 | R1 | −0.139 | −0.006 | −0.153 |

**All scores negative.** No single branch with positive
(Δdeg − |Δecho|).

## 2. Hypothesis adjudication

Phase 1.1b §4.2 hypothesis (predicted source per branch):
- **R1 alone: BIG FS regression, neutral DT** — **CONFIRMED** (FS_movement
  −0.153 matches v3.18 C.E whole-arg FS_static −0.151 magnitude;
  R1 IS the dominant FS regression driver)
- **G1 alone: DT win, FS unchanged** — **PARTIALLY DISCONFIRMED**
  (G1 produces DT_movement +0.071 win as expected, but ALSO produces
  FS regression −0.046/−0.062 — F3.1 v3 mic-excess evidence is more
  fragile than predicted; broadening from filter_converged ~5% to
  fq_usable ~50%+ over-attributes NE in transitional FS frames)
- **P1 alone: small FS gain** — **DISCONFIRMED** (predicted small FS
  GAIN; actual is FS regression −0.024/−0.052; coh2 EMA slow-rise
  tuning fires too often when fq_usable is ON, allowing transient
  echo through)

The v3.18 C.E whole-arg DT_static +0.214 source is **none of
G1/P1/R1 alone**. Most likely an INTERACTION effect:
- G1 + R1 combined: G1 changes nearend evidence at the gain stage,
  R1 changes whether residual estimator runs in linear or render mode
  → multiplicative effect on Wiener gain that lifts DT_static
- Or: the v3.18 C.E whole-arg also affected branches we didn't
  enumerate (e.g., audit-instrumentation paths at lines 3343-3392
  that COULD mutate dt_per_bin under different gating)

Worth noting: 60-case Δdeg has higher variance than 800-case;
±0.020 is within noise on 14 DT_static cases (n=14). v3.18 C.E
whole-arg was on a different 60-case set (per docs/v3_18_c_e_closeout.md
§3 — different cohort sizing). Cohort drift may mask any genuine
DT_static effect.

## 3. Kill criterion adjudication

Per [docs/v3_19_phase1_2_per_branch_flag_design.md](v3_19_phase1_2_per_branch_flag_design.md) §7
+ plan §Phase 1 kill:

> 1.4 sweep finds no single branch with positive (Δdeg − |Δecho|) →
> close Phase 1; pivot to Phase 2 only

**TRIGGERED.** All three branches have negative score. Phase 1
closes here.

Phase 1.5 combined config (planned next sprint) is **skipped** per
strict kill criterion. Rationale:
- G1 + P1 combined would add DT_movement (+0.071 + 0.018 ≈ +0.089)
  but also add FS_movement regression (−0.062 + −0.052 ≈ −0.114).
  Pareto wall holds.
- G1 + R1 combined would match v3.18 C.E whole-arg shape (similar
  Pareto trade-off magnitude).
- All-3 combined ≈ v3.18 C.E whole-arg result (already tested, FAIL).
- No combination credibly meets PRIMARY (DT_static ≥ +0.020) given
  every single branch is at −0.001 to −0.006 DT_static.

## 4. Disposition

### Phase 1 CLOSED — substrate retained per §0.4

Three per-branch flags stay in `AecConfig` default-OFF:
- `c_e_branch_force_render_use_fq_usable: bool = False`
- `c_e_branch_dt_per_bin_use_fq_usable: bool = False`
- `c_e_branch_coh2_ema_use_fq_usable: bool = False`

Plus 3 env-var bridges. All flag-OFF byte-equal verified at Phase 1.3
(5/5 md5 PASS). Substrate ready for v3.20+ if Phase 0.1 Volterra
arc surfaces a per-branch rationale (e.g., Volterra detector gates a
specific per-branch combination).

### Phase 2 (C.D-α retry combined with Phase 1 wiring) NOT viable

Phase 2 explicitly depended on a Phase 1 winning combo to layer C.D-α
on top of. Without Phase 1 winner, Phase 2 reduces to "re-run v3.18
C.D-α alone" which is already CLOSED (silent failure). **Phase 2
SKIPPED.**

### Phase 3 (B FilterMisadjustment retry with fq_usable) PROCEEDS

Independent of Phase 1 outcome. Uses C.B's `fq_usable` substrate
directly to fix v3.18 Phase B's silent failure (0/61 fires from
`refined_usable` 5% coverage gap). Per plan §Phase 3.

### Phase 0 auto-triggers per Phase 0.1 §5

Conditions:
- ✓ Phase 1 closes early (sprint 1.4 kill, ~2 sprints saved
  from 1.5/1.6/1.7)
- ✓ Phase 0.4 fold-in in Phase 1.1b SURVEY only — no SuppressionGain
  port absorbing the freed budget
- Pending: user §0.7 mid-cycle authorisation for Volterra
  Option (B) design lock fold-in (5 sprints doc-only) OR D-γ.fast
  retry (4 sprints, 60-case re-bench)

Recommended order for freed budget:
1. **Phase 3 first** (B FilterMisadjustment retry, 4-6 sprints) — has
   strongest substrate-on-disk readiness, independent risk profile
2. **Then Volterra Option (B) design lock** OR **D-γ.fast retry** —
   user choice at Phase 4 closeout

## 5. Cross-references

- [docs/v3_19_phase1_2_per_branch_flag_design.md](v3_19_phase1_2_per_branch_flag_design.md) — design lock + kill criterion
- [docs/v3_19_phase1_3_byte_equal_verdict.md](v3_19_phase1_3_byte_equal_verdict.md) — implementation
- [docs/v3_19_phase1_1a_resfilter_branch_inventory.md](v3_19_phase1_1a_resfilter_branch_inventory.md) — branch enumeration
- [docs/v3_19_phase1_1b_aec3_suppressiongain_annotation.md](v3_19_phase1_1b_aec3_suppressiongain_annotation.md) — AEC3 mapping
- [docs/v3_18_c_e_closeout.md](v3_18_c_e_closeout.md) — v3.18 C.E whole-arg evidence
- [docs/v3_19_phase0_1_volterra_eval.md](v3_19_phase0_1_volterra_eval.md) — Volterra Option (B) auto-trigger conditions
- [docs/v3_19_phase0_3_d_gamma_retry.md](v3_19_phase0_3_d_gamma_retry.md) — D-γ.fast auto-trigger
- Bench results: `/tmp/v3_19_phase1_4_results_{baseline,g1,p1,r1}/result.md`

## 6. Sprint count update

| Sprint | Status |
|---|---|
| 1.1a | DONE (committed 0f57dac) |
| 1.1b | DONE (committed 0f57dac) |
| 1.2 | DONE (committed a7e7885) |
| 1.3 | DONE (committed a7e7885; byte-equal PASS) |
| 1.4 | DONE (this verdict; PRIMARY FAIL all 3 branches) |
| 1.5 | **SKIPPED** per kill criterion |
| 1.6 | **SKIPPED** (no shipping config) |
| 1.7 | **SKIPPED** (no shipping config) |

Phase 1 actual: **5 sprints** (vs 9-13 planned). Saved 4-8 sprints
for Phase 3 + Phase 0 auto-trigger work.
