# v3.21 poor_coarse rescue (Gap A+B+C+D+E) 12-case AECMOS Verdict

**Variants compared**: M0 (anchor) · M_full_delay (current candidate) · M_full_rescue (M_full_delay + use_aec3_poor_coarse_rescue_copy=True; Gap C E_refined override included)

**Gate 0 byte-equal** (M0 vs plain BALANCED): PASS


## Per-case AECMOS

> Δ_full = M_full_delay − M0. Δ_rescue = M_full_rescue − M0. Δ_R−FD = M_full_rescue − M_full_delay.

| Watch | Case (short) | Bucket | Metric | M0 | M_full_delay Δ | M_full_rescue Δ | Δ(R−FD) | v3.21.6 |
|-------|-------------|--------|--------|-----|----------------|-----------------|---------|---------|
|    | ZJYUt0O0AEKSQ9LJ8z7t0A_doubl | DT_mvmt | deg | 3.058 | -0.148 | -0.078 | +0.070 | 2.270 |
| W1 | wVYSGVTTakih9twI4xlDWQ_doubl | DT_mvmt | deg | 3.205 | +0.046 | +0.198 | +0.152 | 2.741 |
| W2 | xFk7igecuke0R5JMfREyDg_doubl | DT_mvmt | deg | 2.881 | -0.988 | -0.793 | +0.195 | 2.319 |
| W4 | MYrVxVEMxkaE7OuyTUmI0Q_doubl | DT_static | deg | 1.724 | +0.096 | -0.335 | -0.431 | 2.166 |
|    | XRTnTUjU5kS0mejzCqyCiw_doubl | DT_static | deg | 2.751 | +0.069 | +0.039 | -0.030 | 3.950 |
|    | jtYTdZm3lUmFVNibJWq8YQ_doubl | DT_static | deg | 2.587 | +0.204 | +0.273 | +0.069 | 2.700 |
|    | nVUnxqHLr0GTN7shWid1Ow_doubl | DT_static | deg | 2.424 | -0.130 | +0.212 | +0.342 | 2.893 |
|    | 0I0XMl3M0ECO0U1N0cJvpg_faren | FS_mvmt | echo | 4.375 | -0.241 | -0.293 | -0.051 | 4.262 |
| W3 | 9xjhiFbGo06hdQIsHTS6qA_faren | FS_static | echo | 4.565 | -2.211 | -2.323 | -0.112 | 2.367 |
|    | qNvSMyUSXUyrDGpOw7s6qg_faren | FS_static | echo | 3.972 | -0.115 | -0.331 | -0.216 | 3.550 |
|    | xQEUtY2pWUi7v1X93TF2AA_faren | FS_static | echo | 3.712 | -0.086 | -0.062 | +0.023 | 3.387 |
|    | 014AzuqPZku2004NbTTmcA_neare | NS | deg | 4.356 | +0.000 | +0.000 | +0.000 | 4.355 |

## Bucket Means (Δ vs M0)

| Bucket | Metric | M_full_delay Δ | M_full_rescue Δ | Δ(R−FD) |
|--------|--------|----------------|-----------------|---------|
| DT_mvmt | deg | -0.363 | -0.224 | +0.139 |
| DT_static | deg | +0.060 | +0.047 | -0.012 |
| FS_mvmt | echo | -0.241 | -0.293 | -0.051 |
| FS_static | echo | -0.804 | -0.906 | -0.102 |
| NS | deg | +0.000 | +0.000 | +0.000 |

## Gate Check

### Composition gates (vs M0)

| Gate | Criterion | M_full_delay | M_full_rescue |
|------|-----------|--------------|---------------|
| G1 | DT bucket mean Δdeg ≥ −0.05 | FAIL | FAIL |
| G2 | No per-case: DT Δdeg < −0.20 OR FS Δecho > +0.20 | FAIL | FAIL |
| G3 | Stress cases each Δ ≥ −0.10 | FAIL | FAIL |
| G4 | FS bucket mean Δecho ≤ +0.05 | PASS | PASS |

### Rescue-specific gates (M_full_rescue vs M_full_delay)

| Gate | Criterion | Result |
|------|-----------|--------|
| R1 | All per-case Δ(R−FD) within ±0.15 | FAIL |
| R2 | 9xjhi FS_static Δecho vs M_full_delay ≥ −0.05 (Cat3 closure) | FAIL |
| R3 | 0I0XMl3M FS_mvmt Δecho vs M_full_delay ≤ +0.10 (Gate0 carryover) | PASS |

## Hard Watch Items

| Tag | Case | Criterion | M_full_delay Δ | M_full_rescue Δ | Δ(R−FD) |
|-----|------|-----------|----------------|-----------------|---------|
| W1 | wVYSGVTTakih9twI4xlDWQ_doubletal | DT_mvmt deg | +0.046 | +0.198 | +0.152 |
| W2 | xFk7igecuke0R5JMfREyDg_doubletal | DT_mvmt deg | -0.988 | -0.793 | +0.195 |
| W3 | 9xjhiFbGo06hdQIsHTS6qA_farend_si | FS_static echo | -2.211 | -2.323 | -0.112 |
| W4 | MYrVxVEMxkaE7OuyTUmI0Q_doubletal | DT_static deg | +0.096 | -0.335 | -0.431 |

## Ship Decision

- **M_full_delay** (baseline): G1–G4 FAIL (gates: ['G1', 'G2', 'G3'])
- **M_full_rescue**: G1–G4 FAIL (gates: ['G1', 'G2', 'G3'])
- **Rescue-specific R1–R3**: FAIL (gates: ['R1', 'R2'])

### Conclusion

**COMPOSITION FAIL** (M_full_rescue, gates: ['G1', 'G2', 'G3']) — stop and report root cause. See per-case table.

---

## Post-FAIL Sanity Audit (2026-05-27)

Full audit in [v3_21_poor_coarse_rescue_attribution.md](v3_21_poor_coarse_rescue_attribution.md).

> **Update**: a separate `coarse_conv` definition audit
> ([v3_21_coarse_conv_definition_audit.md](v3_21_coarse_conv_definition_audit.md))
> found that the "coarse_conv = 0% every variant" reading reported below
> was a diagnostic-script bug (wrong int16²→float² scale + wrong ratio
> bar). AEC3-correct re-audit: 9xjhi reaches AEC3 RELAXED bar 11.2% of
> frames (M_full_delay) → 12.2% (M_full_rescue); STRICT bar 0% / 0%.
> The NO-SHIP verdict below is UNCHANGED because AECMOS was measured on
> actual audio output, independent of the convergence metric.
> Codebase-wide scale audit found NO equivalent bug in production code:
> [v3_21_int16_hop_scale_codebase_audit.md](v3_21_int16_hop_scale_codebase_audit.md).

**Sanity gates** (confirms FAIL is **NOT** an implementation bug):

| Audit task | Result | Evidence |
|------------|--------|----------|
| T1 — threshold/time semantics | PASS | AEC3 5 blocks × 64 = 320 samples = 20 ms; Python `round(5×64/160) = 2` hops × 160 = 320 samples = 20 ms (match). Hangover: AEC3 25 blocks = 100 ms; Python 10 hops = 100 ms (match). No 64/160 mixing. Sole gap = irreducible decision-point granularity (5 obs vs 2 obs over same 20 ms). |
| T2 — E_refined override correctness | PASS | Single `complete_update()` call per frame (mutually exclusive override/None branches). `_deferred_update_pending` cleared every frame. `partition_idx` advances exactly once. Byte-equal flag-OFF: 25/25 PASS. |

**Per-case attribution table** (M_full_rescue − M_full_delay):

| Case | bucket | AECMOS Δ | copy_fire frac | E_refined_override frac | use_coarse Δ | ul Δ | coarse_conv | Root cause |
|------|--------|----------|----------------|-------------------------|--------------|------|-------------|------------|
| MYrVxVEM | DT_static deg | **−0.431** | 0.2%→2.8% | 2.8% | −7.4% (rises to refined) | +1.3% | 0%→0% | rescue lowers cond2 → `usable_linear` ticks up → SuppressionGain over-suppresses NE in DT |
| qNvSMyUS | FS_static echo | **−0.216** | 1.0%→9.6% | 9.6% | −3.2% | +0.0% | 0%→0% | frequent W resets perturb refined FS convergence; coarse residual on selected frames +50% |
| 9xjhi | FS_static echo | **−0.112** | 0.9%→4.3% | 4.3% | +9.7% (rises to coarse) | +0.0% | 0%→0% | cond1 +12.9pp routes more to un-converged coarse (e2_coarse on coarse-selected +29%); Cat3 gap NOT closed |
| xFk7 | DT_mvmt deg | **+0.195** | 0.3%→5.7% | 5.7% | −15.7% (rises to refined) | +0.0% | 0%→0% | cond2 −19.2pp; rescue's intended mechanism works for genuinely diverged DT_mvmt |

**Key finding**: `coarse_conv = 0%` in EVERY case × variant. AEC3 rescue copies refined W into shadow on fire hops, then drives the same-hop coarse update with E_refined (Gap C). But the shadow PBFDAF NLMS re-diverges under the current gain family — the copied W never produces a converged coarse path. The effect on AECMOS is dominated by URO routing shifts and `usable_linear` consumer side-effects, not by any actual improvement in coarse quality.

### Final Verdict

**Gap C strict AEC3 poor-coarse rescue = NO-SHIP default-OFF substrate.**

- No 800-case.
- Flag `use_aec3_poor_coarse_rescue_copy` stays in codebase at default-OFF; byte-equal safe; can be re-enabled for future shadow-convergence research.
- **Conditional gating / FS-only rescue / convergence-qualified rescue** = beyond-AEC3 optimization; belongs in v3.22, NOT v3.21 alignment scope.

### Nores Artifact Guard

9xjhi nores LF improvement attributed to Bundle A in earlier verdicts **does not** mean the internal farend-singletalk nores artifact is closed. This 12-case verdict is **NOT** a closure of that issue — closure requires a separate diagnostic (cohort + listen-test).
