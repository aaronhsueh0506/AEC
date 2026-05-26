# v3.21.13 — B+E combo (partition_summed_x2 + UseLinearFilterOutput) verdict (CLOSED)

**Date**: 2026-05-23
**Verdict**: **BE FAILS 5-CRITERION SHIP GATE. Keep E as paused substrate; pivot to v3.22 / downstream trust redesign.** Mechanism: B's cleaner filter triggers `usable_linear` latch much earlier, neutralizing E's Y-fallback safety valve. Filter-input parity + final-output parity together cannot escape the Cat C state-layer trap.

## Variants under intended HPF policy (mic ON / ref OFF)

| | A (v3.21.6 baseline) | B (partition_summed_x2) | E (UseLinearFilterOutput) | BE (combo) |
|---|---|---|---|---|
| `use_partition_summed_x2_for_h_error_gain` | False | True | False | True |
| `use_linear_filter_output_selection_for_final_output` | False | False | True | True |

A is the v3.21.6 production baseline. B + E + BE are all default-OFF flags.

## 12-case AECMOS bucket Δ vs A baseline

| Bucket | B | E | **BE** |
|---|---:|---:|---:|
| FS_static Δecho | +0.055 | -0.138 | **+0.006** |
| FS_movement Δecho | +0.152 | -0.007 | +0.064 |
| **DT_static Δecho** | **+0.215** | -0.145 | **+0.177** |
| **DT_static Δdeg** | **-0.938** | -0.335 | **-0.952** |
| DT_movement Δecho | -0.213 | -0.142 | -0.317 |
| DT_movement Δdeg | +0.363 | +0.298 | +0.419 |
| NE | 0 | 0 | 0 |

**BE essentially reproduces B's profile.** DT_static Δdeg -0.952 ≈ B's -0.938. The E mechanism's stress recovery is gone when B is combined.

## Per-case DT_static deg

| case | A | B | E | **BE** | BE-A | BE-B | **BE-E** |
|---|---:|---:|---:|---:|---:|---:|---:|
| MYrVxVEM (stress) | 2.869 | 1.523 | 2.718 | **1.575** | -1.294 | +0.052 | **-1.143** |
| XRTnTUjU (stress) | 3.929 | 2.676 | 3.326 | **2.705** | -1.224 | +0.029 | **-0.621** |
| jtYTdZm3 (normal) | 2.347 | 2.670 | 2.323 | 2.660 | +0.313 | -0.010 | +0.337 |
| nVUnxqHLr (stress, worst) | 3.223 | 1.748 | 2.663 | **1.621** | -1.602 | -0.127 | **-1.042** |

BE = essentially B on stress (BE-B per stress case: +0.05, +0.03, -0.13).
BE is up to **-1.14 deg WORSE than E alone** on stress cases (MYrVxVEM, nVUnxqHLr).

## Per-case FS_static echo

| case | A | B | E | **BE** | BE-A |
|---|---:|---:|---:|---:|---:|
| 9xjhi (canonical nores target) | 2.194 | 2.100 | 1.944 | 2.109 | -0.085 |
| qNvSMyU | 3.673 | 3.568 | 3.539 | 3.486 | -0.187 |
| xQEUtY2 | 3.346 | 3.710 | 3.315 | 3.636 | +0.290 |

FS_static bucket Δecho +0.006 (mean): BE preserves B's FS echo cancellation level.

## nores LF/MF/HF mean band ratios

| Variant | nores_lf | nores_mf | nores_hf |
|---|---:|---:|---:|
| A | 0.5372 | 0.4549 | 0.9340 |
| B | **0.4008** | 0.3723 | **0.6293** |
| E | 0.5372 | 0.4549 | 0.9340 (identical to A) |
| **BE** | **0.4008** | 0.3723 | **0.6293** (identical to B) |

E does NOT affect nores (post-filter change; `*_ours_nores.wav` is pre-RES linear residual). BE inherits B's 25% LF artifact reduction vs A.

## Use-capture rate (KEY MECHANISTIC EVIDENCE)

| case | E_cap% | **BE_cap%** | ratio |
|---|---:|---:|---:|
| **XRTnTUjU (stress)** | **59.4%** | **7.4%** | **8.0× less** |
| nVUnxqHLr | 4.1% | 4.0% | tied |
| MYrVxVEM | 3.2% | 3.2% | tied |
| jtYTdZm3 (normal) | 2.2% | 2.3% | tied |
| xFk7igec DT_mvmt | 3.9% | 3.9% | tied |
| ZJYUt0O0A DT_mvmt | 5.9% | 5.9% | tied |
| wVYSGV DT_mvmt | 3.7% | 3.7% | tied |
| 9xjhi FS | 5.4% | 7.6% | 1.4× more |
| qNvSMyU FS | 17.5% | 13.6% | 0.8× less |
| xQEUtY2 FS | 12.4% | 7.1% | 0.6× less |
| 0I0XMl3M FS_mvmt | 10.3% | 8.4% | 0.8× less |
| NE | 100.0% | 100.0% | tied |

**THE MECHANISM IS CONFIRMED**:
- On XRTnTUjU (the only stress case where E's Y-fallback genuinely activates), partition_summed_x2 collapses use_capture from **59.4 % → 7.4 %**
- B makes the linear filter clean enough that `usable_linear` latches True early on XRTnTUjU → output uses E (residual) for 93% of frames → falls back into the Cat C trap
- On other stress cases (nVUnxqHLr, MYrVxVEM, xFk7igec), E's use_capture was already low (~3-4 %) — those cases don't benefit from E because they latch True from the start. BE = B on those cases.

E's safety valve only works on the SINGLE stress case (XRTnTUjU) where the latch genuinely struggles to engage. The moment B speeds up convergence, the latch engages → E is bypassed.

## 5-criterion ship gate evaluation

| # | Criterion | Result | Verdict |
|---:|---|---|---|
| 1 | Stress cases recover vs B | BE-B per stress: +0.05 / +0.03 / -0.13 — **no material recovery** | **FAIL** |
| 2 | No FS_static echo regression worse than -0.05 bucket mean | FS_static Δecho +0.006 | PASS |
| 3 | No DT_static deg regression worse than -0.1 bucket mean | DT_static Δdeg **-0.952** (9.5× bar) | **HARD FAIL** |
| 4 | No normal DT per-case catastrophic regression | jtYTdZm3 +0.313; DT_mvmt all improve vs A | PASS |
| 5 | nores LF artifact improves vs A | LF 0.40 vs A 0.54 (25% reduction) | PASS |

**Overall**: 3/5 pass, 2/5 critical fail (1 + 3). **BE FAILS the ship gate.**

## Structural conclusion

This combo cycle confirms (with three independent evidence streams) that v3.21.7 Cat C is a **state-layer latch problem**, not solvable by AEC3-parity changes at the filter or final-output stages:

| Evidence | What it shows |
|---|---|
| v3.21.12 (D = B + raw E²) WORSE than B | Cleaner filter input → cleaner residual → latch fires harder → MORE Cat C damage |
| v3.21.13 E alone has -0.335 DT_static residual | E helps when latch is OFF (False regime); but latch True regime still has same damage |
| **v3.21.13 BE = B** | B's cleaner residual collapses E's False regime (XRTnTUjU 59% → 7% use_capture); E's safety valve loses its window |

The proximate cause is the `usable_linear` 4-gate AND with `external_delay OR convergence_seen` gate-3 ([project_usable_linear_gate3_latch_bug]). Once convergence_seen latches True, the system trusts the linear filter for the rest of the utterance — even on stress utterances where convergence was brief / unstable / pre-DT.

Three potential fixes for state-layer Cat C (all OUT OF v3.21.x AEC3 PARITY SCOPE):

1. **`convergence_seen` decay/release** — AEC3 has the same latch but different downstream RES; ours over-relies on the latch
2. **AEC3 RES semantic delta audit** — find where AEC3's SuppressionGain behaves differently when latch True but recent NE
3. **PBFDKF-specific NE-presence override** — when DT speech detected, force `usable_linear = False` per-frame regardless of latch

Per user's locked v3.21.x rule, none of these are v3.21.x parity. They are either v3.22 PBFDKF-specific (option 3) or audit of an AEC3 semantic mismatch in RES (option 2 — could be v3.21.x if a real AEC3 mismatch is found).

## Final disposition

| Variant | Disposition | Reason |
|---|---|---|
| A | Production default — keep | v3.21.6 baseline |
| B | Status quo — paused Cat C BLOCKED-STRESS (v3.21.7 unchanged) | nores LF helpful; Cat C blocker unchanged by BE |
| E (v3.21.13) | **PAUSED SUBSTRATE — default OFF** | First material Cat C recovery on XRTnTUjU; mixed Pareto vs A; defeated by B-combo |
| **BE** | **CLOSE no-ship** | Fails ship gate criteria 1 + 3; mechanism evidence confirms latch is the structural blocker |

E remains a valid AEC3-parity port (code retained, byte-equal default OFF). It can ship if the latch arc closes separately.

## Recommended next steps for user (out of v3.21.x scope; needs explicit authorisation)

1. **v3.22 latch redesign** — `convergence_seen` decay / per-frame override / NE-presence force-False. Explicitly NOT AEC3 parity (would deviate from AEC3 baseline). Requires user authorisation to open v3.22.

2. **Downstream RES AEC3-parity audit** — read AEC3 `suppression_gain.cc` / `residual_echo_estimator.cc` carefully for semantic deltas vs our implementation when `usable_linear = True`. If a real AEC3 mismatch is found, it could be v3.21.x. Independent investigation.

3. **Stay at v3.21.6 + retain v3.21.7 (B) and v3.21.13 (E) as default-OFF substrate**. Mark v3.21.x AEC3-parity arc as exhausted. v3.21.x final disposition: 7 attempts, 0 ship candidates. The structural problem is downstream of all AEC3 parity points we've audited.

## Forbidden actions

- Do NOT ship BE as production default
- Do NOT delete BE code (both flags remain default-OFF substrate)
- Do NOT run 800-case on BE (12-case AECMOS already disqualifies; 800-case is wasted compute)
- Do NOT pivot to v3.22 latch redesign without explicit user authorisation
- Do NOT add gate-3 counter / FA-AND / shadow-converged precondition in v3.21.x scope
- Do NOT delete v3.21.13 E flag — it is the first material Cat C recovery candidate and remains valid AEC3 parity
