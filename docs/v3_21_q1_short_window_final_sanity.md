# v3.21 Q1 Short-Window Final Sanity — 12-case AECMOS

**Date:** 2026-05-25
**Scope:** Final narrow sanity on two shorter Q1 windows; no amplitude increase, no SR/EPV guard, no W-ratio guard, no H_error spike.
**Cohort:** Same 12 cases as Phase 3 (`wav/v3_21_8_cohort/`).
**Config:** BALANCED + `enable_delay_est=True` + `enable_cng=True`.

## Variants

| Label | Config | Total hops | Pre-run expectation |
|---|---|---|---|
| A  | baseline OFF | — | reference |
| SA | 2× / 25+25 | 50 hops = 0.5 s | Q1 expires at hop 50; SR fires at hop 79 → SR fires 29 hops after expiry → expected: no amplification |
| SB | 2× / 40+40 | 80 hops = 0.8 s | Q1 expires at hop 80; SR fires at boundary (q1_rem ≈ 0-1) → expected: borderline |

**Phase 3 root cause recap:** xFk7igecuk delay_first at frame 104; SR fires at frame 183 (hop 79 post-event). Variant C (100 hops) had q1_rem=20 at SR → 79 hops of elevated lc built 7.9% W excess → SR amplified to 17% → persisted 27 s to DT speech → Δdeg=−0.110.

**Pre-run expectation for SA:** Q1 expires at hop 50 (frame 154), SR fires 29 hops later at frame 183. Expectation was that W_SA/W_A ≈ 1.000 at SR time because no elevated lc during hops 51–79. This expectation is **falsified by the trace** (see §SR Trace below).

---

## Gate Summary

Same three gates as Phase 3:
- G1: No DT_mvmt per-case Δdeg < −0.05
- G2: No directional catastrophic (DT Δdeg < −0.20 or FS Δecho > +0.20)
- G3: FS bucket Pareto: FS mean Δecho ≤ +0.05

| Variant | G1 | G2 | G3 | Overall |
|---|---|---|---|---|
| SA | FAIL | PASS | PASS | **FAIL** |
| SB | FAIL | PASS | PASS | **FAIL** |

SA Gate 1 failures: xFk7igecuk Δdeg=-0.097
SA Gate 3: FS_mvmt Δecho=-0.019 OK  FS_static Δecho=-0.014 OK

SB Gate 1 failures: xFk7igecuk Δdeg=-0.117
SB Gate 3: FS_mvmt Δecho=-0.022 OK  FS_static Δecho=-0.029 OK

---

## Bucket Means Δ vs A

Primary metric: DT→deg (higher=better), FS→echo (lower=better).

| Bucket | N | A | SA | ΔSA | SB | ΔSB |
|---|---|---|---|---|---|---|
| DT_mvmt | 3 | deg=3.047 | 3.062 | +0.015 | 3.064 | +0.016 |
| DT_static | 4 | deg=2.372 | 2.482 | +0.110 | 2.471 | +0.099 |
| FS_mvmt | 1 | echo=4.411 | 4.392 | -0.019 | 4.389 | -0.022 |
| FS_static | 3 | echo=3.924 | 3.910 | -0.014 | 3.895 | -0.029 |
| NS | 1 | deg=4.359 | 4.359 | +0.000 | 4.359 | +0.000 |

---

## Per-case DT_mvmt Δdeg (G1 gate)

| Case | A_deg | SA_deg | ΔSA | SA G1 | SB_deg | ΔSB | SB G1 |
|---|---|---|---|---|---|---|---|
| ZJYUt0O0AE | 3.058 | 3.166 | +0.108 | PASS | 3.165 | +0.107 | PASS |
| wVYSGVTTak | 3.205 | 3.239 | +0.034 | PASS | 3.263 | +0.059 | PASS |
| xFk7igecuk | 2.879 | 2.782 | -0.097 | **FAIL** | 2.763 | -0.117 | **FAIL** |

## Per-case DT_static Δdeg

| Case | A_deg | SA_deg | ΔSA | SB_deg | ΔSB | Note |
|---|---|---|---|---|---|---|
| MYrVxVEMxk | 1.730 | 1.795 | +0.064 | 1.747 | +0.016 |  |
| XRTnTUjU5k | 2.750 | 2.780 | +0.030 | 2.780 | +0.030 | Phase 2.1 sensitive |
| jtYTdZm3lU | 2.586 | 2.615 | +0.029 | 2.614 | +0.028 |  |
| nVUnxqHLr0 | 2.421 | 2.737 | +0.316 | 2.744 | +0.323 | EPV suppression sensitive |

## FS Cases — Echo Δ (G2 directional)

| Case | Label | A_echo | SA_echo | ΔSA | SB_echo | ΔSB | G2 (SA/SB) |
|---|---|---|---|---|---|---|---|
| 0I0XMl3M0E | FS_mvmt | 4.411 | 4.392 | -0.019 | 4.389 | -0.022 | PASS/PASS |
| 9xjhiFbGo0 | FS_static | 4.569 | 4.570 | +0.001 | 4.570 | +0.001 | PASS/PASS |
| qNvSMyUSXU | FS_static | 3.594 | 3.594 | +0.001 | 3.594 | +0.001 | PASS/PASS |
| xQEUtY2pWU | FS_static | 3.610 | 3.566 | -0.044 | 3.521 | -0.089 | PASS/PASS |

---

## SR and EPV Trace

Trace on xFk7igecuk (Phase 3 failing case) and nVUnxqHLr (EPV suppression case).

| Case | delay_first | SR1 frame | q1_rem_SA@SR1 | q1_rem_SB@SR1 | EPV_A | EPV_SA | EPV_SB |
|---|---|---|---|---|---|---|---|
| xFk7igecuk | 104 | 183 | 0 | 1 | N | N | N |
| nVUnxqHLr0 | 119 | None | None | None | Y | N | N |

`q1_rem_SA/SB@SR1`: Q1 counter remaining in SA/SB at the first SR frame.
0 = Q1 already expired when SR fires (safe). Positive = SR fires within Q1 window (potential excess seed).

### W_norm at SR+10 (amplification check on xFk7igecuk)

| Variant | W_norm at SR+10 | Ratio vs A |
|---|---|---|
| A  | 112.25 | 1.000 |
| SA | 127.61 | **1.137** |
| SB | 130.54 | **1.163** |
| C (Phase 3, ref) | — | 1.169 |

**Key finding:** SA has q1_rem=0 at SR1 (Q1 already expired when SR fires) yet W_SA/W_A = 1.137 — a 13.7% excess, compared to Variant C's 16.9%. The pre-run expectation of "≈ 1.000 because Q1 expired before SR" is **falsified.**

**Mechanism (structural):** W excess built during the 50 Q1 hops (frames 104–153) propagates multiplicatively through subsequent filter updates. At steady lc (frames 154–182), both W_SA and W_A grow at similar rates but W_SA starts from a higher level — the ratio ≈ 1.065-1.070 is preserved, not decayed. When SR fires at frame 183, the pre-existing 6–7% W_SA excess is amplified to 13.7% by the SR post-hangover Kalman explosion (same mechanism as Variant C, smaller seed, smaller but still damaging amplification).

**Implication:** There is no sustained-lc Q1 window length that avoids SR amplification. Any Q1 window long enough to build W excess will persist that excess to SR time, regardless of whether Q1 is still active when SR fires. The excess is baked into the filter state and does not decay at steady lc. Only a Q1 window of effectively 0 hops (= no Q1) avoids the amplification — but that provides zero benefit.

---

## Conclusion and Disposition

**Both SA and SB FAIL.** No safe parameter surface found for the sustained-lc leakage-window approach.

**Structural finding (this sanity):** SA (50-hop window, expires 29 hops before SR) has W_SA/W_A = 1.137 at SR+10 — not ≈ 1.000 as expected. W excess built during Q1 persists multiplicatively after Q1 expires; it does not decay at steady lc. SR amplifies whatever excess is present at its fire time, regardless of whether Q1 is still active. This closes the parameter search: there is no window length in the sustained-lc approach that avoids SR amplification on cases where SR fires within the post-delay era.

**Complete closure record:**

| Attempt | Config | Status | Mechanism |
|---|---|---|---|
| Phase 2 | 5× / 250+100 | FAIL G1+G2 | mis-calibrated amplitude; multiplicative EPV |
| Phase 2.1 C | 2× / 50+50 | PASS 9-case only | timing coincidence (EPV fires after Q1; SR not in 9-case DT_mvmt) |
| Phase 3 C | 2× / 50+50 | FAIL G1 | SR at hop 79 within Q1; 7.9% W excess → 17% at SR+10 → −0.110 DT |
| Phase 3.1 guard | C + SR/EPV terminate | FAIL G1 | 79-hop pre-SR excess is seed; 7-hop hangover insufficient to close gap |
| SA (this) | 2× / 25+25 | FAIL G1 | Q1 expires before SR but W excess (built over 50 hops) persists; amplified to 13.7% |
| SB (this) | 2× / 40+40 | FAIL G1 | borderline (q1_rem=1 at SR); excess 16.3%; near-identical to C |

**Structural conclusion:** Sustained-lc elevation (the leakage-window approach) builds filter-state W excess that propagates multiplicatively. No window length avoids this on cases where SR fires in the post-delay era. Fix paths:
- **Full AEC3 delay-change chain** (H_error=10000 reset + SetConfig(initial) + AecState initial_state reset on delay_first; AEC3 does **NOT** zero W — `ZeroFilter(13,13)` is a no-op at default partition count, `SetSizePartitions(12,true)` → `ZeroFilter(13,12)` is also no-op): this is NOT a parameter variant of the closed candidate — the chain does not include the leakage-window lc_gain elevation, so no W_C/W_A divergence builds and SR has no excess to amplify. This is proper v3.21.x AEC3 parity work — **OPEN**. See `docs/v3_21_alignment_roadmap_refresh.md`.
- W norm-ratio guard: terminate Q1 early if W_C/W_A exceeds threshold; novel mechanism beyond AEC3 scope — requires new authorization.

**`use_q1_true_delay_transient_leakage` (leakage-window-only approach, no W/H_error reset) is CLOSED as no-ship default-OFF substrate.** Parameter space exhausted; no safe window length exists under the sustained-lc model.

**AEC3 delay-change parity arc OPEN.** The leakage-window candidate omitted H_error=10000 spike, config switch, and AecState reset — the core of AEC3's mechanism. (AEC3 does not zero W on delay change; ZeroFilter(13,13) is a no-op at default partition count.) Its failure does not close the AEC3 delay-change chain. Production `main` unchanged.

---

## Phase History

| Phase | Variant | Result | Root cause / note |
|---|---|---|---|
| Phase 2 | 5× / 250+100 | FAIL | mis-calibrated; Gate 1 DT_mvmt regression |
| Phase 2.1 C | 2× / 50+50 | PASS 9-case | timing-dependent (EPV fires 182 hops after Q1) |
| Phase 3 C | 2× / 50+50 | FAIL 12-case | SR at hop 79 within Q1; W excess 7.9%→17%→13% at DT |
| Phase 3.1 guard | C + SR/EPV terminate | FAIL | pre-SR excess (79 hops) is seed; 7-hop hangover insufficient |
| SA (this) | 2× / 25+25 | FAIL G1 (Δdeg=−0.097) | Q1 expires before SR but W excess persists; W_SA/W_A=1.137 at SR+10 |
| SB (this) | 2× / 40+40 | FAIL G1 (Δdeg=−0.117) | q1_rem=1 at SR; W_SB/W_A=1.163; near-identical to C |
