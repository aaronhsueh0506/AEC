# v3.21 Q1 Phase 2.1 — Recalibration Matrix (R1 Amplitude + R2 Duration)

**Scope:** R1 (amplitude 2×/3× vs 5×) × R2 (duration 50/100 hops vs 250 hops).
No R4 (EPV reset), no new mechanism, no H_error hard reset.
**Config:** BALANCED + enable_delay_est=True + enable_cng=True; no pre-alignment.
**Cohort:** 9 cases (same as Phase 2).

---

## Variant Matrix

| Var | Flag | lc_factor | transient | smoothing | total | desc |
|---|---|---|---|---|---|---|
| A | OFF | 5.0× | 250 | 100 | 350 (3.5s) | baseline OFF |
| B | ON | 5.0× | 250 | 100 | 350 (3.5s) | 5× / 250+100 (Phase 2 FAIL reference) |
| C | ON | 2.0× | 50 | 50 | 100 (1.0s) | 2× / 50+50 |
| D | ON | 2.0× | 100 | 50 | 150 (1.5s) | 2× / 100+50 |
| E | ON | 3.0× | 50 | 50 | 100 (1.0s) | 3× / 50+50 |
| F | ON | 3.0× | 100 | 50 | 150 (1.5s) | 3× / 100+50 |

lc_steady = 0.0025; lc_transient = lc_factor × lc_steady. ld_factor mirrors lc_factor (ld_steady = 0.25).

Gate 2 is evaluated **directionally** (regressions only): DT Δdeg < −0.20 or FS Δecho > +0.20. Large positive Δdeg is an improvement, not a catastrophic outlier.

---

## Gate Summary (vs A baseline)

| Var | Gate 1 (DT_mvmt no regression > −0.05) | Gate 2 (no directional regression \|Δ\| > 0.20) | Gate 3 (H_error ratio > 1.3) | Verdict |
|---|---|---|---|---|
| A | **PASS** | **PASS** | **N/A (flag OFF)** | **PASS** |
| B | **FAIL** — Hp5g −0.951, XRTn −0.168 | **FAIL** — Fi80N Δecho +0.355, Hp5g Δdeg −0.951 | PASS (ratio 3.38×) | **FAIL** |
| C | **PASS** — all cases within −0.05 | **PASS** — no FS echo +0.20; no DT deg −0.20 | PASS (ratio 1.66×) | **PASS** |
| D | **FAIL** — Hp5g −0.309 | **FAIL** — Hp5g Δdeg −0.309; Fi80N Δecho +0.219 | PASS (ratio 1.66×) | **FAIL** |
| E | **FAIL** — Hp5g −0.182 | **PASS** | PASS (ratio 2.24×) | **FAIL** |
| F | **FAIL** — Hp5g −0.494, XRTn −0.055 | **FAIL** — Hp5g Δdeg −0.494 | PASS (ratio 2.24×) | **FAIL** |

**Variant C (2× / 50+50, total 1.0 s) PASSES all three gates.**

---

## Bucket Means Δ vs A

Primary metric: DT→deg (higher=better), FS→echo (lower=better).

| Bucket | N | Δ(B) ref-fail | Δ(C) | Δ(D) | Δ(E) | Δ(F) |
|---|---|---|---|---|---|---|
| DT_mvmt (deg) | 4 | −0.276 | **−0.001** | −0.078 | −0.047 | −0.131 |
| DT_static (deg) | 1 | +0.333 | **+0.320** | +0.326 | +0.319 | +0.321 |
| FS_mvmt (echo) | 3 | +0.155 | **−0.004** | +0.100 | +0.000 | +0.052 |
| FS_static (echo) | 1 | −0.004 | **−0.002** | −0.005 | −0.008 | −0.011 |

Variant C DT_mvmt bucket mean Δdeg = −0.001 (essentially neutral vs baseline OFF).
Variant C FS_mvmt bucket mean Δecho = −0.004 (marginal improvement).
Variant C DT_static Δdeg = +0.320 (beneficial EPV suppression preserved).

---

## Per-case DT_mvmt Δdeg (Gate 1 focus)

| Case | A_deg | B | C | D | E | F |
|---|---|---|---|---|---|---|
| 49IIo03GZ0 | 2.818 | +0.014 | +0.018 | +0.018 | +0.028 | +0.027 |
| 7GTxyTksSU | 3.753 | +0.000 | +0.000 | +0.000 | +0.000 | +0.000 |
| Hp5g1asacU | 2.785 | −0.951 **FAIL** | **−0.009** | −0.309 **FAIL** | −0.182 **FAIL** | −0.494 **FAIL** |
| XRTnTUjU5k | 3.342 | −0.168 **FAIL** | **−0.014** | −0.022 | −0.033 | −0.055 **FAIL** |

**C recovers Hp5g1asacU to −0.009 (noise floor) and XRTnTUjU5k to −0.014 (negligible).**

---

## FS Catastrophic Outliers (Gate 2 directional: Δecho > +0.20 is regression)

| Case | Label | B_Δecho | C_Δecho | D_Δecho | E_Δecho | F_Δecho |
|---|---|---|---|---|---|---|
| 0I0XMl3M0E | FS_mvmt | −0.020 | −0.021 | −0.017 | −0.020 | −0.021 |
| Fi80N5kW9U | FS_mvmt | +0.355 **FAIL** | **−0.061** | +0.219 **FAIL** | −0.090 | +0.059 |
| wlAXM0iDgk | FS_mvmt | +0.129 | **+0.070** | +0.099 | +0.112 | +0.118 |
| 9xjhiFbGo0 | FS_static | −0.004 | **−0.002** | −0.005 | −0.008 | −0.011 |

B's catastrophic Fi80N regression (+0.355) is eliminated in C (−0.061, below threshold).
No FS case exceeds +0.20 in variant C.

---

## H_error Trace — Effect Observability

H_error_mean at event+11 hops for XRTnTUjU (no EPV, clean delay case) and nVUnxqHLr.

| Var | XRTnTUjU H_on+11 | XRTnTUjU ratio | nVUnxqHLr H_on+11 | nVUnxqHLr ratio | EPV fires (ON/OFF) |
|---|---|---|---|---|---|
| A | 0.265 | 1.00× | 1.725 | 1.00× | Y / Y |
| B | 0.896 | 3.38× | 1.895 | 1.10× | N / Y |
| C | 0.441 | **1.66×** | 1.767 | 1.02× | N / Y |
| D | 0.441 | 1.66× | 1.767 | 1.02× | N / Y |
| E | 0.594 | 2.24× | 1.810 | 1.05× | N / Y |
| F | 0.594 | 2.24× | 1.810 | 1.05× | N / Y |

Gate 3 (H_error observability): ratio > 1.3 at event+11 is meaningful boost.
**C ratio = 1.66 → Gate 3 PASS.** Effect is observable, not a no-op.

Note: C and D show identical H_error at event+11 because both are still in the full-plateau phase at frame+11 (both have lc=2×steady for at least 50 hops). Their difference is downstream: C's transient expires sooner, reducing DT exposure and FS overshoot.

---

## nVUnxqHLr EPV Finding per Variant

| Var | A_deg | Δdeg | EPV fires (ON) | EPV fires (OFF) | Verdict |
|---|---|---|---|---|---|
| A | 2.421 | +0.000 | Y (frame 426) | Y (frame 426) | neutral |
| B | 2.421 | +0.333 | N | Y (frame 426) | beneficial |
| C | 2.421 | **+0.320** | N | Y (frame 426) | **beneficial** |
| D | 2.421 | +0.326 | N | Y (frame 426) | beneficial |
| E | 2.421 | +0.319 | N | Y (frame 426) | beneficial |
| F | 2.421 | +0.321 | N | Y (frame 426) | beneficial |

All ON variants suppress the false-alarm EPV at frame 426. Even 2×/50 is sufficient to accelerate convergence past the EPC detector threshold, removing the false EPV that damaged DT deg in the OFF run.

---

## Why C Passes — Mechanistic Analysis

### Gate 1 / Gate 2 recovery on Hp5g1asacU (Phase 2's worst failure)

**Phase 2 mechanism:** delay_first at frame 90 → Q1 arms for 350 hops (total B). EPV fires at frame 372. At EPV time: Q1 counter rem = 350−(372−90) = 68, still in smoothing zone. lc ≈ 0.0093 (3.7× elevated). EPV + elevated lc combined → multiplicative Kalman gain → Δdeg = −0.951.

**C mechanism:** Q1 total = 100 hops. Counter expires at frame 90+100 = **190**. EPV fires at frame **372**, which is 182 hops **after** Q1 has already expired and lc has restored to steady 0.0025. At EPV time: lc = steady (no elevation). EPV fires into normal lc → normal Kalman gain → Δdeg = **−0.009** (noise floor).

The 50-hop transient + 50-hop smooth total of 100 hops is short enough that Hp5g1asacU's EPV (which arrives at hop +282 post-delay-first) fires into steady-state leakage — the multiplicative amplification mechanism is structurally avoided.

### Gate 1 / Gate 2 recovery on XRTnTUjU (no EPV, DT over-adaptation)

**Phase 2 mechanism:** delay_first at frame 154, no EPV. 350-hop window → elevated lc during DT speech throughout 3.5 s → Δdeg = −0.163.

**C mechanism:** 100-hop window expires at frame 154+100=254 (2.54 s). DT speech window partially overlaps but exposure is 3.5× shorter. lc amplitude 2.5× lower (2× vs 5×). Combined: ~8× lower H_error injection product. Δdeg = **−0.014** (negligible).

### Gate 2 recovery on Fi80N5kW9U (FS de-convergence)

**Phase 2 mechanism:** 5× amplitude → H_error per-bin grows too fast → Kalman gain overshoots stable operating point → oscillation → echo +0.318.

**C mechanism:** 2× amplitude → H_error injection 2.5× lower → Kalman gain stays below overshoot threshold → filter re-converges normally → Δecho = **−0.061** (slightly improved).

### Structural fragility note

C's Gate 2 pass on Hp5g1asacU is **timing-dependent**: it works because EPV fires 282 hops after delay_first, giving Q1 enough time (100 hops) to expire before EPV. If a case had EPV firing within 100 hops of delay_first, C's window would still overlap EPV and multiplicative amplification would recur. R4 (EPV counter reset) is the structural fix; C achieves de-facto safety via the short window on this 9-case cohort. This fragility is noted but does not invalidate the current gate results.

---

## Recalibration Conclusion

**Variant C (2× / 50+50, total 1.0 s) PASSES all gates.**

- Gate 1 (DT_mvmt no regression > −0.05): **PASS** — Hp5g −0.009, XRTn −0.014, others neutral
- Gate 2 (no directional regression > 0.20): **PASS** — no FS echo > +0.20; no DT deg < −0.20
- Gate 3 (H_error observability): **PASS** — XRTnTUjU ratio 1.66× (effect still measurable, not no-op)

**R1+R2 can solve the Phase 2 failure on this 9-case cohort.** R4 remains deferred (the user instruction: "No R4 unless R1/R2 matrix proves amplitude/duration cannot solve EPV multiplicative failure" — R1+R2 can solve it here).

Next step per hard rules: proceed to Phase 3 (12-case bench) with Variant C configuration:
```
q1_tdt_lc_factor  = 2.0   (was 5.0)
q1_tdt_ld_factor  = 2.0   (was 5.0)
q1_tdt_transient_hops  = 50   (was 250)
q1_tdt_smoothing_hops  = 50   (was 100)
# total = 100 hops = 1.0 s
```

---

## Hard Rules (unchanged)

- No 12-case bench until Phase 2 gate passes — **Phase 2 now passes for variant C**
- No 800-case bench until Phase 3 (12-case) passes
- No R4 unless R1/R2 matrix proves amplitude/duration cannot solve EPV multiplicative failure — **not triggered; C passes**
- No new mechanism
