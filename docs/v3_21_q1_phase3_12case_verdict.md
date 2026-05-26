# v3.21 Q1 Phase 3 — 12-case AECMOS Verdict

**Candidate:** Variant C — `use_q1_true_delay_transient_leakage=True`, `lc_factor=2.0`, `transient=50`, `smoothing=50` (total 100 hops = 1.0 s).
**Trigger source:** `delay_first` / `delay_shift` only. No EPV reset / no R4 / no new mechanism.
**Config:** BALANCED + `enable_delay_est=True` + `enable_cng=True`; no pre-alignment.
**Cohort:** 12 cases (`wav/v3_21_8_cohort/`).

---

## Gate Results

**Overall Phase 3 verdict: FAIL**

| Gate | Criterion | Result |
|---|---|---|
| Gate 1 | DT_mvmt: no per-case Δdeg < −0.05 | **FAIL** — xFk7igecuk Δdeg=−0.110 |
| Gate 2 | Directional: no DT Δdeg < −0.20 or FS Δecho > +0.20 | **PASS** |
| Gate 3 | FS Pareto: FS bucket mean Δecho ≤ +0.05 | **PASS** |
| Byte-equal OFF | flag=False produces byte-equal output vs standard config | **PASS** |

Phase 3 hard rule consequence: Q1 candidate (`use_q1_true_delay_transient_leakage`, Variant C) is **closed as no-ship substrate**. Do not tune further without new evidence.

---

## Bucket Means Δ vs A (baseline OFF)

Primary metric: DT→deg (higher=better), FS→echo (lower=better).

| Bucket | N | A (baseline) | C (candidate) | Δ(C) |
|---|---|---|---|---|
| DT_mvmt | 3 | deg=3.047 | deg=3.052 | +0.005 |
| DT_static | 4 | deg=2.378 | deg=2.475 | **+0.096** |
| FS_mvmt | 1 | echo=4.462 | echo=4.445 | −0.017 |
| FS_static | 3 | echo=3.371 | echo=3.373 | +0.002 |
| NS | 1 | deg=4.358 | deg=4.358 | +0.000 |

DT_mvmt bucket mean (+0.005) is misleadingly neutral: the two passing cases (+0.088, +0.038) mask the failing case (−0.110). Gate evaluation is per-case, not bucket mean.

---

## Per-case DT_mvmt Δdeg (Gate 1: no regression > −0.05)

| Case | A_deg | C_deg | Δdeg | Gate 1 |
|---|---|---|---|---|
| ZJYUt0O0AE | 3.060 | 3.147 | +0.088 | PASS |
| wVYSGVTTak | 3.203 | 3.241 | +0.038 | PASS |
| xFk7igecuk | 2.878 | 2.769 | −0.110 | **FAIL** |

---

## Per-case DT_static Δdeg (sensitive cases marked)

| Case | A_deg | C_deg | Δdeg | Note |
|---|---|---|---|---|
| MYrVxVEMxk | 1.731 | 1.740 | +0.009 | |
| XRTnTUjU5k | 2.750 | 2.781 | +0.031 | Phase 2.1 sensitive (DT_mvmt version in Phase 2.1 cohort) |
| jtYTdZm3lU | 2.587 | 2.611 | +0.024 | |
| nVUnxqHLr0 | 2.446 | 2.767 | **+0.321** | Phase 2.1 EPV finding case — benefit preserved |

---

## FS Cases — Echo Δ (Gate 2 directional: Δecho > +0.20 = regression)

| Case | Label | A_echo | C_echo | Δecho | Gate 2 |
|---|---|---|---|---|---|
| 0I0XMl3M0E | FS_mvmt | 4.462 | 4.445 | −0.017 | PASS |
| 9xjhiFbGo0 | FS_static | 2.892 | 3.004 | +0.112 | PASS |
| qNvSMyUSXU | FS_static | 3.594 | 3.594 | +0.001 | PASS |
| xQEUtY2pWU | FS_static | 3.626 | 3.521 | −0.105 | PASS |

No FS catastrophic outlier. 9xjhiFbGo0 +0.112 is below the +0.20 gate.

---

## H_error Trace and Event-Source Verification

**Gate: lc elevation must be triggered ONLY by `delay_first` / `delay_shift` events. EPV and shadow_rise must NOT trigger Q1.**

| Case | Event src | Event frame | H_on+11 | H_off+11 | Ratio | EPV_on | EPV_off | SR_on | q1_rem@EPV |
|---|---|---|---|---|---|---|---|---|---|
| XRTnTUjU5k | delay_first | 109 | 0.209 | 0.182 | 1.15× | N | N | N | N/A |
| nVUnxqHLr0 | delay_first | 119 | 1.767 | 1.725 | 1.02× | N | Y | Y | N/A |

**Event-source verification PASS:** Both traced cases trigger Q1 only on `delay_first`. No EPV or shadow_rise triggers observed.

**nVUnxqHLr0 EPV suppression confirmed:** EPV_on=N (EPV suppressed), EPV_off=Y (EPV fires in baseline) — benefit Δdeg=+0.321 preserved. Shadow_rise fires in ON variant but does not trigger Q1 (correct — Q1 gated to delay events only).

**XRTnTUjU5k note:** This is the DT_static version (no movement) in the 12-case cohort. Ratio=1.15× is below the Phase 2.1 Gate 3 threshold (1.3×), but no EPV interaction and AECMOS Δdeg=+0.031 (positive). The DT_static version has fixed echo path — Q1 fires once at startup and expires before any DT speech windows.

---

## Sensitive Cases from Phase 2.1

Cases recovered in Phase 2.1 (9-case cohort, `wav/aec_challenge_blind/`):

| Case | Phase 2.1 Δ | Phase 3 status |
|---|---|---|
| Hp5g1asacU (DT_mvmt) | Δdeg=−0.009 (recovered from −0.951) | NOT in 12-case cohort |
| Fi80N5kW9U (FS_mvmt) | Δecho=−0.061 (recovered from +0.318) | NOT in 12-case cohort |
| XRTnTUjU5k (DT_static in 12-case) | Δdeg=−0.014 (DT_mvmt version in Phase 2.1) | Δdeg=+0.031 (DT_static) — PASS |
| nVUnxqHLr0 (DT_static, EPV) | Δdeg=+0.320 (EPV suppression) | Δdeg=+0.321 — preserved |

---

## Failure Diagnosis: xFk7igecuke0R5JMfREyDg_doubletalk_with_movement

### Observed: Δdeg=−0.110 (Gate 1 FAIL; below catastrophic −0.20 Gate 2 threshold)

xFk7igecuk is a DT_mvmt case not in the Phase 2.1 9-case cohort. The Δdeg=−0.110 regression stems from W_C bias accumulated during Q1+SR interaction — there is no DT overlap with the Q1 window. Per-frame trace: dt_from_energy=0 throughout frames 104–203 (the entire Q1 window); DT speech occurs at t=28–36 s, 27 seconds after Q1 expires.

**Root cause (per-frame trace audit — see `docs/v3_21_q1_phase3_xfk7igec_failure_audit.md`):**

Per-frame trace shows dt_from_energy=0 throughout the entire Q1 window (frames 104–203, t=1.04–2.04 s). DT speech occurs at t=28–36 s, 27 seconds after Q1 expires. The mechanism is a Q1+SR interaction, not DT-during-Q1:

1. **delay_first at frame 104 (t=1.04 s)**: W resets, Q1 arms. Q1 elevated lc (2×) causes W_C to grow 7.9% faster than W_A over 79 hops (frames 104–182). No DT during this period.
2. **shadow_rise at frame 183 (t=1.83 s, q1_rem=20)**: SR fires within Q1 window in BOTH variants (Q1 does not trigger SR). SR resets H_error to 100 (ceil). Post-hangover Kalman explosion (H_error≈24 = 68× steady-state): W_A: 8.47→113, W_C: 9.13→132. The pre-existing 7.9% W_C excess amplifies to **17%**.
3. **W_C excess persists 27 seconds**: EPC remains active, W freezes at W_A=80.27 / W_C=89.84 (ratio=1.119) for frames 1150–2700 (t=11.5–27 s). At first DT frame 2852 (t=28.52 s): W_A=72.3, W_C=81.9, ratio=**1.133**. W is frozen during DT (EPC active). The larger W_C echo estimate, accumulated from Q1+SR aggressive early adaptation that locked to echo-path states that subsequently changed under movement, gives less accurate cancellation during DT → Δdeg=−0.110.

- In ZJYUt0O0AE and wVYSGVTTak, SR does not fire within the Q1 window (or fires later), so the W divergence does not amplify → net AECMOS gain (+0.088, +0.038).
- In xFk7igecuk, SR fires within Q1 at q1_rem=20 → W divergence amplified → persists to late DT speech.

**Classification B** (per root-cause audit): Q1 mechanism conceptually correct; the 100-hop sustained-elevation approach is movement-unsafe when shadow_rise fires within the Q1 window.

**Structural fragility confirmed:** Variant C's safety was timing-dependent, as noted in Phase 2.1 analysis. The SR-within-Q1 failure mode exposes a new vulnerability not present in the 9-case cohort: none of the Phase 2.1 DT_mvmt cases had SR firing within the Q1 window close enough to amplify W divergence to the DT-speech period.

---

## Conclusion

**Phase 3 FAIL.** Variant C (2× / 50+50 = 1.0 s) fails Gate 1 on `xFk7igecuke0R5JMfREyDg` (DT_mvmt Δdeg=−0.110).

Per hard rules (verbatim):
> "if fail, close Q1 candidate as no-ship substrate and do not tune further without new evidence."

**Q1 candidate `use_q1_true_delay_transient_leakage` (Variant C, R1+R2) is CLOSED as no-ship substrate.**

Disposition:
- R1+R2 amplitude/duration matrix is exhausted: variants B/D/E/F failed on Phase 2.1 9-case; C passed Phase 2.1 but fails Phase 3 12-case.
- R4 (EPV counter reset) was deferred because R1+R2 appeared to solve the 9-case cohort. Phase 3 reveals R1+R2 alone does not have a safe parameter point: shorter windows (C) avoid EPV multiplicative failure but expose a Q1+SR interaction failure on new movement cases.
- **Phase 3.1 re-entry guard (2026-05-25): FAIL** — `use_q1_terminate_on_non_delay_epc` terminates Q1 on SR/EPV while active, restoring steady leakage same-hop. Trace on xFk7igecuk: V1 PASS (Q1 terminates at SR183), V2 PASS (lc restored to steady), V3 FAIL (W_D/W_A=1.169 at SR+10 — same 17% excess as C; criterion: <1.12). Root cause of guard failure: 79 pre-SR Q1 hops elevated H_error_D to the same level as H_error_C; 7 hangover frames at steady lc are insufficient to close the gap; K_D ≈ K_C at frame 190 → W_D = W_C post-explosion. Guard is mechanically correct but architecturally insufficient — the pre-SR excess is already the seed of failure. 12-case AECMOS NOT authorized. See `docs/v3_21_q1_phase3_reentry_guard_design.md`.
- **AEC3 comparison:** AEC3 does not terminate its delay transient on `gain_change` events (`RefinedFilterUpdateGain` gain_change path is a TODO stub — no H_error reset, no SetConfig(initial)). AEC3 has no `shadow_rise` event (PBFDKF-specific Phase D mechanism). Therefore `use_q1_terminate_on_non_delay_epc` is NOT strict AEC3 parity — it is a PBFDKF integration guard. Reclassification: `use_q1_true_delay_transient_leakage` = AEC3 parity candidate (delay_change transient path); `use_q1_terminate_on_non_delay_epc` = PBFDKF integration adapter only. See `docs/v3_21_q1_phase3_xfk7igec_failure_audit.md` §5.
- **Effective fix paths (all require new authorization):** (1) Window shorter than 79 hops — SR fires at hop 79 in xFk7igecuk; no universal safe value found in R1+R2 matrix. (2) AEC3-style H_error spike (kHErrorInitial=10000 reset) on delay_first — fast transient decay before SR can amplify; fundamental redesign, not a Q1 parameter tweak. (3) W norm-ratio guard during Q1 — terminate Q1 early if W_C/W_A exceeds threshold; requires tracking W_A reference state; novel mechanism beyond current AEC3 scope.
- Code modules stay default-OFF research substrate. No production change.

**Scope note:** This closure applies to `use_q1_true_delay_transient_leakage` (leakage-window approach only — no W reset, no H_error spike, no config switch, no AecState reset). Fix path (2) above is not a parameter variant of this candidate; it represents the full AEC3 delay-change chain (W=0 + H_error=10000 + SetConfig(initial) + AecState initial_state reset), which has not been tested. The failure of the leakage-window approach does not close the AEC3 delay-change parity arc. See `docs/v3_21_full_delay_change_chain_plan.md`.

---

## Config (Phase 3 candidate — CLOSED no-ship)

```
q1_tdt_lc_factor  = 2.0   (was 5.0)
q1_tdt_ld_factor  = 2.0   (was 5.0)
q1_tdt_transient_hops = 50    (was 250)
q1_tdt_smoothing_hops = 50    (was 100)
# total = 100 hops = 1.0 s
# STATUS: CLOSED — Phase 3 FAIL on xFk7igecuk DT_mvmt
```
