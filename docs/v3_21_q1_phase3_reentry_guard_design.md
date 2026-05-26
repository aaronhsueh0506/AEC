# v3.21 Q1 Phase 3.1 — Re-entry Guard Design and Trace Verdict

**Candidate:** Variant C + `use_q1_terminate_on_non_delay_epc=True`  
**Scope:** Design + per-frame trace on xFk7igecuke0R5JMfREyDg_doubletalk_with_movement.  
**Date:** 2026-05-25

---

## 1. Design Rationale

Phase 3 failed because shadow_rise fired at frame 183 (q1_rem=20, within Q1 window).
The Q1+SR interaction amplified the pre-SR W_C excess from 7.9% to 17.4%.
Root cause (per `docs/v3_21_q1_phase3_xfk7igec_failure_audit.md`):
Q1 elevated lc (2×) for 79 hops before SR fired → W_C 7.9% above W_A → SR explosion amplified to 17%.

**Guard hypothesis:** If Q1 terminates immediately at SR (restoring steady lc same-hop), then:
- lc in D = lc in A from SR frame onward
- H_error growth rate in D = H_error growth rate in A during SR hangover (7 frames)
- Kalman gain K_D ≈ K_A at post-hangover frame 190
- W excess ratio would not be amplified beyond the pre-SR 7.9%

---

## 2. Guard Mechanism

**`use_q1_terminate_on_non_delay_epc`** (default-OFF, requires `use_q1_true_delay_transient_leakage=True`):

On `epv_event.fired and not _epv_suppressed` OR `rise_event.fired` while `_q1_tdt_rem > 0`:
1. `_q1_tdt_rem = 0` — Q1 counter halted
2. `filter._leakage_converged = _q1_tdt_lc_steady` — restored same-hop
3. `filter._leakage_diverged = _q1_tdt_ld_steady` — restored same-hop
4. `_diag['q1_terminated_by'] = 'shadow_rise' | 'epv'`

`delay_shift` during active Q1 re-arms Q1 (as before — true delay event).

**Classification:** This is a **PBFDKF integration guard**, NOT an AEC3 parity mechanism.
AEC3 has no shadow_rise event and does not terminate its delay transient on gain_change events.
See Section 5 (AEC3 comparison) and the updated audit doc.

---

## 3. Implementation Sites

| File | Line range | Change |
|---|---|---|
| `python/modules/config.py` | after `q1_tdt_ld_factor` | `use_q1_terminate_on_non_delay_epc: bool = False` |
| `python/modules/orchestrator.py` | EPV site (`if epv_event.fired`) | Guard block: rem=0 + lc restore + diag |
| `python/modules/orchestrator.py` | SR site (`if rise_event.fired`) | Guard block: rem=0 + lc restore + diag |
| `python/modules/orchestrator.py` | Q1 block | `q1_rem` and `q1_terminated_by` added to `_diag` |

All changes default-OFF. Byte-equal when flag is False.

---

## 4. Trace Verification (xFk7igecuke0R5JMfREyDg)

Script: `python/v3_21_q1_phase3p1_trace.py`  
Variants: A (OFF), C (ON/no guard), D (ON+guard)

### Key frame table (excerpt)

| Frame | t_s | qC | qD | lc_A | lc_C | lc_D | W_A | W_C | W_D | W_D/W_C |
|---|---|---|---|---|---|---|---|---|---|---|
| 104 | 1.040 | 100 | 100 | 0.00250 | 0.00500 | 0.00500 | 3.73 | 3.73 | 3.73 | 1.000 |
| 182 | 1.820 | 22 | 22 | 0.00250 | 0.00360 | 0.00360 | 8.39 | 9.04 | 9.04 | 1.000 |
| **183** | **1.830** | **21** | **21→0** | 0.00250 | 0.00355 | **0.00250** | 8.47 | 9.13 | 9.13 | 1.000 |
| 184 | 1.840 | 20 | **0** | 0.00250 | 0.00350 | **0.00250** | 8.47 | 9.13 | 9.13 | 1.000 |
| 190 | 1.900 | 14 | 0 | 0.00250 | 0.00320 | 0.00250 | **113.06** | **132.76** | **132.76** | **1.000** |
| 2852 | 28.520 | 0 | 0 | 0.00250 | 0.00250 | 0.00250 | 72.29 | 81.89 | 81.81 | 0.999 |

### Verification results

| Check | Criterion | Result | Value |
|---|---|---|---|
| V1 | Q1 terminates at SR1 (frame 183) in D | **PASS** | term_frame=183, SR1=183 |
| V2 | lc_D == lc_A at SR1+1 | **PASS** | lc_D=0.002500 lc_A=0.002500 |
| V3 | W_D/W_A at SR1+10 < 1.10 (no 17% amplification) | **FAIL** | W_D/W_A=1.174 = same as C |
| V4 | No new SR/EPV events introduced in D | **PASS** | new SR frames: none |

**V3 FAIL — guard does not prevent the 17% amplification.** W_D = W_C = 132.76 at frame 190. The guard is mechanically correct (V1/V2 pass) but architecturally insufficient.

---

## 5. Root Cause: Why the Guard Fails on xFk7igecuk

### Mechanism decomposition

**Pre-SR phase (frames 104–182, 79 hops):**
Both C and D use elevated lc (2×, same Q1 window active). W_C = W_D grow faster than W_A.
At frame 183 (SR fires): W_A=8.47, W_C=W_D=9.13 (ratio 1.079). H_error also elevated for C/D vs A.

**SR fires at frame 183:**
Guard activates: D's lc restored to steady (same as A). C keeps elevated lc.

**Hangover phase (frames 184–189, 7 frames, W frozen):**
H_error refreshes each frame:
- A: steady lc → H_error grows slowly
- D: steady lc (restored) → H_error grows at same rate as A
- C: elevated lc → H_error grows faster than D

BUT: H_error for D at frame 183 is ALREADY elevated vs A (from 79 hops of elevated lc).
Even with steady lc from frame 184, H_error_D cannot drop — it only grows more slowly.
Over 7 hangover frames, H_error_D remains near H_error_C (ceiling=100 for both).

**Post-hangover (frame 190):**
H_error_A ≈ H_error_D ≈ H_error_C (all near ceiling from the elevated starting point).
Kalman gain K_A ≈ K_D ≈ K_C.
W update: W_190 = W_183 + K × error.
With identical K and identical starting W_183 for C and D (both=9.13):
W_D_190 = W_C_190 = 132.76.

**Conclusion:** The guard's lc restoration at frame 183 does not reduce H_error for D relative to C.
H_error_D at frame 183 is already elevated from 79 hops of Q1. Seven hangover frames at steady lc
are insufficient to bring H_error_D to H_error_A's level. The Kalman gain at frame 190 is the same
for A, C, and D. The 7.9% pre-SR W_C excess is amplified to 17.4% identically in C and D.

### Structural gap

The guard is correct for the scenario it was designed for (SR fires VERY early in Q1,
before significant excess builds). For xFk7igecuk, SR fires at q1_rem=20 (79/100 of the
window consumed). The pre-SR excess is 7.9%, which is already the seed of the failure.

---

## 6. AEC3 Architecture Note (New Findings — see audit doc update)

AEC3 on delay_change:
- `Subtractor::HandleEchoPathChange` → full filter reset + `SetConfig(refined_initial)` + `SetSizePartitions(initial.length_blocks)` + `AecState` initial_state reset
- AEC3 exits the delay transient via `TransitionTriggered()` / `ExitInitialState()`, NOT via later gain_change or EPC events
- `gain_change` in AEC3: rate-limited, `RefinedFilterUpdateGain` gain_change path is TODO (no H_error reset, no SetConfig(initial)), AecState ERLE reset(false) only. Does NOT terminate the delay transient.
- AEC3 has no shadow_rise event (no PBFDKF divergence detector)

**Reclassification:**
- Q1 true-delay transient leakage (`use_q1_true_delay_transient_leakage`) = **AEC3 parity candidate** for the delay_change transient path (with different implementation: sustained lc vs H_error spike)
- SR/EPV terminate guard (`use_q1_terminate_on_non_delay_epc`) = **PBFDKF integration guard** — prevents Phase D pseudo-events (SR/EPV) from contaminating the Q1 delay transient. NOT AEC3 parity; NOT a parity obligation. Only needed if the integration creates unwanted cross-event coupling.

---

## 7. Disposition

| Item | Status |
|---|---|
| Guard implementation | **COMPLETE** (config flag + 2 orchestrator sites) |
| xFk7igecuk trace | **V3 FAIL** — guard does not prevent 17% W excess |
| 12-case AECMOS | **NOT AUTHORIZED** — trace criterion not met |
| 800-case | NOT AUTHORIZED |

**Q1 arc remains CLOSED (Phase 3 FAIL verdict stands).**

The guard does not fix xFk7igecuk. Effective fix paths (all require new authorization):

| Path | Description | Constraint |
|---|---|---|
| Shorter window | Total hops < 79 (SR latency for this case). Variant C at 100 hops was already the shortest that passed 9-case Phase 2.1. | SR latency varies by case; no universal safe value in R1+R2 matrix |
| AEC3 spike approach | Replace sustained lc elevation with H_error=10000 reset on delay_first. Fast spike decays before SR amplification window. | Fundamental redesign; not a Q1 parameter tweak |
| Extended R4 as counter reset | Reset Q1 counter AND W_C excess on SR. Would need W to be reset or W_C forced to W_A on SR. | Hard rule: no W.fill(0); no W reset within Q1 |

None of these are authorized under current hard rules.
Code stays default-OFF. Production `main` unchanged.
