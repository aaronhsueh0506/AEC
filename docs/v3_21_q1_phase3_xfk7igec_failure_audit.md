# xFk7igecuk Phase 3 Failure Audit

**Case:** `xFk7igecuke0R5JMfREyDg_doubletalk_with_movement`  
**Phase 3 result:** Variant C Δdeg = −0.110 (Gate 1 FAIL; threshold −0.05).  
**Scope:** Read-only per-frame trace. No retuning, no new variant, no 800-case.  
**Deliverable from:** `python/v3_21_q1_xfk7_audit.py`

---

## 1. Event Timing

| Field | Value |
|---|---|
| Event source | `delay_first` |
| Event frame | 104 |
| Event time | 1.040 s |
| Q1 transient window | frames 104–153 (t = 1.04–1.54 s) |
| Q1 smooth window | frames 154–203 (t = 1.54–2.04 s) |
| Q1 total duration | 100 hops = 1.0 s |
| Shadow_rise #1 fires (ON + OFF) | Frame 183, t = 1.830 s — **within Q1 window, q1_rem=20** |
| Shadow_rise #2 fires (ON + OFF) | Frame 469, t = 4.690 s — after Q1 expired |
| EPV fires | No (neither ON nor OFF variant) |

---

## 2. DT Activity vs Q1 Window — Original Diagnosis INCORRECT

The Phase 3 verdict stated the failure mechanism as:

> *"delay event fires during or just before an active DT speech segment → elevated lc (2×) for 100 hops overlaps DT speech → over-adaptation → Δdeg = −0.110."*

**This is wrong.** Per-frame trace:

| Metric | Value |
|---|---|
| dt_from_energy in Q1 window (frames 104–203) | 0.000 throughout — DT NOT active |
| First DT frame | 2852 (t = 28.52 s) — **27 seconds after Q1 window** |
| Last DT frame | 3582 (t = 35.82 s) |
| DT-active frames total | 54 of 3678 frames |

DT speech occurs at t = 28–36 s. Q1 window is t = 1.04–2.04 s. There is no DT during the Q1 window.

---

## 3. Actual Failure Mechanism: Q1 + Shadow_rise Interaction

The real mechanism is a three-stage W_norm divergence chain.

### Stage 1 — Q1 lc builds W_C excess (frames 104–182, 79 hops)

At delay_first (frame 104), W resets (both A and C identical: W=3.73). Q1 elevates lc_C = 2 × lc_steady → H_error grows faster in C → slightly higher Kalman gain → W_C grows 6.5% faster than W_A. No DT throughout.

| Frame | t_s | W_A | W_C | Ratio | dt_energy |
|---|---|---|---|---|---|
| 104 | 1.040 | 3.732 | 3.732 | 1.000 | 0.000 |
| 150 | 1.500 | 7.951 | 8.466 | 1.065 | 0.000 |
| 182 | 1.820 | 8.394 | 9.057 | 1.079 | 0.000 |

### Stage 2 — SR amplifies divergence (frame 183, q1_rem=20)

Shadow_rise fires at frame 183 (t=1.83 s), within Q1 window (q1_rem=20). SR fires identically in BOTH A and C — Q1 does not trigger or suppress SR. SR resets H_error to 100 (ceil) for both. Main filter update blocked during 7-frame hangover (frames 183–189).

At frame 190 hangover ends. H_error ≈ 24 for both (nearly identical). H_error=24 is 68× above steady-state H_error ≈ 0.35 → large Kalman gain → both W_A and W_C undergo a large one-shot update.

**Critical asymmetry:** W_C was already 7.9% larger than W_A when SR fired. The large SR-driven update amplifies this:

| Frame | t_s | W_A | W_C | Ratio | Note |
|---|---|---|---|---|---|
| 182 | 1.820 | 8.394 | 9.057 | 1.079 | pre-SR |
| 183 | 1.830 | 8.474 | 9.130 | 1.078 | SR fires, W frozen |
| 190 | 1.900 | 113.062 | 132.765 | 1.174 | post-hangover Kalman explosion |
| 200 | 2.000 | 109.107 | 127.704 | 1.170 | Q1 expired, ratio persists |

Pre-existing 7.9% W_C excess amplifies to **17%** via SR post-hangover Kalman gain.

### Stage 3 — W_C excess persists to DT speech (t = 28 s)

A second SR fires at frame 469 (q1_rem=0, Q1 expired): W_A=93.7, W_C=112.8 (ratio 1.204). EPC remains active throughout the remainder of the clip. During the extended EPC period, W is completely frozen at W_A=80.273, W_C=89.838 (ratio=1.119) from frame 1150 to 2700 (t=11.5–27 s). At first DT frame 2852 (t=28.52 s): W_A=72.295, W_C=81.886, ratio=**1.133**.

During DT speech (frames 2852–3582), W is frozen in both variants. W_C carries a 13.3% norm excess accumulated from the Q1+SR interaction 27 seconds earlier. The larger W_C⋅X echo estimate is less accurate at DT time (accumulated from more aggressive early adaptation that locked to echo path states that subsequently changed under movement) → worse residual → RES suppression mis-calibrated → Δdeg = −0.110.

---

## 4. Complete W_norm Trajectory

| Frame | t_s | W_A | W_C | Ratio | dt_C | Event |
|---|---|---|---|---|---|---|
| 100 | 1.00 | 21.63 | 21.63 | 1.000 | 0.000 | pre-event |
| 104 | 1.04 | 3.73 | 3.73 | 1.000 | 0.000 | delay_first |
| 150 | 1.50 | 7.95 | 8.47 | 1.065 | 0.000 | Q1 active |
| 183 | 1.83 | 8.47 | 9.13 | 1.078 | 0.000 | **SR fires (q1_rem=20)** |
| 200 | 2.00 | 109.1 | 127.7 | 1.170 | 0.000 | post-SR explosion |
| 469 | 4.69 | 93.7 | 112.8 | 1.204 | 0.000 | 2nd SR fires (q1_rem=0) |
| 1150 | 11.5 | 80.27 | 89.84 | 1.119 | 0.000 | W frozen starts |
| 2700 | 27.0 | 80.27 | 89.84 | 1.119 | 0.000 | W frozen ends |
| 2852 | 28.5 | 72.30 | 81.89 | **1.133** | 0.521 | **first DT — W_C excess persists** |
| 3600 | 36.0 | 51.90 | 62.32 | 1.201 | 0.047 | late DT |

---

## 5. AEC3 Parity: DT Gating in RefinedFilterUpdateGain

**Finding: AEC3 has NO doubletalk/nearend gate in RefinedFilterUpdateGain.**

From `docs/aec3_extracts/src/aec3/refined_filter_update_gain.cc:91–126`:

```
Compute() gates: PoorSignalExcitation | saturated_capture | startup (call_counter).
H_error += leakage × erl  — always runs, no DT/nearend check.
```

The absence of a DT gate is not a missing port — it is the same in AEC3. AEC3 also over-adapts during any post-delay-change speech period, relying on the rapid H_error decay (from kHErrorInitial=10000) to terminate the high-gain period quickly.

### 5a. AEC3 delay_change behavior (exact)

`Subtractor::HandleEchoPathChange` on delay_change fires a **full reset cascade**:
1. `filter.HandleEchoPathChange()` on main + coarse filters — W coefficients zeroed, H_error reset to `kHErrorInitial = 10000`
2. `SetConfig(refined_initial, coarse_initial, /*allow_overriding_settings=*/true)` — leakage rates set to initial transient values (`leakage_converged=5e-3/block`, `leakage_diverged=5e-1/block`), `config_change_duration_blocks=250` transitions in place
3. `SetSizePartitions(initial.length_blocks, /*adapt=*/true)` — partition count can be reset to initial (shorter filter, faster re-acquisition)
4. `AecState` initial_state reset — re-enters full startup mode; convergence latches cleared

**AEC3 exits the delay transient via `TransitionTriggered()` / `ExitInitialState()`** — internal convergence criteria inside AecState, NOT via a later gain_change or EPC event. Once the filter converges to the new delay, the transient expires on its own schedule.

### 5b. AEC3 gain_change behavior (exact)

`gain_change` in AEC3 (fired when `echo_path_variability.gain_change != kNone`):
- Rate-limited by `gain_change_hangover` (consecutive gain_change events suppressed)
- `RefinedFilterUpdateGain::Compute()` gain_change path is **TODO** — no H_error reset, no `SetConfig(initial)`, no `SetSizePartitions`. The comment at `refined_filter_update_gain.cc:128–138` marks this as not yet implemented.
- `AecState::HandleEchoPathChange(gain_change=true)` → ERLE reset to `false` only; does NOT trigger initial_state reset
- **gain_change does NOT terminate the delay transient.** AEC3's delay transient continues through gain_change events until `TransitionTriggered()` fires.

### 5c. AEC3 has no shadow_rise

AEC3 has no `shadow_rise` event. Our `shadow_rise` detector is a PBFDKF-specific Phase D mechanism: it fires when the shadow filter error grows relative to the main filter, indicating filter mistracking in a way that PBFDKF's Kalman dynamics can generate but AEC3's NLMS shadow filter does not.

### 5d. Our Q1 strategy and its architectural classification

**AEC3's delay_change strategy (HandleEchoPathChange):**
- H_error_.fill(10000) — hard reset to kHErrorInitial
- call_counter_ = 0 — startup gate re-arms (blocks first size_partitions frames)
- → Very large initial Kalman gain spike; decays fast once filter update runs
- → Transient exits via internal convergence criteria (TransitionTriggered)

**Our Q1 (Variant C) strategy:**
- H_error NOT reset — starts from converged floor (~0.35)
- lc elevated 2× for 100 hops — H_error grows slowly from floor
- → Moderate, sustained gain elevation over 100 hops

AEC3's spike approach and Q1's sustained approach differ fundamentally in the post-change adaptation trajectory. AEC3's approach causes a fast one-time W update; Q1's approach causes sustained elevated adaptation. In the movement context, AEC3's spike terminates before SR can amplify it; Q1's sustained window is still active when SR fires at frame 183.

**This is not a missing AEC3 guard** — it is a structural difference in the post-delay-change adaptation strategy that becomes unfavorable in the movement context.

### 5e. Re-entry guard architectural classification (Phase 3.1 finding)

The Phase 3.1 re-entry guard (`use_q1_terminate_on_non_delay_epc`) was designed to terminate Q1 when shadow_rise fires within the Q1 window. Trace analysis shows:
- Guard correctly terminates Q1 at SR frame 183 and restores steady lc
- BUT W_D/W_A at frame 190 = 1.174 — **same 17% excess as Variant C without guard**
- Guard has negligible effect on xFk7igecuk (W_D ≡ W_C at all post-SR frames)

Root cause of guard failure: the 7.9% pre-SR W_C excess (accumulated over 79 Q1 hops) is already the seed of the amplification. Restoring lc at frame 183 does not reduce H_error_D below H_error_C because H_error is already elevated from 79 hops of Q1; 7 hangover frames at steady lc are insufficient to close the gap.

**Architectural classification:**
- `use_q1_true_delay_transient_leakage` = **AEC3 parity candidate** (delay_change transient path)
- `use_q1_terminate_on_non_delay_epc` = **PBFDKF integration guard** — prevents Phase D pseudo-events (SR/EPV) from contaminating the Q1 delay transient; NOT AEC3 parity; NOT required by AEC3 design. Only needed if the integration creates unwanted cross-event coupling. For xFk7igecuk this guard is insufficient; for other cases (SR fires very early in Q1) it may help marginally.

See `docs/v3_21_q1_phase3_reentry_guard_design.md` for full trace and disposition.

---

## 6. Classification

**Classification: B — Q1 mechanism correct, movement-context parameterisation unsafe.**

| Question | Finding |
|---|---|
| Event source | `delay_first`, frame 104, t=1.04 s |
| DT overlap with Q1 window | **Zero** — DT is at t=28–36 s, 27 s after Q1 |
| W_C excess at SR fire time | +7.9% (from Q1 elevated lc over 79 hops) |
| W_C excess post-SR explosion | +17.4% (SR Kalman gain amplifies pre-existing excess) |
| W_C excess at first DT | +13.3% (persists 27 s through frozen W period) |
| AEC3 has DT gate in RefinedFilterUpdateGain | No — not a missing port |
| SR fires within Q1 window | Yes — frame 183, q1_rem=20 — key amplifier |
| EPV interaction | No EPV fires |
| Root cause | Q1 lc → W_C divergence → SR amplification → W_C excess at DT time |
| Classification | **B** |

**Why B and not A, C, or D:**
- Not A: no missing AEC3 guard; AEC3 also has no DT gate
- Not C: not a DSP limit; AEC3's alternative strategy (H_error reset vs lc elevation) avoids the issue in different way
- Not D: mechanism fully traced
- B: Q1 mechanism is conceptually correct (accelerate post-change convergence) but the sustained-elevation approach is movement-unsafe when SR fires within the Q1 window

---

## 7. Implication for Q1 Arc

**Phase 3 FAIL verdict stands.** The corrected mechanism reinforces it: 100-hop sustained lc elevation in a DT+movement case causes SR-amplified W divergence that persists to DT speech.

**Phase 3.1 re-entry guard FAILS on xFk7igecuk:** Per `docs/v3_21_q1_phase3_reentry_guard_design.md`, the guard terminates Q1 at SR correctly but W_D/W_A = 1.174 at frame 190 (same 17% excess as C). The pre-SR W_C excess (79 hops of Q1 elevation) is the root cause; restoring lc at SR time does not undo the pre-existing excess nor prevent the SR Kalman explosion from amplifying it. 12-case AECMOS NOT authorized.

**Structural fixes implied by this diagnosis (all require new authorization):**

1. **Shorter total window:** A window shorter than the SR latency post-delay_first (here: 79 hops) would expire before SR fires. R1+R2 matrix already showed variants shorter than 100 hops fail on other cases; no universal safe value found.

2. **AEC3-style H_error spike:** Replace Q1's sustained lc elevation with AEC3's delay_change H_error reset (kHErrorInitial=10000). Spike decays fast (call_counter gate); both A and C start from the same high H_error, so no systematic W_C excess builds. This is a fundamental redesign, not a Q1 parameter tweak.

3. **W norm-ratio guard during Q1:** If W_C/W_A exceeds a threshold during Q1, restore steady lc early. Requires tracking W_A reference state. Novel mechanism beyond current AEC3 scope.

None of these are authorized under current hard rules.

---

## 8. Correction to Phase 3 Verdict Mechanism Statement

`docs/v3_21_q1_phase3_12case_verdict.md` Section "Failure Diagnosis" states:

> *"delay event fires during or just before an active DT speech segment → elevated lc (2×) for 100 hops overlaps DT speech → over-adaptation → Δdeg = −0.110"*

**Correct mechanism (this audit):**

> Q1 elevated lc (2×, 100 hops) causes W_C to grow 7.9% above W_A before shadow_rise fires at t=1.83 s (frame 183, q1_rem=20, within Q1 window). SR's post-hangover Kalman explosion amplifies the divergence to 17%. W_C excess of 13.3% persists 27 seconds through subsequent path changes and frozen-W period. At DT speech time (t=28–36 s), W_C echo estimate is less accurate → Δdeg = −0.110. **No DT is active during the Q1 window.**

The FAIL verdict, gate result, and arc closure statement in the Phase 3 document remain correct. Only the mechanistic explanation was wrong.
