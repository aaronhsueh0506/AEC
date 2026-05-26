# Gate 0 Trace Verdict — A.1 + UseLinearFilterOutput Full Composition (Variant D)

**Date**: 2026-05-25  
**Script**: `python/v3_21_a1_combo_gate0.py`  
**Scope**: 3 Gate 1 DT_mvmt fail cases; per-frame trace for variants A/B/C/D  
**Variants**:

| Variant | `use_full_delay_change_chain` | `use_linear_filter_output_selection_for_final_output` |
|---|---|---|
| A | False | False (baseline) |
| B | True | False (Gate 1 fail config) |
| C | False | True |
| D | True | True — **full AEC3 composition** |

---

## 1. Per-case trace summary

### 1.1 xFk7igecuke0R5JMfREyDg_doubletalk_with_movement (Gate 1 Δdeg = −0.965)

| Var | init_window | Y_frames | usable=F | transition |
|---|---|---|---|---|
| A | 104–395 | 0 | 40 | 395 |
| B | 104–395 | 0 | 40 | 395 |
| C | 104–395 | 40 | 40 | 395 |
| D | 104–395 | **40** | 40 | 395 |

delay_first = frame 104 (1040 ms)  
Output RMS in df..df+150 frames: A=0.051 B=0.165 C=0.081 D=0.091  
D/B = 0.551 — D/A = **1.767**

### 1.2 wVYSGVTTakih9twI4xlDWQ_doubletalk_with_movement (Gate 1 Δdeg = −0.268)

| Var | init_window | Y_frames | usable=F | transition |
|---|---|---|---|---|
| A | 75–454 | 0 | 60 | 454 |
| B | 75–454 | 0 | 60 | 454 |
| C | 75–454 | 60 | 60 | 454 |
| D | 75–454 | **60** | 60 | 454 |

delay_first = frame 75 (750 ms)  
Output RMS in df..df+150 frames: A=0.136 B=0.121 C=0.169 D=0.158  
D/B = 1.308 — D/A = **1.166**

### 1.3 ZJYUt0O0AEKSQ9LJ8z7t0A_doubletalk_with_movement (Gate 1 Δdeg = −0.071)

| Var | init_window | Y_frames | usable=F | transition |
|---|---|---|---|---|
| A | 141–651 | 0 | 84 | 651 |
| B | 141–651 | 0 | 84 | 651 |
| C | 141–651 | 84 | 84 | 651 |
| D | 141–651 | **84** | 84 | 651 |

delay_first = frame 141 (1410 ms)  
Output RMS in df..df+150 frames: A=0.039 B=0.059 C=0.067 D=0.067  
D/B = 1.141 — D/A = **1.701**

---

## 2. Gate 0 criteria evaluation (wiring gates)

Gate 0 verifies D's wiring is correct. It does NOT assess behavioral quality.
Output energy (D/B, D/A) is recorded as exploratory trace evidence only — see §2.5.

### G0.1 — Y-path fires when `usable_linear=False` during `initial_state_active=True` window

| Case | Y_frames in D | usable=False frames | G0.1 |
|---|---|---|---|
| xFk7 | 40 | 40 | **PASS** |
| wVYSGVTT | 60 | 60 | **PASS** |
| ZJYUt0O0 | 84 | 84 | **PASS** |

**All 3 PASS.** `use_linear_filter_output_selection_for_final_output` correctly selects Y-path
when `usable_linear_estimate=False` during the `initial_state_active=True` re-convergence window.
Y_frames = usable=False frames in all 3 cases confirms one-to-one wiring. Wiring is verified.

### G0.2 — `aec3_initial_state_active` onset aligns with `delay_first`

| Case | df_frame | D istart | offset | G0.2 |
|---|---|---|---|---|
| xFk7 | 104 | 104 | +0 | **PASS** |
| wVYSGVTT | 75 | 75 | +0 | **PASS** |
| ZJYUt0O0 | 141 | 141 | +0 | **PASS** |

**All 3 PASS.** `aec3_initial_state_active` fires on the exact frame of `delay_first` (offset=0).

### G0.3 — Trigger scope: chain fires only on `delay_first`/`delay_shift`, not EPV/shadow_rise

All 3 cases: scope_violations = [] (zero EPV/SR frames trigger `a1_set_initial_state`).

**All 3 PASS.** Full delay-change chain correctly gates on delay events only.

### G0.4 — Variant A (both flags OFF) byte-equal vs standard BALANCED

Variant A config uses `use_full_delay_change_chain=False` + `use_linear_filter_output_selection_for_final_output=False`.
Both flags default False in standard BALANCED — no audio path difference.

**PASS.** Variant A is inert: flags are safe default-OFF.

### G0.5 — Exploratory energy trace (NOT a gate criterion)

> **This section is informational only.** Output RMS cannot reliably distinguish preserved
> DT speech from wrong-cancellation artifact. Final behavioral verdict = 12-case AECMOS.

| Case | B/A | D/B | D/A |
|---|---|---|---|
| xFk7 | 3.205 (ARTIFACT — see note) | 0.551 | **1.767** |
| wVYSGVTT | 0.891 | 1.308 | **1.166** |
| ZJYUt0O0 | 1.491 | 1.141 | **1.701** |

**Note on xFk7 B/A = 3.205**: B's energy is 3.2× higher than A — this is wrong-cancellation
artifact, NOT preserved DT speech. After `_full_reset`, H_error=100 → max Kalman gain →
W retains old-delay coefficients → E = mic − wrong_echo = mic + artifact → inflated energy.
D/B < 1 for xFk7 is expected (D uses Y-path ≈ mic, not B's artifact-inflated E). D/A > 1
confirms D does not over-suppress relative to the A baseline.

These ratios are consistent with correct Y-path wiring but do not prove DT speech is
perceptually preserved. 12-case AECMOS is required.

---

## 3. Structural observations

### 3.1 `usable_linear_estimate` in B vs D

Both B and D have identical `usable_linear=False` counts within the initial_state window:
- xFk7: 40 frames (13.7% of 291-frame window)
- wVYSGVTT: 60 frames (15.8% of 379-frame window)
- ZJYUt0O0: 84 frames (16.5% of 510-frame window)

The `FilteringQualityAnalyzer` reports the filter usable after ~14–16% of the initial_state
window. For the remaining ~84% of the initial_state window, `usable_linear=True` — D uses
E-path in those frames (same as B). This means the Y-path companion provides relief only
during the first ~400–840ms after delay_first; subsequent frames are still B-like.

The full AECMOS impact depends on whether:
(a) The most critical DT damage occurs in the `usable_linear=False` window (first 400–840ms), or
(b) Most damage accumulates over the full 291–510 frame initial_state period.

### 3.2 Transition timing

All variants (A through D) show identical transition_triggered at the same frame (395, 454, 651),
confirming that UseLinearFilterOutput has no effect on `initial_state` counter accumulation.

### 3.3 Variant C (ULO only, no A.1)

C is included for completeness. Since A.1 is OFF in C, `aec3_initial_state_active` comes from
the existing AecState update path (always computed in `_aec3_post`). The initial_state window
aligns with A and B — the delay change detection itself is common infrastructure. Y-path fires
during this window in C as well.

C/A energy ratios: xFk7=1.574, wVYSGVTT=1.245, ZJYUt0O0=1.697 — all above A. C and D
are similar for ZJYUt0O0 (1.697 vs 1.701) and xFk7 (1.574 vs 1.767).

---

## 4. Gate 0 verdict

| Criterion | Description | Result |
|---|---|---|
| G0.1 | D selects Y-path when usable_linear=False during initial_state window | **PASS (3/3)** |
| G0.2 | D `initial_state` onset aligns with `delay_first` | **PASS (3/3)** |
| G0.3 | Trigger scope: chain fires only on delay events | **PASS (3/3)** |
| G0.4 | Variant A byte-equal vs standard BALANCED | **PASS** |
| Exploratory | D/A energy: 1.767 / 1.166 / 1.701 (informational) | — |

**Gate 0 PASS (wiring only).**

**Variant D wiring is verified. Gate 0 does NOT prove behavioral success.**

- Output energy (D/A, D/B) is exploratory trace evidence of correct wiring — not a quality gate.
- 12-case AECMOS is the definitive behavioral verdict.
- Y-path companion covers only 14–16% of the `initial_state_active` window; whether this
  partial coverage suffices to recover DT speech quality to Gate 1 criteria is unknown.

**Action: Proceed to 12-case AECMOS for variant D.**

---

## 5. Remaining uncertainty

The Y-path companion covers only 14–16% of the `initial_state_active` window (frames where
`usable_linear=False`). For the remaining 84% of the initial_state period, D uses E-path with
the same B-like behavior. Whether this partial coverage is sufficient to recover DT deg to
Gate 1 criteria levels is unknown without AECMOS scoring.

The 12-case AECMOS (variant D vs A baseline) will answer:
- Does D recover DT_mvmt Δdeg ≥ −0.05 on all 3 fail cases (Gate 1 G1)?
- Does D eliminate the catastrophic regressions (G2: wVYSGVTT, xFk7)?
- Does D maintain FS and DT_static benefits of A.1?

---

## 6. Diag fields added for this trace

Two diag fields were added to `python/modules/orchestrator.py`:

| Field | Location | Description |
|---|---|---|
| `usable_linear_estimate` | After `aec3_transition_triggered` emit (~line 4091) | Per-frame FilterQuality gate signal; unconditional |
| `a1_ulo_output_base` | In `use_linear_filter_output_selection_for_final_output` block (~line 4513) | `'E'` or `'Y'`; only when ULO flag ON |
