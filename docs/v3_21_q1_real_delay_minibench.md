# v3.21 Q1 Path C — Real Delay-Event Mini-Bench

**Date:** 2026-05-25  
**Status:** COMPLETE — evidence collected; Q1 stance updated to (b-revised)  
**Script:** `python/v3_21_q1_minibench.py`  
**Trace output:** `out_v3_21_q1_minibench/` (9 × `_diag.json` + `summary.json`)

---

## Background

Path 1 (real-delay-event scan, 2026-05-24) found zero existing artifact captures of
`event_delay_first` / `event_delay_shift` because all prior trace scripts set
`enable_delay_est=False`. Path C is the surgical 5-10 case mini-bench that re-runs
with `enable_delay_est=True` (BALANCED production default) to obtain actual delay-event
evidence.

No algorithm changes. No AECMOS scoring. No 800-case. Trace-only.

---

## §1 Cohort and Config

**Config:** `AecConfig.from_preset(AecPreset.BALANCED, enable_cng=True, enable_delay_est=True)`  
No manual ref pre-alignment — delay estimator handles ref ring buffer internally.

| # | Label | Stem (truncated) | Dur |
|---|-------|-----------------|-----|
| 1 | DT_mvmt | 49IIo03GZ0CYQOmeA3A0BA_doubletalk_with_movement | 38 s |
| 2 | DT_mvmt | 7GTxyTksSUqCnP5y0ILG4A_doubletalk_with_movement | 37 s |
| 3 | DT_mvmt | Hp5g1asacUCt5rJVLO1FuQ_doubletalk_with_movement | 38 s |
| 4 | DT_mvmt | XRTnTUjU5kS0mejzCqyCiw_doubletalk_with_movement | 37 s |
| 5 | FS_mvmt | 0I0XMl3M0ECO0U1N0cJvpg_farend_singletalk_with_movement | 24 s |
| 6 | FS_mvmt | Fi80N5kW9U6nwaoS04O3vQ_farend_singletalk_with_movement | 22 s |
| 7 | FS_mvmt | wlAXM0iDgkm06i7UdRww1w_farend_singletalk_with_movement | 21 s |
| 8 | DT_static (guard) | nVUnxqHLr0GTN7shWid1Ow_doubletalk | 43 s |
| 9 | FS_static (guard) | 9xjhiFbGo06hdQIsHTS6qA_farend_singletalk | 22 s |

---

## §2 Event Count Table

```
label        stem                                              dur  d_first  d_shift   epv  shdw_rise  any_delay
DT_mvmt      49IIo03GZ0CYQOmeA3A0BA_doubletalk_with_moveme    38        1        0     0          0          1
DT_mvmt      7GTxyTksSUqCnP5y0ILG4A_doubletalk_with_moveme    37        0        0     0          0          0
DT_mvmt      Hp5g1asacUCt5rJVLO1FuQ_doubletalk_with_moveme    38        1        0     1          0          1
DT_mvmt      XRTnTUjU5kS0mejzCqyCiw_doubletalk_with_moveme    37        1        0     0          0          1
FS_mvmt      0I0XMl3M0ECO0U1N0cJvpg_farend_singletalk_with    24        1        0     0          0          1
FS_mvmt      Fi80N5kW9U6nwaoS04O3vQ_farend_singletalk_with    22        1        0     0          0          1
FS_mvmt      wlAXM0iDgkm06i7UdRww1w_farend_singletalk_with    21        1        3     0          3          4
DT_static    nVUnxqHLr0GTN7shWid1Ow_doubletalk                43        1        0     1          0          1
FS_static    9xjhiFbGo06hdQIsHTS6qA_farend_singletalk         22        1        0     1          1          1
──────────────────────────────────────────────────────────────────────────────────────────────────────────────
TOTAL                                                                   8        3     3          4         11
```

**Real delay events (delay_first + delay_shift): 11 across 9 cases.**

---

## §3 Detailed Findings

### §3.1 delay_first pattern (8/9 cases)

`delay_first` fires when the delay estimator first reaches solid confidence
(`conf = 1.0`, `is_solid = True`) and Path A gate `_current_delay < 0` is true.

| Case | fire time | delay acquired | H_error before event |
|---|---|---|---|
| 49IIo03GZ0 | t=1.37 s | 1332 samp (83.3 ms) | ~0.3 (deep equilibrium) |
| Hp5g1asacU | t=0.90 s | 452 samp (28.3 ms) | ~0.6-0.8 (mid-convergence) |
| XRTnTUjU | t=1.54 s | 1000 samp (62.5 ms) | ~0.2 (deep equilibrium) |
| 0I0XMl3M0E | t=1.15 s | 1192 samp (74.5 ms) | ~0.3 (recently converged from 10k) |
| Fi80N5kW9U | t=1.03 s | 1620 samp (101.3 ms) | ~0.2-0.3 (deep equilibrium) |
| wlAXM0iDgkm | t=3.39 s | 3892 samp (243.3 ms) | ~0.7 (mid-convergence) |
| nVUnxqHLr | t=1.19 s | 404 samp (25.3 ms) | ~2.6 (still converging from 10k) |
| 9xjhiFbGo | t=0.69 s | 256 samp (16.0 ms) | ~19-63 (still high, early in convergence) |

**Observation:** In 6/8 cases, H_error is at or near equilibrium (~0.2–0.8) by the time
`delay_first` fires. The filter ran misaligned for 0.7–1.6 s before delay acquisition,
converging to a wrong-alignment echo path during that window.

### §3.2 H_error response at delay_first — NO reset

The most critical Q1 finding: `delay_first` does **not** reset H_error to 10000.

Example: `nVUnxqHLr` case (delay_first at frame 119, t=1.19 s):

```
frame  t_s   H_mean   d_samp  event
 111   1.11    2.618       -1    -
 ...
 118   1.18    2.610       -1    -
 119   1.19    2.575      404   DF   ← delay_first fires; no H_error reset
 120   1.20    2.575      404    -
 ...
 128   1.28    2.575      404    -   ← flat ~10 frames (reset_filter_derived_state effect)
 129   1.29    2.053      404    -
 130   1.30    1.725      404    -   ← H_error resumes declining normally
```

H_error stays at its pre-event value (2.575) and continues natural decay.
The code path: `_reset_filter_derived_state(reason='delay_first')` resets DTD, EPC,
ERLE, and filter-output-derived state — but NOT `H_error_per_bin`.
`handle_echo_path_change(delay_change=True)` (which does the H_error reset) is
gated by `use_aec3_epc_classification=False` (production default) and never fires.

### §3.3 Contrast: EPV resets H_error to 10000

For comparison, `nVUnxqHLr` also fires EPV at frame 426 (t=4.26 s):

```
frame  t_s   H_mean   d_samp  event
 423   4.23    0.119      404    -
 424   4.24    0.142      404    -
 425   4.25    0.138      404    -
 426   4.26 10000.000     404   EPV  ← H_error reset to 10000 by handle_echo_path_change
 427   4.27   100.000     404    -   ← immediately clipped to H_ERROR_CEIL = 100
 428   4.28   100.000     404    -
 ...
 433   4.33    73.581     404    -   ← slow exponential decay via leakage
 444   4.44    41.986     404    -
```

EPV fires `handle_echo_path_change(delay_change=True)` unconditionally (Phase D
wiring, not gated on `use_aec3_epc_classification`). H_error jumps to 10000,
immediately clips to 100, then decays via leakage × erl.

**H_error response summary by event type:**

| Event | H_error response in production | AEC3 response |
|---|---|---|
| delay_first | **no reset** — H_error continues from pre-event value | `SetConfig(refined_initial, immediate=true)` → leakage_converged = 5e-3/block for 2.5 s |
| delay_shift | **no reset** — slight step change then normal evolution (see §3.4) | `SetConfig(refined_initial, immediate=true)` (same as delay_first) |
| EPV | **reset to 10000 → clip to 100** → slow decay | `SetConfig(refined_initial, immediate=true)` |
| shadow_rise | **reset to 10000 → clip to 100** → slow decay | `SetConfig(refined_initial, immediate=true)` |

### §3.4 delay_shift pattern — wlAXM0iDgkm (3 events)

`wlAXM0iDgkm_farend_singletalk_with_movement` is the only case with `delay_shift`:
3 events at t=7.13, 14.37, 17.99 s; all conf=1.0.

**Event 1 (t=7.13, Δ=−208 samp, delay 3892→3684 samp):**
```
frame  t_s   H_mean   d_samp  event
 712   7.12    1.472     3892    -
 713   7.13    1.362     3684   DS   ← delay_shift; no H_error reset; slight step-down
 714   7.14    0.796     3684    -   ← abrupt drop (not a 10000 reset)
 715   7.15    0.789     3684    -
 ...
 730   7.30    1.008     3684    -   ← recovering toward previous level
```

The H_error drops sharply from 1.47 → 0.796 immediately after the shift.
This is a `_arc_m_q_boost` + `_p_max_override` + `force_delay()` effect from
`_reset_filter_derived_state` — NOT a 10000 reset. The filter's mu/P path changes
but H_error stays near equilibrium.

**Event 2 (t=14.37, Δ=−484 samp):**
Pre-event H_error ≈ 19.1 (elevated, not at floor). After shift: H_error continues
slowly declining from 19.1 → 19.0 → ... (no discontinuity). The elevated H_error
here indicates the filter is still adapting from a previous disturbance.

**Event 3 (t=17.99, Δ=+620 samp):**
Pre-event H_error ≈ 14.1. After shift: drops from 14.0 → 13.7 → 11.5 over ~20
frames. Gradual, not a 10000 reset.

### §3.5 delay_first fires even in static cases

Guard cases without physical movement:
- `nVUnxqHLr0_doubletalk` (DT_static): delay_first at t=1.19 s (delay=404 samp).
  This is expected — every case has an initial delay estimation step.
- `9xjhiFbGo0_farend_singletalk` (FS_static): delay_first at t=0.69 s (delay=256 samp).

Both fire `delay_first` because the BALANCED preset's delay estimator always runs
(default `enable_delay_est=True`). The static guard cases fire `delay_first` just
as often as movement cases.

---

## §4 Q1 Implication Analysis

### §4.1 Evidence class upgrade

Path 1 result (zero hits) was a **structural artifact** of `enable_delay_est=False`
in trace scripts. Path C confirms:

> Real `delay_first` and `delay_shift` events DO occur in production config (8/9 cases
> fire `delay_first`; 1/9 additionally fires 3 `delay_shift`). The delay estimator is
> active and acquires delay within 0.7–3.4 s of case start.

The Q1 evidence class advances from "insufficient — zero real events in cohort" to
"evidence collected — H_error response at real delay events observed."

### §4.2 The functional gap is confirmed and precisely characterised

**AEC3 response to delay change** (delay_first or delay_shift in AEC3 taxonomy):  
→ `SetConfig(refined_initial, immediate=true)`: leakage_converged = 5e-3/block (100× AEC3
steady) for `initial_state_seconds = 2.5 s`; then `SetConfig(refined, immediate=false)`
smoothly returns to 5e-5/block over 1 s.  
Effect: H_error decay rate 100× AEC3 steady for 2.5 s post-event.

**PBFDKF response to delay_first / delay_shift**:  
→ `_reset_filter_derived_state` (DTD / EPC / ERLE / filter-output derived state reset).  
→ H_error_per_bin: **unchanged** (no reset, no leakage burst).  
→ Always-on elevated steady leakage (leakage_converged ≈ 1e-3/block = 20× AEC3 steady,
  5× below AEC3 transient peak).

**PBFDKF response to EPV / shadow_rise**:  
→ `handle_echo_path_change(delay_change=True)`: H_error.fill(10000) → clips to 100.  
→ This IS a reset response, but 100× more aggressive than AEC3's transient leakage bump
  (hard reset to ceiling vs leakage burst from current value).

### §4.3 Asymmetric H_error treatment — delay events vs EPV/shadow_rise

The current production code applies **opposite H_error responses** to different EPC
trigger classes:

| Class | H_error treatment | Aggressiveness |
|---|---|---|
| delay_first, delay_shift | no change | neutral (always-on steady leakage applies) |
| EPV, shadow_rise | hard reset to 10000 → 100 | very aggressive (far beyond AEC3 transient) |

AEC3 applies the **same** transient leakage burst (`SetConfig(refined_initial)`) to
all four classes.

This asymmetry means:
- A physical speaker movement (EPV) → H_error fully reset → slow re-adaptation from ceiling
- A delay-path change (delay_shift) → H_error unchanged → continues adapting from current state

Whether this asymmetry is harmful depends on the operational regime:
- delay_first: filter ran misaligned for ~1–3 s, H_error already at wrong-alignment
  equilibrium → NOT resetting means it re-adapts from wrong-path state. With elevated
  steady leakage, re-adaptation speed is ~5× AEC3 steady but 5× slower than AEC3 transient.
- delay_shift (wlAXM0iDgkm event 2): H_error ≈ 19 (elevated, filter not converged) →
  NOT resetting may actually help if the elevated state reflects genuine non-convergence
  that delay_shift is supposed to address.

### §4.4 Updated Q1 stance (b-revised)

The trace evidence supports closing Q1 on stance **(b-revised)** — "documented functional
equivalence with identified asymmetry":

**(b-revised) Documented functional adaptation with asymmetric EPC response:**
- PBFDKF elevated steady leakage (1e-3/block converged, 1e-1/block diverged) provides
  always-on moderate re-adaptation, equivalent to ~20× AEC3 steady in converged regime
  and ~2× AEC3 steady in diverged regime.
- delay_first / delay_shift: H_error not reset. Re-adaptation proceeds from wrong-path
  equilibrium via always-on elevated steady leakage. Functionally weaker than AEC3
  transient burst for the first 2.5 s post-event.
- EPV / shadow_rise: H_error hard-reset to 10000 → 100. More aggressive than AEC3
  transient in the first few frames, then slower (steady leakage from 100 vs AEC3
  transient 5e-3/block continuing from lower base).
- No two-state machine, no smoothing window, no class-specific burst.

This is NOT stance (a) (strict port) — implementing (a) would require
`use_aec3_epc_classification=True` + `SetConfig(refined_initial)` on all four classes.
This is NOT stance (c) (intentional v3.22 divergence) — the leakage values already exist
and are functional adaptations, not designed deviations.

---

## §5 Conclusion

| Question | Answer |
|---|---|
| Do real delay events exist in production? | YES — 8/9 cases fire `delay_first`; 1/9 fires 3 `delay_shift` |
| When does delay_first fire? | t=0.7–3.4 s after case start (first solid delay estimate) |
| H_error state at delay_first? | Near or at equilibrium in 6/8 cases (0.2–0.8); still converging in 2/8 |
| H_error response to delay_first/shift? | **NO RESET** — H_error unchanged; filter re-adapts from current state |
| H_error response to EPV/shadow_rise? | **HARD RESET** to 10000 → 100 (Phase D wiring, always fires) |
| Functional gap vs AEC3? | AEC3 applies same transient burst to all 4 classes; PBFDKF applies hard reset to EPV/SR only, no burst for delay events |
| Q1 verdict? | **(b-revised)** — documented functional adaptation with asymmetric response; NOT stance (a) implementation |

**Q1 remains OPEN as a parity question.** The evidence now precisely characterises the
gap (no leakage burst on delay_first/shift). Closing Q1 on stance (b-revised) means
documenting this asymmetry as a known functional divergence from AEC3 without code change.
Choosing stance (a) requires adding `use_aec3_epc_classification=True` and wiring
`SetConfig(refined_initial)` to all four EPC trigger classes — this is scope for a
separate, explicitly authorised code change.

**Hard rules (unchanged from design note rev 4):**
- No code change in this audit.
- No 800-case AECMOS.
- Q1 choice between (a)/(b-revised) deferred to explicit user decision.
- Q2/Q3/Q4 remain CLOSED / DEFERRED per design note rev 4.
