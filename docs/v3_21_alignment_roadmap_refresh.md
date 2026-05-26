# v3.21.x AEC3 Alignment Roadmap Refresh

**Date:** 2026-05-25
**Status:** ACTIVE — doc-only; no code, no benchmark
**Production baseline:** v3.21.6 (unchanged)

---

## Context

v3.21.x = AEC3 alignment / parity / documented functional adaptation.
v3.22 = beyond-AEC3 optimisation. Do not migrate AEC3 parity gaps into v3.22.

When a partial port fails, the conclusion is that the partial port was wrong — not that the AEC3 mechanism is wrong.

Prior arc (`use_q1_true_delay_transient_leakage`) tested a leakage-window-only approach (no W reset, no H_error spike, no config switch, no AecState reset). It failed because sustained-lc elevation builds W excess that propagates after the window expires and gets amplified by shadow_rise. This failure does not invalidate the full AEC3 delay-change mechanism. See `docs/v3_21_q1_short_window_final_sanity.md` for the complete closure record.

---

## A. Main Remaining v3.21.x Alignment Work

### A.1 — Full Delay-Change Chain Port

**Status: OPEN — design doc complete; implementation requires authorization.**

Design doc: `docs/v3_21_a1_delay_change_chain_design.md`.

AEC3 performs a full lifecycle reset cascade on `delay_change`. The prior partial
candidate (`use_q1_true_delay_transient_leakage`) omitted every structural component
and is CLOSED. The full chain design is distinct and documented.

**AEC3 chain (confirmed from source — see design doc §1 for exact code refs):**

| Step | AEC3 component | Action | Our port status |
|---|---|---|---|
| 1 | `refined_filters_[ch]->HandleEchoPathChange()` | Tail-zero (`[current,max)` = 1 partition at default) | N/A — structural no-op (Q2 fixed-size closed) |
| 2 | `coarse_filter_[ch]->HandleEchoPathChange()` | Same | N/A |
| 3 | `refined_gains_[ch]->HandleEchoPathChange(delay_change)` | `H_error_.fill(10000)` + `call_counter=0` + `poor_excitation=1000` | Ported, gated OFF — included in `use_full_delay_change_chain` |
| 4 | `coarse_gains_[ch]->HandleEchoPathChange()` | Counter resets (no H_error on coarse) | Ported, gated OFF — included in `use_full_delay_change_chain` |
| 5 | `refined_gains_[ch]->SetConfig(refined_initial, true)` | `leakage_converged` 100×, `leakage_diverged` 10×; transient phase 2.5 s | **Missing** — Q1 design decision (strict port Q1-A or functional adaptation Q1-B) |
| 6 | `coarse_gains_[ch]->SetConfig(coarse_initial, true)` | NLMS `rate` 1.29× for transient | **Missing** — Q1 coarse sub-question |
| 7 | `SetSizePartitions(initial.length_blocks, true)` | Shrink 13→12 partitions | N/A — Q2 CLOSED |
| 8 | `AecState::HandleEchoPathChange` → `_full_reset` | `initial_state.reset()` + ERLE/ERL/FQ reset | Ported for delay_shift; **delay_first gap** — fix in §3.2 |
| 9 | `TransitionTriggered` → `ExitInitialState` | Smooth revert over 1 s; SetConfig(steady, false) | Accessor ported; orchestrator consumer missing — Q3 (follows Q1) |
| 10 | `SuppressionGain::SetInitialState(true/false)` | Sets suppression initial state | Call sites missing — safe to add (A.2: zero effect at default config) |

**Open decisions (see design doc §3):**
- **Q1:** Leakage config split — Q1-A (strict AEC3 2.5 s transient machine) or Q1-B (document existing elevated-steady leakage as functional adaptation). Resolve via H_error trace audit on delay-event cohort.
- **Q3:** ExitInitialState orchestrator hook — required if Q1-A; skipped if Q1-B.
- **delay_first AecState gap:** Fix `_aec3_pending_delay_change` wiring for delay_first.

**Trigger scope:**
- `delay_first` / `delay_shift` ONLY
- NOT shadow_rise, NOT gain_change (AEC3 gain_change branch is a TODO stub), NOT EPV

**Primary pre-AECMOS validation:** xFk7igecuk W ratio at SR+10 < 1.05.
If ≥ 1.05, H_error reset has not taken effect — do not proceed to AECMOS.

**Flag:** `use_full_delay_change_chain` (default OFF; `use_q1_true_delay_transient_leakage` stays CLOSED).

---

### A.2 — SuppressionGain::SetInitialState Audit

**Status: CLOSED — already covered; no parity gap. See `docs/v3_21_a2_suppression_gain_initial_state_audit.md`.**

**Verdict: Already covered. No design change needed.**

`SetInitialState(true/false)` gates two consumers in AEC3 suppression_gain.cc:
1. `GetMinGain`: disables LF smoothing when `initial_state_=true AND lf_smoothing_during_initial_phase=false`.
2. `DominantNearendDetector::Update`: clears NE state when `initial_state_=true AND use_during_initial_phase=false`.

**Key finding:** AEC3's defaults are `lf_smoothing_during_initial_phase=true` and `use_during_initial_phase=true`. Under these defaults, `initial_state_` has **zero behavioral effect** in both AEC3 and our port. Our Python `suppression_gain.py` has `set_initial_state()` implemented with identical conditions; our defaults match AEC3.

**A.1 dependency only:** A.1 implementation (Step 9) will add `set_initial_state(True/False)` call sites at delay_change and ExitInitialState. These calls are mechanically correct and safe to add. No behavioral change under default config — they become meaningful only if `lf_smoothing_during_initial_phase` is later switched to `False`.

**Not a missing parity gap. Not a functional adaptation. Mechanism ported and configs matched.**

**Closure scope guard:** This closure applies under the default AEC3 / BALANCED config only (`lf_smoothing_during_initial_phase=True`, `use_during_initial_phase=True`). If either of these defaults is changed — in AEC3 upstream or in our BALANCED preset — the `initial_state_` gate becomes behaviorally active and A.2 must be reopened to verify parity.

---

### A.3 — Downstream AEC3 State Signals Closure Audit

**Status: CLOSED — no live consumer under production default config for all three signals. See `docs/v3_21_a3_downstream_signals_audit.md`.**

Three AEC3 state signals exist as local variables in `AecState::Update()` and are forwarded only to the TM update path. Under production default config (`transparent_mode_enabled=False`, Legacy TM), none have live downstream consumers.

| Signal | AEC3 consumer | Our status | Verdict |
|---|---|---|---|
| `coarse_filter_converged_relaxed` | Per-channel local → feeds `any_coarse_filter_converged` only; not an independent export | Computed as audit trace; not independently wired (correct) | CLOSED — absorbed by `any_coarse_filter_converged` |
| `any_coarse_filter_converged` | HMM TM only (field trial) — Legacy TM marks it `/* unused */` | Not wired to TM; HMM TM not ported; Legacy TM ignores it (correct) | CLOSED — no live consumer (HMM field trial OFF; Legacy TM ignores signal) |
| `all_filters_diverged` | Legacy TM (default) — used to reset diverged-sequence streak | Proxy wired to Legacy TM via `aec_state.py:333-343`; proxy definition differs from AEC3 strict definition | CLOSED — proxy dormant because `transparent_mode_enabled=False` production default |

**A.1 dependency verdict:** A.1 does **not** create new live consumers. TM is enabled by a construction-time config flag, independent of delay_change. A.1 full delay-change chain sets AecState `initial_state=True` but does not enable or reset TM. Signal consumer status is unchanged post-A.1.

**Closure scope guard:** This closure holds only under production default config (`transparent_mode_enabled=False`, Legacy TM). If either `transparent_mode_enabled=True` OR HMM TransparentMode is ever enabled, reopen A.3: the `all_filters_diverged` proxy (`not any_filter_converged AND divergence > 1.0`) differs from AEC3's strict definition (`min(e2_refined, e2_coarse) > 1.5×y2`) and must be replaced with an AEC3-derived calculation. `any_coarse_filter_converged` would still have no consumer (Legacy TM ignores it; HMM TM not ported).

---

## B. Items Closed — Do Not Pursue

| Item | Status | Reason |
|---|---|---|
| Q1 leakage-window tuning (`use_q1_true_delay_transient_leakage`) | CLOSED no-ship | All parameter variants exhausted; structural failure. See `docs/v3_21_q1_short_window_final_sanity.md`. |
| Q2 dynamic partition sizing | CLOSED — fixed-size functional adaptation | PBFDKF fixed partition is valid PBFDKF adaptation. Only reopen if A.1 full-chain design reveals a specific interaction. |
| SR/EPV terminate guard (`use_q1_terminate_on_non_delay_epc`) | CLOSED — not AEC3 parity | AEC3 has no shadow_rise event. Guard is PBFDKF integration adapter only. Also failed as guard (pre-SR W excess is the seed). Do not pursue. |
| Stationarity default-OFF | CLOSED — documented incompatibility | AEC3 default-ON stationarity not compatible with incomplete PBFDKF detector port. Documented in v3.21.5 safety parity. Do not re-A/B. |
| HMM TransparentMode as standalone candidate | CLOSED — no live consumer under current production config | `transparent_mode_enabled=False` default; HMM state machine has no downstream effect. Only revisit if A.3 audit finds live consumer requiring TM ON. |
| `partition_summed_x2` (v3.21.7) as Cat C fix | CLOSED no-ship | X² source parity correct but Cat C is downstream (state-layer latch), not at filter level. |
| UseRefinedOutput | CLOSED — field-trial substrate | AEC3 `use_coarse_filter_output_` defaults False; reclassified as non-production parity. |
| A.2 noise_gate + A.3 poor_excitation (shadow 5-protection) | CLOSED no-ship | 800-case: 6 FS catastrophic regressions, worst −1.054 dB. |
| Further Q1 window shrinking | STOP — tuning, not AEC3 alignment | The leakage-window approach is inherently wrong for any window length when SR fires in the post-delay era. Tuning parameters does not fix the mechanism. |
| NN residual / beyond-AEC3 | Deferred v3.22 | Out of v3.21.x scope by user rule. |

---

## C. Items Deferred — Cohort Required

| Item | Condition for reopening |
|---|---|
| A.4 narrowband mask for shadow | Only with tonal-rich / narrowband cohort (current cohorts lack tonal dominance; 0 meaningful fire). |
| A.5 saturation gate for shadow | Only with clipped / saturation cohort (0/90 fire on stress cases). |
| W norm-ratio guard during Q1 | Novel mechanism beyond AEC3 scope; requires new explicit authorization + separate design. Not a v3.21.x AEC3 parity item. |
| 800-case bench for any candidate | Only after design + 12-case gate pass. |

---

## Next Concrete Work (In Sequence)

1. **A.1 design — COMPLETE.** See `docs/v3_21_a1_delay_change_chain_design.md`.
   Open decisions requiring resolution before implementation:
   - **Q1:** Strict port (Q1-A) or documented functional adaptation (Q1-B). Resolve via H_error trace audit on a delay-event cohort (not the 12-case, which has zero delay_first/delay_shift fires). Trace `H_error_per_bin` post delay_first vs AEC3 transient leakage profile.
   - **Q3:** ExitInitialState consumer in orchestrator — required if Q1-A, skipped if Q1-B.
   - Neither Q1 nor Q3 reopens A.2 or A.3.

2. **A.1 implementation** — Requires explicit authorization after Q1/Q3 decisions are recorded.
   Minimum steps regardless of Q1: Steps 3–4 (H_error reset), delay_first AecState gap fix (§3.2-A), Step 9 call sites (set_initial_state).
   Additional if Q1-A: Steps 5–6 two-state leakage machine + Q3-A ExitInitialState hook.

---

## Production State

- **v3.21.6:** current production baseline. Unchanged.
- v3.21.7 through v3.21.20: all default-OFF substrate. None shipped to production.
- All v3.21.x research flags: default-OFF. Do not change production preset without explicit authorization + 800-case bench PASS.
- Later v3.21.x research provides evidence and substrate, not production-ready code, unless explicitly shipped with a version bump.
