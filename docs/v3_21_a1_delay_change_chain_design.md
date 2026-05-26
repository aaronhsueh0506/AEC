# v3.21 A.1 — Full Delay-Change Chain Port Design

**Date:** 2026-05-25
**Status:** DESIGN DOC — no code, no benchmark
**Scope:** v3.21.x AEC3 parity. No v3.22 framing.
**Depends on:** A.2 CLOSED, A.3 CLOSED (no interaction under default config)
**Supersedes:** `docs/v3_21_delay_change_chain_design_note.md` (prior draft, rev 4)

---

## 0. Scope and exclusions

**This document produces:** a complete port/adaptation design for the AEC3
delay-change chain, covering all 9 chain steps, with per-step porting status,
design options, risk, and a smallest-gate-first validation plan.

**This document does NOT:** implement any code, run any benchmark, reopen
A.2/A.3, or authorize any 800-case run.

**Excluded explicitly:**
- `use_q1_true_delay_transient_leakage` (leakage-window-only, no W/H_error reset)
  — CLOSED no-ship, all parameter variants exhausted. See
  `docs/v3_21_q1_short_window_final_sanity.md`. This candidate omitted every
  structural element of the AEC3 chain and is architecturally orthogonal to the
  design below.
- v3.22 / beyond-AEC3 work — out of scope per version rule.
- Q2 (dynamic partition sizing) — CLOSED fixed-size functional adaptation
  (see §4.5). Do not reopen.
- A.2 and A.3 — CLOSED under default config. A.1 Step 9 adds set_initial_state
  call sites per A.2 closure (zero behavioral effect at default). A.3 TM signal
  consumers remain dormant. Neither closure is invalidated by this design.

---

## 1. AEC3 source chain — complete reference

### 1.1 Top-level dispatch (`echo_remover.cc:385-421`)

```
if (echo_path_variability.AudioPathChanged()):
    subtractor_.HandleEchoPathChange(echo_path_variability)     ← §1.2
    aec_state_.HandleEchoPathChange(echo_path_variability)      ← §1.3
    if (delay_change != kNone):
        suppression_gain_.SetInitialState(true)                 ← §1.4 Step 9

if (aec_state_.TransitionTriggered()):                          ← §1.5
    subtractor_.ExitInitialState()                              ← §1.5
    suppression_gain_.SetInitialState(false)                    ← §1.5 Step 9
```

**Trigger scope (confirmed):** `delay_change != kNone` includes
`EchoPathVariability::DelayAdjustment::NEW_DETECTED_DELAY` and any non-None
variant. In our pipeline, this maps to `delay_first` and `delay_shift` only.
`gain_change`, EPV, and `shadow_rise` are excluded — they route through separate
branches (gain_change fires only `refined_gains_->HandleEchoPathChange` with no
H_error reset; EPV/shadow_rise have no AEC3 equivalent).

### 1.2 `Subtractor::HandleEchoPathChange::full_reset` (`subtractor.cc:148-175`)

Fires per capture channel in this order, only on `delay_change != kNone`:

| Step | AEC3 call | Effect | Config constants |
|---:|---|---|---|
| 1 | `refined_filters_[ch]->HandleEchoPathChange()` | `ZeroFilter(current_size_partitions_, max_size_partitions_, &H_)` — zeros partitions `[current, max)` only (tail zero) | — |
| 2 | `coarse_filter_[ch]->HandleEchoPathChange()` | Same tail-zero for coarse filter | — |
| 3 | `refined_gains_[ch]->HandleEchoPathChange(delay_change)` | `H_error_.fill(kHErrorInitial)` (`= 10000.f`); counter reset (see below) | `kHErrorInitial=10000`, `kPoorExcitationCounterInitial=1000` |
| 4 | `coarse_gains_[ch]->HandleEchoPathChange()` | Counter reset only; **no H_error on coarse** | — |
| 5 | `refined_gains_[ch]->SetConfig(refined_initial, true)` | Switch refined gain to transient config (`immediate=true` → no smoothing); `leakage_converged`: 5e-5→**5e-3** (100×), `leakage_diverged`: 5e-2→**5e-1** (10×) | `refined_initial.leakage_converged=5e-3`, `refined_initial.leakage_diverged=5e-1`, `refined.leakage_converged=5e-5`, `refined.leakage_diverged=5e-2` |
| 6 | `coarse_gains_[ch]->SetConfig(coarse_initial, true)` | Switch coarse gain to transient config (`immediate=true`); `rate`: 0.7→**0.9** (1.29× NLMS step) | `coarse_initial.rate=0.9`, `coarse.rate=0.7` |
| 7 | `refined_filters_[ch]->SetSizePartitions(refined_initial.length_blocks, true)` | Shrink refined filter from 13 → 12 partitions (`immediate=true`); zeros partition 12 | `refined_initial.length_blocks=12`, `refined.length_blocks=13` |
| 8 | `coarse_filter_[ch]->SetSizePartitions(coarse_initial.length_blocks, true)` | Same for coarse | same defaults |

**Counter semantics in Step 3** (`refined_filter_update_gain.cc:53-68`):

```cpp
if (delay_change != kNone):  H_error_.fill(10000)
if (!gain_change):
    poor_excitation_counter_ = kPoorExcitationCounterInitial  // = 1000 blocks = 4 s
    call_counter_ = 0
```

The skip-update gate fires when `call_counter_ <= size_partitions` (13 blocks = 52 ms)
OR `poor_excitation_counter_ < size_partitions`. At HEPC, both gates arm — the
52 ms gate is the effective filter freeze period.

**Step 5 leakage config — `UpdateCurrentConfig()` mechanics**
(`refined_filter_update_gain.cc:143-174`):

`SetConfig(target, immediate)` sets `target_config_` and
`config_change_duration_blocks_`. If `immediate=true`, `config_change_counter_=0`
→ `UpdateCurrentConfig` applies target instantly. If `immediate=false`,
`config_change_counter_ = config_change_duration_blocks_ = 250` (default = 1000 ms)
→ linear blend from `old_target_config_` to `target_config_` over 250 blocks.

H_error update equation (every Compute() block, after the gain is applied):
```cpp
H_error_[k] += current_config_.leakage_converged * erl[k];  // if E²_refined ≤ E²_coarse
H_error_[k] += current_config_.leakage_diverged * erl[k];   // if E²_refined > E²_coarse
H_error_[k] = clamp(H_error_[k], error_floor, error_ceil)
```
Under transient config: leakage values are 100×/10× larger → H_error rebuilds
faster → Kalman gain stays large longer → faster re-tracking of new delay path.

### 1.3 `AecState::HandleEchoPathChange::full_reset` (`aec_state.cc:143-171`)

Fires on `delay_change != kNone`:

| Step | AEC3 call | Effect |
|---:|---|---|
| 6a | `filter_analyzer_.Reset()` | Resets filter misadjustment analyzer |
| 6b | `capture_signal_saturation_ = false` | Clears saturation latch |
| 6c | `strong_not_saturated_render_blocks_ = 0` | Resets the gate counter that drives TransitionTriggered |
| 6d | `blocks_with_active_render_ = 0` | Resets active-render accumulator |
| 6e | `initial_state_.Reset()` | Re-enters initial state; makes `TransitionTriggered` fire again when the convergence threshold is next crossed |
| 6f | `transparent_state_->Reset()` | Resets TM state (dormant under TM=OFF default) |
| 6g | `erle_estimator_.Reset(true)` | Full ERLE reset (delay_change=True resets subbands fully) |
| 6h | `erl_estimator_.Reset()` | Resets ERL estimator |
| 6i | `filter_quality_state_.Reset()` | Resets FilteringQualityAnalyzer |

### 1.4 `suppression_gain_.SetInitialState(true)` (`echo_remover.cc:404-407`)

Called on `delay_change != kNone` only (not on `gain_change`). Sets
`initial_state_ = true` on SuppressionGain. Under AEC3 default config
(`lf_smoothing_during_initial_phase=True`, `use_during_initial_phase=True`),
this has **zero behavioral effect** — both consumers of `initial_state_` are
gated by their respective defaults. See A.2 closure.

### 1.5 `TransitionTriggered` → `ExitInitialState` (`echo_remover.cc:418-421`, `subtractor.cc:177-186`)

`aec_state_.TransitionTriggered()` fires as a one-shot edge signal when
`strong_not_saturated_render_blocks_` crosses `initial_state_seconds × kNumBlocksPerSecond`
= 2.5 s × 250 = **625 blocks** (default; conservative mode doubles to 5 s).

When triggered, `subtractor_.ExitInitialState()` runs per channel:

| Step | AEC3 call | Effect |
|---:|---|---|
| 8a | `refined_gains_[ch]->SetConfig(refined, false)` | Revert to steady refined config; `immediate=false` → linear blend over 250 blocks = **1000 ms** back to `leakage_converged=5e-5, leakage_diverged=5e-2` |
| 8b | `coarse_gains_[ch]->SetConfig(coarse, false)` | Revert coarse to `rate=0.7`; same 1 s smoothing |
| 8c | `refined_filters_[ch]->SetSizePartitions(refined.length_blocks, false)` | Grow refined from 12 → 13 partitions; `immediate=false` → smooth |
| 8d | `coarse_filter_[ch]->SetSizePartitions(coarse.length_blocks, false)` | Same for coarse |

Then `suppression_gain_.SetInitialState(false)` — Step 9, A.2 closure (zero
behavioral effect at default config).

---

## 2. Our current production behavior — per step

Production baseline: v3.21.6. Key flag: `use_aec3_handle_echo_path_change = False`
(default). This flag gates the filter H_error reset and shadow filter counter
reset on delay events.

| Chain step | AEC3 action | Our production state (v3.21.6) | Status |
|---:|---|---|---|
| 1 — Refined W tail-zero | `ZeroFilter(current, max, &H_)` — zeros partition `[12, 13)` | No `_active_size_partitions` var; fixed `n_partitions=6`. ZeroFilter(6,6) would zero 0 partitions. Q2 CLOSED. | **N/A — structural no-op at fixed-size** |
| 2 — Coarse W tail-zero | Same | Same | **N/A — same** |
| 3 — Refined H_error reset | `H_error_.fill(10000)` + counter reset (`call_counter=0`, `poor_excitation=1000`) | `PBFDKF.handle_echo_path_change(delay_change=True)` exists and does `H_error_per_bin.fill(10000)`. BUT: gated by `use_aec3_handle_echo_path_change=False` in production. Counter resets also behind the flag at `delay_first`/`delay_shift`. | **PARTIALLY PORTED — gated OFF in production** |
| 4 — Coarse counter reset | `coarse_gains_->HandleEchoPathChange()` → counter reset | `PBFDAF.handle_echo_path_change(delay_change=True)` exists. Same gating as Step 3. | **PARTIALLY PORTED — gated OFF in production** |
| 5 — SetConfig(refined_initial) | Switch `leakage_converged` 100×, `leakage_diverged` 10× for 2.5 s | **NOT IMPLEMENTED.** PBFDKF has no two-state leakage machine. Always-on elevated steady leakage (`leakage_converged ≈ 1e-3` = 20× AEC3 steady; `leakage_diverged ≈ 1e-1` = 2× AEC3 steady). The Q1 open design question. | **MISSING — Q1 design decision** |
| 6 — SetConfig(coarse_initial) | Switch coarse `rate` 1.29× for 2.5 s | **NOT IMPLEMENTED.** Shadow PBFDAF has single fixed `mu` (no `_p_max_override` on PBFDAF — the `isinstance(filt, PBFDKF)` guard skips PBFDAF everywhere). No coarse transient rate split. | **MISSING — Q1 sub-question (coarse side)** |
| AecState §1.3 steps 6a–6i | Full AecState reset | Our `AecState._full_reset()` mirrors all 9 sub-steps (`aec_state.py:360-374`). Called on delay_shift via `_aec3_pending_delay_change`. **delay_first status: `_aec3_pending_delay_change` is NOT set on delay_first** (delay_first calls `_reset_filter_derived_state` only; `_aec3_pending_delay_change` assignment at `orchestrator.py:1951` is in the delay_shift branch). | **PARTIAL — delay_shift: PORTED; delay_first: MISSING** |
| Step 9 ON — SetInitialState(true) | `suppression_gain_.SetInitialState(true)` on delay_change | Not called. A.2 closure: zero behavioral effect at default config. Call site belongs to A.1 Step 9. | **MISSING CALL SITE — A.2 verified safe to add** |
| §1.5 — TransitionTriggered check | `aec_state_.TransitionTriggered()` → ExitInitialState | `aec_state.transition_triggered()` accessor exists in our port (`aec_state.py:165-166`) and `InitialState.transition_triggered()` fires correctly. **But:** no consumer in orchestrator. `_aec3_post` does NOT check `transition_triggered()` or call ExitInitialState or exit_leakage_config. | **MISSING CONSUMER — Q3 design decision** |
| Step 9 OFF — SetInitialState(false) | `suppression_gain_.SetInitialState(false)` on TransitionTriggered | Not called. Same A.2 note. | **MISSING CALL SITE — coupled to §1.5 consumer** |
| ExitInitialState steps 8a–8d | Revert leakage configs + sizes over 1 s | **NOT IMPLEMENTED.** No two-state leakage machine → nothing to revert. If Q1 strict port is chosen, ExitInitialState must call `PBFDKF.set_leakage_config(steady, immediate=False)` + analogous on PBFDAF. | **MISSING — conditional on Q1 decision** |

**Summary of gaps:**
- Steps 1–2: N/A (Q2 CLOSED, structural no-op)
- Steps 3–4: ported but gated OFF (`use_aec3_handle_echo_path_change=False`); enabling them is part of `use_full_delay_change_chain`
- Steps 5–6: missing (Q1 design decision)
- AecState §1.3: delay_shift ported; **delay_first AecState full_reset missing**
- ExitInitialState: accessor exists; orchestrator consumer missing (Q3, couples to Q1)
- Step 9 call sites: missing, safe to add (A.2 closure)

---

## 3. Design options per component

### 3.1 Steps 3–4 — H_error reset + counter reset

**Status:** Mechanism exists under flag. Include unconditionally in `use_full_delay_change_chain`.

| Option | Description | Risk |
|---|---|---|
| **3.1-A (Recommended)** | Include H_error reset + counter reset in `use_full_delay_change_chain` scope. Both refined (H_error=10000, call_counter=0, poor_excitation=400) and shadow (call_counter=0, poor_excitation=400) fire on delay_first/delay_shift. This consolidates `use_aec3_handle_echo_path_change` behavior into the new flag. Old flag can be retired or left as a no-op. | Low. H_error reset is the primary pre-AECMOS gate (xFk7igecuk W ratio). Already validated structurally in v3.21 Phase D trace. |
| 3.1-B | Keep separate flag (`use_aec3_handle_echo_path_change` gates H_error; `use_full_delay_change_chain` adds Steps 5–9). | Adds combinatorial test burden. Not recommended. |

**Chosen:** 3.1-A. `use_full_delay_change_chain=True` implies H_error reset + counter reset.

### 3.2 AecState delay_first gap

**Status:** `_aec3_pending_delay_change` is only set on delay_shift. On delay_first,
`_reset_filter_derived_state` runs but AecState._full_reset does not fire.

**Impact:** Without AecState._full_reset on delay_first, `initial_state_.reset()` does
not fire → `strong_not_saturated_render_blocks_` is not cleared → `TransitionTriggered`
may never re-fire (it already fired at stream start and won't fire again unless reset).
This also means ERLE/ERL/FilterQuality are not reset on delay_first — a real parity gap.

| Option | Description | Risk |
|---|---|---|
| **3.2-A (Recommended)** | Add `_aec3_pending_delay_change = _DA.NEW_DETECTED_DELAY` to the delay_first branch (mirroring delay_shift). Under `use_full_delay_change_chain`, this queues AecState._full_reset for the next `_aec3_post` frame on delay_first. | Low. `_full_reset` code path is already validated for delay_shift. |
| 3.2-B | Handle inline in `_aec3_post` with a separate pending flag for delay_first. | Same outcome, more flags. Not recommended. |

**Chosen:** 3.2-A. Wire `_aec3_pending_delay_change` for delay_first under the new flag.

### 3.3 Q1 — Steps 5–6 leakage config split (the central design question)

**The AEC3 mechanism:** On delay_change, both refined and coarse filter gains switch
to a transient config — refined `leakage_converged` 100× larger, `leakage_diverged`
10× larger; coarse `rate` 1.29× larger. These hold for `initial_state_seconds=2.5 s`
then revert linearly over `config_change_duration_blocks=1 s` on ExitInitialState.

**Our current state:** PBFDKF runs always-elevated steady-state leakage
(`leakage_converged ≈ 1e-3` = 20× AEC3 steady = 5× below AEC3 transient;
`leakage_diverged ≈ 1e-1` = 2× AEC3 steady = 5× below AEC3 transient). No two-state
machine, no transient bound, no smoothing window. Shadow PBFDAF has a single fixed `mu`.

**`_p_max_override` / `_p_floor_beta` / `_arc_m_q_boost` are NOT Q1 adapters** —
they operate on the legacy P matrix (demoted diagnostic state, `filters.py:411-414`),
not on `H_error_per_bin` or `leakage_*`. See `docs/v3_21_q1_mapping_audit.md`.

| Option | Description | Risk | Validation required |
|---|---|---|---|
| **Q1-A — Strict AEC3 port** | Introduce two-state leakage config. On delay_first/delay_shift: `PBFDKF.set_leakage_config('transient', immediate=True)` → sets `_leakage_converged=5e-3`, `_leakage_diverged=5e-1` (AEC3 transient values). Shadow PBFDAF: `mu_scale=1.29×` for transient phase. On TransitionTriggered: `PBFDKF.set_leakage_config('steady', immediate=False)` → linear blend over 250 hops (~1000 ms) to `leakage_converged=5e-5`, `leakage_diverged=5e-2`. Requires Q3-A (ExitInitialState hook). Code surface ~50 lines. **This is the strict parity path.** | Medium. Changing post-HEPC Kalman gain rebuild rate on delay events affects tracking. Must pass xFk7igecuk W ratio gate and 12-case before 800-case. | Trace H_error_per_bin trajectory vs AEC3 reference profile post-delay_first. xFk7igecuk W ratio gate. 12-case AECMOS. 800-case. |
| **Q1-B — Documented functional adaptation (current code state)** | Document PBFDKF's always-elevated steady leakage (`aec3_scale.py:LEAKAGE_*_PER_HOP_DEFAULT`, `filters.py:424-429`) as the v3.21.x functional adaptation for the AEC3 transient config concept. Caveat: (i) no two-state machine, no 2.5 s transient bound, no 1 s smoothing window; (ii) per-block values biased toward AEC3 transient (20× steady, 5× below AEC3 transient); (iii) always-on means every frame is in "transient mode" relative to AEC3 steady, but without a reset event the benefit doesn't concentrate around delay events. **Doc-only, no code.** | None (doc). | Trace H_error_per_bin trajectory post-delay_first: verify always-elevated profile produces comparable per-frame Kalman gain to AEC3 2.5 s transient profile. If comparable: closure. If not: switch to Q1-A. |
| Q1-C — Intentional divergence | Explicitly declare elevated steady leakage as PBFDKF-specific behavior beyond AEC3 with separate justification (e.g. PBFDKF Kalman dynamics differ from NLMS and require continuously higher leakage). Framed as v3.21.x documented divergence, not a parity gap. | None (doc). Requires explicit user authorization to frame as intentional divergence. | None additional. |

**Recommendation:** Begin with Q1-B (trace audit) to determine empirically whether the
always-elevated steady leakage provides functional equivalence. If the trace shows the
post-delay H_error trajectory is materially worse than AEC3 transient, implement Q1-A.
Decision gate: **xFk7igecuk W ratio at SR+10 < 1.05** — if ratio ≥ 1.05, the H_error
reset alone did not take effect; Q1-B is insufficient and Q1-A is required.

**Q1 coarse side (Step 6, shadow PBFDAF rate):** Follows the Q1 decision.
Under Q1-A: add transient `mu_scale=1.29×` on shadow PBFDAF for the 2.5 s window,
reverted on TransitionTriggered. Under Q1-B: document current fixed `mu` as functional
adaptation (PBFDAF is NLMS — step size is already fresh every frame regardless of
filter age; the AEC3 rate boost may not transfer to our PBFDAF benefit structure).

### 3.4 Q3 — ExitInitialState orchestrator hook

**Status:** `aec_state.transition_triggered()` accessor is correctly implemented
(`aec_state.py:165-166`, `initial_state.py:60-61`). No orchestrator consumer.

| Option | Description | Risk | Couples to |
|---|---|---|---|
| **Q3-A — Add orchestrator consumer** | In `_aec3_post`, after `aec_state.update()`, check `self._aec3_state.transition_triggered()`. If True: call `PBFDKF.exit_leakage_config(immediate=False)` (start 1 s smooth revert) + `PBFDAF.exit_transient_rate(immediate=False)` + `suppression_gain.set_initial_state(False)`. | Low if Q1-A chosen and two-state machine implemented. Otherwise no-op. | **Required if Q1-A chosen.** |
| **Q3-B — No consumer (doc-only)** | If Q1-B or Q1-C chosen: no transient bound exists to exit. Add comment in `_aec3_post` noting the transition_triggered accessor is available for future use. | None. | **Default if Q1-B/Q1-C chosen.** |
| Q3-C — Audit-only accessor in orchestrator | Add a trace-only diag emit for `transition_triggered` state, no action. ~3 lines. | None. | Optional regardless of Q1 decision. |

### 3.5 Steps 1–2 and 7–8 — W tail-zero and SetSizePartitions

**Status:** Q2 CLOSED — documented fixed-size functional adaptation.

At AEC3 default, tail-zero (`ZeroFilter(12, 13)`) zeros exactly 1 partition.
Our fixed-size architecture has no tail beyond `n_partitions=6`. The operation
is a structural no-op. SetSizePartitions (12→13 grow on ExitInitialState) is
also not applicable.

**Disposition:** No implementation. No design work. Closure rationale:
(a) PBFDKF Kalman P-matrix shrink/restore is lossy; (b) `H_error_per_bin.fill(10000)`
already provides per-bin fast-adaptation independently of partition count; (c) AEC3
default shrinks by 1 partition (8%) with 1 s smoothing — disproportionate porting cost
for small magnitude. Documented here as the canonical closure reference.

**`W.fill(0)` (full W zero, non-AEC3):** Current `handle_echo_path_change(zero_filter=True)`
exists and zeros the entire W array. AEC3's equivalent zeros only 1 partition.
`W.fill(0)` remains a non-AEC3 default-OFF substrate — do not include in
`use_full_delay_change_chain` unless separately justified.

### 3.6 Step 9 — SuppressionGain::SetInitialState call sites

**Status:** `suppression_gain.set_initial_state()` method fully ported
(`suppression_gain.py:529-530`). Orchestrator never calls it. A.2 closure:
zero behavioral effect at default config (`lf_smoothing_during_initial_phase=True`,
`use_during_initial_phase=True`).

**Action:** Add two call sites as part of `use_full_delay_change_chain`:
1. `self._aec3_sg.set_initial_state(True)` — in `_aec3_post` when delay_change fires
   (same frame that queues AecState._full_reset)
2. `self._aec3_sg.set_initial_state(False)` — in `_aec3_post` when `transition_triggered()`
   fires (Q3-A path)

Risk: None under default config (A.2 verified). Calls are mechanically correct
per AEC3 source (`echo_remover.cc:406, 420`). Add regardless of Q1 decision
(the calls are safe and complete the parity record).

---

## 4. Flag design

### 4.1 Flag name and scope

```
use_full_delay_change_chain: bool = False   # config.py
```

When `True`, enables on `delay_first` and `delay_shift` only:
1. `PBFDKF.handle_echo_path_change(delay_change=True)` → H_error=10000, call_counter=0, poor_excitation=400
2. `PBFDAF.handle_echo_path_change(delay_change=True)` → counter resets
3. `_aec3_pending_delay_change = _DA.NEW_DETECTED_DELAY` for both delay_first and delay_shift (fixes delay_first AecState gap)
4. `aec3_sg.set_initial_state(True)` — Step 9
5. *(Q1-A only)* Switch PBFDKF to transient leakage profile; switch PBFDAF `mu_scale=1.29×`
6. *(Q3-A only, requires Q1-A)* Arm ExitInitialState check in `_aec3_post`; on `transition_triggered()`: revert leakage/rate + `aec3_sg.set_initial_state(False)`

Items 1–4 are implementation-independent of Q1. Items 5–6 depend on the Q1 decision.

### 4.2 Relationship to existing flags

| Existing flag | Relationship to new flag |
|---|---|
| `use_aec3_handle_echo_path_change` (default OFF) | Overlaps with Steps 3–4 above. Under `use_full_delay_change_chain=True`, the H_error reset is included. The old flag can be treated as absorbed; it need not be independently enabled. |
| `use_q1_true_delay_transient_leakage` (default OFF) | CLOSED, mutually exclusive. Both must not be True simultaneously (they have different W-state semantics). Add assertion or doc note. |
| `use_aec3_zero_filter_on_epc` / M2 W.fill(0) (default OFF) | Independent. The full chain does NOT include `W.fill(0)` — AEC3's equivalent is tail-zero (1 partition, structural no-op in our architecture). Keep separate default-OFF substrate. |

### 4.3 Trigger scope guard

The flag must fire on `delay_first` and `delay_shift` **only**. EPV, shadow_rise,
and gain_change must be explicitly excluded:
- EPV/shadow_rise: PBFDKF-specific events; AEC3 has no equivalent; do not treat as delay_change
- gain_change: AEC3's gain_change branch in `RefinedFilterUpdateGain::HandleEchoPathChange`
  is a TODO stub — no H_error reset, no SetConfig(initial). gain_change fires
  `AecState.erle_estimator_.Reset(false)` only (partial reset). Do not include in full chain.

### 4.4 Pre-implementation guards

The following guards apply to every validation run (Gate 0/1/2). They must be
verified before any ON-variant trace or AECMOS result is interpreted as evidence
for or against A.1.

**Guard 1 — Flag isolation**

`use_full_delay_change_chain` is the only flag allowed in a non-default state during
Gate 0/1/2 validation. The following flags must be confirmed OFF before any A.1 result
is interpreted:

- `use_q1_true_delay_transient_leakage` — CLOSED; structurally incompatible (no W/H_error reset)
- `use_q1_terminate_on_non_delay_epc` — CLOSED; SR/EPV guard, not AEC3 parity
- `use_aec3_zero_filter_on_epc` and any W.fill(0) substrate flags — independent, default OFF
- Any other prior Q1 leakage-window or SR/EPV guard flags

If any of the above are ON during a Gate 0/1/2 run, the run is **invalid** and must not
be used to judge A.1.

**Guard 2 — Gate 0 byte-equal precheck**

Before any ON-variant trace (`use_full_delay_change_chain=True`) is interpreted, the
OFF variant (`use_full_delay_change_chain=False`) must be byte-equal to the current
production baseline (anchor: `docs/bench/v3_21_3aadd2d_baseline/`).

Procedure: `python3 python/check_byte_equal.py` with current config. Must report
`=== 25/25 PASS, 0 FAIL ===`. If any case fails, stop and debug harness/config — a
failing precheck means the baseline is already diverged from production and any Δ will
misattribute the source.

**Guard 3 — AecState delay_first confirmation**

Gate 0 must explicitly confirm that under `use_full_delay_change_chain=True`, the
delay_first event triggers the same AecState full_reset state changes as delay_shift.
Required trace fields (per-frame, xFk7igecuk case):

- `initial_state_active()` transitions False → True at the frame immediately following
  delay_first (the first `_aec3_post` frame that resolves `_aec3_pending_delay_change`)
- SubbandErleEstimator reset and FilterQuality reset diag counters both fire at that frame
- `transition_triggered()` fires exactly once after sufficient strong render accumulates

If `initial_state_active()` does not transition to True, the delay_first AecState
wiring (§3.2 option 3.2-A) is broken. Fix before continuing Gate 0.

**Guard 4 — Strict trigger scope (runtime confirmation)**

The chain must NOT fire on EPV, shadow_rise, or gain_change events. Confirm via
per-frame diag on a case that exercises these events without a preceding delay_first
or delay_shift:

- EPV event fires (`echo_path_variability_fired` diag = True) but no delay_first/delay_shift
  in the same or adjacent frames → no H_error reset, no AecState._full_reset, no
  `set_initial_state(True)` emitted
- Shadow_rise event fires (regime handler `shadow_rise_event` diag = True) but no
  delay_first/delay_shift → same: no chain fire
- If the chain fires on either of the above, it is a trigger scope bug; fix before Gate 0
  measurement proceeds.

---

## 5. Validation plan — smallest gate first

### Gate 0 — Structural correctness trace (no benchmark)

Before any AECMOS measurement:

**Target case:** xFk7igecuk (delay_first at frame 104, SR at frame 183).
**Variants:**
- A: baseline OFF (`use_full_delay_change_chain=False`)
- B: full chain ON (`use_full_delay_change_chain=True`)

**Per-frame trace fields to collect:**
- `H_error_per_bin` mean and max immediately post delay_first (frames 104–110) for
  both main (PBFDKF) and shadow (PBFDAF)
- `_aec3_pending_delay_change` resolution frame (confirm AecState._full_reset fires)
- `aec_state.initial_state_active()` for frames 104–500 (confirm initial_state=True post-delay_first, then falls after TransitionTriggered if Q3-A implemented)
- `aec_state.transition_triggered()` — confirm fires once after convergence
- W_norm per frame, frames 104–200 (the SR+10 window)
- `_leakage_converged` / `_leakage_diverged` in effect (if Q1-A: should switch to transient values)

**Primary gate (from roadmap):**
W_B / W_A ratio at SR+10 (frame 193) **< 1.05**.
- If ratio ≥ 1.05: H_error reset has not taken effect, or elevated leakage is creating W divergence. Investigate before proceeding.
- If ratio < 1.05: proceed to Gate 1.

**Why this gate:** The leakage-window approach produced ratios of 1.137–1.169 at SR+10.
The full AEC3 chain resets H_error=10000 for both ON and OFF variants of the filter,
causing identical large initial Kalman gain → no systematic W divergence builds → ratio
should be near 1.000. Any ratio ≥ 1.05 indicates the chain is not functioning as designed.

**H_error trace check:** At frame 105 (first hop after delay_first), B's
H_error_per_bin should be ≈ 10000 everywhere. A's should be ≈ whatever the
pre-delay-first converged value was (expected: small, < 1).

**AecState check:** B's `initial_state_active()` should be True from frame ~106
(first `_aec3_post` frame with the pending delay_change) onwards. A's may or may
not be True (depends on stream position at delay_first).

### Gate 1 — 12-case AECMOS

Only after Gate 0 passes.

**Standard 12-case cohort** (same as prior v3.21.x candidates).
**Variants:** A (baseline) vs B (full chain ON).
**Criteria:**
1. No per-case Δdeg < −0.05 on any DT case (G1)
2. No directional catastrophic: Δdeg < −0.20 on DT or Δecho > +0.20 on FS (G2)
3. FS bucket mean Δecho ≤ +0.05 (G3)
4. DT_static bucket mean Δdeg ≥ 0.00 (G4)
5. Cat C stress cases (nVUnxqHLr, XRTnTUjU) non-regress: Δdeg ≥ −0.05 each (G5)

If G1–G5 all pass: proceed to Gate 2.
If any fail: trace the failing case per the Phase A protocol from the v3.21.20
plan — find the actual bug before changing the design.

### Gate 2 — 800-case bench

Only after Gate 1 passes. Standard config: `preset=balanced / filter=832 (52 ms) / --cng / --parallel / --workers 4`.

**Hard bars:**
- DT_static Δdeg ≥ −0.010 (mean)
- FS_static Δecho ≤ +0.030 (mean)
- No Cat C catastrophic: no per-case Δdeg < −0.30 on DT_static
- Pareto comparison: beats v3.21.6 on at least one of (deg, echo) without regressing the other by more than 0.010

If Gate 2 passes: A.1 is implementation-complete; bump `__version__` and CHANGELOG.
If fails: return to design with 800-case per-case trace.

---

## 6. Risk table

| Component | Option | Code surface | Production risk | Gate |
|---|---|---|---|---|
| Steps 3–4 (H_error + counter reset) | 3.1-A | 5 lines (wire delay_first into new flag; remove old flag guard) | Low — already validated in Phase D trace | Gate 0 |
| AecState delay_first gap | 3.2-A | 2 lines (`_aec3_pending_delay_change` in delay_first branch) | Low — mirrors existing delay_shift behavior | Gate 0 |
| Step 9 call sites (set_initial_state) | Always | 4 lines (2 True, 2 False) | Zero — A.2 verified no behavioral effect at default | Gate 0 check |
| Q1-B (functional adaptation doc) | No code | 0 lines | None | Trace audit only |
| Q1-A (strict leakage config port) | New | ~50 lines | Medium — changes Kalman gain rebuild rate post-HEPC | Gate 0 → Gate 1 → Gate 2 |
| Q3-A (ExitInitialState hook) | Conditional on Q1-A | ~20 lines | Low if Q1-A is correct | Gate 0 (transition_triggered() fires correctly) |
| Q3-B (no consumer) | No code | 0 lines | None | N/A |

---

## 7. Non-goals and hard rules

- **No implementation until design is authorized.** This document is design only.
- **No 800-case bench until Gate 1 (12-case) passes.**
- **No reopening A.2 or A.3** — neither closure guard is triggered by this design.
  Q1-A does not change `lf_smoothing_during_initial_phase` or `use_during_initial_phase`.
  Q1-A does not enable TM.
- **`use_q1_true_delay_transient_leakage` stays CLOSED** — the leakage-window approach
  is architecturally incompatible with the full chain (no W/H_error reset, no config
  switch). Its failure does not predict the full chain's behavior.
- **W.fill(0) excluded from full chain** — AEC3's tail-zero is 1 partition (structural
  no-op in our architecture). Full-W zero is a non-AEC3 substrate; do not include.
- **gain_change excluded from trigger scope** — AEC3 gain_change branch is a TODO stub;
  including it would over-apply the chain.
- **EPV and shadow_rise excluded** — PBFDKF-specific events; no AEC3 equivalent.

---

## 8. Resolved questions (from prior design note)

| Q | Resolution | Authority |
|---|---|---|
| Q2 (dynamic partition sizing) | CLOSED — fixed-size functional adaptation. H_error reset covers per-bin fast-adaptation without partition change. AEC3 default is 1-partition shrink, small magnitude, costly to port. | `docs/v3_21_delay_change_chain_design_note.md` §2.5 rev 4; §4.2 here |
| Q4 (W.fill(0) vs tail-zero) | CLOSED — follows Q2. Tail-zero is structural no-op at fixed partitions. W.fill(0) stays default-OFF non-AEC3 substrate. | `docs/v3_21_delay_change_chain_design_note.md` §3.4; §3.5 here |
| Q3 (ExitInitialState) | Deferred to Q1 decision. Q3-A if Q1-A chosen; Q3-B if Q1-B/Q1-C. | §3.4, §4.1 here |
| Q1 (transient leakage config split) | OPEN — two options: Q1-A (strict port) or Q1-B (documented functional adaptation). Resolve via H_error trace audit on delay-event cohort before implementation. | §3.3 here |
| A.2 (set_initial_state consumers) | CLOSED — zero behavioral effect at default. Add Step 9 call sites as part of full chain (Step 9 action). | `docs/v3_21_a2_suppression_gain_initial_state_audit.md` |
| A.3 (downstream TM signals) | CLOSED — no live consumer under TM=OFF. A.1 does not create new consumers. | `docs/v3_21_a3_downstream_signals_audit.md` |
| AecState delay_first gap | Newly identified here. Fix via 3.2-A. | §2 mapping table, §3.2 |

---

## 9. Files relevant to implementation (when authorized)

Read before touching any code:
- `python/modules/orchestrator.py:1860-1970` — delay_first and delay_shift trigger sites
- `python/modules/orchestrator.py:3400-3480` — `_aec3_post` call frame (AecState update + diag)
- `python/modules/filters.py:482-506` — `PBFDKF.handle_echo_path_change`
- `python/modules/filters.py:167-193` — `PBFDAF.handle_echo_path_change`
- `python/modules/state/aec_state.py:219-230` — `handle_echo_path_change` dispatch
- `python/modules/state/aec_state.py:360-374` — `_full_reset`
- `python/modules/state/initial_state.py` — `InitialState.reset()` + `transition_triggered()`
- `python/modules/state/aec_state.py:165-166` — `transition_triggered()` accessor
- `python/modules/residual/suppression_gain.py:529-530` — `set_initial_state()`
- `python/modules/config.py` — flag definition location
- `python/modules/aec3_scale.py:82-88` — H_ERROR_INIT, LEAKAGE_* constants

AEC3 source references:
- `docs/aec3_extracts/src/aec3/echo_remover.cc:385-421` — top-level dispatch
- `docs/aec3_extracts/src/aec3/subtractor.cc:148-186` — full_reset + ExitInitialState
- `docs/aec3_extracts/src/aec3/refined_filter_update_gain.cc:53-68, 143-174` — HandleEchoPathChange + UpdateCurrentConfig
- `docs/aec3_extracts/src/aec3/aec_state.cc:143-171` — AecState::HandleEchoPathChange
- `docs/aec3_extracts/api/audio/echo_canceller3_config.h:88-118` — all config defaults
