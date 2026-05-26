# v3.21 Audit B — M1 / M4 / SubtractorOutputAnalyzer state-reset parity (2026-05-24, rev 2)

> **⚠ W=0 CORRECTION (2026-05-25)**: References in this doc to "OPEN v3.21.x parity gap" for
> ZeroFilter / tail-zero / M2 W-zeroing semantics are **RETRACTED**. Verified in
> `adaptive_fir_filter.cc:506-509`: `ZeroFilter(current=13, max=13)` = empty range = no-op;
> `SetSizePartitions(12, true)` → `ZeroFilter(13, 12)` = also no-op. AEC3 retains stale W
> in partitions 0–11. Python `zero_filter=False` **matches** AEC3 — both retain stale W.
> Python `W.fill(0)` (zero_filter=True) = non-AEC3 ablation; NOT a v3.21 alignment candidate.
> M2 tail-zero parity gap is **CLOSED** (no active-W zeroing in either system at default size).
> See `docs/v3_21_linear_filter_alignment_sweep.md` §1.2 S1 and
> `docs/v3_21_linear_filter_alignment_closure_plan.md` for authoritative corrected matrix.

**Status**: paper audit, complete on the M1 / M4 question. Read-only.
No code change in this round. No 12-case / 800-case run.
M2 / M3 / W=0 / usable_linear / PathChangeRegimeHandler / RES tuning
all out of scope for THIS audit's intervention.

**Goal**: map full AEC3 echo-path-change state-reset chain; compare
with our current pipeline; identify true AEC3-parity gaps vs PBFDKF
divergence vs trace-needed vs no-op.

> **Audit B close scope** (clarified rev 2): this audit closes the
> **M1 / M4 question** only. The wider EPC / AEC3 reset parity arc is
> **partially closed** — M2 exact semantics (W tail-zero +
> SetSizePartitions + transient config) and the initial-state chain
> (SuppressionGain::SetInitialState + Subtractor::ExitInitialState)
> remain **OPEN v3.21.x parity gaps**. See §5 + §8.

---

## 0. TL;DR

| Item | Verdict | Why |
|---|---|---|
| M1 (`SubtractorOutputAnalyzer::HandleEchoPathChange`) | **CLOSED: NO-OP / IRRELEVANT for our architecture** | AEC3's M1 reset is overwritten one line later by `Update()`'s recompute. No observable surface in AEC3 normal flow. Our pipeline has no equivalent latch to reset. Porting would have no behavioural impact. |
| M4 | **CLOSED: speculative label, no concrete AEC3 source** | No AEC3 file matches. Struck from open list. |
| `gain_change_hangover` rate-limit (3 blocks/frame) | **TRACE-NEGATIVE on 3-case cohort; DEFER** | Min HEPC gap on cohort = 251 frames (≫ 3) → no observable surface on these cases. Defer port until a fast-fire cohort is identified; not closed. |
| **M2 exact semantics**: `ZeroFilter(current, max)` (tail-zero) + `SetConfig(refined_initial, true)` + `SetSizePartitions(refined_initial.length_blocks, true)` chain on delay_change | **OPEN v3.21.x structural parity gap; no current ship candidate** | AEC3's "W reset on delay change" is NOT `W.fill(0)`. It is (a) tail-only zero, (b) switch gain config to transient profile, (c) shrink filter to initial length + zero displaced partitions. Three pieces together. Our Phase E M2 (`W.fill(0)`) is incomplete and over-aggressive. **Resolution requires exact-semantic design note BEFORE any code or true-delay cohort run.** Not v3.22 unless design intentionally deviates beyond AEC3. |
| `Subtractor::ExitInitialState` (transient → steady config swap on `TransitionTriggered`) | **OPEN v3.21.x parity gap (catalogued)** | Paired with M2 exact semantics in AEC3 — `SetConfig(initial)` on EPC and `SetConfig(steady)` on ExitInitialState. We have neither. Out of THIS audit's intervention scope but included in the same design note. |
| `SuppressionGain::SetInitialState(true)` on delay_change + `SetInitialState(false)` on ExitInitialState | **OPEN v3.21.x parity gap (catalogued)** | Our `SuppressionGain` has no `SetInitialState`. Out of this audit; separate SuppressionGain audit. |
| `AecState::HandleEchoPathChange` full_reset on delay_change | **PORTED** | `state/aec_state.py:219-224` matches AEC3 semantics. |
| `AecState::HandleEchoPathChange` `erle_estimator.Reset(false)` on gain_change | **HANDLER PORTED + AEC3-VALID; SOURCE-MAPPING IS PBFDKF-SPECIFIC** | The handler itself is AEC3-correct (`aec_state.cc:165-167`). The source-mapping (firing it from EPV/shadow_rise — events AEC3 doesn't have) is PBFDKF-specific legacy compensation. |
| Same-frame ordering: `bridge.filter_converged` captured BEFORE `_aec3_state.handle_echo_path_change` | **NO stale-state risk on Phase D path** | For Phase D (no W=0), bridge=True is correct. For Phase E (M2 W=0), bridge would be stale True — but Phase E destabilisation is dominated by the missing M2 exact-semantics chain (the M2 row above). |

---

## 1. AEC3 source map (file:line evidence)

### 1.1 Top-level dispatch — `echo_remover.cc:385-411`

```cpp
if (echo_path_variability.AudioPathChanged()) {
  if (echo_path_variability.gain_change) {
    if (gain_change_hangover_ == 0) {
      gain_change_hangover_ = kMaxBlocksPerFrame;   // = 3
      // log
    } else {
      echo_path_variability.gain_change = false;    // suppressed
    }
  }
  subtractor_.HandleEchoPathChange(echo_path_variability);   // line 401
  aec_state_.HandleEchoPathChange(echo_path_variability);    // line 402
  if (delay_change != kNone) {
    suppression_gain_.SetInitialState(true);                 // line 406
  }
}
if (gain_change_hangover_ > 0) --gain_change_hangover_;
// ...
if (aec_state_.TransitionTriggered()) {
  subtractor_.ExitInitialState();                            // line 419
  suppression_gain_.SetInitialState(false);                  // line 420
}
subtractor_.Process(...)                                     // line 424
// ... residual estimator, suppression gain ...
aec_state_.Update(...)                                       // somewhere later
```

### 1.2 `Subtractor::HandleEchoPathChange` — `subtractor.cc:148-175`

```cpp
const auto full_reset = [&]() {
  for (size_t ch = 0; ch < num_capture_channels_; ++ch) {
    refined_filters_[ch]->HandleEchoPathChange();                          // partial W zero
    coarse_filter_[ch]->HandleEchoPathChange();                            // partial W zero
    refined_gains_[ch]->HandleEchoPathChange(echo_path_variability);       // H_error, counters
    coarse_gains_[ch]->HandleEchoPathChange();                             // counters
    refined_gains_[ch]->SetConfig(config_.filter.refined_initial, true);   // *** transient config
    coarse_gains_[ch]->SetConfig(config_.filter.coarse_initial, true);     // *** transient config
    refined_filters_[ch]->SetSizePartitions(initial_length, true);         // *** shrink + tail zero
    coarse_filter_[ch]->SetSizePartitions(initial_length, true);           // *** shrink + tail zero
  }
};
if (delay_change != kNone) full_reset();
if (gain_change) {
  for (...) refined_gains_[ch]->HandleEchoPathChange(echo_path_variability);  // redundant if delay also true
}
```

### 1.3 `AdaptiveFirFilter::HandleEchoPathChange` — `adaptive_fir_filter.cc:506-509`

```cpp
void AdaptiveFirFilter::HandleEchoPathChange() {
  // TODO(peah): Check the value and purpose of the code below.
  ZeroFilter(current_size_partitions_, max_size_partitions_, &H_);
  // Only zeros UNUSED tail (from current to max). Active filter H_[0..current] is preserved!
}
```

The actual W zero-out for the active partitions comes from
`SetSizePartitions(initial, true)` which calls `ZeroFilter(old, current)` —
shrinking to initial size and zeroing displaced partitions. AEC3's
"W=0 on delay change" is **not** a `W.fill(0)`.

### 1.4 `RefinedFilterUpdateGain::HandleEchoPathChange` — `refined_filter_update_gain.cc:53-68`

```cpp
void RefinedFilterUpdateGain::HandleEchoPathChange(
    const EchoPathVariability& echo_path_variability) {
  if (echo_path_variability.gain_change) {
    // TODO(bugs.webrtc.org/9526) Handle gain changes.
  }
  if (echo_path_variability.delay_change != DelayAdjustment::kNone) {
    H_error_.fill(kHErrorInitial);                          // 10000
  }
  if (!echo_path_variability.gain_change) {
    poor_excitation_counter_ = kPoorExcitationCounterInitial;   // 1000
    call_counter_ = 0;
  }
}
```

Note: gain_change branch is TODO STUB. Counter resets fire only when
`!gain_change` — i.e. on `delay_change` only, OR on a no-op event with
both flags false (shouldn't happen — `AudioPathChanged` would be false).

### 1.5 `CoarseFilterUpdateGain::HandleEchoPathChange` — `coarse_filter_update_gain.cc:34-37`

```cpp
void CoarseFilterUpdateGain::HandleEchoPathChange() {
  poor_signal_excitation_counter_ = 0;
  call_counter_ = 0;
}
```

Unconditional counter reset on any HEPC (delay or gain via full_reset
path; on gain_change-only, NOT called — see `Subtractor::HandleEchoPathChange`
line 170-174 which only calls `refined_gains_->HandleEchoPathChange`,
not coarse).

### 1.6 `AecState::HandleEchoPathChange` — `aec_state.cc:143-171`

```cpp
const auto full_reset = [&]() {
  filter_analyzer_.Reset();
  capture_signal_saturation_ = false;
  strong_not_saturated_render_blocks_ = 0;
  blocks_with_active_render_ = 0;
  initial_state_.Reset();
  if (transparent_state_) transparent_state_->Reset();
  erle_estimator_.Reset(true);        // delay_change=true
  erl_estimator_.Reset();
  filter_quality_state_.Reset();
};
if (delay_change != kNone) full_reset();
else if (gain_change) erle_estimator_.Reset(false);     // gain only
if (subtractor_analyzer_reset_at_echo_path_change_) {   // M1, field-trial-on by default
  subtractor_output_analyzer_.HandleEchoPathChange();
}
```

### 1.7 `SubtractorOutputAnalyzer::HandleEchoPathChange` (M1) — `subtractor_output_analyzer.cc:64-66`

```cpp
void SubtractorOutputAnalyzer::HandleEchoPathChange() {
  std::fill(filters_converged_.begin(), filters_converged_.end(), false);
}
```

### 1.8 M1 consumer trace

`subtractor_output_analyzer_.ConvergedFilters()` is consumed in exactly
two places, both inside `AecState::Update`:

- `aec_state.cc:250` — `erle_estimator_.Update(..., ConvergedFilters())`
- `aec_state.cc:253` — `erl_estimator_.Update(ConvergedFilters(), ...)`

Both calls are AFTER `subtractor_output_analyzer_.Update(...)` at
`aec_state.cc:193` which **fully recomputes** `filters_converged_[ch]`
based on this frame's `e²` and `y²`:

```cpp
// subtractor_output_analyzer.cc:54
filters_converged_[ch] =
    refined_filter_converged || coarse_filter_converged_strict;
```

So M1's reset at `aec_state.cc:169` is **overwritten one execution step
later** by the recompute at line 193. The field-trial kill-switch
(`subtractor_analyzer_reset_at_echo_path_change_`) suggests this was
added defensively; it has no observable surface in the standard flow.

---

## 2. Mechanism table — `delay_change` (kNewDetectedDelay / kBufferFlush)

| # | AEC3 side-effect | AEC3 file:line | Our equivalent | Status |
|---|---|---|---|---|
| 1 | `refined_filters_[ch]->HandleEchoPathChange()` — zero unused W tail only (`ZeroFilter(current, max)`) | `subtractor.cc:152` → `adaptive_fir_filter.cc:506-509` | Phase E `PBFDKF.handle_echo_path_change(zero_filter=True)` = `W.fill(0)` (entire W) | **OPEN v3.21.x parity gap** — our zero is over-aggressive (entire W); AEC3 zeros tail only. Resolution paired with rows 6+8 (the M2 exact-semantics chain). No current ship candidate; see §8.1. |
| 2 | `coarse_filter_[ch]->HandleEchoPathChange()` — same as #1 for coarse | `subtractor.cc:153` → `adaptive_fir_filter.cc:506-509` | Phase E `PBFDAF.handle_echo_path_change(zero_filter=True)` = `W.fill(0)` | **OPEN v3.21.x parity gap** — same as row 1, coarse side. Paired with rows 7+9. |
| 3 | `refined_gains_[ch]->HandleEchoPathChange(delay_change)` — `H_error_.fill(kHErrorInitial=10000)` | `subtractor.cc:154` → `refined_filter_update_gain.cc:59-62` | Phase E/D `PBFDKF.handle_echo_path_change(delay_change=True)` → `H_error_per_bin.fill(10000)` | **ported** |
| 4 | `refined_gains_[ch]->HandleEchoPathChange(!gain_change)` — `poor_excitation_counter=1000, call_counter=0` | `subtractor.cc:154` → `refined_filter_update_gain.cc:64-67` | Phase E/D `PBFDAF.handle_echo_path_change(!gain_change)` → `poor_excitation_counter=400, call_counter=0` | **ported** (kPoorExcitationCounterInitial unit-conversion: AEC3 1000 blocks ≈ our 400 hops) |
| 5 | `coarse_gains_[ch]->HandleEchoPathChange()` — counter reset | `subtractor.cc:155` → `coarse_filter_update_gain.cc:34-37` | Same call path via shadow `PBFDAF.handle_echo_path_change` | **ported** |
| 6 | `refined_gains_[ch]->SetConfig(refined_initial, true)` — switch refined gain config to TRANSIENT profile | `subtractor.cc:156` | NOT IMPLEMENTED — we have a single steady refined config | **OPEN v3.21.x structural parity gap; no current ship candidate** — requires introducing a transient `refined_initial` profile distinct from steady `refined` (PBFDKF Kalman Q/R/P scales appropriate for fast-track phase). Resolution requires §8.1 exact-semantics design note before any code or true-delay cohort. NOT v3.22 unless design intentionally deviates beyond AEC3. |
| 7 | `coarse_gains_[ch]->SetConfig(coarse_initial, true)` — same for coarse | `subtractor.cc:157` | NOT IMPLEMENTED — single steady coarse config | **OPEN v3.21.x structural parity gap; no current ship candidate** — paired with row 6; same §8.1 design-note prerequisite. |
| 8 | `refined_filters_[ch]->SetSizePartitions(refined_initial.length_blocks, true)` — shrink to initial length + zero displaced partitions | `subtractor.cc:158-159` | NOT IMPLEMENTED — partitions fixed at PBFDKF construction | **OPEN v3.21.x structural parity gap; no current ship candidate** — requires either dynamic `_active_size_partitions` runtime variable with zero-on-shrink semantics OR documented decision to keep fixed (with separate justification). §8.1 design note. NOT v3.22 unless intentional deviation. |
| 9 | `coarse_filter_[ch]->SetSizePartitions(coarse_initial.length_blocks, true)` — same for coarse | `subtractor.cc:160-161` | NOT IMPLEMENTED — partitions fixed | **OPEN v3.21.x structural parity gap; no current ship candidate** — paired with row 8; same §8.1 prerequisite. |
| 10 | `aec_state_.HandleEchoPathChange(delay)` → `filter_analyzer.Reset()` | `aec_state.cc:146` | `state/aec_state.py:_full_reset` calls `_filter_analyzer.reset()` if FA enabled | **ported** (gated on `filter_analyzer_enabled` flag) |
| 11 | `capture_signal_saturation_ = false` | `aec_state.cc:147` | `state/aec_state.py:_full_reset` sets `_capture_signal_saturation = False` | **ported** |
| 12 | `strong_not_saturated_render_blocks_ = 0` | `aec_state.cc:148` | `state/aec_state.py:_full_reset` resets the same | **ported** |
| 13 | `blocks_with_active_render_ = 0` | `aec_state.cc:149` | same | **ported** |
| 14 | `initial_state_.Reset()` | `aec_state.cc:150` | `state/aec_state.py:_full_reset` calls `_initial_state.reset()` | **ported** |
| 15 | `transparent_state_->Reset()` (if present) | `aec_state.cc:151-153` | `state/aec_state.py:_full_reset` resets transparent state | **ported** |
| 16 | `erle_estimator_.Reset(true)` — clears + resets `_blocks_since_reset=0` | `aec_state.cc:154` | `state/aec_state.py:_full_reset` calls `_erle_estimator.reset(delay_change=True)` | **ported** |
| 17 | `erl_estimator_.Reset()` | `aec_state.cc:155` | same | **ported** |
| 18 | `filter_quality_state_.Reset()` | `aec_state.cc:156` | same | **ported** |
| 19 | `subtractor_output_analyzer_.HandleEchoPathChange()` — M1 = `filters_converged_.fill(false)` | `aec_state.cc:168-170` → `subtractor_output_analyzer.cc:64-66` | NOT PORTED — we have no `filters_converged_` vector; we recompute `_aec3_converged` each frame in orchestrator:3889 | **CLOSED NO-OP / IRRELEVANT for our architecture** — see §4.4 |
| 20 | `suppression_gain_.SetInitialState(true)` | `echo_remover.cc:406` | NOT PORTED — our `SuppressionGain` has no `SetInitialState` method | **OPEN v3.21.x parity gap (catalogued)** — separate SuppressionGain audit; not addressed by §8.1 (different subsystem). |

**Affects residual echo / usable_linear / ERLE / ERL / FQ / SuppressionGain?**

- #1 / #2 (W tail-zero): touches FILTER output → all downstream consumers
- #3 (H_error reset): Kalman re-arm → filter convergence speed → ERLE / FQ (via filter convergence state)
- #4 / #5 (counter reset): startup gate behaviour → filter doesn't adapt for ~400 hops → all downstream consumers see uncancelled echo
- #6-9 (config / size): would re-tune Kalman / NLMS gains for transient phase → not currently in our path
- #10-18 (AecState full_reset): clears ERLE / ERL / FilterQuality / saturation / initial-state / transparent-mode → affects RES gain + UsableLinearEstimate + TransparentMode
- #19 (M1): NO-OP (overwritten by Update line 193)
- #20 (SuppressionGain initial): would reset Wiener gain to conservative initial state → affects RES output until transition triggered

## 3. Mechanism table — `gain_change` (without delay_change)

| # | AEC3 side-effect | AEC3 file:line | Our equivalent | Status |
|---|---|---|---|---|
| 1 | `refined_gains_[ch]->HandleEchoPathChange(gain_change)` — **TODO STUB**, does nothing for the gain branch | `subtractor.cc:170-174` → `refined_filter_update_gain.cc:55-57` | Phase D wires EPV/shadow_rise as `delay_change=True` → triggers H_error reset; Phase F flag re-maps to gain_change=True → triggers TODO STUB (no-op) on filter side, AECMOS shows DT_static −0.104 regression | **divergent on purpose** — Phase D's `delay_change=True` source-mapping is PBFDKF-specific legacy compensation (see §5) |
| 2 | NO W reset on gain-only | `subtractor.cc:170-174` (no full_reset call) | Phase E gates W=0 on `delay_change && zero_filter`; Phase F classification removes both → W=0 no-ops on gain-only events | **ported** (W=0 correctly skipped on gain-only) |
| 3 | NO coarse filter or coarse counter reset on gain-only | `subtractor.cc:170-174` (no coarse calls) | Phase F: coarse `PBFDAF.handle_echo_path_change(gain_change=True)` → `if not gain_change` guard skips counter reset | **ported** |
| 4 | NO refined counter reset on gain-only | `refined_filter_update_gain.cc:64-67` (`if !gain_change` guard) | Phase F: `PBFDAF.handle_echo_path_change` same guard | **ported** |
| 5 | `aec_state_.HandleEchoPathChange(gain)` → `erle_estimator_.Reset(false)` only | `aec_state.cc:165-167` | `state/aec_state.py:222-224` matches exactly | **ported and AEC3-VALID** |
| 6 | `subtractor_output_analyzer_.HandleEchoPathChange()` (M1) — same as delay branch | `aec_state.cc:168-170` | NOT PORTED | **NO-OP / IRRELEVANT** |
| 7 | NO `suppression_gain_.SetInitialState(true)` (only on delay_change) | `echo_remover.cc:404-407` | NOT PORTED on either side | **missing**, OUT OF SCOPE |
| 8 | `gain_change_hangover` rate-limit (`gain_change` suppressed to once per ≤3 blocks) | `echo_remover.cc:386-398` | NOT PORTED — EPV / shadow_rise can re-fire every block | **TRACE BEFORE PORT** — may matter if our sources fire faster than 3 blocks/event |

**The handler itself (#5) is AEC3 parity.** What is PBFDKF-specific
is the SOURCE-MAPPING decision to fire it from EPV and shadow_rise —
events AEC3 doesn't have. AEC3 fires gain_change only from caller-
provided external `echo_path_gain_change`. Our internal EPV + shadow_rise
+ legacy companion stack is novel; the use of the gain_change handler
is a reasonable adapter from our novel events to AEC3's existing
handler.

---

## 4. Ordering audit

### 4.1 AEC3 order (per capture frame, `echo_remover.cc:385-424`)

```
A. Subtractor::HandleEchoPathChange       ← W=0, H_error reset (filter side)
B. AecState::HandleEchoPathChange         ← AecState reset + M1 reset
C. SuppressionGain::SetInitialState(true) ← only on delay_change
D. (transition triggered? ExitInitialState + SetInitialState(false))
E. Subtractor::Process                    ← compute fresh e, y, e², y²
                                            using POST-reset filter
F. ... residual estimator + suppression gain ...
G. AecState::Update                       ← consumes fresh post-Process state
   ├ aec_state.cc:193 subtractor_output_analyzer_.Update(...) ← recomputes filters_converged_
   ├ aec_state.cc:248 erle_estimator_.Update(...)
   ├ aec_state.cc:250 erle_estimator_.Update(..., ConvergedFilters())
   ├ aec_state.cc:253 erl_estimator_.Update(ConvergedFilters(), ...)
   └ ... rest of Update ...
```

### 4.2 Our order (per `AEC.process` frame, `orchestrator.py`)

```
1. Delay block (orchestrator.py:1832-1960)
   └ delay_first / delay_shift:
        ├ filter.handle_echo_path_change()  ← if Phase F flag ON (default OFF)
        └ set _aec3_pending_delay_change    ← queue for step 5
2. filter.process(near_end, far_end, mu)    ← produces e, y, e², y²
                                              using PRE-EPC W
3. EPC detection block (orchestrator.py:2492-2666)
   ├ EPV fire:        filter.handle_echo_path_change()  ← Phase D/F gates
   │                  set _aec3_pending_gain_change
   └ shadow_rise fire: same
4. _aec3_post(raw_output, ...)
   ├ compute _refined_conv / _coarse_conv from THIS frame's e, y, e², y²
   │   (orchestrator.py:3889)  _aec3_converged = ...
   ├ build FilterStateBridge(filter_converged=_aec3_converged, ...)
   │   (orchestrator.py:3892)
   ├ if _aec3_pending_*: _aec3_state.handle_echo_path_change(variability)
   │   (orchestrator.py:3946)  _full_reset / ERLE reset
   └ _aec3_state.update(bridge=..., ...)
       (orchestrator.py:3970) reads bridge.filter_converged
```

### 4.3 Same-frame `bridge.filter_converged` stale-state risk

**Does our order create stale `bridge.filter_converged` or stale analyzer state?**

| Scenario | bridge.filter_converged value | AEC3 equivalent | Stale? |
|---|---|---|---|
| Phase D, EPV/shadow_rise (no W=0) | True (filter is converged this frame; only H_error reset queued) | After AEC3 HEPC+Process, filters_converged_=True (filter still converged) | **NOT stale**: values agree |
| Phase E, EPV/shadow_rise (W=0 applied at step 3) | True (computed from pre-EPC filter.process at step 2) | After AEC3 HEPC+Process with W=0, filters_converged_=False (e≈mic) | **STALE**: bridge=True; AEC3 would have False |
| Phase F (no W=0, no H_error reset on gain) | True | Same as Phase D | **NOT stale** |

**Phase E stale-state risk** confirmed but **NOT THE PRIMARY PHASE E
FAILURE MODE**. Phase E catastrophe is dominated by:
1. W=0 → 60ms blank window (startup gate freeze) → mic-passthrough.
2. Explosive W recovery (~2× normal) → over-cancellation of DT speech.

The stale `bridge.filter_converged=True` is a secondary contributor —
filter_quality might think the filter is still good despite W=0, but
this is subordinate to the audio damage from #1/#2 above.

**For the Phase D + Phase G path (production)**: NO stale-state risk
exists. Bridge is correct.

### 4.4 M1 ordering interaction

M1's `filters_converged_.fill(false)` at `aec_state.cc:169` happens BEFORE
`subtractor_output_analyzer_.Update(...)` at `aec_state.cc:193`. The Update
call fully recomputes `filters_converged_[ch]` from the current
subtractor output. M1's reset has no observable surface.

**In our port**: even if we ported M1, we'd have nothing to reset
(we don't cache a `filters_converged_` vector — we recompute
`_aec3_converged` each frame at `orchestrator.py:3889`).
Bridge is built from THAT, then immediately consumed by Update.
M1 port would be a no-op variable in our system.

---

## 5. Verdict

### 5.1 OPEN v3.21.x structural parity gaps (still parity, not v3.22)

| Gap | Status | Recommendation |
|---|---|---|
| **M2 exact semantics** — `ZeroFilter(current, max)` (tail-zero only) + `SetConfig(refined_initial, true)` (transient gain config) + `SetSizePartitions(refined_initial.length_blocks, true)` (shrink + zero displaced partitions) on delay_change | **OPEN v3.21.x structural parity gap; no current ship candidate** | Resolution requires (a) introducing a transient `refined_initial` config distinct from steady `refined` (we have only one), (b) introducing dynamic `SetSizePartitions` (our partitions are fixed at construction), (c) replacing our `W.fill(0)` with the AEC3 tail-zero semantic. **Next step**: a design note that maps the AEC3 dynamic-config + dynamic-partition machinery onto our PBFDKF architecture, BEFORE any code. Then a true-delay-event cohort to validate. Do NOT chase as v3.22 unless the design intentionally deviates beyond AEC3. |
| `Subtractor::ExitInitialState` — transient → steady config swap on `aec_state_.TransitionTriggered()` | **OPEN v3.21.x parity gap** | Paired with M2 exact semantics: AEC3 sets `refined_initial` config on EPC and `refined` config on ExitInitialState. We need both halves or neither. Catalogue here; include in the same design note as M2 semantics. |
| `SuppressionGain::SetInitialState(true)` on delay_change + `SetInitialState(false)` on ExitInitialState | **OPEN v3.21.x parity gap** | Our `SuppressionGain` has no `SetInitialState` method. AEC3's `suppression_gain_.SetInitialState(true)` puts the Wiener gain into conservative initial-state output for the transient phase. Catalogue here; separate SuppressionGain audit. Do NOT bundle into the M2-semantics design note (different subsystem). |
| `gain_change_hangover` rate-limit (3 blocks/frame) | **TRACE-NEGATIVE on 3-case cohort only; DEFER** | See §5.4. Min HEPC gap = 251 frames (≫ 3) → no observable surface on this cohort. Defer port; revisit when a fast-fire cohort surfaces. |

### 5.2 PBFDKF-specific compensation (NOT AEC3 parity; legacy production behaviour kept)

| Item | Notes |
|---|---|
| Phase D source-mapping (EPV/shadow_rise → `filter.handle_echo_path_change(delay_change=True)`) | PBFDKF-specific compensation. Phase D's filter-side H_error reset is load-bearing on PBFDKF Kalman re-arm. AEC3's `RefinedFilterUpdateGain::HandleEchoPathChange` on gain_change is a TODO STUB (no H_error reset). The handler API is AEC3's; the source-mapping + the H_error-reset-via-delay-flag is PBFDKF compensation. |
| Phase G source-mapping (EPV/shadow_rise → `_aec3_pending_gain_change = True` → `_aec3_state.handle_echo_path_change(gain_change=True)` → `erle_estimator.Reset(false)`) | The HANDLER (`erle_estimator.Reset(false)`) is AEC3-valid parity (`aec_state.cc:165-167`). The SOURCE-MAPPING (firing it from EPV/shadow_rise — events AEC3 doesn't have) is PBFDKF-specific. |

### 5.3 NO-OP / IRRELEVANT for our architecture (do not port)

| Item | Why no-op |
|---|---|
| M1 (`SubtractorOutputAnalyzer::HandleEchoPathChange`) | AEC3's M1 reset is overwritten by `Update` line 193 → no observable surface in AEC3. We have no equivalent latch. Porting changes nothing. **CLOSED for our current architecture.** |
| M4 | Speculative label; no concrete AEC3 source identified. **Struck from the open list.** |

### 5.4 Trace result: `gain_change_hangover` → NO-OP on this cohort

Independent variable: do EPV / shadow_rise / delay re-fire within ≤ 3
blocks of a prior fire on the 3-case cohort?

Method: scan existing trace JSONs in `out_v3_21_20_phase_e_trace/`
for HEPC fires (H_error reset = 10000 proxy for any EPV/shadow_rise/delay)
and measure min gap between consecutive fires per case/variant.

Result:

| Case | Variant | Fire frames | Min gap (frames) |
|---|---|---|---:|
| jtYTdZm | A_off | [289, 540, 3115] | 251 |
| jtYTdZm | E_on | [289, 3111, 3666] | 555 |
| jtYTdZm | F_off / F_on | [] | n/a |
| nVUnxqHLr | A_off | [1112, 1567] | 455 |
| nVUnxqHLr | E_on | [1112, 3655, 4241] | 586 |
| nVUnxqHLr | F_off / F_on | [] | n/a |
| wVYS (mvmt) | A_off | [261, 642, 3657] | 381 |
| wVYS (mvmt) | E_on | [261, 697, 1054, 3657] | 357 |
| wVYS (mvmt) | F_off / F_on | [] | n/a |

**Min gap across all 3 cases × A_off/E_on = 251 frames (2.51 seconds at
hop=10 ms).** Well above the 3-block hangover threshold AEC3 imposes.

**Conclusion**: `gain_change_hangover` would be no-op on this cohort
(it never triggers because our EPV/shadow_rise events are intrinsically
sparse). The hangover port is **deferred** until a cohort with
fast-fire pattern is identified (would need a different dataset; out of
scope for this round).

### 5.5 No implementation proposed in this round

Per audit rules: **no implementation in this round**.

**Audit B closes the M1 / M4 question only**, not the entire EPC parity
arc. The mechanism landscape is now mapped:

- ~14 mechanisms already ported (matches AEC3 parity).
- **3 mechanisms remain OPEN as v3.21.x structural parity gaps** (M2 exact
  semantics + Subtractor::ExitInitialState + SuppressionGain::SetInitialState).
- 2 source-mappings classified PBFDKF-specific legacy compensation
  (Phase D + Phase G EPV/shadow_rise dispatch).
- 1 mechanism (M1) **CLOSED no-op/irrelevant for our current architecture**.
- 1 label (M4) **struck — no concrete AEC3 source**.
- 1 mechanism (`gain_change_hangover`) trace-negative on this 3-case
  cohort → **defer**, not close.

See §8 for the next-step plan (design note before code).

---

## 6. Wording corrections for existing docs

Apply per user direction:

- "AecState ERLE reset on gain_change is AEC3-valid."
- "EPV/shadow_rise as gain_change sources are PBFDKF-specific."
- "Therefore current Phase D/G behavior is legacy PBFDKF compensation, not v3.21 parity."

Both [docs/v3_21_epc_architecture_audit.md](v3_21_epc_architecture_audit.md)
and [docs/v3_21_phase_g_verdict.md](v3_21_phase_g_verdict.md) updated to
reflect this distinction. Specifically: the AecState handler is parity;
the SOURCE-mapping is PBFDKF compensation.

---

## 7. Conclusion

After full audit of the AEC3 echo-path-change state-reset chain across
`echo_remover.cc`, `subtractor.cc`, `adaptive_fir_filter.cc`,
`refined_filter_update_gain.cc`, `coarse_filter_update_gain.cc`,
`aec_state.cc`, `subtractor_output_analyzer.{cc,h}`:

### 7.1 What this audit closes

- **M1 is NO-OP / IRRELEVANT for our current architecture** —
  overwritten by `Update` one line later in AEC3 itself; our port has
  no equivalent latch to reset. Closing the M1 question.
- **M4 label struck** — no concrete AEC3 source identified.
- **`gain_change_hangover` trace-negative on 3-case cohort** — min HEPC
  gap = 251 frames (≫ 3). Defer port until a fast-fire cohort surfaces.

### 7.2 What remains OPEN (the wider EPC parity arc)

- **M2 exact semantics** — `ZeroFilter(current, max)` + `SetConfig(refined_initial, true)` + `SetSizePartitions(refined_initial, true)` chain on delay_change. Our `W.fill(0)` is incomplete + over-aggressive. **OPEN v3.21.x structural parity gap; no current ship candidate; needs exact-semantic design note BEFORE any true-delay cohort run.**
- **`Subtractor::ExitInitialState`** — transient → steady config swap on transition trigger. Paired with M2 semantics; same design note.
- **`SuppressionGain::SetInitialState(true)` on delay_change + `SetInitialState(false)` on ExitInitialState** — separate audit; out of this audit's scope but catalogued as v3.21.x parity gap.

### 7.3 Reframe of Phase D / G mechanism (rev 2)

| Component | Classification |
|---|---|
| Filter-side `H_error reset` triggered from EPV/shadow_rise (Phase D) | PBFDKF-specific legacy compensation. AEC3's `RefinedFilterUpdateGain::HandleEchoPathChange` on gain_change is a TODO STUB; our use of the `delay_change=True` flag to trigger H_error reset is novel to PBFDKF. |
| AecState `erle_estimator.Reset(false)` triggered from EPV/shadow_rise (Phase G isolation showed this is load-bearing bit-level) | HANDLER is AEC3-valid parity (`aec_state.cc:165-167`); SOURCE-MAPPING is PBFDKF-specific (EPV/shadow_rise are not AEC3 events). |

### 7.4 Status of the EPC parity arc

**Partially closed**. The M1 / M4 sub-question is closed. The wider arc
(M2 exact semantics + initial-state chain) remains OPEN.

Production main is unchanged. M2 W=0 / M3 / usable_linear /
PathChangeRegimeHandler all remain OPEN where applicable.

---

## 8. Next steps (no immediate code)

### 8.1 Build a small design note for exact AEC3 delay-change semantics in our PBFDKF architecture

**Drafted (2026-05-24): [docs/v3_21_delay_change_chain_design_note.md](v3_21_delay_change_chain_design_note.md).**

The note covers (per scope rules: design only, not a patch plan):

1. **The AEC3 delay-change reset chain at the filter level**:
   - `AdaptiveFirFilter::HandleEchoPathChange` → `ZeroFilter(current, max)` (tail only).
   - `RefinedFilterUpdateGain::SetConfig(refined_initial, true)` — switch gain config to transient profile.
   - `AdaptiveFirFilter::SetSizePartitions(refined_initial.length_blocks, true)` — shrink to initial length, `ZeroFilter(old_size, current_size)` for displaced partitions.
   - `CoarseFilterUpdateGain::SetConfig(coarse_initial, true)` + `AdaptiveFirFilter::SetSizePartitions` for coarse.
   - All four happen together inside `Subtractor::HandleEchoPathChange::full_reset()` (`subtractor.cc:150-163`).

2. **The paired return path on `aec_state_.TransitionTriggered()`**:
   - `Subtractor::ExitInitialState` → `SetConfig(refined, false)` + `SetSizePartitions(refined.length_blocks, false)` (immediate=false means smooth-transition over `config_change_duration_blocks`).
   - `SuppressionGain::SetInitialState(false)`.

3. **Our PBFDKF gap analysis**:
   - We have ONE `refined` config (no `refined_initial`). Need to introduce a transient `refined_initial` profile that maps onto PBFDKF Kalman Q / R / P scales appropriate for the initial fast-tracking phase.
   - Partitions are fixed at construction. Need to either (a) allocate at max size + use `_active_size_partitions` runtime variable + zero displaced partitions on size change, or (b) leave fixed and document this divergence explicitly.
   - The W tail-zero (`ZeroFilter(current, max)`) is meaningful only if size changes; if we don't do size changes, our W tail beyond `current_size_partitions` is structurally zero already.

4. **Decision matrix per gap**:
   - For each missing piece (transient config / dynamic partitions / tail-only zero / ExitInitialState transition smoothing), say: keep PBFDKF behaviour (closed as intentional v3.22 divergence ONLY IF we decide to deviate) OR introduce AEC3 mechanism (v3.21.x parity port).
   - For each "introduce" decision, sketch the touch points in `python/modules/filters.py` and the AecState / orchestrator hook points.

5. **Cost / risk**:
   - Introducing transient config = new dataclass field + 2-3 SetConfig touch points. Low code surface.
   - Introducing dynamic partitions = larger surface; touches PBFDKF P matrix shape, X_buf shape, W shape.
   - Going only tail-zero without size change = trivial, ~3-line edit, but might not help unless size change is also implemented (AEC3 ships them together).

6. **Validation prerequisite**:
   - **Only after the design note clarifies what to implement**, identify a true-delay-event cohort (12-case has zero such events) and run M2 + ExitInitialState together as paired changes. NEVER ship M2 alone.

### 8.2 Defer until §8.1 design note exists

- Phase G shipped flag-default-OFF (no production change).
- M2 W=0 substrate remains default-OFF.
- `gain_change_hangover` defer until fast-fire cohort.
- SuppressionGain::SetInitialState defer to separate audit.

### 8.3 Do NOT do (per rules)

- No code change in this audit round.
- No 12-case / 800-case run.
- No M2 / M3 re-enable.
- No usable_linear / PathChangeRegimeHandler / RES tuning.
- No "fix M2 by porting tail-zero" without the §8.1 design note first.
