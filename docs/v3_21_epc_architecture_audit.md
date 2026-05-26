# v3.21 EPC / EchoPathVariability Architecture Audit (2026-05-24, Round 2)

Round-2 audit after user directive 2026-05-24: classifier bug repair +
event-handler matrix + AEC3 ordering audit + 3-case A/E/F trace +
explicit answers A–F + ONE next experiment.

**Rules honoured this round:**
- Do not implement new algorithm behaviour (only classifier-bug repair).
- Do not run 800-case.
- Do not choose F.1 / M1 / M2 split.
- Do not touch usable_linear directly.
- Do not classify Phase D as v3.21.x parity if its benefit is not AEC3-equivalent.

**Sources**: existing trace JSONs in `out_v3_21_20_phase_e_trace/`
(nVUnxqHLr / jtYTdZm / wVYS × A_off / E_on / F_off / F_on) +
direct read of `python/modules/{orchestrator,filters,epc,state/aec_state,dataclasses}.py`.

> **Correction (Phase G, 2026-05-24, see [docs/v3_21_phase_g_verdict.md](v3_21_phase_g_verdict.md))**:
> §3 + §7 of this doc cited `erle_reset_signal=2` as evidence that "AecState
> fires identically in A_off and F_off". That signal is actually a diagnostic
> LABEL keyed off `self.epc_active` (orchestrator:3415-3429, "do not actually
> reset"), NOT the real AecState dispatch. The real dispatch is gated by the
> `_aec3_pending_gain_change` setter at the EPV/shadow_rise sites. Both A_off
> and F_off DO queue that setter, so AecState DOES fire in both — the
> conclusion stands, but the evidence cited was wrong. Phase G isolation
> (skipping the setter at EPV/shadow_rise) showed the real AecState ERLE
> reset is bit-level load-bearing (not byte-equal) but AECMOS-neutral on the
> 3 conflict cases.

---

## 0. TL;DR (one-page summary)

| Item | Finding | Action this round |
|---|---|---|
| Classifier bug | `python/modules/epc.py:29` referenced `AecEventType` without importing it — would raise `NameError` if the `'delay'` branch were ever taken | **Fixed** (added `AecEventType` to import). Verified all 4 source cases. No algorithm-output change (path was unreachable under current flags). |
| F_on vs F_off | **Byte-equal on all 3 conflict cases.** M2 W=0 never fires on the 12-case under Phase F | M2 has no observable surface on this cohort; no decision possible |
| Phase D load-bearing mechanism | Isolated to **filter-side H_error + counter reset**. AecState `erle_reset_signal` fires in BOTH A and F (queued via `_aec3_pending_gain_change`) | Reframe per **Audit B (2026-05-24)**: the gain_change HANDLER (AecState `erle_estimator.Reset(false)`) IS AEC3 parity; the SOURCE-MAPPING (EPV/shadow_rise → gain_change) is PBFDKF-specific. Filter-side H_error reset is NOT in AEC3 (TODO STUB). Net: Phase D / G is **legacy PBFDKF compensation**, not v3.21 parity overall — keep in production. |
| Same-frame ordering | Bridge `filter_converged` captured BEFORE `_aec3_state.handle_echo_path_change`. Stale True can re-engage filter_quality after a reset | Document as risk; not the dominant Phase E failure mode (filter-side W=0 + startup gate is) |
| Single next experiment | **Phase G — decouple filter-side H_error reset from AEC3 pending queue.** Run filter.handle_echo_path_change only; do not set `_aec3_pending_gain_change` at EPV/shadow_rise sites | **Recommendation only** — gate this on user approval; do not implement now |

---

## 1. AEC3 EPC reference (authoritative)

### 1.1 EchoPathVariability data structure

Three orthogonal flags constructed by `block_processor.cc`:

| Flag | Type | Set by | Frequency |
|---|---|---|---|
| `delay_change` | `DelayAdjustment` enum: `kNone` / `kNewDetectedDelay` / `kBufferFlush` | `RenderDelayBuffer::AlignFromDelay()` returns true | Rare (real delay-estimator step OR buffer overrun) |
| `gain_change` | `bool` | External `echo_path_gain_change` parameter — **AEC3 has NO internal gain detector** | Caller-driven (none in our wiring) |
| `clock_drift` | `bool` | `delay_controller_->HasClockdrift()` | Detected by clock skew |

### 1.2 AEC3 dispatch chain (`echo_remover.cc::ProcessCapture`)

```
HandleEchoPathChange(EchoPathVariability)  // CALLED FIRST in capture frame
  ├─ Subtractor::HandleEchoPathChange:
  │   ├─ delay_change != kNone → AdaptiveFirFilter::ZeroFilter(current_size, max_size) for refined + coarse
  │   │                            Zeroes W partitions in [current..max) ONLY.
  │   │                            Steady state (current=max=13): NO-OP — W PRESERVED.
  │   │                            Initial stage (current=12, max=13): zeroes tail partition 12.
  │   │                            Python W.fill(0) is NOT equivalent — it is more aggressive
  │   │                            and is classified as default-OFF ablation, NOT AEC3 parity.
  │   │                          + RefinedFilterUpdateGain::HandleEchoPathChange:
  │   │                            ├─ H_error reset to kHErrorInitial=10000
  │   │                            ├─ poor_excitation_counter = kPoorExcitationCounterInitial
  │   │                            └─ call_counter = 0
  │   │                          + CoarseFilterUpdateGain::HandleEchoPathChange:
  │   │                            ├─ poor_excitation_counter reset
  │   │                            └─ call_counter = 0
  │   └─ gain_change=true       → RefinedFilterUpdateGain TODO STUB (refined_filter_update_gain.cc:56)
  │                              (does NOTHING — explicit TODO marker)
  ├─ AecState::HandleEchoPathChange:
  │   ├─ delay_change != kNone → _full_reset() (filter_analyzer, ERLE, ERL, filter_quality, saturation)
  │   └─ gain_change=true       → erle_estimator_.Reset(false) ONLY
  └─ SuppressionGain::SetInitialState(true)  [delay_change only]

→ THEN Subtractor::Process()    // filter step happens AFTER HEPC
→ THEN AecState::Update()       // consumes fresh post-HEPC filter state
```

### 1.3 Critical AEC3 observations

1. **gain_change on RefinedFilterUpdateGain is a TODO STUB**. AEC3's own
   implementation does NOT reset H_error on pure gain change.
2. **AEC3 has NO internal gain detector**. `echo_path_gain_change` is caller-
   driven; default in our wiring = `false` always.
3. **AEC3 FilterMisadjustmentEstimator** (`subtractor.cc:336-367`) is filter
   SCALING, NOT an EPC trigger.
4. **AEC3 dispatch happens BEFORE subtractor.Process** — so when filter.process
   runs the W=0 takes effect on the SAME frame's e/y2 computation (e≈mic,
   e²>y² → filter_converged=False).
5. **No AEC3 regime handler** equivalent to our `PathChangeRegimeHandler`.

---

## 2. Our v3.21.6 EPC event sources

### 2.1 Event-source inventory

| Source | Detection (Phase 1 of pipeline) | AEC3 equivalent |
|---|---|---|
| **delay_first** | First valid delay acquired (`_current_delay<0` → solid PAR) | `delay_change=kNewDetectedDelay` ✓ true parity |
| **delay_shift** | `|new_delay − cur| > 32 samples`, conf ≥ 0.5, 2-frame consistency gate | `delay_change=kNewDetectedDelay` ✓ true parity |
| **EPV** | Fast/slow far-power EMA divergence > 6 dB (gain-ratio detector) | None native; closest mapping = `gain_change=true` (which AEC3's filter side ignores via TODO STUB) |
| **shadow_rise** | `(main_err+shadow_err) > prev × 1.3` AND `|Δ|/sum < 0.3` AND not stationary | **No AEC3 equivalent at all** — synthetic PBFDKF mistracking signal |
| **PathChangeRegimeHandler (boost_q / main_paused / reverse_copy)** | Per-frame regime selection — independent of EPC | **No AEC3 equivalent** — v3.21 invention, load-bearing on cohort tail |

---

## 3. Event-source × handler matrix (this round's deliverable)

Columns are the side-effects each event source can dispatch. Cells:
**✓** = fires by default; **F-only** = fires when `use_aec3_epc_classification`
flag is ON; **E-only** = fires when `use_aec3_zero_filter_on_epc` ON;
**legacy** = always fires (current production main, no flag gate); **─** = not fired.

| Source | filter W=0 (M2) | filter H_error reset | filter call_counter reset | filter poor_excitation reset | AecState `_full_reset` | AecState `erle_estimator.reset` | PathChangeRegimeHandler | boost_q | p_max_override | p_floor_beta | _erl_estimate cap | _epc_render_forced | _apply_epc_state_reset | _maybe_mark_diverged | _f_e3_handle_epc_fire | dtd_coherence dampening |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **delay_first** | F-only E-only | F-only ¹ | F-only ¹ | F-only ¹ | (via `_aec3_pending_delay_change`) ² | (via `_aec3_pending_delay_change`) ² | continues every frame ³ | ─ | ─ | ─ | ─ | ─ | ─ | legacy | ─ | ─ |
| **delay_shift** | F-only E-only | F-only ¹ | F-only ¹ | F-only ¹ | legacy (via `_aec3_pending_delay_change=NEW_DETECTED_DELAY`) | (via `_aec3_pending_delay_change`) ² | continues every frame ³ | legacy | legacy (PBFDKF only) | ─ | ─ | ─ | ─ | legacy | ─ | ─ |
| **EPV** | E-only ⁴ | **legacy** (Phase D wiring: `delay_change=True`) | **legacy** ¹ | **legacy** ¹ | ─ | legacy (via `_aec3_pending_gain_change=True`) ² | continues every frame ³ | legacy ⁵ | legacy (PBFDKF only) ⁵ | legacy ⁵ | legacy ⁵ | legacy ⁵ | legacy ⁵ | legacy ⁵ | legacy ⁵ | ─ |
| **shadow_rise** | E-only ⁴ | **legacy** ¹ | **legacy** ¹ | **legacy** ¹ | ─ | legacy (via `_aec3_pending_gain_change=True`) ² | continues every frame ³ | legacy ⁵ | legacy (PBFDKF only) ⁵ | legacy ⁵ | legacy ⁵ | legacy ⁵ | legacy ⁵ | legacy ⁵ | legacy ⁵ | legacy (× 0.3) |
| **PathChangeRegimeHandler** | ─ | ─ | ─ | ─ | ─ | ─ | n/a (this IS the handler) | per decision | per decision | ─ | ─ | ─ | ─ | ─ | ─ | ─ |

Notes:
¹ Always fires via `filter.handle_echo_path_change(delay_change=True, ...)` (Phase D wiring uses `delay_change=True` for EPV/shadow_rise; Phase F flag re-maps to `gain_change=True`, which makes the H_error reset NOT fire because `if not gain_change:` guards counter resets in `PBFDAF.handle_echo_path_change`, and `if delay_change:` guards H_error reset in `PBFDKF.handle_echo_path_change`).
² AecState dispatch is queued via `_aec3_pending_gain_change` / `_aec3_pending_delay_change`, then consumed once per `_aec3_post` call. Always fires regardless of Phase D / F flag.
³ `PathChangeRegimeHandler.update` runs every frame at line ~2456 region. Independent of EPC event detection. Boost_q / reverse_copy / pause_main decisions are computed from shadow_err vs main_err balance + DT gate, not from EPC fire.
⁴ When `use_aec3_zero_filter_on_epc=True` AND classification ON: re-mapped to `gain_change=True` → W=0 no-ops because `if delay_change and zero_filter:` guard fails. When classification OFF: `delay_change=True` AND `zero_filter=True` → W=0 fires. M2 effectively only fires at EPV/shadow_rise when E_on AND classification OFF (= the E_on variant from the bench).
⁵ Fired by the EPV / shadow_rise else-branches (`if self.config.aec_event_classification_enabled` selects between `_handle_gain_change_soft` and the full legacy companion stack). With `aec_event_classification_enabled=False` (current main), the full legacy stack fires.

### Concrete fire-count evidence from existing 12-case trace

| Source | A_off (Phase D production) | E_on (Phase D + M2) | F_off (Phase F classification) | F_on (Phase F + M2) |
|---|:---:|:---:|:---:|:---:|
| delay_first / delay_shift M2 W=0 | n/a (sites not wired in A) | n/a (sites not wired in E) | 0 fires across 3 cases | 0 fires across 3 cases |
| EPV M2 W=0 | n/a | nVUnxqHLr 0, jtYTdZm 1, wVYS 0 | n/a (M2 gated off via classification) | 0 (re-mapped to gain_change) |
| shadow_rise M2 W=0 | n/a | nVUnxqHLr 4, jtYTdZm 3, wVYS 4 | n/a | 0 |
| H_error reset count | 2 / 3 / 3 | 3 / 3 / 4 | **0 / 0 / 0** | **0 / 0 / 0** |
| `erle_reset_signal` jumps | every fire frame | every fire frame | every fire frame | every fire frame |

**Key empirical isolation**:
- `erle_reset_signal` (AecState ERLE reset signal) fires IDENTICALLY in A_off and F_off → AecState side is not the divergence.
- H_error reset count: 2–4 in A, **0** in F → filter-side H_error reset is the entire Phase D / Phase F divergence.
- F_on and F_off are **byte-identical** on the 12-case → M2 W=0 has zero observable surface here.

---

## 4. Ordering: AEC3 vs ours

### 4.1 AEC3 dispatch order (per capture frame, `echo_remover.cc`)

```
1. HandleEchoPathChange(variability)
     ├─ Subtractor::HandleEchoPathChange      ─→ W=0, H_error reset
     └─ AecState::HandleEchoPathChange         ─→ _full_reset / ERLE reset
2. Subtractor::Process(near, far)              ─→ produces e, y, e², y²
                                                    [W=0 → e≈near → e²>y² → not converged]
3. AecState::Update(...)                       ─→ reads FRESH post-Process filter state
```

### 4.2 Our dispatch order (per `AEC.process` frame)

```
1. Delay block (line ~1832-1960)
     └─ delay_first / delay_shift:
          ├─ filter.handle_echo_path_change()           [Phase F gates W=0 + H_error reset]
          └─ set _aec3_pending_delay_change             [queue for step 5]
2. filter.process(near_end, far_end, mu)                ─→ produces this frame's e, y, e², y²
                                                          (uses W set in step 1 if event fired)
3. EPC detection block (line ~2492-2666)
     ├─ EPV:  filter.handle_echo_path_change()          [Phase D/F: H_error reset and/or W=0]
     │        + legacy companion stack
     │        + set _aec3_pending_gain_change           [queue for step 5]
     └─ shadow_rise: same shape
        ⚠ filter.process already ran this frame using OLD W;
          handle_echo_path_change here only affects NEXT frame's filter.process
4. _aec3_post(raw_output, ...)
     ├─ compute _refined_conv / _coarse_conv from THIS frame's e, y, e², y²  (line 3889)
     ├─ build FilterStateBridge(filter_converged=_aec3_converged, ...)        (line 3892)
     │                                                  ⚠ bridge captured BEFORE step 5
     ├─ if _aec3_pending_*: _aec3_state.handle_echo_path_change(variability)  (line 3946)
     │                                                  → _full_reset / ERLE reset
     └─ _aec3_state.update(bridge=..., ...)             (line 3970)
                                                          ⚠ reads bridge.filter_converged
                                                          which is STALE (pre-reset value)
```

### 4.3 Same-frame stale-state risk (marked, not promoted to root cause)

`AecState.update` at orchestrator:3970 consumes `bridge.filter_converged`
which was captured at line 3892 BEFORE the inline
`handle_echo_path_change` call at line 3946. If the reset wiped
`filter_quality` internal counters, the very next instruction
(`_aec3_state.update`) re-feeds them with the **pre-reset** filter_converged.

- **EPV / shadow_rise on Phase D (no W=0)**: filter is still converged after
  H_error reset (filter weight intact). bridge.filter_converged=True;
  filter_quality stays engaged. Behaviour matches user intent (the filter
  IS still good, just needs faster mu).
- **Phase E (W=0)**: bridge captured this frame's pre-W=0 e/y² → reports
  filter_converged=True for this frame; AecState reset clears latches; next
  frame's update STILL sees stale True. Filter_quality could remain
  "convergence_seen" even though the next frame's filter is a mic-pass-through.
  **Secondary contributor** to Phase E's catastrophe; the primary mechanism
  is still the filter-side W=0 + startup gate freeze + explosive recovery.

**Recommendation**: not promoted to root cause this round; Phase E is
already incompatible at the filter level. If Phase G is approved, recheck
this risk under Phase G semantics.

---

## 5. 3-case A / E / F trace summary

First A-vs-E weight divergence is the first EPV/shadow_rise fire in each
case (Phase E's M2 W=0 takes effect there). Frames listed are around that
fire frame. Variant column: A = A_off (Phase D production), E = E_on
(Phase D + M2+M3), F = F_off (Phase F classification).

### nVUnxqHLr (DT_static, first conflict @ frame 1112 / t=11.12s)

| frame | var | w_norm | H_err | call_cnt | poor_exc | epc_act | shadow_w | out_rms |
|---:|:---:|---:|---:|---:|---:|---:|---:|---:|
| 1111 | A | 11.43 | 0.1 | 601 | 1512 | 0 | 11.25 | 0.0021 |
| 1111 | E | 11.43 | 0.1 | 601 | 1512 | 0 | 11.25 | 0.0021 |
| 1111 | F | 11.43 | 0.1 | 601 | 1512 | 0 | 11.25 | 0.0021 |
| **1112** | A | 11.37 | **10000.0** | **0** | **400** | 1 | 11.32 | 0.0021 |
| **1112** | E | **0.00** | **10000.0** | **0** | **400** | 1 | **0.00** | 0.0021 |
| **1112** | F | 11.37 | **0.1** ⚠ | **602** ⚠ | **1513** ⚠ | 1 | 11.32 | 0.0021 |
| 1117 | A | 11.37 | 100.0 | 5 | 405 | 1 | 11.18 | 0.0014 |
| 1117 | E | 0.00 | 100.0 | 5 | 405 | 1 | 5.37 | **0.0087** |
| 1117 | F | 11.24 | 0.1 | 607 | 1518 | 1 | 11.18 | 0.0017 |
| 1122 | A | 15.62 | 1.7 | 10 | 410 | 1 | 11.15 | 0.0008 |
| 1122 | E | **36.93** | 1.8 | 10 | 410 | 1 | 5.85 | 0.0008 |
| 1122 | F | 11.26 | 0.0 | 612 | 1523 | 1 | 11.15 | 0.0020 |

Note the F row at frame 1112: epc_act=1 fires (EPC detection works) but
H_err stays 0.1 and counters keep incrementing (602 / 1513) — Phase F's
classification correctly causes the filter-side handler to no-op. Phase
D's "wrong" classification caused the reset that A and E share.

### jtYTdZm (DT_static, first conflict @ frame 289 / t=2.89s)

| frame | var | w_norm | H_err | call_cnt | poor_exc | epc_act | shadow_w | out_rms |
|---:|:---:|---:|---:|---:|---:|---:|---:|---:|
| 288 | A | 18.27 | 0.8 | 286 | 689 | 0 | 17.32 | 0.0010 |
| **289** | A | 18.30 | **10000.0** | **0** | **400** | 1 | 17.39 | 0.0014 |
| **289** | E | **0.00** | **10000.0** | **0** | **400** | 1 | **0.00** | 0.0014 |
| **289** | F | 18.30 | **0.8** | **287** | **690** | 1 | 17.39 | 0.0014 |
| 299 | A | 30.13 | 78.1 | 10 | 410 | 1 | 17.18 | 0.0148 |
| 299 | E | **51.55** | 82.5 | 10 | 410 | 1 | 5.23 | 0.0136 |
| 299 | F | 18.06 | 1.0 | 297 | 700 | 1 | 17.18 | 0.0113 |

### wVYS (DT_movement, first conflict @ frame 261 / t=2.61s)

| frame | var | w_norm | H_err | call_cnt | poor_exc | epc_act | shadow_w | out_rms |
|---:|:---:|---:|---:|---:|---:|---:|---:|---:|
| 260 | A | 28.00 | 1.4 | 150 | 661 | 0 | 59.24 | 0.0019 |
| **261** | A | 28.00 | **10000.0** | **0** | **400** | 1 | 59.24 | 0.0017 |
| **261** | E | **0.00** | **10000.0** | **0** | **400** | 1 | **0.00** | 0.0017 |
| **261** | F | 28.00 | **1.5** | **151** | **662** | 1 | 59.24 | 0.0017 |
| 263 | A | 28.00 | 100.0 | 2 | 402 | 1 | 59.24 | 0.0833 |
| 263 | E | 0.00 | 100.0 | 2 | 402 | 1 | 0.00 | **0.1892** ⚠ |
| 263 | F | 28.00 | 1.5 | 153 | 664 | 1 | 59.24 | 0.0833 |
| 266 | A | 28.00 | 100.0 | 5 | 405 | 1 | 59.72 | 0.0349 |
| 266 | E | 0.00 | 100.0 | 5 | 405 | 1 | 9.52 | **0.3280** ⚠ |
| 266 | F | 28.78 | 1.3 | 156 | 667 | 1 | 59.72 | 0.0355 |
| 281 | A | 66.97 | 21.3 | 17 | 420 | 1 | 67.92 | 0.0143 |
| 281 | E | 51.37 | 21.7 | 17 | 420 | 1 | 45.36 | 0.0190 |
| 281 | F | 34.20 | 1.1 | 168 | 682 | 1 | 67.92 | 0.0042 |

### What the three cases say in one line each

- **A vs E differ ONLY in W**: same H_err reset, same counters, same epc_act, same shadow_w (in A) — the M2 W=0 is the entire delta.
- **A vs F differ ONLY in H_err and counters**: same W (intact), same epc_act, same shadow_w — Phase F's no-op on the filter side is the entire delta.
- **A's recovery is faster than F's** (W grows from 28→48→67 on wVYS vs F's 28→28→34) because A's H_err=10000 makes the Kalman gain large for the next ~10 frames.

**Per-output total RMS across whole utterance** (sum of |out_rms| across all frames):

| Case | A_off | E_on | F_off | F_on |
|---|---:|---:|---:|---:|
| nVUnxqHLr | 99.86 | 99.09 | **107.24** (+7.4%) | 107.24 |
| jtYTdZm | 91.79 | 95.98 | **107.70** (+17.3%) | 107.70 |
| wVYS | 198.00 | 201.13 | **209.85** (+6.0%) | 209.85 |

Phase F's removal of filter-side H_err reset causes 6–17% more output
energy — that's the residual echo Phase D was cancelling via fast Kalman
recovery, leaking through under Phase F.

---

## 6. Classifier bug (fixed this round)

### 6.1 Bug

`python/modules/epc.py:29` returns `AecEvent(delay_change=AecEventType.DELAY_NEW_DETECTED)`
but the file's import statement was:

```python
from .dataclasses import AecEvent, EpcEvent, RegimeHandlerDecision  # missing AecEventType
```

A call to `classify_epc_event(EpcEvent(True, 'delay'))` would raise
`NameError: name 'AecEventType' is not defined`. The bug was latent
because:
- The `'delay'` branch is reachable only when `EchoPathChangeDetector.force_delay()`
  is invoked (via `delay_shift` site in orchestrator:1923).
- That site only calls `classify_epc_event` when `aec_event_classification_enabled=True`.
- `aec_event_classification_enabled` defaults False; not enabled in any
  current bench / production config.
- The other two sources (`'epv'`, `'shadow_rise'`) return `AecEvent(gain_change=True)`
  which doesn't reference the unimported `AecEventType` — so those code
  paths worked even when the flag was enabled.

### 6.2 Fix

```python
# python/modules/epc.py:16
from .dataclasses import AecEvent, AecEventType, EpcEvent, RegimeHandlerDecision
```

One-line import addition. No algorithm-output change. Verified with all
four input cases (`'delay'` / `'epv'` / `'shadow_rise'` / not-fired).

### 6.3 Verification

```
delay        -> delay_change=new_detected gain_change=False  OK
epv          -> delay_change=none         gain_change=True   OK
shadow_rise  -> delay_change=none         gain_change=True   OK
not-fired    -> delay_change=none         gain_change=False  OK
```

---

## 7. Explicit answers (A–F)

### A. Why does Phase D help some cases despite being AEC3-wrong?

Phase D maps EPV/shadow_rise to `delay_change=True`, which triggers
**filter-side H_error reset to 10000 + call_counter=0 + poor_excitation=400**.
This is a Kalman re-arming: the next ~10 frames get a very large Kalman
gain because H_error is the dominant numerator. Filter W can grow/shift
rapidly to track the new echo path.

Evidence: per-frame trace shows A's W jumping 28→48→67 on wVYS while F's
W barely moves (28→28→34) over the same 20 frames after fire. A's larger
W matches the actual echo gain better → less echo leaks through.

This is **NOT AEC3-correct**: AEC3's `RefinedFilterUpdateGain` on
`gain_change=true` is a TODO STUB that does nothing. AEC3 doesn't need
H_error reset because AEC3's main "shadow" is NLMS, and NLMS step size
is always 0.5/X² regardless of "filter age" — no H_error analogue exists.

### B. Why does Phase E fail?

Phase E ADDS M2 W=0 on top of Phase D's H_error reset. The combination
is a filter-starvation cascade:

1. W=0 zeros the entire filter weight.
2. PBFDAF/PBFDKF startup gate (`call_counter <= n_partitions OR
   poor_excitation < n_partitions → return` in `filters.py`) freezes the
   filter for the next ~60ms (400-hop poor_excitation counter must drain
   before adapts resume).
3. During the 60ms blank window, e ≈ mic (no echo subtraction). The
   output is mic-passthrough — full echo leaks (out_rms 0.06 → 0.19 →
   0.33 on wVYS vs A's 0.04 → 0.08 → 0.04).
4. When the gate releases at frame ~1117, H_error=100 is still ~100× larger
   than A's H_error=100 (A's counter was decaying from 100 toward steady
   state; E's was zeroed and re-armed from W=0 baseline). The next mu is
   gigantic → W jumps 0 → 24 → 38 (overshoots A's 11→16→15 by ~2×).
5. The overshot W persists for hundreds of frames → over-cancellation of
   double-talk speech → degraded DT_static deg.

**Phase E is structurally incompatible with our PBFDKF+startup-gate
architecture.** Not a tuning bug; an architectural collision.

### C. Why does Phase F remove Phase D benefit?

Phase F's classification fix correctly maps EPV/shadow_rise to
`gain_change=True`. Under that classification, our
`PBFDAF.handle_echo_path_change` and `PBFDKF.handle_echo_path_change`
implementations gate H_error / counter resets on
`if not gain_change:` and `if delay_change:` respectively — so the
filter-side reset NO-OPS.

AecState side: still fires (`_aec3_state.handle_echo_path_change` is
queued via `_aec3_pending_gain_change=True`, and `erle_reset_signal=2`
appears in both A and F trace at fire frames — proves AecState fires
identically). The thing Phase F removes is the FILTER-SIDE H_error
reset, which is the very mechanism Phase D was load-bearing on.

Per-utterance output RMS jumps 6–17% under F → residual echo Phase D
was cancelling via fast Kalman recovery leaks through.

### D. Is Phase D benefit compensating for missing AEC3 M1/M4 reset, or for our own PBFDKF downstream state?

**Compensating for our own PBFDKF Kalman intrinsic property.**

Evidence:
- AecState-side reset (ERLE reset via `_aec3_pending_gain_change`)
  fires identically in A_off and F_off (`erle_reset_signal=2` in both).
- The ONLY divergence between A and F is filter-side H_error/counter reset.
- Therefore the load-bearing mechanism is **filter-side, not AecState-side**.
- Filter-side H_error reset is the Kalman gain re-arm. It has no AEC3
  analogue because AEC3 uses NLMS (no H_error).
- M1 (`SubtractorOutputAnalyzer::HandleEchoPathChange`) and M4 (whatever
  is meant by M4 — not yet ported) are upstream of `filter_converged`
  signal but don't touch H_error. They aren't the mechanism Phase D rides on.

**Verdict per user's rule** ("classify non-AEC3 compensation as v3.22
candidate only after evidence"): Phase D's load-bearing benefit is
**not AEC3-equivalent in the filter-side H_error reset path**. It is
PBFDKF-specific compensation. Strictly, this puts it in **v3.22
(intentional divergence) territory, not v3.21.x parity**.

**Reframe per Audit B ([docs/v3_21_audit_b_m1_m4_parity.md](v3_21_audit_b_m1_m4_parity.md)):**
The `AecState::HandleEchoPathChange(gain_change)` handler itself
(which calls `erle_estimator.Reset(false)`) IS AEC3-valid parity
(`aec_state.cc:165-167`). What is PBFDKF-specific is the **source
mapping** — firing AEC3's gain_change handler from our internal EPV +
shadow_rise detectors (AEC3 has no such detectors; it fires gain_change
only from caller-provided external signals). The filter-side H_error
reset on these events is the part with no AEC3 equivalent.

Since it is already in production main (it predates the parity effort),
we KEEP it and document it as legacy PBFDKF compensation — we do NOT
call it v3.21.x parity in the changelog or design docs going forward.

### E. Is shadow_rise truly safe to map to AEC3 gain_change, or should it be treated as a legacy-only event with no AEC3 handler?

**Treat as legacy-only event.** Reasons:

1. shadow_rise has no AEC3 analogue. AEC3's `FilterMisadjustmentEstimator`
   is filter SCALING, not an EPC trigger.
2. Mapping to AEC3's `gain_change` is misleading: AEC3 reserves
   `gain_change` for caller-driven external level changes, not for
   internal filter-mistracking signals. Our shadow_rise is the latter.
3. Mapping to `gain_change` causes confusion: it makes the AEC3 dispatch
   chain LOOK like it fires for shadow_rise, but the only side-effect
   that matters (filter H_error reset) is provided by our PBFDKF-only
   handler, NOT by anything AEC3-equivalent.
4. Documenting shadow_rise as "legacy-only PBFDKF event" makes the
   architecture honest: there's a PBFDKF mistracking detector
   (shadow_rise) that triggers a PBFDKF-only re-arm (H_error / counter
   reset). No AecState dispatch needed.

**This implies Phase G (see §8) should NOT queue
`_aec3_pending_gain_change` on shadow_rise either.**

### F. What is the single next independent variable to test?

**The independent variable: filter-side H_error reset, decoupled from
AEC3 pending queue at EPV/shadow_rise sites.**

See §8 for the full Phase G design.

---

## 8. Recommended next experiment (ONE independent variable)

### Phase G — decouple filter-side H_error reset from AEC3 pending queue

**Independent variable:** at EPV/shadow_rise EPC sites:
- KEEP filter.handle_echo_path_change(delay_change=True, gain_change=False, zero_filter=False)
  (the load-bearing Phase D filter-side reset)
- DROP set `_aec3_pending_gain_change = True`
  (because AecState reset is decorative on these events — ERLE reset
  doesn't help; the load-bearing mechanism is filter-side)

**Hold constant** (everything else):
- delay_first / delay_shift: keep current behaviour
  (`_aec3_pending_delay_change = NEW_DETECTED_DELAY` + Phase F filter
  dispatch when classification ON)
- PathChangeRegimeHandler: unchanged
- All other companion machinery: unchanged
- All other flags: defaults

**Predicted outcomes** (cheap, falsifiable, 12-case → if PASS, then 800-case):

| Outcome | What it means |
|---|---|
| 12-case byte-equal to A_off (no output diff) | AecState reset on EPV/shadow_rise is decorative. Phase D's benefit is purely filter-side. Recommend permanently dropping the AecState dispatch on these sources. Honest documentation: shadow_rise/EPV are PBFDKF-only events. |
| 12-case differs from A_off, in particular if DT_static deg drops | AecState ERLE reset on EPV/shadow_rise IS contributing something. Keep both pieces. Refine the mechanism description. |
| 12-case identical to A_off PLUS subsequent 800-case PASS | Ready to ship as v3.21.21 minor refactor: code clarification, no algorithm change. |

**Why this experiment is right per audit rules**:
- One variable (filter side vs both sides).
- Does NOT touch usable_linear.
- Does NOT add M1 / M2 / W=0 behaviour.
- Does NOT combine fixes (Phase G is independent of Phase F classification flag and independent of Phase E M2).
- Result is falsifiable: byte-equal or not.
- If PASS, opens path to honest mechanism documentation and code simplification.
- If FAIL, gives evidence about which AecState side-effect actually matters.

**NOT included in this audit's deliverable**:
- Phase G implementation (forbidden by user rules: "Do not implement new behavior").
- 800-case run (forbidden until 12-case Phase G is validated).
- Closing M2 (separately gated; do not decide).

### Alternative we considered and rejected

Option α (original audit's recommendation): trace-only 800-case audit of
`delay_first` / `delay_shift` fire rates on production main. Rejected
this round because:
- Requires running 800-case (against rule).
- Doesn't address the load-bearing Phase D mechanism (which is the actual
  source of every conflict observed so far).
- Phase G isolates the more important variable cheaper.

### NOT acceptable per audit rules (re-statement)

- Combining Phase G + classification fix + W=0 in one candidate.
- 800-case run before Phase G 12-case is validated.
- Closing M2 without first validating Phase G (we cannot recommend
  M2 ship until Phase D's mechanism is honestly understood and refactored).

---

## 9. Conclusion

Phase D is **not AEC3 parity**. It is **PBFDKF-specific compensation**
for the Kalman-gain-decay-over-time property that AEC3's NLMS does not
have. Per the rule "classify non-AEC3 compensation as v3.22 candidate":
strictly, Phase D belongs in v3.22 (intentional divergence). Practically,
it is already in production main and load-bearing → KEEP, document
honestly.

Phase E (M2 W=0 on EPV/shadow_rise) is **structurally incompatible**
with our PBFDKF + startup-gate + companion-stack architecture. Not a
tuning bug; an architectural collision. Do not retry.

Phase F (correct AEC3 classification) is **AEC3-correct on paper but
net negative empirically** because it removes Phase D's filter-side
H_error reset (the load-bearing mechanism). Do not ship as-is.

The single next independent variable to test is **Phase G**: keep Phase
D's filter-side H_error reset, but stop queueing `_aec3_pending_gain_change`
at EPV/shadow_rise sites. This decouples the load-bearing mechanism from
the decorative AecState dispatch, validating which side actually matters
and opening a path to honest mechanism documentation.

**Classifier bug fix is the only code change this round.** No new
algorithm behaviour, no benchmark run.

**Awaiting user authorization** for Phase G 12-case design + run.
