# v3.21 A.2 — SuppressionGain::SetInitialState Audit

**Date:** 2026-05-25
**Status:** CLOSED — already covered; no parity gap under default AEC3 / BALANCED config
**Scope:** Read-only audit. No code, no benchmark.

---

## 1. AEC3 Mechanism

### Call sites (echo_remover.cc)

| Location | Call | Trigger |
|---|---|---|
| echo_remover.cc:406 | `suppression_gain_.SetInitialState(true)` | `echo_path_variability.delay_change != kNone` |
| echo_remover.cc:420 | `suppression_gain_.SetInitialState(false)` | `aec_state_.TransitionTriggered()` |

`initial_state_` defaults `true` at construction (`suppression_gain.h:146`).
So AEC3 starts in initial state and exits when `TransitionTriggered()` fires after stream-start convergence.
On any subsequent `delay_change` it re-enters initial state.

### Consumers of `initial_state_` (suppression_gain.cc)

**Consumer 1 — `GetMinGain` (cc:257-258):**
```cpp
if (!initial_state_ || suppressor_config.lf_smoothing_during_initial_phase) {
    // LF smoothing: floor min_gain[k] at last_gain[k] * dec for LF bands
    // Prevents gain dropping too fast after strong nearend.
}
```
Effect: when `initial_state_=true` and `lf_smoothing_during_initial_phase=false`, LF smoothing
is disabled — gains can drop faster during initial convergence (more aggressive LF suppression).

**Consumer 2 — `DominantNearendDetector::Update` (called via GetGain cc:417):**
```cpp
dominant_nearend_detector_->Update(nearend, echo, comfort_noise, initial_state_);
```
In the detector: if `initial_state_ && !use_during_initial_phase`, clears trigger/hold counters
and forces NE state = false. Otherwise update proceeds normally.

### AEC3 default configuration

From `echo_canceller3_config.h:223`:
```cpp
bool lf_smoothing_during_initial_phase = true;   // AEC3 default
```
From `DominantNearendDetection` config:
```cpp
bool use_during_initial_phase = true;             // AEC3 default
```

**With AEC3 defaults:**
- Consumer 1: condition `(!initial_state_ || true)` = **always true** → LF smoothing always active regardless of `initial_state_`
- Consumer 2: `use_during_initial_phase=true` → `initial_state_` is **ignored** by the NE detector

**Conclusion:** Under AEC3 default configuration, `SetInitialState(true/false)` has **zero behavioral effect** on suppression computation. The flag is a non-default config knob.

---

## 2. Our Port

### `set_initial_state()` — implementation (suppression_gain.py:529-530)

```python
def set_initial_state(self, state: bool) -> None:
    self._initial_state = bool(state)
```
Field `self._initial_state = True` at construction. ✓

### Consumer 1 — `_get_min_gain` (suppression_gain.py:677)

```python
if not self._initial_state or self._config.lf_smoothing_during_initial_phase:
    # LF smoothing block
```
**Identical condition** to AEC3. ✓

### Consumer 2 — NE detector update (suppression_gain.py:580)

```python
self._dominant_nearend.update(
    nearend_spectrum, echo_for_det, comfort_noise_spectrum, self._initial_state
)
```
`_DominantNearendDetector.update` (suppression_gain.py:379): checks `initial_state and not c.use_during_initial_phase`. ✓

### Our default configuration

```python
# SuppressorConfig:
lf_smoothing_during_initial_phase: bool = True    # matches AEC3 default

# DominantNearendConfig:
use_during_initial_phase: bool = True             # matches AEC3 default
```

Both match AEC3 defaults. ✓

### Orchestrator wiring

`orchestrator.py:3446-3479`: `_initial_state_active` is computed but is **trace-only** (read-only
diagnostic). `suppression_gain.set_initial_state()` is never called by the orchestrator today.

This is not a gap under current production code because A.1 (full delay-change chain) has
not been implemented — there is no delay_change handler to call it. The call will be added
as part of A.1 implementation.

---

## 3. Mapping Table

| AEC3 consumer | What it does | Our equivalent | Behavioral gap? |
|---|---|---|---|
| `SetInitialState(true)` on delay_change | Sets `initial_state_=true` | `set_initial_state()` method exists | No gap in mechanism; call site not yet wired (A.1 scope) |
| `SetInitialState(false)` on TransitionTriggered | Sets `initial_state_=false` | method exists | Same: A.1 scope |
| `GetMinGain` initial_state gate | Disables LF smoothing when `initial_state_=true AND lf_smoothing_during_initial_phase=false` | Identical Python condition | **No gap with default config** — both default to `lf_smoothing_during_initial_phase=true`, condition always true, LF smoothing always active |
| NE detector initial_state arg | Clears NE state when `initial_state_=true AND use_during_initial_phase=false` | Identical Python logic | **No gap** — both default `use_during_initial_phase=true`, initial_state ignored |

---

## 4. Verdict

**CLOSE — already covered. No design change needed.**

Rationale:

1. **Mechanism is fully ported.** `set_initial_state()` exists, `self._initial_state` is consumed in
   the same two code paths as AEC3, with identical conditions.

2. **Defaults match AEC3.** `lf_smoothing_during_initial_phase=True` and `use_during_initial_phase=True`
   match AEC3's defaults. Under these defaults, `initial_state_` has zero behavioral effect in
   both AEC3 and our port — the flag gates features that require non-default config to activate.

3. **Behavioral gap is zero under default config.** No suppression path produces different output
   whether `initial_state_` is `true` or `false` when both callee flags are at their defaults.

4. **Missing call site is A.1 scope, not A.2.** The orchestrator does not yet call `set_initial_state()`
   on delay_change or TransitionTriggered because A.1 full delay-change chain has not been
   implemented. Adding those calls is part of A.1 Step 9, not a separate A.2 fix. Since the
   calls have no behavioral effect with default config, their absence does not create a parity gap.

5. **Not a documented functional adaptation.** The mechanism is present and correctly configured.
   There is nothing to adapt — AEC3's own defaults neutralize the flag.

---

## 5. A.1 Dependency Flag

A.1 implementation (Step 9 of the delay-change chain) will add two call sites:
- `suppression_gain.set_initial_state(True)` on `delay_first` / `delay_shift`
- `suppression_gain.set_initial_state(False)` on `ExitInitialState`

These calls are mechanically correct and safe to add (no behavioral change under default config).
If in the future the suppression tuning switches to `lf_smoothing_during_initial_phase=False`, the
initial-state gate becomes behaviorally active; the wiring must be in place first. A.1 achieves
that.

**No further A.2 work required.**

**Closure scope guard:** This closure holds only under the default AEC3 / BALANCED config (`lf_smoothing_during_initial_phase=True`, `use_during_initial_phase=True`). If either of these defaults changes — in AEC3 upstream or in our BALANCED preset — the `initial_state_` gate becomes behaviorally active and A.2 must be reopened to verify parity.

---

## 6. Files Read

- `docs/aec3_extracts/src/aec3/suppression_gain.cc` — full implementation
- `docs/aec3_extracts/src/aec3/suppression_gain.h` — declaration + defaults
- `docs/aec3_extracts/src/aec3/echo_remover.cc` — call sites
- `docs/aec3_extracts/api/audio/echo_canceller3_config.h:223` — `lf_smoothing_during_initial_phase=true`
- `python/modules/residual/suppression_gain.py` — full Python port
- `python/modules/orchestrator.py:3446-3484` — orchestrator wiring (trace-only)
