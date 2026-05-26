# v3.21 A.3 — Downstream AEC3 State Signals Closure Audit

**Date:** 2026-05-25
**Status:** CLOSED — no live consumer under production default config for all three signals
**Scope:** Read-only audit. No code, no benchmark.

---

## 1. Signal Definitions (AEC3 source)

All three signals are computed by `SubtractorOutputAnalyzer::Update`
(`subtractor_output_analyzer.cc:26-62`) and passed as local variables in `AecState::Update`
(`aec_state.cc:190-274`). They are NOT stored as persistent fields on `AecState` — they exist
only for the duration of a single `Update()` call, then are forwarded to consumers within that
same call.

### `coarse_filter_converged_relaxed` (per-channel local)

```cpp
// subtractor_output_analyzer.cc:50-51
bool coarse_filter_converged_relaxed =
    e2_coarse < 0.3f * y2 && y2 > kConvergenceThresholdLowLevel;
// kConvergenceThresholdLowLevel = 20 * 20 * kBlockSize
```
This is a **per-channel local variable**, not an output of the analyzer. It is immediately
OR'd into `any_coarse_filter_converged` (see below) and has no other consumer.

### `any_coarse_filter_converged` (analyzer output)

```cpp
// subtractor_output_analyzer.cc:58-59
*any_coarse_filter_converged =
    *any_coarse_filter_converged || coarse_filter_converged_relaxed;
```
Output is the OR of `coarse_filter_converged_relaxed` across all capture channels.
Uses **relaxed thresholds** (e2_coarse < 0.3×y2, y2 floor = 20² × kBlockSize = 400×64 = 25600)
vs the strict `coarse_filter_converged_strict` (e2_coarse < 0.05×y2, y2 floor = 50² × 64 = 160000).

### `all_filters_diverged` (analyzer output)

```cpp
// subtractor_output_analyzer.cc:52-53, 60
float min_e2 = std::min(e2_refined, e2_coarse);
bool filter_diverged = min_e2 > 1.5f * y2 && y2 > 30.f * 30.f * kBlockSize;
*all_filters_diverged = *all_filters_diverged && filter_diverged;  // AND across channels
```
True iff `min(e2_refined, e2_coarse) > 1.5 × y2` AND `y2 > 57600` for **all** capture channels.
This is a strict divergence condition (worst filter output exceeds 1.5× captured signal).

---

## 2. AEC3 Consumer Map

| Signal | AEC3 consumer | Usage |
|---|---|---|
| `coarse_filter_converged_relaxed` | *(none — local only)* | Feeds into `any_coarse_filter_converged` only |
| `any_coarse_filter_converged` | `TransparentModeImpl::Update` (HMM variant, cc:100) | HMM observation: `int out = (int)any_coarse_filter_converged` — drives P(transparent) update |
| `any_coarse_filter_converged` | `LegacyTransparentModeImpl::Update` (cc:150) | `bool /* any_coarse_filter_converged */` — **explicitly unused** |
| `all_filters_diverged` | `LegacyTransparentModeImpl::Update` (cc:190-196) | Resets `diverged_sequence_size_`; if streak ≥ 60 blocks, resets `non_converged_sequence_size_` → suppresses TM fire |
| `all_filters_diverged` | `TransparentModeImpl::Update` (cc:54) | `bool /* all_filters_diverged */` — **explicitly unused** |

**Selector:** AEC3 picks the TM implementation via `TransparentMode::Create()` (transparent_mode.cc:232-245):
- `bounded_erl=true` → `nullptr` (TM disabled)
- field trial `WebRTC-Aec3TransparentModeHmm` enabled → `TransparentModeImpl` (HMM)
- default → `LegacyTransparentModeImpl`

AEC3's production default is `LegacyTransparentModeImpl`, which ignores `any_coarse_filter_converged`
and uses `all_filters_diverged`.

---

## 3. Our Port Status

### `coarse_filter_converged_relaxed`

Computed in `orchestrator.py:3934-3936` as `_coarse_conv_relaxed`, gated by
`config.coarse_filter_converged_relaxed_enabled` (default `False`).
Used only for audit trace counters (`_coarse_relaxed_trace`). Not threaded into bridge / AecState / TM.
**Correctly not threaded — AEC3 itself treats this as a local intermediate.**

### `any_coarse_filter_converged`

Equals `coarse_filter_converged_relaxed` in our single-channel context.
Not passed to `transparent_mode.update()` — correct, because we ported the **Legacy** TM which
ignores it. HMM TM is not ported (comment in `transparent_mode.py:4`: "HMM variant NOT ported").

### `all_filters_diverged`

Computed as a **proxy** in `aec_state.py:333-335`:
```python
all_filters_diverged = (
    (not any_filter_converged) and bridge.divergence_indicator > 1.0
)
```
Passed to `transparent_mode.update(all_filters_diverged=...)` in `aec_state.py:340`.
Consumed by `transparent_mode.py:115`: `if not all_filters_diverged: self._diverged_sequence_size = 0`.

**Proxy vs AEC3 definition gap:**
- AEC3: `min(e2_refined, e2_coarse) > 1.5 × y2` — requires hard divergence in BOTH filters
- Our proxy: `(not any_filter_converged) AND bridge.divergence_indicator > 1.0` — broader condition;
  fires on any non-convergence with elevated divergence indicator

This gap is real but dormant: it only affects TM behaviour, and TM is disabled by default.

---

## 4. Signal × Consumer × Status Table

| Signal | AEC3 consumer | Our status | Wired? | Behavioral gap? | Active under production config? |
|---|---|---|---|---|---|
| `coarse_filter_converged_relaxed` | (per-channel local → feeds `any_coarse_filter_converged`) | Computed as audit trace; not independently wired | N/A | None — AEC3 doesn't export it either | No |
| `any_coarse_filter_converged` | HMM TM (field trial) | Not wired to TM; HMM TM not ported | No | HMM TM not ported (field trial only) | No — HMM TM field trial OFF; Legacy TM ignores signal |
| `any_coarse_filter_converged` | Legacy TM (default in AEC3) | Not passed — correct | N/A | None — Legacy TM explicitly ignores it (`/* unused */`) | No |
| `all_filters_diverged` | Legacy TM | Proxy wired to Legacy TM | Yes (proxy) | Proxy definition differs from AEC3 strict definition | No — TM disabled (`transparent_mode_enabled=False`) |
| `all_filters_diverged` | HMM TM | Not wired — HMM TM not ported | No | None — HMM TM explicitly ignores it (`/* unused */`) | No |

---

## 5. A.1 Dependency Check

The roadmap noted: "If A.1 full delay-change chain activates a more complete AecState path,
signals may gain live consumers."

Analysis:

**A.1 chain components** (Steps 1-9 per `v3_21_alignment_roadmap_refresh.md`):
- Steps 1-4: filter H_error reset and SetConfig
- Step 5: SetSizePartitions (Q2 fixed-size adaptation — not reopened)
- Step 6: AecState::HandleEchoPathChange → `initial_state=True` + ERLE/ERL/FilterQuality reset
- Step 7: TransitionTriggered
- Step 8: ExitInitialState
- Step 9: SuppressionGain::SetInitialState (zero behavioral effect under default config — see A.2 closure)

**Does A.1 activate TM?**

No. TransparentMode is enabled by `transparent_mode_enabled` config flag — a construction-time
decision independent of delay_change events. A.1 causes AecState to enter `initial_state`,
which resets ERLE/ERL/FilterQuality but does NOT enable or reset TM (TM has its own
`Reset()` method; AecState's `_full_reset` does not call `transparent_mode.Reset()`).

**Does A.1 cause any of the three signals to gain a live consumer?**

No. The signals are only consumed inside the TM update path. TM is disabled by default and
A.1 does not enable it. After A.1 implementation, all three signals remain no-consumer under
production config.

**Verdict: No A.1 dependency. A.1 does not change the consumer status of any of these signals.**

---

## 6. Verdict

**CLOSED — no live consumer under production default config for all three signals.**

| Signal | Verdict | Reason |
|---|---|---|
| `coarse_filter_converged_relaxed` | CLOSED — absorbed by `any_coarse_filter_converged` | AEC3 treats this as a per-channel intermediate only; not an independent signal export |
| `any_coarse_filter_converged` | CLOSED — no live consumer | Only HMM TM consumes it; HMM TM is behind a field trial not ported; Legacy TM (our port) explicitly ignores it |
| `all_filters_diverged` | CLOSED — proxy wired; no live consumer | Legacy TM code wired; proxy definition differs from AEC3 strict definition; dormant because `transparent_mode_enabled=False` in production |

**No implementation required for A.3 under current production config.**

**Closure scope guard:** This closure holds only under production default config (`transparent_mode_enabled=False`, Legacy TM). If either `transparent_mode_enabled=True` OR HMM TransparentMode is ever enabled, reopen A.3:
- `any_coarse_filter_converged`: still no live consumer even then (Legacy TM marks it `/* unused */`; HMM TM is not ported)
- `all_filters_diverged`: proxy gap becomes active — proxy (`not any_filter_converged AND divergence > 1.0`) differs from AEC3's strict definition (`min(e2_refined, e2_coarse) > 1.5 × y2`) in both frequency and timing. Replace with an AEC3-derived calculation from filter error spectra at that point.

---

## 7. Files Read

- `docs/aec3_extracts/src/aec3/subtractor_output_analyzer.cc` — signal computation
- `docs/aec3_extracts/src/aec3/subtractor_output_analyzer.h` — function signature
- `docs/aec3_extracts/src/aec3/aec_state.cc:190-274` — consumer call sites
- `docs/aec3_extracts/src/aec3/transparent_mode.cc` — both TM consumer implementations
- `docs/aec3_extracts/src/aec3/transparent_mode.h` — Update() signature
- `python/modules/state/aec_state.py:333-343` — our proxy computation + TM wiring
- `python/modules/state/transparent_mode.py` — our Legacy TM port + update() signature
- `python/modules/orchestrator.py:3918-3952` — `coarse_filter_converged_relaxed` audit trace
- `python/modules/config.py:135-150` — `coarse_filter_converged_relaxed_enabled` flag
