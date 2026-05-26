# D2 — Full AEC3 Initial-State Delay-Change Chain: Design Note

**Status**: Design-only. NOT implemented. NOT closed by D1 failure.
**Date**: 2026-05-26
**Context**: `docs/v3_21_m_full_delay_classification_audit.md` §Gate 0 Trace Plan — D2 specification.

---

## Why D2 is needed

Gate 0 traces (C, C2, D1 all FAIL on wVYSGVTT) revealed a two-layer root cause:

**Layer 1 — once_converged never latches (primary blocker)**:
When `use_full_delay_change_chain=True` fires at delay_first:
- `handle_echo_path_change(delay_change=True)` resets H_error from working value (~0.41) to 2.0
  and resets filter taps W
- DT contamination + A.3 poor-excitation gating (active during movement) prevent filter adaptation
- `once_converged` gate requires sustained low ERLE — never met after the H_error reset
- Without `once_converged`, plateau can re-fire after any suppression window expires
- Proof: M_full_no_plateau (plateau fully suppressed) shows ul=96.3% = M_D level; once_converged
  latches normally when plateau is fully absent → once_converged IS achievable if not interrupted

**Layer 2 — Missing AEC3 SetConfig(initial) context (Category A gap)**:
AEC3 `HandleEchoPathChange()` + `SetConfig(refined_initial)` + `SetSizePartitions(initial)` +
`ExitInitialState` form a coordinated recovery framework. Ports D1 tested only leakage rates
(partial). Full D2 must address all components.

---

## AEC3 Initial-State Chain: Component Inventory

### Component 1: `HandleEchoPathChange()` — already implemented

**AEC3 source**: `adaptive_fir_filter.cc::HandleEchoPathChange()`
```cpp
void AdaptiveFirFilter::HandleEchoPathChange() {
    ZeroFilter(current_size_partitions_, max_size_partitions_);
}
```

**ZeroFilter semantics (critical)**:
- `ZeroFilter(current, max)` zeroes W partitions in `[current..max)` only
- Steady state: current=max=13 → **no-op**; W is PRESERVED
- Initial stage: current=12, max=13 → zeroes tail partition 12 only

**Python `handle_echo_path_change(zero_filter=False)`**: Already implements no-op (zero_filter=False
is the correct flag for steady-state operation). This matches AEC3.

**Status**: IMPLEMENTED (correct, no change needed).

### Component 2: `SetConfig(refined_initial)` leakage switch — D1 partial port

**AEC3 source**: `echo_canceller3_config.h` (lines 102-107):
```cpp
RefinedConfiguration refined_initial = {
    .leakage_converged = 0.005f,  // per block; Python: 0.0125 per hop (TRANSIENT_PER_HOP)
    .leakage_diverged  = 0.5f,    // per block; Python: 1.25 per hop (TRANSIENT_PER_HOP)
    .error_floor       = 0.001f,  // same as normal refined — NO CHANGE
    .error_ceil        = 2.f,     // same as normal refined
    // noise_gate = 20075344.f (same as normal refined)
};
```
NOTE: Prior D1/D2 docs incorrectly listed `leakage_converged=0.01`, `leakage_diverged=0.1`,
`error_floor=0.5`. These were wrong source mappings. Corrected 2026-05-26.

**D1 port** (`use_full_delay_setconfig_initial`):
- Switches `_leakage_converged` to 0.0125, `_leakage_diverged` to 1.25 (leakage only).
  These equal `LEAKAGE_CONVERGED/DIVERGED_TRANSIENT_PER_HOP` — the constants are aliased.
- `error_floor` UNCHANGED: AEC3 `refined_initial.error_floor=0.001` equals normal refined.
- Does NOT port coarse filter initial config (see Component 3).
- Restores on ExitInitialState (`TransitionTriggered`) in `_aec3_post`.

**D1 status (CORRECTED 2026-05-26)**: Prior D1 used wrong constants (0.01/0.1 per block →
0.025/0.25 per hop). D1 verdict based on those constants is WITHDRAWN. D1_corrected (0.005/0.5
per block → 0.0125/1.25 per hop) must be traced. See Gate 0 re-run results below.

**D2 delta over D1_corrected for refined config** (leakage already covered by D1_corrected):
```python
# D1_corrected already sets:
#   filter._leakage_converged = 0.0125  (LEAKAGE_CONVERGED_SETCONFIG_INITIAL)
#   filter._leakage_diverged  = 1.25    (LEAKAGE_DIVERGED_SETCONFIG_INITIAL)
# error_floor UNCHANGED (refined_initial.error_floor = 0.001 = normal refined)
# D2 NEW: shadow_mu_initial = 0.643 (coarse rate 0.9 adapted via T1.1 scaling)
```

**Adaptation note**: Leakage rates hop-scaled in `aec3_scale.py`.
`error_floor` is NOT a D2 candidate — AEC3 `refined_initial.error_floor=0.001` is the same
as `refined.error_floor=0.001` (no change between initial and normal configs).

### Component 3: `SetConfig(coarse_initial)` — NOT yet ported

**AEC3 source**: `echo_canceller3_config.h`
```cpp
CoarseConfiguration coarse_initial = {
    .rate       = 0.9f,            // vs 0.7f normal (higher learning rate)
    .noise_gate = 20075344.f,      // same as normal
    .length_blocks = 12,           // vs 13 normal (SetSizePartitions 12→13 over startup)
};
```

AEC3 also switches the coarse/shadow filter to a higher learning rate (0.9 vs 0.7) during
initial state. In Python, the shadow NLMS step-size is `shadow_mu_nlms=0.5`. There's no
"initial-state rate" concept in our shadow filter.

**Adaptation challenge**: Our shadow filter doesn't separate "converged" vs "initial" mu configs.

**Shadow mu_initial derivation** (hop=160, PBFDAF shadow, 5 partitions, SpectralSum denom):

```
AEC3 coarse system (normal state):
  hop       = 64 samples @ 16 kHz → 250 hops/s
  n_part    = 13
  X2        = SpectralSum: Σ_{p=0}^{12} |X_fft[p]|²   (13 partitions, float[-1,1]²)
  mu[k]     = coarse.rate / X2[k]  (CoarseFilterUpdateGain)
  rate_norm = 0.7
  → per-hop step = 0.7 / X2[k]
  → per-second update = 0.7 × 250 = 175.0 / X2[k] / s

AEC3 coarse system (initial state):
  rate_init = 0.9  (coarse_initial.rate)
  → per-second update = 0.9 × 250 = 225.0 / X2[k] / s
  rate boost factor = 225.0 / 175.0 = 1.286×

Python shadow NLMS (A.1 ON, normal state):
  hop       = 160 samples @ 16 kHz → 100 hops/s
  n_part    = 5  (= filter_length // block_size = 832 // 320 rounded)
  X2        = SpectralSum: Σ_{p=0}^{4} |X_buf[p]|²   (5 partitions, float[-1,1]²)
  mu_eff[k] = shadow_mu_nlms / X2[k]
  shadow_mu_nlms = 0.5 (current default)
  → per-second update = 0.5 × 100 = 50.0 / X2[k] / s

Per-unit-signal adjustment for partition count:
  X2_AEC3 ≈ (13/5) × X2_Python  (more partitions → larger spectral sum)
  → AEC3 effective per-second step per X2_Python unit:
     = 175.0 / (13/5 × X2_Python) = 175.0 × 5 / (13 × X2_Python) = 67.3 / X2_Python / s
  → Python normal:
     = 50.0 / X2_Python / s
  T1.1 ratio = 50.0 / 67.3 = 0.74× (Python 26% slower per second — <2× threshold, acceptable)

Shadow mu_initial derivation:
  AEC3 initial boost = 1.286× over normal
  → Python mu_initial = shadow_mu_nlms × 1.286 = 0.5 × 1.286 = 0.643
  → per-second = 0.643 × 100 = 64.3 / X2_Python / s
  → vs AEC3 initial = 225.0 × 5 / (13 × X2_Python) = 86.5 / X2_Python / s
  → ratio = 64.3 / 86.5 = 0.74× (same T1.1 gap maintained — consistent)

Equivalently: mu_initial = 0.5 × (0.9 / 0.7) = 0.643
```

**Why the derivation does NOT produce strict parity**:
1. AEC3 coarse uses FIR partitioned filter; Python shadow uses NLMS — different update equations
2. Partition count mismatch (13 vs 5) changes X2_sum magnitude; per-unit adjustment above is approximate
3. T1.1 already accepts a 0.74× gap for the normal-state mapping; initial-state has the same residual gap

**Classification**: `shadow_mu_initial = 0.643` is a **candidate adaptation** (NOT strict AEC3 port).
The proportional rate boost (0.9/0.7) is directly derived from AEC3 source, but the absolute mu value
inherits the T1.1 gap. If D2 is implemented, label as "Category A port with documented 0.74× adaptation gap."

To port coarse_initial:
- Add `full_delay_initial_shadow_mu: float = 0.643` to AecConfig
- Switch shadow `_mu` to 0.643 at delay_first, restore to 0.5 on ExitInitialState

**D2 classification**: This is a **documented adaptation** (not strict parity) because:
- AEC3 coarse and shadow have different filter structures (FIR vs NLMS)
- Partition count mismatch (13 vs 5) requires approximate scaling
- The T1.1 0.74× per-second gap carries forward to initial-state mapping

**Status**: NOT implemented in D1. Must be explicitly added in D2 implementation.

### Component 4: `SetSizePartitions(initial_length)` — adaptive difficulty

**AEC3 source**: In initial state, AEC3 uses `length_blocks=13` for coarse (same as normal).
For refined filter, `max_filter_length` is fixed; `SetSizePartitions` adjusts `current_size`
during startup to allow incremental partition activation.

**Python PBFDKF**: Uses fixed partition count (determined by `filter_length // block_size`).
There is NO incremental partition activation concept in our PBFDKF.

**Portability assessment**:
- Python PBFDKF partition shape is FIXED: `n_partitions = filter_length // block_size`
- AEC3 SetSizePartitions for startup path (0→13 partitions over time) CANNOT be directly
  ported to PBFDKF without fundamental restructuring
- This component is a **structural incompatibility** — not a configuration gap

**D2 documentation**: Mark as "structural adaptation — SetSizePartitions cannot be directly
ported to fixed-partition PBFDKF; treat as acknowledged gap in D2 design."

### Component 5: `ExitInitialState` (TransitionTriggered restore) — already in D1

**AEC3 source**: `InitialState::Transition()` called when FilteringQualityAnalyzer
determines sufficient data has been gathered (`TransitionTriggered()`).

**Python port (D1)**: `_aec3_post` detects `TransitionTriggered` via
`_aec3_state.initial_state_active()` going False → restores `_leakage_converged`
to `LEAKAGE_CONVERGED_PER_HOP_DEFAULT` and `_leakage_diverged` to `LEAKAGE_DIVERGED_PER_HOP_DEFAULT`.

**D2 delta**: Must also restore `shadow._mu` to 0.5 (normal value) on restore if coarse rate was switched.

**Status**: D1_corrected has the core ExitInitialState restore (corrected leakage). D2 extends it for shadow mu restore.

---

## D2 Root-Cause Gap vs the `once_converged` problem

**Critical finding from C2 trace (2026-05-26)**:

Gate 0 C2 shows: even if the first plateau (frame 462 in B) is suppressed,
`once_converged` never latches → second plateau fires at frame 1051.

This means the primary blocker is NOT just about the SetConfig parameters. The fundamental issue is:
1. H_error reset to 2.0 at delay_first
2. DT contamination + A.3 gating during movement
3. Filter never re-converges → ERLE threshold for `once_converged` never met
4. Plateau fires (as symptom of non-convergence)
5. Filter resets again → loop

D2's coarse rate increase (shadow mu 0.5→0.643) may accelerate shadow filter convergence,
potentially reducing DT contamination perception during movement. However:
- If DT contamination + A.3 gating prevents filter adaptation entirely, faster shadow mu
  alone cannot recover `once_converged`
- `once_converged` depends on windowed ERLE ≥ threshold, not just shadow tracking speed

**What D2 addresses**:
- Category A gap: SetConfig(refined_initial) corrected leakage + SetConfig(coarse_initial) rate (adapted)
- Corrected leakage rates may affect H_error trajectory during initial window
- Does NOT directly fix the DT contamination or A.3 gating interaction

**What D2 does NOT address**:
- `once_converged` latch depends on windowed ERLE, not just H_error
- A.3 poor-excitation gating blocks filter updates during movement regardless of SetConfig
- The plateau detector itself (Python-only) has no AEC3 equivalent — AEC3 would
  not reset the filter on this signal at all

**D2 prediction**: D2 may partially reduce the once_converged delay, but cannot guarantee
full recovery without also addressing A.3 interaction during movement. D2 is worth implementing
as the next step (Category A alignment closure), but should NOT be framed as a guaranteed fix.

---

## D2 Implementation Checklist (requires user authorisation before implementation)

1. **Add fields to `AecConfig`** (default values preserve byte-equal baseline):
   - `use_full_delay_setconfig_initial_d2: bool = False`  # D2 supersedes D1_corrected flag
   - `full_delay_initial_shadow_mu: float = 0.643`       # coarse_initial.rate=0.9 adapted

   NOTE: No `h_error_floor` field — `refined_initial.error_floor=0.001` equals normal refined.

2. **Modify `orchestrator.py`** (delay_first and delay_shift chain blocks):
   - At trigger: set corrected leakage (same as D1_corrected) AND `shadow._mu = 0.643`
   - On ExitInitialState (TransitionTriggered): restore leakage to defaults AND `shadow._mu = 0.5`

3. **Do NOT implement** SetSizePartitions — document as structural incompatibility.

4. **Trace fields to add** for D2 validation:
   - `h_error_floor_active` (True when initial floor is active)
   - `shadow_mu_initial_active` (True when initial mu is active)
   - `once_conv_latch_frame` — primary success metric
   - `plateau_fire_frame` — confirm suppression or prevention

5. **Test only after user authorises D2 implementation**.

---

## D2 vs C2 Interaction

If D2 is implemented AND C2 is also ON:
- D2 corrected leakage + faster shadow mu may allow `once_converged` to latch sooner
- Once `once_converged` latches, C2 suppression is released early (via `and not self._filter_once_converged`)
- Optimal: D2 helps `once_converged` latch, C2 provides plateau safety net while D2 takes effect

Recommendation: implement and test D2 standalone first. If D2 alone fixes `once_converged` timing,
plateau suppression (C/C2) may not be needed.

---

## Summary Table: D2 Components

| Component | AEC3 Source | D1_corrected Status | D2 Status | Portability |
|-----------|-------------|---------------------|-----------|-------------|
| HandleEchoPathChange no-op | adaptive_fir_filter.cc | DONE | DONE | Strict parity |
| SetConfig refined leakage (converged 0.005, diverged 0.5) | echo_canceller3_config.h | DONE (corrected) | DONE | Strict parity (hop-scaled) |
| SetConfig error_floor | echo_canceller3_config.h | N/A (0.001 = normal) | N/A (0.001 = normal) | No change needed |
| SetConfig coarse rate (0.9) | echo_canceller3_config.h | NOT DONE | NOT DONE — D2 new | Documented adaptation (mu scaling) |
| SetSizePartitions (length_blocks=12) | adaptive_fir_filter.cc | NOT PORTABLE | NOT PORTABLE | Structural incompatibility (PBFDKF fixed shape) |
| ExitInitialState (leakage restore) | InitialState.cc | DONE | DONE | Strict parity |
| ExitInitialState (shadow mu restore) | InitialState.cc | N/A | NOT DONE — D2 extension | Documented adaptation |

**Classification**: D2 = partial AEC3 alignment (Category A) with one documented structural adaptation
(SetSizePartitions) and one documented parameter adaptation (coarse rate scaling).
NOT "strict AEC3 parity" — must be labeled "Category A port with documented adaptations."
