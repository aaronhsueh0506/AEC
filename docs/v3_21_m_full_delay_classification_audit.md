# v3.21 M_full_delay Classification Audit

**Date**: 2026-05-26  
**Scope**: wVYSGVTT_DT_mvmt usable_linear collapse (−46.9pp) root-cause audit  
**Prior session error**: wVYSGVTT collapse was incorrectly attributed to `p3h_diverged` fires at frames 462/1084.  
**This document**: Corrects that attribution and provides the definitive classification.

---

## Audit Summary (Verdict First)

**Primary finding: Category B — Python-only FilterPlateauDetector is the SOLE cause.**

- `diverged_reset_enabled = False` in BALANCED config → `p3h_diverged` CANNOT fire. Prior attribution was wrong.
- Actual reset source at frames 461 and 1083: `reason='plateau'` (FilterPlateauDetector).
- FilterPlateauDetector has NO AEC3 equivalent — Python-only safety net.
- M_full_no_plateau (plateau suppressed): usable_linear = **96.3%**, exactly M_D level.
- Conclusion: H_error reset itself does not prevent convergence. The Python plateau detector fires because `_filter_once_converged` never latches in M_full_delay (H_error reset at frame 75 delays initial convergence past the plateau grace window under sustained DT movement + A.3 poor-excitation suppression).

**Secondary finding: Category A — missing AEC3 SetConfig/SetSizePartitions steps.**

- `use_full_delay_change_chain` omits `SetConfig(refined_initial, true)` and `SetSizePartitions(refined_initial.length_blocks, true)` from AEC3's `Subtractor::HandleEchoPathChange()`.
- These switch filter gains to "initial" config (higher leakage rates, lower error floor) during recovery period.
- This is a genuine AEC3 parity gap, but NOT the direct cause of the wVYSGVTT collapse (Category A port incompleteness, not regression cause).

**Classification: Category B (primary) + Category A component (secondary, incomplete port)**

The prior `docs/v3_21_m_full_delay_trace.md` §Classification "Class B (PBFDKF + larger-μ incompatible with PathChangeRegimeHandler)" was **wrong** and must be corrected. This is NOT a PBFDKF architecture incompatibility — it is a Python-only safety net (FilterPlateauDetector) that fires because the H_error reset delays initial convergence past the plateau window.

---

## Audit Task 1: Full Chain Completeness

### AEC3 `Subtractor::HandleEchoPathChange()` (subtractor.cc)

```cpp
refined_filters_[ch]->HandleEchoPathChange();
  // AdaptiveFirFilter::ZeroFilter(current_size_partitions_, max_size_partitions_)
  // Zeroes tail partitions in range [current_size..max_size).
  // AEC3 default: refined max=13, initial=12.
  //   · initial stage (current=12, max=13): zeroes partition 12 only (tail extension)
  //   · steady state (current=13, max=13): current==max → no-op (W preserved, NOT zeroed)
coarse_filter_[ch]->HandleEchoPathChange();
  // Same ZeroFilter semantics; coarse max=13, initial=12 → same no-op in steady state
refined_gains_[ch]->HandleEchoPathChange(...);         // H_error = 10000, poor_excit = 1000
coarse_gains_[ch]->HandleEchoPathChange();             // H_error (coarse) reset
refined_gains_[ch]->SetConfig(config_.filter.refined_initial, true);  // ← MISSING in Python
coarse_gains_[ch]->SetConfig(config_.filter.coarse_initial, true);    // ← MISSING in Python
refined_filters_[ch]->SetSizePartitions(config_.filter.refined_initial.length_blocks, true);  // ← MISSING
coarse_filter_[ch]->SetSizePartitions(config_.filter.coarse_initial.length_blocks, true);     // ← MISSING
```

### AEC3 `EchoRemoverImpl::ProcessCapture()` on delay_change

```cpp
subtractor_.HandleEchoPathChange();        // above
aec_state_.HandleEchoPathChange();         // full_reset → FilteringQualityAnalyzer::Reset() (gate2 only)
suppression_gain_.SetInitialState(true);   // ported as set_initial_state(True)
// ... later, on TransitionTriggered():
subtractor_.ExitInitialState();            // partially ported: set_initial_state(False)
suppression_gain_.SetInitialState(false);  // ported
```

### Python `use_full_delay_change_chain` chain (orchestrator.py ~line 1889)

| Step | AEC3 | Python | Status |
|------|------|--------|--------|
| `filter.handle_echo_path_change()` — ZeroFilter(tail [current..max)) | ✓ | Python: W.fill(0) all partitions — **more aggressive** (see ZeroFilter note) | DIVERGES |
| `shadow.handle_echo_path_change()` — ZeroFilter(tail [current..max)) | ✓ | Python: shadow W.fill(0) all partitions — **more aggressive** | DIVERGES |
| `refined_gains.HandleEchoPathChange()` — H_error=10000, poor_excit=1000 | ✓ | ✓ (via filters.py) | PRESENT |
| `coarse_gains.HandleEchoPathChange()` — H_error reset | ✓ | ✓ | PRESENT |
| `SetConfig(refined_initial, true)` — switch leakage/floor/ceil to initial config | ✓ | ✗ | **MISSING** |
| `SetConfig(coarse_initial, true)` | ✓ | ✗ | **MISSING** |
| `SetSizePartitions(refined_initial.length_blocks, true)` | ✓ | ✗ | **MISSING** |
| `SetSizePartitions(coarse_initial.length_blocks, true)` | ✓ | ✗ | **MISSING** |
| `aec_state_.HandleEchoPathChange()` — gate2 reset only | ✓ | Partial (see gate-reset note) | PARTIAL |
| `suppression_gain_.SetInitialState(true)` | ✓ | ✓ | PRESENT |
| `ExitInitialState()` + `SetInitialState(false)` on TransitionTriggered | ✓ | ✓ (partial) | PRESENT |

**ZeroFilter semantics note (CORRECTION — remove all "W → 0 for all partitions" language)**:
`AdaptiveFirFilter::HandleEchoPathChange()` calls `ZeroFilter(current_size_partitions_, max_size_partitions_)`,
which zeroes only the **tail** range `[current_size..max_size)`. AEC3 default: refined max=13,
initial=12. Consequence:
- **initial stage** (current=12, max=13): zeroes partition 12 only (1 tail partition)
- **steady state** (current=13, max=13): `current == max` → **no-op; W is preserved, not zeroed**

Python's `handle_echo_path_change()` calls `W.fill(0)` (or equivalent), zeroing ALL partitions.
This is MORE aggressive than AEC3, not equivalent. Claims that "Python W.fill(0) = strict AEC3"
or "AEC3 W[i] = 0 for all partitions on delay change" are **INCORRECT and must not appear in
any document or code comment**. For steady-state delay changes (the typical case), AEC3 preserves
the existing W and only resets H_error (via the gains' HandleEchoPathChange).

**Gate-reset divergence note**: AEC3 `FilteringQualityAnalyzer::Reset()` resets ONLY gate2
(`filter_update_blocks_since_reset`) and preserves gate1 (`filter_update_blocks_since_start`) and
`convergence_seen`. Python's `_reset_filter_derived_state()` → `_reset_aec3_post()` creates a
**fresh AecState instance**, resetting ALL fields including gate1 and `convergence_seen`. This
makes Python more aggressive on each reset cycle, but is pre-existing (M_D also does this on
delay_first). The extra aggressiveness is not the cause of the plateau regression.

**Missing SetConfig content** (`echo_canceller3_config.h` lines 102-110):
```
refined_initial: leakage_converged=0.005 (vs 0.00005 normal refined → 100×),
                 leakage_diverged=0.5 (vs 0.05 normal refined → 10×),
                 error_floor=0.001 (same as normal refined — no change),
                 error_ceil=2.0 (same), noise_gate=20075344 (same)
coarse_initial:  rate=0.9 (vs 0.7 normal), length_blocks=12 (vs 13 normal),
                 noise_gate=20075344 (same)
```
NOTE: Prior docs incorrectly listed `leakage_converged=0.01`, `leakage_diverged=0.1`,
`error_floor=0.5`. These were wrong source mappings, corrected 2026-05-26.
Python PBFDKF has no "initial vs non-initial" config concept. This is a **v3.21 port incompleteness** but is NOT the direct cause of the wVYSGVTT collapse (see Task 4 — plateau suppression fully recovers usable_linear without SetConfig fix).

---

## Audit Task 2: wVYSGVTT Frame-Level Trace

### Confirmed `_reset_filter_derived_state` call sites (monkeypatched)

| Reset # | Frame | Reason | once_conv at fire | Source |
|---------|-------|--------|-------------------|--------|
| 1 | 75 | `delay_first` | False | use_full_delay_change_chain path (correct) |
| 2 | 461 | `plateau` | False | FilterPlateauDetector (Python-only, NO AEC3 equiv) |
| 3 | 1083 | `plateau` | False | FilterPlateauDetector (2nd fire, max_attempts=2) |

**Confirmed**: `diverged_reset_enabled = False` in BALANCED config (config.py). `p3h_diverged` requires `diverged_reset_enabled=True` → CANNOT fire. Prior attribution of frames 462/1084 to `p3h_diverged` was WRONG.

### Gate states around plateau fires

**Around frame 461 (first plateau fire):**
```
frames 456–461:  ul=True,  g1=True,  g2=True,  g3c=True,  g3e=True,  once_conv=False
frame 461:       plateau fires inside process() call  
frame 462+:      ul=False, g1=False, g2=False, g3c=False, g3e=True,  once_conv=False
```

**Around frame 1083 (second plateau fire):**
```
frames 1078–1083: ul=True,  g1=True,  g2=True,  g3c=True,  g3e=True,  h_err≈0.29
frame 1083:       plateau fires inside process() call
frame 1084+:      ul=False, g1=False, g2=False, g3c=False, g3e=True
```

### Recovery duration after each reset

| Reset frame | Recovery start | Hops to recover | Why so slow |
|-------------|---------------|-----------------|-------------|
| 75 (M_D) | ~115 | ~40 | Normal gate1+gate2; gate3_ext_delay=True shortcut works |
| 75 (M_full) | ~115 | ~40 | Same as M_D after delay_first; plateau has not fired yet |
| 461 (M_full plateau 1) | ~579 | ~118 | gate1=40 + gate2=20 from 0; poor_excitation (A.3) skips updates during movement → gate2 increments slowly |
| 1083 (M_full plateau 2) | ~2685 | ~1602 | A.3 even more suppressive during 2nd recovery; prolonged movement |

**Why `once_conv` never latches in M_full_delay**: After delay_first at frame 75, H_error reset to 2.0 (clipped from 10000). Larger μ (H_error=2.0) should speed re-convergence, but:
1. A.3 `use_poor_excitation_gate_for_shadow` skips many updates during movement frames
2. DT contamination during movement: ERLE remains < 6 dB from filter perspective  
3. Plateau grace window = 400 frames. Frame 75+400 = 475, plateau already eligible before `convergence_seen` can latch

In M_D, no H_error reset → μ stays small → ERLE builds more gradually BUT filter stays near converged values → `convergence_seen` latches BEFORE grace period expiry → plateau never fires.

### H_error trajectory (M_D vs M_full_delay, wVYSGVTT)

```
M_D:       frame 75 H_error ~0.25 (no reset), recovers via normal decay. Gate2 T at ~115.
M_full:    frame 75 H_error → 2.0, decays toward ~0.30 by frame 456. Gate2 T at ~115.
           Frame 461: plateau fires → H_error reset again (new AecState). Gate2 F.
           Frame 579: Gate2 T recovered.
           Frame 1083: plateau fires again → H_error reset. Gate2 F.
           Frame 2685: Gate2 T recovered (~1602 hops: A.3 suppression during 2nd movement).
```

---

## Audit Task 3: AEC3 Source Equivalence

### Does AEC3 have a FilterPlateauDetector equivalent?

**No.** Exhaustive search of AEC3 source (`aec_state.cc`, `echo_remover.cc`, `filtering_quality_analyzer.cc`, `subtractor.cc`, `refined_filter_update_gain.cc`) confirms:

- AEC3 has no "plateau detector" concept
- AEC3 does not call `HandleEchoPathChange()` in response to prolonged non-convergence
- AEC3's only `HandleEchoPathChange()` trigger is a detected `AudioPathChanged` event (actual echo path change)
- AEC3 relies on `CoarseFilterUpdateGain` and `RefinedFilterUpdateGain` with their own gain schedules to handle non-convergence gracefully — no emergency reset path

FilterPlateauDetector was introduced in Python as a safety net for PBFDKF filter stall scenarios with no AEC3 equivalent. It is not AEC3 alignment code.

### Does AEC3 reset `convergence_seen` / gate1 on divergence after delay change?

**No.** `FilteringQualityAnalyzer::Reset()` (called by `aec_state_.HandleEchoPathChange()`) ONLY resets:
- `filter_update_blocks_since_reset_ = 0` (gate2)

It does NOT reset:
- `filter_update_blocks_since_start_` (gate1)
- `convergence_seen_` (gate3 latch)

Python's `_reset_aec3_post()` creates a fresh AecState, resetting ALL including gate1 and `convergence_seen`. This divergence is pre-existing (present in M_D's delay_first reset) and not the cause of the plateau regression. It means Python is more aggressive on each reset, which is a separate Category A gap worth documenting.

### Does AEC3 have analogous p3h_diverged behavior?

`PathChangeRegimeHandler.update()` returns a `RegimeHandlerDecision` (boost_q, pause_main, reverse_copy) — no divergence-triggered reset. `p3h_diverged` in Python requires `diverged_reset_enabled=True`, which is default-OFF in BALANCED. AEC3 `Subtractor` has no equivalent divergence-detection reset — the NLMS shadow and main filters are structurally independent.

---

## Audit Task 4: Diagnostic Variant M_full_no_plateau

### Variant definition

Monkeypatched `_reset_filter_derived_state` to skip `reason='plateau'` fires (silently drop). `reason='delay_first'` and `reason='delay_shift'` still fire normally. All other M_full_delay flags ON.

### Results

| Variant | Resets | Reset frames | usable_linear | Δvs M_D |
|---------|--------|-------------|---------------|---------|
| M_D | 1 | 75 (delay_first) | **96.3%** | baseline |
| M_full_delay | 3 | 75, 461, 1083 | **49.5%** | −46.9pp |
| M_full_no_plateau | 1 | 75 (delay_first) | **96.3%** | 0.0pp |

**M_full_no_plateau = exact M_D level.** This definitively proves:

1. FilterPlateauDetector is the **sole cause** of the wVYSGVTT collapse in M_full_delay.
2. The H_error reset itself (10000→2.0 at delay_first) does NOT prevent convergence — the filter recovers normally if plateau doesn't intervene.
3. Suppressing plateau restores full usable_linear without any other change.

### Nores LF in M_full_no_plateau

9xjhi LF nores energy: unchanged vs M_full_delay (same as M_D at 18.72 dB). Artifact closure maintained in no-plateau variant. Confirms plateau suppression has no downstream RES artifact impact on nores cases.

---

## Audit Task 5: A.3 Poor-Excitation Duration Analysis

### AEC3 `kPoorExcitationCounterInitial = 1000`

From `refined_filter_update_gain.cc`:
```cpp
constexpr int kPoorExcitationCounterInitial = 1000;
// HandleEchoPathChange(): poor_excitation_counter = kPoorExcitationCounterInitial;
// Each frame where excitation is not poor: counter decrements by 1
// Each frame where excitation IS poor: counter increments by 1
// Early return (skip filter update) when: poor_excitation_counter >= n_partitions
```

AEC3: hop=64 samples → 250 hops/s → 1000 hops = **4.0 seconds** maximum poor-excitation block duration after delay change.

Python: hop=160 samples → 100 hops/s → 1000 hops = **10.0 seconds** maximum duration.

This is a **pre-existing hop-rate scaling gap** (same as T1.1 analysis). The functional duration in wall-clock time is 2.5× longer in Python than AEC3. However:
- This gap exists in **both M_D and M_full_delay** (A.3 is ON in both)
- The plateau fires because of the H_error reset in M_full_delay → delayed convergence → poor_excitation counter stays high longer during sustained movement
- Fixing the hop-rate scaling (1000 → 400 hops for equivalent AEC3 wall-clock duration) would be a v3.21 parity correction but is **not the root cause of the plateau regression**

### A.3 interaction with plateau

In M_full_delay after delay_first reset (frame 75), A.3 poor-excitation gate skips filter updates during movement-active frames. This slows gate2 increment rate (gate2 only increments when filter update actually runs). Combined with H_error=2.0 (larger μ → ERLE harder to build in DT scenario), the filter doesn't "look converged" by frame 461, so `convergence_seen` never latches, and plateau fires.

In M_D after delay_first (frame 75), H_error is NOT reset → μ stays at converged value (~0.25) → ERLE builds quickly → `convergence_seen` latches before frame 461 → plateau never eligible.

**A.3 hop-rate scaling** is a Category A parity gap (pre-existing, not introduced by M_full_delay). Documenting for v3.21 roadmap but not the regression root cause.

---

## Verdict

### Category assignments

| Issue | Category | Direct cause of wVYSGVTT collapse? |
|-------|----------|------------------------------------|
| FilterPlateauDetector fires at frames 461, 1083 | **B** (Python-only, no AEC3 equiv) | **YES — sole cause** |
| `SetConfig(refined_initial)` missing from delay chain | **A** (incomplete port) | No |
| `SetSizePartitions` missing from delay chain | **A** (incomplete port) | No |
| Python W.fill(0) vs AEC3 ZeroFilter (no-op in steady state) | **A** (over-aggressive weight clear) | No — existing behavior in both M_D and M_full_delay |
| `_reset_aec3_post` resets gate1+`convergence_seen` (AEC3 resets only gate2) | **A** (pre-existing divergence) | No — present in M_D too |
| A.3 poor-excitation hop-rate scaling (1000 hops = 10s Python vs 4s AEC3) | **A** (pre-existing scaling gap) | No — present in M_D too |

### Final classification: Category B (with Category A component)

**Category B**: The wVYSGVTT usable_linear collapse in M_full_delay is caused by a **Python-only** safety mechanism (FilterPlateauDetector) that has no AEC3 equivalent and is not part of the AEC3 delay-change chain. The AEC3 full delay-change chain itself is not the cause of the regression.

**Category A component**: The AEC3 delay-change chain as implemented in Python is **incomplete** in three ways:
1. Missing `SetConfig(initial)` — no initial/non-initial leakage config concept in Python PBFDKF
2. Missing `SetSizePartitions` — no active-partition-count gating during initial window
3. Python W.fill(0) over-clears (all partitions) vs AEC3 ZeroFilter which is a no-op in steady state (current=max=13)

All three are v3.21 port gaps, not PBFDKF architectural incompatibilities. Porting SetConfig(initial)
and SetSizePartitions equivalents to Python PBFDKF is v3.21.x scope (not v3.22).

**Implication for M_full_delay disposition**: This is NOT a case where M_full_delay must be deferred simply because PBFDKF is incompatible with AEC3 architecture. The incompatibility is specifically that Python's PBFDKF accumulates a plateau condition that AEC3 NLMS never encounters (because AEC3 doesn't have FilterPlateauDetector and never needs it). Two v3.21.x resolution paths exist:

1. **Narrow fix** (Variant C): Suppress plateau when within `initial_state_active` window after `delay_first`. Python adaptation, not AEC3 port. Fast.
2. **SetConfig port** (Variant D): Add initial-config leakage concept to PBFDKF (no W.fill(0)); switch to higher-leakage gains after delay change; ExitInitialState restores normal. Closes Category A gap, may prevent plateau mechanistically.

Both paths are v3.21.x scope. Gate 0 trace (Variants C + D) determines which is sufficient.

**Correcting the prior "deferred v3.22" classification**: The prior session wrote "deferred to v3.22 (requires PathChangeRegimeHandler redesign for larger-μ scenarios)" based on the wrong p3h_diverged attribution. Since p3h_diverged never fires (diverged_reset_enabled=False), no PathChangeRegimeHandler redesign is needed. Gate 0 trace will determine the minimal v3.21.x fix.

---

## Gate 0 Trace Plan

**Purpose**: Determine whether narrow plateau suppression (Variant C) is sufficient, or whether
full SetConfig/SetSizePartitions port (Variant D) is needed. No ship/no-ship verdict from Gate 0 alone.
No AECMOS. Primary cases: wVYSGVTT_DT_mvmt (regression case) + 9xjhiFbG_FS_static (nores artifact case).
All 8-case cohort secondary.

### Variant A — M_D baseline (reference, data already collected)

```python
# M_D = M_A_ceil2 with use_full_delay_change_chain=False
cfg.use_full_delay_change_chain = False
```
Use existing M_D trace data as reference for all Gate 0 fields.

### Variant B — M_full_delay current (data already collected)

```python
# M_full_delay current = M_D + use_full_delay_change_chain=True (no other changes)
cfg.use_full_delay_change_chain = True
```
Use existing M_full_delay trace data. Baseline for Variant C/D deltas.

### Variant C — M_full_delay + plateau suppression in post-delay initial chain window
**Classification: Python functional adaptation — NOT a strict AEC3 port.**
(AEC3 has no FilterPlateauDetector equivalent; this variant suppresses a Python-only mechanism
within the window where AEC3's SetConfig(initial) would otherwise accelerate convergence.)

```python
# Same as Variant B, PLUS:
# FilterPlateauDetector.update() skips fire when:
#   (1) initial_state_active is True (i.e., delay_first fired, ExitInitialState not yet called)
#   AND (2) once_converged is False
# No change to plateau behavior outside the initial_state window.
cfg.use_full_delay_change_chain = True
cfg.use_full_delay_plateau_suppression = True  # candidate; default-OFF
```
**Hypothesis**: wVYSGVTT usable_linear recovers to M_D level. Plateau suppressed at frames 461
and 1083 (both within initial_state window). 7 other cases unaffected (plateau never eligible).

**Verify**:
- Does `initial_state_active` gate the plateau correctly for the two fire frames (461, 1083)?
- Does `initial_state_active` exit before the plateau would be eligible on other cases?
- Are there cases where plateau fires OUTSIDE initial_state window that should remain active?

### Variant C2 — M_full_delay + plateau suppression with fixed-hop counter
**Classification: Python functional adaptation — NOT a strict AEC3 port.**
(Addresses the 8-frame gap that caused Variant C to fail: initial_state exits at frame 454,
plateau fires at frame 462. C2 uses a fixed 500-hop window from delay_first instead.)

```python
cfg.use_full_delay_change_chain = True
cfg.use_full_delay_plateau_suppression_fixed_hops = True  # default-OFF
cfg.plateau_suppression_fixed_hops = 500  # 500 hops = 5s @ hop=160
# Suppresses plateau while: (frame - trigger) < 500 AND not once_converged
```

**Hypothesis**: 500-hop window covers the critical plateau fire at frame 462 (387 hops after
delay_first=75). If once_converged latches during that window, suppression releases early.

### Variant D1 — WITHDRAWN (wrong constants)
**D1 as originally implemented used wrong AEC3 source values (0.01/0.1 per block instead of**
**0.005/0.5 per block). D1 verdict is RESCINDED. Results are INVALID for AEC3 parity claim.**
See D1_corrected below.

### Variant D1_corrected — M_full_delay + SetConfig(refined_initial) leakage CORRECTED
**Classification: Partial AEC3 alignment — Category A port (corrected leakage only).**
D1_corrected uses the correct AEC3 source values:
```
refined_initial.leakage_converged = 0.005f → 0.0125 per hop (= LEAKAGE_CONVERGED_TRANSIENT_PER_HOP)
refined_initial.leakage_diverged  = 0.5f   → 1.25 per hop   (= LEAKAGE_DIVERGED_TRANSIENT_PER_HOP)
refined_initial.error_floor = 0.001f       → SAME as normal refined (no change)
```
Does NOT port coarse rate=0.9. D1_corrected failure does NOT close D2.

```python
# Same as Variant B, PLUS corrected SetConfig(refined_initial) leakage rates:
#   On delay_first: _leakage_converged=0.0125, _leakage_diverged=1.25
#   On ExitInitialState: restore to LEAKAGE_CONVERGED/DIVERGED_PER_HOP_DEFAULT
#   NOT ported: coarse rate=0.9, SetSizePartitions
cfg.use_full_delay_change_chain = True
cfg.use_full_delay_setconfig_initial = True  # D1 flag uses corrected constants now
```

**Hypothesis**: Corrected (higher) leakage during initial window may change H_error trajectory →
`convergence_seen` may latch before plateau grace window.

### Gate 0 Mandatory Trace Fields

Collect for all variants on wVYSGVTT_DT_mvmt. Also collect A/B on 9xjhiFbG_FS_static.

| Field | What to capture | Why |
|-------|----------------|-----|
| `delay_first_frame` | Integer frame of delay_first event | Confirm gate |
| `delay_shift_frame` | Integer frame (if any) | EPV/shadow_rise gate |
| `plateau_fire_frames` | List of (frame, reason) for all resets | Main diagnostic |
| `initial_state_enter_frame` | Frame delay_first triggers SetInitialState(True) | Variant C gate |
| `initial_state_exit_frame` | Frame ExitInitialState fires | Variant C gate |
| `h_error_post_reset` | H_error_mean at frame+1 after each reset | Confirm clamp to 2.0 |
| `h_error_at_plateau_minus10` | H_error_mean 10 frames before each plateau fire | Convergence state |
| `active_n_partitions` | If Variant D: partition count in use during initial window | SetSizePartitions proxy |
| `config_leakage_converged` | If Variant D: actual leakage_converged value per frame | SetConfig proxy |
| `ul_gate1_frac` | Fraction of frames with gate1=True | gate1 recovery rate |
| `ul_gate2_frac` | Fraction of frames with gate2=True | gate2 recovery rate |
| `ul_gate3_frac` | Fraction of frames with gate3 (ext_delay OR conv_seen) = True | gate3 |
| `ul_gate4_frac` | Fraction of frames with gate4=True (transparent=OFF, always True) | gate4 |
| `usable_linear_frac` | Overall usable_linear fraction | Top-line metric |
| `convergence_seen_latch_frame` | First frame where convergence_seen latches (per segment) | Latch timing |
| `once_conv_latch_frame` | First frame where once_converged sets | Plateau eligibility gate |
| `W_norm_after_delay` | ||W|| at delay_first+1, delay_first+40, delay_first+100 | Weight state |
| `e2_refined_mean` | Mean refined residual energy | Convergence quality |
| `e2_coarse_mean` | Mean coarse residual energy | Convergence quality |
| `nores_LF_db` | 0–500 Hz nores energy | Artifact gate: wVYSGVTT and 9xjhi |
| `nores_MF_db` | 500–2000 Hz nores energy | Artifact gate |
| `nores_HF_db` | 2000–8000 Hz nores energy | Artifact gate |
| `epv_or_shadow_rise_fires` | Count EPV / shadow_rise triggers during initial_state window | Confirm NOT triggering delay chain |

### Gate 0 Decision Rule

After Gate 0 trace collected and reviewed:

- **Variant C sufficient** → Implement `use_full_delay_plateau_suppression` flag (Python functional adaptation; v3.21.x); SetConfig/SetSizePartitions remains open v3.21.x Category A item.
- **Variant C2 sufficient but C insufficient** → Implement fixed-hop variant; SetConfig gap open.
- **Variant D1/D2 prevents plateau mechanistically** → Closes Category A gap + resolves Category B as side effect.
- **None of C/C2/D1/D2 recovers wVYSGVTT** → Mark as unresolved v3.21 parity gap; do NOT move to v3.22; M_D proceeds to 12-case as composition candidate separately.

**See §Gate 0 Results below for actual outcome.**

---

## Gate 0 Results (2026-05-26)

### Trace: wVYSGVTT_DT_mvmt (primary regression case, delay_first=75)

| Variant | ul% | once_conv latch | plateau fires | Notes |
|---------|-----|-----------------|---------------|-------|
| A (M_D) | 96.3% | frame 424 | — | Reference |
| B (M_full_delay) | 49.5% | never | 462, 1084 | Baseline regression |
| C (initial_state gate) | 52.0% | never | 1060 | FAIL: initial_state exits frame 454; plateau fires 462 (8-frame gap) |
| C2 (fixed 500-hop) | 52.7% | never | 1051 | FAIL: first plateau suppressed (462→absent); second fires at 1051 |
| D1 | WITHDRAWN | — | — | **WRONG CONSTANTS** (0.01/0.1 not 0.005/0.5) — verdict rescinded |
| D1_corrected (leakage 0.005/0.5) | 49.4% | never | 470, 1084 | FAIL: plateau shifted 462→470 (+8 hops marginal); once_conv never latches |

### Trace: 9xjhiFbGo06_FS_static (nores artifact guard)

All variants A/B/C2/D1_corrected: ul=92.4%, plateau at 1133 (unchanged). nores_LF: A=45.23 dB, B/C2=45.58 dB (+0.35 dB), D1_corrected=45.20 dB (−0.03 dB). No regression. 9xjhi is safe across all variants.

### Trace: xFk7_DT_mvmt, XRTnTUjU_DT_static

xFk7: A/B/C2 identical (ul=96.1%). D1_corrected: ul unchanged but nores_LF +1.06 dB and once_conv delayed 1069→2710 — **leakage_diverged=1.25 hurts xFk7 convergence**. XRTnTUjU: C2=92.0% (+1.1pp vs A/B=90.9%) — C2 suppresses plateau at frame 450. D1_corrected XRTnTUjU: plateau still fires at 450 (leakage doesn't suppress), nores_LF +0.47 dB.

### Root cause finding — PRIMARY BLOCKER: `once_converged` never latches

**C2 diagnostic isolated the true root cause**: C2 suppresses the first plateau (frame 462 → absent), but `once_converged` still never latches because:
1. H_error reset to 2.0 at delay_first → larger μ → ERLE harder to build under DT movement
2. A.3 poor-excitation gating skips filter updates during movement → ERLE convergence suppressed
3. After C2 window expires (500 hops), detector resumes from frozen state → second plateau fires at frame 1051

**M_full_no_plateau proof** (from §Task 4): When ALL plateaus are suppressed, ul=96.3% = M_D. This confirms `once_converged` CAN latch if plateaus never fire — but plateaus fire BECAUSE once_conv never latches before them. It is a circular dependency: convergence is suppressed → plateau fires → resets convergence → repeat.

**D1_corrected finding (2026-05-26)**: Corrected leakage (0.0125/1.25 per hop) IS active
(leakage_initial_frac=10.3% on wVYSGVTT). H_error before plateau: 1.7068 at frame 460
(vs B's 0.6139) — leakage_diverged=1.25 pushes H_error HIGHER. However, higher H_error
means LESS convergence (larger μ, harder ERLE under DT), so once_conv still never latches.
Plateau shifts 462→470 (+8 hops marginal). Side effect: xFk7 once_conv delayed 1069→2710
and nores_LF +1.06 dB — leakage_diverged=1.25 is too aggressive for cases without major delay change.
Conclusion: corrected leakage rate is NOT the binding constraint. DT+A.3 gating dominate.

**Classification update**: The primary blocker is `once_converged` never latching (not plateau window size). Any fix must either (a) prevent plateau from firing long enough for once_conv to latch, OR (b) help once_conv latch faster before plateau becomes eligible.

### Gate 0 Verdict: UNRESOLVED v3.21 parity gap

- **Variant C**: FAIL (8-frame window gap)
- **Variant C2**: FAIL (once_conv never latches; second plateau fires after 500-hop window expires)
- **Variant D1**: WITHDRAWN — wrong constants (0.01/0.1 per block; AEC3 is 0.005/0.5). Verdict rescinded.
- **Variant D1_corrected**: FAIL (plateau shifts 462→470 marginal; once_conv still never; xFk7 nores +1.06 dB)
- **D2 (full AEC3 initial chain)**: NOT YET IMPLEMENTED — see `docs/v3_21_m_full_delay_d2_design.md`

M_full_delay remains **OPEN v3.21.x parity gap**. M_D = M_A_ceil2 is the provisional composition candidate for 12-case AECMOS (separately authorized). D2 implementation requires user authorization.

---

## Options Forward (Updated 2026-05-26)

### Option A: Implement D2 — Full AEC3 initial chain port

See `docs/v3_21_m_full_delay_d2_design.md` for full spec. D2 adds beyond D1_corrected:
- Corrected leakage (same as D1_corrected: 0.0125/1.25 per hop) — already traces show this alone is insufficient
- `shadow_mu_initial = 0.643` (coarse_initial.rate=0.9 adapted via T1.1 scaling)
- ExitInitialState restores leakage + shadow_mu
- SetSizePartitions: **documented as structural incompatibility** (PBFDKF fixed partition shape)
- NOTE: error_floor is NOT a D2 item — AEC3 `refined_initial.error_floor=0.001` equals normal refined

D2 may partially affect once_conv timing via faster shadow convergence, but cannot guarantee full recovery. D1_corrected already shows leakage alone is insufficient; D2 adds the coarse rate component.

**Requires user authorization before implementation.**

### Option B: Extended C2 + D2 combined

Implement D2 to close Category A gap; keep C2 as safety net. With D2's coarse rate=0.9 (shadow mu=0.643) providing faster shadow convergence, once_conv may latch within C2's 500-hop suppression window. However, D1_corrected already showed the leakage component is insufficient alone — the shadow mu component is the new unknown. Test as standalone D2 first.

**Requires user authorization for D2 implementation; C2 already implemented.**

### Option C: Mark M_full_delay as unresolved v3.21 parity gap (current state)

- **Classification**: Category B (Python-only plateau, fixable) + Category A (SetConfig incomplete + W over-clear)
- **NOT "PBFDKF-incompatible"**: correct framing is "Python safety net fires because once_conv can't latch before plateau window under H_error reset + DT movement"
- **NOT deferred to v3.22**: this is v3.21 AEC3 alignment work; gap remains OPEN in v3.21.x
- M_D (= M_A_ceil2) proceeds to 12-case AECMOS as composition candidate (separately authorized)
- M_full_delay stays open pending D2 implementation + trace

**User decision required before proceeding with Option A or B.**

---

## Supporting Evidence

```
Diagnostic runs (wVYSGVTT_DT_mvmt, 3666 frames):
  M_D:             resets=[75(delay_first)]                     UL=3531/3666=96.3%
  M_full_delay:    resets=[75(delay_first),461(plateau),1083(plateau)]  UL=1813/3666=49.5%
  M_full_no_plateau: resets=[75(delay_first)]                   UL=3531/3666=96.3%
  
  diverged_reset_enabled = False (BALANCED config) → p3h_diverged CANNOT fire.
  Prior frames 462/1084 = gate2 first False after plateau fires on 461/1083.
```
