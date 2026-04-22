# AEC3 Full Architecture Analysis — deep-dive vs this repo

Date: 2026-04-22
Status: reference doc for architectural decisions. Not a work item.
Supersedes: `docs/aec3_architecture_alignment.md` (short note), which
was the seed of this analysis.

Scope: WebRTC AEC3 mainline (fetched 2026-04-22) vs our system
(`python/aec.py`, branch `feature/b16-raw-dt-jump-veto`).

---

## 1. Executive summary

Two-day debug on PZ7V FS#84 converged on: **our shadow filter cannot
differentiate from main at echo-path-gain events** because both are
Kalman and share the same denominator saturation. B-16 (Stage 1B+1C
complete) is a veto-layer band-aid at the `raw_dt` site. Fixing the
shadow differentiation at the **algorithm level** is the long-term
correct path, and matches the AEC3 mainline design.

This doc:

1. Maps AEC3 dual-filter architecture (refined Kalman + coarse NLMS)
   and explains why our system's PBFDKF-shadow-on-PBFDKF degrades
   into a scaled duplicate.
2. Deep-dives two cross-cutting themes: **filter state copy
   mechanics** and **reset taxonomy (5 distinct granularities)**.
3. Catalogs 15 architectural gaps with priorities.
4. Queues 4 high-priority experiments after B-16 merge.

**Top-line conclusions**:

- B-16 is a short-term band-aid; keep it after Stage 1D as defensive
  layer.
- `shadow-as-NLMS` is the highest-confidence follow-up (aligns with
  AEC3 mainline, not invention).
- Filter-misadjustment W-scaling is a second non-optional protection
  layer that we are missing (expected to close part of the 4.6 dB gap
  between B-16's −20.42 dB and the target −25 dB on PZ7V).
- Reset taxonomy gap: we conflate 3 distinct reset types (delay /
  gain / recalibration) into `EPC large`, losing precision.

**Reading order**:

- First-time reader: §1 → §2 → §5 (15-gap table) → §6 (top-4
  experiments) → §8 (roadmap).
- Deep-dive on copy: §3 only.
- Deep-dive on reset: §4 only.

---

## 2. AEC3 dual-filter architecture

### 2.1 Algorithm contrast (refined vs coarse)

Both filters are instances of `AdaptiveFirFilter` (same W format,
same `AdaptPartitions` / `ApplyFilter` / FD-partitioned convolution
backend). They differ **only** in how per-bin gain `G[k]` is computed
each block.

| Step | Refined (`refined_filter_update_gain.cc`, Kalman) | Coarse (`coarse_filter_update_gain.cc`, NLMS) |
|---|---|---|
| `mu[k]` | `H_error[k] / (0.5·H_error[k]·X²[k] + n·E²_refined[k])` | `rate / X²[k]` |
| `P`-state update | `H_error[k] -= 0.5·mu[k]·X²[k]·H_error[k]` | none |
| Q injection | `H_error += leakage·erl` (dynamic, bifurcated by diverged flag) | none |
| W update | `W += mu · E_refined` (conj X implicit in FD conv) | `W += mu · E_coarse` |
| Clamping | `H_error` clamped to `[error_floor, error_ceil]` | none |

Kalman equivalence:
- `H_error ≡ P` (per-bin error covariance)
- `mu ≡ K` (per-bin Kalman gain)
- `0.5·X²` ≡ measurement scaling (the 0.5 is an AEC3-specific half-Hertz
  weighting that also appears in P update)
- `n·E²_refined ≡ R` (measurement noise scaled by partition count)
- `leakage·erl ≡ Q` (process noise, dynamically injected)

**Critical observation — Q (H_error) leakage injection** is
bifurcated:
```
if (E²_refined <= E²_coarse || disallow_leakage_diverged):
    H_error += leakage_converged · erl    // small (preserve P)
else:
    H_error += leakage_diverged · erl     // large (reopen learning)
```
This means: **when coarse is better than refined, refined opens up
its learning** (increases uncertainty, accepts larger step). This is
the primary mechanism by which AEC3 recovers from refined-stuck
situations.

Our PBFDKF has an analogous Q injection (`Q_gated = Q_modulated *
far_active_mask + Q_floor·(1-mask)`) but **no bifurcation based on a
coarse reference**. Our Q always comes from `Q_high × q_scale`;
there is no "refined diverged from a coarse truth → open up more"
path.

### 2.2 Why AEC3 chose this combination

Design intent (inferable from code + comments):

- **Refined (Kalman)** produces the audio output. Kalman's
  covariance tracking gives stable, high-quality convergence in
  steady state, and low misadjustment at high ERLE. It is
  **expensive** to restart (P must re-accumulate).
- **Coarse (NLMS)** is the fast tracker. Fixed step `rate/X²`
  guarantees it always responds to the latest error. No P state
  means nothing to restart. It is the **sanity check** against
  refined.

Three uses of the coarse filter:
1. **Reference for refined diverged detection**: `E²_refined >
   E²_coarse` enables the `leakage_diverged` Q branch above.
2. **Warm-start target for refined**: when coarse consistently
   outperforms refined (`poor_coarse_filter_counters` logic,
   inverted — actually AEC3 uses the coarse-worse direction; see
   §3.1), trigger reset paths.
3. **Independent echo path change detector**: since coarse adapts
   faster, `shadow_adv = e_main/e_shadow` (our signal) or its
   inverse has genuine separation power. The Kalman
   self-normalization that pins our PBFDKF shadow to ~1.0 **does
   not apply** to NLMS, because NLMS `mu` does not depend on P.

### 2.3 Why our system's PBFDKF-on-PBFDKF fails

Demonstrated in Tasks 3-4 of this debug (see
`docs/b15_superseded.md` for full analysis):

- Both filters have `K ∝ P·conj(X) / (Σ P|X|² + R)`.
- Shadow `Q_shadow = 3.5 × Q_main` → `P_shadow ≈ 3.5 × P_main`.
- At echo events, `Σ P|X|²` dominates R; numerator and denominator
  both scale with P → **K_ratio collapses to 1.0**.
- Observed: `main_err / shadow_err = 1.00 … 1.05` during the full
  PZ7V t=9.9-10.5s onset (Task 3 diag, 460 ms window).
- `shadow_adv` never crosses the `epc_small_shadow_adv = 1.3` guard
  → EPC never fires at onset.
- Bit-exact confirmation: `experiment/pbfdkf-no-epc-baseline` with
  `AEC_EXP_NO_EPC=1` produces identical −13.83 dB leak. EPC is
  cosmetic at onset.

---

## 3. Filter state copy mechanics (theme 1)

### 3.1 AEC3 copy: refined → coarse, single direction

Call site: `subtractor.cc`:
```cpp
poor_coarse_filter_counters_[ch] =
    output.e2_refined < output.e2_coarse
        ? poor_coarse_filter_counters_[ch] + 1
        : 0;
if (poor_coarse_filter_counters_[ch] >= 5) {
    coarse_filter_[ch]->SetFilter(
        refined_filters_[ch]->SizePartitions(),
        refined_filters_[ch]->GetFilter());
    coarse_filter_reset_hangover_[ch] =
        config_.filter.coarse_reset_hangover_blocks;
}
```

Condition: **5 consecutive blocks** (~50 ms at 10 ms block) of
`e_refined < e_coarse` ⇒ overwrite coarse with refined's W **and
size**.

Transferred state:
- Filter W: all partitions, FD complex weights. `SetFilter` can
  change partition count (coarse may be shorter; refined size wins
  on copy).
- **Not transferred**: any Kalman state (coarse has none; refined
  keeps its own H_error untouched).

### 3.2 Cross-algorithm transfer — three key questions

**(a) W coefficient format compatibility** ✓

AEC3: both filters are `AdaptiveFirFilter` instances. W format
(FD complex, partition-indexed) is identical. Only the per-block
`G[k]` computation (refined gain module vs coarse gain module)
differs. **Copying W across them is structurally safe.**

Our system: same — `PBFDKF` inherits from `PBFDAF`; W storage
identical. `copy_weights_from` (L782, L980) copies W bit-for-bit.

**(b) Kalman state — how to handle when copy direction involves
a Kalman filter on the receiving side**

AEC3: does not face this. Coarse receives, coarse has no P. The
direction refined→coarse is **Kalman→NLMS**, where "accept the W,
forget P" is the natural reading.

Our system (PBFDKF-both): both sides have P. `copy_weights_from`
on PBFDKF (L980) copies W only and **explicitly does not copy
P/Q**, documented rationale at L980-984:
> Copy W only. P/Q are confidence states the destination has
> accumulated from its own learning history — copying them
> contaminates the Kalman gain. _error_psd/R left alone for the
> same reason. After W copy, P may temporarily mismatch but Kalman
> self-corrects within a few frames thanks to the per-frame
> _error_psd EMA.

This is a **deliberate design choice**: keep destination's own P
so its self-corrective denominator logic applies to the newly
copied W.

**Future question (PBFDKF main + PBFDAF shadow experiment)**: if
shadow (NLMS) outperforms main (Kalman), shadow → main copy:
- Main's W is replaced by shadow's W.
- Main's P is destination's own → **option A keep**: accept
  W/P mismatch, let Kalman's self-correction absorb (matches our
  current L980 logic, fine).
- **Option B reset P**: fill `P_INIT` on the assumption that W is
  new and uncertainty is large. More conservative.
- **Option C ramp-up**: scale P by 2×–10× temporarily, let it
  decay over N blocks. Most defensive.

Recommendation for the future experiment: **Option A first** (keep
existing L980 semantics). Add Option B/C as fallback if smoke test
shows P mismatch causes instability. Align with AEC3 philosophy of
"keep destination's state history, let self-correction converge".

**(c) Post-copy hangover — vicious-cycle prevention**

AEC3 (`subtractor.cc`):
```cpp
const bool disallow_leakage_diverged =
    coarse_filter_reset_hangover_[ch] > 0;
refined_gains_[ch]->Compute(..., disallow_leakage_diverged, ...);
```
For `coarse_reset_hangover_blocks` blocks after a coarse reset, the
refined gain module is forbidden from entering the
`leakage_diverged` branch (high-Q open-up). Reason: immediately
after coarse is reset to refined's W, `e_coarse` will briefly look
better than `e_refined` (identical adaptation from identical W,
with coarse's simpler NLMS often producing lower transient error),
which would otherwise fire the diverged-leakage branch, over-adapt
refined, which would cause real coarse divergence, triggering
another coarse reset → loop.

**Default value of `coarse_reset_hangover_blocks`**: configurable in
`EchoCanceller3Config`, default tbd (not fetched in this session —
see §9).

Our system: **no equivalent**. Copy hysteresis (L3146-3147: 3
consecutive frames `shadow_err < main_err × 0.65` AND 10-frame
streak) prevents trigger oscillation, but **no post-copy
inhibition** of Q injection or adaptation acceleration. Potential
vicious cycle not observed in practice because our PBFDKF-PBFDKF
shadow rarely satisfies the 0.65 threshold anyway (Kalman
saturation keeps `shadow_err ≈ main_err`). If we switch shadow to
NLMS, copy events will become more frequent, and this gap may
manifest.

### 3.3 Our system's copy mechanics (observed)

Code: `aec.py:3107-3158`.

- **Bidirectional**: both `shadow → main` and `main → shadow`
  trigger on symmetric `< threshold × 0.65`.
- Trigger: `shadow_frame_count >= 50` (warmup) AND `far_active` AND
  `error_is_normal` (`main_err < _copy_err_baseline × 4`) AND
  `not epc_active` AND `_saturation_level < 0.3` AND `threshold`
  comparison AND hysteresis 3 AND streak 10.
- No size change (partitions fixed at construction).
- Uses `copy_weights_from`: W copied; PBFDKF path explicitly
  leaves P/Q untouched (per L980-984 rationale).

### 3.4 Comparison table

| Aspect | AEC3 | Our system | Gap |
|---|---|---|---|
| Direction | refined → coarse (single) | bidirectional | design choice |
| Trigger | 5 consecutive blocks `e_ref < e_coarse` | `shadow_err < main_err × 0.65` + 3-frame hyst + 10-frame streak + 5 guards | different semantic |
| Size change | `SetFilter` can resize | size fixed at construction | minor, not needed for current use |
| Kalman P on receiver | N/A (coarse has no P) | P kept (L980 doc) | fine for PBFDKF-PBFDKF; decision pending for mixed |
| Post-copy hangover | `disallow_leakage_diverged` for `coarse_reset_hangover_blocks` | none | **🟡 gap; defer until shadow-as-NLMS to materialise** |
| Cross-algorithm copy | refined→coarse is Kalman→NLMS | N/A (same class) | will matter for shadow-as-NLMS |

### 3.5 Copy strategy decision tree for shadow-as-NLMS experiment

Future scope: PBFDKF main + PBFDAF shadow. Decision tree (non-
binding, to be finalised in spec for that experiment):

```
Event: shadow_err < main_err × threshold, sustained N frames

(1) Direction: shadow → main only? (AEC3-aligned, simpler)
    vs bidirectional (current)?
    → Recommend single direction (shadow → main only).
      Main → shadow is redundant: shadow has no state worth
      preserving (NLMS), and shadow's W is used only as a
      sanity check; nothing breaks if it trails main temporarily.

(2) On shadow → main copy:
    main.W := shadow.W  (straight copy, formats match)
    main.P := ?         (Option A keep | B reset to P_INIT | C ramp-up)
    → Recommend A keep; add post-copy hangover (guard against
      Option-A contamination cycle).

(3) Post-copy hangover:
    for N blocks after copy:
      suppress Q_high injection (keep q_scale at steady-state min)
      OR
      forbid further shadow → main copies
    → Recommend the latter (simpler, matches AEC3 intent).
```

---

## 4. Reset taxonomy (theme 2)

### 4.1 AEC3 has 5 distinct reset granularities

#### 4.1.1 Delay-change full reset

**Trigger** (from `block_processor.cc`):
- `render_event_ == kRenderOverrun` → `DelayAdjustment::kBufferFlush`
- `delay_controller_->GetDelay()` returns new delay AND
  `AlignFromDelay` returns true → `kNewDetectedDelay`

**Propagation**: `EchoPathVariability{gain_change, kBufferFlush |
kNewDetectedDelay, clock_drift}` → `echo_remover_->ProcessCapture`
→ `aec_state_.HandleEchoPathChange(variability)`.

**Action set** (from `aec_state.cc` on non-kNone delay):
```
refined_filter.HandleEchoPathChange()       # internal filter clear
coarse_filter.HandleEchoPathChange()
refined_gains.HandleEchoPathChange()
    → H_error_.fill(10000)                   # P → initial huge
    → poor_excitation_counter → 1000
    → call_counter → 0
coarse_filter.SetSizePartitions(initial)    # shrink back to short
filter_analyzer_.Reset()                     # impulse-response analysis
erle_estimator_.Reset(delay_change=true)     # + blocks_since_reset=0
erl_estimator_.Reset()                       # blocks_since_reset=0
filter_quality_state_.Reset()
initial_state_.Reset()                       # cold-start resumes
transparent_state_->Reset()                  # HMM / legacy state
subtractor_output_analyzer_.HandleEchoPathChange()
```

Total: ~10 subsystems reset. "Nuke from orbit."

#### 4.1.2 Gain-change partial reset (ERLE-only)

**Trigger**: `EchoPathVariability.gain_change == true` passed into
`ProcessCapture` from outside (`echo_path_gain_change` param,
upstream detection).

**Action**:
```
aec_state: erle_estimator_.Reset(delay_change=false)
           → subsystems reset, blocks_since_reset preserved
refined_filter_update_gain.HandleEchoPathChange(gain_change=true):
  TODO (bugs.webrtc.org/9526): Handle gain changes.
  // Currently no H_error reinitialization on gain_change.
  poor_excitation_counter and call_counter also not reset in this branch.
```

This is the key asymmetry: gain-change is **detected** and flagged,
but the filter-side response is an **acknowledged TODO** upstream.
ERLE is reset; filter W/P/Q continue as if nothing happened.
Design rationale (from context): gain change does not alter echo
path shape; W remains valid. Only the ratio-based metrics (ERLE)
need recalibration.

#### 4.1.3 Coarse-from-refined reset (+hangover)

Covered in §3.1-3.2. This is a **reset of one filter only**, not
a global reset. It is the "shadow got too wrong, snap back to main"
mechanism.

#### 4.1.4 Refined filter W scaling (misadjustment recalibration)

**Trigger**: `FilterMisadjustmentEstimator` in `subtractor.cc`
monitors `e²_refined / y²` smoothed over `n_blocks`, tracks
`inv_misadjustment_` via asymmetric EMA:
```
if (update < inv_misadjustment_) OR (overhang_ > 0):
    inv_misadjustment_ += 0.1 · (update - inv_misadjustment_)
```
When `IsAdjustmentNeeded()`:
```
scale = GetMisadjustment()
refined_filter->ScaleFilter(scale)       # W *= scale (all partitions)
impulse_response *= scale                 # external copy
ScaleFilterOutput(y, scale, e, s)         # rescale current block's outputs
```

This is **not a reset**, it is a **recalibration**. W keeps shape
and phase; only magnitude is adjusted. It runs continuously (no
hangover) and does not touch P/Q/H_error.

**Our system has no equivalent.** This is gap #5 below.

#### 4.1.5 Initial-state transition reset

**Trigger** (`aec_state.cc`):
- `initial_state_ = strong_render_blocks < 5s` (conservative) or
  `< 2.5s` (default)
- `transition_triggered_ = !initial_state_ && prev_initial_state`
  (fires for one block when the flag flips)

**Action**: `erle_estimator_.Reset(delay_change=false)` + signals
downstream (e.g. `subtractor` exits initial state, `suppression_gain`
resets to `SetInitialState(false)` per `echo_remover.cc`).

**Purpose**: during cold-start, ERLE estimates are unreliable (filter
not yet converged, render buffer noise). Discard startup-phase ERLE
on transition to steady state, start fresh.

### 4.2 AEC3 reset orchestration call graph

```
external caller (audio_processing_impl)
  → BlockProcessor::ProcessCapture(echo_path_gain_change=bool)
      constructs: EchoPathVariability(gain_change, delay_change)
      delay_change detected here from render_event / delay_controller
      passes → echo_remover_->ProcessCapture(variability)
          order of ops (per echo_remover.cc):
          (1) subtractor->Process(variability)
                subtractor.cc: if variability.delay_change != kNone:
                               refined_gain.HandleEchoPathChange(variability)
                               coarse_gain.HandleEchoPathChange(variability)
                               etc.
          (2) aec_state_.Update(...)
                aec_state.cc: if variability.AudioPathChanged():
                              HandleEchoPathChange(variability)
          (3) if aec_state_.TransitionTriggered():
                suppression_gain->SetInitialState(false)
                subtractor->ExitInitialState()
          (4) suppression_gain_->Compute(...)
          (5) suppression_filter_->ApplyGain(...)
```

### 4.3 Our EPC 3-tier vs AEC3 5-type

#### 4.3.1 EPC none

No action. Covers the "nothing happened" case. No AEC3 analog (AEC3
simply does not call HandleEchoPathChange).

#### 4.3.2 EPC small (`_epc_level = 'small'`)

Code: `aec.py:3207-3254`. Trigger: `shadow_adv >
epc_small_shadow_adv (1.3)` AND sustained guards.

Action (read from code):
```
self.epc_active = True
self._epc_level = 'small'
self.epc_hangover_count = max(..., small_hangover)
```
Downstream effect (L3208: coherence confidence × 0.7 reduction;
L3424 raw_dt = 0.0 during epc_active; mu_scale handling elsewhere).

**Filter W / P / Q untouched.** Semantic close to AEC3 gain-change
partial — but we reduce coherence confidence, not ERLE.

#### 4.3.3 EPC large (`_epc_level = 'large'`)

Trigger: `shadow_adv > epc_large_shadow_adv (2.0)` AND
`epc_magnitude > epc_large_total_rise (3.0)`.

Action (read from L3203-3220):
```
self._epc_level = 'large'
self.dtd_coherence.confidence *= 0.3
self.epc_hangover_count = self.config.epc_hangover
self.epc_active = True
```
Plus L2957-2958 on delay shift:
```
self.epc_active = True
self.epc_hangover_count = epc_hangover
for filt in [self.filter, self.shadow_filter]:
    if hasattr(filt, 'Q'):
        filt.Q = filt.Q_high.copy()               # Q reset to Q_high
        filt._p_max_override = np.float32(1.0)    # P_MAX stretch
```
Plus P floor reset (L934 area): `P_floor = Q_high × _p_floor_beta`
with `beta=P_FLOOR_BETA_RESET=1.0` during transient (vs default 0.1).

**Effect**: filter W stays, P bounds loosen, Q jumps back to Q_high,
R reset (via `_error_psd × R_scale_smooth` path). Approximates AEC3
delay-change-full for W-preservation, but NO `coarse.SetSizePartitions`
shrink, NO `filter_analyzer.Reset`, NO `initial_state.Reset`.

### 4.4 Reset-granularity gap analysis

| # | AEC3 reset type | Our closest | Gap | Priority |
|---|---|---|---|---|
| R1 | Delay full (10+ subsystems) | `EPC large` on delay shift (partial) | no filter_analyzer/initial_state reset; no coarse size shrink | 🟡 medium |
| R2 | Gain partial (ERLE-only) | `EPC small` (coherence confidence reduction) | different metric; gain_change itself undetected upstream | 🟡 medium |
| R3 | Coarse-from-refined + hangover | `shadow copy` bidirectional, no hangover | direction + hangover missing | 🟡 medium (materialises w/ shadow-as-NLMS) |
| R4 | Misadjustment W scaling | **none** | entirely missing | 🔴 **high** (Gap #5) |
| R5 | Initial-transition reset | blocks_since_init tracked, no flipped-edge reset action | minor | 🟢 low |

---

## 5. 15-gap comprehensive comparison

| # | Feature | AEC3 implementation | Our implementation | Severity | PZ7V impact | Other |
|---|---|---|---|---|---|---|
| 1 | **Dual-filter same algorithm** | refined Kalman + coarse NLMS (distinct algos) | shadow = same class as main (PBFDKF on PBFDKF) | 🔴 | root cause of `shadow_adv≈1.0` at onset | affects all EPC reliability |
| 2 | **Shadow algorithm** | NLMS coarse w/ fixed `rate/X²` | Kalman w/ Q×3.5 | 🔴 | same as #1 | same |
| 3 | **Shadow copy direction** | refined→coarse single | bidirectional | 🟡 | minor | irrelevant if both Kalman |
| 4 | **Post-copy hangover** | `disallow_leakage_diverged` for N blocks | none | 🟡 | current: N/A; future: needed for NLMS shadow | post shadow-as-NLMS |
| 5 | **Filter W misadjustment scaling** | `ScaleFilter(scale)` per `FilterMisadjustmentEstimator` | none | 🔴 | partial recovery mechanism for W drift/clipping aftermath; could close part of PZ7V 4.6dB gap | DT stability too |
| 6 | **Dynamic Q leakage bifurcation** | `leakage_converged` vs `leakage_diverged` branch | single `q_scale` via mu_mean | 🟡 | no "reopen learning when shadow says main is wrong" | smooth adaptation |
| 7 | **Reset taxonomy granularity** | 5 distinct types | 3-tier EPC | 🟡 | EPC-large conflates R1 + partial R3 | maintenance |
| 8 | **Initial state transition** | edge-triggered flip → flush ERLE | blocks_since_init passive | 🟢 | negligible | cold-start cases |
| 9 | **Usable-linear gate** | `FilteringQualityAnalyzer`: sufficient_data_startup (0.4s active render) + sufficient_data_reset (0.2s) + convergence_seen + !transparent | `_filter_converged` bool only | 🔴 | protects suppression against using garbage linear output | transparent/dropout edge cases |
| 10 | **Transparent mode** | HMM classifier (+ legacy), reduces suppression for headsets | none | 🟡 | headset / loop-back minimal-echo cases | user-experience |
| 11 | **Filter analyzer (peak, gain, HP pre-filter)** | high-pass cutoff ~600Hz, peak detection, consistent estimate, delay inference | ad hoc (no centralised analyzer) | 🟡 | supports #9; no HP pre-filter → low-freq bias in filter analysis | medium |
| 12 | **Gain-change detection source** | external input (`echo_path_gain_change` param to ProcessCapture) | inferred via shadow_adv (which fails at onset in our Kalman-twin setup) | 🟡 | could be extracted from new detector (e.g. B-16 raw_dt-jump signal) | integration |
| 13 | **Misadjustment estimator** | tracks `e²/y²` asymmetric EMA, IsAdjustmentNeeded check | none | 🔴 (supports #5) | long-term drift correction | medium-long term |
| 14 | **Stationarity estimator (render side)** | dedicated `stationarity_estimator.cc` | `_is_stationary_far` based on CV² | 🟢 | equivalent, different impl | none |
| 15 | **Subband nearend detector / dominant nearend** | dedicated modules | `dtd_coherence.confidence + dtd_divergence` | 🟢 | functionally adequate | none |

Severity legend: 🔴 high (structural, blocks progress), 🟡 medium
(worthwhile improvement), 🟢 low (corner case or different impl of
same idea).

---

## 6. Priority experiments (4 high)

### 6.1 Shadow-as-NLMS (Gap #1 + #2, the root cause)

**Why**: Kalman-on-Kalman shadow degenerates to ~1.0 advantage at
echo events because Kalman `K` self-normalises when `Σ P|X|²`
dominates R. NLMS `mu = rate/X²` has no such pinning.

**AEC3 reference**: `coarse_filter_update_gain.cc`, role
documented throughout `subtractor.cc` and `echo_remover.cc`.

**Our current code**:
- `aec.py:2525-2555` — shadow construction, same `FilterClass` as
  main.
- `aec.py:980-984` — `copy_weights_from` PBFDKF path, W-only
  transfer rationale.

**Proposed change (sketch, not implemented)**:
- In `AEC.__init__`, when `config.use_kalman` (PBFDKF main) and
  `config.enable_shadow`:
  - Construct shadow as `PBFDAF(**shadow_kwargs_without_kalman)`.
  - Introduce `shadow_mu` config (plausible start: 0.3-0.5 matching
    AEC3 coarse rate; needs per-case smoke test).
  - Drop `shadow_q_ratio` (irrelevant for NLMS shadow).
- `copy_weights_from` PBFDKF side: accept source is PBFDAF,
  W-only copy already matches; add assert that source W shape
  matches.
- Copy gate: keep existing direction bidirectional for phase-1,
  add post-copy hangover (Gap #4, §3.2c).

**Expected impact**:
- `shadow_adv > 1.3` will trigger at echo-path-gain events because
  NLMS shadow responds within 1-2 blocks vs refined's multi-block
  Kalman settle.
- EPC small/large guards operate as originally designed.
- PZ7V leak expected to reduce from −20.42 dB (B-16 alone) toward
  target −25 dB, with the remaining gap closed by filter
  misadjustment (§6.2).

**Risk**:
- Shadow PBFDAF will have different R self-regulation
  (`_R = _error_psd` EMA) vs PBFDKF Kalman denominator. May
  require `alpha_r_scale` / `leak` tuning (current shadow_kwargs
  already sets `alpha_r_scale=1.03, leak=1e-6` per L2547-2548 for
  the PBFDKF-shadow case; need to verify these apply cleanly when
  the shadow is a real PBFDAF).
- Copy gate may fire more frequently (NLMS shadow diverges/
  reconverges faster). Without post-copy hangover (Gap #4), risk
  of cycle.
- Full 800-case benchmark required; cannot piggyback on B-16
  smoke.

**Branch name**: `experiment/shadow-as-nlms`

**Dependencies**:
- B-16 merged first (keeps veto as defensive layer).
- Spec `docs/spec_shadow_nlms_refactor.md` with strict §5
  invariants analogous to B-16.
- Gap #4 (post-copy hangover) folded into this experiment
  because it only manifests with NLMS shadow.

### 6.2 Filter misadjustment W scaling (Gap #5 + #13)

**Why**: AEC3's `FilterMisadjustmentEstimator` → `ScaleFilter`
is a continuous recalibration that corrects long-term W magnitude
drift (e.g. caused by soft-clipping upstream, ERL tracker
mismatch, DT-partial adaptation). Over many frames the refined
filter can accumulate a ~1-2 dB magnitude offset without ever
triggering reset logic; misadjustment scaling fixes this.

**AEC3 reference**: `subtractor.cc` —
`FilterMisadjustmentEstimator` block.

**Our current code**: no equivalent. P clamping (L968-970) bounds
P but does not touch W magnitude. EPC-large sets P_FLOOR_BETA_RESET
but leaves W alone.

**Proposed change (sketch)**:
- New state: `_misadjustment_smoothed`, update via asymmetric EMA
  on `|e|² / |y|²` (one decay rate when shrinking, overhang guard
  when expanding).
- Trigger condition: equivalent of AEC3's `IsAdjustmentNeeded`
  (smoothed misadjustment crosses configured ratio AND signal
  stable).
- Action: `self.filter.W *= scale`, `self.filter.P *= scale²`
  (Option A) OR leave P alone (Option B). Need to decide; AEC3
  doesn't touch its P analogue (H_error) during ScaleFilter.

**Expected impact**:
- Closes long-term accumulation drift that B-16 and
  shadow-as-NLMS do not address.
- On PZ7V specifically: expected marginal (the event is a single
  gain jump, not long-term drift), but on recordings with
  repeated soft-clips or slow ERL shifts, should measurably
  improve.

**Risk**:
- ScaleFilter is a "trust me" operation — if the estimator is
  wrong, we scramble W wholesale. Extensive smoke test required.
- Misadjustment estimator itself is non-trivial state machine.

**Branch name**: `experiment/filter-misadjustment`

**Dependencies**: independent of shadow-as-NLMS (operates on
refined/main W only). Can be parallel but safer sequential after
shadow-as-NLMS so benchmark baseline is stable.

### 6.3 Usable-linear gate (Gap #9)

**Why**: our suppression stage reads `self.filter.echo_spec` and
uses it as linear-model echo estimate unconditionally after
`_filter_converged`. AEC3 gates this with a layered sanity check
(`FilteringQualityAnalyzer.usable_linear`) that requires multiple
orthogonal conditions simultaneously. When any fails, AEC3 falls
back to non-linear suppression only.

**AEC3 reference**: `aec_state.cc` — `FilteringQualityAnalyzer`:
```
usable_linear =
    sufficient_data_startup (>0.4s active render)
    && sufficient_data_reset (>0.2s since reset)
    && (external_delay || convergence_seen)
    && !transparent_mode
```

**Our current code**: `_filter_converged` is a single bool that
latches on 10 consecutive ERLE>5dB frames and stays True until
reset. `self.filter.echo_spec` is consumed wherever linear output
is needed, with no "quality" gate.

**Proposed change (sketch)**:
- New state: `_usable_linear = self._filter_converged AND
  blocks_since_reset > N_reset_blocks AND not transparent AND
  convergence_stable`
- Downstream consumers (e.g. RES linear echo estimate, ENR
  calculation) use `_usable_linear` instead of `_filter_converged`
  as the gate.
- When `_usable_linear == False`, downstream falls back to
  non-linear / energy-based estimates.

**Expected impact**:
- Protection during filter-unreliable regimes (post-EPC recovery,
  transparent-mode-like headset scenarios).
- Should reduce "filter insists it's fine but downstream
  exploiting wrong linear estimate" artefacts. Hard to localise
  without specific regression probe.

**Risk**:
- Potentially conservative: if `_usable_linear` trips too
  aggressively, we sacrifice linear-AEC's quality advantage.
- Requires careful tuning of N_reset_blocks and convergence_stable.

**Branch name**: `experiment/usable-linear-gate`

**Dependencies**: depends on Gap #13 (filter analyzer) for some
components (`convergence_seen`), so **sequence #13 before #9** or
combine into one branch (`experiment/filter-quality-gating`).

### 6.4 Filter analyzer (Gap #11, supports #9)

**Why**: AEC3's `filter_analyzer.cc` centralises impulse-response
analysis: high-pass pre-filter (~600 Hz cutoff) to remove DC bias,
peak detection, gain tracking (monotonic-during-startup ratchet),
consistent-estimate detector for `convergence_seen`.

**AEC3 reference**: `filter_analyzer.cc`.

**Our current code**: ad-hoc — ERL tracking is fragmented across
`_filter_erle_est`, `_fb_erle_est`, `_mic_far_tracker`,
`_filter_converged`. No single point of truth.

**Proposed change (sketch)**: introduce a `FilterAnalyzer`
helper class that produces:
- `peak_index`, `filter_delay_blocks`
- `filter_gain` (with monotonic startup rule)
- `consistent_estimate` bool (peak delay stable for 1.5s active
  render)

Used by `_usable_linear` and future code. No behavioural change
on own; pairs with #9.

**Expected impact**: **no direct metric improvement**. Refactor
enabler.

**Risk**: large refactor surface; regression risk from
consolidating existing ad-hoc tracking.

**Branch name**: `experiment/filter-analyzer` (combine with #9 as
`experiment/filter-quality-gating`).

---

## 7. Medium and low priority gaps (non-experiment track)

### Medium (🟡)

- **Gap #3** (copy direction): only matters if we abandon
  bidirectional. Fold into shadow-as-NLMS experiment choice.
- **Gap #4** (post-copy hangover): fold into shadow-as-NLMS.
- **Gap #6** (dynamic Q bifurcation): interesting on its own but
  correlated with shadow-as-NLMS. Experiment branch idea:
  `experiment/leakage-bifurcation`; defer.
- **Gap #7** (reset taxonomy rationalisation): refactor quality
  improvement. Low direct ROI.
- **Gap #10** (transparent mode): user-experience; relevant for
  headset scenarios we're not benchmarking yet. Defer.
- **Gap #12** (gain-change detection source): B-16's sustained
  raw_dt-jump signal is a natural candidate for an explicit
  `gain_change` flag we set on our `EchoPathVariability`
  analogue. Low cost once we have #6.

### Low (🟢)

- **Gap #8** (initial-state flipped-edge reset): marginal,
  accept.
- **Gap #14** (stationarity estimator): functionally adequate.
- **Gap #15** (subband nearend detector): functionally adequate.

---

## 8. Implementation roadmap

Sequencing principle: **never compound experimental architecture
changes**. Merge-by-merge, one diffable change at a time.

### Phase 1 (in progress) — B-16

- Branch: `feature/b16-raw-dt-jump-veto` (current HEAD).
- Stage 1D 800-case benchmark running (task `bkb9ki965`, ~78% at
  this writing).
- Merge decision per §7-§9 of `spec_b16_raw_dt_jump_veto.md`.
- If MERGE: keep as defensive layer throughout remaining phases.
- If ABANDON: diagnose, no code lost (AEC_FIX_B16=0 default).

### Phase 2 (1-2 weeks) — Shadow-as-NLMS

- Branch: `experiment/shadow-as-nlms` (off post-B-16 main).
- Spec: `docs/spec_shadow_nlms_refactor.md`. Must include:
  - Option A/B/C decision for `main.P` on shadow→main copy
  - Post-copy hangover mechanism (Gap #4 bundled)
  - Feature flag `AEC_EXP_SHADOW_NLMS` (default OFF)
- Smoke test: B-16 Stage-1C 7-case battery (PZ7V + 4 DT + 2
  FS-non-onset).
- Full 800-case benchmark. **Stricter criteria** than B-16: hard
  floor no regression > 0.03 dB per scenario.
- Merge if ΔFS_echo ≥ 0.10 dB AND all other metrics flat/
  positive.

### Phase 3 (1-2 weeks) — Filter misadjustment

- Branch: `experiment/filter-misadjustment`.
- Spec: `docs/spec_filter_misadjustment.md`.
- Feature flag `AEC_EXP_MISADJ` (default OFF).
- Smoke test + 800-case benchmark; merge criteria as Phase 2.

### Phase 4 (2-3 weeks) — Filter-quality gating (combines Gaps #11 + #9)

- Branch: `experiment/filter-quality-gating`.
- Refactor scope; split into:
  - Sub-phase 4a: introduce FilterAnalyzer class (Gap #11), no
    behaviour change.
  - Sub-phase 4b: introduce `_usable_linear` gate (Gap #9),
    consume FilterAnalyzer.
- Longer benchmark window; likely requires diagnostic deep-dive
  on specific regressing cases.

### Phase 5 (opportunistic) — Medium gaps as benchmark drives

- Gap #6 (leakage bifurcation): if Phase 2+3 merge but PZ7V-like
  still regresses elsewhere.
- Gap #12 (gain-change detection integration): lightweight
  integration of B-16's signal into a formal `EchoPathVariability`
  flag.
- Gap #7 (reset taxonomy): refactor quality, defer until a good
  window (no simultaneous experiments).

### Phase 6 (defer) — Low gaps

- Gap #10 (transparent mode): when headset / loop-back metrics
  are added to benchmark.
- Gap #8, #14, #15: accept current.

---

## 9. Known limitations of this analysis

### 9.1 Not fetched (should fetch before acting on corresponding gaps)

- `echo_canceller3_config.cc` — config defaults (e.g.
  `coarse_reset_hangover_blocks` actual value, all referenced
  "configurable, default tbd" in this doc).
- `suppression_gain.cc` — RES counterpart.
- `residual_echo_estimator.cc` — residual echo estimate.
- `stationarity_estimator.cc` — render stationarity (our Gap #14
  counterpart).
- `subband_nearend_detector.cc`, `dominant_nearend_detector.cc` —
  DTD counterparts (Gap #15).
- `reverb_model.cc`, `reverb_decay_estimator.cc` — reverb.
- `matched_filter.cc` — delay estimation internals.
- `render_signal_analyzer.cc` — render signal analysis.
- `signal_dependent_erle_estimator.cc` — ERLE details (mentioned
  by `erle_estimator.cc` Reset).
- Second half of `adaptive_fir_filter.cc` — implementations of
  `SetFilter`, `ScaleFilter`, `Reset` (method signatures in .h
  captured in §2.1 table; implementations not inspected).

### 9.2 Inferences made without full source

- `coarse_reset_hangover_blocks` default: "typical ~100ms"
  repeated in some analyses — unverified from source.
- `filter_analyzer` high-pass coefficients quoted from
  user's prior analysis: `[0.7929742, -0.36072128, -0.47047766]`
  — not independently verified in this session.
- AEC3's internal relationship between `gain_change` detection
  upstream and `BlockProcessorImpl::ProcessCapture`'s
  `echo_path_gain_change` parameter is via the caller
  `audio_processing_impl`; not traced.

### 9.3 Our aec.py code paths not exhaustively traced for gaps

- We read `copy_weights_from`, `_update_weights`, EPC dispatch
  (L3203+), basic shadow construction. We did not trace all 11
  `reset()` implementations across the 13 grep-identified classes
  (filter, RES, DTD, etc.) to see which have AEC3 analogs.
- `reset()` sweep and per-class AEC3 mapping would be a useful
  follow-up.

### 9.4 Benchmark correlation

At writing, B-16 Stage 1D benchmark is in progress. The 800-case
metric delta will either corroborate or refute the §6.1 prediction
("B-16 alone ~−6 dB on PZ7V, shadow-as-NLMS needed for full
recovery"). Update §6 phase expectations once Stage 1D lands.

---

## 10. References

### AEC3 source URLs (fetched 2026-04-22, branch `refs/heads/main`)

Fetched and analysed in this doc:

- refined_filter_update_gain.cc: https://webrtc.googlesource.com/src/+/refs/heads/main/modules/audio_processing/aec3/refined_filter_update_gain.cc
- coarse_filter_update_gain.cc: https://webrtc.googlesource.com/src/+/refs/heads/main/modules/audio_processing/aec3/coarse_filter_update_gain.cc
- subtractor.cc: https://webrtc.googlesource.com/src/+/refs/heads/main/modules/audio_processing/aec3/subtractor.cc
- subtractor_output_analyzer.cc: https://webrtc.googlesource.com/src/+/refs/heads/main/modules/audio_processing/aec3/subtractor_output_analyzer.cc
- aec_state.cc: https://webrtc.googlesource.com/src/+/refs/heads/main/modules/audio_processing/aec3/aec_state.cc
- filter_analyzer.cc: https://webrtc.googlesource.com/src/+/refs/heads/main/modules/audio_processing/aec3/filter_analyzer.cc
- adaptive_fir_filter.h: https://webrtc.googlesource.com/src/+/refs/heads/main/modules/audio_processing/aec3/adaptive_fir_filter.h
- adaptive_fir_filter.cc (partial): https://webrtc.googlesource.com/src/+/refs/heads/main/modules/audio_processing/aec3/adaptive_fir_filter.cc
- echo_path_variability.cc: https://webrtc.googlesource.com/src/+/refs/heads/main/modules/audio_processing/aec3/echo_path_variability.cc
- erle_estimator.{h,cc}: https://webrtc.googlesource.com/src/+/refs/heads/main/modules/audio_processing/aec3/erle_estimator.h, .cc
- erl_estimator.cc: https://webrtc.googlesource.com/src/+/refs/heads/main/modules/audio_processing/aec3/erl_estimator.cc
- block_processor.cc: https://webrtc.googlesource.com/src/+/refs/heads/main/modules/audio_processing/aec3/block_processor.cc
- echo_remover.cc: https://webrtc.googlesource.com/src/+/refs/heads/main/modules/audio_processing/aec3/echo_remover.cc
- transparent_mode.cc: https://webrtc.googlesource.com/src/+/refs/heads/main/modules/audio_processing/aec3/transparent_mode.cc
- render_delay_controller.cc: https://webrtc.googlesource.com/src/+/refs/heads/main/modules/audio_processing/aec3/render_delay_controller.cc

### Internal references

- `docs/spec_b16_raw_dt_jump_veto.md` — current veto layer (Phase 1).
- `docs/aec3_architecture_alignment.md` — seed note for this doc.
- `docs/b15_superseded.md` — historical B-15 failed-attempt record.
- `python/aec.py` — our implementation (specific line refs
  throughout sections).
- `experiment/pbfdkf-no-epc-baseline` branch — bit-exact evidence
  that EPC is cosmetic at PZ7V onset (Gap #1 empirical support).

### Internal code line anchors (grep verified 2026-04-22)

- Shadow construction: `aec.py:2525-2555`
- Shadow Q assignment: `aec.py:2552-2555`
- Copy gate orchestration: `aec.py:3100-3158`
- PBFDKF `copy_weights_from` rationale: `aec.py:980-984`
- PBFDAF `copy_weights_from` (shared W+R): `aec.py:782-786`
- EPC level dispatch: `aec.py:3203-3254`
- EPC delay-shift trigger: `aec.py:2957-2962`
- PBFDKF Kalman update (saturating denominator): `aec.py:877-971`
- PBFDKF state init: `aec.py:829-862`
- PBFDAF state init: `aec.py:613-650`

---

## 11. Changelog

- 2026-04-22 (initial): first version. 6 core AEC3 modules
  analysed from earlier context + 8 more fetched this session
  (adaptive_fir_filter.h, erle_estimator.{h,cc}, erl_estimator.cc,
  refined_filter_update_gain.cc HandleEchoPathChange,
  block_processor.cc, echo_remover.cc, transparent_mode.cc,
  render_delay_controller.cc partial). 15-gap table compiled.
  4 high-priority experiments queued.
