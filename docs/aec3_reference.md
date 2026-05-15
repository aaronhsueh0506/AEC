# AEC3 (WebRTC) Reference Notes

Consolidated reference for WebRTC AEC3 architecture/algorithm choices, used as design comparison for our PBFDKF-based AEC.

---

## 1. Architectural alignment with our codebase

# AEC3 architecture alignment — shadow algorithm mismatch

Date: 2026-04-22
Status: forward-looking note. **Not scheduled work.** Follow-up after
B-16 merge decision.

## 1. WebRTC AEC3 mainline dual-filter architecture

AEC3 uses **two filters of different algorithmic classes**:

### 1.1 Refined filter (the output-producing main filter)

- Source:
  https://webrtc.googlesource.com/src/+/refs/heads/main/modules/audio_processing/aec3/refined_filter_update_gain.cc
- **Algorithm: Kalman-like gain update**
  ```
  mu[k] = H_error[k] / (H_error[k]·X²[k] + n·E²[k])
  P_update → H_error[k] evolves per-bin
  ```
- Slow, high-quality adaptation. Corresponds to the `refined` role:
  stable output, careful learning.

### 1.2 Coarse filter (the shadow / backup tracker)

- Source:
  https://webrtc.googlesource.com/src/+/refs/heads/main/modules/audio_processing/aec3/coarse_filter_update_gain.cc
- **Algorithm: NLMS with fixed rate**
  ```
  mu = rate / X²
  ```
- Fast, simpler adaptation. Corresponds to the `coarse` role:
  responsive to changes (echo path shifts, gain jumps), copy W to
  refined when outperforming.

**Key design point**: the two filters are NOT the same class with
different hyper-parameters. They use **fundamentally different
update mechanisms** — Kalman for stability, NLMS for responsiveness.

## 2. Current system (this repo, PBFDKF mode)

Code reference: `python/aec.py:2548-2554`:

```python
self.shadow_filter = FilterClass(**shadow_kwargs)   # same class as main
...
if self.config.use_kalman:
    self.shadow_filter.Q_high = self.filter.Q_high * self.config.shadow_q_ratio
    self.shadow_filter.Q_low  = self.filter.Q_low  * self.config.shadow_q_ratio
    self.shadow_filter.Q      = self.shadow_filter.Q_high.copy()
```

- When `AEC_MODE=PBFDKF`, `FilterClass = PBFDKF` and shadow is also
  PBFDKF.
- Only differentiation: `shadow_q_ratio` (default 3.5) scales
  `Q_high`/`Q_low`, i.e. Kalman process-noise variance.
- Both filters follow the **same update equation** (line 877-971);
  they share the same denominator-saturation math.

## 3. Mismatch vs AEC3 (PBFDKF mode)

| Aspect | AEC3 mainline | This repo (PBFDKF mode) |
|---|---|---|
| Main algo | Kalman (refined) | Kalman (PBFDKF) |
| Shadow algo | **NLMS (coarse)** | **Kalman (PBFDKF)** ← mismatch |
| Shadow differentiation | Algorithm class | Q ratio only |
| Onset behaviour | Coarse NLMS responsive | Shadow K-saturates |
| Copy direction | refined → coarse (reset on re-convergence) | shadow → main (when shadow outperforms) |

### 3.1 Why Q-ratio alone fails (Task diagnostic confirmation)

Probed on PZ7V onset (Task 3-4, this branch):

- `Q_ratio = 3.50` exactly (config honoured)
- `P_ratio = 3.50 → 3.71` (grows faithfully with Q)
- **`K_ratio → 1.0`** at echo event: Kalman denominator
  `Σ P·|X|² + R + δ` is dominated by `Σ P·|X|²` term when signal is
  high; that term scales with P in both filters, so ratio cancels.
- Downstream: `shadow_err / main_err ≈ 1.00-1.05` throughout
  PZ7V t=9.9-10.5s (EMA probe result)
- `shadow_adv` never crosses EPC guard `1.3` → EPC never fires at
  onset (confirmed separately: `experiment/pbfdkf-no-epc-baseline`
  shows `AEC_EXP_NO_EPC` yields bit-exact −13.83 dB leak at PZ7V)

The root cause of shadow ineffectiveness in PBFDKF mode is
**structural, not parametric**: Kalman self-normalization neutralises
any P-based differentiation during high-signal regimes, precisely
the regime where EPC-trigger signal is needed.

## 4. Proposed fix (future, separate scope)

**Direction A (minimal)**: make PBFDKF-mode shadow use the existing
PBFDAF (NLMS) class, not PBFDKF:

```python
# Pseudo — not implemented
if self.config.use_kalman:
    # Main is PBFDKF; make shadow PBFDAF to mimic AEC3 coarse
    self.shadow_filter = PBFDAF(**shadow_kwargs_without_kalman)
    # shadow_mu ≈ config.mu (NLMS fixed rate), not Q ratio
```

- Main PBFDKF stays as-is (output quality unchanged)
- Shadow PBFDAF responds to echo-path-gain jumps via NLMS dynamics
  (no K-saturation)
- `shadow_adv` becomes genuine differentiator → EPC guards work
  as originally intended

**Direction B (symmetric)**: apply to PBFDAF mode too — shadow as
something orthogonal (e.g. a lower-mu PBFDAF with different
step-size rule). Lower priority.

## 5. Expected impact

- Shadow actually differentiates at onsets; `shadow_adv > 1.3`
  naturally triggers EPC during PZ7V-class events
- EPC guards (B-4a/4b/12) work as originally designed; no need to
  band-aid raw_dt via B-16
- B-16 still has defensive value (raw_dt bug is structural, EPC
  may not always fire fast enough); keep as belt-and-suspenders
- Potential PZ7V recovery: if shadow-driven EPC fires at onset and
  resets filter Q_high, filter may re-converge faster than B-16's
  460ms veto window → room for > −25 dB target (currently B-16
  alone reaches −20.42 dB)

## 6. Implementation risks

### 6.1 Copy gate compatibility

Current copy gate at `aec.py:3136-3145` uses
`self.filter.copy_weights_from(self.shadow_filter)` which on PBFDKF
copies W only (per line 979-981 `copy_weights_from` comment: "P/Q
are confidence states … copying them contaminates the Kalman gain").

If shadow becomes PBFDAF:
- PBFDAF W is the same FD complex-weight shape as PBFDKF W — copy
  is structurally compatible
- PBFDAF has no Q/P state — nothing to copy, nothing to contaminate
- PBFDKF main's P is learned from its own error → already decoupled
  from shadow state, so no change needed
- PBFDAF `_R` (measurement-noise tracker) is separate from PBFDKF
  `R` which is derived from `_error_psd × R_scale_smooth`; copy
  semantics need explicit review

### 6.2 Shadow step-size tuning

- Current: `shadow_q_ratio=3.5` (irrelevant once shadow is NLMS)
- New: need `shadow_mu` for NLMS shadow. Plausible starting point
  from AEC3 source: `mu = rate / X²` with `rate ≈ 0.3-0.5`
- Requires smoke test on several FS / DT / onset cases before
  800-case

### 6.3 Invariants that must hold

- PBFDAF class is a drop-in instance; main remains PBFDKF
- Main filter output (`self.filter`) is the only one producing
  audio output; shadow is EPC-trigger / copy-gate only
- `enable_dtd=False` path (legacy formula, used by eval) must not
  regress

### 6.4 Benchmark risk

- 800-case Pareto may shift meaningfully. This is a **larger scope**
  than B-16 (architectural, not a veto layer).
- Possible downside: DT/NE cases where current shadow-PBFDKF's
  smoothness actually helped copy-gate stability; PBFDAF shadow
  might over-trigger copies → output artefacts.
- Must be gated behind a flag with full bit-exact fallback path.

## 7. Suggested stage plan (not scheduled)

1. **B-16 Stage 1D** (in progress): finish 800-case benchmark. If
   PASS, merge `feature/b16-raw-dt-jump-veto` → main. If FAIL,
   diagnose / abandon B-16 first.
2. **After B-16 merge decision**: open `experiment/shadow-as-nlms`
   branch off main. Do not combine with B-16 scope.
3. Write `docs/spec_shadow_nlms_refactor.md`. Stage 1A audit must be
   **strict** — this touches core architectural invariants.
4. Smoke test on same 7 cases used in B-16 Stage 1C.
5. Full 800-case benchmark, same methodology as B-16 Stage 1D.
6. Merge decision: SAFE / MARGINAL / UNSAFE per same rubric.

## 8. Confidence and risk posture

**Confidence (+)**: this is explicitly aligning to WebRTC AEC3
mainline, which is a mature open-source reference. Shadow-as-NLMS is
**not an invention**, it is the upstream design. Deviation cost
(current Kalman-shadow) was identified as structural root cause by
two independent probes (shadow_adv plateau, Kalman K-saturation
math). Design direction carries higher prior than typical new-feature
work.

**Risk (−)**: scope is bigger than B-16. Multiple coupling points
(copy gate, tuning, benchmark regression risk) need careful
sequencing. Must not ship under time pressure; Stage 1A audit is
load-bearing.

## 9. References

- `docs/spec_b15_raw_dt_delay_alignment.md` + `docs/b15_superseded.md`
  — historical raw_dt fix attempt, superseded by B-16
- `docs/spec_b16_raw_dt_jump_veto.md` — current B-16 veto layer
- `docs/aec_methods.md §6.4` — internal dual-filter design intent
  (at introduction time, 2026-03-13, shadow was conservative
  mu×0.5 for divergence recovery; role inverted by v1.12.0 to
  aggressive Q×3 for copy-gate trigger)
- `python/aec.py:2525-2554` — shadow filter instantiation
- `python/aec.py:877-971` — PBFDKF `_update_weights` (shared by
  main and current shadow, the saturating denominator is here)
- `experiment/pbfdkf-no-epc-baseline` branch — empirical evidence
  that EPC never fires at PZ7V onset (bit-exact leak with EPC off)

---

## 2. Full architecture analysis

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

### 9.1 Fetch status

**v3.18 Phase 0.1 update (2026-05-15)**: complete AEC3 source tree
pinned at commit `9310b29acd` under
[docs/aec3_extracts/src/aec3/](aec3_extracts/src/aec3/) (127 files,
1.0 MB). All formerly-unfetched modules are now available locally.
Per-module read status:

**Phase 0.2 read (RES gain pipeline)** — see [docs/aec3_residual_pipeline_mapping.md](aec3_residual_pipeline_mapping.md):
- ✓ `suppression_gain.cc` — read 2026-05-15
- ✓ `residual_echo_estimator.cc` — read 2026-05-15

**Phase 0.3 read (NE detectors, Gap #15 revisit)** — see [docs/aec3_subband_ne_detector_mapping.md](aec3_subband_ne_detector_mapping.md):
- ✓ `subband_nearend_detector.cc` — read 2026-05-15
- ✓ `dominant_nearend_detector.cc` — read 2026-05-15

**Phase 0.4 read (reverb)** — see [docs/aec3_reverb_mapping.md](aec3_reverb_mapping.md):
- ✓ `reverb_model.cc` — read 2026-05-15
- ✓ `reverb_decay_estimator.cc` — read 2026-05-15

**Available locally but not yet read** (Phase A/B/C will read on-demand):
- `echo_canceller3_config.cc` — config defaults (e.g.
  `coarse_reset_hangover_blocks` actual value, all referenced
  "configurable, default tbd" in this doc). Phase 0.6 read if hard-bar
  tuning needs defaults.
- `stationarity_estimator.cc` — render stationarity (our Gap #14
  counterpart). Read on-demand if Phase D needs stationarity input.
- `matched_filter.cc` — delay estimation internals. NOT needed (v3.16
  C6 audit H2 closed DelayEst arc).
- `render_signal_analyzer.cc` — render signal analysis. Phase A.5 read
  if shadow NLMS tuning needs render-side context.
- `signal_dependent_erle_estimator.cc` — ERLE details. Phase D read
  if D-α subband detector wires through ERLE.
- Second half of `adaptive_fir_filter.cc` — `SetFilter`, `ScaleFilter`,
  `Reset` implementations. **Phase B will read** (ScaleFilter port).
- `aec_state.cc` second half — `UsableLinearEstimate` /
  `GetReverbFrequencyResponse` / `GetResidualEchoScaling`
  implementations. **Phase C will read**.
- `filter_analyzer.cc` second half — Phase C C.1 will read.

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

## 11. RES-side gap table (v3.18 Phase 0 audit, 2026-05-15)

Compiled from Phase 0.2-0.4 module reads. The v3.13-v3.17 15-gap table
(§5) focuses on linear-filter / shadow / EPC machinery. This table
focuses on **post-linear (RES) machinery** that the prior table only
sampled (Gaps #9 + #15).

| RES Gap # | AEC3 mechanism | Our analogue | Per-bin? | Severity | Phase |
|---|---|---|---|---|---|
| R1 | `ResidualEchoEstimator` (separate 2-module pipeline) | `ResFilter` monolith (9 stages) | both per-bin | LOW — convention difference | n/a |
| R2 | `nearend_params_` ↔ `normal_params_` 2-way per-bin mask swap | scalar `effective_dt` floor lift | swap is per-bin | **HIGH** — primary NE HF protection mechanism | D-β (Phase D core) |
| R3 | `DominantNearendDetector` LF-only binary + hysteresis | broadband `dt_indicator` + `shadow_dt` → scalar `effective_dt` | scalar | MEDIUM | partial replacement of D-α |
| R4 | `SubbandNearendDetector` HF-quiet-when-voiced cue (echo-agnostic) | none | per-band cue | **HIGH** — only non-residual NE detector | D-α |
| R5 | `WeightEchoForAudibility` 3-band echo weighting | `block_lf` 3-band ENR tilt (v3.14 Arc R) | per-band | NONE — equivalent | shipped v3.14 |
| R6 | `X2_noise_floor[k]` minimum-statistics tracker | none | per-bin | MEDIUM — likely matters for pcb1N FS quiet | D-aux1 |
| R7 | `LimitHighFrequencyGains` 20-28 avg → bins ≥29 cap | `hf_cap` ([_stage_gain_postprocess](../python/aec.py#L3181)) | per-bin | MEDIUM — different rule; audit before port | C-aux audit |
| R8 | `LimitLowFrequencyGains` 3-bin equalize (gain[0]=gain[1]=min(gain[1],gain[2])) | none | per-bin | LOW | optional D-aux |
| R9 | `ReverbModel` per-bin EMA tail addition | none | per-bin | MEDIUM — pcb1N FS_static specific | v3.19 backlog |
| R10 | `ReverbDecayEstimator` adaptive T60 via filter regression | none (audit-only flags) | scalar input | MEDIUM | v3.19 backlog |
| R11 | `EarlyReverbLengthEstimator` 9-section overlap regression | none | scalar input | LOW — only meaningful with R9 + R10 | v3.19 backlog |
| R12 | `reverb_decay` mild ↔ adaptive switch on NE state | none | scalar | LOW — depends on R9 | v3.19 backlog |
| R13 | `UsableLinearEstimate` multi-gate (startup 0.4s + reset 0.2s + convergence_seen + !transparent) | `_filter_converged` (single bool) + `usable_linear_estimate` (half-built) | scalar | HIGH — Gap #9 in §5 | C |
| R14 | `R2` vs `R2_unbounded` dual estimate (NE detector sees unbounded) | not exposed | per-bin × 2 | MEDIUM — Phase D wiring | D dependency |
| R15 | `SaturatedEcho` Y2-passthrough | ad-hoc in `_stage_gain_compute` | per-bin | NONE | n/a |
| R16 | `GetResidualEchoScaling` stationarity-driven per-bin scaling | none (we have `_is_stationary_far` only as input gate) | per-bin | LOW — depends on R9/R10 stationarity logic first | v3.19 backlog |
| R17 | `LowNoiseRenderDetector` (block-level X² magnitude check) | `_far_power` gating equivalent | scalar | NONE | n/a |
| R18 | `EchoGeneratingPower` gated-max over render window | `_far_psd_smoothed` simpler EMA | per-bin | NONE — ours arguably more stable | n/a |
| R19 | `UpperBandsGain` scalar > 8 kHz | N/A (mono 16 kHz single-band) | — | NONE | n/a |
| R20 | `MovingAverageSpectrum` nearend smoother (configurable blocks) | implicit through per-frame consumption | per-bin | NONE | n/a |

### 11.1 Severity legend

- **HIGH** — structural difference, mapped to a concrete user-facing
  symptom (DT debt, NE HF damage, cohort tail catastrophe). Worth porting.
- **MEDIUM** — code-level difference, may improve specific cohorts.
  Port if Phase D capacity allows; otherwise v3.19+.
- **LOW** — small mechanism, cheap to add but limited expected value.
  Bundle with related changes.
- **NONE** — we already have equivalent OR mechanism is irrelevant
  (e.g. multi-channel-only).

### 11.2 Phase D scope condensation

From R1-R20:
- **D-α** (subband detector): R4 standalone. 1-2 sprints.
- **D-β** (two-way per-bin mask swap): R2 + R14. 4-6 sprints.
- **D-γ** (D-α + D-β + D-aux1): R2 + R4 + R6 + R14. 7-10 sprints.

Phase D **WILL NOT** port R9/R10/R11/R12/R16 (reverb arc) — deferred
to v3.19. Phase D **WILL NOT** port R7 differently from what we have
— audit-only sprint added to Phase C.

### 11.3 v3.17 substrate fold-in

| Substrate | Phase D D-γ consumes? |
|---|---|
| v3.17 B.2 `reverb_decay_estimate` audit flag | No — reverb scope is v3.19+ |
| v3.17 C.1 14-flag BALANCED-only finding | Yes — Phase E promotion sprint can include the new D-γ flags after Phase D ships |
| v3.13 E2 Path 3 DT debt (DT_static -0.050) | Targeted by D-β — recovery hard bar Δdeg ≥ +0.005 |
| 0I0XMl3M cohort tail | Partially — D-α subband detector may stabilize NE state on this case |
| xrtntuju 5-clip DT NE positive windows | Targeted by D-β nearend_params per-bin HF relaxation |

### 11.4 Decision for Phase 0.6

Recommendation: **Phase D OPEN at D-γ scope** (R2 + R4 + R6 + R14, 7-10
sprints). Hard bar per v3.18 plan §Phase D. Kill criterion = NE Δdeg <
+0.003 OR DT Δdeg < +0.003.

---

## 12. Changelog

- 2026-05-15 (v3.18 Phase 0.5): §9.1 fetch list refreshed — Phase 0.1
  cloned full AEC3 tree to `docs/aec3_extracts/src/aec3/` (commit
  `9310b29acd`). Phase 0.2 / 0.3 / 0.4 module reads complete, mapped to
  3 new docs: `aec3_residual_pipeline_mapping.md`,
  `aec3_subband_ne_detector_mapping.md`, `aec3_reverb_mapping.md`.
  New §11 RES-side gap table compiled (20-row R1-R20).
- 2026-04-22 (initial): first version. 6 core AEC3 modules
  analysed from earlier context + 8 more fetched this session
  (adaptive_fir_filter.h, erle_estimator.{h,cc}, erl_estimator.cc,
  refined_filter_update_gain.cc HandleEchoPathChange,
  block_processor.cc, echo_remover.cc, transparent_mode.cc,
  render_delay_controller.cc partial). 15-gap table compiled.
  4 high-priority experiments queued.

---

## 13. WebRTC AEC2 (legacy) vs AEC3 differences

# Ours balanced vs WebRTC AEC2 — 差距分析報告

> 800-case AEC Challenge blind test, AECMOS local ONNX, balanced preset, FL=512.
> 對比目的：找出我們在 DT deg / NE deg 輸給 WebRTC AEC2 的根因。

---

## 結論先講

**我們不是「整體比較差」，而是「壓得比較深」的單向 trade-off。**

- ✅ **Linear AEC 全面贏**：FS ERLE 18.2 vs 14.3 dB、DT ERLE 5.9 vs 3.8 dB
- ✅ **AECMOS FS echo 贏 +0.093**（壓 echo 更乾淨）
- ❌ **AECMOS DT deg 輸 -0.255、NE deg 輸 -0.097**（語音也被壓掉）
- ⚠️ **DT echo 輸 -0.032**（幾乎打平）

**根因（單一）**：RES 在純 DT / 純 NE 場景下過度壓抑近端語音。Linear AEC 已經做得比 AEC2 好，但 RES 把該保留的語音也一起壓了。

**證據**：DT-deg worst 30 個 case 中，**28 個的 DT-echo 是正的**（贏 AEC2 0.1+）—
也就是「同一個 case，我們 echo 壓得更乾淨，但語音也被壓掉了」。

---

## 1. 完整對比表（800 cases）

### Linear-level (eval_aec_challenge.py)

| 指標 | Ours balanced | WebRTC AEC2 | 差距 |
|------|--------------|-------------|------|
| FS ERLE | **18.2 dB** | 14.3 dB | +3.9 dB ✅ |
| NE SDR | -2.1 dB | -3.3 dB | +1.2 dB ✅ |
| DT ERLE | **5.9 dB** | 3.8 dB | +2.1 dB ✅ |

**Linear AEC 三項全勝。**

### AECMOS Mean

| Scenario | Metric | Ours | AEC2 | Δ | N |
|----------|--------|------|------|---|---|
| FS | echo↑ | **3.577** | 3.484 | **+0.093** ✅ | 300 |
| FS | deg↑ | 4.999 | 4.999 | -0.000 | 300 |
| NE | echo↑ | 4.998 | 4.998 | +0.000 | 200 |
| NE | deg↑ | 4.001 | **4.098** | **-0.097** ❌ | 200 |
| DT | echo↑ | 4.230 | 4.262 | -0.032 | 300 |
| DT | deg↑ | 2.134 | **2.389** | **-0.255** ❌ | 300 |

### Movement vs Non-movement breakdown

| Scenario | Metric | NoMv Δ | Mv Δ | NoMv N | Mv N |
|----------|--------|--------|------|--------|------|
| FS | echo | +0.105 | +0.078 | 169 | 131 |
| DT | echo | **-0.063** | +0.017 | 186 | 114 |
| DT | deg | **-0.191** | **-0.360** | 186 | 114 |
| NE | deg | -0.097 | — | 200 | 0 |

**觀察**：
- DT deg 在 movement 下退步更嚴重（-0.36 vs -0.19）→ movement / EPC 場景 RES 反應太敏感
- DT echo non-movement 微負（-0.063），movement 反而正（+0.017）→ 非常接近，echo 處理沒有顯著差距

---

## 2. Win / Lose / Tie 統計

> 閾值：delta > +0.1 算 win，delta < -0.1 算 lose，介於算 tie。

| Scenario | Metric | Win | Lose | Tie | Win % |
|----------|--------|-----|------|-----|-------|
| FS | echo | 122 | 150 | 28 | 40.7% |
| FS | deg | 0 | 0 | 300 | — |
| NE | echo | 0 | 0 | 200 | — |
| NE | deg | 10 | **83** | 107 | 5.0% |
| DT | echo | 102 | 143 | 55 | 34.0% |
| DT | deg | 103 | **158** | 39 | 34.3% |

**觀察**：
- **NE deg 大量 lose (83 vs 10)** — 純近端場景 RES 過壓
- **DT deg 也大量 lose (158 vs 103)** — 但仍有 1/3 case 我們贏，不是全面退步
- **FS echo win/lose 接近**（122 vs 150）— 平均贏 0.09 是少數高 score case 拉起來的

---

## 3. Delta 分佈 Percentiles

> Delta = Ours − AEC2，正值代表我們贏。

| Scenario | Metric | p10 | p25 | p50 | p75 | p90 |
|----------|--------|-----|-----|-----|-----|-----|
| FS | echo | -1.107 | -0.437 | -0.099 | +0.637 | +1.621 |
| NE | deg | **-0.265** | **-0.174** | **-0.066** | +0.002 | +0.051 |
| DT | echo | -0.652 | -0.344 | -0.084 | +0.292 | +0.692 |
| DT | deg | **-1.303** | **-0.644** | **-0.142** | +0.223 | +0.571 |

**觀察**：
- **DT deg p10 = -1.303**：worst 10% 的 case 我們輸超過 1 分（嚴重壓抑）
- **NE deg p25 = -0.174**：1/4 的近端場景輸超過 0.17（穩定地壓得太狠）
- **FS echo p10/p90 = ±1**：分佈很寬，極端 case 兩邊都有

---

## 4. Top-10 Worst Cases

### DT deg worst (我們語音保留輸最多)

| Case (前 25 char) | Mv | Ours | AEC2 | Δ |
|-------------------|----|------|------|---|
| nyT6FUUdu0W8UpvjP1rRgQ_dt_w | M | 1.543 | 4.114 | **-2.570** |
| W0zK3dv0QE2YckPArTGXCg_dt | | 1.478 | 3.766 | -2.289 |
| Tgtk8jp1zkqmKzsmdrKt0g_dt | | 1.766 | 3.851 | -2.085 |
| nyT6FUUdu0W8UpvjP1rRgQ_dt | | 1.587 | 3.662 | -2.075 |
| Tgtk8jp1zkqmKzsmdrKt0g_dt_w | M | 1.776 | 3.831 | -2.055 |
| wHmBm7VHfkysBOhjoAXkNA_dt_w | M | 1.305 | 3.310 | -2.005 |
| XTqo1aOXDEiqyWTFK99I5Q_dt_w | M | 1.910 | 3.844 | -1.933 |
| W4r0UCjieEuM0u930spvug_dt_w | M | 1.791 | 3.723 | -1.932 |
| VNkNShj97UajHDVbSmIG0g_dt_w | M | 1.974 | 3.892 | -1.917 |
| sLi810BoekuU3HSx14LT7A_dt | | 1.744 | 3.641 | -1.897 |

→ Worst 20 中 **11 個是 movement**（55%），movement 案例佔比顯著高於整體比例（38%）。

### DT echo worst (我們 echo 壓不過 AEC2)

| Case | Mv | Ours | AEC2 | Δ |
|------|----|------|------|---|
| Y7w0W4v9BEihm8Z06BxZfQ_dt | | 2.924 | 4.575 | -1.650 |
| IxgmaPghzUGnR6sxrbGU3Q_dt | | 3.309 | 4.795 | -1.486 |
| r7U6JmcRl0ibIh0mN3CP9g_dt | | 3.172 | 4.573 | -1.401 |
| s0oJqM6Y1UCHSVmHmgsx4Q_dt_w | M | 3.282 | 4.627 | -1.345 |
| JjCzlhn3gEiBQvfJtPNJ9A_dt_w | M | 2.942 | 4.220 | -1.278 |

→ Worst 20 中 7 個 movement（35%），跟整體比例接近 — DT echo 的退步**不是 movement-bound**。

### NE deg worst

| Case | Ours | AEC2 | Δ |
|------|------|------|---|
| IvAFDPUFEk0GuW2VKRHznA_ne | 2.817 | 3.590 | -0.773 |
| OlEt0Ltk0UKogetYnqirSw_ne | 3.573 | 4.248 | -0.675 |
| SFvlSygv4ke9wCrv8LWvYQ_ne | 3.015 | 3.658 | -0.643 |
| E0l0WVPQjEi6AmtbvfSYLA_ne | 3.560 | 4.110 | -0.550 |

→ NE 全部 non-movement（test set 沒有 NE+movement）。退步集中在「ours 3.0-3.6 / AEC2 3.6-4.2」區間，是穩定的小幅 over-suppression。

---

## 5. 跨 Metric 交叉分析（最關鍵）

### DT-deg worst 30 個 case，DT-echo 表現如何？

| 同時失分 | 數量 |
|---------|------|
| DT-deg worst 30 中 DT-echo **也輸** (Δ < -0.1) | **2 / 30** |
| DT-deg worst 30 中 DT-echo **打平** (\|Δ\| < 0.1) | ~3 / 30 |
| DT-deg worst 30 中 DT-echo **我們贏** (Δ > +0.1) | **~25 / 30** |

→ **這是最關鍵的證據**：在 DT 語音被壓最嚴重的 30 個 case，**幾乎全部 echo 是壓得比 AEC2 更乾淨的**。

意思是 — RES 在這些 case 把整個 DT 段都當 echo 壓掉了，所以：
- echo 很乾淨 ✅（壓過頭）
- 但語音也跟著消失 ❌

### NE-deg worst 也是 DT-deg worst？

`32 cases lose both DT-deg and NE-deg` — 32 個近端場景同時在兩個 metric 都輸 0.1+。代表這些 case 的近端訊號，無論在 DT 還是 NE 場景下，我們都會壓掉。

---

## 6. 根因假設

### (A) RES 太激進整路 ← **最可能**

證據：
- DT-deg worst 30 中 28 個 DT-echo 為正 → 同一段同時壓 echo 和語音
- NE deg 大量輸（83 lose / 10 win）→ 純近端場景也壓
- Linear AEC 三項全勝 → 不是 filter 的問題
- balanced preset 的 g_min_db = -55dB，比 AEC2 的內部設定更激進

### (B) DT 偵測太敏感

證據：
- DT deg movement 退步更嚴重（-0.36 vs non-movement -0.19）
- Movement 場景下 dt_indicator / EPC 都更敏感

### (C) NE protection 不足

證據：
- NE deg p25 = -0.17（穩定的小幅退步）
- 32 個 case 在 DT + NE 兩個場景都輸 → 跨場景共通問題

排除：
- ❌ Linear AEC 不夠強 — 我們 ERLE 全勝
- ❌ Stationary far-end 沒處理好 — v13 已修，且 worst case 不集中在穩態場景

---

## 7. 修正方向建議（不在本次 scope 內）

優先順序由高到低：

1. **降低 RES 預設激進度**：
   - balanced preset 的 `g_min_db` 從 -55dB → -45dB（接近 AEC2 內部值）
   - 或新增「ultra-mild」preset 對應 AEC2 的取向
2. **加強 NE protection**：
   - `ne_protect_db` 從 -16dB → -10dB
   - per-bin NE gate 的 ne_erle_gate 提高
3. **DT 偵測門檻拉緊**：
   - dt_indicator 從 0.0-0.8 → 0.0-0.6（縮小作用範圍）
   - movement 場景下 RES 進入更保守模式
4. **逐 case 微調 ENR 兩段映射**：
   - DT 段 enr_t/enr_s 調寬，減少 false-positive 壓抑

---

## 8. Trade-off 討論

### 值不值得追？

**追的代價**：
- DT echo 平均會降（目前贏 -0.032 → 追語音可能變 -0.1+）
- FS echo 平均會降（目前贏 +0.093 → 可能變持平）
- 整體 Linear AEC ERLE 不變（這是 RES 層的事）

**追的好處**：
- DT deg +0.25（追到跟 AEC2 持平）
- NE deg +0.10
- 對「會議、近端品質優先」場景更友善

### 是否該追？

**取決於應用場景**：
- 📞 **語音通話 / 會議**：應該追，近端品質重要
- 🎬 **直播 / podcast**：應該追，語音品質是核心
- 🔇 **強回聲環境 / 嘈雜場景**：不該追，當前 trade-off 合理
- 🎯 **產品定位「echo 殺手」**：不該追，當前是賣點

### 推薦行動

1. **不改 balanced 預設**（避免破壞現有用戶）
2. **把現有 mild preset 再調更保守**（以對標 AEC2 的近端品質為目標）
3. **跑一次 mild preset 800-case** 看是否已經接近 AEC2

---

## 附錄：分析方法

### 工具

- `python/eval_aec_challenge.py --old-aec --preset balanced --filter 512`
- `/tmp/AEC-Challenge/AECMOS/AECMOS_local/Run_1663915512_Stage_0.onnx`
- `python/eval_aecmos_local.py` (參考)
- ad-hoc 分析腳本：`/tmp/analyze_vs_old_aec.py`
- raw scores: `/tmp/aec_vs_old_raw.json`

### 流程

1. 跑 800-case Old WebRTC AEC2 → `output/{stem}_old_aec.wav`
2. 對每個 case 用本地 ONNX AECMOS 評分 ours + old_aec
3. 計算 delta = ours - old_aec
4. 統計 mean / movement breakdown / win-lose-tie / percentiles
5. 找 top-20 worst cases
6. 跨 metric 交叉分析

### 範圍限制

- 只跑 balanced preset（mild/aggressive/maximum 沒測）
- 純數據分析，沒有 spectrogram / 沒有聽音檔
- 不修改任何代碼或參數
