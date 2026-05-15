# AEC3 Reverb Mapping (Phase 0.4)

**v3.18 Phase 0.4 deliverable** — maps WebRTC AEC3 `reverb_model.cc` +
`reverb_decay_estimator.cc` (commit `9310b29acd`) against our existing
reverb-related state (none in production; audit-only flags from
v3.17 B.2 closure).

Inputs:
- AEC3 `reverb_model.cc` (54 lines)
- AEC3 `reverb_model.h` (55 lines)
- AEC3 `reverb_decay_estimator.cc` (411 lines)
- AEC3 `reverb_decay_estimator.h` (117 lines)
- Our state: no per-bin reverb model. v3.17 B.2 closure substrate at
  [docs/v3_17_b2_reverb_aware_closure.md](v3_17_b2_reverb_aware_closure.md)
- Phase 0.2 cross-ref §2.1 item 3 (reverb tail addition in
  `ResidualEchoEstimator`)

## 1. `ReverbModel` — the tail PSD adder

[reverb_model.cc:41](aec3_extracts/src/aec3/reverb_model.cc#L41):

```cpp
// Per-bin EMA with per-bin scaling
void ReverbModel::UpdateReverb(
    span<const float> power_spectrum,                  // X[k]² of render
    span<const float> power_spectrum_scaling,          // ScaleFilter freq response
    float reverb_decay) {                              // scalar T60 decay
  if (reverb_decay > 0) {
    for (size_t k = 0; k < power_spectrum.size(); ++k) {
      reverb_[k] = (reverb_[k] + power_spectrum[k] * power_spectrum_scaling[k])
                   * reverb_decay;
    }
  }
}
```

**Plus a no-frequency-shaping variant** for non-linear fallback:

```cpp
void ReverbModel::UpdateReverbNoFreqShaping(
    span<const float> power_spectrum,
    float power_spectrum_scaling,                      // scalar ep_gain
    float reverb_decay) {
  // identical except scaling[k] becomes constant
  reverb_[k] = (reverb_[k] + power_spectrum[k] * power_spectrum_scaling)
               * reverb_decay;
}
```

**Properties**:
- Pure first-order IIR per bin: `r[k] ← (r[k] + scaled_X[k]) * decay`
- Steady-state: `r[k] = scaled_X[k] * decay / (1 - decay)`
- **No state machine, no triggers** — every frame adds new render power
  and decays old reverb. Add `r[k]` to `R2[k]` in `ResidualEchoEstimator::AddReverb`
- Tiny code (54 lines), but **load-bearing on long-reverb cases**

**Inputs needed from upstream**:
1. `power_spectrum_scaling[k]` — per-bin frequency response of the
   adaptive filter (from `aec_state.GetReverbFrequencyResponse()`)
2. `reverb_decay` — scalar T60 estimate (from `aec_state.ReverbDecay()`,
   ultimately from `ReverbDecayEstimator`)

## 2. `ReverbDecayEstimator` — the T60 estimator

The non-trivial module (411 lines). Estimates the **scalar reverb decay
rate** `decay_` by analyzing the **time-domain filter impulse response**
(refined filter, not shadow).

### 2.1 Algorithm pipeline

[reverb_decay_estimator.cc:106](aec3_extracts/src/aec3/reverb_decay_estimator.cc#L106) `Update`:

```
if stationary_signal: return
if filter unusable (filter_size mismatch / filter_delay_blocks == 0
                    / !usable_linear_filter): ResetDecayEstimation; return
smoothing = max(prev_smoothing, filter_quality * 0.2)
if smoothing == 0: return

if block_to_analyze < filter_length_blocks:
    AnalyzeFilter(filter)              # accumulate one block of energy data
    block_to_analyze++
else:
    EstimateDecay(filter, peak_block)  # final estimation step
```

### 2.2 `AnalyzeFilter` — accumulate log-energy by block

[reverb_decay_estimator.cc:226](aec3_extracts/src/aec3/reverb_decay_estimator.cc#L226):

For each block-to-analyze in the filter impulse response:
1. Compute per-coefficient `h²[k]` (squared filter coefficients)
2. `AnalyzeBlockGain`: detect if the block is "adapting" (energy
   changing too fast) or "below noise floor" (energy small)
3. Count consecutive "good" blocks (`estimation_region_candidate_size_`)
4. If in late-reverb region: accumulate `log2(h²[k])` into late-reverb
   linear regressor
5. Always: accumulate into early-reverb length regressor

### 2.3 `EstimateDecay` — fit linear regression in log-domain

[reverb_decay_estimator.cc:162](aec3_extracts/src/aec3/reverb_decay_estimator.cc#L162):

Once the filter is fully analyzed:
1. Verify filter quality: `first_reverb_gain > 4 × tail_gain` (sufficient
   decay) AND `> 2 × tail_gain` (valid filter) AND `peak_energy < 100`
2. Estimate early-reverb length via `EarlyReverbLengthEstimator::Estimate()`
   (analyzes tilt of 9 overlapping 6-block sections)
3. `late_reverb_decay_estimator.Estimate()` returns slope of
   `log2(h²)` over late reverb blocks
4. `decay = 2^(slope × kFftLengthBy2)` — converts log-domain slope
   back to per-block decay factor
5. Clamp `[kMinDecay=0.02, kMaxDecay=0.95]` (~15 ms - 1 s RT60)
6. EMA toward current decay: `decay_ += smoothing × (decay - decay_)`

### 2.4 `EarlyReverbLengthEstimator` — sub-module

[reverb_decay_estimator.cc:306-409](aec3_extracts/src/aec3/reverb_decay_estimator.cc#L306):

9 overlapping 6-block sections, each runs a linear regression on
log-energy. Section `k` is flagged "early reverb" if:
- `numerator_k > numerator_for_decay_1.1` (energy not decaying, slope > 0)
  OR
- `numerator_k < numerator_for_decay_0.8` AND `numerator_k < 0.9 ×
  min_numerator_tail` (energy decaying faster than tail — early reflection)

Returns last index `k` flagged → early reverb size in blocks.

### 2.5 `decay_` and `mild_decay_`

[reverb_decay_estimator.cc:98](aec3_extracts/src/aec3/reverb_decay_estimator.cc#L98):

Two decay values maintained:
- `decay_` — adaptive (from `EstimateDecay`)
- `mild_decay_` — fixed at `|config.ep_strength.nearend_len|`

`aec_state.ReverbDecay(mild)` picks one:
- `mild = dominant_nearend` → `mild_decay_` (gentler decay in NE mode,
  preserves NE longer; per [docs/aec3_reference.md](aec3_reference.md))
- else → adaptive `decay_`

This is the **second NE-aware mask** in AEC3's residual pipeline (the
first being `nearend_params_` per-bin masks): reverb tail decays slower
in NE mode → larger residual estimate → more cautious suppression.

## 3. Our existing reverb state

From v3.17 B.2 closure ([docs/v3_17_b2_reverb_aware_closure.md](v3_17_b2_reverb_aware_closure.md)):

| Flag / state | Location | Status | What it does |
|---|---|---|---|
| `reverb_decay_estimate` | `AecConfig` | audit-only | Substrate from v3.17 B.2 attempt; default OFF |
| C9 audit script (`docs/c9_reverb_aware_audit.md`) | tools/ | shipped | Identifies pcb1N FS_static reverb cohort (N=1) |
| `_reverb_tail_psd` / `_reverb_decay_*` | not present | — | We have no per-bin reverb tail tracker |

**No production code path adds a reverb-tail PSD to residual echo
estimate**. Our `_stage_residual_model` computes `residual_echo_psd`
from `echo_psd / erle_factor + leakage` (ERLE-divide) plus a `near_spec`
direct method — neither captures the late-reverb tail that AEC3
explicitly models.

**Effect on pcb1N FS_static**: our residual estimate decays as fast as
the linear-filter echo estimate decays (which is fast — the filter
quickly converges past the direct path), so by the time the reverb
tail is the dominant echo, our residual model has zero handle on it,
which means RES gains rise → reverb leaks through.

## 4. Mapping table

| AEC3 component | Our analogue | Per-bin? | Gap severity |
|---|---|---|---|
| `ReverbModel::UpdateReverb` (per-bin EMA tail) | none | per-bin | **HIGH** for reverb cohort |
| `ReverbModel::UpdateReverbNoFreqShaping` | none | per-bin (constant scale) | LOW (would need non-linear-mode RES first) |
| `aec_state.ReverbDecay(mild)` mild vs adaptive switch | none | scalar | MEDIUM — depends on AECMOS evidence for reverb cohort |
| `aec_state.GetReverbFrequencyResponse()` (per-bin filter freq response) | available via `PBFDKF` weights → easy to expose | per-bin | LOW — substrate exists |
| `ReverbDecayEstimator::AnalyzeFilter` (per-block log² accumulation) | none | scalar | HIGH if we want adaptive T60 |
| `ReverbDecayEstimator::EstimateDecay` (linear regression in log domain) | none | scalar | HIGH same |
| `EarlyReverbLengthEstimator` (9-section overlap linear regression) | none | scalar | MEDIUM — only matters if we want to separate early-vs-late tail |
| `FastApproxLog2f` (fast log2) | available via numpy log2 (no perf concern in Python) | scalar | NONE |

## 5. Phase 0.4 closure — Phase D and Phase E impact

### 5.1 Reverb is OUT of Phase D scope

Per v3.18 plan §"Phase D" (NE detector port), reverb is explicitly
deferred. Reasons:
1. The plan's Phase D ROI estimate assumes the AEC3 NE detector + 2-way
   mask swap closes 0I0XMl3M / xrtntuju / FS_static cohort. Reverb model
   is a **separate ROI** on **pcb1N FS_static** specifically.
2. The full reverb port (ReverbModel + ReverbDecayEstimator) is 5-8
   sprints standalone, matching v3.17 B.2 closure's LOE estimate.
3. Phase D D-γ already pushes plan LOE to 26-37 sprints; adding reverb
   port would push to 32-45.

### 5.2 Where reverb belongs in the v3.18+ roadmap

**v3.18 verdict**: reverb port stays in backlog as **v3.19 candidate
arc**. Substrate from v3.17 B.2 (C9 audit script) is preserved.

**Phase D-γ DOES NOT subsume reverb improvement** — the two-way mask
swap addresses NE confidence routing, not late-tail reverb modelling.
pcb1N FS_static is expected to remain unfixed after Phase D ships.

### 5.3 Minor Phase D / Phase A interaction

`ReverbDecayEstimator::Update` gates on `usable_linear_filter` — which
is exactly the Phase C `usable_linear_estimate` substrate. So if Phase C
lands first, it provides the gating signal that an eventual reverb port
would consume. **Phase ordering implication**: Phase C → reverb arc
(v3.19+). No Phase 0 implication.

## 6. Cheap audit candidate (Phase D-aux)

A **small, low-risk addition** worth considering for Phase D:

**Phase D-aux2 (audit-only)**: expose existing PBFDKF refined-filter
impulse response → compute reverb_decay scalar via simple
last-block-energy / first-block-energy ratio (without full AEC3 linear
regression).

- Trace it (no algorithm change)
- See if it correlates with pcb1N FS_static reverb subjective annoyance
- If correlation > 0.5, prioritize full reverb arc for v3.19
- If correlation < 0.3, deprioritize / close

**Cost**: 1-2 sprints, default-OFF trace flag, byte-equal sanity.

**Value**: builds evidence base for v3.19 reverb arc OR formally closes
the "reverb mechanism" hypothesis.

## 7. Files this doc maps

**AEC3 sources analysed (Phase 0.4)**:
- [docs/aec3_extracts/src/aec3/reverb_model.cc](aec3_extracts/src/aec3/reverb_model.cc) (54 lines)
- [docs/aec3_extracts/src/aec3/reverb_model.h](aec3_extracts/src/aec3/reverb_model.h) (55 lines)
- [docs/aec3_extracts/src/aec3/reverb_decay_estimator.cc](aec3_extracts/src/aec3/reverb_decay_estimator.cc) (411 lines)
- [docs/aec3_extracts/src/aec3/reverb_decay_estimator.h](aec3_extracts/src/aec3/reverb_decay_estimator.h) (117 lines)

**Our state**:
- v3.17 B.2 closure: [docs/v3_17_b2_reverb_aware_closure.md](v3_17_b2_reverb_aware_closure.md)
- C9 audit script: `docs/c9_reverb_aware_audit.md`
- No production-side reverb code in `python/aec.py`

**Cross-references**:
- Phase 0.2: [docs/aec3_residual_pipeline_mapping.md](aec3_residual_pipeline_mapping.md)
  §2.1 item 3 (reverb tail in ResidualEchoEstimator)
- Phase 0.3: [docs/aec3_subband_ne_detector_mapping.md](aec3_subband_ne_detector_mapping.md)
- Phase 0.6 decision: `docs/v3_18_phase0_decision.md` (pending)
- v3.18 plan: `~/.claude/plans/se-aec-aec-main-hazy-lynx.md`
