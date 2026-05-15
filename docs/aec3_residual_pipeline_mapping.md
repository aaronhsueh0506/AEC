# AEC3 Residual Pipeline Mapping (Phase 0.2)

**v3.18 Phase 0.2 deliverable** — maps WebRTC AEC3 `suppression_gain.cc` +
`residual_echo_estimator.cc` (commit `9310b29acd`) against our `ResFilter`
9-stage pipeline. Source under
[docs/aec3_extracts/src/aec3/](aec3_extracts/src/aec3/).

Inputs:
- AEC3 `suppression_gain.cc` (514 lines)
- AEC3 `residual_echo_estimator.cc` (432 lines)
- Our `python/aec.py::ResFilter` (~1500 lines, `_stage_*` pipeline at
  [python/aec.py:2670](../python/aec.py#L2670) ff.)
- Reference: [docs/aec3_reference.md](aec3_reference.md) (2026-04-22, §6).

## 1. Structural difference — 2 modules vs 1 monolith

| AEC3 | Ours |
|---|---|
| `ResidualEchoEstimator::Estimate` → produces `R2[ch][k]` (per-bin residual echo power, dual: clamped + unbounded) | `ResFilter._stage_residual_model` → produces `residual_echo_psd` (per-bin) |
| `SuppressionGain::GetGain` → consumes `R2` + `nearend_spectrum` + `comfort_noise` → outputs per-bin `low_band_gain[k]` + scalar `high_bands_gain` | `ResFilter._stage_gain_compute` + `_stage_gain_postprocess` + `_stage_temporal_smoothing` + `_stage_noise_floor_cng` → outputs per-bin `gain[k]` |
| 2 narrow modules, clean inputs/outputs | 9 stages, ~25 state fields threading, mode coupling between stages |

**Implication for Phase D port**: AEC3's input-side cut is `R2` (per-bin
residual echo power); our analogue is `residual_echo_psd` at
[python/aec.py:2670](../python/aec.py#L2670). Port boundary is feasible
at that exact seam — we already have a per-bin tensor at that point.

## 2. Residual-echo estimator structure (AEC3)

`ResidualEchoEstimator::Estimate` ([residual_echo_estimator.cc:193](aec3_extracts/src/aec3/residual_echo_estimator.cc)):

```
if SaturatedEcho:
    R2[k] = Y2[k]                          # mic-power passthrough
elif UsableLinearEstimate:                  # ← Phase C gate (Gap #9)
    R2[k] = S2_linear[k] / Erle[k]         # per-bin ERLE-divide
    R2 += echo_reverb.reverb()              # per-bin reverb tail
else:                                       # non-linear fallback
    X2[k] = EchoGeneratingPower(...)       # gated max-power over window
    X2[k] -= stationary_gate_slope * X2_noise_floor[k]
    R2[k] = X2[k] * echo_path_gain         # scalar gain × per-bin X2
    if model_reverb_in_nonlinear_mode:
        R2 += echo_reverb.reverb()
if UseStationarityProperties:
    R2[k] *= residual_scaling[k]           # per-bin stationarity scaling
```

Dual outputs:
- `R2` — used downstream by `SuppressionGain::LowerBandGain`
- `R2_unbounded` — uses `ErleUnbounded()` (no NE-onset compensation), fed
  to `DominantNearendDetector` for NE-state decision (avoids the
  feedback loop: NE detector sees pre-clamp residual)

### 2.1 Key per-bin mechanisms

1. **Per-bin ERLE** (`aec_state.Erle(onset_compensated)`) — from
   `ErleEstimator` (not in this file; mapped in v3.18 Phase 0 separate
   deep-dive needed if Phase D opens).
2. **`X2_noise_floor[k]`** — minimum-statistics tracker per-bin
   ([line 343-358](aec3_extracts/src/aec3/residual_echo_estimator.cc#L343)):
   ```
   if X[k] < floor[k]: floor[k] = X[k]; counter[k] = 0
   elif counter[k] >= hold: floor[k] = max(floor[k]*1.1, min_floor)
   else: counter[k]++
   ```
   We don't have this. `noise_floor_cng` is downstream of gain
   (comfort-noise fill, not echo estimator input subtraction).
3. **Reverb tail** (`echo_reverb.UpdateReverb` / `AddReverb`) — per-bin
   EMA `reverb[k] = (reverb[k] + X[k]*scale[k]) * decay`. We don't have
   per-bin reverb modelling — only the audit-only `reverb_decay_*`
   flags from v3.17 B.2 close.
4. **`residual_scaling[k]`** — `aec_state.GetResidualEchoScaling()` —
   per-bin scaling from stationarity-based audibility. We don't expose
   per-bin stationarity influence on residual.

### 2.2 Saturated echo handling

AEC3 sets `R2 = Y2` (mic-power passthrough). This effectively says "I
don't trust my residual estimate, attenuate as much as the mic energy
allows." We have equivalent in
`ResFilter._stage_gain_compute::satrurated_echo` ad-hoc branch but
embedded in the gain stage, not in residual estimator.

## 3. Suppression-gain structure (AEC3)

`SuppressionGain::GetGain` ([suppression_gain.cc:385](aec3_extracts/src/aec3/suppression_gain.cc#L385)):

```
DominantNearendDetector.Update(nearend, R2_unbounded, comfort_noise)
                                                  ↓
                              IsNearendState() →  ↓
                                                  ↓
LowerBandGain(...):
    GetMaxGain(): max_gain[k] = clamp(last_gain[k] * inc_factor, ...)
    for ch:
        nearend_smoother.Average(suppressor_input[ch], nearend)
        weighted_residual_echo = WeightEchoForAudibility(R2[ch])     # 3-band tilt
        GetMinGain(...): min_gain[k] = noise_limit / weighted_echo[k]
        GainToNoAudibleEcho(nearend, weighted_echo, comfort_noise):
            # per-bin masking decision using per-bin enr/emr thresholds
            for k:
                enr = echo[k] / (nearend[k]+1)
                emr = echo[k] / (masker[k]+1)
                if enr > enr_transparent_[k] and emr > emr_transparent_[k]:
                    g = (enr_suppress_[k] - enr) / (enr_suppress_[k] - enr_transparent_[k])
                    g = max(g, emr_transparent_[k] / emr)
                G[k] = g
        G[k] = clip(G[k], min_gain[k], max_gain[k])
        low_band_gain[k] = min(low_band_gain[k], G[k])     # mc reduce across channels
    LimitLowFrequencyGains:  gain[0] = gain[1] = min(gain[1], gain[2])
    if (not IsNearendState) or clock_drift or conservative_hf:
        LimitHighFrequencyGains:                                    # conservative HF cap
            min_upper = min(gain[20..28])
            for k>=29: gain[k] = min(gain[k], min_upper)
    last_gain[:] = gain[:]
    Sqrt(gain)                                                       # power → amplitude

high_bands_gain = UpperBandsGain(...)                                # scalar for >8kHz bands
```

### 3.1 Per-bin masking parameters

`GainParameters::SetConfig` ([suppression_gain.cc:487](aec3_extracts/src/aec3/suppression_gain.cc#L487))
interpolates LF→HF thresholds across `kFftLengthBy2Plus1 = 65` bins:

```
for k in [0..64]:
    if k <= last_lf_band: a = 0
    elif k < first_hf_band: a = (k - last_lf_band) / (first_hf_band - last_lf_band)
    else: a = 1
    enr_transparent_[k] = (1-a)*lf.enr_transparent + a*hf.enr_transparent
    enr_suppress_[k]    = (1-a)*lf.enr_suppress    + a*hf.enr_suppress
    emr_transparent_[k] = (1-a)*lf.emr_transparent + a*hf.emr_transparent
```

Two parameter sets co-exist: `nearend_params_` (used when
`IsNearendState`) and `normal_params_` (FS / DT-medium). Switch is
full-band binary based on DominantNearendDetector — but **the params
themselves are genuinely per-bin** via the LF→HF interpolation.

### 3.2 NE HF protection mechanisms (user's primary concern)

Five distinct mechanisms protect NE high-frequency content in AEC3:

| Mechanism | Location | What it does |
|---|---|---|
| **Per-bin enr/emr thresholds** | `GainToNoAudibleEcho` | Different `enr_transparent_[k]` per bin (LF lower, HF higher → HF more tolerant of residual echo when there's NE energy) |
| **`nearend_params_` vs `normal_params_`** | `LowerBandGain` 2-way switch | Full param-set swap when dominant NE detected — typically `nearend.mask_hf.enr_transparent` is much higher than `normal.mask_hf.enr_transparent` (less aggressive HF suppression in NE mode) |
| **`WeightEchoForAudibility`** | residual-echo path before `GainToNoAudibleEcho` | 3-band threshold weighting (bins 0-3 LF / 3-7 MF / 7-65 HF), each band has its own `audibility_threshold_*`. Below threshold → echo weight smoothly goes to 0 (i.e., very quiet echo isn't fought against, freeing NE budget) |
| **`LimitHighFrequencyGains` conservative cap** | post `GainToNoAudibleEcho` | When filter is unconverged (`conservative_hf_suppression` set), HF bins ≥29 get capped at avg of bins 20-28. Prevents HF leakage when filter is bad. **But this is HF *suppression*, not *protection*** — it caps the gain upward, which **hurts** NE HF. Set OFF during dominant-NE. |
| **`SubbandNearendDetector`** alt | replaces `DominantNearendDetector` if enabled | Detects NE when subband1 < threshold × subband2 AND subband1 > snr×noise. Two-region structural check — more sensitive than the full-band low-frequency-energy check in `DominantNearendDetector`. **This is the per-band NE detection user asked about (§9.1 Phase 0.3 doc).** |

### 3.3 `WeightEchoForAudibility` detail — 3-band tilt

[suppression_gain.cc:88](aec3_extracts/src/aec3/suppression_gain.cc#L88):

```
threshold_lf = floor_power * audibility_threshold_lf       # bins 0-3
threshold_mf = floor_power * audibility_threshold_mf       # bins 3-7
threshold_hf = floor_power * audibility_threshold_hf       # bins 7-65
weighted_echo[k] = echo[k]  if echo[k] >= threshold[band(k)]
                 = echo[k] * max(0, 1 - ((threshold-echo[k])*norm)^2)  otherwise
```

**This is identical in structure to our v3.14 Arc R `block_lf` 3-band ENR
tilt** ([aec.py:794 `res_per_band_enr`](../python/aec.py#L794)). Confirms
our per-band tilt was on the right track architecturally — but AEC3's
genuinely-per-bin masking thresholds (§3.1) are a strict superset.

## 4. Mapping table — AEC3 vs our ResFilter

| AEC3 component | Our `ResFilter` analogue | Per-bin? | Gap severity |
|---|---|---|---|
| `R2 = S2_linear / Erle` (linear mode) | `_stage_residual_model` per-bin `psd` from EER | Both per-bin | LOW — equivalent (both depend on `Erle` quality which is upstream) |
| `R2 = X2 * echo_path_gain` (non-linear fallback) | not exposed — we always run linear; `usable_linear_estimate` is half-built (Phase C) | — | MEDIUM — would need plumbing if Phase C lands |
| `EchoGeneratingPower` (gated max over render window) | `_far_psd_smoothed` simpler EMA | Both per-bin | LOW |
| `X2_noise_floor[k]` minimum-statistics | none | per-bin | HIGH — explains some FS-quiet over-suppression |
| `WeightEchoForAudibility` 3-band tilt | Arc R `block_lf` 3-band ENR (already shipped v3.14) | per-band coarse | LOW — equivalent |
| `dominant_nearend_detector` (or subband) → 2-way param swap | aggregate `dtd_coherence + dtd_divergence` → scalar `effective_dt` | scalar | **HIGH** — Phase 0.3 deep-dive (see [docs/aec3_subband_ne_detector_mapping.md](aec3_subband_ne_detector_mapping.md)) |
| `GainToNoAudibleEcho` per-bin enr/emr | `_stage_gain_compute` per-bin via `eer` + `coh2` + scalar `effective_dt` | mostly per-bin | **HIGH** — our per-bin gain is gated by scalar `effective_dt`, not per-bin NE confidence |
| `last_nearend / last_echo` smoothing (one-block) | `_dt_per_bin_last` exists ([aec.py:2286](../python/aec.py#L2286)) | per-bin | LOW |
| `GetMinGain` (residual-floor based) | `spectral_floor` (per-bin) + `ne_g_floor` (per-bin) | per-bin | LOW — equivalent |
| `GetMaxGain` (last-gain inc cap) | implicit in temporal smoothing | per-bin | LOW |
| `LimitLowFrequencyGains` (gain[0]=gain[1]=min(gain[1],gain[2])) | not present | per-bin | LOW — small fix candidate |
| `LimitHighFrequencyGains` 20-28 avg cap on bins ≥29 | `hf_cap` ([_stage_gain_postprocess](../python/aec.py#L3181)) — different rule | per-bin | MEDIUM — different approach, ours may be more / less conservative; needs A/B audit |
| `reverb_model` per-bin tail | none (only audit flags) | per-bin | MEDIUM — see [aec3_reverb_mapping.md](aec3_reverb_mapping.md) |
| `Sqrt(gain)` (power→amplitude domain) | implicit through PSD-vs-amplitude convention | — | NONE — convention difference, not algorithmic |
| `UpperBandsGain` scalar for >8 kHz | N/A (we are 16 kHz mono single-band) | — | NONE |

## 5. Critical findings for Phase D scope decision

### 5.1 Confirmed: AEC3 RES has structural advantage on NE confidence injection

**Our pipeline**: `dtd_coherence + dtd_divergence` → scalar `effective_dt`
→ this scalar enters per-bin gain compute as a uniform floor lift. So
**per-bin gain modulation depends on per-bin `coh²(k)` and `eer(k)`, but
NE-state confidence is scalar**.

**AEC3 pipeline**: `DominantNearendDetector` (or `SubbandNearendDetector`)
→ binary `IsNearendState()` → switches **entire param set**
`nearend_params_` (per-bin masks) vs `normal_params_` (per-bin masks).
So **per-bin masking thresholds are themselves per-bin AND switch based on
NE detector**.

The asymmetry isn't "per-bin vs scalar gain" (we're per-bin too) — it's
"per-bin **NE-aware mask shape**" (AEC3 has 2 distinct per-bin profiles
swapping atomically) vs "per-bin gain modulated by scalar NE
confidence" (ours).

### 5.2 The gap that could close v3.13 E2 Path 3 DT debt

`v3.13 E2 Path 3` shipped a delay-window enlargement that gave FS_static
+0.107 Δecho but unmasked DT_static −0.050 / DT_movement −0.025 — root
cause: linear filter now catches more echo → RES sees lower S2_linear →
residual estimate too low → gain too aggressive in DT regime where
`effective_dt` doesn't lift the floor enough.

**Mechanism that could fix this in AEC3**: nearend_params have higher
`enr_transparent_` → in NE/DT mode, residual must be much louder relative
to nearend before any suppression kicks in. We don't have the equivalent
"swap entire mask profile when NE-confident" — we only lift a floor.

**Phase D candidate**: port `nearend_params_` / `normal_params_` two-way
swap with per-bin LF→HF interpolated profile, driven by an NE detector
(Phase 0.3 will decide subband or aggregate variant).

### 5.3 Mechanisms NOT worth porting from AEC3 RES

- `UpperBandsGain` scalar high-band — we're 16 kHz mono, single-band
- `LowNoiseRenderDetector` — our `_far_power` gating equivalent exists
- `EchoGeneratingPower` gated-max window — our `_far_psd_smoothed` is
  simpler and arguably more stable; A/B not worth doing
- `SaturatedEcho` Y2-passthrough — we have equivalent ad-hoc path

### 5.4 Mechanisms with mid-tier ROI

- `X2_noise_floor` minimum-statistics → would benefit pcb1N-class
  reverb cases (substrate: `reverb_decay_estimator.cc` exists)
- `LimitHighFrequencyGains` 20-28 avg cap — already have `hf_cap` but
  algorithm differs; **audit before port** (could be regression source)
- `LimitLowFrequencyGains` 3-bin equalize — trivial, low risk

## 6. Phase D decision input (feeds Phase 0.6)

**Recommendation given evidence so far**: **Phase D OPEN** with scope
narrowed to:

1. **D-core**: per-bin masking profile swap (`nearend_params_` /
   `normal_params_` two-way) driven by NE detector decision
   - Wire: 2 per-bin profile tables (~65 floats × 2 × 3 thresholds) in
     `AecConfig`
   - Consumer: `_stage_gain_compute` per-bin masking using
     `enr_transparent_[k]` instead of scalar threshold
   - Detector: Phase 0.3 chooses subband vs aggregate
2. **D-aux1**: `X2_noise_floor` minimum-statistics per-bin (likely +
   pcb1N reverb cohort gain)
3. **D-aux2**: `WeightEchoForAudibility`-equivalent — already shipped
   v3.14 Arc R, **mark this gap closed**

**Out-of-scope for D** (kept in v3.19+ backlog):
- Reverb port (deferred to dedicated v3.19 reverb arc per Phase 0.4 doc)
- Non-linear-mode RES (depends on Phase C `usable_linear_estimate` first)
- `LimitHighFrequencyGains` rewrite — audit-only sprint, not port

**Estimated LOE**:
- D-core alone: 5-7 sprints
- D-core + D-aux1: 7-10 sprints
- D-core wired to subband detector: +2 sprints

**Hard bar (D-core)** from v3.18 plan:
- NE bucket Δdeg ≥ +0.010
- DT bucket Δdeg ≥ +0.005 (recovers v3.13 E2 Path 3 debt)
- FS Δecho ≥ -0.010
- cohort tail Δecho ≥ -0.05

## 7. Files this doc maps

**AEC3 sources analysed (Phase 0.2)**:
- [docs/aec3_extracts/src/aec3/suppression_gain.cc](aec3_extracts/src/aec3/suppression_gain.cc) (514 lines)
- [docs/aec3_extracts/src/aec3/suppression_gain.h](aec3_extracts/src/aec3/suppression_gain.h) (155 lines)
- [docs/aec3_extracts/src/aec3/residual_echo_estimator.cc](aec3_extracts/src/aec3/residual_echo_estimator.cc) (432 lines)
- [docs/aec3_extracts/src/aec3/residual_echo_estimator.h](aec3_extracts/src/aec3/residual_echo_estimator.h) (95 lines)

**Our analogues**:
- [python/aec.py:2111](../python/aec.py#L2111) `class ResFilter`
- [python/aec.py:2670](../python/aec.py#L2670) `_stage_residual_model`
- [python/aec.py:2833](../python/aec.py#L2833) `_stage_gain_compute`
- [python/aec.py:794](../python/aec.py#L794) `res_per_band_enr`
- [python/aec.py:566](../python/aec.py#L566) `res_unified_gain_floor`

**Cross-references**:
- Phase 0.3 — NE detectors: [docs/aec3_subband_ne_detector_mapping.md](aec3_subband_ne_detector_mapping.md)
- Phase 0.4 — reverb: [docs/aec3_reverb_mapping.md](aec3_reverb_mapping.md)
- Phase 0.6 — Phase D decision: `docs/v3_18_phase0_decision.md` (pending)
- v3.18 plan: `~/.claude/plans/se-aec-aec-main-hazy-lynx.md`
