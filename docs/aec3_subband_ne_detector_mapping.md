# AEC3 Subband Nearend Detector Mapping (Phase 0.3)

**v3.18 Phase 0.3 deliverable** — maps WebRTC AEC3 `subband_nearend_detector.cc`
+ `dominant_nearend_detector.cc` (commit `9310b29acd`) against our
aggregate `dt_indicator + shadow_dt → effective_dt` scalar chain.

Inputs:
- AEC3 `subband_nearend_detector.cc` (88 lines)
- AEC3 `subband_nearend_detector.h` (57 lines)
- AEC3 `dominant_nearend_detector.cc` (83 lines)
- AEC3 `dominant_nearend_detector.h` (57 lines)
- Our chain at [python/aec.py:3666-3674](../python/aec.py#L3666) (DTD
  → effective_dt scalar) and [python/aec.py:2833](../python/aec.py#L2833)
  ff. (`_stage_gain_compute` consumes scalar `effective_dt`)
- Phase 0.2 cross-ref: [docs/aec3_residual_pipeline_mapping.md](aec3_residual_pipeline_mapping.md)

## 1. AEC3 NE detector interface (`NearendDetector` base class)

Both detectors implement the same contract:

```cpp
virtual void Update(
    span<...kFftLengthBy2Plus1> nearend_spectrum,         // |Y[k]|² post-linear
    span<...kFftLengthBy2Plus1> residual_echo_spectrum,   // R2[k] from ResidualEchoEstimator
    span<...kFftLengthBy2Plus1> comfort_noise_spectrum,   // |N̂[k]|² estimate
    bool initial_state);
virtual bool IsNearendState() const;       // single binary
virtual void SetConfig(...);
```

Critical observations:
1. **Input is per-bin spectrum** (3 spectra × 65 bins × N_capture_ch).
   The detector has full per-bin information available.
2. **Output is a single bool** (`IsNearendState`). Per-bin information
   gets compressed to a scalar. **This is structurally identical to our
   `effective_dt`** — except ours is a 0..1 scalar with continuous semantics,
   theirs is binary with trigger/hold counters.
3. The detector polymorphism (subband vs dominant) is **config-driven**
   at construct time in `SuppressionGain`, not runtime-switchable.

## 2. `DominantNearendDetector` (the default)

[dominant_nearend_detector.cc:32](aec3_extracts/src/aec3/dominant_nearend_detector.cc#L32):

```cpp
low_frequency_energy = sum(spectrum[1..16])              // bins 1-15 LF integration

for each capture channel:
    ne_sum   = low_freq_energy(nearend_spectrum)
    echo_sum = low_freq_energy(residual_echo_spectrum)
    noise_sum= low_freq_energy(comfort_noise_spectrum)

    # Trigger gate
    if (!initial_state OR use_during_initial_phase)
       AND echo_sum < enr_threshold * ne_sum            # NE dominates echo
       AND ne_sum > snr_threshold * noise_sum:          # NE above noise floor
        trigger_counters[ch]++
        if trigger_counters[ch] >= trigger_threshold:
            hold_counters[ch] = hold_duration
            trigger_counters[ch] = trigger_threshold     # saturate
    else:
        trigger_counters[ch] = max(0, trigger_counters[ch]-1)

    # Fast exit
    if echo_sum > enr_exit_threshold * ne_sum
       AND echo_sum > snr_threshold * noise_sum:
        hold_counters[ch] = 0

    hold_counters[ch] = max(0, hold_counters[ch]-1)
    nearend_state_ |= (hold_counters[ch] > 0)            # OR across channels
```

**Properties**:
- LF-only (bins 1-15). HF is invisible to this detector.
- Hysteresis via `trigger_threshold` (rise) + `hold_duration` (fall) +
  fast-exit on strong echo.
- Multi-channel OR (any channel flags → state high).
- Uses **residual echo R2** (post-linear-filter, post-ERLE), not raw
  mic. So it benefits from upstream filter performance.

**Closest analogue in ours**: nothing single. Our `dt_indicator` is
energy-based (mic vs ref), `shadow_dt` is shadow-filter-based.
`effective_dt = max(dt_for_fs, shadow_dt)` is the OR-equivalent.

## 3. `SubbandNearendDetector` (the alternative)

[subband_nearend_detector.cc:35](aec3_extracts/src/aec3/subband_nearend_detector.cc#L35):

```cpp
for each capture channel:
    smoother.Average(nearend_spectrum[ch], smoothed_nearend)    # MovingAverageSpectrum
    noise_pow_sub1   = avg(noise[subband1.low .. subband1.high])
    ne_pow_sub1      = avg(smoothed_nearend[subband1.low .. subband1.high])
    ne_pow_sub2      = avg(smoothed_nearend[subband2.low .. subband2.high])

    if  ne_pow_sub1 < nearend_threshold * ne_pow_sub2          # SUBBAND1 quieter than SUBBAND2
        AND ne_pow_sub1 > snr_threshold * noise_pow_sub1:      # but still above noise
        nearend_state_ |= true
```

**Properties**:
- Smoothed nearend spectrum (block-EMA, configurable via
  `nearend_average_blocks`).
- Two configurable bands (`subband1` low/high indices, `subband2` low/high
  indices). In the default config these typically map to:
  - `subband1` = upper band (HF region)
  - `subband2` = lower band (voice region)
  - Trigger: HF (sub1) is quieter than voice-band (sub2) by a threshold
    AND HF is above noise floor.
- **Does NOT use `residual_echo_spectrum`** (param is unused —
  marked `/* residual_echo_spectrum */`).
- Decision logic is **structural NE detection** ("looks like speech
  formant pattern: voiced LF energy + quiet HF"), not residual-vs-NE
  comparison.

**Why this exists alongside DominantNearendDetector**:
- DominantNearendDetector can be **fooled by strong LF echo** that
  passes its ENR threshold (echo_sum > enr_threshold × ne_sum → state low
  even though NE is present)
- SubbandNearendDetector is **echo-agnostic** — it asks "does the
  nearend spectrum look speech-like" rather than "is residual echo
  smaller than nearend"
- Useful in **cohort tail** cases where residual estimate is unreliable

## 4. Side-by-side: AEC3 NE detection vs ours

| Property | AEC3 DominantNearend | AEC3 SubbandNearend | Our `effective_dt` |
|---|---|---|---|
| Frequency scope | LF bins 1-15 only | 2 configurable bands (typ. HF vs voice) | Broadband energy ratio + shadow-filter signal |
| Input | NE + R2 + noise | NE + noise (no R2) | mic_pwr / ref_pwr energies + shadow_e2_main/e2_shadow |
| Decision domain | Binary (with hold counter) | Binary | Continuous `[0,1]` scalar (post-max) |
| Hysteresis | Trigger threshold + hold duration + fast-exit | None (instant per-frame from smoother EMA) | None (per-frame `max(dt_for_fs, shadow_dt)`) |
| HF visibility | None | High (sub1 is typically HF) | Low (energy is broadband) |
| Voice-pattern detection | No | Yes (HF-quiet-when-voiced cue) | No |
| Echo-robustness | Fooled by LF echo | Echo-agnostic (doesn't use R2) | shadow-based variant is echo-aware |
| Wired into per-bin gain? | Yes — 2-way mask profile swap | Same | No — only floor-lift via scalar |
| Loss when reduced to scalar | Trigger/hold + multi-channel-OR squashes per-bin info | Single sub1/sub2 comparison preserves a structural cue | Energy-vs-energy is already scalar; per-bin info never enters |

## 5. The structural gap

**AEC3 path**: per-bin spectrum → NE detector (binary) → swap entire
per-bin mask profile (`nearend_params_` vs `normal_params_`) → per-bin gain.

**Per-bin info pathway**: AEC3's "per-bin mask profile" already carries
**LF→HF interpolated thresholds** ([Phase 0.2 §3.1](aec3_residual_pipeline_mapping.md)),
so even though the NE detector compresses to a single bool, the
DOWNSTREAM per-bin mask geometry preserves per-bin granularity. The
detector's job is to decide WHICH mask profile to use, not WHAT to do
per bin.

**Our path**: broadband energy ratio → `effective_dt` (scalar) →
floor-lift in per-bin gain.

**Where the loss happens**:
1. **Detector → State**: Both AEC3 and ours compress per-bin to scalar.
   But AEC3 does this AFTER seeing per-bin spectrum (with subband detector,
   structural cues survive); ours does this BEFORE (energy is already
   scalar input).
2. **State → Gain**: AEC3 swaps per-bin parameter shape based on state.
   Ours just lifts a scalar floor. **This is the bigger loss**.

## 6. Direct port-vs-redesign options for Phase D

### Option D-α: Port `SubbandNearendDetector` standalone (CHEAP)

**What**: Add `ne_state_subband` boolean from sub1/sub2 power compare on
post-linear NE spectrum.

**Wire**: Phase 0.2 §6 D-core consumer — use `ne_state_subband` to
choose between two pre-tuned per-bin mask profiles.

**Cost**:
- ~150 lines in Python (smoother + sub1/sub2 power sum + comparator)
- 6 new config fields (sub1_low/high, sub2_low/high, nearend_threshold,
  snr_threshold)
- Detector trace + audit-only flag for v3.18 Phase D.1

**ROI**: HIGH on FS_static reverb cohort + cases where DT_movement
suffers from broadband effective_dt being too quick to swing. Likely
modest on user's main concern (NE HF protection) because it gives a
better DETECTOR but not yet a better per-bin DECISION.

### Option D-β: Port `nearend_params_` two-way mask swap (CRUCIAL)

**What**: Pre-tune two per-bin mask threshold tables
`enr_transparent_lf2hf[k]`, `enr_suppress_lf2hf[k]`,
`emr_transparent_lf2hf[k]` × {`ne_params_*`, `normal_params_*`}.
On per-frame NE-state, swap which set drives `_stage_gain_compute`.

**Wire**:
- New `ResFilter._per_bin_mask_lookup` initialised from
  `AecConfig.res_per_bin_mask_lf` / `_hf` × 2 sets
- `_stage_gain_compute` uses `enr_t[k]` from current profile
- Profile switch driven by `aec_state.is_nearend_state` (Option D-α
  feeds this) OR by existing `effective_dt > 0.5` if D-α not done

**Cost**:
- ~300 lines in `_stage_gain_compute` (per-bin enr/emr decision rewrite)
- 12 new config fields (2 sets × 3 thresholds × 2 lf/hf anchors)
- Migration sprint for 35+ `effective_dt` consumers in gain stages
- Per-batch byte-equal verification

**ROI**: HIGHEST on user's main concern (NE HF protection). The 2nd
per-bin mask set (`nearend_params_`) explicitly relaxes HF suppression
when NE is detected — which is THE mechanism AEC3 uses for NE HF
preservation that our scalar floor can't replicate.

### Option D-γ: Port both (FULL)

D-α + D-β together. Subband detector feeds two-way mask swap. ~12-15
sprints LOE; matches AEC3 RES path geometrically.

## 7. NE HF protection — direct answer to user question 2026-05-15

User asked: "webrtc 怎麼做 residual 可以學的", in NE HF context.

**The mechanism**: AEC3 has **two per-bin masking parameter sets**.
`nearend_params_` has substantially higher
`enr_transparent_[k]` / `enr_suppress_[k]` / `emr_transparent_[k]` in
the HF range — meaning when NE is detected, the suppressor demands much
more residual echo before it starts attenuating HF bins. The
**NE detector** decides which set to use; the **per-bin LF→HF mask**
gives HF its protection.

Our ResFilter has:
- Per-bin gain compute (good — already there)
- Per-band `block_lf` ENR tilt (good — v3.14 Arc R; equivalent to
  `WeightEchoForAudibility` 3-band weighting)
- Scalar `effective_dt` floor lift (this is the missing piece — when
  NE is detected, we lift a uniform floor but don't relax HF-specific
  thresholds)

**The fix per AEC3 pattern**: Phase D-β port. The substrate from D-α
(subband detector) is the input; the meat is D-β (two-set per-bin mask
swap). D-γ is the full AEC3-shape solution.

## 8. v3.17 closure cross-reference

v3.17 B.2 closure (`docs/v3_17_b2_reverb_aware_closure.md`) flagged
"single-feature trigger N=1 cohort + RES override unsolved; full
mechanism R&D needs 5-8 sprints" — this is exactly what D-γ addresses.
The "RES override" v3.17 B.2 couldn't crack is the per-bin mask profile
swap, and the "multi-feature classifier" is essentially what
SubbandNearendDetector does (HF-quiet-when-voiced cue).

v3.17 B.1 closure (movement-rate DelayEst) had the target 0I0XMl3M
regress because EMA jittered → filter resets cycled. D-α + D-β do NOT
go upstream of the filter, so they don't share the failure mode. The
expected ROI for 0I0XMl3M from D-γ is small (it's a delay tracking
problem), but xrtntuju 5-clip DT NE cohort + FS_static reverb cases
should benefit.

## 9. Phase D opening recommendation

**Phase D OPEN**, scope = Option **D-γ** (full AEC3 RES NE pathway):

- D.1 (1-2 sprints): port `SubbandNearendDetector` standalone, default
  OFF, audit-only trace (sub1/sub2 power + smoother state)
- D.2 (1 sprint): pre-tune two per-bin mask profile tables
  (`nearend_*` / `normal_*` × 65 bins × 3 thresholds) from AEC3 defaults
  + adjust to our LF/HF anchor convention
- D.3 (2-3 sprints): rewrite `_stage_gain_compute` per-bin masking to
  use looked-up `enr_transparent_[k]` / `enr_suppress_[k]` /
  `emr_transparent_[k]`, default-OFF behind `res_aec3_mask_profiles`
- D.4 (1-2 sprints): migrate 35+ legacy `effective_dt` floor-lift
  consumers to either keep (FS path) or replace (NE/DT path) per
  per-bin compute
- D.5 (1 sprint): tune two profile sets — 60-case grid + nores listen
  on xrtntuju 5-clip + pcb1N FS_static + 0I0XMl3M cohort tail
- D.6 (1 sprint): 800-case AECMOS bench + nores listen + ship gate

**Total LOE**: 7-10 sprints

**Hard bar** (v3.18 plan §"Phase D"):
- NE bucket Δdeg ≥ +0.010 (HF protection improvement)
- DT bucket Δdeg ≥ +0.005 (recovers v3.13 E2 Path 3 DT debt)
- FS Δecho ≥ -0.010
- cohort tail Δecho ≥ -0.05

**Kill criterion**: NE Δdeg < +0.003 OR DT Δdeg < +0.003 → close per
§0.4, retain substrate.

## 10. Files this doc maps

**AEC3 sources analysed (Phase 0.3)**:
- [docs/aec3_extracts/src/aec3/subband_nearend_detector.cc](aec3_extracts/src/aec3/subband_nearend_detector.cc) (88 lines)
- [docs/aec3_extracts/src/aec3/subband_nearend_detector.h](aec3_extracts/src/aec3/subband_nearend_detector.h) (57 lines)
- [docs/aec3_extracts/src/aec3/dominant_nearend_detector.cc](aec3_extracts/src/aec3/dominant_nearend_detector.cc) (83 lines)
- [docs/aec3_extracts/src/aec3/dominant_nearend_detector.h](aec3_extracts/src/aec3/dominant_nearend_detector.h) (57 lines)
- [docs/aec3_extracts/src/aec3/nearend_detector.h](aec3_extracts/src/aec3/nearend_detector.h) — abstract base (not read in detail; trivial polymorphism interface)

**Our analogues**:
- [python/aec.py:3666-3674](../python/aec.py#L3666) `dt_for_fs` /
  `effective_dt` compute
- [python/aec.py:2833](../python/aec.py#L2833) `_stage_gain_compute`
  per-bin gain with scalar `effective_dt` floor-lift
- [python/aec.py:3178-3222](../python/aec.py#L3178)
  `ne_confidence = dt_per_bin` interpolation — closest existing per-bin
  NE-aware logic (but `dt_per_bin` derives from `max(effective_dt,
  1-coh2)`, still bottlenecked by scalar `effective_dt`)

**Cross-references**:
- Phase 0.2: [docs/aec3_residual_pipeline_mapping.md](aec3_residual_pipeline_mapping.md)
- Phase 0.4 (reverb): [docs/aec3_reverb_mapping.md](aec3_reverb_mapping.md) (pending)
- Phase 0.6 decision: `docs/v3_18_phase0_decision.md` (pending)
- v3.17 B.2 closure: [docs/v3_17_b2_reverb_aware_closure.md](v3_17_b2_reverb_aware_closure.md)
- v3.18 plan: `~/.claude/plans/se-aec-aec-main-hazy-lynx.md`
