# v3.13 E4.S5 design lock — Subtractive NLP suppressor

**Status**: DESIGN LOCKED. No code lands in S5 (per plan §3.3 — S5 is
design + isolated module spec). S6 implementation follows this spec.

**Date**: 2026-05-13

## Decision summary

| Aspect | Decision |
|---|---|
| **Mask family** | Fixed attenuation, **harmonic-pinned** |
| **Mask floor** | `g_min_e4_db = -12 dB` → mask ≥ 0.25 |
| **Mask upper bound** | 1.0 (no amplification) |
| **Harmonic pinning** | Gaussian at H1=f0, H2=2f0, ..., HK ≤ 6 kHz; σ = 50 Hz |
| **Per-bin attenuation depth** | proportional to `nl_confidence × harmonic_weight[k]` |
| **Time smoothing** | α = 0.7 (per plan §6.3 default) |
| **Frequency smoothing** | 3-bin [0.25, 0.5, 0.25] after time smoothing |
| **NE protection guard** | Hard block when F3.1 v3 mic_excess > 0.5 |
| **Cohort tail guard** | Implicit via detector gating (filter_state) |
| **Ramp-up** | 5-frame attack on nl_confidence cross-up |
| **Position** | Post-RES time-domain, standalone FFT/iFFT pair |
| **Default OFF** | `e4_nlp_suppress_enabled = False` until S7 acceptance |

## Why fixed attenuation (not Wiener / harmonic subtractive)

Per plan §6.1 trade-offs:

| Family | Trade-off |
|---|---|
| **Fixed attenuation** | Simple, bounded, easy to verify byte-equal default-OFF. **Chosen.** |
| Wiener-style | Requires NL energy estimate per-bin — recursive on what we're computing. Risk of feedback loop. |
| Harmonic subtractive (spectral subtraction) | Phase coupling risk; introduces musical noise; canonical complaint of spectral subtraction. |

Fixed attenuation suits the listen-validated NL cohort (E4.S1) because:
1. Type 1 (loudspeaker physical NL): harmonic comb structure; pinning
   to H_k positions selectively attenuates harmonics.
2. Type 2 (transmission codec NL): bandpass intermodulation has
   pitched fundamental; harmonic pinning catches the periodic
   component while leaving non-pitched residual untouched.
3. Bounded `[0.25, 1.0]` range cannot cause runaway suppression — the
   P50 / P58 trap (NE damage via over-suppression) is structurally
   prevented by the floor.

## Mask formula

Per hop, given detector outputs:
- `nl_confidence ∈ [0, 1]` (S4.1 acceptance: 0 when in NE bucket)
- `pitch_lag` (samples)
- `pitch_strength` (autocorr peak; high when actually pitched)

Fundamental frequency:
```
f0 = sample_rate / pitch_lag        # 80-500 Hz for human voice
```

Harmonic frequencies (K such that H_K ≤ 6000 Hz):
```
H_k = k * f0,  k = 1, 2, ..., K
```

Per-bin harmonic weight (Gaussian sum over harmonics):
```
sigma_hz = 50.0                     # ~3 bins at 31 Hz freq resolution
freq[bin] = bin * (sample_rate / fft_size)
harmonic_weight[bin] = sum_k exp(-((freq[bin] - H_k)^2) / (2 * sigma_hz^2))
harmonic_weight[bin] = clip(harmonic_weight[bin], 0, 1)
```

Raw per-bin mask:
```
attenuation_db_per_bin[k] = g_min_e4_db * harmonic_weight[k] * nl_confidence
mask_raw[k] = 10^(attenuation_db_per_bin[k] / 20)
mask_raw[k] = clip(mask_raw[k], 0.25, 1.0)     # floor + ceiling
```

At full attenuation (`nl_confidence=1` and `harmonic_weight=1`):
mask_raw = 10^(-12/20) = 0.251 → 12 dB attenuation.
At zero confidence: mask_raw = 1.0 (no attenuation).
At non-harmonic bins: mask_raw = 1.0 (only pinned bins are touched).

## Time + frequency smoothing

```
mask_time[k] = 0.7 * mask_time_prev[k] + 0.3 * mask_raw[k]
mask_smooth[k] = 0.25*mask_time[k-1] + 0.5*mask_time[k] + 0.25*mask_time[k+1]
```

Edge bins (k=0, k=N-1) use one-sided smoothing.

## NE protection guard

The detector already produces `nl_confidence = 0` when:
- `filter_state not in {refined_usable, coarse_learning}` (DT regime)
- `not far_active` (no echo)
- `raw_output_rms < 0.05` (no residual to flag)
- `mic_rms_ema / raw_rms_ema < 1.05` (NE bucket — no cancellation)

Three layers of NE protection. Suppressor adds a fourth as
defence-in-depth, mirroring the F3.1 v3 pattern at
[aec.py:2451-2454](../python/aec.py#L2451-L2454):

```
# Read F3.1 v3 mic_excess from RES diagnostic
ne_evidence_per_bin = res.get_mic_excess_per_bin()  # [0, 1] per bin
mask_smooth[k] = 1.0 if ne_evidence_per_bin[k] > 0.5 else mask_smooth[k]
```

When mic excess is high (NE voice in this bin), suppressor defers to
RES's per-bin evidence. This is consistent with F3.1 v3 promotion in
v3.10.6.

## Cohort tail invariant guard

The detector hard-gates on `filter_state in {refined_usable,
coarse_learning}`. PathChangeRegimeHandler decisions (`main_paused`,
`boost_q`, `reverse_copy`) cause `filter_state` to transition out of
these states briefly (typically to `coarse_learning` or `diverged`
during catastrophe recovery), so detector outputs nl_confidence=0
automatically. Cohort tail guard is **implicit via detector gating**.

S6 implementation must verify this: the qNvSMyU 7-case set must have
nl_confidence = 0 during regime handler fire windows.

## Ramp-up on detection onset

To prevent click artifacts on NL onset (when nl_confidence transitions
0 → high), attenuation is ramped over 5 hops (~50 ms):

```
nl_confidence_ramp = nl_confidence
if was_zero_prev_frame:
    ramp_counter = 5
if ramp_counter > 0:
    nl_confidence_ramp *= (1 - ramp_counter / 5.0)
    ramp_counter -= 1
```

This mirrors the raised-cosine fade used by `enable_td_constraint`
PBFDKF time-domain constraint.

## Position in pipeline

Post-RES, pre-output. Suppressor takes:
- Input: `final_output` (time-domain hop from ResFilter)
- Input: `nl_confidence`, `pitch_lag`, `pitch_strength` from detector
- Optional input: `mic_excess_per_bin` from RES diagnostic (for NE
  protection layer)
- Output: time-domain hop with harmonic suppression applied

Computational cost:
- 1 FFT (post-RES output) + 1 iFFT (after masking) per hop
- O(N log N) where N = window size (1024 at sr=16k)
- ~40k mul per hop ≈ 4 Mop/sec (negligible)

S6 may optimize by sharing FFT with ResFilter's internal spectrum.
Out of scope for S5 design.

## Synth NL test corpus spec

Per plan §3.3 S5, build a 30-clip synth corpus before bench:

**10 clean NE clips**: NE only, no echo, no NL. Sample 10 random NE
cases from the 800-case bench. Expected suppressor output:
mask ≈ 1.0 (NE detector gating prevents fires; suppressor passes
through). Acceptance: linear ERLE delta = 0 ± 0.5 dB.

**10 clean FS clips**: FS_static cases with M3 < 5.5 (control low-NL).
Sample 10 from E4.S1 audit JSON. Expected: occasional fires on
pitched echo tail; some suppression. Acceptance: linear ERLE delta
≥ +1.0 dB (light suppression OK).

**10 synth FS_NL clips**: take 10 FS_static cases with M3 ≈ 6 (mild
pitched residual). Apply `tanh(mic × 3.0)` to mic input pre-AEC to
inject controlled NL. Expected: clear NL signature, detector fires
sustained, suppressor attenuates harmonics 6-12 dB. Acceptance:
linear ERLE delta ≥ +3.0 dB.

**Note**: The 5 listen-validated NL cohort is NOT in synth corpus —
they remain the production listen-pool reference for S6+. Synth corpus
provides a controlled, reproducible test bed that doesn't deplete
production verification cases.

## Acceptance gates for S6 implementation

Per plan §3.3 S5 hard gate:

1. **Synth NE clips** linear ERLE delta = 0 ± 0.5 dB
2. **Synth FS clips** linear ERLE delta ≥ +1.0 dB
3. **Synth FS_NL clips** linear ERLE delta ≥ +3.0 dB
4. Byte-equal output when `e4_nlp_suppress_enabled = False`
5. Cohort tail (qNvSMyU) nl_confidence = 0 during regime fires
6. Default OFF respected

If isolated bench fails on bars 1-3, return to S2/S5 design lock with
revised detector or mask.

## Trap analysis

| Trap | Predecessor failure | Mitigation in this design |
|---|---|---|
| P50 (NE damage) | `nearend_protect_dt` broke FS Δecho −1.328 (8× audibility bound) | Mask floor `−12 dB / 0.25` hard bounded; F3.1 v3 mic_excess guard adds 4th NE protection layer; NE detector gating already 0% FP |
| P52 (cohort tail) | ShadowCopyController retirement broke catastrophe defence | Suppressor is post-RES additive; PBFDKF/regime untouched; implicit gating via filter_state ensures mask=1.0 during regime fires |
| P55 (separation) | Enzner-Vary canonical discriminator 7.01 dB on 20 dB bar | Detector already validated 5/5 listen-validated separation; per-bucket NE 0%; suppressor inherits detector reliability |
| P58 (4-cap retired) | AEC3 ERLE-divide killed FS Δecho −0.674 | Mask stacks multiplicatively with RES 4-cap; preserves base `g_min_db=−20 dB` floor; 4-cap untouched |
| F2.2 (EMA threshold) | Failed at 17 reg / 8 imp | Mask formula has no threshold sweep; bounded operations only |
| F-E5 (saturation gate) | 0% fire on listen 8 | Suppressor consumes nl_confidence (post-cancel-ratio gated), not saturation |
| **Musical noise** (spectral subtraction classic) | Generic ASR trap | Time + frequency smoothing (3-bin + α=0.7 EMA) standard mitigation |
| **Phase coupling** (subtractive families) | Audible distortion | Not applicable: chose multiplicative mask, not subtractive |

## Plan implications

- **E4.S6 (isolated module bench)**: 2 sprints. Synth corpus + impl +
  bench. Hard gate at synth ERLE thresholds before proceeding.
- **E4.S7 (800-case A/B)**: 1 sprint. CRITICAL — this is the
  "P50 / P52 / P55 / P58 kill-zone" where prior refactors failed.
  Expected outcome: borderline pass with FS positive, DT slight
  negative; tuning in S8.
- **E4.S8 (NE guard tuning)**: 2 sprints if S7 partial pass.
- **E4.S9-S12 (acceptance)**: 4 sprints.

Total remaining: 9 sprints (S6 × 2, S7, S8 × 2, S9-S12 × 4) to v3.13
production candidate.

## Out of scope (S5)

- Implementation (S6)
- 800-case A/B (S7)
- NE guard tuning (S8)
- xrtntuju listen regression (S10)
- C-port handoff (S12)

## References

1. **WebRTC AEC3 SubtractiveNLP** (Chromium source). Post-RES NL
   suppression placement; pitch-driven harmonic attenuation. Same
   architectural pattern.

2. **Boll, S. F. (1979).** "Suppression of acoustic noise in speech
   using spectral subtraction." IEEE Trans. ASSP 27. Classic
   subtractive RES framework; multiplicative mask is the modern
   stable variant that avoids Boll's musical-noise problem.

3. **Hänsler-Schmidt 2008.** "Acoustic Echo Cancellation" textbook,
   chapter on non-linear residual echo suppression. Multiplicative
   gain with floor is canonical.

4. **AEC.S1 → S4.1 listen evidence** (this project). Both NL
   sub-types (loudspeaker physical, transmission codec) preserve
   pitched fundamental. Harmonic-pinning catches both without
   broadband over-suppression.

## Verification rules followed

1. Plan §6 default recommendation (fixed attenuation) adopted with
   harmonic pinning add-on derived from E4.S1 spectral analysis.
2. Mask bounds `[0.25, 1.0]` hard-coded in design — cannot be
   tuned out.
3. NE protection 4-layer defence-in-depth (detector gates + mic
   excess guard).
4. Cohort tail invariant implicit via detector filter_state gating.
5. Synth corpus avoids depleting listen-validated production set.

## Artifacts to produce in S6

- `python/aec.py`: new `SubtractiveNLPSuppressor` class (~100 LoC)
- `python/aec.py`: AecConfig fields:
  `e4_nlp_suppress_enabled` (default False),
  `e4_nlp_suppress_g_min_db` (default -12.0),
  `e4_nlp_suppress_time_alpha` (default 0.7),
  `e4_nlp_suppress_harmonic_sigma_hz` (default 50.0),
  `e4_nlp_suppress_ramp_frames` (default 5)
- `tools/research/e4_s6_synth_corpus.py`: synth NL injection harness
- `tools/research/e4_s6_isolated_bench.py`: isolated suppressor bench
- `results/v3_13_e4_s6_synth/`: synth corpus output
- `docs/v3_13_e4_s6_verdict.md`: isolated bench verdict
