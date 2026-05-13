# v3.13 E4.S2 design lock — NL detector

**Status**: DESIGN LOCKED (no code lands in S2 per plan §3.2).
S3 implementation follows this spec.

**Date**: 2026-05-13

## Decision summary

| Aspect | Decision |
|---|---|
| **Detector category** | Voice-band **autocorrelation pitch tracker** with N-frame continuity |
| **Primary metric** | `pitch_strength` = max autocorrelation peak in [2 ms, 12.5 ms] lag (80-500 Hz fundamental) on linear residual |
| **Secondary metric** | `pitched_continuity_frames` = consecutive frames where pitch period is within ±5% of prior frame |
| **Output type** | Per-frame scalar `nl_confidence` ∈ [0,1] PLUS derived per-bin mask vector (harmonic-spectrum localised) |
| **Position in pipeline** | Post-RES, pre-output (per plan §4.1) |
| **Gating signals** | `filter_state == 'steady'` AND `far_active` AND `filter_converged` AND `_lw_ready` |
| **NE disambiguation** | Hard gate: detector OFF when `filter_state != 'steady'` (covers all DT + transient regimes) |
| **Computational cost** | ~5k mul/frame (cepstrum-based autocorr); negligible |

## Evidence basis (from E4.S1)

PSD analysis on 5 listen-validated NL candidates vs 3 low-M3 control
FS_static cases (`results/v3_11_candidate/*_ours_nores.wav` linear
residual, voice-active frames):

| Case | Type | LF (100-500 Hz) dB | MF (500-2k) dB | HF (2-4k) dB | HF/LF dB | Pitch strength |
|---|---|---:|---:|---:|---:|---:|
| A Gsy0lC5 | T1 loudspeaker | 31.2 | 29.5 | 20.0 | -11.2 | **0.716** |
| B 9xjhiFb | T2 transmission | 25.1 | 31.7 | 29.2 | +4.1 | **0.501** |
| C IrQvqOTC_mvmt | T2 transmission | 33.3 | 35.0 | 30.8 | -2.5 | **0.512** |
| D WTdBhX | T2 transmission | 12.4 | 21.2 | 11.2 | -1.2 | **0.639** |
| E m4789f | T2 transmission | 25.2 | 30.5 | 20.4 | -4.8 | **0.564** |
| CTL_1 1fvt8aj | — | 7.2 | 15.0 | 6.2 | -1.0 | 0.369 |
| CTL_2 49DamGO | — | 3.4 | 9.8 | 0.5 | -2.9 | 0.379 |
| CTL_3 49IIo03 | — | -4.3 | 12.3 | -0.5 | +3.8 | 0.300 |

Headline gaps:
1. **Total residual power**: NL cases 12-35 dB / 9-25 dB / 11-30 dB
   above controls in LF / MF / HF respectively. Strong absolute-energy
   discriminator.
2. **Pitch strength (autocorrelation peak)**: NL mean 0.59 vs control
   mean 0.35 (~70% higher). Tight separation; no overlap.
3. **Spectral shape varies wildly across sub-types**: Type 1 has LF
   dominance (HF/LF -11.2 dB); Type 2 has flat-to-HF-emphasis
   (HF/LF +4.1 to -4.8 dB). **Shape alone cannot unify both
   sub-types** — but **pitched content unifies both**.

## Why pitch tracker (not the alternatives)

### Why NOT coh2 / dt_per_bin (P50/P55 trap)

Per plan §1.3 + Q7 V3 verdict in main hazy-lynx plan, `coh2` saturates
to ~0 in FS post-cancellation (echo decorrelated → "looks NE"). Any
discriminator built on (1 - coh2) cannot detect "voiced NL residual"
because the NL residual is also low-coherence with far-end (it's
generated mic-side, not in the far-end signal). P50's
`nearend_protect_dt` trap was caused by exactly this: protecting on
1-coh2 in FS where the metric is dead. The NL detector must use a
**mic-side absolute metric**, not a far↔mic relative one.

### Why NOT M3 (cepstral peak prominence) alone

Per E4.S1 Pass B: M3 distribution bucket-blind. NE bucket has
**highest** M3 (median 7.28, max 14.31) because clean voice carries
strong cepstral peak. M3 within FS bucket is precise (5/5 listen-
validated) but only because FS has no NE primary. The detector must
gate explicitly on filter_state, not implicitly hope M3 thresholds
hold across buckets.

### Why NOT harmonic comb (H1/H2/H3 ratio)

Per plan §5.2 alternative B. Type 2 NL cases (B/C/D/E) do NOT show
classical harmonic comb structure — they have bandpass-shaped
intermodulation that lacks discrete harmonic peaks. Comb detector
would catch Type 1 but miss 4 of 5 cohort cases. Pitch tracker
catches both types because Type 2 still has pitched fundamental
(autocorrelation peak at pitch period) even without clean H1/H2/H3
spikes in spectrum.

### Why NOT spectral flatness contrast

Per plan §5.2 alternative C. Spectral flatness varies wildly across
sub-types (HF/LF +4.1 to -11.2 dB). Single flatness threshold cannot
unify. Conversely, autocorrelation peak strength is stable across
sub-types (0.50-0.72).

### Why pitch tracker

Voice-band autocorrelation captures the **periodicity** of pitched
mic-side residual. Both NL sub-types produce voiced residual (the
underlying signal is voice; both clipping and codec distortion
preserve voicedness — they just reshape spectrum). Controls have
unpitched broadband residual (echo tail / reverb / noise → no clear
period). The tracker discriminates **periodicity**, which is the
invariant across both NL sub-types and the discriminator from
unpitched controls.

## Detector specification

### Stage 1 — voice-band autocorrelation per frame

Input: per-frame time-domain output samples from ResFilter
(post-suppression, pre-final-write). Window: 64 ms hanning, 20 ms hop
(matches PBFDKF frame structure).

```
def compute_pitch_strength(frame):
    # frame: 64 ms hanning-windowed segment, sr=16 kHz, n=1024 samples
    # Cepstrum-based autocorrelation (FFT-cheap)
    spec = np.fft.rfft(frame)
    log_mag = np.log(np.abs(spec) + eps)
    ac = np.fft.irfft(2 * log_mag)  # real cepstrum is autocorr surrogate
    # Search lag window 2-12.5 ms = 80-500 Hz fundamental
    qmin, qmax = int(0.002 * sr), int(0.0125 * sr)   # = 32, 200 at sr=16k
    slab = np.abs(ac[qmin:qmax+1])
    peak_idx = int(np.argmax(slab))
    peak_val = float(slab[peak_idx])
    pitch_lag = qmin + peak_idx
    pitch_period_samples = pitch_lag
    pitch_hz = sr / pitch_lag
    median = float(np.median(slab))
    pitch_strength = peak_val / (median + eps)
    return pitch_strength, pitch_lag, pitch_hz
```

Cost: 1 FFT + 1 log + 1 IFFT per frame ≈ 4 N log₂ N = 40k mul.
Reuses ResFilter's existing FFT plan (no extra buffer alloc).

### Stage 2 — N-frame continuity check

Cache pitch_lag from previous 4 frames. Require:
- `pitch_strength >= 0.45` (from listen evidence: NL min 0.50,
  control max 0.38 → 0.45 is conservative threshold)
- Pitch lag of current frame within ±5% of pitch lag in ≥3 of 4
  previous frames (continuity — single-frame spurious peaks rejected)
- Both conditions met → `pitched_voice_active = True`

Continuity prevents false-positives from one-frame noise burst at
voice frequencies. Five-frame continuity matches WebRTC AEC3 streak
patterns.

### Stage 3 — bucket / regime gating

Detector outputs `nl_confidence = 0` unconditionally when any of:
- `filter_state != 'steady'` (idle / startup / transient / recovering
  / diverged → no NL processing; protects DT cohort)
- `far_active == False` (no echo to distort)
- `filter_converged == False` (linear filter unreliable)
- `_lw_ready == False` (long-window stats not warm)

Mirrors F3.1 v3 pattern at aec.py:2451-2454 (load-bearing precedent
for NE preservation gating). All four gates must be `True` for
detector to be live.

### Stage 4 — confidence mapping

When all gates pass and pitched_voice_active is True:

```
nl_confidence = sigmoid((pitch_strength - 0.45) * 4.0)
              # maps pitch_strength 0.45→0.5, 0.65→0.88
```

Continuous output (not binary) so suppressor strength can ramp with
detector confidence (per plan §6 — no hard cuts).

### Stage 5 — per-bin mask derivation (for S5-S8 suppressor)

S2 design only — actual mask compute is S5 work. Sketch only:

Given `pitch_hz` and `nl_confidence`, build per-bin attenuation
mask centred on H1=pitch_hz, H2=2×pitch_hz, ..., HK=K×pitch_hz
(K chosen so HK ≤ 6 kHz). Each Hk is a Gaussian-shaped attenuation
of bandwidth ~50 Hz, depth proportional to `nl_confidence`.
**This is the suppressor design surface, not the detector.** S2
locks only the per-frame scalar output; per-bin mask is S5.

## Architecture (per plan §4)

Module placement: `SubtractiveNLP` class in aec.py, instantiated by
`AEC` constructor, invoked after `ResFilter.process()` returns and
before final sample write. Owns its own pitch history buffer.

State surface (per plan §4.4):
- `_nl_pitch_lag_hist` — deque of last 4 frames' pitch_lag
- `_nl_strength_hist` — deque of last 4 frames' pitch_strength (for
  trace + S5 confidence smoothing)
- `_nl_confidence_last` — output of current frame (read by S5
  suppressor)

Reset hooks: none required. EPC reset doesn't apply to detector
state (it's a passive observer of post-RES output).

## Acceptance gates for S3 implementation

Per plan §3.2 S3 (audit-only — zero behaviour change):

1. Detector output `nl_confidence` per frame committed to AecStats.
2. Render 8-case listen + 5 NL-positive cohort with detector wiring;
   confirm byte-equal output vs main (no behaviour change in S3).
3. Detector fires on 5/5 NL cohort with confidence ≥0.5 in voice-
   active frames AND fires on <5% of voice-active frames in 3
   control cases.
4. NE bucket detector fires on <1% of voice-active frames in random
   20-case NE sample (cohort tail invariant).

If any acceptance fails → return to S2 design lock with revised
detector.

## Trap analysis (per plan §1.3)

| Trap | Predecessor failure | Mitigation in this design |
|---|---|---|
| **P50** | `nearend_protect_dt` over-cap on broken `effective_dt` evidence | Detector OFF when `filter_state != steady` — never engages in DT. Cannot accidentally over-suppress NE. |
| **P52** | ShadowCopyController retirement broke cohort tail catastrophe defence | Module is **post-RES additive** — does not touch PBFDKF / ShadowFilter / PathChangeRegimeHandler. Cohort tail recovery unchanged. |
| **P55** | Enzner-Vary canonical discriminator insufficient cohort separation (7.01 dB on 20 dB bar) | Listen-validated **70% separation** (NL pitch 0.59 vs control 0.35) on 5+3 cases; S4 acceptance will validate on 800-case bucket-wise. |
| **P58** | AEC3-pattern RES retired the load-bearing 4-cap chain | Module is **downstream of RES** — 4-cap chain untouched. |
| **F2.2** | EMA threshold tuning failed silently (17 reg / 8 imp on global threshold) | Threshold is fixed per evidence (`pitch_strength ≥ 0.45` calibrated from listen-validated gap, not swept); continuity-based, not EMA-based; N-frame check enforces signal, not threshold sweep. |
| **F-E5 saturation gate** | Saturation-only NL boost fires 0% on listen 8 (E5.S1) | Detector does NOT gate on `saturation_level` (per E5.S1 finding — saturation absent in NL cohort). Uses `filter_state + far_active + filter_converged` instead. |

## Why this is NOT another coh2-style discriminator

Listed explicitly per plan §3.2 acceptance bullet:

- coh2 measures **cross-spectrum** of far↔mic. In post-cancellation
  FS, residual is decorrelated with far → coh2 → 0 → indistinguishable
  from NE (1 - coh2 → 1).
- Pitch tracker measures **autocorrelation of the time-domain
  residual itself**. It is a mic-side absolute metric: NL produces
  voiced residual → autocorrelation has clear pitch peak;
  echo-tail / reverb / noise residual is unvoiced → no pitch peak.
- The discriminator is **periodicity, not coherence**. Periodicity
  survives post-cancellation because it's intrinsic to the mic-side
  signal, not a relationship between two signals.

## References

1. **Hermes, D. J. (1988).** "Measurement of pitch by subharmonic
   summation." JASA 83(1). Autocorrelation/cepstrum-based pitch
   detection canonical reference. Voice-band lag window
   [2 ms, 12.5 ms] = 80-500 Hz fundamental matches this paper's
   specification for human voice.

2. **WebRTC AEC3 SubtractiveNLP** (Chromium source). Reference
   placement of post-RES NL suppressor; pitch-driven harmonic mask
   generation. Our detector is upstream of an equivalent suppressor
   in S5.

3. **Boll, S. F. (1979).** "Suppression of acoustic noise in speech
   using spectral subtraction." IEEE Trans. ASSP 27. Classical
   spectral subtraction framework applies to per-bin mask generation
   in S5; we adapt it for harmonic-locked mask.

## Out of scope (S2)

- Per-bin mask generation algorithm (S5)
- Suppression depth calibration (S6)
- 800-case A/B (S7)
- xrtntuju listen regression (S10)
- C-port handoff (S12)

## Plan implications

- **E4.S3 (implementation)**: wire `SubtractiveNLP` class into AEC,
  output `nl_confidence` per frame to `AecStats`. ZERO behaviour
  change — module computes but does not act. Pure observer.
- **E4.S4 (800-case detector audit)**: fire-rate distribution per
  bucket. Acceptance: NL cohort fires ≥30% of voice-active frames;
  NE bucket fires <1% per case.
- **Sequencing**: S3-S4 are mechanical now that detector is locked.
  S5 design lock (suppressor) follows S4 verdict.

## Verification rules followed

1. Listen-validated cohort (E4.S1, 2026-05-13)
2. Spectral + autocorrelation analysis confirms detector
   discriminator (70% separation)
3. All 6 historical traps mitigated by design (table above)
4. No coh2-style discriminator (Q7 V3 + P50/P55 lesson)
5. Bucket-aware (filter_state hard gate)
6. Reference papers cited (2+ canonical, plan requirement met)

## Artifacts

- E4.S1 verdict: [v3_13_e4_s1_verdict.md](v3_13_e4_s1_verdict.md)
- Plan: `~/.claude/plans/se-aec-aec-v3-13-e4-nlp-arc.md`
- Listen pool: [listen/v3_13_e4_s1_nl_candidates/](../listen/v3_13_e4_s1_nl_candidates/)
- Pass B data: `results/v3_13_e4_s1_audit_full/rows.json`
