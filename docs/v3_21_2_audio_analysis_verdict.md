# U1 Audio analysis verdict — v3.21.2 candidate (T1) HF formant preservation

**Date**: 2026-05-20
**Branch**: v3.21.2 HEAD (commit eab0d52)
**Baseline**: v3.21.0 (commit a537b65)

## Question

The v3.21.2 candidate fixes a bin-index unit-conversion bug in SuppressionGain
(HF cap was firing at 937 Hz instead of 3750 Hz, mask interpolation at
156-250 Hz instead of 625-1000 Hz). 800-case AECMOS reported DT_static deg
+0.092 / DT_movement deg +0.114. Does this AECMOS gain actually correspond
to recovery of voice formant energy that was being damaged in v3.21.0?

## Method

5 cases selected from the v3.21.0 DT_static worst-deg list ([docs/bench/v3_21_3aadd2d_baseline/balanced_aec3_result.md](bench/v3_21_3aadd2d_baseline/balanced_aec3_result.md))
— these are cases where the HF damage hypothesis predicts the largest fix impact.

For each case:

1. Render at `git checkout a537b65` (v3.21.0 baseline) — produces `audio_analysis_u1/<stem>_baseline.wav`
2. Render at `git checkout v3.21.2 HEAD` (T1 candidate) — produces `audio_analysis_u1/<stem>_t1.wav`
3. Compute STFT (`nperseg=512` matching production FFT) on mic / baseline_out / t1_out
4. Average band power across active frames (top 50% mic-energy frames as VAD proxy)
5. Compute per-band preservation ratio: `output_band_E / mic_band_E`
6. Compute T1 vs baseline improvement in dB per band

Voice formant reference bands (Chinese /i/ specifically):
- F1 ≈ 300 Hz, F2 ≈ 2300 Hz, F3 ≈ 3000 Hz

| Band | Range | Voice content |
|---|---|---|
| LF | 0-300 Hz | F0 fundamental + sub-formant; control region (no fix expected) |
| F1 | 300-1000 Hz | First formant region (B1+B2 mask interpolation fix target) |
| F2-F3 | 1-4 kHz | Critical formants (lgb HF cap fix target) |
| HF | 4-8 kHz | Sibilance / fricatives; mixed (cap region) |

Script: [audio_analysis_u1/analyze_bands.py](../audio_analysis_u1/analyze_bands.py).
Raw output: [audio_analysis_u1/band_analysis_results.txt](../audio_analysis_u1/band_analysis_results.txt).

## Per-case results

| Case | baseline AECMOS deg | ΔLF | ΔF1 | **ΔF2-F3** | ΔHF |
|---|---:|---:|---:|---:|---:|
| QK70KpLuZ0O43BBSWEZvHg | 1.249 | +0.03 | +0.07 | **+0.28** | +0.69 |
| sYQK1rJlwU2XCy20n0Sx9g | 1.272 | +0.07 | +0.05 | **+0.56** | -0.19 |
| yc5bFUGsR0GSfiGwTTpRWg | 1.308 | +0.17 | +0.78 | **+0.91** | -0.20 |
| QkRkwwFKVEar0WtcuvJsZg | 1.309 | +0.23 | +0.66 | **+0.47** | -0.32 |
| SgKY30fjT0G8e3kQL0RHSQ | 1.331 | +0.14 | +0.07 | **+0.18** | +0.61 |
| **Mean** | | **+0.13** | **+0.32** | **+0.48** | **+0.12** |

(All values are T1 vs baseline preservation-ratio change in dB. Positive = T1 preserves more energy in that band.)

## Findings

1. **F2-F3 (1-4 kHz) preservation: +0.48 dB mean, all 5 cases positive (+0.18 to +0.91 dB)**.
   This is the critical voice-formant band that the user reported as damaged
   ("400 Hz 以上就被砍" / Chinese /i/ distortion). T1 quantitatively recovers
   this band consistently across all 5 cases. The largest single-case gain
   (+0.91 dB on yc5bFUGsR0GSfiGwTTpRWg) corresponds to nearly 23% more F2-F3
   energy passing through the post-filter.

2. **F1 (300-1000 Hz) preservation: +0.32 dB mean, all 5 cases positive (+0.05 to +0.78 dB)**.
   The B1+B2 fix (mask interpolation boundary 156/250 Hz → 625/1000 Hz)
   moves F1 from the aggressive `mask_hf` zone into the interpolation zone,
   recovering F1 energy. Magnitude smaller than F2-F3 because F1 is closer
   to the boundary.

3. **LF (0-300 Hz) preservation: +0.13 dB mean — control region**.
   Below F1; not directly affected by either fix. Small positive trend
   probably reflects slightly less LF over-suppression as the post-filter
   stays in nearend mode more often (downstream effect of more accurate
   detection in F2-F3).

4. **HF (4-8 kHz): mixed, +0.12 dB mean but range -0.32 to +0.69 dB**.
   This is the HF cap region (cap now at 4 kHz instead of 937 Hz). Cases
   where mic has voice sibilance content in 4-8 kHz show preservation gains
   (e.g. QK70 +0.69, SgKY +0.61); cases where this band is mostly echo
   show slight regression (-0.19 to -0.32). Consistent with the FS_echo
   regression seen in the 800-case bench (T1 lets more HF echo through
   above 4 kHz, accepting that as Pareto cost).

## Verdict

**PASS** — T1 quantitatively recovers voice formant energy (F1 + F2-F3)
across all 5 worst-baseline-deg DT_static cases. The mechanism predicted
by the bin-index unit-conversion fix matches the observed energy
preservation profile:

- B1+B2 (mask interpolation) → F1 recovery
- lgb (HF cap) → F2-F3 recovery
- LF unaffected (control)
- HF mixed (Pareto cost = FS echo leak in voice-poor cases)

The AECMOS DT_static deg gain (+0.092) corresponds to real perceptual
formant preservation, not a spurious metric improvement.

## Limitations

- matplotlib unavailable in current Python env → no spectrogram PNGs
  produced. Quantitative band-power table substitutes; full visual
  spectrogram inspection deferred to future cycle if needed.
- VAD proxy = top 50% mic-energy frames; coarser than a real VAD. Active
  frames almost certainly include both nearend speech and FS-only echo
  segments, which slightly understates the F2-F3 gain (FS-only frames
  contribute echo energy that doesn't represent voice).
- HALT-U1 condition NOT triggered (HF improvement was clear). Proceed
  to U5 (FS recovery sweep).

## Files

| Artifact | Path |
|---|---|
| Analysis script | [audio_analysis_u1/analyze_bands.py](../audio_analysis_u1/analyze_bands.py) |
| Raw output | [audio_analysis_u1/band_analysis_results.txt](../audio_analysis_u1/band_analysis_results.txt) |
| 5 baseline output wavs | `audio_analysis_u1/<stem>_baseline.wav` |
| 5 T1 output wavs | `audio_analysis_u1/<stem>_t1.wav` |
| Input mic/ref wavs | `wav/aec_challenge_blind/doubletalk/<stem>_doubletalk_{mic,lpb}.wav` |
