# E5.S1 verdict — per-frame saturation audit on listen 8-case

**Date**: 2026-05-13
**Branch**: `feature/v3.11-route-a`
**Predecessors**: [v3_13_e2_s1_verdict.md](v3_13_e2_s1_verdict.md), [v3_13_e2_s2_verdict.md](v3_13_e2_s2_verdict.md)
**Status**: **E5.S1 closed — F-E5 bundle gates do not fire on any of the 8 listen cases; "clipping" listen is NL distortion below the 0.95 sample-clip threshold**

## TL;DR

The listen exercise tagged cases 02 / 06 / 07 / 08 with "clipping"
or "嚴重 NL" descriptions, but **literal full-scale clipping is
absent on every case**. mic clip-rate at threshold 0.99 is 0.00% on
all eight, and the production `SaturationDetector`'s smoothed
`_saturation_level` max stays below 0.09 across the whole 800-case
pool of 8 cases — well below the 0.3 / 0.5 / 0.8 gate cliffs that
the F-E5 bundle and the existing always-on safety clamps are wired
to.

This means:

1. **F-E5 (the v3.11 sat-handling bundle) cannot help these 8 cases.**
   E5-1/E5-2/E5-3/E5-4 gates **all fire 0% of frames** on every case.
   The same is true for the production always-on clamps (`mic clip
   freeze ≥0.8`, `softclip ref ≥0.1`, `shadow safety ≥0.5`) — none
   of them activate.

2. **The audible "clipping" is acoustic-path NL distortion** —
   loudspeaker → mic acoustic path saturation that arrives at the
   mic ADC already low-pass-smoothed; peaks land in the 0.7-0.95
   band, below SaturationDetector's 0.95 threshold. The detector
   was designed around digital clipping signatures (consecutive
   peak-equal samples), which this acoustic NL does not produce.

3. **Two failure modes inside Group B**:
   - **Group B-mic_hot** (cases 02 / 07 / 08): mic frame peaks p99
     ≥ 0.9, mic-only sat-events present, online `_sat_mic` rises
     only to 0.04-0.09. Speaker-side NL likely.
   - **Group B-quiet** (cases 03 / 06): mic peaks moderate, low
     mic-lpb correlation. Listen labels (E1 mic quality, occasional
     glitch) suggest non-saturation root cause.

The E5 arc therefore should not target the F-E5 thresholds. The
gates can be **lowered** (E5.S2 candidate) or replaced with an
**acoustic-NL detector** keyed on mic-lpb correlation in the 0.7-
0.95 band (E5.S3 candidate). Otherwise the bundle is dead weight
on this part of the corpus.

## Per-case audit table

### Sample-level clip rates + offline saturation distribution

| # | Stem | scenario | mic peak (dBFS) | mic clip% @95/99/999 | lpb peak (dBFS) | lpb clip% @95/99 | mic longest 95-run (ms) | mic-only 95% frames | mic-lpb r | E2.S1 group | Listen |
|---|---|---|---:|---|---:|---|---:|---:|---:|---|---|
| 01 | 7GTxyTks | FS_static  | -0.00 | 0.00/0.00/0.000% | -0.39 | 0.00/0.00 | 0.06 | 0.09% | 0.247 | A | delay ~10000 samp + clipping |
| 02 | IrQvqOTC | FS_static  | 0.00 | 0.00/0.00/0.000% | -12.64 | 0.00/0.00 | 0.06 | 0.41% | 0.259 | B | delay ~1200 + clipping + radio |
| 03 | pcb1Nh0Z | FS_static  | -2.58 | 0.00/0.00/0.000% | -0.51 | 0.00/0.00 | 0.00 | 0.00% | 0.071 | B | gain變化, mic品質差 (E1) |
| 04 | S22FCqKD | FS_static  | -0.00 | 0.00/0.00/0.000% | -4.38 | 0.00/0.00 | 0.06 | 0.17% | -0.010 | A | delay ~9800 |
| 05 | hVqUmGvI | FS_static  | -7.99 | 0.00/0.00/0.000% | -6.29 | 0.00/0.00 | 0.00 | 0.00% | -0.010 | A | delay ~6000 |
| 06 | 5bJUo1K3 | FS_movement| -0.89 | 0.00/0.00/0.000% | -11.21 | 0.00/0.00 | 0.00 | 0.00% | **0.412** | B | delay ~1800 + clipping |
| 07 | IrQvqOTC | FS_movement| -0.00 | 0.00/0.00/0.000% | -6.02 | 0.00/0.00 | 0.06 | 0.16% | **0.579** | B | delay ~640 + 嚴重 NL |
| 08 | S22FCqKD | FS_movement| -0.02 | 0.02/0.01/0.000% | -10.22 | 0.00/0.00 | 0.25 | **1.11%** | -0.007 | A | delay ~10000 + occasional sat |

### Per-frame mic peak quantile distribution (10 ms frames)

| # | mic frame peak: p50 | p75 | p90 | p95 | p99 | p100 |
|---|---:|---:|---:|---:|---:|---:|
| 01 | 0.01 | 0.29 | 0.82 | 0.84 | 0.90 | 1.00 |
| 02 | 0.19 | 0.78 | 0.84 | 0.88 | 0.93 | 1.00 |
| 03 | 0.32 | 0.41 | 0.49 | 0.53 | 0.60 | 0.74 |
| 04 | 0.20 | 0.38 | 0.52 | 0.61 | 0.80 | 1.00 |
| 05 | 0.13 | 0.29 | 0.39 | 0.40 | 0.40 | 0.40 |
| 06 | 0.23 | 0.66 | 0.76 | 0.80 | 0.85 | 0.90 |
| 07 | **0.68** | **0.75** | **0.80** | **0.85** | **0.90** | 1.00 |
| 08 | 0.12 | 0.33 | 0.55 | 0.69 | 0.96 | 1.00 |

Case 07 (median 0.68!) operates **near full-scale for sustained
periods**. Cases 01 / 02 / 06 / 08 spend p75-p90 of their frames
above 0.66. Cases 03 / 05 are genuinely quieter (peaks under 0.74
even at p100).

### Online smoothed saturation values (production `_saturation_level`)

| # | online sat_level mean | p95 | max | online sat_mic max | online sat_ref max |
|---|---:|---:|---:|---:|---:|
| 01 | 0.003 | 0.012 | 0.022 | 0.044 | 0.000 |
| 02 | 0.010 | 0.025 | 0.037 | 0.073 | 0.000 |
| 03 | 0.000 | 0.001 | 0.004 | 0.007 | 0.000 |
| 04 | 0.001 | 0.005 | 0.016 | 0.031 | 0.000 |
| 05 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| 06 | 0.000 | 0.002 | 0.008 | 0.017 | 0.000 |
| 07 | 0.006 | 0.020 | 0.043 | 0.086 | 0.000 |
| 08 | 0.002 | 0.006 | 0.013 | 0.026 | 0.000 |

**No case ever crosses 0.1 on `_saturation_level`. `_sat_ref` is
zero everywhere** (lpb-side clip is below 0.95 on every case). The
mic-side sat indicator caps at 0.086 (case 07), which is **6×
under the f_e5_mic_softclip_threshold (0.3) and 12× under the
shadow-safety threshold (0.5)**.

### F-E5 gate fire-rates (production path, F-E5 disabled)

All 8 gates / 8 cases: **0.00%** fire rate. Confirmed by running
the bench-equivalent AEC stream on each case and inspecting per-
frame conditions.

| Gate | Trigger | Fires on any of 8 cases? |
|---|---|---|
| E5-1 mic softclip | `sat_mic > 0.3` | **No** |
| E5-2 main mu freeze | `_saturation_level > 0.5` | **No** |
| E5-3 error_psd reset | prev>0.5 & now<0.2 | **No** |
| E5-4 shadow_rise mask | `_saturation_level > 0.5` | **No** |
| [prod] mic clip freeze | `_sat_mic > 0.8` | **No** |
| [prod] softclip ref | `_sat_ref > 0.1` | **No** |
| [prod] shadow safety | `_saturation_level > 0.5` | **No** |

## Cross-reference with E2.S1 / listen findings

### Listen ↔ measurement reconciliation

| Listen description | Sample-clip evidence | Saturation EMA | Likely actual cause |
|---|---|---|---|
| "clipping" | 0.00% @95 (all) | < 0.04 max | acoustic NL, NOT digital clip |
| "嚴重 NL / 喇叭推爆" | 0.00% @95 (case 07) | 0.043 max | speaker driver NL → mic-lpb correlation 0.58 |
| "occasional sat" | 1.11% frames mic>0.95 (case 08) | 0.013 max | brief mic events, lpb quiet → mic-side artifact |
| "radio-like" (case 02) | 0.00% @99 | 0.037 max | gain limiting / heavy compression on far speech |
| "mic 品質差 (E1)" | 0.00% (case 03) | 0.004 max | low-end mic colouration, not saturation |

### Group breakdown (refined from E2.S1)

E2.S1 split the 8 cases into Group A (delay-broken: 01/04/05/08)
and Group B (delay-OK: 02/03/06/07). E5.S1 refines Group B:

- **Group B-mic_hot** (02 / 07): mic median frame peak ≥ 0.43, mic-
  lpb correlation 0.25-0.58. Listen labels: "clipping + radio" /
  "嚴重 NL". → Acoustic-path NL.
- **Group B-quiet** (03 / 06): mic median frame peak ≤ 0.43, low or
  modest mic-lpb correlation. Listen labels: "mic 品質差" /
  "clipping". → Mic-side issues / E1 quality / brief glitches.

Note: **case 06's mic-lpb r = 0.412** is suspicious (highish for
a "quiet" case). Median peak = 0.32 but the correlation is high,
indicating mic clipping co-occurs with lpb activity. Possibly the
"clipping" listen for case 06 is a smaller-amplitude NL than 07
but the same speaker-path mechanism.

Case 08 (Group A by delay): mic-only 1.11% sat-events with
mic-lpb r = -0.007 (independent). The "occasional sat" listen
note matches isolated mic-side events not driven by lpb.

## Which F-E5 bundle gates apply per case?

**Answer**: None of them, on any case. The bundle is calibrated for
sat_level thresholds (0.3 / 0.5 / 0.8) that are **2-30× higher**
than what `SaturationDetector` produces on this 8-case pool.

Why doesn't F-E5 fire even on cases where mic is at -0.00 dBFS?

`SaturationDetector.detect()` (`python/aec.py:1667`) counts:
- samples with `|x| > 0.95` → `clip_count`
- consecutive identical peak-equal samples → `consec_count`
- `raw_sat = min((clip_count + 2*consec_count) / n, 1.0)` per
  hop=160 frame
- EMA: attack α=0.3, release α=0.98

For case 07 (densest near-clip), `mic_clip_95 = 9.94e-06` = ~10
samples in the entire 25 s file. Per-frame `raw_sat` is at most
~0.06 (for the single frame containing all 10 samples), and the
asymmetric EMA (slow release) raises it to a steady-state ~0.04.

**The detector is keyed on "consecutive equal peak samples" — the
digital-clipping flat-top signature.** Acoustic NL distortion
(speaker driver clipping, recorded after the room low-pass) does
not produce flat tops; it produces softly compressed peaks that
land in the 0.7-0.95 range with normal sample-to-sample variation.
So the consecutive-detector contributes 0 to `raw_sat`.

This explains the F-E5 dead zone on the listen cases.

## Hypotheses for E5.S2 / E5.S3 / E5.S4

### E5.S2 — Lower SaturationDetector thresholds + acoustic NL extension

**Hypothesis**: replace the 0.95 fixed sample threshold with an
amplitude-distribution-based detector (e.g. peak-to-RMS ratio per
frame, or fraction of samples in [0.7, 1.0]). Re-bench F-E5 with
the new detector; the 4 gates may already do the right thing once
they actually fire.

Risk: changing the detector touches `_saturation_level` semantics
across the whole pipeline (8 read sites). Must be one-shot opt-in
flag.

### E5.S3 — mic-lpb correlation-based acoustic NL gate

**Hypothesis**: per-frame Pearson r between (mic_frame_peak,
lpb_frame_peak) — already computed in this audit — is the right
signal. r > 0.4 on a frame window indicates speaker-induced
mic NL.

Use this metric to:
- gate `_dt_residual_scale` upward (suppress RES gain when r high)
- raise `over_sub` for the RES echo model
- pause main filter mu (similar to E5-2)

Group 06 / 07 / 08 would benefit; Group 03 / 05 would be untouched
(low r).

### E5.S4 — Acoustic-path NL injection into RES echo model

**Hypothesis**: the linear AEC model fails on speaker NL because
the loudspeaker → mic path is a memoryless nonlinearity that the
linear filter cannot represent. Add a memoryless polynomial NL
model (Hammerstein-style) BEFORE the linear filter for the FS path.

Risk: doubles RES echo model state; only worth it if E5.S3 confirms
that "more echo-model power on mic-hot frames" actually reduces FS
leak on cases 02 / 06 / 07.

**Recommended ordering**: E5.S2 first (cheap, validates whether
F-E5 gates are right and only thresholds are wrong). E5.S3 second.
E5.S4 only if E5.S3 lands positive.

## Critical files reference

- `python/aec.py:1653` — `SaturationDetector` (the detector that
  refuses to fire on acoustic NL)
- `python/aec.py:5615-5637` — saturation detection + F-E5 / E5-1 /
  E5-3 wire-up in `AEC.process()`
- `python/aec.py:5812-5828` — mic clip emergency + F-E5 / E5-2 mu
  freeze
- `python/aec.py:6086-6094` — F-E5 / E5-4 shadow_rise mask
- `python/aec.py:589-603` — F-E5 config knobs (all default OFF)
- `tools/research/e5_s1_sat_audit.py` — this audit harness
- `results/v3_13_e5_s1_audit/` — per-case JSON + summary.md

## Sources

- v3.12 worst-FS listen exercise (`listen/v3_12_worst_fs/`)
- E2.S1 delay audit (`docs/v3_13_e2_s1_verdict.md`)
- F-E5 bundle (v3.11 Phase 1 Sprint 11-12 PASS, default OFF in
  BALANCED; `python/aec.py:589`)
- `SaturationDetector` algorithm (`python/aec.py:1653-1709`)
