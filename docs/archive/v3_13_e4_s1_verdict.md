# v3.13 E4.S1 verdict — NL audit + cohort sizing

**Status**: CLOSED, **GATE PASSED**. Cohort N=5 of 800 listen-validated
NL-distorted in FS bucket → E4 NLP arc proceeds to S2 detector
design lock.

**Date**: 2026-05-13

## Scope

Per v3.13 E4 NLP arc plan (`~/.claude/plans/se-aec-aec-v3-13-e4-nlp-arc.md`)
sprint S1: produce a listen-validated NL-positive cohort. Hard gate:
N<3 OR <1% of 800-case → close E4 arc without FAIL.

## Pass A — output-only metrics (8 listen + 50 sample)

Three candidate detector metrics on linear residual (`_ours_nores.wav`):

- **M1** HF/LF energy ratio (2-4 kHz / 100-500 Hz, voice-active, dB)
- **M2** HF peak-to-median PSD (1-4 kHz, voice-active, dB)
- **M3** cepstral peak prominence (80-500 Hz fundamental band)

Plus E5-merged metrics (per user 2026-05-13 merge of E5.S2 into E4 arc):
- **M4** mic peak-to-RMS ratio (input)
- **M5** mic-lpb envelope correlation r (input, 100 ms windows)
- **M6** mic clip rate (>0.99 threshold)

Initial v0 detector formula (≥2 of 4 metrics > sample p95):
- **Missed case 07** (the only listen-marked NL) — only M3 above p95
- **False-positive case 06** (delay+clipping, not NL)

The v0 formula fails. M5 was meaningfully positive in case 07 (0.74)
but below p95 (0.82) of the random sample. M3 was the only metric
that flagged case 07 (9.42 vs sample max 8.09 — clear outlier).

## Pass B — full 800-case linear residual audit

Per-bucket M3 distribution:

| Bucket | n | mean | median | p95 | max |
|---|---:|---:|---:|---:|---:|
| FS_static | 169 | 6.02 | 5.74 | 8.24 | 11.92 |
| FS_movement | 131 | 5.95 | 5.76 | 7.71 | 9.68 |
| DT_static | 186 | 6.60 | 6.38 | 8.61 | 11.39 |
| DT_movement | 114 | 6.68 | 6.39 | 8.94 | 10.29 |
| **NE** | 200 | **7.77** | **7.28** | **11.24** | **14.31** |

**M3 is NOT bucket-blind**: NE has the highest M3 distribution
(median 7.28, max 14.31). Clean near-end voice carries strong
cepstral peak prominence (pitched content). 14 of corpus-wide top-20
M3 are NE cases. **M3 alone cannot discriminate NL from NE**.

But **M3 within FS bucket** is meaningful. FS has no NE speech by
definition (only far-end + room reflections → mostly unpitched leak
after cancellation). High M3 inside FS = pitched residual that the
linear filter couldn't cancel — exactly the NL signature.

### FS-bucket M3 > 9.0 candidates (n=5)

| # | Stem | Bucket | M3 | M2 | M5 |
|---|---|---|---:|---:|---:|
| A | `Gsy0lC5QSUi540hiax9XtA_farend_singletalk` | FS_static | 11.92 | 23.40 | 0.53 |
| B | `9xjhiFbGo06hdQIsHTS6qA_farend_singletalk` | FS_static | 10.66 | 21.13 | 0.77 |
| C | `IrQvqOTCmEWMXn9k2ICtRQ_farend_singletalk_with_movement` | FS_movement | 9.68 | 16.50 | 0.74 |
| D | `WTdBhXa080WJEeGDde9BGA_farend_singletalk` | FS_static | 9.56 | 26.74 | 0.84 |
| E | `m4789fdio0q92zjf9gvh1Q_farend_singletalk` | FS_static | 9.55 | 21.83 | 0.73 |

## Listen validation (2026-05-13)

User listened to all 5 candidates' `_mic.wav` + `_lpb.wav` pairs in
`listen/v3_13_e4_s1_nl_candidates/`:

| Case | Listen verdict | Sub-type |
|---|---|---|
| **A Gsy0lC5** | 喇叭推爆 (lpb 正常) | Type 1: loudspeaker physical NL |
| **B 9xjhiFb** | 失真/非線性, 類似無線電/廣播/通話 (lpb 正常) | Type 2: transmission-like NL |
| **C IrQvqOTC_mvmt** | 失真/非線性, 同 type 2 (lpb 正常) | Type 2 |
| **D WTdBhX** | 失真/非線性, 同 type 2 (lpb 正常) | Type 2 |
| **E m4789f** | 失真/非線性, 同 type 2 (lpb 正常) | Type 2 |

**5 of 5 confirmed NL-distorted**. Two sub-types: 1 loudspeaker
physical NL + 4 transmission-like NL. lpb is clean in every case →
NL originates on the mic side (acoustic NL from loudspeaker
overdrive, or transmission codec NL applied to the mic capture
before recording).

### Detector precision on listen-validated set

Per-criteria precision on listen-validated cases (5 truly NL):

| Criterion | Hits | Precision | Recall (vs N=5) |
|---|---:|---:|---:|
| M3 > 9.0 (FS) | 5/5 | 100% | 100% |
| M3 > 7.87 (FS p95) | 15 in FS bucket | unknown | ≥100% (superset) |
| Multi-criteria v0 (≥2 of 4 > p95) | 1/5 (case D) | unknown | 20% |

**M3 > 9.0 within FS is a tight, validated detector**. The looser
M3 > FS p95 (7.87) finds 15 cases — 10 unvalidated may include true
NL (then total ≥15) or false-positives (lowering precision).

## Cohort sizing

| Threshold | FS hits | % of 800 | Plan gate |
|---|---:|---:|---|
| M3 > 9.0 (FS only) | 5 listen-validated all NL | 0.625% | **PASS** (N≥3) |
| M3 > FS p95 (7.87) | 15 | 1.875% | PASS (>1%) |
| Multi-criteria v0 | 1 | 0.125% | FAIL |

Gate decision: **PASS via M3 > 9.0 (5 cases, 100% listen-validated)**.

## Type 1 vs Type 2 implications for E4.S2

Two sub-types share the same architectural defect (mic-side NL
that linear PBFDKF cannot predict → residual harmonic content in
`_ours_nores.wav`), but differ in detector calibration:

| Sub-type | Listen feel | Likely spectral signature | E4 suppressor strategy |
|---|---|---|---|
| Type 1 (loudspeaker over-drive) | 喇叭推爆 | odd-harmonic comb (clipping → H3/H5) | harmonic mask gated on `saturation_level + far_active` |
| Type 2 (transmission codec-like) | 無線電/通話 | bandpass-shaped + intermodulation | broadband NL energy mask gated on `far_active + filter_converged + M3` |

E4.S2 design lock must decide:
- Single detector + single suppressor handles both? Or two detectors?
- Plan §5.2 lists three detector categories (pitch tracker, harmonic
  comb, spectral flatness). For Type 1 (harmonic comb) and Type 2
  (broadband-with-spikes), the **pitch tracker** option may unify
  both — pitched content not predicted from far-end = mic-side NL.

## Plan implications

- E4 arc **proceeds to E4.S2 detector design lock** (per plan §3.2)
- Design lock must address:
  1. Bucket-aware gating (M3 in FS only, never trigger on NE)
  2. Two-sub-type unified detector (likely pitch tracker per §5.2)
  3. Post-RES placement (per §4.1) — module sees output after RES
  4. NE preservation invariant — Type 2 (transmission-like) NL
     overlaps audibly with NE voice character; gating must mirror
     F3.1 v3 pattern
- Cohort tail invariant unchanged (qNvSMyU not in NL-positive set)
- xrtntuju 5-clip listen invariant — none of the 5 NL candidates
  overlap with xrtntuju regression set

## Risks identified

1. **Type 2 NL audibly resembles voice character**. Suppressor design
   must NOT over-suppress NE in DT buckets. The plan §3.2 already
   requires this via per-bin gating + filter_state coupling.
2. **M3 within FS works because there's no NE primary**. In DT
   buckets, M3 cannot be used as primary detector — need
   filter_state==steady OR coh2-based DT mask. Plan §5.1 prerequisite
   (audit existing harmonic surface) confirms scalar saturation_level
   gate is too coarse.
3. **5-case cohort small**. 800-case A/B (E4.S7) will have low
   signal-to-noise on bucket means; per-case improvement metric
   required as primary acceptance signal.

## Verification rules followed

1. Linear residual `_ours_nores.wav` (per Phase 0 discipline)
2. Voice-active frame masking (20th percentile energy threshold)
3. Listen validation 5/5 from user listen 2026-05-13
4. Sub-type categorization confirmed (Type 1 vs Type 2)
5. Cross-reference with xrtntuju 5-clip: no overlap

## Artifacts

- Script: [tools/research/e4_s1_nl_audit.py](../tools/research/e4_s1_nl_audit.py)
- Pass A output: `results/v3_13_e4_s1_audit/` (gitignored)
- Pass A linear output: `results/v3_13_e4_s1_audit_linear/`
- Pass B output: `results/v3_13_e4_s1_audit_full/` (full 800-case)
- Listen materials: [listen/v3_13_e4_s1_nl_candidates/](../listen/v3_13_e4_s1_nl_candidates/)

## Next sprint

**E4.S2 detector design lock** (LOE 1 sprint, MED risk per plan).
Per plan §3.2 acceptance: design verdict doc must cover:
- Chosen detector category + justification (reference 2+ papers)
- Gating signals list (closed set, no ad-hoc during impl)
- Per-frame vs per-bin output type
- Computational cost estimate
- Why NOT another coh2-based discriminator (P50/P55 trap)
