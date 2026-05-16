# v3.13 E4.S4 verdict — 800-case detector fire-rate audit

**Status**: CLOSED with **DETECTOR REFINEMENT REQUIRED** before S5
suppressor. The detector fires on pitched residual generally — it
cannot distinguish FS-NL from NE-voice / DT-NE under current gating.
S5 design BLOCKED on S4.1 (NE/DT false-positive elimination).

**Date**: 2026-05-13

## Headline numbers

**Bucket aggregate fire rates** (audit-only, e4_nlp_enabled=True;
output byte-equal to baseline):

| Bucket | n | mean | median | p25 | p75 | p95 | max |
|---|---:|---:|---:|---:|---:|---:|---:|
| FS_static | 169 | 4.64% | 1.42% | 0.19% | 5.91% | 20.34% | 44.11% |
| FS_movement | 131 | 4.25% | 1.52% | 0.17% | 5.68% | 15.88% | 35.47% |
| DT_static | 186 | 3.59% | 3.42% | 1.45% | 4.99% | 9.19% | 13.89% |
| DT_movement | 114 | 4.04% | 3.68% | 1.82% | 5.61% | 9.39% | 22.00% |
| **NE** | 200 | **0.25%** | **0.00%** | 0.00% | 0.00% | 1.19% | 14.31% |

## Acceptance per design lock §3.2 S4

| Bar | Result |
|---|---|
| Byte-equal output (flag OFF) | ✓ PASS (verified in S3) |
| NL cohort ≥30% per case | **2/5 PASS** (Gsy0lC5 44.1%, 9xjhiFb 30.7%); 3 cohort cases in 20-24% range |
| NE bucket <1% per case | **PARTIAL** — aggregate 0.25% ✓ but 13/200 cases exceed 1%, 2/200 exceed 5%, worst 14.31% |
| All NL cohort max conf ≥ 0.5 | ✓ all 5 hit ≥0.83 |

NL cohort full table:

| Stem | Bucket | Fire rate | Max conf | PASS≥30%? |
|---|---|---:|---:|:---:|
| Gsy0lC5 (Type 1) | FS_static | 44.1% | 0.87 | YES |
| 9xjhiFb (Type 2) | FS_static | 30.7% | 0.83 | YES |
| WTdBhX (Type 2) | FS_static | 23.9% | 0.86 | NO |
| m4789f (Type 2) | FS_static | 21.1% | 0.87 | NO |
| IrQvqOTC_mvmt (Type 2) | FS_movement | 20.7% | 0.86 | NO |

## NE bucket worst-20 (the failure)

| Stem | Fire rate | Max conf | Mean conf |
|---|---:|---:|---:|
| `LHsrJBRGnUKiMC2mihEr0g_nearend_singletalk` | **14.31%** | 0.85 | 0.80 |
| `M2nCWTe4nUWo8IOhj2IwNg_nearend_singletalk` | **7.80%** | 0.84 | 0.78 |
| `JkDl1iAjcUelM740jzUz0A_nearend_singletalk` | 3.44% | 0.84 | 0.82 |
| `qkGW9Frbs0Gq5gdfsztA2g_nearend_singletalk` | 3.22% | 0.83 | 0.73 |
| `KgZ0y2EQJ0a4jvtsznBrvw_nearend_singletalk` | 2.80% | 0.85 | 0.78 |
| ... (8 more between 1.0-2.3%) | | | |

When NE detector fires, it fires CONFIDENTLY (mean_conf 0.73-0.82).
This is not random low-conf noise — these are real false positives
where clean NE voice produces a pitched signature that mimics NL.

## Root cause — detector cannot discriminate FS-NL from NE-voice

The current gating signals all evaluate to TRUE in NE bucket as well
as FS-NL:

| Signal | NE bucket | FS-NL bucket | Discriminative? |
|---|---|---|---|
| `far_active` | Often TRUE (lpb has voice) | TRUE | ✗ no |
| `filter_state in {refined_usable, coarse_learning}` | TRUE (filter happily idle since echo coupling = 0) | TRUE | ✗ no |
| `raw_output_rms > 0.05` | TRUE when speaking | TRUE | ✗ no |
| Pitched content via autocorr | TRUE (clean voice is pitched) | TRUE | ✗ no |
| N-frame continuity | TRUE | TRUE | ✗ no |

NE cases have all the same signals as FS-NL. The fundamental
difference — **"filter is actually canceling something"** — is not
captured in any of the current gating signals.

In FS-NL: filter cancels far-correlated echo; what remains is
mic-side NL distortion → pitched residual is unwanted.

In NE: filter doesn't cancel anything (no echo to cancel); residual
≈ mic = clean NE voice → pitched content is the desired signal.

Both look identical to the current detector.

## Additional finding — FS top fires include NL candidates outside cohort

Top-10 FS-bucket fire rates not in the 5-case NL cohort:

| Stem | Bucket | Fire rate | Was on listen 8? |
|---|---|---:|---|
| `VgSXlJJEI02dytkMm5UTzA_farend_singletalk_with_movement` | FS_mvmt | 35.5% | no |
| `sKXucFp4FUCJKo5d0G54Og_farend_singletalk` | FS_static | 28.4% | no |
| `hVqUmGvIlkO0LBUoE06Q3w_farend_singletalk` | FS_static | 27.1% | **YES (case 05 delay-broken)** |
| `NSdFS8g1dkCCAMRgHWLILQ_farend_singletalk` | FS_static | 20.9% | no |
| `pmzLFdKTzEixfU0l0furvA_farend_singletalk` | FS_static | 20.8% | no |
| `wr54weKzNkOcZ07hB04kzA_farend_singletalk` | FS_static | 20.7% | no |
| `vIf0JCJXwUuM90ngXBE18g_farend_singletalk` | FS_static | 19.8% | no |
| (3 more at 18-19%) | | | |

7+ FS cases not in the original cohort fire at NL-cohort-equivalent
rates. Two interpretations:
1. M3 > 9.0 screening (E4.S1) was too tight — missed real NL cases
   that the detector now picks up.
2. Detector firing pattern includes pitched residuals from filter
   non-convergence (delay-broken, partial NL, harmonic echo tail)
   not just true NL.

`hVqUmGvIlkO0LBUoE06Q3w` is listen case 05 with note "delay ~6000".
Not listen-marked NL, but fires at 27% — suggests pitched residual
from delay-broken filter also triggers detector. Detector is
over-broad.

## DT bucket cross-check

DT top-10 fire rates (all between 10-22%):

| Stem | Bucket | Fire rate |
|---|---|---:|
| `IrQvqOTC_doubletalk_with_movement` | DT_mvmt | **22.0%** |
| `XzmU8Bbs_doubletalk` | DT_static | 13.9% |
| `TGZ5Wq0S_doubletalk_with_movement` | DT_mvmt | 13.5% |
| ... (7 more between 10-13%) | | |

`IrQvqOTC_doubletalk_with_movement` = same speaker/recording stem as
case 07 (FS_movement version). Both variants fire similarly (22%
DT, 20.7% FS) — consistent with mic-side NL surviving across
bucket variants. This is real NL detection.

But the other DT top-10 cases (XzmU8Bbs etc.) likely include
NE-voice-driven false positives, same root cause as NE bucket
worst-20.

## What S5 suppressor would do with this detector

If S5 used current S4 detector as-is:
- Suppress ~30% of clean NE voice frames in LHsrJBRGn → audible
  damage to NE voice
- Suppress ~22% of DT frames in IrQvqOTC_doubletalk_mvmt — uncertain
  whether harmful (real NL?) or NE-damaging (false-pos)
- 7+ FS cases outside cohort also get suppressed — may or may not
  be NL

The 14.31% NE FP rate makes the detector unsuitable for promotion
to S5. The xrtntuju 5-clip listen regression test would fail
catastrophically — `LHsrJBRGn_doubletalk` is one of those clips,
and its NE counterpart fires at 14.31%.

## S4.1 path forward — discriminator

The detector needs a "filter is canceling something" signal. Three
candidate approaches:

### Option A — `mic_rms / raw_output_rms ratio`
- In NE: ratio ≈ 1 (filter doesn't cancel)
- In FS converged: ratio ≫ 1 (mic was large, residual is small)
- In FS-NL: ratio is moderate (partial cancellation, NL prevents
  full convergence) — needs measurement
- Risk: case 07-style heavy NL may have ratio near 1 too →
  false negative on the worst NL cases

### Option B — sustained filter ERLE (long-window EMA > threshold)
- Filter ERLE is positive when filter is canceling
- Sustained ERLE > X dB over N seconds = "this is a real echo path"
- Risk: same as Option A — heavy NL cases have low ERLE

### Option C — far↔raw_output cross-correlation low
- In NE: far and raw_output (≈mic) are uncorrelated (NE doesn't
  echo back) — ratio low
- In FS converged: far and residual uncorrelated by definition
  (filter removed correlation) — ratio low too
- In FS-NL: still uncorrelated because NL isn't linear → ratio low
- **NOT discriminative.** Discard.

### Option D — long-window far-vs-mic energy ratio
- `mic_lw / far_lw` over long window
- In NE: mic dominated by NE voice, ratio ≈ 1 or higher (depending
  on far volume)
- In FS: mic dominated by echo, ratio depends on ERL coupling
- Not clean — depends on relative volumes

**Recommendation**: Option A (mic_rms / raw_output_rms ratio over
long window) is the most direct test. Add as a fourth gate. Sweep
threshold like S3.1 did for raw_output_rms.

## Plan implications

- E4 arc S5+ suppressor BLOCKED on S4.1
- E4.S4.1: add `mic_rms / raw_output_rms` long-window ratio gate;
  re-run S4 800-case audit; verify NE worst-case < 1% AND NL cohort
  ≥ 30%
- If S4.1 fails to bring NE < 1% while keeping NL ≥ 30% → return
  to E4.S2 design lock with revised primary metric (back to listen
  evidence; investigate why LHsrJBRGn NE voice is so pitched in
  cepstrum)
- E4 arc currently in honest "detector audit revealed flaw,
  refinement needed" state — not a failure, just iteration

## Caveats

- 7+ additional FS cases fire at NL-cohort-equivalent rates. Some
  may be legitimate NL not caught by initial M3 > 9.0 screening.
  Listen validation on top-10 FS-outside-cohort would size true
  cohort more accurately. Defer to S4.2.

## Acceptance verdict

S4 acceptance MIXED. Promotion to S5 BLOCKED.

- ✓ Detector signal is real (FS top fires 20-44% vs NE median 0%)
- ✓ Listen-validated NL cohort all caught (5/5 hit, max conf ≥0.83)
- ✗ NE per-case bound (<1%) fails on 13/200 cases
- ✗ NL ≥30% bound fails on 3/5 cases

## Verification rules followed

1. Byte-equal output (audit-only, flag-gated)
2. Full 800-case bench (preset BALANCED, fl 832, cng True, j=4)
3. Pre-align with production default max=1024 (Path 3 shipped)
4. Bucket classification matches eval_aec_challenge harness
5. NL cohort = E4.S1 listen-validated set (no scope creep)
6. NE 200-case audit (full bucket, not sample)

## Artifacts

- Script: [tools/research/e4_s4_detector_audit.py](../tools/research/e4_s4_detector_audit.py)
- Output: `results/v3_13_e4_s4_audit/summary.md` + `rows.json` + `bucket_stats.json` (gitignored)
- Prior verdicts: [v3_13_e4_s1_verdict.md](v3_13_e4_s1_verdict.md),
  [v3_13_e4_s2_design_lock.md](v3_13_e4_s2_design_lock.md),
  [v3_13_e4_s3_verdict.md](v3_13_e4_s3_verdict.md)

## Next sprint

**E4.S4.1**: add `mic_rms / raw_output_rms` long-window ratio gate
(Option A above). Threshold sweep then re-audit on 800-case. Target:
NE per-case < 1% AND NL cohort ≥ 30% retained.
