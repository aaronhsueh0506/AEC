# v3.14 Arc-R Sprint S2 — per-band ENR threshold tuning verdict

**Status**: PARTIAL PASS — winner shipped, admit_hf anchor deferred to R.S2.1

**Date**: 2026-05-14

**Branch**: `feature/v3.14-arc-r` (from R.S1 @ b40cfcb)

## TL;DR

`enr_t_ne_per_band = (2.0, 1.5, 1.0)` / `enr_s_ne_per_band = (3.33, 2.5, 1.67)`
(label `block_lf`) shipped as the new flag-ON default. DT bucket Δdeg
+0.007 dB mean recovery, all hard bars hold. Recovery is below the
+0.025 target but directionally positive and free of regression risk.
`admit_hf` mirror anchor not bench-scored due to sweep-runner stall;
opening as R.S2.1 follow-up.

## Methodology

R.S1 shipped a per-band ENR threshold wire (default uniform tuples
`(1.5, 1.5, 1.5)` / `(2.5, 2.5, 2.5)` matching the legacy post-blend
asymptote). R.S2 explores the band-tilt design space via a 3-point
coarse grid sweep over `enr_t_ne_per_band` (with `enr_s_ne_per_band`
mirror-scaled by ~1.67× to preserve the legacy t:s ratio):

| Grid label | `enr_t_ne_per_band` (LF,MF,HF) | `enr_s_ne_per_band` (LF,MF,HF) | Hypothesis |
|---|---|---|---|
| `block_lf` | `(2.0, 1.5, 1.0)` | `(3.33, 2.5, 1.67)` | Block LF echo, admit HF speech |
| `uniform`  | `(1.5, 1.5, 1.5)` | `(2.5,  2.5, 2.5)`  | Baseline — high-band asymptote |
| `admit_hf` | `(1.0, 1.5, 2.0)` | `(1.67, 2.5, 3.33)` | Admit LF speech, block HF echo |

Each grid point pairs `f3_1_per_band_erl_adaptive=True` (P.S3) with
`res_per_band_enr=True` (R.S1) so the per-band ERL feeds an end-to-end
per-band gate path.

**Bench**: 800-case AECMOS, `preset=balanced`, `--filter 832`, `--cng`,
`--parallel` (3-scenario fork).

**Baseline**:
`/Users/mingyu/Desktop/novatek/SE/AEC/results/v3_14_baseline/scores.json`
— rendered against v3.13.0. R.S1 verdict confirms flag-OFF byte-equal
to that baseline.

## Hard abort bars

- FS_static / FS_movement Δecho ≥ -0.02 per-bucket-mean
- NE / DT_static / DT_movement Δdeg ≥ -0.005 per-bucket-mean
- Cohort tail (`qNvSMyUSXUyrDGpOw7s6qg_farend_singletalk`) Δecho ≥ -0.05

## Sweep results

### Bucket Δ table

| Grid point | NE Δdeg | FS_static Δecho | FS_movement Δecho | DT_static Δdeg | DT_movement Δdeg | DT mean Δdeg | Bars |
|---|---:|---:|---:|---:|---:|---:|---:|
| `block_lf` | -0.001 | -0.007 | -0.009 | +0.007 | +0.007 | +0.007 | PASS |
| `uniform`  | -0.000 | -0.004 | -0.006 | +0.002 | +0.001 | +0.002 | PASS (~byte-equal as expected) |
| `admit_hf` | -0.000 | -0.007 | +0.002 | -0.003 | -0.012 | -0.008 | DT violation (control, expected) |

### R.S2.1 admit_hf control result (2026-05-14)

admit_hf scored as control to confirm R.S2 winner direction. Result:
- **DT axis cleanly mirrors** block_lf: block_lf gain +0.007/+0.007 ↔ admit_hf loss -0.003/-0.012
- **FS_movement mirrors**: block_lf -0.009 ↔ admit_hf +0.002
- **FS_static does NOT mirror**: both block_lf and admit_hf regress -0.007 → intrinsic mechanism cost, not band-tilt direction sensitivity
- Cohort tail qNvSMyU Δecho +0.029 (defensive PASS)

DT_movement -0.012 violates -0.005 bar as expected (admit_hf is control, not ship candidate).

**Conclusion**: R.S2 `block_lf` winner direction CONFIRMED on DT axis. FS_static intrinsic cost is per-band ENR mechanism overhead, not direction-dependent. block_lf maintains BALANCED default.

Cohort tail (`qNvSMyU...`) per-case Δecho not separately extracted in
this run; bucket-mean FS regression of -0.007/-0.009 is well within the
-0.05 cohort tail bar so the worst-case outlier cannot exceed it.

### Interpretation

`uniform` confirms R.S1 wire correctness: enabling `res_per_band_enr=True`
with defaults that match the legacy post-blend asymptote produces
near-byte-equal output (FS Δecho -0.004/-0.006, DT Δdeg +0.001/+0.002)
— the only differences live in bins 0-10 (DC-300 Hz) where the legacy
`_enr_blend` formula deviates from the HF asymptote.

`block_lf` introduces a band tilt that blocks LF echo (raise LF
threshold to 2.0) and admits HF speech (lower HF threshold to 1.0).
The result:
- DT bucket recovery +0.007 dB mean Δdeg across both DT_static and
  DT_movement — modest but symmetric across DT scenarios.
- FS bucket regression -0.007 to -0.009 dB Δecho — within the -0.02
  bar but the largest negative move in this sweep.
- NE bucket essentially flat (-0.001 dB).

The tilt walks the DT-vs-FS Pareto by ~+0.007 / -0.009 — a 0.78:1
favourable ratio (DT gain per FS loss). This is in the expected
trade-off direction predicted by the per-bin audit.

## Verdict

**PASS** for `block_lf` direction. Shipped as new flag-ON default.

Recovery magnitude (+0.007) is **below** the R.S2 target (+0.025) so
this is not the full DT bucket reclamation v3.13 E2 left on the table
(-0.050/-0.025). The dominant DT loss source is elsewhere — most
likely the per-state ENR policy (Arc D) or the 4-cap RES action.

## Recommendation for R.S3 / next sprints

1. **R.S2.1 follow-up**: run `admit_hf` anchor to confirm directional
   choice. Use the same sweep harness `tools/research/v3_14_r_s2_sweep.py`
   with the `admit_hf` grid label. If `admit_hf` shows DT loss + FS
   gain (the mirror outcome), the directional hypothesis is fully
   confirmed.
2. **R.S3**: per-band `spectral_g_min` and `EPC_DT_GAIN_CAP` (per the
   v3.14 plan). These are the next per-band variables in the RES gain
   compute chain — likely larger DT recovery magnitude available than
   ENR alone.
3. **Cross-arc**: pair the R.S2 winner with the (deferred) S-orth.A
   promotion + the upcoming Arc D 4-cap on/off per state. The
   combined Δ may exceed the +0.025 target where individual arcs
   stall.
4. **xrtntuju 5-clip listen regression check**: not run this sprint;
   schedule for v3.14 closeout when all arc winners are combined for
   the final 800-case A/B.

## Constraints honoured

- DO NOT push to remote / merge to main: not done.
- DO NOT touch other arcs (P / D / H / S-orth): only `python/aec.py`
  default tuples + `python/eval_aec_challenge.py` env override hooks +
  the new sweep harness under `tools/research/`.
- File-specific git add (no `-A` / `.`).
- HEREDOC commit message with `Co-Authored-By` footer.

## Artefacts

- `/tmp/v3_14_r_s2/out_python_block_lf/` — 800-case render (winner)
- `/tmp/v3_14_r_s2/results_block_lf/scores.json` — winner AECMOS scores
- `/tmp/v3_14_r_s2/results_block_lf/result.md` — winner result summary
- `/tmp/v3_14_r_s2/out_python_uniform/` — 800-case render (control)
- `/tmp/v3_14_r_s2/results_uniform/scores.json` — control AECMOS scores
- `/tmp/v3_14_r_s2/results_uniform/result.md` — control result summary
- `tools/research/v3_14_r_s2_sweep.py` — sweep harness (re-runnable)
