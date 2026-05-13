# v3.13 E2.S5 verdict — Path 3 deployable (default max=1024)

**Status**: SHIPPED. `eval_aec_challenge.estimate_delay` default
`max_delay_ms` raised from 250 → 1024 ms. Env override
`AEC_MAX_DELAY_MS` preserved for A/B.

**Date**: 2026-05-13

## Why this ships

E2.S2 800-case AECMOS bench already established the trade-off:

| Bucket | n | Δecho | Δdeg | Notes |
|---|---:|---:|---:|---|
| FS_static | 169 | +0.107 | -0.000 | strong win (top S22FC +13.5 dB linear ERLE per Phase 0) |
| FS_movement | 131 | +0.018 | -0.000 | win |
| DT_static | 186 | +0.014 | -0.050 | bucket Δdeg violation, **NOT** localised NE damage |
| DT_movement | 114 | +0.006 | -0.025 | bucket Δdeg violation, **NOT** localised NE damage |
| NE | 200 | +0.000 | -0.002 | invariant ✓ |

The audible-anchored regression gates all PASS:

| Gate | Result | Margin |
|---|---|---|
| **xrtntuju 5+2 DT NE clips Δdeg** | 4 unchanged, 2 improved (afHuFvfl +0.158, LHsrJBRGn +0.006) | 0 regressors |
| **Cohort tail qNvSMyU Δecho** | -0.004 | P52 floor -0.05 → 12× margin |
| **NE bucket Δdeg** | -0.002 | rule -0.005 → within margin |
| **Phase 0 linear ERLE on DT regressors** | All 7 show linear *cancellation increase* +1.14 to +3.61 dB | proves linear filter not degraded |

The DT bucket-mean Δdeg violation (10× / 5× the −0.005 rule) is the
**post-RES manifestation of better front-end linear cancellation**:
once the long-delay echo finally gets cancelled, the RES sees less
"echo evidence" and dials back its DT-preservation safety margin,
producing a small but bucket-wide AECMOS deg dip. This is RES
over-confidence in DT, not actual NE damage — confirmed by:

1. Phase 0 linear-ERLE on all 7 DT regressors shows +1.14 to +3.61 dB
   (more cancellation, less leak — linear filter is *better*, not
   worse)
2. xrtntuju anchored listen-positive clips: 0 regressors out of 7
3. NE bucket invariant: -0.002 well within rule (no actual NE damage)

Per user decision 2026-05-13, the DT bucket Δdeg unmasking is
accepted in exchange for the FS gain, with the v3.14+ Phase 3 RES
canonical refactor (per-state ENR tuple) earmarked as the fix.

## What changed in code

`python/eval_aec_challenge.py::estimate_delay` default `max_delay_ms`
parameter raised from `250.0` to `1024.0`. The env override
`AEC_MAX_DELAY_MS` continues to work and now serves the inverse role:
A/B regression to the old behavior is `AEC_MAX_DELAY_MS=250`.

Production preset `BALANCED` is unchanged. C-port is unchanged.
Filter length stays at 832 samples (52 ms). Only the offline bench
pre-align search window widens.

This is a **bench-harness fix**, not an algorithm change. The defect
this corrects: bench-side GCC-PHAT had a 250 ms search window while
the in-pipeline `DelayEstimator` runs with `max_delay_ms=1024`.
Aligning the two windows eliminates the regime where the bench
pre-align lands on a wrong cross-correlation peak even though the
algorithm itself could have found the right one.

## Risk assessment

| Risk | Severity | Mitigation |
|---|---|---|
| Regression on production C-port | NONE | C-port owns its own delay; bench harness only |
| Regression on cohort tail | NONE | Δecho -0.004 (12× margin to P52 hard floor) |
| Regression on NE-damage listen set | NONE | xrtntuju 0 reg / 2 improve |
| Bucket-mean DT Δdeg violation | ACCEPTED | RES unmasking confirmed; v3.14+ Phase 3 will recalibrate |
| Future bench A/B confusion | LOW | env override preserved; `AEC_MAX_DELAY_MS=250` reverts |

## Out of scope (per E2 plan)

- F-DelayTrack period tuning — E2.S3 closed NEGATIVE (flat)
- Classifier-gated conditional Path 3 — E2.S4 closed INFORMATIVE
  (classifier under max=250 mis-classifies pre-align-broken cases as
  STATIC; not a clean gate)
- DT RES regression mechanism — deferred to v3.14+ Phase 3 RES
  canonical refactor

## Plan implications

- E2 arc closes with Path 3 shipped + diagnostic-only classifier
- v3.13 next sprint candidates:
  - E5.S2 NL detector (recommended merge with E4 arc S2-S4)
  - v3.13.X dual filter incremental sub-arc (B6 / F-E1 / F-DelayTrack
    default-ON / B1 / C2)
- v3.14+ Phase 3 deferred: DT RES per-state ENR tuple

## Verification rules followed

1. Baseline JSON not stale (v3_11_candidate baseline scored 5:46 AM
   same day as Path 3 bench at 5:45 PM)
2. 800-case j4 `preset=balanced fl=832 cng=True` (matches v3.13 rule)
3. Cohort tail qNvSMyU Δecho ≥ -0.05 ✓ (-0.004)
4. NE bucket Δdeg ≥ -0.005 ✓ (-0.002)
5. xrtntuju 5-clip ✓ (anchored regressors none)
6. FS bucket Δecho ≥ -0.02 ✓ (+0.018, +0.107)
7. DT bucket Δdeg ≥ -0.005 ✗ → accepted as RES unmasking per Phase 0
   linear evidence (commit 2a82668)

## Artifacts

- Code change: `python/eval_aec_challenge.py::estimate_delay`
- Bench output: `results/v3_13_e2_s2_path3_800case/` (gitignored)
- Scores: `results/v3_13_e2_s2_path3_scores/{result.md,scores.json}`
- Baseline: `results/v3_11_candidate/scores.json`
- Prior verdicts: `docs/v3_13_e2_s2_verdict.md`,
  `docs/v3_13_e2_s3_verdict.md`, `docs/v3_13_e2_s4_verdict.md`
