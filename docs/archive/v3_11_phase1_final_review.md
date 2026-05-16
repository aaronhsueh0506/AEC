# v3.11 Phase 1 Final Review — Flag Promotion Recommendations

**Date**: 2026-05-13
**Branch**: `feature/v3.11-route-a`
**Baseline**: v3.10.6 (`results/v3_10_5_main/`)
**Verification framework**: linear-only evaluation (`tools/research/linear_eval.py`)

This document consolidates Phase 1 ablation results (8 sprints) and
delivers per-flag promote / hold / discard recommendations for v3.11
GA.

---

## Phase 1 ablation summary table

| Sprint | Flag | Verdict | Bucket mean ΔerleA | Cohort qNvSMyU | Cases > 0.1 dB | Top case |
|---|---|---|---|---|---|---|
| 1-2  | `shadow_r_reset_enabled` (B5) | **PASS** | +0.003 / +0.004 (FS) | +0.005 | 4 | TGZ5 +0.517 |
| 3-4  | `shadow_mu_state_aware` (B6) | **PASS** | +0.002 / +0.007 (FS) | +0.010 | 6 (5+/1-) | TGZ5 +0.520 |
| 5-6  | `f_e1_enabled` (F-E1) | **NEUTRAL** | +0.000 all | +0.000 | 0 (corpus N/A) | n/a |
| 7-8  | `f_delaytrack_enabled` (F-DT) | **NEUTRAL** | +0.000 all | +0.000 | 0 (corpus N/A) | n/a |
| 9-10 | `f_e3_enabled` (F-E3) | **FAIL** | -0.110 / -0.068 (FS) | +0.000 | 100+ regress | -1.037 |
| 11-12| `f_e5_enabled` (F-E5) | **PASS** | +0.002 / +0.001 (FS) | +0.000 | 14 | sKXucFp4 +0.348 |
| 13-14| `diverged_reset_enabled` + `diverged_reset_triple_and` | **PASS** | +0.002 (FS_static) | +0.000 | 9 | JTlOYE +0.176 |
| stacked | all 4 PASS together | **PASS** | +0.004 / +0.011 (FS) | +0.010 | 21 | TGZ5 +0.547 |

Hard-abort criteria across all PASS sprints + stacked: all pass.

---

## Per-flag recommendation

### Promote to balanced (enable by default)

#### 1. `shadow_r_reset_enabled` (B5) — **PROMOTE**

- Yang 2017 symmetric R-reset; pairs cleanly with already-promoted F2.3
- Mechanism: correctness-fix (closes asymmetry where shadow.R stayed
  stale after F2.3 reset only main.R)
- Cohort tail: +0.005 (neutral, no regression)
- Bucket means: positive in all FS/DT (within noise)
- Only DT regression case (TGZ5) is positive trade-off (Δne +1.667 dB)
- **Risk**: LOW. Symmetric to validated F2.3 pattern. No new mechanism.

#### 2. `f_e5_enabled` (F-E5) — **PROMOTE**

- Saturation handling: mic soft-clip + main mu sat-gate + error_psd
  post-sat reset + shadow_rise sat mask
- Bucket means flat, but largest single-case improvement of any Phase 1
  sprint (+0.348 dB on sKXucFp4FUCJKo5d0G54Og)
- 14 cases > 0.01 dB, predominantly FS bucket
- All 4 sub-fixes are correctness improvements (close known gaps E5-1
  through E5-4)
- Cohort tail: 0.000 (no saturation events on that case)
- **Risk**: LOW-MEDIUM. The 4 sub-fixes change input/learning behavior
  during saturation; effect contained because saturation events are rare.

#### 3. `diverged_reset_enabled` + `diverged_reset_triple_and` — **PROMOTE PAIR**

- Unblocks the dead-code `diverged_reset_enabled` path safely
- Triple-AND `shadow_advantage > 2.0` is the critical guard: it
  discriminates true divergence from movement-tracking (the F2.2
  EMA failure pattern)
- Cohort tail: 0.000 (F2.2 EMA gave -0.500 / -0.648 — confirmed
  recurrence-avoidance)
- 9 cases moved; top case JTlOYE shows dual improvement
  (ΔerleA +0.176 AND Δne +0.174)
- **Risk**: MEDIUM. Reset action `_reset_filter_derived_state` is
  heavyweight (clears DT / EPC / ERLE / mu state). Triple-AND keeps
  fire rate very low (1.1% of cases on bench). Promoted as a pair
  because triple-AND without master enable is no-op.

### Keep default OFF (ship the flag, don't enable)

#### 4. `shadow_mu_state_aware` (B6) — **HOLD, defer promotion**

- Bucket means slightly better than B5 alone (FS_movement +0.007 vs
  +0.004), 5 FS improvements / 1 FS regression
- BUT contradicts B5 on wlAXM0i FS_static (B5 +0.131 vs B6 -0.210);
  stacked outcome dominated by B6 (-0.207 dB)
- wlAXM0i baseline ERLE_active = 1.66 dB; B6 brings it to 1.45 dB —
  above audibility floor but a real regression
- Cohort tail: +0.010 (neutral)
- **Risk**: MEDIUM. The wlAXM0i contradiction needs investigation
  before promotion. Recommendation: ship flag, run wlAXM0i listen
  test in next sprint, promote only if listening shows no degradation.

#### 5. `f_e1_enabled` (F-E1) — **HOLD, available for downstream**

- 0 cases moved on 800-case bench
- Mechanism correct per design but corpus doesn't contain extreme
  ERL cases (real ERL < 0.001 or far_pwr sustained near 1e-4)
- Available behind flag for downstream products that observe extreme-
  coupling edge cases in their field samples
- **Risk**: N/A on bench. Flag is correctness-only.

#### 6. `f_delaytrack_enabled` (F-DT) — **HOLD, available for downstream**

- 1 case moved by 0.01 dB on 800-case bench
- Mechanism correct (variance-based delay reliability) but on this
  corpus, the old PAR-confidence gate and the new variance gate agree
- Available for deployments observing gradual delay drift (long
  teleconference sessions, thermal effects)
- **Risk**: N/A on bench.

### Discard

#### 7. `f_e3_enabled` (F-E3) — **DISCARD**

- 4 / 6 bucket-mean abort criteria FAILED (FS_static -0.110, others
  worse than -0.02 threshold)
- Root cause: EPC detector fires too frequently in normal speech
  (shadow_rise on intermittent silence boundaries) → F-E3
  misidentifies as "consecutive gain change" → extends hangover →
  ERLE drops
- Cohort tail guard worked (qNvSMyU 0.000) but didn't save FS_static
- Would need redesign of EPC consecutive-detection signature
  before any consecutive-event logic can be safely deployed
- **Action**: keep flag default OFF, do NOT promote, do NOT ship as
  "use at own risk". Document the FAIL pattern as a learning.

---

## Stacked promotion impact

The 3 recommended promotions (B5 + F-E5 + diverged_reset triple-AND)
together produce:

| Bucket | Stacked ΔerleA | Stacked Δne_pres |
|---|---|---|
| FS_static    | +0.003-0.004 | -0.001 to -0.007 |
| FS_movement  | +0.007-0.011 | -0.011 to -0.019 |
| DT_static    | +0.001-0.002 | -0.001 to +0.008 |
| DT_movement  | +0.000 | +0.001 |
| NE           | +0.000 | +0.000 |

(Range covers individual vs stacked since each sub-flag has narrow
effect; stacked is roughly additive on the affected case subset.)

Cohort tail invariant preserved: qNvSMyU stays within ±0.01 dB.

---

## Open items deferred to v3.12+

- **Phase 2** (C2 wiring filter_state → RES): infrastructure work for
  Phase 3. Skipped in v3.11 because Phase 3 is required to consume it.
- **Phase 3** (RES canonical refactor / Q7 V3 verdict):
  - Audit gain_floor fire-rates
  - Unify into canonical Beroutti/Ephraim-Malah/Hänsler-Schmidt 6-step
  - 4-cap ranked priority
  - Per-state ENR tuple (C2 policy consumption)
  - Byte-equal validation in steady state vs v3.10.6
- **F-HFR** (Hybrid-Frequency-Resolution AKF 2023 per-band Q/R) —
  Phase 4 long-arc
- **F-Harm** (WebRTC AEC3 harmonic pitch tracker) — Phase 4
- **E4 NLP** (non-linear processor) — Phase 4 dedicated arc

The Q7 V3 verdict (RES coherence broken) remains the main bottleneck.
Phase 1 confirmed that front-end fixes alone produce only narrow
per-case effects — bucket-mean improvements require addressing RES.

---

## Recommended next actions

1. **Apply promotions**: enable `shadow_r_reset_enabled`,
   `f_e5_enabled`, `diverged_reset_enabled`, `diverged_reset_triple_and`
   in the BALANCED preset (`AecConfig.from_preset`).
2. **Re-render 800-case** with promoted balanced (sanity check that
   the four flags compound as expected on production preset).
3. **Run AECMOS scoring** vs v3.10.6 baseline scores.json (verify
   full-pipeline production metric does not regress; linear-only
   eval shows no regression, but RES interaction needs verification).
4. **Bump version** v3.10.6 → v3.11.0 if AECMOS passes.
5. **Tag** v3.11.0 and update CHANGELOG.

Items 2-5 require user authorisation (versioning + tagging is a
release action). This report stops at the recommendation.
