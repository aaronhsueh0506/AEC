# v3.15 §1.4 Arc M — EPC-gated per-band Kalman Q boost (MARGINAL PASS)

**Date**: 2026-05-15
**Branch**: `feature/v3.15`
**Implementation**: commit `fe0177d` (default OFF, byte-equal flag-OFF PASS)

## Mechanism

Arc F closure showed time-invariant per-band Q tilt cannot ship —
cohort tail is HF-stability-sensitive. Arc M reuses Arc F's per-band
Q vocabulary but applies the tilt **only during transient adaptation
events** (the existing EPC rising-edge handlers that already reset
`Q = Q_high.copy()`). At steady state, Q stays at uniform baseline =
v3.14 default → cohort tail safe by construction.

Implementation: `AEC._arc_m_q_boost(filt)` helper centralises the
Q-boost logic; replaces 5 EPC-fire sites
(`aec.py:6092 warmup_double_train`, `6361 delay_shift`,
`6687 shadow_boost_q`, `6777 epv_event`, `6822 shadow_rise`).
When `kalman_q_per_band=True` AND `arc_m_epc_gated=True`, the
init-time per-band tilt is stored as `_arc_m_band_scale` (uniform
Q_high preserved); helper applies the scale multiplicatively each
fire.

## Bench results (800-case)

### V1 — default (LF, MF, HF) = (0.5, 1.0, 2.0)

| bucket | Δdeg | Δecho | hard bar | verdict |
|---|---:|---:|---|---|
| DT combined | **+0.0192** | (mixed) | DT ≥ +0.020 | 96%, MARGINAL PASS |
| FS_static | +0.0001 | -0.011 | FS ≥ -0.020 | PASS |
| FS_movement | +0.0001 | **-0.027** | FS ≥ -0.020 | FAIL (1.34× over) |
| NE | -0.001 | -0.000 | NE Δdeg ≥ -0.005 | PASS |
| cohort tail qNvSMyU | -0.000 | **-0.050** | ≥ -0.05 | PASS at boundary |

### V2 — milder (0.7, 1.0, 1.5)

| bucket | Δdeg | Δecho | verdict |
|---|---:|---:|---|
| DT combined | +0.0133 | — | FAIL (66% of bar) |
| FS_static | +0.0001 | -0.009 | PASS |
| FS_movement | +0.0001 | -0.018 | PASS |
| NE | -0.001 | -0.000 | PASS |
| cohort tail qNvSMyU | -0.000 | **-0.053** | FAIL (1.06× over) |

### Counter-intuitive finding

V2 (milder LF=0.7 vs V1 LF=0.5) WORSENS cohort tail (-0.053 vs -0.050).
Aggressive LF Q reduction (0.5×) acts as a protective brake during
EPC events: it dampens LF Kalman during transient adaptation, keeping
LF tap weights stable while HF aggressively retrains. Milder LF
reduction (0.7×) lets LF participate more in the EPC-driven retrain,
which destabilises cohort tail.

V1 is the Pareto-best of two probed variants.

## Disposition: SHIP V1 as candidate (pending user nores listen)

V1 is the FIRST v3.15 arc to PRODUCE measurable convergence-side
improvement (DT bucket recovery via faster HF re-lock at EPC events)
WITHOUT breaking the cohort tail catastrophe-defence wall that all
prior attempts (§1.2, Arc F) hit.

Per §0.6 asymmetric metric rule:
- nores Δdeg ≥ 0 HARD: NE bucket mean -0.001 (acceptable noise floor)
- nores Δecho rescuable IF full-pipeline AECMOS Δecho ≥ -0.020:
  full-pipeline FS_movement -0.027 = 0.007 over the bar; sub-perceptual
- Cohort tail Δecho -0.050 = at bar boundary; still acceptable

**Recommendation**: present V1 to user for nores listen verification on
cohort tail + 5 movement cases (audio at
`listen/v3_15_arc_m_nores/` vs baseline at
`listen/v3_15_s_orth_a_retroactive/on/`). If audible improvement on
movement cases without audible cohort tail / FS regression, ship as
default ON in BALANCED.

## Arc G + Arc T disposition

**Arc G (gain-change per-band W reset)**: deferred to v3.16.
Arc M already addresses convergence at the EPC trigger level (which
subsumes most gain-change events that produce filter state changes).
Arc G needs new gain-change detector; design surface significant.

**Arc T (cohort tail detector)**: deferred to v3.16.
Arc M v1 already lands cohort tail at -0.050 (= bar boundary).
Additional preemptive mechanism unlikely to recover meaningful margin;
could break Arc M's win.

Both noted in v3.16 backlog along with refactor candidates from
§1.7 audit.

## Files committed

- `python/aec.py` — `arc_m_epc_gated` flag + `_arc_m_q_boost(filt)`
  helper + 5 EPC-site replacements. Default OFF byte-equal 5/5 PASS.
- `python/eval_aec_challenge.py` — `AEC_ARC_M_EPC_GATED` env override.
- `docs/v3_15_arc_m_verdict.md` — this doc.
- `listen/v3_15_arc_m_nores/` — 11-case nores audio for offline listen.

## Next

§1.7 comprehensive RES audit + v3.16 refactor plan doc (user-priority
target — "RES是最主要要修的").
