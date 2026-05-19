# diverged_reset triple-AND verdict

**Phase**: v3.11 Phase 1 Sprint 13-14
**Flag**: `diverged_reset_triple_and` + `diverged_reset_enabled` (default OFF)
**Result**: **PASS — marginal positive, F2.2 EMA failure pattern avoided**

## Mechanism

The existing `diverged_reset_enabled` has 4 gates (streak / cooldown /
filter_state=='diverged' / once_converged). F2.2 tried EMA-based streak
detection but **FAILED 800-case** (17 regressions / 8 improvements):
EMA confused shadow-tracking-during-movement with true divergence.

Sprint 13-14 adds the triple-AND condition: `shadow_advantage > 2.0`.
This requires shadow to also be ahead — shadow tracking during room
movement keeps `shadow_advantage` near 1.0, so the triple-AND blocks
the reset on those false-positive cases.

Effective gate (all must hold):
1. `diverged_reset_enabled` (master)
2. `_filter_once_converged` (don't reset before first convergence)
3. cooldown == 0 (avoid rapid re-fires)
4. `filter_state == 'diverged'` (state classifier agrees)
5. streak >= 50 frames (counter or EMA-based per `use_diverged_streak_ema`)
6. **NEW**: `shadow_advantage > 2.0` (only when triple-AND flag enabled)

## Verification

**Baseline**: v3.10.6 (`results/v3_10_5_main/`)
**Candidate**: v3.10.6 + `AEC_DIVERGED_RESET=1 AEC_DIVERGED_RESET_TRIPLE=1`
(`results/sprint_diverged_triple_on/`)

### Bucket-mean Δ vs baseline

| Bucket | n | Δerle_full | Δerle_active | Δne_pres |
|---|---|---|---|---|
| FS_static    | 169 | +0.002 | +0.002 | +0.002 |
| FS_movement  | 131 | +0.000 | +0.000 | -0.000 |
| DT_static    | 186 | -0.000 | -0.000 | -0.000 |
| DT_movement  | 114 | +0.000 | +0.000 | -0.000 |
| NE           | 200 | +0.000 | +0.000 | +0.000 |

All bucket means within ±0.002 dB noise floor.

### Per-case impact (|Δ| > 0.01 dB)

9 / 800 cases moved. Notable cases:

| Stem | Bucket | ΔerleA | Δne |
|---|---|---|---|
| JTlOYE0ckEqENX83rxjl0Q FS_static | FS_static | **+0.176** | **+0.174** |
| P10GsQvhskKx3fB06Zv4Yg FS_static | FS_static | +0.167 | -0.036 |
| QEeKiaNiDECfqXTRrDFWWw FS_static | FS_static | +0.075 | -0.049 |
| SvVpGsVXY0WeklKK5tnI3Q FS_static | FS_static | -0.107 | +0.016 |
| oEyXuSCCw0qdJ0J16FGfcQ FS_static | FS_static | +0.024 | +0.094 |

Top case `JTlOYE` shows **dual improvement** (ERLE +0.176 AND NE_pres
+0.174) — clear win for cases where divergence detection fires
correctly and reset recovers the filter.

### Cohort tail (critical F2.2 anti-recurrence)

`qNvSMyUSXUyrDGpOw7s6qg_farend_singletalk`: ΔerleA = **+0.000**, Δne = -0.000

The triple-AND `shadow_advantage > 2.0` condition successfully
**blocked** the reset on the cohort tail case — which is exactly the
F2.2 failure pattern (movement tracker triggering on cohort tail
where shadow tracks main). qNvSMyU has shadow tracking main during
the cohort-tail event, so shadow_advantage stays near 1.0, fails the
new gate, no reset fires.

### Comparison to F2.2 (prior FAIL)

| Sprint | Approach | Cohort tail | Per-case |
|---|---|---|---|
| F2.2 v1 (EMA, threshold 0.7) | EMA-based streak | -0.500 (FAIL) | 17 reg / 8 imp |
| F2.2 v2 (EMA, threshold 0.85) | EMA-based streak | -0.648 (FAIL) | 11 reg / 5 imp |
| Sprint 13-14 (triple-AND) | + shadow_adv > 2.0 | **0.000** | 6 imp / 3 reg* |

*Note: bucket means here use a different evaluation (linear_eval vs
AECMOS). F2.2 was AECMOS-based and saw worse cohort-tail behavior.

## Hard abort criteria

| Criterion | Threshold | Actual | Pass? |
|---|---|---|---|
| Cohort tail qNvSMyU | ≥ -0.05 | +0.000 | ✓ |
| FS_static bucket mean | ≥ -0.02 | +0.002 | ✓ |
| FS_movement bucket mean | ≥ -0.02 | +0.000 | ✓ |
| DT_static bucket mean | ≥ -0.02 | +0.000 | ✓ |
| DT_movement bucket mean | ≥ -0.02 | +0.000 | ✓ |
| Linear ERLE Δ ≥ -0.5 | OK | min: -0.107 | ✓ |
| NE bucket | unchanged | +0.000 | ✓ |

All pass.

## Decision

- **Mechanism**: triple-AND condition implemented correctly; the
  `shadow_advantage > 2.0` gate successfully discriminates true
  divergence (shadow ahead) from movement-tracking (shadow ≈ main).
- **F2.2 anti-recurrence**: confirmed — cohort tail is preserved,
  no regression in the case class that broke F2.2.
- **Impact scope**: 9 / 800 cases (1.1%) — narrow but real, with one
  notable dual-improvement case (+0.176 ERLE + +0.174 NE pres).
- **Net effect**: 6 cases improve substantially, 3 cases regress
  mildly (worst -0.107 on SvVpGsVXY); net positive on the affected
  subset.

**Flag state**: keep both flags default OFF for now. Final promotion
decision deferred to "Final review" at end of Phase 1.

**Why this matters**: this is the first sprint where the
`diverged_reset_enabled` dead code becomes safely usable. The pair
`diverged_reset_enabled + diverged_reset_triple_and` may be worth
promoting together as a stacked enable in balanced preset, since the
triple-AND is the protective gate that makes the reset safe.
