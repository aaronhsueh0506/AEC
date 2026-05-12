# B6 verdict — state-aware shadow µ schedule

**Phase**: v3.11 Phase 1 Sprint 3-4
**Flag**: `shadow_mu_state_aware` (default OFF)
**Result**: **PASS — marginal positive, recommend stacked validation with B5 before final review**

## Mechanism

Current shadow_mu_scale is binary (1.0 or 0.1) based on far_excited +
saturation_safe only; it ignores `main_paused` (regime handler holding W)
and the filter_state classification. When main is paused, shadow keeps
adapting at full speed → next `reverse_copy` pulls a shadow that drifted
while main was held. When filter_state is 'diverged', shadow trains on a
broken main reference, poisoning the next copy.

B6 adds 4-band schedule with explicit precedence (pause > safety >
caution > default):
- main_paused OR filter_state == 'diverged' → 0.0 (don't drift)
- NOT (far_excited AND saturation_safe) → 0.1 (safety, existing)
- filter_state == 'suspicious_dt' → 0.5 (DT caution)
- else → 1.0 (normal)

Uses previous frame's filter_state via `self._prev_filter_state` cache
(classifier runs at end of process()). Initialised to 'idle' in __init__
and reset() to preserve cold-start behaviour.

## Verification

**Baseline**: v3.10.6 800-case rendering (`results/v3_10_5_main/`).
**Candidate**: v3.10.6 + `AEC_SHADOW_MU_STATE=1` (`results/sprint_b6_on/`).

### Bucket-mean Δ vs baseline

| Bucket | n | Δerle_full | Δerle_active | Δne_pres |
|---|---|---|---|---|
| FS_static    | 169 | +0.002 | +0.002 | -0.007 |
| FS_movement  | 131 | +0.007 | +0.007 | -0.011 |
| DT_static    | 186 | +0.001 | +0.001 | -0.001 |
| DT_movement  | 114 | -0.000 | -0.000 | +0.002 |
| NE           | 200 | +0.000 | +0.000 | +0.000 |

All bucket means within ±0.011 dB noise floor.

### Per-case impact distribution (|Δ| > 0.1 dB)

| Bucket | ΔerleA: + / − | Δne: + / − |
|---|---|---|
| FS_static | 2 / 1 | 2 / 5 |
| FS_movement | 3 / 0 | 0 / 1 |
| DT_static | 0 / 0 | 0 / 3 |
| DT_movement | 0 / 0 | 1 / 0 |
| NE | 0 / 0 | 0 / 0 |

5 FS improvements with single regression. The single FS regression case
`wlAXM0iDgkm06i7UdRww1w_farend_singletalk` saw ΔerleA = −0.210 dB (baseline
1.66 → 1.45 dB, still above audibility floor). This case improved under
B5 alone (+0.131 dB), so B5 and B6 have partially-overlapping effects on
the regime handler — they will need stacked validation to confirm net
sign.

### Cohort tail

`qNvSMyUSXUyrDGpOw7s6qg_farend_singletalk`: ΔerleA = +0.010 dB (neutral,
no regression).

### Top affected cases

| Stem | Bucket | ΔerleA | Δne |
|---|---|---|---|
| TGZ5Wq0SCUCOXPsfee3uMQ FS_movement | FS_movement | **+0.520** | −1.435 |
| KSN5Jrzo7kaixP0z8xfr4Q FS_static | FS_static | +0.323 | +0.126 |
| wlAXM0iDgkm06i7UdRww1w FS_static | FS_static | **−0.210** | −0.455 |
| MkSLte0FTkqybGcLTwA3Tw FS_movement | FS_movement | +0.188 | −0.048 |
| sLWe8bfYbkGwX1W3PzI1PQ FS_static | FS_static | +0.162 | −0.410 |
| w0QrMwsZ5kGoJjRWvP0iKg FS_movement | FS_movement | +0.139 | −0.010 |

In FS context Δne < 0 means more aggressive linear cancellation
(less mic-signal preservation), which is desirable. Δerle gains are
larger than the single wlAXM0i regression magnitude.

### TGZ5 cross-sprint comparison

| Stem | B5 ΔerleA | B5 Δne | B6 ΔerleA | B6 Δne |
|---|---|---|---|---|
| TGZ5...doubletalk | -0.694 | +1.667 | +0.051 | -0.030 |
| TGZ5...farend_movement | +0.517 | -1.417 | +0.520 | -1.435 |

B5 DT regression (with NE-pres compensation) is NOT present in B6 — B6
keeps DT behaviour closer to baseline. FS_movement gain is nearly
identical between B5 and B6.

## Hard abort criteria

| Criterion | Threshold | Actual | Pass? |
|---|---|---|---|
| Cohort tail Δecho | ≥ −0.05 | +0.010 | ✓ |
| FS_static bucket mean Δ | ≥ −0.02 | +0.002 | ✓ |
| FS_movement bucket mean Δ | ≥ −0.02 | +0.007 | ✓ |
| Linear ERLE bucket mean Δ ≥ −0.5 | min: -0.000 (DT_movement) | ✓ |
| Per-case worst regression | within tolerance | wlAXM0i: -0.210 (baseline 1.66 → 1.45 still above floor) | ✓ |

All pass.

## Decision

- **B6 mechanism**: works as designed; precedence ordering correct;
  state-machine wiring stable across __init__ / reset() / end-of-frame.
- **Impact scope**: 5+1 FS bucket cases affected, similar narrow scope
  to B5 (cohort tail catastrophe-defence territory only).
- **Cross-sprint interaction**: TGZ5 family confirms B5 and B6 cover
  overlapping but distinct regime-handler firing patterns.
- **Cohort tail invariant**: preserved.

**Flag state**: keep `shadow_mu_state_aware = False` (default OFF) for
now. Final promotion decision deferred to "Final review" at end of
Phase 1.

**Outstanding question to resolve at final review**: do B5 + B6
*compound* on the FS-improving cases (TGZ5 FS_movement gets +0.520
either way), or *cancel* on the wlAXM0i FS_static case (B5 +0.131,
B6 −0.210)? A stacked B5+B6 800-case render at end of Phase 1 will
adjudicate; for now we keep both flags isolated for clean ablation.
