# v3.15 §1.6 Arc F — per-band Kalman Q schedule CLOSED (substrate)

**Date**: 2026-05-15
**Branch**: `feature/v3.15`
**Substrate retained**: commit `c9bbe87` (default OFF, byte-equal flag-OFF PASS)

## Two variants bench'd against `results/v3_14_baseline/scores.json`

### V1 — (LF, MF, HF) = (0.5, 1.0, 2.0): aggressive HF boost

| bucket | Δdeg | Δecho | hard bar |
|---|---:|---:|---|
| DT combined | **+0.029** ✓ | (mixed) | DT ≥ +0.020 ✓ PASS |
| FS_static | +0.0001 | -0.018 | FS ≥ -0.020 ✓ PASS |
| FS_movement | +0.0001 | **-0.030** ✗ | FS ≥ -0.020 FAIL (1.5× over) |
| NE | -0.001 | -0.000 | NE Δdeg ≥ -0.005 ✓ PASS (mean) |
| cohort tail qNvSMyU | -0.000 | **-0.067** ✗ | cohort tail Δecho ≥ -0.05 FAIL (1.34× over) |

Two NE cases damaged: `wJVPo4lex...` Δdeg -0.143; `E0l0WVPQ...` Δdeg -0.150.

### V2 — (LF, MF, HF) = (1.0, 1.0, 1.5): conservative HF boost

| bucket | Δdeg | Δecho | hard bar |
|---|---:|---:|---|
| DT combined | **+0.008** ✗ | (mixed) | DT ≥ +0.020 FAIL (38%) |
| FS_static | +0.0001 | -0.006 | FS ≥ -0.020 ✓ PASS |
| FS_movement | +0.0001 | -0.010 | FS ≥ -0.020 ✓ PASS |
| NE | -0.001 | -0.000 | NE Δdeg ≥ -0.005 ✓ PASS |
| cohort tail qNvSMyU | -0.000 | **-0.063** ✗ | cohort tail Δecho ≥ -0.05 FAIL (1.26× over) |

V2 trades DT win for FS recovery, but cohort tail STILL fails by 0.013 dB.

## Mechanism analysis: steady-state stability wall

PBFDKF Q controls Kalman process noise — higher Q = filter more responsive
to perturbations. Both variants raise Q in HF (and V1 also lowers it in LF).
Effects:

- **Convergence speed (target gain)**: higher HF Q → faster HF re-lock when
  echo path changes. This shows up as DT bucket Δdeg gain (V1 +0.029 dB)
  because faster convergence means less time stuck in `coarse_learning`,
  which the §1.1 audit identified as the DT-NE compression source.
- **Steady-state misadjustment (cost)**: higher HF Q means filter "wobbles"
  more around optimal weights at steady state, leaking ECHO at the bins
  where Q is raised. This shows up as FS bucket Δecho regression and
  catastrophic cohort tail Δecho regression.

The cohort tail (`qNvSMyU`) is by-design HF-sensitive — it's the catastrophe
class where shadow + PathChangeRegimeHandler is load-bearing. Even modest
uniform-time HF Q boost (V2 ×1.5) violates the -0.05 cohort tail bar.

The trade-off is FUNDAMENTAL to time-invariant Q tuning: any uniform-time
HF boost costs steady-state stability proportionally to the boost.

## Disposition

**Arc F as standalone CLOSED CANNOT SHIP.** Both variants violate one or
more hard bars:
- V1 wins DT but breaks FS_movement + cohort tail (catastrophe defence)
- V2 narrowly passes FS but loses DT below hard bar AND still breaks cohort tail

**Substrate retained**: `kalman_q_per_band` flag + `kalman_q_band_scales`
tuple stay as default OFF research substrate. Reusable by Arc M.

## Why Arc F is the right SUBSTRATE for Arc M (not standalone)

Arc M = movement-aware adaptive Q. The CANONICAL mechanism applies high Q
**only when the filter needs to adapt** (state change / movement detector
fires) and falls back to baseline Q at steady state. This avoids the
steady-state stability cost while keeping the convergence speed benefit.

Arc F provides the per-band Q vocabulary (LF/MF/HF distinct values).
Arc M provides the time-variation gating (boost only during transients).
Combined:

```
Q_effective(bin, t) = baseline_Q(bin) +
                      movement_active(t) * boost_Q(bin)
```

where `baseline_Q` = uniform v3.14 default (cohort tail safe), `boost_Q`
applied per-band only during transient windows. This is the correct
canonical implementation of "faster HF re-lock without steady-state
stability cost".

## Plan revision

Roll Arc F substrate forward to Arc M (§1.4). Arc M S1 will design the
movement detector + transient gating; Arc M S2 will wire `boost_Q` per-band
on top of Arc F substrate (with `kalman_q_per_band` activated).

Arc F closure is NOT a setback — it produces the per-band Q infrastructure
that Arc M needs without committing to standalone shipping. This matches
the original plan §1.6 framing ("prerequisite for Arc M's per-band Q boost").

## v3.13 E2 Path 3 DT debt status

Arc F V1 closed +0.029 dB of DT debt (58% of -0.050) BUT broke FS bars.
The mechanism is real; just not deployable as standalone. Awaiting Arc M
gating to make it shippable.

## Files committed

- `python/aec.py` — Arc F config flags + per-band Q tilt at AEC.__init__.
  Default OFF byte-equal sanity 5/5 PASS.
- `python/eval_aec_challenge.py` — env overrides
  `AEC_KALMAN_Q_PER_BAND`, `AEC_KALMAN_Q_BAND_SCALES`.
- `docs/v3_15_arc_f_closure.md` — this doc.

## Next

Arc M (§1.4): movement-aware adaptive Q boost. Will reuse Arc F's per-band
Q infrastructure as substrate. Design: variance-EMA detector → transient
window → boost HF Q for N frames → release. Target: +0.030+ dB DT recovery
without cohort tail regression.
