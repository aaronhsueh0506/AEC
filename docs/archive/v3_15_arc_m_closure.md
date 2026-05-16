# v3.15 §1.4 Arc M — EPC-gated movement-aware Q boost CLOSED (substrate)

**Date**: 2026-05-15
**Branch**: `feature/v3.15`
**Substrate retained**: `arc_m_epc_gated` flag (default OFF) on top of Arc F
substrate (`kalman_q_per_band`). Reusable by Arc G or future arcs.

## Two variants bench'd against `results/v3_14_baseline/scores.json`

### V1 — band scales (LF, MF, HF) = (0.5, 1.0, 2.0): aggressive HF boost

| bucket | Δecho | Δdeg | hard bar |
|---|---:|---:|---|
| NE | -0.0000 | -0.0009 | NE Δdeg ≥ -0.005 ✓ PASS |
| DT_static | -0.0086 | **+0.0150** | DT Δdeg ≥ -0.005 ✓ PASS (recovers) |
| DT_movement | -0.0131 | **+0.0233** | DT Δdeg ≥ -0.005 ✓ PASS (recovers) |
| FS_static | -0.0107 | +0.0000 | FS Δecho ≥ -0.020 ✓ PASS |
| FS_movement | **-0.0269** | +0.0000 | FS Δecho ≥ -0.020 ✗ FAIL (1.34× over) |
| cohort tail qNvSMyU | -0.0496 | -0.0001 | cohort tail Δecho ≥ -0.05 ✓ PASS (margin 0.0004) |

### V2 — band scales (0.7, 1.0, 1.5): conservative HF boost

| bucket | Δecho | Δdeg | hard bar |
|---|---:|---:|---|
| NE | -0.0000 | -0.0009 | NE Δdeg ≥ -0.005 ✓ PASS |
| DT_static | -0.0059 | **+0.0118** | DT Δdeg ≥ -0.005 ✓ PASS (recovers) |
| DT_movement | -0.0123 | **+0.0148** | DT Δdeg ≥ -0.005 ✓ PASS (recovers) |
| FS_static | -0.0088 | +0.0000 | FS Δecho ≥ -0.020 ✓ PASS |
| FS_movement | **-0.0182** | +0.0000 | FS Δecho ≥ -0.020 ✓ PASS (margin 0.0018) |
| cohort tail qNvSMyU | **-0.0531** | -0.0000 | cohort tail Δecho ≥ -0.05 ✗ FAIL (1.06× over) |

## Mechanism analysis: EPC gating doesn't escape Arc F trade-off

Arc M's EPC gating was intended to apply per-band HF Q boost ONLY during
transient EPC-active windows, leaving steady-state Q untouched. The
hypothesis: cohort tail catastrophe defence needs steady-state stability,
EPC windows are short → steady-state untouched → cohort tail safe.

**The hypothesis fails because cohort tail IS exactly when EPC fires
heaviest.** PathChangeRegimeHandler's 6-gate AND fires on `qNvSMyU` class
during the catastrophe window — that is its load-bearing defence role.
Boosting Q during EPC-active windows = boosting Q during the cohort tail
catastrophe = same steady-state stability cost Arc F had.

## V1 vs V2 — symmetric trade-off

The two variants split the trade-off, neither escapes it:

| Variant | DT win | FS_movement | cohort tail |
|---|---|---|---|
| V1 (0.5/1.0/2.0) | strong (+0.023 deg) | FAIL -0.027 | PASS -0.0496 |
| V2 (0.7/1.0/1.5) | medium (+0.015 deg) | PASS -0.018 | FAIL -0.0531 |

This is the SAME wall as Arc F (uniform-time HF boost). The boost magnitude
controls where the wall hits, not whether it hits.

## Disposition

**Arc M as standalone CLOSED CANNOT SHIP.** Both variants violate one or
more hard bars; the violation surface is a fundamental trade-off, not a
tuning gap.

**Substrate retained**: `arc_m_epc_gated` flag (default OFF) + `_arc_m_q_boost`
helper + per-band scale storage at filter init stay as research substrate on
top of `kalman_q_per_band`.

## Why Arc M closure is informative for Arc G

Arc G (gain-change per-band W reset) targets a DIFFERENT mechanism: discrete
W-reset per affected band when ERL drift is detected. This does NOT modify
Q (steady-state stability untouched) and does NOT fire during normal EPC
windows (gain-change detector is orthogonal to EPC). So the Arc F/M wall
does not apply to Arc G.

Arc T (cohort tail detector) is also orthogonal — it targets RES preemption,
not Kalman Q. The Arc F/M wall is specific to Q-modification arcs.

## v3.13 E2 Path 3 DT debt status

Arc M V1 closed +0.020 dB of DT debt (40% of -0.050) BUT broke FS_movement.
Arc M V2 closed +0.013 dB of DT debt (26% of -0.050) BUT broke cohort tail.
The DT-NE compression mechanism is real and reachable by per-band Q, but
not deployable without breaking other invariants. DT debt closure target
moves to Arc G + Arc T composite or post-v3.15 RES refactor.

## Files committed

- `python/aec.py` — Arc M `_arc_m_q_boost` helper + EPC-gated per-band scale
  storage; `arc_m_epc_gated` config flag. Default OFF byte-equal sanity.
- `python/eval_aec_challenge.py` — env override `AEC_ARC_M_EPC_GATED`.
- `docs/v3_15_arc_m_closure.md` — this doc.

## Next

Arc G (§1.4) — gain-change per-band W reset. Mechanism orthogonal to
Arc F/M Q-modification trade-off. Consumes Arc P per-band ERL substrate.
