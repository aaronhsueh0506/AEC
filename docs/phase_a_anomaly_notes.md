# Phase A Anomaly Notes

Per P52 v1.1 §8.3: anomalies are logged here; not actioned in Phase A.

## A.0 — ShadowCopyController retirement

**A1 (2026-05-11). Shadow R-EMA α = 0.95, not the v1.1 §2.4.2 spec value 0.9.**

Both `PBFDKF` main and the shadow `PBFDKF` instance inherit `_alpha_r = 0.95`
([aec.py:1048](../python/aec.py#L1048)). The shadow filter constructor at
[aec.py:4027-4039](../python/aec.py#L4027) only overrides `Q_high` / `Q_low` /
`Q` (×`shadow_q_ratio=3.0`); it does not touch the R-EMA constant. As a result
production runs **main α_R = shadow α_R = 0.95**, shrinking the gap that v1.1
§2.4.2 Difference-1 ("main slow / shadow fast on R") relies on.

Per §0.4 forbidden-actions list and §8.3 anomaly-notes rule, this is **not
modified in Phase A**. Phase A measurements (A.0 ERLE validation, A0 pre-flight
discriminator) run as-is on the production α values. If Phase A reaches T1 / T2,
the constants are still locked.

Post-Phase-C is the earliest legitimate time to revisit shadow α_R per §5.1.

**A2 (2026-05-11). PBFDKF main `_alpha_r = 0.95` already faster than v1.0 §2.4.2 spec α_main = 0.99.**

Logged in v1.1 design doc §2.4.2 itself. Repeated here so Phase A implementors
encounter it during A.0. Same disposition as A1: not modified.

**A3 (2026-05-11). No shadow-at-init H copy exists.**

[aec.py:4020-4039](../python/aec.py#L4020) constructs a fresh `FilterClass`
shadow with zero weights. v1.0 §I3 permits a one-shot stream-start copy from
main → shadow at init but does not require it. Current code doesn't do it; A.0
does not add it. After A.0, shadow continues to start from zero W and evolves
independently under its own Q-boosted Kalman recursion.
