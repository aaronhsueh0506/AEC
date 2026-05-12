# F2.2 v2 Diverged-Streak EMA threshold=0.85 — 800-case Verdict

**Date**: 2026-05-12  
**Branch**: `feature/f2-2-diverged-reset`  
**Config**: EMA α=0.95, threshold=0.85, `diverged_reset_enabled=True`  
**Verdict**: **FAIL — threshold adjustment ineffective; mechanism fundamentally unsuitable**

---

## Motivation

F2.2 v1 (threshold=0.7) failed with 17 regression / 8 improvement and regression sum
−3.50 vs improvement +1.22 (2.9× skew). Root cause: TC≈24 frames fires during normal
FS_movement shadow-leads-main periods. Option A from v1 recommendations: raise threshold
0.7 → 0.85 (TC≈37 frames).

---

## Bench configuration

| param | value |
|---|---|
| preset | balanced |
| filter_length | 832 (52 ms @ 16k) |
| cng | True |
| workers | j=4 |
| cases | 800 |
| AEC_DIVERGED_STREAK_EMA | 1 |
| AEC_DIVERGED_RESET | 1 |
| AEC_DIVERGED_STREAK_THRESHOLD | 0.85 |

Baseline: 800/800, 0 errors, 13.4 min  
Flag-ON: 800/800, 0 errors, 13.4 min

---

## Bucket means

| Bucket | n | baseline echo | v2 echo | Δecho | baseline deg | v2 deg | Δdeg |
|---|---:|---:|---:|---:|---:|---:|---:|
| FS_static   | 169 | 3.542 | 3.539 | −0.004 | 4.999 | 4.999 | ±0.000 |
| FS_movement | 131 | 3.586 | 3.581 | −0.005 | 4.999 | 4.999 | ±0.000 |
| DT_static   | 186 | 4.126 | 4.123 | −0.003 | 2.367 | 2.369 | +0.001 |
| DT_movement | 114 | 4.100 | 4.100 | ±0.000 | 2.337 | 2.333 | −0.003 |
| NE          | 200 | 4.998 | 4.998 | ±0.000 | 4.011 | 4.011 | ±0.000 |

Bucket means all < ±0.01 — deceptive, same pattern as v1.

---

## Why this is still FAIL: worse per-case distribution than v1

| metric | v1 (threshold=0.70) | v2 (threshold=0.85) |
|---|---|---|
| Cases Δecho < −0.05 (regression) | 17 | **11** |
| Cases Δecho > +0.05 (improvement) | 8 | **5** |
| Cases \|Δecho\| > 0.10 | 19 | **8** |
| Sum of echo regressions | −3.50 | **−2.405** |
| Sum of echo improvements | +1.22 | **+0.612** |
| Regression:Improvement ratio | 2.9× | **3.9×** |
| Worst single case Δecho | −0.504 | **−0.648** |
| Cohort tail (`qNvSMyUSXUyrDGp...`) | Δecho=0.000 | Δecho=0.000 |

Raising the threshold reduced the number of false positive fires (11 vs 17 regression
cases) but made each false positive more damaging. The worst case regression worsened
from −0.504 → −0.648 on the **same case**
(`tl5UFRCXZkyL6EoWVl09xA_farend_singletalk_with_movement`).

---

## Why higher threshold makes individual regressions worse

With threshold=0.85, the EMA must sustain above 0.85 for ~37 frames before firing (vs
24 frames at 0.7). For `tl5UFRCXZkyL6EoWVl09xA`, shadow leads main during a movement
phase (legitimate adaptation, not divergence). At 0.7, the false reset fired after 24
frames — discarding 24 frames of good main state. At 0.85, the reset fires at 37 frames
— discarding 37 frames of good main state accumulated during the movement. The later
the false reset fires, the more converged state is thrown away.

This is a fundamental structural problem with threshold tuning: the same threshold that
reduces false-positive frequency makes each false positive more destructive.

---

## Root cause: movement-adaptive state misclassified as divergence

The `_p3f_diverged_streak` / EMA accumulates whenever `main_err_ratio > 1.3` (shadow
30% better than main). In FS_movement cases, the echo path changes continuously; shadow
at Q×3.5 adapts faster than main during movement phases. Sustained shadow-lead periods
of 20–40 frames are normal tracking behavior, not divergence. No threshold in the EMA
approach can distinguish "shadow tracking faster" from "main diverged" without additional
gating on path-change detection.

---

## Disposition

- F2.2 v2 CLOSED FAIL.
- The EMA mechanism + threshold approach cannot be recalibrated to work. The structural
  issue (EMA can't distinguish movement tracking from divergence) requires a different
  detection axis — e.g., Option C from v1 (gate EMA accumulation only when `_epc_active`
  AND `_far_power_high`), not threshold tuning.
- Branch `feature/f2-2-diverged-reset` kept open if Option C is pursued.
- `diverged_reset_enabled` remains default OFF in `balanced`.
- The hard counter `_p3f_diverged_streak` (threshold=50) remains dead code, as before.
