# F2.2 Diverged-Streak EMA + P3h Reset — 800-case Verdict

**Date**: 2026-05-12  
**Branch**: `feature/f2-2-diverged-reset`  
**Commit**: bae2622  
**Verdict**: **FAIL — calibration too aggressive**

---

## What was changed

- `_p3f_diverged_streak` hard counter replaced with EMA tracker (α=0.95,
  TC≈20 frames, threshold=0.7) so single low-error frames do not zero evidence.
- `diverged_reset_enabled=True` passed via `AEC_DIVERGED_STREAK_EMA=1
  AEC_DIVERGED_RESET=1` env vars.
- Mechanism direction is correct; the hard counter was genuine dead code
  (oscillation near the 1.3 ratio boundary resets it before reaching 50).

---

## Bench configuration

| param | value |
|---|---|
| preset | balanced |
| filter_length | 832 (52 ms @ 16k) |
| cng | True |
| workers | j=4 |
| cases | 800 |

Baseline (flag OFF): 800/800, 0 errors, 9.1 min  
Flag-ON: 800/800, 0 errors, 8.8 min

---

## Bucket means

| Bucket | n | baseline echo | ON echo | Δecho | baseline deg | ON deg | Δdeg |
|---|---:|---:|---:|---:|---:|---:|---:|
| FS_static   | 169 | 3.542 | 3.537 | −0.006 | 4.999 | 4.999 | ±0.000 |
| FS_movement | 131 | 3.586 | 3.582 | −0.003 | 4.999 | 4.999 | ±0.000 |
| DT_static   | 186 | 4.126 | 4.123 | −0.004 | 2.367 | 2.362 | −0.005 |
| DT_movement | 114 | 4.100 | 4.098 | −0.002 | 2.337 | 2.336 | −0.001 |
| NE          | 200 | 4.998 | 4.998 | ±0.000 | 4.011 | 4.011 | ±0.000 |

Bucket means all < ±0.01 → appear neutral.

---

## Why this is FAIL: per-case distribution reveals calibration problem

Bucket means are deceptive here. The reset fires frequently enough to cause
large swings on individual cases, which cancel out in the mean:

| metric | value |
|---|---|
| Cases with \|Δecho\| > 0.05 | 17 regression / 8 improvement |
| Cases with \|Δ\| > 0.10 | 19 / 800 |
| Sum of echo improvements | +1.22 |
| Sum of echo regressions  | −3.50 |
| Worst single case Δecho  | −0.504 (`tl5UFRCXZkyL6EoWVl09xA_farend_singletalk_with_movement`) |
| DT worst bidirectional   | Δecho−0.316 / Δdeg−0.447 (`z4PqfBhq2E01IDBkTH0gnw_doubletalk`) |
| Cohort tail case (`qNvSMyUSXUyrDGp...`) | Δecho=0.000 — unaffected |

The regression total (−3.50) outweighs improvement total (+1.22) by 2.9×.
This means the reset fires more often when it shouldn't (false positive
divergence) than when it should.

---

## Root cause of calibration failure

The EMA with α=0.95, threshold=0.7 fires after ~24 consecutive frames where
`main_err_ratio > 1.3` (shadow 30% better than main).

In `farend_singletalk_with_movement` cases, shadow (running at Q×3.5) adapts
faster than main during path changes. This creates sustained periods of
shadow > main without the filter being pathologically diverged — just slower.
The EMA accumulates and triggers a full filter reset, discarding good main
filter state and forcing reconvergence.

The cohort tail case (`qNvSMyUSXUyrDGp`) is unaffected because it already
scores echo=4.049 in baseline — not the kind of diverged case the reset targets.
The cases that trigger the reset are movement cases where shadow naturally leads.

---

## Recommendations for next attempt (F2.2 v2)

Three options, in order of preference:

**Option A — raise EMA threshold** (lowest risk):  
Change `diverged_streak_ema_threshold` from 0.7 → 0.85.  
At α=0.95: reaches 0.85 after ~37 consecutive hit frames (vs 24 at 0.7).  
Significantly tightens the false-positive rate.

**Option B — tighten diverged_th**:  
Raise `_diverged_th` from 1.3 → 1.5 or 2.0 (shadow must be 50–100% better
than main before accumulating EMA). Shadow adapting faster during movement
typically produces ratios in 1.1–1.4 range; genuine divergence produces ratios
> 2.

**Option C — add movement gate**:  
Only update EMA when `_epc_active` is True (path change detected) OR
`_far_power_high` and `_path_change_handler` not in active-movement phase.
Avoids accumulating evidence during normal movement adaptation.

---

## Disposition

- Flag stays **default OFF** in `balanced`.  
- Branch `feature/f2-2-diverged-reset` kept open for v2 recalibration.  
- The structural EMA mechanism (unit tests: 12/12) is correct. Only threshold
  needs adjustment.  
- Do not merge to main in current form.
