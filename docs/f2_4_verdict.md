# F2.4 mu holdoff no-reset — 800-case Verdict

**Date**: 2026-05-12  
**Branch**: `feature/f2-2-diverged-reset`  
**Config**: `mu_holdoff_no_reset=True` — holdoff counter only armed on fresh DT onset (holdoff==0); not reset if already counting  
**Baseline**: production `balanced` (results/f2_2_baseline/scores.json)  
**Verdict**: **CONDITIONAL PASS — net positive, but DT worst-case regression requires monitoring**

---

## What was tested

`_update_simple_mu_ratio` previously reset `holdoff=20` on every DT frame. In marginal DT
(ratio oscillating around `_simple_mu_ratio`), the holdoff counter was continuously refreshed,
preventing mu from releasing indefinitely — a form of S7 stale-state bug. Fix: `holdoff=20`
is only written when `holdoff==0` (fresh onset), so ongoing DT frames that re-trigger the
attack path can no longer extend the holdoff window.

---

## Bench configuration

| param | value |
|---|---|
| preset | balanced |
| filter_length | 832 (52 ms @ 16k) |
| cng | True |
| workers | j=4 |
| cases | 800 |
| AEC_MU_HOLDOFF_NO_RESET | True |

Ablation: 800/800, 0 errors, 13.5 min

---

## Bucket means

| Bucket | n | baseline echo | new echo | Δecho | baseline deg | new deg | Δdeg |
|---|---:|---:|---:|---:|---:|---:|---:|
| FS_static   | 169 | 3.542 | 3.543 | **+0.001** | 4.999 | 4.999 | ±0.000 |
| FS_movement | 131 | 3.586 | 3.589 | **+0.003** | 4.999 | 4.999 | ±0.000 |
| DT_static   | 186 | 4.126 | 4.126 | **+0.000** | 2.367 | 2.368 | +0.001 |
| DT_movement | 114 | 4.100 | 4.099 | **−0.001** | 2.337 | 2.339 | +0.003 |
| NE          | 200 | 4.998 | 4.998 | **−0.000** | 4.011 | 4.011 | ±0.000 |

---

## Per-case distribution

| metric | value |
|---|---|
| Cases Δecho < −0.05 (regression) | **4** |
| Cases Δecho > +0.05 (improvement) | **7** |
| Cases \|Δecho\| > 0.10 | **4** |
| Sum of echo regressions | −0.484 |
| Sum of echo improvements | +0.569 |
| Regression:Improvement ratio | **0.9×** (net positive) |
| Worst single case Δecho | −0.204 (`jtYTdZm3lUmFVNibJWq8YQ_doubletalk_with_movement`) |
| 2nd worst Δecho | −0.113 (`XV5L2dn3S06M9GBEu1q3DA_doubletalk`) |
| Best single case Δecho | +0.114 (`S5cyhx02u00eJzrHTxVTwQ_farend_singletalk`) |
| Cohort tail (`qNvSMyUSXUyrDGp...`) | Δecho=−0.003, Δdeg=+0.000 |

---

## Root cause of DT regression cases

The fix is asymmetric: it benefits scenarios where marginal DT falsely extended holdoff (mu
stuck low during genuine FS), but it hurts scenarios where genuine DT oscillates around the
threshold. In genuine DT with a fluctuating ratio:

- **Baseline**: each ratio < threshold frame resets holdoff=20 → mu stays low → echo suppressed.
- **Fix**: holdoff only set once; once the 20-frame window expires, mu ramps up via α=0.95 →
  even if DT continues, mu partially releases → residual echo passes through.

The two worst regressions are both DT cases (`doubletalk_with_movement`, `doubletalk`),
confirming this mechanism. The ratio < threshold oscillation in these cases is genuine DT
activity, not marginal / false detection.

---

## Why the aggregate is still positive

The S7 bug (marginal DT indefinitely suppressing mu in FS-like segments) affects more cases
than the genuine DT oscillation scenario — hence 7 improvements outweigh 4 regressions in
count, and sum (+0.569) > |sum_reg| (0.484). The bucket means are flat because the worst
DT_movement case is offset by DT_movement improvements on other clips.

---

## Disposition

- F2.4 **CONDITIONAL PASS**.
- `mu_holdoff_no_reset=True` is net-positive but carries a known worst-case DT regression
  (−0.204) on one clip. Recommend:
  1. Promote to `balanced` with flag=True.
  2. Add a secondary guard: if `mu < 0.25` at holdoff expiry and echo-error is still elevated,
     re-arm holdoff once (single re-arm only) → prevents mu snap-back mid-genuine-DT without
     re-introducing the marginal-DT freeze.
  3. The re-arm guard is a Phase 2 refinement; the bare flag is already a net win and safe
     to ship as the worst individual case (−0.204) is a single DT_movement clip, not a
     systematic bucket regression.
- Worst case clip `jtYTdZm3lUmFVNibJWq8YQ_doubletalk_with_movement` added to regression
  monitoring list alongside `qNvSMyUSXUyrDGp...` cohort tail.
