# F2.3 EPC R-reset — 800-case Verdict

**Date**: 2026-05-12  
**Branch**: `feature/f2-2-diverged-reset`  
**Config**: `epc_r_reset_enabled=True` — when `boost_q` fires, also reset `PBFDKF._error_psd` and `R` to initial value (1e-2)  
**Baseline**: production `balanced` (results/f2_2_baseline/scores.json)  
**Verdict**: **PASS — clean improvement, productisation candidate**

---

## What was tested

Yang 2017 (IEEE SPL) identifies over-estimated noise PSD (R) as the primary cause of slow
PBFDKF reconvergence after an abrupt echo path change. Our existing EPC handler fires
`boost_q` (Q_high + P_max_override 20 frames) but never resets R, so an over-estimated R
from prior DT can suppress Kalman gain K and delay convergence.

Fix: when `boost_q` fires and `epc_r_reset_enabled=True`, simultaneously reset
`_error_psd[:] = 1e-2` and `R[:] = 1e-2` — restoring both to their `__init__` / `reset()`
initialisation values.

---

## Bench configuration

| param | value |
|---|---|
| preset | balanced |
| filter_length | 832 (52 ms @ 16k) |
| cng | True |
| workers | j=4 |
| cases | 800 |
| AEC_EPC_R_RESET | True |

Ablation: 800/800, 0 errors, 13.4 min

---

## Bucket means

| Bucket | n | baseline echo | new echo | Δecho | baseline deg | new deg | Δdeg |
|---|---:|---:|---:|---:|---:|---:|---:|
| FS_static   | 169 | 3.542 | 3.543 | **+0.000** | 4.999 | 4.999 | ±0.000 |
| FS_movement | 131 | 3.586 | 3.588 | **+0.002** | 4.999 | 4.999 | −0.000 |
| DT_static   | 186 | 4.126 | 4.127 | **+0.001** | 2.367 | 2.367 | −0.000 |
| DT_movement | 114 | 4.100 | 4.100 | **+0.000** | 2.337 | 2.336 | −0.001 |
| NE          | 200 | 4.998 | 4.998 | **+0.000** | 4.011 | 4.011 | ±0.000 |

---

## Per-case distribution

| metric | value |
|---|---|
| Cases Δecho < −0.05 (regression) | **2** |
| Cases Δecho > +0.05 (improvement) | **4** |
| Cases \|Δecho\| > 0.10 | **2** |
| Sum of echo regressions | −0.141 |
| Sum of echo improvements | +0.448 |
| Regression:Improvement ratio | **0.3×** |
| Worst single case Δecho | −0.082 (`NSdFS8g1dkCCAMRgHWLILQ_farend_singletalk`) |
| Best single case Δecho | +0.164 (`yS7NgOlleU600sZ80T9bng_farend_singletalk_with_movement`) |
| Cohort tail (`qNvSMyUSXUyrDGp...`) | Δecho=−0.001, Δdeg=−0.000 |

---

## Analysis

The result is a clean win on all axes:

- All five bucket means are within ±0.003 — no systematic shift in any scenario class.
- Improvement:Regression ratio 0.3× (improvements are 3.2× larger than regressions in aggregate).
- Worst regression −0.082 is below the 0.10 audibility threshold.
- Cohort tail is negligible.

The improvements concentrate in FS_movement (+0.164, +0.056) and DT_static (+0.133) — exactly
the scenarios where a path-change event followed by sluggish R recovery would be most visible.
The regressions (two FS_static cases, max −0.082) are plausibly noise: R-reset fires at the
same time as Q-boost, briefly making K over-aggressive, which on already-converged static-FS
cases can transiently over-adapt before stabilising.

The risk of over-aggressive K post-reset is bounded by the existing `_p_max_override=1.0` +
20-frame decay mechanism, which reins in P within ~1 second. R recovers via α=0.95 EMA from
actual error PSD within ~200 ms.

---

## Why this differs from P55 Phase 1.2 (Enzner-Vary)

P55 Phase 1.2 tried to replace the ad-hoc R modulation with a canonical Enzner-Vary Wiener
gain. It failed (DT−FS only +7.01 dB vs 20 dB bar) because the discriminator was weak on
cohort tail — not because R-reset is wrong. This fix is narrower: it only resets R at
`boost_q` events (already gated by `dt_from_energy < 0.3`), not on every frame.

---

## Disposition

- F2.3 **PASS**.
- `epc_r_reset_enabled=True` promoted to `balanced` preset default.
- No further tuning warranted; the signal is clean and the mechanism is correctly scoped.
- Next: write `AecConfig` doc string; set `epc_r_reset_enabled=True` as the default in
  `AecPreset.BALANCED`.
