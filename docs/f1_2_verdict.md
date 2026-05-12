# F1.2 PathChangeRegimeHandler gate_mode Ablation — 800-case Verdict

**Date**: 2026-05-12  
**Branch**: `feature/f2-2-diverged-reset`  
**Config**: `regime_gate_mode='streak_only'` (AEC3-style 5-block consecutive, no DT energy gate)  
**Baseline**: production `regime_gate_mode='energy'` (legacy `dt_from_energy < 0.3` DT guard)  
**Verdict**: **FAIL — energy gate is load-bearing; streak_only causes severe FS regression**

---

## What was tested

PathChangeRegimeHandler fires `boost_q` + `pause_main` + `reverse_copy` decisions when it
detects an echo path change. Two gate modes:

- `energy` (production): path-change decisions are additionally gated by `dt_from_energy < 0.3`
  (far-end energy is high, DT unlikely). This prevents the regime from activating during
  doubletalk when far energy is suppressed by near-end speech.
- `streak_only` (ablation): path-change fires on 5 consecutive blocks where shadow error
  < main error × 0.65. No DT energy gate. Mirrors WebRTC AEC3 shadow-copy streak rule.

---

## Bench configuration

| param | value |
|---|---|
| preset | balanced |
| filter_length | 832 (52 ms @ 16k) |
| cng | True |
| workers | j=4 |
| cases | 800 |
| AEC_REGIME_GATE_MODE | streak_only |

Baseline (energy gate): 800/800, 0 errors  
Ablation (streak_only): 800/800, 0 errors, 13.4 min

---

## Bucket means

| Bucket | n | baseline echo | streak echo | Δecho | baseline deg | streak deg | Δdeg |
|---|---:|---:|---:|---:|---:|---:|---:|
| FS_static   | 169 | 3.542 | 3.515 | **−0.028** | 4.999 | 4.999 | ±0.000 |
| FS_movement | 131 | 3.586 | 3.567 | **−0.019** | 4.999 | 4.999 | ±0.000 |
| DT_static   | 186 | 4.126 | 4.117 | −0.009 | 2.367 | 2.379 | +0.011 |
| DT_movement | 114 | 4.100 | 4.092 | −0.008 | 2.337 | 2.347 | +0.010 |
| NE          | 200 | 4.998 | 4.998 | −0.000 | 4.011 | 4.011 | ±0.000 |

---

## Per-case distribution

| metric | value |
|---|---|
| Cases Δecho < −0.05 (regression) | **46** |
| Cases Δecho > +0.05 (improvement) | 14 |
| Cases \|Δecho\| > 0.10 | **33** |
| Sum of echo regressions | **−10.443** |
| Sum of echo improvements | +1.655 |
| Regression:Improvement ratio | **6.3×** |
| Worst single case Δecho | −0.760 (`Y91uE2tRg0SUB2a9XjT30w_farend_singletalk_with_movement`) |
| Best single case Δecho | +0.269 (`kZogUfYct0qMwSqvRTwOVg_doubletalk_with_movement`) |
| Cohort tail (`qNvSMyUSXUyrDGp...`) | Δecho=−0.010, Δdeg=−0.000 |

---

## Root cause: energy gate prevents pause_main from firing during genuine FS

The `dt_from_energy < 0.3` gate in `energy` mode ensures `pause_main` and `boost_q` only
fire when far-end energy is strong (i.e., FS conditions where path-change adaptation is
safe). Without this gate, `streak_only` triggers the regime handler on any 5-block streak
regardless of DT state. In FS scenarios, this causes:

1. **pause_main fires during legitimate FS adaptation**: main filter frozen, echo passes.
2. **boost_q destabilises convergence**: unnecessary Q inflation in already-converged
   FS periods increases adaptation noise.
3. **DT Δdeg +0.011/+0.010 improvement**: the DT slight benefit (NE preserved more)
   confirms the handler fires more in DT too — but the FS echo cost is 6.3× the DT gain.

The asymmetry is telling: DT improvement is real but small; FS regression is large and
widespread (46 cases with |Δecho| > 0.05, 33 cases > 0.10).

---

## Why WebRTC AEC3 streak rule works there but not here

AEC3 shadow is NLMS (no Kalman state), operates at a fixed step size, and the
shadow-copy trigger transfers weights only — no P/Q/R state involved. Our
PathChangeRegimeHandler's `pause_main` + `boost_q` involve Kalman state manipulation
(P_max_override, Q_high) that is significantly more disruptive when falsely triggered.
The AEC3 streak rule was designed for a stateless NLMS copy, not for a Kalman regime
handler.

---

## Disposition

- F1.2 CLOSED FAIL.  
- `regime_gate_mode='energy'` confirmed as the correct production gate — the DT energy
  guard is load-bearing on FS performance.
- `regime_gate_mode='streak_only'` and `AecConfig.regime_gate_mode` field retained as
  research/ablation knob (default OFF).  
- No further recalibration of `streak_only` warranted: the mechanism mismatch (NLMS
  assumption vs Kalman state manipulation) is structural.
