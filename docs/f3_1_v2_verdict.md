# F3.1 v2 verdict — `not epc_active` gate (2026-05-12)

## TL;DR

**F3.1 v2 = marginal trade-off, not a clear win**. Adding `not epc_active` to the gate helps 2 of 5 FS_movement outliers substantially (s0oJqM6Y +0.070, Khk1qeM +0.084) but barely touches the worst case (Lsa5Wpw stays at Δecho −0.301), and loses some DT_static NE-preservation gains (KOy0eft +0.060 → −0.037). Bucket means net wash (FS_movement +0.0009, DT_static −0.0008). Lsa5Wpw is not an EPC-hangover problem and needs separate diagnosis.

Decision (recommended): **ship F3.1 v2** as the F3.1 productisation candidate — the gate is principled (skip EPC hangover where erl/long_window are most stale) and unit-test/bench-clean, even if AECMOS net effect is small. Defer Lsa5Wpw / Wv6yp6N1 / KOy0eft to a future per-case audit pass.

## 800-case j4 BALANCED bench — F3.1 v1 vs v2 vs main

### Bucket means

| Bucket | n | main | F3.1 v1 | F3.1 v2 |
|---|---:|---:|---:|---:|
| FS_static echo   | 169 | 3.542 | 3.541 | 3.541 |
| FS_movement echo | 131 | 3.586 | 3.580 | **3.581** |
| DT_static deg    | 186 | 2.367 | 2.368 | 2.367 |
| DT_movement deg  | 114 | 2.337 | 2.334 | 2.334 |
| NE deg           | 200 | 4.011 | 4.011 | 4.011 |

### Δ vs main baseline

| Bucket | v1 Δecho | v1 Δdeg | v2 Δecho | v2 Δdeg |
|---|---:|---:|---:|---:|
| FS_static   | −0.0011 | +0.0000 | −0.0010 | +0.0000 |
| FS_movement | **−0.0055** | +0.0000 | **−0.0046** | +0.0000 |
| DT_static   | +0.0003 | **+0.0006** | +0.0006 | **−0.0002** |
| DT_movement | +0.0017 | −0.0030 | +0.0015 | −0.0024 |
| NE          | +0.0000 | +0.0000 | +0.0000 | +0.0000 |

All bucket means still safely inside the `bench_aecmos.py` bars (FS echo ≥ −0.02, NE deg ≥ −0.01). v2 inches FS_movement closer to baseline at the cost of a marginal DT_static deg drift.

### 5 FS_movement outliers (Δecho vs main)

| stem | v1 | v2 | Δ(v2 − v1) |
|---|---:|---:|---:|
| Lsa5Wpw   | **−0.301** | **−0.301** | 0.000 |
| s0oJqM6Y  | −0.213 | **−0.143** | **+0.070** |
| wr54weK   | −0.167 | −0.161 | +0.006 |
| pG9Bikvr  | −0.140 | −0.142 | −0.002 |
| Khk1qeM   | −0.075 | **+0.009** | **+0.084** |

Two cases (s0oJqM6Y, Khk1qeM) substantively recovered. Lsa5Wpw is untouched — its bad bursts (t = 13s / 15s / 16s from the time-localised A/B) happen at frames where `filter_converged=True AND epc_active=False`, i.e. *outside* the v2 gate skipping. Need a different gate condition or accept it.

### DT_static gainers tracked

| stem | v1 Δdeg | v2 Δdeg |
|---|---:|---:|
| tl5UFRCX | +0.178 | +0.165 |
| i2BU43nm | +0.174 | +0.174 |
| Wv6yp6N1 | +0.063 | +0.037 |
| **KOy0eft** | **+0.060** | **−0.037** |

KOy0eft flipped from gain to regress (−0.097). v2's gate suppresses F3.1 firing during EPC hangover, and DT_static contains DT scenarios where EPC may fire from talker switches — losing F3.1 in those windows removes some NE-preservation that v1 had.

### |Δ|>0.05 histogram (v2 vs main)

| Bucket | echo regress | echo gain | deg regress | deg gain |
|---|---:|---:|---:|---:|
| FS_static   | 3 | 1 | 0 | 0 |
| FS_movement | **4** | 1 | 0 | 0 |
| DT_static   | 2 | 2 | 3 | 2 |
| DT_movement | 0 | 1 | 4 | 1 |
| NE          | 0 | 0 | 0 | 0 |

Compared to F3.1 v1:
- FS_static echo regress 4 → 3
- FS_movement echo regress 5 → 4
- DT_static deg gain 4 → **2** (lost two gainers)
- DT_movement deg regress 4 → 4 (unchanged)

## Mechanism

The outlier audit (instrumentation script `tools/research/f3_1_outlier_audit.py`) showed:

- F3.1 fires only **5–32 %** of frames on FS_movement outliers (rest are `filter_converged=False`).
- Among those fires, time-localised A/B diff shows the perturbation is **sparse + bursty** (most 1-second windows show 0 % rms diff, a few show 30–150 %).
- The high-diff bursts cluster in two regimes:
  1. EPC hangover with fleeting convergence — `_long_window_far_psd` / `_erl_estimate` pinned to previous room. v2 skips these.
  2. Background-noise-dominated frames in FS_singletalk (mic_rms 4 × far_rms with non-far-correlated content) — F3.1 correctly says "preserve" but AECMOS labels any non-NE energy in FS as echo. v2 does NOT skip these.

Lsa5Wpw belongs to regime (2), not (1). That is why v2 leaves it unchanged.

## What this says about F3.1

- **Mechanism is correct**: the metric finds genuine per-bin "this energy is not echo-shaped" signal.
- **AECMOS-leverage is small** because (a) F3.1 fires rarely (<10 % on most FS cases, 0 % on NE), and (b) the gain compute downstream still has scalar `effective_dt` gating much of the per-bin decision.
- The plan's expected leverage was through F3.1 + F3.2 together. F3.1 v2 alone is principled and safe; the productisable case for F3.1 is built when F3.2 propagates per-bin evidence into the scalar-gated paths.

## Decision

1. **F3.1 v2 is the F3.1 productisation candidate** (commit 0e08de8). Default OFF; flag = `use_mic_excess_evidence`.
2. **Open `next-step` items, not blocking F3.2:**
   - Lsa5Wpw / Wv6yp6N1 / KOy0eft per-case audit (regime 2 mechanism, background-noise labeling).
   - Possible secondary gate: `not is_high_mic_low_far_ratio` (mic_rms / far_rms < N) to suppress F3.1 in background-noise frames.
3. **F3.2 (per-bin effective_dt) remains the next experimental arc**, independent of erl_estimate accuracy.

## Files / artefacts

- Code: `python/aec.py` + `python/res_refactored/gain_computer.py` (commit 0e08de8)
- Audit script: `tools/research/f3_1_outlier_audit.py`
- Bench output: `/tmp/f3_1_v2_on/` (800 WAVs)
- Scores: `/tmp/f3_1_v2_scores/{scores.json, result.md}`
- Unit tests: `python/test_f3_1_mic_excess.py` (6/6), `python/test_f2_1_epc_state_reset.py` (4/4), `python/test_p52_regime.py` (13/13) all green
- Per-frame trace CSVs: `/tmp/f3_1_audit/` (10 files: 5 outliers × {OFF, ON})
