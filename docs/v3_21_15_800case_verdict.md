# v3.21.15 A.2 + A.3 combined — 800-case verdict

Cycle: 2026-05-23
Cohort: `wav/aec_challenge_blind/` (800 cases: 186 DT_static / 114 DT_movement / 169 FS_static / 131 FS_movement / 200 NE)
V0 baseline: `out_800_v3_21_15_V0/` (current HEAD config, all v3.21.14 flags OFF)
V3 candidate: `out_800_v3_21_15_V3/` (`AEC_SHADOW_NOISE_GATE=1 AEC_SHADOW_POOR_EXCITATION=1`; A.1/A.4/A.5 OFF)
Config: `preset=balanced / filter=832 / --cng / --parallel / --workers 4` (CLAUDE.md standard)

## TL;DR

**Pareto trade-off, not a strict ship.** V3 trades **FS echo cancellation
for DT deg preservation** at roughly matched magnitude:
- DT_static Δdeg bucket **+0.027** mean / ΣΔdeg = **+5.08** across 186 cases
- FS_static Δecho bucket **−0.029** mean / ΣΔecho = **−4.93** across 169 cases
- 0 DT catastrophic regressions (< −0.50) ✓
- **6 FS catastrophic regressions** (< −0.50 echo), worst −1.054 on wlAXM0iD ✗

The 12-case 6/7 PASS did NOT generalise — 12-case favoured A.2 (noise_gate)
on positive subset (xQEUtY2 +0.179). 800-case reveals high-variance
behaviour: A.2 and/or A.3 hard-zero shadow updates in conditions where
the conservative shadow under-tracks and feeds bad convergence signal to
the main filter on FS singletalk.

**Recommendation**: do NOT ship V3 (A.2 + A.3 combined) as-is. Run
attribution split (V1 = A.2 only / V2 = A.3 only on 800-case) to identify
which sub-step causes FS catastrophic regressions; the other (if any)
may ship individually.

## 800-case bucket means + Δ

| Bucket | N | V0 echo | V0 deg | V3 echo | V3 deg | Δecho | Δdeg |
|---|---:|---:|---:|---:|---:|---:|---:|
| DT_movement | 114 | 4.188 | 2.464 | 4.184 | 2.472 | **−0.004** | **+0.007** |
| DT_static | 186 | 4.249 | 2.409 | 4.236 | 2.436 | **−0.013** | **+0.027** |
| FS_movement | 131 | 3.532 | 4.999 | 3.518 | 4.999 | **−0.014** | +0.000 |
| FS_static | 169 | 3.656 | 4.999 | 3.627 | 4.999 | **−0.029** | +0.000 |
| NE | 200 | 4.998 | 4.038 | 4.998 | 4.038 | −0.000 | −0.000 |

## Per-case distribution (Δ = V3 − V0)

| Bucket | metric | median | mean | std | p10 | p90 | worst | best |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| DT_static | echo | +0.000 | −0.013 | 0.074 | −0.061 | +0.027 | −0.639 | +0.229 |
| DT_static | deg | +0.000 | **+0.027** | 0.137 | −0.055 | +0.126 | −0.367 | **+0.842** |
| DT_movement | echo | +0.000 | −0.004 | 0.036 | −0.027 | +0.014 | −0.171 | +0.137 |
| DT_movement | deg | +0.001 | +0.007 | 0.090 | −0.058 | +0.045 | −0.200 | +0.635 |
| FS_static | echo | −0.000 | **−0.029** | 0.157 | −0.085 | +0.035 | **−1.054** | +0.444 |
| FS_movement | echo | −0.001 | −0.014 | 0.116 | −0.045 | +0.030 | **−0.894** | +0.294 |
| NE | deg | +0.000 | −0.000 | 0.005 | +0.000 | +0.000 | −0.044 | +0.032 |

Most cases are near-neutral (median ≈ 0); the bucket means are driven by
heavy tails.

## Strict Pareto outcome counts (800 cases)

| Outcome | Count | Definition |
|---|---:|---|
| Strict WIN (both metrics ≥, sum > 0.01) | 60 | A.2/A.3 helps cleanly |
| Strict LOSE (both metrics ≤, sum < −0.01) | 58 | A.2/A.3 hurts cleanly |
| MIXED trade-off | 248 | gain on one metric, loss on other |
| NEUTRAL (|Δ| < 0.01 both) | 434 | A.2/A.3 doesn't fire / no effect |

~46 % of cases see ANY effect (366 cases). Of those, win:lose is roughly
symmetric (60 vs 58), with 248 trade-off cases dominating. Pareto-mixed
pattern.

## Worst-20 DT Δdeg regression

| Case | Bucket | V0 deg | Δdeg |
|---|---|---:|---:|
| Y4zG6bHup06zWMoq3OvZqQ_doubletalk | DT_static | 3.181 | **−0.367** |
| KOy0eftktkuJf180xtXudg_doubletalk | DT_static | 2.004 | **−0.253** |
| zpiSOkxpHkCs5SqdOo5ZIQ_doubletalk_with_movement | DT_movement | 3.790 | −0.200 |
| m6ciKvH6AEe7Yi2ptKjj1g_doubletalk_with_movement | DT_movement | 2.117 | −0.184 |
| WtQs4a0YeU2B0dQWhS7gmg_doubletalk | DT_static | 2.313 | −0.177 |

Worst 5 all ≥ −0.40. Only 2/300 DT cases below −0.20 (Y4zG6bHup0 / KOy0eftktk).
**0 catastrophic** (< −0.50). Per CLAUDE.md ship-gate convention (worst DT < −0.30 disallowed),
Y4zG6bHup0 −0.367 marginally violates the −0.30 bar.

## Worst-20 FS Δecho regression

| Case | Bucket | V0 echo | Δecho |
|---|---|---:|---:|
| wlAXM0iDgkm06i7UdRww1w_farend_singletalk | FS_static | 3.811 | **−1.054** |
| XXz0qkUSd0GT4dsywxpfJg_farend_singletalk_w_movement | FS_movement | 3.873 | **−0.894** |
| NNdxDj6FEk6CAwvbW01bUg_farend_singletalk | FS_static | 4.118 | **−0.812** |
| zykCkY0BZEWhtSbeZJm7pw_farend_singletalk | FS_static | 3.760 | **−0.709** |
| qkGW9Frbs0Gq5gdfsztA2g_farend_singletalk_w_movement | FS_movement | 3.707 | **−0.705** |
| PYGYvZSlIUuUakjw9XwS9g_farend_singletalk | FS_static | 4.232 | **−0.636** |
| ksP3OuSnpUa9Si2ttiUSoA_farend_singletalk | FS_static | 3.896 | −0.444 |
| SFvlSygv4ke9wCrv8LWvYQ_farend_singletalk | FS_static | 3.661 | −0.433 |
| JJ1abErD8USzCkA9Oosd1Q_farend_singletalk | FS_static | 4.358 | −0.430 |
| o2wfdvOGwU6M8Fmn2dCvOA_farend_singletalk | FS_static | 3.087 | −0.385 |

**6 catastrophic FS regressions** (< −0.50), 4 of them FS_static, 2 FS_movement.
All 6 cases start from "moderate echo cancellation" (V0e 3.7–4.4) and drop
to "poor echo cancellation" (V3e ≈ 2.8–3.4). Real per-case losses, not
noise.

## Best-10 gains (for context)

| Case | Bucket | Δecho | Δdeg |
|---|---|---:|---:|
| qVd1gtwQ0k2lVRqPVp1NKQ_doubletalk | DT_static | +0.026 | **+0.842** |
| nyT6FUUdu0W8UpvjP1rRgQ_doubletalk | DT_static | +0.107 | **+0.658** |
| Hp5g1asacUCt5rJVLO1FuQ_doubletalk_with_movement | DT_movement | +0.063 | **+0.635** |
| KSN5Jrzo7kaixP0z8xfr4Q_farend_singletalk | FS_static | **+0.444** | 0 |
| orvXZE0juUeRPAAdjZSqoA_farend_singletalk | FS_static | **+0.426** | 0 |
| Fi80N5kW9U6nwaoS04O3vQ_farend_singletalk_w_movement | FS_movement | **+0.294** | 0 |

Big DT deg gains (up to +0.842) exist on the win side. FS echo gains (up
to +0.444) exist too. The trade-off is mixed-bag: A.2 + A.3 helps some
cases substantially, hurts others substantially.

## Mechanism hypothesis (NOT trace-confirmed)

A.2 (noise_gate hard-zero on low X²) is high-variance:
- ON xQEUtY2 (12-case favourable): noise_gate prevents shadow update on
  noisy bins → cleaner shadow tracking → +0.179 echo
- ON wlAXM0iD (800-case worst FS regressor): noise_gate fires too often
  → shadow under-tracks → main filter convergence signal stale →
  echo cancellation regresses −1.054

A.3 (poor_excitation startup) is more predictable but combined with A.2
amplifies variance. Without per-case trace, can't split attribution
between A.2 and A.3 contribution to the 6 catastrophic regressions.

## Plan disposition (per v3.21.15 disposition matrix)

> **12-case PASS, 800-case FAIL**: retain A.2 + A.3 as default-OFF
> substrate; root-cause 800-case regression; possibly split A.2 / A.3
> individual ship

**Outcome**: V3 (A.2 + A.3 combined) **CLOSED no-ship**. Substrate
retained (default-OFF, env hooks shipped). Recommend split-attribution
to determine if A.2 alone or A.3 alone is individually ship-eligible.

## Recommended next step: split-attribution 800-case

Run two more 800-case renders:
- `out_800_v3_21_15_V1/` with `AEC_SHADOW_NOISE_GATE=1` only (A.2 alone)
- `out_800_v3_21_15_V2/` with `AEC_SHADOW_POOR_EXCITATION=1` only (A.3 alone)

Compare per-bucket + worst-N to V0. If one of A.2/A.3 carries the FS
catastrophic regressions and the other is clean, the clean one becomes
v3.21.15 ship candidate (split from combo).

Three plausible outcomes:
1. A.2 carries FS regressions, A.3 clean → ship A.3 as v3.21.15
2. A.3 carries FS regressions, A.2 clean → ship A.2 as v3.21.15
3. Both contribute → CLOSED, A.2 + A.3 stay substrate; v3.21.x roadmap
   pivots to next AEC3 parity candidate (B-cluster)

## Production state (unchanged)

- `main` unchanged; `__version__` unchanged (v3.21.6)
- All 5 v3.21.14 flags retained as default-OFF substrate
- Env hooks unchanged (`AEC_SHADOW_NOISE_GATE` / `_POOR_EXCITATION` / etc.)
- No production code changes proposed

## Artefacts

- `out_800_v3_21_15_V0/` + `results/800_v3_21_15_V0/` (baseline)
- `out_800_v3_21_15_V3/` + `results/800_v3_21_15_V3/` (A.2 + A.3 combo)
- 12-case prior: `out_12_v3_21_14_{A,A2,A3}/` + `out_12_v3_21_15_V3/`
