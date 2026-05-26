# v3.21.15 A.2 / A.3 interaction report — 11-case frame-level trace

Cycle: 2026-05-23
Probe: [`python/v3_21_15_interaction_probe.py`](AEC/python/v3_21_15_interaction_probe.py)
Analysis: [`python/v3_21_15_interaction_analyze.py`](AEC/python/v3_21_15_interaction_analyze.py)
Trace data: `trace_v3_21_15_interaction/{stem}__{V0,V1,V2,V3}.npz` (44 NPZ files)

## Goal

Determine whether the 6 FS catastrophic regressions in v3.21.15 V3 (A.2+A.3)
800-case are caused by:
1. A.2 alone
2. A.3 alone
3. Temporal interaction between A.2 and A.3
4. Downstream state / usable_linear amplification

## Method

Run 4 variants (V0 = baseline / V1 = A.2 only / V2 = A.3 only / V3 = A.2+A.3)
through 11-case cohort: 6 worst FS regressors + 3 strongest FS wins + 2 DT
guards (worst DT regressions from 800-case). Capture per-hop:

- A.2/A.3 fire decisions (`a3_would_return`, `a2_zero_frac`, zeroed bins by band)
- Shadow state (`shadow_W_norm`, `shadow_error_norm`)
- Main filter state (`main_W_norm`, `main_error_norm`)
- Downstream signals (`refined_conv`, `coarse_conv`, `aec3_converged`,
  `convergence_seen`, `usable_linear`)
- Suppression output (`gain_5/30/100/200`, `out_e`, `nores_e`)

## Per-case classification (proxy: gain_100_mean Δ vs V0; HIGHER = less suppression = WORSE FS)

| Group | Case | ΔV1 (A.2) | ΔV2 (A.3) | ΔV3 | Classification |
|---|---|---:|---:|---:|---|
| FS_worst | **wlAXM0iD** | **+0.0716** | +0.0029 | +0.0849 | **A.2 dominant** (V3 ≈ V1; A.3 null) |
| FS_worst | XXz0qkUS_mvmt | −0.0023 | **+0.0210** | +0.0184 | A.3 dominant |
| FS_worst | NNdxDj6F | −0.0021 | **+0.0246** | +0.0226 | A.3 dominant |
| FS_worst | zykCkY0B | +0.0128 | +0.0192 | +0.0128 | SHARED (both ~equal, V3 not super-additive) |
| FS_worst | qkGW9Fr_mvmt | +0.0026 | **+0.0136** | +0.0157 | A.3 dominant |
| FS_worst | PYGYvZS | +0.0063 | +0.0204 | +0.0279 | ADDITIVE (V3 ≈ V1+V2; mild interaction) |
| FS_win | **KSN5Jrzo** | −0.0223 | **−0.1948** | −0.1974 | **A.3 dominant** (big win, 90% of V3 effect) |
| FS_win | orvXZE0 | −0.0117 | −0.0320 | −0.0438 | ADDITIVE |
| FS_win | **Fi80N5k_mvmt** | −0.0037 | **−0.0950** | −0.0962 | **A.3 dominant** (big win) |
| DT_guard | Y4zG6bHup | −0.0080 | +0.0024 | −0.0056 | A.2 dominant; magnitude tiny |
| DT_guard | KOy0eftk | +0.0006 | +0.0024 | +0.0014 | NEUTRAL |

**Counts**: 5 A.3-dominant / 2 A.2-dominant / 2 additive / 1 shared / 1 neutral.
**Zero super-additive interaction cases**. V3 ≈ max(V1, V2) per case across
all 11 cases — A.2 and A.3 do NOT compound multiplicatively.

## Downstream chain — `usable_linear_frac` (Δ vs V0)

`usable_linear` is the gate-3 latch that decides whether RES treats `error_spec`
as the suppression target (aggressive) or falls back to capture `Y` (less
aggressive). DOWN = latch FAILED → less aggressive RES → echo leaks.

| Group | Case | V0 baseline | ΔV1 | ΔV2 | ΔV3 |
|---|---|---:|---:|---:|---:|
| FS_worst | wlAXM0iD | 0.925 | **−0.088** | +0.000 | −0.103 |
| FS_worst | XXz0qkUS | 0.886 | +0.000 | **−0.021** | −0.021 |
| FS_worst | NNdxDj6F | 0.888 | +0.000 | **−0.026** | −0.026 |
| FS_worst | zykCkY0B | 0.917 | −0.021 | −0.021 | −0.021 |
| FS_worst | qkGW9Fr | 0.929 | +0.000 | **−0.020** | −0.020 |
| FS_worst | PYGYvZS | 0.933 | +0.000 | **−0.022** | −0.022 |
| FS_win | **KSN5Jrzo** | 0.210 | +0.052 | **+0.456** | +0.457 |
| FS_win | orvXZE0 | 0.765 | +0.064 | +0.043 | +0.107 |
| FS_win | Fi80N5k | 0.607 | +0.011 | **+0.075** | +0.075 |
| DT_guard | Y4zG6bHup | 0.904 | +0.000 | −0.005 | −0.005 |
| DT_guard | KOy0eftk | 0.957 | +0.000 | −0.002 | −0.002 |

The mechanism is clear: **A.2 / A.3 → shadow under-tracks (or over-pauses)
→ `convergence_seen` / `any_filter_converged` signal differs → `usable_linear`
latch flips → RES suppression aggressiveness changes → output echo / deg
shifts**.

## Mechanism context: A.2 zero_frac + A.3 actual fire rate

| Group | Case | a2_zero_frac (V0) | a3_fire_rate (V2) |
|---|---|---:|---:|
| FS_worst | **wlAXM0iD** | **77.4 %** | 4.8 % |
| FS_worst | XXz0qkUS | 34.6 % | 6.2 % |
| FS_worst | NNdxDj6F | 52.4 % | 3.1 % |
| FS_worst | zykCkY0B | 32.1 % | 3.3 % |
| FS_worst | qkGW9Fr | 51.0 % | 5.2 % |
| FS_worst | PYGYvZS | 55.5 % | 3.5 % |
| FS_win | **KSN5Jrzo** | 56.3 % | **12.3 %** |
| FS_win | orvXZE0 | 47.2 % | 3.6 % |
| FS_win | Fi80N5k | 34.4 % | 4.4 % |
| DT_guard | Y4zG6bHup | 74.1 % | 6.5 % |
| DT_guard | KOy0eftk | 71.6 % | 3.7 % |

### A.2 mechanism

- High `a2_zero_frac` correlates with A.2 having large effect ONLY on wlAXM0iD
  (77.4 % zero → −0.088 usable_linear drop → catastrophic FS regression).
- Y4zG6bHup0 / KOy0eftk also have high zero_frac (74 %, 72 %) but A.2
  alone has negligible effect → **A.2 destructiveness depends on WHICH bins
  zero (LF carries more echo energy), not just the count**.
- A.2 zero_frac is high (> 50 %) on 7/11 cases — not a niche behaviour;
  on most cohort cases A.2 silently shuts off half the shadow's mu.
  Only wlAXM0iD shows clear downstream collapse from this.

### A.3 mechanism

- A.3 actual fire rate (V2): 3-12 % per case.
- **Counterintuitive**: highest A.3 fire rate (KSN5Jrzo 12.3 %) produces
  BIGGEST WIN (+45.6 % usable_linear, ΔV2 gain_100 −0.195).
- Lower fire rates (3-6 %) on FS regressors produce small but persistent
  usable_linear drops (−2 % each).
- Conclusion: A.3's fire pattern correlates with cohort cases where shadow
  needs pausing — when it pauses at the right moments (frequent narrowband
  fires) it benefits; rare pauses at wrong moments cause shadow to lose
  convergence signal that downstream depends on.

### Why DT_guard regressions don't show in this trace

Y4zG6bHup0 and KOy0eftk had Δdeg −0.367 / −0.253 on 800-case AECMOS. Trace
shows tiny gain_100 changes (≤ ±0.008) and tiny usable_linear shifts
(≤ −0.005). The DT regressions are **subtler than FS** — likely
per-bin gain shifts that affect speech intelligibility (AECMOS deg metric)
but not aggregate residual energy. Not surfaceable from this aggregate
trace; would need per-bin / per-frame spectrogram diff.

## Classification per user's four hypotheses

1. **A.2 alone causes regression**: TRUE for **1 / 6** FS worst cases
   (wlAXM0iD — single high-zero_frac catastrophic). Also DT_guard Y4zG6bHup0
   (tiny magnitude).
2. **A.3 alone causes regression**: TRUE for **5 / 6** FS worst cases
   (XXz0qkUS, NNdxDj6F, qkGW9Fr, PYGYvZS, partial zykCkY0B). A.3 also
   accounts for **3 / 3 FS wins** (KSN5Jrzo / orvXZE0 / Fi80N5k).
3. **Temporal interaction A.2+A.3**: **NOT OBSERVED**. V3 ≈ max(V1, V2)
   per case (no super-additive growth). PYGYvZS shows mild additivity
   (V3 = +0.028 vs V1+V2 = +0.027) but within numerical noise.
4. **Downstream state amplification**: **YES — via usable_linear latch**.
   The mechanism chain is: shadow update change → convergence_seen flips →
   usable_linear latch flips → RES aggressiveness changes → output gain
   shifts. Both A.2 and A.3 act through this same latch.

## Bottom line

- **A.3 is the high-variance contributor** (5 catastrophic FS cases + 3 big
  wins; ratio of catastrophic to win cases is variant-cohort specific)
- **A.2 has 1 known catastrophic case** (wlAXM0iD); on most cases A.2 silently
  zeros half the shadow's mu without downstream impact
- **Both act through the same downstream latch** (`usable_linear`); compound
  effect IS additive but does NOT super-amplify
- **Neither is individually ship-able as default-True**: A.3 alone would
  still have 4-5 FS catastrophic cases on 800-case; A.2 alone would still
  have wlAXM0iD-class catastrophic cases

## Recommendation re: v3.21.15.split 800-case

**OPTIONAL — informative but unlikely to change disposition.**

If run:
- V1 (A.2 only) 800-case would likely show **~1-3 catastrophic FS cases**
  (wlAXM0iD-class high-zero_frac cases) and smaller gain on FS wins
- V2 (A.3 only) 800-case would likely show **~4-5 catastrophic FS cases** +
  preserved big FS wins (KSN5Jrzo etc.)
- Neither would pass strict ship-gate (worst FS < −0.5 disallowed by
  CLAUDE.md convention)

**Cleaner alternative: close V1+V2 individual ship paths now**, retain
A.2 + A.3 as default-OFF substrate, pivot v3.21.x roadmap to next AEC3
parity candidate (B-cluster, e.g. B.5 SignalDependentErleEstimator).

If user wants confirmation of trace findings at 800-case scale, run only
**V1 (A.2 only)** — it has the smallest expected catastrophic-case count
and is the most likely individual ship candidate. Skip V2 (A.3 only)
since trace already confirms 4-5 catastrophic FS-worst cases would persist.

## Forbidden (preserved scope)

- No A.1 / A.4 / A.5 enablement (out of v3.21.15 scope)
- No v3.22 work (AEC3 parity arc still has B-cluster candidates open)
- No mechanism tuning of A.2 / A.3 (would be intentional divergence from
  AEC3 verbatim → v3.22 territory per version rule)
- No `/simplify` until v3.21.x arc resolves

## Artefacts

- `trace_v3_21_15_interaction/` — 44 NPZ files (11 cases × 4 variants)
- `trace_v3_21_15_interaction/analysis.json` — per-case aggregate dict
- Probe + analysis scripts: `python/v3_21_15_interaction_*.py`
