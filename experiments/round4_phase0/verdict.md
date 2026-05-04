# Round 4 — Phase 0 verdict

**Branch**: `algo/round4-perbin-res`  •  **Date**: 2026-05-03
**Baseline**: `baseline_v381_seeded` (deg/echo locked)
**Cases**: 800 (300 FS / 300 DT / 200 NE)

## Per-bucket means (ENR-active frames)

| Bucket | n | coh2_v | res/err_v | ne/err_v | g_voice_mean | g_voice_min | echo_dom_pct |
|---|---:|---:|---:|---:|---:|---:|---:|
| FS_static    | 169 | 0.166 | 7.012 | 0.977 | 0.478 | 0.188 | 4.40% |
| FS_movement  | 131 | 0.158 | 6.753 | 0.932 | 0.445 | 0.177 | 3.97% |
| NE           | 200 | 0.007 | 0.591 | 0.481 | 0.200 | 0.154 | 0.08% |
| DT_static    | 186 | 0.144 | 6.071 | 0.991 | 0.561 | 0.265 | 3.46% |
| DT_movement  | 114 | 0.147 | 5.713 | 0.963 | 0.562 | 0.276 | 3.32% |

NE has near-zero coh2 (no far signal) and near-zero classifier hits — safe.

## DT_movement worst-20 vs best-20 (locked by baseline_v381_seeded.deg)

| field | worst-20 | best-20 | Δ | direction |
|---|---:|---:|---:|---|
| coh2_mean_voice | 0.144 | 0.146 | -0.002 | **flat** |
| res_over_err_voice | **6.60** | **3.93** | **+2.66** | worst residual ↑ |
| ne_over_err_voice | 0.793 | 1.063 | -0.270 | worst NE-floor ↓ |
| g_voice_mean | 0.535 | 0.581 | -0.046 | worst already lower |
| g_voice_min | **0.233** | **0.330** | **-0.097** | worst already pressed harder |
| g_voice_p10 | 0.329 | 0.424 | -0.095 | same |
| echo_dominant_bin_pct | **0.023** | **0.030** | **-0.007** | classifier fires LESS on worst |

worst-20 mean baseline_deg = 1.397; best-20 mean = 3.312.

## Go/No-Go gate (per plan thresholds)

- `g_voice_mean = 0.535 > 0.5` ✓
- `res_over_err_voice = 6.60 > 0.4` ✓

→ **Plan says PROCEED with R1.**

## Yellow flag (not anticipated by plan)

The worst-20 cases already have **lower g** (more suppressed) and **fewer**
echo-dominant classifier hits than best-20. Translation:

1. The residual estimator IS firing strongly on worst (`res_over_err_voice` +2.66) —
   it has the right intuition.
2. But `coh2` stays ~0.14 across both worst and best (no separation). Since R1's
   classifier requires `coh2 > 0.5`, it almost never fires — and when it does, it
   fires more on best-20 than worst-20.
3. Suppressor is already moderate-to-low on worst-20 (`g_voice_p10 = 0.33`).
   Pushing it further may not help: AECMOS deg measures perceived speech damage,
   and the worst-20 cases (baseline_deg ≈ 1.4) are likely already ENR-blowing-up
   territory — adding more attenuation worsens speech damage without recovering
   echo suppression.

This pattern echoes Round 3 D3: RES end gets the signal but acting on it doesn't
help baseline-worst-20 (whose damage is upstream of the suppressor).

## R1 expectations

Running R1 v1 (cap=0.5, thresholds (0.5, 0.5, 0.3)) per plan. **Predicted outcome**:
- Echo bin pct hit-rate ~3% globally → small Δ on bucket means.
- Wins concentrate on mid-baseline DT cases (R3-style), not worst-20.
- FS may regress slightly because FS classifier hit-rate is also 3–4%.
- Most likely: sub-acceptance like R3.

If R1 v1 confirms this, R2/R3/R4 sub-branches will inherit the same problem:
classifier targets wrong population. Then close R4.

## DT_static worst-20 (sanity)

`echo_dominant_bin_pct`: worst 3.9% vs best 2.7% (**+1.2pp, only bucket where
worst > best**). DT_static may respond to R1 better than DT_mv.

## FS worst-20 (sanity)

FS worst-20 has *lower* classifier hit-rate (2.3–2.7% vs 4.8%) AND higher
ne_over_err. Means R1 will not aggressively touch FS worst → less risk of FS
regression. Good for safety.

## Decision

**PROCEED with R1 v1** per plan. Bench will confirm whether Phase 0's yellow
flag plays out as predicted.
