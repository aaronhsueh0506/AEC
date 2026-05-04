# Round 4 — Phase 0 per-bin RES analysis

Cases: 800 (baseline = baseline_v381_seeded, deg/echo locked)

## Per-bucket means (ENR active frames only)

| Bucket | n | coh2_voice | res/err_v | ne/err_v | noise/err | g_voice_mean | g_voice_min | g_voice_p10 | echo_dom_pct |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| FS_static | 169 | 0.166 | 7.012 | 0.977 | 0.455 | 0.478 | 0.188 | 0.280 | 0.0440 |
| FS_movement | 131 | 0.158 | 6.753 | 0.932 | 0.340 | 0.445 | 0.177 | 0.258 | 0.0397 |
| NE | 200 | 0.007 | 0.591 | 0.481 | 0.137 | 0.200 | 0.154 | 0.178 | 0.0008 |
| DT_static | 186 | 0.144 | 6.071 | 0.991 | 0.644 | 0.561 | 0.265 | 0.361 | 0.0346 |
| DT_movement | 114 | 0.147 | 5.713 | 0.963 | 0.738 | 0.562 | 0.276 | 0.370 | 0.0332 |

### DT_movement worst-20 vs best-20 (locked by baseline deg)

worst-20 mean deg = 1.397, best-20 mean deg = 3.312

| field | worst-20 | best-20 | Δ | sep |
|---|---:|---:|---:|---|
| coh2_mean_full | 0.132 | 0.129 | +0.003 | no |
| coh2_mean_voice | 0.144 | 0.146 | -0.002 | no |
| res_over_err_mean_full | 6.853 | 4.417 | +2.436 | YES |
| res_over_err_mean_voice | 6.598 | 3.934 | +2.664 | YES |
| ne_over_err_mean_full | 2.717 | 2.723 | -0.006 | no |
| ne_over_err_mean_voice | 0.793 | 1.063 | -0.270 | weak |
| noise_over_err_mean_full | 0.612 | 0.723 | -0.111 | weak |
| g_voice_mean | 0.535 | 0.581 | -0.046 | no |
| g_voice_min | 0.233 | 0.330 | -0.097 | weak |
| g_voice_p10 | 0.329 | 0.424 | -0.095 | weak |
| echo_dominant_bin_pct | 0.023 | 0.030 | -0.007 | weak |

### DT_static worst-20 vs best-20 (locked by baseline deg)

worst-20 mean deg = 1.337, best-20 mean deg = 3.454

| field | worst-20 | best-20 | Δ | sep |
|---|---:|---:|---:|---|
| coh2_mean_full | 0.141 | 0.115 | +0.026 | weak |
| coh2_mean_voice | 0.159 | 0.133 | +0.026 | weak |
| res_over_err_mean_full | 6.727 | 4.841 | +1.886 | weak |
| res_over_err_mean_voice | 6.505 | 4.565 | +1.940 | weak |
| ne_over_err_mean_full | 2.741 | 2.895 | -0.154 | no |
| ne_over_err_mean_voice | 0.939 | 1.181 | -0.242 | weak |
| noise_over_err_mean_full | 0.542 | 0.612 | -0.070 | weak |
| g_voice_mean | 0.551 | 0.559 | -0.009 | no |
| g_voice_min | 0.241 | 0.303 | -0.062 | weak |
| g_voice_p10 | 0.338 | 0.393 | -0.055 | weak |
| echo_dominant_bin_pct | 0.039 | 0.027 | +0.012 | YES |

### FS_movement worst-20 vs best-20 (locked by baseline echo)

worst-20 mean echo = 2.581, best-20 mean echo = 4.476

| field | worst-20 | best-20 | Δ | sep |
|---|---:|---:|---:|---|
| coh2_mean_full | 0.118 | 0.146 | -0.029 | weak |
| coh2_mean_voice | 0.132 | 0.163 | -0.030 | weak |
| res_over_err_mean_full | 5.235 | 7.598 | -2.363 | YES |
| res_over_err_mean_voice | 4.741 | 7.646 | -2.904 | YES |
| ne_over_err_mean_full | 3.215 | 2.059 | +1.156 | YES |
| ne_over_err_mean_voice | 1.436 | 0.739 | +0.697 | YES |
| noise_over_err_mean_full | 0.504 | 0.318 | +0.186 | YES |
| g_voice_mean | 0.475 | 0.495 | -0.019 | no |
| g_voice_min | 0.254 | 0.183 | +0.071 | weak |
| g_voice_p10 | 0.334 | 0.277 | +0.057 | weak |
| echo_dominant_bin_pct | 0.023 | 0.048 | -0.025 | YES |

### FS_static worst-20 vs best-20 (locked by baseline echo)

worst-20 mean echo = 2.220, best-20 mean echo = 4.558

| field | worst-20 | best-20 | Δ | sep |
|---|---:|---:|---:|---|
| coh2_mean_full | 0.147 | 0.139 | +0.008 | no |
| coh2_mean_voice | 0.173 | 0.152 | +0.020 | weak |
| res_over_err_mean_full | 4.714 | 7.519 | -2.805 | YES |
| res_over_err_mean_voice | 4.044 | 7.369 | -3.325 | YES |
| ne_over_err_mean_full | 3.007 | 2.213 | +0.794 | weak |
| ne_over_err_mean_voice | 1.047 | 0.978 | +0.068 | no |
| noise_over_err_mean_full | 0.387 | 0.337 | +0.050 | weak |
| g_voice_mean | 0.458 | 0.535 | -0.077 | weak |
| g_voice_min | 0.215 | 0.224 | -0.009 | no |
| g_voice_p10 | 0.306 | 0.322 | -0.017 | no |
| echo_dominant_bin_pct | 0.027 | 0.048 | -0.021 | YES |

## Go/No-Go gate (DT_movement worst-20 by baseline deg)

- worst-20 g_voice_mean: 0.535
- worst-20 g_voice_min: 0.233
- worst-20 g_voice_p10: 0.329
- worst-20 res_over_err_voice: 6.598
- worst-20 ne_over_err_voice: 0.793
- worst-20 echo_dominant_bin_pct: 0.0233

**Decision**: PROCEED with R1. Echo bin not pressed (g_voice_mean > 0.5)
AND residual dominant (res/err_voice > 0.4). Per-bin classifier should
have signal to act on.
