# Round 3 — Phase 0 trace analysis

Cases: 800 (baseline = baseline_v381_seeded)

## Per-bucket means

| Bucket | n | once% | epc% | epc_max_run | pmax% | pfloor% | delay_shift | scale_med | pre_epc_dt_mean |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| FS_static | 169 | 64% | 0.40 | 475 | 0.18 | 0.16 | 0.0 | 0.79 | 0.108 |
| FS_movement | 131 | 69% | 0.48 | 585 | 0.15 | 0.14 | 0.0 | 0.78 | 0.140 |
| NE | 200 | 0% | 0.00 | 0 | 0.00 | 0.00 | 0.0 | 1.00 | 0.000 |
| DT_static | 186 | 71% | 0.57 | 1546 | 0.31 | 0.30 | 0.0 | 0.95 | 0.251 |
| DT_movement | 114 | 63% | 0.51 | 1464 | 0.19 | 0.19 | 0.0 | 0.82 | 0.231 |

### DT_movement worst-20 vs best-20

worst-20 mean deg = 1.397, best-20 mean deg = 3.312

| field | worst-20 | best-20 | Δ | separating? |
|---|---:|---:|---:|---|
| epc_pct | 0.573 | 0.354 | +0.220 | YES |
| epc_max_run | 1638.700 | 932.100 | +706.600 | YES |
| pmax_pct | 0.185 | 0.197 | -0.012 | no |
| pfloor_pct | 0.184 | 0.188 | -0.004 | no |
| delay_first | 0.000 | 0.000 | +0.000 | no |
| delay_shift | 0.000 | 0.000 | +0.000 | no |
| epv | 1.400 | 0.400 | +1.000 | YES |
| shadow_rise | 7.900 | 4.700 | +3.200 | YES |
| transitions | 9.350 | 5.100 | +4.250 | YES |
| init_pct | 0.400 | 0.612 | -0.213 | YES |
| once_conv | 0.700 | 0.600 | +0.100 | weak |
| pre_epc_dt_mean | 0.246 | 0.241 | +0.005 | no |
| pre_epc_dt_p90 | 0.372 | 0.330 | +0.042 | weak |
| pre_epc_dt_count | 2185.700 | 1261.650 | +924.050 | YES |
| scale_med | 1.010 | 19.924 | -18.914 | YES |
| scale_mean | 153.458 | 140.212 | +13.245 | no |
| scale_p10 | 0.250 | 0.967 | -0.717 | YES |

### DT_static worst-20 vs best-20

worst-20 mean deg = 1.337, best-20 mean deg = 3.454

| field | worst-20 | best-20 | Δ | separating? |
|---|---:|---:|---:|---|
| epc_pct | 0.615 | 0.396 | +0.219 | YES |
| epc_max_run | 1690.950 | 999.600 | +691.350 | YES |
| pmax_pct | 0.321 | 0.328 | -0.008 | no |
| pfloor_pct | 0.319 | 0.308 | +0.011 | no |
| delay_first | 0.000 | 0.000 | +0.000 | no |
| delay_shift | 0.000 | 0.000 | +0.000 | no |
| epv | 1.650 | 0.850 | +0.800 | YES |
| shadow_rise | 9.900 | 8.250 | +1.650 | weak |
| transitions | 11.600 | 9.150 | +2.450 | weak |
| init_pct | 0.307 | 0.566 | -0.259 | YES |
| once_conv | 0.800 | 0.550 | +0.250 | YES |
| pre_epc_dt_mean | 0.272 | 0.219 | +0.053 | weak |
| pre_epc_dt_p90 | 0.420 | 0.276 | +0.145 | YES |
| pre_epc_dt_count | 2414.950 | 1489.550 | +925.400 | YES |
| scale_med | 2.415 | 5.828 | -3.414 | YES |
| scale_mean | 109.012 | 212.690 | -103.678 | YES |
| scale_p10 | 0.517 | 0.941 | -0.425 | YES |

### FS_movement worst-20 vs best-20

worst-20 mean echo = 2.581, best-20 mean echo = 4.476

| field | worst-20 | best-20 | Δ | separating? |
|---|---:|---:|---:|---|
| epc_pct | 0.234 | 0.582 | -0.347 | YES |
| epc_max_run | 427.900 | 813.000 | -385.100 | YES |
| pmax_pct | 0.078 | 0.135 | -0.057 | YES |
| pfloor_pct | 0.063 | 0.130 | -0.066 | YES |
| delay_first | 0.000 | 0.000 | +0.000 | no |
| delay_shift | 0.000 | 0.000 | +0.000 | no |
| epv | 0.200 | 1.000 | -0.800 | YES |
| shadow_rise | 3.000 | 5.450 | -2.450 | YES |
| transitions | 3.200 | 6.500 | -3.300 | YES |
| init_pct | 0.742 | 0.357 | +0.385 | YES |
| once_conv | 0.400 | 0.800 | -0.400 | YES |
| pre_epc_dt_mean | 0.117 | 0.157 | -0.040 | weak |
| pre_epc_dt_p90 | 0.211 | 0.384 | -0.173 | YES |
| pre_epc_dt_count | 575.200 | 1349.600 | -774.400 | YES |
| scale_med | 13.227 | 1.268 | +11.959 | YES |
| scale_mean | 117.051 | 23.667 | +93.384 | YES |
| scale_p10 | 2.450 | 0.312 | +2.138 | YES |

### FS_static worst-20 vs best-20

worst-20 mean echo = 2.220, best-20 mean echo = 4.558

| field | worst-20 | best-20 | Δ | separating? |
|---|---:|---:|---:|---|
| epc_pct | 0.108 | 0.395 | -0.287 | YES |
| epc_max_run | 132.000 | 592.950 | -460.950 | YES |
| pmax_pct | 0.076 | 0.149 | -0.073 | YES |
| pfloor_pct | 0.050 | 0.144 | -0.095 | YES |
| delay_first | 0.000 | 0.000 | +0.000 | no |
| delay_shift | 0.000 | 0.000 | +0.000 | no |
| epv | 0.050 | 1.500 | -1.450 | YES |
| shadow_rise | 3.000 | 6.850 | -3.850 | YES |
| transitions | 3.150 | 8.650 | -5.500 | YES |
| init_pct | 0.864 | 0.415 | +0.449 | YES |
| once_conv | 0.300 | 0.750 | -0.450 | YES |
| pre_epc_dt_mean | 0.090 | 0.132 | -0.042 | YES |
| pre_epc_dt_p90 | 0.171 | 0.342 | -0.171 | YES |
| pre_epc_dt_count | 237.900 | 958.500 | -720.600 | YES |
| scale_med | 30.433 | 1.059 | +29.373 | YES |
| scale_mean | 110.930 | 33.420 | +77.510 | YES |
| scale_p10 | 3.363 | 0.291 | +3.072 | YES |

## Hypothesis tests

### D1 + D2 (DT_movement worst-20 specifics)

- mean delay_shift count: 0.0
- mean epc_pct: 57.3%
- mean epc_max_run: 1639 frames (16.4s)
- pfloor active fraction across all DT_mv: 18.53%
- median delay_shift count across DT_mv: 0
- DT_mv cases where delay_shift > 0: 0/114

### D3 (pre-EPC DT signal)

- DT_movement: mean pre_epc_dt = 0.231, count of cases with pre_epc_dt_mean > 0.3: 62/114
- DT_static : mean pre_epc_dt = 0.251
- FS_static  : mean pre_epc_dt = 0.108 (sanity: should be near 0)

### M1 (filter scale ratio)

- DT_movement worst-20 median scale: 0.80
- DT_movement best-20  median scale: 0.76
- FS_static median scale: 0.79 (reference)

Interpretation: scale_ratio = filter_echo_psd_mean / (far_psd_mean × erl_estimate). If <<1 → filter underestimates echo magnitude (misadjustment).
