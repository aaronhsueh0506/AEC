# Phase 0 — AEC3-state population analysis

Cases analyzed: 800 (baseline_v381 baseline)

## Per-bucket state means

| Bucket | n | once_conv% | init_pct | dominant_pct | usable_v1_pct | usable_v2_pct | transitions | reset_2_pct |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| FS_static | 169 | 64% | 0.50 | 0.04 | 0.10 | 0.09 | 9.9 | 0.39 |
| FS_movement | 131 | 69% | 0.46 | 0.06 | 0.06 | 0.06 | 8.3 | 0.47 |
| NE | 200 | 0% | 1.00 | 0.00 | 0.00 | 0.00 | 0.0 | 0.00 |
| DT_static | 186 | 71% | 0.37 | 0.16 | 0.06 | 0.06 | 10.2 | 0.57 |
| DT_movement | 114 | 63% | 0.46 | 0.14 | 0.03 | 0.03 | 6.8 | 0.51 |

## Worst-DT vs random — do states separate populations?

Sort each bucket by deg ascending. Compare top-20 (worst) vs bottom-20 (best) on each state field.

### DT_movement

worst-20 mean deg = 1.408, best-20 mean deg = 3.313

| field | worst-20 | best-20 | Δ (worst-best) | separating? |
|---|---:|---:|---:|---|
| init_pct | 0.359 | 0.612 | -0.254 | yes |
| dominant_pct | 0.139 | 0.124 | +0.015 | no |
| usable_v1_pct | 0.028 | 0.034 | -0.006 | no |
| usable_v2_pct | 0.027 | 0.033 | -0.006 | no |
| transitions | 9.400 | 5.100 | +4.300 | yes |
| erl_final | 0.347 | 0.343 | +0.004 | no |
| dt_from_zero_count | 670.850 | 638.400 | +32.450 | no |

### DT_static

worst-20 mean deg = 1.342, best-20 mean deg = 3.454

| field | worst-20 | best-20 | Δ (worst-best) | separating? |
|---|---:|---:|---:|---|
| init_pct | 0.307 | 0.566 | -0.259 | yes |
| dominant_pct | 0.165 | 0.133 | +0.033 | no |
| usable_v1_pct | 0.079 | 0.038 | +0.041 | yes |
| usable_v2_pct | 0.078 | 0.037 | +0.041 | yes |
| transitions | 11.600 | 9.150 | +2.450 | yes |
| erl_final | 0.308 | 0.339 | -0.031 | no |
| dt_from_zero_count | 197.250 | 585.450 | -388.200 | yes |

## FS regression-risk cases — low baseline echo

Sort FS_static + FS_movement by echo ascending. Top-20 (worst FS echo) vs bottom-20.

worst-20 echo = 2.085, best-20 echo = 4.589

| field | worst-20 | best-20 | Δ |
|---|---:|---:|---:|
| init_pct | 0.957 | 0.372 | +0.585 |
| dominant_pct | 0.008 | 0.098 | -0.090 |
| usable_v1_pct | 0.007 | 0.131 | -0.124 |
| usable_v2_pct | 0.007 | 0.130 | -0.123 |
| transitions | 0.550 | 8.000 | -7.450 |
| erl_final | 0.263 | 0.225 | +0.038 |
| dt_from_zero_count | 294.600 | 169.350 | +125.250 |

## Filter once_converged rate per bucket

| bucket | n | once_conv | not_once_conv |
|---|---:|---:|---:|
| FS_static | 169 | 109 (64%) | 60 (36%) |
| FS_movement | 131 | 90 (69%) | 41 (31%) |
| NE | 200 | 0 (0%) | 200 (100%) |
| DT_static | 186 | 132 (71%) | 54 (29%) |
| DT_movement | 114 | 72 (63%) | 42 (37%) |

## Phase 0 verdict

Top 3 separating fields on DT_movement worst-20 vs best-20 (by |Δ|):

- **dt_from_zero_count**: Δ = +32.450
- **transitions**: Δ = +4.300
- **init_pct**: Δ = -0.254
