# Round 7 — Phase 0 filter-trajectory analysis

Cases: 800 (baseline = baseline_v381_seeded, rank-locked by deg)
DT_movement worst-20 mean baseline_deg = 1.397
DT_movement best-20 mean baseline_deg = 3.312

Go/No-Go gate (upgraded):
- continuous: effect_size ≥ 0.5 AND |Δ| ≥ category floor (0.05 / 1 dB / 5 pp)
- event-count: worst-20 hit-rate ≥ 6/20 AND ≥ 2× best-20 hit-rate
- post-event: continuous rule + ≥ 8/20 worst-20 with non-zero sample

## Per-bucket means (sanity check)

| Bucket | n | once_conv% | shadow_adv | inst_erle_sm | nores_echo_proxy | res_gain | far_active% |
|---|---:|---:|---:|---:|---:|---:|---:|
| FS_static | 169 | 49.8 | 1.406 | 4.688 | 17.391 | 0.347 | 96.8 |
| FS_movement | 131 | 53.9 | 1.094 | 3.482 | 10.744 | 0.324 | 96.2 |
| NE | 200 | 0.0 | 0.993 | 0.983 | 0.324 | 0.917 | 16.6 |
| DT_static | 186 | 63.2 | 1.100 | 4.749 | 9.853 | 0.428 | 62.1 |
| DT_movement | 114 | 53.6 | 1.062 | 3.106 | 4.369 | 0.422 | 63.0 |

## DT_movement worst-20 vs best-20 (rank-locked by baseline_deg)

### transition events (event-count rule)

| field | worst-20 | best-20 | gate | reason |
|---|---:|---:|---|---|
| r7_event_delay_first_count | 0.000 | 0.000 | no | event-count: w_hit=0/20, b_hit=0/20, ratio=0.0x |
| r7_event_delay_shift_count | 0.000 | 0.000 | no | event-count: w_hit=0/20, b_hit=0/20, ratio=0.0x |
| r7_event_epv_count | 1.400 | 0.400 | **PASS** | event-count: w_hit=11/20, b_hit=5/20, ratio=2.2x |
| r7_event_shadow_rise_count | 7.900 | 4.700 | no | event-count: w_hit=14/20, b_hit=12/20, ratio=1.2x |

### delay dynamics

| field | worst-20 | best-20 | gate | reason |
|---|---:|---:|---|---|
| r7_delay_samples_mean | -1.000 | -1.000 | no | continuous: effect=0.00, |Δ|=0.0000 (floor=0.05) |
| r7_delay_samples_max | -1.000 | -1.000 | no | continuous: effect=0.00, |Δ|=0.0000 (floor=0.05) |
| r7_delay_delta_max_abs | 0.000 | 0.000 | no | continuous: effect=0.00, |Δ|=0.0000 (floor=0.05) |
| r7_delay_delta_count | 0.000 | 0.000 | no | event-count: w_hit=0/20, b_hit=0/20, ratio=0.0x |

### P-override / forced state (pct rule)

| field | worst-20 | best-20 | gate | reason |
|---|---:|---:|---|---|
| r7_p_max_active_pct | 0.185 | 0.197 | no | continuous: effect=0.05, |Δ|=0.0122 (floor=0.05) |
| r7_p_floor_active_pct | 0.184 | 0.188 | no | continuous: effect=0.02, |Δ|=0.0038 (floor=0.05) |
| r7_epc_force_active_pct | 0.044 | 0.028 | no | continuous: effect=0.24, |Δ|=0.0158 (floor=0.05) |

### convergence (pct rule)

| field | worst-20 | best-20 | gate | reason |
|---|---:|---:|---|---|
| r7_once_conv_pct | 0.601 | 0.388 | no | continuous: effect=0.49, |Δ|=0.2133 (floor=0.05) |

### output power (dB / continuous)

| field | worst-20 | best-20 | gate | reason |
|---|---:|---:|---|---|
| r7_far_active_pct | 0.637 | 0.631 | no | continuous: effect=0.02, |Δ|=0.0053 (floor=0.05) |
| r7_nores_pwr_db_mean | -35.807 | -32.954 | no | continuous: effect=0.09, |Δ|=2.8535 (floor=1.0) |
| r7_final_pwr_db_mean | -47.692 | -42.260 | no | continuous: effect=0.19, |Δ|=5.4323 (floor=1.0) |
| r7_nores_echo_proxy_mean | 1.013 | 1.551 | no | continuous: effect=0.01, |Δ|=0.5383 (floor=0.05) |
| r7_nores_echo_proxy_max | 11.358 | 224.306 | no | continuous: effect=0.05, |Δ|=212.9484 (floor=0.05) |
| r7_res_required_gain_mean | 0.400 | 0.475 | no | continuous: effect=0.28, |Δ|=0.0749 (floor=0.05) |

### adaptation signals (continuous)

| field | worst-20 | best-20 | gate | reason |
|---|---:|---:|---|---|
| r7_shadow_advantage_mean | 0.944 | 1.311 | no | continuous: effect=0.32, |Δ|=0.3677 (floor=0.05) |
| r7_inst_erle_smooth_mean | 3.608 | 2.747 | no | continuous: effect=0.18, |Δ|=0.8604 (floor=0.05) |
| r7_main_err_smooth_mean | 176.447 | 647.287 | **PASS** | continuous: effect=1.05, |Δ|=470.8394 (floor=0.05) |
| r7_shadow_err_smooth_mean | 187.812 | 583.949 | **PASS** | continuous: effect=0.92, |Δ|=396.1379 (floor=0.05) |
| r7_filter_w_norm_mean | 7.160 | 17.826 | **PASS** | continuous: effect=0.98, |Δ|=10.6655 (floor=0.05) |
| r7_shadow_w_norm_mean | 8.801 | 30.785 | **PASS** | continuous: effect=1.21, |Δ|=21.9837 (floor=0.05) |
| r7_mu_scale_mean | 0.795 | 0.716 | **PASS** | continuous: effect=0.65, |Δ|=0.0795 (floor=0.05) |

### post-event recovery (continuous + sample-count guard)

| field | worst-20 | best-20 | gate | reason |
|---|---:|---:|---|---|
| r7_post_delay_first_inst_erle_mean | 0.000 | 0.000 | no | post-event: only 0/20 worst cases have samples |
| r7_post_delay_shift_inst_erle_mean | 0.000 | 0.000 | no | post-event: only 0/20 worst cases have samples |
| r7_post_epv_inst_erle_mean | 5.704 | 1.939 | no | continuous: effect=0.42, |Δ|=3.7650 (floor=0.05) |
| r7_post_shadow_rise_inst_erle_mean | 6.157 | 4.273 | no | continuous: effect=0.21, |Δ|=1.8843 (floor=0.05) |

## Phase 0 Go/No-Go decision

PASS-gate fields: **6**

| field | worst-20 | best-20 | reason |
|---|---:|---:|---|
| r7_event_epv_count | 1.400 | 0.400 | event-count: w_hit=11/20, b_hit=5/20, ratio=2.2x |
| r7_main_err_smooth_mean | 176.447 | 647.287 | continuous: effect=1.05, |Δ|=470.8394 (floor=0.05) |
| r7_shadow_err_smooth_mean | 187.812 | 583.949 | continuous: effect=0.92, |Δ|=396.1379 (floor=0.05) |
| r7_filter_w_norm_mean | 7.160 | 17.826 | continuous: effect=0.98, |Δ|=10.6655 (floor=0.05) |
| r7_shadow_w_norm_mean | 8.801 | 30.785 | continuous: effect=1.21, |Δ|=21.9837 (floor=0.05) |
| r7_mu_scale_mean | 0.795 | 0.716 | continuous: effect=0.65, |Δ|=0.0795 (floor=0.05) |

**Decision: GO** — proceed to Phase 0.5 signal split + Phase 1.
Top separators above point to which transition / signal type is binding.

## Worst-20 DT_mv per-case detail (top separators)

| stem | base_deg | event_epv_count | main_err_smooth_mean | shadow_err_smooth_mean | filter_w_norm_mean | shadow_w_norm_mean |
|---|---:|---:|---:|---:|---:|---:|
| XV5L2dn3S06M9GBEu1q3DA_doublet | 1.16 | 0.000 | 32.110 | 36.528 | 4.782 | 5.600 |
| wHmBm7VHfkysBOhjoAXkNA_doublet | 1.26 | 3.000 | 76.992 | 85.114 | 4.283 | 6.031 |
| WqEYNwalSUebZxaeYVay2g_doublet | 1.30 | 0.000 | 779.785 | 826.075 | 2.414 | 3.943 |
| xofDX004bkqiOv9YOxmGVQ_doublet | 1.31 | 3.000 | 3.263 | 4.286 | 5.241 | 5.696 |
| XuiheB7eUkyJA2XzFIovHQ_doublet | 1.32 | 0.000 | 76.579 | 89.739 | 10.771 | 11.552 |
| VgSXlJJEI02dytkMm5UTzA_doublet | 1.36 | 0.000 | 759.315 | 754.767 | 20.757 | 25.814 |
| I2bme08keUmAnyJRKNYDGQ_doublet | 1.38 | 2.000 | 27.684 | 32.055 | 6.864 | 8.112 |
| QkRkwwFKVEar0WtcuvJsZg_doublet | 1.39 | 3.000 | 323.184 | 344.895 | 8.217 | 8.938 |
| w0QrMwsZ5kGoJjRWvP0iKg_doublet | 1.39 | 1.000 | 80.074 | 77.054 | 7.352 | 8.882 |
| Xvyiz1o0cEijZQ8DT9mB2w_doublet | 1.39 | 0.000 | 213.833 | 233.854 | 3.889 | 5.950 |
| ql7yTcebJU20VE5qpW0kCA_doublet | 1.40 | 0.000 | 128.799 | 138.869 | 11.492 | 13.575 |
| X7Ua9txMj0aws848JPEbOg_doublet | 1.40 | 0.000 | 6.088 | 6.569 | 1.765 | 2.480 |
| WcK0OrF6ukW03fViPXTQjQ_doublet | 1.42 | 1.000 | 95.288 | 106.669 | 5.406 | 7.242 |
| nyT6FUUdu0W8UpvjP1rRgQ_doublet | 1.43 | 1.000 | 113.427 | 115.596 | 10.791 | 13.138 |
| nlSSRl4k50Gq2mIRYlMBCg_doublet | 1.46 | 3.000 | 39.311 | 43.167 | 8.616 | 9.962 |
| V0JqgjlrB0Ke9y91r0rxNw_doublet | 1.47 | 0.000 | 65.150 | 77.790 | 1.532 | 2.008 |
| sKXucFp4FUCJKo5d0G54Og_doublet | 1.50 | 3.000 | 311.360 | 326.411 | 9.180 | 11.518 |
| SwfEwuGDlkWYy9pb4H00eQ_doublet | 1.53 | 3.000 | 144.483 | 176.356 | 8.591 | 9.389 |
| w5XDRNfB2Ei2UoUDtrTkzg_doublet | 1.54 | 0.000 | 147.406 | 149.685 | 4.649 | 9.022 |
| zOiK6oSHp0ib3nHvzLKbRQ_doublet | 1.55 | 5.000 | 104.815 | 130.752 | 6.610 | 7.168 |
