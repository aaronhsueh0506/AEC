# v3.21 M_full_delay Trace (M_D vs M_full_delay)

**Date**: 2026-05-26
**M_D** = M_A_ceil2 (Bundle B+C+A + use_aec3_h_error_ceil=True, use_full_delay_change_chain=False)
**M_full_delay** = M_D + use_full_delay_change_chain=True
**No AECMOS / no 800-case / no verdict / no version bump**

> Key mechanism — two distinct AEC3 HandleEchoPathChange components:
>
> (1) filter/shadow handle_echo_path_change() → AdaptiveFirFilter::ZeroFilter(current_size, max_size):
>     Zeroes ONLY tail partitions in [current..max). AEC3 default: max=13, initial=12.
>     · Steady state (current=13, max=13): current==max → NO-OP. W is preserved, not zeroed.
>     · Initial stage (current=12, max=13): zeroes partition 12 (tail) only.
>     Python does W.fill(0) on all partitions — more aggressive, not AEC3-equivalent.
>
> (2) refined_gains.HandleEchoPathChange() → H_error = 10000, poor_excit = 1000 (separate call).
>
> M_D: neither call runs on delay_first (only gate2 reset via _reset_aec3_post). H_error preserved.
> M_full_delay: both (1) + (2) run at delay_first. H_error 10000 clipped to 2.0 (use_aec3_h_error_ceil=True).

## §1 Delay Events (all cases)

| Case | events | frame | type | M_D event count | M_full_delay event count |
|------|--------|-------|------|-----------------|--------------------------|
| 9xjhiFbG_FS_static | 1 | 69 | delay_first | 1 | 1 |
| xFk7igec_DT_mvmt | 1 | 104 | delay_first | 1 | 1 |
| nVUnxqHL_DT_static | 1 | 119 | delay_first | 1 | 1 |
| XRTnTUjU_DT_static | 1 | 109 | delay_first | 1 | 1 |
| MYrVxVEM_DT_static | 1 | 85 | delay_first | 1 | 1 |
| wVYSGVTT_DT_mvmt | 1 | 75 | delay_first | 1 | 1 |
| qNvSMyUS_FS_static | 1 | 189 | delay_first | 1 | 1 |
| jtYTdZm3_DT_static | 1 | 246 | delay_first | 1 | 1 |

## §2 usable_linear Delta (M_full_delay vs M_D)

| Case | M_D UL | M_full UL | Δpp | M_D g2 | M_full g2 | initial_state (M_full) | a1_on |
|------|--------|-----------|-----|--------|-----------|------------------------|-------|
| 9xjhiFbG_FS_static | 92.4% | 92.4% | +0.0pp | 93.0% | 93.0% | 35.4% | 349 |
| xFk7igec_DT_mvmt | 96.1% | 96.1% | +0.0pp | 96.2% | 96.2% | 10.7% | 291 |
| nVUnxqHL_DT_static | 95.9% | 95.9% | +0.0pp | 96.0% | 96.0% | 12.5% | 418 |
| XRTnTUjU_DT_static | 91.2% | 91.2% | +0.0pp | 91.2% | 91.2% | 75.0% | 489 |
| MYrVxVEM_DT_static | 95.6% | 95.6% | +0.0pp | 95.6% | 95.6% | 25.1% | 406 |
| wVYSGVTT_DT_mvmt | 96.3% | 49.5% | -46.9pp | 96.3% | 49.5% | 79.6% | 379 |
| qNvSMyUS_FS_static | 82.5% | 82.5% | +0.0pp | 85.7% | 85.7% | 45.9% | 629 |
| jtYTdZm3_DT_static | 97.5% | 97.5% | +0.0pp | 97.8% | 97.8% | 13.5% | 250 |

## §3 H_error Window Around delay_first (detail stems)

> rel=0 is the delay_first frame; pre=5 / post=50 frames captured.
> h_error_mean: mean of H_error_per_bin; aec3_IS: initial_state_active;
> g2: ul_gate2_reset; UL: usable_linear; a1_SI: set_initial_state event

### 9xjhiFbG_FS_static

| rel | M_D h_err | M_full h_err | M_D g2 | M_full g2 | M_full IS | M_full a1_SI |
|-----|-----------|-------------|--------|-----------|-----------|--------------|
| -5 | 0.7159 | 0.7159 | T | T | T | — |
| -4 | 0.7103 | 0.7103 | T | T | T | — |
| -3 | 0.5304 | 0.5304 | T | T | T | — |
| -2 | 0.4910 | 0.4910 | T | T | T | — |
| -1 | 0.4971 | 0.4971 | T | T | T | — |
| +0 | 0.4904 | 2.0000 | F | F | T | on |
| +1 | 0.4900 | 2.0000 | F | F | T | on |
| +2 | 0.4983 | 2.0000 | F | F | T | on |
| +3 | 0.5081 | 2.0000 | F | F | T | on |
| +4 | 0.5218 | 2.0000 | F | F | T | on |
| +5 | 0.5276 | 2.0000 | F | F | T | on |
| +6 | 0.5394 | 1.9605 | F | F | T | on |
| +7 | 0.5404 | 1.8804 | F | F | T | on |
| +8 | 0.5473 | 1.8070 | F | F | T | on |
| +9 | 0.5565 | 1.7479 | F | F | T | on |
| +10 | 0.5647 | 1.7166 | F | F | T | on |
| +11 | 0.5813 | 1.7170 | F | F | T | on |
| +12 | 0.5980 | 1.7196 | F | F | T | on |
| +13 | 0.6081 | 1.7070 | F | F | T | on |
| +14 | 0.6219 | 1.6954 | F | F | T | on |
| +15 | 0.6172 | 1.6287 | F | F | T | on |
| +16 | 0.6280 | 1.6283 | F | F | T | on |
| +17 | 0.6408 | 1.6128 | F | F | T | on |
| +18 | 0.6471 | 1.5794 | F | F | T | on |
| +19 | 0.6395 | 1.5108 | F | F | T | on |
| +20 | 0.6370 | 1.4379 | F | F | T | on |
| +21 | 0.6338 | 1.3780 | F | F | T | on |
| +22 | 0.6255 | 1.3102 | F | F | T | on |
| +23 | 0.6076 | 1.2226 | F | F | T | on |
| +24 | 0.5846 | 1.1382 | F | F | T | on |
| +25 | 0.5504 | 1.0326 | F | F | T | on |
| +26 | 0.5515 | 1.0114 | F | F | T | on |
| +27 | 0.5614 | 1.0056 | F | F | T | on |
| +28 | 0.5722 | 1.0113 | F | F | T | on |
| +29 | 0.5820 | 1.0204 | F | F | T | on |
| +30 | 0.5823 | 1.0140 | F | F | T | on |
| +31 | 0.5857 | 1.0166 | F | F | T | on |
| +32 | 0.5874 | 1.0093 | F | F | T | on |
| +33 | 0.5894 | 1.0068 | F | F | T | on |
| +34 | 0.5928 | 1.0076 | F | F | T | on |
| +35 | 0.5928 | 1.0076 | F | F | T | on |
| +36 | 0.5928 | 1.0076 | F | F | T | on |
| +37 | 0.5928 | 1.0076 | F | F | T | on |
| +38 | 0.5928 | 1.0076 | F | F | T | on |
| +39 | 0.5928 | 1.0076 | F | F | T | on |
| +40 | 0.5928 | 1.0076 | F | F | T | on |
| +41 | 0.5928 | 1.0076 | F | F | T | on |
| +42 | 0.5928 | 1.0076 | F | F | T | on |
| +43 | 0.5928 | 1.0076 | F | F | T | on |
| +44 | 0.5928 | 1.0076 | F | F | T | on |
| +45 | 0.6129 | 1.0267 | F | F | T | on |
| +46 | 0.6337 | 1.0463 | F | F | T | on |
| +47 | 0.6524 | 1.0631 | F | F | T | on |
| +48 | 0.6660 | 1.0758 | F | F | T | on |
| +49 | 0.6709 | 1.0706 | F | F | T | on |
| +50 | 0.6659 | 1.0467 | T | T | T | on |

### xFk7igec_DT_mvmt

| rel | M_D h_err | M_full h_err | M_D g2 | M_full g2 | M_full IS | M_full a1_SI |
|-----|-----------|-------------|--------|-----------|-----------|--------------|
| -5 | 0.2178 | 0.2178 | T | T | T | — |
| -4 | 0.2236 | 0.2236 | T | T | T | — |
| -3 | 0.1991 | 0.1991 | T | T | T | — |
| -2 | 0.2099 | 0.2099 | T | T | T | — |
| -1 | 0.2299 | 0.2299 | T | T | T | — |
| +0 | 0.2331 | 2.0000 | F | F | T | on |
| +1 | 0.2249 | 2.0000 | F | F | T | on |
| +2 | 0.1474 | 2.0000 | F | F | T | on |
| +3 | 0.1430 | 2.0000 | F | F | T | on |
| +4 | 0.1306 | 2.0000 | F | F | T | on |
| +5 | 0.1201 | 2.0000 | F | F | T | on |
| +6 | 0.1378 | 1.5919 | F | F | T | on |
| +7 | 0.1509 | 1.0935 | F | F | T | on |
| +8 | 0.1491 | 0.8564 | F | F | T | on |
| +9 | 0.1227 | 0.5654 | F | F | T | on |
| +10 | 0.1258 | 0.5715 | F | F | T | on |
| +11 | 0.1426 | 0.5835 | F | F | T | on |
| +12 | 0.1583 | 0.5933 | F | F | T | on |
| +13 | 0.1758 | 0.6105 | F | F | T | on |
| +14 | 0.1905 | 0.6197 | F | F | T | on |
| +15 | 0.1967 | 0.4432 | F | F | T | on |
| +16 | 0.1959 | 0.3790 | F | F | T | on |
| +17 | 0.1530 | 0.2865 | F | F | T | on |
| +18 | 0.1470 | 0.2628 | F | F | T | on |
| +19 | 0.1224 | 0.2342 | F | F | T | on |
| +20 | 0.1299 | 0.2203 | F | F | T | on |
| +21 | 0.1446 | 0.2033 | F | F | T | on |
| +22 | 0.1603 | 0.1958 | F | F | T | on |
| +23 | 0.1704 | 0.2039 | F | F | T | on |
| +24 | 0.1523 | 0.1802 | F | F | T | on |
| +25 | 0.1549 | 0.1904 | F | F | T | on |
| +26 | 0.1644 | 0.1834 | F | F | T | on |
| +27 | 0.1465 | 0.1619 | F | F | T | on |
| +28 | 0.1596 | 0.1786 | F | F | T | on |
| +29 | 0.1600 | 0.1804 | F | F | T | on |
| +30 | 0.1680 | 0.1746 | F | F | T | on |
| +31 | 0.1888 | 0.1937 | F | F | T | on |
| +32 | 0.2089 | 0.2080 | F | F | T | on |
| +33 | 0.2247 | 0.2235 | F | F | T | on |
| +34 | 0.2156 | 0.2026 | F | F | T | on |
| +35 | 0.2256 | 0.2190 | F | F | T | on |
| +36 | 0.2080 | 0.2185 | F | F | T | on |
| +37 | 0.2061 | 0.2324 | F | F | T | on |
| +38 | 0.2270 | 0.2494 | F | F | T | on |
| +39 | 0.2442 | 0.2657 | F | F | T | on |
| +40 | 0.2560 | 0.2738 | T | T | T | on |
| +41 | 0.2724 | 0.2819 | T | T | T | on |
| +42 | 0.2725 | 0.2858 | T | T | T | on |
| +43 | 0.2724 | 0.2907 | T | T | T | on |
| +44 | 0.2848 | 0.2834 | T | T | T | on |
| +45 | 0.3007 | 0.2947 | T | T | T | on |
| +46 | 0.3012 | 0.3025 | T | T | T | on |
| +47 | 0.2997 | 0.3075 | T | T | T | on |
| +48 | 0.2988 | 0.3168 | T | T | T | on |
| +49 | 0.3075 | 0.3265 | T | T | T | on |
| +50 | 0.3136 | 0.3251 | T | T | T | on |

### XRTnTUjU_DT_static

| rel | M_D h_err | M_full h_err | M_D g2 | M_full g2 | M_full IS | M_full a1_SI |
|-----|-----------|-------------|--------|-----------|-----------|--------------|
| -5 | 0.1208 | 0.1208 | F | F | T | — |
| -4 | 0.1244 | 0.1244 | F | F | T | — |
| -3 | 0.1283 | 0.1283 | F | F | T | — |
| -2 | 0.1322 | 0.1322 | F | F | T | — |
| -1 | 0.1383 | 0.1383 | F | F | T | — |
| +0 | 0.1435 | 2.0000 | F | F | T | on |
| +1 | 0.1483 | 2.0000 | F | F | T | on |
| +2 | 0.1580 | 2.0000 | F | F | T | on |
| +3 | 0.1736 | 2.0000 | F | F | T | on |
| +4 | 0.1749 | 2.0000 | F | F | T | on |
| +5 | 0.1811 | 2.0000 | F | F | T | on |
| +6 | 0.1913 | 1.5739 | F | F | T | on |
| +7 | 0.1983 | 1.3458 | F | F | T | on |
| +8 | 0.2067 | 1.1993 | F | F | T | on |
| +9 | 0.2119 | 1.1476 | F | F | T | on |
| +10 | 0.2167 | 1.0119 | F | F | T | on |
| +11 | 0.2351 | 1.0110 | F | F | T | on |
| +12 | 0.2576 | 1.0287 | F | F | T | on |
| +13 | 0.2576 | 1.0287 | F | F | T | on |
| +14 | 0.2709 | 0.9542 | F | F | T | on |
| +15 | 0.2738 | 0.7968 | F | F | T | on |
| +16 | 0.2746 | 0.6822 | F | F | T | on |
| +17 | 0.2711 | 0.5703 | F | F | T | on |
| +18 | 0.2723 | 0.4957 | F | F | T | on |
| +19 | 0.2717 | 0.4477 | F | F | T | on |
| +20 | 0.2599 | 0.4150 | F | F | T | on |
| +21 | 0.2685 | 0.4012 | F | F | T | on |
| +22 | 0.2699 | 0.3858 | F | F | T | on |
| +23 | 0.2745 | 0.3806 | F | F | T | on |
| +24 | 0.2864 | 0.3889 | F | F | T | on |
| +25 | 0.2986 | 0.3961 | F | F | T | on |
| +26 | 0.3147 | 0.4112 | F | F | T | on |
| +27 | 0.3281 | 0.4257 | F | F | T | on |
| +28 | 0.3435 | 0.4369 | F | F | T | on |
| +29 | 0.3543 | 0.4439 | F | F | T | on |
| +30 | 0.3543 | 0.4439 | F | F | T | on |
| +31 | 0.3543 | 0.4439 | F | F | T | on |
| +32 | 0.3543 | 0.4439 | F | F | T | on |
| +33 | 0.3543 | 0.4439 | F | F | T | on |
| +34 | 0.3543 | 0.4439 | F | F | T | on |
| +35 | 0.3543 | 0.4439 | F | F | T | on |
| +36 | 0.3543 | 0.4439 | F | F | T | on |
| +37 | 0.3543 | 0.4439 | F | F | T | on |
| +38 | 0.3543 | 0.4439 | F | F | T | on |
| +39 | 0.3543 | 0.4439 | F | F | T | on |
| +40 | 0.3543 | 0.4439 | F | F | T | on |
| +41 | 0.3543 | 0.4439 | F | F | T | on |
| +42 | 0.3543 | 0.4439 | F | F | T | on |
| +43 | 0.3543 | 0.4439 | F | F | T | on |
| +44 | 0.3543 | 0.4439 | F | F | T | on |
| +45 | 0.3543 | 0.4439 | F | F | T | on |
| +46 | 0.3543 | 0.4439 | F | F | T | on |
| +47 | 0.3543 | 0.4439 | F | F | T | on |
| +48 | 0.3543 | 0.4439 | F | F | T | on |
| +49 | 0.3543 | 0.4439 | F | F | T | on |
| +50 | 0.3543 | 0.4439 | F | F | T | on |

### wVYSGVTT_DT_mvmt

| rel | M_D h_err | M_full h_err | M_D g2 | M_full g2 | M_full IS | M_full a1_SI |
|-----|-----------|-------------|--------|-----------|-----------|--------------|
| -5 | 0.4023 | 0.4023 | F | F | T | — |
| -4 | 0.3716 | 0.3716 | F | F | T | — |
| -3 | 0.3607 | 0.3607 | F | F | T | — |
| -2 | 0.3817 | 0.3817 | F | F | T | — |
| -1 | 0.3937 | 0.3937 | F | F | T | — |
| +0 | 0.4006 | 2.0000 | F | F | T | on |
| +1 | 0.4147 | 2.0000 | F | F | T | on |
| +2 | 0.4256 | 2.0000 | F | F | T | on |
| +3 | 0.4332 | 2.0000 | F | F | T | on |
| +4 | 0.4391 | 2.0000 | F | F | T | on |
| +5 | 0.4490 | 2.0000 | F | F | T | on |
| +6 | 0.4488 | 1.8549 | F | F | T | on |
| +7 | 0.4661 | 1.7907 | F | F | T | on |
| +8 | 0.4793 | 1.7058 | F | F | T | on |
| +9 | 0.4925 | 1.6557 | F | F | T | on |
| +10 | 0.4892 | 1.4999 | F | F | T | on |
| +11 | 0.5023 | 1.4624 | F | F | T | on |
| +12 | 0.4902 | 1.2390 | F | F | T | on |
| +13 | 0.5026 | 1.2095 | F | F | T | on |
| +14 | 0.5121 | 1.1760 | F | F | T | on |
| +15 | 0.5126 | 1.1111 | F | F | T | on |
| +16 | 0.5119 | 1.0744 | F | F | T | on |
| +17 | 0.5151 | 1.0478 | F | F | T | on |
| +18 | 0.5263 | 1.0239 | F | F | T | on |
| +19 | 0.5418 | 1.0168 | F | F | T | on |
| +20 | 0.5550 | 1.0201 | F | F | T | on |
| +21 | 0.5671 | 1.0167 | F | F | T | on |
| +22 | 0.5737 | 1.0078 | F | F | T | on |
| +23 | 0.5837 | 0.9915 | F | F | T | on |
| +24 | 0.5838 | 0.9893 | F | F | T | on |
| +25 | 0.5849 | 1.0072 | F | F | T | on |
| +26 | 0.5849 | 1.0072 | F | F | T | on |
| +27 | 0.5849 | 1.0072 | F | F | T | on |
| +28 | 0.5849 | 1.0072 | F | F | T | on |
| +29 | 0.5849 | 1.0072 | F | F | T | on |
| +30 | 0.5849 | 1.0072 | F | F | T | on |
| +31 | 0.5849 | 1.0072 | F | F | T | on |
| +32 | 0.5849 | 1.0072 | F | F | T | on |
| +33 | 0.5849 | 1.0072 | F | F | T | on |
| +34 | 0.5849 | 1.0072 | F | F | T | on |
| +35 | 0.5849 | 1.0072 | F | F | T | on |
| +36 | 0.5849 | 1.0072 | F | F | T | on |
| +37 | 0.5849 | 1.0072 | F | F | T | on |
| +38 | 0.5849 | 1.0072 | F | F | T | on |
| +39 | 0.5849 | 1.0072 | F | F | T | on |
| +40 | 0.5849 | 1.0072 | F | F | T | on |
| +41 | 0.5849 | 1.0072 | F | F | T | on |
| +42 | 0.5849 | 1.0072 | F | F | T | on |
| +43 | 0.5849 | 1.0072 | F | F | T | on |
| +44 | 0.5849 | 1.0072 | F | F | T | on |
| +45 | 0.5849 | 1.0072 | F | F | T | on |
| +46 | 0.6092 | 1.0319 | F | F | T | on |
| +47 | 0.6318 | 1.0519 | F | F | T | on |
| +48 | 0.6369 | 1.0204 | F | F | T | on |
| +49 | 0.6241 | 0.9490 | F | F | T | on |
| +50 | 0.6236 | 0.9452 | F | F | T | on |

## §4 r2 / Gain Delta (M_full_delay vs M_D)

| Case | Δr2_lf | Δr2_mf | Δr2_hf | Δgain_lf | Δgain_mf | Δgain_hf |
|------|--------|--------|--------|----------|----------|----------|
| 9xjhiFbG_FS_static | 1.008× | 1.089× | 1.102× | -0.0025 | -0.0073 | -0.0282 |
| xFk7igec_DT_mvmt | 2.551× | 1.480× | 1.929× | -0.0370 | -0.0102 | -0.0187 |
| nVUnxqHL_DT_static | 0.925× | 1.056× | 0.957× | -0.0151 | +0.0025 | -0.0013 |
| XRTnTUjU_DT_static | 1.300× | 1.224× | 1.006× | -0.0423 | -0.0071 | +0.0004 |
| MYrVxVEM_DT_static | 0.843× | 1.074× | 0.789× | -0.0002 | -0.0084 | -0.0056 |
| wVYSGVTT_DT_mvmt | 0.801× | 0.775× | 0.267× | +0.0612 | +0.0829 | +0.0588 |
| qNvSMyUS_FS_static | 0.913× | 1.019× | 0.880× | +0.0134 | -0.0234 | +0.0798 |
| jtYTdZm3_DT_static | 0.782× | 0.615× | 0.939× | +0.0143 | -0.0012 | +0.0121 |

## §5 Nores Band Energy (M_full_delay)

### LF (0–500 Hz) — Δ vs M_D and Δ vs M0

| Case | M0 dB | M_D dB | M_full dB | Δ_vs_M_D | Δ_vs_M0 |
|------|-------|--------|-----------|----------|---------|
| 9xjhiFbG_FS_static | 24.80 | 18.72 | 19.06 | +0.34 | -5.74 |
| xFk7igec_DT_mvmt | 6.16 | 7.56 | 7.72 | +0.16 | +1.56 |
| nVUnxqHL_DT_static | 9.42 | 9.57 | 9.56 | -0.01 | +0.14 |
| XRTnTUjU_DT_static | 5.01 | 5.06 | 5.02 | -0.04 | +0.01 |
| MYrVxVEM_DT_static | -0.22 | 0.43 | -0.02 | -0.45 | +0.20 |
| wVYSGVTT_DT_mvmt | 16.92 | 15.88 | 16.13 | +0.25 | -0.79 |
| qNvSMyUS_FS_static | -2.35 | -2.38 | -2.41 | -0.03 | -0.05 |
| jtYTdZm3_DT_static | 13.63 | 12.12 | 12.13 | +0.01 | -1.51 |

### MF (500–2000 Hz)

| Case | M0 dB | M_D dB | M_full dB | Δ_vs_M_D | Δ_vs_M0 |
|------|-------|--------|-----------|----------|---------|
| 9xjhiFbG_FS_static | 23.41 | 25.49 | 25.46 | -0.03 | +2.05 |
| xFk7igec_DT_mvmt | 12.87 | 11.18 | 11.36 | +0.18 | -1.51 |
| nVUnxqHL_DT_static | 9.75 | 7.24 | 7.25 | +0.01 | -2.50 |
| XRTnTUjU_DT_static | 8.49 | 8.57 | 8.58 | +0.01 | +0.09 |
| MYrVxVEM_DT_static | 7.13 | 3.86 | 3.98 | +0.12 | -3.15 |
| wVYSGVTT_DT_mvmt | 20.12 | 19.66 | 20.84 | +1.18 | +0.73 |
| qNvSMyUS_FS_static | 5.21 | 5.52 | 5.48 | -0.04 | +0.28 |
| jtYTdZm3_DT_static | 10.48 | 9.04 | 8.92 | -0.12 | -1.56 |

### HF (2000–8000 Hz)

| Case | M0 dB | M_D dB | M_full dB | Δ_vs_M_D | Δ_vs_M0 |
|------|-------|--------|-----------|----------|---------|
| 9xjhiFbG_FS_static | 21.76 | 23.25 | 23.17 | -0.08 | +1.41 |
| xFk7igec_DT_mvmt | 8.18 | 0.68 | 0.69 | +0.01 | -7.49 |
| nVUnxqHL_DT_static | 0.12 | 0.53 | 0.54 | +0.01 | +0.42 |
| XRTnTUjU_DT_static | 0.66 | 0.94 | 1.04 | +0.10 | +0.37 |
| MYrVxVEM_DT_static | -4.34 | -10.08 | -9.80 | +0.28 | -5.46 |
| wVYSGVTT_DT_mvmt | 14.41 | 14.90 | 15.03 | +0.13 | +0.62 |
| qNvSMyUS_FS_static | -11.03 | -10.96 | -10.98 | -0.02 | +0.05 |
| jtYTdZm3_DT_static | 2.92 | 3.32 | 3.31 | -0.01 | +0.38 |

## §6 M_full_delay Determination

### wVYSGVTT Collapse Root Cause (CORRECTED 2026-05-26)

> **Prior version of this section incorrectly attributed resets at frames 462/1084 to
> `p3h_diverged` (PathChangeRegimeHandler). This was WRONG.**
> Full classification audit: `docs/v3_21_m_full_delay_classification_audit.md`.

**Observation**: wVYSGVTT usable_linear collapses 96.3% → 49.5% (−46.9pp) in M_full_delay.

**Confirmed reset call sites** (monkeypatched `_reset_filter_derived_state`):

| Reset # | Frame | Reason | once_conv | Source |
|---------|-------|--------|-----------|--------|
| 1 | 75 | `delay_first` | False | use_full_delay_change_chain (correct) |
| 2 | 461 | `plateau` | False | **FilterPlateauDetector** (Python-only) |
| 3 | 1083 | `plateau` | False | **FilterPlateauDetector** (2nd fire) |

**Why prior attribution was wrong**:
- `diverged_reset_enabled = False` in BALANCED config → `p3h_diverged` CANNOT fire.
- `PathChangeRegimeHandler.update()` returns `RegimeHandlerDecision` only (no divergence reset).
- Frames 462/1084 (gate2 T→F) are the frames AFTER plateau fires on 461/1083, not the cause.

**Correct mechanism**:
1. delay_first at frame 75 → `_reset_filter_derived_state(reason='delay_first')` → fresh AecState → gate1/gate2 both reset to 0.
2. M_full_delay additionally calls gains.HandleEchoPathChange() → H_error=10000 clipped to 2.0. M_full_delay also calls filter.handle_echo_path_change() → Python W.fill(0) (all partitions; more aggressive than AEC3's ZeroFilter which is a no-op in steady state).
3. H_error=2.0 → larger μ → slower ERLE build during DT movement + A.3 poor-excitation skipping updates → `convergence_seen` never latches before plateau grace window.
4. FilterPlateauDetector: grace_frames=400, eligible from ~frame 475. Conditions met (ERLE < 6 dB, far_active > 0.5, dt_signal > 0.10 for 50 consecutive) → **fires at frame 461** (within grace window because `once_conv=False`).
5. Second plateau fire at frame 1083 (2nd attempt, max_attempts=2). Recovery after 2nd reset takes 1602 hops due to intensified A.3 gating during sustained movement.
6. **M_D does NOT trigger plateau**: no H_error reset → μ stays at converged value (~0.25) → `convergence_seen` latches before frame 461 → plateau never eligible.

**Gate 0 trace results (2026-05-26)**: See §7 below.

**Diagnostic variant M_full_no_plateau** (plateau fires suppressed, all other M_full flags ON):
```
M_D:               usable_linear = 96.3%  (1 reset: delay_first at 75)
M_full_delay:      usable_linear = 49.5%  (3 resets: 75, 461, 1083)
M_full_no_plateau: usable_linear = 96.3%  (1 reset: same as M_D — exact match)
```
FilterPlateauDetector is the **sole cause** of the wVYSGVTT collapse.

**Why 7 other cases are NOT affected**:
- In all 7 cases, `convergence_seen` latches BEFORE the plateau grace window expires — plateau detector never becomes eligible even with H_error reset.
- wVYSGVTT is the only case with both: (a) prolonged sustained movement causing slow ERLE build, AND (b) A.3 poor-excitation suppression slowing gate2 increment.

### Classification (CORRECTED)

**Category B (primary): Python-only FilterPlateauDetector — not a PBFDKF architectural incompatibility.**

- FilterPlateauDetector has NO AEC3 equivalent. AEC3 NLMS shadow/main never encounters "stall" conditions that require a plateau reset.
- This is NOT "PBFDKF incompatible with AEC3 delay-change architecture." It is a Python safety net that fires due to the interaction of H_error reset + DT movement + A.3 gating.
- Fixable within v3.21.x without architectural changes (see classification_audit.md §Options Forward).

**Category A (secondary): Missing `SetConfig(initial)` and `SetSizePartitions` from AEC3 delay chain.**

- AEC3's `Subtractor::HandleEchoPathChange()` also calls `SetConfig(refined_initial, true)` (higher leakage rates) and `SetSizePartitions` (initial filter length).
- Python PBFDKF has no initial/non-initial config concept. These are real port incompleteness items.
- NOT the direct cause of the wVYSGVTT collapse (M_full_no_plateau proves plateau is sole cause).

**Supersedes prior classification**: "Class B (PBFDKF + larger-μ re-tracking incompatible with PathChangeRegimeHandler)" was based on incorrect p3h_diverged attribution and is RESCINDED.

### Nores Artifact Gate

- 9xjhi LF: M0=24.80 → M_D=18.72 → M_full=19.06 (Δ_vs_M_D=+0.34 dB). Still −5.74 dB vs M0. **Artifact closure maintained.**
- wVYSGVTT nores MF: M_D=19.66 → M_full=20.84 (Δ+1.18 dB). Caused by usable_linear collapse → RES uses near_psd instead of error_psd → less suppression.

### Verdict

**M_full_delay = CANNOT SHIP as-is.** Root cause: FilterPlateauDetector (Python-only).

- wVYSGVTT: 3 resets (2 plateau), usable_linear −46.9pp. DT audibility risk severe.
- 7/8 other cases: effectively identical to M_D (Δul = 0.0pp).
- Classification: **Category B** (Python-only FilterPlateauDetector) + **Category A** (incomplete SetConfig chain). NOT a PBFDKF architectural incompatibility.

### Implications for 12-case AECMOS Gate

M_full_delay cannot be included in 12-case AECMOS without the wVYSGVTT regression.
Options (from `docs/v3_21_m_full_delay_classification_audit.md`):

1. **Narrow fix** (`use_full_delay_plateau_suppression` flag): Suppress plateau when within delay_first grace window. Fast, v3.21.x scope.
2. **Port SetConfig(initial)**: Add initial/non-initial gain concept to PBFDKF. Closes Category A gap, may also prevent plateau. More work.
3. **Defer to v3.22** with correct classification: M_D proceeds to 12-case AECMOS; M_full_delay documented as fixable (Category B, not architecturally blocked).

**Do not proceed to 12-case AECMOS until user selects one of these options.**


---

## §7 Gate 0 Trace Results (2026-05-26)

**Script**: `python/v3_21_gate0_trace.py` (Variants C + D implemented, trace-only)
**No AECMOS. No ship/no-ship verdict.**

### Variant definitions

| Variant | Config | Classification |
|---------|--------|----------------|
| A | M_D baseline (use_full_delay_change_chain=False) | Reference |
| B | M_full_delay current (chain=True, no C/D) | Known regression |
| C | chain=True + use_full_delay_plateau_suppression=True | Python functional adaptation (NOT AEC3 port) |
| D | chain=True + use_full_delay_setconfig_initial=True | Strict AEC3 alignment port (Category A) |

### Primary case — wVYSGVTT_DT_mvmt

| Variant | usable_linear | Δvs A | plateau_fires (wall frame) | once_conv_latch | initial_state exit |
|---------|--------------|-------|---------------------------|-----------------|-------------------|
| A (M_D) | **96.3%** | — | none | 424 | 454 |
| B (M_full) | **49.5%** | −46.9pp | 462, 1084 | never | 454 |
| C (plateau suppressed via initial_state) | **52.0%** | −44.3pp | 1060, 3641 | never | 454 |
| D (SetConfig initial leakage 10×) | **49.5%** | −46.9pp | 462, 1084 | never | 454 |

H_error trajectories (filter mean):

| Frame | A | B | C | D |
|-------|---|---|---|---|
| delay_first+1 | 0.415 (no reset) | 2.000 (clipped from 10000) | 2.000 | 2.000 |
| 200 | 0.775 | 0.794 | 0.794 | 0.782 |
| 400 | 0.908 | 0.799 | 0.799 | 0.810 |
| 461 | 0.726 | 0.703 | 0.703 | 0.757 |
| h_err 10f before plateau | — | 0.614 (at 452) | 0.278 (at 1050) | 0.662 (at 452) |

Leakage info (Variant D only): leakage_converged switched from 0.0025→0.025 for 379 frames
(frames 75–454 = initial_state window), then restored to 0.0025 at frame 454.

### Gate-out check (EPV/shadow_rise)

**Confirmed NOT triggering the delay chain.** Both 9xjhi and wVYSGVTT show identical behavior
across all 4 variants for the gate-out check — the delay chain fires only on delay_first/delay_shift
path, not on EPV or shadow_rise signals.

### Secondary case — 9xjhiFbG_FS_static (nores artifact)

| Variant | usable_linear | Δvs A | plateau_fires | delay_first |
|---------|--------------|-------|--------------|-------------|
| A | 92.4% | — | 1133 | 69 |
| B | 92.4% | 0.0pp | 1133 | 69 |
| C | 92.4% | 0.0pp | 1133 | 69 |
| D | 92.4% | 0.0pp | 1133 | 69 |

**All variants identical.** The delay chain fires (delay_first at frame 69, H_error reset to 2.0),
but 9xjhi is FS (no DT contamination) so the filter converges fast and `once_conv` latches well
before plateau grace window → no regression on any variant. Nores artifact unchanged.

---

### Gate 0 Root-Cause Analysis

#### Why Variant C (initial_state gate) fails

The plateau suppression gate uses `self._aec3_state.initial_state_active()`. This exits at wall
frame **454** (driven by AEC3 FilteringQualityAnalyzer convergence detection — same for all variants).

The critical plateau fire in Variant B is at wall frame **462** — **8 frames AFTER** the
`initial_state_active` window closes.

Timeline:
```
Frame  0:  initial_state_active = True (from AEC3 InitialState start)
Frame 75:  delay_first fires; H_error → 2.0; set_initial_state(True) called
Frame 400: plateau grace expires (plateau.frame_count=400, counting from frame 0)
Frames 400–454: criteria_met but suppressed in C (initial_state_active=True)
Frame 454: initial_state_active exits (TransitionTriggered)
           Variant C suppression gate DROPS HERE
Frame 454: plateau counter = consecutive_match accumulated during 400–454 = ~54 frames
Frame 462: Variant B plateau fires (criteria met for 50+ consecutive frames)
           Variant C: suppression already expired → plateau fires too at 462? No...
```

Actually in Variant C, during frames 0–454 the `plateau_detector.update()` is NOT called (the
suppression gate skips the call entirely). So the plateau detector's internal `_frame_count` stays
at 0 throughout. When suppression lifts at frame 454, the detector starts counting from its own
frame_count=0. Grace expires at detector frame 400 = wall frame 854. Consecutive needed: 50.
Plateau fires at wall frame ~1060 (confirmed by trace: 1060).

**Consequence**: Variant C delays the first plateau from frame 462 → 1060, but does NOT prevent it.
`once_conv` never latches (confirmed: latch_frame=−1 for C), so the second plateau also fires
(at 3641). usable_linear = 52.0% (slightly better than B's 49.5% due to different reset timing,
but still −44.3pp vs A).

**Design fix required**: Replace `initial_state_active()` gate with a fixed timer anchored at
delay_first. The plateau fires at delay_first+387 frames = 462. A fixed window of 500 frames
from delay_first would cover the critical fire. This is a different design (Variant C2).

#### Why Variant D (SetConfig initial leakage) fails

Leakage switch active for 379 frames (75–454). Effect on H_error:
- Variant B at frame 461: H_error_mean = 0.614 → 0.703 (peak), then 0.703
- Variant D at frame 461: H_error_mean = 0.662 → 0.757 (slightly higher due to 10× leakage refresh)

The higher leakage increases the H_error floor during the initial window, but the ERLE test for
`once_conv` (windowed ERLE ≥ threshold) depends on filter echo suppression quality, not H_error
directly. The DT contamination + A.3 poor-excitation gating keep ERLE below threshold regardless.

`once_conv` requires filter output ERLE ≥ convergence threshold for sustained frames. This is
determined by the actual filter W state, not by H_error leakage rate. In wVYSGVTT, the DT
movement prevents W from converging to the echo path → ERLE stays low → `once_conv` never fires.

SetConfig(initial) leakage change is irrelevant to the actual convergence driver (W adaption quality
under DT conditions). The plateau fires identically to Variant B.

---

### Gate 0 Decision

Per the decision rule (both C and D fail):

**→ UNRESOLVED v3.21 parity gap**

- Do NOT move to v3.22 — this is v3.21 AEC3 alignment work
- Do NOT run 12-case AECMOS on M_full_delay
- M_D proceeds to 12-case AECMOS separately (user authorisation required)
- Documented open item: Variant C2 (fixed timer window) as next diagnostic step before
  closing or shipping M_full_delay

**Corrected Variant C2 specification** (not yet implemented; requires user approval to implement):
```python
# In _reset_filter_derived_state: track _full_delay_chain_trigger_frame
# In plateau check:
_frames_since_chain_trigger = (
    self._frame_count - getattr(self, '_full_delay_chain_trigger_frame', 0)
)
_plateau_chain_suppressed = (
    bool(getattr(self.config, 'use_full_delay_change_chain', False))
    and bool(getattr(self.config, 'use_full_delay_plateau_suppression', False))
    and _frames_since_chain_trigger < 500  # covers delay_first+387=462 plus margin
    and not self._filter_once_converged
)
```

**NOT an architectural dead-end.** M_full_no_plateau showed 96.3% recovery (=M_D level) when
plateau is fully suppressed. The mechanism works; only the gate window definition in Variant C
is wrong (8 frames too narrow). Variant C2 with fixed timer is the simplest fix path.

**EPV/shadow_rise confirmed safe**: zero impact on any variant → not a risk factor.
**9xjhi confirmed safe**: all variants identical → no nores regression risk from Variant C2.
