# v3.21 T1.2 Bundle B Sub-Ladder Trace

**Date**: 2026-05-26  
**T1.2 corrected noise gate**: FILTER_NOISE_GATE_POWER_FLOAT = 0.01870 (from 20075344 int16²)
**Old (incorrect) noise gate**: NOISE_GATE_POWER_FLOAT = 0.02562 (from 27509562, suppression path)  
**Band bins** (fft_size=512, 31.25 Hz/bin): LF=slice(1, 16, None) (31-500 Hz), MF=slice(16, 64, None) (500-2000 Hz), HF=slice(64, 257, None) (2000-8000 Hz)

---

## Table 1: shadow_W_norm_mean

| Case (bucket) | M0 | B1 | B2 | B3 | B4 | B5 |
|---|---|---|---|---|---|---|
| 9xjhiFbG (FS_static) | 83.007 | 80.727 | 74.356 | 71.824 | 71.828 | 71.828 |
| xFk7igec (DT_mvmt) | 17.693 | 20.413 | 14.584 | 14.562 | 14.561 | 14.561 |
| nVUnxqHL (DT_static) | 14.828 | 17.420 | 14.937 | 14.927 | 14.927 | 14.927 |
| XRTnTUjU (DT_static) | 2.181 | 3.242 | 3.123 | 3.144 | 3.144 | 3.144 |
| MYrVxVEM (DT_static) | 13.042 | 12.746 | 12.557 | 12.553 | 12.553 | 12.553 |
| wVYSGVTT (DT_mvmt) | 79.722 | 91.887 | 81.309 | 81.305 | 81.305 | 81.305 |
| qNvSMyUS (FS_static) | 1.818 | 2.101 | 1.823 | 1.820 | 1.820 | 1.820 |
| jtYTdZm3 (DT_static) | 21.856 | 24.645 | 19.942 | 19.888 | 19.822 | 19.822 |

## Table 2: e2_coarse_total (spectral, mean over utterance)

| Case (bucket) | M0 | B1 | B2 | B3 | B4 | B5 |
|---|---|---|---|---|---|---|
| 9xjhiFbG (FS_static) | 2.359e+03 | 4.744e+03 | 4.354e+03 | 4.330e+03 | 4.330e+03 | 4.330e+03 |
| xFk7igec (DT_mvmt) | 1.264e+02 | 2.390e+02 | 1.893e+02 | 1.891e+02 | 1.892e+02 | 1.892e+02 |
| nVUnxqHL (DT_static) | 1.020e+02 | 1.124e+02 | 9.452e+01 | 9.454e+01 | 9.454e+01 | 9.454e+01 |
| XRTnTUjU (DT_static) | 6.748e+01 | 8.499e+01 | 7.839e+01 | 7.844e+01 | 7.844e+01 | 7.844e+01 |
| MYrVxVEM (DT_static) | 1.055e+01 | 1.593e+01 | 1.553e+01 | 1.551e+01 | 1.551e+01 | 1.551e+01 |
| wVYSGVTT (DT_mvmt) | 6.985e+02 | 1.217e+03 | 1.145e+03 | 1.145e+03 | 1.145e+03 | 1.145e+03 |
| qNvSMyUS (FS_static) | 4.444e+01 | 4.602e+01 | 4.591e+01 | 4.591e+01 | 4.591e+01 | 4.591e+01 |
| jtYTdZm3 (DT_static) | 1.628e+02 | 2.780e+02 | 2.673e+02 | 2.674e+02 | 2.833e+02 | 2.833e+02 |

## Table 3: A2 zero_frac — all-band (new threshold)

> **zero_frac = fraction of frequency bins where X²_sum < noise_gate**  
> Δ(B2−B1): B2 applies the gate; B1 does not. Shows gate effect.
> old_B2: counterfactual with old 0.02562 threshold (delta shows correction benefit.)

| Case | M0 | B1 | B2(new) | B2_old | Δnew | Δnew−Δold |
|---|---|---|---|---|---|---|
| 9xjhiFbG (FS_static) | 31.0% | 31.0% | 31.0% | 34.5% | +0.0pp | -3.4pp |
| xFk7igec (DT_mvmt) | 58.9% | 58.9% | 58.9% | 61.0% | +0.0pp | -2.1pp |
| nVUnxqHL (DT_static) | 61.2% | 61.2% | 61.2% | 63.4% | +0.0pp | -2.1pp |
| XRTnTUjU (DT_static) | 70.0% | 69.9% | 69.9% | 71.1% | +0.0pp | -1.2pp |
| MYrVxVEM (DT_static) | 67.8% | 67.8% | 67.8% | 69.8% | +0.0pp | -2.0pp |
| wVYSGVTT (DT_mvmt) | 68.4% | 68.4% | 68.4% | 70.4% | +0.0pp | -2.0pp |
| qNvSMyUS (FS_static) | 41.5% | 41.5% | 41.5% | 46.1% | +0.0pp | -4.6pp |
| jtYTdZm3 (DT_static) | 65.0% | 65.0% | 65.0% | 68.4% | +0.0pp | -3.4pp |

> **Note**: A2_zero_frac counts what fraction of bins *would be* zeroed.
> For M0/B1 (gate OFF), this is the *potential* gate fraction (counterfactual).
> For B2+ (gate ON), this matches the actual zeroed fraction.

## Table 4: A2 zero_frac per band — B2 (corrected threshold)

| Case | all | LF(31-500Hz) | MF(500-2000Hz) | HF(2000-8000Hz) |
|---|---|---|---|---|
| 9xjhiFbG (FS_static) | 31.0% | 17.2% | 20.7% | 34.7% |
| xFk7igec (DT_mvmt) | 58.9% | 49.2% | 53.9% | 60.9% |
| nVUnxqHL (DT_static) | 61.2% | 44.5% | 58.5% | 63.2% |
| XRTnTUjU (DT_static) | 69.9% | 64.2% | 65.0% | 71.6% |
| MYrVxVEM (DT_static) | 67.8% | 48.5% | 61.1% | 71.1% |
| wVYSGVTT (DT_mvmt) | 68.4% | 53.8% | 59.9% | 71.7% |
| qNvSMyUS (FS_static) | 41.5% | 18.5% | 25.4% | 47.4% |
| jtYTdZm3 (DT_static) | 65.0% | 40.7% | 42.1% | 72.7% |

## Table 5: mu_eff_mean (far-active frames)

> T1.1 analytical result: Python (B1 ON) vs AEC3 rate ratio ≈ 0.74× per second (< 2× threshold).
> This table confirms the mu_eff order-of-magnitude matches.

| Case | M0_mu_mean | B1_mu_mean | B2_mu_mean | B1/M0 ratio | B1_mu_p95 |
|---|---|---|---|---|---|
| 9xjhiFbG (FS_static) | 5.294e+00 | 2.377e+01 | 4.132e+00 | 4.49 | 7.659e+01 |
| xFk7igec (DT_mvmt) | 8.116e+00 | 4.830e+01 | 3.592e+00 | 5.95 | 1.932e+02 |
| nVUnxqHL (DT_static) | 8.392e+00 | 2.708e+01 | 4.046e+00 | 3.23 | 6.643e+01 |
| XRTnTUjU (DT_static) | 1.197e+01 | 5.795e+01 | 2.981e+00 | 4.84 | 1.343e+02 |
| MYrVxVEM (DT_static) | 1.421e+01 | 5.073e+01 | 3.934e+00 | 3.57 | 1.264e+02 |
| wVYSGVTT (DT_mvmt) | 1.252e+01 | 5.774e+01 | 4.269e+00 | 4.61 | 1.619e+02 |
| qNvSMyUS (FS_static) | 1.198e+01 | 3.493e+01 | 4.713e+00 | 2.92 | 8.593e+01 |
| jtYTdZm3 (DT_static) | 4.723e+01 | 8.761e+01 | 4.397e+00 | 1.86 | 3.240e+02 |

## Table 6: e2_coarse per band (B2) — spectral attribution

| Case | total | LF | MF | HF | dominant |
|---|---|---|---|---|---|
| 9xjhiFbG (FS_static) | 4.354e+03 | 6.874e+02 | 2.760e+03 | 9.018e+02 | MF |
| xFk7igec (DT_mvmt) | 1.893e+02 | 7.842e+01 | 6.193e+01 | 4.868e+01 | LF |
| nVUnxqHL (DT_static) | 9.452e+01 | 5.854e+01 | 2.970e+01 | 5.929e+00 | LF |
| XRTnTUjU (DT_static) | 7.839e+01 | 2.820e+01 | 4.175e+01 | 8.313e+00 | MF |
| MYrVxVEM (DT_static) | 1.553e+01 | 5.854e+00 | 9.320e+00 | 3.035e-01 | MF |
| wVYSGVTT (DT_mvmt) | 1.145e+03 | 5.669e+02 | 4.417e+02 | 1.335e+02 | LF |
| qNvSMyUS (FS_static) | 4.591e+01 | 6.153e+00 | 3.899e+01 | 7.288e-01 | MF |
| jtYTdZm3 (DT_static) | 2.673e+02 | 1.663e+02 | 8.851e+01 | 1.158e+01 | LF |

## Table 7: coarse_conv_frac (e2_coarse < 0.05·y2 sustained)

| Case | M0 | B1 | B2 | B3 | B4 | B5 |
|---|---|---|---|---|---|---|
| 9xjhiFbG (FS_static) | 0.0% | 0.1% | 0.1% | 0.1% | 0.1% | 0.1% |
| xFk7igec (DT_mvmt) | 0.4% | 0.1% | 0.2% | 0.2% | 0.2% | 0.2% |
| nVUnxqHL (DT_static) | 0.4% | 0.0% | 0.1% | 0.1% | 0.1% | 0.1% |
| XRTnTUjU (DT_static) | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% |
| MYrVxVEM (DT_static) | 2.3% | 0.7% | 0.9% | 0.9% | 0.9% | 0.9% |
| wVYSGVTT (DT_mvmt) | 5.5% | 4.1% | 4.1% | 4.1% | 4.1% | 4.1% |
| qNvSMyUS (FS_static) | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% |
| jtYTdZm3 (DT_static) | 5.6% | 5.0% | 5.0% | 5.0% | 5.1% | 5.1% |

## Table 8: e2_ratio (e2_coarse / e2_refined) — lower = coarse closer to refined

| Case | M0 | B1 | B2 | B3 | B4 | B5 |
|---|---|---|---|---|---|---|
| 9xjhiFbG (FS_static) | 0.60 | 1.21 | 1.11 | 1.11 | 1.11 | 1.11 |
| xFk7igec (DT_mvmt) | 13.87 | 26.35 | 21.01 | 20.97 | 20.98 | 20.98 |
| nVUnxqHL (DT_static) | 33.71 | 62.35 | 92.01 | 92.02 | 92.02 | 92.02 |
| XRTnTUjU (DT_static) | 247.64 | 313.51 | 290.27 | 290.15 | 290.15 | 290.15 |
| MYrVxVEM (DT_static) | 4.15 | 4.36 | 3.45 | 3.44 | 3.44 | 3.44 |
| wVYSGVTT (DT_mvmt) | 194.62 | 163.48 | 157.07 | 157.07 | 157.08 | 157.08 |
| qNvSMyUS (FS_static) | 216.13 | 224.39 | 226.63 | 226.60 | 226.61 | 226.61 |
| jtYTdZm3 (DT_static) | 55.43 | 66.55 | 62.44 | 62.44 | 65.24 | 65.24 |

## Table 9: A3/A4/A5 gate fire rates

| Case | B3_a3_skip | B4_a4_mask | B5_a5_sat |
|---|---|---|---|
| 9xjhiFbG (FS_static) | 2.5% | 0.4% | 0.0% |
| xFk7igec (DT_mvmt) | 0.4% | 0.1% | 0.0% |
| nVUnxqHL (DT_static) | 0.5% | 0.0% | 0.0% |
| XRTnTUjU (DT_static) | 0.7% | 0.0% | 0.0% |
| MYrVxVEM (DT_static) | 1.5% | 0.1% | 0.0% |
| wVYSGVTT (DT_mvmt) | 0.6% | 0.0% | 0.0% |
| qNvSMyUS (FS_static) | 3.8% | 0.5% | 0.0% |
| jtYTdZm3 (DT_static) | 0.3% | 0.0% | 0.0% |

## Per-Frame Worst Windows (B2) — xFk7 / 9xjhi / wVYSG

### 9xjhiFbG (FS_static)

| rank | frame | e2_coa | e2_coa_lf | e2_coa_mf | e2_coa_hf | w_norm | a2_zero |
|---|---|---|---|---|---|---|---|
| 1 | 1030 | 4.660e+05 | 1.686e+03 | 4.538e+05 | 1.060e+04 | 117.820 | 0.0% |
| 2 | 2134 | 3.994e+05 | 1.499e+03 | 3.897e+05 | 8.174e+03 | 96.942 | 0.0% |
| 3 | 1031 | 1.796e+05 | 2.604e+03 | 1.660e+05 | 1.096e+04 | 115.673 | 0.0% |
| 4 | 2009 | 1.694e+05 | 1.324e+03 | 1.667e+05 | 1.310e+03 | 70.861 | 0.0% |
| 5 | 905 | 1.681e+05 | 7.535e+02 | 1.659e+05 | 1.450e+03 | 73.991 | 0.0% |
| 6 | 2135 | 1.463e+05 | 2.070e+03 | 1.352e+05 | 8.939e+03 | 95.291 | 0.0% |
| 7 | 50 | 1.440e+05 | 1.297e+05 | 7.927e+03 | 6.171e+03 | 61.886 | 0.4% |
| 8 | 906 | 1.253e+05 | 9.637e+01 | 1.220e+05 | 3.197e+03 | 79.091 | 0.0% |
| 9 | 49 | 1.175e+05 | 1.090e+05 | 2.894e+03 | 4.748e+03 | 60.481 | 0.4% |
| 10 | 2010 | 1.146e+05 | 1.935e+02 | 1.114e+05 | 3.015e+03 | 75.583 | 0.0% |

### xFk7igec (DT_mvmt)

| rank | frame | e2_coa | e2_coa_lf | e2_coa_mf | e2_coa_hf | w_norm | a2_zero |
|---|---|---|---|---|---|---|---|
| 1 | 3142 | 5.774e+04 | 5.617e+04 | 1.529e+03 | 3.500e+01 | 35.051 | 0.0% |
| 2 | 3141 | 4.136e+04 | 3.928e+04 | 1.950e+03 | 9.608e+01 | 36.546 | 0.0% |
| 3 | 3143 | 3.939e+04 | 3.821e+04 | 1.106e+03 | 7.103e+01 | 52.832 | 0.0% |
| 4 | 3140 | 1.498e+04 | 1.377e+04 | 1.096e+03 | 1.119e+02 | 36.411 | 0.0% |
| 5 | 3107 | 1.197e+04 | 5.122e+01 | 1.162e+01 | 1.191e+04 | 44.298 | 0.0% |
| 6 | 3108 | 1.106e+04 | 1.804e+01 | 1.046e+01 | 1.103e+04 | 43.300 | 0.0% |
| 7 | 3114 | 1.052e+04 | 9.662e-01 | 1.439e+00 | 1.051e+04 | 35.067 | 0.0% |
| 8 | 3078 | 7.219e+03 | 2.351e+01 | 6.191e+03 | 1.005e+03 | 50.208 | 0.0% |
| 9 | 98 | 6.996e+03 | 9.337e+02 | 5.793e+03 | 2.634e+02 | 8.904 | 0.0% |
| 10 | 3110 | 6.833e+03 | 7.593e+00 | 6.099e+00 | 6.819e+03 | 40.230 | 0.0% |

### wVYSGVTT (DT_mvmt)

| rank | frame | e2_coa | e2_coa_lf | e2_coa_mf | e2_coa_hf | w_norm | a2_zero |
|---|---|---|---|---|---|---|---|
| 1 | 3631 | 1.120e+05 | 8.263e+04 | 2.779e+04 | 7.037e+02 | 76.644 | 0.0% |
| 2 | 3630 | 9.158e+04 | 5.769e+04 | 3.304e+04 | 3.462e+02 | 155.659 | 0.4% |
| 3 | 2758 | 5.784e+04 | 1.110e+04 | 4.627e+04 | 4.244e+02 | 80.958 | 0.0% |
| 4 | 2759 | 4.413e+04 | 1.186e+04 | 3.190e+04 | 3.669e+02 | 81.158 | 0.0% |
| 5 | 2757 | 3.955e+04 | 1.295e+04 | 2.654e+04 | 5.303e+01 | 80.416 | 0.0% |
| 6 | 2796 | 3.871e+04 | 3.646e+04 | 1.602e+03 | 1.073e+01 | 87.969 | 0.0% |
| 7 | 2797 | 3.246e+04 | 2.513e+04 | 6.857e+03 | 3.300e+01 | 86.987 | 0.0% |
| 8 | 3025 | 3.164e+04 | 3.010e+04 | 1.446e+03 | 4.620e+01 | 89.761 | 0.0% |
| 9 | 3024 | 2.884e+04 | 2.783e+04 | 9.046e+02 | 9.286e+01 | 90.048 | 0.0% |
| 10 | 2760 | 2.631e+04 | 9.543e+03 | 1.659e+04 | 1.682e+02 | 80.931 | 0.0% |

## Verdict

Determination rules (per T1.2 plan):
1. **B2 vs B1 zero_frac**: zero_frac MUST decrease with corrected threshold (new 0.01870 < old 0.02562 → fewer bins zeroed → gate more permissive)
2. **B2 vs B1 e2_coarse**: if improves → constant correction helps convergence
3. **B2 vs B1 e2_coarse**: if worsens → threshold not main variable; revisit SpectralSum depth / mu scaling
4. **B1 vs M0 mu_eff_mean**: confirm T1.1 analytical ratio 0.74× per second

See Tables 1-9 for detailed breakdown. Append conclusion to `docs/v3_21_bundle_b_formula_correction.md`.
