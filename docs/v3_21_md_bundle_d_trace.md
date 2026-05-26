# v3.21 M_D Bundle D Trace (M_A_ceil2 Baseline)

**Date**: 2026-05-26  
**Config**: M_A_ceil2 = Bundle B+C+A + use_aec3_h_error_ceil=True  
**M_D = M_A_ceil2 (no new algorithm flags; filter_analyzer_enabled already ON)**  
**No AECMOS / no 800-case / no verdict / no version bump**

> Primary focus: wVYSGVTT, xFk7, 9xjhi, XRTnTUjU

## §1 usable_linear_frac + Gate Breakdown

> Gate 1: filter_update_blocks_since_start > 40  
> Gate 2: filter_update_blocks_since_reset > 20  
> Gate 3 ext: external delay known  
> Gate 3 conv: convergence_seen latch

| Case | UL_frac | g1 | g2 | g3_ext | g3_conv | FA_consistent | conv_latch | delay_events |
|------|--------|----|----|----|--------|--------------|-----------|-------------|
| 9xjhiFbG_FS_static | 92.4% | 93.0% | 93.0% | 96.8% | 80.9% | 72.3% | 222 | 1 |
| xFk7igec_DT_mvmt | 96.1% | 96.2% | 96.2% | 97.2% | 96.5% | 90.8% | 129 | 1 |
| nVUnxqHL_DT_static | 95.9% | 96.0% | 96.0% | 97.2% | 92.9% | 90.6% | 304 | 1 |
| XRTnTUjU_DT_static | 91.2% | 91.2% | 91.2% | 96.9% | 18.6% | 0.0% | 241 | 1 |
| MYrVxVEM_DT_static | 95.6% | 95.6% | 95.6% | 97.8% | 95.3% | 86.6% | 156 | 1 |
| wVYSGVTT_DT_mvmt | 96.3% | 96.3% | 96.3% | 98.0% | 96.2% | 90.3% | 140 | 1 |
| qNvSMyUS_FS_static | 82.5% | 85.7% | 85.7% | 93.0% | 53.9% | 0.0% | 1239 | 1 |
| jtYTdZm3_DT_static | 97.5% | 97.8% | 97.8% | 93.3% | 97.7% | 87.8% | 53 | 1 |

## §2 Delay and FA Details

| Case | delay_blocks | min_direct_path | ext_delay_src_dist |
|------|------------|----------------|-------------------|
| 9xjhiFbG_FS_static | 0.09 | 0.000 | is_solid:96% none:3% |
| xFk7igec_DT_mvmt | 0.03 | 0.000 | is_solid:97% none:2% |
| nVUnxqHL_DT_static | 0.02 | 0.000 | is_solid:97% none:2% |
| XRTnTUjU_DT_static | 0.75 | 0.000 | is_solid:96% none:3% |
| MYrVxVEM_DT_static | 0.01 | 0.000 | is_solid:97% none:2% |
| wVYSGVTT_DT_mvmt | 0.01 | 0.000 | is_solid:97% none:2% |
| qNvSMyUS_FS_static | 1.02 | 0.000 | is_solid:92% none:7% |
| jtYTdZm3_DT_static | 0.07 | 0.000 | is_solid:93% none:6% |

## §3 URO cond1/cond2 (confirmation vs M_C)

| Case | use_refined | cond1 | cond2 | e2_ref | e2_coa |
|------|------------|-------|-------|--------|--------|
| 9xjhiFbG_FS_static | 52.9% | 42.6% | 6.1% | 12.416 | 18.362 |
| xFk7igec_DT_mvmt | 60.1% | 16.2% | 30.2% | 0.351 | 0.476 |
| nVUnxqHL_DT_static | 69.9% | 14.9% | 23.8% | 0.290 | 0.341 |
| XRTnTUjU_DT_static | 70.1% | 9.7% | 28.3% | 0.279 | 0.307 |
| MYrVxVEM_DT_static | 63.8% | 24.4% | 25.2% | 0.087 | 0.051 |
| wVYSGVTT_DT_mvmt | 64.7% | 23.8% | 16.3% | 3.227 | 3.133 |
| qNvSMyUS_FS_static | 49.9% | 3.1% | 48.7% | 0.199 | 0.174 |
| jtYTdZm3_DT_static | 68.1% | 30.5% | 6.9% | 0.456 | 0.455 |

## §4 r2 and Gain Per Band

| Case | r2_lf | r2_mf | r2_hf | gain_lf | gain_mf | gain_hf |
|------|-------|-------|-------|---------|---------|---------|
| 9xjhiFbG_FS_static | 3.735e+10 | 4.928e+09 | 1.534e+09 | 0.370 | 0.588 | 0.577 |
| xFk7igec_DT_mvmt | 9.631e+08 | 1.476e+09 | 4.350e+07 | 0.705 | 0.578 | 0.550 |
| nVUnxqHL_DT_static | 2.641e+08 | 1.204e+09 | 5.489e+06 | 0.741 | 0.571 | 0.667 |
| XRTnTUjU_DT_static | 8.804e+06 | 1.033e+07 | 1.166e+08 | 0.884 | 0.847 | 0.788 |
| MYrVxVEM_DT_static | 1.278e+08 | 1.809e+09 | 6.642e+05 | 0.736 | 0.613 | 0.729 |
| wVYSGVTT_DT_mvmt | 1.572e+08 | 3.776e+08 | 2.465e+07 | 0.751 | 0.798 | 0.809 |
| qNvSMyUS_FS_static | 5.051e+07 | 2.196e+07 | 2.300e+05 | 0.386 | 0.517 | 0.557 |
| jtYTdZm3_DT_static | 1.992e+08 | 3.129e+09 | 4.229e+06 | 0.781 | 0.534 | 0.663 |

## §5 Gain Reason Distribution LF / HF

| Case | LF reason | HF reason |
|------|-----------|-----------|
| 9xjhiFbG_FS_static | G:85% max:12% min:2% | G:77% max:22% |
| xFk7igec_DT_mvmt | G:94% max:4% min:0% | G:95% max:4% |
| nVUnxqHL_DT_static | G:89% max:8% min:1% | G:92% max:7% |
| XRTnTUjU_DT_static | G:97% max:2% min:0% | G:94% max:5% |
| MYrVxVEM_DT_static | G:92% max:6% min:1% | G:93% max:6% |
| wVYSGVTT_DT_mvmt | G:90% max:8% min:1% | G:95% max:4% |
| qNvSMyUS_FS_static | G:87% max:9% min:2% | G:78% min:17% max:3% |
| jtYTdZm3_DT_static | G:85% max:11% min:3% | G:93% max:6% |

## §6 Nores Band Energy (M_D) — Δ vs M0

> M_D nores = M_A_ceil2 with enable_res=False

### Nores LF (0–500 Hz)

| Case | M0 dB | M_D dB | Δ |
|------|-------|--------|---|
| 9xjhiFbG_FS_static | 24.80 | 18.72 | -6.07 |
| xFk7igec_DT_mvmt | 6.16 | 7.56 | +1.40 |
| nVUnxqHL_DT_static | 9.42 | 9.57 | +0.15 |
| XRTnTUjU_DT_static | 5.01 | 5.06 | +0.05 |
| MYrVxVEM_DT_static | -0.22 | 0.43 | +0.65 |
| wVYSGVTT_DT_mvmt | 16.92 | 15.88 | -1.04 |
| qNvSMyUS_FS_static | -2.35 | -2.38 | -0.03 |
| jtYTdZm3_DT_static | 13.63 | 12.12 | -1.51 |

### Nores MF (500–2000 Hz)

| Case | M0 dB | M_D dB | Δ |
|------|-------|--------|---|
| 9xjhiFbG_FS_static | 23.41 | 25.49 | +2.08 |
| xFk7igec_DT_mvmt | 12.87 | 11.18 | -1.68 |
| nVUnxqHL_DT_static | 9.75 | 7.24 | -2.51 |
| XRTnTUjU_DT_static | 8.49 | 8.57 | +0.08 |
| MYrVxVEM_DT_static | 7.13 | 3.86 | -3.26 |
| wVYSGVTT_DT_mvmt | 20.12 | 19.66 | -0.46 |
| qNvSMyUS_FS_static | 5.21 | 5.52 | +0.31 |
| jtYTdZm3_DT_static | 10.48 | 9.04 | -1.44 |

### Nores HF (2000–8000 Hz)

| Case | M0 dB | M_D dB | Δ |
|------|-------|--------|---|
| 9xjhiFbG_FS_static | 21.76 | 23.25 | +1.49 |
| xFk7igec_DT_mvmt | 8.18 | 0.68 | -7.50 |
| nVUnxqHL_DT_static | 0.12 | 0.53 | +0.42 |
| XRTnTUjU_DT_static | 0.66 | 0.94 | +0.28 |
| MYrVxVEM_DT_static | -4.34 | -10.08 | -5.74 |
| wVYSGVTT_DT_mvmt | 14.41 | 14.90 | +0.49 |
| qNvSMyUS_FS_static | -11.03 | -10.96 | +0.06 |
| jtYTdZm3_DT_static | 2.92 | 3.32 | +0.40 |

## §7 Worst-Frame Detail (top-8 by e2_coarse) — 4 Detail Stems

### 9xjhiFbG_FS_static

| frame | UL | g1 | g2 | g3e | g3c | FA | e2_coa | e2_ref | URO | c1 | c2 | r2_lf | gain_lf |
|-------|----|----|----|----|-----|-------|--------|-----|----|----|-------|---------|
| 1030 | T | T | T | T | T | T | 1733.3823 | 7.5273 | F | F | F | 6.607e+09 | 0.089 |
| 2134 | T | T | T | T | T | T | 1396.4039 | 12.2383 | F | F | F | 6.176e+09 | 0.094 |
| 1031 | T | T | T | T | T | T | 637.4972 | 9.9577 | F | F | F | 9.035e+09 | 0.076 |
| 905 | T | T | T | T | T | T | 607.5347 | 7.8001 | F | F | F | 2.469e+09 | 0.137 |
| 2009 | T | T | T | T | T | T | 561.6162 | 10.8343 | F | F | F | 2.381e+09 | 0.149 |
| 50 | F | F | F | F | F | F | 526.9506 | 31.6515 | F | F | F | 3.586e+07 | 1.000 |
| 476 | T | T | T | T | T | T | 464.6578 | 12.5998 | F | F | F | 4.003e+10 | 0.064 |
| 2135 | T | T | T | T | T | T | 462.6535 | 12.4723 | F | F | F | 9.125e+09 | 0.077 |

### xFk7igec_DT_mvmt

| frame | UL | g1 | g2 | g3e | g3c | FA | e2_coa | e2_ref | URO | c1 | c2 | r2_lf | gain_lf |
|-------|----|----|----|----|-----|-------|--------|-----|----|----|-------|---------|
| 3142 | T | T | T | T | T | T | 220.5746 | 1.4645 | F | F | F | 7.650e+09 | 0.108 |
| 3141 | T | T | T | T | T | T | 156.3159 | 0.7103 | F | F | F | 6.588e+09 | 0.117 |
| 3143 | T | T | T | T | T | T | 155.0851 | 2.1118 | F | F | F | 1.078e+10 | 0.091 |
| 3140 | T | T | T | T | T | T | 53.6327 | 0.7172 | F | F | F | 2.693e+09 | 0.182 |
| 98 | F | T | T | F | F | F | 27.2837 | 23.4219 | F | F | F | 2.677e+08 | 1.000 |
| 100 | F | T | T | F | F | F | 20.5876 | 19.5381 | F | F | F | 2.675e+08 | 1.000 |
| 99 | F | T | T | F | F | F | 20.1659 | 14.7373 | F | F | F | 2.741e+08 | 1.000 |
| 97 | F | F | F | F | F | F | 19.4640 | 18.8118 | F | F | F | 2.644e+08 | 1.000 |

### XRTnTUjU_DT_static

| frame | UL | g1 | g2 | g3e | g3c | FA | e2_coa | e2_ref | URO | c1 | c2 | r2_lf | gain_lf |
|-------|----|----|----|----|-----|-------|--------|-----|----|----|-------|---------|
| 3094 | T | T | T | T | F | F | 22.5311 | 0.0367 | F | F | F | 7.139e+05 | 1.000 |
| 2935 | T | T | T | T | F | F | 11.6818 | 0.8782 | F | F | F | 3.264e+04 | 1.000 |
| 2936 | T | T | T | T | F | F | 10.8877 | 0.1713 | F | F | F | 1.618e+06 | 1.000 |
| 3096 | T | T | T | T | F | F | 9.4136 | 0.0225 | F | F | F | 4.002e+06 | 1.000 |
| 3095 | T | T | T | T | F | F | 8.3133 | 0.0970 | F | F | F | 7.475e+05 | 1.000 |
| 3243 | T | T | T | T | F | F | 7.5482 | 0.0305 | F | F | F | 2.122e+07 | 0.491 |
| 3242 | T | T | T | T | F | F | 6.6070 | 0.0194 | F | F | F | 2.036e+07 | 0.578 |
| 2872 | T | T | T | T | F | F | 5.6523 | 0.8994 | F | F | F | 7.729e+06 | 1.000 |

### wVYSGVTT_DT_mvmt

| frame | UL | g1 | g2 | g3e | g3c | FA | e2_coa | e2_ref | URO | c1 | c2 | r2_lf | gain_lf |
|-------|----|----|----|----|-----|-------|--------|-----|----|----|-------|---------|
| 3631 | T | T | T | T | T | T | 439.1758 | 0.4008 | F | F | F | 3.425e+07 | 1.000 |
| 3630 | T | T | T | T | T | T | 358.0065 | 0.3348 | F | F | F | 2.946e+07 | 1.000 |
| 2756 | T | T | T | T | T | T | 97.1749 | 8.7586 | F | F | F | 4.147e+09 | 0.064 |
| 3628 | T | T | T | T | T | T | 68.7312 | 5.3326 | F | F | F | 3.915e+07 | 1.000 |
| 2753 | T | T | T | T | T | T | 65.2652 | 10.7713 | F | F | F | 1.042e+09 | 0.127 |
| 2889 | T | T | T | T | T | T | 62.7732 | 37.2115 | F | F | F | 1.911e+07 | 1.000 |
| 3015 | T | T | T | T | T | T | 58.8370 | 15.0728 | F | F | F | 7.076e+08 | 0.125 |
| 2754 | T | T | T | T | T | T | 58.7908 | 8.8351 | F | F | F | 1.308e+09 | 0.113 |

## §8 M_D Determination

**M_D = M_A_ceil2 (no new flags)**. If all metrics below are consistent with M_A_ceil2 trace,
label: **Bundle D trace-only COMPLETE**.

Key checks:
- wVYSGVTT usable_linear: expect ~96.3% (recovered from 49.5% in old M_A)
- FA_consistent: should reflect actual FilterAnalyzer state under ceil=2 convergence
- delay_events: confirm count per case for M_full_delay gate
- nores LF delta: 9xjhi should remain ≤ Δ−6 dB (same as M_A_ceil2)
