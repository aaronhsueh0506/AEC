# v3.21 Tier-C Validation — corrected/reverted shipped flags (isolated)

**Cases**: 800  
**Config**: balanced / filter=832 / cng / hop=160  
**Baseline**: V0_prod (current production)

Matched-magnitude AECMOS Pareto: less echo auto-lifts deg — NOT a win.

## Bucket means (absolute)

| Bucket | metric | V0_prod | V1_active_render | V2_fft_off | V3_reverb_off | V4_dne_off | V5_all_corrected |
|---|---|---|---|---|---|---|---|
| DT_mvmt | deg | 2.159 | 2.173 | 2.160 | 2.159 | 2.094 | 2.096 |
| DT_static | deg | 2.061 | 2.070 | 2.059 | 2.061 | 1.986 | 1.998 |
| FS_mvmt | echo | 3.343 | 3.328 | 3.341 | 3.343 | 3.454 | 3.433 |
| FS_static | echo | 3.480 | 3.447 | 3.480 | 3.480 | 3.562 | 3.539 |
| NE | deg | 3.942 | 3.943 | 3.935 | 3.942 | 3.928 | 3.929 |

## Δ vs V0_prod (negative = regression on own metric)

| Bucket | metric | V1_active_render | V2_fft_off | V3_reverb_off | V4_dne_off | V5_all_corrected |
|---|---|---|---|---|---|---|
| DT_mvmt | deg | +0.013 | +0.000 | +0.000 | -0.065 | -0.063 |
| DT_static | deg | +0.009 | -0.001 | +0.000 | -0.075 | -0.062 |
| FS_mvmt | echo | -0.014 | -0.001 | +0.000 | +0.111 | +0.090 |
| FS_static | echo | -0.033 | -0.000 | +0.000 | +0.082 | +0.059 |
| NE | deg | +0.001 | -0.007 | +0.000 | -0.014 | -0.014 |

## Catastrophic regressions vs V0 (Δ < −0.20 on own metric)

| Variant | n_catastrophic | worst Δ | worst case |
|---|---:|---:|---|
| V1_active_render | 25 | -1.431 | `IxgmaPghzUGnR6sxrbGU3Q_farend_singletalk` |
| V2_fft_off | 0 | -0.180 | `ZBc1WgCwmEa2M5gKDuVWkw_farend_singletalk_with_` |
| V3_reverb_off | 0 | +0.000 | `` |
| V4_dne_off | 51 | -0.745 | `WtQs4a0YeU2B0dQWhS7gmg_doubletalk` |
| V5_all_corrected | 49 | -1.043 | `IxgmaPghzUGnR6sxrbGU3Q_farend_singletalk` |

## nores LF (FS_static): V5_all_corrected vs V0_prod

N=169  mean Δ=+0.00 dB  max regression=+0.00 dB  (regressions Δ>+1dB: 0)

9xjhi: V0=29.70 dB  V5=29.70 dB  Δ=+0.00 dB

