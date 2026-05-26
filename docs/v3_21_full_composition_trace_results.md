# v3.21 Full AEC3 Composition Trace — Results

**Date**: 2026-05-26  
**Ladder**: M0 → M_R0 → M_B_corrected → M_C → M_A → M_D → M_full_delay

> **Measurement status (v3 corrected — 2026-05-26)**: Full 7-step ladder traced (all cases, including M_full_delay). H_ERROR_CEIL confirmed — all steady_excl20=0.0% at M_A (use_aec3_h_error_ceil=True resolves saturation). Bundle A/D/M_full_delay all fully measured. **AECMOS not yet run** — pending user authorisation. Prior stale notes (H_ERROR_CEIL retracted, A/D/M_full_delay not measured) are superseded.

## § Bundle B: Shadow/Coarse PBFDAF — M0 vs M_B

| Case | M0 shadow_W_norm | M_B_corr shadow_W_norm | M0 e2_coarse | M_B_corr e2_coarse | M_B_corr A2_zero_frac | M_B_corr A3_skip_frac | M_B_corr coarse_conv_frac |
|------|-----------------|----------------------|------------|------------------|---------------------|---------------------|--------------------------|
| 9xjhiFbG_FS_static | 83.007 | 72.321 | 9.211 | 16.914 | 6.2% | 1.4% | 0.1% |
| xFk7igec_DT_mvmt | 17.693 | 16.158 | 0.493 | 0.773 | 7.0% | 0.2% | 0.2% |
| nVUnxqHL_DT_static | 14.828 | 15.290 | 0.398 | 0.367 | 3.9% | 0.1% | 0.2% |
| XRTnTUjU_DT_static | 2.182 | 3.174 | 0.263 | 0.306 | 2.9% | 0.2% | 0.0% |
| MYrVxVEM_DT_static | 13.042 | 12.553 | 0.041 | 0.060 | 6.5% | 0.5% | 0.9% |
| wVYSGVTT_DT_mvmt | 79.722 | 80.284 | 2.727 | 3.877 | 5.2% | 0.2% | 4.3% |
| qNvSMyUS_FS_static | 1.818 | 1.776 | 0.174 | 0.179 | 12.1% | 2.1% | 0.0% |
| jtYTdZm3_DT_static | 21.856 | 21.352 | 0.635 | 4.485 | 19.0% | 0.2% | 5.0% |

## § Bundle B URO Coarse-Frame Attribution — M_C

> Stats computed only on frames where URO selected coarse output.

| Case | coarse_frames | e2_coa@coa | e2_ref@coa | y2@coa | s2_coa@coa | shadow_W@coa | A2_zero@coa | A3_skip@coa |
|------|--------------|-----------|-----------|-------|-----------|------------|------------|------------|
| 9xjhiFbG_FS_static | 779 | 10.678 | 10335.159 | 17.724 | 15.290 | 77.893 | 12.2% | 1.2% |
| xFk7igec_DT_mvmt | 2249 | 0.620 | 14.916 | 0.563 | 0.719 | 14.879 | 8.0% | 0.0% |
| nVUnxqHL_DT_static | 1277 | 0.280 | 2.103 | 0.302 | 0.117 | 14.779 | 4.5% | 0.0% |
| XRTnTUjU_DT_static | 1113 | 0.190 | 0.244 | 0.195 | 0.020 | 3.334 | 5.4% | 0.0% |
| MYrVxVEM_DT_static | 2013 | 0.083 | 6.787 | 0.191 | 0.144 | 13.126 | 10.8% | 0.7% |
| wVYSGVTT_DT_mvmt | 1797 | 2.195 | 11.804 | 5.064 | 4.547 | 78.813 | 6.8% | 0.0% |
| qNvSMyUS_FS_static | 1973 | 0.085 | 0.161 | 0.156 | 0.058 | 1.288 | 13.7% | 2.4% |
| jtYTdZm3_DT_static | 1721 | 2.677 | 27.817 | 0.824 | 3.019 | 23.490 | 32.5% | 0.0% |

## § Bundle C: URO Root-Cause — M_C

| Case | use_refined_frac | cond1_frac | cond2_frac | e2_ratio(coa/ref) | form_transition_frac |
|------|-----------------|-----------|-----------|-----------------|---------------------|
| 9xjhiFbG_FS_static | 64.4% | 32.6% | 7.6% | 0.005 | 17.5% |
| xFk7igec_DT_mvmt | 38.9% | 36.2% | 56.7% | 0.083 | 34.4% |
| nVUnxqHL_DT_static | 70.3% | 12.8% | 25.9% | 0.440 | 30.2% |
| XRTnTUjU_DT_static | 68.1% | 10.5% | 30.6% | 1.133 | 16.6% |
| MYrVxVEM_DT_static | 49.0% | 43.4% | 45.0% | 0.017 | 9.0% |
| wVYSGVTT_DT_mvmt | 51.0% | 27.6% | 37.9% | 0.490 | 29.3% |
| qNvSMyUS_FS_static | 26.5% | 6.6% | 72.6% | 0.880 | 11.4% |
| jtYTdZm3_DT_static | 53.2% | 44.5% | 39.0% | 0.282 | 8.6% |

## § Bundle A: H_error Contamination-Aware Audit — M_C vs M_A

> NOTE: H_error_mean is contaminated by frames where `far_hop_energy <= 1e-4` skips `_update_weights()` (and thus `_h_error_refresh()`), leaving H_error at init=10000.  
> Use `h_error_steady_at_ceil_exclN` (excluding N hops after reset) as the true ceiling-saturation signal.  
> H_ERROR_CEIL verdict: **CONFIRMED** — all cases steady_excl20=0.0% at M_A (use_aec3_h_error_ceil=True). Ceiling saturation fully resolved. Prior retracted verdict superseded by v3 corrected trace.

| Case | step | H_mean(contaminated) | reset_frames | above_ceil_frac | post_refresh_at_ceil | steady_excl6 | steady_excl20 | steady_excl50 |
|------|------|---------------------|------------|---------------|---------------------|------------|--------------|--------------|
| 9xjhiFbG_FS_static | M_C | 22.987 | 4 | 0.2% | 0.8% | 0.5% | 0.0% | 0.0% |
| 9xjhiFbG_FS_static | M_A | 18.804 | 4 | 0.2% | 0.0% | 0.0% | 0.0% | 0.0% |
| xFk7igec_DT_mvmt | M_C | 165.339 | 60 | 1.6% | 0.6% | 0.1% | 0.0% | 0.0% |
| xFk7igec_DT_mvmt | M_A | 160.585 | 59 | 1.6% | 0.0% | 0.0% | 0.0% | 0.0% |
| nVUnxqHL_DT_static | M_C | 151.998 | 65 | 1.5% | 0.3% | 0.1% | 0.0% | 0.0% |
| nVUnxqHL_DT_static | M_A | 151.381 | 65 | 1.5% | 0.0% | 0.0% | 0.0% | 0.0% |
| XRTnTUjU_DT_static | M_C | 172.538 | 60 | 1.7% | 0.2% | 0.0% | 0.0% | 0.0% |
| XRTnTUjU_DT_static | M_A | 172.145 | 60 | 1.7% | 0.0% | 0.0% | 0.0% | 0.0% |
| MYrVxVEM_DT_static | M_C | 161.763 | 63 | 1.6% | 1.1% | 0.7% | 0.5% | 0.3% |
| MYrVxVEM_DT_static | M_A | 286.616 | 113 | 2.9% | 0.0% | 0.0% | 0.0% | 0.0% |
| wVYSGVTT_DT_mvmt | M_C | 45.713 | 14 | 0.4% | 3.3% | 2.7% | 1.9% | 0.8% |
| wVYSGVTT_DT_mvmt | M_A | 30.364 | 11 | 0.3% | 0.0% | 0.0% | 0.0% | 0.0% |
| qNvSMyUS_FS_static | M_C | 227.851 | 61 | 2.3% | 0.2% | 0.0% | 0.0% | 0.0% |
| qNvSMyUS_FS_static | M_A | 227.237 | 61 | 2.3% | 0.0% | 0.0% | 0.0% | 0.0% |
| jtYTdZm3_DT_static | M_C | 15.576 | 2 | 0.1% | 6.4% | 6.1% | 5.5% | 4.9% |
| jtYTdZm3_DT_static | M_A | 27.880 | 10 | 0.3% | 0.0% | 0.0% | 0.0% | 0.0% |

## § Nores Artifact Gate — FS cases (linear-only LF band energy)

> Δ vs M0 LF energy. M_B/M_C: negligible change on 9xjhi (Δ ≈ −0.02 dB) — shadow/URO changes alone do not reduce the artifact. M_A/M_D/M_full_delay: 9xjhi Δ LF = −6.03/−6.03/−5.83 dB — primary closure criterion MET. qNvSMyUS Δ ≈ 0 throughout (secondary/watch case; M0 LF already near-zero, not a primary artifact case).

| Case | M0 | M_R0 | M_B_corrected | M_C | M_A | M_D | M_full_delay |
|------|------|------|------|------|------|------|------|
| 9xjhiFbG_FS_static | 24.80 | +0.06 | -0.02 | -0.02 | -6.03 | -6.03 | -5.83 |
| qNvSMyUS_FS_static | -2.35 | -0.00 | +0.03 | +0.03 | -0.05 | -0.05 | -0.07 |

> Closure rule: Δ LF at M_A or later vs M0 < −0.5 dB → linear layer reduced artifact → can proceed to AECMOS.  
> **9xjhiFbG verdict: CLOSED** — M_A Δ LF = −6.03 dB ✓. Bundle A (H_error parity, instantaneous E²) is the mechanism; linear layer is the primary artifact source. qNvSMyUS is a secondary/watch case (Δ ≈ −0.05 dB; M0 baseline near-zero, not a primary artifact case). Residual-layer-only attribution is NOT supported by the data.

## § M_full_delay vs M_D Delta (delay-event cases)

| Case | event_type | event_frame | mid_utterance | M_D usable_linear | Mfd usable_linear | Δul | M_D coarse_conv | Mfd coarse_conv | Δcc | Mfd nores LF Δvs M0 |
|------|-----------|------------|--------------|-----------------|------------------|-----|----------------|----------------|-----|---------------------|
| 9xjhiFbG_FS_static | delay_first | 69 | YES | 92.4% | 92.4% | +0.0% | 0.0% | 0.0% | +0.0% | -5.83 |
| xFk7igec_DT_mvmt | delay_first | 104 | YES | 96.1% | 96.1% | +0.0% | 0.2% | 0.1% | -0.0% | +1.53 |
| nVUnxqHL_DT_static | delay_first | 119 | YES | 95.9% | 95.9% | +0.0% | 0.1% | 0.1% | +0.0% | +0.15 |
| XRTnTUjU_DT_static | delay_first | 109 | YES | 91.2% | 91.2% | +0.0% | 0.0% | 0.0% | -0.0% | +0.03 |
| MYrVxVEM_DT_static | delay_first | 85 | YES | 95.6% | 95.6% | +0.0% | 0.9% | 1.0% | +0.1% | +0.03 |
| wVYSGVTT_DT_mvmt | delay_first | 75 | YES | 49.5% | 96.3% | +46.9% | 3.0% | 4.4% | +1.3% | -1.08 |
| qNvSMyUS_FS_static | delay_first | 189 | YES | 82.5% | 82.5% | +0.0% | 0.0% | 0.0% | +0.0% | -0.07 |
| jtYTdZm3_DT_static | delay_first | 246 | YES | 97.7% | 97.7% | +0.0% | 5.4% | 5.1% | -0.4% | -1.45 |

## § Bundle D: usable_linear + FA + Delay Events — M_D

| Case | usable_linear_frac | ul_gate3_conv_frac | ul_gate3_ext_frac | FA_consistent_frac | delay_event_count | M_full_delay? |
|------|-------------------|-------------------|------------------|------------------|-----------------|--------------|
| 9xjhiFbG_FS_static | 92.4% | 80.9% | 96.8% | 71.0% | 1 | YES |
| xFk7igec_DT_mvmt | 96.1% | 96.5% | 97.2% | 90.8% | 1 | YES |
| nVUnxqHL_DT_static | 95.9% | 92.9% | 97.2% | 90.3% | 1 | YES |
| XRTnTUjU_DT_static | 91.2% | 18.6% | 96.9% | 0.0% | 1 | YES |
| MYrVxVEM_DT_static | 95.6% | 95.8% | 97.8% | 86.3% | 1 | YES |
| wVYSGVTT_DT_mvmt | 49.5% | 44.5% | 98.0% | 17.5% | 1 | YES |
| qNvSMyUS_FS_static | 82.5% | 53.7% | 93.0% | 0.0% | 1 | YES |
| jtYTdZm3_DT_static | 97.7% | 98.0% | 93.3% | 87.3% | 1 | YES |

## § Full Ladder: usable_linear_frac × step

| Case | M0 | M_R0 | M_B_corrected | M_C | M_A | M_D |
|------|-------|------|------|------|------|------|
| 9xjhiFbG_FS_static | 94.6% | 94.6% | 92.4% | 92.4% | 92.4% | 92.4% |
| xFk7igec_DT_mvmt | 96.1% | 96.1% | 96.1% | 96.1% | 96.1% | 96.1% |
| nVUnxqHL_DT_static | 95.9% | 95.9% | 95.9% | 95.9% | 95.9% | 95.9% |
| XRTnTUjU_DT_static | 40.6% | 40.6% | 91.2% | 91.2% | 91.2% | 91.2% |
| MYrVxVEM_DT_static | 96.8% | 96.8% | 96.8% | 96.8% | 95.6% | 95.6% |
| wVYSGVTT_DT_mvmt | 96.3% | 96.3% | 96.3% | 96.3% | 49.5% | 49.5% |
| qNvSMyUS_FS_static | 82.5% | 82.5% | 84.2% | 84.2% | 82.5% | 82.5% |
| jtYTdZm3_DT_static | 97.8% | 97.8% | 97.8% | 97.8% | 97.7% | 97.7% |

## § Full Ladder: coarse_conv_frac × step

| Case | M0 | M_R0 | M_B_corrected | M_C | M_A | M_D |
|------|-------|------|------|------|------|------|
| 9xjhiFbG_FS_static | 0.0% | 0.0% | 0.1% | 0.1% | 0.0% | 0.0% |
| xFk7igec_DT_mvmt | 0.4% | 0.4% | 0.2% | 0.2% | 0.2% | 0.2% |
| nVUnxqHL_DT_static | 0.4% | 0.5% | 0.2% | 0.2% | 0.1% | 0.1% |
| XRTnTUjU_DT_static | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% |
| MYrVxVEM_DT_static | 2.3% | 2.3% | 0.9% | 0.9% | 0.9% | 0.9% |
| wVYSGVTT_DT_mvmt | 5.5% | 5.4% | 4.3% | 4.3% | 3.0% | 3.0% |
| qNvSMyUS_FS_static | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% |
| jtYTdZm3_DT_static | 5.6% | 5.8% | 5.0% | 5.0% | 5.4% | 5.4% |

## § Full Ladder: H_error_mean (raw, contaminated) × step

> These are raw means including reset/silent frames at init=10000.  
> Use steady_at_ceil_exclN columns in the audit table for clean signal.

| Case | M0 | M_R0 | M_B_corrected | M_C | M_A | M_D |
|------|-------|------|------|------|------|------|
| 9xjhiFbG_FS_static | 35.614 | 35.040 | 22.987 | 22.987 | 18.804 | 18.804 |
| xFk7igec_DT_mvmt | 165.461 | 165.344 | 165.339 | 165.339 | 160.585 | 160.585 |
| nVUnxqHL_DT_static | 152.686 | 152.038 | 151.998 | 151.998 | 151.381 | 151.381 |
| XRTnTUjU_DT_static | 172.612 | 172.566 | 172.538 | 172.538 | 172.145 | 172.145 |
| MYrVxVEM_DT_static | 156.817 | 156.523 | 161.763 | 161.763 | 286.616 | 286.616 |
| wVYSGVTT_DT_mvmt | 68.819 | 45.903 | 45.713 | 45.713 | 30.364 | 30.364 |
| qNvSMyUS_FS_static | 227.943 | 227.866 | 227.851 | 227.851 | 227.237 | 227.237 |
| jtYTdZm3_DT_static | 20.731 | 15.839 | 15.576 | 15.576 | 27.880 | 27.880 |
