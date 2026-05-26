# v3.21 Residual-Stage Attribution Trace

**Date**: 2026-05-26  
**Variants**: M0 (all flags OFF) | A2_only | A2_A3  
**Cases**: 9xjhiFbG (FS_static G1) | qNvSMyUS (FS_static G2) | xFk7igec (DT_mvmt G3) | XRTnTUjU (DT_static opt)  
**Band defs** (n_freqs=257, Δf=31.25 Hz): LF=bins 0–6 (0–188 Hz) | MF=bins 7–64 (219–2000 Hz) | HF=bins 65–256 (2031–8000 Hz)  

---

## Questions Under Investigation

**Status: R0 parity audit COMPLETE (2026-05-26). All R0 items ruled out as causes of A2 regression. A2/A3 regression = Class C (PBFDKF structural) in isolation. Composition verdict still required before NOSHIP — see §Attribution Verdict.**

1. **9xjhi: first downstream divergence point of full echo crash?** → See §Q1. First signal: H_error HF frame 40. reverb_tail collapses 5× (direct_energy 1.88e3→3.67e2). R²/gain diverge frame 119. Root cause: W magnitude drops 5× under A2 → tail_response = direct × avg_decay drops 5×. **R0.4 filter_quality parity gap causes SLOWER adaptation (skip_frac 56%→79%) — NOT amplification. Class C (isolated).**
2. **nores LF improves but full echo worsens — residual-model too-low R², or gain-reason binding wrong?** → See §Q2. R² drops 2.5× (r2_lf_mean M0=9.94e10 → A2=3.99e10). gain_reason='G' throughout — NOT a gain-rule binding issue. Wedge mechanism confirmed: W magnitude drops → reverb model collapses → R² drops → SG opens gain. **Class C (isolated).**
3. **qNvSMy guard regression — same mechanism as 9xjhi?** → See §Q3. Different mechanism: FilterAnalyzer delay_blocks shifts (3.4→2.6) under A2 → wrong partition used as "direct" → avg_decay inflates 3× → R² inflates 3× (r2_lf M0=7.39e8 → A2=2.37e9). R0.1 filter gate NOT the cause (ng_delta constant across variants). **Class C (isolated); FilterAnalyzer delay_blocks shift is structural consequence of A2 changing W energy distribution.**
4. **xFk7 DT regression — same mechanism, or different path?** → See §Q4. Different: A3 fires diverged-leakage false positive in DT (e2_ref > e2_coa → H_error 28×). A2 alone causes FilterAnalyzer collapse (delay_blocks 4→0). R0 gaps not involved. **Class C (isolated); Bundle B/C composition required.**
5. **Residual echo_model.noise_gate_power — int16/fp32 scaling bug?** → See §Q5 and R0.1/R0.2 in r0_parity_audit. Filter gate B-confirmed (27509562 vs AEC3 20075344, 1.37×). BUT: ng_delta is IDENTICAL across M0/A2/A2_A3 — the gate constant mismatch does NOT interact with A2. R0.2 (residual gate) only fires during warmup nonlinear frames.
6. **Render window parity: Python pre=0/post=1 vs AEC3 pre=1/post=1?** → See §Q6 and R0.3. Confirmed A-class parity gap (minor; nonlinear mode only). Candidate flag: `use_aec3_echo_generating_power_window`. NOT a cause of A2 regression.

---

## C. Noise Gate Unit Audit

### Filter update gate vs Residual echo model gate

| Path | Python constant | int16² value | float[-1,1]² | AEC3 value | Ratio Py/AEC3 |
|------|----------------|--------------|--------------|------------|---------------|
| Filter (refined+coarse) | NOISE_GATE_POWER_FLOAT | 27509562 | 0.02562 | 20075344 int16² = 0.01870 | **1.370×** |
| Residual (echo_model) | 27509562.0 (raw) | 27509562 | 0.02562 | 27509.42 (spec) | **1000.0×** |

**Python filter gate == Python residual gate**: YES (same constant)  
**AEC3 filter gate (20075344 int16²) ≠ Python filter gate (27509562 int16²)**: Python is 1.37× higher.  
**Residual gate comparison**: Python uses 27509562 int16² vs AEC3 spec 27509.42. If AEC3 value is in int16² units: ratio 1000× (likely a unit discrepancy — see note below).  

**Note on echo_model.noise_gate_power**: AEC3 reports `echo_model.noise_gate_power=27509.42f` (float). Our Python stores `27509562.0` as int16², and after `_PSD_SCALE` rescaling both the stored constant and x2 (far_psd) are in int16² space, so the comparison IS dimensionally consistent within our system. However, the two constants (filter gate=20075344 int16² vs residual gate=27509562 int16²) differ by a factor of 1.37×. Python uses the SAME scalar (27509562) for both paths — filter gate and residual gate. AEC3 uses DIFFERENT constants for these two paths. Classification: **B — Python unit bug** (wrong constant for filter gate; filter should use 20075344 int16²).

## Render Window Parity

| Param | Python | AEC3 |
|-------|--------|------|
| render_pre_window_size | 0 | 1 |
| render_post_window_size | 1 | 1 |
| render_history_size | 2 (slots) | 3 (slots) |

Python's pre=0 means the echo-generating power window does NOT look back before the current frame. AEC3's pre=1 catches render transients that occurred ONE block before the current block. Effect: in fast-attack FS scenes, Python may underestimate X² for one block → R² drops slightly → marginally less suppression. Classification: **A — AEC3 parity gap** (minor; pre=0 vs pre=1).

---

## Case: 9xjhiFbG (FS_static, Gate G1)


### A. Filter / H_error (per-band means)

| Signal                 | M0 early | M0 mid  | M0 late | A2 early | A2 mid  | A2 late | ΔA2(late)| A2A3 early | A2A3 mid | A2A3 late | ΔA2A3(late) |
|------------------------|----------|---------|---------|----------|---------|---------|---------|----------|---------|-----------|-------------|

| h_lf                   | 31.1366 |  0.6668 |  0.4890 |  0.4786 |  0.3232 |  0.3121 | -0.177 |  0.4103 |  0.2470 |  0.2514 | -0.238 |
| h_mf                   | 36.1652 |  1.5970 |  1.3857 |  1.1238 |  0.8565 |  0.8923 | -0.493 |  0.9423 |  0.7154 |  0.7474 | -0.638 |
| h_hf                   | 42.4766 |  6.5840 |  3.6057 |  2.3129 |  1.7042 |  1.9386 | -1.667 |  1.8160 |  1.2792 |  1.4491 | -2.157 |
| h_at_floor_frac        |  0.0000 |  0.0000 |  0.0000 |  0.0000 |  0.0000 |  0.0000 | +0.000 |  0.0000 |  0.0000 |  0.0000 | +0.000 |
| h_at_ceil_frac         |  0.0205 |  0.0000 |  0.0000 |  0.0000 |  0.0000 |  0.0000 | +0.000 |  0.0000 |  0.0000 |  0.0000 | +0.000 |


### B. e2_refined vs e2_coarse per band

| Signal                 | M0 early | M0 mid  | M0 late | A2 early | A2 mid  | A2 late | ΔA2(late)| A2A3 early | A2A3 mid | A2A3 late | ΔA2A3(late) |
|------------------------|----------|---------|---------|----------|---------|---------|---------|----------|---------|-----------|-------------|

| e2ref_lf               | 98045.0769 |  58.3318 |  51.6684 | 1214.7375 |  68.9783 |  70.4624 | +18.794 | 1208.8983 |  77.3385 |  70.6278 | +18.959 |
| e2ref_mf               | 751663.4603 | 1915.9468 | 1197.9364 | 9321.1170 | 1695.9830 | 1714.4982 | +516.562 | 9369.3237 | 1840.2085 | 1870.3567 | +672.420 |
| e2ref_hf               | 12489.0266 | 680.5475 | 635.7029 | 1372.2954 | 927.4960 | 739.8194 | +104.116 | 1376.3922 | 936.7076 | 746.7182 | +111.015 |
| e2coa_lf               |  92.9096 |  61.0860 |  67.9340 |  93.2322 |  71.4970 |  70.5725 | +2.638 |  95.8411 |  68.4927 |  72.0437 | +4.110 |
| e2coa_mf               | 1575.8342 | 1723.9053 | 1397.4656 | 1458.2707 | 1500.4770 | 1461.0104 | +63.545 | 1521.1758 | 1444.6653 | 1481.3492 | +83.884 |
| e2coa_hf               | 753.5492 | 724.7988 | 700.7643 | 781.7568 | 757.6308 | 734.4601 | +33.696 | 746.1496 | 737.2222 | 749.1631 | +48.399 |


### C. A3 diverged-leakage fire rate (A2_A3 only)

| Signal                 | M0 early | M0 mid  | M0 late | A2 early | A2 mid  | A2 late | ΔA2(late)| A2A3 early | A2A3 mid | A2A3 late | ΔA2A3(late) |
|------------------------|----------|---------|---------|----------|---------|---------|---------|----------|---------|-----------|-------------|

| a3_div_lf              |   n/a |   n/a |   n/a |   n/a |   n/a |   n/a |   n/a |  0.5557 |  0.4956 |  0.5215 |   n/a |
| a3_div_mf              |   n/a |   n/a |   n/a |   n/a |   n/a |   n/a |   n/a |  0.6505 |  0.5951 |  0.5912 |   n/a |
| a3_div_hf              |   n/a |   n/a |   n/a |   n/a |   n/a |   n/a |   n/a |  0.5551 |  0.5171 |  0.5073 |   n/a |


### D. RES chain — mode, ERLE, R²

| Signal                 | M0 early | M0 mid  | M0 late | A2 early | A2 mid  | A2 late | ΔA2(late)| A2A3 early | A2A3 mid | A2A3 late | ΔA2A3(late) |
|------------------------|----------|---------|---------|----------|---------|---------|---------|----------|---------|-----------|-------------|

| usable_linear          |  0.8897 |  1.0000 |  1.0000 |  0.8897 |  0.9344 |  1.0000 | +0.000 |  0.8897 |  0.9344 |  1.0000 | +0.000 |
| refined_conv           |  0.2374 |  0.3282 |  0.4497 |  0.1299 |  0.1564 |  0.2179 | -0.232 |  0.1257 |  0.1439 |  0.1802 | -0.270 |
| coarse_conv            |  0.0014 |  0.0000 |  0.0000 |  0.0000 |  0.0000 |  0.0000 | +0.000 |  0.0028 |  0.0000 |  0.0000 | +0.000 |
| erle_5                 |  1.0416 |  1.1288 |  1.0676 |  1.1799 |  1.1248 |  1.2128 | +0.145 |  1.1578 |  1.1889 |  1.2375 | +0.170 |
| erle_30                |  1.1963 |  1.6321 |  2.4219 |  1.1718 |  1.5509 |  1.7253 | -0.697 |  1.1873 |  1.5671 |  1.5222 | -0.900 |
| erle_100               |  1.1379 |  2.1516 |  3.4055 |  1.2534 |  1.4805 |  1.5951 | -1.810 |  1.1914 |  1.4607 |  1.3833 | -2.022 |
| erle_hf_mean           |  1.1312 |  1.6558 |  2.2042 |  1.2205 |  1.3977 |  1.6174 | -0.587 |  1.2272 |  1.4183 |  1.5117 | -0.692 |
| r2_5                   | 75082939412.8401 | 88878707051.9777 | 86403565603.0391 | 42603147884.8848 | 23796856543.8048 | 36448168997.1844 | -49955396605.855 | 41249296083.2311 | 17213593032.9836 | 38047952260.7374 | -48355613342.302 |
| r2_30                  | 13768603779.9210 | 35713432940.6927 | 25283408727.4190 | 1883319751.6187 | 6249245773.0114 | 7318918548.0447 | -17964490179.374 | 1270787403.1145 | 5714550581.0627 | 8659686008.6257 | -16623722718.793 |
| r2_100                 | 5968960465.8931 | 19225251487.9106 | 4477282180.0447 | 1349600132.9120 | 2491444712.6513 | 2462302089.2626 | -2014980090.782 | 969349893.0483 | 1270066107.8208 | 1621676476.6480 | -2855605703.397 |
| r2_to_s2_ratio         | 2907419709497206467059582619512471552.0000 | 668.9900 | 234.4574 | 2907419709497206467059582619512471552.0000 | 3505371173184357163026498589768548352.0000 | 178.7171 | -55.740 | 2907419709497206467059582619512471552.0000 | 3505371173184357163026498589768548352.0000 | 211.0102 | -23.447 |
| reverb_freq_resp_tail_max |  3.1703 |  5.7471 |  2.9620 |  0.5434 |  0.5249 |  0.5678 | -2.394 |  0.3896 |  0.4189 |  0.6396 | -2.322 |


### E. SuppressionGain — dominant_ne, gain, reason

| Signal                 | M0 early | M0 mid  | M0 late | A2 early | A2 mid  | A2 late | ΔA2(late)| A2A3 early | A2A3 mid | A2A3 late | ΔA2A3(late) |
|------------------------|----------|---------|---------|----------|---------|---------|---------|----------|---------|-----------|-------------|

| is_nearend_state       |  0.0000 |  0.0000 |  0.0000 |  0.0000 |  0.0000 |  0.0000 | +0.000 |  0.0000 |  0.0000 |  0.0000 | +0.000 |
| gain_5                 |  0.4051 |  0.2274 |  0.1748 |  0.4113 |  0.4465 |  0.2243 | +0.049 |  0.4096 |  0.4560 |  0.2376 | +0.063 |
| gain_30                |  0.5059 |  0.2660 |  0.2241 |  0.6294 |  0.5677 |  0.3573 | +0.133 |  0.6726 |  0.5824 |  0.3890 | +0.165 |
| gain_100               |  0.5708 |  0.2293 |  0.2196 |  0.5759 |  0.4878 |  0.2533 | +0.034 |  0.6353 |  0.5311 |  0.2987 | +0.079 |
| min_gain_5             |  0.2090 |  0.2118 |  0.1586 |  0.2177 |  0.2577 |  0.1598 | +0.001 |  0.2167 |  0.2554 |  0.1606 | +0.002 |
| min_gain_100           |  0.2067 |  0.2346 |  0.1941 |  0.2067 |  0.2346 |  0.1941 | +0.000 |  0.2067 |  0.2346 |  0.1941 | +0.000 |
| G_pre_clip_5           |  0.3668 |  0.2148 |  0.1604 |  0.3735 |  0.4202 |  0.1860 | +0.026 |  0.3724 |  0.4148 |  0.2006 | +0.040 |
| G_pre_clip_100         |  0.5413 |  0.2363 |  0.2067 |  0.5229 |  0.4697 |  0.2358 | +0.029 |  0.5953 |  0.5251 |  0.2763 | +0.070 |
| gain_reason_5          | G | G | G | G | G | G |  | G | G | G |  |
| gain_reason_100        | G | G | G | G | G | G |  | G | G | G |  |
| hf_lim_applied         |  1.0000 |  1.0000 |  1.0000 |  1.0000 |  1.0000 |  1.0000 | +0.000 |  1.0000 |  1.0000 |  1.0000 | +0.000 |
| s2_to_x2_ratio_hf      | 231.3952 | 1161.8662 | 152.3454 | 52.5758 | 184.2032 | 21.0486 | -131.297 | 59.0432 | 194.6778 | 23.8936 | -128.452 |

### First Divergence Frame

**A2_only** (N=2188 frames):
- First H_error HF >5% diff: frame 40
- First ERLE-100 >5% diff: frame 386
- First R²-100 >20% diff: frame 119
- First gain-100 >5% diff: frame 119
- **First signal to diverge: H_error at frame 40**

**A2_A3** (N=2188 frames):
- First H_error HF >5% diff: frame 40
- First ERLE-100 >5% diff: frame 386
- First R²-100 >20% diff: frame 119
- First gain-100 >5% diff: frame 119
- **First signal to diverge: H_error at frame 40**

## Case: qNvSMyUS (FS_static_guard, Gate G2)


### A. Filter / H_error (per-band means)

| Signal                 | M0 early | M0 mid  | M0 late | A2 early | A2 mid  | A2 late | ΔA2(late)| A2A3 early | A2A3 mid | A2A3 late | ΔA2A3(late) |
|------------------------|----------|---------|---------|----------|---------|---------|---------|----------|---------|-----------|-------------|

| h_lf                   | 238.9150 |  0.1145 |  0.0882 | 238.8828 |  0.0988 |  0.0906 | +0.002 | 238.8766 |  0.0844 |  0.0722 | -0.016 |
| h_mf                   | 239.0045 |  0.2656 |  0.2103 | 238.8859 |  0.1344 |  0.1113 | -0.099 | 238.8793 |  0.1195 |  0.0964 | -0.114 |
| h_hf                   | 239.6065 |  0.8266 |  0.6280 | 239.5188 |  0.7768 |  0.6857 | +0.058 | 239.4553 |  0.7142 |  0.5296 | -0.098 |
| h_at_floor_frac        |  0.0000 |  0.0000 |  0.0002 |  0.0013 |  0.0001 |  0.0016 | +0.001 |  0.0042 |  0.0019 |  0.0037 | +0.003 |
| h_at_ceil_frac         |  0.0309 |  0.0000 |  0.0000 |  0.0309 |  0.0000 |  0.0000 | +0.000 |  0.0309 |  0.0000 |  0.0000 | +0.000 |


### B. e2_refined vs e2_coarse per band

| Signal                 | M0 early | M0 mid  | M0 late | A2 early | A2 mid  | A2 late | ΔA2(late)| A2A3 early | A2A3 mid | A2A3 late | ΔA2A3(late) |
|------------------------|----------|---------|---------|----------|---------|---------|---------|----------|---------|-----------|-------------|

| e2ref_lf               |   1.3479 |   1.4704 |   1.7645 |   1.6459 |   3.0298 |   4.7868 | +3.022 |   1.7137 |   2.9046 |   3.4514 | +1.687 |
| e2ref_mf               |  81.4456 |  50.6809 |  20.9114 |  80.8784 |  64.0483 |  22.8790 | +1.968 |  80.4305 |  62.6633 |  21.8787 | +0.967 |
| e2ref_hf               |   1.5266 |   0.5934 |   0.4700 |   1.5152 |   0.5363 |   0.4764 | +0.006 |   1.5191 |   0.4996 |   0.4448 | -0.025 |
| e2coa_lf               |   0.4801 |   1.9200 |   1.0188 |   0.6298 |   2.0060 |   0.9755 | -0.043 |   0.5451 |   1.9463 |   1.1413 | +0.123 |
| e2coa_mf               |  80.0738 |  35.9551 |  13.5677 |  80.8974 |  35.5382 |  12.9616 | -0.606 |  80.3397 |  35.6046 |  13.9860 | +0.418 |
| e2coa_hf               |   1.4311 |   0.4123 |   0.3068 |   1.4313 |   0.3649 |   0.2945 | -0.012 |   1.4243 |   0.3766 |   0.3131 | +0.006 |


### C. A3 diverged-leakage fire rate (A2_A3 only)

| Signal                 | M0 early | M0 mid  | M0 late | A2 early | A2 mid  | A2 late | ΔA2(late)| A2A3 early | A2A3 mid | A2A3 late | ΔA2A3(late) |
|------------------------|----------|---------|---------|----------|---------|---------|---------|----------|---------|-----------|-------------|

| a3_div_lf              |   n/a |   n/a |   n/a |   n/a |   n/a |   n/a |   n/a |  0.8209 |  0.7608 |  0.7154 |   n/a |
| a3_div_mf              |   n/a |   n/a |   n/a |   n/a |   n/a |   n/a |   n/a |  0.8026 |  0.7767 |  0.7364 |   n/a |
| a3_div_hf              |   n/a |   n/a |   n/a |   n/a |   n/a |   n/a |   n/a |  0.7877 |  0.7772 |  0.7404 |   n/a |


### D. RES chain — mode, ERLE, R²

| Signal                 | M0 early | M0 mid  | M0 late | A2 early | A2 mid  | A2 late | ΔA2(late)| A2A3 early | A2A3 mid | A2A3 late | ΔA2A3(late) |
|------------------------|----------|---------|---------|----------|---------|---------|---------|----------|---------|-----------|-------------|

| usable_linear          |  0.6020 |  0.9116 |  1.0000 |  0.6020 |  0.9116 |  1.0000 | +0.000 |  0.6020 |  0.9116 |  1.0000 | +0.000 |
| refined_conv           |  0.0000 |  0.0011 |  0.0011 |  0.0000 |  0.0000 |  0.0011 | +0.000 |  0.0000 |  0.0011 |  0.0000 | -0.001 |
| coarse_conv            |  0.0000 |  0.0000 |  0.0000 |  0.0000 |  0.0000 |  0.0000 | +0.000 |  0.0000 |  0.0000 |  0.0000 | +0.000 |
| erle_5                 |  1.0000 |  1.0000 |  1.0000 |  1.0000 |  1.0000 |  1.0000 | +0.000 |  1.0000 |  1.0000 |  1.0000 | +0.000 |
| erle_30                |  1.0000 |  1.0000 |  1.0000 |  1.0000 |  1.0000 |  1.0000 | +0.000 |  1.0000 |  1.0000 |  1.0000 | +0.000 |
| erle_100               |  1.0000 |  1.0000 |  1.0000 |  1.0000 |  1.0000 |  1.0000 | +0.000 |  1.0000 |  1.0000 |  1.0000 | +0.000 |
| erle_hf_mean           |  1.0000 |  1.0000 |  1.0000 |  1.0000 |  1.0000 |  1.0000 | +0.000 |  1.0000 |  1.0000 |  1.0000 | +0.000 |
| r2_5                   | 548440337.2774 | 433490271.9873 | 784599777.2579 | 944668882.5681 | 1602437984.1379 | 2693899429.0662 | +1909299651.808 | 731779925.7171 | 1568736047.9131 | 3119087557.8549 | +2334487780.597 |
| r2_30                  | 85199498.0605 | 296146222.8682 | 120103412.0335 | 51227465.9751 | 133049913.2920 | 74239877.0550 | -45863534.979 | 78384288.4148 | 482429042.2937 | 125013522.8922 | +4910110.859 |
| r2_100                 | 1073820.3691 | 1974157.3627 | 1283778.7248 | 466401.5704 | 1748677.7555 | 1236630.8431 | -47147.882 | 404929.9249 | 1189631.1696 | 906789.4282 | -376989.297 |
| r2_to_s2_ratio         | 3317747028959728063083975365799968768.0000 | 638827933530638717170045081391988736.0000 |  2.6777 | 3317747028959728063083975365799968768.0000 | 638827933530638717170045081391988736.0000 |  1.6597 | -1.018 | 3317747028959728063083975365799968768.0000 | 638827933530638717170045081391988736.0000 |  7.3167 | +4.639 |
| reverb_freq_resp_tail_max |  0.0000 |  0.0039 |  0.0167 |  0.0000 |  0.0000 |  0.0069 | -0.010 |  0.0000 |  0.0082 |  0.0129 | -0.004 |


### E. SuppressionGain — dominant_ne, gain, reason

| Signal                 | M0 early | M0 mid  | M0 late | A2 early | A2 mid  | A2 late | ΔA2(late)| A2A3 early | A2A3 mid | A2A3 late | ΔA2A3(late) |
|------------------------|----------|---------|---------|----------|---------|---------|---------|----------|---------|-----------|-------------|

| is_nearend_state       |  0.0465 |  0.0000 |  0.0000 |  0.0465 |  0.0000 |  0.0000 | +0.000 |  0.0465 |  0.0000 |  0.0000 | +0.000 |
| gain_5                 |  0.4490 |  0.3495 |  0.4198 |  0.4541 |  0.3500 |  0.3966 | -0.023 |  0.4508 |  0.3152 |  0.3235 | -0.096 |
| gain_30                |  0.5811 |  0.3051 |  0.3392 |  0.5404 |  0.3748 |  0.3661 | +0.027 |  0.5669 |  0.2981 |  0.2825 | -0.057 |
| gain_100               |  0.5454 |  0.3241 |  0.3812 |  0.5400 |  0.3300 |  0.3756 | -0.006 |  0.5385 |  0.3286 |  0.3698 | -0.011 |
| min_gain_5             |  0.1541 |  0.2080 |  0.2133 |  0.1532 |  0.2072 |  0.2129 | -0.000 |  0.1535 |  0.2064 |  0.2125 | -0.001 |
| min_gain_100           |  0.1607 |  0.1973 |  0.1882 |  0.1647 |  0.2191 |  0.1955 | +0.007 |  0.1704 |  0.2001 |  0.1886 | +0.000 |
| G_pre_clip_5           |  0.4175 |  0.3273 |  0.3709 |  0.4247 |  0.3327 |  0.3413 | -0.030 |  0.4211 |  0.2573 |  0.2566 | -0.114 |
| G_pre_clip_100         |  0.5026 |  0.2527 |  0.2941 |  0.5025 |  0.2949 |  0.3005 | +0.006 |  0.5016 |  0.2638 |  0.2877 | -0.006 |
| gain_reason_5          | G | G | G | G | G | G |  | G | G | G |  |
| gain_reason_100        | G | G | G | G | G | G |  | G | G | G |  |
| hf_lim_applied         |  0.9535 |  1.0000 |  1.0000 |  0.9535 |  1.0000 |  1.0000 | +0.000 |  0.9535 |  1.0000 |  1.0000 | +0.000 |
| s2_to_x2_ratio_hf      |  0.3159 |  0.7037 |  0.3618 |  0.3441 |  0.8438 |  0.2997 | -0.062 |  0.4156 |  0.8562 |  0.2407 | -0.121 |

### First Divergence Frame

**A2_only** (N=2686 frames):
- First H_error HF >5% diff: frame 67
- First ERLE-100 >5% diff: frame None
- First R²-100 >20% diff: frame 235
- First gain-100 >5% diff: frame 224
- **First signal to diverge: H_error at frame 67**

**A2_A3** (N=2686 frames):
- First H_error HF >5% diff: frame 67
- First ERLE-100 >5% diff: frame None
- First R²-100 >20% diff: frame 235
- First gain-100 >5% diff: frame 126
- **First signal to diverge: H_error at frame 67**

## Case: xFk7igec (DT_mvmt, Gate G3)


### A. Filter / H_error (per-band means)

| Signal                 | M0 early | M0 mid  | M0 late | A2 early | A2 mid  | A2 late | ΔA2(late)| A2A3 early | A2A3 mid | A2A3 late | ΔA2A3(late) |
|------------------------|----------|---------|---------|----------|---------|---------|---------|----------|---------|-----------|-------------|

| h_lf                   | 166.8768 |  0.2772 |  0.2961 | 149.1376 |  0.0983 |  0.1316 | -0.164 | 149.1060 |  0.0678 |  8.9439 | +8.648 |
| h_mf                   | 168.9019 |  1.6630 |  1.1487 | 149.2505 |  0.1991 |  0.2251 | -0.924 | 149.1916 |  0.1475 |  9.7647 | +8.616 |
| h_hf                   | 169.7921 |  1.3839 |  1.1872 | 149.4679 |  0.3811 |  0.3246 | -0.863 | 149.3811 |  0.2737 |  9.2104 | +8.023 |
| h_at_floor_frac        |  0.0000 |  0.0000 |  0.0000 |  0.0002 |  0.0000 |  0.0002 | +0.000 |  0.0011 |  0.0039 |  0.0010 | +0.001 |
| h_at_ceil_frac         |  0.0367 |  0.0000 |  0.0000 |  0.0205 |  0.0000 |  0.0000 | +0.000 |  0.0205 |  0.0000 |  0.0077 | +0.008 |


### B. e2_refined vs e2_coarse per band

| Signal                 | M0 early | M0 mid  | M0 late | A2 early | A2 mid  | A2 late | ΔA2(late)| A2A3 early | A2A3 mid | A2A3 late | ΔA2A3(late) |
|------------------------|----------|---------|---------|----------|---------|---------|---------|----------|---------|-----------|-------------|

| e2ref_lf               |  19.6780 |   6.3343 |  13.2039 |   4.6766 |   6.3343 |  15.0765 | +1.873 |   4.7867 |   6.3343 |  39.1543 | +25.950 |
| e2ref_mf               | 2335.7374 |  25.1100 | 1096.4166 | 170.2147 |  25.1101 |  94.6169 | -1001.800 | 170.3542 |  25.1101 | 213.3471 | -883.069 |
| e2ref_hf               | 2620.3485 |   0.4322 | 958.2151 |   8.8755 |   0.4322 |   8.0774 | -950.138 |   8.2785 |   0.4322 |  60.3338 | -897.881 |
| e2coa_lf               |   4.0085 |   6.3343 |  13.5091 |   2.8450 |   6.3343 |  14.0457 | +0.537 |   2.4630 |   6.3343 |  15.2971 | +1.788 |
| e2coa_mf               | 111.3436 |  25.1101 | 123.1412 |  82.0667 |  25.1101 | 101.6020 | -21.539 |  81.9408 |  25.1101 | 111.3564 | -11.785 |
| e2coa_hf               |  65.7613 |   0.4322 |  33.5150 |   6.0253 |   0.4322 |   5.2874 | -28.228 |   5.8902 |   0.4322 |   7.3659 | -26.149 |


### C. A3 diverged-leakage fire rate (A2_A3 only)

| Signal                 | M0 early | M0 mid  | M0 late | A2 early | A2 mid  | A2 late | ΔA2(late)| A2A3 early | A2A3 mid | A2A3 late | ΔA2A3(late) |
|------------------------|----------|---------|---------|----------|---------|---------|---------|----------|---------|-----------|-------------|

| a3_div_lf              |   n/a |   n/a |   n/a |   n/a |   n/a |   n/a |   n/a |  0.6169 |  0.4996 |  0.5891 |   n/a |
| a3_div_mf              |   n/a |   n/a |   n/a |   n/a |   n/a |   n/a |   n/a |  0.6027 |  0.4824 |  0.5526 |   n/a |
| a3_div_hf              |   n/a |   n/a |   n/a |   n/a |   n/a |   n/a |   n/a |  0.6126 |  0.4930 |  0.6222 |   n/a |


### D. RES chain — mode, ERLE, R²

| Signal                 | M0 early | M0 mid  | M0 late | A2 early | A2 mid  | A2 late | ΔA2(late)| A2A3 early | A2A3 mid | A2A3 late | ΔA2A3(late) |
|------------------------|----------|---------|---------|----------|---------|---------|---------|----------|---------|-----------|-------------|

| usable_linear          |  0.9142 |  1.0000 |  1.0000 |  0.8746 |  1.0000 |  1.0000 | +0.000 |  0.8746 |  1.0000 |  1.0000 | +0.000 |
| refined_conv           |  0.0891 |  0.0000 |  0.0807 |  0.1749 |  0.0000 |  0.1334 | +0.053 |  0.1865 |  0.0000 |  0.1013 | +0.021 |
| coarse_conv            |  0.0099 |  0.0000 |  0.0008 |  0.0091 |  0.0000 |  0.0008 | +0.000 |  0.0099 |  0.0000 |  0.0008 | +0.000 |
| erle_5                 |  1.0000 |  1.0000 |  1.0095 |  1.0040 |  1.0000 |  1.0119 | +0.002 |  1.0036 |  1.0000 |  1.0022 | -0.007 |
| erle_30                |  1.0035 |  1.0000 |  1.0101 |  1.2043 |  1.4013 |  1.1752 | +0.165 |  1.1014 |  1.1299 |  1.0443 | +0.034 |
| erle_100               |  1.0583 |  1.3641 |  2.8452 |  1.5400 |  3.7151 |  3.2849 | +0.440 |  1.4931 |  2.8735 |  2.0669 | -0.778 |
| erle_hf_mean           |  1.0882 |  1.2658 |  1.5606 |  1.2073 |  1.7897 |  2.0208 | +0.460 |  1.2330 |  1.8458 |  1.4879 | -0.073 |
| r2_5                   | 35158589630.0819 |  0.0000 | 15989990330.5954 | 3775727897.3975 |  0.0000 | 6350231351.5217 | -9639758979.074 | 4130489952.2848 |  0.0000 | 21863381081.1026 | +5873390750.507 |
| r2_30                  | 21407938693.7955 |  0.0000 | 7057200608.2577 | 2104073604.1085 |  0.0000 | 10441321206.7998 | +3384120598.542 | 1952949306.7908 |  0.0000 | 7385558174.7331 | +328357566.475 |
| r2_100                 | 2573733934.1197 |  0.0000 | 201695343.8998 | 189697078.4121 |  0.0000 | 204326534.3586 | +2631190.459 | 182285739.4552 |  0.0000 | 263526260.6202 | +61830916.720 |
| r2_to_s2_ratio         | 11684247291878073724937734814630412288.0000 |  0.0000 | 21301.4237 | 11790974145832021899815300283389444096.0000 |  0.0000 | 2277.0685 | -19024.355 | 11790974145832021899815300283389444096.0000 |  0.0000 | 1696.3097 | -19605.114 |
| reverb_freq_resp_tail_max | 429.3549 | 497.5973 | 252.6290 |  0.2448 |  0.4664 |  0.3621 | -252.267 |  0.2161 |  0.2950 |  0.8070 | -251.822 |


### E. SuppressionGain — dominant_ne, gain, reason

| Signal                 | M0 early | M0 mid  | M0 late | A2 early | A2 mid  | A2 late | ΔA2(late)| A2A3 early | A2A3 mid | A2A3 late | ΔA2A3(late) |
|------------------------|----------|---------|---------|----------|---------|---------|---------|----------|---------|-----------|-------------|

| is_nearend_state       |  0.0000 |  0.0536 |  0.0000 |  0.0437 |  0.0536 |  0.0659 | +0.066 |  0.0437 |  0.0536 |  0.0000 | +0.000 |
| gain_5                 |  0.2267 |  1.0000 |  0.3522 |  0.4297 |  1.0000 |  0.5012 | +0.149 |  0.4296 |  1.0000 |  0.4951 | +0.143 |
| gain_30                |  0.3255 |  1.0000 |  0.3640 |  0.4370 |  1.0000 |  0.2634 | -0.101 |  0.4495 |  1.0000 |  0.2691 | -0.095 |
| gain_100               |  0.2236 |  1.0000 |  0.2770 |  0.3346 |  1.0000 |  0.2507 | -0.026 |  0.3167 |  1.0000 |  0.3167 | +0.040 |
| min_gain_5             |  0.1308 |  1.0000 |  0.1667 |  0.1600 |  1.0000 |  0.1954 | +0.029 |  0.1588 |  1.0000 |  0.1991 | +0.032 |
| min_gain_100           |  0.1147 |  1.0000 |  0.1660 |  0.1213 |  1.0000 |  0.1773 | +0.011 |  0.1213 |  1.0000 |  0.1886 | +0.023 |
| G_pre_clip_5           |  0.1829 |  1.0000 |  0.2716 |  0.3298 |  1.0000 |  0.4174 | +0.146 |  0.3316 |  1.0000 |  0.4353 | +0.164 |
| G_pre_clip_100         |  0.1880 |  1.0000 |  0.2080 |  0.2732 |  1.0000 |  0.2000 | -0.008 |  0.2538 |  1.0000 |  0.2334 | +0.025 |
| gain_reason_5          | G | G | G | G | G | G |  | G | G | G |  |
| gain_reason_100        | G | G | G | G | G | G |  | G | G | G |  |
| hf_lim_applied         |  1.0000 |  0.9464 |  1.0000 |  0.9563 |  0.9464 |  0.9341 | -0.066 |  0.9563 |  0.9464 |  1.0000 | +0.000 |
| s2_to_x2_ratio_hf      | 1593632.8520 | 37.1477 | 998.5586 | 855.0654 |  0.5314 |  2.1007 | -996.458 | 444.1451 |  0.5454 | 230.6209 | -767.938 |

### First Divergence Frame

**A2_only** (N=3678 frames):
- First H_error HF >5% diff: frame 76
- First ERLE-100 >5% diff: frame 342
- First R²-100 >20% diff: frame 144
- First gain-100 >5% diff: frame 144
- **First signal to diverge: H_error at frame 76**

**A2_A3** (N=3678 frames):
- First H_error HF >5% diff: frame 76
- First ERLE-100 >5% diff: frame 337
- First R²-100 >20% diff: frame 145
- First gain-100 >5% diff: frame 145
- **First signal to diverge: H_error at frame 76**

## Case: XRTnTUjU (DT_static, Gate opt)


### A. Filter / H_error (per-band means)

| Signal                 | M0 early | M0 mid  | M0 late | A2 early | A2 mid  | A2 late | ΔA2(late)| A2A3 early | A2A3 mid | A2A3 late | ΔA2A3(late) |
|------------------------|----------|---------|---------|----------|---------|---------|---------|----------|---------|-----------|-------------|

| h_lf                   | 174.7569 |  0.1841 |  0.2929 | 174.6961 |  0.0766 |  0.1328 | -0.160 | 174.6735 |  0.0632 |  0.1209 | -0.172 |
| h_mf                   | 174.8195 |  0.2937 |  0.2968 | 174.7181 |  0.1537 |  0.1327 | -0.164 | 174.7023 |  0.1426 |  0.1248 | -0.172 |
| h_hf                   | 175.1345 |  0.3819 |  0.3086 | 175.0679 |  0.3333 |  0.2246 | -0.084 | 175.0313 |  0.3071 |  0.1933 | -0.115 |
| h_at_floor_frac        |  0.0001 |  0.0000 |  0.0000 |  0.0004 |  0.0000 |  0.0002 | +0.000 |  0.0040 |  0.0000 |  0.0019 | +0.002 |
| h_at_ceil_frac         |  0.0241 |  0.0000 |  0.0000 |  0.0241 |  0.0000 |  0.0000 | +0.000 |  0.0241 |  0.0000 |  0.0000 | +0.000 |


### B. e2_refined vs e2_coarse per band

| Signal                 | M0 early | M0 mid  | M0 late | A2 early | A2 mid  | A2 late | ΔA2(late)| A2A3 early | A2A3 mid | A2A3 late | ΔA2A3(late) |
|------------------------|----------|---------|---------|----------|---------|---------|---------|----------|---------|-----------|-------------|

| e2ref_lf               |   0.1664 |   3.0656 |   2.3172 |   0.2561 |   3.0656 |   2.5663 | +0.249 |   0.2181 |   3.0656 |   2.3840 | +0.067 |
| e2ref_mf               |   9.9810 |  78.4457 |  89.9379 |  10.8250 |  78.4458 |  90.3083 | +0.370 |  11.7396 |  78.4458 |  91.8195 | +1.882 |
| e2ref_hf               |  14.2554 |   4.7224 |   8.7503 |  15.6317 |   4.7224 |   7.8452 | -0.905 |  15.6624 |   4.7224 |   7.8030 | -0.947 |
| e2coa_lf               |   0.0221 |   3.0656 |   2.3227 |   0.0181 |   3.0656 |   2.3190 | -0.004 |   0.0181 |   3.0656 |   2.3162 | -0.006 |
| e2coa_mf               |   4.0762 |  78.4458 |  92.6039 |   4.0386 |  78.4458 |  92.9496 | +0.346 |   3.9712 |  78.4458 |  92.8545 | +0.251 |
| e2coa_hf               |  11.8273 |   4.7224 |   7.4771 |  11.8417 |   4.7224 |   7.4042 | -0.073 |  11.8345 |   4.7224 |   7.5203 | +0.043 |


### C. A3 diverged-leakage fire rate (A2_A3 only)

| Signal                 | M0 early | M0 mid  | M0 late | A2 early | A2 mid  | A2 late | ΔA2(late)| A2A3 early | A2A3 mid | A2A3 late | ΔA2A3(late) |
|------------------------|----------|---------|---------|----------|---------|---------|---------|----------|---------|-----------|-------------|

| a3_div_lf              |   n/a |   n/a |   n/a |   n/a |   n/a |   n/a |   n/a |  0.6177 |  0.0990 |  0.5069 |   n/a |
| a3_div_mf              |   n/a |   n/a |   n/a |   n/a |   n/a |   n/a |   n/a |  0.6088 |  0.0988 |  0.4732 |   n/a |
| a3_div_hf              |   n/a |   n/a |   n/a |   n/a |   n/a |   n/a |   n/a |  0.5937 |  0.0977 |  0.4834 |   n/a |


### D. RES chain — mode, ERLE, R²

| Signal                 | M0 early | M0 mid  | M0 late | A2 early | A2 mid  | A2 late | ΔA2(late)| A2A3 early | A2A3 mid | A2A3 late | ΔA2A3(late) |
|------------------------|----------|---------|---------|----------|---------|---------|---------|----------|---------|-----------|-------------|

| usable_linear          |  0.4613 |  0.0000 |  0.7711 |  0.4613 |  0.0000 |  0.7711 | +0.000 |  0.4613 |  0.0000 |  0.7711 | +0.000 |
| refined_conv           |  0.0496 |  0.0000 |  0.0000 |  0.0157 |  0.0000 |  0.0000 | +0.000 |  0.0148 |  0.0000 |  0.0000 | +0.000 |
| coarse_conv            |  0.0000 |  0.0000 |  0.0000 |  0.0000 |  0.0000 |  0.0000 | +0.000 |  0.0000 |  0.0000 |  0.0000 | +0.000 |
| erle_5                 |  1.0000 |  1.0000 |  1.0000 |  1.0000 |  1.0000 |  1.0000 | +0.000 |  1.0000 |  1.0000 |  1.0000 | +0.000 |
| erle_30                |  1.0000 |  1.0000 |  1.0000 |  1.0000 |  1.0000 |  1.0000 | +0.000 |  1.0000 |  1.0000 |  1.0000 | +0.000 |
| erle_100               |  1.0000 |  1.0000 |  1.0000 |  1.0000 |  1.0000 |  1.0000 | +0.000 |  1.0000 |  1.0000 |  1.0000 | +0.000 |
| erle_hf_mean           |  1.0000 |  1.0000 |  1.0000 |  1.0000 |  1.0000 |  1.0000 | +0.000 |  1.0000 |  1.0000 |  1.0000 | +0.000 |
| r2_5                   | 32658607.9312 |  0.0000 | 58011269.4845 | 73134993.6880 |  0.0000 | 161507401.0069 | +103496131.522 | 69309373.9542 |  0.0000 | 95963128.1924 | +37951858.708 |
| r2_30                  | 53263482.9997 |  0.0000 | 15977310.0474 | 54342105.1723 |  0.0000 | 31657865.7845 | +15680555.737 | 45524653.3791 |  0.0000 | 21887410.1964 | +5910100.149 |
| r2_100                 | 220646734.9262 |  0.0000 | 102840440.1567 | 323973353.5952 |  0.0000 | 52399209.1205 | -50441231.036 | 379026559.4095 |  0.0000 | 52788618.2384 | -50051821.918 |
| r2_to_s2_ratio         | 19670406865397988898438247723531501568.0000 |  0.0000 |  0.8117 | 19670406865397988898438247723531501568.0000 |  0.0000 |  0.8158 | +0.004 | 19670406865397988898438247723531501568.0000 |  0.0000 |  0.8134 | +0.002 |
| reverb_freq_resp_tail_max |  0.0054 |  0.0000 |  0.0000 |  0.0150 |  0.0000 |  0.0000 | +0.000 |  0.0130 |  0.0000 |  0.0000 | +0.000 |


### E. SuppressionGain — dominant_ne, gain, reason

| Signal                 | M0 early | M0 mid  | M0 late | A2 early | A2 mid  | A2 late | ΔA2(late)| A2A3 early | A2A3 mid | A2A3 late | ΔA2A3(late) |
|------------------------|----------|---------|---------|----------|---------|---------|---------|----------|---------|-----------|-------------|

| is_nearend_state       |  0.0000 |  0.3046 |  0.0444 |  0.0000 |  0.3046 |  0.0888 | +0.044 |  0.0000 |  0.3046 |  0.1036 | +0.059 |
| gain_5                 |  0.6256 |  1.0000 |  0.8708 |  0.6020 |  1.0000 |  0.8149 | -0.056 |  0.6221 |  1.0000 |  0.8202 | -0.051 |
| gain_30                |  0.6321 |  1.0000 |  0.8986 |  0.6201 |  1.0000 |  0.8703 | -0.028 |  0.6111 |  1.0000 |  0.8805 | -0.018 |
| gain_100               |  0.5710 |  1.0000 |  0.6739 |  0.5654 |  1.0000 |  0.6366 | -0.037 |  0.5566 |  1.0000 |  0.6411 | -0.033 |
| min_gain_5             |  0.4377 |  1.0000 |  0.3659 |  0.4363 |  1.0000 |  0.3432 | -0.023 |  0.4364 |  1.0000 |  0.3469 | -0.019 |
| min_gain_100           |  0.3943 |  1.0000 |  0.3238 |  0.3907 |  1.0000 |  0.3040 | -0.020 |  0.3906 |  1.0000 |  0.2879 | -0.036 |
| G_pre_clip_5           |  0.6009 |  1.0000 |  0.8623 |  0.5707 |  1.0000 |  0.8167 | -0.046 |  0.5920 |  1.0000 |  0.8133 | -0.049 |
| G_pre_clip_100         |  0.5489 |  1.0000 |  0.6815 |  0.5442 |  1.0000 |  0.6372 | -0.044 |  0.5331 |  1.0000 |  0.6460 | -0.035 |
| gain_reason_5          | G | G | G | G | G | G |  | G | G | G |  |
| gain_reason_100        | G | G | G | G | G | G |  | G | G | G |  |
| hf_lim_applied         |  1.0000 |  0.6954 |  0.9556 |  1.0000 |  0.6954 |  0.9112 | -0.044 |  1.0000 |  0.6954 |  0.8964 | -0.059 |
| s2_to_x2_ratio_hf      | 4166198367523669208255692800.0000 | 5712362878459032853695954944.0000 | 1208708870400544156275441664.0000 | 1686419641600680022029369344.0000 | 2262116311761583782104137728.0000 | 478784254613495342662942720.0000 | -729924615787048882331975680.000 | 1405656670063759961251381248.0000 | 1823707799013019751983087616.0000 | 386144053612432874482434048.0000 | -822564816788111281793007616.000 |

### First Divergence Frame

**A2_only** (N=3487 frames):
- First H_error HF >5% diff: frame 76
- First ERLE-100 >5% diff: frame None
- First R²-100 >20% diff: frame 257
- First gain-100 >5% diff: frame 94
- **First signal to diverge: H_error at frame 76**

**A2_A3** (N=3487 frames):
- First H_error HF >5% diff: frame 76
- First ERLE-100 >5% diff: frame None
- First R²-100 >20% diff: frame 257
- First gain-100 >5% diff: frame 94
- **First signal to diverge: H_error at frame 76**

---

## Attribution Verdict

**R0 parity audit COMPLETE (2026-05-26). All R0 blocker items cleared.**  
**A2/A3 regression = Class C (PBFDKF structural mismatch) in isolation on all three regression cases.**  
**Isolation verdict is NOT a NOSHIP close. Full composition (M_B→M_C→M_A) still required before any NOSHIP verdict.**

Key R0 finding that unblocks classification: R0.4 filter_quality parity gap causes `skip_frac` to INCREASE under A2 (56%→79% for 9xjhi), meaning reverb adapts LESS frequently — NOT faster. The binary quality proxy is non-blocking for the A2 regression verdict. The tail collapse is driven entirely by W magnitude reduction (direct_energy −5×), which is a PBFDKF-structural consequence of A2's E² denominator change.

| Case | First divergence signal | Frame | Crash-site observation | **Isolation verdict** | R0 clearance |
|------|------------------------|-------|------------------------|-----------------------|-------------|
| 9xjhiFbG (FS_static G1) | H_error HF | 40 | W direct_energy 1.88e3→3.67e2 (−5×) → tail_response −5× → R²↓ → gain↑ → echo leaks | **C (isolated; composition required)** | R0.4 CLEARED: skip_frac 56%→79% (slower, not faster); R0.1 CLEARED: ng_delta constant across all variants |
| qNvSMyUS (FS_static G2) | H_error HF | 67 | delay_blocks shift 3.4→2.6 → wrong reverb partition → avg_decay inflates 3× → R² over-estimate | **C (isolated; composition required)** | R0.1 CLEARED: ng_delta constant; delay_blocks shift is structural consequence of A2 W energy redistribution |
| xFk7igec (DT_mvmt G3) | H_error HF | 76 | delay_blocks collapses 4.0→0.0 under A2 (FilterAnalyzer failure); A3 DT false-positive 62% HF bins; H_error 28× inflation | **C (isolated; composition required)** | R0 ALL CLEARED: FilterAnalyzer collapse and A3 DT false-positive both structural consequences of A2 W change |
| XRTnTUjU (DT_static opt) | H_error HF | 76 | Small gain deltas only; no echo crash observed | **D** | R0 not blocking; mechanism not isolable from current trace granularity |

### Classification Legend
- **A** — AEC3 parity gap (can close in v3.21 without changing algorithm)
- **B** — Python unit/scale bug (can close in v3.21)
- **C (isolated)** — PBFDKF structural mismatch confirmed in isolation; NOT a NOSHIP close; full composition (M_B→M_C→M_A) required before verdict
- **D** — evidence insufficient (need finer trace)

---

## Q&A Analysis

### Q1. 9xjhi — First downstream divergence point of full echo crash

**Answer**: H_error HF diverges at frame 40. First echo-path signal: reverb_freq_resp_tail_max collapses starting frames 40–119; R²-100 and gain-100 diverge at frame 119. ERLE-100 diverges frame 386 (secondary — 267 frames after R²/gain).

**Causal chain (confirmed)**:
1. A2 switches the E² denominator in mu from 0.95-EMA `_error_psd` to instantaneous per-bin `|error_spec|²`. The instantaneous version is noisier.
2. Different mu → W evolves on a different trajectory from frame 40.
3. `ReverbFrequencyResponse.tail_response` is estimated from |W|² per partition. Collapse: reverb_freq_resp_tail_max = M0=2.96 → A2=0.57 → A2_A3=0.64 (late phase).
4. R² = S²/ERLE + reverb_tail. Reverb_tail is dominant for 9xjhi (FS_static, reverberant). Collapse: r2_100 late M0=4.48B → A2=2.46B → A2_A3=1.62B.
5. Lower R² → SG gain rises: gain_100 late M0=0.220 → A2=0.253 → A2_A3=0.299.
6. Higher gain → more echo passes → AECMOS echo regression.

`gain_reason='G'` throughout — NOT a min_gain or limiter issue.

**R0.4 RESOLVED (non-blocking)**: R0 trace shows `skip_frac` INCREASES under A2 (56%→79% for 9xjhi) — Python's binary filter_quality causes the reverb model to update LESS frequently, not more. The tail collapse is NOT amplified by faster binary updates; it reflects genuine W magnitude reduction (direct_energy 1.88e3→3.67e2). R0.4 parity gap (binary vs continuous consistent_estimate) is a Class A substrate item but is NOT a causal path for the A2 regression. Classification: **Class C (isolated)** — W trajectory change is PBFDKF structural. Full composition still required before NOSHIP.

---

### Q2. Why nores LF improves but full AECMOS echo worsens

**Answer**: The two outputs diverge because the reverb model creates a wedge between direct-path echo cancellation quality and post-filter echo residual estimate.

- **nores (linear output = filter error)**: A2's per-bin instantaneous E² makes mu more precise per-bin in LF steady state. H_error_lf: M0=0.49 → A2=0.31. Direct echo subtraction improves → nores LF energy decreases.
- **Full output (post-filter)**: ResidualEchoEstimator receives reverb_freq_resp_tail_max as input to R². The reverb model tracks |W|² — when W changes trajectory under A2, tail_response collapses even though the actual room hasn't changed. SG thinks "less residual echo" → higher gain → actual residual passes through.
- **The wedge**: Direct cancellation improves in the filter domain; but the reverb echo power estimate in the residual domain sees a different (collapsed) tail → SG under-suppresses the remaining echo.

**Root cause RESOLVED (Class C)**: R0.4 trace confirms binary filter_quality causes SLOWER reverb updates (skip_frac 56%→79%). The tail collapse is not a filter_quality parity artifact — it is a genuine W trajectory change caused by A2's E² denominator switch. Direct echo cancellation improves (H_error_lf: 0.49→0.31) but the reverb model loses its reference because the |W|² partition landscape shifts. Classification: **Class C (isolated)** — PBFDKF structural; wedge mechanism confirmed.

---

### Q3. qNvSMy guard regression — same mechanism or different?

**Answer**: Same family (A2 changes W) but different path — reverb is near-zero for qNvSMy; mechanism is direct LF e2_ref rise.

**Trace evidence**: e2ref_lf late: M0=1.76 → A2=4.79 → A2_A3=3.45. e2coa_lf stable (M0=1.02 → A2=0.98). ERLE=1.0 throughout. reverb_tail≈0.

**Causal chain**: A2's instantaneous E² denominator introduces per-frame noise into mu at low-echo LF bins. Erratic LF mu → refined filter oscillates at LF rather than converging → e2_ref_lf rises. Shadow filter (unaffected by A2) maintains stable LF performance. Full-output path uses refined error → LF echo regression.

**Smaller magnitude vs 9xjhi**: No reverb_tail amplifier. Only the direct S²/ERLE path is active.

**R0.1 RESOLVED (non-blocking for A2 verdict)**: R0 trace shows `ng_delta` (extra bins zeroed by Python vs AEC3 gate threshold) is IDENTICAL across M0/A2/A2_A3 variants. A2 changes the E² denominator in mu — it does NOT change X², so it has zero interaction with the filter noise gate threshold. The LF mu instability is structural (A2's instantaneous E² is noiser at low-echo bins) — not gate-related. Classification: **Class C (isolated)**. R0.1 filter gate constant remains a standalone B-class candidate (`use_aec3_filter_noise_gate_power`) independent of A2.

---

### Q4. xFk7 DT regression — same mechanism or different?

**Answer**: DIFFERENT mechanism from the FS cases. Primary failure is A3's per-bin diverged-leakage gate producing false positives in DT context.

**Causal chain (confirmed)**:
1. In doubletalk: mic = echo_path(far) + nearend. error_spec = mic − echo_estimate = residual_echo + nearend.
2. A3 gate: `e2_ref > e2_coa` → "diverged" → diverged leakage (0.25/hop vs 0.0025/hop, 100×).
3. Shadow filter (e2_coa source) adapts on far-end path; during DT, e2_ref (includes nearend) >> e2_coa (echo-only) → gate fires false positive at 62% HF bins.
4. H_error inflates: h_hf A2_only=0.325 → A2_A3=9.123 (28×). h_at_ceil_frac=0.0077 (A2_A3).
5. High H_error → large mu → filter partially adapts to nearend (filter corruption).
6. reverb_tail collapses (M0=252.6 → A2=0.36 → A2_A3=0.81). Δdeg=−0.836.

**A3's structural conflict in DT**: The gate `e2_ref > e2_coa` cannot separate "filter diverged" from "nearend inflating e2_ref." No threshold change fixes this — requires DT detection gating A3, which means Bundle B must be present (shadow gates provide better e2_coa signal).

**DEFERRED reason**: A3's false-positive rate (62% HF) is structural in DT context. But the full-composition effect (whether Bundle B shadow gates reduce e2_coa noise and reduce false-positive rate) is untested. Verdict requires M_B → M_C → M_A composition.

---

### Q5. Residual echo_model.noise_gate_power — int16/fp32 scaling bug?

**Answer**: Filter gate is a confirmed B-class bug. Residual gate is a B-class candidate requiring unit verification.

- **Filter gate (B-confirmed)**: Python NOISE_GATE_POWER_FLOAT = 27509562 int16² (0.02562 float); AEC3 filter.coarse.noise_gate = 20075344 int16² (0.01870 float). Python 1.37× higher → gates more bins → fewer W update bins.
- **Residual gate (B-candidate)**: Python EchoModelConfig.noise_gate_power = 27509562.0 (int16²); AEC3 echo_model.noise_gate_power = 27509.42f. Ratio ~1000×. If both in int16² space, Python residual gate is far more aggressive than AEC3 (zeros more nonlinear-mode bins). Only affects `usable_linear=False` frames. Requires AEC3 source unit verification before declaring bug.

See R0.1 and R0.2 in docs/v3_21_residual_r0_parity_audit.md.

---

### Q6. Render window parity: Python pre=0/post=1 vs AEC3 pre=1/post=1

**Answer**: Python pre=0, post=1 → history_size=2. AEC3 pre=1, post=1 → history_size=3. Python misses one pre-block of X² when estimating echo generating power in nonlinear mode.

**Effect**: One-block underestimation on fast-attack FS scenes; transient-relevant, steady-state-minor.

**Classification: A — AEC3 parity gap.** Candidate flag: `use_aec3_echo_generating_power_window`. See R0.3 in docs/v3_21_residual_r0_parity_audit.md.

---

## Summary: Crash-Site and Parity Classification Table

**R0 audit complete 2026-05-26. All R0 blocker items resolved.**

| Finding | Crash-site classification | A2/A3 verdict | R0 status |
|---------|--------------------------|---------------|-----------|
| 9xjhi echo crash: W direct_energy −5× → tail_response −5× → R²↓ → gain↑ | reverb/SG confirmed; W magnitude drop is the root signal | **C (isolated)** | R0.4 CLEARED: skip_frac ↑ (slower); R0.1 CLEARED: ng_delta invariant |
| nores improves but full echo worsens | wedge confirmed: direct cancel improves (H_error_lf ↓); reverb model loses reference (W landscape shifts) | **C (isolated)** | CLEARED: binary quality causes slower updates — wedge is pure PBFDKF structural |
| qNvSMy guard: delay_blocks 3.4→2.6 → avg_decay 3× inflate → R² over-estimate | FilterAnalyzer partition-selection structural; direct LF path affected | **C (isolated)** | R0.1 CLEARED: ng_delta invariant; delay_blocks shift is W-energy redistribution consequence |
| xFk7 DT: A3 DT false-positive (62% HF), delay_blocks 4→0, H_error 28× | FilterAnalyzer collapse + A3 DT gate conflict structural | **C (isolated)** | ALL CLEARED; requires Bundle B composition for full-composition verdict |
| XRTnTUjU DT static opt | Small gain deltas only | **D** | R0 not blocking; need finer per-bin trace |
| Filter noise gate constant (R0.1) | — | **B-confirmed (standalone)** | Python 27509562 vs AEC3 20075344 (1.37×); NOT cause of A2 regression; candidate `use_aec3_filter_noise_gate_power` |
| Residual noise gate constant (R0.2) | — | **B-candidate (standalone)** | Python 27509562 vs AEC3 27509.42 (~1000× if same units); warmup-only effect |
| Render pre-window (R0.3) | — | **A (standalone)** | Python pre=0 vs AEC3 pre=1; nonlinear mode only; candidate `use_aec3_echo_generating_power_window` |
| ReverbFrequencyResponse filter_quality (R0.4) | — | **A (standalone; non-blocking)** | Binary 0/1 vs AEC3 fractional consistent_estimate; causes SLOWER updates under A2 (non-blocking for A2 verdict); long-term port deferred |

**A2/A3 isolation verdict: Class C (PBFDKF structural mismatch) on all three regression cases.**  
**This is NOT a NOSHIP close.** Composition verdict requires M_B→M_C→M_A ladder (hazy-lynx plan). Standalone B/A items (R0.1/R0.2/R0.3/R0.4) are independent of composition and can be proposed as default-OFF candidates on a separate track.
