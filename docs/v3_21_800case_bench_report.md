# v3.21 800-case Benchmark Report — C1-C6 wall-clock alignment (M_full_delay = all-ON) vs M0 (all-OFF == plain BALANCED)

**Date**: 2026-05-29  
**Cases**: 695/800  
**Config**: preset=balanced / filter=832 (52ms) / cng / hop=160 (workers-count does not affect scores)

**Byte-equal precheck** (M0 vs plain BALANCED): PASS ✓

## Flag Composition

> C1/C3/C4/C6 wall-clock EMA/IIR alignment flags (per-4ms-block AEC3 constants → per-10ms-hop).

| Flag | M0 | M_full_delay |
|------|-----|--------------|
| `use_aec3_wallclock_subband_erle_smoothing` | OFF | ON |
| `use_aec3_wallclock_fullband_erle_smoothing` | OFF | ON |
| `use_aec3_wallclock_low_noise_render_iir` | OFF | ON |
| `use_aec3_active_render_threshold_shadow_epc` | OFF | ON |

---

## Production Ledger (M_full_delay vs M0 = v3.21.6 anchor)

### Bucket Means

| Bucket | Metric | N | Δ_mean | Δ_std | Worst Δ | Best Δ |
|--------|--------|---|--------|-------|---------|--------|
| DT_mvmt | deg | 114 | +0.048 | 0.146 | -0.379 | +0.561 |
| DT_static | deg | 186 | +0.026 | 0.153 | -0.479 | +1.102 |
| FS_mvmt | echo | 131 | -0.054 | 0.136 | -1.039 | +0.269 |
| FS_static | echo | 169 | -0.049 | 0.192 | -1.514 | +0.998 |
| NE | deg | 95 | +0.005 | 0.024 | -0.044 | +0.144 |

### Worst-5 per Bucket (Δ vs M0)

**DT_mvmt** (metric=deg):
| Case | M0 | M_full | Δ |
|------|----|--------|---|
| `IrQvqOTCmEWMXn9k2ICtRQ_doubletalk_with_movement` | 2.631 | 2.253 | -0.379 |
| `QEeKiaNiDECfqXTRrDFWWw_doubletalk_with_movement` | 3.263 | 2.912 | -0.350 |
| `wY00iJ3cE0aQsjt0m1tC0g_doubletalk_with_movement` | 1.984 | 1.644 | -0.340 |
| `S22FCqKDWUyymN1YbpItIw_doubletalk_with_movement` | 2.733 | 2.547 | -0.186 |
| `oQK3bVihI0qel9As840Zzw_doubletalk_with_movement` | 2.642 | 2.466 | -0.176 |

**DT_static** (metric=deg):
| Case | M0 | M_full | Δ |
|------|----|--------|---|
| `SvVpGsVXY0WeklKK5tnI3Q_doubletalk` | 3.403 | 2.924 | -0.479 |
| `zpiSOkxpHkCs5SqdOo5ZIQ_doubletalk` | 2.008 | 1.618 | -0.390 |
| `JtodX3Ug6Eu5TYu0HN5IOw_doubletalk` | 1.783 | 1.440 | -0.343 |
| `XtUWcfsuykC5fTu7DdAnnw_doubletalk` | 1.998 | 1.662 | -0.336 |
| `OLjlc92QWU6fwuN4ytCPQg_doubletalk` | 2.162 | 1.949 | -0.212 |

**FS_mvmt** (metric=echo):
| Case | M0 | M_full | Δ |
|------|----|--------|---|
| `iOyPaxX11UOaUkcscKhq1A_farend_singletalk_with_move` | 3.093 | 2.054 | -1.039 |
| `4pN9yn7mhEa5iDiKnr5jlw_farend_singletalk_with_move` | 3.927 | 3.522 | -0.406 |
| `XTqo1aOXDEiqyWTFK99I5Q_farend_singletalk_with_move` | 3.128 | 2.803 | -0.325 |
| `tl5UFRCXZkyL6EoWVl09xA_farend_singletalk_with_move` | 3.570 | 3.254 | -0.316 |
| `iyuYIcszXku7BWYOOwqh5Q_farend_singletalk_with_move` | 4.112 | 3.804 | -0.307 |

**FS_static** (metric=echo):
| Case | M0 | M_full | Δ |
|------|----|--------|---|
| `ql7yTcebJU20VE5qpW0kCA_farend_singletalk` | 3.977 | 2.464 | -1.514 |
| `9bSnA8CNBUSsJeIVCELzSQ_farend_singletalk` | 3.172 | 2.598 | -0.574 |
| `pPBidi8oyUarZCGUrKJEsg_farend_singletalk` | 4.320 | 3.831 | -0.490 |
| `wlAXM0iDgkm06i7UdRww1w_farend_singletalk` | 2.746 | 2.285 | -0.462 |
| `TcuegeGKL06ysmFP0RxjyQ_farend_singletalk` | 2.870 | 2.442 | -0.428 |

**NE** (metric=deg):
| Case | M0 | M_full | Δ |
|------|----|--------|---|
| `wcP1HWQqv0aX1WAguwuUPg_nearend_singletalk` | 2.900 | 2.855 | -0.044 |
| `XnfMDZLl0U2WvLRphiGJ6A_nearend_singletalk` | 3.919 | 3.884 | -0.035 |
| `WH0jN3PY40es2S0LsxmkkQ_nearend_singletalk` | 3.879 | 3.858 | -0.021 |
| `OLjlc92QWU6fwuN4ytCPQg_nearend_singletalk` | 3.561 | 3.540 | -0.021 |
| `S73GmH9ok0GBbaG3esxbQQ_nearend_singletalk` | 3.941 | 3.921 | -0.020 |

### Catastrophic Cases (vs M0)

> DT Δdeg < −0.20  OR  FS Δecho < −0.20 vs M0  (both metrics: higher = better)

**40 catastrophic case(s):**

| Case | Bucket | Δ |
|------|--------|---|
| `ql7yTcebJU20VE5qpW0kCA_farend_singletalk` | FS_static | -1.514 |
| `iOyPaxX11UOaUkcscKhq1A_farend_singletalk_with_move` | FS_mvmt | -1.039 |
| `9bSnA8CNBUSsJeIVCELzSQ_farend_singletalk` | FS_static | -0.574 |
| `pPBidi8oyUarZCGUrKJEsg_farend_singletalk` | FS_static | -0.490 |
| `SvVpGsVXY0WeklKK5tnI3Q_doubletalk` | DT_static | -0.479 |
| `wlAXM0iDgkm06i7UdRww1w_farend_singletalk` | FS_static | -0.462 |
| `TcuegeGKL06ysmFP0RxjyQ_farend_singletalk` | FS_static | -0.428 |
| `4pN9yn7mhEa5iDiKnr5jlw_farend_singletalk_with_move` | FS_mvmt | -0.406 |
| `zpiSOkxpHkCs5SqdOo5ZIQ_doubletalk` | DT_static | -0.390 |
| `IrQvqOTCmEWMXn9k2ICtRQ_doubletalk_with_movement` | DT_mvmt | -0.379 |
| `o2wfdvOGwU6M8Fmn2dCvOA_farend_singletalk` | FS_static | -0.368 |
| `QEeKiaNiDECfqXTRrDFWWw_doubletalk_with_movement` | DT_mvmt | -0.350 |
| `JtodX3Ug6Eu5TYu0HN5IOw_doubletalk` | DT_static | -0.343 |
| `MnzrTJXYiUCPRMjVzLk25w_farend_singletalk` | FS_static | -0.341 |
| `wY00iJ3cE0aQsjt0m1tC0g_doubletalk_with_movement` | DT_mvmt | -0.340 |
| `XtUWcfsuykC5fTu7DdAnnw_doubletalk` | DT_static | -0.336 |
| `XTqo1aOXDEiqyWTFK99I5Q_farend_singletalk_with_move` | FS_mvmt | -0.325 |
| `tl5UFRCXZkyL6EoWVl09xA_farend_singletalk_with_move` | FS_mvmt | -0.316 |
| `IxgmaPghzUGnR6sxrbGU3Q_farend_singletalk` | FS_static | -0.309 |
| `iyuYIcszXku7BWYOOwqh5Q_farend_singletalk_with_move` | FS_mvmt | -0.307 |
| `kOGPX6kHskOaKSZdLGNz8A_farend_singletalk_with_move` | FS_mvmt | -0.295 |
| `OXCtw0FVhUWGUBil6Uucdw_farend_singletalk` | FS_static | -0.287 |
| `sx6mxKBQpkq520m64BwUdQ_farend_singletalk_with_move` | FS_mvmt | -0.268 |
| `r7U6JmcRl0ibIh0mN3CP9g_farend_singletalk` | FS_static | -0.268 |
| `yZs0i8NpJkypsV8QyvduzQ_farend_singletalk` | FS_static | -0.260 |
| `zONvcX0qYkuaAViV5PXcYg_farend_singletalk_with_move` | FS_mvmt | -0.257 |
| `qu84vhur1UaDJPH2eCmMZA_farend_singletalk` | FS_static | -0.253 |
| `VgSXlJJEI02dytkMm5UTzA_farend_singletalk_with_move` | FS_mvmt | -0.250 |
| `yM2wHof9U06yVPJfemZ3hg_farend_singletalk` | FS_static | -0.249 |
| `JtodX3Ug6Eu5TYu0HN5IOw_farend_singletalk` | FS_static | -0.247 |
| `knHK8NbAlESKGG4ZVuPOLg_farend_singletalk` | FS_static | -0.247 |
| `wlAXM0iDgkm06i7UdRww1w_farend_singletalk_with_move` | FS_mvmt | -0.243 |
| `VGlWeOPC6UiXSq4SYPiKpw_farend_singletalk` | FS_static | -0.238 |
| `s0oJqM6Y1UCHSVmHmgsx4Q_farend_singletalk` | FS_static | -0.230 |
| `MYrVxVEMxkaE7OuyTUmI0Q_farend_singletalk_with_move` | FS_mvmt | -0.226 |
| `DS5pS5fZwEG4X0NNifTh0w_farend_singletalk` | FS_static | -0.223 |
| `IrQvqOTCmEWMXn9k2ICtRQ_farend_singletalk` | FS_static | -0.218 |
| `OLjlc92QWU6fwuN4ytCPQg_doubletalk` | DT_static | -0.212 |
| `pmzLFdKTzEixfU0l0furvA_farend_singletalk` | FS_static | -0.209 |
| `sSC9qgbMXEaS2DtvpALBOQ_farend_singletalk` | FS_static | -0.201 |

---

## Alignment Ledger (M_full_delay vs AEC3 behavioral reference)

> AEC3 scores from `bin/aec3_cli` run on 12-case cohort (2026-05-27).

> Only these 12 cases have known AEC3 reference scores.


| Case | Bucket | Metric | AEC3 | M_full | Δ_vs_AEC3 | Δ_vs_M0 | Status |
|------|--------|--------|------|--------|-----------|---------|--------|
| `ZJYUt0O0AEKSQ9LJ8z7t0A_doubletalk_with_mov` | DT_mvmt | deg | 2.177 | 2.251 | +0.074 | +0.010 | ✓ PASS |
| `wVYSGVTTakih9twI4xlDWQ_doubletalk_with_mov` | DT_mvmt | deg | 1.540 | 2.922 | +1.382 | +0.314 | ✓ PASS |
| `xFk7igecuke0R5JMfREyDg_doubletalk_with_mov` | DT_mvmt | deg | 1.275 | 1.817 | +0.542 | +0.202 | ✓ PASS |
| `MYrVxVEMxkaE7OuyTUmI0Q_doubletalk` | DT_static | deg | 1.275 | 1.460 | +0.185 | +0.147 | ✓ PASS |
| `XRTnTUjU5kS0mejzCqyCiw_doubletalk` | DT_static | deg | 2.062 | 2.072 | +0.010 | +0.056 | ✓ PASS |
| `jtYTdZm3lUmFVNibJWq8YQ_doubletalk` | DT_static | deg | 2.298 | 1.955 | -0.343 | +0.109 | ⚠ FAIL |
| `nVUnxqHLr0GTN7shWid1Ow_doubletalk` | DT_static | deg | 1.547 | 2.072 | +0.525 | +0.068 | ✓ PASS |
| `0I0XMl3M0ECO0U1N0cJvpg_farend_singletalk_w` | FS_mvmt | echo | 4.296 | 4.430 | +0.134 | -0.001 | ✓ PASS |
| `9xjhiFbGo06hdQIsHTS6qA_farend_singletalk` | FS_static | echo | 3.442 | 1.729 | -1.713 | -0.014 | ⚠ KNOWN EXCEPTION |
| `qNvSMyUSXUyrDGpOw7s6qg_farend_singletalk` | FS_static | echo | 3.596 | 4.102 | +0.506 | +0.001 | ✓ PASS |
| `xQEUtY2pWUi7v1X93TF2AA_farend_singletalk` | FS_static | echo | 4.219 | 3.489 | -0.730 | -0.075 | ⚠ FAIL |
| `014AzuqPZku2004NbTTmcA_nearend_singletalk` | NE | deg | 4.164 | 4.393 | +0.229 | +0.001 | ✓ PASS |

### Alignment Bucket Summary

| Bucket | N | Mean Δ_vs_AEC3 | Status |
|--------|---|----------------|--------|
| DT_mvmt | 3 | +0.666 | ✓ PASS |
| DT_static | 4 | +0.094 | ✓ PASS |
| FS_mvmt | 1 | +0.134 | ✓ PASS |
| FS_static | 3 | -0.646 | ⚠ FAIL (9xjhi exception) |
| NE | 1 | +0.229 | ✓ PASS |

### Alignment Catastrophics (vs AEC3 ref)

> DT worse than AEC3 by > 0.10 deg  OR  FS worse than AEC3 by > 0.30 echo

**3 alignment catastrophic(s):**

| Case | Bucket | Δ_vs_AEC3 | AEC3 |
|------|--------|-----------|------|
| `jtYTdZm3lUmFVNibJWq8YQ_doubletalk` | DT_static | -0.343 | 2.298 |
| `9xjhiFbGo06hdQIsHTS6qA_farend_singletalk` | FS_static | -1.713 | 3.442 |
| `xQEUtY2pWUi7v1X93TF2AA_farend_singletalk` | FS_static | -0.730 | 4.219 |

---

## 9xjhi Watchlist — FS_static Cases

**Total FS_static cases**: 169

**9xjhi itself**: M0=1.743 M_full=1.729 Δvs_M0=-0.014 Δvs_AEC3=-1.713 (AEC3=3.442)

**FS_static regressions vs M0** (Δecho < −0.05, echo drops = worse): 69 case(s)

| Case | M0_echo | M_full_echo | Δ |
|------|---------|-------------|---|
| `ql7yTcebJU20VE5qpW0kCA_farend_singletalk` | 3.977 | 2.464 | -1.514 |
| `9bSnA8CNBUSsJeIVCELzSQ_farend_singletalk` | 3.172 | 2.598 | -0.574 |
| `pPBidi8oyUarZCGUrKJEsg_farend_singletalk` | 4.320 | 3.831 | -0.490 |
| `wlAXM0iDgkm06i7UdRww1w_farend_singletalk` | 2.746 | 2.285 | -0.462 |
| `TcuegeGKL06ysmFP0RxjyQ_farend_singletalk` | 2.870 | 2.442 | -0.428 |
| `o2wfdvOGwU6M8Fmn2dCvOA_farend_singletalk` | 3.211 | 2.842 | -0.368 |
| `MnzrTJXYiUCPRMjVzLk25w_farend_singletalk` | 3.777 | 3.436 | -0.341 |
| `IxgmaPghzUGnR6sxrbGU3Q_farend_singletalk` | 4.525 | 4.216 | -0.309 |
| `OXCtw0FVhUWGUBil6Uucdw_farend_singletalk` | 3.305 | 3.018 | -0.287 |
| `r7U6JmcRl0ibIh0mN3CP9g_farend_singletalk` | 3.789 | 3.521 | -0.268 |
| `yZs0i8NpJkypsV8QyvduzQ_farend_singletalk` | 3.907 | 3.647 | -0.260 |
| `qu84vhur1UaDJPH2eCmMZA_farend_singletalk` | 3.862 | 3.609 | -0.253 |
| `yM2wHof9U06yVPJfemZ3hg_farend_singletalk` | 3.214 | 2.965 | -0.249 |
| `JtodX3Ug6Eu5TYu0HN5IOw_farend_singletalk` | 3.026 | 2.779 | -0.247 |
| `knHK8NbAlESKGG4ZVuPOLg_farend_singletalk` | 4.313 | 4.067 | -0.247 |
| `VGlWeOPC6UiXSq4SYPiKpw_farend_singletalk` | 3.385 | 3.146 | -0.238 |
| `s0oJqM6Y1UCHSVmHmgsx4Q_farend_singletalk` | 2.840 | 2.609 | -0.230 |
| `DS5pS5fZwEG4X0NNifTh0w_farend_singletalk` | 2.223 | 2.000 | -0.223 |
| `IrQvqOTCmEWMXn9k2ICtRQ_farend_singletalk` | 2.291 | 2.074 | -0.218 |
| `pmzLFdKTzEixfU0l0furvA_farend_singletalk` | 2.538 | 2.329 | -0.209 |

**FS_static large improvements vs M0** (Δecho > +0.50): 1 case(s)

| Case | M0_echo | M_full_echo | Δ |
|------|---------|-------------|---|
| `HIMqDWjSoECJFtIP0TM9bg_farend_singletalk` | 3.462 | 4.460 | +0.998 |

---

## nores LF Artifact Check (FS_static only)

> LF band = 0–500 Hz of `_ours_nores` (linear output; enable_res=False).

> Δ_lf = M_full_nores_LF − M0_nores_LF (negative = improvement).

> 9xjhi target: Δ_lf ≈ −6 dB (Bundle A linear-layer fix confirmed).

**9xjhi nores LF**: M0=29.70 dB  M_full=29.82 dB  Δ=+0.12 dB  ⚠ REGRESSION

**FS_static nores LF Δ summary** (N=169): mean=-0.05 dB  std=0.26 dB  regressions (Δ > +1 dB): 1

**nores LF regressions (Δ > +1 dB): 1 cases**

| Case | M0_LF (dB) | M_full_LF (dB) | Δ (dB) |
|------|-----------|----------------|--------|
| `Y5Xx0GOZs0SmryzlY1KC2Q_farend_singletalk` | 4.44 | 5.54 | +1.09 |

---

## Conclusion

**Overall verdict: NO-SHIP**

40 production catastrophic(s) + 3 alignment catastrophic(s). Investigate before shipping.


### Production ledger summary

- **DT_mvmt** (deg): N=114 mean Δ=+0.048
- **DT_static** (deg): N=186 mean Δ=+0.026
- **FS_mvmt** (echo): N=131 mean Δ=-0.054
- **FS_static** (echo): N=169 mean Δ=-0.049
- **NE** (deg): N=95 mean Δ=+0.005

### Alignment ledger summary (12 cases vs AEC3)

- **DT_mvmt**: N=3 mean Δ_vs_AEC3=+0.666
- **DT_static**: N=4 mean Δ_vs_AEC3=+0.094
- **FS_mvmt**: N=1 mean Δ_vs_AEC3=+0.134
- **FS_static**: N=3 mean Δ_vs_AEC3=-0.646
- **NE**: N=1 mean Δ_vs_AEC3=+0.229

**Alignment catastrophics** (stop if present):
  - `jtYTdZm3lUmFVNibJWq8YQ_doubletalk` DT_static Δ=-0.343 (AEC3=2.298)
  - `9xjhiFbGo06hdQIsHTS6qA_farend_singletalk` FS_static Δ=-1.713 (AEC3=3.442)
  - `xQEUtY2pWUi7v1X93TF2AA_farend_singletalk` FS_static Δ=-0.730 (AEC3=4.219)

---

*Auto-generated by `python/v3_21_800case_bench.py`.*

*No code changes. No merge. No version bump.*

