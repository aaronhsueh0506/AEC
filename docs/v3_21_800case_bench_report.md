# v3.21 800-case Benchmark Report — M_full_delay vs M0

**Date**: 2026-05-27  
**Cases**: 695/800  
**Config**: preset=balanced / filter=832 / cng / workers=4

**Byte-equal precheck** (M0 vs plain BALANCED): PASS ✓

## Flag Composition

| Flag | M0 | M_full_delay |
|------|-----|--------------|
| `use_partition_summed_x2_for_h_error_gain` | OFF | ON |
| `use_current_e2_refined_in_h_error_denominator` | OFF | ON |
| `use_per_bin_h_error_refresh` | OFF | ON |
| `use_aec3_h_error_ceil` | OFF | ON |
| `use_aec3_filter_noise_gate_power` | OFF | ON |
| `use_partition_summed_x2_for_shadow_mu` | OFF | ON |
| `use_aec3_noise_gate_for_shadow` | OFF | ON |
| `use_poor_excitation_gate_for_shadow` | OFF | ON |
| `use_narrowband_mask_for_shadow` | OFF | ON |
| `use_saturation_gate_for_shadow` | OFF | ON |
| `use_refined_output_selection_for_linear_path` | OFF | ON |
| `form_linear_filter_crossfade_enabled` | OFF | ON |
| `use_full_delay_change_chain` | OFF | ON |

---

## Production Ledger (M_full_delay vs M0 = v3.21.6 anchor)

### Bucket Means

| Bucket | Metric | N | Δ_mean | Δ_std | Worst Δ | Best Δ |
|--------|--------|---|--------|-------|---------|--------|
| DT_mvmt | deg | 114 | -0.033 | 0.417 | -1.289 | +1.109 |
| DT_static | deg | 186 | -0.017 | 0.488 | -1.420 | +1.775 |
| FS_mvmt | echo | 131 | -0.071 | 0.386 | -1.341 | +0.904 |
| FS_static | echo | 169 | -0.042 | 0.371 | -1.281 | +1.099 |
| NE | deg | 95 | +0.034 | 0.125 | -0.221 | +0.713 |

### Worst-5 per Bucket (Δ vs M0)

**DT_mvmt** (metric=deg):
| Case | M0 | M_full | Δ |
|------|----|--------|---|
| `xFk7igecuke0R5JMfREyDg_doubletalk_with_movement` | 2.895 | 1.605 | -1.289 |
| `kGFOHthwrUWqHCLkBYIQnA_doubletalk_with_movement` | 3.258 | 2.386 | -0.872 |
| `IrQvqOTCmEWMXn9k2ICtRQ_doubletalk_with_movement` | 3.569 | 2.744 | -0.825 |
| `nlSSRl4k50Gq2mIRYlMBCg_doubletalk_with_movement` | 2.525 | 1.707 | -0.818 |
| `OmB0Ht0hmE2crVnftAEtsw_doubletalk_with_movement` | 2.930 | 2.224 | -0.706 |

**DT_static** (metric=deg):
| Case | M0 | M_full | Δ |
|------|----|--------|---|
| `XRTnTUjU5kS0mejzCqyCiw_doubletalk` | 3.905 | 2.485 | -1.420 |
| `nVUnxqHLr0GTN7shWid1Ow_doubletalk` | 3.226 | 1.950 | -1.276 |
| `Y7w0W4v9BEihm8Z06BxZfQ_doubletalk` | 3.317 | 2.081 | -1.235 |
| `sUQrHEPAoEmIvHclpi1tRQ_doubletalk` | 2.866 | 1.634 | -1.232 |
| `49IIo03GZ0CYQOmeA3A0BA_doubletalk` | 3.921 | 2.780 | -1.141 |

**FS_mvmt** (metric=echo):
| Case | M0 | M_full | Δ |
|------|----|--------|---|
| `xb7eJJF0Vki6Yl3y4B7oJA_farend_singletalk_with_move` | 4.188 | 2.847 | -1.341 |
| `z4PqfBhq2E01IDBkTH0gnw_farend_singletalk_with_move` | 3.731 | 2.523 | -1.208 |
| `lxLsvT1rY0mdtZuRogM06Q_farend_singletalk_with_move` | 3.853 | 2.658 | -1.195 |
| `lV0kQN0hR0ySmE0bQhuYbw_farend_singletalk_with_move` | 4.042 | 2.918 | -1.124 |
| `zzCIhneJ8UKTWZ48U0kRXw_farend_singletalk_with_move` | 4.277 | 3.376 | -0.901 |

**FS_static** (metric=echo):
| Case | M0 | M_full | Δ |
|------|----|--------|---|
| `ualhjKw9zU6bheF6oQjnCw_farend_singletalk` | 4.173 | 2.892 | -1.281 |
| `7GTxyTksSUqCnP5y0ILG4A_farend_singletalk` | 3.771 | 2.656 | -1.115 |
| `ksP3OuSnpUa9Si2ttiUSoA_farend_singletalk` | 3.853 | 2.957 | -0.895 |
| `TcuegeGKL06ysmFP0RxjyQ_farend_singletalk` | 3.684 | 2.886 | -0.798 |
| `IrQvqOTCmEWMXn9k2ICtRQ_farend_singletalk` | 3.055 | 2.272 | -0.783 |

**NE** (metric=deg):
| Case | M0 | M_full | Δ |
|------|----|--------|---|
| `DW3NNHoKl02cAfEkahzsGg_nearend_singletalk` | 3.172 | 2.952 | -0.221 |
| `JkDl1iAjcUelM740jzUz0A_nearend_singletalk` | 4.112 | 3.942 | -0.170 |
| `x7Frsl5wTEKmog3A6lL57g_nearend_singletalk` | 3.475 | 3.368 | -0.108 |
| `XnfMDZLl0U2WvLRphiGJ6A_nearend_singletalk` | 4.324 | 4.258 | -0.067 |
| `014AzuqPZku2004NbTTmcA_nearend_singletalk` | 4.355 | 4.355 | +0.000 |

### Catastrophic Cases (vs M0)

> DT Δdeg < −0.20  OR  FS Δecho < −0.20 vs M0  (both metrics: higher = better)

**183 catastrophic case(s):**

| Case | Bucket | Δ |
|------|--------|---|
| `XRTnTUjU5kS0mejzCqyCiw_doubletalk` | DT_static | -1.420 |
| `xb7eJJF0Vki6Yl3y4B7oJA_farend_singletalk_with_move` | FS_mvmt | -1.341 |
| `xFk7igecuke0R5JMfREyDg_doubletalk_with_movement` | DT_mvmt | -1.289 |
| `ualhjKw9zU6bheF6oQjnCw_farend_singletalk` | FS_static | -1.281 |
| `nVUnxqHLr0GTN7shWid1Ow_doubletalk` | DT_static | -1.276 |
| `Y7w0W4v9BEihm8Z06BxZfQ_doubletalk` | DT_static | -1.235 |
| `sUQrHEPAoEmIvHclpi1tRQ_doubletalk` | DT_static | -1.232 |
| `z4PqfBhq2E01IDBkTH0gnw_farend_singletalk_with_move` | FS_mvmt | -1.208 |
| `lxLsvT1rY0mdtZuRogM06Q_farend_singletalk_with_move` | FS_mvmt | -1.195 |
| `49IIo03GZ0CYQOmeA3A0BA_doubletalk` | DT_static | -1.141 |
| `lV0kQN0hR0ySmE0bQhuYbw_farend_singletalk_with_move` | FS_mvmt | -1.124 |
| `7GTxyTksSUqCnP5y0ILG4A_farend_singletalk` | FS_static | -1.115 |
| `MYrVxVEMxkaE7OuyTUmI0Q_doubletalk` | DT_static | -0.970 |
| `yc5bFUGsR0GSfiGwTTpRWg_doubletalk` | DT_static | -0.945 |
| `NY3kZioAm0KwR45wIVe2Sg_doubletalk` | DT_static | -0.924 |
| `WAx9ADn1O00xxkqYq0hPlg_doubletalk` | DT_static | -0.924 |
| `W0J6iZv7ZkmHOobCToob4A_doubletalk` | DT_static | -0.911 |
| `S22FCqKDWUyymN1YbpItIw_doubletalk` | DT_static | -0.906 |
| `zzCIhneJ8UKTWZ48U0kRXw_farend_singletalk_with_move` | FS_mvmt | -0.901 |
| `ksP3OuSnpUa9Si2ttiUSoA_farend_singletalk` | FS_static | -0.895 |
| `kGFOHthwrUWqHCLkBYIQnA_doubletalk_with_movement` | DT_mvmt | -0.872 |
| `wlAXM0iDgkm06i7UdRww1w_doubletalk` | DT_static | -0.835 |
| `XTqo1aOXDEiqyWTFK99I5Q_doubletalk` | DT_static | -0.827 |
| `IrQvqOTCmEWMXn9k2ICtRQ_doubletalk_with_movement` | DT_mvmt | -0.825 |
| `nlSSRl4k50Gq2mIRYlMBCg_doubletalk_with_movement` | DT_mvmt | -0.818 |
| `kZogUfYct0qMwSqvRTwOVg_doubletalk` | DT_static | -0.817 |
| `TcuegeGKL06ysmFP0RxjyQ_farend_singletalk` | FS_static | -0.798 |
| `IrQvqOTCmEWMXn9k2ICtRQ_farend_singletalk` | FS_static | -0.783 |
| `49DamGOwmUWGCn23bmI8xw_farend_singletalk` | FS_static | -0.776 |
| `yZs0i8NpJkypsV8QyvduzQ_farend_singletalk` | FS_static | -0.709 |
| `VGlWeOPC6UiXSq4SYPiKpw_farend_singletalk` | FS_static | -0.709 |
| `OmB0Ht0hmE2crVnftAEtsw_doubletalk_with_movement` | DT_mvmt | -0.706 |
| `zpiSOkxpHkCs5SqdOo5ZIQ_doubletalk_with_movement` | DT_mvmt | -0.701 |
| `p0mhFbhV6kGJgjd0RTTIIw_farend_singletalk` | FS_static | -0.700 |
| `KSN5Jrzo7kaixP0z8xfr4Q_farend_singletalk_with_move` | FS_mvmt | -0.698 |
| `iOyPaxX11UOaUkcscKhq1A_farend_singletalk` | FS_static | -0.689 |
| `4pN9yn7mhEa5iDiKnr5jlw_farend_singletalk_with_move` | FS_mvmt | -0.678 |
| `SwfEwuGDlkWYy9pb4H00eQ_doubletalk_with_movement` | DT_mvmt | -0.676 |
| `xSh5yXWiP02K0UkYdkZ0cA_doubletalk` | DT_static | -0.675 |
| `4wYZCudp4Umtu9lVi0304g_farend_singletalk` | FS_static | -0.674 |
| `Fi80N5kW9U6nwaoS04O3vQ_farend_singletalk_with_move` | FS_mvmt | -0.674 |
| `vF1LKDSGbUGtp0pR6Fzb3A_farend_singletalk` | FS_static | -0.673 |
| `xnpFE06ShUea4Jn1Wu7EzQ_doubletalk` | DT_static | -0.669 |
| `XuguA1uJAE0bWT0xXRDdeA_farend_singletalk_with_move` | FS_mvmt | -0.662 |
| `XR7EqSNfH06yk30150jF6g_doubletalk` | DT_static | -0.661 |
| `o7yLy0sI9kCpV0HgSDfL8A_doubletalk` | DT_static | -0.654 |
| `wlAXM0iDgkm06i7UdRww1w_doubletalk_with_movement` | DT_mvmt | -0.651 |
| `qu84vhur1UaDJPH2eCmMZA_doubletalk` | DT_static | -0.650 |
| `kGFOHthwrUWqHCLkBYIQnA_doubletalk` | DT_static | -0.635 |
| `zddPqpp1a06xttKdc0iNTA_farend_singletalk_with_move` | FS_mvmt | -0.632 |
| `j0awp3hXrkCSqhR748U3iQ_farend_singletalk` | FS_static | -0.625 |
| `m6ciKvH6AEe7Yi2ptKjj1g_doubletalk_with_movement` | DT_mvmt | -0.614 |
| `V6Mw0Ti8RUSkzvMGB4WGiw_doubletalk` | DT_static | -0.612 |
| `kOGPX6kHskOaKSZdLGNz8A_farend_singletalk_with_move` | FS_mvmt | -0.600 |
| `9bSnA8CNBUSsJeIVCELzSQ_farend_singletalk` | FS_static | -0.597 |
| `MYrVxVEMxkaE7OuyTUmI0Q_farend_singletalk_with_move` | FS_mvmt | -0.594 |
| `WnDjVFWmC0m0WhVq22mRlQ_doubletalk_with_movement` | DT_mvmt | -0.584 |
| `OjdIdZgJDk6hLAQL07KORA_doubletalk` | DT_static | -0.580 |
| `Je6gJ7y1PECStwxnrOe9aA_farend_singletalk_with_move` | FS_mvmt | -0.574 |
| `mrmDEdQMpk6hJnMqn59pOQ_farend_singletalk` | FS_static | -0.574 |
| `8KO3KPpljkiwh06qjaVdWw_farend_singletalk` | FS_static | -0.569 |
| `IqtJR4tjJkWrwUjYorz0Og_farend_singletalk_with_move` | FS_mvmt | -0.569 |
| `sLWe8bfYbkGwX1W3PzI1PQ_doubletalk` | DT_static | -0.557 |
| `s90M7MOTBkqaV4nQPLhKbA_doubletalk` | DT_static | -0.555 |
| `WJC7Ri8s0E2qIrgvcXtoiQ_farend_singletalk_with_move` | FS_mvmt | -0.530 |
| `0I0XMl3M0ECO0U1N0cJvpg_doubletalk` | DT_static | -0.518 |
| `WAx9ADn1O00xxkqYq0hPlg_doubletalk_with_movement` | DT_mvmt | -0.518 |
| `oc0eVAlCbEiTTPNZmV4pMQ_farend_singletalk` | FS_static | -0.517 |
| `pmzLFdKTzEixfU0l0furvA_farend_singletalk` | FS_static | -0.514 |
| `nV9v63E5CUKtKTjha8dtdQ_farend_singletalk_with_move` | FS_mvmt | -0.505 |
| `lV0kQN0hR0ySmE0bQhuYbw_doubletalk` | DT_static | -0.501 |
| `DS5pS5fZwEG4X0NNifTh0w_farend_singletalk` | FS_static | -0.496 |
| `xvACDxradUuKNYImFSd1ww_doubletalk_with_movement` | DT_mvmt | -0.495 |
| `sKXucFp4FUCJKo5d0G54Og_doubletalk_with_movement` | DT_mvmt | -0.493 |
| `uS9t2QYDckeO7SnQNYZVcg_farend_singletalk` | FS_static | -0.485 |
| `wlAXM0iDgkm06i7UdRww1w_farend_singletalk` | FS_static | -0.469 |
| `W4r0UCjieEuM0u930spvug_doubletalk` | DT_static | -0.468 |
| `TGZ5Wq0SCUCOXPsfee3uMQ_farend_singletalk_with_move` | FS_mvmt | -0.466 |
| `o2wfdvOGwU6M8Fmn2dCvOA_farend_singletalk` | FS_static | -0.463 |
| `WCLKkH4DiEmAHQue4iNyOA_doubletalk` | DT_static | -0.462 |
| `TRSNunEou0aqmBCGIC8B7A_farend_singletalk_with_move` | FS_mvmt | -0.460 |
| `XGDaZuEkE0WU4IN0Yi4XtA_doubletalk` | DT_static | -0.459 |
| `WnDjVFWmC0m0WhVq22mRlQ_doubletalk` | DT_static | -0.458 |
| `W0J6iZv7ZkmHOobCToob4A_doubletalk_with_movement` | DT_mvmt | -0.451 |
| `ql7yTcebJU20VE5qpW0kCA_doubletalk` | DT_static | -0.447 |
| `V6Mw0Ti8RUSkzvMGB4WGiw_doubletalk_with_movement` | DT_mvmt | -0.445 |
| `Uc4dmejgWUCTvn0XZbMTBw_farend_singletalk_with_move` | FS_mvmt | -0.444 |
| `W0zK3dv0QE2YckPArTGXCg_doubletalk_with_movement` | DT_mvmt | -0.436 |
| `wY00iJ3cE0aQsjt0m1tC0g_doubletalk_with_movement` | DT_mvmt | -0.421 |
| `kwolfjBXWEOJmdbDdFoTVQ_doubletalk_with_movement` | DT_mvmt | -0.411 |
| `tl5UFRCXZkyL6EoWVl09xA_farend_singletalk_with_move` | FS_mvmt | -0.411 |
| `zONvcX0qYkuaAViV5PXcYg_doubletalk` | DT_static | -0.409 |
| `sRCs6SKo6kC0xire475q0A_doubletalk_with_movement` | DT_mvmt | -0.408 |
| `HAxmF7v4dE0itSp5R5B3Dw_farend_singletalk` | FS_static | -0.408 |
| `JtodX3Ug6Eu5TYu0HN5IOw_doubletalk` | DT_static | -0.400 |
| `hF9Lfjcn9kGQ4430uAbINA_farend_singletalk_with_move` | FS_mvmt | -0.397 |
| `WCLKkH4DiEmAHQue4iNyOA_farend_singletalk` | FS_static | -0.396 |
| `sUQrHEPAoEmIvHclpi1tRQ_farend_singletalk` | FS_static | -0.388 |
| `w5XDRNfB2Ei2UoUDtrTkzg_doubletalk` | DT_static | -0.387 |
| `YCmmUCd3aEWd0V4s7MJQ8g_doubletalk` | DT_static | -0.386 |
| `XTqo1aOXDEiqyWTFK99I5Q_doubletalk_with_movement` | DT_mvmt | -0.382 |
| `zONvcX0qYkuaAViV5PXcYg_farend_singletalk_with_move` | FS_mvmt | -0.381 |
| `kZogUfYct0qMwSqvRTwOVg_farend_singletalk` | FS_static | -0.377 |
| `y1VEXHeaH0K0eKC5tlE7rg_doubletalk` | DT_static | -0.372 |
| `kZogUfYct0qMwSqvRTwOVg_doubletalk_with_movement` | DT_mvmt | -0.371 |
| `Y5Xx0GOZs0SmryzlY1KC2Q_farend_singletalk` | FS_static | -0.369 |
| `N2rQLbnp2UOg2QFRaggbDw_doubletalk_with_movement` | DT_mvmt | -0.368 |
| `oxSdYr0mzESqEpSyHlztug_farend_singletalk_with_move` | FS_mvmt | -0.367 |
| `Khk1qeMXFUuvFhw3YRSm0w_doubletalk` | DT_static | -0.365 |
| `xNr7L0xsLUG4B9oUqW0V4Q_doubletalk` | DT_static | -0.364 |
| `lH20r2skzU02a647xYoFoA_farend_singletalk_with_move` | FS_mvmt | -0.354 |
| `pU21kfoo0UOz0fPMJFfydg_farend_singletalk` | FS_static | -0.354 |
| `xofDX004bkqiOv9YOxmGVQ_doubletalk` | DT_static | -0.329 |
| `XuiheB7eUkyJA2XzFIovHQ_doubletalk` | DT_static | -0.328 |
| `qiQL0BUPxk0YtpnP7JGfNg_doubletalk` | DT_static | -0.322 |
| `I2bme08keUmAnyJRKNYDGQ_doubletalk_with_movement` | DT_mvmt | -0.321 |
| `P10GsQvhskKx3fB06Zv4Yg_farend_singletalk` | FS_static | -0.316 |
| `yM2wHof9U06yVPJfemZ3hg_farend_singletalk` | FS_static | -0.314 |
| `0luXwWjGEEC9G5nf0yTVXw_farend_singletalk_with_move` | FS_mvmt | -0.314 |
| `m4789fdio0q92zjf9gvh1Q_farend_singletalk_with_move` | FS_mvmt | -0.313 |
| `VtEW0vwTO0eEQaCBSfjjnQ_farend_singletalk` | FS_static | -0.312 |
| `xA63038UDkGgZvGqHr0Kiw_doubletalk` | DT_static | -0.309 |
| `m6ciKvH6AEe7Yi2ptKjj1g_farend_singletalk_with_move` | FS_mvmt | -0.307 |
| `XV5L2dn3S06M9GBEu1q3DA_doubletalk` | DT_static | -0.307 |
| `w9Cji060a0Ss7zLxa05Xhw_farend_singletalk` | FS_static | -0.305 |
| `GDyfzBkhxEiDbnRZGGOrQQ_farend_singletalk` | FS_static | -0.305 |
| `zOiK6oSHp0ib3nHvzLKbRQ_doubletalk_with_movement` | DT_mvmt | -0.305 |
| `pPBidi8oyUarZCGUrKJEsg_doubletalk` | DT_static | -0.303 |
| `tl5UFRCXZkyL6EoWVl09xA_farend_singletalk` | FS_static | -0.301 |
| `sKXucFp4FUCJKo5d0G54Og_farend_singletalk` | FS_static | -0.298 |
| `wHmBm7VHfkysBOhjoAXkNA_doubletalk` | DT_static | -0.293 |
| `urm5FZsuoEGEayow6ckb0w_farend_singletalk_with_move` | FS_mvmt | -0.292 |
| `OX2l6zV7nkmmSkVA3ETLKg_farend_singletalk_with_move` | FS_mvmt | -0.292 |
| `nyT6FUUdu0W8UpvjP1rRgQ_farend_singletalk_with_move` | FS_mvmt | -0.290 |
| `hvY1v0viv0yMdAXKa2y1aw_doubletalk_with_movement` | DT_mvmt | -0.290 |
| `UmlD9X38NECNoJKm0oyf4w_farend_singletalk` | FS_static | -0.287 |
| `S22FCqKDWUyymN1YbpItIw_doubletalk_with_movement` | DT_mvmt | -0.286 |
| `Hq00pd6Ey0mGtuMFRoF79w_farend_singletalk_with_move` | FS_mvmt | -0.285 |
| `xnpFE06ShUea4Jn1Wu7EzQ_doubletalk_with_movement` | DT_mvmt | -0.283 |
| `ql7yTcebJU20VE5qpW0kCA_doubletalk_with_movement` | DT_mvmt | -0.282 |
| `waxU019BVEacr7vK6v00mQ_doubletalk_with_movement` | DT_mvmt | -0.273 |
| `orvXZE0juUeRPAAdjZSqoA_doubletalk` | DT_static | -0.272 |
| `Ixf70mgKwkCoFYq32586cw_farend_singletalk_with_move` | FS_mvmt | -0.272 |
| `ZWq0X5sPiUe0lQjZdCPSeQ_doubletalk` | DT_static | -0.268 |
| `oEyXuSCCw0qdJ0J16FGfcQ_doubletalk` | DT_static | -0.267 |
| `wY3Rp9YEjkOke09hMwfsjg_doubletalk` | DT_static | -0.265 |
| `G04O5YNkmkSzDB28fSacyg_farend_singletalk` | FS_static | -0.262 |
| `NY3kZioAm0KwR45wIVe2Sg_farend_singletalk` | FS_static | -0.260 |
| `iyuYIcszXku7BWYOOwqh5Q_farend_singletalk_with_move` | FS_mvmt | -0.258 |
| `kHsrUmyfT0O0RYtusGuQyQ_farend_singletalk_with_move` | FS_mvmt | -0.255 |
| `wVYSGVTTakih9twI4xlDWQ_doubletalk` | DT_static | -0.254 |
| `KgZ0y2EQJ0a4jvtsznBrvw_farend_singletalk` | FS_static | -0.252 |
| `UmlD9X38NECNoJKm0oyf4w_doubletalk` | DT_static | -0.249 |
| `SFvlSygv4ke9wCrv8LWvYQ_doubletalk` | DT_static | -0.247 |
| `s0oJqM6Y1UCHSVmHmgsx4Q_doubletalk_with_movement` | DT_mvmt | -0.242 |
| `Je6gJ7y1PECStwxnrOe9aA_doubletalk_with_movement` | DT_mvmt | -0.239 |
| `lzEZpNXmy0KWtSGT6td00g_farend_singletalk_with_move` | FS_mvmt | -0.238 |
| `PXfMWCKVykukw7Se9Aq7wQ_farend_singletalk_with_move` | FS_mvmt | -0.238 |
| `wJVPo4lexUK40x0nuK0KWg_doubletalk` | DT_static | -0.237 |
| `hVqUmGvIlkO0LBUoE06Q3w_farend_singletalk` | FS_static | -0.233 |
| `ML4MF3Mea0yurjceNQPfNA_farend_singletalk_with_move` | FS_mvmt | -0.232 |
| `Y4zG6bHup06zWMoq3OvZqQ_doubletalk` | DT_static | -0.231 |
| `QEeKiaNiDECfqXTRrDFWWw_doubletalk_with_movement` | DT_mvmt | -0.230 |
| `NNdxDj6FEk6CAwvbW01bUg_farend_singletalk` | FS_static | -0.230 |
| `XXz0qkUSd0GT4dsywxpfJg_farend_singletalk_with_move` | FS_mvmt | -0.227 |
| `khqZY41lNEyIvMf2ZNJuVA_farend_singletalk` | FS_static | -0.226 |
| `VNgRsWxMdkaUx1gKV9W1Zw_farend_singletalk_with_move` | FS_mvmt | -0.226 |
| `DW3NNHoKl02cAfEkahzsGg_nearend_singletalk` | NE | -0.221 |
| `V0JqgjlrB0Ke9y91r0rxNw_doubletalk` | DT_static | -0.220 |
| `SFvlSygv4ke9wCrv8LWvYQ_farend_singletalk` | FS_static | -0.218 |
| `JLNgGcvTNEqbTDbc28wLkg_farend_singletalk` | FS_static | -0.217 |
| `wWeNtFK0dEG9Wub40bB15A_doubletalk` | DT_static | -0.215 |
| `o7yLy0sI9kCpV0HgSDfL8A_farend_singletalk` | FS_static | -0.212 |
| `OLjlc92QWU6fwuN4ytCPQg_farend_singletalk` | FS_static | -0.212 |
| `N94NwopiZEyNnraWHLMDcg_doubletalk` | DT_static | -0.211 |
| `XnfMDZLl0U2WvLRphiGJ6A_doubletalk_with_movement` | DT_mvmt | -0.210 |
| `QK70KpLuZ0O43BBSWEZvHg_farend_singletalk_with_move` | FS_mvmt | -0.207 |
| `9xjhiFbGo06hdQIsHTS6qA_farend_singletalk` | FS_static | -0.204 |
| `KOy0eftktkuJf180xtXudg_doubletalk` | DT_static | -0.204 |
| `49IIo03GZ0CYQOmeA3A0BA_farend_singletalk` | FS_static | -0.202 |
| `Ja8OngfthkOCmL8ldcRNyg_farend_singletalk_with_move` | FS_mvmt | -0.202 |
| `Tgtk8jp1zkqmKzsmdrKt0g_doubletalk_with_movement` | DT_mvmt | -0.201 |
| `LQhlYoXXiUevFuxMKwWB0Q_farend_singletalk` | FS_static | -0.200 |

---

## Alignment Ledger (M_full_delay vs AEC3 behavioral reference)

> AEC3 scores from `bin/aec3_cli` run on 12-case cohort (2026-05-27).

> Only these 12 cases have known AEC3 reference scores.


| Case | Bucket | Metric | AEC3 | M_full | Δ_vs_AEC3 | Δ_vs_M0 | Status |
|------|--------|--------|------|--------|-----------|---------|--------|
| `ZJYUt0O0AEKSQ9LJ8z7t0A_doubletalk_with_mov` | DT_mvmt | deg | 2.177 | 3.074 | +0.897 | +1.044 | ✓ PASS |
| `wVYSGVTTakih9twI4xlDWQ_doubletalk_with_mov` | DT_mvmt | deg | 1.540 | 2.955 | +1.415 | +0.799 | ✓ PASS |
| `xFk7igecuke0R5JMfREyDg_doubletalk_with_mov` | DT_mvmt | deg | 1.275 | 1.605 | +0.330 | -1.289 | ✓ PASS |
| `MYrVxVEMxkaE7OuyTUmI0Q_doubletalk` | DT_static | deg | 1.275 | 1.847 | +0.572 | -0.970 | ✓ PASS |
| `XRTnTUjU5kS0mejzCqyCiw_doubletalk` | DT_static | deg | 2.062 | 2.485 | +0.423 | -1.420 | ✓ PASS |
| `jtYTdZm3lUmFVNibJWq8YQ_doubletalk` | DT_static | deg | 2.298 | 2.832 | +0.534 | +0.243 | ✓ PASS |
| `nVUnxqHLr0GTN7shWid1Ow_doubletalk` | DT_static | deg | 1.547 | 1.950 | +0.403 | -1.276 | ✓ PASS |
| `0I0XMl3M0ECO0U1N0cJvpg_farend_singletalk_w` | FS_mvmt | echo | 4.296 | 4.184 | -0.112 | +0.074 | ✓ PASS |
| `9xjhiFbGo06hdQIsHTS6qA_farend_singletalk` | FS_static | echo | 3.442 | 2.049 | -1.393 | -0.204 | ⚠ KNOWN EXCEPTION |
| `qNvSMyUSXUyrDGpOw7s6qg_farend_singletalk` | FS_static | echo | 3.596 | 3.594 | -0.002 | -0.042 | ✓ PASS |
| `xQEUtY2pWUi7v1X93TF2AA_farend_singletalk` | FS_static | echo | 4.219 | 3.788 | -0.431 | +0.440 | ⚠ FAIL |
| `014AzuqPZku2004NbTTmcA_nearend_singletalk` | NE | deg | 4.164 | 4.355 | +0.191 | +0.000 | ✓ PASS |

### Alignment Bucket Summary

| Bucket | N | Mean Δ_vs_AEC3 | Status |
|--------|---|----------------|--------|
| DT_mvmt | 3 | +0.881 | ✓ PASS |
| DT_static | 4 | +0.483 | ✓ PASS |
| FS_mvmt | 1 | -0.112 | ✓ PASS |
| FS_static | 3 | -0.609 | ⚠ FAIL (9xjhi exception) |
| NE | 1 | +0.191 | ✓ PASS |

### Alignment Catastrophics (vs AEC3 ref)

> DT worse than AEC3 by > 0.10 deg  OR  FS worse than AEC3 by > 0.30 echo

**2 alignment catastrophic(s):**

| Case | Bucket | Δ_vs_AEC3 | AEC3 |
|------|--------|-----------|------|
| `9xjhiFbGo06hdQIsHTS6qA_farend_singletalk` | FS_static | -1.393 | 3.442 |
| `xQEUtY2pWUi7v1X93TF2AA_farend_singletalk` | FS_static | -0.431 | 4.219 |

---

## 9xjhi Watchlist — FS_static Cases

**Total FS_static cases**: 169

**9xjhi itself**: M0=2.252 M_full=2.049 Δvs_M0=-0.204 Δvs_AEC3=-1.393 (AEC3=3.442)

**FS_static regressions vs M0** (Δecho < −0.05, echo drops = worse): 77 case(s)

| Case | M0_echo | M_full_echo | Δ |
|------|---------|-------------|---|
| `ualhjKw9zU6bheF6oQjnCw_farend_singletalk` | 4.173 | 2.892 | -1.281 |
| `7GTxyTksSUqCnP5y0ILG4A_farend_singletalk` | 3.771 | 2.656 | -1.115 |
| `ksP3OuSnpUa9Si2ttiUSoA_farend_singletalk` | 3.853 | 2.957 | -0.895 |
| `TcuegeGKL06ysmFP0RxjyQ_farend_singletalk` | 3.684 | 2.886 | -0.798 |
| `IrQvqOTCmEWMXn9k2ICtRQ_farend_singletalk` | 3.055 | 2.272 | -0.783 |
| `49DamGOwmUWGCn23bmI8xw_farend_singletalk` | 4.360 | 3.584 | -0.776 |
| `yZs0i8NpJkypsV8QyvduzQ_farend_singletalk` | 4.070 | 3.361 | -0.709 |
| `VGlWeOPC6UiXSq4SYPiKpw_farend_singletalk` | 3.650 | 2.941 | -0.709 |
| `p0mhFbhV6kGJgjd0RTTIIw_farend_singletalk` | 4.151 | 3.450 | -0.700 |
| `iOyPaxX11UOaUkcscKhq1A_farend_singletalk` | 3.559 | 2.870 | -0.689 |
| `4wYZCudp4Umtu9lVi0304g_farend_singletalk` | 4.175 | 3.501 | -0.674 |
| `vF1LKDSGbUGtp0pR6Fzb3A_farend_singletalk` | 4.282 | 3.609 | -0.673 |
| `j0awp3hXrkCSqhR748U3iQ_farend_singletalk` | 3.853 | 3.229 | -0.625 |
| `9bSnA8CNBUSsJeIVCELzSQ_farend_singletalk` | 4.015 | 3.418 | -0.597 |
| `mrmDEdQMpk6hJnMqn59pOQ_farend_singletalk` | 3.827 | 3.253 | -0.574 |
| `8KO3KPpljkiwh06qjaVdWw_farend_singletalk` | 3.754 | 3.185 | -0.569 |
| `oc0eVAlCbEiTTPNZmV4pMQ_farend_singletalk` | 4.164 | 3.647 | -0.517 |
| `pmzLFdKTzEixfU0l0furvA_farend_singletalk` | 3.150 | 2.636 | -0.514 |
| `DS5pS5fZwEG4X0NNifTh0w_farend_singletalk` | 3.203 | 2.707 | -0.496 |
| `uS9t2QYDckeO7SnQNYZVcg_farend_singletalk` | 3.820 | 3.335 | -0.485 |

**FS_static large improvements vs M0** (Δecho > +0.50): 11 case(s)

| Case | M0_echo | M_full_echo | Δ |
|------|---------|-------------|---|
| `KSN5Jrzo7kaixP0z8xfr4Q_farend_singletalk` | 2.141 | 3.239 | +1.099 |
| `JtodX3Ug6Eu5TYu0HN5IOw_farend_singletalk` | 3.144 | 4.242 | +1.098 |
| `m5uBFyhN5UCVODFvL0KOeQ_farend_singletalk` | 3.337 | 4.078 | +0.741 |
| `1fvt8ajGxk2OhS7UglBjoA_farend_singletalk` | 3.329 | 4.027 | +0.698 |
| `sLi810BoekuU3HSx14LT7A_farend_singletalk` | 3.638 | 4.325 | +0.687 |
| `wr54weKzNkOcZ07hB04kzA_farend_singletalk` | 2.749 | 3.404 | +0.655 |
| `sYQK1rJlwU2XCy20n0Sx9g_farend_singletalk` | 2.771 | 3.417 | +0.646 |
| `t3A1ZgiaeUqQF8dYeYfE6Q_farend_singletalk` | 3.621 | 4.265 | +0.645 |
| `Tgtk8jp1zkqmKzsmdrKt0g_farend_singletalk` | 3.311 | 3.931 | +0.620 |
| `geGGo8g9UE2MYkdjxtHm8w_farend_singletalk` | 3.583 | 4.092 | +0.509 |

---

## nores LF Artifact Check (FS_static only)

> LF band = 0–500 Hz of `_ours_nores` (linear output; enable_res=False).

> Δ_lf = M_full_nores_LF − M0_nores_LF (negative = improvement).

> 9xjhi target: Δ_lf ≈ −6 dB (Bundle A linear-layer fix confirmed).

**9xjhi nores LF**: M0=32.46 dB  M_full=29.85 dB  Δ=-2.62 dB  ✓ (improvement maintained)

**FS_static nores LF Δ summary** (N=169): mean=-2.36 dB  std=3.39 dB  regressions (Δ > +1 dB): 16

**nores LF regressions (Δ > +1 dB): 16 cases**

| Case | M0_LF (dB) | M_full_LF (dB) | Δ (dB) |
|------|-----------|----------------|--------|
| `P10GsQvhskKx3fB06Zv4Yg_farend_singletalk` | 22.51 | 23.58 | +1.07 |
| `KSN5Jrzo7kaixP0z8xfr4Q_farend_singletalk` | 8.35 | 9.43 | +1.08 |
| `OjdIdZgJDk6hLAQL07KORA_farend_singletalk` | 10.52 | 11.61 | +1.09 |
| `r7U6JmcRl0ibIh0mN3CP9g_farend_singletalk` | 8.70 | 9.80 | +1.10 |
| `XXz0qkUSd0GT4dsywxpfJg_farend_singletalk` | 12.73 | 13.90 | +1.17 |
| `o97uExi7MEqzbanuWK6CCw_farend_singletalk` | 23.73 | 24.93 | +1.20 |
| `XNOJLHM8e0aUHP4rWEAAtQ_farend_singletalk` | 24.57 | 25.81 | +1.24 |
| `m5uBFyhN5UCVODFvL0KOeQ_farend_singletalk` | 18.76 | 20.15 | +1.39 |
| `8KO3KPpljkiwh06qjaVdWw_farend_singletalk` | 5.80 | 7.23 | +1.42 |
| `LeV1uF4j10Whm0FPG80tmw_farend_singletalk` | 19.44 | 20.97 | +1.53 |
| `NjVSdrdPXU2mB5uiecGSjg_farend_singletalk` | 17.16 | 18.73 | +1.57 |
| `xA63038UDkGgZvGqHr0Kiw_farend_singletalk` | 11.63 | 14.40 | +2.77 |
| `m4789fdio0q92zjf9gvh1Q_farend_singletalk` | 29.30 | 32.64 | +3.34 |
| `TZ6TJFCbfkKAVrS64Sf08Q_farend_singletalk` | 5.01 | 8.61 | +3.61 |
| `7GTxyTksSUqCnP5y0ILG4A_farend_singletalk` | 23.84 | 27.70 | +3.87 |
| `xQEUtY2pWUi7v1X93TF2AA_farend_singletalk` | 15.54 | 21.48 | +5.94 |

---

## Conclusion

**Overall verdict: NO-SHIP**

183 production catastrophic(s) + 2 alignment catastrophic(s). Investigate before shipping.


### Production ledger summary

- **DT_mvmt** (deg): N=114 mean Δ=-0.033
- **DT_static** (deg): N=186 mean Δ=-0.017
- **FS_mvmt** (echo): N=131 mean Δ=-0.071
- **FS_static** (echo): N=169 mean Δ=-0.042
- **NE** (deg): N=95 mean Δ=+0.034

### Alignment ledger summary (12 cases vs AEC3)

- **DT_mvmt**: N=3 mean Δ_vs_AEC3=+0.881
- **DT_static**: N=4 mean Δ_vs_AEC3=+0.483
- **FS_mvmt**: N=1 mean Δ_vs_AEC3=-0.112
- **FS_static**: N=3 mean Δ_vs_AEC3=-0.609
- **NE**: N=1 mean Δ_vs_AEC3=+0.191

**Alignment catastrophics** (stop if present):
  - `9xjhiFbGo06hdQIsHTS6qA_farend_singletalk` FS_static Δ=-1.393 (AEC3=3.442)
  - `xQEUtY2pWUi7v1X93TF2AA_farend_singletalk` FS_static Δ=-0.431 (AEC3=4.219)

---

*Auto-generated by `python/v3_21_800case_bench.py`.*

*No code changes. No merge. No version bump.*

