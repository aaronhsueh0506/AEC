# AEC3 RefinedFilterUpdateGain X² source parity fix — 800-case verdict

Config:
- Mechanism: align PBFDKF RefinedFilterUpdateGain X² source with AEC3.
  X²[k] = Σ_p |X_buf[p][k]|² (partition-summed render power, matching
  AEC3 `render_buffer.cc::SpectralSum`) instead of latest partition only.
  Used in: mu denominator / noise gate (silent-far floor) / H_error decay.
  W update outer product still uses per-partition X[p] — direction UNCHANGED.
- A_off: pre-parity (X² = latest partition only)
- B_on : parity ON (X² = partition-summed, AEC3 parity)
- Both runs: mic HPF ON / ref HPF OFF (intended HPF baseline),
  preset balanced, filter 832 (52ms), --cng, --parallel j4

Acceptance (3-way INDEPENDENT split):
1. nores LF artifact reduced on cohort tail — see 6-case audit (NOT in this report).
2. 800-case AECMOS Pareto-safe vs A_off — THIS report.
3. XRTnTUjU_DT_static stress: SEPARATE state-guard arc.
   Any Δdeg here reflects PRE-EXISTING gate-3 latch bug exposed by parity fix,
   NOT a formula problem (see project_usable_linear_gate3_latch_bug memory).
   Handled by separate state-guard work, NOT by reverting parity fix.

Cases: A=800, B=800, common=800.

XRTnTUjU_DT_static = stress / no-clean-convergence case (per project_xrtntuju_dt_static_stress memory). EXCLUDED from normal aggregate; kept as worst-N stress. Parity fix EXPOSES the gate-3 binary latch on convergence_seen; it does NOT cause it. Reverting parity to "hide" this would also revert the nores improvement and the AEC3 alignment.

## Bucket means (Δ vs A_off, NORMAL aggregate — XRTnTUjU_DT_static excluded)

| bucket | n_A | n_B | A_echo | B_echo | Δecho | A_deg | B_deg | Δdeg |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| DT_movement | 114 | 114 | 4.188 | 4.145 | -0.043 | 2.464 | 2.582 | +0.118 |
| DT_static | 185 | 185 | 4.249 | 4.217 | -0.032 | 2.401 | 2.522 | +0.121 |
| FS_movement | 131 | 131 | 3.532 | 3.475 | -0.057 | 4.999 | 4.999 | -0.000 |
| FS_static | 169 | 169 | 3.656 | 3.615 | -0.042 | 4.999 | 4.999 | -0.000 |
| NE | 200 | 200 | 4.998 | 4.998 | -0.000 | 4.038 | 4.078 | +0.039 |

## Bucket means (Δ vs A_off, INCLUDING stress, for completeness)

| bucket | n_A | n_B | A_echo | B_echo | Δecho | A_deg | B_deg | Δdeg |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| DT_movement | 114 | 114 | 4.188 | 4.145 | -0.043 | 2.464 | 2.582 | +0.118 |
| DT_static | 186 | 186 | 4.249 | 4.219 | -0.030 | 2.409 | 2.523 | +0.114 |
| FS_movement | 131 | 131 | 3.532 | 3.475 | -0.057 | 4.999 | 4.999 | -0.000 |
| FS_static | 169 | 169 | 3.656 | 3.615 | -0.042 | 4.999 | 4.999 | -0.000 |
| NE | 200 | 200 | 4.998 | 4.998 | -0.000 | 4.038 | 4.078 | +0.039 |

## XRTnTUjU_DT_static stress watch

A_off: echo=4.081, deg=3.929
B_on : echo=4.487, deg=2.676
**Δecho = +0.406, Δdeg = -1.253** (stress tolerance bar = −2.0 deg; state-guard arc target, NOT a parity-fix revert)

## Top 20 worst Δdeg per bucket (NORMAL, XRTnTUjU excluded)

### DT_movement
| stem | A_echo | B_echo | Δecho | A_deg | B_deg | Δdeg |
|---|---:|---:|---:|---:|---:|---:|
| `xFk7igecuke0R5JMfREyDg_doubletalk_with_movement` | 4.012 | 4.025 | +0.013 | 2.882 | 1.825 | -1.057 |
| `xvACDxradUuKNYImFSd1ww_doubletalk_with_movement` | 3.821 | 4.099 | +0.278 | 2.904 | 2.012 | -0.892 |
| `nlSSRl4k50Gq2mIRYlMBCg_doubletalk_with_movement` | 4.282 | 4.456 | +0.174 | 2.470 | 1.590 | -0.880 |
| `ql7yTcebJU20VE5qpW0kCA_doubletalk_with_movement` | 4.040 | 4.268 | +0.228 | 2.108 | 1.412 | -0.697 |
| `OmB0Ht0hmE2crVnftAEtsw_doubletalk_with_movement` | 4.518 | 4.804 | +0.286 | 2.915 | 2.218 | -0.697 |
| `Wv6yp6N1L0WqQ6ZLn6nD8g_doubletalk_with_movement` | 3.534 | 3.643 | +0.109 | 2.868 | 2.301 | -0.568 |
| `WnDjVFWmC0m0WhVq22mRlQ_doubletalk_with_movement` | 4.295 | 4.563 | +0.268 | 1.775 | 1.253 | -0.522 |
| `u0X5XB2KzEGduXtfWfjGDw_doubletalk_with_movement` | 4.614 | 4.819 | +0.205 | 2.601 | 2.086 | -0.515 |
| `W0J6iZv7ZkmHOobCToob4A_doubletalk_with_movement` | 4.399 | 4.535 | +0.136 | 2.020 | 1.513 | -0.507 |
| `WAx9ADn1O00xxkqYq0hPlg_doubletalk_with_movement` | 4.150 | 4.163 | +0.012 | 3.433 | 2.934 | -0.499 |
| `XqvGR01tJkan17zltLs38Q_doubletalk_with_movement` | 4.452 | 4.285 | -0.167 | 3.948 | 3.463 | -0.486 |
| `iyuYIcszXku7BWYOOwqh5Q_doubletalk_with_movement` | 4.887 | 4.858 | -0.029 | 2.032 | 1.548 | -0.484 |
| `wY00iJ3cE0aQsjt0m1tC0g_doubletalk_with_movement` | 4.392 | 4.400 | +0.007 | 3.106 | 2.636 | -0.471 |
| `sRCs6SKo6kC0xire475q0A_doubletalk_with_movement` | 3.882 | 4.143 | +0.261 | 3.113 | 2.662 | -0.450 |
| `WqEYNwalSUebZxaeYVay2g_doubletalk_with_movement` | 3.867 | 3.957 | +0.090 | 2.617 | 2.179 | -0.438 |
| `m6ciKvH6AEe7Yi2ptKjj1g_doubletalk_with_movement` | 3.614 | 3.926 | +0.312 | 2.117 | 1.721 | -0.396 |
| `zpiSOkxpHkCs5SqdOo5ZIQ_doubletalk_with_movement` | 4.077 | 4.165 | +0.087 | 3.790 | 3.452 | -0.338 |
| `KSN5Jrzo7kaixP0z8xfr4Q_doubletalk_with_movement` | 3.877 | 4.197 | +0.321 | 3.041 | 2.753 | -0.288 |
| `hvY1v0viv0yMdAXKa2y1aw_doubletalk_with_movement` | 4.012 | 4.036 | +0.024 | 2.823 | 2.537 | -0.286 |
| `XTqo1aOXDEiqyWTFK99I5Q_doubletalk_with_movement` | 4.789 | 4.853 | +0.064 | 1.834 | 1.630 | -0.204 |

### DT_static
| stem | A_echo | B_echo | Δecho | A_deg | B_deg | Δdeg |
|---|---:|---:|---:|---:|---:|---:|
| `nVUnxqHLr0GTN7shWid1Ow_doubletalk` | 4.687 | 4.530 | -0.157 | 3.223 | 1.748 | -1.475 |
| `MYrVxVEMxkaE7OuyTUmI0Q_doubletalk` | 4.065 | 4.660 | +0.595 | 2.869 | 1.523 | -1.346 |
| `W0J6iZv7ZkmHOobCToob4A_doubletalk` | 4.576 | 4.949 | +0.373 | 2.752 | 1.652 | -1.100 |
| `xnpFE06ShUea4Jn1Wu7EzQ_doubletalk` | 4.126 | 4.479 | +0.353 | 3.853 | 2.777 | -1.076 |
| `49IIo03GZ0CYQOmeA3A0BA_doubletalk` | 4.400 | 4.462 | +0.063 | 3.923 | 2.851 | -1.072 |
| `wJVPo4lexUK40x0nuK0KWg_doubletalk` | 4.187 | 4.192 | +0.005 | 3.906 | 2.947 | -0.959 |
| `WAx9ADn1O00xxkqYq0hPlg_doubletalk` | 4.409 | 4.560 | +0.152 | 2.825 | 1.871 | -0.955 |
| `ql7yTcebJU20VE5qpW0kCA_doubletalk` | 3.852 | 3.740 | -0.111 | 3.149 | 2.237 | -0.911 |
| `xofDX004bkqiOv9YOxmGVQ_doubletalk` | 4.448 | 4.717 | +0.269 | 3.011 | 2.106 | -0.905 |
| `yc5bFUGsR0GSfiGwTTpRWg_doubletalk` | 4.544 | 4.720 | +0.175 | 1.987 | 1.306 | -0.681 |
| `xNr7L0xsLUG4B9oUqW0V4Q_doubletalk` | 3.912 | 4.290 | +0.378 | 2.458 | 1.793 | -0.666 |
| `wyx9K4tvB0qmAoFaCLBeuA_doubletalk` | 4.250 | 4.049 | -0.201 | 3.428 | 2.782 | -0.646 |
| `YCmmUCd3aEWd0V4s7MJQ8g_doubletalk` | 4.453 | 4.548 | +0.095 | 2.492 | 1.896 | -0.596 |
| `o7yLy0sI9kCpV0HgSDfL8A_doubletalk` | 4.052 | 4.181 | +0.128 | 2.648 | 2.094 | -0.554 |
| `zzCIhneJ8UKTWZ48U0kRXw_doubletalk` | 4.465 | 4.244 | -0.221 | 2.872 | 2.359 | -0.513 |
| `0I0XMl3M0ECO0U1N0cJvpg_doubletalk` | 4.333 | 4.576 | +0.242 | 2.388 | 1.895 | -0.493 |
| `kOGPX6kHskOaKSZdLGNz8A_doubletalk` | 4.061 | 3.973 | -0.088 | 3.567 | 3.079 | -0.488 |
| `X9dNu4tXqUqx7qXBrpcbLA_doubletalk` | 3.939 | 3.848 | -0.091 | 2.601 | 2.125 | -0.476 |
| `w5XDRNfB2Ei2UoUDtrTkzg_doubletalk` | 4.415 | 4.103 | -0.312 | 2.808 | 2.338 | -0.470 |
| `WtQs4a0YeU2B0dQWhS7gmg_doubletalk` | 4.438 | 4.646 | +0.208 | 2.313 | 1.859 | -0.454 |

### FS_movement
| stem | A_echo | B_echo | Δecho | A_deg | B_deg | Δdeg |
|---|---:|---:|---:|---:|---:|---:|
| `V6Mw0Ti8RUSkzvMGB4WGiw_farend_singletalk_with_movement` | 3.469 | 4.113 | +0.643 | 4.999 | 4.998 | -0.001 |
| `QkRkwwFKVEar0WtcuvJsZg_farend_singletalk_with_movement` | 4.020 | 4.263 | +0.243 | 4.999 | 4.998 | -0.001 |
| `I2bme08keUmAnyJRKNYDGQ_farend_singletalk_with_movement` | 3.385 | 4.168 | +0.782 | 5.000 | 4.999 | -0.001 |
| `kwolfjBXWEOJmdbDdFoTVQ_farend_singletalk_with_movement` | 3.930 | 4.065 | +0.135 | 4.999 | 4.998 | -0.001 |
| `z4PqfBhq2E01IDBkTH0gnw_farend_singletalk_with_movement` | 3.623 | 3.743 | +0.120 | 5.000 | 4.999 | -0.001 |
| `4pN9yn7mhEa5iDiKnr5jlw_farend_singletalk_with_movement` | 3.856 | 4.258 | +0.402 | 5.000 | 4.999 | -0.001 |
| `mXuYaMbcZka0TpdHDdTlWA_farend_singletalk_with_movement` | 3.644 | 4.082 | +0.437 | 4.999 | 4.998 | -0.001 |
| `xFk7igecuke0R5JMfREyDg_farend_singletalk_with_movement` | 2.625 | 3.659 | +1.035 | 5.000 | 4.999 | -0.001 |
| `jtYTdZm3lUmFVNibJWq8YQ_farend_singletalk_with_movement` | 4.478 | 4.505 | +0.028 | 4.999 | 4.998 | -0.001 |
| `nlSSRl4k50Gq2mIRYlMBCg_farend_singletalk_with_movement` | 3.560 | 3.670 | +0.110 | 5.000 | 4.999 | -0.001 |
| `Tgtk8jp1zkqmKzsmdrKt0g_farend_singletalk_with_movement` | 4.095 | 4.149 | +0.054 | 4.999 | 4.999 | -0.001 |
| `XTqo1aOXDEiqyWTFK99I5Q_farend_singletalk_with_movement` | 4.178 | 4.385 | +0.207 | 5.000 | 4.999 | -0.001 |
| `hvY1v0viv0yMdAXKa2y1aw_farend_singletalk_with_movement` | 3.708 | 4.182 | +0.475 | 4.999 | 4.999 | -0.000 |
| `oQK3bVihI0qel9As840Zzw_farend_singletalk_with_movement` | 2.712 | 3.639 | +0.927 | 5.000 | 5.000 | -0.000 |
| `vjW8NP6JgUC3ved1NRJwbQ_farend_singletalk_with_movement` | 3.779 | 3.920 | +0.141 | 5.000 | 4.999 | -0.000 |
| `N2rQLbnp2UOg2QFRaggbDw_farend_singletalk_with_movement` | 3.481 | 3.783 | +0.302 | 5.000 | 4.999 | -0.000 |
| `Fi80N5kW9U6nwaoS04O3vQ_farend_singletalk_with_movement` | 3.145 | 3.514 | +0.368 | 5.000 | 4.999 | -0.000 |
| `VNkNShj97UajHDVbSmIG0g_farend_singletalk_with_movement` | 3.362 | 3.005 | -0.357 | 5.000 | 4.999 | -0.000 |
| `hF9Lfjcn9kGQ4430uAbINA_farend_singletalk_with_movement` | 3.629 | 3.469 | -0.160 | 4.999 | 4.999 | -0.000 |
| `xYuPW7feGkyc8a1rfcDv9w_farend_singletalk_with_movement` | 4.249 | 4.231 | -0.019 | 4.998 | 4.998 | -0.000 |

### FS_static
| stem | A_echo | B_echo | Δecho | A_deg | B_deg | Δdeg |
|---|---:|---:|---:|---:|---:|---:|
| `sRCs6SKo6kC0xire475q0A_farend_singletalk` | 3.611 | 3.673 | +0.063 | 4.999 | 4.996 | -0.003 |
| `nZQu09pizke3LNOn6uaU0A_farend_singletalk` | 4.009 | 4.102 | +0.093 | 4.999 | 4.997 | -0.002 |
| `mXuYaMbcZka0TpdHDdTlWA_farend_singletalk` | 3.209 | 4.153 | +0.944 | 4.999 | 4.998 | -0.001 |
| `U70FA4mdkEu0FDuj0nTBdA_farend_singletalk` | 3.181 | 3.995 | +0.814 | 5.000 | 4.998 | -0.001 |
| `JtodX3Ug6Eu5TYu0HN5IOw_farend_singletalk` | 3.240 | 4.248 | +1.008 | 5.000 | 4.999 | -0.001 |
| `1fvt8ajGxk2OhS7UglBjoA_farend_singletalk` | 3.159 | 4.029 | +0.871 | 5.000 | 4.999 | -0.001 |
| `SR68lGQwTUy508j0P8BKZQ_farend_singletalk` | 3.898 | 4.113 | +0.215 | 4.999 | 4.998 | -0.001 |
| `Y7w0W4v9BEihm8Z06BxZfQ_farend_singletalk` | 4.031 | 4.463 | +0.432 | 5.000 | 4.999 | -0.001 |
| `XTqo1aOXDEiqyWTFK99I5Q_farend_singletalk` | 4.138 | 4.241 | +0.103 | 5.000 | 4.999 | -0.001 |
| `KgZ0y2EQJ0a4jvtsznBrvw_farend_singletalk` | 4.049 | 3.967 | -0.082 | 4.999 | 4.998 | -0.001 |
| `Y4zG6bHup06zWMoq3OvZqQ_farend_singletalk` | 3.771 | 4.036 | +0.265 | 4.999 | 4.999 | -0.001 |
| `zykCkY0BZEWhtSbeZJm7pw_farend_singletalk` | 3.760 | 4.165 | +0.405 | 5.000 | 4.999 | -0.001 |
| `Pu8CtSffMUiINQAhSKvlfw_farend_singletalk` | 4.141 | 4.372 | +0.231 | 5.000 | 4.999 | -0.001 |
| `yMXhYzMNDUiLK00Tf3fW8w_farend_singletalk` | 4.230 | 4.302 | +0.073 | 4.999 | 4.999 | -0.001 |
| `vF1LKDSGbUGtp0pR6Fzb3A_farend_singletalk` | 4.366 | 4.209 | -0.157 | 4.999 | 4.998 | -0.001 |
| `0KjzXA3g20qsd8zmSekADw_farend_singletalk` | 3.936 | 4.126 | +0.191 | 4.999 | 4.998 | -0.001 |
| `SUYzW4QT30yxKUq7OGvZKg_farend_singletalk` | 3.813 | 3.928 | +0.115 | 4.999 | 4.999 | -0.001 |
| `w9Cji060a0Ss7zLxa05Xhw_farend_singletalk` | 3.910 | 3.831 | -0.079 | 4.998 | 4.998 | -0.001 |
| `LQhlYoXXiUevFuxMKwWB0Q_farend_singletalk` | 4.295 | 4.239 | -0.056 | 4.998 | 4.997 | -0.001 |
| `o2wfdvOGwU6M8Fmn2dCvOA_farend_singletalk` | 3.087 | 3.879 | +0.792 | 5.000 | 4.999 | -0.001 |

### NE
| stem | A_echo | B_echo | Δecho | A_deg | B_deg | Δdeg |
|---|---:|---:|---:|---:|---:|---:|
| `XnfMDZLl0U2WvLRphiGJ6A_nearend_singletalk` | 4.996 | 4.995 | -0.001 | 4.328 | 4.280 | -0.048 |
| `UQbN1vEVO0avUTkFYtQ0mg_nearend_singletalk` | 4.999 | 4.999 | -0.001 | 4.140 | 4.129 | -0.011 |
| `wJVPo4lexUK40x0nuK0KWg_nearend_singletalk` | 4.999 | 4.998 | -0.000 | 4.212 | 4.202 | -0.009 |
| `NN7yhG2XTEqq46X8X0yLfA_nearend_singletalk` | 4.998 | 4.998 | -0.000 | 4.196 | 4.193 | -0.003 |
| `yN0NYysKnUW1PsMYdU4tQA_nearend_singletalk` | 5.000 | 5.000 | +0.000 | 3.374 | 3.372 | -0.002 |
| `014AzuqPZku2004NbTTmcA_nearend_singletalk` | 4.999 | 4.999 | +0.000 | 4.354 | 4.354 | +0.000 |
| `021g8E0mLEWnaPGZo209gA_nearend_singletalk` | 4.999 | 4.999 | +0.000 | 4.545 | 4.545 | +0.000 |
| `05bNc8DKykeqiFyvLWlXmA_nearend_singletalk` | 4.997 | 4.997 | +0.000 | 4.475 | 4.475 | +0.000 |
| `06Q90a0wkkulvuJBJGQqzQ_nearend_singletalk` | 4.997 | 4.997 | +0.000 | 4.359 | 4.359 | +0.000 |
| `06S6EY1JpU2qpe409kUaew_nearend_singletalk` | 4.997 | 4.997 | +0.000 | 4.259 | 4.259 | +0.000 |
| `08Mo9p6KVUapuF8PNAMWaw_nearend_singletalk` | 4.999 | 4.999 | +0.000 | 3.515 | 3.515 | +0.000 |
| `0B5E1viJNEGA7aU0ZlALQQ_nearend_singletalk` | 4.996 | 4.996 | +0.000 | 4.177 | 4.177 | +0.000 |
| `0CD12tXmdU65rRTkf0okKg_nearend_singletalk` | 4.998 | 4.998 | +0.000 | 4.415 | 4.415 | +0.000 |
| `0FcYh9r6G0qQrbX7taxQ6g_nearend_singletalk` | 4.997 | 4.997 | +0.000 | 4.526 | 4.526 | +0.000 |
| `0R3gOcj9uE0Abp3sjrti0w_nearend_singletalk` | 4.994 | 4.994 | +0.000 | 4.140 | 4.140 | +0.000 |
| `0b8WY2HTrEOOT2Jf0br3hQ_nearend_singletalk` | 4.996 | 4.996 | +0.000 | 4.375 | 4.375 | +0.000 |
| `17ng7Oa1k0qZI0pV0KznQw_nearend_singletalk` | 4.998 | 4.998 | +0.000 | 4.381 | 4.381 | +0.000 |
| `2RLNXS3YSEeq4lmFpF9QfA_nearend_singletalk` | 4.997 | 4.997 | +0.000 | 4.396 | 4.396 | +0.000 |
| `2f9FDnH8u0q829njqEuVrQ_nearend_singletalk` | 4.998 | 4.998 | +0.000 | 3.456 | 3.456 | +0.000 |
| `2wFE5X7bmEWTjphPVFtatg_nearend_singletalk` | 4.997 | 4.997 | +0.000 | 4.238 | 4.238 | +0.000 |

## Top 20 worst Δecho per bucket (NORMAL, XRTnTUjU excluded)

### DT_movement
| stem | A_echo | B_echo | Δecho | A_deg | B_deg | Δdeg |
|---|---:|---:|---:|---:|---:|---:|
| `IrQvqOTCmEWMXn9k2ICtRQ_doubletalk_with_movement` | 2.765 | 1.642 | -1.123 | 3.537 | 3.711 | +0.175 |
| `kwolfjBXWEOJmdbDdFoTVQ_doubletalk_with_movement` | 4.510 | 3.980 | -0.530 | 2.227 | 2.297 | +0.070 |
| `y2ZCo1jA6kGdWZ0MgoaZ5w_doubletalk_with_movement` | 4.070 | 3.552 | -0.518 | 2.613 | 3.325 | +0.712 |
| `V0JqgjlrB0Ke9y91r0rxNw_doubletalk_with_movement` | 4.524 | 4.007 | -0.517 | 1.636 | 2.296 | +0.660 |
| `N2rQLbnp2UOg2QFRaggbDw_doubletalk_with_movement` | 4.131 | 3.682 | -0.449 | 1.720 | 1.927 | +0.207 |
| `49IIo03GZ0CYQOmeA3A0BA_doubletalk_with_movement` | 4.465 | 4.024 | -0.441 | 2.811 | 2.741 | -0.070 |
| `zzCIhneJ8UKTWZ48U0kRXw_doubletalk_with_movement` | 4.499 | 4.085 | -0.413 | 1.849 | 2.177 | +0.328 |
| `7GTxyTksSUqCnP5y0ILG4A_doubletalk_with_movement` | 4.100 | 3.693 | -0.407 | 1.472 | 2.125 | +0.653 |
| `kZogUfYct0qMwSqvRTwOVg_doubletalk_with_movement` | 4.218 | 3.827 | -0.391 | 1.833 | 2.593 | +0.760 |
| `TGZ5Wq0SCUCOXPsfee3uMQ_doubletalk_with_movement` | 3.966 | 3.595 | -0.371 | 2.285 | 2.775 | +0.491 |
| `ZJYUt0O0AEKSQ9LJ8z7t0A_doubletalk_with_movement` | 4.526 | 4.181 | -0.345 | 1.991 | 3.146 | +1.156 |
| `SwfEwuGDlkWYy9pb4H00eQ_doubletalk_with_movement` | 4.503 | 4.176 | -0.327 | 2.476 | 2.388 | -0.088 |
| `XGDaZuEkE0WU4IN0Yi4XtA_doubletalk_with_movement` | 4.320 | 4.001 | -0.319 | 2.028 | 2.946 | +0.918 |
| `wVYSGVTTakih9twI4xlDWQ_doubletalk_with_movement` | 4.199 | 3.892 | -0.307 | 2.203 | 3.192 | +0.989 |
| `QkRkwwFKVEar0WtcuvJsZg_doubletalk_with_movement` | 4.555 | 4.265 | -0.290 | 1.417 | 2.519 | +1.102 |
| `W4r0UCjieEuM0u930spvug_doubletalk_with_movement` | 4.796 | 4.522 | -0.275 | 2.052 | 2.249 | +0.197 |
| `sx6mxKBQpkq520m64BwUdQ_doubletalk_with_movement` | 3.951 | 3.689 | -0.262 | 3.513 | 3.388 | -0.125 |
| `V6Mw0Ti8RUSkzvMGB4WGiw_doubletalk_with_movement` | 4.426 | 4.171 | -0.255 | 2.934 | 2.911 | -0.023 |
| `lV0kQN0hR0ySmE0bQhuYbw_doubletalk_with_movement` | 3.919 | 3.668 | -0.251 | 1.846 | 2.889 | +1.044 |
| `qVd1gtwQ0k2lVRqPVp1NKQ_doubletalk_with_movement` | 4.743 | 4.493 | -0.250 | 2.717 | 2.941 | +0.224 |

### DT_static
| stem | A_echo | B_echo | Δecho | A_deg | B_deg | Δdeg |
|---|---:|---:|---:|---:|---:|---:|
| `XCXcCwUPY0GmrtqtJ6xY2g_doubletalk` | 4.283 | 3.697 | -0.586 | 1.819 | 2.283 | +0.464 |
| `yM2wHof9U06yVPJfemZ3hg_doubletalk` | 4.427 | 3.862 | -0.565 | 2.452 | 2.776 | +0.324 |
| `s0oJqM6Y1UCHSVmHmgsx4Q_doubletalk` | 4.086 | 3.547 | -0.539 | 1.592 | 2.442 | +0.850 |
| `uS9t2QYDckeO7SnQNYZVcg_doubletalk` | 4.094 | 3.570 | -0.524 | 2.127 | 2.722 | +0.595 |
| `Wv6yp6N1L0WqQ6ZLn6nD8g_doubletalk` | 4.212 | 3.713 | -0.498 | 1.730 | 2.351 | +0.622 |
| `XnfMDZLl0U2WvLRphiGJ6A_doubletalk` | 4.337 | 3.889 | -0.448 | 1.458 | 2.205 | +0.747 |
| `WYKA2zSbcE2gRBPHvMLQZw_doubletalk` | 4.152 | 3.739 | -0.413 | 1.990 | 3.144 | +1.154 |
| `VNgRsWxMdkaUx1gKV9W1Zw_doubletalk` | 4.399 | 4.006 | -0.393 | 1.460 | 1.896 | +0.436 |
| `XTqo1aOXDEiqyWTFK99I5Q_doubletalk` | 4.296 | 3.903 | -0.392 | 2.471 | 2.506 | +0.035 |
| `UmlD9X38NECNoJKm0oyf4w_doubletalk` | 4.337 | 3.950 | -0.387 | 2.260 | 1.978 | -0.281 |
| `kz23X4pDSEiPmWtw2Qx00Q_doubletalk` | 4.379 | 4.006 | -0.373 | 2.223 | 2.307 | +0.084 |
| `TGZ5Wq0SCUCOXPsfee3uMQ_doubletalk` | 4.089 | 3.743 | -0.346 | 2.120 | 3.337 | +1.217 |
| `IUp2c0A4yEyttmLfJ3t0Xw_doubletalk` | 4.552 | 4.207 | -0.346 | 1.854 | 2.414 | +0.560 |
| `mrmDEdQMpk6hJnMqn59pOQ_doubletalk` | 4.057 | 3.715 | -0.343 | 3.022 | 3.448 | +0.427 |
| `xYuPW7feGkyc8a1rfcDv9w_doubletalk` | 4.549 | 4.210 | -0.339 | 1.303 | 1.937 | +0.634 |
| `y3EDAxnBRUCjAZ6iLUxs0w_doubletalk` | 4.656 | 4.317 | -0.338 | 2.338 | 2.707 | +0.369 |
| `x7VuXKV4LUu0MSdRHyOeTg_doubletalk` | 4.460 | 4.142 | -0.319 | 2.697 | 3.130 | +0.433 |
| `w5XDRNfB2Ei2UoUDtrTkzg_doubletalk` | 4.415 | 4.103 | -0.312 | 2.808 | 2.338 | -0.470 |
| `JjCzlhn3gEiBQvfJtPNJ9A_doubletalk` | 4.529 | 4.246 | -0.284 | 2.060 | 2.724 | +0.664 |
| `XPzc0mE02UGND0qMzmW52g_doubletalk` | 4.431 | 4.161 | -0.270 | 2.661 | 3.681 | +1.020 |

### FS_movement
| stem | A_echo | B_echo | Δecho | A_deg | B_deg | Δdeg |
|---|---:|---:|---:|---:|---:|---:|
| `nV9v63E5CUKtKTjha8dtdQ_farend_singletalk_with_movement` | 3.941 | 2.888 | -1.054 | 4.999 | 5.000 | +0.001 |
| `XuguA1uJAE0bWT0xXRDdeA_farend_singletalk_with_movement` | 3.258 | 2.552 | -0.706 | 5.000 | 5.000 | +0.000 |
| `Je6gJ7y1PECStwxnrOe9aA_farend_singletalk_with_movement` | 4.075 | 3.400 | -0.675 | 4.999 | 5.000 | +0.001 |
| `s0oJqM6Y1UCHSVmHmgsx4Q_farend_singletalk_with_movement` | 2.955 | 2.355 | -0.600 | 5.000 | 5.000 | +0.000 |
| `lxLsvT1rY0mdtZuRogM06Q_farend_singletalk_with_movement` | 4.001 | 3.414 | -0.587 | 4.999 | 4.999 | +0.000 |
| `Ixf70mgKwkCoFYq32586cw_farend_singletalk_with_movement` | 3.988 | 3.448 | -0.540 | 4.999 | 4.999 | +0.000 |
| `QK70KpLuZ0O43BBSWEZvHg_farend_singletalk_with_movement` | 3.797 | 3.260 | -0.537 | 5.000 | 5.000 | +0.000 |
| `ZWq0X5sPiUe0lQjZdCPSeQ_farend_singletalk_with_movement` | 2.918 | 2.386 | -0.532 | 5.000 | 5.000 | +0.000 |
| `Hq00pd6Ey0mGtuMFRoF79w_farend_singletalk_with_movement` | 3.854 | 3.322 | -0.532 | 4.999 | 5.000 | +0.000 |
| `Ja8OngfthkOCmL8ldcRNyg_farend_singletalk_with_movement` | 4.415 | 3.905 | -0.510 | 4.999 | 4.999 | +0.001 |
| `ML4MF3Mea0yurjceNQPfNA_farend_singletalk_with_movement` | 4.028 | 3.519 | -0.510 | 4.999 | 4.999 | +0.001 |
| `0luXwWjGEEC9G5nf0yTVXw_farend_singletalk_with_movement` | 2.770 | 2.270 | -0.501 | 5.000 | 5.000 | +0.000 |
| `nyT6FUUdu0W8UpvjP1rRgQ_farend_singletalk_with_movement` | 3.928 | 3.445 | -0.483 | 4.999 | 5.000 | +0.000 |
| `SgKY30fjT0G8e3kQL0RHSQ_farend_singletalk_with_movement` | 3.132 | 2.658 | -0.473 | 5.000 | 5.000 | +0.000 |
| `zONvcX0qYkuaAViV5PXcYg_farend_singletalk_with_movement` | 3.151 | 2.681 | -0.471 | 5.000 | 5.000 | +0.000 |
| `XXz0qkUSd0GT4dsywxpfJg_farend_singletalk_with_movement` | 3.873 | 3.411 | -0.462 | 4.999 | 5.000 | +0.001 |
| `WYKA2zSbcE2gRBPHvMLQZw_farend_singletalk_with_movement` | 3.286 | 2.859 | -0.427 | 5.000 | 5.000 | +0.000 |
| `SwfEwuGDlkWYy9pb4H00eQ_farend_singletalk_with_movement` | 3.908 | 3.483 | -0.425 | 5.000 | 5.000 | +0.000 |
| `sx6mxKBQpkq520m64BwUdQ_farend_singletalk_with_movement` | 3.222 | 2.824 | -0.398 | 5.000 | 5.000 | +0.000 |
| `hVqUmGvIlkO0LBUoE06Q3w_farend_singletalk_with_movement` | 2.891 | 2.505 | -0.386 | 5.000 | 5.000 | +0.000 |

### FS_static
| stem | A_echo | B_echo | Δecho | A_deg | B_deg | Δdeg |
|---|---:|---:|---:|---:|---:|---:|
| `7GTxyTksSUqCnP5y0ILG4A_farend_singletalk` | 3.692 | 2.281 | -1.411 | 4.999 | 5.000 | +0.001 |
| `wlAXM0iDgkm06i7UdRww1w_farend_singletalk` | 3.811 | 2.417 | -1.394 | 4.999 | 5.000 | +0.001 |
| `9bSnA8CNBUSsJeIVCELzSQ_farend_singletalk` | 4.040 | 3.339 | -0.701 | 4.999 | 5.000 | +0.001 |
| `IrQvqOTCmEWMXn9k2ICtRQ_farend_singletalk` | 3.025 | 2.384 | -0.641 | 5.000 | 5.000 | +0.000 |
| `ksP3OuSnpUa9Si2ttiUSoA_farend_singletalk` | 3.896 | 3.271 | -0.625 | 4.998 | 5.000 | +0.001 |
| `sUQrHEPAoEmIvHclpi1tRQ_farend_singletalk` | 3.498 | 2.907 | -0.591 | 5.000 | 5.000 | +0.000 |
| `hVqUmGvIlkO0LBUoE06Q3w_farend_singletalk` | 2.715 | 2.145 | -0.570 | 5.000 | 5.000 | +0.000 |
| `khqZY41lNEyIvMf2ZNJuVA_farend_singletalk` | 3.954 | 3.394 | -0.559 | 4.999 | 5.000 | +0.001 |
| `j0awp3hXrkCSqhR748U3iQ_farend_singletalk` | 3.959 | 3.401 | -0.558 | 4.999 | 5.000 | +0.001 |
| `NY3kZioAm0KwR45wIVe2Sg_farend_singletalk` | 4.379 | 3.826 | -0.552 | 4.999 | 4.999 | +0.000 |
| `p0mhFbhV6kGJgjd0RTTIIw_farend_singletalk` | 4.092 | 3.550 | -0.541 | 4.999 | 5.000 | +0.000 |
| `IxgmaPghzUGnR6sxrbGU3Q_farend_singletalk` | 4.278 | 3.750 | -0.528 | 4.999 | 5.000 | +0.001 |
| `SFvlSygv4ke9wCrv8LWvYQ_farend_singletalk` | 3.661 | 3.144 | -0.517 | 5.000 | 5.000 | +0.000 |
| `JLNgGcvTNEqbTDbc28wLkg_farend_singletalk` | 3.048 | 2.544 | -0.504 | 5.000 | 5.000 | +0.000 |
| `yM2wHof9U06yVPJfemZ3hg_farend_singletalk` | 3.993 | 3.504 | -0.490 | 4.999 | 4.999 | -0.000 |
| `rEVrdY1tHE00AGubEOasVA_farend_singletalk` | 3.862 | 3.408 | -0.454 | 4.999 | 4.999 | +0.001 |
| `yZs0i8NpJkypsV8QyvduzQ_farend_singletalk` | 4.062 | 3.616 | -0.447 | 4.999 | 5.000 | +0.000 |
| `pU21kfoo0UOz0fPMJFfydg_farend_singletalk` | 3.154 | 2.737 | -0.417 | 5.000 | 5.000 | +0.000 |
| `r7U6JmcRl0ibIh0mN3CP9g_farend_singletalk` | 3.464 | 3.062 | -0.402 | 5.000 | 5.000 | +0.000 |
| `vSZmpMJI0kKv30P2GhgV1Q_farend_singletalk` | 3.246 | 2.867 | -0.379 | 5.000 | 5.000 | +0.000 |

### NE
| stem | A_echo | B_echo | Δecho | A_deg | B_deg | Δdeg |
|---|---:|---:|---:|---:|---:|---:|
| `lxLsvT1rY0mdtZuRogM06Q_nearend_singletalk` | 4.997 | 4.992 | -0.005 | 4.335 | 4.439 | +0.103 |
| `M2nCWTe4nUWo8IOhj2IwNg_nearend_singletalk` | 5.000 | 4.995 | -0.004 | 4.078 | 4.477 | +0.399 |
| `0bFTCgvGGUaLnVt5LZHXaA_nearend_singletalk` | 5.000 | 4.996 | -0.004 | 3.833 | 4.413 | +0.579 |
| `uLl640xveUuHp2kEtOCTeQ_nearend_singletalk` | 4.997 | 4.995 | -0.003 | 4.164 | 4.224 | +0.059 |
| `SswMECiwjUKY4BWidXhehg_nearend_singletalk` | 5.000 | 4.997 | -0.003 | 3.967 | 4.269 | +0.302 |
| `OLjlc92QWU6fwuN4ytCPQg_nearend_singletalk` | 5.000 | 4.997 | -0.002 | 4.082 | 4.232 | +0.150 |
| `qkGW9Frbs0Gq5gdfsztA2g_nearend_singletalk` | 4.999 | 4.997 | -0.002 | 3.883 | 4.140 | +0.258 |
| `jtxWjQFfUUqNKJHxJKLIJA_nearend_singletalk` | 4.996 | 4.995 | -0.002 | 4.029 | 4.139 | +0.111 |
| `LHsrJBRGnUKiMC2mihEr0g_nearend_singletalk` | 4.999 | 4.997 | -0.002 | 4.178 | 4.247 | +0.070 |
| `09fKZjX0QkCzKPTRe29DaQ_nearend_singletalk` | 4.997 | 4.995 | -0.002 | 4.098 | 4.202 | +0.105 |
| `XnfMDZLl0U2WvLRphiGJ6A_nearend_singletalk` | 4.996 | 4.995 | -0.001 | 4.328 | 4.280 | -0.048 |
| `wlAXM0iDgkm06i7UdRww1w_nearend_singletalk` | 5.000 | 4.999 | -0.001 | 3.298 | 3.884 | +0.587 |
| `wyx9K4tvB0qmAoFaCLBeuA_nearend_singletalk` | 4.999 | 4.999 | -0.001 | 3.891 | 4.246 | +0.355 |
| `DW3NNHoKl02cAfEkahzsGg_nearend_singletalk` | 5.000 | 4.999 | -0.001 | 3.024 | 3.350 | +0.326 |
| `S73GmH9ok0GBbaG3esxbQQ_nearend_singletalk` | 5.000 | 4.999 | -0.001 | 3.214 | 3.836 | +0.622 |
| `UQbN1vEVO0avUTkFYtQ0mg_nearend_singletalk` | 4.999 | 4.999 | -0.001 | 4.140 | 4.129 | -0.011 |
| `3O8efztOJk6Y4kB5qmQ2wQ_nearend_singletalk` | 4.997 | 4.997 | -0.001 | 4.068 | 4.144 | +0.076 |
| `wcP1HWQqv0aX1WAguwuUPg_nearend_singletalk` | 4.999 | 4.999 | -0.000 | 3.793 | 4.053 | +0.261 |
| `mUX1GNN0nEyuL009m50dtg_nearend_singletalk` | 5.000 | 4.999 | -0.000 | 4.183 | 4.298 | +0.116 |
| `OlEt0Ltk0UKogetYnqirSw_nearend_singletalk` | 4.998 | 4.998 | -0.000 | 4.015 | 4.063 | +0.048 |
