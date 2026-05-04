# Bench result — current

Dataset: /Users/mingyu/Desktop/novatek/SE/AEC/wav/aec_challenge_blind
Output dir: experiments/r7_p2_mix_r10/eval_output
Cases scored: 800

## Bucket means

| Bucket | n | echo (↑) | deg (↑) |
|---|---:|---:|---:|
| FS_static | 169 | 3.540 | 5.000 |
| FS_movement | 131 | 3.685 | 5.000 |
| DT_static | 186 | 3.956 | 2.453 |
| DT_movement | 114 | 3.880 | 2.444 |
| NE | 200 | 4.998 | 3.989 |

## Δ vs baseline (baseline_v381_seeded)

| Bucket | Δecho | Δdeg | verdict |
|---|---:|---:|---|
| FS_static | -0.228 | +0.000 | FS echo regress |
| FS_movement | -0.142 | +0.000 | FS echo regress |
| DT_static | -0.296 | +0.197 | ok |
| DT_movement | -0.240 | +0.158 | ok |
| NE | +0.000 | -0.018 | NE deg regress |

## Worst-20 per bucket

### FS_static (sorted by echo ascending)

| stem | echo |
|---|---:|
| `7GTxyTksSUqCnP5y0ILG4A_farend_singletalk` | 1.411 |
| `pcb1Nh0Z3k0WS9a7gBEuqg_farend_singletalk` | 1.693 |
| `JtodX3Ug6Eu5TYu0HN5IOw_farend_singletalk` | 1.698 |
| `S22FCqKDWUyymN1YbpItIw_farend_singletalk` | 1.789 |
| `9xjhiFbGo06hdQIsHTS6qA_farend_singletalk` | 1.809 |
| `IrQvqOTCmEWMXn9k2ICtRQ_farend_singletalk` | 1.816 |
| `pmzLFdKTzEixfU0l0furvA_farend_singletalk` | 1.836 |
| `VJfVUwJs4k25ziMNvJb43A_farend_singletalk` | 1.845 |
| `hVqUmGvIlkO0LBUoE06Q3w_farend_singletalk` | 1.848 |
| `Gsy0lC5QSUi540hiax9XtA_farend_singletalk` | 1.954 |
| `JteZUZ4JYkeD4k2rcVbqHg_farend_singletalk` | 2.087 |
| `geGGo8g9UE2MYkdjxtHm8w_farend_singletalk` | 2.130 |
| `s90M7MOTBkqaV4nQPLhKbA_farend_singletalk` | 2.243 |
| `QYHsugUAcUWEQ9WghnG0Jw_farend_singletalk` | 2.292 |
| `LeV1uF4j10Whm0FPG80tmw_farend_singletalk` | 2.300 |
| `zddPqpp1a06xttKdc0iNTA_farend_singletalk` | 2.325 |
| `sUQrHEPAoEmIvHclpi1tRQ_farend_singletalk` | 2.335 |
| `wTgNyKxp0kihhU2K96tHgA_farend_singletalk` | 2.342 |
| `o97uExi7MEqzbanuWK6CCw_farend_singletalk` | 2.358 |
| `N2rQLbnp2UOg2QFRaggbDw_farend_singletalk` | 2.380 |

### FS_movement (sorted by echo ascending)

| stem | echo |
|---|---:|
| `XTqo1aOXDEiqyWTFK99I5Q_farend_singletalk_with_movement` | 1.796 |
| `hVqUmGvIlkO0LBUoE06Q3w_farend_singletalk_with_movement` | 1.972 |
| `S22FCqKDWUyymN1YbpItIw_farend_singletalk_with_movement` | 1.974 |
| `iOyPaxX11UOaUkcscKhq1A_farend_singletalk_with_movement` | 2.038 |
| `0luXwWjGEEC9G5nf0yTVXw_farend_singletalk_with_movement` | 2.121 |
| `IrQvqOTCmEWMXn9k2ICtRQ_farend_singletalk_with_movement` | 2.257 |
| `OXCtw0FVhUWGUBil6Uucdw_farend_singletalk_with_movement` | 2.273 |
| `SgKY30fjT0G8e3kQL0RHSQ_farend_singletalk_with_movement` | 2.307 |
| `TGZ5Wq0SCUCOXPsfee3uMQ_farend_singletalk_with_movement` | 2.311 |
| `VNgRsWxMdkaUx1gKV9W1Zw_farend_singletalk_with_movement` | 2.346 |
| `WH0jN3PY40es2S0LsxmkkQ_farend_singletalk_with_movement` | 2.354 |
| `N2rQLbnp2UOg2QFRaggbDw_farend_singletalk_with_movement` | 2.363 |
| `IqtJR4tjJkWrwUjYorz0Og_farend_singletalk_with_movement` | 2.501 |
| `JjCzlhn3gEiBQvfJtPNJ9A_farend_singletalk_with_movement` | 2.508 |
| `yyvS0Ljh1k0AHMx6cxtNyg_farend_singletalk_with_movement` | 2.573 |
| `XXz0qkUSd0GT4dsywxpfJg_farend_singletalk_with_movement` | 2.600 |
| `XuguA1uJAE0bWT0xXRDdeA_farend_singletalk_with_movement` | 2.619 |
| `vjW8NP6JgUC3ved1NRJwbQ_farend_singletalk_with_movement` | 2.638 |
| `sx6mxKBQpkq520m64BwUdQ_farend_singletalk_with_movement` | 2.652 |
| `5bJUo1K3uEmMrGa9UhGyVg_farend_singletalk_with_movement` | 2.682 |

### DT_static (sorted by deg ascending)

| stem | deg |
|---|---:|
| `QK70KpLuZ0O43BBSWEZvHg_doubletalk` | 1.231 |
| `p0mhFbhV6kGJgjd0RTTIIw_doubletalk` | 1.249 |
| `SgKY30fjT0G8e3kQL0RHSQ_doubletalk` | 1.252 |
| `X0OraQOKtkCVriR0uj0WBQ_doubletalk` | 1.306 |
| `pPBidi8oyUarZCGUrKJEsg_doubletalk` | 1.328 |
| `XuiheB7eUkyJA2XzFIovHQ_doubletalk` | 1.336 |
| `XtUWcfsuykC5fTu7DdAnnw_doubletalk` | 1.357 |
| `V6Mw0Ti8RUSkzvMGB4WGiw_doubletalk` | 1.384 |
| `QkRkwwFKVEar0WtcuvJsZg_doubletalk` | 1.386 |
| `yc5bFUGsR0GSfiGwTTpRWg_doubletalk` | 1.413 |
| `sYQK1rJlwU2XCy20n0Sx9g_doubletalk` | 1.427 |
| `NY3kZioAm0KwR45wIVe2Sg_doubletalk` | 1.513 |
| `xWOALEtAwk2oABUz3vAv6w_doubletalk` | 1.522 |
| `0I0XMl3M0ECO0U1N0cJvpg_doubletalk` | 1.525 |
| `WH7rA6R2zkyopKUrcq9p3A_doubletalk` | 1.558 |
| `wLRC9k23Bk6kmBzZdbkHTA_doubletalk` | 1.579 |
| `wY3Rp9YEjkOke09hMwfsjg_doubletalk` | 1.637 |
| `qVd1gtwQ0k2lVRqPVp1NKQ_doubletalk` | 1.650 |
| `nVUnxqHLr0GTN7shWid1Ow_doubletalk` | 1.654 |
| `Uc4dmejgWUCTvn0XZbMTBw_doubletalk` | 1.659 |

### DT_movement (sorted by deg ascending)

| stem | deg |
|---|---:|
| `XV5L2dn3S06M9GBEu1q3DA_doubletalk_with_movement` | 1.148 |
| `wHmBm7VHfkysBOhjoAXkNA_doubletalk_with_movement` | 1.224 |
| `WqEYNwalSUebZxaeYVay2g_doubletalk_with_movement` | 1.328 |
| `XuiheB7eUkyJA2XzFIovHQ_doubletalk_with_movement` | 1.331 |
| `X7Ua9txMj0aws848JPEbOg_doubletalk_with_movement` | 1.360 |
| `I2bme08keUmAnyJRKNYDGQ_doubletalk_with_movement` | 1.431 |
| `xofDX004bkqiOv9YOxmGVQ_doubletalk_with_movement` | 1.439 |
| `VgSXlJJEI02dytkMm5UTzA_doubletalk_with_movement` | 1.440 |
| `V0JqgjlrB0Ke9y91r0rxNw_doubletalk_with_movement` | 1.517 |
| `QkRkwwFKVEar0WtcuvJsZg_doubletalk_with_movement` | 1.532 |
| `w5XDRNfB2Ei2UoUDtrTkzg_doubletalk_with_movement` | 1.554 |
| `zOiK6oSHp0ib3nHvzLKbRQ_doubletalk_with_movement` | 1.563 |
| `WH7rA6R2zkyopKUrcq9p3A_doubletalk_with_movement` | 1.567 |
| `ql7yTcebJU20VE5qpW0kCA_doubletalk_with_movement` | 1.577 |
| `wWeNtFK0dEG9Wub40bB15A_doubletalk_with_movement` | 1.583 |
| `SwfEwuGDlkWYy9pb4H00eQ_doubletalk_with_movement` | 1.588 |
| `Xvyiz1o0cEijZQ8DT9mB2w_doubletalk_with_movement` | 1.598 |
| `qkGW9Frbs0Gq5gdfsztA2g_doubletalk_with_movement` | 1.603 |
| `OmB0Ht0hmE2crVnftAEtsw_doubletalk_with_movement` | 1.606 |
| `kwolfjBXWEOJmdbDdFoTVQ_doubletalk_with_movement` | 1.732 |

### NE (sorted by deg ascending)

| stem | deg |
|---|---:|
| `SR68lGQwTUy508j0P8BKZQ_nearend_singletalk` | 3.007 |
| `kz23X4pDSEiPmWtw2Qx00Q_nearend_singletalk` | 3.019 |
| `kOtW70qgikKm0F9OEQw22A_nearend_singletalk` | 3.052 |
| `DW3NNHoKl02cAfEkahzsGg_nearend_singletalk` | 3.119 |
| `A50fBOz02kag8CVAhBOF8A_nearend_singletalk` | 3.149 |
| `x7Frsl5wTEKmog3A6lL57g_nearend_singletalk` | 3.157 |
| `08Mo9p6KVUapuF8PNAMWaw_nearend_singletalk` | 3.186 |
| `4pN9yn7mhEa5iDiKnr5jlw_nearend_singletalk` | 3.287 |
| `2f9FDnH8u0q829njqEuVrQ_nearend_singletalk` | 3.319 |
| `pPBidi8oyUarZCGUrKJEsg_nearend_singletalk` | 3.347 |
| `mAJHXrZ5QU62Do1OqTSL6Q_nearend_singletalk` | 3.370 |
| `x1pZr2uWBkulb77zcCfAJA_nearend_singletalk` | 3.370 |
| `LAdLsNzbcE66K0Zs1NE5QA_nearend_singletalk` | 3.371 |
| `IvAFDPUFEk0GuW2VKRHznA_nearend_singletalk` | 3.417 |
| `SFvlSygv4ke9wCrv8LWvYQ_nearend_singletalk` | 3.436 |
| `XCXcCwUPY0GmrtqtJ6xY2g_nearend_singletalk` | 3.448 |
| `E0l0WVPQjEi6AmtbvfSYLA_nearend_singletalk` | 3.457 |
| `wJVPo4lexUK40x0nuK0KWg_nearend_singletalk` | 3.471 |
| `VNkNShj97UajHDVbSmIG0g_nearend_singletalk` | 3.480 |
| `mljUO9k4gUiYCHgXSsfo5A_nearend_singletalk` | 3.484 |
