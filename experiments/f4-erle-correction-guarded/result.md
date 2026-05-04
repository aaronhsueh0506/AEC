# Bench result — f4_erle_guarded

Dataset: /Users/mingyu/Desktop/novatek/SE/AEC/wav/aec_challenge_blind
Output dir: experiments/f4-erle-correction-guarded/output
Cases scored: 800

## Bucket means

| Bucket | n | echo (↑) | deg (↑) |
|---|---:|---:|---:|
| FS_static | 169 | 3.766 | 4.999 |
| FS_movement | 131 | 3.831 | 4.999 |
| DT_static | 186 | 4.253 | 2.254 |
| DT_movement | 114 | 4.123 | 2.284 |
| NE | 200 | 4.998 | 4.007 |

## Δ vs baseline (baseline_v381)

| Bucket | Δecho | Δdeg | verdict |
|---|---:|---:|---|
| FS_static | -0.001 | +0.000 | ok |
| FS_movement | -0.001 | -0.000 | ok |
| DT_static | +0.001 | -0.003 | ok |
| DT_movement | +0.002 | -0.004 | ok |
| NE | -0.000 | +0.000 | ok |

## Worst-20 per bucket

### FS_static (sorted by echo ascending)

| stem | echo |
|---|---:|
| `7GTxyTksSUqCnP5y0ILG4A_farend_singletalk` | 1.312 |
| `9xjhiFbGo06hdQIsHTS6qA_farend_singletalk` | 1.790 |
| `IrQvqOTCmEWMXn9k2ICtRQ_farend_singletalk` | 1.830 |
| `hVqUmGvIlkO0LBUoE06Q3w_farend_singletalk` | 1.870 |
| `pcb1Nh0Z3k0WS9a7gBEuqg_farend_singletalk` | 1.920 |
| `VJfVUwJs4k25ziMNvJb43A_farend_singletalk` | 1.949 |
| `S22FCqKDWUyymN1YbpItIw_farend_singletalk` | 1.965 |
| `JteZUZ4JYkeD4k2rcVbqHg_farend_singletalk` | 2.158 |
| `N2rQLbnp2UOg2QFRaggbDw_farend_singletalk` | 2.201 |
| `JtodX3Ug6Eu5TYu0HN5IOw_farend_singletalk` | 2.297 |
| `geGGo8g9UE2MYkdjxtHm8w_farend_singletalk` | 2.299 |
| `pmzLFdKTzEixfU0l0furvA_farend_singletalk` | 2.310 |
| `LeV1uF4j10Whm0FPG80tmw_farend_singletalk` | 2.453 |
| `s90M7MOTBkqaV4nQPLhKbA_farend_singletalk` | 2.463 |
| `iOyPaxX11UOaUkcscKhq1A_farend_singletalk` | 2.551 |
| `m4789fdio0q92zjf9gvh1Q_farend_singletalk` | 2.568 |
| `60JogIkuEECYHrwVT3UYwg_farend_singletalk` | 2.569 |
| `lV0kQN0hR0ySmE0bQhuYbw_farend_singletalk` | 2.596 |
| `zddPqpp1a06xttKdc0iNTA_farend_singletalk` | 2.659 |
| `GDyfzBkhxEiDbnRZGGOrQQ_farend_singletalk` | 2.677 |

### FS_movement (sorted by echo ascending)

| stem | echo |
|---|---:|
| `S22FCqKDWUyymN1YbpItIw_farend_singletalk_with_movement` | 1.917 |
| `iOyPaxX11UOaUkcscKhq1A_farend_singletalk_with_movement` | 2.000 |
| `hVqUmGvIlkO0LBUoE06Q3w_farend_singletalk_with_movement` | 2.105 |
| `SgKY30fjT0G8e3kQL0RHSQ_farend_singletalk_with_movement` | 2.235 |
| `OXCtw0FVhUWGUBil6Uucdw_farend_singletalk_with_movement` | 2.293 |
| `IrQvqOTCmEWMXn9k2ICtRQ_farend_singletalk_with_movement` | 2.330 |
| `JtodX3Ug6Eu5TYu0HN5IOw_farend_singletalk_with_movement` | 2.432 |
| `JjCzlhn3gEiBQvfJtPNJ9A_farend_singletalk_with_movement` | 2.523 |
| `N2rQLbnp2UOg2QFRaggbDw_farend_singletalk_with_movement` | 2.529 |
| `yyvS0Ljh1k0AHMx6cxtNyg_farend_singletalk_with_movement` | 2.606 |
| `0luXwWjGEEC9G5nf0yTVXw_farend_singletalk_with_movement` | 2.608 |
| `XTqo1aOXDEiqyWTFK99I5Q_farend_singletalk_with_movement` | 2.671 |
| `XuguA1uJAE0bWT0xXRDdeA_farend_singletalk_with_movement` | 2.763 |
| `VNkNShj97UajHDVbSmIG0g_farend_singletalk_with_movement` | 2.772 |
| `TGZ5Wq0SCUCOXPsfee3uMQ_farend_singletalk_with_movement` | 2.795 |
| `xFk7igecuke0R5JMfREyDg_farend_singletalk_with_movement` | 2.931 |
| `LHsrJBRGnUKiMC2mihEr0g_farend_singletalk_with_movement` | 2.984 |
| `ML4MF3Mea0yurjceNQPfNA_farend_singletalk_with_movement` | 3.014 |
| `IqtJR4tjJkWrwUjYorz0Og_farend_singletalk_with_movement` | 3.035 |
| `sx6mxKBQpkq520m64BwUdQ_farend_singletalk_with_movement` | 3.056 |

### DT_static (sorted by deg ascending)

| stem | deg |
|---|---:|
| `p0mhFbhV6kGJgjd0RTTIIw_doubletalk` | 1.224 |
| `XuiheB7eUkyJA2XzFIovHQ_doubletalk` | 1.233 |
| `X0OraQOKtkCVriR0uj0WBQ_doubletalk` | 1.235 |
| `QK70KpLuZ0O43BBSWEZvHg_doubletalk` | 1.239 |
| `MYrVxVEMxkaE7OuyTUmI0Q_doubletalk` | 1.249 |
| `XtUWcfsuykC5fTu7DdAnnw_doubletalk` | 1.269 |
| `SgKY30fjT0G8e3kQL0RHSQ_doubletalk` | 1.278 |
| `pPBidi8oyUarZCGUrKJEsg_doubletalk` | 1.299 |
| `yc5bFUGsR0GSfiGwTTpRWg_doubletalk` | 1.327 |
| `sUQrHEPAoEmIvHclpi1tRQ_doubletalk` | 1.327 |
| `xofDX004bkqiOv9YOxmGVQ_doubletalk` | 1.334 |
| `XV5L2dn3S06M9GBEu1q3DA_doubletalk` | 1.339 |
| `QkRkwwFKVEar0WtcuvJsZg_doubletalk` | 1.370 |
| `XCXcCwUPY0GmrtqtJ6xY2g_doubletalk` | 1.403 |
| `nlSSRl4k50Gq2mIRYlMBCg_doubletalk` | 1.411 |
| `V6Mw0Ti8RUSkzvMGB4WGiw_doubletalk` | 1.411 |
| `sYQK1rJlwU2XCy20n0Sx9g_doubletalk` | 1.417 |
| `XGDaZuEkE0WU4IN0Yi4XtA_doubletalk` | 1.439 |
| `NY3kZioAm0KwR45wIVe2Sg_doubletalk` | 1.450 |
| `0I0XMl3M0ECO0U1N0cJvpg_doubletalk` | 1.476 |

### DT_movement (sorted by deg ascending)

| stem | deg |
|---|---:|
| `XV5L2dn3S06M9GBEu1q3DA_doubletalk_with_movement` | 1.157 |
| `wHmBm7VHfkysBOhjoAXkNA_doubletalk_with_movement` | 1.254 |
| `xofDX004bkqiOv9YOxmGVQ_doubletalk_with_movement` | 1.297 |
| `WqEYNwalSUebZxaeYVay2g_doubletalk_with_movement` | 1.315 |
| `XuiheB7eUkyJA2XzFIovHQ_doubletalk_with_movement` | 1.350 |
| `VgSXlJJEI02dytkMm5UTzA_doubletalk_with_movement` | 1.378 |
| `QkRkwwFKVEar0WtcuvJsZg_doubletalk_with_movement` | 1.386 |
| `I2bme08keUmAnyJRKNYDGQ_doubletalk_with_movement` | 1.386 |
| `ql7yTcebJU20VE5qpW0kCA_doubletalk_with_movement` | 1.396 |
| `Xvyiz1o0cEijZQ8DT9mB2w_doubletalk_with_movement` | 1.401 |
| `nyT6FUUdu0W8UpvjP1rRgQ_doubletalk_with_movement` | 1.408 |
| `X7Ua9txMj0aws848JPEbOg_doubletalk_with_movement` | 1.409 |
| `WcK0OrF6ukW03fViPXTQjQ_doubletalk_with_movement` | 1.411 |
| `V0JqgjlrB0Ke9y91r0rxNw_doubletalk_with_movement` | 1.460 |
| `w0QrMwsZ5kGoJjRWvP0iKg_doubletalk_with_movement` | 1.471 |
| `nlSSRl4k50Gq2mIRYlMBCg_doubletalk_with_movement` | 1.520 |
| `SwfEwuGDlkWYy9pb4H00eQ_doubletalk_with_movement` | 1.534 |
| `WH7rA6R2zkyopKUrcq9p3A_doubletalk_with_movement` | 1.536 |
| `sKXucFp4FUCJKo5d0G54Og_doubletalk_with_movement` | 1.553 |
| `w5XDRNfB2Ei2UoUDtrTkzg_doubletalk_with_movement` | 1.555 |

### NE (sorted by deg ascending)

| stem | deg |
|---|---:|
| `SR68lGQwTUy508j0P8BKZQ_nearend_singletalk` | 2.903 |
| `kz23X4pDSEiPmWtw2Qx00Q_nearend_singletalk` | 2.963 |
| `SFvlSygv4ke9wCrv8LWvYQ_nearend_singletalk` | 3.075 |
| `kOtW70qgikKm0F9OEQw22A_nearend_singletalk` | 3.139 |
| `A50fBOz02kag8CVAhBOF8A_nearend_singletalk` | 3.191 |
| `DW3NNHoKl02cAfEkahzsGg_nearend_singletalk` | 3.261 |
| `pPBidi8oyUarZCGUrKJEsg_nearend_singletalk` | 3.296 |
| `2f9FDnH8u0q829njqEuVrQ_nearend_singletalk` | 3.318 |
| `4pN9yn7mhEa5iDiKnr5jlw_nearend_singletalk` | 3.324 |
| `IvAFDPUFEk0GuW2VKRHznA_nearend_singletalk` | 3.362 |
| `08Mo9p6KVUapuF8PNAMWaw_nearend_singletalk` | 3.364 |
| `E0l0WVPQjEi6AmtbvfSYLA_nearend_singletalk` | 3.373 |
| `mAJHXrZ5QU62Do1OqTSL6Q_nearend_singletalk` | 3.379 |
| `LAdLsNzbcE66K0Zs1NE5QA_nearend_singletalk` | 3.434 |
| `x1pZr2uWBkulb77zcCfAJA_nearend_singletalk` | 3.434 |
| `x7Frsl5wTEKmog3A6lL57g_nearend_singletalk` | 3.454 |
| `mljUO9k4gUiYCHgXSsfo5A_nearend_singletalk` | 3.479 |
| `VNkNShj97UajHDVbSmIG0g_nearend_singletalk` | 3.479 |
| `XCXcCwUPY0GmrtqtJ6xY2g_nearend_singletalk` | 3.483 |
| `pcb1Nh0Z3k0WS9a7gBEuqg_nearend_singletalk` | 3.484 |
