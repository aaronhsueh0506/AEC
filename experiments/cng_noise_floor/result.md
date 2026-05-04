# Bench result — cng_run2

Dataset: /Users/mingyu/Desktop/novatek/SE/AEC/wav/aec_challenge_blind
Output dir: experiments/cng_noise_floor/output
Cases scored: 800

## Bucket means

| Bucket | n | echo (↑) | deg (↑) |
|---|---:|---:|---:|
| FS_static | 169 | 3.767 | 4.999 |
| FS_movement | 131 | 3.834 | 4.999 |
| DT_static | 186 | 4.252 | 2.257 |
| DT_movement | 114 | 4.124 | 2.284 |
| NE | 200 | 4.998 | 4.006 |

## Δ vs baseline (baseline_v381)

| Bucket | Δecho | Δdeg | verdict |
|---|---:|---:|---|
| FS_static | +0.000 | +0.000 | ok |
| FS_movement | +0.002 | +0.000 | ok |
| DT_static | -0.000 | -0.000 | ok |
| DT_movement | +0.002 | -0.004 | ok |
| NE | -0.000 | -0.000 | ok |

## Worst-20 per bucket

### FS_static (sorted by echo ascending)

| stem | echo |
|---|---:|
| `7GTxyTksSUqCnP5y0ILG4A_farend_singletalk` | 1.318 |
| `9xjhiFbGo06hdQIsHTS6qA_farend_singletalk` | 1.791 |
| `IrQvqOTCmEWMXn9k2ICtRQ_farend_singletalk` | 1.828 |
| `hVqUmGvIlkO0LBUoE06Q3w_farend_singletalk` | 1.886 |
| `S22FCqKDWUyymN1YbpItIw_farend_singletalk` | 1.924 |
| `pcb1Nh0Z3k0WS9a7gBEuqg_farend_singletalk` | 1.934 |
| `VJfVUwJs4k25ziMNvJb43A_farend_singletalk` | 1.993 |
| `JteZUZ4JYkeD4k2rcVbqHg_farend_singletalk` | 2.151 |
| `N2rQLbnp2UOg2QFRaggbDw_farend_singletalk` | 2.201 |
| `geGGo8g9UE2MYkdjxtHm8w_farend_singletalk` | 2.297 |
| `JtodX3Ug6Eu5TYu0HN5IOw_farend_singletalk` | 2.305 |
| `pmzLFdKTzEixfU0l0furvA_farend_singletalk` | 2.331 |
| `LeV1uF4j10Whm0FPG80tmw_farend_singletalk` | 2.469 |
| `s90M7MOTBkqaV4nQPLhKbA_farend_singletalk` | 2.474 |
| `iOyPaxX11UOaUkcscKhq1A_farend_singletalk` | 2.485 |
| `60JogIkuEECYHrwVT3UYwg_farend_singletalk` | 2.555 |
| `m4789fdio0q92zjf9gvh1Q_farend_singletalk` | 2.564 |
| `lV0kQN0hR0ySmE0bQhuYbw_farend_singletalk` | 2.625 |
| `zddPqpp1a06xttKdc0iNTA_farend_singletalk` | 2.645 |
| `GDyfzBkhxEiDbnRZGGOrQQ_farend_singletalk` | 2.678 |

### FS_movement (sorted by echo ascending)

| stem | echo |
|---|---:|
| `S22FCqKDWUyymN1YbpItIw_farend_singletalk_with_movement` | 1.917 |
| `iOyPaxX11UOaUkcscKhq1A_farend_singletalk_with_movement` | 2.010 |
| `hVqUmGvIlkO0LBUoE06Q3w_farend_singletalk_with_movement` | 2.108 |
| `SgKY30fjT0G8e3kQL0RHSQ_farend_singletalk_with_movement` | 2.235 |
| `OXCtw0FVhUWGUBil6Uucdw_farend_singletalk_with_movement` | 2.295 |
| `IrQvqOTCmEWMXn9k2ICtRQ_farend_singletalk_with_movement` | 2.342 |
| `JtodX3Ug6Eu5TYu0HN5IOw_farend_singletalk_with_movement` | 2.465 |
| `JjCzlhn3gEiBQvfJtPNJ9A_farend_singletalk_with_movement` | 2.525 |
| `N2rQLbnp2UOg2QFRaggbDw_farend_singletalk_with_movement` | 2.555 |
| `yyvS0Ljh1k0AHMx6cxtNyg_farend_singletalk_with_movement` | 2.578 |
| `0luXwWjGEEC9G5nf0yTVXw_farend_singletalk_with_movement` | 2.591 |
| `XTqo1aOXDEiqyWTFK99I5Q_farend_singletalk_with_movement` | 2.673 |
| `TGZ5Wq0SCUCOXPsfee3uMQ_farend_singletalk_with_movement` | 2.745 |
| `XuguA1uJAE0bWT0xXRDdeA_farend_singletalk_with_movement` | 2.805 |
| `VNkNShj97UajHDVbSmIG0g_farend_singletalk_with_movement` | 2.863 |
| `xFk7igecuke0R5JMfREyDg_farend_singletalk_with_movement` | 2.916 |
| `IqtJR4tjJkWrwUjYorz0Og_farend_singletalk_with_movement` | 2.994 |
| `LHsrJBRGnUKiMC2mihEr0g_farend_singletalk_with_movement` | 3.018 |
| `pmzLFdKTzEixfU0l0furvA_farend_singletalk_with_movement` | 3.055 |
| `sx6mxKBQpkq520m64BwUdQ_farend_singletalk_with_movement` | 3.076 |

### DT_static (sorted by deg ascending)

| stem | deg |
|---|---:|
| `p0mhFbhV6kGJgjd0RTTIIw_doubletalk` | 1.220 |
| `X0OraQOKtkCVriR0uj0WBQ_doubletalk` | 1.237 |
| `QK70KpLuZ0O43BBSWEZvHg_doubletalk` | 1.240 |
| `XuiheB7eUkyJA2XzFIovHQ_doubletalk` | 1.242 |
| `MYrVxVEMxkaE7OuyTUmI0Q_doubletalk` | 1.247 |
| `XtUWcfsuykC5fTu7DdAnnw_doubletalk` | 1.271 |
| `SgKY30fjT0G8e3kQL0RHSQ_doubletalk` | 1.280 |
| `pPBidi8oyUarZCGUrKJEsg_doubletalk` | 1.303 |
| `sUQrHEPAoEmIvHclpi1tRQ_doubletalk` | 1.323 |
| `yc5bFUGsR0GSfiGwTTpRWg_doubletalk` | 1.325 |
| `xofDX004bkqiOv9YOxmGVQ_doubletalk` | 1.333 |
| `XV5L2dn3S06M9GBEu1q3DA_doubletalk` | 1.350 |
| `nlSSRl4k50Gq2mIRYlMBCg_doubletalk` | 1.380 |
| `sYQK1rJlwU2XCy20n0Sx9g_doubletalk` | 1.381 |
| `V6Mw0Ti8RUSkzvMGB4WGiw_doubletalk` | 1.405 |
| `XCXcCwUPY0GmrtqtJ6xY2g_doubletalk` | 1.411 |
| `QkRkwwFKVEar0WtcuvJsZg_doubletalk` | 1.424 |
| `XGDaZuEkE0WU4IN0Yi4XtA_doubletalk` | 1.443 |
| `NY3kZioAm0KwR45wIVe2Sg_doubletalk` | 1.445 |
| `0I0XMl3M0ECO0U1N0cJvpg_doubletalk` | 1.450 |

### DT_movement (sorted by deg ascending)

| stem | deg |
|---|---:|
| `XV5L2dn3S06M9GBEu1q3DA_doubletalk_with_movement` | 1.153 |
| `wHmBm7VHfkysBOhjoAXkNA_doubletalk_with_movement` | 1.258 |
| `xofDX004bkqiOv9YOxmGVQ_doubletalk_with_movement` | 1.298 |
| `WqEYNwalSUebZxaeYVay2g_doubletalk_with_movement` | 1.315 |
| `XuiheB7eUkyJA2XzFIovHQ_doubletalk_with_movement` | 1.338 |
| `VgSXlJJEI02dytkMm5UTzA_doubletalk_with_movement` | 1.386 |
| `Xvyiz1o0cEijZQ8DT9mB2w_doubletalk_with_movement` | 1.386 |
| `QkRkwwFKVEar0WtcuvJsZg_doubletalk_with_movement` | 1.395 |
| `ql7yTcebJU20VE5qpW0kCA_doubletalk_with_movement` | 1.396 |
| `X7Ua9txMj0aws848JPEbOg_doubletalk_with_movement` | 1.398 |
| `nyT6FUUdu0W8UpvjP1rRgQ_doubletalk_with_movement` | 1.416 |
| `I2bme08keUmAnyJRKNYDGQ_doubletalk_with_movement` | 1.426 |
| `WcK0OrF6ukW03fViPXTQjQ_doubletalk_with_movement` | 1.432 |
| `V0JqgjlrB0Ke9y91r0rxNw_doubletalk_with_movement` | 1.449 |
| `sKXucFp4FUCJKo5d0G54Og_doubletalk_with_movement` | 1.496 |
| `w0QrMwsZ5kGoJjRWvP0iKg_doubletalk_with_movement` | 1.509 |
| `zOiK6oSHp0ib3nHvzLKbRQ_doubletalk_with_movement` | 1.543 |
| `SwfEwuGDlkWYy9pb4H00eQ_doubletalk_with_movement` | 1.558 |
| `w5XDRNfB2Ei2UoUDtrTkzg_doubletalk_with_movement` | 1.561 |
| `WH7rA6R2zkyopKUrcq9p3A_doubletalk_with_movement` | 1.564 |

### NE (sorted by deg ascending)

| stem | deg |
|---|---:|
| `SR68lGQwTUy508j0P8BKZQ_nearend_singletalk` | 2.903 |
| `kz23X4pDSEiPmWtw2Qx00Q_nearend_singletalk` | 2.962 |
| `SFvlSygv4ke9wCrv8LWvYQ_nearend_singletalk` | 3.074 |
| `kOtW70qgikKm0F9OEQw22A_nearend_singletalk` | 3.139 |
| `A50fBOz02kag8CVAhBOF8A_nearend_singletalk` | 3.192 |
| `DW3NNHoKl02cAfEkahzsGg_nearend_singletalk` | 3.276 |
| `pPBidi8oyUarZCGUrKJEsg_nearend_singletalk` | 3.296 |
| `2f9FDnH8u0q829njqEuVrQ_nearend_singletalk` | 3.318 |
| `4pN9yn7mhEa5iDiKnr5jlw_nearend_singletalk` | 3.322 |
| `08Mo9p6KVUapuF8PNAMWaw_nearend_singletalk` | 3.361 |
| `IvAFDPUFEk0GuW2VKRHznA_nearend_singletalk` | 3.361 |
| `E0l0WVPQjEi6AmtbvfSYLA_nearend_singletalk` | 3.374 |
| `mAJHXrZ5QU62Do1OqTSL6Q_nearend_singletalk` | 3.380 |
| `x1pZr2uWBkulb77zcCfAJA_nearend_singletalk` | 3.432 |
| `LAdLsNzbcE66K0Zs1NE5QA_nearend_singletalk` | 3.433 |
| `x7Frsl5wTEKmog3A6lL57g_nearend_singletalk` | 3.435 |
| `mljUO9k4gUiYCHgXSsfo5A_nearend_singletalk` | 3.479 |
| `XCXcCwUPY0GmrtqtJ6xY2g_nearend_singletalk` | 3.483 |
| `pcb1Nh0Z3k0WS9a7gBEuqg_nearend_singletalk` | 3.484 |
| `VNkNShj97UajHDVbSmIG0g_nearend_singletalk` | 3.485 |
