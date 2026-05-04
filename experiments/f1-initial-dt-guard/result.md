# Bench result — f1_initial_dt_guard

Dataset: /Users/mingyu/Desktop/novatek/SE/AEC/wav/aec_challenge_blind
Output dir: experiments/f1-initial-dt-guard/output
Cases scored: 800

## Bucket means

| Bucket | n | echo (↑) | deg (↑) |
|---|---:|---:|---:|
| FS_static | 169 | 3.769 | 4.999 |
| FS_movement | 131 | 3.830 | 4.999 |
| DT_static | 186 | 4.252 | 2.258 |
| DT_movement | 114 | 4.124 | 2.284 |
| NE | 200 | 4.998 | 4.007 |

## Δ vs baseline (baseline_v381)

| Bucket | Δecho | Δdeg | verdict |
|---|---:|---:|---|
| FS_static | +0.002 | -0.000 | ok |
| FS_movement | -0.001 | -0.000 | ok |
| DT_static | +0.000 | +0.001 | ok |
| DT_movement | +0.003 | -0.004 | ok |
| NE | +0.000 | +0.000 | ok |

## Worst-20 per bucket

### FS_static (sorted by echo ascending)

| stem | echo |
|---|---:|
| `7GTxyTksSUqCnP5y0ILG4A_farend_singletalk` | 1.323 |
| `IrQvqOTCmEWMXn9k2ICtRQ_farend_singletalk` | 1.826 |
| `9xjhiFbGo06hdQIsHTS6qA_farend_singletalk` | 1.827 |
| `hVqUmGvIlkO0LBUoE06Q3w_farend_singletalk` | 1.875 |
| `S22FCqKDWUyymN1YbpItIw_farend_singletalk` | 1.919 |
| `pcb1Nh0Z3k0WS9a7gBEuqg_farend_singletalk` | 1.921 |
| `VJfVUwJs4k25ziMNvJb43A_farend_singletalk` | 2.020 |
| `JteZUZ4JYkeD4k2rcVbqHg_farend_singletalk` | 2.140 |
| `N2rQLbnp2UOg2QFRaggbDw_farend_singletalk` | 2.204 |
| `geGGo8g9UE2MYkdjxtHm8w_farend_singletalk` | 2.300 |
| `JtodX3Ug6Eu5TYu0HN5IOw_farend_singletalk` | 2.302 |
| `pmzLFdKTzEixfU0l0furvA_farend_singletalk` | 2.335 |
| `LeV1uF4j10Whm0FPG80tmw_farend_singletalk` | 2.455 |
| `s90M7MOTBkqaV4nQPLhKbA_farend_singletalk` | 2.467 |
| `iOyPaxX11UOaUkcscKhq1A_farend_singletalk` | 2.528 |
| `m4789fdio0q92zjf9gvh1Q_farend_singletalk` | 2.552 |
| `lV0kQN0hR0ySmE0bQhuYbw_farend_singletalk` | 2.605 |
| `60JogIkuEECYHrwVT3UYwg_farend_singletalk` | 2.653 |
| `zddPqpp1a06xttKdc0iNTA_farend_singletalk` | 2.660 |
| `o97uExi7MEqzbanuWK6CCw_farend_singletalk` | 2.667 |

### FS_movement (sorted by echo ascending)

| stem | echo |
|---|---:|
| `S22FCqKDWUyymN1YbpItIw_farend_singletalk_with_movement` | 1.922 |
| `iOyPaxX11UOaUkcscKhq1A_farend_singletalk_with_movement` | 2.012 |
| `hVqUmGvIlkO0LBUoE06Q3w_farend_singletalk_with_movement` | 2.114 |
| `SgKY30fjT0G8e3kQL0RHSQ_farend_singletalk_with_movement` | 2.233 |
| `OXCtw0FVhUWGUBil6Uucdw_farend_singletalk_with_movement` | 2.298 |
| `IrQvqOTCmEWMXn9k2ICtRQ_farend_singletalk_with_movement` | 2.375 |
| `JtodX3Ug6Eu5TYu0HN5IOw_farend_singletalk_with_movement` | 2.467 |
| `0luXwWjGEEC9G5nf0yTVXw_farend_singletalk_with_movement` | 2.495 |
| `JjCzlhn3gEiBQvfJtPNJ9A_farend_singletalk_with_movement` | 2.502 |
| `N2rQLbnp2UOg2QFRaggbDw_farend_singletalk_with_movement` | 2.508 |
| `yyvS0Ljh1k0AHMx6cxtNyg_farend_singletalk_with_movement` | 2.578 |
| `XTqo1aOXDEiqyWTFK99I5Q_farend_singletalk_with_movement` | 2.666 |
| `XuguA1uJAE0bWT0xXRDdeA_farend_singletalk_with_movement` | 2.749 |
| `TGZ5Wq0SCUCOXPsfee3uMQ_farend_singletalk_with_movement` | 2.779 |
| `VNkNShj97UajHDVbSmIG0g_farend_singletalk_with_movement` | 2.823 |
| `xFk7igecuke0R5JMfREyDg_farend_singletalk_with_movement` | 2.928 |
| `pmzLFdKTzEixfU0l0furvA_farend_singletalk_with_movement` | 2.929 |
| `LHsrJBRGnUKiMC2mihEr0g_farend_singletalk_with_movement` | 2.985 |
| `IqtJR4tjJkWrwUjYorz0Og_farend_singletalk_with_movement` | 3.020 |
| `sx6mxKBQpkq520m64BwUdQ_farend_singletalk_with_movement` | 3.070 |

### DT_static (sorted by deg ascending)

| stem | deg |
|---|---:|
| `p0mhFbhV6kGJgjd0RTTIIw_doubletalk` | 1.215 |
| `X0OraQOKtkCVriR0uj0WBQ_doubletalk` | 1.235 |
| `XuiheB7eUkyJA2XzFIovHQ_doubletalk` | 1.236 |
| `QK70KpLuZ0O43BBSWEZvHg_doubletalk` | 1.239 |
| `MYrVxVEMxkaE7OuyTUmI0Q_doubletalk` | 1.243 |
| `XtUWcfsuykC5fTu7DdAnnw_doubletalk` | 1.270 |
| `SgKY30fjT0G8e3kQL0RHSQ_doubletalk` | 1.271 |
| `pPBidi8oyUarZCGUrKJEsg_doubletalk` | 1.302 |
| `sUQrHEPAoEmIvHclpi1tRQ_doubletalk` | 1.317 |
| `yc5bFUGsR0GSfiGwTTpRWg_doubletalk` | 1.333 |
| `xofDX004bkqiOv9YOxmGVQ_doubletalk` | 1.346 |
| `XV5L2dn3S06M9GBEu1q3DA_doubletalk` | 1.354 |
| `QkRkwwFKVEar0WtcuvJsZg_doubletalk` | 1.393 |
| `sYQK1rJlwU2XCy20n0Sx9g_doubletalk` | 1.396 |
| `NY3kZioAm0KwR45wIVe2Sg_doubletalk` | 1.412 |
| `V6Mw0Ti8RUSkzvMGB4WGiw_doubletalk` | 1.415 |
| `nlSSRl4k50Gq2mIRYlMBCg_doubletalk` | 1.428 |
| `XCXcCwUPY0GmrtqtJ6xY2g_doubletalk` | 1.433 |
| `XGDaZuEkE0WU4IN0Yi4XtA_doubletalk` | 1.438 |
| `0I0XMl3M0ECO0U1N0cJvpg_doubletalk` | 1.489 |

### DT_movement (sorted by deg ascending)

| stem | deg |
|---|---:|
| `XV5L2dn3S06M9GBEu1q3DA_doubletalk_with_movement` | 1.155 |
| `wHmBm7VHfkysBOhjoAXkNA_doubletalk_with_movement` | 1.257 |
| `WqEYNwalSUebZxaeYVay2g_doubletalk_with_movement` | 1.292 |
| `xofDX004bkqiOv9YOxmGVQ_doubletalk_with_movement` | 1.296 |
| `XuiheB7eUkyJA2XzFIovHQ_doubletalk_with_movement` | 1.350 |
| `I2bme08keUmAnyJRKNYDGQ_doubletalk_with_movement` | 1.375 |
| `Xvyiz1o0cEijZQ8DT9mB2w_doubletalk_with_movement` | 1.377 |
| `VgSXlJJEI02dytkMm5UTzA_doubletalk_with_movement` | 1.382 |
| `X7Ua9txMj0aws848JPEbOg_doubletalk_with_movement` | 1.402 |
| `ql7yTcebJU20VE5qpW0kCA_doubletalk_with_movement` | 1.409 |
| `WcK0OrF6ukW03fViPXTQjQ_doubletalk_with_movement` | 1.419 |
| `QkRkwwFKVEar0WtcuvJsZg_doubletalk_with_movement` | 1.428 |
| `w0QrMwsZ5kGoJjRWvP0iKg_doubletalk_with_movement` | 1.459 |
| `V0JqgjlrB0Ke9y91r0rxNw_doubletalk_with_movement` | 1.460 |
| `nlSSRl4k50Gq2mIRYlMBCg_doubletalk_with_movement` | 1.478 |
| `nyT6FUUdu0W8UpvjP1rRgQ_doubletalk_with_movement` | 1.480 |
| `SwfEwuGDlkWYy9pb4H00eQ_doubletalk_with_movement` | 1.505 |
| `sKXucFp4FUCJKo5d0G54Og_doubletalk_with_movement` | 1.518 |
| `WH7rA6R2zkyopKUrcq9p3A_doubletalk_with_movement` | 1.527 |
| `kwolfjBXWEOJmdbDdFoTVQ_doubletalk_with_movement` | 1.529 |

### NE (sorted by deg ascending)

| stem | deg |
|---|---:|
| `SR68lGQwTUy508j0P8BKZQ_nearend_singletalk` | 2.903 |
| `kz23X4pDSEiPmWtw2Qx00Q_nearend_singletalk` | 2.963 |
| `SFvlSygv4ke9wCrv8LWvYQ_nearend_singletalk` | 3.081 |
| `kOtW70qgikKm0F9OEQw22A_nearend_singletalk` | 3.139 |
| `A50fBOz02kag8CVAhBOF8A_nearend_singletalk` | 3.191 |
| `DW3NNHoKl02cAfEkahzsGg_nearend_singletalk` | 3.269 |
| `pPBidi8oyUarZCGUrKJEsg_nearend_singletalk` | 3.295 |
| `2f9FDnH8u0q829njqEuVrQ_nearend_singletalk` | 3.318 |
| `4pN9yn7mhEa5iDiKnr5jlw_nearend_singletalk` | 3.323 |
| `IvAFDPUFEk0GuW2VKRHznA_nearend_singletalk` | 3.361 |
| `08Mo9p6KVUapuF8PNAMWaw_nearend_singletalk` | 3.363 |
| `E0l0WVPQjEi6AmtbvfSYLA_nearend_singletalk` | 3.380 |
| `mAJHXrZ5QU62Do1OqTSL6Q_nearend_singletalk` | 3.381 |
| `LAdLsNzbcE66K0Zs1NE5QA_nearend_singletalk` | 3.432 |
| `x1pZr2uWBkulb77zcCfAJA_nearend_singletalk` | 3.433 |
| `x7Frsl5wTEKmog3A6lL57g_nearend_singletalk` | 3.436 |
| `mljUO9k4gUiYCHgXSsfo5A_nearend_singletalk` | 3.479 |
| `VNkNShj97UajHDVbSmIG0g_nearend_singletalk` | 3.480 |
| `XCXcCwUPY0GmrtqtJ6xY2g_nearend_singletalk` | 3.483 |
| `pcb1Nh0Z3k0WS9a7gBEuqg_nearend_singletalk` | 3.484 |
