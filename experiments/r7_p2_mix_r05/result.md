# Bench result — current

Dataset: /Users/mingyu/Desktop/novatek/SE/AEC/wav/aec_challenge_blind
Output dir: experiments/r7_p2_mix_r05/eval_output
Cases scored: 800

## Bucket means

| Bucket | n | echo (↑) | deg (↑) |
|---|---:|---:|---:|
| FS_static | 169 | 3.703 | 4.999 |
| FS_movement | 131 | 3.815 | 4.999 |
| DT_static | 186 | 4.114 | 2.333 |
| DT_movement | 114 | 4.012 | 2.349 |
| NE | 200 | 4.998 | 3.997 |

## Δ vs baseline (baseline_v381_seeded)

| Bucket | Δecho | Δdeg | verdict |
|---|---:|---:|---|
| FS_static | -0.065 | +0.000 | FS echo regress |
| FS_movement | -0.012 | +0.000 | ok |
| DT_static | -0.137 | +0.076 | ok |
| DT_movement | -0.108 | +0.063 | ok |
| NE | +0.000 | -0.010 | ok |

## Worst-20 per bucket

### FS_static (sorted by echo ascending)

| stem | echo |
|---|---:|
| `7GTxyTksSUqCnP5y0ILG4A_farend_singletalk` | 1.304 |
| `9xjhiFbGo06hdQIsHTS6qA_farend_singletalk` | 1.816 |
| `IrQvqOTCmEWMXn9k2ICtRQ_farend_singletalk` | 1.828 |
| `hVqUmGvIlkO0LBUoE06Q3w_farend_singletalk` | 1.847 |
| `S22FCqKDWUyymN1YbpItIw_farend_singletalk` | 1.909 |
| `pcb1Nh0Z3k0WS9a7gBEuqg_farend_singletalk` | 1.921 |
| `VJfVUwJs4k25ziMNvJb43A_farend_singletalk` | 1.922 |
| `JtodX3Ug6Eu5TYu0HN5IOw_farend_singletalk` | 1.956 |
| `JteZUZ4JYkeD4k2rcVbqHg_farend_singletalk` | 2.160 |
| `pmzLFdKTzEixfU0l0furvA_farend_singletalk` | 2.167 |
| `geGGo8g9UE2MYkdjxtHm8w_farend_singletalk` | 2.193 |
| `N2rQLbnp2UOg2QFRaggbDw_farend_singletalk` | 2.376 |
| `XTqo1aOXDEiqyWTFK99I5Q_farend_singletalk` | 2.383 |
| `LeV1uF4j10Whm0FPG80tmw_farend_singletalk` | 2.424 |
| `s90M7MOTBkqaV4nQPLhKbA_farend_singletalk` | 2.457 |
| `wTgNyKxp0kihhU2K96tHgA_farend_singletalk` | 2.499 |
| `60JogIkuEECYHrwVT3UYwg_farend_singletalk` | 2.511 |
| `m4789fdio0q92zjf9gvh1Q_farend_singletalk` | 2.515 |
| `Gsy0lC5QSUi540hiax9XtA_farend_singletalk` | 2.567 |
| `yM2wHof9U06yVPJfemZ3hg_farend_singletalk` | 2.572 |

### FS_movement (sorted by echo ascending)

| stem | echo |
|---|---:|
| `S22FCqKDWUyymN1YbpItIw_farend_singletalk_with_movement` | 1.951 |
| `iOyPaxX11UOaUkcscKhq1A_farend_singletalk_with_movement` | 2.015 |
| `hVqUmGvIlkO0LBUoE06Q3w_farend_singletalk_with_movement` | 2.042 |
| `OXCtw0FVhUWGUBil6Uucdw_farend_singletalk_with_movement` | 2.270 |
| `IrQvqOTCmEWMXn9k2ICtRQ_farend_singletalk_with_movement` | 2.322 |
| `0luXwWjGEEC9G5nf0yTVXw_farend_singletalk_with_movement` | 2.339 |
| `SgKY30fjT0G8e3kQL0RHSQ_farend_singletalk_with_movement` | 2.447 |
| `N2rQLbnp2UOg2QFRaggbDw_farend_singletalk_with_movement` | 2.450 |
| `JjCzlhn3gEiBQvfJtPNJ9A_farend_singletalk_with_movement` | 2.499 |
| `TGZ5Wq0SCUCOXPsfee3uMQ_farend_singletalk_with_movement` | 2.523 |
| `VNgRsWxMdkaUx1gKV9W1Zw_farend_singletalk_with_movement` | 2.617 |
| `XuguA1uJAE0bWT0xXRDdeA_farend_singletalk_with_movement` | 2.664 |
| `JtodX3Ug6Eu5TYu0HN5IOw_farend_singletalk_with_movement` | 2.668 |
| `IqtJR4tjJkWrwUjYorz0Og_farend_singletalk_with_movement` | 2.687 |
| `yyvS0Ljh1k0AHMx6cxtNyg_farend_singletalk_with_movement` | 2.687 |
| `VNkNShj97UajHDVbSmIG0g_farend_singletalk_with_movement` | 2.739 |
| `sx6mxKBQpkq520m64BwUdQ_farend_singletalk_with_movement` | 2.864 |
| `ML4MF3Mea0yurjceNQPfNA_farend_singletalk_with_movement` | 2.901 |
| `pmzLFdKTzEixfU0l0furvA_farend_singletalk_with_movement` | 2.923 |
| `xFk7igecuke0R5JMfREyDg_farend_singletalk_with_movement` | 2.928 |

### DT_static (sorted by deg ascending)

| stem | deg |
|---|---:|
| `QK70KpLuZ0O43BBSWEZvHg_doubletalk` | 1.229 |
| `X0OraQOKtkCVriR0uj0WBQ_doubletalk` | 1.234 |
| `p0mhFbhV6kGJgjd0RTTIIw_doubletalk` | 1.235 |
| `XuiheB7eUkyJA2XzFIovHQ_doubletalk` | 1.253 |
| `SgKY30fjT0G8e3kQL0RHSQ_doubletalk` | 1.259 |
| `XtUWcfsuykC5fTu7DdAnnw_doubletalk` | 1.290 |
| `MYrVxVEMxkaE7OuyTUmI0Q_doubletalk` | 1.293 |
| `yc5bFUGsR0GSfiGwTTpRWg_doubletalk` | 1.349 |
| `QkRkwwFKVEar0WtcuvJsZg_doubletalk` | 1.353 |
| `pPBidi8oyUarZCGUrKJEsg_doubletalk` | 1.363 |
| `V6Mw0Ti8RUSkzvMGB4WGiw_doubletalk` | 1.373 |
| `xofDX004bkqiOv9YOxmGVQ_doubletalk` | 1.384 |
| `NY3kZioAm0KwR45wIVe2Sg_doubletalk` | 1.419 |
| `XV5L2dn3S06M9GBEu1q3DA_doubletalk` | 1.460 |
| `sYQK1rJlwU2XCy20n0Sx9g_doubletalk` | 1.465 |
| `XGDaZuEkE0WU4IN0Yi4XtA_doubletalk` | 1.491 |
| `qVd1gtwQ0k2lVRqPVp1NKQ_doubletalk` | 1.502 |
| `WH7rA6R2zkyopKUrcq9p3A_doubletalk` | 1.503 |
| `sUQrHEPAoEmIvHclpi1tRQ_doubletalk` | 1.511 |
| `0I0XMl3M0ECO0U1N0cJvpg_doubletalk` | 1.511 |

### DT_movement (sorted by deg ascending)

| stem | deg |
|---|---:|
| `XV5L2dn3S06M9GBEu1q3DA_doubletalk_with_movement` | 1.165 |
| `wHmBm7VHfkysBOhjoAXkNA_doubletalk_with_movement` | 1.224 |
| `WqEYNwalSUebZxaeYVay2g_doubletalk_with_movement` | 1.286 |
| `XuiheB7eUkyJA2XzFIovHQ_doubletalk_with_movement` | 1.323 |
| `xofDX004bkqiOv9YOxmGVQ_doubletalk_with_movement` | 1.334 |
| `X7Ua9txMj0aws848JPEbOg_doubletalk_with_movement` | 1.380 |
| `VgSXlJJEI02dytkMm5UTzA_doubletalk_with_movement` | 1.387 |
| `I2bme08keUmAnyJRKNYDGQ_doubletalk_with_movement` | 1.399 |
| `ql7yTcebJU20VE5qpW0kCA_doubletalk_with_movement` | 1.417 |
| `Xvyiz1o0cEijZQ8DT9mB2w_doubletalk_with_movement` | 1.428 |
| `QkRkwwFKVEar0WtcuvJsZg_doubletalk_with_movement` | 1.440 |
| `V0JqgjlrB0Ke9y91r0rxNw_doubletalk_with_movement` | 1.490 |
| `zOiK6oSHp0ib3nHvzLKbRQ_doubletalk_with_movement` | 1.500 |
| `SwfEwuGDlkWYy9pb4H00eQ_doubletalk_with_movement` | 1.500 |
| `wWeNtFK0dEG9Wub40bB15A_doubletalk_with_movement` | 1.530 |
| `w5XDRNfB2Ei2UoUDtrTkzg_doubletalk_with_movement` | 1.557 |
| `WcK0OrF6ukW03fViPXTQjQ_doubletalk_with_movement` | 1.558 |
| `w0QrMwsZ5kGoJjRWvP0iKg_doubletalk_with_movement` | 1.573 |
| `WH7rA6R2zkyopKUrcq9p3A_doubletalk_with_movement` | 1.597 |
| `OmB0Ht0hmE2crVnftAEtsw_doubletalk_with_movement` | 1.637 |

### NE (sorted by deg ascending)

| stem | deg |
|---|---:|
| `SR68lGQwTUy508j0P8BKZQ_nearend_singletalk` | 3.000 |
| `kz23X4pDSEiPmWtw2Qx00Q_nearend_singletalk` | 3.003 |
| `kOtW70qgikKm0F9OEQw22A_nearend_singletalk` | 3.078 |
| `x7Frsl5wTEKmog3A6lL57g_nearend_singletalk` | 3.162 |
| `DW3NNHoKl02cAfEkahzsGg_nearend_singletalk` | 3.182 |
| `A50fBOz02kag8CVAhBOF8A_nearend_singletalk` | 3.188 |
| `08Mo9p6KVUapuF8PNAMWaw_nearend_singletalk` | 3.232 |
| `4pN9yn7mhEa5iDiKnr5jlw_nearend_singletalk` | 3.299 |
| `E0l0WVPQjEi6AmtbvfSYLA_nearend_singletalk` | 3.321 |
| `SFvlSygv4ke9wCrv8LWvYQ_nearend_singletalk` | 3.323 |
| `pPBidi8oyUarZCGUrKJEsg_nearend_singletalk` | 3.324 |
| `2f9FDnH8u0q829njqEuVrQ_nearend_singletalk` | 3.351 |
| `mAJHXrZ5QU62Do1OqTSL6Q_nearend_singletalk` | 3.386 |
| `x1pZr2uWBkulb77zcCfAJA_nearend_singletalk` | 3.400 |
| `IvAFDPUFEk0GuW2VKRHznA_nearend_singletalk` | 3.401 |
| `LAdLsNzbcE66K0Zs1NE5QA_nearend_singletalk` | 3.410 |
| `XCXcCwUPY0GmrtqtJ6xY2g_nearend_singletalk` | 3.465 |
| `wJVPo4lexUK40x0nuK0KWg_nearend_singletalk` | 3.468 |
| `VNkNShj97UajHDVbSmIG0g_nearend_singletalk` | 3.476 |
| `mljUO9k4gUiYCHgXSsfo5A_nearend_singletalk` | 3.483 |
