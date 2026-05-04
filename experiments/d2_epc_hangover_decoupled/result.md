# Bench result — d2_epc_hangover

Dataset: /Users/mingyu/Desktop/novatek/SE/AEC/wav/aec_challenge_blind
Output dir: experiments/d2_epc_hangover_decoupled/output
Cases scored: 800

## Bucket means

| Bucket | n | echo (↑) | deg (↑) |
|---|---:|---:|---:|
| FS_static | 169 | 3.710 | 4.999 |
| FS_movement | 131 | 3.720 | 4.999 |
| DT_static | 186 | 4.242 | 2.261 |
| DT_movement | 114 | 4.100 | 2.297 |
| NE | 200 | 4.998 | 4.007 |

## Δ vs baseline (baseline_v381_seeded)

| Bucket | Δecho | Δdeg | verdict |
|---|---:|---:|---|
| FS_static | -0.058 | +0.000 | FS echo regress |
| FS_movement | -0.108 | +0.000 | FS echo regress |
| DT_static | -0.009 | +0.004 | ok |
| DT_movement | -0.020 | +0.011 | ok |
| NE | +0.000 | +0.000 | ok |

## Worst-20 per bucket

### FS_static (sorted by echo ascending)

| stem | echo |
|---|---:|
| `7GTxyTksSUqCnP5y0ILG4A_farend_singletalk` | 1.313 |
| `9xjhiFbGo06hdQIsHTS6qA_farend_singletalk` | 1.779 |
| `IrQvqOTCmEWMXn9k2ICtRQ_farend_singletalk` | 1.826 |
| `hVqUmGvIlkO0LBUoE06Q3w_farend_singletalk` | 1.886 |
| `pcb1Nh0Z3k0WS9a7gBEuqg_farend_singletalk` | 1.911 |
| `S22FCqKDWUyymN1YbpItIw_farend_singletalk` | 1.918 |
| `VJfVUwJs4k25ziMNvJb43A_farend_singletalk` | 2.029 |
| `Gsy0lC5QSUi540hiax9XtA_farend_singletalk` | 2.062 |
| `pmzLFdKTzEixfU0l0furvA_farend_singletalk` | 2.164 |
| `JteZUZ4JYkeD4k2rcVbqHg_farend_singletalk` | 2.168 |
| `N2rQLbnp2UOg2QFRaggbDw_farend_singletalk` | 2.203 |
| `m4789fdio0q92zjf9gvh1Q_farend_singletalk` | 2.212 |
| `geGGo8g9UE2MYkdjxtHm8w_farend_singletalk` | 2.284 |
| `JtodX3Ug6Eu5TYu0HN5IOw_farend_singletalk` | 2.291 |
| `iOyPaxX11UOaUkcscKhq1A_farend_singletalk` | 2.336 |
| `OXCtw0FVhUWGUBil6Uucdw_farend_singletalk` | 2.445 |
| `s90M7MOTBkqaV4nQPLhKbA_farend_singletalk` | 2.466 |
| `LeV1uF4j10Whm0FPG80tmw_farend_singletalk` | 2.479 |
| `sLWe8bfYbkGwX1W3PzI1PQ_farend_singletalk` | 2.553 |
| `60JogIkuEECYHrwVT3UYwg_farend_singletalk` | 2.561 |

### FS_movement (sorted by echo ascending)

| stem | echo |
|---|---:|
| `S22FCqKDWUyymN1YbpItIw_farend_singletalk_with_movement` | 1.859 |
| `iOyPaxX11UOaUkcscKhq1A_farend_singletalk_with_movement` | 2.003 |
| `hVqUmGvIlkO0LBUoE06Q3w_farend_singletalk_with_movement` | 2.009 |
| `0luXwWjGEEC9G5nf0yTVXw_farend_singletalk_with_movement` | 2.020 |
| `IrQvqOTCmEWMXn9k2ICtRQ_farend_singletalk_with_movement` | 2.048 |
| `IqtJR4tjJkWrwUjYorz0Og_farend_singletalk_with_movement` | 2.221 |
| `OXCtw0FVhUWGUBil6Uucdw_farend_singletalk_with_movement` | 2.294 |
| `TGZ5Wq0SCUCOXPsfee3uMQ_farend_singletalk_with_movement` | 2.296 |
| `JtodX3Ug6Eu5TYu0HN5IOw_farend_singletalk_with_movement` | 2.369 |
| `SgKY30fjT0G8e3kQL0RHSQ_farend_singletalk_with_movement` | 2.394 |
| `5bJUo1K3uEmMrGa9UhGyVg_farend_singletalk_with_movement` | 2.432 |
| `JjCzlhn3gEiBQvfJtPNJ9A_farend_singletalk_with_movement` | 2.468 |
| `N2rQLbnp2UOg2QFRaggbDw_farend_singletalk_with_movement` | 2.478 |
| `XTqo1aOXDEiqyWTFK99I5Q_farend_singletalk_with_movement` | 2.613 |
| `yyvS0Ljh1k0AHMx6cxtNyg_farend_singletalk_with_movement` | 2.620 |
| `xFk7igecuke0R5JMfREyDg_farend_singletalk_with_movement` | 2.630 |
| `VNkNShj97UajHDVbSmIG0g_farend_singletalk_with_movement` | 2.737 |
| `QEeKiaNiDECfqXTRrDFWWw_farend_singletalk_with_movement` | 2.766 |
| `XuguA1uJAE0bWT0xXRDdeA_farend_singletalk_with_movement` | 2.775 |
| `LeV1uF4j10Whm0FPG80tmw_farend_singletalk_with_movement` | 2.776 |

### DT_static (sorted by deg ascending)

| stem | deg |
|---|---:|
| `p0mhFbhV6kGJgjd0RTTIIw_doubletalk` | 1.216 |
| `QK70KpLuZ0O43BBSWEZvHg_doubletalk` | 1.231 |
| `X0OraQOKtkCVriR0uj0WBQ_doubletalk` | 1.238 |
| `XuiheB7eUkyJA2XzFIovHQ_doubletalk` | 1.245 |
| `MYrVxVEMxkaE7OuyTUmI0Q_doubletalk` | 1.247 |
| `XtUWcfsuykC5fTu7DdAnnw_doubletalk` | 1.260 |
| `SgKY30fjT0G8e3kQL0RHSQ_doubletalk` | 1.275 |
| `pPBidi8oyUarZCGUrKJEsg_doubletalk` | 1.302 |
| `sUQrHEPAoEmIvHclpi1tRQ_doubletalk` | 1.315 |
| `yc5bFUGsR0GSfiGwTTpRWg_doubletalk` | 1.319 |
| `xofDX004bkqiOv9YOxmGVQ_doubletalk` | 1.352 |
| `XV5L2dn3S06M9GBEu1q3DA_doubletalk` | 1.356 |
| `QkRkwwFKVEar0WtcuvJsZg_doubletalk` | 1.378 |
| `NY3kZioAm0KwR45wIVe2Sg_doubletalk` | 1.384 |
| `XCXcCwUPY0GmrtqtJ6xY2g_doubletalk` | 1.391 |
| `V6Mw0Ti8RUSkzvMGB4WGiw_doubletalk` | 1.395 |
| `nlSSRl4k50Gq2mIRYlMBCg_doubletalk` | 1.425 |
| `sYQK1rJlwU2XCy20n0Sx9g_doubletalk` | 1.426 |
| `0I0XMl3M0ECO0U1N0cJvpg_doubletalk` | 1.441 |
| `XGDaZuEkE0WU4IN0Yi4XtA_doubletalk` | 1.447 |

### DT_movement (sorted by deg ascending)

| stem | deg |
|---|---:|
| `XV5L2dn3S06M9GBEu1q3DA_doubletalk_with_movement` | 1.158 |
| `wHmBm7VHfkysBOhjoAXkNA_doubletalk_with_movement` | 1.271 |
| `WqEYNwalSUebZxaeYVay2g_doubletalk_with_movement` | 1.299 |
| `xofDX004bkqiOv9YOxmGVQ_doubletalk_with_movement` | 1.307 |
| `XuiheB7eUkyJA2XzFIovHQ_doubletalk_with_movement` | 1.316 |
| `w0QrMwsZ5kGoJjRWvP0iKg_doubletalk_with_movement` | 1.354 |
| `VgSXlJJEI02dytkMm5UTzA_doubletalk_with_movement` | 1.363 |
| `I2bme08keUmAnyJRKNYDGQ_doubletalk_with_movement` | 1.389 |
| `X7Ua9txMj0aws848JPEbOg_doubletalk_with_movement` | 1.393 |
| `Xvyiz1o0cEijZQ8DT9mB2w_doubletalk_with_movement` | 1.395 |
| `QkRkwwFKVEar0WtcuvJsZg_doubletalk_with_movement` | 1.398 |
| `ql7yTcebJU20VE5qpW0kCA_doubletalk_with_movement` | 1.398 |
| `WcK0OrF6ukW03fViPXTQjQ_doubletalk_with_movement` | 1.415 |
| `nyT6FUUdu0W8UpvjP1rRgQ_doubletalk_with_movement` | 1.436 |
| `nlSSRl4k50Gq2mIRYlMBCg_doubletalk_with_movement` | 1.457 |
| `V0JqgjlrB0Ke9y91r0rxNw_doubletalk_with_movement` | 1.471 |
| `SwfEwuGDlkWYy9pb4H00eQ_doubletalk_with_movement` | 1.517 |
| `w5XDRNfB2Ei2UoUDtrTkzg_doubletalk_with_movement` | 1.536 |
| `zOiK6oSHp0ib3nHvzLKbRQ_doubletalk_with_movement` | 1.544 |
| `qkGW9Frbs0Gq5gdfsztA2g_doubletalk_with_movement` | 1.564 |

### NE (sorted by deg ascending)

| stem | deg |
|---|---:|
| `SR68lGQwTUy508j0P8BKZQ_nearend_singletalk` | 2.903 |
| `kz23X4pDSEiPmWtw2Qx00Q_nearend_singletalk` | 2.964 |
| `SFvlSygv4ke9wCrv8LWvYQ_nearend_singletalk` | 3.065 |
| `kOtW70qgikKm0F9OEQw22A_nearend_singletalk` | 3.139 |
| `A50fBOz02kag8CVAhBOF8A_nearend_singletalk` | 3.191 |
| `DW3NNHoKl02cAfEkahzsGg_nearend_singletalk` | 3.254 |
| `pPBidi8oyUarZCGUrKJEsg_nearend_singletalk` | 3.295 |
| `2f9FDnH8u0q829njqEuVrQ_nearend_singletalk` | 3.318 |
| `4pN9yn7mhEa5iDiKnr5jlw_nearend_singletalk` | 3.323 |
| `IvAFDPUFEk0GuW2VKRHznA_nearend_singletalk` | 3.362 |
| `08Mo9p6KVUapuF8PNAMWaw_nearend_singletalk` | 3.362 |
| `E0l0WVPQjEi6AmtbvfSYLA_nearend_singletalk` | 3.364 |
| `mAJHXrZ5QU62Do1OqTSL6Q_nearend_singletalk` | 3.379 |
| `x1pZr2uWBkulb77zcCfAJA_nearend_singletalk` | 3.433 |
| `LAdLsNzbcE66K0Zs1NE5QA_nearend_singletalk` | 3.433 |
| `x7Frsl5wTEKmog3A6lL57g_nearend_singletalk` | 3.442 |
| `mljUO9k4gUiYCHgXSsfo5A_nearend_singletalk` | 3.479 |
| `XCXcCwUPY0GmrtqtJ6xY2g_nearend_singletalk` | 3.483 |
| `pcb1Nh0Z3k0WS9a7gBEuqg_nearend_singletalk` | 3.484 |
| `VNkNShj97UajHDVbSmIG0g_nearend_singletalk` | 3.488 |
