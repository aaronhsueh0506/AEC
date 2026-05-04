# Bench result — current

Dataset: /Users/mingyu/Desktop/novatek/SE/AEC/wav/aec_challenge_blind
Output dir: experiments/r7_p2_mix_r20/eval_output
Cases scored: 800

## Bucket means

| Bucket | n | echo (↑) | deg (↑) |
|---|---:|---:|---:|
| FS_static | 169 | 3.261 | 5.000 |
| FS_movement | 131 | 3.354 | 5.000 |
| DT_static | 186 | 3.660 | 2.736 |
| DT_movement | 114 | 3.637 | 2.718 |
| NE | 200 | 4.998 | 3.985 |

## Δ vs baseline (baseline_v381_seeded)

| Bucket | Δecho | Δdeg | verdict |
|---|---:|---:|---|
| FS_static | -0.507 | +0.001 | FS echo regress |
| FS_movement | -0.473 | +0.000 | FS echo regress |
| DT_static | -0.592 | +0.480 | ok |
| DT_movement | -0.483 | +0.431 | ok |
| NE | +0.000 | -0.022 | NE deg regress |

## Worst-20 per bucket

### FS_static (sorted by echo ascending)

| stem | echo |
|---|---:|
| `Gsy0lC5QSUi540hiax9XtA_farend_singletalk` | 1.492 |
| `pcb1Nh0Z3k0WS9a7gBEuqg_farend_singletalk` | 1.611 |
| `JtodX3Ug6Eu5TYu0HN5IOw_farend_singletalk` | 1.629 |
| `9xjhiFbGo06hdQIsHTS6qA_farend_singletalk` | 1.642 |
| `7GTxyTksSUqCnP5y0ILG4A_farend_singletalk` | 1.667 |
| `pmzLFdKTzEixfU0l0furvA_farend_singletalk` | 1.778 |
| `VJfVUwJs4k25ziMNvJb43A_farend_singletalk` | 1.787 |
| `IrQvqOTCmEWMXn9k2ICtRQ_farend_singletalk` | 1.790 |
| `S22FCqKDWUyymN1YbpItIw_farend_singletalk` | 1.792 |
| `hVqUmGvIlkO0LBUoE06Q3w_farend_singletalk` | 1.843 |
| `WTdBhXa080WJEeGDde9BGA_farend_singletalk` | 1.955 |
| `s90M7MOTBkqaV4nQPLhKbA_farend_singletalk` | 1.966 |
| `iOyPaxX11UOaUkcscKhq1A_farend_singletalk` | 1.995 |
| `JteZUZ4JYkeD4k2rcVbqHg_farend_singletalk` | 2.017 |
| `TcuegeGKL06ysmFP0RxjyQ_farend_singletalk` | 2.020 |
| `DS5pS5fZwEG4X0NNifTh0w_farend_singletalk` | 2.061 |
| `QYHsugUAcUWEQ9WghnG0Jw_farend_singletalk` | 2.069 |
| `XTqo1aOXDEiqyWTFK99I5Q_farend_singletalk` | 2.149 |
| `yM2wHof9U06yVPJfemZ3hg_farend_singletalk` | 2.150 |
| `o97uExi7MEqzbanuWK6CCw_farend_singletalk` | 2.159 |

### FS_movement (sorted by echo ascending)

| stem | echo |
|---|---:|
| `WH0jN3PY40es2S0LsxmkkQ_farend_singletalk_with_movement` | 1.771 |
| `hVqUmGvIlkO0LBUoE06Q3w_farend_singletalk_with_movement` | 1.805 |
| `XTqo1aOXDEiqyWTFK99I5Q_farend_singletalk_with_movement` | 1.821 |
| `0luXwWjGEEC9G5nf0yTVXw_farend_singletalk_with_movement` | 1.868 |
| `xFk7igecuke0R5JMfREyDg_farend_singletalk_with_movement` | 1.898 |
| `VNgRsWxMdkaUx1gKV9W1Zw_farend_singletalk_with_movement` | 1.984 |
| `S22FCqKDWUyymN1YbpItIw_farend_singletalk_with_movement` | 2.058 |
| `TGZ5Wq0SCUCOXPsfee3uMQ_farend_singletalk_with_movement` | 2.062 |
| `iOyPaxX11UOaUkcscKhq1A_farend_singletalk_with_movement` | 2.069 |
| `zONvcX0qYkuaAViV5PXcYg_farend_singletalk_with_movement` | 2.108 |
| `XXz0qkUSd0GT4dsywxpfJg_farend_singletalk_with_movement` | 2.116 |
| `IqtJR4tjJkWrwUjYorz0Og_farend_singletalk_with_movement` | 2.140 |
| `IrQvqOTCmEWMXn9k2ICtRQ_farend_singletalk_with_movement` | 2.143 |
| `N2rQLbnp2UOg2QFRaggbDw_farend_singletalk_with_movement` | 2.196 |
| `SgKY30fjT0G8e3kQL0RHSQ_farend_singletalk_with_movement` | 2.200 |
| `OXCtw0FVhUWGUBil6Uucdw_farend_singletalk_with_movement` | 2.236 |
| `wlAXM0iDgkm06i7UdRww1w_farend_singletalk_with_movement` | 2.271 |
| `ZWq0X5sPiUe0lQjZdCPSeQ_farend_singletalk_with_movement` | 2.292 |
| `XuguA1uJAE0bWT0xXRDdeA_farend_singletalk_with_movement` | 2.327 |
| `yyvS0Ljh1k0AHMx6cxtNyg_farend_singletalk_with_movement` | 2.355 |

### DT_static (sorted by deg ascending)

| stem | deg |
|---|---:|
| `p0mhFbhV6kGJgjd0RTTIIw_doubletalk` | 1.243 |
| `SgKY30fjT0G8e3kQL0RHSQ_doubletalk` | 1.252 |
| `QK70KpLuZ0O43BBSWEZvHg_doubletalk` | 1.316 |
| `sYQK1rJlwU2XCy20n0Sx9g_doubletalk` | 1.463 |
| `wLRC9k23Bk6kmBzZdbkHTA_doubletalk` | 1.543 |
| `V6Mw0Ti8RUSkzvMGB4WGiw_doubletalk` | 1.544 |
| `QkRkwwFKVEar0WtcuvJsZg_doubletalk` | 1.546 |
| `qVd1gtwQ0k2lVRqPVp1NKQ_doubletalk` | 1.589 |
| `XuiheB7eUkyJA2XzFIovHQ_doubletalk` | 1.606 |
| `xb7eJJF0Vki6Yl3y4B7oJA_doubletalk` | 1.631 |
| `XtUWcfsuykC5fTu7DdAnnw_doubletalk` | 1.678 |
| `pPBidi8oyUarZCGUrKJEsg_doubletalk` | 1.687 |
| `wY3Rp9YEjkOke09hMwfsjg_doubletalk` | 1.700 |
| `nVUnxqHLr0GTN7shWid1Ow_doubletalk` | 1.755 |
| `UmlD9X38NECNoJKm0oyf4w_doubletalk` | 1.807 |
| `0I0XMl3M0ECO0U1N0cJvpg_doubletalk` | 1.838 |
| `nZQu09pizke3LNOn6uaU0A_doubletalk` | 1.924 |
| `OjdIdZgJDk6hLAQL07KORA_doubletalk` | 1.927 |
| `Uc4dmejgWUCTvn0XZbMTBw_doubletalk` | 1.951 |
| `Je6gJ7y1PECStwxnrOe9aA_doubletalk` | 1.969 |

### DT_movement (sorted by deg ascending)

| stem | deg |
|---|---:|
| `XV5L2dn3S06M9GBEu1q3DA_doubletalk_with_movement` | 1.258 |
| `WqEYNwalSUebZxaeYVay2g_doubletalk_with_movement` | 1.324 |
| `wHmBm7VHfkysBOhjoAXkNA_doubletalk_with_movement` | 1.388 |
| `I2bme08keUmAnyJRKNYDGQ_doubletalk_with_movement` | 1.513 |
| `WH7rA6R2zkyopKUrcq9p3A_doubletalk_with_movement` | 1.519 |
| `XuiheB7eUkyJA2XzFIovHQ_doubletalk_with_movement` | 1.588 |
| `w5XDRNfB2Ei2UoUDtrTkzg_doubletalk_with_movement` | 1.595 |
| `V0JqgjlrB0Ke9y91r0rxNw_doubletalk_with_movement` | 1.602 |
| `zOiK6oSHp0ib3nHvzLKbRQ_doubletalk_with_movement` | 1.648 |
| `X7Ua9txMj0aws848JPEbOg_doubletalk_with_movement` | 1.675 |
| `VgSXlJJEI02dytkMm5UTzA_doubletalk_with_movement` | 1.743 |
| `u0X5XB2KzEGduXtfWfjGDw_doubletalk_with_movement` | 1.813 |
| `xofDX004bkqiOv9YOxmGVQ_doubletalk_with_movement` | 1.823 |
| `QkRkwwFKVEar0WtcuvJsZg_doubletalk_with_movement` | 1.860 |
| `iyuYIcszXku7BWYOOwqh5Q_doubletalk_with_movement` | 1.886 |
| `xSh5yXWiP02K0UkYdkZ0cA_doubletalk_with_movement` | 1.894 |
| `Xvyiz1o0cEijZQ8DT9mB2w_doubletalk_with_movement` | 1.902 |
| `XCXcCwUPY0GmrtqtJ6xY2g_doubletalk_with_movement` | 1.952 |
| `TGZ5Wq0SCUCOXPsfee3uMQ_doubletalk_with_movement` | 1.962 |
| `qkGW9Frbs0Gq5gdfsztA2g_doubletalk_with_movement` | 2.074 |

### NE (sorted by deg ascending)

| stem | deg |
|---|---:|
| `SR68lGQwTUy508j0P8BKZQ_nearend_singletalk` | 2.999 |
| `kOtW70qgikKm0F9OEQw22A_nearend_singletalk` | 3.015 |
| `DW3NNHoKl02cAfEkahzsGg_nearend_singletalk` | 3.047 |
| `kz23X4pDSEiPmWtw2Qx00Q_nearend_singletalk` | 3.056 |
| `08Mo9p6KVUapuF8PNAMWaw_nearend_singletalk` | 3.141 |
| `A50fBOz02kag8CVAhBOF8A_nearend_singletalk` | 3.160 |
| `4pN9yn7mhEa5iDiKnr5jlw_nearend_singletalk` | 3.263 |
| `2f9FDnH8u0q829njqEuVrQ_nearend_singletalk` | 3.271 |
| `mAJHXrZ5QU62Do1OqTSL6Q_nearend_singletalk` | 3.321 |
| `LAdLsNzbcE66K0Zs1NE5QA_nearend_singletalk` | 3.369 |
| `pPBidi8oyUarZCGUrKJEsg_nearend_singletalk` | 3.377 |
| `x7Frsl5wTEKmog3A6lL57g_nearend_singletalk` | 3.404 |
| `IvAFDPUFEk0GuW2VKRHznA_nearend_singletalk` | 3.432 |
| `SFvlSygv4ke9wCrv8LWvYQ_nearend_singletalk` | 3.433 |
| `XCXcCwUPY0GmrtqtJ6xY2g_nearend_singletalk` | 3.442 |
| `x1pZr2uWBkulb77zcCfAJA_nearend_singletalk` | 3.450 |
| `VNkNShj97UajHDVbSmIG0g_nearend_singletalk` | 3.486 |
| `yN0NYysKnUW1PsMYdU4tQA_nearend_singletalk` | 3.489 |
| `pcb1Nh0Z3k0WS9a7gBEuqg_nearend_singletalk` | 3.498 |
| `m6x1OA5vH0mUhq1QZELg8g_nearend_singletalk` | 3.508 |
