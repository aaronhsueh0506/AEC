# Bench result — current

Dataset: /Users/mingyu/Desktop/novatek/SE/AEC/wav/aec_challenge_blind
Output dir: out_python_v3_21_2_u53/
Cases scored: 800

## Bucket means

| Bucket | n | echo (↑) | deg (↑) |
|---|---:|---:|---:|
| FS_static | 169 | 3.582 | 4.999 |
| FS_movement | 131 | 3.509 | 4.999 |
| DT_static | 186 | 4.188 | 2.481 |
| DT_movement | 114 | 4.166 | 2.485 |
| NE | 200 | 4.998 | 4.054 |

## Δ vs baseline (current)

| Bucket | Δecho | Δdeg | verdict |
|---|---:|---:|---|
| FS_static | -0.147 | +0.000 | FS echo regress |
| FS_movement | -0.117 | +0.000 | FS echo regress |
| DT_static | -0.049 | +0.094 | ok |
| DT_movement | -0.048 | +0.115 | ok |
| NE | -0.000 | +0.003 | ok |

## Worst-20 per bucket

### FS_static (sorted by echo ascending)

| stem | echo |
|---|---:|
| `pcb1Nh0Z3k0WS9a7gBEuqg_farend_singletalk` | 2.036 |
| `LN18k5r8t00C9DulUd809A_farend_singletalk` | 2.124 |
| `WTdBhXa080WJEeGDde9BGA_farend_singletalk` | 2.224 |
| `s90M7MOTBkqaV4nQPLhKbA_farend_singletalk` | 2.227 |
| `JteZUZ4JYkeD4k2rcVbqHg_farend_singletalk` | 2.397 |
| `9xjhiFbGo06hdQIsHTS6qA_farend_singletalk` | 2.399 |
| `KSN5Jrzo7kaixP0z8xfr4Q_farend_singletalk` | 2.438 |
| `Gsy0lC5QSUi540hiax9XtA_farend_singletalk` | 2.474 |
| `LeV1uF4j10Whm0FPG80tmw_farend_singletalk` | 2.584 |
| `wWeNtFK0dEG9Wub40bB15A_farend_singletalk` | 2.598 |
| `lV0kQN0hR0ySmE0bQhuYbw_farend_singletalk` | 2.603 |
| `ql7yTcebJU20VE5qpW0kCA_farend_singletalk` | 2.715 |
| `zddPqpp1a06xttKdc0iNTA_farend_singletalk` | 2.719 |
| `QYHsugUAcUWEQ9WghnG0Jw_farend_singletalk` | 2.759 |
| `pU21kfoo0UOz0fPMJFfydg_farend_singletalk` | 2.775 |
| `o2wfdvOGwU6M8Fmn2dCvOA_farend_singletalk` | 2.776 |
| `JtodX3Ug6Eu5TYu0HN5IOw_farend_singletalk` | 2.790 |
| `hVqUmGvIlkO0LBUoE06Q3w_farend_singletalk` | 2.819 |
| `VJfVUwJs4k25ziMNvJb43A_farend_singletalk` | 2.833 |
| `sLWe8bfYbkGwX1W3PzI1PQ_farend_singletalk` | 2.869 |

### FS_movement (sorted by echo ascending)

| stem | echo |
|---|---:|
| `yyvS0Ljh1k0AHMx6cxtNyg_farend_singletalk_with_movement` | 1.828 |
| `IrQvqOTCmEWMXn9k2ICtRQ_farend_singletalk_with_movement` | 1.925 |
| `iOyPaxX11UOaUkcscKhq1A_farend_singletalk_with_movement` | 2.090 |
| `JjCzlhn3gEiBQvfJtPNJ9A_farend_singletalk_with_movement` | 2.182 |
| `OXCtw0FVhUWGUBil6Uucdw_farend_singletalk_with_movement` | 2.241 |
| `kZogUfYct0qMwSqvRTwOVg_farend_singletalk_with_movement` | 2.313 |
| `SgKY30fjT0G8e3kQL0RHSQ_farend_singletalk_with_movement` | 2.424 |
| `zddPqpp1a06xttKdc0iNTA_farend_singletalk_with_movement` | 2.425 |
| `ZWq0X5sPiUe0lQjZdCPSeQ_farend_singletalk_with_movement` | 2.470 |
| `0luXwWjGEEC9G5nf0yTVXw_farend_singletalk_with_movement` | 2.486 |
| `pmzLFdKTzEixfU0l0furvA_farend_singletalk_with_movement` | 2.533 |
| `kz23X4pDSEiPmWtw2Qx00Q_farend_singletalk_with_movement` | 2.549 |
| `5bJUo1K3uEmMrGa9UhGyVg_farend_singletalk_with_movement` | 2.574 |
| `LeV1uF4j10Whm0FPG80tmw_farend_singletalk_with_movement` | 2.583 |
| `sRCs6SKo6kC0xire475q0A_farend_singletalk_with_movement` | 2.652 |
| `Xv7jH2KcBEWqdpbT000HQA_farend_singletalk_with_movement` | 2.654 |
| `wr54weKzNkOcZ07hB04kzA_farend_singletalk_with_movement` | 2.689 |
| `wlAXM0iDgkm06i7UdRww1w_farend_singletalk_with_movement` | 2.693 |
| `hVqUmGvIlkO0LBUoE06Q3w_farend_singletalk_with_movement` | 2.697 |
| `TGZ5Wq0SCUCOXPsfee3uMQ_farend_singletalk_with_movement` | 2.714 |

### DT_static (sorted by deg ascending)

| stem | deg |
|---|---:|
| `QK70KpLuZ0O43BBSWEZvHg_doubletalk` | 1.267 |
| `sYQK1rJlwU2XCy20n0Sx9g_doubletalk` | 1.281 |
| `QkRkwwFKVEar0WtcuvJsZg_doubletalk` | 1.327 |
| `SgKY30fjT0G8e3kQL0RHSQ_doubletalk` | 1.332 |
| `xYuPW7feGkyc8a1rfcDv9w_doubletalk` | 1.334 |
| `XV5L2dn3S06M9GBEu1q3DA_doubletalk` | 1.367 |
| `yc5bFUGsR0GSfiGwTTpRWg_doubletalk` | 1.371 |
| `X0OraQOKtkCVriR0uj0WBQ_doubletalk` | 1.497 |
| `Uc4dmejgWUCTvn0XZbMTBw_doubletalk` | 1.503 |
| `wY3Rp9YEjkOke09hMwfsjg_doubletalk` | 1.558 |
| `wLRC9k23Bk6kmBzZdbkHTA_doubletalk` | 1.560 |
| `pPBidi8oyUarZCGUrKJEsg_doubletalk` | 1.581 |
| `XnfMDZLl0U2WvLRphiGJ6A_doubletalk` | 1.598 |
| `S5cyhx02u00eJzrHTxVTwQ_doubletalk` | 1.616 |
| `XtUWcfsuykC5fTu7DdAnnw_doubletalk` | 1.621 |
| `Pu8CtSffMUiINQAhSKvlfw_doubletalk` | 1.631 |
| `kGFOHthwrUWqHCLkBYIQnA_doubletalk` | 1.643 |
| `NY3kZioAm0KwR45wIVe2Sg_doubletalk` | 1.669 |
| `xWOALEtAwk2oABUz3vAv6w_doubletalk` | 1.669 |
| `wHmBm7VHfkysBOhjoAXkNA_doubletalk` | 1.684 |

### DT_movement (sorted by deg ascending)

| stem | deg |
|---|---:|
| `wHmBm7VHfkysBOhjoAXkNA_doubletalk_with_movement` | 1.193 |
| `XV5L2dn3S06M9GBEu1q3DA_doubletalk_with_movement` | 1.255 |
| `X7Ua9txMj0aws848JPEbOg_doubletalk_with_movement` | 1.356 |
| `7GTxyTksSUqCnP5y0ILG4A_doubletalk_with_movement` | 1.446 |
| `I2bme08keUmAnyJRKNYDGQ_doubletalk_with_movement` | 1.447 |
| `xofDX004bkqiOv9YOxmGVQ_doubletalk_with_movement` | 1.468 |
| `N2rQLbnp2UOg2QFRaggbDw_doubletalk_with_movement` | 1.503 |
| `uLl640xveUuHp2kEtOCTeQ_doubletalk_with_movement` | 1.526 |
| `WnDjVFWmC0m0WhVq22mRlQ_doubletalk_with_movement` | 1.532 |
| `VgSXlJJEI02dytkMm5UTzA_doubletalk_with_movement` | 1.547 |
| `WcK0OrF6ukW03fViPXTQjQ_doubletalk_with_movement` | 1.591 |
| `Xvyiz1o0cEijZQ8DT9mB2w_doubletalk_with_movement` | 1.689 |
| `w0QrMwsZ5kGoJjRWvP0iKg_doubletalk_with_movement` | 1.691 |
| `nyT6FUUdu0W8UpvjP1rRgQ_doubletalk_with_movement` | 1.692 |
| `xvACDxradUuKNYImFSd1ww_doubletalk_with_movement` | 1.702 |
| `VNkNShj97UajHDVbSmIG0g_doubletalk_with_movement` | 1.731 |
| `zzCIhneJ8UKTWZ48U0kRXw_doubletalk_with_movement` | 1.787 |
| `XCXcCwUPY0GmrtqtJ6xY2g_doubletalk_with_movement` | 1.788 |
| `zOiK6oSHp0ib3nHvzLKbRQ_doubletalk_with_movement` | 1.813 |
| `m6ciKvH6AEe7Yi2ptKjj1g_doubletalk_with_movement` | 1.823 |

### NE (sorted by deg ascending)

| stem | deg |
|---|---:|
| `kz23X4pDSEiPmWtw2Qx00Q_nearend_singletalk` | 2.924 |
| `LeV1uF4j10Whm0FPG80tmw_nearend_singletalk` | 2.964 |
| `DW3NNHoKl02cAfEkahzsGg_nearend_singletalk` | 2.980 |
| `S73GmH9ok0GBbaG3esxbQQ_nearend_singletalk` | 3.103 |
| `SR68lGQwTUy508j0P8BKZQ_nearend_singletalk` | 3.176 |
| `VNkNShj97UajHDVbSmIG0g_nearend_singletalk` | 3.227 |
| `IP5Pznnj10qkIqqrGMFGMg_nearend_singletalk` | 3.239 |
| `QEeKiaNiDECfqXTRrDFWWw_nearend_singletalk` | 3.357 |
| `yN0NYysKnUW1PsMYdU4tQA_nearend_singletalk` | 3.384 |
| `4pN9yn7mhEa5iDiKnr5jlw_nearend_singletalk` | 3.391 |
| `SFvlSygv4ke9wCrv8LWvYQ_nearend_singletalk` | 3.435 |
| `pPBidi8oyUarZCGUrKJEsg_nearend_singletalk` | 3.448 |
| `x7Frsl5wTEKmog3A6lL57g_nearend_singletalk` | 3.454 |
| `2f9FDnH8u0q829njqEuVrQ_nearend_singletalk` | 3.456 |
| `mAJHXrZ5QU62Do1OqTSL6Q_nearend_singletalk` | 3.482 |
| `kOtW70qgikKm0F9OEQw22A_nearend_singletalk` | 3.482 |
| `A50fBOz02kag8CVAhBOF8A_nearend_singletalk` | 3.504 |
| `IvAFDPUFEk0GuW2VKRHznA_nearend_singletalk` | 3.508 |
| `08Mo9p6KVUapuF8PNAMWaw_nearend_singletalk` | 3.515 |
| `XYaVZ4A9B06EYZcGugD3oQ_nearend_singletalk` | 3.534 |
