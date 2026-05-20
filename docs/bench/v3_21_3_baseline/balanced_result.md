# Bench result — current

Dataset: /Users/mingyu/Desktop/novatek/SE/AEC/wav/aec_challenge_blind
Output dir: out_python_v3_21_3_codex/
Cases scored: 800

## Bucket means

| Bucket | n | echo (↑) | deg (↑) |
|---|---:|---:|---:|
| FS_static | 169 | 3.532 | 4.999 |
| FS_movement | 131 | 3.472 | 4.999 |
| DT_static | 186 | 4.161 | 2.506 |
| DT_movement | 114 | 4.135 | 2.511 |
| NE | 200 | 4.998 | 4.054 |

## Δ vs baseline (current)

| Bucket | Δecho | Δdeg | verdict |
|---|---:|---:|---|
| FS_static | -0.050 | +0.000 | FS echo regress |
| FS_movement | -0.037 | +0.000 | FS echo regress |
| DT_static | -0.028 | +0.025 | ok |
| DT_movement | -0.032 | +0.026 | ok |
| NE | +0.000 | +0.000 | ok |

## Worst-20 per bucket

### FS_static (sorted by echo ascending)

| stem | echo |
|---|---:|
| `s90M7MOTBkqaV4nQPLhKbA_farend_singletalk` | 1.968 |
| `LN18k5r8t00C9DulUd809A_farend_singletalk` | 2.047 |
| `pcb1Nh0Z3k0WS9a7gBEuqg_farend_singletalk` | 2.111 |
| `9xjhiFbGo06hdQIsHTS6qA_farend_singletalk` | 2.197 |
| `lV0kQN0hR0ySmE0bQhuYbw_farend_singletalk` | 2.210 |
| `JteZUZ4JYkeD4k2rcVbqHg_farend_singletalk` | 2.211 |
| `WTdBhXa080WJEeGDde9BGA_farend_singletalk` | 2.224 |
| `KSN5Jrzo7kaixP0z8xfr4Q_farend_singletalk` | 2.438 |
| `Gsy0lC5QSUi540hiax9XtA_farend_singletalk` | 2.474 |
| `wWeNtFK0dEG9Wub40bB15A_farend_singletalk` | 2.486 |
| `QYHsugUAcUWEQ9WghnG0Jw_farend_singletalk` | 2.538 |
| `LeV1uF4j10Whm0FPG80tmw_farend_singletalk` | 2.584 |
| `zddPqpp1a06xttKdc0iNTA_farend_singletalk` | 2.629 |
| `o2wfdvOGwU6M8Fmn2dCvOA_farend_singletalk` | 2.632 |
| `ql7yTcebJU20VE5qpW0kCA_farend_singletalk` | 2.667 |
| `TZ6TJFCbfkKAVrS64Sf08Q_farend_singletalk` | 2.760 |
| `pU21kfoo0UOz0fPMJFfydg_farend_singletalk` | 2.783 |
| `JtodX3Ug6Eu5TYu0HN5IOw_farend_singletalk` | 2.790 |
| `orvXZE0juUeRPAAdjZSqoA_farend_singletalk` | 2.819 |
| `hVqUmGvIlkO0LBUoE06Q3w_farend_singletalk` | 2.819 |

### FS_movement (sorted by echo ascending)

| stem | echo |
|---|---:|
| `yyvS0Ljh1k0AHMx6cxtNyg_farend_singletalk_with_movement` | 1.759 |
| `IrQvqOTCmEWMXn9k2ICtRQ_farend_singletalk_with_movement` | 1.902 |
| `iOyPaxX11UOaUkcscKhq1A_farend_singletalk_with_movement` | 2.090 |
| `JjCzlhn3gEiBQvfJtPNJ9A_farend_singletalk_with_movement` | 2.193 |
| `OXCtw0FVhUWGUBil6Uucdw_farend_singletalk_with_movement` | 2.241 |
| `ZWq0X5sPiUe0lQjZdCPSeQ_farend_singletalk_with_movement` | 2.295 |
| `kZogUfYct0qMwSqvRTwOVg_farend_singletalk_with_movement` | 2.321 |
| `SgKY30fjT0G8e3kQL0RHSQ_farend_singletalk_with_movement` | 2.325 |
| `zddPqpp1a06xttKdc0iNTA_farend_singletalk_with_movement` | 2.353 |
| `0luXwWjGEEC9G5nf0yTVXw_farend_singletalk_with_movement` | 2.494 |
| `wlAXM0iDgkm06i7UdRww1w_farend_singletalk_with_movement` | 2.526 |
| `Xv7jH2KcBEWqdpbT000HQA_farend_singletalk_with_movement` | 2.547 |
| `LeV1uF4j10Whm0FPG80tmw_farend_singletalk_with_movement` | 2.564 |
| `5bJUo1K3uEmMrGa9UhGyVg_farend_singletalk_with_movement` | 2.626 |
| `yxKXWLezBUeCpdYkIQOT0A_farend_singletalk_with_movement` | 2.637 |
| `sRCs6SKo6kC0xire475q0A_farend_singletalk_with_movement` | 2.659 |
| `TGZ5Wq0SCUCOXPsfee3uMQ_farend_singletalk_with_movement` | 2.670 |
| `hVqUmGvIlkO0LBUoE06Q3w_farend_singletalk_with_movement` | 2.711 |
| `kz23X4pDSEiPmWtw2Qx00Q_farend_singletalk_with_movement` | 2.720 |
| `wr54weKzNkOcZ07hB04kzA_farend_singletalk_with_movement` | 2.747 |

### DT_static (sorted by deg ascending)

| stem | deg |
|---|---:|
| `QK70KpLuZ0O43BBSWEZvHg_doubletalk` | 1.267 |
| `sYQK1rJlwU2XCy20n0Sx9g_doubletalk` | 1.281 |
| `SgKY30fjT0G8e3kQL0RHSQ_doubletalk` | 1.328 |
| `xYuPW7feGkyc8a1rfcDv9w_doubletalk` | 1.334 |
| `QkRkwwFKVEar0WtcuvJsZg_doubletalk` | 1.338 |
| `XV5L2dn3S06M9GBEu1q3DA_doubletalk` | 1.367 |
| `Uc4dmejgWUCTvn0XZbMTBw_doubletalk` | 1.503 |
| `wY3Rp9YEjkOke09hMwfsjg_doubletalk` | 1.558 |
| `wLRC9k23Bk6kmBzZdbkHTA_doubletalk` | 1.560 |
| `Pu8CtSffMUiINQAhSKvlfw_doubletalk` | 1.575 |
| `pPBidi8oyUarZCGUrKJEsg_doubletalk` | 1.581 |
| `XnfMDZLl0U2WvLRphiGJ6A_doubletalk` | 1.598 |
| `yc5bFUGsR0GSfiGwTTpRWg_doubletalk` | 1.602 |
| `X0OraQOKtkCVriR0uj0WBQ_doubletalk` | 1.620 |
| `kGFOHthwrUWqHCLkBYIQnA_doubletalk` | 1.643 |
| `WtQs4a0YeU2B0dQWhS7gmg_doubletalk` | 1.667 |
| `NY3kZioAm0KwR45wIVe2Sg_doubletalk` | 1.669 |
| `xWOALEtAwk2oABUz3vAv6w_doubletalk` | 1.669 |
| `XtUWcfsuykC5fTu7DdAnnw_doubletalk` | 1.685 |
| `S5cyhx02u00eJzrHTxVTwQ_doubletalk` | 1.711 |

### DT_movement (sorted by deg ascending)

| stem | deg |
|---|---:|
| `wHmBm7VHfkysBOhjoAXkNA_doubletalk_with_movement` | 1.225 |
| `XV5L2dn3S06M9GBEu1q3DA_doubletalk_with_movement` | 1.337 |
| `7GTxyTksSUqCnP5y0ILG4A_doubletalk_with_movement` | 1.446 |
| `xofDX004bkqiOv9YOxmGVQ_doubletalk_with_movement` | 1.475 |
| `I2bme08keUmAnyJRKNYDGQ_doubletalk_with_movement` | 1.492 |
| `VgSXlJJEI02dytkMm5UTzA_doubletalk_with_movement` | 1.526 |
| `yc5bFUGsR0GSfiGwTTpRWg_doubletalk_with_movement` | 1.572 |
| `WcK0OrF6ukW03fViPXTQjQ_doubletalk_with_movement` | 1.580 |
| `N2rQLbnp2UOg2QFRaggbDw_doubletalk_with_movement` | 1.617 |
| `uLl640xveUuHp2kEtOCTeQ_doubletalk_with_movement` | 1.647 |
| `X7Ua9txMj0aws848JPEbOg_doubletalk_with_movement` | 1.701 |
| `xvACDxradUuKNYImFSd1ww_doubletalk_with_movement` | 1.702 |
| `nyT6FUUdu0W8UpvjP1rRgQ_doubletalk_with_movement` | 1.727 |
| `WnDjVFWmC0m0WhVq22mRlQ_doubletalk_with_movement` | 1.759 |
| `w0QrMwsZ5kGoJjRWvP0iKg_doubletalk_with_movement` | 1.790 |
| `VNkNShj97UajHDVbSmIG0g_doubletalk_with_movement` | 1.817 |
| `sKXucFp4FUCJKo5d0G54Og_doubletalk_with_movement` | 1.825 |
| `w5XDRNfB2Ei2UoUDtrTkzg_doubletalk_with_movement` | 1.825 |
| `ql7yTcebJU20VE5qpW0kCA_doubletalk_with_movement` | 1.833 |
| `zOiK6oSHp0ib3nHvzLKbRQ_doubletalk_with_movement` | 1.852 |

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
