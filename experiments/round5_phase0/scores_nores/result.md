# Bench result — current

Dataset: /Users/mingyu/Desktop/novatek/SE/AEC/wav/aec_challenge_blind
Output dir: /tmp/r5_nores_view
Cases scored: 800

## Bucket means

| Bucket | n | echo (↑) | deg (↑) |
|---|---:|---:|---:|
| FS_static | 169 | 2.620 | 5.000 |
| FS_movement | 131 | 2.616 | 5.000 |
| DT_static | 186 | 3.164 | 3.306 |
| DT_movement | 114 | 3.153 | 3.283 |
| NE | 200 | 4.998 | 4.120 |

## Δ vs baseline (baseline_v381_seeded)

| Bucket | Δecho | Δdeg | verdict |
|---|---:|---:|---|
| FS_static | -1.148 | +0.001 | FS echo regress |
| FS_movement | -1.211 | +0.001 | FS echo regress |
| DT_static | -1.087 | +1.049 | ok |
| DT_movement | -0.967 | +0.997 | ok |
| NE | -0.000 | +0.113 | ok |

## Worst-20 per bucket

### FS_static (sorted by echo ascending)

| stem | echo |
|---|---:|
| `XXz0qkUSd0GT4dsywxpfJg_farend_singletalk` | 1.441 |
| `JtodX3Ug6Eu5TYu0HN5IOw_farend_singletalk` | 1.450 |
| `9xjhiFbGo06hdQIsHTS6qA_farend_singletalk` | 1.484 |
| `WTdBhXa080WJEeGDde9BGA_farend_singletalk` | 1.490 |
| `7GTxyTksSUqCnP5y0ILG4A_farend_singletalk` | 1.548 |
| `khqZY41lNEyIvMf2ZNJuVA_farend_singletalk` | 1.561 |
| `s90M7MOTBkqaV4nQPLhKbA_farend_singletalk` | 1.570 |
| `TcuegeGKL06ysmFP0RxjyQ_farend_singletalk` | 1.574 |
| `S22FCqKDWUyymN1YbpItIw_farend_singletalk` | 1.583 |
| `iOyPaxX11UOaUkcscKhq1A_farend_singletalk` | 1.597 |
| `Gsy0lC5QSUi540hiax9XtA_farend_singletalk` | 1.601 |
| `sUQrHEPAoEmIvHclpi1tRQ_farend_singletalk` | 1.608 |
| `XTqo1aOXDEiqyWTFK99I5Q_farend_singletalk` | 1.609 |
| `XoP0KjC4G0W1SCRtg9l0LQ_farend_singletalk` | 1.612 |
| `ksP3OuSnpUa9Si2ttiUSoA_farend_singletalk` | 1.628 |
| `pcb1Nh0Z3k0WS9a7gBEuqg_farend_singletalk` | 1.680 |
| `TZ6TJFCbfkKAVrS64Sf08Q_farend_singletalk` | 1.690 |
| `QYHsugUAcUWEQ9WghnG0Jw_farend_singletalk` | 1.717 |
| `veoTpvS3mkaNkmCI6iEMVA_farend_singletalk` | 1.718 |
| `IrQvqOTCmEWMXn9k2ICtRQ_farend_singletalk` | 1.721 |

### FS_movement (sorted by echo ascending)

| stem | echo |
|---|---:|
| `WH0jN3PY40es2S0LsxmkkQ_farend_singletalk_with_movement` | 1.420 |
| `XTqo1aOXDEiqyWTFK99I5Q_farend_singletalk_with_movement` | 1.435 |
| `xFk7igecuke0R5JMfREyDg_farend_singletalk_with_movement` | 1.449 |
| `0luXwWjGEEC9G5nf0yTVXw_farend_singletalk_with_movement` | 1.470 |
| `TGZ5Wq0SCUCOXPsfee3uMQ_farend_singletalk_with_movement` | 1.486 |
| `ZWq0X5sPiUe0lQjZdCPSeQ_farend_singletalk_with_movement` | 1.550 |
| `VgSXlJJEI02dytkMm5UTzA_farend_singletalk_with_movement` | 1.589 |
| `XXz0qkUSd0GT4dsywxpfJg_farend_singletalk_with_movement` | 1.602 |
| `5bJUo1K3uEmMrGa9UhGyVg_farend_singletalk_with_movement` | 1.615 |
| `LeV1uF4j10Whm0FPG80tmw_farend_singletalk_with_movement` | 1.636 |
| `nV9v63E5CUKtKTjha8dtdQ_farend_singletalk_with_movement` | 1.645 |
| `XuguA1uJAE0bWT0xXRDdeA_farend_singletalk_with_movement` | 1.651 |
| `Fi80N5kW9U6nwaoS04O3vQ_farend_singletalk_with_movement` | 1.685 |
| `zddPqpp1a06xttKdc0iNTA_farend_singletalk_with_movement` | 1.691 |
| `oQK3bVihI0qel9As840Zzw_farend_singletalk_with_movement` | 1.706 |
| `VNgRsWxMdkaUx1gKV9W1Zw_farend_singletalk_with_movement` | 1.707 |
| `kOGPX6kHskOaKSZdLGNz8A_farend_singletalk_with_movement` | 1.708 |
| `yxKXWLezBUeCpdYkIQOT0A_farend_singletalk_with_movement` | 1.784 |
| `ql7yTcebJU20VE5qpW0kCA_farend_singletalk_with_movement` | 1.785 |
| `hVqUmGvIlkO0LBUoE06Q3w_farend_singletalk_with_movement` | 1.802 |

### DT_static (sorted by deg ascending)

| stem | deg |
|---|---:|
| `SgKY30fjT0G8e3kQL0RHSQ_doubletalk` | 1.223 |
| `p0mhFbhV6kGJgjd0RTTIIw_doubletalk` | 1.306 |
| `wY3Rp9YEjkOke09hMwfsjg_doubletalk` | 1.776 |
| `I2bme08keUmAnyJRKNYDGQ_doubletalk` | 1.859 |
| `UmlD9X38NECNoJKm0oyf4w_doubletalk` | 1.918 |
| `Pu8CtSffMUiINQAhSKvlfw_doubletalk` | 2.177 |
| `pPBidi8oyUarZCGUrKJEsg_doubletalk` | 2.201 |
| `OjdIdZgJDk6hLAQL07KORA_doubletalk` | 2.261 |
| `QK70KpLuZ0O43BBSWEZvHg_doubletalk` | 2.302 |
| `yZs0i8NpJkypsV8QyvduzQ_doubletalk` | 2.387 |
| `kg9YJVP17k2YTFuPQTOsdA_doubletalk` | 2.410 |
| `vSZmpMJI0kKv30P2GhgV1Q_doubletalk` | 2.413 |
| `QkRkwwFKVEar0WtcuvJsZg_doubletalk` | 2.415 |
| `Je6gJ7y1PECStwxnrOe9aA_doubletalk` | 2.423 |
| `xb7eJJF0Vki6Yl3y4B7oJA_doubletalk` | 2.471 |
| `XuiheB7eUkyJA2XzFIovHQ_doubletalk` | 2.569 |
| `nVUnxqHLr0GTN7shWid1Ow_doubletalk` | 2.584 |
| `0I0XMl3M0ECO0U1N0cJvpg_doubletalk` | 2.603 |
| `MeQ3WL4hykKuT2761h0xFg_doubletalk` | 2.604 |
| `pU21kfoo0UOz0fPMJFfydg_doubletalk` | 2.606 |

### DT_movement (sorted by deg ascending)

| stem | deg |
|---|---:|
| `WqEYNwalSUebZxaeYVay2g_doubletalk_with_movement` | 1.494 |
| `I2bme08keUmAnyJRKNYDGQ_doubletalk_with_movement` | 1.557 |
| `u0X5XB2KzEGduXtfWfjGDw_doubletalk_with_movement` | 1.726 |
| `XV5L2dn3S06M9GBEu1q3DA_doubletalk_with_movement` | 1.815 |
| `wHmBm7VHfkysBOhjoAXkNA_doubletalk_with_movement` | 2.059 |
| `zOiK6oSHp0ib3nHvzLKbRQ_doubletalk_with_movement` | 2.077 |
| `w5XDRNfB2Ei2UoUDtrTkzg_doubletalk_with_movement` | 2.099 |
| `VgSXlJJEI02dytkMm5UTzA_doubletalk_with_movement` | 2.227 |
| `V0JqgjlrB0Ke9y91r0rxNw_doubletalk_with_movement` | 2.267 |
| `iyuYIcszXku7BWYOOwqh5Q_doubletalk_with_movement` | 2.291 |
| `kg9YJVP17k2YTFuPQTOsdA_doubletalk_with_movement` | 2.449 |
| `nlSSRl4k50Gq2mIRYlMBCg_doubletalk_with_movement` | 2.491 |
| `QkRkwwFKVEar0WtcuvJsZg_doubletalk_with_movement` | 2.491 |
| `waxU019BVEacr7vK6v00mQ_doubletalk_with_movement` | 2.597 |
| `sRCs6SKo6kC0xire475q0A_doubletalk_with_movement` | 2.614 |
| `xSh5yXWiP02K0UkYdkZ0cA_doubletalk_with_movement` | 2.648 |
| `W0J6iZv7ZkmHOobCToob4A_doubletalk_with_movement` | 2.657 |
| `49IIo03GZ0CYQOmeA3A0BA_doubletalk_with_movement` | 2.689 |
| `TGZ5Wq0SCUCOXPsfee3uMQ_doubletalk_with_movement` | 2.726 |
| `WH0jN3PY40es2S0LsxmkkQ_doubletalk_with_movement` | 2.762 |

### NE (sorted by deg ascending)

| stem | deg |
|---|---:|
| `kz23X4pDSEiPmWtw2Qx00Q_nearend_singletalk` | 3.062 |
| `SR68lGQwTUy508j0P8BKZQ_nearend_singletalk` | 3.091 |
| `mAJHXrZ5QU62Do1OqTSL6Q_nearend_singletalk` | 3.327 |
| `4pN9yn7mhEa5iDiKnr5jlw_nearend_singletalk` | 3.398 |
| `2f9FDnH8u0q829njqEuVrQ_nearend_singletalk` | 3.407 |
| `DW3NNHoKl02cAfEkahzsGg_nearend_singletalk` | 3.463 |
| `yN0NYysKnUW1PsMYdU4tQA_nearend_singletalk` | 3.480 |
| `pPBidi8oyUarZCGUrKJEsg_nearend_singletalk` | 3.490 |
| `08Mo9p6KVUapuF8PNAMWaw_nearend_singletalk` | 3.514 |
| `toGZLON0MUWE7tOHhRJIOA_nearend_singletalk` | 3.538 |
| `IvAFDPUFEk0GuW2VKRHznA_nearend_singletalk` | 3.563 |
| `x7Frsl5wTEKmog3A6lL57g_nearend_singletalk` | 3.590 |
| `SFvlSygv4ke9wCrv8LWvYQ_nearend_singletalk` | 3.617 |
| `nVUnxqHLr0GTN7shWid1Ow_nearend_singletalk` | 3.629 |
| `A50fBOz02kag8CVAhBOF8A_nearend_singletalk` | 3.651 |
| `urm5FZsuoEGEayow6ckb0w_nearend_singletalk` | 3.654 |
| `UmlD9X38NECNoJKm0oyf4w_nearend_singletalk` | 3.668 |
| `jBqDg8bmpES0bGBYbsbtIw_nearend_singletalk` | 3.671 |
| `m6x1OA5vH0mUhq1QZELg8g_nearend_singletalk` | 3.693 |
| `JteZUZ4JYkeD4k2rcVbqHg_nearend_singletalk` | 3.695 |
