[C 使用手冊](../c_user_manual_zh_TW.md)

## 附錄 A：`aec_wav` 命令列工具

`aec_wav` 是**示範與驗證工具**，不是交付的函式庫介面。它的旗標與 C API 沒有一對一
關係，正式整合請用 [§2](../c_user_manual_zh_TW.md#2-quick-start) – [§8](integration.md#8-進階整合-seam) 的 API。

```bash
make -C c_impl                                     # 建置（產出 aec_wav）
BIN="$(make -s -C c_impl print-bin-dir)"

"$BIN"/aec_wav <mic.wav> <ref.wav> <out.wav> [options]
```

完整旗標（取自程式的參數解析，無其他旗標）：

| 旗標 | 說明 |
|---|---|
| `--level {0\|1\|2\|3}` | 預設 `1`（balanced）；`0` = mild、`1` = balanced、`2` = aggressive、`3` = no_floor（即 `AecLevel` 的數值）。只收整串十進位數字，名稱、空字串、`1x` 一律以結束碼 2 拒絕；`--preset` 仍可作為別名 |
| `--cng` | 開啟舒適噪音 |
| `--no-cng` | 關閉舒適噪音 |
| `--no-res` | 關閉殘留回音抑制 |
| `--fft-size <n>` | 覆寫 FFT 大小：16 kHz 可用 256/512，48 kHz 為 1024 |
| `--delay-mode {matched\|fixed\|external}` | 覆寫 `cfg.delay_mode`（預設 `matched`）；`external` 對應 C API 的 `AEC_DELAY_EXTERNAL_ALIGNED` |
| `--delay-num-filters <n>` | 覆寫 `cfg.delay_num_filters`（僅 `matched` 合法，1–5，預設 5） |
| `--fixed-delay <samples>` | 覆寫 `cfg.fixed_delay_samples`（僅 `fixed` 合法，native-rate sample，≥0） |
| `--print-mem-size` | 印出 resolved grid／mode／n 與 static-pool byte 拆解後**直接退出**：仍會讀兩個輸入 WAV 的取樣率，但不寫輸出檔、不跑任何一個 hop（見下方「`--print-mem-size` 輸出」） |
| `--debug` | 每秒印一行 `aec_debug_status()` 狀態 |

> 兩個容易誤解的地方：**沒有** `--filter-length` 或 `--filter-length-ms` 這個旗標
> （濾波器長度由取樣率決定，不是可設定的欄位；`--no-shadow`、`--no-hpf`、`--delay-est-period` 也已移除，
> 帶它們會以結束碼 2 拒絕）；`--cng` 不是「開啟一個預設關閉的功能」，
> 舒適噪音**預設就是開的**，`--cng` 與 `--no-cng` 只是明確覆寫。

> `--delay-mode`／`--delay-num-filters`／`--fixed-delay` 三個旗標的值**未經
> CLI 驗證就直接寫入 `AecConfig`**——`aec_resolve()` 是唯一的範圍與
> mode／欄位相容性權威（[§6](configuration.md#6-aecconfig-完整參考)），CLI 不做第二次檢查也不做 clamp。組合非法
> 時（例如 `--fixed-delay` 給負值、`--delay-mode fixed` 沒給 `--fixed-delay`、
> `--delay-mode external` 卻改了 `--delay-num-filters`）`aec_create()` 會
> fail-fast 印出訊息後以離開碼 4 結束，不會被靜默忽略或改成別的設定。

### `--print-mem-size` 輸出

不寫輸出檔、不 `aec_create()` 一個真正的實例，只呼叫 `aec_get_mem_breakdown()`
（[§6.7](configuration.md#67-靜態記憶體拆解aec_get_mem_breakdown)）後印一行到 **stdout**（其餘旗標一律印到 stderr）：

```text
mem: sr=16000 fft=256 hop=128 mode=matched n=5 fixed_delay_samples=-1 total_bytes=239760 estimator_bytes=29248 ring_bytes=33104
```

欄位依序：resolved 後的 `sample_rate`／`fft_size`／`hop_size`、`delay_mode`、
`delay_num_filters`、`fixed_delay_samples`、`aec_get_mem_size()` 回傳的總
byte 數、`delay_aec_get_mem_size()` 回傳的 estimator byte 數（非 `matched`
時為 0）、對齊環形緩衝的 byte 數（`external` 時為 0）。三個 byte 欄位與 [§4](lifecycle.md#4-生命週期)
表格是同一個 `aec_get_mem_breakdown()` 呼叫算出來的，數字對不上就是文件
過期，以實際輸出為準。組合非法時走與 `aec_create()` 相同的 fail-fast 路徑
（離開碼 4），不會印出任何 `mem:` 行。

```bash
"$BIN"/aec_wav mic.wav ref.wav /dev/null --print-mem-size --delay-mode fixed --fixed-delay 1600
```

### Duty census 診斷輸出

正常處理完（非 `--print-mem-size`）結尾，`aec_wav` 會在既有的 `Processed ...`
摘要行之後多印一行 matched-filter duty-cycle **實測**（非估計）engagement
（`delay_selection.md` 的選型量測要用；見 [§8.4](integration.md#84-duty-cycle-engagement-census)）：

```text
duty: matched_filter_ran=250/250 hops (100.00% engagement, 0.00% saved)
```

上面是 16 kHz／256、`matched`、2 s 輸入的實測輸出。duty-cycle 降頻的週期是函式庫內部值（預設 0 = 每個 hop 全速分析），
`aec_wav` 沒有旗標能改它，所以這一行在預設組態下恆為 100%；降頻只在測試用的內部鉤子下出現。

`delay_mode != matched` 時（`fixed`／`external`）從未建過 matched filter，沒有成本可省，印：

```text
duty: matched_filter_ran=0/0 hops (n/a -- no matched-filter estimator in delay_mode=fixed)
```

診斷用途，不影響音訊路徑（見 [§8.4](integration.md#84-duty-cycle-engagement-census)）。

輸入 / 輸出：

- 輸入只接受 **PCM16** 或 **IEEE float32** WAV。PCM24／PCM32 會被拒絕。
- 多聲道輸入只取**第一個聲道**。
- mic/ref 取樣率必須相同，處理長度取兩者較短者；不足一個 hop 的尾端不處理。
- 輸出預設為 float32 WAV。設 `AEC_OUT_FLOAT=0` 可改輸出 PCM16。

離開碼：

| 碼 | 意義 |
|---:|---|
| 0 | 成功 |
| 1 | 參數不足（印出用法） |
| 2 | 未知的 level（不是 0–3 的整串數字）或未知的選項 |
| 3 | 無法開啟輸入檔或無法寫入輸出檔 |
| 4 | 取樣率不支援、mic/ref 取樣率不一致，或訊號格點無效 |

---

## 回報問題

請附上：mic/ref 測試片段、完整的呼叫方式（CLI 命令或 `AecConfig` 設定）、
sample rate 與 level、使用的 BACKEND，以及加 `--debug` 跑 `aec_wav` 的輸出
（每秒一行 `aec_debug_status()`）。
