[C 使用手冊](../c_user_manual_zh_TW.md)

## 4. 生命週期

兩條路徑二選一。輸出**逐位元相同**（同一 BACKEND 下；`make test-static-aec` 驗證）。

### A. Heap 路徑（`aec_create`）

```c
AecConfig cfg;
Aec aec;                                    /* Aec 結構由呼叫端持有 */

aec_config_from_level(&cfg, AEC_LEVEL_BALANCED, 16000);
if (aec_create(&aec, &cfg) != 0) { /* 設定無效或記憶體不足 */ }

/* ... 每個 hop 呼叫 aec_process(&aec, mic, ref, out) ... */

aec_destroy(&aec);                          /* 必須剛好一次，釋放內部 arena */
```

`aec_create()` 內部只做**一次** 16-byte 對齊的 heap 配置（單一 arena，ARM32／ARM64 相同）。之後 `aec_process()` 與
`aec_reset()` 都不會再配置記憶體、不會做 I/O。

### B. 呼叫端自備記憶池（`aec_get_mem_size` + `aec_init`）

沒有 heap 的嵌入式目標用這條。整個實例（含 `Aec` 結構本身）都放進呼叫端的記憶池：

```c
AecConfig cfg;
aec_config_from_level(&cfg, AEC_LEVEL_BALANCED, 16000);

size_t need = aec_get_mem_size(&cfg);       /* 0 = 設定無效 */
if (need == 0) { /* 設定無效 */ }

void *pool = NULL;
if (posix_memalign(&pool, 16, need) != 0) { /* 配置失敗 */ }

Aec *aec = aec_init(pool, need, &cfg);      /* 注意：回傳 Aec*，不是 out-param */
if (!aec) { /* pool 太小、未對齊 16 bytes，或設定無效 */ }

/* ... aec_process(aec, mic, ref, out) — aec 已經是指標，不需要 & ... */

aec_destroy(aec);                           /* 記憶池實例：不釋放記憶體（只把 handle 標成已銷毀），可省略 */
free(pool);                                 /* 記憶池由呼叫端自行回收 */
```

重點：

- **記憶池基底位址必須 16-byte 對齊**，否則 `aec_init()` 回傳 `NULL`。
- `mem_size` 小於 `aec_get_mem_size()` 一個 byte 就會被拒絕。
- 這條路徑**完全不呼叫 malloc**。`aec_destroy()` 對記憶池實例不釋放任何記憶體（只把 handle 標成已銷毀），
  重複呼叫安全，也可以直接省略後回收整個 pool。

### C. 記憶池描述子與陣列實例（`aec_get_mem_requirements` + `aec_init_ex`）

`aec_get_mem_requirements(&cfg, &req)` 對有效的設定（單聲道或陣列）回 0 並填入 32 byte 的 `AecMemReq`（`bytes` 等於
`aec_get_mem_size()`，另有描述子／佈局版本、backend、建置旗標雜湊與對齊）；設定無效回 -1 並把 `req` 清零。
`aec_init_ex(mem, size, &cfg, &req)` 帶非 NULL 的描述子時，要求它與目前建置的描述子完全一致（`reserved` 為 0、`req.bytes` 與
`size` 都不小於目前需要的 byte 數），用來擋掉「舊建置查到的記憶池配給新函式庫」。`aec_init(mem, size, &cfg)` 就是描述子為 `NULL` 的 `aec_init_ex()`。
描述子不可存檔或自行拼湊，每次建置重新查。

`AecConfig.n_mics = AEC_ARRAY_MICS`（4）建出陣列實例：四條 lane、共用的 delay 估計器與對齊後 far、融合緩衝都在**同一塊**記憶池，
一次 init 完成。呼叫序列與行為見 [§8.8](integration.md#88-麥克風陣列實例n_mics--4)；建立如下：

```c
AecConfig cfg;
aec_config_from_level(&cfg, AEC_LEVEL_BALANCED, 16000);
cfg.n_mics = AEC_ARRAY_MICS;               /* 4：陣列實例；residual 預設 FULL */
cfg.capture_proxy_channel = 0;             /* 共用 delay 估計器吃哪一路 mic */

AecMemReq req;                             /* 描述子：本次建置需要的記憶池 */
if (aec_get_mem_requirements(&cfg, &req) != 0) { /* cfg 無效 */ }

void *pool = NULL;
if (posix_memalign(&pool, 16, (size_t)req.bytes) != 0) { /* 配置失敗 */ }

Aec *aec = aec_init_ex(pool, (size_t)req.bytes, &cfg, &req);
if (!aec) { /* 記憶池太小／未對齊，或描述子與目前建置不符 */ }
```

收尾與單聲道相同（`aec_destroy(aec)`，記憶池由呼叫端回收；heap 路徑用 `aec_create()` 並剛好 `aec_destroy()` 一次）。

### 記憶池大小（實測，balanced level，其餘為預設值）

**記憶池大小會隨 `delay_mode` 與 `delay_num_filters` 改變**，因為 matched
filter bank、降取樣 render ring 與兩個 lag histogram 都是依 init 設定從記憶池
切出來的。下表是 NE10 backend 的實測值：

| 格點 | n=1 | n=2 | n=3 | n=4 | n=5（預設） |
|---|---:|---:|---:|---:|---:|
| 8 kHz / 256（legacy） | 226,944 B | 232,672 B | 238,400 B | 244,128 B | 249,856 B |
| 16 kHz / 256 | 320,848 B | 326,576 B | 332,304 B | 338,032 B | 343,760 B |
| 16 kHz / 512 | 433,728 B | 439,456 B | 445,184 B | 450,912 B | 456,640 B |
| 48 kHz / 1024 | 1,055,648 B | 1,061,376 B | 1,067,104 B | 1,072,832 B | 1,078,560 B |

> **這些絕對值會隨 `Aec` / `AecConfig` 的結構大小整批移動（`sizeof(AecConfig)`
> 目前是 228 B），而所有 delta 不變**（每個 filter 5,728 B、`FIXED` 的 byte/ms
> 公式、mode 之間的差值）。所以不能沿用記下的常數，**必須重新查詢
> `aec_get_mem_size()` 並重新 init**。
>
> 本節的絕對值由 `make test-delay-num-filters` 與 `aec_wav --print-mem-size` 量得；
> §4 的表格與[附錄 A](wav_cli.md#附錄-aaec_wav-命令列工具) `--print-mem-size` 的範例輸出都等於同一次量測的結果。

每少一個 matched filter 固定省 **5,728 B**（每一格點都一樣）：coefficients
2,048 B ＋ accumulated error 512 B ＋ render ring 1,536 B ＋ highest-peak
histogram 1,536 B ＋ pre-echo histogram 96 B。所以 `n=1` 相對預設 `n=5` 省
22,912 B。

非 `MATCHED` 模式不配置 estimator，而且**參考訊號環形緩衝的大小也依 mode
不同**（16 kHz/256）：

| `delay_mode` | 記憶池 | 相對 `MATCHED n=5` |
|---|---:|---:|
| `MATCHED` n=5（預設） | 343,760 B | — |
| `MATCHED` n=1 | 320,848 B | −22,912 B |
| `FIXED`，`fixed_delay_samples = 0` | 179,264 B | −164,496 B |
| `FIXED`，`fixed_delay_samples = 400`（25 ms） | 180,864 B | −162,896 B |
| `FIXED`，`fixed_delay_samples = 1600`（100 ms） | 185,664 B | −158,096 B |
| `FIXED`，`fixed_delay_samples = 8000`（500 ms） | 211,264 B | −132,496 B |
| `EXTERNAL_ALIGNED` | 178,752 B | −165,008 B |

#### 環形緩衝大小公式

環形緩衝的容量由**唯一一個** helper 決定（C：`aec.c` 的
`aec_ref_ring_samples()`；Python：`modules/config.py` 的
`ref_ring_samples()`），`aec_get_mem_size()` 與 init 的 pool carve 共用同一個
呼叫，不可能對不上：

| `delay_mode` | 環形緩衝容量（sample） |
|---|---|
| `MATCHED` | `max(delay_buffer_ms, max_delay_ms + 4096)` 換算成 sample |
| `FIXED` | `fixed_delay_samples + hop_size` |
| `EXTERNAL_ALIGNED` | 0（完全不配置） |

`FIXED` 的公式是**緊上界**，不是估算：設 `T` 為目前已寫入的總 sample 數，
環形緩衝持有 `[T-rs, T)`；每個 hop 先寫入最新的 `hop` 個 sample（`T` 前進），
再讀取 `[T-d-hop, T-d)` 來做延遲補償，因此正確性條件就是 `rs >= d + hop`。
`FIXED` 的 `d` 是不可變的（沒有 estimator，`aec_reset()` 也是重新填入同一個
值），所以這是「對一個常數取緊上界」，不是「對會變動的量取最壞情況」——
沒有任何 headroom 項有意義，`max_delay_ms` / `delay_buffer_ms` 在這個 mode
下完全不生效（改它們不會改變記憶池大小）。取等號是安全的：`rs == d + hop`
時讀取起點正好落在寫入游標上，也就是這個 hop 的寫入唯一沒有覆蓋到的那一段。

`fixed_delay_samples = 0` 是退化情形：仍配置一條 hop 長度、只寫不讀的緩衝
（偏移為 0 時根本不需要讀），這樣 `FIXED` 只會有一條環形緩衝程式路徑。
`EXTERNAL_ALIGNED` 與 `FIXED(0)` 的記憶池差距因此正好是一條 hop 長度的
陣列（16 kHz/256 為 512 B）。

換算成 byte：每個 sample 4 B，再對齊到 16 B。以 16 kHz/256（hop 128）為例，
每多 1 ms 的 `fixed_delay_samples` 就多 64 B；48 kHz/1024 每多 1 ms 多 192 B。

各格點的 `FIXED` 實測值（balanced level）：

| 格點 | fixed=0 | 25 ms | 100 ms | 500 ms |
|---|---:|---:|---:|---:|
| 16 kHz / 256 | 179,264 B | 180,864 B | 185,664 B | 211,264 B |
| 16 kHz / 512 | 292,656 B | 294,256 B | 299,056 B | 324,656 B |
| 48 kHz / 1024 | 652,080 B | 656,880 B | 671,280 B | 748,080 B |

> **注意**：`fixed_delay_samples` 很大時 `FIXED` 也可能比 `MATCHED n=5` 更
> 耗記憶體（48 kHz/1024、2500 ms 為 1,132,080 B，高於 `MATCHED n=5` 的
> 1,078,560 B）——環形緩衝終究要放得下那個延遲。要用大 fixed delay 又要省
> 記憶體，正確作法是讓呼叫端自己補償掉大延遲後改用 `EXTERNAL_ALIGNED`。

另外一個明顯的開關（16 kHz/256，相對 343,760 B）：

| 設定 | 記憶池 | 差異 |
|---|---:|---:|
| `enable_shadow = 0` | 314,848 B | −28,912 B |

單聲道實例的記憶池**不隨** `AecConfig.residual` 與 `enable_cng` 改變（相關緩衝區一律配置）；
**陣列實例會隨 `residual` 改變**（見下節）。

上表全是 64-bit 主機建置的實測值，僅供規劃參考。**實際請以
`aec_get_mem_size(&cfg)` 的回傳值為準**——它會隨設定、backend 與目標 ABI
改變。`fft_size`、`sample_rate`、`delay_mode`、`delay_num_filters` 任何一項
改變都必須重新查詢並重新 init（pool 佈局不同）。

### 陣列實例的記憶池

`n_mics = 4`，balanced level，其餘為預設值（`MATCHED`、`delay_num_filters = 5`），NE10 backend、64-bit 主機實測
（`aec_get_mem_size()` 與 `aec_get_mem_breakdown()`；`sizeof(Aec)` 6,368 B、`sizeof(AecConfig)` 228 B）：

| 格點 | `STATE` | `ESTIMATE` | `FULL`（預設） |
|---|---:|---:|---:|
| 16 kHz / 256 | 616,912 B | 712,448 B | 720,352 B |
| 16 kHz / 512 | 920,464 B | 1,110,208 B | 1,124,256 B |
| 48 kHz / 1024 | 2,062,960 B | 2,441,120 B | 2,470,528 B |

`aec_get_mem_breakdown()` 的欄位（`FULL`）：

| 格點 | `total_bytes` | `estimator_bytes` | `ring_bytes` | `lane_bytes` | `fusion_bytes` | `Aec` 標頭 |
|---|---:|---:|---:|---:|---:|---:|
| 16 kHz / 256 | 720,352 | 33,936 | 66,576 | 588,800 | 24,672 | 6,368 |
| 16 kHz / 512 | 1,124,256 | 33,936 | 67,600 | 971,968 | 44,384 | 6,368 |
| 48 kHz / 1024 | 2,470,528 | 35,312 | 200,720 | 2,141,248 | 86,880 | 6,368 |

- 陣列池 = `Aec` 標頭（`sizeof(Aec)` 對齊 16 B，上表最後一欄）+ `estimator_bytes` + `ring_bytes` + `lane_bytes` + `fusion_bytes`。
  **標頭不在任何一個拆解欄位內**，所以四個欄位加起來比 `total_bytes` 少一個標頭。
- `lane_bytes`：四條 lane 池合計，每條 147,200／242,992／535,312 B（`ESTIMATE`、`FULL`）；`STATE` lane 不切它提早離開後不會讀的殘餘回音估計器與舒適噪音陣列，每條 124,880／198,656／446,944 B。
  lane 由內部的 `aec_lane_init()` 建立，比同格點單聲道 `EXTERNAL_ALIGNED` 實例（178,752／291,632／650,032 B）少：借用陣列的同一個 FFT 控制代碼；
  省去 lane 契約永遠不跑的配置——delay-shift 快照（只有 `MATCHED` 估計器會用）、合成與增益緩衝（OLA、`eout`、SuppressionGain 陣列；只有 `FULL` 才套增益）。
  單聲道實例同樣沒有 render-history 列與 `x2_past`，穩態歷史列剛好是環形長度，shadow 濾波器是主濾波器的 follower（同輸入、同清除、同分區步進，
  直接共用主濾波器的遠端歷史，也沒有它不形成的 `near_spec`／`error_spec_windowed`／`spec_scratch`）。
- `fusion_bytes`：控制區塊、麥克風／對齊後 far／融合緩衝、四條 lane 共用的一個 FFT 控制代碼（7,152／13,552／26,352 B），`FULL` 另含共用的抑制增益階段（內部 `aec_res_get_mem_size()`：7,904／14,048／29,408 B）；
  **不含** lane 池。`STATE` 沒有融合緩衝（10,512／17,936／32,784 B）；`ESTIMATE` 有融合緩衝但沒有增益階段（16,768／30,336／57,472 B）。
- `estimator_bytes`／`ring_bytes` 與單聲道同義，但環形緩衝的容量依陣列的公式（[§8.8](integration.md#88-麥克風陣列實例n_mics--4)）：
  `MATCHED` 為 `ceil(max_delay_ms × sample_rate / 1000) + 2 × hop + 1` 個 sample，`FIXED` 為 `fixed_delay_samples + hop_size`，`EXTERNAL_ALIGNED` 為 0。
- 其他 `delay_mode`（`FULL`）：`EXTERNAL_ALIGNED` 為 619,840／1,022,720／2,234,496 B；`FIXED`、`fixed_delay_samples = 0` 為 620,352／1,023,744／2,236,544 B；
  `FIXED`、100 ms 為 626,752（1600 sample）／1,030,144（1600 sample）／2,255,744 B（4800 sample）。

**尚未做到**：遠端側狀態仍是每條 lane 各一份（共用的是 delay 估計器、對齊後 far、FFT 控制代碼，以及 hop 內由 lane 0 借給 lane 1–3 的遠端 FFT 結果）；
lane 約占 `FULL` 陣列池的 82%／86%／87%。
這些是主機的實測值，**實際請以 `aec_get_mem_requirements()` 為準**。

### `aec_reset`

```c
aec_reset(&aec);    /* 或 aec_reset(aec) */
```

清掉所有執行期狀態（延遲估計、filter、double-talk、收斂狀態），但**保留設定與記憶體
配置**。處理下一段**獨立**串流（例如批次處理下一個檔案）前一定要呼叫，否則前一段的
狀態會汙染下一段。

`aec_reset()` 是一次「硬 post 鏈 reset」：函式庫清掉自己的 OLA 尾巴、把舒適噪音種子設回 `AEC_CNG_SEED`（42），
並讓 `aec_post_reset_serial()` 加 1；`aec_reset()` 是獨立的一次呼叫，不在任何處理呼叫之內，它回傳時 serial 已經是 +1 後的值。自己在函式庫外合成的整合者靠這個 serial 同步清自己的 OLA 與 CNG 狀態（見
[§8.3](integration.md#83-aecresidualview-與-aeclinearview)），不要反過來呼叫 `aec_reset()`。

陣列實例的 `aec_reset()` 一次重設四條 lane、共用的 delay 估計器與對齊環、融合用的增益階段與舒適噪音產生器；`aec_post_reset_serial()` 是
四條 lane 計數之和，所以一次 `aec_reset()` 讓它前進 **4**（實測 0→4→8；單聲道為 1），請只比較「是否與自己存的不同」。

### 什麼不能改

`sample_rate`、`fft_size`、`filter_length`、`n_partitions`、`delay_mode`、
`delay_num_filters`、`fixed_delay_samples` 決定記憶體佈局，**建立後不可更改**。要換設定必須 destroy 後重新 create（記憶池路徑則是重新 `aec_init`）。
陣列實例另外還有 `n_mics`、`capture_proxy_channel`、`max_delay_ms`，以及 `residual`（陣列的池大小隨它改變）；
`delay_backward_quarantine_*` 在建立時換算成 hop 數，同樣不可在執行期更改。
也不要在處理中途直接改 `aec.cfg` 的內容。

---

## 5. 錯誤語意表

這個函式庫混用了數種錯誤回報慣例，請逐一對照，不要類推：

| 函式 | 成功 | 失敗 | 傳入 `NULL` handle |
|---|---|---|---|
| `aec_create` | 回傳 `0` | 回傳 `-1`（設定無效、記憶體不足） | 回傳 `-1`（安全） |
| `aec_get_mem_size` | 回傳位元組數（> 0） | 回傳 **`0`** | 回傳 `0`（安全） |
| `aec_init` | 回傳 `Aec*` | 回傳 **`NULL`**（設定無效、pool 太小、基底未 16-byte 對齊） | 回傳 `NULL`（安全） |
| `aec_destroy` | — | 不會失敗；可重複呼叫 | 直接返回（安全）。destroy 之後該 handle 被標成已銷毀：所有入口回 `AEC_ERR_ARG`／`-1`／`0`／NULL／清零的結構，直到再次 `aec_create`／`aec_init`（`aec_hop_size` 讀 `0`）。handle 或 pool 記憶體本身已被釋放或改寫則屬未定義 |
| `aec_reset` | — | 不會失敗 | 直接返回（安全）；已銷毀的 handle 也是 no-op |
| `aec_process`、`aec_process_linear`、`aec_process_residual` | 回傳 `AEC_OK`（0） | 負值 `AecStatus`：`AEC_ERR_ARG`（NULL 或引數不合法）、`AEC_ERR_MODE`（`residual` 不夠）、`AEC_ERR_SEQUENCE`（呼叫順序不對）；失敗時實例保持原狀、不寫任何輸出 | 回傳 `AEC_ERR_ARG`（安全） |
| `aec_process_linear_shared_far_from`（`aec_internal.h`） | 同上 | 同上；無效 owner 退回本地計算；對陣列實例回 `AEC_ERR_MODE` | 回傳 `AEC_ERR_ARG`（安全） |
| 陣列實例（`n_mics = 4`）的 `aec_process`、`aec_process_linear`、`aec_process_residual` | 回傳 `AEC_OK` | `aec_process`：`AEC_ERR_MODE`（陣列用 `aec_process_linear` + `aec_process_residual`）；`aec_process_residual`：`residual = STATE` 回 `AEC_ERR_MODE`，`FULL` 的同 hop 第二次回 `AEC_ERR_SEQUENCE`；輸入含非有限值回 `AEC_ERR_ARG`；`aec_process_residual` 的權重（NULL、非有限、全零）或 `beamformed_error`（含非有限值）不合法也回 `AEC_ERR_ARG`，實例完全不動、該 hop 保持開啟；某條 lane 或融合在引數檢查之後拒絕該 hop（例如有限的極大權重使投影溢位）回 **`AEC_ERR_DSP`** | 回傳 `AEC_ERR_ARG`（安全） |
| `aec_init_ex`、`aec_get_mem_requirements` | `aec_init_ex` 回傳 `Aec*`；`aec_get_mem_requirements` 回傳 `0` 並填入 `AecMemReq` | `aec_init_ex`：`NULL`（同 `aec_init`，另加描述子與目前建置不符）；`aec_get_mem_requirements`：`-1` 並把 `req` 清零 | 回傳 `NULL`／`-1`（安全） |
| `aec_get_array` | 回傳 `AEC_OK`，填入 `AecArrayView` | `AEC_ERR_ARG`（`struct_size` 太小或 `a` 為 NULL）；`AEC_ERR_SEQUENCE`（陣列：尚無 `aec_process_linear()`；單聲道：後級尚未跑） | 回傳 `AEC_ERR_ARG`（安全） |
| `aec_apply_external_realign` | `1`（warm）／`0`（filter 重置） | `-1`：NULL、`delay_mode` 不是 `EXTERNAL_ALIGNED`，或對象是陣列實例（它自己對四條 lane 做 realign） | 回傳 `-1`（安全） |
| `aec_hop_size` | 回傳 hop sample 數 | — | 回傳 `0`（安全；已銷毀的 handle 也是 `0`） |
| `aec_far_fft_real_compute_count` | 回傳累計次數 | — | **會 crash** |
| `aec_get_residual`、`aec_get_linear` | 回傳 `AEC_OK`，填入結構 | `AEC_ERR_ARG`（`struct_size` 太小：結構完全不被寫入）；`aec_get_residual` 另有 `AEC_ERR_MODE`（`residual` 為 `OFF`）、`AEC_ERR_SEQUENCE`（後級尚未跑） | 回傳 `AEC_ERR_ARG`，結構除 `struct_size` 外清零 |
| `aec_apply_gain`、`aec_apply_cng` | 回傳 `AEC_OK` | `AEC_ERR_MODE`（`residual` 不是 `FULL`）、`AEC_ERR_SEQUENCE`（該 hop 尚未處理完）、`AEC_ERR_ARG` | 回傳 `AEC_ERR_ARG`（安全） |
| `aec_debug_status` | 回傳 `void`，填入結構 | — | 把 `*out` 清零（`out` 為 `NULL` 則直接返回；已銷毀的 handle 同樣清零） |
| `aec_get_last_timing` | 無 | 把 `*out` 清零 | `a` 為 `NULL` 清零 `out`；`out` 為 `NULL` 直接返回（安全） |
| `aec_get_mem_breakdown` | 回傳 `AEC_OK`（0）並填入結構 | 回傳 `AEC_ERR_ARG`（`cfg` 無效）並把 `*out` 清零 | `cfg` 為 `NULL` 回 `AEC_ERR_ARG`；`out` 為 `NULL` 回 `AEC_ERR_ARG`（安全） |
| `aec_resolve_signal_grid` | 回傳 `AEC_OK`（0）並填入 `AecSignalGrid`（`out` 可為 `NULL`，只做檢查） | 回傳 `AEC_ERR_ARG`（沒有這組 `(sample_rate, fft_size)`，含 `fft_size = 0`），`*out` 不被寫入 | 不適用（傳入 int） |
| `aec_linear_is_cancelling` | 回傳 `1`（正在消除）或 `0` | — | 回傳 `0`（安全；`0` 代表「沒有證據顯示正在消除」，不代表 filter 壞掉） |
| `aec_is_valid_sample_rate` | 支援回傳 `1` | 不支援回傳 `0` | 不適用（傳入 int） |

必須知道的三件事：

1. **處理函式回傳 `AecStatus`，只報告呼叫本身的錯誤。** 回傳值是 `AEC_OK`（0）或負的錯誤碼
   （`AEC_ERR_ARG`、`AEC_ERR_MODE`、`AEC_ERR_SEQUENCE`；陣列實例另有 `AEC_ERR_DSP`），被拒絕的呼叫完全不動實例。
   陣列實例有兩個例外：`AEC_ERR_DSP` 表示某條 lane 或融合拒絕了該 hop，**整個實例已像 `aec_reset()` 一樣重設**；
   陣列的 `aec_process_linear()` 因輸入含非有限值回 `AEC_ERR_ARG` 時，實例完全不動，上一個 hop 的檢視仍可讀。
   `aec_get_mem_breakdown`、`aec_resolve_signal_grid` 也回 `AecStatus`（0 = 成功），不再是 `1` = 成功。
   設定問題在建立階段（`aec_create` / `aec_init`）擋掉；之後 pointer 有效、長度正確、串流同步仍是呼叫端的責任
   （`mic`／`ref` 的長度函式庫無從檢查）。
   線性輸出隨 `AecConfig.residual` 而不同：`OFF` 使用較簡單的步長規則，`STATE`／`ESTIMATE`／`FULL` 共用同一條線性路徑，所以要拿線性部分與完整鏈比對時，只能在 `STATE` 以上比。
2. **`aec_debug_status` 在 handle 為 `NULL`（或已銷毀）時把你傳入的結構清零。** 檢視函式（`aec_get_residual`、
   `aec_get_linear`）則回傳 `AEC_ERR_ARG`，並把結構除 `struct_size` 外清零。
3. **`aec_get_mem_size` 用 `0` 代表失敗**，不是回傳負值。務必檢查 `== 0`
   再拿去配置記憶體。

設定驗證在 `aec_create` / `aec_init` / `aec_get_mem_size` 三者內部都會先執行，
任一欄位超出 [§6](configuration.md#6-aecconfig-完整參考) 的允許範圍就整組拒絕。

---
