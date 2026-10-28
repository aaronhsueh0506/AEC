> [索引](c_user_manual_zh_TW.md) · 下一部分：[c_user_manual_zh_TW_02_config.md](c_user_manual_zh_TW_02_config.md) <!-- split-nav -->
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

aec_destroy(aec);                           /* 對記憶池實例是 no-op，可省略 */
free(pool);                                 /* 記憶池由呼叫端自行回收 */
```

重點：

- **記憶池基底位址必須 16-byte 對齊**，否則 `aec_init()` 回傳 `NULL`。
- `mem_size` 小於 `aec_get_mem_size()` 一個 byte 就會被拒絕。
- 這條路徑**完全不呼叫 malloc**。`aec_destroy()` 對記憶池實例是真正的 no-op，
  重複呼叫安全，也可以直接省略後回收整個 pool。

### 記憶池大小（實測，balanced level，其餘為預設值）

**記憶池大小會隨 `delay_mode` 與 `delay_num_filters` 改變**，因為 matched
filter bank、降取樣 render ring 與兩個 lag histogram 都是依 init 設定從記憶池
切出來的。下表是 NE10 backend 的實測值：

| 格點 | n=1 | n=2 | n=3 | n=4 | n=5（預設） |
|---|---:|---:|---:|---:|---:|
| 8 kHz / 256（legacy） | 246,080 B | 251,808 B | 257,536 B | 263,264 B | 268,992 B |
| 16 kHz / 256 | 343,088 B | 348,816 B | 354,544 B | 360,272 B | 366,000 B |
| 16 kHz / 512 | 471,808 B | 477,536 B | 483,264 B | 488,992 B | 494,720 B |
| 48 kHz / 1024 | 1,139,824 B | 1,145,552 B | 1,151,280 B | 1,157,008 B | 1,162,736 B |

> **這些絕對值會隨 `Aec` / `AecConfig` 的結構大小整批移動（`sizeof(AecConfig)`
> 目前是 224 B），而所有 delta 不變**（每個 filter 5,728 B、`FIXED` 的 byte/ms
> 公式、mode 之間的差值）。所以不能沿用記下的常數，**必須重新查詢
> `aec_get_mem_size()` 並重新 init**。
>
> 本節的絕對值由 `make test-delay-num-filters` 與 `aec_wav --print-mem-size` 量得；
> §4 的表格與[附錄 A](c_user_manual_zh_TW_05_appendix.md#附錄-aaec_wav-命令列工具) `--print-mem-size` 的範例輸出都等於同一次量測的結果。

每少一個 matched filter 固定省 **5,728 B**（每一格點都一樣）：coefficients
2,048 B ＋ accumulated error 512 B ＋ render ring 1,536 B ＋ highest-peak
histogram 1,536 B ＋ pre-echo histogram 96 B。所以 `n=1` 相對預設 `n=5` 省
22,912 B。

非 `MATCHED` 模式不配置 estimator，而且**參考訊號環形緩衝的大小也依 mode
不同**（16 kHz/256）：

| `delay_mode` | 記憶池 | 相對 `MATCHED n=5` |
|---|---:|---:|
| `MATCHED` n=5（預設） | 366,000 B | — |
| `MATCHED` n=1 | 343,088 B | −22,912 B |
| `FIXED`，`fixed_delay_samples = 0` | 201,504 B | −164,496 B |
| `FIXED`，`fixed_delay_samples = 400`（25 ms） | 203,104 B | −162,896 B |
| `FIXED`，`fixed_delay_samples = 1600`（100 ms） | 207,904 B | −158,096 B |
| `FIXED`，`fixed_delay_samples = 8000`（500 ms） | 233,504 B | −132,496 B |
| `EXTERNAL_ALIGNED` | 200,992 B | −165,008 B |

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
| 16 kHz / 256 | 201,504 B | 203,104 B | 207,904 B | 233,504 B |
| 16 kHz / 512 | 330,736 B | 332,336 B | 337,136 B | 362,736 B |
| 48 kHz / 1024 | 736,256 B | 741,056 B | 755,456 B | 832,256 B |

> **注意**：`fixed_delay_samples` 很大時 `FIXED` 也可能比 `MATCHED n=5` 更
> 耗記憶體（48 kHz/1024、2500 ms 為 1,216,256 B，高於 `MATCHED n=5` 的
> 1,162,736 B）——環形緩衝終究要放得下那個延遲。要用大 fixed delay 又要省
> 記憶體，正確作法是讓呼叫端自己補償掉大延遲後改用 `EXTERNAL_ALIGNED`。

另外一個明顯的開關（16 kHz/256，相對 366,000 B）：

| 設定 | 記憶池 | 差異 |
|---|---:|---:|
| `enable_shadow = 0` | 323,120 B | −42,880 B |

`enable_res` 與 `enable_cng` **不影響**記憶池大小（相關緩衝區一律配置）。

上表全是 64-bit 主機建置的實測值，僅供規劃參考。**實際請以
`aec_get_mem_size(&cfg)` 的回傳值為準**——它會隨設定、backend 與目標 ABI
改變。`fft_size`、`sample_rate`、`delay_mode`、`delay_num_filters` 任何一項
改變都必須重新查詢並重新 init（pool 佈局不同）。

### `aec_reset`

```c
aec_reset(&aec);    /* 或 aec_reset(aec) */
```

清掉所有執行期狀態（延遲估計、filter、double-talk、收斂狀態），但**保留設定與記憶體
配置**。處理下一段**獨立**串流（例如批次處理下一個檔案）前一定要呼叫，否則前一段的
狀態會汙染下一段。

`aec_reset()` 是一次「硬 post 鏈 reset」：函式庫清掉自己的 OLA 尾巴、把舒適噪音種子設回 `AEC_CNG_SEED`（42），
並讓 `aec_post_reset_serial()` 加 1；`aec_reset()` 是獨立的一次呼叫，不在任何處理呼叫之內，它回傳時 serial 已經是 +1 後的值。自己在函式庫外合成的整合者靠這個 serial 同步清自己的 OLA 與 CNG 狀態（見
[§8.3](c_user_manual_zh_TW_04_seam_troubleshooting.md#83-aecrescontext-欄位)），不要反過來呼叫 `aec_reset()`。

### 什麼不能改

`sample_rate`、`fft_size`、`filter_length`、`n_partitions`、`delay_mode`、
`delay_num_filters`、`fixed_delay_samples` 決定記憶體佈局，**建立後不可更改**。要換設定必須 destroy 後重新 create（記憶池路徑則是重新 `aec_init`）。
也不要在處理中途直接改 `aec.cfg` 的內容。

---

## 5. 錯誤語意表

這個函式庫混用了數種錯誤回報慣例，請逐一對照，不要類推：

| 函式 | 成功 | 失敗 | 傳入 `NULL` handle |
|---|---|---|---|
| `aec_create` | 回傳 `0` | 回傳 `-1`（設定無效、記憶體不足） | 回傳 `-1`（安全） |
| `aec_get_mem_size` | 回傳位元組數（> 0） | 回傳 **`0`** | 回傳 `0`（安全） |
| `aec_init` | 回傳 `Aec*` | 回傳 **`NULL`**（設定無效、pool 太小、基底未 16-byte 對齊） | 回傳 `NULL`（安全） |
| `aec_destroy` | — | 不會失敗 | 直接返回（安全） |
| `aec_reset` | — | 不會失敗 | **會 crash**（無 NULL 檢查） |
| `aec_process` | 回傳 `void` | **無法回報失敗** | **會 crash**（無 NULL 檢查） |
| `aec_process_context` | 回傳 `void` | 同上 | **會 crash** |
| `aec_process_context_shared_far_from` | 回傳 `void` | 無法回報失敗；無效 owner 退回本地計算 | 直接返回（安全） |
| `aec_hop_size` | 回傳 hop sample 數 | — | **會 crash** |
| `aec_far_fft_real_compute_count` | 回傳累計次數 | — | **會 crash** |
| `aec_get_res_context` | 回傳 `void`，填入結構 | — | **靜默 no-op**（你的結構完全不被寫入） |
| `aec_debug_status` | 回傳 `void`，填入結構 | — | **靜默 no-op**（你的結構完全不被寫入） |
| `aec_get_last_timing` | 無 | 把 `*out` 清零 | `a` 為 `NULL` 清零 `out`；`out` 為 `NULL` 直接返回（安全） |
| `aec_get_mem_breakdown` | 回傳 `1` | 回傳 `0`（`cfg` 無效）並把 `*out` 清零 | `cfg` 為 `NULL` 回 `0`；`out` 為 `NULL` 回 `0`（安全） |
| `aec_linear_is_cancelling` | 回傳 `1`（正在消除）或 `0` | — | 回傳 `0`（安全；`0` 代表「沒有證據顯示正在消除」，不代表 filter 壞掉） |
| `aec_is_valid_sample_rate` | 支援回傳 `1` | 不支援回傳 `0` | 不適用（傳入 int） |

必須知道的三件事：

1. **處理函式沒有 per-call 的錯誤碼。** `aec_process()` 回傳 `void`。所有能被偵測的
   錯誤都在建立階段（`aec_create` / `aec_init`）擋掉，之後就只能靠呼叫端保證
   pointer 有效、長度正確、串流同步。
2. **兩個「填入結構」的函式在 handle 為 `NULL` 時靜默 no-op**，不會清空也不會標記你
   傳入的結構。若你沒有先初始化就讀它，讀到的是未初始化的值。
3. **`aec_get_mem_size` 用 `0` 代表失敗**，不是回傳負值。務必檢查 `== 0`
   再拿去配置記憶體。

設定驗證在 `aec_create` / `aec_init` / `aec_get_mem_size` 三者內部都會先執行，
任一欄位超出 [§6](c_user_manual_zh_TW_02_config.md#6-aecconfig-完整參考) 的允許範圍就整組拒絕。

---

> [索引](c_user_manual_zh_TW.md) · 下一部分：[c_user_manual_zh_TW_02_config.md](c_user_manual_zh_TW_02_config.md) <!-- split-nav -->
