# AEC C 使用手冊（繁體中文）

> **版本：4.0.0rc1**（對應 `python/aec.py` 的 `__version__`，本手冊唯一的版本標示）
>
> 本手冊是給**函式庫使用者**看的整合說明：什麼時候呼叫哪個函式、參數怎麼設。
> 演算法內部設計請看 [`aec_methods.md`](aec_methods.md)。

## 你需要 include 的 header

```c
#include "aec.h"        /* c_impl/include/aec.h — 標準整合只需要這一個 */
```

一般整合只 include `aec.h`。另有兩個 header 屬於**進階、範圍較窄的對外介面**，
只有要在 AEC 之外自己跑一套 residual-echo suppression 的整合才需要
（本專案的四麥克風 pipeline 就是這樣用的，見 `4ch_pipelines/4aec_nr_res.c`）：

| Header | 何時需要 |
|---|---|
| `suppression_gain.h` | 自己建一個 `SuppressionGain`，對 beamformer 之後的訊號做 RES |
| `aec3_balanced_config.h` | 取用與 balanced preset 相同的 RES 調校常數 |

`c_impl/include/` 底下其餘約 27 個 header（`aec3_post.h`、`pbfdkf.h`、`delay_aec3.h` 等）
全部是**內部實作**用的，會隨版本改變，請勿直接 include，也不要依賴其中任何型別或函式。
注意 `aec.h` 目前會把 `struct Aec` 完整攤開，因此它們仍必須留在 include 路徑上
才能編譯——「留在路徑上」不等於「可以使用」。

`aec.h` 本身會 include 共用 DSP 函式庫（`audio_common`）的 `fft_wrapper.h` 與 `hpf.h`，
所以編譯時兩個 include 路徑都要給（見 §2）。

---

## 1. 這個函式庫是什麼／不是什麼

單聲道聲學回音消除（AEC）。每個 `Aec` 實例處理**一支麥克風 + 一路參考訊號**：

| 訊號 | 意義 |
|---|---|
| `mic` | 麥克風收到的訊號：近端語音 + 環境噪音 + 喇叭耦合進來的回音 |
| `ref` | **實際送去喇叭播放**的訊號（playback loopback / render reference） |
| `out` | 消除回音後的單聲道輸出 |

`ref` 必須是裝置真正播出去的內容，不能拿任意一段遠端錄音代替。

### 支援的取樣率與訊號格點

`sample_rate` 與 `fft_size` 是**成對**檢查的，下表以外的組合一律被 `aec_create()` /
`aec_init()` / `aec_get_mem_size()` 拒絕：

| sample_rate | fft_size | hop（每次呼叫的 sample 數） | hop 時間 | n_partitions（預設 filter 長度下） |
|---:|---:|---:|---:|---:|
| 8000 | 256 | 128 | 16.000 ms | 4 |
| 16000 | 256（預設） | 128 | 8.000 ms | 7 |
| 16000 | 512 | 256 | 16.000 ms | 4 |
| 48000 | 1024（預設） | 512 | 10.667 ms | 6 |

16 kHz 有兩個合法格點，預設是 256。要用 512 必須在 `aec_create()` 前自行覆寫
`cfg.fft_size`。可用 `aec_is_valid_sample_rate()` 先檢查取樣率
（8000/16000/48000 回傳 1，其餘回傳 0）。

### 硬性限制

| 限制 | 說明 |
|---|---|
| **只支援單聲道** | 一個實例 = 1 mic + 1 ref。沒有 mic array、沒有 stereo reference。多路訊號請各自建立獨立實例。 |
| **延遲只能是正的** | 回音必須**晚於**參考訊號抵達（`mic` 落後 `ref`）。內部是對參考訊號的環形緩衝做回看，負延遲無法表示，必須在上游先對齊。 |
| **沒有非線性回音模型** | 只有線性 filter + 殘留回音抑制。喇叭嚴重失真、破音造成的非線性回音無法建模。 |
| **取樣率限 8/16/48 kHz** | 其他取樣率請在外部先重新取樣。 |
| **mic/ref 必須同一時鐘** | 兩路取樣率要相同且同步。長時間的 clock drift 會讓延遲估計與 filter 收斂失效，函式庫不負責補償。 |
| **不是 thread-safe** | 單一實例不可被多執行緒同時呼叫（例外見 §8 的 streaming API）。 |

---

## 2. Quick start

### 建置

`audio_common` 需與本 repo 同層（sibling）。兩邊都要建，而且**必須用同一個 BACKEND**：

```bash
make -C ../audio_common lib BACKEND=kiss     # 共用 DSP 函式庫
make -C c_impl          lib BACKEND=kiss     # 本函式庫 libaec.a
```

> ⚠️ 兩個 repo 的預設 BACKEND **不一樣**（本 repo 預設 `kiss`，`audio_common` 預設
> `ne10`）。不明確指定會拿到不匹配的組合，請永遠明確帶上 `BACKEND=`。

產出路徑會依編譯參數雜湊命名，用各自的查詢目標取得，不要自己拼路徑：

```bash
AEC_LIB="$(make -s -C c_impl          print-lib-path BACKEND=kiss)"
AC_LIB="$(make  -s -C ../audio_common print-lib-path BACKEND=kiss)"
```

### 連結

```bash
cc -std=c99 -O2 -ffp-contract=off app.c \
   -Ic_impl/include -I../audio_common/include \
   "$AEC_LIB" "$AC_LIB" -lm -o app
```

`-ffp-contract=off` 是**必要**的：允許 FMA 合併會改變遞迴狀態（HPF、偵測器累加器）
的數值，導致行為與驗證過的版本不一致。`libaec.a` 不含共用 DSP 的目的檔，
`libaudio_common.a` 一定要一起連結。

### 最小可用範例

```c
#include <stdlib.h>
#include "aec.h"

/* 由你的音訊來源填入一個 hop 的 mic/ref；沒有資料時回傳 0 結束。 */
extern int  app_read_hop(float *mic, float *ref, int hop);
extern void app_write_hop(const float *out, int hop);

int run_aec(int sample_rate)
{
    AecConfig cfg;
    Aec       aec;
    float    *mic, *ref, *out;
    int       hop;

    aec_config_from_preset(&cfg, AEC_PRESET_BALANCED, sample_rate);
    if (aec_create(&aec, &cfg) != 0) return -1;      /* 0 = 成功，-1 = 失敗 */

    hop = aec_hop_size(&aec);                        /* 每次剛好餵這麼多 sample */
    mic = (float *)malloc((size_t)hop * sizeof *mic);
    ref = (float *)malloc((size_t)hop * sizeof *ref);
    out = (float *)malloc((size_t)hop * sizeof *out);

    if (mic && ref && out) {
        while (app_read_hop(mic, ref, hop)) {
            aec_process(&aec, mic, ref, out);        /* 回傳 void，不會失敗 */
            app_write_hop(out, hop);
        }
    }

    free(mic); free(ref); free(out);
    aec_destroy(&aec);                               /* heap 實例必須剛好呼叫一次 */
    return (mic && ref && out) ? 0 : -1;
}
```

---

## 3. 資料契約

### Sample 格式與尺度

- 型別一律是 `float`（32-bit），**不是** `int16`。
- 標稱範圍是 **[-1.0, +1.0]**。從 PCM16 轉入時除以 32768。
- 超出 [-1, 1] 不會被拒絕，但飽和偵測（`saturation_threshold`，預設 0.95）會判定為削波。
- AEC 不做輸出 peak limiting；`out` 可能短暫超過 [-1, 1]。需要限制峰值時，
  請在整條 AEC/beamforming/NR 流程的最後加 AGC/DRC/limiter；轉成 PCM 時仍須飽和裁切。

### Frame 與 hop

**每次 `aec_process()` 必須剛好傳入 `aec_hop_size()` 個 sample。**

這些處理函式都沒有長度參數，也不檢查長度：

- 給少了 → 讀到未初始化記憶體（無聲的資料毀損，不會報錯）
- 給多了 → 多的部分直接被忽略

上游若是變動長度的 buffer，請自行加一層 ring buffer 切成固定 hop。
`mic`、`ref`、`out` 由呼叫端自行配置，各自獨立、長度至少一個 hop。

### 延遲契約

`mic` 與 `ref` 之間的聲學延遲由內建的線上延遲估計處理（`enable_delay_est`，預設開啟）。

**可擷取的延遲上限是編譯期固定的，不是 `max_delay_ms`。** 這點容易誤解：

| 取樣率 | 可擷取上限 |
|---:|---:|
| 8 kHz | 1216 ms |
| 16 kHz | **608 ms** |
| 48 kHz | **608 ms** |

上限來自 matched filter 的搜尋長度（`delay_aec3.h` 的
`DA_MAX_FILTER_LAG` × 4 倍降取樣）。48 kHz 輸入會先抗混疊降到 16 kHz，
估計結果再乘 3 回傳 native-rate samples，所以 wall-clock 範圍仍為
約 608 ms，不是把 9728 直接除以 48 kHz。此上限與設定無關，
**調 `max_delay_ms` 不會提高它**。（608 ms 是幾何全 span；若再計入
matched filter 的 reliability 條件 `lag < FILTER_SIZE-10`，實際可靠取得
peak 的上界約 509 ms——`nn_integration_interface.md` 引用的就是後者，兩
個數字定義不同、不矛盾。）
`max_delay_ms`（預設 1024 ms）只決定參考訊號環形緩衝要開多大；調大只是多花記憶體。
超過上表的延遲請在上游先粗對齊，實機仍應先量測總延遲。

**超出可靠上界（~509 ms）的真延遲不會「估不到」，而是可能在範圍內
高信心誤鎖**（confidence 只量 histogram 一致性、不量匹配品質）——機制
細節、blind-test 語料的實測延遲分佈（2021 全集 p50≈48 ms / p90≈225 ms /
max 923 ms、約 5% 超出 509 ms）與板端/離線的建議做法（系統延遲用
`fixed_delay_samples` 由系統層補償），見
`delay_estimator_design_zh_TW.md`。

延遲估計**不能**修正這些問題：mic/ref 取樣率不同、持續的 clock drift、
送錯的 reference 訊號。這些必須在上游解決。

離線處理且檔案已精準對齊時，可設 `cfg.enable_delay_est = 0` 省掉首次擷取的擾動。
線上產品請維持開啟。

### 演算法延遲（實測）

以脈衝訊號量測輸出峰值位移的結果：

| 格點 | `enable_res = 1`（預設） | `enable_res = 0` |
|---|---:|---:|
| 8 kHz / 256 | 16.000 ms（1 hop） | 0 ms |
| 16 kHz / 256 | 8.000 ms（1 hop） | 0 ms |
| 16 kHz / 512 | 16.000 ms（1 hop） | 0 ms |
| 48 kHz / 1024 | 10.667 ms（1 hop） | 0 ms |

開啟殘留回音抑制時固定多一個 hop 的延遲；關閉後線性路徑沒有額外延遲。
沒有任何 look-ahead。

---

## 4. 生命週期

兩條路徑二選一。輸出**逐位元相同**（同一 BACKEND 下實測 200 hop 完全一致）。

### A. Heap 路徑（`aec_create`）

```c
AecConfig cfg;
Aec aec;                                    /* Aec 結構由呼叫端持有 */

aec_config_from_preset(&cfg, AEC_PRESET_BALANCED, 16000);
if (aec_create(&aec, &cfg) != 0) { /* 設定無效或記憶體不足 */ }

/* ... 每個 hop 呼叫 aec_process(&aec, mic, ref, out) ... */

aec_destroy(&aec);                          /* 必須剛好一次，釋放內部 arena */
```

`aec_create()` 內部只做**一次** malloc（單一 arena）。之後 `aec_process()` 與
`aec_reset()` 都不會再配置記憶體、不會做 I/O。

### B. 呼叫端自備記憶池（`aec_get_mem_size` + `aec_init`）

沒有 heap 的嵌入式目標用這條。整個實例（含 `Aec` 結構本身）都放進呼叫端的記憶池：

```c
AecConfig cfg;
aec_config_from_preset(&cfg, AEC_PRESET_BALANCED, 16000);

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

### 記憶池大小（實測，balanced preset，其餘為預設值）

| 格點 | KISS backend | NE10 backend |
|---|---:|---:|
| 8 kHz / 256 | 276,928 B | 276,320 B |
| 16 kHz / 256 | 381,008 B | 380,400 B |
| 16 kHz / 512 | 510,080 B | 508,704 B |
| 48 kHz / 1024 | 1,166,928 B | 1,164,016 B |

上表是 64-bit 主機建置的實測值，僅供規劃參考。**實際請以 `aec_get_mem_size(&cfg)`
的回傳值為準**——它會隨設定、backend 與目標 ABI 改變。

影響記憶池大小最明顯的兩個開關（16 kHz/256 KISS，相對於上表 381,008 B）：

| 設定 | 記憶池 | 差異 |
|---|---:|---:|
| `enable_delay_est = 0` | 249,936 B | −131,072 B |
| `enable_shadow = 0` | 348,448 B | −32,560 B |

`enable_res` 與 `enable_cng` **不影響**記憶池大小（相關緩衝區一律配置）。

### `aec_reset`

```c
aec_reset(&aec);    /* 或 aec_reset(aec) */
```

清掉所有執行期狀態（延遲估計、filter、double-talk、收斂狀態），但**保留設定與記憶體
配置**。處理下一段**獨立**串流（例如批次處理下一個檔案）前一定要呼叫，否則前一段的
狀態會汙染下一段。

### 什麼不能改

`sample_rate`、`fft_size`、`filter_length`、`n_partitions` 決定記憶體佈局，
**建立後不可更改**。要換設定必須 destroy 後重新 create（記憶池路徑則是重新 `aec_init`）。
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
| `aec_process_context_shared_far` | 回傳 `void` | 同上 | **會 crash** |
| `aec_analyze_render` | 回傳 `AecBufferingEvent` | 以事件值回報緩衝狀況，非錯誤碼 | **會 crash** |
| `aec_process_capture` | 回傳 `AecBufferingEvent` | 同上 | **會 crash** |
| `aec_last_buffering_event` | 回傳 `AecBufferingEvent` | — | **會 crash** |
| `aec_hop_size` | 回傳 hop sample 數 | — | **會 crash** |
| `aec_far_fft_real_compute_count` | 回傳累計次數 | — | **會 crash** |
| `aec_get_res_context` | 回傳 `void`，填入結構 | — | **靜默 no-op**（你的結構完全不被寫入） |
| `aec_debug_status` | 回傳 `void`，填入結構 | — | **靜默 no-op**（你的結構完全不被寫入） |
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
任一欄位超出 §6 的允許範圍就整組拒絕。

---

## 6. `AecConfig` 完整參考

**永遠從 `aec_config_from_preset()`（或 `aec_config_defaults()`）開始**，再覆寫少數必要
欄位。這兩個函式會先 `memset` 整個結構，所以未列出的欄位都會被歸零；自己手動填一個
未初始化的 `AecConfig` 幾乎一定會被驗證擋下。

```c
AecConfig cfg;
aec_config_from_preset(&cfg, AEC_PRESET_BALANCED, 16000);
cfg.enable_cng = 0;          /* 只覆寫你真的要改的 */
```

`aec_config_defaults()` 給的就是 balanced 的值；`aec_config_from_preset()` 只在此之上
改一個欄位（見 §7）。以下所有預設值與範圍都直接取自程式碼。

### 6.1 訊號格點（建立後不可更改）

| 欄位 | 型別 | 預設 | 允許範圍 | 說明／何時調整 |
|---|---|---|---|---|
| `sample_rate` | `int` | 呼叫端指定 | 8000 / 16000 / 48000 | 必填。與音訊來源一致。 |
| `fft_size` | `int` | 8k→256、16k→256、48k→1024 | 須與 `sample_rate` 成對（見 §1） | 只有 16 kHz 有第二選擇（512）。改成 512 會讓 hop 變 16 ms、延遲加倍，但每秒呼叫次數減半。 |
| `filter_length` | `int` | `sr × 52/1000`（48 kHz 為 `sr × 64/1000`）→ 8k:416、16k:832、48k:3072 | 1 – 4096 | **單位是 sample，不是 ms。** 涵蓋的回音路徑長度。大房間／長殘響時調高，代價是記憶體與運算量上升。 |
| `n_partitions` | `int` | `0`（自動） | `0` 或 1 – 256 | 留 `0` 即可，由 `filter_length` 與 hop 自動推導。 |

### 6.2 模組開關（全部只接受 0 或 1）

| 欄位 | 預設 | 說明／何時調整 |
|---|---:|---|
| `enable_res` | `1` | 殘留回音抑制。關閉後輸出是純線性 filter 殘差、延遲降為 0，但回音殘留明顯變多。診斷時可關（見 §9）。 |
| `enable_cng` | `1` | 舒適噪音。若下游還有另一套噪音處理／CNG，**請關掉這個**，兩層疊加會讓噪音底變得不自然。 |
| `enable_delay_est` | `1` | 線上延遲估計。只有在離線且檔案已精準對齊時才關。關閉會少掉 131,072 B 記憶體（16 kHz）。 |
| `enable_highpass` | `1` | mic 路徑的高通濾波（見 `highpass_cutoff_hz`）。上游已有高通時可關。 |
| `enable_saturation` | `1` | 削波偵測與參考訊號軟限幅。建議維持開啟。 |
| `enable_shadow` | `1` | 輔助 filter，用於偵測回音路徑變化。關閉會少掉 32,560 B（16 kHz），但路徑突變後的恢復會變慢。 |
| `saturation_softclip_ref` | `1` | 對參考訊號套用軟限幅。`enable_saturation = 0` 時無作用。 |
| `return_res_context` | `0` | 進階整合用，見 §8。 |
| `spatial_linear_context` | `0` | 進階整合用，見 §8。**只有在 `enable_res = 0` 且 `return_res_context = 1` 時才合法**，其他組合會被驗證拒絕。 |

### 6.3 抑制強度

| 欄位 | 型別 | 預設 | 允許範圍 | 說明／何時調整 |
|---|---|---:|---|---|
| `min_gain_floor_far_active_db` | `float` | `-28.0` | -300 – 50 | **這是三個 preset 唯一的差別**，也是主要的強度旋鈕。遠端在說話時允許的最低增益。**調低**（例如 -38）＝回音壓得更乾淨，但 double-talk 時近端更容易被壓掉；**調高**（例如 -20）＝近端保留較好，回音殘留較多。優先用 preset（§7）而不是自己填數字。 |
| `min_gain_floor_dt_db` | `float` | `-16.0` | -300 – 50 | double-talk 期間專用的最低增益，與上一項分開。**調高**可在雙方同時說話時多保留近端。需 `dt_aware_res_floor_enabled = 1` 才生效。 |
| `dt_aware_res_floor_enabled` | `int` | `1` | 0 / 1 | 是否啟用上面這個 double-talk 專用下限。 |
| `dt_aware_recovery_soft` | `int` | `1` | 0 / 1 | 近端最近有聲音時，改用非破壞性的方式做重新對齊，避免把近端語音削掉。 |

### 6.4 延遲估計

| 欄位 | 型別 | 預設 | 允許範圍 | 說明／何時調整 |
|---|---|---:|---|---|
| `max_delay_ms` | `float` | `1024.0` | 0 – 60000 | **參考訊號環形緩衝的尺寸下限（ms）**，不是搜尋上限（見 §3 延遲契約）。調高只是多配記憶體，不會提高可擷取的延遲。 |
| `delay_buffer_ms` | `float` | `2048.0` | 0 – 120000 | 參考訊號環形緩衝長度（ms）。實際樣本數取 `delay_buffer_ms` 與 `max_delay_ms + 4096 samples` 兩者較大者。調高 `max_delay_ms` 時通常要一併調高。 |
| `delay_est_init_s` | `float` | `0.3` | 0 – 3600 | 延遲估計值需維持穩定多久才視為已鎖定（秒）。 |
| `delay_est_period_s` | `float` | `0.5` | 0 – 3600 | 鎖定後的重新確認週期（秒）。回音路徑常變動（裝置會移動）可調短。 |
| `delay_acquire_protect_converged` | `int` | `1` | 0 / 1 | filter 已收斂時，保護它不被延遲重新擷取破壞。 |
| `delay_acquire_warm_transfer` | `int` | `1` | 0 / 1 | 首次擷取到延遲時，把已學到的回音模型平移過去而不是歸零，避免開場一秒左右出現一段明顯回音。建議維持開啟。 |
| `delay_acquire_inst_erle_db` | `float` | `4.0` | -100 – 100 | 上一項的觸發門檻（dB）：既有模型要夠好才值得平移。 |
| `delay_acquire_protect_inst_erle` | `int` | `0` | 0 / 1 | 另一種較保守的做法（直接擋掉重新對齊），預設關閉。 |

### 6.5 以 10 ms 為單位撰寫的時間欄位 ⚠️

這六個欄位有特殊語意，是最容易誤用的一組：

> **你填的數值會被當成「N × 10 ms 的實際時長」，函式庫在建立時自動換算成目前格點的
> hop 數。** 它們**不是**直接的 hop 計數。

例如在 16 kHz/256（hop = 8 ms）下填 `warmup_frames = 100`，實例內部會變成 125 hop，
也就是 1000 ms；在 48 kHz 下會變成 94 hop，同樣約 1000 ms。**換算後的時長在所有格點
上一致**，這正是這個設計的目的。換算寫進實例自己的設定副本，你傳入的 `AecConfig`
不會被改動。

| 欄位 | 型別 | 預設 | 換算後時長 | 允許範圍 | 說明／何時調整 |
|---|---|---:|---:|---|---|
| `warmup_frames` | `int` | `100` | ≈ 1000 ms | 0 – 1000000 | 啟動期長度：這段期間 filter 用較積極的方式收斂。冷啟動收斂太慢可調高。 |
| `epc_hangover` | `int` | `20` | ≈ 200 ms | 0 – 1000000 | 偵測到回音路徑改變後的加速恢復期長度。裝置常移動可調高。 |
| `ne_recent_hold` | `int` | `150` | ≈ 1500 ms | 0 – 1000000 | 「近端最近說過話」旗標的保持時間，餵給 §6.3 的 double-talk 保護。近端一停就被壓掉時可調高。 |
| `filter_misadjustment_stable_frames` | `int` | `30` | ≈ 300 ms | 0 – 1000000 | 判定 filter 已穩定所需的持續時間。 |
| `filter_misadjustment_hangover_frames` | `int` | `100` | ≈ 1000 ms | 0 – 1000000 | 上述判定的保持時間。 |
| `shadow_err_alpha` | `float` | `0.80` | （平滑係數，非時長） | 0 – 1 | 輔助 filter 誤差的平滑係數，同樣會依格點重新換算（16 kHz/256 下實際為 0.8365）。不建議調整。 |

`ne_recent_sustain`（預設 `3`，範圍 0 – 1000000）是**真正的次數計數**（連續幾次事件），
**不會**被換算，不要跟上表混淆。

### 6.6 其餘可設定欄位

這些是內部調校參數，預設值是整體調過的。除非有明確量測依據，**請維持預設**。

| 欄位 | 型別 | 預設 | 允許範圍 | 說明 |
|---|---|---:|---|---|
| `mu` | `float` | `0.3` | 0 – 10 | 主 filter 適應步進。調高收斂快但更容易發散。 |
| `delta` | `float` | `1e-8` | 0 – 1 | 適應運算的除零保護下限。 |
| `highpass_cutoff_hz` | `float` | `80.0` | `enable_highpass = 1` 時須 > 0 且 **< 0.45 × sample_rate**；關閉時為 0 – 20000 | mic 路徑高通截止頻率。低頻風噪嚴重可調高。 |
| `saturation_threshold` | `float` | `0.95` | 0 – 10 | 判定削波的振幅門檻（相對滿刻度 1.0）。 |
| `shadow_mu_min` | `float` | `0.5` | 0 – 10 | 輔助 filter 的最小步進。 |
| `shadow_mu_nlms` | `float` | `0.5` | 0 – 10 | 輔助 filter 的標稱步進。 |
| `epc_total_rise` | `float` | `1.5` | 0 – 1000 | 判定回音路徑改變的能量上升倍率門檻。誤判太多可調高。 |
| `epc_delta_threshold` | `float` | `0.3` | 0 – 1000 | 同上，變化量門檻。 |
| `epc_mu_floor` | `float` | `0.5` | 0 – 10 | 路徑改變恢復期間的步進下限。 |
| `kalman_q_high` | `float` | `1e-3` | 0 – 1 | 未收斂／路徑改變時的模型不確定度。 |
| `kalman_q_low` | `float` | `1e-6` | 0 – 1 | 已收斂後的模型不確定度。 |
| `ne_recent_threshold` | `float` | `0.3` | 0 – 1000 | 判定近端有聲音的能量門檻。近端偵測過鈍可調低。 |
| `filter_misadjustment_scale_min` | `float` | `0.5` | 0 – 1000 | 回音估計修正倍率下限。 |
| `filter_misadjustment_scale_max` | `float` | `2.0` | 0 – 1000 | 回音估計修正倍率上限。 |

---

## 7. Preset 選擇

```c
aec_config_from_preset(&cfg, AEC_PRESET_BALANCED, 16000);
```

三個 preset **只差一個欄位**：`min_gain_floor_far_active_db`。其餘欄位完全相同
（已逐位元比對確認）。所以 preset 就是「回音壓多乾淨」對「近端保留多完整」的單一取捨軸。

| Preset | `min_gain_floor_far_active_db` | 適用情境 |
|---|---:|---|
| `AEC_PRESET_MILD` | `-20.0` | 近端音質優先。喇叭音量小、耦合弱的場景；可接受較多回音殘留。 |
| `AEC_PRESET_BALANCED` | `-28.0` | **預設，一般通話建議值。** |
| `AEC_PRESET_AGGRESSIVE` | `-38.0` | 回音抑制優先。大音量、喇叭與麥克風靠很近、耦合強的場景；double-talk 時近端較容易被壓抑。 |

先用 preset 調整，不要一開始就自己填 `min_gain_floor_far_active_db`。三個值之間需要
中間值時，才直接覆寫該欄位。傳入超出 enum 範圍的 preset 值會退回 balanced 的設定。

---

## 8. 進階整合 seam

一般使用者不需要這一節。只有在你要自己接上外部的噪音抑制／後處理，或是要驅動多個
實例時才會用到。

### 8.1 Context-only 處理路徑

如果你**完全不使用** `aec_process()` 產出的時域音訊，只想讀頻域中間結果自己做後處理，
用 `aec_process_context()`：它跑完整條線性 filter 與後級運算並填好 `AecResContext`，
但不做輸出限幅、也不寫 `out` buffer（沒有 `out` 參數）。

```c
AecConfig cfg;
aec_config_from_preset(&cfg, AEC_PRESET_BALANCED, 16000);
cfg.enable_res          = 0;   /* 後級的抑制不套用到時域訊號 */
cfg.return_res_context  = 1;   /* 但後級仍然執行並公開結果 */
aec_create(&aec, &cfg);

while (app_read_hop(mic, ref, hop)) {
    aec_process_context(&aec, mic, ref);

    AecResContext ctx;
    aec_get_res_context(&aec, &ctx);
    /* 在這裡讀 ctx，必須在下一次處理呼叫之前 */
}
```

> **四個處理進入點可以自由混用。**
>
> `aec_process()`、`aec_process_capture()`、`aec_process_context()`、
> `aec_process_context_shared_far()` 推進的是完全相同的狀態，差別只在於
> **要不要把結果複製到 `out`**。同一個實例上可以任意交錯呼叫，中間不需要
> `aec_reset()`。
>
> 選哪一個純粹是成本考量：下游若不讀 `out`，用 context-only 版本可以省下
> 一次 `hop` 長度的 memcpy。

### 8.2 多實例共用遠端 FFT

多個實例每個 hop 都看到**完全相同**的參考訊號時，可讓其中一個算 FFT、其餘共用：

```c
aec_process_context(&lane[0], mic[0], ref);      /* lane 0 自己算 */
AecResContext ctx0;
aec_get_res_context(&lane[0], &ctx0);

for (int ch = 1; ch < n_lanes; ++ch)
    aec_process_context_shared_far(&lane[ch], mic[ch], ref, ctx0.far_spec);
```

契約：

- `shared_far_spec` 必須正好是**同一個 hop**、由**同一段參考時域訊號**算出來的頻譜。
- 即使提供了 `shared_far_spec`，`ref` 參數**仍然必須傳**（參考訊號的非 FFT 用途還在）。
- 傳 `NULL` 就退回 `aec_process_context()` 的行為（自己算）。
- `aec_far_fft_real_compute_count()` 是唯讀計數器，可用來驗證共用真的生效
  （純共用的實例會一直維持 0）。

### 8.3 `AecResContext` 欄位

呼叫 `aec_get_res_context()` 取得。頻域欄位只有在 `enable_res = 1` **或**
`return_res_context = 1` 時才有效，否則為 `NULL`。

| 欄位 | 說明 |
|---|---|
| `n_freqs` / `hop_size` | 陣列長度。所有頻域陣列長度都是 `n_freqs`。 |
| `error_spec` | 線性輸出的頻譜 E(f)，**audio 尺度**（與 [-1,1] 時域一致）。 |
| `echo_spec` / `near_spec` | 對應的回音估計與擷取訊號頻譜，滿足 `near_spec = error_spec + echo_spec`。 |
| `res_gain` | 抑制增益，振幅 [0, 1]。**`spatial_linear_context = 1` 時為 `NULL`** —— 代表「沒有計算」，若誤讀成全 0 會變成「全部靜音」。**使用前務必檢查非 NULL。** |
| `r2` | 殘留回音功率譜。**int16² 尺度**：要換成 audio 功率尺度須除以 32768²。 |
| `comfort_noise` | 舒適噪音功率譜。同樣是 int16² 尺度。 |
| `linear_hop` / `formed_hop` | 時域中間結果。 |
| `far_power`、`erle_factor`、`dt_indicator`、`divergence`、`saturation_level`、`erl_estimate`、`shadow_dt`、`is_stationary_dt`、`filter_converged`、`filter_once_converged`、`epc_active` | 供外部判斷／診斷用的純量狀態。 |

> ⚠️ 所有 pointer 欄位都直接指向實例內部的 per-hop 緩衝區，**不是複本**。
> 不可修改、不可 free，而且必須在**下一次對同一實例的任何處理呼叫之前**讀完或複製走。
> 四個進入點（`aec_process`、`aec_process_context`、`aec_process_context_shared_far`、
> `aec_process_capture`）都會就地覆寫這些緩衝區。

### 8.4 分離的 render／capture 進入點

播放與錄音不是由同一個 callback 交付時使用：

```c
AecBufferingEvent e1 = aec_analyze_render(&aec, ref);        /* 放入一個 render hop */
AecBufferingEvent e2 = aec_process_capture(&aec, mic, out);  /* 消耗並處理一個 mic hop */
```

兩者一樣是一次剛好一個 hop。單執行緒下先 `aec_analyze_render(ref)` 緊接
`aec_process_capture(mic, out)`，輸出與 `aec_process(mic, ref, out)` **逐位元相同**
（實測 200 hop 完全一致且不產生任何事件）。

回傳的事件（也可用 `aec_last_buffering_event()` 讀最近一次）：

| 事件 | 意義 |
|---|---|
| `AEC_BUF_NONE` | 正常 |
| `AEC_BUF_RENDER_UNDERRUN` | capture 時緩衝區沒有 render 資料，本次以靜音參考訊號處理 |
| `AEC_BUF_RENDER_OVERRUN` | render 端超出緩衝容量，資料被丟棄 |
| `AEC_BUF_API_SKEW` | 兩端呼叫節奏異常（診斷用） |

內部 FIFO 的實際容量（依格點而定）：8 kHz 512 ms、16 kHz/256 512 ms、
16 kHz/512 512 ms、48 kHz 341 ms。這是用來吸收兩端**呼叫排程抖動**的，
**不是**聲學延遲緩衝，也**不能**用來補償 clock drift。持續出現 underrun／overrun
代表上游排程有問題，請修排程而不是加大 FIFO。

**執行緒契約**：只支援 single-producer / single-consumer——恰好一個執行緒呼叫
`aec_analyze_render()`，恰好另一個（可以不同）執行緒呼叫 `aec_process_capture()`。
在這個形狀下兩者可以完全並行、不需要外部鎖。除此之外的任何形狀都**不在保護範圍**，
需要呼叫端自行序列化，包括：同一側有第二個執行緒；兩端還在呼叫時就
`aec_reset()` / `aec_destroy()`；以及在 streaming 進行中對同一實例並行呼叫
`aec_process()`。

注意上面這條限制是**執行緒**層面的，與 §8.1 的進入點選擇無關 —— 四個處理進入點
本身可以自由混用。

---

## 9. 常見問題排除

先確認基本面：`ref` 真的是送去喇叭的訊號嗎？mic/ref 取樣率相同且同步嗎？
每次呼叫真的餵了剛好 `aec_hop_size()` 個 sample 嗎？這三項有任何一項不成立，
調參數都沒有意義。

### 9.1 回音殘留太多

依序處理：

1. **先分離問題。** 設 `cfg.enable_res = 0` 跑一次，聽線性殘差。
   - 回音仍然很大 → 問題在**線性側**：`ref` 送錯、延遲超出 §3 的可擷取上限、
     回音路徑比 `filter_length` 長，或兩路取樣率不同。往下走第 2、3 點。
   - 線性殘差已經很乾淨 → 問題在**抑制強度**，跳到第 4 點。
2. **檢查延遲。** 用 `aec_debug_status()` 讀 `delay_samples`：`-1` 代表還沒擷取到延遲。
   若始終是 `-1`，確認遠端有足夠能量，並確認實際延遲落在 §3 表列的可擷取上限內。
3. **加長 filter。** 大房間或長殘響時調高 `filter_length`（單位是 **sample**）。
   必須 destroy 後重新 create，並重新評估記憶體與運算量。
4. **加強抑制。** 改用 `AEC_PRESET_AGGRESSIVE`；或直接把
   `min_gain_floor_far_active_db` **調低**。代價是 double-talk 時近端更容易被壓。

### 9.2 Double-talk 時近端被壓掉

| 動作 | 方向 |
|---|---|
| 換較溫和的 preset | `AEC_PRESET_BALANCED` → `AEC_PRESET_MILD` |
| 或直接調 `min_gain_floor_far_active_db` | **調高**（例如 -28 → -20） |
| 放寬 double-talk 專用下限 `min_gain_floor_dt_db` | **調高**（例如 -16 → -12），需 `dt_aware_res_floor_enabled = 1` |
| 近端一停就被壓掉 → 延長保護 `ne_recent_hold` | **調高**（單位是 10 ms） |
| 近端偵測太不敏感 → `ne_recent_threshold` | **調低** |
| 確認 `dt_aware_recovery_soft = 1` | 維持開啟 |

**不要**用關閉 `enable_res` 來解決這個問題——那會讓回音殘留大幅上升。

### 9.3 啟動時收斂太慢／前一秒回音明顯

| 動作 | 方向 |
|---|---|
| 確認 `delay_acquire_warm_transfer = 1` | 維持開啟。關閉會在首次擷取到延遲時把已學到的模型歸零，正是「開場一秒有回音」的典型成因。 |
| 延長啟動期 `warmup_frames` | **調高**（單位是 10 ms，預設 ≈ 1000 ms） |
| 若冷啟動時 `delay_acquire_inst_erle_db` 門檻擋住了平移 | **調低**（預設 4.0 dB） |
| 應用層面 | filter 收斂需要一段有意義的遠端能量。可在暖機期間先靜音輸出或播提示音。 |

`aec_debug_status()` 的 `erle_windowed_db` 與 `filter_converged` 可用來觀察收斂進度。

### 9.4 裝置移動時回音突然變大

回音路徑改變後需要重新收斂，短暫洩漏屬正常，通常會自動恢復。若恢復太慢或太頻繁：

| 動作 | 方向 |
|---|---|
| 確認 `enable_shadow = 1` | 維持開啟——關掉它偵測路徑變化的能力會明顯變差 |
| 延長加速恢復期 `epc_hangover` | **調高**（單位是 10 ms，預設 ≈ 200 ms） |
| 路徑變化偵測太鈍 → `epc_total_rise` | **調低**（預設 1.5） |
| 縮短延遲重新確認週期 `delay_est_period_s` | **調低**（預設 0.5 s） |
| 加長 `filter_length` | **調高**，讓模型涵蓋更多可能的路徑 |

### 9.5 其他

| 現象 | 先檢查 | 處理方向 |
|---|---|---|
| 長時間後回音逐漸變差 | mic/ref clock drift | 改用共同時鐘，或在上游做 drift 補償。函式庫不處理。 |
| 批次處理第二個檔案結果異常 | 沿用了前一個檔案的狀態 | 每個獨立串流前呼叫 `aec_reset()` |
| 輸出有突波／發散 | 削波、`ref` 送錯、延遲劇烈變化 | 維持 `enable_saturation`、`enable_shadow` 開啟；檢查輸入是否已削波 |
| 噪音底聽起來不自然、會抽動 | CNG 疊了兩層 | 下游若已有噪音處理，設 `enable_cng = 0` |
| streaming API 持續 underrun／overrun | 兩端排程或 producer 太慢 | 修排程。不要把 FIFO 當成 clock drift 的補償手段。 |
| 建立實例一直失敗 | 設定被驗證拒絕 | 對照 §6 的允許範圍；特別注意 `sample_rate`/`fft_size` 必須成對合法，以及 `spatial_linear_context` 的組合限制 |
| 換 backend 後輸出不同 | KISS 與 NE10 的 FFT 實作不同 | 這是預期行為。同一 backend 下 heap 與記憶池路徑才保證逐位元相同。 |

---

## 附錄 A：`aec_wav` 命令列工具

`aec_wav` 是**示範與驗證工具**，不是交付的函式庫介面。它的旗標與 C API 沒有一對一
關係，正式整合請用 §2 – §8 的 API。

```bash
make -C c_impl BACKEND=kiss                        # 建置（產出 aec_wav）
BIN="$(make -s -C c_impl print-bin-dir BACKEND=kiss)"

"$BIN"/aec_wav <mic.wav> <ref.wav> <out.wav> [options]
```

完整旗標（取自程式的參數解析，無其他旗標）：

| 旗標 | 說明 |
|---|---|
| `--preset {mild\|balanced\|aggressive}` | 預設 `balanced` |
| `--cng` | 開啟舒適噪音 |
| `--no-cng` | 關閉舒適噪音 |
| `--no-delay-est` | 關閉線上延遲估計 |
| `--no-res` | 關閉殘留回音抑制 |
| `--no-shadow` | 關閉輔助 filter |
| `--no-hpf` | 關閉 mic 路徑高通 |
| `--fft-size <n>` | 覆寫 FFT 大小：16 kHz 可用 256/512，48 kHz 為 1024 |
| `--debug-level <0..3>` | 0=關閉、1=摘要、2=每 frame、3=完整 |
| `--debug-log <path>` | 將 log 導向檔案（預設 stderr） |
| `--debug-trace <path>` | 每 frame 的 CSV trace |
| `--debug` | 每秒印一行 `aec_debug_status()` 狀態 |

> 兩個容易誤解的地方：**沒有** `--filter-length` 或 `--filter-length-ms` 這個旗標
> （`filter_length` 只能透過 C API 設定）；`--cng` 不是「開啟一個預設關閉的功能」，
> 舒適噪音**預設就是開的**，`--cng` 與 `--no-cng` 只是明確覆寫。

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
| 2 | 未知的 preset 或未知的選項 |
| 3 | 無法開啟輸入檔或無法寫入輸出檔 |
| 4 | 取樣率不支援、mic/ref 取樣率不一致，或訊號格點無效 |

---

## 回報問題

請附上：mic/ref 測試片段、完整的呼叫方式（CLI 命令或 `AecConfig` 設定）、
sample rate 與 preset、使用的 BACKEND，以及問題發生前後至少一秒的
`--debug-level 2 --debug-log <path>` 輸出。
