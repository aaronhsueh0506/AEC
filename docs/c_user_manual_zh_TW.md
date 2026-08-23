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
（本專案的四麥克風 pipeline 就是這樣用的，見 `4ch_aec_bf_nr_res/4aec_nr_res.c`）：

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

| sample_rate | fft_size | hop（每次呼叫的 sample 數） | hop 時間 | n_partitions（預設 filter 長度下） | 狀態 |
|---:|---:|---:|---:|---:|---|
| 8000 | 256 | 128 | 16.000 ms | 4 | **legacy**（非產品正式格點） |
| 16000 | 256（預設） | 128 | 8.000 ms | 7 | 產品 |
| 16000 | 512 | 256 | 16.000 ms | 4 | 產品（另一選擇） |
| 48000 | 1024（預設） | 512 | 10.667 ms | 6 | 產品 |

**核心不接受 `fft_size = 0`（「自動」）——格點必須是明確值。** 16 kHz 有
兩個合法格點，函式庫不會替你猜是哪一個；`fft_size = 0` 會被
`aec_create()` / `aec_init()` / `aec_get_mem_size()` 拒絕（和任何不成對的
組合一樣）。

填值的正確做法只有一條鏈：`aec_config_defaults()` / `aec_config_from_preset()`
是**唯一**會替你填預設格點的地方（8k→256、16k→256、48k→1024，同
`aec_default_fft_size()`），填完之後核心不再猜第二次。要用 16 kHz 的 512
格點，就在 `aec_create()` 前自行覆寫 `cfg.fft_size = 512`。CLI／上層應用
可以替沒指定的使用者選產品預設，但**呼叫函式庫前必須先 resolve 成非零
`fft_size`**（`example/aec_wav.c` 就是這樣做的）。

查詢用 API：

- `aec_resolve_signal_grid(sample_rate, fft_size, &grid)` —— 全函式庫唯一
  的格點解析器。成功回傳 1 並填入 `AecSignalGrid{sample_rate, fft_size,
  frame_size, hop_size, n_freqs, is_legacy}`；失敗回傳 0 且**不動** `*out`。
  驗證、`aec_get_mem_size()`、`aec_create()`、`aec_init()` 全部走這一支，
  所以「配記憶體時看到的格點」＝「init 看到的」＝「`aec_process()` 實際
  消耗的 hop」。`out` 可傳 NULL 當純檢查用。
- `aec_is_valid_sample_rate(sample_rate)` —— 「這個取樣率至少有一個合法
  格點嗎」（8000/16000/48000 回傳 1）。**注意：回傳 1 不代表任何
  `fft_size` 都可以**，成對關係仍要 resolve。
- `aec_default_fft_size(sample_rate)` —— 該取樣率的產品預設 `fft_size`
  （不支援的取樣率回 0）。

`fft_size`、`sample_rate`、`delay_mode`、`delay_num_filters` 都是
**init-time immutable**：任何一項改變都必須重新查記憶體需求並重新 init。

> 8 kHz 是 **legacy 格點**：本函式庫與 Audio_ALG MONO pipeline 支援它
> （既有 8 kHz 測試路徑仍在用），但它不是產品正式格點，Audio_ALG 的
> 4 聲道 pipeline 也不支援（該 API 只保證 16k/256、16k/512、48k/1024
> 三個產品格點）。`AecSignalGrid.is_legacy` 會為它、且只為它設為 1。

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

> ⚠️ 兩個 repo 的預設 BACKEND **邏輯不一樣**：本 repo（`c_impl/Makefile`）不論編譯目標一律
> 預設 `ne10`（`BACKEND ?= ne10`）；`audio_common` 則是自動偵測（編譯目標有 ARM NEON
> → `ne10`，否則退回 `kiss`）。兩者在 NEON 目標上結果一致（都是 `ne10`），但在**非
> NEON 的桌機/CI host** 上會不匹配（本 repo 仍拿 `ne10`，`audio_common` 退回
> `kiss`）。不明確指定會拿到不匹配的組合，請永遠明確帶上 `BACKEND=`。

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

`mic` 與 `ref` 之間的對齊方式由 **`cfg.delay_mode`（三選一，init 時決定、
instance 存活期間不可變）** 決定。這是唯一的真實來源：

| `delay_mode` | 需要的欄位 | 行為 | 何時用 |
|---|---|---|---|
| `AEC_DELAY_MATCHED`（預設） | `delay_num_filters` = 1–5 | 內建 matched filter 線上估計延遲，參考環形緩衝套用 | 延遲未知或會變動（一般線上產品） |
| `AEC_DELAY_FIXED` | `fixed_delay_samples` ≥ 0 | **不建 estimator**；配置一條剛好夠用的參考環形緩衝（`fixed_delay_samples + hop_size`）套用固定延遲 | 硬體路徑固定、bring-up 已量測出系統延遲 |
| `AEC_DELAY_EXTERNAL_ALIGNED` | 無 | **不建 estimator、不建環形緩衝**；呼叫端保證傳進來的 `ref` 已對齊 | HAL 已對齊，或離線且檔案已精準對齊 |

非法組合一律**拒絕**（`aec_create` / `aec_init` 失敗、`aec_get_mem_size`
回 0），不會被靜默忽略或修正：`MATCHED` / `EXTERNAL_ALIGNED` 帶
`fixed_delay_samples ≥ 0`、`FIXED` 沒帶 `fixed_delay_samples`、
非 `MATCHED` 卻改了 `delay_num_filters`、以及任何模式下的
`delay_num_filters = 0`，都會被擋下。

> **`enable_delay_est` 已 deprecated。** 它現在只是 `delay_mode` 的相容
> 轉譯層（`aec_config_resolve_delay()` 在驗證前跑一次）：`= 1`（預設）代表
> 「沿用 `delay_mode`」；`= 0` 且 `fixed_delay_samples ≥ 0` → `FIXED`；
> `= 0` 且沒帶 → `EXTERNAL_ALIGNED`。轉譯後只有 `delay_mode` 是真值，
> 這個欄位會被改寫成對應的鏡像值。舊程式不需改動即可繼續運作，但新程式
> 請直接寫 `delay_mode`；此欄位未來會移除。

以下的可擷取上限**只適用於 `MATCHED`**（`FIXED` 沒有搜尋這回事，
環形緩衝自動配置成剛好夠用的 `fixed_delay_samples + hop_size`，
上限只受 `fixed_delay_samples` 的 `120 × sample_rate` 驗證上界限制）。

**可擷取的延遲上限由 `delay_num_filters` 決定，不是 `max_delay_ms`。**
這點容易誤解。下表是預設 `delay_num_filters = 5`：

| 取樣率 | 可擷取上限 |
|---:|---:|
| 8 kHz | 1216 ms |
| 16 kHz | **608 ms** |
| 48 kHz | **608 ms** |

上限來自 matched filter 的搜尋長度（`delay_aec3.h` 的
`DA_MAX_FILTER_LAG_FOR(n)` × 4 倍降取樣，n = `delay_num_filters`）。
48 kHz 輸入會先抗混疊降到 16 kHz，
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

另有一種**延遲在範圍內**也會誤鎖的形態：估計器先鎖對，十幾個 hop 後改鎖
一個更早的值（pre-echo 誤歸因）。它同樣是持續、自洽的錯答案，`MATCHED`
專屬。`delay_backward_quarantine_enabled`（§6.4，預設關閉）能把這種誤鎖
延後一個可設定的窗，但**不能治好它**；機制與實測見
`delay_estimator_design_zh_TW.md` §2。

延遲估計**不能**修正這些問題：mic/ref 取樣率不同、持續的 clock drift、
送錯的 reference 訊號。這些必須在上游解決。

離線處理且檔案已精準對齊時，可設
`cfg.delay_mode = AEC_DELAY_EXTERNAL_ALIGNED`（舊寫法 `cfg.enable_delay_est = 0`
效果相同）省掉首次擷取的擾動。線上產品請維持 `MATCHED`。

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

**記憶池大小會隨 `delay_mode` 與 `delay_num_filters` 改變**，因為 matched
filter bank、降取樣 render ring 與兩個 lag histogram 都是依 init 設定從記憶池
切出來的（不再是編譯期最大值）。下表是 KISS backend 的實測值：

| 格點 | n=1 | n=2 | n=3 | n=4 | n=5（預設） |
|---|---:|---:|---:|---:|---:|
| 8 kHz / 256（legacy） | 252,768 B | 258,496 B | 264,224 B | 269,952 B | 275,680 B |
| 16 kHz / 256 | 356,864 B | 362,592 B | 368,320 B | 374,048 B | 379,776 B |
| 16 kHz / 512 | 485,936 B | 491,664 B | 497,392 B | 503,120 B | 508,848 B |
| 48 kHz / 1024 | 1,144,144 B | 1,149,872 B | 1,155,600 B | 1,161,328 B | 1,167,056 B |

NE10 backend 在每一格都少 608 B（16 kHz/512 與 48 kHz/1024 少 1,376 B / 2,912 B）；
差異來自 FFT handle，與 delay 設定無關。

> **這些絕對值會隨 `Aec` / `AecConfig` 的結構大小整批移動。** 上表含 duty
> census 的兩個 `unsigned long long` 計數器（§8.5）與 backward-jump
> quarantine 的兩個設定欄位＋倒數欄位（§6.4）；`sizeof(AecConfig)` 目前是
> **204 B**。任何一次結構欄位增減，**每個 mode／n／格點的絕對 byte 數都同幅
> 移動，而所有 delta 不變**（每個 filter 5,728 B、`FIXED` 的 byte/ms 公式、
> mode 之間的差值）。所以舊呼叫端不能沿用先前記下的常數，**必須重新查詢
> `aec_get_mem_size()` 並重新 init**（見 CHANGELOG ABI 段）。
>
> 本節的絕對值於 2026-08-19 以 `make test-delay-num-filters`（KISS/NE10 各跑
> 一次）重新量測。這一輪同時**校正**了文件內部原本就不一致的地方：§4／§5 的
> 表格落後一代（16 B），§7 `--print-mem-size` 的範例輸出落後兩代（32 B）。
> 兩處現在都等於同一次量測的結果。

每少一個 matched filter 固定省 **5,728 B**（每一格點都一樣）：coefficients
2,048 B ＋ accumulated error 512 B ＋ render ring 1,536 B ＋ highest-peak
histogram 1,536 B ＋ pre-echo histogram 96 B。所以 `n=1` 相對預設 `n=5` 省
22,912 B。

非 `MATCHED` 模式不配置 estimator，而且**參考訊號環形緩衝的大小也依 mode
不同**（16 kHz/256 KISS）：

| `delay_mode` | 記憶池 | 相對 `MATCHED n=5` |
|---|---:|---:|
| `MATCHED` n=5（預設） | 379,776 B | — |
| `MATCHED` n=1 | 356,864 B | −22,912 B |
| `FIXED`，`fixed_delay_samples = 0` | 215,280 B | −164,496 B |
| `FIXED`，`fixed_delay_samples = 400`（25 ms） | 216,880 B | −162,896 B |
| `FIXED`，`fixed_delay_samples = 1600`（100 ms） | 221,680 B | −158,096 B |
| `FIXED`，`fixed_delay_samples = 8000`（500 ms） | 247,264 B | −132,496 B |
| `EXTERNAL_ALIGNED` | 214,768 B | −165,008 B |

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

各格點的 `FIXED` 實測值（KISS backend，balanced preset）：

| 格點 | fixed=0 | 25 ms | 100 ms | 500 ms |
|---|---:|---:|---:|---:|
| 16 kHz / 256 | 215,280 B | 216,880 B | 221,680 B | 247,280 B |
| 16 kHz / 512 | 344,864 B | 346,464 B | 351,264 B | 376,864 B |
| 48 kHz / 1024 | 740,592 B | 745,392 B | 759,792 B | 836,592 B |

> **注意**：`fixed_delay_samples` 很大時 `FIXED` 也可能比 `MATCHED n=5` 更
> 耗記憶體（48 kHz/1024、2500 ms 為 1,220,576 B，高於 `MATCHED n=5` 的
> 1,167,056 B）——環形緩衝終究要放得下那個延遲。要用大 fixed delay 又要省
> 記憶體，正確作法是讓呼叫端自己補償掉大延遲後改用 `EXTERNAL_ALIGNED`。

另外一個明顯的開關（16 kHz/256 KISS，相對 379,776 B）：

| 設定 | 記憶池 | 差異 |
|---|---:|---:|
| `enable_shadow = 0` | 347,216 B | −32,560 B |

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
| `fft_size` | `int` | 8k→256、16k→256、48k→1024（由 `aec_config_defaults()` 填入，＝`aec_default_fft_size()`） | 須與 `sample_rate` 成對（見 §1）；**`0`（自動）會被拒絕** | 只有 16 kHz 有第二選擇（512）。改成 512 會讓 hop 變 16 ms、延遲加倍，但每秒呼叫次數減半。核心不會替你猜格點——所有入口都走同一支 `aec_resolve_signal_grid()`。 |
| `filter_length` | `int` | `sr × 52/1000`（48 kHz 為 `sr × 64/1000`）→ 8k:416、16k:832、48k:3072 | 1 – 4096 | **單位是 sample，不是 ms。** 涵蓋的回音路徑長度。大房間／長殘響時調高，代價是記憶體與運算量上升。 |
| `n_partitions` | `int` | `0`（自動） | `0` 或 1 – 256 | 留 `0` 即可，由 `filter_length` 與 hop 自動推導。 |

### 6.2 模組開關（全部只接受 0 或 1）

| 欄位 | 預設 | 說明／何時調整 |
|---|---:|---|
| `enable_res` | `1` | 殘留回音抑制。關閉後輸出是純線性 filter 殘差、延遲降為 0，但回音殘留明顯變多。診斷時可關（見 §9）。 |
| `enable_cng` | `1` | 舒適噪音。若下游還有另一套噪音處理／CNG，**請關掉這個**，兩層疊加會讓噪音底變得不自然。 |
| `enable_delay_est` | `1` | **DEPRECATED**——`delay_mode` 的相容轉譯層（見 §3 延遲契約與 §6.4）。設 0 等同 `EXTERNAL_ALIGNED`（或帶 `fixed_delay_samples` 時等同 `FIXED`）。`EXTERNAL_ALIGNED` 會少掉 165,008 B 記憶體（16 kHz/256，見 §5 記憶池表）。新程式請直接寫 `delay_mode`。 |
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
| `delay_mode` | `AecDelayMode` | `AEC_DELAY_MATCHED` | `MATCHED` / `FIXED` / `EXTERNAL_ALIGNED` | **對齊方式的唯一真實來源**（見 §3 延遲契約的三態表）。init 時決定、instance 存活期間不可變（各 mode 的 pool 佈局不同，要改必須 destroy + 重新 init）。非法組合一律拒絕。 |
| `fixed_delay_samples` | `int` | `-1` | `-1`，或 `FIXED` 時 0 – `120 × sample_rate` | **只在 `FIXED` 有意義**：bring-up 量測到的系統延遲，單位是 native-rate sample。其他 mode 必須維持 `-1`（帶值會被拒絕，不會被忽略）。它同時**決定參考環形緩衝的大小**：`FIXED` 只配置 `fixed_delay_samples + hop_size` 個 sample（緊上界，見 §5 記憶池表的公式），`max_delay_ms` / `delay_buffer_ms` 在此 mode 無效。 |
| `max_delay_ms` | `float` | `1024.0` | 0 – 60000 | **`MATCHED` 專用**：參考訊號搜尋環形緩衝的尺寸下限（ms），不是搜尋上限（見 §3 延遲契約）。調高只是多配記憶體，不會提高可擷取的延遲。`FIXED` / `EXTERNAL_ALIGNED` 下完全無效（見 §5 環形緩衝大小公式）。 |
| `delay_buffer_ms` | `float` | `2048.0` | 0 – 120000 | **`MATCHED` 專用**：參考訊號搜尋環形緩衝長度（ms）。實際樣本數取 `delay_buffer_ms` 與 `max_delay_ms + 4096 samples` 兩者較大者。調高 `max_delay_ms` 時通常要一併調高。`FIXED` / `EXTERNAL_ALIGNED` 下完全無效。 |
| `delay_est_init_s` | `float` | `0.3` | 0 – 3600 | 延遲估計值需維持穩定多久才視為已鎖定（秒）。 |
| `delay_est_period_s` | `float` | `0.5` | 0 – 3600 | 鎖定後的重新確認週期（秒）。回音路徑常變動（裝置會移動）可調短。 |
| `delay_num_filters` | `int` | `5` | 1 – 5（**只在 `MATCHED`**；其他 mode 必須維持 5） | Matched-filter bank 大小（**算力旋鈕**）。可靠搜尋上限隨之縮小：n=1→125ms／2→221ms／3→317ms／4→413ms／5→509ms；每少一組省 ~4.2 MMAC/s 的 full-rate 搜尋算力，**並省 5,728 B 記憶池**（bank、render ring、兩個 histogram 都依 n 從記憶池切出，見 §5 記憶池表）。縮小的正確場景是**上游已先提供 coarse-compensated far**（HAL/系統層自行移除 bulk delay 後才餵進來）、matched filter 只需追殘差的 `MATCHED` 部署——注意這與 `FIXED` 無關：`FIXED`（`fixed_delay_samples`）根本不建 matched filter，`EXTERNAL_ALIGNED` 連 estimator/ring 都沒有。bench/dataset 一律維持 5（見 `delay_estimator_design_zh_TW.md` §5）。`FIXED` / `EXTERNAL_ALIGNED` 根本沒有 matched filter，改這個值會被拒絕（不是忽略）；`0` 在任何 mode 都是錯誤，不是「關閉延遲估計」的入口。 |
| `delay_acquire_protect_converged` | `int` | `1` | 0 / 1 | filter 已收斂時，保護它不被**首次**延遲擷取破壞（Path A）。 |
| `delay_backward_quarantine_enabled` | `int` | `0` | 0 / 1 | 上一項的姊妹旗標，管的是**已經鎖定之後的延遲變更**（Path B），預設**關閉**（關閉時輸出與未加此機制的版本逐位元相同）。開啟後只隔離**一種**候選：比現行延遲**更早**（backward／pre-echo 方向）、且線性 filter 在**目前對齊上仍明顯在消除回音**的新估計。候選還必須先通過 Path B 本來的兩個門檻（與現行延遲相差 > 32 樣本、`delay_aec3_confidence()` ≥ 0.5），所以隔離只會花在「不隔離就會被接受」的候選上。**往後跳（延遲變大）永不隔離**，pre-echo 誤歸因不會產生那個方向；**首次取得對齊也不受影響**（那是 `delay_acquire_protect_converged` 的事）。隔離**有上限**（見下一列）：到期就採用。判別依據是 `aec_linear_is_cancelling()`——windowed ERLE > 2.5 dB **或**近期 inst-ERLE 峰值 > `delay_acquire_inst_erle_db`；兩個讀數在其機制未跑過時都是 0，因此指標不存在時自動失效（fail-open），不會誤擋。消除能力崩掉時**立即**放行，不必等到期；這是 hard replacement 常見但不是所有真實路徑變化都必然具備的現象。**沒有「佔優」提前放行**：estimator 對外只有 0／0.5／1.0 三級 confidence，那是 histogram 一致性、不帶品質資訊（誤鎖候選一樣會到 REFINED），Path B 本來就已經在閘它，所以「候選一直被提出」本身就是唯一可用的持續性證據，而窗長就是在量它。**它只延後、不治癒**：誤鎖被推遲一個窗，真正的 pre-echo 修復是 estimator 本身的工作。方向判斷與上限兩者都是必要條件：不看方向、又不設上限的判別式（「只要還在消除就拒絕所有不同候選」）在 multipath／新增路徑場景會無限期否決真正的新路徑（實測舊/新增益 0.4/0.5 到 hop 850 仍不重鎖、0.5/0.5 永不重鎖），現行判別式則四種增益全部重鎖、最壞只慢一個窗。**是否開啟屬部署決策**，需先用實機音檔確認該裝置真的會出現這種誤鎖再開；預設 OFF 是 byte-exact 的安全值。`FIXED` / `EXTERNAL_ALIGNED` 不會重新決定對齊，此旗標對它們無作用。 |
| `delay_backward_quarantine_s` | `float` | `1.0` | 0 – 3600 | 上一項的隔離窗長（秒），在 `aec_create()` 時按解析後的 hop／取樣率換算成 estimation cycle 數（最小 1，所以小於一個 hop 的窗仍會隔離整整一個 cycle；要關閉請用上一列的 enable，不是把這裡設 0）。enable 為 0 時完全無作用。**預設 1.0 秒的依據**（16 kHz／hop 256 = 62 hops）：下限來自誤鎖動態——pre-echo 重鎖只在首次鎖定後 17 個 hop 就發生，窗必須明顯長於它才涵蓋得到誤鎖最具破壞力的那段（62 是 17 的 3.6 倍）；上限來自真實變化的成本——在多路徑測試場景中 estimator 自己產生新候選就要 488–1515 個 hop，多等 62 個 hop 是其中的 4–13%，而且只有在消除能力**沒有**崩掉時才需要等（崩掉就立即放行）。實測（16 kHz／fft 512／hop 256，true 6400 誤鎖 4800）：窗 0.5/1.0/2.0/4.0 s = 31/62/125/250 hops，誤鎖採用點就從 hop 49 精確移到 80/111/174/299——「到期即採用」是唯一的放行機制，沒有永久 veto 的路徑。 |
| `delay_acquire_warm_transfer` | `int` | `1` | 0 / 1 | 首次擷取到延遲時，把已學到的回音模型平移過去而不是歸零，避免開場一秒左右出現一段明顯回音。建議維持開啟。 |
| `delay_acquire_inst_erle_db` | `float` | `4.0` | -100 – 100 | 上一項的觸發門檻（dB）：既有模型要夠好才值得平移。 |
| `delay_acquire_protect_inst_erle` | `int` | `0` | 0 / 1 | 另一種較保守的做法（直接擋掉重新對齊），預設關閉。 |

> `delay_backward_quarantine_*` 的「有上限」指**同一段連續符合條件的
> backward episode**：候選轉為 forward／無效、confidence 中斷，或
> cancellation 證據消失時會解除本次隔離；之後再出現的 backward
> episode 會重新計時。Cancellation collapse 常見於路徑完全替換，但
> 不是 multipath／新增路徑的必然特徵。

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

### 6.6b 逐階段耗時（`aec_get_last_timing`）

```c
typedef struct AecStageTiming AecStageTiming;   /* delay_us / frontend_us / linear_us / res_us */
void aec_get_last_timing(const Aec* a, AecStageTiming* out);
```

最近一個 hop 內四個最大階段的 wall-clock 成本（微秒，預設用
`CLOCK_MONOTONIC`）。純診斷：引擎自己不回讀，處理結果不受影響。`a` 為 `NULL`
時把 `out` 清零而不是失敗；`out` 不得為 `NULL`。

- `delay_us`：延遲估計與環形對齊——matched-filter bank、它周圍的 duty-cycle
  記帳、參考環寫入與補償讀出。`AEC_DELAY_EXTERNAL_ALIGNED` 下恆為 0（沒有環也
  沒有估計器），`AEC_DELAY_FIXED` 下很小（有環，但 matched filter 不跑）。
  `AEC_DELAY_MATCHED` 下這一項經常是整個 hop 最大的單一階段，所以它獨立回報而
  不是併進 `frontend_us`。
- `frontend_us`：進入點到主濾波器之前的其餘部分——mic HPF、飽和偵測、
  render activity、`mu_scale`、mic-clip、RSA 更新，以及 shadow filter。
  **不含** `delay_us`，兩者不會重複計算。
- `linear_us`：主自適應濾波器；若本實例自己算 far-end FFT（而不是透過
  `aec_process_context_shared_far()` 借用），FFT 也算在這裡。
- `res_us`：AEC3 post/RES 區塊。只要 `enable_res` **或** `return_res_context`
  其中之一為真就會跑，所以 context-only 的呼叫端也會看到真實數字。

四者**刻意不等於**整個呼叫：餘額是 stationarity refresh、`e2_coarse`/ERL
publish、DT analyzer、EPV、`shadow_rise`、misadjustment estimator、power EMA
與收斂偵測。要呈現完整分解的呼叫端得自己把餘額算出來。

每次 `aec_process*()` 進入時清零，所以一個 hop 只會回報它真的跑到的階段。
解析度是微秒，低於 1 µs 的階段會讀到 0。

#### 預設關閉

⚠ **量測要建置時開啟**，否則四個欄位每個 hop 都讀 0：

```bash
make PROFILE=1              # 等同 EXTRA_CFLAGS=-DAEC_STAGE_TIMING=1
```

每 hop 每實例七次 clock 讀取，所以 release build 一次都不做——會被切掉的診斷
不會不小心出貨。那個 0 就是呼叫端分辨「沒量」與「量了但很快」的唯一線索；
想要分解的消費端必須建置**函式庫**時就帶旗標，只開自己的顯示旗標沒有用。

`EXTRA_CFLAGS` 與 `PROFILE` 都折進 config hash，所以 profile build 與 release
build 各有自己的 `bin/<backend>-<hash>/`，不會互相交出過期的 object。

#### 目標平台沒有 `CLOCK_MONOTONIC` 時

`clock_gettime(CLOCK_MONOTONIC, ...)` 是 POSIX 而非 C99，精簡 libc 的目標平台
不一定有。這種平台這樣開：

```bash
make PROFILE=1 EXTRA_CFLAGS='-DAEC_NOW_US=board_timer_us -include my_timer.h'
```

`AEC_NOW_US` 要是**純識別字**（本 repo 的 FP policy 不允許 `EXTRA_CFLAGS` 出現
括號），指向一個不收參數、回傳 `uint32_t` 微秒的函式，宣告由整合者自備。指定
之後預設實作既不編譯也不連結，`<time.h>` 也不會被 include。

替代時鐘必須**單調**：會倒退的 stamp 會讓無號減法產生接近 32-bit 全距的荒謬
數值，而不是一個小的錯誤值。指向一個永遠回傳同一個值的函式是合法用法——每個
階段都讀 0，只剩對一個沒人回讀的結構做幾次純量寫入。

兩支包裝這個函式庫的 pipeline 各自有同名慣例的覆寫點
（`AUDIO_PIPELINE_NOW_US`、`FOUR_AEC_NR_RES_NOW_US`），刻意不共用：一個元件的
時鐘屬於該元件自己的建置契約。

### 6.7 靜態記憶體拆解（`aec_get_mem_breakdown`）

```c
typedef struct AecMemBreakdown {
    size_t total_bytes;       /* == aec_get_mem_size(cfg)                 */
    size_t estimator_bytes;   /* delay_aec3_get_mem_size(); 0 非 MATCHED  */
    size_t ring_bytes;        /* 對齊環形緩衝；0 為 EXTERNAL_ALIGNED     */
} AecMemBreakdown;

int aec_get_mem_breakdown(const AecConfig* cfg, AecMemBreakdown* out);
```

把 §4 記憶池大小表格背後的算法變成一個可查詢的 API：`total_bytes` 就是
`aec_get_mem_size(cfg)` 本身（同一次呼叫算出來，不是另一套算式），
`estimator_bytes`／`ring_bytes` 是這個總量裡兩個有名字的子集，用的是
`aec_get_mem_size()` 內部本來就在用的同一批 helper（`delay_aec3_get_mem_size()`、
`aec_derive_dims()` 的 ring 大小 out-param）——查詢這個函式不可能得到與
`aec_carve()` 實際切出來的 pool 佈局不同的答案。失敗條件與
`aec_get_mem_size()` 完全一致：`cfg` 為 `NULL` 或未通過 `aec_validate_config()`
時回傳 0（並把 `*out` 清零）。`example/aec_wav.c` 的 `--print-mem-size`
（附錄 A）就是這個 API 的一個瘦封裝。

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

### 7.1 執行期切換 preset（`aec_set_preset`）

```c
int aec_set_preset(Aec* a, AecPreset preset, float ramp_ms);
```

不需要重建實例。三個 preset 只差 `min_gain_floor_far_active_db`，而該欄位是抑制器
每個 hop 只讀一次的單一純量下限，所以換 preset 是**重新指定目標值**，不是重建：
濾波器、延遲狀態、每一條平滑歷史、far-active latch 與 DominantNearend 計數器全部
原封不動繼續跑。

```c
#include "aec.h"

AecConfig cfg;
aec_config_from_preset(&cfg, AEC_PRESET_BALANCED, 16000);
Aec* a = aec_init(pool, aec_get_mem_size(&cfg), &cfg);

for (;;) {
    /* 使用者把強度旋鈕轉到 aggressive：在兩個 hop 之間呼叫，
     * 100 ms 內以 dB 線性走到新的地板 */
    if (ui_strength_changed())
        (void)aec_set_preset(a, AEC_PRESET_AGGRESSIVE, 100.0f);

    aec_process(a, mic, ref, out);   /* 各恰好 hop 個 float32 */
}
```

`ramp_ms` 語意：

| `ramp_ms` | 行為 |
|---|---|
| `0.0f` | 下一個 hop 就套用，**不是錯誤**。落點與「用該 preset 從頭建一個新實例」持有的 float 完全相同 |
| `> 0.0f` | 以 dB 為單位線性走過去，每次 hop 走一步，最後**精確落在**目標值（不是漸近逼近）。上限 60 秒，超過即拒絕 |

有 ramp 的理由：mild ↔ aggressive 是 18 dB 的落差，而這個地板是硬性 clamp，
下游沒有任何東西會替它平滑。100 ms 是合理的起點，但這是呼叫端的選擇，
函式庫沒有寫死任何政策。

**回傳與拒絕**：成功回 `0`。以下情況回 `-1` 且**什麼都不寫**——`a == NULL`、
`preset` 超出 enum 範圍、`ramp_ms` 非有限值或超出 `[0, 60000]`。注意這裡與
`aec_config_from_preset()` 的差異：那是 config factory，遇到不認得的值**退回
balanced**；這是 setter，有呼叫端可以回報，所以**直接拒絕**。

**呼叫紀律**：在兩個 hop 之間呼叫，與 `aec_process()`（或你實際使用的處理進入點）
序列化。**非 thread-safe**。ramp 進行中再呼叫一次，會從**當前的 live 值**重新起走，
既不會彈回舊目標、也不會沿用舊的步進比例。`aec_reset()` 保留目標值、丟棄 ramp 進度
（目標存在 config、進度存在 state，這是刻意分開的）。

**底層 primitive**。真正做事的是抑制器自己的入口，它是 public 的，因為四麥克風產品
共用的那個 post 級抑制器是一個沒有任何 `Aec` 擁有的 `SuppressionGain`：

```c
#include "suppression_gain.h"   /* §「你需要 include 的 header」列的進階 header */

int suppression_gain_set_split_floor_far_active_db(SuppressionGain* sg,
                                                   float db, float ramp_ms);
```

`db` 就是 `AecConfig.min_gain_floor_far_active_db` 那個 dB-power 量，並且用**驗證器
自己的**範圍 `[-300, 50]` 檢查——執行期呼叫不可能裝進一個 init 當初會拒絕的值。
`ramp_ms` 語意與上表相同。它同樣不動任何平滑歷史、latch、DominantNearend 計數器
或 `initial_state`。回 `0` 或 `-1`（NULL 或引數超出範圍），`-1` 時什麼都不寫：
所有值都先算完、驗完，才寫第一個 store。

**四麥克風產品請勿對 lane 呼叫**。四條 lane 都以 `spatial_linear_context` 建立，
根本不會走到 `suppression_gain_get_gain()`，所以對 lane 呼叫 `aec_set_preset()`
**依建構方式即為無效操作**——塑形輸出的 gain 來自該產品自己那一個共用的 post 級
抑制器。請改用四麥克風核心自己的 setter（`four_aec_nr_res_set_aec_preset()`），
它針對的就是那個共用實例。

**A/B 量測時該預期什麼**（先讀這段再下結論）：far-active 地板只在
**far-active 且非 double-talk** 的 hop 上生效。double-talk 期間套用的是 DT 地板，
而 DT 地板在三個 preset 之間**完全相同**；far-active latch 尚未觸發之前，
套用的是 far-silent 地板。同一個 gain 還決定注入的 comfort noise 量
（振幅正比於 `sqrt(1 − G_res²)`，所以地板壓得越深、CNG 反而越多）。三件事合起來
的結果是：**整段錄音的平均值移動幅度會小於 dB 落差所暗示的量**，而且一個只量
echo／degradation 的 A/B 會把 CNG 的變化錯記到別的機制頭上。請在 echo 對齊
或 degradation 對齊的條件下比較，並實際試聽。

Python 對應：`AEC.set_preset(preset, ramp_ms=0.0)`（同樣的語意；不合法的 preset 或
`ramp_ms` 丟 `ValueError`，實例不變）。回歸測試：`make test-runtime-preset`
與 `python/tests/test_runtime_preset.py`。

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

### 8.5 Duty-cycle engagement census

`AecDebugStatus`（§8，`aec_debug_status()`）多了兩個計數器：

```c
unsigned long long duty_hops_total;  /* 經過 delay 區塊的 hop 數        */
unsigned long long duty_hops_run;    /* 其中真的跑了 matched filter 的數 */
```

背景：duty-cycle 機制（`Aec` struct 裡 `duty_*` 欄位的說明）在估計穩定
（confidence 1.0）且 `delay_est_init_s` 秒不變後，把 matched-filter
分析降到 1-in-K，其文件註解宣稱這樣可省下「約 90% 的 matched-filter
成本」——這是一個**設計數字**，只有機制真的完成降頻的 hop 才會實現；估計
持續變動的回音路徑上，機制每次都要重新武裝，實際降頻可能遠低於設計值。
這兩個計數器把**實測**engagement（`duty_hops_run / duty_hops_total`）變成
每次執行都能讀到的數字，取代原本只能假設的設計估計值。

- 只在 `delay_mode == AEC_DELAY_MATCHED` 時累積（`has_delay == 1` 才會進入
  計數的程式路徑）；`FIXED`／`EXTERNAL_ALIGNED` 從未建過 matched filter，
  兩個計數器永遠是 0——不是「估計器跑了但沒省到」，而是根本沒有機制可省。
- 累積範圍是「自 `aec_create`／`aec_init` 或最近一次 `aec_reset()` 以來」，
  與 `AecDebugStatus` 其餘欄位的累積規則一致；`aec_reset()` 會把兩個計數器
  歸零，因為跨越 reset 的比值會把兩段不同的 duty 狀態混在一起，沒有意義。
  在計數的 call site（消耗 `run_filter` 的那一行）遞增，不可能與實際執行
  的機制脫鉤。
- **純診斷**：兩個計數器只被寫入、從未被函式庫自己讀回，加起來是熱路徑上
  兩個整數遞增，不影響任何音訊輸出（byte-exact 證明見 CHANGELOG 產品化
  A 線 step 4 條目）。
- `example/aec_wav.c` 每次執行結尾印一行（附錄 A「Duty census 診斷輸出」）。
- 永久回歸測試：`make test-duty-census`（`test/test_duty_census.c`）。

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
| 若有開 `delay_backward_quarantine_enabled`（預設 0） | 路徑**新增**（舊反射還在、消除能力沒崩）時，往前跳的新對齊最多晚一個 `delay_backward_quarantine_s` 才被採用；覺得恢復慢就調短窗長或關掉這個旗標（見 §6.4） |

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
| `--delay-mode {matched\|fixed\|external}` | 覆寫 `cfg.delay_mode`（預設 `matched`）；`external` 對應 C API 的 `AEC_DELAY_EXTERNAL_ALIGNED` |
| `--delay-num-filters <n>` | 覆寫 `cfg.delay_num_filters`（僅 `matched` 合法，1–5，預設 5） |
| `--fixed-delay <samples>` | 覆寫 `cfg.fixed_delay_samples`（僅 `fixed` 合法，native-rate sample，≥0） |
| `--print-mem-size` | 印出 resolved grid／mode／n 與 static-pool byte 拆解後**直接退出**，不開音檔輸出、不跑任何一個 hop（見下方「`--print-mem-size` 輸出」） |
| `--debug-level <0..3>` | 0=關閉、1=摘要、2=每 frame、3=完整 |
| `--debug-log <path>` | 將 log 導向檔案（預設 stderr） |
| `--debug-trace <path>` | 每 frame 的 CSV trace |
| `--debug` | 每秒印一行 `aec_debug_status()` 狀態 |

> 兩個容易誤解的地方：**沒有** `--filter-length` 或 `--filter-length-ms` 這個旗標
> （`filter_length` 只能透過 C API 設定）；`--cng` 不是「開啟一個預設關閉的功能」，
> 舒適噪音**預設就是開的**，`--cng` 與 `--no-cng` 只是明確覆寫。

> `--delay-mode`／`--delay-num-filters`／`--fixed-delay` 三個旗標的值**未經
> CLI 驗證就直接寫入 `AecConfig`**——`aec_validate_config()` 是唯一的範圍與
> mode／欄位相容性權威（§2.1），CLI 不做第二次檢查也不做 clamp。組合非法
> 時（例如 `--fixed-delay` 給負值、`--delay-mode fixed` 沒給 `--fixed-delay`、
> `--delay-mode external` 卻改了 `--delay-num-filters`）`aec_create()` 會
> fail-fast 印出訊息後以離開碼 4 結束，不會被靜默忽略或改成別的設定。

### `--print-mem-size` 輸出

不開音檔、不 `aec_create()` 一個真正的實例，只呼叫 `aec_get_mem_breakdown()`
（§6.7）後印一行到 **stdout**（其餘旗標一律印到 stderr）：

```text
mem: sr=16000 fft=256 hop=128 mode=matched n=5 fixed_delay_samples=-1 total_bytes=379760 estimator_bytes=33936 ring_bytes=131072
```

欄位依序：resolved 後的 `sample_rate`／`fft_size`／`hop_size`、`delay_mode`、
`delay_num_filters`、`fixed_delay_samples`、`aec_get_mem_size()` 回傳的總
byte 數、`delay_aec3_get_mem_size()` 回傳的 estimator byte 數（非 `matched`
時為 0）、對齊環形緩衝的 byte 數（`external` 時為 0）。三個 byte 欄位與 §4
表格是同一個 `aec_get_mem_breakdown()` 呼叫算出來的，數字對不上就是文件
過期，以實際輸出為準。組合非法時走與 `aec_create()` 相同的 fail-fast 路徑
（離開碼 4），不會印出任何 `mem:` 行。

```bash
"$BIN"/aec_wav mic.wav ref.wav /dev/null --print-mem-size --delay-mode fixed --fixed-delay 1600
```

### Duty census 診斷輸出

正常處理完（非 `--print-mem-size`）結尾，`aec_wav` 會在既有的 `Processed ...`
摘要行之後多印一行 matched-filter duty-cycle **實測**（非估計）engagement
（產品化計畫 §7 量測方法要用；見 §8.5）：

```text
duty: matched_filter_ran=379/1250 hops (30.32% engagement, 69.68% saved)
```

`delay_mode != matched` 時（`fixed`／`external`，或舊式
`--no-delay-est`）從未建過 matched filter，沒有成本可省，印：

```text
duty: matched_filter_ran=0/0 hops (n/a -- no matched-filter estimator in delay_mode=fixed)
```

診斷用途，不影響音訊路徑（見 §8.5 的 byte-exact 證明）。

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
