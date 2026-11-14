# AEC C 使用手冊

[文件首頁](html/index.html) · [生命週期與錯誤碼](reference/lifecycle.md) · [設定](reference/configuration.md) · [Level](reference/levels.md) · [後級串接與故障排查](reference/integration.md) · [WAV 工具](reference/wav_cli.md)

## 你需要 include 的 header

```c
#include "aec.h"        /* c_impl/include/aec.h — 標準整合只需要這一個 */
```

整合只需要 `aec.h`：它同時宣告單聲道與陣列實例的全部公開介面，包括 `aec_apply_cng_spec()` 舒適噪音填補
（[§8.5](reference/integration.md#85-舒適噪音填補aec_apply_cng_spec)；自己組合頻域後處理鏈時用）。
殘留回音抑制增益階段不是獨立的公開介面：單聲道的在 `Aec` 裡，四麥克風融合後的在陣列實例裡。

`c_impl/include/` 只放一個 header：`aec.h`。`c_impl/include/internal/` 放其餘全部：`aec_internal.h`（給 pipeline 用的函式庫內部介面）、
`aec_balanced_config.h`（產生的各取樣率表）、`aec_post.h`、`pbfdkf.h`、`delay_aec.h`、`suppression_gain.h` 等內部實作 header，
不屬於公開介面；只用公開 API 的整合者只需要 `-I include`；`Aec` 是不透明型別
（`typedef struct Aec Aec;`），呼叫端拿不到結構內容，也不能以值持有或取 `sizeof(Aec)`。

`aec.h` 本身會 include 共用 DSP 函式庫（`audiolib`）的 `fft_wrapper.h`，所以編譯時兩個 include 路徑都要給（見 §2）。

---

## 1. 這個函式庫是什麼／不是什麼

聲學回音消除（AEC）。每個 `Aec` 實例處理**一支麥克風 + 一路參考訊號**（`AecConfig.n_mics = 4` 時是四支麥克風的陣列實例，見下方 §2 末的「四麥克風陣列實例」）：

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

填值的正確做法只有一條鏈：`aec_config_from_level()`
是**唯一**會替你填預設格點的地方（8k→256、16k→256、48k→1024，同
`aec_default_fft_size()`），填完之後核心不再猜第二次。要用 16 kHz 的 512
格點，就在 `aec_create()` 前自行覆寫 `cfg.fft_size = 512`。CLI／上層應用
可以替沒指定的使用者選產品預設，但**呼叫函式庫前必須先 resolve 成非零
`fft_size`**（`example/aec_wav.c` 就是這樣做的）。

查詢用 API：

- `aec_resolve_signal_grid(sample_rate, fft_size, &grid)` —— 全函式庫唯一
  的格點解析器。成功回傳 `AEC_OK`（0）並填入 `AecSignalGrid{sample_rate, fft_size,
  frame_size, hop_size, n_freqs, is_legacy}`；失敗回傳 `AEC_ERR_ARG` 且**不動** `*out`。
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

> 8 kHz 是 **legacy 格點**：本函式庫與 Audio_ALG 的 mono 範例支援它
> （既有 8 kHz 測試路徑仍在用），但它不是產品正式格點，Audio_ALG 的
> 4 聲道範例也不支援；`n_mics = 4` 的陣列實例（`aec_get_mem_requirements()`／`aec_init_ex()`）只接受 16k/256、16k/512、48k/1024
> 三個產品格點。`AecSignalGrid.is_legacy` 會為它、且只為它設為 1。

### 硬性限制

| 限制 | 說明 |
|---|---|
| **麥克風數只有 1 或 4** | 一般實例 = 1 mic + 1 ref；`n_mics = 4` 的陣列實例 = 4 mic（交錯成 `float[hop][4]`）+ 1 ref，beamformer 在函式庫外（§8.8）。沒有其他麥克風數、沒有 stereo reference。其他路數請各自建立獨立實例。 |
| **延遲只能是正的** | 回音必須**晚於**參考訊號抵達（`mic` 落後 `ref`）。內部是對參考訊號的環形緩衝做回看，負延遲無法表示，必須在上游先對齊。 |
| **沒有非線性回音模型** | 只有線性 filter + 殘留回音抑制。喇叭嚴重失真、破音造成的非線性回音無法建模。 |
| **取樣率限 8/16/48 kHz** | 其他取樣率請在外部先重新取樣。 |
| **mic/ref 必須同一時鐘** | 兩路取樣率要相同且同步。長時間的 clock drift 會讓延遲估計與 filter 收斂失效，函式庫不負責補償。 |
| **不是 thread-safe** | 單一實例不可被多執行緒同時呼叫；處理進入點之間、`aec_reset()` 與 `aec_destroy()` 都要由呼叫端序列化。 |

---

## 2. Quick start

### 建置

`audiolib` 需與本 repo 同層（sibling）。兩邊都要建：

```bash
make -C ../audiolib/common lib     # 共用 DSP 函式庫
make -C c_impl          lib     # 本函式庫 libaec.a
```

> FFT 後端只有 NE10（ARM NEON）。`BACKEND` 預設且唯一合法值是 `ne10`，指定其他值
> 兩個 repo 的 Makefile 都會直接報錯；`SIMD=0` 則是同一後端改走 NE10 的純量 C kernel。

產出路徑會依編譯參數雜湊命名，用各自的查詢目標取得，不要自己拼路徑：

```bash
AEC_LIB="$(make -s -C c_impl          print-lib-path)"
COMMON_LIB="$(make  -s -C ../audiolib/common print-lib-path)"
```

### 連結

```bash
cc -std=c99 -O2 -ffp-contract=off app.c \
   -Ic_impl/include -I../audiolib/common/include \
   "$AEC_LIB" "$COMMON_LIB" -lm -o app
```

`-ffp-contract=off` 是**必要**的：允許 FMA 合併會改變遞迴狀態（HPF、偵測器累加器）
的數值，導致行為與驗證過的版本不一致。`libaec.a` 不含共用 DSP 的目的檔，
`libaudio_common.a` 一定要一起連結。`libaudio_common.a` 含一個 C++ 翻譯單元（NE10 的 generic-radix kernel），
所以 Makefile 以 C++ driver 連結（`LINK = $(CXX)`）；用 `cc` 連結時，若出現 C++ runtime 的未定義符號，
改用 `c++`／`g++` 當連結器，或補上該工具鏈的 C++ runtime 函式庫。

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
    Aec      *aec;
    float    *mic, *ref, *out;
    int       hop;

    aec_config_from_level(&cfg, AEC_LEVEL_BALANCED, sample_rate);
    aec = aec_create(&cfg);                          /* NULL = 設定無效或記憶體不足 */
    if (!aec) return -1;

    hop = aec_hop_size(aec);                         /* 每次剛好餵這麼多 sample */
    mic = (float *)malloc((size_t)hop * sizeof *mic);
    ref = (float *)malloc((size_t)hop * sizeof *ref);
    out = (float *)malloc((size_t)hop * sizeof *out);

    if (mic && ref && out) {
        while (app_read_hop(mic, ref, hop)) {
            aec_process(aec, mic, ref, out);         /* 回傳 AecStatus；引數有效時為 AEC_OK */
            app_write_hop(out, hop);
        }
    }

    free(mic); free(ref); free(out);
    aec_destroy(aec);                                /* heap 實例必須剛好呼叫一次，之後指標失效，不可再 free */
    return (mic && ref && out) ? 0 : -1;
}
```

### 四麥克風陣列實例

`cfg.n_mics = AEC_ARRAY_MICS`（4）時，一次 `aec_init()`／`aec_create()` 建出整條四麥克風線性流程（四條 lane、共用的 delay 估計器與對齊後 far、融合緩衝，同一塊記憶池）。
Beamformer 與合成**不在函式庫內**，由呼叫端提供。建立與記憶池描述子用 `aec_get_mem_requirements()` + `aec_init_ex()`；每個 hop：

1. `aec_process_linear(a, mics, ref)`：`mics` 是交錯的 `float[hop][4]`，`ref` 是 `float[hop]`；
2. `aec_get_array(a, &av)`：各 lane 的 `error_spec` 給呼叫端的 beamformer；
3. `aec_process_residual(a, &beam)`：`beam.weights` 是聲道優先的 `Complex[4][n_freqs]`（融合為 `Σ w · lane`，不取共軛），`beam.beamformed_error` 是同一次 beamformer 的頻譜或 `NULL`；
4. `aec_get_residual()` 的 `res_gain`／`aec_apply_gain(a, g_nr, spec)`／`aec_apply_cng(a, g_nr, spec)`（`FULL`），再由呼叫端反轉換與 overlap-add；`aec_post_reset_serial()` 變動時先清自己的 OLA 尾巴。

各 `residual` 模式的行為（`STATE` 的 `aec_process_residual()` 回 `AEC_ERR_MODE`、`FULL` 同 hop 第二次回 `AEC_ERR_SEQUENCE`、融合檢視的 `formed_hop`／`echo_spec` 為 `NULL`）、
`aec_process()`／`aec_apply_external_realign()` 在陣列實例上被拒絕、可編譯的完整範例、記憶池大小與目前尚未做到的項目（遠端側歷史與狀態仍每條 lane 各一份（只依賴遠端的逐 hop 分析已由 lane 0 算一次、其餘沿用，不改記憶池）），見
[§8.8](reference/integration.md#88-麥克風陣列實例n_mics--4) 與 [§4 陣列記憶池](reference/lifecycle.md#陣列實例的記憶池)。

---

## 3. 資料契約

### Sample 格式與尺度

- 型別一律是 `float`（32-bit），**不是** `int16`。
- 標稱範圍是 **[-1.0, +1.0]**。從 PCM16 轉入時除以 32768。
- 超出 [-1, 1] 不會被拒絕，但內部的飽和偵測（門檻 0.95）會判定為削波。
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


以下的延遲範圍**只適用於 `MATCHED`**（`FIXED` 沒有搜尋這回事，
環形緩衝自動配置成剛好夠用的 `fixed_delay_samples + hop_size`，
上限只受 `fixed_delay_samples` 的 `120 × sample_rate` 驗證上界限制）。

**可擷取的延遲範圍由 `delay_num_filters` 決定，與環形緩衝大小無關。**
這點容易誤解。下表是預設 `delay_num_filters = 5` 時搜尋範圍的幾何全 span：

| 取樣率 | 搜尋範圍的幾何全 span |
|---:|---:|
| 8 kHz | 1216 ms |
| 16 kHz | **608 ms** |
| 48 kHz | **608 ms** |

單聲道 `MATCHED` 的參考環是依 estimator 實際能回報的最大延遲配置的（`delay_aec_max_delay_samples()`，n = 5 時 8,148 sample，
`make test-delay-num-filters` 釘住）：8 kHz 約 1,018 ms，16 kHz 與 48 kHz 約 509 ms，也就是下文的「可靠上界」。

範圍來自 matched filter 的搜尋長度（內部 `delay_aec.h` 的
`DA_MAX_FILTER_LAG_FOR(n)` × 4 倍降取樣，n = `delay_num_filters`）。
48 kHz 輸入會先抗混疊降到 16 kHz，
估計結果再乘 3 回傳 native-rate samples，所以 wall-clock 範圍仍為
約 608 ms，不是把 9728 直接除以 48 kHz。此範圍與環形緩衝大小無關，
**環形緩衝的大小不會提高它**。（608 ms 是幾何全 span；若再計入
matched filter 的 reliability 條件 `lag < FILTER_SIZE-10`，實際可靠取得
peak 的上界約 509 ms——`nn_integration_interface.md` 引用的就是後者，兩
個數字定義不同、不矛盾。）
參考訊號環形緩衝的大小是內部決定的，只影響記憶體：單聲道 `MATCHED` 取 estimator 可回報的最大延遲加 `hop_size`
（`4 × ((n−1) × 384 + 501) + hop_size` sample，n = `delay_num_filters`；48 kHz 的第一項乘 3），所以它永遠放得下 estimator 報得出的延遲；
陣列實例的共用環取 `max_delay_ms`（內部值，1024 ms）換算的 sample 加 `2 × hop + 1`（[§4](reference/lifecycle.md#環形緩衝大小公式)）。
超過約 509 ms（8 kHz 約 1,018 ms）的延遲請在上游先粗對齊，實機仍應先量測總延遲。

**超出可靠上界（~509 ms）的真延遲不會「估不到」，而是可能在範圍內
高信心誤鎖**（confidence 只量 histogram 一致性、不量匹配品質）——機制
細節、blind-test 語料的實測延遲分佈（2021 全集 p50≈48 ms / p90≈225 ms /
max 923 ms、約 5% 超出 509 ms）與板端/離線的建議做法（系統延遲用
`fixed_delay_samples` 由系統層補償），見
`delay_estimator_design_zh_TW.md`。

`confidence` 量的是 histogram 一致性、不是匹配品質，所以「延遲狀態穩定」
不等於對齊正確；需要確認時消費端可自行判斷（例如比對
`aec_linear_is_cancelling()`）。細節見 `delay_estimator_design_zh_TW.md` §2。

延遲估計**不能**修正這些問題：mic/ref 取樣率不同、持續的 clock drift、
送錯的 reference 訊號。這些必須在上游解決。

離線處理且檔案已精準對齊時，可設
`cfg.delay_mode = AEC_DELAY_EXTERNAL_ALIGNED`
省掉首次擷取的擾動。線上產品請維持 `MATCHED`。

### 演算法延遲（實測）

以脈衝訊號量測輸出峰值位移的結果：

| 格點 | `residual = AEC_RESIDUAL_FULL`（預設） | `residual = AEC_RESIDUAL_OFF` |
|---|---:|---:|
| 8 kHz / 256 | 16.000 ms（1 hop） | 0 ms |
| 16 kHz / 256 | 8.000 ms（1 hop） | 0 ms |
| 16 kHz / 512 | 16.000 ms（1 hop） | 0 ms |
| 48 kHz / 1024 | 10.667 ms（1 hop） | 0 ms |

開啟殘留回音抑制時固定多一個 hop 的延遲；關閉後線性路徑沒有額外延遲。
`residual = FULL` 用 `aec_process()` 時，建立或 `aec_reset()` 之後的**第一個輸出 hop 全是 0**（OLA 尾巴是空的，輸入從第二個 hop 才出現；
16 kHz／256、0.3 振幅正弦輸入、無參考的實測：第一個 hop 的輸出 rms 為 0，第二個 hop 起接上輸入）。`aec_reset()` 之後重現與剛建立時完全相同的起頭。
引數被拒絕的呼叫（NULL、非有限的輸入樣本）回負值 `AecStatus`、完全不寫 `out`——呼叫端要自己決定那個 hop 播什麼（例如重送上一個 hop 或靜音）。
沒有任何 look-ahead。若外部要把 dry／linear 路徑與抑制後的輸出混合，必須先補償
這一個 hop（16 kHz／256 為 128 個樣本、8 ms），否則會聽成額外的回音。

---
