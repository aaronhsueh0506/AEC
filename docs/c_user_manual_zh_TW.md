# AEC C 使用手冊（繁體中文）

> **版本：4.0.0rc1**（對應 `python/aec.py` 的 `__version__`，本手冊唯一的版本標示）
>
> 本手冊是給**函式庫使用者**看的整合說明：什麼時候呼叫哪個函式、參數怎麼設。
> 演算法內部設計請看 [`aec_methods.md`](aec_methods.md)。
>
> HTML 版（含可編譯範例與時序圖）：[呼叫使用](html/aec_usage.html)、
> [流程時序圖](html/aec_sequence.html)、[外部參數與公開函式](html/aec_params.html)、
> [內部參數](html/aec_params_internal.html)。

- [AEC C 使用手冊（繁體中文）](#aec-c-使用手冊繁體中文) <!-- split-toc -->
  - [你需要 include 的 header](#你需要-include-的-header) <!-- split-toc -->
  - [1. 這個函式庫是什麼／不是什麼](#1-這個函式庫是什麼不是什麼) <!-- split-toc -->
    - [支援的取樣率與訊號格點](#支援的取樣率與訊號格點) <!-- split-toc -->
    - [硬性限制](#硬性限制) <!-- split-toc -->
  - [2. Quick start](#2-quick-start) <!-- split-toc -->
    - [建置](#建置) <!-- split-toc -->
    - [連結](#連結) <!-- split-toc -->
    - [最小可用範例](#最小可用範例) <!-- split-toc -->
  - [3. 資料契約](#3-資料契約) <!-- split-toc -->
    - [Sample 格式與尺度](#sample-格式與尺度) <!-- split-toc -->
    - [Frame 與 hop](#frame-與-hop) <!-- split-toc -->
    - [延遲契約](#延遲契約) <!-- split-toc -->
    - [演算法延遲（實測）](#演算法延遲實測) <!-- split-toc -->
  - [4. 生命週期](c_user_manual_zh_TW_01_lifecycle.md#4-生命週期) <!-- split-toc -->
    - [A. Heap 路徑（`aec_create`）](c_user_manual_zh_TW_01_lifecycle.md#a-heap-路徑aec_create) <!-- split-toc -->
    - [B. 呼叫端自備記憶池（`aec_get_mem_size` + `aec_init`）](c_user_manual_zh_TW_01_lifecycle.md#b-呼叫端自備記憶池aec_get_mem_size--aec_init) <!-- split-toc -->
    - [記憶池大小（實測，balanced preset，其餘為預設值）](c_user_manual_zh_TW_01_lifecycle.md#記憶池大小實測balanced-preset其餘為預設值) <!-- split-toc -->
    - [`aec_reset`](c_user_manual_zh_TW_01_lifecycle.md#aec_reset) <!-- split-toc -->
    - [什麼不能改](c_user_manual_zh_TW_01_lifecycle.md#什麼不能改) <!-- split-toc -->
  - [5. 錯誤語意表](c_user_manual_zh_TW_01_lifecycle.md#5-錯誤語意表) <!-- split-toc -->
  - [6. `AecConfig` 完整參考](c_user_manual_zh_TW_02_config.md#6-aecconfig-完整參考) <!-- split-toc -->
    - [6.1 訊號格點（建立後不可更改）](c_user_manual_zh_TW_02_config.md#61-訊號格點建立後不可更改) <!-- split-toc -->
    - [6.2 模組開關（全部只接受 0 或 1）](c_user_manual_zh_TW_02_config.md#62-模組開關全部只接受-0-或-1) <!-- split-toc -->
    - [6.3 抑制強度](c_user_manual_zh_TW_02_config.md#63-抑制強度) <!-- split-toc -->
    - [6.4 延遲估計](c_user_manual_zh_TW_02_config.md#64-延遲估計) <!-- split-toc -->
    - [6.5 以 10 ms 為單位撰寫的時間欄位 ⚠️](c_user_manual_zh_TW_02_config.md#65-以-10-ms-為單位撰寫的時間欄位-) <!-- split-toc -->
    - [6.6 其餘可設定欄位](c_user_manual_zh_TW_02_config.md#66-其餘可設定欄位) <!-- split-toc -->
    - [6.6b 逐階段耗時（`aec_get_last_timing`）](c_user_manual_zh_TW_02_config.md#66b-逐階段耗時aec_get_last_timing) <!-- split-toc -->
    - [6.6c Live observer（實驗性）](c_user_manual_zh_TW_02_config.md#66c-live-observer實驗性) <!-- split-toc -->
    - [6.7 靜態記憶體拆解（`aec_get_mem_breakdown`）](c_user_manual_zh_TW_02_config.md#67-靜態記憶體拆解aec_get_mem_breakdown) <!-- split-toc -->
  - [7. Preset 選擇](c_user_manual_zh_TW_03_preset.md#7-preset-選擇) <!-- split-toc -->
    - [7.1 執行期切換 preset（`aec_set_preset`）](c_user_manual_zh_TW_03_preset.md#71-執行期切換-presetaec_set_preset) <!-- split-toc -->
  - [8. 進階整合 seam](c_user_manual_zh_TW_04_seam_troubleshooting.md#8-進階整合-seam) <!-- split-toc -->
    - [8.1 Context-only 處理路徑](c_user_manual_zh_TW_04_seam_troubleshooting.md#81-context-only-處理路徑) <!-- split-toc -->
    - [8.2 多實例共用遠端 FFT](c_user_manual_zh_TW_04_seam_troubleshooting.md#82-多實例共用遠端-fft) <!-- split-toc -->
    - [8.3 `AecResContext` 欄位](c_user_manual_zh_TW_04_seam_troubleshooting.md#83-aecrescontext-欄位) <!-- split-toc -->
    - [8.4 Duty-cycle engagement census](c_user_manual_zh_TW_04_seam_troubleshooting.md#84-duty-cycle-engagement-census) <!-- split-toc -->
    - [8.5 獨立的 RES 級（`aec_res.h`）](c_user_manual_zh_TW_04_seam_troubleshooting.md#85-獨立的-res-級aec_resh) <!-- split-toc -->
    - [8.6 線性對齊 context（`aec_get_linear_context`）](c_user_manual_zh_TW_04_seam_troubleshooting.md#86-線性對齊-contextaec_get_linear_context) <!-- split-toc -->
    - [8.7 單聲道整鏈：AEC → NR → RES 增益融合 → 合成](c_user_manual_zh_TW_04_seam_troubleshooting.md#87-單聲道整鏈aec--nr--res-增益融合--合成) <!-- split-toc -->
  - [9. 常見問題排除](c_user_manual_zh_TW_04_seam_troubleshooting.md#9-常見問題排除) <!-- split-toc -->
    - [9.1 回音殘留太多](c_user_manual_zh_TW_04_seam_troubleshooting.md#91-回音殘留太多) <!-- split-toc -->
    - [9.2 Double-talk 時近端被壓掉](c_user_manual_zh_TW_04_seam_troubleshooting.md#92-double-talk-時近端被壓掉) <!-- split-toc -->
    - [9.3 啟動時收斂太慢／前一秒回音明顯](c_user_manual_zh_TW_04_seam_troubleshooting.md#93-啟動時收斂太慢前一秒回音明顯) <!-- split-toc -->
    - [9.4 裝置移動時回音突然變大](c_user_manual_zh_TW_04_seam_troubleshooting.md#94-裝置移動時回音突然變大) <!-- split-toc -->
    - [9.5 其他](c_user_manual_zh_TW_04_seam_troubleshooting.md#95-其他) <!-- split-toc -->
  - [附錄 A：`aec_wav` 命令列工具](c_user_manual_zh_TW_05_appendix.md#附錄-aaec_wav-命令列工具) <!-- split-toc -->
    - [`--print-mem-size` 輸出](c_user_manual_zh_TW_05_appendix.md#--print-mem-size-輸出) <!-- split-toc -->
    - [Duty census 診斷輸出](c_user_manual_zh_TW_05_appendix.md#duty-census-診斷輸出) <!-- split-toc -->
  - [回報問題](c_user_manual_zh_TW_05_appendix.md#回報問題) <!-- split-toc -->
## 你需要 include 的 header

```c
#include "aec.h"        /* c_impl/include/aec.h — 標準整合只需要這一個 */
```

只要自己用 `aec_process()` 取得時域輸出，就只需要 `aec.h`。另有一個 header 屬於
**對外公開、範圍較窄的介面**，在自己組合頻域後處理鏈時需要（單聲道整鏈；以及直接驅動多通道核心、
在 beamformer 之後自備 `AecRes` 的呼叫端——Audio_ALG 的 4ch 單元不含 RES，由應用在單元之後自己建立，見 [§8.5](c_user_manual_zh_TW_04_seam_troubleshooting.md#85-獨立的-res-級aec_resh)）：

| Header | 何時需要 |
|---|---|
| `aec_res.h` | `aec_cng_fill()` 舒適噪音填補（單聲道、多通道兩種串接都用）；以及對 beamformer 之後的融合頻譜算 RES 增益的 `AecRes`（[§8.5](c_user_manual_zh_TW_04_seam_troubleshooting.md#85-獨立的-res-級aec_resh)、[§8.6](c_user_manual_zh_TW_04_seam_troubleshooting.md#86-線性對齊-contextaec_get_linear_context)） |

`suppression_gain.h` 與 `aec_balanced_config.h` 是內部 header。Audio_ALG 的範例與多通道單元
只 include `aec.h` 與 `aec_res.h`，不直接 include 這兩個。

`c_impl/include/` 底下其餘 27 個 header（`aec_post.h`、`pbfdkf.h`、`delay_aec.h`、`suppression_gain.h` 等）
全部是**內部實作**用的，會隨版本改變，請勿直接 include，也不要依賴其中任何型別或函式。
注意 `aec.h` 目前會把 `struct Aec` 完整攤開，因此它們仍必須留在 include 路徑上
才能編譯——「留在路徑上」不等於「可以使用」。

`aec.h` 本身會 include 共用 DSP 函式庫（`audiolib`）的 `fft_wrapper.h` 與 `hpf.h`，
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

> 8 kHz 是 **legacy 格點**：本函式庫與 Audio_ALG 的 mono 範例支援它
> （既有 8 kHz 測試路徑仍在用），但它不是產品正式格點，Audio_ALG 的
> 4 聲道線性 AEC + BF 單元也不支援（該 API 只保證 16k/256、16k/512、48k/1024
> 三個產品格點）。`AecSignalGrid.is_legacy` 會為它、且只為它設為 1。

### 硬性限制

| 限制 | 說明 |
|---|---|
| **只支援單聲道** | 一個實例 = 1 mic + 1 ref。沒有 mic array、沒有 stereo reference。多路訊號請各自建立獨立實例。 |
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

上限來自 matched filter 的搜尋長度（`delay_aec.h` 的
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

`confidence` 量的是 histogram 一致性、不是匹配品質，所以「延遲狀態穩定」
不等於對齊正確；需要確認時消費端可自行判斷（例如比對
`aec_linear_is_cancelling()`）。細節見 `delay_estimator_design_zh_TW.md` §2。

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
沒有任何 look-ahead。若外部要把 dry／linear 路徑與抑制後的輸出混合，必須先補償
這一個 hop（16 kHz／256 為 128 個樣本、8 ms），否則會聽成額外的回音。

---

