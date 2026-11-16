[C 使用手冊](../c_user_manual_zh_TW.md)

## 6. `AecConfig` 完整參考

`AecConfig` 只有 **12 個欄位**，都是產品會設的輸入。**永遠從 `aec_config_from_level()` 開始**，再覆寫少數必要欄位：

```c
AecConfig cfg;
aec_config_from_level(&cfg, AEC_LEVEL_BALANCED, 16000);   /* sample_rate、fft_size、level 與其餘欄位的預設 */
cfg.enable_cng = 0;                                       /* 只覆寫你真的要改的 */
```

`aec_config_from_level()` 先把整個結構歸零，再填 `sample_rate`、該取樣率的產品 `fft_size`、`level` 與其餘欄位的預設值；
列舉範圍外的 level 讀作 `AEC_LEVEL_BALANCED`。自己手動填一個未初始化的 `AecConfig` 幾乎一定會被驗證擋下，
而歸零的結構等於 `AEC_LEVEL_MILD`（level 0）。取樣率不是 8000／16000／48000 時（含 0 與負值），
`fft_size` 是 `0`，`aec_get_mem_size()` 回傳 0、`aec_init()` 回傳 NULL、`aec_create()` 回傳 NULL；函式本身不會溢位或中止。

**其餘一切都不是呼叫端的設定。** 濾波器長度（52 ms；48 kHz 為 64 ms）、步進、shadow filter、高通、飽和偵測、
各偵測器門檻、延遲估計的時間常數、抑制器的 floor 值等，全部由函式庫依 `(sample_rate, fft_size, n_mics, level)`
推導（庫內部的 `AecTuning`，不在公開標頭）；level 之外沒有第二個強度旋鈕，`aec_set_level()` 是唯一的執行期調整。
§6.5–§6.6c 保留這些值作為**唯讀參考**，說明 pool 大小與行為從哪裡來。以下所有預設值與範圍都直接取自程式碼。

| 欄位 | 型別 | 預設 | 允許範圍 | 小節 |
|---|---|---|---|---|
| `sample_rate` | `int` | 呼叫端指定 | 8000 / 16000 / 48000 | [§6.1](#61-訊號格點建立後不可更改) |
| `fft_size` | `int` | 該取樣率的產品格點 | 與 `sample_rate` 成對；`0` 被拒絕 | [§6.1](#61-訊號格點建立後不可更改) |
| `n_mics` | `int` | `1` | `1` 或 `AEC_ARRAY_MICS`（4） | [§6.8](#68-陣列實例欄位n_mics--4) |
| `level` | `AecLevel` | `AEC_LEVEL_BALANCED` | 0 – 3 | [§6.3](#63-抑制強度) |
| `residual` | `AecResidualMode` | `AEC_RESIDUAL_FULL` | `OFF`（0）… `FULL`（3） | [§6.2](#62-殘留階段與舒適噪音) |
| `delay_mode` | `AecDelayMode` | `AEC_DELAY_MATCHED` | `MATCHED` / `FIXED` / `EXTERNAL_ALIGNED` | [§6.4](#64-延遲估計) |
| `fixed_delay_samples` | `int` | `-1` | `-1`，或 `FIXED` 時 0 – `120 × sample_rate` | [§6.4](#64-延遲估計) |
| `delay_num_filters` | `int` | `5` | 1 – 5（只在 `MATCHED`） | [§6.4](#64-延遲估計) |
| `enable_cng` | `int` | `1` | 0 / 1 | [§6.2](#62-殘留階段與舒適噪音) |
| `capture_proxy_channel` | `int` | `0` | 0 – 3（單聲道必須是 0） | [§6.8](#68-陣列實例欄位n_mics--4) |
| `delay_backward_quarantine_enabled` | `int` | `0` | 0 / 1（單聲道必須是 0） | [§6.8](#68-陣列實例欄位n_mics--4) |
| `delay_backward_quarantine_s` | `float` | `1.0` | 0 – 3600 | [§6.8](#68-陣列實例欄位n_mics--4) |

### 6.1 訊號格點（建立後不可更改）

| 欄位 | 型別 | 預設 | 允許範圍 | 說明／何時調整 |
|---|---|---|---|---|
| `sample_rate` | `int` | 呼叫端指定 | 8000 / 16000 / 48000 | 必填。與音訊來源一致。 |
| `fft_size` | `int` | 8k→256、16k→256、48k→1024（由 `aec_config_from_level()` 填入，＝`aec_default_fft_size()`） | 須與 `sample_rate` 成對（見 [§1](../c_user_manual_zh_TW.md#1-這個函式庫是什麼不是什麼)）；**`0`（自動）會被拒絕** | 只有 16 kHz 有第二選擇（512）。改成 512 會讓 hop 變 16 ms、延遲加倍，但每秒呼叫次數減半。核心不會替你猜格點——所有入口都走同一支 `aec_resolve_signal_grid()`。 |

濾波器長度不是欄位：由取樣率決定，`sr × 52/1000`（48 kHz 為 `sr × 64/1000`）→ 8k：416、16k：832、48k：3072 sample；
分割數是 `ceil(濾波器長度 ÷ hop)`。不支援的取樣率派生值為 `0`，與 `fft_size` 一樣在建立時被拒絕。

### 6.2 殘留階段與舒適噪音

| 欄位 | 預設 | 說明／何時調整 |
|---|---:|---|
| `residual` | `AEC_RESIDUAL_FULL` | `AecResidualMode`：後級（殘留回音抑制）做到哪裡，允許範圍 `AEC_RESIDUAL_OFF`（0）…`AEC_RESIDUAL_FULL`（3），範圍外會被驗證拒絕。`FULL`＝完整抑制（`aec_process()` 套用增益、加舒適噪音並合成）；`OFF`＝只有線性 filter，輸出是線性 filter 的輸出（`AecLinearView.linear_hop`）、延遲降為 0，但回音殘留明顯變多，診斷時可用（見 [§9](integration.md#9-常見問題排除)）。`STATE`、`ESTIMATE` 給自己套增益的整合者，見 [§8](integration.md#8-進階整合-seam)。不影響**單聲道**實例的記憶池大小；`n_mics = 4` 的陣列實例則隨它改變，且不接受 `OFF`（[§6.8](#68-陣列實例欄位n_mics--4)）。 |
| `enable_cng` | `1` | 舒適噪音，0 或 1。若下游還有另一套噪音處理／CNG，**請關掉這個**，兩層疊加會讓噪音底變得不自然。 |

內部的模組開關（mic 路徑高通 80 Hz、削波偵測與參考訊號軟限幅、shadow filter）一律開啟，不可設定。
shadow filter 佔 27,888 B（16 kHz/256，見 [§4](lifecycle.md#4-生命週期) 記憶池表），用於偵測回音路徑變化。

### 6.3 抑制強度

`AecConfig.level`（`AecLevel`，0 – 3）是**唯一**的抑制強度設定，只收數字：0＝`AEC_LEVEL_MILD`、1＝`AEC_LEVEL_BALANCED`（預設）、
2＝`AEC_LEVEL_AGGRESSIVE`、3＝`AEC_LEVEL_NO_FLOOR`；範圍外的值被驗證拒絕。選擇方式與取捨見 [§7](levels.md#7-level-選擇)。
函式庫由 level 查表帶出三個內部值（`aec_level_floors()` 可讀）：

| level | far-active 最低增益 | double-talk 最低增益 | split floor |
|---|---:|---:|---|
| `MILD` | -20 dB | -20 dB | 開 |
| `BALANCED` | -28 dB | -20 dB | 開 |
| `AGGRESSIVE` | -38 dB | -30 dB | 開 |
| `NO_FLOOR` | （不用） | （不用） | 關 |

`aec_set_level()` 在執行期改變它，並把新 level 記為實例的 level。double-talk 專用下限是否生效、近端保護的時間常數等都是內部常數。

### 6.4 延遲估計

| 欄位 | 型別 | 預設 | 允許範圍 | 說明／何時調整 |
|---|---|---:|---|---|
| `delay_mode` | `AecDelayMode` | `AEC_DELAY_MATCHED` | `MATCHED` / `FIXED` / `EXTERNAL_ALIGNED` | **對齊方式的唯一真實來源**（見 [§3](../c_user_manual_zh_TW.md#延遲契約) 延遲契約的三態表）。init 時決定、instance 存活期間不可變（各 mode 的 pool 佈局不同，要改必須 destroy + 重新 init）。非法組合一律拒絕。 |
| `fixed_delay_samples` | `int` | `-1` | `-1`，或 `FIXED` 時 0 – `120 × sample_rate` | **只在 `FIXED` 有意義**：bring-up 量測到的系統延遲，單位是 native-rate sample。其他 mode 必須維持 `-1`（帶值會被拒絕，不會被忽略）。它同時**決定參考環形緩衝的大小**：`FIXED` 只配置 `fixed_delay_samples + hop_size` 個 sample（緊上界，見 [§4](lifecycle.md#4-生命週期) 記憶池表的公式）。 |
| `delay_num_filters` | `int` | `5` | 1 – 5（**只在 `MATCHED`**；其他 mode 必須維持 5） | Matched-filter bank 大小（決定記憶體與可搜尋的延遲範圍）。可靠搜尋上限隨之縮小：n=1→125ms／2→221ms／3→317ms／4→413ms／5→509ms；每少一組省 ~4.2 MMAC/s 的搜尋算力（每 hop 全速分析時），**並省 11,264 B 記憶池（單通道 MATCHED，48 kHz 為 23,552 B；陣列的共用 estimator 為 5,120 B）**（bank、render ring、兩個 histogram 都依 n 從記憶池切出，見 [§4](lifecycle.md#4-生命週期) 記憶池表）。縮小的正確場景是**上游已先提供 coarse-compensated far**（HAL/系統層自行移除 bulk delay 後才餵進來）、matched filter 只需追殘差的 `MATCHED` 部署——注意這與 `FIXED` 無關：`FIXED`（`fixed_delay_samples`）根本不建 matched filter，`EXTERNAL_ALIGNED` 連 estimator/ring 都沒有。bench/dataset 一律維持 5（見 `delay_estimator_design_zh_TW.md` §5）。`FIXED` / `EXTERNAL_ALIGNED` 根本沒有 matched filter，改這個值會被拒絕（不是忽略）；`0` 在任何 mode 都是錯誤，不是「關閉延遲估計」的入口。 |

估計器的時間常數是內部常數：延遲估計值需穩定 0.3 s 才視為鎖定，之後 matched filter 預設每個 hop 都分析（不做 duty-cycle 節流）；
`MATCHED` 的參考環形緩衝取 estimator 可回報的最大延遲加 `hop_size`（見 [§4](lifecycle.md#4-生命週期) 環形緩衝大小公式；陣列實例的共用環同式），
它只決定記憶體，**不是**可擷取的延遲上限（上限由 `delay_num_filters` 決定，環永遠放得下 estimator 報得出的延遲）。

### 6.5 以 10 ms 為單位撰寫的時間常數（內部，唯讀參考）⚠️

函式庫內有六個以「N × 10 ms」撰寫的時間常數（呼叫端不能設定），建立實例時依目前格點換算成 hop 數：

> **數值是「N × 10 ms 的實際時長」，不是直接的 hop 計數。**

例如在 16 kHz/256（hop = 8 ms）下 `warmup_frames = 100` 在實例內部變成 125 hop，也就是 1000 ms；在 48 kHz 下變成 94 hop，同樣約 1000 ms。
**換算後的時長在所有格點上一致**，這正是這個設計的目的。換算寫進實例自己的 tuning 副本（`Aec::tun`），不改動傳入的 `AecConfig`。

| 內部常數 | 預設 | 換算後時長 | 作用 |
|---|---:|---:|---|
| `warmup_frames` | `100` | ≈ 1000 ms | 啟動期長度：這段期間 filter 用較積極的方式收斂。 |
| `epc_hangover` | `20` | ≈ 200 ms | 偵測到回音路徑改變後的加速恢復期長度。 |
| `ne_recent_hold` | `150` | ≈ 1500 ms | 「近端最近說過話」旗標的保持時間，餵給 double-talk 保護。 |
| `filter_misadjustment_stable_frames` | `30` | ≈ 300 ms | 判定 filter 已穩定所需的持續時間。 |
| `filter_misadjustment_hangover_frames` | `100` | ≈ 1000 ms | 上述判定的保持時間。 |
| `shadow_err_alpha` | `0.80` | （平滑係數，非時長） | 輔助 filter 誤差的平滑係數，同樣依格點重新換算（16 kHz/256 下實際為 0.8365）。 |

`ne_recent_sustain`（`3`）**不會**被換算，固定維持字面值 `3`，不要跟上表混淆——但原因不是它是真正的次數計數：它的計數器每個 hop 都會遞增或
歸零、沒有跳過條件，所以「連續 3 次」就是連續 3 個 hop 的時長，會隨格點變動（16 kHz/
256 預設格點約 24 ms，16 kHz/512 約 48 ms，48 kHz/1024 約 32 ms）。固定字面值 3 是
因為三個格點目前各自的行為都已個別驗證過；換算成單一時長會讓 16 kHz/512 格點變成
只需 2 個 hop，量測後會在該格點引入新的回音誤觸發。

### 6.6 其餘內部調校值（唯讀參考）

預設值是整體調過的，呼叫端無法更動（測試經 `aec_testing_*` 鉤子覆寫，見 `test/aec_test_tuning.h`）。

| 內部值 | 預設 | 允許範圍（驗證） | 說明 |
|---|---:|---|---|
| `mu` | `0.3` | 0 – 10 | 主 filter 適應步進。 |
| `delta` | `1e-8` | 0 – 1 | 適應運算的除零保護下限。 |
| `highpass_cutoff_hz` | `80.0` | 高通開啟時須 > 0 且 **< 0.45 × sample_rate** | mic 路徑高通截止頻率。 |
| `saturation_threshold` | `0.95` | 0 – 10 | 判定削波的振幅門檻（相對滿刻度 1.0）。 |
| `shadow_mu_min` / `shadow_mu_nlms` | `0.5` / `0.5` | 0 – 10 | 輔助 filter 的最小／標稱步進。 |
| `epc_total_rise` / `epc_delta_threshold` | `1.5` / `0.3` | 0 – 1000 | 判定回音路徑改變的能量上升倍率與變化量門檻。 |
| `ne_recent_threshold` | `0.3` | 0 – 1000 | 判定近端有聲音的能量門檻。 |
| `filter_misadjustment_scale_min` / `_max` | `0.5` / `2.0` | 0 – 1000 | 回音估計修正倍率下限／上限。 |
| `dt_aware_recovery_soft`、`dt_aware_res_floor_enabled`、`delay_acquire_protect_converged`、`delay_acquire_warm_transfer`（均 `1`）、`delay_acquire_inst_erle_db`（`4.0` dB） | | 0 / 1；-100 – 100 | 近端最近有聲音時改用非破壞性的重新對齊；double-talk 專用下限；首次延遲擷取時保護已收斂的 filter 並把學到的回音模型平移過去（需 inst-ERLE 峰值超過門檻）。 |

### 6.6b 逐階段耗時（`aec_get_last_timing`）

```c
typedef struct AecStageTiming {                 /* aec.h；各欄單位微秒 */
    uint32_t delay_us, frontend_us, linear_us, res_us, steering_us;
} AecStageTiming;
void aec_get_last_timing(const Aec* a, AecStageTiming* out);
```

最近一個 hop 內五個最大階段的 wall-clock 成本（微秒，預設用
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
  `aec_process_linear_shared_far_from()` 借用），FFT 也算在這裡。
- `steering_us`：主濾波器與 post 區塊之間的全部工作——stationarity refresh、
  `e2_coarse`/ERL publish 與 coarse rescue、double-talk analyzer、EPV、
  `shadow_rise`、misadjustment estimator。它們都不直接動樣本，只決定下一個
  hop 怎麼適應，但確實是每個 hop 的真實成本。
- `res_us`：AEC3 post/RES 區塊。只要 `AecConfig.residual` 不是 `AEC_RESIDUAL_OFF`
  就會跑，所以拆開進入點（`aec_process_residual()`）的呼叫端也會看到真實數字。

五者**刻意不等於**整個呼叫：餘額是 `res_us` 收尾之後才跑的 per-sample power
EMA 與收斂偵測。要呈現完整分解的呼叫端得自己把餘額算出來。

每次 `aec_process*()` 在引數檢查之後清零，所以一個 hop 只會回報它真的跑到的階段；被引數檢查拒絕（NULL、已銷毀 handle、非有限樣本）的呼叫不動記錄。
解析度是微秒，低於 1 µs 的階段會讀到 0。

#### 預設開啟

量測在預設建置就有（`aec.h` 的 `AEC_STAGE_TIMING` 預設 1；Makefile 的 `PROFILE` 預設 1，
並把 `-DAEC_STAGE_TIMING=$(PROFILE)` 放進 `CFLAGS`）。每 hop 每實例七次 clock 讀取：
桌面主機上一次讀取約 19 ns，所以單聲道實例每 hop 約 0.13 µs（16 kHz／256 與 48 kHz／1024
的 hop 成本約 15–45 µs，差異在量測雜訊內）；四麥克風陣列約三十次讀取。輸出逐位元不變。

⚠ **要關掉就在建置函式庫時關**，五個欄位會每個 hop 都讀 0，archive 也不再引用 `clock_gettime`：

```bash
make PROFILE=0              # 等同在 CFLAGS 加 -DAEC_STAGE_TIMING=0
```

不要用 `EXTRA_CFLAGS=-DAEC_STAGE_TIMING=0` 關閉：Makefile 已經帶了
`-DAEC_STAGE_TIMING=$(PROFILE)`，兩個值不同會是重複定義。那個 0 就是呼叫端分辨
「沒量」與「量了但很快」的唯一線索；消費端的 `#if AEC_STAGE_TIMING` 要用與函式庫
相同的值（`aec.h` 預設 1，與預設建置一致）。

`PROFILE` 折進 config hash（`CFG_SIG`），所以 `PROFILE=0` 與預設建置各有自己的
`bin/<backend>-<hash>/`，不會互相交出過期的 object；`EXTRA_CFLAGS` 同樣折進去。

#### 目標平台沒有 `CLOCK_MONOTONIC` 時

`clock_gettime(CLOCK_MONOTONIC, ...)` 是 POSIX 而非 C99，精簡 libc 的目標平台
不一定有，預設建置會連結失敗。這種平台要嘛用 `make PROFILE=0` 關掉量測，要嘛換掉時鐘：

```bash
make EXTRA_CFLAGS='-DAEC_NOW_US=board_timer_us -include my_timer.h'
```

`AEC_NOW_US` 要是**純識別字**（本 repo 的 FP policy 不允許 `EXTRA_CFLAGS` 出現
括號），指向一個不收參數、回傳 `uint32_t` 微秒的函式，宣告由整合者自備。指定
之後預設實作既不編譯也不連結，`<time.h>` 也不會被 include。

替代時鐘必須**單調**：會倒退的 stamp 會讓無號減法產生接近 32-bit 全距的荒謬
數值，而不是一個小的錯誤值。指向一個永遠回傳同一個值的函式是合法用法——每個
階段都讀 0，只剩對一個沒人回讀的結構做幾次純量寫入。

包裝這個函式庫的上層元件若有自己的計時覆寫點，刻意不共用：一個元件的
時鐘屬於該元件自己的建置契約。

### 6.6c Live observer（實驗性，內部）

**預設關閉，關閉時輸出逐位元不變。未進任何出貨 level，沒有公開開關**（只有測試鉤子與 `bench_rtf --live-observer` 的 `AEC_TESTING` 建置能開）。

機制：第三個獨立 PBFDAF 實例（NLMS，跟 `shadow_filter` 同一個 struct），讀
遠端 ring 比主濾波器的延遲補償讀取早 24 ms。每 hop 把
自己的誤差能量 EMA 跟主濾波器（R² 實際採用的那條選線，不是恆取 main）的
誤差能量 EMA 比較，只有在觀察器沒有比主濾波器差（corroboration）時才信任
其證據；證據夾在主濾波器殘留回音功率以內，經 `max(r2, evidence)` 注入
residual-echo R²（只到 1000 Hz 的低頻段），從不碰輸出、
delay 或任一濾波器的視窗／taps。內部值：`live_observer_shift_ms`＝24.0、`live_observer_max_hz`＝1000.0、
`live_observer_mu`＝0.3（範圍 (0, 2]）、`live_observer_corrob_ms`＝100.0、`live_observer_corrob_margin_db`＝0.0（須有限）。

記憶體增量（`aec_testing_get_mem_size()` 實測，delay_num_filters=5，balanced
level）：8 kHz/256 +29,472 B、16 kHz/256 +38,768 B、16 kHz/512
+58,656 B、48 kHz/1024 +141,648 B。算量增量用 `bench_rtf --live-observer`
在目標格點量測（`make bench AEC_TESTING=1` 建置）；固定開銷來自觀察器每 hop 3 個與分接數
無關的 FFT。

### 6.7 靜態記憶體拆解（`aec_get_mem_breakdown`）

```c
typedef struct AecMemBreakdown {
    size_t total_bytes;       /* == aec_get_mem_size(cfg)                 */
    size_t estimator_bytes;   /* delay_aec_get_mem_size(); 0 非 MATCHED  */
    size_t ring_bytes;        /* 對齊環形緩衝；0 為 EXTERNAL_ALIGNED     */
    size_t lane_bytes;        /* 陣列：四條 lane 池合計；單聲道為 0      */
    size_t fusion_bytes;      /* 陣列：控制區塊、融合緩衝、共用 FFT 控制代碼與增益階段；單聲道為 0 */
} AecMemBreakdown;

/* AEC_OK（0）並填入 *out；cfg 為 NULL 或無效則 AEC_ERR_ARG 並把 *out 清零 */
int aec_get_mem_breakdown(const AecConfig* cfg, AecMemBreakdown* out);
```

陣列實例（`n_mics = 4`）的 `lane_bytes`／`fusion_bytes` 才非 0，`estimator_bytes`／`ring_bytes` 是共用估計器與共用對齊環的；
`estimator_bytes`、`ring_bytes`、`lane_bytes`、`fusion_bytes` 四者相加**不等於** `total_bytes`：差的是 `Aec` 標頭
（`Aec` 標頭，對齊 16 B），它不在任何一個欄位之內（實測數字見 [§4](lifecycle.md#陣列實例的記憶池)）。

把 [§4](lifecycle.md#4-生命週期) 記憶池大小表格背後的算法變成一個可查詢的 API：`total_bytes` 就是
`aec_get_mem_size(cfg)` 本身（同一次呼叫算出來，不是另一套算式），
`estimator_bytes`／`ring_bytes` 是這個總量裡兩個有名字的子集，用的是
`aec_get_mem_size()` 內部本來就在用的同一批 helper（`delay_aec_get_mem_size()`、
`aec_derive_dims()` 的 ring 大小 out-param）——查詢這個函式不可能得到與
`aec_carve()` 實際切出來的 pool 佈局不同的答案。失敗條件與
`aec_get_mem_size()` 完全一致：`cfg` 為 `NULL` 或未通過 `aec_resolve()`
時回傳 `AEC_ERR_ARG`（並把 `*out` 清零）。`example/aec_wav.c` 的 `--print-mem-size`
（[附錄 A](wav_cli.md#附錄-aaec_wav-命令列工具)）就是這個 API 的一個瘦封裝。

### 6.8 陣列實例欄位（`n_mics = 4`）

| 欄位 | 型別 | 預設 | 允許範圍 | 說明 |
|---|---|---:|---|---|
| `n_mics` | `int` | `1` | `1` 或 `AEC_ARRAY_MICS`（4） | `4` 建出四麥克風陣列實例：四條 lane、共用 delay 估計器與對齊後 far、融合緩衝都在同一塊記憶池，一次 `aec_init()`（或 `aec_create()`）。呼叫序列見 [§8.8](integration.md#88-麥克風陣列實例n_mics--4)。建立時，不可變。 |
| `capture_proxy_channel` | `int` | `0` | `0` – 3（單聲道必須是 0） | 共用 delay 估計器的 capture 輸入取哪一路麥克風；`delay_backward_quarantine_*` 也以這一路 lane 的消除狀態為準。 |
| `delay_backward_quarantine_enabled` | `int` | `0` | 0 / 1（單聲道必須是 0） | 只在 `MATCHED` 有意義。比目前 delay 更早的新估計，在 capture-proxy lane 仍在消除期間先保留 `delay_backward_quarantine_s` 秒再採用；消除失效就立刻採用。前向移動與首次取得不受影響。 |
| `delay_backward_quarantine_s` | `float` | `1.0` | 0 – 3600 | 上一項的保留時間（秒），建立時換算成 hop 數（下限 1）。 |

以 `make AEC_NO_ARRAY=1` 建置的單聲道函式庫沒有陣列實例：`n_mics` 不是 1 時 `aec_get_mem_size()` 回 0、`aec_init()` 回 NULL、`aec_create()` 回 NULL，`aec_get_array()` 與 `aec_array_*()` 診斷函式不存在（呼叫即連結錯誤）。flag 折進 config hash，與一般建置各有自己的 `bin/<backend>-<hash>/`；實測 `libaec.a` 的 `__TEXT` 減少約 21 kB（SIMD=1，160,122 → 139,234 B；SIMD=0 為 140,678 → 122,914 B）。

陣列實例對既有欄位的規則：

- `residual` 必須是 `STATE`、`ESTIMATE` 或 `FULL`（`OFF` 被拒絕），並改變池大小；
- 格點只接受 16 kHz／256、16 kHz／512、48 kHz／1024；
- `delay_mode`、`delay_num_filters`、`fixed_delay_samples` 描述**共用**的對齊，每條 lane 一律以 `EXTERNAL_ALIGNED` 吃對齊後的 far；
  共用環形緩衝大小與單聲道同式，取 estimator 可回報的最大延遲加 `hop_size`（見 [§6.4](#64-延遲估計)），duty-cycle 保持時間固定 0.3 s；
- 其餘內部調校值（濾波器長度、步進、shadow filter、偵測器門檻…）與 level 的內部值原樣交給每條 lane；level 只作用在融合後的一次增益階段，lane 自己的 level 沒有效果；
- `aec_set_level()` 重新設定融合的增益階段（`FULL`）並記錄新 level；`STATE`／`ESTIMATE` 沒有增益階段，呼叫回 0 但只更新記錄的 level。

---
