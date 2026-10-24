> [索引](c_user_manual_zh_TW.md) · 上一部分：[c_user_manual_zh_TW_01_lifecycle.md](c_user_manual_zh_TW_01_lifecycle.md) · 下一部分：[c_user_manual_zh_TW_03_preset.md](c_user_manual_zh_TW_03_preset.md) <!-- split-nav -->
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
改 [§7](c_user_manual_zh_TW_03_preset.md#7-preset-選擇) 表列的三個 floor 欄位（`min_gain_floor_far_active_db`、`min_gain_floor_dt_db`、`min_gain_split_floor_enabled`）。以下所有預設值與範圍都直接取自程式碼。

**參數分級**：建議對外開放的只有 `sample_rate`、`fft_size`、`filter_length`、`delay_mode`、
`fixed_delay_samples`、`delay_num_filters`、`enable_res`、`enable_cng` 與 preset（經由
`min_gain_floor_far_active_db`／`min_gain_floor_dt_db`／`min_gain_split_floor_enabled` 三個欄位）；
進階外部為 `enable_highpass`、`highpass_cutoff_hz`、`enable_shadow`、`delay_est_period_s`；
`return_res_context`、`spatial_linear_context` 是串鏈整合者專用；`enable_delay_est` 已 deprecated；
其餘 38 個欄位（§6.5 與 §6.6 的欄位、live observer 等）屬內部調校參數，不建議開放。
逐欄分級、出處行與理由見 [外部參數](html/aec_params.html) 與 [內部參數](html/aec_params_internal.html)。

### 6.1 訊號格點（建立後不可更改）

| 欄位 | 型別 | 預設 | 允許範圍 | 說明／何時調整 |
|---|---|---|---|---|
| `sample_rate` | `int` | 呼叫端指定 | 8000 / 16000 / 48000 | 必填。與音訊來源一致。 |
| `fft_size` | `int` | 8k→256、16k→256、48k→1024（由 `aec_config_defaults()` 填入，＝`aec_default_fft_size()`） | 須與 `sample_rate` 成對（見 [§1](c_user_manual_zh_TW.md#1-這個函式庫是什麼不是什麼)）；**`0`（自動）會被拒絕** | 只有 16 kHz 有第二選擇（512）。改成 512 會讓 hop 變 16 ms、延遲加倍，但每秒呼叫次數減半。核心不會替你猜格點——所有入口都走同一支 `aec_resolve_signal_grid()`。 |
| `filter_length` | `int` | `sr × 52/1000`（48 kHz 為 `sr × 64/1000`）→ 8k:416、16k:832、48k:3072 | 1 – 4096 | **單位是 sample，不是 ms。** 涵蓋的回音路徑長度。大房間／長殘響時調高，代價是記憶體與運算量上升。 |
| `n_partitions` | `int` | `0`（自動） | `0` 或 1 – 256 | 留 `0` 即可，由 `filter_length` 與 hop 自動推導。 |

### 6.2 模組開關（全部只接受 0 或 1）

| 欄位 | 預設 | 說明／何時調整 |
|---|---:|---|
| `enable_res` | `1` | 殘留回音抑制。關閉後輸出是純線性 filter 殘差、延遲降為 0，但回音殘留明顯變多。診斷時可關（見 [§9](c_user_manual_zh_TW_04_seam_troubleshooting.md#9-常見問題排除)）。 |
| `enable_cng` | `1` | 舒適噪音。若下游還有另一套噪音處理／CNG，**請關掉這個**，兩層疊加會讓噪音底變得不自然。 |
| `enable_delay_est` | `1` | **DEPRECATED**——`delay_mode` 的相容轉譯層（見 [§3](c_user_manual_zh_TW.md#延遲契約) 延遲契約與 §6.4）。設 0 等同 `EXTERNAL_ALIGNED`（或帶 `fixed_delay_samples` 時等同 `FIXED`）。`EXTERNAL_ALIGNED` 會少掉 165,008 B 記憶體（16 kHz/256，見 [§4](c_user_manual_zh_TW_01_lifecycle.md#4-生命週期) 記憶池表）。新程式請直接寫 `delay_mode`。 |
| `enable_highpass` | `1` | mic 路徑的高通濾波（見 `highpass_cutoff_hz`）。上游已有高通時可關。 |
| `enable_saturation` | `1` | 削波偵測與參考訊號軟限幅。建議維持開啟。 |
| `enable_shadow` | `1` | 輔助 filter，用於偵測回音路徑變化。關閉會少掉 42,880 B（16 kHz/256，見 [§4](c_user_manual_zh_TW_01_lifecycle.md#4-生命週期) 記憶池表），但路徑突變後的恢復會變慢。 |
| `saturation_softclip_ref` | `1` | 對參考訊號套用軟限幅。`enable_saturation = 0` 時無作用。 |
| `return_res_context` | `0` | 進階整合用，見 [§8](c_user_manual_zh_TW_04_seam_troubleshooting.md#8-進階整合-seam)。 |
| `spatial_linear_context` | `0` | 進階整合用，見 [§8](c_user_manual_zh_TW_04_seam_troubleshooting.md#8-進階整合-seam)。`1`：不計算這條 lane 自己的 `res_gain`。`2`（`AEC_CONTEXT_LINEAR_ONLY`）：呼叫端只讀 context 的線性半邊（`formed_hop`、`error_spec`、`echo_spec`、`near_spec`、`far_spec`、delay 與判定欄位）時使用，post 區塊在 AecState 更新後停止，跳過 R²／comfort noise／抑制增益；線性輸出與 `0`／`1` **逐位元相同**，`res_gain`、`r2`、`comfort_noise` 讀為 `NULL`、`res_floor_protect` 為 0，本機實測 `aec_process_context()` 省約 11–14 %。**只有在 `enable_res = 0` 且 `return_res_context = 1` 時才合法**，其他組合與 0..2 以外的值會被驗證拒絕。 |

### 6.3 抑制強度

| 欄位 | 型別 | 預設 | 允許範圍 | 說明／何時調整 |
|---|---|---:|---|---|
| `min_gain_floor_far_active_db` | `float` | `-28.0` | -300 – 50 | **mild/balanced preset 唯一的差別**，也是主要的強度旋鈕。遠端在說話時允許的最低增益。**調低**（例如 -38）＝回音壓得更乾淨，但 double-talk 時近端更容易被壓掉；**調高**（例如 -20）＝近端保留較好，回音殘留較多。優先用 preset（[§7](c_user_manual_zh_TW_03_preset.md#7-preset-選擇)）而不是自己填數字。 |
| `min_gain_floor_dt_db` | `float` | `-20.0` | -300 – 50 | double-talk 期間專用的最低增益，與上一項分開；aggressive preset 把這個欄位也調低到 `-30.0`（mild/balanced 仍共用 `-20.0`），不然 double-talk 觸發時三個 preset 會蓋回同一個值,聽不出強度差。**調高**可在雙方同時說話時多保留近端。需 `dt_aware_res_floor_enabled = 1` 才生效。 |
| `dt_aware_res_floor_enabled` | `int` | `1` | 0 / 1 | 是否啟用上面這個 double-talk 專用下限。 |
| `dt_aware_recovery_soft` | `int` | `1` | 0 / 1 | 近端最近有聲音時，改用非破壞性的方式做重新對齊，避免把近端語音削掉。 |

### 6.4 延遲估計

| 欄位 | 型別 | 預設 | 允許範圍 | 說明／何時調整 |
|---|---|---:|---|---|
| `delay_mode` | `AecDelayMode` | `AEC_DELAY_MATCHED` | `MATCHED` / `FIXED` / `EXTERNAL_ALIGNED` | **對齊方式的唯一真實來源**（見 [§3](c_user_manual_zh_TW.md#延遲契約) 延遲契約的三態表）。init 時決定、instance 存活期間不可變（各 mode 的 pool 佈局不同，要改必須 destroy + 重新 init）。非法組合一律拒絕。 |
| `fixed_delay_samples` | `int` | `-1` | `-1`，或 `FIXED` 時 0 – `120 × sample_rate` | **只在 `FIXED` 有意義**：bring-up 量測到的系統延遲，單位是 native-rate sample。其他 mode 必須維持 `-1`（帶值會被拒絕，不會被忽略）。它同時**決定參考環形緩衝的大小**：`FIXED` 只配置 `fixed_delay_samples + hop_size` 個 sample（緊上界，見 [§4](c_user_manual_zh_TW_01_lifecycle.md#4-生命週期) 記憶池表的公式），`max_delay_ms` / `delay_buffer_ms` 在此 mode 無效。 |
| `max_delay_ms` | `float` | `1024.0` | 0 – 60000 | **`MATCHED` 專用**：參考訊號搜尋環形緩衝的尺寸下限（ms），不是搜尋上限（見 [§3](c_user_manual_zh_TW.md#延遲契約) 延遲契約）。調高只是多配記憶體，不會提高可擷取的延遲。`FIXED` / `EXTERNAL_ALIGNED` 下完全無效（見 [§4](c_user_manual_zh_TW_01_lifecycle.md#4-生命週期) 環形緩衝大小公式）。 |
| `delay_buffer_ms` | `float` | `2048.0` | 0 – 120000 | **`MATCHED` 專用**：參考訊號搜尋環形緩衝長度（ms）。實際樣本數取 `delay_buffer_ms` 與 `max_delay_ms + 4096 samples` 兩者較大者。調高 `max_delay_ms` 時通常要一併調高。`FIXED` / `EXTERNAL_ALIGNED` 下完全無效。 |
| `delay_est_init_s` | `float` | `0.3` | 0 – 3600 | 延遲估計值需維持穩定多久才視為已鎖定（秒）。 |
| `delay_est_period_s` | `float` | `0.0` | 0 – 3600 | matched filter 的分析週期（秒），算力節流旋鈕。`0`（預設）＝每個 hop 都分析（全速）。`> 0` 時，延遲估計值維持 solid 且穩定 `delay_est_init_s` 後，matched filter 只每 K 個 hop 實跑一次（K ＝ `round(period ÷ hop 時長) ÷ 5`，整數除法、下限 1；其餘 hop 仍餵入資料、只是不分析）；估計變得不穩、或 windowed ERLE 較峰值下滑超過 6 dB 時回到全速。調大省 MIPS、但延遲變動偵測變慢，也可能改變 matched filter 鎖到的 lag。 |
| `delay_num_filters` | `int` | `5` | 1 – 5（**只在 `MATCHED`**；其他 mode 必須維持 5） | Matched-filter bank 大小（決定記憶體與可搜尋的延遲範圍）。可靠搜尋上限隨之縮小：n=1→125ms／2→221ms／3→317ms／4→413ms／5→509ms；預設 `delay_est_period_s = 0`（每 hop 全速分析）時，每少一組省 ~4.2 MMAC/s 的搜尋算力，**並省 5,728 B 記憶池**（bank、render ring、兩個 histogram 都依 n 從記憶池切出，見 [§4](c_user_manual_zh_TW_01_lifecycle.md#4-生命週期) 記憶池表）。縮小的正確場景是**上游已先提供 coarse-compensated far**（HAL/系統層自行移除 bulk delay 後才餵進來）、matched filter 只需追殘差的 `MATCHED` 部署——注意這與 `FIXED` 無關：`FIXED`（`fixed_delay_samples`）根本不建 matched filter，`EXTERNAL_ALIGNED` 連 estimator/ring 都沒有。bench/dataset 一律維持 5（見 `delay_estimator_design_zh_TW.md` §5）。`FIXED` / `EXTERNAL_ALIGNED` 根本沒有 matched filter，改這個值會被拒絕（不是忽略）；`0` 在任何 mode 都是錯誤，不是「關閉延遲估計」的入口。 |
| `delay_acquire_protect_converged` | `int` | `1` | 0 / 1 | filter 已收斂時，保護它不被**首次**延遲擷取破壞（Path A）。 |
| `delay_acquire_warm_transfer` | `int` | `1` | 0 / 1 | 首次擷取到延遲時，把已學到的回音模型平移過去而不是歸零，避免開場一秒左右出現一段明顯回音。建議維持開啟。 |
| `delay_acquire_inst_erle_db` | `float` | `4.0` | -100 – 100 | 上一項的觸發門檻（dB）：既有模型要夠好才值得平移。 |

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

`ne_recent_sustain`（預設 `3`，範圍 0 – 1000000）**不會**被換算，固定維持字面值
`3`，不要跟上表混淆——但原因不是它是真正的次數計數：它的計數器每個 hop 都會遞增或
歸零、沒有跳過條件，所以「連續 3 次」就是連續 3 個 hop 的時長，會隨格點變動（16 kHz/
256 預設格點約 24 ms，16 kHz/512 約 48 ms，48 kHz/1024 約 32 ms）。固定字面值 3 是
因為三個格點目前各自的行為都已個別驗證過；換算成單一時長會讓 16 kHz/512 格點變成
只需 2 個 hop，量測後會在該格點引入新的回音誤觸發。

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
| `epc_mu_floor` | `float` | `0.5` | 0 – 10 | **保留欄位，C 庫內沒有任何使用處**（只有宣告、預設值與範圍驗證），調它不會有任何效果。 |
| `kalman_q_high` | `float` | `1e-3` | 0 – 1 | 未收斂／路徑改變時的模型不確定度。 |
| `kalman_q_low` | `float` | `1e-6` | 0 – 1 | 已收斂後的模型不確定度。 |
| `ne_recent_threshold` | `float` | `0.3` | 0 – 1000 | 判定近端有聲音的能量門檻。近端偵測過鈍可調低。 |
| `filter_misadjustment_scale_min` | `float` | `0.5` | 0 – 1000 | 回音估計修正倍率下限。 |
| `filter_misadjustment_scale_max` | `float` | `2.0` | 0 – 1000 | 回音估計修正倍率上限。 |

### 6.6b 逐階段耗時（`aec_get_last_timing`）

```c
typedef struct AecStageTiming AecStageTiming;   /* delay_us / frontend_us / linear_us / steering_us / res_us */
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
  `aec_process_context_shared_far_from()` 借用），FFT 也算在這裡。
- `steering_us`：主濾波器與 post 區塊之間的全部工作——stationarity refresh、
  `e2_coarse`/ERL publish 與 coarse rescue、double-talk analyzer、EPV、
  `shadow_rise`、misadjustment estimator。它們都不直接動樣本，只決定下一個
  hop 怎麼適應，但確實是每個 hop 的真實成本。
- `res_us`：AEC3 post/RES 區塊。只要 `enable_res` **或** `return_res_context`
  其中之一為真就會跑，所以 context-only 的呼叫端也會看到真實數字。

五者**刻意不等於**整個呼叫：餘額是 `res_us` 收尾之後才跑的 per-sample power
EMA 與收斂偵測。要呈現完整分解的呼叫端得自己把餘額算出來。

每次 `aec_process*()` 進入時清零，所以一個 hop 只會回報它真的跑到的階段。
解析度是微秒，低於 1 µs 的階段會讀到 0。

#### 預設關閉

⚠ **量測要建置時開啟**，否則五個欄位每個 hop 都讀 0：

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

包裝這個函式庫的上層元件各自有同名慣例的覆寫點（例如 Audio_ALG 的 4 聲道
線性 AEC + BF 單元用 `FOUR_AEC_NR_RES_NOW_US`），刻意不共用：一個元件的
時鐘屬於該元件自己的建置契約。

### 6.6c Live observer（實驗性）

**預設關閉（`live_observer_enabled = 0`），關閉時輸出逐位元不變。未進任何
出貨 preset，不要在產品路徑上開啟。**

機制：第三個獨立 PBFDAF 實例（NLMS，跟 `shadow_filter` 同一個 struct），讀
遠端 ring 比主濾波器的延遲補償讀取早 `live_observer_shift_ms`。每 hop 把
自己的誤差能量 EMA 跟主濾波器（R² 實際採用的那條選線，不是恆取 main）的
誤差能量 EMA 比較，只有在觀察器沒有比主濾波器差（corroboration）時才信任
其證據；證據夾在主濾波器殘留回音功率以內，經 `max(r2, evidence)` 注入
residual-echo R²（只到 `live_observer_max_hz` 的低頻段），從不碰輸出、
delay 或任一濾波器的視窗／taps。

| 欄位 | 型別 | 預設 | 允許範圍 | 說明 |
|---|---|---:|---|---|
| `live_observer_enabled` | `int` | `0` | 0 / 1 | 總開關。 |
| `live_observer_shift_ms` | `float` | `24.0` | 0 – 1e9 | 觀察器讀遠端 ring 比主濾波器早多少 ms。 |
| `live_observer_max_hz` | `float` | `1000.0` | 0 – 1e9 | 證據注入 R² 的頻率上限；高於此頻率的 bin 永遠不受影響。 |
| `live_observer_mu` | `float` | `0.3` | (0, 2] | 觀察器自己的 NLMS 步進。 |
| `live_observer_corrob_ms` | `float` | `100.0` | 0 – 1e9 | corroboration EMA 的時間常數。 |
| `live_observer_corrob_margin_db` | `float` | `0.0` | 必須是有限值 | corroboration 容忍度（dB）；觀察器誤差 EMA 超過主濾波器誤差 EMA 這個容忍倍率就不信任。 |

記憶體增量（`aec_get_mem_size()` 實測，delay_num_filters=5，balanced
preset）：8 kHz/256 +29,472 B、16 kHz/256（預設）+38,768 B、16 kHz/512
+58,656 B、48 kHz/1024 +141,648 B。算量增量用 `bench_rtf --live-observer`
在目標格點量測（`make bench` 建置）；固定開銷來自觀察器每 hop 3 個與分接數
無關的 FFT。

### 6.7 靜態記憶體拆解（`aec_get_mem_breakdown`）

```c
typedef struct AecMemBreakdown {
    size_t total_bytes;       /* == aec_get_mem_size(cfg)                 */
    size_t estimator_bytes;   /* delay_aec_get_mem_size(); 0 非 MATCHED  */
    size_t ring_bytes;        /* 對齊環形緩衝；0 為 EXTERNAL_ALIGNED     */
} AecMemBreakdown;

int aec_get_mem_breakdown(const AecConfig* cfg, AecMemBreakdown* out);
```

把 [§4](c_user_manual_zh_TW_01_lifecycle.md#4-生命週期) 記憶池大小表格背後的算法變成一個可查詢的 API：`total_bytes` 就是
`aec_get_mem_size(cfg)` 本身（同一次呼叫算出來，不是另一套算式），
`estimator_bytes`／`ring_bytes` 是這個總量裡兩個有名字的子集，用的是
`aec_get_mem_size()` 內部本來就在用的同一批 helper（`delay_aec_get_mem_size()`、
`aec_derive_dims()` 的 ring 大小 out-param）——查詢這個函式不可能得到與
`aec_carve()` 實際切出來的 pool 佈局不同的答案。失敗條件與
`aec_get_mem_size()` 完全一致：`cfg` 為 `NULL` 或未通過 `aec_validate_config()`
時回傳 0（並把 `*out` 清零）。`example/aec_wav.c` 的 `--print-mem-size`
（[附錄 A](c_user_manual_zh_TW_05_appendix.md#附錄-aaec_wav-命令列工具)）就是這個 API 的一個瘦封裝。

---

> [索引](c_user_manual_zh_TW.md) · 上一部分：[c_user_manual_zh_TW_01_lifecycle.md](c_user_manual_zh_TW_01_lifecycle.md) · 下一部分：[c_user_manual_zh_TW_03_preset.md](c_user_manual_zh_TW_03_preset.md) <!-- split-nav -->
