[C 使用手冊](../c_user_manual_zh_TW.md)

## 8. 進階整合 seam

一般使用者不需要這一節。只有在你要自己接上外部的噪音抑制／後處理，或是要驅動多個
實例時才會用到。

### 8.1 三個處理進入點與 `AecConfig.residual`

一般使用只呼叫 `aec_process()`。要自己接外部的增益（NR、NN 後濾波、多 lane 融合）時，
把一個 hop 拆成兩段：`aec_process_linear()` 跑線性 stage（前端、delay、shadow 與主 filter），
`aec_process_residual()` 緊接著跑後級（FORM、AecState、r2、舒適噪音、`G_res`，做到 `AecConfig.residual` 允許的程度）。
後級**不套用任何增益、不合成**：結果從 `aec_get_residual()` 讀，由呼叫端自己乘增益、加舒適噪音、做反轉換
（可用 `aec_apply_gain()`／`aec_apply_cng()`，見 §8.3）。

`AecConfig.residual`（`AecResidualMode`，預設 `AEC_RESIDUAL_FULL`）取代舊版 `AecConfig` 的三個後級旗標（舊名與對照見 [`sdk_migration_output_guard.md`](../sdk_migration_output_guard.md) §3）：

| 模式 | 後級做到哪裡 |
|---|---|
| `AEC_RESIDUAL_OFF` | 只有線性 stage；沒有後級，`aec_process_residual()` 與 `aec_get_residual()` 回 `AEC_ERR_MODE`。`aec_process()` 的 `out` 是線性殘差。 |
| `AEC_RESIDUAL_STATE` | 被選中的線性殘差 E(f) 與其回音、近端頻譜、渲染底座、`usable_linear` 與狀態旗標；沒有 `r2`、舒適噪音、增益。 |
| `AEC_RESIDUAL_ESTIMATE` | STATE 加上殘留回音功率譜 `r2` 與舒適噪音功率譜（增益由別處算的 lane）。 |
| `AEC_RESIDUAL_FULL` | ESTIMATE 加上抑制增益 `G_res`；`aec_process()` 會套用它，並加舒適噪音、做函式庫自己的合成。 |

三個進入點（回傳 `AecStatus`：`AEC_OK`、`AEC_ERR_ARG`、`AEC_ERR_MODE`、`AEC_ERR_SEQUENCE`）：

- `aec_process(a, mic, ref, out)`：線性 + 後級；`FULL` 時 `out` 是完成的 hop，低於 `FULL` 時 `out` 是線性殘差。
- `aec_process_linear(a, mic, ref)`：第一段；`OFF` 時這一呼叫就結束整個 hop。
- `aec_process_residual(a, NULL)`：第二段，必須緊接在同一個 hop 的 `aec_process_linear()` 之後（`beam` 參數給陣列實例，單聲道傳 `NULL`，見 [§8.8](#88-麥克風陣列實例n_mics--4)）。

線性輸出隨 `AecConfig.residual` 而不同：`OFF` 使用較簡單的步長規則，`STATE`／`ESTIMATE`／`FULL` 共用同一條線性路徑，所以要拿線性部分與完整鏈比對時，只能在 `STATE` 以上比。

```c
AecConfig cfg;
aec_config_from_level(&cfg, AEC_LEVEL_BALANCED, 16000);
cfg.residual = AEC_RESIDUAL_FULL;   /* 後級算出 r2 / 舒適噪音 / G_res，但由呼叫端套用 */
Aec *aec = aec_create(&cfg);

while (app_read_hop(mic, ref, hop)) {
    aec_process_linear(aec, mic, ref);
    aec_process_residual(aec, NULL);

    AecLinearView lin;  lin.struct_size = sizeof lin;
    AecResidualView rv; rv.struct_size  = sizeof rv;
    aec_get_linear(aec, &lin);
    aec_get_residual(aec, &rv);
    /* 在這裡讀 lin／rv，必須在下一次處理呼叫之前 */
}
```

舊旗標組合到新模式的對照（計算與位元結果不變）在 [`sdk_migration_output_guard.md`](../sdk_migration_output_guard.md) §3。

> **分析 seam 與渲染底座是兩組不同的頻譜。**
>
> - **分析 seam**：`formed_hop`、`error_spec`、`echo_spec`、`near_spec`（`= error_spec + echo_spec`）。
>   任何進入點與 `residual` 模式（`aec_process()`、split、`STATE`／`ESTIMATE`／`FULL`）下都是被選中
>   （refined 或 coarse）的**真實線性殘差**及其配對的回音估計；輸出守衛永遠不改它們。`error_spec` 是
>   `[前一個 formed hop | 本 hop 的 formed hop]` 的 STFT，所以 unity IFFT 加 overlap-add 在每個 hop
>   （包含守衛觸發的 hop）都重建「前一個 formed hop」，晚一個 framing hop。AecState、ERLE、R² 與各 PSD
>   都讀這一組；AIAEC 的 `linear_error`、訓練特徵或任何需要可配對波形／STFT 的消費端也讀這一組。
> - **渲染底座**：`aec_get_residual().base_spec`（§8.3）。輸出守衛在**每個 hop、每個模式**求值一次，時點在
>   本 hop 的 usable 判定確定之後、任何 context 提前離開之前。規則（`output_capture_when_linear_unusable`，
>   預設開啟）：本 hop 線性估計不可用，且 `sum(|error_spec|²) > sum(|error_spec + echo_spec|²)`
>   （嚴格大於、加窗頻譜、float32 pairwise 歸約，C 與 Python 的歸約相同，所以相同頻譜得到相同決定；兩邊各自算 STFT，上游頻譜約差 1e-7 相對值，所以 C 對 Python 的契約是：除了雙方都 unusable 且相對能量差 `|Σ|E|²−Σ|Y|²|/Σ|E|²` 落在 1e-6 tie band 內的 hop，每個 hop 的決定都一致；band 內的 hop 在對拍的每個 cell 逐一計數並印出，band 外的不一致即失敗，固定的近門檻 fixture 要求完全相等；同樣的捨入也可能落在自適應濾波器的逐 bin 判定上（refined 濾波器 H_error leakage 比較 `e2_refined[k] <= e2_coarse[k]`，或噪聲門 `X2[k] >= gate`），這類 cell 在 `pyc_parity.py` 登記為 T-01。T-01 是已知的數值分岔，不是等價：照跑時 C 與 Python 的原始輸出可以超過音訊容許值（`revert.rOza.16k512` 最大差約 0.0527），只有強制判定的實驗能說明差異來自登記的平手；只有在守衛、serial、delay 閘門照跑全綠，且 Python 依 C 的方向強制登記的判定後每個閘門都綠時才是 INFO；`res_gain` 有三道閘門：以各 bin 近端功率 `|E + echo|²` 加權的相對 L2（上限 1e-4）、距該 hop 近端功率峰值 20 dB 內各 bin 的最大差（上限 3e-3），以及在有輸出底座的 cell 上，應用實際渲染的 `output_base_spec * res_gain` 逐 hop 相對 L2（距最響 hop 60 dB 內的每個 hop，上限 3e-3），全部 bin 的最大差只印出；每一份比較的資料都必須非空、兩邊等長且是有限值，否則該 cell 失敗），底座就是擷取訊號
>   `error_spec + echo_spec`（`guard_active = 1`），否則底座是 `error_spec`。自己合成音訊的消費端要渲染
>   **`output_base_spec × 增益`**，不是 `error_spec × 增益`；全模式的 `aec_process()` 內部就是這麼算，
>   所以拆開的路徑與一體的路徑得到同樣的音訊（`aec_apply_gain()` 就是對底座乘增益）。底座**不**與 `formed_hop` 配對：守衛 hop 上它是頻域的
>   切換，對它做 unity 反轉換不會重建任何 formed hop。
>
> 守衛只在線性估計不可用時才可能觸發，因此多半發生在第一個 usable 判定之前，但從未變成 usable 的片段
> 會一直觸發，不只在開頭幾秒。
>
> 既有 SDK 應用若還在渲染 `error_spec`，編譯與執行都不會報錯，但會失去守衛；遷移步驟、前後程式碼與驗收測試見
> [`sdk_migration_output_guard.md`](../sdk_migration_output_guard.md)，並以 `AEC_OUTPUT_GUARD_API_VERSION`（`aec.h`）在建置期檢查。

> **兩種形式可以逐 hop 混用。**
>
> `aec_process()` 與 `aec_process_linear()` + `aec_process_residual()`
> 推進完全相同的濾波器與偵測器狀態；差別只在函式庫自己的合成：overlap-add 尾巴與舒適噪音產生器只在
> `aec_process()` 與 `aec_apply_cng()` 推進。同一個實例上可以任意交錯呼叫，中間不需要 `aec_reset()`，
> 但交錯後 `aec_process()` 的 `out` 不再等於一路 `aec_process()` 的 `out`（缺了拆開那幾個 hop 的 OLA 尾巴）；
> 每個 hop 的 `AecLinearView`／`AecResidualView` 則逐位元相同。
>
> 選哪一個純粹看誰來套增益：下游自己乘增益時用拆開的形式，省掉函式庫的合成。

### 8.2 多實例共用遠端 FFT

多個實例每個 hop 都看到**完全相同**的參考訊號時，可讓其中一個算 FFT、其餘共用：

```c
Aec *lane[4];                                    /* 各自 aec_create(&cfg)（單聲道、同一組態）得到的指標 */
const float *mic[4];                             /* 每條 lane 本 hop 的麥克風 */
int n_lanes = 4;
aec_process_linear(lane[0], mic[0], ref);        /* lane 0 自己算 */
for (int ch = 1; ch < n_lanes; ++ch)
    aec_process_linear_shared_far_from(lane[ch], mic[ch], ref, lane[0]);   /* aec_internal.h */
/* 之後每條 lane 各自 aec_process_residual(lane[ch], NULL)（residual 不是 OFF 時） */
```

契約：

- `far_owner` 必須先完成同一個 hop，且處理的是**同一段參考時域訊號**；借用的是它的遠端頻譜及其 AEC magnitude²。
- 即使借用了遠端頻譜，`ref` 參數**仍然必須傳**（參考訊號的非 FFT 用途還在）。
- `aec_far_fft_real_compute_count()` 是唯讀計數器，可用來驗證共用真的生效
  （純共用的實例會一直維持 0）。

`far_owner` 必須先完成同一 hop，且各 lane 的 far-buffer overlap history 必須一致；
只有本 hop 的 `ref` 相同還不夠。若同一 hop 有 lane 採 warm realign、其他採 soft reset，
該 hop 應讓各 lane 自行計算，下一 hop 歷史恢復同步後再共用。
各 lane 仍保留自己的 spectrum、magnitude history 與 `x2_cache`，不保留 owner pointer；
previous-hop RSA 不會提前讀到 producer 的 current-hop history。
`NULL`、自身或 grid 不相容的 owner 會退回本地計算；其他 stale-owner／歷史錯配由呼叫者負責。
AEC magnitude² 必須來自同一個 AEC kernel，不能任意以其他 power 計算取代。

### 8.3 `AecResidualView` 與 `AecLinearView`

兩個唯讀檢視都在該 hop 的處理呼叫之後、下一次處理呼叫之前讀。`struct_size` 是**輸入**（先設成
`sizeof` 該結構）：比已知結構小就回 `AEC_ERR_ARG` 且**完全不寫**呼叫端的結構；較大的值（較新的呼叫端結構）可接受，
只寫已知的前綴，回傳時 `struct_size` 為已知結構的大小。

`aec_get_residual(a, &rv)`：後級對最後一個 hop 所知的一切。指標欄位隨 `AecConfig.residual` 而定：
`STATE` 有 `formed_hop`、`error_spec`、`echo_spec`、`near_spec`、`base_spec` 與旗標，`r2`／`comfort_noise`／`res_gain` 為 `NULL`；
`ESTIMATE` 多 `r2`、`comfort_noise`；`FULL` 再多 `res_gain`。`residual == OFF` 回 `AEC_ERR_MODE`（檢視除
`struct_size` 外清零）；後級尚未跑（拆開的第一段之後）回 `AEC_ERR_SEQUENCE`。

| 欄位 | 說明 |
|---|---|
| `n_freqs` / `hop_size` | 陣列長度。所有頻域陣列長度都是 `n_freqs`。 |
| `error_spec` | formed 線性輸出的頻譜 E(f)，**audio 尺度**（與 [-1,1] 時域一致）。是分析 seam：任何模式下都是真實的線性殘差，輸出守衛不會改它（見 §8.1）；要渲染音訊請用 `base_spec`。 |
| `echo_spec` / `near_spec` | 對應的回音估計與擷取訊號頻譜，滿足 `near_spec = error_spec + echo_spec`。 |
| `far_spec` | 主 filter 自己的矩形區塊遠端頻譜（不是 sqrt-Hann，不是鏈上標準頻譜）。 |
| `base_spec` / `guard_active` | 渲染底座（守衛已套用）與本 hop 守衛是否觸發。 |
| `res_gain` | 抑制增益，振幅 [0, 1]。**低於 `FULL` 時為 `NULL`** —— 代表「沒有計算」，若誤讀成全 0 會變成「全部靜音」。**使用前務必檢查非 NULL。** |
| `r2` | 殘留回音功率譜。**int16² 尺度**：要換成 audio 功率尺度須除以 32768²。`STATE` 時為 `NULL`。 |
| `comfort_noise` | 舒適噪音功率譜。同樣是 int16² 尺度。`STATE` 時為 `NULL`。 |
| `formed_hop` | 時域 formed 線性殘差，與 `error_spec` 配對。 |
| `res_floor_protect` | 本 hop 是否啟用 double-talk RES floor 保護。它就是近端最近有聲音 latch 的實際決定；自己做後級的消費端應讀這個欄位，不要用 `dt_indicator` 另外推導一套規則（陣列實例：`max_dt > 0.2`，`max_dt` 只取權重非零的 lane；單聲道：近端最近有聲音的 latch）。 |
| `usable_linear` | 本 hop 的可用線性估計判定。`0` 時 `r2` 是保守的非線性估計（遠端功率乘上預設 echo-path 增益），並非由線性回音估計推導，可能遠高於根本不存在的殘留回音。 |
| `post_reset_serial` | 與 `aec_post_reset_serial()` 相同的計數（見下）。 |
| `all_converged`、`max_dt`、`max_saturation`、`clock_drift`、`delay_changed`、`delay_locked`、`guard_lanes`、`far_hop` | 後級增益階段本 hop 的輸入摘要（單聲道：來自自己的線性 stage），給自己做後級的消費端使用。 |

`aec_get_linear(a, &lin)`（`AecLinearView`）：線性 stage 的檢視，`residual == OFF` 時就是整個 hop 的結果。
含 `linear_hop`（refined PBFDKF 輸出，FORM 選擇之前）、主 filter 的回音／近端／遠端頻譜、`aligned_far_hop`、delay 狀態（§8.6），
以及線性 stage 的純量狀態：`far_power`、`erle_factor`、`dt_indicator`、`divergence`、`saturation_level`、`erl_estimate`、
`shadow_dt`、`is_stationary_dt`、`filter_converged`、`filter_once_converged`、`epc_active`（供外部判斷／診斷用）。

**增益輔助函式與 reset serial**

```c
Complex out_spec[N];                       /* N >= n_freqs */
aec_apply_gain(aec, extra_gain_or_NULL, out_spec);   /* out = base_spec * min(G_res, extra_gain)；aec 是 Aec* */
aec_apply_cng(aec, extra_gain_or_NULL, out_spec);    /* 在被增益切掉的 bin 加舒適噪音 */
uint32_t serial = aec_post_reset_serial(aec);        /* 不消耗、可重複讀、uint32 回繞 */
```

- `aec_apply_gain()` 需要 `AEC_RESIDUAL_FULL` 且該 hop 已處理完（否則 `AEC_ERR_MODE`／`AEC_ERR_SEQUENCE`）；
  `extra_gain`（例如 NR 的增益，n_freqs 個 [0,1] 值）可為 `NULL`，那就是函式庫自己的輸出階段。
  `aec_apply_cng()` 用實例內的產生器（種子 `AEC_CNG_SEED`，每次硬 post 鏈 reset 重設），傳入與 `aec_apply_gain()` 相同的
  `extra_gain`；要不要加舒適噪音（`enable_cng`）由呼叫端決定，不要的話不呼叫即可。
  無實例的形式 `aec_apply_gain_spec()`／`aec_apply_cng_spec()` 用於呼叫端自己持有增益、底座與產生器狀態時
  （`aec_cng_fill()`（`aec_internal.h`，§8.5）是底層實作）。
- `base_spec` 指向一塊專用緩衝區（守衛沒觸發時就是 `error_spec`），CNG 與其他暫存區不會覆寫它，
  有效到下一次處理或 reset 呼叫。
- `aec_post_reset_serial()` 計數「硬 post 鏈 reset」，共兩類來源：
  1. `aec_reset()`：**獨立的一次呼叫，不在任何處理呼叫之內**；它回傳時 serial 已經 +1，下一次處理呼叫從清空狀態開始。
  2. 處理呼叫（`aec_process`／`aec_process_linear`）**內部**的確認 delay 變更使 filter 重啟：硬位移（Path B，沒有
     DT-aware soft realign）、沒有 warm transfer 也沒有 DT-aware soft 取得的首次取得（Path A）。這類 reset 發生在
     該 hop 的處理內部、該 hop 的合成**之前**。
  每次計數時函式庫都會清自己的 OLA 尾巴並把 CNG 種子設回 `AEC_CNG_SEED`（42）。
  **不**算的有：bad-relock 還原（soft 還原到 shift 前的 delay 與 tap 快照，OLA 與 CNG 照常延續）、warm-transfer 與
  soft（DT-aware）取得／realign、`aec_apply_external_realign()`、只有 `AecLinearView.generation` 變動。
- **外部合成的協定**：消費端自己保存「上次處理過的 serial」。每次處理呼叫後讀到的值若與它不同，必須在
  **本 hop 的第一次 CNG 抽樣之前、也在本 hop 的反轉換／OLA push 之前**（`aec_reset()` 之後則是之後處理的第一個 hop），清掉自己的 OLA 尾巴（例如
  `ac_istft_reset`）並把自己的 CNG 狀態設成 `AEC_CNG_SEED`，然後存下新值。**不可**因此呼叫
  `aec_reset()`。用 `aec_apply_cng()` 時，產生器在實例內，函式庫自己重設；OLA 尾巴仍是消費端的。函式庫不動 NR 與 NN 的狀態，它們要不要跟著 reset 是應用的政策。

> ⚠️ 所有 pointer 欄位都直接指向實例內部的 per-hop 緩衝區，**不是複本**。
> 不可修改、不可 free，而且必須在**下一次對同一實例的任何處理呼叫之前**讀完或複製走。
> 處理進入點（`aec_process`、`aec_process_linear`）都會就地覆寫這些緩衝區。

### 8.4 Duty-cycle engagement census

`AecDebugStatus`（`aec_debug_status()`）含兩個計數器：

```c
unsigned long long duty_hops_total;  /* 經過 delay 區塊的 hop 數        */
unsigned long long duty_hops_run;    /* 其中真的跑了 matched filter 的數 */
```

duty-cycle 機制在估計穩定（confidence 1.0）且 `delay_est_init_s` 秒不變後，
把 matched-filter 分析降到 1-in-K；估計持續變動的回音路徑上機制每次都要
重新武裝，實際降頻比例取決於回音路徑。這兩個計數器提供**實測**
engagement（`duty_hops_run / duty_hops_total`）。

- 降頻週期是函式庫內部值，預設 0（每個 hop 全速分析），沒有公開設定；預設組態下 `duty_hops_run == duty_hops_total`。
- 只在 `delay_mode == AEC_DELAY_MATCHED` 時累積（`has_delay == 1` 才會進入
  計數的程式路徑）；`FIXED`／`EXTERNAL_ALIGNED` 從未建過 matched filter，
  兩個計數器永遠是 0——不是「估計器跑了但沒省到」，而是根本沒有機制可省。
- 累積範圍是「自 `aec_create`／`aec_init` 或最近一次 `aec_reset()` 以來」，
  與 `AecDebugStatus` 其餘欄位的累積規則一致；`aec_reset()` 會把兩個計數器
  歸零，因為跨越 reset 的比值會把兩段不同的 duty 狀態混在一起，沒有意義。
  在計數的 call site（消耗 `run_filter` 的那一行）遞增，與實際執行的機制一致。
- **純診斷**：兩個計數器只被寫入、從未被函式庫自己讀回，加起來是熱路徑上
  兩個整數遞增，不影響任何音訊輸出。
- `example/aec_wav.c` 每次執行結尾印一行（[附錄 A「Duty census 診斷輸出」](wav_cli.md#duty-census-診斷輸出)）。
- 永久回歸測試：`make test-duty-census`（`test/test_duty_census.c`）。


### 8.5 舒適噪音填補（`aec_apply_cng_spec`）

> 殘留回音抑制增益沒有獨立的公開 API：單聲道的 `G_res` 在 `Aec` 裡（`aec_get_residual().res_gain`，見 §8.7），
> 四麥克風線性 AEC + BF 之後的那一次 `G_res` 在陣列實例裡（[§8.8](#88-麥克風陣列實例n_mics--4)）。
> 強度旋鈕一律是 `aec_set_level()`。

`aec_apply_cng_spec()`（宣告在 `aec.h`）是 `aec_apply_cng()` 的明確陣列形式，讓呼叫端自己持有產生器狀態；它與 `Aec` 內部輸出階段共用**唯一**的舒適噪音實作 `aec_cng_fill()`（宣告在 `aec_internal.h`，函式庫內部與測試用，`nr_floor` 由呼叫端選）：

```c
#include "aec.h"

aec_apply_cng_spec(&rng, spec, cn, g_res, g_nr, n_bins);
```

| 函式 | 說明 |
|---|---|
| `aec_apply_cng_spec(&rng, spec, cn, g_res, g_nr, n)` | 在 bin `1 … n-2` 加舒適噪音，每個分量加 `ng · (nf · 表值)`，乘積順序固定：`nf = sqrt(max(cn/32768², 0))`、`ng = sqrt(max(1-g_res², 0))`，有 NR 增益時 `ng ← ng · max(g_nr, nr_floor)`，即振幅 `sqrt(cn/32768²)·sqrt(1-g_res²)·max(g_nr, nr_floor)`。相位由 `rng` 遞推：每個 bin 一步、依 bin 由小到大（`rng = (rng·69069+1) & 0x7FFFFFFF`，取高 5 bit 查 32 點 √2·sin 表，虛部取 +8）；bin 0 與 n-1 不加噪音也不抽樣。`g_nr` 可為 NULL（整個因子省略）。種子用 `AEC_CNG_SEED`（42）。 |

內部以全增益 `G` 當 `g_res`、`g_nr = NULL` 呼叫 `aec_cng_fill()`，所以 `g_nr == NULL` 時公開呼叫與 `aec_process()` 內部的舒適噪音在相同輸入下逐位元相同
（同樣的 sqrt、同樣的乘積順序、同樣的抽樣排程）。`aec_apply_cng()`／`aec_apply_cng_spec()` 的 `nr_floor` 固定為
`AEC_CNG_NR_GAIN_FLOOR`。Python 孿生是 `modules/cng.py:cng_fill`。

RES 增益階段的實作（`src/aec_res.c`，內部標頭 `include/internal/aec_res_internal.h`，不安裝）供陣列實例使用，由 `make test-aec-res`
（設定與 lane 逐位元相等、API 契約、render ×32768 縮放、離開初始狀態、飽和後復原限制）、`make test-zero-heap`、
`make audit-no-stdio` 測試。完整頁面見 [`docs/html/reference/aec_res.html`](../html/reference/aec_res.html)。

### 8.6 線性對齊檢視（`aec_get_linear`）

給需要「與線性 filter 讀到的**同一段**對齊後參考訊號」的後級（頻域 NN 後濾波、
多 lane 的 RES）使用的唯讀檢視：

```c
AecLinearView lin;
lin.struct_size = sizeof lin;
aec_get_linear(aec, &lin);    /* 在該 hop 的處理呼叫之後、下一次處理呼叫之前讀 */
```

| 欄位 | 說明 |
|---|---|
| `hop_size`／`n_freqs` | `linear_hop`、`aligned_far_hop` 與頻譜陣列的長度 |
| `linear_hop` | refined 的 PBFDKF 線性輸出（FORM 選擇之前）；formed 線性殘差在 `aec_get_residual().formed_hop`（`residual` 不是 `OFF` 時） |
| `aligned_far_hop` | 本 hop 線性 filter 實際讀的參考訊號，**指向實例內部緩衝區**，不是複本，下一次處理或 reset 呼叫後失效 |
| `delay_samples`／`delay_confidence` | 目前套用的環形緩衝偏移（尚未取得對齊為 `-1`）與估計器的 0.0／0.5／1.0 信心 |
| `delay_state` | `AEC_LINEAR_DELAY_UNLOCKED`（`aligned_far_hop` 是**未補償**的原始參考訊號）、`LOCKED`、`CHANGED`（偏移在**這個 hop** 內改變，應清掉依賴舊對齊的快取） |
| `generation` | 每次環形偏移改變（首次取得、確認的位移、reset）加 1，飽和不回繞；用來判斷跨 hop 的快取是否失效，只比較 `delay_samples` 會漏掉兩次輪詢之間的 A→B→A |

`delay_mode` 對這些欄位的語意：`MATCHED` 如上；`FIXED` 的 `delay_samples` 恆等於
`fixed_delay_samples`、信心恆為 1.0、`generation` 只在 `aec_reset()` 變動，且前
`ceil(fixed_delay_samples / hop)` 個 hop 因為環形緩衝尚未填滿而回 `UNLOCKED`；
`EXTERNAL_ALIGNED` 的 `aligned_far_hop` 就是呼叫端自己的參考訊號，
`delay_samples` 為 0、恆為 `LOCKED`，`generation` 只被 `aec_apply_external_realign()` 與 reset 推進。

另一個唯讀查詢 `aec_linear_is_cancelling(a)` 回 `1` 代表線性 filter 此刻**有證據**正在
消除回音（上一個 hop 的 windowed ERLE 超過 2.5 dB，或近期 inst-ERLE 峰值超過
`delay_acquire_inst_erle_db`），回 `0` 只代表沒有證據，不代表 filter 壞掉。

### 8.7 單聲道整鏈：AEC → NR → RES 增益融合 → 合成

單聲道沒有單一的包裝函式：`Aec`、`AnrDenoiser`、合成用的 `AcIstft` 由整合者各自初始化，
只有串接順序是整合者的責任。Audio_ALG 的 `pipelines/mono/aec_nr_res/main.c`（每 hop 的增益序列
是該檔的 `mono_form()`）就是用上面的公開介面接起來的範例；每個範例 `main` 自己初始化它用到的函式庫並寫出這段序列。
AEC 以 `AecConfig.residual = AEC_RESIDUAL_FULL`（預設）建立並用拆開的兩段進入點驅動
（所以 `res_gain`、`r2`、`comfort_noise` 都有值），AEC 自己**不**把抑制套到時域，
抑制增益留給整合者與 NR 的增益合併後一次套用；單聲道的 RES 就在 `Aec` 裡面：增益是
`Aec` 自己算的 `rv.res_gain`（由 `aec_get_residual()` 取得），整合者不需要另外建立增益階段。
每個 hop 的呼叫順序：

```
mic, ref (hop 個 float)
  │
  ▼  aec_process_linear(a, mic, ref)           線性 stage
  ▼  aec_process_residual(a, NULL)             post 區塊（FORM、r2、舒適噪音、G_res）；不寫 out
  ▼  aec_get_residual(a, &rv)                  error_spec / near_spec / far_spec / r2 /
  │                                            comfort_noise / res_gain / 純量判定；
  │                                            rv.base_spec：渲染底座（守衛已套用）
  ▼  aec_post_reset_serial(a)                  與自己存的 serial 不同 → 先 ac_istft_reset 再繼續
  │                                            （實例內的 CNG 產生器由函式庫自己重設）
  ▼  aec_get_linear(a, &lin)                   aligned_far_hop、delay_state
  │      （以上指標都借自 AEC 的 per-hop 緩衝，下一次 AEC 處理呼叫前有效）
  ▼  anr_process_gain_with_residual_psd(nr,  NR 增益 g_nr：r2 / 32768² 併入噪音底
  │         rv.base_spec, r2 / 32768²)        （僅在 rv.usable_linear 為真時；否則 anr_process_gain(nr, rv.base_spec) 純降噪）
  │                                            ；回傳的內部陣列就是 g_nr；單聲道出貨鏈估計在 base_spec（見下方要點）
  ▼  aec_apply_gain(a, g_nr, spec)             spec[k] = rv.base_spec[k] · min(g_nr[k], res_gain[k])；不是 error_spec
  ▼  aec_apply_cng(a, g_nr, spec)              只填被 RES 切掉的 bin，量依 g_nr 縮放（下限 AEC_CNG_NR_GAIN_FLOOR）；
  │                                            產生器在實例內，起點 AEC_CNG_SEED
  ▼  ac_istft_push(istft, spec, out)           反 FFT + sqrt-Hann 合成 + OLA → hop 個 sample
```

要點：

- **哪一個進入點**：下游自己套增益時用 `aec_process_linear()` + `aec_process_residual()`，省掉函式庫的合成與 hop 長度的
  memcpy；只做 AEC（不接 NR／RES）的組態才用 `aec_process()` 直接寫 `out`。
- **增益取最小值，不是相乘**：`g_total = min(G_nr, G_res)`（`aec_apply_gain()` 的 `extra_gain` 就是 `G_nr`）。`G_res`（取 min 之前的值）
  同時決定舒適噪音的量，所以填補只反映 AEC 的抑制，NR 增益再以 `-10 dB`
  （`AEC_CNG_NR_GAIN_FLOOR`，`0.31622777f`）為下限縮放填補量。
- **渲染 `rv.base_spec`，不是 `rv.error_spec`**：底座已套用輸出守衛（§8.1），`aec_apply_gain()` 已經替你這麼做。
- **NR 的輸入頻譜**：函式庫不替你選；分析用的 `error_spec` 與渲染用的 `base_spec` 只在守衛 hop 上不同（其餘 hop 逐位元相同）。
  出貨的單聲道鏈（Audio_ALG `pipelines/mono/aec_nr_res`）把 `base_spec` 交給 `anr_process_gain*()`：NR 估計的就是增益要乘上去的那張頻譜。
  出貨的 4ch 鏈（`array_sample_anr_gain()`）交 `error_spec`，即 beamformer 輸出 E。`error_spec`／`formed_hop` 是 AI 模型與 ERLE 類統計
  用的分析 seam。Audio_ALG 之外的板端 sample 目前把 `error_spec` 交給 NR，這是已知差異，不是建議；新的整合以出貨單聲道鏈為準。
- **舒適噪音產生器在 `Aec` 裡**（`aec_apply_cng()`），起點是 `AEC_CNG_SEED`（42，`aec.h`），與 `aec_process()`
  內部相同，硬 post 鏈 reset 時函式庫自己重設。每個 hop 在 `ac_istft_push` 之前先比對 `aec_post_reset_serial()`：值變了就
  `ac_istft_reset()`、存下新值（§8.3）；**不要**為此呼叫 `aec_reset()`。自己持有產生器狀態、改走 `aec_apply_cng_spec()` 的整合者
  則要同時把 `rng` 設回 `AEC_CNG_SEED`。NR 狀態、暫存增益要不要一起清，是應用自己的政策，函式庫不動它們。
- **`aec_set_level()` 對這條鏈有效**：單聲道鏈的 `G_res` 來自 `Aec` 自己的
  post 區塊，所以強度旋鈕直接用 `aec_set_level()`（[§7.1](levels.md#71-執行期切換-levelaec_set_level)）；四麥克風陣列實例的
  `G_res` 在實例內，同樣用 `aec_set_level()`（它重設融合後增益階段的強度，lane 自己的強度沒有作用，§8.8）。
- 下游若另有噪音抑制或 CNG，不要在 `aec_apply_cng_spec` 之外再疊一層（[§6.2](configuration.md#62-殘留階段與舒適噪音)）。

測試：單聲道鏈的行為由 `Audio_ALG` 的 `pipelines/mono/aec_nr_res/tests`（`test_mono_chain.c`）涵蓋；本函式庫端的
對應契約是 `make test-aec-res`、`make test-linear-context`、`make test-process-context` 與 `make test-split-entry`。
元件頁見 [`docs/html/reference/aec.html`](../html/reference/aec.html)（細節頁 [`docs/html/reference/aec_detail.html`](../html/reference/aec_detail.html)）。

### 8.8 麥克風陣列實例（`n_mics = 4`）

`AecConfig.n_mics = AEC_ARRAY_MICS`（4）時，一次 `aec_init()`／`aec_create()` 建出整條四麥克風的線性流程：
四條 lane（每條是一個 `EXTERNAL_ALIGNED` 的單聲道 `Aec`）、**一個**共用的 delay 估計器與**一份**共用的對齊後 far，
以及融合緩衝，全部在同一塊呼叫端記憶池。Beamformer（SRP-PHAT、BF、VAD 等）**不在函式庫內**：呼叫端在 lane 與融合之間
自己算權重，再交給 `aec_process_residual()`；合成（反轉換與 overlap-add）同樣留在函式庫外。
本節只描述陣列實例本身。

**設定**（其餘的內部調校值原樣交給每條 lane）：

| 欄位 | 預設 | 陣列實例的規則 |
|---|---|---|
| `n_mics` | `1` | `1` 或 `AEC_ARRAY_MICS`（4），其他值被拒絕。 |
| `capture_proxy_channel` | `0` | `[0, 3]`：共用 delay 估計器的 capture 輸入取哪一路麥克風。單聲道必須是 `0`。`MATCHED` 時**只有這一路**餵給估計器；該路失效或被遮住，陣列就鎖不到 delay，四條 lane 以未對齊的 far 運作。 |
| `residual` | `FULL` | `STATE`／`ESTIMATE`／`FULL`；`OFF` 被拒絕。**陣列的記憶池大小隨它改變**（見 [§4](lifecycle.md#陣列實例的記憶池)）。 |
| `delay_mode`、`delay_num_filters`、`fixed_delay_samples` | 同單聲道 | 描述**共用**的對齊；每條 lane 一律以 `EXTERNAL_ALIGNED` 吃共用的對齊後 far。`MATCHED` 的共用環形緩衝為 `ceil(1024 ms × sample_rate / 1000) + 2 × hop + 1` 個 sample（`FIXED` 為 `fixed_delay_samples + hop_size`），duty-cycle 的保持時間固定為 0.3 s。 |
| `level` | `BALANCED` | 只作用在融合後的一次增益階段（`aec_set_level()` 同）；lane 自己的 level 沒有效果。融合增益的 DT 下限在最差 lane 的雙講指標高於 0.2 時一律啟用；lane 自己仍用內部的近端保護門檻決定回音路徑變化後的恢復方式。融合增益是獨立的增益階段（近端功率取 `|near|²`，所有貢獻 lane 都收斂後再以 `|E|²` 為上限；融合後的 r2 同時當有界與無界估計），不保證與單聲道實例的增益相同。 |
| `delay_backward_quarantine_enabled`、`delay_backward_quarantine_s` | `0`、`1.0` | 只在 `MATCHED` 有意義，`_s` 範圍 `[0, 3600]`。開啟時，比目前 delay **更早**的新估計在 capture-proxy lane 仍在消除期間先被保留 `_s` 秒再採用；消除失效就立刻採用。前向移動與首次取得不受影響。單聲道必須維持 `0`。 |
| 格點 | — | 只接受 16 kHz／256、16 kHz／512、48 kHz／1024；8 kHz 被拒絕。 |

共用 delay 的變更採兩段式接納：第一次取得立即採用；之後的移動必須超過 32 個 sample，且在候選值存活的 3 個 hop 內、
以差距小於 16 個 sample 再被提出一次，才會採用並對四條 lane 做 realign（`AecDelayState.changed` 在那一個 hop 為 1）。

**建立**（`aec_get_mem_requirements()` 取得描述子，`aec_init_ex()` 用它檢查記憶池）：

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

`aec_init(pool, bytes, &cfg)` 與 `aec_init_ex(..., NULL)` 相同，只是不檢查描述子。`aec_create()` 也接受 `n_mics = 4`（heap，一次 16-byte 對齊的配置，回傳實例指標）。
描述子不可存檔或自行拼湊，每次建置重新查。

**每個 hop 的呼叫序列**：

```
mics (交錯 float[hop][4])、ref (float[hop])
  │
  ▼  aec_process_linear(a, mics, ref)       共用 delay → 對齊後 far → 四條 lane 的線性與 lane 後級
  ▼  aec_get_array(a, &av)                  各 lane 的 error_spec／base_spec／formed_hop、aligned_ref、共用 delay 狀態
  ▼  （呼叫端的 beamformer）                 讀 av.lane_error_spec[ch]，輸出權重 W[4][n_freqs] 與波束頻譜 E
  ▼  aec_process_residual(a, &beam)         融合四條 lane（FULL 另算一次 G_res）
  ▼  aec_get_residual(a, &rv)               融合後的 E、base、r2、舒適噪音與旗標（選用）
  ▼  aec_post_reset_serial(a)               與自己存的不同 → 先清自己的 OLA 尾巴
  ▼  aec_apply_gain(a, g_nr, spec)          spec = base_spec · min(G_res, g_nr)（FULL）
  ▼  aec_apply_cng(a, g_nr, spec)           同一個 g_nr；不要舒適噪音就不呼叫
  ▼  呼叫端的反轉換 + overlap-add
```

```c
/* 呼叫端自己的元件（不在函式庫內） */
void app_beamform(const AecArrayView *av, Complex *weights, Complex *beam_error);
void app_reset_ola(void);                  /* 清自己的 OLA 尾巴 */
void app_synthesize(const Complex *spec);  /* 反轉換 + overlap-add */

/* 一個 hop：mics 是交錯的 float[hop][4]，ref 是 float[hop]；g_nr 可為 NULL */
int app_run_hop(Aec *aec, const float *mics, const float *ref,
                const float *g_nr, uint32_t *last_serial)
{
    static Complex weights[AEC_ARRAY_MICS * 129];   /* 聲道優先 [4][n_freqs] */
    static Complex beam_error[129], spec[129];      /* 16 kHz / 256：n_freqs = 129 */
    AecArrayView av;
    AecBeam beam;
    int rc;

    rc = aec_process_linear(aec, mics, ref);        /* 共用 delay + 四條 lane */
    if (rc != AEC_OK) return rc;

    av.struct_size = sizeof av;
    rc = aec_get_array(aec, &av);                   /* 各 lane 的 error_spec */
    if (rc != AEC_OK) return rc;
    app_beamform(&av, weights, beam_error);

    beam.weights = weights;
    beam.beamformed_error = beam_error;             /* 或 NULL：由權重重建 */
    rc = aec_process_residual(aec, &beam);          /* 融合 + 一次 G_res */
    if (rc != AEC_OK) return rc;

    if (aec_post_reset_serial(aec) != *last_serial) {
        app_reset_ola();                            /* 在本 hop 的 OLA push 之前 */
        *last_serial = aec_post_reset_serial(aec);
    }
    rc = aec_apply_gain(aec, g_nr, spec);           /* spec = base * min(G_res, g_nr) */
    if (rc != AEC_OK) return rc;
    rc = aec_apply_cng(aec, g_nr, spec);            /* 同一個 g_nr */
    if (rc != AEC_OK) return rc;
    app_synthesize(spec);
    return AEC_OK;
}
```

權重與波束頻譜的契約：

- `AecBeam.weights` 是聲道優先的 `Complex[n_mics][n_freqs]`，融合為 `Σ_ch w_ch · lane_ch`，**不取共軛**。必須全部有限且不全為零，否則 `aec_process_residual()` 回 `AEC_ERR_ARG`（該 hop 保持開啟，可用正確的權重再呼叫一次）。
- 某條 lane 的權重在所有 bin 都是 0 時視為「無貢獻」：它不參與 `max_dt`、`max_saturation`、`all_converged`、`usable_linear` 與 `guard_lanes` 的彙整（例如旁路 beamformer 的 `[1,0,0,0]`）。
- `AecBeam.beamformed_error` 是**同一次** beamformer 呼叫算出的頻譜，由呼叫端持有、必須活到本 hop 的檢視讀完為止；`aec_get_residual().error_spec`（以及沒有 lane 守衛觸發時的 `base_spec`）會直接指向它。傳 `NULL` 則由權重重建。含非有限值（NaN／Inf）時 `aec_process_residual()` 與權重非有限一樣先檢查、回 `AEC_ERR_ARG`：實例完全不動、不 reset，該 hop 保持開啟，上一個 hop 的檢視仍可讀。
- 對這份呼叫端自己的緩衝就地呼叫 `aec_apply_gain(a, g, 同一個緩衝)` 是允許的（結果與另給輸出緩衝逐位元相同）；之後 `error_spec`／`base_spec` 讀到的就是已乘增益的資料，所以同一 hop 不要就地套兩次。輸出不可與 `extra_gain` 或函式庫內部緩衝重疊。

**各 `residual` 模式的行為**（陣列實例）：

| 模式 | `aec_process_residual()` | `aec_get_residual()`／增益 | 說明 |
|---|---|---|---|
| `STATE` | `AEC_ERR_MODE` | `aec_get_residual()`、`aec_apply_gain()`、`aec_apply_cng()` 回 `AEC_ERR_MODE` | 只有四條 lane，沒有融合：波束後的頻譜由呼叫端用 `aec_get_array()` 的 `lane_error_spec`／`lane_base_spec` 自行處理與渲染。池最小。 |
| `ESTIMATE` | `AEC_OK`；同一個 hop 可再呼叫（無狀態的重新融合，例如換一組權重） | `r2`、`comfort_noise` 有值，`res_gain` 為 `NULL`；`aec_apply_*` 回 `AEC_ERR_MODE` | 融合但不算增益（增益由呼叫端別處算）。 |
| `FULL` | `AEC_OK`；**同一個 hop 第二次呼叫回 `AEC_ERR_SEQUENCE`**（增益階段每 hop 只前進一次） | 全部有值；`aec_apply_gain()`／`aec_apply_cng()` 可用 | 融合後的頻譜算**一次** `G_res`。 |

順序與錯誤規則：

- `aec_process_residual()` 之前沒有同一 hop 的 `aec_process_linear()`，或 `aec_get_array()` 在第一個 `aec_process_linear()` 之前：`AEC_ERR_SEQUENCE`。
- `aec_process_linear()` 一呼叫，上一個 hop 的融合檢視就失效：在新 hop 的 `aec_process_residual()` 之前，`aec_get_residual()`、`aec_apply_gain()`、`aec_apply_cng()` 回 `AEC_ERR_SEQUENCE`。在後級之前再呼叫一次 `aec_process_linear()` 會取代尚未完成的那個 hop。
- 輸入含非有限值：`aec_process_linear()` 回 `AEC_ERR_ARG`，被拒絕的呼叫不改變任何狀態：上一個 hop 的檢視仍可讀，待處理的 linear 幀保留，逐階段耗時記錄不動。
- 某條 lane 或融合在引數檢查之後拒絕了該 hop（內部一致性檢查失敗，或有限的極大權重使投影溢位）：`AEC_ERR_DSP`（-4），**整個實例已像 `aec_reset()` 一樣重設**，serial 前進。
- `aec_set_level()` 重新設定融合後的增益階段（`FULL`）；lane 自己的 level 沒有作用。`STATE`／`ESTIMATE` 沒有增益階段，呼叫回 0 但依構造無效。

**陣列實例上被拒絕的呼叫**：

| 呼叫 | 結果 |
|---|---|
| `aec_process()` | `AEC_ERR_MODE`（陣列要四路麥克風：用 `aec_process_linear()`） |
| `aec_process_linear_shared_far_from()` | `AEC_ERR_MODE`（四條 lane 的遠端 FFT 與只依賴遠端的逐 hop 分析已由實例自己共用：lane 0 算、lane 1–3 借用；輸出與全部自算位元相同）；`far_owner` 本身是陣列實例時退回本地計算 |
| `aec_apply_external_realign()` | `-1`（共用對齊改變時實例自己對四條 lane 做 realign） |

**檢視**：

- `aec_get_residual()`：`formed_hop` 與 `echo_spec` 為 **`NULL`**（每條 lane 的 formed hop 在 `AecArrayView.lane_formed_hop`）。`error_spec` 是波束後的 E（`beamformed_error` 或由權重重建），`near_spec` 是 `Σ w · lane near`，`far_spec` 是 lane 0 的遠端頻譜，`base_spec` 在沒有貢獻 lane 的守衛觸發時是 E、否則是 `Σ w · lane base`；`r2` 是各 lane 殘留回音向量以權重**相干**相加後的功率，`comfort_noise` 是 `Σ |w|² · lane comfort noise` 的功率和；`guard_lanes` 的第 ch 位元是貢獻 lane ch 的守衛觸發，`guard_active = (guard_lanes != 0)`；`post_reset_serial` 是四條 lane 的 reset 計數之和。
- `aec_get_array()`：`AecArrayView` 的 `lane_error_spec[4]`（分析頻譜，beamformer 的輸入）、`lane_base_spec[4]`（渲染底座）、`lane_formed_hop[4]`、`lane_guard_active[4]`、`aligned_ref`（餵給每條 lane 的 far）與 `delay`（`AecDelayState`：`delay_samples`、`confidence`、`solid`、`changed`、`estimator_calls`、`estimator_updates`）。單聲道實例也可呼叫，回 lane 0 一路（後級跑完之後）。
- `aec_get_linear()`：回 lane 0 的線性狀態，對齊欄位（`aligned_far_hop`、`delay_samples`、`delay_confidence`、`delay_state`、`generation`）換成共用的那一份。
- 所有指標有效到下一次處理或 reset 呼叫。`aec_reset()` 不清 lane 的緩衝：reset 前取得的指標仍可讀（過期的值，不是錯誤），但已不再對應任何 hop，要重新取檢視。

**reset serial 與 OLA**：陣列的 `aec_post_reset_serial()` 是**四條 lane 計數之和**（模 2³²），所以 `aec_reset()` 讓它前進 **4**（單聲道是 1），處理中某條 lane 的硬 reset 也使它移動。只比較「是否與自己存的不同」，不要假設步長。函式庫自己持有的舒適噪音產生器在每次後級呼叫時依 serial 自動重設為 `AEC_CNG_SEED`；呼叫端仍要在本 hop 的 OLA push 之前清自己的 OLA 尾巴（§8.3）。

**診斷**（唯讀，自建立或 `aec_reset()` 以來累計；單聲道實例回中性值）：`aec_far_fft_real_compute_count()`（四條 lane 的總和，正常每 hop +1）、`aec_array_realign_warm_lane_count()`／`aec_array_realign_soft_lane_count()`、`aec_array_matched_filter_count()`、`aec_array_linear_aec_count()`，以及 `-DAEC_STAGE_TIMING=1` 時的 `aec_array_get_timing()`（`delay_us`、`frontend_us`、`linear_us`、`lane_res_us`、`fuse_us`）。

**目前尚未做到的**：

- **遠端側狀態仍是每條 lane 各一份，沒有共用。** 目前共用的只有 delay 估計器、對齊後 far，以及 hop 內由 lane 0 借給 lane 1–3 的遠端 FFT 結果。
- 陣列實例與先前由 Audio_ALG 組成的 4 聲道流程的逐位元一致，只在 Apple arm64／Apple clang 17／NE10 上對照過（2026-11-13 確認後，一次性的對照檔已除役、不再隨附）；其他主機與板端未驗證。

測試：`make test-aec-array`（陣列實例逐 hop 跑 `STATE`／`ESTIMATE`／`FULL` 與契約檢查）、`make test-aec-array-2` 與 `make test-aec-array-flow`（`AEC_TESTING` 建置的狀態機檢查）。這幾支測試各自產生一串雜湊行；要在下一次改動前後比對，先用 `AEC_TEST_BASELINE_WRITE=<目錄>` 從完成版取基準，改動後以 `AEC_TEST_BASELINE=<目錄>` 比對（用法見 `c_impl/README.md`）。

### 8.9 外部對齊的重新對齊（`aec_apply_external_realign`）

只用於 `delay_mode = AEC_DELAY_EXTERNAL_ALIGNED` 的單聲道實例：呼叫端自己把 `ref` 對齊後餵入，對齊量改變（例如重新量測到 far 多延遲 12 個樣本）時，在 hop 之間告訴函式庫，不必 `aec_reset()`：

```c
int delta = new_alignment_samples - old_alignment_samples;   /* 新減舊，單位：樣本 */
int rc = aec_apply_external_realign(aec, delta);       /* aec 是 Aec* */
if (rc < 0) {
    /* NULL、delay_mode 不是 EXTERNAL_ALIGNED，或這是陣列實例（它自己對四條 lane 做 realign） */
} else if (rc == 1) {
    /* warm transfer：delay_acquire_warm_transfer = 1、filter 正在消除回音且位移落在 tap 範圍內，學到的 IR 被平移 */
} else {
    /* rc == 0：只重置 filter（delta == 0 時什麼都不做）；輸出維持連續 */
}
```

規則：

- 與 `aec_process*()` 一樣不是 thread-safe，必須在 hop 之間呼叫；
- `delta != 0` 時一定推進 `AecLinearView.generation`（§8.6），依賴舊對齊的快取要清掉；`delta == 0` 回 0、什麼都不動；
- `delta` 為負時，最多約 `ceil((|delta| + tap 偏移) / hop) + 1` 個 hop 會露出回音；
- 是否已在消除可先查 `aec_linear_is_cancelling()`（§8.6），但函式本身已依此在 warm transfer 與重置之間選擇，不必由呼叫端判斷。

---

## 9. 常見問題排除

先確認基本面：`ref` 真的是送去喇叭的訊號嗎？mic/ref 取樣率相同且同步嗎？
每次呼叫真的餵了剛好 `aec_hop_size()` 個 sample 嗎？這三項有任何一項不成立，
調參數都沒有意義。

### 9.1 回音殘留太多

依序處理：

1. **先分離問題。** 設 `cfg.residual = AEC_RESIDUAL_OFF` 跑一次，聽線性殘差。
   - 回音仍然很大 → 問題在**線性側**：`ref` 送錯、延遲超出 [§3](../c_user_manual_zh_TW.md#3-資料契約) 的可擷取上限、
     回音路徑比濾波器長度（52 ms；48 kHz 為 64 ms）長，或兩路取樣率不同。往下走第 2、3 點。
   - 線性殘差已經很乾淨 → 問題在**抑制強度**，跳到第 4 點。
2. **檢查延遲。** 用 `aec_debug_status()` 讀 `delay_samples`：`-1` 代表還沒擷取到延遲。
   若始終是 `-1`，確認遠端有足夠能量，並確認實際延遲落在 [§3](../c_user_manual_zh_TW.md#3-資料契約) 表列的可擷取上限內。
3. **回音路徑比濾波器長。** 濾波器長度由取樣率決定，不是設定值；回音路徑明顯超過它（大房間、長殘響）時，
   在上游縮短路徑或先做 bulk delay 補償（`FIXED` / `EXTERNAL_ALIGNED`），不能在這個函式庫內加長。
4. **加強抑制。** 改用 `AEC_LEVEL_AGGRESSIVE`（`aec_set_level()` 可在執行期切換）。代價是 double-talk 時近端更容易被壓。

### 9.2 Double-talk 時近端被壓掉

| 動作 | 方向 |
|---|---|
| 換較溫和的 level | `AEC_LEVEL_BALANCED` → `AEC_LEVEL_MILD`（level 是唯一的強度設定；floor 值與近端保護門檻是內部常數，不可單獨調整） |

**不要**用 `AEC_RESIDUAL_OFF`（關閉後級）來解決這個問題——那會讓回音殘留大幅上升。

### 9.3 啟動時收斂太慢／前一秒回音明顯

| 動作 | 方向 |
|---|---|
| 啟動期長度、首次延遲擷取時把已學到的模型平移過去（而不是歸零）的 warm transfer 等收斂相關常數 | 內部常數，已調好且不可設定；「開場一秒有回音」的常見成因是延遲還沒擷取到（`aec_debug_status()` 的 `delay_samples` 為 `-1`）。 |
| 應用層面 | filter 收斂需要一段有意義的遠端能量。可在暖機期間先靜音輸出或播提示音。 |

`aec_debug_status()` 的 `erle_windowed_db` 與 `filter_converged` 可用來觀察收斂進度。

### 9.4 裝置移動時回音突然變大

回音路徑改變後需要重新收斂，短暫洩漏屬正常，通常會自動恢復。若恢復太慢或太頻繁：

| 動作 | 方向 |
|---|---|
| 回音路徑變化偵測、加速恢復期、shadow filter 與延遲變動偵測的常數 | 內部常數，不可設定；恢復時間由它們與 hop 決定（換算後的時長見 [§6.5](configuration.md#65-以-10-ms-為單位撰寫的時間常數內部唯讀參考)） |
| 延遲真的在變（藍牙、HAL 重取樣） | 改用 `EXTERNAL_ALIGNED`，由上游提供對齊後的 far，並在對齊改變時呼叫 `aec_apply_external_realign()` |

### 9.5 其他

| 現象 | 先檢查 | 處理方向 |
|---|---|---|
| 長時間後回音逐漸變差 | mic/ref clock drift | 改用共同時鐘，或在上游做 drift 補償。函式庫不處理。 |
| 批次處理第二個檔案結果異常 | 沿用了前一個檔案的狀態 | 每個獨立串流前呼叫 `aec_reset()` |
| 輸出有突波／發散 | 削波、`ref` 送錯、延遲劇烈變化 | 檢查輸入是否已削波、`ref` 是否送對；削波偵測與 shadow filter 一律開啟，沒有可關的開關 |
| 自己渲染後級頻譜的舊程式，升級 SDK 後守衛 hop 偏大聲或 `aec_reset()` 後有舊尾巴 | 還在渲染 `error_spec`、沒處理 `aec_post_reset_serial()`、視窗或 CNG 種子與函式庫不同 | 照 [`sdk_migration_output_guard.md`](../sdk_migration_output_guard.md) 的清單遷移，並跑其中的逐位元驗收 |
| 噪音底聽起來不自然、會抽動 | CNG 疊了兩層 | 下游若已有噪音處理，設 `enable_cng = 0` |
| 建立實例一直失敗 | 設定被驗證拒絕 | 對照 [§6](configuration.md#6-aecconfig-完整參考) 的允許範圍；特別注意 `sample_rate`/`fft_size` 必須成對合法，以及 `residual` 必須在 `AEC_RESIDUAL_OFF`…`AEC_RESIDUAL_FULL` 範圍內；陣列實例（`n_mics = 4`）另外要求 `residual` 不是 `OFF`、格點是 16k/256、16k/512、48k/1024 之一、`capture_proxy_channel` 在 0–3（[§8.8](#88-麥克風陣列實例n_mics--4)） |
| `SIMD=0` 與 `SIMD=1` 的輸出不同 | NEON kernel 與純量 kernel 的加總順序、融合乘加不同 | 這是預期行為。同一組建置旗標下 heap 與記憶池路徑才保證逐位元相同。 |

---
