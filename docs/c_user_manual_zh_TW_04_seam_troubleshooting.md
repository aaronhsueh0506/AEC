> [索引](c_user_manual_zh_TW.md) · 上一部分：[c_user_manual_zh_TW_03_level.md](c_user_manual_zh_TW_03_level.md) · 下一部分：[c_user_manual_zh_TW_05_appendix.md](c_user_manual_zh_TW_05_appendix.md) <!-- split-nav -->
## 8. 進階整合 seam

一般使用者不需要這一節。只有在你要自己接上外部的噪音抑制／後處理，或是要驅動多個
實例時才會用到。

### 8.1 Context-only 處理路徑

如果你**完全不使用** `aec_process()` 產出的時域音訊，只想讀頻域中間結果自己做後處理，
用 `aec_process_context()`：它跑完整條線性 filter 與後級運算並填好 `AecResContext`，
但不做輸出限幅、也不寫 `out` buffer（沒有 `out` 參數）。

```c
AecConfig cfg;
aec_config_from_level(&cfg, AEC_LEVEL_BALANCED, 16000);
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

> **這個模式下 formed 線性 seam 可能改送擷取訊號。**
>
> `enable_res = 0` 時，後級最後那個「套用抑制並寫出時域訊號」的階段不會執行，它內建的
> 過輸出保護也就跟著不執行——那道保護守的是 `aec_process()` 真正輸出的音訊，守不到只給
> 外部後處理讀的 seam。所以在 context-only 模式下改由 formed 線性輸出的選擇器承擔：當
> filtering-quality 分析器**尚未**判定線性估計可用、且被選中的殘差在這個 hop 的能量
> **高於**擷取訊號時，選擇器會把擷取訊號本身當成 refined／coarse 之外的**第三個候選**
> 送出。殘差比被減數還大，代表線性估計不是在消除回音，而是在加訊號。
>
> 對讀 `AecResContext` 的人的影響：這些 hop 上 `error_spec` / `formed_hop` 是擷取訊號而
> 不是線性殘差，`echo_spec` 隨之趨近 0（`near_spec = error_spec + echo_spec` 的恆等式仍
> 然成立；連續落在擷取訊號時才會趨近 0，單一 hop 因為前半個 block 仍是前一 hop 的
> formed 輸出而不會恰為 0）。切換由既有的 30-sample 交叉淡入承接，所以這是**穩態的
> 退回**，不是逐樣本的硬上界——過渡中那 30 個樣本是兩者的混合。
>
> 這條規則由 `output_capture_when_linear_unusable` 控制（預設開啟），且**只**在
> context-only 生效。`enable_res = 1` 的產品路徑不受影響：那裡仍由後級輸出階段自己的
> 保護負責，輸出逐位元不變；兩個保護同時作用會讓後者的頻域硬切覆蓋掉前者剛啟動的
> 交叉淡入。Audio_ALG 的 4ch mcab flow（`mcab.h`）的四條 lane 全部落在這個模式。

> **三個處理進入點可以自由混用。**
>
> `aec_process()`、`aec_process_context()`、
> `aec_process_context_shared_far_from()`
> 推進完全相同的狀態，輸出複製與遠端分析共用的成本不同。
> 同一個實例上可以任意交錯呼叫，中間不需要
> `aec_reset()`。
>
> 選哪一個純粹是成本考量：下游若不讀 `out`，用 context-only 版本可以省下
> 一次 `hop` 長度的 memcpy。

### 8.2 多實例共用遠端 FFT

多個實例每個 hop 都看到**完全相同**的參考訊號時，可讓其中一個算 FFT、其餘共用：

```c
aec_process_context(&lane[0], mic[0], ref);      /* lane 0 自己算 */
for (int ch = 1; ch < n_lanes; ++ch)
    aec_process_context_shared_far_from(&lane[ch], mic[ch], ref, &lane[0]);
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

### 8.3 `AecResContext` 欄位

呼叫 `aec_get_res_context()` 取得。頻域欄位只有在 `enable_res = 1` **或**
`return_res_context = 1` 時才有效，否則為 `NULL`。

| 欄位 | 說明 |
|---|---|
| `n_freqs` / `hop_size` | 陣列長度。所有頻域陣列長度都是 `n_freqs`。 |
| `error_spec` | formed 線性輸出的頻譜 E(f)，**audio 尺度**（與 [-1,1] 時域一致）。**在 context-only 模式（`enable_res = 0` 且 `return_res_context = 1`）下不保證是線性殘差**：線性估計尚未被判定可用、且選中的殘差能量高於擷取訊號時，送出的是擷取訊號本身（見 §8.1）。 |
| `echo_spec` / `near_spec` | 對應的回音估計與擷取訊號頻譜，滿足 `near_spec = error_spec + echo_spec`。 |
| `res_gain` | 抑制增益，振幅 [0, 1]。**`spatial_linear_context` 為 1 或 2 時為 `NULL`** —— 代表「沒有計算」，若誤讀成全 0 會變成「全部靜音」。**使用前務必檢查非 NULL。** |
| `r2` | 殘留回音功率譜。**int16² 尺度**：要換成 audio 功率尺度須除以 32768²。`spatial_linear_context = 2` 時為 `NULL`。 |
| `comfort_noise` | 舒適噪音功率譜。同樣是 int16² 尺度。`spatial_linear_context = 2` 時為 `NULL`。 |
| `linear_hop` / `formed_hop` | 時域中間結果。 |
| `res_floor_protect` | 本 hop 是否啟用 double-talk RES floor 保護。它就是近端最近有聲音 latch 的實際決定；自己做後級的消費端應讀這個欄位，不要用 `dt_indicator` 另外推導一套規則（Audio_ALG 的 4ch fuse 目前只讓有貢獻的 lane 參與 `max_dt`／`max_sat`／`all_converged`／`usable_linear`（`McabFrame` 欄位）的彙整，不讀本欄位）。 |
| `usable_linear` | 本 hop 的可用線性估計判定。`0` 時 `r2` 是保守的非線性估計（遠端功率乘上預設 echo-path 增益），並非由線性回音估計推導，可能遠高於根本不存在的殘留回音。 |
| `far_power`、`erle_factor`、`dt_indicator`、`divergence`、`saturation_level`、`erl_estimate`、`shadow_dt`、`is_stationary_dt`、`filter_converged`、`filter_once_converged`、`epc_active` | 供外部判斷／診斷用的純量狀態。 |

> ⚠️ 所有 pointer 欄位都直接指向實例內部的 per-hop 緩衝區，**不是複本**。
> 不可修改、不可 free，而且必須在**下一次對同一實例的任何處理呼叫之前**讀完或複製走。
> 三個進入點（`aec_process`、`aec_process_context`、
> `aec_process_context_shared_far_from`）都會就地覆寫這些緩衝區。

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

- 只在 `delay_mode == AEC_DELAY_MATCHED` 時累積（`has_delay == 1` 才會進入
  計數的程式路徑）；`FIXED`／`EXTERNAL_ALIGNED` 從未建過 matched filter，
  兩個計數器永遠是 0——不是「估計器跑了但沒省到」，而是根本沒有機制可省。
- 累積範圍是「自 `aec_create`／`aec_init` 或最近一次 `aec_reset()` 以來」，
  與 `AecDebugStatus` 其餘欄位的累積規則一致；`aec_reset()` 會把兩個計數器
  歸零，因為跨越 reset 的比值會把兩段不同的 duty 狀態混在一起，沒有意義。
  在計數的 call site（消耗 `run_filter` 的那一行）遞增，與實際執行的機制一致。
- **純診斷**：兩個計數器只被寫入、從未被函式庫自己讀回，加起來是熱路徑上
  兩個整數遞增，不影響任何音訊輸出。
- `example/aec_wav.c` 每次執行結尾印一行（[附錄 A「Duty census 診斷輸出」](c_user_manual_zh_TW_05_appendix.md#duty-census-診斷輸出)）。
- 永久回歸測試：`make test-duty-census`（`test/test_duty_census.c`）。


### 8.5 獨立的 RES 級（`aec_res.h`）

多條線性 AEC lane 先融合、再對融合後的頻譜算**一次**殘留回音抑制時使用（四麥克風
線性 AEC + BF 之後的 post-beam RES）。`AecRes` 不屬於任何 `Aec`，也不屬於 Audio_ALG 的 4ch 流程層
`mcab`（`pipelines/4ch_aec_bf_nr_res/mcab.h`）：mcab 是 flow 層而不是 library，四條 lane 與（MATCHED 時）
共用的 `DelayAec` 由應用建立後以指標注入，mcab 只管共用 delay 的對齊狀態、驅動這四條 lane（`mcab_process_front()`）與 `mcab_fuse()`
（fuse 之後 `frame.g_res_ctx` 為 NULL），沒有 `enable_res`／RES 強度欄位，beamformer 也在 mcab 之外。
要 RES 的 4ch 應用在自己的 init 裡 `aec_res_init()`（設定來自 `aec_config_from_level(level, sample_rate)`，
`fft_size` 取 flow 的格點），每 hop 在 fuse 之後用 Audio_ALG 的 `mcab_sample_res_input(&frame, &in)`
（`pipelines/4ch_aec_bf_nr_res/mcab_sample_post.h`）組出 `AecResInput` 再呼叫 `aec_res_process(res, &in)`，並自己持有、設強度（`aec_res_set_level()`）與重設（`aec_res_reset()`）；
範例見 Audio_ALG 的 `pipelines/4ch_aec_bf_nr_res/mcab_static.c`、`mcab_raw.c`、`mcab_external_static.c`
與 `pipelines/dfn/4ch_aec_bf_dfn_res/main.c`。單聲道不用它，RES 就在 `Aec` 裡（`ctx.res_gain`，見 §8.7）。
它把 `Aec` 內部那個抑制器獨立出來，設定來自 lane 自己用的
同一個建構函式，所以同一個 `AecConfig` 建出的 `AecRes` 與 lane 內的抑制器設定逐位元相同。

```c
#include "aec_res.h"

size_t bytes = aec_res_get_mem_size(&cfg);        /* 0 = cfg 無效 */
AecRes* res  = aec_res_init(mem, bytes, &cfg);    /* mem 16-byte 對齊；NULL = 失敗 */

AecResInput in = {0};
in.error_spec = fused_error;   in.near_spec = fused_near;      /* audio 尺度 */
in.r2 = fused_r2;              in.comfort_noise = fused_comfort; /* int16² 尺度 */
in.render_hop = aligned_ref;   /* hop_size 個樣本，audio 尺度 */
in.all_converged = all_ok;     in.max_dt = worst_dt;
in.max_saturation = worst_sat; in.clock_drift = drift;

const float* g_res = aec_res_process(res, &in);   /* n_bins 個增益，[0,1] */
aec_cng_fill(&rng, spec, fused_comfort, g_res, g_nr, 0.31622777f, aec_res_n_bins(res));
```

| 函式 | 說明 |
|---|---|
| `aec_res_get_mem_size(cfg)` | 所需 byte 數；`cfg` 為 NULL 或無效回 0。只有 `sample_rate`、`fft_size`、兩個最小增益地板與 `min_gain_split_floor_enabled` 會影響抑制器（`aec.c` 的 `aec_build_sg_config`）；其餘欄位僅參與驗證。 |
| `aec_res_init(mem, bytes, cfg)` | 在呼叫端的記憶體內建立；不配置記憶體、不用 stdio、不需 destroy。記憶體 NULL／未對齊／過小或 `cfg` 無效回 NULL。 |
| `aec_res_reset(r)` | 清除抑制器全部狀態；**保留**目前的 far-active／dt 地板目標與 `min_gain_split_floor_enabled` 閂（含 `aec_res_set_level` 改過的——這三者都存在 `r->sg_cfg` 鏡像裡，reset 的 `rebuild()` 就是從這個鏡像重新 `suppression_gain_init()`）。 |
| `aec_res_process(r, in)` | 一個 hop。回傳的 `n_bins` 個增益由 `r` 擁有，到下一次對 `r` 的呼叫前有效；`r`／`in` 或任一輸入陣列為 NULL 回 NULL。 |
| `aec_res_set_level(r, level, ramp_ms)` | 與 `aec_set_level()` 同語意：far-active 地板用 `ramp_ms` 走（範圍 `[0, 60000]`），dt 地板與 `min_gain_split_floor_enabled` 閂一律立即套用；含 `no_floor` 的切入/切出。拒絕時（NULL、out-of-enum、`ramp_ms` 超界）什麼都不寫。 |
| `aec_res_split_floor(r, &live, &target)` | 讀 far-active 地板（功率比）：`live` 是現行值，`target` 是正走向的值，後者跨 reset 保留。 |
| `aec_res_n_bins(r)`／`aec_res_hop_size(r)` | 陣列長度（`fft_size/2 + 1`、`fft_size/2`）。 |
| `aec_cng_fill(&rng, spec, cn, g_res, g_nr, nr_floor, n)` | 在 bin `1 … n-2` 加舒適噪音：`sqrt(cn/32768²)·sqrt(1-g_res²)·max(g_nr, nr_floor)` 乘上由 `rng` 遞推選出的隨機相位（`rng = (rng·69069+1) & 0x7FFFFFFF`，取高 5 bit 查 32 點 √2·sin 表，虛部取 +8）。`g_nr` 可為 NULL（視為 1）。 |

`aec_res_process` 做的前處理：近端功率 `|near|²`（`all_converged` 時取 `min(|error|², |near|²)`）乘 32768²；
參考乘 32768；`all_converged` 讓抑制器離開初始狀態；`max_dt > 0.2` 啟用 double-talk 地板；
`max_saturation > 0.5` 視為飽和回音。`r2` 同時當作 bounded 與 unbounded 殘留估計。
`aec_cng_fill` 就是 Audio_ALG 範例（例如 `pipelines/mono_aec_nr_res/main.c` 的 `mono_form()`）每個 hop 呼叫的那個函式，**不同於** `Aec` 內部的批次化填補
（`inv_psd_scale`＋`sk_fast_sqrt_f32`），兩者的輸出不可互相比對。

測試：`make test-aec-res`（設定與 lane 逐位元相等、API 契約、render ×32768 縮放、離開初始狀態、
飽和後復原限制）、`make test-zero-heap`、`make audit-no-stdio`。
完整頁面見 [`docs/html/aec_res.html`](html/aec_res.html)。

### 8.6 線性對齊 context（`aec_get_linear_context`）

給需要「與線性 filter 讀到的**同一段**對齊後參考訊號」的後級（頻域 NN 後濾波、
多 lane 的 RES）使用的唯讀檢視：

```c
AecLinearContext lin;
aec_get_linear_context(&aec, &lin);   /* 在該 hop 的處理呼叫之後、下一次處理呼叫之前讀 */
```

| 欄位 | 說明 |
|---|---|
| `hop_size` | `formed_linear_hop`、`aligned_far_hop` 的長度 |
| `formed_linear_hop` | `enable_res` 或 `return_res_context` 為 1 時是 formed 線性殘差；否則是原始線性輸出 |
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
只有串接順序是整合者的責任。Audio_ALG 的 `pipelines/mono_aec_nr_res/main.c`（每 hop 的增益序列
是該檔的 `mono_form()`）就是用上面的公開介面接起來的範例；每個範例 `main` 自己初始化它用到的函式庫並寫出這段序列。
AEC 以 `enable_res = 0`、`return_res_context = 1` 建立（`spatial_linear_context` 維持 level
的 `0`，所以 `res_gain`、`r2`、`comfort_noise` 都有值），AEC 自己**不**把抑制套到時域，
抑制增益留給整合者與 NR 的增益合併後一次套用；單聲道的 RES 就在 `Aec` 裡面：增益是
`Aec` 自己算的 `ctx.res_gain`（由 `aec_get_res_context()` 取得），整合者不需要另外 `aec_res_init()` 一個 `AecRes`。
每個 hop 的呼叫順序：

```
mic, ref (hop 個 float)
  │
  ▼  aec_process_context(a, mic, ref)          線性 filter + post 區塊；不寫 out
  ▼  aec_get_res_context(a, &ctx)              error_spec / near_spec / far_spec /
  │                                            r2 / comfort_noise / res_gain / 純量判定
  ▼  aec_get_linear_context(a, &lin)           aligned_far_hop、delay_state
  │      （以上指標都借自 AEC 的 per-hop 緩衝，下一次 AEC 處理呼叫前有效）
  ▼  anr_process_gain(nr, ctx.error_spec,      NR 增益：r2 / 32768² 併入噪音底
  │                     r2 / 32768², NULL)     （僅在 ctx.usable_linear 為真時；否則純降噪）
  │                                            ；anr_get_gain() 取得 g_nr
  ▼  g_total[k] = min(g_nr[k], ctx.res_gain[k])
  ▼  spec[k]    = ctx.error_spec[k] · g_total[k]
  ▼  aec_cng_fill(&rng, spec, ctx.comfort_noise, ctx.res_gain, g_nr,
  │               0.31622777f, n_freqs)        只填被 RES 切掉的 bin，量依 g_nr 縮放
  ▼  ac_istft_push(istft, spec, out)           反 FFT + sqrt-Hann 合成 + OLA → hop 個 sample
```

要點：

- **哪一個進入點**：下游不讀時域輸出時用 `aec_process_context()`，省一次 hop 長度的
  memcpy；只做 AEC（不接 NR／RES）的組態才用 `aec_process()` 直接寫 `out`。
- **增益取最小值，不是相乘**：`g_total = min(G_nr, G_res)`。`G_res`（取 min 之前的值）
  同時決定舒適噪音的量，所以填補只反映 AEC 的抑制，NR 增益再以 `-10 dB`
  （`0.31622777f`）為下限縮放填補量。
- **`aec_cng_fill` 的亂數狀態 `rng` 由呼叫端持有**，Audio_ALG 範例碼的種子是 `0x9e3779b9`
  （mono 範例的 `CNG_RNG_SEED`、4ch 的 `MCAB_SAMPLE_CNG_RNG_SEED`）；`aec_reset()` 之外，整合者的 reset 還要同時清 NR 狀態、合成 OLA 與暫存增益，並把 `rng` 設回種子。
- **`aec_set_level()` 對這條鏈有效**：單聲道鏈的 `G_res` 來自 `Aec` 自己的
  post 區塊，所以強度旋鈕直接用 `aec_set_level()`（[§7.1](c_user_manual_zh_TW_03_level.md#71-執行期切換-levelaec_set_level)）；四麥克風線性 AEC + BF 之後的
  `G_res` 來自應用在 4ch mcab flow 之後自己建立的 `AecRes`，才需要對它呼叫 `aec_res_set_level()`（§8.5）。
- 下游若另有噪音抑制或 CNG，不要在 `aec_cng_fill` 之外再疊一層（[§6.2](c_user_manual_zh_TW_02_config.md#62-模組開關全部只接受-0-或-1)）。

測試：單聲道鏈的行為由 `Audio_ALG` 的 `pipelines/mono_aec_nr_res` 測試涵蓋；本函式庫端的
對應契約是 `make test-aec-res`、`make test-linear-context` 與 `make test-process-context`。
元件頁見 [`docs/html/aec.html`](html/aec.html)（細節頁 [`docs/html/aec_detail.html`](html/aec_detail.html)）。

---

## 9. 常見問題排除

先確認基本面：`ref` 真的是送去喇叭的訊號嗎？mic/ref 取樣率相同且同步嗎？
每次呼叫真的餵了剛好 `aec_hop_size()` 個 sample 嗎？這三項有任何一項不成立，
調參數都沒有意義。

### 9.1 回音殘留太多

依序處理：

1. **先分離問題。** 設 `cfg.enable_res = 0` 跑一次，聽線性殘差。
   - 回音仍然很大 → 問題在**線性側**：`ref` 送錯、延遲超出 [§3](c_user_manual_zh_TW.md#3-資料契約) 的可擷取上限、
     回音路徑比 `filter_length` 長，或兩路取樣率不同。往下走第 2、3 點。
   - 線性殘差已經很乾淨 → 問題在**抑制強度**，跳到第 4 點。
2. **檢查延遲。** 用 `aec_debug_status()` 讀 `delay_samples`：`-1` 代表還沒擷取到延遲。
   若始終是 `-1`，確認遠端有足夠能量，並確認實際延遲落在 [§3](c_user_manual_zh_TW.md#3-資料契約) 表列的可擷取上限內。
3. **加長 filter。** 大房間或長殘響時調高 `filter_length`（單位是 **sample**）。
   必須 destroy 後重新 create，並重新評估記憶體與運算量。
4. **加強抑制。** 改用 `AEC_LEVEL_AGGRESSIVE`；或直接把
   `min_gain_floor_far_active_db` **調低**。代價是 double-talk 時近端更容易被壓。

### 9.2 Double-talk 時近端被壓掉

| 動作 | 方向 |
|---|---|
| 換較溫和的 level | `AEC_LEVEL_BALANCED` → `AEC_LEVEL_MILD` |
| 或直接調 `min_gain_floor_far_active_db` | **調高**（例如 -28 → -20） |
| 放寬 double-talk 專用下限 `min_gain_floor_dt_db` | **調高**（例如 -20 → -16），需 `dt_aware_res_floor_enabled = 1` |
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
| 延遲變動偵測變慢（`delay_est_period_s` > 0 時） | **調低**，或維持預設 `0.0`（每個 hop 都分析） |
| 加長 `filter_length` | **調高**，讓模型涵蓋更多可能的路徑 |

### 9.5 其他

| 現象 | 先檢查 | 處理方向 |
|---|---|---|
| 長時間後回音逐漸變差 | mic/ref clock drift | 改用共同時鐘，或在上游做 drift 補償。函式庫不處理。 |
| 批次處理第二個檔案結果異常 | 沿用了前一個檔案的狀態 | 每個獨立串流前呼叫 `aec_reset()` |
| 輸出有突波／發散 | 削波、`ref` 送錯、延遲劇烈變化 | 維持 `enable_saturation`、`enable_shadow` 開啟；檢查輸入是否已削波 |
| 噪音底聽起來不自然、會抽動 | CNG 疊了兩層 | 下游若已有噪音處理，設 `enable_cng = 0` |
| 建立實例一直失敗 | 設定被驗證拒絕 | 對照 [§6](c_user_manual_zh_TW_02_config.md#6-aecconfig-完整參考) 的允許範圍；特別注意 `sample_rate`/`fft_size` 必須成對合法，以及 `spatial_linear_context` 的組合限制 |
| `SIMD=0` 與 `SIMD=1` 的輸出不同 | NEON kernel 與純量 kernel 的加總順序、融合乘加不同 | 這是預期行為。同一組建置旗標下 heap 與記憶池路徑才保證逐位元相同。 |

---

> [索引](c_user_manual_zh_TW.md) · 上一部分：[c_user_manual_zh_TW_03_level.md](c_user_manual_zh_TW_03_level.md) · 下一部分：[c_user_manual_zh_TW_05_appendix.md](c_user_manual_zh_TW_05_appendix.md) <!-- split-nav -->
