# SDK 遷移清單：輸出守衛（output guard）與後級 API

給**自己用後級頻譜（`error_spec` 等）做渲染（合成時域輸出）**的既有 SDK 應用，以及用過舊版
context API 的應用。只呼叫 `aec_process()`（`AecConfig.residual` 維持預設 `AEC_RESIDUAL_FULL`）並直接取
`out` 的應用，輸出位元不變，不必看本文。

> **本版是不相容的 API 變更（輸出位元不變）。** 舊版的 `aec_process_context()`、`aec_get_res_context()`、
> `aec_get_output_base()`、`aec_get_linear_context()`、`AecResContext`、`AecOutputBase`、`AecLinearContext`，
> 以及 `AecConfig` 的 `enable_res`／`return_res_context`／`spatial_linear_context` 三個旗標與 `AEC_CONTEXT_*` 常數都已移除，
> 改為 `AecConfig.residual` 與 `aec_process_linear()`／`aec_process_residual()`／`aec_get_linear()`／`aec_get_residual()`。
> 本文出現的上述**舊名**都只在對照表與「舊寫法」中，標明為舊名。同一輪還有建立、設定與標頭配置的不相容變更（`Aec` 不透明、
> `aec_create()` 回傳指標、`AecConfig` 只剩 12 個欄位等），舊名與新寫法在 §3.3。

相關文件：`reference/integration.md` §8.1（進入點與分析 seam／渲染底座）、§8.3
（`aec_get_residual()`／`aec_apply_gain()`／`aec_post_reset_serial()` 的完整語意）、§8.7（單聲道整鏈）；
公開宣告在 `c_impl/include/aec.h`（`AEC_OUTPUT_GUARD_API_VERSION`、`aec_get_residual`、`aec_apply_gain`、
`aec_apply_cng`、`aec_apply_cng_spec`、`aec_post_reset_serial`、`aec_synthesis_window`）。

## 1. 先講結論：舊程式不會報錯，只會悄悄失去守衛

輸出守衛的規則：本 hop 線性估計不可用，且 `sum|error_spec|²` 大於 `sum|error_spec + echo_spec|²`
（殘差比擷取訊號還大）時，渲染的底座改用擷取訊號 `error_spec + echo_spec`，而不是 `error_spec`。
`aec_process()` 內部一直是這樣做；後級檢視也提供同一個底座（`aec_get_residual().base_spec`）。

如果你的程式還是把 **`error_spec × 增益`** 送去反轉換：

- 這是**自己渲染頻譜**的程式碼，編譯過、執行不報錯、也沒有任何回傳值或旗標會告訴你；
- 但守衛 hop 上你的輸出比 `aec_process()` 的輸出大聲（殘差比擷取還大）；
- 另外你的合成視窗、舒適噪音種子、delay 硬重置時的 OLA／CNG 處理也可能和函式庫內部不同。

新增的 getter 無法偵測「呼叫端還在用 `error_spec` 渲染」，所以這件事只能靠**你主動遷移**。
`AEC_OUTPUT_GUARD_API_VERSION` 是給你放進自己建置的編譯期標記（見 §2），它不會強迫沒加檢查的舊程式中斷。
`AecResidualView.error_spec` 的名稱與語意沿用舊版（它仍是分析用的真實線性殘差），所以只改渲染的程式不會因改名而編譯失敗；
但使用舊 context 型別與函式的程式會因這些名稱已移除而編譯失敗（這一項是明確的，照 §3 改寫即可）。

不在守衛範圍內：時域 seam（`AecResidualView.formed_hop`、`AecLinearView.linear_hop`）永遠是分析殘差，
沒有時域渲染底座。直接播放或重新合成這兩個串流的應用拿不到輸出守衛，需要守衛就必須改走頻域路徑。

## 2. 版本旗標

`aec.h` 定義 `#define AEC_OUTPUT_GUARD_API_VERSION 2`，代表下列同時存在：
`aec_get_residual().base_spec`（舊版是 `aec_get_output_base()`）、`aec_apply_gain()`／`aec_apply_cng()`、
`aec_post_reset_serial()`、`aec_synthesis_window()`、`aec_apply_cng_spec()` 與 `AEC_CNG_SEED`
（都在 `aec.h`）。版本 1 是舊的 `aec_get_output_base()` 形式。
把下面幾行放進你渲染後級頻譜的那個檔案；SDK 比這一版舊時會在編譯期停下來，而不是靜默通過：

```c
#include "aec.h"
#include "audio_stft.h"   /* AcIstft */

#if !defined(AEC_OUTPUT_GUARD_API_VERSION) || AEC_OUTPUT_GUARD_API_VERSION < 2
#error "aec.h is older than the output-guard API 2: update the SDK before rendering residual spectra"
#endif
```

之後輸出守衛相關 API 若再有不相容的變動，這個數字會加一；你的檢查寫成 `== 目前驗過的版本`
還是 `>=`，由你決定要不要在版本升級時被迫重新驗證。

## 3. 遷移步驟總表

### 3.1 設定與進入點（舊名 → 新名）

| 舊名（已移除） | 新寫法 |
|---|---|
| `enable_res = 0`、`return_res_context = 0` | `cfg.residual = AEC_RESIDUAL_OFF`，用 `aec_process()`（`out` 是線性殘差）或 `aec_process_linear()` |
| `spatial_linear_context = AEC_CONTEXT_LINEAR_ONLY`（需 `enable_res = 0`、`return_res_context = 1`） | `AEC_RESIDUAL_STATE` |
| `spatial_linear_context = AEC_CONTEXT_NO_GAIN`（同上） | `AEC_RESIDUAL_ESTIMATE` |
| `enable_res = 0`、`return_res_context = 1`、`spatial_linear_context = 0`，用 `aec_process_context()` | `AEC_RESIDUAL_FULL`，用 `aec_process_linear()` + `aec_process_residual()` |
| `enable_res = 1`，用 `aec_process()` | `AEC_RESIDUAL_FULL`（預設），用 `aec_process()` |
| `aec_process_context_shared_far_from()` | `aec_process_linear_shared_far_from()`（`aec_internal.h`），之後 `aec_process_residual()` |
| `aec_get_res_context()` → `AecResContext` | `aec_get_residual()` → `AecResidualView`（`struct_size` 要先設） |
| `aec_get_output_base()` → `AecOutputBase`（`base_spec`、`guard_active`） | 同一個 `AecResidualView` 的 `base_spec`、`guard_active` |
| `aec_get_linear_context()` → `AecLinearContext` | `aec_get_linear()` → `AecLinearView`（`formed_linear_hop` 改讀 `aec_get_residual().formed_hop`；未經 FORM 的線性輸出是 `AecLinearView.linear_hop`） |
| `AecResContext` 的純量狀態（`far_power`、`erle_factor`、`dt_indicator`、`divergence`、`saturation_level`、`erl_estimate`、`shadow_dt`、`is_stationary_dt`、`filter_converged`、`filter_once_converged`、`epc_active`） | `AecLinearView` 的同名欄位；`res_floor_protect`、`usable_linear` 在 `AecResidualView` |
| 舊的組合限制（`spatial_linear_context` 必須搭配 `enable_res = 0`、`return_res_context = 1`） | 取消；只剩 `residual` 必須在 `OFF`…`FULL` 範圍內 |
| 回傳值：`aec_process()` 回 `void`，getter 回 `0`／`-1` | 回 `AecStatus`（`AEC_OK`＝0，錯誤為負值：`AEC_ERR_ARG`、`AEC_ERR_MODE`、`AEC_ERR_SEQUENCE`、`AEC_ERR_DSP`） |

### 3.3 設定、建立與生命週期（同一輪的其他不相容變更）

下表的舊名都已移除，名稱取自上一版公開標頭（`c_impl/include/aec.h`、`aec_res.h`）。輸出位元不變；改的是呼叫端程式碼要怎麼寫。
各項的現行語意見 `reference/lifecycle.md`、`reference/configuration.md` 與 `reference/levels.md`。

| 舊名（已移除） | 新寫法 |
|---|---|
| `Aec x; aec_create(&x, &cfg);`、`calloc(1, sizeof(Aec))`、`sizeof(Aec)`、`x.field`／`x->field` | `Aec` 不透明：`Aec* a = aec_create(&cfg);`（`aec.h` 只剩 `typedef struct Aec Aec;`）。呼叫端不能以值持有它，也不能取大小或讀欄位 |
| `aec_destroy(&x);` 再 `free(...)`（heap 實例） | `aec_destroy(a);` 剛好一次，它釋放那一整塊；之後指標失效，**不可再 `free`、再 destroy 或再讀**。`aec_init()` 的 pool 實例不變：`aec_destroy()` 不釋放記憶體，pool 仍由呼叫端回收 |
| `aec_config_defaults(&cfg, sr)` 與手動填 `cfg.filter_length`、`cfg.mu`、`cfg.enable_shadow`、`cfg.enable_highpass`、`cfg.max_delay_ms`、`cfg.delay_est_period_s` 等 | `aec_config_from_level(&cfg, level, sr)`。`AecConfig` 只剩 12 個欄位（`aec.h`），其餘是函式庫內部由 `(sample_rate, fft_size, n_mics, level)` 推導的調校值，呼叫端不能設定 |
| `cfg.min_gain_floor_far_active_db`、`cfg.min_gain_floor_dt_db`、`cfg.min_gain_split_floor_enabled` | `cfg.level`（`AecLevel`，0–3）；三個值由 level 查表帶出 |
| `AEC_PRESET_MILD`／`_BALANCED`／`_AGGRESSIVE`／`_NO_FLOOR`、`AecPreset` | `AEC_LEVEL_*`、`AecLevel`（值相同：0、1、2、3）。名稱解析（字串轉 level）不在函式庫內；CLI 的 `--level` 只收 0–3 的數字，其他值以結束碼 2 拒絕 |
| `aec_set_level(a, level, ramp_ms)` | `aec_set_level(a, level)`：far-active 地板的 ramp 固定 100 ms |
| `DA_NUM_FILTERS`、`DA_FILTER_SIZE`、`DA_FILTER_INTRA_SHIFT`、`DA_DOWN_SAMPLING_FACTOR` | `AEC_DELAY_NUM_FILTERS_MAX`、`AEC_DELAY_FILTER_SIZE`、`AEC_DELAY_FILTER_INTRA_SHIFT`、`AEC_DELAY_DOWN_SAMPLING_FACTOR`（`aec.h`，無別名） |
| `aec_res.h`：`AecRes`、`AecResInput`、`aec_res_init()`、`aec_res_process()`、`aec_res_set_level()` | 增益階段不再是公開介面。單聲道的 `G_res` 在 `Aec` 裡（`aec_get_residual().res_gain`）；多路融合用陣列實例（`AecConfig.n_mics = 4`，`aec_process_residual(a, &beam)`），強度用 `aec_set_level()` |
| Audio_ALG 的 `mcab_*` 四聲道流程 | `AecConfig.n_mics = AEC_ARRAY_MICS`（4）的陣列實例：`aec_process_linear()` → `aec_get_array()` → 呼叫端的 beamformer → `aec_process_residual()`；見 `reference/integration.md` §8.8 |
| 由 `include/` 取用內部標頭（`aec_post.h`、`pbfdkf.h` 等）裡的常數與型別 | `AEC_CNG_SEED`、`AEC_PSD_SCALE_F`／`AEC_PSD_SCALE_INV_F`、`AecStageTiming`、`aec_level_name()` 在 `aec.h`；`include/` 只剩 `aec.h`、`aec_internal.h`、`aec_balanced_config.h`、`aec_debug.h`，其餘內部標頭在 `src/` |
| `aec_debug_set_level()`、`aec_debug_set_log()`、`aec_wav` 的 `--debug-level`、`--debug-log`、`--no-shadow`、`--no-hpf`、`--delay-est-period` | 已刪除（前兩者只儲存、從不讀取；後三個對應的欄位已是內部值）。`aec_debug_set_trace()` 與 `--debug-trace` 仍在 |
| 合成視窗 `aec_synthesis_window()` 指向靜態表 | 指向實例自己的 pool（init／create 時建好）：借用它的 `AcIstft` 不得比 `Aec` 活得久（§6） |
| 存檔的 `AecMemReq` 描述子（`AEC_LAYOUT_VERSION` 1） | `AEC_LAYOUT_VERSION` 2：舊描述子被 `aec_init_ex()` 拒絕；每次建置重新查詢（`aec_init()`／`aec_create()` 不比對描述子） |

位元結果：同一份輸入、同樣的組態，新舊對應的模式輸出逐位元相同（函式庫端曾以 `make test-split-entry`、`make test-split-entry-2` 與 `make test-split-entry-3` 對舊 API 產生的
雜湊逐行驗證；這些一次性的對照檔已於 2026-11-13 確認一致後除役，之後的改動由新版本取基準，見 `c_impl/README.md` 的 “Line-stream baselines”）。

### 3.2 自己渲染的應用要做的事

| # | 要做的事 | 舊寫法 | 新寫法 |
|---|---|---|---|
| 1 | 編譯期檢查 | 無（或版本 1） | §2 的 `#if`／`#error` |
| 2 | 渲染底座 | `ctx.error_spec[k] × g[k]` | `aec_apply_gain(aec, extra_gain, out_spec)`，或自己乘：`rv.base_spec[k] × g[k]`，`rv` 來自 `aec_get_residual()` |
| 3 | reset serial | 無，或自己從 delay 變動推測 | 每 hop 比對 `aec_post_reset_serial()`，變了就清 OLA 尾巴（§5） |
| 4 | 合成視窗 | `AcIstft` 預設 sqrt-Hann | 向 `aec_synthesis_window()` 借表（§6） |
| 5 | 舒適噪音 | 自己的 CNG，自選種子 | `aec_apply_cng()`（實例內產生器，函式庫自己重設），或自己持有狀態的 `aec_apply_cng_spec()`，種子 `AEC_CNG_SEED`（§7） |
| 6 | 驗收 | 與舊輸出比較 | 與同一份輸入下 `aec_process()` 逐位元比較（§9） |

分析用途（`r2`、`error_spec` 給 NR 或外部增益階段的輸入、ERLE 類統計、訓練特徵、需要與 `formed_hop` 配對的重建）
**繼續讀 `error_spec`**，不要改成底座；底座只用來渲染。

## 4. 前後對照程式碼

### 4.1 舊寫法（守衛消失；`AecResContext` 為舊名）

```c
#include "aec.h"
#include "audio_stft.h"

void my_comfort_noise(uint32_t* rng, Complex* spec, const AecResContext* ctx);

/* BEFORE (old API): renders error_spec, default-window AcIstft, own CNG with its own seed, no reset handling. */
static void old_render_hop(AcIstft* istft, uint32_t* my_rng, Complex* spec,
                           const AecResContext* ctx, float* out) {
    int k;
    for (k = 0; k < ctx->n_freqs; ++k) {
        spec[k].r = ctx->error_spec[k].r * ctx->res_gain[k];
        spec[k].i = ctx->error_spec[k].i * ctx->res_gain[k];
    }
    my_comfort_noise(my_rng, spec, ctx);
    ac_istft_push(istft, spec, out);
}
```

### 4.2 新寫法（單聲道與 lane 共用的渲染函式）

單聲道：用預設的 `AEC_RESIDUAL_FULL` 建立 AEC，每 hop `aec_process_linear()` + `aec_process_residual()`，增益就是
`rv.res_gain`，不接 NR 時 `aec_apply_gain(aec, NULL, spec)` 即「底座 × `res_gain`」。
lane（`residual` 為 `STATE` 或 `ESTIMATE`）：`rv.res_gain` 為 `NULL`，增益與 `comfort_noise` 由應用自己的來源提供
（例如外部另算的 `G_res`），把它們傳進同一個函式即可；渲染的頻譜一樣是該 lane 的 `rv.base_spec`。
下面是單一渲染器，單聲道與 lane 各持有一份（每個要渲染的 `Aec` 各有自己的 `AcIstft`、`cng_rng` 與 `last_serial`）。
`renderer_render_hop()` 對呼叫端給的增益做渲染；單聲道且增益就是 `res_gain` 時，`mono_render_hop()` 改走函式庫的
`aec_apply_gain()`／`aec_apply_cng()`（兩者逐位元相同）。

```c
#include <stdlib.h>
#include <string.h>

typedef struct OutputRenderer {
    AcIstft* istft;
    void*    istft_mem;
    Complex* spec;          /* n_freqs scratch */
    uint32_t cng_rng;       /* comfort-noise LCG state, owned by the application (aec_apply_cng_spec form only) */
    uint32_t last_serial;   /* last aec_post_reset_serial() value this renderer has handled */
    int      n_freqs;
} OutputRenderer;

static void renderer_free(OutputRenderer* r) {
    free(r->istft_mem);
    free(r->spec);
    memset(r, 0, sizeof *r);
}

/* Call once after aec_create()/aec_init(), before the first hop. */
static int renderer_init(OutputRenderer* r, const Aec* aec, int fft_size) {
    AcStftConfig sc;
    const float* win = NULL;
    int win_n = 0;
    size_t sz;
    memset(r, 0, sizeof *r);
    if (aec_synthesis_window(aec, &win, &win_n) != 0 || win_n != fft_size) return -1;
    ac_stft_config_default(&sc, fft_size);
    sc.window = win;                               /* borrowed: the table is owned by the Aec */
    sz = ac_istft_get_mem_size(&sc);
    if (sz == 0 || posix_memalign(&r->istft_mem, 16, sz) != 0) return -1;
    r->istft = ac_istft_init(r->istft_mem, sz, &sc);
    r->n_freqs = fft_size / 2 + 1;
    r->spec = (Complex*)calloc((size_t)r->n_freqs, sizeof(Complex));
    if (!r->istft || !r->spec) { renderer_free(r); return -1; }
    r->cng_rng = AEC_CNG_SEED;
    r->last_serial = aec_post_reset_serial(aec);   /* 0 on a fresh instance */
    return 0;
}

/* The reset-serial step shared by both renderers: BEFORE the first CNG draw and BEFORE ac_istft_push of this hop.
 * Never reset the AEC for this. */
static void renderer_sync_serial(OutputRenderer* r, const Aec* aec) {
    uint32_t serial = aec_post_reset_serial(aec);
    if (serial != r->last_serial) {
        ac_istft_reset(r->istft);
        r->cng_rng = AEC_CNG_SEED;
        r->last_serial = serial;
    }
}

/* One hop, called right after aec_process_linear() + aec_process_residual() on the same instance.
 *   gain     applied to the base (single-channel, NR off: rv.res_gain; with NR: min(g_nr, g_res))
 *   g_res    the RES gain that decides how much comfort noise is added
 *   cn       comfort-noise PSD (int16^2 scale), NULL = comfort noise off
 *   g_nr     NR gain or NULL; its floor in the noise is AEC_CNG_NR_GAIN_FLOOR
 * Returns 0, or -1 when the view is unavailable. *guard_active is optional. */
static int renderer_render_hop(OutputRenderer* r, const Aec* aec, const float* gain, const float* g_res,
                               const float* cn, const float* g_nr, float* out,
                               int* guard_active) {
    AecResidualView rv;
    int k;

    renderer_sync_serial(r, aec);

    memset(&rv, 0, sizeof rv);
    rv.struct_size = sizeof rv;
    if (aec_get_residual(aec, &rv) != AEC_OK || rv.n_freqs != r->n_freqs || !gain) return -1;
    if (guard_active) *guard_active = rv.guard_active;

    /* (2) the rendering base, not error_spec */
    for (k = 0; k < r->n_freqs; ++k) {
        r->spec[k].r = rv.base_spec[k].r * gain[k];
        r->spec[k].i = rv.base_spec[k].i * gain[k];
    }
    /* (3) comfort noise with the library helper; seed and draw schedule match the internal stage */
    if (cn && g_res) aec_apply_cng_spec(&r->cng_rng, r->spec, cn, g_res, g_nr, r->n_freqs);
    ac_istft_push(r->istft, r->spec, out);
    return 0;
}

/* Single channel, NR off, residual FULL: the library's own helpers do the gain and the comfort noise. */
static int mono_render_hop(OutputRenderer* r, Aec* aec, int use_cng, float* out) {
    renderer_sync_serial(r, aec);
    if (aec_apply_gain(aec, NULL, r->spec) != AEC_OK) return -1;          /* base_spec * res_gain */
    if (use_cng && aec_apply_cng(aec, NULL, r->spec) != AEC_OK) return -1;   /* instance generator */
    ac_istft_push(r->istft, r->spec, out);
    return 0;
}
```

每 hop 的呼叫（單聲道）：

```c
static void run_mono(Aec* aec, OutputRenderer* r, int use_cng, const float* mic, const float* ref,
                     float* out, int n_hops, int hop) {
    int h;
    for (h = 0; h < n_hops; ++h) {
        aec_process_linear(aec, mic + (size_t)h * hop, ref + (size_t)h * hop);
        aec_process_residual(aec, NULL);
        mono_render_hop(r, aec, use_cng, out + (size_t)h * hop);
    }
}
```

要點：

- `use_cng` 就是你建立 `Aec` 時的 `cfg.enable_cng`。拆開的路徑本身不加舒適噪音，加不加由渲染器決定（`aec_apply_cng()` 不看 `enable_cng`，不要的話不呼叫）。
- 有 NR 的單聲道鏈照 §8.7 的順序：`aec_apply_gain(aec, g_nr, spec)` 與 `aec_apply_cng(aec, g_nr, spec)` 傳同一個 `g_nr`
  （舒適噪音的 NR 因子下限是 `AEC_CNG_NR_GAIN_FLOOR`）；改走 `renderer_render_hop()` 時 `gain` 傳 `g_total`，`g_res` 仍傳 `rv.res_gain`。
- `rv.base_spec` 指向函式庫內部緩衝區，只在下一次處理或 reset 呼叫前有效；`rv.struct_size` 是輸入，要先設。
- `AecResidualView` 內的指標都不要保存到下一次處理呼叫之後。

## 5. reset serial 協定（呼叫順序要一字不差）

`aec_post_reset_serial()` 計數「硬 post 鏈 reset」：`aec_reset()`，以及處理呼叫內部的硬 delay 重啟
（首次取得沒有 warm transfer 也沒有 DT-aware soft 取得，或確認位移沒有 DT-aware soft realign）。
bad-relock 還原、warm transfer、soft 取得／realign、`aec_apply_external_realign()` **不會**使它移動。
每次計數時函式庫都會清自己的 OLA 尾巴並把 CNG 種子設回 `AEC_CNG_SEED`；外部合成的應用必須做一樣的事
（用 `aec_apply_cng()` 時，產生器在實例內，函式庫自己重設，應用只需要清自己的 OLA 尾巴）。

每個 hop 的順序：

```text
1. aec_process_linear(aec, mic, ref); aec_process_residual(aec, NULL)   處理這個 hop（內部若硬 reset，serial 已經加一）
2. serial = aec_post_reset_serial(aec)
3. serial != last_serial ?
     是 -> ac_istft_reset(istft); cng_rng = AEC_CNG_SEED; last_serial = serial
4. aec_get_residual / aec_apply_gain                取增益與底座
5. spec = base_spec * gain; aec_apply_cng / aec_apply_cng_spec(...)    第一次 CNG 抽樣在步驟 3 之後
6. ac_istft_push(istft, spec, out)                  這個 hop 的 OLA push 在步驟 3 之後
```

規則：

- 步驟 3 必須在**這個 hop 的第一次 CNG 抽樣之前**與**這個 hop 的 `ac_istft_push` 之前**。晚一個 hop 才清，
  輸出就會多一個 hop 的舊 OLA 尾巴，和 `aec_process()` 對不上。
- **不可以**因 serial 變動而呼叫 `aec_reset()`：函式庫已經重置完了，再重置會把剛開始學的狀態清掉，而且 serial 又加一。
- 你自己呼叫 `aec_reset()` 時，serial 在它回傳時已經加一；不需要另外清你的渲染器，下一個處理過的 hop 在步驟 3 會自動處理。
  不要在呼叫 `aec_reset()` 的地方再手動清一次（重複清沒有壞處，但會讓你以為協定寫在兩個地方）。
- `renderer_init()` 在建立後把 `last_serial` 設成當時的 `aec_post_reset_serial()`（新實例為 0）。
- NR、NN 與其他下游狀態**函式庫不動**；它們要不要跟著 serial 變動而 reset，是應用自己的政策，
  協定只規定渲染器自己的 OLA 尾巴與 CNG 種子。
- 多個要各自渲染的 `Aec`（例如每條 lane）各有自己的 serial、`AcIstft`、`cng_rng`。

## 6. 合成視窗

函式庫內部合成用的視窗表與 `AcIstft` 預設的 sqrt-Hann 在最後一個 float 位元上不同。要與 `aec_process()`
逐位元相同，`AcIstft` 必須借用 `aec_synthesis_window()` 的表（§4.2 的 `renderer_init()` 已示範）：

- 表長 `n` 必須等於你設給 `AcIstft` 的 `fft_size`（`AecConfig` 建立時的格點）；
- 指標唯讀、歸 `Aec` 所有、在實例壽命內不變；`AcIstft` 只借用，**不可比 `Aec` 活得久**（先 free `AcIstft`，再 `aec_destroy`）；
- 回傳 `-1`（NULL 引數）時輸出參數不被寫入。

## 7. 舒適噪音函式與種子

- 用 `aec_apply_cng()`，或 `aec_apply_cng_spec(&rng, spec, cn, g_res, g_nr, n_freqs)`（`aec.h`；`g_nr` 的下限固定為 `AEC_CNG_NR_GAIN_FLOOR`），不要自己實作：
  函式庫內部輸出階段呼叫的就是同一個實作（`g_nr == NULL` 時逐位元相同）。`aec_apply_cng()` 是前者的實例形式
  （產生器在 `Aec` 內）；`aec_apply_cng_spec()` 的 `rng` 由呼叫端持有。
- `rng` 起點是 `AEC_CNG_SEED`（42）；每個 hop 在 bin `1 … n-2` 各抽一次，bin 0 與 `n-1` 不抽。
- `cn` 用 `rv.comfort_noise`（int16² 尺度），`g_res` 用決定舒適噪音量的 RES 增益（不是取 min 之後的增益）。
- 種子在兩個時機設回 `AEC_CNG_SEED`：建立時，以及每次 §5 的 serial 變動（`aec_apply_cng()` 由函式庫自己處理）。
  其他時候**不要**重設，包含 bad-relock 還原那一 hop（serial 沒動，CNG 照常延續）。
- 不要在 `aec_apply_cng()`／`aec_apply_cng_spec()` 外再疊一層舒適噪音；`enable_cng = 0` 時不呼叫它。

## 8. 輸出會怎麼變

遷移後，與你舊版輸出相比：

- **守衛 hop**：輸出由「殘差 × 增益」變成「擷取訊號 × 增益」，所以這些 hop 比舊版安靜（守衛觸發的條件就是殘差比擷取大）。
  守衛只在線性估計不可用時才可能觸發，多半在開頭，但從未變成可用的片段會一直觸發。
- **CNG 開啟的列**：函式庫內部本來就用種子 42。只有「舊程式用別的種子」的應用，舒適噪音的相位序列會改變
  （雜訊的統計量相同、逐樣本不同）。原本就用 `AEC_CNG_SEED` 的應用，這一項沒有變化。
- **差異不只出現在事件當下**：種子或 OLA 的差異發生在某個 hop 之後，會沿著 CNG 抽樣序列與 OLA 一直延續到下一次重設，
  所以「只比對事件那幾個 hop」會漏掉後面的差異。
- **含 bad-relock 還原的片段**：C 的還原現在是 soft 還原（回到還原前的 delay，OLA 尾巴與 CNG 延續，serial 不動；
  先前 C 走硬重置鏈），所以這類片段從還原那一 hop 起，輸出與舊版函式庫不同，不論你的應用有沒有遷移。
  遷移後的渲染器在還原那一 hop **不可**重設 OLA 或 CNG（不重設才與 `aec_process()` 一致）。沒有還原的片段不受這一項影響。
- **合成視窗**：最後一個 float 位元差。
- **不變**：`error_spec`／`echo_spec`／`near_spec`／`formed_hop`／`r2`／`res_gain` 等分析值，以及任何不渲染後級頻譜的用法；
  API 改名本身不改任何輸出位元。

因此**不要拿遷移後的輸出與舊版輸出做逐位元比較**當作驗收。驗收看 §9。

## 9. 驗收測試（可直接複製）

做法：同一份輸入、同樣的 `AecConfig`（含 `fft_size`、`enable_cng`，RES 與 CNG 設定相同，不接 NR），
建立兩個 `Aec`，兩者的 `residual` 都是 `AEC_RESIDUAL_FULL`：

- 一體實例用 `aec_process()` 取 `out`；
- 拆開實例用 `aec_process_linear()` + `aec_process_residual()` 加 §4.2 的渲染器。

每個 hop 的輸出必須**逐位元相同**，不接受容差。模型：`c_impl/test/test_guard_unify.c`
（外部應用是 `app_init()`／`app_synth()`，多實例組態在 `make_cfg()`，比對是 `run_stream()` 內的 `bits_eq_f()`；
刺激訊號是 `gen_echo_stream()`、`gen_uncorrelated_stream()`；`make test-guard-unify` 執行）。

```c
#include <stdio.h>

/* Returns 0: bit-identical and the stimulus exercised the guard and a reset; 1: output differs;
 * 2: identical but the stimulus never fired the guard or never moved the serial (the test proved nothing). */
static int guard_acceptance(const float* mic, const float* ref, int n_hops, int sample_rate, int fft_size) {
    AecConfig cfg;
    Aec* full = NULL;
    Aec* split = NULL;
    OutputRenderer r;
    float *out_full = NULL, *out_split = NULL;
    int hop, h, rc = 0;
    long diff_hops = 0, guard_hops = 0, serial_moves = 0;
    uint32_t prev_serial = 0;

    aec_config_from_level(&cfg, AEC_LEVEL_BALANCED, sample_rate);   /* use the level your product ships */
    cfg.fft_size = fft_size;
    cfg.residual = AEC_RESIDUAL_FULL;                               /* both instances: same config */
    if ((full = aec_create(&cfg)) == NULL || (split = aec_create(&cfg)) == NULL) return 1;
    hop = aec_hop_size(full);
    out_full = (float*)calloc((size_t)hop, sizeof(float));
    out_split = (float*)calloc((size_t)hop, sizeof(float));
    if (!out_full || !out_split || renderer_init(&r, split, fft_size) != 0) return 1;
    prev_serial = aec_post_reset_serial(split);

    for (h = 0; h < n_hops; ++h) {
        const float* m = mic + (size_t)h * hop;
        const float* f = ref + (size_t)h * hop;
        AecResidualView rv;
        int guard = 0;
        uint32_t s;

        if (h == n_hops / 2) { aec_reset(full); aec_reset(split); }   /* a reset in the middle of the stream */

        if (aec_process(full, m, f, out_full) != AEC_OK) { rc = 1; break; }
        if (aec_process_linear(split, m, f) != AEC_OK || aec_process_residual(split, NULL) != AEC_OK) { rc = 1; break; }
        memset(&rv, 0, sizeof rv);
        rv.struct_size = sizeof rv;
        if (aec_get_residual(split, &rv) != AEC_OK || !rv.res_gain ||
            renderer_render_hop(&r, split, rv.res_gain, rv.res_gain,
                                cfg.enable_cng ? rv.comfort_noise : NULL, NULL, out_split, &guard) != 0) {
            rc = 1; break;
        }
        if (memcmp(out_full, out_split, (size_t)hop * sizeof(float)) != 0) diff_hops++;
        guard_hops += guard;
        s = aec_post_reset_serial(split);
        if (s != prev_serial) serial_moves++;
        prev_serial = s;
    }
    printf("guard acceptance: %d hops, %ld differing, %ld guard hops, %ld serial moves\n",
           n_hops, diff_hops, guard_hops, serial_moves);
    if (rc == 0 && diff_hops != 0) rc = 1;
    if (rc == 0 && (guard_hops == 0 || serial_moves == 0)) rc = 2;

    renderer_free(&r);                 /* the AcIstft borrows the Aec's window: free it first */
    aec_destroy(split); aec_destroy(full);
    free(out_full); free(out_split);
    return rc;
}
```

使用方式與注意：

- **刺激訊號必須真的觸發守衛與 reset**，否則「全部相同」什麼也沒證明；上面的函式在這種情況回傳 2。
  遠端與近端不相關、線性濾波器只會加能量的訊號（`gen_uncorrelated_stream()` 那一類）會讓守衛一直觸發；
  含 delay 位移的回音串流會觸發硬 reset；中段的 `aec_reset()` 保證 serial 至少動一次。
  請用你產品實際的取樣率與 `fft_size`，至少各跑一個格點。
- 組態要與產品一致（level、`enable_cng`、delay 模式）；一體與拆開兩個實例必須是同一份 `AecConfig`，
  差別只在使用哪組進入點。
- **證明這個測試會紅**（做一次就好）：把渲染器故意弄壞，確認回傳 1。三種弄壞法各自對應一項遷移步驟：
  1. `rv.base_spec` 換成 `rv.error_spec`（步驟 2：守衛 hop 上會不同）；
  2. 拿掉 serial 比對（步驟 3：`aec_reset()` 之後的 hop 會不同）；
  3. 不借 `aec_synthesis_window()` 的表，用 `AcIstft` 預設視窗（步驟 4：會差最後一個位元）；
  另可把 `cng_rng` 起點改成別的值（步驟 5：CNG 開啟時會不同）。
- 有 NR 的產品：驗收仍只比 AEC-only 路徑（NR 關閉）。NR 之後的輸出沒有一體實例的對照物。
- 這個測試只確認「拆開渲染」與「函式庫內部渲染」一致；是否該用守衛的產品判斷不在它的範圍。

## 10. 清單

- [ ] 設定改為 `AecConfig.residual`，進入點與檢視改為 §3.1 的新名；程式裡沒有任何已移除的舊名。
- [ ] §3.3 的建立與設定舊名也已清除：`Aec` 只以指標持有、heap 實例只 `aec_destroy()` 一次（沒有再 `free`）、強度只用 `level`。
- [ ] 渲染後級頻譜的檔案加了 §2 的 `#if`／`#error`（版本 2）。
- [ ] 渲染用 `rv.base_spec`（或 `aec_apply_gain()`）；分析仍用 `error_spec`。
- [ ] 每個要渲染的 `Aec` 都有 serial 比對，且順序在 CNG 抽樣與 `ac_istft_push` 之前；沒有因此呼叫 `aec_reset()`。
- [ ] `AcIstft` 借 `aec_synthesis_window()` 的表，且先於 `Aec` 釋放。
- [ ] 舒適噪音走 `aec_apply_cng()` 或 `aec_apply_cng_spec()`，起點 `AEC_CNG_SEED`，只在建立與 serial 變動時重設。
- [ ] 沒有直接播放或重新合成 `formed_hop`／`linear_hop` 而期待有守衛。
- [ ] §9 的驗收在你實際使用的每個格點回傳 0，且做過一次「故意弄壞會變紅」。
- [ ] NR／NN 是否跟隨 serial 變動重設，已有應用層政策並寫在你自己的程式裡。
