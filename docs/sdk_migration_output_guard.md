# SDK 遷移清單：輸出守衛（output guard）

給**自己用 `AecResContext` 頻譜做渲染（合成時域輸出）**的既有 SDK 應用。只呼叫 `aec_process()` 並直接取
`out` 的應用不受影響，不必看本文。

相關文件：`c_user_manual_zh_TW_04_seam_troubleshooting.md` §8.1（分析 seam 與渲染底座）、§8.3
（`aec_get_output_base()`／`aec_post_reset_serial()` 的完整語意）、§8.7（單聲道整鏈）；
公開宣告在 `c_impl/include/aec.h`（`AEC_OUTPUT_GUARD_API_VERSION`、`aec_get_output_base`、
`aec_post_reset_serial`、`aec_synthesis_window`）與 `c_impl/include/aec_res.h`（`aec_cng_fill`）。

## 1. 先講結論：舊程式不會報錯，只會悄悄失去守衛

輸出守衛的規則：本 hop 線性估計不可用，且 `sum|error_spec|²` 大於 `sum|error_spec + echo_spec|²`
（殘差比擷取訊號還大）時，渲染的底座改用擷取訊號 `error_spec + echo_spec`，而不是 `error_spec`。
`aec_process()` 內部一直是這樣做；現在 context 路徑也提供同一個底座。

如果你的程式還是把 **`ctx.error_spec × 增益`** 送去反轉換：

- 編譯照過、執行不報錯、也沒有任何回傳值或旗標會告訴你；
- 但守衛 hop 上你的輸出比 `aec_process()` 的輸出大聲（殘差比擷取還大）；
- 另外你的合成視窗、舒適噪音種子、delay 硬重置時的 OLA／CNG 處理也可能和函式庫內部不同。

新增的 getter 無法偵測「呼叫端還在用 `error_spec` 渲染」，所以這件事只能靠**你主動遷移**。
`AEC_OUTPUT_GUARD_API_VERSION` 是給你放進自己建置的編譯期標記（見 §2），它不會強迫沒加檢查的舊程式中斷。
`AecResContext.error_spec` 的名稱與語意沒有改（它仍是分析用的真實線性殘差），所以不會有舊程式因改名而編譯失敗。

不在守衛範圍內：時域 seam（`AecResContext.formed_hop`、`AecLinearContext.formed_linear_hop`）永遠是分析殘差，
沒有時域渲染底座。直接播放或重新合成這兩個串流的應用拿不到輸出守衛，需要守衛就必須改走頻域路徑。

## 2. 版本旗標

`aec.h` 定義 `#define AEC_OUTPUT_GUARD_API_VERSION 1`，代表下列五項同時存在：
`aec_get_output_base()`、`aec_post_reset_serial()`、`aec_synthesis_window()`、
`aec_cng_fill()`（`aec_res.h`）與 `AEC_CNG_SEED`（`aec_post.h`，由 `aec.h` 引入）。
把下面幾行放進你渲染 context 頻譜的那個檔案；SDK 比這一版舊時會在編譯期停下來，而不是靜默通過：

```c
#include "aec.h"
#include "aec_res.h"      /* aec_cng_fill */
#include "audio_stft.h"   /* AcIstft */

#if !defined(AEC_OUTPUT_GUARD_API_VERSION) || AEC_OUTPUT_GUARD_API_VERSION < 1
#error "aec.h is older than the output-guard API: update the SDK before rendering context spectra"
#endif
```

之後輸出守衛相關 API 若再有不相容的變動，這個數字會加一；你的檢查寫成 `== 目前驗過的版本`
還是 `>=`，由你決定要不要在版本升級時被迫重新驗證。

## 3. 遷移步驟總表

| # | 要做的事 | 舊寫法 | 新寫法 |
|---|---|---|---|
| 1 | 編譯期檢查 | 無 | §2 的 `#if`／`#error` |
| 2 | 渲染底座 | `ctx.error_spec[k] × g[k]` | `ob.base_spec[k] × g[k]`，`ob` 來自 `aec_get_output_base()` |
| 3 | reset serial | 無，或自己從 delay 變動推測 | 每 hop 比對 `aec_post_reset_serial()`，變了就清 OLA 尾巴並重設 CNG 種子（§5） |
| 4 | 合成視窗 | `AcIstft` 預設 sqrt-Hann | 向 `aec_synthesis_window()` 借表（§6） |
| 5 | 舒適噪音 | 自己的 CNG，自選種子 | `aec_cng_fill()`，種子 `AEC_CNG_SEED`（§7） |
| 6 | 驗收 | 與舊輸出比較 | 與同一份輸入下全模式 `aec_process()` 逐位元比較（§9） |

分析用途（`r2`、`error_spec` 給 NR／`AecRes` 的輸入、ERLE 類統計、訓練特徵、需要與 `formed_hop` 配對的重建）
**繼續讀 `error_spec`**，不要改成底座；底座只用來渲染。

## 4. 前後對照程式碼

### 4.1 舊寫法（守衛消失）

```c
#include "aec.h"
#include "audio_stft.h"

void my_comfort_noise(uint32_t* rng, Complex* spec, const AecResContext* ctx);

/* BEFORE: renders error_spec, default-window AcIstft, own CNG with its own seed, no reset handling. */
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

單聲道：`enable_res = 0`、`return_res_context = 1` 建立 AEC，增益就是 `ctx.res_gain`，不接 NR。
lane（`spatial_linear_context` 為 1 或 2）：`ctx.res_gain` 為 `NULL`，增益與 `comfort_noise` 由應用自己的來源提供
（例如 `aec_res_process()` 的輸出），把它們傳進同一個函式即可；渲染的頻譜一樣是該 lane 的 `ob.base_spec`。
下面是單一渲染器，單聲道與 lane 各持有一份（每個要渲染的 `Aec` 各有自己的 `AcIstft`、`cng_rng` 與 `last_serial`）。

```c
#include <stdlib.h>
#include <string.h>

typedef struct OutputRenderer {
    AcIstft* istft;
    void*    istft_mem;
    Complex* spec;          /* n_freqs scratch */
    uint32_t cng_rng;       /* comfort-noise LCG state, owned by the application */
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

/* One hop, called right after aec_process_context() on the same instance.
 *   gain     applied to the base (single-channel, NR off: ctx.res_gain; with NR: min(g_nr, g_res))
 *   g_res    the RES gain that decides how much comfort noise is added
 *   cn       comfort-noise PSD (int16^2 scale), NULL = comfort noise off
 *   g_nr     NR gain or NULL; nr_floor its floor (0.31622777f in the single-channel chain)
 * Returns 0, or -1 when the base is unavailable. *guard_active is optional. */
static int renderer_render_hop(OutputRenderer* r, const Aec* aec, const float* gain, const float* g_res,
                               const float* cn, const float* g_nr, float nr_floor, float* out,
                               int* guard_active) {
    AecOutputBase ob;
    uint32_t serial = aec_post_reset_serial(aec);
    int k;

    /* (1) BEFORE the first CNG draw and BEFORE ac_istft_push of this hop. Never reset the AEC for this. */
    if (serial != r->last_serial) {
        ac_istft_reset(r->istft);
        r->cng_rng = AEC_CNG_SEED;
        r->last_serial = serial;
    }

    memset(&ob, 0, sizeof ob);
    ob.struct_size = sizeof ob;
    if (aec_get_output_base(aec, &ob) != 0 || ob.n_freqs != r->n_freqs || !gain) return -1;
    if (guard_active) *guard_active = ob.guard_active;

    /* (2) the rendering base, not error_spec */
    for (k = 0; k < r->n_freqs; ++k) {
        r->spec[k].r = ob.base_spec[k].r * gain[k];
        r->spec[k].i = ob.base_spec[k].i * gain[k];
    }
    /* (3) comfort noise with the library helper; seed and draw schedule match the internal stage */
    if (cn && g_res) aec_cng_fill(&r->cng_rng, r->spec, cn, g_res, g_nr, nr_floor, r->n_freqs);
    ac_istft_push(r->istft, r->spec, out);
    return 0;
}

/* Single channel, NR off: the gain is ctx.res_gain and comfort noise follows cfg.enable_cng. */
static int mono_render_hop(OutputRenderer* r, const Aec* aec, int use_cng, float* out) {
    AecResContext ctx;
    aec_get_res_context(aec, &ctx);
    if (!ctx.res_gain) return -1;
    return renderer_render_hop(r, aec, ctx.res_gain, ctx.res_gain,
                               use_cng ? ctx.comfort_noise : NULL, NULL, 0.0f, out, NULL);
}
```

每 hop 的呼叫（單聲道）：

```c
static void run_mono(Aec* aec, OutputRenderer* r, int use_cng, const float* mic, const float* ref,
                     float* out, int n_hops, int hop) {
    int h;
    for (h = 0; h < n_hops; ++h) {
        aec_process_context(aec, mic + (size_t)h * hop, ref + (size_t)h * hop);
        mono_render_hop(r, aec, use_cng, out + (size_t)h * hop);
    }
}
```

要點：

- `use_cng` 就是你建立 `Aec` 時的 `cfg.enable_cng`。context 路徑本身不加舒適噪音，加不加由渲染器決定。
- 有 NR 的單聲道鏈照 §8.7 的順序：增益取 `min(g_nr, res_gain)`，再傳 `g_nr` 與 `0.31622777f` 給
  `renderer_render_hop()`；`gain` 傳 `g_total`，`g_res` 仍傳 `ctx.res_gain`。
- `ob.base_spec` 指向函式庫內部緩衝區，只在下一次處理或 reset 呼叫前有效；`ob.struct_size` 是輸入，要先設。
- `AecResContext` 與 `AecOutputBase` 內的指標都不要保存到下一次處理呼叫之後。

## 5. reset serial 協定（呼叫順序要一字不差）

`aec_post_reset_serial()` 計數「硬 post 鏈 reset」：`aec_reset()`，以及處理呼叫內部的硬 delay 重啟
（首次取得沒有 warm transfer 也沒有 DT-aware soft 取得，或確認位移沒有 DT-aware soft realign）。
bad-relock 還原、warm transfer、soft 取得／realign、`aec_apply_external_realign()` **不會**使它移動。
每次計數時函式庫都會清自己的 OLA 尾巴並把 CNG 種子設回 `AEC_CNG_SEED`；外部合成的應用必須做一樣的事。

每個 hop 的順序：

```text
1. aec_process_context(aec, mic, ref)             處理這個 hop（內部若硬 reset，serial 已經加一）
2. serial = aec_post_reset_serial(aec)
3. serial != last_serial ?
     是 -> ac_istft_reset(istft); cng_rng = AEC_CNG_SEED; last_serial = serial
4. aec_get_res_context / aec_get_output_base       取增益與底座
5. spec = base_spec * gain; aec_cng_fill(...)      第一次 CNG 抽樣在步驟 3 之後
6. ac_istft_push(istft, spec, out)                 這個 hop 的 OLA push 在步驟 3 之後
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

- 用 `aec_cng_fill(&rng, spec, cn, g_res, g_nr, nr_floor, n_freqs)`（`aec_res.h`），不要自己實作：
  函式庫內部輸出階段呼叫的就是它（`g_nr == NULL` 時逐位元相同）。
- `rng` 由呼叫端持有，起點是 `AEC_CNG_SEED`（42）；每個 hop 在 bin `1 … n-2` 各抽一次，bin 0 與 `n-1` 不抽。
- `cn` 用 `ctx.comfort_noise`（int16² 尺度），`g_res` 用決定舒適噪音量的 RES 增益（不是取 min 之後的增益）。
- 種子在兩個時機設回 `AEC_CNG_SEED`：建立時，以及每次 §5 的 serial 變動。其他時候**不要**重設，
  包含 bad-relock 還原那一 hop（serial 沒動，CNG 照常延續）。
- 不要在 `aec_cng_fill` 外再疊一層舒適噪音；`enable_cng = 0` 時不呼叫它。

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
- **不變**：`error_spec`／`echo_spec`／`near_spec`／`formed_hop`／`r2`／`res_gain` 等分析值，以及任何不渲染 context 頻譜的用法。

因此**不要拿遷移後的輸出與舊版輸出做逐位元比較**當作驗收。驗收看 §9。

## 9. 驗收測試（可直接複製）

做法：同一份輸入、同樣的 `AecConfig`（含 `fft_size`、`enable_cng`，RES 與 CNG 設定相同，不接 NR），
建立兩個 `Aec`：

- 全模式實例 `enable_res = 1`、`return_res_context = 0`，用 `aec_process()` 取 `out`；
- context 實例 `enable_res = 0`、`return_res_context = 1`，用 `aec_process_context()` 加 §4.2 的渲染器。

每個 hop 的輸出必須**逐位元相同**，不接受容差。模型：`c_impl/test/test_guard_unify.c`
（外部應用是 `app_init()`／`app_synth()`，雙實例組態在 `make_cfg()`，比對是 `run_stream()` 內的 `bits_eq_f()`；
刺激訊號是 `gen_echo_stream()`、`gen_uncorrelated_stream()`；`make test-guard-unify` 執行）。

```c
#include <stdio.h>

/* Returns 0: bit-identical and the stimulus exercised the guard and a reset; 1: output differs;
 * 2: identical but the stimulus never fired the guard or never moved the serial (the test proved nothing). */
static int guard_acceptance(const float* mic, const float* ref, int n_hops, int sample_rate, int fft_size) {
    AecConfig cf, ck;
    Aec* full = (Aec*)calloc(1, sizeof(Aec));
    Aec* split = (Aec*)calloc(1, sizeof(Aec));
    OutputRenderer r;
    float *out_full = NULL, *out_split = NULL;
    int hop, h, rc = 0;
    long diff_hops = 0, guard_hops = 0, serial_moves = 0;
    uint32_t prev_serial = 0;

    aec_config_from_level(&cf, AEC_LEVEL_BALANCED, sample_rate);   /* use the level your product ships */
    cf.fft_size = fft_size;
    ck = cf;
    cf.enable_res = 1; cf.return_res_context = 0;                  /* full mode */
    ck.enable_res = 0; ck.return_res_context = 1;                  /* context mode, same CNG setting */
    if (!full || !split || aec_create(full, &cf) != 0 || aec_create(split, &ck) != 0) return 1;
    hop = aec_hop_size(full);
    out_full = (float*)calloc((size_t)hop, sizeof(float));
    out_split = (float*)calloc((size_t)hop, sizeof(float));
    if (!out_full || !out_split || renderer_init(&r, split, fft_size) != 0) return 1;
    prev_serial = aec_post_reset_serial(split);

    for (h = 0; h < n_hops; ++h) {
        const float* m = mic + (size_t)h * hop;
        const float* f = ref + (size_t)h * hop;
        AecResContext ctx;
        int guard = 0;
        uint32_t s;

        if (h == n_hops / 2) { aec_reset(full); aec_reset(split); }   /* a reset in the middle of the stream */

        aec_process(full, m, f, out_full);
        aec_process_context(split, m, f);
        aec_get_res_context(split, &ctx);
        if (!ctx.res_gain ||
            renderer_render_hop(&r, split, ctx.res_gain, ctx.res_gain,
                                ck.enable_cng ? ctx.comfort_noise : NULL, NULL, 0.0f, out_split, &guard) != 0) {
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
    free(split); free(full); free(out_full); free(out_split);
    return rc;
}
```

使用方式與注意：

- **刺激訊號必須真的觸發守衛與 reset**，否則「全部相同」什麼也沒證明；上面的函式在這種情況回傳 2。
  遠端與近端不相關、線性濾波器只會加能量的訊號（`gen_uncorrelated_stream()` 那一類）會讓守衛一直觸發；
  含 delay 位移的回音串流會觸發硬 reset；中段的 `aec_reset()` 保證 serial 至少動一次。
  請用你產品實際的取樣率與 `fft_size`，至少各跑一個格點。
- 組態要與產品一致（level、`enable_cng`、delay 模式）；全模式與 context 實例必須是同一份 `AecConfig`，只有
  `enable_res`／`return_res_context` 不同。
- **證明這個測試會紅**（做一次就好）：把渲染器故意弄壞，確認回傳 1。三種弄壞法各自對應一項遷移步驟：
  1. `ob.base_spec` 換成 `ctx.error_spec`（步驟 2：守衛 hop 上會不同）；
  2. 拿掉 serial 比對（步驟 3：`aec_reset()` 之後的 hop 會不同）；
  3. 不借 `aec_synthesis_window()` 的表，用 `AcIstft` 預設視窗（步驟 4：會差最後一個位元）；
  另可把 `cng_rng` 起點改成別的值（步驟 5：CNG 開啟時會不同）。
- 有 NR 的產品：驗收仍只比 AEC-only 路徑（NR 關閉）。NR 之後的輸出沒有全模式的對照物。
- 這個測試只確認「拆開渲染」與「函式庫內部渲染」一致；是否該用守衛的產品判斷不在它的範圍。

## 10. 清單

- [ ] 渲染 context 頻譜的檔案加了 §2 的 `#if`／`#error`。
- [ ] 渲染用 `ob.base_spec`；分析仍用 `error_spec`。
- [ ] 每個要渲染的 `Aec` 都有 serial 比對，且順序在 CNG 抽樣與 `ac_istft_push` 之前；沒有因此呼叫 `aec_reset()`。
- [ ] `AcIstft` 借 `aec_synthesis_window()` 的表，且先於 `Aec` 釋放。
- [ ] 舒適噪音走 `aec_cng_fill()`，起點 `AEC_CNG_SEED`，只在建立與 serial 變動時重設。
- [ ] 沒有直接播放或重新合成 `formed_hop`／`formed_linear_hop` 而期待有守衛。
- [ ] §9 的驗收在你實際使用的每個格點回傳 0，且做過一次「故意弄壞會變紅」。
- [ ] NR／NN 是否跟隨 serial 變動重設，已有應用層政策並寫在你自己的程式裡。
