# AEC C User Manual（繁體中文）

本手冊說明 `c_impl/` 內單麥克風、單播放參考訊號 AEC 的建置、命令列操作與 C API 整合方式。演算法細節請參考 `aec_methods.md`；既有英文文件 `c_user_and_integration_guide.md` 保留作為延伸參考。

> 驗證基準：AEC commit `f9e7123`（`main`，2026-06-26）。本手冊以當前 `c_impl/include/aec.h`、`c_impl/example/aec_wav.c` 與 `c_impl/Makefile` 為準。

## 1. 模組用途與訊號定義

AEC 用於 full-duplex hands-free 場景，處理一組同步的單聲道訊號：

- `mic`：麥克風 capture，包含近端語音、背景噪聲與喇叭耦合回聲。
- `ref`：實際送往喇叭的 playback loopback／render reference。
- `out`：AEC 線性濾波、residual echo suppression 與 limiter 後的單聲道輸出。

`ref` 不能使用任意的遠端錄音檔替代；它必須對應裝置真正播放的內容。mic/ref 必須使用相同 sample rate 並保持時鐘同步。長期 clock drift 會使 delay estimation 與 filter convergence 失效。

生產路徑包含 HPF、saturation handling、online delay estimation、PBFDKF main filter、shadow filter、echo-path-change handling、AEC3-aligned residual suppression 與可選的 comfort noise。

## 2. 建置與連結

```bash
cd c_impl

make            # bin/aec_wav
make lib        # bin/libaec.a
make debug      # 加入 -g -DAEC_DEBUG
```

預設使用 repo 內附的 KISS FFT。ARM 平台可改用 NE10：

```bash
make NE10_DIR=/path/to/Ne10 CC=aarch64-linux-gnu-gcc
make lib NE10_DIR=/path/to/Ne10 CC=aarch64-linux-gnu-gcc
```

整合 application 時必須保留 `-ffp-contract=off`：

```bash
cc -std=c99 -O2 -ffp-contract=off app.c \
   -Ic_impl/include -Lc_impl/bin -laec -lm -o app
```

省略 `-ffp-contract=off` 可能讓 compiler 將運算融合成 FMA，造成 HPF、detector 與遞迴 state 和驗證版本不一致。若使用 NE10，link 命令還要加入 NE10 library path 與 `-lNE10`。

## 3. 命令列工具

```bash
# balanced preset
./bin/aec_wav mic.wav ref.wav out.wav

# preset
./bin/aec_wav mic.wav ref.wav out.wav --preset gentle
./bin/aec_wav mic.wav ref.wav out.wav --preset aggressive

# 模組開關
./bin/aec_wav mic.wav ref.wav out.wav --cng
./bin/aec_wav mic.wav ref.wav out.wav --no-cng
./bin/aec_wav mic.wav ref.wav out.wav --no-delay-est
./bin/aec_wav mic.wav ref.wav out.wav --no-res
./bin/aec_wav mic.wav ref.wav out.wav --no-shadow
./bin/aec_wav mic.wav ref.wav out.wav --no-hpf

# 診斷
./bin/aec_wav mic.wav ref.wav out.wav --debug-level 2
./bin/aec_wav mic.wav ref.wav out.wav --debug-level 2 --debug-log /tmp/aec.log
./bin/aec_wav mic.wav ref.wav out.wav --debug-trace /tmp/aec_trace.csv
```

未知 preset 或 option 會回傳 exit code 2，不會悄悄退回 balanced。`--debug-level` 的有效使用範圍為 0 到 3。

### 3.1 Preset

三個 preset 目前只改變 far-end active 時的最低 suppression gain floor：

| Preset | `min_gain_floor_far_active_db` | 方向 |
|---|---:|---|
| `gentle` | -20 dB | 近端保留優先，允許較多 residual echo |
| `balanced` | -28 dB | 預設／一般通話 |
| `aggressive` | -38 dB | 回聲抑制優先，double-talk 近端損失風險較高 |

### 3.2 WAV contract

- 建議／驗證 sample rate：8、16、48 kHz。
- 輸入支援 PCM16、PCM32 與 IEEE float32 WAV；多聲道只取第一聲道。
- mic/ref sample rate 必須相同，且應從同一時間基準開始。
- CLI 處理兩個輸入中較短的長度；最後不足一個 hop 的尾端不處理。
- `aec_wav` 未設定 `AEC_OUT_FLOAT` 時會主動選擇單聲道 float32 WAV，方便 parity 比較。
- 要輸出 PCM16：`AEC_OUT_FLOAT=0 ./bin/aec_wav ...`；要明確輸出 float32：`AEC_OUT_FLOAT=1 ./bin/aec_wav ...`。

底層 `wav_io.h` 本身的 writer 預設是 PCM16，但 `aec_wav.c` 在開啟 writer 前會設定 float32；不要混淆兩者的預設行為。

## 4. Lockstep C API：完整 lifecycle

每次 `aec_process()` 必須剛好傳入 `aec_hop_size()` 個 float sample。函式沒有 length 參數，少給會越界讀取，多給則不會被處理。

以下範例可存成 `aec_stream.c`，並以 `cc -std=c99 -ffp-contract=off -c aec_stream.c -Ic_impl/include` 編譯。`read_hop` 成功提供一個完整 block 時回傳 1、EOF 回傳 0、I/O 錯誤回傳負值。

```c
/* AEC_STREAM_EXAMPLE_BEGIN */
#include <stddef.h>
#include <stdlib.h>

#include "aec.h"

typedef int (*AecReadHop)(float *mic, float *ref, int hop, void *user);
typedef int (*AecWriteHop)(const float *out, int hop, void *user);

int run_aec_stream(int sample_rate, AecPreset preset,
                   AecReadHop read_hop, AecWriteHop write_hop,
                   void *user)
{
    AecConfig cfg;
    Aec aec;
    float *mic = NULL;
    float *ref = NULL;
    float *out = NULL;
    int hop;
    int rc = -1;
    int read_rc;

    if (!read_hop || !write_hop) return -1;

    aec_config_from_preset(&cfg, preset, sample_rate);
    cfg.enable_cng = 1;
    cfg.enable_delay_est = 1;

    if (aec_create(&aec, &cfg) != 0)
        return -1;

    hop = aec_hop_size(&aec);
    mic = (float *)malloc((size_t)hop * sizeof(float));
    ref = (float *)malloc((size_t)hop * sizeof(float));
    out = (float *)malloc((size_t)hop * sizeof(float));
    if (!mic || !ref || !out) goto cleanup;

    while ((read_rc = read_hop(mic, ref, hop, user)) == 1) {
        aec_process(&aec, mic, ref, out);
        if (write_hop(out, hop, user) != 0) goto cleanup;
    }
    if (read_rc < 0) goto cleanup;
    rc = 0;

cleanup:
    free(mic);
    free(ref);
    free(out);
    aec_destroy(&aec);
    return rc;
}
/* AEC_STREAM_EXAMPLE_END */
```

若要重用同一個 instance 處理另一條**獨立**串流，先呼叫 `aec_reset(&aec)`；不能讓前一條串流的 delay、filter、double-talk 與 convergence state 洩漏到下一條。

`aec_process()` 回傳型別是 `void`。應在 create 階段檢查 `aec_create()`，並由上層保證 pointer、block size、sample format 與 stream synchronization 正確。

## 5. Decoupled render/capture API

若 playback/render 與 microphone/capture 不是由同一 callback 交付，可使用：

```c
AecBufferingEvent e1 = aec_analyze_render(&aec, ref_hop);
AecBufferingEvent e2 = aec_process_capture(&aec, mic_hop, out_hop);
```

兩個函式仍然各收一個完整 hop。內部約 320 ms 的 render FIFO 是用來吸收 API scheduling jitter，不是 acoustic echo delay buffer。

事件處理範例：

```c
static void count_aec_event(AecBufferingEvent event,
                            unsigned *underruns, unsigned *overruns)
{
    if (event == AEC_BUF_RENDER_UNDERRUN) ++*underruns;
    if (event == AEC_BUF_RENDER_OVERRUN) ++*overruns;
}

/* render callback */
count_aec_event(aec_analyze_render(&aec, ref), &underruns, &overruns);

/* capture callback */
count_aec_event(aec_process_capture(&aec, mic, out),
                &underruns, &overruns);
```

- `AEC_BUF_RENDER_UNDERRUN`：capture 找不到 buffered render；本次以 silent render 處理。
- `AEC_BUF_RENDER_OVERRUN`：render FIFO 已滿；最舊的 hop 被丟棄。
- `AEC_BUF_API_SKEW`：render/capture 呼叫順序異常的診斷事件。
- `aec_last_buffering_event()` 可讀取最近一次 capture event。

Lockstep 呼叫 `aec_analyze_render(ref)` 後立即呼叫 `aec_process_capture(mic, out)`，應與 `aec_process(mic, ref, out)` 一致。

「decoupled」不等於 instance 內建 thread synchronization。若兩個 OS thread 同時呼叫同一個 `Aec`，application 必須自行序列化；否則請將 render/capture block 送到同一 audio worker。多條通話串流必須各自建立 instance。

## 6. Frame、delay 與 latency contract

hop 由 sample rate 推導：

| Sample rate | Hop | 時間 |
|---:|---:|---:|
| 8 kHz | 80 samples | 10 ms |
| 16 kHz | 160 samples | 10 ms |
| 48 kHz | 480 samples | 10 ms |

內部 analysis block 是兩個 hop，使用 50% overlap。啟用 residual suppression 時通常要把一個 hop（約 10 ms）的 OLA 納入 latency budget；關閉 RES 後為線性路徑，沒有這段 post-filter OLA 延遲。

online delay estimator 預設開啟，最大搜尋範圍由 `max_delay_ms` 控制。它不能修復 mic/ref 不同 sample rate、持續 clock drift 或錯誤的 render reference。

離線資料若已精準 pre-align，可在 create 前設 `cfg.enable_delay_est = 0`，避免初次 acquisition 造成 filter state 調整。線上產品通常維持開啟。

## 7. 重要 `AecConfig`

先呼叫 `aec_config_from_preset()`，再做少量必要 override：（`aec.h` 也宣告了 `aec_config_defaults()`，可取得不含 preset 差異化欄位的基礎預設值，兩者都是建立 `AecConfig` 的起點。）

```c
AecConfig cfg;
aec_config_from_preset(&cfg, AEC_PRESET_BALANCED, 16000);
cfg.enable_cng = 1;
cfg.enable_delay_est = 1;
cfg.enable_highpass = 1;
```

重要欄位：

| 欄位 | 預設 | 說明 |
|---|---:|---|
| `sample_rate` | caller 指定 | 建立後不可更換 |
| `filter_length` | 832 samples | 16 kHz 時約 52 ms；不是 millisecond 欄位 |
| `enable_cng` | 1 | comfort noise |
| `enable_delay_est` | 1 | online delay estimator |
| `enable_highpass` | 1 | mic 端 80 Hz HPF |
| `enable_saturation` | 1 | clipping detection／ref soft clip |
| `enable_shadow` | 1 | shadow adaptive filter |
| `enable_res` | 1 | AEC3 post-filter suppression |
| `max_delay_ms` | 1024 | delay 搜尋上限 |
| `warmup_frames` | 100 | 10 ms hop 下約 1 秒 |
| `delay_acquire_warm_transfer` | 1 | 首次 delay acquisition 時的 warm tap-transfer（v3.24.1）：搬移已學到的 IR 而非歸零，讓 realign 前的 cold-start 抑制效果得以保留 |
| `delay_acquire_inst_erle_db` | 4.0 | double；warm tap-transfer 的 inst-ERLE 峰值門檻（dB），超過才觸發搬移 |
| `dt_aware_recovery_soft` | 1 | double-talk 期間允許 soft（非破壞性）Path A/B／EPV／shadow-rise realign |
| `dt_aware_res_floor_enabled` | 1 | 啟用 double-talk 專用的 RES 最低 gain floor |
| `min_gain_floor_dt_db` | -16.0 | double-talk 期間套用的 RES gain floor（dB），與 preset 的 `min_gain_floor_far_active_db` 分開設定 |
| `ne_recent_threshold` | 0.3 | double；near-end「近期出現」判定的能量門檻（用 double 對齊 Python `float(0.3)`） |
| `ne_recent_hold` | 150 | near-end 近期出現旗標的 hold 框數 |
| `ne_recent_sustain` | 3 | 判定 near-end 近期出現前需連續高於門檻的框數 |
| `filter_misadjustment_stable_frames` | 30 | FilterMisadjustmentEstimator（AEC3 ScaleFilter）判定穩定所需框數 |
| `filter_misadjustment_hangover_frames` | 100 | misadjustment 狀態的 hangover 框數 |
| `filter_misadjustment_scale_min` | 0.5 | misadjustment scale 修正下限 |
| `filter_misadjustment_scale_max` | 2.0 | misadjustment scale 修正上限 |

`filter_length` 是 sample 數，預設固定為 832；換到 8／48 kHz 時若要維持相同毫秒數，application 應在 create 前自行換算。`n_partitions=0` 會由 create 根據 filter length 與 hop 推導。

改變 sample rate、filter length、partition count 或其他會改變 allocation/layout 的欄位時，必須 destroy 再 create。不要在 process 途中直接修改 `aec.cfg`。

## 8. 進階：`AecResContext` integration seam

只有需要自行串接 NR、NN post-filter 或外部 suppression gain 的整合者才需要此介面。一般 AEC 使用 `aec_process()` 即可。

建立前設定：

```c
AecConfig cfg;
aec_config_from_preset(&cfg, AEC_PRESET_BALANCED, 16000);
cfg.enable_res = 0;          /* time output 保持 linear residual */
cfg.return_res_context = 1;  /* post block 仍計算並公開 seam */
```

每次 process 後：

```c
AecResContext ctx;
aec_process(&aec, mic, ref, linear_out);
aec_get_res_context(&aec, &ctx);
```

主要欄位：

- `error_spec`：windowed linear error `E(f)`，audio amplitude scale。
- `res_gain`：AEC3 suppression amplitude gain，範圍 `[0, 1]`。
- `r2`：residual echo PSD，使用 int16² scale；轉成 `|E|²` audio scale 時除以 `32768²`。
- `comfort_noise`：CNG PSD，同樣使用 int16² scale。
- `far_power`、`dt_indicator`、`divergence`、`filter_converged`：外部 gate／診斷資訊。

這些 pointer alias AEC instance 的內部 per-hop buffer：不可修改、不可 free，且必須在下一次 `aec_process()`／`aec_process_capture()`、reset 或 destroy 前使用或複製。`return_res_context=0` 時，不應假設 frequency seam pointer 有效。

## 9. Resource ownership 與 thread safety

- `Aec` storage 由 caller 持有；`aec_create()` 會配置其內部動態資源。
- `aec_destroy()` 必須和成功的 `aec_create()` 成對，且只呼叫一次。
- `aec_reset()` 清 state 但保留 config／allocation，適合同規格的新串流。
- `mic`、`ref`、`out` buffer 由 caller 持有，至少包含一個 hop。
- `Aec` 是 stateful、非 reentrant；同一 instance 不可無同步並行呼叫。
- processing path 不提供 per-call error code；buffer underrun／overrun只在 decoupled API 以 event 回報。

## 10. Troubleshooting

| 現象 | 先檢查 | 建議 |
|---|---|---|
| 回聲幾乎沒有下降 | ref 不是實際 playback loopback | 先修正 reference routing，再調參數 |
| 長時間後回聲變差 | mic/ref clock drift | 使用共同 clock 或上層 drift compensation |
| 開始約一秒回聲較多 | filter 尚未 convergence／delay acquisition | 保持 far-end energy，確認 delay search 可涵蓋實際延遲 |
| double-talk 近端被壓低 | preset 太 aggressive | 先改 balanced/gentle，不要先關閉整個 RES |
| far-end only 仍有殘響尾 | acoustic path 超過 filter | 增加 `filter_length` 後重新 create，並重新評估記憶體與 CPU |
| 輸出出現突波／發散 | clipping、錯誤 ref、delay 劇變 | 保留 saturation/shadow/EPC，開 debug trace 定位 |
| async API 持續 underrun | render/capture 排程或 producer 太慢 | 監控 event，修正 scheduling；不要把 FIFO 當 clock-drift 修正器 |
| batch 第二個檔案結果異常 | 沿用前一檔 state | 每個獨立檔案前 `aec_reset()` |
| Python/C 細微不一致 | build flags 或 FFT backend 不同 | 確認 `-ffp-contract=off`、preset、CNG、sample format 與 backend |

回報問題時請附 mic/ref 測試片段、完整命令、sample rate、preset、CNG/delay 設定，以及問題前後至少一秒的 debug log 或 trace。
