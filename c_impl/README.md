# AEC C Implementation — PBFDAF

C 語言 AEC 實作，僅支援 **PBFDAF**（Partitioned Block Frequency Domain Adaptive Filter）模式。

## 編譯

```bash
make          # 編譯 CLI 工具
make lib      # 編譯靜態函式庫 (libaec.a)
make clean    # 清除編譯產物
```

## CLI 使用

```bash
# 基本用法（DTD + RES 預設開啟）
./bin/aec_wav mic.wav ref.wav output.wav

# 調整參數
./bin/aec_wav mic.wav ref.wav output.wav --mu 0.5 --filter 2048

# 關閉 DTD / RES
./bin/aec_wav mic.wav ref.wav output.wav --no-dtd
./bin/aec_wav mic.wav ref.wav output.wav --no-res

# 開啟 Shadow Filter (雙濾波器)
./bin/aec_wav mic.wav ref.wav output.wav --enable-shadow

# 調整 RES 最小增益
./bin/aec_wav mic.wav ref.wav output.wav --res-gmin -25
```

### CLI 參數

| 參數 | 預設值 | 說明 |
|------|--------|------|
| `--mu <value>` | 0.3 | 步長 |
| `--filter <samples>` | 1024 | 濾波器長度 (samples) |
| `--no-dtd` | - | 關閉發散偵測 |
| `--no-res` | - | 關閉殘餘回音抑制 |
| `--enable-res` | 預設開啟 | 開啟殘餘回音抑制 |
| `--res-gmin <dB>` | -20 | RES 最小增益 (dB) |
| `--enable-shadow` | - | 開啟 Shadow Filter (雙濾波器) |
| `--no-shadow` | 預設關閉 | 關閉 Shadow Filter |

## API

```c
#include "aec.h"

// 建立 AEC（預設開啟 DTD + RES，Shadow 關閉）
AecConfig config = aec_default_config(16000);
config.mu = 0.3f;
config.filter_length = 1024;
// config.enable_shadow = true;  // 可選：開啟雙濾波器自動修正

Aec* aec = aec_create(&config);
int hop_size = aec_get_hop_size(aec);  // 256 @ 16kHz

while (has_audio) {
    aec_process(aec, mic_buf, ref_buf, out_buf);

    float erle = aec_get_erle(aec);           // ERLE (dB)
    float conf = aec_get_dtd_confidence(aec); // DTD 信心度 [0,1]
}

aec_destroy(aec);
```

### 主要 API 函式

| 函式 | 說明 |
|------|------|
| `aec_create(config)` | 建立 AEC 實例 |
| `aec_destroy(aec)` | 釋放資源 |
| `aec_process(aec, near, far, out)` | 處理一個 hop（256 samples） |
| `aec_reset(aec)` | 完全重置 |
| `aec_retrain(aec)` | 重置權重，保留緩衝區 |
| `aec_get_erle(aec)` | 取得 ERLE (dB) |
| `aec_get_dtd_confidence(aec)` | 取得 DTD 信心度 [0,1] |
| `aec_get_hop_size(aec)` | Hop size (256 @ 16kHz) |

## 處理流程

```
far-end ──┐
           v
near-end ──> PBFDAF (Main) ──> Shadow Compare ──> DTD ──> RES ──> Limiter ──> output
              ^                  (optional)         |
              └── mu_scale <────────────────────────┘

DTD (Dual Detector):
  1. Divergence: output > input × 1.2 → 降 mu（偵測 filter 發散）
  2. Coherence:  MSC(error, far) 低 + error 能量高 → 降 mu（偵測 double-talk）
  confidence = max(div, coh) → mu_scale = 1.0 - conf × 0.95

Shadow Filter (optional):
  - 保守步長 (mu × 0.5)，永遠全速更新（不受 DTD 控制）
  - 若 shadow error < main error × 0.8 → 複製權重 + echo_spec 到 main
```

1. **PBFDAF**: 分區頻域自適應濾波，估計並消除回音
2. **Shadow Filter** (可選): 保守步長的影子濾波器，發散時自動修正主濾波器
3. **DTD (Dual Detector)**: Divergence 偵測 filter 發散 + Coherence 偵測 double-talk，透過 mu_scale 連續控制更新速率
4. **RES**: 殘餘回音抑制，頻域譜減法進一步消除殘餘回音
5. **Output Limiter**: 確保輸出不超過麥克風振幅

## 檔案結構

```
c_impl/
├── include/
│   ├── aec.h              # 主要 API
│   ├── aec_types.h        # 配置結構與預設值
│   ├── subband_nlms.h     # PBFDAF 介面
│   ├── res_filter.h       # 殘餘回音抑制器
│   └── fft_wrapper.h      # FFT 封裝
├── src/
│   ├── aec.c              # 主協調器
│   ├── subband_nlms.c     # PBFDAF 實作
│   ├── res_filter.c       # RES 實作
│   └── fft_wrapper.c      # KISS FFT 封裝
├── example/
│   ├── main.c             # CLI 工具
│   └── wav_io.h           # WAV 讀寫
├── lib/kiss_fft/          # FFT 函式庫
└── Makefile
```

## 與 NR 整合

```c
#include "aec.h"
// #include "mmse_lsa.h"  // NR 模組

AecConfig aec_cfg = aec_default_config(16000);
aec_cfg.enable_res = true;
aec_cfg.res_g_min_db = -25.0f;
// aec_cfg.enable_shadow = true;  // 可選：雙濾波器

Aec* aec = aec_create(&aec_cfg);

while (has_audio) {
    aec_process(aec, mic_buf, ref_buf, aec_out);
    // mmse_lsa_process(lsa, aec_out, final_out);
}
```
