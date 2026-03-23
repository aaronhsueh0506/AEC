# AEC C Implementation — PBFDKF + Shadow + WOLA RES

C 語言 AEC 實作，對齊 Python AEC v1.15.0（SUBBAND PBFDKF 模式）。

## 特點

- **PBFDKF**（Partitioned Block Frequency-Domain Kalman Filter）— Kalman 永遠開啟
- **Shadow Filter** — 雙濾波器自動修正，永遠啟用
- **WOLA RES** — Weighted Overlap-Add 殘餘回音抑制（sqrt-Hann 窗，frame=320, FFT=512）
- **HPF** — 2 階 Butterworth IIR 高通濾波器（80 Hz）
- **三階段 Preset** — BALANCED / AGGRESSIVE / MAXIMUM
- **Output Limiter** — 平滑增益限制（輸出不超過麥克風振幅）

## 編譯

```bash
make          # 編譯 CLI 工具 → bin/aec_wav
make lib      # 編譯靜態函式庫 → bin/libaec.a
make debug    # 附 debug symbols
make clean    # 清除編譯產物
```

## CLI 使用

```bash
# 基本用法（BALANCED preset，RES + HPF 預設開啟）
./bin/aec_wav mic.wav ref.wav output.wav

# 選擇 preset
./bin/aec_wav mic.wav ref.wav output.wav --preset aggressive
./bin/aec_wav mic.wav ref.wav output.wav --preset maximum

# 關閉 RES / HPF
./bin/aec_wav mic.wav ref.wav output.wav --no-res
./bin/aec_wav mic.wav ref.wav output.wav --no-hpf

# 調整濾波器長度（預設 1600 samples = 100ms @ 16kHz）
./bin/aec_wav mic.wav ref.wav output.wav --filter 2400
```

### CLI 參數

| 參數 | 預設值 | 說明 |
|------|--------|------|
| `--preset <name>` | balanced | 預設模式：balanced / aggressive / maximum |
| `--no-res` | - | 關閉殘餘回音抑制 |
| `--no-hpf` | - | 關閉高通濾波器 |
| `--filter <samples>` | 1600 | 濾波器長度 (samples) |

### Preset 參數差異

| 參數 | BALANCED | AGGRESSIVE | MAXIMUM |
|------|----------|------------|---------|
| `res_g_min_db` | -25 | -35 | -40 |
| `res_over_sub_base` | 2.5 | 4.0 | 6.0 |
| `res_over_sub_scale` | 4.0 | 6.0 | 8.0 |
| `res_dt_reduction` | 3.5 | 2.0 | 0.0 |
| `res_ne_protect_db` | -8 | -5 | -2 |
| `res_spectral_floor_db` | -25 | -30 | -35 |
| `shadow_q_ratio` | 3.0 | 4.0 | 5.0 |
| `shadow_mu_min` | 0.5 | 0.7 | 0.9 |
| `kalman_q_high` | 1e-3 | 3e-3 | 1e-2 |
| `warmup_frames` | 100 | 150 | 200 |

## API

```c
#include "aec.h"

// 從 preset 建立配置
AecConfig cfg = aec_config_from_preset(AEC_PRESET_BALANCED, 16000);
// 或自訂參數：
// cfg.filter_length = 2400;
// cfg.enable_res = 0;

Aec* aec = aec_create(&cfg);
int hop = aec_get_hop_size(aec);  // 160 @ 16kHz

while (has_audio) {
    aec_process(aec, mic_buf, ref_buf, out_buf);  // 每次處理 hop 個 samples

    float erle = aec_get_erle(aec);           // 平滑 ERLE (dB)
    int converged = aec_is_converged(aec);    // 收斂狀態
}

aec_destroy(aec);
```

### 主要 API 函式

| 函式 | 說明 |
|------|------|
| `aec_create(config)` | 建立 AEC 實例 |
| `aec_destroy(aec)` | 釋放資源 |
| `aec_reset(aec)` | 完全重置所有狀態 |
| `aec_process(aec, near, far, out)` | 處理一個 hop（160 samples） |
| `aec_get_hop_size(aec)` | 取得 hop size |
| `aec_get_erle(aec)` | 取得平滑 ERLE (dB) |
| `aec_get_erle_instant(aec)` | 取得瞬時 ERLE (dB) |
| `aec_is_converged(aec)` | 取得收斂狀態 |
| `aec_get_config(aec)` | 取得目前配置 |

## 處理流程

```
                    ┌───────────────────────────────────┐
far-end ──> HPF ──> │  PBFDKF (Main)    PBFDKF (Shadow) │
                    │  mu_scale=var     mu_scale=1.0     │
near-end ─> HPF ──> │       ↓ copy gate ↕                │
                    └──────────┬────────────────────────┘
                               ↓
                    Simple Variable Mu (Valin 2007 RER)
                               ↓
                    Convergence Detection (ERLE > 6dB × 10 frames)
                               ↓
                    WOLA RES (Residual Echo Suppressor)
                      - Coherence-based near-end gate
                      - Dynamic over-subtraction
                      - Spectral floor + rate limiting + CNG
                               ↓
                    Output Limiter → Noise Gate → output
```

### 核心演算法

1. **PBFDKF**：分區頻域 Kalman 濾波器，overlap-save 結構（hop=160, FFT=512）
2. **Shadow Filter**：保守 Q（main Q × ratio），永遠全速更新，bidirectional copy gate
3. **Simple Variable Mu**：取代 DTD，RER-based（far_power/error_power ratio），asymmetric EMA + holdoff
4. **WOLA RES**：sqrt-Hann 窗 WOLA 分析合成（frame=320），coherence 近端保護，per-bin 增益控制
5. **HPF**：2 階 Butterworth IIR，80 Hz 截止，消除 DC offset 與低頻干擾
6. **Output Limiter**：平滑增益限制，確保輸出不超過麥克風振幅

## 檔案結構

```
c_impl/
├── include/
│   ├── aec.h              # 主要 API
│   ├── aec_types.h        # 配置結構、Preset、Factory functions
│   ├── pbfdkf.h           # PBFDKF 介面（Kalman 自適應濾波器）
│   ├── res_filter.h       # WOLA 殘餘回音抑制器
│   ├── hpf.h              # 高通濾波器
│   └── fft_wrapper.h      # FFT 封裝（kiss_fft）
├── src/
│   ├── aec.c              # 主協調器（Shadow + mu + convergence + limiter）
│   ├── pbfdkf.c           # PBFDKF 實作（FDKF Kalman）
│   ├── res_filter.c       # WOLA RES 實作
│   ├── hpf.c              # Butterworth IIR HPF
│   └── fft_wrapper.c      # KISS FFT 封裝
├── example/
│   ├── main.c             # CLI 工具
│   └── wav_io.h           # WAV 讀寫（header-only）
├── lib/kiss_fft/          # KISS FFT 函式庫
└── Makefile
```

## 與 NR 整合

```c
#include "aec.h"
// #include "mmse_lsa.h"  // NR 模組

AecConfig cfg = aec_config_from_preset(AEC_PRESET_BALANCED, 16000);
Aec* aec = aec_create(&cfg);

while (has_audio) {
    aec_process(aec, mic_buf, ref_buf, aec_out);
    // nr_process(nr, aec_out, final_out);  // AEC → NR pipeline
}

aec_destroy(aec);
```

## 注意事項

- C 版本不包含 Delay Estimation，預設濾波器長度為 1600 samples（100ms）以應對未對齊的延遲
- 輸入格式：16-bit mono WAV，16 kHz
- Shadow filter 永遠啟用，無需手動開關
- Kalman 永遠開啟，無 NLMS fallback
