# AEC - Acoustic Echo Cancellation

回音消除模組，支援三種濾波器模式，搭配雙講偵測 (DTD) 和殘餘回音抑制 (RES)。

## 濾波器模式

| 模式 | CLI | 演算法 | 延遲 | 適用場景 |
|------|-----|--------|------|----------|
| **time** | `--mode time` | 時域 NLMS | 10ms | 短回音、低延遲需求 |
| **freq** | `--mode freq` | 頻域 NLMS (單一 block) | 16ms | 中等回音、平衡效能 |
| **subband** | `--mode subband` | 分區頻域 NLMS (PBFDAF) | 16ms | 長回音路徑、快速收斂 |

### 演算法原理

#### 1. 時域 NLMS (`--mode time`)
```
y_hat(n) = w^T * x(n)           // 估計回音
e(n) = d(n) - y_hat(n)          // 誤差信號
mu_eff = mu / (||x||^2 + delta) // 正規化步長
w = w + mu_eff * e(n) * x(n)    // 權重更新
```
- 逐樣本處理，延遲最低
- 複雜度: O(N) per sample
- hop_size: 160 samples (10ms @ 16kHz)

#### 2. 頻域 NLMS (`--mode freq`)
```
X = FFT(x_block)                // 頻域轉換
Y_hat = W * X                   // 頻域濾波 (單一 block)
y_hat = IFFT(Y_hat)             // 時域輸出
```
- 單一 FFT block，n_partitions=1
- 複雜度: O(N log N) per block
- hop_size: 256 samples (16ms @ 16kHz)

#### 3. 分區頻域 NLMS (`--mode subband`)
```
Y_hat = sum(W[p] * X_buf[p])    // 多分區卷積
```
- 分區塊 (Partitioned Block FDAF) 處理長回音路徑
- n_partitions = ceil(filter_length / hop_size)
- 收斂更快，適合 200-300ms 回音路徑

## 快速開始

### 編譯 C 版本

```bash
cd c_impl
make
```

### 使用 C 版本

```bash
# 時域模式 (預設)
./bin/aec_wav mic.wav ref.wav output.wav

# 頻域模式
./bin/aec_wav mic.wav ref.wav output.wav --mode freq

# 分區頻域模式 (長回音路徑)
./bin/aec_wav mic.wav ref.wav output.wav --mode subband

# 調整參數
./bin/aec_wav mic.wav ref.wav output.wav --mode subband --mu 0.5 --filter 300
```

### Python 版本

```bash
cd python
pip install numpy soundfile

# 時域模式 (預設)
python aec.py mic.wav ref.wav output.wav

# 頻域模式
python aec.py mic.wav ref.wav output.wav --mode freq

# 分區頻域模式 + RES
python aec.py mic.wav ref.wav output.wav --mode subband --enable-res

# 調整參數
python aec.py mic.wav ref.wav output.wav --mu 0.5 --filter-ms 300
```

## 系統架構

```
Reference Signal (far-end/喇叭播放)
          |
          v
+-------------------+     +---------------------------+
| Microphone Signal |---->| Double-Talk Detector (DTD)|
| (near-end)        |     +---------------------------+
+-------------------+                |
                                     v
                     +---------------------------+
                     | Adaptive Filter           |
                     | (time / freq / subband)   |
                     +---------------------------+
                                     |
                                     v
                     +---------------------------+
                     | Residual Echo Suppressor  |
                     | (RES Post-Filter)         |
                     +---------------------------+
                                     |
                                     v
                               Clean Output
                                     |
                                     v
                     +---------------------------+
                     | LSA v3-2 (Noise Reduction)|
                     +---------------------------+
```

## API 使用

### C 語言

```c
#include "aec.h"

// 建立 AEC (時域模式)
AecConfig config = aec_default_config(16000);
config.filter_mode = AEC_MODE_TIME;  // 或 AEC_MODE_FREQ / AEC_MODE_SUBBAND
config.mu = 0.3f;
config.enable_dtd = true;

Aec* aec = aec_create(&config);

// 注意：hop_size 根據模式不同
int hop_size = aec_get_hop_size(aec);  // time:160, freq/subband:256

while (has_audio) {
    aec_process(aec, mic_buf, ref_buf, out_buf);
    float erle = aec_get_erle(aec);
}

aec_destroy(aec);
```

### Python

```python
from aec import AEC, AecConfig, AecMode

# 時域模式
config = AecConfig(mode=AecMode.TIME, mu=0.3)
aec = AEC(config)

# 頻域模式
config = AecConfig(mode=AecMode.FREQ)
aec = AEC(config)

# 分區頻域模式 + RES
config = AecConfig(
    mode=AecMode.SUBBAND,
    enable_res=True,
    res_g_min_db=-20.0
)
aec = AEC(config)

# 注意：hop_size 根據模式不同
hop_size = aec.hop_size

while has_audio:
    output = aec.process(mic_block, ref_block)
```

## 參數說明

### 濾波器參數

| 參數 | 預設值 | 範圍 | 說明 |
|------|--------|------|------|
| `mu` | 0.3 | 0.1-0.8 | 步長，越大收斂越快但穩態誤差增加 |
| `filter_length_ms` | 250 | 100-500 | 濾波器長度 (ms) |
| `dtd_threshold` | 0.6 | 0.4-0.8 | DTD 閾值 |

### RES 參數 (僅 freq/subband 模式)

| 參數 | 預設值 | 範圍 | 說明 |
|------|--------|------|------|
| `res_g_min_db` | -20 | -30 ~ -10 | 最小增益 (主要強度控制) |
| `res_over_sub` | 1.5 | 1.0-3.0 | 過減因子 |
| `res_alpha` | 0.8 | 0.5-0.95 | 增益平滑係數 |

### 模式選擇指南

| 應用場景 | 推薦模式 | mu | filter_length_ms |
|----------|----------|-----|------------------|
| 手機/耳機 (短回音) | time | 0.3 | 100-150 |
| 會議室 | freq | 0.3 | 200-250 |
| 智慧音箱 (長回音) | subband | 0.2 | 300-400 |
| 即時通訊 (低延遲) | time | 0.3 | 150 |

## 檔案結構

```
AEC/
├── README.md
├── c_impl/
│   ├── include/
│   │   ├── aec.h              # 主要 API
│   │   ├── aec_types.h        # 配置結構 (AecFilterMode)
│   │   ├── nlms_filter.h      # 時域 NLMS
│   │   ├── subband_nlms.h     # 頻域 NLMS
│   │   ├── dtd.h              # 雙講偵測器
│   │   ├── res_filter.h       # 殘餘回音抑制器
│   │   └── fft_wrapper.h      # FFT 介面
│   ├── src/
│   │   ├── aec.c              # 主協調器 (三模式支援)
│   │   ├── nlms_filter.c
│   │   ├── subband_nlms.c
│   │   ├── dtd.c
│   │   ├── res_filter.c
│   │   └── fft_wrapper.c
│   ├── example/
│   │   ├── main.c             # CLI 工具
│   │   ├── aec_lsa_pipeline.c # AEC+LSA 整合範例
│   │   └── wav_io.h
│   ├── lib/kiss_fft/
│   └── Makefile
├── python/
│   └── aec.py                 # Python 實作 (三模式支援)
└── docs/
    └── DEVLOG.md              # 開發紀錄
```

## 與 LSA 整合

```c
// AEC + LSA 處理鏈
AecConfig aec_cfg = aec_default_config(16000);
aec_cfg.filter_mode = AEC_MODE_SUBBAND;
aec_cfg.enable_res = true;
aec_cfg.res_g_min_db = -25.0f;

MmseLsaConfig lsa_cfg = mmse_lsa_default_config(16000);
lsa_cfg.g_min_db = -15.0f;

Aec* aec = aec_create(&aec_cfg);
MmseLsaDenoiser* lsa = mmse_lsa_create(&lsa_cfg);

// 注意：需要處理不同 hop_size
int aec_hop = aec_get_hop_size(aec);  // 256 for subband
int lsa_hop = mmse_lsa_get_hop_size(lsa);  // 可能不同

while (has_audio) {
    aec_process(aec, mic_buf, ref_buf, aec_out);
    mmse_lsa_process(lsa, aec_out, final_out);
}
```

## 效能指標

| 模式 | ERLE | 複雜度 | 收斂時間 | hop_size |
|------|------|--------|----------|----------|
| time | 15-20 dB | O(N) | 0.5-2s | 160 |
| freq | 18-22 dB | O(N log N) | 0.3-1s | 256 |
| subband | 20-25 dB | O(N log N) | 0.2-0.8s | 256 |
| + RES | +10-15 dB | O(K) | - | - |

## 參考文獻

1. Haykin, S. "Adaptive Filter Theory"
2. Benesty, J. et al. "Advances in Network and Acoustic Echo Cancellation"
3. Sondhi, M.M. "An adaptive echo canceller" (1967)
4. Ferrara, E.R. "Fast implementations of LMS adaptive filters" (1980)
