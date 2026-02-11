# AEC - Acoustic Echo Cancellation

回音消除模組，支援時域和頻域自適應濾波器，搭配雙講偵測 (DTD) 和殘餘回音抑制 (RES)。

## 功能特色

- **時域 NLMS**: 基礎自適應濾波器
- **頻域 NLMS (Subband)**: 效能更佳，適合長回音路徑
- **雙講偵測 (DTD)**: 防止近端說話時濾波器發散
- **殘餘回音抑制 (RES)**: Post-filter 額外抑制 10-20 dB

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
                     | NLMS Adaptive Filter      |
                     | (Time-domain / Subband)   |
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

## 快速開始

### 編譯 C 版本

```bash
cd c_impl
make
```

### 使用

```bash
# 基本用法 (時域 NLMS)
./bin/aec_wav mic.wav ref.wav output.wav

# 調整參數
./bin/aec_wav mic.wav ref.wav output.wav --mu 0.5 --filter 300

# 關閉 DTD
./bin/aec_wav mic.wav ref.wav output.wav --no-dtd
```

### Python 版本

```bash
cd python
pip install numpy soundfile

# 時域 NLMS
python aec.py mic.wav ref.wav output.wav

# 頻域 NLMS + RES
python aec.py mic.wav ref.wav output.wav --mode subband --enable-res

# 調整參數
python aec.py mic.wav ref.wav output.wav --mu 0.5 --filter-ms 300
```

## 檔案結構

```
AEC/
├── README.md
├── c_impl/
│   ├── include/
│   │   ├── aec.h              # 主要 API
│   │   ├── aec_types.h        # 配置結構
│   │   ├── nlms_filter.h      # 時域 NLMS
│   │   ├── subband_nlms.h     # 頻域 NLMS
│   │   ├── dtd.h              # 雙講偵測器
│   │   ├── res_filter.h       # 殘餘回音抑制器
│   │   └── fft_wrapper.h      # FFT 介面
│   ├── src/
│   │   ├── aec.c
│   │   ├── nlms_filter.c
│   │   ├── subband_nlms.c
│   │   ├── dtd.c
│   │   ├── res_filter.c
│   │   └── fft_wrapper.c
│   ├── example/
│   │   ├── main.c             # 命令列工具
│   │   ├── aec_lsa_pipeline.c # AEC+LSA 整合範例
│   │   └── wav_io.h
│   ├── lib/kiss_fft/
│   └── Makefile
├── python/
│   └── aec.py                 # Python 參考實作 (含 Subband + RES)
└── docs/
    └── DEVLOG.md              # 開發紀錄
```

## API 使用

### C 語言

```c
#include "aec.h"

// 建立 AEC
AecConfig config = aec_default_config(16000);
config.mu = 0.3f;
config.filter_length_ms = 250;
config.enable_dtd = true;
config.enable_res = true;      // 啟用 RES
config.res_g_min_db = -20.0f;  // RES 最小增益

Aec* aec = aec_create(&config);

// 處理音訊
int hop_size = aec_get_hop_size(aec);
while (has_audio) {
    aec_process(aec, mic_buf, ref_buf, out_buf);
    float erle = aec_get_erle(aec);
}

aec_destroy(aec);
```

### Python

```python
from aec import AEC, AecConfig, AecMode

# 時域 NLMS
config = AecConfig(mu=0.3, filter_length_ms=250)
aec = AEC(config)

# 頻域 NLMS + RES
config = AecConfig(
    mode=AecMode.SUBBAND,
    enable_res=True,
    res_g_min_db=-20.0
)
aec = AEC(config)

while has_audio:
    output = aec.process(mic_block, ref_block)
```

## 參數說明

### AEC 參數

| 參數 | 預設值 | 範圍 | 說明 |
|------|--------|------|------|
| `mu` | 0.3 | 0.1-0.8 | 步長，越大收斂越快 |
| `filter_length_ms` | 250 | 100-500 | 濾波器長度 (ms) |
| `dtd_threshold` | 0.6 | 0.4-0.8 | DTD 閾值 |

### RES 參數

| 參數 | 預設值 | 範圍 | 說明 |
|------|--------|------|------|
| `res_g_min_db` | -20 | -30 ~ -10 | 最小增益 (主要強度控制) |
| `res_over_sub` | 1.5 | 1.0-3.0 | 過減因子 |
| `res_alpha` | 0.8 | 0.5-0.95 | 增益平滑係數 |

### 建議配置

| 應用場景 | mu | filter_length_ms | mode |
|----------|-----|------------------|------|
| 手機/耳機 | 0.3 | 150 | time |
| 會議室 | 0.2 | 250 | subband |
| 智慧音箱 | 0.2 | 400 | subband + RES |

## 與 LSA 整合

```c
// AEC + LSA 處理鏈
AecConfig aec_cfg = aec_default_config(16000);
aec_cfg.enable_res = true;
aec_cfg.res_g_min_db = -25.0f;  // 較強的回音抑制

MmseLsaConfig lsa_cfg = mmse_lsa_default_config(16000);
lsa_cfg.g_min_db = -15.0f;      // 較強的降噪

Aec* aec = aec_create(&aec_cfg);
MmseLsaDenoiser* lsa = mmse_lsa_create(&lsa_cfg);

while (has_audio) {
    aec_process(aec, mic_buf, ref_buf, aec_out);
    mmse_lsa_process(lsa, aec_out, final_out);
}
```

## 效能指標

| 方法 | ERLE | 複雜度 | 收斂時間 |
|------|------|--------|----------|
| Time-domain NLMS | 15-20 dB | O(N) | 0.5-2s |
| Subband NLMS | 18-25 dB | O(N log N) | 0.3-1s |
| + RES | +10-15 dB | O(K) | - |

## 參考文獻

1. Haykin, S. "Adaptive Filter Theory"
2. Benesty, J. et al. "Advances in Network and Acoustic Echo Cancellation"
3. Sondhi, M.M. "An adaptive echo canceller" (1967)
