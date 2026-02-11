# AEC - Acoustic Echo Cancellation

回音消除模組，使用 NLMS 自適應濾波器搭配雙講偵測 (DTD)。

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
                     +---------------------------+
                                     |
                                     v
                     +---------------------------+
                     | (Future: RES Post-Filter) |
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
# 基本用法
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

# 基本用法
python aec.py mic.wav ref.wav output.wav

# 調整參數
python aec.py mic.wav ref.wav output.wav --mu 0.5 --filter-ms 300
```

## 檔案結構

```
AEC/
├── README.md           # 本文件
├── c_impl/
│   ├── include/
│   │   ├── aec.h           # 主要 API
│   │   ├── aec_types.h     # 配置結構
│   │   ├── nlms_filter.h   # NLMS 濾波器
│   │   ├── dtd.h           # 雙講偵測器
│   │   └── fft_wrapper.h   # FFT 介面
│   ├── src/
│   │   ├── aec.c
│   │   ├── nlms_filter.c
│   │   ├── dtd.c
│   │   └── fft_wrapper.c
│   ├── example/
│   │   ├── main.c          # 命令列工具
│   │   └── wav_io.h        # WAV 讀寫
│   ├── lib/
│   │   └── kiss_fft/       # FFT 函式庫
│   └── Makefile
├── python/
│   └── aec.py              # Python 參考實作
└── docs/
    └── DEVLOG.md           # 開發紀錄
```

## API 使用

### C 語言

```c
#include "aec.h"

// 建立 AEC
AecConfig config = aec_default_config(16000);  // 16kHz
config.mu = 0.3f;                              // 步長
config.filter_length_ms = 250;                 // 濾波器長度
config.enable_dtd = true;                      // 啟用 DTD

Aec* aec = aec_create(&config);

// 處理音訊 (每次 hop_size 個樣本)
int hop_size = aec_get_hop_size(aec);
float mic_buf[hop_size], ref_buf[hop_size], out_buf[hop_size];

while (has_audio) {
    // 讀取輸入...
    aec_process(aec, mic_buf, ref_buf, out_buf);
    // 寫入輸出...

    // 查詢狀態
    float erle = aec_get_erle(aec);
    bool dtd = aec_is_dtd_active(aec);
}

// 釋放資源
aec_destroy(aec);
```

### Python

```python
from aec import AEC, AecConfig

# 建立 AEC
config = AecConfig(
    sample_rate=16000,
    mu=0.3,
    filter_length_ms=250,
    enable_dtd=True
)
aec = AEC(config)

# 處理音訊
hop_size = aec.hop_size
while has_audio:
    output = aec.process(mic_block, ref_block)

    # 查詢狀態
    erle = aec.get_erle()
    dtd = aec.is_dtd_active()
```

## 參數說明

### 強度調整

| 參數 | 預設值 | 說明 |
|------|--------|------|
| `mu` | 0.3 | 步長，越大收斂越快但穩態誤差增加 |
| `filter_length_ms` | 250 | 濾波器長度，需大於回音路徑 |
| `dtd_threshold` | 0.6 | DTD 閾值，越低越容易判定為雙講 |

### 建議配置

| 應用場景 | mu | filter_length_ms |
|----------|-----|------------------|
| 手機/耳機 | 0.3 | 150 |
| 會議室 | 0.2 | 250 |
| 智慧音箱 | 0.2 | 400 |

## 與 LSA 整合

```c
// AEC + LSA 處理鏈
AecConfig aec_cfg = aec_default_config(16000);
MmseLsaConfig lsa_cfg = mmse_lsa_default_config(16000);

Aec* aec = aec_create(&aec_cfg);
MmseLsaDenoiser* lsa = mmse_lsa_create(&lsa_cfg);

while (has_audio) {
    // 1. AEC: 消除回音
    aec_process(aec, mic_buf, ref_buf, aec_out);

    // 2. LSA: 降噪
    mmse_lsa_process(lsa, aec_out, final_out);
}

aec_destroy(aec);
mmse_lsa_destroy(lsa);
```

## 效能指標

- **ERLE (Echo Return Loss Enhancement)**: 回音消除量，典型 15-25 dB
- **收斂時間**: 取決於 mu 和回音路徑，典型 0.5-2 秒
- **計算量**: O(N) per sample，N = filter_length

## 已知限制

1. 目前僅支援 16kHz 取樣率
2. 時域 NLMS 對長回音路徑計算量較大
3. 尚未實作 Residual Echo Suppressor (RES)

## 後續計畫

- [ ] Subband NLMS (頻域實作)
- [ ] Residual Echo Suppressor
- [ ] 多取樣率支援
- [ ] SIMD 優化

## 參考文獻

1. Haykin, S. "Adaptive Filter Theory"
2. Benesty, J. et al. "Advances in Network and Acoustic Echo Cancellation"
