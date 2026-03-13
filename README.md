# AEC - Acoustic Echo Cancellation

回音消除模組，Python 支援四種濾波器模式，搭配發散偵測 (DTD) 和殘餘回音抑制 (RES)。

> C 實作僅支援 PBFDAF 模式，詳見 [c_impl/README.md](c_impl/README.md)。

## 濾波器模式

| 模式 | CLI | 演算法 | 延遲 | 適用場景 |
|------|-----|--------|------|----------|
| **nlms** | `--mode nlms` | 時域 NLMS | 16ms | 一般用途（預設） |
| **freq** | `--mode freq` | 頻域 NLMS (單一 block) | 16ms | 中等回音、平衡效能 |
| **subband** | `--mode subband` | 分區頻域 NLMS (PBFDAF) | 16ms | 長回音路徑、快速收斂 |
| **lms** | `--mode lms` | 時域 LMS (固定步長) | 16ms | 穩態環境、極低資源 |

## 快速開始

### Python

```bash
cd python
pip install numpy soundfile

# 時域 NLMS 模式 (預設，DTD 開啟，RES 關閉)
python3 aec.py mic.wav ref.wav output.wav

# 時域 LMS 模式 (固定步長)
python3 aec.py mic.wav ref.wav output.wav --mode lms

# 頻域模式
python3 aec.py mic.wav ref.wav output.wav --mode freq

# 分區頻域模式 + RES
python3 aec.py mic.wav ref.wav output.wav --mode subband --enable-res

# 分區頻域模式 + Shadow Filter (雙濾波器)
python3 aec.py mic.wav ref.wav output.wav --mode subband --enable-shadow

# 調整參數
python3 aec.py mic.wav ref.wav output.wav --mu 0.5 --filter 1024
```

### C (PBFDAF only)

```bash
cd c_impl && make
./bin/aec_wav mic.wav ref.wav output.wav                    # DTD + RES 預設開啟
./bin/aec_wav mic.wav ref.wav output.wav --no-res            # 關閉 RES
./bin/aec_wav mic.wav ref.wav output.wav --enable-shadow     # 開啟 Shadow Filter
```

詳見 [c_impl/README.md](c_impl/README.md)。

## 系統架構

```
Reference Signal (far-end/喇叭播放)
          |
          v
+-------------------+     +---------------------------+
| Microphone Signal |---->| Adaptive Filter (Main)    |
| (near-end)        |     | (lms/nlms/freq/subband)   |
+-------------------+     +---------------------------+
                                     |
                          +----------+----------+
                          |                     |
                          v                     v
               +------------------+  +------------------------+
               | Shadow Filter    |  | DTD (Dual Detector)    |
               | (optional, 可選) |  |  Divergence: output>in |
               +------------------+  |  Coherence: MSC(e,x)   |
                          |          +------------------------+
                          |                     |
                          |           Coherence 主導, Divergence fallback
                          |           mu_scale = 1 - conf×0.95
                          +----------+----------+
                                     |
                                     v
                     +---------------------------+
                     | RES Post-Filter (optional) |
                     +---------------------------+
                                     |
                                     v
                     +---------------------------+
                     | Output Limiter            |
                     | (output ≤ mic amplitude)  |
                     +---------------------------+
                                     |
                                     v
                               Clean Output
```

## API 使用

```python
from aec import AEC, AecConfig, AecMode

# 時域 NLMS 模式（預設 DTD 開啟，RES 關閉）
config = AecConfig(mode=AecMode.NLMS, mu=0.3)
aec = AEC(config)

# 分區頻域模式 + RES
config = AecConfig(
    mode=AecMode.SUBBAND,
    enable_res=True,
    res_g_min_db=-20.0
)
aec = AEC(config)

# 分區頻域模式 + Shadow Filter (雙濾波器自動修正)
config = AecConfig(
    mode=AecMode.SUBBAND,
    enable_shadow=True
)
aec = AEC(config)

# 處理
hop_size = aec.hop_size  # 256 samples (16ms @ 16kHz)
while has_audio:
    output = aec.process(mic_block, ref_block)
    erle = aec.get_erle()
    conf = aec.get_dtd_confidence()
```

## 參數說明

### 濾波器參數

| 參數 | 預設值 | 範圍 | 說明 |
|------|--------|------|------|
| `mu` | 0.3 (NLMS) / 0.01 (LMS) | NLMS: 0.1-0.8, LMS: 0.001-0.05 | 步長 |
| `filter_length` | 512 (NLMS/LMS), 1024 (SUBBAND) | 256-4096 | 濾波器長度 (samples) |
| `enable_dtd` | True | - | DTD：僅 FREQ/SUBBAND（Divergence + Coherence），詳見 [docs/dtd_design.md](docs/dtd_design.md) |
| `enable_res` | False (Python) / True (C) | - | 殘餘回音抑制 |
| `enable_shadow` | False | - | Shadow filter（僅 freq/subband，需要 DTD 開啟，見下方說明） |

### Shadow Filter 參數 (僅 freq/subband 模式，需要 DTD)

Shadow filter（雙濾波器）使用一個保守步長的影子濾波器持續追蹤回音路徑，當主濾波器發散時自動修正。
這是 WebRTC AEC3 和 SpeexDSP 的核心機制。

> **注意**：Shadow 依賴 DTD 保護。若同時使用 `--no-dtd --enable-shadow`，shadow 會自動停用並印出警告。

| 參數 | 預設值 | 範圍 | 說明 |
|------|--------|------|------|
| `shadow_mu_ratio` | 0.5 | 0.1-0.8 | Shadow mu = main mu × ratio |
| `shadow_copy_threshold` | 0.8 | 0.5-1.0 | Shadow error < main error × threshold 時複製權重 |
| `shadow_err_alpha` | 0.95 | 0.9-0.99 | Error energy EMA 平滑係數 |
| `shadow_dtd_mu_min` | 0.2 | 0.1-0.5 | Shadow DTD mu 最低比例（比 main 的 5% 更寬鬆） |
| `shadow_copy_hysteresis` | 3 | 1-10 | 連續 N frames 符合條件才觸發 copy |

**與 DTD 的關係**：兩者互補，可同時啟用。DTD 降低 mu 防止惡化，shadow 在背景持續追蹤並自動修正。

### RES 參數 (僅 freq/subband 模式)

| 參數 | 預設值 | 範圍 | 說明 |
|------|--------|------|------|
| `res_g_min_db` | -20 | -30 ~ -10 | 最小增益 (主要強度控制) |
| `res_over_sub` | 1.5 | 1.0-3.0 | 過減因子 |
| `res_alpha` | 0.8 | 0.5-0.95 | 增益平滑係數 |

### 模式選擇指南

| 應用場景 | 推薦模式 | mu | filter_length |
|----------|----------|-----|---------------|
| 手機/耳機 (短回音) | nlms | 0.3 | 512 |
| 會議室 (中等回音) | nlms | 0.3 | 1024 |
| 智慧音箱 (長回音) | subband | 0.2 | 1024-4096 |
| 穩態環境 (極低資源) | lms | 0.01 | 512 |

### 繪圖工具

```bash
# 繪製 AEC 結果（DTD 紅底標示）
python3 plot_aec_results.py ../wav/ --mode subband

# 開啟 RES
python3 plot_aec_results.py ../wav/ --mode subband --enable-res

# 開啟 Shadow Filter
python3 plot_aec_results.py ../wav/ --mode subband --enable-shadow
```

## 效能指標

| 模式 | ERLE | 複雜度 | 收斂時間 |
|------|------|--------|----------|
| lms | 10-15 dB | O(N) | 1-5s |
| nlms | 15-20 dB | O(N) | 0.5-2s |
| freq | 18-22 dB | O(N log N) | 0.3-1s |
| subband | 20-25 dB | O(N log N) | 0.2-0.8s |
| + RES | +10-15 dB | O(K) | - |

## 檔案結構

```
AEC/
├── README.md                  # 本文件（Python 為主）
├── c_impl/                    # C 實作 (PBFDAF only)
│   ├── README.md              # C 版本文檔
│   ├── include/               # 標頭檔
│   ├── src/                   # 原始碼
│   ├── example/               # CLI 工具
│   └── Makefile
├── python/
│   ├── aec.py                 # Python 實作 (四模式)
│   ├── plot_aec_results.py    # 結果繪圖 (含 DTD 紅底)
│   └── gen_sim_data.py        # 測試資料生成
├── docs/
│   ├── aec_methods.md         # 演算法文檔
│   ├── dtd_design.md          # DTD 設計文檔（完整說明）
│   └── DEVLOG.md              # 開發紀錄
└── wav/                       # 測試音檔
```

## 參考文獻

1. Haykin, S. "Adaptive Filter Theory"
2. Benesty, J. et al. "Advances in Network and Acoustic Echo Cancellation"
3. Sondhi, M.M. "An adaptive echo canceller" (1967)
4. Ferrara, E.R. "Fast implementations of LMS adaptive filters" (1980)
