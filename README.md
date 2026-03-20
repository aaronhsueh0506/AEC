# AEC - Acoustic Echo Cancellation

回音消除模組，Python 支援四種濾波器模式，搭配 Shadow Filter（預設）和殘餘回音抑制 (RES)。

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

# 推薦：分區頻域模式 + Shadow + RES（預設 Shadow 開啟、DTD 關閉）
python3 aec.py mic.wav ref.wav output.wav --mode subband --enable-res

# 啟用 DTD（進階，特定場景）
python3 aec.py mic.wav ref.wav output.wav --mode subband --enable-res --enable-dtd

# 純 PBFDAF（debug 用，關閉 Shadow）
python3 aec.py mic.wav ref.wav output.wav --mode subband --enable-res --no-shadow

# 時域 NLMS 模式
python3 aec.py mic.wav ref.wav output.wav

# 調整參數
python3 aec.py mic.wav ref.wav output.wav --mu 0.5 --filter 1024
```

### C (PBFDAF only)

```bash
cd c_impl && make
./bin/aec_wav mic.wav ref.wav output.wav                    # DTD + RES 預設開啟（C 版本仍用 DTD）
./bin/aec_wav mic.wav ref.wav output.wav --no-res            # 關閉 RES
./bin/aec_wav mic.wav ref.wav output.wav --enable-shadow     # 開啟 Shadow Filter（C 版本需手動開啟）
```

詳見 [c_impl/README.md](c_impl/README.md)。

## 系統架構

```
Reference Signal (far-end)        Microphone Signal (near-end)
          |                                  |
          v                                  |
+---------------------+                     |
| Delay Estimation    |                     |
| (GCC-PHAT / fixed)  |                     |
+---------------------+                     |
          |                                  |
          v                                  |
+---------------------+                     |
| Reference Alignment |                     |
| (ring buffer delay) |                     |
+---------------------+                     |
          |                                  |
          +----------------------------------+
                         |
                         v
              +---------------------------+
              | PBFDAF (Main Filter)      |
              | Adaptation: FDKF or NLMS  |
              +---------------------------+
                         |
              +----------+----------+
              |                     |
              v                     v
   +------------------+  +------------------------+
   | Shadow Filter    |  | DTD (可選)             |
   | (預設開啟)       |  |  Divergence + Coherence|
   +------------------+  +------------------------+
              |
      copy gate: far_active + not_dt
              |
              v
         +---------------------------+
         | RES Post-Filter           |
         | (coherence-based EER,     |
         |  spectral subtraction)    |
         +---------------------------+
                         |
                         v
         +---------------------------+
         | Output Limiter            |
         +---------------------------+
                         |
                         v
                   Clean Output
```

## API 使用

```python
from aec import AEC, AecConfig, AecMode

# 預設：subband + Shadow Filter（DTD 關閉）
config = AecConfig(mode=AecMode.SUBBAND)
aec = AEC(config)

# subband + Shadow + RES（推薦）
config = AecConfig(
    mode=AecMode.SUBBAND,
    enable_res=True,
    res_g_min_db=-20.0
)
aec = AEC(config)

# subband + DTD + Shadow（雙重保護，進階）
config = AecConfig(
    mode=AecMode.SUBBAND,
    enable_dtd=True
)
aec = AEC(config)

# 時域 NLMS 模式（無 Shadow/DTD）
config = AecConfig(mode=AecMode.NLMS, mu=0.3)
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
| `filter_length` | 512 (NLMS/LMS), 1024 (FREQ/SUBBAND) | 256-4096 | 濾波器長度 (samples) |
| `enable_dtd` | False | - | DTD：僅 FREQ/SUBBAND（Divergence + Coherence），詳見 [docs/dtd_design.md](docs/dtd_design.md) |
| `enable_res` | False (Python) / True (C) | - | 殘餘回音抑制 |
| `use_kalman` | False | - | FDKF adaptation（per-bin Kalman gain 取代 NLMS） |
| `enable_delay_est` | True | - | 自動延遲估計（GCC-PHAT） |
| `max_delay_ms` | 250.0 | 50-500 | 延遲搜尋範圍上限 (ms) |
| `fixed_delay_samples` | -1 | -1 or ≥0 | 若 ≥0，使用固定延遲（跳過估計） |
| `enable_shadow` | True | - | Shadow filter（僅 freq/subband，預設開啟，見下方說明） |

### Shadow Filter 參數 (僅 freq/subband 模式，預設開啟)

Shadow filter（雙濾波器）使用 full mu 的背景濾波器持續追蹤回音路徑，當主濾波器發散時透過 copy gate 自動修正。
這是 WebRTC AEC3 和 SpeexDSP 的核心機制，v1.6.0 起為預設保護策略。

> Shadow 預設單獨使用（≈ WebRTC/SpeexDSP 做法），可加 `--enable-dtd` 啟用雙重保護。
> 詳見 [docs/aec_methods.md §6.4](docs/aec_methods.md) 的組合行為表。

| 參數 | 預設值 | 範圍 | 說明 |
|------|--------|------|------|
| `shadow_copy_threshold` | 0.5 | 0.3-1.0 | Shadow error < main error × threshold 時複製權重 |
| `shadow_err_alpha` | 0.95 | 0.9-0.99 | Error energy EMA 平滑係數 |
| `shadow_mu_min` | 0.5 | 0.1-0.8 | 主濾波器 DT 時 mu 下限（shadow-only 模式的輕量保護） |
| `shadow_copy_hysteresis` | 3 | 1-10 | 連續 N frames 符合條件才觸發 copy |

**設計要點**：
- Shadow 使用 full mu（1.0），不受 DT 影響，持續追蹤回音路徑
- Copy gate 條件：`far_active AND not_dt AND shadow_err < main_err × 0.5`（連續 3 frames）
- Copy 只複製 weights，不切換 output（避免 output 不連續）
- 50-frame warm-up guard：收斂前不允許 copy（避免未收斂時的誤判退化）
- 可與 DTD 互補：DTD 降低 mu 防止惡化，shadow 在背景持續追蹤並自動修正

### RES 參數 (僅 freq/subband 模式)

RES 使用 OLA + sqrt-Hann 窗框架避免 frame 邊界不連續和棋盤頻譜（musical noise）。
增益公式為 spectral subtraction Wiener gain：`G = max(1 - over_sub × EER, g_min)`，DT 時 error 含近端語音 → EER 自然降低 → 壓制自動減少。

| 參數 | 預設值 | 範圍 | 說明 |
|------|--------|------|------|
| `res_g_min_db` | -40 | -40 ~ -10 | 最小增益 (主要強度控制) |
| `res_over_sub` | 10.0 | 1.0-15.0 | 過減因子（越高壓制越強） |
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
# 繪製 AEC 結果（預設 Shadow 開啟）
python3 plot_aec_results.py ../wav/ --mode subband

# 開啟 RES
python3 plot_aec_results.py ../wav/ --mode subband --enable-res

# 加入 DTD（雙重保護）
python3 plot_aec_results.py ../wav/ --mode subband --enable-dtd
```

## Benchmark 比較（AEC Challenge）

### Farend Single-Talk（10 cases）— ERLE

測試條件：subband + Shadow + FDKF + RES + delay pre-alignment（v1.9.0）。
對照：SpeexDSP (FL=2048)、WebRTC AEC3。
指標：`ERLE = 10·log10(mean(mic²) / mean(output²))`，全段平均。

| 指標 | Ours | SpeexDSP | WebRTC AEC3 |
|------|------|----------|-------------|
| Mean ERLE | **14.3 dB** | 7.3 dB | 18.2 dB |

### Doubletalk（10 synthetic cases, SER=-5dB）— PESQ + ERLE

| 指標 | Ours | SpeexDSP | WebRTC AEC3 |
|------|------|----------|-------------|
| Mean ERLE | **9.4 dB** | 4.3 dB | 8.7 dB |
| Mean PESQ | **1.24** | 1.41 | 1.09 |

> **解讀**：
> - Farend ERLE 14.3 dB，較 v1.8.4 的 5.1 dB 提升 +9.2 dB，接近 AEC3（18.2 dB）。
> - Doubletalk ERLE 9.4 dB 已超越 AEC3（8.7 dB），PESQ 1.24 亦優於 AEC3（1.09）。
> - 主要改善來源：delay pre-alignment（+8dB）、RES over-subtraction 修正（+6dB）、FDKF（+1dB）。
> - 詳見 [docs/aec_improve_v3.md](docs/aec_improve_v3.md)。

### 工具

```bash
# 執行 AEC Challenge benchmark（需 speexdsp Python binding + WebRTC AEC3 CLI）
cd AEC
python3 python/eval_aec_challenge.py wav/aec_challenge/ --aec3 --speex
```

## 效能指標

| 模式 | ERLE | 複雜度 | 收斂時間 |
|------|------|--------|----------|
| lms | 10-15 dB | O(N) | 1-5s |
| nlms | 15-20 dB | O(N) | 0.5-2s |
| freq | 18-22 dB | O(N log N) | 0.3-1s |
| subband | 20-25 dB | O(N log N) | 0.2-0.8s |
| + RES | +2-4 dB | O(K) | - |

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
│   ├── aec.py                 # Python 實作 (四模式 + FDKF + Delay Est.)
│   ├── eval_aec_challenge.py  # AEC Challenge benchmark 評測
│   ├── plot_aec_results.py    # 結果繪圖 (含 DTD 紅底)
│   └── gen_sim_data.py        # 測試資料生成
├── docs/
│   ├── aec_methods.md         # 演算法文檔
│   ├── aec_improve_v3.md      # v1.9.0 改善紀錄（Delay+FDKF+RES）
│   ├── dtd_design.md          # DTD 設計文檔（完整說明）
│   └── DEVLOG.md              # 開發紀錄
└── wav/                       # 測試音檔
```

## 參考文獻

1. Haykin, S. "Adaptive Filter Theory"
2. Benesty, J. et al. "Advances in Network and Acoustic Echo Cancellation"
3. Sondhi, M.M. "An adaptive echo canceller" (1967)
4. Ferrara, E.R. "Fast implementations of LMS adaptive filters" (1980)
