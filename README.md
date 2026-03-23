# AEC - Acoustic Echo Cancellation

回音消除模組（v1.15.0），Python 支援四種濾波器模式，搭配 FDKF（頻域卡爾曼濾波器）、Shadow Filter 和殘餘回音抑制 (RES)。支援三級 Preset（BALANCED / AGGRESSIVE / MAXIMUM）控制 echo 壓制強度，搭配 per-bin near-end gate 精確 DT 保護。

> C 實作僅支援 PBFDAF 模式，詳見 [c_impl/README.md](c_impl/README.md)。

## 濾波器模式

| 模式 | CLI | 演算法 | 延遲 | 適用場景 |
|------|-----|--------|------|----------|
| **nlms** | `--mode nlms` | 時域 NLMS | 10ms | 一般用途（預設） |
| **fdaf** | `--mode fdaf` | 頻域 NLMS (單一 block) | 10ms | 中等回音、平衡效能 |
| **subband** | `--mode subband` | 分區頻域 PBFDAF + FDKF | 10ms | 長回音路徑、最佳品質 |
| **lms** | `--mode lms` | 時域 LMS (固定步長) | 10ms | 穩態環境、極低資源 |

## 快速開始

### Python

```bash
cd python
pip install numpy soundfile

# 推薦：PBFDAF + FDKF + Shadow + RES（最佳品質配置，FDKF 預設開啟）
python3 aec.py mic.wav ref.wav output.wav --mode subband --enable-res

# 使用 Preset（控制 echo 壓制強度）
python3 aec.py mic.wav ref.wav output.wav --mode subband --enable-res --preset balanced
python3 aec.py mic.wav ref.wav output.wav --mode subband --enable-res --preset aggressive
python3 aec.py mic.wav ref.wav output.wav --mode subband --enable-res --preset maximum

# 無 FDKF（標準 NLMS adaptation，需在 API 中設定 use_kalman=False）
python3 aec.py mic.wav ref.wav output.wav --mode subband --enable-res

# 啟用 DTD（進階，特定場景）
python3 aec.py mic.wav ref.wav output.wav --mode subband --enable-res --enable-dtd

# FDAF 模式（單一 FFT block，中等回音）
python3 aec.py mic.wav ref.wav output.wav --mode fdaf --enable-res

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
          v                                  v
+---------------------+           +---------------------+
| High-Pass Filter    |           | High-Pass Filter    |
| (2nd-order Butter.) |           | (2nd-order Butter.) |
+---------------------+           +---------------------+
          |                                  |
          v                                  |
+---------------------+                     |
| Saturation Detect   |                     |
| + Soft-Clip Ref     |                     |
+---------------------+                     |
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
              |  Two-stage Q control      |
              +---------------------------+
                         |
              +----------+----------+
              |                     |
              v                     v
   +------------------+  +------------------------+
   | Shadow Filter    |  | DTD (可選)             |
   | (預設開啟)       |  |  Divergence + Coherence|
   | Q = main Q × 3.0 |  +------------------------+
   +------------------+
              |
      copy gate: far_active + not_dt
      hysteresis: 5 consecutive frames
              |
              v
         +---------------------------+
         | RES Post-Filter           |
         | (coherence-based EER,     |
         |  spectral subtraction,    |
         |  OLA + sqrt-Hann)         |
         +---------------------------+
                         |
                         v
         +---------------------------+
         | Output Limiter            |
         | (smoothed gain clamp)     |
         +---------------------------+
                         |
                         v
                   Clean Output
```

### 各模組功能概覽

| 模組 | 功能 | 詳細說明 |
|------|------|----------|
| **High-Pass Filter** | 移除 DC 偏移、50/60Hz 電源雜訊、低頻隆隆聲 | 2 階 Butterworth IIR，截止 80Hz |
| **Saturation Detect** | 偵測 speaker 飽和失真，soft-clip reference | 避免非線性 echo 破壞濾波器建模 |
| **Delay Estimation** | 自動估計 mic-ref 延遲 | GCC-PHAT，週期性重估（每 2s） |
| **Reference Alignment** | Ring buffer 延遲補償 | 讓自適應濾波器專注學習 echo path，不浪費 capacity 建模 delay |
| **PBFDAF** | 分區頻域自適應濾波器 | 多 partition 頻域 echo 估計與消除 |
| **FDKF** | 頻域卡爾曼濾波器 adaptation | Per-bin Kalman gain 取代 NLMS 固定步長，two-stage Q 控制 |
| **Shadow Filter** | 背景濾波器（dual-filter 架構） | 更積極的 Q 設定，copy gate 自動修正 main filter |
| **DTD** | 雙講偵測（可選） | Divergence + Coherence 偵測器，連續 mu scaling |
| **RES** | 殘餘回音抑制 | EER-based spectral subtraction，OLA + sqrt-Hann 完美重建 |
| **Output Limiter** | 安全網 | 保證 output 永不超過 mic amplitude |

## API 使用

```python
from aec import AEC, AecConfig, AecMode, AecPreset

# 推薦：使用 Preset（自動配置 RES + adaptive filter 參數，FDKF 預設開啟）
config = AecConfig.from_preset(AecPreset.BALANCED,
                               sample_rate=16000, mode=AecMode.SUBBAND,
                               enable_res=True)
aec = AEC(config)

# 更強 echo 壓制（犧牲部分近端品質）
config = AecConfig.from_preset(AecPreset.AGGRESSIVE,
                               sample_rate=16000, mode=AecMode.SUBBAND,
                               enable_res=True)
aec = AEC(config)

# 手動配置（無 preset，FDKF 預設開啟）
config = AecConfig(
    mode=AecMode.SUBBAND,
    enable_res=True,
)
aec = AEC(config)

# 標準配置：subband + Shadow（無 FDKF、無 RES）
config = AecConfig(mode=AecMode.SUBBAND)
aec = AEC(config)

# 時域 NLMS 模式（無 Shadow/DTD）
config = AecConfig(mode=AecMode.NLMS, mu=0.3)
aec = AEC(config)

# 處理
hop_size = aec.hop_size  # 160 samples (10ms @ 16kHz)
while has_audio:
    output = aec.process(mic_block, ref_block)
    erle = aec.get_erle()
    conf = aec.get_dtd_confidence()
```

## 參數說明

### 濾波器參數

| 參數 | 預設值 | 範圍 | 說明 |
|------|--------|------|------|
| `mu` | 0.5 (SUBBAND) / 0.3 (NLMS) | 0.1-0.8 | 步長（FDKF 模式下由 Kalman gain 取代） |
| `filter_length` | 512 (NLMS/LMS/FDAF/SUBBAND) | 256-4096 | 濾波器長度 (samples) |
| `enable_dtd` | False | - | DTD：僅 FDAF/SUBBAND（Divergence + Coherence），詳見 [docs/dtd_design.md](docs/dtd_design.md) |
| `enable_res` | False (Python) / True (C) | - | 殘餘回音抑制 |
| `use_kalman` | True | - | FDKF adaptation（per-bin Kalman gain，搭配 two-stage Q），預設開啟 |
| `enable_delay_est` | True | - | 自動延遲估計（GCC-PHAT） |
| `max_delay_ms` | 250.0 | 50-500 | 延遲搜尋範圍上限 (ms) |
| `fixed_delay_samples` | -1 | -1 or ≥0 | 若 ≥0，使用固定延遲（跳過估計） |
| `enable_highpass` | True | - | 高通濾波器（移除 DC + 低頻雜訊） |
| `highpass_cutoff_hz` | 80.0 | 20-200 | 高通截止頻率 (Hz) |
| `enable_saturation_detect` | True | - | 飽和偵測（非線性 echo 處理） |
| `enable_shadow` | True | - | Shadow filter（僅 fdaf/subband，預設開啟） |

### AecPreset 系統

三級 preset 控制 **RES 後處理** 和 **自適應濾波器** 的積極程度，提供 echo 壓制 vs 近端品質的 trade-off：

| Preset | Echo 壓制 | 近端品質 | 適用場景 |
|--------|-----------|----------|----------|
| **BALANCED** | 適中 | 最佳保留 | 會議通話（預設） |
| **AGGRESSIVE** | 強 | 輕微損傷 | 免持電話、車用 |
| **MAXIMUM** | 最強 | 明顯損傷 | 揚聲器對話、高回聲環境 |

**Preset 控制的參數**：

| 參數 | BALANCED | AGGRESSIVE | MAXIMUM | 說明 |
|------|----------|------------|---------|------|
| **RES 層** | | | | |
| `res_g_min_db` | -25 | -35 | -40 | RES 最小增益 |
| `res_over_sub_base` | 2.5 | 4.0 | 6.0 | 過減因子基底 |
| `res_over_sub_scale` | 4.0 | 6.0 | 8.0 | 過減因子隨 ERLE 縮放 |
| `res_dt_reduction` | 3.5 | 2.0 | 0.0 | DT 時 over_sub 降低倍率 |
| `res_ne_protect_db` | -8 | -5 | -2 | Per-bin 近端保護上限（越高=保護越弱） |
| `res_spectral_floor_db` | -25 | -30 | -35 | 頻譜底噪估計 |
| `shadow_q_ratio` | 3.0 | 4.0 | 5.0 | Shadow Q 倍率 |
| **Adaptive 層** | | | | |
| `shadow_mu_min` | 0.5 | 0.7 | 0.9 | DT 時 mu floor（越高 = 越積極更新） |
| `warmup_frames` | 100 | 150 | 200 | 強制高 mu 的 warmup 幀數 |
| `kalman_q_high` | 1e-3 | 3e-3 | 1e-2 | FDKF Q_high（越高 = 越快收斂） |

```python
from aec import AecConfig, AecPreset, AecMode

# Preset + 自訂覆蓋
config = AecConfig.from_preset(
    AecPreset.AGGRESSIVE,
    sample_rate=16000,
    mode=AecMode.SUBBAND,
    enable_res=True,
    filter_length=2048,   # 覆蓋 preset 以外的參數
)
```

### FDKF 參數 (Frequency Domain Kalman Filter)

FDKF 使用 per-bin Kalman gain 取代 NLMS 的固定步長 mu，自動為每個頻率 bin 選擇最佳 adaptation rate。
搭配 two-stage Q（process noise）控制：Q_high 快速收斂，Q_low 穩態追蹤。

**核心公式**：
```
K[k] = P[k] × |X[k]|² / (|X[k]|² × P[k] + R[k])   # Per-bin Kalman gain
W[k] += K[k] × E[k] × conj(X[k])                     # Weight update
P[k] = (1 - K[k] × X[k]) × P[k] + Q[k]               # Covariance update
R[k] = α × R[k] + (1-α) × |E[k]|²                     # Measurement noise (from error)
```

**Two-stage Q 運作機制**：

| 階段 | Q 值 | K (Kalman gain) | 行為 |
|------|------|-----------------|------|
| 收斂前 (Q_high) | `kalman_q_high`（預設 1e-3） | 0.5~0.9 | 積極更新，打破 R deadlock |
| 收斂後 (Q_low) | 1e-5 | 0.01~0.1 | 穩態追蹤，不干擾已收斂權重 |
| 切換條件 | — | — | 連續 10 幀 ERLE > 6dB |

**R self-reinforcing deadlock（Q_low 固定時的問題）**：
```
高 error → 高 R → 低 K → 慢更新 → 仍然高 error（惡性循環）
```
Q_high = 1e-3 解決此問題：高 Q 持續注入 P → 即使 R 高，K 仍維持合理值。

| 參數 | 預設值 | 說明 |
|------|--------|------|
| `use_kalman` | True | 啟用 FDKF adaptation（v1.15.0 起預設開啟） |
| `kalman_q_high` | 1e-3 | 快速收斂 process noise（Preset 可調） |
| `warmup_frames` | 100 | 強制高 mu 的 warmup 幀數（Preset 可調） |
| P_init | 0.5 | Per-partition per-bin 初始 error covariance |
| Q_low | 1e-5 | 穩態追蹤 process noise |
| R_init | 1e-2 | 初始 measurement noise |
| α_R | 0.95 | R 估計 EMA 平滑係數 |

### Shadow Filter 參數 (僅 fdaf/subband 模式，預設開啟)

Shadow filter（雙濾波器）使用更積極參數的背景濾波器持續追蹤回音路徑，當 shadow 表現更好時透過 copy gate 自動將權重複製到 main filter。
這是 WebRTC AEC3 和 SpeexDSP 的核心機制，v1.6.0 起為預設保護策略。

FDKF 模式下，shadow 使用 `shadow_q_ratio` 倍的 Q 值，確保 shadow 收斂更快，copy gate 能有效觸發。

> Shadow 預設單獨使用（≈ WebRTC/SpeexDSP 做法），可加 `--enable-dtd` 啟用雙重保護。
> 詳見 [docs/aec_methods.md §6.4](docs/aec_methods.md) 的組合行為表。

| 參數 | 預設值 | 範圍 | 說明 |
|------|--------|------|------|
| `shadow_copy_threshold` | 0.7 | 0.3-1.0 | Shadow error < main error × threshold 時複製權重 |
| `shadow_err_alpha` | 0.85 | 0.8-0.99 | Error energy EMA 平滑係數 |
| `shadow_mu_min` | 0.5 | 0.1-0.8 | 主濾波器 DT 時 mu 下限（shadow-only 模式的輕量保護） |
| `shadow_copy_hysteresis` | 5 | 1-10 | 連續 N frames 符合條件才觸發 copy |
| `shadow_q_ratio` | 3.0 | 1.0-5.0 | Shadow Q = main Q × ratio（FDKF 模式） |

**Shadow filter 在 FDKF 模式下的 Q 設定**：

| 階段 | main Q | shadow Q | 效果 |
|------|--------|----------|------|
| 收斂前 (Q_high) | `kalman_q_high` | Q_high × `shadow_q_ratio` | Shadow 更快收斂，copy gate 更早觸發 |
| 收斂後 (Q_low) | 1e-5 | Q_low × `shadow_q_ratio` | Shadow 仍比 main 積極，echo path 變化時更快偵測 |
| Echo path 變化 | 慢速恢復 | ratio × 快速恢復 | Shadow copy 觸發，main 快速跟上 |

**設計要點**：
- Shadow 使用 full mu（1.0），不受 DT 影響，持續追蹤回音路徑
- Copy gate 條件：`far_active AND not_dt AND NOT epc_active`
- Copy 門檻：`shadow_err < main_err × 0.7`（連續 5 frames）
- Copy 只複製 weights + P（FDKF 模式），不切換 output（避免不連續）
- 50-frame warm-up guard：收斂前不允許 copy（避免未收斂時的誤判退化）
- 雙向 copy：main 比 shadow 好時也會反向 copy（main → shadow）
- 可與 DTD 互補：DTD 降低 mu 防止惡化，shadow 在背景持續追蹤並自動修正

### RES 參數 (僅 fdaf/subband 模式)

RES（Residual Echo Suppressor）使用 OLA + sqrt-Hann 窗框架，避免 frame 邊界不連續和棋盤頻譜（musical noise）。
增益公式為 spectral subtraction Wiener gain：`G = max(1 - over_sub × EER, g_min)`。

**Per-bin near-end gate（v1.14.0）**：使用 far-end ↔ error 的 coherence² (coh2) 做 per-bin 近端偵測：
- `coh2[k]` 高（echo bin）→ 允許壓到 g_min（激進壓制）
- `coh2[k]` 低（near-end bin）→ 提高 floor 至 `ne_protect_db`（保護近端）
- 取代舊版 broadband `dt_indicator` → 解決 FS 時 echo 被誤判為 DT 的問題

| 參數 | 預設值 | 範圍 | 說明 |
|------|--------|------|------|
| `res_g_min_db` | -25 | -40 ~ -10 | 最小增益（echo-only bin 的壓制下限） |
| `res_ne_protect_db` | -10 | -2 ~ -10 | Per-bin 近端保護上限（Preset 可調） |
| `res_over_sub` | 3.0 | 1.0-15.0 | 過減因子（動態：base+scale×erle_factor，DT 時降低） |
| `res_alpha` | 0.8 | 0.5-0.95 | 增益平滑係數 |

### 模式選擇指南

| 應用場景 | 推薦模式 | 推薦配置 |
|----------|----------|----------|
| 最佳品質（會議系統、智慧音箱） | subband | `--enable-res`（FDKF + Shadow + RES，FDKF 預設開啟） |
| 一般用途（手機/耳機） | nlms | 預設（無 Shadow/DTD） |
| 中等回音 | fdaf | `--enable-res` |
| 極低資源（MCU） | lms | `--mu 0.01` |

### 繪圖工具

```bash
# 繪製 AEC 結果（預設 Shadow 開啟）
python3 plot_aec_results.py ../wav/ --mode subband

# 開啟 RES
python3 plot_aec_results.py ../wav/ --mode subband --enable-res

# 加入 DTD（雙重保護）
python3 plot_aec_results.py ../wav/ --mode subband --enable-dtd
```

## Benchmark 比較（AEC Challenge, v1.15.0）

### 三級 Preset 比較 — ERLE

測試條件：subband + Shadow + FDKF + RES（per-bin gate）+ delay pre-alignment + HPF + saturation detect。
Frame/hop: 320/160 (20ms/10ms), FFT: 512, filter_length: 512。

#### Farend Single-Talk（15 cases）

| 指標 | BALANCED | AGGRESSIVE | MAXIMUM | SpeexDSP | WebRTC AEC3 |
|------|----------|------------|---------|----------|-------------|
| Mean ERLE | 12.4 dB | **15.7 dB** | 15.0 dB | 6.9 dB | 25.8 dB |

#### Doubletalk（15 real cases）

| 指標 | BALANCED | AGGRESSIVE | MAXIMUM | SpeexDSP | WebRTC AEC3 |
|------|----------|------------|---------|----------|-------------|
| Mean ERLE | 4.2 dB | **7.2 dB** | **7.9 dB** | 1.3 dB | 3.7 dB |

> **解讀**：
> - **v1.15.0 frame size 統一後 FS ERLE 提升**：AGGRESSIVE 14.7→15.7 dB (+1.0 dB)。
> - **DT ERLE 碾壓 AEC3**：AGGRESSIVE 7.2 dB vs AEC3 3.7 dB（+3.5 dB）。
> - **AGGRESSIVE 推薦**：FS ERLE 15.7 dB, DT ERLE 7.2 dB — 最佳平衡點。
> - FS echo_mos 差距主要來自 AEC3 的 NLP（非線性後處理），而非線性濾波器性能。
> - 詳見 [docs/aec_improve_v7.md](docs/aec_improve_v7.md)。

### 工具

```bash
# 執行 AEC Challenge benchmark（需 speexdsp Python binding + WebRTC AEC3 CLI）
cd AEC
python3 python/eval_aec_challenge.py wav/aec_challenge/ --aec3 --speex

# 指定 preset
python3 python/eval_aec_challenge.py wav/aec_challenge/ --preset aggressive

# 比較全部 preset
python3 python/eval_aec_challenge.py wav/aec_challenge/ --all-presets

# AECMOS 評估（需 speechmos + onnxruntime，Python 3.13+）
/opt/homebrew/bin/python3 python/eval_aecmos.py wav/aec_challenge/

# AECMOS 比較全部 preset
/opt/homebrew/bin/python3 python/eval_aecmos.py wav/aec_challenge/ --all-presets
```

## 效能指標

| 模式 | ERLE | 複雜度 | 收斂時間 |
|------|------|--------|----------|
| lms | 10-15 dB | O(N) | 1-5s |
| nlms | 15-20 dB | O(N) | 0.5-2s |
| fdaf | 18-22 dB | O(N log N) | 0.3-1s |
| subband | 20-25 dB | O(N log N) | 0.2-0.8s |
| subband + FDKF | 25-30 dB | O(N log N) | 0.1-0.5s |
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
│   ├── aec.py                 # Python 實作 (LMS/NLMS/FDAF/SUBBAND + FDKF + Delay Est.)
│   ├── eval_aec_challenge.py  # AEC Challenge benchmark 評測 (ERLE/PESQ)
│   ├── eval_aecmos.py         # AECMOS 評測 (echo_mos/deg_mos)
│   ├── plot_aec_results.py    # 結果繪圖 (含 DTD 紅底)
│   └── gen_sim_data.py        # 測試資料生成
├── docs/
│   ├── aec_methods.md         # 演算法文檔（完整演算法說明）
│   ├── aec_improve_v3.md      # v1.9.0 改善紀錄（Delay+FDKF+RES）
│   ├── aec_improve_v4.md      # v1.11.0 改善紀錄（warmup+adaptive g_min+AECMOS）
│   ├── aec_improve_v5.md      # v1.12.0 改善紀錄（Two-stage Q+Shadow Q ratio）
│   ├── aec_improve_v6.md      # v1.13.0 改善紀錄（Preset adaptive filter 層）
│   ├── aec_improve_v7.md      # v1.14.0 改善紀錄（Per-bin near-end gate）
│   ├── aec_improve_v8.md      # v1.15.0 改善紀錄（Frame/hop 統一 + 清理 + 重命名）
│   ├── dtd_design.md          # DTD 設計文檔（完整說明）
│   └── DEVLOG.md              # 開發紀錄
└── wav/                       # 測試音檔
```

## 參考文獻

1. Haykin, S. "Adaptive Filter Theory"
2. Benesty, J. et al. "Advances in Network and Acoustic Echo Cancellation"
3. Sondhi, M.M. "An adaptive echo canceller" (1967)
4. Ferrara, E.R. "Fast implementations of LMS adaptive filters" (1980)
5. Enzner, G. et al. "Frequency-domain adaptive Kalman filter for acoustic echo control" (2006)
