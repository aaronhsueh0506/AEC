# AEC - Acoustic Echo Cancellation

回音消除模組（v2.0.0），Python 支援 PBFDKF（頻域卡爾曼濾波器）、Multi-ERLE、Shadow Filter 和 RES（殘餘回音抑制）。

**v2.0.0 主要改進**：

- **Multi-ERLE**：`FilterErleEstimator`（per-bin，`|echo_spec|²/|error_spec|²`）+ `FullbandErleEstimator`（broadband 交叉驗證），打破 `erle_per_bin` 循環依賴
- **dt_indicator 源頭修正**：用當前幀 `raw_output` 即時 ERLE（3 幀 EMA 平滑）壓制高 coupling FS 的假 DT 偵測
- **ENR 重構**：`nearend_est = max(raw_nearend × dt_indicator, noise_floor)`，FS 時 ENR >> 10 全壓（含非線性破音），DT 時語音保留
- **ne_confidence = dt_indicator**：直接用已修正的 dt_indicator 控制 ENR 門檻混合，移除 `fs_confidence` 間接路徑
- **CNG 凍結式底噪追蹤**：壓制中（gain < 0.9）凍結 noise_psd 上升，避免學習到回音形狀
- **Bug fixes**：Shadow copy 補 Q stage、移除 nearend state machine dead code、移除 output noise gate dead code

**Blind test 成績（fl=512, AEC Challenge Interspeech 2021, 800 cases, AECMOS）**：

| Method | SR | FL | FS echo↑ | DT echo↑ | DT deg↑ | NE deg↑ |
|--------|----|----|----------|----------|---------|---------|
| Linear (balanced) | 16k | 512 | 2.710 | 3.167 | 3.198 | 4.069 |
| Speex (SpeexDSP 1.2) | 16k | 2048 | 2.808 | 3.368 | 3.225 | 4.128 |
| Old WebRTC AEC (M120) | 16k | — | 3.484 | 4.262 | 2.389 | 4.098 |
| **mild** | 16k | 512 | 3.272 | 3.943 | **2.267** | 3.988 |
| **balanced** | 16k | 512 | 3.615 | 4.232 | 2.095 | 3.975 |
| **aggressive** | 16k | 512 | 3.771 | 4.375 | 2.044 | 3.962 |
| **maximum** | 16k | 512 | **3.912** | 4.505 | 2.025 | 3.940 |
| WebRTC AEC3 (M120) | 16k | 768 | 3.875 | **4.538** | 1.850 | 3.454 |

> maximum preset FS echo **3.912 超越 AEC3**（3.875）。DT deg 全 preset 優於 AEC3（2.0+ vs 1.850）。NE deg 全 preset 大幅領先 AEC3（+0.5）。

> Pipeline（AEC+NR+RES）成績見 [Audio_ALG](https://github.com/aaronhsueh0506/Audio_ALG) README。

## 規格與限制

### 輸入規格

| 項目 | 規格 |
|------|------|
| 取樣率 | 8000 / 16000 / 48000 Hz（自動計算 frame/fft/hop） |
| 位元深度 | 16-bit PCM / 32-bit float |
| 通道數 | 單聲道（mono） |
| 幀長度 / Hop | 20ms / 10ms（依 sample rate 自動配置） |
| 濾波器長度 | 64ms（預設，可調。512@8k, 1024@16k, 3072@48k） |
| 延遲 | 10ms（1 hop，algorithmic delay） |

### 適用場景

- 全雙工免持通話（手機、智慧音箱、會議系統）
- 單通道 AEC（1 mic + 1 ref）
- 回音路徑 ≤ 64ms（預設），可透過 `filter_length` 參數延長

### 使用注意事項

| 項目 | 說明 |
|------|------|
| **收斂時間** | Filter 需要約 0.5-2 秒的遠端信號才能收斂（ERLE > 10dB）。收斂前 RES 使用保守的 direct echo estimate |
| **Warmup 期** | 前 80-100 幀（0.8-1.0 秒）為 warmup 階段，不進行收斂判定，避免誤判 |
| **延遲對齊** | 若 mic-ref 延遲 > filter_length，需啟用 delay estimation（`--enable-delay-est`）或手動指定延遲 |
| **遠端信號要求** | 遠端需要有足夠的能量和頻譜豐富度才能有效收斂。純靜音或極低能量段不會觸發 filter 更新 |
| **Echo path 變化** | 支援 Echo Path Change (EPC) 偵測，自動提高 P_MAX 加速重新收斂。但快速大幅變化（如移動設備）仍會暫時增加殘餘 echo |
| **Preset 選擇** | mild 最保守（DT deg 最佳 2.267），maximum 最激進（FS echo 最佳 3.912）。推薦 balanced 作為通用設定 |

### 已知限制

| 限制 | 說明 |
|------|------|
| **不支援 32kHz** | 僅支援 8/16/48 kHz，32kHz 需外部 resample |
| **單通道** | 不支援多麥克風陣列 |
| **線性 echo 為主** | 非線性 echo（speaker 飽和）靠 RES 壓制，無獨立非線性模型 |
| **高 coupling FS** | 30% 幀仍存在 dt_indicator 偏高（filter 間歇失效），影響 FS echo |
| **高 coupling DT** | inst_erle_fast 修正可能誤壓 DT 語音（echo 與語音能量相近時） |
| **CNG** | 底噪追蹤在遠端持續講話時效果有限（noise_psd 可能偏高） |
| **延遲估計** | GCC-PHAT 在低 SNR 或強非線性場景可能不準確 |

支援四級 Preset（MILD / BALANCED / AGGRESSIVE / MAXIMUM）控制 echo 壓制強度。

> C 實作已對齊 Python v1.17.0：PBFDKF（Kalman P_init/P_MAX/Q gating/far-end gate 修正）+ Shadow + WOLA RES v2（ENR masking / direct echo est / reverb tail）+ HPF + Preset，詳見 [c_impl/README.md](c_impl/README.md)。

## 濾波器模式

| 模式 | CLI | 演算法 | 延遲 | 適用場景 |
|------|-----|--------|------|----------|
| **nlms** | `--mode nlms` | 時域 NLMS | 10ms | 一般用途（預設） |
| **fdaf** | `--mode fdaf` | 頻域 NLMS (單一 block) | 10ms | 中等回音、平衡效能 |
| **pbfdaf** | `--mode pbfdaf` | 分區頻域 PBFDAF (NLMS) | 10ms | 長回音路徑 |
| **pbfdkf** | `--mode pbfdkf` | 分區頻域 PBFDKF (Kalman) | 10ms | 長回音路徑、最佳品質（推薦） |
| **lms** | `--mode lms` | 時域 LMS (固定步長) | 10ms | 穩態環境、極低資源 |

## 快速開始

### Python

```bash
cd python
pip install numpy soundfile

# 推薦：PBFDKF + Shadow + RES（Kalman adaptation，最佳品質）
python3 aec.py mic.wav ref.wav output.wav --mode pbfdkf --enable-res

# 使用 Preset（控制 echo 壓制強度，四級：mild / balanced / aggressive / maximum）
python3 aec.py mic.wav ref.wav output.wav --mode pbfdkf --enable-res --preset mild        # 最保守
python3 aec.py mic.wav ref.wav output.wav --mode pbfdkf --enable-res --preset balanced    # 推薦
python3 aec.py mic.wav ref.wav output.wav --mode pbfdkf --enable-res --preset aggressive
python3 aec.py mic.wav ref.wav output.wav --mode pbfdkf --enable-res --preset maximum

# PBFDAF 模式（NLMS adaptation，不使用 Kalman）
python3 aec.py mic.wav ref.wav output.wav --mode pbfdaf --enable-res

# 啟用 per-frame 診斷輸出（ERLE, mu, gain, convergence 等）
python3 aec.py mic.wav ref.wav output.wav --mode pbfdkf --enable-res --preset balanced --diag

# 啟用 DTD（進階，特定場景）
python3 aec.py mic.wav ref.wav output.wav --mode pbfdkf --enable-res --enable-dtd

# FDAF 模式（單一 FFT block，中等回音）
python3 aec.py mic.wav ref.wav output.wav --mode fdaf --enable-res

# 純 PBFDAF（debug 用，關閉 Shadow）
python3 aec.py mic.wav ref.wav output.wav --mode pbfdaf --enable-res --no-shadow

# 時域 NLMS 模式
python3 aec.py mic.wav ref.wav output.wav

# 調整參數
python3 aec.py mic.wav ref.wav output.wav --mu 0.5 --filter 1024
```

### C (PBFDKF + Shadow + WOLA RES)

```bash
cd c_impl && make
./bin/aec_wav mic.wav ref.wav output.wav                           # BALANCED preset（預設）
./bin/aec_wav mic.wav ref.wav output.wav --preset aggressive       # AGGRESSIVE preset
./bin/aec_wav mic.wav ref.wav output.wav --preset maximum          # MAXIMUM preset
./bin/aec_wav mic.wav ref.wav output.wav --no-res                  # 關閉 RES
./bin/aec_wav mic.wav ref.wav output.wav --no-hpf                  # 關閉 HPF
./bin/aec_wav mic.wav ref.wav output.wav --filter 2400             # 自訂濾波器長度
```

C 版本已對齊 Python v1.28.1（PBFDKF 模式），包含 PBFDKF Kalman（correct K, P_MAX=0.5）+ Shadow + WOLA RES v2（ENR masking / direct echo est / reverb tail / CNG）+ HPF + 四級 Preset。
不含：Multi-ERLE、dt_indicator ERLE correction、DTD、Delay Estimation、Saturation Detection、LMS/NLMS/FDAF 模式。待同步至 v2.0.0。

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
              | PBFDKF (Main Filter)      |
              | Kalman adaptation         |
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
         | RES Post-Filter (v2)      |
         | (ENR masking / Wiener,    |
         |  direct echo est + ERLE,  |
         |  reverb tail, OLA)        |
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
| **PBFDAF** | 分區頻域自適應濾波器（NLMS base） | 多 partition 頻域 echo 估計與消除 |
| **PBFDKF** | 頻域卡爾曼濾波器（繼承 PBFDAF） | Per-bin Kalman gain 取代 NLMS 固定步長，two-stage Q 控制 |
| **Shadow Filter** | 背景濾波器（dual-filter 架構） | 更積極的 Q 設定，copy gate 自動修正 main filter |
| **DTD** | 雙講偵測（可選） | Divergence + Coherence 偵測器，連續 mu scaling |
| **RES** | 殘餘回音抑制 | ENR masking（預設）/ Wiener / spectral sub，direct echo est + reverb tail，OLA + sqrt-Hann |
| **Output Limiter** | 安全網 | 保證 output 永不超過 mic amplitude |

## API 使用

```python
from aec import AEC, AecConfig, AecMode, AecPreset

# 推薦：使用 Preset（自動配置 RES + adaptive filter 參數，FDKF 預設開啟）
config = AecConfig.from_preset(AecPreset.BALANCED,
                               sample_rate=16000, mode=AecMode.PBFDKF,
                               enable_res=True)
aec = AEC(config)

# 更強 echo 壓制（犧牲部分近端品質）
config = AecConfig.from_preset(AecPreset.AGGRESSIVE,
                               sample_rate=16000, mode=AecMode.PBFDKF,
                               enable_res=True)
aec = AEC(config)

# 手動配置（無 preset，FDKF 預設開啟）
config = AecConfig(
    mode=AecMode.PBFDKF,
    enable_res=True,
)
aec = AEC(config)

# 標準配置：PBFDKF + Shadow（無 RES）
config = AecConfig(mode=AecMode.PBFDKF)
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
| `mu` | 0.5 (PBFDKF) / 0.3 (NLMS) | 0.1-0.8 | 步長（PBFDKF 模式下由 Kalman gain 取代） |
| `filter_length` | 512 (NLMS/LMS/FDAF) / 1024 (PBFDAF/PBFDKF) | 256-4096 | 濾波器長度 (samples)。CLI 依 mode + sample_rate 自動設定：PB=sr×64ms, 非PB=sr×32ms |
| `enable_dtd` | False | - | DTD：僅頻域模式（Divergence + Coherence），詳見 [docs/dtd_design.md](docs/dtd_design.md) |
| `enable_res` | False (Python) / True (C) | - | 殘餘回音抑制 |
| `use_kalman` | True | - | FDKF adaptation（per-bin Kalman gain，搭配 two-stage Q），預設開啟 |
| `enable_delay_est` | True | - | 自動延遲估計（GCC-PHAT） |
| `max_delay_ms` | 250.0 | 50-500 | 延遲搜尋範圍上限 (ms) |
| `fixed_delay_samples` | -1 | -1 or ≥0 | 若 ≥0，使用固定延遲（跳過估計） |
| `enable_highpass` | True | - | 高通濾波器（移除 DC + 低頻雜訊） |
| `highpass_cutoff_hz` | 80.0 | 20-200 | 高通截止頻率 (Hz) |
| `enable_saturation_detect` | True | - | 飽和偵測（非線性 echo 處理） |
| `enable_shadow` | True | - | Shadow filter（僅頻域模式，預設開啟） |

### AecPreset 系統

四級 preset 控制 **RES 後處理** 和 **自適應濾波器** 的積極程度，提供 echo 壓制 vs 近端品質的 trade-off：

| Preset | Echo 壓制 | 近端品質 | 適用場景 |
|--------|-----------|----------|----------|
| **MILD** | 輕 | 最佳保留 | 會議通話、近端品質優先 |
| **BALANCED** | 適中 | 良好 | 一般用途（推薦預設） |
| **AGGRESSIVE** | 強 | 輕微損傷 | 免持電話、車用 |
| **MAXIMUM** | 最強 | 明顯損傷 | 揚聲器對話、高回聲環境 |

**Preset 控制的參數**：

| 參數 | MILD | BALANCED | AGGRESSIVE | MAXIMUM | 說明 |
|------|------|----------|------------|---------|------|
| **RES v2 層** | | | | | |
| `res_enr_scale` | 1.0 | 0.85 | 0.7 | 0.5 | ENR 門檻縮放（越低越激進） |
| `res_reverb_decay` | 0.6 | 0.65 | 0.7 | 0.8 | Reverb 衰減率 |
| `res_reverb_gain` | 0.8 | 1.4 | 2.0 | 3.0 | Reverb 貢獻 |
| **RES 層** | | | | | |
| `res_g_min_db` | -35 | -45 | -60 | -72 | RES 最小增益 |
| `res_over_sub_base` | 2.5 | 4.0 | 6.0 | 10.0 | 基礎 over-subtraction |
| `res_over_sub_scale` | 4.0 | 7.0 | 10.0 | 15.0 | ERLE-scaled over-sub |
| `res_dt_reduction` | 3.5 | 2.0 | 1.0 | 0.0 | DT 時放鬆壓制量 |
| `res_ne_protect_db` | -12 | -16 | -20 | -35 | Per-bin 近端保護上限 |
| `res_spectral_floor_db` | -25 | -32 | -40 | -50 | 頻譜底噪估計 |
| `shadow_q_ratio` | 3.0 | 3.5 | 4.0 | 5.0 | Shadow Q 倍率 |
| **Adaptive 層** | | | | | |
| `shadow_mu_min` | 0.5 | 0.6 | 0.7 | 0.9 | DT 時 mu floor |
| `warmup_frames` | 150 | 150 | 150 | 200 | 強制高 mu 的 warmup 幀數 |
| `kalman_q_high` | 2e-4 | 1.5e-4 | 1e-4 | 1e-4 | PBFDKF Q_high |

```python
from aec import AecConfig, AecPreset, AecMode

# Preset + 自訂覆蓋
config = AecConfig.from_preset(
    AecPreset.AGGRESSIVE,
    sample_rate=16000,
    mode=AecMode.PBFDKF,
    enable_res=True,
    filter_length=2048,   # 覆蓋 preset 以外的參數
)
```

### PBFDKF 參數 (Partitioned Block Frequency Domain Kalman Filter)

PBFDKF（繼承 PBFDAF）使用 per-bin Kalman gain 取代 NLMS 的固定步長 mu，自動為每個頻率 bin 選擇最佳 adaptation rate。
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
| 收斂前 (Q_high) | `kalman_q_high`（預設 1e-4） | 0.3~0.7 | 積極更新，打破 R deadlock |
| 收斂後 (Q_low) | `kalman_q_low`（預設 1e-7） | 0.01~0.05 | 穩態追蹤，不干擾已收斂權重 |
| 切換條件 | — | — | 連續 6 幀 single-talk ERLE > 4dB |

**R self-reinforcing deadlock（Q_low 固定時的問題）**：
```
高 error → 高 R → 低 K → 慢更新 → 仍然高 error（惡性循環）
```
Q_high = 1e-4 解決此問題：高 Q 持續注入 P → 即使 R 高，K 仍維持合理值。搭配 P_MAX=0.02 和 per-bin Q gating 防止 weight jitter。

| 參數 | 預設值 | 說明 |
|------|--------|------|
| `use_kalman` | True | True=PBFDKF（Kalman），False=PBFDAF（NLMS） |
| `kalman_q_high` | 1e-4 | 快速收斂 process noise（所有 Preset 統一） |
| `kalman_q_low` | 1e-7 | 穩態追蹤 process noise |
| `warmup_frames` | 100 | 強制高 mu 的 warmup 幀數（Preset 可調） |
| P_init | 0.01 | Per-partition per-bin 初始 error covariance |
| P_MAX | 0.02 | P clamp 上限（EPC 期間臨時放寬至 0.1, 30 frames） |
| R_init | 1e-2 | 初始 measurement noise |
| α_R | 0.95 | R 估計 EMA 平滑係數 |

### Shadow Filter 參數 (僅頻域模式，預設開啟)

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
| 收斂前 (Q_high) | `kalman_q_high` (1e-4) | Q_high × `shadow_q_ratio` | Shadow 更快收斂，copy gate 更早觸發 |
| 收斂後 (Q_low) | `kalman_q_low` (1e-7) | Q_low × `shadow_q_ratio` | Shadow 仍比 main 積極，echo path 變化時更快偵測 |
| Echo path 變化 | 慢速恢復 | ratio × 快速恢復 | Shadow copy 觸發，main 快速跟上 |

**設計要點**：
- Shadow 使用 full mu（1.0），不受 DT 影響，持續追蹤回音路徑
- Copy gate 條件：`far_active AND not_dt AND NOT epc_active`
- Copy 門檻：`shadow_err < main_err × 0.7`（連續 5 frames）
- Copy 只複製 weights + P（FDKF 模式），不切換 output（避免不連續）
- 50-frame warm-up guard：收斂前不允許 copy（避免未收斂時的誤判退化）
- 雙向 copy：main 比 shadow 好時也會反向 copy（main → shadow）
- 可與 DTD 互補：DTD 降低 mu 防止惡化，shadow 在背景持續追蹤並自動修正

### RES 參數 (僅頻域模式)

RES v2（Residual Echo Suppressor）使用 OLA + sqrt-Hann 窗框架，支援三種增益公式：

**增益公式（`res_gain_type`）**：
- **`"enr"`（預設）**: ENR masking（仿 AEC3 `GainToNoAudibleEcho`），不需要 nearend PSD 估計，FS 自然完全抑制
- `"wiener"`: Wiener gain `G = nearend / (nearend + β × echo)`，DT 保護好但 FS 可能 gain pumping
- `"spectral_sub"`: Legacy spectral subtraction `G = 1 - over_sub × EER`

**Echo 估計（`res_echo_method`）**：
- **`"direct"`（預設）**: 使用自適應濾波器 echo spectrum + per-bin ERLE tracking，快速追蹤
- `"coherence"`: Legacy coherence-based EER

**附加功能**：
- **Reverb tail**: 指數衰減模型捕捉晚期殘響（`res_enable_reverb`）
- **4-block 近端平均**: 平滑近端 PSD，減少幀間增益抖動
- **頻率域後處理**: DC 一致性 + HF cap（仿 AEC3 `PostprocessGains`）
- **Per-bin near-end gate**: coherence² 做 per-bin 近端偵測（v1.14.0+）

| 參數 | 預設值 | 範圍 | 說明 |
|------|--------|------|------|
| `res_gain_type` | `"enr"` | enr/wiener/spectral_sub | 增益公式（Preset 預設 enr） |
| `res_echo_method` | `"direct"` | direct/coherence | Echo 估計方法（Preset 預設 direct） |
| `res_enr_scale` | 1.0 | 0.1-2.0 | ENR 門檻縮放（<1 更激進，1.0=AEC3 默認） |
| `res_enable_reverb` | True | - | Reverb tail model（Preset 預設開啟） |
| `res_reverb_decay` | 0.5 | 0.3-0.9 | Reverb 指數衰減率 |
| `res_reverb_gain` | 0.5 | 0.0-2.0 | Reverb 貢獻縮放 |
| `res_g_min_db` | -35 | -72 ~ -10 | 最小增益（echo-only bin 的壓制下限） |
| `res_ne_protect_db` | -12 | -35 ~ -6 | Per-bin 近端保護上限（越低保護越少，Preset 可調） |
| `res_over_sub` | 3.0 | 1.0-15.0 | 過減因子（wiener/spectral_sub 用，ENR 不使用） |
| `res_alpha` | 0.8 | 0.5-0.95 | 增益平滑係數 |

### 模式選擇指南

| 應用場景 | 推薦模式 | 推薦配置 |
|----------|----------|----------|
| 最佳品質（會議系統、智慧音箱） | pbfdkf | `--enable-res`（Kalman + Shadow + RES） |
| 一般用途（手機/耳機） | nlms | 預設（無 Shadow/DTD） |
| 中等回音 | fdaf | `--enable-res` |
| 極低資源（MCU） | lms | `--mu 0.01` |

### 繪圖工具

```bash
# 繪製 AEC 結果（預設 Shadow 開啟）
python3 plot_aec_results.py ../wav/ --mode pbfdkf

# 開啟 RES
python3 plot_aec_results.py ../wav/ --mode pbfdkf --enable-res

# 加入 DTD（雙重保護）
python3 plot_aec_results.py ../wav/ --mode pbfdkf --enable-dtd
```

## Benchmark 比較

測試條件：PBFDKF + Shadow + RES v2 + delay pre-alignment + HPF + saturation detect。
Frame/hop: 20ms/10ms（自動依 sample rate 配置），filter_length: 100ms。以下測試使用 16kHz。
評估指標：AECMOS（echo_mos↑ = echo 少, deg_mos↑ = 語音損傷少）。

### AEC Challenge Interspeech 2021 Blind Test（800 cases）

#### Overall

| Method | FS echo↑ | DT echo↑ | DT deg↑ | NE deg↑ |
|--------|----------|----------|---------|---------|
| Speex (SpeexDSP v1.2) | 2.808 | 3.368 | 3.225 | 4.128 |
| Old WebRTC AEC (AEC2) | 3.484 | 4.262 | 2.389 | 4.098 |
| **Ours mild** | 3.420 | 4.105 | **2.257** | 4.005 |
| **Ours balanced** | 3.710 | 4.368 | 2.093 | 3.997 |
| **Ours aggressive** | 3.830 | 4.468 | 2.051 | 3.990 |
| **Ours maximum** | **3.951** | 4.527 | 2.044 | 3.976 |
| WebRTC AEC3 | 3.875 | **4.538** | 1.850 | 3.454 |

#### No Movement

| Method | FS echo↑ | DT echo↑ | DT deg↑ |
|--------|----------|----------|---------|
| Speex | 2.847 | 3.427 | 3.179 |
| Old WebRTC AEC | 3.457 | 4.331 | 2.304 |
| **Ours mild** | 3.383 | 4.149 | 2.229 |
| **Ours balanced** | 3.703 | 4.399 | 2.073 |
| **Ours aggressive** | 3.843 | 4.503 | 2.020 |
| **Ours maximum** | **3.960** | 4.555 | 2.009 |
| WebRTC AEC3 | 3.871 | **4.566** | 1.858 |

#### With Movement

| Method | FS echo↑ | DT echo↑ | DT deg↑ |
|--------|----------|----------|---------|
| Speex | 2.757 | 3.272 | 3.301 |
| Old WebRTC AEC | 3.519 | 4.149 | 2.528 |
| **Ours mild** | 3.467 | 4.033 | 2.302 |
| **Ours balanced** | 3.720 | 4.317 | 2.125 |
| **Ours aggressive** | 3.814 | 4.411 | 2.103 |
| **Ours maximum** | **3.939** | 4.482 | 2.102 |
| WebRTC AEC3 | 3.882 | **4.492** | 1.836 |

> - **maximum preset FS echo 超越 AEC3**（overall 3.951 vs 3.875, no-move 3.960 vs 3.871）
> - **NE deg 全 preset 大幅領先 AEC3**（~4.0 vs 3.454, +0.5）— 近端語音幾乎無損
> - Speex 版本：SpeexDSP 1.2rc3；WebRTC AEC3：M120 (2024)；Old WebRTC AEC：M120 AEC2

### 工具

```bash
# 執行 AEC Challenge benchmark（需 speexdsp Python binding + WebRTC AEC3 CLI）
cd AEC
python3 python/eval_aec_challenge.py wav/aec_challenge/ --aec3 --speex

# 指定 preset
python3 python/eval_aec_challenge.py wav/aec_challenge/ --preset aggressive

# 比較全部 preset
python3 python/eval_aec_challenge.py wav/aec_challenge/ --all-presets

# AECMOS 評估（需 speechmos + onnxruntime≤1.16.3 + numpy<2，用 venv）
source .venv/bin/activate
python3 python/eval_aecmos.py wav/aec_challenge/

# AECMOS 比較全部 preset
python3 python/eval_aecmos.py wav/aec_challenge/ --all-presets
```

## 效能指標

| 模式 | ERLE | 複雜度 | 收斂時間 |
|------|------|--------|----------|
| lms | 10-15 dB | O(N) | 1-5s |
| nlms | 15-20 dB | O(N) | 0.5-2s |
| fdaf | 18-22 dB | O(N log N) | 0.3-1s |
| pbfdaf | 20-25 dB | O(N log N) | 0.2-0.8s |
| pbfdkf | 25-30 dB | O(N log N) | 0.1-0.5s |
| + RES | +2-4 dB | O(K) | - |

## 檔案結構

```
AEC/
├── README.md                  # 本文件（Python 為主）
├── c_impl/                    # C 實作 (PBFDKF + Shadow + WOLA RES + HPF + Preset)
│   ├── include/               # aec.h, aec_types.h, pbfdkf.h, res_filter.h, hpf.h
│   ├── src/                   # aec.c, pbfdkf.c, res_filter.c, hpf.c, fft_wrapper.c
│   ├── example/               # main.c (CLI), wav_io.h
│   └── Makefile
├── python/
│   ├── aec.py                 # Python 實作 (PBFDAF/PBFDKF + Shadow + RES + Delay Est.)
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
│   ├── aec_improve_v9.md      # v1.19.x 改善紀錄（Echo 抑制 + 診斷 + Preset 修正）
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
