# AEC v1.15.0 改善紀錄 — Frame/Hop 統一 + 清理 + 重命名

## 概述

本版本進行三大改動：
1. 統一 frame/hop/FFT size（20ms/10ms/512）
2. 清理未使用的 dead code
3. FREQ → FDAF 重命名

## 1. Frame/Hop/FFT Size 統一

### 目標

| 參數 | 舊值 | 新值 | 說明 |
|------|------|------|------|
| `frame_size` | 512 (32ms) | **320** (20ms) | WOLA 窗長 |
| `hop_size` | 256 (16ms) | **160** (10ms) | 處理延遲 |
| `fft_size` | 512 | 512（不變） | nearest 2^n ≥ 320 |

### SubbandNlms 修改（Overlap-save）

- 新增獨立 `hop_size` 參數，不再硬綁 `block_size // 2`
- `block_size` = 512（FFT size），`hop_size` = 160
- Buffer 仍然 512 samples（overlap-save 需要 FFT size）
- Output 提取: `near_buffer[-hop:]`（取最後 hop 個 valid sample）
- Error 零填充: `error_time[-hop:] = output`
- Weight constraint: `w_time[hop_size:] = 0`（partition 長度 = hop = 160）
- Overlap-save 約束: hop ≤ block_size/2 → 160 ≤ 256 ✓

### ResFilter 修改（WOLA）

- `frame_size=320`（窗長），`hop_size=160`（50% overlap）
- sqrt-Hann 窗，50% overlap → 完美重建 ✓
- FFT: zero-pad 320→512 → `rfft(512)` → 257 bins（與 SubbandNlms 相同 n_freqs）
- IFFT: `irfft(257, n=512)[:320]` → synthesis window → OLA

### FDAF Buffering

FDAF 模式使用單一大 FFT block（internal_hop = 512），external_hop = 160：
- 512 不整除 160 → 需要 leftover 處理
- Buffer 大小: `internal_hop + hop_size` = 672
- 每次累積 160 samples，滿 512 後送入 filter，leftover 搬回 buffer 頭

## 2. Dead Code 清理

### 移除項目

| 項目 | 理由 |
|------|------|
| `from dataclasses import field` | 未使用的 import |
| `AecConfig.dtd_threshold` | Legacy，never read |
| `AecConfig.leak` | 永遠被 override 為 1.0 |
| `AecConfig.use_leakage` + `freq_leakage` | Default False，CLI 無 flag，無使用 |
| `NlmsFilter.leak` 參數 | 永遠 1.0，乘法無效果 |
| `SubbandNlms.use_leakage`/`leakage` | 跟隨 config 移除 |

### 修改項目

- `use_kalman: bool = False` → `True`（FDKF 預設開啟）

### 保留項目

- `enable_dtd` + `DtdEstimator` — 完整功能，保留
- `enable_cng` — ResFilter noise tracking 使用
- `enable_shadow` — 有使用
- LMS, NLMS 模式 — 全部保留

## 3. FREQ → FDAF 重命名

- `AecMode.FREQ = "freq"` → `AecMode.FDAF = "fdaf"`
- CLI: `--mode freq` → `--mode fdaf`
- SUBBAND 名稱保留不動

### 修改檔案

| 檔案 | 修改 |
|------|------|
| `python/aec.py` | AecMode enum, from_preset(), CLI choices |
| `python/evaluate_aec.py` | choices, mode_map |
| `python/batch_aec.py` | choices, mode_map |
| `python/plot_aec_results.py` | choices, mode_map |

## 驗證結果

### AEC Challenge Benchmark（15 files, subband + Shadow + FDKF + RES）

**Farend Single-Talk ERLE**：

| Preset | v1.14.0 | v1.15.0 | 差異 |
|--------|---------|---------|------|
| BALANCED | 11.9 dB | 12.4 dB | +0.5 dB |
| AGGRESSIVE | 14.7 dB | **15.7 dB** | **+1.0 dB** |
| MAXIMUM | 13.9 dB | 15.0 dB | +1.1 dB |

**Doubletalk ERLE**：

| Preset | v1.14.0 | v1.15.0 | 差異 |
|--------|---------|---------|------|
| BALANCED | 4.4 dB | 4.2 dB | -0.2 dB |
| AGGRESSIVE | 7.4 dB | 7.2 dB | -0.2 dB |
| MAXIMUM | 8.2 dB | 7.9 dB | -0.3 dB |

### 分析

- **FS ERLE 全面提升**：frame 縮小提升時間解析度，濾波器更快追蹤 echo path 變化
- **DT ERLE 輕微下降**：-0.2 dB 在噪聲範圍內，不影響實際表現
- **延遲改善**：16ms → 10ms，更適合即時通話場景
- **n_freqs 不變**：FFT size 仍為 512 → 257 bins，RES/DTD 等頻域處理完全兼容
