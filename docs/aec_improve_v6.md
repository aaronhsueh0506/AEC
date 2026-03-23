# AEC v1.13.0 改善紀錄

## 概述

基於 v1.12.0（AecPreset 僅控制 RES 層）架構，擴展 preset 至自適應濾波器層，新增 `kalman_q_high`、`warmup_frames` 為可配置參數，讓 preset 同時控制濾波器收斂速度和 echo 壓制強度。

---

## 動機

v1.12.0 的三級 preset 只調整 RES 後處理參數，adaptive filter 層完全相同。
實驗發現：

- 單純降低 DT mu floor (`shadow_mu_min` 0.5→0.1) 同時傷害 FS 和 DT 的 echo_mos
- FDKF 在 DT clips 需要 ~2 秒達 6dB ERLE，但多數 DT clips 只有 1-3 秒 far-end only 時間
- 正確方向：**提高** filter 收斂速度（更高 mu/Q/warmup），而非降低 DT 時的 mu

## 設計

### 新增 AecConfig 參數

| 參數 | 類型 | 預設值 | 說明 |
|------|------|--------|------|
| `kalman_q_high` | float | 1e-3 | FDKF Q_high — 越高收斂越快 |
| `warmup_frames` | int | 100 | 強制高 mu 幀數 — 越長開頭收斂越快 |

`shadow_mu_min` 已存在（預設 0.5），不需新增。

### Preset 參數（RES + Adaptive）

| 參數 | BALANCED | AGGRESSIVE | MAXIMUM | 效果 |
|------|----------|------------|---------|------|
| **RES 層** | | | | |
| res_g_min_db | -25 | -35 | -40 | 最小增益 |
| res_over_sub_base | 2.5 | 4.0 | 6.0 | 過減因子基底 |
| res_over_sub_scale | 4.0 | 6.0 | 8.0 | 過減縮放 |
| res_dt_reduction | 3.5 | 2.5 | 1.5 | DT 降低倍率 |
| res_spectral_floor_db | -25 | -30 | -35 | 頻譜底噪 |
| shadow_q_ratio | 3.0 | 4.0 | 5.0 | Shadow Q 倍率 |
| **Adaptive 層** | | | | |
| shadow_mu_min | 0.5 | 0.7 | 0.9 | DT mu floor |
| warmup_frames | 100 | 150 | 200 | Warmup 幀數 |
| kalman_q_high | 1e-3 | 3e-3 | 1e-2 | FDKF Q_high |

### 實作細節

1. **AEC.__init__()**：FDKF 建立後，用 `config.kalman_q_high` 覆蓋 main filter 的 Q_high/Q，再乘 `shadow_q_ratio` 設定 shadow filter
2. **AEC.__init__() + reset()**：`self._warmup_frames = self.config.warmup_frames` 取代 hardcoded 100

---

## Benchmark 結果

### Farend Single-Talk（15 cases, AEC Challenge）

**ERLE**：

| Preset | Mean ERLE | SpeexDSP | WebRTC AEC3 |
|--------|-----------|----------|-------------|
| BALANCED | 11.5 dB | 6.9 dB | 25.8 dB |
| AGGRESSIVE | 12.2 dB | 6.9 dB | 25.8 dB |
| MAXIMUM | 12.9 dB | 6.9 dB | 25.8 dB |

**AECMOS**：

| Preset | echo_mos | deg_mos |
|--------|----------|---------|
| BALANCED | 3.19 | 5.00 |
| AGGRESSIVE | **3.30** | 5.00 |
| MAXIMUM | 3.38 | 5.00 |

### Doubletalk（15 real cases）

**AECMOS**：

| Preset | echo_mos | deg_mos |
|--------|----------|---------|
| BALANCED | 3.02 | **3.52** |
| AGGRESSIVE | **3.21** | 3.29 |
| MAXIMUM | 3.33 | 3.10 |
| SpeexDSP | 2.76 | 3.90 |
| WebRTC AEC3 | 4.39 | 2.51 |

### 改善幅度（vs v1.12.0 baseline = no preset）

| Preset | FS echo_mos | DT echo_mos | DT deg_mos |
|--------|-------------|-------------|------------|
| BALANCED | -0.01 | -0.01 | ±0 |
| AGGRESSIVE | **+0.10** | **+0.19** | -0.23 |
| MAXIMUM | +0.18 | +0.31 | -0.42 |

---

## 分析

### 為何提高 mu/Q 改善 DT echo_mos

DT clips 的結構通常是：1-3 秒 far-end only → double-talk 開始。

- FDKF 需要 ~2 秒 reach 6dB ERLE
- 更高的 `kalman_q_high` 和 `warmup_frames` 加速初始收斂
- Filter 在 DT 開始前收斂得更好 → DT 期間殘留 echo 更少
- 更高的 `shadow_mu_min` 讓 filter 在 DT 期間也持續積極更新

### Trade-off 機制

| 參數 | 增大效果 | 副作用 |
|------|----------|--------|
| kalman_q_high ↑ | 收斂更快 | 穩態噪聲略增 |
| warmup_frames ↑ | 初始收斂更完整 | 延長高 mu 期間 |
| shadow_mu_min ↑ | DT 時更積極消 echo | 近端語音受汙染 |

AGGRESSIVE 是推薦的平衡點：FS echo_mos 3.30、DT echo_mos 3.21、DT deg_mos 3.29。

---

## 驗證方式

```bash
cd /path/to/AEC

# ERLE + PESQ — 全部 preset
python3 python/eval_aec_challenge.py wav/aec_challenge/ --aec3 --speex --all-presets

# AECMOS — 全部 preset
/opt/homebrew/bin/python3 python/eval_aecmos.py wav/aec_challenge/ --all-presets

# 單一 preset
python3 python/eval_aec_challenge.py wav/aec_challenge/ --preset aggressive
```

---

## 後續方向

- FS echo_mos 與 AEC3 仍有 ~1.1 差距（3.30 vs 4.47），主要受限於無 NLP
- NN-based NLP 預期可進一步大幅提升 echo_mos
- C 版本需同步更新 preset adaptive 層參數
