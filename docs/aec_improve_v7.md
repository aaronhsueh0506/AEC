# AEC v1.14.0 改善紀錄

## 概述

基於 v1.13.0（Preset adaptive filter 層）架構，新增 per-bin near-end gate 取代 broadband dt_indicator，解決 FS 時 echo 被誤判為 DT 導致 g_min 被抬高的問題。同時微調 preset 參數。

---

## 動機

v1.13.0 的 RES 使用 broadband `dt_indicator` 控制 g_min：

```python
dt_indicator = clip(1 - far_pwr / (mic_pwr + far_pwr), 0, 0.8)
effective_g_min = g_min + (dt_g_min - g_min) * dt_indicator
```

**問題**：FS 時 `mic ≈ far`（echo 衰減小）→ `dt_indicator ≈ 0.5` → g_min 被抬到接近 -10dB → echo-only 時段無法激進壓制。

這不是「缺少 hard suppression」，而是**自己的 DT 偵測器在 FS 時誤觸發**。

## 核心改動：Per-bin Near-end Gate

### 原理

利用已有的 per-bin `coh2`（far-end ↔ error coherence²）取代 broadband `dt_indicator`：

| coh2 值 | 含義 | g_min 行為 |
|---------|------|-----------|
| ≈ 1.0 | bin k 與 far-end 高度相關 → echo | 允許壓到 `res_g_min_db`（激進壓制） |
| ≈ 0.0 | bin k 與 far-end 不相關 → near-end | 提高 floor 到 `ne_protect_db`（保護） |

### 實作

```python
# 移除 broadband dt_indicator → g_min 的影響
# 改用 per-bin coh2 gate
ne_protection = (1.0 - coh2) * erle_factor    # [n_freqs]
ne_g_min_ceil = 10 ** (self.ne_protect_db / 20)
ne_g_floor = effective_g_min + (ne_g_min_ceil - effective_g_min) * ne_protection
ne_g_floor = np.maximum(ne_g_floor, effective_g_min)
spectral_g_min = np.maximum(spectral_g_min, ne_g_floor)
```

- `erle_factor` gate：未收斂時 `ne_protection=0`，不干擾初始壓制
- Per-bin 精度：DT 時，echo bin 仍被壓制，near-end bin 被保護

### coh2 收斂後 alpha 調整

收斂後 rise alpha 更保守，coh2 在 echo-only 段更穩定：

```python
if filter_converged:
    a_coh_rise = 0.90   # TC≈160ms（穩定 echo tracking）
    a_coh_drop = 0.50   # TC≈25ms（快速 DT protection）
else:
    a_coh_rise = 0.80
    a_coh_drop = 0.50
```

---

## Preset 參數微調

### 新增參數

| 參數 | 預設值 | 說明 |
|------|--------|------|
| `res_ne_protect_db` | -10.0 | Per-bin 近端保護上限 (dB) |

### Preset 設定

| 參數 | BALANCED | AGGRESSIVE | MAXIMUM |
|------|----------|------------|---------|
| `res_ne_protect_db` | -8 | -5 | -2 |
| `res_dt_reduction` | 3.5 | 2.0 | 0.0 |

其他參數不變（見 v1.13.0）。

---

## Benchmark 結果

### Farend Single-Talk（15 cases, AEC Challenge）

| Preset | echo_mos | ERLE | vs v1.13.0 |
|--------|----------|------|------------|
| BALANCED | 3.18 | 11.9 dB | ≈ baseline |
| AGGRESSIVE | **3.36** | **14.7 dB** | +0.06 |
| MAXIMUM | 3.41 | 13.9 dB | +0.03 |
| SpeexDSP | 3.09 | 6.9 dB | — |
| WebRTC AEC3 | 4.47 | 25.8 dB | — |

### Doubletalk（15 real cases）

| Preset | echo_mos | deg_mos | ERLE | vs v1.13.0 echo_mos |
|--------|----------|---------|------|---------------------|
| BALANCED | 3.06 | **3.51** | 4.4 dB | +0.04 |
| AGGRESSIVE | **3.42** | 3.22 | **7.4 dB** | **+0.21** |
| MAXIMUM | 3.59 | 2.59 | 8.2 dB | +0.26 |
| SpeexDSP | 2.76 | 3.90 | 1.3 dB | — |
| WebRTC AEC3 | 4.39 | 2.51 | 3.7 dB | — |

---

## 分析

### 為何 DT echo_mos 大幅改善

舊版 broadband `dt_indicator` 在 FS 和 DT 都把 g_min 抬到 -10dB → echo-only bin 也被限制。

新版 per-bin gate：
- **FS**：coh2 高 → ne_protection≈0 → g_min 不受限 → 壓制更深
- **DT**：echo bin coh2 高 → 仍壓制；near-end bin coh2 低 → 保護

結果：echo bin 在任何場景都被有效壓制，DT echo_mos +0.21。

### 為何 DT deg_mos 幾乎不變

Per-bin 精度：near-end 佔主導的 bin 的 coh2 低 → ne_g_floor 接近 `ne_protect_db` → 近端語音被保護。

BALANCED/AGGRESSIVE 的 deg_mos 變化 < 0.1。MAXIMUM 因 `ne_protect_db=-2dB` 太弱，deg_mos 降到 2.59（但仍 > AEC3 2.51）。

### DT ERLE 碾壓 AEC3

AGGRESSIVE DT ERLE 7.4 dB vs AEC3 3.7 dB（+3.7 dB）。原因：
1. Per-bin gate 允許 echo bin 激進壓制
2. FDKF 收斂速度優勢（kalman_q_high=3e-3 + warmup=150）
3. AEC3 的 NLP 在 DT 時退縮，我們的 per-bin gate 只退縮 near-end bin

---

## 驗證方式

```bash
cd /path/to/AEC

# ERLE + PESQ — 全部 preset
python3 python/eval_aec_challenge.py wav/aec_challenge/ --aec3 --speex --all-presets

# AECMOS — 全部 preset
/opt/homebrew/bin/python3 python/eval_aecmos.py wav/aec_challenge/ --all-presets
```

---

## 後續方向

- FS echo_mos 與 AEC3 仍有 ~1.1 差距（3.36 vs 4.47），需 NN-based NLP
- C 版本需同步更新 per-bin gate + preset 參數
