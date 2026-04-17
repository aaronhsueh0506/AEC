# Linear AEC 之後：Residual Echo Suppression (RES)

**範疇**：adaptive filter 收斂之後，RES 在做什麼、為什麼需要。
**對象**：內部技術分享 / 外部 presentation。詳細實作見 [aec_methods.md](aec_methods.md)。

---

## 1. Linear AEC 做完後還剩什麼？

自適應濾波器（PBFDKF）對 echo path 建模，理想情況：

```
error = mic - filter × ref   ≈ 0  (FS) 或 ≈ near_speech (DT)
```

實際情況下，error 裡還剩 4 類殘留：

| 類型 | 來源 | 為什麼線性 AEC 壓不掉 |
|------|------|---------------------|
| **Residual linear echo** | filter 未完全收斂、echo path 變化 | 模型誤差（天花板：30-40 dB ERLE）|
| **Non-linear echo** | speaker 失真、clipping、AGC | 線性模型無法建模 |
| **Late reverberation** | 房間反射尾巴（> filter length） | 模型容量不夠 |
| **Filter divergence** | DT 期間錯誤收斂、delay shift | 收斂慢於語音瞬變 |

→ 這些靠 **RES post-filter** 壓制。

---

## 2. RES 的核心決策

在每個頻帶 $k$ 每個時刻 $n$ 回答：

> **這個 bin 的能量有多少來自 echo vs 近端語音？**

輸出：一個 gain $G(n, k) \in [g_{\min}, 1]$。

```
final_output[k] = error[k] × G(n, k)
```

- $G → 1$：相信近端，不壓
- $G → g_{\min}$：認定殘餘 echo，大幅衰減

---

## 3. 架構總覽

```
                       ┌──────────────── residual_echo_psd ────┐
                       │                                        ↓
error_spec ──→ [PSD] ──┼──→ [Direct Method]                  [Wiener Gain]
                       │    echo_psd × per_bin_ERLE              ↓
                       └──→ [Render Method]                    G(n,k)
                            far_psd × ERL_estimate               │
                                                                 ↓
                                                              [Modifiers]
                                                       ENR masking, g_min,
                                                       reverb, DT protect,
                                                       CNG, rate limit
                                                                 ↓
                                                              output
```

**兩層設計**：先算基本 Wiener gain，再用一連串 modifiers 修正。

---

## 4. 殘餘 echo PSD 估計（最關鍵）

分成兩條路，動態切換：

### 4.1 Direct Method（filter 已收斂）

```
residual_echo_psd = echo_psd_from_filter × (1 / per_bin_ERLE)
```

- `echo_psd_from_filter`：線性 filter 輸出的 echo 估計能量
- `per_bin_ERLE`：每個頻帶當前的 echo reduction 能力（由 `echo_spec² / error_spec²` 估）

**好處**：直接反映 filter 當前狀態。
**風險**：filter 未收斂時 `echo_spec` 不可信。

### 4.2 Render Method（filter 未收斂 / EPC 後）

```
residual_echo_psd = far_psd × ERL_estimate
```

- `ERL_estimate`：從 `mic_pwr / far_pwr` 動態學到的實際耦合（Bug 6 v2.4.0 改為動態 ceiling）
- 不依賴 filter 品質，只要知道 ref 就能給出保守估計

**切換機制**：`erle_factor = clip(ERLE / 10 dB, 0, 1)`
- `erle_factor → 0`：完全用 render method
- `erle_factor → 1`：完全用 direct method

---

## 5. 修正器（Modifiers）

### 5.1 ENR Masking（Echo-to-Nearend Ratio）

核心 gate：

```
ENR = residual_echo_psd / nearend_est
G   = shaped(ENR; threshold_t, slope_s)
```

- `nearend_est = max(error_psd - residual_echo_psd, dt_shaped × error_psd)`
- DT 情況下，`dt_per_bin` 驅動 `nearend_est` 抬高 → ENR 降低 → 較不壓制
- **Bug 7（v2.4.0）**：`dt_per_bin ** 2` → `** 1.3`，修正 mid-DT 過度壓制

### 5.2 Dynamic g_min

```
g_min_eff = g_min × far_activity  +  (1 - far_activity)
```

- 遠端沉默：`g_min_eff → 1`（不壓制，避免破壞底噪）
- 遠端活躍：`g_min_eff → g_min`（最大壓制）

### 5.3 Reverb Tail Model

```
reverb_psd[n] = decay × reverb_psd[n-1] + (1-decay) × far_psd[n]
residual_echo_psd += reverb_gain × reverb_psd × reverb_gate
```

- 處理超出 filter length 的後期混響
- `reverb_gate` 依近端活動關閉，避免壓語音

### 5.4 DT 保護路徑（5 條）

每條都接 `effective_dt = max(dt_indicator, energy_dt, shadow_dt)`：

| 路徑 | 作用 |
|------|------|
| `dt_per_bin` 抬 `nearend_est` | ENR 降低 |
| `fs_confidence` 降低 | threshold 放寬 |
| HF cap bypass | 高頻不設上限 |
| `dt_temporal` 驅動 rise/fall α | 上升更快 |
| ENR threshold relaxation | 直接移動 threshold |

### 5.5 Gain 後處理

- **Rate limit**：`max_rise / max_drop`（防 pumping）
- **Spectral-shape-preserving floor**：保留語音頻譜形狀
- **CNG**：遠端活躍時用追蹤的底噪填充極靜音段

---

## 6. 為什麼要 Multi-ERLE？

單一 ERLE 會有循環依賴：

```
ERLE = near_psd / error_psd
但 near_psd = error_psd - residual_echo_psd
     residual_echo_psd = echo_psd / ERLE   ← 循環
```

解法：兩個 ERLE estimator 並行

| ERLE | 計算方式 | 特性 |
|------|---------|------|
| **FilterErle** | `echo_spec² / error_spec²` | per-bin，不依賴 near_psd，打破循環 |
| **FullbandErle** | `near_psd / error_psd` broadband | 穩定但慢，FS-only 更新 |

用 `confidence = f(filter_erle_mean, fullband_erle)` 加權混合。

**Bug 2（v2.4.0）**：修正 FilterErle α 在 DT 期間反向加速更新的漏洞。

---

## 7. Preset — 壓制強度調整

四組預設，主要差在 RES 參數：

| Preset | `over_sub_base` | `reverb_gain` | `g_min` | 場景 |
|--------|----------------|--------------|---------|------|
| mild | 1.2 | 0.8 | -20 dB | 高品質通話 |
| **balanced** | 2.5 | 1.4 | -30 dB | 一般通話（預設）|
| aggressive | 3.5 | 2.0 | -35 dB | 車用、高噪 |
| maximum | 5.0 | 3.0 | -40 dB | 強回音 IoT |

**設計哲學**：所有 DT 保護邏輯四個 preset 共用；只有壓制強度逐漸提高。

---

## 8. 成績（AEC Challenge 2021 blind set, 800 cases, AECMOS）

| Preset | FS echo ↑ | DT echo ↑ | DT deg ↑ | NE deg ↑ |
|--------|----------|----------|---------|---------|
| Linear only | 2.71 | 3.16 | 3.23 | — |
| mild | 3.16 | 3.88 | **2.44** | **4.02** |
| **balanced (v2.4.0)** | **3.49** | 4.16 | 2.29 | 4.01 |
| aggressive | 3.62 | 4.29 | 2.23 | 3.99 |
| maximum | 3.72 | 4.41 | 2.16 | 3.96 |
| _WebRTC AEC3_ | 3.88 | 4.54 | 1.85 | 3.45 |
| _WebRTC AEC2_ | 3.48 | 4.26 | 2.39 | 4.10 |

重點：
- balanced FS echo 接近 AEC3，DT deg 大幅領先（+0.44）
- NE deg 全面領先 AEC3（+0.51~0.57）

---

## 9. 已知限制 / 失效場景

| 場景 | 現象 | 處置 |
|------|------|------|
| Cold start 前 1-2 s | echo 壓制不足 | 走 render method 保守估計 |
| 高耦合 small device | DT echo leak ~1-2 dB | 提高 preset |
| 穩態 far-end + 弱語音 | 近端語音被吸收 | 應用層 VAD + ducking |
| 快速設備移動 | EPC 重新收斂 ~200 ms | 自動恢復 |
| 遠端語音瞬斷 | 100-200 ms CNG 過渡 | 正常行為 |

---

## 10. 總結

| | 線性 AEC | RES |
|---|---------|-----|
| 目標 | 建模 echo path | 決定 "剩下的能量歸誰" |
| 方法 | Kalman / NLMS / LMS | PSD estimation + Wiener gain + modifiers |
| 依賴 | ref 信號品質 | 線性 AEC 品質 + 多路 DT/ERLE 信號 |
| 失效來源 | 收斂慢、延遲錯、非線性 | 高耦合、穩態 far-end、EPC |

→ RES 是線性 AEC 的**互補層**，不是替代層。兩者合作才達到 AEC Challenge 天花板等級。

---

## 延伸閱讀

- [aec_methods.md](aec_methods.md) — 完整演算法 spec
- [dtd_design.md](dtd_design.md) — DTD 詳細設計
- [CHANGELOG_v2.4.0.md](CHANGELOG_v2.4.0.md) — 最新修正
- [CHANGELOG_v2.3.0.md](CHANGELOG_v2.3.0.md) — v2.3.0 milestones（含大部分 RES 架構定型）
