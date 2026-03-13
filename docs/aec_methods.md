# AEC 方法論：Double-Talk Detection 與 Post-Filter

---

## Part 1: Double-Talk Detection (DTD) 與發散偵測

### 1.1 問題定義

AEC 自適應濾波器在以下情況需要控制更新：
- **Double-Talk (DT)**：近端語音與遠端語音同時存在，繼續更新會將近端語音當 echo 學習，導致濾波器發散
- **Echo Path Change**：回聲路徑突然改變（換裝置、人移動），需要重新收斂
- **Filter Divergence**：任何原因導致 output 品質比 input 更差

#### DTD vs 發散偵測的本質差異

| | DTD | 發散偵測 |
|---|---|---|
| 目標 | 偵測「近端有人在講話」 | 偵測「濾波器壞掉了」 |
| 時機 | 在更新前判斷 | 在更新後檢查 |
| 反應 | 凍結/降低更新速率 | 降低更新速率 + 可能 reset |
| 代表 | Geigel, NCC | WebRTC AEC3 |

實務上兩者效果類似——都是在適當時機降低 step size。

---

### 1.2 經典 DTD 方法

#### Geigel DTD

```
if |d[n]| > threshold × max(|x[n-k]|, k=0..L-1):
    → Double-talk detected
```

- 計算量極小，不依賴濾波器狀態
- **❌ 不適用於 AEC**：設計用於線路回聲消除 (LEC)，ERL ≈ -6dB。在 AEC 中 echo gain ≈ 1.0，`|mic|` 幾乎永遠 > `threshold × |ref|`，導致 100% 假觸發

#### Normalized Cross-Correlation (NCC)

```
ξ = |corr(e, x)| / (||e|| × ||x||)
ξ < threshold → Double-talk
```

- 比 Geigel 適合 AEC，但需要長窗口、且依賴濾波器已部分收斂（循環依賴）
- threshold 難調

#### Error-to-Echo Ratio

```
ratio = E[e²] / E[ŷ²] > threshold → Divergence
```

- 嚴重循環依賴：ratio 依賴濾波器品質，濾波器品質依賴 DTD
- 靜音→語音轉場時 ratio 瞬間飆升（假觸發）
- **❌ 不建議作為主要 DTD**

---

### 1.3 現代方法

#### WebRTC AEC3 — 發散偵測 + Step-size Bounding ✅ 本專案採用

**核心思想**：不做明確 DTD，改為偵測 output 是否比 input 更差

```
energy_ratio = mean(output²) / mean(near²)
peak_ratio = max(|output|) / max(|near|)
ratio = max(energy_ratio, peak_ratio)

if ratio > divergence_factor:     confidence += attack
elif ratio > 1.0:                 confidence += attack × (ratio - 1.0)
else:                             confidence -= release
```

**Confidence → mu_scale**：
```
mu_scale = 1.0 - confidence × (1.0 - mu_min_ratio)
# confidence=0 → mu_scale=1.0（正常）
# confidence=1 → mu_scale=0.05（幾乎凍結，保留 5% 微量更新）
```

**Output Limiter（安全網）**：
```
if max(|output|) > max(|near|):
    output *= max(|near|) / max(|output|)
```

**參數**：

| 參數 | 值 | 說明 |
|------|-----|------|
| `divergence_factor` | 1.5 | output > 1.5× input → 發散 |
| `confidence_attack` | 0.3 | 上升速率 |
| `confidence_release` | 0.05 | 下降速率（慢放） |
| `mu_min_ratio` | 0.05 | 最低 mu 比例 |
| `warmup_frames` | 200 (freq) / 50 (time) | 初始收斂期不偵測 |

**優點**：不依賴濾波器品質、幾乎無假觸發、連續 mu scaling、output limiter 兜底

**缺點**：被動（要等到 output 超過才反應），但 output limiter 補償

#### SpeexDSP — 雙濾波器法

```
Background filter (BG): 永遠更新，較小 step size
Foreground filter (FG): 用於實際輸出

每個 block:
  1. BG 更新權重
  2. if MSE(BG) < MSE(FG) × factor:
         FG ← BG
```

- ✅ 最穩健——BG 壞掉不影響 FG
- ✅ 自動處理 echo path change
- ❌ 2× 計算量和記憶體
- ❌ BG→FG 轉移可能不連續

#### Coherence-based DTD (MSC)

利用 Magnitude Squared Coherence `C_de(f)` 判斷近端語音 presence。
- Per-bin DTD，理論紮實
- 需要長窗口估計，不適合短延遲場景

#### 方法比較

| 方法 | 計算量 | 適用 AEC | 準確度 | 穩健性 | 本專案 |
|------|--------|----------|--------|--------|--------|
| Geigel | 極低 | ❌ | 低 | 低 | — |
| NCC | 中 | △ | 中 | 中 | — |
| Error/Echo Ratio | 低 | ❌ | 低 | 低 | — |
| **WebRTC 發散偵測** | **低** | **✅** | **高** | **高** | **✅ 採用** |
| 雙濾波器 (SpeexDSP) | 高 (2×) | ✅ | 最高 | 最高 | — |
| MSC Coherence | 中-高 | ✅ | 高 | 中 | — |

---

## Part 2: AEC Post-Filter 方法

### 2.1 問題定義

自適應濾波器的 echo cancellation 通常只能達到 15-25 dB ERLE。殘餘回聲 (residual echo) 仍然可聞，需要 post-filter 進一步抑制。

Post-filter 的目標：
- 抑制殘餘回聲（額外 10-20 dB）
- 保持近端語音品質（不引入 distortion）
- 不引入 musical noise

---

### 2.2 Residual Echo Suppression (RES) — Spectral Subtraction ✅ 本專案採用

**原理**：在頻域估計殘餘回聲 spectrum，從 error spectrum 中減去

```
Ŝ_echo(f) = |Ŷ(f)|²                     # echo estimate power spectrum
G(f) = max(1 - α × Ŝ_echo(f) / |E(f)|², G_min)
output(f) = G(f) × E(f)
```

**參數**：
| 參數 | 說明 | 典型值 |
|------|------|--------|
| `alpha` (over-subtraction) | 殘餘回聲估計的放大倍數 | 1.0-2.0 |
| `G_min` | 最小增益（防止過度抑制） | -20 dB ~ -30 dB |
| `smoothing` | 頻譜平滑係數 | 0.7-0.9 |

**優點**：
- ✅ 計算量低（已在頻域，直接乘 gain）
- ✅ 和 PBFDAF 天然整合（echo spectrum 直接可用）
- ✅ 參數少，容易調整

**缺點**：
- Musical noise（可用 smoothing 緩解）
- 過度抑制時近端語音 distortion
- 依賴 echo estimate 品質

**本專案狀態**：`res_filter.c` / `ResFilter` class 已實作，但 AEC 主流程中尚未啟用（TODO）

---

### 2.3 Wiener Filter Post-Filter

**原理**：根據 SNR 估計計算最優 Wiener gain

```
SNR(f) = S_speech(f) / S_noise+echo(f)
G(f) = SNR(f) / (1 + SNR(f))
```

等同於 MMSE 最優解（最小化 MSE）。

**實作方式**：
- 需要估計 speech spectrum 和 residual echo + noise spectrum
- 通常使用 decision-directed approach（Ephraim-Malah）估計 a priori SNR

**優點**：
- ✅ 理論上最優（MMSE sense）
- ✅ Musical noise 比 spectral subtraction 少

**缺點**：
- 需要可靠的 noise + echo power 估計
- 計算量比 spectral subtraction 高
- 參數較多

#### WebRTC AEC3 的做法

WebRTC AEC3 使用類似 Wiener 的 comfort noise injection：
```
G(f) = 1 - echo_spectrum(f) / (error_spectrum(f) + echo_spectrum(f))
```
並加入 comfort noise 填補被抑制的頻帶，避免 "dead silence" 不自然感。

---

### 2.4 Coherence-based Post-Filter

**原理**：利用 mic 和 reference 之間的 coherence 估計回聲成分

```
C_xd(f) = |S_xd(f)|² / (S_xx(f) × S_dd(f))

G(f) = 1 - C_xd(f)
# coherence 高 → 回聲成分大 → 抑制
# coherence 低 → 近端語音或雜訊 → 保留
```

**優點**：
- 不依賴自適應濾波器的 echo estimate
- 理論基礎紮實

**缺點**：
- 需要長窗口估計 coherence（延遲）
- 近端和遠端同時有語音時 coherence 估計不準
- 計算量中-高

---

### 2.5 Non-Linear Processing (NLP)

**原理**：在 ERLE 不足時進行非線性抑制

```
if ERLE < target_ERLE:
    # 對殘餘回聲頻帶做硬抑制或 soft gating
    G(f) = comfort_level    # e.g., -40 dB
```

常見於商用 AEC（如 Qualcomm Fluence、Apple AEC）：
- 檢測 "echo only" 段落（near-end VAD off + far-end active）
- 在這些段落做激進抑制（接近 mute）
- 在 double-talk 段落不做 NLP（避免 clipping 近端語音）

**優點**：
- ✅ 可以達到 > 40 dB ERLE
- ✅ Echo 幾乎完全聽不到

**缺點**：
- ❌ 需要準確的 near-end VAD
- ❌ Double-talk 時可能損害近端語音
- ❌ 轉場（echo-only → double-talk）容易有 artifact

---

### 2.6 Post-Filter 方法比較

| 方法 | 計算量 | 抑制量 | 語音品質 | Musical Noise | 本專案 |
|------|--------|--------|----------|---------------|--------|
| **RES (Spectral Sub.)** | **低** | **中 (10-20dB)** | **中-高** | **中** | **✅ 已實作** |
| Wiener Filter | 中 | 中-高 | 高 | 低 | — |
| WebRTC (Wiener+CN) | 中 | 中-高 | 高 | 低 | — |
| Coherence-based | 中-高 | 中 | 高 | 低 | — |
| NLP (hard suppress) | 低 | 極高 (>40dB) | 低-中 | 無 | — |

---

### 2.7 本專案 Post-Filter 狀態

**已實作**：
- `ResFilter` class（Python `aec.py`）— spectral subtraction based RES
- `res_filter.c`（C）— 同上

**尚未啟用**：
- `aec_process()` 中 RES 標記為 TODO
- 需要從 PBFDAF 取得 echo spectrum 和 error spectrum 才能計算 gain

**未來改進方向**：
1. 啟用 RES post-filter（需要 spectrum access refactoring）
2. 考慮 Wiener + comfort noise（如果 musical noise 成為問題）
3. Echo-only 段落的 NLP 可進一步提升 ERLE

---

## 參考資料

- WebRTC AEC3: `webrtc/modules/audio_processing/aec3/`
- SpeexDSP: `speexdsp/libspeex/mdf.c`
- Haykin, "Adaptive Filter Theory", Chapter 13
- Benesty et al., "Advances in Network and Acoustic Echo Cancellation"
- ITU-T G.168: Digital Network Echo Cancellers
- Ephraim & Malah, "Speech Enhancement Using a Minimum Mean-Square Error Short-Time Spectral Amplitude Estimator", IEEE 1984
- Valin, "On Adjusting the Learning Rate in Frequency Domain Echo Cancellation with Double-Talk", IEEE 2007
