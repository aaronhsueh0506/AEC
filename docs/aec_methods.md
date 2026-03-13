# AEC 方法介紹文檔

## 目錄

1. [自適應濾波器演算法總覽](#1-自適應濾波器演算法總覽)
2. [LMS - 最小均方演算法](#2-lms---最小均方演算法)
3. [NLMS - 歸一化最小均方演算法](#3-nlms---歸一化最小均方演算法)
4. [頻域 NLMS](#4-頻域-nlms)
5. [PBFDAF - 分區塊頻域自適應濾波器](#5-pbfdaf---分區塊頻域自適應濾波器)
6. [DTD / 發散偵測](#6-dtd--發散偵測)
7. [RES Post-Filter - 殘餘回聲抑制後濾波器](#7-res-post-filter---殘餘回聲抑制後濾波器)
8. [Post-Filter 方法比較](#8-post-filter-方法比較)
9. [Subband 分頻方式說明](#9-subband-分頻方式說明)
10. [NR Subband 適用性分析](#10-nr-subband-適用性分析)

---

## 1. 自適應濾波器演算法總覽

| 方法 | 模式 | 更新公式 | 複雜度 | 典型 μ | 適用場景 |
|------|------|---------|--------|--------|---------|
| LMS | `AEC_MODE_LMS` | `w += μ·e·x` | O(N) | 0.001~0.05 | 穩態環境、極低資源 |
| NLMS | `AEC_MODE_NLMS` | `w = leak·w + μ/(‖x‖²+δ)·e·x` | O(N) | 0.1~0.8 | 一般用途、低延遲 |
| 頻域 NLMS | `AEC_MODE_FREQ` | `W += μ/(P+δ)·E·conj(X)` | O(N log N) | 0.1~0.8 | 中等 echo path |
| PBFDAF | `AEC_MODE_SUBBAND` | 多 partition 頻域更新 | O(N log N) | 0.1~0.8 | 長 echo path (300ms+) |

### 信號流程

```
far_end (ref) ─────┐
                    ├──→ [Adaptive Filter] ──→ echo_estimate (ŷ)
near_end (mic) ─┐  │                              │
                │  │                              │
                ▼  │                              ▼
                └──┤    error = near_end - echo_estimate
                   │                              │
                   │         ┌────────────────────┘
                   │         │
                   │    [Divergence Detection]
                   │         │
                   │    confidence ──→ mu_scale = 1.0 - conf × (1.0 - min_ratio)
                   │                        │
                   │                   [Weight Update w/ mu_scale]
                   │
                   │    [Output Limiter: out ≤ mic]
                   │
                   └──→ [RES Post-Filter] ──→ output (echo-cancelled)
```

---

## 2. LMS - 最小均方演算法

### 原理

LMS (Least Mean Squares) 是最簡單的自適應濾波演算法，由 Widrow & Hoff (1960) 提出。
使用固定步長 μ 直接沿梯度方向更新權重。

### 公式推導

**回聲估計：**
```
ŷ[n] = Σ(k=0 to L-1) w[k] · x[n-k] = w^T · x[n]
```

**誤差信號：**
```
e[n] = d[n] - ŷ[n]
```
其中 d[n] 為近端（麥克風）信號，ŷ[n] 為回聲估計。

**成本函數：**
```
J(w) = E[e²[n]] = E[(d[n] - w^T·x[n])²]
```

**梯度：**
```
∇J ≈ -2·e[n]·x[n]    (瞬時梯度估計)
```

**權重更新：**
```
w[n+1] = w[n] + μ · e[n] · x[n]
```

### 參數

| 參數 | 典型值 | 說明 |
|------|--------|------|
| μ (step size) | 0.001~0.05 | 步長，越大收斂越快但越不穩定 |
| filter_length | 512 (32ms@16kHz) | 濾波器長度 (samples)，可配置 |

### 穩定性條件

```
0 < μ < 2 / (L · σ_x²)
```
其中 L 為濾波器長度，σ_x² 為輸入功率。因為 μ 固定且不隨輸入功率調整，
在輸入能量變化大的場景容易不穩定，需要使用很小的 μ。

### 優缺點

**優點：**
- 計算最簡單，每次更新 O(N)
- 實現容易，無需除法運算
- 適合極低資源嵌入式環境

**缺點：**
- 收斂速度慢（固定步長）
- 對輸入信號功率敏感，μ 需根據信號強度調整
- 在信號能量變化大的場景容易發散

### 本專案實現

在 `nlms_filter.c` 中，LMS 和 NLMS 共用同一套程式碼，透過 `normalize` 旗標切換：

```c
if (filter->normalize) {
    mu_eff = (filter->mu * mu_scale) / (filter->power_sum + filter->delta);  // NLMS
} else {
    mu_eff = filter->mu * mu_scale;  // LMS (固定步長)
}
```

LMS 模式下 `leak = 1.0`（無權重洩漏）。

---

## 3. NLMS - 歸一化最小均方演算法

### 原理

NLMS (Normalized LMS) 在 LMS 基礎上加入功率歸一化，使步長自動隨輸入功率調整。
這是 AEC 最常用的時域演算法。

### 公式

**功率歸一化：**
```
‖x[n]‖² = Σ(k=0 to L-1) x²[n-k]
mu_eff = μ / (‖x[n]‖² + δ)
```

**權重更新（含洩漏）：**
```
w[n+1] = leak · w[n] + mu_eff · e[n] · x[n]
```

### 參數

| 參數 | 典型值 | 說明 |
|------|--------|------|
| μ (step size) | 0.3 | 歸一化後的步長 |
| δ (regularization) | 1e-8 | 防止除零 |
| leak | 0.9999 | 權重洩漏，防止權重爆炸 |
| filter_length | 512 (32ms@16kHz) | 濾波器長度 (samples)，可配置，circular buffer 跨 block 保留歷史 |

### 穩定性條件

```
0 < μ < 2    (歸一化後穩定範圍大幅增加)
```

### 功率估計優化

使用滑動窗口的增量更新避免每次重算：
```
power_sum += x_new² - x_old²
```

### 與 LMS 比較

| | LMS | NLMS |
|---|---|---|
| 步長 | 固定 μ | μ / (‖x‖² + δ) |
| 收斂速度 | 慢 | 快 |
| 穩定性 | 差（依賴信號功率） | 好（自動適應） |
| 計算量 | O(N) | O(N) + 功率計算 |
| 典型 μ | 0.001~0.05 | 0.1~0.8 |

### 優缺點

**優點：**
- 收斂速度快，對信號功率變化自適應
- 延遲最低（逐樣本處理）
- 實現簡單

**缺點：**
- 計算複雜度 O(N²)（每個樣本需遍歷所有權重）
- 不適合長 echo path（filter_length > 5000），計算量 O(filter_length × hop_size) per block

---

## 4. 頻域 NLMS

### 原理

將 NLMS 搬到頻域，利用 FFT 實現高效的濾波和更新。
使用 Overlap-Save 方法，單一 FFT block 處理。

### 公式

**回聲估計（頻域）：**
```
Ŷ[k] = W[k] · X[k]    (逐 bin 乘法)
```

**功率估計（平滑）：**
```
P[k] = α_p · P[k] + (1 - α_p) · |X[k]|²
```

**權重更新（頻域）：**
```
mu_eff[k] = μ / (P[k] + δ)
W[k] += mu_eff[k] · E[k] · conj(X[k])
```

**時域約束（防止頻域混疊）：**
```
1. w = IFFT(W)
2. w[hop:fft_size] = 0    (截斷後半部)
3. W = FFT(w)
```

### 參數

| 參數 | 典型值 | 說明 |
|------|--------|------|
| μ | 0.3 | 步長 |
| δ | 1e-8 | 正則化 |
| α_power | 0.9 | PSD 平滑因子 |
| fft_size | 512 | FFT 大小 |
| hop_size | 256 (fft_size/2) | 每次處理樣本數 |

### Overlap-Save 流程

```
1. 緩衝區 = [舊半段 | 新半段]  (512 樣本)
2. X = FFT(far_buffer)
3. D = FFT(near_buffer)
4. Ŷ = W · X
5. ŷ = IFFT(Ŷ)
6. e = d[256:512] - ŷ[256:512]  (取後半段有效輸出)
7. E = FFT([0...0 | error])      (零填充前半段)
8. 更新 W
```

### 優缺點

**優點：**
- O(N log N) 計算複雜度
- 頻域逐 bin 歸一化，自然的頻域白化效果

**缺點：**
- 延遲較 NLMS 模式大（hop_size = 256 = 16ms）
- 單一 block 長度限制可覆蓋的 echo path

---

## 5. PBFDAF - 分區塊頻域自適應濾波器

### 原理

Partitioned Block Frequency Domain Adaptive Filter (PBFDAF) 將長濾波器分成多個 partition，
每個 partition 獨立在頻域更新。這是處理長 echo path 最高效的方法。

### 核心概念

```
Echo path = [Partition 0 | Partition 1 | ... | Partition P-1]
             ←─ hop ──→  ←─ hop ──→         ←─ hop ──→

P = ceil(filter_length / hop_size)
  = ceil(1024 / 256)
  = 4 partitions (@ 16kHz, 64ms echo path)
```

### 公式

**回聲估計（多 partition 疊加）：**
```
Ŷ[k] = Σ(p=0 to P-1) W[p][k] · X_buf[p][k]
```
其中 X_buf[p] 是 p 個 hop 之前的遠端頻譜。

**功率估計：**
```
P[k] = α · P[k] + (1-α) · |X[k]|²
```

**權重更新（逐 partition）：**
```
mu_eff[k] = μ / (P[k] + δ)
W[p][k] += mu_eff[k] · E[k] · conj(X_buf[p][k])
```

**時域約束（逐 partition）：**
```
for each partition p:
    w[p] = IFFT(W[p])
    w[p][hop:block] = 0
    W[p] = FFT(w[p])
```

### 參數

| 參數 | 典型值 | 說明 |
|------|--------|------|
| μ | 0.3 | 步長 |
| δ | 1e-8 | 正則化 |
| α_power | 0.9 | PSD 平滑因子 |
| fft_size | 512 | FFT 大小（block size） |
| hop_size | 256 | 每次處理樣本數 |
| n_partitions | 4 | 分區數（64ms / 16ms） |

### 與頻域 NLMS 的區別

| | 頻域 NLMS (FREQ) | PBFDAF (SUBBAND) |
|---|---|---|
| Partition 數 | 1 | P (多個) |
| 最大 echo path | ~16ms | 250ms+ |
| 記憶體 | W[n_freqs] | W[P][n_freqs] |
| 計算量 | O(N log N) | O(P·N log N) |

### 優缺點

**優點：**
- 高效處理長 echo path（300ms+）
- O(N log N) 複雜度（相對時域 O(N²)）
- 頻域白化提供更好的收斂特性

**缺點：**
- 延遲 = hop_size（16ms）
- 記憶體需求較大（P 個 partition 的權重）
- 實現複雜度較高

---

## 6. DTD / 發散偵測

### 6.1 問題定義

AEC 自適應濾波器在以下情況需要控制更新：
- **Double-Talk (DT)**：近端與遠端同時說話，繼續更新會把近端語音當 echo 學習
- **Echo Path Change**：回聲路徑改變，需要重新收斂
- **Filter Divergence**：output 品質比 input 更差

**所有四個模式（LMS、NLMS、FREQ、SUBBAND）都使用發散偵測。**

#### DTD vs 發散偵測

| | 傳統 DTD | 發散偵測（本專案採用） |
|---|---|---|
| 目標 | 偵測「近端有人在講話」 | 偵測「濾波器壞掉了」 |
| 時機 | 在更新前判斷 | 在更新後檢查 |
| 反應 | 凍結更新 | 連續 mu scaling |
| 代表 | Geigel, NCC | WebRTC AEC3 |

### 6.2 經典 DTD 方法（不推薦）

#### Geigel DTD

```
if |d[n]| > threshold × max(|x[n-k]|, k=0..L-1):
    → Double-talk detected
```

- 計算量極小，不依賴濾波器狀態
- **❌ 不適用於 AEC**：設計用於線路回聲消除 (LEC)，ERL ≈ -6dB。在 AEC 中 echo gain ≈ 1.0，`|mic|` 幾乎永遠 > `threshold × |ref|`，導致 100% 假觸發
- C 版本有 Geigel 實作（`dtd.c`），僅供參考

#### Normalized Cross-Correlation (NCC)

```
ξ = |corr(e, x)| / (||e|| × ||x||)
ξ < threshold → Double-talk
```

- 比 Geigel 適合 AEC，但需要長窗口、且依賴濾波器已部分收斂（循環依賴）

#### Error-to-Echo Ratio

```
ratio = E[e²] / E[ŷ²] > threshold → Divergence
```

- 嚴重循環依賴：ratio 依賴濾波器品質，濾波器品質依賴 DTD
- 靜音→語音轉場時 ratio 瞬間飆升（假觸發）
- **❌ 不建議作為主要 DTD**（本專案 v1.2.0 曾使用，v1.3.0 移除）

### 6.3 WebRTC-style 發散偵測 ✅ 本專案採用

**核心思想**：不做明確 DTD，改為偵測 output 是否比 input 更差

```
output_energy = mean(output²)
near_energy = mean(near_end²)
output_peak = max(|output|)
near_peak = max(|near_end|)

energy_ratio = output_energy / (near_energy + ε)
peak_ratio = output_peak / (near_peak + ε)
ratio = max(energy_ratio, peak_ratio)    // 取兩者較大值

if near_energy < 1e-10 and near_peak < 1e-6:
    confidence -= release               // 靜音 → 正常
elif ratio > divergence_factor:
    confidence += attack                 // 嚴重發散
elif ratio > 1.0:
    confidence += attack × (ratio - 1.0) // 輕微發散（比例反應）
else:
    confidence -= release                // 正常
```

**為什麼同時用 energy + peak**：
- Energy-based：捕捉整體發散（mean level）
- Peak-based：捕捉瞬態尖峰（mean 可能正常但個別 sample 超過）

#### Confidence → mu_scale

```
mu_scale = 1.0 - confidence × (1.0 - mu_min_ratio)
# confidence=0 → mu_scale=1.0（正常更新）
# confidence=1 → mu_scale=0.05（幾乎凍結，保留 5% 微量更新）
```

使用連續 mu_scale 而非二元凍結：完全凍結會導致濾波器停滯，5% 最低 mu 保持微量追蹤。

#### Output Limiter（安全網）

```
near_peak = max(|near_end|)
out_peak = max(|output|)
if out_peak > near_peak and near_peak > 1e-6:
    output *= near_peak / out_peak
```

保證 output 永遠不超過 mic amplitude，即使發散偵測來不及反應。

#### Warmup 機制

濾波器剛啟動時需要一段時間收斂，此期間不啟用偵測：

| 模式 | Warmup 幀數 | 時間 (@hop=256, sr=16kHz) |
|------|-------------|--------------------------|
| **LMS / NLMS** | 50 幀 | ~0.8 秒 |
| **FREQ / SUBBAND** | 200 幀 | ~3.2 秒 |

### 6.4 其他現代 DTD 方法

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

### 6.5 DTD 方法比較

| 方法 | 計算量 | 適用 AEC | 準確度 | 穩健性 | 本專案 |
|------|--------|----------|--------|--------|--------|
| Geigel | 極低 | ❌ | 低 | 低 | — |
| NCC | 中 | △ | 中 | 中 | — |
| Error/Echo Ratio | 低 | ❌ | 低 | 低 | v1.2（已移除） |
| **WebRTC 發散偵測** | **低** | **✅** | **高** | **高** | **✅ v1.3 採用** |
| 雙濾波器 (SpeexDSP) | 高 (2×) | ✅ | 最高 | 最高 | — |
| MSC Coherence | 中-高 | ✅ | 高 | 中 | — |

### 6.6 參數

| 參數 | 典型值 | 說明 |
|------|--------|------|
| divergence_factor | 1.5 | output > 1.5× input → 發散 |
| confidence_attack | 0.3 | confidence 上升速率 |
| confidence_release | 0.05 | confidence 下降速率（慢放） |
| mu_min_ratio | 0.05 | 最低 mu 比例（不完全凍結） |
| warmup_frames | 200 (freq) / 50 (time) | 初始收斂期不偵測 |

### 6.7 NLMS 額外保護：權重 Norm 約束

NLMS 還有一個額外的安全機制：

```
w_norm = ||weights||
if w_norm > max_w_norm (預設 4.0):
    weights *= max_w_norm / w_norm
```

防止 warmup 期間的短暫 double-talk 導致權重無限增長。

---

## 7. RES Post-Filter - 殘餘回聲抑制後濾波器

### 目的

自適應濾波器無法完全消除回聲（受限於收斂速度、非線性失真等），
RES post-filter 在頻域對殘餘回聲進行額外 10~20 dB 的抑制。

### 核心公式

**PSD 估計（指數平滑）：**
```
echo_psd[k]  = α_psd · echo_psd[k]  + (1 - α_psd) · |Ŷ[k]|²
error_psd[k] = α_psd · error_psd[k] + (1 - α_psd) · |E[k]|²
```
其中 α_psd = 0.9。

**Echo-to-Error Ratio (EER)：**
```
EER[k] = echo_psd[k] / (error_psd[k] + ε)
```

- EER 高 → 殘餘回聲多 → 需要更多抑制
- EER 低 → 回聲已被消除 → 不需要抑制

**增益計算：**
```
G[k] = 1 / (1 + α_os · EER[k])
```

| EER 值 | G 值 | 含義 |
|--------|------|------|
| 0 | 1.0 | 無回聲，不抑制 |
| 0.5 | 0.57 | 中等抑制 |
| 1.0 | 0.4 | 較強抑制 |
| → ∞ | → 0 | 最大抑制 |

**增益下限（Floor）：**
```
G[k] = max(G[k], g_min)
```
例如 g_min_db = -20 dB → g_min = 0.1，防止過度抑制造成不自然的靜音。

**非對稱時間平滑：**
```
if G[k] < gain_smooth[k]:        // Attack（回聲突然出現）
    α_g = 0.3                    // 快速反應
else:                             // Release（回聲消失）
    α_g = 0.8                    // 緩慢恢復

gain_smooth[k] = α_g · gain_smooth[k] + (1 - α_g) · G[k]
```

- **快攻（attack = 0.3）：** 回聲突然出現時快速壓制
- **慢放（release = 0.8）：** 回聲消失後緩慢恢復，避免音樂噪聲

**遠端靜音自動釋放：**
```
if (far_power < 1e-6):
    G[k] = 1.0    // 遠端無信號時不抑制
```

### 參數

| 參數 | 典型值 | 說明 |
|------|--------|------|
| α_os (over_sub) | 1.5 | 過減因子，越大抑制越強 |
| g_min_db | -20 dB | 最小增益限制 |
| α_psd | 0.9 | PSD 平滑因子 |
| α_attack | 0.3 | 攻擊平滑（快） |
| α_release | 0.8 | 釋放平滑（慢） |

### 輸出

```
E_out[k] = gain_smooth[k] · E[k]
```

### 典型效能

- 語音主導時：G ≈ 1.0（最小影響）
- 回聲突發時：快速壓低（10~20 dB）
- 噪聲環境：G = 1.0（不抑制背景噪聲，那是 NR 的工作）

### 本專案狀態

- `ResFilter` class（Python `aec.py`）和 `res_filter.c`（C）已實作
- `aec_process()` 中 RES 標記為 TODO，需要 spectrum access refactoring 才能啟用

---

## 8. Post-Filter 方法比較

### Wiener Filter Post-Filter

```
SNR(f) = S_speech(f) / S_noise+echo(f)
G(f) = SNR(f) / (1 + SNR(f))
```

- 理論上最優（MMSE sense），musical noise 比 spectral subtraction 少
- 需要可靠的 noise + echo power 估計，計算量較高

WebRTC AEC3 使用類似 Wiener 的做法加 comfort noise injection：
```
G(f) = 1 - echo_spectrum(f) / (error_spectrum(f) + echo_spectrum(f))
```

### Coherence-based Post-Filter

```
C_xd(f) = |S_xd(f)|² / (S_xx(f) × S_dd(f))
G(f) = 1 - C_xd(f)
```

- 不依賴自適應濾波器的 echo estimate，理論紮實
- 需要長窗口估計 coherence

### Non-Linear Processing (NLP)

```
if ERLE < target_ERLE:
    G(f) = comfort_level    # e.g., -40 dB
```

- 商用 AEC 常見（Qualcomm Fluence、Apple AEC）
- 在 "echo only" 段落做激進抑制，double-talk 時不做
- 可達 > 40 dB ERLE，但需要準確的 near-end VAD

### 比較總結

| 方法 | 計算量 | 抑制量 | 語音品質 | Musical Noise | 本專案 |
|------|--------|--------|----------|---------------|--------|
| **RES (Spectral Sub.)** | **低** | **中 (10-20dB)** | **中-高** | **中** | **✅ 已實作** |
| Wiener Filter | 中 | 中-高 | 高 | 低 | — |
| WebRTC (Wiener+CN) | 中 | 中-高 | 高 | 低 | — |
| Coherence-based | 中-高 | 中 | 高 | 低 | — |
| NLP (hard suppress) | 低 | 極高 (>40dB) | 低-中 | 無 | — |

### 未來改進方向

1. 啟用 RES post-filter（需要 spectrum access refactoring）
2. 考慮 Wiener + comfort noise（如果 musical noise 成為問題）
3. Echo-only 段落的 NLP 可進一步提升 ERLE

---

## 9. Subband 分頻方式說明

### 目前的「SUBBAND」模式

**重要澄清：** 本專案中 `AEC_MODE_SUBBAND` 並非傳統的分析-合成 filter bank 分頻，
而是 **PBFDAF（Partitioned Block Frequency Domain Adaptive Filter）**。

### 頻率分解方式

使用 **FFT 線性等距 bin** 分解：

```
bin 間距 = sample_rate / fft_size

頻率範圍 = [0, sample_rate/2]
bin 數量 = fft_size/2 + 1
```

| Sample Rate | FFT Size | bin 間距 | bin 數量 | 頻率範圍 |
|------------|----------|---------|---------|---------|
| 8 kHz | 256 | 31.25 Hz | 129 | 0~4000 Hz |
| 16 kHz | 512 | 31.25 Hz | 257 | 0~8000 Hz |
| 48 kHz | 1024 | 46.88 Hz | 513 | 0~24000 Hz |

### 不同 Sampling Rate 的自動適應

frame/hop 使用 samples 為單位定義，根據 sample rate 自動計算（next power of 2 >= ~32ms）：

| Sample Rate | frame_size (samples) | hop_size (samples) | FFT size | 對應毫秒 |
|------------|---------------------|-------------------|----------|---------|
| 8 kHz | 256 | 128 | 256 | 32ms / 16ms |
| 16 kHz | 512 | 256 | 512 | 32ms / 16ms |
| 48 kHz | 1024 | 512 | 1024 | ~21ms / ~11ms |

**關鍵設計：** frame_size = FFT size，無需零填充，50% overlap。

### 為什麼不用 Filter Bank？

傳統 subband AEC 使用 QMF、DFT filter bank 等將信號分成多個子帶，
每個子帶獨立運行低 sample rate 的自適應濾波器。

本專案選擇 PBFDAF 的原因：
1. **效能等價：** PBFDAF 在頻域逐 bin 處理，等同於最細粒度的 subband
2. **實現簡潔：** 不需要 filter bank 的分析-合成架構
3. **無 aliasing 問題：** FFT 是完美重建的
4. **參數簡單：** 只需 fft_size、hop_size，不需 band 數、prototype filter 等

---

## 10. NR Subband 適用性分析

### 問題：NR 是否也適用 Subband 處理？

### 目前 NR 的處理方式

NR 已經是 **per-bin（逐頻率 bin）處理**，這是最細粒度的 subband：

```python
# NR 逐 bin 處理
for k in range(n_freqs):  # 257 bins @ FFT=512
    gain[k] = f(SNR[k], noise_psd[k], ...)
    enhanced[k] = gain[k] * noisy[k]
```

每個 frequency bin 有獨立的：
- 噪聲 PSD 估計
- SNR 估計
- 增益計算

### 如果改用 Bark/MEL Subband？

將 257 個 bin 分組為 ~24 個 Bark band 或 ~40 個 MEL band：

```
Bark Band 1: bins 0-2     (0~94 Hz)
Bark Band 2: bins 3-4     (94~188 Hz)
...
Bark Band 20: bins 100-150 (3125~4688 Hz)
...
```

每組 band 共享同一個增益值。

### 比較分析

| | Per-Bin (目前) | Bark/MEL Subband |
|---|---|---|
| **頻率解析度** | 31.25 Hz/bin | 低頻高、高頻低 |
| **增益數量** | 257 | ~24 (Bark) / ~40 (MEL) |
| **計算量** | 257 次增益計算 | ~24~40 次 |
| **精確度** | 最高 | 較低（同 band 共用增益） |
| **音樂噪聲** | 可能較多 | 較少（band 內平均效果） |
| **語音保留** | 最佳 | 可能模糊相鄰頻率的語音 |

### 結論

**Per-bin 處理更精確，建議維持現狀。**

原因：
1. NR 的增益計算（MMSE/LSA）計算量不大，257 bin vs 24 band 的節省有限
2. 瓶頸在 FFT（O(N log N)），不在增益計算
3. Per-bin 提供最精確的噪聲估計和增益控制
4. Bark/MEL 分組會降低頻率解析度，可能影響語音品質

**唯一考慮改用 Subband 的場景：**
- 計算資源極度受限（MCU 等級）
- 可接受品質下降以換取計算量減少
- 此時建議使用 MEL scale（~40 bands），兼顧感知特性

---

## 附錄 A: 效能指標

### ERLE (Echo Return Loss Enhancement)

```
ERLE = 10 · log₁₀(E[d²] / E[e²])    (dB)
```

| ERLE 範圍 | 品質評價 |
|----------|---------|
| < 10 dB | 差，明顯回聲 |
| 10~20 dB | 可接受 |
| 20~30 dB | 良好 |
| > 30 dB | 優秀 |

### 收斂時間

| 模式 | 典型收斂時間 |
|------|------------|
| LMS | 5~15 秒 |
| NLMS | 2~5 秒 |
| FREQ NLMS | 2~5 秒 |
| PBFDAF | 1~3 秒 |

### 延遲

| 模式 | 延遲 (16kHz) |
|------|-------------|
| LMS/NLMS | 16 ms (hop=256) |
| FREQ | 16 ms (hop=256) |
| SUBBAND (PBFDAF) | 16 ms (hop=256) |

---

## 附錄 B: 參數調整指南

| 參數 | 影響 | 調整方向 |
|------|------|---------|
| μ (step size) | 收斂速度 vs 穩定性 | 不穩定→降低；收斂慢→提高 |
| filter_length | echo path 覆蓋 | TIME/LMS/SUBBAND 可配置（預設 512/1024），FREQ 固定=hop_size |
| δ (regularization) | 數值穩定性 | 保持 1e-8 |
| leak | 權重衰減（TIME only） | 接近 1.0，防止權重爆炸 |
| divergence_factor | 發散靈敏度 | 降低→更敏感（更快降 mu）；提高→更寬鬆 |
| mu_min_ratio | 最低更新量 | 不建議設為 0（停滯）；0.05 適合大部分場景 |
| res_over_sub | 殘餘回聲抑制強度 | 提高→更強抑制，風險失真 |
| res_g_min_db | 最大抑制量 | 降低→更深抑制，-20dB 通常夠用 |

---

## 參考文獻

1. Haykin, S. "Adaptive Filter Theory" (4th Edition)
2. Widrow, B. & Hoff, M.E. "Adaptive switching circuits" (1960) — LMS 原始論文
3. Sondhi, M.M. "An adaptive echo canceller" (1967)
4. Benesty, J. et al. "Advances in Network and Acoustic Echo Cancellation"
5. Valin, J.M. "On Adjusting the Learning Rate in Frequency Domain Echo Cancellation with Double-Talk", IEEE 2007
6. Ephraim, Y. & Malah, D. "Speech Enhancement Using a MMSE Short-Time Spectral Amplitude Estimator", IEEE 1984
7. ITU-T G.168: Digital Network Echo Cancellers
8. WebRTC AEC3: `webrtc/modules/audio_processing/aec3/`
9. SpeexDSP: `speexdsp/libspeex/mdf.c`
