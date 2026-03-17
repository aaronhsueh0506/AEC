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

> **實作狀態：** Python only（C 版本僅實作 PBFDAF）

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

> **實作狀態：** Python only（C 版本僅實作 PBFDAF）

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

> **實作狀態：** Python only（C 版本僅實作 PBFDAF）

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
| block_size | next_pow2(2×FL) | fft_size (固定 512) |
| _internal_hop | block_size/2 (= FL) | 256 (= hop_size) |
| 最大 echo path | ~16ms | 250ms+ |
| 記憶體 | W[n_freqs] | W[P][n_freqs] |
| 計算量 | O(N log N) | O(P·N log N) |
| DTD cadence | 每 FL/2/hop_size frames | 每 frame |
| DT 處理粒度 | 粗（整個 FL 一次更新） | 細（256 samples 一次更新） |

**FREQ 模式的根本限制**：filter_length 與 block_size 耦合。FL 越長 → block_size 越大 →
每次 weight update 涵蓋更多 samples → DT 期間污染更嚴重。SUBBAND 解耦了這兩者，
filter_length 只影響 partition 數，不影響 block_size 和 DTD cadence。

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

## 6. DTD — 自適應更新控制

### 6.1 問題定義

AEC 自適應濾波器在以下情況需要控制更新：
- **Double-Talk (DT)**：近端與遠端同時說話，繼續更新會把近端語音當 echo 學習
- **Echo Path Change**：回聲路徑改變，需要重新收斂
- **Filter Divergence**：output 品質比 input 更差

### 6.1.1 DTD 整體運作機制

本專案使用**三層防護**控制自適應更新（僅 FREQ/SUBBAND 模式，LMS/NLMS 無 DTD — 詳見 [dtd_design.md §3.5](dtd_design.md)）：

```
每個 hop (256 samples, 16ms @ 16kHz):

  ┌─────────────────────────────────────────────────────────┐
  │ 1. PBFDAF 處理                                          │
  │    output = filter.process(near, far, mu_scale)         │
  │    → 產生 output、error_spec、echo_spec、far_spec       │
  └───────────────┬─────────────────────────────────────────┘
                  │
  ┌───────────────▼─────────────────────────────────────────┐
  │ 2. Shadow Filter (可選)                                  │
  │    shadow_mu = 1.0 - conf × (1-0.2) ← 寬鬆 DTD（最低 20%）│
  │    shadow.process(near, far, shadow_mu)  ← 只更新 weights│
  │    50-frame warm-up → 連續 3 frames shadow_err < main_err │
  │    × 0.8 → copy weights（不切換 output）                   │
  └───────────────┬─────────────────────────────────────────┘
                  │
  ┌───────────────▼─────────────────────────────────────────┐
  │ 3. Divergence Detector                                   │
  │    ratio = max(energy_ratio, peak_ratio)                 │
  │    → 更新 div_confidence [0,1]                           │
  │                                                          │
  │    偵測目標：output 比 input 更差 (ratio > 1.2)           │
  │    適用場景：filter 發散、嚴重失真                         │
  └───────────────┬─────────────────────────────────────────┘
                  │
  ┌───────────────▼─────────────────────────────────────────┐
  │ 4. Coherence Detector                                    │
  │    coherence = MSC(error_spec, far_spec)                 │
  │    → 更新 coh_confidence [0,1]                           │
  │                                                          │
  │    偵測目標：error 含近端語音（跟 far-end 不相關）          │
  │    適用場景：收斂後的 double-talk（divergence 偵測不到）    │
  └───────────────┬─────────────────────────────────────────┘
                  │
  ┌───────────────▼─────────────────────────────────────────┐
  │ 5. mu_scale 計算                                        │
  │    Coherence 主導，Divergence 為 Fallback                │
  │    confidence 帶記憶衰減 (prev × 0.9)                     │
  │    mu_scale = 1.0 - confidence × (1.0 - 0.05)           │
  │    EPC 時 mu_scale ≥ 0.5（維持追蹤能力）                   │
  └───────────────┬─────────────────────────────────────────┘
                  │
  ┌───────────────▼─────────────────────────────────────────┐
  │ 6. RES Post-Filter (可選)                                │
  │    OLA + sqrt-Hann 窗，EER-based 增益                    │
  │    DTD-aware: over_sub × (1-dtd_conf)，DT 時自動放鬆     │
  │    cross-freq smoothing + 慢 attack，抑制殘餘回音 2~4 dB  │
  └───────────────┬─────────────────────────────────────────┘
                  │
  ┌───────────────▼─────────────────────────────────────────┐
  │ 7. Output Limiter (安全網)                               │
  │    if |output| > |mic|: output *= |mic| / |output|      │
  └─────────────────────────────────────────────────────────┘
```

**關鍵設計原則**：

1. **不完全凍結**：mu_scale 最低 0.05（而非 0），保留微量追蹤能力，避免 DT 結束後無法恢復
2. **Coherence 主導，Divergence 為 Fallback**：Coherence 偵測「快要壞」（事前），Divergence 偵測「已經壞」（事後）。Coherence 啟動時忽略 Divergence，避免 echo path change 時誤壓 mu
3. **Confidence 記憶衰減**：conf = max(raw, prev × 0.9)，避免偵測器交替時保護空窗
4. **Warmup 保護**：前 50 幀（~0.8s）不啟用任何偵測，讓濾波器和 PSD 估計穩定
5. **Shadow filter 寬鬆 DTD**：DT 時 mu 最低保留 20%（vs main 的 5%），平衡保護和追蹤能力
6. **Echo Path Change (EPC)**：shadow-based ΔE 偵測，維持 mu≥0.5 讓 filter 追蹤新路徑

**各偵測器的覆蓋範圍**：

| 場景 | Divergence | Coherence | Shadow |
|------|-----------|-----------|--------|
| Filter 發散 (output > input) | ✅ fallback | — | ✅ 自動修正（shadow→main）|
| 收斂後 double-talk | ❌ ratio < 1 | ✅ 主要偵測 | ✅ 寬鬆 DTD 追蹤 |
| Echo path change | ❌ 被 Coherence 抑制 | EPC 壓低 conf | Main 學更快，#6 同步 shadow |
| 未收斂 single-talk | ✅ 若 ratio > 1.2 | ❌ coherence 高 | ✅ 保守追蹤 |
| 靜音 | — release | — release | — 持續更新 |

**Confidence 時間行為**：

| 偵測器 | Attack | Release | Hangover | DT→ST 轉換 |
|--------|--------|---------|----------|------------|
| Divergence | 0.3/block | 0.05/block | 15 blocks (240ms) | ~560ms |
| Coherence | 0.3/block | 0.1/block | 3 blocks (48ms) | ~208ms |

Coherence DTD 使用較短 hangover 和較快 release，因為 PSD EMA smoothing (α=0.85)
本身已提供 ~6 block 的平滑效果（相當於內建 hangover），不需要額外長時間保持。

#### DTD vs 發散偵測

| | 傳統 DTD | 本專案做法 |
|---|---|---|
| 目標 | 偵測「近端有人在講話」 | Divergence：偵測發散；Coherence：偵測 DT |
| 時機 | 在更新前判斷 | 在更新後檢查 |
| 反應 | 凍結更新 | 連續 mu scaling |
| 代表 | Geigel, NCC | Output-vs-Input + MSC Coherence |

### 6.2 經典 DTD 方法（不推薦）

#### Geigel DTD

```
if |d[n]| > threshold × max(|x[n-k]|, k=0..L-1):
    → Double-talk detected
```

- 計算量極小，不依賴濾波器狀態
- **❌ 不適用於 AEC**：設計用於線路回聲消除 (LEC)，ERL ≈ -6dB。在 AEC 中 echo gain ≈ 1.0，`|mic|` 幾乎永遠 > `threshold × |ref|`，導致 100% 假觸發
- C 版本曾有 Geigel 實作（`dtd.c`），已移除

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

### 6.3 Output-vs-Input 發散偵測 ✅ 本專案採用

**核心思想**：不做明確 DTD，改為偵測 output 是否比 input 更差

> **注意**：此方法並非 WebRTC AEC3 原生做法。WebRTC AEC3 使用 dual filter（main + shadow）隱式處理發散，沒有顯式 DTD。我們的方法是獨立設計的 output-vs-input 比較機制。

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
elif ratio > 1.2:
    confidence += attack × (ratio - 1.2) // 輕微發散（比例反應，閾值 1.2）
else:
    // Proportional release: faster when ratio well below 1.0
    release_scale = max(1.0 - ratio, 0.2)  // 0.2x ~ 1.0x
    confidence -= release × (1.0 + 4.0 × release_scale)  // 正常
```

**為什麼 mild threshold = 1.2（而非 1.0）**：
- ratio ≈ 1.0 只代表濾波器尚未收斂，不代表發散
- 若閾值為 1.0，warmup 結束後 DTD 會在收斂期間誤觸發，降低 mu 形成死循環
- 1.0-1.2 範圍的 output 僅比 input 大 <1 dB，有 Output Limiter 兜底，不會產生可聽的失真

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
| **全模式** | 50 幀 | ~0.8 秒 |

### 6.4 Dual Filter (Shadow Filter) — 本專案可選

**核心思想**：使用兩個自適應濾波器，shadow filter 永遠以保守步長更新，main filter 用於輸出。
當 shadow 表現更好時，將 shadow 權重複製到 main，天然解決發散問題。

這是 WebRTC AEC3 和 SpeexDSP 的核心機制。

```
Main filter:   mu = config.mu × mu_scale (DTD 控制，最低 5%)
Shadow filter: mu = config.mu × 0.5 × shadow_mu_scale (寬鬆 DTD，最低 20%)

每個 block:
  1. Main: output = process(near, far, mu_scale)
  2. Shadow: shadow.process(near, far, shadow_mu_scale)  // 只更新 weights
  3. Smooth error energy:
     main_err_smooth   = α × main_err_smooth   + (1-α) × main_err
     shadow_err_smooth = α × shadow_err_smooth + (1-α) × shadow_err
  4. Warm-up guard: 前 50 frames 不允許 copy（兩個 filter 都需要先收斂）
  5. Copy hysteresis: 連續 3 frames shadow_err < main_err × 0.8 才觸發
     if shadow_err_smooth < main_err_smooth × threshold:
         main.W ← shadow.W        // 只複製權重（不切換 output）
         main_err_smooth = shadow_err_smooth
```

**設計要點**：
- **不切換 output**：copy 只複製 weights，不用 `output = shadow_out`，避免 output 不連續
- **50-frame warm-up**：收斂前兩個 filter 的 error 都不穩定，比較無意義，強行 copy 會退化
- **寬鬆 DTD**：shadow DTD mu 最低 20%（vs main 的 5%），保留更多追蹤能力

**與 DTD 的互補**：
- DTD 偵測發散後降低 mu → 防止 main filter 惡化
- Shadow filter 同時以保守 mu 持續追蹤 → 當 main 發散時自動修正
- 兩者可同時啟用，互不衝突

**參數**：

| 參數 | 預設值 | 說明 |
|------|--------|------|
| enable_shadow | false | 啟用 shadow filter |
| shadow_mu_ratio | 0.5 | shadow mu = main mu × ratio |
| shadow_copy_threshold | 0.8 | shadow_err < main_err × threshold 時複製 |
| shadow_err_alpha | 0.95 | error energy EMA 平滑係數 |

**記憶體影響**：額外 ~35KB（SubbandNlms ~34KB + output buffer 1KB），可接受。

**與 SpeexDSP 比較**：

| | SpeexDSP | 本專案 |
|---|---|---|
| 命名 | Background / Foreground | Shadow / Main |
| 保守濾波器 | BG (永遠更新) | Shadow (永遠更新) |
| 輸出濾波器 | FG (條件更新) | Main (DTD mu_scale) |
| 複製方向 | BG → FG | Shadow → Main |
| 額外偵測 | 無 (純雙濾波器) | 可選 DTD |

### 6.5 Coherence-Based Double-Talk Detection ✅ 本專案採用（FREQ/SUBBAND）

> 完整 DTD 設計文檔見 [dtd_design.md](dtd_design.md)。

Divergence detection 的限制：收斂後的 double-talk 中，output ≈ near_speech < mic，ratio < 1.0，
偵測不到。Error 包含近端語音（跟 far-end 不相關），filter 持續更新導致權重漂移。

**適用範圍**：FREQ/SUBBAND 模式。LMS/NLMS 因收斂速度慢，coherence DTD 會在 double-talk
場景下過度降低 mu 導致惡性循環（詳見 dtd_design.md §3.5），預設僅使用 divergence DTD。

**解法**：用 error-reference Magnitude Squared Coherence (MSC) 偵測 DT。

```
Smoothed PSDs (EMA smoothing on cross-spectrum):
  S_ex[k] = α × S_ex[k] + (1-α) × error_spec[k] × conj(far_spec[k])
  S_ee[k] = α × S_ee[k] + (1-α) × |error_spec[k]|²
  S_xx[k] = α × S_xx[k] + (1-α) × |far_spec[k]|²

Broadband coherence (voice-band weighted, ratio-of-sums):
  C_avg = Σ(w[k]×|S_ex[k]|²) / (Σ(w[k]×S_ee[k]×S_xx[k]) + ε)
  voice_weight: 300-4kHz → 3.0, <100Hz/>6kHz → 0.3, 其餘 → 1.0

Spectra source:
  SUBBAND: 直接用 FDAF 的 error_spec/far_spec（每 frame）
  FREQ: 獨立 FL-point FFT sliding buffer（每 FL/2 samples）
        → 解耦 DTD 與 FDAF 的大 block_size

Decision (hysteresis):
  if C_avg > coh_high(0.6):     no DT → release confidence
  elif C_avg < coh_low(0.3) AND error_energy > 0.1 × far_energy:
                                 DT detected → attack confidence
  else:                          ambiguous → slow release
```

**三種情境的行為**：

| 情境 | Coherence | NER (error/far) | DT 判定 |
|------|-----------|-----------------|---------|
| 未收斂 (single-talk) | 中~高 (residual echo correlated) | 低 (<0.01) | ❌ 不觸發 |
| 收斂後 (single-talk) | 低 (error ≈ noise) | 極低 (~0) | ❌ 不觸發 |
| 收斂後 (double-talk) | 低 (near-end uncorrelated) | 高 (>0.1) | ✅ 觸發 |

**跟 divergence detection 的關係**：互補，取 `max(div_conf, coh_conf)` 作為最終 confidence。
Divergence 偵測 filter 發散，coherence 偵測近端語音汙染。

**Shadow filter 在 DT 期間**：寬鬆 DTD 保護（mu 最低 20%，vs main 的 5%）。
Shadow mu 本身已很小 (main_mu × 0.5)，DT 期間漂移速度有限，DT 結束後會重新收斂。

> **注意**：Shadow filter 僅在 SUBBAND 模式下有效。FREQ 模式因 buffering 機制（block_size > hop_size），
> shadow filter 和 RES post-filter 會自動跳過。

### 6.6 DTD 方法比較

| 方法 | 計算量 | 適用 AEC | 準確度 | 穩健性 | 本專案 |
|------|--------|----------|--------|--------|--------|
| Geigel | 極低 | ❌ | 低 | 低 | — |
| NCC | 中 | △ | 中 | 中 | — |
| Error/Echo Ratio | 低 | ❌ | 低 | 低 | v1.2（已移除） |
| **Output-vs-Input 發散偵測** | **低** | **✅** | **高** | **高** | **✅ 預設啟用** |
| **Coherence-based DT** | **中** | **✅** | **高** | **高** | **✅ 預設啟用** |
| **Dual Filter (Shadow)** | **高 (2×)** | **✅** | **最高** | **最高** | **✅ 可選** |

### 6.7 參數

**Divergence detection**：

| 參數 | 典型值 | 說明 |
|------|--------|------|
| divergence_factor | 1.5 | output > 1.5× input → 嚴重發散 |
| mild_threshold | 1.2 | ratio > 1.2 才觸發輕微發散偵測 |
| confidence_attack | 0.3 | confidence 上升速率 |
| confidence_release | 0.05 | confidence 下降速率（慢放） |
| mu_min_ratio | 0.05 | 最低 mu 比例（不完全凍結） |
| warmup_frames | 50 | 初始收斂期不偵測 |

**Coherence-based DT detection**：

| 參數 | 典型值 | 說明 |
|------|--------|------|
| dtd_coh_alpha | 0.85 | PSD smoothing (~6 block 時間常數) |
| dtd_coh_high | 0.6 | coherence > 此值 → 非 DT |
| dtd_coh_low | 0.3 | coherence < 此值 → DT |
| dtd_coh_energy_floor | 0.1 | error_energy > floor × far_energy 才觸發 |
| dtd_coh_hangover | 3 | DT 偵測後的持續 block 數（參考 WebRTC ≈3 blocks） |
| dtd_coh_release | 0.1 | confidence 下降速率（比 divergence 的 0.05 更快） |

**Coherence DTD hangover 設計**：PSD EMA smoothing (α=0.85) 本身已提供 ~6 block 的平滑效果，
因此 coherence DTD 不需要像 Geigel DTD 那樣長的 hangover（15 blocks）。
WebRTC AEC3 的 stationarity estimator 也只用 ~3 blocks hangover。
較短的 hangover + 較快的 release 讓 DT→ST 轉換約 ~208ms（vs 舊的 ~560ms），
改善了短 DT 間隔的時間解析度。

### 6.8 NLMS 額外保護：權重 Norm 約束

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
RES post-filter 對殘餘回聲進行額外 2~4 dB 的抑制。

### 架構：OLA + sqrt-Hann 窗

RES 使用獨立的 Overlap-Add 框架（與 PBFDAF 的 Overlap-Save 分離），
避免 frame 邊界不連續和棋盤頻譜（musical noise）：

```
1. 接收 hop-size (256) 的時域 error 信號
2. Sliding buffer：拼接前一 hop 和當前 hop → block_size (512) 的 frame
3. 分析窗：sqrt-Hann × frame → FFT
4. 頻域增益：EER-based gain → cross-freq smoothing → temporal smoothing
5. 合成窗：IFFT → sqrt-Hann × frame
6. Overlap-Add：累加到 OLA buffer，輸出前 hop
```

**為什麼用 sqrt-Hann**：分析窗 × 合成窗 = Hann，50% overlap-add 完美重建
（`Σ hann[n + k·hop] = 1`），不會引入振幅調制。

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

**DTD-aware 增益計算：**
```
over_sub_eff = α_os × (1.0 - dtd_conf)    // DT 時降低壓制強度
G[k] = 1 / (1 + over_sub_eff · EER[k])
```

| dtd_conf | over_sub_eff | 行為 |
|----------|-------------|------|
| 0.0 | 1.5 | 正常壓制（far-end single talk） |
| 0.5 | 0.75 | 減半壓制（疑似 DT） |
| 1.0 | 0.0 | 完全 bypass（確定 DT，保護近端語音） |

**為什麼需要 DTD-aware**：DT 時 error = 近端語音 + 殘餘回音。若不降低 over_sub，
RES 會把近端語音也一起壓掉，造成語音失真。

**增益下限（Floor）：**
```
G[k] = max(G[k], g_min)
```
例如 g_min_db = -20 dB → g_min = 0.1，防止過度抑制造成不自然的靜音。

**跨頻率平滑（Cross-frequency smoothing）：**
```
G[k] = moving_average(G, width=3)[k]    // 3-bin moving average
```
消除相鄰 bin 間的孤立 gain 峰谷，減少 musical noise（棋盤頻譜的主因之一）。

**非對稱時間平滑：**
```
if G[k] < gain_smooth[k]:        // Attack（回聲突然出現）
    α_g = 0.6                    // 中速反應（避免 binary-like switching）
else:                             // Release（回聲消失）
    α_g = 0.8                    // 緩慢恢復

gain_smooth[k] = α_g · gain_smooth[k] + (1 - α_g) · G[k]
```

- **中速攻（attack = 0.6）：** time constant ~2.5 frames，避免太快切換造成不連續
- **慢放（release = 0.8）：** 回聲消失後緩慢恢復，避免音樂噪聲

**遠端靜音自動釋放：**
```
if (far_power < 1e-6):
    G[k] = 1.0    // 遠端無信號時不抑制
```

### 參數

| 參數 | 典型值 | 說明 |
|------|--------|------|
| α_os (over_sub) | 1.5 | 過減因子，越大抑制越強（DT 時自動降低） |
| g_min_db | -20 dB | 最小增益限制 |
| α_psd | 0.9 | PSD 平滑因子 |
| α_attack | 0.6 | 攻擊平滑（中速，避免 musical noise） |
| α_release | 0.8 | 釋放平滑（慢） |
| cross_freq_width | 3 bins | 跨頻率平滑寬度 |

### 典型效能

- 語音主導時：G ≈ 1.0（最小影響）
- 回聲突發時：壓低 2~4 dB（ERLE 提升）
- DT 時：自動放鬆壓制，保護近端語音
- 噪聲環境：G = 1.0（不抑制背景噪聲，那是 NR 的工作）

### 本專案狀態

- Python：`ResFilter` class（`aec.py`）已啟用，使用 OLA + sqrt-Hann + DTD-aware
- C：`res_filter.c` 已實作（尚未同步 OLA 架構）

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

1. 考慮 Wiener + comfort noise（進一步降低 musical noise）
2. Echo-only 段落的 NLP 可進一步提升 ERLE
3. C 版本同步 OLA + sqrt-Hann + DTD-aware 架構

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
