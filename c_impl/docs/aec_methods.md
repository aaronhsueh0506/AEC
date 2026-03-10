# AEC 方法介紹文檔

## 目錄

1. [自適應濾波器演算法總覽](#1-自適應濾波器演算法總覽)
2. [LMS - 最小均方演算法](#2-lms---最小均方演算法)
3. [NLMS - 歸一化最小均方演算法](#3-nlms---歸一化最小均方演算法)
4. [頻域 NLMS](#4-頻域-nlms)
5. [PBFDAF - 分區塊頻域自適應濾波器](#5-pbfdaf---分區塊頻域自適應濾波器)
6. [DTD - 雙講偵測](#6-dtd---雙講偵測)
7. [RES Post-Filter - 殘餘回聲抑制後濾波器](#7-res-post-filter---殘餘回聲抑制後濾波器)
8. [Subband 分頻方式說明](#8-subband-分頻方式說明)
9. [NR Subband 適用性分析](#9-nr-subband-適用性分析)

---

## 1. 自適應濾波器演算法總覽

| 方法 | 模式 | 更新公式 | 複雜度 | 典型 μ | 適用場景 |
|------|------|---------|--------|--------|---------|
| LMS | `AEC_MODE_LMS` | `w += μ·e·x` | O(N) | 0.001~0.05 | 穩態環境、極低資源 |
| NLMS | `AEC_MODE_TIME` | `w = leak·w + μ/(‖x‖²+δ)·e·x` | O(N) | 0.1~0.8 | 一般用途、低延遲 |
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
                   │    [DTD Detection]
                   │         │
                   │    dtd_active? ──→ No  ──→ [Weight Update]
                   │         │                        │
                   │        Yes ──→ Skip Update        │
                   │                                   ▼
                   │                          w[n+1] = f(w[n], e, x)
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
    mu_eff = filter->mu / (filter->power_sum + filter->delta);  // NLMS
} else {
    mu_eff = filter->mu;  // LMS (固定步長)
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
- 延遲較 TIME 模式大（hop_size = 256 = 16ms）
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

## 6. DTD - 雙講偵測

### 目的

偵測近端和遠端同時說話（double-talk）的狀況。雙講時必須停止權重更新，
否則自適應濾波器會把近端語音當作回聲來消除，導致濾波器發散。

### 方法 1: Geigel DTD

**原理：** 比較近端信號和遠端信號的振幅比。

```
判斷條件: |d[n]| > θ · max(|x[n-k]|)   for k ∈ [0, window_length)
```

| 參數 | 典型值 | 說明 |
|------|--------|------|
| θ (threshold) | 0.6 | 門限，越低越敏感 |
| window_length | 4000 | 最大值追蹤窗口（= filter_length） |

**運作方式：**
- 維護一個環形緩衝區追蹤遠端信號的近期最大值
- 當近端超過遠端最大值的 θ 倍時，判定為雙講
- 快速更新：若新值 >= 當前最大值直接替換
- 定期重算：每圈環形緩衝區重新計算一次最大值

### 方法 2: 能量比 DTD

**原理：** 比較誤差信號與回聲估計的能量比。

```
near_energy  = α · near_energy  + (1-α) · d²[n]
error_energy = α · error_energy + (1-α) · e²[n]
echo_energy  = α · echo_energy  + (1-α) · ŷ²[n]

ratio = error_energy / (echo_energy + ε)

判斷條件: ratio > energy_threshold
```

| 參數 | 典型值 | 說明 |
|------|--------|------|
| α_energy | 0.95 | 能量平滑因子 |
| energy_threshold | 0.4 | 能量比門限 |

**解釋：** 當誤差能量遠大於回聲估計時，代表有額外的近端信號（雙講）。

### 組合判決

```
dtd_detected = geigel_detected OR energy_detected
```

### Hangover 機制

防止 DTD 快速開關造成的不穩定：

```
if (detected):
    hangover_count = hangover_max    (例如 15 幀)
    dtd_active = true
elif (hangover_count > 0):
    hangover_count--
    dtd_active = true                (hangover 期間仍然啟用)
else:
    dtd_active = false
```

| 參數 | 典型值 | 說明 |
|------|--------|------|
| hangover_max | 15 幀 | 持續抑制時間 (~240ms @ 16ms/frame) |

### 信心度追蹤

```
if (detected):
    confidence = min(confidence + 0.1, 1.0)     // 快速上升
else:
    confidence = max(confidence - 0.02, 0.0)    // 緩慢下降
```

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

---

## 8. Subband 分頻方式說明

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

## 9. NR Subband 適用性分析

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
| LMS/NLMS (TIME) | 16 ms (hop=256) |
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
| dtd_threshold | 雙講靈敏度 | 降低→更敏感；提高→更保守 |
| hangover_frames | DTD 持續時間 | 增加→更穩定但收斂更慢 |
| res_over_sub | 殘餘回聲抑制強度 | 提高→更強抑制，風險失真 |
| res_g_min_db | 最大抑制量 | 降低→更深抑制，-20dB 通常夠用 |
