# AEC 方法介紹文檔

## 目錄

1. [自適應濾波器演算法總覽](#1-自適應濾波器演算法總覽)
2. [LMS - 最小均方演算法](#2-lms---最小均方演算法)
3. [NLMS - 歸一化最小均方演算法](#3-nlms---歸一化最小均方演算法)
4. [頻域 NLMS](#4-頻域-nlms)
5. [PBFDAF - 分區塊頻域自適應濾波器](#5-pbfdaf---分區塊頻域自適應濾波器)
5.1. [FDKF - 頻域卡爾曼濾波器](#51-fdkf---頻域卡爾曼濾波器)
6. [DTD / 發散偵測](#6-dtd--發散偵測)
7. [RES Post-Filter - 殘餘回聲抑制後濾波器](#7-res-post-filter---殘餘回聲抑制後濾波器)
8. [Post-Filter 方法比較](#8-post-filter-方法比較)
9. [Subband 分頻方式說明](#9-subband-分頻方式說明)
10. [NR Subband 適用性分析](#10-nr-subband-適用性分析)
11. [前處理模組](#11-前處理模組)
12. [收斂控制與 Output Limiter](#12-收斂控制與-output-limiter)

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

## 5.1 FDKF — 頻域卡爾曼濾波器

> **實作狀態：** Python（`PBFDKF._update_weights()`），C 版本尚未同步。
> **啟用方式：** `AecConfig(use_kalman=True)` 或 CLI `--use-kalman`

### 原理

FDKF（Frequency-Domain Kalman Filter）將 PBFDAF 的 NLMS adaptation 替換為 per-bin Kalman filter。
NLMS 對所有頻率 bin 使用同一個步長 μ（或 per-bin μ/P[k]），而 FDKF 為每個 bin 獨立計算最佳 Kalman gain K[k]，根據該 bin 的 signal-to-noise 特性自動調整 adaptation rate。

**與 NLMS 的關鍵差異**：

| | NLMS | FDKF |
|---|---|---|
| 步長控制 | 固定 μ / (P[k] + δ) | Kalman gain K = f(P, X, R) |
| 每 bin 獨立 | 部分（power normalization） | 完全（P, R per-bin） |
| 收斂-穩態 tradeoff | μ 固定 → 無法同時最佳化 | Two-stage Q 自動切換 |
| 彩色信號處理 | 差（power normalization 有限） | 好（per-bin 自動適應） |
| 記憶體額外開銷 | 0 | P[n_part, n_freq] + Q + R + error_psd |

### 數學推導

**狀態空間模型**（每個頻率 bin k 獨立）：

系統模型：
```
W[k, n+1] = W[k, n] + w[k, n]        w ~ N(0, Q)    (process noise)
D[k, n]   = X[k, n] × W[k, n] + v[k, n]  v ~ N(0, R)    (measurement noise)
```

其中：
- `W[k, n]`：第 k 個 bin 的 echo path 權重（待估計狀態）
- `X[k, n]`：far-end reference 頻譜
- `D[k, n]`：near-end microphone 頻譜（= echo + near speech + noise）
- `Q`：process noise covariance — 模型化 echo path 的時變性
- `R`：measurement noise covariance — 近端語音 + 背景噪聲 + 殘餘 echo

**Kalman filter 更新（per-partition per-bin）**：

```
Prediction:
  P_pred[k] = P[k] + Q[k]                          (已隱含在 covariance update)

Kalman gain:
  K[k] = P[k] × conj(X[k]) / (|X[k]|² × P[k] + R[k])

Weight update:
  W[k] += K[k] × E[k]                              (E = error spectrum)

Covariance update:
  P[k] = (1 - K[k] × X[k]) × P[k] + Q[k]
  P[k] = max(P[k], δ)                               (numerical stability)
```

**Measurement noise R 估計**：
```
error_psd[k] = α_R × error_psd[k] + (1 - α_R) × |E[k]|²
R[k] = max(error_psd[k], δ)
```

R 從 error spectrum 的 PSD 估計，代表 noise + residual echo 的功率。
當 echo path 匹配良好時，error 主要是 noise → R 低 → K 適中。
當 echo path 不匹配時，error 包含殘餘 echo → R 高 → K 降低（保守更新）。

### Two-Stage Q：解決 R Self-Reinforcing Deadlock

**問題：固定低 Q 的惡性循環**

```
初始狀態：P 小（低 Q 無法注入足夠 P）
    → K = P×X/(X²P+R) ≈ small/R → K 很小
    → Weight update 很慢
    → Error 仍然大（echo 未被消除）
    → R = smoothed(|E|²) 持續很高
    → K = P/(P×X²+R) ≈ P/R → K 更小
    → 收斂進一步減慢
    ... 惡性循環 ...
```

這是 FDKF 的根本問題：R 從 error PSD 估計，未收斂時 error 很大 → R 很大 → K 被壓低 → 收斂更慢 → R 持續高。

**解法：Two-Stage Q**

```
Phase 1 — 快速收斂（Q = Q_high = 1e-3）：
  高 Q 持續注入 P → P 維持在 0.1~0.5
  → K = P×X²/(X²P+R) 分子夠大
  → 即使 R 高（1e-2），K 仍 ≈ 0.3~0.8
  → 濾波器快速收斂
  → Error 下降 → R 下降 → 正向循環啟動

Phase 2 — 穩態追蹤（Q = Q_low = 1e-5）：
  收斂後 error 小 → R 小（~1e-4）
  低 Q → P 自然衰減至 ~1e-4
  → K ≈ 0.01~0.1
  → 微量追蹤，不干擾已收斂權重

切換條件：
  連續 10 幀 instantaneous ERLE > 6dB → 判定收斂
  → Q 從 Q_high 切換到 Q_low（main + shadow 同時切換）
```

**Kalman gain K 的動態行為**：

```
# 假設某 bin 的典型值

# Phase 1 (未收斂, Q_high=1e-3):
P ≈ 0.3, |X|² ≈ 0.1, R ≈ 1e-2
K = 0.3 × 0.1 / (0.1 × 0.3 + 0.01) = 0.03 / 0.04 = 0.75
→ 接近 NLMS μ=0.75，非常積極

# Phase 2 (已收斂, Q_low=1e-5):
P ≈ 1e-4, |X|² ≈ 0.1, R ≈ 1e-4
K = 1e-4 × 0.1 / (0.1 × 1e-4 + 1e-4) = 1e-5 / 2e-5 = 0.5
→ 但 P 很小所以 K×E 的絕對更新量很小
→ 有效追蹤，不會過度修改已收斂權重
```

### DT 保護

FDKF 的 DT 保護透過 `mu_scale` 施加在 K 上：

```python
K *= mu_scale  # mu_scale ∈ [0.05, 1.0]
```

DT 時 mu_scale → 0.05 → K 被壓到 5% → 幾乎凍結更新。
結合 R 的自然保護（DT 時 error 含近端語音 → R 升高 → K 本身就降低），雙重保護。

### 實作細節

**初始化**（`PBFDKF.__init__()`）：
```python
if self.use_kalman:
    self.P = np.ones((n_partitions, n_freqs)) * 0.5    # per-partition per-bin
    self.Q_high = np.ones(n_freqs) * 1e-3              # fast convergence
    self.Q_low  = np.ones(n_freqs) * 1e-5              # stable tracking
    self.Q = self.Q_high.copy()                         # start fast
    self.R = np.ones(n_freqs) * 1e-2                    # initial measurement noise
    self._error_psd = np.ones(n_freqs) * 1e-2
    self._alpha_r = 0.95                                # R smoothing
```

**Per-partition 更新**（`_update_kalman()`）：
```python
for p in range(n_partitions):
    X = X_buf[p_idx]
    X_power = |X|² + δ

    # Kalman gain
    K = (P[p] × conj(X)) / (X_power × P[p] + R + δ)
    K *= mu_scale  # DT protection

    # Weight update
    W[p] += K × error_spec

    # Covariance update
    KX = real(K × X)
    P[p] = max((1 - KX) × P[p] + Q, δ)

    # Time-domain truncation (prevent circular convolution)
    w_time = IFFT(W[p])
    w_time[hop:] = 0
    W[p] = FFT(w_time)
```

**重置**（`reset()`）：
```python
if self.use_kalman:
    self.P.fill(0.5)        # 與 init 一致
    self.R.fill(1e-2)
    self._error_psd.fill(1e-2)
    self.Q[:] = self.Q_high  # 重置回 Q_high（重新快速收斂）
```

**權重複製**（`copy_weights_from()`）：
```python
def copy_weights_from(self, src):
    self.W[:] = src.W
    if self.use_kalman and hasattr(src, 'P'):
        self.P[:] = src.P    # FDKF 需要 P 和 W 一起複製
```

### 參數

| 參數 | 值 | 說明 |
|------|-----|------|
| P_init | 0.5 | 初始 error covariance，不宜太大（K→1 發散）也不宜太小（收斂慢） |
| Q_high | 1e-3 | 快速收斂 process noise，越大 K 越大收斂越快，但穩態 misadjustment 越大 |
| Q_low | 1e-5 | 穩態 process noise，越小穩態越穩定但追蹤越慢 |
| R_init | 1e-2 | 初始 measurement noise（從 error PSD 動態估計） |
| α_R | 0.95 | R 估計 EMA 係數（越大 R 越平滑、反應越慢） |
| 收斂門檻 | ERLE > 6dB, 10 幀 | Q 切換觸發條件 |

### 與 NLMS 的性能比較

| 指標 | NLMS (PBFDAF) | FDKF (Two-stage Q) |
|------|---------------|---------------------|
| FS echo_mos | 2.76 | **3.20** (+0.44) |
| DT echo_mos | ~2.8 | **3.03** |
| DT deg_mos | ~3.5 | **3.52** |
| 收斂速度 | 中（μ=0.5 固定） | 快（K 自適應，Q_high 驅動） |
| 穩態 ERLE | 中 | 高（Q_low 後 K 自動降低） |
| 額外複雜度 | 0 | +1 array multiply + 1 division per bin |

### 參考文獻

- Enzner, G. & Vary, P. "Frequency-domain adaptive Kalman filter for acoustic echo control in hands-free telephones", Signal Processing, 2006
- Yang, J. et al. "Frequency-domain adaptive Kalman filter with fast recovery of abrupt echo-path changes", IEEE Signal Processing Letters, 2017

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
  │ 2. Shadow Filter (預設開啟)                               │
  │    shadow_mu = 1.0  ← 固定 full mu（AEC3 style）          │
  │    shadow.process(near, far, 1.0)  ← 只更新 weights       │
  │    Copy gate: far_active + not_dt → 連續 3 frames          │
  │    shadow_err < main_err × 0.5 → copy weights              │
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
  │    EER-based Wiener gain，DT 時 EER 自然降低 → 壓制減少  │
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
Main filter:   mu = config.mu × mu_scale
               DTD 模式: mu_scale 由 coherence/divergence 控制（最低 5%）
               Shadow-only: mu_scale 由 simple_mu_ratio 控制（最低 shadow_mu_min=50%）
Shadow filter: mu = config.mu × 0.5 × 1.0  (固定 full mu，AEC3 background filter style)

每個 block:
  1. Main: output = process(near, far, mu_scale)
  2. Shadow: shadow.process(near, far, 1.0)     // full mu, 只更新 weights
  3. Smooth error energy:
     main_err_smooth   = α × main_err_smooth   + (1-α) × main_err
     shadow_err_smooth = α × shadow_err_smooth + (1-α) × shadow_err
  4. Warm-up guard: 前 50 frames 跳過 copy 邏輯
  5. Copy gate: far_active AND not_dt AND NOT epc_active 才允許比較
     not_dt: DTD 模式用 dtd_conf < 0.3, 否則用 far/error ratio > 0.3
  6. Copy hysteresis: 連續 5 frames shadow_err < main_err × 0.7 才觸發
     if shadow_err_smooth < main_err_smooth × 0.7:
         main.W ← shadow.W        // 只複製權重（不切換 output）
         if FDKF: main.P ← shadow.P  // FDKF 需複製 P
         main_err_smooth = shadow_err_smooth
  7. Bidirectional: 若 main_err < shadow_err × 0.7，反向 copy main → shadow
```

**設計要點**：
- **不切換 output**：copy 只複製 weights（+ FDKF P），不用 `output = shadow_out`，避免 output 不連續
- **50-frame warm-up**：收斂前兩個 filter 的 error 都不穩定，比較無意義，強行 copy 會退化
- **Shadow full mu（1.0）**：shadow 不做 DT 保護，讓它自由追蹤（AEC3 style）
- **threshold（0.7）**：需 shadow 比 main 好 30%+ 才 copy，防止 DT 污染權重被 copy
- **Copy gate**：遠端靜音或 DT 或 EPC 期間不允許 copy（far_active + not_dt + not epc_active gate）
- **Bidirectional copy**：main 比 shadow 好時也會反向 copy（main → shadow），保持兩者同步
- **FDKF P 同步**：copy weights 時同時複製 P（error covariance），否則 K 計算不匹配

**與 DTD 的互補**：
- DTD 模式：DTD 預防式降 mu + Shadow 事後修正，雙重保護
- Shadow-only 模式（預設）：main filter 用 simple_mu_ratio 輕量保護，shadow copy 修正
- 兩者可同時啟用（`--enable-dtd`），互不衝突

#### FDKF 模式下的 Shadow Q Ratio

FDKF 模式下，shadow 和 main 都使用 Kalman adaptation。若兩者 Q 相同，收斂速度相同，
`shadow_err ≈ main_err` → copy gate 永不觸發 → shadow 形同虛設。

**解法**：`shadow_q_ratio = 3.0`

```python
# AEC.__init__() — shadow 建立後
if self.config.use_kalman and hasattr(self.shadow_filter, 'Q_high'):
    self.shadow_filter.Q_high = self.filter.Q_high * self.config.shadow_q_ratio  # 3e-3
    self.shadow_filter.Q_low  = self.filter.Q_low  * self.config.shadow_q_ratio  # 3e-5
    self.shadow_filter.Q      = self.shadow_filter.Q_high.copy()
```

Shadow Q 更高 → K 更大 → 收斂更快 → `shadow_err < main_err × 0.7` → copy gate 觸發。

| 階段 | main Q | shadow Q | shadow K | 效果 |
|------|--------|----------|----------|------|
| 收斂前 | 1e-3 | 3e-3 | 比 main 高 ~30% | Shadow 更快收斂，frame 5 起可觸發 copy |
| 收斂後 | 1e-5 | 3e-5 | 比 main 高 ~30% | Shadow 更敏感追蹤 echo path 變化 |
| Echo path 變化 | 慢速恢復 | 快速恢復 | — | Shadow ~3 幀偵測 → copy → main 快速跟上 |

**參數**：

| 參數 | 預設值 | 說明 |
|------|--------|------|
| enable_shadow | **true** | 啟用 shadow filter（預設開啟） |
| shadow_mu_ratio | 1.0 | shadow mu = main mu × ratio |
| shadow_copy_threshold | **0.7** | shadow_err < main_err × threshold 時複製（需 30%+ 優勢） |
| shadow_err_alpha | 0.85 | error energy EMA 平滑係數 |
| shadow_mu_min | **0.5** | Shadow-only 模式 main filter DT mu floor（50%） |
| shadow_copy_hysteresis | 5 | 連續 N frames 才觸發複製 |
| shadow_q_ratio | **3.0** | FDKF 模式：shadow Q = main Q × ratio（越大越積極） |

**記憶體影響**：額外 ~35KB（PBFDKF ~34KB + output buffer 1KB），可接受。

**三方比較（WebRTC AEC3 / SpeexDSP / 本專案）**：

| | WebRTC AEC3 | SpeexDSP | 本專案 |
|---|---|---|---|
| 架構 | PBFDAF + dual filter | MDF + dual filter | PBFDAF + Shadow（預設）+ 可選 DTD |
| 主要濾波器 | Main (refined) | Foreground (FG) | Main (simple mu ratio / DTD mu_scale) |
| 備份濾波器 | Shadow (coarse) | Background (BG) | Shadow (full mu, AEC3 style) |
| DT 處理 | 隱式（dual filter 比較） | 隱式（variable μ, Valin 2007） | Shadow-only: simple mu ratio; DTD: Coherence + Divergence |
| mu 控制 | Adaptive（convergence-based） | Variable（optimal NLMS theory） | Shadow-only: [0.5, 1.0]; DTD: [0.05, 1.0] |
| Copy 觸發 | shadow_err < main_err × threshold | error_FG > error_BG（連續 N frames） | far_active + not_dt + 連續 3 frames + 50-frame warmup |
| Copy 行為 | Shadow → Main + output switch | BG → FG | Shadow → Main（只 copy weights） |
| Post-filter | Wiener-like suppressor + comfort noise | NLP (spectral subtraction) | EER-based RES（spectral subtraction gain） |
| DT 反應速度 | 慢（等 error 翻轉） | 中（variable μ 自動降低） | Shadow-only: 中; DTD: 快（2-3 frames） |
| 計算量 | 永遠 2x filter | 永遠 2x filter | 預設 2x（shadow）; DTD-only: 1x + DTD |
| 參數數量 | 少（copy_threshold, mu） | 少（variance smooth factors） | Shadow-only: 2-3 個; +DTD: 6+ |
| Shadow 是否必要 | ✅ 必須（唯一 DT 機制） | ✅ 必須（唯一 DT 機制） | 預設開啟，可用 DTD-only 替代 |

### 6.4.1 DTD × Shadow 組合行為

本專案的 DTD 和 Shadow 可獨立啟用，提供四種組合：

| | DTD off | DTD on |
|---|---|---|
| **Shadow off** | 無保護（僅 output limiter）。適合無 DT 場景 | **DTD-only**：Coherence 預防式降 mu。`--enable-dtd --no-shadow` |
| **Shadow on** | **Shadow-only（預設）**：≈ WebRTC/SpeexDSP 做法。免調參 | **DTD+Shadow**：`--enable-dtd` |

#### 各組合的 DT 保護行為

| 組合 | DT 反應速度 | 保護機制 | weights 汙染程度 | 適用場景 |
|------|-----------|---------|---------------|---------|
| 無保護 | — | Output limiter 兜底 | 嚴重（全速學習 near-end） | 純 far-end 場景、測試用 |
| DTD-only | 快（2-3 frames） | 降 mu → 減速更新（預防式） | 極少（mu 降到 5%） | 嵌入式、低資源 |
| **Shadow-only（預設）** | 中（simple mu ratio） | Shadow copy 修正 + mu 保護 | 中等（copy gate 保護） | **推薦，免調參** |
| DTD+Shadow | 快 | DTD 預防 + Shadow fallback | 極少 + safety net | 品質優先、EPC 場景 |

#### 算力影響

| 組合 | 額外計算量 | 額外記憶體 | 說明 |
|------|----------|----------|------|
| 無保護 | 0 | 0 | 基礎 PBFDAF |
| DTD-only | +DTD（PSD smoothing + coherence） | +~2KB（PSD buffers） | 極低 overhead |
| Shadow-only | +1x PBFDAF | +~35KB（完整 filter） | 計算量翻倍 |
| DTD+Shadow | +DTD + 1x PBFDAF | +~37KB | 最重 |

#### 實測結果（AEC Challenge, 34 files, FL=1024, subband+RES）

| 模式 | Mean ERLE | NE-Ret Mean | NE-Ret Med | vs AEC3 |
|------|-----------|-------------|------------|---------|
| DTD only | 6.3 dB | -3.0 dB | -1.6 dB | 23W/11L |
| **Shadow only（預設）** | **7.0 dB** | **-3.9 dB** | **-1.6 dB** | **27W/7L** |
| DTD+Shadow | 6.5 dB | -3.8 dB | -1.7 dB | 23W/11L |

Shadow only ERLE 最高（+0.7 dB vs DTD），勝率最佳（27W）。
Median NE-Ret 與 DTD 完全相同（-1.6），Mean 差距來自 fid 1 white noise outlier。

### 6.4.2 三種架構的優劣分析

**本專案（顯式 DTD + 可選 Shadow）的優勢**：

| 優勢 | 說明 |
|------|------|
| 預防式 DT 保護 | Coherence DTD 在 2-3 frames 內偵測 DT → 立即降 mu。Dual-filter-only 是事後修正 |
| 連續 mu scaling | mu_scale ∈ [0.05, 1.0] 精細控制更新速率。Dual filter 只有 binary 決策（copy or not） |
| 可省計算 | DTD-only 不需 2x filter（省 ~35KB memory + 1x PBFDAF） |

**本專案的劣勢**：

| 劣勢 | 說明 |
|------|------|
| 更多參數 | Coherence DTD 有 6+ 參數要調。WebRTC/SpeexDSP 只需 copy_threshold + mu_ratio |
| DTD 過度保護風險 | DTD 太敏感 → mu 長期壓低 → 收斂慢 |
| Warmup 依賴 | 需 50-frame（~0.8s）等 PSD 穩定。Dual-filter 從 frame 0 就保護 |

**WebRTC/SpeexDSP（純 Dual Filter）的優勢**：

| 優勢 | 說明 |
|------|------|
| 概念簡潔 | 一個機制處理所有場景（DT、divergence、EPC） |
| 免調參 | 不需 DTD 閾值 |
| 實戰驗證 | WebRTC 在 Chrome/Android 數十億設備驗證 |

**WebRTC/SpeexDSP 的劣勢**：

| 劣勢 | 說明 |
|------|------|
| 永遠 2x 計算 | Shadow 必須始終運行 |
| 事後修正 | Main diverge → shadow copy。中間有數 frame 汙染 output |
| Post-filter 缺 DT 資訊 | 無法在 DT 時智能減少壓制 |

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

**增益計算（Wiener gain）：**
```
G[k] = 1 / (1 + α_os · EER[k])
```

當 `α_os = 1.0` 時等價於標準 Wiener gain：`G = error_psd / (error_psd + echo_psd)`。

**DT 天生保護**：DT 時 error = 近端語音 + 殘餘回音 → `error_psd` 增大 → `EER` 自然降低 → `G` 接近 1.0 → 壓制自動減少。這與 SpeexDSP 的 Wiener post-filter 機制相同，不需要額外的 DTD 資訊。

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
| α_os (over_sub) | 1.0 | 過減因子（1.0 = 標準 Wiener gain） |
| g_min_db | -20 dB | 最小增益限制 |
| α_psd | 0.9 | PSD 平滑因子 |
| α_attack | 0.6 | 攻擊平滑（中速，避免 musical noise） |
| α_release | 0.8 | 釋放平滑（慢） |
| cross_freq_width | 3 bins | 跨頻率平滑寬度 |

### 典型效能

- 語音主導時：G ≈ 1.0（最小影響）
- 回聲突發時：壓低 2~4 dB（ERLE 提升）
- DT 時：EER 自然降低 → 壓制自動減少（Wiener 天生特性）
- 噪聲環境：G = 1.0（不抑制背景噪聲，那是 NR 的工作）

### 本專案狀態

- Python：`ResFilter` class（`aec.py`）已啟用，使用 OLA + sqrt-Hann + Wiener gain
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
3. C 版本同步 OLA + sqrt-Hann + Wiener gain 架構

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

## 11. 前處理模組

### 11.1 High-Pass Filter（高通濾波器）

> **實作**：`HighPassFilter` class，2 階 Butterworth IIR

**目的**：移除 DC 偏移、50/60Hz 電源線干擾、低頻機械振動。
若不移除 DC，自適應濾波器會在 DC bin 投入大量 capacity 去學習一個慢變的 offset。

**實作方式**：
```
cutoff = 80 Hz
bilinear transform → 2nd-order IIR (a0, a1, a2, b0, b1, b2)
Direct Form II: y[n] = b0*x[n] + b1*x[n-1] + b2*x[n-2] - a1*y[n-1] - a2*y[n-2]
```

分別對 mic 和 ref 各建一個 HPF instance，確保兩路有一致的相位延遲。

### 11.2 Saturation Detector（飽和偵測）

> **實作**：`SaturationDetector` class

**問題**：Speaker 飽和時產生非線性失真（削波、諧波），echo path 不再是線性的，
線性自適應濾波器無法正確建模 → ERLE 下降。

**做法**：
```python
sat_ratio = count(|sample| > 0.95) / len(signal)  # [0, 1]
# sat_ratio > 0.1 → 啟動 soft-clipping:
soft_clipped = tanh(signal × 1.5) / tanh(1.5)
```

Soft-clip reference 讓 filter 看到的 reference 更接近 speaker 實際輸出的失真信號，
改善 echo estimate 的準確度。同時 RES 會加大 `over_sub` 補償非線性殘餘 echo。

### 11.3 Delay Estimation（延遲估計）

> **實作**：`DelayEstimator` class，GCC-PHAT

**問題**：真實系統中 mic 和 ref 之間有 system delay（speaker→mic 傳播 + device buffer），
若不補償，adaptive filter 必須用部分 capacity 建模純粹的延遲，浪費 filter length。

AEC Challenge 資料的典型 delay：23~127ms。

**做法**：
1. **累積 cross-spectrum**：收集 `init_seconds`（0.5s）的 mic/ref 資料
2. **GCC-PHAT 估計**：
   ```python
   R_xy = Σ X_ref^* × X_mic / |X_ref^* × X_mic|   # PHAT 白化
   r_xy = IFFT(R_xy)
   delay = argmax(r_xy[0:max_delay])
   ```
3. **週期性重估**：每 `period_seconds`（2s）重新估計，若差距 > 32 samples 才更新
4. **Ring buffer 延遲補償**：far-end 寫入 ring buffer，讀取時偏移 delay samples

**關鍵發現**：
- Plain cross-correlation（不做 PHAT whitening）在真實資料上更穩健
- GCC-PHAT（full whitening）在 reverberant 環境中 6/10 cases 估計錯誤
- 評測腳本 (`eval_aec_challenge.py`) 使用全信號 offline 估計以獲得最佳精度

### 11.4 Reference Alignment（參考信號對齊）

使用 ring buffer 實現：
```python
# 寫入
_ref_ring[write_pos:write_pos+hop] = far_end
write_pos = (write_pos + hop) % ring_size

# 讀取（延遲 d samples）
read_pos = (write_pos - hop - d) % ring_size
far_aligned = _ref_ring[read_pos:read_pos+hop]
```

Ring buffer 大小 = `max_delay_samples + 4096`，確保有足夠的 look-back 空間。

---

## 12. 收斂控制與 Output Limiter

### 12.1 收斂偵測

**目的**：判斷 adaptive filter 是否已學到有效的 echo path，用於：
1. Two-stage Q 切換（Q_high → Q_low）
2. DTD 啟用門檻（收斂前不啟用 coherence/divergence 偵測）
3. RES dynamic over_sub 計算
4. Per-bin mu_scale（收斂後啟用）

**判定條件**：
```python
inst_erle = 10 × log10(near_power / raw_error_power)
if inst_erle > 6.0:
    conv_counter += 1
else:
    conv_counter = 0

if conv_counter >= 10:  # 連續 10 幀 (~160ms)
    filter_converged = True
    # 切換 Q_high → Q_low（main + shadow）
    for filt in [main_filter, shadow_filter]:
        filt.Q = filt.Q_low.copy()
```

6dB 門檻 + 10 幀連續要求 → 避免偶發的 ERLE spike 誤判。
收斂後不會回退（one-shot），因為正常操作中 ERLE 不應持續低於 6dB。

### 12.2 Warmup 機制

前 50 幀（~0.8s）不啟用以下功能：
- Shadow filter copy gate
- DTD divergence/coherence 偵測

原因：filter 和 PSD 估計尚未穩定，過早啟動這些機制會導致誤判。

Shadow-only 模式下，前 100 幀（~1.6s）使用 warmup mu：
```python
if warmup_frames > 0:
    mu_scale = min(1.0, max(0.7, ratio + 0.3))  # floor at 0.7
```
確保冷啟動階段 mu 不被過度壓低。

### 12.3 Output Limiter

**安全網**：確保 output 永不超過 mic amplitude，即使 DTD 或 adaptation 來不及反應。

```python
near_peak = max(|near_end|)
out_peak = max(|output|)
if out_peak > near_peak > 1e-6:
    target_gain = near_peak / out_peak
else:
    target_gain = 1.0

# Smoothed gain: avoid frame-boundary clicking
if target_gain < limiter_gain:
    alpha = 0.3   # attack: compress quickly
else:
    alpha = 0.8   # release: recover slowly
limiter_gain = alpha × limiter_gain + (1-alpha) × target_gain
output *= limiter_gain
```

**設計要點**：
- 使用 smoothed gain 而非 instantaneous clamp，避免 frame 邊界的 click 聲
- Attack 快（α=0.3）：一旦超過立即壓縮
- Release 慢（α=0.8）：避免 gain pumping（gain 在壓/放之間快速抖動）

### 12.4 Output Noise Gate

遠端有活動但 output 已極低時（< -20dB of mic），進一步 soft fade 降低底噪：

```python
snr = out_power / near_power
if snr < 0.01:  # -20dB
    output *= snr / 0.01  # soft fade
```

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
| res_over_sub | 殘餘回聲抑制強度 | 1.0=Wiener，提高→更強抑制，風險失真 |
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
