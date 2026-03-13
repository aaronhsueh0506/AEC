# DTD (Double-Talk Detection) 設計文檔

本文件詳細說明 AEC 模組中 DTD 的完整設計，包含演算法、參數、實作細節。

> Python 與 C 實作完全一致，本文以 Python 程式碼為主要參考。

---

## 1. 架構總覽

本專案使用 **雙偵測器 + 可選 Shadow Filter** 三層架構：

```
每個 hop (256 samples, 16ms @ 16kHz):

  ┌─────────────────────────────────────────────────────────┐
  │ 1. 自適應濾波器處理                                       │
  │    output = filter.process(near, far, mu_scale)         │
  │    output = near_end - echo_estimate  （即 error signal）│
  └───────────────┬─────────────────────────────────────────┘
                  │
  ┌───────────────▼─────────────────────────────────────────┐
  │ 2. Shadow Filter（可選，見第 5 節）                        │
  │    shadow_out = shadow.process(near, far, 1.0)          │
  │    若 shadow 更好 → 複製權重到 main                       │
  └───────────────┬─────────────────────────────────────────┘
                  │
  ┌───────────────▼─────────────────────────────────────────┐
  │ 3. Divergence Detector（所有模式）                        │
  │    ratio = max(energy_ratio, peak_ratio)                │
  │    → 更新 div_confidence [0,1]                           │
  └───────────────┬─────────────────────────────────────────┘
                  │
  ┌───────────────▼─────────────────────────────────────────┐
  │ 4. Coherence Detector（FREQ/SUBBAND 預設啟用）              │
  │    coherence = MSC(error_spec, far_spec)                │
  │    → 更新 coh_confidence [0,1]                           │
  └───────────────┬─────────────────────────────────────────┘
                  │
  ┌───────────────▼─────────────────────────────────────────┐
  │ 5. mu_scale 計算                                        │
  │    confidence = max(div_confidence, coh_confidence)      │
  │    mu_scale = 1.0 - confidence × (1.0 - 0.05)           │
  │    → 用於下一個 hop 的濾波器更新                           │
  └───────────────┬─────────────────────────────────────────┘
                  │
  ┌───────────────▼─────────────────────────────────────────┐
  │ 6. Output Limiter（安全網）                               │
  │    if |output| > |mic| → 縮放                            │
  └─────────────────────────────────────────────────────────┘
```

**各模式支援情況**：

| 功能 | LMS | NLMS | FREQ | SUBBAND |
|------|-----|------|------|---------|
| Divergence DTD | — | — | ✅ | ✅ |
| Coherence DTD | — | — | ✅ | ✅ |
| Shadow Filter | — | — | ✅ | ✅ |
| RES Post-Filter | — | — | ✅ | ✅ |
| Output Limiter | ✅ | ✅ | ✅ | ✅ |

**為什麼 LMS/NLMS 不啟用任何 DTD**：

經過業界調研，時域慢收斂濾波器沒有有效的 DTD 方案：

| 方法 | 問題 |
|------|------|
| Geigel | AEC echo gain≈1.0 → 100% 假觸發（設計用於 LEC, ERL≈-6dB） |
| NCC | 循環依賴：需要 filter 部分收斂，但 DT 時 filter 被污染 |
| Coherence DTD | 降低 mu → 慢收斂更慢 → 惡性循環（ERLE 3.4→0.7 dB） |
| VSS-NLMS | DT-robust 版本核心也是 cross-correlation，同樣問題 |
| Two-Path/Shadow | Background 也是慢收斂，DT 時一樣被污染無法恢復 |

LMS/NLMS 的 Divergence detector 在正常 DT 場景下也無實質幫助
（output < input → ratio < 1.0 → 不觸發）。Output Limiter 已提供安全網。

**結論**：需要 double-talk robustness → 使用 FREQ/SUBBAND 模式。

---

## 2. Divergence Detector

### 2.1 目標

偵測 **output 比 input 更差** 的情況，即濾波器發散。

這不是傳統的 DTD（偵測近端語音），而是 output-vs-input 品質偵測。
當濾波器權重錯誤時，echo_estimate 反而增加能量，output > input。

### 2.2 演算法

```python
# 計算 energy 和 peak 兩種 ratio
output_energy = mean(output²)
near_energy   = mean(near_end²)
output_peak   = max(|output|)
near_peak     = max(|near_end|)

energy_ratio = output_energy / (near_energy + 1e-10)
peak_ratio   = output_peak / (near_peak + 1e-10)
ratio        = max(energy_ratio, peak_ratio)

# Confidence 更新
if near_energy < 1e-10 and near_peak < 1e-6:
    # 靜音 → 正常 release
    confidence -= release

elif ratio > divergence_factor (1.5):
    # 嚴重發散 → full attack
    confidence += attack

elif ratio > mild_threshold (1.2):
    # 輕微發散 → 比例 attack
    confidence += attack × (ratio - 1.2)

else:
    # 正常 → proportional release (ratio 越低 release 越快)
    release_scale = max(1.0 - ratio, 0.2)
    confidence -= release × (1.0 + 4.0 × release_scale)
```

### 2.3 設計決策

**為什麼 mild_threshold = 1.2（不是 1.0）**：
- ratio ≈ 1.0 代表濾波器尚未收斂，不代表發散
- 若閾值 1.0，warmup 結束後 DTD 會在收斂期間誤觸發，降低 mu 形成死循環
- 1.0-1.2 範圍的 output 僅比 input 大 <1 dB，有 Output Limiter 兜底

**為什麼同時用 energy + peak**：
- Energy-based：捕捉整體發散（sustained divergence）
- Peak-based：捕捉瞬態尖峰（transient spikes that mean misses）

**為什麼 proportional release**：
- ratio 遠低於 1.0 時（filter 收斂良好），快速 release 讓 mu 恢復
- ratio 接近 1.0 時（filter 尚未完全收斂），慢 release 保持警戒

### 2.4 參數

| 參數 | 預設值 | 說明 |
|------|--------|------|
| `divergence_factor` | 1.5 | ratio > 此值 = 嚴重發散 |
| mild_threshold | 1.2 | ratio > 此值 = 輕微發散（硬編碼） |
| `confidence_attack` | 0.3 | confidence 上升速率 |
| `confidence_release` | 0.05 | confidence 下降速率 |
| `warmup_frames` | 50 | 初始不偵測期（~0.8s） |

---

## 3. Coherence Detector

### 3.1 目標

偵測 **error 中包含近端語音**（double-talk），這是 divergence detector 偵測不到的場景。

收斂後的 double-talk：output ≈ near_speech < mic，ratio < 1.0，divergence 不觸發。
但 error 中包含近端語音（與 far-end 不相關），若繼續更新，權重會漂移。

### 3.2 演算法

使用 error-reference Magnitude Squared Coherence (MSC)：

```python
# 1. 取得頻譜（FREQ/SUBBAND 模式）
#    直接使用 filter 內部的 error_spec, far_spec

# 2. EMA 平滑 PSD 估計
S_ex[k] = α × S_ex[k] + (1-α) × error_spec[k] × conj(far_spec[k])
S_ee[k] = α × S_ee[k] + (1-α) × |error_spec[k]|²
S_xx[k] = α × S_xx[k] + (1-α) × |far_spec[k]|²

# 3. 寬頻 coherence（ratio-of-sums，非 per-bin average）
coherence = Σ|S_ex[k]|² / (Σ(S_ee[k] × S_xx[k]) + 1e-10)

# 4. 能量檢查：避免低能量時假觸發
has_energy = (Σ S_ee > energy_floor × Σ S_xx) AND (Σ S_xx > 1e-10)

# 5. Hysteresis 判定
if coherence > coh_high (0.6):
    # 高相關 → 殘餘回音 → 非 DT → release
    _update_confidence(detected=False)
elif coherence < coh_low (0.3) AND has_energy:
    # 低相關 + 有能量 → 近端語音 → DT
    _update_confidence(detected=True)
else:
    # 模糊區 → 緩慢 release
    confidence -= release × 0.5
```

### 3.3 三種情境分析

| 情境 | Error 內容 | Coherence | NER | DT 判定 |
|------|-----------|-----------|-----|---------|
| 未收斂 (single-talk) | 殘餘回音（與 far-end 相關） | 中~高 | 低 | ❌ 不觸發 |
| 收斂後 (single-talk) | ≈ 噪音（能量極低） | 低 | 極低 | ❌ 不觸發（energy check） |
| 收斂後 (double-talk) | 近端語音（與 far-end 無關） | 低 | 高 | ✅ 觸發 |

**關鍵**：energy check 防止「收斂後 single-talk」的假觸發。error 能量極低時，coherence 統計不穩定，
energy_floor 確保只在 error 有實質能量時才判定 DT。

### 3.4 Hangover 機制

```python
def _update_confidence(detected):
    if detected:
        hangover_count = hangover_max  # 重置 hangover
        confidence += attack
    elif hangover_count > 0:
        hangover_count -= 1
        confidence -= release × 0.5   # hangover 期間慢 release
    else:
        confidence -= release          # hangover 結束後正常 release
```

Coherence DTD 使用較短的 hangover (3 blocks ≈ 48ms)，因為 PSD EMA smoothing (α=0.85)
本身已提供 ~6 block 的平滑效果。相比之下，時域 DTD（如 Geigel）需要較長的 hangover
因為其偵測是 sample-level 的，波動較大。

**DT→ST 轉換時序**：

```
DT 結束
  → Coherence 開始上升（PSD smoothing ~6 blocks）
  → 觸發 release
  → Hangover 3 blocks（慢 release）
  → 正常 release 10 blocks
  → Total ≈ 208ms
```

### 3.5 為什麼 LMS/NLMS 完全不用 DTD

LMS/NLMS 不啟用任何 DTD（包括 divergence 和 coherence），原因是慢收斂的根本限制。

**所有 DTD 方案都不適用**：

| 方法 | 為什麼不行 |
|------|-----------|
| Coherence DTD | 降 mu → 慢收斂更慢 → 惡性循環（ERLE 3.4→0.7 dB） |
| Divergence DTD | 正常 DT 時 output < input → ratio < 1.0 → 不觸發，無實質幫助 |
| Geigel | AEC echo gain≈1.0 → 100% 假觸發 |
| NCC | 循環依賴濾波器收斂 |
| VSS-NLMS | DT-robust 版本核心也是 cross-correlation |
| Two-Path/Shadow | Background 也慢收斂，DT 時被污染無法恢復 |

**LMS/NLMS 的安全機制**：
- **Output Limiter**：output 永遠不超過 mic amplitude（硬限制）
- **Weight norm constraint**（NLMS）：防止權重爆炸
- 這兩層已足以防止可聽的失真

**需要 DT robustness → 使用 FREQ/SUBBAND 模式。**

### 3.6 參數

| 參數 | 預設值 | 說明 |
|------|--------|------|
| `dtd_coh_alpha` | 0.85 | PSD EMA smoothing 係數（~6 block 時間常數） |
| `dtd_coh_high` | 0.6 | coherence > 此值 → 非 DT（release） |
| `dtd_coh_low` | 0.3 | coherence < 此值 → DT（attack） |
| `dtd_coh_energy_floor` | 0.1 | error_energy > floor × far_energy 才觸發 |
| `dtd_coh_hangover` | 3 | DT 偵測後的持續 block 數 |
| `dtd_coh_release` | 0.1 | confidence 下降速率 |

**Hysteresis (0.3/0.6) 的由來**：業界常見設定。0.3 和 0.6 之間的 gap 防止 coherence 在邊界附近
時 confidence 頻繁切換（flicker）。

---

## 4. Confidence 合併與 mu_scale

兩個偵測器各自維護 confidence ∈ [0, 1]，最終取 max：

```python
confidence = max(div_confidence, coh_confidence)
mu_scale   = 1.0 - confidence × (1.0 - mu_min_ratio)

# confidence=0.0 → mu_scale=1.00  正常更新
# confidence=0.5 → mu_scale=0.525 半速更新
# confidence=1.0 → mu_scale=0.05  幾乎凍結（保留 5% 追蹤）
```

**為什麼取 max 而非平均**：任一偵測器發現問題就應降低 mu。
Divergence 和 coherence 偵測的是不同異常，不需要兩者同時觸發。

**為什麼不完全凍結（5% 最低）**：
- DT 結束後濾波器需要恢復能力
- 完全凍結 → 濾波器停滯 → 恢復更慢
- 5% 微量更新提供 gradual recovery

**mu_scale 的應用方式**：
- LMS：`mu_eff = mu × mu_scale`
- NLMS：`mu_eff = (mu × mu_scale) / (power_sum + delta)`
- FREQ/SUBBAND：`mu_eff = mu × mu_scale`（per-block）

---

## 5. Shadow Filter（可選）

### 5.1 原理

使用保守步長的影子濾波器持續追蹤回音路徑。當主濾波器發散時，
shadow 因步長小而漂移有限，可自動修正主濾波器。

這是 WebRTC AEC3 和 SpeexDSP 的核心機制。

```
Main filter:   mu = config.mu × mu_scale    ← DTD 控制
Shadow filter: mu = config.mu × 0.5 × 1.0   ← 永遠全速（不受 DTD 控制）

每個 block:
  1. main_out   = main.process(near, far, mu_scale)
  2. shadow_out = shadow.process(near, far, 1.0)
  3. 平滑 error energy (EMA α=0.95):
     main_err   = α × main_err   + (1-α) × mean(main_out²)
     shadow_err = α × shadow_err + (1-α) × mean(shadow_out²)
  4. if shadow_err < main_err × 0.8:
       main.weights ← shadow.weights
       main.echo_spec ← shadow.echo_spec  (RES 一致性)
       output = shadow_out
       main_err = shadow_err
```

### 5.2 與 DTD 的互補

| 場景 | DTD 做什麼 | Shadow 做什麼 |
|------|-----------|--------------|
| Filter 發散 | 降低 mu 防止惡化 | 自動複製修正 |
| Double-talk | 降低 mu 保護權重 | 背景持續追蹤 |
| Echo path change | — | 保守追蹤新路徑 |

Shadow filter 僅適用於 FREQ/SUBBAND 模式（需要頻域權重結構）。

### 5.3 參數

| 參數 | 預設值 | 說明 |
|------|--------|------|
| `enable_shadow` | false | 啟用 shadow filter |
| `shadow_mu_ratio` | 0.5 | shadow mu = main mu × ratio |
| `shadow_copy_threshold` | 0.8 | shadow_err < main_err × threshold 時複製 |
| `shadow_err_alpha` | 0.95 | error energy EMA 平滑 |

---

## 6. Output Limiter（安全網）

最後一道防線，保證 output 永遠不超過 mic amplitude：

```python
near_peak = max(|near_end|)
out_peak  = max(|output|)
if out_peak > near_peak > 1e-6:
    output *= near_peak / out_peak
```

**原理**：正確的 echo cancellation 中，output = mic - echo ≤ mic。
如果 output > mic，代表 echo_estimate 錯誤（可能反向加了能量），直接縮放。

---

## 7. Warmup 機制

濾波器剛啟動時權重為零，output ≈ input，ratio ≈ 1.0。
如果此時啟用偵測，ratio > 1.0 可能觸發 divergence detection，降低 mu，
形成「偵測到發散 → 降低 mu → 更慢收斂 → 更久看起來像發散」的死循環。

所以前 50 幀（~0.8 秒）不啟用任何偵測，讓濾波器自由收斂：

```python
if frame_count < warmup_frames:
    return 0.0  # confidence 固定為 0
```

---

## 8. 經典 DTD 方法評估（為什麼不用）

### 8.1 Geigel DTD

```
if |d[n]| > threshold × max(|x[n-k]|):
    → Double-talk detected
```

- 設計用於 Line Echo Cancellation (LEC)，ERL ≈ -6dB
- 在 AEC 中 echo gain ≈ 1.0，`|mic|` 幾乎永遠 > `threshold × |ref|`
- **結果：100% 假觸發**，濾波器永遠無法更新
- 本專案 C 版本曾有實作（`dtd.c`），已移除

### 8.2 Normalized Cross-Correlation (NCC)

```
ξ = |corr(e, x)| / (||e|| × ||x||)
ξ < threshold → Double-talk
```

- 比 Geigel 適合 AEC，但：
  - 需要長窗口才能穩定估計 correlation
  - 依賴濾波器已部分收斂（循環依賴）
  - 未收斂時 error ≈ echo → ξ 高 → 看起來像 single-talk → 不降 mu → 可能 OK
  - 但 DT 偵測延遲較大

### 8.3 Error-to-Echo Ratio

```
ratio = E[e²] / E[ŷ²] > threshold → Divergence
```

- 嚴重循環依賴：ratio 依賴濾波器品質，濾波器品質依賴 DTD
- 靜音→語音轉場時 ratio 瞬間飆升（假觸發）
- 本專案 v1.2.0 曾使用，v1.3.0 因不穩定移除

---

## 9. 各偵測器覆蓋範圍

| 場景 | Divergence | Coherence | Shadow |
|------|-----------|-----------|--------|
| Filter 發散 (output > input) | ✅ 主要偵測 | — | ✅ 自動修正 |
| 收斂後 double-talk | ❌ ratio < 1 | ✅ 主要偵測 | ✅ 背景追蹤 |
| 未收斂 single-talk | ✅ 若 ratio > 1.2 | ❌ coherence 高 | ✅ 保守追蹤 |
| Echo path change | ✅ 若 output 變差 | — | ✅ 追蹤新路徑 |
| 靜音 | — release | — release | — 持續更新 |

---

## 10. 業界參考

### WebRTC AEC3
- 不用顯式 coherence DTD；靠 dual filter (main + shadow) 隱式處理
- Stationarity estimator: `kHangoverBlocks ≈ 3`, `kAlpha = 0.004`
- Misadjustment overhang = 4 blocks

### SpeexDSP (mdf.c)
- 不用 coherence DTD；用 foreground/background filter variance 比較
- `VAR1_SMOOTH = 0.36`, `VAR2_SMOOTH = 0.7225`
- Saturation hangover = M+1 frames

### 本專案定位
- 結合顯式 DTD (divergence + coherence) 和可選 shadow filter
- 比 WebRTC 簡單（不需 HMM、stationarity estimator）
- 比 SpeexDSP 靈活（支援 4 種濾波器模式）

---

## 11. Python / C 實作一致性

Python 和 C 的 DTD 實作**完全等價**：

| 組件 | Python | C |
|------|--------|---|
| Divergence detection | `DtdEstimator._detect_divergence()` | `aec.c` 內聯 |
| Coherence detection | `DtdEstimator._detect_coherence()` | `aec.c` 內聯 |
| mu_scale 公式 | `_compute_mu_scale()` | `aec.c:276` |
| Hangover | `_update_confidence()` | `aec.c:433` |
| Config 參數 | `AecConfig` dataclass | `AecConfig` struct |

所有閾值、smoothing 係數、attack/release 速率完全相同。
差異僅在實作風格（Python 用 class/method，C 用 inline code）。

> **注意**：C 版本僅支援 PBFDAF (subband) 模式，因此不需要 LMS/NLMS 的 FFT-based coherence DTD。
