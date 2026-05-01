# PBFDKF + Shadow Filter + DTD 演算法介紹

**版本**：v3.8.1（PBFDKF 內核含 v3.7.0 G1 KX-blended P-update；shadow / DTD 結構自 v2.8.1 起穩定）  
**適用場景**：全雙工免持通話（手機、智慧音箱、車用、視訊會議）  
**範圍**：線性 AEC 核心（不含 RES、CNG 後處理）

---

## 一、問題定義

全雙工通話中，喇叭播放遠端語音（far-end），麥克風同時接收到：

```
mic[n] = echo[n] + nearend[n] + noise[n]
```

其中 echo[n] = 喇叭訊號 ref[n] 經過房間脈衝響應後在麥克風收到的回音。

**目標**：估計 echo，從 mic 中扣除，輸出乾淨的近端語音 error[n]。

```
error[n] = mic[n] - echo_estimate[n]
```

---

## 二、PBFDKF — Partitioned Block Frequency Domain Kalman Filter

### 2.1 為什麼需要頻域分區

回音路徑長度（echo tail）通常為 20–100ms。以 16kHz 為例，100ms = 1600 samples。  
時域 NLMS 每幀需更新 1600 個 taps，運算量 O(N)。  

頻域處理（FFT）讓相關運算降至 O(N log N)，且允許 per-bin 獨立控制。  
**Partitioned Block（PB）**：將長 filter 切成多個短 partition，每個 partition 對應一段歷史 ref，支援任意長度而不犧牲效率。

### 2.2 核心公式

每個頻率 bin k，每幀執行以下 Kalman 更新：

```
K[k]  = P[k] × |X[k]|² / (|X[k]|² × P[k] + R[k])    ← Kalman gain
W[k] += K[k] × E[k] × conj(X[k])                       ← weight update
P[k]  = (1 - K[k] × |X[k]|²) × P[k] + Q[k]            ← covariance update
R[k]  = α_R × R[k] + (1-α_R) × |E[k]|²                ← measurement noise (from error)
```

| 變數 | 意義 |
|------|------|
| W[k] | per-bin Kalman filter weight（複數）|
| X[k] | per-bin reference spectrum |
| E[k] | per-bin error spectrum（echo 消除後）|
| P[k] | per-bin error covariance（系統信心）|
| R[k] | per-bin measurement noise（由 error 估計）|
| K[k] | per-bin Kalman gain（自動步長）|
| Q[k] | per-bin process noise（控制追蹤速度）|

### 2.3 Two-Stage Q 控制

**問題**：R 直接來自 error_psd，DT 或噪音時 error ↑ → R ↑ → K ↓ → 更新停滯（R self-reinforcing deadlock）。

**解法**：兩段式 Q 設定：

| 階段 | Q 值 | Kalman gain K | 行為 |
|------|------|---------------|------|
| 未收斂（Q_high）| 1.5×10⁻⁴ | 0.3–0.7（大）| 積極更新，打破 R deadlock |
| 已收斂（Q_low）| 1×10⁻⁷ | 0.01–0.05（小）| 穩態追蹤，不干擾已收斂權重 |

**切換條件**：連續 6 幀 single-talk ERLE > 4 dB → 切換到 Q_low  
**額外保護**：P 加入 clamp（P_MAX = 0.02），EPC 觸發時臨時放寬至 0.1（30 frames）

### 2.4 與 NLMS 的比較

| 指標 | NLMS | PBFDKF |
|------|------|--------|
| 步長控制 | 固定 μ，手動調整 | Kalman gain K，自動適應 |
| 收斂速度 | 中（0.5–2s）| 快（0.1–0.5s）|
| 穩態精度 | 固定雜訊 | 低（K 小） |
| echo path 變化 | 慢恢復 | Q_high 狀態快速重收斂 |
| DT 保護 | 固定 mu_scale | per-bin K 自動降低 |
| 計算量 | O(N) | O(N log N) |

---

## 三、Shadow Filter — 雙濾波器策略

### 3.1 動機

單一 filter 面臨根本矛盾：
- 學習太快 → DT 時學到近端語音，破壞 echo estimate
- 學習太慢 → echo path 改變後（如設備移動）恢復太慢

### 3.2 架構

同時維護兩個 Kalman filter：

```
                    ref, mic
                       │
          ┌────────────┴────────────┐
          │                         │
    ┌─────▼──────┐           ┌──────▼─────┐
    │  Main Filter│           │Shadow Filter│
    │  保守學習   │           │  激進學習  │
    │  輸出給使用者│           │  背景追蹤  │
    └─────┬───────┘           └──────┬─────┘
          │                          │
          │     error 比較           │
          │  shadow_err < main_err × 0.7？
          │         連續 5 frames？   │
          └──────── 是 → copy ◄──────┘
```

**Shadow Q 設定**：shadow Q = main Q × `shadow_q_ratio`（預設 3.5×）

| 階段 | main Q | shadow Q | 效果 |
|------|--------|----------|------|
| 未收斂 | 1.5×10⁻⁴ | 5.25×10⁻⁴ | Shadow 更快收斂 |
| 已收斂 | 1×10⁻⁷ | 3.5×10⁻⁷ | Shadow 仍比 main 積極 |

### 3.3 Copy Gate 條件

shadow 的 weights 複製到 main 需同時滿足：

1. far_active（遠端有訊號，echo path 可觀測）
2. NOT DT（非雙講期，避免 copy 錯誤收斂結果）
3. NOT epc_active（EPC 期間 shadow 也可能不穩定）
4. shadow_err_smooth < main_err_smooth × 0.7（連續 5 frames）

**雙向 copy**：main 比 shadow 更好時也會 copy main → shadow（防止 shadow 偏離）

**50-frame warm-up guard**：startup 期間不允許 copy

### 3.4 效果

- Echo path 變化（設備移動）：shadow 約 100–200ms 先完成重收斂 → copy → main 快速恢復（無此機制需 1–3 秒）
- DT 期間：main 繼續輸出舊有穩定權重，shadow 在背景嘗試追蹤
- 冷啟動：shadow 比 main 更快到達 copy 門檻，加速整體收斂

---

## 四、DTD — Double-Talk Detection

### 4.1 問題

DT（Double-Talk）= 遠端和近端同時說話。  
若 filter 在 DT 期間照常更新，會誤學近端語音為 echo path → 收斂後把語音消掉。

**控制目標**：偵測到 DT → 降低 filter adaptation rate（μ_scale ↓）

### 4.2 三線驅動 effective_dt

```python
effective_dt = max(dt_from_energy, dt_from_shadow, dt_from_coherence)
```

三個獨立 DT 訊號取最大值，互補覆蓋不同場景的 blindspot：

#### 訊號一：Energy DT（dt_from_energy）

```python
max_echo_expected = far_pwr / erl_estimate   # 預期最多能有多少 echo
dt_from_energy = max(0, (mic_pwr - max_echo_expected) / mic_pwr)
```

- 概念：mic 能量超過 ERL 估計的 echo 上限 → 必然有近端語音
- 優點：不依賴 filter 收斂，cold start 也能偵測
- 盲點：高 ERL coupling 場景（喇叭離麥很近）可能誤觸

#### 訊號二：Shadow DTD（dt_from_shadow）

```python
dt_from_shadow = max(0, shadow_err_smooth / main_err_smooth - 1.0) × scale
```

- 概念：shadow（激進）比 main（保守）的 error 大 → 可能有語音讓 shadow 偏離
- 優點：無 ERLE correction 偏差，在高 coupling DT 更準確
- 設計：FDKF 模式下才啟用（shadow filter 存在才有此訊號）

#### 訊號三：Coherence-based DTD（dt_from_coherence）

利用跨頻率 PSD 估計 mic-ref 同調性 C²：

```
C²[k] = |S_fe[k]|² / (S_ff[k] × S_ee[k])
```

| C² 值 | 意義 |
|-------|------|
| 接近 1 | mic 與 ref 高度相關 → 純 echo |
| 接近 0 | 低相關 → DT 或近端語音為主 |

- `S_fe`：mic-ref 跨 PSD（指數平滑，α = 0.65）
- `S_ff`：ref PSD，`S_ee`：error PSD
- Fullband coherence 取 per-bin C² 的均值（語音頻段加權）
- DT 時全頻段 C² 同時下降 → fullband 指標更穩健

### 4.3 有效 DT → Filter 保護

```python
mu_scale = 1.0 - effective_dt × (1.0 - mu_min)
```

- `effective_dt = 1.0`：mu_scale = mu_min（預設 0.6），filter 幾乎停止更新
- `effective_dt = 0.0`：mu_scale = 1.0，正常更新
- 連續插值，無硬切換，避免增益跳變

FDKF 模式下，mu_scale 轉換為 Q scaling：

```python
# DT 時降低 Q，等效於降低步長
Q_effective = Q_high × (1 - effective_dt) + Q_low × effective_dt
```

### 4.4 EPC（Echo Path Change）Detection

```python
# 綜合發散分數
divergence = α × divergence + (1-α) × max(energy_ratio, peak_ratio)
# 同時監測 filter weight 變化速度
epc_active = divergence > threshold AND delta_norm > delta_threshold
```

EPC 觸發後：
- P_MAX 臨時放寬至 0.1（30 frames）→ K 升高 → filter 快速追蹤新 echo path
- `_filter_converged` 重置 → 需重新累積收斂判定

---

## 五、各模組交互關係

```
far-end (ref) ──HPF──→ ring buffer（延遲對齊）
                                │
                                ↓
mic ──────────HPF──────────→ PBFDKF main filter
                            │     ↑ copy (conditional)
                            │   Shadow filter
                            │     ↑ 若 shadow_err < main_err × 0.7 (5 frames)
                            │
effective_dt ←─ max(energy DT, shadow DTD, coherence DT)
                            │
effective_dt → mu_scale → Q_effective (Kalman step size)
effective_dt → DT gate on shadow copy（DT 時禁止 copy）
effective_dt → 後端 RES DT threshold relaxation（此文件不涵蓋）
                            │
EPC detector ────────────→ P_MAX 放寬 + filter reset flag
                            │
                            ↓
                     error signal (linear AEC output)
                            │
                     [Output Limiter]（安全網：output ≤ mic）
                            │
                            ↓
                     clean output（線性 AEC 完成）
```

---

## 六、效能基線（Linear AEC only，無 RES/CNG）

測試集：AEC Challenge Interspeech 2021 Blind Test（800 cases），AECMOS 評測。

| Method | FS echo↑ | DT echo↑ | DT deg↑ | NE deg↑ |
|--------|----------|----------|---------|---------|
| Speex (SpeexDSP 1.2) | 2.808 | 3.368 | 3.225 | 4.128 |
| WebRTC AEC2 | 3.484 | 4.262 | 2.389 | 4.098 |
| **PBFDKF Linear (balanced)** | **2.714** | **3.162** | **3.225** | — |
| WebRTC AEC3 (reference) | 3.875 | 4.538 | 1.850 | 3.454 |

> Linear 指 PBFDKF + Shadow，不含 RES 後處理。FS echo 2.71 是因為線性模型無法完全消除 speaker 飽和造成的非線性 echo，後端 RES 負責處理殘餘。

**加入 RES（Preset balanced）**：FS echo 3.564 / DT echo 4.117 / DT deg 2.410 / NE deg 4.011

---

## 七、主要參數

| 參數 | 預設值 | 說明 |
|------|--------|------|
| `filter_length` | 512 (32ms @16kHz) | 可覆蓋的 echo path 長度 |
| `n_partitions` | 4 | filter_length / hop_size，決定計算量 |
| `kalman_q_high` | 1.5×10⁻⁴ | 收斂前 process noise（所有 preset 統一）|
| `kalman_q_low` | 1×10⁻⁷ | 穩態 process noise |
| `P_init` | 0.01 | per-bin 初始 error covariance |
| `P_MAX` | 0.02 | P clamp 上限（EPC 時放寬至 0.1）|
| `enable_shadow` | True | 雙濾波器（預設開啟）|
| `shadow_q_ratio` | 3.5 | shadow Q = main Q × ratio |
| `shadow_copy_threshold` | 0.7 | shadow_err < main_err × 0.7 觸發 copy |
| `shadow_copy_hysteresis` | 5 | 連續 5 frames 滿足才 copy |
| `mu_min` (shadow_mu_min) | 0.6 | DT 時 mu_scale 下限 |
| `warmup_frames` | 150 | 強制高 mu 的 warmup 幀數 |
| `enable_delay_est` | True | GCC-PHAT 自動延遲估計 |

---

## 八、實作說明

Python 參考實作：`python/aec.py`，class `AecInstance`（[python/aec.py](../python/aec.py)）

```python
from aec import AEC, AecConfig, AecMode

# PBFDKF + Shadow（無 RES）
config = AecConfig(mode=AecMode.PBFDKF, enable_shadow=True, enable_res=False)
aec = AEC(config)

hop_size = aec.hop_size   # 160 samples (10ms @ 16kHz)
while has_audio:
    output = aec.process(mic_block, ref_block)
    erle = aec.get_erle()
    conf = aec.get_dtd_confidence()
```

C 實作（對齊 Python v2.0.0+v2.5.0）：`c_impl/src/aec.c`、`c_impl/src/pbfdkf.c`

---

## 九、Known Limitations

| 場景 | 行為 | 原因 |
|------|------|------|
| 冷啟動前 0.5–2s | 回音殘留明顯 | Filter 需 far-end 訊號收斂 |
| 設備快速移動 | 約 200ms 回音殘留 | EPC 觸發後重收斂期間 |
| movement + 雙講同時 | echo 無法完全抑制 | DT gate 抑制 adaptation，filter 無法追蹤新 echo path（已系統性驗證，三輪 ablation 確認為線性 AEC 架構固有限制）|
| 延遲超過 filter_length | 消除不完全 | 需調大 filter_length 或啟用 delay estimation |
| 嚴重非線性失真 | 部分殘留 | 線性模型無法建模 speaker 飽和（由後端 RES 處理）|
