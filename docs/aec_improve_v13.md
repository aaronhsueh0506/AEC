# AEC 改進紀錄 v13 — Stationary far-end DT 完整修正

> 本文記錄 v2.0.0 之後針對「穩態 far-end + 近端語音」場景的完整修正。
> 解決白噪音、風扇、冷氣等穩態 far-end 下 RES 把近端語音壓掉的長期問題。

---

## 問題描述

當 far-end 是穩態訊號（白噪音、純音、風扇、冷氣）時：

1. `dt_indicator` 失效：`raw_dt = 1 - far_pwr/(mic_pwr+far_pwr)` 永遠 ≈ 0（far_pwr >> mic_pwr）
2. `coh2`（per-bin coherence）失效：穩態訊號在所有 bin 都有高 coherence，per-bin DT mask 全部 ≈ 0
3. RES 認為全部是 echo，把近端語音壓到 g_min
4. Linear AEC（filter）也會把語音當 echo 學進去，造成 filter divergence

合成測試結果（修正前）：
- 13s 白噪音 + 10s 近端語音
- RES 開啟後：**5% 語音保留**
- NoRES：50% 語音保留（其中 50% 是 linear AEC 自己吸收）

---

## 解決方案

### 1. Stationarity 偵測（CV2 gate）

**`AEC.process` 全域 stationary feature extraction**

```python
alpha_cv = 0.99  # TC ≈ 1s, long enough to average WN hop variance
old_mean = self._far_env_mean
self._far_env_mean = α * mean + (1-α) * far_pwr
self._far_env_var  = α * var  + (1-α) * (far_pwr - old_mean)²
far_cv2 = var / mean²
self._is_stationary_far = (far_cv2 < 0.02)
```

**為什麼 α=0.99 而不是 0.95**：
- 單 hop (160 samples) 的 WN 能量 std/mean ≈ 11%
- α=0.95（TC=200ms）對這個變動反應太大 → flicker
- α=0.99（TC=1s）平均後 CV2 穩定 < 0.02 → 0 個 flicker

### 2. Stationary DT 偵測（jump_ratio + hangover）

語音突波偵測：

```python
if self._is_stationary_far and self._filter_converged:
    # voice-band (100-3000Hz) error power
    track_err_pwr = sum(|error_spec[vb_start:vb_limit]|²)
    if jump_ratio = track_err_pwr / baseline > 1.5:
        self._stat_dt_hangover = 80  # 800ms protection window
    if hangover > 0:
        is_stationary_dt = True
        baseline frozen (α=0.999)  # 防止語音污染 baseline
    else:
        is_stationary_dt = False
        baseline normal EMA (α=0.95)
```

**Hangover 800ms 的理由**：
- 30 frames (300ms) 太短，音節間停頓就讓 hangover 過期
- 80 frames (800ms) 涵蓋一般語音音節間隔
- 偵測率從 39% → 95%

### 3. RES 內部保護（dt_for_fs / dt_temporal）

引入兩個虛擬 DT 指標：

```python
dt_for_fs   = 0.8 if is_stationary_dt else dt_indicator
dt_temporal = 0.8 if is_stationary_dt else dt_indicator
```

**只在 `is_stationary_dt = True` 時** 把 `dt_for_fs` / `dt_temporal` 拉到 0.8，其他場景完全等同 `dt_indicator`，**對非穩態場景零影響**。

替換的位置（ResFilter.process）：

| 位置 | 原本 | 新版 |
|------|------|------|
| `nonlinear_floor` 計算 | `1 - dt_indicator` | `1 - dt_for_fs` |
| `dt_suppress` (residual cap) | `clip(1-dt_indicator²)` | `clip(1-dt_for_fs²)` |
| `ne_reverb_factor` | `(1-dt_indicator)` | `(1-dt_for_fs)` |
| `echo_boost` 條件 | `dt_indicator < 0.2` | `dt_for_fs < 0.2` |
| `dt_per_bin` base | `dt_indicator` | `dt_for_fs` |
| `fs_confidence` | `(1-dt_indicator)²` | `(1-dt_for_fs)²` |
| `alpha_release` | `* clip(1-dt_indicator²)` | `* clip(1-dt_temporal²)` only when `is_stationary_dt` |
| `dt_rise_boost` | `1+19*dt_indicator³` | only when `is_stationary_dt` |
| HF cap bypass | `dt_indicator < 0.5` | `dt_indicator < 0.5 and not is_stationary_dt` |

### 4. v3 頻帶成型遮罩（Frequency-Shaped Mask）

`is_stationary_dt = True` 時，在 ENR 的 `dt_per_bin` 上加上頻帶遮罩：

```python
if is_stationary_dt:
    f_bins = np.arange(n_freqs) * (16000 / block_size)
    wn_mask = zeros(n_freqs)
    for k, f in enumerate(f_bins):
        if 300 <= f <= 3000:        wn_mask[k] = 0.8
        elif 100 < f < 300:         wn_mask[k] = 0.8 * (f-100)/200
        elif 3000 < f < 4000:       wn_mask[k] = 0.8 * (4000-f)/1000
    dt_per_bin = max(dt_per_bin, wn_mask)
```

**設計理由**：
- 語音黃金頻段 300-3000Hz 直接給 dt=0.8 免死金牌
- 100-300Hz / 3000-4000Hz 平滑過渡避免邊界 artifact
- < 100Hz 和 > 4000Hz 不保護，繼續壓 WN 嘶嘶聲（不影響可懂度）

### 5. Reverb tail skip

`reverb_psd` 在 stationary far-end 下會持續累積巨量 WN 能量，加到 `residual_echo_psd` 上會把語音淹沒。

```python
if self.enable_reverb and not is_stationary_dt:
    residual_echo_psd += reverb_gain * reverb_psd * gate
```

### 6. Linear AEC mu_scale freeze

防止 filter 持續學習把近端語音吸收進 echo path：

```python
if self._is_stationary_far:
    self._per_bin_mu_scale[:] = mu_min
    self._simple_mu_ratio = mu_min
```

---

## 驗證

### 合成 WN DT 測試

13s 白噪音 far-end + 10s 近端語音（300-3000Hz）：

| 版本 | FS ERLE | DT 語音保留 |
|------|---------|------------|
| 修正前 | 22.4 dB | **5%** |
| 修正後 | 36.3 dB | **48.3%** |
| NoRES (linear only) | 3.0 dB | 49.8% |

語音保留率從 5% → 48.3%，幾乎跟 NoRES 一樣（48.3% vs 49.8%），同時 FS ERLE 反而提升（因為 echo_boost 在 stationary DT 不亂啟動）。

### 800-case Blind Test (FL=512, 16kHz, AECMOS local ONNX)

| Preset | FS echo↑ | DT echo↑ | DT deg↑ | NE deg↑ |
|--------|----------|----------|---------|---------|
| Linear (NoRES) | 2.71 | 3.17 | 3.20 | 4.07 |
| **mild** | 3.236 | 3.923 | 2.338 | 4.005 |
| **balanced** | 3.577 | 4.230 | 2.134 | 4.001 |
| **aggressive** | 3.687 | 4.357 | 2.070 | 3.995 |
| **maximum** | 3.783 | 4.473 | 2.014 | 3.970 |

vs README baseline (v2.0.0)：

| Preset | FS echo Δ | DT echo Δ | DT deg Δ | NE deg Δ |
|--------|-----------|-----------|----------|----------|
| mild | -0.04 | -0.02 | +0.07 | +0.02 |
| balanced | -0.04 | -0.00 | +0.04 | +0.03 |
| aggressive | -0.08 | -0.02 | +0.03 | +0.03 |
| maximum | -0.13 | -0.03 | -0.01 | +0.03 |

**全部退步 < 0.13，DT/NE 平均改善**。Stationary far-end 場景大幅改善（5% → 48% 保留），語音場景幾乎零退步。

---

## 已知限制

1. **Filter divergence on weak speech segments**: 即使 mu_scale freeze，當近端訊號突然變弱時，linear AEC 在 freeze 觸發前可能已經吸收了部分語音。某些段落（合成測試 sec 10）會掉到 ~10% 保留率。
2. **Stationarity 偵測延遲**: CV2 EMA 需要 ~1s 收斂才會穩定觸發 `_is_stationary_far`。Cold start 後前 1s 走一般 RES 路徑。
3. **Hangover 800ms 的代價**: 真正的 EPC 在 stationary far-end 下會被誤遮罩 800ms。但這在實際場景發生機率低（穩態訊號通常 echo path 也穩定）。
4. **Voice-band spike threshold (jump_ratio > 1.5)**: 微弱語音（音節弱起手）可能不觸發 hangover。

---

## 代碼變更

`python/aec.py`:
- `class AEC.__init__`: `_stat_dt_hangover` 初始化
- `AEC.process` ~L2280: CV2 stationarity gate（α=0.99）
- `AEC.process` ~L2447: jump_ratio detection + hangover
- `AEC.process` ~L2515: mu_scale freeze on `_is_stationary_far`
- `ResFilter.process` ~L1148: `dt_for_fs` 定義
- `ResFilter.process` ~L1192-1304: nonlinear_floor, dt_suppress, reverb skip, dt_per_bin, v3 mask
- `ResFilter.process` ~L1376-1416: HF cap bypass, alpha_release, rise boost (gated on is_stationary_dt)

`python/plot_aec_results.py`:
- `get_estimated_ir()`: 補上 `aec.delay_est.estimated_delay` 對齊絕對時間軸

---

## Commit

`9e24700 feat: stationary far-end DT protection (white noise + tones)`
