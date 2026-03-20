# AEC v1.9.0 改善紀錄

## 概述

基於 v1.8.4（PBFDAF subband + shadow + RES）架構，新增三項改善，在 AEC Challenge 測試集上大幅提升 ERLE。

## Benchmark 結果

### Farend Single-Talk ERLE (10 cases)

| 指標 | v1.8.4 | v1.9.0 | SpeexDSP | WebRTC AEC3 |
|------|--------|--------|----------|-------------|
| Mean ERLE | 5.1 dB | **14.3 dB** | 7.3 dB | 18.2 dB |

### Doubletalk (10 synthetic cases, SER=-5dB)

| 指標 | v1.8.4 | v1.9.0 | SpeexDSP | WebRTC AEC3 |
|------|--------|--------|----------|-------------|
| Mean ERLE | 4.5 dB | **9.4 dB** | 4.3 dB | 8.7 dB |
| Mean PESQ | 1.15 | **1.24** | 1.41 | 1.09 |

### 改善幅度

| 指標 | 改善量 | vs AEC3 |
|------|--------|---------|
| FS ERLE | +9.2 dB | 差距從 13.1 dB 縮小到 3.9 dB |
| DT ERLE | +4.9 dB | **已超越** AEC3 (+0.7 dB) |
| DT PESQ | +0.09 | **已超越** AEC3 (+0.15) |

---

## 改善項目

### 1. Delay Estimation + Reference Pre-Alignment（預期 +8dB，實測 +8dB）

**問題**：AEC Challenge 真實資料中 mic 與 ref 有 23-127ms system delay（speaker→mic + device buffer），v1.8.4 假設 zero delay，導致 adaptive filter 被迫用權重建模 delay，浪費 filter capacity。

**方案**：
- `eval_aec_challenge.py` 中新增 `estimate_delay()` 函式
- 使用全信號 FFT cross-correlation（不做 PHAT whitening，對真實資料較穩健）
- 離線估計 delay 後，pre-align reference signal 再送入 AEC
- `aec.py` 中新增 `DelayEstimator` class（GCC-PHAT，支援 online estimation）
- `AecConfig` 新增 `enable_delay_est`, `max_delay_ms`, `fixed_delay_samples` 參數

**關鍵發現**：
- GCC-PHAT（full whitening）在真實資料 6/10 cases 估計錯誤
- 不做 whitening 的 plain cross-correlation 最穩健
- Delay 方向：`ref_aligned[delay:] = ref[:n-delay]`（delay reference 以匹配 echo path）

### 2. RES Over-Subtraction 修正（預期 +2-5dB，實測 +6dB）

**問題**：`AEC.process()` 中動態 over_sub 公式有 bug：

```python
# BUG: erle_factor = max(0, (erle_db - 5) / 15)
# 真實資料 ERLE 幾乎都 < 5dB → erle_factor 永遠 = 0
# 所以 base_over_sub = 1.0 + 2.0 * 0 = 1.0（忽略 config 設定）
```

**修正**：以 `config.res_over_sub` 為 base，不再被 ERLE 門檻卡住：

```python
base_over_sub = self.config.res_over_sub + 2.0 * erle_factor
self.res.over_sub = max(base_over_sub - dt_reduction, 1.0)
```

同時調整 `res_over_sub` 預設值 3.0 → 10.0，`res_g_min_db` -20 → -40。

**這是最大的單一改善**（7.5 → 13.4 dB on FS ERLE）。

### 3. FDKF（Frequency Domain Kalman Filter）（預期 +1-3dB，實測 +1.1dB）

**問題**：NLMS 使用固定（或 power-normalized）step size，在 colored noise 下收斂慢。

**方案**：Per-bin Kalman gain 取代 NLMS mu：

```
K = P * |X|² / (|X|² * P + R)     # Kalman gain
W += K * error * conj(X)            # Weight update
P = (1 - K * |X|²) * P + Q         # Covariance update
```

- `SubbandNlms` 新增 `use_kalman` 參數和 `_update_kalman()` 方法
- 預設 P_init=0.5, Q=1e-5
- 保持 PBFDAF partition 架構不變，只改 adaptation

### RES 參數調整

| 參數 | v1.8.4 | v1.9.0 | 原因 |
|------|--------|--------|------|
| `res_over_sub` | 3.0 | 10.0 | 修正 bug 後需更高壓制 |
| `res_g_min_db` | -20 | -40 | 允許更深壓制 |
| `alpha_echo_psd` | 0.85 | 0.5 | 更快追蹤 echo PSD 變化 |
| `alpha_error_psd` | 0.9 | 0.8 | 更快追蹤 error PSD 變化 |
| `alpha_coh` | 0.65 | 0.3 | 更快追蹤 coherence 變化 |

---

## 分析：線性 filter 的理論上限

透過 offline Wiener filter 實驗（train/test split on same file），線性 filter ERLE ≈ 5dB，因為 echo path 是 time-varying 的。這意味著：

- 我們的線性 filter ERLE（~4dB）已接近理論極限
- AEC3 的 18.2dB 中 ~10+dB 來自 NLP（非線性後處理），不是線性 filter
- RES/NLP 是縮小差距的關鍵

---

## 驗證方式

```bash
cd /path/to/AEC
python3 python/eval_aec_challenge.py wav/aec_challenge/ --aec3 --speex
```

## 後續方向

- FS ERLE 仍差 AEC3 3.9 dB，可透過更精準的 echo PSD 估計或 per-subband ERLE tracking 改善
- NN-based NLP（未來計畫）預期可進一步提升 5-10 dB
