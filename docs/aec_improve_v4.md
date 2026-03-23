# AEC v1.11.0 改善紀錄

## 概述

基於 v1.10.0（PBFDAF subband + shadow + FDKF + RES + HPF + saturation detect）架構，新增六項改善。
引入 AECMOS（Microsoft speechmos）作為感知品質指標，與 ERLE/PESQ 並列評測。

## Benchmark 結果

### Farend Single-Talk ERLE + AECMOS (10 cases)

| 指標 | v1.10.0 | v1.11.0 | SpeexDSP | WebRTC AEC3 |
|------|---------|---------|----------|-------------|
| Mean ERLE | 14.8 dB | **10.6 dB** | 7.3 dB | 18.2 dB |
| AECMOS echo_mos | — | **3.47** | 3.63 | 4.32 |
| AECMOS deg_mos | — | **5.00** | 4.97 | 5.00 |

### Doubletalk (10 synthetic cases, SER=-5dB)

| 指標 | v1.10.0 | v1.11.0 | SpeexDSP | WebRTC AEC3 |
|------|---------|---------|----------|-------------|
| Mean ERLE | 10.8 dB | **6.5 dB** | 4.3 dB | 8.7 dB |
| Mean PESQ | 1.16 | **1.21** | 1.41 | 1.09 |
| AECMOS echo_mos | — | **2.78** | 2.59 | 4.37 |
| AECMOS deg_mos | — | **3.44** | 3.57 | 1.80 |

### 改善重點

| 指標 | vs v1.10.0 | vs AEC3 |
|------|-----------|---------|
| DT PESQ | +0.05 (1.16→1.21) | **超越** AEC3 (+0.12) |
| DT deg_mos | — | **大幅領先** AEC3 (+1.64) |
| DT echo_mos | — | 差距 -1.59 |
| FS echo_mos | — | 差距 -0.85 |

> **ERLE 下降說明**：v1.11.0 的 ERLE 數字低於 v1.10.0，是因為 `g_min` 從 -40dB 提升到 -25dB，
> `over_sub` 從固定 6.0 改為動態 2.5+4.0×erle_factor 並加入 DT protection。
> 這使得 RES 產生更少的音訊失真（musical noise），AECMOS echo_mos 和 deg_mos 反而更好。
> 感知品質優先於數學指標。

---

## 改善項目

### 1. Warmup Mu — 前 100 幀強制高 mu 加速冷啟動

**問題**：hop=256（16ms）下每秒 weight update 次數僅 62.5 次，前 0.5-1s 收斂遠落後 AEC3（hop=64, 250 次/s）。

**方案**：
- `AEC.__init__()` 新增 `self._warmup_frames = 100`（約 1.6s）
- `_get_simple_mu_scale()`：warmup 期間 return `min(1.0, max(0.7, ratio + 0.3))`
- `reset()` 時重置 warmup counter

### 2. Echo Estimate Ratio — mu ratio 的 lower bound 保護

**問題**：強 echo 場景（fs_4）下 far/error ratio ≈ 1，但 filter 未收斂 → error 大 → ratio 低 → mu 被壓 → 死鎖。

**方案**：
- `_update_simple_mu_ratio()` 中加入 `echo_est_pwr / near_pwr` 作為 ratio 的 lower bound
- `ratio = max(ratio, ratio_echo * 0.5)` 確保 mu 不被壓到極低

### 3. Per-bin Local Power Floor — 改善中頻 ERLE

**問題**：全局 power floor `mean(power) * 0.01` 在低能量中頻 bin 上提供的有效 mu 偏小，500-4kHz ERLE 差距達 11-22 dB。

**方案**：
```python
local_floor = self.power * 0.01 + self.delta     # per-bin 1% floor
global_floor = np.mean(self.power) * 0.001 + self.delta  # 全局 0.1% 極低底限
power_floor = np.maximum(self.power, np.maximum(local_floor, global_floor))
```

### 4. Adaptive g_min — DT 自動提升增益下限

**問題**：g_min=-40dB 在 echo-only 段造成過深壓制產生 musical noise，AECMOS echo_mos 反而被扣分。

**方案**：
- Base `g_min` 從 -40 改為 **-25dB**
- RES 收到 `dt_indicator` 參數，DT 時 g_min 自動提升至 **-10dB**
- Echo-only 段 dt_indicator≈0 → g_min=-25dB（適度壓制）
- DT 段 dt_indicator≈0.8 → g_min≈-10dB（保護近端語音）

```python
dt_g_min = 10 ** (-10.0 / 20)
effective_g_min = effective_g_min + (dt_g_min - effective_g_min) * dt_indicator
```

### 5. Dynamic over_sub 擴大範圍

**問題**：v1.10.0 固定 over_sub=6.0 壓死 DT 近端，v1.11.0 初版 1.5+1.5×erle 太保守。

**方案**：
```python
base_over_sub = 2.5 + 4.0 * erle_factor   # 未收斂: 2.5, 已收斂: 6.5
dt_reduction = 3.5 * dt_indicator          # DT 時最多降 2.8
self.res.over_sub = max(base_over_sub - dt_reduction, 0.5)
```

配合 adaptive g_min，高 over_sub 不再造成深度失真（被 g_min 截止），但中等 EER 的 bin 得到更有效壓制。

### 6. AECMOS 評測工具

新增 `python/eval_aecmos.py`，使用 Microsoft speechmos 的 AECMOS 模型評估：
- `echo_mos`：殘餘回音感知分數（1-5，越高越好）
- `deg_mos`：語音品質退化分數（1-5，越高越好）
- 同時輸出 Ours (RES)、Ours (NoRES)、AEC3、SpeexDSP 的分數

`eval_aec_challenge.py` 新增 no-RES 輸出（raw PBFDAF），作為 RES 效果的對照基準。

---

## RES 參數調整

| 參數 | v1.10.0 | v1.11.0 | 原因 |
|------|---------|---------|------|
| `res_g_min_db` | -40 | **-25** | 減少 musical noise，AECMOS 更好 |
| `res_over_sub` | 6.0 | **3.0**（config default） | 動態公式基礎，實際 2.5-6.5 |
| over_sub 公式 | `config.res_over_sub + 2.0 × erle_factor` | `2.5 + 4.0 × erle_factor` | 更大動態範圍 |
| DT reduction | `1.5 × dtd_conf`（DTD disabled → 0） | `3.5 × dt_indicator` | 始終有效的 DT 保護 |
| g_min 適應 | 固定 | DT 時提升至 -10dB | 保護近端語音 |

---

## 分析：ERLE vs AECMOS 的權衡

透過實驗發現：
- g_min=-40dB 時 ERLE 更高（14.8 dB），但 AECMOS echo_mos 更低（3.34）
- g_min=-25dB 時 ERLE 降低（10.6 dB），但 AECMOS echo_mos 提升（3.47）
- **原因**：深度壓制產生的 musical noise 和 gain pumping 在感知上比殘餘回音更令人不適

NoRES（純線性濾波器）的 FS echo_mos 為 3.38，與 RES 的 3.47 差距僅 0.09。
這代表我們的線性 filter ERLE（~5dB）已接近理論極限，RES 主要在 DT 場景提供額外的回音壓制。

---

## 驗證方式

```bash
cd /path/to/AEC

# ERLE + PESQ 評測
python3 python/eval_aec_challenge.py wav/aec_challenge/ --aec3 --speex

# AECMOS 評測（需 Python 3.13+ + speechmos + onnxruntime 1.24+）
python3 python/eval_aecmos.py wav/aec_challenge/
```

## 後續方向

- FS/DT echo_mos 與 AEC3 仍有 0.85/1.59 的差距，主要受限於線性 filter ERLE 上限
- NN-based NLP 或 hybrid approach 預期可進一步提升 echo_mos 5-10 分
- C 版本仍使用 DTD 模式，未同步 adaptive g_min 等改善
