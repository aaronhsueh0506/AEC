# AEC 改進紀錄 v12 — v1.18.0 → v1.28.0

> 本文記錄 v1.18.0 到 v1.28.0 的所有改動、測試結果和實驗紀錄。

---

## 成績對比

### Blind Test（AEC Challenge 2021, 800 cases, fl=512, BALANCED preset）

| 指標 | v1.18.0 | v1.27.0 | v1.28.0 | 變化(total) | AEC3 | vs AEC3 |
|------|---------|---------|---------|------------|------|---------|
| FS echo_mos | 3.210 | 3.709 | **3.713** | **+0.503** | 3.963 | -0.250 |
| DT echo_mos | 4.000 | 4.273 | **4.370** | **+0.370** | 4.440 | -0.070 |
| DT deg_mos | 2.215 | 2.187 | **2.092** | -0.123 | 2.258 | -0.166 |
| NE deg_mos | 4.018 | 4.005 | **3.996** | -0.022 | 3.530 | **+0.466** |

### AEC Challenge（15 files, fl=512, BALANCED preset）

| 指標 | v1.18.0 | v1.27.0 | AEC3 |
|------|---------|---------|------|
| FS echo_mos | 3.210 | 3.34 | 3.39 |
| DT deg_mos | 3.21 | 3.11 | 3.13 |

---

## 有效改動（已保留）

### 1. Kalman R scaling by mu_scale（v1.22.0）
**位置**: `PBFDKF._update_weights()`
```python
R_scale = 0.3 + 0.7 * (1.0 - mu_mean)  # FS: 0.3, DT: 1.0
self.R = self.R * R_scale
```
**效果**: FS echo +0.12, DT deg -0.10
**原理**: FS 時降低 R 讓 Kalman gain K 更大 → filter 更快追蹤 echo path。
DT 時 R 不降 → K 小 → 保護 filter weights 不被近端污染。
**消耗**: 1 multiply per bin per frame

### 2. Q_low 1e-7 → 1e-5（v1.22.0）
**位置**: `AecConfig`, `PBFDKF.__init__()`
**效果**: 含在 #1 的效果中
**原理**: Q_low 太小（1e-7）讓 Kalman 的 P（error covariance）萎縮到零，
K 永久接近 0，filter 停止學習。1e-5 讓 P 維持合理下界。
**消耗**: 0

### 3. 收斂判定更嚴格 + warmup guard（v1.23.0）
**位置**: `AEC.process()` 收斂偵測區塊
```python
# 6dB × 6 frames → 10dB × 10 frames
# warmup 期間不進入收斂判定
if not self._filter_converged and self.near_power > 1e-8 and self._warmup_frames <= 0:
    if inst_erle > 10.0:  # was 6.0
        if self._conv_counter >= 10:  # was 6
```
**效果**: FS echo +0.06, linear ERLE +0.12
**原理**: 避免 filter 在收斂中途就切到 Q_low（停止學習）。
**消耗**: 0

### 4. R alpha 0.95 → 0.98（v1.23.0）
**位置**: `PBFDKF.__init__()`
**效果**: 含在 #3 的效果中
**原理**: R 追蹤 error_psd 的 EMA。0.95（TC=20 frames）太快，
echo 突變時 R 暴漲壓死 K。0.98（TC=50 frames）更穩定。
**消耗**: 0

### 5. ENR offset +1.0 → adaptive（v1.24.0）⭐ 最關鍵
**位置**: `ResFilter.process()` ENR 計算
```python
# 舊：enr = residual_echo / (nearend_est + 1.0)
# 新：enr = residual_echo / (self.error_psd + 1e-10)
```
**效果**: FS echo +0.17, DT echo +0.14, DT deg -0.10
**原理**: +1.0 offset 對 PSD 量級 0.01-10 太大，導致 ENR 永遠 < 1.0，
RES gain 無法到達 g_min。改成直接用 error_psd 做分母（AEC3 style）。
**消耗**: 0

### 6. ENR 公式改 echo/error（v1.25.0）
**位置**: `ResFilter.process()`
```python
# 舊：nearend_est = max(error - echo, 0); enr = echo / (nearend_est + offset)
# 新：enr = residual_echo_psd / (self.error_psd + 1e-10)
```
**效果**: DT deg +0.06, FS echo 持平
**原理**: `error - echo` 在 echo ≈ error 時 nearend_est ≈ 0 → enr 暴漲 → DT 語音被壓。
直接用 error_psd 做分母（AEC3 風格）避免減法的不穩定。
**消耗**: 0（反而少一個 max 運算）

### 7. Fixed g_min（v1.24.0）
**位置**: `ResFilter.process()` effective_g_min 計算
```python
# 舊：effective_g_min = g_min + (1 - g_min) * (1 - far_activity)
# 新：effective_g_min = g_min  （固定值）
```
**效果**: FS echo +0.10, DT deg -0.02
**原理**: 舊版在 far_activity 低時抬高 g_min → 低 echo 段不壓制（output 停在 -10dB）。
AEC3 用固定 g_min，gain 純由 ENR 決定。
**消耗**: 0（反而少幾個運算）

### 8. CNG crossfade sqrt(1-G²)（v1.26.0）⭐ 效果顯著
**位置**: `ResFilter.process()` CNG injection
```python
# 舊：只在 far_power <= 1e-4 時注入 comfort noise
# 新：cn_gain = sqrt(1 - G²)，suppression 越深 noise 越大
cn_gain = np.sqrt(np.maximum(1.0 - self.gain_smooth ** 2, 0.0))
enhanced_spec += cn_gain * noise_mag * exp(1j * random_phase)
```
**效果**: FS echo +0.22, DT deg +0.03（全面改善）
**原理**: Deep suppression 時 output 接近零（不自然靜音），
AECMOS 可能因此扣分。CNG 填充讓靜音段自然。
**消耗**: 每 frame 1 個 sqrt + 1 個 random phase 生成

### 9. Continuous attack speed blend（v1.27.0）⭐ 效果顯著
**位置**: `ResFilter.process()` gain temporal smoothing
```python
# 舊：if nearend_state: alpha=0.85 else: alpha=0.3（binary）
# 新：
fs_confidence = far_activity * (1 - dt_indicator) ** 2
alpha_attack = alpha_slow + (alpha_fast - alpha_slow) * fs_confidence
```
**效果**: FS echo +0.32 vs binary nearend_state 版
**原理**: binary nearend_state 在 FS 78% 誤判為 nearend → slow attack → FS 壓不深。
用 continuous `fs_confidence` 避免 binary 判斷的問題。
**消耗**: 2 multiply + 1 power per frame

### 10. EMR noise masking（v1.26.0）
**位置**: `ResFilter.process()` ENR gain 計算後
```python
emr = residual_echo_psd / (self.noise_psd + 1e-10)
g_emr = clip(emr_transparent / emr, 0, 1)
g = max(g, g_emr)  # echo < noise → don't suppress
```
**效果**: ~0（noise_psd 在多數 case 很小）
**原理**: echo 被背景噪音遮蔽時不需壓制，減少不必要的語音損傷。
**消耗**: 1 divide per bin per frame

### 11. Q modulation DT（v1.26.0）
**位置**: `PBFDKF._update_weights()`
```python
q_scale = 0.1 + 0.9 * mu_mean  # FS: 1.0, DT: 0.1
Q_modulated = self.Q * q_scale
```
**效果**: ~0（對 AECMOS 影響微小）
**原理**: DT 時降低 Q → filter 不把 nearend error 當 echo path 變化。
Kalman 專屬（NLMS 沒有 Q）。
**消耗**: 1 multiply per bin per frame

### 12. 其他小改動
- Shadow copy threshold 0.7→0.65, hysteresis 5→3
- Q_gated soft floor Q×0.05
- alpha_coh 0.3→0.65
- ERLE cap HF 4→16, LF 8→32
- Echo boost DT guard (0.5×coh2, DT off)
- EPC on delay shift
- Preset 重校準（四級適配新 ENR 架構）

---

## 測試但 revert / 無效的改動

### noise_psd continuous tracking
```python
# 讓 noise_psd 在所有時間更新（不只 far-end silence）
noise_candidate = alpha * noise_psd + (1-alpha) * error_pwr
noise_psd = min(noise_candidate, error_pwr * 2)
```
**結果**: FS echo **-0.24**, DT deg +0.35
**原因**: far-end active 時 error_psd 含 echo → noise_psd 被 echo 污染 → CNG noise 太大

### R alpha adaptive (0.98 - 0.10 × mu_mean)
```python
alpha_r_eff = 0.98 - 0.10 * mu_mean   # DT→0.98, FS→0.88
```
**結果**: 效果微小（±0.01）
**原因**: R scaling 已經在做類似的 mu-based 調整

### NE_TRIGGER_FRAMES 12→5
**結果**: 無明顯效果
**原因**: nearend 進入速度不是瓶頸，是 nearend detection 準確性

### filter_length 1024
**結果**: FS echo +0.003 vs fl=512
**原因**: RES 是瓶頸，不是 linear filter length

### coh2-based nearend detection
```python
avg_coh2 = mean(coh2[8:])
if avg_coh2 < 0.2: nearend
```
**結果**: FS 100% nearend（更差）
**原因**: coh2 在 filter ERLE 低時只有 0.1-0.19，FS 和 DT 都低

### filter-based |H(f)|² path gain
```python
H_f = np.sum(self.filter.W, axis=0)
filter_pg = np.abs(H_f) ** 2
```
**結果**: FS echo -0.4
**原因**: fl=512 的 filter 只建模部分 echo path → |H|² 嚴重低估

### NL echo floor (near_psd/far_psd × far_psd)
**結果**: FS echo +0.14, DT deg -0.18（多種版本都有 trade-off）
**原因**: near_psd 在 DT 含近端 → path_gain 被污染 → FS/DT 混淆

### Audibility weighting
```python
audibility = clip(residual_echo / noise_floor, 0, 1)
residual_echo *= audibility
```
**結果**: FS echo -0.02
**原因**: 降低了 residual echo → ENR 降低 → gain 升高

### Implicit DTD (energy + ERLE drop)
```python
dt_energy = mic_power > 4.0 * far_power
dt_erle = inst_erle < erle_smooth * 0.5
dt_detected = dt_energy or dt_erle
```
**結果**: FS 48% 誤觸發
**原因**: FS 高 echo coupling 也讓 mic > 4× far

### Coherence-based gain blend
```python
g_coh = (1 - coh2) / (1 + (over_sub - 1) * coh2)
g = (1 - erle_factor) * g_coh + erle_factor * g_enr
```
**結果**: FS echo -0.50
**原因**: coh2 在 FS 太低（0.15）→ g_coh ≈ 0.53 → 壓不下去

---

## 架構差異分析（vs WebRTC AEC3）

### 已對齊
- ENR masking with two-tuning（nearend/normal mode）
- Shadow filter（dual-filter divergence control）
- EPC detection（echo path change → Q reset）
- CNG crossfade（sqrt(1-G²)）
- EMR noise masking
- Fixed g_min（不受 far_activity 影響）

### 仍有差距
| 功能 | AEC3 | 我們 | 影響 |
|------|------|------|------|
| Nearend detection | ENR + SNR>30 | ENR only（78% FS false positive）| FS echo -0.25 |
| ERLE estimator | 3 個（fullband+subband+signal-dep） | 1 個 | ENR 精度 |
| Stationarity estimator | 有 | 無 | echo/noise 區分 |
| Delay estimation | Matched filter | GCC-PHAT + histogram | 部分 case delay 不準 |
| Filter length | 768 (48ms) | 512 (32ms) | 可接受 |

### 根本限制
- **Nearend detection 78% FS false positive**：所有靠 nearend_state 的機制都受影響。
  ENR-based detection 在 filter ERLE 低時不可靠（echo_psd 低估 → ENR 低 → 誤判）。
  coh2-based detection 也不行（coh2 在 FS 也低）。
  目前用 continuous attack blend 繞過此問題。

---

## Filter Length 與資源消耗

| filter_length | 時間 | partitions | 記憶體 (含 shadow) | vs fl=512 AECMOS |
|--------------|------|-----------|-------------------|-----------------|
| 512 | 32ms | 4 | ~48KB | baseline |
| 1024 | 64ms | 7 | ~84KB | +0.003 FS echo |
| 3200 | 200ms | 20 | ~240KB | 未測（RES 是瓶頸） |

**結論**: fl=512 足夠，加長 filter 不改善 AECMOS（RES suppression 是主要瓶頸）。
