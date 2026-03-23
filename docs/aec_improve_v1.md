# AEC 修改建議

## 問題總覽

| 優先 | 位置 | 問題 | 預期改善 |
|---|---|---|---|
| 🔴 P0 | `SubbandNlms.process()` L257–265 | Per-partition time-domain truncation 邏輯錯誤 | ERLE 大幅提升，尤其長 echo path |
| 🔴 P0 | `SubbandNlms.process()` L255 | Power floor 用 `max * 1e-4`，低頻 bin step size 爆大 | 收斂速度與穩定性 |
| 🟡 P1 | `ResFilter.process()` L369–396 | EER 計算有誤：far_scale 無物理意義、coh2 與 eer_linear 單位不同直接 max | 減少 residual echo，降低 near-end distortion |
| 🟡 P1 | `AEC.__init__()` + `_compute_mu_scale()` | Coherence DTD 在收斂前 warmup 不足，產生 false positive | 收斂期穩定性 |
| 🟡 P1 | `SubbandNlms.process()` L247–249 | Error spec 前半 zero-pad 引入 circular convolution aliasing | Weight update 品質 |

---

## P0-1：移除 per-partition time-domain truncation，改用 leakage

### 問題說明

`SubbandNlms.process()` 第 257–265 行，每個 partition 做了 time-domain truncation：

```python
# 現有程式碼（有問題）
for p in range(self.n_partitions):
    p_idx = (curr_p - p) % self.n_partitions
    grad = self.error_spec * np.conj(self.X_buf[p_idx])
    self.W[p] += mu_eff * grad

    # Constraint: time-domain truncation  ← 這段有問題
    w_time = np.fft.irfft(self.W[p], self.block_size)
    w_time[hop:] = 0
    self.W[p] = np.fft.rfft(w_time)
```

**根本原因**：`w_time[hop:] = 0` 把每個 partition 的 weights 都截成前 `hop` 個 sample，等於強迫所有 partition 都建模 echo path 的第 0 段，後面的 partition 學不到任何東西。這抵消了 multi-partition 的所有優勢，讓 PBFDAF 退化成 single-partition FDAF。

### 修改方式

刪除 time-domain truncation，改用 leakage 維持穩定性：

```python
# 修正後
for p in range(self.n_partitions):
    p_idx = (curr_p - p) % self.n_partitions
    grad = self.error_spec * np.conj(self.X_buf[p_idx])
    self.W[p] = 0.9999 * self.W[p] + mu_eff * grad
    # 移除 time-domain truncation（w_time 那三行完全刪除）
```

leakage 係數 `0.9999` 可以做成 config 參數（對應現有的 `AecConfig.leak`，目前只用在 `NlmsFilter` 但 `SubbandNlms` 沒用到）。

---

## P0-2：修正 per-bin power floor

### 問題說明

`SubbandNlms.process()` 第 255 行：

```python
# 現有程式碼（有問題）
power_floor = np.maximum(self.power, np.max(self.power) * 1e-4)
```

當某些 bin 的 `self.power` 遠低於 `max * 1e-4`（例如靜音頻段），floor 仍可能很低，導致那些 bin 的 `mu_eff` 過大，weights 爆炸或震盪。

### 修改方式

```python
# 修正後：用全局平均做 floor，更能反映真實訊號水位
global_floor = np.mean(self.power) * 0.01 + self.delta
power_floor = np.maximum(self.power, global_floor)
mu_eff = (self.mu * mu_scale) / (power_floor * self.n_partitions + self.delta)
```

---

## P1-1：修正 ResFilter 的 EER 計算

### 問題說明

`ResFilter.process()` 第 369–396 行，有兩個獨立問題：

**問題 A**：`far_scale` 無物理意義

```python
# 現有程式碼（有問題）
far_scale = min(far_power / (np.mean(error_pwr) + 1e-10), 1.0)
echo_pwr_scaled = echo_pwr_linear * far_scale
```

當 filter 已收斂（echo estimate 準確）時，這個縮放反而把準確的 echo PSD 壓低，導致殘餘 echo 沒被充分 suppress。

**問題 B**：`coh2` 與 `eer_linear` 單位不同直接 `max`

```python
# 現有程式碼（有問題）
eer_linear = self.echo_psd / (self.error_psd + eps)   # 值域 [0, ∞)
eer = np.maximum(eer_linear, coh2)                     # coh2 值域 [0, 1]
```

`eer_linear` 可以 > 1（echo 估計 > error，filter 過衝時），`coh2` 最大是 1。直接 max 在 `eer_linear` > 1 的情況下 `coh2` 完全失效；在 `eer_linear` < 1 時又可能因為 `coh2` 虛高導致過度 suppress。

### 修改方式

```python
# 修正後

# A: 移除 far_scale，直接使用 echo estimate PSD
self.echo_psd = self.alpha_psd * self.echo_psd + (1 - self.alpha_psd) * echo_pwr_linear
self.error_psd = self.alpha_psd * self.error_psd + (1 - self.alpha_psd) * error_pwr

# B: coh2 作為 soft gate，而不是直接 max
eps = 1e-10
eer_linear = self.echo_psd / (self.error_psd + eps)
# coherence 高（echo 佔比大）→ 維持 suppress；coherence 低（near-end 多）→ 減半 suppress
eer = eer_linear * (0.5 + 0.5 * coh2)

# Gain 計算改用線性 Wiener-inspired 公式（原來是 1/(1+over_sub*eer)，維持不動）
g = 1.0 / (1.0 + self.over_sub * eer)
g = np.maximum(g, self.g_min)
g[quiet_mask] = 1.0
```

---

## P1-2：Coherence DTD warmup 與 convergence flag 綁定

### 問題說明

`AEC.__init__()` 第 749 行，coherence DTD 的 warmup 是固定 50 frames：

```python
warmup = 50
```

但 50 frames × 256 samples / 16000 Hz = 0.8 秒，遠早於 filter 實際收斂（通常需要 2–5 秒）。收斂前 `error ≈ mic`，`coherence(error, far)` 非常高 → DTD 判定「沒有 double talk」→ 繼續以 full mu update → 看起來正常但實際上 DTD 保護完全失效。

`_filter_converged` flag 在 `_compute_mu_scale()` 有使用（第 904 行），但 coherence DTD 的 warmup 和它是獨立的。

### 修改方式

在 `AEC.process()` 中，coherence DTD 更新前加 convergence check：

```python
# 修正後：在 dtd_coherence.detect_block() 呼叫前加條件
if self.dtd_coherence and self._filter_converged:
    # 原有的 detect_block() 呼叫...
```

同時調高收斂判斷門檻，從 3 dB 改為 6 dB（第 1115 行）：

```python
# 現有
if inst_erle > 3.0:
    self._filter_converged = True

# 修正後
if inst_erle > 6.0:
    self._filter_converged = True
```

---

## P1-3：修正 error spec 的 zero-padding aliasing

### 問題說明

`SubbandNlms.process()` 第 247–249 行：

```python
# 現有程式碼（有問題）
error_time = np.zeros(self.block_size, dtype=np.float32)
error_time[hop:] = output      # 前半 zero-pad
self.error_spec = np.fft.rfft(error_time)
```

在 overlap-save 框架中，把 error 前半 zero-pad 後取 FFT，再用這個 `error_spec` 去更新 weights，會引入 circular convolution 的 aliasing 誤差，讓 gradient 估計偏移。

### 修改方式

overlap-save 正確做法：error 不需要 zero-pad，直接使用完整 block 的 error 做 FFT，weight update 本身的 linear constraint 靠前面的 partition circular buffer 保證：

```python
# 修正後

# Step 1: 計算完整 block 的 error（近端 - echo estimate）
echo_time = np.fft.irfft(self.echo_spec, self.block_size)
full_error = self.near_buffer - echo_time          # 完整 block，不只後半
output = full_error[hop:]                          # 輸出仍取後半

# Step 2: error spec 用完整 block（不 zero-pad）
self.error_spec = np.fft.rfft(full_error)

# Weight update 不變
```

---

## 參數建議

修改完成後，建議一併調整以下預設值：

| 參數 | 現有值 | 建議值 | 原因 |
|---|---|---|---|
| `AecConfig.leak` | `0.99999` | 只用於 NlmsFilter，SubbandNlms 改由程式碼內 `0.9999` 控制 | 統一管理 |
| `SubbandNlms.alpha_power` | `0.9` | `0.92` | 避免 power estimate 對突發訊號反應過快，影響 step size 穩定性 |
| `AecConfig.mu`（SUBBAND） | `0.5` | `0.3`（修正 truncation 後不需要大 mu 補償） | P0-1 修正後收斂速度已夠 |
| `AecConfig.res_over_sub` | `1.0` | `1.5` | P1-1 修正 EER 計算後，over_sub 需要補償原先被低估的 eer |

---

## 測試驗證方式

建議依序驗證：

1. **P0-1 驗證**：對比修改前後的 ERLE（cumulative），長 echo path（filter_length >= 1024）改善最明顯
2. **P0-2 驗證**：觀察 weights 在靜音後重新接音時是否有震盪（oscillation）消失
3. **P1-1 驗證**：double talk 段落的 near-end 音質（是否有過度壓制）
4. **P1-2 驗證**：觀察 confidence_history 前幾秒是否有異常高值消失
5. **P1-3 驗證**：收斂速度（達到 20 dB ERLE 需要的 frame 數）