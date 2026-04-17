# AEC v2.5.0 Changelog

**日期**：2026-04-17
**相對 v2.4.0**：RES/Kalman C-parity 對齊 + Python PBFDKF mu 修正；DT deg +0.003，其他持平

---

## 採用的修正

### RES FS-parity（C 端已於 v2.4.0 同步，本版 Python 補齊）

| 項目 | Python 位置 | C 位置 | 說明 |
|------|------------|--------|------|
| dt_per_bin 指數 1.3→1.1 | `aec.py` ResFilter.process | `res_filter.c` | 更軟的 DT mid-range 保護（dt=0.5：0.41→0.46）|
| render-based switching threshold | `aec.py` L1277 | `res_filter.c` | `far/error` 替代原本 `echo/error`，C/Py 對齊 |
| fb_erle near_power 平滑化 | `aec.py` L1190 | `res_filter.c` | 用 4-block 平滑 near_psd，C/Py 對齊 |
| filter_erle cap 統一 [0.5, 200] | `aec.py` L820 | `res_filter.c` | 原 C cap LF32/HF16（1/6 too tight），C/Py 對齊 |
| render-based gate `far_power > 1e-4` | `aec.py` L1268 | `res_filter.c` | 避免 frame 0 silence 時 latch render_based |
| ne_physical_floor = error_psd × 0.05 | `aec.py` L1448 | `res_filter.c` | 防止 ENR 在靜音 FS 段爆大 |
| DT ENR threshold relax（effective_dt > 0.4 時 1.5×）| `aec.py` L1463 | `res_filter.c` | high-coupling DT 邊界保護 |

### Kalman float32 對齊（Python 降精度至 float32 = C）

| 項目 | 說明 |
|------|------|
| `np.float32(self.delta)` in denominator | 防止 delta 字面量 1e-8（float64）把 denominator 升到 float64 → K 變 complex128 |
| `np.float32(1.0) - KX` in P update | 防止 P update 中間值升 float64 |
| `KX = np.real(...).astype(np.float32)` | 確保 KX 為 float32 |

### Python PBFDKF mu_ratio 修正

**問題**：PBFDKF + 無 RES 模式下，Python 從不呼叫 `_update_simple_mu_ratio`（ratio 停在 1.0）；C 永遠呼叫。

**修法**：補上 `if not self.res and not enable_dtd and not converged: _update_simple_mu_ratio(raw_output, far_end)`

**影響**：bare Kalman corr 0.9895 → 0.9919；production（RES 開啟）不受影響。

---

## 800-case 分數（balanced preset）

| 版本 | FS echo | DT echo | DT deg | NE deg |
|------|---------|---------|--------|--------|
| v2.3.0 | 3.504 | 4.163 | 2.282 | 4.008 |
| v2.4.0 | 3.487 | 4.164 | 2.293 | 4.007 |
| **v2.5.0** | **3.482** | **4.162** | **2.296** | **4.007** |

全部在 noise band（ΔDT deg +0.003、ΔFS echo -0.005）。

---

## C-Python 對比（v2.5.0 狀態）

| 場景 | Correlation | 說明 |
|------|------------|------|
| NE 單講 | ~1.0000 | 完全一致 |
| DT | 0.97-0.99 | 強對齊 |
| FS with residual | ~0.90 | kiss_fft vs numpy.fft 精度累積（不可消除）|

FS 剩餘 0.10 gap 根本原因：Kalman 正回饋將 FFT library 的 1e-6 精度差異累積為 ~5% W 差，在近完美 echo cancellation 時放大為 50% output ratio 差。非 logic bug，為 FFT library 差異固有限制。

---

## 同步狀態

- `main` + `feature/static-memory` 均已套用上述修正
- `feature/static-memory` 不含 Bug 6（分支結構體無 `erl_estimate` 欄位）
- C 側（`c_impl/`）：`res_filter.c` + `aec.c` + `pbfdkf.c` 已同步
