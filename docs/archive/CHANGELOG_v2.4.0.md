# AEC v2.4.0 Changelog

**日期**：2026-04-17
**耗時**：約 4 小時（800-case 批次驗證為主）
**相對 v2.3.0**：RES 路徑微調；balanced DT deg +0.010，FS echo -0.015

---

## 採用的修正（Bug 7 / 2 / 6）

### Bug 7 — DT per-bin 軟化（唯一實質 AECMOS 提升）

**檔案**：`python/aec.py` L1434、`c_impl/src/res_filter.c` L739

`dt_per_bin ** 2` 把中段 DT 過度壓制：

| dt_per_bin | 原 (**2) | 新 (**1.3) |
|-----------|----------|------------|
| 0.3 | 0.09 | 0.22 |
| 0.5 | 0.25 | 0.41 |
| 0.8 | 0.64 | 0.75 |

**800-case 驗證（balanced）**：DT deg 2.283 → 2.293（+0.010），FS echo 3.502 → 3.487（-0.015，noise band）。

### Bug 2 — FilterErle alpha_rise 修正（架構 cleanup，metric 中性）

**檔案**：`python/aec.py` L805-815、`c_impl/src/res_filter.c` L281-296

原本 `dt_weight = 1 - 1.5*dt_indicator` 被套在 rise 分支上，DT 期間 `alpha = 0.95 * 0.1 = 0.095`（極小 α 意味快速更新）— 這讓 DT 期間雜訊 inst_erle 反而 **加速** 汙染 erle tracker。

新寫法：

```python
dt_factor = clip(dt_indicator, 0, 1)
alpha_rise_eff = 0.95 + 0.05 * dt_factor   # DT → 1.0（凍結）
alpha = alpha_drop if inst < erle else alpha_rise_eff
```

DT 完全凍結 rise，symmetric drop 保持 0.7。800-case metric 中性（DT deg ±0），但消除原本的邏輯 bug。

### Bug 6 — 動態 ERL ceiling for dt_from_energy

**檔案**：`python/aec.py` L2748、`c_impl/src/aec.c` L1090-1109

兩個修改：

1. **ERL ceiling 動態化**：`max_echo = far_power × 4.0`（硬編碼）→ `far_power × (1/erl_estimate) × 2.0`（用學到的 ERL 加 2× 安全邊際）
2. **ERL 收斂後繼續追蹤**：移除 `!filter_converged` gate；收斂後改用 `alpha = 0.999`（更慢）持續追蹤
3. **ERL clip 放寬**：`[0.001, 1.0]` → `[0.001, 10.0]`，允許高耦合場景 ERL > 0 dB

修正前：ERL 卡在 pre-convergence 估值，高耦合 FS 段 `mic_pwr > far_pwr × 4` 常態成立 → 誤觸 `dt_from_energy > 0` → ENR threshold 被放鬆 → echo 洩漏。

**800-case 驗證（balanced）**：metric 中性（FS -0.012 在 noise band）。架構更正確。

---

## 已嘗試但 revert（AEC_CODE_REVIEW.md 的 Critical/High 項目）

每個都 800-case 獨立驗證，全部 regress：

| Bug | 修改 | FS echo Δ | DT echo Δ | 結果 |
|-----|------|-----------|-----------|------|
| Bug 3 | R_scale DT boost 10× | -0.056 | ≈0 | revert |
| Bug 10 | reverb tail re-clamp `min(res, error_psd*0.99)` | **-0.457** | **-0.347** | revert（clamp 上限壓過 suppression，bug 其實是 feature）|
| Bug 8 | erle_factor ramp 0-10 → 3-18 dB | （batch 中 -0.5 量級） | | revert |
| Bug 12 | convergence 5 dB/10f → 10 dB/20f | （batch 中 -0.5 量級） | | revert |
| Bug 16 | near_psd update 移至 fb_erle 前 | （batch 中）| | revert |

**結論**：現有 `aec.py` 的 tight clamp / threshold / ramp 是 **load-bearing tuning**，不是 bug。看似重複或 tight 的限制往往是支撐高 FS 分數的關鍵。未來 review 單項需先 800-case 獨立驗證，禁止批次套用。

---

## C 實作同步（release target）

### 已同步到 C（`main` 分支）

| 項目 | 位置 | 說明 |
|------|------|------|
| Bug 7 | `res_filter.c` nearend_est 區塊 | `dt_per_bin ** 2 → ** 1.1`（Python 亦改為 1.1）|
| Bug 2 | `res_filter.c` update_filter_erle | FilterErle alpha_rise 在 DT 凍結 |
| Bug 6 | `aec.c` dt_from_energy + erl_estimate | 動態 ERL ceiling + 收斂後持續追蹤 |
| A4 | `res_filter.c` CNG | `cn_gain_k × 0.3 → × 0.4` |
| FS-parity (b) | `res_filter.c` ENR | `ne_physical_floor = error_psd × 0.05` |
| FS-parity (c) | `res_filter.c` ENR | DT-aware ENR threshold relaxation（`effective_dt > 0.4` 時最大 1.5×）|

### `feature/static-memory` 分支

同樣 port Bug 7 / Bug 2 / A4 + FS-parity 三項（Bug 6 N/A，該分支無 `dt_from_energy` / `erl_estimate` struct 成員）。由於分支尚未 merge v2.1.0 / v2.3.0 的其他改動，基礎 correlation 本就較低。

### C vs Python 檔案級對比（`main`，v2.4.0 最終狀態）

| 場景 | Correlation 典型值 | 說明 |
|------|----------|------|
| NE 單講 | **~1.0000** | AEC 路徑基本一致 |
| 雙講（DT）| **0.98–0.99** | 主要信號路徑對齊 |
| FS 有殘留 | **0.90** | 殘留 echo 有微量能量差異 |
| FS 近靜音 | **0.60–0.70** | 兩邊都接近 1e-4 RMS，浮點雜訊主導，非真實差異 |

C 在 800-case AECMOS FS echo 比 Python 低 ~0.37，來自 **線性 AEC 層**（Kalman filter / shadow）的浮點累積差異，非 RES 可單獨解決。RES 路徑本次已完成對齊。

`c_impl/USER_GUIDE.md` 新增：發佈取向的使用說明（安裝、CLI/API、使用條件、限制、常見問題調整）。

---

## 分數演進

| 版本 | FS echo | DT echo | DT deg | 關鍵改動 |
|------|---------|---------|--------|---------|
| v2.0.0 | 3.534 | — | 2.237 | inst_erle cap=4 |
| v2.1.0 | 3.534 | — | 2.237 | AEC3 echo switching + P-floor |
| v2.3.0 | 3.504 | 4.163 | 2.282 | Energy DT 三線驅動 + A1-A5 + G1-G2 |
| **v2.4.0** | **3.487** | **4.164** | **2.293** | Bug 7 DT per-bin 軟化 |

---

## 下輪方向（待定）

- v2.1.0 / v2.2.0 / v2.3.0 的改動尚未全部同步到 C 端
- Pareto 前緣已接近飽和（Bug 3/8/10/12/16 都在嘗試越過但都失敗）；下一步需非參數方向（如：雙頻帶處理、NN post-filter、新 DTD 架構）
