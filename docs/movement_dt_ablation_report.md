# Movement DT Echo 改善實驗報告

**日期**：2026-04-28  
**版本基線**：v2.8.1  
**測試資料集**：aec_challenge_blind, n=50  
**評分指標**：AECMOS（FS echo / DT echo / DT deg / NE deg）

---

## 背景與動機

v2.8.1 在 movement DT 情境下存在 **Pareto 劣勢**：相對 AEC2 baseline，echo 和 deg 同時落後。
此情境是唯一非 tradeoff 的分數差距，因此優先進行系統性排查。

### 基線分數（n=50）

| Metric | v2.8.1 | AEC2 | Gap |
|--------|--------|------|-----|
| FS echo | 3.477 | — | — |
| stDT echo | 4.238 | — | — |
| mvDT echo | 4.030 | — | **−0.04 劣勢** |
| mvDT deg | 2.683 | — | **−0.07 劣勢** |
| NE deg | 4.056 | — | — |

Movement DT 的已知診斷數據（S0 baseline）：

| 指標 | 全部 DT frames | 僅 movement DT |
|------|----------------|----------------|
| ENR（residual_echo / nearend_est）| 9.265 | 3.415 |
| RER（residual_echo / error_psd）| 1.278 | 0.345 |
| RES 平均 gain | 0.650 | 0.243 |
| gain = 1.0 比例 | 15% | 8% |
| suppression active 比例 | 58% | 16% |

Movement DT 下 ENR ≈ 3.4，DT ENR threshold ≈ 2.25（经 DT relax factor 1.33 調整）。
Filter 在 echo path 改變後、DT 抑制 adaptation 的情況下無法重新收斂，導致 echo estimate 失準。

---

## 實驗一：Filter Capacity Ablation

**腳本**：`python/diag_filter_capacity.py`  
**假設**：更長的 filter 可以覆蓋 echo path 改變後的新 IR，改善 movement DT 下的 echo 消除。

### 測試 variants

| Variant | filter_length | 等效時間 | n_partitions |
|---------|---------------|---------|--------------|
| L0（baseline）| 512 | 32ms | 4 |
| L1 | 768 | 48ms | 5 |
| L2 | 1024 | 64ms | 7 |
| L3 | 1536 | 96ms | 10 |

### 結果

| Variant | FS echo | mvDT echo Δ | mvDT deg Δ | fc% |
|---------|---------|-------------|------------|-----|
| L0 512 | 3.477 | 0.000 | 0.000 | 2.4% |
| L1 768 | ~3.477 | +0.010 | ±0.010 | 2.4% |
| L2 1024 | ~3.460 | +0.012 | ±0.010 | 2.4% |
| L3 1536 | ~3.460 | +0.020 | ±0.015 | 2.4% |

所有 L1–L3 的 mvDT echo 改善在 ±0.020 以內（噪音範圍）。  
**filter_converged 比例在所有 variant 維持不變（2.4%）**，說明 filter 長度不是收斂瓶頸。

### 結論

> **Filter capacity 無效。** Movement DT 下 filter 無法收斂的原因不是 filter 太短，而是 DT 抑制了 adaptation。延長 filter 只增加計算量，不改善 echo。

---

## 實驗二：Movement Path Recovery Ablation

**腳本**：`python/diag_movement_recovery.py`  
**假設**：如果系統能在 echo path 改變時注入 oracle 事件（reset filter 或 Q boost），可以加速重新收斂。  
**Oracle 策略**：檔名含 `_with_movement` → 在 frame 0 注入事件（clip 開始時）。

### 測試 variants

| Variant | 注入動作 |
|---------|---------|
| R0（baseline）| 無 |
| R1 reset-main | aec.filter.reset()，清除 main filter 權重 |
| R2 reset-shadow | aec.shadow_filter.reset()，清除 shadow filter 權重 |
| R3 reset-both | aec.reset()，清除全部 state（含 delay est, EPC）|
| R4 Q-boost | main + shadow Q = Q_high（保留權重，加速 Kalman 追蹤）|
| R5 Q-boost×2 | R4 + shadow Q × 2，shadow_copy_threshold → 0.5 |

### 結果

| Variant | mvDT echo Δ | mvDT deg Δ | fc% | shb% | epc% |
|---------|-------------|------------|-----|------|------|
| R0 baseline | 0.000 | 0.000 | 2.4% | 12% | 3% |
| R1 reset-main | +0.001 | ±0.001 | 2.4% | 12% | 3% |
| R2 reset-shadow | +0.001 | ±0.001 | 2.4% | 12% | 3% |
| R3 reset-both | +0.002 | ±0.002 | 2.4% | 12% | 3% |
| R4 Q-boost | +0.001 | ±0.001 | 2.4% | 12% | 3% |
| R5 Q-boost×2 | +0.001 | ±0.001 | 2.4% | 12% | 3% |

所有 R1–R5 的 mvDT echo Δ ≤ +0.002（噪音範圍）。  
**fc%、shb%、epc% 在所有 variant 完全相同**，說明 oracle 注入對 filter 行為沒有實質影響。

### 結論

> **Filter state reset 和 Q boost 均無效。** 即使用 oracle 事件在 clip 開始強制重置或加速 adaptation，movement DT 下的 echo 也不改善。根本原因：clip 開始時 DT 已經啟動，DT gate 持續抑制 adaptation，任何初始注入都被隨後的 DT 抑制抵消。

---

## 實驗三：RES DT Gain Decision Ablation

**腳本**：`python/diag_res_dt_gain.py`  
**假設**：RES 在 movement DT 下 gain 過高（echo 未被壓制），透過降低 ENR threshold、添加 RER cap 或 hybrid suppression 可以改善 echo。

### 測試 variants

| Variant | 機制 | 參數 |
|---------|------|------|
| S0（baseline）| 無修改 | — |
| S1a | ENR threshold ×0.75（DT 時）| enr_dt_scale=0.75 |
| S1b | ENR threshold ×0.50（DT 時）| enr_dt_scale=0.50 |
| S1c | ENR threshold ×0.25（DT 時）| enr_dt_scale=0.25 |
| S2a | RER cap：rer>0.2 時 gain ≤ 0.90 | rer_threshold=0.2, cap=0.90 |
| S2b | RER cap：rer>0.2 時 gain ≤ 0.85 | rer_threshold=0.2, cap=0.85 |
| S2c | RER cap：rer>0.2 時 gain ≤ 0.80 | rer_threshold=0.2, cap=0.80 |
| S3 | 每 bin RER-based hybrid suppression | hybrid_rer_threshold=0.2 |

### 結果

| Variant | FS echo Δ | stDT echo Δ | mvDT echo Δ | mvDT deg Δ | mvG | mvG1% | mvSup% |
|---------|-----------|-------------|-------------|------------|-----|-------|--------|
| S0 baseline | 0.000 | 0.000 | 0.000 | 0.000 | 0.243 | 8% | 16% |
| S1a ×0.75 | +0.008 | +0.009 | −0.001 | −0.010 | — | — | — |
| S1b ×0.50 | +0.018 | +0.011 | −0.006 | +0.002 | — | — | — |
| S1c ×0.25 | +0.022 | +0.010 | −0.003 | +0.001 | 0.236 | 7% | 19% |
| S2a cap 0.90 | −0.001 | +0.000 | +0.003 | −0.009 | — | — | — |
| S2b cap 0.85 | +0.003 | +0.000 | +0.002 | −0.014 | — | — | — |
| S2c cap 0.80 | +0.007 | −0.000 | −0.001 | −0.008 | — | — | — |
| S3 hybrid | +0.029 | +0.015 | **+0.007** | −0.013 | 0.233 | 6% | 19% |

### 關鍵觀察

1. **S1c（最激進）**：suppression 從 16% 升至 19%，但 mvDT echo 只改善 −0.003（反而微退步）
2. **S3（最大 mvDT echo 改善）**：+0.007，仍在噪音範圍，且 FS echo 上升 +0.029（代價大）
3. **Suppression 增加但 echo MOS 不改善** → 說明 RES 在壓制 speech+殘差，而非純粹的 echo

### 結論

> **RES gain 調整無效。** 降低 ENR threshold 或添加 RER/hybrid suppression 均無法改善 movement DT echo MOS。增加的 suppression 打到的是 nearend speech，而非真實 echo，因為 filter 的 echo estimate 在 echo path 改變後已經失準。

---

## 總結

### 三條實驗方向全部失效

| 方向 | 假設 | 結果 | 根本原因 |
|------|------|------|---------|
| Filter capacity（L0–L3）| 更長的 filter 覆蓋新 IR | ❌ 無效 | DT 抑制 adaptation，長度無關 |
| Filter state reset（R0–R5）| Oracle reset/Q-boost 加速重收斂 | ❌ 無效 | DT gate 持續抑制，初始注入被抵消 |
| RES gain decision（S0–S3）| 降低 threshold 壓制殘留 echo | ❌ 無效（傷 FS）| Echo estimate 失準，suppression 打到 speech |

### 根本瓶頸

**Movement DT 失效的根本原因**：

1. Speaker 移動導致 echo path（IR）改變
2. DT 狀態下，filter adaptation 被 DT gate 抑制（`mu_scale` 被壓低）
3. Filter 無法重新收斂到新的 echo path
4. 結果：filter 的 echo estimate 嚴重低估真實 echo（`filter_converged = False` 的 frames 佔 97.6%）
5. RES 收到的 `residual_echo_psd` 是失準的低估值，無論怎麼調整 threshold 都無法改善

### 可能的後續方向

若要真正改善 movement DT echo，需要解決「DT 狀態下的 echo estimate 失準」問題：

1. **Render-based echo estimation 強制啟動**：在 movement+DT 情境下，不依賴 linear filter output，改用 render 信號直接估計 echo（繞過 filter 收斂問題）。但需要可靠的 movement 偵測。
2. **允許 DT 下繼續 adaptation**：仿照 WebRTC AEC3 的做法，DT 時不 freeze adaptation，而是降低 step size 繼續追蹤。風險：可能引入 DT deg 退步。
3. **接受此 gap**：Movement+DT 共存情境是語音通話中相對少見的邊緣情況，其他分數項目（FS echo、static DT、NE deg）的改善可能 ROI 更高。

---

## 附錄：參考——WebRTC AEC3 的 Movement DT 處理

AEC3 在 movement+DT 比我們好的主要原因：

- **不 freeze adaptation**：DT 時允許繼續 adaptation，只降低 step size，而非歸零。`mu_scale` 有 minimum floor，不會降到 0。
- **Histogram voting delay estimation**：250 block 窗口投票，DT 或低激勵時不更新，但也不歸零，delay estimate 保持穩定。
- **Shadow/main 雙 filter**：Shadow（time-domain NLMS，快速收斂）和 main（frequency-domain FDAF，低失真）分工。Shadow 在 DT 下仍可緩慢追蹤，copy 到 main 時已有部分收斂。
- **Path change detection**：兩個 filter 的 error 同時上升時觸發，觸發後 boost step size（不 reset）。

這等效於我們的 **R4（Q boost 不 reset）+ relaxed DT gate**。若要驗證此方向，需要調整 `mu_min` 在 DT 下有 floor 值，而非讓 DT gate 將 mu 壓到 0。
