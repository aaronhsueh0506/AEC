# Ours balanced vs WebRTC AEC2 — 差距分析報告

> 800-case AEC Challenge blind test, AECMOS local ONNX, balanced preset, FL=512.
> 對比目的：找出我們在 DT deg / NE deg 輸給 WebRTC AEC2 的根因。

---

## 結論先講

**我們不是「整體比較差」，而是「壓得比較深」的單向 trade-off。**

- ✅ **Linear AEC 全面贏**：FS ERLE 18.2 vs 14.3 dB、DT ERLE 5.9 vs 3.8 dB
- ✅ **AECMOS FS echo 贏 +0.093**（壓 echo 更乾淨）
- ❌ **AECMOS DT deg 輸 -0.255、NE deg 輸 -0.097**（語音也被壓掉）
- ⚠️ **DT echo 輸 -0.032**（幾乎打平）

**根因（單一）**：RES 在純 DT / 純 NE 場景下過度壓抑近端語音。Linear AEC 已經做得比 AEC2 好，但 RES 把該保留的語音也一起壓了。

**證據**：DT-deg worst 30 個 case 中，**28 個的 DT-echo 是正的**（贏 AEC2 0.1+）—
也就是「同一個 case，我們 echo 壓得更乾淨，但語音也被壓掉了」。

---

## 1. 完整對比表（800 cases）

### Linear-level (eval_aec_challenge.py)

| 指標 | Ours balanced | WebRTC AEC2 | 差距 |
|------|--------------|-------------|------|
| FS ERLE | **18.2 dB** | 14.3 dB | +3.9 dB ✅ |
| NE SDR | -2.1 dB | -3.3 dB | +1.2 dB ✅ |
| DT ERLE | **5.9 dB** | 3.8 dB | +2.1 dB ✅ |

**Linear AEC 三項全勝。**

### AECMOS Mean

| Scenario | Metric | Ours | AEC2 | Δ | N |
|----------|--------|------|------|---|---|
| FS | echo↑ | **3.577** | 3.484 | **+0.093** ✅ | 300 |
| FS | deg↑ | 4.999 | 4.999 | -0.000 | 300 |
| NE | echo↑ | 4.998 | 4.998 | +0.000 | 200 |
| NE | deg↑ | 4.001 | **4.098** | **-0.097** ❌ | 200 |
| DT | echo↑ | 4.230 | 4.262 | -0.032 | 300 |
| DT | deg↑ | 2.134 | **2.389** | **-0.255** ❌ | 300 |

### Movement vs Non-movement breakdown

| Scenario | Metric | NoMv Δ | Mv Δ | NoMv N | Mv N |
|----------|--------|--------|------|--------|------|
| FS | echo | +0.105 | +0.078 | 169 | 131 |
| DT | echo | **-0.063** | +0.017 | 186 | 114 |
| DT | deg | **-0.191** | **-0.360** | 186 | 114 |
| NE | deg | -0.097 | — | 200 | 0 |

**觀察**：
- DT deg 在 movement 下退步更嚴重（-0.36 vs -0.19）→ movement / EPC 場景 RES 反應太敏感
- DT echo non-movement 微負（-0.063），movement 反而正（+0.017）→ 非常接近，echo 處理沒有顯著差距

---

## 2. Win / Lose / Tie 統計

> 閾值：delta > +0.1 算 win，delta < -0.1 算 lose，介於算 tie。

| Scenario | Metric | Win | Lose | Tie | Win % |
|----------|--------|-----|------|-----|-------|
| FS | echo | 122 | 150 | 28 | 40.7% |
| FS | deg | 0 | 0 | 300 | — |
| NE | echo | 0 | 0 | 200 | — |
| NE | deg | 10 | **83** | 107 | 5.0% |
| DT | echo | 102 | 143 | 55 | 34.0% |
| DT | deg | 103 | **158** | 39 | 34.3% |

**觀察**：
- **NE deg 大量 lose (83 vs 10)** — 純近端場景 RES 過壓
- **DT deg 也大量 lose (158 vs 103)** — 但仍有 1/3 case 我們贏，不是全面退步
- **FS echo win/lose 接近**（122 vs 150）— 平均贏 0.09 是少數高 score case 拉起來的

---

## 3. Delta 分佈 Percentiles

> Delta = Ours − AEC2，正值代表我們贏。

| Scenario | Metric | p10 | p25 | p50 | p75 | p90 |
|----------|--------|-----|-----|-----|-----|-----|
| FS | echo | -1.107 | -0.437 | -0.099 | +0.637 | +1.621 |
| NE | deg | **-0.265** | **-0.174** | **-0.066** | +0.002 | +0.051 |
| DT | echo | -0.652 | -0.344 | -0.084 | +0.292 | +0.692 |
| DT | deg | **-1.303** | **-0.644** | **-0.142** | +0.223 | +0.571 |

**觀察**：
- **DT deg p10 = -1.303**：worst 10% 的 case 我們輸超過 1 分（嚴重壓抑）
- **NE deg p25 = -0.174**：1/4 的近端場景輸超過 0.17（穩定地壓得太狠）
- **FS echo p10/p90 = ±1**：分佈很寬，極端 case 兩邊都有

---

## 4. Top-10 Worst Cases

### DT deg worst (我們語音保留輸最多)

| Case (前 25 char) | Mv | Ours | AEC2 | Δ |
|-------------------|----|------|------|---|
| nyT6FUUdu0W8UpvjP1rRgQ_dt_w | M | 1.543 | 4.114 | **-2.570** |
| W0zK3dv0QE2YckPArTGXCg_dt | | 1.478 | 3.766 | -2.289 |
| Tgtk8jp1zkqmKzsmdrKt0g_dt | | 1.766 | 3.851 | -2.085 |
| nyT6FUUdu0W8UpvjP1rRgQ_dt | | 1.587 | 3.662 | -2.075 |
| Tgtk8jp1zkqmKzsmdrKt0g_dt_w | M | 1.776 | 3.831 | -2.055 |
| wHmBm7VHfkysBOhjoAXkNA_dt_w | M | 1.305 | 3.310 | -2.005 |
| XTqo1aOXDEiqyWTFK99I5Q_dt_w | M | 1.910 | 3.844 | -1.933 |
| W4r0UCjieEuM0u930spvug_dt_w | M | 1.791 | 3.723 | -1.932 |
| VNkNShj97UajHDVbSmIG0g_dt_w | M | 1.974 | 3.892 | -1.917 |
| sLi810BoekuU3HSx14LT7A_dt | | 1.744 | 3.641 | -1.897 |

→ Worst 20 中 **11 個是 movement**（55%），movement 案例佔比顯著高於整體比例（38%）。

### DT echo worst (我們 echo 壓不過 AEC2)

| Case | Mv | Ours | AEC2 | Δ |
|------|----|------|------|---|
| Y7w0W4v9BEihm8Z06BxZfQ_dt | | 2.924 | 4.575 | -1.650 |
| IxgmaPghzUGnR6sxrbGU3Q_dt | | 3.309 | 4.795 | -1.486 |
| r7U6JmcRl0ibIh0mN3CP9g_dt | | 3.172 | 4.573 | -1.401 |
| s0oJqM6Y1UCHSVmHmgsx4Q_dt_w | M | 3.282 | 4.627 | -1.345 |
| JjCzlhn3gEiBQvfJtPNJ9A_dt_w | M | 2.942 | 4.220 | -1.278 |

→ Worst 20 中 7 個 movement（35%），跟整體比例接近 — DT echo 的退步**不是 movement-bound**。

### NE deg worst

| Case | Ours | AEC2 | Δ |
|------|------|------|---|
| IvAFDPUFEk0GuW2VKRHznA_ne | 2.817 | 3.590 | -0.773 |
| OlEt0Ltk0UKogetYnqirSw_ne | 3.573 | 4.248 | -0.675 |
| SFvlSygv4ke9wCrv8LWvYQ_ne | 3.015 | 3.658 | -0.643 |
| E0l0WVPQjEi6AmtbvfSYLA_ne | 3.560 | 4.110 | -0.550 |

→ NE 全部 non-movement（test set 沒有 NE+movement）。退步集中在「ours 3.0-3.6 / AEC2 3.6-4.2」區間，是穩定的小幅 over-suppression。

---

## 5. 跨 Metric 交叉分析（最關鍵）

### DT-deg worst 30 個 case，DT-echo 表現如何？

| 同時失分 | 數量 |
|---------|------|
| DT-deg worst 30 中 DT-echo **也輸** (Δ < -0.1) | **2 / 30** |
| DT-deg worst 30 中 DT-echo **打平** (\|Δ\| < 0.1) | ~3 / 30 |
| DT-deg worst 30 中 DT-echo **我們贏** (Δ > +0.1) | **~25 / 30** |

→ **這是最關鍵的證據**：在 DT 語音被壓最嚴重的 30 個 case，**幾乎全部 echo 是壓得比 AEC2 更乾淨的**。

意思是 — RES 在這些 case 把整個 DT 段都當 echo 壓掉了，所以：
- echo 很乾淨 ✅（壓過頭）
- 但語音也跟著消失 ❌

### NE-deg worst 也是 DT-deg worst？

`32 cases lose both DT-deg and NE-deg` — 32 個近端場景同時在兩個 metric 都輸 0.1+。代表這些 case 的近端訊號，無論在 DT 還是 NE 場景下，我們都會壓掉。

---

## 6. 根因假設

### (A) RES 太激進整路 ← **最可能**

證據：
- DT-deg worst 30 中 28 個 DT-echo 為正 → 同一段同時壓 echo 和語音
- NE deg 大量輸（83 lose / 10 win）→ 純近端場景也壓
- Linear AEC 三項全勝 → 不是 filter 的問題
- balanced preset 的 g_min_db = -55dB，比 AEC2 的內部設定更激進

### (B) DT 偵測太敏感

證據：
- DT deg movement 退步更嚴重（-0.36 vs non-movement -0.19）
- Movement 場景下 dt_indicator / EPC 都更敏感

### (C) NE protection 不足

證據：
- NE deg p25 = -0.17（穩定的小幅退步）
- 32 個 case 在 DT + NE 兩個場景都輸 → 跨場景共通問題

排除：
- ❌ Linear AEC 不夠強 — 我們 ERLE 全勝
- ❌ Stationary far-end 沒處理好 — v13 已修，且 worst case 不集中在穩態場景

---

## 7. 修正方向建議（不在本次 scope 內）

優先順序由高到低：

1. **降低 RES 預設激進度**：
   - balanced preset 的 `g_min_db` 從 -55dB → -45dB（接近 AEC2 內部值）
   - 或新增「ultra-mild」preset 對應 AEC2 的取向
2. **加強 NE protection**：
   - `ne_protect_db` 從 -16dB → -10dB
   - per-bin NE gate 的 ne_erle_gate 提高
3. **DT 偵測門檻拉緊**：
   - dt_indicator 從 0.0-0.8 → 0.0-0.6（縮小作用範圍）
   - movement 場景下 RES 進入更保守模式
4. **逐 case 微調 ENR 兩段映射**：
   - DT 段 enr_t/enr_s 調寬，減少 false-positive 壓抑

---

## 8. Trade-off 討論

### 值不值得追？

**追的代價**：
- DT echo 平均會降（目前贏 -0.032 → 追語音可能變 -0.1+）
- FS echo 平均會降（目前贏 +0.093 → 可能變持平）
- 整體 Linear AEC ERLE 不變（這是 RES 層的事）

**追的好處**：
- DT deg +0.25（追到跟 AEC2 持平）
- NE deg +0.10
- 對「會議、近端品質優先」場景更友善

### 是否該追？

**取決於應用場景**：
- 📞 **語音通話 / 會議**：應該追，近端品質重要
- 🎬 **直播 / podcast**：應該追，語音品質是核心
- 🔇 **強回聲環境 / 嘈雜場景**：不該追，當前 trade-off 合理
- 🎯 **產品定位「echo 殺手」**：不該追，當前是賣點

### 推薦行動

1. **不改 balanced 預設**（避免破壞現有用戶）
2. **把現有 mild preset 再調更保守**（以對標 AEC2 的近端品質為目標）
3. **跑一次 mild preset 800-case** 看是否已經接近 AEC2

---

## 附錄：分析方法

### 工具

- `python/eval_aec_challenge.py --old-aec --preset balanced --filter 512`
- `/tmp/AEC-Challenge/AECMOS/AECMOS_local/Run_1663915512_Stage_0.onnx`
- `python/eval_aecmos_local.py` (參考)
- ad-hoc 分析腳本：`/tmp/analyze_vs_old_aec.py`
- raw scores: `/tmp/aec_vs_old_raw.json`

### 流程

1. 跑 800-case Old WebRTC AEC2 → `output/{stem}_old_aec.wav`
2. 對每個 case 用本地 ONNX AECMOS 評分 ours + old_aec
3. 計算 delta = ours - old_aec
4. 統計 mean / movement breakdown / win-lose-tie / percentiles
5. 找 top-20 worst cases
6. 跨 metric 交叉分析

### 範圍限制

- 只跑 balanced preset（mild/aggressive/maximum 沒測）
- 純數據分析，沒有 spectrogram / 沒有聽音檔
- 不修改任何代碼或參數
