# AEC 修改建議 v4

## ⚠️ 測試前必讀

**所有 RES 相關改動需要 `--enable-res` flag 才會生效。**

建議測試指令：
```bash
# Baseline（確認基準）
python aec.py mic.wav ref.wav output.wav --mode subband --enable-res

# 測試 filter 長度（不改程式碼，直接跑）
python aec.py mic.wav ref.wav output.wav --mode subband --enable-res --filter 2048

# 完整測試（改完程式碼後）
python aec.py mic.wav ref.wav output.wav --mode subband --enable-res --filter 2048
```

---

## 改動總覽

| 項目 | 位置 | 風險 | 預期效果 |
|---|---|---|---|
| A | `ResFilter.process()` gain formula | 低 | 高 ERLE bin 壓制更乾淨 |
| B | `AecConfig` filter_length 預設值 | 無（參數調整） | 線性消除 ERLE 提升 |

---

## A：修正 suppressor gain formula

### 問題說明

現有的 gain formula（第 398 行）：

```python
g = 1.0 / (1.0 + self.over_sub * eer)
```

這個公式在 `eer` 很高時（echo 幾乎佔滿 error，filter 已收斂），g 趨近 `1/(1+over_sub)` 而非 0。以 `over_sub=1.5` 為例，即使 echo 完全主導，g 最小只到 `0.4`（−8 dB），距離 `g_min`（−20 dB）還有 12 dB 的 suppression 沒有發揮。

Wiener filter 形式在高 eer 時更積極壓制，且公式更接近 AEC3 的 suppressor 行為。

### 修改方式

**`ResFilter.process()` 第 398 行：**

```python
# 改前
g = 1.0 / (1.0 + self.over_sub * eer)

# 改後（Wiener filter gain）
g = np.maximum(1.0 - self.over_sub * eer, self.g_min)
```

**同時調整 `over_sub` 預設值。** Wiener 形式的 `over_sub` 語義不同（直接作為 eer 的係數，不是分母的一部分），原本 `1.5` 在新公式下會讓 `eer > 0.67` 的 bin 全部壓到 `g_min`，過於激進。建議改為：

```python
# AecConfig（第 66 行）
res_over_sub: float = 1.2   # 原 1.0（已在 v1 改為 1.5），Wiener 形式下調回 1.2
```

或在 `ResFilter.__init__()` 的 default 參數：

```python
def __init__(self, ..., over_sub: float = 1.2, ...):
```

### 行為對比

| eer 值 | 舊公式 g（over_sub=1.5） | 新公式 g（over_sub=1.2） | 說明 |
|---|---|---|---|
| 0.1 | 0.87 | 0.88 | 低 echo 佔比：行為幾乎相同 |
| 0.5 | 0.57 | 0.40 | 中等：新公式壓制稍強 |
| 0.8 | 0.45 | 0.04（→ g_min） | 高 echo 佔比：新公式大幅壓制 |
| 1.0 | 0.40 | g_min | Echo 主導：新公式壓到底 |

---

## B：增加 partition 數量

### 問題說明

`filter_length` 決定了 PBFDAF 能建模的最大 echo tail 長度。預設 1024 samples @ 16kHz = 64ms，對應 4 個 partition。如果實際 echo tail 超過 64ms（大房間、藍牙喇叭延遲），超出部分的 echo 線性濾波器完全無法消除，只能靠 RES 補救，但 RES 的 suppression 有 `g_min` 下限。

### 修改方式

**不需要改程式碼**，直接用 `--filter` 參數：

```bash
# 2048 samples = 128ms，8 partitions
python aec.py mic.wav ref.wav output.wav --mode subband --enable-res --filter 2048

# 4096 samples = 256ms，16 partitions（針對藍牙或大房間場景）
python aec.py mic.wav ref.wav output.wav --mode subband --enable-res --filter 4096
```

如果測試結果顯示 2048 有改善，可以把 `AecConfig` 和 `main()` 的預設值更新：

```python
# AecConfig（第 38 行）
filter_length: int = 2048   # 原 512

# main()（第 1280 行）
if aec_mode == AecMode.SUBBAND:
    filter_length = 2048    # 原 1024
```

### 注意

`filter_length` 加倍會讓每個 block 的計算量增加，但 PBFDAF 的 FFT size 由 `frame_size`（512）決定，不隨 `filter_length` 增大。增加的只是 partition 數量（更多的 `W[p] * X_buf[p]` 乘法），對 Python 的影響不大，C 實作下幾乎可以忽略。

---

## 驗證方式

**A 的驗證**：在 echo-only 段落（沒有近端說話）對比改前改後的 ERLE，改後應該在 filter 收斂後的高 ERLE 階段有額外 2–5 dB 的提升。如果 ERLE 反而下降，代表 `over_sub` 調太高，可以降到 `1.0` 再試。

**B 的驗證**：先跑 `--filter 1024`（baseline）再跑 `--filter 2048`，如果 ERLE 有明顯提升代表你的測試場景 echo tail > 64ms，2048 有效。如果沒有差異，代表 echo tail 本來就在 64ms 內，不需要增加。

**A + B 一起測**：

```bash
python aec.py mic.wav ref.wav output.wav --mode subband --enable-res --filter 2048
```

兩者改動完全獨立，沒有交互影響，可以分開測也可以一起測。

---

## 測試步驟與結果記錄

請依序執行以下步驟，每步記錄 Mean ERLE、Median ERLE 和 vs AEC3 勝率。

### Step 1：確認 baseline（不改任何程式碼）

```bash
python aec.py mic.wav ref.wav output.wav --mode subband --enable-res
```

| 指標 | 結果 |
|---|---|
| Mean ERLE | ___ dB |
| Median ERLE | ___ dB |
| vs AEC3 勝率 | ___ % |
| 備註 | 預期接近 4.6 dB / 3.5 dB / 24%（commit ab4b361） |

---

### Step 2：測試 B（filter 長度，不改程式碼）

```bash
# 2048
python aec.py mic.wav ref.wav output.wav --mode subband --enable-res --filter 2048
```

| filter_length | Mean ERLE | Median ERLE | vs AEC3 勝率 | 結論 |
|---|---|---|---|---|
| 1024（baseline） | ___ dB | ___ dB | ___ % | — |
| 2048 | ___ dB | ___ dB | ___ % | 採用 / 退化 / 無差異 |
| 4096（若 2048 有改善才測） | ___ dB | ___ dB | ___ % | 採用 / 退化 / 無差異 |

**判斷**：2048 vs 1024 差距 > 0.5 dB → 採用，更新預設值；差距 ≤ 0.5 dB → echo tail 在 64ms 內，B 無效，不更新。

---

### Step 3：測試 A（Wiener gain，需改程式碼）

先改程式碼（`ResFilter.process()` 第 398 行 + `over_sub = 1.2`），再逐步測試不同 `over_sub` 值。

```bash
# 使用 Step 2 決定的最佳 filter_length
python aec.py mic.wav ref.wav output.wav --mode subband --enable-res --filter <best> --res-over-sub 1.2
```

> 注意：若程式碼沒有 `--res-over-sub` CLI 參數，直接改 `AecConfig.res_over_sub` 預設值。

| over_sub | Mean ERLE | Median ERLE | vs AEC3 勝率 | 結論 |
|---|---|---|---|---|
| 1.5（舊公式 baseline，供對比） | ___ dB | ___ dB | ___ % | — |
| 1.0 | ___ dB | ___ dB | ___ % | 採用 / 退化 |
| 1.2 | ___ dB | ___ dB | ___ % | 採用 / 退化 |
| 1.5 | ___ dB | ___ dB | ___ % | 採用 / 退化 |

**判斷**：若所有 over_sub 值都退化 → Wiener 形式不適合現有 EER 估計，還原舊公式；若有改善 → 採用最佳 over_sub。

---

### Step 4：A + B 最佳組合

```bash
python aec.py mic.wav ref.wav output.wav --mode subband --enable-res --filter <B最佳> --res-over-sub <A最佳>
```

| 指標 | 結果 | vs baseline 差距 |
|---|---|---|
| Mean ERLE | ___ dB | +___ dB |
| Median ERLE | ___ dB | +___ dB |
| vs AEC3 勝率 | ___ % | +___ % |

---

### 結果彙整

| 項目 | 建議內容 | 測試結果 | 採用 | 原因 |
|---|---|---|---|---|
| A | Wiener gain（`1 - over_sub * eer`） + over_sub 調整 | Mean ___ dB | ✅ / ❌ | |
| B | filter_length 2048 | Mean ___ dB | ✅ / ❌ | |