# AEC 開發紀錄

## 版本歷史

### v1.4.1 (2026-03-17) - RES 改為純 Wiener gain + Shadow-only 支援

#### 改動內容

1. **RES 移除 DTD-aware，改為純 Wiener gain**
   - 移除 `dtd_conf` 參數，`over_sub` 預設 1.5→1.0（標準 Wiener）
   - DT 保護改由 EER 天生特性：DT 時 error_psd↑ → EER↓ → gain↑（自動減少壓制）
   - 與 SpeexDSP Wiener post-filter 一致

2. **Shadow-only 模式**
   - 移除 shadow 對 DTD 的依賴 guard（≈ WebRTC/SpeexDSP 做法）
   - 新增三方架構比較（WebRTC AEC3 / SpeexDSP / 本專案）
   - 新增 DTD×Shadow 四種組合行為表

---

### v1.4.0 (2026-03-17) - RES OLA 重寫 + Shadow 修正

#### 改動內容

1. **RES 改用 OLA + sqrt-Hann 窗**
   - 修正棋盤頻譜（musical noise）和無線電不連續感
   - 根因：舊版無窗函數、attack 太快（0.3）、無跨頻率平滑、無 crossfade
   - 分析窗 × 合成窗 = Hann，50% overlap-add 完美重建
   - 加入 3-bin cross-frequency smoothing 消除孤立 gain 峰谷
   - Attack 放慢 0.3→0.6（time constant 1.4→2.5 frames）

2. **Shadow filter 修正**
   - 加入 50-frame warm-up guard：收斂前不允許 copy
   - 移除 `output = shadow_out`：copy 只複製 weights，不切換 output
   - 修正：舊版在 frame 3, 8 就觸發 copy，把未收斂的 weights 複製到 main 導致退化

4. **ERLE 改用 cumulative average**
   - 舊版 EMA（α=0.95）只看最後幾個 sample → fileid_1 結尾靜音顯示 0 dB
   - 新版 `near_power_sum / error_power_sum` 全段平均

5. **FREQ mu 0.2→0.3**
   - 配合 mode default mu table 統一

#### 驗證結果（FL=1024, subband）

| Config | fileid_0 (DT) | fileid_1 (far only) | fileid_2 (alternating) |
|--------|---------------|---------------------|----------------------|
| DTD only | 7.0 dB | 21.0 dB | 1.4 dB |
| DTD+Shadow | 6.7 dB | 21.0 dB | 1.3 dB |
| DTD+RES | 8.3 dB | 25.1 dB | 1.7 dB |

Shadow 不再退化（差距 <0.3 dB），RES 在 far-only 場景提升 +4 dB。

---

### v1.3.0 (2025-03-13) - DTD 改為 WebRTC-style 發散偵測

#### 改動內容

1. **DTD 從 error-based 改為 WebRTC-style 發散偵測**
   - 移除 error/echo ratio DTD（循環依賴：ratio 依賴濾波器品質）
   - 改用 output vs input 比較：`ratio = max(energy_ratio, peak_ratio)`
   - 同時使用 energy-based + peak-based 偵測（peak 捕捉瞬態尖峰）
   - 所有模式（LMS/NLMS/FREQ/SUBBAND）統一使用發散偵測

2. **二元凍結 → confidence-based mu scaling**
   - `update_weights: bool` → `mu_scale: float`（Python + C 全部改）
   - `mu_scale = 1.0 - confidence × (1.0 - mu_min_ratio)`
   - confidence=0 → mu_scale=1.0, confidence=1 → mu_scale=0.05
   - 避免完全凍結導致濾波器停滯

3. **Output Limiter（安全網）**
   - `if max(|output|) > max(|near|): output *= max(|near|) / max(|output|)`
   - 保證 output 永遠不超過 mic amplitude，即使 DTD 來不及反應

4. **C code 同步更新**
   - `aec_types.h`：DTD 參數改為 `dtd_divergence_factor`, `dtd_mu_min_ratio`, `dtd_confidence_attack/release`
   - `aec.c`：inline 發散偵測 + output limiter + mu_scale
   - `subband_nlms.c/h`、`nlms_filter.c/h`：`update_weights` → `mu_scale`

5. **新增文檔** `docs/aec_methods.md`
   - DT 方法比較：Geigel, NCC, Error/Echo, WebRTC, SpeexDSP, MSC
   - Post-filter 方法比較：RES, Wiener, Coherence, NLP

#### 修正的問題

- fileid_0: error-based DTD 假觸發 → 凍結權重在壞狀態 → output 超過 mic
- fileid_2 @ 4.5s: 靜音→語音轉場 error ratio 飆升 → 錯誤觸發 DTD
- Geigel DTD 在 AEC 中 100% 假觸發（echo gain ≈ 1.0）

#### 驗證結果

- Python: 全 4 模式 × 3 fileid 均通過（output 不超過 mic）
- C: 編譯通過

---

### v1.2.0 (2025-03-12) - DTD 全面改進

#### 改動內容

1. **DTD 擴展至所有模式**
   - 原本只有 FREQ/SUBBAND 有 DTD，NLMS/LMS 依賴 leak 防 double-talk
   - 現在所有四個模式都使用 error-based DTD
   - 原因：leak=0.99999 不足以防止 double-talk 權重漂移

2. **DTD Warmup 分模式設定**
   - NLMS/LMS: 50 幀 (~0.8s) — 時域逐樣本更新收斂快
   - FREQ/SUBBAND: 200 幀 (~3.2s) — 頻域需更多 block 建立功率估計
   - 原因：warmup 太長 → double-talk 開始時 DTD 來不及啟動

3. **DTD Holdover 機制**
   - 當 far-end 停止但 near-end 仍在說話時，保持 DTD active
   - 原因：far-end 停止後 echo_energy 歸零 → 收斂保護條件失敗 → DTD 關閉 →
     ref_buffer 殘留數據被 near-end speech 污染 → 權重在一幀內爆炸
   - 實測：fileid_2 在 9.52s→9.60s 之間，weights 從 peak@200 爆炸到 peak@1018

4. **NLMS max_w_norm 調整**
   - 2.0 → 1.5
   - 原因：max_w_norm=2.0 時 fileid_2 output peak (0.867) > mic peak (0.667)

5. **ERLE 計算修正**
   - `get_erle()` 加 eps 保護，避免 log(0) 和信號結尾靜音期的不穩定

#### 修正的問題

- NLMS fileid_2 estimated IR 完全錯誤（peak 在 filter 末尾而非 tap 200）
- NLMS fileid_2 output 比 mic 還大（信號放大而非消除回音）
- `get_erle()` 在信號靜音期回傳不穩定值

---

### v1.1.0 (2025-03-11) - 四模式調校與 DTD 改進

#### 改動內容

1. **DTD 能量尺度修正**
   - FREQ/SUBBAND 的 echo_energy 原本取自頻域 `echo_spec`，與時域 error_energy 尺度不匹配
   - 改用 `echo_est_time = near_end - output`，統一為時域能量

2. **DTD Warmup 機制**（首次引入，針對 FREQ/SUBBAND）
   - 200 幀 warmup，防止收斂初期 DTD 誤觸發導致 false-lock
   - 收斂保護提升至 10%（echo_energy > 10% near_energy）

3. **FREQ 預設 mu 調整**
   - 0.3 → 0.1（單一 FFT block 的 FDAF，mu=0.3 過大導致信號放大）

4. **NLMS leak 調整**
   - 0.9999 → 0.99999
   - 原因：leak=0.9999 使收斂精度只到 14 dB，weights 只達真值的 39%

5. **Plot 改進**
   - 標題不再被切掉
   - True IR 只對 gen_sim_data 產生的檔案顯示（fileid >= 1）
   - IR 圖 x 軸顯示完整 filter length
   - Filter length 依模式自動選擇（SUBBAND=1024，其他=512）

---

### v1.0.0 (2025-02-11) - 初始版本

#### 實作內容

1. **時域 NLMS 自適應濾波器** (`nlms_filter.c`)
   - 標準化最小均方演算法
   - 循環緩衝區實作
   - 可調整步長 (mu)、正則化 (delta)、權重洩漏 (leak)
   - 支援單樣本和區塊處理

2. **雙講偵測器 (DTD)** (`dtd.c`)
   - Geigel 方法：比較近端與遠端能量
   - 能量比方法：誤差能量與回音估計能量比
   - Hangover 機制防止頻繁切換

3. **主協調器** (`aec.c`)
   - 整合 NLMS 和 DTD
   - 串流架構 (hop-size 處理)
   - ERLE 估計功能

4. **Python 參考實作** (`python/aec.py`)
   - 與 C 版本演算法一致
   - 便於研究和驗證

#### 設計決策

- **時域 NLMS 優先**：先建立基線，後續再加入 Subband NLMS
- **區塊級 DTD**：簡化實作，降低計算量
- **與 LSA v3-2 相容**：相同的 hop_size (160 samples @ 16kHz)

#### 已知限制（v1.0.0 時）

- 目前僅支援 16kHz 取樣率
- 時域 NLMS 對長回音路徑計算量較大
- 尚未實作 Residual Echo Suppressor (RES)
- *(v1.1.0+ 已加入 Subband NLMS; v1.3.0 已實作 RES class 但尚未啟用)*

---

## 待辦事項

### Phase 2: 進階功能

- [x] 實作 Subband NLMS (PBFDAF 頻域)
- [x] DTD 改為 WebRTC-style 發散偵測
- [x] 啟用 RES post-filter（Python OLA + sqrt-Hann + Wiener gain）
- [ ] 加入非線性處理 (NLP)
- [ ] 延遲估計模組

### Phase 3: 優化

- [ ] SIMD 優化 (NEON/SSE)
- [ ] 記憶體優化
- [ ] 即時效能測試

### Phase 4: 整合

- [x] AEC + NR pipeline（`Audio_ALG/pipelines/`）
- [ ] 完整測試套件
- [ ] 效能基準測試

---

## 演算法說明

### NLMS (Normalized Least Mean Squares)

```
輸入: d[n] = s[n] + y[n]  (近端語音 + 回音)
      x[n]                (遠端參考信號)

1. 回音估計: y_hat[n] = w^T * x[n]
2. 誤差信號: e[n] = d[n] - y_hat[n]
3. 正規化步長: mu_eff = mu / (||x[n]||^2 + delta)
4. 權重更新: w = leak * w + mu_eff * e[n] * x[n]

輸出: e[n] (回音消除後的信號)
```

### DTD / 發散偵測 (WebRTC-style)

**發散偵測**（v1.3.0 起採用）：
```
energy_ratio = mean(output²) / mean(near²)
peak_ratio = max(|output|) / max(|near|)
ratio = max(energy_ratio, peak_ratio)

ratio > divergence_factor → confidence 上升
ratio < 1.0             → confidence 下降

mu_scale = 1.0 - confidence × (1.0 - mu_min_ratio)
```

搭配 output limiter 確保 output 永遠不超過 mic amplitude。
詳見 `docs/aec_methods.md`。

---

## 參數調校建議

| 參數 | 說明 | 建議範圍 | 預設值 |
|------|------|----------|--------|
| `mu` | 步長 | 0.1 - 0.8 | 0.3 |
| `filter_length` | 濾波器長度 (samples) | 256 - 4096 | 512 |
| `leak` | 權重洩漏 (NLMS only) | 0.9999 - 0.99999 | 0.99999 |
| `dtd_divergence_factor` | 發散判定倍數 | 1.2 - 2.0 | 1.5 |
| `dtd_mu_min_ratio` | 發散時最低 mu 比例 | 0.01 - 0.1 | 0.05 |
| `dtd_confidence_attack` | Confidence 上升速率 | 0.1 - 0.5 | 0.3 |
| `dtd_confidence_release` | Confidence 下降速率 | 0.01 - 0.1 | 0.05 |

### 調校原則

1. **mu 較大**: 收斂快，但穩態誤差大，對雙講敏感
2. **mu 較小**: 收斂慢，但更穩定
3. **filter_length**: 需大於實際回音路徑長度（NLMS/LMS/SUBBAND 可配置，FREQ 固定=hop_size）
4. **divergence_factor**: 較低 → 更敏感（更快降 mu），較高 → 更寬鬆
5. **mu_min_ratio**: 不建議設為 0（完全凍結會導致濾波器停滯）

---

## 測試記錄

### 合成回音測試

```
測試條件:
- 回音延遲: 100ms
- 回音衰減: -6dB
- 濾波器長度: 250ms
- 步長 (mu): 0.3

預期結果:
- 收斂時間: < 1 秒
- 穩態 ERLE: 15-20 dB
```

### 真實錄音測試

待補充。

---

## 參考文獻

1. Haykin, S. "Adaptive Filter Theory" (4th Edition)
2. Sondhi, M.M. "An adaptive echo canceller" (1967)
3. Benesty, J. et al. "Advances in Network and Acoustic Echo Cancellation"
4. Geigel, R.L. "Automatic gain control method" - DTD 演算法基礎
