# AEC 開發紀錄

## 版本歷史

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

#### 已知限制

- 目前僅支援 16kHz 取樣率
- 時域 NLMS 對長回音路徑計算量較大
- 尚未實作 Residual Echo Suppressor (RES)

---

## 待辦事項

### Phase 2: 進階功能

- [ ] 實作 Subband NLMS (頻域)
- [ ] 實作 Residual Echo Suppressor (RES)
- [ ] 加入非線性處理 (NLP)
- [ ] 延遲估計模組

### Phase 3: 優化

- [ ] SIMD 優化 (NEON/SSE)
- [ ] 記憶體優化
- [ ] 即時效能測試

### Phase 4: 整合

- [ ] AEC + LSA 整合範例
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

### DTD (Double-Talk Detection)

**Geigel 方法:**
```
DTD = 1, if |d[n]| > threshold * max(|x[n-k]|), k=0..L-1
```

**能量比方法:**
```
DTD = 1, if E_error / E_echo > threshold
```

當 DTD 偵測到雙講時，停止權重更新以防止濾波器發散。

---

## 參數調校建議

| 參數 | 說明 | 建議範圍 | 預設值 |
|------|------|----------|--------|
| `mu` | 步長 | 0.1 - 0.8 | 0.3 |
| `filter_length` | 濾波器長度 (samples) | 256 - 4096 | 512 |
| `dtd_threshold` | DTD 閾值 (error/echo ratio) | 1.0 - 4.0 | 2.0 |
| `leak` | 權重洩漏 (NLMS only) | 0.9999 - 0.99999 | 0.99999 |

### 調校原則

1. **mu 較大**: 收斂快，但穩態誤差大，對雙講敏感
2. **mu 較小**: 收斂慢，但更穩定
3. **filter_length**: 需大於實際回音路徑長度（NLMS/LMS/SUBBAND 可配置，FREQ 固定=hop_size）
4. **dtd_threshold**: 較低可能誤判為雙講，較高可能漏判

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
