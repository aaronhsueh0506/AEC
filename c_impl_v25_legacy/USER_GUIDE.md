# AEC C — User Guide

回音消除（AEC）模組 C 語言實作使用說明。

---

## 1. 快速開始

### 1.1 編譯

```bash
cd c_impl
make          # CLI 工具：bin/aec_wav
make lib      # 靜態函式庫：bin/libaec.a
```

### 1.2 CLI 使用

```bash
# 基本用法（balanced preset）
./bin/aec_wav <mic.wav> <ref.wav> <output.wav>

# 指定壓制強度
./bin/aec_wav mic.wav ref.wav out.wav --preset mild          # 最保守
./bin/aec_wav mic.wav ref.wav out.wav --preset balanced      # 推薦
./bin/aec_wav mic.wav ref.wav out.wav --preset aggressive
./bin/aec_wav mic.wav ref.wav out.wav --preset maximum       # 最強壓制

# 關閉 RES（輸出純線性 AEC 結果，用於 debug）
./bin/aec_wav mic.wav ref.wav out.wav --no-res

# 自動估計 mic/ref 延遲
./bin/aec_wav mic.wav ref.wav out.wav --enable-delay-est
```

### 1.3 API 使用

```c
#include "aec.h"
#include "aec_types.h"

// 1. 依取樣率建立 config
AecConfig cfg = aec_config_from_preset(AEC_PRESET_BALANCED, 16000);

// 2. 建立 AEC instance
Aec* aec = aec_create(&cfg);

// 3. 逐幀處理（10ms hop，16kHz → 160 samples/frame）
float mic[160], ref[160], out[160];
while (read_audio(mic, ref, 160)) {
    aec_process(aec, mic, ref, out);
    write_audio(out, 160);
}

// 4. 釋放
aec_destroy(aec);
```

---

## 2. 使用條件

### 2.1 輸入規格

| 項目 | 規格 |
|------|------|
| 取樣率 | 8000 / 16000 / 48000 Hz |
| 位元深度 | 16-bit PCM 或 32-bit float |
| 通道數 | 單聲道（mono） |
| 幀長度 / Hop | 20ms / 10ms（自動依取樣率配置） |
| 濾波器長度 | 預設 64ms（可調：512@8k、1024@16k、3072@48k） |

### 2.2 適用場景

- 全雙工免持通話（手機、智慧音箱、會議系統）
- 單 mic + 單 reference signal
- 回音路徑長度 ≤ 64ms（預設）

### 2.3 使用前提

| 前提 | 說明 |
|------|------|
| **Reference 信號正確** | `ref.wav` 必須是送到 speaker 的原始訊號（playback loopback） |
| **延遲對齊** | mic 相對 ref 的延遲需 ≤ filter length。若不確定，啟用 `--enable-delay-est` |
| **Mic 和 Ref 同步** | 兩路取樣率、起點需一致；有 drift 會嚴重影響 ERLE |
| **遠端有足夠活動** | 純靜音或極低能量的 ref 無法驅動 filter 收斂 |

---

## 3. 限制

### 3.1 演算法限制

| 限制 | 說明 |
|------|------|
| **不支援 32 kHz** | 僅支援 8/16/48 kHz，其他需外部 resample |
| **單通道** | 不支援 mic array 或 stereo reference |
| **線性 AEC 為主** | 非線性回音（speaker 飽和、clipping）靠 RES 壓制，無獨立非線性模型 |
| **正延遲限制** | 僅支援 mic 晚於 ref。負延遲（driver bug / SW loopback 異常）需應用層先對齊 |
| **回音路徑長度** | 預設 64ms；若實際路徑較長需調大 `filter_length`（代價：記憶體、運算量）|

### 3.2 特定場景注意

| 場景 | 行為 |
|------|------|
| **Cold start（剛啟動）** | 前 0.5–2 秒 filter 未收斂，echo 壓制較弱；RES 會走保守 render-based 估計 |
| **Echo path 大幅變化** | 設備移動、手遮住 speaker → filter 會 trigger EPC 重新收斂，需 ~200ms |
| **高耦合（small device）** | mic 和 speaker 距離近，echo 能量接近語音；DT 段可能偶見 ~1–2 dB echo leak |
| **穩態遠端（風扇/白噪）** | 純穩態信號會觸發 stationary DT 分支，使用保守 RES；弱語音段可能吸收 |
| **遠端靜音後** | far silence 後殘留 noise gate 約 100–200ms |

### 3.3 資源消耗

| 項目 | 值（fl=512 @16 kHz） |
|------|------|
| 記憶體（filter + shadow） | ~200 KB |
| 記憶體（含 RES + delay est） | ~280 KB |
| 每幀 FFT | 2 × 512-point |
| FFT 函式庫 | kiss_fft（single precision）|

---

## 4. 常見問題與調整

### 4.1 殘留回音太多

**症狀**：遠端講話時仍聽得到回音洩漏。

檢查順序：

1. **確認 ref 信號正確**：`--no-res` 輸出仍有明顯 echo → 線性 AEC 未收斂 → 多半是 ref 錯誤、延遲不對、或取樣率不同步。
2. **延遲對齊**：啟用 `--enable-delay-est`；若仍壓不掉，量測 mic–ref 實際延遲是否 > filter length。
3. **增強 RES**：`--preset aggressive` 或 `maximum`（代價：DT 語音稍受壓）。
4. **延長 filter**：`--filter 1024` 或更大（適用於大型會議室、長 reverb）。

### 4.2 雙方講話（DT）時近端被壓

**症狀**：雙方同時講話時，近端語音聲音變小或斷斷續續。

- **降低 preset**：`--preset mild` 或 `balanced`。
- 不推薦自行修改 RES 內部參數 — 內部壓制/保護邏輯相互耦合，單獨調整容易顧此失彼。

### 4.3 啟動慢、前幾秒 echo 明顯

**說明**：filter 收斂需要 0.5–2 秒的遠端活動訊號。這是正常的 adaptive filter 行為。

- 若需更快收斂：`--preset aggressive`（初期 Q 較大）。
- Warmup 期間建議應用層不要啟用輸出（或播放 "connecting..." 提示音）。

### 4.4 設備移動時回音變大

**說明**：echo path 改變 → EPC 偵測觸發 → filter 重新收斂，約 200 ms 內會有短暫 echo 洩漏。

- 通常自動恢復，無需調整。
- 若頻繁發生（手持裝置），可提高 filter length 增加模型容量。

### 4.5 穩態遠端 + 弱語音被吸收

**說明**：當遠端是穩態訊號（冷氣、風扇、持續白噪）時，linear AEC 可能把微弱近端語音的起手段誤判為 echo 吸收。

- 整體語音保留率 ~48%（接近 NoRES 50%），但個別 100 ms 弱語音段可能掉到 ~10%。
- 若穩態 far-end 是主要場景，考慮應用層加入 VAD + ducking。

### 4.6 想聽線性 AEC 輸出做 debug

```bash
./bin/aec_wav mic.wav ref.wav linear_only.wav --no-res
```

可比較有無 RES 的差異，判斷 echo 來源（filter 未收斂 vs RES 不足）。

---

## 5. Preset 選擇指南

| Preset | FS 回音壓制 | DT 近端保留 | 適用場景 |
|--------|------------|------------|---------|
| **mild** | 中 | 佳 | 高品質通話、音樂場景 |
| **balanced** | 高 | 中 | 一般通話（推薦預設）|
| **aggressive** | 很高 | 中偏弱 | 車用、高噪環境 |
| **maximum** | 極高 | 弱 | 強回音設備、IoT 小喇叭 |

> FS = 遠端單講（Far-end Singletalk）；DT = 雙方同講（Double-talk）。

---

## 6. 聯絡

問題回報或功能需求請開 issue。演算法原理與內部設計見 `docs/` 目錄。
