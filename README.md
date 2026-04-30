# AEC - Acoustic Echo Cancellation

回音消除模組（v3.7.0），Python + C 兩套實作支援 PBFDKF（頻域卡爾曼濾波器）、Multi-ERLE、Shadow Filter 和 RES（殘餘回音抑制）。

**v3.7.0 vs v3.6.1 BALANCED 800-case AECMOS（2026-04-30，AEC Challenge Interspeech 2021）**：

| Preset / version | FS_st echo↑ | FS_mv echo↑ | NE deg↑ | DT_st echo↑ | DT_st deg↑ | DT_mv echo↑ | DT_mv deg↑ |
|--------|-----------|-----------|--------|-----------|----------|-----------|----------|
| **v3.7.0 (BALANCED)** | **3.880** | **3.957** | 3.991 | **4.264** | 2.220 | **4.163** | **2.212** |
| v3.6.1 (BALANCED) | 3.877 | 3.954 | 3.991 | 4.257 | 2.224 | 4.158 | 2.208 |
| Δ v3.7.0 | **+0.003** | **+0.003** | 0 | **+0.007** | -0.004 | **+0.005** | **+0.004** |
| *WebRTC AEC2 (ref)* | *3.457* | *3.519* | *4.098* | *4.331* | *2.304* | *4.149* | *2.528* |
| *gap vs AEC2 (v3.7.0)* | *+0.423* | *+0.438* | *−0.107* | *−0.067* | *−0.084* | *+0.014* | *−0.316* |

> **v3.7.0 主要改進**（PR-G1）：PBFDKF P-covariance update 用 mu_mean-blended KX，避免 DT 期間 P 跟 W decoupling 造成 Kalman 過自信、DT 結束後 recovery 慢。**首次達成全 5 bucket 同向改善（或持平）**——DT_mv deg 是 v3.6.1 最大 gap (−0.320)，G1 收回 +0.004。詳見 [docs/CHANGELOG.md](docs/CHANGELOG.md)。



**v2.8.1（2026-04-28）— Movement DT 系統性排查 + Code cleanup**：

- Movement DT 三輪 ablation 全部結案（filter capacity / filter state reset / RES gain decision），確認根本瓶頸為：DT gate 抑制 adaptation，filter echo estimate 在 echo path 改變後失準，無法靠 downstream 調整彌補。詳見 [docs/movement_dt_ablation_report.md](docs/movement_dt_ablation_report.md)
- 移除三輪實驗產生的暫時性 ResFilter 參數（`res_enr_dt_scale`、`res_rer_threshold`、`res_rer_gain_cap`、`res_hybrid_rer_threshold`），恢復 v2.8.0 production code
- **分數與 v2.8.0 完全相同**（code cleanup only）

**v2.8.0 主要改進（2026-04-19）— RESv3 Phase B（Linear failure fallback）**：

- **E8a — Linear AEC 失敗場景 aggressive mode**：偵測 `erl_estimate > 1.2` 或 `_using_render_based AND erle_factor < 0.2`，且 `effective_dt < 0.2`（排除 DT），強制 `residual_echo_psd → error_psd × 0.9`，ENR 大幅上升 → gain 壓到 g_min。處理 JteZUZ / 9xjhi 等 linear filter 無法收斂的 FS tail cases。
- **800-case 驗證**：FS echo 3.554 → 3.564 (+0.010)，DT echo/deg/NE 持平 v2.7.0（所有 metric ≥ v2.7.0）。
- **後續計畫（Phase A）**：per-band dominance-based dual-hypothesis gain 取代 ENR gate，推動 FS echo 向 AEC3 3.875 靠攏（目標差距 +0.32）。

**v2.7.0 主要改進（2026-04-19）— 突破 DSP Pareto**：

- **E6 — Min-statistics noise floor**：ResFilter 固定 `g_min=-55dB` 改為**動態 noise-floor gain**。每幀追蹤 quiet-floor（Speex/WebRTC AEC3 風格），以 `noise_floor_gain = sqrt(noise_psd / |spec|²)` 作 per-bin 下限。效果：小聲近端語音只要高於統計 noise floor 就會被保留（解決原先 Tgtk8 等 case 的 40% silence 問題 → 22%）。
- **CNG 預設開啟**：balanced preset 加上 `enable_cng=True`，填補被壓制 bin 的 dead air，AECMOS 不再因「靜音跳動」扣分。
- **800-case 驗證**：FS echo 3.462 → **3.554** (+0.092), DT deg 2.308 → **2.410** (+0.102), DT echo 4.155 → 4.117 (-0.038), NE deg 4.007 → 4.011 (+0.004). **FS 和 DT deg 同時上升**（首次突破 Pareto）。

**v2.6.0 主要改進（2026-04-18）**：

- **P2 — Render-based physical ceiling**：`residual_echo_psd` 加入物理上限 `min(erl_estimate × 2.0, 1.0) × far_psd`，防止 ENR 因 filter echo_spec 高估而觸發過度壓制。
- **A1 — EPC 期間 DT 保護**：EPC active 時 `shadow_dt` 由 0 改為 `0.08 × max(dt_from_energy, dt_from_shadow)`，緩解 EPC 誤觸時的 DT 語音壓制（特別是 movement 場景）。
- **800-case 驗證**：DT deg 2.296 → 2.308（+0.012），FS echo 3.482 → 3.462（-0.020，在容忍範圍）

**v2.5.0 主要改進（2026-04-17）**：

- **RES C/Python 7 項對齊**：dt_per_bin ** 1.1、render-based threshold 公式、fb_erle smoothing input、filter_erle cap 統一 [0.5,200]、ne_physical_floor、render gate 條件、DT ENR relax
- **Kalman float32 parity**：Python 消除 delta/KX float64 升精度，與 C float32 對齊
- **PBFDKF mu_ratio 修正**：無 RES 路徑補上 `_update_simple_mu_ratio`（C/Py 一致）
- **800-case 分數**：DT deg 2.296（+0.014 vs v2.3.0），FS echo 3.482，NE deg 4.007

**v2.4.0 主要改進（2026-04-17）**：

- **Bug 7 — DT per-bin 軟化**：`dt_per_bin ** 2` → `** 1.3`，修正 mid-range DT 過度壓制（dt=0.5 的近端壓制 0.25→0.41，dt=0.3 的近端壓制 0.09→0.22）。800-case 驗證 DT deg +0.010
- **Bug 2 — FilterErle alpha_rise 修正**：DT 期間凍結 rise（`alpha → 1`），移除原本 dt_weight 反向加速 rise 的漏洞。架構更乾淨，metric 中性
- **Bug 6 — 動態 ERL ceiling**：`dt_from_energy` 的 echo ceiling 由硬編碼 4× 改為 `1/erl_estimate × 2`，高耦合場景不再誤觸 ENR 放鬆。ERL 持續追蹤至收斂後（alpha 0.99→0.999）
- **C 同步**：所有上述修正已同步至 `c_impl/src/res_filter.c` 和 `c_impl/src/aec.c`
- **Code Review 失敗紀錄**：Bug 3（R_scale DT boost）、Bug 8（erle_factor ramp）、Bug 10（reverb re-clamp）、Bug 12（convergence threshold）、Bug 16（near_psd order）經 800-case 驗證皆造成 FS regression，已 revert

**v2.3.0 主要改進（2026-04-16）**：

- **Pre-filter Energy DT signal** + **Shadow DTD 互補覆蓋** + **三線驅動 effective_dt**（5 條 DT 保護路徑完整啟動）
- **Blindspot fixes**：S_ee decay 同步、delay reset 帶 ResFilter、動態 ERL 估計、GCC-PHAT PAR gate、EPC render-forced 延長至 200ms
- **Render-based 穩定性** + **Shadow copy 防護** + **Code cleanup**（詳見 [CHANGELOG_v2.3.0.md](docs/CHANGELOG_v2.3.0.md)）

**Blind test 成績（fl=512, AEC Challenge Interspeech 2021, 800 cases, AECMOS）**：

**v2.8.1 balanced preset**：FS echo **3.564** / DT echo 4.117 / DT deg 2.410 / NE deg 4.011（同 v2.8.0）

**v2.8.0 balanced preset**：FS echo **3.564** / DT echo 4.117 / DT deg 2.410 / NE deg 4.011

**v2.7.0 balanced preset**：FS echo 3.554 / DT echo 4.117 / DT deg 2.410 / NE deg 4.011

**v2.6.0 balanced preset**：FS echo 3.462 / DT echo 4.155 / DT deg 2.308 / NE deg 4.007

**v2.5.0 balanced preset**：FS echo 3.482 / DT echo 4.162 / DT deg 2.296 / NE deg 4.007

**歷史總表（v2.3.0）**：

| Method | FS echo↑ | DT echo↑ | DT deg↑ | NE deg↑ |
|--------|----------|----------|---------|---------|
| PBFDKF Linear (balanced) | 2.714 | 3.162 | 3.225 | — |
| Speex (SpeexDSP 1.2) | 2.808 | 3.368 | 3.225 | 4.128 |
| WebRTC AEC2 | 3.484 | 4.262 | 2.389 | 4.098 |
| **mild** | 3.155 | 3.882 | **2.441** | **4.019** |
| **balanced v2.3.0** | 3.502 | 4.163 | 2.283 | 4.008 |
| **balanced v2.4.0** | 3.487 | 4.164 | 2.293 | 4.007 |
| **balanced v2.5.0** | 3.482 | 4.162 | 2.296 | 4.007 |
| **balanced v2.6.0** | 3.462 | 4.155 | 2.308 | 4.007 |
| **balanced v2.7.0** | 3.554 | 4.117 | 2.410 | 4.011 |
| **balanced v2.8.0** | **3.564** | **4.117** | **2.410** | **4.011** |
| **aggressive** | **3.621** | 4.289 | 2.231 | 3.994 |
| **maximum** | **3.722** | 4.412 | 2.162 | 3.961 |
| WebRTC AEC3 | 3.875 | **4.538** | 1.850 | 3.454 |

> v2.4.0 僅重新驗證 balanced preset；其他 preset 未跑完整 800-case（Bug 7/2/6 對 mild/aggressive/maximum 的 RES 參數結構相同，預期 DT deg 亦有類似 +0.01 改善幅度，但未實測驗證）。

**FS echo — No Movement / With Movement 分解**（169 no-mv / 131 mv）：

| Method | no-mv FS echo↑ | mv FS echo↑ | ALL |
|--------|----------------|-------------|-----|
| PBFDKF Linear (balanced) | 2.721 | 2.705 | 2.714 |
| **mild** | 3.134 | 3.181 | 3.155 |
| **balanced** | 3.464 | 3.551 | **3.502** |
| **aggressive** | 3.549 | 3.713 | **3.621** |
| **maximum** | 3.638 | 3.830 | **3.722** |
| WebRTC AEC3 | 3.524 | 3.602 | 3.558 |

**DT — No Movement / With Movement 分解**（186 no-mv / 114 mv）：

| Method | no-mv echo↑ | mv echo↑ | no-mv deg↑ | mv deg↑ | ALL echo | ALL deg |
|--------|------------|---------|------------|---------|---------|---------|
| PBFDKF Linear (balanced) | 3.162 | 3.162 | 3.233 | 3.213 | 3.162 | 3.225 |
| **mild** | 3.908 | 3.839 | 2.437 | 2.447 | 3.882 | **2.441** |
| **balanced** | 4.192 | 4.116 | 2.303 | 2.251 | 4.163 | 2.283 |
| **aggressive** | 4.310 | 4.254 | 2.266 | 2.173 | 4.289 | 2.231 |
| **maximum** | 4.431 | 4.383 | 2.204 | 2.094 | 4.412 | 2.162 |
| WebRTC AEC3 | 4.216 | 4.147 | 2.213 | 2.209 | 4.190 | 2.211 |

> **balanced FS echo 3.502 穩固超越 AEC2**（+0.018）。**mild DT deg 2.441 超越 AEC2**（+0.052）。全 preset DT deg 大幅領先 AEC3（+0.31~+0.59）、NE deg 領先 AEC3（+0.51~+0.57）。Movement cases 在 aggressive/maximum preset 表現更突出（FS echo mv > no-mv），得益於 B3 delay reset + B4 動態 ERL。

> Pipeline（AEC+NR+RES）成績見 [Audio_ALG](https://github.com/aaronhsueh0506/Audio_ALG) README。

## 規格與限制

### 輸入規格

| 項目 | 規格 |
|------|------|
| 取樣率 | 8000 / 16000 / 48000 Hz（自動計算 frame/fft/hop） |
| 位元深度 | 16-bit PCM / 32-bit float |
| 通道數 | 單聲道（mono） |
| 幀長度 / Hop | 20ms / 10ms（依 sample rate 自動配置） |
| 濾波器長度 | 64ms（預設，可調。512@8k, 1024@16k, 3072@48k） |
| 延遲 | 10ms（1 hop，algorithmic delay） |

### 適用場景

- 全雙工免持通話（手機、智慧音箱、會議系統）
- 單通道 AEC（1 mic + 1 ref）
- 回音路徑 ≤ 64ms（預設），可透過 `filter_length` 參數延長

### 使用注意事項

| 項目 | 說明 |
|------|------|
| **收斂時間** | Filter 需要約 0.5-2 秒的遠端信號才能收斂（ERLE > 10dB）。收斂前 RES 使用保守的 direct echo estimate |
| **Warmup 期** | 前 80-100 幀（0.8-1.0 秒）為 warmup 階段，不進行收斂判定，避免誤判 |
| **延遲對齊** | 若 mic-ref 延遲 > filter_length，需啟用 delay estimation（`--enable-delay-est`）或手動指定延遲 |
| **遠端信號要求** | 遠端需要有足夠的能量和頻譜豐富度才能有效收斂。純靜音或極低能量段不會觸發 filter 更新 |
| **Echo path 變化** | 支援 Echo Path Change (EPC) 偵測，自動提高 P_MAX 加速重新收斂。但快速大幅變化（如移動設備）仍會暫時增加殘餘 echo |
| **Preset 選擇** | mild 最保守（DT deg 最佳 2.267），maximum 最激進（FS echo 最佳 3.912）。推薦 balanced 作為通用設定 |

### 已知限制

| 限制 | 說明 |
|------|------|
| **不支援 32kHz** | 僅支援 8/16/48 kHz，32kHz 需外部 resample |
| **單通道** | 不支援多麥克風陣列 |
| **線性 echo 為主** | 非線性 echo（speaker 飽和）靠 RES 壓制，無獨立非線性模型 |
| **高 coupling FS** | 30% 幀仍存在 dt_indicator 偏高（filter 間歇失效），影響 FS echo |
| **高 coupling DT** | inst_erle_fast 修正可能誤壓 DT 語音（echo 與語音能量相近時） |
| **CNG** | 底噪追蹤在遠端持續講話時效果有限（noise_psd 可能偏高） |
| **延遲估計** | GCC-PHAT 在低 SNR 或強非線性場景可能不準確 |
| **延遲方向** | 僅支援正延遲（mic 晚於 ref）。負延遲場景（software loopback、driver latency 異常）需由應用層確保 ref 信號正確對齊 |
| **Stationary far-end + 弱語音段**（v13） | 穩態 far-end（白噪音/風扇/冷氣）+ 近端語音場景下，linear AEC 仍可能在弱訊號段（突發停頓、低能量音節）將語音吸收進 echo path。整體語音保留率可達 ~48%（接近 NoRES 50%），但個別 ~100ms 段會掉到 ~10%。詳見 [aec_improve_v13.md](docs/aec_improve_v13.md) |
| **Stationarity 偵測延遲**（v13） | CV² EMA 需要約 1s 才能穩定觸發 `_is_stationary_far`。Cold start 後前 1s 走一般 RES 路徑 |
| **Stationary DT hangover** | 800ms hangover 在穩態 far-end 場景下會延遲 EPC（echo path change）偵測。實際場景發生機率低（穩態訊號通常 echo path 也穩定） |
| **語音弱起手** | 微弱語音（音節弱起手）的 voice-band spike 可能不觸發 stationary DT hangover（jump_ratio < 1.5） |

### 資源消耗（C 實作，fl=512 @16kHz）

| 項目 | 值 |
|------|-----|
| 記憶體（main + shadow filter） | ~200 KB |
| 記憶體（含 RES + delay est） | ~280 KB |
| 每幀計算量 | ~2 × 512-FFT + Kalman update (257 bins × 4 partitions) |
| FFT 函式庫 | kiss_fft（single precision, MIT license） |

> C vs Python correlation: 0.971。差異來自 kiss_fft vs numpy.fft 演算法精度（max 9.83e-7 per FFT call，~113× float32 epsilon），在 Kalman 正回饋中累積。

支援四級 Preset（MILD / BALANCED / AGGRESSIVE / MAXIMUM）控制 echo 壓制強度。

> C 實作對齊 Python **v2.0.0**（correlation 0.971）+ v2.5.0 RES 修正（7 項對齊 + Kalman float32 + mu_ratio）。**v2.1.0 / v2.2.0 / v2.3.0 的 ResFilter 改動尚未完整同步到 C 端**，計畫在下一輪移植。使用文件見 [c_impl/USER_GUIDE.md](c_impl/USER_GUIDE.md)，移植計畫詳見 [docs/DEVLOG.md](docs/DEVLOG.md)。

## 濾波器模式

| 模式 | CLI | 演算法 | 延遲 | 適用場景 |
|------|-----|--------|------|----------|
| **nlms** | `--mode nlms` | 時域 NLMS | 10ms | 一般用途（預設） |
| **fdaf** | `--mode fdaf` | 頻域 NLMS (單一 block) | 10ms | 中等回音、平衡效能 |
| **pbfdaf** | `--mode pbfdaf` | 分區頻域 PBFDAF (NLMS) | 10ms | 長回音路徑 |
| **pbfdkf** | `--mode pbfdkf` | 分區頻域 PBFDKF (Kalman) | 10ms | 長回音路徑、最佳品質（推薦） |
| **lms** | `--mode lms` | 時域 LMS (固定步長) | 10ms | 穩態環境、極低資源 |

## 快速開始

### Python

```bash
cd python
pip install numpy soundfile

# 推薦：PBFDKF + Shadow + RES（Kalman adaptation，最佳品質）
python3 aec.py mic.wav ref.wav output.wav --mode pbfdkf --enable-res

# 使用 Preset（控制 echo 壓制強度，四級：mild / balanced / aggressive / maximum）
python3 aec.py mic.wav ref.wav output.wav --mode pbfdkf --enable-res --preset mild        # 最保守
python3 aec.py mic.wav ref.wav output.wav --mode pbfdkf --enable-res --preset balanced    # 推薦
python3 aec.py mic.wav ref.wav output.wav --mode pbfdkf --enable-res --preset aggressive
python3 aec.py mic.wav ref.wav output.wav --mode pbfdkf --enable-res --preset maximum

# PBFDAF 模式（NLMS adaptation，不使用 Kalman）
python3 aec.py mic.wav ref.wav output.wav --mode pbfdaf --enable-res

# 啟用 per-frame 診斷輸出（ERLE, mu, gain, convergence 等）
python3 aec.py mic.wav ref.wav output.wav --mode pbfdkf --enable-res --preset balanced --diag

# 啟用 DTD（進階，特定場景）
python3 aec.py mic.wav ref.wav output.wav --mode pbfdkf --enable-res --enable-dtd

# FDAF 模式（單一 FFT block，中等回音）
python3 aec.py mic.wav ref.wav output.wav --mode fdaf --enable-res

# 純 PBFDAF（debug 用，關閉 Shadow）
python3 aec.py mic.wav ref.wav output.wav --mode pbfdaf --enable-res --no-shadow

# 時域 NLMS 模式
python3 aec.py mic.wav ref.wav output.wav

# 調整參數
python3 aec.py mic.wav ref.wav output.wav --mu 0.5 --filter 1024
```

### C (PBFDKF + Shadow + WOLA RES)

```bash
cd c_impl && make
./bin/aec_wav mic.wav ref.wav output.wav                           # BALANCED preset（預設）
./bin/aec_wav mic.wav ref.wav output.wav --preset aggressive       # AGGRESSIVE preset
./bin/aec_wav mic.wav ref.wav output.wav --preset maximum          # MAXIMUM preset
./bin/aec_wav mic.wav ref.wav output.wav --no-res                  # 關閉 RES
./bin/aec_wav mic.wav ref.wav output.wav --no-hpf                  # 關閉 HPF
./bin/aec_wav mic.wav ref.wav output.wav --filter 2400             # 自訂濾波器長度
```

C 版本已對齊 Python v2.0.0（PBFDKF 模式），包含：
- PBFDKF Kalman（correct K, P_MAX=0.5, global denominator）
- Shadow filter + bidirectional copy
- Multi-ERLE（FilterErleEstimator + FullbandErleEstimator）
- dt_indicator ERLE correction（3-frame EMA smoothed inst_erle）
- ENR 重構（dt × nearend_est）
- WOLA RES v2（完整重寫，對齊 Python）
- CNG freeze-gated noise tracking（×0.3 attenuation）
- GCC-PHAT Delay Estimation（inline, 250ms max）
- HPF + 四級 Preset

C vs Python correlation: **0.971**（frame 1-10 完美一致 corr=1.0，剩餘差異為 kiss_fft vs numpy.fft 演算法精度）。

不含：DTD、Saturation Detection、LMS/NLMS/FDAF 模式。

## 系統架構

```
Reference Signal (far-end)        Microphone Signal (near-end)
          |                                  |
          v                                  v
+---------------------+           +---------------------+
| High-Pass Filter    |           | High-Pass Filter    |
| (2nd-order Butter.) |           | (2nd-order Butter.) |
+---------------------+           +---------------------+
          |                                  |
          v                                  |
+---------------------+                     |
| Saturation Detect   |                     |
| + Soft-Clip Ref     |                     |
+---------------------+                     |
          |                                  |
          v                                  |
+---------------------+                     |
| Delay Estimation    |                     |
| (GCC-PHAT / fixed)  |                     |
+---------------------+                     |
          |                                  |
          v                                  |
+---------------------+                     |
| Reference Alignment |                     |
| (ring buffer delay) |                     |
+---------------------+                     |
          |                                  |
          +----------------------------------+
                         |
                         v
              +---------------------------+
              | PBFDKF (Main Filter)      |
              | Kalman adaptation         |
              |  Two-stage Q control      |
              +---------------------------+
                         |
              +----------+----------+
              |                     |
              v                     v
   +------------------+  +------------------------+
   | Shadow Filter    |  | DTD (可選)             |
   | (預設開啟)       |  |  Divergence + Coherence|
   | Q = main Q × 3.0 |  +------------------------+
   +------------------+
              |
      copy gate: far_active + not_dt
      hysteresis: 5 consecutive frames
              |
              v
         +---------------------------+
         | RES Post-Filter (v2)      |
         | (ENR masking / Wiener,    |
         |  direct echo est + ERLE,  |
         |  reverb tail, OLA)        |
         +---------------------------+
                         |
                         v
         +---------------------------+
         | Output Limiter            |
         | (smoothed gain clamp)     |
         +---------------------------+
                         |
                         v
                   Clean Output
```

### 各模組功能概覽

| 模組 | 功能 | 詳細說明 |
|------|------|----------|
| **High-Pass Filter** | 移除 DC 偏移、50/60Hz 電源雜訊、低頻隆隆聲 | 2 階 Butterworth IIR，截止 80Hz |
| **Saturation Detect** | 偵測 speaker 飽和失真，soft-clip reference | 避免非線性 echo 破壞濾波器建模 |
| **Delay Estimation** | 自動估計 mic-ref 延遲 | GCC-PHAT，週期性重估（每 2s） |
| **Reference Alignment** | Ring buffer 延遲補償 | 讓自適應濾波器專注學習 echo path，不浪費 capacity 建模 delay |
| **PBFDAF** | 分區頻域自適應濾波器（NLMS base） | 多 partition 頻域 echo 估計與消除 |
| **PBFDKF** | 頻域卡爾曼濾波器（繼承 PBFDAF） | Per-bin Kalman gain 取代 NLMS 固定步長，two-stage Q 控制 |
| **Shadow Filter** | 背景濾波器（dual-filter 架構） | 更積極的 Q 設定，copy gate 自動修正 main filter |
| **DTD** | 雙講偵測（可選） | Divergence + Coherence 偵測器，連續 mu scaling |
| **RES** | 殘餘回音抑制 | ENR masking（預設）/ Wiener / spectral sub，direct echo est + reverb tail，OLA + sqrt-Hann |
| **Output Limiter** | 安全網 | 保證 output 永不超過 mic amplitude |

## API 使用

```python
from aec import AEC, AecConfig, AecMode, AecPreset

# 推薦：使用 Preset（自動配置 RES + adaptive filter 參數，FDKF 預設開啟）
config = AecConfig.from_preset(AecPreset.BALANCED,
                               sample_rate=16000, mode=AecMode.PBFDKF,
                               enable_res=True)
aec = AEC(config)

# 更強 echo 壓制（犧牲部分近端品質）
config = AecConfig.from_preset(AecPreset.AGGRESSIVE,
                               sample_rate=16000, mode=AecMode.PBFDKF,
                               enable_res=True)
aec = AEC(config)

# 手動配置（無 preset，FDKF 預設開啟）
config = AecConfig(
    mode=AecMode.PBFDKF,
    enable_res=True,
)
aec = AEC(config)

# 標準配置：PBFDKF + Shadow（無 RES）
config = AecConfig(mode=AecMode.PBFDKF)
aec = AEC(config)

# 時域 NLMS 模式（無 Shadow/DTD）
config = AecConfig(mode=AecMode.NLMS, mu=0.3)
aec = AEC(config)

# 處理
hop_size = aec.hop_size  # 160 samples (10ms @ 16kHz)
while has_audio:
    output = aec.process(mic_block, ref_block)
    erle = aec.get_erle()
    conf = aec.get_dtd_confidence()
```

## 參數說明

### 濾波器參數

| 參數 | 預設值 | 範圍 | 說明 |
|------|--------|------|------|
| `mu` | 0.5 (PBFDKF) / 0.3 (NLMS) | 0.1-0.8 | 步長（PBFDKF 模式下由 Kalman gain 取代） |
| `filter_length` | 512 (NLMS/LMS/FDAF) / 1024 (PBFDAF/PBFDKF) | 256-4096 | 濾波器長度 (samples)。CLI 依 mode + sample_rate 自動設定：PB=sr×64ms, 非PB=sr×32ms |
| `enable_dtd` | False | - | DTD：僅頻域模式（Divergence + Coherence），詳見 [docs/dtd_design.md](docs/dtd_design.md) |
| `enable_res` | False (Python) / True (C) | - | 殘餘回音抑制 |
| `use_kalman` | True | - | FDKF adaptation（per-bin Kalman gain，搭配 two-stage Q），預設開啟 |
| `enable_delay_est` | True | - | 自動延遲估計（GCC-PHAT） |
| `max_delay_ms` | 250.0 | 50-500 | 延遲搜尋範圍上限 (ms) |
| `fixed_delay_samples` | -1 | -1 or ≥0 | 若 ≥0，使用固定延遲（跳過估計） |
| `enable_highpass` | True | - | 高通濾波器（移除 DC + 低頻雜訊） |
| `highpass_cutoff_hz` | 80.0 | 20-200 | 高通截止頻率 (Hz) |
| `enable_saturation_detect` | True | - | 飽和偵測（非線性 echo 處理） |
| `enable_shadow` | True | - | Shadow filter（僅頻域模式，預設開啟） |

### AecPreset 系統

四級 preset 控制 **RES 後處理** 和 **自適應濾波器** 的積極程度，提供 echo 壓制 vs 近端品質的 trade-off：

| Preset | Echo 壓制 | 近端品質 | 適用場景 |
|--------|-----------|----------|----------|
| **MILD** | 輕 | 最佳保留 | 會議通話、近端品質優先 |
| **BALANCED** | 適中 | 良好 | 一般用途（推薦預設） |
| **AGGRESSIVE** | 強 | 輕微損傷 | 免持電話、車用 |
| **MAXIMUM** | 最強 | 明顯損傷 | 揚聲器對話、高回聲環境 |

**Preset 控制的參數**：

| 參數 | MILD | BALANCED | AGGRESSIVE | MAXIMUM | 說明 |
|------|------|----------|------------|---------|------|
| **RES v2 層** | | | | | |
| `res_enr_scale` | 1.0 | 0.85 | 0.7 | 0.5 | ENR 門檻縮放（越低越激進） |
| `res_reverb_decay` | 0.6 | 0.65 | 0.7 | 0.8 | Reverb 衰減率 |
| `res_reverb_gain` | 0.8 | 1.4 | 2.0 | 3.0 | Reverb 貢獻 |
| **RES 層** | | | | | |
| `res_g_min_db` | -35 | -45 | -60 | -72 | RES 最小增益 |
| `res_over_sub_base` | 2.5 | 4.0 | 6.0 | 10.0 | 基礎 over-subtraction |
| `res_over_sub_scale` | 4.0 | 7.0 | 10.0 | 15.0 | ERLE-scaled over-sub |
| `res_dt_reduction` | 3.5 | 2.0 | 1.0 | 0.0 | DT 時放鬆壓制量 |
| `res_ne_protect_db` | -12 | -16 | -20 | -35 | Per-bin 近端保護上限 |
| `res_spectral_floor_db` | -25 | -32 | -40 | -50 | 頻譜底噪估計 |
| `shadow_q_ratio` | 3.0 | 3.5 | 4.0 | 5.0 | Shadow Q 倍率 |
| **Adaptive 層** | | | | | |
| `shadow_mu_min` | 0.5 | 0.6 | 0.7 | 0.9 | DT 時 mu floor |
| `warmup_frames` | 150 | 150 | 150 | 200 | 強制高 mu 的 warmup 幀數 |
| `kalman_q_high` | 2e-4 | 1.5e-4 | 1e-4 | 1e-4 | PBFDKF Q_high |

```python
from aec import AecConfig, AecPreset, AecMode

# Preset + 自訂覆蓋
config = AecConfig.from_preset(
    AecPreset.AGGRESSIVE,
    sample_rate=16000,
    mode=AecMode.PBFDKF,
    enable_res=True,
    filter_length=2048,   # 覆蓋 preset 以外的參數
)
```

### PBFDKF 參數 (Partitioned Block Frequency Domain Kalman Filter)

PBFDKF（繼承 PBFDAF）使用 per-bin Kalman gain 取代 NLMS 的固定步長 mu，自動為每個頻率 bin 選擇最佳 adaptation rate。
搭配 two-stage Q（process noise）控制：Q_high 快速收斂，Q_low 穩態追蹤。

**核心公式**：
```
K[k] = P[k] × |X[k]|² / (|X[k]|² × P[k] + R[k])   # Per-bin Kalman gain
W[k] += K[k] × E[k] × conj(X[k])                     # Weight update
P[k] = (1 - K[k] × X[k]) × P[k] + Q[k]               # Covariance update
R[k] = α × R[k] + (1-α) × |E[k]|²                     # Measurement noise (from error)
```

**Two-stage Q 運作機制**：

| 階段 | Q 值 | K (Kalman gain) | 行為 |
|------|------|-----------------|------|
| 收斂前 (Q_high) | `kalman_q_high`（預設 1e-4） | 0.3~0.7 | 積極更新，打破 R deadlock |
| 收斂後 (Q_low) | `kalman_q_low`（預設 1e-7） | 0.01~0.05 | 穩態追蹤，不干擾已收斂權重 |
| 切換條件 | — | — | 連續 6 幀 single-talk ERLE > 4dB |

**R self-reinforcing deadlock（Q_low 固定時的問題）**：
```
高 error → 高 R → 低 K → 慢更新 → 仍然高 error（惡性循環）
```
Q_high = 1e-4 解決此問題：高 Q 持續注入 P → 即使 R 高，K 仍維持合理值。搭配 P_MAX=0.02 和 per-bin Q gating 防止 weight jitter。

| 參數 | 預設值 | 說明 |
|------|--------|------|
| `use_kalman` | True | True=PBFDKF（Kalman），False=PBFDAF（NLMS） |
| `kalman_q_high` | 1e-4 | 快速收斂 process noise（所有 Preset 統一） |
| `kalman_q_low` | 1e-7 | 穩態追蹤 process noise |
| `warmup_frames` | 100 | 強制高 mu 的 warmup 幀數（Preset 可調） |
| P_init | 0.01 | Per-partition per-bin 初始 error covariance |
| P_MAX | 0.02 | P clamp 上限（EPC 期間臨時放寬至 0.1, 30 frames） |
| R_init | 1e-2 | 初始 measurement noise |
| α_R | 0.95 | R 估計 EMA 平滑係數 |

### Shadow Filter 參數 (僅頻域模式，預設開啟)

Shadow filter（雙濾波器）使用更積極參數的背景濾波器持續追蹤回音路徑，當 shadow 表現更好時透過 copy gate 自動將權重複製到 main filter。
這是 WebRTC AEC3 和 SpeexDSP 的核心機制，v1.6.0 起為預設保護策略。

FDKF 模式下，shadow 使用 `shadow_q_ratio` 倍的 Q 值，確保 shadow 收斂更快，copy gate 能有效觸發。

> Shadow 預設單獨使用（≈ WebRTC/SpeexDSP 做法），可加 `--enable-dtd` 啟用雙重保護。
> 詳見 [docs/aec_methods.md §6.4](docs/aec_methods.md) 的組合行為表。

| 參數 | 預設值 | 範圍 | 說明 |
|------|--------|------|------|
| `shadow_copy_threshold` | 0.7 | 0.3-1.0 | Shadow error < main error × threshold 時複製權重 |
| `shadow_err_alpha` | 0.85 | 0.8-0.99 | Error energy EMA 平滑係數 |
| `shadow_mu_min` | 0.5 | 0.1-0.8 | 主濾波器 DT 時 mu 下限（shadow-only 模式的輕量保護） |
| `shadow_copy_hysteresis` | 5 | 1-10 | 連續 N frames 符合條件才觸發 copy |
| `shadow_q_ratio` | 3.0 | 1.0-5.0 | Shadow Q = main Q × ratio（FDKF 模式） |

**Shadow filter 在 FDKF 模式下的 Q 設定**：

| 階段 | main Q | shadow Q | 效果 |
|------|--------|----------|------|
| 收斂前 (Q_high) | `kalman_q_high` (1e-4) | Q_high × `shadow_q_ratio` | Shadow 更快收斂，copy gate 更早觸發 |
| 收斂後 (Q_low) | `kalman_q_low` (1e-7) | Q_low × `shadow_q_ratio` | Shadow 仍比 main 積極，echo path 變化時更快偵測 |
| Echo path 變化 | 慢速恢復 | ratio × 快速恢復 | Shadow copy 觸發，main 快速跟上 |

**設計要點**：
- Shadow 使用 full mu（1.0），不受 DT 影響，持續追蹤回音路徑
- Copy gate 條件：`far_active AND not_dt AND NOT epc_active`
- Copy 門檻：`shadow_err < main_err × 0.7`（連續 5 frames）
- Copy 只複製 weights + P（FDKF 模式），不切換 output（避免不連續）
- 50-frame warm-up guard：收斂前不允許 copy（避免未收斂時的誤判退化）
- 雙向 copy：main 比 shadow 好時也會反向 copy（main → shadow）
- 可與 DTD 互補：DTD 降低 mu 防止惡化，shadow 在背景持續追蹤並自動修正

### RES 參數 (僅頻域模式)

RES v2（Residual Echo Suppressor）使用 OLA + sqrt-Hann 窗框架，支援三種增益公式：

**增益公式（`res_gain_type`）**：
- **`"enr"`（預設）**: ENR masking（仿 AEC3 `GainToNoAudibleEcho`），不需要 nearend PSD 估計，FS 自然完全抑制
- `"wiener"`: Wiener gain `G = nearend / (nearend + β × echo)`，DT 保護好但 FS 可能 gain pumping
- `"spectral_sub"`: Legacy spectral subtraction `G = 1 - over_sub × EER`

**Echo 估計（`res_echo_method`）**：
- **`"direct"`（預設）**: 使用自適應濾波器 echo spectrum + per-bin ERLE tracking，快速追蹤
- `"coherence"`: Legacy coherence-based EER

**附加功能**：
- **Reverb tail**: 指數衰減模型捕捉晚期殘響（`res_enable_reverb`）
- **4-block 近端平均**: 平滑近端 PSD，減少幀間增益抖動
- **頻率域後處理**: DC 一致性 + HF cap（仿 AEC3 `PostprocessGains`）
- **Per-bin near-end gate**: coherence² 做 per-bin 近端偵測（v1.14.0+）

| 參數 | 預設值 | 範圍 | 說明 |
|------|--------|------|------|
| `res_gain_type` | `"enr"` | enr/wiener/spectral_sub | 增益公式（Preset 預設 enr） |
| `res_echo_method` | `"direct"` | direct/coherence | Echo 估計方法（Preset 預設 direct） |
| `res_enr_scale` | 1.0 | 0.1-2.0 | ENR 門檻縮放（<1 更激進，1.0=AEC3 默認） |
| `res_enable_reverb` | True | - | Reverb tail model（Preset 預設開啟） |
| `res_reverb_decay` | 0.5 | 0.3-0.9 | Reverb 指數衰減率 |
| `res_reverb_gain` | 0.5 | 0.0-2.0 | Reverb 貢獻縮放 |
| `res_g_min_db` | -35 | -72 ~ -10 | 最小增益（echo-only bin 的壓制下限） |
| `res_ne_protect_db` | -12 | -35 ~ -6 | Per-bin 近端保護上限（越低保護越少，Preset 可調） |
| `res_over_sub` | 3.0 | 1.0-15.0 | 過減因子（wiener/spectral_sub 用，ENR 不使用） |
| `res_alpha` | 0.8 | 0.5-0.95 | 增益平滑係數 |

### 模式選擇指南

| 應用場景 | 推薦模式 | 推薦配置 |
|----------|----------|----------|
| 最佳品質（會議系統、智慧音箱） | pbfdkf | `--enable-res`（Kalman + Shadow + RES） |
| 一般用途（手機/耳機） | nlms | 預設（無 Shadow/DTD） |
| 中等回音 | fdaf | `--enable-res` |
| 極低資源（MCU） | lms | `--mu 0.01` |

### 繪圖工具

```bash
# 繪製 AEC 結果（預設 Shadow 開啟）
python3 plot_aec_results.py ../wav/ --mode pbfdkf

# 開啟 RES
python3 plot_aec_results.py ../wav/ --mode pbfdkf --enable-res

# 加入 DTD（雙重保護）
python3 plot_aec_results.py ../wav/ --mode pbfdkf --enable-dtd
```

## Benchmark 比較

測試條件：PBFDKF + Shadow + RES v2 + delay pre-alignment + HPF + saturation detect。
Frame/hop: 20ms/10ms（自動依 sample rate 配置），filter_length: 100ms。以下測試使用 16kHz。
評估指標：AECMOS（echo_mos↑ = echo 少, deg_mos↑ = 語音損傷少）。

### AEC Challenge Interspeech 2021 Blind Test（800 cases, AECMOS local ONNX Run_1663915512_Stage_0）

#### Overall

| Method | FS echo↑ | DT echo↑ | DT deg↑ | NE deg↑ |
|--------|----------|----------|---------|---------|
| Linear (NoRES, balanced) | 2.71 | 3.17 | 3.20 | 4.07 |
| Speex (SpeexDSP v1.2) | 2.808 | 3.368 | 3.225 | 4.128 |
| WebRTC AEC2 | 3.484 | 4.262 | 2.389 | 4.098 |
| **Ours mild** | 3.236 | 3.923 | **2.338** | 4.005 |
| **Ours balanced** | 3.577 | 4.230 | 2.134 | 4.001 |
| **Ours aggressive** | 3.687 | 4.357 | 2.070 | 3.995 |
| **Ours maximum** | **3.783** | 4.473 | 2.014 | 3.970 |
| WebRTC AEC3 | 3.875 | **4.538** | 1.850 | 3.454 |

#### No Movement

| Method | FS echo↑ | DT echo↑ | DT deg↑ |
|--------|----------|----------|---------|
| Speex | 2.847 | 3.427 | 3.179 |
| WebRTC AEC2 | 3.457 | 4.331 | 2.304 |
| **Ours mild** | 3.214 | 3.954 | 2.310 |
| **Ours balanced** | 3.562 | 4.269 | 2.112 |
| **Ours aggressive** | 3.651 | 4.384 | 2.045 |
| **Ours maximum** | **3.732** | 4.496 | 1.992 |
| WebRTC AEC3 | 3.871 | **4.566** | 1.858 |

#### With Movement

| Method | FS echo↑ | DT echo↑ | DT deg↑ |
|--------|----------|----------|---------|
| Speex | 2.757 | 3.272 | 3.301 |
| WebRTC AEC2 | 3.519 | 4.149 | 2.528 |
| **Ours mild** | 3.264 | 3.873 | 2.383 |
| **Ours balanced** | 3.597 | 4.166 | 2.168 |
| **Ours aggressive** | 3.733 | 4.313 | 2.109 |
| **Ours maximum** | **3.848** | 4.435 | 2.050 |
| WebRTC AEC3 | 3.882 | **4.492** | 1.836 |

> - **NE deg 全 preset 大幅領先 AEC3**（~4.0 vs 3.454, +0.5）— 近端語音幾乎無損
> - **DT deg 全 preset 優於 AEC3**（2.0+ vs 1.850）
> - **v13 Stationary DT 保護**：白噪音/風扇/穩態 far-end + 近端語音場景，語音保留從 5% → 48%（合成測試），語音場景分數零退步
> - 版本：SpeexDSP 1.2rc3；WebRTC AEC2 / AEC3：M72

### 工具

```bash
# 執行 AEC Challenge benchmark（需 speexdsp Python binding + WebRTC AEC3 CLI）
cd AEC
python3 python/eval_aec_challenge.py wav/aec_challenge/ --aec3 --speex

# 指定 preset
python3 python/eval_aec_challenge.py wav/aec_challenge/ --preset aggressive

# 比較全部 preset
python3 python/eval_aec_challenge.py wav/aec_challenge/ --all-presets

# AECMOS 評估（需 speechmos + onnxruntime≤1.16.3 + numpy<2，用 venv）
source .venv/bin/activate
python3 python/eval_aecmos.py wav/aec_challenge/

# AECMOS 比較全部 preset
python3 python/eval_aecmos.py wav/aec_challenge/ --all-presets
```

## 效能指標

### 演算法收斂與抑制能力

| 模式 | ERLE | 複雜度 | Cold start 收斂 |
|------|------|--------|----------------|
| lms | 10-15 dB | O(N) | 1-5s |
| nlms | 15-20 dB | O(N) | 0.5-2s |
| fdaf | 18-22 dB | O(N log N) | 0.3-1s |
| pbfdaf | 20-25 dB | O(N log N) | 0.2-0.8s |
| pbfdkf | 25-30 dB | O(N log N) | 0.1-0.5s |
| + RES | +2-4 dB | O(K) | - |

### 收斂時間細節（PBFDKF + Shadow，FL=512 @16kHz）

| 階段 | 時間 | 條件 |
|------|------|------|
| Warmup（不更新收斂判定） | 80 frames（800ms） | Cold start 後固定 |
| Cold start → ERLE > 10 dB | 0.3-0.5s | 寬頻 far-end + 含 echo |
| Cold start → ERLE > 20 dB | 0.5-1.0s | 寬頻 far-end + 含 echo |
| Cold start → ERLE > 30 dB | 1.5-3.0s | 寬頻 far-end + 含 echo |
| EPC 後 re-convergence | 0.3-0.8s | P_MAX 暫時放寬 30 frames |
| Stationary far-end CV² 收斂 | ~1.0s | α=0.99 EMA TC≈1s |
| Stationary DT hangover | 800ms | jump_ratio > 1.5 後維持 |

### 系統延遲與資源消耗

| 項目 | 值 |
|------|-----|
| Algorithmic delay | 10ms (1 hop) |
| Frame size / Hop | 20ms / 10ms（@16kHz） |
| FFT size | 512（@16kHz, 自動依 SR 配置） |
| 記憶體（C, main+shadow filter） | ~200 KB |
| 記憶體（C, 含 RES + delay est） | ~280 KB |
| 每 frame 計算量 | 2 × 512-FFT + Kalman update（257 bins × 4 partitions） |

### 各 preset 在 8s clean speech 的最終 ERLE（PBFDKF + RES）

| Preset | 平均 ERLE | FS echo↑ (AECMOS) | DT echo↑ | DT deg↑ | 推薦場景 |
|--------|-----------|-------------------|---------|---------|---------|
| **mild** | 16-18 dB | 3.236 | 3.923 | 2.338 | 會議、近端品質優先 |
| **balanced** | 18-22 dB | 3.577 | 4.230 | 2.134 | **預設**、通用場景 |
| **aggressive** | 19-23 dB | 3.687 | 4.357 | 2.070 | 高 echo 環境 |
| **maximum** | 20-25 dB | 3.783 | 4.473 | 2.014 | 極端 echo / demo |

> 數據：800-case AEC Challenge blind test，本地 AECMOS ONNX (Run_1663915512_Stage_0)，FL=512。

### 場景特殊處理

| 場景 | 機制 | 觸發條件 | 效果 |
|------|------|---------|------|
| Echo Path Change | EPC detector + P_MAX 放寬 | error 突升 + delta 變化 | 0.5-1s 內 re-converge |
| Stationary far-end DT | CV² + jump_ratio + v3 mask | far CV²<0.02 + 語音突波 | 語音保留 5% → 48% |
| Movement DT | Online delay re-estimation | 距離變化 | 動態追蹤新 echo path |
| 弱信號 / 靜音 | Far activity gate | far_pwr < 1e-4 | 暫停 filter 更新 |

## 檔案結構

```
AEC/
├── README.md                  # 本文件（Python 為主）
├── c_impl/                    # C 實作 (PBFDKF + Shadow + WOLA RES + HPF + Preset)
│   ├── include/               # aec.h, aec_types.h, pbfdkf.h, res_filter.h, hpf.h
│   ├── src/                   # aec.c, pbfdkf.c, res_filter.c, hpf.c, fft_wrapper.c
│   ├── example/               # main.c (CLI), wav_io.h
│   └── Makefile
├── python/
│   ├── aec.py                 # Python 實作 (PBFDAF/PBFDKF + Shadow + RES + Delay Est.)
│   ├── eval_aec_challenge.py  # AEC Challenge benchmark 評測 (ERLE/PESQ)
│   ├── eval_aecmos.py         # AECMOS 評測 (echo_mos/deg_mos)
│   ├── plot_aec_results.py    # 結果繪圖 (含 DTD 紅底)
│   └── gen_sim_data.py        # 測試資料生成
├── docs/
│   ├── aec_methods.md                  # 演算法完整技術文件
│   ├── aec_algorithm_guide.html        # 技術介紹（HTML，含圖表）
│   ├── pbfdkf_shadow_intro.md          # PBFDKF + Shadow + DTD 演算法介紹（報告用）
│   ├── movement_dt_ablation_report.md  # Movement DT 三輪 ablation 結論報告
│   ├── aec_improve_v12.md              # v1.20 改善紀錄
│   ├── aec_improve_v13.md              # v2.x 改善紀錄（Stationary far-end DT）
│   ├── dtd_design.md                   # DTD 設計文件
│   ├── phase_kappa_report.md           # Phase κ 報告
│   ├── DEVLOG.md                       # 開發紀錄
│   └── TODO.md                         # 待辦與結案實驗記錄
└── wav/                       # 測試音檔
```

## 參考文獻

1. Haykin, S. "Adaptive Filter Theory"
2. Benesty, J. et al. "Advances in Network and Acoustic Echo Cancellation"
3. Sondhi, M.M. "An adaptive echo canceller" (1967)
4. Ferrara, E.R. "Fast implementations of LMS adaptive filters" (1980)
5. Enzner, G. et al. "Frequency-domain adaptive Kalman filter for acoustic echo control" (2006)
