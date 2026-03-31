# AEC 開發紀錄

## 版本歷史

### v1.28.0 (2026-03-31) - Bug Fixes: CNG Gate, Coherence Reset, fs_confidence Dedup

**Bug fixes:**
- **CNG noise tracking gate**: `far_activity < 0.1` gate 取代無條件更新
  - 原版在 far-end silence 時無條件更新 noise_psd，DT 時 near-end speech 可能污染 noise
  - 初版用 `dt_indicator < 0.1` 太嚴格（echo 衰減尾巴讓 dt_indicator 殘留 > 0.1 → noise_psd 全零 → CNG 失效）
  - 最終改用 `far_activity < 0.1`：EMA TC≈800ms，真正靜音才更新，DT 天然隔離
- **_coh2_smooth proper init/reset**: 從 lazy `hasattr` 改為 `__init__` + `reset()` 正確初始化
  - 舊版 hasattr lazy init 在 reset() 後狀態殘留（跨檔案污染）
- **fs_confidence single definition**: 移除重複計算（原本在 ne_g_floor 和 attack blend 各算一次）
- **NE nores output**: eval_aec_challenge.py 補上 nearend_singletalk 的 ours_nores 輸出

**Blind test (fl=512, balanced, 800 cases):**
| Metric | v1.27.0 | v1.28.0 | Change |
|--------|---------|---------|--------|
| FS echo_mos | 3.709 | **3.713** | +0.004 |
| DT echo_mos | 4.273 | **4.370** | **+0.097** |
| DT deg_mos | 2.187 | 2.092 | -0.095 |
| NE deg_mos | 4.005 | 3.996 | -0.009 |

### v1.27.0 (2026-03-31) - AEC3-Style RES + Continuous Attack + CNG Crossfade

**Architecture improvements (v1.25-v1.27):**
- ENR formula: `echo/(error-echo+offset)` → `echo/error` (AEC3 style, no subtraction)
- Continuous attack speed: `fs_confidence = far_activity × (1-dt)²` replaces binary nearend_state
- CNG crossfade: `cn_gain = sqrt(1 - G²)` fills suppressed regions with comfort noise
- EMR noise masking: don't suppress echo below noise floor
- Q modulation: Q × (0.1 + 0.9 × mu) during DT
- Preset recalibration for new ENR scale

**Tested but reverted/ineffective:**
- noise_psd continuous tracking: hurt FS echo -0.24 (noise level too high during far-end active)
- R alpha adaptive (0.98-0.10×mu): marginal effect
- NE_TRIGGER_FRAMES 5: no benefit vs 12
- filter_length 1024: +0.003 FS echo vs 512, not worth cost
- coh2-based nearend detection: coh2 too low in FS (0.1-0.19)
- filter-based |H(f)|² path gain: too small at fl=512
- Audibility weighting: FS echo -0.02

**Final blind test (fl=512, preset balanced):**
| Metric | v1.18.0 (original) | v1.27.0 | Change | AEC3 |
|--------|-------------------|---------|--------|------|
| FS echo_mos | 3.210 | **3.709** | **+0.499** | 3.963 |
| DT echo_mos | 4.000 | **4.273** | **+0.273** | 4.440 |
| DT deg_mos | 2.215 | **2.187** | -0.028 | 2.258 |
| NE deg_mos | 4.018 | **4.005** | -0.013 | 3.530 |

### v1.24.0 (2026-03-30) - RES Suppression Breakthrough

**Critical bug fix: ENR offset +1.0 → adaptive (committed in v1.23.0)**
- `enr = echo / (nearend + 1.0)` — +1.0 was too large for PSD scale (0.01-10)
- ENR was always < 1.0, preventing gain from reaching g_min
- Fix: `enr = echo / (nearend + mean(error) * 0.01)`
- Result: FS echo +0.17 (3.40→3.57), DT echo +0.14

**RES architecture improvements (AEC3-style):**
- **Fixed g_min**: removed far_activity gating on g_min (was raising floor during low echo)
  - AEC3 uses fixed g_min; gain is purely ENR-driven
  - Fixes low-echo segments where output stayed at -10dB instead of -40dB
- **Nearend-aware attack speed**: fast attack (α=0.3) in FS, slow (α=0.7) in DT
  - FS: gain drops to g_min in ~5 frames (50ms) instead of ~30 frames (300ms)
  - DT: gain changes slowly to protect near-end speech
- **Nearend ENR threshold recalibrated**: 1.09→3.0 (LF), 0.1→0.3 (HF)
  - Old threshold 1.09 was designed for +1.0 offset (enr ≤ 1)
  - New scale after offset fix: enr can reach 100+, threshold 3.0 is appropriate
- **Soft upper bound**: residual_echo_psd ≤ 4× error_psd
  - Prevents echo_psd spikes (20× error) from crushing DT speech
- **Convergence detection**: ERLE > 10dB × 10 frames (was 6dB × 6)
- **R alpha**: 0.95 → 0.98 (slower R tracking)
- **Warmup guard**: no convergence detection during warmup

**Blind test results (fl=512, preset balanced):**
| Metric | v1.18.0 (original) | v1.24.0 | Change | AEC3 |
|--------|-------------------|---------|--------|------|
| FS echo_mos | 3.210 | **3.625** | **+0.415** | 3.963 |
| DT echo_mos | 4.000 | **4.257** | **+0.257** | 4.440 |
| DT deg_mos | 2.215 | 2.064 | -0.151 | 2.258 |
| NE deg_mos | 4.018 | 4.011 | -0.007 | 3.530 |

### v1.22.0 (2026-03-29) - Kalman R-Deadlock Fix + RES Tuning + AEC3 Gap Analysis

**Kalman filter convergence fix (biggest impact):**
- R scaling by mu_scale: FS R×0.3 (K large), DT R×1.0 (protect weights)
- Q_low 1e-7→1e-5: prevent P from dying after convergence
- EPC triggered on delay shift: filter re-converges after delay change
- Linear-only ERLE improved from 2.05 to 2.50 (+0.45 on blind test)

**RES/NLP improvements:**
- Q_gated soft floor (Q×0.05): weak bins don't freeze
- alpha_coh 0.3→0.65: more stable coherence tracking
- ERLE cap HF 4→16, LF 8→32: more accurate ERLE tracking
- Echo boost DT guard: 2.0→0.5 in FS, off in DT
- coh2-weighted echo floor: per-bin residual echo estimation
- BALANCED preset: g_min -45→-55, over_sub 4/7→5/9, dt_reduction 2→2.5

**Shadow filter:**
- shadow_copy_threshold 0.7→0.65, hysteresis 5→3

**Documentation:**
- aec_methods.md §9.1: Filter Length / Partition design guide

**Blind test results (fl=512, preset balanced):**
| Metric | v1.18.0 | v1.22.0 | Change |
|--------|---------|---------|--------|
| FS echo_mos | 3.210 | 3.336 | +0.126 |
| DT echo_mos | 4.000 | 4.025 | +0.025 |
| DT deg_mos | 2.215 | 2.146 | -0.069 |

### v1.18.0 (2026-03-24) - Reverb Render Signal + Divergence Suppression + Delay Tracking

#### 改動內容

1. **Reverb tail 改用 render signal 估計**
   - 改前：`reverb_psd = decay × reverb_psd + (1-decay) × |echo_spec|²`（依賴 filter 建模品質）
   - 改後：`reverb_psd = decay × reverb_psd + (1-decay) × |far_spec|²`（直接用 render signal）
   - `far_spec` 代表揚聲器正在播放的 render energy，無論 filter 收了多少都正確
   - Filter 未建模的 reverb tail 部分也能正確估計 → suppressor 更積極壓制殘餘 reverb
   - **ERLE +1.4 dB**（BALANCED: 5.1 → 6.5 dB）

2. **Divergence-triggered suppression**
   - 新增 `_divergence_indicator`：smoothed EMA [0,1]，追蹤 filter 發散狀態
   - 觸發條件：收斂後 `inst_erle_linear < 0.63`（ERLE < -2dB = filter 放大而非消除）
   - RES gain override：`divergence > 0.3` → `g ≤ 0.01 + 0.99 × (1 - divergence)`
   - 仿 AEC3 transparent mode：filter 發散時強制壓制，防止 echo 穿透
   - Benchmark 無退化（固定 echo path 不會觸發 divergence）

3. **延遲追蹤頻率加快 + 漂移保護**
   - `delay_est_period_s`: 2.0 → **0.5**（每 0.5 秒重估）
   - `delay_est_init_s`: 0.5 → **0.3**（更快取得第一次估計）
   - 漂移保護：延遲大幅改變（>32 samples）需連續兩次估計一致才更新
   - 防止 GCC-PHAT 單次誤判導致延遲跳變
   - Benchmark 無退化（固定延遲測試集不受影響，效果在真實裝置延遲漂移場景）

#### Benchmark 結果（AEC Challenge, 15 files, subband + FDKF + Shadow + RES）

| Preset | v1.17.0 | v1.18.0 | 提升 |
|--------|---------|---------|------|
| BALANCED | 5.1 dB | **6.5 dB** | **+1.4 dB** |
| AGGRESSIVE | — | **8.6 dB** | — |
| MAXIMUM | — | **9.1 dB** | — |

#### 設計決策

- **為何 reverb 用 far_psd 而非 echo_pwr**：filter 只建模 echo path 前段（direct path），reverb tail 建模不完整 → echo_pwr 低估 → suppressor 放行殘餘 reverb。far_psd 直接反映揚聲器播放能量，與 filter 品質無關。
- **為何 divergence 用 smoothed EMA**：單幀 ERLE 波動大，直接用會誤觸發。α=0.9 attack + 0.95 decay 提供約 10 幀（100ms）的平滑，避免瞬態假陽性。
- **為何延遲需要兩次確認**：GCC-PHAT 在低 SNR 或近端語音活動時可能產生錯誤峰值。連續兩次一致（差 < 16 samples）才確認是真正的延遲漂移。

---

### v1.17.0 (2026-03-24) - Kalman 修正 + C 實作全同步 Python v1.17.0

#### 改動內容

1. **Kalman 參數修正（Python + C）**
   - P_init: 0.5 → **0.01**（防止啟動時 K≈33 造成 weight jitter / buzzing）
   - P_MAX: 無 → **0.02**（clamp 上限，防止 P 爆炸）
   - Q_high: 1e-3/3e-3/1e-2（per-preset）→ **1e-4**（所有 preset 統一，消除格子狀 artifact）
   - Q_low: 1e-5 hardcoded → **1e-7**（configurable via `kalman_q_low`）
   - Per-bin Q gating: 僅在 `power[k] > mean_power*0.01+1e-6` 的 bin 注入 Q（靜音 bin 不加 Q）
   - Far-end gate: `total_power > delta*nfreq` → **`far_hop_energy > 1e-6`**（~-60 dBFS，更精確的活動偵測）

2. **Preset 大幅加強（AGGRESSIVE / MAXIMUM）**
   - AGGRESSIVE: g_min=-35→**-60**, enr_scale=0.7→**0.3**, over_sub_base=4→**6**, ne_protect=-5→**-3**, reverb_gain=1.0→**2.0**
   - MAXIMUM: g_min=-40→**-72**, enr_scale=0.5→**0.15**, over_sub_base=6→**10**, ne_protect=-2→**-1**, reverb_gain=1.5→**3.0**
   - 設計哲學：「傷到語音沒關係」，最大限度壓制殘餘 echo

3. **C 實作全面同步 Python v1.17.0**
   - `pbfdkf.c`: P_init=0.01, P_MAX=0.02, Q gating, far-end gate, configurable q_low
   - `res_filter.c`: RES v2（ENR masking, direct echo est, 4-block near-end avg, reverb tail, freq postprocessing）
   - `aec_types.h`: 新增 `ResEchoMethod`/`ResGainType` enum, `kalman_q_low` + RES v2 config 欄位
   - `aec.c`: 接線所有新參數到 PBFDKF 和 RES
   - `example/main.c`: 印出 RES v2 / Reverb / Kalman Q_low 參數

4. **移除失敗的測試 code**
   - Q decay 方案（無效，移除）
   - Debug print（移除）

#### 設計決策

- **為何 Q_high=1e-4 而非 1e-3**: Q_high=1e-3 在 startup 造成 K 值過高，weight 在 DT 時劇烈震動產生格子狀 spectrogram artifact。1e-4 配合 P_init=0.01 和 P_MAX=0.02 足以打破 R deadlock 同時避免 jitter。
- **為何所有 preset 統一 Q_high**: 不同 Q_high 導致 AGGRESSIVE 收斂反而比 BALANCED 差（更高 Q → 更大 jitter → 更差 ERLE）。統一 1e-4 後所有 preset 共享相同收斂品質，差異化改由 RES preset 控制。
- **為何 P_MAX=0.02**: P_init=0.01 正常運作，但在某些場景 P 可能回升。0.02 提供 2× headroom 同時防止失控。

---

### v1.16.0 (2026-03-24) - RES v2: ENR Masking + Direct Echo Estimation + Reverb Tail

#### 改動內容

1. **ENR masking 增益公式（取代 spectral subtraction）**
   - 新增 `res_gain_type="enr"` 作為預設增益公式（仿 AEC3 `GainToNoAudibleEcho`）
   - `ENR = residual_echo / (nearend_est + 1.0)`，不需要估計 nearend PSD
   - 頻率相關門檻：LF `enr_t=0.3, enr_s=0.4`、HF `enr_t=0.07, enr_s=0.1`（平滑過渡 bins 5-10）
   - Soft gate: `enr < enr_t → g=1.0`、`enr > enr_s → g=0`、中間線性內插
   - FS 場景 nearend≈0 → ENR 很大 → 自然完全抑制，無 gain pumping
   - 保留 `"wiener"` 和 `"spectral_sub"` 供比較

2. **獨立 ENR 門檻參數 `res_enr_scale`**
   - ENR 門檻與 `over_sub` 解耦，避免語義反轉問題
   - `scale < 1.0` → 門檻更低 → 更激進壓制
   - Preset: BALANCED=1.0（AEC3 默認）、AGGRESSIVE=0.7、MAXIMUM=0.5

3. **Direct echo estimation（取代 coherence-based）**
   - `res_echo_method="direct"`: 使用自適應濾波器的 echo spectrum + per-bin ERLE 追蹤
   - Per-bin ERLE: `inst_erle = near_psd / error_psd`，EMA α=0.95，3-bin 頻率平滑
   - 兩種估計融合：`residual = (1-erle_factor) × direct + erle_factor × erle_corrected`
   - 僅在安全條件下更新（far_active + not_dt + converged）

4. **4-block 近端移動平均**（仿 AEC3 `nearend_average_blocks=4`）
   - Ring buffer 平滑近端 PSD，減少幀間增益抖動
   - `near_psd = mean(near_psd_buf[0:4], axis=0)`

5. **Reverb tail model**
   - `reverb_psd = decay × reverb_psd + (1-decay) × echo_pwr`
   - `residual_echo_psd += reverb_gain × reverb_psd × far_activity`
   - Preset: BALANCED decay=0.5/gain=0.5、AGGRESSIVE 0.6/1.0、MAXIMUM 0.7/1.5

6. **頻率域後處理**（仿 AEC3 `PostprocessGains`）
   - DC 一致性：`g[:2] = min(g[1], g[2])`
   - HF cap: `g[17:] ≤ g[16]`（~2kHz 以上不超過 2kHz 增益）

7. **收緊增益上升速度**
   - `eff_rise = max_rise ^ (0.5 + 0.5 × (1 - far_activity))`
   - Far-end 活躍時上升更慢，防止 echo burst 穿透

8. **Wiener gain 修正**（保留供比較）
   - Noise floor 提高至 `mean(error_psd) × 0.01`（原 0.001），防止 FS gain pumping

9. **輸出檔名修正**
   - `{uuid}_fs_ours.wav` → `{uuid}_farend_singletalk_ours.wav`（與原始 AEC Challenge 命名一致）
   - eval_aec_challenge.py / eval_aecmos.py 同步更新

10. **其他**
    - `.gitignore` 新增 `wav/aec_challenge/` 和 `wav/aec_challenge_blind/`（本地 dataset 不入版控）
    - `benchmark_competitors.py` 新增 AEC3 Linear 支援
    - `eval_aec_challenge.py` 新增 `AEC_GAIN_TYPE` env var 支援（A/B test 用）

#### A/B Test 結果

**小資料集（15 cases, BALANCED preset）**：

| 指標 | v1.15.0 | Wiener (fixed) | ENR (v2) |
|------|---------|----------------|----------|
| FS ERLE | 11.9 dB | 17.2 dB | **16.2 dB** |
| FS echo_mos | 3.18 | 3.45 | **3.42** |
| DT echo_mos | 3.06 | 3.66 | **3.67** |
| DT deg_mos | 3.51 | 3.34 | **3.06** |

**Blind test（555 cases, BALANCED preset, ENR old scale）**：

| 指標 | Wiener | ENR | 差異 |
|------|--------|-----|------|
| FS ERLE | 13.1 dB | **13.4 dB** | +0.3 |
| FS echo_mos | 3.263 | **3.478** | **+0.215** |
| DT echo_mos | 3.876 | 3.868 | ~持平 |
| DT deg_mos | 2.686 | **2.703** | +0.017 |
| NE echo_mos | 4.997 | 4.998 | ~持平 |
| NE deg_mos | 4.080 | 4.077 | ~持平 |

**結論**: ENR 在 FS echo_mos 上顯著領先 Wiener (+0.215)，其他持平。選擇 ENR 作為預設。

#### 設計決策

- **為何選 ENR 而非 Wiener**: Wiener gain 在 FS 場景 `nearend_est → 0` 時不穩定（gain pumping），即使加了 noise floor 修正仍不如 ENR 穩定。ENR masking 不需要 nearend 估計，FS 時自然趨近 0，DT 時自然放行。
- **為何 ENR 用獨立 `res_enr_scale` 而非 `over_sub`**: `over_sub` 越高 = 越激進（spectral_sub/wiener），但 ENR 中 `scale` 越高 = 門檻越高 = 越寬鬆，語義相反。獨立參數更直觀。
- **ENR 不需要動態縮放**: `residual_echo_psd` 已隨收斂狀態自適應（收斂前保守 → 壓多；收斂後準確 → 壓剛好），不需要像 spectral_sub 那樣動態調整 over_sub。

---

### v1.15.1 (2026-03-23) - C 實作完整重寫對齊 Python v1.15.0

#### 改動內容

1. **C 實作完整重寫**
   - 對齊 Python v1.15.0 SUBBAND 模式（PBFDKF + Shadow + WOLA RES + HPF + Preset）
   - `SubbandNlms` → `Pbfdkf`（FDKF Kalman always on，移除 NLMS fallback）
   - 獨立 `hop_size=160` 和 `block_size=512`（不再硬綁 block_size/2）
   - RES 改為 WOLA（sqrt-Hann 窗, frame=320, zero-pad→512）
   - 新增 HPF（2nd-order Butterworth IIR, 80Hz cutoff）
   - 三級 Preset：`--preset balanced|aggressive|maximum`
   - 預設 filter_length=1600（100ms，因無 delay estimation 需較長濾波器）

2. **新增/重寫檔案**
   - `include/aec_types.h` — AecPreset enum + AecConfig struct + factory functions
   - `include/pbfdkf.h` + `src/pbfdkf.c` — PBFDKF（取代 subband_nlms）
   - `include/res_filter.h` + `src/res_filter.c` — WOLA RES（完整重寫）
   - `include/hpf.h` + `src/hpf.c` — 新增 HPF
   - `include/aec.h` + `src/aec.c` — 協調器重寫（Shadow + simple mu + convergence）
   - `example/main.c` — CLI 重寫（preset 選擇）
   - 移除: `subband_nlms.h`, `subband_nlms.c`（由 pbfdkf 取代）

3. **C 版本包含**
   - PBFDKF Kalman（per-bin K, two-stage Q, time-domain constraint）
   - Shadow filter（dual-filter, copy gate with hysteresis, bidirectional copy）
   - Simple variable mu（Valin 2007 RER-inspired, asymmetric EMA + holdoff）
   - WOLA RES（coherence NL echo PSD, per-bin near-end gate, dynamic g_min, spectral floor, rate limiting, CNG）
   - HPF（2nd-order Butterworth IIR）
   - Output limiter（smoothed gain clamp）
   - Convergence detection（10 consecutive frames ERLE > 6dB → Q_high → Q_low）

4. **C 版本不含**
   - DTD（Double-Talk Detection）
   - Delay Estimation（GCC-PHAT）
   - Saturation Detection
   - LMS/NLMS/FDAF 模式

#### 驗證結果

使用 AEC Challenge farend_singletalk 測試（filter_length=1600, 10 partitions）：

| Preset | ERLE (C, 無 delay est) |
|--------|----------------------|
| BALANCED | 5.9 dB |
| AGGRESSIVE | 5.5 dB |
| MAXIMUM | 3.9 dB |

> 注：C 版本無 delay estimation，需較長濾波器補償。Python v1.15.0 同一 file 使用 delay est (414 samples) + 4 partitions 達 5.8 dB。

---

### v1.15.0 (2026-03-23) - Frame/Hop 統一 + Dead Code 清理 + FREQ→FDAF 重命名

#### 改動內容

1. **統一 frame/hop/FFT size**
   - `frame_size`: 512 → **320** (20ms @ 16kHz)
   - `hop_size`: 256 → **160** (10ms @ 16kHz)
   - `fft_size`: 512（不變，nearest 2^n ≥ 320）
   - 延遲從 16ms 降至 **10ms**

2. **SubbandNlms overlap-save 調整**
   - 新增獨立 `hop_size` 參數（不再硬綁 `block_size // 2`）
   - Output 提取改為 `buffer[-hop:]`（取最後 hop 個 valid sample）
   - Weight constraint: `w_time[hop_size:] = 0`（partition 長度 = hop = 160）
   - n_partitions: `ceil(filter_length / 160)` = 4 partitions（was 4 with hop=256, fl=1024）

3. **ResFilter WOLA 調整**
   - `frame_size=320`（WOLA 窗長），`hop_size=160`（50% overlap，sqrt-Hann 完美重建 ✓）
   - FFT: zero-pad 320→512 → `rfft(512)` → 257 bins（與 SubbandNlms 相同 n_freqs）
   - IFFT: `irfft(257, n=512)[:320]` → synthesis window → OLA

4. **FDAF buffering 修正**
   - FDAF 模式 internal_hop (512) 不整除 external_hop (160)
   - Buffer 擴大至 `internal_hop + hop_size` 並加入 leftover 處理

5. **FREQ → FDAF 重命名**
   - `AecMode.FREQ` → `AecMode.FDAF`
   - CLI: `--mode freq` → `--mode fdaf`
   - 所有相關檔案同步更新（evaluate_aec.py, batch_aec.py, plot_aec_results.py 等）

6. **Dead code 清理**
   - 移除 `from dataclasses import field`（未使用 import）
   - 移除 `AecConfig.leak`（永遠被 override 為 1.0）
   - 移除 `AecConfig.use_leakage` + `freq_leakage`（default False，CLI 無 flag，無使用）
   - 移除 `AecConfig.dtd_threshold`（legacy，never read）
   - 移除 `NlmsFilter.leak` 參數（永遠 1.0，乘法無效果）
   - 移除 `SubbandNlms.use_leakage`/`leakage` 參數

7. **`use_kalman` 預設改為 True**
   - `AecConfig.use_kalman: bool = False` → `True`
   - FDKF 為推薦配置，預設開啟更符合使用習慣

#### 驗證結果（AEC Challenge, 15 files, FL=512, subband + Shadow + FDKF + RES）

**Farend Single-Talk**：

| Preset | ERLE | vs v1.14.0 |
|--------|------|------------|
| BALANCED | 12.4 dB | +0.5 dB |
| AGGRESSIVE | **15.7 dB** | **+1.0 dB** |
| MAXIMUM | 15.0 dB | +1.1 dB |

**Doubletalk**：

| Preset | ERLE | vs v1.14.0 |
|--------|------|------------|
| BALANCED | 4.2 dB | -0.2 dB |
| AGGRESSIVE | **7.2 dB** | -0.2 dB |
| MAXIMUM | 7.9 dB | -0.3 dB |

**關鍵發現**：
- FS ERLE 全面提升（AGGRESSIVE +1.0 dB），frame size 縮小提升時間解析度
- DT ERLE 輕微下降（-0.2 dB），在噪聲範圍內，可接受
- 延遲從 16ms 降至 10ms，更適合即時通話場景

---

### v1.14.0 (2026-03-23) - Per-bin Near-end Gate + Preset 微調

#### 改動內容

1. **Per-bin near-end gate（取代 broadband dt_indicator）**
   - 舊版：broadband `dt_indicator = 1 - far/(mic+far)` → FS 時 mic≈far → dt_indicator≈0.5 → g_min 被抬到 -10dB → echo-only 無法激進壓制
   - 新版：per-bin `coh2` 控制 g_min floor
     - `coh2[k]` 高（echo bin）→ ne_protection≈0 → 允許壓到 g_min
     - `coh2[k]` 低（near-end bin）→ ne_protection≈1 → floor 提高到 `ne_protect_db`
   - 僅在收斂後啟用（`erle_factor` gate）

2. **新增 AecConfig 參數**
   - `res_ne_protect_db: float = -10.0`：per-bin 近端保護上限

3. **coh2 收斂後 alpha 調整**
   - 收斂後 rise alpha 0.80→0.90（TC≈160ms），coh2 在 echo-only 段更穩定接近 1.0
   - Drop alpha 不變（0.50），DT 時快速反應

4. **Preset 參數微調**

   | 參數 | BALANCED | AGGRESSIVE | MAXIMUM |
   |------|----------|------------|---------|
   | res_ne_protect_db | -8 | -5 | -2 |
   | res_dt_reduction | 3.5 | 2.0 (was 2.5) | 0.0 (was 1.5) |

#### 驗證結果（AEC Challenge, 15 files, FL=1024, subband + Shadow + FDKF + RES）

**Farend Single-Talk AECMOS**：

| Preset | echo_mos | ERLE | vs v1.13.0 |
|--------|----------|------|------------|
| BALANCED | 3.18 | 11.9 dB | ≈ baseline |
| AGGRESSIVE | **3.36** | **14.7 dB** | **+0.06** |
| MAXIMUM | 3.41 | 13.9 dB | +0.03 |

**Doubletalk AECMOS**：

| Preset | echo_mos | deg_mos | ERLE | vs v1.13.0 echo_mos |
|--------|----------|---------|------|---------------------|
| BALANCED | 3.06 | **3.51** | 4.4 dB | +0.04 |
| AGGRESSIVE | **3.42** | 3.22 | **7.4 dB** | **+0.21** |
| MAXIMUM | 3.59 | 2.59 | 8.2 dB | +0.26 |

**關鍵發現**：
- Per-bin gate 是 DT echo_mos 改善的主力：AGGRESSIVE +0.21（3.21→3.42），deg_mos 僅 -0.07
- DT ERLE 大幅提升：AGGRESSIVE 7.4 dB vs AEC3 3.7 dB（+3.7 dB）
- FS echo_mos 改善有限（+0.06），AECMOS 對 FS echo 評估更偏感知而非 ERLE
- MAXIMUM DT deg_mos 2.59 仍 > AEC3 2.51

---

### v1.13.0 (2026-03-23) - Preset Adaptive Filter 層參數

#### 改動內容

1. **AecConfig 新增可配置參數**
   - `kalman_q_high: float = 1e-3`：FDKF Q_high 收斂速度（原 hardcoded）
   - `warmup_frames: int = 100`：強制高 mu 的 warmup 幀數（原 hardcoded 100）

2. **AecPreset 新增 Adaptive Filter 層控制**
   - v1.12.0 的 preset 僅控制 RES 後處理層
   - v1.13.0 擴展至自適應濾波器層：`shadow_mu_min`、`warmup_frames`、`kalman_q_high`
   - 更高的值 = 更快收斂 + 更強 echo 消除，代價是 DT 時近端品質降低

   | 參數 | BALANCED | AGGRESSIVE | MAXIMUM |
   |------|----------|------------|---------|
   | shadow_mu_min | 0.5 | 0.7 | 0.9 |
   | warmup_frames | 100 | 150 | 200 |
   | kalman_q_high | 1e-3 | 3e-3 | 1e-2 |

3. **AEC.__init__() 套用 config 參數**
   - FDKF 模式：用 `config.kalman_q_high` 設定 main filter 的 Q_high/Q，再乘 `shadow_q_ratio` 給 shadow
   - warmup: 使用 `config.warmup_frames` 取代 hardcoded 100

#### 驗證結果（AEC Challenge, 15 files, FL=1024, subband + Shadow + FDKF + RES）

**Farend Single-Talk AECMOS**：

| Preset | echo_mos | deg_mos | vs v1.12.0 (no preset) |
|--------|----------|---------|------------------------|
| BALANCED | 3.19 | 5.00 | ≈ baseline |
| AGGRESSIVE | **3.30** | 5.00 | **+0.10** |
| MAXIMUM | 3.38 | 5.00 | +0.18 |

**Doubletalk AECMOS**：

| Preset | echo_mos | deg_mos | 備註 |
|--------|----------|---------|------|
| BALANCED | 3.02 | 3.52 | 近端保留最佳 |
| AGGRESSIVE | **3.21** | 3.29 | **推薦平衡點** |
| MAXIMUM | 3.33 | 3.10 | echo 最低，近端損傷明顯 |

**關鍵發現**：
- Adaptive 層參數是 echo_mos 改善的主力（shadow_mu_min + kalman_q_high 加速收斂）
- AGGRESSIVE 推薦：FS echo_mos 3.30 (+0.10), DT echo_mos 3.21 (+0.19), DT deg_mos 3.29 (-0.23) — 可接受的 trade-off
- 即使 MAXIMUM 的 DT deg_mos (3.10) 仍大幅領先 AEC3 (2.51)

---

### v1.12.0 (2026-03-23) - FDKF Two-stage Q + Shadow Q Ratio

#### 改動內容

1. **FDKF Two-stage Q**
   - 新增 `Q_high = 1e-3`（快速收斂）和 `Q_low = 1e-5`（穩態追蹤）
   - 收斂判定：連續 10 幀 ERLE > 6dB → 切換 Q_high → Q_low
   - 解決 R self-reinforcing deadlock：低 Q → 低 P → 低 K → 高 R → 惡性循環
   - FS echo_mos +0.41（2.76 → 3.17）

2. **三項 Bug 修正**
   - Bug 1: Shadow filter 缺少 `use_kalman`，導致 shadow 跑 NLMS 而 main 跑 FDKF
   - Bug 2: `reset()` P 初始值不一致（0.1 vs init 的 0.5），Q 未重置為 Q_high
   - Bug 3: `copy_weights_from()` 未複製 P（FDKF 需要 P 和 W 一起複製）

3. **Shadow Q Ratio**
   - 新參數 `shadow_q_ratio: float = 3.0`
   - Shadow Q = main Q × ratio → shadow 收斂更快 → copy gate 開始觸發
   - 解決同 Q 下 copy gate 永不觸發的問題
   - FS echo_mos +0.03（3.17 → 3.20）

#### 驗證結果（AEC Challenge, 15 files, FL=1024, subband + Shadow + FDKF + RES）

| 指標 | v1.11.0 | v1.12.0 | SpeexDSP | WebRTC AEC3 |
|---|---|---|---|---|
| FS ERLE | 10.6 dB | **11.7 dB** | 6.9 dB | 25.8 dB |
| FS echo_mos | 2.76 | **3.20** | 3.09 | 4.47 |
| DT ERLE | — | **4.3 dB** | 1.3 dB | 3.7 dB |
| DT echo_mos | — | **3.03** | 2.76 | 4.39 |
| DT deg_mos | — | **3.52** | 3.90 | 2.51 |

**關鍵發現**：
- Two-stage Q 是最大的單一改善（FS echo_mos +0.41）
- DT deg_mos 大幅領先 AEC3（+1.01），近端語音保留最好
- DT ERLE 超越 AEC3（4.3 vs 3.7）
- 測試集從 10 cases 擴充至 15 cases

---

### v1.6.0 (2026-03-20) - Shadow only 預設模式

#### 架構決策

| 模式 | 預設 | 說明 |
|---|---|---|
| Shadow filter | ✅ 預設開啟 | 自適應，免調參，業界慣例（WebRTC/Speex） |
| DTD | ❌ 預設關閉 | 保留為進階選項，`--enable-dtd` 開啟 |

#### 改動內容

1. **預設模式改為 Shadow only**
   - `enable_dtd: False`（原 True），`enable_shadow: True`（原 False）
   - CLI：`--no-dtd` 移除，改為 `--enable-dtd`；`--enable-shadow` 改為 `--no-shadow`

2. **Shadow copy gate 加嚴**
   - `shadow_copy_threshold: 0.8 → 0.5`：需 shadow 比 main 好 50% 以上才 copy
   - 加入 far_active + not_dt gate：遠端靜音或 DT 期間不 copy（AEC3 style）
   - 非 DTD 模式用 `_simple_mu_ratio > 0.6` 判斷 not_dt

3. **Shadow filter mu 改為固定 1.0**
   - 舊版：DTD 模式用 `1.0 - conf × (1-0.2)`，非 DTD 用 simple variable mu
   - 新版：一律 `shadow_mu_scale = 1.0`（AEC3 background filter 做法）
   - 測試證實 shadow mu 保護對 NE-Ret 幾乎無影響，threshold 才是關鍵

4. **Main filter mu 保護（shadow_mu_min）**
   - 新參數 `shadow_mu_min: 0.5`：DT 時 main filter mu 最低保留 50%
   - `_get_simple_mu_scale()` 改用 `shadow_mu_min` 取代 `dtd_mu_min_ratio`
   - `_update_simple_mu_ratio()` attack alpha 0.5→0.3，holdoff 30→20（加快 DT 反應）

5. **benchmark_competitors.py 更新**
   - `run_ours()` 不再 hardcode `enable_dtd=True`，使用 AecConfig 預設值
   - 新增 `--enable-dtd` / `--no-shadow` CLI 參數
   - 輸出顯示 DTD/Shadow 配置

#### shadow_mu_min Sweep 結果

| shadow_mu_min | Mean ERLE | vs AEC3 | NE-Ret Mean |
|---|---|---|---|
| 0.05 | 6.2 dB | 21W | -4.2 dB |
| 0.1 | 6.3 dB | 22W | -4.2 dB |
| 0.2 | 6.5 dB | 23W | -4.2 dB |
| 0.3 | 6.7 dB | 24W | -4.2 dB |
| **0.5** | **7.0 dB** | **27W** | **-4.3 dB** |

NE-Ret 不受 shadow_mu_min 影響（差異來自 fid 1 white noise outlier）。

#### Copy gate 優化結果

| Config | NE-Ret Mean | NE-Ret Med | ERLE | vs AEC3 |
|---|---|---|---|---|
| threshold=0.8（舊） | -4.3 | -1.6 | 7.0 | 27W |
| **threshold=0.5（新）** | **-3.9** | **-1.6** | **7.0** | **27W** |

threshold 收嚴是 NE-Ret 改善主因（-4.3→-3.9），shadow mu 保護無效。

#### 驗證結果（AEC Challenge, 34 files, FL=1024, subband+RES）

| 模式 | Mean ERLE | Med ERLE | NE-Ret Mean | NE-Ret Med | vs AEC3 |
|---|---|---|---|---|---|
| DTD only | 6.3 dB | 5.0 dB | -3.0 dB | -1.6 dB | 23W/11L |
| **Shadow only（v1.6.0）** | **7.0 dB** | **5.6 dB** | **-3.9 dB** | **-1.6 dB** | **27W/7L** |
| DTD+Shadow | 6.5 dB | 4.9 dB | -3.8 dB | -1.7 dB | 23W/11L |
| SpeexDSP | 3.1 dB | 2.0 dB | -3.2 dB | -1.2 dB | — |
| WebRTC AEC3 | 7.3 dB | 4.4 dB | -5.9 dB | -1.8 dB | — |

**關鍵發現**：
- Median NE-Ret 完全相同（-1.6），Mean 差距來自 fid 1（white noise outlier: -55.6 vs -32.9）
- 排除 fid 1：Shadow NE-Ret ≈ -2.0 vs DTD ≈ -1.9，差距僅 0.1 dB
- ERLE 大幅領先 DTD（+0.7 dB），勝率 27W vs 23W
- Shadow only 在真實音訊上 NE-Ret 與 DTD 等效，ERLE 明顯更好，且免調參

---

### v1.5.1 (2026-03-20) - RES Wiener gain formula 改進

#### 改動內容

1. **RES gain formula 改為 spectral subtraction 形式**
   - 舊版 `g = 1/(1 + over_sub * eer)`：harmonic form，eer=1.0 時 g=0.5 (-6dB)，永遠不觸及 g_min
   - 新版 `g = max(1 - over_sub * eer, g_min)`：spectral subtraction，eer ≥ 1/over_sub 時直接壓到 g_min
   - 更接近 AEC3 suppressor 行為

2. **over_sub 1.0 → 3.0**
   - 新公式下 over_sub 語義不同（直接乘數 vs 分母），需重新調校
   - Sweep 結果：over_sub 越高 ERLE 越好，Near-end retention 不隨 over_sub 變化
   - NE-Ret 不變原因：far-end 靜音時 EER 本來就很低，gain ≈ 1.0

3. **benchmark 新增 Near-end Retention 指標**
   - 定義：far-end 靜音段的 `10*log10(mean(output²)/mean(mic²))`
   - 0 dB = 完美保留，> -1 dB 可接受，< -3 dB 有感失真
   - 新增 `--res-over-sub` CLI flag

#### over_sub Sweep 結果（新 Wiener gain formula）

| over_sub | Mean ERLE | vs AEC3 | NE-Ret Mean | NE-Ret Median |
|----------|-----------|---------|-------------|---------------|
| 1.0 | 5.1 dB | 12W/22L | -4.6 dB | -1.5 dB |
| 1.5 | 5.5 dB | 14W/20L | -4.7 dB | -1.6 dB |
| 2.0 | 5.8 dB | 16W/18L | -4.7 dB | -1.7 dB |
| 2.5 | 6.1 dB | 18W/16L | -4.7 dB | -1.7 dB |
| **3.0** | **6.2 dB** | **20W/14L** | -4.8 dB | -1.7 dB |

NE-Ret 幾乎不變 → 安全使用 over_sub=3.0。

#### 驗證結果（AEC Challenge, 34 files, FL=1024, subband DTD+RES）

| 指標 | v1.5.0 | v1.5.1 | SpeexDSP | WebRTC AEC3 |
|------|--------|--------|----------|-------------|
| Mean ERLE | 4.6 dB | **6.2 dB** | 3.1 dB | 7.3 dB |
| Median ERLE | 3.5 dB | **4.8 dB** | 2.0 dB | 4.4 dB |
| vs AEC3 | 8W/26L | **20W/14L** | — | — |
| NE-Ret Mean | — | -4.8 dB | -3.2 dB | -5.9 dB |
| NE-Ret Median | — | -1.7 dB | -1.2 dB | -1.8 dB |

v1.5.1 提升 +1.6 dB (Mean)，首次反超 AEC3 勝率（59%）。
AEC3 的 NE-Ret 比 Ours 更差（-5.9 vs -4.8），Ours 在 ERLE/NE-Ret tradeoff 上優於 AEC3。

#### 已驗證但未採用的修改

- **v4 B（FL=2048）**：+0.1 dB 無效，echo tail 在 64ms 內

---

### v1.5.0 (2026-03-19) - Power floor 修正 + RES EER soft gate + Benchmark

#### 改動內容

1. **SubbandNlms power floor 修正（P0-2）**
   - 舊版 `global_floor = max(power) * 0.01`：高能量 bin 主導 floor → 低能量 bin floor 過高 → mu_eff 過小 → 收斂慢
   - 新版 `global_floor = mean(power) * 0.01`：各 bin 獲得合理的 floor，低能量 bin 不再被壓制

2. **RES EER 改用 coherence soft gate（P1-1）**
   - 舊版 `eer = max(eer_linear, coh2)`：coh2 直接當 eer 下限，DT 時 coh2 仍高 → 近端語音被過度壓制
   - 新版 `eer = eer_linear * (0.5 + 0.5 * coh2)`：coherence 作為乘法 gate（0.5–1.0），DT 時自然降低壓制

3. **DTD coherence 綁定收斂狀態（P1-2）**
   - Coherence DTD 僅在 `_filter_converged=True` 後啟用
   - 收斂閾值 3 dB → 6 dB（更嚴格，避免未收斂時啟用 coherence DTD 誤判）

4. **新增 benchmark 工具 `benchmark_competitors.py`**
   - 自動比較 Ours vs SpeexDSP vs WebRTC AEC3
   - 計算 per-file ERLE、勝敗統計、mean/median

#### 驗證結果（AEC Challenge, 34 files, FL=1024, subband DTD+RES）

| 指標 | v1.4.1 (前版) | v1.5.0 | SpeexDSP | WebRTC AEC3 |
|------|--------------|--------|----------|-------------|
| Mean ERLE | 3.8 dB | 4.6 dB | 3.1 dB | 7.3 dB |
| Median ERLE | 2.7 dB | 3.5 dB | 2.0 dB | 4.4 dB |

v1.5.0 vs v1.4.1 提升 +0.8 dB (Mean)，大幅勝過 SpeexDSP，與 AEC3 差距 ~2.7 dB。

#### 已驗證但未採用的修改

- **v2 improve.md**：5 項修改全部造成退化（Mean 4.6→1.6），主因 P0-2 power normalization raw sum
- **v3 improve.md**：coherence-based echo PSD (`coh²×S_ee`) 理論正確但實測退化（4.6→3.7），coherence 統計收斂太慢
- **結論**：FDAF echo estimate 雖有 window mismatch，但 instantaneous 特性提供必要的壓制 margin

---

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
- [x] 延遲估計模組

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
| `enable_shadow` | Shadow filter（預設開啟） | — | True |
| `shadow_copy_threshold` | Shadow copy 門檻 | 0.3 - 0.8 | 0.5 |
| `shadow_mu_min` | Shadow-only mu floor | 0.1 - 0.5 | 0.5 |
| `enable_dtd` | DTD（預設關閉，進階選項） | — | False |
| `dtd_divergence_factor` | 發散判定倍數（DTD 模式） | 1.2 - 2.0 | 1.5 |
| `dtd_mu_min_ratio` | 發散時最低 mu 比例（DTD 模式） | 0.01 - 0.1 | 0.05 |

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
