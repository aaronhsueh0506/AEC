# AEC Phase κ — 開發紀錄與 SEP 實驗結果

> 日期：2026-04-26  
> 範圍：Phase κ bug fix（κ-1 ~ κ-4）+ Scene Separability Experiments（SEP-1 ~ SEP-5）

---

## 1. 版本背景

| 版本 | commit | 說明 |
|---|---|---|
| v2.6.0 | — | Phase ι 架構整理完成，FS echo 超越 AEC2 的基準版本 |
| v2.7.0 | — | Phase κ 起點，新增 SEP-3 diagnostics（9 個新 feature） |

---

## 2. 800-case AECMOS 基準分數

### 2-A. v2.6.0 baseline（Phase ι 後）

800 cases = FS×300 + DT×300 + NE×200（AEC Challenge blind set）

| Scenario | n | echo_mean | deg_mean |
|---|---:|---:|---:|
| farend_singletalk | 300 | 3.457 | 4.999 |
| doubletalk | 300 | 3.994 | 2.451 |
| nearend_singletalk | 200 | 4.998 | 4.093 |

### 2-B. Blind test final（Phase ι → 4 flag combinations）

比較基準：(0,0) = v2.6.0 baseline

| combo | FS_echo | DT_echo | DT_deg | NE_deg |
|---|---:|---:|---:|---:|
| (0,0) baseline | 3.475 | 4.034 | 2.448 | 4.005 |
| (1,0) B-16 | 3.475 | 4.033 | 2.449 | 4.005 |
| (0,1) Phase 2 | 3.445 | 4.039 | 2.458 | 4.012 |
| (1,1) stacked | 3.447 | 4.037 | 2.459 | 4.012 |

Movement vs static subset（FS + DT）：

| Scenario | subset | (0,0) | Δ(1,0) B-16 | Δ(0,1) Phase 2 | Δ(1,1) stacked |
|---|---|---:|---:|---:|---:|
| farend_singletalk | ALL (n=300) | 3.475 | +0.000 | -0.030 | -0.028 |
| farend_singletalk | movement (n=131) | 3.571 | -0.000 | -0.016 | -0.011 |
| farend_singletalk | static (n=169) | 3.400 | +0.001 | -0.040 | -0.042 |
| doubletalk | ALL (n=300) | 4.034 | -0.001 | +0.005 | +0.003 |
| doubletalk | movement (n=114) | 3.978 | -0.002 | -0.017 | -0.019 |
| doubletalk | static (n=186) | 4.068 | -0.001 | +0.018 | +0.016 |

### 2-C. Phase κ-4 vs Phase ι（800-case deg score）

比較基準：`deg_iota`（Phase ι / v2.6.0 equivalent）

| Scenario | subset | n | ι_deg | κ4_deg | Δ |
|---|---|---:|---:|---:|---:|
| farend_singletalk | static | 169 | 4.9994 | 4.9994 | +0.0000 |
| farend_singletalk | movement | 46 | 4.9995 | 4.9996 | +0.0001 |
| farend_singletalk | **ALL** | **215** | **4.9994** | **4.9994** | **+0.0000** |
| doubletalk | static | 186 | 2.4328 | 2.4337 | +0.0010 |
| doubletalk | movement | 114 | 2.4218 | 2.4003 | **-0.0215** |
| doubletalk | **ALL** | **300** | **2.4286** | **2.4210** | **-0.0076** |
| nearend_singletalk | ALL | 200 | 4.0116 | 4.0113 | -0.0003 |

DT movement 子集 -0.022（Fix E 在 DT+movement 場景改善明顯），FS 和 NE 不受影響。

---

## 3. Phase κ 改動紀錄

### κ-1-A（commit `198582d`）

**問題**：Shadow filter 在 DT 期間照常以全速學習，near-end speech 被學進 shadow filter。  
**修正**：DT-aware shadow gating — `_dt_from_energy` 高時降低 shadow adaptation rate，防止 NE learn-in。

### κ-1-C/D（commit `c662c84`）

**問題**：單一 frame 的大 mic 衝擊（burst）觸發 `epc_level='large'`，引發過長的 EPC refractory，壓制正常 copy gate。  
**修正**：
- **Fix C**：EPC large refractory 機制，large EPC 後強制冷卻 N frames
- **Fix D**：large EPC 須連續 sustained 才觸發，單次 burst 不算

### κ-2（commit `35b0925`）— Diagnostic only

新增 diagnostics，無行為改變：
- Copy gate 內部狀態：`error_is_normal`、`copy_err_baseline`、`err_baseline_ratio`、`adv_raw`、`adv_margin`
- Mic 失真代理指標：`mic_peak`、`mic_rms`、`mic_crest`、`mic_hi_ratio`（crest factor 和高幅值 sample 比例，偵測 soft clipping）

### κ-3-E（commits `0fca385` + `8651b04`）

**問題**：DT+movement 場景中，standard copy gate 因 `error_is_normal=False`（baseline 不可靠）而長期封閉，shadow filter 學到更好的 echo path 但無法 copy 回 main。  
**修正**：Fix E — 平行於標準 gate 的 sustained-advantage copy gate：
- 放寬 `error_is_normal` 要求
- 須連續 `_DT_COPY_SUSTAIN_FRAMES=4` frames 維持 shadow advantage 才 fire
- Cooldown `_DT_COPY_COOLDOWN_FRAMES=200` frames（anti-chase）
- 每 session 上限 `_DT_COPY_MAX_SESSION=3`（防 copy-chase loop）
- Advantage 必須在 `(1/0.99, 1.538)` 窗口內（排除 NE 偏差引起的 phantom advantage）

### κ-4（commit `7235499`）

兩個 Fix E 安全強化：

**Fix E.2 — DT-presence arm gate**：Fix E 只有在 `dt_from_energy ≥ 0.50` 持續 4 frames 後才 arm，防止 FS/NE 段（dte≈0）誤觸發。

**err_ratio floor**：`err_baseline_ratio ≥ 1.0` — main filter 必須高於 baseline 才允許 copy。防止 main 已低於 baseline 時 Fix E 仍 copy（backwards copy），避免把好的 main filter 替換掉。

---

## 4. Scene Separability 實驗（SEP-1 ~ SEP-5）

### 目標

找到能在 per-frame 或 per-file 層面區分以下兩個場景的 feature，作為 scene classifier 的輸入：

- **Set A（Bad DT）**：echo ≥ 4.0, deg < 2.5，共 127 files — AEC 在 DT 中聲學表現差
- **Set B（Movement FS）**：`_with_movement_` farend singletalk，共 131 files — 喇叭移動造成 echo path change

成功條件：任一 feature overlap（A vs B）< 0.7。

---

### SEP-1（per-frame，25 個現有 features）

**結論：全部 overlap = 1.000（POOR）**

測試所有 post-AEC error 相關 feature：ERLE、w_delta、coh2、effective_dt 等。  
根本原因：兩個場景都造成 filter 發散，per-frame 的 filter state 特徵無法區分發散原因。

---

### SEP-2（per-frame，3 個新 features）

**結論：全部 overlap = 1.000（POOR）**

| Feature | A_p50 | B_p50 | Overlap |
|---|---:|---:|---:|
| speech_band_coh2_mean | 0.2012 | 0.2079 | 0.995 |
| w_delta_norm | 0.0432 | 0.0412 | 1.000 |
| per_band_error_excess_mean | 0.1976 | 0.2317 | 1.000 |

---

### SEP-3（per-frame，9 個 scene features）— v2.7.0 新增 diagnostics

**結論：Phase 1 FAIL，全部 overlap = 1.000**

新增至 `aec.py` 的 diagnostics（pure diagnostic，無行為改變）：

| Feature | 說明 |
|---|---|
| `capture_to_echo_ratio` | Y²/S²_linear（AEC3 DominantNearendDetector） |
| `dominant_nearend_raw` | capture_to_echo_ratio > 4.0 |
| `filter_usable` | converged AND dt<0.4 AND NOT epc |
| `mic_far_coh_mean/speech` | near_spec vs far_spec coherence |
| `mic_excess_over_echo` | mic_pwr vs ERL×far |
| `movement_confidence` | rule-based（epc/render_based/w_delta） |
| `near_speech_confidence` | weighted composite |
| `scene_state` | DT/MV_FS/FS/NE/LOW_CONF/SAT |

致命問題：scene classifier 完全無法區分兩組。

```
scene_state 分布：
  Bad DT:    MOVEMENT_FS 44.8%  LOW_CONF 27.4%  FS 16.1%  NE 7.0%  DT 4.7%
  Mov FS:    MOVEMENT_FS 41.5%  LOW_CONF 29.9%  FS 16.1%  NE 6.4%  DT 6.1%
```

Bad DT 只有 4.7% 被正確歸類為 DOUBLE_TALK（通過條件需 >60%）。  
原因：Bad DT 觸發 EPC → `movement_confidence` 升高 → 被歸為 MOVEMENT_FS。

---

### SEP-4（per-frame，5 個 temporal pattern features）

**假設**：Movement FS = sudden w_delta spike 後恢復；Bad DT = 持續高 w_delta 無恢復。

新增到 `aec.py` 的 ring buffer diagnostics（pure diagnostic，無行為改變）：

| Feature | 機制 |
|---|---|
| `w_delta_recent_max` | 30-frame ring buffer 的 max |
| `w_delta_recent_mean` | 30-frame ring buffer 的 mean |
| `w_delta_spike_ratio` | max / (mean + 1e-6)，高 = sudden event |
| `erle_recovery_trend` | sign(erle_now − erle_50f_ago) |
| `frames_since_epc` | 自上次 EPC 事件後的 frame 數 |

**結論：全部 overlap = 1.000（POOR），假設錯誤**

| Feature | A_p50 | B_p50 | Overlap |
|---|---:|---:|---:|
| w_delta_spike_ratio | 1.3116 | 1.3057 | 1.000 |
| erle_recovery_trend | 0.0000 | 0.0000 | 1.000 |
| frames_since_epc | 0.0 | 123.0 | 1.000 |
| w_delta_recent_max | 0.0628 | 0.0570 | 1.000 |
| w_delta_recent_mean | 0.0460 | 0.0419 | 1.000 |

Time trace 顯示 Movement FS 也有大量 sustained high w_delta，並非只有 spike。  
`frames_since_epc` p50 有差（Bad DT=0, Mov FS=123），但 per-frame overlap 仍 = 1.0（因兩邊都有 epc=0 和 epc=1 的 frame）。

---

### SEP-5（per-file aggregates，12 features）

**假設**：per-frame 無法區分，但 file level 的 EPC 持續性與恢復率可以。

| Feature | A_p50 | B_p50 | Overlap |
|---|---:|---:|---:|
| max_consecutive_epc_frames | 1389 | 389 | **0.995** |
| sustained_dt_ratio | 0.154 | 0.222 | 0.999 |
| file_epc_fraction | 0.492 | 0.330 | 1.000 |
| clean_adaptation_window_ratio | 0.267 | 0.273 | 1.000 |
| recovery_success_after_epc | 0.500 | 0.500 | 1.000 |
| erle_p25 | 0.024 | 0.188 | 1.000 |
| erle_p50 | 0.253 | 0.289 | 1.000 |
| erle_p75 | 0.431 | 0.378 | 1.000 |
| time_to_recovery_after_epc | 1.0 | 1.0 | 1.000 |
| file_w_delta_p90 | 0.122 | 0.116 | 1.000 |
| far_only_window_ratio | 0.703 | 0.745 | 1.000 |
| max_consecutive_clean_adapt_frames | 239 | 233 | 1.000 |

**結論：Phase 1 FAIL，rule-based separation 不可行**

根本原因：Bad DT 和 Movement FS 在 filter state 效果完全相同。兩組的 file-level 分布都極寬：
- `file_epc_fraction`：Bad DT 分布 0.07→0.95，Mov FS 分布 0.00→0.97 — 完全重疊
- `recovery_success_after_epc` 兩邊 p50 都是 0.50 — EPC 結束後 Bad DT 也常短暫恢復再發散
- `max_consecutive_epc_frames` 是 12 個 feature 中差異最大的，但 Mov FS 的高端（p90=2043）仍重疊 Bad DT

---

## 5. SEP 總結與下一步

### 實驗總覽

| 實驗 | feature 類型 | 最佳 overlap | 結果 |
|---|---|---:|---|
| SEP-1 | per-frame，25 個舊 features | 1.000 | FAIL |
| SEP-2 | per-frame，3 個新 error features | 0.995 | FAIL |
| SEP-3 | per-frame，9 個 scene features | 1.000 | FAIL（scene 分類錯誤率 >40%） |
| SEP-4 | per-frame，5 個 temporal ring buffer | 1.000 | FAIL（假設錯誤） |
| SEP-5 | per-file，12 個 aggregate features | 0.995 | FAIL |

### 根本限制

從 AEC filter 內部觀察，Bad DT 和 Movement FS **在 filter state 層面效果完全一致**：
- 兩者都讓 filter 發散
- 兩者都觸發 EPC
- 兩者都沒有 ERLE
- 兩者的 w_delta、coherence、energy ratio 分布完全重疊

AEC 只能觀測「filter 壞了」，無法從內部信號判斷是「人在說話」還是「喇叭在移動」。

### 候選下一步

| 方向 | 說明 |
|---|---|
| **Delay drift** | Movement 造成 echo path delay 系統性飄移，加 `delay_change_rate` diagnostic；Bad DT 不改 delay |
| **Near-end VAD** | 有 near-end speech → DT；沒有 → Movement FS（需外部 VAD 或 mic 音訊特徵） |
| **Learned classifier** | 放棄 rule-based，用 audio features 訓練小分類模型（scene classifier as ML model） |
