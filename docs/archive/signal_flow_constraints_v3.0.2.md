# AEC v3.0.2 Signal Flow & Hardcoded Constraints

**目的**：DT (especially movement-DT) 表現輸 AEC2，多次單點調整無效。要找出整個訊號流的所有寫死值/門檻/限制，定位多點協調修改的位置。

**範圍**：v3.0.2 (post Phase A+B+C refactor)。`balanced` preset。16 kHz / hop 160。

---

## 0. Top-level pipeline

```
                        ┌─ saturation_detect ─┐
   mic ─┬─→ HPF80 ──────┴─────────────┬───→ PBFDKF main filter ──┐
        │                              │                           │
   ref ─┴─→ HPF80 → delay_est → align ─┘   (mu_scale gates)        │
                                       │   ┌── PBFDKF shadow ─┐    │
                                       └─→ │   (full-speed)   │    │
                                           └─→ shadow_copy ctrl┘    │
                                                                    │
            ┌────────── DTD (energy/shadow/coh) ─────────────┐      │
            │            EchoPathChange detector              │      │
            │            FilterConvergenceAnalyzer            │      │
            │            RenderActivityDetector → AecState    │      │
            └─────────────────────────────────────────────────┘      │
                                                                    │
   error ──→ ResidualEchoEstimator (stage 1+2) ──→ stage 3+4 ──→ ENR gain ──→ output_limiter
                          ↑                            ↑           ↑
                  echo_psd / coh2 / erl_estimate   over_sub     ne_protect / spectral_floor
```

---

## 1. Pre-processing

### 1a. HPF80 (DC + low-freq removal)
- `highpass_cutoff_hz = 80` Hz, biquad on mic and ref.
- 不被任何其他 state 影響。

### 1b. Saturation detector
- `saturation_threshold = 0.95` (|sample| 超過視為 clipping)。
- Output `_saturation_level ∈ [0,1]`，由 EMA 平滑。
- **影響**：
  - mic_clip > 0.8 → `mu_scale = 0.0`、`gain_smooth = g_min`（凍結 + 強壓）
  - sat_level > 0.5 → shadow filter `mu_scale = 0.1`（不再 1.0）
  - sat_level > 0.5 → RES `force_render = True`（render-based echo）
  - sat_level > 0.3 → `_nonlinear_frames++`，hangover 5 frames，nonlinear boost residual_echo
  - sat_level > 0.05 + far_active → harmonic-floor 注入 HF (500-4000Hz) 的 LF echo mean × (0.1+0.4·sat)
  - shadow copy gate: `not_saturating = sat_level < 0.3`
  - over_sub += `saturation_over_sub_boost · sat_level`（base + 3.0·sat）

### 1c. Delay estimator (GCC-PHAT)
- `delay_est_period_s = 0.25` (movement) 或 0.5 (static)
- `delay_est_init_s = 0.2` / 0.3
- `max_delay_ms = 250`
- 收斂條件：`_n_updates ≥ 3 AND _last_par > 5.0`
- **觸發 EPC**：
  - First acquisition：filter / shadow / RES `reset()` + `_convergence.mark_diverged()` + Q→Q_high
  - Delay shift > 32 samples 且兩次 |new−pending|<16 → `_epc_det.force_delay()` + Q→Q_high + P_max_override 30f + `mark_diverged()`

---

## 2. Filter convergence detector ([aec.py FilterConvergenceAnalyzer](AEC/python/aec.py#L2260))

**收斂規則**（**這條規則 gate 了下游 6+ 個決策**）：
- `inst_ERLE > 5.0 dB` 連續 10 frames（warmup_done=True 後）AND far_active (>1e-4)
- 收斂瞬間：filter Q → Q_low（穩定追蹤模式）

**Divergence indicator EMA**（post-converge only）：
- `inst_ERLE_lin < 0.63 (=ERLE<−2dB)` → div=1，否則 0
- EMA α=0.9，pre-conv decay 0.95

**`mark_diverged()` 觸發來源** (4 個)：
- delay first acquisition / delay shift consistent / EPV gain trigger / shadow_rise trigger

**下游消費 `filter_converged`**：
| 消費端 | 行為 |
|---|---|
| Filter mu_scale (startup_dt 路徑) | `_eff_dt > 0.35 AND far_active AND not once_converged` → mu_scale ≥ `startup_dt_mu_min` (預設 0.0)|
| EPV trigger | gated by `filter_converged AND not active AND not main_paused` |
| Shadow_rise trigger | gated by `filter_converged` |
| EPC hangover tick | gated by `filter_converged AND shadow_filter` ← **stuck bug** |
| ERL EMA α | converged 0.999 / unconverged 0.99 |
| Stationary DT detector | gated by `is_stationary AND filter_converged` |
| RES coh2 EMA rates | converged α_rise=0.90/α_drop=0.50；unconverged 0.80/0.50 |
| RES startup_dt cond | `eff_dt>0.35 AND far_active AND not converged` |
| RES startup_dt_once cond | `eff_dt>0.35 AND far_active AND not once_converged` |
| RES min_ne_from_dt scale | gated by `_startup_dt_cond` (not once_converged) |
| RES echo_boost | gated by `erle_factor > 0.3` (proxy for converged) |

→ **關鍵 cascading**：filter 沒收斂時，下游 11+ 個決策全部走「pre-conv」/「startup_dt」分支，互相耦合。

---

## 3. Filter mu_scale gating

### 3a. mu_scale 計算 (`_compute_mu_scale` / `_get_simple_mu_scale`)
- 非 DTD：`raw_dt = max(_dt_from_energy, simple_dt × 0.5)` (where `simple_dt = 1.0 - far/(mic+far)`)
- inst_erle 校正：`raw_dt /= min(_inst_erle_smooth, 4.0)` if smooth>2.0 (cap at 4.0)
- DTD 模式：取 max(divergence, coherence DTD confidence)
- mu_min floor：`epc_active → mu_floor = 0.5`

### 3b. Startup_dt floor 覆寫
- `if eff_dt > 0.35 AND far > 1e-4 AND not once_converged AND startup_dt_mu_min > 0.0`
- → mu_scale = max(mu_scale, startup_dt_mu_min) (預設 0.0 → 不啟用)

### 3c. Mic clipping emergency
- mic_clip > 0.8 → mu_scale = 0.0 + RES gain → g_min

### 3d. Per-bin mu_scale (post-RES, only no-DTD + converged)
- `per_bin_eer = clip(echo_psd / error_psd, 0, 1)`
- `per_bin_mu = mu_min + (1-mu_min)·per_bin_eer` (mu_min = `shadow_mu_min` 0.6 BALANCED)
- stationary_dt → freeze at `mu_min`
- pre-conv：`per_bin_mu_scale = None`（用 simple_mu_ratio）

### 3e. main_mu gate (shadow ctrl 決策)
- `main_mu = 0.0 if shadow_copy_ctrl.main_paused else mu_scale`

---

## 4. Shadow filter & copy gate

### 4a. Shadow filter update
- `shadow_q_ratio = 3.5` (BALANCED) → shadow Q_high = main Q_high × 3.5
- `far_excited = far_pwr > 1e-4`，`saturation_safe = sat_level < 0.5`
- shadow mu_scale = `1.0 if (far_excited AND saturation_safe) else 0.1`
- → 即使 DT 也以 1.0 適應；shadow 預期會追到 NE 但 `shadow_advantage` 訊號用來偵測 DT

### 4b. err_smooth EMA
- `shadow_err_alpha = 0.80` (BALANCED) → TC ≈ 5 frames

### 4c. Shadow DTD (DoubleTalkAnalyzer.update_shadow_dt)
- gate：`shadow_frame_count ≥ 50 AND far_excited`
- `shadow_advantage = main_err / shadow_err`
- `dt_from_shadow_raw = clip((adv - 1.5) / 3.0, 0, 1)`（offset/scale 寫死於 BALANCED）
- 70/30 EMA smooth

### 4d. Shadow copy controller ([ShadowCopyController](AEC/python/aec.py#L2739))
- gate (warmup) `frame ≥ 50`
- `is_stable_fs`: `far_active AND err_balance < 0.3 AND not epc_active`
- `_copy_err_baseline` EMA TC=200 frames (0.995/0.005)
- `error_is_normal`: `main_err < copy_err_baseline · 4.0 + 1e-10`
- DT-safe gate (`gate_mode='energy'` 預設): `dt_from_energy < 0.3`
- `copy_allowed = far_active AND error_is_normal AND not epc_active AND sat<0.3 AND dt_safe`
- 觸發 pause+Q_boost：`copy_counter ≥ shadow_copy_hysteresis (=3) AND streak ≥ 10`
- pause hangover：`epc_hangover (=20)` frames
- reverse copy（main 顯著好）：`main_err < shadow_err × 0.65 AND error_is_normal` → shadow ← main

### 4e. EPC triggers ([EchoPathChangeDetector](AEC/python/aec.py#L2613))
- **EPV** (gain ratio)：`fast_TC=0.98(50f) / slow_TC=0.999(1000f)`，threshold `< 0.25 OR > 4.0`（±6 dB）
  - gate：`filter_converged AND not active AND not main_paused`
  - 觸發後：Q→Q_high, P_max=1.0(30f), P_floor_beta=1.0(30f), `_render_forced=20f`, `erl ≤ 0.3`, `mark_diverged()`
- **Shadow rise**：`total_err > prev × 1.5 AND |Δerr|/total < 0.3`
  - gate：`filter_converged AND shadow_filter`
  - 觸發後：dtd_coh confidence × 0.3, Q→Q_high, P_max=1.0(30f), P_floor_beta=1.0(30f), `_render_forced=20f`, `erl ≤ 0.3`, `mark_diverged()`
- **EPC hangover tick** ([aec.py:3773](AEC/python/aec.py#L3773))：
  - **gated on `filter_converged AND shadow_filter`** ← **never-converge 卡死的 stuck bug**
  - hangover=20，每幀 -1，到 0 時 active=False

---

## 5. ERL estimate 學習

```python
if far_pwr > 1e-4:
    raw_dt_ratio = raw_err_pwr / far_pwr
    if raw_dt_ratio < 2.0:                       # gate：殘差不能太大（NE 主導 skip）
        inst_erl = clip(mic_pwr/far_pwr, 0.001, 10.0)
        alpha_erl = 0.99 if not converged else 0.999    # 收斂後極慢學習
        _erl_estimate = α·_erl + (1-α)·inst_erl
```

**`_erl_estimate` cap on EPC**：每次 EPV / shadow_rise / delay-shift 觸發 → `_erl ≤ 0.3`。

**下游用途**：
- DT energy detector (`max_echo_expected = far × 1/erl × 2.0`)
- ResidualEchoEstimator render-based path (`render_echo = far_psd × erl`)
- RES render-based ceiling (`min(erl·2.0, 1.0) × far_psd`)
- RES E8a linear-failed gate：`erl > 1.2 OR (using_render AND erle_factor<0.2)` → residual = max(residual, error_psd × 0.9)

→ **關鍵問題**：每次 EPC 觸發都把 ERL cap 到 0.3 (=−10dB ERL)。如果 EPC 反覆觸發或 stuck，ERL 永遠學不上去 → DT energy detector 永遠用 ERL=0.3 算，dt_from_energy 不準。

---

## 6. ResidualEchoEstimator (legacy mode, [aec.py:2398](AEC/python/aec.py#L2398))

兩 stage：
- **Stage 1** (linear blend)：
  - `confidence = compute_erle_confidence(filter_erle.erle, fb_erle.fb_erle)`
  - `erle_corrected = max(confidence·filter_erle + (1-confidence)·1.0, 0.5)`
  - `direct_est = echo_psd`
  - `erle_est = echo_psd / erle_corrected`
  - `nonlinear_floor = error_psd · coh2 · far_activity · (1 - dt_for_fs)`（only if far>1e-4）
  - `direct_est = max(direct_est, nonlinear_floor)`，erle_est 同
  - `residual = (1 - erle_factor) × direct + erle_factor × erle_est`
- **Stage 2** (render-based switch)：
  - `enr = far / mean(error_psd)`
  - `switching_threshold = 0.5 · clip(enr/(enr+1), 0.3, 0.7)` (ENR-adaptive: 0.15-0.35)
  - hysteresis 0.05，min hold 5 frames
  - `force_render = epc_active OR sat>0.5 OR not filter_converged`
  - `want_render = erle_factor < threshold OR force_render`
  - if using_render：`blend = clip(1 - erle_factor/threshold, 0, 1)`，`residual = (1-blend)·linear + blend·(far_psd·erl)`

---

## 7. RES post-residual processing (stages 3-4)

### 7a. Nonlinear / harmonic boost
- nonlinear (sat>0.3 sustained 5 frames): `residual *= (1.0 + sat)` (1.3-2.0×)
- harmonic (sat>0.05): HF (500-4000Hz) floor = LF mean × (0.1 + 0.4·sat)

### 7b. Physical ceilings (4 caps applied to `residual_echo_psd`)
1. `residual ≤ echo_psd × 2.0`
2. `residual ≤ error_psd`
3. `residual ≤ error_psd × dt_suppress` where `dt_suppress = clip(1 - dt_for_fs², 0.1, 1.0)`
4. (if far>1e-4) `residual ≤ far_psd × min(erl·2.0, 1.0)`

### 7c. Echo boost (post-converged)
- gate: `far>1e-4 AND erle_factor>0.3`
- `echo_boost = 1.0 + 0.5·coh2 if dt_for_fs<0.2 else 1.0`
- `residual *= echo_boost`

### 7d. E8a linear-failed fallback — **REMOVED in v3.7.1+v3.8.0+v3.8.1**
**Do NOT port to C.** Three structurally related branches were ablated:
- v3.7.1 PR-B: `using_render AND erle_factor<0.2` branch (was firing 35% of DT_mv frames at false-negative effective_dt)
- v3.8.0 ABL-1: error_based_floor `error_psd × far_conf × fallback_factor` in residual_echo_estimator
- v3.8.0 ABL-2: Y2 fallback `mic_psd × 0.5` (gated render_based + error_peak > 0.05)
- v3.8.1 ABL-4: ERL-based branch `erl > 1.2` (dead code, fire rate 0% verified)

All shared the same architectural mistake: using a signal that contains
near-end speech (`error_psd` / `mic_psd`) as an "echo proxy" floor → DT
near-end gets structurally over-suppressed. AEC3-aligned policy: only
`render_based_echo = far × ERL` floors the residual_echo estimate.

---

## 8. RES gain decision (ENR path)

```python
nearend_est = max(error_psd - residual, 0)               # raw
dt_per_bin = max(effective_dt_scalar, 1 - coh2)          # per-bin
if is_stationary_dt: dt_per_bin = max(dt_per_bin, _stat_dt_mask)
dt_shaped = dt_per_bin ** 1.1
nearend_est = max(raw_nearend × dt_shaped, noise_floor=mean(error_psd)·0.01)
min_ne_from_dt = error_psd × dt_shaped                   # FLOOR
nearend_est = max(nearend_est, min_ne_from_dt)
nearend_est = max(nearend_est, error_psd × 0.05)         # physical floor

enr = residual / nearend_est
```

### 8a. ENR threshold (per-bin via blend HF cut from bin 5)
- `enr_t_ne = (1-blend)·2.0 + blend·1.5` (NE-side)
- `enr_s_ne = (1-blend)·3.0 + blend·2.5`
- `enr_t_fs = (1-blend)·0.3·scale + blend·0.07·scale` (FS-side, scale=`enr_scale`=0.85 BALANCED)
- `enr_s_fs = (1-blend)·0.4·scale + blend·0.1·scale`
- DT relax (eff_dt > 0.4)：`enr_t_ne *= (1 + (eff_dt-0.4)/0.6 · 0.5)` max 1.5×
- 混合：`enr_t = ne_conf · enr_t_ne + (1-ne_conf) · enr_t_fs`
- `enr_s_safe = max(enr_s, enr_t + 0.2)` (min gate width)
- `gain = clip((enr_s - enr) / (enr_s - enr_t), 0, 1)`

### 8b. EMR (echo-to-masker ratio masking)
- `emr_transparent = 0.1` (寫死)
- `g_emr = clip(emr_transparent / (emr+1e-10), 0, 1)`
- `g = max(g, g_emr)` (echo<noise → 不壓)

### 8c. ne_g_floor (per-bin near-end protection)
- `ne_erle_gate = max(erle_factor, 0.3)`
- `ne_protection = (1-coh2) · ne_erle_gate · (1 - fs_confidence)` 其中 `fs_confidence = far_activity · (1-eff_dt)²`
- `ne_g_min_ceil = 10^(ne_protect_db/20)` (BALANCED -16dB → 0.158)
- `ne_g_floor = effective_g_min + (ne_g_min_ceil - effective_g_min) · ne_protection`
- `spectral_g_min = max(spectral_g_min, ne_g_floor)`

### 8d. EPC_DT cap
- `epc_dt = epc_active AND effective_dt > 0.35` → `g = min(g, 0.85)` (寫死 EPC_DT_GAIN_CAP)

### 8e. Cross-bin smoothing + HF cap
- 3-bin kernel [0.25, 0.5, 0.25]
- DC consistency (bins 0,1 取 bin 2)
- HF cap (bins > ~500Hz): if `eff_dt < 0.5 AND not is_stationary_dt` → cap to bin@500Hz value

### 8f. Divergence override
- `divergence > 0.3` → `g = min(g, 0.01 + 0.99·(1-div))`

### 8g. Temporal smoothing (鎖在 alpha_release_light = 0.5 - 0.2·dt_temporal)
- 一連串 attack/release 邏輯，rate-clamped by `max_drop_db_per_frame = 6` / `max_rise_db_per_frame = 6`

### 8h. Spectral floor
- `noise_floor_gain = clip(...)` 由 noise_psd 估計，min-stat
- `gain = max(gain, noise_floor_gain)` (final floor)

---

## 9. Output limiter
- `target_gain = near_peak / out_peak` if out > near
- attack 0.3 / release 0.8 EMA on limiter_gain

---

## 寫死/限制總表（按重要性排序）

### A. Cascading-critical（一改影響整片）
| Constant | Value | 位置 | 影響 |
|---|---|---|---|
| `inst_ERLE > 5dB × 10 frames` | 5dB / 10 | FilterConv L2266 | filter_converged 條件，gate 11+ 下游 |
| `_erl ≤ 0.3 on EPC fire` | 0.3 | aec.py L3517,3540 | ERL cap 卡死，DT energy detector / RES render 全受影響 |
| `_render_forced = 20f` | 20 (=epc_hangover) | aec.py L3516,3539 | RES 強制 render-based 20 幀 |
| `epc_hangover = 20f` | 20 | config.epc_hangover | EPC active 時長 |
| `dt_from_energy < 0.3` | 0.3 | ShadowCopyCtrl L2806 | shadow copy 全面封鎖門檻 |
| `mark_diverged() on EPC` | 4 sites | aec.py L3531,3549,3731,3759 | filter convergence 重置 → cascading |
| `EPC hangover tick gated on converged` | gate | aec.py L3747,3773 | **stuck bug 在這裡** |

### B. Threshold-critical
| Constant | Value | 位置 | 影響 |
|---|---|---|---|
| `epc_delta_threshold` | 0.3 | config | shadow_rise 觸發條件 |
| `epc_total_rise` | 1.5 | config | shadow_rise 觸發條件 |
| EPV ±6dB band | 0.25/4.0 | EPCDet L2666 | EPV 觸發 |
| `shadow_copy_threshold` | 0.65 | config | shadow_better 比例 |
| `shadow_dtd_offset` | 1.5 | config | dt_from_shadow 啟動點 |
| `shadow_dtd_advantage_scale` | 3.0 | config | dt_from_shadow 飽和點 |
| Stationary CV² | 0.02 | RenderActDet L2204 | stationary far 偵測 |
| RES `dt_per_bin shaping` | `**1.1` | aec.py L1644 | DT 形狀 (太陡會壓 NE 太多) |
| `min_gate_width` | 0.2 | aec.py L1690 | ENR gate FS-end 軟化 |
| `EPC_DT_GAIN_CAP` | 0.85 | aec.py L1738 | DT+EPC 強制壓制 |
| `divergence_gain` 觸發 | 0.3 | aec.py L1759 | filter 散逸時 hard cap |

### C. Hangover counter
| Counter | Value | 位置 |
|---|---|---|
| EPC hangover | 20 frames | config.epc_hangover |
| Shadow pause hangover | = epc_hangover (耦合) | ShadowCopyCtrl |
| EPC render-forced | = epc_hangover | aec.py L3516,3539 |
| Stat DT hangover | 80 frames (800ms) | aec.py L3454 |
| Filter conv counter | 10 frames | FilterConv L2270 |
| Shadow copy hysteresis | 3 frames | config |
| Shadow advantage streak | 10 frames | ShadowCopyCtrl L2755 |
| RES render min hold | 5 frames | ResidualEcho L2493 |
| RES nonlinear hangover | 5 frames | aec.py L1506 |
| dt_from_energy fast/slow EMA | 0.3/0.7 rise, 0.9/0.1 decay | DTAnalyzer L2371 |

### D. EMA / smoothing TC
| EMA | α | TC | 用途 |
|---|---|---|---|
| `_far_env_mean` (CV²) | 0.99 | ~1s | render activity stationary |
| `_dt_from_energy` rise/fall | 0.3/0.9 | 4f / 30f | energy DT smooth |
| `_dt_from_shadow` | 0.7→0.3 active, 0.95 decay | 5f / 100f | shadow DT smooth |
| `shadow_err_smooth` | 0.80 (`shadow_err_alpha`) | 5f | shadow err |
| `_copy_err_baseline` | 0.995 | 200f | shadow copy gate |
| `epv_gain_fast` | 0.98 | 50f | EPV |
| `epv_gain_slow` | 0.999 | 1000f | EPV |
| `_erle_window_*` | 0.999 | ~10s | erle_factor base |
| `_inst_erle_smooth` | 0.7→0.3 (att) | inst correction |
| `_divergence_indicator` | 0.9 active / 0.95 decay | 10f / 20f | divergence DTD |
| `coh2_smooth` rise/drop (post-conv) | 0.90/0.50 | 10f/2f | DT 保護 |
| ERL `_erl_estimate` | 0.99 unconv / 0.999 conv | 100f/1000f | echo path gain |
| RES gain temporal | 0.5 release light, varies | 主要 release rate |

### E. Pre-conv / startup_dt 路徑（多路徑都走「保守」分支）
| 路徑 | 條件 | 行為 |
|---|---|---|
| `force_render` | `not filter_converged` | RES 用 render-based echo |
| `_startup_dt_curr` | `eff_dt>0.35 AND far>1e-4 AND not converged` | 觸發 startup_dt |
| `_startup_dt_once` | `eff_dt>0.35 AND far>1e-4 AND not once_converged` | 觸發 once-startup_dt |
| `min_ne_from_dt scale` | `_startup_dt_once_cond AND scale != 1.0` | NE floor 縮放 |
| coh2 EMA rates | `not converged` → 0.80 rise (slower) | DT detection 鈍 |
| ERL α | `not converged` → 0.99 (faster but still slow) | ERL 學不快 |
| RES echo_boost | `erle_factor>0.3` (proxy) | post-conv 補強 |
| RES E8a fallback | `erl>1.2 OR (using_render AND erle_factor<0.2) AND eff_dt<0.2` | force residual=error*0.9 |

---

## 觀察 — Hot Spots（多點協調修改候選位置）

### Hot Spot 1: filter_converged → 11+ cascading
單一 boolean gate 11 個下游決策。當 movement-DT 導致 filter 永不收斂 (worst-10 顯示)：
- EPC hangover 卡死（已知 bug）
- ERL EMA α 用 0.99（不慢但永遠 capped 在 0.3 因為 EPC fire 不斷）
- RES `force_render` 永遠 True
- Coh2 EMA 用 unconverged rates
- ne_g_floor 用 erle_factor proxy（一定 < 0.3 → ne_erle_gate locked = 0.3）
- post-conv echo_boost 不啟用

→ 改方案：把 `filter_converged` 拆成更細粒度的 health signal：
  - `filter_has_useful_estimate` (= once_converged AND not actively diverging)
  - `filter_is_tracking_well` (= 連續 N frames 內 inst_ERLE 平均 > X)
  - `filter_in_recovery` (= EPC hangover OR mark_diverged 後 < N frames)

不同 consumer 用不同 signal。

### Hot Spot 2: ERL 0.3 cap 鎖死
EPV/shadow_rise/delay-shift 觸發時都做 `_erl = min(_erl, 0.3)`。若 EPC 反覆觸發或 stuck active → ERL 永遠在 0.3 (=−10dB)。但實際耦合可能 −20dB 甚至 −30dB。這時：
- `dt_from_energy = max(0, (mic - far·1/0.3·2.0)/mic) = max(0, (mic - 6.67·far)/mic)` 
  - 真實 mic ≈ 0.01·far (−20dB ERL) 時 dt = max(0, -666·far/mic) = 0 → **dt_from_energy 永遠不會啟動**
- RES render-based echo `= far·0.3`，遠超實際 echo → over-suppression
- RES E8a 觸發條件 `erl > 1.2` 永遠不會發生

→ 改方案：
  - ERL cap 應隨 EPC age 衰減：剛 fire = 0.3，5f 後逐步放開
  - 或 ERL 雙軌：`_erl_safe` (capped) for echo expectation；`_erl_learned` (繼續學) 用於 dt_from_energy 內部估計
  - 當 EPC fire 是 stuck 狀態 (hangover_count 不變) 應該識別並停止 cap

### Hot Spot 3: dt_from_energy 鎖死下游 shadow copy
`dt_from_energy < 0.3` 是 shadow copy 唯一 DT gate。若 dt_from_energy 永遠是 0（因 ERL cap），DT 真的發生時 copy 不會被攔住 → shadow（追到 NE 的）會 poison main。但目前 v2.8.1 表現尚 OK 因為 streak gate 也卡（10 frames shadow advantage 才 fire）。

→ 改方案：拿掉 dt_from_energy 唯一性，加多重訊號的 DT gate（OR/AND 邏輯）。Phase C1 實驗 coherence gate 失敗，但 *combined* 可能有效。

### Hot Spot 4: EPC fire side effects 太多
單次 EPV/shadow_rise 觸發同時做 7 件事：Q→Q_high, P_max=1.0(30f), P_floor_beta=1.0(30f), `_render_forced=20f`, `_erl ≤ 0.3`, `mark_diverged()`, dtd_coh × 0.3。

→ 改方案：拆成 fine-grained event。e.g. EPV 偵測到 +6dB gain（變大）vs −6dB（變小）行為應該不同；shadow_rise 觸發時 ERL 不一定該 cap。

### Hot Spot 5: RES startup_dt vs converged 路徑差異劇烈
- post-converged：echo_boost +0.5·coh2，erle_factor 帶動 over_sub_base=5+9·erle_factor → 14
- pre-converged：強制 render-based，over_sub 還是低（erle_factor 低），但 ne_g_floor 鎖在 0.3
- 兩個世界中間沒有過渡，filter 收斂瞬間行為跳變

→ 改方案：用 confidence weight 取代 binary converged flag。`erle_smoothness` (= post-conv ERLE std) 作為 confidence proxy。

### Hot Spot 6: dt_per_bin shaping `** 1.1`
`dt_shaped_per_bin = dt_per_bin ** 1.1`。前面註解寫過 `**2.0` 太陡踩 51% case。1.1 是試出來的妥協。但這個 shaping 直接 gate `min_ne_from_dt = error_psd · dt_shaped`，是 NE 保留的 floor。對 DT 場景影響 deg 巨大。

→ 改方案：dt_per_bin shaping 應該 **per-bin per-context** 動態。例如：DT-onset (rapid rise) 用 `**0.7` (擴張 NE)，DT-sustained 用 `**1.5` (壓縮 NE 讓 echo 過 gate)。

---

## 跟 AEC2 比的可能差距方向（待 step 2 數據驗證）

從 worst-10 已知：ours bimodal — 有些 case 大贏 (-1.387 echo)，有些大輸 (+1.292 echo)。整體 std=0.586。

**hypothesis**：差距集中在「filter 沒收斂或處於 stuck EPC」狀態的 case。AEC2 在這些 case 可能：
- 收斂判定條件較鬆（更容易進入 converged 模式）
- ERL 沒有 hard cap，可以學到更小（更準的 echo expectation）
- 無 cascading filter_converged dependency
- 不對 EPC 下這麼多重 side effect（更接近 just Q-boost）

**Step 2 要驗證**：
1. 對 300 個 DT case 做 per-case ranking（包含 static DT + movement DT）
2. 把 case 按 filter 收斂時間分桶（never / 0-10s / 10-20s / >20s）
3. 各桶 ours vs AEC2 echo+deg
4. 在 worst case 內統計 EPC fire 次數、stuck %、ERL 軌跡、main_paused、render-forced %
5. 找出「ours 最輸 AEC2」的 case 特徵
