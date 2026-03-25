# AEC v1.19.x 改善紀錄 — Echo 抑制 + 診斷 + Preset 修正

## 概述

本系列版本針對實際音檔測試發現的多個問題進行修正：
1. 啟動時 echo 洩漏 + 近端被壓（v1.19.0）
2. 新增 per-frame 診斷輸出（v1.19.1）
3. Warmup 靜音浪費 + mu 恢復太慢（v1.19.2）
4. CLI preset 覆蓋 bug + mu_ratio 靜音損壞（v1.19.3）
5. 收斂偵測放寬 + 預收斂近端保護（v1.19.4）
6. Preset ne_protect_db 語義反轉 + enr_scale 修正（v1.19.5）

---

## v1.19.0: Fix echo leakage + near-end suppression (c58b63f)

### 問題

- Far-end echo 在輸出中仍然存在
- 開頭第一句 near-end 被壓到

### 修正

| 修改 | 說明 |
|------|------|
| Warmup DT-aware | Warmup 期間尊重 `_simple_mu_ratio`，DT 時不強制高 mu |
| RES near-end gate floor | 預收斂 `ne_erle_gate = max(erle_factor, 0.2)` |
| Pre-convergence mu_min | 0.5 → 0.3，改善 DT 保護 |
| Echo estimate boost | 預收斂時用 `far_psd * 0.3` 作為 echo 估計下限 |

---

## v1.19.1: Per-frame diagnostic output (e941ede, f978a1a)

### 功能

新增 `--diag` CLI flag，每 0.5s 輸出一行診斷資訊：

```
[ 2.0s] ERLE= 4.9dB mu=0.89 far_act=1.00 g_mean=-17.8dB g_min=-19.8dB eff_gmin=-34.4dB conv=Y div=0.46
```

### 追蹤指標

| 指標 | 說明 |
|------|------|
| ERLE | 瞬時 ERLE (dB) |
| mu | mu_scale（步長縮放） |
| far_act | Far-end activity (0~1) |
| g_mean / g_min | RES gain 平均/最小值 (dB) |
| eff_gmin | 有效 g_min floor (dB)，含 far_activity 調整 |
| conv | 收斂狀態 (Y/N) |
| div | 發散指標 (0~1) |

### Bug fix

- `args` 在 `process_wav_files()` 中未定義 → 改為 `diag: bool` 參數傳入

---

## v1.19.2: Fix warmup wasted on silence (8574304)

### 問題

100 frames warmup 在 1.5s 的 far-end 靜音期就用完了，far-end 真正開始時已無 warmup 加速。

### 修正

- 新增 `_warmup_far_active` flag，warmup 只在 far-end 有信號時才計數
- `_update_simple_mu_ratio()` 新增靜音→活躍轉換時 quick reset（ratio < 0.1 → 0.8）

---

## v1.19.3: Fix CLI preset override + silence mu_ratio (12f3684, b3499e0)

### 問題 1: CLI preset 被覆蓋

`common_kw` 無條件包含 `res_g_min_db=args.res_g_min`（預設 -20.0），
`from_preset()` 的 `defaults.update(kwargs)` 讓 CLI 預設值覆蓋 preset 設定。

用 `--preset balanced` 時，`res_g_min_db` 被覆蓋：-35 → -20dB（少了 15dB 抑制深度）。

**修正**: 只在使用者明確指定 `--res-g-min` 且值不等於預設時才傳入。

### 問題 2: mu_ratio 靜音損壞

靜音期 `_simple_mu_ratio` 被壓低到 < 0.2，warmup DT 保護誤觸發。

**修正**: 靜音期（far + error 都 < 1e-6）不更新 ratio，保持初始值。

---

## v1.19.4: 收斂偵測放寬 + 預收斂近端保護 (8e39c4a)

### 問題

Filter 始終不收斂：ERLE 最高 5.4dB（門檻 6dB），DT 時 ERLE 掉到 -18dB 重置 counter。

### 修正

| 修改 | 舊值 | 新值 |
|------|------|------|
| ERLE 門檻 | 6 dB | **4 dB** |
| 連續 frames | 10 | **6** |
| DT 處理 | 重置 counter | **跳過不更新**（保留 counter 進度） |
| erle_factor 斜率 | (erle-3)/15 | **(erle-2)/8**（ERLE=5dB 時 0.13→0.375） |
| 預收斂 ne_erle_gate | 0.2 | **0.3** |

DT 偵測：`_simple_mu_ratio > 0.5` 視為 single-talk，才更新 counter。

---

## v1.19.5: 修復 preset ne_protect_db 語義反轉 (56041cc)

### 問題

AGGRESSIVE/MAXIMUM preset 比 BALANCED **壓得更少**，近端也被傷到：
- 2.5s g_mean: BALANCED=-15.9dB, AGGRESSIVE=-8.9dB, MAXIMUM=-7.0dB（完全反了）

### 根因

`ne_g_min_ceil = 10^(ne_protect_db/20)` 是近端保護能把 gain floor 拉高到的上限。
**ne_protect_db 越接近 0 = ceiling 越高 = 保護越強 = 抑制越少。**

| Preset | ne_protect_db (舊) | ceiling (舊) | ne_protect_db (新) | ceiling (新) |
|--------|-------------------|-------------|-------------------|-------------|
| BALANCED | -12 | 0.251 | -12 (不變) | 0.251 |
| AGGRESSIVE | **-3** | **0.708** | **-20** | **0.100** |
| MAXIMUM | **-1** | **0.891** | **-35** | **0.018** |

同時 `enr_scale` 太低導致近端 bin 被誤判為 echo：

| Preset | enr_scale (舊) | enr_scale (新) |
|--------|---------------|---------------|
| BALANCED | 1.0 | 1.0 (不變) |
| AGGRESSIVE | **0.3** | **0.7** |
| MAXIMUM | **0.15** | **0.5** |

### Preset 參數對照表（修正後）

| 參數 | BALANCED | AGGRESSIVE | MAXIMUM |
|------|----------|------------|---------|
| g_min_db | -35 | -60 | -72 |
| ne_protect_db | -12 | -20 | -35 |
| enr_scale | 1.0 | 0.7 | 0.5 |
| over_sub_base | 2.5 | 6.0 | 10.0 |
| dt_reduction | 3.5 | 1.0 | 0.0 |
| spectral_floor_db | -25 | -40 | -50 |
| reverb_decay | 0.5 | 0.7 | 0.8 |
| reverb_gain | 0.5 | 2.0 | 3.0 |
| alpha_echo_psd | 0.5 | 0.3 | 0.2 |
| alpha_error_psd | 0.6 | 0.4 | 0.3 |
| shadow_mu_min | 0.5 | 0.7 | 0.9 |
| warmup_frames | 100 | 150 | 200 |

### Preset 設計哲學

- **BALANCED**: 中等抑制 + 保護近端 → 通話品質優先
- **AGGRESSIVE**: 深抑制 + 少保護 → echo 更乾淨，近端可接受的損傷
- **MAXIMUM**: 最大抑制 + 最少保護 → echo 盡可能消除，允許明顯近端損傷

三個 preset 的區別主要在**抑制深度**（g_min）和**近端保護程度**（ne_protect_db），
而非偵測靈敏度（enr_scale 差異適度）。
