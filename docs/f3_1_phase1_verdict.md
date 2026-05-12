# F3.1 Phase 1 verdict — per-bin mic-energy excess evidence (2026-05-12)

## Scope

`feature/f3-1-mic-excess-evidence` 從 main `bd6ba03` 切出。實作 `AecConfig.use_mic_excess_evidence`（default OFF）gating `ResFilter._stage_gain_compute` 的新 `dt_per_bin` 分支：

```
dt_per_bin[k] = clip(max(error_psd[k] − far_lw[k]·ERL_est, 0) / error_psd[k], 0, 1)
```

當 `filter_converged AND _long_window_n_updates > 0` 同時成立才 fire；否則 fallback 到 legacy `(1 − coh2)`。鏡像同步到 `python/res_refactored/gain_computer.py`。

## 800-case j4 BALANCED bench

- Baseline (flag-OFF on F3.1 branch): `/tmp/f3_1_baseline/`, 800/800 cases 0 errors
- F3.1 (flag-ON): `/tmp/f3_1_on/`, 800/800 cases 0 errors
- AECMOS scoring: `/tmp/f3_1_baseline_scores/` + `/tmp/f3_1_on_scores/`

### Bucket means

| Bucket | n | base.echo | f3.1.echo | Δecho | Δdeg | verdict |
|---|---:|---:|---:|---:|---:|---|
| FS_static   | 169 | 3.542 | 3.541 | **−0.001** | +0.000 | ok |
| FS_movement | 131 | 3.586 | 3.580 | **−0.006** | +0.000 | ok |
| DT_static   | 186 | 4.126 | 4.127 | +0.000 | **+0.001** | ok |
| DT_movement | 114 | 4.100 | 4.101 | +0.002 | **−0.003** | ok |
| NE          | 200 | 4.998 | 4.998 | +0.000 | +0.000 | ok |

**All 5 buckets within acceptance bars** (`bench_aecmos.py:218`：FS Δecho ≥ −0.02、NE Δdeg ≥ −0.01)。對照 P50 nearend_protect_dt 失敗的 −1.328 mean，F3.1 mean 落在 −0.006 = **220 倍 less bad**。

### Histogram of |Δ| > 0.05

| Bucket | echo regress | echo gain | deg regress | deg gain |
|---|---:|---:|---:|---:|
| FS_static   | 4 | 3 | 0 | 0 |
| **FS_movement** | **5** | 1 | 0 | 0 |
| DT_static   | 2 | 1 | 3 | 4 |
| DT_movement | 0 | 2 | 4 | 1 |
| NE          | 0 | 0 | 0 | 0 |

### Top movers (per case)

**FS_movement worst Δecho（最值得注意）：**
- `Lsa5WpwTpUeb7C9dc9RXuQ_farend_singletalk_with_movement` Δecho **−0.301** (3.880 → 3.579)
- `s0oJqM6Y1UCHSVmHmgsx4Q_farend_singletalk_with_movement` Δecho **−0.213**
- `wr54weKzNkOcZ07hB04kzA_farend_singletalk_with_movement` Δecho −0.167
- `pG9Bikvr40Ct1kUtch95kw_farend_singletalk_with_movement` Δecho −0.140

**DT_static top Δdeg gainers（NE 保護目標）：**
- `tl5UFRCXZkyL6EoWVl09xA_doubletalk` Δdeg **+0.178** (with Δecho +0.075)
- `i2BU43nmM00qbI1MKr2njQ_doubletalk` Δdeg **+0.174** (with Δecho −0.063)
- `Wv6yp6N1L0WqQ6ZLn6nD8g_doubletalk` Δdeg +0.063

**Cohort tail（P52 outlier）：**
- `qNvSMyUSXUyrDGpOw7s6qg_farend_singletalk` Δecho=+0.000 Δdeg=+0.000 → F3.1 沒 fire（filter never converges → gate fail → legacy fallback）。設計成功隔離 cohort tail。

**NE bucket：** 全 200 case Δ=0.000。F3.1 在 NE 永不 fire（far silent → `_long_window_n_updates` 0 → gate fail）。設計符合預期。

## 機制解讀

F3.1 確實有 fire（NE bucket 全 0 vs 其他 bucket 有非零 Δ 證明 gate 條件區分有效）。但效果分裂：

1. **DT_static**: 2-way 移動，**正向 4 例 vs 負向 3 例**，最大 gainer +0.178 deg。F3.1 目標方向（DT 中保 NE）有出現，只是分布稀疏（4/186 case ≥ 0.05 gain）。
2. **FS_movement**: 5 例 echo regress vs 1 例 echo gain 的 **不對稱 tail**。Movement 期間 EPC 觸發頻繁，`erl_estimate` EMA（α=0.999, TC≈10s）跟不上 path change，舊 ERL 餵 metric → mis-attribute → 過度保 NE → echo 漏出。這是 plan 已預警的 risk（§3.1 ERL_est lag）。
3. **NE / cohort tail**: gate 設計把這兩個區域完全擋下，安全。

### 為何 bucket mean delta 這麼小

兩個來源：
- F3.1 替換的是 `dt_per_bin` 一個 axis，但 gain_compute 還有 scalar `effective_dt`（aec.py:2399）控 ENR relax、HF cap gate、spectral floor lift（aec.py:1972, 2031, 2179）— **scalar gate 沒動，per-bin discrimination 出不來**。F3.2（plan §3.7）就是把 scalar effective_dt 改 per-bin 解這個 leverage。
- 在 FS 後 cancel 的 frame，`fs_confidence` 接近 1，`ne_g_floor` 公式 `(1 − coh2) × erle_gate × (1 − fs_confidence)` 已被 fs_confidence 壓到 ≈ 0 — coh2 saturation bug 在這裡是 cosmetic 的，F3.1 改了沒影響。Bug 真正咬到的是 transitional / pre-converged 場景，但那裡 `filter_converged` gate 又把 F3.1 擋下。

結論：**F3.1 single-axis fix 在 scalar gate 仍然控場的 architecture 下，leverage 不足**。Plan 提的「F3.1 + F3.2 同時做」才是完整解。

## Verdict: **GREEN-PASS (slim)** — proceed to F3.2

- ✅ 所有 bucket mean 過 acceptance bars
- ✅ 沒有重蹈 P50 / P58 失敗模式（P50 mean −1.328，F3.1 mean −0.006 = 220× less bad）
- ✅ Cohort tail 由 gate 設計隔離
- ✅ DT_static 有方向正確的 NE preservation signal（+0.178 max gainer）
- ⚠️ FS_movement −0.301 worst case 值得後續 listen check + 機制 audit
- ⚠️ 整體效應太弱，**F3.1 alone 不是 production candidate**

## 建議 next step：F3.2 — effective_dt 改 per-bin

F3.2 把 `effective_dt`（aec.py:2399 scalar）替換成 per-bin vector：
```
effective_dt[k] = combine(scalar_dt, excess_per_bin[k] normalized)
```
HF cap gate（aec.py:2179）、spectral floor lift（aec.py:1972）、ENR relax（aec.py:2031）改用 per-bin gate。同樣 default OFF flag。預期效果：把 F3.1 per-bin 已有的 NE evidence 接到下游 scalar-gated 決策，**leverage 才解開**。

## Phase 1 結束前的 follow-up（不阻 F3.2）

1. **Listen check FS_movement Lsa5Wpw...** 確認 −0.301 echo MOS drop 在 audibility 上是否真的 audible，或是 AECMOS metric 過敏。
2. **記 down 機制 hypothesis**：F3.1 在 path change 後幾秒內 erl_estimate stale → 過度 NE protection。對應 Q2 fix F2.3（Yang 2017 R-reset + erl reset on EPC）可以順便解這個。

## Files / artefacts

- Code: `python/aec.py` + `python/res_refactored/gain_computer.py`（commits 0eb318d / 721891d / 765d340 / 8560c42 / cec1c1f）
- Runner: `tools/research/f3_1_bench_runner.py`
- Bench output: `/tmp/f3_1_baseline/` + `/tmp/f3_1_on/` (800 + 800 WAVs)
- Scores: `/tmp/f3_1_baseline_scores/{scores.json, result.md}` + `/tmp/f3_1_on_scores/{scores.json, result.md}`
- Unit tests: `python/test_f3_1_mic_excess.py`（6/6 pass）
- P52 regression: 13/13 pass
- Logs: `/tmp/f3_1_baseline_v2.log`, `/tmp/f3_1_on_v2.log`, `/tmp/f3_1_baseline_score.log`, `/tmp/f3_1_on_score.log`
