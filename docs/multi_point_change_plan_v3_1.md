# v3.1 多點協調修改計畫（DT echo gap 改善）

## ⚠️ POST-MORTEM (Phase 1 跑完後更新)

Phase 1（ERLE 5→3 + EPC unstuck + ERL cap decay-back + ERL outlier）跑 800-case：
- FS −0.070（FS_movement −0.127 嚴重 regress）
- DT 幾乎不動（−0.011 echo / −0.006 deg）
- NE 不變

**結論**：原計畫的「filter convergence + EPC stuck + ERL cap」軸**不是 DT echo gap 的主軸**，且**EPC stuck 是 FS_movement 的 fail-safe**（render-forced 鎖死等於用穩定 ERL × far 估計 echo，filter 可動但 RES 不信任 filter）。

## 🎯 真正 root cause（重新分析後）

[aec.py:1525](AEC/python/aec.py#L1525) `residual_echo_psd ≤ self.echo_psd × 2.0` cap：

| filter 狀態 | echo_psd | render-based 估計 | cap 結果 | 後果 |
|---|---|---|---|---|
| 已收斂 | 可信 | 不啟用或補強 | 合理 | 正確 |
| 未收斂 | ≈ 0 | far × erl（合理） | **min(合理, ≈0) ≈ 0** | **echo 完全不抑制** |

**worst-15 static-DT 全部 render_based_pct=98%**（系統認為要用 render）但 cap 把它砍掉。Δecho +1.5~+1.8（漏 echo）+ Δdeg −2.0~−2.4（NE 過度保留）就是這個 bug 的指紋。

**修法**（v3.1.0-rc2）：cap 在 `_using_render_based=True` 時跳過：
```python
if not self._residual_est.using_render_based:
    residual_echo_psd = np.minimum(residual_echo_psd, self.echo_psd * 2.0)
```

單行修改，與 Phase 1 三個 axes 都不衝突。預期 DT echo 大幅改善，FS 幾乎不變（FS filter 通常有學到，cap 仍套用），NE 不變。

---


## Context（Step 1 + Step 2 整合）

300-case DT 評測 vs AEC2 baseline：

| 觀察 | 數據 |
|---|---|
| 永不收斂率 | 50% (150/300) |
| ours 贏 echo | 30% |
| ours 贏 deg | 59% |
| Δecho 平均 | +0.252（ours 輸） |
| Δdeg 平均 | −0.184（ours 贏） |
| render-based 占比 | 95–99% across all buckets |
| main_paused 占比 | 0% (shadow→main pause 失效) |
| Late-converged echo gap | 1-5s +0.356 / 5-15s +0.699 / 15s+ +1.332 |
| ERL 異常 (erl_max>2) | 多 case 學到 +6dB（離譜） |

**根因綜合**（從 Step 1 signal flow 文件 + Step 2 數據交叉驗證）：

> **`filter_converged` 是 11+ 個下游決策的 single boolean gate**，且 5dB×10frames 條件對 DT 場景太嚴。50% 的 DT case 永不滿足 → 整套 detector / RES 卡在「pre-conv defensive」模式。pre-conv 模式包含 `force_render=98%`、`ne_g_floor` 鎖在 -16dB 等保守設定。**defensive 模式抑制 NE 太強**（贏 deg）但**對真實 echo 抑制不夠**（輸 echo），因為它假設 filter 不可信，過度倚賴 render-based echo estimate（受 ERL 0.3 cap 影響估計失真）。

> AEC2 不需要這個強收斂判定就能出好聲音 — 它的 echo suppression 機制獨立於 filter 收斂狀態。

---

## 修改集 V3.1

修改原則：**同時動 5-7 個位置形成 coherent shift**。單點動只會踩到別的限制（已經被 11 個 ablation 證實）。

### 修改 1：放寬 convergence 條件（破除「永不收斂」）

[FilterConvergenceAnalyzer](AEC/python/aec.py#L2260)
- `CONV_ERLE_DB = 5.0 → 3.0`（5dB→3dB）
- `CONV_FRAMES = 10`（保持）
- 加 secondary path：`once_converged = True if (sustained inst_ERLE > 1.5dB for 30 frames)` 作為更弱的 "filter useful" 訊號

理由：50% DT case 在 5dB×10 上失敗。3dB×10 在 worst-15 cases 中觀察到 inst_ERLE 經常達到 2-4dB 但達不到 5dB。3dB 是合理的「有用」門檻。secondary path 給更慢但更鬆的二級指示。

預期影響：
- never-converge 比例從 50% 降到估計 20-30%
- 觸發後 11 個下游決策（包含 force_render）會切到 post-conv 路徑
- 風險：filter quality 較差時被當「已收斂」處理 → echo_psd 不可信 → ENR 計算錯

### 修改 2：拆分 filter_converged 為多 signal

[AecState](AEC/python/aec.py#L2526) 新增 properties：
```python
filter_useful = once_converged AND not actively_diverging  # for RES echo_psd trust
filter_tracking = filter_converged AND erle_factor > 0.3   # for tight gates
filter_in_recovery = epc_hangover_count > 0 OR (mark_diverged 後 < 30f)
```

下游消費端對應遷移：
- `force_render`：用 `not filter_useful` (取代 not filter_converged)
- `ERL EMA α`：用 `filter_tracking` 切 0.999（穩定）vs 0.99（學習）
- `coh2 EMA rates`：用 `filter_useful` 切 post-conv vs pre-conv
- `startup_dt`：用 `not filter_useful`
- post-conv `echo_boost`：用 `filter_tracking`

理由：拆細後不同 consumer 用適合的 health signal。force_render 不再被 EPC stuck 死綁。

### 修改 3：EPC stuck bug fix + ERL cap decay-back

[aec.py L3773](AEC/python/aec.py#L3773) `tick_hangover()` 移出 `(shadow_filter and filter_converged)` gate：
```python
# Tick every frame (was only inside converged gate → stuck bug)
self._epc_det.tick_hangover()
```

[aec.py L3517,3540](AEC/python/aec.py#L3517) ERL cap 改成 decay 回放：
```python
# OLD: self._erl_estimate = min(self._erl_estimate, 0.3)
# NEW: 加 _erl_cap_remaining 計數器
self._erl_cap_value = 0.3
self._erl_cap_frames = 30  # cap 30 frames then decay
# 主迴圈每幀套用 erl_cap dynamic：
if self._erl_cap_frames > 0:
    erl_dynamic_cap = self._erl_cap_value + (1.0 - self._erl_cap_value) * (1 - self._erl_cap_frames/30)
    self._erl_estimate = min(self._erl_estimate, erl_dynamic_cap)
    self._erl_cap_frames -= 1
```

理由：v2.8.1 cap 0.3 永久（直到下次 EPC 才更新），導致 ERL learning 卡在 0.3。Decay-back 讓 ERL 在 EPC 後 30 frames 內逐步回到自由學習。

### 修改 4：`_render_forced` 縮短 + 用 fade-out

[aec.py L3516,3539](AEC/python/aec.py#L3516)
- `_epc_render_forced_remaining = 20 → 8`（200ms → 80ms）
- ResidualEchoEstimator 加 `_render_forced_blend`：剩 8f 全 render，剩 5f blend 50%, 剩 0 全 linear

理由：EPC fire 後 200ms 強制 render-based 太久。filter Q-boost 後快速恢復，render-forced 應該更短。

### 修改 5：放寬 shadow copy gate / 重新激活 main_paused

[ShadowCopyController](AEC/python/aec.py#L2739)
- DT-safe gate 改成 OR：`dt_safe = (dt_from_energy < 0.3) OR (dt_from_coherence < 0.4 AND erle_factor > 0.2)`
- streak 條件 `≥10 → ≥7`
- baseline EMA TC 從 200f 縮到 50f（讓 DT 場景下 baseline 更新更快）

理由：300-case 顯示 main_paused 在所有 DT bucket 都是 0%，shadow filter 的訊號完全沒被用上。

### 修改 6：ERL outlier 防護

[aec.py L3501-3502](AEC/python/aec.py#L3501)
```python
# OLD: if raw_dt_ratio < 2.0:
# NEW: 加 ERL 輸入 clip
inst_erl = clip(mic_pwr/far_pwr, 0.001, 10.0)
if raw_dt_ratio < 2.0 AND inst_erl < 1.5:  # extra: 不學 ERL > 1.5
    self._erl_estimate = α·_erl + (1-α)·inst_erl
```

理由：worst-15 看到 erl_max=3.6, 1.6, 1.5 等異常值。ERL > 1.0 物理上不合理（echo 不可能比 mic 大），> 1.5 是 NE 主導的偽訊號。應該擋掉。

### 修改 7：dt_per_bin shaping 動態化

[aec.py L1644](AEC/python/aec.py#L1644)
```python
# OLD: dt_shaped_per_bin = dt_per_bin ** 1.1
# NEW:
if self._epc_det.active OR not aec_state.filter_useful:
    # defensive: gentle shaping (preserves NE) ... already what 1.1 does
    dt_shaped_per_bin = dt_per_bin ** 1.1
elif erle_factor > 0.4 AND coh2_mean > 0.5:
    # high confidence: aggressive shaping (suppresses echo)
    dt_shaped_per_bin = dt_per_bin ** 1.5
else:
    dt_shaped_per_bin = dt_per_bin ** 1.2
```

理由：worst case 中 ours 比 aec2 deg 更好但 echo 差很多。在 high-confidence 區段允許更陡 shaping → 換 deg 拿 echo。

---

## 預期效果（粗估）

| 修改 | 預期 Δecho | 預期 Δdeg | 風險 |
|---|---|---|---|
| 1: ERLE 5→3 dB | +0.10 | -0.05 | filter quality 較差時 echo_psd 不可信 |
| 2: split filter signals | +0.05 | 中性 | 大量 wiring 改動 |
| 3: EPC unstuck + ERL decay | +0.05 | -0.10 | EPC 行為改變影響 cascading |
| 4: render-forced 短 | +0.05 | -0.05 | EPC 後 echo leak 風險 |
| 5: shadow copy 放寬 | +0.02 | -0.02 | shadow 污染 main 風險 |
| 6: ERL outlier 防護 | +0.05 | 中性 | 真實高 ERL 場景被擋 |
| 7: dt_per_bin 動態 | +0.10 | -0.10 | 高信心場景過度抑制 |
| **合計（樂觀）** | **+0.30~0.40** | **−0.30** | 多點交互效果未知 |

→ 目標：把 Δecho mean 從 +0.252 → 接近 0（即跟 AEC2 持平 echo），同時 Δdeg 從 −0.184 → −0.05（仍小幅領先 AEC2 deg）。

---

## 實施順序與驗證

**Phase 1**（最高 leverage，先做）：
- 修改 1（ERLE 5→3 dB）
- 修改 3（EPC unstuck + ERL decay-back）
- 修改 6（ERL outlier 防護）

**測 5-case smoke 確認 parity 不一致符合預期**（會破 parity，但破壞應該只在 DT 段）。
**跑 800-case AECMOS** 看主指標。如果 Δecho 改善 ≥ +0.10 且 Δdeg 退步 < −0.10，繼續 Phase 2。

**Phase 2**：
- 修改 4（render-forced 短）
- 修改 7（dt_per_bin 動態）

**Phase 3**：
- 修改 2（split filter signals — wiring 改動最大，最後做）
- 修改 5（shadow copy 放寬）

每階段 commit + 800-case 驗證。**整體目標：v3.1.0 = filter health 多訊號 + EPC unstuck + ERL 動態 + RES 緩條件**。

---

## 風險與 fallback

1. **若 Phase 1 echo 沒改善但 deg 退步**：表示 ERLE 5→3 讓不可信的 filter 被當收斂，下游用了垃圾 echo_psd → 退回 5dB 改試 4dB 或加 secondary "ERLE std" 條件
2. **若 ERL outlier 防護把好 case 的 ERL 擋住**：放寬 to `< 2.0`
3. **若 dt_per_bin 動態化過度抑制 NE**：減小 1.5 → 1.3
4. **若全部負向**：退回 v3.0.2 (aec.py reset)，重新分析

每個修改都有 git commit，可單獨 revert。
