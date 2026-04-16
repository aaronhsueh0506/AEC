# AEC v2.3.0 Changelog & Milestones

**日期**：2026-04-16
**耗時**：約 16 小時（單一 session，含 code review、實驗、debug、revert、收版）
**有效改動時間 vs revert 時間**：約 8h 有效 + 8h 嘗試後 revert（BUG-1+2 ~3h、D-3 guard 系列 ~1.5h、D7 ~1h、P4 over_sub ~0.5h、其他 ~2h）

---

## Milestones

### M1：Energy DT 三線驅動（~5h，含 revert 嘗試）
**問題**：inst_erle correction 把 `dt_indicator` 壓到 ≈0，5 條 DT 保護路徑只有 ENR relaxation 一條被驅動。
**解法**：Pre-filter energy DT signal（`mic_pwr > 4 × far_pwr`）+ shadow DTD 合併為 `effective_dt`，驅動 dt_per_bin / fs_confidence / HF cap / dt_temporal / ENR relaxation 全部 5 條線。
**debug 過程**（含失敗嘗試）：
- Energy DT 只接 ENR relaxation → 800-case DT deg +0.001（無效）→ ~1h
- Problem 4 over_sub 用 effective_dt → zero regression（over_sub 在 ENR 是 dead code）→ ~0.5h
- 分離 P1 vs P4 → 確認 P4 無效不是抵消 → ~0.5h
- 接齊三條線 → smoke DT deg +0.08 → ~0.5h
- Shadow DTD 接回 → smoke 2.350→2.407 → ~0.5h
- NE→FS hangover：far_active gate → smoke FS echo -0.043 → -0.003 → ~0.5h
- D-3 收斂 guard 硬歸零 → DT deg -0.033 → revert → ~0.5h
- D-3 erle_factor soft guard → DT deg -0.028 → revert → ~0.5h
- Dynamic ne_physical_floor → 不如 v4 → revert → ~0.5h
**最終成效**：balanced DT deg 2.237→2.298（+0.061）

### M2：Shadow DTD offset（~1.5h，含分佈分析）
**問題**：shadow_advantage 在 FS 高耦合 post-convergence 仍有 43/253 幀 > 0.3 的 false positive。
**debug 過程**（含分析時間）：
- 硬歸零 guard → DT deg -0.033 → revert → ~0.3h（已在 M1 嘗試過，再次確認）
- erle_factor soft guard → DT deg -0.028 → revert → ~0.3h
- 固定 0.3 soft guard → 介於兩者之間 → revert → ~0.2h
- 測量 5 FS + 6 DT case shadow_advantage 分佈 → 確認 offset 1.5 安全 → ~0.5h
- 套用 offset 1.5 → 800-case 驗證 → ~0.2h
**最終成效**：FS echo 3.485→3.498（+0.013），DT deg 微退 -0.011

### M3：Blindspot Fixes A1-A5 + G1-G2（~2h）
**A1 S_ee decay**：1 行 fix，FS echo +0.011（far silence 後 coh2 恢復加快）
**A2 delay reset 帶 ResFilter**：1 行，清除 stale PSD
**A4 EPC 後 ERL partial reset**：1 行，防止 echo path change 後 ERL 過高
**A5 GCC-PHAT PAR gate**：10 行，防止 DT onset 時 delay 誤估
**G1 EPC render-forced 5→20 幀**：1 行，EPC 後 filter 需要 200ms 重新收斂
**G2 render-based override**：5 行，epc_active / saturation 強制 render-based
**800-case 驗證**：FS echo 3.498→3.509（+0.011），DT deg 2.287→2.278（-0.009，G2 divergence 是原因）

### M4：G2 divergence 修復 + D7 測試（~1.5h，含 D7 revert）
**問題**：G2 的 `divergence > 0.5` 在 DT 期間誤觸發 → DT deg -0.017
**debug 過程**：
- G2 帶 divergence 跑 800-case → DT deg 2.278（退步來源確認）→ ~0.3h
- G2 移除 divergence + D7 4dB 一起跑 800-case → FS echo 3.485（退步）→ ~0.3h
- Revert D7 跑 smoke → FS echo 回到 3.654 → 確認 D7 是主因 → ~0.3h
- G2 移除 divergence 單獨保留 → 800-case 確認 → ~0.3h
**最終成效**：DT deg 2.278→2.282（回升），FS echo 3.504

### M5：B4 動態 ERL（~1.5h debug）
**問題**：`echo_path_gain = 0.01`（-40dB）低估真實 ERL 10-50×，225/500 幀用 render-based 幾乎無壓制
**debug 過程**：
- 初始 DT gate `raw_dt < 0.5` → pre-convergence 全部被 block → ERL 停在 0.1
- 放寬到 `raw_dt < 2.0` → 95% pass rate，ERL 0.1→0.67
**最終成效**：pre-convergence FS echo 壓制顯著改善

### M6：BUG-1+2 窗函數統一嘗試（~3h，最終 revert）
**debug 過程**：
- Rectangular error_spec 直接傳 → smoke FS -0.053 → ~0.3h
- 800-case 確認 FS -0.040 / DT -0.027 → ~0.5h（等跑 + 分析）
- 分析 error_psd ratio → 實際 0.944×（非 2.67×）→ 推翻原假設 → ~0.3h
- Floor 補償 0.05→0.02, 0.01→0.004 → 無效 → revert → ~0.3h
- enr_scale 0.85→0.65→0.5 → 無效 → revert → ~0.3h
- 實作 windowed spec（sqrt-Hann）→ smoke OK → ~0.3h
- 800-case windowed spec → FS -0.023 → revert → ~0.5h（等跑 + 分析）
- coh2 alpha 0.90→0.93 微調 → 和 windowed 一起退步 → revert → ~0.2h
**結論**：主要影響是 coherence 計算改變，不是能量偏移。mixed-window bias（sqrt-Hann near × rect echo）在 ERLE < 20dB 不可忽略。需獨立 retune session。

### M7：Code Cleanup（~1h）
- B1-B5：dead code 刪除（alpha_release_base、dt_temporal 重複、eer guard、ne_erle_gate、hasattr→bool）
- C1-C4：per-frame 常數移 init（stat_dt_mask、enr_blend、hf_cap_bin、harmonic bins）
- G3：shadow copy advantage streak gate（10 frames）
- G4：diagnostics 19 keys
- G5：render-based hold time（5 frames）

---

## Bug Tracker

| ID | 嚴重度 | 標題 | 發現方式 | 修復狀態 |
|---|---|---|---|---|
| BUG-A1 | HIGH | S_ee far-silence 不 decay，coh2 bias | code review | ✅ v2.3.0 |
| BUG-A2 | HIGH | delay reset 不帶 ResFilter，stale PSD | code review | ✅ v2.3.0 |
| BUG-B1 | MED | shadow filter 傳 fft_size 而非 block_size | code review | ✅ v2.3.0（不影響運行但語義錯誤）|
| BUG-B4 | HIGH | render-based echo_path_gain=0.01 低估 50× | code review + 數據驗證 | ✅ v2.3.0（動態 ERL）|
| BUG-G2 | MED | G2 divergence override DT 誤觸發 | 800-case 驗證 | ✅ v2.3.0（移除 divergence）|
| BUG-D7 | LOW | max_drop_db 6dB pumping | 聽感 | ❌ revert（FS echo 代價 -0.024）|
| BUG-12 | HIGH | coherence 窗函數不一致 | 理論分析 | ❌ 延後（需獨立 retune session）|

---

## 分數演進

| 版本 | FS echo | DT deg | 關鍵改動 |
|------|---------|--------|---------|
| v2.0.0 (erle_cap) | 3.534 | 2.237 | inst_erle cap=4 |
| v2.1.0 | 3.534 | 2.237 | AEC3 echo switching + P-floor |
| v4 (energy DT) | 3.485 | 2.298 | Energy DT + Shadow DTD 三線驅動 |
| B1234 | 3.498 | 2.287 | + shadow offset + delay reset + ERL |
| Review v1 | 3.509 | 2.278 | + A1/A2/A4/A5/G1/G2 |
| **v2.3.0 Final** | **3.504** | **2.282** | + G2 fix + G3/G5 + cleanup |

---

## 實驗記錄（測試過但不採用）

| 改動 | 結果 | 原因 |
|------|------|------|
| BUG-1+2 rectangular error_spec | FS -0.040 / DT -0.027 | alpha 不匹配 |
| BUG-1+2 windowed spec | FS -0.023 | mixed-window bias |
| D7 max_drop 4dB | FS -0.024 | onset suppression 不足 |
| G2 divergence override | DT -0.017 | DT 期間誤觸發 |
| D-3 convergence guard 硬歸零 | DT -0.033 | onset 保護被砍 |
| D-3 erle_factor soft guard | DT -0.028 | 仍太粗糙 |
| Dynamic ne_physical_floor | 不如 v4 | 三線驅動已覆蓋 |
| Energy DT 只接 ENR relaxation | DT +0.001 | 需要接齊全部線才有效 |
| over_sub 用 effective_dt | zero regression | over_sub 在 ENR 路徑是 dead code |
