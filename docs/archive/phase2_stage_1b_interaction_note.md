# Phase 2 Stage 1B: B-16 × shadow-as-NLMS interaction observation

Date: 2026-04-22. Branch `feature/phase2-shadow-as-nlms`. Non-urgent follow-up note.

## Observation

PZ7V FS#84 onset window (t=10–11 s, 1 s slice), full-file leak (21 s), from Stage 1B smoke test 9-case × 4-combo matrix:

| Flag combo | Onset window (dB) | Δ_onset vs (0,0) | Full-file (dB) | Δ_full vs (0,0) |
|---|---:|---:|---:|---:|
| (0,0) baseline | −13.796 | — | −17.062 | — |
| (1,0) B-16 only | −20.287 | **−6.49** | −17.410 | −0.35 |
| (0,1) Phase 2 only | −13.796 | 0.00 | **−24.427** | **−7.37** |
| (1,1) stacked | −18.046 | **−4.25** | **−26.217** | **−9.16** |

**Key finding**: (1,1) onset = −18.05 dB is **2.24 dB worse than (1,0)** = −20.29 dB. Two layers 在 onset window 有 partial negative interaction。Full-file 方向相反, (1,1) −26.22 dB 是 4-combo 最佳 (vs (0,1) −24.43, (1,0) −17.41)。

## Hypothesis

B-16 veto ([aec.py:3405](../python/aec.py#L3405)) 觸發時把 `raw_dt` 替換為 `_raw_dt_ema_slow` (~0.05)。`_dt_from_shadow` ([aec.py:3106-3110](../python/aec.py#L3106-L3110)) 作為 DTD 的一部分, 其估算 pipeline 對 raw_dt 的精確性敏感。

假設鏈:
1. B-16 flag=1 → raw_dt 於 onset 2-frame jump 期間被替換為 0.05
2. `_dt_from_shadow` 估算下游 signal 受影響 (可能 under-estimate DT)
3. shadow→main copy gate (L3135+) 在 `error_is_normal` guard 與 `copy_allowed` guard 下依 err ratio 觸發; copy gate 不直接用 `_dt_from_shadow`, 但 EPC large 的 `dt_signal < 0.3` guard 用
4. (1,1) 下 EPC 更易 fire (或相反 — Stage 1B telemetry 只測了 (0,1), 未測 (1,1))
5. 若 EPC 於 (1,1) fire 但 (0,1) 不 fire, 導致不同 re-learning trajectory → 不同 onset-window leak

此為假說, 未驗證。

## Impact assessment

- **Full-file (21 s) 上 (1,1) 仍是 4-combo 最佳** (−9.16 dB 改善), stacked net benefit 明確。
- **Onset window 的 −2.24 dB interaction** 是 subtle issue, Phase 2 不 block, Phase 3 再議。
- 若 AECMOS 於 Stage 1D 反映 onset 區域權重高於 full-file, 這個 interaction 會被放大, 需 re-evaluate。

## Follow-up (Stage 1D 驗證項)

Stage 1D 800-case 必看:

1. **Movement subset (1,1) vs (1,0) onset 區域退化幅度**:
   - 若 movement subset 明顯 (e.g. mean Δ > +0.3 dB onset regression), interaction 對 mobile 場景關鍵
   - 若 movement subset 不明顯, interaction 來源可能是 static onset 特有

2. **Non-movement subset 是否也有同 pattern**:
   - Non-movement 800-case 主力是 FS onset cases (Cat-A 類 PZ7V)
   - 若全域可見, 非 PZ7V-specific, 需 orthogonalization (下面)

3. **Cat-A 13 cases mean onset window 的 (1,1) vs (1,0)**:
   - Stage 1D wav-level secondary metric (spec v3 §7.2)
   - 比 AECMOS 更直接量 onset-specific interaction

## Potential orthogonalization (Phase 3 material)

若 Stage 1D 確認 (1,1) onset interaction 是全域 pattern, Phase 3 candidate:

- **Option A — B-16 veto 期間暫停 copy gate evaluation**: 於 `_diag_b16_veto_active` 為 True 的 frame, 跳過 [aec.py:3135-3155](../python/aec.py#L3135-L3155) 的 copy gate 判斷。代價: copy gate 少 2–5 frames fire 機會 (B-16 veto 典型持續 ≤ 5 frames)。
- **Option B — _dt_from_shadow 不用 B-16-modified raw_dt**: 另開一個 `raw_dt_unmodified` tap 給 shadow DTD 用。代價: 重複計算 branch, 程式碼複雜度上升。
- **Option C — 不 orthogonalize, 把 B-16 default ON 且 shadow_copy_hysteresis 調短**: 接受 (1,1) 作 production combo, 用 hysteresis 補償。代價: B-16 從 defensive 變 always-on, 背離 Phase 1 merge rationale。

Phase 3 決策前需 Stage 1D 數據 + user judgment。

## References

- [docs/spec_phase2_shadow_as_nlms.md](spec_phase2_shadow_as_nlms.md) §9.1 v3 dual-window
- [docs/spec_b16_raw_dt_jump_veto.md](spec_b16_raw_dt_jump_veto.md) §3.3.1 raw_dt EMA semantics
- Stage 1B smoke data: `/tmp/p2_b16_{0,1}_p2_{0,1}.json`
- Stage 1B EPC telemetry PZ7V (0,1): `/tmp/p2_telemetry_01_pz7v.log`
