# Session handoff — 2026-04-21

Status snapshot for next Claude Code session.

## Section 1: 本 session 重大發現

**PZ7V FS#84 +15 dB leak 元兇定位完成** — `raw_dt` formula 有 latent
delay-mismatch bug，**v2.8.0 之前就存在**。

- `raw_dt = 1 - far_pwr/(mic_pwr + far_pwr)` 比錯時間窗：mic[t] 含 echo
  from far[t-delay]，但跟 far[t] 比能量。echo onset 瞬間 mic_pwr 暴增
  到 far_pwr 的 38x → raw_dt → 0.97 → 誤判 DT → RES 不壓 echo → leak
- OLD/NEW raw_dt 公式**bit-exact identical**（diagnostic 驗證）
- OLD 看似沒 bug 純粹是 EPC 持續 refresh hangover 把 raw_dt 每 frame
  zero 掉的 lucky side effect
- NEW EPC guards (B-4a/4b/12: dt_signal<0.3, shadow_adv>1.3, sat_safe)
  設計**本身正確**（防止 DT 期 EPC 誤觸發），只是把 pre-existing bug 暴露

**參考**：
- Commit `1917c25` — docs/PZ7V_FS84_root_cause.md（完整 analysis）
- Commit `6ec7264` — WIP: PZ7V 元兇定位 + _FIX flag 基礎設施

## Section 2: 已排除嫌疑完整清單

PZ7V single-file 測過所有版本，**全部 bit-identical to full（−13.83 dB）**：

| 版本 | 結果 |
|---|---|
| `_FIX_A=0` | identical to full |
| `_FIX_B3A=0` | identical |
| `_FIX_B3B=0` | identical |
| `_FIX_B3C=0` | identical |
| `_FIX_B7=0` | identical |
| `_FIX_B11=0` | identical |
| `min_kalman` (全 6 個 OFF) | identical |
| `EPC_MULTI_LEVEL=0` (binary 也含相同 guards) | identical |
| B-9 shadow saturation gate revert | identical |
| B-6 DTD/RES feedback (per_bin_eer source) revert | identical |
| Power EMA gate revert (PBFDAF.process) | identical |
| `_update_per_frame_state` 流程 gate revert | identical |
| PBFDKF `_update_weights` body OLD-graft (Round 1) | identical |
| PBFDKF + `_update_per_frame_state` no-op (Round 2) | identical |
| AB3 state machine forced OFF + Set B disabled | identical |

**Reverse graft** (NEW PBFDAF+PBFDKF 移植進 OLD aec.py)：
- 結果 −33.10 dB ≈ OLD baseline −32.43
- → **PBFDKF/PBFDAF class rewrite 完全清白**

**OLD force epc=False** (force OLD's epc_active = False everywhere)：
- raw_dt 同樣飆到 0.975 (bit-exact 跟 NEW 一致)
- 10-11s output −14.20 dB（跟 NEW −13.83 幾乎一樣）
- → **確認 raw_dt bug 是 root cause，OLD 只是被 EPC 意外掩蓋**

## Section 3: 800-case bisect 狀態

| Run | 完成 | FS_echo | DT_echo | DT_deg | NE_deg | Δ vs baseline |
|---|---|---|---|---|---|---|
| **baseline** | ✓ | 3.621 | 4.115 | 2.386 | 4.003 | — |
| **full** (all fixes) | ✓ | 3.474 | 4.034 | 2.445 | 4.005 | FS −0.147, DT_echo −0.081 |
| bisect_a (Fix A OFF) | ✓ | 3.467 | 4.032 | 2.451 | 4.006 | FS −0.154, DT_echo −0.083 |
| bisect_b3a (B3A OFF) | ✓ | 3.486 | 4.043 | 2.431 | 4.004 | FS −0.135, DT_echo −0.072 |
| bisect_b3b (B3B OFF) | ✓ | 3.472 | 4.034 | 2.447 | 4.006 | FS −0.149, DT_echo −0.081 |
| bisect_b3c (B3C OFF) | 🔄 running 22:08 | — | — | — | — | — |
| bisect_b7 (B7 OFF) | 🔄 running 22:08 | — | — | — | — | — |
| bisect_b11 (B11 OFF) | 🔄 running 22:08 | — | — | — | — | — |

3 路並行 ETA：23:30–00:30 完成 800-case generation；scoring 各約 5 分鐘。

## Section 4: 待辦

1. 等 b3c/b7/b11 完成 + AECMOS 評分
2. 看完整 6 路 ΔFS 表，判斷個別 flag 對 −0.147 退步的貢獻比重
3. 規劃 **組 4.5 / B-15: raw_dt delay alignment fix**
   - 把 `far_pwr` 換成 delay-aligned echo power 估計
   - 例如 `raw_dt = max(0, 1 - echo_est_pwr / mic_pwr)` where
     `echo_est_pwr = sum(|filter.echo_spec|²)`
   - 包進 `AEC_FIX_B15` flag, default ON, 舊公式留在 OFF 分支
   - 驗證：PZ7V t=10s dt_indicator 應穩在 < 0.3
4. B-15 完成後跑 final 800-case 確認 −0.147 收斂
5. 與 linear/speex/aec2/aec3 final 比較表

## Section 5: 下個 session 開場 prompt

複製貼給 Claude Code：

```
讀 docs/handoff_to_next_session.md 跟 docs/PZ7V_FS84_root_cause.md。
執行 bash:
  cd /Users/mingyu/Desktop/novatek/SE/AEC/python
  for x in bisect_b3c bisect_b7 bisect_b11; do
    ls baselines/$x | wc -l
    grep -A1 "MEAN" /tmp/score_$x.log 2>/dev/null | head -20
  done
確認 3 路 bisect 完成狀態 (1600 files 表示 done)。如有未 score
的，跑 python3 eval_aecmos_local.py ../wav/aec_challenge_blind/ -o
baselines/<name>。回報所有路的 ΔFS/ΔDT_echo vs baseline。
不要開始新 fix，等我下指令。
```

## 工具 / 路徑提示

- 主要 aec.py: `python/aec.py` (commit 6ec7264)
- _FIX flags 基礎設施 line 27-55
- 單 file 測試 script: `python/_run_single.py`
- baseline 完整 wav: `python/baselines/pbfdkf_balanced/`
- OLD v2.8.0 aec.py: `/tmp/aec_OLD_v280.py`
- Diagnostic backup state: `/tmp/aec_current_state_before_bisect.py`
- Eval cmd template:
  ```
  AEC_MODE=PBFDKF AEC_EPC_MULTI_LEVEL=1 AEC_FIX_*=... \
    python3 eval_aec_challenge.py ../wav/aec_challenge_blind/ \
    --preset balanced --output-dir baselines/<name>
  ```
- AECMOS scoring: `python3 eval_aecmos_local.py ../wav/aec_challenge_blind/ -o baselines/<name>`
