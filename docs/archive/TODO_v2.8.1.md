# AEC 待辦事項（v2.8.1 後）

## 當前進行中

（movement DT 三輪 ablation 全部結案，目前無進行中項目）

## 確認不做（已測試無效或退步）

### Movement DT 相關（2026-04-28 三輪 ablation 結案）

| 項目 | 原因 |
|------|------|
| Filter capacity 延長（L1 768 / L2 1024 / L3 1536）| mvDT echo Δ ≤ +0.020（噪音範圍）；filter_converged% 在所有 variant 維持 2.4%，DT 抑制 adaptation 才是瓶頸 |
| Oracle movement 偵測 + filter reset（R1/R2/R3）| mvDT echo Δ ≤ +0.002；fc%/shb%/epc% 完全不變，DT gate 在 reset 後持續抑制 adaptation |
| Oracle movement 偵測 + Q boost（R4/R5）| 同上；即使 oracle 注入 Q_high 也無效 |
| RES ENR threshold scaling（S1a/b/c, ×0.75/0.50/0.25）| 增加 suppression 16%→19%，但 mvDT echo 不改善；因 filter echo estimate 在 echo path 改變後已失準，suppression 打到 speech |
| RES RER fallback gain cap（S2a/b/c）| mvDT echo Δ ≤ +0.003（噪音範圍），同上根本原因 |
| RES hybrid RER suppression（S3）| mvDT echo +0.007（噪音範圍），但 FS echo +0.029（代價不值得）|
| GCC-PHAT oracle delay policy（P1-P3, R1-R4, 8 variants）| 全部無效或退步；movement DT 不是 delay estimate 問題 |

### 其他

| 項目 | 原因 |
|------|------|
| EPC hangover deadlock fix（3c3d85c） | FS echo -0.121, DT echo -0.065；epc_active stuck True 意外維持高 mu + 強壓制對 movement 有利 |
| Per-bin dt_suppress 3.2 + dt_residual_scale 3.1 | 50-case 零效果：coh2 在 DT 也全局偏低；且 RES gain 在 DT 本就 ≈ 1.0，改 scaling 無影響 |
| Experiment D（移除 dt_residual_scale） | 零效果：ENR 恆 < threshold，RES 透明，scaling 無關 |
| Section 9.3 acoustic ERL detector | 零效果：movement+DT 同時發生時 dt_from_energy gate 阻擋觸發；拿掉 gate 則語音誤判 |
| AEC_CODE_REVIEW Bug 3/8/10/12/16 | R_scale / ramp / reverb clamp / conv threshold / near_psd order，全部 regress |
| BUG-1+2 rectangular error_spec | alpha 不匹配，全面退步 |
| Square Root Kalman | P 是純量，矩陣正定性問題不存在 |
| shadow_q_ratio > 5.0 | shadow 不穩，copy gate 品質下降 |
| Soft delay shift | Hard-shift + EPC 更快收斂 |
| D7 max_drop 4dB | FS echo 代價 -0.024 不值得 |

## 中優先（結構完整性，不影響分數）

| # | 來源 | 項目 | 說明 |
|---|------|------|------|
| 5 | CODE_REVIEW E1 | ResFilter init/reset 分離 | `__init__` 結尾呼叫 `reset()`，immutable config vs runtime state 分開 |
| 6 | CODE_REVIEW E2 | AecResContext 加 erl_estimate | 外部 RES 需要動態 ERL |
| 7 | CODE_REVIEW D5 | 48kHz filter_length 64ms | 48kHz RIR 更長，32ms 不夠 |

## 低優先（效能/清理，C 移植時做）

| # | 項目 | 說明 |
|---|------|------|
| 8 | PBFDKF _update_weights 向量化 | 7 partitions × 2 FFT/IFFT per frame → 矩陣運算 |
| 9 | HighPassFilter → scipy.signal.lfilter | sample-by-sample loop，48kHz 瓶頸 |
| 10 | ring buffer wrap 優化 | np.concatenate → 預分配 buffer |

## C 移植待同步

| 項目 | Python 版本 | 狀態 |
|------|------------|------|
| v2.1.0–v2.3.x 改動 | Energy DT, shadow offset, 動態 ERL, PAR, EPC 延長 | 未開始 |
| v2.4.0 Bug 7/2/6 | dt_per_bin ** 1.3, FilterErle fix, 動態 ERL ceiling | ✅ 2026-04-17 |
| v2.8.x per-bin DT（3.1+3.2） | coh2 per-bin dt_suppress + dt_residual_scale | 待 Python 驗證後 |

## 長期架構（分數穩定後）

Section 8 State Machine（STARTUP_DT / EPC_FS / EPC_DT / RECOVERY 等），Section 9.2 DT 訊號分層（dt_for_adaptation / nearend_protect_score / echo_suppress_confidence）。

*更新於 2026-04-28（v2.8.1）*
