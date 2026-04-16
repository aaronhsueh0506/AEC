# AEC 待辦事項（v2.3.0 後）

## 高優先（影響分數或正確性）

| # | 來源 | 項目 | 說明 | 狀態 |
|---|------|------|------|------|
| 1 | CODE_REVIEW A3 | GCC-PHAT 負延遲 | 寫入 README 限制條件，客戶端責任 | ✅ v2.3.1 |
| 2 | CODE_REVIEW D4 | `_wn_err_baseline` 初始化 | 非 stationary 時緩慢追蹤 baseline | ✅ v2.3.1 |
| 3 | CODE_REVIEW D2 | erle_factor ramp 2→0 dB | `erle/10.0`，搭配 B4 動態 ERL | ⏳ 800-case 驗證中 |
| 4 | CODE_REVIEW D3 | shadow_err_alpha 0.85→0.80 | 更快追蹤 shadow advantage | ⏳ 800-case 驗證中 |
| 5 | CODE_REVIEW D6 | kalman_q_low 1e-5→1e-6 | 更低 misadjustment | ⏳ 800-case 驗證中 |
| 6 | BLINDSPOTS B5 | BUG-1+2 coh2 窗函數統一 | 需獨立 session 從頭 retune | ⏭️ C 移植後再做 |

## 中優先（結構/完整性）

| # | 來源 | 項目 | 說明 |
|---|------|------|------|
| 7 | CODE_REVIEW E1 | ResFilter init/reset 分離 | `__init__` 結尾呼叫 `reset()`，immutable config vs runtime state 分開 |
| 8 | CODE_REVIEW E2 | AecResContext 加 erl_estimate | 外部 RES 需要 B4 的動態 ERL |
| 9 | CODE_REVIEW D5 | 48kHz filter_length 64ms | 48kHz RIR 更長，32ms 不夠 |
| 10 | CODE_REVIEW D1 | MILD shadow_dtd_offset=1.2 | mild DT deg 2.441 已 OK，可跳過 |

## 低優先（效能/清理，C 移植時做）

| # | 來源 | 項目 | 說明 |
|---|------|------|------|
| 11 | BLINDSPOTS P1 | PBFDKF _update_weights 向量化 | 7 partitions × 2 FFT/IFFT per frame → 矩陣運算 |
| 12 | BLINDSPOTS P2 | HighPassFilter → scipy.signal.lfilter | sample-by-sample loop，48kHz 瓶頸 |
| 13 | BLINDSPOTS P3 | ERLE EMA closed-form decay | sample-by-sample loop → dot product |
| 14 | CODE_REVIEW F5 | ring buffer wrap 優化 | np.concatenate → 預分配 buffer |

## 下一步：C 移植（v2.3.1 收版後）

| # | 項目 | Python 版本 | 狀態 |
|---|------|------------|------|
| 21 | v2.1.0 改動移植 | AEC3 echo switching, P-floor, inst_erle cap, light release, BUG-3 | 未開始 |
| 22 | v2.3.x 改動移植 | Energy DT, shadow offset, 動態 ERL, PAR, EPC 延長, cleanup | 未開始 |

## C 移植後：BUG-1+2 重新嘗試

| # | 項目 | 說明 |
|---|------|------|
| 6 | BUG-1+2 coh2 窗函數統一 | 獨立 session，從頭 retune。之前兩次嘗試失敗（rectangular FS -0.040、windowed FS -0.023） |

## 架構重構（BUG-1+2 後）

| # | 來源 | 項目 | 說明 |
|---|------|------|------|
| 15 | GPT 1.1 | AEC/RES 3 層架構分離 | Linear AEC / State estimation / Postfilter |
| 16 | GPT 1.3 | DT 三信號分層 | echo presence / near-end presence / filter trust |
| 17 | GPT 1.4 | AEC.process() 拆分子函式 | 7 stage functions |
| 18 | GPT 3.1 | Kalman 控制邏輯分層 | P update / W update / hard reset / external trust |
| 19 | GPT 4.1 | RES gain 分層結構 | primary / modifier / clamp / temporal |
| 20 | GPT 4.2 | AecResContext 完整重設計 | signal / confidence / mode+event 三類本體 |

---

## 已確認不做的項目

| 項目 | 原因 |
|------|------|
| Square Root Kalman | P 是純量，矩陣正定性問題不存在 |
| np.roll 替代滑動賦值 | allocate 新 array，反而更慢 |
| shadow_q_ratio > 5.0 | shadow 不穩，copy gate 品質下降 |
| scipy.linalg.blas | 不適合 per-bin element-wise 運算 |
| Soft delay shift | Hard-shift + EPC 更快收斂 |
| D7 max_drop 4dB | FS echo 代價 -0.024 不值得 |
| BUG-1+2 rectangular error_spec | alpha 不匹配，全面退步 |

*更新於 2026-04-16*
