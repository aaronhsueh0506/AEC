# AEC delay estimator 產品化：選型與驗證

範圍：`AEC/`、`Audio_ALG/pipelines/`、`Audio_ALG/AIAEC/`。現行 API 與行為以
`c_user_manual_zh_TW.md`、`delay_estimator_design_zh_TW.md`、公開 header 及測試為準；
本文規定**選 delay mode／`delay_num_filters` 的量測方法**與**驗證項目**。

## 1. 三個 delay 預算必須分開配置

`delay_num_filters`、PBFDKF `filter_length` 與 AIAEC TA `D`應是三個
彼此獨立的 init/export 參數，不得由其中一個自動推導另一個，也不得
用單一「總 delay range」把三者相加：

| 參數 | 負責的問題 | 不能取代的能力 |
|---|---|---|
| matched-filter `n` | 搜尋、鎖定 bulk far→mic delay，驅動 reference alignment | 不負責建模房間 echo tail |
| PBFDKF `filter_length` | 在 reference 已對齊後，建模 causal echo path/RIR tail | 不是 bulk-delay acquisition window，也無法吸收負 residual |
| TA `D` | 在送入 NN 的 far 上搜尋 current/past frame，範圍為 `(D-1)*model_hop` | 不會修復 PBFDKF 的 delay lock，也不會延長 PBFDKF coefficients |

三者 API 獨立，但系統能力仍有耦合條件：

```text
matched filter/external align 必須先將 bulk delay 收進可用 residual
0 <= positive residual + effective acoustic tail < PBFDKF filter span
0 <= NN input residual <= (D - 1) * model_hop       # causal TA only
```

- `aligned_far` 模式：matched filter 負責 bulk delay，PBFDKF 負責 linear
  cancellation，小 `D` 只吸收量化誤差、drift 與 delay-change transient。
  matched filter 未鎖定或鎖錯時必須 fail-open，不能把 TA 當成額外
  matched-filter bank。
- `raw_far` 模式：TA 可搜尋總 far/model-input delay，但 PBFDKF 仍需
  自己的 reference alignment。這只對 E2E-AECNR 有機會成為大範圍備援；
  RES+NR 不得依賴 TA 補救已失鎖的 linear AEC。

因此縮小 `n` 前要分別量測：產品 bulk-delay 分佈選 `n/mode`；
對齊後 residual + RIR tail 選 PBFDKF `filter_length`；NN 實際輸入的
residual-frame lag 分佈選 `D`。若有 external coarse delay，可將大幅固定
delay 移除後再配小 `n` + 小 `D`，但不能宣稱總覆蓋範圍等於
`MF reach + filter_length + TA reach`。

## 2. 選 mode／n 的量測方法

每個 SKU/route 收集 far→mic residual delay：

- cold boot。
- warm start。
- suspend/resume。
- route switch。
- buffer size/codec mode change。
- CPU loading／underrun recovery。
- clock drift。
- echo-path movement。

使用：

```text
p99.9 residual + jitter + drift + safety margin < bank reliable reach
```

起始候選（不可取代實測）：

| 產品條件 | 起始候選 |
|---|---|
| 固定硬體路徑、量測後幾乎不變 | `FIXED` |
| HAL 已對齊且保證 residual | `EXTERNAL_ALIGNED` |
| residual 約 8～100 ms | `MATCHED n=1` |
| residual < 約 180 ms | `MATCHED n=2` |
| residual < 約 280 ms | `MATCHED n=3` |
| 延遲來源不確定 | `MATCHED n=5` |
| 可能 >509 ms | 外部 coarse delay；增加 ring/max_delay 不會增加 matched-filter reach |

外部 coarse alignment 要刻意留下小幅正 residual（初始可驗 8～16 ms），不要過補成負 residual。

實測的 duty-cycle engagement 用 `aec_wav` 結尾的 `duty:` 診斷行取得
（`c_user_manual_zh_TW_04_seam_troubleshooting.md` §8.4）。

## 3. 驗證項目

### 3.1 結構／bit-exact gate

- grid：16k/256/128、16k/512/256、48k/1024/512。
- C：NE10 × SIMD=0/1 × shadow on/off × CNG on/off。
- mono/4ch static-vs-heap/caller-pool parity。
- 4ch matched-filter instance count 永遠 1。
- fixed/external modes 不出現 matched-filter update。
- reset 後 ring、generation token、delay state 正確。
- `mem(n=1) < ... < mem(n=5)`、`FIXED mem < MATCHED n=1 mem`、
  `EXTERNAL_ALIGNED mem < FIXED mem`（`make test-delay-num-filters`）。

### 3.2 Audio gate

- far-end static/movement。
- double-talk static/movement。
- near-end-only。
- delay step/path change。
- clock drift。
- short clips、非 hop 整除 tail。
- native 48 kHz recordings，不只升採樣素材。
- AECMOS echo/deg、ERLE/residual RMS、applied-delay switches、acquisition time、deep-regression counts。
- 部署 n<5 時，AIAEC 訓練／推論要用同一組 delay 設定驗證 acquisition/path-change residual；
  不能只用 n=5 dataset 分數宣稱 train/deploy 一致。

### 3.3 不放行條件

以下任一成立即不放行：

- 任一 mode 有隱式 fallback。
- get-mem-size/carve 不 lockstep。
- processing path allocation。
- 4ch 不只一個 shared estimator。
- aggregate 通過但 deep-regression/tail 超門檻。
- AIAEC contract 與實際 deployed delay config 不一致。
