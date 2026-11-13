# Delay Estimator（matched filter）設計說明

機制、語意警示、`num_filters` 的取捨、blind-test 語料延遲分佈與 48 kHz 適用性。
所有數字可由文中列出的工具/測試重現。

相關文件：`c_user_manual_zh_TW.md` §延遲契約（使用者視角的上限表）、
`nn_integration_interface.md`（509 ms 可靠上界的引用點）、
`aec_methods.md`（模組全景圖）。

## 1. 機制總覽

訊號路徑（Python spec：`python/modules/delay/`；C：`c_impl/src/delay_aec.c`）：

```
render/capture（per hop）
  └─ 48 kHz 時先過 _Resample48 抗混疊 sidechain（÷3 → 16 kHz-equivalent）
  └─ 外→內 edge-chunker：切成 64-sample inner block（4 ms）
       └─ Decimator ÷4（elliptic LP + HP biquad）→ 4 kHz sub-block（16 samples）
       └─ DownsampledRenderBuffer（ring，2064 ds samples）
       └─ MatchedFilter：n=5 個 NLMS 互相關 filter，錯開排列
       └─ MatchedFilterLagAggregator：histogram 投票 → DelayEstimate
  └─ orchestrator Path A（首次取得）/ Path B（變更確認）→ ref ring 讀取偏移
```

幾何（downsampled 4 kHz 域，1 ds sample = 0.25 ms）：

| 常數 | 值 | wall-clock |
|---|---|---|
| sub-block | 16 ds samples | 4 ms |
| filter window（`window_size_sub_blocks=32`） | 512 ds | 128 ms |
| 相鄰 filter 錯開（`alignment_shift_sub_blocks=24`） | 384 ds | 96 ms |
| filter 數 `num_filters` | 5 | — |
| 幾何全 span | `(5−1)×384+512 = 2048 ds` | 512 ms |
| **可靠上界**（reliable 需 `peak < FILTER_SIZE−10`） | 2037 ds | **509.25 ms** |
| `get_max_filter_lag()` 公式值（histogram 配置用，多算一個 shift） | 2432 ds | 608 ms |

每個 sub-block，每個 filter 做一次 NLMS 更新與誤差量測；勝者 =
`reliable` 且 `error_sum` 最小的 filter，`reliable ⟺ peak 在窗內邊界外
且 error_sum < matching_filter_threshold × 當下 16-sample capture 能量`。
勝者的 lag 丟進 aggregator 的 250-estimate histogram；同一 bin 累積
超過 `converged=20` 票即 latch `REFINED`（confidence 1.0）。

## 2. 語意警示

1. **confidence 量的是「一致性」，不是「匹配品質」**。`error_sum` 在
   `MatchedFilter.update` 內就被丟棄，不進 aggregator。一個持續出現的
   錯誤候選（例如真延遲超出搜尋範圍時、窗內的訊號自相關假峰）會
   正常累票拿到 confidence 1.0，且 latch 永不自動歸零。
2. **超出可靠上界（509 ms）的真延遲不會「估不到」，而是「高信心估錯」**。
   實測案例：真延遲 579.44 ms 的遠端單講錄音，5-filter 在 3.7 s 鎖定
   32 ms、confidence 1.0、殘差 548 ms、永不自我修正。收緊
   `matching_filter_threshold` 無法擋（假/真匹配的誤差比分佈重疊，
   實測掃描 0.03–0.25 全域無分離點）。
3. **鎖定後出口極少**：latch 後 NLMS 平滑降為 `smoothing_slow=0.1`、
   每 ~500 ms 的 soft reset 只清 filter 不清 histogram、orchestrator
   Path B 需要估計值變動 >32 samples 才動作。錯誤鎖定實務上只有
   `aec_reset()` 能清，而重置後幾何限制不變、會鎖回同一個錯誤值。
4. **回報的是 dominant peak**。aggregator 用 highest-peak 直方圖判定品質，
   production 回報的延遲就是這個 dominant 候選；pre-echo 候選只供 AEC3
   對照與診斷，不決定對齊位置（若回報 pre-echo 候選，錯值會天生自洽、
   confidence 天生是 1.0，任何「連續 K 次確認」閘門都擋不住）。由
   `test_delay_dominant_selection`（Python 與 C）釘住。
5. **dominant-peak 近似平手時，C 與 Python 可能落在不同路徑。** 例如舊反射與
   新路徑增益相等的 multipath 場景，aggregator 的投票幾乎正好平手；兩個
   port 的 tie-break 規則相同（bin index 最小者勝，bin-to-lag 映射方向一致），
   落點由各自測試雜訊實現的具體樣本決定，不是 port 之間的缺口。

## 3. `num_filters=5` 的取捨

- upstream 參考實作中它是 `EchoCanceller3Config::Delay::num_filters` 的
  **純靜態預設值**（validator 只防呆 0..5000），沒有任何依延遲需求自動
  調整的機制。
- 本實作逐字沿用，C port 逐字鏡像，**未獨立 A/B 驗證**。
- 與我們的 hop/filter_length 沒有推導關係：matched filter 活在自己的
  4 ms sub-block 網格；`filter_length`（16 kHz 時 832 taps = 52 ms）管的
  是「對齊之後」線性濾波器能模的殘差+回聲尾長。系統不變量是
  **鎖定後殘差恆正且 <10 ms**（64-sample 量化 + headroom 恆提早 32–92
  samples），與 filter 數無關。
- 兩個常數與 upstream 不同、沒有記錄理由：
  `matching_filter_threshold 0.3`（upstream 0.2）、`smoothing_slow 0.1`
  （upstream 0.7）。任何變更須走 800-case 隔離驗證。

## 4. 超出範圍的延遲：系統層責任（設計立場）

upstream 對 >512 ms 延遲的官方解是**外部 delay hint**（OS 回報音訊緩衝
延遲、`use_external_delay_estimator` 直接繞過 matched filter）——即
「系統延遲由系統回報，演算法只追殘差」。本 repo 的對應物：

- **板端（系統延遲已知）**：bring-up 量測後設 `fixed_delay_samples`
  （estimator 不建構，搜尋算力 = 0）。校正誤差直接變成殘差，須納入
  bring-up 驗收（目標：殘差 < 10 ms 等級）。
- **離線 dataset 生成（真延遲可離線量測）**：per-clip 用
  `wav/measure_blind_delay.py` 同法量測真延遲 → 餵 `fixed_delay_samples`
  或剔除超界 clip，並以「estimator 估計 vs 離線真值」做 QA 檢核。
- **bench（不可用任何 pre-align，維持誠實計分）**：現行幾何照跑；
  超界案例（見 §6，約 5%）如實反映在分數上，upstream 參考實作同樣
  受此限制，相對比較公平。

## 5. 算力與縮小方向

C 端目前每 filter、每 64-sample block 的 dot／NLMS MAC 上限為
`512 + 16×512 + 16×512 = 16,896` MAC：一次初始 render 能量、
16 次 dot，以及全部通過 excitation／saturation gate 時的 16 次 NLMS 更新。
後續 render 能量用原順序的 sliding recurrence。下表以 250 block/s 計算 bank 上限，
未含 sliding recurrence、error sum、decimator、peak scan、histogram 與
scalar bookkeeping；不是 CPU 實測。
穩態實際分析比例由現有 duty state machine 與 watchdog 決定，不能固定除以 10。

RAM 欄是 `delay_aec_get_mem_size()` 的**實測**回傳值（不是估計），
16 kHz / hop 128；48 kHz / hop 512 每列再加 1,376 B 的抗混疊 sidechain
scratch（只有 48 kHz 才配置）。

| n | 可靠覆蓋 | full-rate bank 上限 | RAM（估計器，16 kHz 實測） |
|---|---|---|---:|
| 5（預設） | 509 ms | 21.120 MMAC/s | 33,936 B |
| 4 | 413 ms | 16.896 MMAC/s | 28,208 B |
| 3 | 317 ms | 12.672 MMAC/s | 22,480 B |
| 2 | 221 ms | 8.448 MMAC/s | 16,752 B |
| 1 | 125 ms | 4.224 MMAC/s | 11,024 B |
| estimator OFF（`FIXED` / `EXTERNAL_ALIGNED`） | — | 0 | 0 B |

NEON 路徑把當次 tap update 與下一個 sample 的 dot 合併走訪，直接使用
剛更新的 tap，減少 coefficient reload。首筆 dot、末筆 update 及無更新分支各自處理；
scalar 路徑保留原演算。`make test-delay-kernels` 對照獨立的分開走訪 reference，
檢查所有 taps、error sum，以及 first-only／last-only／no-update 等 gate。

縮 n 不動 window/shift/threshold（每個窗的統計行為與 upstream 完全一
致），是偏離最小的縮法。

**Config 通路**：Python `AecConfig.delay_num_filters`
（1..5，`__post_init__` fail-fast 驗證）→ orchestrator → LegacyDelayShim
→ `EchoPathDelayEstimator(num_filters=…)`；C `AecConfig::delay_num_filters`
（`aec_resolve_valid_config` 拒絕超界）→ `delay_aec_get_mem_size()` /
`delay_aec_init()` → runtime `DaMatchedFilter::num_filters`。

**RAM 隨 n 縮**：`DelayAec` 是 metadata + pool pointers，bank／accumulated
error／render ring／兩個 lag histogram 全部由 `aec_get_mem_size(cfg)` 依
resolved `(sample_rate, hop, n)` 從呼叫端記憶池切出，每少一組 filter 固定省
**5,728 B**（每個格點都一樣），n=1 相對 n=5 省 22,912 B。`DA_NUM_FILTERS`(=5)
只是驗證用的幾何上限，不 size 任何東西。零 heap；`n` 與格點一樣是 init-time
immutable（要改必須重新查大小並重新 init）。

Histogram bins 用 `n*SHIFT + SIZE` 的公式（多帶一組 filter 的餘裕），不是算術上最緊的
`(n-1)*SHIFT + SIZE`。這是必要的：pre-echo histogram 的 bin 數決定 windowed
local-max scan 走幾個 32-bin 視窗，用最緊公式時 n=2 只會掃 1 個視窗（bin 0..31），
但該 bank 自己就能報到 bin 55——等於砍掉一半搜尋範圍。此公式在 n=1..5 都剛好涵蓋
可達 bin，且多出的視窗恆為全零（永遠贏不了 scan），所以每個 n 的行為都與
「陣列固定配到 n=5 上限」時完全一致。

使用場景：板端 fixed_delay 補償後 nf=1~2；bench/dataset 一律維持 5
（發佈分數的量測幾何）。

### 5.1 實務延遲量級參考（按裝置/傳輸路徑）

各部署形態的 bulk delay 典型量級（經驗值，供選 mode/n 的第一輪判斷；
正式選值仍以 bring-up 實測公式為準——見
`reference/delay_selection.md` §「p99.9 residual +
jitter + drift + safety margin < bank reliable reach」）：

| 部署形態 | 典型 bulk delay | 性質 | 建議 mode / n |
|---|---|---|---|
| 嵌入式 SoC、ref 走內部 loopback | ~0–50 ms | **固定、可量測**（firmware buffer 大小定死） | `FIXED`（bring-up 量一次），或 `MATCHED` n=1 |
| USB／類比會議裝置 | 20–100 ms | 準固定 | `FIXED`＋小殘差，或 `MATCHED` n=1 |
| PC／手機軟體堆疊 | 50–300 ms | 變動（OS buffer、resampler） | `MATCHED` n=2~3 |
| 藍牙輸出 | 150–500 ms+ | 變動、重連會跳 | `MATCHED` n=4~5 |
| challenge 語料（未知裝置錄音） | 實測見 §6：2021 全集 max 923 ms（~5% 超出 509 ms）；2023@48k max 674 ms（~1% 超界） | 無系統 hint，只能自己找 | `MATCHED` n=5（bench/dataset 固定幾何） |

原則（設計立場詳見 §4）：**系統延遲是系統層的知識**——firmware
buffer 大小本來就已知，或 bring-up 用 loopback 量一次——用
`fixed_delay_samples` 補掉（estimator 整個不建、搜尋算力=0），演算法
只留小範圍追殘差/drift。這也是 upstream 的哲學：OS 以
buffer-delay hint 回報延遲，matched filter 是沒有 hint 時的自救手段。
延遲不可知的形態（藍牙類）才需要大 n 全程搜尋。

## 6. Blind-test 語料延遲普查

工具：`wav/measure_blind_delay.py`（NCC + 裁齊長度避開 zero-pad 假峰 +
dominance 判準 + undecidable 閘；方法以 579.44 ms 已知案例與合成
延遲自驗）。注意：GCC-PHAT 在 mic/lpb 檔長不一致時會把長度差放大成
假峰（例如 600.00 ms 假峰 = 檔長差），故本工具用 plain NCC。

**Interspeech 2021 blind set（16 kHz，287 對，全集）**：

| 場景 | n(可測) | p50 | p90 | max | >509 ms | >128 ms |
|---|---|---|---|---|---|---|
| farend_singletalk(+movement) | 128 | 48 ms | 226 ms | 923 ms | 7 | 25 |
| doubletalk(+movement) | 101 | 47 ms | 219 ms | 923 ms | 5 | 21 |
| nearend_singletalk | 5（28 無回聲不可測） | 81 ms | 268 ms | 349 ms | 0 | 2 |

- 超出可靠覆蓋（509 ms）的尾巴約 **5%**（12/234 可測 FST+DT）——
  這些案例在現行幾何下會發生 §2 的高信心誤鎖。
- 少數量測為負值/低 ncc（如 −95 ms、ncc 0.07）屬弱相關低信心案例。

**ICASSP 2023 blind set（48 kHz，每場景 100 case＝300 case / 600 檔）**：

| 場景 | n(可測) | p50 | p90 | max | >509 ms | >128 ms |
|---|---|---|---|---|---|---|
| farend_singletalk | 69 | 159 ms | 342 ms | **674 ms** | 1 | 38 |
| farend_singletalk_with_movement | 23 | 168 ms | 299 ms | 406 ms | 0 | 18 |
| doubletalk | 74 | 79 ms | 262 ms | 338 ms | 0 | 33 |
| doubletalk_with_movement | 20 | 104 ms | 255 ms | 263 ms | 0 | 9 |
| nearend_singletalk | 22（78 無回聲不可測） | 82 ms | 294 ms | 673 ms | 1 | 9 |

- 2023 的超界尾巴約 **1%**（2/208 可測 FST+DT+NST-洩漏，674/673 ms），
  遠小於 2021 的 5%；分佈整體偏高於 2021（p50 79–168 ms vs 48 ms）但
  99% 落在現行 509 ms 覆蓋內。

## 7. 48 kHz 適用性

**可以直接餵 48 kHz wav**：`process_wav_files` 自動從檔頭讀取取樣率並
解析 grid（48 k → frame 1024 / hop 512 / filter_length 3072 = 64 ms，
`config.py` fail-fast 驗證非法組合）；delay sidechain 行為級驗證通過
（同一合成 300 ms 延遲，48 k 與 16 k 鎖定值差 **0.000 ms**；ERLE 正常
上升；無 NaN）。覆蓋上限以 ms 計與 16 k 相同（509/608 ms）。

**真實 48 kHz clip 抽驗（ICASSP-2023 集 10 clips 跨 0–674 ms delay 頻段）**：
9/9 在覆蓋內的 clip 全部鎖對 delay（誤差 +2.8～+5.7 ms、恆提早=headroom 契約）；
消除量實測 4–13 dB（far-active 能量域；三個 instant-ERLE 尾段平均偏低的 clip
經 far-active gating 重驗均為 10–12.8 dB 真實消除——instant-ERLE 的靜音段稀釋是
量測 artifact，不是引擎問題）；674 ms 超界 clip 為**誠實未鎖定**（未誤鎖）。
→ 48 kHz 以現行參數可用於 dataset 生成與評測。

注意事項：

1. **調校未驗證是官方已記載限制**（CHANGELOG `[4.0.0]` Known
   limitations）：48 kHz 只有結構性驗證與上述抽驗，沒有系統性 AECMOS 調校驗證。
2. **驅動腳本預設**：`eval_aec_challenge.py --filter` 預設 `-1`（自動：
   <44.1 kHz 為 52 ms、≥44.1 kHz 為 64 ms）；`run_one_case.py` 與
   `eval_manifest90.py` 的 `--filter` 預設 832（16 kHz-authored 的絕對 sample 數），
   48 kHz 須顯式覆寫為 3072，`run_one_case.py` 另須 `--sample-rate 48000`
   （不符會 fail-fast 報錯）。
3. C↔Python 的 48 kHz 差異（f32/f64 Kalman 捨入放大所致，容差見
   `c_impl/test/parity_aec_e2e.c`）只影響跨語言 parity 測試與 C 部署驗證，
   Python-only 的 dataset 生成不受影響。
4. 未 retime 常數（`_simple_mu_ratio` 四常數、`erle_coh_gate_alpha`）為
   全 grid 共通；48 k/1024 的 hop（10.667 ms）最接近原 authoring grid，
   漂移幅度（+4~7%）是各 grid 中最小。
