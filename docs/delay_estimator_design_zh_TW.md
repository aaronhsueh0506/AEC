# Delay Estimator（matched filter）設計說明

2026-08-14 首版：整併本輪 root-cause 調查（FST 579 ms 誤鎖案例）、upstream 溯源、
blind-test 語料延遲普查與 48 kHz 適用性稽核的結果。所有數字可由文中列出的
工具/測試重現。

相關文件：`c_user_manual_zh_TW.md` §延遲契約（使用者視角的上限表）、
`nn_integration_interface.md`（509 ms 可靠上界的引用點）、
`aec_methods.md`（模組全景圖）。

## 1. 機制總覽

訊號路徑（Python spec：`python/modules/delay/`；C：`c_impl/src/delay_aec3.c`）：

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
| `get_max_filter_lag()` 公式值（histogram 配置用，overclaim 一個 shift） | 2432 ds | 608 ms |

每個 sub-block，每個 filter 做一次 NLMS 更新與誤差量測；勝者 =
`reliable` 且 `error_sum` 最小的 filter，`reliable ⟺ peak 在窗內邊界外
且 error_sum < matching_filter_threshold × 當下 16-sample capture 能量`。
勝者的 lag 丟進 aggregator 的 250-estimate histogram；同一 bin 累積
超過 `converged=20` 票即 latch `REFINED`（confidence 1.0）。

## 2. 語意警示（本輪 root-cause 的核心教訓）

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

## 3. `num_filters=5` 的出處（溯源結論）

- upstream 參考實作中它是 `EchoCanceller3Config::Delay::num_filters` 的
  **純靜態預設值**（validator 只防呆 0..5000），沒有任何依延遲需求自動
  調整的機制；至少 2019-10 → 2026-05 六年未變。
- 本 repo 於 2026-05-17（`77af915`）逐字沿用，之後 C port（`c5c09c4`）
  再逐字鏡像 —— **未曾獨立 A/B 驗證**（對照：`down_sampling_factor` 的
  ds8 選項曾以 60-case AECMOS 實測並因遠端單講退化而移除，是唯一被
  驗證過的幾何常數）。
- 與我們的 hop/filter_length 沒有推導關係：matched filter 活在自己的
  4 ms sub-block 網格；`filter_length`（16 kHz 時 832 taps = 52 ms）管的
  是「對齊之後」線性濾波器能模的殘差+回聲尾長。系統不變量是
  **鎖定後殘差恆正且 <10 ms**（64-sample 量化 + headroom 恆提早 32–92
  samples），與 filter 數無關。
- 兩處引入當日即與 upstream 分岔、無記錄理由的常數：
  `matching_filter_threshold 0.3`（upstream 0.2）、`smoothing_slow 0.1`
  （upstream 0.7）。維持現狀；任何變更須走 800-case 隔離驗證。

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

C 端實測模型（`delay_aec3.c`；每 filter 每 block 無條件 8704 MAC、
NLMS 8192 MAC、末端 accumulated-error 常數 8192 MAC；250 block/s；
鎖定穩定 0.3 s 後 duty-cycle 1-in-10）：

| n | 可靠覆蓋 | full-rate | duty 穩態 | RAM（估計器） |
|---|---|---|---|---|
| 5（現行） | 509 ms | 23.2 MMAC/s | 2.32 | ~32 KB |
| 3 | 317 ms | 14.7（−37%） | 1.47 | ~20 KB |
| 2 | 221 ms | 10.5（−55%） | 1.05 | ~16 KB |
| 1 | 125 ms | 6.3（−73%） | 0.63 | ~12 KB |
| estimator OFF（`fixed_delay_samples`） | — | 0 | 0 | ~0 |

縮 n 不動 window/shift/threshold（每個窗的統計行為與 upstream 完全一
致），是偏離最小的縮法。前置條件：板端 fixed_delay 補償後的殘差
profile（bring-up 量測）；`num_filters` 目前無 config 通路（Python 寫死
於 estimator 預設值、C 為編譯期 `DA_NUM_FILTERS`），縮小前需先開通。

## 6. Blind-test 語料延遲普查（2026-08-14 實測）

工具：`wav/measure_blind_delay.py`（NCC + 裁齊長度避開 zero-pad 假峰 +
dominance 判準 + undecidable 閘；方法先以 579.44 ms 已知案例與合成
延遲自驗）。注意：GCC-PHAT 在 mic/lpb 檔長不一致時會把長度差放大成
假峰（曾實測 600.00 ms 假峰 = 檔長差），故本工具用 plain NCC。

**Interspeech 2021 blind set（16 kHz，287 對，全集）**：

| 場景 | n(可測) | p50 | p90 | max | >509 ms | >128 ms |
|---|---|---|---|---|---|---|
| farend_singletalk(+movement) | 128 | 48 ms | 226 ms | 923 ms | 7 | 25 |
| doubletalk(+movement) | 101 | 47 ms | 219 ms | 923 ms | 5 | 21 |
| nearend_singletalk | 5（28 無回聲不可測） | 81 ms | 268 ms | 349 ms | 0 | 2 |

- 超出可靠覆蓋（509 ms）的尾巴約 **5%**（12/234 可測 FST+DT）——
  這些案例在現行幾何下會發生 §2 的高信心誤鎖。
- 少數量測為負值/低 ncc（如 −95 ms、ncc 0.07）屬弱相關低信心案例。

**ICASSP 2023 blind set（48 kHz 確認，小批 24 對，非全集）**：

| 場景 | n | p50 | max | >509 ms |
|---|---|---|---|---|
| doubletalk(+movement) | 8 | ~46 ms | 81 ms | 0 |
| farend_singletalk | 8 | 62 ms | 349 ms | 0 |
| nearend_singletalk | 8 | 7/8 不可測（正確）；1 檔洩漏 349 ms | | 0 |

小批樣本內全部落在現行覆蓋內；全集統計待需要時擴充（同一工具直接跑）。

## 7. 48 kHz 適用性（2026-08-14 稽核結論）

**可以直接餵 48 kHz wav**：`process_wav_files` 自動從檔頭讀取取樣率並
解析 grid（48 k → frame 1024 / hop 512 / filter_length 3072 = 64 ms，
`config.py` fail-fast 驗證非法組合）；delay sidechain 行為級驗證通過
（同一合成 300 ms 延遲，48 k 與 16 k 鎖定值差 **0.000 ms**；ERLE 正常
上升；無 NaN）。覆蓋上限以 ms 計與 16 k 相同（509/608 ms）。

注意事項：

1. **調校未驗證是官方已記載限制**（CHANGELOG 4.0.0rc Known
   limitations）：48 kHz 至今只有結構性驗證（不崩潰/有限值），從未用
   原生 48 kHz 素材做 AECMOS/ERLE 調校驗證。dataset 生成可直接跑，但
   輸出品質不可假設等同 16 kHz 調校水準，須抽查。
2. **驅動腳本陷阱**：`eval_aec_challenge.py --filter` 預設 2048、
   `run_one_case.py` 預設 832，皆為 16 kHz-authored 絕對 sample 數；
   48 kHz 須顯式覆寫（3072）或走 config 自動解析路徑。
   `run_one_case.py` 另須 `--sample-rate 48000`（不符會 fail-fast 報錯）。
3. C↔Python 的 48 kHz parity 差異（−23.8 dB、corr 0.99998，f32/f64
   Kalman 捨入放大所致）只影響跨語言 parity 測試與 C 部署驗證，
   Python-only 的 dataset 生成不受影響。
4. 未 retime 常數（`_simple_mu_ratio` 四常數、`erle_coh_gate_alpha`）為
   全 grid 共通的已記錄決策/待辦；48 k/1024 的 hop（10.667 ms）最接近
   原 authoring grid，漂移幅度（+4~7%）反而是各 grid 中最小。
