# AEC delay estimator 產品化實作計畫

狀態：規劃文件，尚未授權 push。  
範圍：`AEC/`、`Audio_ALG/pipelines/`、`Audio_ALG/pipelines/4ch_pipelines/`、`Audio_ALG/AIAEC/`。  
目標：讓產品能在初始化時選擇 delay 模式與 matched-filter bank 大小，同時讓 C static pool 的實際 RAM 隨設定縮小；另將 pre-echo 正確性修復與 PBFDKF delay controller 重新拆開驗證。

## 0. 不可混在一起的兩條工作線

### A. 產品化／算量與記憶體控制

- `delay_num_filters=1..5`。
- `MATCHED`／`FIXED`／`EXTERNAL_ALIGNED` 三種明確模式。
- mono、4ch、Python、C、CLI、AIAEC contract 全部接通。
- 預設 `MATCHED + n=5` 必須對既有穩定版本 byte-exact。
- 這一線不應包含 pre-echo、hysteresis 或 filter 行為調整。

### B. Pre-echo／applied-delay controller

- 修正 previous-winner prefix error 是正確性修復。
- 目前本地 `H=2 + N_RELEASE=16` 不符合 release gate，不可隨 A 線一起推送。
- 先完成 Python/C raw-delay trace parity，再設計 PBFDKF-aware safe-window controller。

## 1. 分支與 baseline 管理

1. 穩定比較基準固定為 AEC `5cd14a0`（已含 configurable bank size，但未含本地 pre-echo／guard 實驗）。
2. AEC 目前 `origin/main..HEAD` 的 8 個 commit 保留在獨立研究 branch，不 push、不更新 `Audio_ALG/lib/aec` pin。
3. A 線從穩定基準建立獨立 branch/worktree；B 線留在實驗 branch。
4. 每個結果標明基準 commit、config、grid、backend、SIMD、音檔 manifest hash。
5. 不得使用 `checkout/reset/stash` 遺失任何未追蹤或搬移檔；操作前先確認四個 repo 狀態。

## 2. 公開 delay mode API

### 2.1 C API

新增明確 enum，不以 `delay_num_filters=0` 暗示關閉：

```c
typedef enum AecDelayMode {
    AEC_DELAY_MATCHED = 0,
    AEC_DELAY_FIXED = 1,
    AEC_DELAY_EXTERNAL_ALIGNED = 2
} AecDelayMode;
```

`AecConfig` 新增或重整為：

```c
AecDelayMode delay_mode;
int delay_num_filters;       /* MATCHED only, 1..5 */
int fixed_delay_samples;     /* FIXED only, native-rate samples, >=0 */
```

驗證規則：

| mode | 合法欄位 | 行為 |
|---|---|---|
| `MATCHED` | `delay_num_filters=1..5` | 建構 estimator 與 reference alignment ring |
| `FIXED` | `fixed_delay_samples>=0` | 不建 estimator；用共用 delay ring 套用固定 delay |
| `EXTERNAL_ALIGNED` | 無 delay 參數 | 不建 estimator、不建 alignment ring；輸入 far 已對齊 |

- `delay_num_filters=0` 永遠是錯誤，不能成為靜默關閉入口。
- 不相容欄位組合 fail-fast；不得忽略使用者輸入。
- 預設為 `MATCHED + n=5`。
- mode 與 n 在 init 時決定，instance 存活期間 immutable；要改必須 reset-free destroy/re-init，因為 pool layout 不同。
- 若保留舊 `enable_delay_est`，只能作短期相容轉譯且不得成為第二個真實來源；release API 最終只保留一套語意。

### 2.2 Python API

Python `AecConfig` 使用相同三態與相同驗證。不得再以：

```python
fixed_delay_samples >= 0
```

隱式覆蓋 `enable_delay_est`。所有 mode 的 state、reset、reference ring 行為要和 C 一致。

### 2.3 Pipeline API

同步加入：

- `AudioPipelineConfig`（mono C）。
- `FourAecNrResConfig`（4ch core C）。
- `AudioPipeline4ChConfig`／ULCNet wrappers。
- Python `FourChannelAecConfig`／`SharedMatchedDelayEstimator`。
- CLI 參數與 usage 文件。

4ch 契約：

- `MATCHED` 永遠只有一個 shared estimator，不得每 lane 建一套。
- 四個 linear PBFDKF lane 的內部 delay estimator 維持關閉。
- `FIXED` 只保留一條 shared fixed-delay ring。
- `EXTERNAL_ALIGNED` 不配置 shared estimator/ring。

### 2.4 Signal grid 也是 init-time 必填契約

不能再只傳 `sample_rate` 就在 library 內猜 FFT，因為 16 kHz 有兩個正式
grid：

| sample rate | FFT / frame | hop |
|---:|---:|---:|
| 16 kHz（預設） | 256 | 128 |
| 16 kHz（alternate） | 512 | 256 |
| 48 kHz | 1024 | 512 |

公開的 AEC、NR、mono pipeline、4ch pipeline init config 都必須包含明確的
`sample_rate` 與 `fft_size`。在本專案固定 `frame_size=fft_size`、
`hop_size=fft_size/2`，由唯一一個 shared resolver 產生 resolved grid；
`get_mem_size()`、validation、init、process contract 必須使用同一份 resolved
結果。

- release-facing core init 不接受 `fft_size=0/auto`；16 kHz 的 256 與 512
  必須由 caller 明確選擇。
- 可以保留 convenience default factory，但它只能回傳一份完整 config
  （16 kHz 明確填 256、48 kHz 明確填 1024），不能讓核心 init 再猜一次。
- CLI 可以在使用者未指定時選產品預設，但呼叫 library 前必須先 resolve 成
  非零 `fft_size`。
- `fft_size`、delay mode、n 都是 init-time immutable；任何一項改變都必須
  重新查 memory requirement 並 re-init。
- internal `DelayAec3` 不需要假裝使用 FFT；它應接收 resolved
  `sample_rate + hop_size + num_filters`，用 hop 配置 edge/resample scratch。
- 三組 grid 都要有「get-mem-size 所見 grid == init 所見 grid == process hop」
  的 lockstep/mutation test。
- 清查並收斂所有只靠 sample rate 的 default 路徑，包括
  `aec_config_from_preset()`、NR `*_default_config*()`、mono/4ch
  `*_default_config()` 與 Python `resolve_signal_grid()`；default factory 可留，
  但不得成為第二套 hidden resolution chain。

## 3. C RAM 必須隨 init config 縮小

### 3.1 現況問題

目前 `DelayAec3` 直接內嵌下列最大尺寸陣列：

- `DaRing.buffer[DA_RING_CAPACITY]`。
- `DaMatchedFilter.filters[DA_NUM_FILTERS][DA_FILTER_SIZE]`。
- `DaMatchedFilter.accumulated_error[DA_NUM_FILTERS][DA_ACC_ERR_SIZE]`。
- highest-peak/pre-echo histogram 最大 bins。
- 48 kHz near/far scratch。

因此 n=1..5 現在只減少迴圈計算，不會等比例減少 `sizeof(Aec)`；estimator off 也只能省外部 reference ring，內嵌 estimator state 還在。

### 3.2 目標架構

維持 caller-owned static pool、零 heap，將 `DelayAec3` 改成小型 metadata/state + pool pointers：

```c
size_t delay_aec3_get_mem_size(
    int sample_rate, int hop_size, int num_filters);

int delay_aec3_init(
    DelayAec3* state,
    void* mem,
    size_t bytes,
    int sample_rate,
    int hop_size,
    int num_filters);
```

實際命名可依 repo 慣例調整，但只能有一套 canonical pool-first lifecycle。

必須由 pool 依 config 配置：

1. `n * DA_FILTER_SIZE` filter coefficients。
2. `n * DA_ACC_ERR_SIZE` accumulated-error rows。
3. 依 n 的實際最大 lag 配置 render ring。
4. 依 n 的實際最大 lag 配置 highest-peak/pre-echo histogram bins。
5. `DA_HIST_WINDOW` 是時間窗，不因 n 改變。
6. instantaneous prefix-error buffer 是單一 filter-size buffer，不因 n 改變。
7. 48 kHz anti-alias scratch 只在 48 kHz 配置；16 kHz 不保留兩份 48 kHz scratch。
8. `FIXED` 不配置 matched estimator，但配置足以涵蓋 fixed delay + hop/headroom 的 alignment ring。
9. `EXTERNAL_ALIGNED` 不配置 estimator 或 delay ring。

`Aec`／`FourAecNrRes` 本體只保留狀態與 pointers，不再內嵌最大 bank arrays。

### 3.3 Lockstep 與安全性

- `delay_aec3_get_mem_size()` 與 init carve 必須使用同一套 checked size helper。
- init 完成後 cursor 必須剛好用完所宣告的 bytes。
- 所有乘法、加法、alignment 做 overflow check。
- pool alignment 保持全專案統一值。
- 不得 fallback 到 heap。
- 不得為了 runtime 改 n 而預配 max=5；runtime 改 n 必須 re-init。
- bump AEC ABI/layout/descriptor version；更新 changelog 與 rebuild-required 說明。

### 3.4 RAM acceptance tests

永久測試至少驗證：

1. 同 grid 下 `mem(n=1) < mem(n=2) < ... < mem(n=5)`。
2. `MATCHED n=5` 的 audio output 對 refactor 前 byte-exact。
3. n=1..5 heap/caller-pool parity。
4. `FIXED mem < MATCHED n=1 mem`。
5. `EXTERNAL_ALIGNED mem < FIXED mem`。
6. mono 與 4ch `--print-mem-size` 列出 resolved grid、mode、n、estimator
   bytes、alignment-ring bytes。
7. 4ch estimator bytes 只出現一次，不是四次。
8. allocator-hook／symbol gate 證明所有 mode processing 期間零 allocation。
9. 故意改壞一個 size/carve 或 grid 傳遞公式時測試必須失敗
   （mutation test）。

不要在實作前寫死預期 KiB；先以實際結構公式產生 golden，再將三組正式 grid 的數字寫入文件。

## 4. Product config 傳遞與 AIAEC contract

### 4.1 Mono／4ch

- 上層 pipeline config 必須真的傳到 `aec_get_mem_size()`、AEC init、shared `DelayAec3` init。
- validation、get-mem-size、init 必須看到完全相同的 resolved config。
- descriptor/config hash 必須包含 `delay_mode`、`delay_num_filters`、`fixed_delay_samples`。
- unknown enum、n=0、n>5、negative fixed delay、mode/欄位衝突全部拒絕。

### 4.2 AIAEC

- `[linear_aec]` 加入 delay mode/n/fixed delay，預設仍為 `MATCHED + n=5`。
- materialization、training、denoise/inference 使用同一 resolved config。
- packed dataset/checkpoint contract 必須包含這三個欄位。
- behavior hash/schema 因新增欄位改版；不得讓舊 checkpoint 靜默載入不同 delay 行為。
- 若預設 n=5/matched 的 waveform 經 byte-equal gate 證明完全不變，可提供「只重蓋 contract、不重生 WAV」工具；否則必須重 materialize linear AEC。
- 產品實際部署 n<5 時，要用同 config 驗證 acquisition/path-change residual；不能只用 n=5 dataset 分數宣稱 train/deploy 一致。

## 5. Pre-echo 正確性與 trace parity

這一階段在 B 線執行，不能和第 2～4 節的 bit-exact refactor 混成同一 commit。

### 5.1 保留的正確性修復

prefix error 必須只由 previous winner filter 計算；synthetic two-arrival case 應得到正確的 `(peak_lag, onset_lag)`。修法需逐行對照 upstream `matched_filter.cc`。

### 5.2 先找 Python/C 分歧

建立同一輸入、同一 cadence、full-rate matched-filter 的逐 observation trace：

- filter index。
- 每個 filter 的 reliable/error_sum/peak lag。
- winner index／winner lag。
- prefix-error/pre-echo lag。
- highest-peak candidate。
- pre-echo histogram candidate。
- quality（coarse/refined）。
- raw reported delay。
- applied delay。
- reset/re-arm/duty 狀態。

驗證順序：

1. synthetic single arrival。
2. synthetic two arrival（early weak + late strong）。
3. onset amplitude sweep。
4. movement／delay step。
5. 目前 9 個 tail clips。
6. full-rate parity 後再打開 production duty cadence。

在 raw trace 的第一個 divergence 沒找到前，不調 H 或 release N。

## 6. 以 PBFDKF safe window 取代固定 H/N 實驗

目前 `H=2 + N_RELEASE=16` 不放行。下一個候選 controller 必須以 PBFDKF 的 causal tap reach 判斷，而不是只比較 `abs(new-current)<=H`。

定義：

```text
residual = raw_onset_delay - applied_bulk_delay
safe window = [front_headroom, filter_length - tail_margin]
```

候選語意：

1. residual 在 safe window 內：不 realign，忽略小 onset jitter。
2. applied 太晚、residual 接近/越過前界：快速往更早方向修正，避免 early echo 變成 non-causal。
3. applied 太早、residual 接近/越過後界：需 refined + 連續相同方向證據後往更晚方向修正。
4. 大幅超窗 path change：立即重定位。
5. 每個 held state 都必須有由幾何邊界保證的 exit；不得存在「raw 已長期穩定但 applied 永遠回不去」的隱式 latch。
6. small realign 優先 shift PBFDKF coefficients/state + output crossfade；只有大 path change 才完整 reset。
7. 所有 duration 以 seconds/ms authoring，依 16k/256/128、16k/512/256、48k/1024/512 retime；真正 event-count 才保留整數次數。

需先寫設計表列出每個狀態、enter/hold/exit、危險方向、reset 行為，再寫 code。

## 7. 產品選 n／mode 的量測方法

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

初始建議，不可取代實測：

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

可另立後續功能：acquisition 使用較大 n，solid 後 freeze 成 fixed/低 duty，遇 route-change、FIFO reset、clockdrift、ERLE collapse、divergence 才 re-arm。這不是本輪必要功能，且不能在沒有可靠 re-arm gate 前預設啟用。

## 8. 驗證矩陣

### 8.1 結構／bit-exact gate

- grid：16k/256/128、16k/512/256、48k/1024/512。
- C：KISS/NE10 × SIMD=0/1 × shadow on/off × CNG on/off。
- Python/C delay trace parity。
- mono/4ch static-vs-heap/caller-pool parity。
- n=5 refactor 前後 WAV SHA-256 byte-equal。
- 4ch matched-filter instance count 永遠 1。
- fixed/external modes不出現 matched-filter update。
- reset 後 ring、generation token、delay state 正確。

### 8.2 Audio gate

- far-end static/movement。
- double-talk static/movement。
- near-end-only。
- delay step/path change。
- clock drift。
- short clips、非 hop 整除 tail。
- native 48 kHz recordings，不只升採樣素材。
- AECMOS echo/deg、ERLE/residual RMS、applied-delay switches、acquisition time、deep-regression counts。
- 新 controller 的最終確認必須用未參與 H/N/safe-window 設計的新 holdout；舊 832-case 只算 development evidence。

### 8.3 Release hard stops

以下任一成立即不 push：

- n=5 bit-exact gate 失敗（純 productization branch）。
- C/Python raw-delay trace 未定位分歧。
- 任一 mode 有隱式 fallback。
- get-mem-size/carve 不 lockstep。
- processing path allocation。
- 4ch 不只一個 shared estimator。
- controller 存在無 exit 的 latch。
- aggregate 通過但 deep-regression/tail 超門檻。
- AIAEC contract 與實際 deployed delay config 不一致。

## 9. 建議 commit／push 順序

每步獨立 commit、每步先測再進下一步：

1. AEC：delay mode config + validation，尚不改 memory layout。
2. AEC：pool-first variable-size `DelayAec3`，n=5 byte-exact。
3. AEC：FIXED／EXTERNAL alignment state與測試。
4. AEC：CLI、memory diagnostics、文件與 ABI/version。
5. push AEC。
6. Audio_ALG：更新 `lib/aec` pin。
7. Audio_ALG：mono pipeline config passthrough。
8. Audio_ALG：4ch one-shared-estimator mode/n/fixed/external。
9. Audio_ALG：Python 4ch parity、AIAEC contract/tooling。
10. 完整 pipeline/AIAEC tests 後 push Audio_ALG。
11. 另開 B 線 commit：pre-echo winner fix與 trace harness。
12. trace parity 完成後才實作 safe-window controller；通過新 holdout 才另行決定是否推送及 bump AIAEC behavior contract。

不得先 bump `Audio_ALG/lib/aec` 到尚未 push 的 standalone commit。

## 10. 最終交付物

- 三態 delay API 與使用範例。
- C/Python/mono/4ch 完整 config parity tests。
- 各 n、mode、grid 的實測 static-pool bytes 與 MAC/RTF 表。
- 產品選型表與 residual-delay 量測工具。
- raw-delay trace comparison report。
- PBFDKF safe-window controller 設計與 A/B 報告。
- AIAEC dataset/checkpoint contract migration 說明。
- CHANGELOG、C user manual、pipeline README、HTML 文件站同步更新。

## 11. Review 附錄（2026-08-16 Claude 審查補充，執行時一併遵守）

1. **AecLinearContext seam 的逐 mode 語意必須明訂**（§2 的缺口；NN pipeline 靠它做 fail-open）：
   `MATCHED`=現行語意不變；`FIXED`=delay_samples=固定值、delay_state=LOCKED、
   confidence=1.0、generation 恆定（init 後不再 bump）；`EXTERNAL_ALIGNED`=
   aligned_far_hop 即輸入 far、delay_samples=0、state=LOCKED、generation 恆定。
   三者都要進 seam 測試。
2. **8 kHz grid 要明確表態**：resolver 的正式 grid 表只列 16k×2＋48k；repo 現有
   8 kHz 路徑（E2E parity、1216 ms 上限表）。決定：8k/256/128 列為 resolver 支援
   的 legacy grid（非產品正式 grid），或明文 deprecate；不得留在「猜」的路徑裡。
3. **NR repo 是否入列**：§2.4 清查清單提到 NR `*_default_config*()`，但 §範圍 未列
   NR。決定：NR 的 grid 顯式化納入 A 線範圍（動 NR repo），或本輪只清 AEC/
   pipeline，NR 另開工作項。不得曖昧。
4. **研究 branch 手術與工具重落地**：8 個 local commit 移至 research branch 後，
   其中兩個中性工具是 A 線需要的——`--delay-num-filters` CLI（113e09e，與
   --delay-hysteresis 同 commit、糾纏 B 線 config 欄位）與 duty census 診斷
   （505d2b3，§7 量測方法要用）。A 線在 §9 step 4 **重新實作**這兩者（不
   cherry-pick，避免把 B 線欄位拖進來）。
5. **§5 trace parity 的 f32 預期**：C delay 鏈是刻意 f32（Python f64），數值
   parity 已退役——「第一個 divergence」很可能是 argmax tie/threshold 邊緣的
   良性 f32 差異。trace harness 必須能分類 benign-f32 vs 結構性分歧（例如同
   observation 的整數決策全同、僅 error_sum 尾數不同=benign），否則會在假分歧
   上卡死。
6. **dataset 重生時點**：A 線預設 byte-exact → aec_behavior_hash 仍因 AST 變動而
   bump，但走 §4.2 的「只重蓋 contract、不重生 WAV」工具（前提=byte-equal gate
   實證）。**AIAEC 正式 dataset 生成排在 A 線落地之後即可開始**（用 MATCHED
   n=5），不必等 B 線——B 線 controller 若日後放行，才觸發真正的 re-materialize。
7. **§3 refactor 的 byte-exact 風險點**（執行提醒）：陣列從內嵌改 pool-carve 時，
   每個 f32 陣列一律 ALIGN16（NEON kernel 對 512-tap 陣列的對齊假設）；init/
   memset 順序不得重排；n=5 SHA byte-equal gate 在 KISS/NE10 × SIMD 0/1 全矩陣跑。
8. 舊 `feature/delay-prescan-anchor` branch（6fb2abc，為 research branch 祖先）
   刪除，避免雙 branch 混淆。
