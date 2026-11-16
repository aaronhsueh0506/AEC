# C ↔ Python 函式對照表（AEC）

> 範圍：`c_impl/src` 全部（28 個 `.c`）加 `c_impl/include/internal` 標頭內的 static inline 重要函式；Python 側為 `python/aec.py` 與 `python/modules/**`（唯讀，受行為雜湊保護，本文件與其測試都不修改它們）。

## 0. 用途、讀法、維護

### 0.1 用途

演算法在 Python 端開發，C 要先和 Python 的效果同步，之後才能優化或改寫。本表回答：每個 C 函式對到哪個 Python 檔案的哪個符號、兩者關係是什麼、已知有哪些落差、有哪些對拍工具或測試。

注意 C 與 Python 的名稱不對稱：Python 側仍帶 `aec3`，C 側沒有：`AEC._aec3_post`、`AEC._aec3_select_linear_filter_output`、`AEC._fire_aec3_misadj_scale`、`AEC._reset_aec3_post`、`PBFDKF._update_weights_aec3`、`modules/aec3_scale.py`、`ReverbDecayEstimator._lazy_init_aec3` 等。這份表記下的就是這種追溯資訊。強度軸的命名也不對稱：C 稱 **level**（`AecLevel`、`aec_set_level`、`aec_config_from_level`），Python 稱 **preset**（`AecPreset`、`AEC.set_preset`、`AecConfig.from_preset`）；**C level == Python preset**，名稱與數字（0..3）相同。第 1 欄寫 C 名稱，第 2 欄寫 Python 名稱。

### 0.2 欄位與符號語法

每張表都是同樣五欄、同樣的標題列（測試以標題文字定位欄位，不靠欄位順序）。

| 欄位 | 寫法 |
|---|---|
| C 檔:函式 | `src/<檔>.c:<函式>`、`include/internal/<檔>.h:<函式>` 或 `include/<檔>.h:<函式>`，路徑相對於 `c_impl/`。§3 的 Python-only 列此欄為 `—`，或 `struct:<C 型別名>`（表示 C 以該型別承載）。 |
| Python 檔:符號 | `` `<路徑>:<符號>` ``，路徑相對於 `python/`，符號是點分隔的 `類別.方法` 或模組層函式／類別名。一格可有多個，以 `<br>` 分隔；`—` 表示範圍內沒有。可在符號後加 `#錨點字串`：表示 C 函式只對應該符號內的一段（例如部分方法內的局部計算只對應某個 C 函式），測試會驗證錨點字串確實出現在該符號的原始碼裡。**符號名為準，行號不寫**（行號會漂移）。 |
| 關係 | 開頭是下列關鍵字之一，後面可接括號說明：`mirror`＝同一運算單元、同數學（數值差異另見落差欄）；`partial`＝只對應一部分，或 C 與 Python 拆分的方式不同，括號內寫缺什麼；`C-only`＝範圍內沒有對應的 Python 函式（記憶體、SIMD 核心、除錯、測試鉤子等 C 專屬基礎設施）；`Python-only`＝範圍內沒有任何 C 函式對應到這個 Python 符號；`未能判斷`＝證據不足，不硬湊。括號說明以「推定：」開頭者，表示對應是讀兩邊程式碼後得出，註解與函式名都沒有直接說明，動到該函式時應複核。 |
| 已知落差 | 落差編號，`；` 分隔，`—` 表示沒有已登記的落差。編號意義見附錄 A。EXPECTED_RED(S-xx) 標記應引用這一欄的編號。 |
| 對拍工具／測試 | 標記語法見附錄 B（`live:`、`e2e:`、`manual:`、`ctest:`、`pytest:`）。`—` 表示沒有直接對拍或專屬測試，只被端到端間接覆蓋。 |

### 0.3 計數規則（逐字，先於數字）

1. **C 函式 `src/*.c`**：以「(檔, 函式名)」為單位，取檔案層級（depth 0）的函式定義，`static` 與非 `static` 都算。同一檔內因 `#if` 分支出現兩次的同名函式只算一個（NEON 與純量分支），括號說明標註「雙胞」；本檔共 5 個：`da_dot_f32`、`da_nlms_update_f32`、`da_max_square_peak_index`、`da_decimator_decimate_pair`、`da_resample48_pair`。`AEC_TESTING` 才編譯的測試鉤子（`aec_testing_*`）算，標為 C-only／測試專用。
2. **header 內 static inline**：`include/internal/aec_balanced_config.h` 與 `include/internal/pbfdkf.h` 的全部 static inline 函式都算（統計歸入 header 欄）；共用 `utility/simd_aec.h` 的函式依規則 3 排除，由 Audio_ALG 對照表統計。這些 header 入口全部是 C-only。
3. **範圍外不算**：`test/`、`example/` 下的 C 檔、`audiolib` 的函式（例如 HPF、FFT、重取樣）、`Audio_ALG/pipelines` 內的 Python 孿生（例如 4ch 後波束 RES、`cng_aec3.py`）。這些在相關列的括號說明或 §3 以文字指出，但不列入統計，測試也不驗證它們。
4. **Python 符號**：`python/aec.py` 與 `python/modules/**` 內以 `ast` 取得的 class、函式、方法（含 `__init__`），不含函式內的巢狀 closure，不匯入模組。
5. **已對應**＝關係為 `mirror` 或 `partial`。每個 C 函式恰好在 §1／§2 出現一列；每個 Python 符號要嘛被至少一列的 Python 欄引用，要嘛列在 §3，兩者互斥且合起來等於 Python 符號總數。

### 0.4 涵蓋率統計

下表由測試重算並比對（`test_stats_block_matches_tables`）；改了對照表要同步改這裡。

<!-- stats:begin -->
| key | 數值 | 說明 |
|---|---|---|
| `c_src_total` | 514 | `src/` 下 `.c` 的 C 函式總數（唯一 (檔, 函式名)） |
| `c_src_mirror` | 221 | 關係為 mirror |
| `c_src_partial` | 124 | 關係為 partial |
| `c_src_c_only` | 169 | 關係為 C-only（含 22 個 `aec_testing_*` 測試鉤子） |
| `c_src_unknown` | 0 | 關係為 未能判斷 |
| `c_hdr_total` | 2 | `include/internal/` 下 header 內 static inline 函式總數（規則 2；共用 SIMD 核心不列入） |
| `c_hdr_c_only` | 1 | 其中 C-only |
| `c_hdr_mapped` | 1 | 其中 mirror 或 partial |
| `c_total` | 516 | C 函式合計（src 加 header） |
| `c_mapped` | 346 | 已對應（mirror 加 partial）合計 |
| `c_c_only` | 170 | C-only 合計 |
| `c_unknown` | 0 | 未能判斷 合計 |
| `inferred` | 28 | 關係說明以「推定：」開頭的列（已含在上列 mirror／partial 內） |
| `py_total` | 575 | Python 符號總數（規則 4） |
| `py_mapped` | 322 | 被 §1／§2 至少一列引用的 Python 符號 |
| `py_only_struct` | 58 | §3 Python-only：C 以資料結構、列舉或常數承載 |
| `py_only_field` | 58 | §3 Python-only：C 以結構欄位或回傳值直接取得 |
| `py_only_none` | 137 | §3 Python-only：C 無此機制 |
| `py_unknown` | 0 | §3 未能判斷 |
<!-- stats:end -->

### 0.5 未能判斷

沒有留空的函式或符號：每個 C 函式都能由名稱一一對應或讀程式碼比對判定。判定的依據強度不同，其中 29 列標為「推定」（關係說明以「推定：」開頭），是這份表最該被複核的部分；其餘列要嘛有 C 或 Python 註解明載，要嘛函式名與方法名一一對應。

### 0.6 如何維護

1. 增、刪、改名 C 函式或 Python 符號之後，直接改本檔對應的列（沒有產生器，本檔就是唯一來源）。
2. 在 `python/` 的上層目錄執行 `python3 -m pytest python/tests/test_c_python_map.py`。測試不匯入任何模組，用 `ast` 讀 `python/aec.py` 與 `python/modules/**`，用內建的小型 C 掃描器讀 `c_impl/src`、`c_impl/include`。失敗訊息會指出是哪一列、哪個符號。
3. 測試依賴本檔的標題列與關係關鍵字；新增標記語法（例如日後的 `pyc:`）要同時改附錄 B 與測試。
4. 新增一個 C 函式：在 §1 對應檔的表加一列，並更新 §0.4 的統計。新增 Python 符號：若有 C 函式對應就在該列引用；沒有就加進 §3。
5. 這份表在 `AEC` 與 `Audio_ALG/lib/aec` 是同一份檔案、同一份測試（`lib/aec` 是鏡像），兩處要一起改。測試用自己所在位置推出 `python/` 與 `c_impl/`，所以原樣複製即可在兩處通過；兩處的 Python 符號集合與 C 函式集合完全相同。

### 0.7 與 `pyc_parity_aec.sh` 的關係

`c_impl/test/pyc_parity_aec.sh`（`make test-pyc-parity`）是 Python↔C 數值對拍腳本；端到端閘門是 `make test-parity-e2e`（只在 `make test-full` 執行）。`make test-pyc-parity` 預設跑 10 個場景的精簡集，`PYC_ARGS=--full` 跑全部場景。分工是：
該腳本以「情境」為單位量 C 對 Python 的數值差，兩層標準（std-math 建置貼近 Python、出貨建置落在固定近似預算內），並以 `EXPECTED_RED(S-xx)` 標記已知落差（登記在 `c_impl/test/pyc_parity/pyc_parity.py`）；本表以「函式」為單位回答「這段 C 對應 Python 哪裡、有哪些已登記落差」。情境紅燈時用本表由 C 函式反查 Python 符號，`EXPECTED_RED(S-xx)` 的編號則引用本表「已知落差」欄與附錄 A。本表不量任何數值，也不取代該腳本。

---

## 1. 逐檔對照（`src/*.c`）

### 1.1 `aec.c`　aec.c（頂層引擎：設定、記憶體、逐 hop 主迴圈、串流 API）

依據：Python 的 `AEC.process` 按 capture、reference alignment、linear filter、RES 與診斷順序呼叫私有 helper（`_update_simple_mu_ratio`、`_fire_aec3_misadj_scale`、`_reset_filter_derived_state`、`_reset_aec3_post`、`_get_simple_mu_scale`、`_update_misadjustment_estimator` 等）；下表對應實際執行該步驟的方法，必要時以 `#錨點字串` 指出局部計算。

| C 檔:函式 | Python 檔:符號 | 關係 | 已知落差 | 對拍工具／測試 |
|---|---|---|---| --- |
| `src/aec.c:aec_legacy10ms_hops` | — | C-only（Python 在呼叫處直接用 `ms_to_hops` 重新取整，沒有獨立函式） | — | `ctest:test_rate_structural.c` |
| `src/aec.c:aec_legacy10ms_alpha` | — | C-only（Python 在呼叫處直接用 `growth_rehop`） | — | `ctest:test_rate_structural.c` |
| `src/aec.c:aec_legacy10ms_rate` | — | C-only（加法型逐 hop 速率的線性換算；C 註解載明 Python 側尚無對應的共用 helper） | — | `ctest:test_rate_structural.c` |
| `src/aec_tuning.c:aec_level_row` | — | C-only（`aec_level_floors` 與 `aec_level_name` 共用的列查找；Python 以 `AecConfig.from_preset` 內的分支取得同樣數值） | — | `ctest:test_runtime_preset.c` |
| `src/aec_tuning.c:aec_level_name` | — | C-only（等級名稱字串，供 CLI 與 log 顯示；列舉之外回 "balanced"，與 `aec_config_from_level` 的處理相同） | — | `ctest:test_runtime_preset.c` |
| `src/aec_tuning.c:aec_level_floors` | `modules/config.py:AecConfig.from_preset` | partial（一次回 far-active 地板 dB、DT 地板 dB 與 split-floor 閂；Python 以 `from_preset` 內的分支取得同樣數值） | D-04 | `ctest:test_runtime_preset.c` |
| `src/aec_tuning.c:aec_config_from_level` | `modules/config.py:AecConfig.from_preset` | mirror（C 只存 `level` 數字，三個 floor 值由 `aec_tuning_defaults` 查表帶出，缺 Python-only 旋鈕；NO_FLOOR 雙邊建構期都有；不支援的取樣率 C 回 `fft_size`＝0、派生的 `filter_length`＝0 並由建立入口拒絕，Python 在 `__post_init__` 直接 `ValueError`） | S-02；S-22；D-02；D-04；D-06；D-07 | `ctest:test_runtime_preset.c`<br>`ctest:test_config_validation.c` |
| `src/aec_tuning.c:aec_find_grid` | — | C-only（格點表查找：取樣率加 fft_size，fft_size 不大於 0 時任一列；`aec_resolve_signal_grid`、`aec_is_valid_sample_rate`、`aec_default_fft_size` 共用） | — | `ctest:test_config_validation.c` |
| `src/aec_tuning.c:aec_resolve_signal_grid` | `modules/config.py:resolve_signal_grid` | mirror | — | `pytest:test_signal_grid.py`<br>`ctest:test_config_validation.c` |
| `src/aec_tuning.c:aec_is_valid_sample_rate` | `modules/config.py:resolve_signal_grid`<br>`modules/config.py:default_fft_size` | partial（推定：Python 沒有獨立的「取樣率是否合法」函式，由格點表比對隱含） | — | `ctest:test_config_validation.c` |
| `src/aec_tuning.c:aec_default_fft_size` | `modules/config.py:default_fft_size` | mirror | — | `pytest:test_signal_grid.py`<br>`ctest:test_config_validation.c` |
| `src/aec.c:aec_ring_active` | `modules/orchestrator.py:AEC._init_delay#_delay_active` | mirror（C 註解：Mirrors Python `_delay_active`） | — | — |
| `src/aec.c:aec_resolved_valid` | `modules/config.py:AecConfig.__post_init__` | partial（公開欄位、level、延遲模式矩陣與陣列規則；內部調校值的範圍檢查在 `aec_tuning_valid`；C 嚴格 fail-closed，Python 幾乎不驗） | S-34；D-08 | `ctest:test_config_validation.c` |
| `src/aec_tuning.c:aec_tuning_valid` | `modules/config.py:AecConfig.__post_init__` | partial（內部調校值的範圍檢查；HPF cutoff 範圍僅 C 檢查，Python 幾乎不驗） | S-04；S-34；D-08 | `ctest:test_config_validation.c` |
| `src/aec.c:mean_sq` | — | C-only（`np.mean(x**2)` 的 float32 等價實作；Python 直接呼叫 numpy） | — | — |
| `src/aec.c:aec_counter_bump` | — | C-only（hop／呼叫計數器的飽和遞增；Python 的整數不會溢位） | — | `ctest:test_counter_saturation.c` |
| `src/aec.c:aec_count_as_int` | — | C-only（`long` 計數器飽和轉 `int`；Python 的整數不會溢位） | — | `ctest:test_counter_saturation.c` |
| `src/aec.c:aec_rearm_excitation` | `modules/orchestrator.py:AEC._rearm_filters_for_delay_change`<br>`modules/filters.py:PBFDAF.handle_echo_path_change` | partial（只處理 PBFDAF 型濾波器；主濾波器走 `pbfdkf_handle_echo_path_change`） | D-10 | — |
| `src/aec.c:aec_reset_filter_latches` | `modules/orchestrator.py:AEC._reset_filter_latches` | mirror | — | `ctest:test_rate_structural.c` |
| `src/aec.c:aec_reset_filter_derived_state` | `modules/orchestrator.py:AEC._reset_filter_derived_state` | mirror | — | — |
| `src/aec.c:aec_delay_value_rejected` | `modules/orchestrator.py:AEC._track_delay_shift#_delay_reject_values` | partial（C 的 reject list 固定 8 格，Python 為無上限 list） | D-03 | — |
| `src/aec.c:aec_apply_delay_shift_hard_reset` | `modules/orchestrator.py:AEC._apply_delay_shift_hard_reset` | mirror | — | — |
| `src/aec.c:aec_apply_delay_shift_soft_revert` | `modules/orchestrator.py:AEC._apply_delay_shift_soft_revert` | mirror（bad-relock 還原：回到 shift 前的 delay，還原 main 的 W 與 `H_error_per_bin`、shadow 的 W，不走硬 reset 鏈，所以 `aec_post_reset_serial()` 不動、OLA 與 CNG 照常延續；退後（target < bad）清遠端歷史，前進則保留；windowed-ERLE 累積、main／shadow 誤差 EMA 與待決 delay 清除。C 在 2026-10-28 才移植，C 預設路徑的輸出因此在有還原的 hop 之後改為與 Python 相同） | — | `ctest:test_guard_unify.c`<br>`pytest:test_guard_unify.py` |
| `src/aec.c:aec_reset_filter_for_realign` | `modules/orchestrator.py:AEC._reset_filter_for_realign` | mirror | — | `ctest:test_external_realign.c` |
| `src/aec.c:get_simple_mu_scale` | `modules/orchestrator.py:AEC._get_simple_mu_scale` | mirror | — | — |
| `src/aec.c:advance_erl_and_dt` | `modules/orchestrator.py:AEC._compute_residual_steering#_erl_estimate`<br>`modules/orchestrator.py:AEC._update_linear_only_steering` | partial（process 內兩處呼叫點共用的 ERL 追蹤加能量 DT 更新） | — | — |
| `src/aec.c:update_simple_mu_ratio` | `modules/orchestrator.py:AEC._update_simple_mu_ratio` | mirror | — | `pytest:test_simple_mu_holdoff_f24.py` |
| `src/aec.c:aec_testing_update_simple_mu_ratio` | — | C-only（測試專用鉤子，只在 `AEC_TESTING` 建置存在） | — | `ctest:test_rate_structural.c` |
| `src/aec.c:aec_testing_reset_filter_latches` | — | C-only（測試專用鉤子，只在 `AEC_TESTING` 建置存在） | — | `ctest:test_rate_structural.c` |
| `src/aec.c:aec_testing_legacy10ms_rate` | — | C-only（測試專用鉤子，只在 `AEC_TESTING` 建置存在） | — | `ctest:test_rate_structural.c` |
| `src/aec.c:aec_testing_track_delay_shift` | — | C-only（測試專用鉤子，只在 `AEC_TESTING` 建置存在：對單聲道 Path B 提交一個指定的 delay 估計，以測兩段式確認的帶寬、確認窗與候選壽命） | — | `ctest:test_delay_admission.c` |
| `src/aec.c:aec_testing_tuning_defaults` | — | C-only（測試專用鉤子，只在 `AEC_TESTING` 建置存在：填入函式庫為該 `AecConfig` 帶出的內部 `AecTuning`） | — | `ctest:test_config_validation.c` |
| `src/aec.c:aec_testing_tuning_valid` | — | C-only（測試專用鉤子，只在 `AEC_TESTING` 建置存在：內部調校值的範圍檢查） | — | `ctest:test_config_validation.c` |
| `src/aec.c:aec_testing_get_mem_size` | — | C-only（測試專用鉤子，只在 `AEC_TESTING` 建置存在：以測試自備的 `AecTuning` 查記憶池大小） | — | `ctest:test_memory_layout.c` |
| `src/aec.c:aec_testing_get_mem_requirements` | — | C-only（測試專用鉤子，只在 `AEC_TESTING` 建置存在：以測試自備的 `AecTuning` 填 AecMemReq） | — | `ctest:test_aec_array_flow.c` |
| `src/aec.c:aec_testing_init` | — | C-only（測試專用鉤子，只在 `AEC_TESTING` 建置存在：以測試自備的 `AecTuning` 在呼叫端 pool 建構） | — | `ctest:test_memory_layout.c` |
| `src/aec.c:aec_testing_create` | — | C-only（測試專用鉤子，只在 `AEC_TESTING` 建置存在：以測試自備的 `AecTuning` 一次配置建構） | — | `ctest:test_rate_structural.c` |
| `src/aec.c:aec_fill_filter_taps` | — | C-only（FilterAnalyzer 的 tap provider 回呼；Python 直接傳 `get_time_domain_filter()` 結果） | — | — |
| `src/aec.c:aec_state_bind` | — | C-only（把 AecState 設定與初始化集中在一處，供 `aec_carve_state` 與 `aec_post_chain_reset` 共用） | — | `ctest:test_memory_layout.c`<br>`pytest:test_aec_reset.py` |
| `src/aec.c:aec_ree_bind` | — | C-only（把殘餘回音估計器的設定與初始化集中在一處，供 `aec_carve_residual` 與 `aec_post_chain_reset` 共用） | — | `ctest:test_memory_layout.c`<br>`pytest:test_aec_reset.py` |
| `src/aec.c:aec_post_chain_reset` | `modules/orchestrator.py:AEC._reset_aec3_post` | mirror（對應 preserve_render_side=True 的路徑） | — | `pytest:test_aec_reset.py` |
| `src/aec.c:aec_create` | `modules/orchestrator.py:AEC.__init__`<br>`modules/orchestrator.py:AEC._init_config` | partial（推定：C 另做驗證與記憶體配置：一次配置 `aec_get_mem_size()` 位元組，以 `aec_init` 建構；回傳實例指標） | D-06；D-07 | `ctest:test_config_validation.c`<br>`ctest:test_heap_alignment.c` |
| `src/aec.c:aec_ref_ring_samples` | `modules/config.py:ref_ring_samples` | partial（FIXED／EXTERNAL_ALIGNED 同式；MATCHED 的 C 環為 estimator 可回報延遲上限＋hop，Python 為 max(delay_buffer_ms, max_delay_ms + 4096)） | D-16 | `ctest:test_linear_context.c`<br>`ctest:test_delay_num_filters.c` |
| `src/aec.c:aec_derive_dims` | `modules/config.py:resolve_signal_grid`<br>`modules/config.py:AecConfig.n_partitions`<br>`modules/config.py:AecConfig.fft_size` | partial（推定：幀幾何走同一個格點解析；C 的 n_partitions 可覆寫） | D-11 | `ctest:test_config_validation.c` |
| `src/aec.c:aec_get_mem_size` | — | C-only（靜態記憶體池尺寸查詢） | — | `ctest:test_static_aec.c` |
| `src/aec.c:aec_get_mem_breakdown` | — | C-only（靜態記憶體池分項；回 `AecStatus`：0 = 成功，`AEC_ERR_ARG` = cfg 無效） | — | `ctest:print_mem_size_ref.c` |
| `src/aec.c:aec_recompute_wallclock_thresholds` | `modules/orchestrator.py:AEC._update_shadow_leakage#_poor_coarse_counter` | partial（C 把建構與 reset 共用的牆鐘門檻重算收成一個函式；Python 在 `process` 內逐 hop 即時以 `blocks_to_hops`／`ms_to_hops` 計算） | — | `ctest:test_reset_parity.c` |
| `src/aec_config_build.c:aec_build_sg_config` | `modules/orchestrator.py:AEC._build_sg_kwargs` | partial（C 不帶 last_lf_band／first_hf_band；C 同時供 `aec_res` 使用；ENR／SU 表留給 `aec_sg_tables_decode` 解碼，EMR 表指向靜態資料） | — | — |
| `src/aec_config_build.c:aec_synth_window_build` | `modules/orchestrator.py:AEC._init_post_synthesis#_aec3_synth_window` | partial（值逐位元相同：Python 以 sqrt-Hann 公式（double 運算再轉 float32）建窗，C 由標頭的半窗表以鏡射與抽樣重建，不呼叫 libm；任一白名單長度都由同一張半窗表而來） | — | `ctest:test_synthesis_window.c`<br>`ctest:test_rate_structural.c` |
| `src/aec_config_build.c:aec_sg_tables_bytes` | — | C-only（四張解碼後 ENR／SU 表的池位元組數，每張補到 16 位元組；單通道實例與增益階段共用） | — | `ctest:test_memory_layout.c` |
| `src/aec_config_build.c:aec_balanced_rate_dims` | — | C-only（依 (sample_rate, fft_size) 取生成表 `AECB_RATE_TABLE` 的調校列；表只在 `aec_config_build.o` 實體化，其餘翻譯單元經此讀取） | — | `ctest:test_config_validation.c`<br>`ctest:test_rate_structural.c` |
| `src/aec_config_build.c:aec_sg_tables_expand` | — | C-only（把費率列的 run-length 碼展開成四張逐 bin 表，值與 Python `_build_gain_params` 逐位元相同；對拍表在生成器輸出的 `test/aec_ref_tables.h`） | — | `ctest:test_config_validation.c`<br>`ctest:test_aec_res.c` |
| `src/aec_config_build.c:aec_sg_tables_decode` | — | C-only（依已解析設定查費率列，把四張 ENR／SU 表解碼進呼叫端的池並設定 tuning 的指標；陣列的增益階段使用） | — | `ctest:test_aec_res.c` |
| `src/aec_tuning.c:aec_resolve_tuned` | — | C-only（驗證 `AecConfig` 加 `AecTuning` 並寫入 `AecResolved`；tuning 為 NULL 時用 `aec_tuning_defaults`；公開入口走 `aec_resolve`，測試鉤子帶自己的 tuning） | — | `ctest:test_config_validation.c` |
| `src/aec_tuning.c:aec_resolve` | — | C-only（`aec_resolve_tuned` 的 tuning＝NULL 形式：由 (sample_rate, fft_size, n_mics, level) 帶出全部內部調校，所有公開尺寸查詢、建構與 AecRes 入口的單一解析點） | — | `ctest:test_config_validation.c` |
| `src/aec_tuning.c:aec_tuning_defaults` | `modules/config.py:AecConfig` | partial（內部調校欄位的預設值，與 Python `AecConfig` 同名同值（`warmup_frames`、`delay_est_period_s` 見 D-02 與 `from_preset`）；floor 三欄由 level 查表；缺 Python-only 旋鈕） | D-02；D-04；D-06 | `ctest:test_config_validation.c`<br>`ctest:test_runtime_preset.c` |
| `src/aec_tuning.c:aec_apply_level` | `modules/orchestrator.py:AEC.set_preset` | partial（單通道與陣列的 `aec_set_level` 共用：寫入 `AecConfig.level` 與該 level 的三個 floor 值；level 不合法時什麼都不寫；執行期 ramp 在增益 stage 內） | D-04 | `ctest:test_runtime_preset.c` |
| `src/aec.c:aec_acquire_delay` | `modules/orchestrator.py:AEC._acquire_delay` | partial（逐 hop 子步驟；保留原始先後順序與 scratch 生命週期） | — | `ctest:test_process_context.c` |
| `src/aec.c:aec_align_reference` | `modules/orchestrator.py:AEC._align_reference` | partial（逐 hop 子步驟；保留原始先後順序與 scratch 生命週期） | — | `ctest:test_process_context.c` |
| `src/aec.c:aec_carve_delay_ring` | `modules/orchestrator.py:AEC._init_delay` | partial（建構步驟；C 綁定既有 pool 切片） | — | `ctest:test_memory_layout.c` |
| `src/aec.c:aec_carve_detectors` | `modules/orchestrator.py:AEC._init_path_tracking`<br>`modules/orchestrator.py:AEC._init_preprocessing` | partial（建構步驟；C 綁定既有 pool 切片） | — | `ctest:test_memory_layout.c` |
| `src/aec.c:aec_carve_filters` | `modules/orchestrator.py:AEC._init_linear_filter`<br>`modules/orchestrator.py:AEC._init_shadow_filter`<br>`modules/orchestrator.py:AEC._init_observer` | partial（建構步驟；C 綁定既有 pool 切片） | — | `ctest:test_memory_layout.c` |
| `src/aec.c:aec_carve_hop_scratch` | `modules/orchestrator.py:AEC.__init__` | partial（建構步驟；C 綁定既有 pool 切片） | — | `ctest:test_memory_layout.c` |
| `src/aec.c:aec_carve_linear_select` | `modules/orchestrator.py:AEC._init_output_state` | partial（建構步驟；C 綁定既有 pool 切片） | — | `ctest:test_memory_layout.c` |
| `src/aec.c:aec_carve_post_backing` | `modules/orchestrator.py:AEC._init_post_synthesis` | partial（建構步驟；C 綁定既有 pool 切片） | — | `ctest:test_memory_layout.c` |
| `src/aec.c:aec_carve_post_scratch` | `modules/orchestrator.py:AEC.__init__` | partial（建構步驟；C 綁定既有 pool 切片） | — | `ctest:test_memory_layout.c` |
| `src/aec.c:aec_carve_residual` | `modules/orchestrator.py:AEC._init_suppression_chain` | partial（建構步驟；C 綁定既有 pool 切片） | — | `ctest:test_memory_layout.c` |
| `src/aec.c:aec_carve_state` | `modules/orchestrator.py:AEC._init_suppression_chain` | partial（建構步驟；C 綁定既有 pool 切片） | — | `ctest:test_memory_layout.c` |
| `src/aec.c:aec_carve_stationarity` | `modules/orchestrator.py:AEC._init_render_analysis` | partial（建構步驟；C 綁定既有 pool 切片） | — | `ctest:test_memory_layout.c` |
| `src/aec.c:aec_carve_suppression` | `modules/orchestrator.py:AEC._init_suppression_chain` | partial（建構步驟；C 綁定既有 pool 切片） | — | `ctest:test_memory_layout.c` |
| `src/aec.c:aec_finish_hop` | `modules/orchestrator.py:AEC._track_erle_convergence` | partial（逐 hop 子步驟；保留原始先後順序與 scratch 生命週期） | — | `ctest:test_process_context.c` |
| `src/aec.c:aec_init_config_state` | `modules/orchestrator.py:AEC._init_config` | partial（建構步驟；C 綁定既有 pool 切片） | — | `ctest:test_memory_layout.c` |
| `src/aec.c:aec_reset_scalar_state` | `modules/orchestrator.py:AEC._init_path_tracking`<br>`modules/orchestrator.py:AEC._init_output_state`<br>`modules/orchestrator.py:AEC.reset` | partial（推定：建構與 `aec_reset` 共用的逐 hop 純量狀態起始值；Python 兩處各寫一份） | — | `ctest:test_reset_parity.c`<br>`ctest:test_memory_layout.c` |
| `src/aec.c:aec_init_scalar_state` | `modules/orchestrator.py:AEC._init_path_tracking`<br>`modules/orchestrator.py:AEC._init_output_state` | partial（建構步驟；C 綁定既有 pool 切片） | — | `ctest:test_memory_layout.c` |
| `src/aec.c:aec_judge_delay_shift` | `modules/orchestrator.py:AEC._judge_delay_shift` | partial（逐 hop 子步驟；保留原始先後順序與 scratch 生命週期） | — | `ctest:test_process_context.c` |
| `src/aec.c:aec_prepare_capture_hop` | `modules/orchestrator.py:AEC._prepare_capture` | partial（逐 hop 子步驟；保留原始先後順序與 scratch 生命週期） | — | `ctest:test_process_context.c` |
| `src/aec.c:aec_prepare_mu` | `modules/orchestrator.py:AEC._update_render_gates` | partial（逐 hop 子步驟；保留原始先後順序與 scratch 生命週期） | — | `ctest:test_process_context.c` |
| `src/aec.c:aec_prepare_post_dt` | `modules/orchestrator.py:AEC._compute_residual_steering` | partial（逐 hop 子步驟；保留原始先後順序與 scratch 生命週期） | — | `ctest:test_process_context.c` |
| `src/aec.c:aec_prepare_rsa` | `modules/orchestrator.py:AEC._prepare_linear_gates` | partial（逐 hop 子步驟；保留原始先後順序與 scratch 生命週期） | — | `ctest:test_process_context.c` |
| `src/aec.c:aec_refresh_stationarity` | `modules/orchestrator.py:AEC._update_render_stationarity` | partial（逐 hop 子步驟；保留原始先後順序與 scratch 生命週期） | — | `ctest:test_process_context.c` |
| `src/aec.c:aec_run_main_filter` | `modules/orchestrator.py:AEC._process_linear` | partial（逐 hop 子步驟；保留原始先後順序與 scratch 生命週期） | — | `ctest:test_process_context.c` |
| `src/aec.c:aec_run_post_chain` | `modules/orchestrator.py:AEC._aec3_post` | partial（逐 hop 子步驟；保留原始先後順序與 scratch 生命週期） | — | `ctest:test_process_context.c` |
| `src/aec.c:aec_run_shadow` | `modules/orchestrator.py:AEC._process_linear`<br>`modules/orchestrator.py:AEC._update_shadow` | partial（逐 hop 子步驟；保留原始先後順序與 scratch 生命週期） | — | `ctest:test_process_context.c` |
| `src/aec.c:aec_track_delay_shift` | `modules/orchestrator.py:AEC._track_delay_shift`<br>`modules/orchestrator.py:AEC._accept_delay_shift` | partial（逐 hop 子步驟；保留原始先後順序與 scratch 生命週期） | — | `ctest:test_process_context.c` |
| `src/aec.c:aec_update_delay_estimator` | `modules/orchestrator.py:AEC._update_delay_estimate` | partial（逐 hop 子步驟；保留原始先後順序與 scratch 生命週期） | — | `ctest:test_process_context.c` |
| `src/aec.c:aec_duty_leak_db` | — | C-only（duty 看門狗的逐 hop ERLE 洩漏量：0.001 dB 在 10 ms 參考 hop，依 hop 時長線性換算；單通道與陣列實例共用） | — | `ctest:test_duty_census.c`<br>`ctest:test_rate_structural.c`<br>`ctest:test_aec_array_flow.c` |
| `src/aec.c:aec_duty_configure` | — | C-only（duty 閘門的 hold／period／leak 一次換算：秒依格點換成 hop，下限 1，period 再除以 `AEC_DUTY_PERIOD_DIVISOR`；單通道與陣列實例共用） | — | `ctest:test_duty_census.c`<br>`ctest:test_aec_array_flow.c` |
| `src/aec.c:aec_duty_reset` | — | C-only（清除 duty 閘門狀態，不動 hold／period／leak；單通道與陣列實例共用） | — | `ctest:test_duty_census.c`<br>`ctest:test_aec_array_flow.c` |
| `src/aec.c:aec_duty_begin` | — | C-only（duty 閘門前半：推進抽樣相位並回傳本 hop 是否跑 matched filter 分析，並計入 `hops_run`；單通道與陣列實例共用） | — | `ctest:test_duty_census.c`<br>`ctest:test_aec_array_flow.c` |
| `src/aec.c:aec_duty_end` | — | C-only（duty 閘門後半：依估計是否穩定與 windowed ERLE 啟動、保持或放棄抽樣；單通道與陣列實例共用） | — | `ctest:test_duty_census.c`<br>`ctest:test_aec_array_flow.c` |
| `src/aec.c:aec_epc_gain_change_recovery` | — | C-only（EPV 與 shadow_rise 兩個觸發共用的回復步驟；Python 在兩處呼叫點各自內聯） | — | `ctest:test_process_context.c` |
| `src/aec.c:aec_update_echo_path` | `modules/orchestrator.py:AEC._update_path_changes` | partial（逐 hop 子步驟；保留原始先後順序與 scratch 生命週期） | — | `ctest:test_process_context.c` |
| `src/aec.c:aec_update_linear_only_steering` | `modules/orchestrator.py:AEC._update_linear_only_steering` | partial（逐 hop 子步驟；保留原始先後順序與 scratch 生命週期） | — | `ctest:test_process_context.c` |
| `src/aec.c:aec_update_misadjustment_erle` | `modules/orchestrator.py:AEC._track_linear_error` | partial（逐 hop 子步驟；保留原始先後順序與 scratch 生命週期） | — | `ctest:test_process_context.c` |
| `src/aec.c:aec_hop_powers` | — | C-only（hop 的參考／麥克風／線性殘差三個均方，各加 1e-10；Python 在各呼叫點內聯） | — | `ctest:test_process_context.c` |
| `src/aec.c:aec_steer_mu` | `modules/orchestrator.py:AEC._update_residual_mu` | partial（逐 hop 子步驟；保留原始先後順序與 scratch 生命週期；殘差級與 `residual = OFF` 的線性步驟共用） | — | `ctest:test_process_context.c` |
| `src/aec.c:aec_update_shadow_dt` | `modules/orchestrator.py:AEC._update_shadow` | partial（逐 hop 子步驟；保留原始先後順序與 scratch 生命週期） | — | `ctest:test_process_context.c` |
| `src/aec.c:aec_update_shadow_quality` | `modules/orchestrator.py:AEC._update_shadow_leakage` | partial（逐 hop 子步驟；保留原始先後順序與 scratch 生命週期） | — | `ctest:test_process_context.c` |
| `src/aec.c:aec_update_windowed_erle` | `modules/orchestrator.py:AEC._track_linear_error` | partial（逐 hop 子步驟；保留原始先後順序與 scratch 生命週期） | — | `ctest:test_process_context.c` |
| `src/aec.c:aec_carve` | `modules/orchestrator.py:AEC.__init__`<br>`modules/orchestrator.py:AEC._seed_poor_excitation_counters`<br>`modules/orchestrator.py:AEC._init_delay`<br>`modules/orchestrator.py:AEC._init_linear_filter`<br>`modules/orchestrator.py:AEC._init_observer`<br>`modules/orchestrator.py:AEC._init_output_state`<br>`modules/orchestrator.py:AEC._init_path_tracking`<br>`modules/orchestrator.py:AEC._init_preprocessing`<br>`modules/orchestrator.py:AEC._init_render_analysis`<br>`modules/orchestrator.py:AEC._init_shadow_filter` | partial（建構全部子模組並綁定記憶體池，並播種 poor-excitation 計數器；Python 只建物件） | D-06；D-07 | `ctest:test_static_aec.c` |
| `src/aec.c:aec_init` | `modules/orchestrator.py:AEC.__init__` | partial（推定：pool-first 建構入口） | — | `ctest:test_lifecycle.c` |
| `src/aec.c:aec_build_flags_hash` | — | C-only（記憶體池描述子的建置指紋：對配置文字而非可調數值取雜湊；Python 沒有 pool） | — | `ctest:test_aec_array.c` |
| `src/aec.c:aec_get_mem_requirements` | — | C-only（填 AecMemReq 描述子，供 `aec_init_ex` 拒絕與新版函式庫不相容的舊 pool；Python 沒有 pool） | — | `ctest:test_aec_array.c` |
| `src/aec.c:aec_init_resolved` | `modules/orchestrator.py:AEC.__init__` | partial（推定：已解析的設定在呼叫端 pool 內建構；`n_mics` 為陣列值時轉交 `aec_array_impl_init`；Python 只建物件） | D-06；D-07 | `ctest:test_lifecycle.c`<br>`ctest:test_static_aec.c` |
| `src/aec.c:aec_create_resolved` | `modules/orchestrator.py:AEC.__init__` | partial（推定：一次對齊配置 `aec_mem_size_resolved()` 位元組、以 `aec_init_resolved` 建構並記下該區塊；`aec_create` 與測試鉤子共用） | D-06；D-07 | `ctest:test_heap_alignment.c` |
| `src/aec.c:aec_mem_size_resolved` | — | C-only（已解析設定的記憶池位元組數；單通道與陣列分流，公開 `aec_get_mem_size` 與測試鉤子共用） | — | `ctest:test_memory_layout.c` |
| `src/aec.c:aec_mem_breakdown_resolved` | — | C-only（已解析設定的記憶池分項） | — | `ctest:test_aec_array.c` |
| `src/aec.c:aec_mem_requirements_resolved` | — | C-only（已解析設定的 AecMemReq 描述子） | — | `ctest:test_aec_array.c` |
| `src/aec.c:aec_init_ex_resolved` | `modules/orchestrator.py:AEC.__init__` | partial（推定：`aec_init_resolved` 加選用的 AecMemReq 核對） | — | `ctest:test_aec_array.c` |
| `src/aec.c:aec_init_ex` | `modules/orchestrator.py:AEC.__init__` | partial（推定：`aec_init` 加選用的 AecMemReq 核對；核對通過後同 `aec_init`） | — | `ctest:test_aec_array.c` |
| `src/aec.c:aec_lane_get_mem_size` | — | C-only（陣列實例的 lane 池尺寸：僅接受 lane 契約的設定，省去該契約不跑的配置；Python 沒有 pool） | — | `ctest:test_memory_layout.c`<br>`ctest:test_aec_array.c` |
| `src/aec.c:aec_lane_init` | — | C-only（在陣列池內建構一條 lane：借用陣列的 FFT 控制代碼，建構結束位置須等於 `aec_lane_get_mem_size`；Python 沒有 pool） | — | `ctest:test_aec_array.c`<br>`ctest:test_aec_array_flow.c` |
| `src/aec.c:aec_init_single` | — | C-only（單通道建構：`aec_init_resolved`（單通道）與 `aec_lane_init`（陣列的 lane）共用，lane 模式略去其契約不跑的配置；Python 沒有 pool） | — | `ctest:test_lifecycle.c`<br>`ctest:test_memory_layout.c` |
| `src/aec.c:aec_lane_contract_ok` | — | C-only（lane 契約檢查：單通道、EXTERNAL_ALIGNED、`residual` 為 `STATE` 或 `ESTIMATE`，其餘設定一律拒絕） | — | `ctest:test_memory_layout.c`<br>`ctest:test_aec_array.c` |
| `src/aec.c:aec_state_lane` | — | C-only（`STATE` lane 的判斷：這種 lane 在 AecState 更新後就結束，carve 略過殘餘回音估計器與舒適噪音陣列） | — | `ctest:test_memory_layout.c`<br>`ctest:test_aec_array.c` |
| `src/aec.c:aec_mem_size_impl` | — | C-only（pool 位元組數的計算：單通道與 lane 共用，lane 模式扣掉契約不跑的配置；`aec_carve` 結束位置必須等於它，否則 `aec_init` 失敗；Python 沒有 pool） | — | `ctest:test_memory_layout.c`<br>`ctest:test_aec_array.c`<br>`ctest:test_split_entry2.c` |
| `src/aec.c:aec_lane_process_linear` | — | C-only（陣列 lane 的線性 stage：lane 0 照常計算並發布只依賴遠端的結果，其餘 lane 取用；`hop`／`fft`／`analysis` 三層各有成立條件；線性檢視不是有限值時回 `AEC_ERR_DSP`，由陣列重設；Python 沒有陣列） | — | `ctest:test_aec_array_flow.c`<br>`ctest:test_aec_array.c`<br>`ctest:test_dsp_refusal.c` |
| `src/aec.c:aec_lane_process_residual` | — | C-only（陣列 lane 的後級 stage：遠端頻譜相同時取 lane 0 的遠端 PSD、縮放後的 render 區塊與活動判定；收尾檢查不通過時回 `AEC_ERR_DSP`，由陣列重設） | — | `ctest:test_aec_array_flow.c`<br>`ctest:test_aec_array.c`<br>`ctest:test_dsp_refusal.c` |
| `src/aec.c:aec_lane_adopt_far_analysis` | — | C-only（把 lane 0 的 RSA 計數器與噪聲底限複製給一條一直沿用 lane 0 結果、自身狀態落後的 lane，讓它之後自己更新） | — | `ctest:test_aec_array_flow.c` |
| `src/aec.c:aec_lane_far_history_equal` | — | C-only（兩條 lane 的遠端歷史（main／shadow 的 far_buffer 與前一 hop 的功率譜）是否位元相同；陣列用它判定誰還能沿用 lane 0 的遠端分析） | — | `ctest:test_aec_array_flow.c` |
| `src/aec.c:aec_destroy` | — | C-only（Python 無手動釋放；堆積實例釋放整塊記憶體） | — | `ctest:test_lifecycle.c` |
| `src/aec.c:aec_hop_size` | `modules/orchestrator.py:AEC.hop_size` | mirror | — | — |
| `src/aec.c:aec_bump_delay_generation` | — | C-only（AecLinearView 的對齊世代記號） | D-12 | `ctest:test_linear_context.c` |
| `src/aec.c:aec_far_fft_real_compute_count` | — | C-only（遠端 FFT 共用的計數儀表） | D-12 | `ctest:test_shared_far_spec.c` |
| `src/aec.c:aec_set_level` | `modules/orchestrator.py:AEC.set_preset` | mirror（NO_FLOOR 雙邊執行期都可切換，無守衛；C 的 far-active ramp 固定 100 ms、只收 level，Python `set_preset` 保留 `ramp_ms` 參數，傳 100.0 等價） | D-04 | `pytest:test_runtime_preset.py`<br>`ctest:test_runtime_preset.c` |
| `src/aec.c:aec_apply_external_realign` | `modules/orchestrator.py:AEC.apply_external_realign` | mirror | — | `pytest:test_external_realign.py`<br>`ctest:test_external_realign.c` |
| `src/aec.c:aec_reset` | `modules/orchestrator.py:AEC.reset` | mirror | — | `pytest:test_aec_reset.py`<br>`ctest:test_reset_parity.c` |
| `src/aec.c:misadj_update` | `modules/orchestrator.py:AEC._update_misadjustment_estimator` | mirror | — | — |
| `src/aec.c:misadj_fire` | `modules/orchestrator.py:AEC._fire_aec3_misadj_scale` | mirror（Python 方法名仍帶 aec3） | — | `ctest:test_counter_saturation.c` |
| `src/aec.c:aec_erle_ring_max_last15` | `modules/orchestrator.py:AEC._acquire_delay#_erle_slope_buf` | partial（Python 在 process 內以 `max(list(self._erle_slope_buf)[-15:] or [0.0])` 內嵌） | — | — |
| `src/aec.c:aec_linear_is_cancelling` | `modules/orchestrator.py:AEC.linear_is_cancelling` | mirror（Python 註解回指此 C 函式） | — | `ctest:test_external_realign.c` |
| `src/aec.c:aec_get_last_timing` | — | C-only（效能計時） | — | `ctest:bench_rtf.c` |
| `src/aec.c:aec_stage_linear` | `modules/orchestrator.py:AEC.process`<br>`modules/orchestrator.py:AEC._maybe_mark_diverged`<br>`modules/orchestrator.py:AEC._dt_from_energy`<br>`modules/orchestrator.py:AEC._dt_from_shadow`<br>`modules/orchestrator.py:AEC.get_erle_instant`<br>`modules/orchestrator.py:AEC._accept_delay_shift`<br>`modules/orchestrator.py:AEC._acquire_delay`<br>`modules/orchestrator.py:AEC._align_reference`<br>`modules/orchestrator.py:AEC._judge_delay_shift`<br>`modules/orchestrator.py:AEC._prepare_capture`<br>`modules/orchestrator.py:AEC._prepare_linear_gates`<br>`modules/orchestrator.py:AEC._process_frequency`<br>`modules/orchestrator.py:AEC._process_linear`<br>`modules/orchestrator.py:AEC._track_delay_shift`<br>`modules/orchestrator.py:AEC._track_erle_convergence`<br>`modules/orchestrator.py:AEC._track_linear_error`<br>`modules/orchestrator.py:AEC._update_delay_estimate`<br>`modules/orchestrator.py:AEC._update_path_changes`<br>`modules/orchestrator.py:AEC._update_render_gates`<br>`modules/orchestrator.py:AEC._update_render_stationarity`<br>`modules/orchestrator.py:AEC._update_residual_mu`<br>`modules/orchestrator.py:AEC._update_shadow`<br>`modules/orchestrator.py:AEC._update_shadow_leakage` | partial（逐 hop 主迴圈的線性 stage：前端、delay、shadow、主 filter 到 windowed ERLE；`aec_process()`／`aec_process_linear()` 共用；C 無 return_formed_output／res_audit／診斷） | T-02；T-17；D-10 | `e2e:parity_aec_e2e`<br>`ctest:test_split_entry.c`<br>`ctest:test_process_context.c` |
| `src/aec.c:aec_stage_residual` | `modules/orchestrator.py:AEC._process_residual` | partial（逐 hop 的後級 stage：post 區塊、後級 mu、收尾；`apply` 時另套增益並合成；`AecConfig.residual` 決定做到哪裡；Python 的 res-context 路徑對應 `FULL` 搭配拆開的進入點） | T-05 | `ctest:test_split_entry.c`<br>`ctest:test_process_context.c` |
| `src/aec.c:aec_process` | `modules/orchestrator.py:AEC.process` | partial（離線包裝：兩個 stage 加輸出複製；`residual` 為 `FULL` 時另套增益、加舒適噪音並合成；結果溢位的 hop 回 `AEC_ERR_DSP`（C 專屬，見 `aec_closed_finite`）） | S-19；T-02 | `e2e:parity_aec_e2e`<br>`ctest:test_split_entry.c`<br>`ctest:test_process_context.c`<br>`ctest:test_dsp_refusal.c` |
| `src/aec.c:aec_process_linear` | `modules/orchestrator.py:AEC.process` | partial（只跑線性 stage 的入口；`residual` 為 `OFF` 時一呼叫結束整個 hop，其餘留待 `aec_process_residual`；結果溢位的 hop 回 `AEC_ERR_DSP`（C 專屬）） | D-12 | `ctest:test_split_entry.c`<br>`ctest:test_process_context.c`<br>`ctest:test_dsp_refusal.c` |
| `src/aec.c:aec_process_linear_shared_far_from` | — | C-only（宣告在 `aec_internal.h`；同 hop 的遠端頻譜與精確 magnitude² 成對共用；各 lane 保留自身歷史） | D-12 | `ctest:test_shared_far_spec.c`<br>`ctest:test_spectral_reuse.c` |
| `src/aec.c:aec_close_linear` | — | C-only（第一階段收尾：`residual == OFF` 時直接結束該 hop，否則把 hop 留在待後級的狀態） | — | `ctest:test_split_entry.c` |
| `src/aec.c:aec_far_donor_ok` | — | C-only（遠端頻譜共用的 donor 資格判斷：另一個同格點的單聲道實例） | D-12 | `ctest:test_shared_far_spec.c` |
| `src/aec.c:aec_input_finite` | — | C-only（輸入含非有限值時在狀態變動前拒絕該 hop） | — | `ctest:test_split_entry.c` |
| `src/aec.c:aec_cvec_finite` | — | C-only（複數向量是否全為有限值；單聲道與陣列的被拒 hop 檢查共用） | — | `ctest:test_dsp_refusal.c`<br>`ctest:test_aec_array.c` |
| `src/aec.c:aec_refuse_hop` | — | C-only（被拒 hop 的出口，單聲道與陣列共用：像 `aec_reset()` 一樣重設、把線性檢視清零（陣列為 lane 0 的），回 `AEC_ERR_DSP`；Python 版沒有這個拒絕：只在有限輸入使結果溢位時才作用，正常輸入（含預設設定）兩邊逐位元一致不受影響，與 ULCNet 有限值檢查同屬 C 端防護） | — | `ctest:test_dsp_refusal.c`<br>`ctest:test_aec_array.c`<br>`ctest:test_aec_array_flow.c` |
| `src/aec.c:aec_linear_finite` | — | C-only（線性 stage 交出的向量（單聲道與陣列每條 lane；`AecLinearView` 的 linear_hop 與三個頻譜）是否有限；Python 版沒有這個拒絕：只在有限輸入使結果溢位時才作用，正常輸入（含預設設定）兩邊逐位元一致不受影響，與 ULCNet 有限值檢查同屬 C 端防護） | — | `ctest:test_dsp_refusal.c` |
| `src/aec.c:aec_closed_finite` | — | C-only（hop 收尾時的檢查（單聲道與陣列每條 lane）：近端／線性輸出功率追蹤值與 `AecResidualView` 向量（依 `residual` 所及）是否有限；Python 版沒有這個拒絕：只在有限輸入使結果溢位時才作用，正常輸入（含預設設定）兩邊逐位元一致不受影響，與 ULCNet 有限值檢查同屬 C 端防護） | — | `ctest:test_dsp_refusal.c` |
| `src/aec.c:aec_process_residual` | `modules/orchestrator.py:AEC.process` | partial（緊接線性 stage 的後級入口，不套用增益、不合成；等同 Python 的 res-context 路徑；結果溢位的 hop 回 `AEC_ERR_DSP`（C 專屬）） | T-05 | `ctest:test_split_entry.c`<br>`ctest:test_process_context.c`<br>`ctest:test_dsp_refusal.c` |
| `src/aec.c:aec_get_residual` | `modules/orchestrator.py:AEC._build_res_context#AecResContext`<br>`modules/orchestrator.py:AEC._post_stash_context`<br>`modules/orchestrator.py:AEC._build_res_context#output_base_spec`<br>`modules/orchestrator.py:AEC._post_stash_context#_res_output_base` | partial（後級檢視：分析 seam 與渲染底座 `base_spec`／`guard_active` 在同一個結構，帶 `struct_size` 檢查；Python 在 process 內建構兩者並放進同一個 res-context 資料類別，欄位集合兩邊不同；底座與 `formed_hop` 不配對） | D-05 | `ctest:test_split_entry.c`<br>`ctest:test_process_context.c`<br>`ctest:test_guard_unify.c`<br>`pytest:test_guard_unify.py` |
| `src/aec.c:aec_gain_ready` | — | C-only（`aec_apply_gain`／`aec_apply_cng` 的前置檢查：需 `FULL` 且該 hop 已處理完） | D-12 | `ctest:test_split_entry.c` |
| `src/aec.c:aec_render_base` | — | C-only（增益套用的底座頻譜：輸出守衛啟動時為守衛的底座，否則為選定的誤差頻譜；`aec_get_residual`／`aec_apply_gain` 共用） | D-12 | `ctest:test_split_entry.c` |
| `src/aec_post.c:aec_apply_gain_spec` | `modules/orchestrator.py:AEC._aec3_post#_output_base * gain` | partial（無實例形式：底座乘增益，可帶外部增益取 min；Python 在 post 內部直接套用） | — | `ctest:test_split_entry.c`<br>`ctest:test_guard_unify.c` |
| `src/aec_post.c:aec_apply_cng_spec` | `modules/cng.py:cng_fill`<br>`modules/orchestrator.py:AEC._post_add_comfort_noise` | partial（無實例形式：以呼叫端持有的產生器狀態加舒適噪音，薄包裝 `aec_cng_fill`） | — | `ctest:test_split_entry.c`<br>`ctest:test_guard_unify.c` |
| `src/aec.c:aec_apply_gain` | — | C-only（實例形式：對 `aec_get_residual().base_spec` 乘 `G_res`，可帶外部增益；Python 無獨立入口） | D-12 | `ctest:test_split_entry.c`<br>`ctest:test_guard_unify.c` |
| `src/aec.c:aec_apply_cng` | — | C-only（實例形式：用實例內產生器加舒適噪音，種子隨硬 reset 重設；Python 無獨立入口） | D-12 | `ctest:test_split_entry.c`<br>`ctest:test_guard_unify.c` |
| `src/aec_post.c:aec_cng_fill` | `modules/cng.py:cng_fill`<br>`modules/cng.py:cng_draw_indices`<br>`modules/cng.py:cng_noise_terms`<br>`modules/orchestrator.py:AEC._post_add_comfort_noise#cng_draw_indices` | mirror（單一數值核心：`aec_post_apply_output` 內部 CNG 與公開 helper 共用同一份 sqrt、乘積順序與逐 bin 抽樣排程；Python 的 `_post_add_comfort_noise` 與 `cng_fill` 共用 `cng_draw_indices`／`cng_noise_terms`；種子 `AEC_CNG_SEED` 兩邊同為 42）| — | `ctest:test_aec_res.c`<br>`ctest:test_guard_unify.c`<br>`pytest:test_guard_unify.py` |
| `src/aec.c:aec_post_reset_serial` | `modules/orchestrator.py:AEC.get_post_reset_serial` | mirror（硬 post 鏈 reset 的不消耗計數器，uint32 回繞；Python 另在 `AecResContext.post_reset_serial` 帶同一個值） | — | `ctest:test_guard_unify.c`<br>`pytest:test_guard_unify.py` |
| `src/aec.c:aec_synthesis_window` | — | C-only（公開存取器：回傳函式庫自己做 OLA 前乘的合成窗與其長度，供外部合成器以 `ac_istft` 借用；Python 側的窗在 `modules/` 內部，沒有對應入口） | — | `ctest:test_synthesis_window.c` |
| `src/aec.c:aec_delay_confidence` | — | C-only（跨 delay 模式的統一信心值；MATCHED 值來自 `delay_aec_confidence`） | D-12 | `ctest:test_duty_census.c` |
| `src/aec.c:aec_debug_status` | — | C-only（AecDebugStatus 唯讀狀態查詢） | D-12 | `ctest:test_duty_census.c` |
| `src/aec.c:aec_get_linear` | — | C-only（AecLinearView 線性 stage 檢視） | D-12 | `ctest:test_linear_context.c`<br>`ctest:test_process_context.c` |

### 1.2 `aec_post.c`　aec_post.c（AEC 後處理驅動：對應 Python 的 `_aec3_post`）

依據：C 的 `aec_post_run` 對應 `AEC._aec3_post` 的整條編排，`aec_post_chain_reset` 對應 `_reset_aec3_post`。Python 的 `AEC._aec3_post` 依序呼叫 spectra、state、residual、gain 與 synthesis helper；下表對應實際執行該階段的方法。

| C 檔:函式 | Python 檔:符號 | 關係 | 已知落差 | 對拍工具／測試 |
|---|---|---|---| --- |
| `src/aec_post.c:aec_post_config_defaults` | — | C-only（把 `_aec3_post` 讀到的欄位收成 AecPostConfig 並給預設；Python 直接讀 AecConfig） | — | — |
| `src/aec_post.c:aec_post_init` | `modules/orchestrator.py:AEC._init_post_synthesis#_aec3_cng_seed`<br>`modules/orchestrator.py:AEC._init_post_state`<br>`modules/orchestrator.py:AEC._init_suppression_chain` | partial（推定：C 建構 post 鏈狀態；Python 在 `AEC.__init__` 內建構同組 `_aec3_*` 狀態） | D-06 | — |
| `src/aec_post.c:aec_post_reset` | `modules/orchestrator.py:AEC._reset_aec3_post`<br>`modules/orchestrator.py:AEC._reset_coherence_state` | mirror（C 註解：mirrors `_reset_aec3_post`，並含 `_reset_coherence_state`） | — | `pytest:test_aec_reset.py` |
| `src/aec_post.c:psd_from_abs` | — | C-only（`mag*mag*PSD_SCALE` 的 float32 運算序等價） | — | — |
| `src/aec_post.c:aec_post_compute_psds` | `modules/orchestrator.py:AEC._post_prepare_spectra#_near_psd_win` | partial（`_aec3_post` stage 1–2） | — | — |
| `src/aec_post.c:aec_post_avg_reverb_slots` | `modules/orchestrator.py:AEC._post_render_reverb#_past_idx` | partial（avg-render-reverb 取用的延遲分區與前一分區的槽位索引；Python 在 `_aec3_post` 內以 `_curr_p`、`_delay_idx`、`_past_idx` 內嵌） | — | `ctest:test_avg_reverb_slots.c` |
| `src/aec_post.c:aec_post_compute_x2_reverb` | `modules/orchestrator.py:AEC._post_render_reverb#x2_reverb_for_erle` | partial（`_aec3_post` stage 3） | — | — |
| `src/aec_post.c:aec_post_compute_coherence` | `modules/orchestrator.py:AEC._post_update_state#_coh_erle_sye` | partial（`_aec3_post` stage 4：Γ²(Ŷ,Y) EMA 與 ERLE coherence gate） | — | — |
| `src/aec_post.c:aec_post_compute_comfort_noise` | `modules/orchestrator.py:AEC._post_comfort_noise#_aec3_cng` | partial（`_aec3_post` stage 5：CNG N2 追蹤） | — | — |
| `src/aec_post.c:aec_post_apply_output` | `modules/orchestrator.py:AEC._post_add_comfort_noise#cng_noise_terms`<br>`modules/orchestrator.py:AEC._post_synthesize` | partial（`_aec3_post` stage 6：對渲染底座套增益、注入 CNG、irfft、OLA；底座已由 `aec_post_eval_output_guard` 在每 hop 決定，這裡不再選；CNG 走 `aec_cng_fill`） | — | `ctest:test_guard_unify.c` |
| `src/aec_post.c:aec_post_eval_output_guard` | `modules/orchestrator.py:AEC._post_output_base`<br>`modules/orchestrator.py:AEC._guard_energy` | mirror（輸出守衛，每 hop 在 usable 判定確定後、任何 context 提前離開之前求值一次；`sum(abs(E)²)` 與 `sum(abs(Y)²)` 同為 float32 pairwise 歸約、嚴格 `>`；相同頻譜得到相同決定；C 對 Python 的對拍允許雙方都 unusable 且相對能量差在 1e-6 tie band 內的 hop 不一致並逐 cell 計數，band 外不一致即失敗） | — | `ctest:test_guard_unify.c`<br>`pytest:test_guard_unify.py` |
| `src/aec_post.c:aec_post_run_observer_step` | `modules/orchestrator.py:AEC._post_observer_residual#_observer_corrob_margin` | partial（live observer 注入步驟；C 缺 `live_observer_inject_dt_energy_max` 守衛） | S-02；D-15 | — |
| `src/aec_post.c:aec_post_run` | `modules/orchestrator.py:AEC._aec3_post`<br>`modules/orchestrator.py:AEC._post_estimate_residual`<br>`modules/orchestrator.py:AEC._post_external_delay`<br>`modules/orchestrator.py:AEC._post_filter_bridge`<br>`modules/orchestrator.py:AEC._post_filter_taps`<br>`modules/orchestrator.py:AEC._post_nearend_power`<br>`modules/orchestrator.py:AEC._post_pending_changes`<br>`modules/orchestrator.py:AEC._post_stationary_residual`<br>`modules/orchestrator.py:AEC._post_suppression_gain` | partial（完整 `_aec3_post` 編排；C 版不含 Python-only 研究旋鈕） | S-22；T-03 | `e2e:parity_aec_e2e` |

### 1.3 `aec_res.c`　aec_res.c（RES 增益階段，內部模組，供陣列實例的融合後增益使用）

依據：本檔是 `SuppressionGain` 的內部包裝（標頭 `include/internal/aec_res_internal.h`，不對外公開），由陣列實例（`src/aec_array.c`）對融合後的頻譜算一次 RES 增益；範圍內（`AEC/python`）沒有對應的 Python 類別。Python 孿生在 `Audio_ALG/pipelines/4ch/pipeline.py` 的 `PostBeamResidualSuppressor`（範圍外，不在本表驗證；落差見 S-26）。`aec_cng_fill`（宣告在 `aec_internal.h`，定義在 `src/aec.c`）的 Python 孿生是範圍內的 `modules/cng.py:cng_fill`（見該列）；`Audio_ALG/pipelines/mono/cng_aec3.py` 是範圍外的另一份應用端副本，不在本表驗證。

| C 檔:函式 | Python 檔:符號 | 關係 | 已知落差 | 對拍工具／測試 |
|---|---|---|---| --- |
| `src/aec_res.c:sg_float_count` | — | C-only（範圍內無 Python 對應；包裝層的記憶體計算） | — | `ctest:test_aec_res.c` |
| `src/aec_res.c:aec_res_get_mem_size_resolved` | — | C-only（範圍內無 Python 對應；已解析設定的增益階段記憶體，陣列實例傳自己的 tuning） | — | `ctest:test_aec_res.c` |
| `src/aec_res.c:aec_res_init_resolved` | — | C-only（範圍內無 Python 對應；已解析設定的增益階段建構） | — | `ctest:test_aec_res.c` |
| `src/aec_res.c:aec_res_get_mem_size` | — | C-only（範圍內無 Python 對應；靜態記憶體） | — | `ctest:test_aec_res.c` |
| `src/aec_res.c:rebuild` | — | C-only（範圍內無 Python 對應；以儲存的設定重建 suppressor） | — | `ctest:test_aec_res.c` |
| `src/aec_res.c:aec_res_init` | — | C-only（範圍內無 Python 對應；孿生為 4ch 管線的 PostBeamResidualSuppressor） | — | `ctest:test_aec_res.c` |
| `src/aec_res.c:aec_res_reset` | — | C-only（範圍內無 Python 對應） | — | `ctest:test_aec_res.c` |
| `src/aec_res.c:aec_res_n_bins` | — | C-only（範圍內無 Python 對應） | — | `ctest:test_aec_res.c` |
| `src/aec_res.c:aec_res_hop_size` | — | C-only（範圍內無 Python 對應） | — | `ctest:test_aec_res.c` |
| `src/aec_res.c:aec_res_process` | — | C-only（範圍內無 Python 對應；內部呼叫 `suppression_gain_get_gain`） | — | `ctest:test_aec_res.c` |
| `src/aec_res.c:aec_res_set_level` | — | C-only（內部函式；只收 level，far-active ramp 固定 `AEC_LEVEL_RAMP_MS` = 100 ms；範圍內無 Python 對應） | S-26；D-04 | `ctest:test_aec_res.c` |
| `src/aec_config_build.c:aec_sg_apply_level` | — | C-only（內部函式；把 level 的 far-active 下限［`AEC_LEVEL_RAMP_MS` ramp］、DT 下限與 split-floor 開關寫進一個 SuppressionGain；`aec_set_level` 與 `aec_res_set_level` 共用；範圍內無 Python 對應） | S-26；D-04 | `ctest:test_aec_res.c`<br>`ctest:test_runtime_preset.c` |
| `src/aec_res.c:aec_testing_res_mem_size` | — | C-only（測試專用鉤子，只在 `AEC_TESTING` 建置存在：單獨一個陣列增益階段的記憶體大小，供 Audio_ALG 的 pyc_parity 4ch replay 在錄製的 frame 上重算 G_res；`test_aec_array_flow.c` 驗證它與陣列自己的 G_res 逐位元相同） | — | `ctest:test_aec_array_flow.c` |
| `src/aec_res.c:aec_testing_res_init` | — | C-only（測試專用鉤子，只在 `AEC_TESTING` 建置存在：建立單獨的增益階段） | — | `ctest:test_aec_array_flow.c` |
| `src/aec_res.c:aec_testing_res_process` | — | C-only（測試專用鉤子，只在 `AEC_TESTING` 建置存在：以 `AecResidualView` 餵增益階段，欄位對映同陣列自己的呼叫） | — | `ctest:test_aec_array_flow.c` |

### 1.5 `aec_state.c`　aec_state.c（AEC 狀態彙整：可用線性、飽和、ERLE／ERL 查詢）

依據：C 的 `aec_state.c` 對應 `python/modules/state/aec_state.py`；`update()` 的子分析器呼叫順序固定，兩邊一致；查詢函式名一一對應 Python 方法。

| C 檔:函式 | Python 檔:符號 | 關係 | 已知落差 | 對拍工具／測試 |
|---|---|---|---| --- |
| `src/aec_state.c:resolve_startup_hops` | `modules/state/aec_state.py:AecState._resolve_startup_hops` | mirror | T-03 | — |
| `src/aec_state.c:aec_state_config_defaults` | `modules/state/aec_state.py:AecStateConfig` | partial（C 填預設；Python 為 dataclass 欄位預設） | T-03 | — |
| `src/aec_state.c:aec_state_init` | `modules/state/aec_state.py:AecState.__init__` | mirror | T-03 | — |
| `src/aec_state.c:aec_state_usable_linear_estimate` | `modules/state/aec_state.py:AecState.usable_linear_estimate` | mirror | T-03 | — |
| `src/aec_state.c:aec_state_saturated_echo` | `modules/state/aec_state.py:AecState.saturated_echo` | mirror | T-03 | — |
| `src/aec_state.c:aec_state_transition_triggered` | `modules/state/aec_state.py:AecState.transition_triggered` | mirror | T-03 | — |
| `src/aec_state.c:aec_state_min_direct_path_filter_delay` | `modules/state/aec_state.py:AecState.min_direct_path_filter_delay` | mirror | T-03 | — |
| `src/aec_state.c:aec_state_erle` | `modules/state/aec_state.py:AecState.erle` | mirror | T-03 | — |
| `src/aec_state.c:aec_state_erle_unbounded` | `modules/state/aec_state.py:AecState.erle_unbounded` | mirror | T-03 | — |
| `src/aec_state.c:aec_state_fullband_erle_log2` | `modules/state/aec_state.py:AecState.fullband_erle_log2` | mirror | T-03 | — |
| `src/aec_state.c:aec_state_get_inst_linear_quality_estimate` | `modules/state/aec_state.py:AecState.get_inst_linear_quality_estimate` | mirror | T-03 | — |
| `src/aec_state.c:aec_state_set_taps_provider` | — | C-only（tap provider 回呼註冊；Python 直接傳入濾波器係數） | — | — |
| `src/aec_state.c:aec_state_update_capture_saturation` | `modules/state/aec_state.py:AecState.update_capture_saturation` | mirror | T-03 | — |
| `src/aec_state.c:aec_state_full_reset` | `modules/state/aec_state.py:AecState._full_reset` | mirror | T-03 | — |
| `src/aec_state.c:aec_state_handle_echo_path_change` | `modules/state/aec_state.py:AecState.handle_echo_path_change` | mirror | T-03 | — |
| `src/aec_state.c:aec_state_update` | `modules/state/aec_state.py:AecState.update` | mirror | T-03 | `ctest:test_counter_saturation.c` |

### 1.6 `aec_scale.c`　aec_scale.c（單位換算：block／ms／hop／dB）

依據：C 的 `aec_scale.c` 對應 `python/modules/aec3_scale.py` 的換算 helper；函式名一一對應（C 側為 `aec_*`，Python 側檔名與函式名仍帶 aec3）。`aec_ms_to_hops` 對的是 `ms_to_hops_f32`（專為對齊 C 而存在），double 版 `ms_to_hops` 在整半格案例差 1 hop（D-09）。

| C 檔:函式 | Python 檔:符號 | 關係 | 已知落差 | 對拍工具／測試 |
|---|---|---|---| --- |
| `src/aec_scale.c:aec_blocks_to_hops` | `modules/aec3_scale.py:blocks_to_hops` | mirror | — | — |
| `src/aec_scale.c:aec_ms_to_hops` | `modules/aec3_scale.py:ms_to_hops_f32`<br>`modules/aec3_scale.py:ms_to_hops` | mirror（對 `ms_to_hops_f32`；double 版 `ms_to_hops` 在整半格案例差 1 hop） | D-09 | `ctest:test_rate_structural.c` |
| `src/aec_scale.c:aec_per_block_rate_to_per_hop` | `modules/aec3_scale.py:per_block_rate_to_per_hop` | mirror | — | — |
| `src/aec_scale.c:aec_per_block_ema_alpha_to_per_hop` | `modules/aec3_scale.py:per_block_ema_alpha_to_per_hop` | mirror | — | — |
| `src/aec_scale.c:aec_per_bin_psd_threshold` | `modules/aec3_scale.py:per_bin_psd_threshold` | mirror | — | — |
| `src/aec_scale.c:aec_block_energy_scale` | `modules/aec3_scale.py:block_energy_scale` | mirror | — | — |
| `src/aec_scale.c:aec_per_block_growth_to_per_hop` | `modules/aec3_scale.py:per_block_growth_to_per_hop` | mirror | — | — |
| `src/aec_scale.c:aec_ema_alpha_from_tau_ms` | `modules/aec3_scale.py:ema_alpha_from_tau_ms` | mirror | — | — |
| `src/aec_scale.c:aec_db_to_linear` | `modules/aec3_scale.py:db_to_linear` | mirror | — | — |
| `src/aec_scale.c:aec_growth_rehop` | `modules/aec3_scale.py:growth_rehop` | mirror | — | `ctest:test_rate_structural.c` |

### 1.7 `linear_filter_output.c`　linear_filter_output.c（線性濾波器輸出選擇：refined 對 coarse）

依據：C 的 `linear_filter_output.c` 對應 `AEC._aec3_select_linear_filter_output`（Python 方法名仍帶 aec3）。

| C 檔:函式 | Python 檔:符號 | 關係 | 已知落差 | 對拍工具／測試 |
|---|---|---|---| --- |
| `src/linear_filter_output.c:sum_sq_f32` | — | C-only（`np.sum(arr**2)` 的 float32 等價） | — | — |
| `src/linear_filter_output.c:linear_filter_select_reset` | `modules/orchestrator.py:AEC._reset_aec3_post#_form_prev_output_time` | partial（推定：持久選擇狀態的重設；Python 在 `_reset_aec3_post` 內重設） | T-03 | — |
| `src/linear_filter_output.c:linear_filter_select` | `modules/orchestrator.py:AEC._aec3_select_linear_filter_output` | partial（C 不記 refined_last_selected） | T-03 | `ctest:test_linear_context.c` |

### 1.8 `pbfdkf.c`　pbfdkf.c（分區頻域濾波器：PBFDAF 基底與 PBFDKF Kalman 增益）

依據：C 的 `pbfdkf.c` 對應 `python/modules/filters.py`。C 的 `PBFDAF` 結構同時是 shadow／observer 濾波器（NLMS 更新）與 `PBFDKF` 的基底，與 Python `PBFDKF(PBFDAF)` 的繼承一致。`pbfdkf_update_weights` 對 Python 的 `_update_weights_aec3`（方法名仍帶 aec3）。Python 有 FDAF／SUBBAND 模式與調校選項，C 只實作 PBFDKF 路徑（D-07）。

| C 檔:函式 | Python 檔:符號 | 關係 | 已知落差 | 對拍工具／測試 |
|---|---|---|---| --- |
| `src/pbfdkf.c:next_pow2` | — | C-only（記憶體配置輔助） | — | — |
| `src/pbfdkf.c:pbfdaf_compute_sizes` | `modules/filters.py:PBFDAF.__init__` | partial（推定：衍生尺寸的計算） | T-03 | — |
| `src/pbfdkf.c:pbfdaf_fill_windows` | `modules/filters.py:PBFDAF.__init__` | partial（推定：td_window 與 sqrt-Hann 分析窗的建表） | T-03 | — |
| `src/pbfdkf.c:pbfdaf_init_scalars` | `modules/filters.py:PBFDAF.__init__` | partial（純量欄位初值） | D-10；T-03 | — |
| `src/pbfdkf.c:pbfdaf_zero_state` | `modules/filters.py:PBFDAF.__init__` | partial（推定：狀態緩衝歸零；Python 以 np.zeros 配置） | T-03 | — |
| `src/pbfdkf.c:pbfdaf_get_mem_size` | — | C-only（靜態記憶體池尺寸查詢） | — | `ctest:test_config_validation.c` |
| `src/pbfdkf.c:pbfdaf_get_mem_size_follower` | — | C-only（lane 內 shadow 濾波器的池尺寸：共用主濾波器的遠端歷史，且不含 lightweight 前端不形成的頻譜） | — | `ctest:test_memory_layout.c` |
| `src/pbfdkf.c:pbfdaf_init_static` | `modules/filters.py:PBFDAF.__init__` | partial（推定：靜態記憶體池路徑） | T-03 | `ctest:test_shared_fft_handle.c`<br>`ctest:test_config_validation.c` |
| `src/pbfdkf.c:pbfdaf_init_static_follower` | `modules/filters.py:PBFDAF.__init__` | partial（推定：同 `pbfdaf_init_static`，但 `X_buf`／`x2_cache` 指向主濾波器的；前提是兩者每 hop 同輸入、同清除、同分區步進，Python 兩個物件各自持有） | — | `ctest:test_aec_array.c` |
| `src/pbfdkf.c:pbfdaf_mem_size_impl` | — | C-only（濾波器 pool 位元組數：一般與 follower 兩種配置共用；Python 沒有 pool） | — | `ctest:test_memory_layout.c` |
| `src/pbfdkf.c:pbfdaf_init_impl` | — | C-only（濾波器在 pool 內的建構：`pbfdaf_init_static` 與 `pbfdaf_init_static_follower` 共用；Python 沒有 pool） | — | `ctest:test_memory_layout.c`<br>`ctest:test_shared_fft_handle.c` |
| `src/pbfdkf.c:pbfdaf_free` | — | C-only（只放掉借用的共用 FFT 參考；池與 FFT 都屬於呼叫端） | — | `ctest:test_config_validation.c` |
| `src/pbfdkf.c:pbfdaf_reset_taps` | `modules/filters.py:PBFDAF.reset_taps` | mirror | T-03 | — |
| `src/pbfdkf.c:pbfdaf_clear_far_history` | `modules/filters.py:PBFDAF.reset_taps` | partial（推定：`reset_taps` 的遠端頻譜歷史清除半段） | T-03 | — |
| `src/pbfdkf.c:pbfdaf_reset` | `modules/filters.py:PBFDAF.reset` | mirror | T-03 | `ctest:test_shared_far_spec.c` |
| `src/pbfdkf.c:rfft_padded` | — | C-only（短輸入補零後的 rfft 輔助） | — | — |
| `src/pbfdkf.c:pbfdaf_frontend` | `modules/filters.py:PBFDAF.process` | partial（共用前端：緩衝位移、FFT、遠端 PSD、回音估計、誤差、加窗誤差；PBFDAF／PBFDKF 共用） | T-03 | — |
| `src/pbfdkf.c:pbfdaf_process` | `modules/filters.py:PBFDAF.process`<br>`modules/filters.py:PBFDAF.complete_update`<br>`modules/filters.py:PBFDAF._update_weights` | partial（NLMS 路徑；C 在單一函式內完成 Python 拆成三個方法的流程；`call_counter` 與 Python 一樣在 mu 全 0 的提早返回前先遞增） | T-03 | — |
| `src/pbfdkf.c:pbfdaf_process_with_far_cmag2` | — | C-only（PBFDAF 精確遠端 magnitude² 快取入口；流程同 process） | T-03 | `ctest:test_spectral_reuse.c` |
| `src/pbfdkf.c:pbfdaf_copy_weights_from` | `modules/filters.py:PBFDAF.copy_weights_from` | mirror | T-03 | — |
| `src/pbfdkf.c:pbfdaf_warm_shift_ir` | `modules/filters.py:PBFDAF.warm_shift_ir` | mirror | T-03 | `pytest:test_external_realign.py` |
| `src/pbfdkf.c:pbfdkf_init_scalars` | `modules/filters.py:PBFDKF.__init__` | partial（Kalman 純量欄位初值，含逐 block 速率換算） | T-03 | `ctest:test_process_context.c` |
| `src/pbfdkf.c:pbfdkf_get_mem_size` | — | C-only（靜態記憶體池尺寸查詢） | — | `ctest:test_config_validation.c` |
| `src/pbfdkf.c:pbfdkf_init_static` | `modules/filters.py:PBFDKF.__init__` | partial（推定：靜態記憶體池路徑） | T-03 | `ctest:test_shared_fft_handle.c`<br>`ctest:test_config_validation.c` |
| `src/pbfdkf.c:pbfdkf_free` | — | C-only（只放掉借用的共用 FFT 參考；池與 FFT 都屬於呼叫端） | — | `ctest:test_config_validation.c` |
| `src/pbfdkf.c:pbfdkf_rearm_kalman` | `modules/filters.py:PBFDKF._rearm_kalman` | partial（C 只重設 error_psd；Python 另重設 R、Q） | T-03 | — |
| `src/pbfdkf.c:pbfdkf_reset` | `modules/filters.py:PBFDKF.reset` | mirror | T-03 | — |
| `src/pbfdkf.c:pbfdkf_reset_taps` | `modules/filters.py:PBFDKF.reset_taps` | mirror | T-03 | — |
| `src/pbfdkf.c:pbfdaf_reset_carried_state` | `modules/filters.py:PBFDAF.reset_carried_state` | mirror（C 另含 `poor_excitation_counter` 的建構值寫入） | D-10；T-03 | `pytest:test_aec_reset.py` |
| `src/pbfdkf.c:pbfdkf_reset_carried_state` | `modules/filters.py:PBFDKF.reset_carried_state` | mirror | T-03 | `pytest:test_aec_reset.py` |
| `src/pbfdkf.c:pbfdkf_handle_echo_path_change` | `modules/filters.py:PBFDKF.handle_echo_path_change` | mirror（C 註解：Mirrors filters.py `_poor_excitation_counter`） | D-10；T-03 | — |
| `src/pbfdkf.c:pbfdaf_get_time_domain_filter_range` | `modules/filters.py:PBFDAF.get_time_domain_filter` | partial（只取部分分區，供 FilterAnalyzer 逐 hop 取一個區域） | T-03 | — |
| `src/pbfdkf.c:pbfdaf_get_time_domain_filter` | `modules/filters.py:PBFDAF.get_time_domain_filter` | mirror | T-03 | — |
| `src/pbfdkf.c:pbfdaf_scale_filter` | `modules/filters.py:PBFDAF.scale_filter` | mirror | T-03 | — |
| `src/pbfdkf.c:pbfdkf_scale_filter` | `modules/filters.py:PBFDKF.scale_filter` | mirror | T-03 | — |
| `src/pbfdkf.c:pbfdkf_h_error_refresh` | `modules/filters.py:PBFDKF._h_error_refresh` | mirror | T-03 | — |
| `src/pbfdkf.c:pbfdkf_update_weights` | `modules/filters.py:PBFDKF._update_weights_aec3`<br>`modules/filters.py:PBFDKF._update_weights` | mirror（Python 的 `_update_weights` 是閘門加包裝） | T-03 | — |
| `src/pbfdkf.c:pbfdkf_process_with_far_cmag2` | `modules/filters.py:PBFDAF.process`<br>`modules/filters.py:PBFDAF.complete_update` | partial（PBFDKF 沒有自己的 `process`，繼承 PBFDAF；C 把 Kalman 路徑獨立成函式，並多一個精確遠端 magnitude² 快取入口，回傳本 hop 的 error magnitude² cache 是否有效） | T-03 | `ctest:test_spectral_reuse.c` |

### 1.9 `delay_aec.c`　delay_aec.c（延遲估計鏈：降頻器、matched filter、lag 聚合、時脈漂移）

依據：C 的 `delay_aec.c` 對應 `python/modules/delay/` 內的 `lag_aggregator.py`（`aggregate()`）、`clockdrift_detector.py`、`matched_filter.py`（`matched_filter_core`、`MatchedFilter.update`）、`echo_path_delay_estimator.py`（`_process_inner_block`、`estimate_delay`）與 `downsampled_ring.py`（`_CascadedBiquad`）；對外包裝 `delay_aec_*` 對應 `legacy_compat.py` 的 `LegacyDelayShim`。C 的整條延遲鏈以 float32 實作，與 Python float64 的差異是刻意的（`delay_aec.h` 明載），不保證逐位元相同（D-03）；`aec.c` 的 duty-cycle 狀態機另以 `delay_aec_accumulate_ex` 驅動（`run_matched_filter=0` 時只餵不分析，Python 永遠分析）。Python 的 `RenderDelayController` 與 pre-echo 偵測在 C 內沒有對應（見 §3）。

| C 檔:函式 | Python 檔:符號 | 關係 | 已知落差 | 對拍工具／測試 |
|---|---|---|---| --- |
| `src/delay_aec.c:delay_aec_dot` | `modules/delay/matched_filter.py:matched_filter_core` | partial（推定：`np.dot` 的 float32 累加等價，用於 error_sum_anchor） | D-03 | `ctest:test_delay_reset.c` |
| `src/delay_aec.c:da_dot_f32` | `modules/delay/matched_filter.py:matched_filter_core` | partial（推定：`h·x` 的 NEON／純量核心；NEON／純量雙胞，算一個函式） | D-03 | — |
| `src/delay_aec.c:da_nlms_update_f32` | `modules/delay/matched_filter.py:matched_filter_core` | partial（推定：`h += alpha*x` 的 NEON／純量核心；NEON／純量雙胞，算一個函式） | D-03 | — |
| `src/delay_aec.c:da_nlms_update_next_dot_f32` | — | C-only（NEON 融合目前 tap 更新與下一個 dot；保留四組累加器、FMA 與 reduction tree） | D-03 | `ctest:test_delay_kernels.c` |
| `src/delay_aec.c:da_max_square_peak_index` | `modules/delay/matched_filter.py:max_square_peak_index` | mirror（NEON／純量雙胞，算一個函式） | D-03 | `ctest:test_delay_kernels.c` |
| `src/delay_aec.c:da_biquad_reset` | `modules/delay/downsampled_ring.py:_CascadedBiquad.reset` | mirror（C 為 float32 DF2T，Python 為 float64） | D-03 | — |
| `src/delay_aec.c:da_biquad_init_antialias` | `modules/delay/downsampled_ring.py:_CascadedBiquad.__init__` | partial（推定：抗混疊級係數；對應 Decimator 的低通級） | D-03 | `ctest:test_delay_kernels.c` |
| `src/delay_aec.c:da_biquad_init_noise_reduction` | `modules/delay/downsampled_ring.py:_CascadedBiquad.__init__` | partial（推定：雜訊抑制級係數；對應 Decimator 的高通級） | D-03 | `ctest:test_delay_kernels.c` |
| `src/delay_aec.c:da_biquad_process1` | `modules/delay/downsampled_ring.py:_CascadedBiquad.process` | mirror（逐樣本；C 為 float32 DF2T，Python 為 float64） | D-03 | `ctest:test_delay_kernels.c` |
| `src/delay_aec.c:da_biquad_init_resample48` | `modules/delay/echo_path_delay_estimator.py:_Resample48.__init__` | partial（推定：48 kHz 到 16 kHz 旁路的 biquad 係數） | D-03 | `ctest:test_delay_kernels.c` |
| `src/delay_aec.c:da_resample48_init` | `modules/delay/echo_path_delay_estimator.py:_Resample48.__init__` | mirror | D-03 | — |
| `src/delay_aec.c:da_resample48_reset` | `modules/delay/echo_path_delay_estimator.py:_Resample48.reset` | mirror | D-03 | — |
| `src/delay_aec.c:da_resample48_channel` | `modules/delay/echo_path_delay_estimator.py:_Resample48.process_decimate` | mirror | D-03 | `ctest:test_delay_kernels.c` |
| `src/delay_aec.c:da_decimator_init` | `modules/delay/downsampled_ring.py:Decimator.__init__` | mirror | D-03 | `ctest:test_delay_kernels.c` |
| `src/delay_aec.c:da_decimator_decimate_generic` | `modules/delay/downsampled_ring.py:Decimator.decimate` | mirror（通用逐樣本走法，64 樣本區塊） | D-03 | `ctest:test_delay_kernels.c` |
| `src/delay_aec.c:da_decimator_decimate` | `modules/delay/downsampled_ring.py:Decimator.decimate` | mirror（對外分派：NEON 版或通用版） | D-03 | `ctest:test_delay_kernels.c` |
| `src/delay_aec.c:da_biquad_same_coefficients` | — | C-only（判斷近端／遠端兩路係數相同，以啟用成對 NEON 路徑） | — | — |
| `src/delay_aec.c:da_decimator_same_coefficients` | — | C-only（同上；只在 NEON 建置存在） | — | `ctest:test_delay_kernels.c` |
| `src/delay_aec.c:da_decimator_decimate_pair` | `modules/delay/downsampled_ring.py:Decimator.decimate` | partial（近端遠端成對降頻的 NEON 優化；NEON／純量雙胞，算一個函式） | D-03 | `ctest:test_delay_kernels.c` |
| `src/delay_aec.c:da_resample48_pair` | `modules/delay/echo_path_delay_estimator.py:_Resample48.process_decimate` | partial（近端遠端成對的 NEON 優化；NEON／純量雙胞，算一個函式） | D-03 | `ctest:test_delay_kernels.c` |
| `src/delay_aec.c:da_ring_reset` | `modules/delay/downsampled_ring.py:DownsampledRenderBuffer.reset` | mirror | — | — |
| `src/delay_aec.c:da_ring_push` | `modules/delay/downsampled_ring.py:DownsampledRenderBuffer.push_sub_block` | mirror | — | — |
| `src/delay_aec.c:da_ring_gather_back` | `modules/delay/downsampled_ring.py:DownsampledRenderBuffer.gather_back` | mirror | — | — |
| `src/delay_aec.c:da_matched_filter_core` | `modules/delay/matched_filter.py:matched_filter_core` | mirror（C 註解：Mirrors matched_filter_core()；float32） | D-03 | `ctest:test_delay_kernels.c`<br>`ctest:test_delay_matched_neon.c` |
| `src/delay_aec.c:da_mf_row` | — | C-only（兩個行主序濾波器組的列存取器） | — | — |
| `src/delay_aec.c:da_matched_filter_init` | `modules/delay/matched_filter.py:MatchedFilter.__init__` | partial（C 無 accumulated／instantaneous error 與 pre-echo 計數） | — | `ctest:test_delay_num_filters.c`<br>`pytest:test_delay_num_filters.py` |
| `src/delay_aec.c:da_matched_filter_reset` | `modules/delay/matched_filter.py:MatchedFilter.reset` | partial（C 無 accumulated error 重設，也沒有 full_reset 參數） | — | — |
| `src/delay_aec.c:da_matched_filter_update` | `modules/delay/matched_filter.py:MatchedFilter.update` | partial（略去 pre-echo 起點搜尋：該估計沒有消費者） | D-03 | `ctest:test_delay_num_filters.c`<br>`pytest:test_delay_num_filters.py` |
| `src/delay_aec.c:da_argmax_incremental_update` | `modules/delay/lag_aggregator.py:_HighestPeakAggregator.aggregate` | partial（推定：直方圖最大值的 O(1) 增量維護；Python 每次重算 argmax） | — | `ctest:test_delay_dominant_selection.c` |
| `src/delay_aec.c:da_highest_peak_reset` | `modules/delay/lag_aggregator.py:_HighestPeakAggregator.reset` | mirror | — | `ctest:test_delay_dominant_selection.c` |
| `src/delay_aec.c:da_highest_peak_init` | `modules/delay/lag_aggregator.py:_HighestPeakAggregator.__init__` | mirror | — | `ctest:test_delay_dominant_selection.c` |
| `src/delay_aec.c:da_highest_peak_aggregate` | `modules/delay/lag_aggregator.py:_HighestPeakAggregator.aggregate` | mirror | — | `ctest:test_delay_dominant_selection.c`<br>`pytest:test_delay_dominant_selection.py` |
| `src/delay_aec.c:da_aggregator_init` | `modules/delay/lag_aggregator.py:MatchedFilterLagAggregator.__init__` | mirror | — | — |
| `src/delay_aec.c:da_aggregator_reset` | `modules/delay/lag_aggregator.py:MatchedFilterLagAggregator.reset` | mirror | — | — |
| `src/delay_aec.c:da_aggregator_reliable_delay_found` | `modules/delay/lag_aggregator.py:MatchedFilterLagAggregator.reliable_delay_found` | mirror | — | — |
| `src/delay_aec.c:da_aggregator_delay_at_highest_peak` | `modules/delay/lag_aggregator.py:MatchedFilterLagAggregator.delay_at_highest_peak` | mirror | — | — |
| `src/delay_aec.c:da_aggregator_aggregate` | `modules/delay/lag_aggregator.py:MatchedFilterLagAggregator.aggregate` | partial（產品政策：回報延遲一律取最高峰，不走 pre-echo 分支） | — | `ctest:test_delay_dominant_selection.c`<br>`pytest:test_delay_dominant_selection.py` |
| `src/delay_aec.c:da_clockdrift_init` | `modules/delay/clockdrift_detector.py:ClockdriftDetector.__init__` | mirror | — | — |
| `src/delay_aec.c:da_clockdrift_update` | `modules/delay/clockdrift_detector.py:ClockdriftDetector.update` | mirror | — | — |
| `src/delay_aec.c:da_hist_window_for_rate` | `modules/delay/lag_aggregator.py:_num_blocks_per_second` | partial（1 秒內的 64 樣本內層區塊數，另夾在 C 的靜態容量內） | — | — |
| `src/delay_aec.c:da_estimator_clear_chain` | — | C-only（降頻器與 render ring 的清零，供 `da_estimator_init` 與 `da_estimator_reset` 共用） | — | `ctest:test_delay_reset.c` |
| `src/delay_aec.c:da_estimator_init` | `modules/delay/echo_path_delay_estimator.py:EchoPathDelayEstimator.__init__` | partial（C 無 pre-echo aggregator 與 old_agg_quality） | — | — |
| `src/delay_aec.c:da_estimator_reset_internal` | `modules/delay/echo_path_delay_estimator.py:EchoPathDelayEstimator._reset` | mirror | — | — |
| `src/delay_aec.c:da_estimator_reset` | `modules/delay/echo_path_delay_estimator.py:EchoPathDelayEstimator.reset` | mirror | — | `ctest:test_delay_reset.c`<br>`pytest:test_delay_reset.py` |
| `src/delay_aec.c:da_estimator_process_inner_block` | `modules/delay/echo_path_delay_estimator.py:EchoPathDelayEstimator._process_inner_block` | partial（C 多一個 run_matched_filter 參數以支援 duty-cycle） | D-03 | `ctest:test_delay_reset.c`<br>`ctest:test_delay_dominant_selection.c` |
| `src/delay_aec.c:da_estimator_estimate_delay` | `modules/delay/echo_path_delay_estimator.py:EchoPathDelayEstimator.estimate_delay` | mirror | D-03 | `ctest:test_delay_reset.c`<br>`ctest:test_delay_dominant_selection.c` |
| `src/delay_aec.c:da_valid_num_filters` | — | C-only（濾波器組大小的拒絕式驗證，不夾制） | — | `ctest:test_delay_num_filters.c` |
| `src/delay_aec.c:da_compute_layout` | — | C-only（記憶體池版面計算） | — | `ctest:test_delay_num_filters.c` |
| `src/delay_aec.c:da_delay_carve` | — | C-only（記憶體池指標綁定） | — | — |
| `src/delay_aec.c:delay_aec_get_mem_size` | — | C-only（靜態記憶體池尺寸查詢） | — | `ctest:test_delay_num_filters.c` |
| `src/delay_aec.c:delay_aec_max_delay_samples` | — | C-only（estimator 可回報延遲上限；單通道 MATCHED 參考環的大小依據） | — | `ctest:test_delay_num_filters.c` |
| `src/delay_aec.c:delay_aec_init` | `modules/delay/legacy_compat.py:LegacyDelayShim.__init__` | partial（C 另做記憶體池綁定與尺寸驗證） | — | `ctest:test_delay_num_filters.c` |
| `src/delay_aec.c:delay_aec_reset` | `modules/delay/legacy_compat.py:LegacyDelayShim.reset` | mirror | — | `ctest:test_delay_reset.c`<br>`pytest:test_delay_reset.py` |
| `src/delay_aec.c:delay_aec_accumulate_ex` | `modules/delay/legacy_compat.py:LegacyDelayShim.accumulate` | partial（C 加 duty-cycle 節流：run_matched_filter=0 時不分析，Python 永遠分析） | D-03 | `ctest:test_duty_census.c`<br>`ctest:test_counter_saturation.c` |
| `src/delay_aec.c:delay_aec_estimated_delay` | `modules/delay/legacy_compat.py:LegacyDelayShim.estimated_delay` | mirror | D-03 | `ctest:test_delay_reset.c` |
| `src/delay_aec.c:delay_aec_confidence` | `modules/delay/legacy_compat.py:LegacyDelayShim.confidence` | mirror（0.0／0.5／1.0 階梯） | D-03 | `ctest:test_delay_dominant_selection.c` |
| `src/delay_aec.c:delay_aec_is_solid` | `modules/delay/legacy_compat.py:LegacyDelayShim.is_solid` | mirror | D-03 | — |
| `src/delay_aec.c:delay_aec_n_updates` | `modules/delay/legacy_compat.py:LegacyDelayShim._n_updates` | mirror | D-03 | `ctest:test_delay_matched_neon.c` |
| `src/delay_aec.c:delay_aec_has_clockdrift` | `modules/delay/legacy_compat.py:LegacyDelayShim.has_clockdrift` | mirror | D-03 | — |

### 1.10 `suppression_gain.c`　suppression_gain.c（殘餘回音抑制增益）

依據：C 的 `suppression_gain.c` 對應 `python/modules/residual/suppression_gain.py`；static 函式名對應 Python 名稱（`_ne_state_for_gain_rules`、`LowNoiseRenderDetector.detect`、`DominantNearendDetector.update`、`_MovingAverageSpectrum.average`、`_get_max_gain`、`_get_min_gain`、`_gain_to_no_audible_echo`、`_limit_lf_gains`、`_limit_hf_gains`、`_lower_band_gain`）。C 的 `suppression_gain_get_gain` 內聯了 `_lower_band_gain` 與 `_sigmoid_blend_weight`。Python 的 `_DominantNearendDetector` 設定類別方法（`set_config`、`_derive_trigger_threshold`）與 `_build_gain_params`、`_MovingAverageSpectrum.update_memory_length` 在 C 內於初始化時展開（見 §3）。

| C 檔:函式 | Python 檔:符號 | 關係 | 已知落差 | 對拍工具／測試 |
|---|---|---|---| --- |
| `src/suppression_gain.c:f32_sum_slice` | — | C-only（切片的 pairwise float32 加總；Python 用 numpy） | — | — |
| `src/suppression_gain.c:suppression_gain_init` | `modules/residual/suppression_gain.py:SuppressionGain.__init__`<br>`modules/residual/suppression_gain.py:_db_to_power_f32`<br>`modules/residual/suppression_gain.py:_build_gain_params` | partial（C 一次完成 config 展開與緩衝綁定） | T-03 | `ctest:test_aec_res.c` |
| `src/suppression_gain.c:suppression_gain_set_initial_state` | `modules/residual/suppression_gain.py:SuppressionGain.set_initial_state` | mirror | T-03 | `ctest:test_aec_res.c` |
| `src/suppression_gain.c:sg_split_floor_db_to_power` | `modules/residual/suppression_gain.py:SuppressionGain._check_split_floor_db`<br>`modules/residual/suppression_gain.py:_db_to_power_f32` | mirror（範圍檢查加 dB 轉功率） | T-03 | `ctest:test_aec_res.c` |
| `src/suppression_gain.c:advance_split_floor_ramp` | `modules/residual/suppression_gain.py:SuppressionGain._advance_split_floor_ramp` | mirror | T-03 | `ctest:test_runtime_preset.c`<br>`pytest:test_runtime_preset.py` |
| `src/suppression_gain.c:suppression_gain_set_split_floor_far_active_db` | `modules/residual/suppression_gain.py:SuppressionGain.set_split_floor_far_active_db` | mirror | D-04；T-03 | `ctest:test_runtime_preset.c`<br>`pytest:test_runtime_preset.py` |
| `src/suppression_gain.c:suppression_gain_set_split_floor_dt_db` | `modules/residual/suppression_gain.py:SuppressionGain.set_split_floor_dt_db` | mirror | T-03 | `ctest:test_runtime_preset.c`<br>`pytest:test_runtime_preset.py` |
| `src/suppression_gain.c:suppression_gain_set_split_floor_enabled` | `modules/residual/suppression_gain.py:SuppressionGain.set_split_floor_enabled` | mirror（兩邊都是立即階躍，無 ramp） | D-04；T-03 | `ctest:test_runtime_preset.c`<br>`pytest:test_runtime_preset.py` |
| `src/suppression_gain.c:suppression_gain_is_dominant_nearend` | `modules/residual/suppression_gain.py:SuppressionGain.is_dominant_nearend` | mirror | T-03 | `ctest:test_rate_structural.c` |
| `src/suppression_gain.c:ne_state_for_gain_rules` | `modules/residual/suppression_gain.py:SuppressionGain._ne_state_for_gain_rules` | mirror | T-03 | — |
| `src/suppression_gain.c:low_noise_render_detect` | `modules/residual/suppression_gain.py:_LowNoiseRenderDetector.detect` | mirror | T-03 | — |
| `src/suppression_gain.c:dominant_nearend_update` | `modules/residual/suppression_gain.py:_DominantNearendDetector.update` | mirror | T-03 | `ctest:test_res_nearend_gate.c`<br>`pytest:test_res_nearend_gate.py` |
| `src/suppression_gain.c:nearend_smoother_average` | `modules/residual/suppression_gain.py:_MovingAverageSpectrum.average` | mirror | T-03 | — |
| `src/suppression_gain.c:weigh_band` | `modules/residual/suppression_gain.py:_weight_echo_for_audibility` | partial（推定：`weigh` 內層函式的 band 權重計算） | T-03 | — |
| `src/suppression_gain.c:weight_echo_for_audibility` | `modules/residual/suppression_gain.py:_weight_echo_for_audibility` | mirror | T-03 | — |
| `src/suppression_gain.c:get_max_gain` | `modules/residual/suppression_gain.py:SuppressionGain._get_max_gain` | mirror | T-03 | — |
| `src/suppression_gain.c:get_min_gain` | `modules/residual/suppression_gain.py:SuppressionGain._get_min_gain` | mirror | T-03 | — |
| `src/suppression_gain.c:gain_to_no_audible_echo` | `modules/residual/suppression_gain.py:SuppressionGain._gain_to_no_audible_echo`<br>`modules/residual/suppression_gain.py:SuppressionGain._sigmoid_blend_weight` | mirror（C 內聯 sigmoid 混合權重） | T-03 | — |
| `src/suppression_gain.c:limit_lf_gains` | `modules/residual/suppression_gain.py:_limit_lf_gains` | mirror | T-03 | — |
| `src/suppression_gain.c:limit_hf_gains` | `modules/residual/suppression_gain.py:_limit_hf_gains` | mirror | T-03 | — |
| `src/suppression_gain.c:suppression_gain_update_dominant_nearend` | `modules/residual/suppression_gain.py:SuppressionGain.get_gain#_dominant_nearend.update` | partial（推定：`get_gain` 開頭的主導近端更新步驟，獨立成函式） | T-03 | `ctest:test_rate_structural.c` |
| `src/suppression_gain.c:suppression_gain_get_gain` | `modules/residual/suppression_gain.py:SuppressionGain.get_gain`<br>`modules/residual/suppression_gain.py:SuppressionGain._lower_band_gain` | mirror（C 內聯 `_lower_band_gain`） | T-03 | `ctest:test_aec_res.c` |
| `src/suppression_gain.c:suppression_gain_render_stats` | `modules/residual/suppression_gain.py:SuppressionGain.get_gain#render_block` | partial（獨立成函式的 render 統計：平方和與平方峰值；Python 在 `get_gain` 內部算） | T-03 | `ctest:test_aec_res.c` |
| `src/suppression_gain.c:suppression_gain_get_gain_stats` | `modules/residual/suppression_gain.py:SuppressionGain.get_gain` | partial（以先算好的 render 統計取增益；`suppression_gain_get_gain` 先算統計再呼叫它） | T-03 | `ctest:test_aec_res.c` |

### 1.11 `residual_echo_estimator.c`　residual_echo_estimator.c（殘餘回音功率估計）

依據：C 的 `residual_echo_estimator.c` 對應 `python/modules/residual/residual_echo_estimator.py`（AEC3 的 `UpdateRenderNoisePower`、`GetEchoPathGain`，Python 的 `_reverb_decay`、`_update_reverb_linear`）。C 版的殘響衰減走靜態設定路徑（未綁 decay estimator），因此 Python 的 `ReverbDecayEstimator` 與 `attach_reverb_decay_estimator` 在 C 內沒有對應（見 §3）。

| C 檔:函式 | Python 檔:符號 | 關係 | 已知落差 | 對拍工具／測試 |
|---|---|---|---| --- |
| `src/residual_echo_estimator.c:ree_buf_slot` | — | C-only（render 緩衝的環形槽索引；Python 用 deque） | — | — |
| `src/residual_echo_estimator.c:ree_appendleft` | — | C-only（Python deque.appendleft 的環形緩衝等價） | — | — |
| `src/residual_echo_estimator.c:ree_buf_at` | — | C-only（Python deque 索引的環形緩衝等價） | — | — |
| `src/residual_echo_estimator.c:ree_init` | `modules/residual/residual_echo_estimator.py:ResidualEchoEstimator.__init__` | mirror | D-06；T-03 | — |
| `src/residual_echo_estimator.c:ree_reset` | `modules/residual/residual_echo_estimator.py:ResidualEchoEstimator.reset` | mirror | T-03 | — |
| `src/residual_echo_estimator.c:ree_update_reverb_models` | `modules/residual/residual_echo_estimator.py:ResidualEchoEstimator.update_reverb_models` | mirror | T-03 | — |
| `src/residual_echo_estimator.c:ree_update_render_noise_power` | `modules/residual/residual_echo_estimator.py:ResidualEchoEstimator._update_render_noise_power` | mirror | T-03 | — |
| `src/residual_echo_estimator.c:ree_echo_path_gain` | `modules/residual/residual_echo_estimator.py:ResidualEchoEstimator._get_echo_path_gain` | mirror | T-03 | — |
| `src/residual_echo_estimator.c:ree_reverb_decay` | `modules/residual/residual_echo_estimator.py:ResidualEchoEstimator._reverb_decay` | partial（只實作靜態設定路徑，decay estimator 未綁定） | T-03 | — |
| `src/residual_echo_estimator.c:ree_add_reverb` | — | C-only（線性與非線性路徑共用的 AddReverb 迴圈；Python 在兩處呼叫點各自內聯） | — | — |
| `src/residual_echo_estimator.c:ree_update_reverb_linear` | `modules/residual/residual_echo_estimator.py:ResidualEchoEstimator._update_reverb_linear` | mirror | T-03 | — |
| `src/residual_echo_estimator.c:ree_estimate` | `modules/residual/residual_echo_estimator.py:ResidualEchoEstimator.estimate` | mirror | T-03 | — |

### 1.12 `reverb_model.c`　reverb_model.c（殘響能量模型）

依據：C 的 `reverb_model.c` 對應 `reverb_model.py`，全部 float32。`ReverbModel.reverb` 為 getter，C 以結構欄位存取（見 §3）。

| C 檔:函式 | Python 檔:符號 | 關係 | 已知落差 | 對拍工具／測試 |
|---|---|---|---| --- |
| `src/reverb_model.c:reverb_model_init` | `modules/residual/reverb_model.py:ReverbModel.__init__` | mirror | — | — |
| `src/reverb_model.c:reverb_model_reset` | `modules/residual/reverb_model.py:ReverbModel.reset` | mirror | — | — |
| `src/reverb_model.c:reverb_model_update_no_freq_shaping` | `modules/residual/reverb_model.py:ReverbModel.update_no_freq_shaping` | mirror | — | — |
| `src/reverb_model.c:reverb_model_update` | `modules/residual/reverb_model.py:ReverbModel.update` | mirror | — | — |

### 1.13 `reverb_frequency_response.c`　reverb_frequency_response.c（殘響頻率響應）

依據：C 的 `reverb_frequency_response.c` 對應 `reverb_frequency_response.py`。`tail_response`、`average_decay` 為 getter，C 以結構欄位存取（見 §3）。

| C 檔:函式 | Python 檔:符號 | 關係 | 已知落差 | 對拍工具／測試 |
|---|---|---|---| --- |
| `src/reverb_frequency_response.c:reverb_freq_resp_init` | `modules/residual/reverb_frequency_response.py:ReverbFrequencyResponse.__init__` | mirror | — | — |
| `src/reverb_frequency_response.c:reverb_freq_resp_reset` | `modules/residual/reverb_frequency_response.py:ReverbFrequencyResponse.reset` | mirror | — | — |
| `src/reverb_frequency_response.c:reverb_freq_resp_update` | `modules/residual/reverb_frequency_response.py:ReverbFrequencyResponse.update` | mirror | — | — |

### 1.14 `stationarity_estimator.c`　stationarity_estimator.c（頻帶／區塊穩態判定）

依據：C 的 `stationarity_estimator.c` 對應 `python/modules/state/stationarity_estimator.py`；static 函式名即 Python 私有方法名（`_NoiseSpectrum`、`_alpha_now`、`_update_hangover`、`_smooth_per_freq`）。Python 模組層函式 `get_residual_echo_scaling` 在 C 內沒有獨立函式（見 §3）。

| C 檔:函式 | Python 檔:符號 | 關係 | 已知落差 | 對拍工具／測試 |
|---|---|---|---| --- |
| `src/stationarity_estimator.c:noise_spectrum_init` | `modules/state/stationarity_estimator.py:_NoiseSpectrum.__init__` | mirror | — | — |
| `src/stationarity_estimator.c:noise_spectrum_reset` | `modules/state/stationarity_estimator.py:_NoiseSpectrum.reset` | mirror | — | — |
| `src/stationarity_estimator.c:noise_spectrum_alpha_now` | `modules/state/stationarity_estimator.py:_NoiseSpectrum._alpha_now` | mirror | — | — |
| `src/stationarity_estimator.c:noise_spectrum_update` | `modules/state/stationarity_estimator.py:_NoiseSpectrum.update` | mirror | — | — |
| `src/stationarity_estimator.c:stationarity_estimator_init` | `modules/state/stationarity_estimator.py:StationarityEstimator.__init__` | mirror | — | — |
| `src/stationarity_estimator.c:stationarity_estimator_reset` | `modules/state/stationarity_estimator.py:StationarityEstimator.reset` | mirror | — | — |
| `src/stationarity_estimator.c:stationarity_estimator_window_hops` | `modules/state/stationarity_estimator.py:StationarityEstimator.__init__` | mirror（歷史環的列數，與 `__init__` 的視窗長度同一運算式；C 供 lane 精確配置） | — | `ctest:test_aec_array.c` |
| `src/stationarity_estimator.c:stationarity_estimator_update_noise_estimator` | `modules/state/stationarity_estimator.py:StationarityEstimator.update_noise_estimator` | mirror | — | — |
| `src/stationarity_estimator.c:update_hangover` | `modules/state/stationarity_estimator.py:StationarityEstimator._update_hangover` | mirror | — | — |
| `src/stationarity_estimator.c:smooth_per_freq` | `modules/state/stationarity_estimator.py:StationarityEstimator._smooth_per_freq` | mirror | — | — |
| `src/stationarity_estimator.c:stationarity_estimator_update_stationarity_flags` | `modules/state/stationarity_estimator.py:StationarityEstimator.update_stationarity_flags` | mirror | — | — |
| `src/stationarity_estimator.c:stationarity_estimator_band_stationary_mask` | `modules/state/stationarity_estimator.py:StationarityEstimator.band_stationary_mask` | mirror | — | — |
| `src/stationarity_estimator.c:stationarity_estimator_is_block_stationary` | `modules/state/stationarity_estimator.py:StationarityEstimator.is_block_stationary` | mirror | — | — |

### 1.15 `filter_analyzer.c`　filter_analyzer.c（濾波器分析：延遲、一致性、峰值）

依據：C 的 `filter_analyzer.c` 對應 `python/modules/state/filter_analyzer.py`；`cd_*` 對應 Python 的 `_ConsistentFilterDetector`。Python 的 `filter_delays_blocks` 為 getter，C 以結構欄位存取（見 §3）。

| C 檔:函式 | Python 檔:符號 | 關係 | 已知落差 | 對拍工具／測試 |
|---|---|---|---| --- |
| `src/filter_analyzer.c:fa_floor_scale` | `modules/state/filter_analyzer.py:_ConsistentFilterDetector.__init__` | partial（推定：48 kHz 格點的 floor 取樣數倍率：Python 在 `__init__` 內以 `scale = 3 if sample_rate == 48000 else 1` 內嵌） | — | — |
| `src/filter_analyzer.c:cd_reset` | `modules/state/filter_analyzer.py:_ConsistentFilterDetector.reset` | mirror | — | — |
| `src/filter_analyzer.c:cd_accumulate_slice` | `modules/state/filter_analyzer.py:_ConsistentFilterDetector.detect` | partial（`detect` 內 `abs` 切片加總與最大值的抽出） | — | — |
| `src/filter_analyzer.c:cd_detect` | `modules/state/filter_analyzer.py:_ConsistentFilterDetector.detect` | mirror | — | — |
| `src/filter_analyzer.c:fa_init` | `modules/state/filter_analyzer.py:FilterAnalyzer.__init__`<br>`modules/state/filter_analyzer.py:_ConsistentFilterDetector.__init__` | mirror | — | `ctest:test_counter_saturation.c` |
| `src/filter_analyzer.c:fa_reset` | `modules/state/filter_analyzer.py:FilterAnalyzer.reset` | mirror | — | — |
| `src/filter_analyzer.c:fa_set_region` | `modules/state/filter_analyzer.py:FilterAnalyzer._set_region` | mirror | — | — |
| `src/filter_analyzer.c:fa_set_taps_provider` | — | C-only（tap provider 回呼註冊；Python 直接傳入濾波器係數） | — | — |
| `src/filter_analyzer.c:fa_preprocess` | `modules/state/filter_analyzer.py:FilterAnalyzer._preprocess` | mirror | — | — |
| `src/filter_analyzer.c:fa_update` | `modules/state/filter_analyzer.py:FilterAnalyzer.update` | mirror | — | `ctest:test_counter_saturation.c` |
| `src/filter_analyzer.c:fa_min_filter_delay_blocks` | `modules/state/filter_analyzer.py:FilterAnalyzer.min_filter_delay_blocks` | mirror | — | — |
| `src/filter_analyzer.c:fa_max_echo_path_gain` | `modules/state/filter_analyzer.py:FilterAnalyzer.max_echo_path_gain` | mirror | — | — |

### 1.16 `filter_delay.c`　filter_delay.c（濾波器延遲彙整）

依據：C 的 `filter_delay.c` 對應 `python/modules/state/filter_delay.py`。Python 的 `direct_path_filter_delays` 為 getter，C 以結構欄位存取（見 §3）；C 的 external_delay 只保留 `reported` 旗標，沒有 samples 值。

| C 檔:函式 | Python 檔:符號 | 關係 | 已知落差 | 對拍工具／測試 |
|---|---|---|---| --- |
| `src/filter_delay.c:filter_delay_init` | `modules/state/filter_delay.py:FilterDelay.__init__` | partial（C 的 external_delay 沒有 quality 欄位） | T-03 | — |
| `src/filter_delay.c:filter_delay_update` | `modules/state/filter_delay.py:FilterDelay.update` | mirror | T-03 | — |
| `src/filter_delay.c:filter_delay_min_direct_path` | `modules/state/filter_delay.py:FilterDelay.min_direct_path_filter_delay` | mirror | T-03 | — |

### 1.17 `filter_quality.c`　filter_quality.c（線性濾波器可用性閘門）

依據：C 的 `filter_quality.c` 對應 `python/modules/state/filter_quality.py`。

| C 檔:函式 | Python 檔:符號 | 關係 | 已知落差 | 對拍工具／測試 |
|---|---|---|---| --- |
| `src/filter_quality.c:fq_init` | `modules/state/filter_quality.py:FilteringQualityAnalyzer.__init__` | mirror | — | `ctest:test_counter_saturation.c` |
| `src/filter_quality.c:fq_reset` | `modules/state/filter_quality.py:FilteringQualityAnalyzer.reset` | mirror | — | — |
| `src/filter_quality.c:fq_update` | `modules/state/filter_quality.py:FilteringQualityAnalyzer.update` | mirror | — | `ctest:test_counter_saturation.c` |
| `src/filter_quality.c:fq_linear_filter_usable` | `modules/state/filter_quality.py:FilteringQualityAnalyzer.linear_filter_usable` | mirror | — | — |

### 1.19 `fullband_erle.c`　fullband_erle.c（全頻帶 ERLE 估計）

依據：C 的 `fullband_erle.c` 對應 `python/modules/state/fullband_erle.py`；`_ErleInstantaneous` 對應 C 的 `inst_*` static 函式。

| C 檔:函式 | Python 檔:符號 | 關係 | 已知落差 | 對拍工具／測試 |
|---|---|---|---| --- |
| `src/fullband_erle.c:fast_log2` | `modules/state/fullband_erle.py:_fast_log2` | mirror | — | — |
| `src/fullband_erle.c:inst_init` | `modules/state/fullband_erle.py:_ErleInstantaneous.__init__` | mirror | — | `ctest:test_neon_twins.c` |
| `src/fullband_erle.c:inst_reset_accumulators` | `modules/state/fullband_erle.py:_ErleInstantaneous.reset_accumulators` | mirror | — | — |
| `src/fullband_erle.c:inst_reset` | `modules/state/fullband_erle.py:_ErleInstantaneous.reset` | mirror | — | — |
| `src/fullband_erle.c:inst_update_max_min` | `modules/state/fullband_erle.py:_ErleInstantaneous._update_max_min` | mirror | — | — |
| `src/fullband_erle.c:inst_update_quality_estimate` | `modules/state/fullband_erle.py:_ErleInstantaneous._update_quality_estimate` | mirror | — | — |
| `src/fullband_erle.c:inst_update` | `modules/state/fullband_erle.py:_ErleInstantaneous.update` | mirror | — | — |
| `src/fullband_erle.c:inst_get_quality_estimate` | `modules/state/fullband_erle.py:_ErleInstantaneous.get_quality_estimate` | mirror（`*valid==0` 對應 Python 回 None） | — | — |
| `src/fullband_erle.c:update_quality_estimate` | `modules/state/fullband_erle.py:FullBandErleEstimator._update_quality_estimate` | mirror | — | — |
| `src/fullband_erle.c:fb_erle_init` | `modules/state/fullband_erle.py:FullBandErleEstimator.__init__` | partial（C 不帶 max_erle_l 參數與 max_erle_lf_log2） | — | `ctest:test_counter_saturation.c` |
| `src/fullband_erle.c:fb_erle_reset` | `modules/state/fullband_erle.py:FullBandErleEstimator.reset` | mirror | — | — |
| `src/fullband_erle.c:fb_erle_update` | `modules/state/fullband_erle.py:FullBandErleEstimator.update` | mirror | — | `ctest:test_counter_saturation.c` |
| `src/fullband_erle.c:fb_erle_log2` | `modules/state/fullband_erle.py:FullBandErleEstimator.fullband_erle_log2` | mirror | — | — |
| `src/fullband_erle.c:fb_erle_get_inst_linear_quality_estimate` | `modules/state/fullband_erle.py:FullBandErleEstimator.get_inst_linear_quality_estimate` | mirror | — | — |

### 1.20 `subband_erle.c`　subband_erle.c（子頻帶 ERLE 估計）

依據：C 的 `subband_erle.c` 對應 `python/modules/state/subband_erle.py`；static 函式名對應 Python 方法名（`_set_max_erle_bands`、`_smoothed_update`、`_update_accumulated_spectra`、`_update_bands`、`_decrease_erle_per_band_for_low_render_signals`）。

| C 檔:函式 | Python 檔:符號 | 關係 | 已知落差 | 對拍工具／測試 |
|---|---|---|---| --- |
| `src/subband_erle.c:set_max_erle_bands` | `modules/state/subband_erle.py:_set_max_erle_bands` | mirror | — | — |
| `src/subband_erle.c:subband_erle_init` | `modules/state/subband_erle.py:SubbandErleEstimator.__init__` | mirror | — | `ctest:test_neon_twins.c` |
| `src/subband_erle.c:subband_erle_reset` | `modules/state/subband_erle.py:SubbandErleEstimator.reset`<br>`modules/state/subband_erle.py:SubbandErleEstimator._reset_accumulated_spectra` | mirror（C 註解：reset 內含 `_reset_accumulated_spectra`） | — | — |
| `src/subband_erle.c:smoothed_update` | `modules/state/subband_erle.py:SubbandErleEstimator._smoothed_update` | mirror | — | — |
| `src/subband_erle.c:update_accumulated_spectra` | `modules/state/subband_erle.py:SubbandErleEstimator._update_accumulated_spectra` | mirror | — | — |
| `src/subband_erle.c:update_bands` | `modules/state/subband_erle.py:SubbandErleEstimator._update_bands` | mirror | — | — |
| `src/subband_erle.c:decrease_erle_for_low_render` | `modules/state/subband_erle.py:SubbandErleEstimator._decrease_erle_per_band_for_low_render_signals` | mirror | — | `ctest:test_neon_twins.c` |
| `src/subband_erle.c:subband_erle_update` | `modules/state/subband_erle.py:SubbandErleEstimator.update` | mirror | — | — |
| `src/subband_erle.c:subband_erle_erle` | `modules/state/subband_erle.py:SubbandErleEstimator.erle` | mirror | — | — |
| `src/subband_erle.c:subband_erle_erle_unbounded` | `modules/state/subband_erle.py:SubbandErleEstimator.erle_unbounded` | mirror | — | — |

### 1.21 `erle_estimator.c`　erle_estimator.c（ERLE 估計器外殼：子頻帶加全頻帶）

依據：C 的 `erle_estimator.c` 對應 `python/modules/state/erle_estimator.py`；先 subband 後 fullband 的順序沿用 .py。

| C 檔:函式 | Python 檔:符號 | 關係 | 已知落差 | 對拍工具／測試 |
|---|---|---|---| --- |
| `src/erle_estimator.c:erle_estimator_init` | `modules/state/erle_estimator.py:ErleEstimator.__init__` | mirror | T-03 | — |
| `src/erle_estimator.c:erle_estimator_reset` | `modules/state/erle_estimator.py:ErleEstimator.reset` | mirror | T-03 | — |
| `src/erle_estimator.c:erle_estimator_update` | `modules/state/erle_estimator.py:ErleEstimator.update` | mirror | T-03 | — |
| `src/erle_estimator.c:erle_estimator_erle` | `modules/state/erle_estimator.py:ErleEstimator.erle` | mirror | T-03 | — |
| `src/erle_estimator.c:erle_estimator_erle_unbounded` | `modules/state/erle_estimator.py:ErleEstimator.erle_unbounded` | mirror | T-03 | — |
| `src/erle_estimator.c:erle_estimator_fullband_erle_log2` | `modules/state/erle_estimator.py:ErleEstimator.fullband_erle_log2` | mirror | T-03 | — |
| `src/erle_estimator.c:erle_estimator_get_inst_linear_quality_estimate` | `modules/state/erle_estimator.py:ErleEstimator.get_inst_linear_quality_estimate` | mirror | T-03 | — |

### 1.22 `erl_estimator.c`　erl_estimator.c（逐頻帶 ERL 估計）

依據：C 的 `erl_estimator.c` 對應 `python/modules/state/erl_estimator.py`。

| C 檔:函式 | Python 檔:符號 | 關係 | 已知落差 | 對拍工具／測試 |
|---|---|---|---| --- |
| `src/erl_estimator.c:erl_estimator_init` | `modules/state/erl_estimator.py:ErlEstimator.__init__` | mirror | — | `ctest:test_counter_saturation.c` |
| `src/erl_estimator.c:erl_estimator_reset` | `modules/state/erl_estimator.py:ErlEstimator.reset` | mirror | — | — |
| `src/erl_estimator.c:erl_estimator_update` | `modules/state/erl_estimator.py:ErlEstimator.update` | mirror | — | `ctest:test_counter_saturation.c` |

### 1.23 `initial_state.c`　initial_state.c（啟動階段旗標）

依據：C 的 `initial_state.c` 對應 `python/modules/state/initial_state.py`。

| C 檔:函式 | Python 檔:符號 | 關係 | 已知落差 | 對拍工具／測試 |
|---|---|---|---| --- |
| `src/initial_state.c:initial_state_init` | `modules/state/initial_state.py:InitialState.__init__` | mirror | — | `ctest:test_counter_saturation.c` |
| `src/initial_state.c:initial_state_reset` | `modules/state/initial_state.py:InitialState.reset` | mirror | — | — |
| `src/initial_state.c:initial_state_update` | `modules/state/initial_state.py:InitialState.update` | mirror | — | `ctest:test_counter_saturation.c` |
| `src/initial_state.c:initial_state_transition_triggered` | `modules/state/initial_state.py:InitialState.transition_triggered` | mirror | — | — |

### 1.24 `render_signal_analyzer.c`　render_signal_analyzer.c（遠端訊號窄頻成分與激發度分析）

依據：C 的 `render_signal_analyzer.c` 對應 `python/modules/render/render_signal_analyzer.py`（AEC3 的窄頻區域與強峰識別）。`rsa_argmax_*`／`rsa_max_abs_*` 是共用 `sk_argmax_f32*`／`sk_max_abs_f32*` 的薄 wrapper（`np.argmax`／`np.max(np.abs())` 的等價掃描；max_abs 的 NaN 不參與最大值，numpy 則回傳 NaN）。

| C 檔:函式 | Python 檔:符號 | 關係 | 已知落差 | 對拍工具／測試 |
|---|---|---|---| --- |
| `src/render_signal_analyzer.c:rsa_init` | `modules/render/render_signal_analyzer.py:RenderSignalAnalyzer.__init__` | mirror | — | — |
| `src/render_signal_analyzer.c:rsa_reset` | `modules/render/render_signal_analyzer.py:RenderSignalAnalyzer.reset` | mirror | — | — |
| `src/render_signal_analyzer.c:identify_small_narrow_band_regions` | `modules/render/render_signal_analyzer.py:RenderSignalAnalyzer._identify_small_narrow_band_regions` | mirror | — | — |
| `src/render_signal_analyzer.c:rsa_argmax_first` | — | C-only（`np.argmax` 的 NEON／純量分派） | — | `ctest:test_neon_twins.c` |
| `src/render_signal_analyzer.c:rsa_max_abs` | — | C-only（`sk_max_abs_f32` 的 wrapper；`np.max(np.abs(x))`；NaN 輸入時 numpy 回傳 NaN，C 略過 NaN） | — | `ctest:test_neon_twins.c` |
| `src/render_signal_analyzer.c:identify_strong_narrow_band_component` | `modules/render/render_signal_analyzer.py:RenderSignalAnalyzer._identify_strong_narrow_band_component` | mirror | — | — |
| `src/render_signal_analyzer.c:rsa_update` | `modules/render/render_signal_analyzer.py:RenderSignalAnalyzer.update` | mirror | — | — |
| `src/render_signal_analyzer.c:rsa_poor_signal_excitation` | `modules/render/render_signal_analyzer.py:RenderSignalAnalyzer.poor_signal_excitation` | mirror | — | — |
| `src/render_signal_analyzer.c:rsa_any_narrow_band_active` | — | C-only（遮罩前的快速判斷；Python 在 filters.py 每次都呼叫遮罩） | — | — |
| `src/render_signal_analyzer.c:rsa_mask_regions_around_narrow_bands` | `modules/render/render_signal_analyzer.py:RenderSignalAnalyzer.mask_regions_around_narrow_bands` | mirror | S-22 | — |

### 1.25 `saturation.c`　saturation.c（輸入飽和偵測與軟削波）

依據：C 的 `saturation.c` 對應 Python 的 `SaturationDetector`（`modules/preprocessing.py`）。

| C 檔:函式 | Python 檔:符號 | 關係 | 已知落差 | 對拍工具／測試 |
|---|---|---|---| --- |
| `src/saturation.c:saturation_init` | `modules/preprocessing.py:SaturationDetector.__init__` | mirror | — | `ctest:test_config_validation.c` |
| `src/saturation.c:saturation_reset` | `modules/preprocessing.py:SaturationDetector.reset` | mirror | — | — |
| `src/saturation.c:saturation_detect` | `modules/preprocessing.py:SaturationDetector.detect` | mirror | T-03 | — |
| `src/saturation.c:saturation_soft_clip` | `modules/preprocessing.py:SaturationDetector.soft_clip` | mirror | T-03 | — |

### 1.26 `saturation_detector.c`　saturation_detector.c（回音飽和偵測，AEC3 SaturationDetector 的移植）

依據：C 的 `saturation_detector.c` 對應 `python/modules/state/saturation_detector.py`。

| C 檔:函式 | Python 檔:符號 | 關係 | 已知落差 | 對拍工具／測試 |
|---|---|---|---| --- |
| `src/saturation_detector.c:saturation_detector_init` | `modules/state/saturation_detector.py:SaturationDetector.__init__` | mirror | — | — |
| `src/saturation_detector.c:saturation_detector_update` | `modules/state/saturation_detector.py:SaturationDetector.update` | mirror | — | — |
| `src/saturation_detector.c:saturation_detector_saturated_echo` | `modules/state/saturation_detector.py:SaturationDetector.saturated_echo` | mirror | — | — |

### 1.27 `detectors.c`　detectors.c（遠端活動、濾波器收斂、雙講偵測）

依據：C 的 `detectors.c` 對應 `python/modules/detectors.py`。三個偵測器都有常設 Python↔C 閘門 `make test-detectors-parity-all`（每次由 Python 重產 golden，四個格點）。Python 的 `RenderActivityDetector.is_active`／`is_stationary`、`FilterConvergenceAnalyzer.converged` 等為 getter，C 以結構欄位或回傳值取得（見 §3）。

| C 檔:函式 | Python 檔:符號 | 關係 | 已知落差 | 對拍工具／測試 |
|---|---|---|---| --- |
| `src/detectors.c:render_activity_init` | `modules/detectors.py:RenderActivityDetector.__init__` | mirror | — | `live:test-detectors-parity-all`<br>`ctest:test_rate_structural.c` |
| `src/detectors.c:render_activity_reset` | `modules/detectors.py:RenderActivityDetector.reset` | mirror | — | — |
| `src/detectors.c:render_activity_update` | `modules/detectors.py:RenderActivityDetector.update` | mirror | — | `live:test-detectors-parity-all`<br>`ctest:test_rate_structural.c` |
| `src/detectors.c:filter_convergence_init` | `modules/detectors.py:FilterConvergenceAnalyzer.__init__` | mirror | — | `live:test-detectors-parity-all`<br>`ctest:test_counter_saturation.c` |
| `src/detectors.c:filter_convergence_reset` | `modules/detectors.py:FilterConvergenceAnalyzer.reset` | mirror | — | — |
| `src/detectors.c:filter_convergence_mark_diverged` | `modules/detectors.py:FilterConvergenceAnalyzer.mark_diverged` | mirror | — | `ctest:test_counter_saturation.c` |
| `src/detectors.c:filter_convergence_update_divergence` | `modules/detectors.py:FilterConvergenceAnalyzer.update_divergence` | mirror | — | `live:test-detectors-parity-all` |
| `src/detectors.c:filter_convergence_update_convergence` | `modules/detectors.py:FilterConvergenceAnalyzer.update_convergence` | mirror | — | `live:test-detectors-parity-all`<br>`ctest:test_counter_saturation.c` |
| `src/detectors.c:doubletalk_init` | `modules/detectors.py:DoubleTalkAnalyzer.__init__` | mirror | — | `live:test-detectors-parity-all` |
| `src/detectors.c:doubletalk_reset` | `modules/detectors.py:DoubleTalkAnalyzer.reset` | mirror | — | — |
| `src/detectors.c:doubletalk_update_shadow_dt` | `modules/detectors.py:DoubleTalkAnalyzer.update_shadow_dt` | mirror | — | `live:test-detectors-parity-all` |
| `src/detectors.c:doubletalk_update_energy_dt` | `modules/detectors.py:DoubleTalkAnalyzer.update_energy_dt` | mirror | — | `live:test-detectors-parity-all` |

### 1.28 `epc_shadow.c`　epc_shadow.c（回音路徑變化偵測與影子濾波器複製控制）

依據：C 的 `epc_shadow.c` 對應 `modules/epc.py` 的 `EchoPathChangeDetector` 與 `PathChangeRegimeHandler`（影子複製控制）。格點相關的建構入口只有 `shadow_copy_init_for_grid`。

| C 檔:函式 | Python 檔:符號 | 關係 | 已知落差 | 對拍工具／測試 |
|---|---|---|---| --- |
| `src/epc_shadow.c:epc_init` | `modules/epc.py:EchoPathChangeDetector.__init__` | mirror | T-03 | `ctest:test_counter_saturation.c` |
| `src/epc_shadow.c:epc_reset` | `modules/epc.py:EchoPathChangeDetector.reset` | mirror | T-03 | — |
| `src/epc_shadow.c:epc_force_delay` | `modules/epc.py:EchoPathChangeDetector.force_delay` | partial（C 不回傳事件，只設 `active` 與 `hangover`） | T-03 | `ctest:test_counter_saturation.c` |
| `src/epc_shadow.c:epc_update_epv` | `modules/epc.py:EchoPathChangeDetector.update_epv` | partial（C 的 EpcEvent 沒有 source 欄位） | T-03 | — |
| `src/epc_shadow.c:epc_update_shadow_rise` | `modules/epc.py:EchoPathChangeDetector.update_shadow_rise` | partial（C 的 EpcEvent 沒有 source 欄位） | T-03 | — |
| `src/epc_shadow.c:epc_tick_hangover` | `modules/epc.py:EchoPathChangeDetector.tick_hangover` | mirror | T-03 | `ctest:test_counter_saturation.c` |
| `src/epc_shadow.c:shadow_copy_init_for_grid` | `modules/epc.py:PathChangeRegimeHandler.__init__` | mirror | T-03 | `ctest:test_counter_saturation.c` |
| `src/epc_shadow.c:shadow_copy_reset` | `modules/epc.py:PathChangeRegimeHandler.reset` | mirror | T-03 | `ctest:test_counter_saturation.c` |
| `src/epc_shadow.c:dt_safe` | `modules/epc.py:PathChangeRegimeHandler._dt_safe` | mirror | T-03 | — |
| `src/epc_shadow.c:shadow_copy_update` | `modules/epc.py:PathChangeRegimeHandler.update` | partial（C 不回傳決策，只更新 `ShadowCopy` 狀態；沒有 boost_q（Kalman Q 增強已從 C 移除），也沒有 reverse_copy） | T-03 | `ctest:test_counter_saturation.c` |

### 1.29 `aec_array.c`　aec_array.c（四麥克風陣列實例：共用 delay、四條 lane、依 beam 權重融合）

依據：本檔是 `AecConfig.n_mics` 為陣列值時，由 `aec_*` 公開入口轉交的實作，對應 4ch 線性 AEC 加後波束 RES 流程。範圍內（`AEC/python`）沒有對應的 Python 類別或函式；Python 孿生在 `Audio_ALG/pipelines/4ch/pipeline.py`（範圍外，不在本表驗證）。各 lane 是一般的單通道 `Aec`（外部對齊 delay 模式），其逐 hop 數學對應 §1.1 的列。因此本檔全部列為 C-only；`aec_array_impl_*` 是同名公開入口在陣列實例上的實作，`aec_array_*` 計數與 `aec_get_array` 兩種實例皆可呼叫。

| C 檔:函式 | Python 檔:符號 | 關係 | 已知落差 | 對拍工具／測試 |
|---|---|---|---| --- |
| `src/aec_array.c:aec_array_get_timing` | — | C-only（各階段計時讀取（delay、前端、線性、lane 殘餘等）；單通道與陣列實例皆可呼叫的效能儀表） | — | `ctest:test_aec_array_flow.c` |
| `src/aec_array.c:aec_array_impl_apply_cng` | — | C-only（陣列實例的 `aec_apply_cng` 實作：對融合頻譜加舒適噪音，產生器種子由陣列實例持有） | — | `ctest:test_aec_array.c`<br>`ctest:test_aec_array_flow.c` |
| `src/aec_array.c:aec_array_impl_apply_gain` | — | C-only（陣列實例的 `aec_apply_gain` 實作：融合底座乘單一 `G_res`，可帶外部增益） | — | `ctest:test_aec_array.c`<br>`ctest:test_aec_array_flow.c` |
| `src/aec_array.c:aec_array_impl_config_ok` | — | C-only（陣列設定的合法性檢查（`n_mics`、`residual` 不可為 OFF、capture_proxy 範圍、三個產品格點，以及導出的 lane 設定可解析）；單通道只檢查隔離欄位） | — | `ctest:test_aec_array.c`<br>`ctest:test_aec_array_flow.c` |
| `src/aec_array.c:aec_array_impl_debug_status` | — | C-only（陣列實例的 `aec_debug_status` 實作：以 lane 0 為底，覆寫共用 delay 狀態、最大 windowed ERLE 與 duty 計數） | — | `ctest:test_aec_array.c`<br>`ctest:test_aec_array_flow.c` |
| `src/aec_array.c:aec_array_impl_destroy` | — | C-only（陣列實例的釋放（逐 lane 銷毀；堆積實例釋放整塊記憶體，池實例只清 ready 旗標）；Python 無手動釋放） | — | `ctest:test_aec_array.c`<br>`ctest:test_aec_array_flow.c` |
| `src/aec_array.c:aec_array_impl_far_fft_count` | — | C-only（陣列實例的 `aec_far_fft_real_compute_count` 實作：四條 lane 計數的飽和總和；遠端 FFT 計數儀表） | — | `ctest:test_aec_array.c`<br>`ctest:test_aec_array_flow.c` |
| `src/aec_array.c:aec_array_impl_get_linear` | — | C-only（陣列實例的 `aec_get_linear` 實作：以 lane 0 的線性檢視為底，對齊遠端、世代與 delay 欄位改用共用值） | — | `ctest:test_aec_array.c`<br>`ctest:test_aec_array_flow.c` |
| `src/aec_array.c:aec_array_impl_get_mem_breakdown` | — | C-only（陣列實例的靜態記憶體分項（估計器、ring、lane、融合）） | — | `ctest:test_aec_array.c`<br>`ctest:test_aec_array_flow.c` |
| `src/aec_array.c:aec_array_impl_get_mem_size` | — | C-only（陣列實例的靜態記憶體尺寸查詢） | — | `ctest:test_aec_array.c`<br>`ctest:test_aec_array_flow.c` |
| `src/aec_array.c:aec_array_impl_get_residual` | — | C-only（陣列實例的 `aec_get_residual` 實作：融合後的 near、R²、舒適噪音、渲染底座檢視） | — | `ctest:test_aec_array.c`<br>`ctest:test_aec_array_flow.c` |
| `src/aec_array.c:aec_array_impl_init` | — | C-only（陣列實例在呼叫端 pool 內的建構（共用 delay、對齊遠端與四條 lane 都切在同一個 pool）） | — | `ctest:test_aec_array.c`<br>`ctest:test_aec_array_flow.c` |
| `src/aec_array.c:aec_array_impl_is_cancelling` | — | C-only（陣列實例的 `aec_linear_is_cancelling` 實作：回報 capture proxy lane 的狀態） | — | `ctest:test_aec_array.c`<br>`ctest:test_aec_array_flow.c` |
| `src/aec_array.c:aec_array_impl_linear_lane` | — | C-only（`aec_get_linear()` 背後的 lane（lane 0）；被拒 hop 時 `aec_refuse_hop` 把它的線性檢視清零） | — | `ctest:test_dsp_refusal.c` |
| `src/aec_array.c:aec_array_impl_post_reset_serial` | — | C-only（陣列實例的 `aec_post_reset_serial` 實作：四條 lane 序號的和（模 2^32）） | — | `ctest:test_aec_array.c`<br>`ctest:test_aec_array_flow.c` |
| `src/aec_array.c:aec_array_impl_process_linear` | — | C-only（陣列實例的 `aec_process_linear` 實作：共用 delay、對齊遠端，再跑四條 lane 的線性 stage 與 lane 殘餘 stage；任一條 lane 的檢查不通過即 `aec_refuse_hop`） | — | `ctest:test_aec_array.c`<br>`ctest:test_aec_array_flow.c`<br>`ctest:test_dsp_refusal.c` |
| `src/aec_array.c:aec_array_impl_process_residual` | — | C-only（陣列實例的 `aec_process_residual` 實作：先驗證權重與 `beamformed_error`（無效回 `AEC_ERR_ARG`，不動實例），再以呼叫端的 beam 權重融合 lane context，`FULL` 另算融合頻譜上的增益；波束成形在函式庫外） | — | `ctest:test_aec_array.c`<br>`ctest:test_aec_array_flow.c` |
| `src/aec_array.c:aec_array_impl_reset` | — | C-only（陣列實例的 `aec_reset` 實作：重設共用 delay、四條 lane 與增益 stage，清對齊緩衝與陣列層級狀態） | — | `ctest:test_aec_array.c`<br>`ctest:test_aec_array_flow.c` |
| `src/aec_array.c:aec_array_impl_set_level` | — | C-only（陣列實例的 `aec_set_level` 實作：先驗證 level 查表，再切換增益 stage 並更新設定內的地板值） | — | `ctest:test_aec_array.c`<br>`ctest:test_aec_array_flow.c` |
| `src/aec_array.c:aec_array_impl_synthesis_window` | — | C-only（陣列實例的 `aec_synthesis_window` 實作：回傳 lane 0 的合成窗；合成在函式庫外） | — | `ctest:test_aec_array.c`<br>`ctest:test_aec_array_flow.c` |
| `src/aec_array.c:aec_array_linear_aec_count` | — | C-only（實際執行線性 AEC 的 lane 數儀表（陣列為 4，單通道為 1）） | — | `ctest:test_aec_array.c`<br>`ctest:test_aec_array_flow.c` |
| `src/aec_array.c:aec_array_matched_filter_count` | — | C-only（matched filter 的實例數儀表（陣列共用一份，不隨 lane 倍增）；單通道與陣列實例皆可呼叫） | — | `ctest:test_aec_array.c`<br>`ctest:test_aec_array_flow.c` |
| `src/aec_array.c:aec_testing_array_pending_delay_candidate` | — | C-only（測試專用鉤子，只在 `AEC_TESTING` 建置存在：讀取共用 delay 估計器待確認候選值；無待決時回 -1） | — | `ctest:test_aec_array.c`<br>`ctest:test_aec_array_flow.c` |
| `src/aec_array.c:aec_array_realign_soft_lane_count` | — | C-only（共用 delay 變動時 lane 以軟方式重對齊的次數儀表） | — | `ctest:test_aec_array.c`<br>`ctest:test_aec_array_flow.c` |
| `src/aec_array.c:aec_array_realign_warm_lane_count` | — | C-only（共用 delay 變動時 lane 以暖啟動方式重對齊的次數儀表） | — | `ctest:test_aec_array.c`<br>`ctest:test_aec_array_flow.c` |
| `src/aec_array.c:aec_get_array` | — | C-only（公開檢視：各 lane 的誤差、底座與守衛旗標、對齊遠端與共用 delay 狀態；陣列與單通道實例皆可呼叫（單通道為 lane 0）） | — | `ctest:test_aec_array.c`<br>`ctest:test_aec_array_flow.c` |
| `src/aec_array.c:aec_testing_array_duty_erle_leak_db` | — | C-only（測試專用鉤子，只在 `AEC_TESTING` 建置存在：讀取 duty 看門狗的 ERLE 洩漏值） | — | `ctest:test_aec_array_flow.c` |
| `src/aec_array.c:aec_testing_array_duty_erle_peak` | — | C-only（測試專用鉤子，只在 `AEC_TESTING` 建置存在：讀取 duty 看門狗的 ERLE 峰值） | — | `ctest:test_aec_array_flow.c` |
| `src/aec_array.c:aec_testing_array_far_spec_provenance` | — | C-only（測試專用鉤子，只在 `AEC_TESTING` 建置存在：讀取 lane 遠端頻譜共用 hop 的來源記號） | — | `ctest:test_aec_array_flow.c` |
| `src/aec_array.c:aec_testing_array_pin_duty_full_rate` | — | C-only（測試專用鉤子，只在 `AEC_TESTING` 建置存在：把 duty 週期釘為 1 hop（matched filter 每 hop 分析）） | — | `ctest:test_aec_array_flow.c` |
| `src/aec_array.c:aec_testing_array_set_far_hoist` | — | C-only（測試專用鉤子，只在 `AEC_TESTING` 建置存在：關閉遠端分析共用，讓每條 lane 全部自算，作為位元比對的對照組） | — | `ctest:test_aec_array_flow.c` |
| `src/aec_array.c:aec_testing_array_far_sync_lanes` | — | C-only（測試專用鉤子，只在 `AEC_TESTING` 建置存在：自上次重置起遠端歷史一直與 lane 0 相同的 lane 集合） | — | `ctest:test_aec_array_flow.c` |
| `src/aec_array.c:aec_testing_array_far_stale_lanes` | — | C-only（測試專用鉤子，只在 `AEC_TESTING` 建置存在：上一 hop 略過自身分析更新、以 lane 0 狀態代替的 lane 集合） | — | `ctest:test_aec_array_flow.c` |
| `src/aec_array.c:aec_testing_array_lane` | — | C-only（測試專用鉤子，只在 `AEC_TESTING` 建置存在：唯讀取得第 ch 條 lane，供與全自算對照組逐位元比對分析狀態） | — | `ctest:test_aec_array_flow.c` |
| `src/aec_array.c:align_render` | — | C-only（依共用 delay 把遠端對齊，寫入 `aligned_ref`；EXTERNAL_ALIGNED 直接複製，其餘走 delay ring；Python 孿生在範圍外的 4ch 管線） | — | `ctest:test_aec_array.c`<br>`ctest:test_aec_array_flow.c` |
| `src/aec_array.c:far_resync` | — | C-only（共用 delay 變動後，逐條比對 lane 與 lane 0 的遠端歷史；不同者移出共用集合，直到下次重置） | — | `ctest:test_aec_array_flow.c` |
| `src/aec_array.c:far_analysis_lanes` | — | C-only（本 hop 沿用 lane 0 遠端分析的 lane；其餘落後的 lane 在 lane 0 執行前先取得它的狀態，再自行更新） | — | `ctest:test_aec_array_flow.c` |
| `src/aec_array.c:array_clear_state` | — | C-only（清除陣列層級的逐 hop、duty 與融合狀態（建構與 `aec_array_impl_reset` 共用）） | — | `ctest:test_aec_array.c`<br>`ctest:test_aec_array_flow.c` |
| `src/aec_array.c:live_array` | — | C-only（取出有效陣列實例的控制區塊；單聲道、NULL 或已銷毀的 handle 回 NULL，診斷函式據此回中性值） | — | `ctest:test_aec_array.c` |
| `src/aec_array.c:array_derive_lane` | — | C-only（由陣列設定導出 lane 設定：單通道、EXTERNAL_ALIGNED delay 模式、lane 殘餘模式（STATE 或 ESTIMATE）） | — | `ctest:test_aec_array.c`<br>`ctest:test_aec_array_flow.c` |
| `src/aec_array.c:array_dims` | `modules/config.py:ref_ring_samples` | partial（解析陣列設定並算出各緩衝尺寸與 pool 總需求；其中 MATCHED 共用參考環與單通道同式＝estimator 可回報延遲上限＋hop。`python/` 沒有陣列實例，環公式對照單通道 helper；陣列的 Python 孿生在 Audio_ALG `pipelines/4ch/pipeline.py`，以 `max_delay_ms` 定環） | D-17 | `ctest:test_aec_array.c`<br>`ctest:test_aec_array_flow.c` |
| `src/aec_array.c:array_gain_ready` | — | C-only（`aec_array_impl_apply_gain`／`apply_cng` 的前置檢查：需 `FULL` 且該 hop 已融合） | — | `ctest:test_aec_array.c`<br>`ctest:test_aec_array_flow.c` |
| `src/aec_array.c:array_setup` | — | C-only（陣列實例的 pool 切片與初始化（靜態 pool 與堆積配置共用）） | — | `ctest:test_aec_array.c`<br>`ctest:test_aec_array_flow.c` |
| `src/aec_array.c:bind_lane_view` | — | C-only（把一條 lane 的 per-hop 緩衝綁成陣列層級的借用視圖（不複製）） | — | `ctest:test_aec_array.c`<br>`ctest:test_aec_array_flow.c` |
| `src/aec_array.c:flow_buffer_size` | — | C-only（陣列流程緩衝（hop 緩衝、融合用複數與浮點緩衝、delay ring）的位元組計算） | — | `ctest:test_aec_array.c`<br>`ctest:test_aec_array_flow.c` |
| `src/aec_array.c:fuse_contexts` | — | C-only（以 beam 權重融合四條 lane 的誤差、near、R²、舒適噪音（R² 為相干、舒適噪音為功率和）） | — | `ctest:test_aec_array.c`<br>`ctest:test_aec_array_flow.c` |
| `src/aec_array.c:lane_contributes` | — | C-only（判斷一條 lane 的權重是否有非零項而參與融合） | — | `ctest:test_aec_array.c`<br>`ctest:test_aec_array_flow.c` |
| `src/aec_array.c:max_lane_erle_windowed` | — | C-only（四條 lane 的 windowed ERLE 最大值，作為 duty 看門狗訊號） | — | `ctest:test_aec_array.c`<br>`ctest:test_aec_array_flow.c` |
| `src/aec_array.c:project_render_base` | — | C-only（把各 lane 的渲染底座依權重投影成融合底座；沒有任何貢獻 lane 的輸出守衛觸發時直接用 beam 的 E） | — | `ctest:test_aec_array.c`<br>`ctest:test_aec_array_flow.c` |
| `src/aec_array.c:update_shared_delay` | — | C-only（共用 delay 估計器的逐 hop 更新（EXTERNAL_ALIGNED／FIXED 直接給值；MATCHED 含 duty 降頻、候選值確認與接受）） | — | `ctest:test_aec_array.c`<br>`ctest:test_aec_array_flow.c` |

## 2. header 內的 static inline

### 2.1 `aec_balanced_config.h`　aec_balanced_config.h（自動產生的 balanced 速率尺寸表；僅列 static inline）

依據：標頭自述由 `python/diag/gen_aec_balanced_config_h.py` 從 live Python balanced 實例產生，DO NOT EDIT BY HAND。

| C 檔:函式 | Python 檔:符號 | 關係 | 已知落差 | 對拍工具／測試 |
|---|---|---|---| --- |
| `include/internal/aec_balanced_config.h:aecb_rate_cfg` | — | C-only（依 sample_rate 與 fft_size 查表；表內容由 `python/diag/gen_aec_balanced_config_h.py` 從 Python balanced 實例產生，範圍內沒有對應的 Python 函式） | — | `ctest:test_config_validation.c` |

### 2.2 `pbfdkf.h`　pbfdkf.h（`include/internal/`，static inline）

依據：函式與 `PBFDAF.tap_span` 同義。

| C 檔:函式 | Python 檔:符號 | 關係 | 已知落差 | 對拍工具／測試 |
|---|---|---|---| --- |
| `include/internal/pbfdkf.h:pbfdaf_tap_span` | `modules/filters.py:PBFDAF.tap_span` | mirror | — | — |

### 2.4 共用 AEC SIMD 核心

AEC 的 C 呼叫端直接包含 `utility/simd_aec.h`。NLMS／Kalman 權重更新、複數模平方、CNG EMA、ERL 追蹤、pairwise 加總與穩態遮罩等共用核心位於 `audiolib/common/utility/simd_aec.h`，由 `Audio_ALG/docs/c_python_function_map.md` 的 AC-SIMD-07 統計；依規則 3，不在本表重複列出。相容別名與搬移紀錄見 CHANGELOG。

PBF/Kalman H-error refresh、subband ERLE accumulation/release、render peak counters、stationarity hangover/smoothing/threshold 與 saturation predicate counts 的 array kernels 位於 `audiolib/common/utility/simd_aec.h`（Audio_ALG 對照表 AC-SIMD-07）；上表各 local wrapper 與 Python 方法對應不變。`ctest:simd_selftest_aec.c` 逐位元比較這八組純量與 SIMD 核心，涵蓋所有 lane 邊界、三種頻點數與狀態分支。

## 3. Python-only：沒有任何 C 函式對應的 Python 符號

下列符號沒有被 §1／§2 任何一列引用。「C 以資料結構、列舉或常數承載」的類別，其方法已在 §1 逐一對照；「C 以結構欄位或回傳值直接取得」的是 Python 的 getter 或薄包裝；「C 無此機制」才是真正的功能缺口（多為研究旋鈕、診斷介面、未上線的參考移植）。

| C 檔:函式 | Python 檔:符號 | 關係 | 已知落差 | 對拍工具／測試 |
|---|---|---|---| --- |
| — | `modules/_rates.py:aec3_blocks_to_our_hops`、`modules/_rates.py:ms_to_hops`、`modules/_rates.py:ms_to_aec3_blocks` | Python-only（C 無此機制；LEGACY 速率 helper：唯一匯入者是未上線的 render_delay_controller.py，C 的換算走 aec_scale.c） | — | — |
| `struct:AecSignalGrid` | `modules/config.py:SignalGrid` | Python-only（C 以資料結構、列舉或常數承載；格點描述） | — | `pytest:test_signal_grid.py` |
| — | `modules/config.py:AecConfig._resolve_delay_mode` | Python-only（C 無此機制；Python 把舊式 `enable_delay_est=False` 折成 `delay_mode`，C 沒有該欄位，`delay_mode` 直接驗證） | — | `pytest:test_delay_mode.py` |
| — | `modules/config.py:AecConfig.psd_scale` | Python-only（C 以結構欄位或回傳值直接取得；常數 `AEC_PSD_SCALE`） | — | — |
| `struct:AecResidualView` | `modules/dataclasses.py:AecResContext` | Python-only（C 以資料結構、列舉或常數承載；res-context 接縫的資料結構，兩邊欄位集合不同） | D-05 | — |
| `struct:RenderActivityResult` | `modules/dataclasses.py:RenderActivityState` | Python-only（C 以資料結構、列舉或常數承載；逐幀遠端活動摘要） | — | — |
| — | `modules/dataclasses.py:FilterConvergenceState` | Python-only（C 以結構欄位或回傳值直接取得；C 由 `filter_convergence_update_convergence` 的回傳值與 `FilterConvergence` 欄位取得） | — | — |
| — | `modules/dataclasses.py:RegimeHandlerDecision` | Python-only（C 以結構欄位或回傳值直接取得；C 讀 `ShadowCopy.main_paused`，沒有 `reverse_copy`） | — | — |
| `struct:EpcEvent` | `modules/dataclasses.py:EpcEvent` | Python-only（C 以資料結構、列舉或常數承載；回音路徑變化事件） | — | — |
| — | `modules/dataclasses.py:AecStats` | Python-only（C 無此機制；C 側最接近的是 `AecDebugStatus`（欄位集合不同）；Python 的 `get_stats` 路徑在 C 沒有） | D-12 | — |
| — | `modules/dataclasses.py:AecEventType`、`modules/dataclasses.py:AecEvent`、`modules/dataclasses.py:AecEvent.audio_path_changed` | Python-only（C 無此機制；AEC3 風格事件分類，orchestrator 未使用（僅 trace）） | — | — |
| — | `modules/debug_logger.py:AecDebugLogger`、`modules/debug_logger.py:AecDebugLogger.__init__`、`modules/debug_logger.py:AecDebugLogger.update` | Python-only（C 無此機制；每秒主控台傾印，只讀 AEC 狀態；C 沒有對應設施） | D-12 | — |
| `struct:ClockdriftLevel` | `modules/delay/clockdrift_detector.py:ClockdriftLevel` | Python-only（C 以資料結構、列舉或常數承載；列舉） | — | — |
| `struct:DaClockdrift` | `modules/delay/clockdrift_detector.py:ClockdriftDetector` | Python-only（C 以資料結構、列舉或常數承載；方法已對照到 `da_clockdrift_*`） | — | — |
| — | `modules/delay/clockdrift_detector.py:ClockdriftDetector.level` | Python-only（C 以結構欄位或回傳值直接取得；`DaClockdrift` 欄位） | — | — |
| `struct:DelayQuality` | `modules/delay/delay_types.py:DelayQuality` | Python-only（C 以資料結構、列舉或常數承載；列舉） | — | — |
| — | `modules/delay/delay_types.py:DelayAdjustment` | Python-only（C 以資料結構、列舉或常數承載；`AEC_POST_DELAY_ADJ_*` 巨集（aec_post.h）） | — | — |
| — | `modules/delay/delay_types.py:DelayEstimate`、`modules/delay/delay_types.py:EchoPathVariability`、`modules/delay/delay_types.py:EchoPathVariability.audio_path_changed` | Python-only（C 無此機制；只被未上線的 render_delay_controller.py 使用） | — | — |
| `struct:DaBiquad` | `modules/delay/downsampled_ring.py:_CascadedBiquad` | Python-only（C 以資料結構、列舉或常數承載；方法已對照到 `da_biquad_*`） | — | — |
| `struct:DaDecimator` | `modules/delay/downsampled_ring.py:Decimator` | Python-only（C 以資料結構、列舉或常數承載；方法已對照到 `da_decimator_*`） | — | — |
| — | `modules/delay/downsampled_ring.py:Decimator.reset` | Python-only（C 以結構欄位或回傳值直接取得；C 的降頻器狀態隨 `da_biquad_reset` 一併重設，沒有獨立函式） | D-03 | — |
| `struct:DaRing` | `modules/delay/downsampled_ring.py:DownsampledRenderBuffer` | Python-only（C 以資料結構、列舉或常數承載；方法已對照到 `da_ring_*`） | — | — |
| — | `modules/delay/downsampled_ring.py:DownsampledRenderBuffer.__init__` | Python-only（C 以結構欄位或回傳值直接取得；C 的環形緩衝由 `da_delay_carve` 綁定、`da_ring_reset` 歸零） | — | — |
| — | `modules/delay/downsampled_ring.py:DownsampledRenderBuffer.newest_index`、`modules/delay/downsampled_ring.py:required_capacity` | Python-only（C 以結構欄位或回傳值直接取得；`DaRing` 欄位與編譯期容量常數） | — | — |
| `struct:DaResample48` | `modules/delay/echo_path_delay_estimator.py:_Resample48` | Python-only（C 以資料結構、列舉或常數承載；方法已對照到 `da_resample48_*`） | — | — |
| `struct:DaEstimator` | `modules/delay/echo_path_delay_estimator.py:EchoPathDelayEstimator` | Python-only（C 以資料結構、列舉或常數承載；方法已對照到 `da_estimator_*`） | — | — |
| — | `modules/delay/echo_path_delay_estimator.py:EchoPathDelayEstimator.consistent_estimate_counter`、`modules/delay/echo_path_delay_estimator.py:EchoPathDelayEstimator.clockdrift_level`、`modules/delay/echo_path_delay_estimator.py:EchoPathDelayEstimator.reliable_delay_found` | Python-only（C 以結構欄位或回傳值直接取得；`DaEstimator` 欄位） | — | — |
| — | `modules/delay/lag_aggregator.py:DelaySelectionThresholds` | Python-only（C 以資料結構、列舉或常數承載；C 以編譯期常數承載門檻） | — | — |
| `struct:DaHighestPeak` | `modules/delay/lag_aggregator.py:_HighestPeakAggregator` | Python-only（C 以資料結構、列舉或常數承載；方法已對照到 `da_highest_peak_*`） | — | — |
| — | `modules/delay/lag_aggregator.py:_HighestPeakAggregator.candidate`、`modules/delay/lag_aggregator.py:_HighestPeakAggregator.histogram` | Python-only（C 以結構欄位或回傳值直接取得；`DaHighestPeak` 欄位） | — | — |
| — | `modules/delay/lag_aggregator.py:_PreEchoLagAggregator`、`modules/delay/lag_aggregator.py:_PreEchoLagAggregator.__init__`、`modules/delay/lag_aggregator.py:_PreEchoLagAggregator.reset`、`modules/delay/lag_aggregator.py:_PreEchoLagAggregator.aggregate`、`modules/delay/lag_aggregator.py:_PreEchoLagAggregator.candidate`、`modules/delay/lag_aggregator.py:_get_down_sampling_block_size_log2` | Python-only（C 無此機制；pre-echo 聚合與其狀態已從 C 移除（回報延遲一律取最高峰）） | — | — |
| `struct:DaLagAggregator` | `modules/delay/lag_aggregator.py:MatchedFilterLagAggregator` | Python-only（C 以資料結構、列舉或常數承載；方法已對照到 `da_aggregator_*`） | — | — |
| `struct:DelayAec` | `modules/delay/legacy_compat.py:LegacyDelayShim` | Python-only（C 以資料結構、列舉或常數承載；方法已對照到 `delay_aec_*`） | — | — |
| — | `modules/delay/legacy_compat.py:LegacyDelayShim.latest_variability` | Python-only（C 無此機制；回傳 EchoPathVariability 的診斷接口） | — | — |
| — | `modules/delay/matched_filter.py:LagEstimate` | Python-only（C 無此機制；pre-echo 延遲估計的回傳型別，C 略過 pre-echo 搜尋） | — | — |
| — | `modules/delay/matched_filter.py:_update_accumulated_error`、`modules/delay/matched_filter.py:_compute_pre_echo_lag` | Python-only（C 無此機制；pre-echo 起點搜尋：C 刻意略過（無消費者）） | — | — |
| `struct:DaMatchedFilter` | `modules/delay/matched_filter.py:MatchedFilter` | Python-only（C 以資料結構、列舉或常數承載；方法已對照到 `da_matched_filter_*`） | — | — |
| — | `modules/delay/matched_filter.py:MatchedFilter.get_best_lag_estimate`、`modules/delay/matched_filter.py:MatchedFilter.get_max_filter_lag`、`modules/delay/matched_filter.py:MatchedFilter.filter_intra_lag_shift`、`modules/delay/matched_filter.py:MatchedFilter.filter_size` | Python-only（C 以結構欄位或回傳值直接取得；`DaMatchedFilter` 欄位） | — | — |
| — | `modules/delay/render_delay_controller.py:_compute_buffer_delay_hops`、`modules/delay/render_delay_controller.py:RenderDelayController`、`modules/delay/render_delay_controller.py:RenderDelayController.__init__`、`modules/delay/render_delay_controller.py:RenderDelayController.consistent_estimate_counter`、`modules/delay/render_delay_controller.py:RenderDelayController.reliable_delay_found`、`modules/delay/render_delay_controller.py:RenderDelayController.has_clockdrift`、`modules/delay/render_delay_controller.py:RenderDelayController.reset`、`modules/delay/render_delay_controller.py:RenderDelayController.get_delay` | Python-only（C 無此機制；LEGACY 參考移植，無上線呼叫者（模組自述）；C 沒有，delay 走 `delay_aec_*`） | — | — |
| — | `modules/detectors.py:_rehop`、`modules/detectors.py:_rehop_ms` | Python-only（C 無此機制；Python 的時間常數重定時 helper；C 在呼叫處直接用 `aec_ms_to_hops`／`aec_growth_rehop`） | — | — |
| `struct:RenderActivity` | `modules/detectors.py:RenderActivityDetector` | Python-only（C 以資料結構、列舉或常數承載；方法已對照到 `render_activity_*`） | — | — |
| — | `modules/detectors.py:RenderActivityDetector.is_active`、`modules/detectors.py:RenderActivityDetector.is_stationary` | Python-only（C 以結構欄位或回傳值直接取得；`is_active` 由 `RenderActivityResult` 回傳，`is_stationary` 取自 `RenderActivity` 的狀態欄） | — | — |
| `struct:FilterConvergence` | `modules/detectors.py:FilterConvergenceAnalyzer` | Python-only（C 以資料結構、列舉或常數承載；方法已對照到 `filter_convergence_*`） | — | — |
| — | `modules/detectors.py:FilterConvergenceAnalyzer.converged`、`modules/detectors.py:FilterConvergenceAnalyzer.once_converged`、`modules/detectors.py:FilterConvergenceAnalyzer.divergence` | Python-only（C 以結構欄位或回傳值直接取得；`FilterConvergence` 欄位） | — | — |
| `struct:DoubleTalk` | `modules/detectors.py:DoubleTalkAnalyzer` | Python-only（C 以資料結構、列舉或常數承載；方法已對照到 `doubletalk_*`） | — | — |
| — | `modules/detectors.py:DoubleTalkAnalyzer.dt_from_energy`、`modules/detectors.py:DoubleTalkAnalyzer.dt_from_shadow`、`modules/detectors.py:DoubleTalkAnalyzer.shadow_advantage` | Python-only（C 以結構欄位或回傳值直接取得；`DoubleTalk` 欄位） | — | — |
| — | `modules/detectors.py:FilterPlateauDetector`、`modules/detectors.py:FilterPlateauDetector.__init__`、`modules/detectors.py:FilterPlateauDetector.reset`、`modules/detectors.py:FilterPlateauDetector.update`、`modules/detectors.py:FilterPlateauDetector.attempts`、`modules/detectors.py:FilterPlateauDetector.last_reset_frame` | Python-only（C 無此機制；Python 類別目前沒有任何實例化處，C 端沒有移植） | — | — |
| — | `modules/enums.py:AecMode`、`modules/enums.py:AecFilterState` | Python-only（C 無此機制；Python 的濾波器模式與狀態列舉（FDAF／PBFDAF／PBFDKF／SUBBAND 等）；C 只實作 PBFDKF 路徑） | D-07 | — |
| `struct:AecLevel` | `modules/enums.py:AecPreset` | Python-only（C 以資料結構、列舉或常數承載；C 的 `AecLevel` 列舉有 `AEC_LEVEL_NO_FLOOR=3`；pytest 比對 aec.h 內 `AEC_LEVEL_*` 那四行的整數值） | D-04 | `pytest:test_preset_arg.py` |
| `struct:AecDelayMode` | `modules/enums.py:AecDelayMode`、`modules/enums.py:AecDelayMode.coerce` | Python-only（C 以資料結構、列舉或常數承載；列舉，`coerce` 只在 Python（接受字串）） | — | `pytest:test_delay_mode.py` |
| — | `modules/epc.py:classify_epc_event` | Python-only（C 無此機制；EpcEvent 到 AecEvent 的分類，orchestrator 未使用） | — | — |
| `struct:EpcDetector` | `modules/epc.py:EchoPathChangeDetector` | Python-only（C 以資料結構、列舉或常數承載；方法已對照到 `epc_*`） | — | — |
| — | `modules/epc.py:EchoPathChangeDetector.active`、`modules/epc.py:EchoPathChangeDetector.hangover_count`、`modules/epc.py:EchoPathChangeDetector.epv_gain_fast`、`modules/epc.py:EchoPathChangeDetector.epv_gain_slow` | Python-only（C 以結構欄位或回傳值直接取得；`EpcDetector` 欄位） | — | — |
| `struct:ShadowCopy` | `modules/epc.py:PathChangeRegimeHandler` | Python-only（C 以資料結構、列舉或常數承載；方法已對照到 `shadow_copy_*`） | — | — |
| — | `modules/epc.py:PathChangeRegimeHandler.main_paused`、`modules/epc.py:PathChangeRegimeHandler.copy_err_baseline`、`modules/epc.py:PathChangeRegimeHandler.copy_counter`、`modules/epc.py:PathChangeRegimeHandler.hys_streak_min`、`modules/epc.py:PathChangeRegimeHandler.copy_err_baseline_retention` | Python-only（C 以結構欄位或回傳值直接取得；`ShadowCopy` 欄位） | — | — |
| — | `modules/filter/filter_state_bridge.py:FilterStateBridge` | Python-only（C 無此機制；快照資料結構，C 不再建立該快照） | — | — |
| `struct:PBFDAF` | `modules/filters.py:PBFDAF` | Python-only（C 以資料結構、列舉或常數承載；方法已對照到 `pbfdaf_*`） | — | — |
| `struct:PBFDKF` | `modules/filters.py:PBFDKF` | Python-only（C 以資料結構、列舉或常數承載；方法已對照到 `pbfdkf_*`） | D-07 | — |
| — | `modules/filters.py:PBFDAF.zero_filter_partitions` | Python-only（C 無此機制；局部清零指定分區；C 的 PBFDAF 只有整體 `pbfdaf_reset_taps`） | — | — |
| `struct:Aec` | `modules/orchestrator.py:AEC` | Python-only（C 以資料結構、列舉或常數承載；方法已對照到 `aec_*`） | — | — |
| — | `modules/orchestrator.py:AEC._filter_converged`、`modules/orchestrator.py:AEC._filter_once_converged`、`modules/orchestrator.py:AEC._divergence_indicator`、`modules/orchestrator.py:AEC._shadow_advantage`、`modules/orchestrator.py:AEC.epc_active`、`modules/orchestrator.py:AEC.epc_hangover_count`、`modules/orchestrator.py:AEC._epv_gain_fast`、`modules/orchestrator.py:AEC._epv_gain_slow` | Python-only（C 以結構欄位或回傳值直接取得；orchestrator 對偵測器欄位的薄包裝 getter，C 直接讀結構欄位） | — | — |
| — | `modules/orchestrator.py:AEC._init_trace_state`、`modules/orchestrator.py:AEC._update_energy_diagnostics`、`modules/orchestrator.py:AEC._update_filter_state_diagnostics`、`modules/orchestrator.py:AEC._update_frequency_diagnostics`、`modules/orchestrator.py:AEC._update_state_diagnostics`、`modules/orchestrator.py:AEC._update_trajectory_diagnostics` | Python-only（C 無此機制；Python 的診斷狀態建立與逐 hop 追蹤） | D-12 | — |
| — | `modules/orchestrator.py:AEC.get_diagnostics`、`modules/orchestrator.py:AEC.get_filter_state`、`modules/orchestrator.py:AEC.get_stats`、`modules/orchestrator.py:AEC.GetStats`、`modules/orchestrator.py:AEC.get_erle`、`modules/orchestrator.py:AEC.get_formed_output` | Python-only（C 無此機制；Python 的診斷與取值 API；C 對應的是 `aec_debug_status`／`aec_get_linear`／`aec_get_residual`，欄位集合不同） | D-05；D-12 | — |
| — | `modules/orchestrator.py:AEC.dump_p53_trace`、`modules/orchestrator.py:AEC.enable_res_audit`、`modules/orchestrator.py:AEC.get_res_audit` | Python-only（C 無此機制；Python-only 的研究／稽核介面） | D-12 | — |
| — | `modules/orchestrator.py:process_wav_files`、`modules/orchestrator.py:main` | Python-only（C 無此機制；Python CLI；C 的 CLI 是 `aec_wav`（另一組旗標）） | S-08；D-13 | — |
| — | `modules/p52_regime_classifier.py:AcousticRegime`、`modules/p52_regime_classifier.py:RegimeClassification`、`modules/p52_regime_classifier.py:AcousticRegimeClassifier`、`modules/p52_regime_classifier.py:AcousticRegimeClassifier.__init__`、`modules/p52_regime_classifier.py:AcousticRegimeClassifier._label_for`、`modules/p52_regime_classifier.py:AcousticRegimeClassifier.classify` | Python-only（C 無此機制；離線分析工具，模組自述不得進入 process()） | — | `pytest:test_p52_regime.py` |
| — | `modules/preprocessing.py:HighPassFilter`、`modules/preprocessing.py:HighPassFilter.__init__`、`modules/preprocessing.py:HighPassFilter.process`、`modules/preprocessing.py:HighPassFilter.reset` | Python-only（C 無此機制；C 側走 audiolib 的共用 float32 HPF（不在 `AEC/c_impl/src`）） | D-01；S-04；T-06 | — |
| `struct:Saturation` | `modules/preprocessing.py:SaturationDetector` | Python-only（C 以資料結構、列舉或常數承載；方法已對照到 `saturation_*`） | — | — |
| `struct:RenderSignalAnalyzer` | `modules/render/render_signal_analyzer.py:RenderSignalAnalyzer` | Python-only（C 以資料結構、列舉或常數承載；方法已對照到 `rsa_*`） | — | — |
| `struct:ReeEchoModelConfig` | `modules/residual/residual_echo_estimator.py:EchoModelConfig`、`modules/residual/residual_echo_estimator.py:EpStrengthConfig`、`modules/residual/residual_echo_estimator.py:ReverbConfig` | Python-only（C 以資料結構、列舉或常數承載；Python 三個設定 dataclass，C 合併成 `ReeEchoModelConfig`） | — | — |
| `struct:ResidualEchoEstimator` | `modules/residual/residual_echo_estimator.py:ResidualEchoEstimator` | Python-only（C 以資料結構、列舉或常數承載；方法已對照到 `ree_*`） | — | — |
| — | `modules/residual/residual_echo_estimator.py:ResidualEchoEstimator.attach_reverb_decay_estimator` | Python-only（C 無此機制；C 的殘響衰減走靜態設定路徑，不綁 decay estimator） | — | — |
| — | `modules/residual/reverb_decay_estimator.py:_symmetric_arithmetic_sum`、`modules/residual/reverb_decay_estimator.py:_block_energy_average`、`modules/residual/reverb_decay_estimator.py:_block_energy_peak`、`modules/residual/reverb_decay_estimator.py:_LateReverbLinearRegressor`、`modules/residual/reverb_decay_estimator.py:_LateReverbLinearRegressor.__init__`、`modules/residual/reverb_decay_estimator.py:_LateReverbLinearRegressor.reset`、`modules/residual/reverb_decay_estimator.py:_LateReverbLinearRegressor.accumulate`、`modules/residual/reverb_decay_estimator.py:_LateReverbLinearRegressor.estimate_available`、`modules/residual/reverb_decay_estimator.py:_LateReverbLinearRegressor.estimate`、`modules/residual/reverb_decay_estimator.py:_EarlyReverbLengthEstimator`、`modules/residual/reverb_decay_estimator.py:_EarlyReverbLengthEstimator.__init__`、`modules/residual/reverb_decay_estimator.py:_EarlyReverbLengthEstimator.reset`、`modules/residual/reverb_decay_estimator.py:_EarlyReverbLengthEstimator.accumulate`、`modules/residual/reverb_decay_estimator.py:_EarlyReverbLengthEstimator.estimate`、`modules/residual/reverb_decay_estimator.py:ReverbDecayEstimator`、`modules/residual/reverb_decay_estimator.py:ReverbDecayEstimator.__init__`、`modules/residual/reverb_decay_estimator.py:ReverbDecayEstimator.decay_value`、`modules/residual/reverb_decay_estimator.py:ReverbDecayEstimator.decay`、`modules/residual/reverb_decay_estimator.py:ReverbDecayEstimator.reset`、`modules/residual/reverb_decay_estimator.py:ReverbDecayEstimator._reset_decay_estimation`、`modules/residual/reverb_decay_estimator.py:ReverbDecayEstimator._lazy_init_aec3`、`modules/residual/reverb_decay_estimator.py:ReverbDecayEstimator.update`、`modules/residual/reverb_decay_estimator.py:ReverbDecayEstimator._update_legacy`、`modules/residual/reverb_decay_estimator.py:ReverbDecayEstimator._update_aec3_strict`、`modules/residual/reverb_decay_estimator.py:ReverbDecayEstimator._analyze_block_gain`、`modules/residual/reverb_decay_estimator.py:ReverbDecayEstimator._analyze_filter`、`modules/residual/reverb_decay_estimator.py:ReverbDecayEstimator._estimate_decay` | Python-only（C 無此機制；`src/` 沒有移植；線上 C 走靜態衰減） | T-03 | — |
| `struct:ReverbFrequencyResponse` | `modules/residual/reverb_frequency_response.py:ReverbFrequencyResponse` | Python-only（C 以資料結構、列舉或常數承載；方法已對照到 `reverb_freq_resp_*`） | — | — |
| — | `modules/residual/reverb_frequency_response.py:ReverbFrequencyResponse.tail_response`、`modules/residual/reverb_frequency_response.py:ReverbFrequencyResponse.average_decay` | Python-only（C 以結構欄位或回傳值直接取得；同名結構欄位） | — | — |
| `struct:ReverbModel` | `modules/residual/reverb_model.py:ReverbModel` | Python-only（C 以資料結構、列舉或常數承載；方法已對照到 `reverb_model_*`） | — | — |
| — | `modules/residual/reverb_model.py:ReverbModel.reverb` | Python-only（C 以結構欄位或回傳值直接取得；同名結構欄位） | — | — |
| — | `modules/residual/suppression_filter.py:apply_suppression` | Python-only（C 無此機制；Python 的增益套用 helper，orchestrator 未呼叫（僅被 package 匯出）；C 在 `aec_post_apply_output` 內套用增益） | — | — |
| `struct:SuppressionGainConfig` | `modules/residual/suppression_gain.py:MaskTuning`、`modules/residual/suppression_gain.py:SuppressorTuning`、`modules/residual/suppression_gain.py:HighFrequencySuppressionConfig`、`modules/residual/suppression_gain.py:EchoAudibilityConfig`、`modules/residual/suppression_gain.py:DominantNearendConfig`、`modules/residual/suppression_gain.py:SuppressorConfig` | Python-only（C 以資料結構、列舉或常數承載；Python 的設定 dataclass；C 合併成 `SuppressionGainConfig`／`SuppressionGainTuning`，Python-only 研究旋鈕沒有欄位） | S-22 | — |
| `struct:SuppressionGain` | `modules/residual/suppression_gain.py:SuppressionGain` | Python-only（C 以資料結構、列舉或常數承載；方法已對照到 `suppression_gain_*`） | — | — |
| — | `modules/residual/suppression_gain.py:_MovingAverageSpectrum`、`modules/residual/suppression_gain.py:_MovingAverageSpectrum.__init__` | Python-only（C 以結構欄位或回傳值直接取得；C 以 `SuppressionGain` 的 ma_buf 欄位承載，`nearend_smoother_average` 為其平均） | — | — |
| — | `modules/residual/suppression_gain.py:_MovingAverageSpectrum.update_memory_length` | Python-only（C 無此機制；Python 的執行期改記憶長度；C 在初始化時定長） | — | — |
| — | `modules/residual/suppression_gain.py:_LowNoiseRenderDetector`、`modules/residual/suppression_gain.py:_LowNoiseRenderDetector.__init__` | Python-only（C 以結構欄位或回傳值直接取得；C 在 `suppression_gain_init` 內初始化，狀態在 `SuppressionGain` 欄位） | — | — |
| — | `modules/residual/suppression_gain.py:_DominantNearendDetector`、`modules/residual/suppression_gain.py:_DominantNearendDetector.__init__`、`modules/residual/suppression_gain.py:_DominantNearendDetector._derive_trigger_threshold`、`modules/residual/suppression_gain.py:_DominantNearendDetector.set_config` | Python-only（C 以結構欄位或回傳值直接取得；C 在 `suppression_gain_init` 內展開設定，狀態在 `SuppressionGain` 欄位） | — | — |
| — | `modules/residual/suppression_gain.py:_DominantNearendDetector.is_nearend_state` | Python-only（C 以結構欄位或回傳值直接取得；`SuppressionGain` 欄位（見 `suppression_gain_is_dominant_nearend`）） | — | — |
| `struct:AecState` | `modules/state/aec_state.py:AecState` | Python-only（C 以資料結構、列舉或常數承載；方法已對照到 `aec_state_*`） | — | — |
| — | `modules/state/aec_state.py:AecState.external_delay_samples` | Python-only（C 無此值：`FilterDelayEstimate` 只有 `reported` 旗標） | — | — |
| `struct:ErlEstimator` | `modules/state/erl_estimator.py:ErlEstimator` | Python-only（C 以資料結構、列舉或常數承載；方法已對照到 `erl_estimator_*`） | — | — |
| `struct:ErleEstimator` | `modules/state/erle_estimator.py:ErleEstimator` | Python-only（C 以資料結構、列舉或常數承載；方法已對照到 `erle_estimator_*`） | — | — |
| `struct:FaConsistentDetector` | `modules/state/filter_analyzer.py:_ConsistentFilterDetector` | Python-only（C 以資料結構、列舉或常數承載；方法已對照到 `cd_*`） | — | — |
| `struct:FilterAnalyzer` | `modules/state/filter_analyzer.py:FilterAnalyzer` | Python-only（C 以資料結構、列舉或常數承載；方法已對照到 `fa_*`） | — | — |
| — | `modules/state/filter_analyzer.py:FilterAnalyzer.filter_delays_blocks` | Python-only（C 以結構欄位或回傳值直接取得；同名結構欄位） | — | — |
| `struct:FilterDelay` | `modules/state/filter_delay.py:FilterDelay` | Python-only（C 以資料結構、列舉或常數承載；方法已對照到 `filter_delay_*`） | — | — |
| — | `modules/state/filter_delay.py:FilterDelay.external_delay_samples` | Python-only（C 無此值：`FilterDelayEstimate` 只有 `reported` 旗標） | — | — |
| — | `modules/state/filter_delay.py:FilterDelay.direct_path_filter_delays` | Python-only（C 以結構欄位或回傳值直接取得；同名結構欄位） | — | — |
| `struct:FilteringQualityAnalyzer` | `modules/state/filter_quality.py:FilteringQualityAnalyzer` | Python-only（C 以資料結構、列舉或常數承載；方法已對照到 `fq_*`） | — | — |
| `struct:ErleInstantaneous` | `modules/state/fullband_erle.py:_ErleInstantaneous` | Python-only（C 以資料結構、列舉或常數承載；方法已對照到 `inst_*`） | — | — |
| — | `modules/state/fullband_erle.py:_ErleInstantaneous.get_inst_erle_log2` | Python-only（C 以結構欄位或回傳值直接取得；同名結構欄位） | — | — |
| `struct:FullBandErleEstimator` | `modules/state/fullband_erle.py:FullBandErleEstimator` | Python-only（C 以資料結構、列舉或常數承載；方法已對照到 `fb_erle_*`） | — | — |
| `struct:InitialState` | `modules/state/initial_state.py:InitialState` | Python-only（C 以資料結構、列舉或常數承載；方法已對照到 `initial_state_*`） | — | — |
| `struct:SaturationDetector` | `modules/state/saturation_detector.py:SaturationDetector` | Python-only（C 以資料結構、列舉或常數承載；方法已對照到 `saturation_detector_*`） | — | — |
| `struct:NoiseSpectrum` | `modules/state/stationarity_estimator.py:_NoiseSpectrum` | Python-only（C 以資料結構、列舉或常數承載；方法已對照到 `noise_spectrum_*`） | — | — |
| — | `modules/state/stationarity_estimator.py:_NoiseSpectrum.power` | Python-only（C 以結構欄位或回傳值直接取得；同名結構欄位） | — | — |
| `struct:StationarityEstimator` | `modules/state/stationarity_estimator.py:StationarityEstimator` | Python-only（C 以資料結構、列舉或常數承載；方法已對照到 `stationarity_estimator_*`） | — | — |
| — | `modules/state/stationarity_estimator.py:get_residual_echo_scaling` | Python-only（C 無此機制；模組層函式；C 內沒有獨立函式，`src/` 也沒有呼叫者） | — | — |
| `struct:SubbandErle` | `modules/state/subband_erle.py:SubbandErleEstimator` | Python-only（C 以資料結構、列舉或常數承載；方法已對照到 `subband_erle_*`） | — | — |
| — | `modules/aec3_scale.py:fft_density_scale` | Python-only（C 無此機制） | — | — |
| — | `modules/aec3_scale.py:nl_r2_norm_power` | Python-only（C 無此機制） | — | — |
| — | `modules/aec3_scale.py:psd_int16_to_float` | Python-only（C 無此機制） | — | — |
| — | `modules/filter/filter_state_bridge.py:build_filter_state_bridge` | Python-only（C 無此機制） | — | — |
| — | `modules/filters.py:PBFDAF.get_error_energy` | Python-only（C 無此機制） | — | — |
| — | `modules/filters.py:PBFDKF.copy_weights_from` | Python-only（C 無此機制） | — | — |
| — | `modules/orchestrator.py:AEC._arc_m_q_boost` | Python-only（C 無此機制；Kalman Q 增強：C 的 PBFDKF 已不保留 Q／R 狀態） | — | — |
| — | `modules/freq_utils.py:bin_to_hz` | Python-only（C 無此機制） | — | — |
| — | `modules/freq_utils.py:hz_to_bin` | Python-only（C 無此機制） | — | — |
| — | `modules/render/render_signal_analyzer.py:RenderSignalAnalyzer.narrow_peak_band` | Python-only（C 無此機制） | — | — |
| — | `modules/state/aec_state.py:AecState.active_render` | Python-only（C 無此機制） | — | — |
| — | `modules/state/aec_state.py:AecState.erl` | Python-only（C 無此機制） | — | — |
| — | `modules/state/aec_state.py:AecState.erl_time_domain` | Python-only（C 無此機制） | — | — |
| — | `modules/state/aec_state.py:AecState.filter_analyzer_consistent` | Python-only（C 無此機制） | — | — |
| — | `modules/state/aec_state.py:AecState.filter_analyzer_max_echo_path_gain` | Python-only（C 無此機制） | — | — |
| — | `modules/state/aec_state.py:AecState.filter_analyzer_peak_index` | Python-only（C 無此機制） | — | — |
| — | `modules/state/aec_state.py:AecState.initial_state_active` | Python-only（C 無此機制） | — | — |
| — | `modules/state/aec_state.py:AecState.saturated_capture` | Python-only（C 無此機制） | — | — |
| — | `modules/state/aec_state.py:AecState.transparent_mode_active` | Python-only（C 無此機制；C 的 `ree_estimate` 與 `fq_update` 的 `transparent_mode` 參數一律傳字面值 0） | — | — |
| — | `modules/state/erl_estimator.py:ErlEstimator.erl` | Python-only（C 無此機制） | — | — |
| — | `modules/state/erl_estimator.py:ErlEstimator.erl_time_domain` | Python-only（C 無此機制） | — | — |
| — | `modules/state/filter_analyzer.py:FilterAnalyzer.any_filter_consistent` | Python-only（C 無此機制） | — | — |
| — | `modules/state/filter_analyzer.py:FilterAnalyzer.get_adjusted_filters` | Python-only（C 無此機制） | — | — |
| — | `modules/state/filter_analyzer.py:FilterAnalyzer.peak_index` | Python-only（C 無此機制） | — | — |
| — | `modules/state/filter_delay.py:FilterDelay.external_delay_reported` | Python-only（C 無此機制） | — | — |
| — | `modules/state/filter_quality.py:FilteringQualityAnalyzer.gate1_pass` | Python-only（C 無此機制） | — | — |
| — | `modules/state/filter_quality.py:FilteringQualityAnalyzer.gate2_pass` | Python-only（C 無此機制） | — | — |
| — | `modules/state/filter_quality.py:FilteringQualityAnalyzer.reset_blocks` | Python-only（C 無此機制） | — | — |
| — | `modules/state/filter_quality.py:FilteringQualityAnalyzer.startup_blocks` | Python-only（C 無此機制） | — | — |
| — | `modules/state/initial_state.py:InitialState.initial_state_active` | Python-only（C 無此機制） | — | — |
| — | `modules/state/stationarity_estimator.py:StationarityEstimator.is_band_stationary` | Python-only（C 無此機制） | — | — |
| — | `modules/state/subband_erle.py:SubbandErleEstimator.erle_during_onsets` | Python-only（C 無此機制） | — | — |

---

## 附錄 A：落差編號

這裡只收本表引用到的編號。S＝sync_bug、D＝文件或註解明載的刻意差異、T＝缺對拍。

| 編號 | 說明 |
|---|---|
| D-01 | mic-path HPF：C 為 float32 DF2T，Python 為 fp64，非逐位元（文件化） |
| D-02 | `delay_est_period_s` 預設 Python 0.5／C 0.0 |
| D-03 | matched-filter duty-cycle 節流、reject list 固定 8 格、延遲鏈為 float32 非逐位元 |
| D-04 | preset 軸（C 稱 level）的承載形式不同：C 為三個查表函式加 `aec_set_level`，Python 為 `AecConfig.from_preset` 內的一張表加 `AEC.set_preset`；NO_FLOOR 兩邊在建構期與執行期都有；C 的執行期 ramp 時間是內部常數 `AEC_LEVEL_RAMP_MS`（100 ms），Python 為呼叫引數 |
| D-05 | `return_formed_output` 為 Python-only；Python 的 res-context 資料類別與 C 的 `AecResidualView`／`AecLinearView` 欄位兩邊不同 |
| D-06 | 預設為 True 的 AEC3 對齊旗標在 C 為編譯期寫死，無法在 runtime 關閉 |
| D-07 | C 只實作 PBFDKF；Python 另有 FDAF／PBFDAF／SUBBAND 與調校選項 |
| D-08 | 設定驗證規則不對稱（C 嚴格） |
| D-09 | `ms_to_hops` 的 double 版與 C 在整半格案例差 1 hop（`ms_to_hops_f32` 與 C 一致） |
| D-10 | `poor_excitation_counter`：C 有飽和上限、Python 無上限，re-arm 初值也不同（唯一消費者是 `counter < n_partitions` 閘，決策不變） |
| D-11 | C 的 `AecTuning.n_partitions` 可覆寫（庫內部，只有測試鉤子能設；公開 `AecConfig` 已無此欄位），Python 為唯讀 property |
| D-12 | C-only API（`aec_process_linear_shared_far_from`（`aec_internal.h`）、`AecConfig.residual` 的 `STATE`／`ESTIMATE` 模式與增益輔助函式 `aec_apply_gain`／`aec_apply_cng`、靜態記憶體、debug status）；Python-only：`get_res_audit`、`dump_p53_trace`、`get_diagnostics` |
| D-13 | CLI／API 兩側獨有項（C `aec_wav` 與 Python `aec.py` 的旗標不同） |
| D-15 | live observer lane 尚未鏡射 soft-realign warm IR-shift、RSA narrowband mu mask；bad-relock 還原（`aec_apply_delay_shift_soft_revert`／`_apply_delay_shift_soft_revert`）只還原 main 與 shadow 的 tap，兩邊都不還原 observer 的 tap；兩邊一致的缺口，不是移植落差 |
| D-16 | 單通道 MATCHED 參考環：C 取 estimator 可回報延遲上限＋hop（`delay_buffer_ms`、`max_delay_ms` 已刪除）；Python 取 max(delay_buffer_ms, max_delay_ms + 4096)。在 Python 預設值（環不小於上限）下兩邊的准入結果相同、輸出逐位元相同（證據：實作版對舊環 683 例逐位元相同，含真實錄音 154 例、真實 8 kHz 30 例、`delay_num_filters` < 5 共 165 例，另有 reach 邊界對照 1,138 例）；Python 把兩個欄位調小到環小於 estimator 可回報上限時，會拒收 C 會接受的延遲 |
| D-17 | 陣列 MATCHED 共用參考環：C 取 estimator 可回報延遲上限＋hop（與單通道同式；C 已無 `max_delay_ms`）；Python 孿生（Audio_ALG `pipelines/4ch/pipeline.py` 的 `_SharedReferenceDelayLine`）取 `max_delay_ms`（1024 ms）換算的 sample＋2 hop＋1。在 Python 預設值（環不小於上限）下兩邊的准入結果相同、輸出逐位元相同（證據：四組 baseline 串流 SIMD 1／0 除 META 外逐位元相同；Audio_ALG 4ch 測試與 `pyc_parity_pipelines.sh` 的 results.json 與改動前相同；reach 邊界對照 36 例相同）；Python 把 `max_delay_ms` 調小到環小於 estimator 可回報上限時，會拒收 C 會接受的延遲 |
| S-02 | live observer 注入守衛 `live_observer_inject_dt_energy_max`（Python 預設 0.5）C 沒有；開 observer 時實測會改輸出 |
| S-04 | HPF cutoff 範圍驗證（`0 < cutoff < 0.45·sr`）只有 C 檢查，Python 不檢查 |
| S-08 | CLI 預設運作點不同（Python `aec.py` 與 C `aec_wav`，兩邊都不給旗標時） |
| S-18 | （已關閉）`epc_mu_floor` 曾是 C-only 的死欄位；已隨 `AecConfig` 瘦身刪除 |
| S-19 | 端到端閘門 `parity_aec_e2e` 的 8 kHz 列超出容差（`TOL_E2E` 2e-2）；`make test-parity-e2e` 把它當預期紅燈 |
| S-22 | Python-only、預設 OFF 的研究旋鈕（C 沒有對應） |
| S-26 | 4ch 後波束 RES 的 AEC preset：DT 地板（Python 版只設 far-active 地板，DT 地板留在預設 -20） |
| S-34 | 設定範圍檢查不對稱：C 的 `validate` 嚴格 fail-closed，Python 幾乎不驗 |
| T-02 | 端到端閘門 `parity_aec_e2e` 的覆蓋：只有一個場景（雙講）、產生器沒有 `--fft` 選項（16k/512 沒有端到端對拍）、容差 2e-2 |
| T-03 | 沒有模組級 Python golden 的 C 模組：目前只有 `make test-detectors-parity-all` 是常設的模組級 Python↔C 閘門，其餘模組靠端到端閘門與 C 自洽測試間接覆蓋 |
| T-05 | 48k/1024 純線性（C `AEC_RESIDUAL_OFF`；Python 的後級旗標全關）瞬態偏差：HPF 關、雙講時約 -65.5 dB，未找到單一歸因 |
| T-06 | HPF 沒有 Python↔C 專屬 golden，只有端到端間接覆蓋 |
| T-17 | mono 端到端 C 對 Python 非逐位元，`error_spec` 若干 hop 後分岔（歸屬 lib/aec） |

## 附錄 B：對拍工具／測試標記

標記一律寫在反引號內，語法為 `<類別>:<名稱>`。測試會驗證每個標記指到的檔案或 Makefile target 仍然存在；它**不**驗證工具的結果，也不驗證該工具有沒有直接呼叫那一列的函式。`live:`、`manual:` 是模組級標記：harness 經由模組的公開函式驅動，同模組內部的 static 函式只被間接覆蓋；`ctest:` 與 `pytest:` 則附在確實涉及該函式（直接或經由其呼叫者）的列上。

| 標記 | 意義 |
|---|---|
| `live:<make target>` | `c_impl/Makefile` 內存在的 target，會即時由 Python 重產 golden 再比對 C。目前唯一的常設 Python↔C 模組級閘門是 `test-detectors-parity-all`（四個格點，取樣率／hop：8k/128、16k/128、16k/256、48k/512），涵蓋 `RenderActivityDetector`、`FilterConvergenceAnalyzer`、`DoubleTalkAnalyzer`（C 端沒有 `FilterPlateauDetector`）。 |
| `e2e:parity_aec_e2e` | `c_impl/test/parity_aec_e2e.c` 加 `python/diag/gen_aec_e2e_golden.py`：端到端對拍，由 `make test-parity-e2e` 建置並執行（16／48 kHz 通過，8 kHz 列為預期紅燈，見 S-19）；覆蓋限制見 T-02。 |
| `manual:<檔名>` | `c_impl/test/` 下沒有接任何 Makefile target 的 harness；目前沒有任何列使用。 |
| `ctest:<檔名>` | `c_impl/test/` 下的 C 測試。多數是 C 自洽的回歸或結構測試，不是 Python 數值對拍；`test_delay_*`、`test_external_realign`、`test_runtime_preset`、`test_res_nearend_gate`、`test_reset_parity` 與同名（或對應）的 pytest 用同一組場景。 |
| `pytest:<檔名>` | `python/tests/` 下的 Python 測試。`test_signal_grid.py` 會讀 `c_impl/src/aec.c` 的文字比對格點表（`AEC_GRID_TABLE`），`test_preset_arg.py` 讀 `aec.h` 內 `AEC_LEVEL_*` 那四行的整數值；其餘只測 Python 行為。 |
