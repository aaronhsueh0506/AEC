# AEC · 回音消除

輸入一支麥克風和同取樣率的播放參考，逐 hop 輸出回音消除後的音訊。C 函式庫提供 PBFDKF 線性消除、延遲估計、殘餘回音抑制（RES）及舒適噪音；`AecConfig.n_mics = 4` 時一次 init 建出四麥克風陣列實例（四條線性 lane、共用 delay 估計器與融合，beamformer 與合成由呼叫端提供）。另有單聲道的 Python 參考實作。

## 先處理一段 WAV

將 `audiolib/` 和 `AEC/` 放在同一層。在 AEC 根目錄執行下列命令。

```sh
make -C c_impl
bin="$(make -s -C c_impl print-bin-dir)"
"$bin/aec_wav" mic.wav ref.wav out.wav --level 1
```

`--level 1`（balanced）是預設強度，0 是 mild、2 是 aggressive、3 是 no_floor。兩個輸入使用 mono WAV，採樣格點支援 8 kHz（256/128）、16 kHz（256/128 或 512/256）、48 kHz（1024/512），括號內是 FFT 長度／hop 樣本數。產品版本為 5.0.0rc1，48 kHz 仍需原生錄音驗收音質。

板端可用 `aec_get_mem_size()` 配置記憶池，`aec_init()` 初始化（桌面可用 `aec_create()` 一次配置、`aec_destroy()` 釋放；`Aec` 是不透明型別），每 hop 呼叫 `aec_process()`（回傳 `AecStatus`）。要自己套增益或串接模型時，把 `AecConfig.residual` 設成需要的深度（`OFF`／`STATE`／`ESTIMATE`／`FULL`），改呼叫 `aec_process_linear()` 與 `aec_process_residual()`，由 `aec_get_residual()` 讀線性殘差、渲染基底與 `res_gain`，再用 `aec_apply_gain()`／`aec_apply_cng()` 渲染；舊版 context API 與已移除名稱的遷移見 [`docs/sdk_migration_output_guard.md`](docs/sdk_migration_output_guard.md)。範例與 reset 順序都從同一入口進入。

手冊入口：[C 使用手冊](docs/c_user_manual_zh_TW.md)（header、格點、建置、最小範例與資料契約）· [靜態記憶體](c_impl/STATIC_MEMORY.md) · [開發與驗證](docs/maintenance/development.md)。

**[開啟 AEC 文件](docs/html/index.html)** · 可在本機離線閱讀。
