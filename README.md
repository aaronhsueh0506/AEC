# AEC · 單聲道回音消除

輸入一支麥克風和同取樣率的播放參考，逐 hop 輸出回音消除後的音訊。C 函式庫提供 PBFDKF 線性消除、延遲估計、殘餘回音抑制（RES）及舒適噪音。另有 Python 參考實作。

## 先處理一段 WAV

將 `audiolib/` 和 `AEC/` 放在同一層。在 AEC 根目錄執行下列命令。

```sh
make -C c_impl
bin="$(make -s -C c_impl print-bin-dir)"
"$bin/aec_wav" mic.wav ref.wav out.wav --level balanced
```

`balanced` 是預設強度。兩個輸入使用 mono WAV，採樣格點支援 8 kHz（256/128）、16 kHz（256/128 或 512/256）、48 kHz（1024/512），括號內是 FFT 長度／hop 樣本數。產品版本為 5.0.0rc1，48 kHz 仍需原生錄音驗收音質。

板端可用 `aec_get_mem_size()` 配置記憶池，`aec_init()` 初始化，每 hop 呼叫 `aec_process()`。串接模型時請確認需要的是線性殘差、下游渲染基底或獨立 RES 增益。範例與 reset 順序都從同一入口進入。

**[開啟 AEC 文件](docs/html/index.html)** · 可在本機離線閱讀。
