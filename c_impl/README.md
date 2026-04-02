# AEC C Implementation — PBFDKF + Shadow + Multi-ERLE + WOLA RES

C 語言 AEC 實作，對齊 Python v2.0.0（correlation 0.971）。

## 特點

- **PBFDKF**（Partitioned Block Frequency-Domain Kalman Filter）
  - Correct Kalman math（global denominator, unscaled K for P update）
  - P_MAX=0.5, R_scale=0.1, alpha_r=0.95
  - dt_indicator ERLE correction（3-frame EMA smoothed inst_erle）
- **Shadow Filter** — 雙濾波器自動修正 + bidirectional copy
- **Multi-ERLE** — FilterErleEstimator + FullbandErleEstimator（打破循環依賴）
- **WOLA RES v2** — 完整重寫對齊 Python
  - ENR: dt × nearend_est（非線性 echo 保護）
  - ne_confidence = dt_indicator（直接控制）
  - Divergence override、cross-frequency smoothing
- **CNG** — 凍結式底噪追蹤（×0.3 attenuation）
- **GCC-PHAT Delay Estimation** — inline, 250ms max
- **四級 Preset** — MILD / BALANCED / AGGRESSIVE / MAXIMUM
- **Output Limiter** — 平滑增益限制

> HPF（80Hz 高通濾波器）已移至應用層。AEC module 不再內建 HPF。

## 編譯

```bash
make          # CLI → bin/aec_wav
make lib      # 靜態函式庫 → bin/libaec.a
make clean    # 清除
```

## CLI 使用

```bash
# 基本用法（balanced preset，含 HPF + delay estimation）
./bin/aec_wav mic.wav ref.wav output.wav

# 指定 preset
./bin/aec_wav mic.wav ref.wav output.wav --preset aggressive

# 關閉 RES（純線性 AEC output）
./bin/aec_wav mic.wav ref.wav output.wav --no-res

# 關閉 HPF
./bin/aec_wav mic.wav ref.wav output.wav --no-hpf

# 啟用 delay estimation
./bin/aec_wav mic.wav ref.wav output.wav --enable-delay-est

# 自訂 filter length
./bin/aec_wav mic.wav ref.wav output.wav --filter 1024
```

## API 使用

```c
#include "aec.h"
#include "aec_types.h"
#include "hpf.h"

// 1. 建立 config（balanced preset）
AecConfig cfg = aec_config_from_preset(AEC_PRESET_BALANCED, 16000);

// 2. 建立 AEC（HPF 已移至應用層，AEC 不做 HPF）
Aec* aec = aec_create(&cfg);

// 3. 建立 HPF（應用層負責）
Hpf* hp_mic = hpf_create(80.0f, 16000);
Hpf* hp_ref = hpf_create(80.0f, 16000);

// 4. 逐幀處理
float mic[160], ref[160], out[160];
while (read_audio(mic, ref, 160)) {
    hpf_process(hp_mic, mic, 160);  // HPF 在 AEC 之前
    hpf_process(hp_ref, ref, 160);
    aec_process(aec, mic, ref, out);
    write_audio(out, 160);
}

// 5. 清除
aec_destroy(aec);
hpf_destroy(hp_mic);
hpf_destroy(hp_ref);
```

## 資源消耗（fl=512 @16kHz）

| 項目 | 值 |
|------|-----|
| 記憶體（main + shadow） | ~200 KB |
| 記憶體（含 RES + delay est） | ~280 KB |
| FFT | kiss_fft（single precision） |
