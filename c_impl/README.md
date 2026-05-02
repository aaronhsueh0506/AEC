# AEC C Implementation (v2 — bit-exact aligned with Python v3.8.1)

C port of the Python `aec.py` v3.8.1 reference. Layout: `include_v2/` + `src_v2/`,
top-level orchestration in `aec_v2.{h,c}`, CLI binary `bin/aec_wav_v2`.

The legacy v2.5-aligned implementation in `include/`, `src/`, and the older
`bin/aec_wav` / `bin/aec_dump` binaries is kept frozen for reference. New code
should use the v2 path.

> User guide → [USER_GUIDE.md](USER_GUIDE.md) (CLI / API / build / limits)
> Integration into a host C/C++ pipeline → [../docs/c_integration_guide.md](../docs/c_integration_guide.md)
> Algorithm reference → [../docs/aec_methods.md](../docs/aec_methods.md)

---

## Parity status

End-to-end SNR vs Python `aec.py` v3.8.1 (`--mode pbfdkf --enable-res`,
BALANCED preset, 16 kHz):

| Scenario              | SNR    | max abs diff |
|-----------------------|--------|--------------|
| farend_singletalk     | 64.15 dB | 3.05e-5 |
| doubletalk            | 70.37 dB | 3.05e-5 |
| nearend_singletalk    | 67.38 dB | 3.05e-5 |

`max_abs_diff = 3.05e-5` is the 16-bit PCM quantization step (1/32768);
internal float32 buffers match within fp32 ULP cumulative drift over thousands
of frames of EMA chains.

### Per-module parity gates (passed)

| Module                        | Gate                                       |
|-------------------------------|--------------------------------------------|
| HighPassFilter                | bit-exact 0 ULP, 4237 frames × 2 sides     |
| SaturationDetector            | bit-exact 0 ULP, 4237 frames × 2 sides     |
| DelayEstimator (GCC-PHAT)     | delay & n_updates exact, PAR rtol < 1e-3   |
| FilterErle / FullbandErle     | fp64 bit-exact, fp32 store-back ULP        |
| RenderActivity / Convergence / DoubleTalkAnalyzer | bit-exact across boolean + counter + fp64 EMAs |
| PBFDKF (G1 KX-blended)        | output / echo / P / R bit-exact pre-reset (30 frames) |

Per-frame state diff (every Python `aec._diag` field): ALL of `erl_estimate`,
`dt_indicator`, `dt_from_energy`, `far_activity`, `main_err_smooth`,
`shadow_err_smooth`, `simple_mu_ratio`, `converged`, `once_converged`,
`using_render_based`, `epc_active` match bit-exact across the entire 4237-frame
test wav.

---

## Module layout

```
c_impl/
├── include_v2/                   v2 API headers (active)
│   ├── aec_v2.h                    top-level orchestration
│   ├── aec_debug.h                 timestamped logger (build-gated)
│   ├── hpf_v2.h                    80 Hz Butterworth IIR HPF
│   ├── saturation_v2.h             ref-side soft-clip + level EMA
│   ├── delay_est_v2.h              GCC-PHAT online delay
│   ├── erle_v2.h                   FilterErle + FullbandErle + confidence
│   ├── detectors_v2.h              RenderActivity + Convergence + DoubleTalk
│   ├── pbfdkf_v2.h                 PBFDAF base + PBFDKF (G1 KX blend)
│   ├── residual_echo_v2.h          ResidualEchoEstimator (legacy mode)
│   ├── res_filter_v2.h             ResFilter (ENR + reverb + min-stat noise + CNG)
│   ├── epc_shadow_v2.h             EchoPathChange + ShadowCopyController
│   └── dtd_v2.h                    DTD (geigel / divergence / coherence — opt-in)
├── src_v2/                       v2 implementations
│   ├── fft_fp64.c                  fp64 radix-2 FFT (matches numpy pocketfft)
│   └── (one .c per header)
├── example/
│   ├── aec_wav_v2.c                active CLI
│   ├── main.c, aec_dump.c          legacy v2.5 CLI binaries (frozen)
│   └── wav_io.h
├── test/parity/                  per-module parity tests + harness sources
├── lib/kiss_fft/                 legacy fp32 FFT (used only by frozen binaries)
├── include/, src/                legacy v2.5 implementation (frozen)
└── Makefile
```

---

## Build

```bash
make aec_wav_v2    # → bin/aec_wav_v2 (active CLI)
make               # legacy v2.5 binary (bin/aec_wav)
make clean
```

CLI compile flags:

```
-O2 -ffp-contract=off -I c_impl/include_v2 -I c_impl/include
```

`-ffp-contract=off` is **required** — FMA contraction breaks bit-exact parity
on HPF / Saturation / detector states.

---

## CLI

```bash
./bin/aec_wav_v2 mic.wav ref.wav out.wav                    # BALANCED
./bin/aec_wav_v2 mic.wav ref.wav out.wav --preset aggressive
./bin/aec_wav_v2 mic.wav ref.wav out.wav --cng              # CNG on (default off)
./bin/aec_wav_v2 mic.wav ref.wav out.wav --no-res           # skip residual filter
./bin/aec_wav_v2 mic.wav ref.wav out.wav --no-shadow
./bin/aec_wav_v2 mic.wav ref.wav out.wav --no-delay-est
./bin/aec_wav_v2 mic.wav ref.wav out.wav --filter-length-ms 64
./bin/aec_wav_v2 mic.wav ref.wav out.wav --debug-level 2    # per-frame log to stderr
```

`--preset` accepts `mild | balanced | aggressive | maximum`. Unknown preset
exits with non-zero status (no silent fallback). Output wav defaults to fp32
PCM for parity-friendly comparison; set env `AEC_FP32_WAV=0` for 16-bit PCM.

---

## API (minimum)

```c
#include "aec_v2.h"

AecV2Config cfg;
aec_v2_config_from_preset(&cfg, AECV2_PRESET_BALANCED, 16000);
cfg.enable_cng = 0;             // default — match Python CLI default
cfg.enable_delay_est = 1;       // default

AecV2 aec;
aec_v2_create(&aec, &cfg);

int hop = aec_v2_hop_size(&aec);   // 160 @16k, 480 @48k
float mic[hop], ref[hop], out[hop];

while (read_block(mic, ref, hop)) {
    aec_v2_process(&aec, mic, ref, out);
    write_block(out, hop);
}

aec_v2_destroy(&aec);
```

The C library expects `hop` samples per call; HPF + saturation detect + delay
estimation happen internally. Configurable behavior is in `AecV2Config`
(mirrors Python `AecConfig` field-by-field).

---

## Resources (fl=52 ms @16 kHz)

| | |
|---|---|
| Memory (main + shadow filter)     | ~200 KB |
| Memory (incl. RES + delay est.)   | ~280 KB |
| Compute / frame                   | 4 × 512-FFT + Kalman update (257 bins × 6 partitions) |
| FFT                               | fp64 radix-2 (no external dep) |

---

## Build-time switches

| Flag | Purpose |
|---|---|
| `-DAEC_DEBUG`           | enable runtime debug log call sites |
| `-DNDEBUG`              | strip debug log strings & assertions (release) |
| `-DAEC_PARITY_HARNESS`  | enable internal-state read/write entry points used by `test/parity/` |

The debug logger writes one line per frame (`[AEC][t=  1.234s][f=  154][PBFDKF] ...`)
to stderr (or a file via `--debug-log <path>`). In release builds the macros
collapse to no-ops with no string literals retained in the binary.
