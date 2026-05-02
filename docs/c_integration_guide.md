# C AEC Integration Guide

**Audience**: developers integrating the v2 C AEC library into a host C/C++
project (Audio_ALG, Novatek SDK, etc.).

**Library version**: aligned bit-exact with Python `aec.py` v3.8.1
(see [CHANGELOG.md v3.8.2 entry](CHANGELOG.md)). Active path is
`c_impl/include_v2/` + `c_impl/src_v2/`. Legacy v2.5-aligned `c_impl/include/`
+ `c_impl/src/` is frozen.

> CLI usage → [c_impl/USER_GUIDE.md](../c_impl/USER_GUIDE.md)
> Per-module parity gate evidence → [c_impl/README.md](../c_impl/README.md)

---

## API surface

```c
#include "aec_v2.h"
```

`AecV2Config` mirrors Python `AecConfig` field-by-field. `AecV2` is the runtime
context (open struct so parity test code can read internal state — keep
client code reading only via accessors).

### Standard mode — full pipeline in one call

```c
AecV2Config cfg;
aec_v2_config_from_preset(&cfg, AECV2_PRESET_BALANCED, /*sr=*/16000);
cfg.enable_cng       = 0;   // default — match Python CLI
cfg.enable_delay_est = 1;   // default

AecV2 aec;
aec_v2_create(&aec, &cfg);
int hop = aec_v2_hop_size(&aec);   // 160 @16k, 480 @48k

float mic[hop], ref[hop], out[hop];
while (read_block(mic, ref, hop)) {
    aec_v2_process(&aec, mic, ref, out);   // HPF + sat + delay + filter + shadow + RES + limiter
    write_block(out, hop);
}

aec_v2_destroy(&aec);
```

`aec_v2_process` runs the entire pipeline (HPF → saturation → delay-est →
PBFDKF main + shadow → EPC + ShadowCopy → ResFilter → limiter). Disable
sub-modules via `cfg.enable_*` flags before `aec_v2_create`.

---

## Inserting noise reduction between linear AEC and RES

**Status (2026-05-02)**: not yet exposed in v2 public API.

Python `aec.py` reference exposes `AecResContext` (legacy `c_impl/include/aec_types.h`)
that decouples linear AEC from ResFilter so a host pipeline can insert NR/post-
filter in between. The same surface needs to be added on top of v2 — design
exists, port pending. Until then:

- Quick option: run AEC with `cfg.enable_residual_filter = 0` to get
  linear-only output. Pipe into NR. RES is currently not callable separately
  in v2 (would need a `res_filter_v2_create_standalone` + context struct).
- Reference target shape: the legacy `aec_process_ex(...)` + `AecResContext`
  flow in `c_impl/src/aec.c` (frozen). New v2 wrapper should mirror this.

`Audio_ALG/pipelines/aec_nr_pipeline.c` currently uses the legacy v2.5 API
and remains the working example. Migration to v2 is queued.

---

## Module ownership (v2)

| Header | Source | Owns |
|---|---|---|
| `aec_v2.h` | `aec_v2.c` | top-level orchestration, control-flow state, `AecV2Config` |
| `hpf_v2.h` | `hpf_v2.c` | 80 Hz Butterworth biquad |
| `saturation_v2.h` | `saturation_v2.c` | clip detection + ref-side soft-clip |
| `delay_est_v2.h` | `delay_est_v2.c` | GCC-PHAT delay estimation, ring buffer |
| `pbfdkf_v2.h` | `pbfdkf_v2.c` | PBFDAF base + PBFDKF (G1 KX-blended P-update), Q schedule, partition rotation |
| `erle_v2.h` | `erle_v2.c` | FilterErleEstimator + FullbandErleEstimator + `compute_erle_confidence` |
| `detectors_v2.h` | `detectors_v2.c` | RenderActivity + FilterConvergence + DoubleTalkAnalyzer |
| `epc_shadow_v2.h` | `epc_shadow_v2.c` | EchoPathChangeDetector + ShadowCopyController |
| `residual_echo_v2.h` | `residual_echo_v2.c` | ResidualEchoEstimator (legacy mode — used by all 4 presets) |
| `res_filter_v2.h` | `res_filter_v2.c` | ResFilter (ENR + reverb tail + min-stat noise + CNG, OLA + sqrt-Hann) |
| `dtd_v2.h` | `dtd_v2.c` | DTD (geigel / divergence / coherence — opt-in; BALANCED preset has DTD off) |
| `aec_debug.h` | `aec_debug.c` | timestamped log infrastructure (build-gated) |
| — | `fft_fp64.c` | fp64 radix-2 FFT (replaces kiss_fft for parity with numpy pocketfft) |

---

## Things you must not do

These cause correctness failures (not just style issues):

### 1. Do not bypass linear AEC and feed mic directly to RES

ResFilter consumes the linear filter's `echo_spec` as its primary echo signal.
Without a converged linear estimate, RES has nothing to suppress.

### 2. Do not feed non-`hop_size` blocks

`aec_v2_process` requires exactly `hop_size` samples per call (160 @16k,
480 @48k, etc.). Use a ring buffer upstream for variable-size input.

### 3. Do not change `filter_length_ms` mid-stream

Partition count is fixed at `aec_v2_create`. Mid-stream changes invalidate
`P`/`Q`/`W` array bounds.

### 4. Do not omit `aec_v2_reset` between independent streams

DT / EPC / convergence state accumulates across calls. For batch processing
of multiple files, call `aec_v2_reset` before each new file. Otherwise a
tonal far-end from file N can trigger EPC false-positives at the start of N+1.

### 5. Do not enable `enable_cng` and run RES output through another CNG layer

CNG fills below-`g_min` bins with shaped noise. Stacking another CNG layer
double-shapes the noise floor → tonal artifacts.

### 6. Do not assume `enable_delay_est = 1` is free in offline mode

Delay estimator runs every frame and triggers filter reset on first
acquisition (causes ~300 ms learning loss). For offline batch with
pre-aligned files, set `cfg.enable_delay_est = 0`.

### 7. Do not run multiple `AecV2` instances sharing FFT state across threads

Each instance allocates its own FFT scratch. Concurrent use of a single
instance from multiple threads is undefined.

### 8. Do not build without `-ffp-contract=off`

FMA contraction at `-O2` breaks bit-exact parity on HPF / saturation / detector
states. The flag is required even in release builds if you want parity with
Python output.

---

## Memory & threading

- **Per-instance heap** (16 kHz / 52 ms / 257 freqs): ~280 KB including main +
  shadow filter + RES + delay-est ring buffer.
- **Threading**: each `AecV2` instance is single-threaded. For multi-stream
  use multiple instances; no shared state.
- **Real-time**: after `aec_v2_create`, `aec_v2_process` does not allocate or
  do I/O. `aec_v2_reset` likewise.
- **Cache footprint**: PBFDKF weights dominate. For 52 ms / 16 kHz / 6
  partitions × 257 bins × 8 bytes (complex64) × 2 (main + shadow) ≈ 25 KB
  per filter, ~50 KB total filter weights.

---

## Build

```bash
cd c_impl/
make aec_wav_v2          # CLI binary → bin/aec_wav_v2

# Manual compile (single-binary embed):
cc -O2 -ffp-contract=off \
   -I include_v2 -I include -I example \
   src_v2/*.c example/aec_wav_v2.c \
   -o bin/aec_wav_v2 -lm
```

Required flags:

- `-O2` (or `-O3`): expected optimization level
- `-ffp-contract=off`: disable FMA contraction (mandatory for parity)
- `-I include_v2 -I include`: legacy `include/` still provides `fft_wrapper.h`
  + `wav_io.h`. v2 sources include `aec_v2.h` etc. from `include_v2/`.

---

## Validation checklist before deployment

1. **Compiler clean**: `-Wall -Werror -O2 -ffp-contract=off` produces zero warnings
2. **Per-module parity**: `make parity_<module>` for HPF / Saturation / Delay /
   Erle / Detectors / PBFDKF — all gates green
3. **End-to-end SNR**: `aec_wav_v2` output vs Python `aec.py` (`--mode pbfdkf
   --enable-res`, same preset) → SNR ≥ 60 dB on representative cases
4. **Bench parity**: 800-case AECMOS — per-bucket scores within ±0.005 of
   Python v3.8.1 baseline (current target — script integration TBD)
5. **Memory check**: `valgrind --leak-check=full ./bin/aec_wav_v2 ...`
6. **Real-time check**: per-frame `aec_v2_process` time < hop duration (10 ms
   @16 kHz)
7. **Reset hygiene**: `create → process → reset → process` reproduces fresh-
   create + process second-invocation output

---

## Reference

- Python source — [python/aec.py](../python/aec.py) (v3.8.1 reference)
- Algorithm methods — [aec_methods.md](aec_methods.md)
- Per-version history — [CHANGELOG.md](CHANGELOG.md)
- WebRTC AEC3 design notes — [aec3_reference.md](aec3_reference.md)
