# AEC C — User Guide

C-language usage guide for the v2 AEC port (active path), aligned bit-exact with
Python `aec.py` v3.8.1.

> Architecture / parity status → [README.md](README.md)
> Algorithm reference → [../docs/aec_methods.md](../docs/aec_methods.md)
> Integration into a host C/C++ pipeline → [../docs/c_integration_guide.md](../docs/c_integration_guide.md)

---

## 1. Quick start

### 1.1 Build

```bash
cd c_impl
make aec_wav_v2          # CLI binary → bin/aec_wav_v2
```

Required compile flags (already in Makefile target):

```
-O2 -ffp-contract=off -I include_v2 -I include
```

`-ffp-contract=off` is mandatory — FMA contraction breaks bit-exact parity.

### 1.2 CLI

```bash
# BALANCED preset (default)
./bin/aec_wav_v2 <mic.wav> <ref.wav> <out.wav>

# Other presets
./bin/aec_wav_v2 mic.wav ref.wav out.wav --preset mild
./bin/aec_wav_v2 mic.wav ref.wav out.wav --preset aggressive
./bin/aec_wav_v2 mic.wav ref.wav out.wav --preset maximum

# Toggles (defaults match Python `aec.py` CLI)
./bin/aec_wav_v2 mic.wav ref.wav out.wav --cng              # enable CNG (default off)
./bin/aec_wav_v2 mic.wav ref.wav out.wav --no-res           # skip residual filter
./bin/aec_wav_v2 mic.wav ref.wav out.wav --no-shadow
./bin/aec_wav_v2 mic.wav ref.wav out.wav --no-delay-est
./bin/aec_wav_v2 mic.wav ref.wav out.wav --no-hpf

# Filter length / reverb override
./bin/aec_wav_v2 mic.wav ref.wav out.wav --filter-length-ms 64
./bin/aec_wav_v2 mic.wav ref.wav out.wav --reverb-decay 0.85 --reverb-gain 1.6

# Debug log (per-frame stderr)
./bin/aec_wav_v2 mic.wav ref.wav out.wav --debug-level 2
./bin/aec_wav_v2 mic.wav ref.wav out.wav --debug-level 2 --debug-log /tmp/aec.log
```

Unknown preset names exit with non-zero status (no silent fallback).

### 1.3 API

```c
#include "aec_v2.h"

AecV2Config cfg;
aec_v2_config_from_preset(&cfg, AECV2_PRESET_BALANCED, /*sr=*/16000);

// Optional: tweak from defaults (preset-derived values already loaded)
cfg.enable_cng       = 0;        // default — match Python CLI default
cfg.enable_delay_est = 1;        // default

AecV2 aec;
aec_v2_create(&aec, &cfg);

int hop = aec_v2_hop_size(&aec); // 160 @16k, 480 @48k
float mic[hop], ref[hop], out[hop];

while (read_block(mic, ref, hop)) {
    aec_v2_process(&aec, mic, ref, out);
    write_block(out, hop);
}

aec_v2_destroy(&aec);
```

The C library expects `hop` samples per call. HPF, saturation detect, delay
estimation, shadow filter, and ResFilter all run inside `aec_v2_process` —
no manual pre-processing needed.

---

## 2. Operating conditions

### 2.1 Input format

| Field | Value |
|---|---|
| Sample rate | 8 / 16 / 48 kHz |
| Bit depth | 16-bit PCM or 32-bit float |
| Channels | mono |
| Frame / hop | 20 ms / 10 ms (auto-derived from SR) |
| Filter length | 52 ms default |

### 2.2 Use case

- Full-duplex hands-free (phone, smart speaker, conferencing, automotive)
- Single mic + single reference signal (mono)
- Echo-path length ≤ filter length (default 52 ms; configurable)

### 2.3 Prerequisites

| | |
|---|---|
| Reference correct      | `ref` must be the playback loopback of what reaches the speaker |
| Delay aligned          | `mic - ref` delay must fit within `filter_length`. Online delay est. (`--enable-delay-est`, default on) handles this if delay < `max_delay_ms` (default 250 ms) |
| Sync mic and ref       | same SR, time-aligned start. Drift will tank ERLE |
| Sufficient far energy  | silent / near-silent ref will not drive convergence |

---

## 3. Limits

### 3.1 Algorithmic

| | |
|---|---|
| Sample rate          | 8 / 16 / 48 kHz only — anything else needs external resample |
| Channels             | mono only — no mic array / stereo ref |
| Nonlinear echo       | linear filter + RES; no dedicated nonlinear model |
| Delay direction      | positive only (mic lags ref). Negative-delay scenarios must be aligned upstream |
| Echo-path length     | configurable via `filter_length_ms` — extending costs memory + compute |

### 3.2 Scenario notes

| | |
|---|---|
| Cold start                   | first 0.5–2 s before filter converges; RES uses conservative render-based estimate |
| Echo-path change             | EPC detector triggers fast re-convergence (~200 ms hangover) |
| High coupling (small device) | mic ≈ speaker proximity; ~1–2 dB residual leak possible during DT |
| Stationary far + weak NE     | linear AEC may absorb very low-energy NE syllables (mitigated by stationary-DT hangover but not fully recoverable) |
| DT-from-frame-0              | NE present from sample 0 prevents convergence — see [../docs/aec_methods.md 附錄 E](../docs/aec_methods.md#附錄-e-dt-from-frame-0-限制) |

### 3.3 Resources (fl=52 ms @16 kHz)

| | |
|---|---|
| Memory (main + shadow filter)     | ~200 KB |
| Memory (incl. RES + delay est.)   | ~280 KB |
| Compute / frame                   | 4 × 512-FFT + Kalman update (257 bins × 6 partitions) |
| FFT                               | fp64 radix-2 (no external dep, ~6 KB code) |

---

## 4. Common adjustments

### 4.1 Residual echo too high

1. **Verify ref**: with `--no-res`, output should be a clean linear-AEC residual. If echo still dominates → ref signal is wrong, mic/ref delay > filter length, or SRs differ.
2. **Delay alignment**: `--enable-delay-est` (default on); if still leaking, measure actual mic–ref delay and bump `--filter-length-ms`.
3. **Stronger RES**: `--preset aggressive` or `maximum` (cost: more NE compression).
4. **Longer filter**: `--filter-length-ms 100` for big rooms / long reverb.

### 4.2 NE clipped during double-talk

- **Lower preset**: `--preset balanced` or `mild`.
- Don't tweak individual RES knobs — preset values are co-tuned.

### 4.3 Slow startup / first-second echo

Filter convergence needs ≥ 0.5 s of meaningful far-end energy. Normal adaptive
behavior. Consider muting output during application warm-up (e.g. play a
"connecting..." cue).

### 4.4 Echo spikes when device moves

Echo path changes → EPC fires → ~200 ms re-convergence with brief leak. Usually
recovers automatically. For frequent movement, increase filter length to
broaden model capacity.

### 4.5 Listen to the linear-AEC residual

```bash
./bin/aec_wav_v2 mic.wav ref.wav linear_only.wav --no-res
```

Lets you separate filter-side issues (linear AEC underperforming) from RES-side
issues (filter OK, post-filter too weak / strong).

### 4.6 Inspect per-frame state

```bash
./bin/aec_wav_v2 mic.wav ref.wav out.wav --debug-level 2 --debug-log /tmp/aec.log
grep "PBFDKF" /tmp/aec.log | head
```

Format: `[AEC][t=  1.234s][f=  154][PBFDKF] mu_mean=... P_mean=...`. Each line
is grep-friendly key=value pairs. Release builds (`-DNDEBUG`) strip log
strings entirely.

---

## 5. Preset selection

| Preset | FS echo suppression | DT NE preservation | Use |
|---|---|---|---|
| **mild**       | medium     | best       | high-fidelity calls, music scenarios |
| **balanced**   | high       | medium     | general calls (recommended default) |
| **aggressive** | very high  | medium-low | automotive, noisy environments |
| **maximum**    | extreme    | weak       | hi-coupling speakerphones, IoT |

FS = far-end singletalk; DT = double-talk.

---

## 6. Output format

`aec_wav_v2` writes 32-bit-float WAV by default (parity-friendly). Set
`AEC_FP32_WAV=0` to force 16-bit PCM:

```bash
AEC_FP32_WAV=0 ./bin/aec_wav_v2 mic.wav ref.wav out.wav
```

Internal pipeline always runs at fp32 sample I/O regardless of file format.

---

## 7. Reporting issues

When reporting, please include:

- input wav files (or describe acoustic scenario)
- exact CLI used
- output of `--debug-level 2 --debug-log <path>` for ≥ 1 s around the issue
- expected vs observed behavior

For internal algorithm details, see [../docs/aec_methods.md](../docs/aec_methods.md).
