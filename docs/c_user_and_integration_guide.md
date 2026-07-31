# AEC C — User & Integration Guide

C implementation of the AEC algorithm. Single mic + single reference,
PBFDKF main filter + shadow filter + post-filter residual echo
suppression. Targets full-duplex hands-free use cases (phone, smart
speaker, conferencing, automotive).

> Algorithm reference → [aec_methods.md](aec_methods.md)
> Changelog → [../CHANGELOG.md](../CHANGELOG.md)

---

## 1. Build

```bash
cd c_impl/
make            # → bin/<backend>-<config-hash>/aec_wav (CLI binary)
make lib        # → bin/<backend>-<config-hash>/libaec.a (static library)
make debug      # build with -g -DAEC_DEBUG
make clean
```

Artifacts land in a config-hashed `bin/<backend>-<config-hash>/` directory
(run `make print-bin-dir` with the same flags to get the exact path, or
`make publish` for a stable `dist/<backend>/current/` handoff path); the
example invocations below elide this prefix for readability.

Required compile flags (already in Makefile):

```
-O2 -ffp-contract=off -I include -I example
```

`-ffp-contract=off` is **mandatory**. FMA contraction at `-O2` produces
non-deterministic state in HPF / saturation / detector accumulators.

---

## 2. CLI usage

```bash
# Default preset (balanced)
./bin/aec_wav <mic.wav> <ref.wav> <out.wav>

# Other presets
./bin/aec_wav mic.wav ref.wav out.wav --preset mild
./bin/aec_wav mic.wav ref.wav out.wav --preset aggressive

# Toggles
./bin/aec_wav mic.wav ref.wav out.wav --cng              # comfort noise (default off)
./bin/aec_wav mic.wav ref.wav out.wav --no-res           # skip residual filter
./bin/aec_wav mic.wav ref.wav out.wav --no-shadow
./bin/aec_wav mic.wav ref.wav out.wav --no-delay-est
./bin/aec_wav mic.wav ref.wav out.wav --no-hpf

# Preset + toggle combined, and CSV debug trace
./bin/aec_wav mic.wav ref.wav out.wav --preset aggressive --no-hpf
./bin/aec_wav mic.wav ref.wav out.wav --debug-trace trace.csv

# Debug log (per-frame stderr or file)
./bin/aec_wav mic.wav ref.wav out.wav --debug-level 2
./bin/aec_wav mic.wav ref.wav out.wav --debug-level 2 --debug-log /tmp/aec.log
```

Unknown preset names or options exit with code 2 (no silent fallback).
Output WAV is fp32 PCM by default; `AEC_OUT_FLOAT=0` forces 16-bit PCM.

### Preset selection

Three presets — they differ **only** in `min_gain_floor_far_active_db`, the
far-active min-gain floor (the single Pareto knob trading echo suppression
against near-end preservation):

| Preset | floor | FS echo suppression | DT NE preservation | Use case |
|---|---|---|---|---|
| **mild** | −20 dB | lower (FS dips below 3.5 by design) | best — near-priority | echo 安靜場景、demo / 試聽、想保留近端原始質感 |
| **balanced** ★ | −28 dB | high (all four ship bars met) | medium | general calls (**recommended default**) |
| **aggressive** | −38 dB | very high — echo-priority | medium-low (deg still >2.0) | automotive, noisy / hi-coupling speakerphones |

---

## 3. C API

### 3.1 Standard heap-based init

```c
#include "aec.h"

AecConfig cfg;
aec_config_from_preset(&cfg, AEC_PRESET_BALANCED, /*sr=*/16000);
cfg.enable_cng       = 0;        // default off
cfg.enable_delay_est = 1;        // default on

Aec aec;
aec_create(&aec, &cfg);

int hop = aec_hop_size(&aec);    // 256 @16k default, 512 @48k
float mic[hop], ref[hop], out[hop];

while (read_block(mic, ref, hop)) {
    aec_process(&aec, mic, ref, out);
    write_block(out, hop);
}

aec_destroy(&aec);
```

`aec_process` runs the entire pipeline (HPF → saturation → delay
estimation → PBFDKF main + shadow → echo-path-change handling → AEC3
post-filter `aec3_post` → limiter). The post-stage is the AEC3-aligned
chain (AecState + ResidualEchoEstimator + SuppressionGain + CNG, OLA);
the legacy 9-stage `ResFilter` was retired in v3.21. Disable sub-modules
via `cfg.enable_*` flags before `aec_create`.

### 3.2 Heap-free init (static memory pool)

For embedded targets without a heap, supply a single pre-allocated pool
instead of calling `aec_create`:

```c
size_t pool_bytes = aec_get_mem_size(&cfg);   /* exact pool size for cfg */
void*  pool = NULL;
if (posix_memalign(&pool, 16, pool_bytes) != 0) { /* allocation failed */ }

Aec* aec = aec_init(pool, pool_bytes, &cfg);  /* returns NULL, not an int status */
if (!aec) { /* pool too small, or NULL/misaligned inputs */ }

/* aec_process(aec, ...) / aec_reset(aec) / aec_destroy(aec) as usual —
 * aec is now a pointer, so no more `&aec` on call sites. */
aec_destroy(aec);
free(pool);   /* caller frees pool itself, after aec_destroy() (see below) */
```

`aec_init` takes the pool directly (no out-param `Aec*` argument) and
returns `Aec*` — `NULL` on failure (pool too small, NULL pointer, or a
misaligned base). Both paths produce **bit-identical** output on a given
backend (verified across all 3 presets and FS / DT / NE scenarios; NE10 vs
KISS output is not bit-identical to *each other* — a pre-existing, expected
difference between the two FFT implementations). At BALANCED / 16 kHz /
52 ms filter / shadow on / RES on / delay-est on the pool is **538,320 B
(525.7 KB)** on the KISS backend (host/reference, `make`, default) or
**534,192 B (521.7 KB)** on the NE10 backend (embedded, `make
BACKEND=ne10`). The `aec_wav` CLI is heap-only; `test_static_aec.c` is
the static-path harness.

**`aec_destroy` is a genuine no-op for pool instances on both backends.**
Everything `aec_init` places — including NE10's three R2C/C2R twiddle
configs (vendored patch P0001, `audio_common/lib/ne10/VENDORED.md`) — lives
inside `pool` and is included in `aec_get_mem_size`'s figure. The call is
safe, idempotent, and optional before releasing/reusing the pool; strict
init-to-destroy zero-heap is allocator-hook-verified by
`test/test_zero_heap_aec.c` on both backends.
Full design notes and per-module breakdown:
[../c_impl/STATIC_MEMORY.md](../c_impl/STATIC_MEMORY.md).

### Module ownership

| Header | Source | Owns |
|---|---|---|
| `aec.h` | `aec.c` | top-level orchestration, `AecConfig`, control flow |
| `hpf.h` (audio_common) | `hpf.c` (audio_common) | 80 Hz Butterworth biquad (f32, DF2-transposed — the shared platform HPF) |
| `saturation.h` | `saturation.c` | clip detection + ref-side soft-clip |
| `delay_aec3.h` | `delay_aec3.c` | AEC3-style matched-filter bank + lag-histogram aggregator + clockdrift detector for delay estimation (replaces the legacy GCC-PHAT estimator) |
| `pbfdkf.h` | `pbfdkf.c` | PBFDAF base + PBFDKF (G1 KX-blended P-update) |
| `detectors.h` | `detectors.c` | RenderActivity + FilterConvergence + DoubleTalkAnalyzer |
| `epc_shadow.h` | `epc_shadow.c` | EchoPathChangeDetector + ShadowCopyController |
| `aec3_post.h` | `aec3_post.c` | **AEC3 post-filter driver** — PSD derivation + coherence-ERLE gate + CNG + OLA (production post-stage) |
| `aec_state.h` / `residual_echo_estimator.h` / `suppression_gain.h` / `reverb_model.h` | resp. `.c` | post-filter sub-modules: AecState · ResidualEchoEstimator · SuppressionGain (ENR/EMR `GainToNoAudibleEcho`) · ReverbModel |
| `aec_debug.h` | `aec_debug.c` | timestamped log infrastructure |

> `AecConfig` has grown beyond what this guide covers — newer fields
> include warm tap-transfer (`delay_acquire_warm_transfer`,
> `delay_acquire_inst_erle_db`), DT-aware recovery
> (`dt_aware_recovery_soft`, `dt_aware_res_floor_enabled`,
> `min_gain_floor_dt_db`, `ne_recent_*`), and filter-misadjustment
> estimation (`filter_misadjustment_*`). This guide does not maintain a
> full config table — read `c_impl/include/aec.h` directly for the
> complete field list.

---

## 4. Integration rules

These cause correctness failures (not just style issues):

1. **Do not bypass the linear filter and feed mic directly to RES.**
   The AEC3 post-filter consumes the linear filter's `echo_spec` as primary
   echo signal. Without a converged linear estimate, RES has nothing to
   suppress.
2. **Do not feed non-`hop_size` blocks.** `aec_process` requires
   exactly `hop_size` samples per call (query `aec_hop_size`; e.g. 256 @16k
   default, 512 @48k). Use a ring
   buffer upstream for variable-size input.
3. **Do not change `filter_length_ms` mid-stream.** Partition count is
   fixed at `aec_create`. Mid-stream changes invalidate `P` / `Q` / `W`
   array bounds.
4. **Do not omit `aec_reset` between independent streams.** DT / EPC /
   convergence state accumulates across calls. For batch processing,
   call `aec_reset` before each new file.
5. **Do not stack CNG layers.** If `enable_cng = 1`, do NOT run RES
   output through another CNG. Stacking double-shapes the noise floor →
   tonal artifacts.
6. **Do not assume `enable_delay_est = 1` is free in offline mode.**
   The delay estimator triggers a filter reset on first acquisition
   (~300 ms learning loss). For offline batch with pre-aligned files,
   set `cfg.enable_delay_est = 0`.
7. **Do not share an `Aec` instance across threads.** Each instance is
   single-threaded. Multi-stream → multiple instances.
8. **Do not build without `-ffp-contract=off`.** FMA contraction
   produces inconsistent detector state.

---

## 5. Operating conditions

### 5.1 Input format

| Field | Value |
|---|---|
| Sample rate | 8 / 16 / 48 kHz |
| Bit depth | 16-bit PCM or 32-bit float |
| Channels | mono |
| Frame / hop | frame=FFT、hop=frame/2；8k 256/128、16k 512/256（可選256/128）、48k 1024/512 |
| Filter length | 52 ms @ 8/16 kHz；64 ms @ 48 kHz |

### 5.2 Prerequisites

| | |
|---|---|
| Reference correct | `ref` must be the playback loopback of what reaches the speaker |
| Delay aligned | `mic - ref` delay must fit within `filter_length`. Online delay estimation handles this if delay < `max_delay_ms` (default 1024 ms) |
| Sync mic and ref | same SR, time-aligned start. Drift will tank ERLE |
| Sufficient far energy | silent / near-silent ref will not drive convergence |

### 5.3 Limits

| | |
|---|---|
| Sample rate | 8 / 16 / 48 kHz only — anything else needs external resample |
| Channels | mono only — no mic array / stereo ref |
| Nonlinear echo | linear filter + RES; no dedicated nonlinear model |
| Delay direction | positive only (mic lags ref). Negative-delay scenarios must be aligned upstream |

### 5.4 Scenario notes

| | |
|---|---|
| Cold start | first 0.5–2 s before filter converges; RES uses conservative render-based estimate |
| Echo-path change | EPC detector triggers fast re-convergence (~200 ms hangover) |
| High coupling (small device) | mic ≈ speaker proximity; ~1–2 dB residual leak possible during DT |
| DT-from-frame-0 | NE present from sample 0 prevents convergence — see [aec_methods.md 附錄 E](aec_methods.md#附錄-e-dt-from-frame-0-限制) |

---

## 6. Resources & threading

| | |
|---|---|
| Static pool, KISS (host/reference, `make`) | 538,320 B (525.7 KB) |
| Static pool, NE10 (embedded, `make BACKEND=ne10`) | 534,192 B (521.7 KB, twiddle configs in-pool since P0001) |
| Compute / frame | 4 × 512-FFT + Kalman update (257 bins × 6 partitions) |
| FFT | KISS FFT (float32; NE10 ARM-NEON opt-in) — ~float32 precision vs numpy `np.fft` |

Both backends are static==dynamic byte-equal (`test_static_aec`); full
per-region breakdown → [`../c_impl/STATIC_MEMORY.md`](../c_impl/STATIC_MEMORY.md).

- **Threading**: each `Aec` instance is single-threaded. Multi-stream →
  multiple instances; no shared state.
- **Real-time**: after `aec_create`, `aec_process` and `aec_reset` do
  not allocate or do I/O.

---

## 7. Debug logging

```bash
./bin/aec_wav mic.wav ref.wav out.wav --debug-level 2 --debug-log /tmp/aec.log
grep "PBFDKF" /tmp/aec.log | head
```

Format: `[AEC][t=  1.234s][f=  154][PBFDKF] mu_mean=... P_mean=...`. Each
line is grep-friendly key=value pairs. Release builds (`-DNDEBUG`) strip
log strings entirely.

### Build-time switches

| Flag | Purpose |
|---|---|
| `-DAEC_DEBUG` | enable runtime debug log call sites |
| `-DNDEBUG` | strip debug log strings & assertions (release) |

---

## 8. Common adjustments

### 8.1 Residual echo too high

1. **Verify ref**: with `--no-res`, output should be a clean linear-AEC
   residual. If echo still dominates → ref signal is wrong, mic/ref
   delay > filter length, or SRs differ.
2. **Delay alignment**: `--no-delay-est` disables the online estimator;
   if leaking, measure actual mic-ref delay and bump `--filter-length-ms`.
3. **Stronger RES**: `--preset aggressive` (cost: more NE compression).
4. **Longer filter**: `--filter-length-ms 100` for big rooms / long
   reverb.

### 8.2 NE clipped during double-talk

- **Lower preset**: `--preset balanced` or `mild`.
- Don't tweak individual RES knobs — preset values are co-tuned.

### 8.3 Slow startup / first-second echo

Filter convergence needs ≥ 0.5 s of meaningful far-end energy. Normal
adaptive behavior. Consider muting output during application warm-up
(e.g. play a "connecting..." cue).

### 8.4 Echo spikes when device moves

Echo path changes → EPC fires → ~200 ms re-convergence with brief leak.
Usually recovers automatically. For frequent movement, increase filter
length to broaden model capacity.

### 8.5 Listen to the linear-AEC residual

```bash
./bin/aec_wav mic.wav ref.wav linear_only.wav --no-res
```

Lets you separate filter-side issues from RES-side issues.

---

## 9. Output format

`aec_wav` writes 32-bit-float WAV by default. Set `AEC_OUT_FLOAT=0` to
force 16-bit PCM:

```bash
AEC_OUT_FLOAT=0 ./bin/aec_wav mic.wav ref.wav out.wav
```

Internal pipeline always runs at fp32 sample I/O regardless of file
format.

---

## 10. Streaming contract & sample-rate scaling

### 10.1 Streaming contract

`aec_process(Aec*, const float* mic, const float* ref, float* out)` is
strictly **1 hop in → 1 hop out per call**. There is no internal queue
that absorbs partial frames; the caller must feed exactly `hop_size`
samples each call.

```
frame_size = fft_size; hop_size = fft_size / 2
8 kHz          → 256 / 128
16 kHz default → 512 / 256   (optional low grid: 256 / 128)
48 kHz         → 1024 / 512
```

Get the value at runtime via `aec_hop_size(&aec)`.

**Internals:** main filter and RES use 50 %-overlap sliding blocks
(`block_size = 2 × hop_size`). On each call the input buffer is
`memmove`-shifted by one hop and the new hop appended; FFT runs on the
full block. From the caller's view this is fully streaming — no
build-up phase past the first call.

**Algorithmic latency:**
- With RES enabled (default): **one hop** (8k/16k default 16 ms、16k low-grid
  8 ms、48k 10.667 ms). The delay comes
  from the RES sqrt-Hann analysis × Hann synthesis OLA.
- With `--no-res` / `cfg.enable_res = 0`: effectively **0 ms**.

There is no look-ahead beyond the current block.

> ⚠️ **Wrong sample count = silent corruption.** `aec_process` does not
> validate the input length (no assert, no length parameter). Feeding
> fewer samples reads uninitialised memory; feeding more is ignored.
> The CLI explicitly chunks `for (i = 0; i + hop <= n; i += hop)` and
> drops any tail shorter than one hop.

### 10.1.1 Decoupled render/capture API (async pipelines) — v3.22.5

For real-time pipelines where the far-end (render) and mic (capture) arrive on
**separate calls / threads and not necessarily 1:1**, use the decoupled API
instead of `aec_process`:

```c
AecBufferingEvent aec_analyze_render(Aec* a, const float* ref);          /* buffer one render hop */
AecBufferingEvent aec_process_capture(Aec* a, const float* mic, float* out); /* consume + process one mic hop */
```

A **320 ms render-hop FIFO** absorbs call-scheduling jitter between the two
streams (this is *not* echo delay — the acoustic delay still lives in the
ref ring buffer). Both calls still take exactly `hop_size` samples.

- **Lockstep equivalence:** one `aec_analyze_render(ref)` immediately followed by
  one `aec_process_capture(mic, out)` is **byte-identical to `aec_process(mic,
  ref, out)`** — the FIFO is pure pass-through in lockstep, so the C-side
  byte-equal regression gate is unaffected.
- **Underrun** (capture with empty FIFO): processed with a silent render hop,
  returns `AEC_BUF_RENDER_UNDERRUN`.
- **Overrun** (render past FIFO capacity): the oldest buffered hop is dropped,
  returns `AEC_BUF_RENDER_OVERRUN`.

Check the returned `AecBufferingEvent` (or `aec_last_buffering_event(&aec)`) to
monitor stream health. Verified by `c_impl/test/stream_sim.c` (lockstep 0/400
hops differ; underrun + overrun detection/recovery pass).

### 10.2 Sample-rate auto-derivation

Most sizes are derived from `sample_rate` inline inside `aec_create()`
— there is no separate `aec_derive_sizes()` function. Caller only sets
`sample_rate` (and optionally the sample-count field `filter_length`); the rest is
automatic:

| Field | 8 kHz | 16 kHz | 48 kHz | Auto? |
|---|---|---|---|---|
| `hop_size` | 128 | 256（可選128） | 512 | ✓ (`fft/2`) |
| `block_size` | 256 | 512（可選256） | 1024 | ✓ (`2 · hop`) |
| `fft_size` | 256 | 512（可選256） | 1024 | ✓ (`block_size`，不補零) |
| `n_freqs` | 129 | 257 default／129 low-grid | 513 | ✓ (`fft/2 + 1`) |
| `n_partitions` | 4 (52 ms) | 4 default／7 low-grid (52 ms) | 6 (64 ms) | ✓ (`ceil(filter_length/hop)`) |
| `ref_ring_size` | 16384 | 32768 | 98304 | ✓ (note 2) |
| RES bin resolution | sr / blk | sr / blk | sr / blk | ✓ |
| `filter_length` default duration | 52 ms | 52 ms | 64 ms (note 1) | ✗ user override |
| `highpass_cutoff_hz` (80) | same | same | same | ✗ Hz, auto-correct |
| `max_delay_ms` (1024) | same | same | same | ✗ ms |
| `saturation_threshold` (0.95) | same | same | same | ✗ amplitude |

> Note 1 — Both Python and C default to 64 ms at 48 kHz and 52 ms below
> 44.1 kHz. `filter_length` itself is a sample count, not a millisecond field.
>
> Note 2 — `ref_ring_size` is the sample count for `cfg.delay_buffer_ms`
> (default 2048 ms), floored at `max_delay_ms`-equivalent samples + 4096
> if that floor is larger. With the default `delay_buffer_ms` (2048)
> and `max_delay_ms` (1024), `delay_buffer_ms` drives the value at all
> three standard sample rates (values above).

### 10.3 Gotchas when changing SR

1. **Tuning floats are 16 kHz-flavoured.** `mu`, `kalman_q_*`,
   `res_g_min_db`, `res_dt_reduction`, etc. were tuned at 16 kHz and
   are *not* SR-scaled. They generally still work at 8 / 48 kHz but
   may need light tuning if you push corner cases.
2. **8 kHz hits the high-frequency edge.** The RES stationary-DT mask
   has Hz fades at 3000–4000 Hz; at 8 kHz Nyquist is 4 kHz so the fade
   is right against the edge. Functional but shaped tightly.
3. **No grid has zero-padding margin.** Every grid deliberately uses
   `block_size == fft_size == 2*hop`; overlap-save remains valid because each
   filter partition is one hop long. Longer filters add partitions rather than
   extending a partition beyond one hop.
4. **Non-standard SRs and non-whitelisted FFT sizes are rejected.** Stick to
   8 / 16 / 48 kHz and the table above; resample upstream otherwise.
5. **Latency is grid-dependent.** RES OLA adds one hop; the hop durations are
   8–16 ms depending on the selected rate/grid, not a universal 10 ms.
6. **Project tuning fields ending in `_frames` and several project EMAs remain
   per-hop values.** For example, preset `warmup_frames=100` lasts 1.6 s on the
   16k/512 default grid and about 1.067 s on 48k/1024. AEC3-derived internal
   constants use the `aec3_scale` wall-clock helpers, but these project-level
   knobs still require per-grid qualification before changing production tuning.

---

## 11. Reporting issues

When reporting, please include:

- input wav files (or describe acoustic scenario)
- exact CLI used
- output of `--debug-level 2 --debug-log <path>` for ≥ 1 s around the
  issue
- expected vs observed behavior
