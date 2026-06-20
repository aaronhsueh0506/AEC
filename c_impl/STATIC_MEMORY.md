# Static Memory API — AEC C

## Goal

Provide a heap-free initialization path for embedded targets that
pre-allocate a single memory pool and slice it per module. The existing
`_create / _destroy` heap API stays valid for desktop / unit-test use.

## Top-level usage

```c
#include "aec.h"

AecConfig cfg;
aec_config_from_preset(&cfg, AEC_PRESET_BALANCED, /*sr=*/16000);

size_t pool_bytes = aec_get_mem_size(&cfg);     /* query needed buffer size */
void*  pool       = your_static_pool_alloc(pool_bytes);  /* 16-byte aligned */

Aec aec;
if (aec_init(&aec, pool, pool_bytes, &cfg) != 0) {
    /* pool too small, or NULL inputs */
}

int hop = aec_hop_size(&aec);
float mic[hop], ref[hop], out[hop];

while (read_block(mic, ref, hop)) {
    aec_process(&aec, mic, ref, out);
    write_block(out, hop);
}

aec_destroy(&aec);   /* no-op for static path; safe for both paths */
/* caller frees `pool` itself */
```

`aec_get_mem_size` returns the exact byte count needed for the supplied
config (sample rate, filter length, optional shadow filter, optional delay
ring). At BALANCED / 16 kHz / 52 ms filter length / shadow on / delay on the
pool is **525,760 bytes (513.4 KB)**. RES is value-typed and always present, so
`enable_res` does not change the pool size; `enable_shadow` and `enable_delay_est`
do (shadow ≈ 61.5 KB, delay ring 128 KB).

## Design pattern

Every sub-module that owns dynamic state exposes:

```c
size_t  module_get_mem_size(...);                 /* required pool bytes */
void    module_init_static(Module* m,
                           void* mem, size_t bytes, ...);  /* place state */
```

`*_init_static` walks the supplied buffer with `ALIGN16` boundaries
(defined in `include/fft_wrapper.h`) and assigns every internal field by
pointer arithmetic. The same `*_free` works for both paths because each
struct carries an `is_static` flag — the static branch early-returns
without freeing.

`*_get_mem_size` and `*_init_static` MUST walk fields in identical order
so the size computation matches the placement.

## Module status

`aec_init` places three kinds of state into the pool:

**1. Modules with their own static API** (`*_get_mem_size` + `*_init_static`),
summed by `aec_get_mem_size`:

| Module                          | Heap path     | Static API |
|---|:-:|:-:|
| `aec` (top-level)               | `aec_create`  | `aec_get_mem_size` + `aec_init` |
| `pbfdkf` (main filter)          | `pbfdkf_init` | `pbfdkf_get_mem_size` + `pbfdkf_init_static` |
| `pbfdaf` (shadow filter)        | `pbfdaf_init` | `pbfdaf_get_mem_size` + `pbfdaf_init_static` |
| `fft_wrapper` (KISS, fully in-pool) | `fft_create` | `fft_get_mem_size` + `fft_init` |

**2. Raw buffers** sliced inline by `aec_init` (no per-module API): the delay
`ref_ring`, the render FIFO, and the RSA counters.

**3. Value-typed sub-modules** embedded directly in the `Aec` struct (plain
`*_init`, no heap, no static API); where they own pointer arrays, `aec_init`
slices those from the pool too:

- `DelayAec3` (AEC3 matched-filter delay + ring), `Hpf`, `Saturation`
- the detectors (`RenderActivity` / `FilterConvergence` / `DoubleTalk`),
  `EpcDetector`, `ShadowCopy` (regime handler), `RenderSignalAnalyzer`
- the whole AEC3 post-filter chain: `Aec3Post`, `AecState` (+`AecStateStorage`),
  `ResidualEchoEstimator`, `SuppressionGain`, `StationarityEstimator`,
  `ReverbModel`, `LinearFilterSelect`, and the run / hop scratch

> The legacy `res_filter` / `filter_erle` / `residual_echo` (pre-AEC3 v3.10
> residual suppressor) and `delay_est` (the retired v3.10 delay module,
> superseded by the value-typed `DelayAec3` above) have all been removed and no
> longer exist — their unused `*_get_mem_size` / `*_init_static` symbols are
> gone with them.

Sub-modules are wired internally by `aec_init` — your code only ever calls the
top-level `aec_get_mem_size` / `aec_init`.

## Verification

`test_static_aec.c` runs the same input through both `aec_create` (heap) and
`aec_init` (static pool) and asserts every output sample is byte-equal:

```bash
gcc -O2 -ffp-contract=off -std=c99 -Iinclude -Iexample -Ilib/kiss_fft \
    test_static_aec.c $(find src -name '*.c' ! -name 'fft_wrapper_ne10.c') \
    lib/kiss_fft/kiss_fft.c -lm -o bin/test_static_aec
./bin/test_static_aec mic.wav ref.wav
# → Pool: 525760 bytes (513.4 KB), frames: N
#   PASS: all <2*N> samples byte-equal (static == dynamic)
```

The two paths produce **bit-identical output** across all presets and all three
scenarios (FS / DT / NE). The `aec_wav` CLI itself is heap-only — it does not
expose a `--static-mem` flag; `test_static_aec.c` is the static-path harness.

## Debug logging

Built-in hooks fire on `_init` / `_destroy` (compile-gated by
`-DAEC_DEBUG`, runtime-gated by `--debug-level`):

The static-path pool log is emitted by `aec_init` / `aec_destroy` when compiled
with `-DAEC_DEBUG` and run at a non-zero debug level. The `aec_wav` CLI is
heap-only (`aec_create`), so to see the static-path lines a harness must call
`aec_init` directly and raise the debug level; the format is:

```text
# [AEC][t= 0.000s][f=    0][Init] static-mem pool=525760 bytes (513.4 KB) sr=16000 hop=160 preset_q=0.001 cng=0
# [AEC][t= ...   ][f= ... ][Init] destroy: static path (no free; caller owns pool)
```

Release builds (`make` without `debug`) strip log strings entirely;
no overhead.

## Memory layout (16 kHz, 52 ms filter, BALANCED, shadow on, delay on)

Measured via `aec_get_mem_size`:

| Region | Size |
|---|---:|
| `ref_ring` delay buffer (2048 ms @ 16 kHz)                 | 128 KB |
| Main `pbfdkf` (W, X_buf, P, FFT cfgs, scratch)             | 69.7 KB |
| Shadow `pbfdaf` (same layout, no Kalman P)                 | 61.5 KB |
| `fft_wrapper` post FFT (KISS cfgs in-pool)                 | 16.6 KB |
| `Aec` struct + AEC3 chain backing arrays (state / RES-est / suppression / stationarity / LFS / run + hop scratch) + render FIFO + RSA counters | ~238 KB |
| **Total**                                                  | **513.4 KB** (525,760 B) |

Sample rates other than 16 kHz scale roughly proportional to hop_size.
