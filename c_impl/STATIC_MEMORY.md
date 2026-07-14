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
ring). At BALANCED / 16 kHz / 52 ms filter length / shadow on / RES on /
delay on the pool is:

| Backend | Pool |
|---|---:|
| KISS (host/reference) | **557,680 B (544.6 KB)** |
| NE10 (embedded)       | **519,232 B (507.1 KB)** |

(NE10's R2C/C2R twiddle configs are backend-internal heap allocations that
live outside the pool either way — see `fft_destroy`/`aec_destroy` note
below.) Both figures are static==dynamic byte-equal, verified by
`test_static_aec.c` (669,920 samples, 50-cycle init/destroy leak loop). The
old figures (539,328 B / 526.7 KB, KISS only) are stale — since then: +24.5 KB
de-stacked instance scratch (13 former `float[8192]`/`float[4096]`
function-local stack arrays moved into the pool, sized by real dims), −2 KB
coherence arrays converted double→float, −3.1 KB PBFDAF process-scratch now
flag-gated off for the PBFDKF base, plus small struct deltas.

RES is value-typed and always present, so `enable_res` does not change the
pool size; `enable_shadow` and `enable_delay_est` do (shadow ≈ 61.5 KB, delay
ring 128 KB — see the per-region breakdown below for the current split).

## Design pattern

Every sub-module that owns dynamic state exposes:

```c
size_t  module_get_mem_size(...);                 /* required pool bytes */
void    module_init_static(Module* m,
                           void* mem, size_t bytes, ...);  /* place state */
```

`*_init_static` walks the supplied buffer with `ALIGN16` boundaries
(defined in `fft_wrapper.h`, now vendored in the shared `audio_common`
layer and pulled in via `-I`) and assigns every internal field by
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
make -C ../../audio_common BACKEND=kiss lib
gcc -O2 -ffp-contract=off -std=gnu99 -Iinclude -Iexample -I../../audio_common/include \
    test_static_aec.c $(find src -name '*.c') \
    ../../audio_common/bin/kiss/libaudio_common.a -lm -o bin/test_static_aec
./bin/test_static_aec mic.wav ref.wav
# → Pool: 557680 bytes (544.6 KB), frames: N   [KISS backend; NE10 -> 519232 B / 507.1 KB]
#   PASS: all <2*N> samples byte-equal (static == dynamic)
```

(Or just `make` from `c_impl/` — the top-level Makefile builds the
`audio_common` archive as an order-only prereq automatically. Build against
the NE10 archive — `make BACKEND=ne10 lib` in `audio_common/` — to reproduce
the NE10 pool figure instead.)

The two paths produce **bit-identical output** across all presets and all three
scenarios (FS / DT / NE), on each backend independently (NE10 vs KISS output is
*not* bit-identical to each other — a pre-existing, expected difference from the
two FFT implementations; NE10+static == NE10+malloc is byte-equal, same as
KISS). The `aec_wav` CLI itself is heap-only — it does not expose a
`--static-mem` flag; `test_static_aec.c` is the static-path harness.

## Debug logging

Built-in hooks fire on `_init` / `_destroy` (compile-gated by
`-DAEC_DEBUG`, runtime-gated by `--debug-level`):

The static-path pool log is emitted by `aec_init` / `aec_destroy` when compiled
with `-DAEC_DEBUG` and run at a non-zero debug level. The `aec_wav` CLI is
heap-only (`aec_create`), so to see the static-path lines a harness must call
`aec_init` directly and raise the debug level; the format is:

```text
# [AEC][t= 0.000s][f=    0][Init] static-mem pool=557680 bytes (544.6 KB) sr=16000 hop=160 preset_q=0.001 cng=0
# [AEC][t= ...   ][f= ... ][Init] destroy: static path (no free; caller owns pool)
```

Release builds (`make` without `debug`) strip log strings entirely;
no overhead.

## Memory layout (16 kHz, 52 ms filter, BALANCED, shadow on, RES on, delay on)

Measured via `aec_get_mem_size`, KISS backend (host/reference build):

| Region | Size |
|---|---:|
| `ref_ring` delay buffer (2048 ms @ 16 kHz)                 | 128 KB |
| Main `pbfdkf` (W, X_buf, P, FFT cfgs, scratch)             | 66.6 KB |
| Shadow `pbfdaf` (same layout, no Kalman P)                 | 61.5 KB |
| `fft_wrapper` post FFT (KISS cfgs in-pool)                 | 16.6 KB |
| `Aec` struct + AEC3 chain backing arrays (state / RES-est / suppression / stationarity / LFS / run + hop scratch, incl. the de-stacked instance scratch) + render FIFO + RSA counters | ~272 KB |
| **Total (KISS)**                                           | **544.6 KB** (557,680 B) |

What moved since the previous figure (526.7 KB / 539,328 B): +24.5 KB of
former function-local stack scratch (13 `float[8192]`/`float[4096]` arrays,
sized by real dims) is now pool-resident instead of stack-resident; −2 KB
from the coherence arrays converting double→float; −3.1 KB because the
PBFDAF process-scratch fields are now flag-gated off for the PBFDKF base
(main filter no longer carries the unused copy); plus small struct deltas
elsewhere.

**NE10 backend (embedded build): 507.1 KB (519,232 B) total** — smaller than
KISS because the R2C/C2R twiddle configs are NE10-internal heap allocations
that live *outside* this pool (released by `fft_destroy`/`aec_destroy`, not by
freeing the pool); everything else in the table above is backend-independent.

Sample rates other than 16 kHz scale roughly proportional to hop_size.
