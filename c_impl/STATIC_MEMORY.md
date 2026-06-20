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
config (sample rate, filter length, optional shadow / RES / delay-est
sub-modules). At BALANCED / 16 kHz / 52 ms filter length / shadow on /
RES on / delay-est on the pool is ~700 KB.

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

| Module                    | Heap path | Static API |
|---|:-:|:-:|
| `aec` (top-level)         | `aec_create`     | `aec_get_mem_size` + `aec_init` |
| `pbfdkf` (with PBFDAF base) | `pbfdkf_init` | `pbfdkf_get_mem_size` + `pbfdkf_init_static` |
| `res_filter`              | `res_filter_init` | `res_filter_get_mem_size` + `res_filter_init_static` |
| `delay_est`               | `delay_est_init` | `delay_est_get_mem_size` + `delay_est_init_static` |
| `filter_erle` (inside res_filter) | `filter_erle_init` | `filter_erle_get_mem_size` + `filter_erle_init_static` |
| `fft_wrapper` (KISS, fully in-pool) | `fft_create` | `fft_get_mem_size` + `fft_init` |
| `hpf`, `saturation`, `detectors`, `epc_shadow`, `residual_echo`, `aec_debug` | value-typed (no heap) | n/a |

Sub-modules with heap allocations are wired internally by `aec_init` —
your code only ever calls the top-level `aec_get_mem_size` / `aec_init`.

## Verification

The CLI binary supports a `--static-mem` flag for quick comparison:

```bash
./bin/aec_wav mic.wav ref.wav out_heap.wav --preset balanced
./bin/aec_wav mic.wav ref.wav out_static.wav --preset balanced --static-mem
md5 out_heap.wav out_static.wav        # must match
```

The two paths produce **bit-identical output** across all four presets
and all three scenarios (FS / DT / NE) on the AEC Challenge dataset.

## Debug logging

Built-in hooks fire on `_init` / `_destroy` (compile-gated by
`-DAEC_DEBUG`, runtime-gated by `--debug-level`):

```bash
make debug                                                  # builds with -DAEC_DEBUG
./bin/aec_wav mic.wav ref.wav out.wav --static-mem --debug-level 1
# stderr:
# [AEC][t= 0.000s][f=    0][Init] static-mem pool=716416 bytes (699.6 KB) sr=16000 hop=160 preset_q=0.001 cng=0
# Processed N frames ...
# [AEC][t= 21.75s][f= 2175][Init] destroy: static path (no free; caller owns pool)
```

Release builds (`make` without `debug`) strip log strings entirely;
no overhead.

## Memory layout (16 kHz, 52 ms, BALANCED preset, shadow on, RES on, delay-est on)

| Region | Approx size |
|---|---:|
| `delay_est` ring + GCC buffers + FFT | ~270 KB |
| Main `pbfdkf` (W, X_buf, P, FFT, scratch) | ~85 KB |
| Shadow `pbfdkf` (same layout)       | ~85 KB |
| `res_filter` (per-bin + window + FFT + erle) | ~270 KB |
| Hop scratch (mic/far/raw/res)       | ~3 KB |
| **Total**                           | **~700 KB** |

Sample rates other than 16 kHz scale roughly proportional to hop_size.
