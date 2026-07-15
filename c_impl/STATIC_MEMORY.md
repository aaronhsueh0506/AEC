# Static Memory API — AEC C

## Goal

Provide a heap-free initialization path for embedded targets that
pre-allocate a single memory pool and slice it per module. The existing
`_create / _destroy` heap API stays valid for desktop / unit-test use.

## Top-level usage

`aec_init()` returns `Aec*` (NULL on failure) — it does **not** take a
caller-owned `Aec` struct by pointer. The pool must be 16-byte aligned;
`posix_memalign` (POSIX hosts) or your platform's aligned allocator both
work — plain `malloc` is not guaranteed to satisfy the alignment check and
`aec_init` will reject a misaligned base.

<!-- docs-smoke:begin -->
```c
#include "aec.h"
#include <stdlib.h>

AecConfig cfg;
aec_config_from_preset(&cfg, AEC_PRESET_BALANCED, /*sr=*/16000);

size_t pool_bytes = aec_get_mem_size(&cfg);   /* query needed buffer size */
void*  pool = NULL;
if (posix_memalign(&pool, 16, pool_bytes) != 0) {
    /* allocation failed */
}

Aec* aec = aec_init(pool, pool_bytes, &cfg);
if (!aec) {
    /* pool too small, or NULL/misaligned inputs — aec_init() returns NULL */
}

int hop = aec_hop_size(aec);
float mic[hop], ref[hop], out[hop];

while (read_block(mic, ref, hop)) {
    aec_process(aec, mic, ref, out);
    write_block(out, hop);
}

/* aec_destroy() on a pool instance is a genuine no-op on BOTH backends:
 * everything — including NE10's R2C/C2R twiddle configs (vendored patch
 * P0001, see audio_common/lib/ne10/VENDORED.md) — lives inside `pool` and
 * is sized into aec_get_mem_size()'s figure. The call is safe, idempotent,
 * and cheap; strict init-to-destroy zero-heap is enforced by
 * test/test_zero_heap_aec.c (allocator-hook counts stay 0 on both
 * backends). The pool is yours to release or reuse afterwards. */
aec_destroy(aec);
free(pool);   /* caller owns `pool`; substitute your platform's deallocator */
```
<!-- docs-smoke:end -->

The sample above is compiled and run on 1 s of silence by
[`test/docs_smoke.sh`](test/docs_smoke.sh) so this snippet cannot silently
drift from the real `aec.h` signatures again.

`aec_get_mem_size` returns the exact byte count needed for the supplied
config (sample rate, filter length, optional shadow filter, optional delay
ring). At BALANCED / 16 kHz / 52 ms filter length / shadow on / RES on /
delay on the pool is:

| Backend | Pool |
|---|---:|
| KISS (host/reference) | **538,320 B (525.7 KB)** |
| NE10 (embedded)       | **534,192 B (521.7 KB)** |

(Both figures now include everything — NE10's three R2C/C2R twiddle configs
are carved from the pool since vendored patch P0001; there are no
backend-internal heap allocations left on the static path, enforced by
`test/test_zero_heap_aec.c`.) Both figures are static==dynamic byte-equal, verified by
`test_static_aec.c` (669,920 samples, 50-cycle init/destroy leak loop). The
previous figures (537,680 B KISS / 533,552 B NE10) grew by exactly
`ALIGN16(hop_size * sizeof(float))` — 640 B at 16 kHz (hop=160) — for the F09
Variant A' streaming-FIFO rewrite's `fifo_zero_ref`: an immutable all-zero
hop carved right after `render_fifo`, used as the underrun reference so the
capture thread never has to write into the ring (see aec.h/aec.c). Before
that: the old figures (539,328 B / 526.7 KB, KISS only) were stale — since
then: +24.5 KB
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
    $(make -s -C ../../audio_common BACKEND=kiss print-lib-path) -lm -o bin/test_static_aec
./bin/test_static_aec mic.wav ref.wav
# → Pool: 538320 bytes (525.7 KB), frames: N   [KISS 16 kHz; per-rate table below]
#   PASS: all <2*N> samples byte-equal (static == dynamic)
```

(Or just `make` from `c_impl/` — the top-level Makefile resolves and builds
the `audio_common` archive automatically as a real, mtime-tracked
prerequisite of `aec_wav`, not just an order-only trigger; see the
Makefile's two-phase `AC_LIB` resolution comment. Artifacts land in
`bin/<backend>-<config-hash>/` — run `make print-bin-dir` (same flags as
your build) to locate them, or `make publish` for a stable
`dist/<backend>/current/` handoff path. Build against the NE10 archive —
`make BACKEND=ne10 lib` in `audio_common/` — to reproduce the NE10 pool
figure instead.)

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
# [AEC][t= 0.000s][f=    0][Init] static-mem pool=538320 bytes (525.7 KB) sr=16000 hop=160 preset_q=0.001 cng=0
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
| `Aec` struct + AEC3 chain backing arrays (state / RES-est / suppression / stationarity / LFS / run + hop scratch, incl. the de-stacked instance scratch) + render FIFO + `fifo_zero_ref` + RSA counters | ~248.6 KB |
| **Total (KISS)**                                           | **525.7 KB** (538,320 B) |

What moved since the previous figure (526.7 KB / 539,328 B): +24.5 KB of
former function-local stack scratch (13 `float[8192]`/`float[4096]` arrays,
sized by real dims) is now pool-resident instead of stack-resident; −2 KB
from the coherence arrays converting double→float; −3.1 KB because the
PBFDAF process-scratch fields are now flag-gated off for the PBFDKF base
(main filter no longer carries the unused copy); plus small struct deltas
elsewhere. The vectorization campaign then removed another **24.7 KB**: the
per-hop `W_all`/`X_buf_all` snapshot buffers are gone (aec3_post now reads
the filter state directly through its const input pointers). Most recently,
the F09 Variant A' streaming-FIFO rewrite (drop-new + consumer catch-up)
added **+640 B** (16 kHz): `fifo_zero_ref`, an immutable all-zero hop
(`ALIGN16(hop_size * sizeof(float))`) carved right after `render_fifo` and
used as the underrun reference, so the capture thread never writes into the
ring itself. Behaviour-neutral otherwise — `fifo_count` stays in the struct
(RETIRED, always 0) purely for field-offset/layout stability.

**NE10 backend (embedded build): 521.7 KB (534,192 B) total** — the three
R2C/C2R twiddle configs (11,440 B each at nfft=512) are carved from this
pool since vendored patch P0001, so the figure is the complete memory
requirement; everything else in the table above is backend-independent.
(Pre-P0001 the configs were NE10-internal heap allocations outside the pool
and the figure understated the true footprint.)

Per-rate pool totals (balanced, ms-derived filter length 52 ms / 64 ms ≥44.1 k,
all dims auto-derived from the hop = 10 ms rule; verified static==dynamic
byte-equal per rate by `test_static_aec <mic> <ref> <sr>`):

| Rate | KISS | NE10 |
|---|---:|---:|
| 8 kHz  (FL 416, 6 part., taps 480)   |   290,672 B |   288,848 B |
| 16 kHz (FL 832, 6 part., taps 960)   |   538,320 B |   534,192 B |
| 48 kHz (FL 3072, 7 part., taps 3360) | 1,253,680 B | 1,244,944 B |

(Each rate grew by exactly `ALIGN16(hop_size * sizeof(float))` for
`fifo_zero_ref` — 320 B @ 8 kHz (hop=80), 640 B @ 16 kHz (hop=160), 1,920 B
@ 48 kHz (hop=480) — over the pre-F09-Variant-A' figures (290,352 / 537,680 /
1,251,760 B KISS; 288,528 / 533,552 / 1,243,024 B NE10).)

Delay-estimator coverage is spec-inherited (fixed /4 decimation, 64-sample
inner blocks, native-sample reporting — mirrors the Python reference, which
hard-rejects any other decimation): max acquirable delay ≈ 1216 ms @ 8 k /
608 ms @ 16 k / **~203 ms @ 48 k**. The 48 kHz bound is a known spec
limitation — changing it must land Python-first.
