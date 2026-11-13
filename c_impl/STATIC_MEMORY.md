# Static-memory API

AEC supports both heap construction (`aec_create`) and caller-owned static
memory (`aec_get_mem_size` + `aec_init`). The processing algorithm and output
are identical within the same FFT backend.

## Contract

- Call `aec_get_mem_size(&cfg)` after the final sample-rate/FFT configuration.
- Provide one writable, 16-byte-aligned pool of at least that size.
- Keep the pool alive until the last `aec_process` and `aec_destroy` call.
- `aec_init` returns `NULL` for an invalid config, alignment or pool size.
- `aec_destroy` releases heap instances. For pool instances it releases nothing
  (it only drops internal references, marks the handle destroyed and may be called again); the caller still
  owns and releases the pool.
- The static path performs no heap allocation from initialization through
  destruction. `test/test_zero_heap_aec.c` enforces this.
- Heap construction also aligns its arena to 16 bytes explicitly; an ILP32
  allocator's natural 8-byte alignment is insufficient for carved subobjects.
- Pool sizes and public structure layouts belong to the compiled ABI. Query
  them on the target; do not reuse an ARM64 pool-size literal or raw instance
  image in an ARM32 program.

## Example

<!-- docs-smoke:begin -->
```c
#include "aec.h"
#include <stdlib.h>

AecConfig cfg;
aec_config_from_level(&cfg, AEC_LEVEL_BALANCED, 16000);

size_t pool_bytes = aec_get_mem_size(&cfg);
void *pool = NULL;
if (pool_bytes == 0 || posix_memalign(&pool, 16, pool_bytes) != 0) {
    return 1;
}

Aec *aec = aec_init(pool, pool_bytes, &cfg);
if (!aec) {
    free(pool);
    return 1;
}

int hop = aec_hop_size(aec);
float *mic = (float *)malloc((size_t)hop * sizeof *mic);
float *ref = (float *)malloc((size_t)hop * sizeof *ref);
float *out = (float *)malloc((size_t)hop * sizeof *out);
if (!mic || !ref || !out) {
    free(mic); free(ref); free(out);
    aec_destroy(aec);
    free(pool);
    return 1;
}

while (read_block(mic, ref, hop)) {
    aec_process(aec, mic, ref, out);
    write_block(out, hop);
}

free(mic); free(ref); free(out);
aec_destroy(aec);
free(pool);
```
<!-- docs-smoke:end -->

The sample is compiled and run by `test/docs_smoke.sh`.

## Supported grids

| Sample rate | FFT / frame | Hop |
|---:|---:|---:|
| 16 kHz | 256 | 128 |
| 16 kHz | 512 | 256 |
| 48 kHz | 1024 | 512 |

The library also retains an 8 kHz legacy grid for standalone compatibility.
In Audio_ALG the mono samples also accept the 8 kHz row; the 4ch flow and the ULCNet samples accept only the 16 kHz and 48 kHz rows above.

Pool size depends on the complete config, backend and target ABI. Do not copy a
checked-in byte count into firmware. Query `aec_get_mem_size` and reserve that
exact result in the board memory map. For orientation, the balanced 16 kHz
build with the NE10 FFT backend (the only backend) at the default
`delay_mode = MATCHED` / `delay_num_filters = 5` is (`make test-delay-num-filters`
prints these):

| FFT / hop | Pool size |
|---:|---:|
| 256 / 128 | 343,760 B |
| 512 / 256 | 456,640 B |

These figures depend on the struct sizes of `Aec`/`AecConfig`; re-run
`aec_get_mem_size()` (or `make test-delay-num-filters`) rather than carrying
them forward.

The delay configuration moves these numbers, because the matched-filter bank,
the down-sampled render ring and both lag histograms are carved from this same
pool at the configured size rather than at a compile-time maximum. Each filter
dropped from the bank costs 5,728 B less on every grid (n=1 is 22,912 B below
the n=5 default); `FIXED` carves no estimator at all, and
`EXTERNAL_ALIGNED` additionally carves no reference ring. `delay_mode` and
`delay_num_filters` are therefore init-time immutable alongside the signal
grid: changing either means re-querying `aec_get_mem_size` and re-initialising.

## Array instance (`AecConfig.n_mics = 4`)

One `aec_init()` places the whole four-microphone linear flow in one caller pool: four lanes (each the pool of a
single-channel `EXTERNAL_ALIGNED` instance), the shared delay estimator and its aligned-far ring, and the fusion
buffers (plus the one residual-gain stage with `AEC_RESIDUAL_FULL`). The per-hop calls are in
`../docs/reference/integration.md` section 8.8. The descriptor form refuses a pool queried from a different build:

```c
AecConfig cfg;
aec_config_from_level(&cfg, AEC_LEVEL_BALANCED, 16000);
cfg.n_mics = AEC_ARRAY_MICS;               /* 4: array instance; residual defaults to FULL */
cfg.capture_proxy_channel = 0;             /* microphone fed to the shared delay estimator */

AecMemReq req;                             /* descriptor of the pool this build needs */
if (aec_get_mem_requirements(&cfg, &req) != 0) { /* invalid cfg */ }

void *pool = NULL;
if (posix_memalign(&pool, 16, (size_t)req.bytes) != 0) { /* allocation failed */ }

Aec *aec = aec_init_ex(pool, (size_t)req.bytes, &cfg, &req);
if (!aec) { /* pool too small / unaligned, or descriptor from a different build */ }
```

Unlike a single-channel instance, the array pool **depends on `AecConfig.residual`** (`OFF` is refused). Measured with
`aec_get_mem_requirements()` on the 64-bit host (balanced, `MATCHED`, `delay_num_filters = 5`, NE10, `sizeof(Aec)` 6,368 B):

| Grid | `STATE` | `ESTIMATE` | `FULL` (default) |
|---|---:|---:|---:|
| 16 kHz / 256 | 616,912 B | 712,448 B | 720,352 B |
| 16 kHz / 512 | 920,464 B | 1,110,208 B | 1,124,256 B |
| 48 kHz / 1024 | 2,062,960 B | 2,441,120 B | 2,470,528 B |

`aec_get_mem_breakdown()` splits the `FULL` pool as follows. The `Aec` header (`sizeof(Aec)` rounded to 16 B, 6,368 B here) belongs
to none of the four sub-fields, so they sum to `total_bytes` minus the header:

| Grid | `total_bytes` | `estimator_bytes` | `ring_bytes` | `lane_bytes` | `fusion_bytes` |
|---|---:|---:|---:|---:|---:|
| 16 kHz / 256 | 720,352 | 33,936 | 66,576 | 588,800 | 24,672 |
| 16 kHz / 512 | 1,124,256 | 33,936 | 67,600 | 971,968 | 44,384 |
| 48 kHz / 1024 | 2,470,528 | 35,312 | 200,720 | 2,141,248 | 86,880 |

- **Lane contract.** The flow stamps the same config on each lane: the array's grid and tunables, `delay_mode = EXTERNAL_ALIGNED`
  (the lanes take the shared aligned far), `residual = ESTIMATE` (`STATE` for a `STATE` array), `n_mics = 1`. Each lane pool is
  147,200 / 242,992 / 535,312 B (`ESTIMATE` and `FULL`; a `STATE` lane omits the residual-echo estimator and comfort-noise arrays its early exit never reads: 124,880 / 198,656 / 446,944 B) and `lane_bytes` is four of them. A lane is built by the internal
  `aec_lane_init()`; against a stand-alone single-channel `EXTERNAL_ALIGNED` instance (178,752 / 291,632 / 650,032 B)
  it borrows the array's one FFT handle and omits what its contract never runs: the delay-shift snapshots (only the `MATCHED`
  estimator takes them), the synthesis and gain buffers (OLA, `eout`, the SuppressionGain arrays; only `FULL` applies a gain) and
  the FFT handle; a `STATE` lane also omits the residual-echo estimator and comfort-noise arrays (its early exit never reads them).
  Every single-channel instance, lane or not, also omits the render-history rows (read only without the echo-generating window),
  has no `x2_past`, keeps the stationarity history at exactly the ring's length, and its shadow filter follows the main filter (same
  input, same clears, same partition steps), so it shares the main filter's far history and has no `near_spec` /
  `error_spec_windowed` / `spec_scratch` (its frontend is lightweight).
- `fusion_bytes` is the control block, the microphone / aligned-far / fusion buffers, the one FFT handle the four lanes share
  (7,152 / 13,552 / 26,352 B) and, in `FULL`, the gain stage (internal `aec_res_get_mem_size()` 7,904 / 14,048 / 29,408 B). It
  does not include the lane pools. `STATE` has no fusion buffers (10,512 / 17,936 / 32,784 B); `ESTIMATE` has them but no gain
  stage (16,768 / 30,336 / 57,472 B).
- `ring_bytes`: `MATCHED` is `ceil(max_delay_ms * sample_rate / 1000) + 2 * hop + 1` samples (`max_delay_ms` <= 4096), `FIXED` is
  `fixed_delay_samples + hop_size`, `EXTERNAL_ALIGNED` none. `FULL` with `EXTERNAL_ALIGNED`: 619,840 / 1,022,720 / 2,234,496 B.

**Not done yet.** The far-side state is still held per lane (what the lanes share is the delay estimator, the aligned far,
the FFT handle and, within a hop, the far FFT result and the far-only per-hop analysis computed by lane 0, which changes
no pool size: lanes 1-3 keep their own copies of that state and update them only when they stop following lane 0); the lanes are
about 82 / 86 / 87 % of the `FULL` array pool. These sizes are host measurements; the target build's
`aec_get_mem_requirements()` is authoritative.

## Ownership model

The top-level pool contains the `Aec` object, delay and reference-ring storage, FFT plans,
main and shadow filter state, AEC3 post-filter state and processing scratch.
All placement follows the same 16-byte alignment rules used by
`aec_get_mem_size`. An array instance holds, in the same way, the four lane pools, the shared estimator and ring, and the
fusion buffers; the lanes are initialised inside the array's pool and nothing is allocated outside it.

Internal modules may expose their own `*_get_mem_size`/`*_init_static`
functions, but application code should use only the top-level AEC API unless it
is implementing a separately documented internal integration.

## Verification

Run the release checks from `c_impl/`:

```bash
make selftest
make test-counter-saturation
make test-delay-reset
make test-rate-structural
make test-process-context
make test-split-entry
make test-shared-far-spec
make test-shared-fft-handle
make test-delay-num-filters
make test-config-validation
make test-linear-context
make test-zero-heap
make test-static-aec TEST_STATIC_MIC=mic.wav TEST_STATIC_REF=ref.wav TEST_STATIC_SR=16000
test/docs_smoke.sh
```

`aec_get_mem_breakdown(&cfg, &out)` splits the same total into `estimator_bytes` and
`ring_bytes` (`aec_wav --print-mem-size` prints it) and, for an array instance, `lane_bytes` and `fusion_bytes`.
The array tests are `make test-aec-array`, `make test-aec-array-2` and `make test-aec-array-flow`. `aec_init()` fails when the carve of a single-channel
instance or a lane does not end exactly at the size `aec_get_mem_size()` reported (`make test-split-entry-2` sweeps every instance kind). Call order, sequence diagrams and the
parameter tables are on [the HTML pages](../docs/html/reference/aec_usage.html).

Heap-vs-pool equality is checked by `make test-static-aec`.
`make test-zero-heap` runs the allocator-hook acceptance test for the complete
static-pool lifecycle.

For firmware builds without hosted I/O, use `NO_STDIO=1` and run
`make audit-no-stdio`. For the complete public lifecycle API,
see [the C user manual](../docs/c_user_manual_zh_TW.md).
