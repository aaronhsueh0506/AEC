# Static-memory API

AEC supports both heap construction (`aec_create`, one allocation that `aec_destroy` frees) and caller-owned static
memory (`aec_get_mem_size` + `aec_init`). `Aec` is opaque: a caller never holds one by value or asks for its size; the
pool size is `aec_get_mem_size()`. The processing algorithm and output are identical within the same FFT backend.

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
- Heap construction also aligns its block to 16 bytes explicitly; an ILP32
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

The sample is compiled and run by `make test-docs-smoke` (`test/docs_smoke.sh`).

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
exact result in the board memory map. For orientation, the balanced single-channel
instance with the NE10 FFT backend (the only backend) at the default
`delay_mode = MATCHED` / `delay_num_filters = 5`, measured with `aec_get_mem_size()` on the 64-bit
host (the same for every `AecConfig.residual`, for `SIMD=0/1` and for `AEC_NO_ARRAY`):

| Sample rate / FFT / hop | Pool size | `EXTERNAL_ALIGNED` |
|---|---:|---:|
| 16 kHz / 256 / 128 | 239,232 B | 176,880 B |
| 16 kHz / 512 / 256 | 351,088 B | 288,224 B |
| 48 kHz / 1024 / 512 | 774,000 B | 643,552 B |
| 8 kHz / 256 / 128 (legacy) | 210,864 B | 148,512 B |

These figures depend on the struct sizes of the library's internals; re-run
`aec_get_mem_size()` (or `aec_wav --print-mem-size`) rather than carrying
them forward. The pool holds the instance and everything it carves, including the synthesis window and the decoded
gain-threshold tables (see "Ownership model"); the code and flash side is separate: `libaec.a` `__TEXT` is 167,730 B at `SIMD=1`
and 147,482 B at `SIMD=0`, and 143,686 B / 125,694 B built with `AEC_NO_ARRAY=1` (host Apple clang, arm64, `size` summed over the
members as in `README.md`; the target toolchain's numbers differ).

The delay configuration moves these numbers, because the matched-filter bank,
the down-sampled render ring and the lag histogram are carved from this same
pool at the configured size rather than at a compile-time maximum. Each filter
dropped from the bank costs 5,120 B less in the estimator on every grid, plus
(single channel, `MATCHED`) a reference ring shorter by 1,536 samples: 11,264 B
per filter at 16 / 8 kHz and 23,552 B at 48 kHz (n=1 is 45,056 B / 94,208 B
below the n=5 default); `FIXED` carves no estimator at all, and
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
`aec_get_mem_requirements()` on the 64-bit host (balanced, `MATCHED`, `delay_num_filters = 5`, NE10, `Aec` header 6,032 B):

| Grid | `STATE` | `ESTIMATE` | `FULL` (default) |
|---|---:|---:|---:|
| 16 kHz / 256 | 420,736 B | 514,688 B | 524,448 B |
| 16 kHz / 512 | 615,536 B | 802,160 B | 820,112 B |
| 48 kHz / 1024 | 1,366,976 B | 1,738,944 B | 1,776,352 B |

`aec_get_mem_breakdown()` splits the `FULL` pool as follows. The `Aec` header (6,032 B here, a multiple of 16) belongs
to none of the four sub-fields, so they sum to `total_bytes` minus the header:

| Grid | `total_bytes` | `estimator_bytes` | `ring_bytes` | `lane_bytes` | `fusion_bytes` |
|---|---:|---:|---:|---:|---:|
| 16 kHz / 256 | 524,448 | 29,248 | 33,104 | 383,232 | 72,832 |
| 16 kHz / 512 | 820,112 | 29,248 | 33,616 | 626,176 | 125,040 |
| 48 kHz / 1024 | 1,776,352 | 30,624 | 99,824 | 1,371,456 | 268,416 |

- **Lane contract.** The flow stamps the same config on each lane: the array's grid and tunables, `delay_mode = EXTERNAL_ALIGNED`
  (the lanes take the shared aligned far), `residual = ESTIMATE` (`STATE` for a `STATE` array), `n_mics = 1`. Each lane pool is
  95,808 / 156,544 / 342,864 B (`ESTIMATE` and `FULL`; a `STATE` lane omits the residual-echo estimator and comfort-noise arrays its early exit never reads: 74,016 / 113,248 / 256,560 B) and `lane_bytes` is four of them. A lane is built by the internal
  `aec_lane_init()`; against a stand-alone single-channel `EXTERNAL_ALIGNED` instance (176,880 / 288,224 / 643,552 B)
  it borrows the array's one FFT handle and omits what its contract never runs: the delay-shift snapshots (only the `MATCHED`
  estimator takes them), the synthesis and gain buffers (synthesis window, OLA, `eout`, the SuppressionGain arrays and the decoded ENR/SU threshold tables; only `FULL` applies a gain) and
  the FFT handle; a `STATE` lane also omits the residual-echo estimator and comfort-noise arrays (its early exit never reads them).
  The four lanes run one after another and each finishes its linear and residual stage before the next starts, so the buffers a lane
  writes before it reads inside one call (the main and shadow filters' transform rows, the post-chain magnitude / PSD / mask rows,
  the FORM and filter-analyzer scratch, the tap span, the unused hop outputs, and the filters' constant window tables) are carved
  once per array and borrowed by all four lanes: 45,360 / 74,784 / 169,520 B (`STATE`: 44,832 / 73,744 / 167,456 B), counted in
  `fusion_bytes`. What a lane keeps from one hop to the next, what the views return, and what the donor lane hands to lanes 1-3
  (far PSD, scaled far block, narrow-band mask) stay per lane.
  Every single-channel instance, lane or not, also omits the render-history rows (read only without the echo-generating window),
  has no `x2_past`, keeps the stationarity history at exactly the ring's length, and its shadow filter follows the main filter (same
  input, same clears, same partition steps), so it shares the main filter's far history and has no `near_spec` /
  `error_spec_windowed` / `spec_scratch` (its frontend is lightweight).
- `fusion_bytes` is the control block, the microphone / aligned-far / fusion buffers, the one FFT handle the four lanes share
  (7,152 / 13,552 / 26,352 B), the lanes' shared per-call buffers above and, in `FULL`, the gain stage (internal
  `aec_res_get_mem_size()` 10,000 / 18,192 / 37,648 B). It does not include the lane pools. `STATE` has no fusion buffers
  (56,288 / 93,648 / 204,256 B); `ESTIMATE` has them but no gain stage (63,072 / 107,088 / 231,008 B).
- `ring_bytes`: `MATCHED` is the estimator's reach plus one hop, as for a single channel: `4 * ((n - 1) * 384 + 501) + hop_size`
  samples (three times the first term at 48 kHz; 8,276 samples at n = 5, 16 kHz / 256), `FIXED` is `fixed_delay_samples +
  hop_size`, `EXTERNAL_ALIGNED` none. `FULL` array with `EXTERNAL_ALIGNED`: 462,096 / 757,248 / 1,645,904 B.

**Not done yet.** The far-side state is still held per lane (what the lanes share is the delay estimator, the aligned far,
the FFT handle, the per-call buffers above and, within a hop, the far FFT result and the far-only per-hop analysis computed by
lane 0: lanes 1-3 keep their own copies of that state, the far history `X_buf` / `x2_cache`, the residual-echo delay rows and the
stationarity history, and update them only when they stop following lane 0, because a realign can give one lane a different far
history); the lanes are about 73 / 76 / 77 % of the `FULL` array pool. These sizes are host measurements; the target build's
`aec_get_mem_requirements()` is authoritative.

## Ownership model

The top-level pool contains the `Aec` object, delay and reference-ring storage, FFT plans,
main and shadow filter state, AEC3 post-filter state and processing scratch.
All placement follows the same 16-byte alignment rules used by
`aec_get_mem_size`. An array instance holds, in the same way, the four lane pools, the shared estimator and ring, and the
fusion buffers; the lanes are initialised inside the array's pool and nothing is allocated outside it.

Two tables are built into the pool by `aec_init()` / `aec_create()` instead of living in flash at full size: the synthesis window
(`fft_size` floats, mirrored from a 513-float half table without libm; the pointer `aec_synthesis_window()` returns lives in the
pool, so a borrower such as an `AcIstft` must not outlive the `Aec`) and the ENR/SU gain-threshold tables of the residual gain
stage (run-length coded in flash, `4 * ALIGN16(n_bins * 4)` B decoded: 2,112 / 4,160 / 8,256 B at 16 kHz / 256, 16 kHz / 512, 48 kHz / 1024). Both exist once
per single-channel instance and once per array, and none in a lane. The EMR thresholds stay a flash constant.

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
make test-docs-smoke
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
