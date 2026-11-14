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
| 16 kHz / 256 / 128 | 239,920 B | 177,568 B |
| 16 kHz / 512 / 256 | 352,288 B | 289,424 B |
| 48 kHz / 1024 / 512 | 776,224 B | 645,776 B |
| 8 kHz / 256 / 128 (legacy) | 211,552 B | 149,200 B |

These figures depend on the struct sizes of the library's internals; re-run
`aec_get_mem_size()` (or `aec_wav --print-mem-size`) rather than carrying
them forward. The pool holds the instance and everything it carves, including the synthesis window and the decoded
gain-threshold tables (see "Ownership model"); the code and flash side is separate: `libaec.a` `__TEXT` is 164,227 B at `SIMD=1`
and 145,667 B at `SIMD=0`, and 142,015 B / 126,643 B built with `AEC_NO_ARRAY=1` (host Apple clang, arm64, `size` summed over the
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
`aec_get_mem_requirements()` on the 64-bit host (balanced, `MATCHED`, `delay_num_filters = 5`, NE10, `Aec` header 6,208 B):

| Grid | `STATE` | `ESTIMATE` | `FULL` (default) |
|---|---:|---:|---:|
| 16 kHz / 256 | 595,744 B | 691,280 B | 701,280 B |
| 16 kHz / 512 | 883,936 B | 1,073,680 B | 1,091,872 B |
| 48 kHz / 1024 | 1,995,712 B | 2,373,872 B | 2,411,520 B |

`aec_get_mem_breakdown()` splits the `FULL` pool as follows. The `Aec` header (6,208 B here, a multiple of 16) belongs
to none of the four sub-fields, so they sum to `total_bytes` minus the header:

| Grid | `total_bytes` | `estimator_bytes` | `ring_bytes` | `lane_bytes` | `fusion_bytes` |
|---|---:|---:|---:|---:|---:|
| 16 kHz / 256 | 701,280 | 29,248 | 66,576 | 571,520 | 27,728 |
| 16 kHz / 512 | 1,091,872 | 29,248 | 67,600 | 938,304 | 50,512 |
| 48 kHz / 1024 | 2,411,520 | 30,624 | 200,720 | 2,074,816 | 99,152 |

- **Lane contract.** The flow stamps the same config on each lane: the array's grid and tunables, `delay_mode = EXTERNAL_ALIGNED`
  (the lanes take the shared aligned far), `residual = ESTIMATE` (`STATE` for a `STATE` array), `n_mics = 1`. Each lane pool is
  142,880 / 234,576 / 518,704 B (`ESTIMATE` and `FULL`; a `STATE` lane omits the residual-echo estimator and comfort-noise arrays its early exit never reads: 120,560 / 190,240 / 430,336 B) and `lane_bytes` is four of them. A lane is built by the internal
  `aec_lane_init()`; against a stand-alone single-channel `EXTERNAL_ALIGNED` instance (177,568 / 289,424 / 645,776 B)
  it borrows the array's one FFT handle and omits what its contract never runs: the delay-shift snapshots (only the `MATCHED`
  estimator takes them), the synthesis and gain buffers (synthesis window, OLA, `eout`, the SuppressionGain arrays and the decoded ENR/SU threshold tables; only `FULL` applies a gain) and
  the FFT handle; a `STATE` lane also omits the residual-echo estimator and comfort-noise arrays (its early exit never reads them).
  Every single-channel instance, lane or not, also omits the render-history rows (read only without the echo-generating window),
  has no `x2_past`, keeps the stationarity history at exactly the ring's length, and its shadow filter follows the main filter (same
  input, same clears, same partition steps), so it shares the main filter's far history and has no `near_spec` /
  `error_spec_windowed` / `spec_scratch` (its frontend is lightweight).
- `fusion_bytes` is the control block, the microphone / aligned-far / fusion buffers, the one FFT handle the four lanes share
  (7,152 / 13,552 / 26,352 B) and, in `FULL`, the gain stage (internal `aec_res_get_mem_size()` 10,000 / 18,192 / 37,648 B). It
  does not include the lane pools. `STATE` has no fusion buffers (11,472 / 19,920 / 36,816 B); `ESTIMATE` has them but no gain
  stage (17,728 / 32,320 / 61,504 B).
- `ring_bytes`: for the array, `MATCHED` is `ceil(1024 ms * sample_rate / 1000) + 2 * hop + 1` samples (the library's internal
  `max_delay_ms` of 1024), `FIXED` is `fixed_delay_samples + hop_size`, `EXTERNAL_ALIGNED` none. A single-channel `MATCHED`
  instance sizes its ring from the estimator's reach, `4 * ((n - 1) * 384 + 501) + hop_size` samples (three times the first term at
  48 kHz). `FULL` array with `EXTERNAL_ALIGNED`: 605,456 / 995,024 / 2,180,176 B.

**Not done yet.** The far-side state is still held per lane (what the lanes share is the delay estimator, the aligned far,
the FFT handle and, within a hop, the far FFT result and the far-only per-hop analysis computed by lane 0, which changes
no pool size: lanes 1-3 keep their own copies of that state and update them only when they stop following lane 0); the lanes are
about 81 / 86 / 86 % of the `FULL` array pool. These sizes are host measurements; the target build's
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
