# Static-memory API

AEC supports both heap construction (`aec_create`) and caller-owned static
memory (`aec_get_mem_size` + `aec_init`). The processing algorithm and output
are identical within the same FFT backend.

## Contract

- Call `aec_get_mem_size(&cfg)` after the final sample-rate/FFT configuration.
- Provide one writable, 16-byte-aligned pool of at least that size.
- Keep the pool alive until the last `aec_process` and `aec_destroy` call.
- `aec_init` returns `NULL` for an invalid config, alignment or pool size.
- `aec_destroy` releases heap instances. It is a no-op for pool instances; the
  caller still owns and releases the pool.
- The static path performs no heap allocation from initialization through
  destruction on KISS and NE10. `test/test_zero_heap_aec.c` enforces this.

## Example

<!-- docs-smoke:begin -->
```c
#include "aec.h"
#include <stdlib.h>

AecConfig cfg;
aec_config_from_preset(&cfg, AEC_PRESET_BALANCED, 16000);

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

The library also retains an 8 kHz legacy grid for standalone compatibility;
Audio_ALG production pipelines accept only the three rows above.

Pool size depends on the complete config, backend and target ABI. Do not copy a
checked-in byte count into firmware. Query `aec_get_mem_size` and reserve that
exact result in the board memory map. For orientation, the balanced 16 kHz
build measured on the current host, at the default `delay_mode = MATCHED` /
`delay_num_filters = 5`, is:

| Backend | 256 / 128 | 512 / 256 |
|---|---:|---:|
| KISS | 385,440 B | 513,968 B |
| NE10 | 384,832 B | 512,592 B |

These four figures were re-measured on both backends after the linear filter
took a per-partition `|X|²` mirror and a far-PSD hold into its own state (and
gave back the far-power EMA array it no longer maintains), so each moved by
+5,664 B at 256 / 128 and +5,120 B at 512 / 256. Every difference below is
unchanged: the growth is per instance, not per delay filter.

The delay configuration moves these numbers, because the matched-filter bank,
the down-sampled render ring and both lag histograms are carved from this same
pool at the configured size rather than at a compile-time maximum. Each filter
dropped from the bank costs 5,728 B less on every grid (n=1 is 22,912 B below
the n=5 default); `FIXED` carves no estimator at all, and
`EXTERNAL_ALIGNED` additionally carves no reference ring. `delay_mode` and
`delay_num_filters` are therefore init-time immutable alongside the signal
grid: changing either means re-querying `aec_get_mem_size` and re-initialising.

## Ownership model

The top-level pool contains the `Aec` object, delay/FIFO storage, FFT plans,
main and shadow filter state, AEC3 post-filter state and processing scratch.
All placement follows the same 16-byte alignment rules used by
`aec_get_mem_size`.

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
make test-shared-far-spec
make test-shared-fft-handle
make test-delay-num-filters
make test-config-validation
make test-linear-context
make test-zero-heap
make test-static-aec TEST_STATIC_MIC=mic.wav TEST_STATIC_REF=ref.wav TEST_STATIC_SR=16000
test/docs_smoke.sh
```

Repeat with `BACKEND=ne10` for the embedded FFT backend. Heap-vs-pool equality
is checked within each backend; KISS and NE10 are not expected to be
bit-identical to each other. `make test-zero-heap` runs the allocator-hook
acceptance test for the complete static-pool lifecycle.

For firmware builds without hosted I/O, use `NO_STDIO=1` and run
`make audit-no-stdio`. For the complete public lifecycle and streaming API,
see [the C user manual](../docs/c_user_manual_zh_TW.md).
