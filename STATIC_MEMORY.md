# Static Memory API — AEC (feature/static-memory)

## Overview

Added static memory (pre-allocated buffer) support to all AEC modules.
Each module now provides `_get_mem_size()` and `_init()` in addition to the
existing `_create()` / `_destroy()` API.

When using `_init()`, no internal malloc is called. The caller provides a
pre-allocated buffer and the module places all internal state via pointer
arithmetic with 16-byte alignment (ALIGN16).

## Target

Novatek embedded platform (no heap / limited heap). Pipeline allocates a
single memory pool via PA/VA, then slices it to each module.

## Changed Files

### Headers
- `c_impl/include/fft_wrapper.h` — Added ALIGN16 macro, `fft_init()`, `fft_get_mem_size()`
- `c_impl/include/hpf.h` — Added `hpf_init()`, `hpf_get_mem_size()`
- `c_impl/include/pbfdkf.h` — Added `pbfdkf_init()`, `pbfdkf_get_mem_size()`
- `c_impl/include/res_filter.h` — Added `res_init()`, `res_get_mem_size()`
- `c_impl/include/aec.h` — Added `aec_init()`, `aec_get_mem_size()`, `aec_context_init()`, `aec_context_get_mem_size()`
- `c_impl/include/aec_types.h` — Added `is_static` to AecResContext

### Sources
- `c_impl/src/fft_wrapper.c` — `is_static`, `fft_get_mem_size()`, `fft_init()`, updated `fft_destroy()`
- `c_impl/src/hpf.c` — `is_static`, extracted `hpf_compute_coeffs()` helper, `hpf_get_mem_size()`, `hpf_init()`, updated `hpf_destroy()`
- `c_impl/src/pbfdkf.c` — `is_static`, `pbfdkf_get_mem_size()` (2D array sizing), `pbfdkf_init()` (2D pointer + data placement), updated `pbfdkf_destroy()`
- `c_impl/src/res_filter.c` — `is_static`, extracted `res_setup_config()` and `res_init_state()` helpers, `res_get_mem_size()`, `res_init()`, updated `res_destroy()`
- `c_impl/src/aec.c` — `is_static`, `aec_get_mem_size()` (sums sub-modules), `aec_init()` (calls sub-module `_init()`), `aec_context_get_mem_size()`, `aec_context_init()`, updated `aec_destroy()` and `aec_context_destroy()`

## Memory Layout (16kHz, frame=320, hop=160, fft=512, n_part=10)

```
Aec struct
├── HPF x2 (mic + ref, via hpf_init)
├── PBFDKF main (via pbfdkf_init)
│   ├── FftHandle (via fft_init)
│   ├── W[10][257] (Complex)     ← filter weights
│   ├── X_buf[10][257] (Complex) ← reference buffer
│   ├── P[10][257] (float)       ← Kalman covariance
│   ├── Q, Q_high, Q_low, R, error_psd [257] each
│   ├── near_buffer[512], far_buffer[512]
│   ├── near/far/echo/error_spec[257] (Complex)
│   ├── power[257], temp_time[512], temp_spec[257] (Complex)
│   └── (2D arrays: pointer array first, then data blocks)
├── PBFDKF shadow (same layout as main)
├── mic_buf[160], ref_buf[160], raw_output[160]
└── (optional) ResFilter (via res_init, when enable_res=1)
```

Total AEC (linear, no RES): ~182 KB

### AecResContext

```
AecResContext struct
├── echo_spec_re[257], echo_spec_im[257]
├── far_spec_re[257], far_spec_im[257]
└── near_spec_re[257], near_spec_im[257]
```

Total per context: ~6.2 KB

## 2D Array Placement (PBFDKF)

For `W[n_partitions][n_freqs]` (Complex), placement is:

```c
// Pointer array first
W = (Complex**)ptr;
ptr += ALIGN16(n_partitions * sizeof(Complex*));

// Then contiguous data blocks
for (int p = 0; p < n_partitions; p++) {
    W[p] = (Complex*)ptr;
    ptr += ALIGN16(n_freqs * sizeof(Complex));
}
```

Same pattern for `X_buf` and `P`.

## API Pattern

```c
// 1. Query size
size_t size = aec_get_mem_size(&config);

// 2. Init in pre-allocated memory
Aec* aec = aec_init(mem, size, &config);

// 3. Process (same as malloc version)
aec_process(aec, mic, ref, out);

// 4. Destroy is no-op (is_static=1 → skip free)
aec_destroy(aec);
```

## Notes

- `_create()` API is unchanged and still works (backward compatible)
- `is_static` flag in each struct prevents `_destroy()` from calling free
- All buffers are 16-byte aligned (ALIGN16 macro)
- PBFDKF core is NOT changed (Kalman math identical)
- Shared `hpf_compute_coeffs()` and `res_setup_config()`/`res_init_state()` helpers avoid code duplication between `_create()` and `_init()` paths
