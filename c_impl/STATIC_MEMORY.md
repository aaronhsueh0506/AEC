# Static Memory API — AEC C

## Goal

Provide a heap-free initialization path for embedded targets (e.g.
Novatek SDK) that pre-allocate a single pool and slice it per module.
The existing `_create / _destroy` heap API stays valid for desktop /
unit-test use.

## Design pattern

Every module that owns dynamic state exposes a companion pair:

```c
size_t  module_get_mem_size(config);              /* required pool bytes */
Type*   module_init(void* mem, size_t bytes,      /* place state in mem */
                     config);
void    module_destroy(Type* m);                  /* no-op for static path */
```

`module_init` walks the supplied buffer with `ALIGN16` boundaries (defined
in `include/fft_wrapper.h`) and places every internal field by pointer
arithmetic. The same `module_destroy` works for both paths because each
struct carries an `is_static` flag — the static branch early-returns
without freeing.

`module_get_mem_size` and `module_init` MUST walk fields in identical
order so the size computation matches the placement.

## Status (2026-05-04)

| Module                    | Heap path | Static API |
|---|:-:|:-:|
| `fft_fp64`                | ✅        | ✅ `fft_get_mem_size / fft_init` |
| `hpf`                     | n/a (value type, embedded) | n/a |
| `saturation`              | n/a (value type, embedded) | n/a |
| `detectors`               | n/a (value type, embedded) | n/a |
| `epc_shadow`              | n/a (value type, embedded) | n/a |
| `residual_echo`           | n/a (value type, embedded) | n/a |
| `aec_debug`               | n/a (no state) | n/a |
| `delay_est`               | ✅        | TODO |
| `erle`                    | ✅        | TODO |
| `dtd`                     | ✅        | TODO |
| `pbfdkf` (2D arrays)      | ✅        | TODO |
| `res_filter`              | ✅        | TODO |
| `aec` (top-level)         | ✅        | TODO |

Top-level user API target:

```c
AecConfig cfg;
aec_config_from_preset(&cfg, AEC_PRESET_BALANCED, 16000);

size_t pool_bytes = aec_get_mem_size(&cfg);
void*  pool       = your_static_pool_alloc(pool_bytes);
Aec*   a          = aec_init(pool, pool_bytes, &cfg);

aec_process(a, mic, ref, out);
aec_destroy(a);   /* no-op when initialised via aec_init */
```

## Debug logging

`AEC_DEBUG_LOG` (in `aec_debug.h`) hooks fire on `_init` / `_destroy`
calls so the embedded target can verify the pool slicing in production.
Enable with `-DAEC_DEBUG` at compile time and `--debug-level 2` at
runtime.
