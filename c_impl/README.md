# AEC C Implementation

C implementation of the AEC algorithm. Top-level orchestration in
`aec.{h,c}`, CLI binary `bin/<backend>-<config-hash>/aec_wav`.

> **User & integration guide** → [../docs/c_user_manual_zh_TW.md](../docs/c_user_manual_zh_TW.md)
> Algorithm reference → [../docs/aec_methods.md](../docs/aec_methods.md)
> Changelog → [../CHANGELOG.md](../CHANGELOG.md)

## Layout

```
c_impl/
├── include/        public headers
├── src/            sources
├── example/
│   ├── aec_wav.c   CLI entry point
│   └── wav_io.h
├── test/           test harnesses (test/modules/: per-module parity checkers)
└── Makefile
```

## Build

```bash
make            # → bin/<backend>-<config-hash>/aec_wav (CLI binary)
make lib        # → bin/<backend>-<config-hash>/libaec.a (static library)
make SIMD=0     # force every optional SIMD path (including matched filter) to scalar
make PROFILE=1  # compile the per-hop stage timing in (aec_get_last_timing(); off by default)
make clean
```

Artifacts land in a config-hashed `bin/<backend>-<config-hash>/` directory
(switching `BACKEND`/`SIMD`/`PROFILE`/`EXTRA_CFLAGS`/`WERROR` always lands in a fresh one
automatically — no stale-object risk, no manual clean needed). Run `make
print-bin-dir` (same flags as your build) to get the exact path, or `make
publish` to copy this build's artifacts to a stable `dist/<backend>/current/`
handoff path.

Compile flags (already in the Makefile): `-O2 -I include -I example`, plus
`-ffp-contract=off -fno-math-errno` appended **last** (see "FP-contraction
policy" below). Both flags are required: no FMA fusion keeps results
reproducible across builds and compilers, and `-fno-math-errno` lets the
guarded AArch64 `sqrtf` in `fast_math.h` lower to a single `FSQRT` with no
errno-checking libm fallback.

The compiler target selects the architecture path; no source change is needed
for ARM32 or ARM64. Shared handwritten DSP NEON kernels require AArch64, so
ARM32 uses their C fallback even when the shared FFT can use ARM32 NEON.
`SIMD=0` forces those fallbacks and does not select the ABI. Pass a matching
CC/CXX/AR/RANLIB toolchain, and obtain static pool sizes from the target build
rather than copying ARM64 sizes. Heap construction requests a 16-byte arena on
both ABIs.

On ARM32 pass `EXTRA_CFLAGS=-mfpu=neon-vfpv4` (`-mfpu=neon-fp-armv8` for AArch32
ARMv8 cores), not plain `-mfpu=neon`: without hardware FMA every explicit
`fmaf()` (PBFDKF update, complex multiply-accumulate, FFT power spectrum,
`aec_post.c`) turns into a libm call. Results are bit-identical, only slower.
ARMv7 Advanced SIMD flushes subnormals to zero and returns the default NaN, so
`SIMD=1` and `SIMD=0` can differ on ARM32 for subnormal or NaN-payload values;
only the NE10 NEON FFT and, on cores with fused FMA (`-mfpu=neon-vfpv4`), the
delay matched-filter kernels (`make test-delay-matched-neon`;
`-DDA_DISABLE_ARM32_NEON` forces scalar) are affected. On such cores the
matched filter is bit-identical to AArch64 NEON for finite, non-subnormal
inputs; cores without fused FMA keep the scalar path.

### FP-contraction policy

`-ffp-contract=off -fno-math-errno` is a repo-wide policy shared by `audiolib`,
`NR/c_impl`, `AEC/c_impl` and `Audio_ALG/pipelines`. In every Makefile the flags
are appended last, after `EXTRA_CFLAGS`, so nothing a caller passes can land
after them. Each Makefile rejects, at parse time, an `EXTRA_CFLAGS` containing
`-Ofast`, `-ffast-math` or `-ffp-contract=<anything>`, and rejects a
command-line `CFLAGS=`/`CXXFLAGS=`/`CPPFLAGS=`/`LDFLAGS=`/`FP_POLICY=` override
(GNU Make would silently ignore the Makefile's own appends to such a variable).
`EXTRA_CFLAGS`/`EXTRA_LDFLAGS` are the supported hooks:

```
$ make EXTRA_CFLAGS=-ffast-math
*** FP policy conflict: CFLAGS/CXXFLAGS/CPPFLAGS/EXTRA_CFLAGS contains -ffast-math; this repo pins -ffp-contract=off; remove -ffast-math from CFLAGS/CXXFLAGS/CPPFLAGS/EXTRA_CFLAGS.  Stop.
$ make CFLAGS=-O3
*** CFLAGS cannot be overridden (origin: command line) -- it would silently drop this Makefile's own flag appends (FP policy, NO_STDIO, include paths); use EXTRA_CFLAGS / EXTRA_LDFLAGS instead.  Stop.
```

`audiolib/common/scripts/audit_fp_contract.sh` disassembles a fixed list of
objects expected to be scalar and fails on any fused multiply-add instruction.
Kernels with explicit fusion (`vfmaq_f32`/`fmaf`, e.g. `utility/simd_aec.h` and
`fft_power` in the NE10 wrapper) are exempt; see the script header.

### No-stdio library builds (`NO_STDIO=1`)

`make lib NO_STDIO=1` produces a `libaec.a` without the per-hop debug trace:
`src/aec_debug.c` (the library's only stdio translation unit) is excluded and
its call site in `aec.c` is compiled out under `AEC_NO_STDIO`, so the archive
has no `fprintf`/`vfprintf`/`stderr` reference. The gate is compile-time and
default-off (`NO_STDIO ?= 0`). Only the `lib` goal supports it: the CLI
(`aec_wav`) uses the `FILE*`-based debug API (`--debug-trace`). Board / MCU
images that link `libaec.a` without hosted stdio should build with
`make lib NO_STDIO=1`; `make audit-no-stdio` builds that archive, checks that the
debug object is absent, scans the archive and a minimal linked consumer for
stdio symbols, and runs the consumer.

## Run

```bash
BIN="$(make -s print-bin-dir)"
"$BIN"/aec_wav mic.wav ref.wav out.wav --level balanced --cng
```

Levels (single residual-echo strength axis `min_gain_floor_far_active_db`,
plus `no_floor`'s separate boolean gate `min_gain_split_floor_enabled`):
`mild` (−20, near-priority) / `balanced` (−28, production) /
`aggressive` (−38, echo-priority) / `no_floor` (floor mechanism off
entirely — `--level 3` or `--level no_floor`, deepest point on the axis,
most near-end loss). `--preset` is accepted as an alias of `--level`; the
Python twins call the same axis "preset" (C level == Python preset). Other
flags: `--cng`, `--debug` (one status line per second), `--debug-trace <path>` (opt-in per-frame CSV of post-filter internals —
`usable_linear`, `fullband_erle`, `r2_mean`, `gain_mean`, `comfort_noise_mean`,
… one row per hop; one NULL test per hop when off). `aec_wav --help` lists all
options.

## Precision & regression anchors

Production C is float32 end-to-end (delay chain, orchestrator scalars,
post/state modules, `residual_echo_estimator`, HPF). The Python reference
(`python/aec.py`, fp64) is the **algorithm spec**; this C port is the float32
**implementation**; Python↔C comparison is **tolerance-based**, never 0/0.

The delay-estimator matched filter is float32/fast-math/duty-cycled
unconditionally (see `include/delay_aec.h`).

The **FFT backend is NE10 (ARM NEON, float32)**; the FFT wrapper and the NE10
backend live in the shared `audiolib` layer (a sibling repo). It is the only
backend (`make BACKEND=ne10`, the default), used by both the malloc path and the
caller-pool path (`aec_get_mem_size`/`aec_init`), and differs from numpy's fp64
`np.fft` by float32 precision. The static path is byte-equal to the malloc path
(`test_static_aec.c`).

**Test tiers** (run from `c_impl/`; failures are collected and listed at the end):

- `make test` — the default tier: every `test-*` target below in sequence, the
  build gates (`test-no-testing-symbols`, `audit-no-stdio`, `audit-profile`), the
  pytest suite (`test-python`), the lean Python↔C parity suite (10 scenes, one
  per mechanism a planted regression is known to move) and the UBSan wrappers
  `test/run_counter_saturation_ubsan.sh` and `test/run_selftest_ubsan.sh`.
  Members whose result cannot differ between SIMD settings (`TEST_SIMD_INDEP` in
  the Makefile, and `audit-profile`) and members that build their own
  configurations (pytest, the parity suite, the sanitizer wrappers) run only
  under `SIMD=1`. The complete default tier is therefore `make test SIMD=1`
  plus `make test SIMD=0`.
- `make test-full` — the default tier plus what is slow or rarely changes: the
  full parity matrix (`make test-pyc-parity PYC_ARGS=--full`, 46 scenes), the
  end-to-end gate `make test-parity-e2e`, the SIMD-independent targets under
  `SIMD=0` too, the UBSan `--use-standard-math` variants,
  `test/run_fp_flag_policy_test.sh` (negative cases of the Makefile flag policy)
  and `test/run_ubsan_signal_handling_test.sh` (the traps of the two UBSan
  wrappers). Under `SIMD=0` it does not repeat the members that take no SIMD
  input.
- Not in either tier, still runnable on their own: `test-detectors-parity`
  (single grid; `test-detectors-parity-all` is a superset), `test/docs_smoke.sh`,
  `test/test_lifecycle.c` and `test/test_wav_io_shim.c` (these two have no
  Makefile target; their build lines are in the files).

**Regression gates:**

- `test/parity_aec_e2e.c` (`make test-parity-e2e`, part of `make test-full`) —
  the end-to-end gate: full `aec_create` → `aec_process` output against the
  Python golden within 2e-2 (1e-1 at 48 kHz) over the ~4186-hop doubletalk case.
- `test/test_counter_saturation.c` (`make test-counter-saturation`) — every
  unbounded hop/call counter stops at its cap: cap-1 → cap, cap → cap,
  no runaway past the cap, and the comparison the counter feeds gives the same
  result at the cap as at a synthetic huge value.
  `test/run_counter_saturation_ubsan.sh` runs the same source under
  `-fsanitize=undefined` (like `test/run_selftest_ubsan.sh`).
- `test/test_rate_structural.c` (`make test-rate-structural`) — per-rate
  structure at each of the four whitelisted grids (8k/256, 16k/256, 16k/512,
  48k/1024): synthesis-window COLA, impulse-through-linear and
  impulse-through-full-post-chain convergence, `aec_get_mem_size`
  monotonicity/level-invariance, `aec_init` (pool) byte-equal to `aec_create`
  (heap), and that `shadow_err_alpha`/`warmup_frames`/`epc_hangover`/
  `ne_recent_hold`/`filter_misadjustment_{stable,hangover}_frames`/
  `EPV_FAST_TC`/`EPV_SLOW_TC` cover approximately the same wall-clock duration
  at every grid.
- **Kernel gates** (NEON twins against their scalar twins; `sk_max_abs_f32` and
  `sk_stationary_mask_u8` also against the loop each replaces): `make selftest`
  (the `sk_` kernels of `utility/simd_aec.h`), `make test-aec-res`,
  `make test-neon-twins` and `make test-delay-kernels`. The last one is the only
  gate that sees the 48 kHz anti-alias pair kernel of `delay_aec.c`: one unit in
  the last place of the filtered stream does not move the integer delay
  estimate. With `SIMD=0` the NEON twin does not exist and `selftest`,
  `test-neon-twins` and `test-delay-kernels` compare scalar code with itself;
  the hop peak magnitude (`sk_max_abs_f32`) and the stationary mask's fold limit
  are then held by the written-out reference loops in `make selftest`.
  "Bit-identical" for a kernel means the same bits except the payload bits of a
  NaN result.

`python/diag/gen_detectors_golden.py` (`make test-detectors-parity`) and
`python/diag/gen_aec_e2e_golden.py` (`make test-parity-e2e`) are the golden
generators with a C checker in this tree; the other `python/diag/gen_*_golden.py`
scripts only dump Python reference state.

## Static-memory (heap-free pool)

For embedded targets, build one pool and place the whole instance in it:

```c
size_t bytes = aec_get_mem_size(&cfg);   /* query at runtime; size is backend/config dependent */
void*  pool  = your_static_alloc(bytes); /* MUST be 16-byte aligned (posix_memalign, etc.) */
Aec*   a     = aec_init(pool, bytes, &cfg);  /* NULL on failure; byte-equal to aec_create output */
/* ... aec_process(a, ...) ... */
aec_destroy(a);   /* pool instance: releases nothing (drops internal references only); idempotent */
free(pool);
```

`aec_init` returns a pointer into `pool` (not an out-param) and is byte-equal to
the malloc path (`test_static_aec.c`). Both memory models ship in the same
library: `aec_create` (heap) and `aec_get_mem_size`/`aec_init` (caller pool) are
always compiled and selected at runtime; which constructor you call is the only
switch. The single NE10 R2C/C2R twiddle config of the one FFT handle shared by the
post-filter, main, shadow and observer filters (`test/test_shared_fft_handle.c`) is
carved from `pool` (vendored patch P0001, see
audiolib/common/lib/ne10/VENDORED.md), so the static path is heap-free —
verified with an allocator hook by `test/test_zero_heap_aec.c`.

Full CLI options, C API reference, integration rules, runtime resource
notes, and validation steps:
[../docs/c_user_manual_zh_TW.md](../docs/c_user_manual_zh_TW.md). HTML pages (call order with a
compile-checked example, sequence diagrams, external/internal parameter tables):
[../docs/html/aec_usage.html](../docs/html/aec_usage.html).
