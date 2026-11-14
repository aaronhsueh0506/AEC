# AEC C Implementation

C implementation of the AEC algorithm. Top-level orchestration in
`aec.{h,c}`, CLI binary `bin/<backend>-<config-hash>/aec_wav`.

> **User & integration guide** → [../docs/c_user_manual_zh_TW.md](../docs/c_user_manual_zh_TW.md)
> Algorithm reference → [../docs/aec_methods.md](../docs/aec_methods.md)
> Changelog → [../CHANGELOG.md](../CHANGELOG.md)

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

Headers. `include/` holds the one header a caller sees: `aec.h` (the API; `Aec` is an opaque `typedef struct Aec Aec;`).
Every other header is in `include/internal/`: `aec_internal.h` (library-internal API for pipelines),
`aec_balanced_config.h` (generated per-rate tables), the body of `struct Aec` (`aec_impl.h`) and the sub-module headers. The
library, the CLI and the tests compile with `-I include -I include/internal` (the Makefile adds both); an integrator using
only the public API needs `-I include`, and an SDK Makefile that compiles `src/*.c` needs `-I include/internal` as well.
White-box tests reach the internal headers through `test/aec_white_box.h`; the public-header checks never add
`include/internal`.

Compile flags (already in the Makefile): `-O2 -I include -I include/internal -I example`, plus
`-ffp-contract=off -fno-math-errno` appended **last** (see "FP-contraction
policy" below). Both flags are required: no FMA fusion keeps results
reproducible across builds and compilers, and `-fno-math-errno` lets the
guarded AArch64 `sqrtf` in `fast_math.h` lower to a single `FSQRT` with no
errno-checking libm fallback.

The compiler target selects the architecture path; no source change is needed
for ARM32 or ARM64. Shared handwritten DSP NEON kernels run on AArch64 and on
32-bit ARM with NEON plus fused FMA (`-mcpu=cortex-a32/a53/a55/a73`,
`-mfpu=neon-vfpv4`, `-mfpu=neon-fp-armv8`); `SK_HAVE_NEON` in
`utility/simd_kernels.h` is the one gate. 32-bit ARM without FMA keeps the C
fallback even when the shared FFT can use ARM32 NEON.
`SIMD=0` forces the fallbacks and does not select the ABI. Pass a matching
CC/CXX/AR/RANLIB toolchain, and obtain static pool sizes from the target build
rather than copying ARM64 sizes. Heap construction requests a 16-byte arena on
both ABIs.

On ARM32 pass `EXTRA_CFLAGS=-mfpu=neon-vfpv4` (`-mfpu=neon-fp-armv8` for AArch32
ARMv8 cores), not plain `-mfpu=neon`: without hardware FMA every explicit
`fmaf()` (PBFDKF update, complex multiply-accumulate, FFT power spectrum,
`aec_post.c`) turns into a libm call. Results are bit-identical, only slower.
ARMv7 Advanced SIMD flushes subnormals to zero and returns the default NaN, so
`SIMD=1` and `SIMD=0` can differ on ARM32 for subnormal or NaN-payload values;
NEON kernels equal their scalar twins there for finite, non-subnormal inputs.
The tests that compare the two call `SK_MATCH_SCALAR_TO_VECTOR_MODE()` first.
The delay matched-filter kernels (`make test-delay-matched-neon`) are bit-identical to AArch64 NEON for
finite, non-subnormal inputs; cores without fused FMA keep the scalar path.

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

The library has no stdio: no source under `src/` includes `<stdio.h>`, and the archive carries no
`fprintf`/`vfprintf`/`stderr` reference. `NO_STDIO` (default `0`) selects no source and no flag; it only keys a separate
obj/bin directory, as it does in audiolib and NR, so `make lib NO_STDIO=1` is the archive a board / MCU image without hosted
stdio links. Only the `lib` goal is meant for it (the CLI `aec_wav` uses stdio). `make audit-no-stdio` is the regression gate:
it builds that archive, scans it and a minimal linked consumer for stdio symbols, and runs the consumer.

### Single-channel library (`AEC_NO_ARRAY=1`)

`make lib AEC_NO_ARRAY=1` builds a `libaec.a` without the four-microphone array instance: `src/aec_array.c` and `src/aec_res.c` are not compiled
(`-DAEC_NO_ARRAY`), `AecConfig.n_mics` other than 1 is refused (`aec_get_mem_size()` 0, `aec_init()` NULL, `aec_create()` NULL), and
`aec_get_array()` and the `aec_array_*()` diagnostics are not defined. Everything single-channel behaves byte for byte as in the full library. On the host
the archive's `__TEXT` drops by about 22 kB at `SIMD=1` (164,227 -> 142,015 B) and 19 kB at `SIMD=0` (145,667 -> 126,643 B); measured as `size "$(make -s print-lib-path [SIMD=0] [AEC_NO_ARRAY=1])" | awk 'NR>1{t+=$1} END{print t}'` after `make lib` with the same flags (Apple clang, arm64). The flag is part of the config hash, so the
two builds keep separate `obj/` and `bin/` directories; the array-only test targets print "SKIPPED" under it. Consumers pass it through to this build
(`make AEC_NO_ARRAY=1` in Audio_ALG/pipelines).

一般呼叫與參數可查閱 [AEC 文件](../docs/html/index.html)。

## Precision & regression anchors

Production C is float32 end-to-end (delay chain, orchestrator scalars,
post/state modules, `residual_echo_estimator`, HPF). The Python reference
(`python/aec.py`, fp64) is the **algorithm spec**; this C port is the float32
**implementation**; Python↔C comparison is **tolerance-based**, never 0/0.

The delay-estimator matched filter is float32/fast-math/duty-cycled
unconditionally (see `include/internal/delay_aec.h`).

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
  full parity matrix (`make test-pyc-parity PYC_ARGS=--full`, every scene), the
  end-to-end gate `make test-parity-e2e`, the SIMD-independent targets under
  `SIMD=0` too, the UBSan `--use-standard-math` variants,
  `test/run_fp_flag_policy_test.sh` (negative cases of the Makefile flag policy)
  and `test/run_ubsan_signal_handling_test.sh` (the traps of the two UBSan
  wrappers). Under `SIMD=0` it does not repeat the members that take no SIMD
  input.
- Not in either tier, still runnable on their own: `test-detectors-parity`
  (single grid; `test-detectors-parity-all` is a superset).

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
- `test/test_split_entry.c` (`make test-split-entry`) — the three entry points
  (`aec_process`, `aec_process_linear`, `aec_process_residual`), the views
  (`aec_get_linear`, `aec_get_residual`) and the gain / comfort-noise helpers
  (`aec_apply_gain`, `aec_apply_cng`) in every `AecConfig.residual` mode on three grids, with the order / argument checks.
  Every field of every hop is hashed into a line stream (see "Line-stream baselines" below). `test/test_process_context.c`
  (`make test-process-context`) holds the same entry points to each other: `aec_process`
  against the split pair on one instance and on two, and `STATE` against `FULL`.
- `test/test_split_entry2.c` (`make test-split-entry-2`) — the second single-channel scene set
  (`test/baseline2_mono.h`): a clipping far end and
  microphone, double talk with the near-end-recent latch, a 50 ms path jump with the Path A/B choice, the bad-relock revert and its reject
  veto, clock drift, resets, `FIXED` and `EXTERNAL_ALIGNED` with `aec_apply_external_realign`, `aec_set_level` ramps in flight on all four
  levels, shadow / high-pass / comfort noise / saturation off, live observer, windowed noise tracker, duty cycle and the 8 kHz grid,
  chained per-group hashes. Also `aec_process` alternating with the split calls hop by hop, and the carve of every instance kind
  ending exactly at `aec_get_mem_size()`.
- `test/test_split_entry3.c` (`make test-split-entry-3`) — the third single-channel scene set
  (its case table is in the file): free-running two-path scenes
  in which the bad-relock judgment (windowed ERLE after a confirmed delay shift against the value before it, 1 dB margin) comes
  out by itself, with the ERLE improving (kept), clearly worse (reverted), 0.5 dB either side of the margin, worse without an
  estimate (kept) and on the soft path, on the three product grids; the test prints every judgment (hop, ERLE before / after,
  outcome) and checks it against the case table.
- `test/test_aec_array.c` (`make test-aec-array`) — the four-microphone array instance
  (`AecConfig.n_mics = 4`) driven hop by hop in `STATE`, `ESTIMATE` and `FULL` (hashed into a line stream), plus
  the contract checks (pool vs heap, pool fills and offsets, reset equals fresh, descriptor).
  `test/test_aec_array_flow.c` (`make test-aec-array-flow`, an `AEC_TESTING` build) holds the
  state-machine groups (shared-delay admission, quarantine, duty cycle,
  far-FFT sharing, lane contract, DSP errors).
- `test/test_aec_array2.c` (`make test-aec-array-2`) — the second array scene set
  (`test/baseline2_array.h`): a clipped and a silent microphone,
  double talk, a clipping far end, `capture_proxy_channel` 1-3, a caller-supplied `beamformed_error`, clock drift, `aec_set_level`
  in flight, lanes without the shadow filter, the 32/36-sample admission band, the backward and forward moves under the quarantine
  and the duty cycle, a reset inside double talk; `STATE`, `ESTIMATE` and `FULL` runs, plus per-hop view invariants.
- **Line-stream baselines.** The five tests above (`test-split-entry`, `-2`, `-3`, `test-aec-array`, `-2`) each print or hash
  their fields into a line stream. No baseline is shipped: the one-off files written by the pre-split API were compared
  bit for bit with this version (all three single-channel sets and both array sets, `SIMD=0` and `SIMD=1`) and retired on
  2026-11-13. `AEC_TEST_BASELINE_WRITE=<dir> make test-split-entry` (likewise the other four, each run from `c_impl/`)
  writes `<dir>/split_entry_simd{0,1}.txt` from the build at hand (`simd0` = `SIMD=0`, `simd1` = NEON; nothing on other
  hosts); `AEC_TEST_BASELINE=<dir>` compares every line with it. Capture the baseline from the finished version before the
  next change, compare after it; the files are valid for the host, compiler and FFT backend that wrote them.
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
aec_destroy(a);   /* pool instance: releases nothing (marks the handle destroyed); idempotent */
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

### Array instance

`AecConfig.n_mics = AEC_ARRAY_MICS` (4) builds the whole four-microphone linear flow in one pool: four lanes (each a
single-channel `EXTERNAL_ALIGNED` instance), one shared delay estimator with one aligned far, and the fusion buffers.
The beamformer and the synthesis stay with the caller. Per hop: `aec_process_linear()` (interleaved `float[hop][4]`),
`aec_get_array()`, the caller's beamformer, `aec_process_residual(a, &beam)`, then `aec_get_residual()` /
`aec_apply_gain()` / `aec_apply_cng()`. Query the pool with `aec_get_mem_requirements()` and build with
`aec_init_ex()`; the pool size depends on `residual` (see `STATIC_MEMORY.md`). Not done yet: the far-side state is
still per lane (only the delay estimator, the aligned far and the far FFT result are shared). A lane omits what its
contract never runs: at 16 kHz / 256 a lane pool is 142,880 B against 177,568 B for a stand-alone
`EXTERNAL_ALIGNED` instance. Contract and example:
[../docs/reference/integration.md](../docs/reference/integration.md) §8.8.

Full CLI options, C API reference, integration rules, runtime resource
notes, and validation steps:
[../docs/c_user_manual_zh_TW.md](../docs/c_user_manual_zh_TW.md). HTML pages (call order with a
compile-checked example, sequence diagrams, external/internal parameter tables):
[../docs/html/reference/aec_usage.html](../docs/html/reference/aec_usage.html).
