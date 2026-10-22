# AEC C Implementation

C implementation of the AEC algorithm. Top-level orchestration in
`aec.{h,c}`, CLI binary `bin/aec_wav`.

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
├── test/modules/   per-module test harnesses (dev tooling)
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

Compile flags (already in Makefile): `-O2 -I include -I example`, plus
`-ffp-contract=off -fno-math-errno` appended **last** (see "Unified
FP-contraction policy" below). Both flags are **required**: no FMA fusion
is load-bearing for build determinism and golden stability across
builds/compilers (see Precision & regression anchors below), and
`-fno-math-errno` lets the guarded AArch64 `sqrtf` in `fast_math.h` lower to
a single `FSQRT` with no errno-checking libm fallback.

The compiler target selects the architecture path; changing source files is
not required when building for ARM32 or ARM64. Shared handwritten DSP NEON
kernels currently require AArch64, so ARM32 uses their C fallback even when
the shared FFT can use ARM32 NEON. `SIMD=0` forces those DSP fallbacks and does
not select the ABI. Pass a matching CC/CXX/AR/RANLIB toolchain, and obtain static
pool sizes from the target build rather than copying ARM64 sizes. Heap
construction explicitly requests a 16-byte arena on both ABIs.

On ARM32 pass `EXTRA_CFLAGS=-mfpu=neon-vfpv4` (`-mfpu=neon-fp-armv8` for AArch32
ARMv8 cores), not plain `-mfpu=neon`: without hardware FMA every explicit `fmaf()`
(PBFDKF update, complex multiply-accumulate, FFT power spectrum, `aec_post.c`)
turns into a libm call. Results are bit-identical, only slower. ARMv7 Advanced
SIMD flushes subnormals to zero and returns the default NaN, so `SIMD=1` and
`SIMD=0` can differ on ARM32 for subnormal or NaN-payload values; only the NE10
NEON FFT and, on cores with fused FMA (`-mfpu=neon-vfpv4`), the delay matched-filter
kernels (`make test-delay-matched-neon`; `-DDA_DISABLE_ARM32_NEON` forces scalar) are affected.
On such cores the matched filter is bit-identical to AArch64 NEON for finite,
non-subnormal inputs; cores without fused FMA keep the scalar path.

### Unified FP-contraction policy

`-ffp-contract=off -fno-math-errno` is no longer just an AEC convention — it is a **repo-wide
policy spanning all four repos** (`audiolib`, `NR/c_impl`, `AEC/c_impl`,
`Audio_ALG/pipelines`): every translation unit any of their Makefiles compile
— each repo's own sources *and* the vendored NE10 C and C++ TUs alike —
builds with both flags. In every one of the four Makefiles the flags are
appended **last** in the CFLAGS/CXXFLAGS assembly (after `EXTRA_CFLAGS`, after
any BACKEND-conditional append, after `WERROR`/`NO_STDIO`), so nothing a
caller passes can land after them and override them — AEC's Makefile previously
carried the contraction flag as the *third* token of the base CFLAGS assignment (before
`EXTRA_CFLAGS` was folded in), before being moved to its current
trailing position. Each Makefile also rejects outright, at parse time, an
`EXTRA_CFLAGS` (or `CFLAGS=` override) containing `-Ofast`, `-ffast-math`, or
`-ffp-contract=<anything>` (all of which would re-enable contraction), e.g.:

```
$ make EXTRA_CFLAGS=-ffast-math
Makefile:111: *** FP policy conflict: CFLAGS/EXTRA_CFLAGS contains -ffast-math; this repo pins -ffp-contract=off; remove -ffast-math from EXTRA_CFLAGS.  Stop.
```

Build hardening rejects a command-line `CFLAGS=`/`CXXFLAGS=`/`CPPFLAGS=`/
`LDFLAGS=`/`FP_POLICY=` override is now rejected outright in all four
Makefiles — GNU Make silently ignores a Makefile's own `+=`/`:=` assignments
to a command-line-set variable, so `make CFLAGS=-O3` used to strip
`-ffp-contract=off` (and `-DAEC_NO_STDIO`, the `-I` paths, `-lm`) while
still building. `EXTRA_CFLAGS`/`EXTRA_LDFLAGS` are the two supported hooks:

```
$ make CFLAGS=-O3
Makefile:119: *** CFLAGS cannot be overridden (origin: command line) -- it would silently drop this Makefile's own flag appends (FP policy, NO_STDIO, include paths); use EXTRA_CFLAGS / EXTRA_LDFLAGS instead.  Stop.
```

`audiolib/common/scripts/audit_fp_contract.sh` is the disassembly-level proof
the flag actually bites: it disassembles a fixed list of TUs expected to be
genuinely scalar (audiolib's `hpf.o`/NE10's scalar-C
objects, NR's three core objects) and fails if any fmadd/fmsub/fnmadd/
fnmsub/fmla/fmls instruction shows up. Shared AEC kernels in `utility/simd_aec.h` — with
its explicit `vfmaq_f32`-family intrinsics, consumed by AEC's TUs — is the
same "legitimate explicit fusion, not compiler contraction" exemption
category that script documents for its own audit list (see that script's
header comment for the full rationale, including two non-obvious
classifications it derived empirically rather than by filename guess).

### No-stdio library builds (`NO_STDIO=1`)

`make lib NO_STDIO=1` produces a `libaec.a` with the per-hop debug trace
compiled out entirely (`src/aec_debug.c` — the library's only stdio
translation unit, `FILE*`/`fprintf`/`vfprintf`/`fputc` — is excluded from the
archive, and its one call site in `aec.c`'s per-frame trace block is
`#ifndef AEC_NO_STDIO`'d out) — no `fprintf`/`vfprintf`/`stderr` reference
anywhere in the archive, defined or undefined. This is a compile-time,
default-OFF gate (`NO_STDIO ?= 0`): default builds (`NO_STDIO=0`, i.e. every
build before this flag existed) are byte-identical to before.

Only the `lib` goal is meant to be built with `NO_STDIO=1` — the CLI
(`aec_wav`) calls the `FILE*`-based debug API directly (`--debug-log` /
`--debug-trace`) and is not expected to compile under this macro. Board / MCU
images that link `libaec.a` without a hosted stdio should build with
`make lib NO_STDIO=1`. Verify with `make audit-no-stdio` (builds the
NO_STDIO=1 archive, `ar -t`s it for the excluded debug object, `nm -A`s it and
a minimal linked stdio-free consumer for stdio symbols, then runs that
consumer) — see the Makefile comment above that target for the full
calibration (the linked executable is fully stdio-free).

## Run

```bash
BIN="$(make -s print-bin-dir)"
"$BIN"/aec_wav mic.wav ref.wav out.wav --preset balanced --cng
```

Presets (single residual-echo strength axis `min_gain_floor_far_active_db`,
plus `no_floor`'s separate boolean gate `min_gain_split_floor_enabled`):
`mild` (−20, near-priority) / `balanced` (−28, production) /
`aggressive` (−38, echo-priority) / `no_floor` (floor mechanism off
entirely — `--preset 3` or `--preset no_floor`, deepest point on the axis,
most near-end loss). Other flags: `--cng`,
`--debug-level <n>` / `--debug-log <path>` (level-gated trace),
`--debug-trace <path>` (opt-in per-frame CSV of post-filter internals —
`usable_linear`, `fullband_erle`, `r2_mean`, `gain_mean`, `comfort_noise_mean`,
… one row per hop; zero hot-path cost when off).

## Streaming API

`aec_process(mic, ref, out)` is the lockstep 1-hop-in/1-hop-out call. For async
pipelines where render and capture arrive on separate calls/threads, v3.22.5 adds
a decoupled pair over a 320 ms render FIFO:

```c
aec_analyze_render(&a, ref);          /* buffer one render hop */
aec_process_capture(&a, mic, out);    /* consume + process one mic hop */
```

Lockstep (`analyze_render` then `process_capture`) is **byte-identical** to
`aec_process`; underrun/overrun return `AEC_BUF_RENDER_UNDERRUN`/`_OVERRUN`. See
[使用手冊 §8.4](../docs/c_user_manual_zh_TW.md) and `test/stream_sim.c`.

## Precision & regression anchors

**Float32 campaign (2026-07-15): Python bit-exact parity is retired
repo-wide.** All production C is now float32 end-to-end — delay chain,
orchestrator scalars, post/state modules, `residual_echo_estimator`, HPF. The Python reference (`python/aec.py`,
fp64) is now the **algorithm spec**; this C port is the float32
**implementation**; Python↔C comparison is **tolerance-based** (~−60 dB
class, correlation 0.99999958) — never 0/0.

The delay-estimator matched filter is the one piece that was already
float32/fast-math/duty-cycled *unconditionally* before the wider campaign
(intentional, sampled at zero AECMOS cost; see `include/delay_aec.h`).

The **FFT backend is NE10 (ARM NEON, float32)** — the FFT wrapper and the
NE10 backend live in the shared `audiolib` layer (a sibling repo); it is
the only backend (`make BACKEND=ne10`, the default), used by both the malloc
path and the caller-pool path (`aec_get_mem_size`/`aec_init`), and differs
from numpy's fp64 `np.fft` by float32 precision. The static path is byte-equal
to the malloc path (`test_static_aec.c`).

**Maintained regression anchors:**

- `test/parity_aec_e2e.c` — the **authoritative end-to-end gate**: full
  `aec_create` → `aec_process` output within a 2e-2 float32 tolerance
  (correlation 0.99999958, RMS ≈ −60 dB below signal, per-sample max ~6e-3
  over 4186 recursive hops — inaudible).
- Staged gates vs the `fp64-baseline` git tag: 60-case stratified AECMOS
  (worst per-case delta −0.021 echo, all bucket means ≤0.002), waveform drift
  median −95 dB, 1-hour soak (delay trajectory identical, power-EMA worst rel
  diff 1.3e-5, final ERLE matching to 4 digits).
- `test/test_counter_saturation.c` (`make test-counter-saturation`) — the
  **permanent counter-saturation regression test**: for
  every unbounded `struct_field += 1` / `-= 1` counter-overflow fix across
  every fixed counter-overflow path, proves cap-1→cap,
  cap→cap (no-op), thousands-of-calls-past-cap runaway-proof, and
  decision-invariance (the boolean/comparison the counter feeds is identical
  at the cap vs. a synthetic huge value standing in for the unbounded value
  the original bug would have produced). Replaces the ad hoc/throwaway UBSan
  probes each prior round wrote to verify these fixes.
  `test/run_counter_saturation_ubsan.sh` builds+runs this same source under
  `-fsanitize=undefined` (same shape as `test/run_selftest_ubsan.sh`).
- `test/test_rate_structural.c` (`make test-rate-structural`) — the
  **permanent per-rate structural regression test** (M5 multi-rate campaign,
  multi-rate contract): at each of the four whitelisted grids (8k/256, 16k/256,
  16k/512, 48k/1024) checks synthesis-window COLA, impulse-through-linear
  and impulse-through-full-post-chain convergence, `aec_get_mem_size`
  monotonicity/preset-invariance, `aec_init` (static pool) byte-equal to
  `aec_create` (heap), and — the top-level (non-AEC3) hop-authored constant
  audit's own regression gate — that `shadow_err_alpha`/`warmup_frames`/
  `epc_hangover`/`ne_recent_hold`/`filter_misadjustment_{stable,hangover}_
  frames`/`EPV_FAST_TC`/`EPV_SLOW_TC` cover approximately the same
  wall-clock duration at every grid. 107 checks total.
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

Module-level Python goldens: `python/diag/gen_detectors_golden.py`
(`make test-detectors-parity`) and `python/diag/gen_aec_e2e_golden.py`
(`make test-parity-e2e`) are the generators with a C checker in this tree;
the other `python/diag/gen_*_golden.py` scripts only dump Python reference
state, and no C replay harness is shipped for them.
See `docs/` for the numpy→C idioms (`np.abs(c64)**2` = scaled-hypot-FMA,
complex×complex FMA, EMA double-coeff) that motivated the original per-module
harness design.

## Static-memory (heap-free pool)

For embedded targets, build one pool and place the whole instance in it:

```c
size_t bytes = aec_get_mem_size(&cfg);   /* query at runtime; size is backend/config dependent */
void*  pool  = your_static_alloc(bytes); /* MUST be 16-byte aligned (posix_memalign, etc.) */
Aec*   a     = aec_init(pool, bytes, &cfg);  /* NULL on failure; byte-equal to aec_create output */
/* ... aec_process(a, ...) ... */
aec_destroy(a);   /* genuine no-op for pool instances; idempotent */
free(pool);
```

`aec_init` returns a pointer into `pool` (not an out-param) and is byte-equal
to the malloc path (`test_static_aec.c`, 0 mismatches). Both memory models
ship in the same library: `aec_create` (heap) and `aec_get_mem_size`/`aec_init`
(caller pool) are always compiled, selected at runtime — `USE_EXT_MEM` or
similar compile flags are not involved; which constructor you call is the only
switch. The three NE10 R2C/C2R twiddle configs (post-filter + main + shadow)
are carved from `pool` (vendored patch P0001, see
audiolib/common/lib/ne10/VENDORED.md), so the static path is strictly
heap-free — allocator-hook-verified by `test/test_zero_heap_aec.c` — and
`aec_destroy` is a genuine, idempotent no-op for pool instances.

Full CLI options, C API reference, integration rules, runtime resource
notes, and validation steps:
[../docs/c_user_manual_zh_TW.md](../docs/c_user_manual_zh_TW.md).
