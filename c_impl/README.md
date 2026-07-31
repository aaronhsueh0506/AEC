# AEC C Implementation

C implementation of the AEC algorithm. Top-level orchestration in
`aec.{h,c}`, CLI binary `bin/aec_wav`.

> **User & integration guide** → [../docs/c_user_and_integration_guide.md](../docs/c_user_and_integration_guide.md)
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
make clean
```

Artifacts land in a config-hashed `bin/<backend>-<config-hash>/` directory
(switching `BACKEND`/`SIMD`/`EXTRA_CFLAGS`/`WERROR` always lands in a fresh one
automatically — no stale-object risk, no manual clean needed). Run `make
print-bin-dir` (same flags as your build) to get the exact path, or `make
publish` to copy this build's artifacts to a stable `dist/<backend>/current/`
handoff path.

Compile flags (already in Makefile): `-O2 -I include -I example`, plus
`-ffp-contract=off` appended **last** (round-3 review B04 — see "Unified
FP-contraction policy" below). `-ffp-contract=off` is **required** (no FMA
fusion — load-bearing for build determinism and golden stability across
builds/compilers; see Precision & regression anchors below).

### Unified FP-contraction policy (round-3 review B04)

`-ffp-contract=off` is no longer just an AEC convention — it is a **repo-wide
policy spanning all four repos** (`audio_common`, `NR/c_impl`, `AEC/c_impl`,
`Audio_ALG/pipelines`): every translation unit any of their Makefiles compile
— each repo's own sources *and* the vendored KISS/NE10 C and C++ TUs alike —
builds with this flag. In every one of the four Makefiles the flag is
appended **last** in the CFLAGS/CXXFLAGS assembly (after `EXTRA_CFLAGS`, after
any BACKEND-conditional append, after `WERROR`/`NO_STDIO`), so nothing a
caller passes can land after it and override it — AEC's Makefile used to
carry the flag as the *third* token of the base CFLAGS assignment (before
`EXTRA_CFLAGS` was folded in), which this review moved to its current
trailing position. Each Makefile also rejects outright, at parse time, an
`EXTRA_CFLAGS` (or `CFLAGS=` override) containing `-Ofast`, `-ffast-math`, or
`-ffp-contract=<anything>` (all of which would re-enable contraction), e.g.:

```
$ make EXTRA_CFLAGS=-ffast-math
Makefile:111: *** FP policy conflict: CFLAGS/EXTRA_CFLAGS contains -ffast-math; this repo pins -ffp-contract=off; remove -ffast-math from EXTRA_CFLAGS.  Stop.
```

Round-4 hardening: a command-line `CFLAGS=`/`CXXFLAGS=`/`CPPFLAGS=`/
`LDFLAGS=`/`FP_POLICY=` override is now rejected outright in all four
Makefiles — GNU Make silently ignores a Makefile's own `+=`/`:=` assignments
to a command-line-set variable, so `make CFLAGS=-O3` used to strip
`-ffp-contract=off` (and `-DAEC_NO_STDIO`, the `-I` paths, `-lm`) while
still building. `EXTRA_CFLAGS`/`EXTRA_LDFLAGS` are the two supported hooks:

```
$ make CFLAGS=-O3
Makefile:119: *** CFLAGS cannot be overridden (origin: command line) -- it would silently drop this Makefile's own flag appends (FP policy, NO_STDIO, include paths); use EXTRA_CFLAGS / EXTRA_LDFLAGS instead.  Stop.
```

`audio_common/scripts/audit_fp_contract.sh` is the disassembly-level proof
the flag actually bites: it disassembles a fixed list of TUs expected to be
genuinely scalar (audio_common's `hpf.o`/`kiss_fft.o`/NE10's scalar-C
objects, NR's three core objects) and fails if any fmadd/fmsub/fnmadd/
fnmsub/fmla/fmls instruction shows up. AEC's own `aec_simd_kernels.h` — with
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
calibration (on this host, `BACKEND=ne10` links a fully stdio-free
executable; `BACKEND=kiss` retains a small, out-of-scope residual from
audio_common's vendored KISS FFT's own error-log macro).

## Run

```bash
./bin/aec_wav mic.wav ref.wav out.wav --preset balanced --cng
```

Presets (single residual-echo strength axis `min_gain_floor_far_active_db`):
`mild` (−20, near-priority) / `balanced` (−28, production) /
`aggressive` (−38, echo-priority). Other flags: `--cng`,
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
[guide §10.1.1](../docs/c_user_and_integration_guide.md) and `test/stream_sim.c`.

## Precision & regression anchors

**Float32 campaign (2026-07-15): Python bit-exact parity is retired
repo-wide.** All production C is now float32 end-to-end — delay chain,
orchestrator scalars, post/state modules, `residual_echo_estimator`, HPF
(`reverb_decay_estimator.c` is the sole remaining `double` file, and it is
dead code with no production caller). The Python reference (`python/aec.py`,
fp64) is now the **algorithm spec**; this C port is the float32
**implementation**; Python↔C comparison is **tolerance-based** (~−60 dB
class, correlation 0.99999958) — never 0/0.

The delay-estimator matched filter is the one piece that was already
float32/fast-math/duty-cycled *unconditionally* before the wider campaign
(intentional, sampled at zero AECMOS cost; see `include/delay_aec3.h`).

The production **FFT backend is KISS FFT (float32)** — the FFT wrapper, KISS
FFT, and NE10 (ARM NEON) backend now live in the shared `audio_common` layer
(a sibling repo); select one via `make BACKEND=kiss` (default, host/reference
build, malloc) or `make BACKEND=ne10` (embedded build, caller pool via
`aec_get_mem_size`/`aec_init`) — which differs from numpy's fp64 `np.fft` by
float32 precision. NE10 vs KISS output is not bit-identical to each other
(pre-existing, expected); each backend's static path is byte-equal to its own
malloc path (`test_static_aec.c`).

**Maintained regression anchors:**

- `test/parity_delay.c` — a **C-regression golden**, regenerated by
  `test/gen_delay_c_golden.c` (checks the delay chain against its own prior
  output, not against Python).
- `test/parity_aec_e2e.c` — the **authoritative end-to-end gate**: full
  `aec_create` → `aec_process` output within a 2e-2 float32 tolerance
  (correlation 0.99999958, RMS ≈ −60 dB below signal, per-sample max ~6e-3
  over 4186 recursive hops — inaudible).
- Staged gates vs the `fp64-baseline` git tag: 60-case stratified AECMOS
  (worst per-case delta −0.021 echo, all bucket means ≤0.002), waveform drift
  median −95 dB, 1-hour soak (delay trajectory identical, power-EMA worst rel
  diff 1.3e-5, final ERLE matching to 4 digits).
- `test/test_counter_saturation.c` (`make test-counter-saturation`) — the
  **permanent counter-saturation regression test** (round-6 review): for
  every unbounded `struct_field += 1` / `-= 1` counter-overflow fix across
  rounds 4-6 (plus the round-6 sweep's own new finds), proves cap-1→cap,
  cap→cap (no-op), thousands-of-calls-past-cap runaway-proof, and
  decision-invariance (the boolean/comparison the counter feeds is identical
  at the cap vs. a synthetic huge value standing in for the unbounded value
  the original bug would have produced). `FilterPlateauDetector`'s trio
  (widened to `int64_t` rather than capped, to avoid corrupting the
  far_ratio/dt_ratio they feed) gets a dedicated ratio-preservation +
  no-wraparound-past-`INT32_MAX` case. Replaces the ad hoc/throwaway UBSan
  probes each prior round wrote to verify these fixes.
  `test/run_counter_saturation_ubsan.sh` builds+runs this same source under
  `-fsanitize=undefined` (same shape as `test/run_selftest_ubsan.sh`).

The remaining per-module `test/parity_*.c` ⟷ `python/diag/gen_*_golden.py`
harnesses are kept as **historical** diagnostics — they compare against fp64
Python goldens predating the campaign and may report drift by design now
that the modules they exercise are float32; some also read f64 golden
scalars into f32 configs via silent narrowing. See
[`test/PARITY_REPORT.md`](test/PARITY_REPORT.md) for the full disposition.
See `docs/` for the numpy→C idioms (`np.abs(c64)**2` = scaled-hypot-FMA,
complex×complex FMA, EMA double-coeff) that motivated the original per-module
harness design.

## Static-memory (heap-free pool)

For embedded targets, build one pool and place the whole instance in it:

```c
size_t bytes = aec_get_mem_size(&cfg);   /* 16k default 512/256 balanced: 543,040 B on current KISS build */
void*  pool  = your_static_alloc(bytes); /* MUST be 16-byte aligned (posix_memalign, etc.) */
Aec*   a     = aec_init(pool, bytes, &cfg);  /* NULL on failure; byte-equal to aec_create output */
/* ... aec_process(a, ...) ... */
aec_destroy(a);   /* genuine no-op for pool instances (both backends); idempotent */
free(pool);
```

`aec_init` returns a pointer into `pool` (not an out-param) and is byte-equal
to the malloc path (`test_static_aec.c`, 0 mismatches). With the KISS FFT
backend the FFT is now **fully heap-free too** — the kiss configs are placed
in the caller pool via `kiss_fft_alloc`'s mem/lenmem API (unlike the old
pocketfft plans, which had to stay on the heap), so the static path makes no
heap allocation at all and `aec_destroy` is a genuine no-op on this backend.
Both memory models ship in the same library: `aec_create` (heap) and
`aec_get_mem_size`/`aec_init` (caller pool) are always compiled, selected at
runtime — `USE_EXT_MEM` or similar compile flags are not involved; which
constructor you call is the only switch. **Under NE10** the three R2C/C2R
twiddle configs (post-filter + main + shadow) are carved from `pool` too
(vendored patch P0001, see audio_common/lib/ne10/VENDORED.md), so the static
path is strictly heap-free on both backends — allocator-hook-verified by
`test/test_zero_heap_aec.c` — and `aec_destroy` is a genuine, idempotent
no-op for pool instances on either backend.

Full CLI options, C API reference, integration rules, runtime resource
notes, and validation steps:
[../docs/c_user_and_integration_guide.md](../docs/c_user_and_integration_guide.md).
