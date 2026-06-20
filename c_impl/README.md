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
make            # → bin/aec_wav (CLI binary)
make lib        # → bin/libaec.a (static library)
make clean
```

Compile flags (already in Makefile): `-O2 -ffp-contract=off -I include
-I example`. `-ffp-contract=off` is **required** (no FMA fusion — load-bearing
for bit-exact parity with the Python reference; see Parity below).

## Run

```bash
./bin/aec_wav mic.wav ref.wav out.wav --preset balanced --cng
```

Presets (single residual-echo strength axis `min_gain_floor_far_active_db`):
`gentle` (−20, near-priority) / `balanced` (−28, production) /
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

## Parity (float32-precision to Python)

The C port's **non-FFT logic is bit-exact** to `python/aec.py` (verified at
v3.23.0 under `-DUSE_STANDARD_MATH` with a numpy-precision FFT). The production
**FFT backend is KISS FFT (float32)** — vendored `lib/kiss_fft/`, with NE10
(ARM NEON) opt-in via `make NE10_DIR=...` — which differs from numpy's fp64
`np.fft` by float32 precision. So the **end-to-end C output aligns with Python
to ~float32 precision, NOT 0/0**: correlation 0.99999958, RMS error ≈ −60 dB
below signal, per-sample max ~6e-3 over 4186 recursive hops (inaudible). The
default `fast_math.h` (approximate `exp`/`sqrt`) adds a further ~1e-5..1e-4 in
the RES stages. Each sub-module has a standalone golden test
(`test/parity_*.c` ⟷ `python/diag/gen_*_golden.py`). The FFT-path module tests'
strict 0/0 checks (written against the now-removed pocketfft backend) have been
converted to the KISS float32 reality: non-FFT and integer/boolean control state
stays asserted **bit-exact**, while FFT-derived values are gated within a small
tolerance (`parity_fft`/`parity_linear_filter_select`/`parity_filter_state_bridge`/
`parity_aec3_post` 1e-4; recursive outputs `parity_pbfdkf`/`parity_aec3_post_run`
2e-2). The one exception is `parity_pbfdkf`'s internal Kalman state (W/H_error/…),
which diverges chaotically through the recursive loop and is reported
diagnostic-only — the gated quantities there are the integer control state and
the linear output. The **authoritative end-to-end gate** is
`test/parity_aec_e2e.c` (output within 2e-2 float32 tolerance).
See [../memory parity rules] and
`docs/` for the numpy→C idioms (`np.abs(c64)**2` = scaled-hypot-FMA,
complex×complex FMA, EMA double-coeff).

## Static-memory (heap-free pool)

For embedded targets, build one pool and place the whole instance in it:

```c
size_t bytes = aec_get_mem_size(&cfg);     /* balanced: 525760 B @ hop=160 */
void*  pool  = your_static_alloc(bytes);   /* 16-byte aligned */
Aec a; aec_init(&a, pool, bytes, &cfg);    /* byte-equal to aec_create */
```

`aec_init` is byte-equal to the malloc path (`test_static_aec.c`, 0 mismatches).
With the KISS FFT backend the FFT is now **fully heap-free too** — the kiss
configs are placed in the caller pool via `kiss_fft_alloc`'s mem/lenmem API
(unlike the old pocketfft plans, which had to stay on the heap), so the static
path makes no heap allocation at all. The static-memory variant lives on the
`feature/static-memory` branch; `main` uses malloc.

Full CLI options, C API reference, integration rules, runtime resource
notes, and validation steps:
[../docs/c_user_and_integration_guide.md](../docs/c_user_and_integration_guide.md).
