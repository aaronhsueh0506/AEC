# AEC C Implementation

C implementation of the AEC algorithm. Top-level orchestration in
`aec.{h,c}`, CLI binary `bin/aec_wav`.

> **User & integration guide** → [../docs/c_user_and_integration_guide.md](../docs/c_user_and_integration_guide.md)
> Algorithm reference → [../docs/aec_methods.md](../docs/aec_methods.md)
> Changelog → [../docs/CHANGELOG.md](../docs/CHANGELOG.md)

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

## Parity (bit-exact to Python)

The C port is **byte-for-byte identical** to `python/aec.py` end-to-end:
per-hop golden 0 mismatches over the full doubletalk case (linear residual +
final output), all three presets; full CLI `wav→wav` 0/669920 fp32 sample
mismatches. Each sub-module has a standalone golden test
(`test/parity_*.c` ⟷ `python/diag/gen_*_golden.py`); the FFT is numpy's
vendored pocketfft (`lib/pocketfft/`). End-to-end gate:
`test/parity_aec_e2e.c`. See
[../memory parity rules] and `docs/` for the numpy→C bit-exact idioms
(`np.abs(c64)**2` = scaled-hypot-FMA, complex×complex FMA, EMA double-coeff).

Full CLI options, C API reference, integration rules, runtime resource
notes, and validation steps:
[../docs/c_user_and_integration_guide.md](../docs/c_user_and_integration_guide.md).
