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
-I example`. `-ffp-contract=off` is required.

Full CLI options, C API reference, integration rules, runtime resource
notes, and validation steps:
[../docs/c_user_and_integration_guide.md](../docs/c_user_and_integration_guide.md).
