# Development guide

This document records repository invariants, validation commands, and
maintenance conventions for anyone changing the implementation. Product users
should start from the root README or the C integration guide instead.

## What this repo is

Single-channel AEC (1 mic + 1 ref reference signal). Two implementations of the
**same algorithm** — Python is the fp64 algorithm spec, C is the float32
production implementation, and their outputs agree within a documented
numerical tolerance (not bit-equal):

- `python/aec.py` — top-level shim that re-exports every public symbol so
  `from aec import AEC, AecConfig, ...` works. The algorithm itself lives
  under `python/modules/`.
- `python/modules/` — algorithm modules organised into the AEC3-aligned
  chain (`state/`, `residual/`, `filter/`, `render/`, `delay/`) plus the
  shared infrastructure (`filters`, `epc`, `detectors`, `preprocessing`,
  `dataclasses`, `config`, `orchestrator`, `debug_logger`).
- `c_impl/` — production C port. Mirrors the Python class structure
  (`PBFDKF`, `AecState`, `SuppressionGain`, etc.). Built with
  `-ffp-contract=off -fno-math-errno` mandatory.

The version is `__version__` in [aec.py](../python/aec.py); the changelog is
[CHANGELOG.md](../CHANGELOG.md).

All production C is float32 end-to-end (delay chain, orchestrator scalars,
post/state modules, `residual_echo_estimator`, mic-path HPF). Python↔C
comparison is **tolerance-based**, never 0/0. The regression anchor is the
end-to-end tolerance gate `c_impl/test/parity_aec_e2e.c`.

The **FFT backend is NE10 (float32, ARM NEON)** for both the malloc build
(`make`) and the embedded deployment (caller pool via
`aec_get_mem_size`/`aec_init`); the static path is byte-equal to the malloc
path. Canonical algorithm reference: [aec_methods.md](aec_methods.md).

## Common commands

### Python single-case (algorithm dev)

```bash
# Standard run (single BALANCED preset)
python3 python/aec.py mic.wav ref.wav out.wav --mode pbfdkf --preset balanced --enable-res --cng

# Single-case with diagnostic 5-panel PNG (waveforms + spectrograms + ERLE)
python3 python/run_one_case.py mic.wav ref.wav out.wav --preset balanced

# Per-second console diagnostics
python3 python/aec.py mic.wav ref.wav out.wav --preset balanced --enable-res --diag
```

### Python 800-case benchmark (the standard bench)

```bash
# Render 800-case AEC Challenge corpus
python3 python/eval_aec_challenge.py wav/aec_challenge_blind/ \
    --preset balanced --filter 832 --cng --parallel -o out_python/ --workers 4

# AECMOS scoring (needs onnxruntime, librosa and soundfile)
python3 python/bench_aecmos.py out_python/ results/
python3 python/bench_aecmos.py out_python/ results/ --baseline /path/to/baseline_scores.json
```

**Standard 800-case config**: `preset=balanced / filter=832 (52 ms) / --cng /
--parallel / --workers 4`. This combination is the reference for every
regression check, A/B, and audit; other settings produce results that are not
comparable.

### Byte-equal regression check (behaviour-neutral changes)

A behaviour-neutral change must produce byte-identical output on the side of
the port it touches. **Python-side edits** use the Python render procedure
below. **C-side edits** are gated with C renders instead (`aec_wav` output +
`cmp`), since Python↔C is tolerance-based, not byte-equal.

**C-side gate** (render with `aec_wav` before/after, compare with `cmp`).
`aec_wav` lands in a config-hashed `bin/<backend>-<config-hash>/` directory;
resolve the exact path with `make print-bin-dir` (same flags as your build),
or use the stable `dist/<backend>/current/` path after `make publish`:

```bash
cd c_impl && make clean && make   # BEFORE editing
bin_before="$(make -s print-bin-dir)"
mkdir -p /tmp/cbe_before /tmp/cbe_after
for f in <stems>; do "$bin_before/aec_wav" "wav/${f}_mic.wav" "wav/${f}_lpb.wav" "/tmp/cbe_before/${f}.wav" --preset balanced --cng; done
# ... make edits ...
cd c_impl && make clean && make   # AFTER editing
bin_after="$(make -s print-bin-dir)"
for f in <stems>; do "$bin_after/aec_wav" "wav/${f}_mic.wav" "wav/${f}_lpb.wav" "/tmp/cbe_after/${f}.wav" --preset balanced --cng; done
for f in /tmp/cbe_after/*.wav; do \
  cmp -s "$f" "/tmp/cbe_before/$(basename "$f")" \
    && echo "MATCH $(basename "$f")" || echo "DIFFER $(basename "$f")"; done
```

Also re-run `test_static_aec` (static == dynamic) and `parity_aec_e2e`
(tolerance gate) — see `c_impl/STATIC_MEMORY.md`.

**Python-side gate** (render a small case set before and after the edit and
compare with `cmp`):

```bash
# BEFORE editing: render a baseline (any small dir of mic/lpb cases)
python3 python/eval_aec_challenge.py wav/<subset>/ --preset balanced \
    --filter 832 --cng --parallel -o /tmp/be_before/ --workers 4
# ... make edits ...
python3 python/eval_aec_challenge.py wav/<subset>/ --preset balanced \
    --filter 832 --cng --parallel -o /tmp/be_after/ --workers 4
# _ours.wav must all be byte-identical:
for f in /tmp/be_after/*_ours.wav; do \
  cmp -s "$f" "/tmp/be_before/$(basename "$f")" \
    && echo "MATCH $(basename "$f")" || echo "DIFFER $(basename "$f")"; done
```

All MATCH before any commit that touches code outside docs.

### C build & run

```bash
cd c_impl
make                  # debug: `make debug` (adds -g -DAEC_DEBUG)
# artifacts land in bin/<backend>-<config-hash>/;
# `make print-bin-dir` (same flags) resolves the exact path, or use
# `make publish` for a stable dist/<backend>/current/ path:
bin="$(make -s print-bin-dir)"
"$bin/aec_wav" mic.wav ref.wav out.wav --preset balanced --cng
"$bin/aec_wav" mic.wav ref.wav out.wav --debug-trace /tmp/aec_trace.csv
```

`-ffp-contract=off -fno-math-errno` in `CFLAGS` is mandatory for build
determinism (no FMA reassociation drift between compilers/builds). It is a
policy shared by `audiolib`, `NR/c_impl`, `AEC/c_impl` and `Audio_ALG/pipelines`;
each Makefile appends the flags last and rejects overrides. See
`c_impl/README.md`'s "FP-contraction policy" section and
`audiolib/common/scripts/audit_fp_contract.sh` (the disassembly-level audit).
Output WAV defaults to fp32 PCM (`AEC_OUT_FLOAT=0` for 16-bit).

### Python tests

```bash
python3 -m pytest python/tests
```

### C tests

```bash
cd c_impl
make test SIMD=1      # default tier; SIMD-independent members run only under SIMD=1
make test SIMD=0
make test-full        # + full Python<->C parity matrix, end-to-end gate, slow script tests
```

`make test` also runs the pytest suite and the lean parity suite
(`make test-pyc-parity`; `PYC_ARGS=--full` for every scenario). The tiers and
their members are listed in `c_impl/README.md`.

## Architecture — what to read before changing things

### Pipeline

```
mic ─► HPF ──────────────────────────────────────────────────────────►
ref ─────► Saturation ─► DelayEst+ref ring ─► PBFDKF ─► error ─► AEC3 post ─► out
                                               │                       │
                                       Shadow filter (PBFDAF/NLMS) AecState + ResidualEchoEstimator
                                       + PathChangeRegimeHandler + SuppressionGain + CNG (OLA)
```

HPF runs on the mic path only; `Saturation` / `EchoPathDelayEstimator` consume
the raw reference.

The post-filter is `AEC._aec3_post` in
[python/modules/orchestrator.py](../python/modules/orchestrator.py), which
drives the AEC3-aligned chain:

  modules/state         — AecState ADT + StationarityEstimator +
                          SubbandErleEstimator (the AEC3 read-only seam
                          for downstream consumers)
  modules/residual      — ResidualEchoEstimator + SuppressionGain
                          (per-bin echo PSD estimate + Wiener gain)
  modules/filter        — FilterStateBridge (per-frame snapshot of the refined
                          filter consumed by AecState)
  modules/render        — RenderSignalAnalyzer (per-bin tonal narrowband
                          mask + poor_signal_excitation gate)
  modules/delay         — EchoPathDelayEstimator + LegacyDelayShim
                          (the `accumulate()` API the orchestrator expects)

`enable_res` gates the post-filter; running with `--enable-res 0` emits
the linear residual at PBFDKF output (used by
`eval_aec_challenge.py`'s `_ours_nores.wav` companion render).

Tight coupling lives in **`PBFDKF` + shadow `PBFDAF` + `PathChangeRegimeHandler`**
— the shadow filter and main filter exchange state via a regime handler
that fires `boost_q` / `reverse_copy` / `main_paused` decisions. PBFDKF
lives in [python/modules/filters.py](../python/modules/filters.py);
PathChangeRegimeHandler in [python/modules/epc.py](../python/modules/epc.py).
The handler is **load-bearing on the cohort tail**; do not remove or bypass it.

### `AecConfig` and presets

Four presets — `mild` / `balanced` / `aggressive` / `no_floor` — are defined in
[python/modules/config.py](../python/modules/config.py) (`from_preset`). All share
the same AEC3 chain and the four base overrides (`enable_cng`,
`shadow_mu_min`, `warmup_frames`, `kalman_q_high`); everything else uses dataclass
defaults. **Knobs are co-tuned** — don't tweak a single field without a full
800-case re-bench.

`balanced` is the production preset. `mild` and `aggressive` are deliberate
**Pareto operating points** on a single residual-echo strength knob —
`min_gain_floor_far_active_db`, the far-active min-gain floor (mild −20 /
balanced −28 / aggressive −38 dB); `aggressive` also deepens
`min_gain_floor_dt_db` to −30 dB, and `no_floor` turns the floor mechanism off
(`min_gain_split_floor_enabled=False`):

- `mild` = near-priority (higher floor → more near-end kept, more echo leak).
- `aggressive` = echo-priority (deeper floor → more echo killed, more near loss).

The DT-deg-vs-echo trade is a single-channel DSP Pareto wall; the strength axis
exposes it rather than hiding it.

### Diagnostic surfaces (do not remove)

- `AecStats` / `get_stats()` ([python/modules/dataclasses.py](../python/modules/dataclasses.py)
  + the AEC method in [python/modules/orchestrator.py](../python/modules/orchestrator.py))
  — per-frame audio-passive trace consumed by `run_one_case.py` plots
  and external research tooling.
- `AecResContext` — exposes `echo_spec` / per-bin Kalman state so the
  linear stage can feed an external (or NN) post-filter;
  `AecConfig.return_res_context = True` switches `aec.process()` return
  type to `(out, AecResContext)`.

## Conventions

- Audio dataset (800 cases): `wav/aec_challenge_blind/{doubletalk,farend_singletalk,nearend_singletalk}/<stem>_{mic,lpb}.wav`.
- CNG determinism: comfort noise uses an internal LCG seeded with 42 at construction and reset (Python and C alike); the algorithm consumes no global RNG state.
- HPF defaults: far-end (ref) HPF=OFF, mic-path HPF=ON.
- macOS: use `python3` (not `python`).
- `active_render` threshold is **5.96e-4** (100² · 64 / 32768²).

## Branch model

`main` carries the production-graded code. Version history lives in
[CHANGELOG.md](../CHANGELOG.md).
