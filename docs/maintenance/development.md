# Development guide

This document records repository invariants, validation commands, and
maintenance conventions for anyone changing the implementation. Product users
should start from the root README or the C integration guide instead.

## What this repo is

AEC for one microphone and one reference signal. The C library also builds a
four-microphone array instance (`AecConfig.n_mics = 4`: four lanes, one shared delay
estimator, fusion buffers; the beamformer stays with the caller); the Python reference
implements the single-channel algorithm only. Two implementations of the
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

The version is `__version__` in [aec.py](../../python/aec.py); the changelog is
[CHANGELOG.md](../../CHANGELOG.md).

All production C is float32 end-to-end (delay chain, orchestrator scalars,
post/state modules, `residual_echo_estimator`, mic-path HPF). Python↔C
comparison is **tolerance-based**, never 0/0. The regression anchor is the
end-to-end tolerance gate `c_impl/test/parity_aec_e2e.c`.

The **FFT backend is NE10 (float32, ARM NEON)** for both the malloc build
(`make`) and the embedded deployment (caller pool via
`aec_get_mem_size`/`aec_init`); the static path is byte-equal to the malloc
path. Canonical algorithm reference: [aec_methods.md](../aec_methods.md).

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

### Fixed-manifest blind A/B (baseline vs candidate worktrees)

`eval/run_py_ab.sh` (Python path, `NO_PREALIGN=1`) and `eval/run_c_ab.sh` (C path, `aec_wav`) render and score the stems in
`eval/manifest_90case_stems.txt`; `eval/ab_report.py` summarises. A change that lands in both ports needs both runners: neither
sees the other's regressions.

The manifest holds 61 cases (30 far-end, 30 doubletalk, 1 near-end; every clip is official AEC Challenge data, as the evaluation
rule requires). The file names keep the original `90case` label: the set started as 90 cases and lost the 29 near-end stems whose
clips are not official. The corpus under `wav/aec_challenge_blind/` is git-ignored local data. `eval/manifest_pin.py prepare`
resolves each stem to its scenario directory and nowhere else, so the frozen set is compared unchanged (do not regenerate it:
`build_manifest_90case.py` run today selects a different set). It stops naming every stem it cannot find, and verifies
`eval/manifest_90case.sha256` (sha256 of the 122 clips by scenario-relative name, of the stems list and of the manifest JSON), so
drift of the corpus or the manifest aborts the run with the list of files. The resolved clips are linked into `$OUT/dataset_view`,
which the renderer and `bench_aecmos.py --dataset` read; `python/eval_manifest90.py` applies the same resolution and check. After
an intended corpus change, re-pin with `python3 eval/manifest_pin.py write --dataset wav/aec_challenge_blind`. `SELF_CHECK=1
eval/run_c_ab.sh` accepts two byte-identical builds: a null run in which every delta must be exactly 0. Run the scripts from a scratch copy of the tree, not from inside the repository.

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
work="$(mktemp -d)"                 # scratch dir for the two renders; delete it when done
cd c_impl && make clean && make   # BEFORE editing
bin_before="$(make -s print-bin-dir)"
mkdir -p "$work/cbe_before" "$work/cbe_after"
for f in <stems>; do "$bin_before/aec_wav" "wav/${f}_mic.wav" "wav/${f}_lpb.wav" "$work/cbe_before/${f}.wav" --level 1 --cng; done
# ... make edits ...
cd c_impl && make clean && make   # AFTER editing
bin_after="$(make -s print-bin-dir)"
for f in <stems>; do "$bin_after/aec_wav" "wav/${f}_mic.wav" "wav/${f}_lpb.wav" "$work/cbe_after/${f}.wav" --level 1 --cng; done
for f in "$work"/cbe_after/*.wav; do \
  cmp -s "$f" "$work/cbe_before/$(basename "$f")" \
    && echo "MATCH $(basename "$f")" || echo "DIFFER $(basename "$f")"; done
```

Also re-run `test_static_aec` (static == dynamic) and `parity_aec_e2e`
(tolerance gate) — see `c_impl/STATIC_MEMORY.md`.

**Python-side gate** (render a small case set before and after the edit and
compare with `cmp`):

```bash
# BEFORE editing: render a baseline (any small dir of mic/lpb cases)
work="$(mktemp -d)"                 # scratch dir; delete it when done
python3 python/eval_aec_challenge.py wav/<subset>/ --preset balanced \
    --filter 832 --cng --parallel -o "$work/be_before/" --workers 4
# ... make edits ...
python3 python/eval_aec_challenge.py wav/<subset>/ --preset balanced \
    --filter 832 --cng --parallel -o "$work/be_after/" --workers 4
# _ours.wav must all be byte-identical:
for f in "$work"/be_after/*_ours.wav; do \
  cmp -s "$f" "$work/be_before/$(basename "$f")" \
    && echo "MATCH $(basename "$f")" || echo "DIFFER $(basename "$f")"; done
```

All MATCH before any commit that touches code outside docs.

### C build & run

```bash
cd c_impl
make                  # debug: `make debug` (adds -g)
# artifacts land in bin/<backend>-<config-hash>/;
# `make print-bin-dir` (same flags) resolves the exact path, or use
# `make publish` for a stable dist/<backend>/current/ path:
bin="$(make -s print-bin-dir)"
"$bin/aec_wav" mic.wav ref.wav out.wav --level 1 --cng
"$bin/aec_wav" mic.wav ref.wav out.wav --debug
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

Two of these read documentation: `python/tests/test_c_python_map.py` (the marker, count and statistics checks of
`docs/c_python_function_map.md`) and `python/tests/test_timing_inventory.py` (`docs/timing_constant_inventory.json`
and the `.md` rendered from it by `python/diag/gen_timing_inventory.py`). `make test-docs-smoke` (in `c_impl/`) compiles and
runs the C sample between the `docs-smoke` markers of `c_impl/STATIC_MEMORY.md`. Run them after a documentation change.

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

### Behaviour-neutral C changes: the line-stream baseline

No golden directory ships in `c_impl/test/`. The four hash-stream tests (`test-split-entry`, `test-split-entry-2`,
`test-aec-array`, `test-aec-array-2`) capture and compare their own output instead, run from `c_impl/`:

```bash
out="$(mktemp -d)"                                    # scratch directory outside the repository; delete it when done
AEC_TEST_BASELINE_WRITE="$out" make test-split-entry  # BEFORE editing: write <test>_simd{0,1}.txt for this build
# ... make the change ...
AEC_TEST_BASELINE="$out" make test-split-entry        # AFTER editing: every line must match
```

Do this for each of the four tests, at `SIMD=1` and `SIMD=0`. A pool-size change shows up only in the `META` lines; every hop hash must stay
identical. The files are valid for the host, compiler and FFT backend that wrote them. The Python-side goldens
(`make test-detectors-parity`, `make test-parity-e2e`) are regenerated from the Python reference on every run and are not checked in.

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
[python/modules/orchestrator.py](../../python/modules/orchestrator.py), which
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
`eval_aec_challenge.py`'s `_ours_nores.wav` companion render). (Python twin; the C `AecConfig` has the single
`residual` field instead, `AEC_RESIDUAL_OFF` being this bypass.)

Tight coupling lives in **`PBFDKF` + shadow `PBFDAF` + `PathChangeRegimeHandler`**
— the shadow filter and main filter exchange state via a regime handler
that fires `boost_q` / `reverse_copy` / `main_paused` decisions. PBFDKF
lives in [python/modules/filters.py](../../python/modules/filters.py);
PathChangeRegimeHandler in [python/modules/epc.py](../../python/modules/epc.py).
The handler is **load-bearing on the cohort tail**; do not remove or bypass it.

### `AecConfig` and presets

(Python twin. The C `AecConfig` has 12 fields and one strength field, `level`; everything below that the Python dataclass
exposes is a library-internal `AecTuning` value in C. See `docs/reference/configuration.md` §6.)

Four presets — `mild` / `balanced` / `aggressive` / `no_floor` — are defined in
[python/modules/config.py](../../python/modules/config.py) (`from_preset`). All share
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

- `AecStats` / `get_stats()` ([python/modules/dataclasses.py](../../python/modules/dataclasses.py)
  + the AEC method in [python/modules/orchestrator.py](../../python/modules/orchestrator.py))
  — per-frame audio-passive trace consumed by `run_one_case.py` plots
  and external research tooling.
- `AecResContext` — exposes `echo_spec` / per-bin Kalman state so the
  linear stage can feed an external (or NN) post-filter;
  `AecConfig.return_res_context = True` switches `aec.process()` return
  type to `(out, AecResContext)`. (Python twin; the C side exposes the same data through `aec_get_linear()` /
  `aec_get_residual()`, see `docs/reference/integration.md` §8.3.)

## Conventions

- Audio dataset (800 cases): `wav/aec_challenge_blind/{doubletalk,farend_singletalk,nearend_singletalk}/<stem>_{mic,lpb}.wav`.
- CNG determinism: comfort noise uses an internal LCG seeded with 42 at construction and reset (Python and C alike); the algorithm consumes no global RNG state.
- HPF defaults: far-end (ref) HPF=OFF, mic-path HPF=ON.
- macOS: use `python3` (not `python`).
- `active_render` threshold is **5.96e-4** (100² · 64 / 32768²).
