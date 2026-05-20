# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this repo is

Single-channel AEC (1 mic + 1 ref reference signal). Two implementations of the
**same algorithm** that must produce **bit-equal output** within numerical
tolerance:

- `python/aec.py` — top-level shim that re-exports every public symbol so
  `from aec import AEC, AecConfig, ...` keeps working. Algorithm itself
  lives under `python/modules/`.
- `python/modules/` — algorithm modules organised into the AEC3-aligned
  chain (`state/`, `residual/`, `filter/`, `render/`, `delay/`) plus the
  legacy shared infrastructure (`filters`, `epc`, `detectors`, `dtd`,
  `preprocessing`, `nlp`, `erle`, `dataclasses`, `config`, `orchestrator`,
  `debug_logger`, `residual_estimator`). Module-by-module map:
  [docs/refactor_modules_layout.md](docs/refactor_modules_layout.md).
- `c_impl/` — production C port. Mirrors the Python class structure
  (`PBFDKF`, `ShadowFilter`, etc.). Built with `-ffp-contract=off` mandatory.

Algorithm version is tracked by `__version__` in [aec.py](python/aec.py)
(currently **3.21.4**). Canonical algorithm reference:
[docs/aec_methods.md](docs/aec_methods.md). Trace-driven evolution
history: [docs/aec_v3_evolution.md](docs/aec_v3_evolution.md). Research
log canonical: [docs/SUMMARY.md](docs/SUMMARY.md).

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

# AECMOS scoring (needs speechmos + onnxruntime ≤1.16.3 + numpy<2 in venv)
python3 python/bench_aecmos.py out_python/ results/
python3 python/bench_aecmos.py out_python/ results/ --baseline /path/to/baseline_scores.json
```

**Standard 800-case config**: `preset=balanced / filter=832 (52 ms) / --cng /
--parallel / --workers 4`. This combination is the reference for every
regression check, A/B, and audit. Deviating from it produces results that
aren't comparable to prior verdicts.

### Byte-equal regression harness (post-cleanup gate)

```bash
# Sample 25 cases (5 per bucket at echo percentiles 0/25/50/75/100), render,
# md5 _ours.wav and _ours_nores.wav, compare to v3.21 anchor.
python3 python/check_byte_equal.py
# Must report `=== 25/25 PASS, 0 FAIL ===` before any commit that touches
# code outside docs.
```

Reference at [docs/bench/v3_21_3aadd2d_baseline/](docs/bench/v3_21_3aadd2d_baseline/).

### C build & run

```bash
cd c_impl
make                  # debug: `make debug` (adds -g -DAEC_DEBUG)
./bin/aec_wav mic.wav ref.wav out.wav --preset balanced --cng
./bin/aec_wav mic.wav ref.wav out.wav --debug-level 2 --debug-log /tmp/aec.log
```

`-ffp-contract=off` in `CFLAGS` is mandatory for Python↔C byte-equal parity.
Output WAV defaults to fp32 PCM (`AEC_FP32_WAV=0` for 16-bit).

### Python tests

```bash
python3 -m pytest python/test_p52_regime.py   # P52 classifier unit tests
```

There is no project-wide pytest collection.

## Architecture — what to read before changing things

### Pipeline (v3.21)

```
mic ─► HPF ──────────────────────────────────────────────────────────►
ref ─────► Saturation ─► DelayEst+RingBuf ─► PBFDKF ─► error ─► AEC3 post ─► out
                                               │                       │
                                       Shadow filter (Q×3.5)   AecState + ResidualEchoEstimator
                                       + PathChangeRegimeHandler + SuppressionGain + CNG (OLA)
```

HPF runs on the mic path only. The ref-path HPF was retired (default
OFF) after the v3.19 ref-flip verdict; downstream `Saturation` /
`EchoPathDelayEstimator` consume the raw reference.

The v3.21 pipeline retires the legacy 9-stage `ResFilter` chain. Its
replacement is `AEC._aec3_post` in [python/modules/orchestrator.py](python/modules/orchestrator.py),
which drives the AEC3-aligned post-filter:

  modules/state         — AecState ADT + StationarityEstimator +
                          SubbandErleEstimator (the AEC3 read-only seam
                          for downstream consumers)
  modules/residual      — ResidualEchoEstimator + SuppressionGain
                          (per-bin echo PSD estimate + Wiener gain)
  modules/filter        — refined filter substrate (incl. RenderSignalAnalyzer
                          + filter_quality + filter_analyzer audit ports)
  modules/render        — RenderSignalAnalyzer (per-bin tonal narrowband
                          mask + poor_signal_excitation gate)
  modules/delay         — EchoPathDelayEstimator + LegacyDelayShim
                          (AEC3 EchoPathDelayEstimator with the legacy
                          `accumulate()` API the orchestrator expects)

`enable_res` gates the post-filter; running with `--enable-res 0` emits
the linear residual at PBFDKF output (used by
`eval_aec_challenge.py`'s `_ours_nores.wav` companion render).

Tight coupling lives in **`PBFDKF` + `ShadowFilter` + `PathChangeRegimeHandler`**
— the shadow filter and main filter exchange state via a regime handler
that fires `boost_q` / `reverse_copy` / `main_paused` decisions. PBFDKF
lives in [python/modules/filters.py](python/modules/filters.py);
PathChangeRegimeHandler in [python/modules/epc.py](python/modules/epc.py).
**Was previously named `ShadowCopyController`** (renamed under P52
Path 3 of the v3.10.6 cycle — kept the audio path identical, added
the regime-classifier anti-loophole test). The handler is
**load-bearing on the cohort tail** (~7/800 cases); do not remove or
bypass it.

### `AecConfig` and presets

Single production preset: `BALANCED`. Defined in
[python/modules/config.py](python/modules/config.py) (`from_preset`).
The five 800-case AECMOS-tuned overrides are `enable_cng`,
`shadow_q_ratio`, `shadow_mu_min`, `warmup_frames`, `kalman_q_high`;
everything else uses dataclass defaults. **Knobs are co-tuned** — don't
tweak a single field without a full 800-case re-bench.

### Diagnostic surfaces (do not remove)

- `AecStats` / `get_stats()` ([python/modules/dataclasses.py](python/modules/dataclasses.py)
  + the AEC method in [python/modules/orchestrator.py](python/modules/orchestrator.py))
  — per-frame audio-passive trace consumed by `run_one_case.py` plots
  and external research tooling.
- `AecResContext` — exposes `echo_spec` / per-bin Kalman state so the
  linear stage can feed an external (or NN) post-filter;
  `AecConfig.return_res_context = True` switches `aec.process()` return
  type to `(out, AecResContext)`.
- `trace_p52_regime_handler` flag (default-OFF) — per-frame regime
  handler trace; classifier in
  [python/modules/p52_regime_classifier.py](python/modules/p52_regime_classifier.py)
  is analysis-only (enforced by
  [python/test_p52_regime.py](python/test_p52_regime.py)::AntiLoopholeTests).

## Conventions

- Audio dataset (800 cases): `wav/aec_challenge_blind/{doubletalk,farend_singletalk,nearend_singletalk}/<stem>_{mic,lpb}.wav`.
- Per-case CNG determinism: `np.random.seed(42)` before each `AEC(cfg)` instantiation.
- HPF defaults locked: far-end (ref) HPF=OFF, mic-path HPF=ON.
- macOS: use `python3` (not `python`); kiss_fft symlink in `c_impl/lib/kiss_fft` → `../../../lib/nr/c_impl/lib/kiss_fft` in the Audio_ALG integration repo.

## Branch model

`main` carries the production-graded code. Current `__version__` is
**3.21.4** — v3.21.4 is an audit / structural-refactor cycle:
closes 4 carry-overs from v3.21.2's plan (time-domain unit-conversion
"bugs" NOT-A-BUG; per-bin H_error refresh retest FAIL on canonical
state; B3 lf_endpoint intermediate values FAIL; ReverbDecayEstimator
NOT-PORTING — estimator dormant in our pipeline). Includes a
structural ms-based config refactor for `hold_duration_ms` /
`noise_floor_hold_ms`. No production code change (byte-equal at default
vs v3.21.3). v3.21.3 was the Codex hygiene cycle. v3.21.2 corrected
the FFT-scale bin-index unit-conversion bug. v3.21.0 retired the
legacy `ResFilter` 9-stage chain in favour of the AEC3-aligned
`_aec3_post` (AecState + ResidualEchoEstimator + SuppressionGain +
CNG). Single production preset: `BALANCED`. See
[CHANGELOG.md](CHANGELOG.md) for full per-version detail.

The active reference set at `docs/` root holds the canonical pipeline +
algorithm documentation; closed-arc verdict / design docs from prior
cycles were retired in the v3.21 cleanup (see [CHANGELOG.md](CHANGELOG.md)
for the per-round delta).
