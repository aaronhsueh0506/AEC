# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this repo is

Single-channel AEC (1 mic + 1 ref reference signal). Two implementations of the
**same algorithm** that must produce **bit-equal output** within numerical
tolerance:

- `python/aec.py` — algorithm reference, ~4500 lines, single file. Where new
  algorithm work happens first.
- `c_impl/` — production C port. Mirrors the Python class structure
  (`PBFDKF`, `ShadowFilter`, `ResFilter`, etc.). Built with
  `-ffp-contract=off` mandatory.

Algorithm version is tracked by `__version__` in [aec.py](python/aec.py).
Canonical algorithm reference: [docs/aec_methods.md](docs/aec_methods.md).
Trace-driven evolution history: [docs/aec_v3_evolution.md](docs/aec_v3_evolution.md).
Research log canonical: [docs/SUMMARY.md](docs/SUMMARY.md).

## Common commands

### Python single-case (algorithm dev)

```bash
# Standard run (BALANCED preset)
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
    --preset balanced --filter 832 --cng -o out_python/ -j 4

# AECMOS scoring (needs speechmos + onnxruntime ≤1.16.3 + numpy<2 in venv)
python3 python/bench_aecmos.py out_python/ results/
python3 python/bench_aecmos.py out_python/ results/ --baseline /path/to/baseline_scores.json
```

**Standard 800-case config**: `preset=balanced / fl=832 (52 ms) / cng=True / j=4`.
This combination is the reference for every regression check, A/B, and audit.
Deviating from it produces results that aren't comparable to prior verdicts.

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

There is no project-wide pytest collection. Tests live alongside the modules
they cover (e.g. `python/test_p52_regime.py`).

## Architecture — what to read before changing things

### Pipeline

```
mic ─► HPF ────────────────────────────────────────────────────►
ref ─► HPF ─► Saturation ─► DelayEst+RingBuf ─► PBFDKF ─► error ─► ResFilter ─► out
                                                  │
                                            Shadow filter (Q×3.5)
                                            + PathChangeRegimeHandler
```

Tight coupling lives in two places:

1. **`PBFDKF` + `ShadowFilter` + `PathChangeRegimeHandler`** ([aec.py:1026](python/aec.py#L1026), [aec.py:3577](python/aec.py#L3577) and surrounds) — the shadow filter and main filter exchange state via a regime handler that fires `boost_q` / `reverse_copy` / `main_paused` decisions. **Was previously named `ShadowCopyController`**; renamed under P52 Path 3 (see [docs/p52_phase_a_verdict.md](docs/p52_phase_a_verdict.md)). The handler is **load-bearing on the cohort tail** (~7/800 cases); do not remove or bypass it.

2. **`ResFilter.process()` 9-stage pipeline** ([aec.py:1390](python/aec.py#L1390) ff.) — residual model → gain compute (softgate_emr / spectral_floor) → gain postprocess (epc_dt_cap / quiet_mask / 3bin_smooth / hf_cap / pre_temporal) → temporal smoothing → noise floor + CNG. State threads through ~25 `self.*` fields per frame. The P52 Phase B refactor extracts each stage into [python/res_refactored/](python/res_refactored/) (subclass-and-delegate pattern; see [docs/p52_phase_b_module_1_verdict.md](docs/p52_phase_b_module_1_verdict.md)).

### `AecConfig` and presets

Five operating points (`MILD`/`SOFT`/`BALANCED`/`AGGRESSIVE`/`MAXIMUM`) defined
in [aec.py:146-365](python/aec.py#L146). **Presets are co-tuned** — every knob
inside a preset is anchored against the 800-case AECMOS bench. Don't tweak
individual RES knobs; switch presets or design a new one with a full
800-case re-bench.

### Diagnostic surfaces (do not remove)

- `AecStats` / `get_stats()` ([aec.py:62-123](python/aec.py#L62) + 4344) — per-frame audio-passive trace consumed by `run_one_case.py` plots and external research tooling.
- `AecResContext` — exposes `echo_spec` / per-bin Kalman state so the linear stage can feed an external (or NN) post-filter; `AecConfig.return_res_context = True` switches `aec.process()` return type to `(out, AecResContext)`.
- `_diag_round5_stages` ([aec.py:1488-1491](python/aec.py#L1488)) — 9-slot voice-band gain trace for RES per-stage inspection.
- `trace_p52_regime_handler` flag (default-OFF) — per-frame regime handler trace; classifier in `python/aec_p52_regime_classifier.py` is analysis-only (enforced by `python/test_p52_regime.py::AntiLoopholeTests`).

### Active refactor: P52 Phase B

`python/res_refactored/` contains the in-progress modular extraction of
`ResFilter`. Each module file (`residual_estimator.py`, `gain_computer.py`,
`spectral_shaper.py`, `temporal_smoother.py`, `noise_floor_cng.py`) holds the
verbatim body of one legacy `ResFilter._stage_*` method. `ResFilterRefactored`
subclasses `ResFilter` and overrides each stage to delegate to the new free
function. The byte-equal validation tools live in
[tools/research/p52_phase_b_b3_byte_equal.py](tools/research/p52_phase_b_b3_byte_equal.py)
(100-case sample) and `p52_phase_b_b4_byte_equal.py` (full 800).

The refactor is governed by the P52 v1.1 design lock
([docs/p52_design_lock_v1.1.md](docs/p52_design_lock_v1.1.md) on
`feature/p52-design-lock-v1.1`). Anti-loophole §5.5 forbids any RES logic
change inside this refactor — extractions must be byte-identical at the
sample level (hard bar §3.6: ≥99.99% within `atol=1e-6, rtol=1e-5`; internal
target 100% exact).

## Conventions

- Audio dataset (800 cases): `wav/aec_challenge_blind/{doubletalk,farend_singletalk,nearend_singletalk}/<stem>_{mic,lpb}.wav`.
- Per-case CNG determinism: `np.random.seed(42)` before each `AEC(cfg)` instantiation. All A.0R / B.3 / B.4 tooling uses seed=42.
- Research outputs: `tools/research/p52_*` produces artefacts under `/tmp/p52_*/` (gitignored) and verdict docs under `docs/p52_*.md` (committed).
- macOS: use `python3` (not `python`); kiss_fft symlink in `c_impl/lib/kiss_fft` → `../../../lib/nr/c_impl/lib/kiss_fft` in the Audio_ALG integration repo.

## Branch model

`main` carries the production-graded code. Current `__version__` is **3.15.0**
(v3.15 arc closeout 2026-05-15, tagged `v3.15.0`). The version-bump path
since the May 12 P52 Path 3 merge:

- `p52-phase-a-closed-path3` (2026-05-12) — Phase A merge baseline.
- `v3.10.6` — F3.1 v3 / F2.3 / F2.4 promoted (xrtntuju 5-clip arc closed).
- `v3.11.0` / `v3.11.1` / `v3.11.2` — Phase 1 promotions
  (B5 / B6 / F-E1 / F-DelayTrack / F-E5 / diverged_reset triple-AND).
- v3.12 — Stage 1 RES exhaustion (5 NEUTRAL closures, no version bump).
- `v3.13.0` (2026-05-14) — v3.13 arc closure: E2 Path 3 SHIPPED
  (FS_static Δecho +0.107); E4 NLP + E5 Saturation closed CANNOT SHIP
  (substrate retained for v3.14); Phase 3 RES audit done.
- `v3.14.0` (2026-05-14) — Arc P (per-band ERL EMA) + Arc R (per-band
  ENR `block_lf` tilt) + Arc S-orth.A (decoupled shadow Kalman state)
  promoted to BALANCED. Arc H (Huber loss) closed CANNOT SHIP. B1 + B2
  housekeeping shipped.
- `v3.15.0` (2026-05-15) — Arc T cohort tail real-time detector
  promoted to BALANCED default ON (byte-equal on audio). Six candidate
  arcs CLOSED CANNOT SHIP (§1.2 / Arc M V1+V2 / Arc G / Arc T S2 wiring
  / Arc M.v3 / Arc F). 13 v3.16 candidates ranked; v3.16 RES refactor
  arc authorised pending kickoff.

See [CHANGELOG.md](CHANGELOG.md) for full per-version detail.

**Active branches (post v3.15.0):**

- `main` — production HEAD at v3.15.0.
- `feature/static-memory` — NR / AEC static-memory pipeline (long-lived).
- `feature/v3.16` — v3.16 RES refactor arc kickoff branch (created at
  v3.15.0 merge). Phase 0 housekeeping (HK-1 + HK-2 + C1 + C1b) +
  Phase 1 foundation (C5 per-state RES interface + C6 DelayEst audit).
  See [docs/v3_15_res_audit_and_refactor_plan.md](docs/v3_15_res_audit_and_refactor_plan.md)
  for the candidate inventory + Phase 0-4 dependency graph.

Feature branches stay file-disjoint per
[design lock §6.4](docs/p52_design_lock_v1.1.md): `feature/p52-phase-a-shadow`
touches shadow/PathChangeRegimeHandler only; `feature/p52-phase-b-refactor`
touches `python/res_refactored/` only. The C port (`c_impl/`) follows Python
once an algorithm change has merged.
