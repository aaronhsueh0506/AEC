# Development guide

This document records repository invariants, validation commands, and
maintenance conventions for anyone changing the implementation. Product users
should start from the root README or the C integration guide instead.

## What this repo is

Single-channel AEC (1 mic + 1 ref reference signal). Two implementations of the
**same algorithm** — Python is the fp64 algorithm spec, C is the float32
production implementation, and their outputs must agree within a documented
numerical tolerance (not bit-equal — see the float32 campaign note below):

- `python/aec.py` — top-level shim that re-exports every public symbol so
  `from aec import AEC, AecConfig, ...` keeps working. Algorithm itself
  lives under `python/modules/`.
- `python/modules/` — algorithm modules organised into the AEC3-aligned
  chain (`state/`, `residual/`, `filter/`, `render/`, `delay/`) plus the
  shared infrastructure (`filters`, `epc`, `detectors`,
  `preprocessing`, `erle`, `dataclasses`, `config`, `orchestrator`,
  `debug_logger`).
- `c_impl/` — production C port. Mirrors the Python class structure
  (`PBFDKF`, `ShadowFilter`, etc.). Built with `-ffp-contract=off` mandatory.

Algorithm version is tracked by `__version__` in [aec.py](../python/aec.py)
(currently **3.23.0**; BALANCED changed in 3.23.0 — no-PA matched-filter
pre-echo fix + DT-deg recovery stack — see CHANGELOG).

**Float32 campaign (2026-07-15): all production C is now float32
end-to-end** — delay chain, orchestrator scalars, post/state modules,
`residual_echo_estimator`, and the mic-path HPF. The optional double-precision
reverb-decay reference port is test-only under `c_impl/test/support/` and is
not part of `libaec.a`. **Python bit-exact parity is retired repo-wide**: the Python
reference (fp64) is now the **algorithm spec**, C is the float32
**implementation**, and Python↔C comparison is **tolerance-based** (~−60 dB
class, correlation 0.99999958) — never 0/0. The regression anchors are now:

- **C-goldens** — `c_impl/test/parity_delay.c` regenerated against its own
  prior output via `c_impl/test/gen_delay_c_golden.c` (the delay chain's
  fast-math/duty-cycled matched filter is checked against itself, not
  Python), and the end-to-end tolerance gate `c_impl/test/parity_aec_e2e.c`.
- **Staged gates vs the `fp64-baseline` git tag** — 60-case stratified
  AECMOS (worst per-case delta −0.021 echo, all bucket means ≤0.002, within
  the established noise bar), waveform drift median −95 dB, and a 1-hour
  soak (delay trajectory identical, power-EMA worst rel diff 1.3e-5, final
  ERLE matching to 4 digits).

The production **FFT backend is KISS FFT (float32)** (host/reference build:
malloc + KISS, `make`, default). The embedded deployment (caller pool + NE10,
`make BACKEND=ne10` + `aec_get_mem_size`/`aec_init`) ships from the same main
branch; NE10 vs KISS output is not bit-identical to each other (pre-existing,
expected), but each backend's static path is byte-equal to its own malloc
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

# AECMOS scoring (needs speechmos + onnxruntime ≤1.16.3 + numpy<2 in venv)
python3 python/bench_aecmos.py out_python/ results/
python3 python/bench_aecmos.py out_python/ results/ --baseline /path/to/baseline_scores.json
```

**Standard 800-case config**: `preset=balanced / filter=832 (52 ms) / --cng /
--parallel / --workers 4`. This combination is the reference for every
regression check, A/B, and audit. Deviating from it produces results that
aren't comparable to prior verdicts.

### Byte-equal regression check (post-cleanup gate)

A behaviour-neutral cleanup must produce byte-identical output on the side of
the port it touches. **Python-side edits** still use the Python render
procedure below. **C-side edits** (this is the standard path since the f32
campaign) are gated with C renders instead — `bin/aec_wav` binary output +
`cmp` — not the Python renderer, since Python↔C is tolerance-based, not
byte-equal (see "What this repo is" above).

**C-side gate** (render with `aec_wav` before/after, compare with `cmp`).
`aec_wav` now lands in a config-hashed `bin/<backend>-<config-hash>/`
directory — resolve the exact path with `make
print-bin-dir` (same flags as your build), or use the stable `dist/kiss/
current/` path after `make publish`:

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
(tolerance gate) — see `c_impl/STATIC_MEMORY.md`. Older parity reports were
retired during the release cleanup and remain recoverable from Git history.

**Python-side gate** (unchanged — render a small case set before and after
the edit and compare with `cmp`):

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
"$bin/aec_wav" mic.wav ref.wav out.wav --debug-level 2 --debug-log /tmp/aec.log
```

`-ffp-contract=off` in `CFLAGS` is mandatory for build determinism and golden
stability (no FMA reassociation drift between compilers/builds) — retained
post-campaign even though Python↔C is no longer a byte-equal target. As of
the build hardening work this is a **unified policy across all four repos**
(`audio_common`, `NR/c_impl`, `AEC/c_impl`, `Audio_ALG/pipelines`): every TU
each Makefile compiles, own code and vendored KISS/NE10 alike, builds with
the flag appended LAST (after `EXTRA_CFLAGS`/`BACKEND`/`WERROR`/`NO_STDIO`)
so nothing can override it, with parse-time rejection of an `EXTRA_CFLAGS`
containing `-Ofast`/`-ffast-math`/`-ffp-contract=<anything>`, and
outright rejection of any command-line `CFLAGS=`/`CXXFLAGS=`/`LDFLAGS=`
override (which would silently disable the Makefile's own appends —
`EXTRA_CFLAGS`/`EXTRA_LDFLAGS` are the supported hooks). See
`c_impl/README.md`'s "Unified FP-contraction policy" section and
`audio_common/scripts/audit_fp_contract.sh` (the disassembly-level PASS/
EXEMPT audit gate) for the full writeup.
Output WAV defaults to fp32 PCM (`AEC_OUT_FLOAT=0` for 16-bit).

### Python tests

```bash
python3 -m pytest python/tests/test_p52_regime.py   # P52 classifier unit tests
```

There is no project-wide pytest collection.

## Architecture — what to read before changing things

### Pipeline (v3.21)

```
mic ─► HPF ──────────────────────────────────────────────────────────►
ref ─────► Saturation ─► DelayEst+RingBuf ─► PBFDKF ─► error ─► AEC3 post ─► out
                                               │                       │
                                       Shadow filter (PBFDAF/NLMS) AecState + ResidualEchoEstimator
                                       + PathChangeRegimeHandler + SuppressionGain + CNG (OLA)
```

HPF runs on the mic path only. The ref-path HPF was retired (default
OFF) after the v3.19 ref-flip verdict; downstream `Saturation` /
`EchoPathDelayEstimator` consume the raw reference.

The v3.21 pipeline retires the legacy 9-stage `ResFilter` chain. Its
replacement is `AEC._aec3_post` in [python/modules/orchestrator.py](../python/modules/orchestrator.py),
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
lives in [python/modules/filters.py](../python/modules/filters.py);
PathChangeRegimeHandler in [python/modules/epc.py](../python/modules/epc.py).
**Was previously named `ShadowCopyController`** (renamed under P52
Path 3 of the v3.10.6 cycle — kept the audio path identical, added
the regime-classifier anti-loophole test). The handler is
**load-bearing on the cohort tail** (~7/800 cases); do not remove or
bypass it.

### `AecConfig` and presets

Three presets — `mild` / `balanced` / `aggressive` (NR-style naming; `mild` was `gentle` until 2026-07-15, same −20 dB parameters) — defined in
[python/modules/config.py](../python/modules/config.py) (`from_preset`). All share
the same AEC3 chain + the four 800-case AECMOS-tuned base overrides (`enable_cng`,
`shadow_mu_min`, `warmup_frames`, `kalman_q_high`); everything else uses dataclass
defaults. **Knobs are co-tuned** — don't tweak a single field without a full
800-case re-bench.

`balanced` is the production preset (all four ship bars met). `mild` and
`aggressive` are deliberate **Pareto operating points** on a single residual-echo
strength knob — `min_gain_floor_far_active_db`, the far-active min-gain floor
(mild −20 / balanced −28 / aggressive −38 dB):

- `mild` = near-priority (higher floor → more near-end kept, more echo leak;
  DT_static deg reaches AEC2's, FS echo drops below balanced's 3.5 bar by design).
- `aggressive` = echo-priority (deeper floor → more echo killed, more near loss;
  beats AEC2 on DT+FS echo, deg stays >2.0 and above AEC3).

The DT-deg-vs-echo trade is a proven single-channel DSP Pareto wall (see CHANGELOG
`[3.22.4]`); the strength axis exposes it honestly rather than hiding it. mild/
aggressive differ from balanced **only** in that one floor field.

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
- Per-case CNG determinism: `np.random.seed(0)` before each `AEC(cfg)` instantiation (see `eval_aec_challenge.py:run_ours`).
- HPF defaults locked: far-end (ref) HPF=OFF, mic-path HPF=ON.
- macOS: use `python3` (not `python`); kiss_fft symlink in `c_impl/lib/kiss_fft` → `../../../lib/nr/c_impl/lib/kiss_fft` in the Audio_ALG integration repo.
- `active_render` threshold is **5.96e-4**, empirically tuned — the strict-AEC3
  value 9.31e-6 was validated as a regression; do not "align" it to the reference.

## Branch model

`main` carries the production-graded code. The current production preset is
**3.23.0** BALANCED: it adds the no-pre-align (no-PA) online-delay fix — the
matched-filter pre-echo `accumulated_error` binning bug (`i//4` → AEC3 cumsum
prefix-error) that had collapsed pre-echo to 0 and corrupted no-PA delay
estimation — plus a default-ON DT-deg recovery stack (`dt_aware_recovery_soft`
+ `dt_aware_res_floor`, `min_gain_floor_dt_db = −20`), and at the time completed
Python↔C bit-exactness under `-DUSE_STANDARD_MATH` (4 production-C port bugs
fixed) — **since superseded by the 2026-07-15 float32 campaign** (see "What
this repo is" above): Python bit-exact parity is retired repo-wide, Python is
now the algorithm spec and C the tolerance-checked float32 implementation. It
supersedes 3.22.2 as production; see CHANGELOG `[3.23.0]`. Earlier frontier
history (3.22.x and v3.21 closeout) lives in [CHANGELOG.md](../CHANGELOG.md).
