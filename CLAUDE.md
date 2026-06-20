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
  shared infrastructure (`filters`, `epc`, `detectors`,
  `preprocessing`, `erle`, `dataclasses`, `config`, `orchestrator`,
  `debug_logger`).
- `c_impl/` — production C port. Mirrors the Python class structure
  (`PBFDKF`, `ShadowFilter`, etc.). Built with `-ffp-contract=off` mandatory.

Algorithm version is tracked by `__version__` in [aec.py](python/aec.py)
(currently **3.23.0**; BALANCED changed in 3.23.0 — no-PA matched-filter
pre-echo fix + DT-deg recovery stack — see CHANGELOG). The Python↔C port is
**bit-exact under `-DUSE_STANDARD_MATH`** (all module parity tests + end-to-end
`parity_aec_e2e`, 0 mismatches, 3 presets); production `fast_math.h` is the only
residual (~1e-5..1e-4 in exp/sqrt stages). Canonical algorithm reference:
[docs/aec_methods.md](docs/aec_methods.md). Architecture flowcharts —
current vs AEC3 reference:
[docs/architecture_v3_22_5_vs_aec3.html](docs/architecture_v3_22_5_vs_aec3.html).

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

A behaviour-neutral cleanup must produce byte-identical output. Render a
small case set before and after the edit and compare with `cmp`:

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
                                       Shadow filter (PBFDAF/NLMS) AecState + ResidualEchoEstimator
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

Three presets — `gentle` / `balanced` / `aggressive` — defined in
[python/modules/config.py](python/modules/config.py) (`from_preset`). All share
the same AEC3 chain + the four 800-case AECMOS-tuned base overrides (`enable_cng`,
`shadow_mu_min`, `warmup_frames`, `kalman_q_high`); everything else uses dataclass
defaults. **Knobs are co-tuned** — don't tweak a single field without a full
800-case re-bench.

`balanced` is the production preset (all four ship bars met). `gentle` and
`aggressive` are deliberate **Pareto operating points** on a single residual-echo
strength knob — `min_gain_floor_far_active_db`, the far-active min-gain floor
(gentle −20 / balanced −28 / aggressive −38 dB):

- `gentle` = near-priority (higher floor → more near-end kept, more echo leak;
  DT_static deg reaches AEC2's, FS echo drops below balanced's 3.5 bar by design).
- `aggressive` = echo-priority (deeper floor → more echo killed, more near loss;
  beats AEC2 on DT+FS echo, deg stays >2.0 and above AEC3).

The DT-deg-vs-echo trade is a proven single-channel DSP Pareto wall (see CHANGELOG
`[3.22.4]`); the strength axis exposes it honestly rather than hiding it. gentle/
aggressive differ from balanced **only** in that one floor field.

### Diagnostic surfaces (do not remove)

- `AecStats` / `get_stats()` ([python/modules/dataclasses.py](python/modules/dataclasses.py)
  + the AEC method in [python/modules/orchestrator.py](python/modules/orchestrator.py))
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

## Branch model

`main` carries the production-graded code. The current production preset is
**3.23.0** BALANCED: it adds the no-pre-align (no-PA) online-delay fix — the
matched-filter pre-echo `accumulated_error` binning bug (`i//4` → AEC3 cumsum
prefix-error) that had collapsed pre-echo to 0 and corrupted no-PA delay
estimation — plus a default-ON DT-deg recovery stack (`dt_aware_recovery_soft`
+ `dt_aware_res_floor`, `min_gain_floor_dt_db = −20`), and completes Python↔C
bit-exactness under `-DUSE_STANDARD_MATH` (4 production-C port bugs fixed). It
supersedes 3.22.2 as production; see CHANGELOG `[3.23.0]`. The frontier history
below is retained for context.

The **3.22.2** BALANCED preset
(`soft_nearend_blend_per_bin`, default ON) + far-active min-gain floor
−28 dB (`min_gain_floor_far_active_db`): the per-bin frequency-selective
near-end protection lets the deeper floor cancel more echo (DT echo +0.113
vs the −22 baseline) at only −0.044 DT deg, all four ship bars met. The
`far_active_floor_db` is the single-knob preset axis (weak −18 / strong
−28+). Built on the v3.22.1 P4 delay-acquire guard, the v3.22.0 split
min-gain floor + default-ON stack (E1+x2+E2+D3+L1+C′), and the v3.21.6.4
AEC3-alignment completion. See CHANGELOG `[3.22.2]` and
[docs/archive/v3_22.md](docs/archive/v3_22.md) for the full flag-campaign evidence.

On top of 3.22.2, a **byte-equal hygiene pass** (CHANGELOG `[Unreleased]`,
`__version__` unchanged): Track A `SuppressionGain`-ctor dedup, Track B
retired 6 dud default-OFF flags (`AecConfig` 109 fields), Track C added the
default-OFF per-bin near-end SPP substrate (`NearendSpp` +
[python/spp_step0_diag.py](python/spp_step0_diag.py)) for the DT frontier —
**NULL verdict** (near-gated cohxd lands on the plain-cohxd Pareto line; the
per-bin near-mask hits the voice-on-voice bin-overlap wall, [docs/archive/v3_22.md](docs/archive/v3_22.md)
§7). All three byte-equal-verified; production behaviour unchanged.

On top of that, **3.22.3** (CHANGELOG `[3.22.3]`) ships the surviving
**isolated parity/correctness candidates** from the Codex source audit:
**P0.1** coherence-gate EMA reset hygiene, **P0.4** analysis-window canonical
sqrt-Hann (true perfect reconstruction), **P0.5** `erle_e2y2_gate_*` preserved
across reset. Output changes vs 3.22.2 but **AECMOS-neutral** (≤0.002 all
buckets). Three parity candidates were **gated** (P0.2a windowed SG-nearend,
P0.2b CNG source, P0.3 C′ selected/windowed) — the audit's "contaminates R²"
framing was refuted (R² is decoupled from `near_psd`); P0.2b is kept documented
in-code as a CN-floor DT-deg lever for the frontier. See [docs/archive/v3_22.md](docs/archive/v3_22.md) §8.

The **v3.21 CLOSE** (branch `v3_21_release`, byte-equal, no algorithm
change — see CHANGELOG `[Unreleased]`) finalised the arc: a hop/fft
conversion audit + 800-case Tier-C validation adjudicated every
"physical-meaning conversion" flag (matched-magnitude AECMOS Pareto),
then hard-coded the surviving default-True alignment flags into their
call sites and deleted all NOSHIP / temp substrate. Net effect on the
two large files: **config 412→230, filters 1057→863** (legacy
P-denominator Kalman body removed; 10 always-True refined/shadow
AEC3-parity flags inlined), orchestrator construction/`_aec3_post`
branches collapsed. Also removed the unused alternate filter line
(`NlmsFilter` + `AecMode.LMS`/`NLMS`) and the dead `erle.py`
back-compat re-export. `active_render` 5.96e-4 is now documented as an
empirically-tuned threshold (the strict-AEC3 9.31e-6 validated as a
regression) — the only flag whose "alignment" label changed.

Byte-equal verified across the cleanup arc (`_ours` + `_ours_nores`
md5, all buckets incl. movement). See [CHANGELOG.md](CHANGELOG.md) for
per-version detail and
[docs/architecture_v3_22_5_vs_aec3.html](docs/architecture_v3_22_5_vs_aec3.html)
for the architectural before/after.
