# AEC — Acoustic Echo Cancellation

> **C 使用與整合**：[AEC C User Manual（繁體中文）](docs/c_user_manual_zh_TW.md)

Single-channel AEC (1 mic + 1 ref) supporting PBFDKF (frequency-domain Kalman),
multi-ERLE, shadow filter, and post-filter residual echo suppression.
Python reference implementation + C implementation.

**Release**: v4.0.0rc1 — Python `aec.py` `__version__ = "4.0.0rc1"`. **Not ship-ready.** Two entries under `[4.0.0]` (the AEC3 Tier-2 constant audit and the 16 kHz default-grid flip) change production 16 kHz output and carry their own "800-case bench not run" disclaimers, and the hop-authored timing audit is incomplete outside `detectors.*` — see `[4.0.0]` "Known gaps". Do not tag or publish until those close. The `4.x` major bump records the public-ABI and output-contract breaks listed under `[4.0.0]` in [CHANGELOG.md](CHANGELOG.md) (custom output limiter removed so output is no longer bit-identical, `AecConfig`/`AecResContext` layout changes, 16 kHz default grid 512/256 → 256/128), **not** a new algorithm generation. The production algorithm is the v3.21 AEC3-aligned `_aec3_post` chain (AecState + ResidualEchoEstimator + SuppressionGain + CNG) with the v3.22 split min-gain floor (DT/NE near-end preservation). **3.23.0** fixes the no-pre-align (no-PA) online-delay path — the matched-filter pre-echo `accumulated_error` binning bug (`i//4` → AEC3 cumsum prefix-error) that had collapsed pre-echo to 0 and corrupted no-PA delay estimation — and ships a default-ON DT-deg recovery stack (`dt_aware_recovery_soft` + `dt_aware_res_floor`, `min_gain_floor_dt_db = −16`). Three Pareto presets — `mild` / `balanced` / `aggressive` — differ only in the far-active min-gain floor; **`balanced` is production** and meets all four ship bars (FS echo >3.5, DT echo >4, DT deg >2, NE deg ≥4). See [CHANGELOG.md](CHANGELOG.md) `[3.23.0]`.

**Float32 campaign** (2026-07-15, on top of 3.23.0): all production C is now
float32 end-to-end (delay chain, orchestrator scalars, post/state modules,
`residual_echo_estimator`, HPF). **Python bit-exact parity is retired
repo-wide** — the Python reference (fp64) is now the **algorithm spec**, C is
the float32 **implementation**, and Python↔C comparison is **tolerance-based**
(~−60 dB class, correlation 0.99999958), not 0/0. The production **FFT
backend is KISS FFT (float32)** on the host/reference build (`make`, malloc);
the embedded deployment (`make BACKEND=ne10`, caller pool via
`aec_get_mem_size`/`aec_init`) ships from the same main branch — NE10 vs KISS
output is not bit-identical to each other (pre-existing), but each backend's
static path is byte-equal to its own malloc path. Regression anchors: C-goldens
(`c_impl/test/parity_delay.c` + `c_impl/test/parity_aec_e2e.c`) and staged
gates vs the `fp64-baseline` tag (60-case stratified AECMOS within noise bar,
waveform drift median −95 dB, 1-hour soak stable). See
[CHANGELOG.md](CHANGELOG.md) `[Unreleased] — 2026-07-15`.

---

## Status snapshot

**800-case AECMOS — ours vs WebRTC AEC3 vs Speex (MDF)**, `balanced /
fl=832 (52 ms) / --cng`, local AECMOS ONNX, full AEC Challenge 2021 blind
corpus (echo↑ deg↑). **No pre-alignment** — every engine self-aligns online
(ours via its in-pipeline matched-filter `EchoPathDelayEstimator`; AEC3 via its
internal `RenderDelayController`; Speex internally), exactly as in production:

| Bucket | n | **ours** echo / deg | AEC3 echo / deg | Speex echo / deg |
|---|---|---|---|---|
| FS_static   | 169 | 3.544 / 4.999 | 3.821 / 4.999 | 2.847 / 5.000 |
| FS_movement | 131 | 3.519 / 4.999 | 3.790 / 4.999 | 2.757 / 5.000 |
| DT_static   | 186 | 4.218 / 2.074 | 4.531 / 1.815 | 3.427 / 3.179 |
| DT_movement | 114 | 4.114 / 2.140 | 4.456 / 1.816 | 3.272 / 3.301 |
| NE          | 200 | 4.998 / 4.021 | 4.999 / 3.410 | 4.998 / 4.128 |

- **Echo cancellation: AEC3 > ours > Speex.** AEC3 cuts the most; Speex is a
  weak canceller (FS ~2.8, DT ~3.3); ours sits in between (FS ~3.5, DT ~4.2).
- **Near-end preservation (deg): Speex > ours > AEC3.** AEC3 pays for its echo
  cancellation with the worst DT deg (1.82) and NE deg (3.41); ours holds the
  middle and **beats AEC3 on every degradation axis** (DT deg 2.07/2.14 vs 1.82,
  NE deg 4.02 vs 3.41) while **approaching AEC3 on echo** (FS_movement 3.519 vs
  3.790, within 0.27). This is the ship target: *approach AEC3 on echo, beat it
  on deg.* All four ship bars met; **FS_movement 3.519 > 3.5** is the primary gate.
- These numbers reflect the v3.23.0 matched-filter pre-echo fix (the delay
  estimator now holds the true echo-path delay online instead of collapsing
  toward 0) plus the default-ON DT-deg recovery stack (soft acquire-reset +
  DT-gated RES floor), which trades a little echo for DT-deg headroom above the
  pre-align reference while keeping all four ship bars.

### Presets — one Pareto knob

Three presets differ **only** in `min_gain_floor_far_active_db`, the far-active
residual-gain floor that trades echo suppression against near-end preservation:

| Preset | floor | character |
|---|---|---|
| `mild`       | −20 dB | near-priority — more near-end kept, more echo leak (FS dips below the 3.5 bar by design). Was `gentle` until 2026-07-15 (NR-style naming, same parameters) |
| **`balanced` ★** | −28 dB | **production** — all four ship bars met |
| `aggressive`   | −38 dB | echo-priority — deeper suppression, more near-end loss (deg stays >2.0, above AEC3) |

`mild` / `aggressive` are deliberate Pareto operating points on the proven
single-channel DT-deg-vs-echo wall; all share the same `_aec3_post` chain and
800-case-tuned base. Full version history → [CHANGELOG.md](CHANGELOG.md).

---

## Specifications & limits

### Input

| Field | Value |
|---|---|
| Sample rate | 8 / 16 / 48 kHz (whitelisted no-padding grids) |
| Bit depth | 16-bit PCM or 32-bit float |
| Channels | mono |
| Frame / hop | 8k: 256/128; 16k: 256/128 default or 512/256; 48k: 1024/512 |
| Filter length | 52 ms @ 8/16 kHz; 64 ms @ 48 kHz (configurable) |
| Algorithmic delay | one hop with RES: 16 ms / 8 ms / 10.667 ms by grid |

### Resource (C, 16 kHz default grid 256/128, fl=52 ms)

| | |
|---|---|
| Static pool | Backend、grid、filter 長度與功能開關都會影響大小；部署時以 `aec_get_mem_size()` 查詢，不使用文件中的固定常數 |
| Static pool, other backend | Query `aec_get_mem_size`; FFT workspace is backend-dependent |
| Compute / frame | 4 × 256-FFT + Kalman update (129 bins × 7 partitions) |
| FFT                                | KISS FFT (float32; NE10 ARM-NEON opt-in) — ~float32 precision vs numpy `np.fft` |

Both backends are static==dynamic byte-equal (`test_static_aec`); full
per-region breakdown → [c_impl/STATIC_MEMORY.md](c_impl/STATIC_MEMORY.md).

### Known limitations (cases this AEC cannot fully solve)

| Limitation | Detail |
|---|---|
| Supported SR              | 8 / 16 / 48 kHz only — 32 kHz / 44.1 kHz need external resample |
| Channel count             | mono only — no mic array / stereo reference |
| Echo nonlinearity         | linear filter + RES; no dedicated nonlinear model. High speaker distortion produces residual harmonic leakage |
| Delay direction           | positive only (mic lags ref). Negative-delay scenarios must be aligned upstream |
| Stationary far + weak NE  | linear AEC may absorb very low-energy NE syllables into the echo path. Stationary-DT hangover mitigates but cannot fully restore worst frames |
| DT-from-frame-0           | NE present from sample 0 prevents the filter from ever converging — see [docs/aec_methods.md 附錄 E](docs/aec_methods.md#附錄-e-dt-from-frame-0-限制) |
| Echo-path > filter length | echo tail beyond `filter_length` (default 52 ms worth of samples) is not modelled. Larger rooms need a longer filter (cost: memory + compute). C-API only — there is no CLI flag for it |
| Mid-stream SR/filter change | not supported. `filter_length` is fixed at construction; SR change requires destroy + re-create |
| Mic/ref clock drift       | sustained sample-rate drift between mic and ref tanks ERLE (filter cannot track) |

### Common problems & adjustments

Both Python (`aec.py`) and C (`c_impl`'s config-keyed `aec_wav`) use the same algorithm; the
adjustments below apply to both unless noted. CLI flag examples shown for
C; equivalent Python flags differ only in syntax (`--mode pbfdkf` etc.).

| Symptom | Diagnosis & adjustment |
|---|---|
| **Residual echo too high (FS / NE)** | 1. With `--no-res`, output should be a clean linear-AEC residual. Echo still dominating → ref signal is wrong, mic-ref delay > filter length, or sample rates differ. 2. `--preset aggressive` for stronger RES (cost: more NE compression). 3. `--filter-length-ms 100` for big rooms / long reverb. |
| **NE clipped during double-talk** | Lower preset → `--preset mild` (−20 dB floor, near-priority: keeps more near-end at the cost of more echo leak). Don't tweak individual RES knobs — preset values are co-tuned. |
| **Slow startup / first-second echo** | Filter convergence needs ≥ 0.5 s of meaningful far energy. Normal adaptive behavior. Consider muting output during application warm-up (e.g. play a "connecting…" cue). |
| **Echo spikes when device moves** | Echo path changes → EPC fires → ~200 ms re-convergence with brief leak. Usually self-recovers. For frequent movement, increase filter length. |
| **Output sounds muffled / pumping in NE-only** | Comfort noise mismatch. Try `--cng` (both C and Python; Python uses `--no-cng` to disable) to shape the noise floor; do NOT stack a second CNG layer downstream. |
| **First file sounds OK but second file glitches in batch processing** | Missed `aec_reset` between files. DT / EPC / convergence state accumulates. Reset before each independent stream. |
| **Linear-AEC residual sounds wrong (separate filter from RES)** | `"$BIN"/aec_wav mic ref linear_only.wav --no-res` after resolving `BIN="$(make -s print-bin-dir)"` in `c_impl/` (or use the Python `--no-res` equivalent). This isolates filter-side issues from RES-side issues. |
| **Per-frame state inspection** | C: `--debug-level 2 --debug-log /tmp/aec.log`, then `grep PBFDKF /tmp/aec.log`. Python: `python3 aec.py mic ref out --diag`. |
| **Detect mic/ref drift or wrong delay** | Run with `--no-delay-est`, supply pre-aligned files, compare output vs the online-delay-est version. Large divergence → drift, or a delay beyond the acquirable ceiling (1216 / 608 / 608 ms at 8 / 16 / 48 kHz — compile-time, not raised by `max_delay_ms`; see the C manual §3). |
| **Build mismatch between Python and C output** | Verify C built with `-ffp-contract=off` (mandatory) and same preset / `--cng` setting. Output WAV defaults to fp32 PCM in C; `AEC_OUT_FLOAT=0` for 16-bit PCM. |

---

## System architecture

```
   mic (near-end)                    ref (far-end)
        │                                  │
        ▼                                  │   (ref-path HPF OFF since v3.19)
   ┌──────────┐                            ▼
   │  HPF     │   (80 Hz, mic only)   ┌──────────────┐
   └────┬─────┘                       │  Saturation  │   (soft-clip ref)
        │                             └────┬─────────┘
        │                                  ▼
        │                             ┌──────────────┐
        │                             │  Delay est.  │   (online EchoPathDelay-
        │                             │  + ring buf  │    Estimator, periodic)
        │                             └────┬─────────┘
        └────────────┬────────────────────┘
                     ▼
              ┌──────────────┐
              │  PBFDKF      │   per-bin Kalman, two-stage Q
              │  main filter │
              └──────┬───────┘
                     │
            ┌────────┴────────┐
            ▼                 ▼
      ┌────────────┐    ┌──────────────────────┐
      │  Shadow    │    │ PathChangeRegime-     │  (echo-path-change +
      │  filter    │    │ Handler (EPC state)   │   shadow-copy state machine)
      └────────────┘    └──────────────────────┘
                     │
                     ▼   error spectrum E(f)
              ┌────────────────────────┐
              │  AEC3 post-filter      │  _aec3_post: AecState +
              │  (_aec3_post)          │  ResidualEchoEstimator (per-bin R²)
              │                        │  + SuppressionGain (ENR/EMR gain)
              └──────┬─────────────────┘  + CNG, OLA + sqrt-Hann
                     ▼
                clean output
```

The v3.21 pipeline retired the legacy 9-stage `ResFilter`; the post-filter is
now `AEC._aec3_post` driving the AEC3-aligned chain (`modules/state`,
`modules/residual`, `modules/render`, `modules/delay`). Algorithm details →
[docs/aec_methods.md](docs/aec_methods.md). Historical PBFDKF/shadow parameter
tables are not current default contracts.

### Future direction — NN post-filter after the linear AEC

The pipeline splits into a **linear stage** (HPF → saturation → delay-est →
PBFDKF + shadow) and the **AEC3 post-filter** (`_aec3_post`: ResidualEcho-
Estimator + SuppressionGain + CNG). Both have stable, freq-domain interfaces,
so a learned model can replace the post-filter — or the NR + RES jointly —
without touching the linear AEC or the front/back transform:

```
              linear AEC                          post-filter
mic ─►──────► [PBFDKF] ──► E(f) ──► [ _aec3_post / SuppressionGain ] ──► out
                              │
                              └─► (or NN-residual / NN-NR / NN-joint on E(f))
```

The seam is the per-frame `AecResContext` (set `AecConfig.return_res_context =
True` → `aec.process()` returns `(linear_out, AecResContext)`). It carries the
linear error spectrum, the per-frame echo estimate, and the AEC3 suppression
gain so a downstream stage runs entirely in the frequency domain:

| Field | Meaning |
|---|---|
| `formed_output` / `formed_hop` | Selected/crossfaded linear hop represented by `error_spec` |
| `error_spec` | E(f) — reconstructing 50%-overlap sqrt-Hann STFT of the formed linear output |
| `echo_spec`, `near_spec` | Matching windowed echo and capture spectra; `near_spec = error_spec + echo_spec` |
| `far_spec` | X(f) — PBFDKF render spectrum (shared reference/diagnostic coordinate) |
| `res_gain`, `comfort_noise` | the AEC3 SuppressionGain + CNG this frame |
| `erle_factor`, `dt_indicator`, `divergence`, `over_sub`, `erl_estimate` | per-frame telemetry |

**Lightweight variant — just the formed linear hop, no RES/CNG context.** A caller that only
wants `formed_output` (e.g. a dataset-gen tool building a "clean linear AEC error" channel,
with no interest in `error_spec`/`res_gain`/telemetry) doesn't need to opt into the full
`AecResContext` — `enable_res=False` with `return_res_context=True` still runs the entire
`_aec3_post` chain (`ResidualEchoEstimator` + `SuppressionGain` + CNG) purely to populate a
context most of whose fields go unread. Set `AecConfig.return_formed_output = True` instead:
`aec.process()`'s return shape/type is **unchanged** (still just `linear_out`, or `(linear_out,
AecResContext)` if `return_res_context` is *also* set), and after each `process()` call
`aec.get_formed_output()` returns the same value `AecResContext.formed_output` would have —
computed by running only the AEC3 `UseRefinedOutput`/`FormLinearFilterOutput` selection-and-
crossfade step, not the heavier gain/CNG chain around it. Independent of `enable_res`: it
does not skip RES/CNG when `enable_res=True` — only an additional value becomes readable.
Note this is the *selected and crossfaded* linear output, not the raw main-filter output,
which is why a consumer needing the linear residual must read it. Byte-identical to
`context.formed_output` in every configuration; see `python/tests/test_formed_output_seam.py`.

This seam is **already exercised** in the
[Audio_ALG](https://github.com/aaronhsueh0506/Audio_ALG) integration repo, whose
`AEC(linear) → echo-aware NR → RES` frequency-domain pipeline folds the residual-echo
PSD `R²` into NR's noise floor (`ξ = S²/(N²+R²)`) so **one** MMSE-LSA gain suppresses
noise + residual echo per-bin, plus a near-end floor that lifts the gain back toward 1
where there is no echo — a single FFT/IFFT for the whole chain, each stage independently
swappable for a neural model. The classical DSP pipeline stays in production / as the
deterministic fallback. NN-integration contract →
[docs/nn_integration_interface.md](docs/nn_integration_interface.md).

---

## Python tooling overview

| Script | Purpose |
|---|---|
| `python/aec.py` | Algorithm reference + single-case CLI. `python3 aec.py mic ref out --mode pbfdkf --preset balanced --enable-res [--cng] [--diag]`. `--diag` prints per-second ERLE / gain stats to stdout. |
| `python/run_one_case.py` | One-call render of a single mic/ref pair plus a 5-panel PNG (mic / ref / out waveforms, mic+out spectrograms, per-frame ERLE). Default writes `<out>.png`; pass `--no-plot` to skip. |
| `python/eval_aec_challenge.py` | Batch render on the AEC Challenge dataset. Supports `--all-presets`, alternative engines (`--aec3`, `--aec3-linear`, `--old-aec`, `--speex`), and parallel scenario processing. Writes `<stem>_ours.wav` etc. into the output dir. |
| `python/bench_aecmos.py` | AECMOS scoring of rendered outputs. Reads a directory of `<stem>_ours.wav`, writes `scores.json` + `result.md`. Optional `--baseline` produces Δ vs a saved scores.json. |
| `python/eval_manifest90.py` | Fixed 90-case partial blind benchmark with ERLE/SDR and scenario-correct AECMOS metrics. |
| `python/tests/` | Python regression tests for reset, grid timing, WOLA/context, and formed-output contracts. |

### Debug logging

| Implementation | How to enable | Output |
|---|---|---|
| Python (`python/aec.py`) | `--diag` on the CLI | per-second console line: ERLE / gain mean / shadow-advantage / DT indicator. For per-frame internal state, instantiate `AEC` programmatically and read `aec._diag` after each `aec.process(...)` call. |
| Python (`python/run_one_case.py`) | always — generates the diagnostic PNG | mic / ref / out waveforms + spectrograms + per-frame ERLE |
| C (`c_impl/bin/<backend>-<config-hash>/aec_wav`) | `--debug-level {0..3}`, optional `--debug-log <path>` | grep-friendly per-frame stderr (or file) lines: `[AEC][t=…s][f=…][PBFDKF] mu_mean=… P_mean=…`. Release builds (`-DNDEBUG`) strip log strings entirely. |

---

## Quick start

### Python

```bash
cd python
pip install numpy soundfile matplotlib

# BALANCED preset, PBFDKF + RES (recommended)
python3 aec.py mic.wav ref.wav out.wav --mode pbfdkf --preset balanced --enable-res

# Other presets (mild = near-priority, aggressive = echo-priority)
python3 aec.py mic.wav ref.wav out.wav --mode pbfdkf --preset {mild|balanced|aggressive} --enable-res

# CNG on
python3 aec.py mic.wav ref.wav out.wav --mode pbfdkf --preset balanced --enable-res --cng

# Per-frame diagnostics (per-second ERLE / gain summary)
python3 aec.py mic.wav ref.wav out.wav --mode pbfdkf --preset balanced --enable-res --diag

# Single case + diagnostic plot (mic/ref/out + spectrograms + ERLE-over-time PNG)
python3 ../python/run_one_case.py mic.wav ref.wav out.wav --preset balanced
```

### C

```bash
cd c_impl
make
BIN="$(make -s print-bin-dir)"

"$BIN"/aec_wav mic.wav ref.wav out.wav                  # BALANCED preset
"$BIN"/aec_wav mic.wav ref.wav out.wav --preset aggressive
"$BIN"/aec_wav mic.wav ref.wav out.wav --cng            # enable CNG (default off)
"$BIN"/aec_wav mic.wav ref.wav out.wav --no-res         # skip residual filter
"$BIN"/aec_wav mic.wav ref.wav out.wav --debug-level 2  # per-frame log
```

C usage / API / integration details
→ [docs/c_user_manual_zh_TW.md](docs/c_user_manual_zh_TW.md).

---

## Python API

```python
from aec import AEC, AecConfig, AecMode, AecPreset

# BALANCED preset (recommended)
config = AecConfig.from_preset(AecPreset.BALANCED,
                               sample_rate=16000,
                               mode=AecMode.PBFDKF,
                               enable_res=True)
aec = AEC(config)

hop = aec.hop_size
while has_audio:
    out = aec.process(mic_block, ref_block)
    erle = aec.get_erle()
```

Preset trade-off:

| Preset | floor | Echo suppression | NE preservation | Use |
|---|---|---|---|---|
| `mild`     | −20 dB | light  | best (near-priority) | NE 過度被壓時、demo / 試聽 |
| `balanced` ★ | −28 dB | medium | good (all ship bars) | general / production default |
| `aggressive` | −38 dB | strong | minor loss (echo-priority) | hands-free, speakerphone, high echo |

Full parameter reference → [docs/aec_methods.md §6 (Configuration)](docs/aec_methods.md).

---

## Benchmark

```bash
# 1) Render Python AEC outputs (BALANCED preset, fl=52 ms, CNG on)
source .venv/bin/activate
python3 python/eval_aec_challenge.py wav/aec_challenge_blind/ \
    --preset balanced --filter 832 --cng -o out_python/

# 2) Score with AECMOS (needs speechmos + onnxruntime ≤1.16.3 + numpy<2)
python3 python/bench_aecmos.py out_python/ results/

# Compare against a saved baseline scores.json
python3 python/bench_aecmos.py out_python/ results/ --baseline /path/to/baseline_scores.json

# All presets at once (Python rendering only)
python3 python/eval_aec_challenge.py wav/aec_challenge_blind/ --all-presets

# C-side release gates (including real-WAV heap/pool byte equality)
make -C c_impl test-static-aec
make -C c_impl test-rate-structural test-process-context
```

---

## File layout

```
AEC/
├── README.md                    # this file
├── python/
│   ├── aec.py                       # algorithm + CLI (single-case process)
│   ├── run_one_case.py              # single-case render + diagnostic plot (PNG)
│   ├── eval_aec_challenge.py        # render outputs on the AEC Challenge dataset
│   ├── bench_aecmos.py              # AECMOS scoring of rendered outputs
│   ├── eval_manifest90.py            # fixed partial blind benchmark
│   └── tests/                        # Python regression tests
├── c_impl/                          # C implementation
│   ├── README.md                    # short landing → docs/c_user_manual_zh_TW.md
│   ├── include/                     # public headers
│   ├── src/                         # sources
│   ├── example/
│   │   ├── aec_wav.c                # CLI entry point
│   │   └── wav_io.h
│   ├── test/modules/                # per-module test harnesses (dev tooling)
│   └── Makefile
├── docs/
│   ├── aec_methods.md                   # canonical algorithm spec (v3.21)
│   ├── archive/                         # historical reports, presentations, and retired parameter guides
│   ├── development_guide.md              # contributor invariants and validation commands
│   ├── nn_integration_interface.md       # NN residual/NR/joint freq-domain seam
│   └── c_user_manual_zh_TW.md            # canonical C user manual (API + CLI appendix)
├── eval/                            # fixed benchmark manifest + builder
└── wav/                             # local/downloaded test audio (mostly ignored)
```

---

## References

1. Haykin, S. — *Adaptive Filter Theory*
2. Benesty, J. et al. — *Advances in Network and Acoustic Echo Cancellation*
3. Enzner, G. et al. — *Frequency-domain adaptive Kalman filter for acoustic echo control* (2006)
4. WebRTC AEC3 source — https://webrtc.googlesource.com/src
