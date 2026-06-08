# AEC — Acoustic Echo Cancellation

Single-channel AEC (1 mic + 1 ref) supporting PBFDKF (frequency-domain Kalman),
multi-ERLE, shadow filter, and post-filter residual echo suppression.
Python reference implementation + C implementation.

**Release**: v3.22.5 (2026-06-07) — Python `aec.py` `__version__ = "3.22.5"`; C port bit-exact (peak |Δ| = 0). The production algorithm is the v3.21 AEC3-aligned `_aec3_post` chain (AecState + ResidualEchoEstimator + SuppressionGain + CNG) with the v3.22 split min-gain floor (DT/NE near-end preservation). v3.22.5 is a cleanup + docs release on the byte-equal **3.22.4** algorithm: the DTD subsystem and dead research flags were removed, the Python CLI now exposes all three presets, and a decoupled C streaming render/capture API shipped. Three Pareto presets — `gentle` / `balanced` / `aggressive` — differ only in the far-active min-gain floor; **`balanced` is production** and meets all four ship bars (FS echo >3.5, DT echo >4, DT deg >2, NE deg ≥4). See [CHANGELOG.md](CHANGELOG.md) `[3.22.5]` and [docs/v3_22_5_release.md](docs/v3_22_5_release.md).

---

## Status snapshot

**800-case AECMOS — ours vs WebRTC AEC3 vs Speex (MDF)**, standard bench
`balanced / fl=832 (52 ms) / --cng`, local AECMOS ONNX, full AEC Challenge
2021 blind corpus (echo↑ deg↑):

| Bucket | n | **ours** echo / deg | AEC3 echo / deg | Speex echo / deg |
|---|---|---|---|---|
| FS_static   | 169 | 3.576 / 4.999 | 3.821 / 4.999 | 2.847 / 5.000 |
| FS_movement | 131 | 3.512 / 4.999 | 3.790 / 4.999 | 2.757 / 5.000 |
| DT_static   | 186 | 4.201 / 2.156 | 4.531 / 1.815 | 3.427 / 3.179 |
| DT_movement | 114 | 4.082 / 2.228 | 4.456 / 1.816 | 3.272 / 3.301 |
| NE          | 200 | 4.998 / 4.047 | 4.999 / 3.410 | 4.998 / 4.128 |

- **Echo cancellation: AEC3 > ours > Speex.** AEC3 cuts the most; Speex is a
  weak canceller (FS ~2.8, DT ~3.3); ours sits in between (FS ~3.5, DT ~4.1).
- **Near-end preservation (deg): Speex > ours > AEC3.** AEC3 pays for its echo
  cancellation with the worst DT deg (1.82) and NE deg (3.41); ours holds the
  middle and **beats AEC3 on every degradation axis** (DT deg 2.16/2.23 vs 1.82,
  NE deg 4.05 vs 3.41) while **approaching AEC3 on echo** (FS_movement 3.512 vs
  3.790, within 0.28). This is the ship target: *approach AEC3 on echo, beat it
  on deg.* All four ship bars met; **FS_movement 3.512 > 3.5** is the primary gate.

### Presets — one Pareto knob

Three presets differ **only** in `min_gain_floor_far_active_db`, the far-active
residual-gain floor that trades echo suppression against near-end preservation:

| Preset | floor | character |
|---|---|---|
| `gentle`       | −20 dB | near-priority — more near-end kept, more echo leak (FS dips below the 3.5 bar by design) |
| **`balanced` ★** | −28 dB | **production** — all four ship bars met |
| `aggressive`   | −38 dB | echo-priority — deeper suppression, more near-end loss (deg stays >2.0, above AEC3) |

`gentle` / `aggressive` are deliberate Pareto operating points on the proven
single-channel DT-deg-vs-echo wall; all share the same `_aec3_post` chain and
800-case-tuned base. Full version history → [CHANGELOG.md](CHANGELOG.md).

---

## Specifications & limits

### Input

| Field | Value |
|---|---|
| Sample rate | 8 / 16 / 48 kHz (frame/fft/hop auto-derived) |
| Bit depth | 16-bit PCM or 32-bit float |
| Channels | mono |
| Frame / hop | 20 ms / 10 ms |
| Filter length | 52 ms default (configurable) |
| Algorithmic delay | 10 ms (1 hop) |

### Resource (C, fl=52 ms @16 kHz)

| | |
|---|---|
| Memory (main+shadow filter)        | ~200 KB |
| Memory (incl. RES + delay est.)    | ~280 KB |
| Compute / frame                    | 4 × 512-FFT + Kalman update (257 bins × 6 partitions) |
| FFT                                | fp64 radix-2 (matches numpy pocketfft fp64-internal precision) |

### Known limitations (cases this AEC cannot fully solve)

| Limitation | Detail |
|---|---|
| Supported SR              | 8 / 16 / 48 kHz only — 32 kHz / 44.1 kHz need external resample |
| Channel count             | mono only — no mic array / stereo reference |
| Echo nonlinearity         | linear filter + RES; no dedicated nonlinear model. High speaker distortion produces residual harmonic leakage |
| Delay direction           | positive only (mic lags ref). Negative-delay scenarios must be aligned upstream |
| Stationary far + weak NE  | linear AEC may absorb very low-energy NE syllables into the echo path. Stationary-DT hangover mitigates but cannot fully restore worst frames |
| DT-from-frame-0           | NE present from sample 0 prevents the filter from ever converging — see [docs/aec_methods.md 附錄 E](docs/aec_methods.md#附錄-e-dt-from-frame-0-限制) |
| Echo-path > filter length | echo tail beyond `filter_length_ms` (default 52 ms) is not modelled. Larger rooms need longer filter (cost: memory + compute) |
| Mid-stream SR/filter change | not supported. `filter_length_ms` is fixed at construction; SR change requires destroy + re-create |
| Mic/ref clock drift       | sustained sample-rate drift between mic and ref tanks ERLE (filter cannot track) |

### Common problems & adjustments

Both Python (`aec.py`) and C (`bin/aec_wav`) use the same algorithm; the
adjustments below apply to both unless noted. CLI flag examples shown for
C; equivalent Python flags differ only in syntax (`--mode pbfdkf` etc.).

| Symptom | Diagnosis & adjustment |
|---|---|
| **Residual echo too high (FS / NE)** | 1. With `--no-res`, output should be a clean linear-AEC residual. Echo still dominating → ref signal is wrong, mic-ref delay > filter length, or sample rates differ. 2. `--preset aggressive` for stronger RES (cost: more NE compression). 3. `--filter-length-ms 100` for big rooms / long reverb. |
| **NE clipped during double-talk** | Lower preset → `--preset gentle` (−20 dB floor, near-priority: keeps more near-end at the cost of more echo leak). Don't tweak individual RES knobs — preset values are co-tuned. |
| **Slow startup / first-second echo** | Filter convergence needs ≥ 0.5 s of meaningful far energy. Normal adaptive behavior. Consider muting output during application warm-up (e.g. play a "connecting…" cue). |
| **Echo spikes when device moves** | Echo path changes → EPC fires → ~200 ms re-convergence with brief leak. Usually self-recovers. For frequent movement, increase filter length. |
| **Output sounds muffled / pumping in NE-only** | Comfort noise mismatch. Try `--cng` (C) / `--enable-cng` (Python) to shape the noise floor; do NOT stack a second CNG layer downstream. |
| **First file sounds OK but second file glitches in batch processing** | Missed `aec_reset` between files. DT / EPC / convergence state accumulates. Reset before each independent stream. |
| **Linear-AEC residual sounds wrong (separate filter from RES)** | `./bin/aec_wav mic ref linear_only.wav --no-res` (or Python `--no-res` equivalent). Lets you isolate filter-side issues from RES-side issues. |
| **Per-frame state inspection** | C: `--debug-level 2 --debug-log /tmp/aec.log`, then `grep PBFDKF /tmp/aec.log`. Python: `python3 aec.py mic ref out --diag`. |
| **Detect mic/ref drift or wrong delay** | Run with `--no-delay-est`, supply pre-aligned files, compare output vs the online-delay-est version. Large divergence → drift or a delay outside `max_delay_ms` (default 1024 ms since v3.10.4; was 250 ms ≤ v3.8.3 / 512 ms in v3.10.0–v3.10.3). |
| **Build mismatch between Python and C output** | Verify C built with `-ffp-contract=off` (mandatory) and same preset / `--cng` setting. Output WAV defaults to fp32 PCM in C; `AEC_FP32_WAV=0` for 16-bit PCM. |

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
[docs/aec_methods.md](docs/aec_methods.md). PBFDKF / shadow overview →
[docs/pbfdkf_shadow_intro.md](docs/pbfdkf_shadow_intro.md).

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
| `error_spec` | E(f) — windowed linear AEC error spectrum (the "noisy" input) |
| `echo_spec` | Ŷ(f) — linear echo estimate (residual-echo reference) |
| `far_spec`, `near_spec` | X(f), mic spectrum |
| `res_gain`, `comfort_noise` | the AEC3 SuppressionGain + CNG this frame |
| `erle_factor`, `dt_indicator`, `divergence`, `over_sub`, `erl_estimate` | per-frame telemetry |

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
| `python/batch_c_eval.py` | Same dataset traversal as `eval_aec_challenge.py`, but invokes `c_impl/bin/aec_wav` instead of running the Python algorithm — used to render C outputs for AECMOS scoring. |

### Debug logging

| Implementation | How to enable | Output |
|---|---|---|
| Python (`python/aec.py`) | `--diag` on the CLI | per-second console line: ERLE / gain mean / shadow-advantage / DT indicator. For per-frame internal state, instantiate `AEC` programmatically and read `aec._diag` after each `aec.process(...)` call. |
| Python (`python/run_one_case.py`) | always — generates the diagnostic PNG | mic / ref / out waveforms + spectrograms + per-frame ERLE |
| C (`c_impl/bin/aec_wav`) | `--debug-level {0..3}`, optional `--debug-log <path>` | grep-friendly per-frame stderr (or file) lines: `[AEC][t=…s][f=…][PBFDKF] mu_mean=… P_mean=…`. Release builds (`-DNDEBUG`) strip log strings entirely. |

---

## Quick start

### Python

```bash
cd python
pip install numpy soundfile matplotlib

# BALANCED preset, PBFDKF + RES (recommended)
python3 aec.py mic.wav ref.wav out.wav --mode pbfdkf --preset balanced --enable-res

# Other presets (gentle = near-priority, aggressive = echo-priority)
python3 aec.py mic.wav ref.wav out.wav --mode pbfdkf --preset {gentle|balanced|aggressive} --enable-res

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

./bin/aec_wav mic.wav ref.wav out.wav                  # BALANCED preset
./bin/aec_wav mic.wav ref.wav out.wav --preset aggressive
./bin/aec_wav mic.wav ref.wav out.wav --cng            # enable CNG (default off)
./bin/aec_wav mic.wav ref.wav out.wav --no-res         # skip residual filter
./bin/aec_wav mic.wav ref.wav out.wav --debug-level 2  # per-frame log
```

C usage / API / integration details
→ [docs/c_user_and_integration_guide.md](docs/c_user_and_integration_guide.md).

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
| `gentle`     | −20 dB | light  | best (near-priority) | NE 過度被壓時、demo / 試聽 |
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

# C run on the same dataset
python3 python/batch_c_eval.py wav/aec_challenge_blind/ -o out_c/ --preset balanced
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
│   └── batch_c_eval.py              # run the C binary in batch on the same dataset
├── c_impl/                          # C implementation
│   ├── README.md                    # short landing → docs/c_user_and_integration_guide.md
│   ├── include/                     # public headers
│   ├── src/                         # sources
│   ├── example/
│   │   ├── aec_wav.c                # CLI entry point
│   │   └── wav_io.h
│   ├── test/modules/                # per-module test harnesses (dev tooling)
│   └── Makefile
├── docs/
│   ├── aec_methods.md                   # canonical algorithm spec (v3.21)
│   ├── aec_algorithm_guide.html         # presentation overview (v3.21)
│   ├── architecture_v3_22_5_vs_aec3.html  # current (v3.22.5) vs AEC3 architecture flowcharts
│   ├── refactor_modules_layout.md       # current module map (v3.21)
│   ├── pbfdkf_shadow_intro.md
│   ├── v3_22_5_release.md                # release summary + 3-way AEC3/Speex comparison
│   ├── nn_integration_interface.md       # NN residual/NR/joint freq-domain seam
│   ├── c_user_and_integration_guide.md  # canonical C CLI / API / integration guide
│   └── bench/v3_21_3aadd2d_baseline/    # 800-case AECMOS anchor + 25-case byte-equal reference
├── bin/                             # optional WebRTC benchmark helpers
│   ├── aec3_cli, aec3_linear_cli, old_aec_cli
└── wav/                             # test audio
```

---

## References

1. Haykin, S. — *Adaptive Filter Theory*
2. Benesty, J. et al. — *Advances in Network and Acoustic Echo Cancellation*
3. Enzner, G. et al. — *Frequency-domain adaptive Kalman filter for acoustic echo control* (2006)
4. WebRTC AEC3 source — https://webrtc.googlesource.com/src
