# AEC — Acoustic Echo Cancellation

Single-channel AEC (1 mic + 1 ref) supporting PBFDKF (frequency-domain Kalman),
multi-ERLE, shadow filter, and post-filter residual echo suppression.
Python reference implementation + C implementation.

**Release**: v3.8.2 (2026-05-02)

---

## Status snapshot

**BALANCED preset, 800-case AECMOS** (AEC Challenge Interspeech 2021,
fl=52 ms, cng=True). Aggregated (no movement split):

| | FS echo↑ | NE deg↑ | DT echo↑ | DT deg↑ |
|---|---|---|---|---|
| **AEC BALANCED** | **3.834** | **4.002** | **4.202** | **2.263** |
| WebRTC AEC2 (ref)| 3.488   | 4.098     | 4.240     | 2.416     |

Per-bucket breakdown (static vs movement):

| | FS_st echo↑ | FS_mv echo↑ | NE deg↑ | DT_st echo↑ | DT_st deg↑ | DT_mv echo↑ | DT_mv deg↑ |
|---|---|---|---|---|---|---|---|
| **AEC BALANCED** | 3.803 | 3.865 | **4.002** | 4.257 | **2.256** | 4.147 | **2.270** |
| WebRTC AEC2 (ref) | *3.457* | *3.519* | *4.098* | *4.331* | *2.304* | *4.149* | *2.528* |

**4-preset operating points** (2026-05-01 rebench). Aggregated:

| Preset | FS echo | NE deg | DT echo | DT deg |
|---|---|---|---|---|
| MILD | 3.678 | **4.013** | 4.048 | **2.379** |
| BALANCED | 3.834 | 4.002 | 4.202 | 2.263 |
| AGGRESSIVE | 3.849 | 3.997 | 4.226 | 2.230 |
| MAXIMUM | **3.898** | 3.986 | **4.263** | 2.189 |

Per-bucket breakdown:

| Preset | FS_st | FS_mv | NE | DT_st echo | DT_st deg | DT_mv echo | DT_mv deg |
|---|---|---|---|---|---|---|---|
| MILD | 3.624 | 3.732 | **4.013** | 4.084 | 2.366 | 4.012 | **2.391** |
| BALANCED | 3.803 | 3.865 | 4.002 | 4.257 | 2.256 | 4.147 | 2.270 |
| AGGRESSIVE | 3.830 | 3.868 | 3.997 | 4.284 | 2.222 | 4.167 | 2.237 |
| MAXIMUM | **3.889** | **3.906** | 3.986 | **4.320** | **2.186** | **4.206** | 2.192 |

Algorithm version history → [docs/CHANGELOG.md](docs/CHANGELOG.md).
Trace-driven evolution (v3.0–v3.4 design rationale) → [docs/aec_v3_evolution.md](docs/aec_v3_evolution.md).

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
| **Residual echo too high (FS / NE)** | 1. With `--no-res`, output should be a clean linear-AEC residual. Echo still dominating → ref signal is wrong, mic-ref delay > filter length, or sample rates differ. 2. `--preset aggressive` or `maximum` for stronger RES (cost: more NE compression). 3. `--filter-length-ms 100` for big rooms / long reverb. |
| **NE clipped during double-talk** | Lower preset → `--preset balanced` or `mild`. Don't tweak individual RES knobs — preset values are co-tuned. |
| **Slow startup / first-second echo** | Filter convergence needs ≥ 0.5 s of meaningful far energy. Normal adaptive behavior. Consider muting output during application warm-up (e.g. play a "connecting…" cue). |
| **Echo spikes when device moves** | Echo path changes → EPC fires → ~200 ms re-convergence with brief leak. Usually self-recovers. For frequent movement, increase filter length. |
| **Output sounds muffled / pumping in NE-only** | Comfort noise mismatch. Try `--cng` (C) / `--enable-cng` (Python) to shape the noise floor; do NOT stack a second CNG layer downstream. |
| **First file sounds OK but second file glitches in batch processing** | Missed `aec_reset` between files. DT / EPC / convergence state accumulates. Reset before each independent stream. |
| **Linear-AEC residual sounds wrong (separate filter from RES)** | `./bin/aec_wav mic ref linear_only.wav --no-res` (or Python `--no-res` equivalent). Lets you isolate filter-side issues from RES-side issues. |
| **Per-frame state inspection** | C: `--debug-level 2 --debug-log /tmp/aec.log`, then `grep PBFDKF /tmp/aec.log`. Python: `python3 aec.py mic ref out --diag`. |
| **Detect mic/ref drift or wrong delay** | Run with `--no-delay-est`, supply pre-aligned files, compare output vs the online-delay-est version. Large divergence → drift or a delay outside `max_delay_ms` (default 250 ms). |
| **Build mismatch between Python and C output** | Verify C built with `-ffp-contract=off` (mandatory) and same preset / `--cng` setting. Output WAV defaults to fp32 PCM in C; `AEC_FP32_WAV=0` for 16-bit PCM. |

---

## System architecture

```
   mic (near-end)                    ref (far-end)
        │                                  │
        ▼                                  ▼
   ┌──────────┐                       ┌──────────┐
   │  HPF     │                       │  HPF     │   (80 Hz Butterworth, both sides)
   └────┬─────┘                       └────┬─────┘
        │                                  ▼
        │                           ┌──────────────┐
        │                           │  Saturation  │   (soft-clip ref)
        │                           └────┬─────────┘
        │                                ▼
        │                           ┌──────────────┐
        │                           │  Delay est.  │   (GCC-PHAT, periodic)
        │                           │  + ring buf  │
        │                           └────┬─────────┘
        │                                │
        └────────────┬───────────────────┘
                     ▼
              ┌──────────────┐
              │  PBFDKF      │   per-bin Kalman, two-stage Q,
              │  main filter │   v3.7.0 G1 KX-blended P-update
              └──────┬───────┘
                     │
            ┌────────┴────────┐
            ▼                 ▼
      ┌────────────┐    ┌──────────────┐
      │  Shadow    │    │  EPC + Shadow│   (echo-path-change + shadow-copy
      │  filter    │    │  copy state  │    state machine)
      │  (Q×3.5)   │    └──────────────┘
      └────────────┘
                     │
                     ▼
              ┌──────────────────┐
              │  ResFilter       │   ENR-mask + reverb tail + min-stat
              │  (post-filter)   │   noise + CNG, OLA + sqrt-Hann
              └──────┬───────────┘
                     ▼
              ┌──────────────┐
              │  Limiter     │   smoothed gain clamp
              └──────┬───────┘
                     ▼
                clean output
```

Algorithm details → [docs/aec_methods.md](docs/aec_methods.md). PBFDKF/shadow/DTD
overview → [docs/pbfdkf_shadow_intro.md](docs/pbfdkf_shadow_intro.md). DTD design
→ [docs/dtd_design.md](docs/dtd_design.md).

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

# Other presets
python3 aec.py mic.wav ref.wav out.wav --mode pbfdkf --preset {mild|aggressive|maximum} --enable-res

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
./bin/aec_wav mic.wav ref.wav out.wav --preset maximum
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

| Preset | Echo suppression | NE preservation | Use |
|---|---|---|---|
| MILD       | light  | best  | conferencing, NE quality first |
| BALANCED   | medium | good  | general / default |
| AGGRESSIVE | strong | minor loss | hands-free, automotive |
| MAXIMUM    | max    | visible loss | speakerphone / very high echo |

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
│   ├── CHANGELOG.md
│   ├── SUMMARY.md                       # R1–R16 research log (canonical)
│   ├── aec_methods.md                   # canonical algorithm reference
│   ├── aec_algorithm_guide.html         # HTML snapshot; canonical = aec_methods.md
│   ├── aec_v3_evolution.md              # historical v3.0–v3.4 design notes
│   ├── aec3_reference.md                # WebRTC AEC3 comparison notes
│   ├── pbfdkf_shadow_intro.md
│   ├── dtd_design.md
│   ├── c_user_and_integration_guide.md  # canonical C CLI / API / integration guide
│   └── archive/                         # superseded specs / change logs
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
