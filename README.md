# AEC — Acoustic Echo Cancellation

Single-channel AEC (1 mic + 1 ref) supporting PBFDKF (frequency-domain Kalman),
multi-ERLE, shadow filter, and post-filter residual echo suppression.
Python reference implementation + C implementation.

**Release**: v3.21.0 (2026-05-19) — Python `aec.py` `__version__ = "3.21.0"`. C port lag follows. The v3.21 release retires the legacy `ResFilter` 9-stage chain in favour of the AEC3-aligned `_aec3_post` (AecState + ResidualEchoEstimator + SuppressionGain + CNG). Single production preset: `BALANCED`. The 5-preset table below is the historical v3.10.4 snapshot and is preserved for reference; v3.21 baseline scores live in [docs/bench/v3_21_3aadd2d_baseline/](docs/bench/v3_21_3aadd2d_baseline/README.md).

---

## Status snapshot

**5-preset operating points** (800-case AECMOS, AEC Challenge Interspeech
2021, fl=52 ms, cng=True). Aggregated (FS / DT echo = static + movement
mean):

| Preset | FS echo↑ | NE deg↑ | DT echo↑ | DT deg↑ |
|---|---|---|---|---|
| **MILD**        | 3.397 | **4.022** | 3.855 | **2.623** |
| SOFT            | 3.565 | 4.020 | 4.015 | 2.461 |
| BALANCED ★      | 3.668 | 4.010 | 4.154 | 2.344 |
| AGGRESSIVE      | 3.686 | 4.006 | 4.178 | 2.319 |
| MAXIMUM         | **3.746** | 3.993 | **4.218** | 2.265 |
| WebRTC AEC2     | 3.488 | 4.098 | 4.240 | 2.416 |

Per-bucket breakdown:

| Preset | FS_st | FS_mv | NE | DT_st echo | DT_st deg | DT_mv echo | DT_mv deg |
|---|---|---|---|---|---|---|---|
| **MILD**        | 3.332 | 3.480 | **4.022** | 3.888 | 2.632 | 3.802 | **2.608** |
| SOFT            | 3.504 | 3.643 | 4.020     | 4.069 | 2.453 | 3.926 | 2.474     |
| BALANCED ★      | 3.641 | 3.704 | 4.010     | 4.217 | 2.328 | 4.051 | 2.370     |
| AGGRESSIVE      | 3.676 | 3.699 | 4.006     | 4.242 | 2.297 | 4.073 | 2.355     |
| MAXIMUM         | **3.748** | **3.743** | 3.993 | **4.285** | **2.240** | **4.109** | 2.307 |

> v3.8.3 (2026-05-05) shifted the gentle end of the preset ladder:
> new MILD is an ultra-light minimum-touch preset (re-bench'd 2026-05-05);
> SOFT (\*) inherits the former v3.8.2 MILD numbers verbatim (params
> unchanged); BALANCED+ unchanged from the 2026-05-01 baseline. NE deg
> ≈ 4.0 is a binding floor of the current architecture — every preset
> clusters in 3.986–4.022.

### What changed since v3.8.3

- **v3.8.4 — Plan A (HF preservation under DT)**: smoothing kernel
  `[0.25, 0.5, 0.25] → [0.1, 0.8, 0.1]` (stops low-band echo gain
  leaking into HF bins, ~10 dB cut measured 4–8 kHz); HF cap anchor
  500 Hz → 2 kHz (vowel formants 1–3 kHz preserved); DT gate
  `effective_dt < 0.5 → < 0.3`; cap skipped when high-band shows NE
  evidence; `_stat_dt_mask` extended 4 → 7 kHz with linear fade.
- **v3.10.0 — Delay + Recovery (WebRTC AEC3 alignment)**: DelayEstimator
  `max_delay_ms` 250 → 512 + `confidence` / `is_solid` properties;
  render ring buffer 1024 ms; two-path delay-acquisition gate
  (acquisition vs shift), `mu_scale` delay-confidence ceiling;
  `FilterPlateauDetector` (resets filter taps when ERLE stuck low for
  50 frames during DT); `ResidualEchoEstimator` long-window far-PSD
  EMA (alpha=0.993).
- **v3.10.1 — Long-window EMA refinements**: EMA updates every
  far-active frame regardless of mode; render-based fallback blends
  70 % long-window + 30 % instantaneous, warmup-gated.
- **v3.10.2 — Codex round 2**: split delay gates into truly independent
  ifs; shared `_reset_filter_derived_state(reason, preserve_render_ema)`
  helper covers plateau and delay-first reset paths.
- **v3.10.3 — Codex round 3 + self-trace fixes**: helper now also
  resets `near_power` EMA (F3); plateau detector uses current-frame
  `dt_signal_present` (F4); `_pending_delay` cleared on reset (F5);
  `__version__` bump (F7); `_pending_delay` TTL = 3 cycles (H1);
  `mu_scale` ceiling skipped during post-reset warmup (H2); plateau
  detector resets cumulative counters on fire (H3); Path B delay
  shift now calls helper (M4).
- **v3.10.4 — Wider delay range + CLI fix**: `max_delay_ms`
  512 → 1024 (matches WebRTC's older AEC ~1 s far-end history;
  AEC3's 512 ms misses BT/mobile skew); render buffer 1024 → 2048 ms
  for headroom. `aec.py` CLI: `--enable-res` / `--cng` use
  `argparse.BooleanOptionalAction` with `default=None` so preset
  values are no longer silently overridden when the user doesn't
  pass the flag (F6).
- **Bench (BALANCED, 800-case)**: v3.10.4 FS 3.668 / NE 4.010 /
  DT echo 4.154 / DT deg 2.344. The −0.130 FS regression vs v3.8.3
  baseline is the locked-in cost of Plan A's smoothing kernel
  change — see CHANGELOG for details and known trade-offs.

Algorithm version history → [docs/CHANGELOG.md](docs/CHANGELOG.md).
Trace-driven evolution (v3.0–v3.4 design rationale) → [docs/aec_v3_evolution.md](docs/aec_v3_evolution.md).
HF preservation research → [docs/research_log_v3.9.x_HF_preservation.md](docs/research_log_v3.9.x_HF_preservation.md).

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
| **NE clipped during double-talk** | Lower preset → `--preset soft` first (= former v3.8.2 mild, light RES with audible echo cleanup), then `--preset mild` for minimum-touch RES. Don't tweak individual RES knobs — preset values are co-tuned. |
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

### Future direction — NN post-filter after the linear AEC

The pipeline above is split into a **linear stage** (HPF → saturation →
delay-est → PBFDKF + shadow) and a **post-filter stage** (ResFilter:
ENR + spectral floor + reverb tail + CNG). Both stages have stable,
well-understood interfaces:

```
                          linear AEC                  post-filter
mic ─►───────────────────► [PBFDKF] ──► error ──► [ ResFilter ] ──► out
                                            │
                                            └─► (or NN model)
```

The classical RES has hit its score plateau on the AEC Challenge 2021
blind set (R10–R16 investigations confirmed; see
[SUMMARY.md](docs/SUMMARY.md)). The next architectural step is to
**replace or complement ResFilter with a neural post-filter** taking
the same inputs the classical RES uses today:

| Input to NN model | Source | Frame |
|---|---|---|
| `error` (linear AEC out) | `aec.process()` raw output | hop |
| `echo_spec` (filter echo estimate) | `AecResContext.echo_spec` | n_freqs complex |
| `near_spec`, `far_spec` | mic / ref FFT | n_freqs complex |
| Detector telemetry (ERLE, dt_indicator, divergence, erle_factor, etc.) | `AecStats` / `AecResContext` | scalar/frame |

The hooks are already in place:

- **Python**: set `AecConfig.return_res_context = True` and
  `aec.process()` returns `(linear_out, AecResContext)`. Feed the
  context into a downstream model and skip the built-in ResFilter
  with `enable_res = False`.
- **C**: same plumbing exists in `c_impl/include/aec.h` —
  `aec_process` can be split into `linear_step` + external RES via
  exposing `AecResContext` (port from Python is the next step on the
  Audio_ALG integration repo).

Candidate models (joint NR + RES + dereverb) and the roadmap for the
NN replacement are in the planning notes; the classical pipeline stays
in production until the NN reaches parity on near-end preservation
(NE deg ≥ 4.0).

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
python3 aec.py mic.wav ref.wav out.wav --mode pbfdkf --preset {mild|soft|aggressive|maximum} --enable-res

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
| MILD       | very light | best (minimum-touch) | echo 安靜、demo / 試聽 |
| SOFT       | light  | very good | 一般通話但 BALANCED 過度壓 NE |
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
