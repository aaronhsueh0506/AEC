# AEC — Acoustic Echo Cancellation

Single-channel AEC (1 mic + 1 ref) supporting PBFDKF (frequency-domain Kalman),
multi-ERLE, shadow filter, and post-filter residual echo suppression.
Python reference implementation + bit-exact-aligned C port.

**Current version**: v3.8.2 (2026-05-02)
**C/Python parity**: end-to-end SNR 64–78 dB across DT / FE / NE scenarios
(max-abs-diff ≈ 16-bit PCM quantization step). See
[c_impl/README.md](c_impl/README.md) for parity gate details.

---

## Status snapshot

**Python v3.8.1 BALANCED, 800-case AECMOS** (AEC Challenge Interspeech 2021,
fl=52ms, cng=True):

| | FS_st echo↑ | FS_mv echo↑ | NE deg↑ | DT_st echo↑ | DT_st deg↑ | DT_mv echo↑ | DT_mv deg↑ |
|---|---|---|---|---|---|---|---|
| **v3.8.1 BALANCED** | 3.803 | 3.865 | **4.002** | 4.257 | **2.256** | 4.147 | **2.270** |
| WebRTC AEC2 (ref)   | *3.457* | *3.519* | *4.098* | *4.331* | *2.304* | *4.149* | *2.528* |

**4-preset operating points** (v3.8.1, 2026-05-01 rebench):

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

### Known limitations

| | |
|---|---|
| Supported SR              | 8 / 16 / 48 kHz only (32 kHz needs external resample) |
| Channel count             | mono only |
| Echo nonlinearity         | linear filter + RES (no dedicated nonlinear model) |
| Delay direction           | positive only (mic lags ref). Negative-delay scenarios must be aligned upstream |
| Stationary far + weak NE  | linear AEC may absorb weak speech into echo path during stationary far-end + low-energy near-end. Hangover protection mitigates but cannot fully restore worst frames |
| DT-from-frame-0           | NE present from sample 0 prevents convergence — see [docs/spec_dt_from_frame_zero.md](docs/spec_dt_from_frame_zero.md) |

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

## Quick start

### Python

```bash
cd python
pip install numpy soundfile

# BALANCED preset, PBFDKF + RES (recommended)
python3 aec.py mic.wav ref.wav out.wav --mode pbfdkf --preset balanced --enable-res

# Other presets
python3 aec.py mic.wav ref.wav out.wav --mode pbfdkf --preset {mild|aggressive|maximum} --enable-res

# CNG on
python3 aec.py mic.wav ref.wav out.wav --mode pbfdkf --preset balanced --enable-res --cng

# Per-frame diagnostics
python3 aec.py mic.wav ref.wav out.wav --mode pbfdkf --preset balanced --enable-res --diag
```

### C (v2 — bit-exact-aligned with Python v3.8.1)

```bash
cd c_impl
make aec_wav_v2

./bin/aec_wav_v2 mic.wav ref.wav out.wav                  # BALANCED preset
./bin/aec_wav_v2 mic.wav ref.wav out.wav --preset maximum
./bin/aec_wav_v2 mic.wav ref.wav out.wav --cng            # enable CNG (default off)
./bin/aec_wav_v2 mic.wav ref.wav out.wav --no-res         # skip residual filter
./bin/aec_wav_v2 mic.wav ref.wav out.wav --debug-level 2  # per-frame log
```

C usage details → [c_impl/USER_GUIDE.md](c_impl/USER_GUIDE.md).
C integration into a C/C++ project → [docs/c_integration_guide.md](docs/c_integration_guide.md).

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
# AECMOS (needs speechmos + onnxruntime ≤1.16.3 + numpy<2)
source .venv/bin/activate
python3 python/eval_aecmos.py wav/aec_challenge/

# All presets
python3 python/eval_aecmos.py wav/aec_challenge/ --all-presets

# C run on dataset (parity with Python --enable-res)
python3 python/batch_c_eval.py wav/aec_challenge_blind/ -o out_c/ --preset balanced
```

---

## File layout

```
AEC/
├── README.md                    # this file
├── python/
│   ├── aec.py                   # reference implementation (v3.8.1)
│   ├── eval_aec_challenge.py
│   ├── eval_aecmos.py
│   ├── batch_c_eval.py
│   ├── dump_state.py            # per-frame state dump (parity tooling)
│   └── parity_export_*.py       # binary exports for C parity tests
├── c_impl/
│   ├── README.md                # C status + parity gate results
│   ├── USER_GUIDE.md            # C CLI / API usage
│   ├── include_v2/              # v2 headers (active port)
│   ├── src_v2/                  # v2 sources (bit-exact w/ Python)
│   ├── example/
│   │   ├── aec_wav_v2.c         # active CLI binary
│   │   └── main.c, aec_dump.c   # legacy v2.5 binaries
│   ├── test/parity/             # per-module parity tests
│   ├── lib/kiss_fft/            # legacy fft (kept for old binaries)
│   ├── include/, src/           # legacy v2.5 implementation (frozen)
│   └── Makefile
├── docs/
│   ├── CHANGELOG.md
│   ├── aec_methods.md           # complete algorithm reference
│   ├── aec_algorithm_guide.html # algorithm guide (HTML, with diagrams)
│   ├── aec_v3_evolution.md      # v3 trace-driven design rationale
│   ├── aec3_reference.md        # WebRTC AEC3 comparison notes
│   ├── pbfdkf_shadow_intro.md   # PBFDKF + shadow + DTD overview
│   ├── dtd_design.md            # DTD detailed design
│   ├── spec_dt_from_frame_zero.md
│   ├── c_integration_guide.md
│   └── archive/                 # superseded specs / change logs
└── wav/                         # test audio
```

---

## References

1. Haykin, S. — *Adaptive Filter Theory*
2. Benesty, J. et al. — *Advances in Network and Acoustic Echo Cancellation*
3. Enzner, G. et al. — *Frequency-domain adaptive Kalman filter for acoustic echo control* (2006)
4. WebRTC AEC3 source — https://webrtc.googlesource.com/src
