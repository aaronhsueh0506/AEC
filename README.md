# AEC — Acoustic Echo Cancellation

> **C 使用與整合**：[AEC C User Manual（繁體中文）](docs/c_user_manual_zh_TW.md)
> · HTML 頁面：[呼叫使用](docs/html/aec_usage.html) · [流程時序圖](docs/html/aec_sequence.html) ·
> [外部參數](docs/html/aec_params.html) · [內部參數](docs/html/aec_params_internal.html)

Single-channel AEC (1 mic + 1 ref) supporting PBFDKF (frequency-domain Kalman),
multi-ERLE, shadow filter, and post-filter residual echo suppression.
Python reference implementation + C implementation.

**Release**: v5.0.0rc1 (`__version__` in `python/aec.py`). **Not ship-ready**:
the open items are the `[4.0.0]` "Known limitations" in
[CHANGELOG.md](CHANGELOG.md), which also holds the version history and the
measurements behind each default.

The production algorithm is the AEC3-aligned `_aec3_post` chain (AecState +
ResidualEchoEstimator + SuppressionGain + CNG) behind the PBFDKF linear stage,
with a split min-gain floor for near-end preservation and a default-ON
double-talk recovery stack (`dt_aware_recovery_soft` +
`dt_aware_res_floor_enabled`, `min_gain_floor_dt_db = −20`). Four Pareto
presets — `mild` / `balanced` / `aggressive` / `no_floor` — share a two-field
strength axis (see "Presets" below); **`balanced` is production**.

**Numeric contract.** Production C is float32 end-to-end. The Python reference
(fp64) is the **algorithm spec**, C is the float32 **implementation**, and
Python↔C comparison is **tolerance-based**, not bit-exact
(`c_impl/test/parity_aec_e2e.c`, 2e-2 at 16 kHz). The **FFT backend is NE10
(float32, ARM NEON)** for both the host build (`make`, malloc) and the embedded
deployment (caller pool via `aec_get_mem_size`/`aec_init`); the pool path is
byte-equal to the malloc path. `SIMD=0` and `SIMD=1` are likewise not
bit-identical: the matched filter's dot product (`c_impl/src/delay_aec.c`)
uses four accumulators plus `vfmaq_f32` (reordered summation *and* fused
rounding), and its NLMS update fuses where the scalar path spells a separate
multiply and add. Byte-exact comparison is therefore only meaningful within
one build configuration.

---

## Status snapshot

**800-case AECMOS — ours vs WebRTC AEC3 vs Speex (MDF).** Full AEC Challenge
2021 blind corpus (echo↑ deg↑), `balanced / fl=832 (52 ms) / --cng`, 16 kHz
256/128 grid (the default), official per-bucket AECMOS segmentation, local
AECMOS ONNX; measured build: commit `02df7cd`. **No pre-alignment** — every
engine self-aligns online (ours via its in-pipeline matched-filter
`EchoPathDelayEstimator`; AEC3 via its internal `RenderDelayController`;
Speex internally), exactly as in production.

| Bucket | n | **ours** echo / deg | AEC3 echo / deg | Speex echo / deg |
|---|---|---|---|---|
| FS_static   | 169 | 3.758 / 4.991 | 4.037 / 4.999 | 2.963 / 5.000 |
| FS_movement | 131 | 3.733 / 4.999 | 4.046 / 4.999 | 2.806 / 5.000 |
| DT_static   | 186 | 4.037 / 3.418 | 4.281 / 2.967 | 3.145 / 3.923 |
| DT_movement | 114 | 3.838 / 3.385 | 4.150 / 2.931 | 2.916 / 3.874 |
| NE          | 200 | 4.998 / 3.969 | 4.999 / 3.410 | 4.998 / 4.128 |

**Echo cancellation: AEC3 > ours > Speex** (ours trails AEC3 by 0.24–0.31,
leads Speex by 0.80–0.93). **Near-end preservation (deg): Speex > ours > AEC3**
on DT/NE (ours beats AEC3 by 0.45–0.56), with FS deg a near-ceiling tie across
all three engines (within 0.009 of 5.0). Caveat: Speex ran at its
script-hardcoded `fl=2048` taps (no `--speex-fl` flag exists), not the
832-tap/52 ms grid ours and AEC3 use — same corpus and method, but not a
tail-length-matched comparison.

### Presets — a strength axis, four operating points

The Python API and CLIs call this axis the *preset*; the C API and C CLIs call it the *level* (`AecLevel`, `AEC_LEVEL_*`, `--level`): **C level == Python preset**, with the same names and the same numbers 0..3 (default `1` = balanced). The old C spellings `AecPreset` / `AEC_PRESET_*` stay in `aec.h`, and `--preset` stays an alias of `--level` in the C CLIs.

`mild` / `balanced` differ **only** in `min_gain_floor_far_active_db`, the
far-active residual-gain floor that trades echo suppression against near-end
preservation. `aggressive` also deepens `min_gain_floor_dt_db` (the floor
used in its place during double-talk) — without it, `dt_protect_active`
hard-overrides the far-active floor with the SAME shared −20 dB for every
preset whenever it fires (it also false-arms in far-end single talk), masking
most of the strength-axis difference on exactly the hops where a user would
expect to hear it. `no_floor` is a fourth tier that removes the floor
mechanism entirely (`min_gain_split_floor_enabled=False`) instead of
deepening it further:

| Preset (C: level) | far-active floor | dt floor | character |
|---|---|---|---|
| `mild`       | −20 dB | −20 dB | near-priority — more near-end kept, more echo leak |
| **`balanced` ★** | −28 dB | −20 dB | **production** |
| `aggressive`   | −38 dB | −30 dB | echo-priority — deeper suppression, more near-end loss; dev-pool (47-case) DT_static echo +0.17/+0.14 MOS vs a shared-dt-floor aggressive, at a bounded near-end cost |
| `no_floor`   | inert (gate off) | inert (gate off) | deepest point on the axis — purely audibility/R²-derived gain; most near-end loss. `AecPreset.NO_FLOOR` / `AEC_LEVEL_NO_FLOOR` (C enum value `3`); the dB columns are never read once `min_gain_split_floor_enabled` is `False` |

`mild` / `aggressive` / `no_floor` are deliberate Pareto operating points on the
single-channel DT-deg-vs-echo trade-off; all share the same `_aec3_post` chain
and base tuning.

#### Retargeting a running instance

The preset (C: level) can be changed on a live instance without rebuilding it — the
filter, the delay lock and every smoothing history carry on. `no_floor`
changes a boolean (`min_gain_split_floor_enabled`), which the ramp below
cannot represent, so it STEPS immediately instead of ramping, as the DT floor
does. Both C (`aec_set_level()` / `aec_res_set_level()`) and Python
(`AEC.set_preset()`) allow switching to or from `no_floor` with no
accept/reject guard: the gate steps on the same call that writes the
far-active and dt floors, so a round trip through `no_floor` always lands on
the target preset's exact values. Construction-time `no_floor`
(`aec_config_from_level()` / `AecConfig.from_preset()`) and runtime
retargeting behave the same in both languages.

```c
int aec_set_level(Aec* a, AecLevel level, float ramp_ms);   /* aec.h */
```

`ramp_ms == 0` applies the far-active floor on the next hop (not an error) and
lands on exactly the floor a fresh instance would hold; a positive value walks
there linearly in dB, bounded at 60 s — worth using, since mild ↔ aggressive is
an 18 dB step into a
hard clamp. Call between hops, serialised with processing; not thread-safe.
Returns `0`, or `-1` with nothing written on NULL / out-of-enum level /
out-of-range `ramp_ms`. Unlike `aec_config_from_level()`, which falls back to
balanced for an unrecognised value, this **refuses** — a setter has a caller to
tell.

The primitives underneath — `suppression_gain_set_split_floor_far_active_db()`
(ramped), `suppression_gain_set_split_floor_dt_db()` (steps immediately —
the DT floor is a hard per-hop override, not a value the gain rule converges
toward, so `ramp_ms` does not apply to it), and
`suppression_gain_set_split_floor_enabled()` (steps immediately — a boolean
has no continuous path to ramp along; gates all three floor variants
together: far-active, far-silent/pure-NE, and DT-lifted) — are internal; a
caller that owns a post-stage suppressor outside any `Aec` (the four-channel
product) uses `aec_res_set_level()` from `aec_res.h`, same semantics.
`aec_level_split_floor_enabled(AecLevel, int*)` is the public lookup for
the gate's value alone, mirroring `aec_level_floor_db()` /
`aec_level_dt_floor_db()`, for a caller that wants the gate without the dB
lookups.

Two things to know before measuring a level change: lanes built with
`spatial_linear_context` never reach the suppressor's gain path, so retargeting
them is inert by construction (a 4-channel product retargets the shared
`AecRes` that the application itself created after the Audio_ALG mcab flow:
`aec_res_set_level(res, level, ramp_ms)`); and the far-active floor only binds on far-active,
non-double-talk hops while the same gain also sets the comfort-noise level, so a
whole-recording average moves less than the dB step suggests. Full contract and
A/B guidance → [docs/c_user_manual_zh_TW_03_level.md §7.1](docs/c_user_manual_zh_TW_03_level.md#71-執行期切換-levelaec_set_level).

---

## Specifications & limits

### Input

| Field | Value |
|---|---|
| Sample rate | 8 / 16 / 48 kHz (whitelisted no-padding grids); 16 / 48 kHz are the product grids |
| Bit depth | 16-bit PCM or 32-bit float |
| Channels | mono |
| Frame / hop | 8k: 256/128; 16k: 256/128 default or 512/256; 48k: 1024/512 |
| Filter length | 52 ms @ 8/16 kHz; 64 ms @ 48 kHz (configurable) |
| Algorithmic delay | one hop with RES: 16 ms / 8 ms / 10.667 ms by grid |

### Resource (C, 16 kHz default grid 256/128, fl=52 ms)

| | |
|---|---|
| Static pool | Backend、grid、filter 長度與功能開關都會影響大小；部署時以 `aec_get_mem_size()` 查詢，不使用文件中的固定常數 |
| Compute / frame | Kalman update over 129 bins × 7 partitions |
| FFT                                | NE10 ARM-NEON FFT (float32) — ~float32 precision vs numpy `np.fft` |

The static pool path is byte-equal to the malloc path (`test_static_aec`); full
per-region breakdown → [c_impl/STATIC_MEMORY.md](c_impl/STATIC_MEMORY.md).

### Known limitations (cases this AEC cannot fully solve)

| Limitation | Detail |
|---|---|
| Supported SR              | 8 / 16 / 48 kHz whitelisted no-padding grids; all four (8k 256/128, 16k 256/128, 16k 512/256, 48k 1024/512) are covered by the C/Python timing-parity gate. Product validation targets 16 / 48 kHz: AECMOS is only meaningful at 16 kHz, and 8 kHz is a legacy standalone grid. Other rates, including 32 kHz / 44.1 kHz, need external resampling. |
| Channel count             | mono only — no mic array / stereo reference |
| Echo nonlinearity         | linear filter + RES; no dedicated nonlinear model. High speaker distortion produces residual harmonic leakage |
| Delay direction           | positive only (mic lags ref). Negative-delay scenarios must be aligned upstream |
| Stationary far + weak NE  | linear AEC may absorb very low-energy NE syllables into the echo path. Stationary-DT hangover mitigates but cannot fully restore worst frames |
| DT-from-frame-0           | NE present from sample 0 prevents the filter from ever converging |
| Echo-path > filter length | echo tail beyond `filter_length` (default 52 ms worth of samples) is not modelled. Larger rooms need a longer filter (cost: memory + compute). C: set `AecConfig.filter_length` (no `aec_wav` flag); Python CLI: `--filter <samples>` |
| Mid-stream SR/filter change | not supported. `filter_length` is fixed at construction; SR change requires destroy + re-create |
| Mic/ref clock drift       | sustained sample-rate drift between mic and ref tanks ERLE (filter cannot track) |

### Common problems & adjustments

Both Python (`aec.py`) and C (`c_impl`'s `aec_wav`) use the same algorithm; the
adjustments below apply to both unless noted. CLI flag examples shown for
C; the Python CLIs have no `--level` and spell the same axis `--preset` (`--level aggressive`
in C is `--preset aggressive` in Python), and otherwise differ only in syntax (`--mode pbfdkf` etc.).

| Symptom | Diagnosis & adjustment |
|---|---|
| **Residual echo too high (FS / NE)** | 1. With `--no-res`, output should be a clean linear-AEC residual. Echo still dominating → ref signal is wrong, mic-ref delay > filter length, or sample rates differ. 2. `--level aggressive` for stronger RES (cost: more NE compression). 3. For big rooms, increase `AecConfig.filter_length` through the C API before `aec_get_mem_size()`/`aec_init()` and recreate the instance. The value is in samples; there is no `--filter-length` or `--filter-length-ms` CLI flag. |
| **NE clipped during double-talk** | Lower level → `--level mild` (−20 dB floor, near-priority: keeps more near-end at the cost of more echo leak). Don't tweak individual RES knobs — level values are co-tuned. |
| **Slow startup / first-second echo** | Filter convergence needs ≥ 0.5 s of meaningful far energy. Normal adaptive behavior. Consider muting output during application warm-up (e.g. play a "connecting…" cue). |
| **Echo spikes when device moves** | Echo path changes → EPC fires → ~200 ms re-convergence with brief leak. Usually self-recovers. For frequent movement, increase filter length. |
| **Output sounds muffled / pumping in NE-only** | Comfort noise mismatch. CNG is on by default in C (`--no-cng` disables) and with any Python preset (`--cng` / `--no-cng` override); check it is enabled, and do NOT stack a second CNG layer downstream. |
| **First file sounds OK but second file glitches in batch processing** | Missed `aec_reset` between files. DT / EPC / convergence state accumulates. Reset before each independent stream. |
| **Linear-AEC residual sounds wrong (separate filter from RES)** | `"$BIN"/aec_wav mic ref linear_only.wav --no-res` after resolving `BIN="$(make -s print-bin-dir)"` in `c_impl/` (or use the Python `--no-res` equivalent). This isolates filter-side issues from RES-side issues. |
| **Per-frame state inspection** | C: `--debug-trace trace.csv` (one CSV row per hop: `usable_linear`, `fullband_erle`, `r2_mean`, `gain_mean`, …). Python: `python3 aec.py mic ref out --diag`. |
| **Detect mic/ref drift or wrong delay** | Run with `--no-delay-est`, supply pre-aligned files, compare output vs the online-delay-est version. Large divergence → drift, or a delay beyond the acquirable ceiling (1216 / 608 / 608 ms at 8 / 16 / 48 kHz — compile-time, not raised by `max_delay_ms`; see the C manual §3). |
| **Build mismatch between Python and C output** | Verify C built with `-ffp-contract=off` (mandatory) and same level (Python: preset) / `--cng` setting. Output WAV defaults to fp32 PCM in C; `AEC_OUT_FLOAT=0` for 16-bit PCM. |

---

## System architecture

```
   mic (near-end)                    ref (far-end)
        │                                  │
        ▼                                  │
   ┌──────────┐                            ▼
   │  HPF     │   (80 Hz, mic only)   ┌──────────────┐
   └────┬─────┘                       │  Saturation  │   (soft-clip ref)
        │                             └────┬─────────┘
        │                                  ▼
        │                             ┌──────────────┐
        │                             │  Delay est.  │   (online EchoPathDelay-
        │                             │  + ring buf  │    Estimator)
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

The post-filter is `AEC._aec3_post`, which drives the AEC3-aligned chain
(`modules/state`, `modules/residual`, `modules/render`, `modules/delay`).
Algorithm details → [docs/aec_methods.md](docs/aec_methods.md).

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
| `erle_factor`, `dt_indicator`, `divergence`, `over_sub`, `erl_estimate` | per-frame telemetry (`over_sub` is a Python-only field, equal to `erle_factor`; the C `AecResContext` has no `over_sub`) |
| `res_floor_protect` | this frame's near-recent-latch DT floor decision; a fused consumer that wants it must combine the contributing lanes' flags itself (Audio_ALG's `mcab_fuse()` does not read it) |
| `usable_linear` | this frame's usable-linear-estimate verdict; `False` means `r2` is the conservative nonlinear estimate (far-end power times a default echo-path gain), not one derived from the linear echo estimate — see [`docs/aec_methods_02_post_filter.md` §3.4](docs/aec_methods_02_post_filter.md#34-suppressiongain) |

**Reading the formed linear hop without building the full `AecResContext`.** A caller that only
wants `formed_output` (e.g. a dataset-gen tool building a "clean linear AEC error" channel, with
no interest in `error_spec`/`res_gain`/telemetry) doesn't need to read the whole context object.
Set `AecConfig.return_formed_output = True` **together with** `enable_res=True` or
`return_res_context=True`: `aec.process()`'s return shape/type is **unchanged** (still just
`linear_out`, or `(linear_out, AecResContext)` if `return_res_context` is *also* set), and after
each `process()` call `aec.get_formed_output()` returns the same value
`AecResContext.formed_output` would have.

This reads back a value the post chain already computed for its own gate — it is **not** a
lighter path that skips that chain. `AecConfig` raises `ValueError` at construction if
`return_formed_output` is set while both `enable_res` and `return_res_context` are false.

Note this is the *selected and crossfaded* linear output, not the raw main-filter output — and in
context-only mode (`enable_res=False`, `return_res_context=True`) the selection can be the capture
itself on a hop the filtering-quality analyzer has not cleared, see §3 of
[docs/aec_methods_02_post_filter.md](docs/aec_methods_02_post_filter.md#3-aec3-post-filter-_aec3_post). Byte-identical to `context.formed_output` in every
supported configuration; see `python/tests/test_formed_output_seam.py`.

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
| C (`c_impl/bin/<backend>-<config-hash>/aec_wav`) | `--debug-trace <path>`, `--debug` | `--debug-trace`: one CSV row per hop (columns in `c_impl/include/aec_debug.h`); `--debug`: one status line per second |

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
python3 run_one_case.py mic.wav ref.wav out.wav --preset balanced
```

### C

```bash
cd c_impl
make
BIN="$(make -s print-bin-dir)"

"$BIN"/aec_wav mic.wav ref.wav out.wav                  # BALANCED level
"$BIN"/aec_wav mic.wav ref.wav out.wav --level 2        # 0 mild | 1 balanced | 2 aggressive | 3 no_floor (names also accepted)
"$BIN"/aec_wav mic.wav ref.wav out.wav --no-cng         # disable CNG (default on)
"$BIN"/aec_wav mic.wav ref.wav out.wav --no-res         # skip residual filter
"$BIN"/aec_wav mic.wav ref.wav out.wav --debug-trace trace.csv  # per-hop CSV trace
```

C usage / API / integration details
→ [docs/c_user_manual_zh_TW.md](docs/c_user_manual_zh_TW.md); call order, sequence diagrams and the
external-vs-internal parameter tables are on the HTML pages under [docs/html/](docs/html/aec_usage.html).

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

# Retarget the strength axis on the running instance (C twin: aec_set_level).
# ramp_ms=0 applies next hop; a positive value walks there linearly in dB.
aec.set_preset(AecPreset.AGGRESSIVE, ramp_ms=100.0)
```

`set_preset()` raises `ValueError` on an unknown preset or an out-of-range
`ramp_ms`, leaving the instance untouched, where `AecConfig.from_preset()` falls
back to balanced. Call between hops only.

Preset trade-off:

| Preset | floor | Echo suppression | NE preservation | Use |
|---|---|---|---|---|
| `mild`     | −20 dB | light  | best (near-priority) | NE 過度被壓時、demo / 試聽 |
| `balanced` ★ | −28 dB | medium | good | general / production default |
| `aggressive` | −38 dB | strong | minor loss (echo-priority) | hands-free, speakerphone, high echo |

Full parameter reference → [docs/aec_methods.md §1 (Tunable parameters)](docs/aec_methods.md).

---

## Benchmark

```bash
# 1) Render Python AEC outputs (BALANCED preset, fl=52 ms, CNG on)
python3 python/eval_aec_challenge.py wav/aec_challenge_blind/ \
    --preset balanced --filter 832 --cng -o out_python/

# 2) Score with AECMOS (needs onnxruntime, librosa and soundfile)
python3 python/bench_aecmos.py out_python/ results/

# Compare against a saved baseline scores.json
python3 python/bench_aecmos.py out_python/ results/ --baseline /path/to/baseline_scores.json

# All presets at once (Python rendering only)
python3 python/eval_aec_challenge.py wav/aec_challenge_blind/ --all-presets

# C-side test tiers. SIMD-independent members run only under SIMD=1, so the default tier is both runs
make -C c_impl test SIMD=1
make -C c_impl test SIMD=0
# adds the full parity matrix, the end-to-end gate and the slow script tests
make -C c_impl test-full SIMD=1

# single gates (including real-WAV heap/pool byte equality)
make -C c_impl test-static-aec
make -C c_impl test-rate-structural test-process-context test-avg-reverb-slots
```

---

## File layout

```
AEC/
├── README.md
├── CHANGELOG.md
├── python/
│   ├── aec.py                       # public re-exports + CLI entry point
│   ├── modules/                     # algorithm (config, orchestrator, delay/, state/, residual/, ...)
│   ├── run_one_case.py              # single-case render + diagnostic plot (PNG)
│   ├── eval_aec_challenge.py        # render outputs on the AEC Challenge dataset
│   ├── bench_aecmos.py              # AECMOS scoring of rendered outputs
│   ├── eval_manifest90.py           # fixed partial blind benchmark
│   ├── preset_arg.py                # shared --preset parser
│   ├── diag/                        # golden and timing-inventory generators
│   └── tests/                       # Python regression tests
├── c_impl/                          # C implementation
│   ├── README.md                    # build, FP policy, streaming, regression gates
│   ├── STATIC_MEMORY.md             # caller-owned pool contract
│   ├── include/                     # public headers
│   ├── src/                         # sources
│   ├── example/                     # aec_wav CLI, wav_io.h
│   ├── test/                        # test harnesses
│   └── Makefile
├── docs/
│   ├── aec_methods.md                   # algorithm specification (index + sections 1, 4-7)
│   ├── aec_methods_01_front_end.md      # algorithm specification, section 2 (front-end blocks)
│   ├── aec_methods_02_post_filter.md    # algorithm specification, section 3 (AEC3 post-filter)
│   ├── c_user_manual_zh_TW.md           # C user manual (index, sections 1-3)
│   ├── c_user_manual_zh_TW_01_lifecycle.md            # C user manual, sections 4-5 (lifecycle, error semantics)
│   ├── c_user_manual_zh_TW_02_config.md               # C user manual, section 6 (AecConfig reference)
│   ├── c_user_manual_zh_TW_03_level.md               # C user manual, section 7 (levels)
│   ├── c_user_manual_zh_TW_04_seam_troubleshooting.md # C user manual, sections 8-9 (integration seams, troubleshooting)
│   ├── c_user_manual_zh_TW_05_appendix.md             # C user manual, appendix A (aec_wav CLI)
│   ├── nn_integration_interface.md      # NN residual/NR/joint freq-domain seam
│   ├── development_guide.md             # contributor invariants and validation commands
│   ├── c_python_function_map.md         # C function <-> Python symbol map
│   ├── timing_constant_inventory.md     # generated hop-authored constant inventory
│   ├── delay_estimator_design_zh_TW.md  # delay estimator design
│   ├── delay_estimator_productization_plan_zh_TW.md
│   ├── linear_filter_evolution.svg      # linear-filter figure
│   ├── timing_constant_inventory.json   # source of timing_constant_inventory.md
│   └── html/                            # HTML pages (usage, sequence diagrams, parameters, API, RES stage)
├── model/                           # AECMOS ONNX model and wrapper
├── eval/                            # fixed benchmark manifest, A/B tooling
└── wav/                             # local/downloaded test audio (mostly ignored)
```

---

## References

1. Haykin, S. — *Adaptive Filter Theory*
2. Benesty, J. et al. — *Advances in Network and Acoustic Echo Cancellation*
3. Enzner, G. et al. — *Frequency-domain adaptive Kalman filter for acoustic echo control* (2006)
4. WebRTC AEC3 source — https://webrtc.googlesource.com/src
