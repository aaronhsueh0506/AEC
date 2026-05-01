# C AEC Integration Guide

**Target audience**: developers integrating the C AEC library into a larger pipeline (Audio_ALG, Novatek SDK, custom C/C++ project).

**Library version reference**: Python `aec.py` v3.8.1 (commit `3844169`). C is being rewritten to match.

---

## ⚠️ Current state warning (2026-05-01)

The C implementation in `c_impl/` is being rewritten. Do not assume v3.8.1 algorithmic parity until rewrite Phase 5 (parity verification) is complete. The legacy v2.5 implementation is preserved at `c_impl_v25_legacy/` for reference only.

If you need a stable C build today, use Python `aec.py` via subprocess or the v2.5 legacy with explicit version label.

---

## API surface

```c
#include "aec.h"
#include "aec_types.h"
#include "res_filter.h"
```

### Standard mode — single-call full pipeline

```c
AecConfig cfg = aec_config_from_preset(AEC_PRESET_BALANCED, 16000);
cfg.enable_cng = 1;
Aec* aec = aec_create(&cfg);
int hop = aec_get_hop_size(aec);

float mic[hop], ref[hop], out[hop];
while (read_audio(mic, ref, hop)) {
    aec_process(aec, mic, ref, out);
    write_audio(out, hop);
}

aec_destroy(aec);
```

### Linear-only mode — for inserting NR (or any post-filter) between linear and RES

```c
AecConfig cfg = aec_config_from_preset(AEC_PRESET_BALANCED, 16000);
cfg.enable_res = 0;            /* disable internal RES */
Aec* aec = aec_create(&cfg);
ResFilter* res = res_create(&res_config_from_aec(&cfg));
AecResContext* ctx      = aec_context_create(aec);
AecResContext* prev_ctx = aec_context_create(aec);

float aec_out[hop], nr_out[hop], res_out[hop];
Complex corrected_echo[n_freqs];
int have_prev = 0;

while (read_audio(mic, ref, hop)) {
    /* Stage 1: linear AEC, exposing context */
    aec_process_ex(aec, mic, ref, aec_out, ctx);

    /* Stage 2: NR (or any post-filter that produces per-bin gain) */
    nr_process(nr, aec_out, nr_out);

    /* Stage 3: RES with NR-gain-corrected echo
     *
     * NR has 1-frame OLA delay → use prev_ctx (delayed by 1 frame). */
    if (have_prev) {
        const float* g = nr_get_gain_per_bin(nr);
        for (int k = 0; k < n_freqs; k++) {
            corrected_echo[k].r = prev_ctx_echo_spec[k].r * g[k];
            corrected_echo[k].i = prev_ctx_echo_spec[k].i * g[k];
        }
        res_process(res, nr_out, corrected_echo,
                    prev_far_spec, prev_near_spec,
                    prev_ctx->far_power, prev_ctx->filter_converged,
                    prev_ctx->erle_factor, prev_ctx->dt_indicator,
                    /* over_sub */ 5.0f,
                    /* divergence */ 0.0f,
                    prev_ctx->is_stationary_dt,
                    /* shadow_dt */ 0.0f,
                    prev_ctx->erl_estimate,
                    /* epc_active */ 0,
                    prev_ctx->saturation_level,
                    res_out);
        write_audio(res_out, hop);
    } else {
        write_audio(aec_out, hop);  /* first frame: no prev_ctx, output linear-only */
    }

    /* Swap ctx ↔ prev_ctx for next frame */
    AecResContext* tmp = prev_ctx; prev_ctx = ctx; ctx = tmp;
    have_prev = 1;
}

aec_context_destroy(ctx);
aec_context_destroy(prev_ctx);
res_destroy(res);
aec_destroy(aec);
```

A working reference implementation lives at `Audio_ALG/pipelines/aec_nr_pipeline.c`.

---

## File-by-file responsibilities

| File | Role | Owns |
|------|------|------|
| `aec.h` / `aec.c` | Top-level orchestration | Hop pacing, mu_scale computation, EPC handling, ctx export, RES dispatch |
| `aec_types.h` | Configuration | `AecConfig` (all tunables), `AecPreset` enum, `AecResContext` (linear→RES handoff) |
| `pbfdkf.h` / `pbfdkf.c` | Linear adaptive filter | Kalman state (W, P, Q), error/echo spec, partition shift, G1 KX blended P-update |
| `res_filter.h` / `res_filter.c` | Residual echo suppression | WOLA, ENR mask, spectral floor, CNG, reverb tail |
| `shadow_filter.h` / `.c` | Background filter | Parallel PBFDKF with separate Q schedule, copy-controller hook |
| `dt_analyzer.h` / `.c` | Double-talk detection | DT-from-frame-zero stats detector, stationary far-end DT detector |
| `multi_erle.h` / `.c` | Convergence tracking | FilterErle (per-bin) + FullbandErle (broadband), confidence |
| `render_activity.h` / `.c` | Far-end activity | Active/stationary detection, hangover |
| `epc_detector.h` / `.c` | Echo path change | EPV (mic energy rise), shadow-rise detection |
| `shadow_copy.h` / `.c` | Shadow-to-main copy gate | 5-state machine, FS-baseline-tracking |
| `delay_estimator.h` / `.c` | Delay alignment | GCC-PHAT, ring buffer, delay shift handling |
| `fft_wrapper.h` / `.c` | FFT abstraction | kiss_fft real/complex, sqrt-Hann window |
| `fast_math.h` | Scalar approximations | sqrtf, log2f, expf for HF inner loops |
| `hpf.h` / `.c` | DC removal pre-filter | 80Hz Butterworth (off by default; pipeline-level use) |

---

## Things you CANNOT do

These are not "bad practice" — they cause correctness or stability failures.

### 1. Do not bypass linear AEC and feed mic directly to RES

```c
res_process(res, mic, /* echo_spec= */ NULL, ...);  /* WRONG */
```

RES uses the linear filter's `echo_spec` (frequency-domain echo estimate) as its primary signal. Without it, RES has no echo to subtract — output ≈ input.

### 2. Do not skip the NR-gain echo correction when inserting NR

```c
res_process(res, nr_out, prev_ctx->echo_spec, ...);  /* WRONG: should be NR-corrected */
```

NR attenuates echo-band frequencies; the saved `echo_spec` from the linear filter is "raw echo" pre-NR. Feeding raw `echo_spec` to RES causes RES to over-suppress (RES sees echo where NR has already removed it → applies more suppression on top → audible NE damage).

### 3. Do not feed current-frame ctx to RES when NR is in-line

NR's MMSE-LSA has a 1-frame OLA latency. If you pass `ctx` (current frame) instead of `prev_ctx` (previous frame), the echo, far_spec, near_spec are misaligned with `nr_out` by one hop → RES gain spectrum is computed against the wrong frame → echo leaks or NE damage depending on signal.

### 4. Do not omit `aec_reset()` between independent input streams

Diagnostic counters and DT/EPC detector states accumulate across calls. For batch processing of multiple files, call `aec_reset()` before each new file. Without reset, a tonal far-end from file N can trigger EPC false-positive at the start of file N+1.

### 5. Do not modify `AecResContext` fields after AEC populated them, except `echo_spec`

The pipeline pattern allows you to multiply `echo_spec_re/im` by NR gain per bin. Other fields (`far_spec`, `near_spec`, `far_power`, `filter_converged`, `erle_factor`, `dt_indicator`, `is_stationary_dt`, `saturation_level`, `erl_estimate`) are passed through unchanged from the linear AEC. Modifying them produces undefined RES behavior.

### 6. Do not enable `enable_cng` and run RES output through another CNG layer

CNG fills below-`g_min` bins with shaped noise. Stacking another CNG layer on top double-shapes the noise floor → tonal artifacts.

### 7. Do not assume `enable_delay_est = 1` is free in offline mode

Delay estimator runs on every frame and triggers filter reset on first acquisition (300ms warmup loss). For offline batch where files are pre-aligned, set `enable_delay_est = 0` and pre-shift inputs.

### 8. Do not change `filter_length` mid-call

PBFDKF partition count is computed at `aec_create()`. Reducing `filter_length` mid-call leaks weights past array bounds.

### 9. Do not feed non-`hop_size` blocks

`aec_process()` requires exactly `hop_size` samples. Variable-size blocks must be ring-buffered upstream.

### 10. Do not run multiple `Aec*` instances on the same `kiss_fft_cfg`

`fft_wrapper` allocates per-instance kiss state. Concurrent use of one `kiss_fft_cfg` from multiple threads = data race.

---

## Memory and threading

- **Per-instance heap**: ~80 KB @ 16kHz / 52ms / 257 freqs (estimate; verify post-rewrite). Static-memory branch in flight will eliminate `malloc`.
- **Threading**: each `Aec*` instance is single-threaded. Concurrent multi-stream = multiple instances. No shared FFT state.
- **Real-time safety**: no I/O, no `printf`, no large `malloc` in `aec_process()` once instance is created (post-rewrite). `aec_reset()` does not allocate.
- **Cache footprint**: PBFDKF filter weights dominate (`filter_length × n_freqs × 8 bytes complex`). For 16kHz/52ms: ~830 KB filter × 2 (main + shadow) ≈ 1.6 MB. May exceed small embedded L2.

---

## Build

```bash
cd c_impl/
make clean && make lib       # builds libaec.a
make example                 # builds bin/aec_wav demo
./bin/aec_wav mic.wav ref.wav out.wav --preset balanced
```

For Audio_ALG pipeline:
```bash
cd Audio_ALG/pipelines/
make aec_nr_pipeline
./aec_nr_pipeline mic.wav ref.wav out.wav balanced
```

---

## Validation checklist before deployment

1. Build with `-Wall -Werror -O2` — zero warnings
2. Run `./bin/aec_wav` on at least 5 representative samples
3. Run 800-case bench (script: `python/eval_aec_challenge.py` C wrapper TBD) — per-bucket scores ≤0.005 vs Python v3.8.1 baseline
4. Memory leak check: `valgrind --leak-check=full ./bin/aec_wav ...`
5. Real-time check: process an N-minute file, ensure `process()` time per frame < hop duration (10ms @ 16kHz)
6. Reset hygiene: `aec_create() → process() → reset() → process()` produces same output as fresh `aec_create()` for second invocation

---

## Reference

- Python source: `python/aec.py` (v3.8.1 reference)
- Architecture spec: `docs/signal_flow_constraints.md` (signal-flow rules + ABL'd code warnings)
- Rewrite plan: `docs/c_rewrite_plan.md` (phase split + parity gates)
- Changelog: `docs/CHANGELOG.md` (v3.7.0 / v3.7.1 / v3.8.0 / v3.8.1 architectural notes)
- Pipeline reference: `Audio_ALG/pipelines/aec_nr_pipeline.c` (working AEC+NR+RES integration)
