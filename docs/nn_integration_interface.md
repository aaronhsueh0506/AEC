# Linear AEC outputs for downstream models

The linear AEC exposes time-domain and frequency-domain views of the same hop. Match the downstream model to the instance grid and use the reset notification when keeping external state.

## The shared grid (fixed per instance)
All stages in one instance share a no-padding grid: **frame=FFT, hop=frame/2,
sqrt-Hann periodic (COLA)**. Supported choices are 8k:256/128,
16k:256/128 (default) or 512/256, and 48k:1024/512. A model MUST match the
selected instance grid; `n_freqs = fft/2+1` is therefore 129, 257, or 513.
The grid cannot change mid-stream.

## The seam: `AecResContext` (per-frame, frequency-domain)
Enable with `AecConfig.return_res_context = True`; then `aec.process(mic_hop, ref_hop)` returns
`(time_out, AecResContext)`. The context IS the per-frame feature vector a model consumes:

| field | shape / type | meaning (model input) |
|---|---|---|
| `error_spec` | (n_freqs,) complex64 | **E(f)** — reconstructing 50%-overlap sqrt-Hann STFT of the selected/crossfaded formed output: the real linear residual, in every mode. Analysis seam; never altered by the output guard. |
| `echo_spec` | (n_freqs,) complex64 | Matching windowed **Ŷ(f)** residual-echo reference. |
| `near_spec` | (n_freqs,) complex64 | Matching windowed capture spectrum; exactly `error_spec + echo_spec`. |
| `far_spec` | (n_freqs,) complex64 | PBFDKF far-end spectrum used by the adaptive filter; not a downstream WOLA synthesis frame. |
| `far_power` | float | mean(far²) — far activity. |
| `filter_converged` | bool | linear filter converged this frame. |
| `erle_factor` | float [0,1] | linear convergence quality. |
| `dt_indicator` | float [0,0.8] | double-talk confidence. |
| `divergence` | float [0,1] | filter divergence indicator. |
| `over_sub` | float | dynamic over-subtraction factor (Python only; equal to `erle_factor`, absent from the C `AecResContext`). |
| `erl_estimate` | float | dynamic echo-return-loss (render-based residual). |
| `raw_output` | (hop_size,) float | Refined PBFDKF output before output selection. This is also what the context-only entry point returns as its audio. It is not the formed seam and not the rendering base. |
| `formed_output` | (hop_size,) float | Current selected/crossfaded hop underlying `error_spec`; use this when a downstream block performs its own STFT. Always the real linear residual. |
| `output_base_spec` | (n_freqs,) complex64 | **Rendering base**: `error_spec`, or `error_spec + echo_spec` (= `near_spec`) on a hop where the output guard fires. Render `output_base_spec * gain`. Does not pair with `formed_output`. |
| `guard_active` | bool | This hop's output guard fired (`output_base_spec` is the capture). |
| `post_reset_serial` | int (uint32, wraps) | Hard post-chain reset counter; see *Reset serial* below. |

These are produced every hop by the production linear AEC with zero extra cost (already computed internally).

**C names.** The C struct `AecResContext` (`c_impl/include/aec.h`) carries the same data under slightly
different names: `raw_output` is `linear_hop`, `formed_output` is `formed_hop`; `res_gain`, `r2`,
`comfort_noise`, `res_floor_protect` and `usable_linear` have the same names. `output_base_spec` / `guard_active` are the separate C getter `aec_get_output_base()` (`AecOutputBase`) and `post_reset_serial` is `aec_post_reset_serial()`; neither is a field of the C `AecResContext`. The C call sequence is
`cfg.enable_res = 0; cfg.return_res_context = 1;` at construction, then per hop
`aec_process_context()` followed by `aec_get_res_context()` (pointers valid until the next process or
reset call). See the [C integration example](html/reference/aec_usage.html).

### Analysis seam vs rendering base

Two groups of fields, never to be substituted for each other:

- **Analysis seam** — `formed_output`, `error_spec`, `echo_spec`, `near_spec`.
  The real selected (refined or coarse) linear residual and its matched echo
  estimate, identical in every API mode (full, context-only, no-gain,
  linear-only). `error_spec` is the STFT of `[previous formed hop | current
  formed hop]`, so a unity IFFT with overlap-add reproduces the previous formed
  hop one framing hop late on every hop. AecState, ERLE, R² and the PSDs read
  this seam, and so should anything that needs a pairable waveform or STFT: an
  AIAEC front end (`linear_error`), training features, a model with its own
  analysis window. The analysis residual can exceed the capture on a guard hop;
  the final audio is protected by the rendering guard below.
- **Rendering base** — `output_base_spec` (C: `aec_get_output_base()` →
  `AecOutputBase{struct_size, n_freqs, guard_active, base_spec}`). The output
  guard is evaluated exactly once per hop, in every mode, after the hop's
  usable-linear verdict is final: when
  `output_capture_when_linear_unusable` is set (default ON), the verdict is
  false and `sum(|error_spec|²) > sum(|error_spec + echo_spec|²)` (strict, float32
  pairwise reduction in both languages, so identical spectra give the identical decision), the base is the capture
  `error_spec + echo_spec`; otherwise it is `error_spec`. Audio-rendering
  consumers (mono, DFN, ULCNet and 4ch lanes, board samples) must multiply
  `output_base_spec` by their gain(s), never `error_spec`; the split path then
  produces what `aec_process()` produces. The base is not a formed hop: on a
  guard hop it is a spectral switch to the capture and a unity inverse
  transform of it reproduces no formed hop. The guard acts only while the
  estimate is unusable, so it fires mostly before the first usable verdict but
  also on clips that never become usable; it is not confined to the first seconds.

Python `AecConfig` rejects `return_formed_output` unless `enable_res` or
`return_res_context` is set: the formed seam is produced by the post chain and
cannot be requested without it, and asking anyway raises at construction.

The case the guard exists for: a loud far end whose echo never reaches the
microphone. The filter has nothing to cancel, the quality analyzer never declares
the estimate usable, and an unguarded rendering would carry far-end energy that
the capture never had. `echo_spec` is not altered by the guard.

C minimum for rendering outside the library, per hop after
`aec_process_context()`: `aec_get_res_context()`, `aec_get_output_base()`,
`spec = base_spec * gain` (`res_gain`, times any NR gain), `aec_cng_fill()`,
then `ac_istft_push()` (audiolib), handling the reset serial below first. With
NR/NN off, the same RES/CNG settings and the library's own synthesis window
(`aec_synthesis_window(aec, &w, &n)`: read-only table of `n == fft_size` floats owned by the
instance, valid for its lifetime; set it as `AcStftConfig.window` before
`ac_istft_get_mem_size()` / `ac_istft_init()`, the `AcIstft` borrows it and must not outlive
the `Aec`; the audiolib default sqrt-Hann differs from it in the last float bit), this is
bit-identical to `aec_process()`.

**What the guard changes for existing consumers.**

- A frequency-domain consumer that still renders `error_spec` instead of the rendering
  base silently loses the guard: on guard hops its output carries the residual that is
  louder than the capture. This is a breaking change for renderers written before
  `aec_get_output_base()` existed; switch them to `output_base_spec * gain`.
- Time-domain seam consumers (`formed_output` / C `formed_hop`, and
  `AecLinearContext.formed_linear_hop`) are not guarded: those streams are the analysis
  residual on every hop, guard hops included, and there is no time-domain rendering
  base. A consumer that plays or re-synthesizes them gets no output guard.
- C `aec_get_output_base()` validates `AecOutputBase.struct_size` (input) before writing:
  a value below `sizeof(AecOutputBase)` returns -1 and writes nothing, a larger value is
  accepted and only the known prefix is written (`struct_size` comes back as
  `sizeof(AecOutputBase)`).

### Reset serial

A hard post-chain reset has two sources. `aec_reset()` is a separate call, not part
of any processing call: the serial already reads +1 when it returns, and the next
processing call starts from the cleared state. A confirmed delay change that restarts
the filter (hard shift, or a first acquisition without warm transfer or DT-aware soft
acquisition) happens inside a hop's processing, before that hop's synthesis. At each
counted reset the library clears its own OLA tail and reseeds its CNG to `AEC_CNG_SEED` (42).
`aec_post_reset_serial()` (Python `get_post_reset_serial()` /
`AecResContext.post_reset_serial`) is a non-consuming, wrapping `uint32` counter
of those events. It does not move on warm or soft realignments, on the bad-relock
revert (a soft revert to the pre-shift delay and tap snapshot: OLA tail and CNG keep
running), or on `aec_apply_external_realign()`. An application that synthesizes outside the
library keeps its own last-handled serial. When the value read after a
processing call (or after `aec_reset()`) differs from it, the application must, before its first
CNG draw and before the inverse-transform/overlap-add push of the hop it renders from that call
(after `aec_reset()`: the first hop processed afterwards), clear its OLA
tail (e.g. `ac_istft_reset`) and reseed its CNG state to `AEC_CNG_SEED`, then
store the new value. It must not reset the AEC in response. The library touches
neither NR nor NN state: whether and how they follow the reset is the
application's policy.

### Comfort noise

`aec_cng_fill(&rng, spec, comfort_noise, g_res, g_nr, nr_floor, n)` adds, for
bins `1 .. n-2`, `ng * (sqrt(cn / 32768²) * table)` per component with
`ng = sqrt(1 - g_res²)` times `max(g_nr, nr_floor)` when an NR gain is given.
It draws one LCG step per bin in ascending order and is the same function the
library's own output stage calls (with `g_nr = NULL` and its total gain as
`g_res`), so without an NR gain it reproduces the in-chain noise bit for bit.
Python: `modules/cng.py:cng_fill`. Start the generator at `AEC_CNG_SEED`.

## The time-domain seam: `AecLinearContext` (aligned far + delay status)
For post-filters that take the **time-domain aligned far-end** as a second input
(e.g. an alignment-attention RES+NR model that performs its own sqrt-Hann STFT
on both streams), `aec_get_linear_context()` exposes:

| field | type | meaning |
|---|---|---|
| `formed_linear_hop` | (hop,) float | formed linear output (the same hop `formed_output` reflects: the real linear residual, never the capture — see *Analysis seam vs rendering base*). |
| `aligned_far_hop` | (hop,) float | **the exact time-domain far the PBFDKF consumed this hop** (aliases the internal buffer; valid until the next process/reset). NOT `far_spec`: that is a rectangular overlap-save FFT, unusable as a sqrt-Hann analysis frame. |
| `delay_samples` | int | applied ring offset. Meaning depends on `delay_mode` — see below. |
| `delay_confidence` | float | 0 / 0.5 / 1 under `MATCHED`; always 1.0 under `FIXED`/`EXTERNAL_ALIGNED` (an out-of-band or contractual alignment carries no estimator uncertainty to report). |
| `delay_state` | enum | `UNLOCKED` (content is RAW far — do not treat as aligned), `LOCKED`, `CHANGED` (offset moved THIS hop — flush any far feature rings / attention history downstream). Per-mode timing below. |
| `generation` | unsigned | bumps on every ring-offset change including the flagless soft-recovery realigns and `aec_reset()`; saturating. Poll it instead of differencing `delay_samples` (transient A→B→A shifts are invisible to differencing). |

### `delay_mode` changes the *timing* of this seam, not just its source
`AecConfig.delay_mode` (`AEC_DELAY_MATCHED` / `AEC_DELAY_FIXED` /
`AEC_DELAY_EXTERNAL_ALIGNED`, init-time immutable, see `aec.h`) is the single
source of truth for far/mic alignment, and each mode reaches `LOCKED` on a
different hop:

| `delay_mode` | `delay_samples` | reaches `LOCKED` |
|---|---|---|
| `MATCHED` (default) | −1 before acquisition, else the estimated ring offset | once the matched filter acquires — needs far activity and `delay_aec_n_updates() ≥ 3` solid updates; never, if the true delay is outside the ~509 ms reliable-peak bound |
| `FIXED` | always `cfg.fixed_delay_samples` (never −1) | once the ring can serve the offset, not hop 0: the first `ceil(fixed_delay_samples / hop)` hops report `UNLOCKED` over RAW far, and the next process call reports `LOCKED` |
| `EXTERNAL_ALIGNED` | always `0` | **hop 0** — the caller has contracted that `ref` is already aligned, so there is nothing to wait for |

The legacy `enable_delay_est=0` toggle is a deprecated mirror of `delay_mode`,
folded into it by `aec_config_resolve_delay()`; its equivalent is
`AEC_DELAY_EXTERNAL_ALIGNED`, which reports `LOCKED` / `delay_samples=0` /
`delay_confidence=1.0` from the very first hop. A consumer that gates a
far-conditioned NN post-filter on `delay_state == LOCKED` must account for
this: under `EXTERNAL_ALIGNED` that gate is open immediately, with no
warm-up wait to rely on; under `FIXED` it is closed for a short, computable
window; only `MATCHED` can stay `UNLOCKED` indefinitely.

Out-of-range bulk delay (beyond the matched filter's ~509 ms reliable-peak
bound at 16 kHz; the 608 ms figure in `c_user_manual_zh_TW.md` is the full
filter-bank geometric span — a different definition, not a contradiction) is
a `MATCHED`-only failure mode: the estimator simply never acquires and the
state stays `UNLOCKED`. `FIXED` and `EXTERNAL_ALIGNED` cannot fail this way —
alignment there is a caller contract, not a search, so it is not detectable
at this seam either way.
Fail-open policy (bypass the far-conditioned model, emit the linear error)
belongs to the integrator. Regression coverage: `test/test_linear_context.c`
(including the far-active/no-echo scenario: the rendering base never carries more energy than the capture) and `test/test_guard_unify.c` (split rendering against `aec_process()`).

The estimator's `confidence` measures histogram consistency, not match quality,
so a `LOCKED` state is not proof that the alignment is correct, and a re-lock
arrives as a normal `CHANGED` hop visible through `generation`: flush far
feature rings on `generation` movement regardless. A consumer that owns its own estimator can
ask the engine the same "is this lane still cancelling" question the engine
asks of itself, via the read-only `aec_linear_is_cancelling(const Aec*)` (`aec.h`);
it returns 0 when the ERLE machinery has not run, i.e. absence of evidence,
never a claim that the filter is broken.

## Connecting a model

A model that accepts a time-domain residual can consume the floating-point formed hop directly. The C field is `AecResContext.formed_hop`. A caller using an aligned far signal can obtain its hop and alignment generation from `aec_get_linear_context()`. Follow the model's contract for far input: Align-ULCNet's PCM stream in Audio_ALG takes the formed error and **raw** far input.

A frequency-domain model can read `error_spec` for analysis. When rendering the guarded signal, obtain the base from `aec_get_output_base()`, apply the required gains, add comfort noise once if configured, then synthesize. Keep independent OLA and model state per stream and handle changes to `aec_post_reset_serial()` before synthesizing that hop. See the [output-guard integration guide](sdk_migration_output_guard.md).
