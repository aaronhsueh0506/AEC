# NN integration interface (residual / NR / joint) — doc-only spec
AEC v3.22.5 · the freq-domain seam for swapping any post-linear stage with a learned model

> Scope: this documents the **existing** interface that lets a neural model replace the residual-echo
> suppressor, the noise reduction, or both, **without touching the linear AEC or the front/back transform**.
> No NN is shipped; the DSP stages remain the default and fallback. Companion pipeline design:
> `Audio_ALG/docs/freq_domain_pipeline_design.md`.

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
| `error_spec` | (n_freqs,) complex64 | **E(f)** — reconstructing 50%-overlap sqrt-Hann STFT of the selected/crossfaded linear output. |
| `echo_spec` | (n_freqs,) complex64 | Matching windowed **Ŷ(f)** residual-echo reference. |
| `near_spec` | (n_freqs,) complex64 | Matching windowed capture spectrum; exactly `error_spec + echo_spec`. |
| `far_spec` | (n_freqs,) complex64 | PBFDKF far-end spectrum used by the adaptive filter; not a downstream WOLA synthesis frame. |
| `far_power` | float | mean(far²) — far activity. |
| `filter_converged` | bool | linear filter converged this frame. |
| `erle_factor` | float [0,1] | linear convergence quality. |
| `dt_indicator` | float [0,0.8] | double-talk confidence. |
| `divergence` | float [0,1] | filter divergence indicator. |
| `over_sub` | float | dynamic over-subtraction factor. |
| `erl_estimate` | float | dynamic echo-return-loss (render-based residual). |
| `raw_output` | (hop_size,) float | Refined PBFDKF output before refined/coarse selection. |
| `formed_output` | (hop_size,) float | Current selected/crossfaded hop underlying `error_spec`; use this when a downstream block performs its own STFT. |

These are produced every hop by the production linear AEC with zero extra cost (already computed internally).

## The time-domain seam: `AecLinearContext` (aligned far + delay status)
For post-filters that take the **time-domain aligned far-end** as a second input
(e.g. an alignment-attention RES+NR model that performs its own sqrt-Hann STFT
on both streams), `aec_get_linear_context()` exposes:

| field | type | meaning |
|---|---|---|
| `formed_linear_hop` | (hop,) float | formed linear error (same hop `formed_output` reflects). |
| `aligned_far_hop` | (hop,) float | **the exact time-domain far the PBFDKF consumed this hop** (aliases the internal buffer; valid until the next process/reset). NOT `far_spec`: that is a rectangular overlap-save FFT, unusable as a sqrt-Hann analysis frame. |
| `delay_samples` | int | applied ring offset; −1 before acquisition. |
| `delay_confidence` | float | 0 / 0.5 / 1. |
| `delay_state` | enum | `UNLOCKED` (content is RAW far — do not treat as aligned), `LOCKED`, `CHANGED` (offset moved THIS hop — flush any far feature rings / attention history downstream). |
| `generation` | unsigned | bumps on every ring-offset change including the flagless soft-recovery realigns and `aec_reset()`; saturating. Poll it instead of differencing `delay_samples` (transient A→B→A shifts are invisible to differencing). |

Out-of-range bulk delay (beyond the matched filter's ~509 ms reliable-peak
bound at 16 kHz; the 608 ms figure in `c_user_manual_zh_TW.md` is the full
filter-bank geometric span — a different definition, not a contradiction) is
not detectable at this seam: the state simply stays `UNLOCKED`.
Fail-open policy (bypass the far-conditioned model, emit the linear error)
belongs to the integrator. Regression coverage: `test/test_linear_context.c`.

## The three swap points (freq-in → freq-out blocks)
Each is a pure function on the shared grid; the network replaces the DSP function, nothing else.

### 1. NN-residual (replace the AEC3 post-filter `SuppressionGain`)
- **in:** `E(f)` post-NR (or pre-NR) + `{Ŷ(f), X(f), erle_factor, dt_indicator, far_power, divergence}`
- **out:** real gain `G_res(f) ∈ [0,1]` (apply `S(f)=G_res·E(f)`) **or** the enhanced complex `S(f)`.
- DSP baseline it replaces: AEC3 `GainToNoAudibleEcho` (ENR/EMR), `suppression_gain.py`.

### 2. NN-NR (replace `denoise_spectrum`)
- **in:** `E(f)` (+ optional running noise estimate)
- **out:** real gain `G_nr(f) ∈ [0,1]` **or** enhanced spectrum.
- DSP baseline it replaces: NR `MmseLsaDenoiser.denoise_spectrum()` (every NR denoiser already exposes this
  freq-in/freq-out entry — the model drops in at the same call site).

### 3. NN-residual+NR (replace NR+RES jointly with one mask)
- **in:** `E(f)` + full `AecResContext`
- **out:** final `S(f)` (one network does denoise + residual-echo suppression in a single complex/real mask).
- This is the highest-leverage seam: a single model consumes the linear AEC's freq context and emits the
  finished near-end spectrum; the chain becomes `[window+FFT] → AEC-linear → NN → [IFFT+OLA]`.

## Wiring contract (how a model is enabled)
1. Run AEC with `return_res_context=True` → stream of `(E(f), ctx)`.
2. Route `E(f)` (+ ctx fields) through the chosen NN block instead of the DSP function.
3. Apply the returned gain/spectrum in the frequency domain.
4. One `irFFT + sqrt-Hann OLA` at the very end (shared back-end).
- The linear AEC, the front-end window+FFT, and the back-end IFFT+OLA are **unchanged** across all three
  swaps. The model needs to honor its selected `sample_rate/fft/hop/n_freqs` contract and return a same-shaped
  gain or spectrum. The DSP path stays as the deterministic fallback (and the A/B reference).

## Why this is safe for a DSP-first release
- The interface is already present (`return_res_context`, `AecResContext`, `denoise_spectrum`, `run_res`)
  and exercised by `aec_nr_pipeline.py --pipeline-mode linear`. Shipping v3.22.5 DSP does not commit to any
  model; it guarantees the seam exists so an NN arc can start without re-architecting the front end.
- Training-data generation is free: the DSP stages produce `(input features = AecResContext, target = clean
  near-end)` pairs on the 800-case corpus and any field recording, at the exact grid the model will run on.
