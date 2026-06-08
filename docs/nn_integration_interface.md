# NN integration interface (residual / NR / joint) — doc-only spec
AEC v3.22.5 · the freq-domain seam for swapping any post-linear stage with a learned model

> Scope: this documents the **existing** interface that lets a neural model replace the residual-echo
> suppressor, the noise reduction, or both, **without touching the linear AEC or the front/back transform**.
> No NN is shipped; the DSP stages remain the default and fallback. Companion pipeline design:
> `Audio_ALG/docs/freq_domain_pipeline_design.md`.

## The shared grid (fixed — see item-13 sensitivity)
All stages operate on one analysis grid: **16 kHz · fft=512 · hop=160 (10 ms) · sqrt-Hann periodic (COLA) ·
257 complex bins** (`n_freqs = fft/2+1`). A model MUST match this grid. Changing it is a structural refactor
(delay block-rate is `250 = 16000/64`, PSD constants are int16²-sum-calibrated at fft=512) — not a free knob.

## The seam: `AecResContext` (per-frame, frequency-domain)
Enable with `AecConfig.return_res_context = True`; then `aec.process(mic_hop, ref_hop)` returns
`(time_out, AecResContext)`. The context IS the per-frame feature vector a model consumes:

| field | shape / type | meaning (model input) |
|---|---|---|
| `near_spec` | (257,) complex64 | **E(f)** — linear AEC error spectrum (mic − Ŵ·X). The "noisy" signal to denoise / suppress. |
| `echo_spec` | (257,) complex64 | **Ŷ(f)** — linear echo estimate. Residual-echo reference. |
| `far_spec` | (257,) complex64 | **X(f)** — far-end (render) spectrum. |
| `far_power` | float | mean(far²) — far activity. |
| `filter_converged` | bool | linear filter converged this frame. |
| `erle_factor` | float [0,1] | linear convergence quality. |
| `dt_indicator` | float [0,0.8] | double-talk confidence. |
| `divergence` | float [0,1] | filter divergence indicator. |
| `over_sub` | float | dynamic over-subtraction factor. |
| `erl_estimate` | float | dynamic echo-return-loss (render-based residual). |
| `raw_output` | (160,) float | time-domain linear output (for reference/fallback). |

These are produced every hop by the production linear AEC with zero extra cost (already computed internally).

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
  swaps. The model only needs to honor the 257-bin / hop-160 / 16 k contract and return a same-shaped
  gain or spectrum. The DSP path stays as the deterministic fallback (and the A/B reference).

## Why this is safe for a DSP-first release
- The interface is already present (`return_res_context`, `AecResContext`, `denoise_spectrum`, `run_res`)
  and exercised by `aec_nr_pipeline.py --pipeline-mode linear`. Shipping v3.22.5 DSP does not commit to any
  model; it guarantees the seam exists so an NN arc can start without re-architecting the front end.
- Training-data generation is free: the DSP stages produce `(input features = AecResContext, target = clean
  near-end)` pairs on the 800-case corpus and any field recording, at the exact grid the model will run on.
