# Bundle B Formula Audit: AEC3 CoarseFilterUpdateGain vs Python Shadow PBFDAF

**Date**: 2026-05-26 (rev 2: 2026-05-26 — rate corrected 0.95→0.7; noise gate constant split clarified)  
**Status**: Audit only — no ship/no-ship verdict. Do NOT interpret as "AEC3 coarse fallback invalid."  
**Source**: `docs/aec3_extracts/src/aec3/coarse_filter_update_gain.cc` vs `python/modules/filters.py` PBFDAF `_update_weights`

> **Rev 2 corrections** (2026-05-26):
> 1. **Rate**: AEC3 single-channel default `coarse.rate = 0.7` (not 0.95). `0.95` is multichannel/field-trial only. `coarse_initial.rate = 0.9` for delay-event transient only.
> 2. **Noise gate units**: `filter.coarse.noise_gate = 20075344.f` is in int16² (convert with `psd_int16_to_float`). `echo_model.noise_gate_power = 27509.42f` is in AEC3 float units (do NOT apply `psd_int16_to_float`). These are different paths and different unit systems.
> 3. **v3.21 alignment target**: single-channel default = `rate=0.7`. All prior entries stating `rate=0.95` for "AEC3 default" were incorrect for the production target.
> 4. **Bundle B failure classification**: Added explicit a/b/c/d/e breakdown; `c` (FFT/window scale) flagged as OPEN pending synthetic test.
> 5. **FFT/Window Scale Analysis**: New section added — synthetic test required before depth-ratio correction can be quantified.

---

## Purpose

The Bundle B sub-ladder showed that A.1 (`use_partition_summed_x2_for_shadow_mu`) makes
`e2_coarse` worse on 6/8 cases, and that A.2+A.1 together cannot restore shadow convergence.
This audit finds whether the problem is a **port mapping mismatch** (wrong constant or wrong
structural coupling) or an **implementation bug** (code does not match its intended formula).

**Key question**: Why does partition-summed X² in the shadow NLMS denominator degrade convergence?

---

## AEC3 CoarseFilterUpdateGain::Compute (ground truth)

Source: `coarse_filter_update_gain.cc` lines 50–84 (extracted verbatim).

```cpp
void CoarseFilterUpdateGain::Compute(
    const std::array<float, kFftLengthBy2Plus1>& render_power,  // = SpectralSum(n_partitions)
    const RenderSignalAnalyzer& render_signal_analyzer,
    const FftData& E_coarse,
    size_t size_partitions,
    bool saturated_capture_signal,
    FftData* G) {
  ++call_counter_;
  UpdateCurrentConfig();
  if (render_signal_analyzer.PoorSignalExcitation()) {
    poor_signal_excitation_counter_ = 0;               // RESET on poor excitation
  }
  // Skip if insufficient excitation OR startup OR saturation
  if (++poor_signal_excitation_counter_ < size_partitions ||
      saturated_capture_signal || call_counter_ <= size_partitions) {
    G->re.fill(0.f); G->im.fill(0.f);
    return;
  }
  // Compute mu
  std::array<float, kFftLengthBy2Plus1> mu;
  const auto& X2 = render_power;    // = SpectralSum(size_partitions)
  for (size_t k = 0; k < kFftLengthBy2Plus1; ++k) {
    if (X2[k] > current_config_.noise_gate) {        // noise_gate = 20075344.f (int16²)
      mu[k] = current_config_.rate / X2[k];          // rate = 0.95f
    } else {
      mu[k] = 0.f;
    }
  }
  render_signal_analyzer.MaskRegionsAroundNarrowBands(&mu);
  for (size_t k = 0; k < kFftLengthBy2Plus1; ++k) {
    G->re[k] = mu[k] * E_coarse.re[k];              // G = mu × E
    G->im[k] = mu[k] * E_coarse.im[k];
  }
}
```

`render_power` is `SpectralSum(size_partitions = 13)`:
```cpp
// subtractor.cc:206–208
render_buffer.SpectralSums(coarse_filter_[0]->SizePartitions(),   // = 13
                            refined_filters_[0]->SizePartitions(), // = 13
                            &X2_coarse, &X2_refined);
```

```cpp
// render_buffer.cc:42–57  SpectralSum
void RenderBuffer::SpectralSum(size_t num_spectra, ...) {
  X2->fill(0.f);
  for (size_t j = 0; j < num_spectra; ++j) {
    for (k) (*X2)[k] += channel_spectrum[k];   // sum of |X[p][k]|² over num_spectra frames
  }
}
```

**AEC3 config for v3.21 alignment** (source: `v3_21_delay_change_chain_design_note.md` §3.1,
`v3_21_a1_delay_change_chain_design.md` §6):

```
# Single-channel default config (v3.21 alignment target):
coarse.length_blocks      = 13       (13 × 64 samples = 832 samples at 16 kHz)
coarse.rate               = 0.7      ← SINGLE-CHANNEL DEFAULT (NOT 0.95)
coarse_initial.rate       = 0.9      ← transient rate during delay-change events (1.29× boost)
coarse.noise_gate         = 20075344.f  (filter adaptation gate; see §Noise Gate below)

# Multichannel / field-trial config only:
# coarse.rate             = 0.95     ← NOT the single-channel production default
```

**Rate config clarification**: The v3.21 alignment target is the **single-channel default**
config (`coarse.rate = 0.7`). The `0.95` value appears only in multichannel or specific
field-trial configurations. All prior audit entries stating `rate=0.95` were incorrect for
the single-channel production alignment target.

`coarse_initial.rate = 0.9` is activated only by Step 6 of the AEC3 delay-change chain
(`SetConfig(coarse_initial, /*immediate=*/true)`). It applies for 2.5 s after a delay event,
then reverts to `coarse.rate = 0.7`. This transient rate boost is not yet ported to Python
(PBFDAF shadow has a single fixed `mu`; no `_p_max_override` guard for PBFDAF).

---

## Python Shadow PBFDAF _update_weights (current implementation)

Source: `python/modules/filters.py` PBFDAF.`_update_weights`, lines 263–352.

**Gate sequence** (flags A.5→A.3→stationary→A.4→A.1→A.2 in order):

```python
# A.5 saturation gate
if _use_saturation_gate and _saturated_capture: return

# A.3 poor_excitation gate
if _use_poor_excitation_gate:
    self._call_counter += 1
    if call_counter <= n_partitions or poor_exc_counter < n_partitions: return

# stationary gate (always-on, not an A.x flag)
if _block_stationary: return

# A.4 narrowband mask
if _use_narrowband_mask: mu_scale_arr *= narrowband_mask

# A.1 denominator
if _use_partition_summed_x2_for_shadow_mu:
    x2_part = (|X_buf|²).sum(axis=0)        # Σ_p |X[p][k]|² over n_partitions frames
    denom = x2_part + delta
else:
    denom = power_floor * n_partitions + delta   # EMA-smoothed power × n_partitions + floor

mu_eff = (self.mu * mu_scale_arr) / denom      # self.mu = shadow_mu_nlms = 0.5 (BALANCED)

# A.2 noise gate
if _use_aec3_noise_gate_for_shadow:
    x2_for_gate = (|X_buf|²).sum(axis=0)     # SpectralSum regardless of A.1
    mu_eff = where(x2_for_gate >= NOISE_GATE_POWER_FLOAT, mu_eff, 0.0)
    # NOISE_GATE_POWER_FLOAT = psd_int16_to_float(27509562.0) = 0.02562

for p in range(n_partitions):
    W[p] += mu_eff * error_spec * conj(X_buf[p])   # FDAF gradient update
```

**Python BALANCED shadow config**:
```
n_partitions   = 6   (filter_length=832 / hop_size=160 → ceil(832/160)=6; runtime-confirmed)
self.mu        = shadow_mu_nlms = 0.5
mu_scale_arr   = [0.5 to 1.0] depending on DT detection
```

---

## Component-by-Component Audit

### 1. X² denominator source (A.1)

| Dimension | AEC3 | Python (A.1 OFF) | Python (A.1 ON) | Classification |
|-----------|------|-----------------|-----------------|----------------|
| X² source | SpectralSum (instantaneous, no EMA) | EMA(power) × n_partitions | SpectralSum (instantaneous) | Port mapping mismatch |
| n_partitions in X² | 13 (coarse.length_blocks) | **6** (Python shadow) | **6** (Python shadow) | Structural divergence |
| Rate constant | `rate = 0.95` | `mu = 0.5` | `mu = 0.5` (unchanged) | Port mapping mismatch |

**Root-cause analysis for "A.1 makes e2_coarse worse"**:

The Python EMA denominator was co-designed with `mu = 0.5`:
- A.1 OFF: `denom = EMA(|X_curr|²) × 5` — stable, smoothed denominator; `mu=0.5` calibrated for this
- A.1 ON: `denom = Σ_p |X[p]|²` — instantaneous sum; `mu=0.5` was NOT calibrated for this

When the echo path has a dominant short lag (most energy in partition 0):
- `EMA(|X_curr|²) × 5 ≈ P_0 × 5` (inflated ×5 by n_partitions multiplier)
- `Σ_p |X[p]|² ≈ P_0 + small_contributions ≈ P_0` (only slight inflation)
- A.1 ON denom < A.1 OFF denom → **A.1 ON mu is larger** (confirmed by trace: mu_eff_mean increases B0→B1)
- Higher mu on instantaneous X² → more aggressive weight update per frame → **over-adaptation during DT or path changes**
- Over-adaptation leads to shadow weight instability → E_coarse residual grows → e2_coarse increases

For an AEC3-compatible shadow, the SpectralSum denominator requires:
- `rate ≈ 0.95` (not 0.5) to match AEC3's effective step size
- `n_partitions ≈ 13` (AEC3 default) or rate rescaled by `5/13` if using 5 partitions
- Poor excitation counter protecting against over-adaptation during low-render periods

**Classification: port mapping mismatch** — A.1 implements SpectralSum correctly, but the rate
constant `mu=0.5` was tuned for EMA denominator and cannot be directly swapped to SpectralSum
without re-calibrating `mu` to account for (a) instantaneous vs. smoothed noise floor difference
and (b) 5 vs. 13 partition depth difference.

This is NOT a bug in the implementation of A.1 itself.

---

### 2. Noise gate (A.2)

#### AEC3 noise gate constant sources (two separate fields, different units)

AEC3 EchoCanceller3Config has two distinct noise-gate constants that serve different roles:

| AEC3 config field | Value | Used in | Unit |
|---|---|---|---|
| `filter.coarse.noise_gate` | `20075344.f` | CoarseFilterUpdateGain — weight update gate | int16² (relative to 32768 full-scale) |
| `echo_model.noise_gate_power` | `27509.42f` | ResidualEchoEstimator — echo model gate | float (AEC3 internal render power) |

These are completely separate paths. `echo_model.noise_gate_power` is consumed by
`residual_echo_estimator.cc:124–126` to gate the echo power estimate; it is NOT used in
the filter weight update. `filter.coarse.noise_gate = 20075344.f` is consumed by
`coarse_filter_update_gain.cc:67–68` to gate the NLMS step.

**Critical note**: `27509.42f` (echo_model) and `20075344.f` (filter) are in different unit
systems. The echo_model value is in AEC3's internal float render-power representation; the
filter value is in int16² units (divide by 32768² to get float). Do NOT apply
`psd_int16_to_float` to `27509.42f` — it is already a float.

#### Python A.2 current vs correct

| Dimension | AEC3 coarse filter | Python A.2 (current) | Python A.2 (corrected) |
|-----------|-------------------|-----------------------|------------------------|
| Gate source | `filter.coarse.noise_gate = 20075344.f` | `echo_model.noise_gate` (wrong path) | `filter.coarse.noise_gate = 20075344.f` |
| Gate threshold (float) | `20075344 / 32768² = 0.01869` | `0.02562` (wrong: ~37% too high) | `0.01869` |
| SpectralSum depth | 13 partitions | 5 partitions | 5 partitions (structural; see §FFT/window) |
| Effective per-partition threshold | `0.01869 / 13 ≈ 1.44e-3` | `0.02562 / 5 ≈ 5.12e-3` | `0.01869 / 5 ≈ 3.74e-3` |
| Over-aggressiveness vs AEC3 | — | **3.6× per partition** | 2.6× (depth mismatch only) |

**Python A.2 uses the wrong constant source**: `NOISE_GATE_POWER_FLOAT` in our code is
derived from the echo model path (`psd_int16_to_float(27509562.0) = 0.02562`), not from the
filter config path. The correct constant for the shadow filter noise gate is
`psd_int16_to_float(20075344.0) = 0.01869`.

The suggested new constant name (audit-only, no production change this round):
```python
FILTER_NOISE_GATE_FLOAT = psd_int16_to_float(20075344.0)  # = 0.01869; AEC3 coarse filter gate
# Do NOT use NOISE_GATE_POWER_FLOAT (= 0.02562 from echo_model path) for filter weight update
```

**SpectralSum depth mismatch remains** even with the correct constant. With 5 partitions
vs AEC3's 13, our SpectralSum is 5/13 ≈ 38% of AEC3's. Whether this requires a depth-scaled
threshold depends on the FFT/window scale analysis (see §FFT/Window Scale below). The
per-partition effective threshold would be 2.6× too high with the correct constant but
mismatched depth. Full correction requires resolving §FFT/Window Scale first.

**Classification: port mapping mismatch (two sub-issues)** — wrong constant source
(`echo_model.noise_gate_power` vs `filter.coarse.noise_gate`) and SpectralSum depth mismatch
(**6** vs 13 partitions; SpectralSum scale ratio 29.6× — see §FFT/Window Scale Analysis).

---

### 3. Poor excitation gate (A.3)

| Dimension | AEC3 | Python A.3 | Classification |
|-----------|------|-----------|----------------|
| Counter increment timing | Only when NOT poor (implicit by always incrementing then resetting on poor) | Explicit `_call_counter` + `_poor_excitation_counter` | **Structural equivalent** |
| Reset trigger | `PoorSignalExcitation()` resets `poor_signal_excitation_counter_` to 0 | `_poor_excitation_counter` increment gated on RSA result | Equivalent semantics |
| Gate threshold | `poor_signal_excitation_counter_ < size_partitions` | `_poor_excitation_counter < n_partitions` | Equivalent (different absolute value: **6** vs 13) |
| Gate order vs saturation | Compound: `++poor_exc < n_parts OR sat OR call_ctr` | Separate: sat → poor_exc → stationary | Minor order difference |

**Classification: structural equivalent** — A.3 correctly ports AEC3's poor excitation counter.
The `call_counter ≤ n_partitions` startup gate is also correctly ported. Minor order difference
(sat before poor_exc in Python vs simultaneous in AEC3) is not observable under normal conditions
since saturation events are rare.

**AEC3 importance note**: A.3's `poor_excitation_counter` is load-bearing in AEC3's shadow design
because it protects the higher-rate (`rate=0.7`) update from adapting during noisy periods.
In our Python shadow with mu=0.5 (EMA denominator), the EMA already provides a similar stabilization
effect, making A.3's marginal contribution smaller. This explains the ~10-25% skip fraction
with limited impact on e2_coarse in the B3 step.

---

### 4. Narrowband mask (A.4)

| Dimension | AEC3 | Python A.4 | Classification |
|-----------|------|-----------|----------------|
| Mask application | `MaskRegionsAroundNarrowBands(&mu)` pre-multiplies mu in-place | `mu_scale_arr *= narrowband_mask` before mu computation | **Structurally equivalent** |
| Mask source | RenderSignalAnalyzer | Same Python RSA ported in `modules/render/` | Correct port |
| Fire rate on 8-case cohort | Not measured | ~1% bins (A.4_mask_frac from sub-ladder) | Very low fire rate |

**Classification: correct port** — A.4 applies the narrowband mask at the correct point in the
compute chain. Very low fire rate (1%) on the 8-case cohort reflects the lack of narrowband tonal
signals in the test set; behavior on true narrowband cases would be as designed.

---

### 5. Saturation gate (A.5)

| Dimension | AEC3 | Python A.5 | Classification |
|-----------|------|-----------|----------------|
| Trigger | `saturated_capture_signal` parameter | `self._saturated_capture` attribute | **Correct port** |
| Action | `G->fill(0)` then return | `return` (no W update) | Equivalent |
| Counter reset | Resets `poor_signal_excitation_counter_` = 0 | No counter reset on sat | Minor difference |
| Fire rate on 8-case cohort | Not measured | ~0% (zero on DT_static; rare on DT_mvmt) | Low fire rate |

**Classification: correct port** — saturation gate logic matches AEC3. The minor difference
(AEC3 resets poor_excitation counter on saturation, Python doesn't) is benign since the counter
self-corrects within n_partitions frames.

---

## mu_eff Actual vs AEC3 Effective mu

For reference, assuming a typical render power P (same signal, SpectralSum over respective
n_partitions). **Corrected rate: AEC3 single-channel default is `rate=0.7`, not 0.95.**

| Configuration | Numerator | Denominator | Effective mu per bin |
|--------------|-----------|-------------|----------------------|
| AEC3 coarse (13 parts, steady-state) | rate = **0.7** | SpectralSum(13) ≈ 13P | **0.7 / 13P = 0.054/P** |
| AEC3 coarse (13 parts, delay-event) | rate_initial = 0.9 | SpectralSum(13) ≈ 13P | 0.9 / 13P = 0.069/P |
| Python A.1 OFF (**6** parts) | mu×mu_scale ≈ 0.5×1.0 | EMA(P)×**6** ≈ **6**P | 0.5 / **6**P = **0.083**/P |
| Python A.1 ON (**6** parts) | mu×mu_scale ≈ 0.5×1.0 | SpectralSum(**6**) ≈ **6**P | 0.5 / **6**P = **0.083**/P |

_(n\_partitions=6 runtime-confirmed from v3\_21\_a2\_a3\_fail\_trace.py; filter\_length=832 / hop=160 → ceil=6)_

**Key insight from rate correction**: Python shadow (both A.1 ON and OFF) has effective mu
≈ **1.54× AEC3's steady-state** (0.083/P vs 0.054/P). This means the Python shadow inherently
adapts faster than AEC3's coarse filter. With A.1 ON on short-lag echo:

For short-lag echo (P_0 >> P_1…P_5, SpectralSum ≈ P_0):
- Python A.1 OFF: denom = EMA(P_0)×**6** ≈ **6**P (inflated, stable) → mu ≈ **0.083**/P
- Python A.1 ON: denom ≈ P_0 (instantaneous, not inflated) → mu spikes to ≈ 0.50/P (**6**× jump)
- **A.1 ON → mu jumps 6× in short-lag case** (confirmed by trace: mu_eff_mean increases B0→B1)
- AEC3 does NOT show this spike because SpectralSum(13) aggregates enough frames to buffer instantaneous drops

With mu_scale (DT) = 0.5:
- Python A.1 ON (DT): `0.5×0.5 / P_0 = 0.25/P_0` vs AEC3 `0.7/13P = 0.054/P` → 4.6× AEC3 during DT
- This 4.6× excess explains over-adaptation during doubletalk on short-lag paths

---

## FFT/Window Scale Analysis (**CLOSED** — synthetic test completed)

**Why this matters**: The noise gate threshold (`filter.coarse.noise_gate = 20075344.f`) and
the rate (`0.7`) were tuned for AEC3's internal X² representation. Python's X² (stored in
`X_buf`) is in a different magnitude range due to different FFT/window conventions.

### Known differences

| Parameter | AEC3 | Python |
|-----------|------|--------|
| FFT size | 128 (kFftLength) | **512** (next pow2 above block=320) |
| Block size | 64 samples (4 ms @ 16 kHz) | 320 samples (20 ms @ 16 kHz) |
| n_freqs (bins) | 65 (kFftLengthBy2Plus1) | **257** (fft_size/2+1) |
| Window for render | kSqrtHanning128 applied to second 64 samples (ZeroPaddedFft) | **No window** on far_buffer (full 320-sample block, rfft/512) |
| Total samples per SpectralSum | 13 × 64 = 832 | **6** × 160 = **960** |

Note: Python total SpectralSum duration (960 samples) is slightly longer than AEC3 (832
samples). The dominant scale difference is from FFT normalization, not duration.

### AEC3 window convention

`aec3_fft.cc` uses `ZeroPaddedFft` for render: zero-pads first 64 samples, applies
`kSqrtHanning128` (sqrt of Hanning(128)) to the next 64 samples, then 128-pt FFT.
The `kSqrtHanning128` window has power sum = Σw² ≈ 128 × 3/8 = 48 (Hanning power factor).
Per-bin X² = (amplitude)² × 48 (approx) for a tone in the window.

Python uses rfft(block=320, fft=512) with NO window → full-amplitude tone; per-bin X² ≈ 
(amplitude × 320/2)² = (amplitude × 160)² → 6400 × amplitude².

### Synthetic test results (from v3\_21\_a2\_a3\_fail\_trace.py)

1-kHz tone, amplitude=0.5, 3 seconds at 16 kHz:

| Path | FFT | block | hop | n_parts | per-part \|X\|² | SpectralSum |
|------|-----|-------|-----|---------|----------------|-------------|
| Python | 512 | 320 | 160 | **6** | 6400.0 | **38400.0** |
| AEC3 sim | 128 | 64 | 64 | 13 | 99.9 | 1298.2 |

**Measured ratios**:
- Per-partition |X|² ratio: **64.1×** (Python / AEC3)
- SpectralSum ratio: **29.6×** (Python SpectralSum(6) / AEC3 SpectralSum(13))

**Implication**: The noise gate threshold and rate constant from AEC3 must be scaled by
**29.6×** (SpectralSum ratio) for use in Python's SpectralSum(6) context.

### Corrected Python constants (audit-only)

| Constant | AEC3 value | AEC3 → float | Python SpectralSum(6) equivalent |
|---|---|---|---|
| `filter.coarse.noise_gate` | `20075344.f` (int16²) | `psd_int16_to_float(20075344) = 0.01869` | `0.01869 × 29.6 ≈ **0.553**` |
| `coarse.rate` (steady-state) | `0.7` | — | `0.7 × 29.6 ≈ **20.7**` (mu numerator for parity) |
| `coarse_initial.rate` (delay-event) | `0.9` | — | `0.9 × 29.6 ≈ **26.6**` |

_(Prior values used py\_npar=5 giving ratio=24.7×: noise gate was 0.461, rate was 17.3 — both 17% too low.)_

### Resolution

The 5/13 (= 38%) depth-ratio assumption was incorrect for two reasons:
1. Python has n\_partitions=**6** (not 5), giving 6/13 = 46%
2. Python has NO FFT window while AEC3 uses SqrtHanning128 → 64.1× per-partition scale
   dominates the total SpectralSum ratio (64.1 × 6/13 = **29.6×**, not 6/13 = 0.46)

The previous 3.6× and 2.6× over-aggressiveness estimates were based on the wrong assumption
that the depth ratio was the only difference. The actual SpectralSum ratio is 29.6×, so any
AEC3 constant must be multiplied by 29.6× when ported to Python's SpectralSum(6) context.

**Status: CLOSED.** Correction factors confirmed by synthetic test.

---

## Gate Interaction Diagram

```
AEC3 CoarseFilterUpdateGain::Compute (per frame):
  render_power = SpectralSum(13)              ← 13-partition sum, 832-sample total
  if poor_exc_ctr < 13 OR sat OR startup:
      G = 0 → return
  mu[k] = rate(0.7) / X2[k]                  ← SpectralSum(13) in denom (0.9 during delay-event)
  if X2[k] <= 20075344/32768²: mu[k]=0       ← filter.coarse.noise_gate (NOT echo_model.noise_gate_power)
  mu *= narrowband_mask
  G = mu × E_coarse
  W[p] += G × conj(X[p])

Python PBFDAF._update_weights with B5 (A.1+A.2+A.3+A.4+A.5 ON):
  if saturated: return                        ← A.5 (correct)
  if call_ctr ≤ 6 OR poor_exc < 6:           ← A.3 (correct, but threshold=6 vs 13)
      return
  if stationary: return                       ← always-on, not in AEC3
  mu_scale_arr *= narrowband_mask             ← A.4 (correct)
  denom = SpectralSum(6) + delta              ← A.1 (SpectralSum correct; mu=0.5 mismatched vs rate=0.7)
  mu_eff = (0.5 × mu_scale) / denom          ← rate 0.5 vs AEC3 0.7; depth 6 vs 13 (see §mu_eff table)
  if SpectralSum(6) <= 0.02562: mu=0         ← A.2 (wrong constant: echo_model; correct=0.553 for SpectralSum(6))
  W[p] += mu_eff × error_spec × conj(X[p])
```

---

## Why A.1 Makes e2_coarse Worse (Summary)

The sub-ladder showed e2_coarse DOUBLED on 9xjhiFbG and xFk7igec with A.1 ON. Root cause:

1. **EMA denominator was a stabilizer**: `EMA(power) × n_partitions` provides a smoother,
   lag-dampened denominator. During active echo (high current render), EMA lags → denom is
   larger → mu is smaller → shadow tracks slowly but stably.

2. **SpectralSum is reactive**: Instantaneous sum responds immediately to current render power.
   During high-render frames, SpectralSum → large → mu small (ok). During sudden render drops
   (path changes, movement), SpectralSum → small → mu spikes → over-adaptation.

3. **mu=0.5 was tuned for EMA stability**: The BALANCED preset's `shadow_mu_nlms=0.5` was
   set through 800-case iteration against the EMA denominator. Switching to SpectralSum without
   re-tuning `mu` changes the effective step size in an uncontrolled way.

4. **A.2 over-gates**: The 3.6× over-aggressive noise gate (wrong constant + wrong depth)
   zeros adaptation on bins where AEC3 would still adapt, selectively removing high-value
   learning on lower-power bins.

---

## Classification Table

| Component | Classification | Reason |
|-----------|---------------|--------|
| A.1 X² denominator (SpectralSum swap) | **Port mapping mismatch** | `mu=0.5` co-tuned with EMA; SpectralSum requires `rate≈0.7` (AEC3 default) and/or depth correction; effective mu is 1.85× AEC3 steady-state |
| A.2 noise gate constant | **Port mapping mismatch** | Used `echo_model.noise_gate_power=27509.42f` (wrong path; float units) instead of `filter.coarse.noise_gate=20075344.f` (correct path; int16² units) |
| A.2 noise gate SpectralSum depth | **Port mapping mismatch** | 6 vs 13 partitions; scale factor = 29.6× (64.1× per-part × 6/13); correct threshold = 0.553 |
| A.3 poor excitation counter | **Structural equivalent** | Semantically correct; threshold 5 vs AEC3's 13 reflects different filter depth |
| A.4 narrowband mask | **Correct port** | Logic matches AEC3; low fire rate on current cohort |
| A.5 saturation gate | **Correct port** | Logic matches AEC3; minor counter-reset difference is benign |
| Rate constant (A.1 denominator) | **Port mapping mismatch** | AEC3 `rate=0.7` (single-channel) vs Python `mu=0.5`; Python mu was tuned for EMA, NOT for SpectralSum denominator |

### Bundle B failure classification (per user directive)

The user requested explicit classification of the Bundle B failure cause:

- **a. wrong constant**: YES — A.2 uses echo_model noise gate (wrong path and wrong unit system)
- **b. wrong rate**: YES — AEC3 `rate=0.7` vs Python `mu=0.5`; both were tuned for their respective denominators but they do not transfer directly; Python 1.85× above AEC3 steady-state even with correct denominator type
- **c. FFT/window scale mismatch**: **CLOSED** — 64.1× per-partition (Python rfft/512, no window vs AEC3 ZeroPaddedFft/128 SqrtHanning128); SpectralSum ratio **29.6×** (6-part Python / 13-part AEC3)
- **d. partition-depth mismatch**: YES — **6** vs 13 partitions; Python total 6×160=960 samples vs AEC3 832; affects noise gate aggressiveness and poor\_excitation threshold
- **e. true architectural mismatch**: NOT YET CONFIRMED — the PBFDAF (Python) and AdaptiveFirFilter (AEC3 coarse) are frequency-domain NLMS equivalents; no known architectural incompatibility beyond the calibration mismatches above

---

## What This Means for Bundle B Closure

**The sub-ladder failure is NOT evidence that AEC3's coarse fallback is invalid.**

AEC3's coarse fallback works because all components are co-calibrated:
- `rate=0.7` (single-channel default) matched to SpectralSum(13)
- `noise_gate=20075344` (filter.coarse path) matched to SpectralSum(13)
- `poor_excitation_counter` threshold = n_partitions = 13, protecting high-rate adaptation
- `coarse_initial.rate=0.9` provides a 1.29× boost for 2.5 s after delay events

Our Python shadow (PBFDAF) diverges with Bundle B because:
1. A.1 swaps denominator type without adjusting `mu` (rate mismatch: effective mu 1.85× AEC3 at steady state; 4.6× during DT on short-lag paths)
2. A.2 uses wrong constant source (echo_model instead of filter path) AND wrong depth (**6** vs 13, scale ratio 29.6×); current NOISE\_GATE\_POWER\_FLOAT=0.02562 is wrong; correct SpectralSum(6) threshold=0.553
3. Both errors compound: A.1 inflates mu on short-lag signals → over-adaptation; A.2 under-gates on low-power bins → shadow learns wrong convergence signals

**Path to fixing Bundle B** (if pursued in future, requires 800-case re-bench):

1. **Correct A.2 constant (confirmed bug)**:
   Use `FILTER_NOISE_GATE_FLOAT = psd_int16_to_float(20075344.0) = 0.01869` for the per-partition
   gate. For SpectralSum(6) context: scaled threshold = `0.01869 × 29.6 ≈ 0.553`.
   Do NOT use `NOISE_GATE_POWER_FLOAT = 0.02562` (echo_model path; different unit system).

2. **FFT/window scale** (**CLOSED** — see §FFT/Window Scale Analysis):
   Synthetic test completed. Per-partition ratio = **64.1×**; SpectralSum(6)/SpectralSum(13) = **29.6×**.
   All AEC3 float constants must be multiplied by 29.6× for Python SpectralSum(6) context.

3. **Re-calibrate `shadow_mu_nlms` for SpectralSum denominator**:
   AEC3 uses `rate=0.7` for SpectralSum(13). Python equivalent with SpectralSum(6):
   `mu_target = 0.7 × (SpectralSum_Py / SpectralSum_AEC3) = 0.7 × 29.6 ≈ **20.7**`
   This is 41× the current `shadow_mu_nlms = 0.5` — confirms that A.1 with the current mu=0.5
   is far outside AEC3's operating range. Re-calibration is mandatory for A.1 to work.

4. **Verify poor_excitation threshold**:
   AEC3 threshold = `size_partitions = 13`. Python A.3 uses `n_partitions = 6` (runtime-confirmed).
   Structurally equivalent (same concept, scaled to actual filter partition count). No fix needed.

These are tuning/wiring corrections, NOT algorithm changes. They require 800-case re-bench.
**Bundle B is not yet closed as SHIP or NOSHIP.** Do NOT interpret the sub-ladder failure as
"AEC3 coarse fallback invalid."

---

## Shadow Update Formula Trace Targets

Per user directive: trace these to attribute Bundle B failure sub-causes:

| Trace field | What it measures | Attribute to |
|---|---|---|
| `mu_eff_mean` | Mean effective step size per frame | Rate/denominator mismatch (b, d) |
| `mu_eff_max` | Max effective step size per frame | Short-lag over-adaptation (b, d) |
| `x2_sum` (= denom - delta when A.1 ON) | SpectralSum(**6**) per frame | Denominator value (d) |
| `noise_gate_zero_frac` | Fraction of bins zeroed by A.2 | Noise gate aggressiveness (a) |
| `A3_skip_frac` | Fraction of frames skipped by poor_excitation | Poor-excitation protection (d) |
| `coarse_conv_frac` | Frames where e2_coarse < 0.5 × y2 | Convergence quality (overall) |
| `shadow_W_norm` | L2 norm of shadow weights | Divergence indicator |

These are already partially available from the sub-ladder trace (B0→B5 step). The `mu_eff_mean`
and `mu_eff_max` fields were added to `filters.py` `_c1c5_trace` in the prior session.
Run B0 (all OFF) vs B1 (A.1 only) to isolate rate/denominator attribution.

---

## Open Items / Status

- **CLOSED (confirmed)**: A.2 constant is from wrong source (echo_model vs filter.coarse); correct = `FILTER_NOISE_GATE_FLOAT = psd_int16_to_float(20075344.0) = 0.01869`; SpectralSum(6) threshold ≈ **0.553** (audit only, not production)
- **CLOSED (confirmed)**: AEC3 single-channel `rate=0.7` (not 0.95); `coarse_initial.rate=0.9` for delay-event transient
- **CLOSED (confirmed)**: `echo_model.noise_gate_power = 27509.42f` is in AEC3 float units, NOT int16²; do NOT apply `psd_int16_to_float`
- **CLOSED (confirmed)**: FFT/window scale — per-partition ratio **64.1×**, SpectralSum(6)/SpectralSum(13) = **29.6×** (synthetic test in `python/v3_21_a2_a3_fail_trace.py`)
- **CLOSED (confirmed)**: n\_partitions = **6** (runtime-confirmed: filter\_length=832 / hop=160 → ceil=6; prior "5" was wrong)
- **CLOSED (confirmed)**: noise gate SpectralSum(6) threshold = **0.553** (0.01869 × 29.6); prior "0.461" (24.7×) and "0.4607" were wrong
- **OPEN**: mu re-calibration with A.1 ON — target `shadow_mu_nlms ≈ 20.7` (0.7 × 29.6); requires 800-case bench; not v3.21.x scope
- **Not closed**: Bundle B — neither SHIP nor NOSHIP; calibration corrections (A.2 constant, mu re-calibration) required before 800-case re-bench
