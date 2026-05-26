# AEC3 Linear/Residual Signal-Flow Audit — v3.21.x Substrate

**Date**: 2026-05-26  
**Scope**: v3.21.x AEC3 alignment substrate. No production default flips. No merge/version bump.  
**Method**: Direct code reading of orchestrator + state + residual + filter modules.  
**Rule**: Do not close a module as useless unless its producer AND consumer are both traced.

---

## Deliverable A — AEC3 Signal-Flow Map

For every signal in the four scope paths: **producer → Python equivalent → status**.

Status codes:
- **MATCH** — Python implementation matches AEC3 contract (same signal, same gating).
- **IMPL-OFF** — port exists behind a default-OFF flag; byte-equal when OFF.
- **APPROX** — port exists but differs from AEC3 in threshold, timing, or source.
- **MISSING** — AEC3 produces and consumes this; Python has no equivalent.
- **NO-CONSUMER** — signal computed but nothing reads it in the audio path.
- **ARCH-DIVERGENCE** — structural difference; exact parity is not achievable without redesign.

### A.1 Linear Filter Path

| Signal | AEC3 producer | AEC3 consumer | Python equivalent | Status |
|---|---|---|---|---|
| `E_refined` (linear residual) | PBFDKF `W·X − Y` per-bin | `UseRefinedOutput`, `FormLinearFilterOutput`, ERLE, ERL | `filter.error_spec_windowed` (complex, windowed) | **MATCH** |
| `S_refined` (echo estimate) | `W·X` per-bin refined | `UseRefinedOutput` s2 compute, RES echo_psd | `filter.echo_spec` (complex) | **MATCH** |
| `Y` (capture spectrum) | subtractor | `UseLinearFilterOutput` fallback, nearend_psd | `filter.near_spec` (complex) | **MATCH** |
| `X` (render spectrum) | render buffer | ERLE/ERL render_psd, coarse update | `filter.far_spec` (complex) | **MATCH** |
| `E_coarse` (shadow residual) | PBFDAF `W_coarse·X − Y` | `UseRefinedOutput` e2_coarse, coarse_converged | `shadow_filter.error_spec` (complex) | **MATCH** (present when shadow ON) |
| `S_coarse` (shadow echo) | PBFDAF `W_coarse·X` | `UseRefinedOutput` s2_coarse, `FormLinearFilterOutput` | `shadow_filter.echo_spec` (complex) | **MATCH** (present when shadow ON) |
| `UseRefinedOutput` predicate | `SubtractorOutput` e2/y2 fields | `FormLinearFilterOutput` path selection | `_aec3_select_linear_filter_output()` at orchestrator:3765 | **IMPL-OFF** (`use_refined_output_selection_for_linear_path=False`) |
| `FormLinearFilterOutput` crossfade | `_refined_filter_output_last_selected` hysteresis, per-sample lerp over kBlockSize | `echo_remover.cc:447-473` → final E_sel, S_sel spectra | **Binary spectral switch** only; no per-sample crossfade | **ARCH-DIVERGENCE** |
| `UseLinearFilterOutput` (Y vs E) | `aec_state.UsableLinearEstimate()` | `echo_remover.cc:475` — nearend_psd input to RES/SG | `use_linear_filter_output_selection_for_final_output` flag | **IMPL-OFF** |
| `S2_linear` → `echo_psd` | selected `S_refined` or `S_coarse` per UseRefinedOutput | `ResidualEchoEstimator.estimate(s2_linear=...)` | `echo_psd = |_sel_echo_spec|² * SCALE` | **MATCH** when URO OFF; **IMPL-OFF** when URO ON |
| `nearend_pwr` (E² vs Y² select) | `echo_remover.cc:452,495-501` E²=min(E²,Y²) when usable | `DominantNearendDetector`, `_gain_to_no_audible_echo` | `np.minimum(error_psd, near_psd)` gated by `e2_y2_clamp_enabled` | **IMPL-OFF** (`e2_y2_clamp_enabled=True` in v3.21.5; verify default) |

**ARCH-DIVERGENCE note — FormLinearFilterOutput:**  
AEC3 crossfades sample-by-sample using `_refined_filter_output_last_selected` across kBlockSize=64 samples when switching coarse↔refined. Python switches the spectrum atomically each hop. On mode-switch frames, this produces a one-hop spectral discontinuity rather than a 4 ms smooth ramp. This cannot be fixed by a flag — it requires per-sample time-domain blending before FFT or equivalent frequency-domain smoothing.

---

### A.2 Filter State / Quality Path

| Signal | AEC3 producer | AEC3 consumer | Python equivalent | Status |
|---|---|---|---|---|
| `refined_converged` | `subtractor_output_analyzer.cc:38-40`: `e²_ref < 0.5·y²` AND `y² > 50²·64` | `any_filter_converged`, ERLE update gate | `_refined_conv` (inline `_aec3_post:3935`) | **MATCH** |
| `coarse_converged` | `subtractor_output_analyzer.cc:42-44`: `e²_coa < 0.05·y²` AND `y² > 50²·64` | `any_filter_converged` | `_coarse_conv` (inline `_aec3_post:3960`) | **MATCH** |
| `coarse_converged_relaxed` | `subtractor_output_analyzer.cc:50-51`: `e²_coa < 0.3·y²` AND `y² > 20²·64` | `TransparentModeImpl` input (only) | `_coarse_conv_relaxed` (audit-only, `coarse_filter_converged_relaxed_enabled`) | **NO-CONSUMER** (TM input bypassed; LegacyTransparentModeImpl ignores it) |
| `any_filter_converged` | `refined_conv OR coarse_conv` | `AecState.update(bridge.filter_converged)`, ERLE/ERL update gate | `_aec3_converged = _refined_conv OR _coarse_conv` → `bridge.filter_converged` | **MATCH** |
| `any_filter_diverged` | `SubtractorOutputAnalyzer` divergence signal | `all_filters_diverged` in TM update | `bridge.divergence_indicator > 1.0` (P_trace × shadow_e_ratio) | **APPROX** (different computation) |
| `all_filters_diverged` | `!any_filter_converged AND any_filter_diverged` | `TransparentMode.update()` | `(not any_filter_converged) AND bridge.divergence_indicator > 1.0` (aec_state.py:333) | **APPROX** |
| `external_delay` quality | `EchoPathDelayEstimator` → `DelayQuality.COARSE` or `REFINED` | `FilteringQualityAnalyzer` gate 3 | Always `DelayQuality.REFINED` when `_delay_active` | **APPROX** (quality always REFINED regardless of confidence) |
| `convergence_seen` latch | `FilteringQualityAnalyzer`: set True on first `any_filter_converged`; no reset except path-change | `usable_linear_estimate` gate 4 | `_convergence_seen` in `filter_quality.py:62` | **MATCH** |
| `usable_linear_estimate` | 4-gate AND: !initial_state ∧ !TM ∧ !saturated ∧ (ext_delay OR conv_seen) | `echo_remover.cc` nearend select, ERLE mode | `aec_state.usable_linear_estimate()` via `FilteringQualityAnalyzer` | **MATCH** (gate logic); **APPROX** (TM input) |
| `filter_analyzer_consistent` | `FilterAnalyzer.any_filter_consistent()` (peak stability) | `TransparentMode` input, usable_linear gate-3 optional AND | `_filter_analyzer.any_filter_consistent()` when `filter_analyzer_enabled=True` | **IMPL-OFF** (`filter_analyzer_enabled=False`); fallback proxy = `conv AND ext_delay` |
| `subtractor_s_refined_max_abs` | `SubtractorOutput.s_refined` peak absolute value | `SaturationDetector.update()` echo scaling | Always `0.0` fed to `aec_state.update()` | **MISSING** (field not computed in orchestrator) |
| `subtractor_s_coarse_max_abs` | `SubtractorOutput.s_coarse` peak absolute value | `SaturationDetector.update()` echo scaling | Always `0.0` | **MISSING** |
| `TransparentMode` state | `LegacyTransparentModeImpl` (HMM not ported) | `usable_linear` gate 2; `echo_remover.cc` gain-pass-through | `transparent_mode.py` LegacyTransparentModeImpl | **IMPL-OFF** (`transparent_mode_enabled=False`); HMM is **ARCH-DIVERGENCE** |

---

### A.3 Residual Path

| Signal | AEC3 producer | AEC3 consumer | Python equivalent | Status |
|---|---|---|---|---|
| `S2_linear` (echo PSD) | `FormLinearFilterOutput` selected `S²` | `ResidualEchoEstimator.estimate(s2_linear)` linear mode | `echo_psd = |selected_echo_spec|² * SCALE` | **MATCH** when URO OFF; **IMPL-OFF** when ON |
| `R2` bounded residual | `REE.estimate()` linear: `S2/ERLE + reverb`; nonlinear: echo model + reverb | `SuppressionGain.get_gain(residual_echo_spectrum)` | `r2` from `_aec3_ree.estimate()` | **MATCH** for linear mode; **APPROX** for nonlinear |
| `R2_unb` unbounded residual | Same as R2 but with unbounded ERLE | `DominantNearendDetector` ENR | `r2_unb` from `_aec3_ree.estimate()` | **MATCH** |
| ERLE per-bin (bounded) | `SubbandErleEstimator`: ratio `X²/Y²` vs `X²/E²` smoothed, gated by `filter_converged` | `R2 = S2 / ERLE` in linear mode | `aec_state.erle()` → `_erle_estimator.erle()` | **MATCH** |
| ERLE per-bin (unbounded) | Same without cap | `R2_unb = S2 / ERLE_unb` | `aec_state.erle_unbounded()` | **MATCH** |
| ERLE SDE refinement | `SignalDependentErleEstimator`: per-section per-subband using `|W|²` and `X²` history | `R2 / R2_unb` precision on high-echo tails | `signal_dependent_erle.py` behind `signal_dependent_erle_sections > 0` | **IMPL-OFF** |
| ERL per-bin | `ErlEstimator`: `X²/Y²` ratio, converged-gated | `min_echo_power` in `_get_min_gain` | `aec_state.erl()` via `ErlEstimator` | **MATCH** |
| Reverb tail (linear mode) | `ReverbModel.update(X², |W|²/X², decay)` → per-bin tail mass | Added to `R2 + R2_unb` | `_reverb_model.update()` in `REE._update_reverb_linear()` | **MATCH** for fixed-decay path |
| `reverb_decay` adaptive | `ReverbDecayEstimator`: spectral auto-correlation of `W` tail | `ReverbModel` decay parameter | `_reverb_decay_est` behind `reverb_cfg.use_adaptive_decay=False` | **IMPL-OFF** |
| `reverb_frequency_response` | `ReverbFrequencyResponse`: per-bin late-reflection shape from filter taps | `R2` per-bin shaping (not just scalar mass) | `_reverb_freq_resp` behind `reverb_cfg.use_freq_response=False` | **IMPL-OFF** |
| `aec_state.get_reverb_frequency_response()` | AecState reverb port (filled by ReverbDecayEstimator) | `REE.estimate()` would read if non-zero | Returns `np.zeros(n_bins)` always — stub | **MISSING** (no fill path; AecState port is a dead stub) |
| Stationarity R² zeroing | `StationarityEstimator.band_stationary()` → zero R² on stationary render bands | Prevents `_get_min_gain` floor from suppressing NE on stationary far-end | `_aec3_stationarity.band_stationary_mask()` → `np.where(mask, 0, r2)` gated by `use_stationarity_properties` | **IMPL-OFF** (config flag default False; stationarity estimator IS instantiated) |
| NE staleness (REE→SG boundary) | AEC3: `dominant_nearend` computed once before both REE and SG | `REE.estimate(dominant_nearend)` and `SG.get_gain()` internal `_dominant_nearend.update()` | Python: `dominant_ne = sg.is_dominant_nearend()` BEFORE `get_gain`; SG updates detector INSIDE `get_gain` → 1-frame stale | **APPROX** (Bug 4; 1-frame timing mismatch on NE transitions) |

---

### A.4 Gain Decision Path

| Signal | AEC3 producer | AEC3 consumer | Python equivalent | Status |
|---|---|---|---|---|
| `nearend_spectrum` (to SG) | `echo_remover.cc:452,495-501`: E² (usable) or Y² (not usable), clamped `min(E²,Y²)` | `_gain_to_no_audible_echo`, `DominantNearendDetector` | `nearend_pwr` in `_aec3_post:4270-4276` | **MATCH** (e2_y2_clamp gated, verify default) |
| `DominantNearendDetector` update | `suppression_gain.cc:373`: `update(nearend, R2_unb, comfort_noise)` | `_ne_state_for_gain_rules()` read by `_lower_band_gain` | `_DominantNearendDetector.update()` called INSIDE `get_gain` | **MATCH** for logic; **APPROX** for timing vs REE |
| `_get_min_gain` floor | `min_echo_power / weighted_R2`, clamped to 1.0 | Lower bound on `G` per bin | `_get_min_gain()` in suppression_gain.py:653 | **MATCH** |
| `_get_max_gain` ramp | `last_gain * inc_factor` (NE: larger inc), floor at `floor_first_increase` | Upper bound on `G` per bin | `_get_max_gain()` in suppression_gain.py:641 | **MATCH** |
| `_limit_lf_gains` | Hard cap on LF bins (≤ ~250 Hz) to prevent LF distortion | Applied always | `_limit_lf_gains()` suppression_gain.py:301 | **MATCH** |
| `_limit_hf_gains` | HF cap when NOT NE state AND conservative_hf_suppression | Prevents suppression of HF NE components | `_limit_hf_gains()` suppression_gain.py:307 | **MATCH** |
| `comfort_noise_spectrum` | `CNG noise floor` from capture EMA | `_gain_to_no_audible_echo` noise term | Per-bin EMA noise PSD (`_aec3_noise_psd`), lazy-init | **APPROX** (AEC3 CNG uses `SubtractedNoiseEstimator`; we use simple EMA) |
| Per-bin gain reason (min/max/ENR/HF) | AEC3 internal — not exposed | Debugging only | **Not present in any diag field** | **MISSING** (diag gap) |
| `LowNoiseRenderDetector` | Detects low-render energy → relaxes gain floor | `_get_min_gain` relaxation path | `_LowNoiseRenderDetector` in suppression_gain.py:229 | **MATCH** |
| `SubbandNearendDetector` | Optional per-subband NE refinement (flag `use_subband_nearend_detection`) | `_ne_state_for_gain_rules` override | `_SubbandNearendDetector` present but flag default False | **IMPL-OFF** |
| `initial_state` gain policy | `SuppressionGain.set_initial_state(True/False)` → `lf_smoothing_during_initial_phase` | Looser LF gain during initial convergence window | `set_initial_state()` method exists; called from A.1 delay-change chain | **IMPL-OFF** (`use_full_delay_change_chain=False`) |

---

## Deliverable B — Missing Substrate (Real Downstream Consumers Only)

Items listed only when AEC3's upstream producer AND downstream consumer are both present or reachable in our code. Items marked "low-impact" are included for completeness but are not blocking.

### B.1 FormLinearFilterOutput crossfade [ARCH-DIVERGENCE, HIGH]

**What AEC3 does**: `echo_remover.cc:447-473` crossfades the output `E_sel` sample-by-sample using a hysteresis flag `_refined_filter_output_last_selected`. On a switch, it ramps over kBlockSize=64 samples using a lerp weight `alpha` from the previous and new outputs.

**What we do**: Switch `error_spec_windowed` and `echo_spec` atomically (single spectrum swap) when `UseRefinedOutput` fires.

**Real consumer**: Final output audio + `nearend_psd/echo_psd` into RES. Spectral discontinuity on switch frames → audible click candidate + per-frame RES spike.

**Substrate needed**: Per-sample crossfade in `_aec3_select_linear_filter_output()` or a frequency-domain crossfade approximation (Hann-weighted blend of adjacent-hop spectra). Cannot be done as a single flag flip.

**Scope**: Only matters when `use_refined_output_selection_for_linear_path=True`.

---

### B.2 `subtractor_s_refined_max_abs` / `s_coarse_max_abs` [MISSING, LOW]

**What AEC3 does**: `SaturationDetector` in `aec_state.cc:219-227` scales its render-based saturation estimate by `max(s_refined_max_abs, s_coarse_max_abs)` — the peak absolute value of the echo estimate (not the capture) in the current block.

**What we do**: Always pass `subtractor_s_refined_max_abs=0.0` and `subtractor_s_coarse_max_abs=0.0` to `aec_state.update()` (orchestrator:4087-4100).

**Real consumer**: `SaturationDetector` → `capture_signal_saturation` → gate 3 of `usable_linear_estimate`. Low impact: saturation events are rare in our corpus and the capture-signal gate path already fires.

**Substrate needed**: Compute `float(np.max(np.abs(self.filter.echo_spec)))` (and shadow equivalent) and pass to `aec_state.update()`. One-liner.

---

### B.3 `aec_state.get_reverb_frequency_response()` stub [MISSING, MEDIUM]

**What AEC3 does**: `ReverbDecayEstimator` fills `AecState._reverb_frequency_response` (a per-bin decay profile). `ResidualEchoEstimator` reads it via `aec_state.get_reverb_frequency_response()`.

**What we do**: `AecState.get_reverb_frequency_response()` returns `np.zeros(n_bins)` always (aec_state.py:178-180). `REE._update_reverb_linear()` calls `_reverb_freq_resp` directly from the REE's own module — bypassing the AecState channel entirely. The AecState stub is dead.

**Real consumer**: Would be `REE.estimate()` → reverb tail per-bin shaping → `r2` accuracy on reverberant cases.

**Substrate needed**: Wire `aec_state.get_reverb_frequency_response()` to return REE's `_reverb_freq_resp.tail_response` when that module is active. Currently a one-liner is impossible because the REE doesn't register itself on AecState — requires a back-reference or a shared accessor.

---

### B.4 `DelayQuality.COARSE` vs `REFINED` distinction [APPROX, MEDIUM]

**What AEC3 does**: `EchoPathDelayEstimator` emits `DelayQuality.COARSE` (first-pass estimate) and `DelayQuality.REFINED` (converged estimate). `FilteringQualityAnalyzer.gate3` requires `quality >= REFINED` before treating the external delay as valid.

**What we do**: Always construct `DelayEstimate(quality=DelayQuality.REFINED)` once `_delay_active` is True (orchestrator:4030). This means `usable_linear` gate 3 flips True as soon as any delay estimate arrives, not when it is refined.

**Real consumer**: `FilteringQualityAnalyzer` gate 3 → `usable_linear_estimate` → ERLE mode, R² linear mode.

**Substrate needed**: Map our delay estimator confidence state (`is_solid`, `delay_status`) to `DelayQuality.COARSE` vs `REFINED`. Requires adding a confidence accessor to `EchoPathDelayEstimator`.

---

### B.5 Per-bin gain reason attribution [MISSING, MEDIUM — diag only]

**What AEC3 does**: Not exposed externally; internal to `_lower_band_gain`.

**What we have**: `gain_5/30/50/100/200` point samples only. No way to know whether min_gain, max_gain, ENR curve, LF cap, or HF cap is the binding constraint at any bin.

**Real consumer**: Root-cause debugging of suppression behavior. No audio path impact.

**Substrate needed**: Return (or trace) a per-bin label array from `_lower_band_gain` indicating which constraint is binding. 5 labels: `enr_curve / min_floor / max_ramp / lf_cap / hf_cap`. Diag-only; gated by `trace_hf_chain`.

---

### B.6 NE state staleness (Bug 4): REE reads pre-gain NE; SG updates inside get_gain [APPROX, LOW]

**What AEC3 does**: `echo_remover.cc` calls `UpdateCaptureLevelEstimate()` → `DominantNearendDetector.Update()` BEFORE `ReverbContributions` computation and BEFORE `SuppressorGain.GetGain()`. Both REE and SG see the same NE state for a given frame.

**What we do**: `dominant_ne = sg.is_dominant_nearend()` reads the PREVIOUS frame's state (orchestrator:4138). `sg.get_gain()` internally calls `_dominant_nearend.update()` → state changes INSIDE the gain call. REE and SG see different NE states on NE-transition frames.

**Real consumer**: `REE.estimate(dominant_nearend=...)` → reverb decay mild/normal selection. On NE transitions: 1-frame wrong reverb decay.

**Substrate needed**: Extract `DominantNearendDetector.update()` call OUT of `get_gain()` and run it in `_aec3_post` before `REE.estimate()`. Requires refactoring `SuppressionGain` interface. Medium complexity.

---

### B.7 `FilterAnalyzer` default-OFF → `TransparentMode` input is APPROX [IMPL-OFF, MEDIUM]

**What AEC3 does**: `FilterAnalyzer.any_filter_consistent()` = peak-index stable AND coherence high for ≥ N frames. This drives TM's `any_filter_consistent` input.

**What we do**: `any_filter_consistent = any_filter_converged AND external_delay is not None` (aec_state.py:330-332 fallback). This fires early (as soon as first convergence + delay estimate) and does not check peak stability.

**Real consumer**: `TransparentMode.update()` → `TM_active` → `usable_linear` gate 2 (when TM enabled).

**Substrate needed**: `filter_analyzer_enabled=True` is the fix. Blocked by FilterAnalyzer needing `get_time_domain_filter()` per-hop IFFT cost — already implemented, just default-OFF. No new code needed, only config cost assessment.

---

## Deliverable C — Diag Schema

Proposed default-OFF trace fields that reconstruct one frame's complete decision chain.  
All fields must be gated by `trace_hf_chain` (already default-OFF in production).  
Fields marked with `[exists]` are already present in `_hf_chain_trace`.

### C.1 Linear filter path trace (add to `_hf_chain_trace`)

```python
# --- C.1 Linear filter selection ---
'uro_fire'             [exists]   # UseRefinedOutput fired
'uro_cond1'            [exists]   # coarse cleaner (cond1)
'uro_cond2'            [exists]   # refined diverged (cond2)
'uro_path'             [exists]   # 'refined' | 'coarse'
'uro_s2_refined'       [exists]   # echo PSD from refined (sum)
'uro_s2_coarse'        [exists]   # echo PSD from coarse (sum)

# NEW — need these:
'ulo_fire'             # UseLinearFilterOutput: True = used E, False = used Y
'ulo_nearend_src'      # 'E' | 'Y' — what nearend_pwr was built from
'form_crossfade_active' # True when coarse↔refined switch occurred this frame (ARCH gap marker)
'sel_echo_psd_sum'     # sum of echo_psd (after URO selection; tracks which path feeds REE)
'sel_error_psd_sum'    # sum of error_psd (same)
```

### C.2 Filter state / quality trace (add to `_hf_chain_trace`)

```python
# --- C.2 Filter state ---
'refined_conv'         [exists]   # e2_refined < 0.5*y2
'coarse_conv'          [exists]   # e2_coarse < 0.05*y2
'aec3_converged'       [exists]   # any_filter_converged
'convergence_seen'     [exists]   # latch
'ext_delay_available'  [exists]   # external_delay is not None

# NEW — need these:
'ext_delay_quality'    # 'COARSE' | 'REFINED' | 'NONE' — delay confidence state
'usable_linear'        [exists]   # final gate output (already in _diag, not in hf_chain)
'initial_state_active' [exists in _diag]  # should mirror to hf_chain
'transparent_mode_active' [exists]  # in hf_chain
'filter_analyzer_consistent' [exists]  # in hf_chain

# Gate-by-gate usable_linear attribution (4 signals):
'ul_gate1_pass'        # !initial_state
'ul_gate2_pass'        # !transparent_mode OR tm_disabled
'ul_gate3_pass'        # ext_delay OR convergence_seen
'ul_gate4_pass'        # !saturated_capture
```

### C.3 Residual path trace (add to `_hf_chain_trace`)

```python
# --- C.3 Residual echo ---
'r2_to_s2_ratio'       [exists]   # R2_sum / S2_sum — ERLE effectiveness
'r2_5/30/100'          [exists]   # per-bin samples
'erle_5/30/100'        [exists]   # per-bin ERLE samples

# NEW — need these:
'erle_sum'             # total ERLE (weighted sum) — single number for trend
'r2_linear_frac'       # fraction of R2 that came from S2/ERLE (vs reverb + noise model)
'r2_reverb_sum'        # reverb tail contribution to R2 (from reverb_model.reverb)
'r2_nonlinear_sum'     # nonlinear echo model contribution to R2
'dominant_ne_pre_ree'  [exists as 'dominant_ne_prev_for_ree']  # NE state fed to REE
'dominant_ne_post_sg'  [exists as 'dominant_ne_after_gain']    # NE state after SG update
'stationarity_kill_frac' [exists as 'r2_mask_kill_ratio']      # fraction of R2 zeroed

# SDE (when enabled):
'sde_erle_refinement_sum'  # |SDE_ERLE - SubbandERLE| mean — shows SDE contribution
```

### C.4 Gain decision trace (add to `_hf_chain_trace`)

```python
# --- C.4 Gain decision ---
'gain_5/30/50/100/200'  [exists]  # per-bin point samples

# NEW — need these:
'gain_reason_5'         # 'enr_curve' | 'min_floor' | 'max_ramp' | 'lf_cap' | 'hf_cap'
'gain_reason_100'       # same for mid-band
'gain_reason_200'       # same for HF
'min_gain_5'            # _get_min_gain value at bin 5 (echo floor)
'min_gain_100'          # same at bin 100
'max_gain_5'            # _get_max_gain value at bin 5 (rate limiter)
'max_gain_100'          # same at bin 100
'nearend_pwr_src'       # 'E_clamped' | 'E_raw' | 'Y' — what nearend_pwr came from
'comfort_noise_sum'     # total comfort noise PSD (tracks noise floor estimate)
'ne_trigger_counter'    [exists]   # DominantNearend trigger counter
'ne_hold_counter'       [exists]   # DominantNearend hold counter
'is_nearend_state'      [exists]   # current NE decision
'hf_cap_fired'          [exists]   # HF limiter active
```

### C.5 Full per-frame reconstruction snapshot (new concept)

One structured `_aec3_frame_snap` dict (not inside `_hf_chain_trace`, separate list) containing the 10 signals needed to reconstruct a single frame's entire decision chain. Useful for side-by-side comparison of two configs at the same frame:

```python
_aec3_frame_snap = {
    # Input spectra (log-power, 5 bins: 0, 32, 64, 128, 256)
    'near_psd_db':    [dB at 5 bins],
    'echo_psd_db':    [dB at 5 bins],   # after URO selection
    'r2_db':          [dB at 5 bins],
    'erle_db':        [dB at 5 bins],

    # State signals
    'usable_linear':  bool,
    'dominant_ne':    bool,
    'initial_state':  bool,
    'tm_active':      bool,

    # Gain output
    'gain_db':        [dB at 5 bins],
    'gain_reason':    [str at 5 bins],   # min/max/enr/lf/hf
}
```

---

## Deliverable D — Experiment Plan

### D.1 Phasing rule

Only run AECMOS after:
1. **Substrate pass**: the target signals have a correct Python producer and consumer traced.
2. **Diag sanity**: `_aec3_frame_snap` on ≥ 3 stress cases shows the expected signal values (not placeholders / zeros / infinity).
3. **Flag-group definition**: the combination to test is named by mechanism, not by variant letter.

No 800-case bench until a grouped bundle passes trace sanity AND 12-case.

---

### D.2 Experiment groups (mechanism-driven, not patch-driven)

#### Group 1 — UseRefinedOutput + crossfade fix (prerequisite: B.1 crossfade substrate)

**What to enable together**:
- `use_refined_output_selection_for_linear_path=True`
- Per-sample or per-hop crossfade implementation (B.1)
- `use_coarse_e2_time_domain_parity=True` (aligns e2_coarse source)

**Trace sanity check** (before 12-case):
- `_aec3_frame_snap` on wVYSGVTT/xFk7 in DT_mvmt window: `uro_fire > 0`, `sel_echo_psd_sum` varies between refined/coarse, `form_crossfade_active` is False or bounded
- Verify: no frame where `sel_echo_psd_sum` spikes by >10 dB vs adjacent frames (crossfade working)

**12-case target**: wVYSGVTT/xFk7 DT_mvmt ≥ 0.0 Δdeg vs A baseline; ZJYUt0O0 ≥ −0.05.

---

#### Group 2 — UseLinearFilterOutput (Y/E select) — independent of Group 1

**What to enable together**:
- `use_linear_filter_output_selection_for_final_output=True`
- Depends on: `usable_linear_estimate()` working correctly (currently MATCH for gate logic)

**Trace sanity check**:
- `ulo_nearend_src` on DT_mvmt stress cases: expect `'Y'` during under-convergence windows, `'E'` after convergence
- `usable_linear` rate on DT_static cases: should be 60%+ of active frames (current baseline)

**12-case target**: DT_static Δdeg ≥ 0.0; no DT_mvmt regression > 0.1 dB.

---

#### Group 3 — ERLE / ERL substrate quality bundle

**What to enable together** (substrate completions, not new logic):
- `subtractor_s_refined_max_abs` pass-through fix (B.2 — one-liner)
- `ext_delay_quality` COARSE/REFINED distinction (B.4)
- SDE (signal_dependent_erle_sections ≥ 1) once B.4 is stable

**Trace sanity check**:
- `erle_5/30/100` on DT_static nVUnxqHLr: should be > 1.0 consistently after convergence
- `erle_sum` trace: should rise monotonically during filter convergence, not stay at 1.0

**12-case target**: No regression in FS_static echo bucket; DT_static ERLE sanity confirmed.

---

#### Group 4 — Reverb + stationarity bundle

**What to enable together**:
- `reverb_cfg.use_adaptive_decay=True` (ReverbDecayEstimator, needs filter taps)
- `reverb_cfg.use_freq_response=True` (ReverbFrequencyResponse)
- Stationarity R² zeroing (already instantiated, just config-gate OFF)
- AecState reverb stub wiring (B.3)

**Trace sanity check**:
- `r2_reverb_sum` non-zero and < 10× `r2_linear_frac` on FS_static cases
- `stationarity_kill_frac` > 0 on cases with known stationary far-end hum (E0l0/wJVP cohort)

**12-case target**: FS_static Δecho ≥ −0.02 (reverb should not hurt); DT_static neutral.

---

#### Group 5 — Gain decision substrate fix bundle

**What to enable together**:
- NE staleness fix (B.6 — extract `_dominant_nearend.update()` out of `get_gain`)
- Per-bin gain reason diag (C.4 additions)
- `SubbandNearendDetector` enable (depends on NE staleness fix first)

**Trace sanity check**:
- `dominant_ne_pre_ree` == `dominant_ne_post_sg` on all DT_static cases (no staleness)
- `gain_reason_100` and `gain_reason_200` per-case distribution: `hf_cap` should fire less on DT frames

**12-case target**: DT_static Δdeg ≥ 0.0; verify no silent gain-rule change from staleness fix.

---

### D.3 Order

Groups are listed by dependency, not priority:

1. **Group 3 first** (substrate correctness — no AECMOS risk, just fixes broken inputs)
2. **Group 1** (requires B.1 crossfade — most complex; don't combine with others until tested alone)
3. **Group 2** (can run in parallel with Group 1 in separate bench once Group 3 is done)
4. **Group 4** (reverb + stationarity; run after Group 2/3 baseline is stable)
5. **Group 5** (gain decision; run last — requires NE staleness fix which touches SuppressionGain interface)

Do not combine groups until each passes 12-case independently.

---

## Summary: Open Items by Priority

| # | Item | Type | Priority | Blocking what |
|---|---|---|---|---|
| B.1 | FormLinearFilterOutput crossfade | ARCH-DIVERGENCE | HIGH | Group 1 (URO) |
| B.4 | DelayQuality COARSE/REFINED | APPROX | MEDIUM | Group 3 |
| B.3 | aec_state.get_reverb_frequency_response() stub wiring | MISSING | MEDIUM | Group 4 |
| B.6 | NE staleness (REE vs SG) | APPROX | MEDIUM | Group 5 |
| B.7 | FilterAnalyzer default-OFF (TM input APPROX) | IMPL-OFF | MEDIUM | TransparentMode accuracy |
| C.1-C.4 | New diag fields | MISSING (diag) | MEDIUM | All groups need trace sanity |
| B.2 | subtractor_s_max_abs = 0 | MISSING | LOW | SaturationDetector accuracy |
| B.5 | Per-bin gain reason | MISSING (diag) | LOW | Group 5 debugging |

---

## Constraints (carry-forward)

- No production default flip in this audit cycle.
- No merge or version bump.
- No 800-case bench until a group bundle passes 12-case.
- Do not close a module based on single-patch verdict. Wait for group trace sanity.
- v3.22 = beyond-AEC3 optimization. Everything in this document is v3.21.x alignment work.
