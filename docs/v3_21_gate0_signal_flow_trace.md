# Gate 0 Signal-Flow Trace — v3.21 AEC3 alignment substrate

**Flags**: all alignment flags OFF (byte-equal baseline)

**Cases**: xFk7 / wVYSGVTT / ZJYUt000 (DT_mvmt)


## Source-flow matrix

| Signal | AEC3 source | Python diag field | Observed (xFk7 / wVYSGVTT / ZJYUt000) | Remaining gap |
|---|---|---|---|---|
| **A — FormLinearFilterOutput (30-sample SignalTransition)** | | | | |
| `uro_fire` | `echo_remover.cc:112-147 UseRefinedOutput` | `UseRefinedOutput` | 0.0% / 0.0% / 0.0% | crossfade not yet verified |
| `uro_cond1` | `echo_remover.cc:117` | `cond_coarse_cleaner` | 0.0% / 0.0% / 0.0% | energy thresholds vs AEC3 int16 |
| `uro_cond2` | `echo_remover.cc:122` | `cond_refined_diverged` | 0.0% / 0.0% / 0.0% | MATCH |
| `uro_path` | `echo_remover.cc:134 FormLinearFilterOutput output` | `uro_path` | refined / refined / refined | crossfade not yet active |
| `form_transition_active` | `signal_transition.cc UpdateBlock` | `form_linear_filter_crossfade_enabled` | 0.0% / 0.0% / 0.0% | OFF by default (flag audit) |
| **B — ext_delay / usable_linear gates** | | | | |
| `ext_delay_present` | `aec_state.cc FilterQuality gate 3 input` | `ext_delay_present` | 97.2% / 98.0% / 96.3% | semantics: always vs trusted-only |
| `ext_delay_source` | `aec_state.cc:203-206 FilterDelay` | `ext_delay_source` | is_solid / is_solid / is_solid | fixed/is_solid/always/none |
| `delay_is_solid` | `RenderDelayController::DelayIsValid` | `delay_is_solid` | 97.2% / 98.0% / 96.3% | Python uses legacy delay_est.is_solid |
| `ul_gate1_startup` | `filter_quality.cc:100 startup_ok` | `gate1_pass()` | 96.2% / 96.3% / 94.1% | MATCH |
| `ul_gate2_reset` | `filter_quality.cc:101 reset_ok` | `gate2_pass()` | 96.2% / 96.3% / 94.1% | MATCH |
| `ul_gate3_ext_delay` | `filter_quality.cc:115 external_delay branch` | `ext_delay_present` | 97.2% / 98.0% / 96.3% | SEMANTICS: Python gives ext_delay when delay_active; AEC3 only when authoritative source reports |
| `ul_gate3_conv_seen` | `filter_quality.cc:110 convergence_seen branch` | `convergence_seen` | 97.1% / 96.9% / 96.3% | MATCH |
| `ul_gate4_not_transparent` | `filter_quality.cc:121 not transparent_mode` | `not transparent_mode_active()` | 100.0% / 100.0% / 100.0% | MATCH |
| `usable_linear` | `aec_state.cc UsableLinearEstimate()` | `usable_linear_estimate()` | 96.1% / 96.3% / 94.1% | MATCH (all 4 gates) |
| **C — SaturationDetector subtractor inputs** | | | | |
| `sat_s_refined_max_abs` | `subtractor_output.cc s_refined max_abs` | `saturation_subtractor_inputs_enabled` | 0.000 / 0.000 / 0.000 | OFF by default (flag audit) |
| `sat_s_coarse_max_abs` | `subtractor_output.cc s_coarse max_abs` | `saturation_subtractor_inputs_enabled` | 0.000 / 0.000 / 0.000 | OFF by default (flag audit) |
| **D — SuppressionGain attribution** | | | | |
| `min_gain_5` | `suppression_gain.cc GetMinGain bin5` | `_last_lower_band_snap[min_gain_5]` | 0.436 / 0.641 / 0.469 | MATCH |
| `max_gain_5` | `suppression_gain.cc GetMaxGain bin5` | `_last_lower_band_snap[max_gain_5]` | 0.498 / 0.705 / 0.631 | MATCH |
| `G_pre_clip_5` | `suppression_gain.cc GainToNoAudibleEcho bin5` | `_last_lower_band_snap[G_pre_clip_5]` | 0.490 / 0.703 / 0.603 | MATCH |
| `gain_reason_5` | `suppression_gain.cc clamp outcome bin5` | `_last_lower_band_snap[gain_reason_5]` | G / G / G | IMPL-ONLY: binding constraint attribution |
| `gain_reason_100` | `suppression_gain.cc clamp outcome bin100` | `_last_lower_band_snap[gain_reason_100]` | G / G / G | IMPL-ONLY: binding constraint attribution |
| `hf_lim_applied` | `suppression_gain.cc HF limiter condition` | `_last_lower_band_snap[hf_lim_applied]` | 98.2% / 99.8% / 100.0% | MATCH: !ne_state || conservative |

## Flag status

| Flag | Default | Purpose |
|---|---|---|
| `form_linear_filter_crossfade_enabled` | False | 30-sample SignalTransition parity |
| `saturation_subtractor_inputs_enabled` | False | SaturationDetector echo-estimate inputs |
| `use_refined_output_selection_for_linear_path` | False | URO per-frame selection |
| `usable_linear_trusted_external_delay_only` | False | Restrict ext_delay to solid/fixed delay |

## Next step: run with flags ON one at a time to isolate behavioral deltas.
