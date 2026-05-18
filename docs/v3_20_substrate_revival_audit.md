# v3.20 Substrate Revival — bug audit + fixes (2026-05-17)

## TL;DR

Phase 2 arcs (2.3/2.4/2.5/2.6/2.7/2.8) were reported CANNOT SHIP in
Bench A/B/C composition results, but a deep trace revealed the
substrate they consume was **structurally dead** — no consumer
actually fired because the upstream signals were broken. The arcs
were never given a fair test.

Root causes (in order of impact):

1. **`saturated_capture` was sourced from `_saturation_level` which is
   far-end-dominated, not mic clipping.** AEC3 `saturated_capture`
   semantics = mic ADC clipping; we were passing True almost every
   frame on FS-loud cases → `strong_not_saturated_render_blocks`
   stuck at 1 → `InitialState` never transitioned → 4-gate
   `linear_filter_usable` always False → ResidualEcho/SuppressionGain
   couldn't use the linear estimate path.
2. **`any_filter_converged` was only `_filter_converged` (PBFDKF's
   sticky flag) which is False on FS_static the whole time.**
   `AecState.convergence_seen` sticky bit never trips → gate (c)
   `delay_or_converge` fails.
3. **`external_delay` always passed as `None`.** Combined with (2),
   gate (c) has no path to unlock.
4. **`any_filter_consistent` duplicated `_filter_converged`.**
   TransparentMode logic reads garbage.
5. **`dominant_nearend` hardcoded `False` in ResidualEstimator call.**
   AEC3 SG and Residual share dominant_ne; ours decoupled.
6. **EchoAudibility.update() never called.** Tier A.3 module landed
   in tree but no per-frame integration into AecState's 15-step order.
7. **DownsampledRenderBuffer.read pointer never advanced.** Stuck at
   0 → matched_filter reads buf[39] (always) instead of the latest
   write position.
8. **DownsampledRenderBuffer size 2048 < max lag span 5960.** Filters
   reading far-lag positions wrap into stale data.
9. **matched_filter inner k-loop walked BACKWARD** (`xi = x_start - k`)
   vs AEC3 FORWARD (`xi = x_start + k`). Filter learned the wrong
   correlation direction.
10. **matched_filter accumulated_error EMA had no energy gate.** AEC3
    requires `error_sum_anchor > kEnergyThreshold` before pre-echo
    accumulator update.
11. **Multiple AEC3-derived constants were not rescaled for our
    pipeline cadence (100 fps vs AEC3 250 fps) and FFT size (512 vs
    128).**

## Constant rescales (100 fps + FFT=512)

| File | Constant | Before | After | Reason |
|---|---|---:|---:|---|
| matched_filter.py | window_size_sub_blocks default | 31 | 32 | AEC3 spec (fps-invariant) |
| erle.py | _AEC3_BLOCKS_TO_HOLD_ERLE | 100 | 40 | AEC3 100@250fps=0.4s → 40@100fps |
| erle.py | _AEC3_BLOCKS_FOR_ONSET_DETECTION | 250 | 100 | AEC3 250@250fps=1.0s → 100@100fps |
| erl.py | _HOLD_LENGTH | 1000 | 400 | AEC3 1000@250fps=4s → 400@100fps |
| residual_estimator_aec3 | _K_DEFAULT_NOISE_FLOOR_HOLD | 50 | 20 | AEC3 50@250fps=0.2s → 20@100fps |
| suppression_gain_aec3 | _K_LOW_BAND_GAIN_LIMIT | 64 | 128 | AEC3 kFftLengthBy2/2 = 32 @ FFT 128 → 128 @ FFT 512 (4kHz boundary) |
| suppression_gain_aec3 | _K_LOW_NOISE_THRESHOLD | 160000 | 640000 | AEC3 50²×kFftLengthBy2 (=64) → 50²×256 |
| suppression_gain_aec3 | _K_FFT_BY2 | 128 | 256 | AEC3 FFT 128/2 → FFT 512/2 (Nyquist mirror) |
| suppression_gain_aec3 | _K_LF_BAND_BOUND | 3 | 12 | AEC3 bin 3 ≈ 375 Hz → bin 12 |
| suppression_gain_aec3 | _K_MF_BAND_BOUND | 7 | 28 | AEC3 bin 7 ≈ 875 Hz → bin 28 |
| suppression_gain_aec3 | TuningConfig.last_lf_band default | 5 | 20 | AEC3 65-bin space → 257-bin (4×) |
| suppression_gain_aec3 | TuningConfig.first_hf_band default | 24 | 96 | AEC3 65-bin → 257-bin (4×) |
| suppression_gain_aec3 | _upper_bands_gain bin range | 1:16 | 1:64 | AEC3 1:kLowBandGainLimit/2 = 1:16 @ 65-bin → 1:64 |
| suppression_gain_aec3 | DominantNE hangover | (none) | 20 blocks | AEC3 50@250fps=0.2s → 20@100fps |
| suppression_gain_aec3 | _upper_bands_gain enr/max_gain | 4.0 / 0.5 hardcoded | ctor args | AEC3 config-driven |

## Wiring fixes

| File / line | Before | After |
|---|---|---|
| orchestrator.py | `saturated_capture = bool(_saturation_level > 0.5)` | `bool(np.max(\|near_end\|) > 20000)` (AEC3 ADC clip threshold) |
| orchestrator.py | `any_filter_converged = bool(_filter_converged)` (sticky PBFDKF, ~0% on FS_static) | OR with per-frame `_e_sum < 0.5 * _y_sum` (AEC3 SubtractorOutputAnalyzer proxy) |
| orchestrator.py | `external_delay = None` | from `RenderDelayController.delay` (stashed when emitted) |
| orchestrator.py | `any_filter_consistent = bool(_filter_converged)` (duplicate) | `bool(_filter_converged)` standalone (separate from converged) |
| orchestrator.py | `dominant_nearend = False` hardcoded | from `res._aec3_suppression_gain.nearend_detector.is_nearend` (lagged 1 frame) |
| orchestrator.py | (no EchoAudibility update) | `EchoAudibility.update(render_spectrum, external_delay_seen, render_block_max_abs)` per frame |
| orchestrator.py | `DownsampledRenderBuffer.read = 0` (init only) | `read = _w` per frame after write (matched_filter reads latest position) |
| orchestrator.py | buffer size 2048 | `next_pow2(max_lag_span + filter_size + sub_block)` = 8192 typical |
| matched_filter.py | inner k-loop `xi = (x_idx_init - base_k)` | `xi = (x_idx_init + base_k)` (AEC3 forward k-walk) |
| matched_filter.py | accumulated_error update unconditional | gated on `error_sum_anchor > _K_ENERGY_THRESHOLD` |
| orchestrator.py | mov_rate uses `matched_error_ratio > 1.5` unconditionally | falls back to legacy EPC when `consistent_estimate_counter ≤ 5` |
| suppression_gain_aec3 | divide-by-zero in `emr_floor` path | guarded with `emr_safe = max(emr, 1e-10)` |

## Substrate trace verification (FS_static case 0KjzXA3g20q...)

| Signal | Before fixes | After fixes |
|---|---:|---:|
| `filter_update_blocks_since_start` (counter) | 1 | 2176 ✓ |
| `linear_filter_usable` | 0.0% | **98.2%** |
| `usable_linear_estimate` | 0.0% | **98.2%** |
| `convergence_seen` (sticky) | False forever | True from frame ~25 ✓ |
| `InitialState` transitioned | False | True ✓ |
| `strong_not_saturated_render_blocks` | 1 | 2176 ✓ |
| EchoAudibility block stationary | 0% (never called) | 40% ✓ |
| matched_filter `reported_lag_estimate` count | 0 | 1374 / 2176 (63%) |
| `consistent_estimate_counter` max | 0 | 0 (still 0 — aggregator never converges) |
| `delay_quality` kRefined | 0 | 0 (not yet emitting) |

**Phase 2.5/2.6/2.7 still won't fire** because matched_filter aggregator
doesn't accumulate 10+ matching votes in any single histogram bin
(winners are spread across filters 0-18). Root cause TBD — possible
NLMS step too aggressive, possible alignment offset between capture
ordering and matched_filter's expected order. Secondary issue;
parked for next iteration.

**Phase 2.3/2.4/2.8 should now fire reliably** via AecState's 98%
`linear_filter_usable` and `usable_linear_estimate`.

## Bench D results (PENDING — 800-case j4 running)

[TBD]

## Phase 3/4 cleanup TODO — auto-derive constants from pipeline settings

All AEC3 substrate constants currently hardcoded for our setup
(hop=160, frame=320, FFT=512, ds_factor=4 → 100 fps + 257 bins).
If sample_rate / hop_size / FFT change, these need re-tuning.

For Phase 3/4 (Python cleanup), refactor to **auto-derive from
`AecConfig.{sample_rate, hop_size, frame_size, fft_size}`**:

| Constant | Current value | Auto-derive from |
|---|---:|---|
| matched_filter `window_size_sub_blocks` | 13 | `round(130ms / sub_block_time)` |
| matched_filter `alignment_shift_sub_blocks` | 2 | `round(20ms / sub_block_time)` |
| matched_filter `num_matched_filters` | 5 | preset-driven (matches AEC3 default) |
| matched_filter `matching_filter_threshold` | 0.8 | function of `filter_size` (smaller filter → looser thr) |
| delay_aec3 `_K_HISTOGRAM_ROLLING_WINDOW` | 100 | `int(1.0 * fps)` (1 sec window) |
| delay_aec3 `Thresholds.initial / converged` | 4 / 6 | `int(0.04 * fps) / int(0.06 * fps)` |
| delay_aec3 `_K_MATCHED_FILTER_WINDOW_SIZE_SUB_BLOCKS` | 13 | mirror matched_filter window_size |
| erle.py `_AEC3_BLOCKS_TO_HOLD_ERLE` | 40 | `int(0.4 * fps)` |
| erle.py `_AEC3_BLOCKS_FOR_ONSET_DETECTION` | 100 | `int(1.0 * fps)` |
| erl.py `_HOLD_LENGTH` | 400 | `int(4.0 * fps)` |
| residual_estimator_aec3 `_K_DEFAULT_NOISE_FLOOR_HOLD` | 20 | `int(0.2 * fps)` |
| suppression_gain_aec3 `_K_LOW_BAND_GAIN_LIMIT` | 128 | `int(fft_size / 4)` (= 4kHz boundary) |
| suppression_gain_aec3 `_K_LOW_NOISE_THRESHOLD` | 640000 | `50 ** 2 * (fft_size / 2)` |
| suppression_gain_aec3 `_K_FFT_BY2` | 256 | `fft_size // 2` |
| suppression_gain_aec3 `_K_LF_BAND_BOUND` | 12 | `round(375 / (sample_rate / fft_size))` (= 375Hz bin) |
| suppression_gain_aec3 `_K_MF_BAND_BOUND` | 28 | `round(875 / (sample_rate / fft_size))` (= 875Hz bin) |
| suppression_gain_aec3 `TuningConfig.last_lf_band` | 20 | scale from AEC3 65-bin → our fft_size//2 + 1 |
| suppression_gain_aec3 `TuningConfig.first_hf_band` | 96 | scale from AEC3 65-bin → our fft_size//2 + 1 |
| suppression_gain_aec3 `_K_FFT_BY2_PLUS_1` | 257 | `fft_size // 2 + 1` |
| suppression_gain_aec3 DominantNE hold_duration | 20 | `int(0.2 * fps)` |
| echo_audibility `_K_HANGOVER_BLOCKS` | 5 | `int(fps / 20)` |
| residual_estimator_aec3 `_K_DEFAULT_FFT_BINS` | 257 | `fft_size // 2 + 1` |
| erle.py `_AEC3_K_FFT_BY2_PLUS_1` | 257 | `fft_size // 2 + 1` |
| reverb.py `_K_FFT_BY2_PLUS_1` | 129 | depends on reverb-side FFT |
| orchestrator `_mf_max_lag` | derived | already auto from `_mf.window_size_sub_blocks` etc. ✓ |

**Open questions** (deferred):
- Phase F sentinel for first-time delay (avoid treating first valid
  delay as "change").
- Matched_filter still leaves filters 3+4 idle — could downsize N=3
  if want to save more CPU.
