# AEC Algorithm Methods

This document is the deep algorithm specification for the production pipeline:
a PBFDKF linear stage followed by an AEC3-aligned post-filter chain
(`AecState` + `ResidualEchoEstimator` + `SuppressionGain` + CNG). The Python
reference (fp64) is the **algorithm spec**; the C port is the float32
**implementation** and agrees with Python to a tolerance (§7). The C structure
mirrors Python class boundaries.

| Companion doc | Purpose |
|---|---|
| [`c_user_manual_zh_TW.md`](c_user_manual_zh_TW.md) | C API, integration, streaming contract |
| [`nn_integration_interface.md`](nn_integration_interface.md) | NN residual/NR/joint swap seams |
| [`../CHANGELOG.md`](../CHANGELOG.md) | Version history |

---

## 1. Pipeline overview

```
mic ─► HPF ────────────────────────────────────────────────────────────────────────►
ref ──────► Saturation ─► EchoPathDelayEstimator ─► reference ring ─────┐
                          (matched filter +                             │
                           lag aggregator +                             ▼
                           clock-drift detector)                   PBFDKF refined
                                                                   filter (Kalman)
                                                                          │
                                                                  Shadow filter (PBFDAF/NLMS,
                                                                  μ = 0.5) + PathChange-
                                                                  RegimeHandler (load-bearing
                                                                  on cohort tail)
                                                                  + RenderSignalAnalyzer
                                                                          │
                                                                       error (e[k])
                                                                          │
                                                                          ▼
                                                                   AEC3 _aec3_post
                                                                   ├─ StationarityEstimator
                                                                   ├─ AecState (read-only ADT
                                                                   │     over its sub-analyzers)
                                                                   ├─ ResidualEchoEstimator
                                                                   │     (per-bin echo PSD)
                                                                   ├─ SuppressionGain
                                                                   │     (Wiener + over-est)
                                                                   ├─ Comfort noise (per-bin
                                                                   │     PSD EMA)
                                                                   └─ sqrt-Hann synthesis OLA
                                                                          │
                                                                          ▼
                                                                         out
```

### Presets — one production point + three operating points

`AecPreset.BALANCED` is the production preset. It shares a narrow tuned base
with the other presets:

```python
AecConfig.from_preset(AecPreset.BALANCED)  # base, equivalent to:
AecConfig(
    enable_cng=True,        # comfort-noise generator on
    shadow_mu_min=0.5,      # shadow / DT mu floor
    warmup_frames=100,      # forced-high-mu startup window
    kalman_q_high=1e-3,     # PBFDKF Q_high convergence speed
)
```

`mild` and `aggressive` are deliberate **Pareto operating points** on the same
strength axis as `balanced`. `mild`/`balanced` differ in exactly one field —
`min_gain_floor_far_active_db`, the far-active min-gain floor. `aggressive`
additionally deepens `min_gain_floor_dt_db` (the floor used in its place
during double-talk, see §3.4): without that second field, `dt_protect_active`
hard-overrides the far-active floor with the SAME shared −20 dB for every
preset whenever it fires (it also false-arms in far-end single talk), so the
presets would be indistinguishable on exactly the hops where a user would
expect to hear the strength difference. `no_floor` is a fourth tier that
removes the floor mechanism (`min_gain_split_floor_enabled=False`) entirely
instead of deepening it further — the deepest point on the axis, with the most
near-end cost.

| preset | far-active floor | dt floor | character |
|---|---|---|---|
| `mild` | −20 dB | −20 dB | near-priority — more near kept, more echo leak |
| `balanced` | −28 dB | −20 dB | production |
| `aggressive` | −38 dB | −30 dB | echo-priority — dev-pool (47-case) DT_static echo +0.17/+0.14 MOS vs the shared-floor point, at a bounded near-end cost |
| `no_floor` | n/a (floor off) | n/a (floor off) | deepest point on the axis — purely audibility/R²-derived gain, no fixed-dB clamp at all; most near-end loss, and it exposes the cold-start R² behaviour the floor otherwise masks |

The DT-deg-vs-echo trade is a single-channel DSP Pareto wall; the strength
axis exposes it.

Everything else uses `AecConfig` dataclass defaults; the AEC3 chain has
its own tuning embedded in `AecStateConfig` / `SuppressionGainConfig`
(`modules/state/aec_state.py`, `modules/residual/suppression_gain.py`).

### Determinism

Comfort noise uses an internal LCG (`seed = seed·69069 + 1`) seeded with 42 at
construction and at reset, in both Python and C; the algorithm consumes no
global RNG state, so a render is reproducible per instance.

### Tunable parameters (`AecConfig`)

`AecConfig` (`modules/config.py`) is the full release surface; the table lists
its principal fields. mild/balanced differ **only** in
`min_gain_floor_far_active_db` (−20 / −28); aggressive additionally sets
`min_gain_floor_dt_db` to −30 (shared −20 default otherwise); `no_floor`
instead sets `min_gain_split_floor_enabled=False`. `from_preset` additionally
pins a shared base (`enable_cng=True`, `shadow_mu_min=0.5`, `warmup_frames=100`,
`kalman_q_high=1e-3`). Everything below uses dataclass defaults. Fields are
grouped by subsystem; **ON**/**OFF** marks the default state of a feature flag.
Frame-count fields marked as a 10 ms authored basis are converted at
construction to preserve wall-clock duration on the selected grid; the table
shows their public configuration values, not necessarily the internal hop count.

| Subsystem | Field | Default | Notes |
|---|---|---|---|
| System / framing | `sample_rate` | 16000 | 8000/16000/48000 |
| | `frame_size` / `hop_size` | auto | frame=FFT、hop=frame/2；8k 256/128、16k 256/128（可選512/256）、48k 1024/512 |
| | `filter_length` | auto | 52 ms (<44.1 kHz), 64 ms (≥44.1 kHz) |
| | `mu` / `delta` | 0.3 / 1e-8 | step size / regularization |
| Residual / CNG | `enable_res` | **ON** | post-filter master gate (§3.7) |
| | `enable_cng` | OFF (preset **ON**) | comfort noise (§3.5) |
| | `cng_n2_windowed_psd` | 0.0 (**OFF**) | feeds the N2 tracker a windowed capture PSD instead of a rectangular one |
| | `comfort_noise_floor_dbfs` | −96.034 | |
| | `enable_td_constraint` | **ON** | |
| C′ ERLE coherence gate | `erle_coh_gate_enabled` | **ON** | Γ²(Ŷ,Y) per-bin ERLE freeze |
| | `erle_coh_gate_threshold` / `_alpha` | 0.5 / 0.05 | |
| Split min-gain floor (§3.4) | `min_gain_split_floor_enabled` | **ON** | OFF for the `no_floor` preset |
| | `min_gain_floor_far_active_db` | −28 | **preset strength knob** (mild −20 / balanced −28 / aggressive −38) |
| | `min_gain_floor_far_silent_db` | −12 | pure-NE floor |
| | `min_gain_far_latch_power` | 1e6 | far-active latch |
| **DT-aware recovery group** | `dt_aware_recovery_soft` | **ON** | soft (non-destructive) delay/EPC recovery (§2.3) |
| | `dt_aware_res_floor_enabled` | **ON** | RES floor lift (§3.4); default source is the near-recent latch alone |
| | `min_gain_floor_dt_db` | −20 | floor used in double-talk; **−30 for aggressive**, shared −20 for mild/balanced |
| | `ne_recent_threshold` | 0.3 | near-recent gate arm threshold |
| | `ne_recent_hold` | 150 | 10 ms authored basis; approximately 1.5 s after grid retiming |
| | `ne_recent_sustain` | 3 | frames above threshold to arm |
| Soft near-end blend | `soft_nearend_blend_enabled` | **ON** | sigmoid ENR tuning blend |
| | `soft_nearend_blend_enr_threshold` / `_softness` | 0.25 / 0.25 | |
| | `soft_nearend_blend_per_bin` | **ON** | per-bin frequency-selective near protection |
| Nonlinear residual | `nl_r2_enabled` | **ON** | Kuech-Kellermann 2nd-order R² (L1) |
| | `nl_r2_alpha` | 0.1 | |
| | `nl_r2_cap_multiple` | 300 | caps r2_nl at 300x the direct term; the raw formula is unbounded (fixed -20dBFS reference, not signal-level-relative) and reaches 1e16-1e18 on loud cold-start render |
| ERLE coordinate | `erle_render_x2_psd_scale` | **ON** | render PSD scale feeding the reverb model |
| | `reverb_tail_strength` | 1.0 | reverb-tail R² scale |
| | `erle_windowed_capture_psd` | **ON** | Y2/E2 sqrt-Hann coordinate consistency |
| | `output_capture_when_linear_unusable` | **ON** | two mutually exclusive roles: E2 output base = Y on diverged frames under `enable_res`; capture admitted as the FORM step's third candidate in context-only (§3) |
| Delay-acquire guard | `delay_acquire_protect_converged` | **ON** | reject spurious late FIRST acquisition (Path A only) |
| Cold-start deadlock (OFF) | `h_error_refresh_erl_floor` | 0.0 (**OFF**) | Kalman-gain cold-start breakers |
| | `h_error_floor_override` | 0.0 (**OFF**) | |
| Shadow filter | `enable_shadow` | **ON** | PBFDAF/NLMS coarse filter |
| | `shadow_mu_min` / `shadow_mu_nlms` | 0.5 / 0.5 | |
| | `shadow_copy_threshold` / `_hysteresis` | 0.65 / 3 | |
| | `shadow_err_alpha` | 0.80 | |
| | `shadow_dtd_advantage_scale` / `shadow_dtd_offset` | 3.0 / 1.5 | |
| Misadjustment estimator | `filter_misadjustment_*` | 100 / 30 / 0.5 / 2.0 | hangover/stable use 10 ms authored basis; scale min/max are dimensionless |
| PBFDKF (Kalman) | `use_kalman` | **ON** | |
| | `kalman_q_high` / `kalman_q_low` | 1e-3 / 1e-6 | |
| | `warmup_frames` | 80 (preset 100) | 10 ms authored basis; preset remains approximately 1.0 s after grid retiming |
| EPC | `epc_delta_threshold` / `epc_total_rise` / `epc_hangover` | 0.3 / 1.5 / 20 | |
| Delay estimation | `enable_delay_est` | **ON** | deprecated mirror of `delay_mode` |
| | `max_delay_ms` / `delay_buffer_ms` | 1024 / 2048 | |
| | `delay_est_period_s` / `delay_est_init_s` | 0.5 / 0.3 | |
| | `delay_par_low_threshold` / `_solid_threshold` | 5.0 / 8.0 | |
| HPF | `enable_highpass` | **ON** | mic-path only |
| | `highpass_cutoff_hz` | 80.0 | |
| Saturation | `enable_saturation_detect` | **ON** | |
| | `saturation_threshold` | 0.95 | |
| | `saturation_softclip_ref` | **ON** | |
| Mode / output | `mode` | `PBFDKF` | |
| | `return_res_context` | OFF | switches `process()` return type (§4); with `enable_res=False` it selects the context-only seam, whose FORM step admits the capture candidate (§3) |
| | `clear_filter_history` | OFF | |

The **DT-aware recovery group** has two halves:
`dt_aware_recovery_soft` softens the delay/EPC recovery (§2.3), gated on the
`ne_recent_*` near-recent latch, and `dt_aware_res_floor_enabled` lifts the RES
floor (§3.4), gated on the same `ne_recent_*` latch. The exact per-hop floor
decision is exported to fused consumers as `res_floor_protect`.

---

## 2. Front-end blocks

### 2.1 HPF (high-pass filter)

2nd-order Butterworth high-pass, 80 Hz, on the mic path only:

* Mic-path HPF: **ON** (suppresses DC + sub-bass before delay alignment).
* Reference HPF: **OFF / bypassed** — the pipeline feeds the raw reference
  straight into `Saturation` → `EchoPathDelayEstimator`.

`HighPassFilter` lives in `modules/preprocessing.py`; the C port uses the shared
`audiolib` HPF (`hpf_init()`).

### 2.2 Saturation detector + soft-clip

`SaturationDetector` (preprocessing.py) tracks the fraction of clipped
samples (|x| > `saturation_threshold`, 0.95) and exposes a smoothed
`saturation_level` ∈ [0, 1] (fast attack, slow release). The orchestrator combines the reference and
mic levels; above 0.5 the capture counts as saturated (`AecState.update_capture_saturation`) and
the filter step sizes are gated down.

`saturation_softclip_ref=True` (default) soft-clips the reference
prior to delay alignment so the refined filter sees a tractable
nonlinear envelope.

### 2.3 Delay estimator (AEC3-aligned)

`modules/delay/`:

* `EchoPathDelayEstimator` (matched filter + downsampled cross-correlation).
* `MatchedFilter` (running matched-filter accumulator).
* `MatchedFilterLagAggregator` (decision over the matched-filter histogram).
* `ClockdriftDetector` (drift over a 30 s window; AEC3 reference is
  7500 blocks ≈ 30 s).
* `DownsampledRenderBuffer` (delay-aligned render history).
* `RenderDelayController` (orchestrator-facing alignment + warmup).

`legacy_compat.LegacyDelayShim` exposes the `DelayEstimator.accumulate()`
API on top of the estimator; the orchestrator imports it via the public
alias so external callers (`from aec import DelayEstimator`) work.

The reference ring holds `max_delay_ms` (1024 ms by default) of skew; the
matched filter's own reliable reach is about 509 ms (see
[`delay_estimator_design_zh_TW.md`](delay_estimator_design_zh_TW.md)). The
estimate feeds the refined filter through a delay-line on the reference.

#### Pre-echo detection (matched-filter accumulated-error)

`MatchedFilter` (`modules/delay/matched_filter.py`) is a bank of `num_filters=5`
NLMS cross-correlators over staggered lag windows (128 ms window, 96 ms
alignment shift). The winning lag is
`max_square_peak_index(h) + n·intra_lag_shift` — the strongest squared tap. AEC3
additionally tracks a **pre-echo lag** so *its* reported delay snaps to the
echo *onset* rather than the strongest tap (which, with reverb, can sit well
after onset). `detect_pre_echo` defaults to True in `EchoPathDelayEstimator`
and the pre-echo lag aggregator runs, but this port **does not report its
candidate**.

⚠ **Deliberate divergence at the selection seam.** Upstream substitutes the
pre-echo candidate for the reported delay while still qualifying it with the
*dominant* peak's histogram — sound there, because its estimator feeds a
`RenderDelayController` that wants the earliest onset. Here the reported sample
delay aligns a short PBFDKF directly, so an onset further ahead of the dominant
echo than that filter spans puts the echo it must cancel outside the filter,
while confidence still reads 1.0 because confidence describes the dominant
histogram and not the substituted delay. The product path therefore reports the
dominant peak; the pre-echo histogram is still computed as the upstream
reference and as a diagnostic. Pinned by `test_delay_dominant_selection`
(Python and C).

The pre-echo locator (`_compute_pre_echo_lag`, after AEC3 `ComputePreEchoLag`)
walks the per-tap-group **accumulated squared error of the filter PREFIX** and
returns the earliest group whose prefix already explains the capture (= onset).
The curve is built in `matched_filter_core` from the cumulative sum of
`h · x_window` over tap groups of `_ACCUMULATED_ERROR_SUBSAMPLE_RATE` (4); the
squared prefix error of each group is accumulated per sub-block sample.

#### DT-aware soft recovery (`dt_aware_recovery_soft`, default ON)

The delay/EPC recovery triggers (Path A first-acquisition, Path B re-lock, EPV,
shadow_rise) normally do a **destructive** full reset — clear the filter, set
Kalman P=1.0, `mark_diverged` — and re-converge from cold. That reset is the
right move at a true far-only path change, but its aggressive re-convergence
tail overfits near-end speech that arrives shortly after, and the online
(no-pre-align) pipeline pays this penalty that offline GCC pre-align never does.
When `dt_aware_recovery_soft=True` and near-end was seen recently
(`_ne_recent_frames > 0`, the held gate described in §3.4), those triggers drop
the full reset and instead **realign softly** — keep the converged filter,
re-adapt at normal step size. Far-end single-talk never arms the gate, so its
recoveries stay aggressive and FS echo depth is preserved. This is the partner
mechanism to the DT-gated RES floor (§3.4).

### 2.4 PBFDKF refined filter

`PBFDKF` (Partitioned Block Frequency Domain Kalman Filter,
`modules/filters.py`) is the production filter. It is a `PBFDAF` with a per-bin
Kalman gain: the overlap-save buffer is 2 × hop, the FFT size is the next power
of two of that (256 on the 16 kHz default grid), and
`n_partitions = ceil(filter_length / hop)`. Kalman state per partition: filter
coefficients `W[p, k]`, error covariance `P[k]`, process noise `Q[k]`.

Convergence schedule: `Q` runs at `kalman_q_high` (1e-3) on start and after an
EPC / delay change, and settles to `kalman_q_low` (1e-6) in steady state;
`warmup_frames` (preset 100) forces the high-step startup window.

The refined filter consumes the `RenderSignalAnalyzer` per-bin mask
(narrow-band tonal regions freeze the W update on those bins) and the
`poor_signal_excitation` flag (insufficient reference excitation also
freezes W).

`FilterStateBridge` (`modules/filter/filter_state_bridge.py`) is the
read-only seam that exposes refined-filter spectra / convergence state
to the AEC3 post-filter chain.

### 2.5 Shadow filter (coarse adapter)

The shadow filter is NLMS (`PBFDAF`) at μ = 0.5, the AEC3 coarse-filter role.
It runs in parallel with the refined filter; its job is to be
the fast-tracker that survives echo path changes, while the refined
filter holds the converged tuning.

`PathChangeRegimeHandler` (`modules/epc.py`) arbitrates between the two
filters:

* Fires `boost_q` / `reverse_copy` / `main_paused` decisions per frame
  based on relative error EMAs, render activity, and the shadow-DT
  signals from `DoubleTalkAnalyzer` (energy-gated).
* Load-bearing on the cohort tail: without it the main filter can diverge to
  anti-correlated output on a nonstationary echo path.
* Must not be removed or bypassed.

### 2.6 Render signal analyzer

`RenderSignalAnalyzer` (`modules/render/render_signal_analyzer.py`)
mirrors AEC3's `render_signal_analyzer.cc`. Two outputs:

1. Per-bin freeze mask for refined-filter gain compute (sustained narrow-band
   peaks zero mu around them).
2. `poor_signal_excitation` flag that zeroes the refined-filter mu when
   a narrow-band counter shows the reference is too tonal / too quiet to excite
   the filter.

---

## 3. AEC3 post-filter (`_aec3_post`)

After the refined filter produces the linear residual error spectrum
`E(k)`, `AEC._aec3_post()` (in `modules/orchestrator.py`) runs the
AEC3-aligned suppression chain.

Its first step is the **FORM** stage, `_aec3_select_linear_filter_output`,
mirrored by `linear_filter_select()` in `c_impl/src/linear_filter_output.c`.
It picks one of **three** time-domain candidates — coarse (shadow), refined,
and the capture itself (`_SEL_COARSE` / `_SEL_REFINED` / `_SEL_CAPTURE`;
`LFS_SEL_*` in C) — carries any change across the 30-sample AEC3
`SignalTransition` crossfade, and publishes the formed hop and its matching
WOLA spectrum as a pair, so `E(k)` entering the rest of the chain is the
FORMED output, not necessarily what the refined filter produced.

The capture candidate is admitted only when all four hold:
`output_capture_when_linear_unusable` is set (default ON), the
FilteringQualityAnalyzer has not declared the linear estimate usable (the
verdict as of the PREVIOUS hop — AecState updates later in `_aec3_post`), the
selected residual carries more hop energy than the capture, and the instance
is in **context-only** mode (`enable_res=False` with
`return_res_context=True`). Under `enable_res` the chain's own E2 output-base
switch still owns that rule and the emitted audio is unchanged; running both
would let E2's frequency-domain hard switch overwrite the crossfade FORM just
started. Because the 30-sample transition stays active, the capture candidate
is a steady-state fallback, not a strict per-sample bound.

The chain itself:

### 3.1 StationarityEstimator

`modules/state/stationarity_estimator.py` mirrors
`stationarity_estimator.cc`. Per-frame per-bin EMA of render PSD plus a
running coefficient-of-variation gate. Two consumers:

* `_aec3_post` zeros R² (residual echo PSD estimate) on stationary bands, so a
  steady far-end hum the filter has nothing to learn from does not drive the
  suppressor against a near end speaking over it. The zeroing fires whenever
  the filter has converged and the stationarity mask does, regardless of the
  usable-linear verdict.
* Refined filter skips W update on block-stationary frames, preventing
  the PBFDKF from learning mic-as-echo coupling against an uncorrelated
  stationary input.

### 3.2 AecState

`modules/state/aec_state.py` is the AEC3-aligned read-only ADT and the single
source of `usable_linear_estimate()`. It wraps the analyzers under
`modules/state/`:

```
AecState
├─ InitialState          (startup gating)
├─ FilterDelay           (filter-peak delay)
├─ FilterAnalyzer        (impulse-response shape consistency)
├─ FilteringQualityAnalyzer (multi-gate usable_linear_estimate)
├─ ErlEstimator          (ERL adaptive EMA)
├─ ErleEstimator         (subband + fullband ERLE)
│    ├─ SubbandErleEstimator
│    └─ FullBandErleEstimator
└─ SaturationDetector    (capture / render saturation flags)
```

### 3.3 ResidualEchoEstimator

`modules/residual/residual_echo_estimator.py` produces the per-bin
residual-echo PSD estimate R²(k). Two paths:

* **Linear**: `R² = ERLE_subband × E²(k)` — the AEC3-aligned ERLE-divide.
  Active when AecState reports `usable_linear_estimate`.
* **Render-based**: `R² = ERL_subband × X²(k)` — uses the long-window
  far-PSD EMA when the linear path is unreliable (epc_active, saturation,
  pre-convergence).

A reverb-tail term from `ReverbModel` (`modules/residual/reverb_model.py`)
is added when reverb estimation is online; tracks per-bin reverb energy
across the impulse response tail using `ReverbDecayEstimator` +
`ReverbFrequencyResponse`.

#### Live observer (default OFF)

`live_observer_enabled` (default OFF) adds a third, independent PBFDAF
instance (C: `aec_post_run_observer_step()` in `aec_post.c`, called from the
post chain rather than the main hop loop because it needs the current hop's
`dominant_ne` / `usable_linear_estimate` verdicts). It reads the far-end ring
`live_observer_shift_ms` earlier than the main filter's own
delay-compensated read, so its echo estimate covers the span just ahead
of the main filter's current window. Its evidence is injected as
`R² = max(R², min(|observer.echo - sel_echo|², |sel_esw|²) × PSD_SCALE)`,
bounded to `live_observer_max_hz` and gated on a corroboration check: the
observer's own residual-error EMA must be no worse than the EMA of the
*selected* linear output's residual over a `live_observer_corrob_ms` time
constant, within `live_observer_corrob_margin_db`; injection is also limited by
`live_observer_inject_dt_energy_max`. It never touches the linear filter, the
delay estimate, or ERLE — only R² before SuppressionGain.

### 3.4 SuppressionGain

`modules/residual/suppression_gain.py` mirrors AEC3
`suppression_gain.cc`. Wiener-style gain on the residual model:

```
G(k) = (S²(k) - α R²(k)) / S²(k)
S²(k) = mic PSD when usable_linear_estimate
      = nearend_excess PSD otherwise
```

with α a frame-state-dependent over-estimation factor. The gain is then
band-limited (HF cap), temporally smoothed (asymmetric attack / release),
and floored by a per-bin spectral-floor mask.

**Split min-gain floor (default ON).** The AEC3 min-gain floor
`min_gain = min_echo_power / R²` collapses to ~0 in double-talk: near-end
inflates the error → ERLE drops → R² spikes → the floor that should protect
near-end vanishes exactly when near-end is present (RES over-suppresses
near-end up to −8 dB; the linear filter only cuts −1.4..−2.9 dB). A
coherence double-talk discriminator does not work (single-lag X–Y
coherence cannot separate reverberant FS from DT). The fix is a power-domain
floor on the deepest suppression, **split by far-end activity**:

```
min_gain(k) = max( min_echo_power / R²(k),  floor )
floor = 10^(-28/10)  if far-active (FS/DT)   # balanced; deep far-active
                                             #   suppression for more echo
                                             #   cancellation
      = 10^(-12/10)  if far-silent (pure NE) # lifts NE near-end, zero echo
                                             #   cost (no echo to leak)
```

Routing uses a per-recording **latch** on instantaneous far energy
(`mean(render_block²) > 1e6`, render int16-scaled — ≈ the orchestrator's own
far-active criterion). It latches from the first far frame so FS/DT use the
gentler floor throughout (no cold-start leak); only recordings where far is
never active keep the strong floor. Config (`AecConfig`):
`min_gain_split_floor_enabled`, `min_gain_floor_far_active_db` (balanced −28;
mild −20 / aggressive −38 — the preset strength knob),
`min_gain_floor_far_silent_db` (−12), `min_gain_far_latch_power` (1e6). The
floors are amplitude-domain dB; the orchestrator constructs `SuppressionGain`
with these as `split_floor_*`. The far-active floor above is further lifted in
double-talk by the DT-aware floor gating below.

**Double-talk floor protection (DT-aware floor gating, default ON).**
The far-active floor (balanced −28 dB) is deliberately aggressive so pure
far-end single-talk cancels deep echo — but applying that same deep floor
*during double-talk* over-suppresses near-end. The DT-gated floor lifts the
floor from `min_gain_floor_far_active_db` to `min_gain_floor_dt_db` *only when
near-end is present*, selectively protecting near-end in DT while keeping the
aggressive far-active floor in pure FS:

```python
# suppression_gain.py (_get_min_gain):
if self._far_active_latched:
    base_floor = (self._split_floor_dt if self._dt_protect_active   # DT
                  else self._split_floor_far_active)                # FS
else:
    base_floor = self._split_floor_far_silent                       # pure NE: −12 dB
```

The `-20 dB` dt-floor default (mild/balanced) is a Pareto choice, not a free
detector improvement: against `-16 dB` it buys far-end-single-talk ERLE and echo
MOS at a small double-talk degradation cost (numbers in the CHANGELOG), and it
does not make the near/far latch more accurate. **aggressive deepens this floor
to −30 dB** instead — without a distinct value here, `dt_protect_active` (which
also false-arms on pure far-end-singletalk hops) hard-overrides the
far-active floor with the SAME −20 dB for every preset, masking most of the
strength-axis difference exactly when a user would expect to hear it.

The orchestrator sets `sg._dt_protect_active` immediately before `get_gain`:

```python
self._aec3_sg._dt_protect_active = bool(
    self.config.dt_aware_res_floor_enabled        # default True
    and _ne_recent)
```

The floor is armed by the energy-based `_ne_recent_frames` latch alone — the
same gate `dt_aware_recovery_soft` uses. Config (`AecConfig`):
`dt_aware_res_floor_enabled`, `min_gain_floor_dt_db` (−20; −30 for aggressive),
`ne_recent_threshold`/`ne_recent_hold`/`ne_recent_sustain`.

### 3.5 Comfort noise (CNG)

When `enable_cng = True` (preset default), `_aec3_post` adds per-bin shaped
noise scaled by `sqrt(1 - G²)` to mask the suppressed spectrum. State:
per-bin noise PSD EMA + smoothed cn gain (init in `AEC.__init__`).

### 3.6 OLA synthesis

sqrt-Hann analysis × sqrt-Hann synthesis = Hann, sums to 1 across 50 %
overlap hops. `_aec3_synth_window` + `_aec3_ola_buf` accumulate the
windowed inverse FFT into the output stream.

### 3.7 enable_res = False bypass

With `enable_res=False` the `_aec3_post` body is skipped and the linear residual
is emitted straight from the refined filter; `eval_aec_challenge.py` writes
that render as `_ours_nores.wav` next to the production render. The gate at
`_aec3_post` block-open uses
`self.config.enable_res or self.config.return_res_context`.

---

## 4. Diagnostic surfaces (load-bearing — do not remove)

### `AecStats` / `get_stats()`
Per-frame audio-passive trace (`modules/dataclasses.py` + the
`AEC.get_stats` method in `modules/orchestrator.py`). Consumed by
`run_one_case.py` plots and external research tooling.

### `AecResContext`
Exposes `echo_spec` / per-bin Kalman state so the linear stage can feed
an external (or NN) post-filter. Enabled via
`AecConfig.return_res_context = True`; switches `aec.process()` return
type to `(out, AecResContext)`.

With `enable_res=False` (context-only) this is the **only** mode in which the
FORM step admits the capture candidate (§3): the context path returns the
linear residual before the E2 output-base switch, so without it the seam would
carry an over-output that the emitted audio never does. `formed_output`,
`error_spec`, `echo_spec` and `near_spec` all follow the selection as a
matched set. `AecConfig.return_formed_output` requires this flag or
`enable_res` and raises at construction otherwise — the formed seam is
produced by the post chain and cannot be requested without it.

### Regime classifier
`modules/p52_regime_classifier.py` is analysis-only; the
`AntiLoopholeTests` in `python/tests/test_p52_regime.py` enforce that the
classifier never modifies audio.

---

## 5. Tooling

* `python/aec.py` — top-level shim + CLI entry point.
* `python/eval_aec_challenge.py` — render 800-case AEC Challenge corpus.
  Standard config: `--preset balanced --filter 832 --cng --parallel --workers 4`.
* `python/bench_aecmos.py` — AECMOS scoring driver
  (needs `onnxruntime`, `librosa` and `soundfile`).
* `python/eval_manifest90.py` — fixed 90-case partial blind benchmark with
  ERLE/SDR and scenario-correct AECMOS metrics.
* `python/run_one_case.py` — single-case dev tool with diagnostic
  5-panel PNG (waveforms + spectrograms + ERLE).
* `c_impl/Makefile` release gates — `test-static-aec`,
  `test-rate-structural`, `test-process-context` and `test-avg-reverb-slots`
  cover C pool parity, supported grids, the external context seam and the
  average-render-reverb ring slots.
* `python/tests/test_p52_regime.py` — pytest for the regime classifier
  anti-loophole contract.

---

## 6. Reference scores

The production-faithful 800-case AECMOS bucket means (ours vs WebRTC AEC3 vs
Speex; `balanced / fl=832 / --cng`, no pre-alignment, official per-bucket
segmentation, 16 kHz 256/128 grid) are in the root
[README](../README.md#status-snapshot) with their measurement conditions. "No
pre-alignment" means the delay is acquired online by the in-pipeline
matched-filter self-align (§2.3), exactly as the shipped algorithm runs.
Offline GCC pre-alignment is a benchmark-only crutch; its numbers are not a
production result.

---

## 7. Python ↔ C precision model

The `c_impl/` port is the production C implementation of the **same algorithm**
as `python/aec.py`. All production C is float32 end-to-end: delay chain,
orchestrator scalars, post/state modules, `residual_echo_estimator`, and the
mic-path HPF. The relationship between the two implementations is:

* **Python (fp64) is the algorithm spec.**
* **C (float32) is the implementation.**
* **Python↔C comparison is tolerance-based**, not 0/0.

The delay-estimator matched filter is float32 fast-math and duty-cycle capable
(see §2.3 and `c_impl/include/delay_aec.h`).

The **FFT backend is NE10 (float32, ARM NEON)** (via the shared
`audiolib` layer), which is not bit-exact to numpy's fp64 `np.fft`. Because
PBFDKF and the post-filter are frequency-domain, this float32 FFT difference
propagates through the whole pipeline, so the **end-to-end C output aligns
with Python to ~float32 precision, not 0/0**. `parity_aec_e2e.c` asserts the
linear `raw_output` and final `out[hop]` stay within a **2e-2 float32
tolerance** at 16 kHz (1e-1 at 48 kHz; measured maxima are tabulated in the
file header) and is the **authoritative C regression gate**.

### Math backends

The shared `audiolib/common/utility/fast_math.h` (sibling repo, header-only)
provides two implementations selected at compile time (orthogonal to the
FFT-backend choice above):

* **Default (production): `fast_math.h` approximations.** `fast_exp` (LUT +
  Taylor), `fast_log`/`fast_log10` (IEEE-754 range reduction + degree-4
  minimax), and `fast_sqrt` (correctly-rounded ARM sqrt: AArch64 hardware
  FSQRT or ARM32 `sqrtf` through hardware/libm; Newton-Raphson is the portable
  non-ARM fallback). `fast_exp` has a
  worst relative error of **3.9e-3** (about 0.034 dB on a gain), which reaches
  the stages that call `exp` (SuppressionGain near-end sigmoid, CNG gain, dB
  conversions) on top of the float32-FFT tolerance above; the ARM
  `fast_sqrt` is correctly rounded and adds none.
* **`-DUSE_STANDARD_MATH`: libm.** Swaps the approximations for `expf`/`logf`/
  `sqrtf` (the `USE_STANDARD_MATH` block at the top of `fast_math.h`); a
  precision/perf toggle only.

`-ffp-contract=off -fno-math-errno` is mandatory in `CFLAGS` on both backends
and on both math modes — the first flag keeps builds deterministic (no fused
multiply-add reassociation), the second makes guarded AArch64 `sqrtf` lower to
`FSQRT` without an errno fallback. The DSP pipeline does not read math
`errno`. The C structure mirrors the Python class boundaries one-to-one
(`PBFDKF`, `AecState`, `SuppressionGain`, `EchoPathDelayEstimator`, …), so a
module-level regression isolates any divergence to a single stage. See
`docs/c_user_manual_zh_TW.md` (build + integration).
