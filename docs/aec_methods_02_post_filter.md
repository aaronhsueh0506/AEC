[Index](aec_methods.md) | Previous: [Part 1: Front-end blocks](aec_methods_01_front_end.md) | Next: (end) <!-- split-nav -->
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
`live_observer_inject_dt_energy_max` (Python only; the C port has no such guard, a known
divergence pinned as scenario S-02 in `c_impl/test/pyc_parity/pyc_parity.py`). It never touches the linear filter, the
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

[Index](aec_methods.md) | Previous: [Part 1: Front-end blocks](aec_methods_01_front_end.md) | Next: (end) <!-- split-nav -->
