[Algorithm reference](../aec_methods.md)

## 3. AEC3 post-filter (`_aec3_post`)

After the refined filter produces the linear residual error spectrum
`E(k)`, `AEC._aec3_post()` (in `modules/orchestrator.py`) runs the
AEC3-aligned suppression chain.

Its first step is the **FORM** stage, `_aec3_select_linear_filter_output`,
mirrored by `linear_filter_select()` in `c_impl/src/linear_filter_output.c`.
It picks one of **two** time-domain candidates — coarse (shadow) and refined
(`_SEL_COARSE` / `_SEL_REFINED`; `LFS_SEL_*` in C) — carries any change across
the 30-sample AEC3 `SignalTransition` crossfade, and publishes the formed hop
and its matching WOLA spectrum as a pair, so `E(k)` entering the rest of the
chain is the FORMED output, not necessarily what the refined filter produced.

The formed output is always the real selected linear residual, in every API
mode (full, context-only, no-gain, linear-only): the capture is never a
candidate, and AecState, ERLE, R², the error PSD and the echo PSD always see
the selected residual. `error_spec` is the STFT of
`[previous formed hop | current formed hop]`, so a unity IFFT with overlap-add
reproduces the previous formed hop one framing hop late, on every hop — hops
where the output guard fires included. These fields are the **analysis seam**:
`formed_hop` / `formed_output`, `error_spec`, `echo_spec` and
`near_spec = error_spec + echo_spec`. The output guard never alters them; it
only chooses the **rendering base** ([§3.8](#38-output-guard-rendering-base-and-post-reset-serial)).

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

The numeric core is one function in each language: C `aec_cng_fill()`
(`aec_res.c`, called by the post chain's own output stage and public for
external synthesizers) and Python `modules/cng.py` (`cng_fill`, which
`_post_add_comfort_noise` shares through `cng_draw_indices` /
`cng_noise_terms`). For bins `1 .. n-2` (DC and Nyquist get no noise and draw
nothing) it adds, per component,

```
ng * (nf * lut)      ng = sqrt(max(1 - g_res², 0)) [* max(g_nr, nr_floor) with an NR gain]
                     nf = sqrt(max(cn / 32768², 0))
                     lut[i] = sqrt(2) * sin(2π i / 32)
```

in this product order, float32 throughout. The phase is one LCG step per bin in
ascending bin order (`rng = (rng * 69069 + 1) & 0x7FFFFFFF`, `ix = rng >> 26`;
real part `lut[ix]`, imaginary part `lut[(ix + 8) & 31]`). The generator starts
at `AEC_CNG_SEED` (42) after construction and after every hard post-chain reset.
With `g_nr == NULL` the public helper is bit-identical to the in-chain noise on
the same inputs; the in-chain stage passes the total gain it applied as
`g_res`. With an NR gain the amplitude is
`sqrt(cn / 32768²) * sqrt(1 - g_res²) * max(g_nr, nr_floor)`.

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

### 3.8 Output guard, rendering base and post reset serial

**Two different things.** The analysis seam (`formed_hop`, `error_spec`,
`echo_spec`, `near_spec`, [§3](#3-aec3-post-filter-_aec3_post)) is the pairable
linear residual that the estimators, an AIAEC front end and any model that
needs a reconstructable waveform read. The **rendering base** is the spectrum
that is multiplied by the gain and synthesized into audio. They differ only on
hops where the guard fires, and the rendering base is not claimed to pair with
`formed_hop`.

**The guard.** It is evaluated exactly once per hop, in every mode (full,
context-only, no-gain, linear-only), after this hop's usable-linear verdict is
final and before any context or early exit. With
`output_capture_when_linear_unusable` set (default ON), the verdict false and
`sum(|E|²) > sum(|Y|²)` (strict `>` over the windowed spectra, `E = error_spec`,
`Y = error_spec + echo_spec`), the base is `Y` and `guard_active` is set;
otherwise the base is `E`. Both energies are float32 magnitudes reduced with the
same float32 pairwise sum in C (`aec_post_eval_output_guard`) and Python
(`_guard_energy`), so identical spectra give the identical decision. C and Python compute the STFT separately, and their
upstream spectra differ by about 1e-7 relative. The C-versus-Python contract is therefore: the decision matches on every
hop except inside a tie band of 1e-6 relative energy margin, `|Σ|E|² − Σ|Y|²| / Σ|E|²`, on hops that are unusable on both
sides (a usable-verdict difference is never a tie). The band is 3.3 times the largest measured near-threshold margin
disagreement (3.3e-7 at 48 kHz/1024; 3.0e-7 at 16 kHz/256) and 4 times the largest margin at any observed mismatch
(2.48e-7); hops inside the band are counted and printed per parity cell, any mismatch outside it fails. The fixed
near-threshold fixtures (identical inputs) keep exact equality. The same rounding can also fall on a per-bin decision inside
the adaptive filter (the refined filter's H_error leakage compare `e2_refined[k] <= e2_coarse[k]`, or the noise gate
`X2[k] >= gate`); the one-bin difference then spreads to the analysis seam. Such a cell is registered as T-01 in
`c_impl/test/pyc_parity/pyc_parity.py`. T-01 is a known numeric divergence, not an equivalence: as run, the raw C and
Python outputs of such a cell can exceed the audio tolerance (`revert.rOza.16k512`: max_abs about 0.0527), and only the
forced-decision experiment shows that the difference is the registered tie. The cell is INFO only while the guard, serial
and delay gates stay green as run and the Python run with the registered decisions forced the way C takes them is green
on every gate. `res_gain` is held by three gates: its relative L2 weighted by each bin's near-end power `|E + echo|²` (at
most 1e-4), its largest difference over the bins within 20 dB of the hop's peak near-end power (at most 3e-3), and, on
the cells that dump the rendering base, the rendered product `output_base_spec * res_gain` per hop (relative L2 at most
3e-3 on every hop within 60 dB of the loudest); its all-bin maximum is printed only. Every compared dump must be
non-empty, of equal length on both sides and finite, or the cell fails.
`echo_spec` is not altered by the guard. The guard can fire for the whole of a
clip whose estimate is never declared usable; it is not confined to the start of
a stream.

**Consumers.** Full mode renders `base * gain`, then CNG, IFFT, synthesis window
and OLA. The context API publishes the base (`AecResContext.output_base_spec`
and `guard_active` in Python; `aec_get_output_base()` returning an
`AecOutputBase` in C) and a consumer that renders itself multiplies
`output_base_spec * gain(s)` — never `error_spec`. The C base aliases a
dedicated buffer that CNG and other scratch never touch, valid until the next
processing or reset call.

**Reset serial.** A hard post-chain reset clears the library's OLA tail and
reseeds its CNG to `AEC_CNG_SEED`. It has two sources. `aec_reset` is a separate
call, not part of any processing call (the serial already reads +1 when it
returns). A confirmed delay change that restarts the filter (hard shift, first
acquisition without warm transfer or soft recovery) happens inside the hop's
processing and before that hop's synthesis. Soft or warm realignments, the
bad-relock revert (a soft revert: no serial bump, OLA tail and CNG keep running)
and `aec_apply_external_realign` do not count. A non-consuming, wrapping `uint32` counter publishes it: C
`aec_post_reset_serial()`, Python `AEC.get_post_reset_serial()` and
`AecResContext.post_reset_serial` (0 on a fresh instance; it keeps counting
across `reset()`). An external synthesizer keeps its own last-handled serial;
when the value read after a processing call differs, it must, before its first
CNG draw and before the IFFT/OLA push of that same hop, clear its OLA tail and
reseed its CNG to `AEC_CNG_SEED`, then store the new serial. It never resets the
AEC in response. The library touches neither NR nor NN state; following the
reset there is the application's policy.

---
