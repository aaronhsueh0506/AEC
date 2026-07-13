# AEC Algorithm Methods (current through v3.23.0)

**Release**: algorithm current through v3.23.0. The split min-gain floor (§3.4)
was added in v3.22.0 on top of the v3.21 pipeline, the three Pareto presets
(gentle/balanced/aggressive) in v3.22.4, and the **DT-aware recovery stack**
(matched-filter pre-echo fix + DT-aware soft recovery + DT-gated RES floor, §2.3
and §3.4) in v3.23.0 — the work that closed the no-pre-align (production-faithful
online self-align) delay-acquisition gap. The C port's **non-FFT logic is
bit-exact** with the Python reference (verified under `-DUSE_STANDARD_MATH`),
except the C delay-estimator matched filter, which intentionally diverges
(unconditional float32 fast-math + duty-cycling; see §7); the
production FFT backend is **KISS FFT (float32)**, so the end-to-end C output
aligns with Python to ~float32 precision (correlation 0.99999958, ≈ −60 dB,
inaudible — not 0/0; see §7). The C structure mirrors Python class boundaries.

This document is the deep algorithm specification for the production
pipeline. The v3.21 release retires the legacy 9-stage `ResFilter`
post-filter in favour of an AEC3-aligned chain
(`AecState` + `ResidualEchoEstimator` + `SuppressionGain` + CNG). See
[architecture_v3_22_5_vs_aec3.html](architecture_v3_22_5_vs_aec3.html)
for the side-by-side flowcharts of the current build and the WebRTC AEC3 reference.

| Companion doc | Purpose |
|---|---|
| [`aec_algorithm_guide.html`](aec_algorithm_guide.html) | Presentation overview |
| [`architecture_v3_22_5_vs_aec3.html`](architecture_v3_22_5_vs_aec3.html) | Architecture flowcharts vs WebRTC AEC3 (the v3.23.0 DT-aware stack is config-only over this pipeline) |
| [`c_user_and_integration_guide.md`](c_user_and_integration_guide.md) | C API, integration, streaming contract |
| [`pbfdkf_shadow_intro.md`](pbfdkf_shadow_intro.md) | PBFDKF + shadow design |
| [`nn_integration_interface.md`](nn_integration_interface.md) | NN residual/NR/joint swap seams |
| [`../CHANGELOG.md`](../CHANGELOG.md) | Version history |

---

## 1. Pipeline overview

```
mic ─► HPF ────────────────────────────────────────────────────────────────────────►
ref ──────► Saturation ─► EchoPathDelayEstimator ─► RingBuf ─────────────┐
                          (matched filter +                              │
                           lag aggregator +                              ▼
                           clock-drift detector)                    PBFDKF refined
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
                                                                   │     over 12 sub-analyzers)
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

### Presets — one production point + two Pareto operating points (v3.22.4)

`AecPreset.BALANCED` is the production preset (all four ship bars met; R1 / R2
deleted the legacy MILD / SOFT / AGGRESSIVE / MAXIMUM 40-knob menu). It shares a
narrow tuned base with the two new strength presets:

```python
AecConfig.from_preset(AecPreset.BALANCED)  # base, equivalent to:
AecConfig(
    enable_cng=True,        # comfort-noise generator on
    shadow_mu_min=0.5,      # shadow / DT mu floor
    warmup_frames=100,      # forced-high-mu startup window
    kalman_q_high=1e-3,     # PBFDKF Q_high convergence speed
)
```

`gentle` and `aggressive` are deliberate **Pareto operating points** that differ
from `balanced` in exactly one field — `min_gain_floor_far_active_db`, the
far-active min-gain floor (the single residual-echo strength knob):

| preset | floor | character |
|---|---|---|
| `gentle` | −20 dB | near-priority — more near kept, more echo leak (DT_static deg ≈ AEC2; FS echo below balanced's 3.5 bar by design) |
| `balanced` | −28 dB | production — all four ship bars met |
| `aggressive` | −38 dB | echo-priority — beats AEC2 on DT+FS echo, deg >2.0 and above AEC3 |

The DT-deg-vs-echo trade is a proven single-channel DSP Pareto wall; the strength
axis exposes it honestly. See CHANGELOG `[3.22.4]` for the full-800 numbers.

Everything else uses `AecConfig` dataclass defaults; the AEC3 chain has
its own tuning embedded in `AecStateConfig` / `SuppressionGainConfig`
(`modules/state/aec_state.py`, `modules/residual/suppression_gain.py`).

### Determinism

Per-case CNG seed: `np.random.seed(42)` before each `AEC(cfg)` instance.
All benchmarking tooling (`eval_aec_challenge.py`, `check_byte_equal.py`,
`run_one_case.py`) seeds before instantiation.

### Tunable parameters (`AecConfig`)

The full release surface is `AecConfig` (`modules/config.py`). The three presets
differ **only** in `min_gain_floor_far_active_db` (gentle −20 / balanced −28 /
aggressive −38); `from_preset` additionally pins a shared base
(`enable_cng=True`, `shadow_mu_min=0.5`, `warmup_frames=100`,
`kalman_q_high=1e-3`). Everything below uses dataclass defaults. Fields are
grouped by subsystem; **ON**/**OFF** marks the default state of a feature flag.

| Subsystem | Field | Default | Notes |
|---|---|---|---|
| System / framing | `sample_rate` | 16000 | 8000/16000/48000 |
| | `frame_size` / `hop_size` | auto | 20 ms / 10 ms (50 % overlap) |
| | `filter_length` | auto | 52 ms (<44.1 kHz), 64 ms (≥44.1 kHz) |
| | `mu` / `delta` | 0.3 / 1e-8 | step size / regularization |
| Residual / CNG | `enable_res` | **ON** | post-filter master gate (§3.7) |
| | `enable_cng` | OFF (preset **ON**) | comfort noise (§3.5) |
| | `comfort_noise_floor_dbfs` | −96.034 | |
| | `enable_td_constraint` | **ON** | |
| C′ ERLE coherence gate | `erle_coh_gate_enabled` | **ON** | Γ²(Ŷ,Y) per-bin ERLE freeze |
| | `erle_coh_gate_threshold` / `_alpha` | 0.5 / 0.05 | |
| Split min-gain floor (§3.4) | `min_gain_split_floor_enabled` | **ON** | |
| | `min_gain_floor_far_active_db` | −28 | **preset strength knob** |
| | `min_gain_floor_far_silent_db` | −12 | pure-NE floor |
| | `min_gain_far_latch_power` | 1e6 | far-active latch |
| **DT-aware recovery group** | `dt_aware_recovery_soft` | **ON** | soft (non-destructive) delay/EPC recovery (§2.3) |
| | `dt_aware_res_floor_enabled` | **ON** | DT-gated RES floor lift (§3.4) |
| | `min_gain_floor_dt_db` | −20 | floor used in double-talk |
| | `ne_recent_threshold` | 0.3 | near-recent gate arm threshold |
| | `ne_recent_hold` | 150 | hold frames after arm |
| | `ne_recent_sustain` | 3 | frames above threshold to arm |
| Soft near-end blend | `soft_nearend_blend_enabled` | **ON** | sigmoid ENR tuning blend (D3) |
| | `soft_nearend_blend_enr_threshold` / `_softness` | 0.25 / 0.25 | |
| | `soft_nearend_blend_per_bin` | **ON** | per-bin frequency-selective near protection |
| Nonlinear residual | `nl_r2_enabled` | **ON** | Kuech-Kellermann 2nd-order R² (L1) |
| | `nl_r2_alpha` | 0.1 | |
| ERLE coordinate fixes | `erle_render_x2_psd_scale` | **ON** | revives reverb model (PSD-scale bug fix) |
| | `reverb_tail_strength` | 1.0 | reverb-tail R² scale |
| | `erle_windowed_capture_psd` | **ON** | E1 Y2/E2 sqrt-Hann coordinate consistency |
| | `output_capture_when_linear_unusable` | **ON** | E2 output base = Y on diverged frames |
| Delay-acquire guard | `delay_acquire_protect_converged` | **ON** | reject spurious late acquisition |
| Cold-start deadlock (OFF) | `h_error_refresh_erl_floor` | 0.0 (**OFF**) | Kalman-gain cold-start breakers |
| | `h_error_floor_override` | 0.0 (**OFF**) | |
| Shadow filter | `enable_shadow` | **ON** | PBFDAF/NLMS coarse filter |
| | `shadow_mu_min` / `shadow_mu_nlms` | 0.5 / 0.5 | |
| | `shadow_copy_threshold` / `_hysteresis` | 0.65 / 3 | |
| | `shadow_err_alpha` | 0.80 | |
| | `shadow_dtd_advantage_scale` / `_offset` | 3.0 / 1.5 | |
| Misadjustment estimator | `filter_misadjustment_*` | 100 / 30 / 0.5 / 2.0 | hangover / stable / scale min/max |
| PBFDKF (Kalman) | `use_kalman` | **ON** | |
| | `kalman_q_high` / `kalman_q_low` | 1e-3 / 1e-6 | |
| | `warmup_frames` | 80 (preset 100) | forced-high-mu window |
| EPC | `epc_delta_threshold` / `epc_total_rise` / `epc_hangover` | 0.3 / 1.5 / 20 | |
| Delay estimation | `enable_delay_est` | **ON** | |
| | `max_delay_ms` / `delay_buffer_ms` | 1024 / 2048 | |
| | `delay_est_period_s` / `delay_est_init_s` | 0.5 / 0.3 | |
| | `delay_par_low_threshold` / `_solid_threshold` | 5.0 / 8.0 | |
| HPF | `enable_highpass` | **ON** | mic-path only (ref HPF retired) |
| | `highpass_cutoff_hz` | 80.0 | |
| Saturation | `enable_saturation_detect` | **ON** | |
| | `saturation_threshold` | 0.95 | |
| | `saturation_softclip_ref` | **ON** | |
| Mode / output | `mode` | `PBFDKF` | |
| | `return_res_context` | OFF | switches `process()` return type (§4) |
| | `clear_filter_history` | OFF | |

The **DT-aware recovery group** (the v3.23.0 addition) is the only default-ON
group whose two halves act in tandem: `dt_aware_recovery_soft` softens the
delay/EPC recovery (§2.3) and `dt_aware_res_floor_enabled` lifts the RES floor
(§3.4), both gated on the shared `ne_recent_*` near-recent latch.

---

## 2. Front-end blocks

### 2.1 HPF (high-pass filter)

80 Hz IIR Butterworth-style on the mic path only. Defaults locked
post-v3.20:

* Mic-path HPF: **ON** (suppresses DC + sub-bass before delay alignment).
* Reference HPF: **OFF / bypassed** — the ref-path HPF was retired
  after the v3.19 HPF-flip verdict showed −0.04 nores deg with no echo
  gain; the pipeline now feeds the raw reference straight into
  `Saturation` → `EchoPathDelayEstimator`.

`HighPassFilter` lives in `modules/preprocessing.py`; mirrors C
`high_pass_filter.{h,c}`.

### 2.2 Saturation detector + soft-clip

`SaturationDetector` (preprocessing.py) tracks the running fraction of
clipped reference samples (|x| > 0.95). The detector populates
`self._saturation_level` (∈ [0, 1]); downstream consumers use it to
freeze the refined filter μ on heavy clipping and to skip the
ResidualEchoEstimator render-based path.

`saturation_softclip_ref=True` (default) soft-clips the reference
prior to delay alignment so the refined filter sees a tractable
nonlinear envelope.

### 2.3 Delay estimator (AEC3-aligned)

`modules/delay/`:

* `EchoPathDelayEstimator` (matched filter + downsampled cross-correlation).
* `MatchedFilter` (running matched-filter accumulator).
* `LagAggregator` (decision over the matched-filter histogram).
* `ClockdriftDetector` (drift over a 30 s window; AEC3 reference is
  7500 blocks ≈ 30 s).
* `DownsampledRing` (delay-aligned render history).
* `RenderDelayController` (orchestrator-facing alignment + warmup).

`legacy_compat.LegacyDelayShim` exposes the old `DelayEstimator.accumulate()`
API on top of the new estimator; orchestrator imports it via the public
alias so external callers (`from aec import DelayEstimator`) keep working.

The estimator supports up to 1024 ms of skew (v3.10.4 widening). It feeds
the refined filter through `RingBuf` (a delay-line on the reference).

#### Pre-echo detection (matched-filter accumulated-error)

`MatchedFilter` (`modules/delay/matched_filter.py`) is a bank of `num_filters=5`
NLMS cross-correlators over staggered lag windows. The winning lag is normally
`MaxSquarePeakIndex(h) + n·intra_lag_shift` — the strongest squared tap. AEC3
additionally tracks a **pre-echo lag** so the reported delay snaps to the true
echo *onset* rather than the strongest tap (which, with reverb, can sit well
after onset). `detect_pre_echo` is **default-True** (`echo_path_delay_estimator.py:61`,
AEC3 strict default `echo_canceller3_config.h:73`): the `PreEchoLagAggregator`
may override the highest-peak lag with the pre-echo candidate when the
accumulated-error curve shows a leading peak.

The pre-echo locator (`ComputePreEchoLag`, AEC3 `matched_filter.cc:516-524`)
walks the per-tap-group **accumulated squared error of the filter PREFIX** and
returns the earliest group whose prefix already explains the capture (= onset).
Correctly populating that curve was the load-bearing fix:

```
# matched_filter_core, lines ~97-108 — per sub-block sample i:
g      = _ACCUMULATED_ERROR_SUBSAMPLE_RATE          # 4
prefix = np.cumsum((h * x_window).reshape(-1, g).sum(axis=1))   # Σ h[:G]·x[:G]
diff   = prefix - y[i]
instantaneous_error_out[:diff.size] += diff * diff  # prefix-error per tap-group
```

The **bug this replaced**: the previous code binned the *full-filter* error by
output-sample index, which left all but the first few bins zero — so the
accumulated-error curve never developed a leading peak and `ComputePreEchoLag`
always collapsed the pre-echo lag to ≈0. With `detect_pre_echo=True` overriding
the highest-peak lag, the reported delay was pulled back toward 0, which broke
online (no-pre-align) delay acquisition. The cumsum prefix-error binning above
restores the AEC3 onset locator; the C port carries the same fix.

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
re-adapt at normal step size (`orchestrator.py:1386,1453,1899,1941`). Far-end
single-talk never arms the gate, so its recoveries stay aggressive and FS echo
depth is preserved. No-pre-align 800-case: DT_static deg 1.993→2.016,
DT_movement 2.071→2.088, FS echo holds >3.5. This is the partner mechanism to
the DT-gated RES floor (§3.4); together they make up the v3.23.0 DT-aware
recovery stack that closed the no-pre-align acquisition gap.

### 2.4 PBFDKF refined filter

`PBFDKF` (Partitioned Block Frequency Domain Kalman Filter,
`modules/filters.py`) is the production filter. Block size 64 (4 ms at
16 kHz), FFT size 128; partition count `n_partitions = filter_length / 64`.
Kalman state per partition: filter coefficients `W[p, k]`, error
covariance `P[k]`, process noise `Q[k]`.

Convergence schedule:

```
mu_scale     = 1.0 for warmup_frames=100 (forced high)
Q_high       = 1e-3 (initial / re-converge mode)
Q_low        = 1e-6 (steady state)
P_max_override frames = 30  on EPC / delay change
```

The refined filter consumes the `RenderSignalAnalyzer` per-bin mask
(narrow-band tonal regions freeze the W update on those bins) and the
`poor_signal_excitation` flag (insufficient reference excitation also
freezes W).

`FilterStateBridge` (`modules/filter/filter_state_bridge.py`) is the
read-only seam that exposes refined-filter spectra / convergence state
to the AEC3 post-filter chain.

### 2.5 Shadow filter (coarse adapter)

v3.21 default: NLMS (`PBFDAF`) at μ = 0.5, the AEC3 coarse-filter role.
The shadow runs in parallel with the refined filter; its job is to be
the fast-tracker that survives echo path changes, while the refined
filter holds the converged tuning.

`PathChangeRegimeHandler` (`modules/epc.py`, formerly `ShadowCopyController`)
arbitrates between the two filters:

* Fires `boost_q` / `reverse_copy` / `main_paused` decisions per frame
  based on relative error EMAs, render activity, and the shadow-DT
  signals from `DoubleTalkAnalyzer` (energy-gated; the opt-in DTD detector
  subsystem was retired in 3.22.5).
* Load-bearing on ~7/800 cohort-tail cases.
* Must not be removed or bypassed (see CLAUDE.md).

### 2.6 Render signal analyzer

`RenderSignalAnalyzer` (`modules/render/render_signal_analyzer.py`)
mirrors AEC3's `render_signal_analyzer.cc`. Two outputs:

1. Per-bin freeze mask for refined-filter gain compute (strong narrowband
   peaks freeze for `strong_peak_freeze_duration = n_partitions` blocks).
2. `poor_signal_excitation` flag that zeroes the refined-filter mu when
   the reference is too quiet for ~1 s.

---

## 3. AEC3 post-filter (`_aec3_post`)

After the refined filter produces the linear residual error spectrum
`E(k)`, `AEC._aec3_post()` (in `modules/orchestrator.py`) runs the
AEC3-aligned suppression chain:

### 3.1 StationarityEstimator

`modules/state/stationarity_estimator.py` mirrors
`stationarity_estimator.cc`. Per-frame per-bin EMA of render PSD plus a
running coefficient-of-variation gate. Two consumers:

* `_aec3_post` zeros R² (residual echo PSD estimate) on stationary bands
  so a steady hum doesn't drive the suppressor.
* Refined filter skips W update on block-stationary frames, preventing
  the PBFDKF from learning mic-as-echo coupling against an uncorrelated
  stationary input (the `9xJH` / `E0l0` / `wJVP` NE-outlier class).

### 3.2 AecState

`modules/state/aec_state.py` is the v3.21 AEC3-aligned read-only ADT.
It wraps 12 sub-analyzers under `modules/state/`:

```
AecState
├─ FilterAnalyzer        (impulse-response shape consistency)
├─ FilteringQualityAnalyzer (multi-gate usable_linear_estimate)
├─ ErlEstimator          (ERL adaptive EMA)
├─ ErleEstimator         (subband ERLE EMA)
├─ FullbandErleEstimator (fullband ERLE EMA)
├─ SubbandErleEstimator  (sub-band ERLE EMA — AEC3-aligned)
├─ SaturationDetector    (proxy to preprocessing detector)
├─ InitialState          (startup gating)
├─ TransparentMode       (no-echo-to-cancel pass-through; default off)
└─ ...                   (StationarityEstimator wiring + render activity)
```

The legacy `AecState` aggregator (formerly `modules/legacy_state.py`)
was retired in v3.21 cleanup R8 because it was a thin facade over the
five legacy detectors with no AEC3-substrate value — the
`modules/state/aec_state.py` port replaces it.

### 3.3 ResidualEchoEstimator

`modules/residual/residual_echo_estimator.py` produces the per-bin
residual-echo PSD estimate R²(k). Two paths:

* **Linear**: `R² = ERLE_subband × E²(k)` — the AEC3-aligned ERLE-divide.
  Active when AecState reports `usable_linear_estimate`.
* **Render-based**: `R² = ERL_subband × X²(k)` — uses the long-window
  far-PSD EMA when the linear path is unreliable (epc_active, saturation,
  pre-convergence). Mirrors AEC3 `residual_echo_estimator.cc:269`.

A reverb-tail term from `ReverbModel` (`modules/residual/reverb_model.py`)
is added when reverb estimation is online; tracks per-bin reverb energy
across the impulse response tail using `ReverbDecayEstimator` +
`ReverbFrequencyResponse`.

The legacy two-path attribution (`attribute_legacy` in
`modules/residual_estimator.py`) is kept for backward-compat callers but
no longer driven by the production path.

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

**Split min-gain floor (v3.22.0, default ON).** The AEC3 min-gain floor
`min_gain = min_echo_power / R²` collapses to ~0 in double-talk: near-end
inflates the error → ERLE drops → R² spikes → the floor that should protect
near-end vanishes exactly when near-end is present (RES over-suppresses
near-end up to −8 dB; the linear filter only cuts −1.4..−2.9 dB). A
coherence double-talk discriminator was proven unworkable (single-lag X–Y
coherence cannot separate reverberant FS from DT). The fix is a power-domain
floor on the deepest suppression, **split by far-end activity**:

```
min_gain(k) = max( min_echo_power / R²(k),  floor )
floor = 10^(-28/10)  if far-active (FS/DT)   # balanced; deep far-active
                                             #   suppression for more echo
                                             #   cancellation → FS echo > AEC2
      = 10^(-12/10)  if far-silent (pure NE) # lifts NE near-end, zero echo
                                             #   cost (no echo to leak)
```

Routing uses a per-recording **latch** on instantaneous far energy
(`mean(render_block²) > 1e6`, render int16-scaled — ≈ the orchestrator's own
far-active criterion). It latches from the first far frame so FS/DT use the
gentler floor throughout (no cold-start leak); only recordings where far is
never active keep the strong floor. Config (`AecConfig`):
`min_gain_split_floor_enabled`, `min_gain_floor_far_active_db` (balanced −28;
gentle −20 / aggressive −38 — the v3.22.4 preset strength knob),
`min_gain_floor_far_silent_db` (−12), `min_gain_far_latch_power` (1e6). The
floors are amplitude-domain dB; the orchestrator constructs `SuppressionGain`
with these as `split_floor_*` (`orchestrator.py:960`). The far-active floor
above is further lifted in double-talk by the DT-aware floor gating below.

**Double-talk floor protection (DT-aware floor gating, v3.23.0, default ON).**
The far-active floor (−28 dB) is deliberately aggressive so pure far-end
single-talk cancels deep echo — but applying that same deep floor *during
double-talk* over-suppresses near-end. The DT-gated floor lifts the floor from
`min_gain_floor_far_active_db` (−28 dB) to `min_gain_floor_dt_db` (**−20 dB**)
*only when near-end is present*, selectively protecting near-end in DT while
keeping the aggressive far-active floor in pure FS:

```python
# suppression_gain.py ~928 (_get_min_gain):
if self._far_active_latched:
    base_floor = (self._split_floor_dt if self._dt_protect_active   # DT: −20 dB
                  else self._split_floor_far_active)                # FS: −28 dB
else:
    base_floor = self._split_floor_far_silent                       # pure NE: −12 dB
```

The orchestrator sets `sg._dt_protect_active` immediately before `get_gain`
(`orchestrator.py:3444`):

```python
self._aec3_sg._dt_protect_active = bool(
    self.config.dt_aware_res_floor_enabled        # default True
    and self._ne_recent_frames > 0)               # near recently present
```

`_ne_recent_frames` is the held "near-end seen recently" gate (same mechanism
that arms `dt_aware_recovery_soft`, §2.3): the `_dt_from_energy` indicator
must stay above `ne_recent_threshold` (0.3) for `ne_recent_sustain` (3) frames
to arm, then holds for `ne_recent_hold` (150) frames (`orchestrator.py:1323-1336`).
Far-end single-talk never sustains the indicator, so FS recoveries keep the −28 dB
floor and FS echo depth is preserved; the energy gate does false-arm a little on
loud FS echo (the bounded FS cost). Config (`AecConfig`):
`dt_aware_res_floor_enabled`, `min_gain_floor_dt_db` (−20),
`ne_recent_threshold`/`ne_recent_hold`/`ne_recent_sustain`. On the
no-pre-align baseline (on top of `dt_aware_recovery_soft`): DT_static deg
2.016→2.074, DT_movement 2.088→2.140 (both exceed the pre-align result), at a
bounded echo cost (DT echo 4.28→4.22, FS echo 3.59→3.54), all ship bars held.

### 3.5 Comfort noise (CNG)

When `enable_cng = True` (default), `_aec3_post` adds per-bin shaped
noise scaled by `sqrt(1 - G²)` to mask the suppressed spectrum. State:
per-bin noise PSD EMA + smoothed cn gain (init in `AEC.__init__`).

### 3.6 OLA synthesis

sqrt-Hann analysis × sqrt-Hann synthesis = Hann, sums to 1 across 50 %
overlap hops. `_aec3_synth_window` + `_aec3_ola_buf` accumulate the
windowed inverse FFT into the output stream.

### 3.7 enable_res = False bypass

The byte-equal harness renders each case twice — once with
`enable_res=True` (production) and once with `enable_res=False`. The
latter skips the entire `_aec3_post` body and emits the linear residual
straight from the refined filter into `_ours_nores.wav`. The gate at
`orchestrator.py:_aec3_post` block-open uses
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

### `trace_p52_regime_handler`
Per-frame regime-handler trace flag (default OFF). Classifier in
`modules/p52_regime_classifier.py` is analysis-only; the
`AntiLoopholeTests` in `python/test_p52_regime.py` enforce that the
classifier never modifies audio.

---

## 5. Tooling

* `python/aec.py` — top-level shim + CLI entry point.
* `python/eval_aec_challenge.py` — render 800-case AEC Challenge corpus.
  Standard config: `--preset balanced --filter 832 --cng --parallel --workers 4`.
* `python/bench_aecmos.py` — AECMOS scoring driver
  (needs `speechmos` + `onnxruntime ≤ 1.16.3` + `numpy < 2`).
* `python/check_byte_equal.py` — 25-case representative byte-equal harness
  (5 per bucket at echo percentiles 0/25/50/75/100); reference at
  [`bench/v3_21_3aadd2d_baseline/`](bench/v3_21_3aadd2d_baseline/README.md).
  Must report `=== 25/25 PASS, 0 FAIL ===` before any commit that touches
  Python code outside `docs/`.
* `python/run_one_case.py` — single-case dev tool with diagnostic
  5-panel PNG (waveforms + spectrograms + ERLE).
* `python/run_e2e_parity.py` — Python ↔ C parity driver.
* `python/batch_c_eval.py` — batch-run the C `aec_wav` binary.
* `python/test_p52_regime.py` — pytest for the regime classifier
  anti-loophole contract.

---

## 6. Reference scores (current — no pre-alignment)

The numbers below are the **production-faithful** 800-case AECMOS bucket means
for the **BALANCED** preset with **no pre-alignment** — the delay is acquired
online by the in-pipeline matched-filter self-align (§2.3), exactly as the
shipped algorithm runs. This is the only result presentation that reflects
production: offline GCC pre-align is a benchmark-only crutch and its numbers are
**not** a current result.

| Bucket       |  echo (↑) |  deg (↑) |
|--------------|----------:|---------:|
| FS_static    |     3.544 |        — |
| FS_movement  |     3.519 |        — |
| DT_static    |     4.218 |    2.074 |
| DT_movement  |     4.114 |    2.140 |
| NE           |         — |    4.021 |

All four ship bars are met:

* **FS echo > 3.5** — FS_static 3.544, FS_movement 3.519 ✓
* **DT echo > 4** — DT_static 4.218, DT_movement 4.114 ✓
* **DT deg > 2.0** — DT_static 2.074, DT_movement 2.140 ✓
* **NE deg ≥ 4** — NE 4.021 ✓

These reflect the v3.23.0 DT-aware recovery stack (matched-filter pre-echo fix
§2.3 + `dt_aware_recovery_soft` §2.3 + DT-gated RES floor §3.4), which closed the
no-pre-align acquisition gap — the DT deg figures (2.074 / 2.140) now exceed what
the offline pre-align crutch produced.

Reference points (AEC2 / AEC3 anchors from the AEC Challenge corpus):

* **AEC2 reference** — DT static / movement deg ≈ 2.39; NE deg ≈ 4.10.
* **AEC3 reference** — DT static / movement deg ≈ 1.85; NE deg ≈ 3.45.

BALANCED beats AEC3 on every DT/NE deg bucket and holds all four echo/deg ship
bars under online self-align.

> **Historical (pre-align, NOT a current result).** The retired v3.21 anchor
> (`docs/bench/v3_21_3aadd2d_baseline/`, commit `3aadd2d`) was scored *with*
> offline GCC pre-alignment, which inflates FS deg to ≈5.0 and shifts the echo
> means; those numbers are kept only as a historical anchor and must never be
> presented as a current production result.

---

## 7. Python ↔ C parity

The `c_impl/` port is the production C implementation of the **same algorithm**.
The **non-FFT C logic is bit-exact** to the Python reference — verified at
v3.23.0 under `-DUSE_STANDARD_MATH` with a numpy-precision FFT: every per-module
parity test (`c_impl/test/parity_*.c` — PBFDKF, delay, AecState,
ResidualEchoEstimator, SuppressionGain, the ERLE/reverb estimators, etc.) plus
the end-to-end `parity_aec_e2e` passed with 0 mismatches across all three
presets over the full ≈4186-hop doubletalk case.

**Permanent exception (2026-07-13)**: the C delay-estimator matched filter now
runs float32 fast-math dot products + duty-cycled analysis **unconditionally**
(sampled at zero AECMOS cost on the 60-case stratified gate), so the delay
chain no longer bit-matches Python by design; `c_impl/test/parity_delay.c` is
since then a **C-regression golden** (regenerated by
`c_impl/test/gen_delay_c_golden.c`), not a Python-parity gate. The v3.23.0
0-mismatch record above is the historical pre-divergence verification.

The **production FFT backend is KISS FFT (float32)** (vendored `lib/kiss_fft/`;
NE10 ARM-NEON opt-in via `make NE10_DIR=...`), which is NOT bit-exact to numpy's
fp64 `np.fft`. Because PBFDKF and the post-filter are frequency-domain, this
float32 FFT difference propagates through the whole pipeline, so the **end-to-end
C output aligns with Python to ~float32 precision, NOT 0/0**: correlation
0.99999958, RMS error ≈ −60 dB below signal, per-sample max ~6e-3 (inaudible).
`parity_aec_e2e.c` now asserts the linear `raw_output` and final `out[hop]` stay
within a **2e-2 float32 tolerance** (not strict equality) and is the
authoritative gate. The per-module FFT-path tests' strict 0/0 checks were
written against the removed pocketfft backend; converting them to per-module
float32 tolerances is a follow-on.

### Math backends

`c_impl/include/fast_math.h` provides two implementations selected at compile
time (orthogonal to the FFT-backend choice above):

* **Default (production): `fast_math.h` approximations.** `fast_exp` (LUT +
  Taylor), `fast_log`/`fast_log10` (IEEE-754 + Taylor), `fast_sqrt`
  (Newton-Raphson). These add a documented **~1e-5 .. 1e-4 tolerance** in the
  stages that call `exp`/`sqrt` (SuppressionGain Wiener mapping, CNG gain, dB
  conversions) — on top of the float32-FFT tolerance above.
* **`-DUSE_STANDARD_MATH`: libm.** Swaps the approximations for `expf`/`logf`/
  `sqrtf` (`fast_math.h:77-84`) — this isolates the non-FFT logic for the parity
  logic checks. With the KISS FFT it does NOT recover end-to-end 0/0 (the
  float32 FFT difference remains); the historical numpy-bit-exact gate used the
  pocketfft backend (since removed).

`-ffp-contract=off` is mandatory in `CFLAGS` on both backends (no fused
multiply-add reassociation), and the C FFT is pocketfft (fp64-internal) to match
numpy. The C structure mirrors the Python class boundaries one-to-one (`PBFDKF`,
`ShadowFilter`, `AecState`, `SuppressionGain`, `EchoPathDelayEstimator`, …), so a
module parity test isolates any divergence to a single stage. See
`python/run_e2e_parity.py` (the parity driver) and
`docs/c_user_and_integration_guide.md` (build + integration).
