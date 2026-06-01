# AEC Algorithm Methods (v3.22.0)

**Release**: v3.22.0 (2026-06-01). Python `aec.py` `__version__ = "3.22.0"`.
C port lag behind Python; the C structure mirrors Python class boundaries.
v3.22.0 adds the split min-gain floor (§3.4) on top of the v3.21 pipeline.

This document is the deep algorithm specification for the production
pipeline. The v3.21 release retires the legacy 9-stage `ResFilter`
post-filter in favour of an AEC3-aligned chain
(`AecState` + `ResidualEchoEstimator` + `SuppressionGain` + CNG). See
[architecture_v3_10_5_vs_v3_21_vs_aec3.html](architecture_v3_10_5_vs_v3_21_vs_aec3.html)
for the side-by-side comparison with v3.10.5 and the WebRTC AEC3 reference.

| Companion doc | Purpose |
|---|---|
| [`aec_algorithm_guide.html`](aec_algorithm_guide.html) | Presentation overview |
| [`architecture_v3_10_5_vs_v3_21_vs_aec3.html`](architecture_v3_10_5_vs_v3_21_vs_aec3.html) | v3.10.5 vs v3.21 vs WebRTC AEC3 comparison |
| [`c_user_and_integration_guide.md`](c_user_and_integration_guide.md) | C API, integration, streaming contract |
| [`refactor_modules_layout.md`](refactor_modules_layout.md) | Module map |
| [`pbfdkf_shadow_intro.md`](pbfdkf_shadow_intro.md) | PBFDKF + shadow design |
| [`dtd_design.md`](dtd_design.md) | DTD algorithm |
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

### Single production preset

`AecPreset.BALANCED` is the only shipped preset (R1 / R2 deleted MILD /
SOFT / AGGRESSIVE / MAXIMUM / legacy BALANCED). Override surface in
`AecConfig.from_preset` is intentionally narrow (5 keys):

```python
AecConfig.from_preset(AecPreset.BALANCED)  # equivalent to:
AecConfig(
    enable_cng=True,        # comfort-noise generator on
    shadow_q_ratio=3.5,     # shadow Q is 3.5× refined Q (PBFDKF only)
    shadow_mu_min=0.5,      # shadow / DT mu floor
    warmup_frames=100,      # forced-high-mu startup window
    kalman_q_high=1e-3,     # PBFDKF Q_high convergence speed
)
```

Everything else uses `AecConfig` dataclass defaults; the AEC3 chain has
its own tuning embedded in `AecStateConfig` / `SuppressionGainConfig`
(`modules/state/aec_state.py`, `modules/residual/suppression_gain.py`).

### Determinism

Per-case CNG seed: `np.random.seed(42)` before each `AEC(cfg)` instance.
All benchmarking tooling (`eval_aec_challenge.py`, `check_byte_equal.py`,
`run_one_case.py`) seeds before instantiation.

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
  based on relative error EMAs, render activity, and DTD state.
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
floor = 10^(-22/10)  if far-active (FS/DT)   # caps DT gain-collapse; only
                                             #   FS cost is deep echo <-40dB
                                             #   (inaudible) → FS echo > AEC2
      = 10^(-12/10)  if far-silent (pure NE) # lifts NE near-end, zero echo
                                             #   cost (no echo to leak)
```

Routing uses a per-recording **latch** on instantaneous far energy
(`mean(render_block²) > 1e6`, render int16-scaled — ≈ the orchestrator's own
far-active criterion). It latches from the first far frame so FS/DT use the
gentler floor throughout (no cold-start leak); only recordings where far is
never active keep the strong floor. Config (`AecConfig`):
`min_gain_split_floor_enabled`, `min_gain_floor_far_active_db` (−22),
`min_gain_floor_far_silent_db` (−12), `min_gain_far_latch_power` (1e6).
800-case: FS echo 3.520 / DT echo 4.042 / DT deg 2.226 / NE deg 4.047
(all four ship thresholds; the only config to pass all four).

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

## 6. Reference scores (v3.21 baseline)

`docs/bench/v3_21_3aadd2d_baseline/balanced_aec3_result.md` records the
800-case AECMOS bucket means at the v3.21 anchor commit (`3aadd2d`):

| Bucket       |    n |  echo (↑) |  deg (↑) |
|--------------|-----:|----------:|---------:|
| FS_static    |  169 |     3.729 |    4.999 |
| FS_movement  |  131 |     3.626 |    4.999 |
| DT_static    |  186 |     4.237 |    2.387 |
| DT_movement  |  114 |     4.215 |    2.371 |
| NE           |  200 |     4.998 |    4.052 |

Reference points (AEC2 / AEC3 anchors from the AEC Challenge corpus):

* **AEC2 reference** — FS / DT echo unknown (closed corpus); DT static
  deg 2.39; DT movement deg 2.39; NE deg 4.10.
* **AEC3 reference** — DT static deg 1.85; DT movement deg 1.85;
  NE deg 3.45.

v3.21 beats AEC3 on every DT/NE deg bucket by +0.52 / +0.52 / +0.60 and
matches AEC2 (~0) on NE / DT deg targets while preserving the v3.10.4
FS-echo gain (+0.10 vs v3.10.4 anchor).

The full per-case 800-case scores are at
`docs/bench/v3_21_3aadd2d_baseline/balanced_aec3_scores.json` and act as
the anchor for the Phase D-4 final-bench verify (must match within
±0.001 dB before tagging `v3.21.0`).
