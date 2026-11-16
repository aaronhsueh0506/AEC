[Algorithm reference](../aec_methods.md)

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

The C `MATCHED` reference ring, single channel and array, is sized to the estimator's reach plus one hop (see `docs/reference/lifecycle.md` §4); the
matched filter's own reliable reach is about 509 ms (see
[`delay_estimator_design_zh_TW.md`](../delay_estimator_design_zh_TW.md)). The
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
reference and as a diagnostic (Python only: the C estimator keeps neither the
accumulated error nor the pre-echo histogram). Pinned by
`test_delay_dominant_selection` (Python and C).

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
(`_ne_recent_frames > 0`, the held gate described in [§3.4](postfilter.md#34-suppressiongain)), those triggers drop
the full reset and instead **realign softly** — keep the converged filter,
re-adapt at normal step size. Far-end single-talk never arms the gate, so its
recoveries stay aggressive and FS echo depth is preserved. This is the partner
mechanism to the DT-gated RES floor ([§3.4](postfilter.md#34-suppressiongain)).

### 2.4 PBFDKF refined filter

`PBFDKF` (Partitioned Block Frequency Domain Kalman Filter,
`modules/filters.py`) is the production filter. It is a `PBFDAF` with a per-bin
Kalman gain: the overlap-save buffer is 2 × hop, the FFT size is the next power
of two of that (256 on the 16 kHz default grid), and
`n_partitions = ceil(filter_length / hop)`. Kalman state per partition: filter
coefficients `W[p, k]`, error covariance `P[k]`, process noise `Q[k]`.

Convergence schedule: `Q` runs at `kalman_q_high` (1e-3) on start and after an
EPC / delay change, and settles to `kalman_q_low` (1e-6) in steady state
(Python only: the gain does not read `Q`, so the C filter keeps no `Q`, `R` or
`kalman_q_*`); `warmup_frames` (preset 100) forces the high-step startup window.

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
