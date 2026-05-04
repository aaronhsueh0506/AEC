# Phase 0 — AEC3 ↔ aec.py state mapping

**Branch**: `algo/f1-f4-investigation`  •  **Date**: 2026-05-02

Maps the four AEC3 runtime states the next round will trace against
the existing v3.8.1 implementation. Goal of Phase 0: surface these
states as diagnostics, run on 800-case (no audio change), and check
whether worst-DT and FS-regression-risk cases separate by these
states. Result decides whether Phase 1+ proceeds.

## 1. `initial_state` (AEC3)

**AEC3 def**: startup phase before filter has demonstrated convergence.
Uses startup-data-accumulator (≥ 0.4 s) + reset-data-accumulator (≥ 0.2 s
post-reset) + convergence-seen counter.

**v3.8.1 analog**: implicit via `_filter_once_converged` (set in
`FilterConvergenceAnalyzer.update_convergence`, `aec.py:2425`). No
explicit duration tracking after first convergence; mark_diverged
clears `_converged` but NOT `_once_converged` (see L2398-2401).

**Phase 0 diagnostic** (`initial_state_active`, per-frame bool):
```python
initial_state_active = (
    not self._filter_once_converged
    or self._frame_count < (self.config.warmup_frames + 50)
)
```
(captures both pre-convergence AND a 50-frame tail after first conv).

**Transition signal** (`initial_transition_triggered`, per-frame bool):
fires once when `_just_converged` flag is True
(`FilterConvergenceState.just_converged`, `aec.py:2365` returned by
`update_convergence` as bool).

## 2. `usable_linear_estimate` (AEC3)

**AEC3 def**: linear filter estimate trustworthy for residual echo
attribution. Requires startup_period_done + reset_period_done +
convergence_seen + transparent_state + filter_analyzer.

**v3.8.1 analog**: `AecState.usable_linear_estimate` (`aec.py:2702-2712`):
```python
return (self._conv.once_converged
        and self._conv.converged
        and not self._epc.active)
```
Lacks: post-reset cooldown, divergence-bounded check, multi-frame
convergence count.

**Phase 0 diagnostic** (`usable_linear_estimate_v2`, per-frame bool):
```python
usable_linear_estimate_v2 = (
    self._aec_state.usable_linear_estimate           # current 3-cond gate
    and self._frame_count > self.config.warmup_frames + 30
    and self._convergence.divergence < 0.3           # FilterConv divergence < 30%
    and self._epc_hangover_count < 1                 # past EPC recovery
)
```

## 3. `dominant_nearend` (AEC3)

**AEC3 def**: per-frame state where near-end speech dominates over
residual echo + noise floor. Used to relax suppression in the linear
suppressor's initial phase.

**v3.8.1 analog**: NONE. Existing per-bin `dt_per_bin = max(eff_dt,
1.0 - coh2)` (aec.py:1711-1714) is a per-bin proxy but not a frame-level
state with hold counter.

**Phase 0 diagnostic** (`dominant_nearend_like_state`, per-frame bool):
computed inside `ResFilter._compute_gain` where all signals are local:
```python
# Mean over voice band (avoid noise-only bins)
ne_est_mean   = np.mean(nearend_est[vb_start:vb_end])
res_psd_mean  = np.mean(residual_echo_psd[vb_start:vb_end])
noise_psd_mean = np.mean(self.noise_psd[vb_start:vb_end]) + 1e-10

dominant_nearend_like = (
    ne_est_mean > 3.0 * res_psd_mean
    and ne_est_mean > 5.0 * noise_psd_mean
    and not initial_state_active
)
```
Hold counter: 5 frames (50 ms). Exported via `_diag` so AEC.process
can read.

**Why these thresholds**: AEC3 uses similar magnitudes (×3 over residual,
×5 over noise) per its DominantNearendDetector source. Match orders of
magnitude; tune if Phase 0 shows population correlation.

## 4. `ERLE reset` taxonomy (AEC3)

**AEC3 def**: distinct reset events (delay shift, gain change, startup
transition, EPC) reset different subsystems — ERLE/RES trust without
filter weights, etc.

**v3.8.1 analog**: `EchoPathChangeDetector` has 3 trigger sources
(delay / epv / shadow_rise) at `aec.py:2719-2723` but downstream all
collapse to `FilterConvergenceAnalyzer.mark_diverged()` + mu_scale
boost. Reset granularity is binary.
- `FilterErleEstimator.reset()` — exists (aec.py:975), called only by
  `AEC.reset()` not by EPC.
- `FullbandErleEstimator.reset()` — same.
- `ResidualEchoEstimator.reset()` — exists (aec.py:2536), called only
  by `AEC.reset()`.

**Phase 0 diagnostic** (`erle_reset_signal`, per-frame source enum,
0=none, 1=startup-tail, 2=epc, 3=divergence-spike): trace which
reset class would fire on each frame. Don't actually reset anything —
just count and label.

```python
if self._just_converged:                      # initial-state ending
    erle_reset_signal = 1
elif self._aec_state.epc_active:              # EPC source
    erle_reset_signal = 2
elif self._convergence.divergence > 0.5:      # divergence event
    erle_reset_signal = 3
else:
    erle_reset_signal = 0
```

## Aggregated per-case stats to dump

For each 800-case run, `bench_aecmos.py` will be extended to record
per-case (averaged over frames):

- `initial_state_pct`: % frames `initial_state_active`
- `transition_count`: count of `initial_transition_triggered` events
- `usable_linear_v2_pct`: % frames `usable_linear_estimate_v2`
- `usable_linear_v1_pct`: % frames `_aec_state.usable_linear_estimate` (control)
- `dominant_nearend_pct`: % frames `dominant_nearend_like_state`
- `erle_reset_dist`: counts per source {0,1,2,3}
- existing: `dt_from_zero_count`, `_filter_once_converged`,
  `_far_active_blocks`, `_erl_estimate`

## Phase 0 acceptance

This is **observe-only**. The decision criteria are *qualitative*:
- Does `dominant_nearend_pct` distribution differ on baseline-worst-20
  DT_movement vs random 20 DT_movement cases?
- Does `usable_linear_v2_pct` differ on cases that were marginal in F3?
- Does `transition_count` cluster on cases that improved or regressed
  in F1?

If at least 2 of these 4 questions show clear separation (>20 pct-pt
gap between worst vs random), Phase 1+ becomes feasible. If 0-1 show
separation, the AEC3 state model itself is not the right discriminator
on this dataset; STOP and try a fundamentally different selector
(per-bin spectral / learned NE classifier).
