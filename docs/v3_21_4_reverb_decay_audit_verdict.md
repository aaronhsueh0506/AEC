# v3.21.4 ReverbDecayEstimator audit verdict — CLOSED NOT-PORTING

**Date**: 2026-05-21
**Branch**: v3.21.4 HEAD
**Trigger**: v3.21.2 plan carry-over — port the missing pieces of
AEC3 `ReverbDecayEstimator` (411 LOC) on top of our 139-LOC simplified
port (`AnalyzeFilter` + `EarlyReverbLengthEstimator` + validation
gates, ~270 LOC missing).

## Audit method

Instrumented `python/modules/residual/reverb_decay_estimator.py` to
trace per-frame decay values + early-return reasons across 4
representative cases (1 FS_static, 1 DT_static, 1 DT_movement worst-
case, 1 NE).

For each case captured:
- decay value distribution
- gate-input distribution (`usable_linear_filter`, `filter_quality`,
  `filter_delay_blocks`, `stationary_block`)
- inside-estimator early-return reasons
- `n_partitions` and `delay_blocks` distribution

## Findings

### 1. Decay stays at default 0.85 across all 4 sampled cases

| Case | n_frames | decay min | decay max |
|---|---:|---:|---:|
| FS_static 9xjhi | 2188 | 0.8500 | 0.8500 |
| DT_static QK70 | 4307 | 0.8500 | 0.8500 |
| DT_static oEyXu (U4.B worst) | 3657 | 0.8500 | 0.8500 |
| NE PZ7V0 | 1093 | 0.8500 | 0.8500 |

The estimator NEVER adapts. The reverb decay value driving
`ResidualEchoEstimator._update_reverb_linear` is **always the static
config default `0.85`** — same as the non-adaptive fallback.

### 2. Architectural granularity mismatch

| Param | Value | Note |
|---|---:|---|
| `n_partitions` | 6 | filter_length=832 / hop_size=160, with partition rounding |
| `K_EARLY_REVERB_MIN_SIZE_BLOCKS` | 3 | (= AEC3 constant, ported unchanged) |
| Required for late split | `delay_blocks ≤ n_partitions - K - 1 = 2` | |
| Actual `delay_blocks` (median) | 1 (9xjhi) / 7 (QK70) / 3 (oEyXu) | |

On 3 of 4 sampled cases, `delay_blocks` exceeds the regression
feasibility window. Even on cases where `delay_blocks=1` passes,
`late_end - late_start = 6 - (1+3) = 2` barely meets the
`min_tail_partitions=2` floor — leaving 2 partitions of data points
for the slope regression (too few for a stable estimate).

AEC3 operates on **64-sample blocks** (not our 160-sample partitions),
giving 13 blocks for a 832-sample filter — plenty of room for the
early/late split. Our partition-level port has fundamentally too few
data points for the regression to work meaningfully.

### 3. Upstream gates intermittently block the estimator

`_aec3_state.usable_linear_estimate()` fires ~89-94% on FS / DT cases
(post-startup) but 0% on the NE case. Multi-layer gating involves:
- FilteringQualityAnalyzer 4-gate AND (startup ≥ 40 hops, reset ≥ 20
  hops, external_delay OR convergence_seen, NOT transparent_mode)
- StationarityEstimator block_stationary flag
- Convergence latch via bridge.filter_converged

Even with our simpler port, gates block enough of the call window that
the regression rarely accumulates evidence before being interrupted by
the next gate-close or recovery event.

### 4. Codex #2 (v3.21.3) recreates the estimator on filter recovery

`_reset_filter_derived_state()` calls `_reset_aec3_post(preserve_render_side=True)`
which recreates `_aec3_ree = ResidualEchoEstimator(...)` → new
`_reverb_decay_est` instance with state reset. On delay_first /
delay_shift / p3h_diverged events the estimator loses all its
accumulated regression evidence. This compounds with finding (3).

## Why a full AEC3 port wouldn't help

The full AEC3 port (270 LOC: `AnalyzeFilter` + `EarlyReverbLengthEstimator`
+ `LateReverbLinearRegressor` + validation gates) operates at the
64-sample block level. To port it faithfully on our partition layout
we'd need:

- IFFT each partition `W[p]` to recover time-domain `h[p*160:(p+1)*160]`
- Run AnalyzeBlockGain on 64-sample sub-blocks within each partition
- Accumulate `log2(h²)` per-sample into the EarlyReverb 9-window
  regressors

This is feasible but doesn't address findings (3) and (4): even with
a faithful port, the upstream gates would still block most update
calls AND the Codex #2 recreate-on-recovery would still erase state.

The full port would only help meaningfully after both:
- Upstream gates are loosened or audited (separate sprint)
- Estimator state is preserved across `_reset_filter_derived_state()`
  (would re-classify it as render-side instead of filter-derived)

## Verdict

**CLOSED NOT-PORTING** for v3.21.4.

The simpler 139-LOC port is dormant in production — its `decay()`
return is always `default_decay=0.85` (= the non-adaptive fallback
value). The full AEC3 port would change zero observable behavior
without first addressing the upstream gating and reset semantics.

Reverb decay constant `0.85` IS the operating reverb value driving
`R²` in `_update_reverb_linear`. This constant has been
empirically-validated across all v3.21.x bench cycles. No bench
movement expected from the port.

## Future work (v3.22+)

Pre-requisites for re-attempting the port:

1. **Loosen upstream gates** — audit `FilteringQualityAnalyzer` 4-gate
   AND; the convergence_seen latch + external_delay-vs-convergence
   either/or is the main blocker on NE-only cases.

2. **Move estimator state to render-side preservation** — add
   `_aec3_reverb_decay_est` as a render-side AEC3 post-state field
   in `_reset_aec3_post(preserve_render_side=True)`. Reverb decay is
   genuinely a property of the ROOM (render side), not the filter
   tap history, so this is a correctness fix as well.

3. **THEN** consider the full AEC3 port if (1)+(2) reveal that the
   simpler port is bench-limiting.

## Files

| Artifact | Path |
|---|---|
| Our current port | [python/modules/residual/reverb_decay_estimator.py](../python/modules/residual/reverb_decay_estimator.py) (139 LOC) |
| AEC3 source reference | [docs/aec3_extracts/src/aec3/reverb_decay_estimator.cc](aec3_extracts/src/aec3/reverb_decay_estimator.cc) (411 LOC) |
| AEC3 header | [docs/aec3_extracts/src/aec3/reverb_decay_estimator.h](aec3_extracts/src/aec3/reverb_decay_estimator.h) |
| Static fallback consumer | [python/modules/residual/residual_echo_estimator.py:362-376](../python/modules/residual/residual_echo_estimator.py#L362-L376) `_reverb_decay()` |
