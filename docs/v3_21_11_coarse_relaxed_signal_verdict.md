# v3.21.11 — AEC3 `coarse_filter_converged_relaxed` signal parity (#1d) verdict (CLOSED)

**Date**: 2026-05-22
**Final verdict**: **NO-OP on this cohort. No consumer in current pipeline. Close as audit-only signal addition, default OFF preserved.**

## Implementation summary

- Flag: `coarse_filter_converged_relaxed_enabled: bool = False` ([config.py](AEC/python/modules/config.py))
- Env: `AEC_COARSE_RELAXED=1`
- Code: signal computed inside `_aec3_post` alongside existing strict `_coarse_conv` ([orchestrator.py](AEC/python/modules/orchestrator.py)). When flag ON: populates `self._coarse_relaxed_trace` counter dict + `self._diag['coarse_conv_relaxed']`. No bridge / state / RES wiring.
- Threshold (matches AEC3 `subtractor_output_analyzer.cc:50-51`):
  - Strict (existing): `e2_coarse < 0.05·y2 AND y2 > (50/32768)²·hop ≈ 3.73e-4`
  - Relaxed (new): `e2_coarse < 0.3·y2 AND y2 > (20/32768)²·hop ≈ 5.96e-5`

## AEC3 consumer audit

AEC3 publishes both signals via `SubtractorOutputAnalyzer::Update` (`subtractor_output_analyzer.cc:48-59`):

- `any_filter_converged` = `refined_filter_converged OR coarse_filter_converged_strict` → consumed by `convergence_seen_` latch + `FilteringQualityAnalyzer.Update()` + ERLE/ERL estimators (`aec_state.cc:193, 277, 415, 433`).
- `any_coarse_filter_converged` = `coarse_filter_converged_relaxed` → consumed **only** by `TransparentModeImpl::Update` HMM observation (`transparent_mode.cc:53, 100`).

The HMM uses the relaxed signal as the binary observation in its 2-state classifier (normal vs transparent). A converged-filter observation under active render lowers the probability of "transparent" (headset / no-echo) state; a non-converged observation raises it.

**Legacy variant** (`LegacyTransparentModeImpl::Update`, `transparent_mode.cc:147-150`) explicitly ignores `any_coarse_filter_converged`:

```cpp
void Update(int filter_delay_blocks,
            bool any_filter_consistent,
            bool any_filter_converged,
            bool /* any_coarse_filter_converged */,   // ← ignored
            bool all_filters_diverged,
            ...)
```

## Our pipeline state

- Our [`TransparentMode`](AEC/python/modules/state/transparent_mode.py) is the **Legacy** variant. Its `update()` signature does not even accept the relaxed signal:
  ```python
  def update(self, *, filter_delay_blocks, any_filter_consistent,
             any_filter_converged, all_filters_diverged,
             active_render, saturated_capture) -> None:
  ```
- `transparent_mode_enabled` defaults `False` (v3.21.6 Sprint P2 retained as audit substrate). With flag OFF, AecState never instantiates the module → `update()` is never called.
- HMM variant **not ported**. No other consumer of `any_coarse_filter_converged` exists in the codebase.

Conclusion: the relaxed signal has **zero downstream consumers in the current pipeline**. Even setting `transparent_mode_enabled = True` would not activate the relaxed-signal path because our Legacy port doesn't read it.

## Byte-equal + fire-rate trace evidence

Per-case full-file probe (default config, flag OFF vs ON):

| case | hops | strict fires | relaxed fires | relaxed-only | byte-equal |
|---|---:|---:|---:|---:|---|
| XRTnTUjU (stress) | 3486 | 0 | 32 | 32 (0.9%) | ✓ |
| nVUnxqHLr (stress) | 4297 | 20 | 236 | 216 (5.0%) | ✓ |
| MYrVxVEM (stress) | 3946 | 89 | 513 | 424 (10.7%) | ✓ |
| xFk7igec (stress) | 3677 | 14 | 360 | 346 (9.4%) | ✓ |
| jtYTdZm3 (normal DT) | 3675 | 229 | 915 | 686 (18.7%) | ✓ |
| 9xjhi (FS converging) | 2187 | 1 | 380 | 379 (17.3%) | ✓ |
| qNvSMyU (FS converging) | 2685 | 0 | 5 | 5 (0.2%) | ✓ |

Observations:

1. **Relaxed signal activates substantially more than strict** on every case — confirming AEC3's looser thresholds work as intended.
2. **`strict_only_fire = 0` across all cases** — confirms the superset relation `relaxed ⊇ strict` holds (lower threshold AND lower y² floor both relax independently, so any strict pass also passes relaxed).
3. **Stress cases (XRTnTUjU, nVUnxqHLr, MYrVxVEM, xFk7igec)** activate the relaxed signal 5-25× more than strict, but absolute fire rate is still low (0.9-10.7%) — shadow NLMS doesn't converge cleanly on no-clean-convergence stress.
4. **9xjhi (FS converging)** is the biggest divergence (strict 0.05% vs relaxed 17.3%) — relaxed catches "good but not perfect" coarse convergence on a clean FS case, exactly the regime AEC3 designed it for.
5. **Byte-equal preserved** on all 7 cases — trace counters add no algorithmic side-effect.

## Hypothetical HMM consequence (NOT to be implemented in v3.21.x)

If we ported the HMM TransparentMode and wired the relaxed signal:

- On 9xjhi FS converging: HMM would observe `any_coarse_filter_converged = True` 17.3 % of frames → lower probability of transparent state → stay in normal mode (correct, echo present).
- On XRTnTUjU stress: HMM would observe `any_coarse_filter_converged = True` only 0.9 % of frames → high probability of transparent state → could enter transparent mode → **suppress echo cancellation entirely**.

The XRTnTUjU scenario is exactly the **opposite** of what we want — we already have heavy echo, and incorrectly entering transparent mode would amplify the regression, not relieve it. HMM TransparentMode is designed for headset / no-echo detection, NOT no-clean-convergence stress. Porting it would be **counter-productive** on the v3.21.7 Cat C stress cluster.

This is consistent with AEC3's design intent: HMM transparent mode handles cases where echo is genuinely absent, not where the filter can't converge despite echo presence.

## Conclusion

`#1d coarse_filter_converged_relaxed` is mathematically a real AEC3 signal. The fix is correct, the trace is informative, and `relaxed ⊇ strict` is confirmed structurally. **But it has zero practical effect on our pipeline** because:

- Only AEC3 consumer is HMM `TransparentModeImpl` (we don't have it)
- Our `LegacyTransparentModeImpl` port explicitly ignores the relaxed signal
- Production has `transparent_mode_enabled = False` anyway
- Hypothetical HMM wiring would HARM the v3.21.7 stress cluster, not help it

v3.21.11 does NOT address the DT stress cluster identified by v3.21.7 Cat C. It is an **audit-only signal addition** — code computes and traces the AEC3 relaxed signal correctly, but no production behaviour change.

This is the third v3.21.x no-op closure with a different root cause:

| Patch | Closure root cause |
|---|---|
| v3.21.9 (#1c) | Threshold-comparison: TD vs Parseval produces same predicate outcome |
| v3.21.10 (#3a) | EMA dilution + overhang gate: AEC3 trigger never fires on our signal regime |
| v3.21.11 (#1d) | **No consumer**: signal exists in AEC3 only to feed HMM TransparentMode, which we don't have |

## Decision

- Close v3.21.11 as audit-only signal addition; default OFF preserved
- Code retained as parity-aligned substrate (cheap to keep; activates if/when HMM TransparentMode is ported)
- Do NOT promote to 800-case (no value to measure — byte-equal on all representative cases)
- Do NOT enable by default (zero benefit until consumer wired)
- Do NOT port HMM TransparentMode in v3.21.x scope — that is "new suppression behavior" per user scope rule, and the trace evidence shows it would harm stress cases anyway

## Remaining v3.21.x parity targets

| Target | Status | Expected effect |
|---|---|---|
| #1c coarse e2/y2 TD parity | **CLOSED no-op (v3.21.9)** | Threshold-bound: both formulas produce same predicate outcome |
| #2a UseRefinedOutput | **CLOSED field-trial substrate (v3.21.8)** | Cat C stress; AEC3 default OFF in production |
| #3a FilterMisadjustment direction | **CLOSED no-op (v3.21.10)** | EMA dilution caps `inv_max` < 1 vs threshold 10 |
| #1d coarse_filter_converged_relaxed | **CLOSED no-op (v3.21.11, THIS DOC)** | No consumer; HMM TM not ported and would harm stress |
| #1e `all_filters_diverged` signal | Pending | Last remaining target. Likely behaves like #1d (signal addition; consumer wiring depends on whether our PathChangeRegimeHandler / TransparentMode have integration points). |

## Forbidden actions (still in force)

- Do NOT port HMM `TransparentModeImpl` in v3.21.x (new suppression behavior)
- Do NOT enable `transparent_mode_enabled` by default
- Do NOT combine with UseRefinedOutput / partition_summed_x2 / coarse_e2_td_parity / misadj_parity / shadow-converged precondition / V2 gate-3 / convergence-counter variants
- Do NOT lower the AEC3 thresholds (20² / 0.3) — they are AEC3 design choices
- Do NOT delete research branches (Cat C preserved evidence)
- Do NOT run /simplify or branch cleanup
