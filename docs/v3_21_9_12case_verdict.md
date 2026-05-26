# v3.21.9 — AEC3 SubtractorOutput::ComputeMetrics coarse-e² time-domain window parity 12-case verdict (CLOSED)

**Date**: 2026-05-22

**Final verdict**: **NO-OP on this cohort. Close v3.21.9 as semantic-only parity, default OFF preserved.**

## Implementation summary

- Flag: `use_coarse_e2_time_domain_parity: bool = False` ([config.py](AEC/python/modules/config.py))
- Env: `AEC_COARSE_E2_TD_PARITY=1`
- Code: `_aec3_post` convergence check ([orchestrator.py](AEC/python/modules/orchestrator.py)) — when flag ON, `_e2_coarse` computed as `sum(self._last_shadow_output_time²)` over hop, matching AEC3 `subtractor_output.cc:38-48` block sum-of-squares. When flag OFF, preserves legacy Parseval-over-fft_size reconstruction.

## Byte-equal verification

- Default OFF (`use_coarse_e2_time_domain_parity = False`) vs prior 12-case `A_baseline`: **24/24 file MD5 PASS** (both `*_ours.wav` and `*_ours_nores.wav` identical).
- Flag ON (full bench): **0/24 file diffs** vs A_baseline. Byte-equal across all 12 cases despite the formula change being mathematically distinct.

## Trace evidence — why byte-equal even with flag ON

The Parseval-over-fft_size and time-domain-over-hop formulas give different `_e2_coarse` magnitudes (~3.2× ratio for stationary signals). However, the `_coarse_conv` predicate is a **threshold comparison** (`_e2_coarse < 0.05 × _y2_time AND _y2_time > _y2_threshold`). In practice this never crosses the threshold under either formula on the 12-case cohort.

Per-case fire rates (flag OFF vs ON):

| case | frames | refined_conv (off / on) | coarse_conv (off / on) | usable_linear (off / on) |
|---|---:|---:|---:|---:|
| FS-9xjhi | 2188 | 814 / 814 | 1 / 1 | 2069 / 2069 |
| FS-mv-0I0XMl3M | 2416 | 0 / 0 | 0 / 0 | 2168 / 2168 |
| STRESS-XRTnTUjU | 3487 | 57 / 57 | **0 / 0** | 1416 / 1416 |

Mechanism:

1. **Stress cases**: shadow NLMS is un-converged → `e_coarse` is large under BOTH formulas → `_e2_coarse < 0.05 × _y2_time` fails under BOTH → `_coarse_conv = False` regardless of formula → no behavioral change.
2. **Normal cases**: `refined_conv` fires first and locks `convergence_seen` latch → downstream behavior is determined by refined-side state, not coarse-side. coarse-side formula doesn't matter.
3. **Edge cases**: where shadow IS clean enough that TD < 0.05 × y2 but Parseval > 0.05 × y2 would exist, but they are absent from the 12-case cohort.

## Conclusion

`#1c coarse e2/y2 window mismatch` is mathematically a real AEC3 parity gap. The fix is correct. **But it has zero practical effect on the 12-case cohort** because:

- Stress cases never trigger `coarse_conv` under either formula (shadow un-converged)
- Normal cases rely on `refined_conv` first-fire → latch already triggered → coarse-side gate doesn't matter

v3.21.9 does NOT address the DT stress cluster identified by v3.21.7 Cat C verdict. It is a **semantic-only parity refresh** — code aligned with AEC3 `SubtractorOutput::ComputeMetrics`, but no measurable production behavior change.

## Decision

- Close v3.21.9 as no-op semantic cleanup, default OFF preserved
- Code retained as parity-aligned substrate (cheap to keep, may matter on future cohorts where shadow converges differently)
- Do NOT promote to 800-case (no value to measure)
- Do NOT enable by default (no harm but no benefit either; default OFF lets future research toggle for ablation)

## Remaining v3.21.x parity targets

| Target | Status | Expected effect |
|---|---|---|
| #1c coarse e2/y2 TD parity | **CLOSED no-op (v3.21.9)** | None (threshold-bound; shadow under-converged or refined-locked) |
| #1d `coarse_filter_converged_relaxed` signal | Pending | Adds downstream consumer signal. Unknown effect — depends on which AecState transitions read it. |
| #1e `all_filters_diverged` signal | Pending | Adds echo-path-change detection trigger. Unknown effect — depends on PathChangeRegimeHandler integration. |
| #3a FilterMisadjustment direction inversion | Pending (v3.23 G.0 A.1 closed INFORMATIVE_NULL on different cohort) | Re-evaluate on stress cohort. Estimator fires 0/20188 on G.0 cohort; may fire here. |

Per the v3.21.x rule (parity only), all four remain candidates. Pivot to v3.22 NOT yet justified — three untested parity targets remain.

## Forbidden actions (still in force)

- Do NOT combine v3.21.9 with UseRefinedOutput / partition_summed_x2 / V2 / shadow-converged precondition
- Do NOT delete research branches (Cat C preserved evidence)
- Do NOT mark v3.21.9 as production-default change (default OFF stays)
