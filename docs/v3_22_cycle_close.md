# v3.22 — Cycle Close: No-Shippable-Change Optimization Audit

**Date**: 2026-05-21
**Branch**: `feature/v3_22_optimization` (NOT merged to main)
**Status**: **CLOSED no-shippable-change**. v3.21.6 remains the current shipping algorithm baseline. v3.22 substrate stays on feature branch as research material; **no version bump, no tag, no merge**.

## TL;DR

| Sprint | Outcome | Reason |
|---|---|---|
| **E** NE-presence augmentation (PRIMARY) | ✅ CLOSED no-leverage | Gain-policy-level proxy can't substitute R²-level zeroing |
| **G.0** unified triage trace | ✅ done | 6 trace fields added; cohort verdict for D/F/G.1/H.3 |
| **D** hybrid residual / nonlinear HF floor | ✅ CLOSED no-leverage | P1 already calibrated S²/X² in HF; median 25-200× above gate |
| **G.1 / H.3** ERLE clamp widening | ✅ CLOSED marginal-no-leverage | 1/4 FS cases; direction is FS-unfavorable |
| **F** reverb tail dead fallback | ✅ CLOSED no-leverage | Catalyst works but DT damage 61-69 cata frames untunable |
| **G.2 / G.3 / G.4 / G.5** | ⏳ deferred | Out of scope for lite triage; require deeper instrumentation |
| **I** config cleanup | ⏳ deferred | No flags to clean up given no ship; future cycle if any |

**Net algorithm ship**: nothing. v3.21.6 stays canonical.

## Substrate left in tree (dormant)

| Item | Type | Default | Reason kept |
|---|---|---|---|
| `e_stat_aware_ne_proxy_enabled` + `_threshold` | AecConfig | False / 0.10 | Sprint E research toggle |
| `reverb_tail_dead_fallback_enabled` + `_threshold_frames` + `_strength` | AecConfig | False / 50 / 0.25 | Sprint F research toggle |
| `transparent_mode_enabled` | AecConfig | False | v3.21.6 P2 research toggle (carry-over) |
| 6 G.0 trace fields | trace_hf_chain | default-OFF | Future-trace substrate (D/F/G.1/H.3 gate evidence) |
| Env hooks `AEC_STAT_NE_PROXY`, `AEC_REVERB_DEAD_*` | run_one_case.py + eval_aec_challenge.py | unset | A/B utility |

All default-OFF; byte-equal verified at every closure step (single-case pcb1Nh md5 `6143992ee53a9866f6fda068137603b4` invariant).

## Why every candidate failed

The v3.22 closures share a single structural root cause: **every candidate operates downstream of, or as a substitute for, the load-bearing v3.21.6 P4 stationarity zeroing safety-net** — and none of those positions can reach the cohort-tail failure mode that the safety-net is currently masking.

### The cohort-tail failure mode (recap from v3.21.6 P4 verdict)

`_DominantNearendDetector` mis-classifies NE under stationary-far conditions. On the cohort tail (xQEUtY2 / wVYSGV / WcK0OrF), `is_nearend_state` rate stays at 0.4-15.3% when NE-speech is actually present 40-60% of the time. The stationarity zeroing (`R²/R²_unb → 0` on stationary-far bins) was retained default-True as a load-bearing safety-net that compensates for this detector failure on stationary-far frames specifically.

### Why each v3.22 candidate failed

- **Sprint E** tried to **substitute** the zeroing with a SuppressionGain-level proxy (augment NE state to include stationary-far). Proxy fires on the right frames (94-100% match per E.0 cross-tab) but operates at ENR/EMR tuning level, downstream of the R² values. The damage R² inflation causes can't be undone by a gain-policy decision. Catastrophic reduction 3.5-17% vs ≥ 50% target.

- **Sprint D** tried to **add a nonlinear HF floor** on the residual. G.0 trace showed s2_to_x2_ratio_hf is already 25-200× above the under-estimation gate — P1's FilterAnalyzer port already calibrated the HF residual. A nonlinear floor would over-suppress where the linear stage is now working correctly.

- **Sprint G.1 / H.3** tried to **widen the HF ERLE clamp** to let linear effectiveness show. Only 1/4 FS converged cases passes the 50% at-max gate; direction (less HF suppression) is FS-unfavorable on the same cohort where v3.21.6's leverage is FS-positive.

- **Sprint F** tried to **inject conservative tail mass** into R²/R²_unb when AEC3's reverb estimator can't update. Catalyst mechanism works (Δg100 = −0.16/−0.18 on tail-dead FS frames) but the render-power-scaled injection unavoidably adds R² on DT bins where NE-speech co-occurs with render. DT damage 61-69 catastrophic frames at strength=0.05 (5× reduced). No safe gating found: the companion safeguards (NE detector, `_filter_converged_enough`) both have known mis-classification patterns on this exact cohort.

### The unifying observation

Every shippable v3.22 candidate operates at the residual-echo / suppression-gain level. The actual leverage is upstream:

```
PBFDKF (filter convergence)
  └─> FilterAnalyzer (P1: convergence signal source)
        └─> _DominantNearendDetector (NE classification) ←—————————┐
              └─> stationarity zeroing (load-bearing safety-net)  │  cohort tail
                    └─> R²/R²_unb (residual echo PSDs)            │  failure mode
                          └─> SuppressionGain (gain rule)         │  originates here
                                └─> output                        │
                                                                  │
v3.22 candidates all live below this line ─────────────────────────┘
```

Any candidate that produces leverage on the cohort-tail mis-classification region also produces damage on the cases where the safety-net already works correctly. The Pareto wall isn't in the residual layer; it's in the detector layer.

## Future direction (not v3.22 scope)

The next plan, **if any**, should target the upstream detector layer, NOT residual fallback / HF cap tuning. Specific design candidates:

1. **PBFDKF-aware NE detector** — replace `_DominantNearendDetector` with a Kalman-state-derived "no echo path" / "echo path present" classifier that doesn't depend on the ENR/SNR comparison the current detector uses (which fails on stationary-far). Could use innovation_var, kalman_p_trace, system_distance_proxy from the existing KalmanStateInterface (P55 design lock substrate).

2. **Stationary-far-specific NE detector** — add a sub-detector that fires when `stationary_block` is True, replacing the current detector's vote on those frames. Less invasive than full replacement; could ship as a hybrid.

3. **ScaleFilter / FilterMisadjustmentEstimator port** — these are AEC3 companion mechanisms our PBFDKF port skipped. Per v3.21.6 P4 verdict, they're hypothesized to be the missing piece that would make `_DominantNearendDetector` fire correctly on stationary-far conditions. Cost: medium-large port + bench cycle.

4. **PBFDKF-state-derived echo-confidence at R² zeroing decision** — replace the current `band_stationary_mask()`-driven zeroing with a Kalman-state-derived per-bin confidence. The non-zero zeroing predicate would no longer depend on the AEC3 stationarity estimator (which is far-end-only); it'd reflect "the PBFDKF Kalman filter believes this bin has converged echo path AND the residual energy is consistent with linear cancellation". Different from candidate 1 because the substitution is at the R² zeroing site, not at the detector itself.

All four candidates are **upstream** of v3.22's scope. None are residual-layer.

### What NOT to try in the next cycle

- Residual fallback variants (mass injection into R² / R²_unb) — Sprint F demonstrated this can't be safely gated
- NE proxy variants at SuppressionGain gain-policy level — Sprint E demonstrated this can't substitute R²-level zeroing
- HF cap / ERLE clamp tuning — G.0 closed (direction unfavorable; signal weak)
- Nonlinear HF floor / hybrid residual — G.0 closed (P1 already calibrated)

These are exhausted directions. Any future v3.23+ cycle that proposes them needs explicit "differs from v3.22 closures because [X]" rationale; default is to reject.

## What the v3.22 cycle did produce

- **6 new trace_hf_chain fields** (D/F/G.1/H.3 gate evidence) — instrumentation substrate for any future cycle
- **3 research-toggle flags** (E + F substrate; mirrors P2/P4 precedent) — dormant code paths kept for A/B if/when a companion mechanism makes them safe
- **5 verdict docs** (E close + G.0 triage + F close + this cycle close + E.0 cross-tab evidence) — cohort-evidence trail for next cycle to inherit
- **2 cohort renders saved** (`/tmp/e0_cohort/` + `/tmp/g0_cohort/` + `/tmp/f_cohort/`) — if any future trace cycle wants the v3.21.6-baseline reference data, it's there

## Operational state

- **main branch**: at v3.21.6 (merge commit `f8aa50c`); shipping baseline; `__version__ = 3.21.6`; tag `v3.21.6` pushed
- **feature/v3_22_optimization branch**: at `873afed` (Sprint F close); 5 commits ahead of main containing the v3.22 substrate
- **No merge to main**: explicit per user directive — the substrate is research-grade not ship-grade
- **No version bump**: __version__ stays at 3.21.6
- **No v3.22 tag**: per close-as-audit decision

If a future cycle revisits any v3.22 dormant flag and demonstrates ship-grade Pareto-positive bench result, the merge / version-bump / tag steps happen THEN. Today's decision is "v3.22 didn't produce shippable algorithm change; document and move on".

## Files

- This cycle close: `docs/v3_22_cycle_close.md`
- v3.22 verdicts: `docs/v3_22_e0_hf_cap_ne_cross_tab_verdict.md`, `docs/v3_22_e_ne_proxy_verdict.md`, `docs/v3_22_g0_triage_verdict.md`, `docs/v3_22_f_reverb_dead_fallback_verdict.md`
- Plan: `~/.claude/plans/se-aec-aec-main-hazy-lynx.md` (this cycle's plan; will be archived once next-cycle plan opens)
- v3.21.6 verdicts (still authoritative for current shipping baseline): `docs/v3_21_6_p{1,2,3,4}_*_verdict.md`, `docs/v3_21_6_cycle_close.md`
- v3.21.6 bench snapshot (current production reference): `docs/bench/v3_21_6_baseline/`
