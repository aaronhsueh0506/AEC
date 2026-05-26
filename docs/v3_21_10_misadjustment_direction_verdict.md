# v3.21.10 — AEC3 FilterMisadjustment direction parity (#3a) verdict (CLOSED)

**Date**: 2026-05-22
**Final verdict**: **NO-OP on this cohort. Close v3.21.10 as semantic-only parity, default OFF preserved.**

## Implementation summary

- Flags: `aec3_misadj_trace_enabled: bool = False` + `use_aec3_filter_misadjustment_parity: bool = False` ([config.py](AEC/python/modules/config.py))
- Env: `AEC_AEC3_MISADJ_TRACE=1` / `AEC_AEC3_MISADJ_PARITY=1` ([eval_aec_challenge.py](AEC/python/eval_aec_challenge.py))
- Code: new `_fire_aec3_misadj_scale` + branched `_update_misadjustment_estimator` / `_check_and_apply_misadjustment_scale` ([orchestrator.py](AEC/python/modules/orchestrator.py)). Mirrors AEC3 `Subtractor::FilterMisadjustmentEstimator` (subtractor.cc:336-358 + subtractor.h:99-128): accumulates `e2_refined` + `y2` over `n_hops=2` (20 ms ≈ AEC3 4-block 16 ms wall-clock), asymmetric EMA on `inv_misadjustment_ = e2/y2`, IsAdjustmentNeeded `> 10`, scale `2/sqrt(inv)` (SHRINK W). Legacy path (echo_psd / far_psd / ERL ratio) preserved when parity flag OFF.

## Direction parity confirmation

The audit ([docs/v3_21_10_misadjustment_direction_audit.md](AEC/docs/v3_21_10_misadjustment_direction_audit.md)) identified opposite failure-mode coverage:

| Aspect | AEC3 | Ours (legacy) |
|---|---|---|
| Observable | `e2_refined / y2` (post-filter error vs mic) | `echo_psd / (far_psd × ERL)` (echo-estimate vs ERL-predicted) |
| Fire condition | `inv > 10` (error >> mic → DIVERGENCE) | `smoothed < 0.5` (echo_est < ERL pred → UNDER-MODELLING) |
| Scale | `2 / sqrt(inv)` → < 1 → SHRINK W | `1 / smoothed` → > 1 → GROW W |

The parity port now provides the AEC3 over-adaptation detection path our legacy didn't cover.

## Byte-equal verification

- Default OFF: byte-equal to v3.21.6 baseline (legacy code path untouched).
- TRACE-ON (no parity): 6/6 case MD5 PASS vs default OFF — verifies trace counters add zero side-effect.
- PARITY-ON (full 12-case representative + partition_summed_x2 OFF): **6/6 byte-equal MD5 PASS vs default OFF** (XRTnTUjU/nVUnxqHLr/MYrVxVEM/xFk7igec/jtYTdZm3/9xjhi). Mechanism is correct AEC3 port; behaviour unchanged because the AEC3 trigger never fires.

## Trace evidence — why AEC3 IsAdjustmentNeeded never fires

Per-case probe (partition_summed_x2 ON to maximise stress, full files):

| case | hops | mic_max | e2_p99 | e2/y2 p99 | overhang fires | inv_max | aec3_would_fire |
|---|---:|---:|---:|---:|---:|---:|---:|
| XRTnTUjU (stress) | 3486 | 0.901 | 3.03 | **45.6** | 0 | 0.000 | 0 |
| nVUnxqHLr (stress, worst regression) | 4297 | 0.572 | 2.00 | 2.0 | 0 | 0.000 | 0 |
| MYrVxVEM (stress) | 3946 | 0.734 | 0.43 | 1.4 | 0 | 0.000 | 0 |
| xFk7igec (stress) | 3677 | 1.000 | 3.78 | 1.5 | 15 | 0.516 | 0 |
| jtYTdZm3 (normal DT) | 3675 | 1.000 | 3.45 | 1.1 | 4 | 0.345 | 0 |
| 9xjhi (normal FS) | 2187 | 0.722 | 31.4 | 1.2 | **1122 (51 %)** | **0.964** | 0 |

Two cascading guards keep `inv` < 10 across the cohort:

1. **Overhang gate**: AEC3's `e2 > n × 7500² × kBlockSize` triggers only on near-full-scale sustained error (amplitude > 0.229 in float [-1,1] = -12.8 dBFS RMS over the accumulation window). On 9xjhi (FS) this fires 51 % of frames; on most stress DT cases it never fires at all because the residual amplitude is far below this floor. Without overhang, AEC3's EMA only flows downward (`update < inv`), and `inv` starts at 0 — so `inv` is permanently 0 when overhang stays silent.

2. **Asymmetric EMA dilution**: even on 9xjhi (51 % overhang fire), `inv` converges toward the median `e2/y2 ≈ 1.0` (= 0 dB ERLE, i.e. error matches mic energy on a per-window basis). The 99-th percentile spikes to 45 on XRTnTUjU are diluted by the 4-block accumulation and the 0.1× EMA rate. Sustained `inv > 10` would require ERLE ≤ -10 dB (filter actively amplifying echo) across multiple consecutive 16 ms windows — that is filter blowup, not local divergence.

XRTnTUjU has 98 isolated hops (2.8 %) where instantaneous `e²/y² > 10`, but the EMA dilution prevents these from registering as a sustained threshold crossing.

## Conclusion

`#3a FilterMisadjustment direction inversion` is now correct in code: the AEC3-parity path provides the over-adaptation detection our legacy formula structurally cannot. The fix is mathematically right and matches AEC3 verbatim.

**However**, AEC3 IsAdjustmentNeeded is by-design a rare emergency-shrink trigger, not an ongoing correction loop:

- It only fires when the filter is in catastrophic blowup (sustained ERLE ≤ -10 dB over 16 ms)
- On the v3.21.7 Cat C stress cohort, divergence is LOCAL (sparse hops with `e²/y²` spikes) but the EMA-smoothed `inv` stays under 1.0
- AEC3 is conservative on purpose: false-positive shrinks would destroy convergence, so the trigger is calibrated for genuine filter failure only

`v3.21.10` does NOT address the DT stress cluster identified by v3.21.7 Cat C. It is a **semantic-only parity refresh** — code aligned with AEC3 `Subtractor::FilterMisadjustmentEstimator`, but no measurable production behaviour change on this cohort.

This is the same outcome class as v3.21.9 (#1c coarse e2/y2 TD window parity): the parity gap is real but threshold-bound to a state our signals don't reach. Aggregate v3.21.x parity-only outcome so far: 1 BLOCKED-STRESS (#2a UseRefinedOutput, v3.21.7), 2 NO-OP semantic refreshes (#1c, #3a), 0 behavioural improvements.

## Decision

- Close v3.21.10 as no-op semantic cleanup, default OFF preserved
- Code retained as parity-aligned substrate (cheap to keep; activates on future cohorts where extreme blowup occurs)
- Do NOT promote to 800-case (no value to measure — byte-equal to OFF on 6/6 representative cases)
- Do NOT enable by default (no harm but no benefit either; default OFF lets future research toggle for ablation)
- Do NOT lower the AEC3 thresholds (7500 amplitude / `inv > 10`) — that would be divergence from AEC3, not parity, and the user's directive locks v3.21.x to parity-only

## Remaining v3.21.x parity targets

| Target | Status | Expected effect |
|---|---|---|
| #1c coarse e2/y2 TD parity | **CLOSED no-op (v3.21.9)** | None (threshold-bound; shadow under-converged or refined-locked) |
| #2a UseRefinedOutput | **CLOSED field-trial substrate (v3.21.8)** | Cat C stress; AEC3 default OFF in production |
| #3a FilterMisadjustment direction | **CLOSED no-op (v3.21.10, THIS DOC)** | None (overhang gate + EMA dilution prevent threshold crossing on cohort signal regime) |
| #1d `coarse_filter_converged_relaxed` signal | Pending | Adds downstream consumer signal. Unknown effect — depends on which AecState transitions read it. |
| #1e `all_filters_diverged` signal | Pending | Adds echo-path-change detection trigger. Unknown effect — depends on PathChangeRegimeHandler integration. |

Per the v3.21.x rule (parity only), #1d and #1e remain candidates. Pivot to v3.22 NOT yet justified — two untested parity targets remain. Reset of pattern is a useful evidence anchor — both remaining targets are downstream SIGNAL additions (not threshold-bound trigger ports), so they may behave structurally differently from the three closed.

## Forbidden actions (still in force)

- Do NOT combine v3.21.10 with UseRefinedOutput / partition_summed_x2 / coarse_e2_td_parity / shadow-converged precondition / V2 gate-3 / convergence-counter variants
- Do NOT delete research branches (Cat C preserved evidence)
- Do NOT mark v3.21.10 as production-default change (default OFF stays)
- Do NOT modify AEC3 threshold constants (7500 amplitude / `inv > 10`) — those are AEC3 design choices; tuning them would convert the patch into intentional divergence
- Do NOT run /simplify or branch cleanup
