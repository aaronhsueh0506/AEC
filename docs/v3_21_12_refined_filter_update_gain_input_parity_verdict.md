# v3.21.12 — RefinedFilterUpdateGain input-parity verdict (CLOSED)

**Date**: 2026-05-22
**Final verdict**: **REJECTED. Filter input parity is NOT the right intervention. v3.21.7 Cat C regression is downstream of the filter (latch / usable_linear / RES). Close C as no-op; close D as WORSE-than-B on Cat C stress. v3.21.12 closes without ship candidate.**

## EXACT / APPROX / WRONG verdict on AEC3 inputs

Per AEC3 `refined_filter_update_gain.cc:103-107`
`mu[k] = H_error[k] / (0.5·H_error[k]·X²[k] + n_part·E²_refined[k])`:

| Input | AEC3 source | Our default (v3.21.6) | Verdict |
|---|---|---|---|
| `X²[k]` | `render_buffer.SpectralSum` = Σ_p `|X_buf[p][k]|²` | latest-partition only | **WRONG (default)** — under-states render energy. Gated by `use_partition_summed_x2_for_h_error_gain` (v3.21.7) |
| `E²_refined[k]` | current-block `SubtractorOutput.E2_refined` = instantaneous per-bin | `self._error_psd` = 0.95 EMA, ~200 ms lag | **WRONG (default)** — lags transients. Gated by NEW `use_current_e2_refined_in_h_error_denominator` (v3.21.12, this round) |
| `n_partitions` | `SizePartitions()` | `self.n_partitions` | EXACT |
| Noise gate | `X² >= noise_gate` | `X² >= NOISE_GATE_POWER_FLOAT` | EXACT |
| `H_error` decay | per-bin `H_error -= 0.5·mu·X²·H_error` | per-bin | EXACT |
| `H_error` refresh selector | per-bin `E²_refined ≤ E²_coarse` | per-bin (flag-ON) | EXACT |

Two structural input mismatches confirmed. Comment in `filters.py:609-611` ("AEC3 also uses a smoothed estimate") was wrong and has been corrected.

## A/B/C/D 12-case AECMOS — bucket Δ vs A baseline

| Bucket | B (partition only) | C (raw E2 only) | D (both ON) |
|---|---:|---:|---:|
| FS_static Δecho | +0.055 | -0.016 | -0.022 |
| FS_static Δdeg | -0.000 | +0.000 | -0.000 |
| FS_movement Δecho | +0.152 | +0.004 | +0.121 |
| FS_movement Δdeg | +0.001 | +0.001 | +0.001 |
| **DT_static Δecho** | **+0.215** | -0.091 | **+0.175** |
| **DT_static Δdeg** | **-0.938** | -0.061 | **-1.034** |
| DT_movement Δecho | -0.213 | -0.141 | -0.255 |
| DT_movement Δdeg | +0.363 | +0.273 | +0.626 |
| NE | ±0 | ±0 | ±0 |

**Critical**: D's DT_static Δdeg = **-1.034**, *worse* than B's -0.938. The "better filter regime" hypothesis failed.

## Per-case DT_static (4 cases — 3 stress + 1 normal)

| case | A_deg | B_deg | C_deg | D_deg | D-A | D-B |
|---|---:|---:|---:|---:|---:|---:|
| MYrVxVEM (stress) | 2.869 | 1.523 | 2.845 | 1.520 | **-1.349** | -0.003 |
| XRTnTUjU (stress) | 3.929 | 2.676 | 3.570 | 2.674 | **-1.255** | -0.002 |
| jtYTdZm3 (normal) | 2.347 | 2.670 | 2.355 | 2.343 | -0.004 | **-0.328** |
| nVUnxqHLr (stress, worst) | 3.223 | 1.748 | 3.356 | 1.694 | **-1.529** | -0.054 |

D's stress regression is essentially equivalent to B's (within ±0.05 deg per case). But D **introduces a new -0.328 deg regression on the normal DT case** (jtYTdZm3) which B doesn't have. This makes D **strictly worse than B** on this 12-case cohort:

- D matches B on the 3 worst stress cases (within ±0.1 deg)
- D loses to B on jtYTdZm3 by 0.328 deg
- B has no normal-DT regression; D does

## Per-case FS_static + DT_movement (selected)

| case | A | B | C | D | D-B |
|---|---:|---:|---:|---:|---:|
| 9xjhi (FS, nores-artifact target) echo | 2.194 | 2.100 | 2.167 | 1.875 | -0.225 |
| qNvSMyU (FS) echo | 3.673 | 3.568 | 3.531 | 3.576 | +0.007 |
| xQEUtY2 (FS) echo | 3.346 | 3.710 | 3.467 | 3.697 | -0.013 |
| wVYSGV (DT_mvmt) deg | 2.203 | 3.192 | 3.111 | 3.826 | +0.634 |

D vs B summary: ~similar on FS, better on DT_movement (wVYSGV +0.634), but the DT_static regression dominates the verdict.

## Mechanism — why D's "cleaner residual" hurts more

Hypothesis from trace phase: D's lower refined-divergence rate (10.8 % strong on XRTnTUjU vs B's 12.1 %) should reduce the cond2-fire / DT-deg damage.

**AECMOS falsifies this.** D's lower divergence translates to MORE `convergence_seen` latching events, not fewer. The v3.21.7 Cat C mechanism is:

```
cleaner residual → refined_conv fires → convergence_seen latches True
                                            ↓
                                       usable_linear stays True for rest of utterance
                                            ↓
                                       SuppressionGain trusts S²_linear path
                                            ↓
                                       DT speech treated as residual echo
                                            ↓
                                       over-suppressed (DT_deg regress)
```

D produces a cleaner residual than B → fires `refined_conv` slightly more / slightly earlier → latches harder → DT damage extends to MORE frames per utterance → marginally worse DT_static Δdeg.

This is exactly the failure mode predicted by [project_usable_linear_gate3_latch_bug]. The root cause is **the latch + downstream RES**, NOT the filter input formula.

## Trace metric vs AECMOS — trace gate was insufficient

The trace gate (`reduce stress refined_divergence`) was met by D (XRTnTUjU 12.1% → 10.8% strong). The plan's tentative interpretation was that this should reduce downstream DT damage. AECMOS proves the opposite:

| Metric | B | D | Better metric value? | AECMOS outcome |
|---|---:|---:|---|---|
| XRTnTUjU div_strong% | 12.1 % | **10.8 %** | D | D AECMOS deg = B (no help) |
| nVUnxqHLr div_strong% | 0.9 % | **0.6 %** | D | D AECMOS deg ≈ B |
| Mean DT_static div_strong% | 3.40 % | **3.00 %** | D | D AECMOS Δdeg -1.034 (worse than B -0.938) |

The trace metric predicts the wrong direction because lower divergence (= more convergence) IS the proximate cause of the latch failure. The plan's trace-gate hypothesis (filter quality → state quality) is **inverted** for the Cat C mechanism.

## Final disposition

| Variant | Disposition | Reason |
|---|---|---|
| A | Production default — keep | v3.21.6 baseline |
| B (`use_partition_summed_x2_for_h_error_gain`) | **Status quo: paused Cat C BLOCKED-STRESS** (v3.21.7 unchanged) | Same as prior verdict; no new evidence to ship |
| **C (`use_current_e2_refined_in_h_error_denominator`)** | **CLOSE no-op (default OFF)** | Trace showed divergence worsens but AECMOS DT_static Δdeg -0.061 — no perceptual damage, no benefit either |
| **D (`B + C` combined)** | **CLOSE REJECTED — worse than B** | Marginal Cat C regression (-1.034 vs -0.938) PLUS new normal DT regression (jtYTdZm3 -0.328) |

Code retained for both flags as default-OFF substrate. The flag implementations are correct AEC3 parity — they just don't fix the actual downstream failure mode.

## Recommendation per plan Step 6

Per plan v3.21.12 Step 6 verdict gate:

> | C/D regresses nores or worsens stress | **CLOSE REJECTED** |

D worsens stress (DT_static Δdeg) compared to B. Decision: **CLOSE REJECTED**.

> | C improves nores but worsens stress (or vice versa) | **PAUSED SUBSTRATE** |

C is essentially neutral on AECMOS (small -0.061 DT_static Δdeg). Trace showed divergence worsening, but no perceptual translation. Decision: **CLOSE no-op-neutral**, retain as substrate.

## Implication for v3.21.x cycle

Pattern so far:

| Patch | Closure cause |
|---|---|
| v3.21.7 #2a partition_summed_x2 | Cat C BLOCKED-STRESS (paused) |
| v3.21.8 UseRefinedOutput | AEC3 field-trial substrate (closed) |
| v3.21.9 #1c coarse e2/y2 TD parity | No-op (threshold-comparison invariant) |
| v3.21.10 #3a FilterMisadjustment direction | No-op (EMA dilution prevents trigger) |
| v3.21.11 #1d coarse_relaxed signal | No-op (no consumer in our pipeline) |
| **v3.21.12 #refined-gain inputs** | **REJECTED (D worse than B; C neutral)** |

**Aggregate**: 0 ship candidates from 6 v3.21.x AEC3-parity attempts so far.

The structural finding is that the v3.21.7 Cat C stress regression is **downstream of the refined filter**:
- Filter improvements (partition_summed_x2, raw E²_refined denominator) increase residual cleanliness
- Cleaner residual triggers `convergence_seen` latch faster / longer
- Latch keeps `usable_linear = True` over DT regions in no-clean-convergence stress
- SuppressionGain over-suppresses DT speech

The **only AEC3 parity target remaining in v3.21.x scope** is #1e `all_filters_diverged` (echo-path-change detection signal). It is unlikely to break the pattern: it's a downstream-consumer signal addition, similar to #1d.

**After #1e**, the v3.21.x cycle exhausts AEC3 parity targets. Per the version-line rule (paused, not removed) the next legitimate move is either:
1. **v3.22 latch/state-layer redesign** (PBFDKF-specific, beyond AEC3) — the structural root cause per the trace/AECMOS evidence. Explicitly OUT of v3.21.x scope.
2. **Audit downstream RES** for AEC3 semantic deltas (`R²` smoothing, `S²_linear` selection, AecState NE detector path) — these could be in-scope v3.21.x if AEC3-parity mismatches found.

Neither is to be opened in this round.

## Forbidden actions (still in force)

- Do NOT enable C / D by default in production
- Do NOT modify AEC3 thresholds (these flags' values match AEC3 verbatim)
- Do NOT delete C / D code (keep as default-OFF parity substrate)
- Do NOT pivot to v3.22 yet — #1e remains untested
- Do NOT add gate-3 counter / FA-AND / shadow-converged precondition in v3.21.x
- Do NOT run 800-case on D — 12-case AECMOS already disqualifies
- Do NOT run /simplify or branch cleanup before #1e closes

## Open decision after #1e closes

If #1e is no-op (likely), v3.21.x AEC3 parity is exhausted. User decision needed:
- Open v3.22 latch redesign (out of AEC3 parity scope)?
- Audit downstream RES for AEC3 semantic mismatches (potentially in scope)?
- Stay at v3.21.6, mark all v3.21.x candidates as substrate-only?
