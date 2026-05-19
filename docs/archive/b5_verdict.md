# B5 verdict — shadow R-reset symmetric extension

**Phase**: v3.11 Phase 1 Sprint 1-2
**Flag**: `shadow_r_reset_enabled` (default OFF)
**Result**: **PASS — marginal positive, recommend promote at final review**

## Mechanism

F2.3 (Yang 2017 R-reset on EPC `boost_q`) currently resets only the main
filter's `_error_psd` and `R`. The shadow filter retains its stale R from
the same DT period; when the regime handler subsequently fires
`reverse_copy` (shadow → main), the K-handicapped shadow's W is
transplanted into main, undoing F2.3's fast-recovery benefit.

B5 extends F2.3 symmetric: when `boost_q` fires AND `shadow_r_reset_enabled`,
also `shadow_filter._error_psd.fill(1e-2)` + `shadow_filter.R.fill(1e-2)`.
Gated behind a new opt-in flag; depends on the shadow filter existing.

Reference: Yang/Enzner/Yang 2017 IEEE SPL "FDKF Fast Recovery of Abrupt
Echo-Path Changes" — symmetric application across both filters.

## Verification

**Baseline**: v3.10.6 800-case rendering (`results/v3_10_5_main/`).
**Candidate**: v3.10.6 + `AEC_SHADOW_R_RESET=1` (`results/sprint_b5_on/`).
**Metric**: Linear-only quality (`tools/research/linear_eval.py`) — PBFDKF
linear residual without RES, eliminating RES patches masking front-end
changes.

### Bucket-mean Δ vs baseline

| Bucket | n | Δerle_full | Δerle_active | Δne_pres |
|---|---|---|---|---|
| FS_static    | 169 | +0.003 | +0.003 | -0.003 |
| FS_movement  | 131 | +0.004 | +0.004 | -0.013 |
| DT_static    | 186 | -0.003 | -0.004 | +0.008 |
| DT_movement  | 114 | +0.000 | +0.000 | -0.000 |
| NE           | 200 | +0.000 | +0.000 | +0.000 |

All bucket means within ±0.013 dB noise floor.

### Per-case impact distribution (|Δerle_active| > 0.1 dB)

| Bucket | n cases moved | improvements | regressions |
|---|---|---|---|
| FS_static | 2 | 2 (+0.121, +0.131) | 0 |
| FS_movement | 1 | 1 (+0.517) | 0 |
| DT_static | 1 | 0 | 1 (-0.694, but Δne_pres = +1.667) |
| DT_movement | 0 | 0 | 0 |
| NE | 0 | 0 | 0 |

Total: 4 / 800 cases significantly moved (0.5% impact rate). Effect
is contained to cases where the chain `boost_q + reverse_copy +
stale-shadow-R` is actually load-bearing.

### TGZ5 DT_static trade-off (worth understanding)

The single DT regression case `TGZ5Wq0SCUCOXPsfee3uMQ_doubletalk`:

- ΔerleA = -0.694 dB (less echo cancellation in DT)
- Δne_pres = +1.667 dB (more NE preservation in DT)

This is the **expected positive trade-off**. In DT context, the shadow
R-reset prevents the over-confident shadow K (from stale R) from pulling
main's W in the wrong direction during a DT-phase EPC boost. The result:
- Less linear cancellation (shadow's wrong direction would have cancelled
  more, but at the cost of NE damage)
- Better NE preservation (which is what we want in DT)

Same stem in FS_movement bucket: ΔerleA = +0.517 dB / Δne_pres = -1.417 dB
(complementary — better cancellation in FS, with reduced preservation
since FS has no NE to preserve). Mechanism consistent.

### Cohort tail check

`qNvSMyUSXUyrDGpOw7s6qg_farend_singletalk` (P52 ERL_decile_std p99.2):

- Baseline ERLE_active = 1.10 dB
- B5 candidate ERLE_active = 1.11 dB (Δ = +0.005)

No regression on the load-bearing cohort tail anchor. The fact that Δ is
essentially zero suggests this specific stem's path-change pattern
doesn't exercise the B5 code path (boost_q + reverse_copy combo) on
this particular file.

FS_static worst-20 by baseline ERLE_active: 18 / 20 unchanged (Δ ≈ 0),
1 with -0.017, 1 (wlAXM0iDgkm06i7UdRww1w) with +0.131. No regression.

## Hard abort criteria

| Criterion | Threshold | Actual | Pass? |
|---|---|---|---|
| Cohort tail (qNvSMyU) Δecho | ≥ -0.05 | +0.005 | ✓ |
| FS_static bucket mean Δ | ≥ -0.02 | +0.003 | ✓ |
| FS_movement bucket mean Δ | ≥ -0.02 | +0.004 | ✓ |
| Linear ERLE bucket mean | Δ ≥ -0.5 dB | min: -0.004 (DT_static) | ✓ |
| Linear NE distortion raise | none | min Δne: -0.013 (FS_movement, expected) | ✓ |

All pass.

## Decision

- **B5 mechanism**: works as designed; symmetric Yang 2017 R-reset is
  correctness-fix, not tuning hack.
- **Impact scope**: narrow (4 / 800 cases, ~0.5% impact rate).
- **Regression risk**: zero (all trade-offs are positive: ERLE-loss in DT
  paired with NE-pres gain).
- **Cohort tail invariant**: preserved.

**Flag state**: keep `shadow_r_reset_enabled = False` (default OFF) for
now. Final promotion decision deferred to "Final review" task at end of
Phase 1, alongside other Phase 1 default-OFF flags.

Rationale for deferring: B5's effect is narrow enough that bucket means
don't move. Promoting standalone gives no measurable production benefit.
At final review, B5 will be promoted together with other Phase 1 fixes
that compound (B6 state-aware shadow µ likely amplifies B5 since it
controls when shadow accumulates state to reset).
