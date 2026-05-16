# v3.18 Phase C.D-α Closeout — leakage_diverged Q-bifurcation (2026-05-16)

**Verdict**: CLOSED — CANNOT SHIP. Substrate retained default-OFF.
PRIMARY DT_static Δdeg = +0.000 (need ≥+0.010), no measurable RES
benefit despite functioning trigger.

## What landed

- `AecConfig.leakage_diverged_*` × 3 fields (enabled / threshold /
  hangover_frames)
- `AEC._check_leakage_diverged()` + `_apply_leakage_diverged()` methods
- Wiring in `AEC.process()` as 5th independent Q-boost trigger
  (after delay_shift / EPV / shadow_rise / regime_handler)
- Per-frame diag: `_diag['leakage_diverged_fired']`
- Env-var bridge `AEC_LEAKAGE_DIVERGED` / `AEC_LEAKAGE_DIVERGED_THRESHOLD`

## C.D-α.2 byte-equal: 5/5 PASS flag-OFF md5

## C.D-α.3 fire-rate gate: PASS

| bucket | stem | n_frames | fires | fq_usable% | shadow_adv_avg |
|---|---|---:|---:|---:|---:|
| FS_mv | 0I0XMl3M_FS_mv | 2415 | 0 | 89.7% | 0.814 |
| FS_tail | qNvSMyU | 2685 | 3 | 94.0% | 1.054 |
| FS_st | 0KjzXA3g | 2175 | 1 | 55.9% | 0.768 |
| DT_st | 0I0XMl3M_DT | 4186 | **6** | 58.1% | 1.054 |
| DT_st | sUQrHEPA | 4236 | 0 | 61.0% | 0.908 |

Total: 10 fires across 5 cases (gate range 5-200 → PASS).

## C.D-α.4 60-case AECMOS: FAIL

### Bucket means (all within ±0.001)

| Bucket | Δecho | Δdeg | bucket guard | verdict |
|---|---:|---:|---|---|
| FS_static | +0.001 | -0.000 | guards pass | OK |
| FS_movement | +0.000 | +0.000 | guards pass | OK |
| **DT_static** | **+0.000** | **+0.000** | **PRIMARY ≥+0.010** | **FAIL** |
| DT_movement | -0.000 | -0.000 | guards pass | OK |
| NE | +0.000 | +0.000 | guards pass | OK |

### Per-case extremes

| stem | bucket | Δecho | Δdeg | notes |
|---|---|---:|---:|---|
| N2rQLbnp_FS_static | FS_static | **+0.0128** | -0.0001 | **recovers Phase A.7 worst-case −0.131 breach** |
| QEeKiaNiD_DT_movement | DT_movement | -0.0005 | -0.0028 | tiny |
| qNvSMyU (cohort tail) | FS_static | +0.0000 | +0.0000 | no effect |

n=61: Δecho mean +0.0002, range [-0.0005, +0.0128]; Δdeg mean -0.0000,
range [-0.0028, +0.0000].

## Mechanism analysis — silent on downstream RES

leakage_diverged trigger fires correctly (10 smoke fires; 60-case
fires unmeasured but likely similar order). Q-boost on refined filter
also fires correctly. Yet RES output deltas are ~0 across all buckets.

Three hypotheses:

1. **Existing triggers cover the same cases**. delay_shift / EPV /
   shadow_rise / regime_handler boost_q already cover the same
   leakage_diverged conditions; the 5th trigger fires on cases that
   already have hangover from one of the existing 4.
   → A 5-trigger overlap audit would confirm; not done in this sprint.

2. **Q-boost magnitude is exhausted**. By the time leakage_diverged
   fires (requires fq_usable=True, which requires 0.4s startup + 0.2s
   reset + convergence seen), the filter has converged; another
   Q-boost adds noise rather than improvement.
   → Consistent with Phase B closeout's observation that converged
   filters don't have much misadjustment to correct.

3. **RES gating bottleneck**. RES still gates on `_filter_converged`
   (not `fq_usable`). Even if main filter improves from Q-boost, RES
   suppression decision doesn't change → no AECMOS movement.
   → Would require Phase C.E (consumer migration) to unblock.

The N2rQLbnp +0.0128 single-case win on FS_static is the only signal
that the mechanism does something — and it recovers a Phase A.7
worst-case breach (the same case went from baseline 3.84 → A.7 flag-ON
3.71 → C.D-α 3.85). Suggests leakage_diverged Q-boost helped main
filter resync after a path-change-like event.

## §6 kill criterion adjudication

PRIMARY FAIL (DT_static Δdeg = +0.000 vs ≥+0.010 required) → close
per §0.4. Substrate retained.

## Substrate retained (default OFF)

- `AecConfig.leakage_diverged_enabled` / `_threshold` / `_hangover_frames`
- `AEC._check_leakage_diverged` + `_apply_leakage_diverged` methods
- `_leakage_diverged_hangover` / `_leakage_diverged_fire_count` state
- Wiring as 5th Q-boost trigger
- Env-var bridge

Re-enable preconditions:
1. Phase C.E migrates RES gating to `fq_usable` (hypothesis 3 unblock)
2. Phase C.D-β (RES per-state ENR) lands first — would let RES react
   to filter state changes that leakage_diverged drives
3. Overlap audit (hypothesis 1) — identify whether leakage_diverged
   fires on novel cases vs ones already handled by existing 4 triggers

## Implications

This is the **first C-series gate FAIL** (C.A / C.B / C.C all PASS).
Pattern shift: behaviour-changing sub-phases are still hitting the
co-tuning wall, even with audit-only substrate (C.A/C.B/C.C) PASSed
upstream. The unblock isn't just having the substrate — it requires
**downstream consumers to consume it**.

This validates the C.1 §3.1 design call: read-only mirror first, then
consumer migration. C.D-α tried to ride the wave too early by adding
a new Q-boost trigger without consumer migration. The right ordering
is:
1. ~~C.A, C.B, C.C audit-only ports (DONE)~~
2. ~~C.D-α leakage_diverged Q trigger (CLOSED — premature)~~
3. **C.E consumer migration** (RES + DTD + regime handler read fq_usable)
4. Then C.D-α can be re-attempted as a Phase C.D follow-up (after C.E
   provides the consumer-side equivalence)

**Re-ordering decision for Phase C remainder**: skip C.D-β (would
follow same trap), go directly to C.E. After C.E lands, revisit C.D
as combined α+β arc with proper consumer-side wiring.

## Cross-references

- [docs/v3_18_c_d1_leakage_diverged_design.md](v3_18_c_d1_leakage_diverged_design.md) — design lock
- [docs/v3_18_c_b_verdict.md](v3_18_c_b_verdict.md) — C.B provides fq_usable
- [docs/v3_18_b_closeout.md](v3_18_b_closeout.md) — Phase B closeout (similar silent-mechanism mode)
- [docs/v3_18_a_closeout.md](v3_18_a_closeout.md) — Phase A closeout
