# AEC v3.8.1 — Round 1–7 Investigation Summary

**Outcome**: 17 ablations across 7 rounds. No variant satisfies the spec gates
(DT_mv deg ≥+0.02, FS echo ≥-0.02, NE deg drop ≤0.01) simultaneously.
v3.8.1 baseline_v381_seeded remains canonical. Classical-suppressor architecture
is at Pareto on this dataset; only NN postfilter remains as plausible ship route.

## Table of contents

- [F1–F4 Investigation Summary](#f1�f4-investigation-summary)
- [Round 2 — AEC3-state-routing investigation summary](#round-2--aec3-state-routing-investigation-summary)
- [Round 3 — Delay/EPC/Filter-misadjustment investigation](#round-3--delay/epc/filter-misadjustment-investigation)
- [Round 4 — Per-bin RES/ENR suppression investigation](#round-4--per-bin-res/enr-suppression-investigation)
- [Round 5 — Binding Audit + Coupled Ablation](#round-5--binding-audit-+-coupled-ablation)
- [Round 6 — RES Refactor + Artifact-Aware Noise Lift](#round-6--res-refactor-+-artifact-aware-noise-lift)
- [Round 7 — Signal-Flow Audit + Filter Trajectory + Output Mix](#round-7--signal-flow-audit-+-filter-trajectory-+-output-mix)


---

# F1–F4 Investigation Summary

**Branch**: `algo/f1-f4-investigation`  •  **Date**: 2026-05-02
**Outcome**: NO MERGE TO MAIN. All four findings produce null
bucket-level Δ on the 800-case BALANCED bench. Branch retained as
audit trail; not promoted.

## Baseline (v3.8.1, BALANCED, fl=52ms, cng=True, 800 cases)

| Bucket | n | echo | deg |
|---|---:|---:|---:|
| FS_static | 169 | 3.767 | 4.999 |
| FS_movement | 131 | 3.832 | 4.999 |
| DT_static | 186 | 4.252 | 2.257 |
| DT_movement | 114 | 4.121 | 2.288 |
| NE | 200 | 4.998 | 4.006 |

NE deg 4.006 hovers 0.006 above the 4.000 binding floor — any sub-branch
that pushed NE deg below 4.000 would have been unshippable per the project
spec. None of the four did, since none had measurable effect.

## Findings × bench Δ

| Finding | Mechanism | DT_mv Δdeg | FS_st Δecho | DT_st Δdeg | NE Δdeg | Verdict |
|---|---|---:|---:|---:|---:|---|
| F2 | (audit only — confirms `res_dt_reduction` and 3 siblings dead under ENR) | — | — | — | — | informational |
| F4 | Tighten inst-ERLE DT cap 4→2 on cold-start signature | -0.004 | -0.001 | -0.003 | +0.000 | REJECT |
| F1 | InitialDtGuard: `dt_residual_scale × 0.7` on cold-start + NE-likely | -0.004 | +0.002 | +0.001 | +0.000 | REJECT (bimodal) |
| F3 | `usable_linear_estimate` gate replaces `not filter_converged` | +0.002 | -0.000 | -0.001 | +0.000 | REJECT (sub-noise) |

All Δ within ±0.004 — well below the 0.02 acceptance threshold.

## Why null (cumulative analysis)

The reviewer's hypothesis: v3.8.1 lacks AEC3-style cold-start /
usable-linear / dominant-nearend / misadjustment-recovery protections.
This is true as architectural fact (per `docs/aec3_reference.md` and
`docs/spec_dt_from_frame_zero.md`).

But on the 800-case AEC Challenge blind set, the population that would
benefit is a minority:

- **F4**: cold-start trigger (`not once_converged AND far_active>200 AND
  erl>0.4`) fires on 2/30 DT cases (0.1% of frames). The `erl>0.4` gate
  is rarely met because most DT cases either converge or have weak echo
  paths that don't drift ERL upward.
- **F1**: same trigger plus `using_render_based AND mic/far>0.3` —
  fires on 5/30 DT (1.4% of frames), 24/120 broader scan. The
  `dt_residual_scale × 0.7` action is *bimodal*: helps the NE-dominant
  subset, hurts the still-converging subset, mean ≈ 0. On baseline-worst-20
  cases F1 mean Δdeg = -0.002, only 3/20 helped > +0.005.
- **F3**: `usable_linear_estimate` differs from `filter_converged` only
  during transient post-converged dips, which the existing
  `_render_based_hold=5` hysteresis already covers. Per-case
  distribution bimodal but slightly positive lean (+0.002 on
  baseline-worst-20).

The cold-start DT-from-frame-0 family — described in
`docs/spec_dt_from_frame_zero.md` as the dominant signature among the
"worst 12" — is *not* the dominant signature among the 800-case
blind set's worst DT_movement cases. The two populations diverge.

## What this rules out / leaves open

**Ruled out** for further work without a different selector:
- Cold-start DT-from-frame-0 mitigations gated on `not once_converged
  AND erl > 0.4` (F1 / F4 trigger style).
- `usable_linear_estimate` as a per-frame attribution switch (F3).
- F5 (dominant-nearend) using raw mic/far ENR — same trigger style
  predicts same null per F1's distribution.

**Open** for a future, separately-scoped investigation:
- Per-bin NE confidence mask (instead of frame-level guard) — would
  help only DT bins, not affect FS bins. Hardcoded `dt_per_bin = max(
  effective_dt, 1.0 - coh2)` exists at aec.py:1711 but coh2 fails on
  stationary far-end (note at L1716-1718). A learned per-bin NE mask
  would address this.
- Movement-induced delay re-acquisition / gain ramps — the dominant
  signature in DT_movement worst cases per the `_with_movement_`
  variants on the blind set. Out of scope for this branch.
- Echo-path nonlinearity beyond filter representation — needs an NN
  postfilter (DTLN-AEC / TEACAEC), tracked in
  `~/.claude/plans/jazzy-brewing-castle.md` per `aec_methods.md`.

## Disposition

Branch `algo/f1-f4-investigation` is local-only and not pushed. Sub-branch
verdicts (`f4-erle-correction-guarded/`, `f1-initial-dt-guard/`,
`f3-usable-linear-gate/`) are committed to the investigation branch as
artifacts. Code on `python/aec.py` reverted to v3.8.1 baseline (b3f62c3 +
no behaviour changes).

To pick this up again: review `experiments/<branch>/verdict.md` for the
specific failure mode, then either widen the trigger / change the action
on a fresh sub-branch, or design a new selector that addresses the
movement / NE-mask gaps above.

---

# Round 2 — AEC3-state-routing investigation summary

**Branch**: `algo/f1-f4-investigation`  •  **Date**: 2026-05-02
**Outcome**: NO MERGE TO MAIN. State-routing patches all null on bench.
Branch retained as audit + diagnostic plumbing (Phase 0 trace fields
on `python/aec.py` are kept; audio-passive).

## Round 2 chronology

| Step | Action | Result |
|---|---|---|
| Phase 0a | Mapped AEC3 states → aec.py | `experiments/aec3_phase0/state_mapping.md` |
| Phase 0b | Added 6 trace-only diag fields to aec.py | `python/aec.py` (audio-passive) |
| Phase 0c | 800-case state dump via `trace_phase0.py` | `experiments/aec3_phase0/states.json` |
| Phase 0d | Worst-vs-best population analysis | `experiments/aec3_phase0/analysis.md` |
| Phase 1 | SKIPPED (Phase 0: usable_v2 ≡ v1) | — |
| Phase 2 | SKIPPED (Phase 0: dominant_pct doesn't separate) | — |
| CNG calib | Two unseeded runs of same code | DT_mv Δdeg ±0.004 — noise floor |
| Seed | `np.random.seed(0)` in `run_ours` per case | `python/eval_aec_challenge.py` |
| Phase 3 | ERLE/RES reset on transitions | `experiments/phase3_erle_reset/verdict.md` |

## Round 2 bench Δ table (all vs seeded baseline)

| Branch | FS_st Δecho | FS_mv Δecho | DT_st Δdeg | DT_mv Δdeg | NE Δdeg |
|---|---:|---:|---:|---:|---:|
| baseline_v381_seeded (reference) | 3.768 | 3.827 | 2.257 | 2.286 | 4.007 |
| Phase 3 (ERLE/RES reset) | +0.001 | -0.000 | +0.000 | +0.000 | +0.000 |
| **CNG noise floor (same-code re-run)** | **+0.000** | **+0.002** | **-0.000** | **-0.004** | **-0.000** |

Phase 3's Δ is INSIDE the noise floor in every direction. Genuine no-op.

## Key finding 1: CNG noise floor invalidates Round 1 deltas

`np.random.randn` unseeded at `aec.py:2009-2010` produces ±0.004
DT_movement deg variance between same-code runs. Round 1 (F1/F3/F4)
showed Δ ≤ ±0.004 — at or below this floor. Those "null" verdicts
were technically correct outcomes but the bench couldn't have detected
small signals if they existed.

Going forward:
- Seeded baseline (`baseline_v381_seeded/scores.json`) is the reference.
- Single-run Δ < 0.005 is uninterpretable without seeded CNG.
- Multi-seed averaging would tighten the floor further (out of scope here).

## Key finding 2: state routing doesn't move bench means on this dataset

Cumulative across F1 + F3 + F4 + Phase 1 (skipped) + Phase 2 (skipped)
+ Phase 3:

- **Cold-start guards** (F1, F4): null bucket, bimodal per-case (was
  signal+noise mix — the bimodal effect from Round 1 was probably CNG).
- **Usable-linear-estimate gate** (F3, Phase 1): no separation, no Δ.
- **DominantNearend** (Phase 2 skipped): population doesn't exist.
- **ERLE / RES trust reset on transitions** (Phase 3): null.

The reviewer's architectural-gap thesis (v3.8.1 lacks AEC3-style
cold-start + usable-linear + dominant-NE protections) accurately
describes what AEC3 has that v3.8.1 lacks, but on this 800-case
blind set those mechanisms don't move the binding constraint.

## Where the binding constraint is (extrapolated)

Phase 0 finding: `transitions` count discriminates worst-DT (worst-20
mean 9.4 vs best-20 mean 5.1). Phase 3 showed fixing the post-transition
trust state doesn't help. So transitions are SYMPTOM not CAUSE.

Plausible causes for further investigation (out of scope here):
1. **Filter W trajectory during transitions** — W-touching territory,
   excluded by user constraint; would need NLMS shadow / coarse filter
   redesign.
2. **Movement-induced delay re-acquisition** — DT_movement is 114 cases;
   movement variant introduces delay shifts that EPC partially handles
   but residual error during re-acquisition propagates to ResFilter.
3. **Echo-path nonlinearity** — beyond the linear filter's
   representational capacity. Solvable only with NN postfilter
   (DTLN-AEC / TEACAEC), per `~/.claude/plans/jazzy-brewing-castle.md`.

## Disposition

- `algo/f1-f4-investigation` retained: contains all verdicts +
  bench data + Phase 0 trace plumbing.
- `python/aec.py` keeps the Phase 0 trace fields (`initial_state_active`,
  `usable_linear_estimate_v2`, `dominant_nearend_like_state`,
  `erle_reset_signal`) — audio-passive, useful for future trace work.
- `python/eval_aec_challenge.py` keeps `np.random.seed(0)` per case —
  bench tooling improvement.
- All sub-branches deleted.
- No code change to suppressor / filter / mu_scale / W path.

## What this rules out cleanly

For any future investigator: don't repeat these.
- Cold-start gating triggers (F1/F4 family) — they fire on a minority
  population.
- usable_linear_estimate as the residual attribution switch (F3 / Phase 1).
- DominantNearend frame-level state machine (Phase 2).
- ERLE / RES trust reset on convergence transitions (Phase 3).

These have all been bench-tested with seeded CNG. Move on to a
different mechanism class.

---

# Round 3 — Delay/EPC/Filter-misadjustment investigation

**Branch**: `algo/f1-f4-investigation`  •  **Date**: 2026-05-03
**Outcome**: NO MERGE TO MAIN. D2 produces real DT improvement but
catastrophic FS regression (Pareto-fail). D3 produces sub-acceptance
DT improvement scattered on mid-baseline cases, not concentrated on
worst-DT. Branch retained as audit + Phase 0 trace plumbing.

## Findings

### Phase 0 trace (no audio change) — 800-case states.json

Strong separators on DT_movement worst-20 vs best-20 (baseline_v381_seeded):

| Field | worst | best | Δ | implication |
|---|---:|---:|---:|---|
| `epc_pct` | 0.573 | 0.354 | +0.22 | EPC stuck longer on worst cases |
| `epc_max_run` | 1639 frames | 932 frames | +707 (≈7s) | tick_hangover frozen |
| `pre_epc_dt_p90` | 0.420 | 0.276 | +0.145 | real DT being zeroed during stuck-EPC |
| `pre_epc_dt_count` | 2415 | 1490 | +925 | thousands of DT-zero frames per case |
| `transitions` | 9.35 | 5.10 | +4.25 | (Round 2 finding, persists) |

### D1 — SKIP

`delay_shift` count = 0 across all 800 cases. `eval_aec_challenge.run_ours`
calls `estimate_delay()` first to globally pre-align ref + enables online
delay-est on movement; the combination eliminates internal `delay_shift`
events. The population D1 was meant to fix doesn't exist in the bench
pipeline.

### M1 — SKIP

`filter_scale_ratio` direction is inverted from hypothesis. Worst-FS-echo
cases have median scale_ratio = 30 (40× overestimate); best-FS = 1.06.
DT_mv worst median = 2.4 vs best = 5.8 — both off but unclear direction.
Misadjustment-correction would need separate FS-overshoot vs DT designs;
out of round 3 scope.

### D2 — REJECT (Pareto-fail)

Decouple `tick_hangover()` from `_filter_converged` gate.

| Variant | DT_mv Δdeg | FS_st Δecho | FS_mv Δecho | NE Δdeg | acceptance |
|---|---:|---:|---:|---:|---|
| D2 unconditional | **+0.011** | **-0.058** | **-0.108** | +0.000 | FAIL FS |
| D2b once_conv-gated | -0.002 | -0.058 | -0.062 | +0.000 | FAIL FS + lost DT |

D2 unconditional delivers the largest DT_mv improvement seen in any
round (+0.011 deg, ~half of acceptance threshold), with concentrated
per-case wins (top winners +0.36, +0.34, +0.19, +0.18, +0.16, +0.15).
But FS catastrophe: PZ7V0 dropped 4.14→2.86 echo (-1.29). Stuck-EPC was
silently providing render-mode protection on FS cases that depended on
it; once unblocked, the linear filter's wrong-path estimate exposes
the FS suppressor.

D2b (gate on `_filter_once_converged`): partial FS recovery (-0.062 vs
-0.108 unconditional) but lost the DT improvement (+0.011 → -0.002).
Suggests both populations (DT wins and FS losses) span both converged
and non-converged buckets.

**Conclusion**: D2 cleanly identifies the right mechanism (stuck-EPC is
real and damages worst-DT), but the cure is bound to the same
protection that helps worst-FS. A surgical fix would need to gate
hangover-tick on additional positive evidence (e.g. shadow_advantage
stable, far_active established, divergence < threshold) — outside
round 3's scope and reverts to "tunable thresholds" territory the
review wanted to avoid.

### D3 — REJECT (sub-acceptance, not concentrated on worst)

Replace `if epc_active: raw_dt = 0.0` with `raw_dt = raw_dt × <fraction>`,
preserving partial DT signal for ResFilter while keeping `is_stationary_dt
= False` for adaptation safety.

| Variant | DT_mv Δdeg | FS_st Δecho | FS_mv Δecho | NE Δdeg | DT_mv worst-20 Δdeg | acceptance |
|---|---:|---:|---:|---:|---:|---|
| D3 (0.15) | +0.003 | -0.001 | -0.007 | +0.000 | -0.000 | sub-acceptance |
| D3b (0.25) | +0.005 | -0.003 | -0.014 | +0.000 | -0.001 | sub-acceptance |

Both variants stay within FS bucket-mean acceptance (≤0.02 regression)
and don't hurt NE. But:
- DT_mv bucket-mean Δdeg ≤ +0.005 (well below +0.02 threshold)
- Mean Δdeg over baseline-worst-20 = ~0 (D3 0/20 helped, D3b 0/20 helped)
- Wins concentrate on **mid-baseline** cases (top winners had baseline_deg
  3.24, 3.63, 2.12) — these were already OK
- Some FS_mv individual cases drop large (D3b: 0luXwWjGEE -0.249,
  XXz0qkUSd -0.206)

The action helps cases where the suppressor side gets useful per-bin
information from the partial DT signal, but the cases that REALLY hurt
on baseline (deg ≈ 1.4) are damaged so far upstream that downstream
DT awareness can't recover them.

**Conclusion**: D3 is in the right direction but the binding population
(baseline-worst DT_mv) is upstream of the suppressor — fixing the
suppressor's DT awareness doesn't help cases where the residual echo
estimate itself is wrong.

## Cumulative across Round 1, 2, 3

| Round | Phase | DT_mv Δdeg | Cost / verdict |
|---|---|---:|---|
| 1 | F4 (cap 4→2) | -0.004 | null + FS within noise |
| 1 | F1 (dt_residual_scale × 0.7) | -0.004 | bimodal per-case (signal+CNG noise) |
| 1 | F3 (usable_linear_v1 gate) | +0.002 | sub-noise |
| 2 | Phase 0 trace + Phase 3 (ERLE/RES reset on transitions) | +0.000 | null with seeded CNG |
| 2 | CNG calibration | — | discovered ±0.004 noise floor |
| 3 | D2 unconditional | **+0.011** | FS catastrophe (-0.108) |
| 3 | D2b once_conv-gated | -0.002 | lost DT, FS still bad |
| 3 | D3 (0.15) | +0.003 | not on worst-20 |
| 3 | D3b (0.25) | +0.005 | not on worst-20, FS_mv -0.014 |

The largest DT_mv signal across all 8 ablations is D2 unconditional
(+0.011), but it's Pareto-bound to FS damage. Every clean (FS-safe)
intervention is sub-acceptance.

## What this rules out

For DT_movement worst-20 (baseline deg ≈ 1.4):
- ❌ State routing (F1, F3, Phase 1, Phase 2, Phase 3, D2 once_conv-gated)
- ❌ Suppressor-side DT-awareness restoration (D3 / D3b)
- ❌ Cold-start DT-from-frame-0 guards (F1, F4)
- ❌ ERLE/RES trust resets (Phase 3)
- ❌ Filter-scale correction (M1 — direction unclear, FS overshoots)

The DT_mv worst-20 cases need either:
1. **Filter weight (W) trajectory fix** during transitions / movement — out
   of round-3 scope per user constraint.
2. **Per-bin spectral NE classifier** that operates upstream of frame-level
   DT — not reachable from state machine layer.
3. **NN postfilter** for residual echo on the cases linear filter can't
   model — separate plan (`~/.claude/plans/jazzy-brewing-castle.md`).

## Disposition

- `algo/f1-f4-investigation` retained, includes Round 3 trace plumbing
  on `python/aec.py` (audio-passive: epc_active_now, epc_hangover_count,
  p_max_override_active, p_floor_beta_active, div_counts, filter_scale_ratio,
  raw_dt_pre_epc) and `python/trace_phase0.py`+`analyze_round3.py`.
- All sub-branches dropped.
- No code change to behavior path. Patches reverted.
- Seeded CNG bench tooling (`np.random.seed(0)` in `run_ours`) kept.

`baseline_v381_seeded` remains the canonical reference baseline for any
future Δ comparison.

---

# Round 4 — Per-bin RES/ENR suppression investigation

**Branch**: `algo/round4-perbin-res` (Phase 0 trace) +
`algo/r4-r1-echo-dominant` (v1) + `algo/r4-r1-echo-dominant-v2` (v2)
**Date**: 2026-05-03
**Outcome**: NO MERGE TO MAIN. R1 v1 + v2 both null (Δ < 0.001 every
bucket). R2/R3/R4 abandoned because they share the same per-bin
classifier / coh2 dependency that Phase 0 already proved doesn't
discriminate worst-DT_mv cases. Branch retained as audit + Phase 0
trace plumbing.

## Phase 0 (per-bin RES diagnostics, audio-passive)

11 new per-frame fields surfaced in `_diag` (coh2_mean_voice,
res_over_err_voice, ne_over_err_voice, g_voice_mean/min/p10,
echo_dominant_bin_pct, etc.). 800-case seeded trace.

### DT_movement worst-20 vs best-20 (locked by baseline_v381_seeded.deg)

| field | worst-20 | best-20 | Δ | direction |
|---|---:|---:|---:|---|
| coh2_mean_voice | 0.144 | 0.146 | -0.002 | **flat** |
| res_over_err_voice | **6.60** | **3.93** | **+2.66** | residual estimator IS firing on worst |
| ne_over_err_voice | 0.793 | 1.063 | -0.270 | worst NE-floor lower |
| g_voice_mean | 0.535 | 0.581 | -0.046 | worst already pressed |
| g_voice_min | **0.233** | **0.330** | **-0.097** | worst already pressed harder |
| g_voice_p10 | 0.329 | 0.424 | -0.095 | same |
| echo_dominant_bin_pct | **0.023** | **0.030** | **-0.007** | classifier hits LESS on worst |

worst-20 mean baseline_deg = 1.397; best-20 mean = 3.312.

### Phase 0 decision

Per plan threshold: PROCEED with R1 (g_voice_mean > 0.5, res/err_voice > 0.4).
But yellow flag: classifier targets wrong population because (a) coh2
stays ~0.14 flat both worst and best, (b) suppressor already presses
worst-20 harder than best-20.

## R1 — Per-bin echo-dominant classifier

Cap gain on bins satisfying `(coh2 > 0.5) & (res/err > 0.5) & (ne/err < 0.3)`.
Conservative AND of three signals; classifier hits ~3% of bins per frame.

| Variant | Cap | DT_mv Δdeg | DT_mv Δecho | FS_st Δecho | FS_mv Δecho | DT_mv worst-20 wins | acceptance |
|---|---:|---:|---:|---:|---:|---:|---|
| R1 v1 | 0.50 | +0.00000 | -0.00001 | +0.00000 | +0.00010 | 0/20 | NULL |
| R1 v2 | 0.30 | -0.00003 | -0.00000 | +0.00002 | +0.00048 | 0/20 | NULL |

Both runs every metric < 0.001 (well under CNG ±0.004 noise floor).

**Why null**: cap=0.5 doesn't matter because g_voice_mean ≈ 0.535 already.
Cap=0.3 does enter the population, but the ~3% of bins where the
classifier fires are mostly bins where g was already low — the cap
gates already-pressed bins. Worst-20 has 2.3% classifier hit rate
(LESS than best-20's 3.0%), so the action concentrates on already-good
cases.

## R2 / R3 / R4 — abandoned

All three depend on the same per-bin signals R1 used:

- **R2 voice-band selective**: AND echo_dominant with voice_band_mask.
  Strictly tighter classifier — fewer hits than R1, same null direction.
- **R3 ne floor per-bin**: uses same `echo_dominant` mask to cut ne floor.
  Same root issue: mask doesn't fire on worst-20.
- **R4 EMR coh2-gated**: gates EMR transparency on `coh2 > 0.5`. Same
  coh2 ~0.14 flat distribution → gate seldom flips → null.

Phase 0 diagnostic already showed that the binding constraint is upstream
of any per-bin suppressor decision. coh2 cannot separate worst from best
DT_movement. Running R2/R3/R4 would burn ~3 hours bench for predictable
nulls.

## Cumulative across Round 1, 2, 3, 4

| Round | Phase | DT_mv Δdeg | Cost / verdict |
|---|---|---:|---|
| 1 | F4 (cap 4→2) | -0.004 | null + FS within noise |
| 1 | F1 (dt_residual_scale × 0.7) | -0.004 | bimodal per-case |
| 1 | F3 (usable_linear_v1 gate) | +0.002 | sub-noise |
| 2 | Phase 3 (ERLE/RES reset on transitions) | +0.000 | null seeded CNG |
| 2 | CNG calibration | — | discovered ±0.004 noise floor |
| 3 | D2 unconditional | **+0.011** | FS catastrophe (-0.108) |
| 3 | D2b once_conv-gated | -0.002 | lost DT, FS still bad |
| 3 | D3 (0.15) | +0.003 | not on worst-20 |
| 3 | D3b (0.25) | +0.005 | not on worst-20, FS_mv -0.014 |
| **4** | **R1 v1 cap=0.5** | **+0.00000** | **null all metrics** |
| **4** | **R1 v2 cap=0.3** | **-0.00003** | **null all metrics** |

Across **10 ablations / 4 rounds**, the largest DT_mv signal remains
Round 3 D2 unconditional (+0.011) — Pareto-bound to FS damage. Every
clean (FS-safe) intervention is sub-acceptance or null.

## What Round 4 added that prior rounds didn't

Phase 0 per-bin diagnostics give a definitive **per-bin negative result**:
- worst-DT_mv `g_voice_min = 0.233` — suppressor already presses voice
  bins down to gain 0.23 in worst cases. There is little room to push
  further without speech damage.
- worst-DT_mv `coh2_voice = 0.14` (flat vs best 0.15) — coherence is
  not a usable per-bin discriminator on this dataset.
- `res_over_err_voice` worst 6.60 / best 3.93 — residual ESTIMATOR
  has the right intuition (worst > best), but the suppressor's response
  to that estimate is already saturated.

## Per plan escape clause

> "如果 R1-R4 都只在 ±0.005 內，代表 classical suppressor 也接近 Pareto，
> v3.8.1 就應該進整合，不再繼續硬榨"

R1 v1 (-0.00001) + R1 v2 (-0.00003) both << 0.005. Diagnostic shows
R2/R3/R4 will share the same outcome. **Recommendation: v3.8.1 進整合**;
further suppressor-side work on this dataset is not productive without
either:

1. **Filter weight (W) trajectory fix during transitions/movement** —
   out of round-4 scope per user constraint.
2. **Per-bin learned NE classifier** (NN postfilter) — separate plan
   `~/.claude/plans/jazzy-brewing-castle.md`.
3. **Different dataset that exposes a different bottleneck** — current
   AEC Challenge blind set has worst-DT cases with baseline_deg ≈ 1.4
   that are upstream-broken; per-bin suppressor cannot recover.

## Disposition

- `algo/round4-perbin-res` retained, includes Round 4 Phase 0 trace
  plumbing on `python/aec.py` (audio-passive: 11 per-bin _diag fields,
  voice_band_mask precompute) + `python/trace_phase0.py` extension
  + `python/analyze_round4.py`.
- `algo/r4-r1-echo-dominant` (v1, cap=0.5) — kept as audit, not merged.
- `algo/r4-r1-echo-dominant-v2` (cap=0.3) — kept as audit, not merged.
- R2 / R3 / R4 sub-branches not created (Phase 0 + R1 invalidated premise).
- No code change to behavior path. R1 patch reverted on round4 base.
- Seeded CNG bench tooling (`np.random.seed(0)` in `run_ours`) kept.

`baseline_v381_seeded` remains canonical reference.

---

# Round 5 — Binding Audit + Coupled Ablation

**Branch**: `algo/round5-coupled-audit` (Phase 0 + audit)
**Sub-branches** (audit-only, no merge):
  `algo/r5-A34-noise-temporal-coupled` (v1),
  `algo/r5-A34-v2-narrow` (v2),
  `algo/r5-A34-v3a-A3only` (v3a),
  `algo/r5-A34-v3b-soft-A3` (v3b)
**Date**: 2026-05-03
**Outcome**: NO MERGE TO MAIN. R5 confirmed Phase 0 hypothesis (downstream
lifts erase soft-gate decisions on worst-DT cases) and produced the first
non-null result in 4 rounds (v2 DT_mv deg +0.023, 15/20 worst-20 cases
improved). But the Pareto trade is bound: any setting that delivers the
DT win regresses FS by ≥0.02; any setting that protects FS loses the
DT win. R5 mapped the full Pareto curve. Branch retained as audit.

## Phase 0 binding audit

3-way bench raw / nores / final on 800 cases produced strongest single
finding of the entire investigation:

| Bucket | nores echo | ours echo | RES echo+ | nores deg | ours deg | RES deg- |
|---|---:|---:|---:|---:|---:|---:|
| FS_static | 2.620 | 3.768 | +1.148 | 5.000 | 4.999 | -0.001 |
| FS_movement | 2.616 | 3.827 | +1.211 | 5.000 | 4.999 | -0.001 |
| NE | 4.998 | 4.998 | +0.000 | 4.120 | 4.007 | -0.113 |
| DT_static | 3.164 | 4.251 | +1.087 | 3.306 | 2.257 | -1.049 |
| DT_movement | 3.153 | 4.120 | +0.967 | 3.283 | 2.286 | -0.997 |

Linear AEC alone wins DT deg by ~1.0; RES adds ~+1 echo at cost of
~-1 deg. Bucket means already ≈ AEC2 reference on every metric.

### Stage gain trajectory (DT_mv worst-20 vs best-20)

| stage | worst-20 g | best-20 g | gap |
|---|---:|---:|---:|
| soft-gate | 0.249 | 0.496 | -0.247 |
| post_temporal | 0.368 | 0.469 | -0.101 |
| **after_noise_lift (final)** | **0.535** | **0.581** | **-0.046** |

Soft-gate's 50% gap (0.25 vs 0.50) erased to 92% (0.535 vs 0.581) by
post_temporal + noise_floor lift, asymmetrically lifting worst more than
best. **Explains why R4 R1 cap=0.5 was null** — any soft-gate intervention
gets lifted by ~0.30 in absolute g terms before reaching output.

## Coupled Test A3+A4 — 4 variants on Pareto curve

Mask: per-bin echo-dominant detector built right after softgate stage.
Frame-level gates: `far_power > 1e-4 AND effective_dt < 0.45`.
Per-bin: `g_softgate < THR_g AND res_over_err > THR_r` (+ optional voice-band).
A3: bypass / soften noise_floor lift on echo-dom bins.
A4: divide attack-side temporal alpha by 3.0 on echo-dom bins.

| | A3 | A4 | mask | FS_st e | FS_mv e | DT_mv e | DT_mv d | NE d | w20 d | w20 impr | b20 d |
|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| v1 | full | 3x | g<0.20, res>1.0 | -0.068 | -0.064 | +0.091 | -0.139 | -0.010 | +0.059 | 10/20 | -0.231 |
| **v2** | full | 3x | g<0.15, res>1.5, voice-only | -0.042 | -0.039 | +0.074 | **+0.023** | -0.004 | **+0.106** | **15/20** | -0.069 |
| v3a | full | OFF | same v2 | -0.059 | -0.050 | +0.062 | +0.022 | -0.002 | +0.091 | 13/20 | -0.053 |
| v3b | 50% | 3x | same v2 | **+0.021** | **+0.030** | +0.051 | -0.021 | -0.002 | +0.015 | 7/20 | -0.051 |

Per-case DT_mv direction (114 cases):
- v1: 12 both-win / 23 both-lose
- v2: 37 both-win / 8 both-lose ← best win count
- v3a: 36 both-win / 11 both-lose
- v3b: 30 both-win / 5 both-lose ← best collateral protection

## Acceptance gate analysis

User spec hard gates (all required):
- DT_mv deg ≥ +0.02
- FS_st echo ≥ -0.02
- FS_mv echo ≥ -0.02

| | DT_mv deg gate | FS_st gate | FS_mv gate |
|---|---:|---:|---:|
| v1 | ✗ (-0.139) | ✗ (-0.068) | ✗ (-0.064) |
| v2 | ✓ (+0.023) | ✗ (-0.042) | ✗ (-0.039) |
| v3a | ✓ (+0.022) | ✗ (-0.059) | ✗ (-0.050) |
| v3b | ✗ (-0.021) | ✓ (+0.021) | ✓ (+0.030) |

**No variant passes all three gates simultaneously.** v2 is closest (passes DT
gate, fails FS by 2× the limit). v3b passes FS by safe margin but flips
DT_mv deg negative.

## A3 vs A4 isolation (v2 → v3a)

Disabling A4 made FS WORSE (-0.039 → -0.050 on FS_mv), not better.
A3 (noise floor bypass) is the FS damage source — over-suppression on
echo-dominant bins creates spectral artifacts AECMOS perceives as quality
loss. The noise_floor lift was preventing those artifacts, not just NE
protection.

A4 (faster temporal attack) on top of A3 acts as a partial counter — its
gain trajectory smoothing absorbs some artifact magnitude. A4 contributes
+0.012 echo with no net FS cost when added to A3.

## Cumulative across Round 1, 2, 3, 4, 5

| Round | Phase / Variant | DT_mv Δdeg (bucket) | DT_mv Δecho | Acceptance verdict |
|---|---|---:|---:|---|
| 1 | F4 (cap 4→2) | -0.004 | n/a | null |
| 1 | F1 (dt_residual_scale × 0.7) | -0.004 | n/a | bimodal |
| 1 | F3 (usable_linear_v1 gate) | +0.002 | n/a | sub-noise |
| 2 | Phase 3 (ERLE/RES reset on transitions) | +0.000 | n/a | null seeded |
| 3 | D2 unconditional | -0.139 | +0.011 | FS catastrophe (-0.108) |
| 3 | D2b once_conv-gated | -0.002 | n/a | lost DT, FS still bad |
| 3 | D3 (0.15) | +0.003 | n/a | sub-acceptance, not worst-20 |
| 3 | D3b (0.25) | +0.005 | n/a | sub-acceptance, FS_mv -0.014 |
| 4 | R1 v1 cap=0.5 | +0.000 | -0.000 | null all metrics |
| 4 | R1 v2 cap=0.3 | -0.000 | -0.000 | null all metrics |
| **5** | **A34 v1 (full A3, wide mask)** | **-0.139** | **+0.091** | FS catastrophic (-0.068) |
| **5** | **A34 v2 (narrowed mask)** | **+0.023** | **+0.074** | DT pass, FS fail (-0.04) |
| **5** | **A34 v3a (A3-only)** | **+0.022** | **+0.062** | DT pass, FS fail (-0.05) |
| **5** | **A34 v3b (soft A3 50%)** | **-0.021** | **+0.051** | FS pass, DT fail (-0.02) |

Across 14 ablations / 5 rounds:
- Largest DT_mv echo signal: v1 +0.091 (Pareto-bound to FS -0.06+)
- First DT_mv deg positive: v2 +0.023 (FS -0.04, fails gate)
- First FS-positive variant: v3b +0.02-0.03 (DT_mv deg -0.021)
- **No variant satisfies all hard gates.**

## What R5 added that prior rounds didn't

1. **Per-stage gain trace infrastructure** (audio-passive, kept on round5
   base): 9 voice-band stage means + raw/nores/final 3-way bench harness.

2. **Definitive Pareto characterization**: Phase 0 + 4 sub-branches map
   the full trade-off curve:
   - Aggressive end (v1): DT echo +0.09 / FS regress -0.07
   - Balanced (v2): DT deg +0.023 / FS regress -0.04
   - Soft end (v3b): FS positive +0.03 / DT deg -0.021
   The curve does not pass through the (DT≥+0.02, FS≥-0.02, NE drop≤0.01)
   corner that the spec requires.

3. **Mechanism confirmed**: soft-gate decisions ARE being erased by
   downstream lifts; preserving them DOES recover DT-bucket signal.
   But the lifts serve dual purpose (NE protection + spectral artifact
   prevention), so bypassing them on echo-dominant bins creates new
   FS-side artifacts that AECMOS perceives as quality loss.

4. **First mechanism-passing audio change in 5 rounds**: v2 produces
   bucket-mean DT_mv echo +0.074 / deg +0.023, with 15/20 worst-20 cases
   directionally improved by ≥+0.02 deg, max winner +0.56. Phase 0
   hypothesis is empirically validated even though FS prevents shipping.

## Why R5 family Pareto can't be broken with mask-only intervention

Stage trajectory shows soft-gate's worst-vs-best 50% gap collapses to 92%
at output. Closing the lift gap requires either:

1. **Better per-bin classifier** — current `g_softgate < 0.15 AND res/err > 1.5`
   fires on FS bins where the over-suppression itself is artifact-creating.
   Need a discriminator that distinguishes "echo to suppress" from "FS bin
   that will create musical noise if suppressed harder". `coh2` is flat
   across worst/best DT (R4 finding), so coherence doesn't help. NE-presence
   detector only helps DT.

2. **Spectral artifact protection independent of NE** — reformulate the
   noise_floor lift to depend on **per-bin gain trajectory smoothness**
   instead of `noise_psd`. Don't lift gains that are stable; do lift gains
   that are spiking. This is structural — outside R5 scope.

3. **NN postfilter** — separate plan `~/.claude/plans/jazzy-brewing-castle.md`.

## Why prior null findings still hold

R4 R1 (per-bin cap on softgate) was null because cap=0.5 sat above
softgate's already-suppressed worst-20 voice mean (~0.25), and cap=0.3
got lifted +0.30 by downstream stages. R5 v2 succeeded where R4 R1
failed by **acting on the lifts themselves**, not the cap.

## Disposition

- All 4 sub-branches retained as audit (no merge to main).
- Round 5 base branch `algo/round5-coupled-audit` retains:
  - 9-stage gain trace plumbing (audio-passive)
  - trace_phase0.py extension for stage gains
  - Phase 0 verdict + 800-case states.json
  - `experiments/round5_phase0/` raw/nores/final 3-way bench scores
  - `experiments/r5_A34_v{1,2,3a,3b}/` per-variant outputs
- No code change to behavior path on round5 base.

## Recommendation for future rounds

R5 plan's escape clause was satisfied: "若所有結果在 ±0.005 內，CNG/noise
floor 不是主因。" — actually NOT this case here. R5 produced large
multi-bucket movements (±0.05–0.15). The conclusion is different:

**The classical suppressor pipeline cannot satisfy the strict acceptance
gates on this dataset.** Phase 0 showed bucket means already ≈ AEC2 — there
is no headroom in the bucket-mean game. The DT_mv worst-20 deg gap
(~+0.5 to +1.0 needed per case to reach AEC2-level NE preservation) is
beyond what mask-based per-bin gating can deliver without creating
FS artifacts.

Future-direction options (all out of R5 scope):

1. **NN postfilter** (TEACAEC / DTLN-AEC): separate plan, expensive but
   only known path to break Pareto on this dataset.
2. **Spectral-trajectory-aware noise floor** (re-formulate noise lift
   condition): structural rewrite of [aec.py:1985-2025](AEC/python/aec.py#L1985).
3. **Filter W trajectory fix during transitions**: addresses upstream
   damage that forces RES into bad decisions. Out of scope per user
   constraint, but R5 Phase 0 trace (`scale_ratio_p10` field exposed)
   keeps the diagnostic line open.

`baseline_v381_seeded` remains canonical reference. v3.8.1 is at Pareto
on classical-suppressor architecture; integration-and-ship is the plan
default per Round 4 conclusion (still holds).

---

# Round 6 — RES Refactor + Artifact-Aware Noise Lift

**Branch**: `algo/round6-res-split` (refactor + Phase 0)
**Sub-branches** (audit-only): `algo/r6-v1-conservative` (blend 0.3),
  `algo/r6-v2-stronger` (blend 0.1)
**Date**: 2026-05-04
**Outcome**: REFACTOR shippable on its own (5 stage methods, bit-exact
parity verified). ARTIFACT-AWARE LIFT hypothesis disproven by v1+v2 — no
sub-branch passes shipping criteria. R5 v2 remains the largest-gain
mask-based variant (still Pareto-bound).

## Phase 1 — RES Responsibility Split (refactor only)

Split `ResFilter.process()` into 5 private stage methods on the same class:

- `_stage_residual_model` — residual_echo_psd + 4 caps + reverb + echo boost
- `_stage_gain_compute` — ENR / Wiener / spectral_sub + EMR + spectral floor
- `_stage_gain_postprocess` — EPC_DT cap + quiet mask + 3-bin + HF + divergence
- `_stage_temporal_smoothing` — split attack/release + rate limit + LF + render ceil
- `_stage_noise_floor_and_cng` — noise_psd tracker + noise-floor lift + CNG + IFFT/OLA

`process()` orchestrates: setup → coh2/PSD update → fs_confidence + ne_g_floor +
spectral_g_min compute → 5 stage calls → stats/diag tail.

**Parity verification** (`python/parity_round6.py`):
- 5-case fixture (FS_st_worst, FS_mv_loss, DT_mv_worst, DT_st_typ, NE)
- Per-frame output SHA256 hash bit-exact across all 5 cases
- All 28 captured `_diag` keys bit-exact
- Verified after each of the 5 extractions individually + after final state

Refactor commit: `9040bb9`. Behavior-preserving — safe to ship independently
of any algorithm change.

## Phase 2 — Artifact-Aware Noise Lift (variant testing)

### Hypothesis

R5 found mask-based lift bypass produces Pareto-bound (DT_mv deg +0.023 / FS
echo -0.042). Hypothesis for R6: smooth-vs-rough gain trajectory distinguishes
"artifact-safe suppression" from "artifact-prone over-suppression". The
noise-floor lift on smooth bins is unnecessary protection; on rough bins
(isolated notch / spectral roughness / temporal jitter) the lift IS
preventing real artifacts. Selectively reduce lift only on smooth +
echo-dominant bins.

### Implementation (R6 v1 commit `a17d96d`)

In `_stage_gain_compute` (after EMR):
```python
echo_dom_mask = (g_softgate < 0.20) & (res/err > 1.0)
                & (far_power > 1e-4) & (effective_dt < 0.45)
```

In `_stage_noise_floor_and_cng`:
```python
spectral_rough = |g[k] - 0.5*(g[k-1]+g[k+1])|        # < 0.10 → smooth
isolated_notch = (g[k] < 0.6*min(g[k-1],g[k+1])) & (min > 0.1)
temporal_delta = |g[k] - prev_final_g[k]|             # < 0.15 → smooth
is_smooth = spec_OK & temp_OK & not_isolated

blend = where(echo_dom & is_smooth, safe_blend, 1.0)
gain_smooth = blend * full_lift + (1-blend) * pre_lift_g
```

Trace fields (audio-passive, surfaced as `r6_*` in `_diag`):
`noise_lift_amount_voice`, `spectral_roughness_voice`,
`isolated_notch_pct_voice`, `temporal_delta_voice`, `echo_dom_mask_voice_pct`.

### Bench results

| Variant | safe_blend | FS_st e | FS_mv e | DT_mv e | DT_mv d | NE d | w20 d | impr |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| baseline_v381_seeded | (n/a) | 0 | 0 | 0 | 0 | 0 | (locked) | (locked) |
| R5 v2 (full bypass) | n/a | -0.042 | -0.039 | +0.074 | +0.023 | -0.004 | +0.106 | 15/20 |
| **R6 v1** | 0.3 | -0.049 | -0.046 | +0.003 | -0.026 | -0.001 | +0.022 | 7/20 |
| **R6 v2** | 0.1 | **-0.103** | **-0.109** | -0.022 | -0.012 | -0.003 | +0.088 | 15/20 |

Acceptance gates (DT_mv deg ≥ +0.02 / FS regress < 0.02 / NE drop ≤ 0.01):
- R6 v1: ALL fail except NE
- R6 v2: ALL fail except NE

### Why the smoothness gate fails

Mathematically R6 v2 with `safe_blend=0.1` is **more aggressive** than R5 v2's
full bypass. R5 v2 kept `pre_lift_g` on echo-dom bins. R6 v2 keeps
`0.9 * pre_lift_g + 0.1 * full_lift` only on (echo_dom & smooth) bins. Since
`pre_lift_g <= full_lift`, the blend produces a slightly higher gain than R5
v2's full bypass — but only on the smooth subset. The math:

| Variant | gain on echo_dom bins | gain on FS smooth-echo bins |
|---|---|---|
| R5 v2 | `pre_lift_g` (full bypass) | `pre_lift_g` |
| R6 v1 | `0.3*full_lift + 0.7*pre_lift_g` (smooth) / `full_lift` (rough) | `0.3*full + 0.7*pre` |
| R6 v2 | `0.1*full_lift + 0.9*pre_lift_g` (smooth) / `full_lift` (rough) | `0.1*full + 0.9*pre` |

On FS, all echo bins are "smooth" (no NE jitter to make them rough). So the
smoothness gate doesn't EXCLUDE FS — it only changes the per-bin blend ratio.
Strong blend (v2) over-suppresses FS more than R5 v2; weak blend (v1)
under-suppresses DT.

Net: smoothness gate is a knob on the same Pareto curve as R5 v2's blend, not
a way to escape it.

### Plan v3 / v4 not run

R6 plan listed:
- v3: artifact-aware lift WITHOUT A4 — N/A (R6 didn't add A4 in the first place)
- v4: artifact-aware lift + A4 — would map another point on the same curve.
  Since v1 and v2 both bracket the Pareto curve and confirm no isolation,
  v4 = (v1 result + small A4 echo improvement at no FS cost) ≈ R5 v2 + smoothness gate. Predicted not to ship; saved 17 min × 2 evals.

## Cumulative across Round 1–6

| Round | Best variant | DT_mv Δdeg (bucket) | FS_st Δecho | Verdict |
|---|---|---:|---:|---|
| 1 | F3 usable_linear gate | +0.002 | sub-noise | null |
| 2 | Phase 3 ERLE reset | 0 | n/a | null seeded |
| 3 | D2 unconditional | -0.139 | -0.058 (+ FS_mv -0.108) | Pareto fail |
| 3 | D3 (0.25) | +0.005 | -0.003 | sub-acceptance |
| 4 | R1 v1 (cap=0.5) | 0 | 0 | null |
| 4 | R1 v2 (cap=0.3) | -0.000 | +0.000 | null |
| 5 | A34 v1 (full A3, wide) | -0.139 | -0.068 | catastrophe |
| 5 | A34 v2 (full A3, narrow) | **+0.023** | -0.042 | DT pass / FS fail |
| 5 | A34 v3a (A3-only) | +0.022 | -0.059 | A4 not damage source |
| 5 | A34 v3b (soft A3 50%) | -0.021 | +0.021 | FS pass / DT fail |
| 6 | v1 (artifact blend 0.3) | -0.026 | -0.049 | hypothesis disproven |
| 6 | v2 (artifact blend 0.1) | -0.012 | -0.103 | hypothesis disproven, more aggressive |

Across 16 ablations / 6 rounds:
- Largest DT_mv echo signal: R5 A34 v1 +0.091 (Pareto-bound)
- Largest DT_mv deg signal: R5 A34 v2 +0.023 (FS bound)
- Largest worst-20 directional improvement: R5 A34 v2 (15/20, mean +0.106)
- **No variant satisfies all hard gates simultaneously.**

## What Round 6 added that prior rounds didn't

1. **Refactored RES pipeline** (`9040bb9`). Same behavior, but every stage
   is now an isolated method with a named interface, capturable independently.
   Future R&D can target one stage without grep-archaeology. Shippable to
   main on its own merit.

2. **`python/parity_round6.py`** — 5-case fixture per-frame parity harness
   (output SHA256 + 28 diag values). Rerunnable for any future ResFilter
   refactor. Self-test deterministic.

3. **R6 trace plumbing** (`a17d96d`) — 5 audio-passive `r6_*` diag fields
   exposing the artifact metrics (kept on round6 base). Useful for future
   debugging even though R6 v1/v2 didn't ship.

4. **Definitive Pareto wall confirmation**: 4 rounds of mask-based and
   1 round of artifact-aware approaches all hit the same wall. The
   noise-floor lift is necessary at full strength for AECMOS-perceived
   audio quality on FS; reducing it on echo-dominant bins (regardless of
   smoothness) creates artifacts in proportion to reduction strength.

## Disposition

- **Refactor commit `9040bb9`**: ship-ready. Recommended merge target:
  `main` (after standard parity bench on 800-case to triple-confirm).
- R6 v1, v2 commits: audit-only, NOT for merge. Hypothesis disproven.
- Round 6 base branch retains refactor + R6 trace plumbing + parity harness.

## Recommendation

1. **Ship the refactor only** — merge `algo/round6-res-split` HEAD (refactor +
   parity harness + R6 trace plumbing) to main after running 800-case bench
   for one final parity confirmation. Algorithm behavior unchanged.
2. **R6 mask + artifact gate**: NOT for shipping. Same conclusion as R5 — the
   classical suppressor is at Pareto on this dataset. Future signal must come
   from upstream (filter trajectory) or from a learned postfilter.
3. **R6 trace plumbing kept**: future R&D can use `r6_*` diag fields for
   per-frame artifact analysis even without enabling the lift logic
   (`r6_artifact_aware_enable=False` is a runtime toggle).

`baseline_v381_seeded` remains canonical reference. v3.8.1 algorithm
unchanged; v3.8.2 C-port unaffected.

---

# Round 7 — Signal-Flow Audit + Filter Trajectory + Output Mix

**Branch**: `algo/round7-signal-split-trajectory` (from `9040bb9` R6 refactor base)
**Sub-experiments**:
  `experiments/round7_phase0/` (P0 trace + verdict),
  `experiments/r7_p1a_epv_thr8/` (R7.1a EPV damping),
  `experiments/r7_p2_mix_r{05,10,20}/` (R7.2 offline mix)
**Date**: 2026-05-04
**Outcome**: NO MERGE. R7 confirmed Phase 0 finding (filter trajectory IS
materially different between worst- and best-DT_mv, effect size up to 1.21)
but every actionable lever inside plan scope produced NULL or Pareto-bound
result. Round 7 closed without ship candidate. Filter trajectory ROOT
(why W never grows on worst cases) is the only remaining mechanism;
plan explicitly out of scope.

## Phase 0 — Filter trajectory audit (GO)

800-case R7 trace + upgraded gate (effect_size ≥ 0.5 + abs floor for
continuous; hit-rate ≥ 6/20 + ≥2× for events; sample-count guard for
post-event). 6 fields pass on DT_movement worst-20 vs best-20:

| field | worst-20 | best-20 | rule | effect / hit |
|---|---:|---:|---|---|
| `r7_shadow_w_norm_mean` | 8.80 | 30.79 | continuous | effect=1.21 |
| `r7_main_err_smooth_mean` | 176.4 | 647.3 | continuous | effect=1.05 |
| `r7_filter_w_norm_mean` | 7.16 | 17.83 | continuous | effect=0.98 |
| `r7_shadow_err_smooth_mean` | 187.8 | 583.9 | continuous | effect=0.92 |
| `r7_mu_scale_mean` | 0.795 | 0.716 | continuous | effect=0.65 |
| `r7_event_epv_count` | 1.40 | 0.40 | event-count | 11/20 vs 5/20 (2.2×) |

**Mechanism reading**: worst-DT_mv filter sits at ~40% of best filter's
norm. mu_scale is pushed harder yet W stays small. EPV fires 2.75× more
often. Signal-energy regime on worst cases is itself smaller; filter
trajectory is structurally different, not just "noisier".

P0 also confirmed `delay_first` / `delay_shift` events fire 0 times across
800-case blind set → drop from Phase 1 priority. Phase 1 reordered to
EPV first.

## R7.1a — EPV weak-filter false-positive damping (NULL)

Variant: env `R7_EPV_WEAK_THR=8.0`. At EPV trigger site, when
`|main filter W|_2 < THR`, skip the entire EPV response (Q_high,
P_max=1.0/30, P_floor=1.0/30, mark_diverged, render_forced, erl_clamp).
Default-off → bit-exact parity vs `baseline_v381_seeded` (verified via
`parity_round6`).

| Bucket | Δecho | Δdeg |
|---|---:|---:|
| FS_static | +0.001 | +0.000 |
| FS_movement | -0.000 | +0.000 |
| DT_static | -0.001 | +0.002 |
| **DT_movement** | -0.001 | **+0.001** |
| NE | +0.000 | +0.000 |

All deltas inside ±0.005 noise floor. DT_mv deg gain 20× below
acceptance gate.

**Mechanism conclusion**: EPV is **symptom**, not cause. Smoke trace
showed wHmBm7VHfk (worst-20 case): baseline 3 EPV fires → THR=8 gave 5
raw fires all suppressed, but `filter_w_norm` only changed 3.89 → 3.88.
Filter doesn't grow even when EPV resets are suppressed. The P0 EPV
separator describes a downstream artifact of the small-filter regime,
not the lever holding W small.

## R7.2 — Offline upper-bound mix (Pareto-bound, NO RATIO PASSES)

Method: `mixed = (1-r) * ours + r * ours_nores`, blanket per-case,
re-scored. 3 ratios = 5%, 10%, 20%. Source = R7.1a eval_output (= baseline
within 0.001).

| ratio | DT_mv Δdeg | DT_mv Δecho | FS_st Δecho | FS_mv Δecho | NE Δdeg |
|---|---:|---:|---:|---:|---:|
| 5%  | **+0.063** | -0.108 | **-0.065** | -0.012 | -0.010 |
| 10% | +0.158 | -0.240 | -0.228 | -0.142 | -0.018 |
| 20% | +0.431 | -0.483 | -0.507 | -0.473 | -0.022 |

Acceptance gates (user spec, w/o DT echo gate):

| gate | 5% | 10% | 20% |
|---|:-:|:-:|:-:|
| DT_mv deg ≥ +0.02 | ✓ | ✓ | ✓ |
| FS_st echo regress < 0.02 | **✗ (-0.065, 3.25×)** | ✗ | ✗ |
| FS_mv echo regress < 0.02 | ✓ | ✗ | ✗ |
| NE deg drop ≤ 0.01 | ✓ | ✗ | ✗ |
| worst20 ≥10/20 | ✓ (11/20) | ✓ | ✓ |

5% passes 4/5 user gates; FS_st fails by 3.25×. Top-1 worst DT_mv case
(XV5L2, base 1.16) only moves +0.006 deg even at 5% — not all worst cases
have headroom in their nores audio.

Including plan §1 acceptance (`DT_mv echo regress < 0.02`): all 3 ratios
fail by 5–22×.

**Pareto reading**: the FS_st loss at 5% mix matches expected algebra —
nores has +1.15 echo penalty on FS_static (R5 Phase 0 binding audit:
nores echo 2.62 vs ours 3.77). Mixing 5% nores adds 0.05 × 1.15 ≈ 0.058
echo penalty on FS_st, which matches the observed -0.065. Conditional
mix can in principle reduce this by gating to DT-only frames, but:

1. Top-1 worst case unmoved by any ratio → conditional mix adds no new
   headroom there.
2. The FS bleed at 5% blanket suggests current per-frame DT detector
   isn't selective enough; tighter gate would also kill DT gain.
3. In-process implementation cost (dual OLA buffer in ResFilter) is the
   most expensive R7 step per plan §2.3.

R7.2 STOP per user pre-condition: "若三個 ratio 全 fail，寫 SUMMARY_round7".

## Architectural finding kept (P0 trace plumbing)

Audio-passive trace fields added at `AEC.process` tail
([aec.py:4419+](AEC/python/aec.py#L4419)) — 14 new `_diag` keys:
delay tracking, event markers, P-override countdowns, raw/nores/final
output power, res_required_gain, nores_echo_proxy. parity_round6 5-case
fixture stays bit-exact with default env.

Plus mechanism trace for R7.1a (`epv_event_raw`, `epv_event_suppressed`,
`filter_w_norm` instantaneous). Default off (env-toggle), no parity risk.

`trace_phase0.py` extended with R7 collection: 9 far-active continuous
signals + per-source 30-frame post-event inst_erle response windows.
Added `--jobs/-j` ProcessPool wrapper for parallel runs.

Architectural signal-overload audit table delivered (plan body, not
re-implemented as P0.5 split). Skipped P0.5 mandatory split per user
guidance: P1 EPV ablation didn't need it; if any future variant hits
coupling, do targeted split then.

## Cumulative across Round 1–7

| Round | Phase / Variant | DT_mv Δdeg | DT_mv Δecho | Verdict |
|---|---|---:|---:|---|
| 1 | F4 (cap 4→2) | -0.004 | n/a | null |
| 1 | F1 (dt_residual_scale × 0.7) | -0.004 | n/a | bimodal |
| 1 | F3 (usable_linear_v1 gate) | +0.002 | n/a | sub-noise |
| 2 | Phase 3 (ERLE/RES reset on transitions) | +0.000 | n/a | null seeded |
| 3 | D2 unconditional | -0.139 | +0.011 | FS catastrophe |
| 3 | D2b once_conv-gated | -0.002 | n/a | lost DT, FS still bad |
| 3 | D3 (0.15) | +0.003 | n/a | sub-acceptance |
| 3 | D3b (0.25) | +0.005 | n/a | FS_mv -0.014 |
| 4 | R1 v1 cap=0.5 | +0.000 | -0.000 | null |
| 4 | R1 v2 cap=0.3 | -0.000 | -0.000 | null |
| 5 | A34 v1 (full A3, wide mask) | -0.139 | +0.091 | FS catastrophe |
| 5 | A34 v2 (narrowed mask) | +0.023 | +0.074 | DT pass, FS fail (-0.04) |
| 5 | A34 v3a (A3-only) | +0.022 | +0.062 | DT pass, FS fail (-0.05) |
| 5 | A34 v3b (soft A3 50%) | -0.021 | +0.051 | FS pass, DT fail |
| 6 | refactor-only (R6) | bit-exact | bit-exact | merge-ready |
| **7** | **R7.1a EPV THR=8** | **+0.001** | **-0.001** | **null** |
| **7** | **R7.2 mix 5% blanket** | **+0.063** | **-0.108** | **FS_st fail (-0.065)** |
| **7** | **R7.2 mix 10% blanket** | +0.158 | -0.240 | FS+NE fail |
| **7** | **R7.2 mix 20% blanket** | +0.431 | -0.483 | FS+NE fail |

Across 17 ablations / 7 rounds:
- No variant satisfies the spec gates (DT deg ≥+0.02, FS echo ≥-0.02,
  NE deg drop ≤0.01) simultaneously.
- R5 v2 came closest to the deg side but failed FS by 2× the limit.
- R7.2 5% mix is the upper bound of case-level mixing — fails FS_st by
  3.25×.

## What R7 added that earlier rounds didn't

1. **Direct filter trajectory measurement** on 800-case: confirmed
   worst-DT_mv filter sits at 40% of best filter's W norm with effect
   size 0.98–1.21. This is the strongest single signal-level finding
   in 7 rounds.
2. **EPV separator validated as symptom**: P0 hit-rate 11/20 vs 5/20
   was confirmed correlative, not causal (R7.1a NULL).
3. **Mix headroom mapped**: 5% blanket mix is the upper bound for
   case-level intervention; +0.063 deg / -0.065 FS_st echo / -0.108
   DT echo. No conditional mix can exceed this.
4. **Trace plumbing kept** as audit-passive on R7 base for any future
   investigation.
5. **Plan correction in P0**: discovered `delay_first` / `delay_shift`
   events fire 0 times on this dataset, so any plan that prioritizes
   delay-shift handling is dead on this benchmark.

## Why R7 cannot break Pareto with in-scope levers

P0 strongest separator is `filter_w_norm` (effect 0.98). The only
mechanisms that can move filter W are inside the PBFDKF kernel:
mu/Q derivation, P_max/P_floor envelope, gradient computation. Plan
§1.3 don't-do list explicitly excludes "shadow NLMS / PBFDKF kernel".

R7 levers in scope (transition-window adaptation, frame-level mix) are
all downstream of the kernel. They can:
- Dampen kernel response to specific triggers (R7.1a tested: NULL because
  filter wasn't being held back BY the trigger response — kernel just
  couldn't grow on that signal regime)
- Blend output with linear-only nores (R7.2 tested: Pareto-bound; FS
  bleed at any non-trivial ratio)

Neither changes the kernel's W trajectory itself.

## Disposition

- All R7 sub-branches retained as audit on `algo/round7-signal-split-trajectory`.
  No merge to main.
- `baseline_v381_seeded` remains canonical.
- v3.8.1 is at Pareto on classical-suppressor architecture.
  Integration-and-ship plan default per Round 4 conclusion still holds.
- R7 trace plumbing kept on the branch (audio-passive); useful diagnostic
  if any future investigation revisits filter trajectory.

## Recommendation for future rounds

R7 establishes that **the filter trajectory itself is binding**, not
the suppressor stage. Levers that don't touch the kernel cannot move
the worst-DT_mv deg score by ≥+0.02. Three out-of-scope options remain:

1. **NN postfilter** (TEACAEC / DTLN-AEC): separate plan
   `~/.claude/plans/jazzy-brewing-castle.md`. Only known path to break
   Pareto on this dataset.
2. **PBFDKF kernel rewrite**: spectral-trajectory-aware mu derivation,
   per-bin process-noise floor that depends on signal-regime not just
   error magnitude. Structural; multi-week.
3. **Filter W warmup heuristic**: special-case very-small-W early
   frames to seed W from a longer-time average / from shadow filter
   when shadow has converged but main hasn't. Lower-risk than (2)
   but still touches kernel state — out of scope per plan don't-do list.

If the user accepts in-scope exhausted, recommend route 1 (NN postfilter)
as the only remaining lever with realistic ship potential.

---

# Round 8 Pre-flight — Trace Correctness Re-evaluation (closed)

**Branch**: `algo/round8-prep-trace-cleanup` (audit-only, NOT MERGED — discarded after verdict)
**Date**: 2026-05-04
**Outcome**: STOP R8 main line. R7's "filter doesn't grow" causal claim
falsified by normalized re-evaluation. Worst-DT_mv `deg` loss is in RES,
not the linear AEC kernel — re-confirms R5/R6/R7 Pareto with corrected
measurements.

## Why R8 was opened

R7's strongest separator was `filter_w_norm` (worst 7.16 vs best 17.83,
effect 0.98) combined with HIGHER `mu_scale` on worst (0.795 vs 0.716).
The reading was: "controller asks for more learning, but actual W
doesn't materialize". Codex direction: P/Q/R effective-update audit.

## Why the premise was wrong (5 measurement findings)

Code review of R7 plumbing surfaced confounds invalidating the causal
direction:

1. `nores_echo_proxy = sqrt(nores_pwr/mic_pwr)` is a power ratio, NOT an
   echo proxy. DT/CNG/silence confound it.
2. `filter_w_norm` separator is signal-level confounded with
   `main_err_smooth` (both small on worst → small-signal regime, not
   necessarily learning failure).
3. CLI `--enable-res / --cng` `store_true` defaults silently overrode
   BALANCED preset's `enable_res=True / enable_cng=True`.
4. eval `--filter` default 2048 ≠ baseline 832 (52 ms @ 16 k).
5. R7 trace lazy state (`_r7_prev_delay`, `_r7_prev_div_counts`) not
   cleared in `AEC.reset()`.

## Phase A — trace correctness fixes (built on branch, then discarded)

Implemented as audit-only / additive plumbing, parity_round6 5-case
bit-exact PASS:

- Renamed power-ratio fields, kept aliases one cycle:
  `nores_to_mic_power_ratio`, `final_to_nores_power_ratio`
- Added real echo proxies: `inst_erle_db`, `nores_inst_erle_db`,
  `echo_psd_to_error_psd_ratio`, `far_to_mic_xcorr0`
- Added level-corrected: `filter_w_norm_per_sqrt_far`,
  `predicted_echo_norm`
- Extended PBFDKF `_kx_trace` (gated, zero-diff when off): K, P, R,
  denom, dW_norm, X_norm, E_norm, predicted_echo_norm, ...
- CLI `--enable-res / --cng` switched to `BooleanOptionalAction
  default=None` so preset values govern when flag absent
- `eval_aec_challenge.py --filter` default 2048 → 832
- `AEC.reset()` clears R7 trace lazy state

Branch was discarded after verdict — these fixes are NOT on main and
would need to be re-applied if a future round wants the trace plumbing.
Findings + fix list recorded above so they don't have to be re-discovered.

## Phase B — light-trace re-evaluation on normalized fields

800-case trace, DT_movement worst-20 vs best-20 (rank-locked by
baseline_v381_seeded.deg). Effect-size + abs-floor gate:

| field | worst | best | effect | gate |
|---|---:|---:|---:|---|
| `filter_w_norm_per_sqrt_far` | 63 664 | 153 117 | **1.00** | PASS |
| `nores_inst_erle_db` (linear AEC alone) | **1.66 dB** | 0.72 dB | 0.31 | sub-gate |
| `inst_erle_db` (full pipeline) | **13.54 dB** | 10.02 dB | 0.45 | sub-gate |
| `echo_psd_to_error_psd_ratio` | 5.50 | 3.72 | 0.21 | sub-gate |
| `\|far_to_mic_xcorr0\|` | 0.231 | 0.215 | 0.13 | sub-gate |
| (R7 reference) `r7_filter_w_norm` | 7.16 | 17.83 | 0.98 | PASS |

**Three answers:**

1. **Level-corrected W still separates** (effect 1.00). Worst W is
   small in both absolute and per-√far-energy terms — the differential
   is real, not a pure signal-regime artifact.
2. **Linear filter is NOT worse on worst — it's slightly better.**
   Worst-20 nores ERLE 1.66 dB > best-20 0.72 dB. Full pipeline ERLE
   also higher on worst (13.5 vs 10.0). `echo_psd_to_error_psd_ratio`
   higher on worst (5.5 vs 3.7) → filter's echo-path estimate explains
   MORE of the residual on worst. Smaller W is the *correct* equilibrium
   for the smaller-far-energy regime; nothing more to learn.
3. **Reference/mic coupling is fine.** xcorr0 worst 0.231 vs best
   0.215 (effect 0.13) → delay tracker / SNR / reference path is not
   the differentiator.

## Where the worst-DT_mv `deg` loss lives

| stage | worst-20 finding | conclusion |
|---|---|---|
| reference / delay | xcorr0 same as best | not the issue |
| linear AEC ERLE | **higher** than best | working well, possibly over-working |
| nores `deg` (R5 P0) | best-of-pipeline | RES is what removes it |
| RES output `deg` | worst | RES suppression damages NE |

R7's `filter_w_norm` separator described the *symptom* (small filter
matches small far energy → strong linear ERLE → RES still triggers
DT-aggressive suppression → NE damaged). R8 Codex premise (P/Q/R
audit) was based on the wrong causal reading.

## Disposition

- Branch `algo/round8-prep-trace-cleanup` discarded (no merge to main).
  Phase A trace-correctness fixes are NOT on main; recorded above for
  future re-application if needed.
- No Phase 1 algorithm changes attempted (premise falsified).
- v3.8.1 `baseline_v381_seeded` remains canonical.
- Heavy Kalman `_kx_trace` decomposition (drafted but not run) skipped:
  Q2 says linear AEC is fine; decomposing its update path won't reveal
  a fixable lever.

## Recommendation (unchanged from R7)

NN postfilter (separate plan) is the remaining route to break Pareto
on this dataset. Classical-suppressor architecture is at Pareto.

---

# Round 9 — AEC3-style FilterQualityState (closed at STOP-1)

**Branch**: `algo/round9-filter-quality-state` (audit-only, NOT MERGED — discarded after verdict)
**Date**: 2026-05-04
**Outcome**: STOP-1 fired at end of Phase 0. R8-prep warning ("purely
filter-side state may not separate") confirmed empirically. R9 closed
without Phase 1; the FilterQualityState cannot sort worst-DT_mv from
best-DT_mv on the cases the bench cares about.

## Why R9 was opened

Architectural problem identified across rounds: ResFilter / residual
estimator / suppressor each re-derive policy from raw flags
(`effective_dt`, `epc_active`, `_filter_converged`, `using_render_based`,
`divergence`). One signal change at one site gets reinterpreted at
another. R9's thesis: centralize into a single `FilterQualityState`
whose categorical output (`residual_policy`) is the only policy input
those consumers see.

Codex direction: AEC3-style state architecture; do NOT touch shadow
NLMS / NN / RES output mix / kernel.

## Phase 0 design (audit-passive plumbing, all discarded)

Extended `AecState` ([aec.py:2696](AEC/python/aec.py#L2696)) in-place
with `compute_quality(frame_count, warmup_frames, erle_factor,
nores_inst_erle_db, echo_psd_to_error_psd_ratio, dt_combined,
epc_event_fired, delay_shift_event, shadow_rise_event)` and 7
properties:

- `linear_usable_for_residual` = `once_conv ∧ conv ∧ ¬epc ∧ div<0.3`
- `linear_usable_for_output` = + `filter_quality_score≥0.5 ∧ ¬transition_recent`
- `initial_state` = `(¬once_conv) ∨ frame<warmup+50`
- `transition_recent` = transition_window_frames > 0 (30-frame window
  armed on EPC fire / delay_shift / shadow_rise)
- `transparent_mode` = `linear_usable_for_output ∧ low echo risk ∧
  high NE confidence ∧ dt_combined<0.2`
- `filter_quality_score` ∈ [0,1] = weighted erle_factor +
  nores_inst_erle_db + once·conv + (1−div)
- `residual_policy` ∈ {LINEAR_TRUSTED, LINEAR_BUT_CAP, RENDER_FALLBACK,
  TRANSITION_HOLD, TRANSPARENT_NEAR}, priority dispatch

Plus the four R8-prep echo proxies (re-applied only on this branch):
`inst_erle_db`, `nores_inst_erle_db`, `echo_psd_to_error_psd_ratio`,
`far_to_mic_xcorr0`. trace_phase0 extended to collect all `r9_*` /
`qstate_*` fields.

parity_round6 5-case fixture: PARITY PASS bit-exact vs golden built
on branch HEAD before any change (audit-only / additive).

## Phase 0 result — STOP-1 fired

DT_movement worst-20 vs best-20, rank-locked by baseline_v381_seeded.deg.
PROCEED criterion: ≥1 critical bucket with |Cohen's d| ≥ 0.5 AND
|Δfraction| ≥ 0.10.

| critical bucket | worst | best | |Δ| | effect | gate |
|---|---:|---:|---:|---:|---|
| `RENDER_FALLBACK` fraction | 90.7 % | 92.5 % | 0.018 | 0.11 | ✗ |
| `TRANSITION_HOLD` fraction | 6.6 % | 4.2 % | 0.024 | 0.25 | ✗ |
| `transition_recent` fraction | 6.6 % | 4.2 % | 0.024 | 0.25 | ✗ |

0/3 critical PASS. 0/11 non-critical PASS. The largest non-critical
effect was `r9_initial_state_pct` (effect 0.49, |Δ| 0.213) but
direction is reversed (best has MORE initial-state frames — likely
shorter clips, not a policy separator).

**All 20 worst-DT_mv AND all 20 best-DT_mv have dominant policy =
`render_fallback`**, with most cases in render_fallback ≥ 80 % of
frames. The state cannot distinguish them.

## Mechanism reading

`linear_usable_for_residual` requires `once_conv AND conv AND not epc
AND div<0.3`. DT_mv cases rarely satisfy this — DT density and
movement-driven `mark_diverged` events keep both populations out of
`linear_usable_for_residual` most of the time. Both worst and best
route to RENDER_FALLBACK.

The differentiator on DT_mv `deg` is therefore NOT "which residual
source to use". Both populations use the same source. The differentiator
is "how aggressively the suppressor cuts NE within RENDER_FALLBACK
mode", which is exactly the R5/R6/R8-prep finding: RES is the binding
stage. State routing isn't the lever.

R8-prep verdict had explicitly flagged this: "purely filter-side state
may not separate". R9 P0 confirmed.

## Disposition

- Branch `algo/round9-filter-quality-state` discarded (no merge to
  main). The FilterQualityState plumbing + 4 P0a echo proxies are NOT
  on main; recorded above for future re-application if a different
  state-side hypothesis emerges.
- No Phase 1 algorithm changes attempted (premise falsified).
- v3.8.1 `baseline_v381_seeded` remains canonical.

## Cumulative across Round 1-7 + R8 prep + R9 P0

| Round | Outcome | Notes |
|---|---|---|
| 1 | F1-F4 all null | trigger style invalidated |
| 2 | Phase 3 / state-routing all null | cold-start / EPC unconditional fail |
| 3 | D2 catastrophic / D3 sub-acceptance | EPC-gated bypass not viable |
| 4 | R1 v1+v2 null | per-bin cap erased downstream |
| 5 | A34 v2 closest (+0.023 deg, FS −0.04) | Pareto curve mapped |
| 6 | refactor-only shippable, lift disproven | RES split clean |
| 7 | EPV damp NULL; mix Pareto-bound | filter trajectory hypothesis raised |
| R8 prep | R7 causal direction falsified | linear AEC fine on worst |
| **R9** | **STOP-1: state cannot sort worst from best** | **both populations dominated by RENDER_FALLBACK** |

## Recommendation (unchanged)

NN postfilter (separate plan) remains the only realistic ship route
to break Pareto on this dataset. Classical-suppressor architecture
is at Pareto across every in-scope lever, including state routing.

---

# Round 10 — WebRTC SuppressionGainCandidate (closed at Phase 0 STOP)

**Branch**: `algo/round10-webrtc-suppression-gain` (audit-only, NOT MERGED — discarded after verdict)
**Date**: 2026-05-04
**Outcome**: STOP at Phase 0. WebRTC-style "no-audible-echo" candidate
gain consumes the same input pool as the current pipeline
(`residual_echo_psd`, `error_psd`, `noise_psd`); inherits the same
residual-model over-estimation bias on worst-DT_mv, producing
DEEPER suppression on the cohort RES is already over-suppressing.
Same R5/R6 Pareto axis confirmed once more.

## Why R10 was opened

Codex direction: rebuild the suppressor's gain calculation using
WebRTC `SuppressionGain` shape — `weighted_echo_audibility +
comfort_noise + nearend → no-audible-echo gain` instead of the current
`residual_echo / nearend → ENR softgate → noise-floor lift`. Phase 0
strict gate: candidate must have DIFFERENT shape from current (per-
cohort direction or magnitude differential), not just a tighter/looser
knob on the same lever.

Don't-do list: no shadow NLMS / kernel / delay estimator / NN / direct
suppressor replacement / grid search.

## Phase 0 plumbing (audit-only, all discarded)

`ResFilter._r10_compute_candidate_gain()` runs ALONGSIDE the current
pipeline after `_stage_noise_floor_and_cng` returns. Never mutates
audio state; writes diagnostic stats to `self._r10_diag`.

Formula (intentionally simple to verify shape, not curve-fit):

```
weighted_echo = max(residual_echo_psd - 1.5·noise_psd, 0)
nearend_proxy = max(error_psd - residual_echo_psd, eps)
target_echo   = noise_psd · 1.0
gain = sqrt((nearend + target_echo) / (nearend + weighted_echo + eps))
gain = clip(sqrt(gain), g_min, 1.0)
```

13 voice-band stats per frame: gain_candidate / gain_current /
weighted_echo / comfort_noise / nearend_proxy / min/max gain /
candidate_delta voice mean + abs + p10 + p90 / higher% / lower%.

`AEC.process` trace tail wires fields to `_diag` with `r10_*` prefix.
`trace_phase0.py` collects 13 fields on far-active frames (same
gating as R5 stage gains).

parity_round6 5-case fixture: PARITY PASS bit-exact vs golden.

## Phase 0 result — STOP

5 cohorts × 20 cases (rank-locked by baseline_v381_seeded; FS by echo,
DT/NE by deg).

| cohort | n | gain_curr | gain_cand | Δ mean | tighten% | relax% |
|---|---:|---:|---:|---:|---:|---:|
| **DT_mv worst-20** | 20 | 0.535 | 0.302 | **-0.232** | **74.4 %** | 12.2 % |
| DT_mv best-20  | 20 | 0.581 | 0.515 | -0.065 | 51.1 % | 29.5 % |
| FS_st worst-20 | 20 | 0.458 | 0.421 | -0.037 | 58.9 % | 32.0 % |
| FS_mv worst-20 | 20 | 0.475 | 0.396 | -0.079 | 62.4 % | 23.3 % |
| NE worst-20    | 20 | 0.138 | 0.096 | -0.042 | 17.2 % |  5.8 % |

| cohort | weighted_echo | nearend_proxy | echo:near ratio |
|---|---:|---:|---:|
| **DT_mv worst** | 12.21 | 0.84 | **14.5×** |
| DT_mv best  | 3.97 | 4.45 | 0.89× |
| FS_st worst | 9.49 | 5.59 | 1.70× |
| FS_mv worst | 11.21 | 2.86 | 3.92× |
| NE worst    | 0.007 | 0.036 | 0.19× |

Both Phase 0 gates fail:

- **Direction differential**: ALL 5 cohorts have negative Δ (uniform
  sign — not regime-aware policy).
- **Worst-20 should preserve speech, not over-suppress**: candidate
  tightens worst-DT_mv 3.6× MORE than best-DT_mv. Wrong direction.

## Mechanism reading

The formula consumes the SAME input pool the current pipeline consumes.
On worst-DT_mv, `residual_echo_psd` is large (over-estimated by RES
residual model — the R5/R6/R8/R9 binding stage). That over-estimation:

1. Inflates `weighted_echo` (worst 12.21 vs best 3.97).
2. Shrinks `nearend_proxy = error_psd − residual_echo_psd` (worst 0.84
   vs best 4.45 — DT_mv worst nearend looks 5× weaker than best in the
   formula's eyes).
3. echo:nearend ratio on worst-DT_mv hits 14.5× — formula concludes
   "echo dominates, compress hard" → gain 0.30 (vs current 0.54).

The candidate is structurally the same lever as current, just a
**stricter** mapping of the same inputs. Inherits the residual-model
bias and produces deeper suppression on the cohort already
over-suppressed. This is the R5/R6/R8/R9 finding once more: the
binding stage is the **residual echo model** (which sets
`residual_echo_psd`), NOT the suppressor's gain formula. Rewriting
the formula on top of the same residual estimate cannot break Pareto.

## Disposition

- Branch `algo/round10-webrtc-suppression-gain` discarded (no merge to
  main). The R10 candidate plumbing is NOT on main; recorded above
  for future re-application if a different residual-model-side
  hypothesis emerges.
- No Phase 1 / 2 / 3 algorithm changes attempted (premise falsified
  at strict Phase 0 gate).
- v3.8.1 `baseline_v381_seeded` remains canonical.

## Cumulative across Round 1-10

| Round | Outcome | Notes |
|---|---|---|
| 1 | F1-F4 all null | trigger style invalidated |
| 2 | Phase 3 / state-routing all null | cold-start / EPC unconditional fail |
| 3 | D2 catastrophic / D3 sub-acceptance | EPC-gated bypass not viable |
| 4 | R1 v1+v2 null | per-bin cap erased downstream |
| 5 | A34 v2 closest (+0.023 deg, FS −0.04) | Pareto curve mapped |
| 6 | refactor-only shippable, lift disproven | RES split clean |
| 7 | EPV damp NULL; mix Pareto-bound | filter trajectory hypothesis raised |
| R8 prep | R7 causal direction falsified | linear AEC fine on worst |
| R9 | STOP-1: state cannot sort worst from best | both populations dominated by RENDER_FALLBACK |
| **R10** | **STOP at P0: candidate is same R5/R6 axis** | **uniform tightening; over-compresses worst-DT_mv 3.6× more** |

## Recommendation (unchanged)

NN postfilter (separate plan) remains the only realistic ship route.
Across 10 rounds, every classical lever that takes `residual_echo_psd`
as input (or anything derived from it) lands on the same Pareto
curve. The remaining structural lever would be a **better
`residual_echo_psd` estimate itself** — i.e., shadow NLMS / kernel
changes, both excluded from R10's scope.
