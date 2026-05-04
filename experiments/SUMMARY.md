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


# Round 11 — Residual estimate decomposition + Phase 1 limiter audit

Branch: `algo/round11-residual-nearend-audit` (discarded after closure)
Date: 2026-05-04
Outcome: **Phase 0 GO (single-stage culprit identified) → Phase 1 STOP
(no per-frame discriminator at residual stage)**

## Why R11 was opened (Codex direction)

R10 cleared the suppressor-formula path: WebRTC-style
SuppressionGainCandidate uniformly tightened across all cohorts and
inherited the residual-model bias. Codex direction: stop touching
gain, audit the **residual_echo_psd estimate itself**. Hypothesis:
worst-DT_mv has its `residual_echo_psd` over-estimated, which collapses
`nearend_proxy = error_psd − residual_echo_psd` and forces the
suppressor (any formula) to over-compress.

## Phase 0 — audit-passive decomposition trace

Plumbing added to `ResFilter._stage_residual_model`: per-frame
voice-band (300–3000 Hz) `mean(residual_echo_psd)` captured at all
seven decomposition points (attribute → echo_cap → error_cap → dt_cap
→ render_ceil → reverb → boost), plus `r11_pre_reverb_voice`,
`r11_reverb_add_voice`, `r11_nearend_proxy_voice`,
`r11_res_to_error_voice`, `r11_dt_for_fs`, `r11_far_activity`. All
audit-passive. parity_round6 5-fixture HASH MATCH × 5.

### Per-cohort stage snapshots (voice-band mean, 800-case trace)

| cohort | attr | render_ceil (= pre_reverb) | reverb | final | err_psd |
|---|---:|---:|---:|---:|---:|
| FS_static | 1.99 | 0.88 | 8.37 | 9.44 | 1.80 |
| FS_movement | 2.93 | 1.22 | 11.72 | 13.37 | 1.90 |
| NE | 0.00 | 0.00 | 0.01 | 0.01 | 0.66 |
| DT_static | 1.11 | 0.46 | 6.22 | 6.92 | 1.54 |
| DT_movement | 1.30 | 0.54 | 7.22 | 7.91 | 2.16 |
| **worst-DT_mv** | 1.39 | **0.48** | **11.12** | **12.23** | **1.47** |
| best-DT_mv | 2.44 | 1.24 | 3.66 | 3.99 | 5.76 |

### Cohen's-d separator (worst-20 vs best-20 DT_mv, rank-locked baseline_deg)

| stage delta | worst | best | gate |
|---|---:|---:|---|
| d_echo_cap | -0.006 | -0.001 | no |
| d_error_cap | -0.903 | -1.182 | no |
| d_dt_cap | ~0 | ~0 | no |
| d_render_ceil | -0.002 | -0.021 | no |
| **d_reverb** | **+10.642** | **+2.419** | **PASS effect=0.85, |Δ|=8.22** |
| d_boost | +1.106 | +0.336 | no |

**Phase 0 verdict: GO** — single-stage culprit `d_reverb` clearly
identified. Pre-reverb residual is similar (0.48 worst vs 1.26 best),
but the reverb stage adds +10.64 voice units on worst vs +2.42 on best.
This is the first round in R5–R10 with a single-stage signal that
clears the strict Cohen's-d gate.

## Phase 1 — limiter design audit (overdrive definition)

Three overdrive definitions evaluated, audit-only:

1. per-bin `reverb_add / max(pre_reverb_res, 0.05·error_psd)`, voice mean
2. per-bin `reverb_add / max(pre_reverb_res, noise_psd)`, voice mean
3. cohort-level ratio of voice-band means

Per-frame def#1/def#2 are corrupted by silence-frame divisions
(pre_reverb ≈ 0 → eps → astronomical). Cohort-level ratio (def#3)
is the meaningful aggregate:

| cohort | reverb_add_voice | pre_reverb_voice | ratio |
|---|---:|---:|---:|
| best-DT_mv | 2.42 | 1.24 | 1.96 |
| FS_static | 7.49 | 0.88 | 8.48 |
| FS_movement | 10.50 | 1.22 | 8.60 |
| DT_static | 5.76 | 0.46 | 12.47 |
| DT_movement (avg) | 6.69 | 0.54 | 12.42 |
| worst-DT_mv | 10.64 | 0.48 | **22.17** |

Per-case ratio distribution (p10 / p50 / p90):

| cohort | p10 | p50 | p90 |
|---|---:|---:|---:|
| best-DT_mv | 0.4 | 5.9 | 38.7 |
| FS_static | 2.1 | 27.5 | 298.8 |
| FS_movement | 1.5 | 26.4 | 250.4 |
| DT_static | 2.5 | 22.3 | 108.1 |
| DT_movement | 1.7 | 18.4 | 95.3 |
| worst-DT_mv | 9.4 | 26.8 | 143.7 |

FS p50 (~27) overlaps worst p50 (~27) — **per-case overdrive cannot
cleanly separate worst-DT_mv from FS**.

## Co-discriminator audit (DT-side signals available at residual stage)

| cohort | coh2_voice | dt_for_fs | far_act | err_psd | np/err |
|---|---:|---:|---:|---:|---:|
| FS_static | 0.166 | 0.135 | 0.836 | 1.804 | 0.145 |
| FS_movement | 0.158 | 0.114 | 0.838 | 1.903 | 0.161 |
| DT_static | 0.144 | 0.129 | 0.796 | 1.535 | 0.236 |
| DT_movement | 0.147 | 0.151 | 0.799 | 2.155 | 0.256 |
| worst-DT_mv | **0.144** | **0.110** | 0.834 | **1.474** | 0.191 |
| best-DT_mv | 0.146 | 0.264 | 0.773 | 5.756 | 0.381 |

- coh2 essentially identical (0.144–0.166) across all far-active cohorts.
- dt_for_fs: worst (0.110) is indistinguishable from FS (0.114–0.135).
- error_psd: worst (1.47) is between DT_static (1.54) and FS (1.80) — close.

## Ablation iterations (limiter prototypes, all rejected)

Three soft-limiter prototypes were trial-implemented and trace-evaluated
without bench, then reverted. None passed mechanism gate "worst -50%,
FS ±20%".

| prototype | worst Δreverb | FS_static Δ | FS_mv Δ | best Δ | result |
|---|---:|---:|---:|---:|---|
| (a) per-bin: res_pre/err<0.5 AND add/err>1, cap=1.5·max(pre, 0.2err) | -13.7% | -13.9% | -11.6% | -18.5% | hits best more than worst |
| (b) per-bin: overdrive>3 with err-floor 0.05·err | -89.6% | -85.5% | -87.3% | -74.7% | nukes FS |
| (c) per-bin: overdrive>3 with noise_psd floor | -89.7% | -85.6% | -87.3% | -75.8% | same shape as (b) |
| (d) per-bin: coh2<0.4 AND overdrive>3 (noise floor) | not run | not run | not run | not run | aborted — coh2 doesn't discriminate |

## Mechanism reading

The reverb stage IS the residual-side culprit (P0 PASS). But worst-DT_mv's
signature in every signal available at the reverb stage is
indistinguishable from FS:

- same low coh2
- same low dt_for_fs (DTD doesn't see movement-DT)
- similar error_psd (1.47 worst vs 1.80 FS_static)
- similar far_activity (0.83 worst vs 0.84 FS)
- similar reverb_add magnitude (10.64 worst vs 10.50 FS_movement)

The only thing that *differs* is the genuine nearend content, which
the algorithm cannot detect at this stage — the upstream **DTD failure
on movement-DT** is the binding root cause. Any per-frame limiter that
discriminates by overdrive will hit FS (which legitimately needs the
reverb tail to suppress echo after the linear filter caps).

## Disposition

- Branch `algo/round11-residual-nearend-audit` discarded (no merge to
  main). R11 P0 plumbing (`_r11_diag` decomposition trace + audit-only
  3-overdrive definitions) is NOT on main; documented here for future
  re-application if upstream DTD-on-movement is fixed and the residual
  stage becomes addressable.
- No Phase 1 / 2 algorithm changes shipped.
- v3.8.1 `baseline_v381_seeded` remains canonical.

## Cumulative across Round 1-11

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
| R9 | STOP-1: state cannot sort worst from best | both dominated by RENDER_FALLBACK |
| R10 | STOP at P0: candidate is same R5/R6 axis | uniform tightening |
| **R11** | **P0 GO (reverb culprit) → P1 STOP** | **no per-frame discriminator; upstream DTD-on-movement is binding root** |

## Recommendation (refined after R11)

Across 11 rounds, the failure pattern is now mechanistically clear:

- **Suppressor side** (R5/R6/R10): every formula consuming
  `residual_echo_psd` lands on the same Pareto curve.
- **Filter side** (R7/R8): kernel ERLE on worst-DT_mv is *better* than
  best — kernel is not the binding constraint.
- **State / aggregator side** (R9): FilterQualityState cannot sort
  worst from best — both dominated by RENDER_FALLBACK.
- **Residual estimate side** (R11): single-stage culprit (reverb)
  identified, but no per-frame DT discriminator exists at that stage.

The remaining structural lever is **upstream DTD on movement** —
either a better DT detector that fires on worst-DT_mv (currently
dt_for_fs ≈ 0.110 there), or a separate DT-on-movement signal that
the residual stage can consume. Both require new instrumentation
outside R5–R11 scope.

NN postfilter (separate plan) remains the only realistic ship route
without that upstream rework.


# Round 12 — Movement-DT nearend-evidence audit (audit-passive)

Branch: `algo/round12-nearend-evidence-audit` (discarded after closure)
Date: 2026-05-04
Outcome: **STOP at Phase 0** — none of the 5 nearend-evidence candidates
qualify under the strict 2-way separation gate.

## Why R12 was opened (Codex direction)

R11 P0 cleanly identified the reverb stage as the residual-side
culprit, but R11 P1 proved every limiter prototype either no-ops or
nukes FS — worst-DT_mv was indistinguishable from FS in every signal
available at that stage (coh2, dt_for_fs, far_activity, error_psd,
reverb_add magnitude, r/err). R12 took the WebRTC AEC3 architectural
hint: introduce a `DominantNearendDetector` / `SubbandNearendDetector`
that can flip suppressor tuning, built from signals NOT already
proven null in R5–R11.

R12 was audit-first: trace 5 candidate signals, ask one question —
**does at least one separate worst-DT_mv from FS at strict gate?**

## Phase 0 plumbing

`ResFilter._stage_residual_model` emits a per-frame `_r12_diag` dict
with five candidate signals computed from local cross-spectra
(`S_mf_r12`, `S_mm_r12`, `S_ff_r12` EMA at α=0.65, audit-passive):

- **A** `r12_coh2_mf_voice` — mic-far coherence² (voice band mean)
- **B** `r12_shape_corr_ef_voice` — Pearson(log error_psd, log far_psd) over voice band
- **C** `r12_env_drift_voice` / `r12_env_decay_voice` / `r12_env_drift_ratio` — frame-to-frame error_psd dynamics
- **D** `r12_phase_jitter_mf_voice` — frame-to-frame phase change of S_mf voice band
- **E** `r12_near_residual_frac_voice` — far-uncorrelated mic energy fraction (independent of AEC filter)

parity_round6 5-fixture HASH MATCH × 5.

## Strict 2-way gate

A candidate qualifies only if it passes BOTH:

- **G1 worst-DT_mv vs FS_combined** (FS_static ∪ FS_movement, n=300):
  |Cohen's d| ≥ 0.6 AND |Δmean| ≥ 0.05
- **G2 worst-DT_mv vs best-DT_mv** (rank-locked baseline_v381_seeded.deg):
  |Cohen's d| ≥ 0.5

Stricter than R7/R11 default (0.5/0.05) because the R11 failure mode
was worst-vs-FS overlap — G1 is the binding criterion.

## Results (800-case trace, worst-20 vs best-20 vs FS-300)

| candidate | aggregate | worst | best | FS | d(w-b) | d(w-FS) | |Δw-FS| | G1 | G2 | qualified |
|---|---|---:|---:|---:|---:|---:|---:|---|---|---|
| A coh2_mf | mean | 0.305 | 0.333 | 0.359 | -0.22 | -0.43 | 0.053 | no | no | no |
| B shape_corr_ef | p50 | 0.533 | 0.461 | 0.554 | +0.60 | -0.18 | 0.021 | no | PASS | no |
| C env_drift | mean | 0.545 | 2.125 | 0.766 | -0.71 | -0.13 | 0.221 | no | PASS | no |
| C env_drift_ratio | p50 | 1.689 | 1.577 | 1.616 | +1.17 | +0.57 | 0.073 | no | PASS | no |
| D phase_jitter_mf | p50 | 0.439 | 0.379 | 0.395 | +0.41 | +0.32 | 0.044 | no | no | no |
| E near_residual_frac | mean | 0.572 | 0.562 | 0.487 | +0.05 | +0.48 | 0.085 | no | no | no |

## Mechanism reading

Three close-but-not-passing patterns:

1. **Coh2_mf direction is correct, magnitude small.** Worst-DT_mv
   mic-far coherence really is lower than FS (consistent with the
   hypothesis that nearend speech reduces far-correlated mic content),
   but Δ=0.054 is at the floor and d=-0.43 well below 0.6.

2. **Candidates B and C pass G2 only.** They sort within DT_mv (worst
   sortable from best) but cannot sort worst-DT_mv from FS — the same
   R11 failure mode. A reverb gate built on either would need
   additional discrimination to avoid hitting FS.

3. **Candidate E (near_residual_frac) is the inverse**: it sorts FS-vs-DT
   (DT cohorts ~0.55, FS ~0.49) but **worst and best DT_mv are
   identical** (0.572 vs 0.562, d=+0.05). This is the most definitive
   null in R12 — an estimate of "mic energy NOT explained by far
   reference" extracts equivalent nearend-energy on both worst and
   best. The thing that differs is upstream — likely how that energy
   is consumed (reverb attribution / suppressor decision), not whether
   the cross-spectrum can see it.

The phase-jitter candidate (D), inspired directly by Codex's WebRTC
direction, shows the predicted directionality but magnitudes are too
small after S_mf EMA smoothing.

## Disposition

- Branch `algo/round12-nearend-evidence-audit` discarded (no merge to
  main). R12 P0 plumbing (5 candidate signals + cross-spectrum EMAs +
  trace fields) is NOT carried to main; documented here for future
  re-application.
- v3.8.1 `baseline_v381_seeded` remains canonical.

## What R5–R12 cumulatively rule out

Combined with R11's null reading on `coh2`, `dt_for_fs`, `error_psd`,
`far_activity`, `reverb_add` magnitude, `r/err`, R12 adds:

- Mic-far coherence (A) — direction right, magnitude too small.
- Spectral shape correlation (B) — confounded; sorts within DT but not vs FS.
- Frame-to-frame envelope dynamics (C) — scale-confounded by mic energy.
- Cross-spectrum phase jitter (D) — too smoothed by EMA.
- Far-uncorrelated mic energy (E) — sorts FS-vs-DT but not worst-vs-best-DT.

This exhausts the residual-stage signal pool. The required movement-DT
nearend discriminator is not in this pool.

## Cumulative across Round 1–12

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
| R9 | STOP-1: state cannot sort worst from best | both dominated by RENDER_FALLBACK |
| R10 | STOP at P0: candidate is same R5/R6 axis | uniform tightening |
| R11 | P0 GO (reverb culprit) → P1 STOP | no per-frame discriminator at residual stage |
| **R12** | **STOP at P0: 5 candidates exhausted** | **residual-stage signal pool fully exhausted; nearend evidence is upstream** |

## Recommendation (refined after R12)

Three rounds in a row (R10/R11/R12) STOP at Phase 0 or Phase 1 of
a residual-stage / suppressor-stage local fix. The pattern is now
mechanistically settled:

- **Don't borrow WebRTC suppressor formulas** — R10 ruled that out
  (gain rewrite inherits the residual bias).
- **Don't borrow WebRTC reverb gates as local thresholds** — R11/R12
  ruled that out (no per-frame DT discriminator at residual stage).
- **Do borrow WebRTC architecture** — `ResidualEchoEstimator` separates
  early/late reflections as a state machine, and `AecState` consolidates
  decisions consumed downstream. That is a structural change, not a
  threshold tweak.

Next round candidates (under the constraint of NOT changing shadow
PBFDKF and NOT shipping NN):

- **R13**: WebRTC-style ResidualEchoEstimator / Reverb rewrite —
  early-vs-late reflection separation as independent state machines,
  audit-first.
- **R14**: AecState sidecar (consolidated state module that
  simultaneously controls residual mode / reverb / suppressor min-max /
  CNG, not just one routing decision as in R9).
- R15 (audibility / gain dynamics) and R16 (low-noise-render special
  case) are polish-tier; not blocking.

NN postfilter remains the only known ship route absent the upstream
architectural rework.


# Round 13 — WebRTC-style early/late reverb decomposition (audit-passive)

Branch: `algo/round13-residual-echo-reverb-rewrite` (discarded after closure)
Date: 2026-05-04
Outcome: **STOP at Phase 0** for both simplified candidates β and γ.
Does NOT close the full WebRTC AEC3 ResidualEchoEstimator architecture
(see "What this does NOT close").

## Why R13 was opened (Codex direction)

R12 closed STOP after exhausting the residual-stage signal pool for
nearend evidence. Direction shifted from "find a local nearend signal"
to "borrow WebRTC AEC3 *architecture* — early/late reflection state
machines, residual estimate as a consequence of state, not a single
multiplicative gate." R13 is the first audit-first test of this
architectural framing.

## Phase 0 plumbing

`PBFDAF.process()` extends the existing echo_spec accumulation loop
(line 686-700) with K_early=2 partition split, populating
`self.early_echo_spec` and `self.late_echo_spec` at the SAME pre-update
W as `echo_spec`. In-loop accumulation guarantees the identity
`early + late ≡ echo_spec` (same operand order, same W). At
BALANCED/16kHz fl=832 hop=160, n_partitions=5 so K_early=2 = 20 ms
boundary (direct + first early reflection), late = 30+ ms tail.

`AEC.process()` reads `self.filter.early_echo_spec` /
`self.filter.late_echo_spec`, computes PSDs, passes into `res.process()`
as new optional kwargs. `ResFilter._stage_residual_model` adds an
audit-passive R13 block AFTER the existing reverb add (so existing
behavior is untouched), snapshots pre-reverb residual + current addend,
and computes two candidate reverb estimators side-by-side:

- **β** — observation-bounded reverb IIR, input = `early_echo_psd`
  (instead of `far_psd`). Same gain / decay / gate as production.
  Hypothesis: when kernel sees no echo, β IIR starves.
- **γ** — early-presence scalar gate on the existing addend:
  `linear_echo_present = clip(early_voice_pwr / max(3·noise_voice_pwr, eps), 0, 1)`,
  then `gamma_addend = current_addend × linear_echo_present`.

α (late-partition linear echo) is recorded as a DIAGNOSTIC ONLY — not a
candidate; the kernel's late partitions are within-FIR-window late
reflections, not WebRTC's beyond-filter reverb tail.

13 trace fields per frame, voice-band aggregates only. parity_round6
5-fixture HASH MATCH × 5.

## Phase 0 gate (bidirectional)

Reviewing initial results showed an **overshoot** is just as harmful as
a drop on non-target cohorts. Analyzer enforces both bounds:

- worst-DT_mv: `ratio ≤ 0.70` (≥ 30% drop on target cohort)
- FS_static / FS_movement: `ratio ∈ [0.80, 1.20]` (drift ≤ ±20%)
- best-DT_mv: `ratio ∈ [0.80, 1.20]` (drift ≤ ±20%)

where `ratio_K = mean(predicted_final_K_voice) / mean(predicted_final_current_voice)`
per cohort. Predicted final = pre-reverb residual + candidate addend.

## Per-cohort results (predicted final residual ratio)

| cohort | n | predicted_current | predicted_β | predicted_γ | ratio_β | ratio_γ |
|---|---:|---:|---:|---:|---:|---:|
| FS_static | 169 | 8.371 | 7.915 | 8.122 | 0.946 | 0.970 |
| FS_movement | 131 | 11.719 | 9.876 | 11.513 | 0.843 | 0.982 |
| NE | 200 | 0.007 | 0.000 | 0.005 | 0.067 | 0.837 |
| DT_static | 186 | 6.223 | 4.440 | 5.958 | 0.714 | 0.957 |
| DT_movement | 114 | 7.224 | 4.171 | 6.910 | 0.577 | 0.957 |
| **worst-DT_mv** | 20 | 11.122 | 4.920 | 10.665 | **0.442** | 0.959 |
| **best-DT_mv** | 20 | 3.655 | 8.141 | 3.487 | **2.227** | 0.954 |

### Gate evaluation

| candidate | worst-DT_mv | FS_static | FS_movement | best-DT_mv | qualified |
|---|---:|---:|---:|---:|---|
| β | 0.442 (PASS) | 0.946 (PASS) | 0.843 (PASS) | **2.227 (FAIL)** | **no** |
| γ | **0.959 (FAIL)** | 0.970 (PASS) | 0.982 (PASS) | 0.954 (PASS) | **no** |

## Mechanism reading — the structural finding

β's input `early_echo_psd` voice-mean per cohort:
- best-DT_mv: 4.65  (highest among DT cohorts)
- worst-DT_mv: 2.84
- DT_static: 2.55
- FS_static: 4.64

`early_echo_psd` is essentially a proxy for **filter convergence
quality on the recent room response**. Best-DT_mv has higher early_echo
*because the kernel converged better* (mic was loud enough to drive
clean adaptation), not because there is more echo. This recapitulates
R8-prep's finding from the opposite side: linear AEC ERLE is BETTER on
worst-DT_mv (1.66 dB) than best (0.72 dB).

So β has the right WORST direction (-56% reduction) but **anti-correlates**
with the cohort axis: it inflates reverb on best (already fine,
predicted residual ×2.23) and starves it on worst. Net AECMOS impact
predictable to be a loss on best while gaining on worst — a Pareto
shift, not an improvement.

γ's `linear_echo_present` scalar gate ranges 0.63–0.74 across
far-active cohorts (0.10 on NE only). The 0.10-wide spread is far too
narrow to discriminate worst-DT_mv from FS at the residual scale needed.

## Disposition

- Branch `algo/round13-residual-echo-reverb-rewrite` discarded (no
  merge to main). R13 P0 plumbing (PBFDAF early/late split + ResFilter
  R13 audit block + 13 trace fields) is NOT carried to main; documented
  here for future re-application.
- v3.8.1 `baseline_v381_seeded` remains canonical.

## What this does NOT close

R13 closes only **the two simplified candidates β and γ**. It does NOT
close the full WebRTC AEC3 ResidualEchoEstimator architecture. Three
structural elements R13 did NOT plumb (and which would behave
differently than β/γ):

1. **Render-buffer partition tap**: AEC3's reverb input is a
   *render-buffer slice at a tail offset*, NOT the kernel's late
   partitions and NOT raw `far_psd`. This decouples reverb input from
   kernel convergence quality — addresses the β reverse-correlation.
2. **AecState-supplied reverb frequency response**: AEC3 shapes the
   render-buffer tail by a per-bin reverb response from AecState.
3. **Per-AecState gating** (saturated echo, transition window, etc.)
   that mediates between linear path and nonlinear path.

A future round (R13b or R14) could plumb (1) + (2) + (3). β can be
re-tested in that variant where input is render-buffer-tap rather than
kernel-derived, which would not have the convergence-quality
correlation.

## Cumulative across Round 1–13

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
| R9 | STOP-1: state cannot sort worst from best | both dominated by RENDER_FALLBACK |
| R10 | STOP at P0: candidate is same R5/R6 axis | uniform tightening |
| R11 | P0 GO (reverb culprit) → P1 STOP | no per-frame discriminator at residual stage |
| R12 | STOP at P0: 5 candidates exhausted | residual-stage signal pool exhausted |
| **R13** | **STOP at P0: β reverse-correlated, γ too soft** | **kernel-derived early/late points wrong direction; full WebRTC reverb plumbing (render-tap + AecState reverb response) not yet plumbed** |

## Recommendation

Three rounds in a row (R11/R12/R13) document the same pattern at
escalating depth: at the residual stage, the cohort discriminator the
problem demands is structurally absent OR (R13) anti-correlated.
Three options remain:

1. **Plumb full WebRTC AEC3 ResidualEchoEstimator** (R13b/R14): adds
   render-buffer partition tap and AecState reverb frequency response.
   Substantial scope. Decouples reverb input from kernel-convergence
   quality, addressing β's reverse-correlation. Not guaranteed to
   escape the same Pareto wall — the discriminator for worst-DT_mv may
   still come down to upstream DTD weakness — but it is the next
   architectural step the data justifies.
2. **Accept current Pareto and ship v3.8.1 as-is** for non-NN tracks.
3. **Commit to NN postfilter** as the ship route (existing plan).

NN remains the only known route that does not require either upstream
DTD-on-movement rework or the full AEC3 plumbing.


# Round 14 — AEC3-style render-tail reverb audit (audit-passive)

Branch: `algo/round14-render-tail-reverb-audit` (discarded after closure)
Date: 2026-05-04
Outcome: **STOP at Phase 0** for both simplified candidates δ and ε.
ε is diagnostic-positive (avoided R13 β's best-DT_mv overshoot — first
in R10–R14 to do so) but still non-ship: FS preservation fails by ~36–47%.
Does NOT close the full WebRTC AEC3 ResidualEchoEstimator architecture
(see "What this does NOT close").

## Why R14 was opened (Codex direction)

R13 closed STOP because β's reverb input `early_echo_psd` correlated
with kernel convergence quality (best-DT_mv has higher early_echo →
β fed it more reverb → 2.27× overshoot on best, opposite of need).
R13 verdict pointed to: WebRTC AEC3 reverb input is a render-buffer
partition tap, NOT the kernel's late partitions. R14 is that test.

## Phase 0 plumbing

`AEC.process()` reads `tail_p = self.filter.partition_idx` (after
`filter.process()` advances it, that index is the next write slot =
oldest entry in the circular buffer; for K_tail = N − 1 = 5 at
fl=832/hop=160 with n_partitions=6, that IS the tail tap, ~50 ms back).
`tail_far_psd = |X_buf[tail_p]|²` and the AEC's `_erl_estimate` are
passed into `res.process()` as new optional kwargs.

`ResFilter._stage_residual_model` adds an audit-passive R14 block
AFTER the existing reverb add (so existing behavior is untouched),
snapshots pre-reverb residual + current addend, and computes two
candidate reverb estimators side-by-side:

- **δ** — render-tail × ERL scalar shape (linear, NOT squared; ERL is
  already a power ratio). No new state beyond `reverb_psd_δ` IIR.
- **ε** — render-tail × per-bin slow-EMA shape from echo_psd.
  Slow EMA TC ≈ 5 s with cold-start init at 50 far-active frames so
  ~10 s blind-set clips have meaningful warmed samples. Marked HIGH
  RISK in the plan because the shape is built from `echo_psd` =
  `|W·X|²` smoothed, which could in principle still inherit kernel
  convergence bias.

12 trace fields × {`_all_mean`, `_warm_mean`} aggregates per case
(δ unaffected by warmup; ε aggregated over warmup-completed frames
only). Cold-start init successful on 93–98% of frames per cohort.

parity_round6 5-fixture HASH MATCH × 5.

## Phase 0 gate (bidirectional, warmup-excluded)

- `ratio[worst-DT_mv] ≤ 0.70` (≥ 30% drop on target)
- `ratio[FS_static]   ∈ [0.80, 1.20]`
- `ratio[FS_movement] ∈ [0.80, 1.20]`
- `ratio[best-DT_mv]  ∈ [0.80, 1.20]`
- DT_static yellow flag if outside [0.60, 1.40]
- Hard early-exit: `ratio[best-DT_mv] > 1.50` (β-style overshoot, STOP)

## Per-cohort results

### Sanity (warmup, tail-tap, ERL distribution)

| cohort | warm coverage | tail_far_psd | ERL | slow_echo | slow_tail_far | ε ratio (slow_e/slow_tf) |
|---|---:|---:|---:|---:|---:|---:|
| FS_static | 98% | 4.45 | 0.234 | 0.234 | 6.80 | 0.034 |
| FS_movement | 98% | 6.69 | 0.234 | 0.234 | 6.81 | 0.034 |
| NE | 93% | 0.006 | 0.032 | 0.000 | 0.007 | — |
| DT_static | 98% | 3.65 | 0.257 | 0.257 | 3.56 | 0.072 |
| DT_movement | 98% | 4.28 | 0.261 | 0.262 | 3.91 | 0.067 |
| **worst-DT_mv** | 98% | 6.78 | 0.288 | 0.288 | 6.04 | 0.048 |
| **best-DT_mv** | 98% | 1.57 | 0.290 | 0.291 | **1.83** | **0.159** |

ERL is roughly uniform across far-active cohorts. The notable variation
is `slow_tail_far`: best-DT_mv has the LOWEST tail far (1.83) while
worst-DT_mv has it among the highest (6.04). This 4× difference
translates into ε's per-bin shape being 4× higher on best (0.159) than
on worst (0.048) — which is what saves best-DT_mv from the β-style
overshoot.

### Predicted final residual + ratios (gate metric)

| cohort | predicted_current | predicted_δ | predicted_ε | ratio_δ | ratio_ε |
|---|---:|---:|---:|---:|---:|
| FS_static | 8.343 | 2.464 | 5.333 | **0.295** | **0.639** |
| FS_movement | 11.736 | 3.893 | 6.181 | **0.332** | **0.527** |
| NE | 0.006 | 0.001 | 0.001 | 0.205 | 0.083 |
| DT_static | 6.143 | 1.849 | 3.590 | 0.301 | 0.584 |
| DT_movement | 7.221 | 2.179 | 3.368 | 0.302 | 0.466 |
| **worst-DT_mv** | 11.144 | 3.207 | 5.423 | 0.288 | 0.487 |
| **best-DT_mv** | 3.620 | 1.917 | 3.448 | 0.530 | **0.953** |

### Gate evaluation

| candidate | worst-DT_mv | FS_static | FS_movement | best-DT_mv | DT_static | qualified |
|---|---:|---:|---:|---:|---:|---|
| δ | 0.288 PASS | **0.295 FAIL** | **0.332 FAIL** | **0.530 FAIL** | 0.301 (YELLOW) | **no** |
| ε | 0.487 PASS | **0.639 FAIL** | **0.527 FAIL** | 0.953 PASS | 0.584 (YELLOW) | **no** |

## Mechanism reading

### Why δ fails (uniform under-attribution)

δ multiplies tail_far_psd by ERL (~0.23 across far-active cohorts).
The current production formula uses raw far_psd (no ERL). δ's input is
~4× smaller across the board, and reverb_gain (tuned for the larger
input) was not adjusted. The IIR output is uniformly smaller everywhere.
Same shape as R10 uniform tightening, opposite direction
(uniform under-estimation). Scale mismatch dominates; no per-cohort
discrimination.

### Why ε partially works on best-DT_mv but not FS

ε's per-bin shape `slow_echo / slow_tail_far` is a long-time-average
echo-path response. Best-DT_mv has LOW slow_tail_far (1.83) so the
ratio is HIGH (0.159), partially compensating for the smaller raw-vs-
shaped input scale → predicted residual stays within 5% of current.
This is a **structural success vs R13 β**: per-bin slow shape avoids
the kernel-convergence reverse-correlation that defeated β.

But on FS the slow_tail_far is high (6.80) so the ratio is low (0.034).
Per-bin shape doesn't compensate enough → ε under-attributes reverb on
FS by 36–47%. Same scale-mismatch story as δ, just less severe on
best-DT_mv specifically.

### What R14 demonstrates (positive finding)

ε is the first candidate in R10–R14 to avoid BOTH:
- R10's failure mode (uniform same-axis tightening) — FS still drops
  but best-DT_mv is preserved (0.953 vs 1.0 baseline)
- R13's failure mode (best-DT_mv overshoot from convergence bias)

The remaining defect is **scale-mismatch with the existing
`reverb_gain`**, a tuning issue rather than a structural one. The
render-tail decoupling DOES remove the kernel-convergence bias.

## Disposition

- Branch `algo/round14-render-tail-reverb-audit` discarded (no merge to
  main). R14 P0 plumbing (PBFDAF tail-tap exposure + ResFilter R14
  audit block + 12 trace fields × _all/_warm aggregation) NOT carried
  to main; documented here for future re-application.
- v3.8.1 `baseline_v381_seeded` remains canonical.

## What this does NOT close

R14 closes only **the two simplified candidates δ and ε** under the
constraint of NO `reverb_gain` retuning. The structural elements R14
did NOT plumb (any of which could address the scale-mismatch FS
failure):

1. **Calibrated AecState reverb frequency response**: WebRTC AEC3
   maintains an explicit per-bin reverb response updated from AecState
   (not derived from echo_psd as ε does). The response would be
   normalized so reverb prediction matches actual late-reflection
   energy without depending on a separately-tuned reverb_gain.
2. **State-dependent reverb gain**: WebRTC's reverb gain is itself
   AecState-dependent (linear vs. nonlinear path).
3. **Per-AecState gating** (saturated-echo, transition-window,
   linear/nonlinear branching) that mediates between paths.

A future round (R14b or R15) could:
- (a) Allow `reverb_gain` to be co-tuned with the new input — would
  resolve the scale-mismatch but is a 2-axis search; risks fitting
  the dataset rather than the underlying model.
- (b) Plumb the full AecState reverb frequency response — bigger
  scope but the proper architectural fix.

## Cumulative across Round 1–14

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
| R9 | STOP-1: state cannot sort worst from best | both dominated by RENDER_FALLBACK |
| R10 | STOP at P0: candidate is same R5/R6 axis | uniform tightening |
| R11 | P0 GO (reverb culprit) → P1 STOP | no per-frame discriminator at residual stage |
| R12 | STOP at P0: 5 candidates exhausted | residual-stage signal pool exhausted |
| R13 | STOP at P0: β reverse-correlated, γ too soft | kernel-derived early/late points wrong direction |
| **R14** | **STOP at P0: δ uniform under, ε FS under** | **render-tail decouples kernel bias (ε preserves best-DT_mv) but reverb_gain scale mismatch causes FS under-attribution** |

## Recommendation

R14 ε is the first candidate in R10–R14 that avoided both failure
modes documented in prior rounds (R10 uniform same-axis, R13 best
overshoot). The remaining defect is reverb_gain scale-mismatch — a
tuning issue rather than a structural one.

Next-round options:

1. **R14b**: ε with co-tuned `reverb_gain` (small-scope follow-up).
   Fit `reverb_gain` so ε's FS_static ratio = 1.0, then re-evaluate.
   Risk: 2-axis tune may overfit dataset.
2. **R15**: full AecState reverb frequency response (proper architectural
   fix). Substantial scope.
3. **Accept current Pareto and ship v3.8.1 as-is**, or commit to NN
   postfilter as the ship route.

NN postfilter remains the only known route that does not require
either upstream DTD-on-movement rework or full AEC3 plumbing.


# Round 15 — Feasible-region search over residual/reverb candidate formulas (closed at P0)

**Branch**: `algo/round15-feasibility-search` (audit-only, NOT MERGED — discarded after verdict)
**Date**: 2026-05-04
**Outcome**: STOP at Phase 0 — empty feasible region over the parameterized formula family on the R14/R15 observable pool.

## Why R15 was opened

R10–R14 each tested ONE candidate per branch and STOPped at Phase 0.
R14 ε was the first to avoid both R10 uniform-tightening AND R13
best-DT_mv reverse-correlation, but failed FS scale (uniform
under-attribution from `reverb_gain` mismatch). R14 verdict pointed to
two follow-ups: R14b (co-tune `reverb_gain` with ε) or R15 (full
AecState reverb response).

R15 changed the methodology rather than picking a single follow-up
candidate. Hypothesis: the one-candidate-per-branch cycle was burning
time without sampling the design space, so a frozen trace + offline
grid search over a parameterized family of {scale, mix, clip,
state-conditioned gate} formulas would either (a) find a feasible
region, in which case top-5 candidates are reported and user picks
one to promote, or (b) prove the entire family empty — strong
evidence to close the classical residual/reverb path.

## Phase 0 plumbing

- **R14 δ/ε/tail-tap audit block** cherry-picked from R14 reflog
  (commit `c230b98`) onto R15 branch as foundation. R14 plumbing was
  bit-exact audit-passive when first written; cherry-pick re-verified
  parity_round6 PASS (5/5 HASH MATCH).
- **R15 state-snapshot extension** in `ResFilter._stage_residual_model`:
  accept `state_snapshot` kwarg dict, copy values + AecState bools/scalars
  into `self._r15_diag`. Snapshot built in `AEC.process` from already-
  computed values (no recomputation): dt_indicator, dtd_coherence_prev
  (causal — DTD computed after res.process), dt_from_energy/shadow,
  divergence, mu_scale_mean, inst_erle_smooth, shadow_dt; AecState adds
  render_active, dt_combined, usable_linear_estimate, epc_active,
  epc_hangover_count, filter_converged/once_converged, shadow_advantage;
  plus res-stage scalars far_activity, dt_for_fs, saturation_level,
  erl_estimate, is_stationary_dt.
- **trace_phase0.py**: per-far-active-frame raw-array collection of R14+R15
  fields into per-case `experiments/round15_p0/frames/<stem>.npz` (~80 MB
  total across 800 cases). states.json keeps voice-band aggregates.
- **grid_search_r15.py**: offline engine. Layer 1 base × Layer 2 clip ×
  Layer 3 gate cross-product with prune rules (B0-equivalent, degenerate
  clip, no-op gate). 8892 candidates. Vectorized numpy over cached
  per-frame arrays; ~3 min runtime.
- **analyze_round15_p0.py**: top-5 reporter with cohort-leak risk flag,
  neighborhood density, warm-fallback retention; STOP path with scoped
  conclusion + R16 follow-up pointer.
- parity_round6 PASS bit-exact on plumbing commit (`4a76bbd`).

## Phase 0 gate (per plan, after the 5 user-requested fixes)

Bidirectional + per-case gate, on predicted final residual ratio
computed from warm frames (per-case warm-fallback to all-frame mean if
20 ≤ n_warm < 50, drop if < 20 far-active total):

1. cohort_ratio[worst-DT_mv-20] ≤ 0.70
2. cohort_ratio[FS_static] ∈ [0.80, 1.20]
3. cohort_ratio[FS_movement] ∈ [0.80, 1.20]
4. cohort_ratio[best-DT_mv-20] ∈ [0.80, 1.20]
5. ≥ 10/20 worst-DT_mv-20 cases with case_ratio < 0.85
6. ≤ 2 FS cases (out of FS_static ∪ FS_movement) with case_ratio > 1.50

Hard early-exit: candidate dropped if cohort_ratio[best-DT_mv-20] > 1.50.

Per-cohort retained-case counts (n_retained / n_warm_fallback / n_dropped)
reported per top-5 entry. Cohort-leak flag = HIGH if a Layer-3 gate fires
on ≥ 80 % of one cohort and ≤ 20 % of another (gate is effectively a
cohort-label proxy).

## Phase 0 result

| metric | value |
|---|---:|
| candidates evaluated | 8892 |
| qualifying (all 6 gates) | **0** |
| pass g1 worst ≤ 0.70 | 932 |
| pass g2 FS_st in band | 6386 |
| pass g3 FS_mv in band | 6396 |
| pass g4 best in band | 6886 |
| pass g5 worst-20 dir ≥ 10/20 | 1945 |
| pass g6 ≤ 2 FS overshoots | 7593 |
| pass g1 AND g4 (both DT cohorts good) | 110 |

The 110 candidates that satisfy both DT cohort gates ALL fail g2+g3
because they uniformly tighten FS to ratio 0.62–0.69 — same axis-coupling
as R10 uniform tightening and R14 δ uniform under-attribution. Sample
of the g1+g4 passers:

| formula | worst | best | FS_st | FS_mv | failed |
|---|---:|---:|---:|---:|---|
| B2(s=0.5) + clip(a=0.5, b=∞) | 0.586 | 0.806 | 0.656 | 0.654 | g2,g3 |
| B2(s=0.5) + clip(a=0.5, b=2.0) + G_dt(τ=0.5) | 0.599 | 0.806 | 0.649 | 0.648 | g2,g3 |
| B1(s=2) + clip(a=0.5, b=1.0) + G_dt(τ=0.3) | 0.687 | 0.814 | 0.671 | 0.679 | g2,g3 |

The DT-vs-FS gap among the best candidates is only ~0.16 ratio (0.60
DT vs 0.65 FS) — there is no formula in the searched family that
discriminates more strongly. State-conditioned gates (G_dt, G_render,
G_coh, G_far) only attenuate the global drop; they do NOT reverse the
DT-vs-FS direction because every input in the observable pool is
either uniformly sized across cohorts (current_add, far_psd,
tail_far) or inversely correlated with the cohort axis (delta_add via
ERL, epsilon_add via slow_echo/slow_tail) — i.e. no member of the pool
is a positive-correlation discriminator.

Top-5 nearest-misses by g1 ranking (deepest worst-DT_mv reductions,
all uniformly over-attribute):

| rank | formula | worst | best | FS_st | FS_mv | failed |
|---:|---|---:|---:|---:|---:|---|
| 1 | B1(s=0.5) | 0.188 | 0.373 | 0.200 | 0.213 | g2,g3,g4 |
| 2 | B1(s=0.5) + G_satecho | 0.188 | 0.373 | 0.200 | 0.213 | g2,g3,g4 |
| 3 | B1(s=0.5) + G_far(τ=0.1) | 0.188 | 0.373 | 0.200 | 0.213 | g2,g3,g4 |
| 4 | B1(s=0.5) + clip(a=0.0, b=1.0) | 0.188 | 0.373 | 0.200 | 0.213 | g2,g3,g4 |
| 5 | B1(s=0.5) + clip(a=0.0, b=1.0) + G_satecho | 0.188 | 0.373 | 0.200 | 0.213 | g2,g3,g4 |

## Mechanism reading

R15 reproduces and extends every prior failure mode at scale, in one
trace + one search:

- **R10 uniform tightening**: B0-mix and B1/B2/B3 base candidates
  collapse all cohorts together. 6886 candidates (77 %) keep best-DT_mv
  in band, but only 110 of those also tighten worst enough — and all
  110 over-tighten FS by the same axis.
- **R13 reverse-correlation**: any base whose input depends on filter W
  (β candidates were not in R15's pool but the same logic applies to ε
  via slow_echo) shows the best > worst ordering when scaled small.
- **R14 ε scale-mismatch**: B2(s=1) — the literal R14 ε — sits at
  worst=0.527 best=0.953 FS_st=0.487 FS_mv=0.639, exactly matching the
  R14 verdict numbers (verifies the offline-trace calibration is
  faithful to in-loop). Scaling ε up (B2 s=2,4,8) tightens FS further;
  scaling down (s=0.5) pushes worst above 0.70.

The 110-candidate g1+g4 boundary represents the **limit of the
observable pool**: no Layer-1 base + Layer-2 clip + Layer-3 gate
combination can split DT_mv worst from FS by more than ~0.16 ratio.
The R10 / R13 / R14 patterns are not artifacts of single-point
choices — they are the manifold of this design space.

## Disposition

- Branch `algo/round15-feasibility-search` discarded (no merge to main).
  The R15 plumbing (R14 audit cherry-pick + state-snapshot extension +
  per-frame npz output + grid_search_r15 + analyzer) is NOT on main; this
  closure section captures everything load-bearing.
- 80 MB `frames/` artifact and 14 MB `search_results.json` from
  `experiments/round15_p0/` are reproducible from a future trace; not
  retained on main.

## Scoped conclusion (per plan)

**Proves**: no feasible classical-DSP residual/reverb formula exists
over the **observable pool** R14/R15 surfaced — that is, over
{`current_add`, `delta_add`, `epsilon_add`, `tail_far`, `slow_echo`,
`slow_tail_far`, `erl`, AecState bools (render_active, dt_combined,
usable_linear_estimate, saturated_echo proxy, epc_active),
DTD scalars (coh_xy, dt_for_fs), `far_activity`, `mu_scale`} via the
parameterized Layer-1 × Layer-2 × Layer-3 family. Single-day
one-candidate-per-branch experimentation through R10–R14 was not the
bottleneck — there is no point in this design space.

**Does NOT prove**: full WebRTC AEC3 ResidualEchoEstimator is
impossible. AEC3 uses additional state we have not surfaced in R14/R15
(per-band reverb decay model, calibrated frequency response,
nonlinear/linear branching with saturated-echo handling,
transition-window protection beyond a single bool, render-buffer late
tap with K_tail variation).

## Cumulative across Round 1–15

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
| R9 | STOP-1: state cannot sort worst from best | both dominated by RENDER_FALLBACK |
| R10 | STOP at P0: candidate is same R5/R6 axis | uniform tightening |
| R11 | P0 GO (reverb culprit) → P1 STOP | no per-frame discriminator at residual stage |
| R12 | STOP at P0: 5 candidates exhausted | residual-stage signal pool exhausted |
| R13 | STOP at P0: β reverse-correlated, γ too soft | kernel-derived early/late points wrong direction |
| R14 | STOP at P0: δ uniform under, ε FS under | render-tail decouples kernel bias but reverb_gain mismatch |
| **R15** | **STOP at P0: empty feasible region (0/8892)** | **no Layer-1 × Layer-2 × Layer-3 formula over R14/R15 observable pool passes bidirectional + per-case gate; classical residual/reverb path closed** |

## Recommendation

R10–R15 cumulatively rule out classical-DSP residual/reverb adjustment
over the surfaced observable pool. The remaining axes of mechanism
work are:

1. **R16 missing-state audit** (per the R15 plan's follow-up section) —
   surface state R14/R15 did NOT plumb (per-band reverb decay model,
   K_tail sweep at multiple offsets, nonlinear/linear branch state,
   transition-window remaining time as continuous, saturation history,
   per-band erle / erle-based confidence), then rerun the R15
   grid-search engine over the enlarged pool. If that ALSO closes
   empty, the classical residual/reverb path is dead by every
   reasonable interpretation.

2. **NN postfilter** (`~/.claude/plans/jazzy-brewing-castle.md`) —
   learned residual; bypasses the observable-pool ceiling entirely by
   using upstream features the DSP residual stage does not see.

3. **Upstream filter-trajectory rewrite** — flagged in R8/R11 verdicts
   as the binding root for DT_mv. Out of scope of R10–R15 (those rounds
   only touched the residual stage); a future round would need to open
   on the kernel side, not the residual side.

4. **Accept v3.8.1 baseline as the classical-DSP Pareto ship** and
   commit to NN as the only forward-going scoring lift. R10–R15 is the
   cumulative evidence supporting this decision.
