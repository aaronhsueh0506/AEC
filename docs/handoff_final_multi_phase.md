# AEC Improvement Multi-Phase Execution — Final Handoff

Date: 2026-04-22。Session: autonomous multi-phase plan (α, β, γ, δ, ε)。
Main tip: `1372198` (Phase 2 Stage 1 merged + earlier commits)。

## §1 Executive summary

**2-layer defensive stack merged**, 2 architectural experiments preserved as unmerged reference branches。

| Phase | Subject | Result | Status |
|---|---|---|---|
| α | Phase 2 Stage 2 threshold tune (0.65→0.55) | **FAIL/MARGINAL** | Branch preserved, not merged |
| β | Gap #5 FilterMisadjustmentEstimator | **FAIL** | Branch preserved, not merged |
| γ | Saturation (B-7/B-9 followups) | SKIPPED | No forward spec; already-merged code |
| δ | DTD/RES (B-6/B-10 followups) | SKIPPED | No forward spec; demoted in bisect plan |
| ε | Final handoff + cleanup | DONE | This doc |

**Production behavior** (all flags default OFF): bit-exact with pre-session main (Phase 2 Stage 1 merged + B-16 flag-gated)。No defaults changed。

## §2 Phase α — Phase 2 Stage 2 Priority 1 (threshold 0.55)

**Status**: FAIL / MARGINAL — branch `experiment/phase2-stage2-threshold-055@155f588` preserved。

| Stage | Verdict |
|---|---|
| 2A spec | `30a4728` |
| 2B impl | `ce4a09a` (15-line diff, 3 invariants PASS) |
| 2C smoke (9 cases) | PASS (PZ7V full-file retained −7.62 dB vs baseline) |
| 2D 800-case | **FAIL**: ΔFS_echo **−0.035** vs Phase 2 baseline (−0.030), slightly **worse** |

**Diagnostic** ([phase2_stage_2_prelim_diagnostic.md](phase2_stage_2_prelim_diagnostic.md)): 3 big-loss cases have 3 distinct mechanisms。Threshold tune only addresses 1/3 (LeV1uF over-aggressive +0.43 dB)。KSN5Jr shadow_adv spike 7.2 unreachable by any threshold。IUp2c0 missed-copy mechanism not fixable by threshold。

**Flag**: `AEC_EXP_SHADOW_COPY_THR` env variable retained on branch — useful mechanism for future tuning iteration。Zero cost to preserve。

## §3 Phase β — Gap #5 FilterMisadjustmentEstimator

**Status**: FAIL — branch `experiment/gap5-misadjustment@ab0eea2` preserved。

### §3.1 β0 probe evidence (preserved, still valid)

PZ7V t=9.95s onset window with B-16 + Phase 2 enabled:

| Metric | Pre-onset | Onset peak |
|---|---:|---:|
| `filter.W` L2 | 7.48 | 7.29 (flat) |
| `filter.echo_spec` L2 | 2.5 | 11.4 (4.6×) |
| err_pwr | 2e−7 | 1.5e−1 (750 000×) |
| e²/y² | −38 dB | **+2.56 dB** |

**Signature**: W magnitude does not track echo path gain change — textbook filter-misadjustment。The problem is **real**;implementation trigger was insufficient。

### §3.2 β4 verdict

| combo | FS_echo | ΔFS vs (1,1) |
|---|---:|---:|
| (1,1) P2 stacked | 3.447 | — |
| gap5 (1,1,1) | **3.397** | **−0.050** |

FAIL per spec §6.3 (aggregate regress > 0.020) and revised rubric (> 0.005)。Static subset −0.058, movement −0.039。

### §3.3 Insight for future revisit

1. Replace `err/mic` trigger with Phase 2's `err_refined / err_coarse` ratio (DT-invariant, less noisy — matches AEC3 mainline design)
2. Explicit `dt_from_shadow < 0.3` gate
3. Less aggressive scale factors (1.2× vs 1.4× with shorter overhang)
4. Upper-bound: don't trigger when output leak is already low

## §4 Phase γ — Saturation (skipped)

No forward spec exists。B-7 (`_FIX_B7` PBFDKF P+W freeze) and B-9 (shadow saturation gate) are **already merged** as Phase 1-era work。[bisect_analysis_and_plan.md](bisect_analysis_and_plan.md) noted: "top-20 零 saturation 案例; B-7/B-9 bisect 顯示影響 < 0.012" — demoted in priority。

Per autonomous plan rule (skip when no forward spec), **skipped**。

## §5 Phase δ — DTD/RES (skipped)

No forward spec exists。B-6 (DTD/RES feedback) already landed as revert, not a forward fix。[bisect_analysis_and_plan.md](bisect_analysis_and_plan.md) noted: "top-20 零 DT 案例, 組 4 降優先"。

Conditional on Phase γ merging (which was skipped) → **skipped**。

## §6 Flag inventory (main HEAD `1372198`)

| Flag | Default | Status | Reference |
|---|---|---|---|
| `AEC_FIX_B16` | `0` | merged EXPERIMENTAL (Phase 1 NEUTRAL, PZ7V wav-level −6.59 dB onset) | [spec_b16_raw_dt_jump_veto.md](spec_b16_raw_dt_jump_veto.md), [b16_stage_1d_results.md](b16_stage_1d_results.md) |
| `AEC_EXP_SHADOW_NLMS` | `0` | merged EXPERIMENTAL (Phase 2 MARGINAL, PZ7V full-file −9.16 dB stacked) | [spec_phase2_shadow_as_nlms.md](spec_phase2_shadow_as_nlms.md) v4, [phase2_stage_1d_results.md](phase2_stage_1d_results.md) |
| `AEC_EXP_SHADOW_COPY_THR` | unset | **branch-only** (experiment/phase2-stage2-threshold-055) | [phase2_stage_2_copy_threshold_tune.md](phase2_stage_2_copy_threshold_tune.md), [phase2_stage_2_results.md](phase2_stage_2_results.md) |
| `AEC_EXP_MISADJUST` | `0` | **branch-only** (experiment/gap5-misadjustment) | [spec_gap5_filter_misadjustment.md](spec_gap5_filter_misadjustment.md), [phase_beta_gap5_results.md](phase_beta_gap5_results.md) |

**Recommended usage**:
- Production: all flags default OFF (bit-exact stable baseline)
- Research / PZ7V-class investigation: `AEC_EXP_SHADOW_NLMS=1` (Phase 2 opt-in, EXPERIMENTAL)
- Full opt-in stack: `AEC_FIX_B16=1 AEC_EXP_SHADOW_NLMS=1` (2-layer defensive, best PZ7V full-file −9.16 dB but AECMOS MARGINAL −0.028)

## §7 Branch topology

```
main@1372198  [production, all merged flags default OFF]
├── experiment/phase2-stage2-threshold-055@155f588  [α MARGINAL/FAIL, threshold tune]
├── experiment/gap5-misadjustment@ab0eea2            [β FAIL, misadjustment estimator]
├── feature/phase2-shadow-as-nlms@4052528            [Phase 2 source branch, already merged]
├── feature/b16-raw-dt-jump-veto                     [Phase 1 source branch, already merged]
└── probe/phase2-d1-shadow-nlms@3682f98              [Stage 0b historical]
```

All experimental branches retained as reference。No branches deleted。

## §8 Baseline measurements (at session start / end)

| Metric (all flags OFF) | Value |
|---|---:|
| FS_echo (800-case AECMOS) | 3.475 |
| DT_echo | 4.034 |
| DT_deg | 2.448 |
| NE_deg | 4.005 |
| PZ7V full-file leak | −17.07 dB |
| PZ7V onset leak | −13.81 dB |
| Cat-A 13 mean full-file leak | −24.78 dB |

End-of-session main is **bit-exact** with start-of-session main (no merges during this session)。

**Best opt-in combo** (B-16 + Phase 2 stacked, `AEC_FIX_B16=1 AEC_EXP_SHADOW_NLMS=1`):

| Metric | Value | Δ vs (0,0) |
|---|---:|---:|
| FS_echo | 3.447 | −0.028 |
| DT_echo | 4.037 | +0.003 |
| PZ7V full | −26.22 dB | **−9.16** |
| PZ7V onset | −18.05 dB | **−4.25** |
| Cat-A 13 mean | −25.86 dB | **−1.08** |

## §9 Deferred / future work

### §9.1 Gap #5 revisit (if Phase 2 tune doesn't resolve static regression)

Preserved branch `experiment/gap5-misadjustment` + β0 probe data give a clean starting point。Replace `err/mic` trigger with `err_refined / err_coarse` (using Phase 2 shadow as coarse) and add `dt_from_shadow < 0.3` gate。Estimated scope: 30-line addition to existing branch。

### §9.2 Phase 2 Stage 2 Priority 2-4 (if aggregate regression to be addressed)

Env flag `AEC_EXP_SHADOW_COPY_THR` mechanism in place on branch。Follow-up could try hysteresis/streak tune (Priority 2) or PBFDAF shadow `alpha_r_scale` tune (Priority 3)。[phase2_stage_2_prelim_diagnostic.md §5](phase2_stage_2_prelim_diagnostic.md) has candidate order。

### §9.3 Gap #16 Render-based estimator (from [aec3_full_architecture_analysis.md §6.5](aec3_full_architecture_analysis.md))

Future-deferred: revisit only if Gap #5 stacked layer can't close residual leak pockets post-tune。Dependencies: Gap #13 Filter Analyzer + robust ERL estimator。Scope ~100 lines。

### §9.4 Not to pursue

- Phase γ (Saturation) — demoted in bisect plan, no target cases
- Phase δ (DTD/RES) — demoted in bisect plan, no target cases
- B-15 raw_dt delay-aligned fix — superseded by B-16 (see [b15_superseded.md](b15_superseded.md))

## §10 Session statistics

- **Commits on main this session**: 0 (no new merges)
- **Commits on branches this session**:
  - `experiment/phase2-stage2-threshold-055`: 4 (spec → impl → Gap #16 doc → results)
  - `experiment/gap5-misadjustment`: 4 (spec → impl → sustain fix → results)
- **Benchmarks run this session**:
  - 800-case × 2 runs (Stage 2 tune, Gap #5 stacked) + AECMOS for each
  - 9-case smoke × 4 combos + 3 big-loss probe + Gap #5 smoke v1/v2
- **Preserved artifacts**:
  - `docs/benchmarks/phase2_stage_2/aecmos_thr055.log.gz`
  - `docs/benchmarks/gap5_misadjust/aecmos_111.log.gz`
  - `docs/benchmarks/gap5_misadjust/beta0_probe_pz7v.log.gz`

## §11 Handoff notes for next session

- Working tree on `main` has pre-existing untracked files from earlier sessions (`python/baselines/*`, various `.md` and `.py` debug artifacts) — **not touched this session**, not part of handoff
- Pre-existing `D python/test_shadow.py` deletion carried across session — **not touched**
- All this session's changes are on reference branches, fully committed
- Next session can cleanly `git checkout` any reference branch for continuation

Session complete。
