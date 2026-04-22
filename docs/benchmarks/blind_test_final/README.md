# Blind Test Final — 800-case 2×2 AECMOS

Date: 2026-04-22。Consolidated from Phase 1 (B-16) + Phase 2 (shadow-as-NLMS) Stage 1D runs。

## §1 Scope

Test set: `wav/aec_challenge_blind/` (AEC Challenge blind 800 cases = FS 300 + DT 300 + NE 200)。

4 flag combinations × 1600 output wav files each (ours + ours_nores):

| tag | `AEC_FIX_B16` | `AEC_EXP_SHADOW_NLMS` | Output dir | AECMOS log |
|---|---:|---:|---|---|
| (0,0) baseline | 0 | 0 | `python/baselines/b16_baseline/` | `docs/benchmarks/b16_stage_1d/baseline_aecmos.log.gz` |
| (1,0) B-16 alone | 1 | 0 | `python/baselines/b16_on/` | `docs/benchmarks/b16_stage_1d/b16on_aecmos.log.gz` |
| (0,1) Phase 2 alone | 0 | 1 | `python/baselines/p2_b16_0_p2_1/` | `docs/benchmarks/phase2_stage_1d/p2_aecmos_01.log.gz` |
| (1,1) stacked | 1 | 1 | `python/baselines/p2_b16_1_p2_1/` | `docs/benchmarks/phase2_stage_1d/p2_aecmos_11.log.gz` |

All 4 runs against same blind set — this IS the production-facing benchmark; no separate "blind-only" test exists in this repo。

## §2 Aggregate Pareto

| combo | FS_echo | DT_echo | DT_deg | NE_deg |
|---|---:|---:|---:|---:|
| (0,0) baseline | 3.475 | 4.034 | 2.448 | 4.005 |
| (1,0) B-16 | 3.475 | 4.033 | 2.449 | 4.005 |
| (0,1) Phase 2 | **3.445** | 4.039 | 2.458 | 4.012 |
| (1,1) stacked | **3.447** | 4.037 | 2.459 | 4.012 |

Δ vs (0,0):

| combo | ΔFS_echo | ΔDT_echo | ΔDT_deg | ΔNE_deg |
|---|---:|---:|---:|---:|
| (1,0) | +0.000 | −0.001 | +0.001 | +0.000 |
| (0,1) | **−0.030** | +0.005 | +0.010 | +0.007 |
| (1,1) | **−0.028** | +0.003 | +0.011 | +0.007 |

## §3 Movement / static subset (FS, DT)

### §3.1 FAREND SINGLETALK (300 cases, 131 movement / 169 static)

| subset | (0,0) | Δ(1,0) | Δ(0,1) | Δ(1,1) |
|---|---:|---:|---:|---:|
| ALL | 3.475 | +0.000 | **−0.030** | **−0.028** |
| movement | 3.571 | −0.000 | **−0.016** | −0.011 |
| static | 3.400 | +0.001 | **−0.040** | **−0.042** |

**Pattern**: FS regression concentrated in **static subset** (−0.040 to −0.042)。Movement subset milder (−0.011 to −0.016)。

### §3.2 DOUBLETALK (300 cases, 114 movement / 186 static)

| subset | (0,0) | Δ(1,0) | Δ(0,1) | Δ(1,1) |
|---|---:|---:|---:|---:|
| ALL | 4.034 | −0.001 | +0.005 | +0.003 |
| movement | 3.978 | −0.002 | **−0.017** | **−0.019** |
| static | 4.068 | −0.001 | **+0.018** | +0.016 |

**Pattern**: DT subsets cancel (movement regress, static improve)。Aggregate slightly positive。

## §4 FS big wins / losses (Δ > ±0.2)

| combo | Big wins | Big losses | Net |
|---|---:|---:|---:|
| (1,0) B-16 | 0 (0 mv) | 0 (0 mv) | 0 |
| (0,1) Phase 2 | **21** (8 mv) | **40** (13 mv) | **−19** |
| (1,1) stacked | **23** (9 mv) | **38** (11 mv) | **−15** |

B-16 AECMOS-neutral。Phase 2 trades 21 wins for 40 losses on FS。Stacking slightly improves net (−15 vs −19), stacking helps marginal cases。

## §5 Cat-A 13-case wav-level secondary

Cat-A = 12 PZ7V-class gain-jump cases + PZ7V itself ([bisect_analysis_and_plan.md](../../bisect_analysis_and_plan.md))。

| combo | Cat-A mean leak | PZ7V full | PZ7V onset |
|---|---:|---:|---:|
| (0,0) baseline | −24.78 dB | −17.07 | −13.81 |
| (1,0) B-16 | −24.85 | −17.43 | **−20.66** |
| (0,1) Phase 2 | −25.68 | −24.43 | −13.80 |
| (1,1) stacked | **−25.86** | **−26.22** | −18.05 |

**Architectural mechanisms separated**:
- B-16 kills PZ7V **onset-window** leak (+6.85 dB), minimal full-file effect
- Phase 2 cuts PZ7V **full-file** leak (+7.37 dB), no onset-window effect
- Stacked: full-file best (−9.16 dB); onset window 2.62 dB **worse than B-16 alone** (interaction, see [phase2_stage_1b_interaction_note.md](../../phase2_stage_1b_interaction_note.md))

## §6 Verdict (vs Stage 1D)

**Consistent with Stage 1D — MARGINAL**:
- AECMOS FS_echo regresses 0.028 aggregate
- Wav-level clear PZ7V-class improvement
- Pareto trade, no free win

Blind test does not reveal different verdict from Stage 1D because Stage 1D **was** the blind test — `wav/aec_challenge_blind/` is the same set used for both。

## §7 Recommendation

### §7.1 Production default

**Keep all flags default OFF** (no change to current main `1372198`):
- (0,0) baseline has lowest risk
- AECMOS FS_echo 0.028 regression exceeds typical "harmless experimental" tolerance
- No metric is strictly dominated by (1,1) for all case classes

### §7.2 Research / opt-in

**Recommended opt-in**: `AEC_FIX_B16=1 AEC_EXP_SHADOW_NLMS=1` (both ON, stacked):
- Best Cat-A wav-level (−25.86 dB mean)
- Best PZ7V full-file (−26.22 dB)
- Stacked big-wins/losses net (−15 vs Phase 2 alone −19)
- Acceptable for PZ7V-class investigation、data collection、non-AECMOS-benchmarked deployment

### §7.3 Next step options

| Option | Rationale | Cost |
|---|---|---|
| **Accept current (do nothing)** | 2-layer stack validated opt-in, Gap #5 / Stage 2 proved current implementations insufficient。Ship as-is。 | Zero |
| **Gap #13 Filter Analyzer** | Provides `err_refined/err_coarse` consistency signal — prerequisite for Gap #5 revisit with correct trigger; could also power alternative Stage 2 gates | ~100 lines, 1 week |
| **Gap #9 Usable-linear gate** | From aec3_full_architecture_analysis.md §6.3, lower complexity than Gap #5, could close residual cases without disturbing stable FS | ~80 lines, 1 week |
| **Phase 2 Stage 2 retry (different priority)** | Priority 2 (hysteresis) or 3 (alpha_r_scale) via existing env flag mechanism on preserved branch | ~20 lines, 1 day |

**Recommendation**: **Option 1 (accept)**。2-layer stack is the current architectural ceiling。Further improvements need new signal sources (Gap #13 / Gap #9) which are ~1-week scoped experiments。Current stack is stable, documented, well-characterized。

## §8 Artifacts

- This doc: `docs/benchmarks/blind_test_final/README.md`
- Analysis script: `docs/benchmarks/blind_test_final/analyze.py`
- Console output: `docs/benchmarks/blind_test_final/summary.txt`
- Raw AECMOS logs (pre-existing): `docs/benchmarks/{b16_stage_1d,phase2_stage_1d}/*.log.gz`
- Raw output wavs: `python/baselines/{b16_baseline,b16_on,p2_b16_0_p2_1,p2_b16_1_p2_1}/`
