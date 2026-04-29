# Phase 2 Stage 1D Results — MARGINAL verdict

Date: 2026-04-22. Branch `feature/phase2-shadow-as-nlms`.
Spec: [spec_phase2_shadow_as_nlms.md](spec_phase2_shadow_as_nlms.md) (v4).
Interaction note: [phase2_stage_1b_interaction_note.md](phase2_stage_1b_interaction_note.md).

## 1. Scope

- 2×2 flag matrix: `AEC_FIX_B16 × AEC_EXP_SHADOW_NLMS` = {(0,0), (1,0), (0,1), (1,1)}
- 800-case AEC Challenge blind set (FS 300 + DT 300 + NE 200)
- Baseline reused: `python/baselines/b16_baseline` = (0,0), `b16_on` = (1,0) from Phase 1 B-16 Stage 1D
- New runs: `p2_b16_0_p2_1` = (0,1), `p2_b16_1_p2_1` = (1,1)
- Movement label: filename substring `_with_movement_`

## 2. AECMOS 800-case aggregate

| combo | FS_echo | DT_echo | DT_deg | NE_deg |
|---:|---:|---:|---:|---:|
| (0,0) baseline | 3.475 | 4.034 | 2.448 | 4.005 |
| (1,0) B-16 | 3.475 | 4.033 | 2.449 | 4.005 |
| (0,1) Phase 2 | **3.445** | 4.039 | 2.458 | 4.012 |
| (1,1) stacked | **3.447** | 4.037 | 2.459 | 4.012 |

Δ vs (0,0):

| combo | ΔFS_echo | ΔDT_echo | ΔDT_deg | ΔNE_deg |
|---:|---:|---:|---:|---:|
| (1,0) | +0.000 | −0.001 | +0.001 | +0.000 |
| (0,1) | **−0.030** | +0.005 | +0.010 | +0.007 |
| (1,1) | **−0.028** | +0.003 | +0.011 | +0.007 |

**Summary**: B-16 NEUTRAL (matches Phase 1)。Phase 2 causes small FS_echo regression (−0.030), small DT_echo / DT_deg / NE_deg improvements。

## 3. Movement vs static subset (FS)

AECMOS FS_echo per subset:

| subset (n) | (0,0) | (1,0) | (0,1) | (1,1) | Δ(1,0) | Δ(0,1) | Δ(1,1) |
|---|---:|---:|---:|---:|---:|---:|---:|
| ALL (300) | 3.475 | 3.475 | 3.445 | 3.447 | +0.000 | **−0.030** | **−0.028** |
| movement (131) | 3.571 | 3.570 | 3.555 | 3.560 | −0.000 | **−0.016** | −0.011 |
| static (169) | 3.400 | 3.401 | 3.360 | 3.359 | +0.001 | **−0.040** | **−0.042** |

AECMOS DT_echo per subset:

| subset (n) | Δ(0,1) | Δ(1,1) |
|---|---:|---:|
| ALL (300) | +0.005 | +0.003 |
| movement (114) | **−0.017** | **−0.019** |
| static (186) | **+0.018** | **+0.016** |

**Key observation**:
- FS regression **concentrated in static** (169 cases, −0.040), not movement (131 cases, −0.011)
- DT exhibits mirror pattern: movement regresses slightly (−0.017), static improves (+0.018)
- Two subsets largely cancel out in DT aggregate; FS 無 cancel because movement is fairly flat

反直覺: [phase2_stage_1b_interaction_note.md](phase2_stage_1b_interaction_note.md) 預測 movement 會較糟 (B-16 × shadow DTD interaction), 實際 static 更糟 — suggests 另一 mechanism at play。

## 4. FS big wins / big losses (0,1) vs (0,0)

Threshold: |Δecho_mos| > 0.2。

| Category | Count | Movement | Static |
|---|---:|---:|---:|
| Big wins (Δ > +0.2) | 21 | 8 | 13 |
| Big losses (Δ < −0.2) | 40 | 13 | 27 |

**Top 5 wins**:
- +1.429 sZ9Egg (static)
- +1.180 JTlOYE (static)
- +0.938 nyT6FUUdu0W8 (movement)
- +0.652 **PZ7V** (static, target case)
- +0.606 yZs0i (static)

**Top 5 losses**:
- −0.722 KSN5Jr (static)
- −0.294 IUp2cA (static)
- −0.234 I2bme (movement)
- −0.234 LeV1uF (static)
- −0.214 HAxmF7v4dE0i (static)

**Net**: big losses outnumber big wins 40 vs 21。Phase 2 is **not systemic regression**; trade-off: fixes a batch of gain-jump cases (inc PZ7V) but disrupts a different batch (mostly static)。

## 5. Wav-level secondary metric (Cat-A 13 cases)

Cat-A source: [bisect_analysis_and_plan.md](bisect_analysis_and_plan.md) top-20 FS gap 的 12 non-movement + PZ7V。

| Metric | (0,0) | (1,0) | (0,1) | (1,1) | Δ(0,1) | Δ(1,1) |
|---|---:|---:|---:|---:|---:|---:|
| Full-file mean (dB) | −24.78 | −24.85 | −25.68 | **−25.86** | −0.90 | **−1.08** |
| Onset window mean (dB) | −22.47 | −23.23 | −23.03 | **−23.50** | −0.56 | **−1.03** |

**Spec v3 §7.2 thresholds**:
- Full-file ≥ 2 dB improvement → actual **1.08 dB** → MARGINAL (未達 PASS)
- Onset ≥ 3 dB improvement → actual **1.03 dB** → MARGINAL (未達 PASS)

**PZ7V individual** (target case):

| Metric | (0,0) | (1,0) | (0,1) | (1,1) | Best Δ |
|---|---:|---:|---:|---:|---:|
| Full-file (dB) | −17.07 | −17.43 | −24.43 | **−26.22** | **−9.16** |
| Onset window (dB) | −13.81 | **−20.66** | −13.79 | −18.04 | **−6.85** (B-16 alone) |

PZ7V PASS++ on full-file;onset 最佳 is (1,0) 單靠 B-16;(1,1) onset 2.62 dB **worse than (1,0) alone** — interaction note 在 PZ7V 上 confirmed。

## 6. §7.4 Matrix interaction

| Metric | Δ(1,0) | Δ(0,1) | Additive expected | Δ(1,1) actual | Pattern |
|---|---:|---:|---:|---:|---|
| AECMOS FS_echo (800) | +0.000 | −0.030 | −0.030 | −0.028 | **Substitution** (Phase 2 dominates) |
| Cat-A full-file (13) | −0.07 | −0.90 | −0.97 | −1.08 | **Slight synergy** (+0.11) |
| Cat-A onset (13) | −0.76 | −0.56 | −1.32 | −1.03 | **Partial cancel** (0.29 worse than additive) |
| PZ7V onset (1) | −6.85 | +0.02 | −6.83 | −4.23 | **Cancel** ((1,1) worse than (1,0) by 2.6 dB) |

PZ7V-specific onset cancel is real (matches Stage 1B observation); aggregate Cat-A onset doesn't cancel。§9.3 "(1,1) < (1,0)" 對 PZ7V-onset 為 True, 對 aggregate 不成立。

## 7. Verdict per spec v3 §9 dual-window

| 判準 | 結果 | Tier |
|---|---|---|
| AECMOS systemic regression > 0.05 dB | −0.030 FS_echo | 容差內 |
| Cat-A full-file ≥ 2 dB improvement | 1.08 dB | MARGINAL |
| Cat-A onset ≥ 3 dB improvement | 1.03 dB | MARGINAL |
| PZ7V full-file ≥ 5 dB improvement | 9.16 dB | PASS++ |
| PZ7V onset ≤ −22 dB | (1,1) −18.04 dB | ≥ baseline −5 dB (PASS v3) |
| DT/NE regression | No systematic | PASS |
| (1,1) aggregate vs (1,0) | ΔFS essentially equal; Cat-A onset 略優 | 容差內 |
| (1,1) PZ7V onset vs (1,0) | (1,1) worse 2.62 dB | §9.3 warning |

**Overall: MARGINAL (tier 2)** — hypothesis validated on target case, aggregate sub-threshold。

## 8. Decision — Option A merge flag-gated OFF

User decision: **merge with `AEC_EXP_SHADOW_NLMS` default OFF** + EXPERIMENTAL status banner。

Rationale:
- Architectural fix validated (PZ7V +9.16 dB demonstrates shadow-as-NLMS correctly reviving copy gate)
- Flag OFF = bit-exact baseline (invariant §5.1), zero production impact
- Preserves spec + artifact + knowledge for Stage 2
- Full revert cost: `git revert` single merge commit

## 9. Stage 2 direction

[phase2_stage_2_prelim_diagnostic.md](phase2_stage_2_prelim_diagnostic.md) 對 3 個 big-loss cases (KSN5Jr, IUp2cA, LeV1uF) 的 copy-gate timeline probe, 用於 Stage 2 tuning priority。

候選 tune parameters (順序需 diagnostic doc 結果確認):
1. `shadow_copy_threshold` (0.65): 提高 → 更嚴格觸發 copy
2. `shadow_copy_hysteresis` (3) / `_shadow_advantage_streak` (min 10): 延長觀察期防止 spurious copy
3. PBFDAF shadow `alpha_r_scale` (1.03) / `leak` (1e-6): 調差異化參數降低 static cases 的 shadow→main copy 誤觸

Stage 2 goal: preserve PZ7V-class fix (−9 dB) while reducing static subset regression (−0.040 → < −0.01)。

## 10. Artifacts

- AECMOS logs:
  - (0,0) `docs/benchmarks/b16_stage_1d/baseline_aecmos.log.gz` (reused)
  - (1,0) `docs/benchmarks/b16_stage_1d/b16on_aecmos.log.gz` (reused)
  - (0,1) `docs/benchmarks/phase2_stage_1d/p2_aecmos_01.log`
  - (1,1) `docs/benchmarks/phase2_stage_1d/p2_aecmos_11.log`
- Per-case AEC output wavs:
  - (0,1) `python/baselines/p2_b16_0_p2_1/`
  - (1,1) `python/baselines/p2_b16_1_p2_1/`
- Cat-A wav-level per-case JSON: `docs/benchmarks/phase2_stage_1d/p2_cat_a.json`
- Analysis scripts: `/tmp/analyze_p2_1d.py`, `/tmp/ms2.py`
