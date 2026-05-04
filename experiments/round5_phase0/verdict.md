# Round 5 — Phase 0 Binding Audit verdict

**Branch**: `algo/round5-coupled-audit`  •  **Date**: 2026-05-03
**Cases**: 800 seeded BALANCED, baseline = `baseline_v381_seeded`

## Headline finding

**RES is doing exactly what it's designed to do (trade DT deg for DT echo)
but the trade is grossly asymmetric on worst-DT_mv cases**. The
binding stage is NOT the soft-gate — it's the downstream temporal +
noise-floor lift, which erases worst-case-targeted suppression more
aggressively than best-case suppression.

## Bucket means: linear (nores) vs full (ours)

| Bucket | nores echo | ours echo | RES echo+ | nores deg | ours deg | RES deg- |
|---|---:|---:|---:|---:|---:|---:|
| FS_static | 2.620 | 3.768 | **+1.148** | 5.000 | 4.999 | -0.001 |
| FS_movement | 2.616 | 3.827 | **+1.211** | 5.000 | 4.999 | -0.001 |
| NE | 4.998 | 4.998 | +0.000 | 4.120 | 4.007 | -0.113 |
| DT_static | 3.164 | 4.251 | **+1.087** | 3.306 | 2.257 | **-1.049** |
| DT_movement | 3.153 | 4.120 | **+0.967** | 3.283 | 2.286 | **-0.997** |

**Linear AEC alone wins DT deg by ~1.0 point** but loses ~1.0 point of DT
echo. RES net contribution: ~+1 echo at cost of ~-1 deg on DT, free
+1.1 echo on FS (no NE → no deg cost), modest -0.1 NE deg.

Compared to AEC2 reference (CHANGELOG): AEC2 is at FS 3.5 / DT 4.26 / NE deg 4.10.
- Our linear-only loses echo to AEC2 by 0.7–1.1 pts (AEC2 has NLP too)
- Our final BEATS AEC2 on FS_st echo (+0.27), DT_st echo (-0.01), DT_mv echo (-0.03 ≈ tie)
- DT_st deg ours 2.257 vs AEC2 ~2.30, DT_mv deg ours 2.286 vs AEC2 ~2.53 (we win deg)

So RES is NOT the bottleneck against AEC2 — final v3.8.1 BALANCED already
≈ AEC2 on every bucket. The DT_mv worst-20 gap (where users want to
recover ≥0.02 deg) is **per-case-asymmetric**, not bucket-mean-deficient.

## DT_mv worst-20 individual case pattern (20/20 same direction)

All 20 worst cases by ours deg (baseline_deg ≈ 1.16–1.55):

| metric | mean Δ (ours − nores) | range |
|---|---:|---|
| Δdeg | **-1.21** (RES costs deg) | −2.07 to −0.18 |
| Δecho | **+1.07** (RES adds echo) | +0.24 to +1.96 |

Out of 20 worst-20 cases, **all 20** have Δecho > 0 (RES helps echo) and
**all 20** have Δdeg < 0 (RES hurts deg). The deg loss exceeds 1.0 on 12/20
cases. The trade-off is real but the per-case ratio is roughly 1:1 (gain
1 echo, lose 1 deg) — much worse than bucket-mean ratio (which is dominated
by mid-case wins).

## Stage gain trajectory — the binding chain

DT_mv worst-20 vs best-20 (locked by ours deg, baseline_v381_seeded scores):

| stage | worst-20 g | best-20 g | gap | gap closure |
|---|---:|---:|---:|---:|
| **soft-gate (post-EMR)** | 0.249 | 0.496 | **-0.247** | 0% (gap full) |
| spectral_floor | 0.256 | 0.504 | -0.248 | flat |
| epc_dt_cap | 0.256 | 0.504 | -0.248 | flat |
| quiet_mask | 0.258 | 0.507 | -0.248 | flat |
| 3bin_smooth | 0.108 | 0.320 | -0.212 | 14% closed |
| hf_cap | 0.072 | 0.253 | -0.181 | 27% closed |
| pre_temporal | 0.237 | 0.466 | -0.229 | (gap re-opens) |
| **post_temporal** | 0.368 | 0.469 | **-0.101** | **59% closed** |
| **after_noise_lift (final)** | **0.535** | 0.581 | **-0.046** | **81% closed** |

**The story**:
1. Soft-gate **correctly** distinguishes worst (0.25) from best (0.50)
   based on residual+coh signals.
2. spectral_floor / epc / quiet_mask preserve the gap.
3. 3bin_smooth + hf_cap pull both down (cross-bin smoothing + HF cap),
   slightly narrowing the gap.
4. **post_temporal lifts worst-20 from 0.237 to 0.368 (+0.131) but lifts
   best-20 only 0.466 → 0.469 (+0.003)** — temporal smoothing
   asymmetrically erases worst-case-targeted suppression.
5. **after_noise_lift lifts worst-20 from 0.368 to 0.535 (+0.167) but
   lifts best-20 only 0.469 → 0.581 (+0.112)** — noise-floor lift again
   adds disproportionately to worst.

Final gain ratio worst:best = 0.535/0.581 = 92% — soft-gate's intended
ratio (0.249/0.496 = 50%) gets erased to near-equality. **RES suppresses
all DT_mv cases similarly at output, regardless of soft-gate's per-frame
echo evidence.**

## Why R4 R1 (per-bin cap on soft-gate) was null

R1 capped gain at the soft-gate stage. From the trajectory above:
- soft-gate g_voice already at ~0.5 average (cap=0.5 had no effect)
- Even cap=0.3 was erased by post_temporal (+0.13) and noise_floor (+0.17)
  on the worst-20 cases — the very cases we needed it to act on.

**The per-bin cap had no chance.** Any soft-gate-level intervention will
be lifted by ~0.30 in absolute g terms by the time it reaches output.

## Decision per plan tree

| Pattern | Result |
|---|---|
| raw_score 已輸 AEC2 (filter bottleneck) | NO — raw wins deg, loses echo (RES adds echo at deg cost) |
| raw OK, final 輸 (RES bottleneck) | PARTIAL — final ≈ AEC2 on bucket means; bottleneck is per-case worst-20 |
| nores 比 final 好 | False on bucket; on worst-20, nores deg is much higher (3.28 vs 2.29) but echo lower (3.15 vs 4.12) — depends on metric |
| final 比 nores 好但仍輸 AEC2 | False — final ≈ AEC2 (within 0.1) on all buckets |

**Real conclusion**: bucket-mean game is over (≈ AEC2). The DT_mv worst-20
problem is a **suppression-symmetry** problem: temporal + noise_floor
stages don't preserve per-case soft-gate decisions. To recover deg on
worst cases without losing echo on best cases, **the lift stages must be
made selective** — bypass on cases (or bins) where soft-gate said "this
is heavy NE", but preserve on cases where soft-gate said "let it through".

## R5 main line

**PROCEED with Coupled Test A — but A3 (noise-floor bypass) and A4
(faster temporal attack) are the targets**, not A2 (spectral_floor)
which Phase 0 shows is not a binding stage.

**Modified A1–A4 priorities**:
- A1 (softgate cap only): SKIP — Phase 0 confirms it's downstream-erased
- A2 (cap + spectral_g_min lower): SKIP — spectral_floor is not binding
- **A3 (cap + noise_floor_gain bypass on echo-dominant)**: PRIMARY
- **A4 (cap + faster temporal attack on echo-dominant)**: PRIMARY

The classifier from R4 (`coh2 > 0.5 & res/err > 0.5 & ne/err < 0.3`) had
3% hit rate. For A3/A4, the classifier needs to fire on the right
**frames**, not bins — gate temporal/noise lift at frame level when
soft-gate confirms echo dominance.

Better classifier candidate: `g_after_softgate < 0.2 AND res_over_err > 1.0`
(soft-gate already said heavy echo; protect that decision through the chain).

## Test B / C / D status

- B (EPC split): still warranted. Phase 0 shows EPC is rarely active in
  worst-20 DT_mv (existing R3 trace: epc_pct = 0.57 on worst, 0.35 on best).
  Less urgent than A.
- C (DT split): same root issue as A — `effective_dt` drives both NE
  protection (lifts gain) and adaptation freeze. Pause until A result.
- D (CNG/noise floor): noise_floor_gain is binding (Phase 0 confirms +0.17
  lift on worst). D3 (noise_floor bypass) is essentially same as A3 — collapse
  D3 into A3.

## Disposition

- Phase 0 trace plumbing kept on `algo/round5-coupled-audit` (audio-passive)
- 9-stage gain trace + raw/nores/final 3-way bench infrastructure ready
- Next: implement A3 + A4 on `algo/r5-A-noise-temporal-coupled` sub-branch
- Mechanism check before bench: must verify `g_after_noise_lift` for worst-20
  drops by ≥ 0.05 vs baseline before running 800-case
