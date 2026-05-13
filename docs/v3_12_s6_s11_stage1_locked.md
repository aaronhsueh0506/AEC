# v3.12 Stage 1 surface five-trial NEUTRAL — strategic decision required

**Date**: 2026-05-13
**Branch**: `feature/v3.11-s9-noise-floor-refine` (parent: `feature/v3.11-route-a`)
**Predecessors**: [v3_12_s6b_verdict.md](v3_12_s6b_verdict.md) /
[v3_12_s7_verdict.md](v3_12_s7_verdict.md) /
[v3_12_s8_verdict.md](v3_12_s8_verdict.md) /
[v3_12_s9_verdict.md](v3_12_s9_verdict.md)
**Status**: **Stage 1 RES surface exhausted — v3.12 GA blocker; user decision needed**

## TL;DR

Five consecutive 800-case AECMOS benches (BALANCED, fl=832, cng=True,
seed=0) on the Stage 1 RES surface all close NEUTRAL with Δ ≈ ±0.001
on every bucket. The benches attack every meaningful gate on
ENR-input both denominator-side and numerator-side, with measured
pre-audit magnitude impact of 7-23 dB on individual signal paths.
Yet the bench output is **structurally locked** to baseline.

| Sprint | Target | Pre-audit FS magnitude | AECMOS verdict |
|---|---|---|---|
| **S6** | `ne_g_floor` removal (denominator) | ~5 dB shift | Δ ≈ 0.000 across all buckets |
| **S6b** | `epc_dt_cap` retirement (denominator) | cap fires 0/2M frames | Δ ≈ 0.000 across all buckets |
| **S7** | `dt_per_bin` unified (denominator) | absorbed by downstream | Δ ≈ 0.000 across all buckets |
| **S10** | `noise_floor_psd` refinement (denominator) | 11.71 dB on FS_static | Δ ≈ 0.000 across all buckets |
| **S11** | `Cap2` FS-loosen (numerator) | 12.34 dB on FS_static | Δ ≈ 0.000 across all buckets |

Worst FS-static case `7GTxyTksSUqCnP5y0ILG4A` echo score:
- baseline (v3_12_s7_off): 1.478
- S10 flag-ON: 1.477
- S11 flag-ON: 1.475

Even on the case with the loudest FS leak, all five Stage 1 attack
directions move AECMOS by ≤0.003 score points (≈ inaudible).

## Mechanism inventory exhausted

The Stage 1 RES surface has 5 known gain-floor / cap paths
(per Q7 V3 verdict in hazy-lynx plan §7). S6-S11 attacked them all:

| Path | Mechanism | Sprint | Result |
|---|---|---|---|
| `ne_g_floor` | gain-floor (denominator) | S6 | NEUTRAL |
| `spectral_floor` | gain-floor (denominator) | not attacked (physical fallback) | — |
| `epc_dt_cap` | gain cap (denominator) | S6b | NEUTRAL (dead code) |
| `quiet_mask` | gain mask (denominator) | not attacked (physical fallback) | — |
| `divergence_floor` | gain floor (denominator) | not attacked (links filter_state) | — |
| `dt_per_bin` | gain weight (denominator) | S7 | NEUTRAL |
| `noise_floor_psd` | nearend_est floor (denominator) | S10 | NEUTRAL |
| `min_ne_from_dt` | nearend_est floor (denominator) | S9-C pre-audit only | shown structurally redundant |
| `ne_physical_floor` | nearend_est floor (denominator) | S9-D pre-audit only | reveals fundamental floor |
| `Cap2` | residual_echo cap (numerator) | S11 | NEUTRAL |
| `Cap1` (`echo × 2`) | residual_echo cap (numerator) | not attacked (S8 binds 0.1%) | dead |
| `Cap3` (`dt_suppress`) | residual_echo cap (numerator) | not attacked (S8 binds 1%) | near-dead |
| `Cap4` (`render_ceil`) | residual_echo cap (numerator) | not attacked (S8 binds 0.3%) | near-dead |

The unattacked items either bind <2% in FS (Cap1/3/4 = ineffective by
S8 measurement) or are physical fallbacks that should not be removed
(spectral_floor, quiet_mask, ne_physical_floor).

## Why the 5-NEUTRAL pattern

The bench data falsifies the **carrier-in-Stage-1 hypothesis** that
drove the v3.12 plan. Specifically:

### Mechanism 1: Gain-pipeline absorption

In v3.11.x BALANCED, the gain pipeline post-processing (softgate
clamp + 3-bin smooth + hf_cap + temporal smoothing + g_min floor =
10^(g_min_db/20)) compresses any Stage 1 magnitude change into the
same output range. When Stage 1 raises ENR by 12 dB:
- If gain was already at `g_min`, no further reduction possible
- If gain was mid-range, softgate exponent absorbs the shift

### Mechanism 2: FS leak is filter residual, not gain residual

In the worst-FS case (`7GTxyTksSUqCnP5y0ILG4A` echo = 1.478), the
gain pipeline is likely already at minimum suppression. The audible
echo is the **residual signal after gain × output**: if filter
residual is X dB above floor and gain is already at `g_min = -X dB`,
output residual = 0 dB above floor = audible.

Reducing gain further beyond `g_min` is impossible — Stage 1
input changes can't help in this regime.

The FS leak carrier therefore lives **upstream of RES** — in the
PBFDKF main filter convergence quality / shadow filter handoff /
EPC handling. These are the Q1/Q2/Q4/Q5 surfaces of the hazy-lynx
plan (filter side), not the Q3/Q6/Q7 surfaces (RES side).

## Strategic decision options

### Option A — Accept v3.11.x as production ceiling

Conclude that the current `BALANCED` preset is at the local
optimum for the existing AEC algorithm. Close v3.12 with no new
promotions. Document the 5-NEUTRAL finding as a hard ceiling.

Pros:
- Respects empirical evidence (5 disciplined sprints)
- No further sprint cost
- v3.11.x already shipped Phase 1 wins (B5/F-E5/diverged_reset)

Cons:
- Worst FS cases still have audible leak (7GTxy echo=1.478)
- Doesn't advance the architecture

### Option B — Pivot to filter-side fixes (hazy-lynx Q1/Q2/Q4/Q5)

Stop attacking Stage 1 RES. Restart from the **filter side**:
PBFDKF, ShadowFilter, PathChangeRegimeHandler, EPC, delay-tracking.
These are the Q1 (dual filter) / Q2 (convergence stability) /
Q4 (state machine) / Q5 (dual filter redesign) surfaces in the
hazy-lynx plan.

Specific candidates (already in plan):
- F-DelayTrack (continuous delay variance)
- F-E3 redesign (EPC consecutive-detection)
- F-HFR per-band Q/R
- Path Change Regime Handler tuning (with cohort tail guard)

Risk: HIGH (touches load-bearing surfaces; cohort tail catastrophe
defence must be preserved).

### Option C — Phase 4 architectural refactor (Q7 V3 unified floor)

Despite the 5-NEUTRAL evidence, proceed with the canonical
Beroutti / Ephraim-Malah / Hänsler-Schmidt 6-step gain refactor.
This restructures the gain pipeline itself (not just the inputs)
and may break the structural lock observed in S6-S11.

Risk: MED-HIGH (architectural change; hard abort if any bucket
regresses). Could take 4-6 weeks across S12-S20 sprints.

### Option D — Listen to worst cases + AECMOS sanity

Before more sprint work, run subjective listen on the 5 worst FS
cases (echo < 2.0 in baseline). Confirm that:
- The echo is audible (AECMOS isn't artifact)
- The S10/S11 outputs sound the same (NEUTRAL is real)

If listens confirm NEUTRAL → Option A or B is the call.
If S11 sounds slightly better than baseline despite Δecho=0.000 →
AECMOS metric isn't measuring the change; proceed with caution.

LOE: 30 min listen + decision.

## Critical invariants (preserved across all 5 NEUTRAL sprints)

- Cohort tail (`qNvSMyUSXUyrDGp`) Δecho preserved
- NE Δdeg ≥ −0.005 (no NE damage)
- DT Δdeg ≥ −0.005
- Byte-equal flag-OFF (each sprint maintains this invariant)

## Sprint cost so far on v3.12 RES surface

| Sprint | Render time | Score time | Total wall clock |
|---|---|---|---|
| S6/S6b/S7 | ~12 min × 2 | ~5 min × 2 | ~35 min (prior) |
| S10 | 12.7 min | 5 min | ~18 min |
| S11 | 11.8 min | 5 min | ~17 min |
| **Pre-audits** | S8 (~13 min) + S9-A/C/D (~40 min total) | — | ~53 min |
| **TOTAL** | — | — | ~3 hours sprint compute + ~12 hours pre-audit/design |

## Recommendation

**Run Option D listen first (30 min)** then make the strategic
decision. If listen confirms NEUTRAL is real, recommend Option A
(accept ceiling) for short-term + Option B (filter-side pivot)
for the v3.13 arc.

Do NOT proceed with Option C (Phase 4 unified-floor) — the
5-NEUTRAL evidence falsifies the premise that the gain pipeline
input layer is the bottleneck. Phase 4 may still be worth doing
as code quality / canonical-coherence cleanup, but it won't
move AECMOS.

## Sources

- 5 NEUTRAL bench results (S6/S6b/S7/S10/S11)
- S8 audit JSON with Stage 1 binding distribution
- S9-A/C/D pre-audit data on nearend_est floor stack
- Q7 V3 verdict (hazy-lynx plan §7)
- v3.11 Phase 1 verdicts (`docs/v3_11_phase1_final_review.md`)
