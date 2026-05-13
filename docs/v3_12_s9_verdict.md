# S9 verdict — noise_floor_psd refinement closes NULL; Phase 4 RES canonical refactor confirmed

**Date**: 2026-05-13
**Branch**: `feature/v3.11-s9-noise-floor-refine` (parent: `feature/v3.11-route-a`)
**Predecessor**: [v3_12_s8_verdict.md](v3_12_s8_verdict.md)
**Status**: **S9 CLOSED — three-trial null (A/C/D); pivot Phase 4**

## TL;DR

Three pre-implementation audits on the production 800-case corpus
(BALANCED, fl=832, cng=True, seed=0) attacked the `nearend_est`
floor stack identified by S8 (43% baseline binding by
`noise_floor_psd`). The three trials yield **FS-bucket release rate
to raw `nearend_est` ≤ 15%** — but mean nearend_est magnitude
reduction grows from 7 dB (A.1) to 23 dB (D). The release-rate
metric tracks **winner-identity** (which floor wins); the magnitude
metric tracks **actual ENR impact** — both matter, and they diverge.

| Trial | Attack target | FS_static release | FS_movement release |
|---|---|---:|---:|
| **S9-A.1** | `noise_floor_psd` scalar × 0.001 (10× lower) | 8.23% | 10.01% |
| **S9-A.2** | `noise_floor_psd` → `error_psd × 0.005` (per-bin) | 11.53% | 14.25% |
| **S9-C.1** | A.2 + `min_ne_from_dt × 0.1` (joint) | 13.61% | 14.18% |
| **S9-C.2** | A.2 + `min_ne_from_dt → 0` (joint) | 13.61% | 14.18% |
| **S9-D** | A.2 + `min_ne → 0` + `ne_physical → 0` (sanity) | **14.05%** | **14.60%** |

**Key finding**: each successive floor attack reveals the next-lower
floor binding. Single-target attacks shift binding from #1 to #2; joint
2-floor attacks shift to #3 (`ne_physical_floor`); only attacking all
3 floors simultaneously could theoretically release FS bins to
`raw_NE × dt_shaped` — and S9-D measures whether even that
suffices. This is the structural pattern Q7 V3 verdict predicted:
the 5-path floor stack is the canonical-coherence bug; patching one
floor at a time can never fix it.

**Decision**: S9 audit closed; **next sprint must distinguish two
hypotheses**. The release-rate metric used in S9-A/C/D measured
winner-identity (which floor wins argmax). But the actual gain-
pipeline driver is **ENR magnitude**, and S9-D shows mean nearend_est
reduction of **23 dB** on FS_static under maximum 3-floor attack —
which would translate to 23 dB ENR rise → potential gain drop →
potential FS leak fix.

Two hypotheses to resolve via S10 flag-and-bench:
- **H1** — nearend_est magnitude is the carrier; FS-gated A.2
  (lower noise_floor coefficient + FS-confidence gate) produces
  Δecho improvement.
- **H2** — magnitude reduction is absorbed by downstream gain
  pipeline (ENR clamps, gain caps, smoothing); bench is NEUTRAL.
  Then Cap2 (residual_echo) becomes the priority pivot.

S10 chooses the simplest candidate (A.2 + FS gate) and benches it.

## Audit lineage

```
S6   ne_g_floor removal              → NEUTRAL (gated by (1-fs_confidence) already)
S6b  epc_dt_cap removal              → NEUTRAL (cap2 fires 0/2M frames)
S7   dt_per_bin_unified              → NEUTRAL (downstream absorbed shift)
S8   downstream nearend_est audit    → DIAGNOSTIC (noise_floor 43%, min_ne 37%)
S9-A noise_floor_psd refinement      → NULL (release 8-14% on FS)
S9-C joint noise_floor + min_ne      → NULL (release 13.6% on FS; ne_physical_floor takes over)
S9-D all-3-floor sanity              → _TBD_ — informs Phase 4 unified-floor target
```

## Step-by-step results

### Step 1 — S9-A pre-audit (Phase 3B v5)

Substrate added 14 counter fields to `ResAuditCounters` (commit on
`feature/v3.11-s9-noise-floor-refine`). Audit hook computes, for each
FS bin (coh² < 0.1) where baseline winner is `noise_floor_psd`, the
hypothetical winner under two candidates:
- A.1 = `mean(error_psd) × 0.001 + ε` (scalar, 10× lower than 0.01)
- A.2 = `error_psd × 0.005 + ε` (per-bin, ~half coefficient on per-bin)

#### FS bucket release fates (800-case)

| bucket | floor bins | A.1 release | A.1 shift→min_ne | A.1 stays | A.2 release | A.2 shift→min_ne | A.2 stays |
|---|---:|---:|---:|---:|---:|---:|---:|
| FS_static | 24,217,878 | **8.23%** | 44.03% | 47.74% | **11.53%** | 88.09% | 0.38% |
| FS_movement | 17,253,416 | **10.01%** | 41.38% | 48.62% | **14.25%** | 84.54% | 1.21% |
| DT_static | 57,540,070 | 29.25% | 19.22% | 51.52% | 58.35% | 41.50% | 0.15% |
| DT_movement | 33,822,834 | 29.98% | 17.37% | 52.65% | 60.19% | 39.65% | 0.16% |
| NE | 25,645,720 | 36.51% | 2.92% | 60.57% | 90.56% | 9.44% | 0.00% |
| GLOBAL | 158,479,918 | 25.27% | 22.39% | 52.33% | 52.00% | 47.72% | 0.28% |

**Magnitude**: A.1 mean dB reduction 7.19–7.93 dB; A.2 mean dB
reduction 11.71–16.24 dB. A.2 intrusion outside baseline floor = 0%
across all buckets (sanity passes).

**Observation**: In FS bins, A.2 shifts 88% of `noise_floor` bins to
`min_ne_from_dt`. The release-to-raw rate (~12%) is bounded by the
fraction of FS bins where `raw_NE × dt_shaped` already exceeds the
runner-up floor. This is structural — refining the scalar coefficient
cannot break it.

### Step 2 — S9-C joint floor attack pre-audit

Joint attacks layer the A.2 noise_floor refinement with min_ne_from_dt
scaling.

- C.1 = A.2 (noise_floor → error_psd × 0.005) + min_ne_from_dt × 0.1
- C.2 = A.2 + min_ne_from_dt → 0

#### Joint release fates (800-case)

| bucket | any-floor bins | **C.1 release** | C.1 still_min_ne | C.1 mean dB | **C.2 release** | C.2 still_min_ne | C.2 mean dB |
|---|---:|---:|---:|---:|---:|---:|---:|
| FS_static | 44,608,860 | **13.61%** | 46.89% | 14.71 | **13.61%** | 0.00% | 14.71 |
| FS_movement | 32,167,056 | **14.18%** | 46.13% | 14.70 | **14.18%** | 0.00% | 14.70 |
| DT_static | 77,620,058 | 48.51% | 25.96% | 14.70 | 48.51% | 0.00% | 14.70 |
| DT_movement | 44,949,014 | 49.26% | 27.00% | 15.11 | 49.27% | 0.00% | 15.11 |
| NE | 26,849,353 | 87.72% | 11.34% | 16.77 | 87.72% | 0.00% | 16.77 |
| GLOBAL | 226,194,341 | 41.55% | 31.43% | 15.03 | 41.55% | 0.00% | 15.03 |

**Critical finding**: C.1 release ≡ C.2 release on every bucket
(13.61% vs 13.61% on FS_static; 14.18% vs 14.18% on FS_movement).
The C.2 elimination of min_ne_from_dt shifts 47% of those bins from
"still_min_ne" to "still_phys" (= `ne_physical_floor = error_psd × 0.05`),
not to "release_to_raw". **`min_ne_from_dt × 0.1` is already low
enough to be non-binding on FS**; the binding constraint is the 3rd
floor.

### Step 3 — S9-D sanity pre-audit

Audit setup attacks all three nearend_est floors simultaneously:
- noise_floor → error_psd × 0.005
- min_ne_from_dt → 0
- ne_physical_floor → 0

Stack reduces to `[raw_NE × dt_shaped, noise_floor_A2, 0, 0]`. Winner
can only be 0 (raw_NE wins) or 1 (tiny noise_floor wins, only when
raw_NE × dt_shaped < 0.005 × error_psd).

| bucket | any-floor bins | **D release** | D still_floor | D mean dB |
|---|---:|---:|---:|---:|
| FS_static | 44,608,860 | **14.05%** | 85.95% | 23.13 |
| FS_movement | 32,167,056 | **14.60%** | 85.40% | 23.06 |
| DT_static | 77,620,058 | 48.80% | 51.20% | 19.75 |
| DT_movement | 44,949,014 | 49.55% | 50.45% | 20.09 |
| NE | 26,849,353 | 87.80% | 12.20% | 17.96 |
| GLOBAL | 226,194,341 | 41.86% | 58.14% | 20.74 |

**Verdict: FS release 14.05% (S9-D) — barely higher than S9-C
(13.61%)**. Attacking all 3 floors produces only +0.44 pct point of
release vs attacking 2. The remaining 85.95% of FS bins stay
floor-bound at `noise_floor_A2 = 0.005 × error_psd` (a value already
10× lower than baseline 0.01).

**Conclusion**: In 86% of FS converged bins,
`raw_nearend_est × dt_shaped < 0.005 × error_psd`. The nearend_est
floor stack is **NOT the FS-leak carrier** — the binding is
intrinsic to the structure of `raw_nearend_est × dt_shaped` being
near-zero in well-cancelled FS bins, which is the **physically
correct** behavior (FS by definition has no NE).

The carrier remains elsewhere. From S8's Stage 1 audit, **Cap2
(`residual_echo_psd ≤ error_psd × mult`)** fires 17-18% on FS bins
— attacks the ENR numerator (residual_echo) rather than denominator
(nearend_est), which is consistent with deflating ENR → inflating
gain → echo leak. Cap2 was identified in S8 but never attacked in
S9 because the floor stack appeared higher leverage.

## S10 plan — flag-and-bench S9-A.2 to discriminate H1 vs H2

### Implementation

- Flag: `res_noise_floor_refined` (default OFF) in AecConfig
- Behavior on FS-confident bins (`coh² < 0.1`):
  - Replace `noise_floor_psd = mean(error_psd) × 0.01` with
    `noise_floor_psd[k] = error_psd[k] × 0.005` (per-bin, lower
    coefficient)
- DT/NE bins (`coh² ≥ 0.1`): unchanged (baseline behavior)

### Acceptance gate

- 800-case byte-equal in flag-OFF ≥ 99.99% vs v3.11.x (Q7 V3 invariant)
- Flag-ON 800-case A/B with AECMOS:
  - **H1 PASS** = FS Δecho ≥ +0.005 AND NE Δdeg ≥ −0.005 AND DT Δdeg ≥ −0.005
    AND cohort tail Δecho ≥ −0.05 → S9-A.2 promotes to BALANCED (v3.11.3)
  - **H1 NEUTRAL** = FS Δecho in [−0.005, +0.005] → S9 closes null;
    pivot S11 = Cap2 pre-audit
  - **H1 FAIL** = FS Δecho < −0.005 OR any other bucket regression →
    S9 closes fail; pivot S11 = Cap2 pre-audit

### S11+ if H1 NEUTRAL/FAIL — Cap2 (residual_echo) pivot

Pre-audit Cap2 (Stage 1 binding 17-18% in FS per S8). Candidates:
- E.1: Cap2 disabled in FS-confident bins
- E.2: Cap2 mult raised (looser) in FS bins
- E.3: Cap2 gated by `1 - effective_dt`

### Critical invariants

- Cohort tail (`qNvSMyUSXUyrDGp`) Δecho ≥ −0.05 (P52 invariant)
- FS bucket Δecho ≥ −0.02 (anti-P50 trap)
- DT/NE bucket Δdeg ≥ −0.005 (anti-P58 trap)
- xrtntuju 5-clip DT regression listen
- Phase 4 unified-floor refactor: deferred to v3.13+ pending S9-A.2/Cap2 outcomes

## Sources

- [v3_12_s8_verdict.md](v3_12_s8_verdict.md) — noise_floor_psd is dominant carrier
- [v3_12_s7_verdict.md](v3_12_s7_verdict.md) — dt_per_bin unified NEUTRAL (carrier downstream)
- [v3_12_s6b_verdict.md](v3_12_s6b_verdict.md) — epc_dt_cap fire 0/2M
- `tools/research/s9_noise_floor_audit.py` — 800-case pre-audit harness
- `results/v3_12_s9_audit/audit.json` — S9-A data
- `results/v3_12_s9c_audit/audit.json` — S9-C data
- `results/v3_12_s9d_audit/audit.json` — S9-D data
- Q7 V3 verdict in `~/.claude/plans/se-aec-aec-main-hazy-lynx.md` §7
