# v3.16 C3 — 4-cap chain stacking audit CLOSED (2026-05-15)

**Status**: CLOSED — NOT STACKING-DRIVEN per §0.4.
**Branch**: `feature/v3.16` (commit pending).
**Sprint**: v3.16 Phase 2 audit-first (per `~/.claude/plans/se-aec-aec-main-hazy-lynx.md` §10).
**Substrate**: `tools/research/v3_16_c3_4cap_stacking_audit.py` retained.

---

## 1. Headline

**Verdict**: Stage-1 4-cap chain (post v3.16 C1: 3-cap chain `quiet_mask
→ 3bin_smooth → hf_cap`) **does NOT stack-attenuate the voice band**.
60-case subset audit: 99.87% of frames have stack=0; 0.13% have stack=1;
**0.00% have stack≥2**.

The DT-NE compression hypothesis H3 (4-cap chain over-fires on DT-NE)
**FALSIFIED** — caps almost never affect voice band cumulatively. C3
mechanism arc (reorder caps to reduce stacking) cannot deliver Phase 2
gain; **C3 CLOSED, redirect Phase 2 effort to C2 (ENR per-state ×
per-band)**.

---

## 2. Empirical analysis

### 2.1 Stack distribution (60-case, 162,791 frames)

| Stack count | Global % |
|---:|---:|
| 0 (no cap fired on voice band) | **99.87 %** |
| 1 (single cap fired) | 0.13 % |
| 2 (two caps stacked) | 0.00 % |
| 3 (all three stacked) | 0.00 % |

### 2.2 Per-bucket stack distribution

| Bucket | n cases | n frames | stk=0 | stk=1 | stk=2 | stk=3 |
|---|---:|---:|---:|---:|---:|---:|
| FS_static | 11 | 25,386 | 100.0 % | 0.0 % | 0.0 % | 0.0 % |
| FS_movement | 11 | 25,371 | 99.8 % | 0.2 % | 0.0 % | 0.0 % |
| DT_static | 14 | 51,572 | 99.7 % | 0.3 % | 0.0 % | 0.0 % |
| DT_movement | 13 | 48,352 | 99.9 % | 0.1 % | 0.0 % | 0.0 % |
| NE | 11 | 12,110 | 100.0 % | 0.0 % | 0.0 % | 0.0 % |

DT bucket has the highest stack-1 rate, but still ≤ 0.3 % of frames.
Stack-2+ is empirically **zero** across all buckets.

### 2.3 Per-cap voice-band fire rate

| Bucket | qm fire | sm fire | hf fire |
|---|---:|---:|---:|
| FS_static | 0.0 % | 0.0 % | 0.0 % |
| FS_movement | 0.0 % | 0.0 % | 0.2 % |
| DT_static | 0.0 % | 0.0 % | 0.3 % |
| DT_movement | 0.0 % | 0.0 % | 0.1 % |
| NE | 0.0 % | 0.0 % | 0.0 % |

`quiet_mask` and `3bin_smooth` fire 0 % on voice band. `hf_cap` fires
< 0.3 % (voice band ends below the HF cap bin, so direct effect rare).

### 2.4 Reconciliation with v3.15 §1.7 audit (which reported cap fire-rates 30-90 %)

The §1.7 audit measured the LOOSE definition of fire-rate ("any bin
post-stage gain differs from pre-stage by > 1e-7"). My audit measures
the STRICT definition ("voice-band MEAN gain dropped by > 0.04 dB").

Both are correct under their definitions. The reconciliation:
- Caps DO modify the gain vector frame-by-frame (loose: 30-90 %).
- But the modifications are mostly:
  1. `quiet_mask` RAISES gain in masked (quiet) bins to 1.0 — this is
     a PASS-THROUGH protection, not attenuation.
  2. `3bin_smooth` averages neighbours — net effect on mean is roughly
     zero; minor smoothing.
  3. `hf_cap` only affects bins ABOVE the cap bin (typically 4 kHz).
     Voice band (100-3500 Hz) is below.
- Net voice-band cumulative attenuation: stack=0 frames gain RISES by
  +0.2 to +0.9 dB across buckets (caps NET HELP voice preservation).

### 2.5 Cumulative voice-band attenuation when caps fire

| Bucket | cum_db @ stk=0 | cum_db @ stk=1 |
|---|---:|---:|
| FS_static | +0.223 | −1.350 |
| FS_movement | +0.858 | −3.608 |
| DT_static | +0.507 | −0.534 |
| DT_movement | +0.434 | −3.578 |
| NE | +0.221 | +3.181 |

Stack=1 frames (the rare ones where any cap moves voice band) DO
attenuate, but they are < 0.3 % of all frames. Not a significant
contributor to user-perceived DT-NE compression.

---

## 3. v3.16 plan implications

### 3.1 C3 mechanism arc — CLOSE

The original C3 hypothesis (4-cap chain reorder reduces DT-NE
over-suppression) is empirically refuted. **No C3 mechanism arc;
no need to invest C5 architecture for C3.**

### 3.2 Phase 0 §3.4 quiet_mask cohort_tail substrate shift (-0.052)

Phase 0 closeout flagged quiet_mask cohort_tail fire-rate dropped from
0.679 (v3.13) → 0.627 (v3.16) — a 5.2 % drop. **C3 audit confirms this
is a fire-rate metric drift, NOT a perceptual change**: voice-band
cumulative effect of quiet_mask is essentially zero (0.0 % voice-band
fire rate across all buckets). The cohort tail fire-rate change
reflects substrate evolution (Arc P + R + S-orth.A + Arc T) shifting
which bins are "quiet enough to mask", but does not affect voice-band
gain in any meaningful way.

**No C3 follow-up sprint required.**

### 3.3 Phase 2 path forward

| Candidate | Status | Recommendation |
|---|---|---|
| **C2** ENR per-state × per-band (subsumes Arc D + ne_g_floor refactor) | per plan §1.1 audit H1+H2 cited as primary DT-NE compression source | **REMAINS Phase 2 priority** if Phase 2 opens |
| **C3** Stage-1 4-cap reorder | empirically refuted | ✓ CLOSED (this doc) |
| **C4** noise_floor / CNG interaction | per plan §1.1 audit H4; not yet audited | LOW priority — audit-first before commit |

### 3.4 C5 architectural investment re-justification

C5 architectural foundation was sized as 3 sprints to BLOCK Phase 2
(C2/C3/C4). With C3 closed, the remaining Phase 2 candidates (C2 + C4)
may not require the full modular interface — could be implemented
directly on `_stage_gain_compute` (per the parallel-branch hygiene
risk noted in plan §0.2 Arc D × Arc R conflict).

**Decision flag**: C5's value is now driven by C2 alone. If C2
implementation can isolate ENR write surface (e.g. C2 introduces its
own per-state-per-band table at the dispatch site without colliding
with other paths), C5 may be deferable to v3.17. Defer C5 decision
until C2 design lock.

### 3.5 Phase 1 / 2 closure tally (Phase 0 + 1 done)

| ID | Phase | Status |
|---|---|---|
| HK-1 | 0 | ✓ DONE (`ea8d320`) |
| HK-2 | 0 | ✓ REFRAMED → C6 + C9 |
| C1 | 0 | ✓ DONE (`d90efdc`) |
| C1b | 0 | ✓ RECLASSIFIED RETAIN |
| C1c | 0 | ✓ DONE (`37a46b2`) |
| C6 | 1 | ✓ CLOSED H2 (`ac7320e`) |
| v3.16-A | 1.5 | ✓ CLOSED CANNOT SHIP (`d181e17`) |
| C5 | 1.6 | DEFERRED (re-evaluate post-C2 design lock) |
| C2 | 2 | candidate (remains primary Phase 2 target) |
| C3 | 2 | ✓ CLOSED (this doc) |
| C4 | 2 | candidate (LOW priority; audit-first) |
| v3.16-B | 3 | gated on C2 |
| C7/C8 | 4 | gated on Phase 3 |
| C9 | 4 | gated on C6 (done); ready to open |

**8 candidates disposed** (HK-1/HK-2/C1/C1b/C1c/C6/v3.16-A/C3); **7 remain**.

---

## 4. Substrate (committed)

- Audit script:
  [`tools/research/v3_16_c3_4cap_stacking_audit.py`](../tools/research/v3_16_c3_4cap_stacking_audit.py)
- Aggregate JSON (gitignored): `/tmp/v3_16_c3_audit/aggregate.json`
- Per-case JSONs (gitignored): `/tmp/v3_16_c3_audit/per_case/<stem>.json`

Zero changes to `python/aec.py` — audit consumed `capture_stages=True`
existing surface. Production behaviour unchanged.

---

## 5. Verdict signed-off

**CLOSED** per §0.4 negative-result acceptance protocol. Stage-1 4-cap
chain stacking is not the source of DT-NE compression. C2 (ENR
per-state × per-band) remains the primary Phase 2 candidate per §1.1
audit H1+H2.

**Next decision point**: open C2 design? Or open C9 (Phase 4
reverb-aware, audible target, gated on C6 done)? Defer to user.
