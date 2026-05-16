# v3.17 Closeout (2026-05-15)

**Branch**: `feature/v3.17` HEAD `62552b7` (off v3.16.0 merge `4cc3b3b`).
**Cycle outcome**: 1 dead-code removal + 1 substrate (Movement-rate
DelayEst, default OFF) + 5 audit/closure docs + 0 audible regressions
+ Phase C preset gradient confirmed monotone.
**Algorithm changes shipped**: 0 (Phase A.1.1 byte-equal in production
ENR mode; B.1 substrate default OFF; B.3/B.2 no code).
**Awaiting**: user §0.7 merge authorisation.

---

## 1. Phase outcomes

| Phase | Sprint | Verdict | Substrate retained |
|---|---|---|---|
| A.1.1 | 1 | ✓ SHIPPED `0859efa` (over_sub gated, dead-code in ENR) | n/a (dead-code removal) |
| A.4 | 1 | ✓ PASS (byte-equal vs v3.16.0 60-case) | — |
| B.1.S1-S3 | 3 | ✗ CLOSED CANNOT SHIP `182cf89` — target case 0I0XMl3M Δecho −0.144, 2/11 movement cases improved, 3/11 regressed | `mov_rate_delay_est_enabled` flag + `delay_est_period_s_fast` + `delay_est_alpha_fast` (3 default-OFF config fields) |
| B.3 | 1 | ✗ CLOSED CANNOT SHIP `30d85c1` — narrow scope (suspicious_dt only) barely fires; DT bucket Δdeg +0.001 (vs hard bar +0.015) | (existing v3.15 §1.2 substrate) |
| B.2 | 1 | ✗ CLOSED CANNOT SHIP `cb677b4` — single-feature trigger N=1 cohort + RES override unsolved; full mechanism R&D needs 5-8 sprints | (existing v3.16 C9 audit script) |
| C.1 | 1 | ✓ COMPLETE `ab4cf2c` — 12 monotone strength knobs documented; 14 BALANCED-only substrate flags identified | — |
| C.2 | 1 | ✓ PASS `62552b7` — all 10 bucket-metric pairs monotone on 60-case AECMOS; gradient interface correct | — |
| C.3 | — | NOT TRIGGERED (C.2 found 0 non-monotone metrics) | — |

## 2. v3.17 production deliverables

What `feature/v3.17` brings if merged to main:

1. **Phase A.1.1** — `over_sub` chain gated behind `gain_type != 'enr'`.
   Dead-code elision in production ENR mode. Byte-equal to v3.16.0
   on 60-case AECMOS (120/120 wav files md5-identical).

2. **B.1 substrate** — 3 default-OFF config flags wired:
   - `mov_rate_delay_est_enabled`
   - `delay_est_period_s_fast` (0.25 s)
   - `delay_est_alpha_fast` (0.2)
   - Env overrides: `AEC_MOV_RATE_DELAY_EST`, `AEC_DELAY_EST_PERIOD_S_FAST`
   - Default OFF: byte-equal to v3.16.0
   - ON path: closed CANNOT SHIP (target regressed); kept as substrate
     for v3.18+ retry with different motion detector / EMA reset design

3. **5 verdict / audit docs**:
   - `docs/v3_17_a_stability_baseline.md`
   - `docs/v3_17_b1_movement_rate_delay_est_closure.md`
   - `docs/v3_17_b3_narrow_per_state_enr_closure.md`
   - `docs/v3_17_b2_reverb_aware_closure.md`
   - `docs/v3_17_c_strength_knob_inventory.md`
   - `docs/v3_17_c_preset_gradient_audit.md`

**v3.17 production algorithm changes = 0**. All Phase B mechanism
arcs closed CANNOT SHIP. Phase A.1.1 is dead-code elision (byte-equal).
Phase C is documentation-only.

## 3. Honest value statement

v3.17 cycle did NOT solve any audible debt:
- v3.13 E2 Path 3 DT debt (DT_static -0.050 / DT_movement -0.025) —
  CONFIRMED unrecoverable through RES-side gate relax (B.3 narrow
  approach barely fires; structural confirmation of v3.15 §1.1 H1
  hypothesis)
- 0I0XMl3M FS_movement -49 dB ERLE — B.1 polling+alpha mechanism
  REGRESSED the target (Δecho -0.144 dB on this very case)
- pcb1N FS_static reverb / fl-undercoverage — multi-sprint mechanism
  R&D needed; v3.17 budget insufficient after B.1+B.3 spend

**What v3.17 DID do**:
- ✓ Confirmed preset gradient (5 operating points) is monotone and
  user-tunable (Phase C)
- ✓ Documented 14-flag substrate asymmetry (BALANCED uses v3.10+
  improvements; other presets don't) — informs v3.18+ scope
- ✓ Removed 1 piece of dead code (over_sub chain in ENR mode)
- ✓ Provided substrate for v3.18+ DelayEst retry (B.1 mechanism wired
  but disabled, ready for swap of motion-detector / EMA-reset design)

**Pattern confirmed across v3.13 / v3.14 / v3.15 / v3.16 / v3.17**:
DSP-only AEC at fl=832/balanced is at perceptual ceiling. Each cycle
shrinks the candidate-arc space by ruling out failed mechanism
families per §0.4. v3.17 closes 3 more (movement-rate DelayEst,
narrow per-state ENR, single-feature reverb trigger).

## 4. v3.18+ candidate backlog

From v3.17 closure docs:

| Item | Origin | LOE | Risk |
|---|---|---|---|
| DelayEst confidence-gated EMA reset (multi-hypothesis tracking) | B.1 closure §4 | 5-8 sprints | HIGH (DelayEst rewrite) |
| Linear-filter convergence improvement (Q tuning + delay tracking) | B.3 + v3.15 §1.2 closure §6 | 8-12 sprints | HIGH (architecture-level) |
| C9.v2 multi-feature reverb-aware classifier + RES override | B.2 closure §3 | 5-8 sprints | MEDIUM (cohort N=1 today) |
| Memory-budget revisit (`fl > 832`) for long-delay coverage | B.2 closure §4 | uncertain | breaks §8 invariant |
| Long-delay parallel shadow filter | B.2 closure §6 | 6-10 sprints | new architecture arc |
| Substrate flag promotion (14 flags ON across all 5 presets) | C.1 finding | 4 × 800-case + listen | bounded |
| Phase-aware NL research-track (§1.9 family) | v3.13 E4/E5 closures | needs user auth | RESEARCH |

## 5. §0.7 merge gate

Per v3.17 plan §0.7: `feature/v3.17` does NOT merge to main without
explicit user authorisation.

**Recommendation**: merge as v3.17.0.

Rationale:
- 0 algorithm regressions vs v3.16.0 (Phase A.4 byte-equal PASS)
- 1 maintenance improvement (dead code removed)
- 1 substrate (default OFF) for future mechanism retries
- 6 closure / audit docs documenting 3 mechanism family closures
  + Phase C gradient confirmation
- v3.17 is a clean "clarification" cycle — no new debt, no new
  bugs, just better-documented limits and one substrate

If user declines merge:
- `feature/v3.17` remains as research substrate
- main stays at v3.16.0
- B.1 substrate stays branch-only
- Closure docs do not propagate to main

## 6. Branch state

```
feature/v3.17 HEAD 62552b7
  62552b7 v3.17 Phase C.2 — Preset gradient audit PASS
  ab4cf2c v3.17 Phase C.1 — Strength knob inventory complete
  cb677b4 v3.17 Phase B.2 CLOSED CANNOT SHIP (per §0.4)
  30d85c1 v3.17 Phase B.3 CLOSED CANNOT SHIP (per §0.4)
  182cf89 v3.17 Phase B.1 CLOSED CANNOT SHIP (per §0.4)
  9ece25d v3.17 Phase B.1.S3 — Add EMA alpha override
  ab8816b v3.17 Phase B.1.S2 — Movement-rate DelayEst wired (default OFF, byte-equal PASS)
  fd3f45b v3.17 Phase A — Stability baseline CLOSED
  0859efa v3.17 Phase A.1.1: gate over_sub chain
  4cc3b3b (main, v3.16.0) Merge v3.16 closeout
```

9 commits. Not pushed. Not merged.
