# v3.17 Phase A — Stability baseline verdict (2026-05-15)

**Branch**: `feature/v3.17` HEAD `0859efa` (off `4cc3b3b` v3.16.0).
**Phase A scope**: dead-code inventory + targeted removal + byte-equal
stability gate vs v3.16.0 production HEAD.

## Phase A sprints — outcome

### A.1.1 — `over_sub` chain dead-code gating ✓ SHIPPED

**Mechanism**: `_stage_dt_indicator` at `aec.py:7384-7389` computed
`dt_reduction → effective_over_sub → self.res.over_sub` every frame in
ENR mode, but `over_sub` is consumed only by
`gain_type=='wiener'` / `'spectral_sub'` paths (lines 3244-3251). All
5 BALANCED presets use `gain_type='enr'` ⇒ writes were dead code.

Same closure family as v3.16-A H1 (Arc T S2 `over_sub × 1.3` boost).

**Implementation**: `_over_sub_live = self.config.res_gain_type != 'enr'`
gate; entire chain wrapped behind this gate. Skipped for ENR; preserved
for wiener / spectral_sub paths.

**Verification**: 5/5 byte-equal sanity PASS on FS_static / FS_movement /
DT_static / NE / pcb1Nh0Z (both `_ours.wav` + `_ours_nores.wav`).
Commit `0859efa`.

### A.1.2-A.1.6 — DEFERRED (no audible/runtime impact)

Quick scan disposition:

| ID | Verdict | Reason |
|---|---|---|
| A.1.2 `_shadow_mu_holdoff` | DEFER | inline comment "kept harmless for future wiring" — explicit design intent to keep |
| A.1.3 P3h reset action | DEFER | needs verify-first sprint; not blocking Phase B |
| A.1.4 line 3680 dead branch | DEFER | already documented as dead-retained; cosmetic |
| A.1.5 line 3310 cap-in-FS | DEFER | needs 800-case fire-rate audit (not free) |
| A.1.6 line 1133 diverged_reset triple-AND | DEFER | gate intentional per v3.11 closure |

These do not block Phase B. Revisit during Phase C cleanup if Phase B
work touches adjacent regions.

### A.2 — Substrate flag triage DEFERRED

42 default-False bool fields across `AecConfig` (counted via grep). Full
triage requires its own multi-sprint audit; v3.16 just closed +6 candidate
arcs and the substrate flags are documented in their respective closure
docs. Re-visit if Phase C tunable-strength interface needs cleanup.

### A.3 — Logic bug review DEFERRED

B-list status from v3.15 §2 punch list:
- B1, B2, B9, B11 — FIXED
- B4, B5 — open (HIGH/MED, v3.15 §10.S1/S2 deferred)
- B6, B10 — LOW (style/process)
- B7, B8 — known-substrate (closed Arc T candidates)

Phase B.1 (Movement-rate DelayEst) does not depend on B4/B5 fixes. Defer
B4/B5 to v3.17 closeout housekeeping or v3.18.

### A.4 — Stability gate ✓ PASS

60-case subset bench rendered:
- baseline: `/tmp/v3_16_a_off/` (v3.16.0 HEAD `4cc3b3b`, 120 wav files)
- treatment: `/tmp/v3_17_a_subset/` (v3.17 A.1.1 HEAD `0859efa`, 120 wav files)

Per-file md5 diff: **0 differences across all 120 files**. Byte-equal
confirmed (expected: A.1.1 is dead-code-elision in ENR mode).

## Phase A — overall outcome

✓ 1 dead-code removal landed (A.1.1) with byte-equal proof
✓ 60-case stability gate PASS — v3.17 HEAD = v3.16.0 byte-equal
✓ Substrate ready for Phase B mechanism arcs

Phase A.2-A.3 deferred items don't block Phase B. v3.17 HEAD `0859efa`
declared as "stable substrate" baseline for Phase B/C.

## Next: Phase B

Per v3.17 plan §Phase B sequencing, three independent mechanism arcs:
- B.1 Movement-rate DelayEst (smallest LOE, 2-3 sprints, 0I0XMl3M target)
- B.2 C9.v2 reverb-aware RES override (5-8 sprints, pcb1N target)
- B.3 C2.v2 narrow per-state ENR (3-4 sprints, DT-NE debt; risk wall)

Order: B.1 → B.3 → B.2 (smallest LOE first, biggest LOE last; B.3 risk
wall handled mid-cycle so failure is known before B.2 sunk cost peaks).
