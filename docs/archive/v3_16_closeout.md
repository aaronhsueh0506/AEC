# v3.16 closeout summary (2026-05-15)

**Status**: DEFINITIVE CLOSEOUT — all audit-able candidates closed.
**Branch**: `feature/v3.16` (10 commits past v3.15.0; updated 2026-05-15 post C4).
**Push status**: NOT PUSHED, NOT MERGED — pending §0.7 user authorisation.
**Plan reference**: `~/.claude/plans/se-aec-aec-main-hazy-lynx.md` §10.

---

## 1. Headline

v3.16 ships **1 production-grade algorithm change** (C1 `epc_dt_cap`
dead-code removal) + **9 disposed candidate arcs** (5 Phase 0 housekeeping
+ 4 mechanism closures). Of 6 remaining candidates, 5 have likely-closure
indicators (FS-vs-DT wall, dead path, low priority, no AECMOS Δ
predicted). Cycle effort delivered substantial **closure evidence**
informing v3.17 scope but minimal user-visible AECMOS Δ.

**Pattern parallels v3.15 closeout**: most v3.16 candidates closed
when run through the §0.4 negative-result acceptance protocol. The
RES architecture is approaching its DSP-only ceiling — same conclusion
as v3.13 (E4/E5/Volterra closures), v3.14 (Arc H closure), and v3.15
(Arc M/G/T S2 wiring closures).

---

## 2. v3.16 commits on `feature/v3.16` (10 commits past `9f70add` v3.15 merge)

| # | Commit | Sprint | Description |
|---|---|---|---|
| 1 | `d90efdc` | C1 | Phase 0 epc_dt_cap dead-code removal (5/5 byte-equal PASS, 0/2,032,022 frame fire-rate) |
| 2 | `b0dc576` | HK-2 reframe + C1b block + C1c new | Phase 0 HK-2 listen analysis (5 causes), C1b ne_g_floor removal blocked by 4/5 byte-equal FAIL, C1c new candidate proposed |
| 3 | `37a46b2` | C1c | Phase 0 audit script ne_g_floor measurement bug fix (pre-max snapshot) |
| 4 | `718d36f` | Phase 0 closeout | plan doc — 800-case audit + C9 + C1b reclassify |
| 5 | `ac7320e` | C6 audit | Phase 1 DelayEst audit CLOSED H2 — lead-lag 1.5%; 0/10 H1 STRONG cases |
| 6 | `d181e17` | v3.16-A | Phase 1.5 force_render OR-in CLOSED CANNOT SHIP — logically subsumed by `not filter_converged` (qNvSMyU byte-identical, lever fires 0/2686) |
| 7 | `d04ba60` | C3 | Phase 2 4-cap stacking audit CLOSED — NOT stacking-driven (99.87% stack=0 on voice band) |
| 8 | `6ba0378` | C9 audit | Phase 4 reverb-aware audit CLOSED — TRIGGER UNDESIGNED (Pearson r fails to separate pcb1N from CTRL; 5-8 sprint LOE needed) |
| 9 | `e42a8a6` | v3.16 closeout summary | Initial closeout summary doc |
| 10 | `35800af` | C4 audit | Phase 2 noise_floor / CNG audit CLOSED — H4 REFUTED (nfl_dom = 0% all buckets; noise_floor is NE-protection floor not compression source) |

**Production change**: only commit `d90efdc` (C1 dead-code removal).
The other 7 commits are audit substrate, closure docs, and plan doc
revisions. **No new algorithm features ship in v3.16.**

---

## 3. Candidate disposition tally

### 3.1 Disposed (10 / 15)

| ID | Phase | Outcome | Commit |
|---|---|---|---|
| HK-1 | 0 | ✓ DONE (B3 fix shipped in v3.15 `ea8d320`) | inherited |
| HK-2 | 0 | ✓ REFRAMED → C6 + C9 inputs | `b0dc576` |
| C1 | 0 | ✓ DONE (epc_dt_cap removal; 5/5 byte-equal PASS) | `d90efdc` |
| C1b | 0 | ✓ RECLASSIFIED RETAIN (ne_g_floor universal floor; folded into C2) | `b0dc576` |
| C1c | 0 | ✓ DONE (audit measurement bug fix) | `37a46b2` |
| C6 | 1 | ✓ CLOSED H2 (audit, no delay arc) | `ac7320e` |
| v3.16-A | 1.5 | ✓ CLOSED CANNOT SHIP (subsume) | `d181e17` |
| C3 | 2 | ✓ CLOSED (NOT stacking-driven) | `d04ba60` |
| C4 | 2 | ✓ CLOSED (H4 REFUTED; noise_floor is NE-protection floor) | `35800af` |
| C9 | 4 | ✓ CLOSED → v3.17 C9.v2 (trigger undesigned) | `6ba0378` |

### 3.2 Remaining (5 / 15) — all non-audit-able

| ID | Phase | Status | Likely outcome |
|---|---|---|---|
| C5 | 1.6 | DEFERRED | Architectural; ZERO AECMOS Δ; defer to v3.17 if C2 implementation can isolate ENR write surface |
| C2 | 2 | candidate | Likely CLOSED — same family as v3.15 §1.2 / Arc D (FS-vs-DT wall confirmed by §1.2 closure, FS Δecho -0.077 to -0.395 dB) |
| v3.16-B | 3 | candidate | Gated on C2; if C2 closes, v3.16-B is blocked |
| C7 | 4 | candidate | Was CLOSED in v3.15 §1.5b; retry needs new detector primitive (depends on C6 outcome — none) |
| C8 | 4 | candidate | LOW priority; partial-decay alt for Arc G (Arc G already CLOSED v3.15) |

---

## 4. Closure pattern analysis

### 4.1 The "logical subsume" pattern (v3.16-A)

v3.16-A's force_render OR-in was mathematically correct but logically
subsumed: `force_render = ... or not filter_converged or (cohort_tail_T)`,
and in cohort tail catastrophe windows the filter is NEVER converged
(0/2686 frames on qNvSMyU), so `not filter_converged = True` already
fires regardless. This is the **same closure family** as v3.15 Arc T S2
(H1 dead code in BALANCED ENR path; H2 wiring no-op).

**Lesson**: cohort tail catastrophe windows are characterised by filter
non-convergence (per definition of catastrophe). Any RES-side trigger
gated on cohort_tail_T is OR-subsumed by `not filter_converged`. To
deliver new behaviour, consumer must operate on a SUBSET of
`not filter_converged` — tighten the gate, not loosen the OR.

### 4.2 The "metric mismatch" pattern (C3 + C9)

C3 audit showed 4-cap chain has 99.87 % stack=0 on voice band — caps
are protection / smoothing, NOT attenuation. The §1.7 audit's loose
fire-rate definition (any-bin change > 1e-7) reported 30-90 % fire on
DT/FS, but voice-band cumulative effect is essentially zero. Mechanism
arc would target the wrong surface.

C9 audit attempted plain Pearson r as a per-frame coupling indicator;
CTRL (Y91uE2t) had WORSE r than pcb1N. The HK-2-reported r=0.071 was
likely a different methodology not reproducible from current trace
surface. Mechanism arc would need 5-8 sprints of detector R&D (vs
original 2-3 estimate).

**Lesson**: predicted ROI on candidate arcs depended on metric
assumptions. When metrics actually measured the relevant phenomenon,
the assumptions broke. Audit-first protocol caught these in 1 sprint
each, saving 8-15 sprints of mechanism wiring that would have closed
anyway.

### 4.3 The "DSP ceiling" pattern (cumulative)

Across v3.13 / v3.14 / v3.15 / v3.16, AEC has now closed:
- 3 amplitude-domain NL attempts (E4 mask / E5 saturation / v3.14 Volterra)
- 5 cohort-tail-targeted RES policies (P52 A.0 / P55 1.2 / P58 / Arc T S2 / v3.16-A)
- 2 dual-filter orthogonality attempts (P52 1.2 Wiener / Arc H Huber)
- 3 movement-aware Q schedules (Arc M V1 / V2 / V3)
- 2 destructive W reset arcs (Arc G / proposed C8 retry)
- 1 RES floor refactor (Phase 3 + C1b retain)

**Pattern**: traditional DSP-only AEC at fl=832 / preset=balanced is
near its perceptual ceiling. Most remaining "audible debt" requires
either (a) deeper architectural rework (delay coverage, NL preprocessor,
phase-aware NL) outside v3.16+ single-cycle scope, or (b) NN augmentation
explicitly out of scope per `feedback_no_nn_mention`.

---

## 5. Recommended next steps (post C4 closure update 2026-05-15)

### 5.1 Option A: Closeout v3.16 + transition to v3.17 planning  [RECOMMENDED]

All 5 audit-able candidates closed (C6 / v3.16-A / C3 / C4 / C9). The
remaining 5 (C5 / C2 / v3.16-B / C7 / C8) are NOT audit-first
candidates — they would require multi-sprint mechanism investment with
high closure risk per the cumulative pattern. v3.17 planning with
focused candidates delivers more value than continuing v3.16.

### 5.2 Option B: Open C2 mechanism despite §1.2 wall warning

C2 was the primary Phase 2 target. Even though §1.2 / Arc D family
showed the FS-vs-DT wall, C2's per-state × per-band variant might
find a narrow window. 3-sprint investment (without C5 if ENR write
surface can be isolated). RISK: closure pattern strongly suggests
this also hits the wall (3-sprint cost for likely-closure outcome).

### 5.3 Recommended: Option A (closeout + v3.17 planning)

Per §0.4 negative-result acceptance + the cumulative closure pattern,
v3.16 has delivered maximum value from audit-first cycles. C4 audit
**confirmed REFUTED** (this update; nfl_dom 0% across all buckets).
C2 risks hitting the same wall as §1.2. The right call is **closeout
v3.16 + transition to v3.17 planning** with focused candidates:

#### v3.17 candidate seed list (preliminary)

1. **C9.v2** — Reverb-aware RES override (re-scoped 5-8 sprints).
   Multi-feature trigger + cohort expansion + RES override design +
   800-case FP guard. pcb1N as primary case + 5+ similar reverb-heavy
   cohort.
2. **Movement-rate DelayEst** (from C6 closure §6 backlog) —
   shorten `period_seconds` 2.0 → 0.25 s when motion detector
   fires. Target: `0I0XMl3M` ERLE p5_bad ≥ −20 dB (vs current −49 dB).
3. **C2.v2 (if user authorises)** — narrow per-state × per-band ENR
   tuning ON A SUBSET of `coarse_learning + suspicious_dt` states,
   leaving `refined_usable` byte-equal. May escape FS-vs-DT wall by
   fixing only the over-suppression in non-converged states.
4. **C4 audit** — noise_floor / CNG interaction (deferred from v3.16
   if not done).
5. **NL detector substrate revival** — v3.13 E5 saturation detector
   was preserved; combine with phase-aware NL research-track if user
   authorises (per `feedback_dsp_only_until_completion`).

### 5.5 Push / merge gate (per §0.7)

`feature/v3.16` has 8 commits past `9f70add` (v3.15 merge). NOT pushed,
NOT merged to main. **Awaits user authorisation**:

- Production change scope: 1 algorithm commit (`d90efdc` C1 dead-code
  removal). 5/5 byte-equal PASS on (FS_static, FS_movement, DT_static,
  NE, pcb1Nh0Z) for both `_ours.wav` and `_ours_nores.wav`. 800-case
  re-audit confirms 0/2,032,022 frame fire-rate.
- Other 7 commits are research substrate (audit scripts, closure docs,
  plan doc revisions) — non-load-bearing for production behaviour.

If user authorises:
- `git checkout main && git merge --no-ff feature/v3.16 -m "Merge v3.16 closeout (C1 dead-code removal; 4 candidate arcs CLOSED CANNOT SHIP)"`
- `git tag v3.16.0`
- Push `main` + `feature/v3.16` (substrate) + `v3.13.0` / `v3.14.x` / `v3.15.0` / `v3.16.0` tags

If user declines: `feature/v3.16` remains as research substrate.
Production HEAD stays at v3.15.0.

---

## 6. v3.17 sprint estimate (preliminary)

| Candidate | Sprints | Notes |
|---|---:|---|
| C9.v2 multi-feature trigger | 5-8 | High-priority audible debt closure |
| Movement-rate DelayEst | 2-3 | C6 backlog item; isolated case |
| C2.v2 narrow per-state ENR | 3-4 | If user authorises; risk same wall |
| C4 audit + (mechanism if PASS) | 1 + 2-3 | Deferred from v3.16 |
| NL detector revival | 5-10 | Research-track; needs user auth |
| **TOTAL (if all open)** | **18-28** | Long cycle |
| **MIN (only C9.v2)** | **5-8** | Targeted closure of pcb1N audible debt |

---

## 7. Closeout signed-off

v3.16 has delivered:
- 1 production change (`d90efdc` C1 epc_dt_cap removal)
- 9 candidate dispositions (5 Phase 0 housekeeping + 4 mechanism
  closures)
- 4 substantial closure verdict docs informing v3.17 scope
- 4 audit substrate scripts retained for v3.17 / future research

**Recommendation**: closeout v3.16, transition to v3.17 planning with
C9.v2 as the highest-ROI candidate. **Awaits §0.7 user merge
authorisation.**
