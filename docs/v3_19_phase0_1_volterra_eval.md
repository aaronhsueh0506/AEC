# v3.19 Phase 0.1 — Volterra non-linear inverse filter evaluation (2026-05-16)

**Status**: DECIDED — Option (A) primary; Option (B) auto-triggered if
Phase 1 closes early.
**Verdict**: Volterra remains a **v3.20+ dedicated cycle**. v3.19 stays
Pareto-focused. If Phase 1 closes early (kill criterion at 1.4 or 1.5),
re-evaluate folding the **design lock + literature fetch** (option B
scope) into the freed-up budget.

## 1. Context

v3.13 closeout 2026-05-14
([docs/v3_13_arc_closure.md](v3_13_arc_closure.md)) declared:

> v3.14 primary arc = Volterra non-linear inverse filter (HIGHEST
> priority, 6+ month dedicated arc). Detector substrate from E4.S2 +
> E5.S3 reusable.

v3.14 / v3.15 / v3.16 / v3.17 / v3.18 all skipped it. Each cycle had
its own primary arc that took precedence (arcs P/R/H/H'/D for v3.14;
arc T cohort detector for v3.15; RES refactor housekeeping for v3.16;
preset gradient audit for v3.17; AEC3 alignment for v3.18). The
substrate (E4.S2 detector + E5.S3 correlation gate) is still on
disk as default-OFF.

## 2. What Volterra solves (vs what v3.18 solved)

| Symptom | Source frame profile | v3.18 lever | Volterra lever |
|---|---|---|---|
| 爆掉 (loudspeaker NL) | FS bucket M3 > 9.0 | unreachable | time-domain NL inverse |
| 無線電 (codec NL) | FS bucket; phase distortion | unreachable | time-domain NL inverse |
| DT NE preserve | DT_static low ENR | C.E +1.2 Δdeg single-case | orthogonal |
| FS echo suppression | FS_static high mic-lpb r | C.E -0.49 single-case worst | orthogonal |

v3.18 C.E reached the **linear/RES Pareto wall**. Volterra reaches the
**amplitude-mask wall** (per v3.13 E4 closeout: "multiplicative
spectral mask cannot reach phase distortion + time-domain transient").
The two are **orthogonal axes**. Solving one does not advance the
other.

## 3. Re-evaluation question

> Given v3.18 C.E finding (Pareto walk has +1.2 Δdeg single-case in
> linear/RES space), is Volterra still highest priority? Or does the
> linear-side ceiling justify exhausting Pareto first?

### Evidence for "exhaust Pareto first"

1. **C.E finding is the largest AECMOS movement of v3.18 cycle**.
   Pareto-walking has empirical headroom — single-case +1.2 Δdeg is
   reproducible.
2. **Pareto Phase 1 LOE is 8-12 sprints**. A shipping algo within
   v3.19 budget is realistic. Volterra is 6+ months — **always**
   exceeds a single cycle.
3. **v3.19 goal**: ≥ 1 algorithm change ships (break v3.17 → v3.18
   streak). Pareto path supports this; Volterra path does not (design
   lock alone ships nothing).
4. **Substrate compounds**: Phase 1 RES per-branch work surfaces the
   exact branches Volterra's detector would need to pin. Doing Pareto
   first informs Volterra design.

### Evidence for "Volterra now"

1. v3.13 declared HIGHEST priority — 5 cycles of skipping is
   accumulated technical debt against canonical roadmap.
2. v3.18 cycle confirmed **AEC3 alignment hits diminishing returns**
   (8 substrate arcs + 0 ships). Pareto is one more attempt at the
   same Pareto wall family.
3. NL listening cohort (5/5 listen-validated) has remained
   user-perceptible across all cycles. Linear/RES improvements don't
   touch it.

### Decision weight

The "Pareto Phase 1 ships within budget" path has **direct AECMOS
evidence** (C.E +1.2 single-case). The "Volterra ships sometime" path
has **literature evidence** (Hammerstein-Wiener / Volterra-Wiener
adaptive papers) but **no AEC3 reference port** in the WebRTC
codebase to mirror. Doing Volterra requires fresh design from
literature; doing Pareto reuses 8 existing substrate arcs.

**Pareto first is the higher-confidence cycle outcome.** Volterra
without Pareto-first risks another 0-ship cycle (Volterra design lock
+ implementation in 6+ months = exceeds v3.19 wall-clock).

## 4. Decision

### Primary: Option (A) — v3.20 dedicated Volterra cycle

- v3.19 stays focused on Phase 1 (C.E Pareto-walking) + Phase 2
  (C.D-α retry) + Phase 3 (B FilterMisadjustment retry) per plan.
- Volterra **stays in v3.20+ backlog** as DEDICATED CYCLE (6+ months,
  not foldable into mixed cycle).
- v3.19 closeout (Phase 4) re-states Volterra as v3.20 candidate
  alongside other v3.20+ arcs.

### Auto-trigger: Option (B) — fold Volterra design lock into freed budget

If Phase 1 closes early (kill criterion at sprint 1.4 single-branch
sweep, or 1.5 combined config), the freed budget triggers Option (B):

- **Sprint V.1** (2 sprints): literature re-survey — Hammerstein /
  Wiener / Volterra-Wiener adaptive identification. Compare AEC3's
  approach (no Volterra port; AEC3 has no NL processor either) vs
  papers Yang 2014 (NLMS-based Volterra) / Stenger-Kellermann (Volterra
  series for AEC) / Birkett-Goubran 1995 (canonical AEC NL).
- **Sprint V.2** (2 sprints): design lock — Volterra kernel order /
  memory length / step size / state integration with AecState back-ref
  / detector reuse (E4.S2 + E5.S3 + C.A consistent_estimate +
  C.E fq_usable as gate).
- **Sprint V.3** (1 sprint): cohort sizing — re-listen 5/5 v3.13 NL
  cohort + new candidates from C.E sUQrHEPA tail; pre-bench oracle
  thresholds.

Total V.1-V.3 = 5 sprints, all doc-only. Output: Volterra design lock
ready for v3.20 implementation kickoff.

### Rejected: Option (C) — v3.19 pivot to Volterra

- v3.19 only has 24-37 sprints; Volterra needs 6+ months (~50+
  sprints) to land an algorithm. Pivoting kills both Pareto AND
  Volterra (Volterra wouldn't finish in v3.19 either).
- Option (C) is logically a no-op; it just delays Pareto without
  delivering Volterra.

## 5. Trigger conditions for auto-Option-B

Phase 0 plan body **does not pre-allocate** V.1-V.3 sprints. Auto-fold
fires only if **all** of the following:

1. Phase 1 sprint 1.4 single-branch sweep finds **no branch** with
   positive (Δdeg − |Δecho|) → Phase 1 closes here, ~6 sprints freed
2. OR Phase 1 sprint 1.5 combined config FAILs PRIMARY or any guard
   → Phase 1 closes here, ~3 sprints freed (1.6+1.7 cancelled)
3. AND no Phase 0.4 fold-in triggered (suppression_gain port absorbed
   the freed budget instead)
4. AND user authorises auto-fold via §0.7 mid-cycle check-in

If trigger conditions met: insert V.1-V.3 between Phase 3 and Phase E
(or Phase 4 if Phase E skipped).

## 6. Substrate inventory for Volterra (already on disk)

| Substrate | Location | Source cycle | Reuse role |
|---|---|---|---|
| E4.S2 voice-band autocorrelation pitch tracker | aec.py SubtractiveNLP class (default-OFF) | v3.13 | NL frame detector |
| E5.S3 mic-lpb correlation r>0.35 in 0.7-0.95 | worktree branches s4a/s4b (preserved) | v3.13 | NL frame detector v2 |
| C.A FilterAnalyzer.consistent_estimate | aec_filter_analyzer.py (default-OFF) | v3.18 | Linear-filter health gate |
| C.B FilteringQualityAnalyzer.fq_usable | aec_filter_quality.py (default-OFF) | v3.18 | NL processor enable gate |
| C.C AecState back-ref | aec.py L5177+ AecState extension | v3.18 | NL state integration |
| 5/5 listen-validated NL cohort | docs/v3_13_e4_*.md | v3.13 | Pre-bench oracle |

**No fresh substrate work required for Volterra design lock.** All
inputs are on disk.

## 7. v3.20+ backlog re-statement

Volterra entry in plan v3.20+ backlog stays:

> Volterra non-linear inverse filter (DEDICATED CYCLE) | 6+ months |
> post-v3.19 closeout; per Phase 0.1 if option (A) | v3.13 closeout

(no change to backlog table; this evaluation confirms the entry).

## 8. Cross-references

- [docs/v3_13_arc_closure.md](v3_13_arc_closure.md) — v3.14 primary
  arc declaration (HIGHEST priority origin)
- [docs/v3_13_e4_s6a_s6b_verdict.md](v3_13_e4_s6a_s6b_verdict.md) —
  amplitude-mask family CLOSED FAIL (NL physics wall)
- [docs/v3_13_e5_closure_verdict.md](v3_13_e5_closure_verdict.md) —
  saturation deepening CLOSED (FS-DT trade-off line)
- [docs/v3_18_c_e_closeout.md](v3_18_c_e_closeout.md) — Pareto
  evidence (+1.2 Δdeg single-case) motivating Pareto-first
- [docs/v3_18_cycle_closeout.md](v3_18_cycle_closeout.md) — v3.18 0
  algorithm-change outcome motivating v3.19 ship-priority
- `~/.claude/plans/se-aec-aec-main-hazy-lynx.md` — v3.19 cycle plan
