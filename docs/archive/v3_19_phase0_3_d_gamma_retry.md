# v3.19 Phase 0.3 — D-γ subband NE detector retry decision (2026-05-16)

**Status**: DECIDED — Option (B): defer D-γ retry to **v3.20+
backlog**, **after Phase 1 RES per-branch findings land**.
**Rationale**: D-γ closed because mask profile swap hit the same
DT-FS Pareto wall as Phase 1 is now investigating. Phase 1's
per-branch finding will tell us **which RES branches** can absorb a
sharper FS suppression curve — exactly the missing input D-γ needed.
Doing D-γ first risks repeating the same closure.

## 1. Origin

v3.18 D-γ closeout
([docs/v3_18_d_gamma_closeout.md](v3_18_d_gamma_closeout.md))
states:

> Useful for: (a) v3.19+ retry once Phase A (shadow NLMS) + Phase B
> (filter misadjustment) recover linear-filter quality (cohort tail
> Δecho gain may absorb the DT cost); (b) v3.20+ Reverb-port arc.

Pre-condition (a) is **partially met** in v3.18 outcome:
- Phase A (shadow NLMS) — CLOSED (qNvSMyU Δecho -0.001, 4 worst-case
  breaches). Linear-filter quality NOT recovered.
- Phase B (filter misadjustment) — CLOSED (0/61 fires). Substrate
  on disk but trigger silent.
- Phase C.A/C.B/C.C — PASSED (audit-only). Linear-filter
  observability ports landed; **no quality recovery yet**.

So the D-γ pre-condition (linear-filter quality recovered) is **not
satisfied** by v3.18.

## 2. Re-evaluation question

Should D-γ retry happen in v3.19 (substrate already in place) or wait
for v3.20+?

### Option (A) — v3.19 Phase 3.5: D-γ retry after Phase 3 B

- Phase A still closed (substrate on disk, OFF). D-γ would re-bench
  with B retry's improved fq_usable + filter_misadjustment ON if Phase
  3 ships.
- LOE: 5-8 sprints (re-bench all 4 D-γ configs from closeout table +
  any new combo with C.E winning branches if Phase 1 ships).
- Risk: same physics wall as v3.18 D-γ closeout. AEC3's
  `normal_params_` (LF anchor (0.3, 0.4, 0.3), HF anchor (0.07, 0.1,
  0.3)) is more aggressive than what our pipeline survives without DT
  damage. Phase 1/2/3 success doesn't change mask aggressiveness; it
  changes which RES branches the mask hits.

### Option (B) — defer to v3.20+ backlog (RECOMMENDED)

- D-γ depends on **per-band NE detection at the RES branch level**.
  Phase 1 per-RES-branch ablation directly surfaces which branches
  benefit from per-band signals (LF NE preserve vs HF FS suppress).
- After Phase 1.4 single-branch sweep ranks branches by (Δdeg − |Δecho|),
  D-γ retry can target **only the helpful branches** with subband NE
  gating — same Pareto-walking discipline as Phase 1.
- Avoids repeating the same whole-mechanism swap that closed v3.18 D-γ.

## 3. Decision

**Option (B): D-γ retry stays in v3.20+ backlog. Trigger condition
sharpened.**

Update v3.20+ backlog entry:

> **D-γ subband NE detector retry** | 5-8 sprints | trigger:
> Phase 1 per-RES-branch findings published (ranked branches with
> measured Δdeg DT_static / Δecho FS_static per RES stage); D-γ
> mask profile then re-applied **only at branches where per-band
> separation has empirical headroom** | source: v3.18 D-γ closeout
> + v3.19 Phase 0.3

This change moves D-γ from **mechanism-port retry** to
**branch-targeted retry**, mirroring Phase 1's discipline.

## 4. What's NOT changing

- D-γ substrate stays default-OFF on disk (per v3.18 closeout): all 8
  env-var overrides + SubbandNearendDetector + DominantNearendDetector
  + mask profile builders preserved.
- No code change in v3.19 Phase 0-4.
- Phase 1.1 RES branch inventory **does not** need to enumerate
  per-band consumers — Phase 1 is per-branch on the
  `filter_converged → fq_usable` axis only. Per-band is a v3.20+
  axis.

## 5. Auto-trigger fold (parallel to Phase 0.1)

If Phase 1 closes early AND Volterra Option (B) does NOT fold in
(per Phase 0.1 §5), **D-γ retry is the next-priority candidate** for
the freed budget. Fold-in scope would be:

- D.1: re-bench dom_only_asym + combined_asym configs against new
  Phase 1 RES branch combo (3-4 sprints, 60-case).
- Kill criterion: same as v3.18 (NE Δdeg ≥ +0.010 PRIMARY +
  DT Δdeg ≥ +0.005 across both buckets + FS Δecho ≥ −0.010).

This is option **D-γ.fast** (4 sprints) vs option (A) full retry
(5-8 sprints). User §0.7 mid-cycle authorisation required.

## 6. Cross-references

- [docs/v3_18_d_gamma_closeout.md](v3_18_d_gamma_closeout.md) — D-γ
  closure + substrate inventory + retry pre-conditions
- [docs/v3_18_phase0_decision.md](v3_18_phase0_decision.md) — D-γ
  hard bar definition
- [docs/v3_18_c_e_closeout.md](v3_18_c_e_closeout.md) — Pareto wall
  evidence shared with D-γ closure
- [docs/v3_13_e5_closure_verdict.md](v3_13_e5_closure_verdict.md) —
  same DT-FS Pareto wall (4 sub-variants closed)
- [docs/v3_19_phase0_1_volterra_eval.md](v3_19_phase0_1_volterra_eval.md) —
  parallel auto-trigger logic
- `~/.claude/plans/se-aec-aec-main-hazy-lynx.md` — v3.19 cycle plan
