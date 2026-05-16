# v3.17 Phase B.2 — C9.v2 reverb-aware RES override CLOSED CANNOT SHIP (2026-05-15)

**Branch**: `feature/v3.17` HEAD `30d85c1`.
**Sprint**: Phase B.2 (per [`docs/v3_17_plan.md`](v3_17_plan.md) §B.2).
**Verdict**: CLOSE per §0.4 — trigger design problem unchanged from
v3.16 C9 closure. Single-feature trigger candidate fires on too many
DT/movement false-positive cases; multi-feature classifier requires
multi-sprint detector R&D that exceeds v3.17 budget after B.1+B.3
sprint spend.
**Substrate retained**: `tools/research/v3_16_c9_reverb_aware_audit.py`.

---

## 1. Adjudication summary

The v3.16 C9 closure (`docs/v3_16_c9_reverb_aware_audit_closure.md`)
was explicit: "Estimated LOE: **5-8 sprints** ... Beyond single-sprint
audit scope". v3.17 plan §B.2 acknowledged this and budgeted 5-8
sprints. After B.1 (CLOSED CANNOT SHIP) + B.3 (CLOSED CANNOT SHIP)
sprint spend, remaining v3.17 budget is insufficient to do justice to
B.2's full mechanism R&D arc.

## 2. Single-feature trigger reanalysis (existing Tier A 10-case data)

From `docs/v3_16_c9_reverb_aware_audit_closure.md` §2.4:

> "The closest single-feature discriminator is `delay > 0.8 × fl_samples`,
> but this fires false-positive on cohort tail / movement / DT cases."

Re-checking the per-case rates from the C9 audit table (§2.1):

| Case | Bucket | delay > 0.8·fl % | C9 target? |
|---|---|---:|---|
| **pcb1N FS** (target) | FS_static | 82.3 % | YES — primary target |
| **qNvSMyU FS** | FS_static | 84.8 % | OVERLAP — cohort tail Arc T scope |
| XRTnTUjU DT | DT_static | 79.7 % | FALSE POSITIVE — would damage DT |
| 0I0XMl3M FS_mvmt | FS_movement | 72.4 % | FALSE POSITIVE — movement, not reverb |
| Hp5g1asac DT_mvmt | DT_movement | 0.0 % | not triggered |
| WH0jN3PY DT_mvmt | DT_movement | 86.2 % | FALSE POSITIVE — DT |
| OX2l6zV7 FS_mvmt | FS_movement | 81.1 % | FALSE POSITIVE — movement |
| Y91uE2t FS CTRL | FS_static | 0.0 % | not triggered (good) |
| (others) | various | 0.0 % | not triggered |

**Cohort breakdown**:
- C9 valid targets: 1 (pcb1N) — qNvSMyU is already in Arc T's scope
- False-positive risk: 5 (XRTnTUjU, 0I0XMl3M, WH0jN3PY, OX2l6zV7,
  plus likely more on full 800-corpus)
- N=1 valid cohort cannot drive a generalisable mechanism.

## 3. Expansion attempt cost-benefit

To find more pcb1N-class cases (FS_static with delay > 0.8·fl), would
need to run the v3.16 C9 audit script on the full 60-case (or 800-case)
subset:
- 60-case audit: ~30 min wall (script wraps full AEC pipeline)
- Add classifier design sprint (multi-feature: delay coverage +
  envelope cross-corr + per-band coh²): 1-2 sprints
- Override design (which RES knob, by how much, per-bin or scalar):
  1-2 sprints
- 800-case FP regression guard sprint: 1 sprint
- Total: 4-6 sprints minimum

**v3.17 budget after B.1 (1) + B.3 (1) + this audit (1)**: realistically
2-3 sprints remain for B.2 + Phase C. Not enough.

## 4. RES override design — also unsolved

Even with a perfect trigger, what does "FS-aggressive RES" mean for
fl-undercoverage cases? The existing knobs:

| Knob | Effect | Concern for pcb1N |
|---|---|---|
| `over_sub` boost | Wiener / spectral_sub paths | DEAD in ENR mode (per Phase A.1.1 closure) |
| Lower `spectral_g_min` (-20 → -30 dB) | Allow deeper notch | pcb1N already at gp5 = -6.27 dB; would push to -10 dB |
| Scale `enr_t_ne` UP | Open ENR gate (less suppression) | OPPOSITE of what we want |
| Scale `enr_t_ne` DOWN | Tighten ENR gate (more suppression) | pcb1N already heavily suppressed; leakage is residual artefact, not gain failure |
| Use shadow filter output | If shadow tracks better | shadow has its own delay, may also fail fl coverage |

The pcb1N audible artefact is residual echo passing through the
linear filter (because filter can't model long delay). RES is
downstream — it can suppress more, but it cannot reconstruct missing
echo path information. **The canonical fix is to extend `fl` beyond
832 (memory-bound, violates §8 invariant) OR add a parallel longer-FIR
shadow filter for high-delay coverage.** Both are major arcs, not B.2
scope.

## 5. Per §0.4 verdict

> "If lever moves bucket-mean < 0.002 dB AND fires < 5% of frames
> after 3 A/B sweeps, ship as substrate for dependent arc"

B.2 not even attempting a wired benchmark — the trigger design + N=1
cohort + override unsolved problems compound. v3.16 C9 closure was
correct: 5-8 sprints needed. v3.17 cannot deliver this responsibly
given B.1 + B.3 spend.

**Action**: no new code. Document closure. Existing C9 audit script
substrate (commit from v3.16 era) preserved for v3.18+ retry.

## 6. v3.18+ canonical path (per analysis above)

pcb1N audible debt requires either:
- **Memory-budget revisit**: extend `fl > 832` (reopens §8 invariant)
- **Long-delay parallel shadow filter**: new architecture arc
- **Reverb-aware RES override**: 5-8 sprint detector R&D (this arc, deferred)

All exceed v3.17 single-cycle scope. v3.13 E2 Path 3 max_delay 1024 ms
shipped (filter operating window doubled), but `fl=832` taps still
limits effective echo path length. pcb1N remains the canonical
"hardware ceiling" artefact case.

## 7. Next: Phase C

Phase C (tunable strength interface) is the v3.17 remaining authorised
work. C.1 strength knob inventory + C.2 preset gradient validation
have NO mechanism wall risk (audit + measurement only) and produce
deliverable substrate. Proceeding directly to Phase C.
