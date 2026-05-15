# v3.18 plan — Large-arc cycle (2026-05-15)

**Branch**: continues on `feature/v3.17` HEAD `87730e4` (v3.17 closeout
unmerged; v3.18 work lands on same branch — at merge time the branch
ships as v3.18.0 or splits into v3.17.0 + v3.18.0 per user pick).
**Scope**: large architectural arcs deferred from v3.13/14/15/16/17.
Per user 2026-05-15: "不是因為架構大就不做" — schedule them in, not
out.
**Predecessors**:
[`docs/v3_17_closeout.md`](v3_17_closeout.md) (small-arc cycle outcome),
[`docs/v3_17_b1_movement_rate_delay_est_closure.md`](v3_17_b1_movement_rate_delay_est_closure.md),
[`docs/v3_17_b2_reverb_aware_closure.md`](v3_17_b2_reverb_aware_closure.md),
[`docs/v3_16_c6_delay_est_audit_verdict.md`](v3_16_c6_delay_est_audit_verdict.md).

---

## Strategic framing

v3.13 → v3.17 cycles exhausted the **1-3 sprint** mechanism space.
Every audible debt that remains needs a **5-12 sprint** architectural
arc:

| Audible debt | Why small arcs failed | Required arc |
|---|---|---|
| v3.13 E2 DT_static Δdeg −0.050 / DT_movement Δdeg −0.025 | RES-side gate relax is trade-off-bound (FS↔DT wall, confirmed v3.15 §1.2 + v3.17 B.3) | **Linear-filter convergence** rewrite |
| 0I0XMl3M FS_movement ERLE −49 dB | DelayEst polling+alpha can't track fast movement (v3.17 B.1) | **DelayEst rewrite** (multi-hypothesis + confidence-gated reset) |
| pcb1N FS reverb / fl-undercoverage | RES single-feature trigger N=1 + override unsolved (v3.16 C9 + v3.17 B.2) | **C9.v2 multi-feature classifier** OR **long-delay shadow** OR **fl > 832** |
| qNvSMyU cohort tail catastrophe | DSP wall (multi-peak ambiguity within fl coverage) | None — DSP-only ceiling |

This is a deliberate **scale-up cycle** trading sprint count for
actual mechanism progress. Estimated total LOE: **26-40 sprints**.

---

## Phase D — DelayEst rewrite (FOUNDATIONAL, 5-8 sprints)

**Why first**: every downstream arc benefits from better delay tracking.
v3.16 C6 + v3.17 B.1 confirmed the current `period_seconds + EMA alpha`
mechanism is insufficient for fast movement / multi-peak cases.

### D.0 Design lock (1 sprint)

Read existing v3.16 C6 trace data (10 cases, all 5 cohorts) + v3.17 B.1
closure root-cause analysis. Design the new DelayEst architecture:

| Component | Mechanism | Inspired by |
|---|---|---|
| **Confidence-gated EMA reset** | Reset cross-spec EMA when PAR drops below threshold AND new candidate emerges. Avoids tracking noise during low-confidence frames | WebRTC AEC3 delay estimator state machine |
| **Multi-hypothesis tracking** | Keep top-K=3 candidate lags through EPC window; commit to one only when separation is decisive | Multi-target tracking literature (Roumeliotis-Bekey 2002) |
| **Sub-sample fractional-delay** (optional, gated on D.0 cost) | Parabolic interpolation around argmax for sub-sample precision | Research-track |
| **PAR-aware Path B threshold** | Current Path B uses fixed 32-sample threshold for "meaningful shift" — should scale with PAR confidence | Phase 3c (P3c) high-PAR fast-path precedent |

Output: `docs/v3_18_d_delay_est_design.md` with mechanism specification,
data flow, byte-equal preservation strategy, hard bar.

**Hard bar (D)**:
- 0I0XMl3M ERLE p5_bad ≥ **-20 dB** (recover 29+ dB from -49 dB baseline)
- qNvSMyU Δecho ≥ **-0.05** (cohort tail catastrophe defence preserved)
- 60-case AECMOS no regression (FS/DT/NE all within bar)
- Default-OFF byte-equal sanity 5/5 cases

### D.1 — Implementation skeleton (1-2 sprints)
Wire new DelayEst class behind flag `delay_est_v2_enabled` (default
OFF). Reuse existing `DelayEstimator` for default path. New class
inherits but overrides `_estimate()` + `accumulate()`.

### D.2 — Multi-hypothesis tracking (1-2 sprints)
Track top-3 lag candidates per `_update_cross_spectrum()` call.
Maintain confidence trajectory per candidate. Commit when:
- PAR(top1) > 2 × PAR(top2) for 3 consecutive estimates AND
- Confidence trajectory monotone-rising for chosen candidate

### D.3 — Confidence-gated EMA reset (1 sprint)
Reset cross-spec when:
- PAR drops below low threshold for N consecutive estimates AND
- A new candidate lag emerges in different region (>200 samples from
  current commit)
Avoid resetting during EPC window if filter is mid-recovery.

### D.4 — Tuning + audit (1 sprint)
Run on v3.16 C6 Tier A 10 cases + 0I0XMl3M extended trace. Verify
mechanism behaves per design intent. Adjust thresholds (K, PAR
multipliers, reset criteria).

### D.5 — 60-case bench + listen verify (1 sprint)
60-case A/B vs v3.16.0 baseline. Per §0.6 metric channel: linear-filter
arc → nores listen PRIMARY + AECMOS regression guard.

**Kill criterion**: if 0I0XMl3M doesn't reach -20 dB OR cohort tail
regresses > -0.05 → close D, retry with different mechanism in v3.19+.

---

## Phase E — Linear-filter convergence rewrite (8-12 sprints, gated on D PASS)

**Why after D**: better delay tracking is a prerequisite for clean
filter convergence evaluation. v3.15 Arc M/G/F/T family all failed
because they tried to fix convergence WITHOUT addressing the delay
tracking signal upstream.

### E.0 Audit + design (2 sprints)
Re-mine v3.15 §1.1 DT-NE audit data + v3.16 C6 trace data with D-rewrite
running. Hypothesis: DT-NE compression in `coarse_learning` state
(v3.15 §1.1 H1) is partly driven by delay-tracking jitter (now
mitigated by D). With cleaner delay, `coarse_learning` may resolve
faster → reduces DT-NE compression naturally.

If hypothesis CONFIRMED: design surgical Q-tuning + filter_state
refinement (smaller intervention than v3.15 Arc M/F).
If REFUTED: design new mechanism (state machine restructure, per-band
mu scaling, or canonical Wiener gain integration).

### E.1-E.6 — Mechanism implementation (6-10 sprints)
TBD based on E.0 audit. Sprint plan defined in
`docs/v3_18_e_filter_convergence_design.md` (to be written after E.0).

**Hard bar (E)**:
- DT_static Δdeg ≥ **+0.020** (40% of v3.13 E2 debt recovery)
- DT_movement Δdeg ≥ **+0.010**
- FS Δecho ≥ -0.020 (no FS wall re-trigger)
- cohort tail Δecho ≥ -0.05
- nores listen verifies linear-filter primary improvement

**Kill criterion**: if DT bucket recovery < +0.005 after 4 tuning
sweeps → close E, RES-side mechanism family permanently confirmed
dead, escalate to phase-aware NL research-track.

---

## Phase F — C9.v2 multi-feature reverb-aware (5-8 sprints, gated on D)

**Why after D**: pcb1N's primary issue per v3.16 C6 audit is
`delay > 0.8 × fl_samples` — a delay-coverage problem. D-rewrite may
not fix coverage (filter taps still capped at 832) but may reduce
delay-tracking noise that masks the underlying issue. Need D outcome
to know what F needs to handle.

### F.0 — Cohort expansion (1-2 sprints)
Run v3.16 C9 audit script on full 60-case subset (vs original Tier A
10-case). Identify all cases with `delay > 0.8 × fl_samples` AND
`bucket == FS_static` (avoid DT/movement FP risk). If cohort ≥ 5 →
proceed. If cohort < 5 → close F (N=1 doesn't justify the arc).

### F.1 — Multi-feature classifier design (1-2 sprints)
Feature set:
- `delay > 0.8 × fl_samples` (binary, per-frame)
- Per-band coherence² mean in voice band (continuous)
- DelayEst `top1/top2 par_ratio` (multi-peak ambiguity)
- Envelope cross-correlation magnitude (alternative to sample-level
  Pearson which failed in v3.16 C9)

Classifier: simple thresholded AND (no learned weights to avoid
overfit on N≤10 cohort).

### F.2 — RES override design + wiring (1-2 sprints)
When trigger fires, modify RES behavior:
- Option A: lower `spectral_g_min` by 6-10 dB (deeper notch on
  fl-uncovered residual)
- Option B: scale `enr_t_ne` UP by 1.5 to open ENR gate (let RES
  attenuate more aggressively on bins where residual is high)
- Pick via F.2 mini A/B

### F.3-F.4 — Bench + listen (2-3 sprints)
pcb1N listen (audible target) + 60-case AECMOS regression guard +
800-case FP regression guard.

**Hard bar (F)**:
- pcb1N audible improvement (subjective listen)
- 60-case bucket means Δ ≥ -0.005 all metrics
- Cohort tail no regression

**Kill criterion**: if pcb1N listen shows no improvement OR FP rate
on full corpus > 5% → close F, defer pcb1N to Phase G architecture
arc.

---

## Phase G — Architecture decision (USER GATE, then 6-10 sprints)

**Why user gate**: G is a strategic architecture choice that affects
all future cycles. Requires user input on §8 invariant relaxation
vs new architecture cost.

### G option 1 — fl > 832 (memory budget revisit)

Relax §8 invariant `fl ≤ 832` for production C. Trade-offs:
- C memory budget exceeded; needs hardware team approval
- Production code changes (`c_impl/` rebuilds)
- Cleanest fix for pcb1N + qNvSMyU sub-sample multi-peak
- LOE: 6-8 sprints for Python prototyping + C port + bench

### G option 2 — Parallel long-delay shadow filter

Add a new long-FIR filter (e.g. fl=2048) running parallel to PBFDKF
+ ShadowFilter. Outputs are switched based on confidence. Trade-offs:
- New architecture component (third filter in pipeline)
- Compute doubles for long-delay path
- May solve pcb1N + 0I0XMl3M (if D-rewrite insufficient)
- LOE: 8-10 sprints architecture + integration

### G option 3 — Defer / decline

Accept current pcb1N audible debt. fl=832 stays. No architectural
change. v3.18 ships with Phase D + E + F + H only.

### G decision sprint

User reviews D-F outcomes + memory budget data + perceptual gain
estimate. Picks G1, G2, or G3.

---

## Phase H — Substrate flag promotion (PARALLEL-SAFE, 4-6 sprints)

**Why parallel**: independent of D/E/F mechanism work (different code
surfaces). Can run during long D/E/F bench cycles.

Promote 14 BALANCED-only substrate flags to MILD / SOFT / AGGRESSIVE /
MAXIMUM presets:

| Sprint | Action |
|---|---|
| H.1 | Audit per-flag impact on each non-BALANCED preset (run preset × flag combinations on 60-case) |
| H.2 | Identify safe-promotion flags (no regression on host preset) |
| H.3 | Promote in batches: 800-case A/B per preset per batch |
| H.4 | Listen verify on cohort tail + DT_static + FS_static |
| H.5 | Update preset definitions in `aec.py:1042-1228` |

**Hard bar (H)**: each promoted flag passes 800-case AECMOS no
regression on the host preset (FS Δecho ≥ -0.02, DT Δdeg ≥ -0.005).

---

## Phase sequencing

```
            ┌─ Phase H (parallel-safe, 4-6 sprints) ─────────────────┐
            │                                                          │
Phase D ────┼─→ Phase E (gated on D, 8-12 sprints) ──┐                │
(5-8 spr)   │                                          │                │
            └─→ Phase F (gated on D, 5-8 sprints) ────┤                │
                                                       │                │
                                              Phase G user gate (after │
                                              D-F outcomes available)  │
                                                                        │
                                              v3.18 closeout + §0.7 ◄──┘
                                              merge auth
```

D blocks E + F. E + F can parallelize on sub-branches (file-disjoint:
E touches PBFDKF + AecState, F touches RES). H runs in background
throughout. G is user-gated, decision point depends on D-F results.

**Total**: 26-40 sprints (D 5-8 + E 8-12 + F 5-8 + G 0-10 + H 4-6).
Realistic execution timeline: 2-4 months of dedicated work.

---

## Verification rules (apply each sprint)

Same §0 discipline as v3.15/16/17:
1. §0.1 — trace before edit / baseline JSON drift / byte-equal flag-OFF /
   standard config / listen / root cause / one-arc per sprint
2. §0.2 — state-mutation disjointness for parallel arcs (E vs F)
3. §0.3 — oracle validation before design lock (especially D.0 → D.1)
4. §0.4 — pre-stated acceptance thresholds (each arc's hard bar above)
5. §0.6 — metric channel rules:
   - D = linear-filter (nores listen PRIMARY)
   - E = linear-filter (nores listen PRIMARY)
   - F = RES output (AECMOS PRIMARY)
   - G = depends on choice
   - H = AECMOS (preset comparison)
6. §0.7 — `feature/v3.17` branch (or `feature/v3.18` if split) not
   merged until full cycle verdict pack reviewed by user

---

## Out-of-scope (explicit)

- **NN any module** — per `feedback_no_nn_mention` /
  `feedback_dsp_only_until_completion`. v3.18 stays DSP-only until
  Phase G decision and v3.18 closeout. Phase-aware NL research-track
  needs separate user authorization (orthogonal to v3.18 cycle).
- **WebRTC AEC3 port** — load-bearing 4-cap RES already same family;
  rewrite ROI ≈ 0
- **MMAE / IMM filter replacement** — PBFDKF + ShadowFilter +
  PathChangeRegimeHandler retirement attempts (5+ closures) confirmed
  load-bearing

---

## Critical files (v3.18 modification targets)

**Phase D**:
- `python/aec.py:1232-1500` — DelayEstimator class (new class
  `DelayEstimatorV2` extends or replaces; flag-gated)
- `python/aec.py:6500-6555` — DelayEst integration site

**Phase E**:
- `python/aec.py:1026-1219` — PBFDKF
- `python/aec.py:3577 ff.` — ShadowFilter
- `python/aec.py:5800-6100` — AecState / filter_state machine
- `python/aec.py:7180-7199` — state classification

**Phase F**:
- `python/aec.py:2789-2985` — ResFilter._stage_gain_compute /
  _stage_gain_postprocess
- New flag: `c9v2_reverb_aware_enabled`

**Phase G** (if G1 or G2):
- `c_impl/` — production C port adjustments
- `python/aec.py:1574-1620` — PBFDKF buffer sizes

**Phase H**:
- `python/aec.py:1042-1228` — preset defaults

---

## v3.17 closeout disposition

`feature/v3.17` HEAD `87730e4` carries the v3.17 closeout. Per user
direction "請你安排下去": do NOT merge v3.17 alone first. v3.17 work +
v3.18 phases all ship together as either:
- `v3.18.0` (subsumes v3.17.x improvements + new arcs)
- OR `v3.17.0` + `v3.18.0` (split commits at merge time)

Decision deferred to v3.18 closeout merge gate.

---

## Risk register

| Risk | Source | Mitigation |
|---|---|---|
| D-rewrite breaks cohort tail catastrophe defence | qNvSMyU sample-level multi-peak (DSP wall per C6 audit) | Default-OFF + byte-equal sanity at every D step; D kill criterion explicit on qNvSMyU Δecho ≥ -0.05 |
| E hits §1.2 FS-vs-DT wall again | v3.15 §1.2 + v3.17 B.3 closure family | E.0 audit re-mines hypothesis with D-rewrite running; if H1 still dominant, E switches to state-machine restructure (not gate-relax) |
| F cohort N < 5 after expansion | v3.16 C9 audit Tier A had N=1 valid (pcb1N) | F.0 explicit gate: cohort < 5 closes F immediately |
| G architecture decision delayed | User-gated, not engineering-gated | G can run last; D + E + F + H ship independently if user defers G |
| v3.18 cycle drags past 4 months | 26-40 sprint LOE estimate | Sprint cadence ~1/day with bench wait; track per-phase progress in `docs/v3_18_<phase>_progress.md`; user can authorise merge at any phase boundary |
| Parallel E + F sub-branches conflict | §0.2 state-mutation disjointness | File-disjoint analysis at E.0 / F.1 design lock; if conflict, sequentialize |

---

## Next sprint

**D.0 design lock** starts immediately on user authorization of this
plan. Output: `docs/v3_18_d_delay_est_design.md`.

If user wants different ordering (e.g. H first to validate substrate
parity before architectural work), redirect.
