# v3.18 Phase C.1 — AecState centralisation + FilteringQualityAnalyzer (design lock, doc-only) — 2026-05-16

**Status**: C.1 DESIGN SPRINT — doc only, no code edits.
**Branch**: `feature/v3.18-aec3-fetch` (HEAD after Phase B closeout).
**Authorisation scope**: C.1 design ONLY. **C.2+ requires user re-auth**
per the 4-of-4 AEC3-port closure lessons learned.
**Pivot context**: Phase A + B both closed 2026-05-16 with shared root
cause — AEC3 mechanisms need AEC3 co-design context. C is the
co-design refactor.

## 0. One-paragraph framing

D-γ / F / A / B all failed because each AEC3 mechanism we imported
expected an AEC3-shaped surrounding system: per-band NE detector
(D-γ), reliable delay_change events (F), refined-filter quality
analyzer (A), `refined_usable`-majority state coverage (B). Phase C
**stops importing mechanisms** and instead **refactors our internal
state to AEC3's ADT shape**. Specifically: `AecState` Python class
(single source of truth for "is filter usable right now"), driven by
an internal `FilterAnalyzer` + `FilteringQualityAnalyzer` (multi-gate
usable flag). Once that's in place, the retained substrate from
D / F / A / B can be revisited as Phase C follow-ups because the
preconditions they expected will exist.

Phase C is **internal refactor**, not AEC3 port. The mechanism we
deploy is ours; the architectural shape is AEC3-aligned.

## 1. Scope inventory — current state surface

### 1.1 State currently scattered across AEC

| Field | Type | Today's owner | Consumers (grep count) |
|---|---|---|---|
| `_filter_converged` | bool | `FilterConvergenceAnalyzer` | 24 sites |
| `_filter_state` | str (enum-like) | `_prev_filter_state` cache | 32 sites |
| `effective_dt` | float | `DoubleTalkAnalyzer` + scattered overrides | 51 sites |
| `usable_linear_estimate` | property | `AEC` | 9 sites (already partial-built per v3.13) |
| `epc_active` | property | `EchoPathChangeDetector` | many sites |
| `_arc_t_cohort_tail_signal` | bool | `AEC` | several sites |
| `_filter_once_converged` | bool | `AEC` | several sites |
| `saturation_level` | float | various | scattered |

Total directly-affected consumer surface: ~120 sites. AEC3's `AecState`
encapsulates the equivalent into ~10 public methods.

### 1.2 AEC3 reference (`docs/aec3_extracts/src/aec3/aec_state.h`)

Public interface AEC3 exposes:

```cpp
bool UsableLinearEstimate() const;           // can RES use linear output?
bool UseLinearFilterOutput() const;          // should pipeline use linear?
bool ActiveRender() const;                   // far-end active?
bool UseStationarityProperties() const;      // is near-end stationary?
float FullBandErleLog2() const;              // filter quality metric
const std::array<float, …>& Erl() const;     // per-bin ERL
int MinDirectPathFilterDelay() const;        // delay estimate
bool SaturatedCapture() const;               // saturation flag
bool SaturatedEcho() const;                  // echo-channel saturation
bool TransparentModeActive() const;          // transparent mode
bool TransitionTriggered() const;            // initial→stable transition
void HandleEchoPathChange(const EPV&);       // event cascade
void Update(…);                              // per-frame update
```

Internal sub-modules AEC3 wraps:
- `FilterAnalyzer` (HP 600 Hz pre-filter + peak detection + consistent_estimate)
- `FilteringQualityAnalyzer` (startup_timer + reset_timer + convergence_seen + !transparent multi-gate)
- `ErleEstimator` / `ErlEstimator`
- `SaturationDetector`
- `InitialState` / `TransitionTriggered` lifecycle

### 1.3 What `refined_usable` means in our code vs AEC3

Our `_filter_state == 'refined_usable'`:
- Set in `FilterConvergenceAnalyzer` when filter is "converged enough"
- Coverage on 3-case trace: 0% / 0% / 38% (per Phase B closeout)

AEC3's `UsableLinearEstimate() == true`:
- Coverage typical 60-80% on similar cases (per AEC3 design docs)
- Multi-gate: startup_timer expired (0.4s) AND last reset > 0.2s ago
  AND convergence_seen ever AND not transparent

The gap: our `'refined_usable'` is a tight one-frame-instant check;
AEC3's is a "you can trust the linear filter NOW because [historical
multi-event sequence]" check. AEC3's broader coverage is the unblock
B needed.

## 2. Phase C ordering — 4 sub-phases

To keep risk bounded after 4-of-4 closures, C is split into 4
**file-disjoint** sub-phases, each with its own kill criterion:

### C.A — FilterAnalyzer port (doc + 5 sprints)
- Subsume v3.14 Arc P (per-band ERL EMA) into a single `FilterAnalyzer`
  class
- Add AEC3-style HP 600 Hz pre-filter + peak detection +
  `consistent_estimate` boolean
- File: new `python/aec_filter_analyzer.py`
- Wired as audit-only; no consumers change behaviour
- Kill: if `consistent_estimate` flag is identical to `_filter_converged`
  on 60-case trace, port adds no information → close

### C.B — FilteringQualityAnalyzer port (5 sprints, gated on C.A PASS)
- New `FilteringQualityAnalyzer` class with AEC3 multi-gate
  `usable_linear_estimate`:
  - `startup_timer` (frames since first far_active ≥ 40 ≈ 0.4s)
  - `reset_timer` (frames since last EPC reset ≥ 20 ≈ 0.2s)
  - `convergence_seen` (filter has been converged at any point in stream)
  - `not transparent` (transparent mode not active)
- File: new `python/aec_filter_quality.py`
- Wired as audit-only initially; one diagnostic field exposes the new
  flag without changing legacy behaviour
- Kill: if multi-gate coverage matches `_filter_converged` coverage
  on 60-case trace, port adds no information → close

### C.C — AecState ADT (6 sprints, gated on C.A + C.B PASS)
- New `AecState` class wrapping the new analyzers + the existing
  scattered state
- Public methods mirror AEC3 (`UsableLinearEstimate`, `ActiveRender`,
  `SaturatedCapture`, `TransparentModeActive`, etc.)
- Initially **read-only mirror** — AecState.update() reads from the
  existing AEC fields each frame
- Consumer migration in C.E (below); C.C is just the wrapper
- Kill: if AecState.UsableLinearEstimate() differs from
  `usable_linear_estimate` property on byte-equal trace → mismatch
  bug, fix before C.E

### C.D — leakage_diverged + per-state RES (5 sprints, gated on C.C PASS)
- This is the original Phase C scope — `leakage_diverged` bifurcation
  in PBFDKF Q injection + RES per-state ENR refactor
- Now feasible because C.B's `FilteringQualityAnalyzer.usable_linear`
  gates the RES decision cleanly
- File-disjoint: touches `aec.py` Q-injection block + `res_*` modules

### C.E — Consumer migration (4-6 sprints, in batches)
- Migrate ~120 scattered consumers to `AecState` API
- Per-batch (5-10 consumers) byte-equal verification
- File: aec.py (incremental) + new `tools/research/c_e_consumer_inventory.py`
- Kill: per batch, if byte-equal breaks → revert that batch, document
  why specific consumer can't migrate, narrow C.E scope

### C.F — 60-case A/B + 800-case ship gate
- 60-case AECMOS with each sub-phase's flag ON one at a time
  (audit-only flags), then combined
- 800-case + nores listen on cohort tail + xrtntuju 5-clip
- Per-§6 hard bar adjudication

Total LOE: 25-30 sprints (was 9-12 in plan — corrected upward given
B closeout's evidence that the AEC3 refactor surface is larger than
plan revision §4 assumed).

## 3. Design decisions

### 3.1 Read-only mirror first, migrate later

C.A + C.B + C.C are **observability layers** initially — they compute
the new metrics + expose them via `_diag['aec_state_*']` traces without
any consumer changing behaviour. This means **all C.A-C.C sprints
preserve byte-equal flag-OFF and flag-ON** (the new flag is "compute
the new metric" which has zero downstream effect by design).

Only C.D + C.E are behaviour-changing — and each is its own re-auth
gate.

### 3.2 No AEC3 source-line copies; reimplement from spec

Per `feedback_no_shortcut_use_canonical.md`: implement the canonical
mechanism ourselves rather than line-port AEC3 code. AEC3 is the
reference for **what** the analyzer outputs; **how** we compute it in
Python is up to us, using existing PBFDKF/PBFDAF accessors.

Practical consequence: we don't depend on AEC3 source still being
present in `docs/aec3_extracts/`. The design lock here references the
public API only.

### 3.3 Cheap pre-bench gates baked in (Phase B lesson)

Phase B's wall-clock saving from the upfront B.4 fire-rate gate is now
standard discipline:
- C.A pre-bench gate: does `consistent_estimate` distribution differ
  from `_filter_converged` distribution? 5-case trace, no bench needed
- C.B pre-bench gate: does `usable_linear_estimate (multi-gate)`
  coverage exceed `_filter_converged` coverage by ≥ 20 percentage
  points on the same 5-case trace?
- C.C pre-bench gate: AecState public-method outputs match the legacy
  scattered fields on byte-equal trace
- C.D + C.E require full 60-case bench (behavioural change)

If C.A pre-gate fails → close C entire (analyzer adds no signal).
If C.B pre-gate fails → close C.D entire (the broader coverage was
the whole point).

### 3.4 What this design does NOT address

- **Delay detector replacement** (matched-filter NLMS + histogram) —
  v3.19+ scope. Phase C provides the FQA + AecState surface that a
  future delay detector can hook into.
- **Per-band NE detector port** (D-γ retry) — substrate retained but
  not re-bench'd in C. Eligible for re-bench only AFTER C.C+C.D ship
  (per-band suppression decision needs `AecState.UsableLinearEstimate`
  per-band gating).
- **Shadow filter type change** (A retry) — substrate retained, eligible
  only AFTER C.E migrates regime handler to read AecState (regime
  handler threshold then becomes derivable from AecState rather than
  hand-tuned per filter type).

### 3.5 v3.18 closeout positioning

Even if all of C.A-C.E lands and PASSes, **v3.18 may still ship 0
algorithm changes** because C is enabling infrastructure, not
mechanism. The user-facing improvement comes from the D-γ / F / A / B
retries that depend on C — and those are v3.19+ scope per this design.

C **does** raise the v3.18 closeout quality: instead of "4 closed CANNOT
SHIP", v3.18 ships "centralised AecState + FQA infrastructure ready
for v3.19+". That's substantive engineering value even with no
algorithm change.

### 3.6 v3.18 alternative: descope C, ship closeout-only

Per Phase B closeout's "Recommendation §" alternative — if C's LOE
(25-30 sprints) exceeds user appetite, an alternative v3.18 closeout
package is feasible:
- Ship 0 algorithm change
- Document all 4 closure decisions + substrate retained
- Surface v3.19+ priority arc inventory
- `feature/v3.17` rolls forward as `v3.17.1` (closeout commits only)

C.1 design captures both paths so user can choose at the re-auth gate
before C.2.

## 4. Reverse-evidence risk register

| # | Risk | Mitigation |
|---|---|---|
| R1 | C is itself an AEC3 port that fails the co-tuning wall | C reimplements rather than imports (§3.2). Public API matches AEC3; internals match our pipeline |
| R2 | `usable_linear_estimate` multi-gate coverage may not exceed `_filter_converged` on our pipeline | C.B pre-bench gate (§3.3) — close C.D / C.E if so. C.A + C.B retained as observability infra |
| R3 | 120-site consumer migration introduces silent byte-equal break | Per-batch (5-10 sites) verification with kill-and-revert protocol |
| R4 | Phase C.D RES per-state refactor regresses FS or DT | C.D has its own 60-case bench gate; substrate retained on failure (matches D-γ / F / A / B pattern) |
| R5 | 25-30 sprint LOE exceeds user appetite for v3.18 | §3.6 alternative: descope to closeout-only ship |
| R6 | C.A's `FilterAnalyzer` overlaps with v3.14 Arc P (per-band ERL EMA) currently in production | C.A explicitly subsumes Arc P; design lock notes which Arc P fields move into FilterAnalyzer to avoid duplication |
| R7 | qNvSMyU cohort tail (0% `refined_usable`) may still be 0% under multi-gate FQA — cohort tail is genuinely non-converging | Accept upfront: C does not target cohort tail. Cohort tail is v3.20+ matched-filter delay arc |

## 5. C.1 design confidence read

**Reading: 70%** (between A's 65% and B's 75%).

Components:
- (+) Internal refactor with read-only mirror first, byte-equal
  preserved through C.A-C.C — low implementation risk
- (+) Cheap pre-bench gates at each sub-phase (Phase B lesson applied)
- (+) Solves the SHARED root cause of 4 closures (co-design surface)
- (+) v3.18 closeout positioning even if no algorithm ships
- (−) 25-30 sprint LOE is large; risk that user appetite for v3.18
  reaches scope limit before C.D+E ship
- (−) Cohort tail (qNvSMyU) explicitly not targeted (R7 acknowledged)
- (−) C is itself an AEC3-shaped port even if not source-line — R1
  mitigation depends on implementation discipline

Confidence above 60% kill threshold. Proceed to C.A design sprint
**after user confirms §3.6 path choice** (full C.A-C.E vs descope to
closeout-only).

## 6. Phase C hard bar (composite across sub-phases)

### C.A pre-bench gate
- `consistent_estimate` distribution DIFFERS from `_filter_converged`
  on 5-case trace by ≥ 10% in either direction (else port adds no signal)

### C.B pre-bench gate
- `usable_linear_estimate (multi-gate)` coverage exceeds
  `_filter_converged` coverage by ≥ 20 percentage points on 5-case trace
  (else multi-gate refactor adds no coverage)

### C.C verification gate
- AecState public-method outputs match legacy scattered fields on
  byte-equal trace (catch wiring bugs)

### C.D 60-case bench (RES per-state)
- PRIMARY: DT bucket Δdeg ≥ +0.015 (v3.13 E2 Path 3 debt 40% recovery
  target, retained from original plan)
- Guards: FS Δecho ≥ -0.010, cohort tail Δecho ≥ -0.05, NE Δdeg ≥ -0.005
- Kill: PRIMARY fail OR any guard fail → close C.D, substrate retained

### C.E batched migration
- Per-batch (5-10 consumers): byte-equal flag-OFF md5 PASS required
- Per-batch revert + scope-narrow on failure

### C.F 800-case ship gate
- Per-§6 hard bar; requires post-C.D + post-C.E user re-auth

## 7. C.1 → C.A hand-off checklist

C.1 design complete when:
- [x] Scope inventory (§1)
- [x] Sub-phase ordering with kill gates (§2)
- [x] Decisions classified (§3)
- [x] Risk register (§4)
- [x] Confidence read (§5)
- [x] Hard bar (§6)
- [x] §3.6 alternative documented for user re-auth gate

C.2 (= C.A.2) commit gate: **user must choose between**:
1. Full Phase C (25-30 sprints, C.A through C.F)
2. Descoped v3.18 closeout-only (no C, ship verdict pack)

If user chooses (1), proceed to C.A.1 sub-design sprint (FilterAnalyzer
detail). If (2), pivot to v3.18 closeout package.

## 8. Sprint preview if Path 1 (full C)

| Sub-phase | Sprints | Output |
|---|---|---|
| C.A | 5 | `aec_filter_analyzer.py` + audit-only traces + pre-bench gate decision |
| C.B | 5 | `aec_filter_quality.py` + multi-gate computation + pre-bench gate decision |
| C.C | 6 | `aec_state.py` (AecState class) + read-only mirror + verification gate |
| C.D | 5 | leakage_diverged Q-bifurcation + RES per-state ENR + 60-case bench |
| C.E | 4-6 | ~120-site consumer migration in batches |
| C.F | 2 | 800-case + ship gate + §0.7 merge pkg |

Total: 27-29 sprints, ~3-4 months wall-clock at typical sprint rate.

## 9. Cross-references

- [docs/v3_18_b_closeout.md](v3_18_b_closeout.md) — Phase B closeout (motivates C)
- [docs/v3_18_a_closeout.md](v3_18_a_closeout.md) — Phase A closeout
- [docs/v3_18_f_closeout.md](v3_18_f_closeout.md) — Phase F closeout
- [docs/v3_18_d_gamma_closeout.md](v3_18_d_gamma_closeout.md) — Phase D-γ closeout
- [docs/v3_18_plan_revision_2026_05_15.md §4](v3_18_plan_revision_2026_05_15.md#L87) — original Phase C expanded scope
- [docs/aec3_reference.md §9.1](aec3_reference.md#L1036) — AEC3 RES module list (consumers of AecState)
- [docs/aec3_extracts/src/aec3/aec_state.h](aec3_extracts/src/aec3/aec_state.h) — AEC3 reference (header only used as API spec)
- [docs/aec3_extracts/src/aec3/filter_analyzer.h](aec3_extracts/src/aec3/filter_analyzer.h) — AEC3 reference
