# P52 Phase C — Closure

**Date**: 2026-05-12
**Phase A closure**: `p52-phase-a-closed-path3` tag on `main`
**Phase B closure**: `p52-phase-b-closed` tag on `main`
**Status**: **CLOSED by construction** (T6 / T7 / T8 outcomes logically implied; no measurement runs executed)

## Scope

P52 v1.1 §4.5 specifies a Phase C decision tree branching on T7 (800-case
AECMOS regression) outcome. The original 4-configuration matrix is:

| # | Configuration |
|---|---|
| 1 | Baseline (production `main`) |
| 2 | Phase A only |
| 3 | Phase B only |
| 4 | Combined |

Path 3 closure of Phase A retired configurations 2 and 4 from this
matrix (Phase A did not introduce a new shadow-driven intervention —
the controller was *renamed + observed*, not modified at the audio
output level). The matrix reduces to Baseline vs Phase B only, and both
configurations are **provably byte-equal** at the audio sample level
(B.4: 800/800 cases; B.5: confirmed via the production
`use_res_refactored` flag).

The T6 / T7 / T8 outcomes are therefore **logically implied by Phase A
+ B sample-level results**; running them as measurements is a formality
that consumes compute without adding evidence.

## T6 / T7 / T8 resolution (by construction)

### T7 — 800-case no regression

| | |
|---|---|
| Bar | ≥ 95% cases within −0.5 dB AECMOS, no systematic regression |
| Construction proof | Phase B audio output byte-identical to Baseline on 800/800 cases (B.4) |
| Implied outcome | **TRIVIALLY PASS** — Δ AECMOS = 0.000 on every case |
| Measurement run | **not executed** |

### T6 — 9 listener-anchored cases improvement

| | |
|---|---|
| Bar | ≥ 2 dB improvement on ≥ 5/9 windows; no window regresses > 1 dB |
| Construction proof | Phase B audio output byte-identical to Baseline; Phase A Path 3 did not introduce a new intervention |
| Implied outcome | **TRIVIALLY FAIL** — Δ AECMOS = 0.000 on each window, no improvement |
| Measurement run | **not executed** |
| Interpretation | 9 listener-anchored cases remain unsolved within P52 scope. Acknowledged at Phase A close ([p52_phase_a_verdict.md](p52_phase_a_verdict.md)); not a Phase C surprise. |

### T8 — Filter state distribution

| | |
|---|---|
| Bar | ≥ 10 pp gain in `mature` / `refined_usable` filter states on the 9 windows |
| Construction proof | Phase A unavailable (Path 3 did not change main filter); Phase B does not touch `PBFDKF` (per §6.4 isolation) |
| Implied outcome | **TRIVIALLY UNCHANGED** — main filter state distribution identical to Baseline |
| Measurement run | **not executed** |

## Phase C decision tree §4.5 applied

§4.5 branch: **"Phase A unavailable + Phase B trivially compatible"**.

| Tree node | Resolution |
|---|---|
| T7 PASS? | Yes (trivially) → continue tree |
| Promote Phase B as production default? | **No** — keep `use_res_refactored: bool = False` as default; the refactor ships as an opt-in flag. Rationale: B.5 measured +1.48% wall-time overhead from subclass dispatch; production default should bear no overhead until a follow-on ResState migration eliminates that cost. |
| Combined config? | Skipped (Phase A unavailable) |
| Phase A retroactive evaluation? | Not opened (Path 3 already closed Phase A; reopening requires a fresh design lock) |

**Phase C verdict**: P52 closes with Phase B refactor shipped behind
opt-in flag. Production behaviour on `main` is **identical** to
pre-P52-Phase-B state at the audio sample level.

## AECMOS baseline collection — deferred gap

Step 4 of the closure plan called for collecting a standalone AECMOS
baseline on the 800-case corpus for use as a reference by future
audio architecture work (Path 2, NN postfilter, etc.).

**Status: deferred.** AECMOS scoring tooling
([python/bench_aecmos.py](../python/bench_aecmos.py)) requires
`speechmos` + `onnxruntime ≤ 1.16.3` + `numpy < 2`, which are not
installed in the system Python 3.9 used by this session and no `.venv`
exists at the AEC repo root. Setting up the scoring environment from
scratch is outside the scope of Phase C closure.

**Follow-on task** (independent of P52): set up an AECMOS scoring
virtualenv per `README.md` §Benchmark and render the baseline. Output
path: `docs/p52_aecmos_baseline.md` (file name reserved). This task is
not blocking P52 closure.

The byte-equal proof is itself a stronger statement than any AECMOS
score — AECMOS Δ = 0.000 is logically implied by sample identity on
325M samples. The deferred baseline collection is for *absolute*
reference points, not for Phase C decision-making.

## Path 2 future-decision evidence summary

P52 Phase A Path 3 surfaced a quantitative profile of the
PathChangeRegimeHandler firing pattern that should anchor any future
Path 2 (regime-aware redesign) design lock:

| Statistic | Value | Source |
|---|---|---|
| Cases classified `wildly_nonstationary` | 7 / 800 | A.0R.7 |
| Cases classified `mildly_nonstationary` | 59 / 800 | A.0R.7 |
| Cases classified `stable` | 734 / 800 | A.0R.7 |
| Heavy-machinery (`boost_q` + `main_paused`) fires across cohort | ~7 / 800 (0.9 %) | A.0R.7 §3 + §3a |
| Catastrophe-target case in wildly cohort | 1 / 7 (target = `qNvSMyU…`) | A.0R.7 §3 |
| Wildly cases with only `reverse_copy` (gentle sync) | 6 / 7 | A.0R.7 §3 |
| Mildly cases with zero heavy-machinery fire | 55 / 59 (93 %) | A.0R.7 §3a |
| Mildly case with bq=17 / pause=188 (comparable to catastrophe target) | 1 (`MkSLte0F…`, std=12.98 dB — inside mildly band) | A.0R.7 §3a |

Key inference for any future Path 2 design:
**ERL_decile_std-anchored regime classification does NOT discriminate
the cases that need the heavy escalation machinery from those that need
only gentle reverse_copy.** Catastrophe-defence firing aligns with a
frame-level divergence trajectory, not a pre-AEC acoustic class.

A Path 2 design lock that arms on `wildly_nonstationary` alone would
catch 1 of ~7 catastrophe candidates (recall ~14%); arming on
`wildly ∪ mildly` would catch 5 of ~7 (recall ~71%) at precision ~7.6%
(5 useful fires out of 66 trigger events). Neither point is attractive
ROI without a different anchor signal.

**Path 2 disposition**: **deferred indefinitely**. Reopening Path 2
requires a fresh design lock that explicitly anchors on the
A.0R.7 + A.0R.7 §3a finding (frame-level divergence signal, not
acoustic regime label) rather than on the v1.0/v1.1 "shadow as
information source" premise.

PathChangeRegimeHandler stays in production as the active divergence
defence — load-bearing on the cohort tail per
[p52_a0_postmortem.md](p52_a0_postmortem.md).

## Trace-chain final state

| Arc | Closure |
|---|---|
| P30 – P51 | closed at P51 (non-NN WebRTC-like input pipeline exhausted; no further DSP gains identified within the trace-chain methodology) |
| P52 Phase A | **closed via Path 3** (rename + observability + classifier; production unchanged at sample level) |
| P52 Phase B | **closed shipping refactor behind opt-in flag** (default OFF; byte-equal) |
| P52 Phase C | **closed by construction** (this doc) |

## Methodology lessons preserved

From prior arcs (carried forward as standing guidance):

- **P44b instrumentation hygiene** — diagnostic surfaces must be
  audio-passive; never let trace flag mutate production state.
- **P46 cross-cohort first** — validate a hypothesis on the full
  800-case corpus before any 9-window deep dive; the 9 windows are
  not statistically representative of the cohort tail.
- **P47.1 pipeline-source-first traversal** — when a downstream metric
  surfaces a regression, walk upstream to the data-source change
  before tuning the downstream knob.

### New methodology lesson from P52

**P52 lesson: post-mortem before retry.**

P52 Task A.0 (retire ShadowCopyController) borderline-FAILed 800-case
with 1 outlier at −0.56 dB / 1.88% frames > 0.1 dB. The instinct under
v1.0 spec was to retune `_alpha_r` or `shadow_q_ratio` and retry A.0.
Instead, post-mortem on the single outlier case (`qNvSMyU…`)
identified the controller as **catastrophic-divergence defence on the
top ~1% of cohort echo-path-nonstationarity**, not a legacy artifact.
This reframe led directly to Path 3 closure of Phase A (rename +
document, preserve production behaviour) instead of an A.0 retune that
would have produced more borderline-FAILs.

**Application**: when a clean test design fails on a small case-tail,
do **not** retune the proposed change before understanding *why those
specific cases regress*. The case-tail is often load-bearing on
production by design; the test was correct, the design premise was
wrong.

## Outstanding deferred items

| Item | Status | Owner |
|---|---|---|
| `ResState` migration (§3.4 target shape) | Deferred to standalone post-P52 task | Future |
| AECMOS baseline collection on `main` post-Phase-B | Deferred pending scoring venv setup | Future |
| C-impl rename `ShadowCopyController` → `PathChangeRegimeHandler` | Deferred (out of scope per Path 3) | Audio_ALG integration repo |
| Path 2 design lock | Deferred indefinitely; requires fresh anchor signal | Future, gated on evidence |

## Cross-references

- Design lock: [p52_design_lock_v1.1.md](p52_design_lock_v1.1.md)
- Phase A verdict: [p52_phase_a_verdict.md](p52_phase_a_verdict.md)
- A.0 post-mortem: [p52_a0_postmortem.md](p52_a0_postmortem.md)
- A.0R.7 regime distribution: [p52_a0r_regime_distribution.md](p52_a0r_regime_distribution.md)
- A.0R.8 runtime sanity: [p52_a0r_runtime_sanity.md](p52_a0r_runtime_sanity.md)
- Phase B verdict: [p52_phase_b_verdict.md](p52_phase_b_verdict.md)
- B.4 byte-equal: [p52_phase_b_b4_verdict.md](p52_phase_b_b4_verdict.md)
- B.5 flag-path validation: [p52_phase_b_b5_verdict.md](p52_phase_b_b5_verdict.md)
- Errata: [p52_errata.md](p52_errata.md)
- Phase C kickoff (now superseded by this closure): [p52_phase_c_kickoff.md](p52_phase_c_kickoff.md)
