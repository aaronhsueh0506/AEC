# E2.S2 verdict — fl scan + envelope + extended pre-alignment

**Date**: 2026-05-13
**Branch**: `feature/v3.11-route-a` (post-substrate merge `60d91f5`)
**Predecessors**: [v3_13_e2_s1_verdict.md](v3_13_e2_s1_verdict.md)
**Status**: **E2.S2 closed — Path 3 (extended bench pre-alignment) is the winner**

## TL;DR

Three parallel paths were investigated for the FS-static 50% Group A
"delay broken" cohort identified in E2.S1:

| Path | Mechanism | C-port impact | Result |
|---|---|---|---|
| **Path 1** | Python `filter_length` scan 832 → 12288 | none (Python-only) | fl=12288 gives FS_static echo **2.125** (vs baseline 1.775) but at 15× memory |
| **Path 2** | Envelope (short-time RMS) GCC-PHAT pre-align prototype | new pre-align module | matches raw GCC at extended search range — superseded by Path 3 |
| **Path 3** | Extend bench `estimate_delay` `max_delay_ms` 250→1024 ms | **none** (bench harness only) | FS_static echo **2.294** at fl=832 — BEATS fl=12288 |

**Path 3 wins by a large margin** on FS_static, at zero C-port cost.
The root cause was a search-range mismatch between the bench
pre-alignment (`max_delay_ms=250`, 4000 samples) and the online
`DelayEstimator` inside AEC (`max_delay_ms=1024`, 16384 samples).
Group A cases have true delays 5653-10882 samples, well outside the
bench search window — GCC-PHAT peak literally cannot be found there,
confidence collapses, the fallback to non-PHAT cross-correlation also
fails (same range cap), and the pre-aligned delay is garbage.

Online tracker subsequently reports huge residual (2647-10467 samples),
but the AEC has no mechanism to re-align the lpb buffer based on the
online estimate (E2.S1 §"Mechanism diagnosis"). So filter_length=832
gets a 5000-10000 sample miss-aligned ref, much of the echo lies
outside filter coverage, residual goes to RES — listen reports
"delay ~10000 samples".

Fix: change one default in
[python/eval_aec_challenge.py:79](../python/eval_aec_challenge.py#L79)
`max_delay_ms=250.0` → `max_delay_ms=1024.0`.

## Bucket-mean AECMOS comparison

8 listen cases, AECMOS echo (higher = better; deg saturates at 5.0):

| Approach | FS_static (n=5) | FS_movement (n=3) | Mem |
|---|---:|---:|---|
| fl=832 max=250 (production baseline) | 1.775 | 1.853 | 1× |
| fl=2080 max=250 | 1.797 | 1.861 | 2.5× |
| fl=4160 max=250 | 1.841 | 1.923 | 5× |
| fl=6400 max=250 | 1.885 | 1.963 | 7.7× |
| fl=12288 max=250 | 2.125 | 1.926 | 15× |
| **fl=832 max=1024 (Path 3)** | **2.294** | 1.860 | **1×** |

Path 3 ΔFS_static = **+0.519** echo vs baseline; ΔFS_movement = +0.007.

FS_static gains dominate because Group A cases (01/04/05/08) had
literally wrong pre-alignment. FS_movement is dominated by Group B
(06/07/08) which had correct pre-alignment but listen complaints
attributed to NL distortion (E4) / saturation (E5) — those bands
are unaffected by pre-align fix.

## Per-case detail — Path 3 vs baseline (FS_static)

| Stem | Group | Bench delay (was → now) | GCC conf (was → now) | echo (was → now) |
|---|---|---|---|---:|
| 7GTxyTks | A | 415 → 10910 | 3.62 → 21.26 | 1.478 → 1.813 (+0.335) |
| IrQvqOTC | B | 1475 → 1475 | 28.47 → 53.50 | 1.831 → 1.831 (+0.000) |
| pcb1Nh0Z | B | 2523 → 2523 | 19.53 → 36.41 | 1.841 → 1.841 (+0.000) |
| **S22FCqKD** | **A** | **0 → 9839** | **8.74 → 32.69** | **1.853 → 4.010 (+2.157)** |
| hVqUmGvI | A | 3006 → 5796 | 4.76 → 13.85 | 1.870 → 1.973 (+0.103) |

Case 04 (S22FCqKD) is the headline — pre-alignment was returning
delay=0 (GCC-PHAT peak literally never inside the 4000-sample window),
so the AEC was seeing completely un-aligned ref. Extended search
finds delay=9839 with conf=32.69 (high confidence) → AEC now sees
properly aligned ref → fl=832 covers residual → echo 1.853 → **4.010**
(near-perfect AECMOS score).

Case 01 (7GTxy) was lazy: even with delay fix, echo improvement is
modest (1.478 → 1.813). Listen note attributes this to E5 (clipping)
on top of E2 (delay). Path 3 fixes E2; E5 deepening sprints (E5.S1-S4)
should pick up the residual.

## Path-by-path detail

### Path 1 — fl scan (Python research only)

Per E2.S1 expectation, fl=12288 (768 ms) covers all Group A residual
delays. Confirmed by the table: FS_static echo 1.775 → 2.125 (+0.350)
at 15× memory. Pareto efficiency: every doubling of fl yields ~0.05
echo improvement on FS_static — diminishing returns.

C-port stays at `filter_length=832` per the v3.13 plan. Path 1 output
is dataset characterization, not a deployment candidate. Confirms
fl alone can recover Group A but at prohibitive memory cost.

### Path 2 — envelope-based pre-alignment

Short-time RMS envelope (20 ms window, 10 ms hop) + GCC-PHAT on the
envelopes, sub-sample refinement via phase-GCC near the envelope
peak. Implementation: [tools/research/e2_s2_envelope_align.py](../tools/research/e2_s2_envelope_align.py)

Result: matches raw GCC-PHAT at extended search range on all 8 cases.
The envelope-vs-raw-GCC equivalence on Group A means the cases were
NOT actually NL-distorted in the way that breaks GCC linearity —
they were simply outside the search window. Once given a 1024 ms
window, raw GCC-PHAT finds them with high confidence.

Path 2 is **superseded by Path 3**: simpler implementation, identical
result, zero new module needed.

(Envelope alignment may still be useful for genuinely NL-distorted
cases — but the listen 8-case did not include such a case in Group A.
Case 07 — the one with "嚴重 NL" listen note — was actually Group B
with GCC conf=21.15, so already handled fine.)

### Path 3 — extended bench pre-alignment

One-line change in
[python/eval_aec_challenge.py:79](../python/eval_aec_challenge.py#L79):
`max_delay_ms=250.0` → `max_delay_ms=1024.0` (matches the online
`DelayEstimator` default).

Implementation tested via standalone script
[tools/research/e2_s2_extended_align_test.py](../tools/research/e2_s2_extended_align_test.py),
which inlines the modified `estimate_delay` and renders the 8 cases
with fl=832 + extended search. Output dir
`results/v3_13_e2_s2_extended_align/`.

GCC confidence on Group A jumped:
- Case 01: 3.62 → 21.26 (5.9×)
- Case 04: 8.74 → 32.69 (3.7×)
- Case 05: 4.76 → 13.85 (2.9×)
- Case 08: 4.11 → 32.05 (7.8×)

All four now well above the confidence=5.0 threshold (which gates
fallback to non-PHAT). The fallback no longer triggers — clean PHAT
peak inside the extended search range.

## Why this wasn't caught before

The bench harness `estimate_delay` was last touched when the online
`DelayEstimator` extended its range from 512 → 1024 ms (v3.10.4 comment
at [python/aec.py:217](../python/aec.py#L217)). The bench function
was forgotten and stayed at the v3.x.0 default of 250 ms.

The bench `estimate_delay` has a confidence=5.0 fallback to non-PHAT,
which masked the issue: instead of returning "no delay found" or
"out of range," it silently returned the best-it-could-do within the
constrained search window. For Group A cases this returned plausible-
looking but wrong delays (0, 415, 3006, 3056) that the AEC then
trusted.

E2.S1 audited the online tracker but only inferred the true total
delay from `pre + online_residual`. It correctly identified the
disagreement but attributed it to "GCC-PHAT confidence too low → wrong
fallback" rather than "search range too small → peak not in window."
Both are technically true but the latter is the actionable root cause.

## 800-case A/B (2026-05-13)

Path 3 rendered full 800-case at `AEC_MAX_DELAY_MS=1024, fl=832,
preset=balanced, cng=True` and scored against `results/v3_11_candidate/`
(v3.11.x production baseline at max=250).

### Bucket means

| Bucket | n | Δecho | Δdeg | v3.13 rule | bench tool verdict |
|---|---:|---:|---:|---|---|
| FS_static | 169 | **+0.107** | -0.000 | ✅ Δecho≥-0.02 | ok |
| FS_movement | 131 | +0.018 | -0.000 | ✅ | ok |
| DT_static | 186 | +0.014 | **-0.050** | ❌ Δdeg≥-0.005 violated 10× | ok (tool only checks FS echo+NE deg) |
| DT_movement | 114 | +0.006 | **-0.025** | ❌ violated 5× | ok |
| NE | 200 | +0.000 | -0.002 | ✅ | ok |

**Cohort tail invariant** `qNvSMyUSXUyrDGp` (P52 hard floor Δecho ≥ -0.05):
- Δecho: 4.024 → 4.020 (Δ -0.004) ✅ within tolerance

**xrtntuju 5-clip DT regression**: both clips Δdeg ≈ 0.000 ✅ preserved.

### Per-case attribution (DT_static worst-7)

The DT_static bucket mean Δdeg = −0.050 is dominated by 7 outliers
(of 186 cases) all with Δdeg < -0.5:

| Stem (DT) | Δdeg | Was/Now deg | Was/Now pre-align delay |
|---|---:|---|---|
| S22FCqKD_doubletalk | -2.045 | 4.157 → 2.112 | 3559 → 9393 |
| JtodX3Ug_doubletalk | -1.913 | 3.442 → 1.529 | 0 → 8524 |
| khqZY41l_doubletalk | -1.585 | 3.252 → 1.668 | 0 → 11859 |
| 7GTxyTks_doubletalk | -1.504 | 3.895 → 2.391 | 3733 → 12608 |
| zzCIhneJ_doubletalk | -0.876 | 2.679 → 1.803 | 176 → 14615 |
| V0Jqgjlr_doubletalk | -0.758 | 2.124 → 1.366 | 756 → 15894 |
| XTqo1aOX_doubletalk | -0.576 | 2.204 → 1.629 | 2711 → 11775 |

**All 7 had broken pre-alignment under max=250** (delays returned
were 0-3733, but true delays per max=1024 are 8524-15894 = 533-993 ms).
The bench was previously ALSO misaligning these DT cases. Extended
alignment now finds the true delay → AEC cancels echo properly → but
in DT mode, the AEC's RES configuration over-suppresses NE speech.

Median Δdeg across DT_static is **0.000** — the rest of the corpus
is unaffected. Bench tool labels "ok" because it only checks FS echo
regression + NE deg regression, but the strict v3.13 rule (`Δdeg ≥ -0.005`
on DT) is violated by the mean.

### Mechanism — why DT regresses

In Group A cases (delay > 250 ms in both FS AND DT variants), the
previous bench measurement was an **artifact** of broken pre-alignment.
Production v3.11.x got Δdeg=4.157 on S22FC_doubletalk because the AEC
was seeing un-aligned ref → couldn't cancel echo at all → NE speech
passed through cleanly → high deg score.

Path 3 reveals that **once the ref is properly aligned**, the AEC
over-suppresses on DT in these long-delay cases. The "Δdeg loss" is
not a regression introduced by Path 3 — it's an **unmasking** of
existing RES behavior that was previously hidden by the bench bug.

Two facts make this attribution clean:
1. Cohort tail and xrtntuju are unaffected (no algorithm change)
2. Median Δdeg = 0.000 on DT_static — only outliers move

### Verdict

**Path 3 reveals an upstream bench harness bug AND an existing AEC
behavior on long-delay DT cases.** The FS recovery is real and
substantial (+0.107 mean, S22FC FS case 1.853→4.010); the DT regression
is bench measurement convergence to underlying AEC behavior, not a
new algorithmic loss.

Per the strict v3.13 plan rules (`DT Δdeg ≥ -0.005`), Path 3 cannot
ship as the new bench default — the DT bucket regression breaches
the hard abort threshold, even if the regression is "unmasking" rather
than "introducing."

Path 3 is documented as a **research finding**. The eval harness
keeps `max_delay_ms=250` as default. The `AEC_MAX_DELAY_MS` env var
override is retained as opt-in for sub-experiments. Next sprint
(E2.S2.b) must address the unmasked DT over-suppression before
the bench harness can be flipped.

## Implementation plan

### Immediate (this sprint) — DO NOT ship bench default flip

Path 3 violates DT Δdeg hard abort. Keep production:
- `python/eval_aec_challenge.py:79` default `max_delay_ms=250.0` UNCHANGED
- `AEC_MAX_DELAY_MS` env var override LANDED (opt-in for research)
- Path 3 results archived for reference

### Next sprint — E2.S2.b: confidence-gated escalation OR DT-aware RES

Two options for unblocking Path 3:

**Option B1 (confidence-gated escalation)**: only escalate to max=1024
if max=250 returns low confidence. For Group B (conf > 5.0 at 250),
keep default → no DT regression. For Group A (conf < 5.0), escalate.
But this still escalates the Group A DT cases (which have the same
low-conf-at-250 property) — likely won't help.

**Option B2 (DT-aware RES configuration)**: investigate why RES
over-suppresses in DT mode on long-delay cases once cancellation
becomes effective. This is upstream of E5 deepening and possibly
overlaps with the deferred Q3 RES re-evaluation. Probably needs
the F3.1 v3 mic-excess metric or per-bin filter_state to gate
RES aggression in DT.

**Option B3 (accept DT regression as truth)**: argue that the
underlying AEC behavior IS the truth and the previous bench
measurements were artificially inflated. Update baseline reference
to Path 3 numbers, then re-evaluate all subsequent sprints relative
to the new baseline. Requires user authorization to change baseline.

Recommendation: defer the decision. Path 3 verdict establishes the
bench bug existence and quantifies trade-off; B1/B2/B3 choice should
be revisited when DT-side investigation has more substrate.

### Followup (future sprints)

**E2.S3 — F-DelayTrack period scan**: FS_movement still needs in-band
tracking. Group A is solved by Path 3; FS_movement residual leakage
(echo 1.853 → 1.860, marginal) means the movement bucket is dominated
by drift, not initial alignment. Period scan 0.05 / 0.10 / 0.25 / 0.50 s
on FS_movement subset planned.

**E2.S4 — movement persistence classifier prototype**: per-frame
movement class dump on 800-case; whether to enable continuous delay
tracking vs static pre-align based on classification.

**E2.S5 — combined deployable A/B**: Path 3 + F-DelayTrack + movement
classifier flag-gated 800-case A/B at fl=832.

## C-port implications

**Zero impact**. Path 3 is bench-harness-side. The C-port AEC's online
`DelayEstimator` already uses `max_delay_ms=1024.0` (Python parity).
The C-port relies on the **caller** to provide pre-aligned ref, so as
long as the integration platform's pre-alignment uses ≥1024 ms search,
Group A is handled. If the integration platform uses 250 ms (legacy),
the C-port AEC's online tracker will see the residual and report it
via `delay_reliable` — but does not currently re-align the ring buffer
on that signal.

Optional future work: have the C-port AEC accept an
"online-tracker-feedback to ring buffer" mode (currently the online
estimate only gates RES, not ref alignment). This would let the
C-port self-correct platform pre-align errors. Out of scope for E2.S2.

## Critical files reference

- [python/eval_aec_challenge.py:79](../python/eval_aec_challenge.py#L79)
  — bench pre-alignment (Path 3 target)
- [python/aec.py:225](../python/aec.py#L225) — AecConfig.max_delay_ms
  (already 1024.0)
- [python/aec.py:5690](../python/aec.py#L5690) — `delay_reliable` consumer
  (RES gate, not ref align — architectural gap)
- [tools/research/e2_s2_fl_scan.py](../tools/research/e2_s2_fl_scan.py)
  — Path 1 fl scan harness
- [tools/research/e2_s2_envelope_align.py](../tools/research/e2_s2_envelope_align.py)
  — Path 2 envelope prototype
- [tools/research/e2_s2_extended_align_test.py](../tools/research/e2_s2_extended_align_test.py)
  — Path 3 extended pre-align test
- `results/v3_13_e2_s2_fl_*/` — fl scan outputs (gitignored)
- `results/v3_13_e2_s2_envelope/` — Path 2 results (gitignored)
- `results/v3_13_e2_s2_extended_align/` — Path 3 output (gitignored)
- `results/v3_13_e2_s2_score_*/` — AECMOS scores (gitignored)

## Sources

- E2.S1 verdict ([v3_13_e2_s1_verdict.md](v3_13_e2_s1_verdict.md))
- Online `DelayEstimator` range history (v3.10.4 changelog at
  [python/aec.py:217](../python/aec.py#L217))
- AEC Challenge blind 8-case worst-FS listen findings (2026-05-13)
