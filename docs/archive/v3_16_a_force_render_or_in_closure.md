# v3.16-A — force_render OR-in CLOSED CANNOT SHIP (2026-05-15)

**Status**: CLOSED per §0.4 negative-result acceptance protocol.
**Branch**: `feature/v3.16` (commit pending).
**Sprint**: v3.16 Phase 1.5 (per `~/.claude/plans/se-aec-aec-main-hazy-lynx.md` §10).
**Predecessor**: Arc T S2 closure
([`v3_15_arc_t_s2_wiring_closure.md`](v3_15_arc_t_s2_wiring_closure.md)).
**Substrate**: retained `arc_t_force_render_or_in: bool = False` default OFF.

---

## 1. Headline

**Verdict**: v3.16-A's `force_render` OR-in is **logically subsumed by
the existing `not filter_converged` clause** in cohort tail catastrophe
windows. Predicted +0.030 cohort tail Δecho is **NOT REALIZED** (qNvSMyU
byte-identical flag-ON vs flag-OFF). Same root-cause family as Arc T S2
(H1+H2 closure) — the cohort tail signal asserts in states where
`force_render` is already True via another clause.

**Substrate retained**: code wired, flag default OFF (byte-equal),
ready for downstream re-use if a future arc decouples filter
convergence from cohort tail detection.

---

## 2. Mathematical analysis of the subsume

### 2.1 The force_render predicate (post-v3.16-A)

```python
# python/aec.py ResidualEchoEstimator.attribute_legacy
force_render = (
    epc_active
    or saturation_level > 0.5
    or not filter_converged
    or (self._arc_t_force_render_or_in_enabled
        and self._arc_t_cohort_tail_signal)
)
```

The OR-in clause `(enabled AND cohort_tail_T)` adds new `True` cases
only when:

> `not epc_active AND saturation_level ≤ 0.5 AND filter_converged
> AND cohort_tail_T`

i.e. only when the filter is CONVERGED but the cohort tail detector
still fires.

### 2.2 Empirical fire rate on qNvSMyU (cohort tail catastrophe)

From C6 audit trace ([`/tmp/v3_16_c6_audit/qNvSMyUSXUyrDGpOw7s6qg_farend_singletalk.json`](/tmp/v3_16_c6_audit/qNvSMyUSXUyrDGpOw7s6qg_farend_singletalk.json)):

| Condition | Frames | % of 2686 |
|---|---:|---:|
| `cohort_tail_T = True` | 453 | 16.9 % |
| `cohort_tail_T = True AND filter_converged = True` | 0 | **0.0 %** |
| `cohort_tail_T = True AND not epc_active` | 453 | 16.9 % |
| v3.16-A LEVER ACTIVE (cohort_tail_T AND filter_converged AND not epc_active) | 0 | **0.0 %** |

The filter NEVER converges on this case (filter_converged_rate ≈ 0 %
across the full 26.86-second clip per C6 summary). Therefore
`not filter_converged = True` in 100 % of cohort_tail_T frames →
`force_render` is already True regardless of v3.16-A.

### 2.3 Per-case A/B verification

**5-case byte-equal sanity (flag OFF, with v3.16-A wiring vs pre-wiring
HEAD `ac7320e`)**: 5/5 PASS on `ours.wav` (commit hash test cases:
`Y91uE2t`, `XqvGR01t DT_mvmt`, `014Azuq NE`, `qNvSMyU`, `pcb1N`).

**Per-case ours.wav md5 (flag ON vs flag OFF, 60-case subset)**:

| Case | Bucket | flag-ON vs flag-OFF |
|---|---|---|
| **qNvSMyU FS** (cohort tail catastrophe) | FS_static | **byte-identical** (md5 `5a08ab76…`) |
| Other 26 / 60 cases | mixed | md5 differs |
| Remaining 34 / 60 cases | mixed | byte-identical |

The cohort tail TARGET case (qNvSMyU) is byte-identical → v3.16-A had
zero effect on the actual catastrophe defence path. The 26 cases with
differing md5 are NOT cohort tail cases; lever fires there
incidentally (cohort_tail_T detector occasionally asserts during
converged FS / DT_movement frames).

---

## 3. AECMOS bench (60-case subset)

`tools/research/v3_15_subset_bench.sh /tmp/v3_16_a_off/`  (flag OFF, baseline)
`tools/research/v3_15_subset_bench.sh /tmp/v3_16_a_on/ 'AEC_ARC_T_FORCE_RENDER_OR_IN=1'`  (flag ON)

### 3.1 Bucket means (n shown is per bucket within 60-case subset)

| Bucket | n | OFF echo | ON echo | Δecho | OFF deg | ON deg | Δdeg |
|---|---:|---:|---:|---:|---:|---:|---:|
| FS_static | 11 | 3.507 | 3.506 | **−0.001** | 4.999 | 4.999 | 0.000 |
| FS_movement | 11 | 3.719 | 3.719 | 0.000 | 4.999 | 4.999 | 0.000 |
| DT_static | 14 | 4.439 | 4.439 | 0.000 | 2.377 | 2.377 | 0.000 |
| DT_movement | 13 | 4.003 | 4.003 | 0.000 | 2.421 | 2.421 | 0.000 |
| NE | 11 | 4.999 | 4.999 | 0.000 | 3.747 | 3.747 | 0.000 |

### 3.2 Diffing-case detail (only 7 / 60 cases have any AECMOS Δ > 1e-6)

| Stem | Bucket | Δecho | Δdeg |
|---|---|---:|---:|
| VNkNShj97UajHDVbSmIG0g_FS | FS_static | −0.005 | 0.000 |
| Y91uE2tRg0… FS_movement | FS_movement | −0.001 | 0.000 |
| XnfMDZLl0… DT_mvmt | DT_movement | 0.000 | −0.001 |
| W0zK3dv0… DT_mvmt | DT_movement | 0.000 | 0.000 |
| Hp5g1asac… DT_mvmt | DT_movement | 0.000 | 0.000 |
| Hq00pd6Ey0… FS_mvmt | FS_movement | +0.001 | 0.000 |
| OX2l6zV7… FS_mvmt | FS_movement | +0.004 | 0.000 |

- **Net Δecho across 7 diffing cases**: −0.001 (essentially noise)
- **Largest single-case Δecho**: −0.005 (VNkNShj97…)
- **0 cohort-tail cases differ** — qNvSMyU is byte-identical.

Per [v3.15 §0.4](../../../.claude/plans/se-aec-aec-main-hazy-lynx.md#§04--negative-result-acceptance-protocol):

> "If lever moves bucket-mean < 0.002 dB AND fires < 5 % of frames
> after 3 A/B sweeps, ship as substrate for dependent arc (DO NOT
> rework). Document fire rate in verdict doc."

All bucket means < 0.005. Lever fire rate on **TARGET** catastrophe
case = 0.0 % (0 / 2686 frames on qNvSMyU). Predicted +0.030 cohort
tail Δecho **NOT REALIZED**. Single tuning sweep sufficient to confirm
mathematical subsume; second sweep would not change the outcome.

---

## 4. Same closure family as Arc T S2

The v3.15 §1.5.S2 Arc T closure
([`v3_15_arc_t_s2_wiring_closure.md`](v3_15_arc_t_s2_wiring_closure.md))
identified two failure modes for the cohort tail signal consumer:

| Hypothesis | Original mechanism | Closure root cause |
|---|---|---|
| **H1** (`over_sub × 1.3` boost) | DEAD CODE in BALANCED (over_sub only read by gain_type='wiener'; all presets use 'enr') | mechanism never executes |
| **H2** (`_using_render_based = True` from AEC level) | OVERWRITTEN by `attribute_legacy` state machine 1 line later | wiring no-op |

v3.16-A targeted H2: rewire the OR INSIDE `attribute_legacy`. The fix
is mathematically correct — it survives the state machine override —
but the signal arrives in a state where `not filter_converged` already
sets `force_render = True`. Result: behavioural no-op for the same
reason H1+H2 were no-ops, just at a different layer of indirection.

**Lesson**: cohort tail catastrophe windows are characterised by
filter non-convergence (per the very definition of catastrophe class).
Any RES-side trigger gated on cohort_tail_T is **OR-subsumed by
not filter_converged**. To deliver new behaviour, the consumer must
operate on a SUBSET of `not filter_converged` — i.e. tighten the gate,
not loosen the OR.

---

## 5. v3.16 plan implications

### 5.1 Phase ordering re-evaluation

Per the aggressive ordering (C6 → v3.16-A → C5) locked 2026-05-15,
v3.16-A was promoted out of Phase 3 to deliver the highest predicted
Δ ASAP. With v3.16-A closed:

| Original ordering | Status post-closure |
|---|---|
| Phase 1.0 C6 audit | ✓ CLOSED H2 (no delay arc opened) |
| Phase 1.5 v3.16-A | ✓ CLOSED CANNOT SHIP (subsume) |
| Phase 1.6 C5 architectural foundation | **NEXT** — sole remaining Phase 1 arc |

Phase 1 reduces to a single arc (C5). After C5 lands, Phase 2 (C2/C3/C4)
opens, and v3.16-B (Phase 3) becomes the next aggressive-priority
candidate — but v3.16-B is gated on C2 (ENR write-conflict
resolution) per `Critical gate logic` in plan §10.

### 5.2 v3.16-B re-prediction

v3.16-B was predicted +0.020 cohort tail Δecho (similar mechanism
family: ENR-path lift via `enr_t_ne × cohort_tail_T_boost`). The
v3.16-A closure reveals a **structural concern**: if v3.16-B's
trigger is also `cohort_tail_T`, and `not filter_converged` is also
True in those windows, does v3.16-B suffer the same subsume?

**Key difference**: v3.16-B modifies `enr_t_ne` (the ENR threshold for
NE-suppression cut-on). The ENR gate fires per-bin based on
`enr = residual_echo_psd / nearend_est`; it is NOT gated on
`filter_converged`. So v3.16-B's effect surface is per-bin
suppression aggressiveness — orthogonal to force_render. The subsume
analysis does NOT apply.

**Action**: v3.16-B can still proceed when C2 completes. Re-validate
predicted Δ at sprint kickoff (re-bench on cohort tail with
`cohort_tail_T_boost` swept).

### 5.3 C9 reverb-aware (Phase 4) implications

C9 was scoped on mic-lpb cross-correlation + far_power gating
(per HK-2 reframing). The v3.16-A closure does NOT impact C9 because
C9 does not depend on `cohort_tail_T` — it has an orthogonal
detector. Scope unchanged.

### 5.4 C7 Arc M.v3 (Phase 4) implications

C7's α / β / γ rescue paths target Arc M's V1+V2 closure. None
target cohort tail catastrophe via `force_render`. Subsume does not
apply. Scope unchanged.

---

## 6. Substrate (committed)

Files modified (5 surgical edits):

1. `python/aec.py` — `AecConfig.arc_t_force_render_or_in: bool = False`
   (~line 914).
2. `python/aec.py` — `ResidualEchoEstimator.__init__` adds two
   per-instance fields (`_arc_t_cohort_tail_signal`,
   `_arc_t_force_render_or_in_enabled`).
3. `python/aec.py` — `ResidualEchoEstimator.attribute_legacy`
   force_render OR-in (~line 4623).
4. `python/aec.py` — AEC.__init__ wires flag onto
   `self.res._residual_est._arc_t_force_render_or_in_enabled`
   (~line 5662).
5. `python/aec.py` — AEC.process per-frame propagates
   `_arc_t_cohort_tail_signal` to `_residual_est`
   (~line 7394).
6. `python/eval_aec_challenge.py` — `AEC_ARC_T_FORCE_RENDER_OR_IN`
   env override (~line 306).

All changes byte-equal-flag-OFF verified (5-case sanity PASS).
**Net LOC delta**: ~25 added, 0 removed.

---

## 7. Verdict signed-off

**CLOSED CANNOT SHIP** per §0.4. Substrate retained as default-OFF
flag for potential downstream re-use (e.g. v3.17 if a "converged-cohort-
tail" sub-detector ever materialises). v3.16 Phase 1.6 **C5
architectural modularisation** is the next sprint.
