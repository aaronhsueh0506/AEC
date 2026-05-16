# v3.18 Plan Revision (2026-05-15, post-Phase 0 audit)

**Trigger**: Phase 0 mapping doc reads + user discussion 2026-05-15 on
matched-filter delay / EchoPathVariability / AecState centralisation.
User decision: **Option 2 — v3.18 收斂 + v3.19+ 排程**.

This doc patches the original v3.18 plan at
`~/.claude/plans/se-aec-aec-main-hazy-lynx.md` rather than rewriting it.

## 1. v3.18 final scope (selected for ship cycle)

```
Phase 0   ✓ COMPLETE (fetch + 4 mapping docs + decision sprint)
Phase D   ✗ CLOSED 2026-05-16 — CANNOT SHIP; substrate retained
            See docs/v3_18_d_gamma_closeout.md
Phase F   ✗ CLOSED 2026-05-16 — CANNOT SHIP; substrate retained
            See docs/v3_18_f_closeout.md
Phase A   ✗ CLOSED 2026-05-16 — CANNOT SHIP; substrate retained
            See docs/v3_18_a_closeout.md
            A.7 60-case bench: PRIMARY FAIL (qNvSMyU Δecho -0.0011);
            4 worst-case breaches. Bucket means OK but mixed.
            3-of-3 AEC3-port arcs (D-γ + F + A) now CLOSED.
Phase B   ✗ CLOSED 2026-05-16 — CANNOT SHIP; substrate retained
            See docs/v3_18_b_closeout.md
            B.4 fire-rate gate FAIL (0 fires across 61 cases).
            Root cause: _erl_estimate semantics + refined_usable
            coverage are co-design gaps. Phase C unblocks.
            4-of-4 AEC3-port arcs (D-γ + F + A + B) now CLOSED.
Phase C   ⏳ IN PROGRESS — user chose Path 1 (full Phase C) 2026-05-16
            C.1 design lock — docs/v3_18_c1_aec_state_design.md
            C.A COMPLETE — docs/v3_18_c_a_verdict.md
              FilterAnalyzer audit-only port; pre-bench + 60-case gates PASS
              consistent_estimate covers 60-93% per bucket vs 0-9% _filter_converged
              Critical: qNvSMyU cohort tail 54.5% consistent (was 0% converged)
            C.B COMPLETE — docs/v3_18_c_b_verdict.md
              FilteringQualityAnalyzer multi-gate; pre-bench + 60-case PASS
              fq_usable covers 52-86% on FS/DT vs 4-9% _filter_converged
              qNvSMyU 96.3% usable; Phase B retry now viable as C.D follow-up
            C.C COMPLETE — AecState extension (back-ref pattern)
              5/5 byte-equal flag-OFF + audit-only invariant PASS
              Existing AecState class extended with C.A/C.B-aware methods
            C.D-α CLOSED CANNOT SHIP — docs/v3_18_c_d_alpha_closeout.md
              PRIMARY DT_static Δdeg = +0.000 (need ≥+0.010) → FAIL
              Trigger fires but downstream RES doesn't react. Substrate
              retained. Re-ordering: pivot to C.E surgical migration.
            C.E CLOSED CANNOT SHIP — docs/v3_18_c_e_closeout.md
              First non-silent behaviour change. Massive Pareto trade-off:
              DT_static Δdeg +0.214 PASS; FS_static Δecho -0.151 FAIL;
              single-case Δdeg up to +1.200; 18 worst-case breaches.
              Hypothesis 3 STRONGLY CONFIRMED — RES is the AECMOS-dominant
              consumer; whole-arg migration too aggressive. Substrate
              retained; 3 follow-up paths in closeout for v3.19+.
            **Phase C executable scope EXHAUSTED for v3.18 cycle.**

v3.18 CYCLE CLOSEOUT — docs/v3_18_cycle_closeout.md (2026-05-16)
  0 algorithm changes; 8 default-OFF substrate arcs ship.
  Awaiting §0.7 user merge auth (Option A: tag v3.18.0 recommended).
Phase E   PENDING (4-6 sprints) — Substrate flag promotion
```

Total v3.18 LOE: 34-49 sprints original budget; **D + F consumed
~10 sprints, both negative**. Remaining cycle 24-39 sprints across A/B/C/E.
Wall-clock: 2-3 months remaining dedicated work.

## 2. v3.19+ backlog (NEW, deferred)

Per option 2, the following AEC3 mechanism imports stay out of v3.18 to
keep ship-cycle size manageable:

| Phase | Title | LOE | Rationale for v3.19+ |
|---|---|---|---|
| **G** | `InitialState` — startup-vs-steady filter config split (`Q_high_initial` / `Q_low_initial` + transition_triggered_) | 4-5 sprints | Independent feature; v3.18 doesn't depend on it; can ship standalone |
| **H** | `TransparentMode` — no-echo auto-bypass | 3-4 sprints | Standalone module; binary on/off feature; touches AEC class entry point |
| **Delay-revisit** | Matched-filter delay estimator (parallel NLMS + histogram aggregator + dual quality tier) | 5-8 sprints | Re-opens v3.16 C6 closure. PBFDKF main filter independent; delay estimator is separate module. Worth re-evaluating given AEC3 uses structurally different mechanism (multiple parallel NLMS + histogram), not GCC-PHAT. |
| **Reverb-port** | `ReverbModel` + `ReverbDecayEstimator` (from Phase 0.4 mapping) | 5-8 sprints | pcb1N FS_static cohort-specific; v3.17 B.2 closure substrate ready |

Total v3.19+ backlog: 17-25 sprints. Distributed across multiple
cycles based on §0.7 user merge auth + 800-case bench results.

## 3. Phase F detail — NEW (EchoPathVariability event classification)

**Rationale**: AEC3 has 3-way event taxonomy `(gain_change,
delay_change ∈ {kNone, kBufferFlush, kNewDetectedDelay}, clock_drift)`
with **asymmetric cascade reset**:

| Event | Subtractor response | AecState response |
|---|---|---|
| `delay_change != kNone` | Full reset (filter weights, gains, partition size all to initial) | Full reset (filter_analyzer, transparent, ERLE, ERL, FQA all reset) |
| `gain_change` | Only `refined_gains_->HandleEchoPathChange()` (Q boost; weights preserved) | Only `erle_estimator_.Reset(false)` |
| `clock_drift` | No direct reset; downstream consumers gated | No direct reset; downstream consumers gated |

Our current `epc_active` conflates these 3 into 1 boolean. Multiple
v3.15-v3.17 closures (`movement-rate DelayEst B.1`, `dt-ne-compression-fix`,
`PathChangeRegimeHandler`) have asymmetric-event-mishandling as
contributing factor in their failure modes.

**Sprints (Phase F)**:

| Sprint | Action | Hard bar |
|---|---|---|
| F.1 | Add `AecEvent` enum + `ClassifyEpcEvent()` helper. Wire `_classified_event` field default OFF | 5-case byte-equal flag-OFF |
| F.2 | Split `Subtractor.HandleEchoPathChange` into `_handle_gain_change()` (soft, ERLE-only reset) + `_handle_delay_change()` (full reset). Default OFF: same as current behaviour | byte-equal |
| F.3 | Wire delay-event detection from `delay_est` (delta > threshold → fire `kNewDetectedDelay`). When flag-ON, ERLE reset on gain-only / full reset on delay-only — vs current "EPC = both" | trace-only A/B |
| F.4 | 60-case A/B with flag-ON | Cohort tail (qNvSMyU) Δecho ≥ -0.05; expect partial recovery on movement cases |
| F.5 | 800-case bench + ship gate | DT Δdeg ≥ +0.005, FS Δecho ≥ -0.020 |

**Hard bar (F)**:
- Cohort tail Δecho ≥ -0.05 (no regression)
- DT bucket Δdeg ≥ +0.005 (expected from cleaner gain-only event handling)
- FS Δecho ≥ -0.020

**Kill criterion (F)**: DT Δdeg < 0 → close per §0.4, substrate retained.

## 4. Phase C scope expansion

Original Phase C (`leakage_diverged + FilteringQualityAnalyzer`)
remains, but EXPANDED to include AecState centralisation core
([docs/aec3_extracts/src/aec3/aec_state.h](aec3_extracts/src/aec3/aec_state.h)).

New scope:
1. Original C.1-C.5 (FilterAnalyzer + FilteringQualityAnalyzer + leakage_diverged + `usable_linear_estimate` migration of 35+ consumers)
2. **NEW C.6**: Wrap scattered state into `AecState` Python class
   - Subsume: `_filter_converged`, `_filter_once_converged`, `_diag_filter_state`, `epc_active`, `_arc_t_*`, `dt_indicator`, `shadow_dt`, `effective_dt`
   - One source of truth for "is filter usable now" and "what is the AEC pipeline state"
3. **NEW C.7**: Subsume `SaturationDetector` and `ErlEstimator` /
   `ErleEstimator` into AecState wrapper
4. **NEW C.8**: 800-case bench + ship gate

Original LOE: 6-8 sprints. Expanded: 9-12 sprints.

Hard bar additions:
- 35+ `effective_dt` consumers migrated; per-batch byte-equal verified
- `AecState.UsableLinearEstimate()` matches AEC3 multi-gate semantics
- No new test failures in `test_p52_regime.py`

## 5. Phase ordering for v3.18

```
                    Phase 0 ✓
                       │
                       ↓
                  Phase D-γ (active)
                       │
                       ↓
              ┌──── Phase F (event classification)
              │  parallel-safe with D wiring (different files)
              ↓
         Phase A + B (parallel sub-branches)
              │
              ↓
              C (FQA + AecState centralisation)
              │
              ↓
              E (preset promotion)
              │
              ↓
       v3.18 closeout + §0.7 merge auth
```

**Why F early**: F is narrow (3-4 sprints), high-confidence, and
makes downstream A/B/C cleaner because they can read classified events
from a unified source rather than re-parsing `epc_active`.

**Why C late**: AecState centralisation is the biggest refactor;
needs A/B/D/F all settled so we know what state needs to be wrapped.

## 6. v3.18 closeout deliverables (per option 2)

1. Phase D verdict doc (open/close + ship verdict per AECMOS hard bars)
2. Phase F verdict doc
3. Phase C verdict doc + 800-case bench
4. Phase A + B verdict docs
5. Phase E migration audit
6. v3.18 closeout doc — what shipped, what closed, what deferred to v3.19+

## 7. v3.19+ phasing proposal

```
v3.19 (first post-v3.18 cycle, ~4 weeks):
  - Phase G: InitialState (startup config split)         4-5 sprints
  - Phase H: TransparentMode                             3-4 sprints
  - v3.19 closeout

v3.20 (second post-v3.18 cycle, ~4 weeks):
  - Phase Delay-revisit: matched-filter delay estimator  5-8 sprints
  - v3.20 closeout

v3.21+:
  - Phase Reverb-port: ReverbModel + ReverbDecayEstimator  5-8 sprints
  - Pre-echo detection (matched_filter sub-feature)         2-3 sprints
```

Each v3.19+ cycle stays 1-arc-per-cycle per [docs/v3_17_closeout.md](v3_17_closeout.md)
pattern. §0.7 user merge auth per cycle.

## 8. Patch to original v3.18 plan

Original plan at `~/.claude/plans/se-aec-aec-main-hazy-lynx.md`:
- Phase 0 ✓ unchanged
- Phase A ✓ unchanged
- Phase B ✓ unchanged
- Phase C — **expand** per §4 above
- Phase D — **in progress**, see also [docs/v3_18_d1_*.md](v3_18_d1_subband_ne_detector.md), [d2](v3_18_d2_mask_profile_substrate.md), [d3](v3_18_d3_mask_shape_swap.md)
- Phase E ✓ unchanged
- **NEW Phase F** — see §3
- ~~Phase G (originally "fl > 832")~~ — closed in original plan, remains closed
- ~~Phase F-DelayEst rewrite (originally closed per v3.16 C6)~~ — **REOPENED as v3.20 candidate** per AEC3 mechanism analysis

## 9. Cross-references

- Phase 0.2 mapping (RES pipeline): [docs/aec3_residual_pipeline_mapping.md](aec3_residual_pipeline_mapping.md)
- Phase 0.3 mapping (NE detectors): [docs/aec3_subband_ne_detector_mapping.md](aec3_subband_ne_detector_mapping.md)
- Phase 0.4 mapping (reverb): [docs/aec3_reverb_mapping.md](aec3_reverb_mapping.md)
- Phase 0.6 decision: [docs/v3_18_phase0_decision.md](v3_18_phase0_decision.md)
- Original plan: `~/.claude/plans/se-aec-aec-main-hazy-lynx.md`
- Phase F-Delay re-opening context: v3.16 C6 audit closed GCC-PHAT;
  AEC3 uses NLMS-matched-filter + histogram which is structurally
  different family.

## 10. Closeout status snapshot (2026-05-16)

### Phase D-γ — CLOSED CANNOT SHIP
4-config 60-case A/B (dom_only_asym / combined_asym / dom_only_bin /
combined_bin) all fail hard bar:

| Config | FS_static Δecho | DT_static Δdeg | NE Δdeg | Verdict |
|---|---:|---:|---:|---|
| dom_only_asym | +0.018 | −0.034 | −0.001 | FAIL |
| combined_asym | +0.005 | −0.012 | −0.001 | FAIL |
| dom_only_bin | +0.154 | −0.057 | −0.005 | FAIL |
| combined_bin | +0.148 | −0.062 | −0.005 | FAIL |

Hard bar: NE Δdeg ≥ +0.010 PRIMARY (none met) + DT Δdeg ≥ +0.005 (none).
Mechanism: AEC3 `normal_params_` mask too aggressive for our downstream
RES chain (spectral_floor / 3-bin smooth / hf_cap / noise_floor co-tuned
with legacy ne_confidence interp); falls on the same DT-FS Pareto line as
v3.13 E5 saturation arc. Detector accuracy *not* the bottleneck (Dominant
NE detector smoke: FS misfire 19.9% → 2.3%; DT 0I0XMl3M recall 6.6% →
29.3%).

Substrate retained default-OFF: Subband + Dominant detectors, mask profile
tables, binary + asymmetric swap modes, 8 AecConfig fields, env-var
overrides. Re-usable when Phase A/B improve linear filter quality (less
echo headroom needed → softer mask profile suffices) OR when Phase C lands
per-band RES chain refactor.

Verdict: [docs/v3_18_d_gamma_closeout.md](v3_18_d_gamma_closeout.md).

### Phase F — CLOSED CANNOT SHIP

F.1 / F.2 / F.3 substrate landed (5-case byte-equal flag-OFF PASS each
sprint). F.4 60-case flag-ON A/B:

| Bucket | Δecho | Δdeg | Hard bar | Verdict |
|---|---:|---:|---|---|
| FS_static | **−0.048** | ±0.000 | Δecho ≥ −0.020 | **FAIL** |
| FS_movement | −0.011 | ±0.000 | OK | OK |
| DT_static | −0.013 | +0.007 | Δdeg ≥ +0.005 | borderline |
| DT_movement | +0.013 | **−0.011** | Δdeg ≥ +0.005 | **FAIL** |
| NE | ±0.000 | ±0.000 | — | OK |

Worst-case protect: sx6mxKBQ FS_mv Δecho −0.081 > −0.05 breach.

Root cause: our `EpcEvent.source` ∈ {'epv','shadow_rise'} doesn't actually
mean "gain-only event" — it means "delay shift that GCC-PHAT missed."
Lightening their reset path strips the Kalman P relax + ERL cap that
carries post-path-change FS suppression. Confirms v3.16 C6 audit
prediction.

Substrate retained default-OFF: AecEventType / AecEvent / classify_epc_event
/ _handle_gain_change_soft / _handle_delay_change_full / asymmetric wiring
at EPV + shadow_rise sites / AEC_EVENT_CLASSIFICATION env-var. Re-usable
once v3.20 Delay-revisit lands matched-filter + histogram delay detector
(EPV/shadow_rise will stop double-covering delay events).

Verdict: [docs/v3_18_f_closeout.md](v3_18_f_closeout.md).

## 11. Phase A design-confidence reassessment (2026-05-16)

**Question raised before Phase A.2 commit**: P52 Phase A audit
(2026-05-11, [docs/p52_phase_a_verdict.md](p52_phase_a_verdict.md))
concluded "PBFDKF shadow works fine, keep it" + P52 v1.1 design lock
explicitly states "Keeps homogeneous PBFDKF main + Q-scaled PBFDKF
shadow." Why are we now converting shadow → NLMS?

### What P52 audit actually answered
P52 closed *retiring ShadowCopyController* (renamed to
PathChangeRegimeHandler instead). It did **not** answer "does PBFDKF
shadow give us orthogonal information vs main, or is it correlated."
That's a different question.

### Evidence supporting Phase A (PBFDKF→NLMS)
1. **v3.15 four arcs CLOSED with same root cause**: Arc M (V1/V2/V3) /
   Arc G / Arc F / Arc T S2 — all used shadow signal as input,
   all failed because shadow + main give same update direction in
   high-signal regime.
2. **K_ratio math** (aec3_reference.md §3.2): in high-signal regime,
   `X·P·X* >> R` → `K → 1/X*` — main and shadow Kalman gains converge
   to same value, Q×3.5 only changes P time constant, not steady-state
   K. So shadow follows main's direction.
3. **AEC3 design choice**: AEC3 explicitly uses NLMS for shadow (PBFDAF
   class). This is structural, not implementation convenience.
4. **Substrate exists**: PBFDAF class at [aec.py:1679](../python/aec.py#L1679);
   PBFDKF inherits from PBFDAF [aec.py:1836](../python/aec.py#L1836).
   Construction change is one-line.

### Reverse evidence (acknowledged risk)
1. **v3.14 Arc S-orth.A shipped** "decoupled shadow Kalman state" — was
   partial answer to coupling problem. Yet v3.15 Arc M/G/F/T still failed.
   May indicate problem is in *signal interpretation*, not filter type.
2. **qNvSMyU cohort tail divergence defence**: P52 audit found
   PathChangeRegimeHandler is load-bearing on cohort tail
   (−0.56 dB outlier without it). NLMS shadow doesn't automatically
   provide same defence; may make outlier worse.
3. **F closeout points to DelayEst**: real bottleneck per F may be
   missing delay_change events (GCC-PHAT gap), not shadow filter type.
   Phase A doesn't address that.
4. **D + F both closed**: pattern suggests AEC3-port arcs hit our
   pipeline's co-tuning wall; Phase A may follow.

### Decision (user-authorised 2026-05-16)
Proceed with **A.1 design sprint only** (doc-only, 1 sprint). A.1
output:
- Updated D.0 audit aligned to current code state
- Design lock incorporates §11 reverse-evidence as risk register
- Explicit kill criterion adapted from §0.4: if A.1 design confidence
  reads < 60%, close before A.2 and pivot to B (lower-risk,
  substrate-ready)

A.2 is the unauthorised commitment point. After A.1, present design +
confidence to user for §0.7-equivalent green-light before code edits.

### Updated Phase A hard bar (per §11)
- PRIMARY: cohort tail (qNvSMyU) Δecho ≥ +0.030 dB **AND** no new
  qNvSMyU outlier worse than P52 baseline (−0.56 dB floor maintained)
- Guards unchanged: DT Δdeg ≥ −0.005, FS Δecho ≥ −0.010, NE ≥ −0.005
- Kill criterion: cohort tail no improvement OR qNvSMyU outlier
  regresses → close per §0.4, substrate retained

