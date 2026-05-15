# v3.15 §1.5b Arc M.v3 CLOSED CANNOT SHIP — T-gate timing mismatch (substrate)

**Date**: 2026-05-15
**Branch**: `feature/v3.15-arc-g` (consolidated)
**Sprint**: §1.5b.S1 PASS (byte-equal sanity); §1.5b.S2 SUBSET FAIL → close per §0.4
**Substrate retained**: `arc_m_t_gated_enabled` flag (default OFF), additive on
top of Arc M V1 (`arc_m_epc_gated`) and Arc T (`arc_t_cohort_detector`) flags.

## Disposition summary

§1.5b CLOSED at S2 (subset bench) per §0.4 kill criterion: H2 + H3 +
H4b FAIL on subset proxy. The mechanism that failed is **structural**
(not tunable): T-gate cannot suppress the destructive `_arc_m_q_boost`
calls because the EPC events that trigger Q boost happen at frames
where `cohort_tail_T` signal is False, OR the suppressed call is on the
shadow filter (no output impact post S-orth.A decoupling).

Per design doc § "If S2 FAIL" — Arc M family permanently CLOSED through v3.15.

## Subset evidence (60-case Tier 1)

Configurations:
- C0 baseline: all v3.15 flags OFF (= v3.14 BALANCED)
- C1 V1 reproduction: `arc_m_epc_gated=1, kalman_q_per_band=1, kalman_q_band_scales=0.5,1.0,2.0`
- C2 §1.5b candidate: C1 + `arc_m_t_gated_enabled=1, arc_t_cohort_detector=1`

### Hard bar adjudication (per §1.5b design doc table)

| Bar | Target | Subset Δ (60 cases) | Result |
|---|---|---:|---|
| H1 cohort_tail Δecho ≥ −0.05 | rescue from V1's −0.0496 | per-case identical to V1 (no rescue) | n/a |
| H2 DT_movement Δdeg ≥ +0.020 | preserve V1's +0.023 | **+0.0011** | **FAIL** by 18× |
| H3 FS_movement Δecho ≥ −0.020 | rescue from V1's −0.027 | **−0.0604** | **FAIL** by 3× |
| H4a NE Δdeg ≥ −0.005 | regression-guard | +0.0000 | PASS |
| H4b DT_static Δdeg ≥ +0.010 | regression-guard | +0.0007 | FAIL |

Subset H3 = −0.0604 dB on FS_movement (subset includes high-noise cohort
tail FS_movement cases — V1 hits hardest there), worse than V1's 800-case
−0.0269. Even on subset proxy with statistical noise, H3 is 3× over the
hard bar and clearly inherits V1's wall.

### Per-bucket ΔERLE_lin (linear filter PRIMARY metric per §0.6)

| Bucket | n | V1 ΔERLE_lin (C1−C0) | M.v3 ΔERLE_lin (C2−C0) |
|---|---:|---:|---:|
| cohort_tail | 6 | **−0.0203** | **−0.0203** (identical) |
| FS_static | 10 | −0.1422 | −0.1422 (identical) |
| FS_movement | 9 | −0.1011 | −0.1011 (identical) |
| NE | 11 | +0.0000 | +0.0000 (identical) |
| DT_static | 13 | −0.0265 | −0.0265 (identical) |
| DT_movement | 11 | −0.0413 | −0.0413 (identical) |

**V1 ΔERLE_lin == M.v3 ΔERLE_lin in EVERY bucket, EVERY decimal place** —
linear filter output is byte-equal between C1 and C2. T-gate has zero
effect on filter convergence. Per-case MD5 verified on `qNvSMyU`:

```
md5 C1 ours_nores.wav = 92de45fd7d932f5598cef8e676b3af7b
md5 C2 ours_nores.wav = 92de45fd7d932f5598cef8e676b3af7b  ← identical
md5 C1 ours.wav       = 7b8efe21278aa88dab69ffca0ce8164a
md5 C2 ours.wav       = 7b8efe21278aa88dab69ffca0ce8164a  ← identical
```

## Mechanism analysis: T-gate timing mismatch

Direct trace on `qNvSMyU` (cohort tail canonical) with V1+T-detector ON:

```
T detector fires:                  373 frames (signal asserts often)
_arc_m_q_boost called total:       5 times in 26.86 sec (EPC events sparse)
  ... where signal=True at fire:    1 (suppressible)
  ... where signal=False at fire:   4 (untouchable)

C2 wrapper invocations:            4 (1 fire suppressed by gate)
  ... all signal=False               (the 1 signal=True case was suppressed)
```

Two failure modes compound:

### Failure mode 1: signal/fire timing mismatch

`_arc_m_q_boost` is invoked at **discrete impulsive events** (5 per
26-sec case): warmup re-arm, delay_shift, shadow_decision.boost_q,
epv_event, rise_event. These are RISING-EDGE detectors that fire at
abrupt path changes.

`cohort_tail_T` is a **persistent detector** (~373 frames asserted of
~2700 total = 14%) tracking the ERL_decile_std proxy. It asserts
during sustained cohort tail catastrophe windows.

At the q_boost fire moments (rising edges), 4 of 5 fires occur in
frames where signal is **False** because:
1. The 1-frame latency (per design doc R1) means signal hasn't
   propagated by the time the gate evaluates
2. Rising-edge events tend to fire AT the boundary of signal assertion,
   not within sustained windows
3. Hysteresis covers FRAMES, not EVENTS — Q boost events that happen
   between hysteresis-released frames see signal=False

Design doc R1 estimated 1-frame leakage as 0.4% of damage. **Reality is
much worse**: the q_boost events themselves are aligned with the
boundary of signal assertion, so 80% (4/5) of fires are "leaks", not
just 0.4%.

### Failure mode 2: suppressed fire was on shadow filter

The 1 fire that the gate did suppress in C2 (the only signal=True moment)
was on the shadow filter. Per S-orth.A decoupling, shadow filter state
does not affect main output path. Q changes on shadow only affect
shadow's residual estimate, which (per Arc T S2 closure) does not
propagate to ours.wav output.

Therefore the entire effect of §1.5b on this case is **mathematically
zero** despite the gate firing once. MD5 confirms: `qNvSMyU` C1 and C2
output is bit-equal.

## Why this is structural (not tunable)

The T-gate design assumes signal/fire temporal alignment. Reality:
- q_boost events are RARE (~5 per case, ~0.2 Hz)
- cohort_tail signal is COMMON during cohort tail (14% of frames asserted)
- These two distributions do NOT overlap meaningfully (the rising-edge
  events happen mostly OUTSIDE the asserted signal windows, with only
  occasional overlap on shadow filter)

To rescue this design would require:
- Option α: extend hysteresis to PRE-assert (predict cohort tail
  before EPC fires) — speculative, no detector primitive
- Option β: extend gate to suppress Q boost for N frames AFTER any
  recent signal assertion (decay hysteresis on the gate) — requires
  detector signal propagation rework
- Option γ: separately gate main vs shadow Q boost (only suppress on
  main, allow shadow) — but the gate is per-call symmetric; would
  need per-filter dispatch

All three options exceed §0.4 "1 tuning sprint" budget. Per design doc:
"Permanently here means: the §1.5b additive flag joins arc_m_epc_gated
as default-OFF research substrate; the Arc M arc closes for v3.15 with
no further sprints."

## Per §0.4 closure protocol

> §0.4 kill criterion: any of the 3 main hard bars FAIL after 1 tuning
> sprint → CLOSE Arc M permanently, V1 substrate stays default-OFF,
> write closure post-mortem documenting which bar broke.

H2, H3, and H4b fail on subset proxy. Mechanism evidence (trace on
qNvSMyU) confirms the failure is structural. **Arc M family permanently
CLOSED through v3.15**.

## Files retained (substrate)

- [python/aec.py](../python/aec.py): `arc_m_t_gated_enabled` config flag (line 852)
  and 5 gate-wrap insertions at q_boost call sites. All gated on flag
  default OFF → byte-equal to pre-§1.5b.
- [python/eval_aec_challenge.py](../python/eval_aec_challenge.py): `AEC_ARC_M_T_GATED_ENABLED` env override.
- [docs/v3_15_arc_m_v3_design.md](v3_15_arc_m_v3_design.md): S1 design doc (kept for v3.16 reference).
- [docs/v3_15_arc_m_v3_closure.md](v3_15_arc_m_v3_closure.md): this doc.

## v3.16+ candidates (deferred)

If a future v3.16 release wants to retry Arc M, the substrate provides:
- `arc_m_epc_gated` + per-band Q tilt infrastructure (Arc F substrate)
- `_arc_t_cohort_tail_signal` field (Arc T substrate)
- `arc_m_t_gated_enabled` gate-wrap framework (this arc's leftover)

Required new design work:
1. Solve signal/fire temporal alignment (Option α/β/γ above or a
   different detector primitive — e.g. predictive cohort tail signal,
   or post-assertion hysteresis on the gate)
2. Per-filter dispatch (main vs shadow) for the gate, so the gate's
   suppressions affect main filter output, not just shadow
3. Re-baseline against new v3.16 substrate (post-§1.7 RES refactor)

These are separate v3.16 candidate arcs, not auto-authorized; user
review per §0.7 before opening any retry.

## v3.15 plan impact

§1.5b row → CLOSED CANNOT SHIP. Combined with prior closures:

| Arc | Status | Verdict |
|---|---|---|
| §1.0 v3.14 substrate re-verification | PASS | Phase A.0 done |
| §1.2 DT-NE compression fix | CLOSED CANNOT SHIP | FS-vs-DT wall |
| §1.3 Arc D merge | DEFERRED | gated post Phase E (now skipped — Phase E mostly closed) |
| §1.4 Arc M (S1+S2) V1+V2 | CLOSED CANNOT SHIP | Arc F wall |
| §1.4 Arc G S3-S5 | CLOSED CANNOT SHIP | W-reset destructive |
| §1.5 Arc T S1 (detector) | PASS (substrate) | signal correctly asserts |
| §1.5 Arc T S2 (RES preempt wiring) | CLOSED (no-op wiring) | H1 dead code in ENR; H2 overwritten by state machine |
| §1.5b Arc M.v3 (T-gated rescue) | **CLOSED CANNOT SHIP** | T-gate timing/scope mismatch |
| §1.6 Arc F per-band Q | CLOSED CANNOT SHIP (substrate) | steady-state stability wall |
| §1.7 RES audit + v3.16 refactor plan | PENDING | doc-only arc |
| §1.8 E1 pcb1N patch | PENDING | single-case |

**v3.15 production deliverables = 0 algorithm changes** (all 8 candidate
arcs closed). v3.15 closeout will be:
- Bug fixes only (B4 quiescent re-sync correctness if applicable; B5)
- §1.7 RES audit doc → v3.16 refactor plan input
- §1.8 pcb1N single-case patch (pending)
- Substrate retention: Arc P + R + S-orth.A on main (pre v3.15);
  Arc G + Arc T + Arc M v1/v2/v3 + Arc F on `feature/v3.15`
  (default OFF substrate)

The v3.13 E2 Path 3 DT debt (DT_static −0.050, DT_movement −0.025)
remains unrecoverable in v3.15 production. Closure target moves to
v3.16 RES refactor (per §1.7 v3.16 plan candidates).
