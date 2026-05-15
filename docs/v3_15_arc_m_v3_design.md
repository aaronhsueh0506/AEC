# v3.15 §1.5b Arc M.v3 — T-gated rescue retry of Arc M V1 (sprint design)

**Date**: 2026-05-15
**Branch**: `feature/v3.15-arc-m-v3` (to be created **after** §1.5 Arc T merges
into `feature/v3.15`; rebase ordering enforced by §0.7).
**Sprint**: §1.5b.S1 (impl + 5-case byte-equal) and §1.5b.S2 (800-case A/B + verdict).
**Substrate retained on close**: `arc_m_t_gated_enabled` flag (default OFF) on
top of Arc M V1 substrate (`arc_m_epc_gated` + `_arc_m_q_boost`) and Arc T
substrate (`arc_t_cohort_detector` + `_arc_t_cohort_tail_signal`).

## Hard bars (all four required)

| # | Bar | V1 result | §1.5b target |
|---|---|---:|---:|
| H1 | cohort tail (`qNvSMyU`) Δecho | **−0.0496** PASS-margin 0.0004 | ≥ −0.030 (rescue ≥ +0.020 dB) |
| H2 | DT_movement Δdeg | **+0.0233** PASS | ≥ +0.020 (≥ 87% retention of V1 win) |
| H3 | FS_movement Δecho | **−0.0269** FAIL (1.34× over) | ≥ −0.020 (rescue ≥ +0.007 dB) |
| H4a | NE Δdeg | −0.0009 PASS | ≥ −0.005 (regression-guard) |
| H4b | DT_static Δdeg | +0.0150 PASS | ≥ +0.010 (regression-guard) |

(V1 column = Arc M V1 row from
[`docs/v3_15_arc_m_closure.md`](v3_15_arc_m_closure.md), against
`results/v3_14_baseline/scores.json`.)

## §0.4 kill criterion

If **any of H1, H2, or H3** FAILS after one tuning sprint (one calibration
pass on the gate threshold or hysteresis), §1.5b CLOSES PERMANENTLY:
- Arc M V1 substrate stays default-OFF.
- Arc M.v3 substrate stays default-OFF (additive flag, no removal).
- Closure post-mortem written to
  `docs/v3_15_arc_m_v3_closure.md` documenting which bar broke and root
  cause (e.g. T-gate FN missed cohort tail = H1 fail; T-gate FP truncated
  V1 wins = H2 fail; FS_movement leak orthogonal to T = H3 fail).

H4a/H4b are regression-guards; failing only those triggers the same close
but the post-mortem is treated as "no harm, no foul" (V1 was already
passing those bars, regression should not be created by the gate).

## §0.6 metric channel

- **PRIMARY**: linear-filter `nores` listen on the DT_movement cohort
  (5-clip subset rebuilt from V1 nores render at e857209). The Arc M Q
  boost effect is upstream of RES, so nores is the correct channel.
- **REGRESSION-GUARD**: full-pipeline AECMOS 800-case bench, focusing on
  cohort tail bucket (`qNvSMyU` + 6 known siblings) + FS_movement bucket.
- **NOT used as primary**: AECMOS DT_movement Δdeg — it conflates RES and
  filter contributions; the V1 +0.023 dB win was already validated as
  filter-domain (NE-bleed compression unwound) by the V1 nores listen.

## §0.7 branch policy

Work happens on `feature/v3.15-arc-m-v3`. Merge into `feature/v3.15`
**only after** §1.5b.S2 PASS (all H1+H2+H3+H4a+H4b green). `main` is
read-only until the user-authorised final v3.15 → main merge that batches
all v3.15 arc closures together.

## Why §1.5b — rationale

Arc M V1 produced a **real, large, filter-domain DT_movement Δdeg win of
+0.0233** (per [`docs/v3_15_arc_m_closure.md:16`](v3_15_arc_m_closure.md#L16))
that was the largest DT_movement filter-side improvement of the v3.15 cycle.
The closure verdict (CANNOT SHIP) was driven by FS_movement Δecho
−0.0269 (1.34× over the −0.020 hard bar) and a marginal cohort tail
Δecho −0.0496 (0.0004 from the bar).

Arc M V1+V2 closure analysis identified the failure mechanism as
**EPC-window ⊃ cohort-tail-catastrophe-window**:
PathChangeRegimeHandler's 6-gate AND fires heaviest on `qNvSMyU` class
during the catastrophe window (load-bearing defence role per
[`docs/v3_15_arc_m_closure.md:39-43`](v3_15_arc_m_closure.md#L39-L43)).
Boosting Q during EPC-active windows therefore boosts Q during the
catastrophe windows, and that's where the FS_movement / cohort tail
damage comes from. V2's symmetric tuning (0.7/1.0/1.5) confirmed the
trade-off is real, not a knob choice — neither variant escapes.

§1.5b's hypothesis is the missing **NOT-cohort-tail** factor. Arc T
([`docs/v3_15_arc_t_s1_design_and_verdict.md`](v3_15_arc_t_s1_design_and_verdict.md))
provides a real-time per-frame `cohort_tail_T` Boolean signal. By gating
Arc M's per-band Q tilt on `(EPC_active AND NOT cohort_tail_T)`, the
boost is applied during EPC windows that are NOT cohort tail (i.e. the
DT_movement convergence-recovery windows that produced the V1 +0.023 dB
win) and is **suppressed** during the EPC windows that ARE cohort tail
(i.e. the catastrophe windows that produced the V1 FS_movement /
cohort tail damage).

Decision-theoretic framing: Arc M V1 is a single-rule policy with two
dominant outcome buckets (DT_movement positive, FS_movement / cohort
tail negative). §1.5b inserts a learned discriminator (Arc T) between
them. If the discriminator separates the buckets cleanly (Arc T's S1
8-case validation: 5/5 TAIL fire + 3/3 CTRL no-fire), the gated policy
inherits V1's positive bucket and avoids V1's negative bucket. If the
discriminator's FN rate on the cohort-tail bucket is >0, H1 weakens
proportionally; if its FP rate on the DT_movement bucket is >0, H2
weakens proportionally. Both rates are bounded by Arc T S1 hard bars.

## Mechanism — gate expression and 5 plug-in sites

### Gate expression

At each `_arc_m_q_boost` call site, replace

```python
self._arc_m_q_boost(filt)
```

with

```python
if not (self.config.arc_m_t_gated_enabled
        and self._arc_t_cohort_tail_signal):
    self._arc_m_q_boost(filt)
```

Reading: when `arc_m_t_gated_enabled` is OFF, the gate is bypassed
and behaviour is byte-equal to current Arc M V1. When the flag is ON,
the per-band Q tilt is suppressed during cohort-tail-signal-asserted
windows; uniform Q baseline (`Q = Q_high.copy()`) is also suppressed in
those windows, leaving the **previous** Q value in place. This is the
correct behaviour: during cohort tail catastrophe we explicitly do NOT
want a Q reset (Arc M V1's failure mode).

Alternative considered & rejected: gate only the per-band scale
multiplication (apply uniform `Q = Q_high.copy()` always, skip
per-band scale during cohort tail). Rejected because the V1 closure
verdict identified the per-band tilt magnitude as the wall-controlling
parameter (V1 vs V2 trade-off is symmetric in tilt magnitude); a
uniform reset would still apply the same Q magnitude on the cohort
tail bucket, only via the LF/MF bands. Skipping the entire boost
preserves the previous (already-tilted-or-not) Q until the next
non-cohort-tail EPC fire reapplies it.

### Signal source

`self._arc_t_cohort_tail_signal: bool`

Initialised to `False` at `AEC.__init__` ([aec.py:5640](../python/aec.py#L5640)),
cleared to `False` on full reset ([aec.py:5994](../python/aec.py#L5994)) and
filter-derived reset ([aec.py:6104](../python/aec.py#L6104)). Updated only
when `arc_t_cohort_detector=True AND erl_update_gate AND inst_erl_raw < 1.5
AND _long_window_n_updates >= 100` ([aec.py:7086-7148](../python/aec.py#L7086-L7148)).
**Default-OFF (Arc T flag OFF) holds the field at `False` for every
frame**, so the §1.5b gate `(arc_m_t_gated_enabled AND _arc_t_cohort_tail_signal)`
evaluates to `False` for every frame and Arc M V1 behaviour is
byte-equal preserved. This is the intended dual-flag composition: §1.5b
is a no-op until BOTH `arc_m_t_gated_enabled=True` AND
`arc_t_cohort_detector=True`.

### 1-frame latency contract

Per [`v3_15_arc_t_s1_design_and_verdict.md` §1.5b dependency contract](v3_15_arc_t_s1_design_and_verdict.md#L164-L175):

- Arc T proxy block at line ~7086-7148 writes `_arc_t_cohort_tail_signal`
  AFTER the per-band ERL update block ends, BEFORE `self.res.process()`
  is invoked at the end of the same frame.
- All 5 `_arc_m_q_boost` invocations (lines 6177, 6446, 6772, 6862, 6907,
  see table below) are **earlier** in the same frame than the Arc T
  compute block.
- Therefore §1.5b reads the **previous frame's** `_arc_t_cohort_tail_signal`
  value. Latency = 1 frame = 10 ms (hop=160 / fs=16 kHz).
- Acceptable: `qNvSMyU` cohort tail `ERL_decile_std` timescale is
  ~2.7 s (per Arc T S1 design). 1-frame (10 ms) latency is 270×
  shorter than the timescale, so the gate fires within the catastrophe
  window with negligible boundary smear (≤ 1 frame at start, ≤ 1 frame
  at end; both bounded by the 200-frame Arc T hysteresis).

### 5 plug-in sites at HEAD `2d521b7`

| # | Line | Call path | Iterates over | Loop-internal? |
|---:|---:|---|---|---|
| 1 | **6177** | `_reset_filter_derived_state` post-reset Q boost (warmup re-arm) | main + shadow | yes |
| 2 | **6446** | `delay_shift` Path A reset (Method M4: re-train against new alignment) | main + shadow | yes |
| 3 | **6772** | `shadow_decision.boost_q` from `PathChangeRegimeHandler.update()` | main only | no |
| 4 | **6862** | `epv_event.fired` (EPV detector EPC source) | main + shadow | yes |
| 5 | **6907** | `rise_event.fired` (shadow_rise detector EPC source) | main + shadow | yes |

Plan §1.5b cited approximate line numbers as `~6124, ~6393, ~6719, ~6809,
~6854`. Verified actual line numbers at HEAD `2d521b7` are 6177, 6446,
6772, 6862, 6907 — uniform +53 line drift from the planning estimate
(likely caused by the Arc T S1 + S2 commits on this worktree adding
~110 lines of state init and proxy compute upstream of the invocation
sites). **Site count, call paths, and per-site semantics are
unchanged** from the plan; only the absolute line numbers differ.

The gate wrapping is identical at all 5 sites (the `filt` variable name
is the same in all loops, including the main-only site #3 where it is
spelled `self.filter`). For sites that iterate over `[main, shadow]`,
the gate evaluates **once per loop iteration** (it does not hoist out
of the loop) — this is intentional: the cost of the conditional check
is negligible (one attribute read + one boolean AND), and keeping the
shape symmetric across all 5 sites simplifies code review and the
byte-equal sanity test.

## Risk register

| # | Risk | Mitigation | Impact if realised |
|---|---|---|---|
| R1 | T-gate 1-frame latency causes Q boost to fire on the first cohort-tail frame before T asserts | Bounded by Arc T hysteresis (200 frames); first-frame leakage ≤ 1 frame's worth of Q tilt out of ~270-frame cohort tail event | H1 weakened by < 1/270 ≈ 0.4% of total damage (≈ −0.0002 dB on H1); negligible vs 0.020 dB rescue target |
| R2 | T detector FN (cohort tail event missed) | S1 validation: 5/5 TAIL fire + 3 CTRL no-fire passed; threshold T_HI=18.5 dB has 0.5 dB margin to nearest CTRL max | If FN occurs on a previously unseen cohort-tail-class case, that case retains V1's full damage; 800-case bench detects this in cohort-tail bucket aggregate |
| R3 | T detector FP (non-tail EPC window asserted as cohort tail) | S1 validation: NE-corruption gate (`inst_erl_raw < 1.5`) eliminated DT_static FP on `NN7yhG2X`; FP rate on 3 CTRL = 0/3 | Each FP suppresses one Q boost in a non-tail EPC window → loss of V1 win on those windows. Bounded by S3 acceptance bar (FP rate ≤ 5% on non-tail cohort) → H2 retention ≥ 95% × V1 win ≈ +0.022 dB still passes ≥ +0.020 |
| R4 | FS_movement Δecho damage is NOT actually inside cohort-tail-T windows | Inferred from V1 closure: V1 FS_movement damage came from the same EPC-window-=-cohort-tail-window mechanism as cohort tail damage. If FS_movement damage is orthogonal to cohort_tail_T, H3 fails | §0.4 kill criterion fires; close arc with post-mortem. Probability est. low (V1+V2 closure analysis explicitly identified the SAME mechanism for FS_movement and cohort tail) |
| R5 | Arc T S2 RES preempt mode (`arc_t_res_preempt_mode=True`) interacts with §1.5b gate | §1.5b enables ONLY `arc_m_t_gated_enabled`; `arc_t_res_preempt_mode` stays default OFF. RES preempt and Q gate are read-disjoint: Q gate reads `_arc_t_cohort_tail_signal`, RES preempt reads same signal at L7268 → orthogonal write paths | None if §1.5b S2 keeps `arc_t_res_preempt_mode=False`; explicit in S2 config |

## Sprint plan

### §1.5b.S1 — impl + 5-case byte-equal

**Scope**: code change only.

1. Add `arc_m_t_gated_enabled: bool = False` to `AecConfig`
   (after the existing `arc_m_epc_gated` field at
   [aec.py:850](../python/aec.py#L850)).
2. Wrap each of the 5 `_arc_m_q_boost(filt)` / `_arc_m_q_boost(self.filter)`
   calls (lines 6177, 6446, 6772, 6862, 6907) with the gate expression.
3. Add `AEC_ARC_M_T_GATED_ENABLED` env override in
   `python/eval_aec_challenge.py` immediately after the existing
   `AEC_ARC_M_EPC_GATED` block at
   [eval_aec_challenge.py:268-271](../python/eval_aec_challenge.py#L268-L271).

**Acceptance bars (S1)**:
- 5-case byte-equal sanity (`atol=0.0`, MD5 identical) with both flags
  OFF vs the pre-§1.5b baseline at HEAD `2d521b7`.
- 5-case selection (independent of S1 Arc T cohort and S2 800-case
  acceptance cohort): pick 1 NE + 1 DT_static + 1 DT_movement +
  1 FS_static + 1 FS_movement, all OUTSIDE the cohort tail bucket and
  the Arc T 8-case validation set.
- Default-OFF byte-equal MUST hold for ALL 4 flag combinations:
  `(arc_m_epc_gated, arc_m_t_gated_enabled) ∈ {OFF/OFF}` vs baseline.
  (Other 3 combinations are tested in S2.)

If S1 byte-equal fails → fix wiring (no behaviour decision).

### §1.5b.S2 — 800-case A/B + verdict

**Scope**: bench + adjudication.

Configuration matrix:

| ID | `arc_m_epc_gated` | `arc_m_t_gated_enabled` | `arc_t_cohort_detector` | `arc_t_res_preempt_mode` | Notes |
|---|---|---|---|---|---|
| C0 | OFF | OFF | OFF | OFF | Baseline (= `results/v3_14_baseline/scores.json`) |
| C1 | ON | OFF | OFF | OFF | Arc M V1 reproduction (sanity vs `docs/v3_15_arc_m_closure.md` V1 row) |
| C2 | ON | ON | ON | OFF | **§1.5b candidate** (T-gated rescue) |

Run all 3 configs end-to-end on the standard 800-case bench
(`preset=balanced / fl=832 / cng=True / j=4`). Compute Δs vs C0 for
each bucket. Adjudicate against the 5 hard bars (H1, H2, H3, H4a, H4b)
on the C2-vs-C0 row.

**S2 PASS condition**: all 5 hard bars pass on C2 vs C0.
**S2 FAIL condition**: any of H1, H2, H3 fails → §0.4 kill criterion
fires; H4a/H4b fail with H1+H2+H3 PASS → close per §0.4 with "no harm
no foul" note in post-mortem.

If S2 PASS, write verdict doc `docs/v3_15_arc_m_v3_verdict.md`,
merge `feature/v3.15-arc-m-v3` into `feature/v3.15`, update §1.5b row
in v3.15 plan to PROMOTED. Defer the productionisation flip
(`balanced` preset gets `arc_m_t_gated_enabled=True` + Arc T flags
ON) to the v3.15 closeout sprint per §0.7.

If S2 FAIL, write closure post-mortem `docs/v3_15_arc_m_v3_closure.md`,
keep both flags default-OFF, update §1.5b row in v3.15 plan to
CLOSED CANNOT SHIP, document substrate retention rationale (consumable
by future Arc T-tier discriminators or NN-classifier-gated arcs).

## Files to modify

- `python/aec.py` (≈ +12 lines):
  - 1 config flag `arc_m_t_gated_enabled: bool = False` after line 850.
  - 5 gate-wrap insertions at lines 6177, 6446, 6772, 6862, 6907 (each
    +1 line of `if not (... and ...):` and +1 line of indentation).
- `python/eval_aec_challenge.py` (≈ +4 lines):
  - 1 env override block `AEC_ARC_M_T_GATED_ENABLED` after line 271.
- `docs/v3_15_arc_m_v3_design.md` (this file, new).
- `docs/v3_15_arc_m_v3_verdict.md` (new on S2 PASS) **OR**
  `docs/v3_15_arc_m_v3_closure.md` (new on S2 FAIL per §0.4).

No changes to `python/res_refactored/`, `c_impl/`, or any other
file. C port follows after Python merges to `main` per the standard
Python-first convention.

## Closure protocol per §0.4 (verbatim)

> §0.4 kill criterion: any of the 3 main hard bars FAIL after 1 tuning
> sprint → CLOSE Arc M permanently, V1 substrate stays default-OFF,
> write closure post-mortem documenting which bar broke.

"Permanently" here means: the §1.5b additive flag
`arc_m_t_gated_enabled` joins `arc_m_epc_gated` as default-OFF
research substrate; the Arc M arc closes for v3.15 with no further
sprints. Future v3.16+ retries are out of v3.15 scope and require a
new design lock + re-authorisation.

"1 tuning sprint" budget: a single threshold/hysteresis recalibration
on Arc T parameters that are read by the §1.5b gate
(`arc_t_threshold_hi_db`, `arc_t_threshold_lo_db`,
`arc_t_hysteresis_frames`). NOT in budget: changes to the Q
band-scales themselves (those are V1 substrate, owned by Arc M closure
verdict and not re-opened); changes to the gate expression (e.g.
adding extra AND clauses); changes to Arc T detection mechanism
(those belong on `feature/v3.15-arc-t`).

## v3.15 plan impact

If §1.5b PASSES:
- Arc M reopens with V1 + T-gate as the productisation candidate. v3.15
  plan §1.5b row → PROMOTED, productisation deferred to v3.15 closeout.
- v3.13 E2 Path 3 DT debt closure target advances by ~+0.022 dB
  (≥ 87% of V1's +0.023 dB), bringing total DT debt closure to
  ~44% of −0.050 dB target (vs prior ~0% with all V1+V2 closed).
- Arc G (§1.4 next slot) remains valid; the Arc F/M wall analysis from
  Arc M closure does NOT predict Arc G failure (Arc G mechanism is
  W-reset, orthogonal to Q modification).
- v3.15 production candidate gains a per-frame T-gated branch in the
  filter; review burden +small (5 gate sites + 1 compose flag + Arc T
  substrate already merged).

If §1.5b FAILS:
- Arc M permanently closed; total v3.15 closure burden = Arc F closed
  + Arc M V1+V2+v3 closed + Arc T (S3 separately decides).
- DT debt closure attribution moves to Arc G + Arc T composite (per
  [`docs/v3_15_arc_m_closure.md:81-84`](v3_15_arc_m_closure.md#L81-L84))
  or post-v3.15 RES per-state ENR refactor.
- Arc T's value as a §1.5b gate substrate is invalidated for Arc M
  specifically, but its independent S3 RES preempt arc remains in scope.
- The v3.15 closeout doc records §1.5b as "third strike on Arc M";
  Arc M family permanently retired through v3.15.
