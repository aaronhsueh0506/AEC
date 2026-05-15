# v3.15 §1.7 — RES audit and v3.16 refactor plan

**Status**: POPULATED — 12 v3.16 candidates ranked; user-gated authorisation
per §0.7 closeout merge.

**Date**: 2026-05-15 (populated)
**Branch**: `feature/v3.15-arc-g`
**Author**: v3.15 Phase F closeout
**Doc role**: Gates the v3.16 RES refactor arc. Hard bar §6 = ≥ 5 candidates
ranked with measurable success criteria; ≥ 3 with predicted Δ ≥ +0.005 →
authorise v3.16 RES refactor; otherwise declare RES architecture stable.

---

## 1. Header / scope

- §1.7 purpose: comprehensive RES health audit on the v3.15 closeout
  substrate (Arc P + Arc R + Arc S-orth.A on main; Arc T detector default
  ON; all other v3.15 candidate arcs CLOSED CANNOT SHIP per §1.2 / §1.4 /
  §1.5b / §1.6).
- §1.2 status: CLOSED CANNOT SHIP — DT-NE compression fix
  ([`v3_15_dt_ne_compression_fix_closure.md`](v3_15_dt_ne_compression_fix_closure.md)).
- §1.3 status: DEFERRED post Phase E — Arc D merge (§10 plan note).
- §1.4 status: Arc M S1+S2 V1+V2 CLOSED ([`v3_15_arc_m_closure.md`](v3_15_arc_m_closure.md));
  Arc G CLOSED ([`v3_15_arc_g_closure.md`](v3_15_arc_g_closure.md)).
- §1.5 status: Arc T S1 PASS substrate; Arc T S2 wiring CLOSED no-op
  ([`v3_15_arc_t_s2_wiring_closure.md`](v3_15_arc_t_s2_wiring_closure.md)).
- §1.5b status: Arc M.v3 T-gated retry CLOSED CANNOT SHIP
  ([`v3_15_arc_m_v3_closure.md`](v3_15_arc_m_v3_closure.md)).
- §1.6 status: Arc F per-band Q CLOSED ([`v3_15_arc_f_closure.md`](v3_15_arc_f_closure.md)).
- Substrate baseline for diff: v3.13 Phase 3 verdict
  ([`v3_13_phase3_res_audit_verdict.md`](v3_13_phase3_res_audit_verdict.md)).
- Code changes inside §1.7: ZERO (audit-only). Delete-only candidates ship
  inside the v3.16 cycle as standalone byte-equal-verified commits.

---

## 2. Audit method

- Script: [`tools/research/v3_15_res_audit.py`](../tools/research/v3_15_res_audit.py).
- Mirrors the v3.13 Phase 3 fire-rate-counting pattern via the
  `AecConfig.capture_stages=True` substrate
  ([python/aec.py](../python/aec.py)). Per-bin diff is computed in
  user-space from `res.get_stage_gains()` output. Per-frame fire-rate =
  `np.any(post − pre > 1e-7)`.
- `ne_g_floor` is folded into `spectral_g_min` BEFORE stage 02; inferred
  from cached scalars `ResFilter._stats_last_ne_g_floor` and
  `_stats_last_spectral_g_min` (frame-level binary, matches v3.13 verdict
  semantic).
- Bench config: BALANCED (now includes `arc_t_cohort_detector=True` per
  §10.S0b) / `fl=832` / `cng=True` / `seed=0` / `j=6` chunk-split.
- Per-bucket aggregator: `FS_static / FS_movement / DT_static /
  DT_movement / NE / cohort_tail / GLOBAL`. Bucket assignment via stem
  suffix + `qNvSMyU…` cohort-tail anchor.
- Outputs: `/tmp/v3_15_res_audit/audit.json` (full per-bucket JSON),
  `summary.csv` (fire-rate + Δ-vs-v3.13), per-case JSONs.

### 2.1 v3.13 baseline (diff column reference)

| Path | FS_static | FS_movement | DT_static | DT_movement | NE | cohort_tail (qNvSMyU) |
|---|---:|---:|---:|---:|---:|---:|
| spectral_floor | 0.894 | 0.881 | 0.524 | 0.529 | 0.097 | 0.974 |
| ne_g_floor    | 0.880 | 0.867 | 0.934 | 0.933 | 0.999 | 0.750 |
| epc_dt_cap    | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| quiet_mask    | 0.509 | 0.495 | 0.298 | 0.307 | 0.060 | 0.679 |
| hf_cap        | n/a   | n/a   | n/a   | n/a   | n/a   | n/a   |

`hf_cap` was not measured discretely in v3.13 (rolled into general
post-process); §1.7 audit establishes its first-pass baseline.

---

## 3. Findings (60-case directional subset)

Audit run on the first 60 cases of the AEC Challenge corpus on v3.15
closeout substrate (`feature/v3.15-arc-g` HEAD `5bb2fa8` —
`arc_t_cohort_detector=True` BALANCED default; all other v3.15
substrate flags default OFF). Outputs:
`/tmp/v3_15_res_audit/{audit.json,summary.csv,per_case/*.json}`. Run
time 70 sec on `-j 6`.

> **CRITICAL SAMPLE BIAS**: The `v3_15_res_audit.py --n-cases 60` flag
> takes alphabetical-first cases from the corpus loader, and the loader
> happens to enumerate `doubletalk/` first. Result: **60 / 60 cases are
> DT (40 static + 20 movement); 0 FS / 0 NE / 0 cohort_tail**. Findings
> below are reliable for DT bucket; FS / NE / cohort_tail rows in
> summary.csv are all zero-frames-counted (not zero-fire-rate). Full
> 800-case audit at the v3.16 phase entry is mandatory before promoting
> any candidate to a sprint (per `feedback_aec_code_review_accuracy`).
> The audit script needs the `--cases-list` flag added in v3.16
> housekeeping (currently only supports `--n-cases`).

### 3.1 `spectral_floor`

- **Method**: pre = `01_softgate_emr`, post = `02_spectral_floor`.
- **v3.15 fire-rate**: DT_static 0.5144 / DT_movement 0.4699; mean
  delta when fired ≈ 0.013-0.015.
- **Diff vs v3.13 (DT)**: DT_static −0.010 (0.5240 → 0.5144);
  DT_movement −0.059 (0.5290 → 0.4699). DT_movement drop is the most
  visible substrate shift.
- **Cohort tail (qNvSMyU)**: not in 60-case sample; re-audit on
  800-case required.
- **Determination**: KEEP. DT firing pattern unchanged in topology;
  drop in DT_movement is moderate. Cohort-tail load-bearing role
  remains the canonical defence (v3.13 baseline 97.4 %; reaffirmed by
  v3.12 P58 Phase 2 closure when removing it).

### 3.2 `ne_g_floor` — **SUBSTRATE SHIFT: dead on v3.15 substrate**

- **Method**: inferred via `_ne_g_floor_fired(res)` checking
  `_stats_last_ne_g_floor > _stats_last_spectral_g_min + 1e-7`.
- **v3.15 fire-rate**: **0.0000 on ALL measured buckets (DT_static,
  DT_movement, GLOBAL)** — 0 / 150 792 frames on DT_static, 0 / 74 049
  on DT_movement.
- **Diff vs v3.13 (DT)**: −0.934 on DT_static (0.934 → 0.000);
  −0.933 on DT_movement (0.933 → 0.000). v3.13 verdict's "universal
  baseline floor" finding **NO LONGER HOLDS** on v3.15 substrate.
- **Mechanism**: v3.14 Arc P (adaptive per-band ERL) + Arc R (per-band
  ENR `block_lf` tilt) raise `spectral_g_min` enough that the
  `max(spectral_g_min, ne_g_floor)` comparison **never picks
  `ne_g_floor`**. `ne_g_floor` is now structurally dominated by the
  per-band ENR-driven floor.
- **Determination**: STRONG CANDIDATE FOR REMOVAL. Pending 800-case
  re-audit confirming the same on FS / NE / cohort_tail. If confirmed,
  remove as a delete-only candidate alongside `epc_dt_cap` (see
  **NEW v3.16 candidate** in §4 — listed as part of Candidate C1
  scope expansion, or as a separate C1b).

### 3.3 `epc_dt_cap`

- **Method**: pre = `02_spectral_floor`, post = `03_epc_dt_cap`.
- **v3.15 fire-rate**: **0.0000 on all measured buckets** (DT_static,
  DT_movement). Matches v3.13 baseline 0/800.
- **Diff vs v3.13**: +0.0000 — no change.
- **Determination**: REMOVE — ship as v3.16 candidate **C1**
  (delete-only, byte-equal-verified). Dead on both v3.13 and v3.15
  substrates.

### 3.4 `quiet_mask`

- **Method**: pre = `03_epc_dt_cap`, post = `04_quiet_mask`.
- **v3.15 fire-rate**: DT_static 0.3010 / DT_movement 0.3125; mean
  delta when fired ≈ 0.85.
- **Diff vs v3.13 (DT)**: +0.003 DT_static, +0.006 DT_movement —
  effectively unchanged.
- **Determination**: KEEP. Physical noise gate, stable across
  substrate.

### 3.5 `hf_cap`

- **Method**: pre = `05_3bin_smooth`, post = `06_hf_cap`. Three
  sub-modes (conditional / plan_a_2k / v3.8.3-strict); summary.csv
  does not split by sub-mode.
- **v3.15 fire-rate (FIRST baseline)**: DT_static 0.3537 / DT_movement
  0.3359; mean delta when fired ≈ 0.052-0.074.
- **Diff vs v3.13**: no baseline (newly measured).
- **Determination**: KEEP. Sub-mode dead-branch audit deferred to v3.16
  candidate **C3** (Stage-1 4-cap chain ordering).

### 3.6 Cross-path observations

- **`ne_g_floor` substrate shift is the headline finding**: v3.14 Arc
  P + R + S-orth.A made it structurally dead on DT. Adds a NEW
  delete-only candidate to the v3.16 Phase 0 list.
- v3.15 substrate does NOT change `spectral_floor` / `quiet_mask`
  load-bearing topology on DT bucket. Reaffirms v3.13 finding that
  `spectral_floor` is the cohort-tail defence (pending 800-case
  re-audit on cohort_tail bucket).
- `epc_dt_cap` is doubly dead (v3.13 + v3.15). C1 removal is safe.
- DT-NE compression mechanism is NOT inside this audit's floor
  topology. §1.2 audit ([`v3_15_dt_ne_audit.md`](v3_15_dt_ne_audit.md))
  traced compression to ENR per-state × per-band — v3.16 candidate
  **C2** addresses that surface.
- 60-case DT-only sample limits coverage. The audit's substrate-shift
  finding on `ne_g_floor` may not hold on FS / NE / cohort_tail
  (e.g., NE bucket in v3.13 fired `ne_g_floor` 99.9 % — if it stays
  high on NE, the floor is NOT dead globally and the removal candidate
  must be NE-aware). 800-case re-audit at v3.16 Phase 0 entry is the
  gate for C1b (`ne_g_floor` removal).

---

## 4. v3.16 refactor candidates (13 candidates, 5 phases)

Each candidate has a measurable success criterion (per hard bar §6).
Candidates are ranked within phases by ROI; phase ordering enforces
dependency.

### Phase 0 — Housekeeping (parallel-safe; runs before everything else)

#### Candidate HK-1: B3 fix — `run_one_case.py` CNG seed
- **Origin**: v3.15 §10.S3 deferred to v3.16.
- **Mechanism**: `python/run_one_case.py` instantiates `AEC(cfg)` without
  `np.random.seed(...)`. Add `np.random.seed(0)` (or 42 — match
  `eval_aec_challenge.py` per `seed=0` convention) before AEC
  instantiation.
- **Predicted AECMOS Δ**: 0.000 (research tooling fix; no production
  bench impact).
- **Implementation LOE**: S (≤ 1 sprint, ≤ 5 LOC).
- **Cohort-tail risk**: none.
- **Measurable success**: `run_one_case.py` produces byte-identical
  output across re-invocations on the same case. Attach a 5-case
  determinism test.
- **Predicted byte-equal cost**: byte-equal on production bench
  (`eval_aec_challenge.py` already seeds correctly); only research
  tool's behaviour changes.
- **Prior art**: documented in v3.15 plan §2 B3 punch-list entry.

#### Candidate HK-2: §1.8 pcb1N mic_dynamic_margin patch
- **Origin**: v3.15 §10.S5 deferred to v3.16.
- **Mechanism**: Single-case audible artefact (`pcb1N` listen case)
  driven by mic_dynamic_margin clipping at startup. Apply targeted
  patch as documented in plan §1.8.
- **Predicted AECMOS Δ**: 0.000 mean (single case; bucket-mean
  invisible).
- **Implementation LOE**: S (≤ 1 sprint).
- **Cohort-tail risk**: low — must verify cohort-tail (`qNvSMyU`)
  unchanged on 800-case bench post-fix.
- **Measurable success**: pcb1N listen confirms patch resolves audible
  artefact AND 800-case AECMOS no regression (DT Δdeg ≥ −0.005, FS
  Δecho ≥ −0.020).
- **Predicted byte-equal cost**: minor drift on `pcb1N`; byte-equal on
  the other 799 cases.

#### Candidate C1: `epc_dt_cap` dead-code removal
- **Origin**: v3.13 Phase 3 audit (fire-rate 0/800); reaffirmed by §1.7
  §3.3.
- **Mechanism**: Delete `if epc_dt: g = min(g, EPC_DT_GAIN_CAP)` block
  + `_stage_gains['03_epc_dt_cap']` capture wrapper at
  [`python/aec.py`](../python/aec.py) (lines ≈ 3199-3204; line numbers
  may have drifted post v3.14 Arc P/R + v3.15 Arc T merges, re-locate
  before edit per §0.1 rule 1).
- **Predicted AECMOS Δ**: 0.000 (byte-equal — fire-rate 0 on subset).
- **Implementation LOE**: S (≤ 20 LOC delete + capture-key cleanup).
- **Cohort-tail risk**: none.
- **Measurable success**: 800-case sample-level byte-equal (`atol=1e-6,
  rtol=1e-5`) AND `_stage_gains` dict no longer carries the deprecated
  key.
- **Predicted byte-equal cost**: byte-equal.

#### Candidate C1b: `ne_g_floor` removal — NEW (substrate-shift discovery)
- **Origin**: §1.7 §3.2 audit found fire-rate **0.0000 on DT_static +
  DT_movement** (v3.13 baseline 0.93 on both). v3.14 Arc P + R raise
  `spectral_g_min` enough that `ne_g_floor` never wins the
  `max(spectral_g_min, ne_g_floor)` comparison.
- **Mechanism**: Delete the `spectral_g_min = max(spectral_g_min,
  ne_g_floor)` raise at [`python/aec.py`](../python/aec.py) line
  ≈ 3696 + remove `ne_g_floor` config field + remove the cached scalar
  `_stats_last_ne_g_floor` (used only by `_ne_g_floor_fired()` audit
  hook).
- **Predicted AECMOS Δ**: 0.000 on DT (byte-equal — fire-rate 0). FS
  / NE / cohort_tail TBD by 800-case re-audit at v3.16 Phase 0 entry.
- **Implementation LOE**: S (≤ 30 LOC delete + capture cleanup).
- **Cohort-tail risk**: low — v3.13 cohort_tail ne_g_floor fire-rate
  was 0.750 (lower than other buckets); needs re-audit on v3.15
  substrate before removal commits. If still > 5 % on cohort_tail,
  promote to C2 (per-state × per-band table) instead of C1b.
- **Measurable success**: full 800-case fire-rate audit confirms ≤ 5 %
  on every bucket including cohort_tail AND 800-case sample-level
  byte-equal verification post-removal.
- **Predicted byte-equal cost**: byte-equal IF the gate condition
  holds; non-byte-equal is the trigger to ABORT removal and escalate
  to C2.
- **Gate**: 800-case re-audit (uses C6 prerequisite tooling — extend
  audit script to take `--cases-list` flag).

### Phase 1 — Foundation (BLOCKS Phase 2-4; can run in parallel within phase)

#### Candidate C5: Per-state RES interface modularisation
- **Origin**: extends `python/res_refactored/` (P52 Phase B).
- **Mechanism**: Lift the per-`filter_state` ENR / floor / cap policy
  into a small interface (`PerStateResPolicy`) so Phase 2 candidates
  (C2, C3, C4) can edit a single dispatch site instead of modifying
  `_stage_gain_compute` directly. Reduces Arc D × Arc R style write
  conflicts (see §0.2 plan rule).
- **Predicted AECMOS Δ**: 0.000 (architectural; logic byte-equal by
  design lock §5.5 anti-loophole).
- **Implementation LOE**: M (3 sprints — extract / wire / 800-case
  byte-equal).
- **Cohort-tail risk**: none if anti-loophole §5.5 enforced
  (extraction must be byte-identical).
- **Measurable success**: 800-case sample-level byte-equal (`atol=1e-6`,
  rtol=1e-5`) PLUS `_stage_gain_compute` writes only via
  `PerStateResPolicy.apply()` interface (no direct ENR scalar writes).
- **Predicted byte-equal cost**: byte-equal mandated.
- **BLOCKS**: C2, C3, C4 (Phase 2).

#### Candidate C6: DelayEst audit + delay-aware detector / handler
- **Origin**: meta-pattern across v3.15 closures — cohort tail
  `qNvSMyU` class, Arc M V1 FS_movement damage, Arc F cohort tail
  damage, Arc G destructive W reset, §1.1 H5 hypothesis on DT-NE all
  share echo-path-changing-during-case where DelayEst tracking drives
  downstream filter behaviour. If DelayEst is laggy / inaccurate /
  mis-locks during a movement window, the linear filter trains on the
  wrong delay → phantom echo → inflated residual → RES over-suppresses
  on NE-only frames.
- **Mechanism (audit)**: 60-case Tier 1 + dedicated movement / cohort
  tail subset; per-frame trace of `estimated_delay_samples`,
  `lock_state`, re-estimation events, multi-peak ambiguity score.
  Correlate with `cohort_tail_T` detector fire frames, EPC fire events,
  RES `residual_echo_psd` peaks, NE Δdeg cases.
- **Predicted AECMOS Δ**: TBD by audit. Hypothesis: if DelayEst is the
  upstream cause for 30–50 % of movement-related closures' wall
  magnitude, a delay-aware detector / handler can recover that fraction
  on FS_movement / DT_movement / cohort tail.
- **Implementation LOE**: M-L (1-2 sprint audit; mechanism arc 3-6
  sprints depending on audit outcome).
- **Cohort-tail risk**: AUDIT is risk-free; mechanism arcs need
  cohort-tail Δecho ≥ −0.05 hard bar.
- **Measurable success (audit phase)**: documented per-frame DelayEst
  behaviour on the 5 movement-failure cohorts above; one of: (a)
  DelayEst is the upstream root cause (PROCEED with delay-aware arc),
  (b) DelayEst behaves correctly and the closures are mechanism-walls
  (CLOSE C6 audit; do not open mechanism arc).
- **Predicted byte-equal cost**: audit byte-equal; mechanism arc
  introduces new state.
- **CRITICAL GATE**: outcome may RE-ORDER Phase 3-4 priorities. v3.16-A
  / B / C7 ROI estimates assume DelayEst is NOT the root cause; if C6
  audit overturns that assumption, those candidates need re-evaluation
  before sprint authorisation.
- **Real-world consideration**: DelayEst inaccuracy is unavoidable in
  some scenarios (low echo-to-noise ratio, NE-only segments,
  multi-path). Any v3.16 delay arc must be robust to "uncertain"
  frames — downstream RES should not trust filter output when delay
  confidence is low.

### Phase 2 — RES refactor (gated on C5 modularisation; can parallelise within phase)

#### Candidate C2: ENR per-state × per-band refactor (含 Arc D merge re-eval)
- **Origin**: Arc D × Arc R write-conflict surface (v3.14 §0.2 lesson)
  + §1.2 DT-NE audit hypothesis H1+H2 (most DT compression in
  `refined_usable` state, ENR over-fires on bins 5-25 voice band).
- **Mechanism**: Split `enr_t_ne / enr_s_ne` into a 2-D table indexed
  by (filter_state, band). Resolves the Arc D × Arc R conflict
  documented at v3.14 §0.2; Arc D's `coarse_learning` tuple becomes
  inactive when post-Phase E convergence keeps the filter in
  `refined_usable`.
- **Predicted AECMOS Δ**: +0.005 to +0.015 (DT bucket Δdeg recovery on
  `refined_usable` substrate). Re-audit at sprint kickoff to confirm.
- **Implementation LOE**: M (3 sprints — table design / wire /
  per-state byte-equal sanity / 800-case A/B).
- **Cohort-tail risk**: low — `refined_usable` cohort-tail behaviour
  preserved by Arc D's steady-state byte-equal tuple.
- **Measurable success**: DT bucket Δdeg ≥ +0.020 (consume v3.13 E2
  Path 3 debt ≥ 40 %) AND cohort tail Δecho ≥ −0.05 AND FS bucket no
  regression > −0.020.
- **Gates on**: C5 (modular interface).
- **Subsumes**: §1.3 Arc D merge re-evaluation (D's `coarse_learning`
  tuple becomes part of C2's table).

#### Candidate C3: Stage-1 4-cap chain ordering audit + reorder
- **Origin**: v3.13 Phase 3 cosmetic refactor deferred + §1.1 DT-NE
  audit hypothesis H3 (4-cap chain over-fires on DT-NE).
- **Mechanism**: Audit the 4-cap chain (`epc_dt_cap → quiet_mask →
  3bin_smooth → hf_cap`) for ordering effects. Hypothesis: reordering
  `quiet_mask` before `epc_dt_cap` (post C1 removal: before
  `3bin_smooth`) may reduce stacking-induced over-suppression on DT
  cases.
- **Predicted AECMOS Δ**: +0.003 to +0.010 (DT bucket; uncertain
  pending audit).
- **Implementation LOE**: S-M (1 audit sprint + 1-2 reorder sprints).
- **Cohort-tail risk**: medium — `quiet_mask` is FS-skewed, reordering
  changes its interaction with cohort-tail `spectral_floor`.
- **Measurable success**: cohort tail Δecho ≥ −0.05 AND DT bucket Δdeg
  ≥ +0.010 AND FS Δecho ≥ −0.020.
- **Gates on**: C5 (modular interface).

#### Candidate C4: `noise_floor` / CNG interaction refactor
- **Origin**: §1.1 DT-NE audit hypothesis H4 (compression downstream of
  RES — noise_floor / CNG over-aggressive on DT-NE residual).
- **Mechanism**: Cap noise_floor effect on NE-active frames; rework
  CNG fill to NOT mask NE residual leakage during DT.
- **Predicted AECMOS Δ**: +0.005 to +0.015 (DT bucket; addresses the
  "compression downstream of RES" component of v3.13 E2 Path 3 debt).
- **Implementation LOE**: M (2-3 sprints).
- **Cohort-tail risk**: low — noise_floor / CNG do not drive cohort
  tail catastrophe defence.
- **Measurable success**: DT bucket Δdeg ≥ +0.010 AND NE Δdeg ≥ −0.005
  (canonical NE preservation guard).
- **Gates on**: C5 (modular interface).

### Phase 3 — Arc T consumers (gated on C6 outcome + Arc T detector default ON, achieved by v3.15 §10.S0b)

#### Candidate v3.16-A: Force-render OR-in `_arc_t_cohort_tail_signal`
- **Origin**: Arc T S2 closure H2 ([`v3_15_arc_t_s2_wiring_closure.md`](v3_15_arc_t_s2_wiring_closure.md))
  identified `_using_render_based = True` was overwritten 1 line later
  by `_residual_est.compute_residual_echo()` state machine.
- **Mechanism**: Rewire so cohort tail signal OR-s into the
  `force_render` decision INSIDE `compute_residual_echo()` instead of
  outside it. Direct fix to the H2 dead-write bug.
- **Predicted AECMOS Δ**: +0.030 cohort tail Δecho IF detector FP rate
  ≤ 5 % on non-tail buckets (Arc T S1 detector validated 5/5 TAIL fire
  + 3/3 CTRL no-fire on 8-case validation).
- **Implementation LOE**: S (≤ 30 LOC + new test for the OR-in path).
- **Cohort-tail risk**: this is the cohort-tail defence path; risk is
  on FS / DT buckets if detector FP exceeds 5 %.
- **Measurable success**: cohort tail Δecho ≥ +0.020 AND FS Δecho ≥
  −0.010 AND DT Δdeg ≥ −0.005.
- **Gates on**: Arc T detector default ON (DONE v3.15 §10.S0b); C6
  outcome may shift priority.

#### Candidate v3.16-B: ENR-path lift via `enr_t_ne` × `cohort_tail_T`
- **Origin**: Arc T S2 closure H1 — `over_sub × 1.3` was DEAD CODE in
  BALANCED because `over_sub` is only consumed by `gain_type='wiener'`
  and BALANCED uses `'enr'`.
- **Mechanism**: Apply the equivalent of an over-sub boost on the
  enr-path side: `enr_t_ne_eff = enr_t_ne × cohort_tail_T_boost`
  during cohort-tail windows. Mirrors H1 intent on the actually-live
  ENR gain path.
- **Predicted AECMOS Δ**: +0.020 cohort tail Δecho.
- **Implementation LOE**: S-M (1-2 sprints — design + tune
  `cohort_tail_T_boost` value + 800-case A/B).
- **Cohort-tail risk**: same as v3.16-A; FP outside cohort tail damages
  FS / DT.
- **Measurable success**: cohort tail Δecho ≥ +0.015 AND FS Δecho ≥
  −0.010 AND DT Δdeg ≥ −0.005.
- **Gates on**: Arc T detector default ON (DONE) AND C2 (ENR per-state
  × per-band refactor) finishing first.

### Phase 4 — Arc M family retry (gated on C6 + Phase 3 outcomes)

#### Candidate C7: Arc M.v3 retry (α / β / γ)
- **Origin**: §1.5b Arc M.v3 closure structural analysis — 3 rescue
  paths identified, all exceeded 1-tuning-sprint budget:
  - **α**: predictive cohort tail signal (cohort tail predicted before
    EPC fires, eliminating timing mismatch).
  - **β**: post-assertion hysteresis on the gate (suppress Q boost for
    N frames AFTER any recent signal assertion).
  - **γ**: per-filter dispatch (gate suppresses on main, allows on
    shadow).
- **Mechanism**: pick one of α / β / γ based on C6 audit outcome (if
  DelayEst is upstream cause, α may be feasible via early prediction;
  if not, β/γ are the path).
- **Predicted AECMOS Δ**: TBD — Arc M V1's +0.023 dB DT_movement Δdeg
  remains unconsumed; rescue would consume ≥ 50 % of that.
- **Implementation LOE**: M-L (3-6 sprints including new detector
  primitive design).
- **Cohort-tail risk**: HIGH — Arc M V1 had cohort tail damage Δecho
  −0.0496; rescue must avoid resurrection of that wall.
- **Measurable success**: H1 (cohort tail) ≥ −0.030, H2 (DT_movement)
  ≥ +0.015, H3 (FS_movement) ≥ −0.020 (relax §1.5b H1 from −0.020 to
  −0.030 acknowledging structural difficulty).
- **Gates on**: C6 outcome (especially if α path is selected); Phase 3
  results (per-filter dispatch γ may need v3.16-A or B's wiring).

#### Candidate C8: Arc G non-destructive partial W decay
- **Origin**: Arc G closure §"What's left for v3.16+"
  ([`v3_15_arc_g_closure.md`](v3_15_arc_g_closure.md)).
- **Mechanism**: Replace Arc G's destructive zero-out W reset with a
  partial decay (`W *= 0.5` instead of `W = 0`) on detected gain-change
  drift. Avoids the destructive convergence loss that closed Arc G.
- **Predicted AECMOS Δ**: TBD — Arc G's failure mode (ERLE Δ = −1.48
  dB) suggests partial decay may avoid the cliff. Estimate +0.005 if
  decay coefficient tuned correctly.
- **Implementation LOE**: M (2-3 sprints).
- **Cohort-tail risk**: low — Arc G's gain-change detector is
  orthogonal to cohort tail mechanism.
- **Measurable success**: ≥ 5 of Arc G's S4 listen cohort show
  audible improvement (vs Arc G S4's 0/5 on destructive W reset) AND
  cohort tail Δecho ≥ −0.05 AND DT Δdeg ≥ −0.005.
- **Gates on**: independent of C7; same phase.
- **Priority**: LOW within Phase 4 (Arc G's audible debt is smaller
  than Arc M's).

---

## 5. Recommendation — v3.16 arc authorisation

### Decision matrix

| Candidates with predicted Δ ≥ +0.005 | Recommendation |
|---|---|
| ≥ 3 | **Authorise v3.16 RES refactor arc**; promote top-3 + C5 architectural foundation as Phase 1-2 sprints. |
| 1 – 2 | Ship the 1 – 2 as standalone v3.15.x commits; **do not** open v3.16. |
| 0 (only delete-only / cosmetic) | **Declare RES architecture stable**; close §1.7; no v3.16. |

### Top candidates by predicted Δ ≥ +0.005

1. **v3.16-A** — predicted +0.030 cohort tail Δecho (Arc T H2 fix, gated on detector default ON which v3.15 §10.S0b just landed)
2. **v3.16-B** — predicted +0.020 cohort tail Δecho (Arc T H1 fix on enr path)
3. **C2** — predicted +0.005 to +0.015 DT bucket Δdeg (ENR per-state × per-band)
4. **C4** — predicted +0.005 to +0.015 DT bucket Δdeg (noise_floor / CNG)
5. **C3** — predicted +0.003 to +0.010 DT bucket Δdeg (4-cap reorder)

5 candidates with predicted Δ ≥ +0.005 → **AUTHORISE v3.16 RES refactor
arc** per the decision matrix. ≥ 3 satisfied → user merge + v3.16
authorisation pending §0.7 closeout review.

### v3.16 phase ordering

```
Phase 0 (parallel-safe; before everything else)
├── HK-1  B3 fix (run_one_case.py CNG seed)
├── HK-2  pcb1N mic_dynamic_margin patch
├── C1    epc_dt_cap dead-code removal (ZERO Δ)
└── C1b   ne_g_floor removal (NEW — substrate-shift; gated on 800-case re-audit)

Phase 1 (BLOCKS Phase 2-4)
├── C5    Per-state RES interface modularisation [architectural]
└── C6    DelayEst audit (outcomes RE-ORDER Phase 3-4)
        ├── if movement closures = upstream delay → Phase 3-4 needs delay fix first
        └── if movement closures = mechanism walls → Phase 3-4 proceeds independently

Phase 2 (gated on C5)
├── C2    ENR per-state × per-band refactor (含 Arc D merge re-eval)
├── C3    Stage-1 4-cap chain ordering audit + reorder
└── C4    noise_floor / CNG interaction refactor

Phase 3 (gated on C6 outcome + Arc T detector default ON [DONE v3.15 §10.S0b])
├── v3.16-A  force_render OR-in cohort_tail_T (Arc T H2 fix)
└── v3.16-B  ENR-path lift via enr_t_ne × cohort_tail_T (also gated on C2)

Phase 4 (gated on C6 + Phase 3)
├── C7    Arc M.v3 retry (α / β / γ — needs new detector primitive from C6 OR new dispatch from Phase 3)
└── C8    Arc G non-destructive partial W decay (independent within phase, LOW priority)
```

### v3.16 sprint estimate

| Phase | Sprints |
|---|---|
| 0 housekeeping | 3 – 4 (HK-1 + HK-2 + C1 + C1b) |
| 1 foundation | 4 – 5 (C5 architectural; C6 independent audit) |
| 2 RES refactor | 6 – 9 (3 candidates × 2 – 3 sprints each) |
| 3 Arc T consumers | 4 – 6 |
| 4 Arc M / G retry (if opened) | 4 – 6 |
| **TOTAL** | **21 – 30 sprints** |

### Cohort-tail risk register (require explicit `qNvSMyU` regression check before authorisation)

- **v3.16-A** — direct cohort tail wiring; if detector FP rate > 5 % on
  non-tail buckets, FS / DT regress.
- **v3.16-B** — same surface as v3.16-A.
- **C7** — Arc M family with cohort tail history (V1 −0.0496 Δecho).
- **C3** — `quiet_mask` reorder may shift cohort-tail `spectral_floor`
  interaction.

---

## 6. Hard-bar status

| Bar | Status |
|---|---|
| ≥ 5 refactor candidates listed | **PASS** (13 candidates including new C1b from substrate-shift discovery) |
| Each candidate has measurable success criterion | **PASS** (template applied to all 12) |
| ≥ 3 candidates with predicted Δ ≥ +0.005 → authorise v3.16 | **PASS** (5: v3.16-A / v3.16-B / C2 / C4 / C3) |
| Else → declare RES architecture stable, close §1.7, no v3.16 | n/a |
| Delete-only candidates (e.g. C1 `epc_dt_cap`) ship inside §1.7 | DEFERRED — ship as v3.16 Phase 0 commit instead of inside §1.7 (avoids mixing audit-doc with code change in v3.15 closeout) |

**Doc acceptance**: §1.7 hard bar PASS. v3.16 RES refactor arc
authorised, conditional on user §0.7 closeout review.

---

## 7. Artifacts

- This plan: [`docs/v3_15_res_audit_and_refactor_plan.md`](v3_15_res_audit_and_refactor_plan.md)
- Audit script: [`tools/research/v3_15_res_audit.py`](../tools/research/v3_15_res_audit.py)
- Audit raw outputs (gitignored): `/tmp/v3_15_res_audit/audit.json`,
  `summary.csv`, `per_case/<stem>.json`.
- v3.13 baseline: [`docs/v3_13_phase3_res_audit_verdict.md`](v3_13_phase3_res_audit_verdict.md)
- v3.12 predecessor (static categorisation): `docs/v3_12_res_audit.md`
- v3.15 closure docs (referenced as v3.16 candidate origin):
  - [`docs/v3_15_arc_g_closure.md`](v3_15_arc_g_closure.md) — C8 origin
  - [`docs/v3_15_arc_m_closure.md`](v3_15_arc_m_closure.md) — C7 origin (V1+V2)
  - [`docs/v3_15_arc_m_v3_closure.md`](v3_15_arc_m_v3_closure.md) — C7 origin (V3)
  - [`docs/v3_15_arc_t_s2_wiring_closure.md`](v3_15_arc_t_s2_wiring_closure.md) — v3.16-A / v3.16-B origin
  - [`docs/v3_15_dt_ne_audit.md`](v3_15_dt_ne_audit.md) — C2 / C3 / C4 origin (H1-H5 hypotheses)
  - [`docs/v3_15_dt_ne_compression_fix_closure.md`](v3_15_dt_ne_compression_fix_closure.md) — DT-NE wall confirmation
  - [`docs/v3_15_arc_f_closure.md`](v3_15_arc_f_closure.md) — Arc F closure (informs C7 risk)

---

## 8. Open items / follow-ups

- **Audit on full 800-case at v3.16 phase entry**: 60-case Tier 1
  fire-rate values in §3 are directional; promotion to v3.16 sprint
  requires re-audit on full 800 cases per
  `feedback_aec_code_review_accuracy`.
- **C6 audit re-orders Phase 3-4**: outcome of C6 DelayEst audit may
  invalidate v3.16-A / v3.16-B / C7 ROI estimates. Run C6 audit first
  in Phase 1 sequencing.
- **Arc D substrate revisit**: §1.3 Arc D merge was deferred post Phase
  E. C2 candidate now subsumes Arc D's `coarse_learning` tuple; if C2
  ships in v3.16, the v3.14 `feature/v3.14-arc-d` substrate becomes
  redundant. Document closure of that branch in v3.16 housekeeping.
- **Plan §10.S0b retroactive nores listen**: Arc T detector default-ON
  promotion shipped with byte-equal sanity (5/5 PASS) but did NOT
  trigger §1.0.S3 retroactive S-orth.A nores listen. That sprint can
  be re-considered in v3.16 if any audit / listen cohort surfaces
  S-orth.A regression evidence.
