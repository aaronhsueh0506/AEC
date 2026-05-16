# v3.15 §1.7 — RES audit and v3.16 refactor plan

**Status**: POPULATED — 15 v3.16 candidates ranked (HK-1 / C1 / C1c
DONE on `feature/v3.16` Phase 0; HK-2 reframed; C1b reclassified;
remaining 10 pending sprint authorisation); user-gated authorisation
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

## 3. Findings (800-case post-C1c, 2026-05-15)

Audit run on full 800-case AEC Challenge corpus on `feature/v3.16`
HEAD `37a46b2` (post-C1c audit-script fix +
`arc_t_cohort_detector=True` BALANCED default; all other v3.15
substrate flags default OFF). Outputs:
`/tmp/v3_16_audit_full/{audit.json,summary.csv,per_case/*.json}`.
Run time 641 sec (10.7 min) on `-j 6` chunk-split.

### 3.1 `spectral_floor`

- **Method**: pre = `01_softgate_emr`, post = `02_spectral_floor`.
- **v3.15 fire-rate (800-case)**: FS_static 0.900 / FS_movement
  0.870 / DT_static 0.525 / DT_movement 0.518 / NE 0.093 /
  cohort_tail 0.974 / GLOBAL 0.598; mean delta when fired
  ≈ 0.011-0.032.
- **Diff vs v3.13**: FS_static +0.006 / FS_movement −0.011 /
  DT_static +0.001 / DT_movement −0.011 / NE −0.004 / cohort_tail
  −0.000 — all within ±0.015, **substrate stable**.
- **Determination**: KEEP. Cohort-tail load-bearing role
  reaffirmed (0.974 fire-rate, identical to v3.13 baseline; v3.12
  P58 Phase 2 confirmed criticality when attempting removal).

### 3.2 `ne_g_floor` — RECOVERED post-C1c [2026-05-15]

- **Original audit bug** (now fixed): `_stats_last_spectral_g_min` was
  written at [`python/aec.py:3742`](../python/aec.py#L3742) **AFTER**
  the `spectral_g_min = np.maximum(spectral_g_min, ne_g_floor)` raise.
  The audit read the POST-max value → `mean(post_max) >= mean(ne_g_floor)`
  by construction → comparison was mathematically False in production
  cases → reported 0 fires regardless of actual activity.
- **C1c fix shipped commit `37a46b2`** — un-gated pre-max diag fields
  added to `ResFilter`, audit script consumes per-bin any-fire flag
  (`_stats_ne_g_floor_any_bin_fired`).
- **Recovered fire-rates** (800-case full corpus, post-fix):

| Bucket | C1c fire-rate | v3.13 baseline | Diff |
|---|---:|---:|---:|
| FS_static | 0.867 | 0.880 | −0.013 |
| FS_movement | 0.856 | 0.867 | −0.011 |
| DT_static | 0.930 | 0.934 | −0.004 |
| DT_movement | 0.924 | 0.933 | −0.009 |
| NE | 0.999 | 0.999 | +0.000 |
| cohort_tail (qNvSMyU) | 0.764 | 0.750 | +0.014 |
| GLOBAL | 0.913 | n/a | — |

  Substrate drift ≤ ±0.014 vs v3.13 baseline across all buckets;
  mechanism is **load-bearing on every bucket** — universal NE-floor
  on NE (0.999 fire-rate ≡ always-fire as expected) and ~90 % on FS
  / DT. Cohort tail slightly UP (+0.014) post v3.14 substrate.
- **Empirical falsification of removal hypothesis**: full removal
  patch on `feature/v3.16` (deleted `ne_evidence`, `ne_protection`,
  `ne_g_min_ceil`, `ne_g_floor`, `np.maximum` raise) broke 4/5
  byte-equal sanity (`_ours.wav`). 5/5 PASS on `_ours_nores.wav`
  confirms the divergence is RES-side, not linear-filter side.
- **Substrate**: removal patch preserved at `git stash@{0}` titled
  "v3.16 C1b ne_g_floor full removal — REVERTED: audit script
  measurement bug …".
- **Determination**: KEEP. ne_g_floor is **load-bearing universal
  NE-protection floor**. C1b reclassified to retain (see §4 C1b
  candidate); per-band refactor folded into C2 scope (Phase 2).

### 3.3 `epc_dt_cap`

- **Method**: pre = `02_spectral_floor`, post = `03_epc_dt_cap`.
- **v3.15 fire-rate (800-case re-audit)**: **0.0000 on ALL buckets**
  including `cohort_tail 0/2686` — 0 / 2 032 022 frames globally.
  Matches v3.13 baseline 0/800.
- **Diff vs v3.13**: +0.0000 — no change.
- **Determination**: REMOVE — ship as v3.16 candidate **C1**
  (delete-only, byte-equal-verified). Dead on both v3.13 and v3.15
  substrates. **5/5 byte-equal sanity PASS** on (FS_static, FS_movement,
  DT_static, NE, pcb1Nh0Z) for both `_ours.wav` and `_ours_nores.wav`.

### 3.4 `quiet_mask`

- **Method**: pre = `03_epc_dt_cap`, post = `04_quiet_mask`.
- **v3.15 fire-rate (800-case)**: FS_static 0.524 / FS_movement
  0.542 / DT_static 0.311 / DT_movement 0.341 / NE 0.060 /
  cohort_tail 0.627 / GLOBAL 0.364; mean delta when fired ≈ 0.86.
- **Diff vs v3.13**: FS_static +0.015 / FS_movement +0.047 /
  DT_static +0.013 / DT_movement +0.034 / NE +0.000 / cohort_tail
  **−0.052** (0.679 → 0.627). Cohort-tail drop is the **most
  visible substrate shift in this audit** — quiet_mask defends
  cohort tail less aggressively post v3.14 substrate.
- **Determination**: KEEP. Stable on FS / DT / NE; cohort_tail
  drop is mechanism-orthogonal (Arc T detector now provides cohort
  tail signal independently). Substrate shift noted for **C3**
  candidate (4-cap reorder needs to evaluate quiet_mask /
  spectral_floor interaction on cohort tail).

### 3.5 `hf_cap`

- **Method**: pre = `05_3bin_smooth`, post = `06_hf_cap`. Three
  sub-modes (conditional / plan_a_2k / v3.8.3-strict); summary.csv
  does not split by sub-mode.
- **v3.15 fire-rate (800-case, FIRST baseline)**: FS_static 0.602 /
  FS_movement 0.613 / DT_static 0.361 / DT_movement 0.374 / NE
  0.019 / cohort_tail 0.649 / GLOBAL 0.409; mean delta when fired
  ≈ 0.034-0.118 (NE Δ highest 0.118).
- **Diff vs v3.13**: no baseline (newly measured 800-case).
- **Determination**: KEEP. **Load-bearing on cohort_tail (0.649)**
  — FS bias intuitive (HF echo unmask). Sub-mode dead-branch audit
  deferred to v3.16 candidate **C3** (Stage-1 4-cap chain
  ordering).

### 3.6 Cross-path observations (post-C1c, 800-case)

- **`ne_g_floor` reclassified KEEP** (was C1b removal candidate).
  Real fire-rate 0.913 GLOBAL across all buckets, ≤ ±0.014 drift vs
  v3.13. NE bucket fires 0.999 (universal floor as designed).
  Per-band refactor folded into C2 scope.
- **`spectral_floor` cohort-tail load-bearing reaffirmed** (cohort
  tail 0.974, identical to v3.13).
- **`quiet_mask` cohort-tail substrate shift detected** (−0.052 from
  v3.13 0.679 → 0.627). Other buckets stable. Note for C3 candidate
  evaluation.
- **`hf_cap` first 800-case baseline established** — load-bearing on
  cohort_tail (0.649) and FS (0.602+); informs C3 ordering audit.
- **`epc_dt_cap` doubly dead** (v3.13 + v3.15 = 0/2,032,022). C1
  removal safe and shipped (commit `d90efdc`).
- **DT-NE compression mechanism** is NOT inside this audit's floor
  topology. §1.2 audit ([`v3_15_dt_ne_audit.md`](v3_15_dt_ne_audit.md))
  traced compression to ENR per-state × per-band — v3.16 candidate
  **C2** addresses that surface.

---

## 4. v3.16 refactor candidates (15 candidates, 5 phases)

Each candidate has a measurable success criterion (per hard bar §6).
Candidates are ranked within phases by ROI; phase ordering enforces
dependency.

### Phase 0 — Housekeeping (parallel-safe; runs before everything else)

#### Candidate HK-1: B3 fix — `run_one_case.py` CNG seed [DONE]
- **Status (2026-05-15)**: SHIPPED in v3.15 Phase D commit `ea8d320`
  (alongside B6 `_prev_filter_state` doc). v3.15 §10.S3 sprint marker
  superseded — fix actually landed during v3.15 housekeeping window
  before v3.16 plan was authored. v3.16 Phase 0 inherits no further
  work for HK-1.
- **Implementation**: [`python/run_one_case.py:90`](../python/run_one_case.py#L90)
  adds `np.random.seed(0)` before `AEC(cfg)` to match
  `eval_aec_challenge.py:325` convention.
- **Verified**: 800-case production bench unaffected (eval script seeds
  per-case independently); CLI re-invocations of `run_one_case.py` on the
  same case now produce byte-identical output.

#### Candidate HK-2: pcb1N artefact characterization [REFRAMED, 2026-05-15]
- **Origin**: v3.15 §10.S5 deferred to v3.16 as "mic_dynamic_margin
  patch". Listen-driven 2026-05-15 investigation **falsifies the
  original mechanism hypothesis** — pcb1N has NO clipping (mic peak
  -2.58 dBFS, online sat 0.000) and no startup margin issue.
- **Actual artefact (per user listen 2026-05-15)**: AEC output
  retains nearly the entire mic spectrum — filter / RES barely act on
  this case. Four contributing causes identified:
  1. **Static delay ~2000 samples (~125 ms @16k) > fl=832 (52 ms)**.
     If pre-align mis-resolves OR online tracker disagrees, the filter
     trains on mis-aligned mic/lpb pairs and never converges.
     mic-lpb cross-correlation `r = 0.071` (FS should be > 0.5)
     consistent with this.
  2. **Mic input characterised as "school PA speaker, low quality"** —
     loudspeaker introduces NL distortion in the echo path; echo is
     not a linear function of lpb. v3.13 E4 + E5 + v3.14 Volterra
     closures established this DSP-only ceiling.
  3. **LPB has "chained" spectral regions with audible boundary
     pops, not visible in time-domain** — codec / encoding artefacts
     in the reference signal itself (transient discontinuities the
     filter cannot model).
  4. **Mic level / gain unstable 80k–150k samples (5–9.4 s)** —
     suggests source-side AGC or ambient gain swings during the
     clip, defeating the steady-state assumptions in the per-band
     ERL EMA and shadow Q-damping (v3.14 Arc P / S-orth.A).
  5. **Mic HF energy not present in farend** — HF spectrum carries
     ambient / mic-intrinsic noise / HF reverb tail beyond the
     speaker output. F3.1 `mic-excess = mic - far*ERL` reads this
     HF surplus as NE evidence, raises `dt_per_bin`, lifts the per-bin
     gate, and lets HF echo + noise pass through the RES gain. This
     is the spectral-floor / quiet_mask substrate-shift surface
     (audit §3.4–3.5) feeding the audible "未處理頻譜" pattern.
- **Why HK-2 quick-patch path closed**: no single-knob fix addresses
  all three causes. Deeper mechanism (reverb-aware RES override OR
  delay-aware filter mode OR NL-detector-driven gain boost) requires
  audit + design + 800-case bench, exceeding "≤ 1 sprint" budget.
- **v3.16 disposition**:
  1. **HK-2 closed per §0.4** — original mechanism hypothesis
     falsified, no quick patch ships.
  2. **pcb1N promoted to v3.16 C6 (DelayEst audit) priority listen
     case** — case-1 of `delay > fl AND low cross-correlation` cohort
     that drives the C6 investigation.
  3. **NEW candidate C9 proposed for v3.16 Phase 4** (gated on C6
     outcome): `reverb_aware_res_override` — when mic-lpb r is
     persistently low + far_power high, switch RES to aggressive mode
     (override ENR thresholds with FS-bias). LOE M (detector + RES
     policy + 800-case to avoid NE/DT false-trigger).
- **Listen pack preserved**: `/tmp/v3_16_hk2_pcb1n_listen/` (mic +
  lpb + ours + ours_nores) for C6 audit reuse.

#### Candidate C1: `epc_dt_cap` dead-code removal [DONE]
- **Status (2026-05-15)**: SHIPPED on `feature/v3.16` — full mechanism
  removal (cap action + state-driven flag substrate + gate
  computation). 5/5 byte-equal sanity PASS (FS_static, FS_movement,
  DT_static, NE, pcb1Nh0Z) for both `_ours.wav` and `_ours_nores.wav`.
  800-case re-audit confirms 0/2,032,022 frame fire-rate across ALL
  buckets including cohort_tail.
- **Files changed**:
  - [`python/aec.py`](../python/aec.py): removed
    `res_state_driven_epc_dt_cap` config field, constructor param +
    self-attr, gate computation block, cap action, `epc_dt` postprocess
    signature param, call-site keyword arg, ResFilter wire-through.
    `_diag_round5_stages[2]` + `_stage_gains['03_epc_dt_cap']` writes
    preserved as aliases of stage 02 for diagnostic backward compat
    (P52 9-slot contract).
  - [`python/res_refactored/spectral_shaper.py`](../python/res_refactored/spectral_shaper.py):
    mirrored signature + cap removal for byte-equal vs legacy.
  - [`python/res_refactored/gain_computer.py`](../python/res_refactored/gain_computer.py)
    + [`__init__.py`](../python/res_refactored/__init__.py): updated
    docstring scope notes to reflect removal.
- **Net LOC change**: ~30 LOC delete + ~12 LOC doc/comment update.

#### Candidate C1b: `ne_g_floor` removal — RECLASSIFIED RETAIN [2026-05-15]
- **Original hypothesis**: ne_g_floor is dead code (audit reported
  0/2,032,022 fires). FALSIFIED.
- **C1c verdict** (after audit-script fix shipped 37a46b2): real
  fire-rate matches v3.13 baseline (~0.93–0.97 on DT buckets),
  ne_g_floor is the **universal NE-protection spectral floor** still
  load-bearing on DT frames. Empirical confirmation: full removal
  patch broke 4/5 byte-equal sanity (`_ours.wav`).
- **Status**: REMOVED-FROM-DELETE-CANDIDATE-LIST. Substrate
  preserved at `git stash@{0}` for reference; do NOT pop unless
  pursuing per-band refactor (see below).
- **Reclassification options for v3.16**:
  1. **Subsume into C2** (preferred) — C2 ENR per-state × per-band
     refactor naturally exposes the per-bin gate that ne_g_floor
     currently provides as a scalar; ne_g_floor becomes a per-band
     ceiling inside C2's table. NO standalone C1b sprint.
  2. **Standalone per-band ne_g_floor refactor** (alternative if C2
     hits scope creep) — split ne_g_floor scalar into 3-band
     vector, tune per-band thresholds. Independent of C2 but
     duplicates effort.
  3. **Subsume into C9** (long-tail) — if reverb-aware RES override
     needs ne_g_floor as a sub-mechanism, C9 lifts it; not the
     primary path.
- **Recommended disposition**: option 1 (subsume into C2). C1b
  closed as standalone candidate in Phase 0; reopened as part of C2
  scope in Phase 2.
- **Implementation LOE**: 0 standalone (folded into C2's M scope).

#### Candidate C1c: fix `ne_g_floor` audit-script measurement bug [DONE 2026-05-15]
- **Status**: SHIPPED on `feature/v3.16` commit `37a46b2`. Audit
  bug fixed via path (a) — un-gated pre-max diag fields added to
  `ResFilter`, audit script consumes per-bin any-fire flag.
- **Files changed**:
  - [`python/aec.py`](../python/aec.py): added 4 pre-max diag fields
    (`_stats_pre_max_spectral_g_min`,
    `_stats_pre_max_spectral_g_min_max`, `_stats_ne_g_floor_max`,
    `_stats_ne_g_floor_any_bin_fired`) computed unconditionally
    at line 3752 BEFORE the `np.maximum` raise.
  - [`tools/research/v3_15_res_audit.py`](../tools/research/v3_15_res_audit.py):
    `_ne_g_floor_fired()` prefers per-bin flag, falls back to scalar
    pre-max comparison on older builds.
- **Recovered fire-rates** (5-case directional, post-fix):
  ne_g_floor DT_static 0.969 / DT_movement 0.974 / GLOBAL 0.971 —
  matches v3.13 baseline (0.934 / 0.933) within +0.04 substrate
  drift expected from v3.14 Arc P + R + S-orth.A.
- **Byte-equal**: 10/10 PASS on 5-case sanity (ours + nores). No
  production behaviour shift — only diag-field captures + audit-
  script read path changed.
- **Unblocks**: C1b reclassification (see below).

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
  tuple becomes part of C2's table); C1b ne_g_floor reclassification
  (per-band ne_g_floor becomes part of C2's table per §C1b option 1).

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

#### Candidate C9: Reverb-aware RES override — NEW [Phase 4, from HK-2 reframing]
- **Origin**: HK-2 reframed 2026-05-15. pcb1N listen analysis
  surfaced four contributing causes (heavy reverb / delay > fl /
  PA-quality NL / lpb codec NL / mic gain instability / HF mic
  energy not in farend). None is patchable by a single knob; the
  filter / RES "barely act" surface is structural — when echo path
  is heavily reverberant, the linear filter's `fl=832` (52 ms) tail
  is shorter than room reverb (~125 ms+), and the F3.1 mic-excess
  evidence path mis-reads HF reverb as NE.
- **Mechanism**: Detect persistently low mic-lpb cross-correlation
  (`r < 0.15` over a sliding window with `far_power > threshold`)
  → switch RES to FS-aggressive mode for the duration: tighten
  `enr_t_ne / enr_s_ne` toward FS values, lift `effective_scale`,
  optionally cap dt_per_bin at HF bins. Override times out after
  K frames of `r > 0.3` or `far_power < threshold`.
- **Predicted AECMOS Δ**: TBD — pcb1N audible improvement is the
  primary success target; cohort-wide AECMOS may move ±0 because
  the affected case fraction is small.
- **Implementation LOE**: M-L (2-3 sprints — detector calibration
  on pcb1N + 3-5 reverb-heavy cohort cases; RES override wire;
  800-case A/B for FS / DT false-trigger guard).
- **Cohort-tail risk**: LOW — cohort_tail mechanism is
  movement-driven (path change detected by ERL_decile_std),
  orthogonal to persistent-reverb detector. Both can fire
  independently without conflict.
- **Measurable success**: pcb1N audible filter / RES action
  improvement (user listen) AND cohort tail Δecho ≥ −0.05 AND
  FS Δecho ≥ −0.020 AND DT Δdeg ≥ −0.005 AND NE Δdeg ≥ −0.005.
- **Gates on**: C6 (DelayEst audit may resolve cause #1 first; if
  pcb1N's primary issue is delay tracking, C9 mechanism scope
  shrinks to NL + reverb-only).
- **Listen pack**: `/tmp/v3_16_hk2_pcb1n_listen/` (mic + lpb + ours
  + ours_nores) preserved for C9 design + tuning.
- **Priority**: LOW-MED (single-case audible debt, but case is
  characteristic of a real-world subset — PA-speaker meeting rooms).

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

**Single-case audible candidates (not in Δ ranking)**: C9 reverb-aware
RES override (HK-2 reframed) — primary success metric is pcb1N
audible improvement, not cohort AECMOS Δ.

### v3.16 phase ordering

```
Phase 0 (parallel-safe; before everything else)
├── HK-1  B3 fix (run_one_case.py CNG seed)                  [DONE ea8d320]
├── HK-2  pcb1N artefact characterization                     [REFRAMED → C6 + C9]
├── C1    epc_dt_cap dead-code removal (ZERO Δ)               [DONE d90efdc]
├── C1b   ne_g_floor removal — RECLASSIFIED retain/refactor   [substrate stash@{0}; defer to C2 / C9]
└── C1c   audit ne_g_floor measurement bug fix                [DONE 37a46b2]

Phase 1 (BLOCKS Phase 2-4)
├── C5    Per-state RES interface modularisation [architectural]
└── C6    DelayEst audit (outcomes RE-ORDER Phase 3-4)
        ├── if movement closures = upstream delay → Phase 3-4 needs delay fix first
        └── if movement closures = mechanism walls → Phase 3-4 proceeds independently

Phase 2 (gated on C5)
├── C2    ENR per-state × per-band refactor (含 Arc D merge re-eval; subsumes C1b's ne_g_floor question)
├── C3    Stage-1 4-cap chain ordering audit + reorder
└── C4    noise_floor / CNG interaction refactor

Phase 3 (gated on C6 outcome + Arc T detector default ON [DONE v3.15 §10.S0b])
├── v3.16-A  force_render OR-in cohort_tail_T (Arc T H2 fix)
└── v3.16-B  ENR-path lift via enr_t_ne × cohort_tail_T (also gated on C2)

Phase 4 (gated on C6 + Phase 3)
├── C7    Arc M.v3 retry (α / β / γ — needs new detector primitive from C6 OR new dispatch from Phase 3)
├── C8    Arc G non-destructive partial W decay (independent within phase, LOW priority)
└── C9    Reverb-aware RES override (NEW from HK-2; gated on C6 — pcb1N audible target)
```

### v3.16 sprint estimate

| Phase | Sprints | Status |
|---|---|---|
| 0 housekeeping | 0 (HK-1 + HK-2 + C1 + C1c DONE; C1b reclassified to C2 surface) | ~50 % done at v3.15 closeout |
| 1 foundation | 4 – 5 (C5 architectural; C6 independent audit) | pending |
| 2 RES refactor | 6 – 9 (3 candidates × 2 – 3 sprints each) | gated on C5 |
| 3 Arc T consumers | 4 – 6 | gated on C6 |
| 4 Arc M / G / C9 (if opened) | 6 – 9 (C7 4-6 + C8 2-3 + C9 2-3) | gated on C6 + Phase 3 |
| **TOTAL** | **20 – 29 sprints** | |

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
| ≥ 5 refactor candidates listed | **PASS** (15 candidates: HK-1 / HK-2 / C1 / C1b / C1c / C5 / C6 / C2 / C3 / C4 / v3.16-A / v3.16-B / C7 / C8 / C9) |
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

- **Audit on full 800-case** [DONE 2026-05-15]: post-C1c full audit
  ran on `feature/v3.16` HEAD `37a46b2`. Outputs at
  `/tmp/v3_16_audit_full/{audit.json,summary.csv,per_case/*.json}`.
  §3 numbers above are 800-case authoritative. v3.16 phase-entry
  re-audit no longer needed unless substrate changes.
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
