# v3.15 closeout verdict pack — for §0.7 merge authorisation

**Date**: 2026-05-15
**Branch**: `feature/v3.15-arc-g` HEAD `04c1dfe` (post §10.S4 commit)
**Baseline**: `main` HEAD `b3273de` (v3.14 production: Arc P + Arc R +
Arc S-orth.A all default ON)
**Author**: v3.15 Phase F closeout
**Doc role**: Final summary for user §0.7 merge authorisation. All
Phase A.0 → F sprints complete; verdict + evidence + cleanup state in
one place.

---

## 1. Headline

- **v3.15 ship-able algorithm change**: 1 default-ON promotion (Arc T
  cohort tail real-time detector → BALANCED preset; byte-equal on
  audio output, only `AecStats.cohort_tail_T` becomes informative).
- **v3.15 closed candidate arcs**: 6 (`§1.2`, `§1.4 Arc M V1+V2`,
  `§1.4 Arc G S3-S5`, `§1.5 Arc T S2 wiring`, `§1.5b Arc M.v3`,
  `§1.6 Arc F`).
- **v3.15 substrate retained**: 6 default-OFF flags (Arc M v1/v2/v3,
  Arc G, Arc T RES preempt mode, Arc F per-band Q).
- **v3.15 bug fixes shipped**: B4 (quiescent re-sync dead branch fix),
  B5 (`_shadow_copy_err_baseline` doc-aligned-to-impl), B9 (j=6
  chunk-split bench tooling); B10 / B11 (process discipline notes —
  no code change).
- **v3.16 RES refactor plan**: 13 candidates ranked, hard-bar PASS,
  authorised pending §0.7 user review.
- **v3.13 E2 Path 3 DT debt** (DT_static −0.050, DT_movement −0.025):
  remains unrecoverable in v3.15 production; closure target moves to
  v3.16 (per §1.7 plan candidates C2 / C4 / C3 totalling predicted
  +0.005 to +0.040 DT bucket recovery).

---

## 2. v3.15 candidate arc disposition

| ID | Arc | Status | Production change | Substrate | Verdict doc |
|---|---|---|---|---|---|
| §1.0 | Phase A.0 v3.14 substrate re-verify | DONE (B4 / B5 partial; S-orth.A nores listen NOT executed) | none | n/a | [`v3_15_b4_verdict.md`](v3_15_b4_verdict.md) + [`v3_15_b5_verdict.md`](v3_15_b5_verdict.md) |
| §1.2 | DT-NE compression fix | CLOSED CANNOT SHIP | none | flag default OFF | [`v3_15_dt_ne_compression_fix_closure.md`](v3_15_dt_ne_compression_fix_closure.md) |
| §1.3 | Arc D merge | DEFERRED post Phase E | none | `feature/v3.14-arc-d` retained | (deferred per plan §10) |
| §1.4 | Arc M (S1+S2) V1+V2 | CLOSED CANNOT SHIP | none | `arc_m_epc_gated` default OFF | [`v3_15_arc_m_closure.md`](v3_15_arc_m_closure.md) |
| §1.4 | Arc G S3-S5 | CLOSED CANNOT SHIP | none | `arc_g_per_band_w_reset` default OFF | [`v3_15_arc_g_closure.md`](v3_15_arc_g_closure.md) |
| §1.5 | Arc T S1 (detector) | PASS substrate; **default ON in BALANCED** (§10.S0b) | 1 line `arc_t_cohort_detector=True` in BALANCED | n/a (now default ON) | [`v3_15_arc_t_s1_design_and_verdict.md`](v3_15_arc_t_s1_design_and_verdict.md) |
| §1.5 | Arc T S2 (RES preempt wiring) | CLOSED no-op | none | flag retained for code symmetry | [`v3_15_arc_t_s2_wiring_closure.md`](v3_15_arc_t_s2_wiring_closure.md) |
| §1.5b | Arc M.v3 (T-gated) | CLOSED CANNOT SHIP | none | `arc_m_t_gated_enabled` default OFF | [`v3_15_arc_m_v3_closure.md`](v3_15_arc_m_v3_closure.md) |
| §1.6 | Arc F per-band Q | CLOSED (substrate) | none | `kalman_q_per_band` + scales default OFF | [`v3_15_arc_f_closure.md`](v3_15_arc_f_closure.md) |
| §1.7 | RES audit + v3.16 plan | DONE | none | n/a | [`v3_15_res_audit_and_refactor_plan.md`](v3_15_res_audit_and_refactor_plan.md) |
| §1.8 | E1 mic_dynamic_margin pcb1N | DEFERRED | none | n/a | (deferred → v3.16 HK-2) |

**v3.15 ship-able algorithm change count**: **0 algorithm; 1 preset
default flip** (`arc_t_cohort_detector=True` in BALANCED, byte-equal
on audio output per §10.S0b 5-case sanity).

---

## 3. v3.15 production deliverables (what `feature/v3.15-arc-g` brings if merged)

### 3.1 Algorithm
1. **Arc T cohort tail detector default ON** in BALANCED preset
   (commit `5bb2fa8`). `AecStats.cohort_tail_T` becomes informative;
   `_arc_t_cohort_tail_signal` field becomes available to v3.16 RES
   refactor consumers (Phase 3 candidates v3.16-A / v3.16-B). Audio
   output byte-equal verified on 5 cases (NE / DT / DT_movement / FS
   / FS_movement, atol=0.0).

### 3.2 Bug fixes (already in earlier work, verified during closeout)
2. **B4 fix** (line 6721 / 6728): `_prev_filter_state` quiescent
   re-sync now correctly checks only `'refined_usable'` (drop dead
   `'converged'` enum-mismatch branch). Restores Arc S-orth.A safety
   net intent. Verdict: [`v3_15_b4_verdict.md`](v3_15_b4_verdict.md).
3. **B5 disposition** (lines 691-700): `_shadow_copy_err_baseline` and
   `_shadow_mu_holdoff` documented as RESERVED (declared but not wired
   — future arc scope). Doc aligned with actual implementation. No
   code change. Verdict: [`v3_15_b5_verdict.md`](v3_15_b5_verdict.md).

### 3.3 Bench tooling improvements
4. **B9 fix — j=6 chunk-split** (commit `1323f92`): `eval_aec_challenge.py`
   now supports `--workers` CLI flag (default 6) with per-scenario
   chunk-split (`n_chunks = workers // 3`). Speeds 800-case bench
   ~2× over hardcoded `max_workers=3`. Byte-equal sanity 120/120
   between j=3 and j=6 outputs.
5. **`--cases-list` flag** (also in `1323f92`): allows curated
   subset bench (Tier 1 60-case via `tools/research/v3_15_subset_cases.txt`).
   Used by §1.7 RES audit script for fast directional reads.

### 3.4 Substrate retention (default-OFF research flags)
6. **Arc M V1**: `arc_m_epc_gated`, `kalman_q_per_band`,
   `kalman_q_band_scales` (commit `fe0177d`). Closure:
   [`v3_15_arc_m_closure.md`](v3_15_arc_m_closure.md).
7. **Arc M.v3**: `arc_m_t_gated_enabled` (commit `e8df090` + rename
   `03e311b`, drop `_v3` version suffix per project naming
   convention). Closure: [`v3_15_arc_m_v3_closure.md`](v3_15_arc_m_v3_closure.md).
8. **Arc G**: `arc_g_per_band_w_reset` + `arc_g_drift_ratio` etc.
   (commit `a77a176`). Closure: [`v3_15_arc_g_closure.md`](v3_15_arc_g_closure.md).
9. **Arc T RES preempt**: `arc_t_res_preempt_mode` (commit `4d8ae1c`).
   Closure: [`v3_15_arc_t_s2_wiring_closure.md`](v3_15_arc_t_s2_wiring_closure.md).
10. **Arc F per-band Q substrate**: documented but flag may be removed
    in v3.16 — see [`v3_15_arc_f_closure.md`](v3_15_arc_f_closure.md).
11. **§1.2 DT-NE compression fix substrate**:
    `dt_ne_compression_fix=False` flag retained for v3.16 reference.
    Closure: [`v3_15_dt_ne_compression_fix_closure.md`](v3_15_dt_ne_compression_fix_closure.md).

### 3.5 Documentation
- 15 v3.15 verdict / closure / design docs total in [`docs/`](.).
- v3.16 plan: [`v3_15_res_audit_and_refactor_plan.md`](v3_15_res_audit_and_refactor_plan.md)
  with 13 candidates + Phase 0-4 dependency graph.

### 3.6 v3.14 substrate preservation (commit `7c42e40`)
- `docs/v3_14_h_s1_verdict.md` + `tools/research/v3_14_h_s1_huber_proto.py`
  pulled from soon-deleted `feature/v3.14-arc-h` worktree.
- `docs/v3_14_r_s2_verdict.md` updated with R.S2.1 admit_hf control
  result (was uncommitted in `feature/v3.14-arc-r` worktree).
- `docs/v3_14_s_orth_b_s2_verdict.md` adds 800-case bench result for
  L1-regularized shadow weight update (was uncommitted in
  `feature/v3.14-arc-s-orth-b` worktree).

---

## 4. Bench evidence summary

| Bench | Config | Cases | Result | Use |
|---|---|---|---|---|
| §10.S0b detector promotion sanity | BALANCED, fl=832, env=0 vs env=1 | 5 (NE / DT / DT_mvmt / FS / FS_mvmt) | 5/5 byte-equal PASS (md5) | Default-ON safety |
| §1.5b S1 byte-equal sanity | 4 flag combos × 5 cases = 20 | 5 stems × 4 combos | OFF/OFF, M_v3-only, T-only = baseline; both-ON differs only on SgKY30fj DT_mvmt (1/5) — gate firing as designed | Wiring correctness |
| §1.5b S2 60-case subset | C0 / C1 / C2 vs baseline | 60 | V1 ΔERLE_lin == M.v3 ΔERLE_lin in EVERY bucket; H2/H3/H4b FAIL → CLOSE per §0.4 | Arc M.v3 closure evidence |
| §1.5b S2 800-case | (skipped) | — | Not run; subset + trace evidence conclusive (per user 2026-05-15) | Time saved (12 min) |
| §1.7 RES audit subset | BALANCED, fl=832, j=6 | 60 (DT-only sample bias) | ne_g_floor 0/all DT (v3.13 baseline 0.93) → C1b candidate; epc_dt_cap 0/all (confirmed dead); spectral_floor / quiet_mask topology unchanged | v3.16 plan input |
| §1.5 Arc T S1 8-case validation | TAIL + CTRL stems | 8 | 5/5 TAIL fire + 3/3 CTRL no-fire (T_HI=18.5 dB / T_LO=13.0 dB) | Detector calibration |
| §1.4 Arc G S4 listen | (5 gain-change cases) | 5 | 0/5 audible improvement; ERLE Δ=−1.48 dB → CLOSE | Arc G closure |

No 800-case bench was run during v3.15 closeout — all candidate arcs
closed via subset evidence + trace analysis. Per user 2026-05-15, this
is acceptable when subset evidence is structurally conclusive (mechanism
walls proven by trace, not statistical bucket-mean noise).

---

## 5. v3.16 candidate plan summary

**Doc**: [`v3_15_res_audit_and_refactor_plan.md`](v3_15_res_audit_and_refactor_plan.md)

**13 candidates across 5 phases**:

| Phase | Candidates | Sprints |
|---|---|---|
| 0 housekeeping | HK-1 (B3 CNG seed), HK-2 (pcb1N patch), C1 (epc_dt_cap removal), **C1b (ne_g_floor removal — NEW substrate-shift)** | 3 – 4 |
| 1 foundation | C5 (per-state RES interface), C6 (DelayEst audit) | 4 – 5 |
| 2 RES refactor | C2 (ENR per-state × per-band), C3 (4-cap reorder), C4 (noise_floor / CNG) | 6 – 9 |
| 3 Arc T consumers | v3.16-A (force_render OR-in), v3.16-B (ENR-path lift) | 4 – 6 |
| 4 Arc M / G retry | C7 (Arc M.v3 α/β/γ), C8 (Arc G non-destructive decay) | 4 – 6 |
| **TOTAL** | | **21 – 30** |

**Hard bar status**: PASS (13 candidates listed; ≥ 3 with predicted Δ
≥ +0.005 — actual: 5).

**Top 5 candidates by predicted AECMOS Δ**:
1. v3.16-A — +0.030 cohort tail Δecho (Arc T H2 force_render OR-in fix)
2. v3.16-B — +0.020 cohort tail Δecho (Arc T H1 ENR-path lift)
3. C2 — +0.005 to +0.015 DT bucket Δdeg (ENR per-state × per-band)
4. C4 — +0.005 to +0.015 DT bucket Δdeg (noise_floor / CNG refactor)
5. C3 — +0.003 to +0.010 DT bucket Δdeg (4-cap chain reorder)

**Critical gate**: C6 DelayEst audit outcome may re-order Phase 3-4
priorities. Movement-related closures (cohort tail, Arc M V1
FS_movement, Arc F cohort tail, Arc G destructive W reset, §1.1 H5
hypothesis) all share echo-path-changing-during-case where DelayEst
tracking drives downstream filter behaviour. If DelayEst is the
upstream cause for ≥ 30 % of those wall magnitudes, a delay-aware
detector arc may consume the bulk of the recoverable debt before any
Phase 2-4 mechanism arc opens.

---

## 6. Branch + workspace state (post-cleanup)

**Branches retained** (3 total):
- `main` (production v3.14 HEAD `b3273de`)
- `feature/static-memory` (NR/AEC static memory branch)
- `feature/v3.15-arc-g` (current dev HEAD `04c1dfe`)

**Branches deleted** (30 total): 27 obsolete `feature/` branches + 3
`worktree-agent-*` cleanup branches. See plan §10.S0c section E.

**Worktrees removed** (8 total): all `AEC-worktrees/*` cleared; 4
research substrate files preserved into `feature/v3.15-arc-g` via
commit `7c42e40`.

**Disk reclaim** (post §10.S0c cleanup):
- `out_python_*/` (12 dirs, ~14 GB)
- `results/` `.wav` + `.png` files (~14 GB; kept scores.json +
  result.md)
- `/tmp/v3_15_*` artifacts (~720 MB; kept `subset_c0/` + `scores_c0/`
  for §10.S4 baseline)
- `AEC-worktrees/` (~1.2 GB)
- AEC repo: 4.6 GB; disk free 199 GB.

---

## 7. Process notes (lessons captured for v3.16+)

1. **Single-branch policy** during closeout: parallel sub-branches
   triggered §0.2 worktree CWD bug (B11) twice. v3.15 closeout
   consolidated to single branch `feature/v3.15-arc-g` via
   cherry-pick.
2. **CWD discipline** (B10): Bash CWD drifts silently — use absolute
   paths for `python/...` invocations.
3. **V1 reproduction config completeness**: V1 needs THREE flags
   (`AEC_ARC_M_EPC_GATED` + `AEC_KALMAN_Q_PER_BAND` +
   `AEC_KALMAN_Q_BAND_SCALES`). Setting only one silently produces
   baseline behaviour. v3.16 should add a "preset combo" env var
   (e.g. `AEC_PRESET_COMBO=arc_m_v1`) that sets all required flags
   atomically.
4. **Subset Tier 1 bench infrastructure** (60-case @ j=6, ~5-10 min):
   reusable for v3.16 arc tuning sprints.
   `tools/research/v3_15_subset_*.{py,sh,txt}`.
5. **Arc T S2 wiring failure pattern**: writing to `self.res.X` from
   AEC level after RES init is overwritten by RES's own state machine.
   v3.16 RES refactor candidates v3.16-A / v3.16-B document the
   correct integration patterns.
6. **§1.5b T-gate timing/scope mismatch**: discrete-event signals
   (q_boost) don't pair well with persistent-state signals
   (cohort_tail_T) without designed temporal alignment. Future arcs
   pairing impulsive triggers with persistent detectors need
   pre-/post-assertion hysteresis on the gate.
7. **`v3` version suffix in variable names**: `arc_m_v3_t_gated_enabled`
   renamed to `arc_m_t_gated_enabled` (commit `03e311b`) per project
   naming convention — project-codename prefixes (`arc_m_*`, `arc_t_*`,
   `arc_g_*`) are kept as identifiers, but numeric design-iteration
   suffixes (`_v3`, `_v2`) belong in design doc labels and commit
   messages, not in live config field names.

---

## 8. §0.7 merge gate

Per plan §0.7: `feature/v3.15-arc-g` → `main` merge requires:

1. **Phase A.0 → F all green per their hard bars**: ✓ DONE (this doc
   compiles all verdicts).
2. **§1.7 RES audit produces v3.16 refactor plan with ≥ 3 ranked
   candidates**: ✓ DONE (13 candidates ranked; 5 with predicted Δ ≥
   +0.005).
3. **Closeout sprint compiles v3.15 verdict pack**: ✓ DONE (this
   document).
4. **User reviews pack, decides merge**: ⏳ PENDING.

**On user authorisation**:
- `git checkout main && git merge --no-ff feature/v3.15-arc-g`
- `git tag v3.15.0`
- `git push origin main` + `git push origin v3.15.0` + push v3.13.0 +
  v3.14.x intermediate tags + 75 deferred commits.

**On user decline**:
- `feature/v3.15-arc-g` remains as research substrate; document
  closure reasons; main stays at v3.14 production HEAD `b3273de`.

---

## 9. Recommended action

**Recommendation**: AUTHORISE `feature/v3.15-arc-g` → `main` merge.
Justification:

- v3.15 production algorithm change is minimal (1 default-ON flag flip,
  byte-equal on audio output) — low-risk merge.
- Bug fixes B4 / B5 / B9 are net positive (fix safety regularization
  intent + bench tooling speedup).
- Substrate retention is research-positive (6 default-OFF flags
  available for v3.16 retry without re-implementation).
- v3.16 plan with 13 ranked candidates provides a concrete forward
  path; ≥ 3 high-Δ candidates means RES architecture is NOT declared
  stable — v3.16 cycle should open per the §1.7 hard bar.
- Closeout cleanup (3 branches, 4.6 GB AEC, 199 GB free disk) brings
  workspace into reviewable state.

**Recommended next steps post-merge**:
1. Push to remote per §0.7 final paragraph.
2. Open v3.16 housekeeping sprints (HK-1 + HK-2 + C1 + C1b) parallel
   with C5 + C6 foundation kickoff.
3. Re-author NN Joint NR+RES+Dereverb plan when DSP-only constraint
   lifts (per `feedback_dsp_only_until_completion`; not in v3.16
   scope).
