# v3.19 Phase 0.2 — Phase E status decision (2026-05-16)

**Status**: DECIDED — Phase E is **conditional Phase 5** in v3.19,
fires only if Phase 1 / 2 / 3 land any shipping algo change.
**Verdict**: Wire Phase E.0 trigger gate at Phase 4 entry; do not
pre-allocate sprints.

## 1. Context

v3.18 plan §Phase H (later renamed Phase E in revision) defined a
4-6 sprint substrate-flag promotion arc:

> Promote 14 BALANCED-only substrate flags to MILD / SOFT /
> AGGRESSIVE / MAXIMUM presets

v3.18 dropped Phase E because 0 algorithm changes shipped → no fresh
flag to promote. But the underlying need — **BALANCED accumulates
flags faster than other presets** — remains unaddressed.

## 2. Current BALANCED-only flag inventory (2026-05-16)

Survey of [python/aec.py:1288-1365](../python/aec.py#L1288) BALANCED
defaults, comparing against MILD / SOFT / AGGRESSIVE / MAXIMUM:

| Flag | First shipped | Source cycle | Currently in non-BALANCED? |
|---|---|---|---|
| `use_mic_excess_evidence` | F3.1 v3 | v3.10 | NO |
| `epc_r_reset_enabled` | F2.3 | v3.10 | NO |
| `mu_holdoff_no_reset` | F2.4 | v3.10 | NO |
| `shadow_r_reset_enabled` | B5 | v3.11 | NO |
| `shadow_state_decoupled` | S-orth.A | v3.14 | NO |
| `f3_1_per_band_erl_adaptive` | Arc-P P.S3 | v3.14 | NO |
| `res_per_band_enr` | Arc-R R.S2 | v3.14 | NO |
| `f_e5_enabled` | F-E5 | v3.11 | NO |
| `diverged_reset_enabled` | F2.5 / triple-AND | v3.11 | NO |
| `diverged_reset_triple_and` | (paired with above) | v3.11 | NO |
| `shadow_mu_state_aware` | B6 | v3.12 | NO |
| `f_e1_enabled` | F-E1 | v3.12 | NO |
| `f_delaytrack_enabled` | F-DelayTrack | v3.12 | NO |
| `arc_t_cohort_detector` | Arc-T S0b | v3.15 | NO |

**Count: 14 BALANCED-only flags** (matches original Phase E target).

Plus v3.18 substrate (all default-OFF in **all** presets, not in
BALANCED either):
- `shadow_class_nlms` / `shadow_mu_nlms` (Phase A)
- `filter_misadjustment_*` 9 flags (Phase B)
- `filter_analyzer_enabled` (Phase C.A)
- `filter_quality_enabled` (Phase C.B)
- `aec_state_enabled` (Phase C.C)
- `leakage_diverged_*` 3 flags (Phase C.D-α)
- `c_e_res_use_fq_usable` (Phase C.E)

These do not trigger Phase E (already default-OFF everywhere; no
preset asymmetry).

## 3. Re-evaluation question

Phase E currently waits on Phase 1/2/3 ship. Two sub-questions:

### Q1: Should Phase E include a v3.19 ship?

**Answer: Yes, conditional**. If Phase 1 (Pareto-walking) ships an
algo change as new BALANCED-only flag (e.g.,
`c_e_branch_xxx_use_fq_usable=True`), Phase E promotes that flag
across presets per the original plan.

### Q2: Should Phase E ALSO clean up the 14 pre-v3.19 BALANCED-only flags?

**Answer: No, defer**. Adding 14-flag promotion to v3.19 doubles
Phase E LOE (4-6 sprints → 8-12 sprints) and doesn't satisfy v3.19's
ship-priority goal — it's housekeeping. Promotion of pre-v3.19 flags
remains valid backlog but moves to v3.20+.

**Rationale**: each pre-v3.19 flag was tuned against BALANCED's
specific (`shadow_q_ratio=3.5`, `kalman_q_high=1e-3`,
`res_g_min_db=-55.0`) operating point. Promoting to MILD
(`shadow_q_ratio=2.0`, `kalman_q_high=2e-3`, `res_g_min_db=-44.0`) or
MAXIMUM (`shadow_q_ratio=5.0`, `kalman_q_high=7e-4`,
`res_g_min_db=-72.0`) needs per-flag × per-preset 60-case audit
(14 × 4 = 56 audits) before any 800-case bench. That's the bulk of
Phase E LOE if all 14 flags get promoted.

For v3.19, Phase E scope is **only the v3.19-shipped flag(s)**.

## 4. Decision

### Phase E trigger gate (at Phase 4 entry)

After Phase 1/2/3 close, before Phase 4 closeout:

```
if (any of Phase 1.7 / 2.3 / 3.5 has shipping algo flag landed)
  AND (flag default ON in BALANCED only):
    enter Phase E (E.1-E.4 per plan)
else:
    skip Phase E; proceed to Phase 4 closeout
```

### Phase E sprint scope (conditional, 2-4 sprints if triggered)

- **E.1**: per-preset 60-case audit of v3.19-shipped flag(s) only
  (1-2 audits per flag, NOT 14 audits).
  Output: `docs/v3_19_e_preset_audit.md`.
- **E.2**: safe-promotion grouping — which presets accept the flag
  with no regression > -0.015 vs preset's native target bucket.
- **E.3**: Promote group 1 (no interaction); 800-case per accepting
  preset.
- **E.4**: Promote group 2 (post-interaction); 800-case per
  accepting preset.

If 0 algo ships in v3.19 → Phase E does **not** trigger. v3.19
closeout (Phase 4) re-states pre-v3.19 14-flag promotion as v3.20+
backlog.

### Pre-v3.19 14-flag promotion scope → v3.20+ backlog

New v3.20+ backlog entry:

> **BALANCED-only flag promotion across presets (14 pre-v3.19
> flags)** | 6-10 sprints | trigger: post-Phase-E completion of any
> v3.19 algo flag promotion, OR standalone housekeeping cycle |
> source: v3.18 Phase H scope deferral + v3.19 Phase 0.2

## 5. Sprint plan body update

No change to sprint plan body. Phase E entry already conditioned on
ship per existing plan §Phase E. This Phase 0.2 doc confirms the
gate logic and trims pre-v3.19 housekeeping out of v3.19 scope.

## 6. Cross-references

- [docs/v3_18_plan.md](v3_18_plan.md) — original Phase H definition
  (renamed Phase E in revision)
- [docs/v3_18_plan_revision_2026_05_15.md](v3_18_plan_revision_2026_05_15.md) — Phase E placement
- [python/aec.py:1288-1365](../python/aec.py#L1288) — BALANCED preset
  defaults (14 BALANCED-only flag survey source)
- `~/.claude/plans/se-aec-aec-main-hazy-lynx.md` — v3.19 cycle plan
  Phase E section
