# Archive

Superseded specs, change logs, and historical analyses. Kept for traceability;
do not treat as authoritative for the current design.

## What's here

### Per-version change logs (now consolidated into `../CHANGELOG.md`)
- `CHANGELOG_v2.3.0.md` / `CHANGELOG_v2.4.0.md` / `CHANGELOG_v2.5.0.md` / `CHANGELOG_v2.5.md`

### Completed C-port planning docs (work delivered as v3.8.2)
- `c_rewrite_plan.md` — original v2.5 → v3.8.1 sync plan
- `c_port_spec.md` — Python class → C struct mapping spec

### Era-bound design notes
- `signal_flow_constraints_v3.0.2.md` — v3.0.2 era audit of hardcoded constants;
  superseded by current `aec_methods.md`
- `TODO_v2.8.1.md` — Was empty; movement-DT ablations all closed

### Improvement tracking (v1.x / v2.x experiments)
- `aec_improve_v12.md` / `aec_improve_v13.md`
- `bisect_analysis_and_plan.md`
- `movement_dt_ablation_report.md`
- `phase_kappa_report.md`
- `phase2_stage_*.md`

### Architecture comparisons (older drafts)
- `aec3_architecture_alignment.md` / `aec3_full_architecture_analysis.md`
  → superseded by `../aec3_reference.md`
- `aec_vs_old_webrtc_analysis.md`

### Specs and dev logs
- `spec_*` — superseded specs (dt_jump_veto, raw_dt alignment, shadow_nlms, …)
- `DEVLOG.md` — historical dev journal
- `b15_superseded.md` / `b16_stage_1d_results.md`
- `handoff_*.md` — session handoffs
- `multi_point_change_plan_v3_1.md`
- `aec_post_linear_presentation.md`

### Baselines
- `baselines/baseline_v3*_vs_aec2.json` — historical 800-case AECMOS scores
