# P52 Phase B Task B.3 — Module 1 ResidualEstimator verdict

**Date**: 2026-05-12
**Branch**: `feature/p52-phase-b-refactor`
**Commits**: pending (this doc + module impl + orchestrator + validation tool)
**Status**: **PASS** (hard bar AND internal 100% target both met)

## Verdict at a glance

| Bar | Result |
|---|---|
| Hard bar §3.6 (≥99.99% samples within `atol=1e-6, rtol=1e-5`) | **PASS** (100% exact) |
| Internal advisory target (100% exact match) | **PASS** |
| Sample size | 100 deterministic cases (seed=42) |
| Total samples compared | 40,814,400 |
| Cases byte-identical (`np.array_equal`) | **100 / 100** |
| Top-10 max abs delta | 0.0 across the board |

## Extraction methodology

`ResFilter._stage_residual_model` (aec.py:1732-1833) was extracted into a free
function `residual_estimator(rf, **kwargs)` in
`python/res_refactored/residual_estimator.py`. The function body is a verbatim
transcription of the 100-line stage method — every `self.X` reference replaced
with `rf.X`, no algebra simplification, no reordering, no threshold change.

The orchestrator `ResFilterRefactored`
(`python/res_refactored/res_filter_refactored.py`) subclasses the legacy
`ResFilter` and overrides exactly one method:

```python
class ResFilterRefactored(ResFilter):
    def _stage_residual_model(self, **kwargs):
        return residual_estimator(self, **kwargs)
```

Modules 2–5 still execute via the inherited legacy `_stage_*` methods. This
keeps each Phase B module migration small and independently verifiable.

### Why subclass + delegate (not yet `ResState`-typed pure function)

The B.1 inventory identified the `_stage_residual_model` body touches 8 distinct
`self.*` fields plus the `_residual_est` sub-object (which holds its own
`using_render_based` flag and internal long-window statistics consumed by
Module 3 via cross-stage read — anomaly A3 in B.1 doc). Migrating that state
onto `ResState` in B.3 alone would either (a) require partial duplication of
state during the transition or (b) ripple into Modules 2–5 simultaneously,
both of which violate the §3.5 "module-by-module byte-equal before next"
spec. The subclass + `rf` adapter preserves single-source-of-truth on every
field; the `ResState` migration is deferred to a post-all-five-extracted pass
once the surface area to refactor is bounded.

This deferral is **strictly within v1.1 spec**: §3.4 specifies `ResState` as a
deliverable of Phase B as a whole, not of each individual module migration.
§5.5 forbids RES logic change; subclass-and-delegate preserves logic exactly.

## Byte-equal validation

Tool: [tools/research/p52_phase_b_b3_byte_equal.py](../tools/research/p52_phase_b_b3_byte_equal.py)
Method: `snapshot` once with the legacy class, once with `aec.ResFilter` monkey-
patched to `ResFilterRefactored`; then `diff` two `.npz` snapshots per stem.
Configuration: balanced preset / fl=832 / cng=True / `np.random.seed(42)` per
case (matches A.0R.8 / A.0R.6 sampling and seed conventions).

Artefacts:
- `/tmp/p52_b3/legacy.npz`, `/tmp/p52_b3/refactored.npz`
- `/tmp/p52_b3/verdict.json`

## Drift analysis

Not applicable — zero drift observed at the sample level. No floating-point
accumulation difference, no operation reordering, no library version
dependency surfaced.

## Anti-loophole compliance

- §5.5 (Phase B may not change RES logic): verbatim copy of stage body; no
  threshold tuning, no operation reorder, no removal/addition of any
  computation.
- §5.4 (shadow filter strict scope): Module 1 has no reference to
  `PathChangeRegimeHandler`, `AcousticRegimeClassifier`, or any shadow state.
- §6.4 (branch isolation): only `python/res_refactored/*` files modified +
  new tooling + new verdict doc. No `aec.py` change in B.3.

## Next steps

- **B.3 Module 2** (`GainComputer`): extract `_stage_gain_compute` (aec.py:1835)
  plus the `epc_dt_cap` block from `_stage_gain_postprocess` (per inventory
  cross-module remapping). Anomaly A2 (Module 2 reads Module 5's previous-
  frame `noise_psd`) requires pass-through via the `rf` adapter — already
  available since both stages still touch `self.noise_psd`. Same subclass-
  and-delegate pattern.
- After all five modules migrated and B.4 confirms 800-case byte-equal,
  migrate state onto `ResState` (§3.4) as a single bulk pass with full
  byte-equal re-verification.

## Cross-references

- Inventory: [docs/research_log_p52_phase_b_inventory.md](research_log_p52_phase_b_inventory.md)
- Anomalies log: [docs/phase_b_anomaly_notes.md](phase_b_anomaly_notes.md)
- Module impl: [python/res_refactored/residual_estimator.py](../python/res_refactored/residual_estimator.py)
- Orchestrator: [python/res_refactored/res_filter_refactored.py](../python/res_refactored/res_filter_refactored.py)
- Validation tool: [tools/research/p52_phase_b_b3_byte_equal.py](../tools/research/p52_phase_b_b3_byte_equal.py)
- Phase A closure: [docs/p52_phase_a_verdict.md](p52_phase_a_verdict.md)
