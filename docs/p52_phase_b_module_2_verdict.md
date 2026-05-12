# P52 Phase B Task B.3 — Module 2 GainComputer verdict

**Date**: 2026-05-12
**Branch**: `feature/p52-phase-b-refactor`
**Status**: **PASS** (hard bar AND internal 100% target both met)

## Verdict at a glance

| Bar | Result |
|---|---|
| Hard bar §3.6 (≥99.99% within `atol=1e-6, rtol=1e-5`) | **PASS** (100% exact) |
| Internal advisory target (100% exact match) | **PASS** |
| Sample size | 100 cases (seed=42) |
| Total samples | 40,814,400 |
| Cases byte-identical | **100 / 100** |
| Top-10 max abs delta | 0.0 |

## Extraction methodology

`ResFilter._stage_gain_compute` (aec.py:1835-1971) extracted into
`python/res_refactored/gain_computer.py::gain_computer(rf, **kwargs)`.
Verbatim transcription (`self.X` → `rf.X`); no algebra simplification,
operation reorder, or threshold change. `ResFilterRefactored` adds the
override:

```python
def _stage_gain_compute(self, **kwargs):
    return gain_computer(self, **kwargs)
```

## Deviation from v1.1 §3.3 mapping (documented)

v1.1 §3.3 places `epc_dt_cap` (diag [2]) inside Module 2. In legacy code
it lives as the first operation of `_stage_gain_postprocess`. Moving it
into `gain_computer()` would require changing the `_stage_gain_compute`
signature (passing `epc_dt` in) and removing the corresponding block from
`_stage_gain_postprocess` — a coordinated two-method override with risk
of cross-coupling bugs.

For B.3 Module 2 the byte-equal-preserving decision is to **leave
`epc_dt_cap` in Module 3** (`_stage_gain_postprocess`). The §3.3 logical
mapping is honored at the time-ordered sequence level (`gain_compute → 
epc_dt_cap → ...` unchanged); the code-locality shift is deferred to the
post-five-extraction `ResState` migration pass, when the orchestrator
owns frame ingestion and module boundaries are under full refactor
control without partial-state-duplication risk.

This deviation is strictly within v1.1 spec: §3.4 specifies `ResState`
as a deliverable of Phase B as a whole, not of each individual module
migration. §5.5 is honored — RES logic unchanged.

## Anti-loophole compliance

- §5.5: verbatim copy; zero numerical change.
- §5.4: no shadow / classifier reference.
- §6.4: only `python/res_refactored/*` + tooling + docs.

## Cross-references

- Module impl: [python/res_refactored/gain_computer.py](../python/res_refactored/gain_computer.py)
- Module 1 verdict: [p52_phase_b_module_1_verdict.md](p52_phase_b_module_1_verdict.md)
- Validation tool: [tools/research/p52_phase_b_b3_byte_equal.py](../tools/research/p52_phase_b_b3_byte_equal.py)
- Artefacts: `/tmp/p52_b3/refactored_m12.npz`, `/tmp/p52_b3/verdict_m12.json`
