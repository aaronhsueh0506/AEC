# P52 Phase B — Verdict (RES modular byte-equal refactor)

**Branch**: `feature/p52-phase-b-refactor`
**Date**: 2026-05-12
**Status**: **CLOSED — PASS**

## What Phase B accomplished

| # | Task | Status | Commit |
|---|---|---|---|
| B.1 | RES stage inventory (9 stages → 5 Modules mapping) | DONE | `76993b7` |
| B.2 | `res_refactored` package: `ResState` + 5 module stubs | DONE | `5d01e5b` |
| B.3 Module 1 | `ResidualEstimator` — `_stage_residual_model` extraction | DONE | `2652478` |
| B.3 Module 2 | `GainComputer` — `_stage_gain_compute` extraction | DONE | `31004aa` |
| B.3 Modules 3-5 | `SpectralShaper` / `TemporalSmoother` / `NoiseFloorAndCng` | DONE | `3bd58c6` |
| B.4 | 800-case full byte-equal sweep | **PASS** | `2c27428` |
| B.5 | `use_res_refactored: bool = False` config flag + AEC swap site | DONE | this commit |

## B.4 hard-bar result

| Bar | Value | Result |
|---|---:|---|
| §3.6 ≥99.99% within `atol=1e-6, rtol=1e-5` | 100.0000% | **PASS** |
| §3.6 zero cases > 0.1 dB voice-band mean gain diff | 0 / 800 | **PASS** |
| Internal advisory target: 100% exact match | 100.0000% | **PASS** |
| Cases byte-identical (`np.array_equal`) | **800 / 800** | — |
| Total samples compared | 325,123,520 | — |

The full corpus reproduced the B.3 100-case sample exactly — no edge-case
cohort behaviour emerged at scale.

## B.5 wiring

`AecConfig.use_res_refactored: bool = False` ([aec.py:182](../python/aec.py#L182)).
`AEC.__init__` ResFilter instantiation site ([aec.py:4011-4015](../python/aec.py#L4011))
selects `ResFilterRefactored` when the flag is True; otherwise legacy
`ResFilter`. Smoke-test (5 cases via the flag path, not monkey-patch) confirms
byte-identical output to legacy:

```
0I0XMl3M0ECO0U1N0cJvpg_doubletalk           identical=True max|delta|=0.000e+00
49IIo03GZ0CYQOmeA3A0BA_doubletalk           identical=True max|delta|=0.000e+00
0I0XMl3M0ECO0U1N0cJvpg_farend_singletalk    identical=True max|delta|=0.000e+00
0KjzXA3g20qsd8zmSekADw_farend_singletalk    identical=True max|delta|=0.000e+00
014AzuqPZku2004NbTTmcA_nearend_singletalk   identical=True max|delta|=0.000e+00
```

Combined with B.4 (800-case monkey-patch path), this validates both the
class-level swap (verified at scale) and the production-flag swap path
(verified for correct wiring).

## Architecture decision: subclass-and-delegate, deferred ResState

All five module extractions follow a uniform pattern:

```python
class ResFilterRefactored(ResFilter):
    def _stage_residual_model(self, **kw):  return residual_estimator(self, **kw)
    def _stage_gain_compute(self, **kw):    return gain_computer(self, **kw)
    def _stage_gain_postprocess(self, **kw):return spectral_shaper(self, **kw)
    def _stage_temporal_smoothing(self, **kw):return temporal_smoother(self, **kw)
    def _stage_noise_floor_and_cng(self, **kw):return noise_floor_cng(self, **kw)
```

Each module is a free function taking `rf: ResFilter` as its first parameter
(adapter for `self`). State stays in the legacy class instance; the function
just relocates the **code body**, not the **state**.

`ResState` migration (§3.4) is **deferred** to a future phase. The B.1
inventory enumerates 25+ cross-module reads/writes of `self.*`; migrating
state in lockstep with logic extraction would require partial-state-
duplication during the transition, violating §3.5's "module-by-module
byte-equal" gate. The deferred approach preserves byte-equal at every step
and bounds the eventual `ResState` PR to state plumbing only (no logic
change), making it independently auditable.

## Documented deviations from v1.1 §3.3 mapping

`epc_dt_cap` (diag [2]) physically remains in Module 3 (`spectral_shaper.py`)
rather than relocated to Module 2 (`gain_computer.py`). The §3.3 logical
mapping is honored at the time-ordered sequence level (gain_compute →
epc_dt_cap → ...); code-locality shift is deferred to the `ResState`
migration pass when the orchestrator owns frame ingestion. Documented in
[p52_phase_b_module_2_verdict.md](p52_phase_b_module_2_verdict.md) §
"Deviation from v1.1 §3.3 mapping".

## Anti-loophole compliance (Phase B totals)

- **§5.5 (no RES logic change)**: verified at the 800-case sample level —
  zero drift, zero `|delta|>0` samples, zero cases > 0.1 dB mean drift.
- **§5.4 (shadow strict scope)**: `python/res_refactored/*` has no
  reference to `PathChangeRegimeHandler`, `AcousticRegimeClassifier`, or
  any shadow state.
- **§6.4 (branch isolation)**: `python/aec.py` change in Phase B is
  bounded exactly to (i) one new `AecConfig` field, (ii) a 4-line swap
  block at the ResFilter instantiation site. The branch did not touch
  `_stage_*` methods, `PBFDKF`, `ShadowFilter`, `PathChangeRegimeHandler`,
  `AcousticRegimeClassifier`, or `RenderSignalAnalyzer`.

## What this unlocks

- **Phase C** can now opt into Module 1-5 extraction by setting
  `use_res_refactored=True` in test configurations. Phase C Combined eval
  (§4.5) runs Baseline / A-only / B-only / Combined — `B-only` and
  `Combined` will set the flag, `Baseline` and `A-only` leave it default-
  OFF.
- A future phase can retire the legacy `_stage_*` methods from `ResFilter`
  once external integrations (C port, downstream tooling) confirm none
  depend on the in-class method names.
- The `ResState` plumbing migration becomes a focused follow-on that
  re-validates the same B.4 800-case bench.

## Cross-references

- Per-module verdicts: [p52_phase_b_module_{1,2,3,4,5}_verdict.md](.)
- B.4 verdict: [p52_phase_b_b4_verdict.md](p52_phase_b_b4_verdict.md)
- Inventory: [research_log_p52_phase_b_inventory.md](research_log_p52_phase_b_inventory.md)
- Anomalies: [phase_b_anomaly_notes.md](phase_b_anomaly_notes.md)
- Design lock: [p52_design_lock_v1.1.md](p52_design_lock_v1.1.md)
- Phase A closure: [p52_phase_a_verdict.md](p52_phase_a_verdict.md)
