# P52 Phase B Task B.4 — Full 800-case byte-equal verdict

**Date**: 2026-05-12
**Branch**: `feature/p52-phase-b-refactor`
**Commit**: pending (this doc + B.4 tool + verdict JSON reference)
**Status**: **PASS** (hard bar AND internal 100% target both met)

## Verdict at a glance

| Bar | Value | Result |
|---|---:|---|
| Hard bar §3.6 (≥ 99.99% within `atol=1e-6, rtol=1e-5`) | **100.0000%** | **PASS** |
| Hard bar §3.6 (zero cases > 0.1 dB voice-band mean gain diff) | **0 / 800** | **PASS** |
| Internal advisory target (100% exact match) | **100.0000%** | **PASS** |
| Sample size | 800 (full corpus) | — |
| Total samples compared | 325,123,520 | — |
| Cases byte-identical (`np.array_equal`) | **800 / 800** | — |

## Methodology

Tool: [tools/research/p52_phase_b_b4_byte_equal.py](../tools/research/p52_phase_b_b4_byte_equal.py)

Two snapshot runs over the full 800-case AEC Challenge corpus:

| Tag | ResFilter class |
|---|---|
| `legacy` | Production `aec.ResFilter` (unchanged from `main@df2d2f8`) |
| `refactored` | `ResFilterRefactored` subclass with **all five** `_stage_*` methods delegated to functions in `python/res_refactored/` (Modules 1-5) |

Per-case `np.random.seed(42)` for CNG determinism. Config:
`balanced / fl=832 / cng=True / enable_shadow=True / mode=PBFDKF`. `j=4`.
Run wall time: legacy 532 s, refactored 539 s.

## Result detail

```
cases_total                            : 800
cases_byte_identical                   : 800
total_samples                          : 325,123,520
samples_exact_match                    : 325,123,520
samples_close_atol_1e-6_rtol_1e-5      : 325,123,520
fraction_exact                         : 1.0
fraction_close                         : 1.0
cases_over_0.1dB_mean_drift            : 0
```

Top-10 max-delta cases: all 0.0 (artefacts at `/tmp/p52_b4/verdict.json`).

## §B.4-supp — per-module byte-equal breakdown (gap closure)

Originally B.3 committed Modules 3-5 in a single commit (`3bd58c6`) and
B.4 measured only the cumulative all-five result. To rule out a
theoretical "compensating drift" scenario where one module's positive
drift cancels another's negative drift, supplementary 100-case isolated
runs were performed by building dynamic `ResFilter` subclasses overriding
only `M1..M{k}` for k ∈ {2, 3, 4, 5}.

| Configuration | Modules overridden | Cases byte-identical | Max abs delta |
|---|---|---:|---:|
| `M1_through_M2` | M1, M2 | **100 / 100** | 0.0 |
| `M1_through_M3` | M1, M2, M3 | **100 / 100** | 0.0 |
| `M1_through_M4` | M1, M2, M3, M4 | **100 / 100** | 0.0 |
| `M1_through_M5` | M1, M2, M3, M4, M5 | **100 / 100** | 0.0 |

Every incremental module addition introduces **zero drift independently**.
The compensating-drift theoretical bug is verified absent. Tool:
[tools/research/p52_phase_b_b4_isolated.py](../tools/research/p52_phase_b_b4_isolated.py).
Artefact: `/tmp/p52_b4/isolated.json`.

## Implication

The subclass-and-delegate refactor of all five RES modules introduces **zero
numeric drift** across the full 800-case bench. The 100-case sample result
from B.3 generalizes exactly to the full corpus — no edge-case cohort
behaviour emerged at scale.

`ResFilterRefactored` is now a **drop-in equivalent** to `ResFilter` at the
audio sample level. The B.5 config flag (`use_res_refactored: bool = False`)
can be wired into `AecConfig` and the production `AEC.__init__` swap site
without behavioural risk.

## Anti-loophole compliance (Phase B running totals)

- §5.5 (Phase B may not change RES logic): verified at the 800-case sample
  level — zero drift in any case, any sample.
- §5.4 (shadow filter strict scope): `python/res_refactored/*` does not
  reference `PathChangeRegimeHandler`, `AcousticRegimeClassifier`, or any
  shadow state.
- §6.4 (branch isolation): `python/aec.py` not modified by Phase B work
  to date; only `python/res_refactored/*`, `tools/research/p52_phase_b_*`,
  and `docs/p52_phase_b_*` touched.

## Next

**B.5**: wire `use_res_refactored: bool = False` flag into `AecConfig`;
modify the single ResFilter instantiation site in `AEC.__init__`
(aec.py:4007) to construct `ResFilterRefactored` when the flag is set.
This is the first and only `aec.py` change Phase B is permitted to make.
After B.5, run a final verification with `use_res_refactored=True` on
the same 800-case bench to confirm the flag wiring itself is correct.

## Cross-references

- Module verdicts: [p52_phase_b_module_{1,2,3,4,5}_verdict.md](.)
- Inventory: [research_log_p52_phase_b_inventory.md](research_log_p52_phase_b_inventory.md)
- Anomalies log: [phase_b_anomaly_notes.md](phase_b_anomaly_notes.md)
- Tool: [tools/research/p52_phase_b_b4_byte_equal.py](../tools/research/p52_phase_b_b4_byte_equal.py)
- Artefacts: `/tmp/p52_b4/{legacy,refactored}.npz`, `/tmp/p52_b4/verdict.json`
