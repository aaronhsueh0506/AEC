# P52 Phase B Task B.1 — RES Stage Inventory

**Branch**: `feature/p52-phase-b-refactor` (forked from `main@8c20f6d`)
**Read-only deliverable.** No code changed in B.1.

## Source-of-truth pointers

- `ResFilter` class: [aec.py:1381](../python/aec.py#L1381)
- `_diag_round5_stages` index legend: [aec.py:1488-1491](../python/aec.py#L1488-L1491)
- `ResFilter.process()` orchestrator: [aec.py:2221-2480](../python/aec.py#L2221)

## Stage call sequence inside `ResFilter.process()`

| Order | Caller line | Method | `_diag_round5_stages` indices written |
|---|---|---|---|
| pre-stages | [2272-2360](../python/aec.py#L2272) | inline in `process()` — coh², EMAs, `far_activity`, `effective_dt` derivation | none (pre-stage state) |
| 1 | [aec.py:2363](../python/aec.py#L2363) | [`_stage_residual_model`](../python/aec.py#L1723) | none (writes `residual_echo_psd`, `reverb_psd`) |
| 2 | [aec.py:2440](../python/aec.py#L2440) | [`_stage_gain_compute`](../python/aec.py#L1826) | `[0]` softgate_emr, `[1]` spectral_floor |
| 3 | [aec.py:2451](../python/aec.py#L2451) | [`_stage_gain_postprocess`](../python/aec.py#L1964) | `[2]` epc_dt_cap, `[3]` quiet_mask, `[4]` 3bin_smooth, `[5]` hf_cap, `[6]` pre_temporal |
| 4 | [aec.py:2462](../python/aec.py#L2462) | [`_stage_temporal_smoothing`](../python/aec.py#L2094) | `[7]` post_temporal |
| 5 | [aec.py:2473](../python/aec.py#L2473) | [`_stage_noise_floor_and_cng`](../python/aec.py#L2151) | `[8]` after_noise_lift |

## Mapping to P52 v1.1 §3.3 Modules

| New module | Existing method(s) covered | `_diag` indices | Notes |
|---|---|---|---|
| **Module 1 `ResidualEstimator`** | `_stage_residual_model` | none | Delegates to `self._residual_est` ([aec.py:1570](../python/aec.py#L1570) — `ResidualEchoEstimator(n_freqs, mode='legacy')`), plus reverb-tail update [aec.py:1810-1818](../python/aec.py#L1810). |
| **Module 2 `GainComputer`** | `_stage_gain_compute` + first leg of `_stage_gain_postprocess` (epc_dt_cap, idx `[2]`) | `[0] [1] [2]` | v1.1 §3.3 says Module 2 covers softgate_emr / spectral_floor / epc_dt_cap. Existing code splits epc_dt_cap into `gain_postprocess`; B.3 must move it to Module 2 to honour the v1.1 mapping. |
| **Module 3 `SpectralShaper`** | remainder of `_stage_gain_postprocess` (quiet_mask, 3bin_smooth, hf_cap, pre_temporal) | `[3] [4] [5] [6]` | Stage `[6]` pre_temporal is the trace snapshot taken before Module 4; not a separate transform. |
| **Module 4 `TemporalSmoother`** | `_stage_temporal_smoothing` | `[7]` | Owns EMA + drop/rise rate limit. |
| **Module 5 `NoiseFloorAndCng`** | `_stage_noise_floor_and_cng` | `[8]` | Owns noise PSD adaptive EMA, noise-floor lift, CNG injection, OLA synth. |

## Module-spanning state (must become explicit `ResState` fields per v1.1 §3.4)

| Field | Init site | Read by | Written by |
|---|---|---|---|
| `echo_psd`, `error_psd` | [aec.py:1528-1535](../python/aec.py#L1528) | Modules 1, 2 | pre-stage inline in `process()` ([2316-2317](../python/aec.py#L2316)) |
| `reverb_psd` | [aec.py:1536](../python/aec.py#L1536) | Module 1 | Module 1 ([1810-1818](../python/aec.py#L1810)) |
| `_residual_est` (sub-state) | [aec.py:1570](../python/aec.py#L1570) `ResidualEchoEstimator` | Module 1 | Module 1 |
| `_coh2_smooth`, `S_fe`, `S_ff`, `S_ee` | init in `__init__` | pre-stage (coh²), Modules 2,3 (via `coh2` arg) | pre-stage inline ([2285-2300](../python/aec.py#L2285)) |
| `far_activity` | init in `__init__` | Module 2 (`effective_g_min`) | pre-stage inline ([2331-2336](../python/aec.py#L2331)) |
| `_diag_coh2_last`, `_dt_per_bin_last` | [aec.py:1483-1498](../python/aec.py#L1483) | downstream callers (audio-passive trace) | Modules 2, 3 |
| `gain_smooth` | [aec.py:1528](../python/aec.py#L1528) | Module 5 (noise-floor lift compare), trace | Module 4 ([2147](../python/aec.py#L2147)) |
| `noise_psd` | [aec.py:1537](../python/aec.py#L1537) | Module 5 (CNG noise gen, noise-floor lift) | Module 5 ([2166, 2173](../python/aec.py#L2166)) |
| `_smooth_cn_gain`, `ola_buf` | init in `__init__` | Module 5 | Module 5 |
| `input_buf` | init in `__init__` | pre-stage (window/FFT) | pre-stage inline ([2256-2257](../python/aec.py#L2256)) |

## Cross-module coupling that complicates the refactor

1. **Pre-stage inline block (`process()` lines 2256-2360)** — windowing, FFT, coh², EMAs, `far_activity`, `effective_dt`. **Not** in any `_stage_*` method. B.2 must decide whether this becomes Module 0 (`PreStage`) or stays in the orchestrator. Recommendation: keep in orchestrator, treat as "frame ingestion" — keeps each Module pure.
2. **`epc_dt_cap` placement** — currently in `_stage_gain_postprocess` but v1.1 §3.3 puts it in Module 2 (GainComputer). B.3 must move the cap block while preserving byte-equality. Order of operations inside `gain_postprocess` matters; `epc_dt_cap` runs first (writes `[2]` before quiet_mask writes `[3]`).
3. **`_residual_est` is a sub-object with internal mutable state** ([aec.py:1570](../python/aec.py#L1570)). B.2 has two options:
   - (a) wrap as a field inside `ResState` (`residual_estimator_state: ResidualEstimatorState`) — the `_residual_est` object is its own state container already; treat its `reset()` + `attribute()` methods as state-transition operators.
   - (b) inline the relevant logic — **rejected**: violates §5.5 ("Phase B may not change RES logic").
   B.3 will adopt (a).
4. **Diag write-back is per-module** ([1929, 1935, 1978, ...](../python/aec.py#L1929)) — each module mutates `self._diag_round5_stages[i]`. B.2 `ResState` should include a `diag_round5_stages: np.ndarray` field so modules write to a returned struct rather than `self`.
5. **`erle_factor` / `dt_indicator` / `effective_dt` / `epc_dt`** — derived in `process()` before Module 2 from upstream caller state (`AecState` etc.). These are **inputs** to Modules 2-5, not state. B.2 must pass them as function parameters, not as `ResState` fields.

## Recommended file layout for `python/res_refactored/`

(Aligned with v1.1 §6.2 file layout.)

```
python/res_refactored/
  __init__.py
  state.py                       # ResState dataclass; ResidualResult / GainResult value-types
  residual_estimator.py          # Module 1 — wraps existing ResidualEchoEstimator
  gain_computer.py               # Module 2 — softgate_emr + spectral_floor + epc_dt_cap
  spectral_shaper.py             # Module 3 — quiet_mask + 3bin_smooth + hf_cap
  temporal_smoother.py           # Module 4 — EMA + rate limit
  noise_floor_cng.py             # Module 5 — noise EMA + lift + CNG + OLA synth
  res_filter_refactored.py       # orchestrator (mirrors ResFilter.process line-for-line)
```

## Anomalies discovered during inventory (logged, not actioned)

- A1: `_stage_residual_model` does not write to `_diag_round5_stages` — only the gain-side stages do. The "9 stages" name in v1.1 §3.3 collapses Module 1 (residual estimation) and an implicit Module-0 pre-stage into a single conceptual stage; the gain pipeline alone is 9 named slots. Inventory documents this; no v1.1 doc change.
- A2: `_stage_gain_compute` reads `self.noise_psd` ([aec.py:1919-1920](../python/aec.py#L1919)) but `noise_psd` is owned by Module 5 — i.e. **Module 2 reads Module 5's previous-frame state**. This is a legitimate cross-module dependency (last-frame noise floor used in current-frame EMR computation). `ResState.noise_psd` must be passed as input to Module 2, not just held inside Module 5. Logged in `phase_b_anomaly_notes.md`.
- A3: `_stage_gain_postprocess` reads `self._residual_est._long_window_far_psd` ([aec.py:2030](../python/aec.py#L2030)). Module 3 reads internal state of Module 1's estimator. Must expose as an accessor (`residual_estimator.long_window_far_psd`) before B.3 Module 3 migration. Logged.

## B.1 verdict

Inventory complete. Mapping is well-defined; cross-module state is enumerated.
Two cross-module reads (A2, A3) are real dependencies that must be plumbed as
inputs in `ResState` / accessor methods — they are not bugs and not in scope
for Phase B (§5.5 forbids fixing them).

Ready to proceed to **Task B.2** (state container + module signature stubs).
