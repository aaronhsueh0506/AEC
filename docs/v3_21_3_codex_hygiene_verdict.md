# v3.21.3 Codex hygiene verdict

**Date**: 2026-05-20
**Branch**: v3.21.3 HEAD
**Predecessor**: v3.21.2 (tagged 1b8246d, branch HEAD)

## Question

Codex review of v3.21.2 flagged 4 hygiene findings — 1 HIGH severity
(state-clearing bug across utterance boundaries) and 3 MEDIUM
(documentation drift, broken contract, dead code). Are they real, and
what's the bench impact of fixing them?

## Findings — all verified, all fixed

### Codex #1 (HIGH) — `AEC.reset()` doesn't clear AEC3 post-state

Verified via line-read of [python/modules/orchestrator.py:984-1121](../python/modules/orchestrator.py#L984-L1121).
`__init__` populates `_aec3_state` / `_aec3_ree` / `_aec3_sg` /
`_aec3_ola_buf` / `_aec3_noise_psd` / `_aec3_noise_initialized` /
`_aec3_smooth_cn_gain` / `_aec3_pending_*` / `_aec3_stationarity` +
counters; `reset()` body never touched them.

**Reachability**: smoke test confirmed `_aec3_noise_initialized` stays
True after `reset()` and `_aec3_ola_buf` retains non-zero samples in
the pre-fix code. Cross-stream reuse (typical NN training pipeline
pattern) carried prior-utterance state into the next utterance.

**Fix**: commit [`81a5103`](../python/modules/orchestrator.py).
`_reset_aec3_post()` helper. 5 new unit tests in
[python/test_aec_reset.py](../python/test_aec_reset.py) — all PASS.

### Codex #2 (MED) — `_reset_filter_derived_state()` stale + incomplete

Verified via line-read of [python/modules/orchestrator.py:1123-1171](../python/modules/orchestrator.py#L1123-L1171).
Docblock promised clearing "RES post-filter state (gain_smooth /
echo_psd / noise_psd / gates)" — that's the LEGACY ResFilter chain
retired in v3.21.0. Body never updated to clear the replacement
(AEC3 post chain).

**Reachability**: called from delay_first (orchestrator.py:1577),
delay_shift (orchestrator.py:1611), p3h_diverged (orchestrator.py:2821).
On 800-case bench, these fire on movement / EPC / divergence cases.

**Fix**: commit [`b2491a8`](../python/modules/orchestrator.py).
Docblock corrected; `_reset_aec3_post(preserve_render_side=True)`
added to the body. Render-side context (stationarity tracker,
non_zero_render_seen, active_hops counter) preserved per the helper's
input-side rule.

### Codex #3 (MED) — `return_res_context=True` dead contract

Verified via line-read of:
- [python/modules/config.py:864-865](../python/modules/config.py#L864-L865) — config field documented to return tuple
- [python/modules/orchestrator.py:1702](../python/modules/orchestrator.py#L1702) — `_res_context = None` (only assignment)
- [python/modules/orchestrator.py:3046-3047](../python/modules/orchestrator.py#L3046-L3047) — `if _res_context is not None: return (result, _res_context)` (dead branch)

Even with `return_res_context=True`, `process()` always returned just
`ndarray` because `_res_context` was never assigned anything non-None.

**Fix**: commit [`80da109`](../python/modules/orchestrator.py).
Populate `_res_context` with `AecResContext(...)` immediately after
`_aec3_post` returns. 2 new unit tests — both PASS. Default path
(`return_res_context=False`) byte-equal preserved.

### Codex #4 (MED) — legacy delay knobs are no-ops

Verified via line-read of:
- [python/modules/delay/legacy_compat.py:21-22](../python/modules/delay/legacy_compat.py#L21-L22) — shim explicitly documents `_period_samples` / `_alpha` as "no-op compat attributes"
- [python/modules/delay/legacy_compat.py:39-41](../python/modules/delay/legacy_compat.py#L39-L41) — `_legacy_kwargs` collected but never consumed
- [python/modules/orchestrator.py:1502-1524](../python/modules/orchestrator.py#L1502) — 17-line conditional wrote to these no-op attrs
- run_one_case.py uses `getattr(..., None)` for `_trace_rows` which the shim never populates → CSV write silently skipped

**Fix**: commit [`fd2cfcd`](../python/modules/orchestrator.py).
5 config fields + 17-line conditional + 2 env var hooks + 5 reference
sites removed (72-line delta). Byte-equal preserved (provably dead).

## Bench impact (800-case AECMOS vs v3.21.2 baseline)

| Bucket | Δecho | Δdeg |
|---|---:|---:|
| FS_static | −0.050 | +0.000 |
| FS_movement | −0.037 | +0.000 |
| DT_static | −0.028 | **+0.025** |
| DT_movement | −0.032 | **+0.026** |
| NE | +0.000 | +0.000 |

### Attribution

- **Codex #1, #3, #4**: zero bench impact (Codex #1 path not exercised
  on fresh-instance-per-case bench; #3 gated by default-OFF config
  flag; #4 provably dead).
- **Codex #2**: is the source of the entire Pareto shift.

### Codex #2 mechanism (why DT_deg up + FS_echo down)

Pre-fix: filter resets via delay_first / delay_shift / p3h_diverged →
PBFDKF taps go to zero. AEC3 post chain keeps prior ERLE / R² /
per-bin smoothers. SuppressionGain sees "we have high ERLE" → applies
confident suppression to the noisy output of the freshly-reset
(un-trained) filter. This produces:

- Aggressive suppression masks the un-trained filter's poor cancellation
  → FS_echo metric looks better than reality.
- Same aggressive suppression also fires on DT speech that's mistakenly
  classified as residual echo → DT_deg degraded.

Post-fix: filter resets AND AEC3 post chain resets together. Fresh
ERLE = min_erle → SuppressionGain backs off until filter re-trains.
This produces:

- Honest FS_echo metric — no fake confidence boost during recovery
  transients. Bench loses the previously-illusory FS gain.
- DT speech preserved during recovery — no over-aggressive suppression
  → DT_deg metric improves.

The Pareto shift exposes the bug as a *correctness vs. metric trade*:
the pre-fix metric advantage on FS was extracting damage from DT in
a way that the AECMOS metric correctly penalised.

## Verdict

**PASS** — all 4 findings verified real, fixed, and tested. Codex #2
introduces a real Pareto shift; framed and shipped as a correctness
improvement rather than a metric regression. v3.21.3 ships.

## Files

| Artifact | Path |
|---|---|
| Codex #1 commit | [`81a5103`](../python/modules/orchestrator.py) + [test_aec_reset.py](../python/test_aec_reset.py) |
| Codex #2 commit | [`b2491a8`](../python/modules/orchestrator.py) |
| Codex #3 commit | [`80da109`](../python/modules/orchestrator.py) + [test_aec_reset.py](../python/test_aec_reset.py) |
| Codex #4 commit | [`fd2cfcd`](../python/modules/orchestrator.py) |
| v3.21.3 baseline | [docs/bench/v3_21_3_baseline/](bench/v3_21_3_baseline/) |

## Next sprints (v3.21.4+ carry-overs)

Still in v3.21 AEC3-alignment arc; ship as future v3.21.x:

- 3 HIGH time-domain unit-conversion bugs (`trigger_threshold` /
  `hold_duration` / `noise_floor_hold` ported as bare ints from AEC3
  4 ms blocks into our 10 ms hops). Direction opposes FS recovery —
  needs careful Pareto management.
- ReverbDecayEstimator full port (`AnalyzeFilter` +
  `EarlyReverbLengthEstimator` + validation gates — ~270 LOC missing
  vs AEC3's 411).
- U4.A v3.21.1 per-bin H_error refresh re-test on canonical state.
- U4.B B3 intermediate value (1000 / 1500 Hz) sweep.
