# v3.21.5 Phase 1 Sprint C — Reverb AEC3 semantic audit verdict

**Date**: 2026-05-21
**Branch**: `feature/v3_21_5_phase1_aec3_parity`
**Status**: **CLOSED — no Phase 1 ship; deferred to Phase 2 Sprint F (RES fallback) + Sprint H (upstream parity)**

## Context

Codex Finding 4 (verified): `ReverbFrequencyResponse.update()` has 3 early-return paths in [python/modules/residual/reverb_frequency_response.py:61-65](../python/modules/residual/reverb_frequency_response.py#L61):

```python
if stationary_block or linear_filter_quality is None:
    return                          # cause (1)/(2)
n_partitions = int(frequency_response.shape[0])
if filter_delay_blocks < 0 or filter_delay_blocks >= n_partitions:
    return                          # cause (3)
```

Phase 1 Sprint C scope: diagnose which early-return path fires + ship only AEC3-source-aligned semantic fix. Fallback (RES-internal divergence) defers to Phase 2 Sprint F.

## Sprint C.0 — Reachability check (cohort tail 5 FS worst cases)

Trace data from `/tmp/sprintA0/` 5 cases × ~2300 frames each (Sprint A.0 cohort). Aggregated `reverb_freq_resp_tail_max` per frame.

| Stem | tail_max mean | tail_dead % |
|---|---:|---:|
| `pcb1Nh0Z3k0WS9a7gBEuqg_farend_singletalk` | 0.0000 | **100%** |
| `LN18k5r8t00C9DulUd809A_farend_singletalk` | 0.0000 | **100%** |
| `s90M7MOTBkqaV4nQPLhKbA_farend_singletalk` | 0.0000 | **100%** |
| `9xjhiFbGo06hdQIsHTS6qA_farend_singletalk` | 0.8786 | 10.2% |
| `lV0kQN0hR0ySmE0bQhuYbw_farend_singletalk` | 0.2601 | 6.8% |

**Aggregated**: 7354/11338 = **64.9%** of frames have reverb tail dead (`tail_max = 0`).

C.0 gate (per plan): `> 50%` → confirm hypothesis, proceed to C.1.

## Sprint C.1 — Early-return cause attribution

Of the 7354 dead frames:

| Cause | Trace signal | Rate |
|---|---|---:|
| (1) `stationary_block=True` | `reverb_stationary_block` True | 17.9% |
| (2) `linear_filter_quality is None` | `reverb_filter_q_present` False | **90.5%** |
| (3) `filter_delay_blocks` out of range | `reverb_delay_within_partitions` False | **96.2%** |

(causes overlap; a dead frame can trigger multiple)

Per-case `delay_in_range%` shows bimodal pattern:
- pcb1Nh0Z3 / LN18k5r8 / s90M7MOT: `delay_in_range = 0%` (delay completely inactive)
- 9xjhiFbG / lV0kQN0h: `delay_in_range = 97-99%` (delay almost always in range, but other causes dominate)

Per-case `filter_q_None%`: 66-100% — `_filter_q` is None most of the time on all cases. From [orchestrator.py:3447](../python/modules/orchestrator.py#L3447):
```python
_filter_q = (1.0 if (_aec3_converged and _filter_converged_enough)
             else None)
```
Returns None unless BOTH conditions hold; on cohort tail FS cases, one or both is often False.

## Sprint C.2 — Direct-path delay semantic fix (would have been Phase 1 ship)

Per plan: if cause (3) dominates AND `aec_state.min_direct_path_filter_delay()` (AEC3 canonical source) is consistently in-range while our current source (`current_delay // hop_size`) is wrong → RES-internal swap is port fidelity → Phase 1 ship.

**Trace `aec3_min_direct_path_blocks` per case**: **0 across all 5 cases × all frames**.

Root cause per [python/modules/state/filter_delay.py:57-60](../python/modules/state/filter_delay.py#L57): when `analyzer_filter_delay_estimates_blocks=None` (current usage), `_filter_delays_blocks` falls back to `[_delay_headroom_blocks]` (typically 0). So `min_direct_path_filter_delay()` returns 0 essentially always — the AEC3 canonical API is not fed valid data because the FilterAnalyzer port is incomplete.

**Both candidates for `_delay_blocks` are degenerate (0)**:
- Current: `current_delay // hop_size` → -1 (delay inactive on 3/5 cases) or 0 (in-range but degenerate)
- AEC3 canonical: `min_direct_path_filter_delay()` → 0 always

Swapping doesn't help. The reverb update path is broken upstream of RES; no RES-internal AEC3-semantic fix exists.

## Sprint C.3 — Reverb tail dead fallback (RES-internal divergence)

**NOT a Phase 1 fix.** Adding `if tail_dead for N frames: use conservative S²/X² floor` is intentional divergence beyond AEC3 (AEC3 assumes complete analyzer; we lack it).

**Deferred to Phase 2 Sprint F** per plan boundary discipline. Will be revisited as v3.22 Sprint F.

## Verdict

**Phase 1 Sprint C: CLOSED — no ship.**

Reasoning:
- C.0 reachability confirmed: 64.9% frames dead
- C.1 attribution: dominant causes are upstream (filter_quality is None, filter_delay analyzer inactive)
- C.2 RES-internal AEC3-canonical swap (`aec_state.min_direct_path_filter_delay()`) returns 0 always → swap is no-op
- Phase 1 acceptance bar: "change must be traceable to AEC3 source line OR restore AEC3 default behavior". No such fix exists within RES scope here.
- Phase 2 Sprint F (RES-internal dead fallback) + Sprint H (upstream filter_quality / filter_delay analyzer port) carry this forward.

## Evidence trail (for v3.22 reference)

Trace CSVs at `/tmp/sprintA0/*.csv` (5 cohort tail cases, full 22 new trace fields). Aggregation script in this verdict's authoring session.

Key reproduction:
```bash
git checkout feature/v3_21_5_phase1_aec3_parity
# For 1 case:
python3 python/run_one_case.py wav/.../FS_case_mic.wav wav/.../FS_case_lpb.wav \
  /tmp/out.wav --preset balanced --trace-hf-chain /tmp/trace.csv
# Trace contains reverb_fallback_active, reverb_freq_resp_tail_max,
# reverb_delay_blocks, reverb_delay_within_partitions,
# reverb_stationary_block, reverb_filter_q_present,
# aec3_min_direct_path_blocks
```

## v3.22 carry-over

- **Phase 2 Sprint F**: implement RES-internal `reverb_tail_dead_fallback` opt-in flag. When `tail_max == 0` for N consecutive frames, fall back to conservative `render_psd * ep_gain` floor. Avoids leaving reverb model permanently empty on incomplete-port path.
- **Phase 2 Sprint H.1**: port AEC3 `FilterAnalyzer.consistent_estimate` to feed `linear_filter_quality` properly (or widen our scalar to fire after warmup).
- **Phase 2 Sprint H.2**: port AEC3 `FilterAnalyzer.FilterDelayBlocks()` per-channel per-block estimate to feed `analyzer_filter_delay_estimates_blocks`, so `min_direct_path_filter_delay()` returns meaningful values.
