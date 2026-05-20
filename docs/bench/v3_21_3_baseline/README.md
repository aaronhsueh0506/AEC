# v3.21.3 AECMOS baseline — `balanced`

**Snapshot**: v3.21.3 release HEAD (= last Codex commit on v3.21.3 branch).
**Render config**: `--preset balanced --filter 832 --cng --parallel --workers 4`,
full 800-case AEC Challenge blind test corpus.
**Scoring**: AECMOS via `python/bench_aecmos.py`.

## What v3.21.3 changed vs v3.21.2

Codex hygiene cycle:

- **Codex #1 (HIGH)**: `AEC.reset()` now clears the AEC3 post-state
  (`_aec3_state` / `_aec3_ree` / `_aec3_sg` / OLA / CNG / stationarity
  tracker / pending events). Cross-stream reuse of an AEC instance no
  longer leaks previous-utterance post-filter state.
- **Codex #2 (MED)**: `_reset_filter_derived_state()` also clears the
  AEC3 post chain on filter recovery paths (delay_first / delay_shift /
  p3h_diverged). Preserves render-side context. The only behavior-
  changing fix in this release.
- **Codex #3 (MED)**: `AecConfig.return_res_context=True` contract
  implemented. `process()` now returns `(output, AecResContext)` as
  documented (was always returning ndarray due to dead `_res_context = None`).
- **Codex #4 (MED)**: removed 5 dead config fields + 17-line dead
  conditional + 2 env var hooks + 5 reference sites
  (`mov_rate_delay_est_enabled` family and `trace_delay_est` family
  became silent no-ops when v3.21 swapped legacy DelayEstimator for
  `LegacyDelayShim`).

See [CHANGELOG.md](../../../CHANGELOG.md) v3.21.3 entry for full
mechanism + Pareto attribution.

## Bucket means (v3.21.3 final)

| Bucket | n | echo | deg |
|---|---:|---:|---:|
| FS_static | 169 | 3.532 | 4.999 |
| FS_movement | 131 | 3.472 | 4.999 |
| DT_static | 186 | 4.161 | 2.506 |
| DT_movement | 114 | 4.135 | 2.511 |
| NE | 200 | 4.998 | 4.054 |

## Δ vs v3.21.2 baseline

| Bucket | Δecho | Δdeg |
|---|---:|---:|
| FS_static | −0.050 | +0.000 |
| FS_movement | −0.037 | +0.000 |
| DT_static | −0.028 | **+0.025** |
| DT_movement | −0.032 | **+0.026** |
| NE | +0.000 | +0.000 |

Pareto shift driven by Codex #2 (the other 3 fixes are no-impact on a
fresh-instance-per-case bench). The pre-fix behavior was extracting
illusory FS_echo gain via stale post-chain ERLE estimates during
filter recovery — Codex #2 returns the post chain to a state honestly
matching the freshly-reset filter, recovering DT speech that had been
over-suppressed.

## Δ vs v3.21.0 baseline (a537b65) — cumulative through v3.21.3

| Bucket | Δecho | Δdeg |
|---|---:|---:|
| FS_static | −0.197 | +0.000 |
| FS_movement | −0.154 | +0.000 |
| DT_static | −0.077 | **+0.119** |
| DT_movement | −0.080 | **+0.141** |
| NE | +0.000 | +0.003 |

## Reference points

| | FS_st echo | FS_mv echo | DT_st echo | DT_mv echo | DT_st deg | DT_mv deg | NE deg |
|---|---:|---:|---:|---:|---:|---:|---:|
| AEC2 | 3.48 | 3.48 | 4.26 | 4.26 | 2.39 | 2.39 | 4.10 |
| AEC3 | 3.88 | 3.88 | 4.54 | 4.54 | 1.85 | 1.85 | 3.45 |
| v3.21.0 (`a537b65`) | 3.729 | 3.626 | 4.237 | 4.215 | 2.387 | 2.371 | 4.052 |
| v3.21.2 | 3.582 | 3.509 | 4.188 | 4.166 | 2.481 | 2.485 | 4.054 |
| **v3.21.3** | 3.532 | 3.472 | 4.161 | 4.135 | **2.506** | **2.511** | 4.054 |

v3.21.3 DT_deg moves further above AEC2 reference (+0.11 on static,
+0.12 on movement); FS_echo remains within ~0.35 dB of AEC3 reference.
The Pareto position is consistently DT-favouring vs both reference
implementations.

## Files in this directory

| File | Purpose |
|---|---|
| `balanced_scores.json` | Per-case AECMOS scores (800 cases) |
| `balanced_result.md` | Bucket-mean summary + worst-20 per bucket |

## Reproducing this baseline

```bash
python3 python/eval_aec_challenge.py wav/aec_challenge_blind/ \
    --preset balanced --filter 832 --cng --parallel \
    -o out_python_v3_21_3/ --workers 4
python3 python/bench_aecmos.py out_python_v3_21_3/ results_v3_21_3/
```
