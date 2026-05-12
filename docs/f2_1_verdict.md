# F2.1 verdict — EPC state reset NEGATIVE (2026-05-12)

## TL;DR

**F2.1 (reset `_erl_estimate` / `_erle_window_*` / `_wn_err_baseline` on EPC fire) closes as FAIL.** The hypothesis that erl_estimate staleness caused the F3.1 FS_movement outlier (Lsa5Wpw Δecho −0.301) is **refuted** by data: F2.1 alone barely moves Lsa5Wpw (Δ −0.011), and F3.1+F2.1 leaves it at Δ −0.287 ≈ same as F3.1 alone. Worse, F2.1 makes the FS bucket regress more under F3.1.

Keep `feature/f3-1-mic-excess-evidence` open with F2.1 as default-OFF research substrate (not productised). F3.1 alone remains GREEN-PASS slim per `docs/f3_1_phase1_verdict.md`.

## 800-case j4 BALANCED bench — 4 conditions

### Bucket-mean deltas vs main baseline

| Bucket | F3.1 alone | F2.1 alone | F3.1+F2.1 |
|---|---:|---:|---:|
| FS_static Δecho   | −0.0011 | −0.0028 | **−0.0051** |
| FS_movement Δecho | −0.0055 | −0.0078 | **−0.0151** |
| DT_static Δecho   | +0.0003 | −0.0005 | −0.0014 |
| DT_movement Δecho | +0.0017 | −0.0004 | +0.0000 |
| NE Δecho          | +0.0000 | +0.0000 | +0.0000 |
| FS_static Δdeg    | +0.0000 | +0.0000 | +0.0000 |
| FS_movement Δdeg  | +0.0000 | +0.0000 | +0.0000 |
| DT_static Δdeg    | +0.0006 | +0.0002 | +0.0010 |
| DT_movement Δdeg  | −0.0030 | +0.0012 | +0.0003 |
| NE Δdeg           | +0.0000 | +0.0000 | +0.0000 |

All bucket means still inside `bench_aecmos.py` bars (FS echo ≥ −0.02, NE deg ≥ −0.01). FS_movement −0.0151 inches closer to the bound than F3.1 alone but does not breach.

### |Δ|>0.05 case count (F3.1+F2.1 vs main)

| Bucket | echo regress | echo gain | deg regress | deg gain |
|---|---:|---:|---:|---:|
| FS_static   | **6** | 2 | 0 | 0 |
| FS_movement | **9** | 1 | 0 | 0 |
| DT_static   | 2 | 0 | 3 | **7** |
| DT_movement | 0 | 0 | 2 | 3 |
| NE          | 0 | 0 | 0 | 0 |

F3.1 alone had 5 FS_movement regress / 1 gain. F3.1+F2.1 has 9/1 — **F2.1 adds 4 more FS_movement regressors**. On the DT side it adds 3 more deg-gain cases (4→7).

### Lsa5Wpw FS_movement outlier (the case F2.1 was supposed to rescue)

| Condition | echo | Δecho vs main |
|---|---:|---:|
| main baseline | 3.880 | — |
| F3.1 alone    | 3.579 | −0.301 |
| **F2.1 alone**    | **3.869** | **−0.011** |
| F3.1+F2.1     | 3.593 | −0.287 |

**F2.1 alone barely touches Lsa5Wpw.** F3.1+F2.1 ≈ F3.1 alone. The erl-staleness hypothesis is refuted.

## Why F2.1 failed

Two design errors in the reset:

1. **`_erl_estimate` reset to 0.1 (init) is *more* conservative than the shipped `min(0.3)` cap.** With a lower erl, F3.1's metric `max(error − far_lw·erl, 0) / error` over-attributes mic energy to NE → over-protects → echo suppression weakens → FS echo MOS drops. This is the *opposite* of what was intended.
2. **`_erle_window_near` / `_erle_window_err` reset to 1e-10 erases useful ERLE history.** The windowed accumulators feed `erle_for_factor` via `erle_windowed = 10·log10((_erle_window_near + 1e-10) / (_erle_window_err + 1e-10))` (aec.py:5173). Zeroing them collapses `erle_factor`, which drops `base_over_sub` (aec.py:5181) → less suppression → more echo leak.

The shipped EPC code is more careful than the F2.1 plan gave it credit for: it caps erl rather than resetting, and leaves the erle windows to decay naturally with α=0.999. That is the *correct* balance — the F2.1 hard-reset is too aggressive.

Smoke trace on Lsa5Wpw confirms F2.1 fires 3 × `shadow_rise` resets, so the wiring is correct; the resets just don't produce the expected upstream improvement to F3.1's metric.

## Re-examining the Lsa5Wpw mechanism

Lsa5Wpw FS_movement frame budget (instrumented from the F3.1 Phase 1 audit, `docs/f3_1_phase1_verdict.md`):

- 2207 frames total (22.1 s of audio).
- F3.1 fires only **140 frames (6.3 %)** — `filter_converged` is False the other 93.7 %.
- Those 140 fires concentrate in short-lived "filter just-converged" windows that flank the movement transitions.

Even when erl_estimate is *reset* to init by F2.1, that init value still mis-matches the new room's true ERL for the entire post-reset settling period. The metric stays wrong; only the *direction* of the error shifts. Lsa5Wpw is fundamentally hard for any per-frame metric that depends on erl_estimate accuracy in motion regimes.

## Decision

1. **F2.1 is not productisable.** Closes; flag stays in code as default-OFF research substrate (matches P55 / P58 pattern).
2. **F3.1 stays GREEN-PASS slim** per `docs/f3_1_phase1_verdict.md`.
3. **Do not chain F2.1 onto F3.1** in any productisation candidate.
4. **F3.2 (per-bin effective_dt) is independent of erl_estimate accuracy** — it changes scalar→per-bin gates inside the gain compute stage, not the metric formula. Proceed to F3.2 as the next experimental arc. The FS_movement regression in F3.1 is a separate concern to address with a different mechanism (likely a tighter F3.1 gate, e.g. `not _epc_render_forced_remaining > 0`, or F3.1-skip-when-recently-paused).

## Followups (deferred, not blocking F3.2)

- Try a *softer* F2.1 variant: cap `_erl_estimate` to a value derived from the freshly-reset filter's instantaneous ERL rather than the init 0.1.
- Investigate the FS_movement worst cluster (`Lsa5Wpw`, `s0oJqM6Y...`, `wr54weK...`, `pG9Bikvr...`, `Khk1qeM...`) to see whether they share an acoustic signature (e.g. very-fast movement, near-end-of-clip transition) that F3.1 could explicitly gate against.

## Files / artefacts

- Code: `python/aec.py` + `python/eval_aec_challenge.py` + `tools/research/f3_1_bench_runner.py` (commit b6719b1)
- Unit tests: `python/test_f2_1_epc_state_reset.py` (4/4 pass)
- Bench output: `/tmp/f2_1_alone/` + `/tmp/f3_1_f2_1/` (800 + 800 WAVs)
- Scores: `/tmp/f2_1_alone_scores/{scores.json, result.md}` + `/tmp/f3_1_f2_1_scores/{scores.json, result.md}`
- Logs: `/tmp/f2_1_alone.log`, `/tmp/f3_1_f2_1.log`, `/tmp/f2_1_alone_score.log`, `/tmp/f3_1_f2_1_score.log`
