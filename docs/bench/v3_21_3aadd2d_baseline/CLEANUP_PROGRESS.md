# v3.21 release cleanup progress (post-compact resume notes)

**Working dir**: `/Users/mingyu/Desktop/novatek/SE/AEC`
**Branch**: `feature/v3.21-aec3-clean`
**Baseline anchor**: commit `3aadd2d` (v3.21 NE fix lag-correct output limiter)
**Cleanup head** (as of compact): commit `ceb9ead` (R5 done)
**Byte-equal harness**: `python3 python/check_byte_equal.py` — must report `=== 25/25 PASS, 0 FAIL ===` after every cleanup round before commit
**Single production preset**: `BALANCED` (which is what `BALANCED_AEC3` was at baseline; `use_aec3_residual` field removed since AEC3 chain is now the only path)

## What the user asked for

> "balanced_aec3 設定成 default, 把其他 preset 移除, 現在這個分數先 commit, 開始整理 code,
> 維持現在 balanced_aec3 結果, 每種 case 各挑幾個出來做 byte equal 對比. 以這份 python
> 跟 docs 要 release 出來的標準來整理 code"

Translation of intent:
- AEC3 chain is the production ship target (preset `balanced` post-rename)
- Other 5 presets (mild / soft / aggressive / maximum / legacy BALANCED) are slated for removal
- Cleanup must preserve byte-equal output (verified via 25-case md5 reference)
- Aggressive cleanup pass to release-ready Python + docs

## Commits done (newest first)

| Commit | Round | Summary | LOC Δ | byte-equal |
|---|---|---|---|---|
| `ceb9ead` | R5 | prune dead local-var prep + dead self.res state writes in process() | +41 / -139 | 25/25 PASS |
| `28ef604` | R4 | drop dead `else: self.res.process(...)` branch (collapse to `_aec3_post(...)`) | +1 / -21 | 25/25 PASS |
| `97509c3` | R3 | remove `use_aec3_residual` config field + 2 runtime if-gates + env-var hook | +15 / -33 | 25/25 PASS |
| `6267de0` | R2 | drop legacy BALANCED config, rename BALANCED_AEC3 → BALANCED | +9 / -98 | 25/25 PASS |
| `c07d428` | R1 | delete MILD/SOFT/AGGRESSIVE/MAXIMUM presets (enums.py / config.py / CLI / scripts) | +10 / -127 | 25/25 PASS |
| `a24d154` | Phase A | 800-case AECMOS baseline + byte-equal reference for 25 cases + check_byte_equal.py harness + docs/bench/v3_21_3aadd2d_baseline/ | +8684 / -0 | — |

## What's left (R6 → final 800-case verify)

### R6 (in_progress at compact time): extract `_residual_est` from ResFilter to AEC

`self.res` (ResFilter instance) is currently retained because 6+ sites in `orchestrator.py` still read `self.res._residual_est` or `self.res.error_psd/echo_psd/near_psd`. After moving `_residual_est` ownership to `AEC.__init__` (replace `self.res._residual_est` reads with `self._residual_est`), and similarly moving `error_psd / echo_psd / near_psd / far_activity` to AEC, the ResFilter instance can be retired in R7.

Specific touch points in `python/modules/orchestrator.py`:
- L672-678 — substrate flag writes (`_arc_t_force_render_or_in_enabled` / `_c_e_branch_force_render_use_fq_usable`)
- L2636-2639 — high-band F3.1 metric read
- L2720-2723 — trace_high_band_metrics extras
- L2894-2896 — `_arc_t_cohort_tail_signal` write
- L2985 — `per_bin_eer = self.res.echo_psd / self.res.error_psd` (active path — NOT gated by `if self.res:`)
- L3047-3049 — trace_high_band_metrics (`error_psd / near_psd / _long_window_far_psd`)

Open question for R6: should the `_residual_est` continue to maintain `_long_window_far_psd` EMA when `res.process()` is no longer called? Currently it doesn't update (stays at zero) and consumer sites gate on `_long_window_n_updates > 0` / `>= 100` so they never fire. If we want this signal to be live again, we'd need to call its `attribute()` from `_aec3_post` or hoist the EMA out as its own helper.

### R7: delete ResFilter class entirely

After R6 the `self.res` instance is unused. Delete:
- `python/modules/res_filter.py` (file)
- `python/modules/res_refactored/` (entire subpackage — P52 Phase B subclass-and-delegate scaffold that never landed)
- `aec.py` L45 re-export `from modules.res_filter import ResFilter, ResFilterEnr, ResFilterWiener`
- `aec.py` L13 docstring mention
- Update `python/test_f3_1_mic_excess.py:30` (imports `from aec import ResFilter` — either rewrite the test or delete if obsolete)

### R8: delete legacy modules + audit untracked

- Delete `python/modules/legacy_state.py` (AecState legacy aggregator; AEC3 chain uses `modules/state/aec_state.py` instead)
- Untangle `self._aec_state` (legacy) from `self._aec3_state` (AEC3) in orchestrator
- The 2 untracked baseline-substrate files (`python/modules/state/filter_analyzer.py`, `python/modules/state/subtractor_output_analyzer.py`) are UNREACHED by current tracked code per import audit done before R5 (state/__init__.py does NOT import them; orchestrator does not use them) — **decision pending**: delete (since R7+R8 removes their downstream-substrate motivation) OR commit as research substrate

### R9: delete `python/modules/legacy_delay.py` if safe

Currently re-exported via `aec.py:33` as `DelayEstimator`. Production orchestrator uses `delay.legacy_compat.LegacyDelayShim` (renamed). Check if any external caller relies on `from aec import DelayEstimator`.

### R10: dead config knob sweep

`python/modules/config.py` likely has ~30-40 fields no longer read after R1-R9. Earlier audit (Round 7 of prior cleanup attempt) identified candidates: `dtd_geigel_threshold`, `dtd_hangover_frames`, etc. Sweep + remove. Also dead methods / dead imports across `dataclasses.py` / `detectors.py` / `dtd.py` / `preprocessing.py`.

### Phase D — release docs + version bump + final 800-case

- D-1: update `CLAUDE.md` (pipeline diagram = AEC3 chain only; preset = single BALANCED), `docs/aec_methods.md`, `docs/refactor_modules_layout.md`
- D-2: `CHANGELOG.md` v3.21.0 entry
- D-3: bump `__version__` in `python/aec.py` from `3.15.0` → `3.21.0`
- D-4: full 800-case bench on cleaned tree, compare per-case against `docs/bench/v3_21_3aadd2d_baseline/balanced_aec3_scores.json`. Must match within ±0.001 dB (byte-equal → exact). Then tag `v3.21.0`.

## Critical: byte-equal gating discipline

After every code change, run `python3 python/check_byte_equal.py`. Must report `25/25 PASS, 0 FAIL` before commit. If a round fails the check, investigate the diverging case (script prints expected vs got md5 per stem) — usually due to accidentally touching code that's actually live for that case. Roll back partial edits as needed.

## Baseline AECMOS bucket means (for sanity vs final bench)

Single preset `BALANCED` (was `BALANCED_AEC3` at baseline):

| Bucket | n | echo (↑) | deg (↑) |
|---|---:|---:|---:|
| FS_static    | 169 | 3.729 | 4.999 |
| FS_movement  | 131 | 3.626 | 4.999 |
| DT_static    | 186 | 4.237 | 2.387 |
| DT_movement  | 114 | 4.215 | 2.371 |
| NE           | 200 | 4.998 | 4.052 |

vs references: beats AEC3 on all deg buckets (+0.52 / +0.52 / +0.60), beats AEC2 on FS_static echo (+0.25), within ~0 of AEC2 on DT/NE bucket targets.

## Commit message style

Each round uses this pattern:

```
v3.21 cleanup R<N>: <one-line summary>

<paragraph of detail>

byte-equal <N>/25 PASS.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
```

## File index for the byte-equal harness

- `python/check_byte_equal.py` — driver
- `docs/bench/v3_21_3aadd2d_baseline/byte_equal_reference.json` — 25 case md5 reference (sampled at echo percentile 0/25/50/75/100 within each bucket)
- `docs/bench/v3_21_3aadd2d_baseline/balanced_aec3_scores.json` — full 800-case baseline scores (anchor for final D-4 verify)
- `docs/bench/v3_21_3aadd2d_baseline/balanced_aec3_result.md` — bucket means + worst-20 per bucket
- `docs/bench/v3_21_3aadd2d_baseline/README.md` — overview + reproduction instructions
