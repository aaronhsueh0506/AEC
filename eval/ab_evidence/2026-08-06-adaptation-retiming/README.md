# Blind A/B — adaptation-constant retiming (2026-08-06)

Evidence for CHANGELOG 4.0.0 item (17). Kept because the verdict ("neutral,
ships") is not reproducible from a summary line: without the per-case scores
there is no way to tell a genuinely flat result from one where two buckets moved
in opposite directions and cancelled in the mean.

## What was compared

Baseline and candidate differ **only** in four files under `python/modules/`,
swapped in and out around each render. `variant_diff/` carries the exact unified
diff; it is five constants and nothing else:

| constant | file | baseline | candidate |
|---|---|---|---|
| `alpha` | `orchestrator.py` | `0.95` | `0.95 ** (16000 / sample_rate)` |
| `_alpha_erl_tracking` / `_alpha_erl_converged` | `orchestrator.py` | `0.99` / `0.999` | `growth_rehop(·, 160, 16000, hop, sr)` |
| `alpha_power` | `filters.py` | `0.9` | `growth_rehop(0.9, 160, 16000, hop, sr)` |
| `_alpha_r` | `filters.py` | `0.95` | `growth_rehop(0.95, **256**, 16000, hop, sr)` |
| `alpha_attack` / `alpha_release` | `preprocessing.py` | `0.3` / `0.98` | `growth_rehop(·, **256**, 16000, hop, sr)` |

`detectors.py` is in the swap set but identical in substance between variants at
these two grids — it was retimed in the earlier batch (item 16) and is held
constant here so this A/B isolates item (17).

Note the two different reference hops (160 vs 256). They are not a typo: the
`_alpha_r` and saturation constants were authored against frame 512 / hop 256
(`e9cb383`, `243d67c`), so retiming them off the 10 ms reference would be wrong
by 1.6x.

## Run identity

| | |
|---|---|
| AEC checkout | `47f4485` + uncommitted retiming work |
| Corpus | AEC Challenge blind set, `wav/aec_challenge_blind` |
| Case list | `eval/manifest_90case_stems.txt` (90 cases, buckets below) |
| Config | `--preset balanced --filter 52 --cng` |
| Grids | 16000/256 (hop 128) and 16000/512 (hop 256) |
| Pre-align | **`NO_PREALIGN=1`** |
| Scorer | `python/bench_aecmos.py --baseline` |
| Harness | `run_ab.sh` (copy of the script that produced these files) |

`NO_PREALIGN=1` is mandatory and is the single most important line in
`run_ab.sh`. The eval driver's **default** is an offline GCC-PHAT pre-align
crutch, which hides exactly the class of timing error this change is about; a
run without it would be worthless here regardless of what the numbers said.

## Result

See `summary_table.md` for the full bucket table. Worst movement across both
grids: **Δecho −0.0169**, **Δdeg −0.0133**, against a ±0.05 abort band. Every
bucket passed.

## What this does NOT establish

- **Nothing about 48 kHz.** Both runs are 16 kHz. 48 kHz has a structural pass
  only (finite, stable, survives reset) — no native 48 kHz material exists in
  this repo, and upsampled 16 kHz carries no energy above 8 kHz, which is the
  band a 48 kHz grid exists to serve.
- **Nothing about the 800-case bench**, which was not re-run for this change.
- **Neutrality is "no harm", not "no effect".** These constants feed detectors
  and EMAs whose outputs are gates; on a speech corpus most of those gates sit
  in the same state either way. A corpus that exercises the gated paths harder
  could separate the two variants where this one does not.

## Files

| file | contents |
|---|---|
| `b2_{base,cand}_{g256,g512}.scores.json` | per-case `bucket`/`echo`/`deg` for all 90 cases, plus bucket means |
| `summary_table.md` | the bucket table above, generated from those four files |
| `variant_diff/*.diff` | exact baseline→candidate source diff |
| `run_ab.sh` | the harness, including the `NO_PREALIGN=1` invocation |
