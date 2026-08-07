# Blind A/B — alpha_r 10 ms anchor + bypass fix (2026-08-07)

Evidence for `d7e94f7`.

## Result: no output change through `Aec` in either shadow mode — but NOT dead

360 case-grid pairs, all byte-identical, decided by SHA-256 per output file:

| run | grids | cases | byte-equal | worst \|diff\| | record |
|---|---|---:|---:|---:|---|
| default config (shadow ON) | 16k/256, 16k/512 | 90 each | 180/180 | 0.0 | `wav_comparison_g{256,512}.json` |
| `--no-shadow` | 16k/256, 16k/512 | 90 each | 180/180 | 0.0 | `noshadow_wav_comparison_g{256,512}.json` |

Those files carry a per-case `base_sha256` / `cand_sha256` / `max_abs_diff` /
`rms_diff` / `sample_rate` / `sample_count`. **The byte-identical claim rests on
them, not on the AECMOS deltas** — an earlier version of this file inferred it
from every bucket delta being exactly `+0.0000`, which is also what a harness
that rendered nothing produces.

But "audio-dead" — which an earlier version of this file and of `d7e94f7`'s
commit message both claimed — is the wrong conclusion, and so is "the scalar
fallback is unreachable". The precise picture:

| path | `error_psd` read? | does `alpha_r` change output? | why |
|---|---|---|---|
| `Aec`, shadow ON (default) | no | no | the per-bin branch (`pbfdkf.c:958`) reads `error_spec` directly |
| `Aec`, shadow OFF | **yes** | **no** | the scalar branch runs, but `e2_coarse_for_refresh` is only assigned inside `if (a->has_shadow)` (`aec.c:1955`), so it stays `0.0f`; `use_conv = (e2_ref_sum <= 0)` is constant-false for any positive `error_psd`, whatever `alpha_r` is |
| direct `pbfdkf_process()` | **yes** | **YES** | a caller that supplies its own `e2_coarse_for_refresh` makes the comparison meaningful, and `alpha_r` then selects the leakage branch |

So `alpha_r` is a **live adaptation constant of the public PBFDKF API** that
happens to be inert through every path the `Aec` wrapper can reach — for two
*different* reasons in the two shadow modes. It is not dead, it must not be
removed or left un-retimed, and the 10 ms anchor matters for any integrator
driving PBFDKF directly, which is a supported entry point (`pbfdkf.h`).

Two corrections this supersedes:

- `d7e94f7`'s "Unlike alpha_power, this one reaches audio" — it does not,
  through `Aec`.
- This file's earlier "AUDIO-DEAD ... the else is unreachable" — the else IS
  reachable with `--no-shadow`; it is *dominated*, which is a different claim
  and does not extend to direct-API callers.

## How the constant is pinned instead

No wrapper-level A/B can observe this constant, so a zero delta from one is not
evidence about it either way. Two tests drive PBFDKF directly and measure the
coefficient the EMA **actually applied**, recovered from two runs that differ
only in `error_psd`'s starting value:

- `test_rate_structural` check (d5), all four grids;
- `python/tests/test_alpha_r_direct_pbfdkf.py`, same four grids, plus a
  wrapper-level assertion that moving `_alpha_r` 0.9598 → 0.10 leaves the AEC
  output bit-identical in **both** shadow modes — the scope claim above, as a
  measurement rather than a description.

Both are mutation-verified. Each of these must make them fail, and does:

| mutation | C result | Python result |
|---|---|---|
| use site back to the `0.95` literal | 22 checks fail | 19 tests fail |
| reference hop back to 256 (the introduce grid) | 21 checks fail | 13 tests fail |
| delete the EMA entirely | 4 checks fail | 20 tests error |

The reference-hop mutation is the one a "does the applied value match the stored
field?" test cannot catch, because both move together. It is caught by asserting
the wall-clock span itself: 194.957 ms, against the 311.932 ms a 16 ms reference
would give.

## What this means

`d7e94f7` stays. The anchor correction (16 ms → 10 ms, TC 311.93 → 194.96 ms) is
right on the evidence, and wiring the live path to the retimed field removes the
same computed-but-ignored trap as `alpha_power`. Through `Aec` it is a
consistency fix with no audio effect; through the public PBFDKF API it is a real
adaptation change, which is why it is pinned by a direct-API test and not by a
bucket average.

Worth carrying into the remaining candidates: **"the constant is read somewhere"
is not the same as "the read is reachable", and neither is the same as "the
read's result can vary".** All three have to be established separately. This
constant clears the first two and fails the third through the wrapper, and it
took two wrong readings to get there — the first checked whether the field was
read, the second checked whether the branch ran, and neither checked whether the
comparison it feeds could come out differently.

## Run identity

| | |
|---|---|
| Baseline | `73ff8db` (`d7e94f7^`), own git worktree |
| Candidate | `d7e94f7`, own git worktree |
| Engine | **C** (`aec_wav`) — `eval_aec_challenge.py` drives Python and cannot see a C-only change |
| Corpus | `wav/aec_challenge_blind`, `eval/manifest_90case_stems.txt` (90 cases) |
| Config | `--preset balanced --cng`, `--fft-size {256,512}` |
| Grids | 16000/256 (hop 128), 16000/512 (hop 256) |
| Harness | `eval/run_c_ab.sh` (canonical; not copied per evidence directory) |

```bash
# default config
eval/run_c_ab.sh <wt-73ff8db> <wt-d7e94f7> <out> "256 512"
# the shadow-OFF path, where the scalar fallback is reachable
eval/run_c_ab.sh <wt-73ff8db> <wt-d7e94f7> <out> "256 512" --no-shadow
```

Baseline and candidate are separate **worktrees**, each built from its own
checkout with `make clean` and `WERROR=1`, so no build can pick up the other
side's sources and a warning introduced by the candidate stops the run.

`NO_PREALIGN` does not apply: it gates the Python driver's offline GCC-PHAT
crutch. `aec_wav` has no pre-align path and always runs the online estimator,
which is the stricter condition.

## Files

| file | contents |
|---|---|
| `{base,cand}_g{256,512}.scores.json` | per-case `bucket`/`echo`/`deg`, 90 cases each, default config |
| `noshadow_{base,cand}_g{256,512}.scores.json` | the same under `--no-shadow` |
| `wav_comparison_g{256,512}.json` | per-case SHA-256, sample count, sample rate, max/RMS diff — the byte-identical claim |
| `noshadow_wav_comparison_g{256,512}.json` | the same for the `--no-shadow` run |
| `source.diff` | the exact baseline→candidate change |
