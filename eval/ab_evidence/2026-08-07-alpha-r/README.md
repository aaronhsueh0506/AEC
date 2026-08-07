# Blind A/B — alpha_r 10 ms anchor + bypass fix (2026-08-07)

Evidence for `d7e94f7`.

## Result: byte-identical output through the whole `Aec` API — but NOT dead

0 of 180 case-grid pairs changed (default config); a further 20-case run with
`--no-shadow` is also byte-identical. But "audio-dead" -- which an earlier
version of this file and of `d7e94f7`'s commit message both claimed -- is the
wrong conclusion, and so is "the scalar fallback is unreachable".

The precise picture, verified in source and empirically:

| path | `error_psd` read? | does `alpha_r` change output? | why |
|---|---|---|---|
| `Aec`, shadow ON (default) | no | no | the per-bin branch (`pbfdkf.c:958`) reads `error_spec` directly |
| `Aec`, shadow OFF | **yes** | **no** | scalar branch runs, but `e2_coarse_for_refresh` is only assigned inside `if (a->has_shadow)` (`aec.c:1955`), so it stays `0.0f`; `use_conv = (e2_ref_sum <= 0)` is constant-false for any positive `error_psd`, whatever `alpha_r` is |
| direct `pbfdkf_process()` | **yes** | **YES** | a caller that supplies its own `e2_coarse_for_refresh` makes the comparison meaningful, and `alpha_r` then selects the leakage rate |

So `alpha_r` is a **live adaptation constant of the public PBFDKF API** that
happens to be inert through every path the `Aec` wrapper can reach — for two
*different* reasons in the two shadow modes. It is not dead, and it must not be
removed or left un-retimed; the 10 ms anchor matters for any integrator driving
PBFDKF directly, which is a supported entry point (`pbfdkf.h` documents it).

Two corrections this supersedes:
- `d7e94f7`'s "Unlike alpha_power, this one reaches audio" -- it does not,
  through `Aec`.
- This file's earlier "AUDIO-DEAD ... the else is unreachable" -- the else IS
  reachable with `--no-shadow`; it is *dominated*, which is a different claim
  and does not extend to direct-API callers.

## What this means

`d7e94f7` stays. The anchor correction (16 ms -> 10 ms, TC 311.93 -> 194.96 ms)
is right on the evidence, and wiring the live path to the retimed field removes
the same computed-but-ignored trap as alpha_power. But it is a consistency fix
with **no audio effect**, not an audio change.

Both bypass constants are therefore AUDIO-DEAD, and both belong in the
auxiliary / dead-output batch: C/Python parity and golden coverage, no AECMOS.

Worth carrying into the remaining batches: "the constant is read somewhere" is
not the same as "the read is reachable". For each remaining retime candidate,
the branch containing the consumer has to be shown live before its A/B result
means anything -- otherwise a zero delta gets misread as "the corpus did not
exercise it" when the real answer is "no corpus can".

## Run identity

| | |
|---|---|
| Baseline | `73ff8db` (`d7e94f7^`), own git worktree |
| Candidate | `d7e94f7`, own git worktree |
| Engine | **C** (`aec_wav`) -- `eval_aec_challenge.py` drives Python and cannot see a C-only change |
| Corpus | `wav/aec_challenge_blind`, `eval/manifest_90case_stems.txt` (90 cases) |
| Config | `--preset balanced --cng`, `--fft-size {256,512}` |
| Grids | 16000/256 (hop 128), 16000/512 (hop 256) |
