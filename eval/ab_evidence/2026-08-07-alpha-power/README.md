# Blind A/B — alpha_power bypass fix (2026-08-07)

Evidence for `5232ab6`, which made the far-power EMA read the grid-retimed
`p->alpha_power` instead of a hardcoded `0.9f`.

## Result: byte-identical output. Not "neutral" — provably zero effect.

180 case-grid pairs, all byte-identical, decided by SHA-256 per output file:

| grids | cases | byte-equal | worst \|diff\| | record |
|---|---:|---:|---:|---|
| 16k/256, 16k/512 | 90 each | 180/180 | 0.0 | `wav_comparison_g{256,512}.json` |

**The byte-identical claim rests on those digests, not on the AECMOS deltas.**
An earlier version of this file inferred it from every bucket delta being
exactly `+0.0000`, which is also what a broken harness produces — and this
harness *had* been broken that way once: its first version reported "rendered
90" while all 90 renders had exited 2 on an invalid flag, and `bench_aecmos.py`
cheerfully writes a well-formed `scores.json` for "Scoring 0 cases".

A zero delta was therefore investigated rather than reported:

- the two binaries differ (`cmp` confirms; the harness now aborts if they do
  not);
- the harness rendered 90/90 and the scorer saw 90 cases at each grid, and the
  rendered / scored / manifest stem **sets** are compared for equality, not just
  their sizes;
- the retimed coefficient genuinely differs from the literal at every grid
  (0.845 / 0.919 / 0.845 / 0.894 against 0.9), and `test_rate_structural` check
  (d4) measures the coefficient the EMA *actually applied*, so the new value
  provably reaches the loop.

**The real reason is that `power[]` has no consumer.** In both ports it is
written (init, reset, cold-start `memcpy`, the EMA itself) and read at exactly
one place — its own cold-start guard:

```c
for (int k = 0; k < K; ++k) pwr_sum += p->power[k];
if (pwr_sum < 1e-10f && far_psd_sum > 1e-10f) { /* cold start */ }
```

Nothing downstream reads it. `sum(power)` crosses `1e-10` on the first hop with
far-end energy under any coefficient in `(0,1)`, so the branch decision is
identical and the array's values never influence anything else. Verified by grep
across `c_impl/src/` and `python/modules/filters.py`.

## What this means for the fix

`5232ab6` is still correct and should stay: it removes a real Python/C
divergence, and it removes a trap where a field is computed, asserted on by a
test, and ignored by the code. But it is a **consistency fix, not an audio
change**, and the original commit message overstated it by saying "the audio
paths did not [match]" — the stored values diverged; no audio consumed them.

Consequence for the campaign: `alpha_power` needs C/Python parity and golden
coverage, and it does not need AECMOS. Removing `power[]`, its coefficient, the
EMA and the pool allocation outright is a separate bit-exact cleanup — it changes
the PBFDKF public struct layout, so it is not folded into a timing change.

`alpha_r` is a different case again — see `../2026-08-07-alpha-r/README.md`. It
is inert through every `Aec` path (for two different reasons depending on shadow
mode) but is a **live** adaptation constant for a direct `pbfdkf_process()`
caller, so unlike `alpha_power` it is neither audio-dead in general nor
removable.

## Run identity

| | |
|---|---|
| Baseline | `d19e90f` (`5232ab6^`), built in its own git worktree |
| Candidate | `5232ab6`, built in its own git worktree |
| Engine | **C** (`aec_wav`), not the Python driver |
| Corpus | `wav/aec_challenge_blind`, `eval/manifest_90case_stems.txt` (90 cases) |
| Config | `--preset balanced --cng`, `--fft-size {256,512}` |
| Grids | 16000/256 (hop 128), 16000/512 (hop 256) |
| Harness | `eval/run_c_ab.sh` (canonical; not copied per evidence directory) |

```bash
eval/run_c_ab.sh <wt-d19e90f> <wt-5232ab6> <out> "256 512"
```

Baseline and candidate are separate **worktrees**, each built from its own
checkout with `make clean` and `WERROR=1`, so no build can pick up the other
side's sources.

**The engine matters here.** `eval_aec_challenge.py` drives the *Python* AEC
(`from aec import AEC`), so it cannot observe a C-only change at all — it would
have reported a zero delta for a completely different reason and the distinction
would have been invisible. `run_c_ab.sh` drives the C CLI instead. That gap is
also why a C/Python divergence could sit in `main` with every benchmark green:
**the repo had no C-path benchmark.**

`NO_PREALIGN` does not apply: it gates the Python driver's offline GCC-PHAT
crutch. `aec_wav` has no pre-align path and always runs the online estimator,
which is the stricter condition.

## Files

| file | contents |
|---|---|
| `{base,cand}_g{256,512}.scores.json` | per-case `bucket`/`echo`/`deg`, 90 cases each |
| `wav_comparison_g{256,512}.json` | per-case SHA-256, sample count, sample rate, max/RMS diff — the byte-identical claim |
| `source.diff` | the exact baseline→candidate change |
