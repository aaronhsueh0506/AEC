# Blind A/B — alpha_r 10 ms anchor + bypass fix (2026-08-07)

Evidence for `d7e94f7`.

## Result: byte-identical output, for the same reason as alpha_power

0 of 180 case-grid pairs changed; all 90 WAVs byte-identical at both grids.

**This corrects a claim made in `d7e94f7`'s own commit message.** That message
said "Unlike alpha_power, this one reaches audio", citing the chain
`error_psd -> e2_ref_sum -> use_converged -> leakage -> H_error_per_bin`. The
chain exists, but it is in a branch that production never takes. Two lines
above it, the source says so:

```c
} else {
    /* scalar fallback: e2_ref_sum = sum(error_psd) (float32); e2_coa scalar.
     * (Not exercised in production: orchestrator sets e2_coarse_per_bin
     * every hop. Kept pairwise-correct for completeness.) */
```

The live path is the per-bin branch, which computes `e2_refined_per_bin` from
the current frame's `error_spec` directly and never reads the smoothed
`error_psd`. The orchestrator sets `e2_coarse_per_bin` unconditionally every
hop (`orchestrator.py:1687,1784`; `aec.c:1959` sets
`e2_coarse_per_bin_valid = 1`), so the `else` is unreachable in production.
PBFDKF's `error_psd` therefore has no live consumer, and neither does `R`,
which the header already documents as "computed in the active path but no
longer feeds the H_error gain".

(The `error_psd` in `aec3_post.c` / `aec_state.c` is a different field on a
different struct -- `aec3_post.h:166` -- not PBFDKF's.)

I read the consumer and stopped there instead of checking whether its branch
runs. That is the same mistake class as trusting a struct field that nothing
reads, one level up.

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
