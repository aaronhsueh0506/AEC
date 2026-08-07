# Blind A/B — F2.4 simple-mu retiming (2026-08-07)

Evidence for `c4b7382`: the simple-mu holdoff (20 hops → 200 ms) and its three
retention alphas (attack 0.3, hold 0.99, release 0.95, all authored at a 10 ms
hop), retimed together as one atomic change.

## ⚠ The abort band fired. This verdict is OPEN.

| engine | grid | FS_static | FS_movement | DT_static | DT_movement | NE |
|---|---|---:|---:|---:|---:|---:|
| **Python** | 16k/256 | **−0.052** | −0.011 | +0.023 | −0.013 | +0.000 |
| Python | 16k/512 | +0.006 | +0.007 | +0.000 | +0.010 | +0.000 |
| C | 16k/256 | −0.032 | +0.021 | −0.023 | −0.008 | +0.000 |
| C | 16k/512 | −0.015 | +0.007 | +0.019 | +0.010 | −0.000 |

(Δecho. Worst Δdeg anywhere is −0.034, Python 16k/256 DT_static.)

**Python 16 kHz/256 FS_static Δecho = −0.052 is past the −0.05 abort band**, so
the run stopped for investigation rather than being reported as a pass. What
follows is that investigation. The decision to accept, re-tune or revert is not
made here.

## The investigation

**It is one clip, and the effect is real.** `HAxmF7v4dE0itSp5R5B3Dw_farend_-
singletalk` moves −0.257 on its own; the other four FS_static cases are +0.003,
+0.008, +0.009 and −0.025. With n=5 that single case carries the bucket mean.

It is not scorer noise. Reproduced deterministically through the driver's own
`run_ours()` entry point at the exact bench configuration, and measured in
physical units rather than MOS — for a far-end-singletalk clip the output *is*
residual echo, so its RMS is the quantity AECMOS-echo tracks:

```
                              residual-echo RMS, dB vs the all-frozen baseline
HAxmF7v4dE0itSp5R5B3Dw                          +0.44
```

Per second: 12 of 22 seconds worse by more than 1 dB, the worst by +10.7 dB.
Audible, not a rounding artifact.

**But the bucket as a whole is neutral, and partial retimes are worse.** Running
each constant alone against the same five clips separates the mechanism from the
outlier (`fs_static_bisect.txt`, `fs_static_bisect.json`):

```
stem                            all four   holdoff_limit    alpha_attack      alpha_hold   alpha_release
HAxmF7v4dE0itSp5R5B3Dw             +0.44           +1.17           -0.03           +1.29           +1.16
49DamGOwmUWGCn23bmI8xw             -0.12           -0.13           -0.16           -0.09           +0.04
1fvt8ajGxk2OhS7UglBjoA             -0.30           +0.17           -0.35           -0.92           -0.72
0KjzXA3g20qsd8zmSekADw             -0.00           +0.00           +0.00           +0.00           -0.00
7GTxyTksSUqCnP5y0ILG4A             +0.00           +0.00           +0.00           -0.00           +0.00
mean                               +0.00           +0.24           -0.11           +0.06           +0.10
```

Two readings, both load-bearing:

1. **All four together is exactly neutral (+0.00 dB mean) on this bucket.** The
   one clip's +0.44 dB is offset by −0.12 and −0.30 on two others. AECMOS
   weights the loss more heavily than the two gains, which is how a bucket that
   is energy-neutral produces a −0.05 mean.
2. **Every partial retime is worse than the full batch.** On the worst clip,
   each of holdoff / hold / release *alone* costs ~+1.2 dB while all four
   together cost +0.44. That is the "their relative rates are what shape the
   response" argument, measured: retiming a subset is not a smaller version of
   this change, it is a worse one. The atomic batch is the correct unit, and
   splitting it to chase the FS_static number would make the result worse.

**The C port does not show it.** The same clip is −0.09 dB in C (2 seconds worse
by >1 dB, 2 better), and C's FS_static stays inside the band at both grids. The
two ports already disagreed on this clip *before* the change — baseline
AECMOS-echo 3.526 (Python) against 4.167 (C) — so it sits on a decision boundary
that the two implementations fall off differently. No stem in this bucket is
sign-consistent across all four configurations:

```
stem                       PY256  PY512   C256   C512     mean
0KjzXA3g20qsd8zmSekADw    +0.003 +0.000 -0.004 +0.000   -0.000
1fvt8ajGxk2OhS7UglBjoA    +0.008 +0.258 -0.034 -0.213   +0.005
49DamGOwmUWGCn23bmI8xw    -0.025 +0.017 -0.136 +0.101   -0.011
7GTxyTksSUqCnP5y0ILG4A    +0.009 -0.004 +0.005 +0.004   +0.004
HAxmF7v4dE0itSp5R5B3Dw    -0.257 -0.242 +0.007 +0.034   -0.115
```

**No other bucket moves.** `FS_movement` (n=25, five times the sample and the
better FS estimator) pools to **+0.006** across all four configurations. Pooled
over the four configurations: DT_movement −0.000, DT_static +0.005, NE +0.000.
The only negative pooled bucket is the n=5 FS_static, at −0.024.

## What this is NOT

It is not a case of the corpus failing to exercise the mechanism. The branch
census (`branch_census_g{256,512}.json`) covers all 90 cases and every branch
fires heavily at both grids:

| grid | holdoff limit | hops | attack (fresh) | attack (ongoing) | hold | release |
|---|---:|---:|---:|---:|---:|---:|
| 16k/256 | 25 | 381320 | 9925 | 83095 | 121377 | 166923 |
| 16k/512 | 12 | 190641 | 8526 | 32189 | 59766 | 90160 |

Two of 90 cases miss at least one branch; the rest exercise all three. A neutral
result on a branch nothing reached would say nothing about its constant — here
that escape is closed.

It is also not a no-op. 83/90 and 82/90 cases changed at the two C grids
(`c_wav_comparison_g{256,512}.json`), worst |diff| 0.71 and 1.15 LSB.

## Open decision

The evidence supports accepting: the bucket is energy-neutral, the one clip's
loss is not reproduced in the other port, the larger FS bucket is positive, and
every alternative that would improve the FS_static number (retiming a subset)
measures worse. But the abort band fired on a stated rule, and the rule exists
so that this call is made deliberately rather than by an author who wants to
proceed. Recorded here, not decided here.

If accepted, the residual is one FS clip at one grid in one port. If rejected,
the constants stay frozen and cover 160 ms at the default grid and 320 ms at the
16 ms grids — which is the defect this campaign exists to remove.

## Run identity

| | |
|---|---|
| Baseline | `ac1df01` (`c4b7382^`), own git worktree |
| Candidate | `c4b7382`, own git worktree |
| Engines | **both** — `aec_wav` (C) and `eval_aec_challenge.py` (Python) |
| Corpus | `wav/aec_challenge_blind`, `eval/manifest_90case_stems.txt` (90 cases) |
| Config | `--preset balanced --filter 52 --cng`, grids 16k/256 and 16k/512 |
| Python | `NO_PREALIGN=1` (mandatory: the driver's default is an offline GCC-PHAT crutch) |
| C | no pre-align path exists; `aec_wav` always runs the online estimator |

```bash
eval/run_c_ab.sh  <wt-ac1df01> <wt-c4b7382> <out> "256 512"
eval/run_py_ab.sh <wt-ac1df01> <wt-c4b7382> <out> "256 512"
python3 eval/simple_mu_branch_census.py --frame-size 256 --out branch_census_g256.json
```

Both harnesses build/render from separate worktrees, and both put the outputs
through `eval/ab_compare.py`: rendered, scored and manifest stem **sets** must be
equal, and every output pair is compared by SHA-256 and per-sample statistics.

## Files

| file | contents |
|---|---|
| `{c,py}_{base,cand}_g{256,512}.scores.json` | per-case `bucket`/`echo`/`deg`, 90 cases each |
| `{c,py}_wav_comparison_g{256,512}.json` | per-case SHA-256, sample count, rate, max/RMS diff |
| `branch_census_g{256,512}.json` | per-case attack/hold/release hit counts over all 90 cases |
| `fs_static_bisect.{txt,json}` | each constant alone against all four together, five FS_static clips |
| `source.diff` | the exact baseline→candidate change |
