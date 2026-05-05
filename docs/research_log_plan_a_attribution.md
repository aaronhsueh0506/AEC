# Research log — Plan A internal attribution (P1.0)

Date: 2026-05-05
Code line: v3.10.4 (release).
Bench: 800-case AEC Challenge blind, BALANCED, fl=52ms, cng=on,
`--parallel`.

## Question

Plan A (v3.8.4) shipped three sub-changes bundled together. The 800-case
bench attributed the cumulative regression to "Plan A" without splitting
which sub-change owned which part of the score change. Before designing
a conditional fix in P1 Phase 1+, we need to know which sub-change is
the actual lever.

The three sub-changes:

- **A1** smoothing kernel `[0.25, 0.5, 0.25] → [0.1, 0.8, 0.1]`
- **A2** HF cap rework: anchor 500 Hz → 2 kHz, gate
  `effective_dt < 0.5 → < 0.3`, skip cap when `high_ne_conf >= 0.3`
- **A3** `_stat_dt_mask` upper edge extended 4 kHz → 7 kHz

## Method

`AecConfig` got three boolean toggles (`plan_a_kernel_tight`,
`plan_a_hf_cap_2k`, `plan_a_stat_mask_7k`, all default True =
v3.10.4 release behaviour). `eval_aec_challenge.py` reads three env
vars (`AEC_PLAN_A_KERNEL`, `AEC_PLAN_A_HF_CAP`, `AEC_PLAN_A_STAT_MASK`)
to flip them. Four BALANCED 800-case benches:

- B1: `AEC_PLAN_A_KERNEL=0`
- B2: `AEC_PLAN_A_HF_CAP=0`
- B3: `AEC_PLAN_A_STAT_MASK=0`
- B4: all three set to 0

## Results

n-weighted aggregates (FS = (169·FS_st + 131·FS_mv)/300; DT same with
186/114; NE = NE bucket only).

| Variant                | FS    | DT echo | DT deg | NE deg | Δ FS vs v3.10.4 | Δ DT deg |
|------------------------|-------|---------|--------|--------|-----------------|----------|
| v3.8.3 baseline        | 3.798 | 4.186   | 2.272  | 4.007  | (reference v3.8.3) | (reference) |
| **v3.10.4 (full Plan A)** | 3.668 | 4.154   | 2.344  | 4.010  | reference       | reference |
| B1 (−A1 kernel)        | 3.654 | 4.156   | 2.353  | 4.011  | −0.014          | +0.009   |
| **B2 (−A2 HF cap)**    | **3.799** | 4.188 | **2.272** | 4.006  | **+0.131**      | **−0.072** |
| B3 (−A3 stat_mask)     | 3.668 | 4.154   | 2.344  | 4.010  | 0               | 0        |
| B4 (−A1 −A2 −A3)       | 3.792 | 4.186   | 2.276  | 4.006  | +0.124          | −0.068   |

Per-bucket (full table for completeness):

| Variant         | FS_st | FS_mv | DT_st e/d   | DT_mv e/d   | NE deg |
|-----------------|-------|-------|-------------|-------------|--------|
| v3.10.4         | 3.641 | 3.704 | 4.217/2.328 | 4.051/2.370 | 4.010  |
| B1              | 3.626 | 3.689 | 4.213/2.337 | 4.043/2.379 | 4.011  |
| **B2**          | **3.777** | **3.827** | 4.248/2.249 | 4.090/2.309 | 4.006  |
| B3              | 3.641 | 3.704 | 4.217/2.328 | 4.051/2.370 | 4.010  |
| B4              | 3.769 | 3.822 | 4.247/2.254 | 4.087/2.313 | 4.006  |

## Conclusions

1. **A2 HF cap rework is the sole driver of the Plan A FS-DT trade.**
   Reverting only the HF cap (B2) restores FS to v3.8.3 baseline
   (3.799 vs 3.798) AND simultaneously gives back all of the DT deg
   gain (2.272 vs baseline 2.272). The whole −0.130 FS / +0.072 DT deg
   trade-off lives in the cap anchor location and gate.

2. **A1 kernel change is essentially noise.** B1 shows FS Δ ≈ −0.014
   (within 800-case noise floor of ~0.01) and DT deg Δ ≈ +0.009 (also
   noise). The kernel rework was the originally-suspected culprit; it
   is not.

3. **A3 stat_mask 4 kHz → 7 kHz extension is completely silent** on
   AECMOS. B3 produces bit-exact identical bucket means to v3.10.4. The
   stationary-DT path is either rarely entered on this dataset, or the
   mask extension does not change downstream gain enough to register.

4. **No interaction.** B2 ≈ B4 (FS 3.799 vs 3.792; DT deg 2.272 vs
   2.276). The three sub-changes act independently.

## Implications for P1 Phase 1+

The conditional gate target moves from "smoothing kernel" to
"**HF cap behaviour**". Concretely, in
`ResFilter._stage_gain_postprocess`:

- When the metric says "this frame has high-band NE present" (DT-like):
  apply Plan A cap (anchor 2 kHz, gate `effective_dt < 0.3`,
  skip on `high_ne_conf` — note the existing skip uses the broken
  `1 - coh2` metric and is therefore ineffective in FS, which is why
  v3.10.5 V1 was a no-op; the new metric must replace this gate).

- When the metric says "this is FS post-cancellation, no NE":
  apply v3.8.3 cap (anchor 500 Hz, gate `effective_dt < 0.5`).

The three toggle flags stay in the config (default True =
v3.10.4 release behaviour) so future investigations can re-isolate.

## Plan B for P1 Phase 1 metric candidates

The P1 plan listed three metric candidates. Given P1.0's finding,
emphasise candidates that fire on **high-band NE presence**, not on
"echo cancellation quality" (that is what `1 - coh2` measures in
practice, and why it saturated). The strongest candidates remain:

1. `m_excess_ratio = mean(max(error_psd[2k:] - α · far_lw[2k:] · erl_est, 0))
   / (mean(error_psd[2k:]) + eps)` — measures uncancelled residual after
   subtracting expected echo.
2. `m_modulation` — high-band envelope CV² (NE has syllabic modulation;
   echo residual is stationary).
3. `m_spectral_flatness` — high-band geometric/arithmetic mean ratio.

The cohort gate in the post-release plan stands; nothing in P1.0
contradicts it. The metric must clear NE ≥ 0.40 and the curated
"Plan A actually helped" DT subset ≥ 0.40 while keeping FS ≤ 0.15.

## Toggle infrastructure left in place

`AecConfig.plan_a_kernel_tight`, `plan_a_hf_cap_2k`,
`plan_a_stat_mask_7k` are now permanent. `eval_aec_challenge.py`
reads `AEC_PLAN_A_KERNEL`, `AEC_PLAN_A_HF_CAP`, `AEC_PLAN_A_STAT_MASK`
env vars. Use these to A/B Phase 2 conditional kernel + cap
implementations against B2 ground truth.

## Reproducibility

```bash
# Reproduce B2 (minus A2 HF cap)
AEC_PLAN_A_HF_CAP=0 python/eval_aec_challenge.py wav/aec_challenge_blind \
  --preset balanced --filter 832 --cng --parallel \
  -o /tmp/bench_p10_b2
python/bench_aecmos.py /tmp/bench_p10_b2 /tmp/bench_p10_b2_scores \
  --label "v3.10.4 minus A2 (HF cap)"
```

Bench output dirs: `/tmp/bench_p10_b{1,2,3,4}` (raw WAVs),
`/tmp/bench_p10_b{1,2,3,4}_scores/` (scoreboards, full result.md
including worst-20 per bucket).
