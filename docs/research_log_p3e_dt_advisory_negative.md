# Research log — P3e DT advisory gate (negative result)

Date: 2026-05-06
Code line: v3.10.4 (advisory toggles default off in shipped build).
Bench preset: balanced / fl=52ms / cng=on / 800-case AEC Challenge blind.

## Question

P3d trace showed that on `7GTxyTksSUqCnP5y0ILG4A_doubletalk` the 788-ms
skew alignment is solved at v3.10.4, but the post-alignment back half
shows ERLE collapsing to 0–negative. DT signals (`dt_shadow` median
0.51, `dt_energy` median 0.24) are present, but the composite
`dt_active` gate that drives `mu_scale` reduction never fires
(`enable_dtd=False` zeroes out `dt_conf` / `dt_coh`).

Hypothesis: route shadow / energy DT evidence directly into a
mu-reduction-only advisory gate (no RES change, no taps freeze) and
the back-half taps will stop degrading.

## Method

Implemented a hit-then-hold (400 ms) advisory state machine that
multiplies `mu_scale` by `dt_advisory_mu_factor` (0.3) when the gate
fires. Three variants benched on full 800-case:

| Variant | Gate condition |
|---|---|
| V1 | `far_active AND dt_shadow > 0.5` |
| V2 | `far_active AND (dt_shadow > 0.5 OR dt_energy > 0.4)` |
| V3 | `far_active AND once_converged AND not post_reset_warmup AND (dt_shadow > 0.5 OR dt_energy > 0.4)` |

## Results

| Bucket | v3.10.4 | V1 | V2 | V3 |
|---|---:|---:|---:|---:|
| FS_static echo  | 3.641 | 3.546 (**−0.095**) | 3.513 (**−0.128**) | 3.630 (−0.011) |
| FS_movement echo| 3.704 | 3.636 (**−0.068**) | 3.590 (**−0.114**) | 3.688 (−0.016) |
| DT_static deg   | 2.328 | 2.402 (+0.074) | 2.432 (+0.104) | 2.335 (+0.007) |
| DT_movement deg | 2.370 | 2.422 (+0.052) | 2.457 (+0.087) | 2.388 (+0.018) |
| NE deg          | 4.013 | flat | flat | 4.010 (flat) |
| 7GT doubletalk echo / deg | 3.366 / 3.895 | — | — | **3.366 / 3.895 (identical)** |

## Findings

### V1 / V2 fail — DT-confound

V1 (shadow only) drops FS_static echo by 0.095. V2 (shadow OR energy)
makes it worse (−0.128). Mechanism: in early FS, the main filter is
still converging; the passive shadow filter happens to beat main
because the relative coherence-vs-error trade-off favours shadow at
that stage. So `dt_shadow > 0.5` fires *before* the filter has
adapted, mu is reduced 0.3×, the main filter learns slower, and
echo leaks. The DT bucket gain of V1 (+0.074 deg) is partly real DT
protection — but it is also partly the same FS-style false fire
benefiting cases that happen to have shadow advantage during DT.

### V3 — guard works for FS, advisory window collapses

Adding `once_converged AND not post_reset_warmup` brings FS back
within ±0.02 of baseline (decisive test passed). But the DT deg
gain shrinks from +0.074/+0.052 → +0.007/+0.018 — about an order of
magnitude. **7GT doubletalk score is bit-identical to v3.10.4
(3.366 / 3.895)**. Either `once_converged` is not reached during
the 7GT post-alignment contamination window (likely — 7GT alignment
fires at t=4.57 s and the filter spends 4–8 s rebuilding ERLE before
once_converged latches), or the back-half DT signals fail the gate
for another reason. Either way, **V3 misses the case that motivated
P3e**.

The bucket-level DT gain that survives V3 (+0.007 / +0.018) is
within bench noise.

## Root-cause reading

Single per-frame `dt_shadow > th` (with or without convergence guard)
cannot distinguish the three states it confounds:

1. **Pre-convergence shadow-beats-main** — FS, no DT, false fire.
2. **Post-reset shadow-beats-main** — FS after delay re-acquisition,
   no DT, false fire (V3 does catch this with `post_reset_warmup`).
3. **Genuine DT after the filter has matured** — what we want to
   gate on.

Convergence guard removes (1) but is too coarse for the difference
between (2) the post-reset rebuild window (where shadow beating
main is still a convergence artefact) and (3) the steady-state DT
signal we want. On 7GT specifically, the contamination window is
*inside* the post-reset rebuild — exactly the region V3 is forced
to ignore.

## Decision

Stop iterating single-threshold + per-frame guard variants. The
shape of the V1 → V2 → V3 sweep makes it clear that the search
space here is FS-protection-vs-DT-coverage with no winning point.

The next step is **P3f: Mini AecState** — adopt the WebRTC AEC3
state-layer pattern (subtractor output analyzer → aec_state →
suppression / adaptation policy) but **trace-only first**. We will:

1. Compute three state layers per frame, written to `_diag` / diag
   CSV, with **no behaviour change**:
   - **Filter state** — `pre_converged / coarse_converged /
     refined_converged / diverged / post_reset_warmup`. Driven by
     relative-error ratios (analogue of WebRTC's
     `e2_refined < 0.5·y2`, `e2_coarse < 0.05·y2`), not by
     `dt_shadow` or absolute-ERLE thresholds.
   - **Nearend state** — `nearend_evidence` only meaningful in
     `refined_converged`; before that, shadow-beats-main means
     "shadow learnt first", not DT.
   - **Usable linear estimate** —
     `delay_solid AND refined_converged AND not diverged`.
2. Validate invariants on bench traces:
   - FS early frames classify as `pre/coarse_converging`, not DT.
   - 7GT post-alignment 4–12 s window passes through
     `coarse_converged → refined_converged`.
   - True DT_static / DT_movement frames classify as
     `refined_converged + nearend_evidence`.
3. Only after invariants are validated, wire mu reduction and
   RES linear-vs-render switching to the state machine. The
   advisory becomes:
   ```
   dt_advisory =
       refined_converged
       AND not post_reset_warmup
       AND nearend_evidence
   ```

P3e advisory toggles (`dt_advisory_enabled`, etc.) are kept in the
`AecConfig` for now (default off, no behaviour change) so V1/V2/V3
can be re-benched against P3f's classifier. They will be removed
or repurposed once P3f lands.

### Notes on adapting WebRTC AEC3's pattern to our codebase

- AEC3 has refined and coarse filters; both adapt. Our codebase has
  a main filter (adaptive) and a *passive* shadow that copy-tracks
  main with delay. So our `e2_coarse / y2` analogue must be derived
  from `pred_err_psd / mic_psd`, not from the shadow filter — the
  shadow ratio is stale.
- WebRTC's e2 thresholds (`< 0.5·y2`, `< 0.05·y2`) should not be
  copied verbatim. They are calibrated for AEC3's refined-coarse
  pair. We will measure what ratios our `pred_err_psd / mic_psd`
  actually takes in `pre / coarse / refined` regimes and pick
  thresholds from the empirical distribution.
- `post_reset_warmup` already exists in our code
  (`_warmup_frames > 0` OR `_p_max_override_frames > 0`); reuse it.

## Files

- Bench output: `/tmp/bench_p3e_v1`, `/tmp/bench_p3e_v2`,
  `/tmp/bench_p3e_v3`
- Score JSONs: `/tmp/bench_p3e_v{1,2,3}_scores/`
- Code: `python/aec.py` `AecConfig.dt_advisory_*`,
  `process()` advisory state machine (≈ line 4597).
