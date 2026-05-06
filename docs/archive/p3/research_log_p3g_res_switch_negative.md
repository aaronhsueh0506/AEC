# Research log — P3g Phase 0 (RES linear-vs-render switch, negative)

Date: 2026-05-06
Code line: v3.10.4 + dry-run diagnostic fields
(`residual_psd_linear`, `residual_psd_render`, `residual_render_blend`).
No behaviour change.

## Question

P3f V1 closed the mu-reduction direction without moving 7GT. P3d had
flagged `using_render = 99 %` post-alignment as part of the diagnosis,
implying the render-based fallback was over-active. The proposed fix:
state-gate the RES residual source — when `usable_linear == True`, lean
toward the linear (filter-`echo_spec`-driven) residual instead of the
render-based fallback.

Per user direction, P3g started with **Phase 0 dry-run only**: instrument
the per-frame linear and render-based residual estimates *without*
changing which one the suppressor consumes, then read the comparison
against the P3f filter_state classification. Phase 1 (soft blend
w = 0.3 / 0.5) would only proceed if the dry-run showed 7GT benefit
without FS leakage.

## Method

In `ResidualEchoEstimator.attribute_legacy`:
- after Stage 1 (ERLE-blended linear residual, line ~3145), stash
  `mean(residual_echo_psd)` as `_last_linear_residual_psd_mean`;
- after Stage 2 (render-based blend, when `_using_render_based`),
  stash `mean(render_based_echo)` and the per-frame `blend` weight.

Surfaced into `_diag` as `residual_psd_linear`, `residual_psd_render`,
`residual_render_blend`. Re-ran the 5-case Phase 2 trace set with
`--trace-aec-state`. Audit script at `/tmp/p3g_audit.py`.

## Sign convention reminder

`residual_echo_psd` is the **echo estimate the Wiener post-filter
consumes**. Larger residual_echo_psd → more aggressive suppression
(more "echo" the post-filter believes it has to remove). So:

```
render_dominance_dB = 10·log10(render_psd) − 10·log10(linear_psd)

> 0  ⇒ render-based estimate is HIGHER  ⇒ more suppression than linear
       would have produced.  Switching to linear would suppress LESS
       (risk of echo leak in FS; potential NE preservation in DT only
       if real NE was the cause of the inflation).
< 0  ⇒ render-based estimate is LOWER   ⇒ less suppression than linear
       would have produced.  Switching to linear would suppress MORE
       (risk of NE damage if linear is over-estimating because of bad
       taps; only safe when filter taps are trustworthy).
```

## Results

Per-state median residual PSD on far-active frames (5 trace cases):

| case | state | n | median lin (dB) | median ren (dB) | dominance (dB) |
|---|---|---:|---:|---:|---:|
| 7GT | suspicious_dt | 628 | **+4.8** | −11.7 | **−16.5** |
| 7GT | refined_usable | 267 | −1.6 | −12.3 | −10.7 |
| 7GT | diverged | 54 | **+17.6** | −9.2 | **−26.8** |
| 7GT | coarse_learning | 989 | −9.3 | −12.6 | −3.3 |
| FS_static | refined_usable | 255 | −26.8 | −15.2 | **+11.7** |
| FS_static | coarse_learning | 605 | −27.4 | −10.3 | +17.1 |
| FS_static | diverged | 128 | −31.7 | −14.0 | +17.7 |
| FS_movement | coarse_learning | 470 | −42.2 | −10.2 | +32.0 |
| DT_static | refined_usable | 294 | −2.1 | −5.7 | −5.1 |

(Render column reads only on frames where `using_render=1`, which is
≥ 73 % of far-active frames in all states; idle excluded as it is
silence-floor.)

Targeted comparisons:

```
[7GT 24–36s NE-evidence] total 599, using_render 599
  linear median: +4.8 dB
  render median: −11.7 dB
  render dominance: −16.5 dB
  ⇒ on the very window we wanted to fix, the linear residual is
    16.5 dB ABOVE the render residual — switching to linear would
    INCREASE suppression based on corrupt taps.

[FS_static usable_linear=True] total 255, using_render 255
  linear median: −26.8 dB
  render median: −15.2 dB
  render dominance: +11.7 dB
  ⇒ render is over-estimating by ~12 dB.  Switching to linear means
    LESS suppression in FS — potential echo leak with no NE upside
    (FS has no NE to preserve).
```

## Findings

### 1. The inversion: render-based is the *protective* path on 7GT

`linear_residual = +4.8 dB` on 7GT NE-contaminated frames means the
filter's `echo_spec` is reporting an echo bigger than the mic itself —
classic post-divergence behaviour: the corrupt taps inflate
`echo_psd`. The render override (`far_psd × erl_estimate`) is
delay-agnostic and does not pass through the bad taps; it produces a
sane, much smaller estimate (−11.7 dB).

The post-filter then sees the smaller render-based estimate and
suppresses *less*. That is the only reason 7GT's deg score is 3.895
and not lower — render-based RES is **already protecting NE** by
de-trusting the corrupt taps.

The P3d observation "`using_render = 99 %` post-alignment" was not a
sign of an over-active fallback; it was the system correctly entering
the protective path because the filter had degraded. Trying to roll
back to linear residual would undo that protection.

### 2. FS_static would lose without DT gain

On FS_static, `usable_linear = True` frames have linear residual
−26.8 dB and render residual −15.2 dB (render dominance +11.7 dB).
Switching to linear (or soft-blending toward it with w = 0.3) would
produce a less aggressive suppression — fine for NE preservation,
except FS has no NE. The trade is FS echo down, deg unchanged.

### 3. P3g closes at Phase 0

Both target effects fail:

- 7GT 24–36s: linear is *worse* than render (+16.5 dB more
  suppression based on corrupt taps).
- FS refined_usable: linear is gentler than render but only because
  render over-estimates; trading echo without a deg gain is a
  one-way regression.

A soft blend (w = 0.3 → 0.5) of the same two paths would inherit the
same trade. There is no Phase 1 worth running.

## What to keep / what to discard

**Keep:**

- Phase 0 dry-run diagnostic fields (`residual_psd_linear`,
  `residual_psd_render`, `residual_render_blend`) on `_diag` and in
  the `--trace-aec-state` CSV. Zero runtime cost when CSV not
  requested. Useful for any future RES experiment that needs to
  audit residual-source behaviour per state.
- The audit script `/tmp/p3g_audit.py` as a template.

**Don't repeat:**

- A linear-vs-render switch decision based on filter convergence
  alone. The convergence signal does not predict whether the linear
  *residual* is trustworthy — corrupt taps can still produce
  `_filter_once_converged = True` and inflate `echo_psd` long after.

## Decision

P3g closes. Toggle proposal (`dt_advisory_use_p3f_state`) for RES
variant **not added** to AecConfig — Phase 0 has already closed it.
Diagnostic fields stay in.

## Files

- Diagnostic fields: `python/aec.py`
  - `ResidualEchoEstimator._last_linear_residual_psd_mean` (≈ line 3148)
  - `ResidualEchoEstimator._last_render_residual_psd_mean` (≈ line 3219)
  - `AEC.process()` `_diag` writes (≈ line 5343)
- Trace fields: `python/run_one_case.py --trace-aec-state`
- Audit: `/tmp/p3g_audit.py`
- 5-case CSVs: `/tmp/p3f_*.csv` (rerun with the new columns)
