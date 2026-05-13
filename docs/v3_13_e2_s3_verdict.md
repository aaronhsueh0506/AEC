# v3.13 E2.S3 verdict — F-DelayTrack `delay_est_period_s` scan on FS_movement

**Status**: CLOSED NEGATIVE (informative). `delay_est_period_s` is **not a
deployable knob on this bench**. Production stays at 0.25s.

**Date**: 2026-05-13

## Scope

Per v3.13 E2 plan, sweep online F-DelayTrack tracker period ∈
{0.05, 0.10, 0.25, 0.50}s on FS_movement (n=131), Path 3 extended bench
pre-align (max_delay_ms=1024) ON for all variants. The question: under
Path 3, does a faster (or slower) tracker period improve linear
cancellation on movement cases?

Per Phase 0 linear-first sequencing (E2.S2 verdict): evaluate on
`_ours_nores.wav` (RES disabled) so RES masking doesn't hide front-end
deltas.

## Headline result

| Period (ms) | n | mean lin ERLE | median | p10 | p90 |
|---:|---:|---:|---:|---:|---:|
| 50 | 131 | 5.573 | 4.966 | 1.646 | 9.962 |
| 100 | 131 | 5.570 | 4.966 | 1.670 | 9.962 |
| **250** | 131 | **5.570** | 5.035 | 1.659 | 9.962 |
| 500 | 131 | 5.568 | 5.035 | 1.668 | 9.962 |

Δ vs production 250ms baseline:

| Period | mean Δ | median Δ | n>+1 dB | n>+3 dB | n<−1 dB | n<−3 dB |
|---:|---:|---:|---:|---:|---:|---:|
| 50ms | +0.002 | 0.000 | 0 | 0 | 0 | 0 |
| 100ms | −0.001 | 0.000 | 0 | 0 | 0 | 0 |
| 500ms | −0.003 | 0.000 | 0 | 0 | 0 | 0 |

Max single-case Δ in either direction: +0.29 dB / −0.12 dB. Below the
noise floor of the linear ERLE metric on FS_movement (cohort spread
already 8+ dB p10→p90).

## Mechanism — why the period doesn't matter on this bench

The bench harness pre-aligns mic/lpb per case via GCC-PHAT before AEC
sees the streams (see `python/eval_aec_challenge.py::estimate_delay`,
now extended to max=1024 under Path 3). The online F-DelayTrack
estimator runs inside the pipeline to compensate **live** delay drift
during a session.

But the AEC Challenge blind dataset has **static within-session
delays** per case (random per case, fixed within a recording). Once
the bench pre-align lands the offset within ±filter_length samples,
there is no drift left for F-DelayTrack to track — its output is
constant across the session for all four periods. Same constant
in, same linear residual out.

This is consistent with the substrate design: F-DelayTrack is a
production safety net for live-drift scenarios (mic/speaker movement
between sessions, lpb buffer jitter, etc.), not a knob that the
offline bench can reward.

## What this rules out / doesn't rule out

**Rules out**: that tuning `delay_est_period_s` alone (within
[0.05, 0.50]s) improves FS_movement on the AEC Challenge bench. The
hypothesis from the v3.13 plan ("period with best FS_movement Δecho")
fails to materialise — there is no such period because the underlying
gradient is zero.

**Doesn't rule out**:
1. Online tracker improvements that *feed back into the reference
   alignment buffer* (i.e., a closed-loop tracker that re-aligns ref
   inside the pipeline, not just monitors delay). This would require
   a small architectural change — F-DelayTrack currently *measures*
   delay but the per-frame ref reaching PBFDKF is the bench-aligned
   stream, not a re-aligned stream.
2. E2.S4 movement persistence classifier — orthogonal axis (detect
   moving cases, switch other parameters), not period.
3. Live-drift datasets (would require synthesizing or recording such
   a corpus; not in v3.13 scope).

## Implication for E2 arc plan

Sprint E2.S3 closes NEGATIVE — no deployable change. Plan implications:

- **E2.S4 movement classifier** (Sprint planned): still relevant; period
  isn't the knob, but persistence detection can drive other knobs
  (e.g., conditional Path 3 confidence gating, conditional fl bump
  toggles on Python ceiling). Recommended to *combine* the classifier
  with closed-loop ref re-alignment investigation rather than period.
- **E2.S5 deployable A/B** (planned): merge Path 3 (already-deployable
  bench fix) with movement classifier gating; **drop F-DelayTrack
  period from the deploy bundle** since it's a no-op on this bench.
- New consideration: should F-DelayTrack output *feed back* into
  ref alignment? Currently it does not (output is consumed by RES
  gating only, see `PathChangeRegimeHandler` `GATE_COH_DELAY` mode
  which production doesn't use). If yes → v3.13.x or v3.14 micro-arc.

## Artifacts

- Renderer: [tools/research/e2_s3_delay_period_scan.py](../tools/research/e2_s3_delay_period_scan.py)
- Summary: [tools/research/e2_s3_linear_summary.py](../tools/research/e2_s3_linear_summary.py)
- Output dirs: `results/v3_13_e2_s3_period_{50,100,250,500}ms/` (gitignored)
- Summary doc: `results/v3_13_e2_s3_linear_summary/summary.md` (gitignored)

## Verification rules followed

1. Path 3 pre-align ON for all 4 variants (bench-side fairness)
2. Linear ERLE on signal-active frames (Phase 0 linear-first discipline)
3. FS_movement bucket only (E2.S3 scope per plan)
4. `_ours_nores.wav` only (RES disabled — no post-RES masking)
5. n=131 cases (full FS_movement bucket)
6. seed=0 inside renderer (deterministic across runs)
