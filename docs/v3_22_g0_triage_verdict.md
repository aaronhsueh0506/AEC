# v3.22 G.0 — Unified Triage Trace Verdict

**Date**: 2026-05-21
**Branch**: `feature/v3_22_optimization`
**Baseline**: v3.21.6 final state (FilterAnalyzer default-True per P1; EchoAudibilityConfig wiring per P3; stationarity zeroing default-True per P4)
**Cohort**: 8 cases (5 FS_static = Sprint A worst-FS-echo; 2 DT_movement + 1 DT_static = P4 worst-deg)
**Trace fields added** (6 new, default-OFF, byte-equal preserved at `e_stat_aware_ne_proxy_enabled=False`): `s2_hf_mean`, `render_hf_mean`, `s2_to_x2_ratio_hf`, `erle_hf_mean`, `erle_hf_at_max_frac`, `reverb_tail_dead_frames`. HF region = bin ≥ 32 (~1 kHz at 16 kHz / 257-bin FFT), matching existing `_e2_y2_frac_hf` convention.

## Summary

| Sprint | Gate | Cases passing | Verdict |
|---|---|---|---|
| **D** hybrid residual / nonlinear HF floor | usable_linear ≥ 20% AND median(s2_to_x2_ratio_hf) < 1e-3 | **0/8** | ✅ **CLOSE no-leverage** |
| **F** reverb tail dead fallback | tail_dead_streak > 50 frames rate > 10% on any bucket | **2/8** | ▶ **PROCEED** to design/sanity |
| **G.1 / H.3** ERLE clamp widening | erle_hf_at_max_frac > 50% on FS converged frames | **1/4 FS** | △ **MARGINAL — close pending F outcome** |

v3.22 ship surface narrows to **Sprint F only** (plus optional G.1/H.3 follow-up after F's cohort sanity).

## Sprint D — CLOSE no-leverage

### Trace evidence

| Case | Bucket | usable_linear rate | s2/x2_hf p25 | p50 | p75 | p90 |
|---|---|---:|---:|---:|---:|---:|
| pcb1Nh | FS_static | 88.5% | 0.617 | 1.764 | 3.862 | 6.829 |
| LN18k5r8 | FS_static | 0.0% | 0.010 | 0.026 | 0.083 | 0.312 |
| s90M7MOT | FS_static | 76.7% | 0.059 | 0.176 | 0.638 | 2.431 |
| 9xjhi | FS_static | 89.5% | 6.527 | 9.420 | 13.64 | 22.85 |
| lV0kQN | FS_static | 97.0% | 3.336 | 8.756 | 20.76 | 32.25 |
| WcK0OrF | DT_movement | 92.4% | 0.102 | 0.121 | 0.270 | 0.996 |
| wVYSGV | DT_movement | 96.1% | 6.488 | 15.26 | 19.63 | 21.70 |
| xQEUtY2 | DT_static | 98.5% | 0.076 | 0.114 | 0.139 | 0.311 |

### Diagnosis

`s2_to_x2_ratio_hf` median (8 cases): smallest is **2.6e-2** (LN18k5r8 — where `usable_linear=0%` so D doesn't apply anyway); next smallest is **0.114** (xQEUtY2). Both are **25-200× above** the 1e-3 gate. D's premise — that S²_linear severely under-estimates HF echo despite P1's improved reverb path — is **falsified**.

Interpretation: v3.21.6 P1 (FilterAnalyzer port) already gave the residual-echo estimator a working reverb tail update on the cohort cases where the linear filter is active. The residual S² is now well-calibrated relative to render X² in HF — adding a nonlinear floor would over-estimate echo, harming gain on FS bins.

### Verdict

**D CLOSE no-leverage** for v3.22. No code change. Hypothesis "linear stage HF under-estimation needs nonlinear safety floor" data-falsified on v3.21.6 baseline.

## Sprint F — PROCEED to design / cohort sanity

### Trace evidence

| Case | Bucket | tail_dead_streak > 50f rate | max streak (frames) | tail_max ≤ 0 rate |
|---|---|---:|---:|---:|
| pcb1Nh | FS_static | 3.2% | 117 | 9.2% |
| **LN18k5r8** | FS_static | **97.9%** | **2358** (entire case) | **100.0%** |
| **s90M7MOT** | FS_static | **67.0%** | 1000 | 73.5% |
| 9xjhi | FS_static | 5.6% | 131 | 10.2% |
| lV0kQN | FS_static | 4.5% | 148 | 6.8% |
| WcK0OrF | DT_movement | 5.8% | 277 | 7.1% |
| wVYSGV | DT_movement | 2.3% | 133 | 3.6% |
| xQEUtY2 | DT_static | 0.7% | 79 | 2.0% |

### Diagnosis

LN18k5r8 has **`reverb_freq_resp_tail_max == 0` on 100% of frames** — the reverb estimator never updates throughout the 23.6 s case. P1 FilterAnalyzer port (which fixed 4/5 cohort cases per P1.2 trace) didn't rescue this one. s90M7MOT has 73.5% tail-zero rate and a 1000-frame (= 10 s) dead streak — same failure mode, less severe.

The other 6 cohort cases have tail-zero rates 2-10% — short transient zeros (likely between filter restarts) that wouldn't trigger F's > 50-frame streak gate.

**DT risk**: the 3 DT cases all have dead-streak > 50f rate ≤ 5.8% (= F fires rarely on DT). Maximum DT streak is 277 frames on WcK0OrF (a single ~3 s episode). F's fallback adding R²_unb safety mass during dead-tail state would not engage on DT NE-speech segments significantly. Risk is bounded.

### Verdict

**F PROCEED** to design + sanity. The 2/8 cohort-tail signal is real (1 case 100% dead, 1 case 67%). v3.21.6 P1 didn't fix LN18k5r8 / s90M7MOT — the FilterAnalyzer-driven reverb update has its own preconditions (sufficient peak consistency, `linear_filter_quality` non-None, etc.) that fail on these cases. F's role is to inject a conservative tail estimate when the AEC3 estimator is unable to update — purely RES-internal divergence (AEC3 has no equivalent fallback).

**Implementation plan**:
- Flag-guarded default-OFF `reverb_tail_dead_fallback_enabled: bool = False`
- Threshold `reverb_tail_dead_threshold_frames: int = 50` (matches gate; ~500 ms)
- Strength `reverb_tail_dead_fallback_strength: float = 0.25` (conservative)
- When `_reverb_tail_dead_counter >= threshold`: `r2_unb += render_psd * ep_gain_sq * strength`
- Cohort sanity on 8-case before any 800-case (mirror Sprint E.1.0 pattern):
  - Gate (a) FS echo improvement measurable on LN18k5r8 / s90M7MOT (target Δgain_100 < 0 on tail-dead frames; tail-dead frames should see more suppression)
  - Gate (b) DT cohort tail no NE damage (Δg100 < -0.3 frame count not increased)
- Only if both gates pass → 800-case bench

## Sprint G.1 / H.3 — MARGINAL — close pending F outcome

### Trace evidence

| Case | Bucket | erle_hf_mean | at_max% (all) | at_max% (FS converged frames) |
|---|---|---:|---:|---:|
| pcb1Nh | FS_static | 1.217 | 14.6% | 23.0% |
| LN18k5r8 | FS_static | 1.000 | 0.0% | n/a (0 converged) |
| s90M7MOT | FS_static | 1.031 | 1.8% | 5.1% |
| 9xjhi | FS_static | 1.529 | 31.6% | 41.3% |
| **lV0kQN** | FS_static | 1.731 | 46.0% | **58.5%** |
| WcK0OrF | DT_movement | 1.599 | 45.0% | 28.9% |
| wVYSGV | DT_movement | 1.312 | 25.0% | 22.2% |
| xQEUtY2 | DT_static | 1.180 | 10.5% | 26.0% |

### Diagnosis

Gate criterion (`at_max > 50% on FS converged`) passes on **only 1 of 4 FS converged cases** (lV0kQN at 58.5%; 9xjhi closest miss at 41.3%). The signal is dispersed — at_max ranges from 5.1% to 58.5% on FS converged frames.

The HF region (bin ≥ 32) spans both `max_erle_l=4.0` (bins 32-127) and `max_erle_h=1.5` (bins 128-256) caps. lV0kQN's 58.5% at_max likely indicates the entire [128:] HF half is binding at the 1.5 cap (= 129/225 = 57% of HF bins). Widening the HF cap from 1.5 toward 4.0 (matching LF) would reduce R² on those bins → higher gain → **less HF suppression**.

This is a DT-friendly trade-off (preserve HF formant fidelity) but FS-unfavorable (more HF echo bypass). It's also the opposite direction of where v3.22 has leverage — the v3.21.6 cumulative bench already trades FS for DT in a Pareto-positive way. Loosening HF suppression further would risk negative FS Δecho.

### Verdict

**G.1 / H.3 CLOSE marginal-no-leverage** for v3.22 minimal-cycle. Single-case signal (lV0kQN) is insufficient evidence; direction (less HF suppression) is FS-unfavorable on the same cohort where v3.21.6's residual leverage is FS-positive.

If Sprint F ships and 800-case bench reveals residual FS-leak on cohort tail despite F's tail-fallback, G.1/H.3 widening can be revisited as a 2-step companion (with the understanding that direction may need to be **tighten** the cap not loosen, depending on F's bench result). Out of scope for this minimal-cycle pass.

## v3.22 ship surface after G.0

```
v3.22 = E.close-no-leverage (substrate dormant) +
        D.close-no-leverage +
        F.try-design-then-sanity-then-bench +
        G.1/H.3.close-marginal +
        G.2/G.3/G.4/G.5.untriaged-lite-G.0 +
        Sprint I.config-cleanup-post-bench
```

If F's cohort sanity fails (= F also closes no-leverage), v3.22 ships as **post-parity optimization audit** — no algorithm change, Sprint I cleanup only, substrate from P2 + E retained as research toggles, plus G.0 trace fields landed as instrumentation substrate.

## Open follow-up if F ships

- **G.3 bounded vs unbounded NE detector**: requires alternative detector simulation; deferred (would need additional ~50-100 LOC instrumentation to run the detector twice per frame)
- **G.4 min_gain ceiling**: requires SuppressionGain internal-state exposure (`min_gain_pre_clip` / `_post_clip`); deferred
- **G.5 LF smoothing stale state**: requires SuppressionGain LF-smoothing state lag exposure; deferred
- **G.2 PBFDKF-specific TM**: requires PBFDKF Kalman-state criterion design (constrained to divergence, not parity); deferred until v3.22 ship state evaluated

These are out of scope for this lite G.0 trace pass. Their priority depends on F's bench outcome.

## Files

- Trace data: `/tmp/g0_cohort/*.csv` (8 cases) + `analyze.py`
- Code: `python/modules/orchestrator.py` (6 new fields in `trace_hf_chain`)
- Byte-equal: verified at default-OFF (single-case pcb1Nh md5 identical pre/post: `6143992ee53a9866f6fda068137603b4`)
