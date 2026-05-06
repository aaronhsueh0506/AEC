# Research log — high-band NE evidence metric (P1 Phase 1, trace-only)

Date: 2026-05-05
Code line: v3.10.4 + P1.0 toggles.
Behaviour change: NONE. This is trace + cohort analysis only.

## Question

After P1.0 isolated the v3.8.4 Plan A FS-DT trade to the **HF cap rework**
([research_log_plan_a_attribution.md](research_log_plan_a_attribution.md)),
P1 Phase 1 needed a metric that distinguishes "DT high-band NE present"
from "FS post-cancellation". The existing skip gate uses
`high_ne_conf = mean(max(effective_dt, 1 - coh2)[2k:])` which saturates
~1 in FS post-cancellation (low coh2 → "NE-like") and is therefore
non-discriminating. This is the v3.10.5 lesson; do not reuse.

## Method

Three candidate metrics computed per-frame, BEFORE stage-5 smoothing
and stage-6 HF cap (see `python/aec.py` `_update_high_band_metrics_diag`).
Per-case median taken for cohort analysis.

1. **m_excess_ratio** with α ∈ {0.5, 1.0, 2.0}:
   `mean(max(error_psd[2k:] - α · far_lw[2k:] · erl_est, 0)) /
   (mean(error_psd[2k:]) + 1e-10)`
   Index 2 kHz = bin 64 at 16 kHz / 256-pt FFT.
2. **m_modulation**: high-band envelope CV² over 32-frame window.
   Window stat = `var / (mean^2 + eps)` of `near_psd[2k:].mean()` per frame.
3. **m_spectral_flatness**: `geo_mean(error_psd[2k:]) /
   (arith_mean(error_psd[2k:]) + 1e-10)`.

## Cohorts

30 cases sampled per cohort (`random.seed(42)`):

- **FS_static** (n=30): `farend_singletalk` minus `_with_movement_*`
- **FS_movement** (n=30): `farend_singletalk_with_movement_*`
- **DT_static** (n=30): `doubletalk` minus `_with_movement_*`
- **DT_movement** (n=30): `doubletalk_with_movement_*`
- **NE** (n=30): `nearend_singletalk_*` (full file)
- **DT_positive** (n=30): doubletalk cases where `v3.10.4 deg − B2 deg ≥ 0.10`
  (Plan A HF cap genuinely helped; from
  `/tmp/bench_v3104_bal_scores/scores.json` minus
  `/tmp/bench_p10_b2_scores/scores.json`)
- **DT_flat** (n=30): doubletalk cases where `|v3.10.4 deg − B2 deg| ≤ 0.02`
  (Plan A made no difference; sanity cohort)

## Per-cohort medians

| Cohort       | m_excess α=0.5 | α=1.0 | α=2.0 | m_modulation | m_spectral_flatness |
|--------------|----------------|-------|-------|--------------|---------------------|
| FS_static    | 0.086          | **0.001** | 0.000 | 1.181        | 0.262               |
| FS_movement  | 0.176          | **0.031** | 0.000 | 0.973        | 0.240               |
| DT_flat      | 0.296          | 0.125 | 0.028 | 0.989        | 0.196               |
| DT_static    | 0.277          | 0.135 | 0.043 | 1.052        | 0.196               |
| DT_movement  | 0.363          | 0.189 | 0.057 | 1.065        | 0.179               |
| **DT_positive** | 0.694       | **0.552** | 0.382 | 0.935        | 0.173               |
| **NE**       | 1.000          | **1.000** | 0.999 | 1.439        | 0.096               |

## AUROC FS-vs-(NE ∪ DT_positive)

| Metric                | AUROC | NE_only | DT_pos_only |
|-----------------------|-------|---------|-------------|
| m_excess_ratio (α=0.5) | 0.868 | 0.989   | 0.747       |
| **m_excess_ratio (α=1.0)** | **0.871** | **0.989** | **0.752** |
| m_excess_ratio (α=2.0) | 0.874 | 0.989   | 0.759       |
| m_modulation          | 0.564 | 0.755   | 0.372       |
| m_spectral_flatness   | 0.200 | 0.103   | 0.297       |

## α-stability of m_excess_ratio

AUROC across α ∈ {0.5, 1.0, 2.0}: 0.868 / 0.871 / 0.874.
Range = **0.006**, well under the < 0.05 gate.

The metric is robust to the `erl_estimate` scaling factor — a key
worry since `erl_estimate` is unreliable in DT and during weak-filter
periods. The high-band excess signal is dominant enough that the
α scaling barely moves AUROC.

## Cohort gate verdicts

A metric "passes" if:
- FS_static median ≤ 0.15 AND FS_movement median ≤ 0.15
- NE median ≥ 0.40
- DT_positive median ≥ 0.40
- AUROC FS-vs-(NE ∪ DT_positive) ≥ 0.85

| Metric             | Verdict | Failed gates                             |
|--------------------|---------|------------------------------------------|
| m_excess α=0.5     | FAIL    | FS_movement median 0.176 > 0.15          |
| **m_excess α=1.0** | **PASS** | (none)                                  |
| m_excess α=2.0     | FAIL    | DT_positive median 0.382 < 0.40 (over-gate) |
| m_modulation       | FAIL    | FS_static 1.18; AUROC 0.56 (≈ random)    |
| m_spectral_flatness | FAIL   | FS 0.26 / NE 0.10 (inverted!); AUROC 0.20 |

## Diagnostic notes

**Why m_modulation fails**: high-band envelope CV² is a property of the
mic-side signal, not the residual. FS-mic side has the far-end speaker's
syllabic modulation just as much as DT-mic has near-end syllabic
modulation, so CV² doesn't separate the two. Only NE bucket lights up
because there's no echo at all, but FS-with-echo and DT-with-echo look
the same at the envelope level.

**Why m_spectral_flatness inverts**: speech (NE) high-band has fricatives
and turbulent components that are spectrally flatter than the relatively
sparse echo residual. Opposite of intuition. Useless as a positive-NE
signal.

**Why m_excess_ratio works**: residual energy beyond the predicted echo
is exactly the right physical signal. NE saturates because there is far
more high-band mic energy than far_lw can predict (no echo). FS
post-cancellation has near-zero residual after removing the predicted
echo. DT_positive cases have NE present and the metric correctly fires.

## Recommendation for P1 Phase 2

**Use `m_excess_ratio` with α = 1.0** as the high-band NE evidence
gate replacing the existing `1 - coh2`-based `high_ne_conf` in stage 6.

Conditional HF cap behaviour (replacing existing stage 6 skip):

```
if metric < θ:                                  # FS-like (no NE)
    apply v3.8.3 cap (anchor 500 Hz, gate effective_dt < 0.5)
elif metric < θ_high:                           # ambiguous
    apply v3.10.4 cap (anchor 2 kHz, gate effective_dt < 0.3)
else:                                           # DT-like (NE clearly present)
    skip cap entirely (preserve high-band)
```

Threshold candidates to sweep in Phase 2:
- θ = 0.20 / 0.30 / 0.40 / 0.50
- θ_high = 0.70 / 0.80 (for the three-way variant) or omit (binary FS-vs-DT)

Per-cohort distribution suggests:
- θ = 0.30 sits at FS_movement p75 boundary (0.31 in trace data)
- θ = 0.50 catches DT_positive median (0.55) but may miss DT_flat → loses
  some Plan A protection
- θ = 0.20 is conservative, gates more cases as DT-like

Bench gate (per Phase 2 implementation):
- FS ≥ +0.05 vs v3.10.4 (recover at least 0.05 of the 0.130 Plan A cost)
- DT deg ≥ −0.02 vs v3.10.4 (preserve most of Plan A's DT improvement)
- NE flat
- 800-case AECMOS, BALANCED preset

## DT_positive cohort stem list

Reproducibility — these 30 cases drove the gate analysis:

```
docs/research_log_high_band_evidence.md (auto-saved during the trace)
```

Source files:
- `/tmp/p1ph1/case_medians.csv` (210 case × 5 metrics × cohort)
- `/tmp/p1ph1/run.log` (per-case trace progress)

## Reproduction

```bash
# Full trace + analysis
python python/trace_high_band_metrics.py
# Outputs to /tmp/p1ph1/
```

The driver embeds cohort selection + per-case AEC run with
`AecConfig.trace_high_band_metrics=True`. Default release behaviour is
preserved (flag defaults to False).
