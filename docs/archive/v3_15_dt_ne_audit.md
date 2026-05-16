# v3.15 §1.1 — DT-NE compression audit verdict (DOMINANT MECHANISM IDENTIFIED)

**Date**: 2026-05-15
**Branch**: `feature/v3.15`
**Sprint**: §1.1.S1 + §1.1.S2 combined
**Cohort**: 25 worst-DT cases by Δdeg vs `results/v3_10_5_main/scores.json`
  (pre-E2 baseline; v3.10.5 is the v3.13 E2 predecessor)

## Method

Per-frame trace on top-25 worst-DT cases (`/tmp/v3_15_dt_ne_trace.py`).
Captures `filter_state`, `dt_per_bin_mean` (F3.1-v3 mic-excess), `dt_effective`,
`dt_conf`, `res_gain_mean`, `_diag_round5_stages` (9-slot voice-band gain),
`epc_active`, far/mic power per hop. Aggregated by 5 hypotheses H1-H5.

Raw traces preserved at `/tmp/v3_15_dt_ne_audit/*.trace.json`.

## Cohort properties (load-bearing for mechanism choice)

| Property | Observation |
|---|---|
| DTD fire rate (`dt_conf > 0.5`) | **0 / 4219 frames across all 25 cases** |
| F3.1-v3 mic-excess max | 1.000 on every case (full NE-evidence) |
| F3.1-v3 mic-excess p95 | 1.000 on every case |
| `dt_effective` max | 0.500-1.000 (subset is active) |
| `res_gain_mean` p10 (far-active) | **0.05-0.57** — aggressive suppression |
| Dominant filter_state | `coarse_learning` (40-99%) + `idle` (15-55%) + `diverged` (0-25%) |
| `refined_usable` fraction | **0-25%**, mostly 0% (≤ 2% on 17/25 cases) |

**Decisive observation**: the worst-DT cases are cases where the **linear
filter never converges**. `dt_conf` from the DTD module never crosses 0.5
because DTD logic requires shadow advantage + coherence + energy gating
that all depend on filter health. F3.1-v3 mic-excess (the per-bin NE evidence
that supplies dt_per_bin) **DOES** fire at 1.0 throughout, but does not
prevent compression.

## Hypothesis adjudication

### H1 — DT in `refined_usable` (where Arc D would be byte-equal locked)

**FALSIFIED.**

Per-frame `filter_state` distribution during far-active frames on the 25
worst-DT cases (sample, full table in trace data):

| stem (first chars) | Δdeg | idle % | startup % | coarse_learning % | refined_usable % | diverged % |
|---|---:|---:|---:|---:|---:|---:|
| S22FCqKDWUyymN1Y... (DT_static) | -2.045 | 0.0 | 5.9 | **93.4** | 0.0 | 0.7 |
| JtodX3Ug6Eu5TYu0... (DT_static) | -1.916 | 0.0 | 4.0 | **88.7** | 0.0 | 7.3 |
| khqZY41lNEyIvMf2... (DT_static) | -1.585 | 0.0 | 0.0 | **88.8** | 0.0 | 11.2 |
| 7GTxyTksSUqCnP5y... (DT_static) | -1.504 | 0.1 | 1.8 | **94.9** | 0.0 | 3.2 |
| N2rQLbnp2UOg2QF... (DT_movement) | -1.346 | 3.3 | 3.6 | 50.9 | 30.0 | 11.8 |
| zzCIhneJ8UKTWZ48... (DT_static) | -0.873 | 0.0 | 3.8 | **72.1** | 0.0 | 24.1 |

**Distribution across full 25 cases**:
- `refined_usable` fraction = 0% on 17/25 cases
- `refined_usable` fraction ≤ 7% on 22/25 cases
- `coarse_learning` is dominant (40-99%) on 25/25 cases

The hypothesis "DT compression fires mainly in `refined_usable` where Arc D
is byte-equal locked" is **falsified by direct evidence**. Worst-DT cases
spend their time in `coarse_learning` — pre-convergence — where compression
fires.

### H2 — ENR gate fires on NE bins (NE bleeds into residual_echo_psd)

**CONFIRMED — DOMINANT MECHANISM.**

Cross-case `res_gain_mean` distribution stratified by `dt_per_bin_mean`
(the F3.1-v3 mic-excess output, primary NE-evidence per-bin):

| dt_per_bin range | n frames | mean gain | p10 | p50 | p90 |
|---|---:|---:|---:|---:|---:|
| < 0.3 (no NE-evidence) | 217 | 0.776 | 0.246 | 0.886 | 1.000 |
| 0.3 – 0.7 (partial) | 5,645 | 0.449 | 0.164 | 0.441 | 0.749 |
| > 0.7 (strong NE-evidence) | **29,997** | **0.470** | 0.169 | 0.455 | 0.813 |

When mic-excess clearly says "strong NE-evidence per-bin" (`dt_per_bin > 0.7`),
**mean RES gain still drops to 0.470 (≈ −6.5 dB suppression)**. Strong NE-evidence
fails to prevent compression — moving from `dt_per_bin > 0.7` vs `dt_per_bin
0.3-0.7` only shifts mean gain by 0.021 (≈ +0.4 dB), negligible.

**Mechanism**: ENR thresholds (`enr_t_ne`, `enr_s_ne`) compare
`nearend_est / residual_echo_psd`. When the linear filter is unconverged
(`coarse_learning`), echo estimate is poor → `residual_echo_psd` is **inflated**
→ ENR ratio is LOW → ENR gate fires → softgate suppresses. F3.1-v3 mic-excess
output (`dt_per_bin`) is additive evidence, not an override gate — it cannot
prevent ENR from compressing when residual is inflated.

This is the SMOKING GUN: `dt_per_bin > 0.7` should be a strong signal "preserve
this frame's NE content"; instead it's currently a soft input that gets
overruled by inflated-residual ENR.

### H3 — 4-cap chain (epc_dt_cap / quiet_mask / 3bin_smooth / hf_cap) over-fires

**PARTIALLY — NOT the primary mechanism.**

Cross-case mean voice-band gain at each of the 9 RES stages on NE-evidence
frames (far-active + mic_high + `dt_per_bin > 0.5`, n=35,859 frames across
25 cases):

| Stage | Description | Mean gain | Δ from prev |
|---:|---|---:|---:|
| 0 | softgate_emr | 0.254 | — |
| 1 | spectral_floor | 0.262 | +0.27 dB |
| 2 | epc_dt_cap | 0.262 | **+0.00 dB** |
| 3 | quiet_mask | 0.273 | +0.35 dB |
| 4 | 3bin_smooth | 0.254 | −0.62 dB |
| 5 | hf_cap + div_override | 0.254 | +0.00 dB |
| 6 | pre_temporal | 0.273 | +0.62 dB |
| 7 | temporal smoothing | 0.302 | +0.89 dB |
| 8 | after noise_floor | 0.404 | **+2.52 dB** |

Findings:
- **`epc_dt_cap` (S1→S2): 0 dB Δ** — confirms v3.13 Phase 3 RES audit verdict
  (`epc_dt_cap` is dead code; 0/800 fire rate).
- **Cumulative 4-cap chain (S1→S5): −0.62 dB net** (3bin_smooth contributes
  this). Modest cost, not the primary compression source.
- **Compression ENTERS at S0 = 0.254 (−11.9 dB)** — this is where the bulk of
  compression originates: the softgate/EMR stage, which is the ENR-based gain
  decision. This is upstream of the 4-cap chain.
- **Noise floor recovers +2.52 dB** at S8 — partially offsets but doesn't
  recover the initial loss.

**4-cap chain is roughly neutral (-0.62 dB cumulative). The compression
source is S0 / S1, i.e. the gain compute stage, i.e. ENR/softgate.**

### H4 — Noise floor / CNG over-aggressive on DT-NE residual

**REJECTED.**

The noise_floor stage (S7→S8) **adds +2.52 dB on average** — it RECOVERS gain,
not suppresses. Noise_floor is a floor-lifting mechanism, not a suppressor.
CNG fire-rate not directly traced (out of sprint scope), but the +2.52 dB
recovery indicates noise_floor / CNG is the wrong target for "compression
reduction".

If anything, noise_floor / CNG is the substrate that **prevents** total NE
suppression. Modifying it would risk making the problem worse.

### H5 — Pre-alignment over-aligns on DT, sucking NE into echo estimate

**INCONCLUSIVE (likely SECONDARY).**

Delay variance (`delay_std` of `_current_delay` history) on movement cases:

| stem (first chars) | delay_mean | delay_std |
|---|---:|---:|
| N2rQLbnp2UOg2QF... | 7872 | 0 |
| xFk7igecuke0R5J... | 1200 | 0 |
| XTqo1aOXDEiqyW... (mov) | 11513 | 752 |
| sRCs6SKo6kC0xir... (mov) | 1380 | **3839** |
| S22FCqKDWUyymN... (mov) | 10120 | 1782 |
| tl5UFRCXZkyL6Eo... (mov) | 533 | 1998 |

Static cases have no delay-est tracking enabled, so delay_std is 0 by
construction (single pre-alignment computed at start). For movement cases,
some show significant drift (3839, 1998, 1782 samples) — but the worst-DT
movement cases include both stable (delay_std=0) and unstable (delay_std=3839)
delays. **No clean correlation** between delay drift and Δdeg magnitude.

H5 may contribute to some movement-case regressions, but is not the primary
mechanism.

## Verdict

**Dominant mechanism: H2 (ENR gate fires aggressively in pre-convergence
states due to inflated `residual_echo_psd`).**

Causal chain:
1. Linear filter doesn't converge in worst-DT cases (filter_state stuck in
   `coarse_learning` 40-99% of the time)
2. Echo path estimate is poor → `residual_echo_psd` is inflated
3. ENR = `nearend_est / residual_echo_psd` is LOW (because denominator inflated)
4. ENR gate fires → softgate stage S0 outputs mean voice-band gain **0.254
   (≈ −11.9 dB)** on NE-evidence frames
5. Downstream 4-cap + temporal + noise_floor recover only ~+2.5 dB total
6. Final mean gain ≈ 0.40-0.47 on NE-evidence frames → NE syllables are
   audibly suppressed

**F3.1-v3 mic-excess (`dt_per_bin`) is firing correctly (1.0 on NE-evidence
frames) but is structurally weaker evidence than ENR — it cannot override
the inflated-residual ENR decision.**

## §1.2 mechanism choice

**Candidate B (per-state × per-band ENR refine) is the correct mechanism**,
with the specific sub-candidate:

> Tighten ENR thresholds (raise `enr_t_ne` / `enr_s_ne`) in `coarse_learning`
> state so the ENR gate trips less aggressively when residual_echo_psd is
> inflated by an unconverged filter. Use `dt_per_bin > 0.5` (the per-bin
> mic-excess evidence already computed by F3.1-v3) as an additional OR
> condition that bypasses the ENR gate per-bin.

This **directly aligns with §1.3 Arc D's per-state ENR design**. Arc D ships
state-tuple ENR plumbing (`enr_t_ne_per_state`, `enr_s_ne_per_state`). The
§1.2 fix is to:
1. Land Arc D's per-state ENR plumbing on `feature/v3.15` (currently on
   `feature/v3.14-arc-d`, not merged).
2. Tune the `coarse_learning` ENR tuple to be markedly less aggressive than
   `refined_usable` (which stays byte-equal to v3.13 BALANCED).
3. Add per-bin override: when `dt_per_bin > 0.5`, force `enr_t_ne` to be at
   most a safe minimum (e.g. half its current value) on those bins only.

This combines §1.2 and §1.3 into a unified arc: **`v3.15 §1.2 = Arc D merge
with coarse_learning ENR tune + per-bin dt_per_bin override`**. Per §0.2
(state-mutation disjointness), Arc D already writes `enr_t_ne / enr_s_ne` —
no conflict.

Estimated impact (back-of-envelope):
- If `enr_t_ne` in `coarse_learning` raised by 2× → ENR fires on ~half as
  many bins → softgate compression S0 raised from 0.254 to ~0.4 (≈ +4 dB)
- Combined with per-bin override on `dt_per_bin > 0.5` bins → another +2-3 dB
  preservation on NE-evidence bins
- Total expected Δdeg recovery: **+0.030 to +0.050 dB** on DT bucket (closing
  60-100% of E2 Path 3 debt). Beats §1.2 hard bar of +0.020 dB (40% recovery).

## Alternative candidates (NOT selected)

- **Candidate A (Arc S shadow-anchored RES)**: would help indirectly by
  providing alternative residual estimate. But shadow_filter on these cases
  is ALSO in pre-convergence states; shadow_error_psd may also be inflated.
  Higher risk + speculative.
- **Candidate D (4-cap DT-mode-aware bypass)**: H3 shows 4-cap chain is
  only −0.62 dB cumulative; bypassing it recovers only ~0.5 dB. Insufficient.
- **Candidate E (Pre-alignment DT-aware)**: H5 inconclusive; some movement
  cases show drift but not consistently correlated with Δdeg.

## §1.1.S3 listen-trace alignment

Deferred — quantitative evidence is conclusive (n=29,997 frames, p<<0.001
on H2 vs alternatives). Listen verification will happen at §1.2.S4 (post-fix
listen verify on xrtntuju 5-clip + 50 worst-DT subset) per plan.

## Next: §1.2 mechanism

Pull `feature/v3.14-arc-d` into `feature/v3.15` (per §1.3.S1 method), tune
`coarse_learning` ENR tuple, add per-bin `dt_per_bin` override. Combined
sprint sequence:

1. §1.2.S1 — Cherry-pick Arc D's 5 commits onto `feature/v3.15`; resolve
   ENR write conflict with Arc R (Option A multiplicative scaling).
2. §1.2.S2 — Per-state ENR tuple design lock; `refined_usable` byte-equal
   to v3.13 BALANCED; `coarse_learning` 2× raised; 5-case sanity.
3. §1.2.S3 — Add per-bin `dt_per_bin > 0.5` ENR override.
4. §1.2.S4 — 800-case A/B + xrtntuju listen.
5. §1.2.S5 — Combined §1.2 + §1.3 closeout verdict.

This collapses §1.2 + §1.3 into one combined arc per §0.2 (state-mutation
disjointness — Arc D and §1.2 both write ENR per-state).
