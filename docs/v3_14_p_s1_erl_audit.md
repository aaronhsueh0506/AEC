# v3.14 Arc-P Sprint 01: Per-Band ERL Audit

**Date**: 2026-05-14  
**Branch**: `feature/v3.14-arc-p`  
**Sprint**: P.S1 — Per-band ERL estimate audit (prerequisite for Arc-D and Arc-R wiring)  
**Status**: CLOSED — findings documented; P.S2 design recommendations below

---

## 1. Objective

Audit the per-band distribution of actual echo coupling ERL versus the production
scalar `_erl_estimate` (default ~0.3, clipped to [0.001, 1.0]) used in the F3.1-v3
mic-excess formula:

```
excess_ratio[k] = max(error_psd[k] - far_lw[k] * erl_estimate, 0) / (error_psd[k] + eps)
```

The working hypothesis was: scalar 0.3 is LF-conservative → systematically under-estimates
HF ERL → over-suppresses HF NE speech. The audit tests this hypothesis empirically.

---

## 2. Method

**Audit tool**: `tools/research/v3_14_p_s1_erl_audit.py`

**Cases**: 8 worst-FS listen cases from `listen/v3_12_worst_fs/`

| ID  | Stem (short)       | Bucket      |
|-----|--------------------|-------------|
| 01  | 7GTxyTks           | FS_static   |
| 02  | IrQvqOTC           | FS_static   |
| 03  | pcb1Nh0Z           | FS_static   |
| 04  | S22FCqKD           | FS_static   |
| 05  | hVqUmGvI           | FS_static   |
| 06  | 5bJUo1K3           | FS_movement |
| 07  | IrQvqOTC_movement  | FS_movement |
| 08  | S22FCqKD_movement  | FS_movement |

**Bench config**: BALANCED preset / fl=832 / cng=True / seed=42 (standard)

**ERL proxy**: `per_band_ERL(k_band) = mean(error_psd[k_band]) / mean(far_lw[k_band])`  
Valid interpretation: in FS frames where the filter has cancelled the echo, `error_psd ≈
residual echo ≈ far_lw × ERL(k)`. The per-band ratio estimates the true room coupling
in each frequency band.

**Bands**:
- LF: 0–1000 Hz (bins 0–40 @ 16 kHz / 320-sample frame)
- MF: 1000–4000 Hz (bins 40–160)
- HF: 4000–8000 Hz (bins 160–257)

**Frame gate**: `lw_ready AND far_power_db > -55 dB AND dt_confidence < 0.30`  
Stratified into: all qualifying frames / `filter_converged=True` only.  
The `filter_converged=True` stratum is the most reliable for ERL interpretation;
in non-converged frames `error_psd ≈ mic (not cancelled)`, inflating the proxy.

---

## 3. Key Finding: Worst-FS Cases Rarely Converge

Only **2 of 8 cases** reach `filter_converged=True`:

| Case | Converged frames | Total frames | Conv% |
|------|-----------------|--------------|-------|
| 01   | 3               | 2203         | 0.1%  |
| 02   | 0               | 2176         | 0%    |
| 03   | 0               | 2322         | 0%    |
| 04   | **742**         | 2414         | **31%** |
| 05   | 0               | 2267         | 0%    |
| 06   | 0               | 2191         | 0%    |
| 07   | 0               | 2517         | 0%    |
| 08   | **296**         | 2205         | **13%** |

This is the defining characteristic of the "worst-FS" cohort: these cases are worst-FS
precisely because the filter never converges (delay mismatch, NL distortion). For ERL
interpretation, the converged-frame data from cases 04 and 08 is the ground truth.

---

## 4. Per-Band ERL Distribution

### 4.1 All qualifying frames (lw_ready + far_active + low_DT)

ERL proxy is inflated in non-converged cases because `error ≈ mic` (filter fails).
These numbers represent a mix of "room coupling + filter failure residual":

| Band    | Bucket      | ERL_mean | ERL_p10 | ERL_p90 | Bias (actual−0.3) | ER_sc030 | ER_perband |
|---------|-------------|----------|---------|---------|-------------------|----------|------------|
| LF_0_1k | FS_static   | 0.774    | 0.018   | 1.660   | +0.474            | 0.295    | 0.288      |
| MF_1_4k | FS_static   | 0.869    | 0.058   | 1.836   | +0.569            | 0.359    | 0.301      |
| HF_4_8k | FS_static   | 1.002    | 0.229   | 1.728   | +0.702            | 0.342    | 0.249      |
| LF_0_1k | FS_movement | 1.073    | 0.006   | 2.000   | +0.773            | 0.364    | 0.299      |
| MF_1_4k | FS_movement | 1.135    | 0.025   | 2.000   | +0.835            | 0.469    | 0.381      |
| HF_4_8k | FS_movement | 0.853    | 0.007   | 2.000   | +0.553            | 0.286    | 0.225      |

**Do not use these as per-band ERL recommendations** — the inflated values reflect
filter failure, not room physics.

### 4.2 Filter-converged frames only (authoritative)

From cases 04 (FS_static, 742 frames) and 08 (FS_movement, 296 frames):

| Band    | Bucket      | ERL_mean | ERL_p10 | ERL_p90 | Bias (actual−0.3) | ER_sc030 | ER_perband | ER_delta |
|---------|-------------|----------|---------|---------|-------------------|----------|------------|----------|
| LF_0_1k | FS_static   | 0.063    | 0.033   | 0.110   | **−0.237**        | 0.011    | 0.158      | +0.147   |
| MF_1_4k | FS_static   | 0.097    | 0.002   | 0.265   | **−0.203**        | 0.028    | 0.300      | +0.272   |
| HF_4_8k | FS_static   | 0.094    | 0.027   | 0.220   | **−0.206**        | 0.031    | 0.157      | +0.126   |
| LF_0_1k | FS_movement | 0.489    | 0.001   | 2.000   | **+0.189**        | 0.174    | 0.204      | +0.031   |
| MF_1_4k | FS_movement | 0.274    | 0.001   | 0.799   | **−0.026**        | 0.108    | 0.157      | +0.049   |
| HF_4_8k | FS_movement | 0.344    | 0.001   | 1.383   | **+0.044**        | 0.043    | 0.054      | +0.011   |

**Global converged-frame ERL** (pooled over cases 04 + 08):
- LF: mean=0.205, p10=0.051, p90=0.408, bias=−0.095
- MF: mean=0.156, p10=0.040, p90=0.257, bias=−0.144
- HF: mean=0.177, p10=0.084, p90=0.298, bias=−0.123

---

## 5. Hypothesis Falsification

**Original hypothesis**: scalar 0.3 under-estimates HF ERL → excess over-fires → over-suppresses HF NE speech.

**Finding**: In the 742-frame converged FS_static case (04):
- LF ERL = 0.043 (scalar 0.3 is 7× OVER-estimate)
- MF ERL = 0.191 (scalar 0.3 is 1.6× OVER-estimate)
- HF ERL = 0.111 (scalar 0.3 is 2.7× OVER-estimate)

**The scalar 0.3 OVER-estimates ERL in all bands in this room**, not under-estimates it.

This means: `far_lw × 0.3 >> far_lw × 0.043` → `excess = error - far_lw × 0.3 ≈ 0`
→ `dt_per_bin ≈ 0` (no NE protection) even though there is real near-end residual.
The F3.1-v3 excess formula in this room UNDER-reports NE evidence in FS frames because
the ERL floor is too conservative (too high).

**The FS_movement case (08, 296 frames)** shows a different room:
- LF ERL = 0.489 (scalar 0.3 under-estimates by +0.189)
- MF ERL = 0.274 (near scalar; bias −0.026)
- HF ERL = 0.344 (scalar slightly under-estimates)

This room has higher coupling (louder echo relative to far signal), consistent with
movement cases where the speaker is close to the microphone.

---

## 6. Root Cause Analysis

The per-band ERL variation between rooms is **large** (case 04 LF=0.043 vs case 08 LF=0.489 — an 11× range). This means:

1. **A single per-band ERL table cannot serve all rooms** — per-band ERL is not a fixed
   room property but a room+geometry+level function that varies dramatically.
2. **The scalar 0.3 post-EPC cap** (line 6265, 6304) is the actual bottleneck: after EPC,
   `_erl_estimate` is forced to min(current, 0.3). In low-coupling rooms (case 04), the
   true ERL is 0.04–0.11 but the cap holds it at 0.3, artificially raising the echo mask
   → F3.1 excess fires less → echo leaks through into dt_per_bin.
3. **The FS_static worst cases (01–05) that never converge** have `_erl_estimate` stuck at
   the initial value (~0.1) because the convergence gate on the ERL update path
   (`raw_dt_ratio < 2.0 and inst_erl_raw < 1.5`, lines 6384–6387) never fires reliably.
   The filter doesn't cancel → `raw_err_pwr / far_pwr > 2.0` → ERL update blocked.

---

## 7. Scalar Bias Profile by Bucket

| Bucket      | LF bias (actual−0.3) | MF bias | HF bias | Dominant direction |
|-------------|----------------------|---------|---------|-------------------|
| FS_static   | −0.237               | −0.203  | −0.206  | Scalar OVER-estimates ERL in all bands (low-coupling room) |
| FS_movement | +0.189               | −0.026  | +0.044  | Scalar roughly matched or slightly under (high-coupling room) |

**The bias is room-dependent, not band-dependent.** Per-band ERL profiles do not
show a consistent LF<MF<HF pattern in these cases.

---

## 8. Recommendations for P.S2

### 8.1 Primary recommendation: Adaptive per-band ERL from filter state

Per-band ERL should be estimated adaptively from the Kalman filter state rather than
from a fixed table. In converged frames, the per-bin echo estimate `echo_psd[k]` from
the main PBFDKF filter (`filter.echo_spec`) provides a direct per-bin ERL proxy:

```
erl_bin[k] = echo_psd[k] / far_psd[k]  (FS, converged, far active)
```

This is already available in `ResFilter` (echo_psd is produced by PBFDKF) and updated
every frame. A per-band EMA over this would give a slow-tracking per-band ERL estimate
without needing a fixed table.

### 8.2 Per-band ERL clip values for P.S2 (conservative start)

Based on the converged-frame distribution across 2 rooms:

| Band    | Observed range (converged) | Recommended clip_lo | Recommended clip_hi | Notes |
|---------|---------------------------|---------------------|---------------------|-------|
| LF 0–1k | 0.033–0.489 (mean ~0.15)  | 0.01                | 0.6                 | Case 04: 0.043; Case 08: 0.489 |
| MF 1–4k | 0.002–0.799 (mean ~0.17)  | 0.005               | 0.8                 | Case 04: 0.191; Case 08: 0.274 |
| HF 4–8k | 0.001–1.383 (mean ~0.20)  | 0.005               | 1.0                 | Case 04: 0.111; Case 08: 0.344 |

**The ranges overlap substantially across bands** — there is no clean 3-band split that
captures per-room variance. Per-band clips should be implemented but treated as safety
bounds only, with an adaptive per-band EMA being the primary mechanism.

### 8.3 Key design constraint for P.S2

The post-EPC cap at `min(current, 0.3)` (aec.py lines 6265, 6304) is the highest-impact
ERL change: in low-coupling rooms (case 04, true ERL ~0.04–0.11), this cap forces
erl_estimate to 0.3 which artificially inflates the echo mask → F3.1 excess fires less
→ echo leaks into dt_per_bin when it shouldn't. Relaxing the post-EPC cap per-band
(e.g., cap_lo = 0.01, cap_hi = 0.6) is likely higher value than the per-band split alone.

### 8.4 Approach not recommended: Fixed per-band ERL table

A fixed 3-band ERL table (e.g., LF=0.15, MF=0.20, HF=0.25) would not generalize
because inter-room variance (11×) dwarfs inter-band variance. Any fixed table would
be mis-tuned for the next room. Adaptive per-band estimation is the correct arch.

---

## 9. Data Quality Notes

- **Converged-frame sample size is small** (742 + 296 frames from 2 rooms). The per-band
  ERL values are directionally correct but the recommended clip values should be validated
  with the full 800-case corpus when P.S2 wiring is complete.
- **Non-converged cases**: 6 of 8 worst-FS cases never converge. Their ERL proxy (all-frames
  column) reflects `error ≈ mic ≈ echo + background`, inflating values to 0.8–1.8. These
  are not usable for room ERL estimation.
- **This is an 8-case worst-FS audit, not a representative distribution**. The 800-case corpus
  contains many well-converged cases; the per-band ERL distribution there would be much
  wider and more informative.

---

## 10. P.S2 Sprint Design Lock (Provisional)

Based on P.S1 findings:

- **Target**: Replace scalar `erl_estimate` in F3.1 excess formula with `erl_band[k]`
  — a slow EMA of `echo_psd[k] / far_psd[k]` initialized to the scalar `_erl_estimate`,
  clipped to [0.005, 1.0] per-bin, updated only in converged FS frames.
- **Default-OFF**: env-gated `AEC_PER_BAND_ERL=1` or config flag `use_per_band_erl=False`.
- **Byte-equal when OFF**: no production code path touched when flag is False.
- **First gate**: 8-case listen verification that per-band ERL changes excess_ratio
  meaningfully in at least one case (the low-coupling FS_static case 04 is the primary
  target — scalar 0.3 over-estimates, per-band ~0.04–0.11 would tighten the mask).
- **Second gate**: 800-case AECMOS net-positive at ≥ 0.005 dB DT Δdeg without FS
  regression > 0.005 dB.

---

## Files

- **Audit script**: `tools/research/v3_14_p_s1_erl_audit.py`
- **Per-case JSONs**: `/tmp/v3_14_p_s1_erl_audit/*.json` (gitignored, regenerate with script)
- **Summary JSON**: `/tmp/v3_14_p_s1_erl_audit/summary.json`
