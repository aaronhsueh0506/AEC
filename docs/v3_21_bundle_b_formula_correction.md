# v3.21 Bundle B Formula Correction

**Date**: 2026-05-26  
**Track 1 output** (prerequisite for M_B_corrected composition step)

---

## T1.1 — A.1 Equivalent Effective Mu Derivation

**Question**: Does `use_partition_summed_x2_for_shadow_mu` (A.1) require a compensating `shadow_mu_nlms` change to match AEC3 coarse's `rate=0.7`?

### Analytical derivation

```
AEC3 coarse (from echo_canceller3_config.cc:99, coarse_filter_update_gain.cc:64-72):
  hop = 64 samples (4 ms @ 16 kHz)
  n_partitions = 13
  rate = 0.7
  mu formula: mu[k] = rate / X2[k]
  X2 = render_power = SpectralSum = Σ_{p=0}^{12} |X[p]|²  (upstream subtractor sum)

Python shadow NLMS (A.1 ON):
  hop = 160 samples (10 ms @ 16 kHz)
  n_partitions = 5 (filter_length 832 / block_size 320 = 2.6 → default 5)
  shadow_mu_nlms = 0.5
  mu formula (A.1 ON): mu[k] = shadow_mu_nlms / Σ_{p=0}^{4} |X_buf[p]|²

Per-unit-signal scaling:
  X2_AEC3 ≈ (13/5) × X2_Python  (more partitions → larger sum)

Effective learning rate per second (X²_pp = X² per partition per bin):
  AEC3:   0.7 / (13 × X²_pp) × (16000/64)  = 0.7/13 × 250 = 13.46 / X²_pp / s
  Python: 0.5 / (5  × X²_pp) × (16000/160) = 0.5/5  × 100 = 10.00 / X²_pp / s
  Ratio: 0.74× (Python ~26% slower per second)
```

**T1.1 conclusion**: ratio 0.74× < 2× threshold → **no candidate flag needed**.  
Keep `shadow_mu_nlms = 0.5`. Document as negligible gap. Trace confirmation below (§T1.1 trace).

---

## T1.2 — A.2 Noise Gate Constant Correction

### AEC3 source confirmation

From `echo_canceller3_config.cc:97-100`:
```cpp
RefinedConfiguration refined = {..., .noise_gate = 20075344.f};
CoarseConfiguration  coarse  = {.length_blocks = 13, .rate = 0.7f, .noise_gate = 20075344.f};
```

Both **refined AND coarse** filter update paths use `noise_gate = 20075344` (int16² scale).  
Threshold semantics: `mu[k] = 0` where `X²[k] < noise_gate` (coarse_filter_update_gain.cc:67-71).

**Python bug (pre-T1.2)**: A.2 flag used `NOISE_GATE_POWER_FLOAT = psd_int16_to_float(27509562) = 0.02562`.  
This was incorrectly ported from AEC3's **suppression/residual path** (`echo_model.noise_gate_power = 27509.42f` — float PSD scale, completely different path).

**Fix (T1.2, 2026-05-26)**:
- Added `FILTER_NOISE_GATE_POWER_FLOAT = psd_int16_to_float(20075344.0) = 0.01870` to `aec3_scale.py`
- A.2 flag now uses `FILTER_NOISE_GATE_POWER_FLOAT` (flag-gated, default-OFF)
- Old `NOISE_GATE_POWER_FLOAT = 0.02562` retained with warning comment (residual path only)
- Files changed: `modules/aec3_scale.py`, `modules/filters.py` (A.2 block), `modules/config.py` (comment)

**Correction magnitude**: 0.02562 → 0.01870, ratio = 1.37×  
Effect: fewer bins zeroed (gate more permissive), consistent with AEC3 production behavior.

**Parseval FFT-size note**: per-bin mean power ≈ 2×energy_td regardless of FFT size  
(Σ|X|²=N×E_td, #bins≈N/2 → mean≈2E_td), so **no FFT-size scaling** needed between AEC3 (FFT=128) and Python (FFT=512) — same threshold value applies.

---

## T1.2 Trace Results (2026-05-26)

8-case cohort × 6 variants: M0 / B1(A1) / B2(A1+A2) / B3(A1+A2+A3) / B4+A4 / B5+A5.  
Full trace data: [`docs/v3_21_t1_2_bundle_b_trace.md`](v3_21_t1_2_bundle_b_trace.md)

### T1.1 trace confirmation — mu_eff_mean (far-active frames)

| Case (bucket) | M0_mu_mean | B1_mu_mean | B1/M0 | B2_mu_mean | B2_mu_p95 |
|---|---|---|---|---|---|
| 9xjhiFbG (FS_static) | 5.29e+00 | 2.38e+01 | **4.5×** | 4.13e+00 | 7.66e+01 |
| xFk7igec (DT_mvmt)   | 8.12e+00 | 4.83e+01 | **5.9×** | 3.59e+00 | 1.93e+02 |
| nVUnxqHL (DT_static) | 8.39e+00 | 2.71e+01 | **3.2×** | 4.05e+00 | 6.64e+01 |
| XRTnTUjU (DT_static) | 1.20e+01 | 5.80e+01 | **4.8×** | 2.98e+00 | 1.34e+02 |
| wVYSGVTT (DT_mvmt)   | 1.25e+01 | 5.77e+01 | **4.6×** | 4.27e+00 | 1.62e+02 |
| jtYTdZm3 (DT_static) | 4.72e+01 | 8.76e+01 | **1.9×** | 4.40e+00 | 3.24e+02 |

**Observation**: B1 (A1 alone) mu is **3-6x higher** than M0, not 0.74x as T1.1 predicted.  
**Explanation**: T1.1 derivation compared Python vs AEC3 coarse both at equal signal level.  
In practice, M0 uses EMA-smoothed `power_floor × n_partitions` which maintains a floor during quiet frames. B1 uses instantaneous `Σ|X_buf|²` which drops to near-zero in quiet/transition frames → mu spikes.  
AEC3's design intent: instantaneous X² normalization + noise gate together = quiet-frame protection. A1 alone (no gate) replicates only half of AEC3's coarse system.

**B2 mu correction**: With A.2 gate active, quiet-frame bins are zeroed → mean mu drops to 3-4e+00, BELOW M0's EMA-protected value. This is correct behavior — A1+A2 together form the AEC3 coarse update system.

**T1.1 conclusion revised**: The 0.74× ratio holds at equal signal level, but M0 vs B1 shows larger swing in practice because of EMA floor differences. **A1 is structurally unsafe without A2** — the two flags form an inseparable pair in AEC3's coarse filter design.

### T1.2 trace — A2 zero_frac (corrected vs old threshold)

| Case (bucket) | new (0.01870) | old (0.02562) | Δ (new − old) |
|---|---|---|---|
| 9xjhiFbG (FS_static)  | 31.0% | 34.5% | **−3.4pp** |
| xFk7igec (DT_mvmt)    | 58.9% | 61.0% | −2.1pp |
| nVUnxqHL (DT_static)  | 61.2% | 63.4% | −2.1pp |
| XRTnTUjU (DT_static)  | 69.9% | 71.1% | −1.2pp |
| MYrVxVEM (DT_static)  | 67.8% | 69.8% | −2.0pp |
| wVYSGVTT (DT_mvmt)    | 68.4% | 70.4% | −2.0pp |
| qNvSMyUS (FS_static)  | 41.5% | 46.1% | **−4.6pp** |
| jtYTdZm3 (DT_static)  | 65.0% | 68.4% | −3.4pp |

T1.2 correction makes gate **1.2-4.6pp more permissive** across all cases. FS cases see larger correction (3-5pp) vs DT cases (1-2pp), consistent with FS having cleaner signal statistics.

### A2 zero_frac per band (B2, corrected threshold)

| Case | all | LF(31-500Hz) | MF(500-2000Hz) | HF(2000-8000Hz) |
|---|---|---|---|---|
| 9xjhiFbG (FS_static)  | 31.0% | 17.2% | 20.7% | 34.7% |
| xFk7igec (DT_mvmt)    | 58.9% | 49.2% | 53.9% | 60.9% |
| nVUnxqHL (DT_static)  | 61.2% | 44.5% | 58.5% | 63.2% |
| wVYSGVTT (DT_mvmt)    | 68.4% | 53.8% | 59.9% | 71.7% |
| jtYTdZm3 (DT_static)  | 65.0% | 40.7% | 42.1% | 72.7% |

HF bins (2000-8000 Hz) have highest zero_frac (35-73%), LF lowest (17-54%). This is physically correct: HF reference signal is typically weaker (room acoustics attenuate HF), so more HF bins fall below the noise gate.

### shadow_W_norm_mean per variant

| Case (bucket) | M0 | B1 | B2 | B5 |
|---|---|---|---|---|
| 9xjhiFbG (FS_static)  | 83.0 | 80.7 | 74.4 | 71.8 |
| xFk7igec (DT_mvmt)    | 17.7 | 20.4 | 14.6 | 14.6 |
| nVUnxqHL (DT_static)  | 14.8 | 17.4 | 14.9 | 14.9 |
| XRTnTUjU (DT_static)  |  2.2 |  3.2 |  3.1 |  3.1 |
| wVYSGVTT (DT_mvmt)    | 79.7 | 91.9 | 81.3 | 81.3 |
| jtYTdZm3 (DT_static)  | 21.9 | 24.6 | 19.9 | 19.8 |

**B1 W_norm inflated vs M0** (wVYSG: +15%, xFk7: +15%, nVUnxqHL: +17%). B2 corrects this — most cases return near or below M0. This confirms that A.1 alone causes shadow W to grow on spurious updates during quiet/transition frames.

### e2_coarse spectral (mean over utterance)

| Case (bucket) | M0 | B1 | B2 | B2/M0 | B2−B1 (relative) |
|---|---|---|---|---|---|
| 9xjhiFbG (FS_static)  | 2.36e+03 | 4.74e+03 | 4.35e+03 | **1.85×** | −8% |
| xFk7igec (DT_mvmt)    | 1.26e+02 | 2.39e+02 | 1.89e+02 | **1.50×** | −21% |
| nVUnxqHL (DT_static)  | 1.02e+02 | 1.12e+02 | 9.45e+01 | **0.93×** | −16% |
| XRTnTUjU (DT_static)  | 6.75e+01 | 8.50e+01 | 7.84e+01 | **1.16×** | −8% |
| MYrVxVEM (DT_static)  | 1.06e+01 | 1.59e+01 | 1.55e+01 | **1.47×** | −3% |
| wVYSGVTT (DT_mvmt)    | 6.99e+02 | 1.22e+03 | 1.15e+03 | **1.64×** | −6% |
| qNvSMyUS (FS_static)  | 4.44e+01 | 4.60e+01 | 4.59e+01 | **1.03×** | −0.2% |
| jtYTdZm3 (DT_static)  | 1.63e+02 | 2.78e+02 | 2.67e+02 | **1.64×** | −4% |

**B2 vs B1**: B2 consistently IMPROVES over B1 (2-21% reduction in e2_coarse). T1.2 determination Rule 2 CONFIRMED.

**B2 vs M0**: Mixed.
- nVUnxqHL: B2 slightly BETTER (−7%)
- Most cases: B2 WORSE than M0 (1.03×–1.85×)

This is expected: the shadow NLMS with instantaneous X² normalization + gate behaves differently from M0's EMA-smoothed PBFDAF. The key question is whether Bundle B (all 5 flags) brings the COARSE filter's convergence contribution close enough to enable URO selection. That requires the full M_B_corrected composition trace.

### coarse_conv_frac per variant

| Case | M0 | B1 | B2 | B5 |
|---|---|---|---|---|
| 9xjhiFbG (FS_static)  | 0.0% | 0.1% | 0.1% | 0.1% |
| xFk7igec (DT_mvmt)    | 0.4% | 0.1% | 0.2% | 0.2% |
| nVUnxqHL (DT_static)  | 0.4% | 0.0% | 0.1% | 0.1% |
| wVYSGVTT (DT_mvmt)    | 5.5% | 4.1% | 4.1% | 4.1% |
| jtYTdZm3 (DT_static)  | 5.6% | 5.0% | 5.0% | 5.1% |

All variants show <6% coarse_conv_frac. This is the fundamental PBFDKF structural limit — the shadow NLMS coarse output does not meet the strict `e2_coarse < 0.05 × y2` criterion on these cases, regardless of Bundle B flags. This is consistent with Class C isolated classification (A2/A3).

### A3/A4/A5 fire rates

| Case | B3_a3_skip | B4_a4_mask | B5_a5_sat |
|---|---|---|---|
| 9xjhiFbG (FS_static)  | 2.5% | 0.4% | 0.0% |
| xFk7igec (DT_mvmt)    | 0.4% | 0.1% | 0.0% |
| qNvSMyUS (FS_static)  | 3.8% | 0.5% | 0.0% |
| Others                | 0.3-1.5% | 0.0-0.1% | 0.0% |

A4/A5 zero or near-zero on 8-case cohort, consistent with v3.21.14 audit. A3 fires on FS cases (poor excitation on single-speaker), low on DT.

---

## M_B_corrected Definition (CONFIRMED)

**Based on T1.1 + T1.2 findings, M_B_corrected is defined as:**

```python
cfg.use_partition_summed_x2_for_shadow_mu = True   # A.1 — mandatory pair with A.2
cfg.use_aec3_noise_gate_for_shadow = True           # A.2 — MUST use FILTER_NOISE_GATE_POWER_FLOAT
                                                    #        (20075344 int16² = 0.01870)
cfg.use_poor_excitation_gate_for_shadow = True      # A.3
cfg.use_narrowband_mask_for_shadow = True           # A.4
cfg.use_saturation_gate_for_shadow = True           # A.5
# T1.1: shadow_mu_nlms = 0.5 UNCHANGED (no candidate flag; 0.74× per-second gap < 2× threshold)
# T1.2: FILTER_NOISE_GATE_POWER_FLOAT replaces NOISE_GATE_POWER_FLOAT in A.2 block
```

**NOT included** (no candidate flag from T1.1): `use_aec3_coarse_shadow_mu_rate` — analytically unnecessary.

**Key design rule**: A.1 and A.2 are **inseparable** — A.1 without A.2 inflates mu by 3-6× in quiet frames. AEC3's coarse filter design always uses X² normalization with noise gate as a unit.

---

## Summary

| Item | Status | Finding |
|---|---|---|
| T1.1 A.1 mu rate gap | **CLOSED — no flag** | 0.74× per second ratio; A1 unsafe without A2 |
| T1.2 A.2 constant correction | **IMPLEMENTED** | 27509562 → 20075344; 1.2-4.6pp gate relaxation |
| A1+A2 pair | **VERIFIED mandatory** | B2 always improves B1; A1 alone inflates mu 3-6× |
| M_B_corrected definition | **CONFIRMED** | A.1+A.2(corrected)+A.3+A.4+A.5 |
| Composition verdict | **PENDING** | coarse_conv <6% on all variants; full M_B_corrected ladder needed |

**Next step**: Run M0 → M_B_corrected → M_C → M_A → M_D on 8 cases (Track 3).  
Attribution gate: confirm `e2_coarse < 0.5·y2` more reliable in M_B_corrected vs M0 before proceeding to M_C.
