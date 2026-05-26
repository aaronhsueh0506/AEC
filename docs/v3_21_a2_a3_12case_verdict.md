# A2_A3 12-case AECMOS Verdict

**Date**: 2026-05-26  
**Candidate**: `use_current_e2_refined_in_h_error_denominator=True` (A2) + `use_per_bin_h_error_refresh=True` (A3)  
**Config**: BALANCED + `enable_cng=True` + `enable_shadow=True`; all other candidate flags explicitly OFF  
**Cohort**: 12 cases (`wav/v3_21_8_cohort/`)  
**Baseline (A = M0)**: all candidate flags OFF = v3.21.6 equivalent  
**NOT included**: Bundle B (shadow gates), URO/crossfade, H_ERROR_CEIL=2.0, full_delay_chain, v3.22 knobs  

## Verdict: **FAIL**

| Gate | Criterion | Result |
|---|---|---|
| G1 | 9xjhiFbG echo ≥ M0−0.05 (nores LF artifact) | **FAIL** (Δecho=-2.512) |
| G2 | qNvSMyUS clean guard \|Δecho\| < 0.2 | **FAIL** (Δecho=-0.328) |
| G3 | No DT Δdeg < −0.2 (catastrophic) | **FAIL** |
| G4 | No DT usable_linear collapse (delta-based) | **PASS** |

---

## Bucket Means Δ vs M0

| Bucket | N | M0 (OFF) | A2_A3 (ON) | Δ(B−A) |
|---|---|---|---|---|
| DT_mvmt | 3 | deg=3.048 | deg=2.909 | -0.138 |
| DT_static | 4 | deg=2.372 | deg=2.585 | +0.213 |
| FS_mvmt | 1 | echo=4.375 | echo=4.250 | -0.124 |
| FS_static | 3 | echo=4.083 | echo=3.107 | -0.976 |
| NS | 1 | deg=4.356 | deg=4.356 | +0.000 |

---

## DT Cases — Δdeg (G3 threshold −0.20) + G4 usable_linear

| Case | Label | A_deg | B_deg | Δdeg | v3.21.6 | Δvs6 | UL_M0 | UL_B | G3 |
|---|---|---|---|---|---|---|---|---|---|
| ZJYUt0O0AEKSQ9 | DT_mvmt | 3.058 | 3.048 | -0.010 | 2.270 | +0.778 | 7.7% | 5.6% | PASS |
| wVYSGVTTakih9t | DT_mvmt | 3.205 | 3.635 | +0.430 | 2.741 | +0.894 | 10.2% | 9.9% | PASS |
| xFk7igecuke0R5 | DT_mvmt | 2.881 | 2.045 | -0.836 | 2.319 | -0.274 | 4.9% | 1.4% | **FAIL** |
| MYrVxVEMxkaE7O | DT_static | 1.724 | 2.171 | +0.447 | 2.166 | +0.005 | 1.8% | 1.4% | PASS |
| XRTnTUjU5kS0me *(pre-existing low UL)* | DT_static | 2.751 | 2.875 | +0.124 | 3.950 | -1.075 | 0.0% | 0.0% | PASS |
| jtYTdZm3lUmFVN | DT_static | 2.587 | 2.389 | -0.198 | 2.700 | -0.311 | 6.9% | 5.8% | PASS |
| nVUnxqHLr0GTN7 | DT_static | 2.424 | 2.904 | +0.480 | 2.893 | +0.011 | 5.3% | 5.7% | PASS |

## FS Cases — Δecho (G1, G2)

| Case | Label | A_echo | B_echo | Δecho | v3.21.6 | Δvs6 | Gate |
|---|---|---|---|---|---|---|---|
| 0I0XMl3M0ECO0U | FS_mvmt | 4.375 | 4.250 | -0.124 | 4.262 | -0.012 |  |
| 9xjhiFbGo06hdQ | FS_static | 4.565 | 2.053 | -2.512 | 2.367 | -0.314 | G1 **FAIL** |
| qNvSMyUSXUyrDG | FS_static | 3.972 | 3.644 | -0.328 | 3.550 | +0.094 | G2 **FAIL** |
| xQEUtY2pWUi7v1 | FS_static | 3.712 | 3.625 | -0.088 | 3.387 | +0.238 |  |

## Nores LF Band (0–200 Hz) — Primary Artifact Case

| Case | M0 LF dB | A2_A3 LF dB | Δ dB |
|---|---|---|---|
| 9xjhiFbGo06hdQ | 18.02 | 11.06 | -6.96 |

Isolation trace reference: A2_A3 nores LF Δ = −4.97 dB (pre-computed).

---

## G4 Pre-existing usable_linear (excluded from collapse check)

| Case | M0 UL | Note |
|---|---|---|
| ZJYUt0O0AEKSQ9 | M0=7.7%(low M0, excluded) | Pre-existing; not caused by A2 |
| wVYSGVTTakih9t | M0=10.2%(low M0, excluded) | Pre-existing; not caused by A2 |
| xFk7igecuke0R5 | M0=4.9%(low M0, excluded) | Pre-existing; not caused by A2 |
| MYrVxVEMxkaE7O | M0=1.8%(low M0, excluded) | Pre-existing; not caused by A2 |
| XRTnTUjU5kS0me | M0=0.0%(pre-existing) | Pre-existing; not caused by A2 |
| jtYTdZm3lUmFVN | M0=6.9%(low M0, excluded) | Pre-existing; not caused by A2 |
| nVUnxqHLr0GTN7 | M0=5.3%(low M0, excluded) | Pre-existing; not caused by A2 |

---

## Conclusion

**FAIL.** A2_A3 does not pass all 12-case gates. Failure summary:

- G1 (9xjhi artifact): Δecho=-2.512 — echo regressed
- G2 (qNvSMy guard): Δecho=-0.328 — guard case regressed
- G3 (DT catastrophic): 1 case(s)
  - xFk7igecuke0R5 Δdeg=-0.836

---

## Config (A2_A3 candidate)

```python
# M0 baseline — all candidate flags OFF
use_partition_summed_x2_for_h_error_gain       = False
use_current_e2_refined_in_h_error_denominator  = False
use_per_bin_h_error_refresh                    = False
use_aec3_h_error_ceil                          = False
use_partition_summed_x2_for_shadow_mu          = False
use_aec3_noise_gate_for_shadow                 = False
use_poor_excitation_gate_for_shadow            = False
use_narrowband_mask_for_shadow                 = False
use_saturation_gate_for_shadow                 = False
use_refined_output_selection_for_linear_path   = False
form_linear_filter_crossfade_enabled           = False
use_full_delay_change_chain                    = False
transparent_mode_enabled                       = False

# A2_A3 variant: only these two flags change
use_current_e2_refined_in_h_error_denominator  = True   # A2
use_per_bin_h_error_refresh                    = True   # A3
```
