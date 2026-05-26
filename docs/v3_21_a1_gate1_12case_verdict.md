# Gate 1 Verdict — A.1 `use_full_delay_change_chain`

**Date**: 2026-05-25  
**Candidate**: `use_full_delay_change_chain=True`  
**Config**: BALANCED + `enable_cng=True` + `enable_shadow=True`; `use_q1_true_delay_transient_leakage=False`, `use_q1_terminate_on_non_delay_epc=False`, `use_aec3_zero_filter_on_epc=False` (all partial flags OFF)  
**Cohort**: 12 cases (`wav/v3_21_8_cohort/`)  
**Baseline (A)**: `use_full_delay_change_chain=False` = v3.21.6 equivalent  

## Gate 1 Verdict: **FAIL**

| Gate | Criterion | Result |
|---|---|---|
| G1 | DT_mvmt: no Δdeg < −0.05 | **FAIL** |
| G2 | No per-case catastrophic (DT Δdeg<−0.20 or FS Δecho>+0.20) | **FAIL** |
| G3 | FS bucket means Δecho ≤ +0.05 | **PASS** |
| G4 | xFk7igecuk DT_mvmt Δdeg ≥ −0.05 | **FAIL** |
| G5 | Flag-OFF byte-equal vs standard BALANCED | **PASS** |
| Trigger scope | Chain fires only on delay_first/delay_shift | **PASS** |

---

## Bucket Means Δ vs A (baseline OFF = v3.21.6)

Primary metric: DT→deg (higher=better), FS→echo (lower=better).

| Bucket | N | A (OFF) | B (ON) | Δ(B−A) |
|---|---|---|---|---|
| DT_mvmt | 3 | deg=3.047 | deg=2.612 | -0.435 |
| DT_static | 4 | deg=2.372 | deg=2.768 | +0.396 |
| FS_mvmt | 1 | echo=4.411 | echo=4.319 | -0.092 |
| FS_static | 3 | echo=3.924 | echo=3.812 | -0.112 |
| NS | 1 | deg=4.359 | deg=4.359 | +0.000 |

---

## Per-case DT_mvmt Δdeg (G1)

| Case | A_deg | B_deg | Δdeg | v3.21.6 | Δvs6 | G1 |
|---|---|---|---|---|---|---|
| ZJYUt0O0AEKSQ9 | 3.058 | 2.987 | -0.071 | 2.270 | +0.717 | **FAIL** |
| wVYSGVTTakih9t | 3.205 | 2.936 | -0.268 | 2.741 | +0.195 | **FAIL** |
| xFk7igecuke0R5 ← xFk7 | 2.879 | 1.914 | -0.965 | 2.319 | -0.405 | **FAIL** |

## Per-case DT_static Δdeg (G2 threshold −0.20)

| Case | A_deg | B_deg | Δdeg | v3.21.6 | Δvs6 | Note |
|---|---|---|---|---|---|---|
| MYrVxVEMxkaE7O | 1.730 | 2.283 | +0.552 | 2.166 | +0.117 |  |
| XRTnTUjU5kS0me | 2.750 | 2.928 | +0.178 | 3.950 | -1.022 | sensitive |
| jtYTdZm3lUmFVN | 2.586 | 3.048 | +0.462 | 2.700 | +0.348 |  |
| nVUnxqHLr0GTN7 | 2.421 | 2.815 | +0.393 | 2.893 | -0.078 | sensitive EPV |

## FS Cases — Δecho (G2 > +0.20; G3 bucket ≤ +0.05)

| Case | Label | A_echo | B_echo | Δecho | v3.21.6 | Δvs6 | G2 |
|---|---|---|---|---|---|---|---|
| 0I0XMl3M0ECO0U | FS_mvmt | 4.411 | 4.319 | -0.092 | 4.262 | +0.057 | PASS |
| 9xjhiFbGo06hdQ | FS_static | 4.569 | 4.332 | -0.237 | 2.367 | +1.965 | PASS |
| qNvSMyUSXUyrDG | FS_static | 3.594 | 3.595 | +0.002 | 3.550 | +0.045 | PASS |
| xQEUtY2pWUi7v1 | FS_static | 3.610 | 3.510 | -0.100 | 3.387 | +0.123 | PASS |

G3 detail: FS_mvmt Δecho=-0.092 OK / FS_static Δecho=-0.112 OK

---

## Trigger Scope Trace

Chain must fire ONLY on `delay_first` / `delay_shift`. EPV and shadow_rise must NOT trigger.

| Case | delay_first | delay_shift | EPV frames | SR frames | Scope | H_on+1 | H_off+1 |
|---|---|---|---|---|---|---|---|
| XRTnTUjU5kS0me | [109] | [] | [] | [] | PASS | 100.0 | 0.2 |
| nVUnxqHLr0GTN7 | [119] | [] | [] | [1111] | PASS | 100.0 | 2.6 |
| xFk7igecuke0R5 | [104] | [] | [] | [1401] | PASS | 100.0 | 0.3 |

H_error ceiling = 100.0. ON should be ~100 at event+1; OFF should be ~0.4.

---

## Comparison with Previous Q1 Leakage-Window Phase 3

| Case | Phase 3 (leakage-only) | A.1 (full chain) | Recovery? |
|---|---|---|---|
| xFk7 DT_mvmt | Δdeg ~−0.05 (previous gate fail) | see G4 above | see G4 |
| XRTnTUjU | not in Phase 3 DT_mvmt | see DT_static above | — |
| nVUnxqHLr | not in Phase 3 DT_mvmt | see DT_static above | — |

---

## Conclusion

**Gate 1 FAIL.** A.1 candidate does not pass all gates on the 12-case cohort.

Failure summary:
- G1 (DT_mvmt regression): 3 case(s) exceed −0.05 threshold
- G2 (catastrophic outlier): 2 case(s)
- G4 (xFk7 DT_mvmt): Δdeg=-0.965

---

## Config (A.1 candidate)

```python
use_full_delay_change_chain = True   # AEC3 full echo-path-change cascade
# All Q1/partial flags OFF (default):
use_q1_true_delay_transient_leakage = False
use_q1_terminate_on_non_delay_epc   = False
use_aec3_zero_filter_on_epc         = False
```
