# Gate 1 Verdict — A.1 Full AEC3 Composition (Variant D)

**Date**: 2026-05-25  
**Script**: `python/v3_21_a1_d_12case.py`  
**Candidate (D)**: `use_full_delay_change_chain=True` + `use_linear_filter_output_selection_for_final_output=True`  
**Config**: BALANCED + `enable_cng=True` + `enable_shadow=True`  
**Cohort**: 12 cases (`wav/v3_21_8_cohort/`)  
**Baseline (A)**: both flags OFF = v3.21.6 equivalent  
**Reference (B)**: `use_full_delay_change_chain=True` only (original Gate 1 FAIL)  

> **Gate 0 note (conservative)**: Prior Gate 0 trace verified that Y-path wiring is
> correct (G0.1 PASS) and that D does not over-suppress vs A in energy terms (D/A > 1).
> These are wiring checks, not quality predictions. AECMOS on 12 cases is the definitive
> verdict on whether Y-path companion materially recovers DT speech quality.

## Gate 1 Verdict: **FAIL**

| Gate | Criterion | Result |
|---|---|---|
| G1 | DT_mvmt: no D−A Δdeg < −0.05 | **FAIL** |
| G2 | No per-case catastrophic: DT D−A Δdeg<−0.20 or FS D−A Δecho>+0.20 | **FAIL** |
| G3 | FS bucket D−A Δecho ≤ +0.05 | **PASS** |
| G4 | xFk7igecuke0R5 DT_mvmt D−A Δdeg ≥ −0.05 | **FAIL** |
| G5 | D trigger scope: chain fires only on delay_first/delay_shift | **PASS** |

---

## Bucket Means — A / B / D

Primary: DT→deg (higher=better), FS/NS→echo (lower=better).
D−A = primary gate comparison. D−B = recovery check (how much B regression D recovers).

| Bucket | N | A (baseline) | B (chain-only) | D (full comp.) | D−A | D−B |
|---|---|---|---|---|---|---|
| DT_mvmt | 3 | deg=3.047 | deg=2.641 | deg=2.655 | -0.392 | +0.014 |
| DT_static | 4 | deg=2.378 | deg=2.781 | deg=2.717 | +0.339 | -0.064 |
| FS_mvmt | 1 | echo=4.462 | echo=4.371 | echo=4.111 | -0.351 | -0.260 |
| FS_static | 3 | echo=3.371 | echo=3.067 | echo=3.045 | -0.326 | -0.022 |
| NS | 1 | deg=4.358 | deg=4.358 | deg=4.359 | +0.000 | +0.000 |

---

## Per-case DT_mvmt (G1 / Recovery)

★ = Gate 0 focus case (confirmed B-fail in prior Gate 1). B−A = original regression.

| Case | A_deg | B_deg | D_deg | D−A | D−B | B−A | G1 | Focus |
|---|---|---|---|---|---|---|---|---|
| ZJYUt0O0AEKSQ9LJ8z7t0A_dou | 3.060 | 2.988 | 2.699 | -0.361 | -0.289 | -0.072 | **FAIL** | ★ |
| wVYSGVTTakih9twI4xlDWQ_dou | 3.203 | 2.934 | 3.093 | -0.110 | +0.159 | -0.269 | **FAIL** | ★ |
| xFk7igecuke0R5JMfREyDg_dou | 2.878 | 2.002 | 2.172 | -0.706 | +0.171 | -0.877 | **FAIL** | ★ |

## Per-case DT_static (G2 / Preservation)

B−A reflects delay-chain improvement. D−B shows whether D erases those gains.

| Case | A_deg | B_deg | D_deg | D−A | D−B | B−A | G2 |
|---|---|---|---|---|---|---|---|
| MYrVxVEMxkaE7OuyTUmI0Q_dou | 1.731 | 2.281 | 2.117 | +0.387 | -0.164 | +0.550 | pass |
| XRTnTUjU5kS0mejzCqyCiw_dou | 2.750 | 2.933 | 3.002 | +0.252 | +0.069 | +0.183 | pass |
| jtYTdZm3lUmFVNibJWq8YQ_dou | 2.587 | 3.051 | 3.057 | +0.470 | +0.006 | +0.464 | pass |
| nVUnxqHLr0GTN7shWid1Ow_dou | 2.446 | 2.859 | 2.693 | +0.246 | -0.167 | +0.413 | pass |

## FS Cases — Δecho (G2 / G3)

| Case | Label | A_echo | B_echo | D_echo | D−A | D−B | G2 |
|---|---|---|---|---|---|---|---|
| 0I0XMl3M0ECO0U1N0cJvpg_far | FS_mvmt | 4.462 | 4.371 | 4.111 | -0.351 | -0.260 | pass |
| 9xjhiFbGo06hdQIsHTS6qA_far | FS_static | 2.892 | 2.089 | 2.191 | -0.701 | +0.102 | pass |
| qNvSMyUSXUyrDGpOw7s6qg_far | FS_static | 3.594 | 3.596 | 3.560 | -0.034 | -0.036 | pass |
| xQEUtY2pWUi7v1X93TF2AA_far | FS_static | 3.626 | 3.516 | 3.383 | -0.243 | -0.133 | pass |

G3: FS_mvmt D−A=-0.351 OK / FS_static D−A=-0.326 OK

---

## Trigger Scope Trace (Variant D)

Chain must fire ONLY on `delay_first` / `delay_shift`. Y-path columns are informational.

| Case | delay_first | delay_shift | EPV | SR | Y_init | Y_total | Scope |
|---|---|---|---|---|---|---|---|
| xFk7igecuke0R5JMfREyDg_dou | [104] | [] | [] | [1401] | 193 | 193 | PASS |
| wVYSGVTTakih9twI4xlDWQ_dou | [75] | [] | [] | [405, 538, 1196, 2828] | 135 | 135 | PASS |
| ZJYUt0O0AEKSQ9LJ8z7t0A_dou | [141] | [] | [465, 897, 1153] | [3772] | 225 | 225 | PASS |

---

## Analysis

**Gate 1 FAIL.** Variant D does not pass all gate criteria.

Failure summary:
- G1 (DT_mvmt regression): 3 case(s) D−A < −0.05
  - ZJYUt0O0AEKSQ9LJ8z7t: D−A=−0.361
  - wVYSGVTTakih9twI4xlD: D−A=−0.110
  - xFk7igecuke0R5JMfREy: D−A=−0.706
- G2 (catastrophic): 2 cases D−A < −0.20 (ZJYUt0O0 −0.361 / xFk7 −0.706)
- G4 (xFk7 DT_mvmt): D−A=−0.706

### Key observations

**Partial D−B recovery (mixed)**: D partially recovers B's regression on xFk7 (+0.171)
and wVYSGVTT (+0.159), but makes ZJYUt0O0 *worse* than B by −0.289. The Y-path companion
helps on the two worst-regression cases but hurts the smallest-regression case — suggesting
UseLinearFilterOutput has asymmetric per-case effects that cannot be predicted from
Gate 0 energy analysis alone.

**Root cause**: The full delay-change chain (W=0, H_error=100, AecState reset) creates a
re-convergence period longer than the `usable_linear=False` window. Y_init in the Gate 1
trace (193/135/225 frames) is larger than the Gate 0 window counts (40/60/84) — the
initial_state window persists or re-fires beyond the first Gate 0 window. Despite Y-path
covering those frames, the E-path damage during the remaining `usable_linear=True` period
still causes severe DT regression. The problem is the convergence duration, not wiring.

**DT_static preserved**: All 4 DT_static cases retain the B improvements (D−A: +0.247
to +0.470). The delay-chain DT_static gains are largely preserved under D.

**FS improved**: G3 passes easily (D−A: −0.351 FS_mvmt / −0.326 FS_static mean). FS
echo is further reduced vs B as well. UseLinearFilterOutput benefits FS unexpectedly.

### Classification implication

Full AEC3 composition (delay-change chain + UseLinearFilterOutput Y-path companion) has
been tested end-to-end. Gate 1 FAIL on all 3 DT_mvmt cases confirms that the Y-path
companion alone cannot bridge the DT speech quality gap during re-convergence. This
constitutes the evidence base for Category A reclassification.

**Category A (reclassification)**: A.1 `use_full_delay_change_chain=True` with full
AEC3 composition (variant D) is structurally incompatible with PBFDKF architecture on
DT_mvmt cases. The incompatibility is in re-convergence duration, not wiring. Both
sub-chains of AEC3's echo-path-change response have been tested; neither alone nor
combined resolves the DT_mvmt regression.

### Next step (requires user authorization)

Per the original gate rules: Gate 1 FAIL with full composition → Category A classification
valid. Functional adaptations R1 (NE gate ERLE reset) and/or R3 (skip ERLE reset on
delay-change) may now be authorized as separate A.1 companion tests.

**These are NOT automatically authorized.** User must explicitly enable R1/R3 testing
before any functional adaptation code is introduced.

---

## Hard Rules (unchanged)

- A.1 remains OPEN until user closes
- No 800-case without explicit user authorization
- No v3.22 framing of A.1
- R1/R3 functional adaptation: Gate 1 FAIL NOW CONFIRMED — requires user authorization before testing

---

## Config

```python
# Variant D — full AEC3 composition
use_full_delay_change_chain = True
use_linear_filter_output_selection_for_final_output = True
# BALANCED preset + enable_cng=True + enable_shadow=True
```
