# Gate 0 Verdict — Variant H (F + C1-C5 shadow protection)

**Date**: 2026-05-26
**Script**: `python/v3_21_a1_h_gate0.py`
**Variant H**: F + `use_partition_summed_x2_for_shadow_mu=True` (C1) + `use_aec3_noise_gate_for_shadow=True` (C2) + `use_poor_excitation_gate_for_shadow=True` (C3) + `use_narrowband_mask_for_shadow=True` (C4) + `use_saturation_gate_for_shadow=True` (C5)
**Variant F**: delay-change chain + ULO + UseRefinedOutput (prior baseline)

## Gate 0: FAIL

| Gate | Criterion | Result |
|---|---|---|
| G0.1 | UseRefinedOutput fires ≥1 DT_mvmt frame | **PASS** |
| G0.2 | cond2 fires in re-convergence window | **PASS** |
| G0.3 | coarse path in DT_mvmt re-convergence windows | **PASS** |
| G0.4 | DT_static stable cond2 rate < 15% | **PASS** |
| G0.H1 | e2_coarse(H)/e2_coarse(F) ≤ 1.1 in coarse windows | **FAIL** |
| G0.H2 | shadow_w_norm ratio [0.20, 5.00] | **PASS** |
| G0.H3 | wVYSGVTT rescue preserved | **PASS** |

## DT_mvmt trace (H)

| Case | frames | uro_fire | cond2 | reconv_cond2 | reconv_coarse | e2_ratio | wn_H/F |
|---|---|---|---|---|---|---|---|
| ZJYUt0O0AEKSQ9LJ8z7t0A_doublet | 3791 | 2158 | 1774 | 66 | 75 | 1.041 | 30.8289/38.2703 |
| wVYSGVTTakih9twI4xlDWQ_doublet | 3666 | 1702 | 1267 | 43 | 52 | 1.246 | 79.2456/81.9237 |
| xFk7igecuke0R5JMfREyDg_doublet | 3678 | 969 | 831 | 33 | 33 | 1.291 | 11.5933/8.9200 |

## DT_static G0.4 (H)

| Case | stable_frames | stable_cond2 | rate | G0.4 |
|---|---|---|---|---|
| MYrVxVEMxkaE7OuyTUmI0Q_doublet | 0 | 0 | 0.0% | PASS |
| XRTnTUjU5kS0mejzCqyCiw_doublet | 0 | 0 | 0.0% | PASS |
| jtYTdZm3lUmFVNibJWq8YQ_doublet | 0 | 0 | 0.0% | PASS |
| nVUnxqHLr0GTN7shWid1Ow_doublet | 0 | 0 | 0.0% | PASS |

## C1-C5 gate fire summary

| Case | C1 A1_frac | C2 zero_frac | C3 skip_frac | C4 mask_frac | C5 skip_frac |
|---|---|---|---|---|---|
| ZJYUt0O0AEKSQ9LJ8z7t0A_doublet | 26.30% | 58.81% | 0.76% | 0.06% | 0.00% |
| wVYSGVTTakih9twI4xlDWQ_doublet | 28.21% | 22.48% | 0.33% | 0.00% | 0.00% |
| xFk7igecuke0R5JMfREyDg_doublet | 40.97% | 20.44% | 0.33% | 0.13% | 0.00% |
| MYrVxVEMxkaE7OuyTUmI0Q_doublet | 33.22% | 23.45% | 0.66% | 0.10% | 0.00% |
| XRTnTUjU5kS0mejzCqyCiw_doublet | 23.66% | 15.32% | 0.34% | 0.00% | 0.00% |
| jtYTdZm3lUmFVNibJWq8YQ_doublet | 49.56% | 43.61% | 0.33% | 0.03% | 0.00% |
| nVUnxqHLr0GTN7shWid1Ow_doublet | 30.41% | 17.13% | 0.28% | 0.04% | 0.00% |

## Hard rules

- A.1 remains OPEN until user closes
- No 800-case without explicit user authorization
- No v3.22 framing of A.1
- R1/R3 functional adaptation: NOT authorized
- No H+G combination until H and G have separate attribution
- ZJYUt0O0 failures with Δframes > 10: mark as Variant G candidate, not H limitation
