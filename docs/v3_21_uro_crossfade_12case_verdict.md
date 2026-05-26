# v3.21 URO + FormLinearFilterOutput Crossfade — 12-case Verdict

**Gate 0**: byte-equal M0 vs BALANCED = PASS


## Per-case AECMOS (Δ vs M0)

| Case | Bucket | Metric | M0 | M1 Δ | M2 Δ | M3 Δ | M4 Δ | v3.21.6 |
|---|---|---|---|---|---|---|---|---|
| ZJYUt0O0AEKSQ9LJ8z7t0A_doubl | DT_mvmt | deg | 3.058 | -0.148 | -0.078 | -0.078 | -0.078 | 2.270 |
| wVYSGVTTakih9twI4xlDWQ_doubl | DT_mvmt | deg | 3.205 | +0.022 | -0.136 | -0.136 | -0.136 | 2.741 |
| xFk7igecuke0R5JMfREyDg_doubl | DT_mvmt | deg | 2.881 | -1.009 | -0.955 | -0.955 | -0.955 | 2.319 |
| MYrVxVEMxkaE7OuyTUmI0Q_doubl | DT_static | deg | 1.724 | -0.301 | -0.315 | -0.315 | -0.315 | 2.166 |
| XRTnTUjU5kS0mejzCqyCiw_doubl | DT_static | deg | 2.751 | +0.049 | +0.076 | +0.076 | +0.076 | 3.950 |
| jtYTdZm3lUmFVNibJWq8YQ_doubl | DT_static | deg | 2.587 | -0.151 | +0.033 | +0.033 | +0.033 | 2.700 |
| nVUnxqHLr0GTN7shWid1Ow_doubl | DT_static | deg | 2.424 | +0.016 | +0.003 | +0.003 | +0.003 | 2.893 |
| 0I0XMl3M0ECO0U1N0cJvpg_faren | FS_mvmt | echo | 4.375 | +0.055 | -0.154 | -0.154 | -0.154 | 4.262 |
| 9xjhiFbGo06hdQIsHTS6qA_faren | FS_static | echo | 4.565 | -1.517 | -1.731 | -1.731 | -1.731 | 2.367 |
| qNvSMyUSXUyrDGpOw7s6qg_faren | FS_static | echo | 3.972 | +0.038 | -0.080 | -0.080 | -0.080 | 3.550 |
| xQEUtY2pWUi7v1X93TF2AA_faren | FS_static | echo | 3.712 | +0.063 | -0.031 | -0.031 | -0.031 | 3.387 |
| 014AzuqPZku2004NbTTmcA_neare | NS | deg | 4.356 | +0.000 | +0.000 | +0.000 | +0.000 | 4.355 |

## Bucket means Δ vs M0

| Bucket | Metric | M1 Δ | M2 Δ | M3 Δ | M4 Δ |
|---|---|---|---|---|---|
| DT_mvmt | deg | -0.378 | -0.389 | -0.389 | -0.389 |
| DT_static | deg | -0.097 | -0.051 | -0.051 | -0.051 |
| FS_mvmt | echo | +0.055 | -0.154 | -0.154 | -0.154 |
| FS_static | echo | -0.472 | -0.614 | -0.614 | -0.614 |
| NS | deg | +0.000 | +0.000 | +0.000 | +0.000 |

## Gate check per variant

| Gate | Criterion | M1 | M2 | M3 | M4 |
|---|---|---|---|---|---|
| G1 | No DT bucket Δdeg < −0.05 | FAIL | FAIL | FAIL | FAIL |
| G2 | No per-case: DT Δdeg < −0.20 OR FS Δecho > +0.20 | FAIL | FAIL | FAIL | FAIL |
| G3 | 4-stress cases each Δ ≥ −0.10 | FAIL | FAIL | FAIL | FAIL |
| G4 | FS bucket Δecho ≤ +0.05 | FAIL | PASS | PASS | PASS |

## Ship decision

- **M1**: FAIL — not a ship candidate
- **M2**: FAIL — not a ship candidate
- **M3**: FAIL — not a ship candidate
- **M4**: FAIL — not a ship candidate