# P53 Step 0 — Innovation Audit Verdict

**Date**: 2026-05-12
**Source data**: `/tmp/p53_audit/audit_summary.json`
**Audit code path**: production main HEAD, `use_res_refactored=False`, `trace_p53_innovation=True`.

## T0 outcome

**Outcome**: `T0.E`

**Reason**: ambiguous: wildly 1.75, stable 5.67 do not fit A/B/C/D thresholds.
The wildly cohort r_voice is *lower* than the stable cohort — the opposite of
the Q-underestimation signature (`wildly r_voice ≥ 3.0 AND stable < 1.5`)
that T0.A would predict. No A/B/C/D branch matches; per design lock §2.5
this is the committed outcome.

## Key finding — production architecture absorbs innovation through adaptive R

Production PBFDKF maintains `R = max(EMA(|error_spec|², α=0.95), δ) · R_scale`
([aec.py:1088-1095](../python/aec.py#L1088)). Because `R` is itself an EMA of
innovation power, `C_obs ≈ R` by construction on any frame where the EMA has
settled. The canonical Kalman orthogonality test (Mehra 1970) assumes `R` is a
fixed measurement-noise variance — that assumption does not hold here, so the
`r = C_obs / C_exp` diagnostic is partially degenerate as a Q-underestimation
detector.

Even accounting for that degeneracy, the observed sign is *contrary* to the
canonical hypothesis:

- Stable cohort `r_voice ≈ 5.67` (innovation > expected — modest "Q-too-small"
  signal on the bulk distribution)
- Wildly cohort `r_voice ≈ 1.75` (innovation ≈ expected — *no* Q-too-small
  signal on the cohort tail where divergence happens)
- Catastrophe target case `qNvSMyU…` `r_voice ≈ 0.37` (innovation < expected
  — the divergence case is *over-fitting* by the canonical metric, not
  under-fitting)

The wildly-cohort low ratio plausibly arises from already-adaptive Q gating
(`Q_modulated = Q · q_scale, q_scale ∈ [0.1, 1.0]` driven by `mu_scale`) plus
PathChangeRegimeHandler's `boost_q` intervention, both of which already lift
`P_diag` on the cases where the catastrophic-defence handler fires. By the
time we measure, `C_exp = far_psd · P_diag + R` is already inflated, so
`r = C_obs / C_exp < 1` even though the underlying issue is genuine path
nonstationarity.

**Conclusion**: the canonical Q-underestimation hypothesis (P53 design lock
§0.3 + §3.1) does not match the cohort. Adaptive-Q (Direction 1), Strong
Tracking P-inflation (Direction 2), and IMM-bank regime switching
(Direction 4) all share the premise that the linear filter's process-noise
model is too rigid; none of those directions has supporting evidence on this
data. Variational Bayesian linear+NL (Direction 3) is high-cost and has no
positive signal either.

## Phase commitments (per design lock §2.4)

- Phase L → **close Phase L**
- Phase R → **R3 (Lee-Kim SPP, least risky)**

## Aggregate r_voice (design-lock formula `C_exp = far_psd * P_diag + R`)

| Category | n cases | r_voice mean (across cases) | r_voice max |
|---|---|---|---|
| stable | 9 | 5.674 | 20.292 |
| listener | 8 | 2.963 | 13.061 |
| wildly | 7 | 1.752 | 5.450 |

## Per-case detail

| Category | Regime (A.0R.3) | Stem | n_frames | n_far_active | r_voice mean (design) | p99 (design) | mean (partition) |
|---|---|---|---|---|---|---|---|
| listener | stable | `LHsrJBRGnUKiMC2mihEr0g_doubletalk` | 1348 | 1348 | 13.061 | 234.554 | 33.811 |
| listener | mildly_nonstationary | `pU21kfoo0UOz0fPMJFfydg_doubletalk` | 1591 | 1591 | 3.315 | 23.781 | 5.081 |
| listener | mildly_nonstationary | `XRTnTUjU5kS0mejzCqyCiw_doubletalk` | 1022 | 1022 | 1.558 | 6.548 | 3.180 |
| listener | mildly_nonstationary | `MkSLte0FTkqybGcLTwA3Tw_farend_singletalk_with_movement` | 1584 | 1584 | 1.553 | 8.517 | 3.571 |
| listener | mildly_nonstationary | `SgKY30fjT0G8e3kQL0RHSQ_doubletalk_with_movement` | 1585 | 1585 | 1.283 | 6.765 | 2.094 |
| listener | mildly_nonstationary | `SUYzW4QT30yxKUq7OGvZKg_farend_singletalk` | 1424 | 1424 | 1.157 | 6.575 | 2.079 |
| listener | mildly_nonstationary | `afHuFvflAkaH7Pr85kheUQ_doubletalk_with_movement` | 1537 | 1537 | 1.062 | 5.474 | 2.087 |
| listener | stable | `XqvGR01tJkan17zltLs38Q_doubletalk_with_movement` | 1447 | 1447 | 0.713 | 3.141 | 1.901 |
| stable | mildly_nonstationary | `uLl640xveUuHp2kEtOCTeQ_nearend_singletalk` | 33 | 33 | 20.292 | 130.885 | 25.293 |
| stable | stable | `WnDjVFWmC0m0WhVq22mRlQ_nearend_singletalk` | 90 | 90 | 14.575 | 160.697 | 20.954 |
| stable | stable | `ZtGitIxrzU0ILwu0HACaaw_doubletalk` | 1334 | 1334 | 3.668 | 11.979 | 4.661 |
| stable | mildly_nonstationary | `KgZ0y2EQJ0a4jvtsznBrvw_doubletalk` | 1131 | 1131 | 3.325 | 53.267 | 7.706 |
| stable | stable | `XNOJLHM8e0aUHP4rWEAAtQ_doubletalk` | 1731 | 1731 | 2.371 | 7.150 | 4.753 |
| stable | stable | `Wv6yp6N1L0WqQ6ZLn6nD8g_doubletalk` | 1418 | 1418 | 1.870 | 5.193 | 2.990 |
| stable | mildly_nonstationary | `yc5bFUGsR0GSfiGwTTpRWg_doubletalk_with_movement` | 2184 | 2184 | 1.774 | 7.759 | 3.907 |
| stable | stable | `w0QrMwsZ5kGoJjRWvP0iKg_doubletalk` | 1296 | 1296 | 1.691 | 5.033 | 3.112 |
| stable | stable | `wr2CEenGL0Oec4b1GW5Zww_doubletalk` | 1297 | 1297 | 1.503 | 5.296 | 2.660 |
| wildly | wildly_nonstationary | `0I0XMl3M0ECO0U1N0cJvpg_farend_singletalk_with_movement` | 1708 | 1708 | 5.450 | 186.980 | 10.836 |
| wildly | wildly_nonstationary | `SgKY30fjT0G8e3kQL0RHSQ_doubletalk` | 1164 | 1164 | 3.228 | 45.159 | 5.889 |
| wildly | wildly_nonstationary | `P10GsQvhskKx3fB06Zv4Yg_farend_singletalk` | 1290 | 1290 | 1.248 | 6.013 | 2.507 |
| wildly | wildly_nonstationary | `s90M7MOTBkqaV4nQPLhKbA_doubletalk` | 1858 | 1858 | 1.052 | 5.055 | 1.746 |
| wildly | wildly_nonstationary | `LQhlYoXXiUevFuxMKwWB0Q_doubletalk` | 1614 | 1614 | 0.751 | 4.878 | 1.564 |
| wildly | wildly_nonstationary | `qNvSMyUSXUyrDGpOw7s6qg_farend_singletalk` | 1749 | 1749 | 0.369 | 2.208 | 0.838 |
| wildly | wildly_nonstationary | `LQhlYoXXiUevFuxMKwWB0Q_farend_singletalk` | 1309 | 1309 | 0.163 | 2.304 | 0.460 |

## Constants (locked at design-lock signing)

- `ALPHA_OBS` = 0.95
- Voice band: 300.0–3000.0 Hz
- T0 thresholds: wildly ≥ 3.0; stable < 1.5; disjoint margin 1.0
- Stable sample seed: `random.Random(42).sample(..., 12)`
- CNG seed per case: `np.random.seed(42)`

## Methodology caveats (recorded for downstream phases)

1. **3 of 12 stable-cohort samples produced 0 frames** (`ON00x93…`,
   `veoTpvS3…`, `vcmW9eY9…` — all `nearend_singletalk` with `|lpb|_max < 0.005`).
   PBFDKF only updates weights when `far_hop_energy > 1e-4`
   ([aec.py:1011](../python/aec.py#L1011)); these cases never trip the
   threshold so the audit captures nothing. Effective stable `n = 9`.
2. **2 of 9 effective stable samples are NE-only with sparse far activity**
   (`uLl640…` 33 frames; `WnDjVFW…` 90 frames). Their `r_voice` means (20.29,
   14.58) dominate the stable aggregate via the small-`far_psd · P_diag`
   denominator. They are *not* representative of bulk stable behaviour; the
   remaining 7 doubletalk stable samples have `r_voice` means in the
   1.5 – 3.7 band. Even with this caveat, no stable sample lies in the
   `< 1.5` target band that T0.A would require.
3. **Whole-recording regime classifier disagrees with the supplied case
   list** on several entries — `LHsrJBRGnUKiMC2mihEr0g_doubletalk` is
   labelled `stable` by `AcousticRegimeClassifier` despite being a listener
   anchor, and three doubletalk stable samples re-classify as
   `mildly_nonstationary`. This is expected — the listener anchor set is
   trace-chain-curated, not regime-classifier-curated. The audit reports both
   labels per case.
4. **`r_voice (design)` vs `r_voice (partition)`** — both formulas computed.
   The partition-aware form (`C_exp = Σ_p P[p]·|X[p]|² + R + δ`, i.e. the
   actual Kalman denominator used in the production gain calculation) is
   reported alongside the design-lock formula. Conclusions are unchanged
   under either form: both show stable ≥ wildly, contrary to T0.A.

Verdict immutable post-write.
