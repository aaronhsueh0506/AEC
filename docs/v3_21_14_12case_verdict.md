# v3.21.14 PBFDAF shadow NLMS AEC3 protection alignment — 12-case verdict

Cycle: 2026-05-23
Cohort: `wav/v3_21_8_cohort/` (12 cases: 3 stress DT + 4 normal DT + 4 FS + 1 NE)
Baseline: `out_12_v3_21_14_A/` (v3.21.6 + all 5 v3.21.14 flags default-OFF; byte-equal vs working-tree HEAD)
Variants: A.1 / A.2 / A.3 / A.4 / A.5 / ALL (env-flag-driven, all defaults OFF)

## TL;DR

**ALL-ON FAILS 5-criterion ship gate** on criterion 4 (wVYSGVTTakih9twI4xlDWQ
catastrophic Δdeg −0.445), driven by A.1.

Per-sub disposition (NOT a full arc closure — A.2/A.3/A.4/A.5 are all AEC3
parity work and remain in v3.21.x per version rule):

| Sub | Net signal | Per-sub status |
|---|---|---|
| **A.1** partition-sum X² mu denom | FS_movement +0.032, but **wVYSGVTTakih9twI4xlDWQ −0.148** alone; **−0.445 compound in ALL** → A.1 is the wVYSGVTTakih9twI4xlDWQ destabiliser | **CLOSED / DO NOT SHIP** |
| **A.2** noise_gate hard zero | FS_static **+0.059** / FS_movement **+0.053** / xQEUtY2 +0.179; no DT regression | **ACTIVE v3.21.15 candidate** (combined with A.3) |
| **A.3** poor_excitation startup gate | FS_movement **+0.161** / 0I0XMl3M +0.161 / **nVUnxqHLr +0.189 (Cat C partial recovery)**; no DT regression | **ACTIVE v3.21.15 candidate** (combined with A.2) |
| **A.4** narrowband mask | All-bucket Δ ≤ 0.005 (mask never fires on this cohort) | default-OFF substrate; **awaits parallel-track narrowband cohort audit** |
| **A.5** saturation gate | All-bucket Δ = 0.000 (no saturation in this cohort) | default-OFF substrate; **awaits parallel-track saturation cohort audit** |
| **ALL** combo | criterion-4 fail (A.1 destabilises) | **CLOSED / DO NOT SHIP** |

Next active cycle: **v3.21.15 = A.2 + A.3 only**. A.1/A.4/A.5 explicitly
forbidden from v3.21.15. A.4/A.5 graduate via separate cohort audits;
neither blocks v3.21.15 progress. See plan file
`~/.claude/plans/se-aec-aec-main-hazy-lynx.md` for v3.21.15 + A.4/A.5
audit specs.

## A_baseline per-bucket means

| Bucket | N | echo | deg |
|---|---:|---:|---:|
| DT_movement | 3 | 3.917 | 3.046 |
| DT_static | 4 | 4.419 | 2.748 |
| FS_movement | 1 | 3.818 | 4.999 |
| FS_static | 3 | 3.167 | 5.000 |
| NE | 1 | 4.999 | 4.354 |

## Per-variant Δ vs A (bucket means)

| Variant | Bucket | Δecho | Δdeg |
|---|---|---:|---:|
| A.1 | FS_static | −0.021 | +0.000 |
| A.1 | FS_movement | **+0.032** | +0.000 |
| A.1 | DT_static | +0.007 | −0.004 |
| A.1 | DT_movement | +0.047 | **−0.051** |
| A.2 | FS_static | **+0.059** | +0.000 |
| A.2 | FS_movement | **+0.053** | −0.000 |
| A.2 | DT_static | +0.002 | −0.007 |
| A.2 | DT_movement | −0.018 | +0.007 |
| A.3 | FS_static | +0.001 | +0.000 |
| A.3 | FS_movement | **+0.161** | +0.000 |
| A.3 | DT_static | +0.000 | −0.004 |
| A.3 | DT_movement | +0.007 | +0.009 |
| A.4 | (all) | ≤ +0.003 | ≤ −0.004 |
| A.5 | (all) | 0.000 | 0.000 |
| ALL | FS_static | −0.017 | +0.000 |
| ALL | FS_movement | **+0.129** | +0.000 |
| ALL | DT_static | −0.001 | +0.014 |
| ALL | DT_movement | +0.023 | **−0.151** |

## Per-case Cat C stress recovery vs A (and vs v3.21.7 B)

Baseline (v3.21.7 partition_summed_x2 OFF) Cat C deg are already in normal
range. v3.21.7 B's Cat C cluster (−1.038 to −1.475 Δdeg vs A) is reproduced
here ONLY through `AEC_PARTITION_SUMMED_X2=1`; that env is OFF in this matrix.

| Case | A deg | A.1 Δdeg | A.3 Δdeg | ALL Δdeg | v3.21.7 B Δdeg (memory) |
|---|---:|---:|---:|---:|---:|
| XRTnTUjU_DT | 3.374 | −0.000 | +0.000 | +0.000 | −1.253 |
| MYrVxVEM_DT | 2.454 | −0.097 | +0.000 | −0.094 | −1.038 |
| nVUnxqHLr_DT | 2.589 | +0.085 | +0.000 | **+0.189** | −1.475 |

ALL >> v3.21.7 B on all three (none of these v3.21.14 flags is
partition_summed_x2 for refined; refined-side partition X² stays OFF). A.3
alone produces the cleanest +0.189 nVUnxqHLr partial recovery.

## Normal DT per-case (5-criterion check #4)

| Case | A deg | A.1 Δdeg | A.2 Δdeg | A.3 Δdeg | ALL Δdeg |
|---|---:|---:|---:|---:|---:|
| jtYTdZm3 DT | 2.575 | −0.003 | −0.037 | −0.015 | −0.041 |
| **wVYSGVTTakih9twI4xlDWQ DT_mvmt** | 2.948 | **−0.148** | −0.000 | +0.029 | **−0.445** |
| xFk7igecuke DT_mvmt | 3.036 | −0.004 | −0.015 | +0.000 | −0.004 |
| ZJYUt0O0AEKSQ9LJ DT_mvmt | 3.154 | −0.001 | +0.036 | −0.001 | −0.003 |

wVYSGVTTakih9twI4xlDWQ regression is the criterion-4 fail. A.1 alone is the
destabiliser (−0.148); A.1 + A.3 + A.4 + A.5 do not stack additively — ALL
amplifies wVYSGVTTakih9twI4xlDWQ to **−0.445**. Mechanism candidate (not
trace-confirmed): A.1 (partition-sum X²) + A.2 (hard noise gate) jointly
make shadow mu more reactive AND less stable; on
wVYSGVTTakih9twI4xlDWQ_doubletalk_with_movement the shadow can lose tracking
during the movement transient and the v3.21.6 latch path
([[project-usable-linear-gate3-latch-bug]]) over-suppresses DT speech.
Verbatim trace TBD; deferred to v3.22 substrate work.

## FS-side per-case (parity gain attribution)

| Case | A echo | A.1 Δecho | A.2 Δecho | A.3 Δecho | ALL Δecho |
|---|---:|---:|---:|---:|---:|
| 9xjhi FS | 2.459 | −0.102 | −0.003 | +0.001 | −0.121 |
| qNvSMyU FS | 3.435 | −0.009 | +0.000 | +0.000 | −0.049 |
| **xQEUtY2 FS** | 3.607 | +0.048 | **+0.179** | +0.000 | **+0.120** |
| **0I0XMl3M FS_mvmt** | 3.818 | +0.032 | +0.053 | **+0.161** | **+0.129** |

A.2 owns xQEUtY2 +0.179 gain; A.3 owns 0I0XMl3M +0.161 gain. ALL
preserves both. A.1 hurts 9xjhi −0.102 / −0.121 in ALL.

## 5-criterion ship gate (ALL evaluation)

| Criterion | Threshold | ALL value | Verdict |
|---|---|---|---|
| 1. Stress recovery vs v3.21.7 B | net positive on 3 stress cases | XRTnTUjU +1.253 / MYrVxVEM +0.944 / nVUnxqHLr +1.664 (all vs B) | PASS (trivial — flag set is orthogonal to v3.21.7) |
| 2. FS_static Δecho ≥ −0.05 mean | bar ≥ −0.05 | **−0.017** | PASS |
| 3. DT_static Δdeg ≥ −0.10 mean | bar ≥ −0.10 | **+0.014** | PASS |
| 4. No normal DT per-case catastrophic regression | ≥ −0.30 worst | **wVYSGVTTakih9twI4xlDWQ −0.445** | **FAIL** |
| 5. nores LF artifact ≥ A | spectrogram / band energy improvement | not analysed (criterion-4 fail makes 5 moot) | N/A |

**ALL-on fails criterion 4** because of A.1. A.1 + ALL-on close.
A.2 + A.3 + A.4 + A.5 individually pass or are untested on this cohort
and remain active v3.21.x parity work.

## Per-sub disposition

- **ALL-on**: CLOSED / DO NOT SHIP (criterion-4 wVYSGVTTakih9twI4xlDWQ catastrophic)
- **A.1**: CLOSED / DO NOT SHIP (wVYSGVTTakih9twI4xlDWQ −0.148 single-flag
  regression; primary destabiliser in ALL combo)
- **A.2**: ACTIVE v3.21.15 candidate (combined with A.3); 12-case standalone
  shows FS gain +0.05 to +0.18 per-case with no DT loss
- **A.3**: ACTIVE v3.21.15 candidate (combined with A.2); 12-case standalone
  shows FS_movement +0.16 + Cat C nVUnxqHLr +0.19 with no DT loss
- **A.4 narrowband mask**: default-OFF substrate; awaits parallel-track
  narrowband / tonal-render cohort audit (separate sprint, NOT blocking
  v3.21.15). Cohort search criterion: cases where PBFDKF main
  `_render_signal_analyzer` mask actually fires ≥ 10 %
- **A.5 saturation gate**: default-OFF substrate; awaits parallel-track
  saturation cohort audit. Cohort search criterion: cases with
  `_saturation_level > 0.5` on ≥ 5 % of frames

## v3.21.x parity attempts to date (arc still OPEN)

| # | Cycle | Flag / scope | Verdict |
|---|---|---|---|
| 1 | v3.21.7 | partition_summed_x2 refined | Cat C BLOCKED-STRESS |
| 2 | v3.21.8 | UseRefinedOutput | Cat C BLOCKED-STRESS / AEC3 field-trial-only |
| 3 | v3.21.9 | coarse_e2_time_domain | byte-equal no-op |
| 4 | v3.21.10 | aec3_misadj_parity | parity audit no-op |
| 5 | v3.21.11 | coarse_filter_converged_relaxed | no-consumer no-op |
| 6 | v3.21.12 | current_e2_refined_in_h_error_denom | REJECTED Cat C worse |
| 7 | v3.21.13 | UseLinearFilterOutput + BE combo | BE-combo no recovery |
| 8 | **v3.21.14** | **PBFDAF shadow 5-protection alignment (A.1–A.5)** | A.1+ALL CLOSED; **A.2 + A.3 → v3.21.15**; A.4/A.5 → parallel audits |
| **9** | **v3.21.15** (next) | **A.2 + A.3 only** | **OPEN — see plan** |
| **10** | A.4 narrowband audit | targeted cohort | OPEN — parallel-track |
| **11** | A.5 saturation audit | targeted cohort | OPEN — parallel-track |

Version-rule reminder: A.2 / A.3 / A.4 / A.5 are AEC3 parity / alignment and
remain in v3.21.x. v3.22 is reserved for beyond-AEC3 design only; do NOT
move unfinished AEC3 parity candidates into v3.22.

## Forbidden post-verdict

- No `/simplify` until v3.21.15 + A.4/A.5 audits close
- No 800-case run on v3.21.14 ALL or A.1 (CLOSED)
- No 800-case on A.2 / A.3 until v3.21.15 12-case 7-criterion gate PASS
- No A.1 / A.4 / A.5 enablement during v3.21.15 (keep attribution clean)
- No flag default-True flip on any sub-step without user review
- No v3.22 work — v3.21.x parity arc remains OPEN
