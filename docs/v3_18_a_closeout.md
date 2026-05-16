# v3.18 Phase A Closeout — Shadow NLMS conversion (2026-05-16)

**Verdict**: CLOSED — CANNOT SHIP. Substrate retained default-OFF.
A.7 60-case bench fails design lock §6 PRIMARY + worst-case hard bars.

## Scope tested

A.2 + A.3 + A.4 + A.5 + A.6 + A.7 all delivered:
- A.2 `shadow_class_nlms: bool = False` + `shadow_mu_nlms: float = 0.5`
  config; PBFDAF construction branch; isinstance guards on 4 PBFDKF-
  attr sites (`_handle_delay_change_full`, delay_shift site, EPV
  legacy, shadow_rise legacy, shadow_r_reset)
- A.3 (corrected per 2026-05-16 user feedback "AEC3 應該沒有 copy 機制"):
  flag-ON skips `reverse_copy` entirely (mirrors AEC3 — refined/coarse
  are independent, no W transfer)
- A.4 (descope): boost_q hangover (`_pause_resume` countdown) intact,
  filter-type agnostic — no code edit needed
- A.5 NLMS rate {0.3, 0.5, 0.7} grid: no measurable downstream
  sensitivity; μ=0.5 (AEC3 default) selected
- A.6 PathChangeRegimeHandler trace: under flag-ON, `reverse_copy`
  decisions surge (9→115 on 0I0XMl3M) because NLMS steady-state error
  > Kalman; `boost_q` paused at 1 fire — qNvSMyU defence path largely
  silent. Threshold retune deferred per design

All sprints: byte-equal flag-OFF (5/5 md5 PASS verified after A.2 and
after A.3 wiring).

## A.7 60-case A/B (BALANCED, fl=832, cng, flag-ON vs clean baseline)

### Bucket means

| Bucket | n | Δecho | Δdeg | Bucket guard | Verdict |
|---|---:|---:|---:|---|---|
| FS_static | 12 | -0.006 | -0.000 | Δecho ≥ -0.010 | OK |
| FS_movement | 9 | -0.009 | -0.000 | Δecho ≥ -0.010 | OK (marginal) |
| DT_static | 13 | -0.008 | +0.014 | Δdeg ≥ -0.005 | OK |
| DT_movement | 7 | +0.003 | +0.015 | Δdeg ≥ -0.005 | OK |
| NE | 20 | +0.000 | +0.000 | Δdeg ≥ -0.005 | OK |

Bucket guards all pass. DT NE-preserve actually improves (+0.014/+0.015 Δdeg).

### §6 PRIMARY — qNvSMyU Δecho ≥ +0.030 dB

| Field | Baseline | Flag-ON | Δ |
|---|---:|---:|---:|
| echo | 3.9558 | 3.9547 | -0.0011 |
| deg | 4.9990 | 4.9990 | +0.0000 |

**PRIMARY: FAIL** — Δecho ≈ 0, no cohort-tail improvement.

### §6 worst-case bar — no sample Δecho < -0.05 dB

| Stem | Bucket | Δecho | Δdeg | Notes |
|---|---|---:|---:|---|
| `kz23X4pDSEiPmWtw2Qx00Q_doubletalk` | DT_static | **-0.141** | +0.052 | breach; partial Δdeg offset |
| `N2rQLbnp2UOg2QFRaggbDw_farend_singletalk` | FS_static | **-0.131** | +0.000 | breach; pure FS loss |
| `hF9Lfjcn9kGQ4430uAbINA_farend_singletalk_with_movement` | FS_movement | **-0.098** | -0.000 | breach; pure FS loss |
| `oQK3bVihI0qel9As840Zzw_doubletalk_with_movement` | DT_movement | **-0.062** | +0.186 | breach; large Δdeg offset |

Four breaches; two are pure FS regression with no Δdeg compensation.

### Top 5 improvements

| Stem | Bucket | Δecho | Δdeg |
|---|---|---:|---:|
| WTdBhXa080W…_FS_static | FS_static | +0.092 | -0.000 |
| hF9Lfjcn9kGQ…_doubletalk | DT_static | +0.050 | +0.077 |
| xuKL15aeq0C…_doubletalk_with_movement | DT_movement | +0.040 | +0.011 |
| SUYzW4QT30y…_farend_singletalk | FS_static | +0.026 | -0.000 |
| uS9t2QYDckeO…_doubletalk | DT_static | +0.025 | +0.024 |

### Stats (n=61)

| Metric | Mean | Median | Min | Max |
|---|---:|---:|---:|---:|
| Δecho | -0.004 | +0.000 | -0.141 | +0.092 |
| Δdeg  | +0.005 | +0.000 | -0.063 | +0.186 |

## Mechanism analysis

Two competing effects observed:

**Positive — DT NE preservation**: bucket-mean Δdeg DT_static +0.014,
DT_movement +0.015. NLMS shadow doesn't accumulate Kalman P-memory,
so when the regime handler's `reverse_copy` action is skipped (A.3
guard), the shadow track is no longer "polluted" by a stale baseline.
This indirectly helps DTD signal cleanliness. Confirmed on
`oQK3bV…_DT_mv` (+0.186 Δdeg, large NE recovery).

**Negative — FS catastrophe**: cases like N2rQLbnp_FS_static (-0.131)
and hF9Lfjcn_FS_mv (-0.098) show large echo regressions with zero
Δdeg upside. Investigation shows these are FS-only samples where the
PBFDKF main filter benefits from the Kalman-shadow `reverse_copy`
resync (PBFDKF shadow occasionally "rescued" a drifted shadow track
that fed back into the regime handler). Removing that rescue path
under flag-ON exposes long-lived shadow tracking errors.

The kz23X4pD_DT_static -0.141 + Δdeg +0.052 case shows a different
pattern: the DT NE preserve gains and FS echo loss are correlated —
classic DT-FS Pareto trade-off (same wall as v3.13 E5).

## Risk validation (against A.1 §11 reverse-evidence)

| Risk # | Prediction | A.7 evidence |
|---|---|---|
| R1 | Arc S-orth.A precedent — coupling may be upstream of filter type | **Supported**. Removing filter coupling (NLMS shadow) gives DT bucket-mean gain but doesn't fix qNvSMyU. Signal-interpretation layer (regime handler thresholds, DTD weighting) is where the next move should be |
| R2 | qNvSMyU defence may worsen under NLMS shadow | **Borderline**. Δecho -0.0011 (no improvement, no regression). The defence path (boost_q + pause_main) rarely fires under NLMS shadow because shadow_err > main_err most of the time. P52 −0.56 dB floor maintained (qNvSMyU at 3.95 dB, well above pass-through) |
| R3 | F-closeout points to DelayEst as real bottleneck | **Supported**. NLMS shadow alone doesn't close cohort tail; matched-filter delay detection (v3.19+ backlog) remains the gating arc |
| R4 | D+F pattern — AEC3-port hits pipeline co-tuning wall | **Strongly supported**. Like D-γ and F, Phase A bucket means look OK but worst-case samples breach. Three AEC3-port arcs in a row hit the same wall |
| R5 | NLMS post-copy cycle | **Moot**. With reverse_copy skipped under flag-ON, there's no copy event to require hangover |
| R6 | μ_nlms wrongly tuned | **Not load-bearing**. A.5 grid showed μ choice has negligible downstream effect (RES dominates) |

## §6 kill criterion adjudication

> A.7 PRIMARY OR HARD GUARD failure → close A per §0.4. Substrate retained.

- PRIMARY (qNvSMyU Δecho ≥ +0.030): **FAIL** (-0.0011)
- Hard guard (worst-case Δecho ≥ -0.05): **FAIL** (4 breaches)
- Bucket guards: PASS

Two of three gate categories fail. Per §0.4 negative-result protocol,
**close Phase A**. Substrate kept as default-OFF for future re-enable
if signal-interpretation arc (Phase C FQA + regime handler retune) or
v3.19+ matched-filter delay arc lands a precondition that flips the
PRIMARY math.

## Substrate retained (default OFF)

- `AecConfig.shadow_class_nlms: bool = False`
- `AecConfig.shadow_mu_nlms: float = 0.5`
- PBFDAF construction branch at [aec.py:6219-6234](../python/aec.py#L6219)
- `isinstance(filt, PBFDKF)` guards on 4 P-override sites
  (legacy PBFDKF byte-equal preserved)
- A.3 `reverse_copy` skip guard at [aec.py:7484](../python/aec.py#L7484)
- `AEC_SHADOW_NLMS` / `AEC_SHADOW_MU_NLMS` env-var overrides in eval bench
- `_classified_event` slot (carries over from F.1 substrate)

Re-enable preconditions (any of these may lift the §6 PRIMARY bar):
1. Phase C FilteringQualityAnalyzer + regime handler retune with
   NLMS-aware thresholds (boost_q decision logic re-derived from NLMS
   error magnitude scale)
2. v3.19+ matched-filter + histogram delay detector (so qNvSMyU's
   path-change events get caught upstream, removing the dependency on
   regime handler defence)
3. Phase B FilterMisadjustmentEstimator + ScaleFilter (orthogonal — may
   tighten main filter on FS, recovering N2rQLbnp / hF9Lfjcn worst cases)

## Implications for v3.18 cycle

- Phase A closes negative on the 60-case bench. **3-of-3 AEC3-port
  arcs in v3.18 have closed CANNOT SHIP** (D-γ + F + A).
- Phase B (FilterMisadjustmentEstimator + ScaleFilter) **next** —
  substrate-ready, lower-risk per §11 pivot decision. Touches
  PBFDKF main filter only, file-disjoint with A's substrate.
- Phase C (expanded — FQA + AecState centralisation) gated on B PASS.
- Phase E (preset promotion) unchanged.
- v3.19+ backlog priority elevates further: matched-filter delay
  detector becomes the gating arc for ANY future shadow-side refactor.

## Cross-references

- [docs/v3_18_plan_revision_2026_05_15.md §11](v3_18_plan_revision_2026_05_15.md#L244) — Phase A design-confidence framing + §6 hard bar
- [docs/v3_18_a1_shadow_nlms_design.md](v3_18_a1_shadow_nlms_design.md) — A.1 design lock (with 2026-05-16 corrections)
- [docs/v3_18_f_closeout.md](v3_18_f_closeout.md) — predecessor closeout; cumulative AEC3-port wall evidence
- [docs/v3_18_d_gamma_closeout.md](v3_18_d_gamma_closeout.md) — first AEC3-port wall instance
- [docs/aec3_reference.md §6.1](aec3_reference.md#L750) — AEC3 shadow design (no W copy confirmed)
- [docs/p52_phase_a_verdict.md](p52_phase_a_verdict.md) / [docs/p52_a0_postmortem.md](p52_a0_postmortem.md) — PathChangeRegimeHandler cohort-tail defence rationale
- `~/.claude/memory/project_v3_15_closeout_outcome.md` — v3.15 four shadow-correlation closures
- `~/.claude/memory/project_delay_est_audit_v3_16.md` — v3.16 C6 audit (GCC-PHAT gap)
