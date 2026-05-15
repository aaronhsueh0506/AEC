# v3.18 Phase F Closeout — EchoPathVariability event classification (2026-05-16)

**Verdict**: CLOSED — CANNOT SHIP. Substrate retained default-OFF.
Pivot motivates **v3.20 Delay-revisit** backlog priority.

## Scope tested

AEC3-aligned 3-tuple event classification + asymmetric cascade reset:
- F.1: `AecEvent` taxonomy + `classify_epc_event` classifier
- F.2: `_handle_gain_change_soft()` (Q-boost only, mirrors AEC3
  `refined_gains_->HandleEchoPathChange()`) + `_handle_delay_change_full()`
  (Q-boost + Kalman P relax + ERL cap + state reset)
- F.3: asymmetric wiring at EPV + shadow_rise sites — when flag ON, both
  use soft reset (gain_change semantics); delay_shift unchanged (already
  full path via `_reset_filter_derived_state`)

All three sprints: byte-equal flag-OFF PASS.

## 60-case A/B (flag-ON vs clean f_baseline, BALANCED / fl=832 / cng)

| Bucket | Δecho | Δdeg | Hard bar | Verdict |
|---|---:|---:|---|---|
| FS_static | **-0.048** | ±0.000 | Δecho ≥ -0.020 | **FAIL** |
| FS_movement | -0.011 | ±0.000 | Δecho ≥ -0.020 | OK |
| DT_static | -0.013 | +0.007 | Δdeg ≥ +0.005 | borderline |
| DT_movement | +0.013 | -0.011 | Δdeg ≥ +0.005 | **FAIL** |
| NE | ±0.000 | ±0.000 | — | OK |

Worst-case protect bar: `sx6mxKBQ` (FS_mv) Δecho **-0.081** > -0.05
breach → individual case failure on cohort-tail-like sample.

## Mechanism failure analysis

The asymmetric reset depended on a clean separation between gain_change
events and delay_change events. In AEC3 this separation is robust
because:

1. `MatchedFilter` (parallel NLMS at different alignment shifts) +
   `MatchedFilterLagAggregator` (histogram-mode latch with kCoarse /
   kRefined tiers) gives a high-confidence delay change signal
   (`EchoPathVariability.delay_change == kNewDetectedDelay`).
2. `EchoPathVariability.gain_change` is fired by a *separate* detector
   tracking far-end level changes; events don't co-fire.

Our system's reality:
- **DelayEstimator** is GCC-PHAT based (v3.16 C6 audit; closed
  CANNOT-FIX for sample-level multi-peak cases like `0I0XMl3M` /
  `qNvSMyU`). It fires `force_delay()` only on extreme alignment shifts
  (|Δdelay| > 32 samples), missing 10-20 sample shifts that AEC3
  catches.
- **EPV detector** (far-power EMA divergence ±6 dB) fires on BOTH gain
  glitches AND real path changes (a real path change shifts which
  far-end energy ends up in the mic, so far-power tracker sees the
  divergence too).
- **shadow_rise detector** (both filter errors rising) fires after a
  real path change has invalidated the filter — also catches some gain
  glitches.

So `source ∈ {'epv', 'shadow_rise'}` doesn't actually mean "gain change
without delay change" — it means "something happened that the
DelayEstimator didn't catch." Treating it as gain_change-only
(soft reset) strips the Kalman P relax + ERL cap that today carries
post-path-change FS suppression. Hence the -0.048 FS_static regression.

This is exactly the failure mode v3.16 C6 audit predicted: without an
AEC3-grade delay detector, you can't refactor the reset cascade per
AEC3 semantics. The C6 closure was correct that **GCC-PHAT** can't fix
the delay tracking gap; what changed is recognizing that **structurally
different detector family** (NLMS-matched-filter + histogram) is what
makes AEC3's `delay_change` reliable enough to gate.

## Substrate retained (default OFF)

- `AecEventType` constants + `AecEvent` dataclass
- `classify_epc_event(EpcEvent) → AecEvent` helper
- `_handle_gain_change_soft()` + `_handle_delay_change_full()` methods
- Asymmetric wiring at EPV + shadow_rise sites (legacy branch
  preserved; new branch behind `aec_event_classification_enabled`)
- `AEC._classified_event` per-frame state slot
- `AEC_EVENT_CLASSIFICATION` env-var override in eval bench

Reusable when v3.20 Delay-revisit lands an AEC3-grade matched-filter +
histogram delay detector. At that point, EPV/shadow_rise will no longer
double-cover real path changes, and the asymmetric reset becomes safe
to enable.

## Implications for v3.18 cycle

- F closes negative on the 60-case bench.
- Phase A (shadow NLMS) and Phase B (filter misadjustment ScaleFilter)
  proceed per plan revision §5 ordering — both are gain_change /
  state-quality work and don't depend on delay classification.
- Phase C (expanded; FQA + AecState centralisation) absorbs the F.1 +
  F.2 substrate when it lands the centralised state model.
- v3.20 (post-v3.18) priority elevates: matched-filter + histogram
  delay estimator becomes the gating arc for any future asymmetric
  reset work.

## Cross-references

- `docs/v3_18_plan_revision_2026_05_15.md` §3 — Phase F sprint plan
- `docs/v3_18_f1_aec_event_taxonomy.md` — F.1 substrate doc
- `docs/v3_18_d_gamma_closeout.md` — D-γ closeout (prior Phase)
- `docs/aec3_extracts/src/aec3/echo_path_variability.{h,cc}` — AEC3 ref
- `docs/aec3_extracts/src/aec3/subtractor.cc:148-175` — AEC3
  HandleEchoPathChange semantics
- `~/.claude/memory/project_delay_est_audit_v3_16.md` — v3.16 C6 audit
- `docs/v3_18_plan_revision_2026_05_15.md` §2 Delay-revisit row —
  v3.20 backlog motivation
