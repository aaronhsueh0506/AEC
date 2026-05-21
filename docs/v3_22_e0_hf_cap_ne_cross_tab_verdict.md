# v3.22 Sprint E.0 — HF cap × NE state cross-tab verdict (IN PROGRESS)

**Date**: 2026-05-21
**Branch**: `feature/v3_22_optimization`
**Status**: E.0 cross-tab observed; frame-level correlation analysis pending; **E.1 design decision GATED** on separability outcome
**Plan**: `~/.claude/plans/se-aec-aec-main-hazy-lynx.md` (v3.22 Sprint E — PRIMARY, post-v3.21.6 framing)
**Prior**: [v3.21.6 P4 verdict](v3_21_6_p4_stationarity_retest_verdict.md) (surviving root cause = `_DominantNearendDetector` mis-classification under stationary-far)

## E.0 cohort trace setup

Rendered 8 cases on v3.21.6 default state (no env overrides; `--trace-hf-chain`):

| Case | Bucket | n frames |
|---|---|---:|
| pcb1Nh0Z3k0WS9a7gBEuqg_farend_singletalk | FS_static | 2318 |
| LN18k5r8t00C9DulUd809A_farend_singletalk | FS_static | 2358 |
| s90M7MOTBkqaV4nQPLhKbA_farend_singletalk | FS_static | 2307 |
| 9xjhiFbGo06hdQIsHTS6qA_farend_singletalk | FS_static | 2188 |
| lV0kQN0hR0ySmE0bQhuYbw_farend_singletalk | FS_static | 2167 |
| WcK0OrF6ukW03fViPXTQjQ_doubletalk_with_movement | DT_movement | 3916 |
| wVYSGVTTakih9twI4xlDWQ_doubletalk_with_movement | DT_movement | 3666 |
| xQEUtY2pWUi7v1X93TF2AA_doubletalk | DT_static | 3988 |

5 FS_static = HF-heavy cohort (Sprint A worst-FS-echo); 3 DT cohort tail = P4 worst-deg cases.

Trace data location: `/tmp/e0_cohort/*.csv`.

## E.0 cross-tab (per-case rates)

| Case | Bucket | HF_cap% | NE% | HFcap\|NE% | HFcap\|~NE% | stat% | stat & ~NE% |
|---|---|---:|---:|---:|---:|---:|---:|
| pcb1Nh | FS_static | 97.2 | 2.8 | 0.0 | 100.0 | 35.8 | 34.3 |
| LN18k5r8 | FS_static | 100.0 | 0.0 | 0.0 | 100.0 | 2.9 | 2.9 |
| s90M7MOT | FS_static | 87.8 | 12.2 | 0.0 | 100.0 | 51.8 | 44.7 |
| 9xjhi | FS_static | 100.0 | 0.0 | 0.0 | 100.0 | 46.0 | 46.0 |
| lV0kQN | FS_static | 96.7 | 3.3 | 0.0 | 100.0 | 90.5 | **87.3** |
| WcK0OrF | DT_movement | 84.7 | 15.3 | 0.0 | 100.0 | 51.1 | 44.8 |
| wVYSGV | DT_movement | 99.6 | 0.4 | 0.0 | 100.0 | 64.0 | **63.8** |
| xQEUtY2 | DT_static | 93.0 | 7.0 | 0.0 | 100.0 | 63.2 | **58.8** |

(`stat%` = `stationary_mask_fired_frac > 0.1`, i.e. mask covered > 10% of bins.)

## E.0 observations

### Observation 1: HF cap × NE state is perfectly anti-correlated

`HFcap | NE = 0.0%` and `HFcap | ~NE = 100.0%` across **all 8 cases**. This is the existing gate `not self._dominant_nearend.is_nearend_state()` working exactly as designed: the cap fires iff NE-state is False.

### Observation 2: Plan's original E.0 gate criterion DOES NOT trigger E.1

The plan's pre-trace E.0 gate stated: "if HF cap fire rate < 30% → E.1 high priority" (the assumption being that HF cap was under-firing → echo bypass). Reality: HF cap fires **84-100%** of frames across all 8 cases. **Original E.1 framing (decouple HF cap → fire more) is wrong for our cohort.**

### Observation 3: The actionable failure mode is the inverse — HF cap fires when it shouldn't

The DT cohort tail (WcK0OrF / wVYSGV / xQEUtY2) and stationary-far-heavy FS cases (lV0kQN, 9xjhi, s90M7MOT) show very high `stat & ~NE` rates (44-87%). These are frames where:

- `stationary_mask_fired_frac > 10%` → at least some bins are flagged as stationary-far (legitimate stationary echo conditions)
- `is_nearend_state = False` → NE detector says "no NE speech present"

Per v3.21.6 P4 verdict, `_DominantNearendDetector` mis-classifies NE under stationary-far conditions (xQEUtY2 stuck at 7.0% NE rate vs Sprint B's documented 7.2% baseline despite P1+P2+P3 ship). So a large fraction of `stat & ~NE` frames are **suspected NE-speech frames where the detector failed**. In those frames, HF cap fires aggressively (100% gate firing) — adding HF-band suppression on top of the far-tuning gain that P4 documented as catastrophic NE destruction.

### Observation 4: Original E.1 reframed

E.1 cannot be "decouple HF cap from NE state → cap fires more". The correct direction is **augment NE state with stationary-far-aware proxy → cap fires less in detector-broken regions**. Three candidate paths:

- **(a) ENR-gate override**: gate cap as `not (is_nearend_state() OR (stat_active AND ENR < threshold))` — let the cap fire only when echo is dominant per ENR
- **(b) PBFDKF-specific NE proxy**: replace binary `is_nearend_state()` with a Kalman-state-derived echo-confidence scalar (intentional divergence, must NOT claim AEC3 parity per [P2 discipline rule](v3_21_6_p2_transparent_mode_audit_verdict.md))
- **(c) Frequency-band restriction**: when `stat_active`, cap only true HF (e.g. > 4 kHz) and preserve mid-band formants

## Decision gate (deferred to frame-level analysis)

Per user directive 2026-05-21:

> **Only when catastrophic frames (Δg100 < -0.3) can be separated from normal-echo frames on ENR/SNR distribution → proceed with E.1 candidate (a) (ENR-gate override).**
>
> **If the distributions overlap → E.1 must take path (b) PBFDKF-specific NE proxy OR (c) frequency-band restriction. Must NOT do plain decouple/fire-more.**

## Pending analysis (frame-level)

To make the gate decision, the follow-up analysis must:

1. **Identify catastrophic frames** per case — render the P4 cohort 3 cases with stationarity OFF vs default (= within-cohort Sprint B's Δg100 < -0.3 definition); collect frame indices
2. **Cross-tab at catastrophic frame indices** — `hf_cap_fired` × `stationary_mask_active` × `~is_nearend_state` × `catastrophic` (4-way) to confirm the suspected mechanism
3. **ENR/SNR distribution overlap test** — at catastrophic frames vs normal-echo frames (within same case), compare ENR + SNR histograms / percentile bands; quantify separability (e.g., AUC or simple percentile non-overlap)
4. **Frequency-band damage profile** — at catastrophic frames, look at gain_5 / gain_30 / gain_50 / gain_100 / gain_200 distribution; identify which bands are most damaged (HF cap-attributable vs broadband detector failure)

Outcomes:

- Separable + HF-band concentrated → E.1 (a) ENR-gate override is the right design
- Separable but broadband damage → E.1 (c) frequency-band restriction better
- Not separable (catastrophic vs normal echo look the same in ENR/SNR) → E.1 (b) PBFDKF-specific NE proxy required (Kalman-state echo confidence)

## Status

E.0 cross-tab observed. Frame-level analysis pending. E.1 design choice deferred until that analysis completes. No code changes yet on `feature/v3_22_optimization`.
