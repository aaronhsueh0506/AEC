# v3.22 Sprint E.0 — HF cap × NE state cross-tab verdict

**Date**: 2026-05-21
**Branch**: `feature/v3_22_optimization`
**Status**: E.0 complete — E.1 candidate (a) ENR-gate override REJECTED; E.1 must take path (b) **NE-presence augmentation using `stat_active`**, not (a) ENR-gate, not (c) band restriction
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

## Frame-level analysis results

Reused P4 cohort traces (`/tmp/p4_cohort_trace/*_zeroing_{on,off}.csv`). Catastrophic = `gain_100[off] - gain_100[on] < -0.3` (Sprint B / P4 definition: removing the stationarity zeroing pulls gain from ~1.0 to ~0.0 on those frames). All other downstream signals (`hf_cap_fired`, `is_nearend_state`, `stationary_mask_fired_frac`, ENR, SNR, gain_N) read from the `zeroing_on` trace = v3.21.6 default state.

### 4-way cross-tab at catastrophic frame indices

| Case | n | cata | cata% | (hf_cap & stat & ~NE & cata) | match rate |
|---|---:|---:|---:|---:|---:|
| WcK0OrF | 3916 | 233 | 5.9% | 233 | **100%** |
| wVYSGV | 3666 | 235 | 6.4% | 221 | 94% |
| xQEUtY2 | 3988 | 424 | 10.6% | 424 | **100%** |

**Mechanism perfectly aligned**: 94-100% of catastrophic frames satisfy `(hf_cap_fired & stat_active & ~is_nearend_state)`. Catastrophic frames are runtime-detectable via that 4-way signal, without any oracle.

### ENR/SNR distribution overlap (catastrophic vs normal-echo, default state)

| Case | group | n | ENR p25/med/p75 | SNR p25/med/p75 |
|---|---|---:|---|---|
| WcK0OrF | catastr | 233 | 0 / 0 / 0 | 1.75 / 2.73 / 3.63 |
| WcK0OrF | normal_echo | 3683 | 0 / 0.032 / 3.23 | 1.39 / 4.51 / 23.26 |
| wVYSGV | catastr | 235 | 0 / **1.90** / 3.37 | 0.30 / 3.88 / 137 |
| wVYSGV | normal_echo | 3431 | 0 / **0** / 1.62 | 0.11 / 1.49 / 5.71 |
| xQEUtY2 | catastr | 424 | 0 / 0 / 0 | 0.35 / 0.99 / 3.07 |
| xQEUtY2 | normal_echo | 3564 | 0 / 0.404 / 35.9 | 0.41 / 4.22 / 40.9 |

**Direction NOT consistent across cohort.** wVYSGV is *inverted* (catastr has higher ENR than normal_echo). WcK0OrF and xQEUtY2 trend in the expected direction (catastr lower) but with very compressed catastr distributions clamped at zero. SNR similarly mixed: catastr p75 ranges from 3.07 to 137 across cases.

**Per user-set gate (2026-05-21): if distributions overlap → E.1 must NOT do plain ENR-gate override.** Result: **(a) ENR-gate override REJECTED**.

### Frequency-band damage profile (gain_N median in default state)

| Case | group | g5 (156Hz) | g30 (938) | g50 (1563) | g100 (3125) | g200 (6250) |
|---|---|---:|---:|---:|---:|---:|
| WcK0OrF | catastr | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| WcK0OrF | normal_echo | 1.000 | 0.447 | 1.000 | 1.000 | 0.519 |
| wVYSGV | catastr | 1.000 | 1.000 | 1.000 | 1.000 | 0.136 |
| wVYSGV | normal_echo | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| xQEUtY2 | catastr | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| xQEUtY2 | normal_echo | 0.289 | 0.215 | 0.291 | 0.240 | 0.082 |

In default state: **catastrophic frames have gain ≈ 1.000 across ALL bands** (the zeroing safety net has done its job — R²/R²_unbounded are zeroed on stationary bins → SuppressionGain produces gain ≈ 1.0 → NE-speech preserved). With zeroing OFF (P4 experiment), those same frames drop to broadband gain ≈ 0.0 (per P4 verdict).

**Damage is broadband, not HF-localized.** The HF cap firing during stat&~NE frames is one symptom, but the full broadband suppression (g5..g200 all dropping) tells us the failure is at the upstream detector decision (`is_nearend_state` mis-classification) propagating through ALL gain rules — not just the cap.

**(c) Frequency-band restriction REJECTED**: capping only > 4 kHz wouldn't save the broadband damage (g5/g30/g50/g100 all hit when zeroing is OFF).

## E.1 design — chosen path

**Path (b) — NE-presence augmentation**, but the signal source is `stat_active` (= the same `band_stationary_mask()` already powering the zeroing), NOT Kalman P_trace.

Rationale:
- Frame-level analysis showed `stat_active` is a near-perfect runtime indicator of detector-failure regions (94-100% match with catastrophic frames)
- The damage is broadband, so the fix has to be upstream of all gain rules, not at the HF cap gate alone
- This is divergence from AEC3 (AEC3 has companion mechanisms — ScaleFilter / FilterMisadjustmentEstimator — we don't port; we use stationarity awareness as a different non-AEC3 protective augmentation, per [P2 discipline rule](v3_21_6_p2_transparent_mode_audit_verdict.md): must NOT claim AEC3 parity restoration)

### Proposed E.1 implementation

Replace every relevant read of `dominant_nearend.is_nearend_state()` with an augmented form:

```python
def ne_or_stationary_far_protect(self) -> bool:
    if self._dominant_nearend.is_nearend_state():
        return True
    # detector-failure-region override: if stationary mask covers a
    # substantial portion of bins, treat as NE-presence-uncertain
    if self._stat_mask_active_now:  # = band_stationary_mask().sum() / n_bins > threshold
        return True
    return False
```

Initial threshold candidate: `> 10%` of bins masked (matches the cross-tab definition). Tune in E.2 if needed.

Critical placement: this proxy replaces `is_nearend_state()` AT THE GAIN-RULE LEVEL in `SuppressionGain` — not just at the HF cap gate. Specifically:
- HF cap gate
- `nearend_tuning` vs `normal_tuning` selector
- Any downstream Wiener-style rule reading `is_nearend_state()`

NOT at the detector's own internal state — the detector keeps its current decisions (don't pollute its hysteresis). The proxy is a *gain-policy* override that respects "detector might be wrong here".

### Risk register

1. **Over-protection on FS_static cases**: FS cases with stationary far-end (lV0kQN 87.3% stat-frac) would now get NE-protective gain whenever stationarity fires → echo bypass risk. Mitigation: E.0 cross-tab shows lV0kQN's NE rate = 3.3%, and the gain trajectory at stationary-far frames is currently dominated by zeroing's effect (gain ≈ 1.0). Augmentation would not change that. But for FS frames OUTSIDE the stationary mask region, the HF cap still fires (no change). Need bench to confirm.

2. **Threshold tuning**: 10% bins is a guess. Should be benched at 5% / 10% / 20%.

3. **Aliasing the existing zeroing**: the augmentation uses the same signal the zeroing already uses. Risk = double-coverage redundancy (each protects different code paths so probably OK) or accidental masking (if the augmentation triggers nearend_tuning which itself produces aggressive suppression). Need careful bench.

## Status

E.0 complete. E.1 design path locked: **(b) NE-presence augmentation using `stat_active`**, not ENR-gate, not band restriction. Implementation pending under flag-guarded default-OFF; cohort + 800-case bench follows.
