# v3.21.6 Sprint P4 — Stationarity default-off re-test VERDICT

**Date**: 2026-05-21
**Branch**: `feature/v3_21_6_parity_completion`
**Status**: ✅ CLOSED — intentionally incompatible with current PBFDKF cohort tail; AEC3 default-off (`use_stationarity_properties=False`) permanently retired
**Plan**: `~/.claude/plans/se-aec-aec-main-hazy-lynx.md` (v3.21.6 Sprint P4)
**Prior verdict**: [`docs/v3_21_5_phase1_b_stationarity_gate_verdict.md`](v3_21_5_phase1_b_stationarity_gate_verdict.md) (Sprint B, v3.21.5)

## P4.0 — Cohort 3-case re-trace gate

Hypothesis (per Round 7 plan):
> After P1 FilterAnalyzer + P2 TransparentMode audit + P3 EchoAudibilityConfig wiring, the upstream signals (notably `is_nearend_state` firing under stationary-far conditions) may have been rescued — flip `use_stationarity_properties=False` (= AEC3 default) and check cohort tail damage.

User-set gate (3 criteria, ALL must pass):
1. Catastrophic gain drops (Δgain_100 < -0.3) disappear
2. `is_nearend_state` rate notably improves vs Sprint B baseline (7.2%)
3. Formant damage (1-4 kHz Δ dB) is eliminated

Method: render 3 Sprint B worst cases twice on current post-P3 baseline; compare per-frame trace + WAV-level formant bands.

| Case | Bucket | n frames |
|---|---|---:|
| WcK0OrF6ukW03fViPXTQjQ_doubletalk_with_movement | DT_movement | 3666 |
| wVYSGVTTakih9twI4xlDWQ_doubletalk_with_movement | DT_movement | 3666 |
| xQEUtY2pWUi7v1X93TF2AA_doubletalk | DT_static | 3988 |

Configurations:
- **ON**: `AEC_E2_Y2_CLAMP=1` (= default; `aec3_post_stationarity_zero_enabled=True`; load-bearing legacy zeroing fires)
- **OFF**: `AEC_E2_Y2_CLAMP=1 AEC_STATIONARITY_ZERO=0` (= AEC3-default-off; alias propagates through P3 wiring to `echo_audibility.use_stationarity_properties=False`)

### Results

| Case | NE_ON% | NE_OFF% | ΔNE | catastrophic g100<-.3 | tot energy Δ% | F1 Δ dB | F2 Δ dB | HF Δ dB |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| WcK0OrF | 15.3 | 15.3 | **+0.0** | **233** | -0.2 | -0.00 | -0.00 | -0.00 |
| wVYSGV | 0.4 | 0.4 | **+0.0** | **235** | -0.5 | -0.01 | -0.04 | -0.01 |
| xQEUtY2 | 7.0 | 7.0 | **+0.0** | **424** | -2.2 | -0.01 | -0.06 | **-0.94** |

### Per-criterion verdict

| Criterion | Result | Detail |
|---|---|---|
| (1) Catastrophic g100 drops disappear | ✗ FAIL | 233 / 235 / 424 frames per case (Sprint B baseline was ~280 on xQEUtY2; xQEUtY2's count *grew* here under the new substrate, but the magnitude of the formant damage per drop is somewhat lower — net still load-bearing) |
| (2) NE state rate improves | ✗ **HARD FAIL** | ΔNE = **+0.0** on all 3 cases. xQEUtY2 stays at 7.0% (vs Sprint B's 7.2% baseline). P1 FilterAnalyzer didn't move the detector under stationary-far conditions; P2 TM was closed disabled; P3 only re-wires config plumbing. **None of the upstream changes touched the `_DominantNearendDetector` decision path that Sprint B identified as the root cause** of the safety-net being load-bearing. |
| (3) Formant damage eliminated | ✗ FAIL | xQEUtY2 HF Δ -0.94 dB (Sprint B -1.39 dB; reduction roughly proportional to fewer-but-different catastrophic frames distribution, still audible-band damage). F1 / F2 micro (≤ 0.06 dB), HF dominant. |

Per user directive on P4.0 gate:
> *若 P4.0 fail，直接 close P4 intentionally-incompatible，v3.21.6 保留 zeroing default True*

All 3 criteria FAIL. → **CLOSE P4 intentionally-incompatible. No 800-case bench run.**

## Root-cause summary

The Sprint B safety-net evidence holds on the **post-P1+P2+P3 baseline**. The stationarity zeroing of `R²` / `R²_unbounded` is a **permanent intentional deviation** from AEC3 default-off behavior, compensating for our incomplete `_DominantNearendDetector` port on stationary-far conditions:
- Under stationary far-end noise (HVAC, room tone, etc.), our detector mis-classifies NE-speech frames as "not NE" (echo path active) → far-tuning gain applied → catastrophic NE destruction
- AEC3 has companion mechanisms (ScaleFilter / FilterMisadjustmentEstimator) that keep `is_nearend_state` correctly firing under stationary-far; we don't port those
- The orchestrator-side zeroing of stationary-band R²/R²_unbounded effectively prevents the destructive gain decisions on those bands
- Removing the zeroing exposes the detector mis-classification consequences directly → 1-2 dB formant attenuation on cohort tail

P1's FilterAnalyzer (direct-path delay + analyzer-derived `any_filter_consistent`) **does not feed into** the `_DominantNearendDetector` ENR/SNR decision path. P3 only re-architectures the config surface. So there was no mechanism for P1+P2+P3 to rescue the detector — the falsified hypothesis is now data-confirmed-falsified.

## Final verdict

✅ **CLOSED — AEC3 default-off (`use_stationarity_properties=False`) permanently retired for our PBFDKF + RES port.**

- Production state unchanged: `aec3_post_stationarity_zero_enabled = True` (= `echo_audibility.use_stationarity_properties = True` after P3 wiring; load-bearing legacy zeroing fires).
- The flag stays as a **research / A-B toggle** indefinitely (env hook `AEC_STATIONARITY_ZERO=0` still works) — not as a "soon-to-flip" experiment.
- Per the [P2 discipline rule](feedback_no_parity_claim_for_divergence.md): any future cycle revisiting this MUST design as **PBFDKF-specific divergence** (e.g., ship the missing AEC3 ScaleFilter / FilterMisadjustmentEstimator companions; or replace the detector with a PBFDKF-Kalman-aware NE detector). **MUST NOT** be framed as "AEC3 parity restoration of `use_stationarity_properties=False`".

### Plan triage policy citation

This verdict places `use_stationarity_properties=False` permanently into **Bucket 3** (closed-out — not in v3.22 scope as AEC3 parity). The companion-mechanism design (which would mechanically allow AEC3-default-off to work) is the divergence path — pre-emptively, v3.22 Sprint G has slot for "DominantNearendDetector parity / companion mechanisms" if a future research budget materializes; it must be labelled as divergence.

### v3.21.6 cycle accounting

| Sprint | Verdict | AEC3 parity outcome |
|---|---|---|
| P1 FilterAnalyzer | ✅ PASS, shipped default-True | Restored |
| P2 TransparentMode | ✅ CLOSED intentionally-incompatible | Permanently retired (PBFDKF peak too noisy) |
| P3 EchoAudibilityConfig | ✅ PASS, byte-equal | Structural parity restored |
| **P4 stationarity** | **✅ CLOSED intentionally-incompatible** | **AEC3 default-off permanently retired (detector port incomplete; safety net load-bearing)** |

v3.21.6 entry gate to v3.22 (= AEC3 parity baseline locked): MET. Every Bucket-1 parity item has a closed verdict; no parity gap remains merely deferred.

## v3.22 Sprint I follow-up

Per [P3 verdict's "Sprint I follow-up" section](v3_21_6_p3_echo_audibility_wiring_verdict.md): since `use_stationarity_properties` is now **permanently True** for our cohort, the top-level `AecConfig.aec3_post_stationarity_zero_enabled` alias can either:

1. **Stay as a research toggle** (keep alias forever; document the v3.22 cleanup as "intentional flag retention for A/B work")
2. **Be removed** in v3.22 Sprint I (canonical surface is `SuppressorConfig.echo_audibility.use_stationarity_properties`); env hook then needs new wiring path

Recommendation for I: option 1 (keep flag) since A/B research is useful for any future companion-mechanism work that may rescue AEC3-default-off behavior. Document explicitly that the flag's default is a **non-AEC3-parity intentional deviation**, not a "to-flip-later" stub.
