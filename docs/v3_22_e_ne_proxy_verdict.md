# v3.22 Sprint E — NE-presence augmentation CLOSED no-leverage

**Date**: 2026-05-21
**Branch**: `feature/v3_22_optimization`
**Status**: **CLOSED no-leverage**. Code substrate stays in tree under default-OFF flag `e_stat_aware_ne_proxy_enabled`; production behaviour unchanged.
**Prior**:
- [v3.21.6 P4 verdict](v3_21_6_p4_stationarity_retest_verdict.md) — surviving root cause = `_DominantNearendDetector` mis-classification under stationary-far
- [v3.22 E.0 cross-tab + frame-level analysis](v3_22_e0_hf_cap_ne_cross_tab_verdict.md) — design path locked to (b) NE-presence augmentation using `stat_active`

**Headline**: Proxy at SuppressionGain gain-policy level fires on the right frames (matches 94-100% of catastrophic indices, exactly as the E.0 cross-tab predicted) but cannot rescue the catastrophic-gain failure mode. The damage originates upstream at the residual-echo PSD level (`R²` / `R²_unbounded` are zeroed on stationary bins inside `_aec3_post`); a gain-rule proxy at SuppressionGain only changes ENR/EMR tuning parameters, which is too far downstream to substitute for the zeroing.

## E.1 implementation summary

Per directive 2026-05-21:
- Flag-guarded default-OFF (`e_stat_aware_ne_proxy_enabled`; threshold `e_stat_aware_ne_proxy_threshold=0.10`)
- Proxy applied at **SuppressionGain gain-policy consumer sites only**:
  - HF cap gate
  - `_get_max_gain` selector
  - `_get_min_gain` LF smoothing dec selector
  - `_gain_to_no_audible_echo` ENR/EMR tuning selector
- `is_dominant_nearend()` public API UNCHANGED (returns raw `is_nearend_state()`) so `_DominantNearendDetector` hysteresis is untouched
- Stationarity mask passed through from orchestrator via `get_gain(stationary_mask=...)`
- Env hooks `AEC_STAT_NE_PROXY` / `AEC_STAT_NE_PROXY_THR` added to `run_one_case.py` + `eval_aec_challenge.py`

Code surface (≈40 LOC):
- `python/modules/config.py`: 2 new AecConfig fields
- `python/modules/residual/suppression_gain.py`: 2 SuppressorConfig fields + `_stat_mask_frac` instance state + `_ne_state_for_gain_rules()` helper + 4 consumer-site read replacements
- `python/modules/orchestrator.py`: propagate flags into SuppressorConfig at init; extend `_need_stationary_mask`; pass `stationary_mask=` to `get_gain`
- `python/run_one_case.py` + `python/eval_aec_challenge.py`: env hook plumbing

byte-equal at default-OFF preserved (no harness re-anchor needed).

## E.1.0 cohort 8-case sanity (FS5 + DT3)

### Section 1 — proxy fire-rate at 5% / 10% / 20% threshold (production state)

| Case | Bucket | fire@5% | fire@10% | fire@20% |
|---|---|---:|---:|---:|
| pcb1Nh | FS_static | 41.7% | 35.8% | 28.6% |
| LN18k5r8 | FS_static | 2.9% | 2.9% | 2.9% |
| s90M7MOT | FS_static | 55.1% | 51.8% | 45.6% |
| 9xjhi | FS_static | 50.7% | 46.0% | 43.8% |
| lV0kQN | FS_static | 93.0% | 90.5% | 88.3% |
| WcK0OrF | DT_movement | 53.3% | 51.1% | 50.3% |
| wVYSGV | DT_movement | 69.2% | 64.0% | 61.4% |
| xQEUtY2 | DT_static | 64.1% | 63.2% | 57.7% |

Fire rates flatten between 10% and 20% — primary threshold 10% behaves identically to a relaxed 5% on most cases. Threshold sweep does not change the outcome below.

### Section 2 — FS gate (b): proxy ON vs production (z-ON + p-OFF)

Δg100 = `gain_100[z-ON + p-ON] − gain_100[z-ON + p-OFF]`.

| Case | Bucket | frames | Δg100 > +0.3 | Δg100 < −0.3 | mean Δg |
|---|---|---:|---:|---:|---:|
| pcb1Nh | FS_static | 2318 | 15 | 0 | +0.0100 |
| LN18k5r8 | FS_static | 2358 | 0 | 0 | +0.0000 |
| s90M7MOT | FS_static | 2307 | 41 | 0 | +0.0187 |
| 9xjhi | FS_static | 2188 | 63 | 0 | +0.0241 |
| lV0kQN | FS_static | 2167 | 20 | 0 | +0.0104 |
| WcK0OrF | DT_movement | 3916 | 14 | 0 | +0.0040 |
| wVYSGV | DT_movement | 3666 | 4 | 0 | +0.0015 |
| xQEUtY2 | DT_static | 3988 | 0 | 0 | +0.0022 |

Small Δg100 > +0.3 frame counts (0-63 per case) — proxy lets a handful of HF-cap-bypassed frames have higher gain (echo-leak risk modest on FS). **Gate (b) weak-pass** (no − direction excursions; magnitudes mostly inside +0.01-0.025 mean).

### Section 3 — DT gate (a): proxy substitutes zeroing?

Compares two perturbations against shared production baseline (z-ON + p-OFF):
- **P4 baseline cata**: frames where `gain_100[z-OFF + p-OFF] − gain_100[z-ON + p-OFF] < −0.3` (= turning zeroing OFF drops gain catastrophically; the failure mode P4 retired)
- **E.1 cata**: frames where `gain_100[z-OFF + p-ON] − gain_100[z-ON + p-OFF] < −0.3` (= turning zeroing OFF *and* enabling proxy)
- Target: E.1 cata << P4 cata (≥ 50% reduction) → proxy substitutes the zeroing

| Case | P4 cata | E.1 cata | reduction |
|---|---:|---:|---:|
| WcK0OrF | 233 | 206 | **11.6%** |
| wVYSGV | 235 | 195 | **17.0%** |
| xQEUtY2 | 424 | 409 | **3.5%** |

**Gate (a) FAIL** on all 3 DT cohort tail cases. Target was ≥ 50% reduction; actual 3.5-17.0%. Proxy only saves a small fraction of the catastrophic frames the zeroing safety-net was preventing.

## Why the proxy can't reach the damage

E.0 evidence (verdict at `8a33e5d`) showed `(hf_cap & stat & ~NE)` matches 94-100% of catastrophic frames — the proxy fires on **exactly the right frames**. The implementation correctness is not the failure point.

The mechanism failure: **catastrophic-gain damage originates at residual-echo PSD level, not at gain-policy decision level**.

- Stationarity zeroing (kept default-True at P3 / P4) zeros `r²` and `r²_unbounded` directly on stationary bins inside `_aec3_post` (orchestrator stage). When the safety-net is ON, the SuppressionGain receives R² = 0 on those bins → it computes ENR ≈ 0 → gain naturally tends toward 1.0. The detector's own NE-state decision is barely consulted in that regime because the ENR-driven cost function already prefers no suppression.
- When zeroing is OFF (P4 retest), R² is correctly populated from `s² / ERLE` for those same bins → ENR ≈ render/echo (potentially high) → SuppressionGain produces aggressive far-end-tuning gain (≪ 1.0 across many bands). The NE-state false-negative then **fails to switch the tuning back to nearend**, allowing the broadband suppression. The damage is broadband (g5..g200 all hit, per E.0 §4-way cross-tab), not HF-localized.
- The Sprint E proxy operates inside `SuppressionGain.get_gain()` — it sees the already-populated R² and only changes which ENR/EMR tuning curve (`nearend_tuning` vs `normal_tuning`) is used to convert R² → gain. That choice influences gain modestly (a few hundredths to a tenth on most frames). It cannot recover the broadband drops that come from a non-zero R² × normal_tuning combination on bins that should have had R² zeroed in the first place.

In short: the zeroing is the **load-bearing mechanism**; the proxy substitutes an upstream input change (zero the R²) with a downstream policy change (pick a softer gain curve). The policy change saves 3-17% of catastrophic frames where the curve choice alone happens to be enough; the remaining 83-97% need the R² actually zeroed.

## Other candidate paths — none viable per directive

The E.0 verdict locked path (b) NE-presence augmentation specifically because:
- **(a) ENR-gate override** was REJECTED at E.0: ENR/SNR distributions of catastrophic vs normal-echo frames don't separate cleanly (wVYSGV inverted direction; xQEUtY2 compressed-at-zero), so a runtime ENR criterion cannot distinguish them.
- **(c) Frequency-band restriction** was REJECTED at E.0: damage is broadband (g5..g200 all drop to ~0 with zeroing OFF); capping HF only would leave LF/mid drops untouched.

With (b) now also failing, the only remaining direction would be to apply the proxy upstream at R²/R²_unbounded level — which is **structurally equivalent to redoing the stationarity zeroing** (P4 already in production as default-True, permanently retired as AEC3-default-off per Bucket 3). Per user directive 2026-05-21: do not move the proxy to R² level.

## Verdict

**Sprint E CLOSED no-leverage** for v3.22 ship set. Substrate stays in tree:
- `e_stat_aware_ne_proxy_enabled: bool = False` (research toggle; mirrors `aec3_post_stationarity_zero_enabled` / `transparent_mode_enabled` precedent)
- Env hook `AEC_STAT_NE_PROXY` retained for any future re-evaluation when companion mechanisms (e.g. an alternative NE detector that replaces `_DominantNearendDetector` entirely) make it viable
- `is_dominant_nearend()` public API unchanged

The Sprint E line ITEM in the v3.22 Bucket 2 trace-gated category resolves as:
- E.0: produced design (b) path-lock + falsified (a)/(c) alternatives
- E.1: implemented (b) at gain-policy level → cohort sanity FAIL → close
- E.2 / E.3: not pursued (Sprint E retired before re-trying LF-only ENR fallback; would face the same residual-echo-vs-gain-policy structural barrier)

The surviving P4 root cause (`_DominantNearendDetector` mis-classification under stationary-far) is permanently mitigated by the production-default zeroing safety-net. Any further attempt to remove the safety-net needs a **non-detector** replacement — most cleanly a Kalman-state-derived echo confidence scalar replacing the binary `is_nearend_state()` at the **residual-echo-estimator level** (R² zeroing decision), not at the gain-policy level. That belongs in a future PBFDKF-specific divergence cycle, not in v3.22.

## v3.22 impact

Sprint E was the v3.22 PRIMARY. Closure leaves v3.22 with:
- Sprint D (secondary): only proceeds if post-v3.21.6 trace shows S²_linear HF under-estimation persists despite P1's improved reverb path
- Sprint F: trace-gated no-op candidate; likely SKIP (P1 already handles tail-dead)
- Sprint G.1-G.5: trace-gated per-item probes (G.0 trace re-collection mandatory on v3.21.6 final baseline)
- Sprint H.3: G.1-gated
- Sprint I: post-bench cleanup

Recommended next step: small-scope G.0 cohort trace (single render + analysis pass) to triage which of G.1-G.5 has any leverage at all; if all trace evidence is negative, prepare for v3.22 minimal-cycle close (no algorithm change; only dormant substrate from P2 + E in tree, optionally with a small cleanup ship in Sprint I).

## Files

- E.0 verdict: [`docs/v3_22_e0_hf_cap_ne_cross_tab_verdict.md`](v3_22_e0_hf_cap_ne_cross_tab_verdict.md)
- E.1 cohort trace: `/tmp/e1_cohort/*.csv` + `analyze.py`
- E.1 code: `python/modules/config.py`, `python/modules/residual/suppression_gain.py`, `python/modules/orchestrator.py`, `python/run_one_case.py`, `python/eval_aec_challenge.py`
- P4 retest cohort baseline: `/tmp/p4_cohort_trace/*_zeroing_{on,off}.csv`
