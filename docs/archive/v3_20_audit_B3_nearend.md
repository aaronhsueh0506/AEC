# v3.20 ADDENDUM 3 — Audit B.3: SuppressionGain `nearend_spectrum` wiring

**Component**: `res_filter_aec3.py:_stage_gain_compute` → `Aec3SuppressionGain.get_gain(nearend_spectrum=...)`

**Date**: 2026-05-17

**Status**: AUDIT COMPLETE; FIX IMPLEMENTED behind `AEC_B3_USE_ERROR_PSD=1` env flag pending verdict.

## Divergence table

| # | OUR port (file:line) | AEC3 source (file:line) | Severity | Note |
|--:|---|---|---|---|
| B.3-1 | `python/modules/res_filter_aec3.py:152` — `nearend = self.near_psd` (= mic² Y2 always) | `docs/aec3_extracts/src/aec3/echo_remover.cc:452` — `nearend_spectrum = aec_state.UsableLinearEstimate() ? E2 : Y2` | **BLOCKER** | When the linear filter is usable, AEC3 passes the post-cancellation error² (E2) so the SG sees what is LEFT to suppress, not raw mic. Passing Y2 unconditionally dilutes the per-bin ENR (= R²/nearend) by the echo magnitude itself and prevents the SG mask gate from firing on the majority of bins. |
| B.3-2 | Same call site — `comfort_noise = self.noise_psd` | `echo_remover.cc:526` — `cng_.NoiseSpectrum()` | OK | Our `noise_psd` is min-statistics over `error_psd` (`res_filter.py:1759`); semantically aligned with AEC3 `NoiseSpectrum()`. No fix needed. |
| B.3-3 | `suppression_gain_aec3.py:302-307` — `DominantNearendDetector.update(nearend_spectrum=nearend.mean(axis=0))` | `dominant_nearend_detector.cc` | OK (coupled to B.3-1) | The detector consumes the same `nearend_spectrum` as SG. Spec is correct (AEC3 source feeds the same buffer). When B.3-1 fix flips to E2, the detector's `is_nearend` decision moves with it — semantically correct per AEC3, but bench must verify no NE deg regression. |

## Probe evidence

**Pre-fix (baseline)** — 5 cohort cases on full AEC3-on stack
(AEC3 SG + AEC3 R² estimator + full AecState + post-stage override).

| case | usable_lin % | amp_min p50 | amp_med p50 | ENR sum-ratio p50 | ERLE p50 (dB) | out_rms p50 |
|---|---:|---:|---:|---:|---:|---:|
| FS_static (9xjhi) | 97.4% | 1.0000 | 1.0000 | 0.017 | 3.13 | 0.2119 |
| FS_movement (0luXw) | 94.2% | 1.0000 | 1.0000 | 0.000 | 1.95 | 0.0844 |
| DT_static (p0mhF) | 36.4% | 1.0000 | 1.0000 | 0.000 | 4.26 | 0.0066 |
| DT_movement (wHmBm) | 94.5% | 1.0000 | 1.0000 | 0.000 | 2.17 | 0.0263 |
| NE (SR68l) | 0.0%  | 1.0000 | 1.0000 | 0.000 | 1.52 | 0.0218 |

`amp_med = 1.0000` everywhere = SG mask gate fires on 0/65 bins on the median frame.
Per-frame R² is concentrated in a handful of bins; on the others R²≈0 so even after the
fix the gate doesn't fire (R²/nearend ≪ transparent_lf=0.5).

**Post-fix (`AEC_B3_USE_ERROR_PSD=1`)** — same 5 cases.

| case | usable_lin % | amp_min p50 | amp_med p50 | ENR sum-ratio p50 | ERLE p50 (dB) | out_rms p50 |
|---|---:|---:|---:|---:|---:|---:|
| FS_static (9xjhi) | 97.4% | **0.4395** | 1.0000 | **0.092** | **3.33** | **0.2017** |
| FS_movement (0luXw) | 94.2% | 1.0000 | 1.0000 | 0.000 | 1.96 | 0.0842 |
| DT_static (p0mhF) | 36.4% | 1.0000 | 1.0000 | 0.000 | 4.26 | 0.0066 |
| DT_movement (wHmBm) | 94.5% | 1.0000 | 1.0000 | 0.000 | 2.23 | 0.0262 |
| NE (SR68l) | 0.0%  | 1.0000 | 1.0000 | 0.000 | 1.52 | 0.0218 |

**Signal-level findings**:
- FS_static (worst): `amp_min` drops 1.000 → 0.439 (SG now suppresses deep on some bins),
  ENR sum-ratio up 5.4×, ERLE +0.20 dB, out_rms −5%.
- Median bin still doesn't fire (R² distribution is the limiter, not the wiring).
- Other 4 buckets: neutral (FS_movement/DT_static/DT_movement have R² ≈ 0 even with E2;
  NE has usable_lin=0% so fix path doesn't fire at all).

**Direct near_psd vs error_psd check on 9xjhi**:
- `res.near_psd_sum` p50 = 8.18e3
- `res.error_psd_sum` p50 = 2.60e3
- `error_psd / near_psd` ratio p50 = 0.36
- frames where error ≤ 0.5 × near: 1724 / 2186 (filter does help)
- frames where error > near (would have hurt): 18 / 2186
- → using E2 is the correct AEC3-aligned signal.

## Fix

**File**: `python/modules/res_filter_aec3.py:152` (inside `_stage_gain_compute`).

```python
nearend = self.near_psd  # AEC3 Y2 path (default)
if (self._aec3_full_state is not None
        and os.environ.get('AEC_B3_USE_ERROR_PSD', '0') == '1'):
    try:
        if self._aec3_full_state.usable_linear_estimate():
            nearend = self.error_psd  # AEC3 E2 path (echo_remover.cc:452)
    except Exception:
        pass
```

(Env flag for clean A/B during 60-case bench; final commit will promote to
default-ON for the AEC3-SG path and remove the env hook.)

## Cold-start risk

`error_psd` starts at 0 (`res_filter.py:326`); cold-start seeds it from the first
non-zero far-end frame (`res_filter.py:1914-1916`), then EMA-blends at α=0.8
(≈5-frame TC). `aec_state.usable_linear_estimate()` should be False during this
window (AEC3 startup gate + filter_converged gate), so the fallback to Y2 naturally
covers it. The smoke probe confirms `usable_lin %` is 0% on NE (no far-end activity)
and ~95% on cases with sustained far-end — gate behaves as expected.

## Decisions

- **Fix is correct per AEC3 source** (`echo_remover.cc:452` is unambiguous).
- **Smoke shows small positive on FS_static, neutral elsewhere** — the AEC3 R²
  estimator output is too small in absolute terms on the worst FS cases for the
  ENR gate to fire on most bins even after the wiring fix. Per-bin R² peak might
  still cross threshold on the bins it does land on (the `amp_min p50` 1.000 → 0.439
  drop reflects this).
- **60-case bench pending** — AECMOS bucket-mean Δ will determine PROMOTE vs CLOSE.
- **If FS_static lifts ≥ +0.05 mean Δecho AND no other column regresses ≥ -0.05**
  → 800-case bench → ship as default-ON for AEC3 SG path.
- **If neutral or regression** → close as audit-only. The fix is still spec-correct
  but the AEC3 SG path on our cohort cannot benefit from it because the limiter is
  upstream (R² magnitude or filter convergence quality), not the wiring.
- Either way the **R² magnitude is the next audit target** (component B.2 — AEC3
  ResidualEchoEstimator). Probe shows R²_sum p50 = 1.5e3 on FS_static, with
  error_psd_sum p50 = 2.6e3 → R² is ~57% of post-cancellation residual.
  AEC3's `default_gain_amplitude` 0.01 vs our 0.1 in the nonlinear path
  could shrink R² further; the delayed-render reverb (`cc:367`) we did not
  port could be inflating it. Audit B.2 needs to walk these.

## Inline-cleanup candidates surfaced in this audit

- `aec3_echo_audibility_floor_power` (config.py) — default 0 disables
  WeightEchoForAudibility; verified in any preset / env / commit?
  → audit during B.4 close-out, may be retire candidate.
- `_aec3_pending_gain` breadcrumb (`res_filter_aec3.py:187`) — only meaningful
  when override is on; safe but adds 3 conditional blocks.
  → retire when `res_aec3_skip_legacy_post_stages` is promoted to default-ON
  (decision in B.5).
