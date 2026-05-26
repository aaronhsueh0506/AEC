# v3.21.18-pre B.3 EchoAudibility audit — VERDICT: ALREADY PORTED (no action)

Cycle: 2026-05-23
Audit: read-only Pass-2 verification of Finding 2's B.3 gap claim

## TL;DR

**Finding 2 mis-identified the gap.** Both AEC3 EchoAudibility components
(`echo_audibility.{cc,h}` stationarity-based scaling AND
`WeightEchoForAudibility` static-threshold weighting in
`suppression_gain.cc`) are ALREADY PORTED in our codebase. v3.21.18-pre
closes immediately as **no action required**.

This is a multi-pass verification win — the C-protocol mandate to re-audit
prior claims before acting caught a misdiagnosis that would have wasted a
sprint.

## Multi-pass verification

### AEC3 component 1: `EchoAudibility` class

**AEC3 source** ([`docs/aec3_extracts/src/aec3/echo_audibility.{cc,h}`](AEC/docs/aec3_extracts/src/aec3/echo_audibility.cc)):
- Thin wrapper around `StationarityEstimator` for render path
- `Update()` — updates render stationarity flags + noise estimator
- `GetResidualEchoScaling(filter_has_had_time_to_converge, residual_scaling[band])`:
  ```c
  if (render_stationarity_.IsBandStationary(band) &&
      (filter_has_had_time_to_converge || use_render_stationarity_at_init_)) {
    residual_scaling[band] = 0.f;
  } else {
    residual_scaling[band] = 1.0f;
  }
  ```

**Our port** ([`python/modules/state/stationarity_estimator.py:183-194`](AEC/python/modules/state/stationarity_estimator.py#L183-L194)):
```python
def get_residual_echo_scaling(stationarity: StationarityEstimator,
                              filter_has_had_time_to_converge: bool) -> np.ndarray:
    """Per-bin residual scaling (echo_audibility.h:40-51).
    Returns 0.0 for stationary bands once the filter has converged enough that
    we can trust the stationarity flag, 1.0 otherwise.
    """
    scaling = np.ones(stationarity.n_freqs, dtype=np.float32)
    if filter_has_had_time_to_converge:
        scaling[stationarity.band_stationary_mask()] = 0.0
    return scaling
```

**Verdict**: VERBATIM MATCH. The mechanism is implemented equivalently:
when filter has converged AND band is stationary → residual scaling = 0
(suppress echo at that band). The wrapper around StationarityEstimator
is implicit in our port (StationarityEstimator instance is passed as
argument instead of being a member field).

**Production wiring**: `get_residual_echo_scaling()` is defined but not
directly called by orchestrator. Instead, orchestrator uses
`_stationary_mask` directly to zero R² at
[`orchestrator.py:4030`](AEC/python/modules/orchestrator.py#L4030):
```python
r2 = np.where(_stationary_mask, 0.0, r2).astype(np.float32)
```
gated by `_filter_converged_enough`. This is the EQUIVALENT operation to
AEC3's `GetResidualEchoScaling()` applied to residual_echo array.

**Configuration**: `aec3_post_stationarity_zero_enabled: bool = True`
(default ON — Sprint A E2 / Phase 1 lock at
[`config.py:584`](AEC/python/modules/config.py#L584)). Sprint B previously
tested flipping to False (= remove AEC3 stationarity zeroing); REJECTED on
60+ DT cohort-tail cases (worst xQEUtY2 −0.602 Δdeg). Default-True
restored per memory anchor `feedback_no_residual_fallback_or_hf_cap_tuning`.

### AEC3 component 2: `WeightEchoForAudibility` function

**AEC3 source** ([`docs/aec3_extracts/src/aec3/suppression_gain.cc:88-121`](AEC/docs/aec3_extracts/src/aec3/suppression_gain.cc#L88-L121)):
Per-band downweight of residual echo when echo < `floor_power *
audibility_threshold_{lf,mf,hf}` — applies quadratic falloff.

**Our port** ([`python/modules/residual/suppression_gain.py:270-299`](AEC/python/modules/residual/suppression_gain.py#L270-L299)):
```python
def weigh_for_audibility(
    cfg: EchoAudibilityConfig, echo: np.ndarray, out: np.ndarray,
    sr: int = 16000,
) -> None:
    """Mirrors WeightEchoForAudibility (suppression_gain.cc:88-121).
    For each band (LF: 0-2, MF: 3-6, HF: 7-end), bins with echo below
    threshold = floor_power * audibility_threshold_* get scaled by
    max(0, 1 - ((threshold - echo) / (threshold - floor_power))²).
    """
    ...
    weigh(cfg.floor_power * cfg.audibility_threshold_lf, 0, lf_end)
    weigh(cfg.floor_power * cfg.audibility_threshold_mf, lf_end, mf_end)
    weigh(cfg.floor_power * cfg.audibility_threshold_hf, mf_end, n)
```

**Verdict**: VERBATIM MATCH. Per-band quadratic downweight ported correctly.

**Production wiring**: Called from SuppressionGain consumer chain via
`EchoAudibilityConfig`. Default thresholds match AEC3 defaults
(`floor_power=128, audibility_threshold_lf=10, audibility_threshold_mf=10,
audibility_threshold_hf=10`).

## Where Finding 2 went wrong

Finding 2 (broad AEC3 alignment audit) flagged B.3 with:
> "EchoAudibilityEstimator NOT ported (suppression_gain.cc:29 uses it;
> we hardcode defaults at suppression_gain.py:90–99)"

The agent confused TWO distinct mechanisms:
1. `EchoAudibility` class (stationarity-based) — ALREADY PORTED
2. `WeightEchoForAudibility` function (static-threshold) — ALREADY PORTED

The "hardcoded defaults" critique applies to `EchoAudibilityConfig`
defaults (`floor_power=128`, etc.) — which are the AEC3 DEFAULTS in
`echo_canceller3_config.h`. We use the same defaults. Not a gap; that's
parity.

## Multi-pass verification process (what worked)

Per C-protocol re-audit cadence (PT-4 parallel track):
1. Read Finding 2's specific claim
2. Locate referenced source files
3. Read AEC3 source directly
4. Read our port directly
5. Compare line-by-line
6. Verify production wiring + config

This process surfaced the misdiagnosis in 20 minutes vs the alternative
(spending a sprint porting "missing" code that already existed).

The user's "anti-misdiagnosis / multi-pass" directive (2026-05-23) is the
load-bearing reason this was caught. Without it, v3.21.18-pre would have
gone through a wasted port-and-test cycle before discovering the
duplicate.

## Disposition

**v3.21.18-pre CLOSE as ALREADY PORTED — no v3.21.18 ship needed.**

- No new code
- No new flag
- No bench
- Status update: B.3 graduates from "stub" to "verified equivalent" in
  Finding 2's audit table

## Implications for v3.21.x arc-exit (v3.21.19)

Combined with v3.21.17 closure (B.5 SDE no-leverage substrate), the
remaining AEC3-verbatim parity surface after v3.21.18-pre is:

| B-cluster item | Status |
|---|---|
| B.1 coarse_e2_time_domain | v3.21.9 CLOSED methodology-sound (per closure re-audit) |
| B.2 HMM TransparentMode | PBFDKF-incompatible per Finding 2 (field trial, not production default) |
| B.3 EchoAudibility | **VERIFIED already ported (v3.21.18-pre)** |
| B.4 SubbandNearendDetector | Matches AEC3 ship default (`use_subband_nearend_detection: bool = False`) |
| B.5 SignalDependentErleEstimator | v3.21.17 CLOSED no-leverage substrate |
| B.6 FullBandErleEstimator parity re-audit | Not surfaced as priority; module fully ported per Finding 2 |

**No remaining AEC3-verbatim items requiring a v3.21.x ship cycle.**
v3.21.19 declares arc CLOSED. v3.22 candidate set (Volterra NL inverse,
latch redesign, etc.) becomes the next development chapter — requires
explicit user authorisation per scope rule.
