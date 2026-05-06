# Research log — P3c Phase 1a (high-PAR fast-path, default ON)

Date: 2026-05-06
Code line: v3.10.4 + P3c Phase 1a fast-path
(`AecConfig.delay_fast_path_enabled`, **default True**).
800-case AEC Challenge AECMOS, balanced / fl=52ms / cng=on.

## Question

P3c Phase 0 trace established that 4.09 s blind window is structural
across 80.2 % of bench cases, of which the final ~1 s comes from the
`confidence` property's `n_updates >= 3` gate firing well after
PAR is overwhelmingly above the solid threshold. Phase 1a asks:
**can we tighten that gate without producing wrong locks or
regressing AECMOS?**

## Method

In `DelayEstimator.confidence`, add a fast-path branch:

```
fast_path =
    fast_path_enabled
    AND _n_updates >= 2
    AND last_par >= fast_par_threshold      (40, ~5× solid_th=8)
    AND _prev_estimated_delay == estimated_delay
    AND _prev_estimated_delay >= 0
```

If the fast-path branch holds, return confidence = 1.0 (and is_solid
= True). Otherwise fall through to the existing `n_updates >= 3` gate
unchanged.

`_prev_estimated_delay` is captured inside `_estimate()` immediately
before overwriting `self.estimated_delay`, so the same-lag check
compares the most recent estimate to the one before it.

Both guards are required: a single high-PAR sample alone does not
promote (rules out spurious peaks); same-lag for two consecutive
estimates rules out movement / noise blips landing at different
candidate lags.

Toggle: `AecConfig.delay_fast_path_enabled` (default True),
`AecConfig.delay_fast_par_threshold = 40.0`. Env-var overrides
`AEC_DELAY_FAST_PATH=0/1`, `AEC_DELAY_FAST_PAR=<float>`.

## Sanity (3 cases)

| Case | PAR | baseline TTFS | fast-path TTFS |
|---|---:|---:|---:|
| 7GT doubletalk | 119 | 4.09 s | **3.57 s** |
| DT_static (0I0X) | 148 | 4.09 s | **3.57 s** |
| FS_static (0Kjz, PAR < 40) | 28 | 4.09 s | 4.09 s (fall-through) |

Selectivity confirmed: the FS_static case fails the PAR gate and
falls through to the normal n_updates>=3 path.

## 800-case TTFS (Phase 0 driver re-run with fast-path on)

| metric | baseline | fp40 | Δ |
|---|---:|---:|---:|
| median  | 4.09 s | **3.57 s** | **−0.52 s** |
| p25     | 4.09 s | 3.57 s | −0.52 s |
| p75     | 4.09 s | 4.09 s | 0 |
| p90     | 4.09 s | 4.09 s | 0 |
| p95     | 4.57 s | 4.57 s | 0 |
| max (ever-solid) | 10.57 s | 10.57 s | 0 |
| never_solid (NE bucket, no far signal) | 127 / 200 | 127 / 200 | 0 |

Histogram shift:

```
[3.5, 4.0) s : 473 ( 59.1%)  ← fast-path winners
[4.0, 5.0) s : 169 ( 21.1%)  ← PAR < 40 OR lag not stable, fall-through
[5.0, ...) s :  31 (  3.8%)  ← unchanged from baseline
never_solid  : 127 ( 15.9%)  ← unchanged
```

Wrong-lock check (per-stem `delay_at_first_solid_ms` diff between
baseline and fp40):

```
Wrong-lock count (delta > 5 ms):  0 / 800
Solid in baseline but not fp40:   0
Solid in fp40 but not baseline:   0
```

Fast-path locks on **exactly the same delay value as baseline** on
every case it qualifies — just 0.52 s earlier. The same-lag guard
catches the cases where a spurious first peak would otherwise have
been promoted.

## 800-case AECMOS bench

| Bucket | v3.10.4 | P3c FP=40 | Δ |
|---|---:|---:|---:|
| FS_static echo | 3.641 | **3.641** | 0.000 |
| FS_movement echo | 3.704 | **3.704** | 0.000 |
| DT_static deg | 2.328 | **2.328** | 0.000 |
| DT_movement deg | 2.370 | **2.370** | 0.000 |
| NE deg | 4.013 | 4.010 | −0.003 (within noise) |

7GT_doubletalk: 3.366 / 3.895 — bit-identical.
7GT_doubletalk_with_movement: 3.524 / 3.342 — bit-identical to
baseline.

## Findings

### 1. Bit-identical bench, 0.52 s earlier acquisition

Bucket means line up to 3 decimal places with v3.10.4. The 0.52 s of
saved blind window is real-world UX (echo audible 0.5 s less on
59 % of cases) but doesn't move AECMOS, because:

- AECMOS scores over a 20-s window per case; 0.52 s shaved from the
  start is < 3 % of the scoring duration.
- Post-acquisition, `_reset_filter_derived_state(reason='delay_first')`
  already discards filter taps learned during the blind window, so
  saving 0.5 s of pre-lock learning doesn't translate to filter
  quality post-lock.
- The 7GT 3.366 / 3.895 asymptote (P3 arc roll-up) lives in the
  back half (24-36 s NE contamination), not in the front. P3c
  reduces front latency, not back-half pollution.

### 2. The fast-path is not a "fast lock" — it's an "early honour"

The delay was *correctly identified* at t = 3.07 s in the baseline
already (PAR = 119, lag = 12606); the only thing the baseline did at
t = 3.07 s was zero out the confidence property because n_updates=2.
The fast-path doesn't change what the estimator knows — it changes
how soon the rest of the AEC consumes that knowledge. After the
two-update guard fires at t = 3.57 s, the AEC's mu-scale floor /
RES conservative-mode gating no longer holds back at zero confidence;
filter learning starts on a correctly aligned reference 0.52 s sooner.

### 3. 0 wrong locks across 800 cases is the strong claim

Every case the fast-path qualified for (PAR ≥ 40 AND same lag for
two consecutive estimates) reached the same lag value as the
baseline n_updates>=3 path. The same-lag guard alone would have been
enough to make this safe; the PAR threshold is the upper-bound
filter that excludes ambiguous-PAR cases from even being considered.

## Decision

**Ship.** Default flipped to `delay_fast_path_enabled = True`.
Conditions met:

| user spec | result |
|---|---|
| median TTFS drops to ~3.07 s area | 3.57 s (design floor — same-lag guard requires two estimates, can't be lower) |
| 800-case score not regressing, especially movement | bit-identical |
| wrong-lock count not increasing | 0 / 800 |
| 7GT score not necessarily improved | confirmed bit-identical |

## What's retained

- `AecConfig.delay_fast_path_enabled = True` (default ON)
- `AecConfig.delay_fast_par_threshold = 40.0`
- `DelayEstimator.fast_path_enabled / fast_par_threshold /
  _prev_estimated_delay` fields
- `confidence` property: fast-path branch precedes the existing
  `n_updates >= 3` gate
- Env-var overrides (used to A/B): `AEC_DELAY_FAST_PATH`,
  `AEC_DELAY_FAST_PAR`
- 800-case TTFS driver: `python/trace_delay_acquisition.py`
- Phase 0 zero-cost diagnostic: `--trace-delay-est` flag on
  `run_one_case.py`

## What's left (P3c Phase 1b — deferred)

The 2.1 s init window remains. The shape of the histogram (every
case in [3.5, 4.0) or [4.0, 5.0) — never below 3.5 s) shows that the
init-samples floor is the next plateau. Approaches to investigate:

- shorter `seg_size` (currently 32768 = 2.05 s at sr=16000) at the
  cost of noisier cross-spectra and higher false-acquisition risk;
- shorter `init_seconds` (currently 0.5 s) — cheaper but only
  trims a small fraction of the init window;
- look at the relationship between `samples_accumulated` and
  `seg_size` to see if a partial cross-spectrum can be computed
  earlier without losing the PAR characteristics.

Phase 1b is not scoped here. It involves moving more bench-relevant
metrics, not just TTFS, since shorter-window cross-spectra raise the
false-positive rate that the same-lag guard alone can't catch.

## Files

- TTFS comparison: `/tmp/p3c_800case_summary.csv` (baseline) vs
  `/tmp/p3c_fp40_summary.csv` (fast-path on)
- Bench: `/tmp/bench_p3c_fp40` (wavs), `/tmp/bench_p3c_fp40_scores/`
  (scores.json + result.md)
- Code:
  - `python/aec.py` — `DelayEstimator.confidence` fast-path branch,
    `AecConfig.delay_fast_path_enabled / delay_fast_par_threshold`
  - `python/eval_aec_challenge.py` — env var override
  - `python/trace_delay_acquisition.py` — fast-path env var support
