# Research log — P3c delay acquisition latency (trace only)

Date: 2026-05-06
Code line: v3.10.4 + zero-cost diagnostic
(`AecConfig.trace_delay_est`, default off; `--trace-delay-est` CLI on
`run_one_case.py`; standalone driver `python/trace_delay_acquisition.py`).
**No behaviour change.**

## Question

The P3 arc closed with 7GT 3.366 / 3.895 declared an AECMOS asymptote
under all single-action interventions. P3d had observed that the
delay does not lock until t = 4.57 s post-start (later refined to
4.09 s on the trace below); the corresponding "blind window" before
acquisition is filter-side garbage that the post-acquisition
`_reset_filter_derived_state(reason='delay_first')` then throws
away.

P3c asks: **is 7GT's 4-second blind window an outlier on this dataset
(in which case it's a 7GT-specific fix) or representative of the
whole bench (in which case any improvement broadcasts)?**

## Method

Two-phase trace:

- **Phase A — 7GT deep trace.** Per-call DelayEstimator instrumentation
  (`trace_delay_est=True`) dumped to CSV via the new `--trace-delay-est`
  flag on `run_one_case.py`. Captures `last_par`, `confidence`,
  `is_solid`, `estimated_delay`, top-3 GCC-PHAT peaks per call.
- **Phase B — 800-case summary.** Standalone driver
  (`python/trace_delay_acquisition.py`) walks every (mic, lpb) pair
  under `wav/aec_challenge_blind`, runs the AEC end-to-end with
  `trace_delay_est=True`, and emits a one-row-per-case summary CSV
  with `time_to_first_solid_s`, `delay_at_first_solid_ms`,
  `max_par_observed`, `ever_solid`. No waveform writes.

## Phase A — 7GT timeline

| time | event | reason |
|---|---|---|
| 0 – 2.11 s | nothing | init window: `samples_accumulated < init_samples` |
| 2.11 s | first cross-spectrum update (n_updates=1) | top1_par jumps 0 → 90.5 |
| 3.07 s | first `did_estimate=True` (n_updates=2), `last_par`=119, `estimated_delay`=12 606 samples / **787.9 ms** | confidence still 0 because gate requires `n_updates >= 3` |
| 4.09 s | `is_solid=True` (n_updates=3, n_updates >= 3 gate clears) | acquisition complete |

**The delay is correctly identified at t = 3.07 s with PAR = 119 (vs
solid threshold = 8), but the `confidence` property hard-zeros the
output for another full second.** That second is pure update-count
gating, not signal-quality gating.

## Phase B — 800-case distribution

```
Time-to-first-solid (s)                                 (ever-solid cases only, n=673)
  median 4.09  mean 4.26  p25 4.09  p75 4.09  p90 4.09  p95 4.57  p99 9.57  max 10.57
```

Histogram (all 800 cases):

```
[ 4.0,  5.0) s :  642 ( 80.2%) ████████████████████████████████████████████████████████
[ 5.0,  6.0) s :   11 (  1.4%) ▌
[ 6.0,  8.0) s :    5 (  0.6%)
[ 8.0, 10.0) s :   10 (  1.2%) ▏
[10.0, 999.0) s:    5 (  0.6%)
never_solid    :  127 ( 15.9%)
```

Per-scenario:

| Bucket | n | median | p90 | max | never_solid |
|---|---:|---:|---:|---:|---:|
| DT_movement | 114 | 4.09 s | 4.09 s | 4.09 s | 0 |
| DT_static   | 186 | 4.09 s | 4.09 s | 5.57 s | 0 |
| FS_movement | 131 | 4.09 s | 4.09 s | 5.57 s | 0 |
| FS_static   | 169 | 4.09 s | 4.09 s | 5.57 s | 0 |
| NE          | 200 | 4.09 s | 9.57 s | 10.57 s | 127 |

7GT's three cases all sit at the median:

```
7GTxyT...doubletalk                    ttfs=4.09s  delay=787.9 ms  par_max=158.6
7GTxyT...doubletalk_with_movement      ttfs=4.09s  delay=795.8 ms  par_max=156.3
7GTxyT...farend_singletalk             ttfs=4.09s  delay=682.2 ms  par_max= 42.7
```

The NE-bucket "never_solid" 127/200 is correct behaviour — those
cases have no far-end signal to estimate against, so PAR stays low
and the gate never opens. Not a defect.

## Findings

### 1. 7GT is the median, not an outlier

The four scenarios that have far-end signal all show essentially a
delta function at 4.09 s — for **80.2 % of 800 cases**, time-to-solid
is exactly 4.09 s. This is structural latency built into the
estimator's init + update-count gating, identical across the
dataset regardless of true delay magnitude or signal quality.

7GT's special property is its 788 ms skew, not its acquisition time.
Once acquired, 7GT loses 4.09 s of pre-lock filter learning the same
as every other case in the bench.

### 2. The blind window is two distinct components

Decomposing 7GT's 4.09 s:

```
0    init               2.11 s    ← seg_size / sample_rate samples must
                                    accumulate before first cross-spectrum
2.11  first update      3.07 s    ← first did_estimate at n_updates=2,
                                    delay correctly identified, PAR≫threshold
3.07  gate              4.09 s    ← confidence stays 0 for ~1 s while
                                    n_updates climbs from 2 → 3
```

The third stage (~1 s) is **a free win**: PAR is already 119 (15× the
solid threshold of 8). A two-of-two confirmation rule
("n_updates >= 2 AND last_par >= solid * K, K ≈ 2") would land on
the same delay 1 s sooner without introducing false positives. Same
rule generalises to every case where PAR is unambiguously above the
threshold after the first update.

The second stage (~1 s) is also tractable in principle (smaller
`seg_size`, faster first estimate) but less of a free lunch: shorter
windows mean noisier cross-spectra and higher false-acquisition risk.

The first stage (~2.1 s) is the hardest: it's a window-length floor.

### 3. What's at stake for the bench

The pre-acquisition window is essentially wasted: the filter learns
against a misaligned reference, then `_reset_filter_derived_state(
reason='delay_first')` discards those taps when acquisition completes.
The cost is:

- 4 s of unsuppressed echo at the start of every case;
- the AECMOS scoring window (`max_len = 20 s`) overlaps this 4 s
  pre-lock zone, so 20 % of every score-relevant frame contains an
  unworking AEC.

If P3c shaves ~1 s off via the gate-tightening above, every case in
the dataset gains ~1 s of working AEC inside its scoring window.

### 4. P3c is the first P3 sub-investigation that touches all cases

P3a–P3h all targeted 7GT. Every action there either had no effect or
a slightly negative one in 7GT, and most regressed FS in the broader
bench. P3c is structurally different — the latency is dataset-wide,
the worst case (10.57 s) and the median (4.09 s) come from the same
mechanism, and the fix lives entirely inside `DelayEstimator`
(no filter / RES / DT / state-machine coupling).

## Decision

**P3c is worth pursuing**. Phase 1 proposal (next round, awaits user
OK):

- **Phase 1a** (lowest risk): tighten the `confidence` property so
  that a single update with overwhelming PAR (e.g. `last_par >=
  par_solid_threshold * 2.0`) can promote `is_solid` at
  `n_updates >= 2` instead of `n_updates >= 3`. Expected effect: ~1 s
  saved on every case where the true delay is unambiguous.
  Estimated worst case: misacquisition on noisy cases where one PAR
  spike is spurious — needs guardrail (e.g. require the same lag for
  two consecutive updates).
- **Phase 1b** (after 1a passes): study reducing `seg_size` /
  `init_samples` to lower the 2.1 s floor. Higher false-positive
  risk; need broader trace evidence before changing.

Both phases stay diagnostic-only until 800-case bench shows the
shorter ttfs translates into AECMOS gain — the P3 arc taught us that
internal metric improvements don't always translate.

## What's retained in v3.10.4 (zero-cost)

Already shipped in v3.10.4 build:
- `AecConfig.trace_delay_est: bool = False`
- `AecConfig.trace_delay_est_path: str = ""`
- `DelayEstimator._trace_rows` capture (active only when
  `trace_delay_est=True`)

Added this round (read-only diagnostics):
- `--trace-delay-est PATH` flag on `run_one_case.py`
- `python/trace_delay_acquisition.py` — 800-case driver

## Files

- 7GT trace: `/tmp/p3c_7gt_delay.csv` (3667 rows, per-call)
- 800-case summary: `/tmp/p3c_800case_summary.csv` (800 rows)
- Summary script: `/tmp/p3c_summarize.py`
- Code:
  - `python/run_one_case.py` `--trace-delay-est` flag
  - `python/trace_delay_acquisition.py` driver
