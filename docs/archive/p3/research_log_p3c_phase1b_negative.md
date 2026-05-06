# Research log — P3c Phase 1b (n_updates=1 gate, negative)

Date: 2026-05-06
Code line: v3.10.4 + P3c Phase 1a (default ON). No behaviour change
in this branch — trace-only investigation.

## Question

P3c Phase 1a fast-path (gate at `n_updates >= 2` with same-lag-twice
guard) drops TTFS median 4.09 s → 3.57 s on 59 % of cases with
0 / 800 wrong locks. The remaining ~3.57 s floor decomposes as:

- ~2.05 s init window (`buf_pos < seg_size = 32768 samples`, where
  `seg_size = 2 × max_delay_samples` = bound on the first
  cross-spectrum frame)
- ~1.0 s between first cross-spectrum (`n_updates = 1`) and the
  second one needed for same-lag confirmation (`seg_hop = 16384
  samples = ~1 s`)

Phase 1b asks: can we eliminate the second ~1 s by gating at
`n_updates >= 1` (single update) with a high-PAR guard? Per user
spec: trace-only first; if the n=1 cross-spectrum produces wrong
locks at any reasonable PAR threshold, STOP — partial-buffer FFT
becomes a separate larger research branch, not this release path.

## Method

Extended `python/trace_delay_acquisition.py` to capture per-update
snapshots: at each `n_updates ∈ {1, 2, 3}`, record the cross-spectrum
state (`time_s`, `top1_lag`, `top1_par`, `top2_par`). Computed
`same_lag_1_to_2 = (nup1_top1_lag == nup2_top1_lag)`. Re-ran the
800-case driver. No behaviour change.

## Results

### Same-lag rate between first and second update

| scenario | same-lag-1→2 | n |
|---|---:|---:|
| overall (ever-solid) | **73.25 %** | 673 |
| DT_movement | 79.82 % | 114 |
| DT_static | 81.72 % | 186 |
| FS_movement | **71.76 %** | 131 |
| FS_static | 79.88 % | 169 |
| NE | 28.77 % | 73 |

Target was ≥ 95 %. Reality is **22 percentage points short of target
on the best scenario**. Movement buckets are the worst.

### Wrong-lock pool (`nup1_top1_lag != nup2_top1_lag`)

180 / 673 = **26.75 %** of ever-solid cases. PAR distribution in
this pool:

```
min=4.72  p25=8.58  median=22.01  p75=42.33  max=199.45
```

The wrong-lock pool reaches **PAR = 199.45** at the first update —
i.e. even an overwhelming-PAR first cross-spectrum can lock on the
wrong lag. No PAR threshold separates correct from wrong cleanly.

### PAR threshold sweep — gate at `n_updates >= 1` if `nup1_top1_par >= X`

| X | cases pass | wrong | wrong-rate among triggered | benefit (TTFS savings ≈ 1 s on cases passing) |
|---:|---:|---:|---:|---|
| 40 (Phase 1a's threshold) | 419 (62.3 %) | 46 | 11.0 % | UNSAFE |
| 60 | 303 (45.0 %) | 27 | 8.9 % | UNSAFE |
| 80 | 220 (32.7 %) | 18 | 8.2 % | UNSAFE |
| 100 | 144 (21.4 %) | 7 | 4.9 % | still > 0 wrong |
| 120 | 94 (14.0 %) | 5 | 5.3 % | still > 0 wrong |
| 150 | 55 (8.2 %) | 2 | 3.6 % | tiny benefit, still > 0 wrong |

**No threshold drives wrong-lock to 0 while preserving meaningful
benefit.** Even at PAR ≥ 150 (extreme), 2 of 55 triggers wrong-lock,
and only 8 % of cases benefit at all.

### 7GT specifically (the case that motivated all P3)

| 7GT variant | nup1_par | nup1_lag | nup2_lag | same |
|---|---:|---:|---:|---|
| doubletalk | 90.5 | 12606 | 12606 | ✓ |
| doubletalk_with_movement | 16.99 | 12731 | 12606 | ✗ |
| farend_singletalk | 9.6 | 10918 | unknown | ✗ |

**2 of 3 7GT variants would WRONG-LOCK** under any n=1 gate. The
exact case the entire P3 arc was trying to help cannot be helped by
this approach.

## Why is the n=1 cross-spectrum so unreliable?

The first cross-spectrum is computed on the very first
`seg_size = 32768` samples of mic and reference. Two structural
issues:

- The signal in those first ~2 seconds may be predominantly silence
  / weak energy, even when the case as a whole has clear far-end
  activity later. PAR can spike high on near-silent input because
  the mean is tiny.
- Movement cases have an actively shifting delay; the GCC-PHAT peak
  in the first 2 seconds reflects whatever lag held *during those
  2 seconds*, which is not necessarily the same lag the second
  segment converges on.

The same-lag-twice guard from Phase 1a is doing real work: it
filters out ~27 % of cases where the first peak is wrong, and
brings the remaining 59 % into the fast-path safely.

## Decision

**STOP.** Per user stop criteria:

| criterion | result |
|---|---|
| wrong locks 0 or 極低 | not met at any X (≥3.6 % at X=150) |
| movement bucket 不容易出現低品質早鎖 | FS_movement same-lag rate 71.76 % |

The 1-update gate cannot be made safe by tightening PAR alone.
The signal isn't clean enough to gate on a single cross-spectrum.

**Partial-buffer FFT (`seg_size` shrinkage at first FFT only) is
explicitly left out of scope** for this release path. It would
require:

1. A separate FFT plan for the first FFT (smaller window),
2. Calibration of cross-spectrum noise floor at smaller window,
3. Re-running the same trace-then-bench cycle to validate.

That's a larger research branch, not a release-path tweak.
Recorded as future work.

## What's left of the original 4 s blind window after the P3c arc

- 2.05 s init window — structural, max_delay-bound, untouched
- 1.0 s n=1 → n=2 gap — same-lag-twice guard rules out lock-early
- 0.5 s n=2 → solid (Phase 1a saved this) — shipped, default ON

The Phase 1a fast-path harvested everything that could be harvested
at the n=2 boundary. Anything below n=2 requires a different
architecture (partial FFT or smaller seg_size at fixed max_delay),
which is not in scope.

## Files

- 800-case trace summary: `/tmp/p3c_phase1b_summary.csv`
- Driver: `python/trace_delay_acquisition.py`
  (extended with per-update `nup{1,2,3}_*` columns)
- This log: `docs/research_log_p3c_phase1b_negative.md`
