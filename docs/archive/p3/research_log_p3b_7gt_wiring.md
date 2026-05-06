# P3b — 7GT delay-est "wiring bug" investigation

## Goal
The earlier session's recovery-event-attribution trace claimed that on
the 7GT doubletalk case the production `DelayEstimator` "locks at lag=8
samples at t=2.05s", while the standalone `python/diagnose_gcc_phat.py`
(P3a) cleanly finds the true 788 ms (12619-sample) skew with PAR≈80.
P3b's job: instrument `DelayEstimator.accumulate()` and trace what the
production estimator actually sees on every frame, to discriminate
between hypotheses (A) input-reshape, (B) gating, (C) early spurious
peak, (D) `_n_updates` race.

## Method
Added an opt-in trace hook (default OFF, no v3.10.4 behaviour change):

- `AecConfig.trace_delay_est: bool = False` (and `trace_delay_est_path`
  reserved for a later flush helper)
- `DelayEstimator.__init__(trace=False)`; when `trace=True`, every call
  to `accumulate()` recomputes the GCC-PHAT over the running EMA cross-
  spectrum in fp64, extracts top-3 peaks (greedy 16-sample lobe
  suppression), and appends a row to `self._trace_rows` with
  `mic_pwr_pre`, `ref_pwr_pre`, `n_updates_pre/post`,
  `samples_accumulated`, `in_init_window`, `init_done`, `did_estimate`,
  `estimated_delay`, `last_par`, `confidence`, `is_solid`, top-3
  `(lag, height, par)`. The estimator's decision logic is untouched —
  the trace recomputes peaks AFTER the production code has already run.

Driver: `/tmp/p3b/run_trace.py` runs preset=balanced, filter=832,
cng=on on `7GTxyTksSUqCnP5y0ILG4A_doubletalk_{mic,lpb}.wav` (no pad)
and writes `/tmp/p3b/7gt_delay_trace.csv` (3667 rows, one per `process()`
call).

## Trace findings

`DelayEstimator` configuration (logged at startup):
- `seg_size = 32768` (2048 ms; next pow2 ≥ 2× max_delay 16384)
- `seg_hop = 16384` (≈ 1.024 s) — so `_n_updates` increments only once
  per ~1 s of audio
- `max_delay_samples = 16384` (1024 ms)
- `_init_samples = 4800` (0.3 s); `_period_samples = 8000` (0.5 s)

Timeline of `_n_updates` and `_estimate()` events:

| call_idx | t (s) | n_up_pre | n_up_post | did_estimate | top1 lag | top1 PAR | confidence | is_solid | _current_delay |
|---:|---:|---:|---:|---|---:|---:|---:|---|---:|
| 0–203 | 0.00–2.03 | 0 | 0 | False | n/a | n/a | 0 | F | -1 |
| 204 | 2.04 | 0 | **1** | False | **12606** | **90.52** | 0 | F | -1 |
| 205–306 | 2.05–3.06 | 1 | 1 | False | 12606 | 90.52 | 0 | F | -1 |
| 307 | 3.07 | 1 | **2** | **True** | 12606 | 119.16 | 0 | F | -1 |
| 308–406 | 3.08–4.06 | 2 | 2 | False | 12606 | 119.16 | 0 | F | -1 |
| 407 | 4.07 | 2 | 3 | True | 12606 | 119.16 | 0 (still <3 in `confidence` boundary) | F | -1 |
| 457 | 4.57 | 3 | 3 | True | **12606** | **120.04** | **1.0** | **T** | **12606** ← Path A first commit |

Across all 3667 frames, the distribution of `estimated_delay` values
is concentrated in {12606, 12607, 12608, 12609, 12619, 12624, 12626} —
all within ~1 ms of the standalone diagnostic's 12619-sample target.
**No "lag=8" estimate was ever produced.** No early Path-A trigger
fires before t=4.57s. `_current_delay` is `12606` for the entire run.

### Root cause

The reported "delay_first fires at t=2.05s with new_delay=8" event
**does not exist on this case in the current codebase**. The bug
hypotheses (A), (B), (C), (D) all fail to fire:

- (A) is irrelevant — even after HPF + soft-clip, the 12606-sample
  peak has PAR > 90 from the very first segment.
- (B) `accumulate()` IS called every frame (3667 calls for ~36.7 s).
- (C) No spurious early peak: top1 lag is 12606 from `_n_updates=1`
  onward; lag=0..32 PAR is dominated by 12606.
- (D) `_n_updates >= 3` gate is in fact protective here: by the time
  it is satisfied, the peak has consolidated to PAR=120 at lag 12606.

The earlier "lag=8 at t=2.05s" report was therefore either (1) from a
**different branch/code state** than v3.10.4 release, (2) from a
**different case** mislabelled as 7GT, or (3) from an instrumentation
bug in the previous trace tool. Production v3.10.4 on 7GT no-pad
balanced/cng/fl=832 correctly locks `_current_delay = 12606` at
t≈4.57 s and holds it for the entire utterance.

This matches **none of (A)–(D)**: the supposed bug is not present.

### Frame-budget observation (not the bug, but worth noting)

There is still a **2.5 s "blind window"** between segment start and
first commit: `seg_size=32768` requires 2.05 s of audio before
`_n_updates` reaches 1, and 4.57 s before all of (`_n_updates ≥ 3`
+ `is_solid` + Path A) fire. During those 4.57 s the AEC filter
adapts against ref-aligned-to-zero (`_current_delay = -1`), which
poisons filter state and forces the post-Path-A reset to discard
~4 s of adaptation. That is a separable convergence-latency issue
(P3c+ candidate), not the delay-estimation correctness bug we set out
to find.

## Fix proposal (none required for the alleged bug)

Since the alleged "wiring bug" is not reproducible in v3.10.4, no fix
is needed. P3c should instead:

1. Re-run the original recovery-event-attribution trace and verify
   whether its "delay_first new_delay=8" event was on a different
   commit or a different case. If it was on the same commit, examine
   the trace tool itself.
2. If the convergence-latency issue (4.5 s blind window before Path A
   fires) is the real performance gap on 7GT, evaluate either:
   a. Reducing `seg_size` for the first acquisition (e.g., a coarse
      decimated-rate pre-search) so initial commit can fire earlier,
      OR
   b. Pre-charging `_n_updates` by overlapping segments more aggressively
      until `is_solid` is reached, then switching back to 50% overlap.
   Both are intrusive and should be benched separately.

## Files added/modified

- `python/aec.py` — added `AecConfig.trace_delay_est` + `trace_delay_est_path`;
  added `trace=False` arg to `DelayEstimator.__init__` plus per-call
  fp64 top-3 instrumentation in `accumulate()`; threaded the flag from
  `AEC.__init__`. All instrumentation gated behind `if self._trace`,
  zero overhead when off.
- `/tmp/p3b/run_trace.py` — driver that runs balanced/cng/fl=832 on
  7GT no-pad and dumps the trace.
- `/tmp/p3b/7gt_delay_trace.csv` — 3667 rows, full per-frame trace.
- `/tmp/p3b/analyze.py` — small post-processor.
- `docs/research_log_p3b_7gt_wiring.md` — this file.
