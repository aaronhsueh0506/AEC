# Research log — P3h sustained-diverged filter reset (negative, single-case dry-run)

Date: 2026-05-06
Code line: v3.10.4 + P3 arc diagnostics
(`diverged_reset_enabled` toggle, default off).

## Question

P3 arc summary identified one state-driven action not yet tried:
**filter reset on sustained `diverged`**. P3f's classifier explicitly
distinguished `suspicious_dt` (mu reduction action) from `diverged`
(reset action) — the latter was deliberately left unwired pending a
focused single-case dry-run.

Per user direction: 7GT_doubletalk single-case only. No 800-case until
the single case visibly moves on AECMOS, not just on internal trace
metrics.

## Method

Added `AecConfig.diverged_reset_enabled` (default False),
`diverged_reset_streak_frames` (default 50), and
`diverged_reset_cooldown_frames` (default 400). When enabled and all
guards hold:

```
if (diverged_reset_enabled
        and self._filter_once_converged
        and self._p3h_reset_cooldown_remaining == 0
        and filter_state == 'diverged'
        and self._p3f_diverged_streak >= diverged_reset_streak_frames):
    self._reset_filter_derived_state(reason='p3h_diverged')
    self._p3h_reset_cooldown_remaining = diverged_reset_cooldown_frames
```

Added `--diverged-reset` / `--diverged-reset-streak` /
`--diverged-reset-cooldown` flags to `run_one_case.py`. New diag
fields `p3h_reset_fired`, `p3h_reset_count`, `p3h_reset_cooldown` in
`_diag` and `--trace-aec-state` CSV.

## Streak threshold tuning

First dry-run with `streak=50` (default) fired 0 resets. Inspected the
baseline trace's longest sustained `diverged` run on 7GT and found
**max = 27 frames**, with 9 separate `diverged` runs of length
`[19, 7, 6, 4, 4, 4, 4, 3, 2]` in 24–36 s. Re-ran with `streak=20`
to make the dry-run informative; 1 reset fires at `t = 26.95 s`.

The 27-frame ceiling is itself useful information — even on the
worst-behaved bench case, the classifier's `diverged` runs are
short bursts, not multi-second plateaus. Whatever fix this case
needs, it is not "the filter sits in `diverged` for seconds" — the
classifier sees the filter recovering, then re-diverging.

## Results

Trace metrics (24–36 s, far-active frames):

| metric | baseline | P3h (1 reset @ 26.95 s) | Δ |
|---|---:|---:|---:|
| ERLE_inst median (dB) | +0.29 | +2.15 | **+1.86** |
| ERLE_win median (dB) | −1.13 | +3.62 | **+4.75** |
| main_err_ratio median | 0.94 | 0.61 | −0.33 |
| p3f_shadow_advantage median | 3.36 | 2.00 | −1.36 |
| state `diverged` frames | 53 | 0 | −53 |
| state `refined_usable` frames | 80 | 187 | +107 |
| state `suspicious_dt` frames | 518 | 139 | −379 |

Internal ERLE / state-recovery metrics show the reset doing exactly
what it is designed to: discard the corrupt taps, re-converge to
`refined_usable`, and the classifier no longer reports `diverged`
or sustained `suspicious_dt`.

AECMOS single-case score (full 36.7 s clip, `talk_type='dt'`):

| metric | baseline | P3h | Δ |
|---|---:|---:|---:|
| echo | 3.491 | 3.502 | +0.011 |
| deg  | 3.226 | 3.197 | **−0.029** |

(Note: these single-case scores are not directly comparable to the
800-case bench means used elsewhere — bench uses a fixed 20-s slice
+ scenario-specific averaging — but they are internally consistent
between the two columns here.)

## Findings

### 1. Internal recovery is real; AECMOS does not reward it

The reset clears the post-NE-contamination corrupt taps, the filter
re-converges on the next far-active block, and the classifier sees
`refined_usable` for 2.3× as many frames in the back half. Yet the
single-case deg score drops by 0.029. This is the same pattern as
P3e / P3f / P3g: the action is internally well-behaved, but the
AECMOS-relevant audio characteristics it produces are no better
than the prior path.

### 2. Why the reset doesn't help: re-learning happens *during* NE

Reset fires at t = 26.95 s; the trace ends at 36.7 s, so the
post-reset window is ≈ 10 s, all of it inside the case's NE-active
region (the same region whose contamination triggered the original
reset). The freshly reset filter then re-learns under NE, ending up
in roughly the same place a few seconds later. The reset replaces
"old corrupt taps + render-fallback de-trusting them" with "fresh
taps re-learning under NE + shorter render protection window";
AECMOS-relevant net change is approximately zero.

### 3. The P3 arc asymptote claim is reinforced

P3e/f/g closed with:

> The architecture (taps ⊕ shadow ⊕ render-fallback) has already
> organised itself near its AECMOS asymptote on 7GT.

P3h adds: even discarding the taps entirely and starting over at the
worst point in the trace doesn't move the AECMOS score in the
direction we want. The asymptote claim now has four independent
negative experiments behind it (V1/V2/V3 mu reduction, P3f state
mu reduction, P3g RES dry-run, P3h reset).

## Decision

**Closed.** Per user stop gate ("AECMOS direction is the deciding
signal"), the deg regression of −0.029 means P3h does not earn
its 800-case risk bench. Toggle stays default off.

## What to keep / what to discard

**Keep:**

- `AecConfig.diverged_reset_enabled` /
  `diverged_reset_streak_frames` / `diverged_reset_cooldown_frames`
  (default off — present for future revisit if a different
  classifier or different action set is tried).
- `_diag` fields `p3h_reset_fired`, `p3h_reset_count`,
  `p3h_reset_cooldown`, in `--trace-aec-state` CSV. Useful for
  future state-machine experiments to count action firings.
- `--diverged-reset[-streak|-cooldown]` flags on `run_one_case.py`
  for ad-hoc reproduction.

**Don't repeat:**

- Whole-filter reset gated only on classifier `diverged` state.
  Internal recovery doesn't translate into AECMOS gain on this
  case. If a future P3 attempts another reset variant, it must
  trigger *before* the contamination starts (require a leading
  indicator, not the post-fact `diverged` symptom), and it must
  ensure the post-reset learning window doesn't run into more NE.

## Files

- `python/aec.py`:
  - `AecConfig.diverged_reset_*` (3 fields)
  - `AEC._p3h_reset_cooldown_remaining`, `_p3h_reset_count`
  - reset action ≈ line 5377
- `python/run_one_case.py`: 3 CLI flags, 2 CSV columns
- Trace runs: `/tmp/p3h_baseline_7gt.{wav,csv}`,
  `/tmp/p3h_on_7gt.{wav,csv}`
- Compare script: `/tmp/p3h_compare.py`
- AECMOS scorer: `/tmp/p3h_aecmos.py`
