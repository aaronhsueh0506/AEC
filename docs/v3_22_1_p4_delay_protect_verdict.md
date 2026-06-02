# v3.22.1 — P4 delay-acquire phantom-mislock guard — SHIP (default ON)

**Date**: 2026-06-02
**Config flag**: `delay_acquire_protect_converged` (default **True** as of 3.22.1)
**Env override** (for A/B): `AEC_DELAY_PROTECT=0|1`
**Branch**: `v3_22`

## Mechanism

Path-A is the first delay acquisition (`_current_delay < 0` → a solid estimate).
It is a heavy action: it resets the filter taps + all filter-output-derived
state and applies the estimated shift. The guard rejects this acquisition when
the linear filter is **already cancelling** (`erle_windowed > 2.5 dB`) at the
current alignment:

```python
_already_cancelling = (
    bool(getattr(self.config, 'delay_acquire_protect_converged', False))
    and float(self._diag.get('erle_windowed', 0.0)) > 2.5)  # dB
if (_delay_eligible and self._current_delay < 0
        and self.delay_est.is_solid and not _already_cancelling):
    self._current_delay = new_delay
    self._reset_filter_derived_state(reason='delay_first', ...)
```

Windowed ERLE (TC ≈ 10 s) is the chosen signal: it decays, so a *real* path
change that stops cancellation lifts the guard and re-acquisition proceeds —
unlike a cumulative ERLE. Instant-ERLE EMA is too noisy (dips to ~1 dB between
far-active bursts); the 10-frame >5 dB latch is too strict (breaks on those
dips).

## Root cause (bench double-alignment)

`eval_aec_challenge.py` pre-aligns the reference with full-signal GCC-PHAT and
*then* the in-pipeline AEC3 matched filter (`EchoPathDelayEstimator`,
decimate-by-4, 608 ms coverage) runs on top. On weak/nonlinear-echo cases the
matched-filter correlation surface is flat, so it latches a **noise peak** at
confidence 1.0 (REFINED) — e.g. kZogUfYc +96 ms, 9xjhi +188 ms, JjCzlhn3 +96 ms
— double-shifting the already-aligned reference. The filter is reset against the
mis-shifted reference and ERLE collapses to ~0 for the rest of the case.

**Production is unaffected**: with raw reference (no GCC pre-align) the matched
filter locks the true 16–32 ms delay. Verified on JjCzlhn3: production raw-ref
mode reaches 5.85 dB far-active ERLE vs the bench's 2.3 dB. The guard never
blocks a legitimate cold acquisition in production (cold filter ERLE ≈ 0 < 2.5),
so it is harmless there — the win is a bench-measurement-artifact recovery.

## 800-case result (vs `splitcfg` baseline; preset=balanced/fl=832/cng/-j)

- **26 cases changed**, net **Σecho +2.85 / Σdeg +0.28** (mean +0.110 / +0.011).
- Wins: kZogUfYc **+2.06**, Xv7jH2 **+1.51** (FS_movement echo); waxU01 **+0.236**,
  w0QrMw +0.193, QkRkww +0.172 (DT_movement deg).
- Bucket-level (small, by design — 26/800 diluted): FS_movement echo
  3.480 → **3.502** (+0.022); everything else ~flat.

## Casualties are AECMOS movement-quirks, not regressions (audio-verified)

- **W0zK3dv0** (DT_movement deg −0.32): divergence is confined to far-active
  sec 7–11 + 27; **near-only frames (sec 12–26) are byte-identical** OFF vs ON;
  coherence(mic, far)=0.58–0.65 in sec 7–10 → **echo-dominant**. The fix removes
  echo (ON suppresses to 0.003 where OFF leaks 0.02–0.10 because the reset
  de-converged the filter), not near-end speech.
- **JtodX, wlAXM0iD** (FS_movement echo, deg pinned at 5.0): overall ON output is
  *quieter* (less total echo energy); AECMOS scores the temporal redistribution
  lower. Same quirk class.

Conclusion: AECMOS echo-score is **non-monotonic with residual echo energy on
movement cases**; the per-case casualties are scoring artifacts, so the fix is
*better* than its aggregate AECMOS numbers suggest.

## Verdict

SHIP default-ON. Zero production risk (harmless without bench pre-align), fixes a
real phantom-mislock bug, net-positive 800-case, no audio-verified regression.

## Not closed by this fix

The guard's `erle_windowed > 2.5` gate misses phantom mislocks where the filter
had not yet converged when the spurious acquisition fires (weak early far, e.g.
JjCzlhn3). These remain bench-artifact-depressed but are correct in production. A
fuller fix would teach the in-pipeline delay estimator not to double-count the
bench pre-align (reject large jumps when the reference is already aligned) — a
bench-harness change, deferred. The aggregate FS echo gap to AEC3 is **not**
explained by this artifact (24-case raw-vs-bench scan: median +0.1 dB, net wash).
