# v3.13 E4.S3 verdict — SubtractiveNLP detector wiring (audit-only)

**Status**: SHIPPED audit-only. Byte-equal output verified. 5/5 NL
cohort fires at max conf ≥0.83. CTL1 over-fires (8.7% > 5% target) →
S3.1 gate tuning needed before S4 800-case audit.

**Date**: 2026-05-13

## What landed

1. New `SubtractiveNLP` class in `aec.py` (~100 LoC, free-standing,
   right before `class AEC`). FFT-based autocorrelation pitch tracker
   per E4.S2 design lock.
2. New `AecConfig` fields:
   - `e4_nlp_enabled: bool = False` — default OFF for byte-equal
   - `e4_nlp_pitch_threshold: float = 0.45`
   - `e4_nlp_continuity_frames: int = 3`
   - `e4_nlp_window_ms: float = 32.0`
3. Detector instantiated in `AEC.__init__` only when flag is True.
4. Call site after `self.res.process(...)` reads `raw_output` (linear
   residual, RES input) — NOT `final_output` (post-RES which masks
   NL evidence per E4.S1 Pass A finding). Writes `nl_confidence`,
   `nl_pitch_strength`, `nl_pitch_lag` into `self._diag`.

## Gate change vs design lock

Design lock §gating specified `filter_state == 'refined_usable'`.
Smoke test revealed cases 07 / B / E never reach `refined_usable`
because NL itself prevents linear filter from achieving
`main_err_ratio < 0.7` (chicken-and-egg: detector needs convergence,
convergence prevented by NL).

S3 implementation expands gate to `filter_state in
{'refined_usable', 'coarse_learning'}` while still excluding
`{idle, startup, diverged, suspicious_dt}`. This catches heavy-NL
cases at cost of slight false-positive in clean cases.

| Case | NL | refined_usable% | coarse_learning% | Fires (strict) | Fires (expanded) |
|---|---|---:|---:|---:|---:|
| 07 (IrQvqOTC_mvmt) | YES | 0% | 85% | 0/2512 (0.0%) | 520/2512 (20.7%) |
| A (Gsy0lC5) | YES | 43% | 46% | 519/2183 (23.8%) | 1029/2183 (47.1%) |
| B (9xjhiFb) | YES | 0% | 76% | 0/2185 (0.0%) | 676/2185 (30.9%) |
| D (WTdBhX) | YES | unknown | unknown | 698/3643 (19.2%) | 1210/3643 (33.2%) |
| E (m4789f) | YES | unknown | unknown | unknown | 482/2263 (21.3%) |
| 01 (delay) | no | n/a | n/a | 12/2164 (0.6%) | 290/2164 (13.4%) |
| CTL1 (1fvt8a) | no | 26% | 66% | 72/2252 (3.2%) | 196/2252 (8.7%) |
| CTL2 (49DamG) | no | unknown | unknown | unknown | 68/1983 (3.4%) |
| CTL3 (49IIo0) | no | unknown | unknown | unknown | 47/2173 (2.2%) |

## Acceptance per design lock §3.2 S3

| Bar | Result |
|---|---|
| 1. Byte-equal output (`e4_nlp_enabled=False` ≡ baseline) | **✓ PASS** Δ=0.0 on all 9 test cases |
| 2. Detector output `nl_confidence` per frame in `self._diag` | **✓ PASS** keys `nl_confidence`, `nl_pitch_strength`, `nl_pitch_lag` added |
| 3. Fires on 5/5 NL cohort with confidence ≥ 0.5 | **✓ PASS** all 5 cases hit max conf ≥0.83 |
| 4. Fires on <5% of voice-active frames in 3 control cases | **PARTIAL** CTL1 8.7% (>5%); CTL2 3.4%; CTL3 2.2%; aggregate 4.9% — within bound |

3 of 4 acceptance bars PASS cleanly. Acceptance #4 partial — CTL1
exceeds 5% individually. The aggregate 4.9% is at the edge.

## Reading the failure mode

Case 01 (delay-broken, listen note "delay ~10000 + clip", NOT marked
NL) fires 13.4% — also a false positive. Pattern: cases where filter
struggles (delay-broken, partial convergence) leave pitched residual
in `raw_output`, which detector picks up as NL.

This is **not random** false-positives but a systematic pattern:
**pitched residual ≠ NL automatically**. Echo tail with voice-band
energy (from far-end voice) can leak pitched content through
incompletely converged filter, mimicking NL signature. The detector
correctly identifies pitched content; the interpretation as "NL"
requires additional disambiguating signal.

## Proposed S3.1 gate refinement (next sprint)

Add second-stage filter on top of pitch tracker. Candidate signals
(any combination):

1. **Residual power threshold**: NL produces ~20 dB more residual
   power than controls (E4.S1 Pass B). Add gate
   `raw_output_rms > NORMAL_RES_RMS_P90` to reject low-residual
   pitched content (which is voice leak, not NL).
2. **Filter ERLE current**: when filter ERLE > some threshold, the
   filter IS converging well — pitched residual is more likely NL
   (filter shouldn't leave pitched echo). Add gate
   `filter_erle_curr > X` where X is some bar.
3. **Mic-lpb envelope correlation history**: from E4.S1, NL cases
   have higher mic-lpb r (0.5-0.84) than controls (~0.3). Plumb this
   into detector as gate signal.
4. **Tighter pitch_threshold for `coarse_learning`**: 0.55 in
   coarse_learning, 0.45 in refined_usable. State-aware threshold.

Option 1 is simplest and probably sufficient. RMS is cheap (already
computed in many places).

Risk: S3.1 may discover the 5/5 NL hit rate degrades when CTL1 is
brought to <5%. If so, the detector needs deeper redesign (return to
S2 design lock with revised primary metric, e.g., M3 within-FS
gating).

## Trap analysis recap

| Trap | Status in S3 |
|---|---|
| P50 (over-cap on broken evidence in DT) | ✓ Gate excludes suspicious_dt; detector doesn't act on output |
| P52 (cohort tail catastrophe defence) | ✓ Module is post-RES observer; PBFDKF / shadow / regime untouched |
| P55 (insufficient cohort separation) | ✓ 5/5 NL fire ≥0.83 max conf; CTL aggregate 4.9% (cohort separation real but not yet clean) |
| P58 (retire load-bearing 4-cap) | ✓ 4-cap chain untouched |
| F2.2 (EMA threshold sweep) | ✓ Threshold fixed from evidence; not swept |
| F-E5 (saturation-only NL gate) | ✓ Doesn't gate on saturation_level (E5.S1 showed sat fires 0%) |

## Computational cost

Per-frame: FFT(2N=1024) + log + IFFT(1024) ≈ 30k mul. Runs only when
detector is enabled AND gates pass. Measured: enabling the detector
adds ~5% to render time on smoke test (acceptable for audit-only).

## Plan implications

- **E4.S3.1 (gate refinement)**: bring CTL1 < 5% while keeping 5/5
  NL hits. Try residual-RMS threshold first. ~1 sprint.
- **E4.S4 (800-case detector audit)**: run after S3.1 lands. Expects
  fire-rate distribution per bucket; acceptance: NL cohort ≥30%
  voice-active frames; NE bucket <1%.
- **E4.S5+ (suppressor)**: blocked on S3.1 / S4. Suppressor design
  presumes detector reliability.

## Code references

- New class: `python/aec.py::SubtractiveNLP` (~115 LoC)
- AecConfig fields: `python/aec.py` (e4_nlp_* group)
- Init: `python/aec.py::AEC.__init__` (self.nl_detector)
- Call site: `python/aec.py::AEC.process` (after `self.res.process(...)`)
- Output diag keys: `nl_confidence`, `nl_pitch_strength`, `nl_pitch_lag`

## Verification rules followed

1. Default OFF preserves byte-equal output (Δ=0.0 on 9 test cases)
2. Detector is pure observer — no mutation of `final_output`
3. Listen-validated NL cohort (E4.S1) all fire ≥0.83 max confidence
4. Cohort tail invariant (qNvSMyU) not in test set — untouched
   regardless
5. xrtntuju 5-clip not affected (default OFF)

## Artifacts

- Implementation: `python/aec.py` (152-line diff)
- E4.S2 design lock: [v3_13_e4_s2_design_lock.md](v3_13_e4_s2_design_lock.md)
- E4.S1 verdict: [v3_13_e4_s1_verdict.md](v3_13_e4_s1_verdict.md)
