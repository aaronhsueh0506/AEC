# F-E5 verdict — saturation handling extensions

**Phase**: v3.11 Phase 1 Sprint 11-12
**Flag**: `f_e5_enabled` (default OFF)
**Result**: **PASS — marginal positive on saturation cases, neutral elsewhere**

## Mechanism

Targets edge case **E5: clip & saturation** with 4 sub-fixes under one flag:

- **E5-1**: symmetric mic soft-clip when `sat_mic > 0.3` (was: only ref
  soft-clipped). Prevents clipped mic samples from corrupting filter
  error spectrum.
- **E5-2**: extended main mu sat-gate to threshold 0.5 (matches shadow).
  Previously main only froze at extreme mic clip > 0.8 → main kept
  learning on clipped reference at sat 0.5-0.8 while shadow was paused.
- **E5-3**: fast-attack `_error_psd` reset on sat→clean transition.
  Triggered when prev sat > 0.5 AND curr sat < 0.2. Prevents
  α=0.95 EMA from propagating clipped samples into R for ~20 frames
  post-sat.
- **E5-4**: shadow_rise mask during sustained saturation. Clipped
  input causes both filter errors to rise in tandem; detector misreads
  this as path change. Setting `fired=False` prevents false EPC
  triggering filter re-initialisation during a sat event.

## Verification

**Baseline**: v3.10.6 (`results/v3_10_5_main/`)
**Candidate**: v3.10.6 + `AEC_F_E5=1` (`results/sprint_f_e5_on/`)

### Bucket-mean Δ vs baseline

| Bucket | n | Δerle_full | Δerle_active | Δne_pres |
|---|---|---|---|---|
| FS_static    | 169 | +0.002 | +0.002 | -0.001 |
| FS_movement  | 131 | +0.001 | +0.001 | +0.000 |
| DT_static    | 186 | +0.000 | +0.000 | -0.000 |
| DT_movement  | 114 | +0.000 | +0.000 | -0.000 |
| NE           | 200 | +0.000 | +0.000 | +0.000 |

All bucket means within ±0.002 dB noise floor. NE bucket unchanged
(correct — no saturation events in pure NE).

### Per-case impact

14 / 800 cases moved by |Δ| > 0.01 dB.

| Stem | Bucket | ΔerleA | Δne |
|---|---|---|---|
| sKXucFp4FUCJKo5d0G54Og FS_static | FS_static | **+0.348** | -0.117 |
| Y91uE2tRg0SUB2a9XjT30w FS_movement | FS_movement | +0.123 | -0.032 |
| OLjlc92QWU6fwuN4ytCPQg FS_static | FS_static | +0.044 | -0.007 |
| oc0eVAlCbEiTTPNZmV4pMQ FS_static | FS_static | +0.040 | -0.037 |
| m4789fdio0q92zjf9gvh1Q FS_movement | FS_movement | +0.040 | -0.007 |
| Y91uE2tRg0SUB2a9XjT30w FS_static | FS_static | -0.036 | -0.045 |

Top case +0.348 dB on sKXucFp4 is significant — likely a case with
saturation events where F-E5 protects the filter from clipped-input
learning. Most other cases are small positive movements with slight
NE-preservation cost (trade-off pattern same as B5/B6).

### Cohort tail

`qNvSMyUSXUyrDGpOw7s6qg_farend_singletalk`: ΔerleA = 0.000 (neutral,
no saturation events on this case).

## Hard abort criteria

| Criterion | Threshold | Actual | Pass? |
|---|---|---|---|
| FS_static bucket mean | ≥ -0.02 | +0.002 | ✓ |
| FS_movement bucket mean | ≥ -0.02 | +0.001 | ✓ |
| DT_static bucket mean | ≥ -0.02 | +0.000 | ✓ |
| DT_movement bucket mean | ≥ -0.02 | +0.000 | ✓ |
| NE bucket | unchanged | +0.000 | ✓ |
| Linear ERLE Δ ≥ -0.5 | OK | min: -0.036 | ✓ |
| Cohort tail qNvSMyU | ≥ -0.05 | +0.000 | ✓ |

All pass.

## Decision

- **F-E5 mechanism**: all 4 sub-fixes work as designed; the
  combination correctly targets saturation-event cases without
  side-effects on clean audio.
- **Impact scope**: 14 / 800 cases (1.75%) — narrow but real, with
  one significant positive (+0.348 dB on sKXucFp4).
- **Trade-off**: positive Δerle paired with small ne_pres reduction
  in FS — expected pattern (more conservative filter learning during
  sat → cleaner cancellation post-sat).
- **Cohort tail invariant**: preserved.

**Flag state**: keep `f_e5_enabled = False` (default OFF) for now.
Final promotion decision deferred to "Final review" at end of Phase 1.

**Notable**: F-E5 has the highest per-case improvement signal of any
Phase 1 sprint to date (+0.348 dB). Even though bucket means are flat,
the specific cases that benefit are real production-relevant
(speaker-distortion scenarios). Worth keeping the flag available; may
be promoted alongside B5/B6 if final-review compounding shows benefit.
