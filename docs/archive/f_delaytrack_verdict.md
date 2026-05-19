# F-DelayTrack verdict — continuous delay variance tracking

**Phase**: v3.11 Phase 1 Sprint 7-8
**Flag**: `f_delaytrack_enabled` (default OFF)
**Result**: **NEUTRAL — mechanism correct, 800-case corpus produces identical gate outcomes**

## Mechanism

Replaces `delay_reliable = self.delay_est.confidence >= 0.5` hard cut on
PAR-derived confidence (aec.py:5246, before fix) with variance-based
stability check:

- Track last 8 valid delay estimates in `self._delay_history` (bounded deque)
- Gate condition: `std(history) < 4 samples AND confidence >= 0.3`

PAR (peak-to-average ratio) is noisy during movement — multiple peaks
compete, causing instantaneous confidence to oscillate. Variance over
recent estimates is a more direct stability signal. Reference pattern:
Switchboard AEC3 continuous delay tracking.

## Verification

**Baseline**: v3.10.6 (`results/v3_10_5_main/`)
**Candidate**: v3.10.6 + `AEC_F_DELAYTRACK=1` (`results/sprint_f_delaytrack_on/`)

### Bucket-mean Δ vs baseline

| Bucket | n | Δerle_full | Δerle_active | Δne_pres |
|---|---|---|---|---|
| FS_static    | 169 | +0.000 | +0.000 | +0.000 |
| FS_movement  | 131 | +0.000 | +0.000 | -0.000 |
| DT_static    | 186 | -0.000 | -0.000 | -0.000 |
| DT_movement  | 114 | +0.000 | +0.000 | -0.000 |
| NE           | 200 | +0.000 | +0.000 | +0.000 |

All values within FP roundoff.

### Per-case impact

- Only 1 / 800 cases had |Δ| > 0.01 dB (`oEyXuSCCw0qdJ0J16FGfcQ`, Δne +0.027)
- Movement buckets (FS_movement n=131, DT_movement n=114): **0 cases affected**
- Cohort tail `qNvSMyU`: Δ ≈ 1e-6 (zero)

### Why no effect

Old gate and new gate produce the same outcome on this corpus:

| Scenario | Old gate (conf >= 0.5) | New gate (variance < 4 AND conf >= 0.3) |
|---|---|---|
| Static delay | True (PAR converges high) | True (variance 0, conf high) |
| Movement | False (PAR drops when delay shifts) | False (variance high) |
| Cold start | False (n_updates < 3) | False (history < 3) |
| Stable + mid-conf (0.3-0.5) | False | True ← only divergence point |

The "stable + mid-confidence" window is narrow on this corpus. Real
production scenarios with weak GCC-PHAT signal but consistent delay
estimates would benefit, but the AEC Challenge dataset has clean enough
audio that PAR either converges high or drops decisively.

The target edge case (gradual creep < 32 samples/frame that doesn't
trigger `delay_shift`) requires synthesised drifting-delay signals not
present in `aec_challenge_blind/`.

## Decision

- **Mechanism**: implemented correctly per design.
- **Impact on 800-case**: zero.
- **Production rationale**: gradual delay creep is a real commercial
  concern (movement / breathing-room thermal drift), and variance-based
  tracking is more robust than PAR confidence. Available behind flag for
  field deployment.

**Flag state**: keep `f_delaytrack_enabled = False` (default OFF). Not
promoting to balanced because 800-case bench shows zero benefit.

**Defer to final review**: same recommendation as F-E1 — keep OFF for
v3.11 GA, available for downstream deployments observing drift-delay
edge cases (e.g., long teleconferencing sessions where speaker
position thermal-drifts during the session).
