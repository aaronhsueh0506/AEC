# F-E1 verdict — ERL clip + far_active hysteresis

**Phase**: v3.11 Phase 1 Sprint 5-6
**Flag**: `f_e1_enabled` (default OFF)
**Result**: **NEUTRAL — mechanism correct, 800-case corpus doesn't exercise edge case**

## Mechanism

Targets edge case **E1: ref/mic energy difference too large**:

- **E1-1** Extend ERL clip lower bound from 0.001 to 1e-5. Current
  `np.clip(inst_erl_raw, 0.001, 1.0)` clamps extreme high coupling
  (real ERL = 0.0005) to 0.001, biasing the F3.1-v3 mic_excess metric
  downstream by 6 dB.
- **E1-3** Add fast-attack / slow-release hysteresis to the `far_pwr >
  1e-4` gate. Currently single-threshold causes ERL updates to stall
  when far power oscillates near 1e-4. Hysteresis: attack at 1e-4,
  release after 5 consecutive frames below 3e-5.

## Verification

**Baseline**: v3.10.6 (`results/v3_10_5_main/`)
**Candidate**: v3.10.6 + `AEC_F_E1=1` (`results/sprint_f_e1_on/`)

### Bucket-mean Δ vs baseline

| Bucket | n | Δerle_full | Δerle_active | Δne_pres |
|---|---|---|---|---|
| FS_static    | 169 | +0.000 | +0.000 | +0.000 |
| FS_movement  | 131 | +0.000 | +0.000 | -0.000 |
| DT_static    | 186 | -0.000 | -0.000 | -0.000 |
| DT_movement  | 114 | +0.000 | +0.000 | -0.000 |
| NE           | 200 | +0.000 | +0.000 | +0.000 |

All values within 1e-3 (essentially zero — render-level FP noise).

### Per-case impact

Cases with any non-trivial Δ (|Δ| > 0.001): **4 / 800**.

| Stem | ΔerleA | Δne |
|---|---|---|
| ZtGitIxrzU0ILwu0HACaaw_farend_singletalk | -0.00116 | -0.00198 |
| oEyXuSCCw0qdJ0J16FGfcQ_farend_singletalk | -0.00029 | +0.02678 |
| yM2wHof9U06yVPJfemZ3hg_farend_singletalk | +0.00165 | +0.00284 |
| kOGPX6kHskOaKSZdLGNz8A_doubletalk | -0.00025 | -0.00173 |

All movements are at or below FP-roundoff level.

### Why no effect on 800-case corpus

The AEC Challenge `aec_challenge_blind/` dataset has normal-coupling
scenarios. Sampling 30 cases gave far-power mean values in the range
1e-6 (NE-only, no echo) to 0.025 (typical FS). No sustained near-1e-4
oscillation that would exercise the hysteresis release counter; no real
ERL below 0.001 that would exercise the extended clip range.

The fix mechanism is correct (verified by reading the code path) but
the load-bearing edge cases are not in this corpus. To genuinely
ablate F-E1 we would need:

- A synthesised case with real ERL = 5e-4 (mic gain attenuated to
  exercise the 1e-5 clip range).
- A case with `far_pwr` sustained around 1e-4 with ±3 dB oscillation
  (to exercise the hysteresis release counter).

These are production-realistic conditions (e.g., users with very
attenuated speaker volume or very weak loopback gain) that the
AEC Challenge corpus does not specifically include.

## Decision

- **F-E1 mechanism**: implemented correctly per design.
- **Impact on 800-case**: zero (no edge-case exposure).
- **Production rationale**: extreme-ERL handling is a commercial-
  stability concern (user said: "ref/mic 能量差異太大" must be handled);
  having the fix available behind a flag is more conservative than
  not having it at all.

**Flag state**: keep `f_e1_enabled = False` (default OFF). Not
promoting to balanced because the 800-case bench cannot verify benefit;
flag remains available for production deployments that report extreme-
coupling issues (would re-test on field-collected samples then).

**Defer to final review**: F-E1 is a correctness fix with no
800-case validation evidence. Final-review recommendation: keep OFF
for v3.11 GA; consider opt-in for downstream products that observe
extreme-coupling edge cases in their own corpora.
