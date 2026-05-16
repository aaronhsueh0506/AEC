# v3.17 Phase B.1 — Movement-rate DelayEst CLOSED CANNOT SHIP (2026-05-15)

**Branch**: `feature/v3.17` HEAD `9ece25d`.
**Sprint**: Phase B.1 (per [`docs/v3_17_plan.md`](v3_17_plan.md) §B.1).
**Verdict**: CLOSE per §0.4 — target case regressed, 2 of 11 affected
movement cases improved, 3 of 11 regressed (net negative).
**Substrate retained**: `mov_rate_delay_est_enabled` flag (default OFF) +
`delay_est_period_s_fast` (0.25) + `delay_est_alpha_fast` (0.2) — kept
for v3.18+ retry with a different motion-detector / EMA-management design.

---

## 1. Mechanism shipped (substrate)

When EPC is active or in hangover, override DelayEstimator state:
- `_period_samples`: `delay_est_period_s` (0.5 s) → `delay_est_period_s_fast` (0.25 s)
- `_alpha`: 0.6 → `delay_est_alpha_fast` (0.2)

Restore baseline when EPC quiet. EPC chosen as motion proxy because it
already fires on echo path change events that drive `force_delay()` chain.

Wiring location: [aec.py:6505-6535](../python/aec.py#L6505) (EPC-gated
period+alpha override before `delay_est.accumulate()`).

## 2. Verification

### 2.1 Default-OFF byte-equal sanity (PASS)
- 60-case subset render with `mov_rate_delay_est_enabled=False`:
  120/120 wav files byte-identical vs v3.16.0 (`/tmp/v3_16_a_off/`)
- Default-OFF code path is dead (write only fires when flag ON)

### 2.2 Lever-fires audit (PASS)
Per-frame instrumentation on case `0I0XMl3M_FS_movement` (target):
- Frames in fast cadence (EPC active+hangover): 1657/2416 = 68.6%
- Frames in baseline cadence: 759/2416
- Estimates fired: 76 (vs ~52 expected at baseline 0.5s rate)
- Lever fires; mechanism is not no-op

### 2.3 60-case AECMOS bench (FAIL — closure trigger)

Bucket means Δ ON vs OFF:

| Bucket | Δecho | Δdeg |
|---|---:|---:|
| FS_static | +0.000 | +0.000 |
| FS_movement | **−0.014** | +0.000 |
| DT_static | +0.000 | +0.000 |
| DT_movement | −0.000 | −0.002 |
| NE | +0.000 | +0.000 |

Per-case Δ on the 11 movement cases that diverged:

| Case | Δecho | Δdeg | verdict |
|---|---:|---:|---|
| **0I0XMl3M FS_movement** (TARGET) | **−0.144** | +0.000 | ✗ TARGET REGRESSED |
| IqtJR4tjJkWrwUjYorz0Og FS_movement | +0.074 | −0.000 | ✓ improved |
| OXCtw0FVhUWGUBil6Uucdw DT_movement | −0.058 | −0.025 | ✗ DT_movement Δdeg fail |
| W0zK3dv0QE2YckPArTGXCg DT_movement | +0.057 | +0.003 | ✓ improved |
| afHuFvflAkaH7Pr85kheUQ FS_movement | −0.081 | +0.000 | ✗ FS regressed |
| (6 other cases) | ±0.005 | ±0.005 | ≈ no change |

Net: 2 improvements, 3 regressions among significant cases. Target case
0I0XMl3M ironically got WORSE.

## 3. Root cause

Faster EMA (alpha 0.6 → 0.2) tracks echo-path changes faster — but
also tracks NOISE faster. EPC-window cross-spectrum is exactly when
the signal is most non-stationary; reducing alpha makes the EMA jitter
between candidate peaks rather than converge to the new peak.

For the target case 0I0XMl3M, the C6 audit showed estimated_delay
jumps 1230 → 4132 → 4369 → 2 across multiple EPC events. Each new EMA
estimate (faster) latches onto a DIFFERENT lag candidate during the
turbulence window, then triggers `Path B delay shift` → 32-sample
threshold check → `_reset_filter_derived_state(reason='delay_shift')`
→ filter re-converges from scratch. More frequent re-convergences
during the same movement window = more catastrophic ERLE.

Polling alone (commit `ab8816b`) couldn't help because the same EMA
queried twice as often gives the same answer. Alpha reduction
(commit `9ece25d`) makes the EMA jitter, which is even worse.

## 4. What would actually fix this (deferred to v3.18+)

Per the v3.16 C6 audit doc warning:
> "no clean tactical fix without major DelayEst rewrite (faster
> update rate may help but introduces noise at low-confidence frames)"

Fundamental fixes require:
- **Confidence-gated EMA reset**: only reset cross-spec when PAR is
  HIGH (not noise). Requires confidence tracking through EPC window.
- **Fractional-sample interpolation**: even with stable cross-spec,
  the fl=832 (52 ms) coverage on movement makes integer-sample lag
  estimates poorly aligned. Needs sub-sample work.
- **Multi-hypothesis tracking**: keep top-K candidate lags during EPC,
  defer commit until one wins decisively. Significant code complexity.

None fit a 2-3 sprint v3.17 window. Defer to v3.18.

## 5. Per §0.4 verdict

> "If lever moves bucket-mean < 0.002 dB AND fires < 5% of frames
> after 3 A/B sweeps, ship as substrate for dependent arc"

Lever fires 68% of frames (well above 5%). Bucket-mean moves
−0.014 (FS_movement) — above 0.002 dB threshold. But MAGNITUDE is
matched by per-case variance (±0.144), not a robust improvement.
Combined with target-case regression: this is a "high lever / high
variance / wrong sign" outcome — substrate is technically valid but
NOT useful.

**Action**: substrate retained (3 config flags), commit as default-OFF.
Mechanism family permanently retired through v3.17. v3.18+ DelayEst
work would need a fundamentally different mechanism (confidence-gated
reset, sub-sample, multi-hypothesis).

## 6. Cohort tail regression check (per §0.5)

`qNvSMyU` (cohort tail): byte-identical between ON and OFF (cohort tail
case has no fast movement, EPC fires only briefly). No regression on
load-bearing cohort tail defence.

## 7. Substrate commits

- `ab8816b` — period override wiring (no-op on output)
- `9ece25d` — alpha override added (output diverges, but wrong sign)
- this doc — closure verdict

## 8. Next: Phase B.3

Per v3.17 plan ordering, B.3 (C2.v2 narrow per-state ENR) is next.
B.3 targets DT-NE compression debt (different mechanism family from
B.1's filter-side movement); B.1 closure does NOT block B.3.
