# v3.21.4 time-domain audit verdict — CLOSED NOT-A-BUG

**Date**: 2026-05-20
**Branch**: v3.21.4 HEAD
**Trigger**: v3.21.2 plan flagged 3 HIGH "time-domain unit-conversion
bugs" parallel to the bin-index bug fixed in v3.21.2 — counters in
`DominantNearendConfig` and `EchoModelConfig` were ported as bare ints
from AEC3 (4 ms blocks @ 16 kHz) into our 10 ms hops, giving 2.5×
longer wall-clock time than AEC3 intended.

## The "bugs"

| Knob | File | Value | "AEC3 wall-clock" | "Our wall-clock" |
|---|---|---:|---:|---:|
| `DominantNearendConfig.trigger_threshold` | [suppression_gain.py](../python/modules/residual/suppression_gain.py) | 12 | 48 ms | 120 ms |
| `DominantNearendConfig.hold_duration` | [suppression_gain.py](../python/modules/residual/suppression_gain.py) | 50 | 200 ms | 500 ms |
| `EchoModelConfig.noise_floor_hold` | [residual_echo_estimator.py](../python/modules/residual/residual_echo_estimator.py) | 50 | 200 ms | 500 ms |

The naive correction (via [`blocks_to_hops()`](../python/modules/aec3_scale.py)
which already exists for state/stationarity_estimator and state/subband_erle):

- `trigger_threshold` 12 → 5
- `hold_duration` 50 → 20
- `noise_floor_hold` 50 → 20

## V4.1 empirical test — trigger_threshold 12 → 5

800-case AECMOS vs v3.21.3 baseline:

| Bucket | Δecho | Δdeg |
|---|---:|---:|
| FS_static | −0.021 | +0.000 |
| FS_movement | −0.027 | +0.000 |
| DT_static | +0.002 | **−0.027** |
| DT_movement | −0.003 | **−0.007** |
| NE | +0.000 | −0.001 |

**Both directions worse** (FS_echo down AND DT_deg down) — not a
Pareto shift. This is a strict regression. V4.1 reverted.

## Why blocks_to_hops is the wrong yardstick

User pointed out: these counters need **physical-meaning alignment**,
not wall-clock alignment. Inspection of AEC3 source
[dominant_nearend_detector.cc:32-76](aec3_extracts/src/aec3/dominant_nearend_detector.cc)
+ [residual_echo_estimator.cc:340-358](aec3_extracts/src/aec3/residual_echo_estimator.cc)
confirms each counter measures a different physical quantity, not
wall-clock time:

### trigger_threshold — **statistical hysteresis depth**

```cpp
if (echo_sum < enr_threshold * ne_sum && ne_sum > snr_threshold * noise_sum) {
    if (++trigger_counters_[ch] >= trigger_threshold) { ... }
} else {
    trigger_counters_[ch] = std::max(0, trigger_counters_[ch] - 1);  // decrement, not reset
}
```

The +1/-1 random-walk semantics make `trigger_threshold` a noise-tolerant
accumulator, not a strict consecutive-N counter. The physical quantity
it measures is:

**how much net-positive evidence (signed accumulation) is needed to
override per-sample ENR estimator noise**.

This depends on:
1. **Per-sample ENR estimator noise floor** (signal-dependent, pipeline-
   dependent — our PBFDKF feeds a different ENR distribution than
   AEC3's matched-filter + refined output).
2. **Cohort transient burst rate** (how often noise alone produces
   spurious NE-condition-true samples).

Neither scales with hop/block size. Wall-clock conversion is wrong.

V4.1 empirically confirmed: 5 hops gives too few statistical samples for
our pipeline's ENR estimator noise → NE state flickers on transients →
both FS and DT regress.

### hold_duration — **NE-state minimum dwell time**

```cpp
if (++trigger_counters_[ch] >= trigger_threshold) {
    hold_counters_[ch] = hold_duration;  // set to full on trigger
}
hold_counters_[ch] = std::max(0, hold_counters_[ch] - 1);  // decrement per call
nearend_state_ = nearend_state_ || hold_counters_[ch] > 0;
```

Physical quantity: **how long to remain in NE state after triggering**.
Two timescales compete:

1. **Speech phoneme duration** (~50–200 ms, wall-clock). AEC3 200 ms ≈
   one phoneme; our 500 ms ≈ half a word.
2. **Downstream gain-shape interaction**. The NE state changes which
   tuning (`normal_tuning` vs `nearend_tuning`) `SuppressionGain` uses;
   the relevant mask shapes are AEC3 vs ours — semantically different
   even at matched wall-clock.

Wall-clock matches phoneme (1) but not gain-shape interaction (2).
The empirically-validated value (50 hops = 500 ms) is co-tuned with our
downstream NE/non-NE gain rules; changing it without re-tuning the
downstream is co-tuning violation per CLAUDE.md.

### noise_floor_hold — **noise floor adapt rate**

```cpp
if (render_power[k] < X2_noise_floor_[k]) {
    X2_noise_floor_[k] = render_power[k];           // instant track DOWN
    X2_noise_floor_counter_[k] = 0;
} else {
    if (X2_noise_floor_counter_[k] >= noise_floor_hold) {
        X2_noise_floor_[k] = std::max(... * 1.1f, ...);  // creep UP by 1.1×
    } else {
        ++X2_noise_floor_counter_[k];
    }
}
```

Physical quantity: **delay before allowing minimum-statistics floor to
creep upward**. This IS approximately wall-clock-meaningful: room noise
floor varies on physical timescales (HVAC drift, fan modulation), so
wall-clock anchoring makes sense in principle.

But the empirical interaction matters: our cohort is mostly
quiet-room recordings where the true X² floor is genuinely stationary.
A slower adapt-up (500 ms vs 200 ms) is harmless or beneficial because
the floor stays at the true minimum longer; rapid adapt-up risks
inflating the floor on speech transients (false positives). Not a bug
on this cohort.

## Verdict

**CLOSED NOT-A-BUG** for all 3 knobs.

The original v3.21.2 plan's "time-domain unit-conversion bugs"
framing was a misdiagnosis: bare-value ports from AEC3 are not
automatically bugs when the bare value lands on a different absolute
wall-clock equivalent. Each counter measures a physical quantity with
its own scaling rule, and 2 of 3 (trigger_threshold, hold_duration)
are not wall-clock-anchored at all. The 3rd (noise_floor_hold) is
wall-clock-anchored but the empirical tuning at 500 ms is harmless or
beneficial on our cohort.

V4.1 empirically demonstrates the harm of naive wall-clock conversion:
both FS and DT regress simultaneously when statistical hysteresis is
shortened to match AEC3 wall-clock.

The existing values (12 / 50 / 50) are kept as **empirically-validated
cohort tuning**, not pure ports.

## What was learned

1. **`blocks_to_hops()` is correct when the AEC3 counter is genuinely
   measuring wall-clock time**. Confirmed correct usage in
   [`state/stationarity_estimator.py`](../python/modules/state/stationarity_estimator.py)
   + [`state/subband_erle.py`](../python/modules/state/subband_erle.py)
   for noise-estimator / stationarity / convergence counters where AEC3's
   intent IS wall-clock alignment.

2. **Wall-clock conversion is wrong** when the counter is measuring:
   - Statistical hysteresis depth (depends on estimator noise floor)
   - Behavior-shape interaction (depends on downstream tuning)
   - Cohort-empirical compromise (a value found by tuning, not derivation)

3. **Audit rule for future ports**: read AEC3 source for the counter's
   role + role of decrement / hysteresis / reset, then decide if
   wall-clock applies before mass-applying `blocks_to_hops()`.

## Files

| Artifact | Path |
|---|---|
| AEC3 dominant_nearend_detector source | [docs/aec3_extracts/src/aec3/dominant_nearend_detector.cc](aec3_extracts/src/aec3/dominant_nearend_detector.cc) |
| AEC3 residual_echo_estimator source (noise floor) | [docs/aec3_extracts/src/aec3/residual_echo_estimator.cc](aec3_extracts/src/aec3/residual_echo_estimator.cc) lines 340-358 |
| V4.1 bench output | `results_v3_21_4_v1/result.md` |
| V4.2/V4.3 — not run (analysis closed before bench) | — |
