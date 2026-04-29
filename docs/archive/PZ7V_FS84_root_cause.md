# PZ7V FS#84 +15 dB leak — root cause analysis

## Symptom

PZ7V_farend_singletalk in 800-case blind test:
- OLD baseline: −29.54 dB at 10-11s (clean)
- NEW full-fix: −13.83 dB at 10-11s (+15 dB leak at echo onset t=10.0-10.4s)

## Root cause: latent `raw_dt` delay-mismatch bug

`AEC.process()`:
```python
raw_dt = 1.0 - far_pwr / (mic_pwr + far_pwr)
```

Mic at frame `t` contains echo from `far[t-delay]`, but the formula compares
against `far[t]` (same-time window). When mic energy spikes due to echo
arrival but far is in a quiet sample, the ratio looks like near-end speech.

At PZ7V t=10.04s:
- mic_pwr = 0.0995, far_pwr = 0.0026 → mic/far = 38x
- raw_dt = 1 - 0.026 = 0.974

This wrongly signals "doubletalk" → ResFilter `effective_dt = 0.8` →
ENR gate stops suppressing echo → 15 dB leak.

## OLD vs NEW: identical formula, different masking

| t | NEW raw_dt | OLD (force epc=False) raw_dt |
|---|---|---|
| 9.96s | 0.241 | 0.241 (bit-exact) |
| 10.04s | 0.975 | 0.975 (bit-exact) |
| 10.10s | 0.885 | 0.885 (bit-exact) |

OLD and NEW raw_dt formulas are 100% identical. OLD's protection was
incidental: EPC's "continuously-refreshing hangover" zeroed raw_dt every
frame at echo onset:

```python
# OLD epc_active=True throughout 9.90-10.15s (hangover refreshed to 20 each frame)
if self.epc_active:
    raw_dt = 0.0
```

Without that masking (`OLD with force epc=False`), OLD also leaks +15 dB:
- OLD baseline 10-11s: −29.54 dB
- OLD force epc=False 10-11s: **−14.20 dB** (essentially same as NEW −13.83)

## Why NEW EPC stopped triggering

NEW added guards (B-4a/4b/12, line 3171-3173):

```python
guards_pass = (not self._is_stationary_far
               AND dt_signal < 0.3
               AND sat_safe)
```

Plus `shadow_adv > 1.3` requirement. At PZ7V t=9.96s, dt_indicator already
elevated (0.24, due to the raw_dt bug starting to manifest). dt_signal soon
> 0.3 → guards fail → EPC blocked → raw_dt no longer masked → vicious cycle.

The guards themselves are CORRECT (preventing DT-period EPC misfire). They
just exposed a pre-existing latent bug in raw_dt.

## Bisection journey (for context)

Before identifying root cause, exhaustive bisection ruled out:

PZ7V single-file (all returned identical to full):
- 6 individual `_FIX_*` flag OFF: A, B-3a, B-3b, B-3c, B-7, B-11
- min_kalman (all 6 OFF combined)
- EPC multi-level OFF (binary fallback ALSO has the new guards)
- B-9 shadow saturation gate revert
- B-6 DTD/RES feedback (per_bin_eer source) revert
- Power EMA gate revert (PBFDAF.process)
- `_update_per_frame_state` flow gate revert
- PBFDKF `_update_weights` body OLD-graft (Round 1)
- PBFDKF `_update_per_frame_state` no-op (Round 2)
- AB3 state machine OFF (`fs_hi_erl_state` forced False) + Set B disabled

Reverse graft (NEW PBFDAF+PBFDKF classes injected into OLD aec.py):
- Output −33.10 dB ≈ OLD baseline → **PBFDKF/PBFDAF rewrite is innocent**

This narrowed the suspect to AEC.process / ResFilter unconditional changes,
which led to the `raw_dt` + EPC-masking discovery via diagnostic prints.

## Fix plan (deferred to 組 4.5 / B-15)

Option A (recommended, treats root cause):
- Replace `far_pwr` in raw_dt with delay-aligned echo power estimate
- e.g. `raw_dt = max(0, 1 - echo_est_pwr / mic_pwr)` where `echo_est_pwr =
  np.sum(|filter.echo_spec|²)`
- Wrap behind `AEC_FIX_B15` flag, default ON
- Verify: PZ7V t=10s dt_indicator stays < 0.3

Options NOT to take:
- Relax EPC guards: defeats B-4a/4b/12 design intent
- Add ad-hoc echo onset detector: band-aid
- Relax mu_eff path: band-aid

## Defer reason

Finishing 800-case bisection of remaining flags (b11, b7, b3c) first, to
quantify what fraction of −0.147 FS regression is explained by the
raw_dt bug vs other unconditional changes.
