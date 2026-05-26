# v3.21.14 A.4 + A.5 fire-rate targeted audit (parallel-track)

Cycle: 2026-05-23
Probe: [`python/a4_a5_fire_rate_probe.py`](AEC/python/a4_a5_fire_rate_probe.py)
Probe measures hypothetical A.4 / A.5 fire rate on shadow filter using the
already-computed main-filter RSA + saturation_level state. Default-OFF
flags are unchanged — probe is read-only.

## Cohorts probed

| Cohort | Cases | Bucket split | Source |
|---|---|---|---|
| 12-case | 12 | 7 DT + 4 FS + 1 NE | `wav/v3_21_8_cohort/` (v3.21.8 stress cohort) |
| 90-case | 90 | 30 DT + 30 FS + 30 NE | `wav/aec_challenge_blind/` (first 30 per bucket) |

## A.4 narrowband mask — `np.any(rsa.counters > 5)`

| Cohort | Mean fire | Max fire | Cases ≥ 10 % | Top fire cases |
|---|---:|---:|---:|---|
| 12-case | 4.80 % | 40.95 % | 1 / 12 | xFk7igecuke0R5JMfREyDg_doubletalk_with_movement (40.9 %) |
| 90-case | 1.95 % | 37.43 % | 2 / 90 | Je6gJ7y1PECStwxnrOe9aA_doubletalk_with_movement (37.4 %) / Je6gJ7y1PECStwxnrOe9aA_doubletalk (37.3 %) |

**Avg bins masked per firing frame**: 5–157 (most cases 5–25 bins; xFk7igecuke
hits 157 bins which is substantial — multiple narrowband peaks simultaneously).

### A.4 audit verdict

- **Mainstream AEC challenge cohort population**: ~2 % of cases fire ≥ 10 %
  (Je6gJ7y1 variants share the same recording). This is a niche
  population, not a broad parity capability.
- **Mask intensity when it fires**: substantial (up to 157 bins per frame),
  suggesting the cases that fire would see meaningful shadow update
  suppression if A.4 enabled.
- **No 800-case AECMOS data**: probe is read-only; need separate A.4-on
  render to measure actual impact.

**Disposition**: A.4 stays **default-OFF substrate**. Not justified for
v3.21.x ship cycle on general AEC challenge data. Re-open only if:
1. We acquire a tonal / narrowband-rich cohort (DTMF / music conference / sustained-tone telephony)
2. ≥ 10 % of cases fire ≥ 10 % on that cohort
3. A/B AECMOS shows net positive

## A.5 saturation gate — `_saturation_level > 0.5`

| Cohort | Mean fire | Max fire | Cases ≥ 5 % |
|---|---:|---:|---:|
| 12-case | 0.00 % | 0.00 % | 0 / 12 |
| 90-case | 0.00 % | 0.00 % | 0 / 90 |

### A.5 audit verdict

- **AEC challenge cohort is clean of saturation events**: 0 firing on 102
  probed cases.
- **A.5 cannot be activated on this cohort**: any A/B would be byte-equal.

**Disposition**: A.5 stays **default-OFF substrate**. Cannot proceed
with v3.21.x ship cycle on current cohort. Re-open only if:
1. We acquire a clipped / saturated cohort (intentionally clipped mic / loud near-end with mic-overload / synthetic clipping)
2. ≥ 5 % of cases fire ≥ 5 % on that cohort
3. A/B AECMOS shows net positive (downstream on the saturated cases)

## Production state — unchanged

- `use_narrowband_mask_for_shadow: bool = False` — substrate, env hook `AEC_SHADOW_NARROWBAND_MASK=1`
- `use_saturation_gate_for_shadow: bool = False` — substrate, env hook `AEC_SHADOW_SATURATION_GATE=1`
- Orchestrator Phase 2 wiring (RSA + saturation propagation to shadow) remains intact

## Artefacts

- `/tmp/a4_a5_12case.csv` — 12-case probe output
- `/tmp/a4_a5_90sample.csv` — 90-case probe output

## Next step for A.4 / A.5

- A.4: search Audio_ALG NR + integration repos for narrowband / tonal-rich
  test data; OR synthesise (DTMF generator + ringtone samples) and re-audit
- A.5: scout clipped-recording corpora; OR generate synthetic clipped
  variants of existing 800-case (mic peak → 1.0 saturate)

Neither is gating on v3.21.15 (A.2 + A.3 ship cycle).
