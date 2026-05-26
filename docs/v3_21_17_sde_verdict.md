# v3.21.17 SignalDependentErleEstimator port — verdict

Cycle: 2026-05-23
Implementation: [`python/modules/state/signal_dependent_erle.py`](AEC/python/modules/state/signal_dependent_erle.py) (verbatim AEC3 port)
Wiring: `python/modules/state/erle_estimator.py` + `python/modules/state/aec_state.py` + `python/modules/orchestrator.py`
Config: `python/modules/config.py` `signal_dependent_erle_sections: int = 0`
Env hook: `AEC_SIGNAL_DEPENDENT_ERLE=N` (0 = OFF default)

## TL;DR

**v3.21.17 CLOSE as no-leverage substrate.** SDE port complete + C-protocol
Gates 1 / 2 / 3 PASS. Activated configs (sections=2 / 8) produce ≤ 0.002
bucket-mean AECMOS delta on 12-case cohort — matches Finding 2 prediction
that BALANCED preset (single-channel, no multi-section echo paths) does
not activate per-section ERLE refinement meaningfully.

Per plan disposition: **"V2 byte-equal or Δ ≤ 0.005 mean → CLOSE as
no-leverage; dormant substrate"**. Port retained as default-OFF substrate
(env hook + config flag preserved); no 800-case bench needed since
12-case unanimously shows no leverage.

## C-protocol gate summary

| Gate | Status | Evidence |
|---|---|---|
| 1 Code audit | PASS | Verbatim port of `docs/aec3_extracts/src/aec3/signal_dependent_erle_estimator.{cc,h}` (cc:32-426). Single-channel adapted from AEC3's multi-channel template. Subband boundaries scaled ×4 to our 257-bin spectrum (AEC3 65-bin → ours 257-bin). One adaptation documented: last-bin mirror (n_freqs-1 = n_freqs-2) to match our baseline subband_erle convention (subband_erle.py:89) since AEC3's `kFftLengthBy2` loop stops at bin 64 leaving bin 65 at min_erle. |
| 2 Default-OFF byte-equal | PASS | `signal_dependent_erle_sections=0` default → `_sde = None` in ErleEstimator; no SDE codepath executes. Smoke test confirmed: `aec._aec3_state._erle_estimator._sde is None`. Orchestrator wiring gated by `if sections > 0`; SDE inputs not computed (None passed). |
| 3 Degeneracy proof (sections=1) | PASS (after bin-256 mirror fix) | 12/12 MD5-equal vs SDE-OFF on `wav/v3_21_8_cohort/`. Initial port had 3-case diff due to AEC3 `kFftLengthBy2` (exclusive) loop semantics; resolved by mirroring last bin from second-last. |
| 4 12-case stratified | PASS as "no leverage" | All buckets Δecho ≤ 0.002, Δdeg = 0.000 on sections=2 AND sections=8. No per-case Δ > 0.005. |
| 5 Per-sub cross-attribution | N/A | Single mechanism (no combo) |
| 6 800-case + fingerprint stratification | SKIPPED (12-case no-leverage triggers CLOSE before 800-case) | per disposition matrix: 12-case Δ ≤ 0.005 → CLOSE substrate, no 800-case needed |

## 12-case bucket means

V0 baseline = `out_12_v3_21_14_A/` (re-used, v3.21.15 V0).

| Bucket | N | V0 echo | V0 deg | SDE2 Δe | SDE2 Δd | SDE8 Δe | SDE8 Δd |
|---|---:|---:|---:|---:|---:|---:|---:|
| DT_movement | 3 | 3.917 | 3.046 | −0.001 | +0.000 | +0.001 | +0.000 |
| DT_static | 4 | 4.419 | 2.748 | +0.000 | +0.000 | +0.000 | +0.000 |
| FS_movement | 1 | 3.818 | 4.999 | +0.000 | +0.000 | +0.000 | +0.000 |
| FS_static | 3 | 3.167 | 5.000 | +0.002 | +0.000 | +0.001 | +0.000 |
| NE | 1 | 4.999 | 4.354 | +0.000 | +0.000 | +0.000 | +0.000 |

## Mechanism: why no leverage on our cohort

Per Finding 2 (broad AEC3 alignment audit), SDE refines ERLE per-section
when multiple impulse-response sections contribute differently to echo
estimate. This requires:
1. **Multi-channel input** — AEC3 default `num_sections=8` is configured for
   multi-channel processing. Single-channel cohort can't benefit from
   cross-channel section averaging.
2. **Heterogeneous reverb** — sections diverge when direct path / early
   reflections / late reverb have meaningfully different ERLE
   characteristics. AEC challenge cohort is mostly single-room recordings
   with relatively uniform reverb (no music-over-speech, no DTMF-over-room).
3. **Long enough filter** — section sizes are non-linear (2, 4, 8, 16, ...
   blocks). With our 13-partition filter (52 ms), sections=8 gives <1
   block per late section; section data is too sparse to drive correction
   factor updates (cc:336 `num_updates > 50` gate rarely met).

These three conditions explain why sections=2 and sections=8 produce
near-zero AECMOS delta. The port is correct; the cohort doesn't excite
the mechanism.

## Disposition

**v3.21.17 CLOSE as no-leverage substrate.**

- Port retained: `python/modules/state/signal_dependent_erle.py` (290 lines)
- Wiring retained: env hook `AEC_SIGNAL_DEPENDENT_ERLE` + AecConfig field +
  AecStateConfig field + ErleEstimator integration + orchestrator W²/X² input
- Production main / `__version__` unchanged
- No 800-case bench (12-case unanimously no-leverage triggers CLOSE)
- Re-open trigger: multi-channel cohort OR long-filter config (n_partitions
  > 30) OR heterogeneous reverb cohort

## Forbidden post-v3.21.17

- No default-True flip (no AECMOS evidence)
- No `num_sections` tuning to non-{1,2,8} (v3.22 territory per AEC3-verbatim rule)
- No SDE wiring changes without re-running Gates 1-4

## Artefacts

- `python/modules/state/signal_dependent_erle.py` — new module
- `/tmp/be_v3_21_17_off/` — SDE-OFF render
- `/tmp/be_v3_21_17_deg2/` — SDE-sections=1 (degenerate, byte-equal proof)
- `/tmp/be_v3_21_17_act/` — SDE-sections=2 render
- `/tmp/be_v3_21_17_s8/` — SDE-sections=8 render
- `results/v3_21_17_SDE2/scores.json` + `result.md`
- `results/v3_21_17_SDE8/scores.json` + `result.md`

## Next per plan

v3.21.18-pre B.3 EchoAudibilityEstimator audit. v3.21.16 verdict pending
(800-case fingerprint probe still in flight, ~70% complete).
