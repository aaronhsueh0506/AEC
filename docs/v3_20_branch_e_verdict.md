# v3.20 Branch E (+ E.2) — Source-vs-port gap closure verdict (2026-05-17)

## Context

Per plan `~/.claude/plans/se-aec-aec-main-hazy-lynx.md` ADDENDUM 2,
Branch E was authorised to close 3 deferred port gaps after
post-Bench-D/E/F/H mask-sweep saturation pointed at structural
deficits. User criterion (locked 2026-05-17): echo and deg both
close-to-parity (within 0.10) or winning vs AEC2 reference on
all 4 columns.

Two sub-bench iterations were run:

- **Branch E**: Reverb wiring fix + WeightEchoForAudibility port
  + min_echo_power config. Default config Gap A active in isolation.
- **Branch E.2**: + AEC3-faithful post-SG override
  (remove `spectral_g_min` clamp in `_stage_temporal_smoothing`;
  remove `noise_floor_gain = max(gain, noise_floor_gain)` clamp in
  `_stage_noise_floor_and_cng`).

## Verified AEC3 source vs our port (re-traced 2026-05-17)

| Layer | AEC3 source | Our port pre-E | Status |
|-------|-------------|----------------|--------|
| **R² linear** `residual_echo_estimator.cc:91-105` | `R² = S²_lin / ERLE` then `+= reverb_power` | Same formula. Reverb wiring was DEAD (`_K_FFT_BY2 = 128` vs AEC3 64 → `ReverbFrequencyResponse.update` raised silently). | **Fixed in E1** |
| **R² nonlinear** `cc:262-291` | `R² = X² × echo_path_gain²`; default early/late gain = 0.01 amp | Default 0.1 amp (100× larger power). Rarely exercised since `usable_linear=98%` post-revival. | Not fixed (low leverage) |
| **R² reverb render** `cc:367-370` | Render power is from `FilterLengthBlocks+1` blocks ago (delayed) | Current render passed | **Not fixed** — would need delayed render buffer; suspected cause of Branch E movement regression |
| **R² scaling** `cc:303-313` | per-bin `GetResidualEchoScaling` only when `UseStationarityProperties` | Same | OK |
| **Onset compensation** `cc:251-254` | `compensated = flag OR !dominant_ne` | `compensated = !dominant_ne` (matches at default config) | OK |
| **SG GainToNoAudibleEcho** `suppression_gain.cc:215-233` | `g = (suppress-enr)/(suppress-transparent)`, NOT clipped inside; outer clamps to [min_gain, max_gain] | We clip [0,1] inside `_gain_to_no_audible_echo`; same effective result | OK |
| **SG GetMinGain** `cc:237-276` | `min_gain[k] = min(1, min_echo_power / weighted_echo[k])` | Same formula. Hardcoded 0.5/1.0 → made config-driven in E3; defaults preserve behaviour. Probed: our R² p50 = 53 → ENR ≈ 0.68 (matches AEC3 normal_lf default 0.5 trigger). **Units are correct between R² and near_psd.** | OK |
| **SG WeightEchoForAudibility** `cc:88-121` | 3-band downweight quiet echo by `(1−x²)` | Skipped → ported in E2 default-OFF. **Re-verified direction: WeightEchoForAudibility makes ENR SMALLER → LESS suppression → protects NE, NOT a FS lever.** Agent diagnosis had this reversed. | Ported (NE-only lever, opt-in) |
| **SG LowerBandGain LF smoothing** `cc:257-272` | `min_gain[k] >= last_gain[k] * max_dec_factor_lf` for LF bins | Same | OK |
| **MovingAverageSpectrum** `cc:308` | `nearend_smoothers_[ch].Average(...)`, default blocks=1 (no-op) | Not implemented; default is no-op so byte-equal at default | OK |
| **Post-SG `suppression_filter.cc:88-183`** | **Minimal**: `noise_gain = sqrt(1−g²); E = g*E + noise_gain*cng; IFFT; OLA; SafeClamp`. **NO floor / no g_min lift / no temporal smooth / no spectral envelope** | 4 extra stages: `_stage_gain_postprocess` (quiet_mask / 3bin / HF cap / divergence), `_stage_temporal_smoothing` (attack/release EMA + `max(g, spectral_g_min)`), `_stage_noise_floor_and_cng` (`max(g, noise_floor_gain)` + CNG addition) | **Branch E.2 added override**: skip `spectral_g_min` + `noise_floor_gain` clamps when AEC3 SG active; keep `noise_psd` learning + additive CNG |

## Bench results (Phase 0 corpus, BALANCED preset, fl=832, cng=True, j=6)

| Bucket | Branch E | Branch E.2 | Phase 0 | AEC2 | AEC3 |
|--------|---------:|-----------:|--------:|-----:|-----:|
| FS_static echo | 3.161 | **3.140** | 3.769 | 3.48 | 3.88 |
| FS_movement echo | 2.877 | 2.868 | (~3.5) | n/a | n/a |
| DT_static echo | 3.712 | 3.719 | 4.184 | 4.26 | 4.54 |
| DT_static deg | 2.993 | **2.962** | 2.305 | 2.39 | 1.85 |
| DT_movement echo | 3.418 | 3.424 | (~3.6) | n/a | n/a |
| DT_movement deg | 3.140 | 3.138 | (~2.4) | n/a | n/a |
| NE deg | 4.010 | 4.010 | 4.005 | 4.10 | 3.45 |

## User criterion check — Branch E.2 (best AEC3 SG variant tested)

| Column | AEC2 | Branch E.2 | Δ vs AEC2 | Within 0.10? |
|---|---:|---:|---:|:---:|
| FS_static echo | 3.48 | 3.140 | **−0.340** | NO |
| DT_static echo | 4.26 | 3.719 | **−0.541** | NO |
| DT_static deg | 2.39 | 2.962 | +0.572 (WIN) | YES |
| NE deg | 4.10 | 4.010 | −0.090 | YES (marginal) |

**FAILS** on FS_static AND DT_static echo. Phase 0 anchor meets
criterion on all 4 columns.

## E.2 mechanism — why removing the floor clamp didn't help aggregate

Per-frame probe (9xjhi FS_static worst, 2102 frames with non-zero R²):

**With floor clamp (Branch E)**:
- AEC3 SG output (pre-NFL) min gain median: 0.0151 (-36 dB) — SG wanted deep
- Post-NFL min gain median: 0.0354 (-29 dB) — clamped UP by 7 dB
- `noise_floor_gain` median: 0.397 (-8 dB)
- 2107/2187 frames have pre-NFL min < 0.05 — SG aimed deep on 96% of frames

**Without floor clamp (Branch E.2)**:
- Single-case smoke (0Kjz... benign FS): ERLE 3.6 → 16.4 dB (+12.8) — works for easy cases.
- 9xjhi (worst FS): output RMS distribution UNCHANGED to 5 decimals (p10 0.02050 vs 0.02041, p50 0.18980 vs 0.18974, p90 0.33210 vs 0.33229).

**Interpretation**: the floor clamp was suppressing GAIN deeper on
easy FS cases, but on the WORST FS cases (which drag down the
bucket average) the gain wasn't going below the floor anyway — SG
itself wasn't going deep enough on those cases, probably because:

- Linear filter doesn't converge cleanly on worst cases → S²_lin
  isn't a good echo estimate → R² doesn't reflect true echo
- ENR gate misfires on those frames (high near_psd from leaked echo
  raises denominator)
- Or there's upstream wiring (`_aec3_r2_override` exception path)
  that bypasses AEC3 SG on those frames

**Worst-case FS is not bottlenecked by post-SG clamps**. It's
bottlenecked upstream — most likely by the linear filter quality on
those audio files, which is OUTSIDE the AEC3 SG path.

## Direction-correction on prior diagnosis (multiple agent errors)

The Plan ADDENDUM 2's framing was wrong on multiple counts:

1. **Gap B `WeightEchoForAudibility`**: Agent said "skipping it makes
   ENR over-aggressive → mask saturates". Direct re-read of
   `suppression_gain.cc:88-121` shows it **down-weights** echo below
   floor → ENR **smaller** → **less** suppression. It protects NE,
   NOT FS. Agent had direction reversed.

2. **Gap C `min_echo_power`**: Agent said hardcoded 0.5/1.0 was off
   by 1e5 vs our pipeline. Probe found our R² is in 10-100 range
   (p50=53), matching near_psd (p50=79), so hardcoded values were
   already correctly scaled. Made configurable for tuning but no
   behaviour change at default.

3. **Gap A reverb wiring**: Real bug (constants 128/129 vs AEC3
   64/65 caused silent ValueError in reverb update for Bench D-H).
   Fixed in E1. Effect on bench: neutral on static, slight
   regression on movement (reverb accumulating stale render across
   path changes — same as AEC3 `cc:367-370` "filter_length+1 blocks
   ago" issue we didn't fix).

4. **FS ceiling NOT at mask saturation**: Agent suggested mask sweep
   saturated due to gate logic. Re-trace shows AEC3 SG **does** want
   to push gain to 0 on 96% of frames; the AGGREGATE is limited by
   the WORST cases where linear filter quality is poor → SG never
   sees a clean R² to act on.

## Outcome — confirmed: AEC3 SG path cannot reach user criterion

Three bench iterations (E, E.2, plus prior Bench D/F/H) all fail
to lift FS_static past ~3.16-3.42 vs AEC2 anchor 3.48. The
ceiling is structural, not tunable via post-SG mods:

- Mask sweep (Bench F/H): FS saturates at 3.42
- Substrate revival + reverb fix (Branch E): FS at 3.16
- AEC3-faithful post-SG removal (Branch E.2): FS at 3.14

**Phase 0 (legacy production) remains the only path meeting user
criterion** on all 4 columns (FS +0.27 vs AEC2 WIN, others within
0.10).

## Decisions (pending user authorisation)

1. **Ship Phase 0 as v3.20 production default**. AEC3 SG path
   confirmed structurally Pareto-bound below AEC2 FS reference.

2. **Retain Branch E1+E2+E3+E.2 changes as substrate landings**
   (default-OFF). All are correctness improvements:
   - Reverb subsystems are alive (were dead-via-silent-exception)
   - WeightEchoForAudibility port available for future NE tuning
   - min_echo_power config knob available
   - AEC3-faithful post-SG path available (matches AEC3 verbatim)

3. **Cycle close-out**: move to v3.20 Phase 4 (Python cleanup) →
   v3.21 (C-port catch-up) → v3.22 (NN/Volterra, DSP-only mandate
   demonstrably exhausted).

4. **NOT fixed (would need separate effort, out of v3.20 scope)**:
   - Delayed-render to reverb update (`FilterLengthBlocks+1` blocks
     ago per `residual_echo_estimator.cc:367`)
   - `default_gain_amplitude` mismatch (we use 0.1, AEC3 uses 0.01)
   - Worst-case FS linear filter quality (the actual aggregate
     bottleneck) — this is NLMS-domain work, not SG-domain

## Files touched (Branch E + E.2)

- `python/modules/reverb.py` — constants 128/129 → 64/65 (E1)
- `python/modules/state.py` — `get_reverb_frequency_response()` +
  `get_reverb_decay()` added (E1)
- `python/modules/residual_estimator_aec3.py` —
  `_update_reverb_linear` reads aec_state frequency response (E1)
- `python/modules/suppression_gain_aec3.py` — ported
  `_weight_echo_for_audibility` (E2); `_get_min_gain` reads config
  (E3); 6 new `audibility_*` ctor kwargs
- `python/modules/res_filter_aec3.py` — plumbs 6 audibility kwargs;
  **`_stage_temporal_smoothing` and `_stage_noise_floor_and_cng`
  overrides drop floor clamps in AEC3-faithful mode** (E.2)
- `python/modules/config.py` — 6 new `aec3_echo_audibility_*`
  knobs (defaults preserve prior behaviour)
- `python/modules/orchestrator.py` — passes audibility config
- `python/eval_aec_challenge.py` — env-var hooks for the 6 knobs

All gates default-OFF or default-neutral; Phase 0 byte-equal
preserved at default config.
