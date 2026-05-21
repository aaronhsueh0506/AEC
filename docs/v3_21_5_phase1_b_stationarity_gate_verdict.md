# v3.21.5 Phase 1 Sprint B — Stationarity AEC3-default-off CLOSED REJECTED

**Date**: 2026-05-21
**Branch**: `feature/v3_21_5_phase1_aec3_parity`
**Commits**: `b34f722` (B.1 flag introduction with default-False), `2b98c13` (B revert to default-True after CLOSED verdict)
**Status**: **CLOSED REJECTED for v3.21.5 cumulative ship** — load-bearing safety net evidence; re-test scheduled for v3.21.6 Sprint P4 after companion mechanisms ported
**Plan**: `~/.claude/plans/se-aec-aec-main-hazy-lynx.md` (Round 7 close)

## AEC3 source citation

[`docs/aec3_extracts/src/aec3/echo_audibility.h:40-51`](../docs/aec3_extracts/src/aec3/echo_audibility.h#L40) — `GetResidualEchoScaling` is hard 0/1.

[`docs/aec3_extracts/src/aec3/residual_echo_estimator.cc:303-313`](../docs/aec3_extracts/src/aec3/residual_echo_estimator.cc#L303) — scaling is gated by `aec_state.UseStationarityProperties()`, which reads `EchoCanceller3Config::EchoAudibility.use_stationarity_properties` (**AEC3 default = false**).

Our pre-fix [`orchestrator.py:3471`](../python/modules/orchestrator.py#L3471) unconditionally zeroed R² on stationary bins whenever the filter had converged, ignoring the AEC3 config. Sprint B introduced `aec3_post_stationarity_zero_enabled` flag with default = False (= AEC3 default) to restore port fidelity.

## Sprint B.0 — Trace from Sprint A cohort (historical setup)

`stationary_mask_fired_frac` / `r2_mask_kill_ratio` computed on Sprint A.0's 5 FS_static worst cases (`pcb1Nh`, `LN18k5r8`, `s90M7MOT`, `9xjhi`, `lV0kQN`). All showed `stationary_mask_active` ≥ 50% with significant `r2_mask_kill_ratio` → strong signal that the unconditional zeroing was active on cohort tail. Decision: proceed to B.1 / B.2 to bench whether removing the zeroing improves Pareto.

## Sprint B.1 — Implementation (commit b34f722, historical)

Wrapped [`orchestrator.py:3471`](../python/modules/orchestrator.py#L3471) stationarity block in `if self.config.aec3_post_stationarity_zero_enabled and _filter_converged_enough:`. Default `aec3_post_stationarity_zero_enabled: bool = False` (= AEC3 default). Env hook `AEC_STATIONARITY_ZERO=1` to opt back into the legacy unconditional zeroing for byte-equal A/B (verified 25/25 match vs v3.21.4 at `AEC_STATIONARITY_ZERO=1`).

## Sprint B.2 — Full 800-case bench (background `bael54gvf`, 2026-05-21)

```
python3 python/eval_aec_challenge.py wav/aec_challenge_blind/ \
    --preset balanced --filter 832 --cng --parallel \
    -o out_v3_21_5_phase1_b/ --workers 9
python3 python/bench_aecmos.py out_v3_21_5_phase1_b/ results_v3_21_5_phase1_b/ \
    --baseline results_v3_21_4_v1/scores.json \
    --label "v3.21.5_phase1_sprint_b_stationarity_gate"
```

### Bucket means vs v3.21.4 baseline

| Bucket | Δecho | Δdeg | direction |
|---|---:|---:|---|
| FS_static | **+0.032** | -0.000 | target direction ✓ |
| FS_movement | **+0.048** | -0.000 | target direction ✓ (bigger than A's +0.035) |
| DT_static | +0.007 | -0.002 | echo↑ deg neutral |
| DT_movement | +0.015 | **-0.035** | echo↑ deg ↓ small |
| NE | -0.000 | +0.001 | neutral |

On bucket means alone Sprint B looks acceptable (FS gain, DT marginal, NE neutral). The damage hides in the per-case distribution.

### Per-case distribution (vs v3.21.4)

| Bucket | n | echo imp>+0.05 | echo reg<-0.05 | deg imp>+0.05 | deg reg<-0.05 | esum_+ | esum_- | dsum_- | worst Δdeg |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| FS_static | 169 | 34 | 8 | 0 | 0 | +5.47 | -0.66 | 0.00 | -0.001 |
| FS_movement | 131 | 38 | 6 | 0 | 0 | +6.81 | -0.69 | 0.00 | -0.001 |
| DT_static | 186 | 25 | 16 | 32 | **32** | +3.01 | -2.22 | -6.86 | **-0.877** (`WcK0OrF`) |
| DT_movement | 114 | 19 | 9 | 12 | **30** | +2.47 | -1.01 | -5.89 | **-0.870** (`wVYSGV`) |
| NE | 200 | 0 | 0 | 3 | 3 | 0.00 | 0.00 | -0.25 | -0.124 |

- FS improvement strong (4.3:1 and 6.3:1 ratios)
- **DT cohort tail: 62 cases Δdeg < -0.05 (DT_static 32 + DT_movement 30)** — exceeds Sprint A's strict halt criterion (d) of "> 30 cases" by 2×

### Top 5 DT worst dreg cases (audio listen targets)

| stem | bucket | Δdeg | Δecho |
|---|---|---:|---:|
| `WcK0OrF6ukW03fViPXTQjQ` | DT_static | **-0.877** | -0.099 |
| `wVYSGVTTakih9twI4xlDWQ` | DT_movement | -0.870 | +0.433 |
| `wr2CEenGL0Oec4b1GW5Zww` | DT_movement | -0.784 | +0.178 |
| `xQEUtY2pWUi7v1X93TF2AA` | DT_static | -0.602 | -0.385 |
| `Y4zG6bHup06zWMoq3OvZqQ` | DT_static | -0.552 | +0.257 |

## Audio listen + spectrogram analysis (`/tmp/sprintB_audio/spectrograms/`)

Rendered baseline (`AEC_STATIONARITY_ZERO=1` = legacy) vs Sprint B (default unset = AEC3-default-off) for all 5 worst-deg cases. Computed Δ panels (Sprint B − baseline in dB on speech-active frames).

### Formant-band ΔdB (speech-active frames, mean)

| Case | Δdeg | F1 (300-800 Hz) | F2 (1-3 kHz) | F3 (3-6 kHz) | HF (6-8 kHz) |
|---|---:|---:|---:|---:|---:|
| WcK0OrF | -0.877 | -0.15 | **-1.31** | -0.99 | -0.60 |
| wVYSGV | -0.870 | -0.47 | **-1.26** | -0.41 | -0.15 |
| wr2CEen | -0.784 | **-1.14** | -0.78 | -0.34 | -0.03 |
| **xQEUtY2** | -0.602 | **-2.12** | **-1.28** | **-1.37** | **-1.39** |
| Y4zG6bH | -0.552 | **-1.73** | -0.77 | -0.77 | -0.79 |

### Sprint A worst case for comparison (qiQL0BUP from Sprint A verdict)

| Case | Δdeg | F1 | F2/F3 | F3+ |
|---|---:|---:|---:|---:|
| qiQL0BUP | -0.526 | -0.20 | -0.10 | -0.35 |

**Sprint B formant damage is 6-10× larger than Sprint A.** Sprint A's small attenuation didn't manifest as audible damage (user spectrogram check: "看起來差不多"). Sprint B's 1-2 dB across F1/F2/F3 formants IS audible-grade attenuation, NOT AECMOS over-reaction.

## xQEUtY2 trace deep-dive (mechanism evidence)

Full per-frame trace at `/tmp/sprintB_audio/trace_xQEUtY2_baseline.csv` (default-True / legacy) and `/tmp/sprintB_audio/trace_xQEUtY2_sprintB.csv` (default-False / AEC3-default-off).

### Whole-case summary

| Metric | baseline (stat_zero ON) | Sprint B (default OFF) |
|---|---:|---:|
| `stationary_mask_active` fire rate | **68.6%** | 68.6% |
| `stationary_mask_fired_frac` (per-bin mean) | 0.534 | 0.534 |
| `is_nearend_state` fire rate | **7.2%** | 7.2% |
| `usable_linear` fire rate | 96.3% | 96.3% |

The detector fires NE only 7.2% of the time on this DT case — `dominant_nearend` is mis-firing under stationary-far conditions. When NE detector says "not NE", `SuppressionGain` uses `nearend_tuning=False` → far-tuning aggressive gain.

### Per-frame gain Δ distribution (Sprint B − baseline)

| Band | mean Δ | min Δ | p1 Δ | p5 Δ | p95 Δ |
|---|---:|---:|---:|---:|---:|
| gain_5 (LF) | -0.051 | -0.992 | **-0.907** | -0.729 | 0.000 |
| gain_30 | -0.050 | -0.979 | -0.909 | -0.634 | 0.000 |
| gain_100 (HF) | -0.050 | -0.964 | -0.882 | -0.617 | 0.000 |
| gain_200 | -0.053 | -0.989 | -0.912 | -0.774 | 0.000 |

Mean Δ looks small (~0.05) but the distribution is heavy-tailed: p1 hits **-0.9** = near-full suppression difference (gain drops from ~1 preserve to ~0 full-suppress).

### Catastrophic damage windows

Frames where `Δgain_100 < -0.3` (severe HF damage):

| seg | t (s) | length | Δgain_100 | Δgain_30 | stationary_mask | NE state |
|---|---|---|---:|---:|---:|---:|
| 0 | 2.33-2.55 | 23 fr | -0.660 | -0.469 | 100% | 0% |
| 1 | 4.02-4.24 | 23 fr | -0.565 | -0.632 | 100% | 0% |
| 2 | 6.08-6.88 | 81 fr | -0.590 | -0.676 | 100% | 0% |
| 3 | 8.77-9.30 | 54 fr | **-0.836** | 0.000 | 100% | 0% |
| 4 | 9.70-10.70 | 101 fr | -0.702 | -0.684 | 100% | 0% |
| 5 | 11.62-11.73 | 12 fr | -0.615 | -0.667 | 100% | 0% |

**6 segments, 294 frames total (7.4% of case), all with `stationary_mask_active=100%` AND `is_nearend_state=0%` simultaneously.** In a 40-s case this is **~2.9 seconds of full NE-speech suppression** when the baseline path was preserving it.

Visual confirmation in `/tmp/sprintB_audio/spectrograms/xQEUtY2_gain_timeseries.png` (gain time series with baseline=black / Sprint B=red, red fill where Sprint B suppresses more) and `compare_5cases_diff.png` (5-case mic / baseline / Sprint B / Δ-dB spectrograms — red stripes mark Sprint B's catastrophic windows on xQEUtY2 and Y4zG6bH).

## Mechanism (cascade root cause)

```
cohort tail DT case (lots of stationary far-end noise: HVAC / fan / room tone)
    ↓
StationarityEstimator correctly fires on 68% of frames / 52% of bins
    ↓
[baseline / Sprint A path] _filter_converged AND stationary_mask → r2 = 0 on those bins
    → SuppressionGain sees no echo on those bins → gain stays high (≈ 1)
    → NE-speech HF / formants preserved ✓
[Sprint B path / AEC3-default-off] mask suppressed → r2 keeps its computed value (~1e11)
    → SuppressionGain treats those bins as echo
    → is_nearend_state only 7.2% (detector mis-fires under stationary-far)
    → far-tuning gain (aggressive) applied to NE-speech bins
    → gain drops to ~0 → NE speech destroyed
```

## Why AEC3-vanilla default-off doesn't fail their cohort but ours does

AEC3 has a stack of companion mechanisms that keep `is_nearend_state` firing correctly under stationary-far conditions:
- `ScaleFilter` + `FilterMisadjustment` (calibrate filter convergence + scale residual estimate)
- `EchoAudibilityConfig` structural wiring (canonical control surface)
- `FilterAnalyzer` (per-bin convergence score feeding reverb + delay)
- `TransparentMode` (gate transitions)

Our v3.21.4 port has incomplete coverage of these. Removing the stationarity zeroing exposes the `is_nearend_state` mis-fire that AEC3-vanilla doesn't suffer from. **The stationarity zeroing in our port is a load-bearing safety net**, not redundant.

This matches `memory/feedback_aec_code_review_accuracy.md` (existing aec.py tuning is load-bearing) and `memory/feedback_no_shortcut_use_canonical.md` (work-arounds only as safety layer).

## Verdict (Round 7 user decision)

**CLOSED REJECTED for v3.21.5 cumulative ship.** Reasoning:
1. Strict halt criterion (d) `> 30 cases Δ < -0.05` exceeded **2×** (62 DT cases)
2. Audio evidence (formant-band Δ) shows **real audible damage**, not AECMOS over-reaction
3. xQEUtY2 trace deep-dive proves mechanism: ~2.9 s of full NE-speech suppression per 40-s case
4. Override that worked for Sprint A (small attenuation, no audible) does NOT apply here — Sprint B damage IS audible
5. The legacy stationarity zeroing is load-bearing safety net, not redundant

## Action (committed in 2b98c13)

- `aec3_post_stationarity_zero_enabled: bool = True` (default reverted from False to True)
- Flag kept for opt-in research (`AEC_STATIONARITY_ZERO=0` to opt into AEC3-default-off A/B)
- Byte-equal 25/25 verified vs v3.21.4 70e7f96 baseline (Sprint B revert restores legacy behavior exactly)
- v3.21.5 cumulative ship path: **A + C2 only** (B excluded; never in shipping cumulative)
- Re-test deferred to **v3.21.6 Sprint P4** AFTER:
  - **P1**: FilterAnalyzer / direct-path delay port (companion delay signal)
  - **P2**: TransparentMode / AecState parity audit (companion NE detector firing)
  - **P3**: EchoAudibilityConfig structural wiring (canonical control surface so P4's re-test happens through `suppressor_config.echo_audibility.use_stationarity_properties=False`, not via the top-level flag)
  - **P4**: re-bench `use_stationarity_properties=False` on the v3.21.6 baseline; if companion mechanisms rescued `is_nearend_state` accuracy, behavior may flip default-off safely

## Triage policy reference

Per `~/.claude/plans/se-aec-aec-main-hazy-lynx.md` "AEC3 Parity Gap Triage Policy", Sprint B is **Bucket 1 (Must fix / evaluate before v3.22)** with cycle home split across v3.21.5 (close rejected) + v3.21.6 P4 (re-test after companions). The flag is retained as deprecated-pending-P3-wiring; P3 promotes the control surface to `EchoAudibilityConfig` and P4 decides the default value through it.

## Reproduction

```bash
git checkout 2b98c13   # v3.21.5 Sprint B revert (default-True restored)
# Verify byte-equal vs v3.21.4:
python3 python/check_byte_equal.py
# Compare to v3.21.4 baseline:
git checkout 70e7f96 && python3 python/check_byte_equal.py 2>&1 | grep got | sort > /tmp/v3214.txt
git checkout feature/v3_21_5_phase1_aec3_parity && python3 python/check_byte_equal.py 2>&1 | grep got | sort > /tmp/current.txt
diff /tmp/v3214.txt /tmp/current.txt  # should be empty

# Reproduce the rejected Sprint B 800-case bench:
AEC_STATIONARITY_ZERO=0 python3 python/eval_aec_challenge.py wav/aec_challenge_blind/ \
    --preset balanced --filter 832 --cng --parallel \
    -o out_v3_21_5_phase1_b_repro/ --workers 9
python3 python/bench_aecmos.py out_v3_21_5_phase1_b_repro/ results_v3_21_5_phase1_b_repro/ \
    --baseline results_v3_21_4_v1/scores.json \
    --label "v3.21.5_phase1_sprint_b_repro"

# Reproduce xQEUtY2 trace deep-dive:
stem=xQEUtY2pWUi7v1X93TF2AA_doubletalk
mic=wav/aec_challenge_blind/doubletalk/${stem}_mic.wav
ref=wav/aec_challenge_blind/doubletalk/${stem}_lpb.wav
AEC_STATIONARITY_ZERO=1 python3 python/run_one_case.py "$mic" "$ref" /tmp/baseline.wav --preset balanced --trace-hf-chain /tmp/baseline.csv
AEC_STATIONARITY_ZERO=0 python3 python/run_one_case.py "$mic" "$ref" /tmp/sprintB.wav  --preset balanced --trace-hf-chain /tmp/sprintB.csv
```

Audio listen pairs: `/tmp/sprintB_audio/{baseline,sprintB}/`. Spectrograms: `/tmp/sprintB_audio/spectrograms/compare_5cases_diff.png` + `xQEUtY2_gain_timeseries.png`.
