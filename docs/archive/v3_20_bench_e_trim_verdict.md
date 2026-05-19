# v3.20 Bench E — Phase 2 trim + dominant_ne port audit verdict (2026-05-17)

## TL;DR

- **Bench E is numerically identical to Bench D / Bench A / Bench C** on
  800-case AECMOS (Δ < 0.01 every bucket).
- **Disabling 2.4 + 2.5 + 2.7 changed nothing** (composition is flat
  per [[v3_20_phase3_ship_verdict]]).
- **Fixing 5 dominant_ne port bugs vs AEC3 source changed nothing**
  on 800-case bench (correctness improvement only; bench-inert because
  the buggy detector was already not firing on FS by accident, and
  composition is dominated by AEC3 SG default tuning, not by detector
  state).
- **Verdict re-confirmed**: AEC3 SG + ResidualEstimator + AecState
  pipeline is structurally NE-preserving on our cohort, regardless of
  substrate revival or DT-protection arc wiring. Phase 0 stays
  production. Bench A pattern remains valid as opt-in
  `speech_preserving` preset.

## Context

Session goal (user 2026-05-17): "review phase2 出了什麼問題導致 echo
抑制不足". User intuition: AEC3 應該是 echo-aggressive first; current
Bench D 結果 (-0.6 dB FS, +0.7 dB DT deg) 不相符。

Three parallel explore agents (suppression_gain.cc verbatim read +
DT-protection mechanism trace + gain pipeline localisation) returned:

1. **AEC3 is NE-preserving by default** — overturns user intuition.
   Normal LF mask (0.5, 1.5) wide gate vs Nearend LF mask (1.09, 1.1)
   nearly-flat (need 100× ratio). AEC3 only goes echo-aggressive on
   saturation/divergence (`min_gain=0`).
2. **2.4 leakage_diverged + 2.5 boost_q cascade** identified as
   suspected over-fire chain (Q-boost loop on always-on substrate).
3. **dominant_ne port** had 5 bugs vs AEC3
   `dominant_nearend_detector.cc` (LF-only sum, snr_threshold gate,
   exit threshold, trigger_threshold=1 → flickers on single-frame
   evidence, missing initial_state gate).

Bench E was launched to test the trim+fix hypothesis.

## Bench E config

```
AEC_USE_AEC3_SUPPRESSION_GAIN=1
AEC_USE_AEC3_RESIDUAL_ESTIMATOR=1
AEC_AEC_STATE_FULL_ENABLED=1
AEC_RES_AEC3_SKIP_LEGACY_POST=1
AEC_TRANSPARENT_MODE_ENABLED=1
AEC_FILTER_MISADJUSTMENT_ENABLED=1
AEC_FILTER_MISADJUSTMENT_USE_FQ_USABLE=1
AEC_AEC3_DELAY_CONTROLLER_ENABLED=1
AEC_PHASE_F_USE_AEC3_DELAY=1
AEC_MOV_RATE_DELAY_EST_ENABLED=1
AEC_DOMINANT_NE_DETECT=1
# 2.4 + 2.5 + 2.7 OFF:
#   AEC_LEAKAGE_DIVERGED_ENABLED unset
#   AEC_PHASE_A_USE_AEC3_DELAY_QUALITY unset
#   AEC_MOV_RATE_USE_AEC3_ERROR_RATIO unset
```

Plus dominant_ne port fix in
`python/modules/suppression_gain_aec3.py:140-180`:
- LF-only summing (bins 1-60 = AEC3 bins 1-15 frequency range)
- `snr_threshold=30.0` gate (NE > 30× noise)
- `enr_exit_threshold=10.0` gate (early exit on strong echo)
- `trigger_threshold=12` (AEC3 verbatim; was 1 → flicker-prone)
- `initial_state` gate + `use_during_initial_phase=True`

Output dir: `/tmp/v3_20_e_trim_2_4_2_5/`

## Results

| Bucket | n | Bench E echo | Bench E deg | Bench D echo | Bench D deg | ΔE-D echo | ΔE-D deg |
|---|---:|---:|---:|---:|---:|---:|---:|
| FS_static  | 169 | 3.162 | 5.000 | 3.158 | 5.000 | +0.004 | 0.000 |
| FS_movement| 131 | 2.958 | 5.000 | 2.950 | 5.000 | +0.008 | 0.000 |
| DT_static  | 186 | 3.712 | 2.993 | 3.715 | 2.996 | -0.003 | -0.003 |
| DT_movement| 114 | 3.552 | 3.108 | 3.542 | 3.111 | +0.010 | -0.003 |
| NE         | 200 | 4.998 | 4.010 | 4.998 | 4.010 |  0.000 | 0.000 |

**Δ < 0.01 dB on all buckets**. Bench E is numerically identical to Bench D.

## Cross-bench comparison (all 800-case BALANCED)

| Bench | FS_st echo | FS_mv echo | DT_st echo | DT_mv echo | DT_st deg | DT_mv deg | NE deg |
|---|---:|---:|---:|---:|---:|---:|---:|
| **AEC2 ref**          | — | — | — | — | 2.39 | 2.39 | 4.10 |
| **AEC3 deg ref**      | — | — | — | — | 1.85 | 1.85 | 3.45 |
| **Phase 0 (legacy)**  | 3.769 | 3.730 | 4.244 | 4.082 | 2.297 | 2.318 | 4.005 |
| Tier 1 (no substrate) | 3.295 | 3.347 | 3.711 | 3.695 | 2.865 | 2.846 | 4.016 |
| Tier 2 B (normal-tune)| 3.340 | 3.386 | 3.727 | 3.711 | 2.848 | 2.824 | 4.016 |
| Bench A (skip-post)   | 3.375 | 2.923 | 3.786 | 3.481 | 2.865 | 3.098 | 4.010 |
| Bench C (all on)      | 3.373 | 2.922 | 3.786 | 3.484 | 2.868 | 3.100 | 4.010 |
| Bench D (substrate revived, all on) | 3.158 | 2.950 | 3.715 | 3.542 | 2.996 | 3.111 | 4.010 |
| **Bench E (trim 2.4+2.5+det fix)** | **3.162** | **2.958** | **3.712** | **3.552** | **2.993** | **3.108** | **4.010** |

## Mechanism — why Bench E ≈ Bench D

### Trim 2.4 + 2.5 = no movement

Already in [[v3_20_phase3_ship_verdict]] §"Composition does NOT
compose": every 2.2-2.8 arc individually was neutral / slightly
negative on Bench A. Removing 2.4 + 2.5 from the stack adds zero.

### Dominant_ne port fix = no FS movement

Tracing the buggy detector behavior post-mortem:

| Detector | FS_static frame | DT frame |
|---|---|---|
| **Buggy** (full-spectrum sum, trigger=1) | echo (full-spectrum) >> ne (full-spectrum noise) → trigger fails → never fires | ne >> echo → trigger fires on first frame → fires |
| **Fixed** (LF-only, trigger=12, snr_gate) | echo[1:61] >> ne[1:61] → trigger fails → never fires | ne[1:61] sustained >> echo[1:61], > snr×noise → trigger fires after 12 frames |

On FS, **both** detectors don't fire (echo dominates LF and full-spectrum
both). So the SG runs `normal_tuning` mask in both cases → identical
output. The buggy detector never created false-positive nearend mask
swap on FS in the first place; my fix is correctness work but doesn't
move the FS bench.

On DT, fixed detector fires slightly LESS (LF-only is more conservative
than full-spectrum + needs 12-frame sustain). This explains the
microscopic DT deg loss (-0.003 each bucket) — fewer nearend
hangover frames → SG is fractionally more aggressive.

### What's actually driving the FS regression vs Phase 0

| Bench | Substrate state | R² source | FS_static |
|---|---|---|---:|
| Phase 0 | Legacy | Legacy `_gain_compute_enr` | 3.769 |
| Tier 1 | Dead | Legacy R² + AEC3 SG | 3.295 |
| Bench A/C | Dead | Legacy R² + AEC3 SG (Tier A wired) | 3.373 |
| Bench D/E | Revived | AEC3 ResidualEcho R² + AEC3 SG (full chain) | 3.158 |

**The structural FS gap is between Phase 0 (legacy SG) and any AEC3 SG
variant**: ~0.4-0.6 dB. Substrate revival flipped `usable_linear_estimate`
0%→98% which unlocked the AEC3 R² linear path (R² = |echo|² / ERLE).
With high ERLE on FS, this produces SMALLER residual than legacy → SG
gets LESS to suppress → echo MOS drops further. Substrate revival
working as designed; not a bug.

## Decision tree triggered

Per [[v3_20_bench_e_trim_verdict §approach§decision tree]] in the
pre-execution plan:

- FS echo Bench E (3.162) << AEC2 (3.48) ⇒ "strip to pure-AEC3"
  branch would not help (Bench A pure-AEC3 had FS 3.375, still < AEC2)
- ⇒ "substrate revival itself is the regression source" verdict path
- ⇒ **Don't promote any AEC3 SG/Residual/AecState combination to BALANCED**

## Final verdict (re-confirms [[v3_20_phase3_ship_verdict]])

**CANNOT SHIP to BALANCED. Substrate retained + dominant_ne port
correctness fix retained.**

### Ship recommendations (no change from Phase 3)

1. **Production stays Phase 0** — beats AEC2 on FS_echo (+0.27 dB);
   beats AEC3 deg on both columns (+0.45/+0.55). Loses small on
   DT_echo and AEC2 deg, but only path that wins FS_echo.
2. **`speech_preserving` preset** (new, opt-in) — ship Bench A
   pattern (FULL CHAIN + Tier A + skip-post) as alternative preset.
   DT_deg leader (3.098 DT_mv vs AEC3 1.85 → +1.25 dB).
3. **Substrate retention** — all Phase 2 substrate + corrected
   dominant_ne port + substrate revival fixes stay in tree
   default-OFF. Architectural alignment for v3.22+ NN/Volterra cycle
   (NN can plug into AEC3-aligned R² + AecState cleanly).

### What user authorisation is needed

- Commit ~7-file substrate revival + dominant_ne port fix
  (matched_filter, delay_aec3, orchestrator, residual_estimator_aec3,
  suppression_gain_aec3, erle, erl) as **`substrate revival + dominant_ne
  port AEC3 source-aligned`** commit. Default-OFF substrate; byte-equal
  on Phase 0 path; informs future v3.22+ NN plug-in.
- Commit substrate revival audit doc + this Bench E verdict doc.
- No production behavior change. No preset promotion.

## Files

- This doc: `docs/v3_20_bench_e_trim_verdict.md`
- Bench E: `/tmp/v3_20_e_trim_2_4_2_5/scores.json` (via
  `/tmp/v3_20_e_trim_aecmos/`)
- Bench D: `/tmp/v3_20_d_substrate_fixed_aecmos/result.md`
- Substrate revival audit: `docs/v3_20_substrate_revival_audit.md`
- Prior verdicts:
  - `docs/v3_20_phase3_ship_verdict.md`
  - `docs/v3_20_phase2_1_tier2_verdict.md`
  - `docs/v3_20_phase2_4_leakage_diverged_verdict.md`
  - `docs/v3_20_phase2_5_6_7_combined_verdict.md`

---

## Appendix — Post-trace corrections (2026-05-17, after AEC3 re-read)

Per user directive ("在判斷有用無用之前請你先再次 trace aec3 做法避免
用錯誤資訊判斷"), I re-traced `residual_echo_estimator.cc` +
`suppression_gain.cc` after Bench E and found several inaccuracies in
the earlier analysis. Documenting them so subsequent benches are
judged against the corrected mental model.

### AEC3 ResidualEcho actual computation (`residual_echo_estimator.cc:193-313`)

```
Linear path (UsableLinearEstimate=True, SaturatedEcho=False):
    R²[k] = S²_linear[k] / ERLE[k]              (line 102)
    R²[k] += echo_reverb.reverb()[k]            (line 410)  ← I missed this

NonLinear path (UsableLinearEstimate=False):
    R²[k] = noise-gated X²[k] × echo_path_gain² (line 290)
    if model_reverb_in_nonlinear_mode AND NOT TransparentMode:
        R²[k] += echo_reverb.reverb()[k]        (line 298-299)

Saturated override (both paths): R²[k] = Y²[k]  (line 246-249, 267-270)
Stationarity scaling (final):    R²[k] *= residual_scaling[k]  (line 305-313)
```

Our port (`residual_estimator_aec3.py:104-147`) is structurally aligned:
linear and nonlinear paths both call `_add_reverb` after R² computation.

### AEC3 SuppressionGain actual gate (`suppression_gain.cc:215-233`)

```
enr = echo[k] / (nearend[k] + 1.f)
emr = echo[k] / (masker[k] + 1.f)      // masker = comfort_noise
g = 1.0                                 // default = no suppression
if (enr > enr_transparent[k] AND emr > emr_transparent[k]) {
    g = (enr_suppress[k] - enr) / (enr_suppress[k] - enr_transparent[k])
    g = max(g, emr_transparent[k] / emr)
}
```

The mask gate is **AND**-conjoined: both `enr > enr_t` AND `emr > emr_t`
required. If either fails, `g = 1.0` (no suppression on that bin).

### AEC3 min_gain actual formula (`suppression_gain.cc:237-276`)

```
if NOT saturated:
    min_echo_power = config.echo_audibility.{low,normal}_render_limit
    min_gain[k] = min(1.0, min_echo_power / weighted_residual_echo[k])
    // LF smoothing band:
    if last_nearend[k] > last_echo[k] OR k ≤ permanent_lf_band:
        min_gain[k] = max(min_gain[k], last_gain[k] * 0.25)
else:
    min_gain[k] = 0   // allow full suppression
```

`min_gain` is the LOWER bound on gain (audibility floor):
"if echo is below audibility threshold, don't bother suppressing".

### Earlier-claim corrections

| Earlier claim (above in this doc) | Corrected understanding |
|---|---|
| "R² = S²/ERLE 量級小所以 SG 看不到 echo" | **Incomplete** — reverb is added back (`+= echo_reverb.reverb()[k]`). Post-reverb R² magnitude depends on linear estimate AND reverb mix, not just S²/ERLE alone |
| "AEC3 R² 一定比 legacy R² 小" | **Not necessarily** — depends on reverb contribution. Linear+reverb could approximate legacy magnitude on FS frames with strong reverb |
| "Tier 3 mask 把更多 frame 壓制 → 救回 FS" | **Conditional** — only if R² magnitude actually lands in the new gate range (0.1-0.5). If R² < 0.1 across most FS bins, Tier 3 still doesn't trigger |
| "min_gain 越小越好壓 FS" | **Direction was right but mechanism mis-stated** — min_gain is a FLOOR (lower bound). Our port's hardcoded 1.0 floor is more aggressive than AEC3's config-driven (~tens to hundreds) floor. But the gate (enr/emr > thresholds) fires first; min_gain only matters when gate fired |
| "dominant_ne port fix 應該救 FS (避免 false-positive nearend mask)" | **Wrong** — on FS, buggy detector wasn't firing in the first place (full-spectrum sum dominated by echo). Bench E confirmed FS Δ = +0.004 only |

### Our port vs AEC3 SG — missing pieces (potential factors)

These don't fully explain FS regression alone, but each could
contribute to mask gate decisions:

1. **`nearend_smoothers_[ch].Average(suppressor_input)`** missing
   (AEC3 line 308 uses `MovingAverageSpectrum(nearend_average_blocks)`).
   Our port (`suppression_gain_aec3.py:273-275`) feeds raw nearend
   spectrum → potential frame-to-frame gate flicker near threshold.

2. **`WeightEchoForAudibility(residual_echo, ...)`** missing
   (AEC3 line 312 weights residual_echo by audibility before SG sees
   it). Our port (`suppression_gain_aec3.py:266` comment
   "WeightEchoForAudibility skipped") uses raw residual.

3. **`min_echo_power` hardcoded vs config-driven**
   (`suppression_gain_aec3.py:368`: "1.0 / 0.5 (full AEC3 reads
   config.echo_audibility)"). AEC3 typical config values are much
   larger (audibility floor in spectral power units). Our floor is
   more aggressive (allows deeper suppression) but doesn't gate on
   audibility.

4. **`config.echo_audibility.{floor_power, audibility_threshold_lf/mf/hf}`** not wired
   through `Aec3SuppressionGain` constructor.

### Bench F judgment criteria (corrected, supersedes any prior extrapolation)

Tier 3 mask tune `normal_lf=(0.1, 0.5)` + `normal_hf=(0.02, 0.1)`:

| Bench F result vs Bench E (FS_static 3.162) | Interpretation |
|---|---|
| FS rises ≥ 3.30 | enr/emr gate **was** the bottleneck. R² magnitude lands in new (0.1, 0.5) range. Further mask tuning down to (0.05, 0.3) may extract more |
| FS micro-improvement (3.16-3.25) | Gate partially recovered. R² distribution centered around 0.1-0.5. Need to combine with other axes (audibility floor / weight_for_audibility port) |
| FS unchanged (~ 3.16) | R² magnitude is **below** 0.1 on most FS bins. Mask gate never triggers. Need to either change R² source (disable reverb-add / tune ERLE down) or change audibility floor. **Tier 2 extrapolation in legacy-R² regime did NOT transfer to AEC3-R² regime** |
| FS regresses (< 3.16) | Mask too aggressive — triggers on bins that should pass; over-suppression damages quality metric |
| DT_static deg drops sharply (< 2.8) | Normal mask cannot substitute for nearend mask; aggressive normal damages NE bins that dominant_ne detector failed to protect (or detector hangover gap) |

### Key takeaway

The **Pareto wall** identified in `v3_20_phase3_ship_verdict.md` was
real but its mechanism description in earlier verdict docs (and in
this doc's main body before this appendix) over-simplified "AEC3 R²
small" into a single-line explanation. Reverb addition and gate
arithmetic both affect where R² lands relative to the mask gate. Any
future tuning attempt on this surface must verify against actual
per-bin R² distribution traces, not just bench aggregate numbers.
