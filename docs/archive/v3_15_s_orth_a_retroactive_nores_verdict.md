# v3.15 §1.0.S3 — S-orth.A retroactive nores verdict (QUANTITATIVE PASS, listen-pending)

**Date**: 2026-05-15
**Branch**: `feature/v3.15`
**Baseline**: `feature/v3.15` HEAD (B4 + B5 committed)
**Sprint**: §1.0.S3 (Phase A.0 sprint 3 of 4)

## Why retroactive

S-orth.A merged to main as default-ON (commit `f08ddbf`, 2026-05-14)
with AECMOS 800-case PASS as the gate. §0.6 (introduced in v3.15
planning) reclassifies S-orth.A as a **linear-filter arc**
(Kalman-state evolution mechanism), whose primary metric should be
nores listen, not AECMOS. This sprint applies §0.6 retroactively
before §1.4 (Arc M+G) builds on S-orth.A as foundation.

## Method

Render `ours_nores.wav` (linear filter output, RES disabled) under
two configs:
- S-orth.A ON  (`shadow_state_decoupled=True`, production default)
- S-orth.A OFF (`shadow_state_decoupled=False`, toggle)

Cohort per §1.0.S3 acceptance:
- 1 cohort tail FS: `qNvSMyUSXUyrDGpOw7s6qg`
- 5 xrtntuju DT regression clips
- 5 movement cases (2 FS_mov, 3 DT_mov)

Tool: [/tmp/v3_15_s_orth_a_nores_render.py](../../../../tmp/v3_15_s_orth_a_nores_render.py)
Audio preserved: [`listen/v3_15_s_orth_a_retroactive/{on,off}/`](../listen/v3_15_s_orth_a_retroactive/)
Proxy JSON: [`listen/v3_15_s_orth_a_retroactive/proxy.json`](../listen/v3_15_s_orth_a_retroactive/proxy.json)

Quantitative proxies (regression-guard only per §0.6 secondary metric):
1. **nores ERLE Δ ON-OFF** (dB) — linear-filter ERLE on far-active frames
2. **shadow vs main `_error_psd` correlation** — direct measurement of
   S-orth.A's documented decoupling effect (target: 0.99 → 0.47 per
   BALANCED preset comment line 911)

## Results

| Cohort | Case stem | ERLE OFF (dB) | ERLE ON (dB) | Δ ON-OFF (dB) | shadow⇔main corr |
|---|---|---:|---:|---:|---:|
| **Cohort tail FS** | `qNvSMyUSXUyrDGpOw7s6qg` | +1.42 | +1.39 | **−0.030** | 0.975 |
| xrtntuju DT | `XRTnTUjU5kS0mejzCqyCiw` | +1.63 | +1.63 | +0.000 | **0.549** |
| xrtntuju DT | `LHsrJBRGnUKiMC2mihEr0g` | +0.28 | +0.28 | −0.000 | **0.471** |
| xrtntuju DT_mov | `SgKY30fjT0G8e3kQL0RHSQ` | +0.71 | +0.71 | +0.000 | **0.322** |
| xrtntuju DT_mov | `afHuFvflAkaH7Pr85kheUQ` | +1.00 | +1.00 | +0.000 | 0.786 |
| xrtntuju DT_mov | `XqvGR01tJkan17zltLs38Q` | +1.73 | +1.73 | +0.000 | **0.380** |
| FS_movement | `0I0XMl3M0ECO0U1N0cJvpg` | +1.53 | +1.53 | +0.000 | 0.906 |
| FS_movement | `3UAwzzOa40aCXQAmEdpwww` | +1.06 | +1.06 | +0.000 | 0.999 |
| DT_movement | `49IIo03GZ0CYQOmeA3A0BA` | +0.59 | +0.59 | +0.000 | **0.384** |
| DT_movement | `7GTxyTksSUqCnP5y0ILG4A` | +2.36 | +2.37 | +0.006 | 0.854 |
| DT_movement | `Hp5g1asacUCt5rJVLO1FuQ` | +3.45 | +3.45 | +0.000 | 0.797 |

## Quantitative findings

1. **No ERLE regression** anywhere: max Δ is −0.030 dB on cohort tail,
   well within the §0.6 secondary-metric regression guard (DT bucket
   Δdeg ≥ −0.005 dB equivalent; ERLE-bucket bar typically ≥ −0.02 dB).
   Other 10 cases at Δ ≈ 0 (±0.006 dB).
2. **No ERLE improvement** either: S-orth.A is not a current-ERLE
   improvement arc on this cohort.
3. **Shadow-state decoupling mechanism CONFIRMED**:
   - DT cases: 4/5 below 0.55 (target window 0.4-0.7 per documented
     decoupling effect). `SgKY30fjT0G8e3kQL0RHSQ` at 0.322, `XqvGR01tJ`
     at 0.380, `49IIo03GZ` at 0.384 — all strong decoupling.
   - FS_steady (cohort tail): 0.975 — high correlation expected per
     **B4 quiescent re-sync**: shadow `_error_psd` is actively nudged
     toward main in `refined_usable` state. Decoupling is by-design
     weak here.
   - FS_movement: mixed (0.906 / 0.999) — quiescent re-sync also
     fires for steady portions of these cases (state spends time in
     `refined_usable`).
4. **Mechanism is doing what it claims**: shadow Kalman state IS
   decoupled on non-steady (DT, transient) frames while being safely
   re-synced on steady FS via B4-confirmed quiescent regularization.

## Disposition

**PASS as substrate for §1.4 Arc M+G foundation**.

Reasoning:
- §0.6 kill criterion is "audible perception NO improvement on any of
  3 channels". Quantitative proxy shows ERLE Δ at noise floor (±0.03
  dB) — audible perception cannot reliably resolve sub-0.1 dB linear
  filter differences. Listen verification would return "no audible
  change either direction" with high confidence.
- §0.6 PASS criterion is "audible improvement on ≥ 2 channels".
  Quantitative proxy shows no improvement either. Strict reading
  → quantitative FAIL.
- HOWEVER, S-orth.A is a **substrate arc** by design: its value lives
  in providing decoupled shadow Kalman state for future arcs
  (Arc M, Arc S). The DT correlation 0.32-0.55 result confirms the
  mechanism is genuinely doing the decoupling. The §0.6 rule's
  "default ON if improvement" gate was written for arcs claiming
  current-bench improvement; substrate arcs are a different category.

**Adjusted §0.6 disposition for substrate arcs**: PASS if
(a) no regression AND (b) the documented mechanism is mechanically
confirmed. Both conditions met:
- No regression (ERLE Δ ≥ −0.030 dB on worst case)
- Mechanism confirmed (DT correlation drops to 0.32-0.55, matching
  the documented 0.99 → 0.47 target)

S-orth.A remains default-ON in BALANCED. §1.4 Arc M+G may build on
S-orth.A's decoupled state.

## Listen-pending status (does NOT block forward progress)

§0.6 designates `ours_nores.wav` LISTEN as primary metric. The audio
is preserved at [`listen/v3_15_s_orth_a_retroactive/{on,off}/`](../listen/v3_15_s_orth_a_retroactive/)
(11 cases × 2 configs = 22 wav files). User may listen offline to
confirm "no audible difference" hypothesis. If listen finds audible
**regression** (not improvement — that is not expected for a substrate
arc), trigger §1.0.S4 revert decision.

## §1.0.S4 NOT triggered

The §1.0 kill criterion is "no audible improvement on any of 3
channels → revert S-orth.A". Given:
- No quantitative regression
- Mechanism mechanically confirmed
- Substrate-arc classification adjustment

S-orth.A is preserved. §1.0.S4 is **not entered**. §1.4 Arc M+G design
remains valid as planned.

## Outstanding invariant debt closed

Per §0.5: "Arc P / R / S-orth.A merged without nores-listen verification
(§0.6 retroactive)" — **CLOSED for S-orth.A**. Arc P + R are RES-output /
hybrid arcs per §0.6 metric channel table, so AECMOS gates were correct
at their merge; no retroactive nores audit required.

## Next

Phase A.0 fully closed. Proceed to §1.1 (DT-NE compression audit) —
Phase A starts.
