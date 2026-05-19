# v3.18 Phase D-γ Closeout (2026-05-16)

**Verdict**: CLOSED — CANNOT SHIP. Substrate retained for v3.19+ reconsideration.

## Scope tested

Per-band NE pathway port from AEC3 — combination of:
- D.1: `SubbandNearendDetector` (structural HF-quiet/voice-loud cue, echo-agnostic)
- D.2: per-bin mask profile tables (`nearend_params_` / `normal_params_` LF→HF interp)
- D.3: atomic binary swap, OR asymmetric per-frame 3-state (FS / NE / uncertain)
- B1: `DominantNearendDetector` (LF energy ratio, echo-aware, hysteresis trigger/hold)
- Combined: `_ne_combined_state = subband OR dominant`

## Final 60-case A/B (commit pending, all four configs flag-OFF byte-equal PASS)

Baseline = production `pathDf` (BALANCED / fl=832 / cng=True / max_delay=1024).

| Config | FS_static Δecho | FS_movement Δecho | DT_static Δdeg | DT_movement Δdeg | NE Δdeg |
|---|---:|---:|---:|---:|---:|
| dom_only_asym | +0.018 | +0.016 | **−0.034** | +0.011 | −0.001 |
| combined_asym | +0.005 | +0.002 | −0.012 | +0.016 | −0.001 |
| dom_only_bin | **+0.154** | +0.043 | **−0.057** | **−0.048** | −0.005 |
| combined_bin | +0.148 | +0.033 | **−0.062** | −0.036 | −0.005 |

Phase D-γ hard bar (per `docs/v3_18_phase0_decision.md`):
- NE Δdeg ≥ **+0.010** (PRIMARY) — **NONE met**
- DT Δdeg ≥ +0.005 across both buckets — **NONE met**
- FS Δecho ≥ −0.010 — all four PASS
- Cohort tail (qNvSMyU) Δecho ≥ −0.05 — not violated

**Kill condition triggered (§0.4)**: both NE failure AND DT regression.

## Mechanism findings

1. **Binary mode**: big FS gain (+0.15 dB FS_static) but big DT crash (~−0.06 dB Δdeg both DT buckets). Same trade-off as D.3 initial.
2. **Asymmetric (per-frame 3-state)**: blunts both directions — DT damage halved (−0.034 → −0.012 with subband added) but FS gain also shrinks (+0.15 → +0.005).
3. **Subband-or-Dominant aggregate**: in asymmetric mode adding Subband softens both directions because mode is gated AND'd against `effective_dt`; in binary mode Subband adds little (Dominant already covers cohort tail).
4. **NE detector accuracy not the bottleneck**: Dominant's echo-aware + hysteresis design caught the FS misfire problem (smoke: FS misfire 19.9% → 2.3%), but the resulting per-bin mask profile (`normal_params_`) is still too aggressive vs our legacy `ne_confidence` interp.

## Why it fails: physics summary

Our system's RES path is co-tuned with the legacy scalar `effective_dt` →
`(enr_t_fs, enr_s_fs)` / `(enr_t_ne, enr_s_ne)` mix. AEC3's `normal_params_`
(LF anchor `(0.3, 0.4, 0.3)`, HF anchor `(0.07, 0.1, 0.3)`) is more
aggressive than what our pipeline survives without DT damage. The
`spectral_g_min` floor and the v3.13 E5 saturation arc (closed CANNOT SHIP,
same trade-off slope) confirm: our pipeline cannot absorb a sharper FS
suppression curve without DT speech leakage.

This matches the v3.13 E5 finding (4 sub-variants S2/S3/S4a/S4b all on the
DT-FS Pareto line, slope ~0.5 dB DT loss per +1 dB FS gain). Both arcs are
the same physics wall: at this point in v3.18 the linear filter + RES
parametrisation is the load-bearing layer; mask-shape swaps trade one
bucket for another.

## Substrate retained (default OFF, gated by `_mask_profile_swap_enabled`)

- `SubbandNearendDetector` compute path in `ResFilter._stage_residual_model`
- `DominantNearendDetector` compute path + hysteresis counters
- `_normal_mask_profile` / `_nearend_mask_profile` per-bin LF→HF interp builders
- `_ne_combined_state` aggregate
- Binary and asymmetric swap modes wired
- 8 env-var overrides + AecConfig fields for tuning sweeps in future arcs

Useful for: (a) v3.19+ retry once Phase A (shadow NLMS) + Phase B
(filter misadjustment) recover linear-filter quality (cohort tail
Δecho gain may absorb the DT cost); (b) v3.20+ Reverb-port arc — AEC3
reverb estimate also feeds the residual model, which is the path AEC3
uses to make `normal_params_` work without DT damage.

## Disposition

- Phase D-γ CLOSED — no ship.
- Substrate retained: do **not** revert. The detectors + profile tables are
  default-OFF and can be re-tuned (anchor sweep + per-band ENR weighting)
  if Phase A/B shift the operating point.
- v3.18 cycle proceeds to Phase F (`EchoPathVariability` event
  classification + cascade reset) per `docs/v3_18_plan_revision_2026_05_15.md`.

## Cross-references

- `docs/v3_18_d1_subband_ne_detector.md` — D.1 substrate
- `docs/v3_18_d2_mask_profile_substrate.md` — D.2 substrate
- `docs/v3_18_d3_mask_shape_swap.md` — D.3 binary mechanism + first 60-case
- `docs/v3_18_phase0_decision.md` — Phase D-γ hard bar definition
- `docs/v3_18_plan_revision_2026_05_15.md` — option-2 cycle scope
- `docs/v3_13_e5_closure.md` — comparable DT-FS Pareto wall, closed
