# v3.21.10 — FilterMisadjustment direction parity audit (#3a)

**Date**: 2026-05-22
**Status**: AUDIT (pre-trace, pre-implementation)
**Scope**: Compare AEC3 `Subtractor::FilterMisadjustmentEstimator` against our Python `_update_misadjustment_estimator` / `_check_and_apply_misadjustment_scale`. Establish formula + direction parity.

## Source references

### AEC3 reference
- [`docs/aec3_extracts/src/aec3/subtractor.h:99-128`](aec3_extracts/src/aec3/subtractor.h) — `FilterMisadjustmentEstimator` class
- [`docs/aec3_extracts/src/aec3/subtractor.cc:336-358`](aec3_extracts/src/aec3/subtractor.cc) — `Update()` body
- [`docs/aec3_extracts/src/aec3/subtractor.cc:237-249`](aec3_extracts/src/aec3/subtractor.cc) — invocation in `Subtractor::Process`
- [`docs/aec3_extracts/src/aec3/adaptive_fir_filter.cc:713-724`](aec3_extracts/src/aec3/adaptive_fir_filter.cc) — `ScaleFilter()`

### Ours (v3.21.6 baseline + v3.21.x parity sub-patches)
- [`python/modules/config.py:71-92`](python/modules/config.py) — config fields
- [`python/modules/orchestrator.py:190-260`](python/modules/orchestrator.py) — `_update_misadjustment_estimator` + `_check_and_apply_misadjustment_scale`
- [`python/modules/orchestrator.py:2332-2333`](python/modules/orchestrator.py) — call site (post-filter, pre-RES)

## Direction parity table

| Aspect | AEC3 | Ours | Match? |
|---|---|---|---|
| **Observable** | `update = e2_acum / y2_acum` <br/> (error² / mic²; both time-domain block sum-of-squares) | `raw_ratio = mean(\|echo_spec\|²) / (mean(\|far_spec\|²) × ERL)` <br/> (echo-estimate PSD / far-PSD × ERL) | **NO** — different signals; ours has no `e²` term at all |
| **Accumulation** | sum over `n_blocks_ = 4` AEC3 blocks (4 × 4 ms = 16 ms wall-clock) | per-hop, no accumulation | **NO** |
| **EMA update path** | `inv_misadjustment_ += 0.1 × (update − inv_misadjustment_)` <br/> gated on `(update < inv_misadjustment_) OR (overhang_ > 0)` | `smoothed = α × smoothed + (1−α) × raw_ratio` <br/> always; asymmetric α | partial |
| **High-energy trigger** | sets `overhang_ = 4` when `e2_acum > n_blocks × 7500² × kBlockSize` (high-error sustain) | none | **NO** |
| **Energy floor** | requires `y2_acum > n_blocks × 200² × kBlockSize` (`~6.1e-3` int16) before any update | requires `_fp_far ≥ 1e-10` | partial |
| **Fire predicate** | `inv_misadjustment_ > 10.0` → fires when **error >> mic** (DIVERGENCE) | `smoothed < 0.5` → fires when **echo_est < 0.5 × far × ERL** (UNDER-MODELLING) | **NO** — opposite failure mode |
| **Scale magnitude** | `2.0 / sqrt(inv_misadjustment_)` — when `inv = 10`, scale ≈ 0.632; when `inv = 100`, scale = 0.2 | `1.0 / max(smoothed, 1e-6)`, clamped to `[0.5, 2.0]` — when `smoothed = 0.3`, proposed scale ≈ 3.33 | **NO** — opposite direction |
| **Scale direction** | **SHRINK** W (scale < 1) | **GROW** W (scale > 1; clamp at 2.0) | **NO** — opposite |
| **P scaling** | not applicable (FIR filter, no Kalman P) | optional via `filter_misadjustment_scale_p` (default False, AEC3-aligned) | partial |
| **Post-fire reset** | `Reset()` zeros `e2_acum_ / y2_acum_ / n_blocks_acum_ / inv_misadjustment_ / overhang_` | sets `smoothed = 1.0`; arms `hangover_remaining = 100` | partial — hangover is ours, not AEC3 |
| **Output adjustment** | also calls `ScaleFilterOutput(y, scale, e_refined, output.s_refined)` to keep current-frame output consistent | applies to next frame only | **NO** |
| **Per-channel** | per-channel estimator + reset hangover | single estimator | partial (we're 1-channel) |

## Semantic interpretation

**AEC3 fires when**: post-filter error has stayed larger than the microphone signal for ~16 ms of sustained near-end energy. Implication: the refined filter is creating spurious energy (over-adapted, mis-aligned, or unstable). Remedy: shrink `W` to dampen.

**Ours fires when**: the smoothed echo-estimate PSD is less than half of `far_psd × ERL` for ≥ 30 stable frames. Implication: filter is under-modelling the true echo (W shrunk relative to ideal). Remedy: grow `W` to lift response.

**These detect different failure modes.** The legacy formula is silent on AEC3-style divergence (over-adaptation, large `e²`), and AEC3's formula is silent on under-modelling (small echo_est).

## Stress-cohort hypothesis

From v3.21.7 Gate 2 + v3.21.8 trace evidence:
- XRTnTUjU + nVUnxqHLr + MYrVxVEM + xFk7igec are no-clean-convergence stress cases
- Under v3.21.7 partition_summed_x2 ON, refined PBFDKF locally diverges: 37% of XRTnTUjU frames see `e2_coarse < e2_refined AND y2 < e2_refined` (UseRefinedOutput cond2 = "refined diverged")
- `y2 < e2_refined` is mathematically `inv_misadjustment_ > 1.0` (using AEC3's variable). With 4-block accumulation and trigger threshold 10, AEC3 would fire IF the divergence is sustained ≥ 16 ms with error 10× larger than mic

**Prediction (to be tested by trace)**: AEC3 formula fires on stress cluster (XRTnTUjU etc.) where legacy formula stays silent (because legacy looks for under-modelling, not over-adaptation).

If trace confirms: enabling AEC3 parity → shrinks W on the diverged frames → reduces cond2-fire rate → may reduce DT-deg damage on stress without normal regression.

## v3.23 G.0 A.1 prior result (must not be confused)

A.1 (INFORMATIVE_NULL, fired 0/20188 G.0-cohort frames) tested the **legacy formula** on the G.0 cohort (movement-tail / P4 stationary-far). It does NOT prove anything about the AEC3 formula. The 0-fire result is consistent with the legacy formula's under-modelling failure mode being absent in G.0; it says nothing about AEC3-style over-adaptation.

This audit re-opens the parity question with the correct (AEC3) formula and the correct (v3.21.7 stress) cohort.

## Implementation plan (no code changes yet)

### Stage 1 — Trace-only (default-OFF)

Add config flag `use_aec3_filter_misadjustment_parity: bool = False`. When OFF:
- Legacy update + fire path unchanged (byte-equal v3.21.6 anchor preserved)
- AEC3 formula computed in parallel as **counter only**:
  - `_aec3_misadj_trace = {'frames_total', 'aec3_would_fire', 'legacy_did_fire', 'both_fire', 'aec3_only', 'legacy_only'}`
- No `W` change, no `P` change, no observable diff

### Stage 2 — Parity ON (flag ON)

When `use_aec3_filter_misadjustment_parity = True`:
- Legacy update + fire path BYPASSED (do not stack)
- AEC3 4-block accumulator + e²/y² ratio + `> 10` threshold + `2/sqrt` scale
- ScaleFilter applied as AEC3 (W shrunk, P untouched per `scale_p=False` default)
- Existing safety gates (`epc_active`, `main_paused`, `_filter_converged`) kept as outer guards — AEC3 doesn't have them, but we keep to avoid firing during transients (matches our v3.18 B.5 framing for legacy)

### Stage 3 — 12-case cohort gate

Cohort:
- 4 stress: XRTnTUjU_DT_static, nVUnxqHLr_DT_static, MYrVxVEM_DT_static, xFk7igec_DT_static
- 4 normal DT: jtYTdZm3_DT_movement, wVYSGV_DT_movement, ZJYUt0O0A_DT_movement, sUQrHEPA_DT_static
- 3 normal FS: FS-9xjhi, FS-qNvSMyU, FS-xQEUtY2, FS-mv-0I0XMl3M
- 1 NE: NE-014AzuqPZku

Bench:
- A_baseline = v3.21.6 baseline (all parity flags OFF including the new one)
- B_v3_21_10 = baseline + `AEC_AEC3_MISADJ_PARITY=1` (this flag ON, all others OFF)
- Standard preset/filter/CNG settings; HPF lock honoured

### Stage 4 — Decision rule (PER USER)

| Outcome on 12-case | Decision |
|---|---|
| AEC3 fires on stress + reduces DT-deg regression there + no normal regression | KEEP v3.21.10 ON in next 12-case stack (test partition_summed_x2 on top) |
| AEC3 fires on stress but DT-deg regression unchanged or worsens | CLOSE (mechanism present but not curative) |
| AEC3 fires on normal cases too and regresses them | CLOSE — direction parity correct but gate insufficient |
| AEC3 doesn't fire on any case in cohort | CLOSE as no-op; move to #1d / #1e downstream signals |

## Forbidden during this arc

- Do NOT revive v3.21.8 UseRefinedOutput as ship candidate
- Do NOT add shadow-converged precondition (deferred to v3.22)
- Do NOT pivot to v3.22 PBFDKF-specific logic
- Do NOT run /simplify or branch cleanup before this parity target closes
- Do NOT stack with partition_summed_x2 / UseRefinedOutput / coarse_e2_td_parity flags until v3.21.10 12-case gate passes
- Do NOT run 800-case until 12-case Stage 4 verdict justifies it
