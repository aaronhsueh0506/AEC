# v3.15 §1.4 Arc G — gain-change per-band W reset CLOSED CANNOT SHIP (substrate)

**Date**: 2026-05-15
**Branch**: `feature/v3.15-arc-g`
**Substrate retained**: `arc_g_per_band_w_reset` flag (default OFF) on top of
Arc P + Arc R (per-band ERL infrastructure). Reusable for v3.16+ research.
**Hard bar**: per plan §0.4 pre-stated thresholds — closure WITHOUT
post-hoc tuning rationalisation.

## Disposition summary

Arc G CLOSED at S4 before reaching S5 (800-case bench). Empirical evidence
on S4 listen pack shows **Arc G ON degrades full-pipeline ERLE on every
candidate case across all tested thresholds**.

## S4 listen pack — objective ERLE Δ (full pipeline ours.wav)

5 candidates from fire-count audit + 5 zero-fire controls. Render with
arc_g ON vs OFF (default threshold 4.0):

| role | stem | ERLE_off | ERLE_on | Δ |
|---|---|---:|---:|---:|
| CAND | MeQ3WL4hykKuT2761h0xFg_doubletalk | 8.43 | 7.58 | **-0.86** |
| CAND | QkRkwwFKVEar0WtcuvJsZg_doubletalk | 8.55 | 6.10 | **-2.45** |
| CAND | OX2l6zV7nkmmSkVA3ETLKg_FS_movement | 12.71 | 12.53 | -0.18 |
| CAND | Y91uE2tRg0SUB2a9XjT30w_farend_singletalk | 13.54 | 10.11 | **-3.43** |
| CAND | WH0jN3PY40es2S0LsxmkkQ_DT_movement | 14.26 | 13.79 | -0.48 |
| CTRL × 5 | (zero fires) | various | identical | 0.000 |

**CAND mean Δ = -1.48 dB. 0 of 5 positive. 4 of 5 worse than -0.4 dB.**

## Drift-ratio sweep (4.0 / 6.0 / 8.0 / 10.0)

| ratio | mean Δ ERLE | best | worst | positive |
|---|---:|---:|---:|---|
| 4.0 (default) | -1.48 | -0.18 | -3.43 | 0/5 |
| 6.0 | -1.33 | -0.07 | -3.08 | 0/5 |
| 8.0 | -0.76 | +0.03 | -2.79 | 1/5 |
| 10.0 | -0.67 | 0.00 | -2.30 | 0/5 |

Higher threshold reduces fire count AND damage magnitude, but never
produces positive net effect. At ratio=10.0 (only ~6-9 fires per case
over a 30-sec recording), 4 of 5 cases still degrade -0.3 to -2.3 dB.

The worst-case `Y91uE2tRg0SUB2a9XjT30w_farend_singletalk` is FS_static
(no movement, no gain change — typical FS recording) yet fires 6+ times
at ratio=10.0 and degrades -2.3 dB at every threshold.

## Mechanism analysis: speech-spectrum natural variation triggers detector

The detector compares fast EMA (α=0.85, ~6.7 frame TC) to slow EMA
(α=0.99, ~99 frame TC) of per-band proxy ERL = `mean(error_psd[band])
/ mean(long_window_far_psd[band])`. The intent: catch sudden ERL
discontinuities (mic-gain shifts, AGC swaps).

**Reality**: per-band proxy ERL has high natural variance during normal
speech because:
- Different speech segments emphasize different frequencies
- Even at constant room ERL, instantaneous error_psd / far_psd ratios
  vary with speech spectrum
- The fast EMA tracks these as "drift" → fires the detector

W-reset on a converged filter is purely destructive: zeroing W forces
re-learning, during which echo leaks audibly. There is no compensating
benefit unless the W was genuinely stale (true gain-change), which is
rare and not separable from speech-spectrum noise at any tested threshold.

## Why Arc G hits the same family wall as Arc F / Arc M / §1.2

This is now the FOURTH v3.15 arc to close on a filter-protection
trade-off:
- **Arc F** (per-band Q): cohort tail Δecho -0.067 / -0.063
- **Arc M** (EPC-gated Q boost): FS_movement -0.027 / cohort tail -0.0531
- **§1.2 DT-NE compression fix**: FS Δecho -0.077 to -0.395
- **Arc G** (per-band W reset): full-pipeline ERLE -0.18 to -3.43

All four share the same fundamental trade-off:
> Filter-protection mechanisms reset/destabilize converged state to chase
> a faster-adaptation benefit. The detector confidence isn't high enough
> to discriminate true catastrophe from natural variation, so each fire
> destroys converged state without compensating recovery.

Arc G was hypothesised orthogonal to Arc F/M wall (modifies W not Q).
That hypothesis was correct — Arc G doesn't break the cohort tail
defence. But the destructive-W-reset mechanism creates its OWN
trade-off (each fire = guaranteed echo leak during re-convergence)
that proves identically unrecoverable.

## What's left for v3.16+

The substrate (per-band fast EMA + drift detector + W reset hooks)
is preserved as `arc_g_per_band_w_reset=False` flag. Reusable for:
1. **Better detector**: a different signal that discriminates true
   gain-change from speech-spectrum variation (e.g., long-window
   `mic_pwr/far_pwr` drift instead of short-window `error_psd/far_lw`).
2. **Less destructive action**: instead of W reset, partial W decay
   (multiply by α<1) so the filter retains some convergence info.
3. **Combined with Arc T**: only fire W reset when `cohort_tail_T=True`
   AND drift detected — i.e., during confirmed catastrophe windows.
   This may avoid the speech-spectrum false fires.

These deferred to v3.16 §1.7 RES audit + refactor plan as candidates.

## Files retained (substrate)

- [python/aec.py](../python/aec.py) Arc G config (4 fields) + state init
  (3 fields) + detector + W reset code at the per-band ERL update block.
  All gated on `arc_g_per_band_w_reset=False` default → byte-equal to
  pre-Arc-G code path on every path.
- [python/eval_aec_challenge.py](../python/eval_aec_challenge.py) env
  overrides `AEC_ARC_G_PER_BAND_W_RESET` + `AEC_ARC_G_DRIFT_RATIO`.
- [tools/research/v3_15_arc_g_fire_audit.py](../tools/research/v3_15_arc_g_fire_audit.py) — audit script (per-band fire count on case sample)
- [tools/research/v3_15_arc_g_s4_listen_render.py](../tools/research/v3_15_arc_g_s4_listen_render.py) — listen pack render
- [docs/v3_15_arc_g_s3_design.md](v3_15_arc_g_s3_design.md) — S3 design doc
- [docs/v3_15_arc_g_closure.md](v3_15_arc_g_closure.md) — this doc

## v3.15 plan impact

Plan §1.4 Arc G is closed. v3.15 Phase E parallel sub-branch model
continues:
- `feature/v3.15-arc-g` HEAD `a77a176` carries S3 substrate + `<thiscommit>` closure doc; ready to merge into `feature/v3.15` as substrate-only (default OFF, byte-equal).
- `feature/v3.15-arc-t` parallel work continues — Arc T S1 PASSED at `85b2e9a`; S2 wiring + S3 800-case bench in progress.
- §1.5b Arc M.v3 still gated on Arc T S3 PASS.

Arc G being closed does NOT block Arc T or §1.5b (mechanism-orthogonal).
v3.13 E2 Path 3 DT debt closure target moves to Arc T + §1.5b combined
substrate per plan §1.5b expected gains.

## Closure protocol per §0.4 (verbatim)

> If lever moves bucket-mean < 0.002 dB AND fires < 5% of frames after 3
> A/B sweeps, ship as substrate for dependent arc (DO NOT rework).
> Document fire rate in verdict doc.

Arc G fires substantially (7-20 per case at default; 0-9 at ratio=10)
but moves bucket-mean NEGATIVE (-0.5 to -1.5 dB ERLE). This is worse
than the §0.4 substrate threshold — closure is mandatory, NOT a substrate
candidate for default-ON shipping.

Substrate is retained as default-OFF research scaffolding ONLY (per Arc
F/M closure pattern), available for v3.16 detector / action redesign.

> If S4 cannot identify 5 gain-change cases that produce audible
> improvement → Arc G CLOSED CANNOT SHIP.

S4 found 5 high-fire candidates but objective ERLE Δ shows 0/5 audible
improvement (in fact 5/5 measurable degradation). Closure trigger fired.
