# v3.15 §1.4 Arc G — gain-change per-band W reset (S3 design + impl)

**Date**: 2026-05-15
**Branch**: `feature/v3.15-arc-g`
**Sprint**: §1.4.S3 (design + impl + byte-equal sanity)
**Substrate retained**: `arc_g_per_band_w_reset` flag (default OFF)

## Why Arc G

Arc M (§1.4 prior) and Arc F (§1.6) closed CANNOT SHIP because both
modify Kalman Q (process noise) — any uniform-time HF Q boost incurs
proportional steady-state misadjustment, breaking cohort tail Δecho
(catastrophe class qNvSMyU is HF-sensitive). EPC-gating Arc M was
intended to escape but failed: PathChangeRegimeHandler 6-gate AND
fires precisely IN the cohort tail catastrophe window, so EPC-gated Q
boost = boost during catastrophe = same wall.

Arc G targets a **physically distinct** failure mode: sudden mic-gain
or path-discontinuity changes that move the room ERL in one band but
not others. Mechanism:
- Modifies `filter.W` (filter weights) instead of `Q` → no steady-state
  misadjustment cost
- Detector is per-band ERL drift (not movement / not EPC) → orthogonal
  to PathChangeRegimeHandler trigger

Per Arc M closure doc: *"Arc G (gain-change per-band W reset) targets
a DIFFERENT mechanism: discrete W-reset per affected band when ERL
drift is detected. This does NOT modify Q (steady-state stability
untouched) and does NOT fire during normal EPC windows (gain-change
detector is orthogonal to EPC). So the Arc F/M wall does not apply
to Arc G."*

## Design

### Mechanism: fast-vs-slow per-band ERL EMA + drift-gated W reset

Maintain a **fast** per-band ERL EMA alongside Arc P's existing
**slow** EMA. When `max(fast,slow) / min(fast,slow) >= drift_ratio`
on band `b` AND PathChangeRegimeHandler is NOT asserting EPC AND the
band is not in cooldown → zero `filter.W[:, bin_range_for_band_b]`
and start a per-band cooldown.

**Mathematical justification for cooldown duration**: after W reset,
the residual `error_psd` shifts to reflect the new ERL → the slow EMA
(α=0.99) needs ~99 frames (~1.6 s at 256-sample hop) to track ~63%
of the new value. The fast EMA (α=0.85, ~6.7 frame time constant)
catches up in ~30 ms. To prevent immediate re-fire from the
fast/slow gap during the slow EMA's catch-up window, cooldown is set
to 100 frames (~1.6 s). After cooldown, projected fast/slow ratio for
a 10× true-ERL change is:
- slow ≈ 0.367 × old + 0.633 × new ≈ 0.67 (when new = 1.0, old = 0.1)
- fast ≈ 1.0 × new ≈ 1.0
- ratio ≈ 1.49 < 4.0 (no re-fire) ✓

### Hyperparameters (defaults)

| Field | Default | Rationale |
|---|---|---|
| `arc_g_per_band_w_reset` | `False` | Master flag; default OFF, byte-equal flag-OFF |
| `arc_g_fast_alpha` | `0.85` | Time constant ~6.7 frames (~100 ms); fast enough to detect sudden gain changes, slow enough to avoid speech-spectrum false positives |
| `arc_g_drift_ratio` | `4.0` | 6 dB asymmetry; conservative to avoid speech-spectrum false positives (typical per-band ERL variance during normal speech is ~2× / 3 dB) |
| `arc_g_cooldown_frames` | `100` | ~1.6 s; per math above ensures no re-fire after slow EMA catches up |

S4 will tune these against gain-change cohort + 800-case false-positive
audit.

### Gating chain (mutually exclusive with PathChangeRegimeHandler)

```
if (Arc P enabled                        # f3_1_per_band_erl_adaptive
    AND filter converged                 # _filter_converged
    AND residual estimator initialized): # res._residual_est warmed up
    update slow per-band ERL EMA
    if Arc G enabled:                    # arc_g_per_band_w_reset
        update fast per-band ERL EMA
        if (EPC NOT asserting             # _epc_render_forced_remaining <= 0
            AND band not in cooldown      # _arc_g_cooldown[bi] == 0
            AND non-trivial signal):      # slow > 1e-6 AND fast > 1e-6
            if max/min ratio >= drift_ratio:
                zero filter.W[:, band_bins]
                set cooldown[bi] = 100
                snap fast EMA to slow (prevent immediate re-fire)
```

**EPC-quiet gate** is load-bearing: it ensures Arc G NEVER fires during
the catastrophe window where PathChangeRegimeHandler is applying its
6-gate defence. This preserves the cohort tail invariant
(`feedback_aec_code_review_accuracy`).

## Plug-in points (cite line numbers)

All changes in `python/aec.py` Arc P per-band ERL update block:

| Location | Change | Lines |
|---|---|---|
| `AecConfig` (after Arc M flag) | 4 new config fields | [aec.py:847-872](../python/aec.py#L847) |
| `AEC.__init__` (after `_per_band_erl` init) | 3 new state fields | [aec.py:5605-5618](../python/aec.py#L5605) |
| Per-band ERL update (inside Arc P gate) | Arc G fast EMA + detector + W reset | [aec.py:7002-7048](../python/aec.py#L7002) |
| `eval_aec_challenge.py` (after Arc M env override) | 2 env overrides | [eval_aec_challenge.py:272-281](../python/eval_aec_challenge.py#L272) |

## Diagnostic surface

`self._arc_g_fire_count[band]` — int64 counter, total fires per band
per stream. S5 800-case bench will dump this for fire-rate audit.

(Optional follow-up): wire to `AecStats` for per-frame trace if
detailed timing needed during S4.

## §1.4.S3 acceptance bars (this sprint)

1. Byte-equal flag-OFF sanity: 5-case (`0KjzXA3g20qsd8zmSekADw`,
   `0I0XMl3M0ECO0U1N0cJvpg`, `49IIo03GZ0CYQOmeA3A0BA`,
   `014AzuqPZku2004NbTTmcA`, `021g8E0mLEWnaPGZo209gA`),
   atol=0.0 — **PASS 10/10 files MD5-identical** (5 cases × ours +
   ours_nores). Validation script: render baseline by stashing WIP
   then `eval_aec_challenge.py /tmp/arc_g_be_subset/`, render
   treatment by popping WIP and same command. Diff via Python md5.
2. Arc G consumes Arc P infrastructure (verified by code reading):
   detector + W reset block is INSIDE the Arc P
   (`f3_1_per_band_erl_adaptive`) gate at [aec.py:6982](../python/aec.py#L6982).
3. EPC-quiet gate active: detector skipped when
   `_epc_render_forced_remaining > 0` (PathChangeRegimeHandler
   catastrophe window).
4. Default-OFF byte-equal verified through MD5 hash equality.

## §1.4.S4 plan (next sprint, NOT this one)

S4 must:
1. **Identify 5 gain-change cases**: run Arc G with low threshold
   (`arc_g_drift_ratio=2.0`) on a 50-100 case sample, sort by
   `_arc_g_fire_count` per band, pick top 5 with audible perceptual
   character of mic-gain shift.  Likely candidates source pool:
   - Movement cases (FS_movement / DT_movement) where ERL changes
     mid-stream
   - PathChangeRegimeHandler-trigger cases (the 7/800 cohort tail
     class) — INTERSECT with NON-EPC-active windows so Arc G can fire
   - Arc D worst-DT cases (DT_movement subset where state transitions
     hint at gain shift)
2. **Tune `arc_g_drift_ratio`** on 800-case false-positive audit:
   pick the largest ratio that still fires on ≥ 4 of 5 gain-change
   cohort, while keeping false-positive frame fraction < 5% on
   non-gain-change cases.
3. **nores listen on the 5 gain-change cases**: render ON vs OFF
   `ours_nores.wav`, A/B listen for audible ERL recovery improvement
   (per §0.6 linear-filter-arc rule, nores is PRIMARY metric channel).

## §1.4.S5 plan (after S4)

1. **800-case A/B**: Arc G ON vs OFF on `feature/v3.15-arc-g` HEAD
   (NOT vs main). Standard config: `--preset balanced --filter 832
   --cng -j 4`.
2. **Hard bars (per §0.6 linear-filter rule)**:
   - nores listen on cohort tail (`qNvSMyU…`) + xrtntuju 5-clip + 5
     gain-change cases shows audible improvement on ≥ 2 of 3 channels
   - 800-case AECMOS regression-guard: DT Δdeg ≥ -0.005, FS Δecho ≥
     -0.020, **cohort tail Δecho ≥ -0.05** (cohort defence preserved)
3. **Cohort tail fire-rate audit**: dump `_arc_g_fire_count` for
   cohort tail cases. Per design, EPC-quiet gate should make
   fire_count near-zero in the catastrophe window. Verify empirically.

## Closure protocol per §0.4

If S4 cannot identify 5 gain-change cases that produce audible
improvement → Arc G CLOSED CANNOT SHIP. Substrate retained as flag
(default OFF) for future arc to consume (per Arc F/M closure pattern).

If S5 800-case shows cohort tail Δecho < -0.05 (catastrophe defence
weakened) → ALSO CLOSED CANNOT SHIP regardless of S4 listen result.
Hard invariant trumps audible improvement.

## Files modified (this sprint)

- `python/aec.py` (+64 lines): config, state init, detector + W reset
- `python/eval_aec_challenge.py` (+9 lines): env overrides
- `docs/v3_15_arc_g_s3_design.md` (this doc, new file)
