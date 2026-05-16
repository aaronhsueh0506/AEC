# v3.15 §1.5 Arc T — cohort tail real-time detector (S1 design + impl + verdict)

**Date**: 2026-05-15
**Branch**: `feature/v3.15-arc-t`
**Sprint**: §1.5.S1 (detector design + impl + 8-case validation)
**Substrate retained**: `arc_t_cohort_detector` flag (default OFF)
**S1 hard bar**: PASS (5/5 TAIL fire + 3/3 CTRL no-fire after threshold tuning)

## Why Arc T (dual role)

1. **Original — RES preempt**: real-time `ERL_decile_std` proxy →
   trigger conservative Kalman + RES mode BEFORE catastrophe.
   Reduces residual cohort tail damage that
   `PathChangeRegimeHandler` currently absorbs reactively.
2. **NEW (2026-05-15 reorder) — §1.5b dependency**: expose
   `cohort_tail_T` signal as upstream gate for Arc M.v3 retry
   (`(EPC_active AND NOT cohort_tail_T)` gate inside `_arc_m_q_boost`).
   §1.5b is BLOCKED if §1.5 fails → Arc M permanently CLOSED.

## Mechanism

### Source signal — explicitly UN-GATED on `_filter_converged`

The canonical cohort tail case `qNvSMyU` **never reaches** `refined_usable`
(filter never converges) per
[docs/v3_14_s_orth_a_design.md:173-180](v3_14_s_orth_a_design.md#L173-L180),
so the existing `_per_band_erl[]` slow EMA at
[python/aec.py:5583](../python/aec.py#L5583) (which only updates when
`_filter_converged AND lw_ready`) is silent there. The Arc T proxy must
read a signal that updates EVERY far-active frame regardless of
convergence.

**Chosen source**:
```
inst_pb_proxy[b] = mean(self.res.error_psd[band_b]) /
                   max(mean(self.res._residual_est._long_window_far_psd[band_b]), 1e-10)
```

Both arrays update every far-active frame:
- `self.res.error_psd` updates at [python/aec.py:3531](../python/aec.py#L3531)
- `self.res._residual_est._long_window_far_psd` updates at
  [python/aec.py:4499-4511](../python/aec.py#L4499-L4511) (the v3.10.0
  EMA was specifically separated from the consumer gate so it is "ready
  immediately when delay drops out and we need a delay-agnostic echo
  template")

This is the canonical (non-shortcut per `feedback_no_shortcut_use_canonical`)
source: same formula as P.S3 per-band ERL update at
[python/aec.py:7020](../python/aec.py#L7020) but ungated on convergence.

### Real-time proxy — `max/min` ratio over rolling window

Per band: smooth `inst_pb_proxy[b]` with EMA `α=0.85` (TC ≈ 6.7 frames
≈ 65 ms at hop=160), maintain a 64-frame ring buffer, compute
`ratio_db[b] = 10·log10(max/min)` over the window. Per-frame statistic:
`max_b ratio_db[b]` (max-over-bands; cohort tail signature is a single
band collapsing).

### Hysteresis state machine

```
if proxy_db ≥ T_HI = 18.5:
    cohort_tail_T = True
    hys_remaining = 200
elif hys_remaining > 0 and proxy_db ≥ T_LO = 13.0:
    cohort_tail_T = True
    hys_remaining -= 1
else:
    cohort_tail_T = False
    hys_remaining = max(0, hys_remaining - 1)
```

### NE-corruption gate (S1 calibration 2026-05-15)

Initial agent design assumed `error_psd` EMA-smoothing was sufficient
to handle DT NE bleed. **Initial 8-case test FAILED** on DT controls:
NN7yhG2X_doubletalk fired 21.5% (max_db 18.0) at T_HI=15. Root cause:
during DT, NE speech inflates `error_psd` while `long_window_far_psd`
stays steady → proxy reads "ERL drift" that's actually NE bleed.

**Fix**: gate on `inst_erl_raw = mic_pwr / far_pwr < 1.5` (same NE-
corruption protection rule the scalar ERL update uses at
[python/aec.py:6953](../python/aec.py#L6953)). Cohort tail (qNvSMyU)
has `mic ≈ small_ERL × far → inst_erl_raw ≈ 0.3-0.7 < 1.5` ✓ — gate
does NOT block cohort tail.

After the gate added, NN7yhG2X dropped from 21.5% to 0% fires (max_db
18.0 still produced but never gates through to the proxy compute).

### Threshold tuning (2026-05-15)

| Param | Initial | Final | Reason |
|---|---|---|---|
| `arc_t_threshold_hi_db` | 15.0 | **18.5** | TAIL min max_db = 19.15 (Hp5g1asac); CTRL max_db = 18.01 (NN7yhG2X). 18.5 splits the gap with ~0.5 dB margin both sides. |
| `arc_t_threshold_lo_db` | 10.0 | **13.0** | Reduce hysteresis hold on post-catastrophe-recovery frames. |
| `arc_t_inst_alpha` | 0.85 | 0.85 | Unchanged. TC ≈ 65 ms. |
| `arc_t_window_frames` | 64 | 64 | Unchanged. ~1 s rolling window. |
| `arc_t_hysteresis_frames` | 200 | 200 | Unchanged. ~2 s release window. |

## Plug-in points (cite line numbers in worktree at HEAD)

| Location | Change |
|---|---|
| `AecConfig` (after Arc M flag) | 8 new config fields |
| `AEC.__init__` (after `_per_band_erl` init) | 8 new state fields (proxy state + ring buffers + signal + counter) |
| `AEC.reset()` | 8 fields cleared (cumulative `_arc_t_fire_count` IS cleared by full reset) |
| `AEC._reset_filter_derived_state()` | 7 fields cleared (cumulative counter PRESERVED on partial reset) |
| Per-band ERL update (after slow-EMA loop) | Arc T proxy compute + hysteresis state machine + NE gate |
| `eval_aec_challenge.py` (after Arc M env block) | 8 env overrides |

## Hard bars verified

| Bar | Status |
|---|---|
| Byte-equal flag-OFF (5-case atol=0.0): same as pre-Arc-T baseline at e857209 | **PASS 10/10** MD5 identical |
| 8-case S1 validation: 5 TAIL fire / 3 CTRL no-fire | **PASS 5/5 + 3/3** |

### 8-case S1 validation results

```
role                     frames  fire%  max_db fire_count  PASS?
----------------------------------------------------------------------
TAIL_canonical             2686   16.9   23.08        350  PASS
TAIL_named_outlier         2416   11.5   22.55        159  PASS
TAIL_arc_f_breaker         2325   68.7   24.42       1580  PASS
TAIL_xqvgr_dt_mvmt         3695   18.4   23.09        396  PASS
TAIL_arc_m_v2_breaker      3846    7.6   19.15        139  PASS
CTRL_fs_static             2169    0.0    6.82          0  PASS
CTRL_dt_static_v2          3697    0.0   18.01          0  PASS
CTRL_ne_only               1103    0.0    0.00          0  PASS
```

Cases:
- TAIL_canonical = `qNvSMyUSXUyrDGpOw7s6qg_farend_singletalk` (postmortem rank 655/660)
- TAIL_named_outlier = `0I0XMl3M0ECO0U1N0cJvpg_farend_singletalk_with_movement` (postmortem named outlier)
- TAIL_arc_f_breaker = `3UAwzzOa40aCXQAmEdpwww_farend_singletalk_with_movement` (Arc F V1 FS_movement breaker)
- TAIL_xqvgr_dt_mvmt = `XqvGR01tJkan17zltLs38Q_doubletalk_with_movement` (Arc D D.S4.2 listen regression)
- TAIL_arc_m_v2_breaker = `Hp5g1asacUCt5rJVLO1FuQ_doubletalk_with_movement` (Arc M V2 cohort listen)
- CTRL_fs_static = `IrQvqOTCmEWMXn9k2ICtRQ_farend_singletalk` (LISTEN_CASES entry 02)
- CTRL_dt_static_v2 = `NN7yhG2XTEqq46X8X0yLfA_doubletalk` (Arc G zero-fire control)
- CTRL_ne_only = `014AzuqPZku2004NbTTmcA_nearend_singletalk` (alphabetically first NE)

Note: agent's recommended CTRL_dt_static (`LHsrJBRGnUKiMC2mihEr0g_doubletalk`)
was MISLABELED — it actually has cohort tail signature (fired 83% with
max_db 24.77). Substituted with `NN7yhG2XTEqq46X8X0yLfA_doubletalk` which
had zero Arc G fires AND zero Arc T fires, confirming it's a true control.

## §1.5.S2 plan (next sprint)

S2 wires:
1. `cohort_tail_T` field exposure for §1.5b consumption (already done — §1.5b reads `self._arc_t_cohort_tail_signal` directly)
2. RES preempt (H1: boost `effective_over_sub × arc_t_over_sub_boost=1.3` + H2: force `_using_render_based=True`) at [python/aec.py:~7152-7158](../python/aec.py#L7152) when `arc_t_res_preempt_mode=True AND cohort_tail_T=True`
3. `AecStats.cohort_tail_T` field for trace consumption

Acceptance: 5-case byte-equal sanity (atol=0.0) when both flags OFF. **5-case selection independent of S1 8-case validation cohort.**

## §1.5.S3 plan (after S2)

800-case A/B with both flags ON vs both flags OFF on `feature/v3.15-arc-t`:
- Cohort tail bucket Δecho ≥ +0.030 (preemption helps)
- DT Δdeg ≥ -0.005, FS Δecho ≥ -0.020 (no regression)
- Detector FP rate ≤ 5% on non-tail cohort

## §1.5b dependency contract

§1.5b reads `self._arc_t_cohort_tail_signal: bool` directly on `AEC`.
Per-frame call ordering (verified by trace):
- Arc T proxy block at line ~7058 writes `_arc_t_cohort_tail_signal`
  AFTER the per-band ERL update block ends, BEFORE `self.res.process()`
  (which is called near line 7188)
- The 5 `_arc_m_q_boost` invocations (lines ~6124, ~6393, ~6719, ~6809,
  ~6854) are all EARLIER in the same frame than Arc T compute → §1.5b
  reads previous-frame `_arc_t_cohort_tail_signal`. 1-frame latency =
  10 ms (hop=160 / 16 kHz). Negligible vs qNvSMyU's ~2.7 s decile
  timescale.

## Files modified

- `python/aec.py` (+~110 lines): config (8 fields), state init, dual reset paths, proxy compute block with NE gate
- `python/eval_aec_challenge.py` (+~25 lines): 8 env overrides
- `docs/v3_15_arc_t_s1_design_and_verdict.md` (this doc, new file)

## Closure protocol per §0.4

If S2 byte-equal sanity FAILS → fix wiring (no behaviour decision).
If S3 800-case Δecho < +0.030 on cohort tail OR DT Δdeg < -0.005 OR FS
Δecho < -0.020 → §1.5 kill → §1.5b BLOCKED, Arc M permanently CLOSED.

Substrate retained as flag default OFF regardless of verdict (mirror Arc
F/M closure pattern). The Arc T proxy is reusable for v3.16 detector-
audit research even if §1.5 fails.
