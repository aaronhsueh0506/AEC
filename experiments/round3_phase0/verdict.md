# Round 3 — Phase 0 verdict

**Branch**: `algo/f1-f4-investigation`  •  **Date**: 2026-05-03

Summary of findings on 800-case seeded baseline (`baseline_v381_seeded`):

## Strong population separators (worst-20 DT_movement vs best-20)

| Field | worst | best | Δ | implication |
|---|---:|---:|---:|---|
| `epc_pct` | 0.573 | 0.354 | **+0.22** | D2: stuck-EPC region is real |
| `epc_max_run` (frames) | 1639 | 932 | **+707** | D2: longest stuck-EPC = 16s (vs 9s) |
| `shadow_rise` count | 7.9 | 4.7 | +3.2 | known: more EPC firing on worst |
| `epv` count | 1.4 | 0.4 | +1.0 | known: more EPC firing on worst |
| `transitions` | 9.35 | 5.10 | +4.25 | (Round 2 finding, persists) |
| `pre_epc_dt_p90` | 0.420 | 0.276 | +0.145 | **D3: real DT being zeroed during EPC** |
| `pre_epc_dt_count` | 2415 | 1490 | +925 | D3: thousands of zeroed-DT frames per case |

## Hypothesis disposition

| User finding | Action | Reason |
|---|---|---|
| **D1** (P_floor boost on delay_shift) | **SKIP** | `delay_shift` fires 0× across all 800 cases. eval_aec_challenge pre-aligns ref via `estimate_delay()` + enables online delay-est on movement; the combination eliminates internal `delay_shift` events. Population doesn't exist in bench pipeline. |
| **D2** (EPC hangover tick decoupling) | **PROCEED** | Strongest separator. EPC duration scales with worst-DT_mv damage. Root cause confirmed: `tick_hangover()` at aec.py:3901 only fires inside `if shadow_filter and _filter_converged` — when `_maybe_mark_diverged` flips `_converged=False`, the tick freezes; EPC active stays True until filter re-converges. On worst cases this is forever. |
| **D3** (separate DT for adaptation vs suppressor) | **PROCEED** | 62/114 DT_movement cases have mean pre_epc_dt > 0.3 during EPC. Worst-20 has p90 = 0.42 (vs best 0.28). The aec.py:4047 `if epc_active: raw_dt = 0.0` zero-gate destroys this signal for the ResFilter. |
| **M1** (filter scale correction) | **SKIP** | Direction is inverted from hypothesis. Worst-FS-echo cases have filter `scale_med = 30` (40× overestimate), best-FS has `scale_med = 1.06`. So filter OVERSHOOTS on worst-FS, not undershoots. DT_mv worst (2.4) and best (5.8) both off but neither under 1.0; no clear DT misadjustment direction. Scale-correction would need separate FS-overshoot vs DT-direction designs — out of round 3 scope. |

## Next steps

1. **D2 patch**: decouple `epc_det.tick_hangover()` from `_filter_converged` gate. Inside the same `if shadow_filter is not None` guard but lift `and self._filter_converged` from the tick. Bench seeded.
2. **D3 patch**: at aec.py:4047, replace `raw_dt = 0.0` with split — keep `raw_dt` zeroed for the path that drives mu_scale / adaptation (which already happens after this line via `_compute_mu_scale`), but pass `pre_epc_raw_dt × 0.15` to ResFilter as the `dt_indicator` kwarg. Bench seeded.
3. Both patches with seeded CNG so Δ is comparable to baseline_v381_seeded.

## Acceptance (from user's spec)

- FS_st / FS_mv echo regression each ≤ 0.02
- NE deg drop ≤ 0.01
- DT_mv deg improvement ≥ 0.02 OR concentrated on baseline-worst with no FS hurt
