# Round 7 — Phase 0 Verdict

**Date**: 2026-05-04
**Branch**: `algo/round7-signal-split-trajectory` (from `9040bb9` R6 refactor base, bit-exact parity vs `baseline_v381_seeded`)
**Dataset**: 800-case blind, BALANCED preset, fl=832, CNG seeded
**Comparison**: DT_movement worst-20 vs best-20 (rank-locked by `baseline_v381_seeded.deg`)
**Worst-20 baseline_deg mean**: 1.397
**Best-20 baseline_deg mean**: 3.214

## Decision: **GO** → proceed to Phase 0.5 + Phase 1

6 R7 fields pass the upgraded gate (effect_size ≥ 0.5 + abs floor, OR event-count
≥ 6/20 worst hit + ≥ 2× best). Filter trajectory IS materially different between
worst-DT_mv and best-DT_mv — the upstream-of-RES hypothesis is empirically
supported.

## Top separators

| field | worst-20 | best-20 | rule | effect / hit |
|---|---:|---:|---|---|
| `r7_shadow_w_norm_mean` | 8.80 | 30.79 | continuous | effect=1.21 |
| `r7_main_err_smooth_mean` | 176.4 | 647.3 | continuous | effect=1.05 |
| `r7_filter_w_norm_mean` | 7.16 | 17.83 | continuous | effect=0.98 |
| `r7_shadow_err_smooth_mean` | 187.8 | 583.9 | continuous | effect=0.92 |
| `r7_mu_scale_mean` | 0.795 | 0.716 | continuous | effect=0.65 |
| `r7_event_epv_count` | 1.40 | 0.40 | event-count | 11/20 vs 5/20 (2.2×) |

## Mechanism reading

The direction is **consistent and counterintuitive in the right way**:

1. **worst-20 has SMALLER filter weight norm** (`filter_w_norm` 7.16 vs 17.83 —
   worst is 40 % of best). The filter on worst cases never grows into the
   echo path the way best cases do.
2. **worst-20 has SMALLER smoothed error magnitudes** (`main_err_smooth` 176 vs
   647). This rules out the naive "worst has bigger residual" reading — the
   *signal-energy regime* on worst cases is itself smaller, and the filter
   sits in a low-energy attractor.
3. **worst-20 runs HIGHER mu_scale** (0.795 vs 0.716) — the controller is
   trying to push adaptation, but the `W` norm stays small. The mu boost
   is not translating into convergence on these cases.
4. **worst-20 fires EPV 2.75× more often** (1.4 vs 0.4 events / case, hit-rate
   11/20 vs 5/20). EPV (echo-path-variation) is being triggered on worst
   cases that other cases don't hit.
5. **`r7_once_conv_pct` 0.601 vs 0.388** is just below the 0.5 effect-size
   bar (0.49) but the direction is consistent: worst cases reach
   `_filter_once_converged` MORE often (likely a low-energy quirk where the
   converged criterion fires before the filter has grown).

Signal NOT separating: `delay_*` events all zero on this dataset (no movement
delay shifts captured in 800-case blind), `r7_p_max_active_pct` /
`r7_p_floor_active_pct` ~equal worst/best, `r7_once_conv_pct` just under bar.

## Implications for Phase 1

The Phase 1 priority order should be **EPV first**, not delay_shift:

1. **R7.1 EPV-window adaptation**: in 100ms post-EPV window, adjust mu / Q
   so the filter has a chance to grow. Priority lifted from "shadow_rise
   first" because delay_shift events are absent on this dataset.
2. **shadow_rise**: still strong (worst 14/20 hit, best 12/20 — 1.18× ratio
   doesn't pass event-count gate but the COUNT differs 7.9 vs 4.7 ×).
   Worth probing as R7.2 with dry-mechanism-first protocol.
3. **delay_first / delay_shift**: drop from Phase 1 (no events on 800-case).
4. **Filter trajectory direct intervention**: out of stated R7 scope. The
   data motivates it but plan said "don't touch PBFDKF kernel". Note the
   diagnostic for any future round.

Plan's transition order (1.1a delay_shift first) needs updating to put EPV
first — the dataset doesn't exercise delay_shift but does exercise EPV.

## Phase 0.5 still mandatory

Even though Phase 1 priority shifts, the signal-split refactor remains
required: the EPV/mu_scale/converged signals each have multiple readers,
and any Phase 1 EPV-window intervention needs an isolated reader to act
on without disturbing other consumers.

## Outputs

- `experiments/round7_phase0/states.json` — 800-case R7 trace
- `experiments/round7_phase0/analysis.md` — per-field gate table
- `experiments/round7_phase0/verdict.md` — this file
