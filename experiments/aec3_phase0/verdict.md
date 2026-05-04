# Phase 0 verdict — population separation analysis

**Branch**: `algo/f1-f4-investigation`  •  **Date**: 2026-05-02
**Trace patch**: 6 diag fields added to `aec.py` (audio-passive), 800-case
state dump via `python/trace_phase0.py`. CNG seeded for reproducibility.

## What separates worst-DT from best-DT

| Field | DT_mv worst-best Δ | DT_st worst-best Δ | Verdict |
|---|---:|---:|---|
| `transitions` | **+4.3** | +2.45 | **separating** — worst cases wobble in/out of convergence ~10×, best cases ~5× |
| `init_pct` | -0.25 | -0.26 | separating, BUT inverted: worst cases stay LESS in init (i.e. they DO converge often, just unstably) |
| `usable_v1_pct` ≡ `usable_v2_pct` | -0.006 | +0.041 | weak / mixed — DT_st worst slightly more usable-linear, DT_mv neutral |
| `dominant_pct` | +0.015 | +0.033 | **NOT separating** — reviewer's DominantNearend hypothesis fails on this dataset |
| `dt_from_zero_count` | +32 | -388 | mixed/inverted — worst DT_st actually has FEWER cold-start triggers |
| `erl_final` | +0.004 | -0.031 | not separating |

## What separates worst-FS-echo from best-FS-echo

| Field | FS worst-best Δ | Note |
|---|---:|---|
| `init_pct` | +0.585 | worst-FS-echo cases stay 96% in init — they never converge |
| `transitions` | -7.45 | logical consequence (no transitions if no convergence) |
| `usable_v1/v2_pct` | -0.124 | logical consequence |
| `dominant_pct` | -0.090 | logical consequence |

Worst-FS cases are **pure cold-start non-converged failures** — any patch
that touches init-state frames will move their scores (good or bad).

## Per-bucket once_converged rate

| Bucket | once_converged |
|---|---:|
| FS_static | 64% |
| FS_movement | 69% |
| **NE** | **0%** (no echo to learn) |
| DT_static | 71% |
| DT_movement | 63% |

NE filter never converges (no far-end echo path to learn). NE deg
4.006 floor is therefore *purely a function of the suppressor on
non-converged signal* — filter convergence is irrelevant to NE.

## What this rules out / supports

**RULES OUT** for this round:
- **Phase 2 (DominantNearend)**: my AEC3-aligned proxy
  (`ne > 3×res AND ne > 5×noise + initial-state gate + hold counter`)
  fires on 14-16% of DT frames but does NOT correlate with worst-DT cases.
  Reviewer's hypothesis "frame-level NE-dominant state should relax
  suppression" lacks dataset support. **SKIP Phase 2** — the state model
  itself is wrong for this dataset.
- **Phase 1 (`usable_linear_estimate_v2` wiring)**: my v2 conditions
  (warmup_done + divergence < 0.3 + epc_hangover < 1) collapse to v1 in
  practice (% nearly identical). My F3 already wired v1; result was null.
  v2 can't do better than v1 here. **SKIP Phase 1** unless v2 is
  redefined (current definition redundant).

**SUPPORTS** for this round:
- **Phase 3 (ERLE / RES reset on transitions)**: `transitions` is the
  strongest discriminator on DT (worst 9.4 vs best 5.1 on DT_mv;
  worst 11.6 vs best 9.2 on DT_st). The hypothesis: when the filter
  loses convergence (`mark_diverged`) and recovers (`just_converged`),
  ERLE / RES trust state retains stale EMA from the prior regime,
  blending into the new regime non-physically. Fresh-state on each
  transition might break the bimodal F1 distribution that came from
  the same population.
- This is genuine **state routing** (per user distinction), not gain
  tuning — we're routing the convergence event to specific subsystem
  resets without changing W / mu_scale / gain.

## Decision

Skip Phase 1 + Phase 2 (failed Phase 0 separation). Proceed directly to
**Phase 3: ERLE / RES reset on each filter-convergence transition**.

Acceptance per user:
- FS_st / FS_mv echo regression ≤ 0.02
- NE deg ≥ baseline-0.01 (NE filter doesn't converge → reset has no
  effect on NE; should be exactly null)
- DT_mv deg improvement ≥ 0.02, OR top worst-DT improves and FS not hurt
  → guarded keeper

## Side note: F1/F3/F4 null may be partly CNG noise

Discovered while validating Phase 0 audio-passivity: AEC v3.8.1 with
CNG enabled is non-deterministic (CNG uses `np.random.randn` unseeded,
aec.py:2009-2010). Two runs of same code on same input differ by
max_abs 2.6e-2 on DT cases. Whether this masks small bench-level
signals (F1/F3/F4 each showed ±0.004 Δ) is open but probably yes —
single-run bench Δ < 0.005 is at or below CNG noise floor.

For Phase 3 bench: seed CNG (`np.random.seed(0)` once before each
run) and / or run the bench twice to estimate run-to-run variance,
so the new Δ can be calibrated against actual noise.
