# v3.13 Phase 3 — RES gain_floor 5-path empirical audit verdict

**Status**: Sprint S4-S5 CLOSED. Empirical audit complete. Findings
substantially **revise** the v3.12 plan's static categorization. The
canonical refactor (Sprint S6-S7) value is **smaller than originally
hypothesized** — see "Implications" section.

**Date**: 2026-05-14

## TL;DR

Empirical audit (instrumented BALANCED render of 800-case AEC Challenge,
fl=832, cng=True) of 5 RES gain-floor paths in `python/aec.py` ResFilter
class. Per-path fire rates (frames where path actually modified gain on
≥1 bin) tabulated by bucket.

**Single biggest finding**: `epc_dt_cap` (line 2024 area, EPC_DT_GAIN_CAP
= 0.85 in `_stage_gain_postprocess`) **fires on 0/800 cases across all
buckets**. The cap is mathematically dead code in BALANCED — the gain
never exceeds 0.85 in conditions where `epc_dt` is true. This empirically
confirms the in-source comment at line 2141 ("cap is largely dead code
there").

Other findings (verdict per path below) **mostly support physical
fallback hypothesis** for the remaining 4 paths. The Q7 V3 verdict's
hypothesis that `ne_g_floor` is the main "fragmentation" source is
**not empirically supported** — `ne_g_floor` fires 88-99% across all
buckets (low skew = 0.13), behaving as a universal baseline floor
rather than a bucket-skewed evidence patch.

## Method

### Code instrumentation

`python/aec.py` ResFilter class — added `_diag_floor_fires` dict +
`_floor_audit_enabled` flag (env-gated `AEC_RES_FLOOR_AUDIT`, default
OFF, ZERO behaviour change). 5 read-only `np.any(...)` increments at
the 5 floor-firing sites. Byte-equal flag-OFF verified on 5 sample
cases (NE / FS_static / FS_movement / DT_static / DT_movement) — 100%
PASS.

| Path | Site | Trigger condition |
|---|---|---|
| `spectral_floor` | line ~1980 | `np.any(spectral_g_min > g)` before `g = max(g, spectral_g_min)` |
| `ne_g_floor` | line ~2492 | `np.any(ne_g_floor > spectral_g_min)` before `spectral_g_min = max(spectral_g_min, ne_g_floor)` |
| `epc_dt_cap` | line ~2024 | `epc_dt is True AND np.any(g > 0.85)` before `g = min(g, 0.85)` |
| `quiet_mask` | line ~2034 | `np.any(quiet_mask & (g < 1.0))` before `g[quiet_mask] = 1.0` |
| `divergence_floor` | line ~2125 | `divergence > 0.3 AND np.any(g > divergence_gain)` |

The `frames_total` counter is incremented unconditionally at the
`ne_g_floor` site (a code path that runs every frame).

Instrumentation, harness scripts (`run_800_audit.py`, `byte_equal_sanity.py`,
`aggregate_audit.py`) preserved in worktree
`worktree-agent-abbbb8ba75683ce4d` for v3.14 reuse.

### Bench

800-case BALANCED render with `AEC_RES_FLOOR_AUDIT=1`, `fl=832`,
`cng=True`, `seed=0`, j=4. Per-case fire counts dumped to
`/tmp/v3_13_phase3/audit_800.json`. Aggregator at
`/tmp/v3_13_phase3/aggregate_audit.py`.

## Per-path fire rate by bucket

Mean rate (median, p99 of per-case rate within bucket).

| Path | FS_static<br>(N=169) | FS_movement<br>(N=131) | DT_static<br>(N=186) | DT_movement<br>(N=114) | NE<br>(N=200) |
|---|---|---|---|---|---|
| spectral_floor | 0.894 (m 0.921, p99 0.999) | 0.881 (m 0.907, p99 0.998) | 0.524 (m 0.532, p99 0.617) | 0.529 (m 0.538, p99 0.636) | 0.097 (m 0.000, p99 0.912) |
| ne_g_floor | 0.880 (m 0.920, p99 1.000) | 0.867 (m 0.923, p99 1.000) | 0.934 (m 0.963, p99 1.000) | 0.933 (m 0.967, p99 1.000) | 0.999 (m 1.000, p99 1.000) |
| **epc_dt_cap** | **0.000** | **0.000** | **0.000** | **0.000** | **0.000** |
| quiet_mask | 0.509 (m 0.521, p99 0.868) | 0.495 (m 0.515, p99 0.850) | 0.298 (m 0.307, p99 0.508) | 0.307 (m 0.334, p99 0.522) | 0.060 (m 0.000, p99 0.737) |
| divergence_floor | 0.006 (m 0.000, p99 0.073) | 0.002 (m 0.000, p99 0.026) | 0.002 (m 0.000, p99 0.035) | 0.003 (m 0.000, p99 0.047) | 0.000 |

### Bucket skew (max-min mean rate; higher = more bucket-fragmented)

| Path | Skew | Min bucket | Max bucket | Skew interpretation |
|---|---:|---|---|---|
| spectral_floor | 0.80 | NE | FS_static | Echo-correlated; fires when filter has residual |
| quiet_mask | 0.45 | NE | FS_static | Inverse echo-correlated; fires when bins are quiet (no echo or post-cancellation) |
| ne_g_floor | 0.13 | FS_movement | NE | Near-universal; behaves like baseline floor |
| divergence_floor | 0.006 | NE | FS_static | Almost dead (rare divergence events) |
| epc_dt_cap | 0.00 | all 0 | all 0 | DEAD CODE |

## Per-path verdict

### `spectral_floor` — PHYSICAL FALLBACK (load-bearing on cohort tail)

Mean fire rate skews FS > DT > NE (89% / 52% / 10%). Skew aligns with
"fires when echo residual is present" — physically expected. Cohort
tail (`qNvSMyUSXUyrDGpOw7s6qg`): **97% fire rate**. spectral_floor is
**load-bearing for the documented cohort tail catastrophe defence**.
Removing or unifying spectral_floor risks P52 trap regression.

**Action**: KEEP as discrete clamp in canonical refactor. Do not unify.

### `ne_g_floor` — UNIVERSAL FLOOR (low bucket skew, near-baseline)

Mean fire rate 88-99% across all buckets, skew only 0.13. **Q7 V3
hypothesis that ne_g_floor is the main "fragmentation source" is NOT
supported** — empirically it behaves as a universal baseline floor,
not a bucket-skewed evidence patch. The fire rate on NE is highest
(99.9%, near 100%) which is consistent with NE protection role; on
FS/DT it's still high (87-93%).

**Action**: this floor IS effectively a baseline. CANDIDATE for
unification into the canonical 6-step gain (Beroutti / Ephraim-Malah /
Hänsler-Schmidt step 6). Architectural simplification only — minimal
behaviour change expected since rate is uniform.

### `epc_dt_cap` — DEAD CODE (0/800 fires)

`g = min(g, 0.85)` when `epc_dt is True` fires 0% across ALL buckets.
The `epc_dt` flag never coincides with `g > 0.85` in BALANCED — either
because `epc_dt` rarely fires OR because when it fires, the gain is
already capped lower by another path. In-source comment at line 2141
already noted this ("cap is largely dead code there"); empirically
confirmed on full corpus.

**Action**: REMOVE in canonical refactor. Code complexity reduction,
zero behaviour impact.

### `quiet_mask` — PHYSICAL NOISE GATE (FS-skewed but expected)

Mean fire rate 51% FS / 30% DT / 6% NE. Skew = 0.45. The skew is
inverse-echo-correlated: FS has more "quiet bins" (echo cancelled in
some bins, gain forced to 1.0 to pass through) than NE (continuous
voice fills all bins).

**Action**: KEEP as discrete clamp. Noise gate is physically distinct
from the gain-suppression floor; merging would conflate semantics.

### `divergence_floor` — EDGE-CASE PROTECTION (almost dead)

Mean fire rate <1% everywhere. Only fires meaningfully on rare FS
cases where filter divergence > 0.3. Worst case `SFvlSygv4ke9wCrv8LWvYQ`
fires 24% (single outlier).

**Action**: KEEP as discrete clamp. Rare but semantically distinct
catastrophe defence; cost is negligible.

## Cohort-tail verification (`qNvSMyU`)

| Path | qNvSMyU fire rate |
|---|---:|
| spectral_floor | **0.974** (load-bearing) |
| ne_g_floor | 0.750 |
| epc_dt_cap | 0.000 (consistent with 800-case dead) |
| quiet_mask | 0.679 |
| divergence_floor | 0.000 |

`spectral_floor` at 97% on cohort tail is **the strongest single
empirical signal in this audit**. Any S6-S7 change touching
spectral_floor must include explicit cohort tail regression check;
ideally do not modify spectral_floor at all.

## Implications for Sprint S6-S7

The originally-planned canonical 6-step refactor (`spectral_floor +
ne_g_floor + epc_dt_cap + quiet_mask + divergence_floor` → unified gain
floor + 4 discrete clamps) **simplifies cleanly to**:

| Path | S6-S7 disposition |
|---|---|
| spectral_floor | KEEP as discrete clamp (cohort tail load-bearing) |
| ne_g_floor | UNIFY into canonical gain floor (truly baseline) |
| epc_dt_cap | REMOVE (dead code) |
| quiet_mask | KEEP as discrete clamp (physical noise gate) |
| divergence_floor | KEEP as discrete clamp (rare edge case) |

**Magnitude of refactor benefit**: SMALL.
- Architectural cleanup: 1 path removed (epc_dt_cap), 1 path absorbed
  (ne_g_floor). Code complexity reduced.
- Behaviour change: ne_g_floor unification likely byte-equal-ish since
  it currently fires 88-99% (already universal); merging it into the
  canonical floor is mostly cosmetic.
- AECMOS expected delta: near-zero (consistent with v3.12 5-NEUTRAL
  closure; RES Stage 1 is at local optimum).

**Strategic recommendation**: Sprint S6-S7 may be deprioritized given
small expected benefit. v3.12 closed Stage 1 RES surface as exhausted
across 5 NEUTRAL sprints; this audit confirms architectural refactor
also has limited surface. Consider deferring S6-S7 to v3.14 along with
4-cap ranked priority (S8-S9) and per-state ENR tuple, OR proceed with
S6-S7 as cosmetic cleanup only (no AECMOS bench expected to move).

## Risks identified

1. **spectral_floor cohort-tail dependency** (97% on qNvSMyU) is the
   single biggest empirical risk. Any S6-S7 change must protect this.
2. **ne_g_floor unification** is the only meaningful candidate for
   unification, but its near-universal fire rate suggests merging it
   into canonical gain floor is mostly cosmetic. Do not over-engineer.
3. **epc_dt_cap removal** is safe (0/800 fires). Can be done as a
   standalone cleanup commit with byte-equal verification.

## Open questions for S8-S9 (4-cap ranked priority + per-state ENR)

This audit did not instrument the 4-cap chain at line 2174-2270 (only
the 5 gain-floor paths). 4-cap chain is `divergence > EPC > quiet >
smooth > HF`. Empirical fire rate audit of 4-cap is still needed for
S8-S9. epc_dt_cap finding here suggests several caps in the 4-cap chain
may also be dead code; deferred to S8-S9 instrumentation.

## v3.13 arc state after this verdict

| Arc | Status |
|---|---|
| E2 Delay | ✅ SHIPPED |
| E4 NLP | ❌ CLOSED CANNOT SHIP (amplitude wall) |
| E5 Saturation | ❌ CLOSED CANNOT SHIP (FS-DT trade-off) |
| **Phase 3 RES audit (S4-S5)** | ✅ **CLOSED — empirical findings**, S6-S7 deprioritized |
| Phase 3 RES unification (S6-S7) | ⏳ candidate — small expected benefit |
| Phase 3 RES 4-cap + ENR (S8-S9) | ⏳ candidate |
| F-HFR per-band Q/R | ⏳ v3.14 candidate |
| E1 mic_dynamic_margin | ⏳ v3.14 candidate |
| **v3.14 Volterra non-linear inverse** | ⏳ canonical breakthrough path |

## Decision needed

Three paths for v3.13 from here:

1. **Proceed with S6-S7 canonical refactor** — cosmetic cleanup
   (remove epc_dt_cap dead code + unify ne_g_floor into canonical
   gain floor). Estimated 1-2 sprints. Expected AECMOS delta: ~0.
   Mostly code-quality benefit.

2. **Do epc_dt_cap removal only as standalone commit** — narrow safe
   cleanup. 1 commit, byte-equal verified. Defer wider canonical
   refactor.

3. **Close v3.13 cleanly** — mark E2 shipped, E4/E5 closed, Phase 3
   audit closed; defer all RES architectural work to v3.14 along with
   Volterra. Move to whatever v3.14 design opens.

Recommend (3) given empirical evidence that RES architecture has
limited remaining surface (5 NEUTRAL v3.12 + this audit) and v3.13
front-end work is complete.

## Artifacts

- This verdict: `docs/v3_13_phase3_res_audit_verdict.md`
- Static predecessor: `docs/v3_12_res_audit.md` (existing)
- Instrumentation: worktree `worktree-agent-abbbb8ba75683ce4d`,
  uncommitted `python/aec.py` diff (preserved for v3.14 reuse)
- Harness: `/tmp/v3_13_phase3/{run_800_audit.py, byte_equal_sanity.py,
  aggregate_audit.py}`
- Raw audit: `/tmp/v3_13_phase3/audit_800.json` (gitignored)
