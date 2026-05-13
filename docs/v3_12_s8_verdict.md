# S8 verdict — Downstream-clamp audit identifies real FS-visible carrier

**Date**: 2026-05-13
**Branch**: `feature/v3.11-s8-downstream-audit` (parent: `feature/v3.11-route-a`)
**Predecessor**: [v3_12_s7_verdict.md](v3_12_s7_verdict.md) §9 re-investigation list
**Status**: **S8 CLOSED — diagnostic complete, S9 candidates ranked**

## TL;DR

The S8 800-case audit (BALANCED, seed=0, fl=832, cng=True) identifies
the **`noise_floor_psd` scalar floor on `nearend_est`** as the dominant
output-visible FS clamp (43% binding in FS bins), with **`min_ne_from_dt`**
as secondary (37%) and **Cap2 (`error_psd × mult`)** as the dominant
Stage 1 binder (18%) feeding both.

The three together account for ~80% of FS bin clamping. The other five
gain-postprocess paths (S6 ne_g_floor / S6b epc_dt_cap / S7 dt_per_bin)
all sit downstream of these clamps, which is why all three sprints
came back NEUTRAL — output-visible per-bin changes are absorbed by
the scalar `noise_floor_psd` before reaching the gain pipeline.

S9 candidates ranked by mechanism centrality + risk:
1. **`noise_floor_psd` refinement** (scalar → per-bin or coefficient lower) — highest centrality, NE risk MED
2. **Cap2 upstream fix** (residual_echo_psd over-estimation in FS) — addresses root cause, P58 trap risk MED
3. **`min_ne_from_dt` refinement** (already partly mitigated by S7; combine with #1 for joint effect)

## Step-by-step results

### Step 3 — Audit substrate extension (commit e741209)
Added 13 new counter fields to `ResAuditCounters`:
- Stage 1 4-cap binding counts + dB-reduction sums (8 fields)
- Nearend_est 4-way floor binding counts (4 fields: raw / noise_floor /
  min_ne_dt / ne_physical)
- 1 frame-counter (s8_stage1_fs_bin_total)

Single-case smoke verified byte-equal vs no-audit run (max diff 0.0
across 451,485 FS bin opportunities on `sZ9Egg0Y` FS_movement).

### Step 4 — 800-case audit (commit f743760)

#### Stage 1 4-cap binding rate (FS bins coh²<0.1)

| bucket | cap1 echo×2 | **cap2 err×mult** | cap3 dt | cap4 rc |
|---|---:|---:|---:|---:|
| FS_static | 0.10% | **18.28%** | 1.56% | 0.34% |
| FS_movement | 0.11% | **17.11%** | 0.94% | 0.29% |
| DT_static | 0.02% | 8.54% | 0.60% | 0.13% |
| DT_movement | 0.02% | 8.32% | 0.26% | 0.07% |
| NE | 0.00% | 0.38% | 0.00% | 0.00% |
| **GLOBAL** | 0.04% | **9.67%** | 0.62% | 0.15% |

**Finding 1**: Cap2 (`residual_echo_psd ≤ error_psd × mult`) is the
**only meaningfully-firing Stage 1 cap in FS** (17-18%). The other
three (echo×2, dt_suppress, render_ceil) are <1% in every bucket.
P58 verdict's "4-cap chain is load-bearing" is more specifically
"Cap2 is load-bearing"; caps 1/3/4 are dead in BALANCED production.

#### Nearend_est 4-way floor binding (FS bins)

| bucket | raw NE | **noise_floor** | **min_ne_dt** | ne_phys |
|---|---:|---:|---:|---:|
| FS_static | 18.81% | **44.08%** | **37.11%** | 0.00% |
| FS_movement | 22.21% | **41.72%** | **36.06%** | 0.00% |
| DT_static | 41.83% | 43.12% | 15.05% | 0.00% |
| DT_movement | 43.23% | 42.72% | 14.05% | 0.00% |
| NE | 52.46% | 45.41% | 2.13% | 0.00% |
| **GLOBAL** | 38.09% | **43.37%** | 18.53% | 0.00% |

**Finding 2**: In FS bins, **81% of `nearend_est` is pinned by either
`noise_floor_psd` (~43%) or `min_ne_from_dt` (~37%)**, not by the
actual raw NE estimate. This is the *exact* clamp the S7 verdict §9
hypothesised. The 67.7% per-bin reduction S7 measured at `dt_per_bin`
flows into `min_ne_from_dt` (which depends on `dt_shaped_per_bin`),
but once `min_ne_from_dt` drops below `noise_floor_psd`, the latter
takes over — output unchanged.

**Finding 3**: `ne_physical_floor = error_psd × 0.05` is **0% binding**
across all buckets. It's strictly dominated by either `min_ne_from_dt`
(error_psd × dt_shaped, with dt_shaped < 1 in NE → potentially < 0.05
× error_psd) or `noise_floor_psd` (scalar). The 0.05 coefficient is
too high to ever win; this is effectively dead code.

**Finding 4**: NE bucket has 52% raw NE binding (highest), 45% noise
floor, 2% min_ne_dt. In NE the raw NE estimate is genuinely large
(real near-end speech) so it wins more often. **NE risk from
`noise_floor_psd` changes is concentrated in non-speech NE bins where
noise floor wins (45%).** This is the bucket to watch in any S9 A/B.

### Counter coverage validation

GLOBAL `s8_stage1_fs_bin_total = 365,380,061` FS bin opportunities
across 800 cases. NE bucket contributes 56M bins, FS_static 55M,
DT_static 133M (largest), FS_movement 41M, DT_movement 79M. NEF total
matches stage1 total within rounding (4-way always picks one binder
when the mask is set).

## Synthesised carrier hierarchy

```
                Stage 1 attribution
                       │
                       ▼
            residual_echo_psd
                       │
       Cap1  Cap2 ←----│  Cap3  Cap4
       0.04% 9.67%      0.62%  0.15%
                       │
                       ▼ (in FS: ~17-18% pinning)
              raw_nearend_est = error_psd - residual_echo_psd
                       │
                  When Cap2 binds aggressively → raw NE ≈ 0
                       │
                       ▼ ne_est = max(raw × dt_shaped,
                                       noise_floor_psd,    ← FS 43%
                                       min_ne_from_dt,     ← FS 37%
                                       ne_physical_floor)  ← FS 0%
                       │
       In FS:  raw 20% / noise_floor 43% / min_ne_dt 37% / phys 0%
                       │
                       ▼
              ENR = residual / nearend_est
              Wiener / softgate / spectral
                       │
                       ▼ (S6/S6b/S7 attempted to change THIS layer)
              gain post-process
              (ne_g_floor / epc_dt_cap /
               dt_per_bin / quiet_mask /
               divergence_floor / 3-bin smooth)
```

The S6/S6b/S7 sprints targeted the *bottom three rows* (gain
post-process and evidence input to gain compute). None of them changed
the top three rows (Cap2 / noise_floor / min_ne_from_dt) which are the
real output-visible bottleneck.

## S9 candidate directions

### S9-A — `noise_floor_psd` refinement (RECOMMENDED FIRST)

**Mechanism change** ([aec.py:2247](../python/aec.py#L2247) area):
```python
# Current scalar
noise_floor_psd = np.mean(self.error_psd) * 0.01 + 1e-10
```

Two sub-options:

- **A.1** Lower coefficient: `× 0.001` (10× smaller floor). Risk:
  ENR division can produce huge values in bins where residual_echo_psd
  is small but non-zero. Need to gate downstream ENR clamp.
- **A.2** Per-bin instead of scalar:
  `noise_floor_psd = error_psd * 0.005` per bin. Removes the cross-bin
  averaging effect that makes scalar dominate quiet FS bins. The
  `ne_physical_floor = error_psd * 0.05` already does this at higher
  coefficient — A.2 with 0.005 would slot under it.

**Pre-implementation audit** (gated, same pattern as S7): instrument
counter on what fraction of FS bins **would** have raw NE win under
A.1 / A.2 instead of noise_floor.

**Risk**: MED. NE bucket has 45% noise_floor binding — over half of
those are likely silence/noise bins where the floor is correctly
preserving NE quality. Hard-abort threshold (NE Δdeg < −0.005) is
specifically designed to catch this.

**ROI**: HIGH. The single largest binder (43% FS) is being modified.
If 30-40% of the 43% binding rate shifts to raw NE, we expect
non-trivial FS Δecho movement (positive direction: smaller floor →
larger ENR → smaller gain → more suppression).

### S9-B — Cap2 upstream attribution fix

**Mechanism**: Cap2 binding at 17-18% in FS means
`residual_echo_psd > error_psd × 1.0` (non-render) or `× 1.5` (render)
in ~18% of FS bins. This is a symptom of upstream over-estimation in
either the ERLE-blended estimate (`attribute_legacy`) or the
render-based echo estimate. Fixing the upstream over-estimation
would reduce Cap2 binding → less aggressive clamping →
`raw_nearend_est` more often non-zero → more raw NE binding.

**Risk**: MED-HIGH. P58 verdict explicitly called Cap2 (and the 4-cap
chain) load-bearing. Touching Cap2 directly risks P58 regression.

**ROI**: MED. Addresses root cause but downstream `noise_floor_psd`
may still dominate in many bins (currently 43%); even if Cap2 is
fully fixed, expect some carry-over to noise_floor.

### S9-C — `min_ne_from_dt` refinement

**Mechanism**: 37% FS binding by `error_psd × dt_shaped × startup_scale
× render_factor`. S7 already partly targets this via dt_per_bin shape,
but the render_factor 0.5 halving and dt_shaped^1.1 saturation work
against it.

**Risk**: LOW (subset of S7's mechanism, already validated as
NE-safe).

**ROI**: LOW alone. Combined with S9-A is multiplicative.

### S9-D — Multi-knob fix (A + C combined)

Lower `noise_floor_psd` AND tighten `dt_shaped` (essentially
restoring S7's unified path but with the noise_floor headroom).

Risk: MED. Two knobs at once is harder to A/B individually if it
fails.

## Recommended sequencing

1. **S9 design lock**: pick S9-A first (single knob, highest
   centrality, lowest implementation cost).
2. **S9 pre-implementation audit**: extend the S8 substrate to count
   "would-be" raw NE wins under candidate A.1 / A.2. Use to choose
   between A.1 and A.2.
3. **S9 byte-equal scaffold + flag-ON A/B** (same Step 5/6 pattern
   as S7).
4. **If S9-A passes**: consider S9-D (combined A+C). If fails:
   investigate S9-B (Cap2 upstream).
5. **If S9-A fails too**: consider closing v3.12 with verdict
   "noise_floor_psd is well-calibrated; further work belongs in
   Phase 4 (NLP arc / F-HFR)".

## Anti-trap reflection (post-S7 lessons applied)

| Trap | How S8 audit addressed |
|---|---|
| **S6b dead-gate** | S8 measured binding rates pre-implementation; only 18.81-22.21% raw NE in FS = guarantees S9-A flag changes will fire. Cap1/3/4 already shown dead — S9 won't target them. |
| **S7 absorption** | The S8 audit *is* the absorption check: it tells us noise_floor_psd is what absorbs S7. S9 design starts from this knowledge. |
| **P50 / P52 / P55 / P58** | S9-A is a downstream clamp change, not a new evidence path (P50 trap) or a new discriminator (P55 trap). Cohort tail catastrophe defence is unrelated to noise_floor. P58 risk only if S9-B is selected. |

## Open questions for review

1. **A.1 (coefficient) vs A.2 (per-bin) for noise_floor**: A.2 is
   structurally cleaner (no mean-cross-bin averaging) but A.1 is a
   single-knob change. Run pre-implementation audit before choosing.
2. **Cap2 audit before S9-A**: should we additionally instrument what
   `residual_echo_psd` pre-Cap2 distribution looks like? Cap2 binding
   18% is symptomatic; understanding the upstream is independent of
   the S9-A fix.
3. **Should `ne_physical_floor = error_psd × 0.05` be retired?** Audit
   shows 0% binding across all buckets. Substrate is cheap to keep,
   but it's clearly dead code. Consider retiring as separate cleanup
   commit.
4. **Phase 4 pivot trigger**: if S9-A also closes NEUTRAL, the next
   layer (Cap2 / Stage 1 attribution mechanism) is dangerously close
   to P58. Should we set "S9 fails → pivot to Phase 4" as policy
   before starting, or re-evaluate at that point?
