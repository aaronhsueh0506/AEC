# S7 verdict — `dt_per_bin` legacy-path unification (`res_dt_per_bin_unified`)

**Date**: 2026-05-13
**Branch**: `feature/v3.11-s7-dt-per-bin` (parent: `feature/v3.11-route-a`)
**Design**: [v3_12_phase3b_v3_design.md §3.1](v3_12_phase3b_v3_design.md) (commit bd762ec)
**Status**: **S7 CLOSED — NEUTRAL** (third Q7 V3 carrier null result after S6 / S6b)

## TL;DR

The S7 800-case A/B verdict found `res_dt_per_bin_unified=True` produces
**±0.001 bucket-mean deltas** vs v3.11.2 baseline. No hard-abort threshold
crossed and no positive criterion met. Per-case distribution shows 25
cases shift >0.01 in Δecho (10 pos / 15 neg) and 16 cases shift >0.01
in Δdeg (5 pos / 11 neg). Cohort tail `qNvSMyU` is byte-equal (Δ=0.0000).

The 67.7% per-bin reduction measured at `dt_per_bin` (pre-implementation
audit) **does not propagate to output** because downstream clamps
(`noise_floor_psd` floor on `nearend_est`, Wiener gain saturation,
3-bin smooth, spectral_floor, 4-cap chain) absorb the per-bin shift.

S6 (ne_g_floor neutralised by `1-fs_confidence`) + S6b (epc_dt_cap dead
gate) + S7 (dt_per_bin downstream-absorbed) jointly form a **three-trial
null** that empirically falsifies the Q7 V3 attribution of FS echo leak
to RES post-process gain caps **or** to evidence-level per-bin DT
indicators. Real FS-visible carrier is elsewhere — re-investigation
required before next sprint.

BALANCED preset unchanged. `res_dt_per_bin_unified` stays default-OFF
research substrate. ResFilter audit counter infrastructure
(`enable_audit_counters` / `get_audit_counters`) retained for future
sprints (durable, replaces per-sprint ad-hoc counters).

## Pre-implementation audit (§6.1 gate) — PASS

Run: `tools/research/s7_dt_per_bin_audit.py`, preset BALANCED,
`np.random.seed(0)`, fl=832, cng=True. 800-case j=4. Output:
`results/v3_12_s7_audit/audit.json`.

| bucket | cases | frames | legacy% | F3.1v3% | legacy+EPC% | target FS bins | reduce% |
|---|---:|---:|---:|---:|---:|---:|---:|
| FS_static | 169 | 387,980 | 96.18 | 3.82 | 32.83 | 770,512 | **67.2** |
| FS_movement | 131 | 295,122 | 97.64 | 2.36 | 33.56 | 514,602 | **68.1** |
| DT_static | 186 | 701,736 | 97.43 | 2.57 | 42.92 | 960,244 | **68.2** |
| DT_movement | 114 | 424,351 | 98.59 | 1.41 | 38.36 | 426,064 | **66.9** |
| NE | 200 | 222,833 | 100.00 | 0.00 | 0.00 | 0 | n/a |
| **GLOBAL** | **800** | **2,032,022** | **97.75** | **2.25** | **33.97** | **2,671,422** | **67.7** |

Gate thresholds per design §6.1:
- Fire rate > 0.5% in target bucket → **PASS** (33.97% globally, 60× the floor)
- Unified reduction ≥ 20% in FS bins → **PASS** (67.7% globally, 3.4× the bar)

Both gates passed with large margin. Audit gave a strong GO signal,
which makes the downstream A/B null result especially informative.

## Step 5 byte-equal scaffold — PASS (100%)

`tools/research/s7_byte_equal_check.py`: 800/800 cases byte-equal vs
`results/v3_12_s3_candidate/` (v3.11.2 baseline). Flag-OFF substrate
is verified safe. Per-bucket match%: 100.00% across FS_static,
FS_movement, DT_static, DT_movement, NE.

## Step 6 flag-ON A/B (AECMOS) — NEUTRAL

Run: `tools/research/s7_bench_runner.py --flag-on`, score via
`python/bench_aecmos.py results/v3_12_s7_on results/v3_12_s7_scores_on
--baseline results/v3_12_s3_candidate/scores.json`.

### Bucket means (Δ vs v3.11.2 baseline)

| Bucket | Δecho | Δdeg | hard-abort? | positive? |
|---|---:|---:|---|---|
| FS_static | −0.001 | +0.000 | no (>−0.02) | no (<+0.02) |
| FS_movement | −0.001 | +0.000 | no (>−0.02) | no (<+0.02) |
| DT_static | +0.000 | −0.001 | no (>−0.005) | no (<+0.005) |
| DT_movement | −0.000 | −0.000 | no (>−0.005) | no (<+0.005) |
| NE | +0.000 | +0.000 | no (>−0.005) | n/a |

No hard-abort threshold crossed. No positive criterion met. Result is
**NEUTRAL** — within bench noise floor.

### Per-case distribution

| metric | cases with \|Δ\| > 0.01 | improvements | regressions |
|---|---:|---:|---:|
| Δecho | 25 | 10 | 15 |
| Δdeg | 16 | 5 | 11 |

Per-case shifts are dominated by regressions (1.5× and 2.2× the
positive count). Bucket means are NEUTRAL because the magnitudes cancel
across cases.

### Worst regressions

| case | bucket | Δecho |
|---|---|---:|
| `KOy0eftktkuJf180xtXudg_farend_singletalk` | FS_static | −0.077 |
| `WH0jN3PY40es2S0LsxmkkQ_farend_singletalk` | FS_static | −0.065 |
| `OXCtw0FVhUWGUBil6Uucdw_farend_singletalk` | FS_static | −0.064 |
| `iOyPaxX11UOaUkcscKhq1A_farend_singletalk` | FS_static | −0.049 |

| case | bucket | Δdeg |
|---|---|---:|
| `KOy0eftktkuJf180xtXudg_doubletalk` | DT_static | −0.111 |
| `W0zK3dv0QE2YckPArTGXCg_doubletalk` | DT_static | −0.046 |
| `yM2wHof9U06yVPJfemZ3hg_doubletalk` | DT_static | −0.035 |

`KOy0eftktkuJf180xtXudg` is the worst case in both FS_static
(Δecho −0.077) and DT_static (Δdeg −0.111) — same recording / room,
regresses in both directions. Cohort tail `qNvSMyU` Δ=0.0000 (byte-equal).

## Mechanism: why the audit-measured 67.7% reduction did not propagate

The audit measured `dt_per_bin` shape AFTER the legacy `(1-coh²)` saturation
path. The unified-hypothetical produces a 67.7% per-bin mean reduction in
FS bins (coh²<0.1). However, the AECMOS-visible output is shaped through a
9-stage pipeline downstream of this signal:

```
dt_per_bin (CHANGED 67.7%)
        │
        ▼  (line 2236-2237)
dt_shaped_per_bin = dt_per_bin ** 1.1
        │
        ▼
nearend_est = max(raw_nearend_est * dt_shaped, noise_floor_psd)
        │              ← CLAMP: when raw_nearend_est is tiny (FS post-cancel),
        │                noise_floor_psd dominates → dt_per_bin change invisible
        ▼
ENR / Wiener gain
        │              ← CLAMP: g_min floor / Beroutti over-sub absorb small shifts
        ▼
gain caps (epc_dt / quiet_mask / divergence_floor / ne_g_floor)
        │              ← CLAMP: ne_g_floor neutralised by (1-fs_confidence) S6;
        │                       epc_dt_cap dead in BALANCED S6b
        ▼
3-bin smooth [0.1, 0.8, 0.1]
        │              ← LOWPASS: smooths the residual per-bin shift
        ▼
spectral_floor
        │              ← CLAMP: physical fallback overrides
        ▼
output gain g[k]       ← ~0 effective per-bin change
```

The dt_per_bin saturation is real mechanism, but in FS post-cancellation
the chain `raw_nearend_est → ε → noise_floor_psd` already pins
`nearend_est` to a noise-floor-dominated value regardless of the
multiplier. The 1.1-power shaping then has a small relative effect.
Bins where dt_per_bin actually flips a downstream decision are rare —
~25/800 cases with |Δ|>0.01 per AECMOS.

## What this overturns

Phase 3B v3 design §2 hypothesised `dt_per_bin` saturation as the
upstream Q7 V3 carrier. S7 falsifies the *carrier-strength* hypothesis
empirically: even with the carrier modified at 67.7% reduction, output
is essentially unchanged. The downstream clamps (specifically the
`noise_floor_psd` floor on `nearend_est`) are the real bottleneck for
output-visible per-bin changes.

S6 + S6b + S7 are now three sequential null results on Q7 V3-attributed
RES gain / evidence carriers:

| sprint | target | result | mechanism that absorbed change |
|---|---|---|---|
| S6 | ne_g_floor evidence swap | byte-equal | `(1-fs_confidence)` zeroes ne_g_floor in FS |
| S6b | epc_dt_cap gate swap | dead gate (0% fire) | legacy `effective_dt AND epc_active` never co-occurs |
| S7 | dt_per_bin upstream blend | bucket Δ≤0.001 | `noise_floor_psd` floor on `nearend_est` absorbs per-bin shift |

The Q7 V3 critique was **correct at mechanism level** (8 patch generations
on broken evidence) but mis-located the *output-visible carrier* in all
three targeted layers. Real carrier lives further downstream (probably
in `noise_floor_psd` itself, or in the 3-bin smooth, or in the spectral
shape of `raw_nearend_est`).

## Anti-trap reflection

| Trap | Original concern | Post-S7 reading |
|---|---|---|
| **P50** (FS Δecho −1.328) | Per-bin evidence change could mis-fire in FS | **Moot** — dt_per_bin change did not propagate at all, regardless of FS vs DT |
| **P52** (cohort tail −0.56) | qNvSMyU vulnerable | **Verified safe** — Δ=0.0000 byte-equal |
| **P55** (DT-FS +7.01 vs 20 dB) | Wiener gain replacement weakness | **Not applicable** — Wiener pipeline unchanged in S7 |
| **P58** (FS Δecho −0.674) | Cap chain weakening | **Verified safe** — Stage 1 residual caps unchanged |

Anti-trap framework worked exactly as intended: zero hard-abort threshold
crossings. The neutral result is *because* the design correctly avoided
the trap pathways, not because the mechanism was unsafe.

## Action taken

1. **BALANCED unchanged**: `res_dt_per_bin_unified` stays default-OFF.
2. **Flag and gate code retained**: `if ... res_dt_per_bin_unified` branch
   in `_stage_gain_compute` simplified to conditional check; legacy
   branch is the default path. Substrate cost is negligible
   (~5 LoC + 1 dataclass field).
3. **Audit counter substrate retained**: `ResFilter.enable_audit_counters`
   and `AEC.enable_res_audit` are durable infrastructure — future
   sprints add new counter keys without per-sprint scrubs.
4. **Audit harness retained**: `tools/research/s7_dt_per_bin_audit.py`
   and `tools/research/s7_bench_runner.py` stay as patterns for
   future RES sprint audits.
5. **No revert**: all commits since `bd762ec` (design lock) are
   substrate-or-flag additions; nothing to revert.

## Next step: re-investigation

The post-Q7-V3 carrier hypotheses after S6 / S6b / S7 are:

| Path | File:line | Why candidate after S7 |
|---|---|---|
| `noise_floor_psd` floor on `nearend_est` | [aec.py:2237](../python/aec.py#L2237) | Pinned the dt_per_bin change to noise floor; this **is** the FS-visible clamp |
| 3-bin smooth kernel `[0.1, 0.8, 0.1]` | [aec.py:2370-2387](../python/aec.py#L2370-L2387) | Code comment at 2399+ admits "Plan A's actual FS cost lives in smoothing kernel"; still untested empirically |
| `raw_nearend_est` source: `error_psd - residual_echo_psd` | [aec.py:2214](../python/aec.py#L2214) | If residual_echo_psd over-estimates (Stage 1 4-cap chain pinning), `raw_nearend_est` → 0 → noise_floor pins regardless of dt_per_bin |
| `spectral_floor` physical fallback | [aec.py:2745-2756](../python/aec.py#L2745-L2756) | Phase 3A flagged as physical fallback (keep); but if it fires at FS frequencies, it overrides any earlier gain shaping |

Recommended Phase 3B v4 (re-design) sequencing:

1. **`noise_floor_psd` audit**: instrument what fraction of FS bins land
   on `nearend_est = noise_floor_psd` (vs `raw_nearend_est * dt_shaped`).
   If this is >50%, no per-bin evidence change upstream can move FS
   without touching the noise floor itself.
2. **Stage 1 residual cap audit**: instrument which of the 4 Stage 1
   caps fires most often (residual_echo_psd ≤ echo×2 / error×1.5 /
   error×dt_suppress / far×2·ERL). If one cap dominates, that is what
   pins `raw_nearend_est` to ε.
3. **3-bin smooth A/B with median kernel** (Option β from Phase 3B v3):
   substitute 3-tap median for [0.1, 0.8, 0.1] convolve. Per code
   comment, this is the v3.8.4-validated bottleneck.
4. **OR: close v3.12 with v3.11.2 as floor + Phase 3A audit as
   documentation deliverable**. Three null results suggests RES post-
   processing is well-calibrated already; further work belongs in
   Phase 4 (NLP arc, F-HFR per-band Q/R) which restructures the
   pipeline rather than retuning evidence.

User authorisation required before next sprint.

## Open questions for review

1. **Three nulls — give up on Q7 V3?** S6 / S6b / S7 are converging on a
   single conclusion: RES post-processing is not the FS leak carrier.
   Should we re-write the Q7 V3 §6 (the original RES audit) to reflect
   this finding, or treat it as a calibration finding?
2. **Worst case `KOy0eftktkuJf180xtXudg`**: per-bucket regression in
   both FS_static (Δecho −0.077) and DT_static (Δdeg −0.111) suggests
   a single-case pathology. Worth a single-case deep dive to identify
   the failure mode before any further per-bin evidence work.
3. **3-bin smooth (Option β) — proceed or stop?** Option β from
   Phase 3B v3 design is the only remaining gain-postprocess candidate.
   Code comment claims it's the v3.8.4-validated bottleneck. With S6 /
   S6b / S7 already null, probability β succeeds is uncertain.
4. **Pivot to Phase 4 long-arc?** F-HFR per-band Q/R restructures the
   Kalman filter rather than tuning RES evidence. Higher-risk-higher-
   reward arc. If Phase 3B v3 is closing without a win, pivoting to
   Phase 4 may be the right move.
