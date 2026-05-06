# Research log — P4B γ²-primary `dt_per_bin` (sub-noise, closed)

Date: 2026-05-06
Code line: `main` at v3.10.4 + P3c Phase 1a (shipped) + commit 291edf1
(Plan B toggle wired, default OFF). 800-case bench compares
`AEC_PLAN_B=1` against `AEC_PLAN_B=0` baseline, both at preset
balanced / fl=52ms / cng=on / parallel max_workers=3.

## Hypothesis under test

`dt_per_bin = max(effective_dt, 1−coh2)` lifts ALL bins to the
frame-scalar `effective_dt` floor when DTD fires moderately
(0.2–0.5). Even bins where γ²(k) clearly says "echo here" get
clamped up to the frame value, over-protecting echo-dominant bins
and forcing a uniform NE floor. Plan B re-derives `dt_per_bin` from
`1−coh2` alone (γ²-primary) and only lifts a soft floor when
`effective_dt > 0.5`:

```python
dt_per_bin_gamma = (1.0 - coh2).astype(np.float32)
if effective_dt > 0.5:
    floor_lift = float((effective_dt - 0.5) * 2.0)
    dt_per_bin = np.maximum(dt_per_bin_gamma, floor_lift)
else:
    dt_per_bin = dt_per_bin_gamma
```

Hypothesis: removing the ambiguous-region (effective_dt 0.2–0.5)
floor recovers per-bin γ²(k) discrimination → reduces NE
over-protection in DT and reduces over-suppression of NE.

## Method

- Wire `AecConfig.plan_b_dt_per_bin_gamma: bool = False` and `--plan-b`
  / `AEC_PLAN_B` overrides.
- Add 7 `_p4b_*` diag fields to `_diag` (no `AecStats` change), surface
  in `--trace-aec-state` CSV.
- 3-case sanity (smoke test only — toggle wires, CSV populates, no
  catastrophe).
- 800-case BALANCED bench (Plan B OFF baseline vs Plan B ON).
- AECMOS scoring per bucket; HF-mean dominance proxy across the
  smoke-test cases.

## Results

### 800-case bench (per-bucket means)

| Bucket | n | baseline echo / deg | Plan B echo / deg | Δ echo | Δ deg |
|---|---:|---:|---:|---:|---:|
| FS_static   | 169 | 3.641 / 4.999 | 3.646 / 4.999 | +0.005 | +0.000 |
| FS_movement | 131 | 3.704 / 4.999 | 3.712 / 4.999 | +0.008 | +0.000 |
| DT_static   | 186 | 4.217 / 2.328 | 4.216 / 2.322 | −0.001 | **−0.006** |
| DT_movement | 114 | 4.051 / 2.370 | 4.054 / 2.363 | +0.003 | **−0.007** |
| NE          | 200 | 4.998 / 4.010 | 4.998 / 4.010 | +0.000 | +0.000 |

**Every Δ has |·| < 0.01 — sub-noise across all 5 buckets.**

Notably:
- DT deg slightly *drops* (−0.006 / −0.007) — the opposite of the
  hypothesized direction.
- FS echo slightly *rises* (which would be the right sign if Plan B
  worked, but the magnitude is below noise floor and DT deg moves
  the wrong way; not interpretable as a real direction).
- NE is identical to 4 decimals.

Single-case 7GT_doubletalk_farend_singletalk pair: identical to
baseline within 0.001.

### HF-mean dominance proxy (3-case smoke set)

`effective_dt > (1 − p4b_coh2_hf_mean)` per frame:

| cohort | frames | dominance% | mean(eff_dt) | mean(1−coh2_hf) |
|---|---:|---:|---:|---:|
| 7GT_doubletalk    | 3667 | **5.15%** | 0.138 | 0.878 |
| FS_static (rep)   | 2176 | **0.00%** | 0.003 | 0.907 |
| NE (rep)          | 1103 | **0.00%** | 0.142 | 1.000 |

Caveat: this is the **HF-mean proxy** — `effective_dt` is compared
against the HF-band mean of `1−coh2`, not per bin. A frame can fail
the proxy yet still have a subset of bins where the floor dominates
and Plan B shifts gain. With 0% / 0% / 5% dominance and bench Δ
sub-noise across 5 buckets, the proxy and the bench agree
unambiguously — no need to upgrade to per-bin.

### Why Plan B is sub-noise

`dt_per_bin = max(effective_dt, 1−coh2[k])` per bin. Plan B differs
from baseline only on bins where `effective_dt > (1 − coh2[k])`. In
this dataset:
- `1−coh2[k]` saturates near 1.0 on most HF bins (FS post-cancel
  AND real DT alike — the failure mode #2 from P4x audit).
- `effective_dt` rarely exceeds 0.5 in DT cohorts (P3f Phase 2
  showed `shadow_advantage ≈ 1.0` across DT bucket means), and
  almost never exceeds the per-bin `1−coh2[k]`.

So the two formulas are nearly identical on nearly all frames /
bins, and AECMOS reflects that.

## Decision

**Plan B closed.** Per B.5 gate "Bit-identical / sub-noise → close
Plan B, go directly to P4W."

Triage matches the expected outcome flagged after the 3-case smoke
test: Plan B addresses Failure 1 (frame-scalar floor lifts all bins
uniformly) but Failure 2 (`1−coh2` saturating in both real DT and
FS post-cancel) governs in this dataset. Patch-level fix on a term
that doesn't dominate cannot move bench.

## What's retained

Default-OFF, zero-cost diagnostics:

| Field group | Source | Purpose |
|---|---|---|
| `p4b_dt_per_bin_mean / hf_mean` | `_stage_gain_compute` | Per-bin DT confidence (HF) |
| `p4b_coh2_hf_mean` | `_stage_gain_compute` | HF coherence² |
| `p4b_effective_dt`, `p4b_is_stationary_dt` | `_stage_gain_compute` | DTD frame-scalar |
| `p4b_gain_hf_mean` | `_stage_gain_postprocess` | HF gain after stage 7 |
| `p4b_res_echo_hf_mean_db` | `_stage_gain_postprocess` | HF residual echo PSD |

These are useful for P4W Phase 0 trace and stay in. Toggle
`plan_b_dt_per_bin_gamma` retained as default-OFF (not removed) for
audit reproducibility.

## Hand-off to P4W

**Constraint passed forward**: any successor must work where
`1−coh2` ≈ 1 in both real-DT and FS-post-cancel frames. Coherence-
based features alone cannot separate them. P4W primary features
(per memory `feedback_p4w_design_axes`):

1. `render_explained_ratio` — render long-window × ERL accountability
   of `error_psd`. Discriminates "echo not yet cancelled" from
   "post-cancel residual" — orthogonal to coherence saturation.
2. `nearend_excess_persistence` — windowed sustained high-band
   excess (NOT a single-frame metric).
3. `filter_state`, `usable_linear` (already wired by P3f).
4. `main_err_ratio`, `shadow_err_ratio` (already wired).

Spectral flatness, single-frame high-band excess, modulation:
diagnostic-only, not gate inputs. (They collapse on FS post-cancel
in the same way `1−coh2` does.)

P4W Phase 1 = state classifier on the four primaries (state fusion;
AUROC FS-vs-(NE + positive-DT) ≥ 0.80 on the **fused** classifier,
not single feature).
P4W Phase 2 = changes *gain policy* (gain floor lift / attack-release),
not the `dt_per_bin` formula.

## Files

- 800-case CSVs: `/tmp/p4b_baseline/`, `/tmp/p4b_on/`
- AECMOS scores: `/tmp/p4b_baseline_scores/`, `/tmp/p4b_on_scores/`
- Driver: `python/eval_aec_challenge.py` with `AEC_PLAN_B` env
- Plan B toggle wired: `python/aec.py` (commit 291edf1)
- This log: `docs/research_log_p4b_gamma_primary.md`
- P4x audit: `docs/research_log_p4x_legacy_bc_audit.md`
