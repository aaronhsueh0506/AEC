# Research log — P4x Plan B/C audit (write-up only)

Date: 2026-05-06
Code line: v3.10.4 + P3c Phase 1a (default ON). No behaviour change
in this branch — audit-only consolidation before P4B.

## Purpose

Consolidate the Phase-0 RES pipeline trace, identify the load-bearing
flaw behind "RES suppresses near-end during DT", and capture the
recoverable shape of the legacy v3.9.0 Plan B / Plan C branches so
that P4B can rebase the right diff onto current `main`.

## RES gain pipeline (9 stages, current `main`)

`ResFilter.process()` orchestrates per-bin gain `g`. Mutations in
order:

| # | Stage | File:line | Mutation |
|---|---|---|---|
| 1 | Residual echo attribution | [python/aec.py:1705-1806](python/aec.py#L1705-L1806) | none (returns `residual_echo_psd`) |
| 2 | ENR/Wiener gain compute | [python/aec.py:1808-1918](python/aec.py#L1808-L1918) | `g` initialized via per-bin ENR gate |
| 3 | EPC_DT cap | [python/aec.py:1928-1931](python/aec.py#L1928-L1931) | `g = min(g, 0.85)` |
| 4 | Quiet mask gate | [python/aec.py:1936](python/aec.py#L1936) | `g[quiet] = 1.0` |
| 5 | 3-bin smoothing kernel | [python/aec.py:1952-1956](python/aec.py#L1952-L1956) | Plan A: `[0.1, 0.8, 0.1]` (default ON) |
| 6 | HF cap @ 2 kHz anchor | [python/aec.py:2004-2014](python/aec.py#L2004-L2014) | Plan A: anchor 2 kHz, fires when `effective_dt < 0.3 AND high_ne_conf < 0.3` |
| 7 | Divergence override | [python/aec.py:2028-2030](python/aec.py#L2028-L2030) | severe cap when filter diverged |
| 8 | Temporal EMA smoothing | [python/aec.py:2036-2089](python/aec.py#L2036-L2089) | split attack/release |
| 9 | Noise-floor lift + CNG | [python/aec.py:2093-2161](python/aec.py#L2093-L2161) | floor lift + optional CNG synth |

Plan A (kernel tight + HF cap @ 2 kHz + stat_mask 7 kHz) is **already
shipped default-ON** in v3.10.4 (commits c2b7d37, 2b6409a). Anything
the old `~/.claude/plans/...tranquil-scroll.md` plan proposed under
"Phase 3" of the v3.8.3 era is moot.

## Root cause of NE suppression during DT

`_stage_gain_compute()` at [python/aec.py:1820-1828](python/aec.py#L1820-L1828):

```python
dt_per_bin = np.maximum(
    np.full(self.n_freqs, effective_dt, dtype=np.float32),
    1.0 - coh2
)
if is_stationary_dt:
    dt_per_bin = np.maximum(dt_per_bin, self._stat_dt_mask)
self._dt_per_bin_last = dt_per_bin
```

`dt_per_bin` then drives:

- `dt_shaped_per_bin = dt_per_bin ** 1.1` ([line 1830](python/aec.py#L1830))
- `nearend_est = max(raw_nearend_est * dt_shaped_per_bin, ...)` ([1831](python/aec.py#L1831))
- `ne_confidence = dt_per_bin` ([1855](python/aec.py#L1855))
- ENR threshold blend `enr_t = ne_confidence * enr_t_ne + (1 − ne_confidence) * enr_t_fs` ([1865](python/aec.py#L1865))
- ENR slope blend `enr_s` ([1866](python/aec.py#L1866))

### Two failure modes of the current formula

**Failure 1 — `np.maximum(effective_dt, 1−coh2)` lifts ALL bins to the
frame-scalar floor.** When DTD says "DT" with `effective_dt = 0.6`,
even bins where γ²(k) clearly says "echo here" get
`dt_per_bin >= 0.6`, over-protecting echo-dominant bins and forcing
a uniform NE floor across the spectrum even when only some bins
actually carry near-end energy.

**Failure 2 — `1−coh2` saturates in two unrelated regimes.** Both
real DT (NE energy uncorrelated with reference → low coh2) and FS
post-cancellation (linear filter removed coherent echo, residual is
small/uncorrelated → low coh2) produce `1−coh2 ≈ 1`. The gate cannot
distinguish them. P3f Phase 2 separately demonstrated that
`effective_dt = max(dt_for_fs, shadow_dt)` is itself unreliable
(`shadow_advantage ≈ 1.0` across DT bucket means), so the floor
contribution offers no extra discrimination.

Net: `dt_per_bin` lifts to ~1.0 in FS post-cancel just as readily as
in real DT, and the rest of the chain (raised ENR threshold, lifted
nearend_est) treats both as "protect NE" — which only helps in DT.

## Already-shipped (not in scope)

- Plan A kernel `[0.1, 0.8, 0.1]` ([aec.py:1952-1956](python/aec.py#L1952-L1956)) — c2b7d37
- HF cap anchor 500 Hz → 2 kHz with conditional gate ([aec.py:2004-2014](python/aec.py#L2004-L2014)) — c2b7d37
- `_stat_dt_mask` extended 4 kHz → 7 kHz ([aec.py:1448-1457](python/aec.py#L1448-L1457)) — c2b7d37
- P3c Phase 1a high-PAR delay fast-path — 52b4b07 / 437005c
- All Plan A toggles (`plan_a_kernel_tight`, `plan_a_hf_cap_2k`,
  `plan_a_stat_mask_7k`) default `True`

## Plan B recovery (commit `c50a0aa`, recoverable from reflog)

**Author**: aaron, 2026-05-05
**Title**: `fix(res): Plan B — per-bin γ²(k) primary in dt_per_bin`
**Scope**: 19-line change, `python/aec.py` only.

### Diff (against the v3.9.0-era base, structurally identical to current `main` for this block)

```diff
-            # Per-bin DT indicator: base from coh2 (works for speech far-end)
-            dt_per_bin = np.maximum(
-                np.full(self.n_freqs, effective_dt, dtype=np.float32),
-                1.0 - coh2
-            )
+            # Per-bin DT indicator. Plan B: γ²(k) is primary, frame-scalar
+            # effective_dt only contributes a soft floor when DTD strongly
+            # fires (> 0.5).
+            dt_per_bin_gamma = (1.0 - coh2).astype(np.float32)
+            if effective_dt > 0.5:
+                # Soft floor: 0 at effective_dt=0.5, full at effective_dt=1
+                floor_lift = float((effective_dt - 0.5) * 2.0)
+                dt_per_bin = np.maximum(dt_per_bin_gamma, floor_lift)
+            else:
+                dt_per_bin = dt_per_bin_gamma
             if is_stationary_dt:
                 dt_per_bin = np.maximum(dt_per_bin, self._stat_dt_mask)
```

### What Plan B claims to fix vs. what it actually fixes

Plan B addresses **Failure 1** (frame-scalar floor lifting all bins
uniformly) — when `effective_dt < 0.5`, the floor is removed entirely
and γ²(k) governs each bin independently; when `effective_dt > 0.5`,
the floor lifts gradually instead of jumping to its full value at any
DT signal.

Plan B does **NOT** address **Failure 2** — γ²(k) still saturates in
FS post-cancellation. The hypothesis under test in P4B: removing the
ambiguous-region floor (`effective_dt` ∈ [0.2, 0.5]) is enough to
recover meaningful per-bin discrimination, even though γ²(k) alone is
not a clean DT signal in FS post-cancel.

This is a **patch-level hypothesis**, not architectural. The
architectural fix needs an independent NE evidence source (high-band
excess, modulation, spectral flatness) — that is **P4W**, deferred.

### Rebase to current `main`

Block in current `main` is at [aec.py:1820-1828](python/aec.py#L1820-L1828).
Identical structure; the diff applies cleanly. P4B implementation
will additionally gate the new branch behind `plan_b_dt_per_bin_gamma`
config toggle (default OFF) for safe A/B.

## Plan C recovery (commit `3ffcf8f`, recoverable from reflog)

**Author**: aaron, 2026-05-05
**Title**: `feat(res): sub-band per-band RES decision layer (4 bands)`
**Scope**: 250 lines + `python/test_subband_res.py` (134 lines new).

### Shape

Replaces frame-scalar `over_sub` / `erle_factor` / `enr_scale` /
`dt_indicator` with per-sub-band 4-vectors broadcast to per-bin at
gain-compute. Sub-bands: 0–1 k / 1–2 k / 2–4 k / 4–8 k. High band
defaults to `1.4 × enr_scale` for NE-friendly suppression.

### Gating concern (blocks P4C without rework)

`over_sub` is **dead code in current ENR path**. It is read only by
the legacy spectral-sub branch at [aec.py:1915](python/aec.py#L1915),
which fires when `gain_type != "enr"`. All shipped presets use
`gain_type == "enr"` (default ENR path uses `dt_per_bin` and ENR
thresholds, not `over_sub`).

Therefore Plan C as written would **NOT affect default behaviour**
on any shipped preset. To make Plan C live, P4C would need to:

1. Either route per-band `over_sub` into the ENR path (new code), or
2. Rework `dt_per_bin` itself into per-band form (different shape than
   Plan C's commit).

Decision: P4C is **not a drop-in successor** to P4B. If P4B has weak
direction but no FS regression, P4C dry-run becomes a question of
which path (1) or (2) above to scaffold — to be decided after B.5
gate, not now.

## Diag fields needed for P4B (to be added in B.1)

### Already exposed (no work needed)

- `dt_confidence`, `erle_inst_db`, `erle_windowed_db`, `divergence`,
  `far_activity`, `res_gain_mean_db`, `res_using_render`,
  `echo_psd_mean_db`, `error_psd_mean_db` — `AecStats` dataclass
  ([aec.py:65-126](python/aec.py#L65-L126))
- `residual_psd_linear`, `residual_psd_render`, `residual_render_blend`
  — `_diag` dict via P3g
- P3f state fields (`filter_state`, `usable_linear`, `main_err_ratio`,
  etc.) — `_diag` via `--trace-aec-state`

### Need to be added (B.1 scope)

7 `_diag`-only fields, all per-frame scalars, written from inside
`ResFilter._stage_gain_compute()` and `_stage_gain_postprocess()`:

| Field | Site | Source |
|---|---|---|
| `p4b_dt_per_bin_mean` | gain_compute, after L1828 | `mean(dt_per_bin)` |
| `p4b_dt_per_bin_hf_mean` | gain_compute | `mean(dt_per_bin[hf_2k:])` |
| `p4b_coh2_hf_mean` | gain_compute | `mean(coh2[hf_2k:])` |
| `p4b_effective_dt` | gain_compute | `effective_dt` |
| `p4b_is_stationary_dt` | gain_compute | `int(is_stationary_dt)` |
| `p4b_gain_hf_mean` | gain_postprocess, end | `mean(g[hf_2k:])` |
| `p4b_res_echo_hf_mean_db` | gain_postprocess | `10·log10(mean(residual_echo_psd[hf_2k:]) + 1e-12)` |

CSV surfacing in `python/run_one_case.py` is **explicit**, not
automatic — header list and per-frame row-build need 7 entries.

## Deliverables

- P3 arc evidence preserved at `docs/archive/p3/` (10 logs, including
  `research_log_p3c_phase1b_negative.md`).
- `docs/SUMMARY.md` Round 8 = canonical entry point.
- This audit log = handoff to P4B.

## Next: P4B

Implement B.1 → B.2 → B.3 → B.4 → B.5 per
`~/.claude/plans/users-mingyu-desktop-novatek-se-aec-pyr-tranquil-scroll.md`.
