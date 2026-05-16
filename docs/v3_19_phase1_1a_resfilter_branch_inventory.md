# v3.19 Phase 1.1a — ResFilter branch inventory (filter_converged consumers)

**Status**: COMPLETE 2026-05-16
**Scope**: enumerate ResFilter branches that read `filter_converged`
(via `ResFilter.process()` parameter or downstream `_stage_*` calls)
and classify each by downstream effect on `effective_g_min` /
`res_gain_mean` / Wiener gain output.

## 1. Method

Searched [python/aec.py](../python/aec.py) for `filter_converged`
(case-sensitive, both bound and free occurrences). Found 92 references.
Filtered out:
- **Stats / diagnostics** (lines 4315, 4316, 4330, 4344, 5318, 9136,
  9361): stats accumulation only, no behaviour change
- **AECState back-ref + entry plumbing** (lines 5230, 5879, 6086,
  7212-9274 in AEC class): caller-side, not RES branches
- **Audit instrumentation** (lines 3343-3392): S7/S8/S9 audit
  counters, zero-cost when `_audit_counters` is None
- **EPC update** (lines 5462, 5474, 5507): outside ResFilter
- **AecConfig flag definitions** (lines 274-289): config-only
- **C.E migration switch entry** (lines 8468-8481): the v3.18 1-line
  switch we're now ablating; not a RES branch — it's the entry point

Remaining: **5 ResFilter branches** that materially gate algorithm
behaviour on `filter_converged`.

## 2. Branch enumeration

### Branch R1 — `attribute_legacy.force_render` OR-condition

**Location**: [python/aec.py:5116-5122](../python/aec.py#L5116)

```python
force_render = (
    epc_active
    or saturation_level > 0.5
    or not filter_converged
    or (self._arc_t_force_render_or_in_enabled
        and self._arc_t_cohort_tail_signal)
)
```

**Stage**: `ResFilter._stage_residual_model` →
`ResidualEchoEstimator.attribute_legacy()`

**Mechanism**: when `not filter_converged`, force render-based
(non-linear) R2 path — bypasses linear ERLE-blended `direct_est /
erle_corrected` formula.

**Downstream effect on residual_echo_psd**:
- Linear path (filter_converged=True): `R2 = ERLE-blended echo_psd /
  erle_corrected` (per-bin, smaller residual when filter is healthy)
- Render path (filter_converged=False): `R2 = X2 * echo_path_gain`
  with EMA long-window blend (delay-agnostic, larger residual)

**fq_usable substitution semantics**: replacing `filter_converged`
with `fq_usable` means render-mode forced ONLY when fq_usable=False
(i.e., AEC3 multi-gate FAILS). Since `fq_usable` is broader (52-86%
coverage vs filter_converged ~5%), this would make linear-mode the
default on far more frames → smaller residual_echo_psd default →
**less suppression by default** → likely BIG FS regression
(consistent with v3.18 C.E whole-arg result: FS_static -0.151).

**Ablation flag candidate**: `c_e_branch_force_render_use_fq_usable`

---

### Branch G1 — `_stage_gain_compute` F3.1 v3 mic-excess-evidence path

**Location**: [python/aec.py:3298-3323](../python/aec.py#L3298)

```python
if (self._use_mic_excess_evidence
        and filter_converged
        and _lw_ready
        and (self._dt_per_bin_unified or not epc_active)):
    far_lw = self._residual_est._long_window_far_psd
    erl_e = float(erl_estimate) if not isinstance(erl_estimate, np.ndarray) else erl_estimate
    excess = np.maximum(self.error_psd - far_lw * erl_e, 0.0)
    excess_ratio = np.clip(excess / (self.error_psd + 1e-10), 0.0, 1.0)
    legacy = 1.0 - coh2
    dt_per_bin = (_BLEND_F31_MIC_EXCESS * excess_ratio
                  + (1.0 - _BLEND_F31_MIC_EXCESS) * legacy)
    if effective_dt > 0.5:
        floor_lift = float((effective_dt - 0.5) * 2.0)
        dt_per_bin = np.maximum(dt_per_bin, floor_lift)
elif self._plan_b_dt_per_bin_gamma:
    dt_per_bin = (1.0 - coh2).astype(np.float32)
    ...
else:
    dt_per_bin = max(effective_dt_full_arr, 1.0 - coh2)  # legacy
```

**Stage**: `_stage_gain_compute`

**Mechanism**: when `filter_converged AND lw_ready AND
(dt_per_bin_unified OR not epc_active)`, replace `(1 - coh2)` legacy
NE evidence with F3.1 v3 mic-excess blend (0.7 F3.1 + 0.3 legacy).

**Downstream effect on dt_per_bin → nearend_est → Wiener gain**:
- F3.1 path: `dt_per_bin` driven by mic-energy excess over
  `far_lw * erl_estimate` → **per-bin NE evidence**, more accurate
  than coherence-saturating (1-coh2) in FS post-cancel
- Legacy path: `(1 - coh2)` saturates to 1 in FS post-cancel
  (coh2→0) → falsely raises NE evidence in FS

`dt_per_bin` then shapes `nearend_est = max(raw * dt_shaped, ...)`
which goes into `g = nearend_est / (nearend_est + residual_echo_psd)`
Wiener formula. Higher dt_per_bin → higher nearend_est → higher gain
in NE-protected bins.

**fq_usable substitution semantics**: `fq_usable=True` more often →
F3.1 path activates more often → DT NE preserve improves (matches
v3.18 C.E DT_static +0.214). FS_static *might* go either way: F3.1
mic-excess is supposed to be MORE accurate than (1-coh2) in FS, so
in principle should reduce false NE evidence. But the `_lw_ready`
gate plus `not epc_active` may still let some mis-attributed FS
frames through.

**Ablation flag candidate**: `c_e_branch_dt_per_bin_use_fq_usable`

---

### Branch P1 — `process()` coh2 EMA asymmetric tuning

**Location**: [python/aec.py:4060-4068](../python/aec.py#L4060)

```python
if filter_converged:
    a_coh_rise = 0.90   # TC≈160ms, stable echo-only tracking
    a_coh_drop = 0.50   # TC≈25ms, fast DT protection
else:
    a_coh_rise = 0.80
    a_coh_drop = 0.50
a_coh = np.where(coh2_raw < self._coh2_smooth, a_coh_drop, a_coh_rise)
self._coh2_smooth = a_coh * self._coh2_smooth + (1.0 - a_coh) * coh2_raw
```

**Stage**: `ResFilter.process()` (top-level orchestrator)

**Mechanism**: when `filter_converged`, slow the coh2 rise time
constant from TC≈80ms (α=0.80) to TC≈160ms (α=0.90). Drop time
constant unchanged (TC≈25ms).

**Downstream effect on coh2 → ne_evidence → ne_g_floor → spectral_g_min**:
Slower coh2 rise = more stable echo-only tracking = less FS
protection lift (good for FS); fast drop preserves DT protection
(good for DT).

**fq_usable substitution semantics**: `fq_usable=True` more often
→ slow-rise tuning activates more often → coh2 stays high longer
in echo-dominant frames → ne_g_floor stays low longer in FS →
**better FS suppression**. Predicted: small FS gain, neutral DT.

**Ablation flag candidate**: `c_e_branch_coh2_ema_use_fq_usable`

---

### Branch P2 — `process()` unified gain_floor mic-excess gate

**Location**: [python/aec.py:4197-4219](../python/aec.py#L4197)

```python
if self._unified_gain_floor and self._use_mic_excess_evidence:
    _lw_ready_floor = (self._residual_est is not None
                       and self._residual_est._long_window_n_updates > 0)
    if filter_converged and _lw_ready_floor and not epc_active:
        far_lw = self._residual_est._long_window_far_psd
        erl_e = ...
        excess = np.maximum(self.error_psd - far_lw * erl_e, 0.0)
        excess_ratio = np.clip(excess / (self.error_psd + 1e-10), 0.0, 1.0)
        legacy_evidence = 1.0 - coh2
        ne_evidence = (_BLEND_F31_MIC_EXCESS * excess_ratio
                       + (1.0 - _BLEND_F31_MIC_EXCESS) * legacy_evidence)
    else:
        ne_evidence = 1.0 - coh2
else:
    ne_evidence = 1.0 - coh2
```

**Stage**: `ResFilter.process()` orchestrator (between residual_model
and gain_compute)

**Mechanism**: when `_unified_gain_floor AND _use_mic_excess_evidence
AND filter_converged AND _lw_ready_floor AND not epc_active`, replace
`(1-coh2)` ne_evidence with F3.1 v3 mic-excess blend.

**Downstream effect on ne_evidence → ne_g_floor → spectral_g_min**:
Same per-bin NE evidence as Branch G1, but applied at a different
stage:
- ne_evidence × ne_erle_gate × (1 - fs_confidence) = ne_protection
- ne_g_floor = effective_g_min + (ne_g_min_ceil - effective_g_min) * ne_protection
- spectral_g_min = max(spectral_g_min, ne_g_floor) ← FLOOR LIFT

**Note**: `_unified_gain_floor` is a separate flag; in current
production it's typically OFF or in S6 audit-mode. So this branch may
not fire in BALANCED defaults. Verify in 1.2 design.

**Ablation flag candidate**: `c_e_branch_unified_gfloor_use_fq_usable`
(low priority if `_unified_gain_floor=False` in BALANCED)

---

### Branch P3 — `process()` startup_dt_curr → spectral_g_min cap

**Location**: [python/aec.py:4244, 4252-4253](../python/aec.py#L4244)

```python
_startup_dt_curr = effective_dt > 0.35 and far_power > 1e-4 and not filter_converged

if _startup_dt_curr and self.startup_dt_gain_floor < 1.0:
    spectral_g_min = np.minimum(spectral_g_min, self.startup_dt_gain_floor)
```

**Stage**: `ResFilter.process()` orchestrator (before gain_compute)

**Mechanism**: when `not filter_converged AND effective_dt > 0.35
AND far_power > 1e-4`, cap `spectral_g_min` from above by
`startup_dt_gain_floor` (default 1.0 = no effect; can be set lower
for ablation).

**Downstream effect on spectral_g_min**: caps maximum suppression
floor during startup-DT frames (allows more suppression even when
filter is not yet converged).

**fq_usable substitution semantics**: `fq_usable=True` more often →
**fewer frames classified as startup_dt_curr** → cap fires less →
spectral_g_min higher → **less suppression in those frames**.

**Note**: BALANCED default `startup_dt_gain_floor=1.0` → cap is
no-op in production today. **This branch is dead code in BALANCED**
(default 1.0 means `if 1.0 < 1.0` is always False).

**Ablation flag candidate**: skip (dead in BALANCED defaults)

---

## 3. Branch summary table

| # | Site | Stage | Default-fire-rate (BALANCED) | Effect on Wiener gain | Predicted fq_usable swap effect |
|---|---|---|---|---|---|
| R1 | residual_estimator force_render | `_stage_residual_model` | ~95% (filter_converged=False ~95%) | LARGE (linear vs render mode) | BIG FS regression (linear default → less default suppression) |
| G1 | gain_compute F3.1 v3 dt_per_bin | `_stage_gain_compute` | ~5% (filter_converged=True, gated) | MEDIUM (per-bin NE evidence) | DT NE preserve win; FS effect uncertain |
| P1 | coh2 EMA asymmetric | `process()` | ~5% slow-rise tuning | SMALL (coh2 stability) | Small FS gain, neutral DT |
| P2 | unified gain_floor mic-excess | `process()` | ~0% (gain_floor flag OFF in BALANCED) | MEDIUM if unified_gfloor enabled | Same shape as G1 (n/a if disabled) |
| P3 | startup_dt_gain_floor cap | `process()` | 0% (dead code in BALANCED, floor=1.0) | n/a | n/a (no behaviour change) |

**Live ablation candidates**: R1, G1, P1 (and conditionally P2).
P3 dropped (dead in BALANCED).

## 4. Hypothesis: which branch dominates v3.18 C.E whole-arg result?

v3.18 C.E whole-arg migration (replace `_filter_converged` with
`fq_usable` everywhere RES.process() reads it) showed:

| Bucket | Δecho | Δdeg | Source |
|---|---:|---:|---|
| FS_static | **−0.151** | +0.000 | Branch R1 likely dominant (force_render disabled → linear default → less suppression) |
| DT_static | −0.096 | **+0.214** | Branch G1 likely dominant (F3.1 v3 dt_per_bin path activates → per-bin NE preserve) |
| DT_movement | +0.033 | +0.234 | G1 + R1 net positive |
| NE | 0.000 | 0.000 | no FS/DT — branches don't fire |

**Working hypothesis (to test in 1.4 single-branch sweep)**:
- R1 alone: BIG FS regression, neutral DT
- G1 alone: DT win, FS unchanged
- P1 alone: small FS gain (consistent with intent)
- R1+G1 combined: matches v3.18 C.E whole-arg result

If hypothesis confirmed, **G1+P1** without R1 is the v3.19 ship
candidate (DT win + small FS gain, no FS regression).

## 5. Adjacencies (for Phase 1.1b SuppressionGain side-by-side)

These branches map to AEC3 SuppressionGain stages as follows
(detailed in 1.1b):

- R1 ↔ AEC3 `ResidualEchoEstimator::Estimate` — 3-mode switch
  (Saturated / UsableLinearEstimate / non-linear fallback). AEC3
  uses `aec_state.UsableLinearEstimate()` instead of
  `filter_converged`. **Direct AEC3 parallel**.
- G1 ↔ AEC3 `SuppressionGain::LowerBandGain` `nearend_smoother` —
  averages suppressor input over 8 frames as nearend evidence. Our
  F3.1 v3 mic-excess is per-frame; AEC3's is multi-frame. **Different
  primitive**, but same role (NE evidence into Wiener gain).
- P1 ↔ AEC3 has fixed coh-style smoothing per `subband_nearend_detector`.
  Our coh2 EMA asymmetry is a different smoother. **No direct AEC3
  parallel** for the asymmetric tuning.
- P2 ↔ AEC3 `SuppressionGain::GainParameters::SetConfig` per-bin
  LF→HF interpolation. Different mechanism.

Phase 1.1b will add line citations + structural mapping per branch.

## 6. Cross-references

- [docs/v3_18_c_e_closeout.md](v3_18_c_e_closeout.md) — Pareto evidence motivating per-branch ablation
- [docs/v3_18_c_e1_consumer_migration_design.md](v3_18_c_e1_consumer_migration_design.md) — whole-arg migration design (now superseded by per-branch approach)
- [docs/v3_19_phase0_4_aec3_unported_modules.md](v3_19_phase0_4_aec3_unported_modules.md) — Phase 1.1b scope
- [docs/aec3_residual_pipeline_mapping.md](aec3_residual_pipeline_mapping.md) — AEC3 SuppressionGain + ResidualEchoEstimator structural docs
- [python/aec.py:5116-5122](../python/aec.py#L5116) — Branch R1
- [python/aec.py:3298-3323](../python/aec.py#L3298) — Branch G1
- [python/aec.py:4060-4068](../python/aec.py#L4060) — Branch P1
- [python/aec.py:4197-4219](../python/aec.py#L4197) — Branch P2
- [python/aec.py:4244, 4252-4253](../python/aec.py#L4244) — Branch P3
