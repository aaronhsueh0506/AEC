# v3.19 Phase 1.1b — AEC3 SuppressionGain side-by-side annotation

**Status**: COMPLETE 2026-05-16
**Scope**: per Phase 1.1a branch (R1, G1, P1, P2, P3), cite the
matching AEC3 line ranges in `docs/aec3_extracts/src/aec3/` and
identify the structural ↔ section mapping. Output informs Phase 1.2
per-branch flag design and v3.20+ SuppressionGain port arc.

## 1. AEC3 inputs (source of authority)

| File | Lines | Role |
|---|---|---|
| `docs/aec3_extracts/src/aec3/aec_state.h` | 50-80 | `UsableLinearEstimate()` definition (single multi-gate predicate) |
| `docs/aec3_extracts/src/aec3/aec_state.cc` | 399-442 | UsableLinearEstimate computation (sufficient_data_to_converge_at_startup AND (external_delay OR convergence_seen) AND NOT transparent_mode) |
| `docs/aec3_extracts/src/aec3/residual_echo_estimator.cc` | 200-280 | R2 producer with `aec_state.UsableLinearEstimate()` switch (lines 210, 242) |
| `docs/aec3_extracts/src/aec3/suppression_gain.cc` | 290-340 | `LowerBandGain` Wiener envelope (`nearend_smoother`, `WeightEchoForAudibility`, `GetMinGain`/`GetMaxGain`, `GainToNoAudibleEcho`) |
| `docs/aec3_extracts/src/aec3/suppression_gain.cc` | 480-512 | `GainParameters::SetConfig` per-bin LF→HF interpolation |

Our `fq_usable` = AEC3's `UsableLinearEstimate()` semantics:
multi-gate (sufficient_startup AND startup_done AND
convergence_seen AND not transparent_mode). Per
[python/aec.py:5274](../python/aec.py#L5274) +
[aec_filter_quality.py](../python/aec_filter_quality.py).

## 2. Per-branch annotation

### Branch R1 — `attribute_legacy.force_render` ↔ AEC3 ResidualEchoEstimator

**Our code** ([python/aec.py:5116-5122](../python/aec.py#L5116)):
```python
force_render = (
    epc_active
    or saturation_level > 0.5
    or not filter_converged
    or (self._arc_t_force_render_or_in_enabled
        and self._arc_t_cohort_tail_signal)
)
```

**AEC3 equivalent**
([residual_echo_estimator.cc:208-242](aec3_extracts/src/aec3/residual_echo_estimator.cc)):
```cpp
is_ml_ree_active_ = neural_residual_echo_estimator_ != nullptr &&
                    aec_state.UsableLinearEstimate();
...
if (!is_ml_ree_active_) {
    if (aec_state.UsableLinearEstimate()) {
        if (aec_state.SaturatedEcho()) {  // → R2 = Y2 (mic passthrough)
            ...
        } else {
            LinearEstimate(S2_linear, aec_state.Erle(...), R2);
        }
        UpdateReverb(...);  AddReverb(R2);
    } else {  // → render-based fallback
        ...
    }
}
```

**Structural mapping**:
- AEC3's `aec_state.UsableLinearEstimate()` is the GATE for linear vs render-based R2
- AEC3 has 3 modes: SaturatedEcho (Y2 passthrough) → UsableLinearEstimate (linear ERLE) → fallback (render-based)
- Ours has 4 OR-conditions for force_render: `epc_active OR saturation OR not filter_converged OR cohort_tail_signal`
- AEC3's `SaturatedEcho()` ≈ our `saturation_level > 0.5`
- AEC3's `!UsableLinearEstimate()` ≈ our `not filter_converged` (with broader semantics)
- AEC3 has NO `epc_active` equivalent (no EPC concept; uses
  `convergence_seen_` reset on echo path change)

**Direct match strength**: HIGH. R1 is the closest 1:1 mapping in
the codebase. The fq_usable substitution semantics for R1 are:
"AEC3-style linear/non-linear gate."

**Per-branch flag design**: `c_e_branch_force_render_use_fq_usable`
controls only the third OR-condition; epc_active / saturation /
cohort_tail_signal stay on `_filter_converged` semantics
(or stay literal flags).

---

### Branch G1 — `_stage_gain_compute` F3.1 v3 dt_per_bin ↔ AEC3 NO direct equivalent

**Our code** ([python/aec.py:3298-3323](../python/aec.py#L3298)):
F3.1 v3 mic-excess blend for per-bin NE evidence.

**AEC3 nearest analog**
([suppression_gain.cc:308](aec3_extracts/src/aec3/suppression_gain.cc#L308)):
```cpp
nearend_smoothers_[ch].Average(suppressor_input[ch], nearend);
```

`nearend_smoothers_` is an 8-frame moving-average of the
`suppressor_input` magnitude per channel. The output `nearend` is
then fed to `GainToNoAudibleEcho(nearend, weighted_residual_echo,
comfort_noise[0], &G)` which computes Wiener-like gain.

**Structural mapping**:
- Our `dt_per_bin` is per-frame; AEC3's `nearend` is 8-frame averaged
- Our F3.1 mic-excess uses `error_psd - far_lw * erl_estimate` as
  evidence; AEC3 uses raw suppressor input magnitude (post-pre-filter)
- AEC3 also has `DominantNearendDetector` (binary flag) AND
  `nearend_smoother` (per-bin level); our G1 fuses these into a
  single per-bin float
- AEC3's nearend smoother does NOT gate on UsableLinearEstimate —
  it's always-on. **AEC3 does not gate NE evidence on filter
  health.**

**Match strength**: LOW. G1 is conceptually similar (per-bin NE
evidence) but mechanistically different. AEC3 ports won't directly
reuse G1's gate logic.

**Per-branch flag design**: `c_e_branch_dt_per_bin_use_fq_usable`
controls only the `filter_converged` clause; the `_lw_ready` and
`(dt_per_bin_unified OR not epc_active)` gates stay literal.

---

### Branch P1 — `process()` coh2 EMA asymmetric ↔ AEC3 NO equivalent

**Our code** ([python/aec.py:4060-4068](../python/aec.py#L4060)):
```python
if filter_converged:
    a_coh_rise = 0.90; a_coh_drop = 0.50  # TC≈160ms / 25ms
else:
    a_coh_rise = 0.80; a_coh_drop = 0.50  # TC≈80ms / 25ms
```

**AEC3 equivalent**: NONE. AEC3 does not maintain a coherence-based
echo signal in this form. AEC3's `coh` analog is in
`subband_nearend_detector.cc` for HF/LF NE classification, not in
the gain pipeline.

**Match strength**: NONE. P1 is unique to our pipeline. v3.20+ port
work does not need to reconcile P1 with AEC3.

**Per-branch flag design**: `c_e_branch_coh2_ema_use_fq_usable`
controls the `if filter_converged` decision; same body otherwise.

---

### Branch P2 — `process()` unified gain_floor ↔ AEC3 GainParameters per-bin LF/HF

**Our code** ([python/aec.py:4197-4219](../python/aec.py#L4197)):
F3.1 v3 mic-excess blend for ne_evidence (when `_unified_gain_floor`
ON; default OFF in BALANCED).

**AEC3 equivalent**
([suppression_gain.cc:480-512](aec3_extracts/src/aec3/suppression_gain.cc#L480)):
```cpp
void SuppressionGain::GainParameters::SetConfig(
    int last_lf_band, int first_hf_band,
    const EchoCanceller3Config::Suppressor::Tuning& tuning) {
  for (int k = 0; k < kFftLengthBy2Plus1; k++) {
    float a;
    if (k <= last_lf_band)       a = 0.f;
    else if (k < first_hf_band)  a = (k - last_lf_band) / float(...);
    else                         a = 1.f;
    enr_transparent_[k] = (1 - a) * lf.enr_transparent + a * hf.enr_transparent;
    ...
  }
}
```

**Structural mapping**:
- AEC3 sets per-bin (`enr_transparent_[k]`, `enr_suppress_[k]`,
  `emr_transparent_[k]`) thresholds via LF→HF linear interpolation,
  configured ONCE at construction
- Ours computes ne_evidence per frame, gates on filter_converged
- AEC3's per-bin masking thresholds are STATIC anchor values; OURS is
  DYNAMIC per-frame mic-excess
- Different design philosophy. AEC3 uses binary
  `dominant_nearend_detector` to switch between two pre-computed
  threshold tables (`nearend_params_` / `normal_params_`); we use a
  per-bin per-frame gradient

**Match strength**: LOW. AEC3's per-bin thresholds and our P2 mic-
excess are different mechanisms. v3.20+ port could replace P2 with
AEC3-style per-bin tables (similar to v3.18 D-γ which closed CANNOT
SHIP — see Phase 0.3 D-γ retry doc).

**Per-branch flag design**: `c_e_branch_unified_gfloor_use_fq_usable`.
LOW PRIORITY — `_unified_gain_floor=False` in BALANCED, branch
doesn't fire.

---

### Branch P3 — `process()` startup_dt_curr cap ↔ AEC3 startup transition logic

**Our code** ([python/aec.py:4244, 4252-4253](../python/aec.py#L4244)):
```python
_startup_dt_curr = effective_dt > 0.35 and far_power > 1e-4 and not filter_converged
if _startup_dt_curr and self.startup_dt_gain_floor < 1.0:
    spectral_g_min = np.minimum(spectral_g_min, self.startup_dt_gain_floor)
```

**AEC3 equivalent**: AEC3 has `InitialState` and
`TransparentModeActive` per AecState; suppressor_config has
`startup_dt_*` analog via
`config.suppressor.high_bands_suppression.enr_threshold_during_startup`
(not in the extracts here). AEC3 startup logic is closely tied to
`InitialState` lifecycle, which is **v3.20+ backlog Tier 4** (per
v3.18 cycle closeout).

**Match strength**: MEDIUM. Conceptually equivalent (special handling
in startup), structurally different (AEC3 lifecycle vs our binary
gate).

**Per-branch flag design**: SKIP. P3 is dead code in BALANCED defaults
(startup_dt_gain_floor=1.0 → cap is no-op). No per-branch flag needed
in v3.19 Phase 1.

---

## 3. Summary table

| # | Branch | AEC3 file | AEC3 lines | Match strength | v3.20+ port relevance |
|---|---|---|---|---|---|
| R1 | force_render | residual_echo_estimator.cc | 208-242 | HIGH (1:1) | High — direct port candidate |
| G1 | dt_per_bin F3.1 | suppression_gain.cc | 308 (nearend_smoother) | LOW | Different primitive; AEC3 won't subsume |
| P1 | coh2 EMA asymmetric | (none) | n/a | NONE | Our-only mechanism; not portable |
| P2 | unified gain_floor | suppression_gain.cc | 480-512 (GainParameters) | LOW | Different design (static thresholds vs dynamic per-frame) |
| P3 | startup_dt cap | (lifecycle in InitialState) | n/a | MEDIUM | v3.20+ Tier 4 InitialState arc |

## 4. Implications for Phase 1.2 design

### 4.1 Per-branch flag count

Phase 1.2 will define **3 live per-branch flags** (R1, G1, P1) plus
**1 conditional flag** (P2 — only useful if `_unified_gain_floor`
becomes ON in BALANCED via separate audit). P3 is skipped.

Flag count cap (per plan §1.2): "5-10 branches expected" — we are at
3-4 actual flags, well within bounds.

### 4.2 Hypothesis refinement

Branch R1 has DIRECT AEC3 parallel (`UsableLinearEstimate` semantic).
This is the strongest case for "fq_usable substitution preserves
algorithm correctness" — AEC3 has been shipping this gate for years.

Branch G1 has NO AEC3 parallel (NE evidence isn't gated on filter
health in AEC3). v3.18 C.E whole-arg result suggests G1 is the
**source of the +1.2 Δdeg single-case win** — but the win is on a
mechanism that AEC3 doesn't validate (i.e., we may be discovering
something AEC3 didn't try, or we may be hitting a fragile sweet spot).

This bifurcates v3.19 risk:
- R1-only swap: well-understood; predict R1 is the dominant FS-cost
  driver (AEC3 with usable-linear=False forces render-based; we
  similarly default to render-based when filter_converged=False).
  Swapping to fq_usable means linear-mode default → matches AEC3's
  steady-state behavior. Risk: **likely FS regression** (because
  fq_usable is broader, so linear-mode fires on FRESHER convergence
  than AEC3 would gate on).
- G1-only swap: less-understood; the C.E +1.2 Δdeg evidence is here.
  Risk: per-frame F3.1 dt_per_bin firing more often may over-attribute
  NE in transitional states (where filter_converged=False but
  fq_usable=True).
- P1-only swap: small effect; coh2 EMA tuning. Low risk.

### 4.3 Recommended Phase 1.4 sweep order

1. **G1 alone** first — most likely to PASS PRIMARY (DT Δdeg ≥ +0.020)
2. **P1 alone** second — small effect, fast confirm
3. **R1 alone** last — most likely to FAIL FS guard (predict
   R1 is the FS regression driver in v3.18 C.E whole-arg result)

If G1 alone passes the bar, ship G1+P1. Skip R1.

## 5. Cross-references

- [docs/v3_19_phase1_1a_resfilter_branch_inventory.md](v3_19_phase1_1a_resfilter_branch_inventory.md) — branch enumeration
- [docs/aec3_residual_pipeline_mapping.md](aec3_residual_pipeline_mapping.md) — AEC3 SuppressionGain + ResidualEchoEstimator structural docs
- [docs/v3_18_c_e_closeout.md](v3_18_c_e_closeout.md) — whole-arg result (G1+R1+P1+P2 combined)
- [docs/v3_19_phase0_4_aec3_unported_modules.md](v3_19_phase0_4_aec3_unported_modules.md) — Phase 1.1b scope decision
- `docs/aec3_extracts/src/aec3/residual_echo_estimator.cc:208-242` — R1 source
- `docs/aec3_extracts/src/aec3/suppression_gain.cc:308` — G1 nearest analog
- `docs/aec3_extracts/src/aec3/suppression_gain.cc:480-512` — P2 nearest analog
