# v3.19 Phase 1.2 — Per-branch flag design lock

**Status**: DESIGN LOCKED 2026-05-16. Confidence 70%.
**Scope**: define 3 live per-branch flags for Phase 1.3 implementation.
Each flag toggles a single ResFilter branch from `_filter_converged`
semantics to `fq_usable` (AecState back-ref to FilteringQualityAnalyzer).

## 1. Flag inventory

Per Phase 1.1a + 1.1b, the live ablation candidates are R1, G1, P1.
P2 conditional (skip unless `_unified_gain_floor=True`); P3 dead in
BALANCED.

### Flag 1: `c_e_branch_force_render_use_fq_usable`

**Default**: `False` (byte-equal to v3.18 production)
**Branch**: R1 — `attribute_legacy.force_render` OR-condition
**Site**: [python/aec.py:5119](../python/aec.py#L5119)

**Mechanism**: when ON, the `not filter_converged` clause in the
force_render OR is replaced by `not fq_usable`.

**Patch shape**:
```python
# Existing
force_render = (
    epc_active
    or saturation_level > 0.5
    or not filter_converged
    or (self._arc_t_force_render_or_in_enabled
        and self._arc_t_cohort_tail_signal)
)

# After patch
_fc_arg = filter_converged
if (self._c_e_branch_force_render_use_fq_usable
        and aec_state is not None
        and getattr(aec_state, '_aec_ref', None) is not None):
    _fc_arg = aec_state.fq_usable()
force_render = (
    epc_active
    or saturation_level > 0.5
    or not _fc_arg
    or (self._arc_t_force_render_or_in_enabled
        and self._arc_t_cohort_tail_signal)
)
```

**Env var**: `AEC_C_E_BRANCH_FORCE_RENDER_FQ_USABLE` (1 / 0)

---

### Flag 2: `c_e_branch_dt_per_bin_use_fq_usable`

**Default**: `False`
**Branch**: G1 — F3.1 v3 mic-excess-evidence dt_per_bin path
**Site**: [python/aec.py:3298-3301](../python/aec.py#L3298)

**Mechanism**: when ON, replaces `filter_converged` in the F3.1 v3
gate with `fq_usable`.

**Patch shape** (in `_stage_gain_compute`, signature already takes
`filter_converged`):
```python
# Existing
if (self._use_mic_excess_evidence
        and filter_converged
        and _lw_ready
        and (self._dt_per_bin_unified or not epc_active)):
    ...

# After patch — needs aec_state passed in
_fc_g1 = filter_converged
if (self._c_e_branch_dt_per_bin_use_fq_usable
        and aec_state is not None
        and getattr(aec_state, '_aec_ref', None) is not None):
    _fc_g1 = aec_state.fq_usable()
if (self._use_mic_excess_evidence
        and _fc_g1
        and _lw_ready
        and (self._dt_per_bin_unified or not epc_active)):
    ...
```

**Required signature change**: `_stage_gain_compute` must take
`aec_state=None` parameter (currently doesn't). Caller in
`process()` line ~4255 already has `aec_state` in scope.

**Env var**: `AEC_C_E_BRANCH_DT_PER_BIN_FQ_USABLE`

---

### Flag 3: `c_e_branch_coh2_ema_use_fq_usable`

**Default**: `False`
**Branch**: P1 — coh2 EMA asymmetric tuning
**Site**: [python/aec.py:4060-4068](../python/aec.py#L4060)

**Mechanism**: when ON, switch the `if filter_converged` condition
to `if fq_usable_p1` for the slow-rise (α=0.90) tuning.

**Patch shape**:
```python
# Existing
if filter_converged:
    a_coh_rise = 0.90
    a_coh_drop = 0.50
else:
    a_coh_rise = 0.80
    a_coh_drop = 0.50

# After patch
_fc_p1 = filter_converged
if (self._c_e_branch_coh2_ema_use_fq_usable
        and aec_state is not None
        and getattr(aec_state, '_aec_ref', None) is not None):
    _fc_p1 = aec_state.fq_usable()
if _fc_p1:
    a_coh_rise = 0.90
    a_coh_drop = 0.50
else:
    a_coh_rise = 0.80
    a_coh_drop = 0.50
```

**Env var**: `AEC_C_E_BRANCH_COH2_EMA_FQ_USABLE`

---

### Flag set: `_c_e_branch_*` instance attributes

Three new ResFilter instance attributes set from AecConfig:
- `self._c_e_branch_force_render_use_fq_usable = config.c_e_branch_force_render_use_fq_usable`
- `self._c_e_branch_dt_per_bin_use_fq_usable = config.c_e_branch_dt_per_bin_use_fq_usable`
- `self._c_e_branch_coh2_ema_use_fq_usable = config.c_e_branch_coh2_ema_use_fq_usable`

(Or read directly off `self.config` if ResFilter holds a config ref.)

## 2. AecConfig changes

Three new bool fields in [python/aec.py:204-289 region](../python/aec.py#L204):

```python
# v3.19 Phase 1 — per-RES-branch C.E migration. Per-branch ablation
# of the v3.18 C.E whole-arg switch; identifies which RES branches
# benefit from fq_usable (AEC3-aligned multi-gate) vs hurt from it.
# All default OFF; byte-equal to v3.18 production.
# Flag-OFF semantics identical to legacy `_filter_converged`.
c_e_branch_force_render_use_fq_usable: bool = False
c_e_branch_dt_per_bin_use_fq_usable: bool = False
c_e_branch_coh2_ema_use_fq_usable: bool = False
```

Note: v3.18's `c_e_res_use_fq_usable` (whole-arg switch) stays in
AecConfig — it's the v3.18 substrate. When ANY per-branch flag is
ON, the whole-arg switch becomes redundant for that branch (per-branch
flag wins). When per-branch flag OFF AND whole-arg OFF, branch
reads `_filter_converged` (production behaviour).

## 3. eval_aec_challenge.py env-var bridges

Three new bridges to insert near existing `AEC_C_E_RES_FQ_USABLE`:

```python
if os.getenv('AEC_C_E_BRANCH_FORCE_RENDER_FQ_USABLE') == '1':
    cfg.c_e_branch_force_render_use_fq_usable = True
if os.getenv('AEC_C_E_BRANCH_DT_PER_BIN_FQ_USABLE') == '1':
    cfg.c_e_branch_dt_per_bin_use_fq_usable = True
if os.getenv('AEC_C_E_BRANCH_COH2_EMA_FQ_USABLE') == '1':
    cfg.c_e_branch_coh2_ema_use_fq_usable = True
```

## 4. Wiring requirements

### 4.1 `_stage_gain_compute` signature change

Currently:
```python
def _stage_gain_compute(self, *, residual_echo_psd, eer, coh2,
                          effective_dt, is_stationary_dt, far_power,
                          filter_once_converged, spectral_g_min, eps,
                          erl_estimate=0.01, filter_converged=False,
                          epc_active=False, filter_state='idle'):
```

After patch:
```python
def _stage_gain_compute(self, *, residual_echo_psd, eer, coh2,
                          effective_dt, is_stationary_dt, far_power,
                          filter_once_converged, spectral_g_min, eps,
                          erl_estimate=0.01, filter_converged=False,
                          epc_active=False, filter_state='idle',
                          aec_state=None):
```

Caller update: `process()` line ~4255 passes `aec_state=aec_state`
(value already in scope).

### 4.2 `attribute_legacy` signature

Already takes `aec_state=None` via `attribute()` dispatcher. The patch
in §1 Branch R1 just needs the `aec_state` reference — it's already
available in `kw['aec_state']` if passed. **Verify** that
`attribute_legacy` receives `aec_state`. Looking at line 5048-5055:
```python
return self.attribute_legacy(**kw)
```
where `kw` is the union of legacy+split kwargs and includes
`aec_state` per `attribute_legacy` signature. **Verify** in 1.3 that
`attribute_legacy` accepts `aec_state` (it currently does NOT — only
`attribute_split` does). This needs fixing as part of 1.3.

**Fix**: add `aec_state=None` to `attribute_legacy` signature; pass
through from caller; use only if R1 flag ON.

### 4.3 `process()` coh2 EMA site

Already inside `process()`, `aec_state` parameter already in scope
(passed from caller).

## 5. Byte-equal flag-OFF guarantee

Each flag-OFF path uses the existing `filter_converged` value,
verbatim. No mutation. No reordering of operations. The guard
chain is:

```python
_fc_xxx = filter_converged
if FLAG_ON and aec_state is not None and back_ref_set:
    _fc_xxx = aec_state.fq_usable()
```

Three guards (flag, aec_state, back_ref) ensure that:
- Flag OFF → `_fc_xxx = filter_converged` (byte-equal)
- Flag ON but aec_state None → `_fc_xxx = filter_converged` (byte-equal)
- Flag ON but back_ref unset → `_fc_xxx = filter_converged` (byte-equal)
- Flag ON + aec_state + back_ref → `_fc_xxx = aec_state.fq_usable()` (NEW behaviour)

## 6. Confidence check (60% PRIMARY bar)

Per Phase 1 plan §1.2 hard bar: design lock + 60% confidence.

**Confidence: 70%**. Justification:
- v3.18 C.E whole-arg result is reproducible empirical evidence (60-case bench)
- Branch decomposition into R1/G1/P1 is clean (no overlapping state mutations)
- Each per-branch flag is a 3-5 line patch; risk is low
- Predicted single-branch effects (Phase 1.1a §3) are testable in 1.4
- Risk: G1 may not be the +1.2 source — could be R1 alone or
  R1+G1 interaction. If 1.4 sweep shows no positive single-branch,
  Phase 1 closes per kill criterion (acceptable outcome — substrate
  retained, Volterra Option (B) auto-trigger fires).

## 7. Pre-stated kill criteria

### Per-flag (sprint 1.4 single-branch sweep)

For each of {R1, G1, P1} alone on 60-case:
- **PASS-positive**: rank by `(Δdeg_DT_static + Δdeg_DT_movement) -
  (|Δecho_FS_static| + |Δecho_FS_movement|)`
- **PASS-skip**: top score < +0.005 → branch dropped from combined
  pool

### Combined (sprint 1.5)

Combined config of top-ranked branches must satisfy all of:
- DT_static Δdeg ≥ **+0.020** (PRIMARY)
- DT_movement Δdeg ≥ −0.005
- FS_static Δecho ≥ −0.010
- FS_movement Δecho ≥ −0.010
- NE Δdeg ≥ −0.005
- worst-case sample Δecho ≥ −0.05 dB
- qNvSMyU cohort tail no regression vs v3.18 baseline

If **any** guard fails → Phase 1 close, Phase 2 only proceeds if
1.5 result has any directional gain (mid-cycle reassessment).

## 8. Cross-references

- [docs/v3_19_phase1_1a_resfilter_branch_inventory.md](v3_19_phase1_1a_resfilter_branch_inventory.md) — branches
- [docs/v3_19_phase1_1b_aec3_suppressiongain_annotation.md](v3_19_phase1_1b_aec3_suppressiongain_annotation.md) — AEC3 mapping
- [docs/v3_18_c_e_closeout.md](v3_18_c_e_closeout.md) — whole-arg evidence
- [docs/v3_18_c_e1_consumer_migration_design.md](v3_18_c_e1_consumer_migration_design.md) — v3.18 whole-arg implementation reference
- [python/aec.py:5274](../python/aec.py#L5274) — `AecState.fq_usable()` accessor
- [python/aec.py:5119](../python/aec.py#L5119) — R1 site
- [python/aec.py:3298](../python/aec.py#L3298) — G1 site
- [python/aec.py:4060](../python/aec.py#L4060) — P1 site
- [python/eval_aec_challenge.py](../python/eval_aec_challenge.py) — env-var bridges insertion point
