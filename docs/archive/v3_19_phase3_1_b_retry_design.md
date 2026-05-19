# v3.19 Phase 3.1 — B FilterMisadjustment retry design (2026-05-16)

**Status**: DESIGN LOCKED 2026-05-16. Confidence 65%.
**Scope**: B trigger redesign with `fq_usable + reset_done` gate +
threshold inversion (`<0.5` → `<2.0` for our ERL semantics). Re-uses
v3.18 Phase B substrate; adds new flag for retry semantics.

## 1. v3.18 Phase B closure cause (recap)

Per [docs/v3_18_b_closeout.md](v3_18_b_closeout.md): B.4 fire-rate
gate FAIL with **0/61 fires**. Root cause:

1. **Stability gate silent**: `refined_usable` coverage 0-38%; never
   reached `stable_count ≥ 30` consecutive frames. qNvSMyU never
   enters refined_usable; DT cases also rare.
2. **Threshold mismatch**: AEC3's `< 0.5` (under-model) calibrated
   for AEC3's smoothed clustering near 1.0. Our smoothed clusters at
   10-42× because `_erl_estimate` is a **conservative budget**
   (starts 0.1, capped 0.3) not "true ERL" — so our ratio is
   inflated by 5-15×. Excursions to 0.07-0.28 happen but are masked
   by gate.

Re-enable preconditions per closeout §"Re-enable":
1. ✓ Phase C FilteringQualityAnalyzer raises usable coverage to
   52-86% (v3.18 C.B PASS verdict)
2. Threshold redesign for our pipeline (this doc)
3. Co-design with our `_erl_estimate` semantics (this doc)

## 2. Phase 3 redesign — 3 changes

### 2.1 Gate change: `refined_usable` → `fq_usable + reset_done`

**Current** ([python/aec.py:6136-6141](../python/aec.py#L6136)):
```python
stable = (
    self._filter_converged
    and getattr(self, '_prev_filter_state', '') == 'refined_usable'
    and not self.epc_active
    and not self._regime_handler.main_paused
)
```

**Phase 3.1 redesign**:
```python
stable = (
    self._aec_state is not None
    and getattr(self._aec_state, '_aec_ref', None) is not None
    and self._aec_state.fq_usable()
    and not self.epc_active
    and not self._regime_handler.main_paused
    and self._misadjustment_reset_done  # NEW: reset hangover satisfied
)
```

`fq_usable` from C.B FilteringQualityAnalyzer (multi-gate:
startup_timer + reset_timer + convergence_seen + not_transparent;
broader than refined_usable).

`_misadjustment_reset_done` is a NEW per-frame flag set after the
filter has been stable for `reset_done_frames=20` frames (no recent
EPC, no recent main_paused, no recent leakage_diverged). Prevents
firing immediately after a reset event when filter state is in flux.

### 2.2 Threshold inversion: `< 0.5` → `< 2.0`

**Rationale**:
- AEC3 smoothed clusters near 1.0 → `< 0.5` catches dramatic
  under-model events (factor of 2× drop)
- Our smoothed clusters at 10-42× (5-15× inflation from conservative
  `_erl_estimate` budget) → equivalent factor-of-2 drop is
  `5-21 → 2.5-10` range
- New threshold `< 2.0` calibrated to catch dips that, on our scale,
  represent actual filter under-modelling (smoothed dropped to
  the same relative position as AEC3's `< 0.5` would catch)

**Cross-validation**: v3.18 closeout single-case data shows excursions
to smoothed_min in {0.067, 0.105, 0.281}. New threshold `< 2.0`
catches all three (would have produced >0 fires on these cases IF
gate were open).

### 2.3 New AecConfig flags (default-OFF)

```python
# v3.19 Phase 3 — B retry with fq_usable gate + threshold inversion
filter_misadjustment_use_fq_usable: bool = False
filter_misadjustment_reset_done_frames: int = 20
filter_misadjustment_threshold_phase3: float = 2.0  # was 0.5
```

When `filter_misadjustment_use_fq_usable=True`:
- Gate uses fq_usable + reset_done (per §2.1)
- Threshold uses `filter_misadjustment_threshold_phase3` (default
  2.0) instead of `filter_misadjustment_threshold` (legacy 0.5)

When `False`: byte-equal to v3.18 Phase B substrate (all gates per
legacy semantics).

### 2.4 Action direction unchanged

When fire fires: `scale = clamp(1.0 / smoothed, scale_min, scale_max)`.

For our smoothed dips (e.g., 0.105 → scale = 9.52, clamped to 2.0):
- `scale_max=2.0` caps the scale-up to 2× per fire
- After fire: `_misadjustment_smoothed = 1.0`, hangover armed for
  100 frames
- Effect: filter coefficients scaled UP, residual reduced,
  smoothed decoupled from immediate post-fire state

## 3. Phase 3 sprint plan (4-6 sprints)

| Sprint | Action | Hard bar |
|---|---|---|
| 3.1 | Design doc (this) | n/a |
| 3.2 | Implement: 3 new AecConfig fields + gate switch + threshold switch + reset_done logic; 5-case byte-equal flag-OFF | byte-equal md5 5/5 |
| 3.3 | Fire-rate gate on 60-case: 10 ≤ total_fires ≤ 200 with new gates | PASS/FAIL |
| 3.4 | If 3.3 PASS: 60-case A/B AECMOS | FS_static Δecho ≥ +0.010 (PRIMARY) AND DT bucket Δdeg ≥ −0.005 AND NE Δdeg ≥ −0.005 |
| 3.5 | If 3.4 PASS: 800-case + ship gate (user §0.7 re-auth) | hard bar all PASS |

### Hard bar (Phase 3) — pre-stated

PRIMARY: **FS_static Δecho ≥ +0.010** (B is FS-side mechanism;
gain expected here).

GUARDS:
- DT_static Δdeg ≥ −0.005
- DT_movement Δdeg ≥ −0.005
- FS_movement Δecho ≥ −0.010
- NE Δdeg ≥ −0.005
- Worst-case sample Δecho ≥ −0.05 dB
- qNvSMyU cohort tail no regression vs v3.19 baseline (per Phase 1.4
  baseline)

### Kill criteria

- 3.3 fire-rate < 10 (still silent) → close, substrate retained per
  §0.4. Conclusion: Phase C wasn't the only blocker.
- 3.3 fire-rate > 200 (over-firing) → re-tune hangover_frames /
  threshold_phase3 / reset_done_frames; one threshold-sweep
  sub-sprint allowed before close
- 3.4 PRIMARY FAIL or any guard FAIL → close

## 4. Risk register

| Risk | Severity | Mitigation |
|---|---|---|
| Over-firing destabilises filter | HIGH | scale_max=2.0 caps single-fire impact; 100-frame hangover prevents back-to-back fires; reset_done_frames=20 prevents fire during reset transients |
| qNvSMyU cohort tail still 0% (fq_usable also 0% there) | MEDIUM | If 3.3 audit shows 0 fires on qNvSMyU specifically, document and continue (qNvSMyU is divergence-defence territory anyway, per [project_p52_task_a0_verdict.md](../../../.claude/projects/-Users-mingyu-Desktop-novatek-SE/memory/project_p52_task_a0_verdict.md)) |
| DT regression from spurious fires during NE-rich frames | MEDIUM | epc_active gate stays; hangover prevents short-burst fires; 3.4 60-case A/B includes DT_static + DT_movement guards |
| Threshold 2.0 too aggressive (over-fires on benign FS frames) | LOW | 3.3 fire-rate gate catches this with explicit upper bound 200 |
| C.B fq_usable wired correctly but qNvSMyU still 0% | LOW | C.B v3.18 verdict reports 52-86% coverage on FS / DT buckets; cohort tail may be lower |

## 5. Cross-references

- [docs/v3_18_b_closeout.md](v3_18_b_closeout.md) — v3.18 Phase B
  closure (silent failure root cause)
- [docs/v3_18_b1_misadjustment_design.md](v3_18_b1_misadjustment_design.md) — v3.18 Phase B design
- [docs/v3_18_c_b_verdict.md](v3_18_c_b_verdict.md) — C.B PASS
  (fq_usable substrate ready)
- [python/aec.py:6098-6163](../python/aec.py#L6098) — current B
  implementation
- [python/aec.py:5274](../python/aec.py#L5274) — `AecState.fq_usable()`
  accessor
- `~/.claude/plans/se-aec-aec-main-hazy-lynx.md` §Phase 3
