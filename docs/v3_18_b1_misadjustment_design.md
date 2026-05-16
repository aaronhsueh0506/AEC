# v3.18 Phase B.1 — FilterMisadjustmentEstimator + ScaleFilter (design lock, doc-only) — 2026-05-16

**Status**: B.1 DESIGN SPRINT — doc only, no code edits.
**Branch**: `feature/v3.18-aec3-fetch` (HEAD after Phase A closeout).
**Pivot context**: Phase A CLOSED CANNOT SHIP 2026-05-16
([docs/v3_18_a_closeout.md](v3_18_a_closeout.md)); §11 fallback
prescribed Phase B as next lower-risk arc.

## 0. One-paragraph framing

AEC3 has a `FilterMisadjustmentEstimator` that tracks long-term echo/error
ratio drift, and a `ScaleFilter` action that rescales the filter weights
(W *= scale) when the estimator detects systematic under-/over-modelling.
We have **no equivalent** — our `_diag['filter_scale_ratio']` already
computes the underlying signal but it's trace-only, never feedback into
W. This is Gap #5 + #13 from
[aec3_reference.md §6.2](aec3_reference.md#L811).

Unlike Phase A (replaced an existing mechanism), B **adds a new
mechanism**. Risk profile is different: false-positive over-correction
could destabilise convergence, but the baseline behaviour is preserved
when the estimator doesn't fire.

## 1. Audit answer — current code state

### 1.1 Existing substrate

| Item | Location | State |
|---|---|---|
| `_diag['filter_scale_ratio']` trace | [aec.py:8683-8689](../python/aec.py#L8683) | `echo_psd_mean / (far_psd_mean × ERL_est)` — same numerator AEC3 uses |
| `_erl_estimate` (scalar ERL) | [aec.py:6260+](../python/aec.py#L6260) | dynamic estimate already maintained per frame |
| `PBFDKF._update_weights` | [aec.py:1916-2027](../python/aec.py#L1916) | only PBFDKF; PBFDAF base has its own at 1807 |
| `AecConfig` knobs | [aec.py:146-365](../python/aec.py#L146) | room for new fields |
| `PBFDAF.copy_weights_from` | [aec.py:1832](../python/aec.py#L1832) | only `self.W[:] = src.W` — scale_filter needs a similar new method |

### 1.2 What's missing for AEC3 alignment

1. **Smoothed misadjustment state** — `_misadjustment_smoothed` (asymmetric
   EMA on `|e|² / |y|²` so the estimator tracks slow drift, not transients)
2. **`IsAdjustmentNeeded` trigger** — combination of:
   - misadjustment > threshold (under-modelling)
   - signal-stable hold (avoid DT-transients triggering)
   - convergence-seen (don't fire during startup; require usable filter)
3. **`scale_filter(scale)` primitive** — multiplicative `W *= scale`
   applied to all partitions; decision on whether `P *= scale²` (Option A)
   or P untouched (Option B per AEC3) — design choice §3.3 below
4. **Wiring**: estimator updates per frame; trigger fires conditionally
   when stable; ScaleFilter applies via PBFDKF method call

## 2. Design lock — flag, state, action

### 2.1 New `AecConfig` fields

```python
# v3.18 Phase B.2 — Filter misadjustment estimator + ScaleFilter.
# Tracks long-term echo/error ratio drift and rescales W when systematic
# under-modelling is detected. AEC3-aligned (Gap #5/#13).
# Flag-OFF (default): no estimator updates, no scale action. Byte-equal.
# Flag-ON: estimator + trigger + scale_filter all active.
filter_misadjustment_enabled: bool = False
# Misadjustment EMA coefficients (asymmetric so under-modelling rises
# slowly but recovery is faster — AEC3-aligned).
filter_misadjustment_alpha_up: float = 0.99   # slow accumulation
filter_misadjustment_alpha_dn: float = 0.95   # fast recovery
# Threshold for IsAdjustmentNeeded: scale_ratio < threshold → under-modelling
# (e.g. 0.5 means echo_psd is half what render says it should be). AEC3 uses
# 0.5; we mirror that.
filter_misadjustment_threshold: float = 0.5
# Min frames between consecutive ScaleFilter fires (hangover).
filter_misadjustment_hangover_frames: int = 100
# Stability gate: filter_converged AND not (epc_active OR DT) must hold for
# this many consecutive frames before scale_filter is allowed to fire.
filter_misadjustment_stable_frames: int = 30
# Scale value applied to W when fire: W *= scale_factor.
# AEC3 uses ratio-derived scale; we conservative-clamp the proposed scale
# so a single fire can't move W by more than 50% in either direction.
filter_misadjustment_scale_min: float = 0.5
filter_misadjustment_scale_max: float = 2.0
# P update on scale (design choice §3.3): True → P *= scale² (Option A,
# Kalman-canonical); False → P untouched (Option B, AEC3-aligned).
filter_misadjustment_scale_p: bool = False
```

**Naming check**: all `filter_misadjustment_*` mechanism descriptors,
no version-iteration suffixes. ✓

### 2.2 New AEC state (lazy-init at first frame to preserve byte-equal)

```python
# In AEC.__init__ (only allocated when flag is True; flag-OFF skips).
if self.config.filter_misadjustment_enabled:
    self._misadjustment_smoothed = 1.0   # initial = neutral
    self._misadjustment_stable_count = 0
    self._misadjustment_hangover_remaining = 0
    self._misadjustment_fire_count = 0   # diagnostic
```

### 2.3 Estimator update (per frame, after main filter processes)

```python
def _update_misadjustment_estimator(self):
    """v3.18 Phase B.2 — asymmetric EMA on filter_scale_ratio.

    scale_ratio = echo_psd_mean / (far_psd_mean × ERL_est)
      - 1.0: filter echo estimate matches render-derived estimate
      - <1.0: filter under-models (W magnitude too small)
      - >1.0: filter over-models (W magnitude too large)

    Asymmetric EMA: when ratio drops (worse), accumulate slowly so we
    don't react to transients; when ratio rises (recovering), update
    fast. This biases the smoothed value toward the long-term drift
    rather than the noise floor.
    """
    if not self.config.filter_misadjustment_enabled:
        return
    _fpw = float(np.mean(np.abs(self.filter.echo_spec) ** 2))
    _fp_far = float(np.mean(np.abs(self.filter.far_spec) ** 2))
    if _fp_far < 1e-10:
        return    # no far-end activity; estimator doesn't update
    raw_ratio = _fpw / (_fp_far * float(self._erl_estimate) + 1e-12)
    if raw_ratio < self._misadjustment_smoothed:
        alpha = self.config.filter_misadjustment_alpha_up
    else:
        alpha = self.config.filter_misadjustment_alpha_dn
    self._misadjustment_smoothed = (
        alpha * self._misadjustment_smoothed
        + (1.0 - alpha) * raw_ratio
    )
```

Callsite: `AEC.process()`, after main `self.filter.process()` returns
and before residual stage. Trace exposure: write
`self._diag['misadjustment_smoothed']`.

### 2.4 Trigger logic — `IsAdjustmentNeeded`

```python
def _check_misadjustment_trigger(self) -> bool:
    """Per AEC3 ScaleFilter trigger: stable convergence + under-model evidence.

    Gates:
      - filter_converged AND _filter_state == 'refined_usable'
      - NOT epc_active, NOT current-frame DT detected
      - misadjustment_smoothed < threshold
      - stable_count ≥ filter_misadjustment_stable_frames
      - hangover countdown == 0
    """
    if not self.config.filter_misadjustment_enabled:
        return False
    if self._misadjustment_hangover_remaining > 0:
        self._misadjustment_hangover_remaining -= 1
        return False
    stable = (
        self._filter_converged
        and getattr(self, '_prev_filter_state', '') == 'refined_usable'
        and not self.epc_active
        and not self._regime_handler.main_paused
    )
    if not stable:
        self._misadjustment_stable_count = 0
        return False
    self._misadjustment_stable_count += 1
    if self._misadjustment_stable_count < self.config.filter_misadjustment_stable_frames:
        return False
    if self._misadjustment_smoothed >= self.config.filter_misadjustment_threshold:
        return False
    return True
```

Trigger fires at most every `(stable_frames + hangover_frames) ≈ 130` frames
under flag-ON (≈ 2.1 sec at 16 kHz, 10ms frame). Coarse correction cadence
matching AEC3.

### 2.5 ScaleFilter action

New method on PBFDAF (inherited by PBFDKF):

```python
def scale_filter(self, scale: float) -> None:
    """Multiplicative W rescale across all partitions.

    AEC3-aligned (subtractor.cc ScaleFilter). Multiplies W by `scale`
    in-place. Optionally rescales P (Kalman covariance) by scale²
    when the caller config requests it (Option A); otherwise P is
    untouched (Option B per AEC3).
    """
    self.W *= np.complex64(scale)

class PBFDKF(PBFDAF):
    def scale_filter(self, scale: float, scale_p: bool = False) -> None:
        super().scale_filter(scale)
        if scale_p:
            self.P *= float(scale) ** 2
```

Decision: keep `scale_p` defaulting to False (AEC3-aligned). Option A
(`scale_p=True`) is what Kalman canonical math would predict (variance
of a scaled state is scale² × original variance), but AEC3 deliberately
doesn't touch P because the Kalman gain `K = P·X*/(X·P·X* + R)` self-
corrects within a few frames anyway. AEC3's empirical choice; we follow.

Caller:

```python
if self._check_misadjustment_trigger():
    proposed_scale = 1.0 / self._misadjustment_smoothed
    scale = max(self.config.filter_misadjustment_scale_min,
                min(self.config.filter_misadjustment_scale_max,
                    proposed_scale))
    self.filter.scale_filter(scale,
        scale_p=self.config.filter_misadjustment_scale_p)
    # Reset smoothed estimate so we don't immediately re-fire.
    self._misadjustment_smoothed = 1.0
    self._misadjustment_hangover_remaining = (
        self.config.filter_misadjustment_hangover_frames)
    self._misadjustment_fire_count += 1
```

## 3. Design decisions

### 3.1 Misadjustment metric — use existing `_diag['filter_scale_ratio']` formula

The trace metric at [aec.py:8683-8689](../python/aec.py#L8683) is already
exactly the AEC3-form `echo_psd / (far_psd × ERL_est)`. The estimator
upgrade simply adds an asymmetric EMA + state. No new math to derive.

### 3.2 Asymmetric EMA tuning

AEC3 uses different time constants for accumulation vs recovery. Defaults
(`alpha_up=0.99`, `alpha_dn=0.95`) give:
- Time constant up (accumulation): ~100 frames = 1 sec
- Time constant down (recovery): ~20 frames = 0.2 sec

Drift accumulates over 1+ seconds before triggering; recovery (e.g.
after a real path change) flushes the smoothed value in 200 ms so the
estimator doesn't keep insisting "filter is under-modelling" after
EPC reset.

### 3.3 Scale-P design choice

**Option A** (`scale_p=True`, Kalman canonical): when W is scaled, the
Kalman state covariance P should also scale by scale². Mathematically
consistent.

**Option B** (`scale_p=False`, AEC3-aligned): leave P alone. K self-
corrects within ~5 frames because high P→large K→large weight update
→W converges to right magnitude through Kalman gain mechanics.

**Decision**: default Option B per AEC3. Document Option A as a
config-switch knob (`filter_misadjustment_scale_p: bool = False`)
for future ablation if B.4 trace shows pathological re-convergence
post-scale.

### 3.4 Conservative scale clamp [0.5, 2.0]

AEC3 doesn't clamp; it trusts the proposed scale. We're more cautious
because of v3.14 Arc H (Huber loss) experience showing that filter-state
mutations can cascade if too aggressive. A 2× over-correction in one
frame is roughly the AEC3 expected magnitude per cycle anyway. Clamp
documented but reviewable in B.4.

### 3.5 Per-bin vs scalar — scalar only for B

AEC3's ScaleFilter is scalar (single multiplier across all bins, all
partitions). Per-band misadjustment correction is a natural extension
but out-of-scope for B (would require per-band ERL_est + per-band
threshold tuning). Scalar version validates the mechanism first; per-
band variant becomes a v3.19+ extension if B ships.

### 3.6 Interaction with shadow filter

Phase B touches **main filter only**. Shadow filter is unaffected
(shadow tracks its own residual independently; misadjustment on shadow
would need a separate estimator anyway, and shadow is auxiliary).

File-disjointness check vs hypothetical concurrent Phase A: A touched
shadow construction + isinstance guards; B touches `PBFDAF.scale_filter`
(new method) + new AEC fields + new AEC method `_update_misadjustment_*`.
**No overlap.** Confirms §0.2 file-disjointness requirement.

### 3.7 Re-enable preconditions from A.7 closeout

A's R-register predicted "AEC3-port arcs hit pipeline co-tuning wall."
B is not strictly a port (it adds new mechanism), but the same wall
could apply. Mitigation:
- B.4 fire-rate audit BEFORE B.5 60-case: confirm trigger fires
  reasonably often (say 0.1-1.0 per case under BALANCED). If trigger
  never fires → mechanism is silent → close cheap.
- B.5 60-case includes worst-case bar (no sample Δecho < -0.05) — same
  bar that killed Phase A. If worst-case breaches without
  compensating cohort-tail gain → close per §0.4.

## 4. Reverse-evidence risk register

| # | Risk | Mitigation |
|---|---|---|
| R1 | False-positive scale corrupts W during DT-partial | Trigger gate excludes `_dt_from_energy > 0` (handled by `epc_active`/`refined_usable` upstream) |
| R2 | Filter just learned a real new path; estimator triggers spurious scale-down | hangover_frames=100; estimator alpha_dn fast-decays smoothed value back toward 1.0 within 200 ms of a successful track |
| R3 | `_erl_estimate` itself is stale → scale_ratio garbage | Stability gate requires `_filter_converged AND refined_usable` — ERL has converged in that regime |
| R4 | Scale-P interaction with `_p_max_override` countdown (active during EPC reset) | `epc_active` excluded from trigger gate; P-override countdown never overlaps with B fire |
| R5 | Cohort tail (qNvSMyU) — does misadjustment correction help or hurt? | qNvSMyU is non-stationary path → estimator unlikely to enter stable regime → trigger silent on this case. Verify in B.4 trace |
| R6 | Cumulative interaction with Arc P (per-band ERL EMA) | Arc P feeds `_erl_estimate` upstream of B; same metric reused. Mutually consistent — if Arc P's ERL is wrong, B's ratio is also wrong, but B's threshold gate prevents action under that condition |

## 5. B.1 design confidence read

**Reading: 75%** (higher than A.1's 65%).

Components:
- (+) Mechanism is **additive**, not replacement — baseline preserved
  when flag-OFF byte-equal AND when flag-ON trigger doesn't fire
- (+) Underlying signal (`filter_scale_ratio`) already trace-validated
  in production code path — not a new measurement
- (+) AEC3 design choice is well-documented and self-contained
- (+) File-disjoint from Phase A/C/E (per §3.6)
- (−) Three AEC3-port arcs in a row failed; pattern says "be cautious"
  even though B isn't a strict port
- (−) Trigger fire-rate unknown until B.4 trace; could be silent on
  the whole 800-case (mechanism never activates → no gain)
- (−) Aggressive scale corrections could destabilise convergence on
  borderline-DT samples (R1/R2 mitigations may be insufficient)

Confidence is above the 60% kill threshold; proceed to B.2 with B.4
fire-rate trace as the first decision gate.

## 6. Updated Phase B hard bar

### B.4 fire-rate trace gate (must pass to authorise B.5 60-case bench)
- Trigger fires **≥ 10 times** across 60-case BALANCED rendering
  (anything less suggests mechanism is silent → close cheap)
- Average proposed scale **within [0.7, 1.5]** range (aggressive
  >2× scales suggest the threshold is wrong, not the filter)
- No frame where flag-ON main_err exceeds 5× flag-OFF main_err (sanity:
  scale didn't catastrophically destabilise the filter)

### B.5 60-case AECMOS (per A.7 lessons learned)
- PRIMARY: cohort tail (qNvSMyU) Δecho ≥ +0.020 dB
  (lower than A's +0.030 because B isn't targeting cohort tail; B targets
  long-term drift which is a different failure mode)
- Bucket guards: DT Δdeg ≥ -0.005, FS Δecho ≥ -0.010, NE Δdeg ≥ -0.005
- Worst-case bar: no sample Δecho < -0.05 dB
- Δdeg-compensated bar: worst-case Δecho between -0.05 and -0.02
  acceptable if same sample's Δdeg ≥ +0.02 (DT NE win offsets FS loss)

### B.6 800-case (productisation, requires post-B.5 user re-auth)
- Same bars as B.5 evaluated on full 800-case
- BALANCED preset only
- nores listen on 0I0XMl3M / qNvSMyU / xrtntuju 5-clip
- Trigger fire distribution: should align with B.4 trace distribution

### Kill criterion (per §0.4)
B.4 fire-rate gate fail → close, substrate retained.
B.5 PRIMARY OR worst-case fail → close, substrate retained.

## 7. B.1 → B.2 hand-off checklist

B.1 design complete when:
- [x] Audit refreshed (§1)
- [x] Construction + estimator + trigger + scale design (§2)
- [x] Decisions classified (§3)
- [x] Reverse-evidence risk register (§4)
- [x] Confidence read (§5)
- [x] Hard bar updated for B.4/B.5/B.6 (§6)

B.2 commit gate: per the **post-A.7 lessons learned**, this design
incorporates clearer kill criteria and a cheap B.4 fire-rate gate
before committing to the 60-case bench. **Authorisation scope: B.2
proceeds under user's "go without stopping" directive 2026-05-16.**

## 8. B.2-B.6 sprint preview

| Sprint | Action | Output |
|---|---|---|
| B.2 | Add `filter_misadjustment_*` config; AEC state init; `_update_misadjustment_estimator` method | 5-case byte-equal flag-OFF md5 PASS |
| B.3 | `PBFDAF.scale_filter` + `PBFDKF.scale_filter` (scale_p option) methods; trigger logic + scale fire wiring | 5-case byte-equal flag-OFF |
| B.4 | Render 60-case flag-ON, capture `_misadjustment_fire_count` distribution per case | fire-rate gate decision |
| B.5 | If B.4 passes: 60-case AECMOS A/B + nores listen on cohort tail | verdict doc |
| B.6 | If B.5 passes: 800-case + ship gate (user re-auth) | B.6 verdict + §0.7 merge package |

## 9. Cross-references

- [docs/v3_18_a_closeout.md](v3_18_a_closeout.md) — Phase A closeout (pivot motivation)
- [docs/v3_18_plan_revision_2026_05_15.md §11](v3_18_plan_revision_2026_05_15.md#L244) — §11 fallback to Phase B
- [docs/aec3_reference.md §6.2](aec3_reference.md#L811) — AEC3 FilterMisadjustmentEstimator + ScaleFilter
- [aec.py:8683-8689](../python/aec.py#L8683) — existing `filter_scale_ratio` trace substrate
- [aec.py:1916-2027](../python/aec.py#L1916) — PBFDKF method group (insertion point for `scale_filter`)
