# v3.18 Phase C.B.1 — FilteringQualityAnalyzer sub-design (2026-05-16)

**Status**: sub-design under [docs/v3_18_c1_aec_state_design.md](v3_18_c1_aec_state_design.md) §2 C.B.
Audit-only port; consumer remains `_filter_converged` until C.E migration.

## 1. Public API

New file `python/aec_filter_quality.py`:

```python
class FilteringQualityAnalyzer:
    """AEC3 FilteringQualityAnalyzer: multi-gate usable_linear_estimate.

    Combines 4 gates (AND-semantic):
      1. startup_timer  (frames since first far_active ≥ STARTUP_FRAMES)
      2. reset_timer    (frames since last EPC reset ≥ RESET_FRAMES)
      3. convergence_seen (sticky: has _filter_converged been True yet?)
      4. not_transparent (we have no transparent-mode equivalent; proxy
         = far_active sometime in last TRANSPARENT_WINDOW frames)

    Outputs:
      - usable (bool): combined multi-gate flag
      - startup_done (bool)
      - reset_done (bool)
      - convergence_seen (bool)
      - far_active_recent (bool)  [transparent proxy]
    """

    # Frame counts at 16k / 10ms hop = 0.01 s per frame
    STARTUP_FRAMES = 40    # 0.4 s
    RESET_FRAMES = 20      # 0.2 s
    TRANSPARENT_WINDOW = 100   # 1.0 s

    def __init__(self):
        self.startup_timer = 0
        self.reset_timer = 0
        self.frames_since_far_active = self.TRANSPARENT_WINDOW + 1
        self.convergence_seen = False
        # Outputs:
        self.usable = False
        self.startup_done = False
        self.reset_done = False
        self.far_active_recent = False

    def update(self, *, far_active: bool, epc_reset_fired: bool,
               filter_converged: bool) -> None:
        # 1. startup_timer increments once far has ever been active.
        if far_active or self.startup_timer > 0:
            self.startup_timer += 1
        # 2. reset_timer increments since last EPC reset; restarted on fire.
        if epc_reset_fired:
            self.reset_timer = 0
        else:
            self.reset_timer += 1
        # 3. convergence_seen: sticky True after first True observation.
        if filter_converged:
            self.convergence_seen = True
        # 4. transparent proxy: far_active observed within window.
        if far_active:
            self.frames_since_far_active = 0
        else:
            self.frames_since_far_active += 1

        self.startup_done = self.startup_timer >= self.STARTUP_FRAMES
        self.reset_done = self.reset_timer >= self.RESET_FRAMES
        self.far_active_recent = (
            self.frames_since_far_active < self.TRANSPARENT_WINDOW)
        self.usable = (
            self.startup_done and self.reset_done
            and self.convergence_seen and self.far_active_recent)

    def reset(self) -> None:
        self.startup_timer = 0
        self.reset_timer = 0
        self.frames_since_far_active = self.TRANSPARENT_WINDOW + 1
        self.convergence_seen = False
        self.usable = False
        self.startup_done = False
        self.reset_done = False
        self.far_active_recent = False
```

## 2. Wiring into `AEC`

Inputs FQA needs per frame:
- `far_active`: from existing `far_active` boolean computed in process()
- `epc_reset_fired`: True when any of `delay_shift / epv_event / shadow_rise` fired this frame
- `filter_converged`: from `self._filter_converged`

Lazy-init guarded by `config.filter_quality_enabled` (new flag).
Update call placed right after C.A's FilterAnalyzer update.

### Config

```python
filter_quality_enabled: bool = False    # C.B audit-only port
```

### Frame update

```python
if self._filter_quality is not None:
    # Reuse existing per-frame booleans (no new computation).
    _epc_reset_this_frame = bool(
        getattr(self, '_epc_reset_fired_this_frame', False))
    self._filter_quality.update(
        far_active=bool(far_active),
        epc_reset_fired=_epc_reset_this_frame,
        filter_converged=bool(self._filter_converged))
    self._diag['fq_usable'] = bool(self._filter_quality.usable)
    self._diag['fq_startup_done'] = bool(self._filter_quality.startup_done)
    self._diag['fq_reset_done'] = bool(self._filter_quality.reset_done)
    self._diag['fq_convergence_seen'] = bool(self._filter_quality.convergence_seen)
```

`_epc_reset_fired_this_frame` is a new per-frame helper bool we set
inside each of the three EPC fire sites (delay_shift, EPV, shadow_rise).
Setting a bool inside an already-existing branch is byte-equal-safe.

## 3. Pre-bench gate (C.B.3)

Per C.1 §6:
> `usable_linear_estimate (multi-gate)` coverage exceeds `_filter_converged`
> coverage by ≥ 20 pp on 5-case trace

Same 5 stems as C.A.3. PASS if coverage delta ≥ 20pp on at least 2 of 5
cases.

Direction matters: `usable` should be MORE selective than
`consistent_estimate` (multi-gate is stricter than single peak-stability)
but MORE permissive than `_filter_converged` (because it doesn't require
the strict per-frame convergence threshold).

Expected ordering across coverage %:
```
_filter_converged ≤ fq_usable ≤ consistent_estimate
```

If `fq_usable` < `_filter_converged`, the multi-gate is broken (too
strict). If `fq_usable` ≈ `consistent_estimate`, the multi-gate adds
no constraint over peak stability (gates are silent).

## 4. C.B sprint sequence

| Sprint | Action | Output |
|---|---|---|
| C.B.1 | This design doc | doc only |
| C.B.2 | `aec_filter_quality.py` + config flag + wiring + `_epc_reset_fired_this_frame` helper + byte-equal flag-OFF | byte-equal flag-OFF 5/5 PASS |
| C.B.3 | 5-case pre-bench gate trace | PASS/FAIL decision |
| C.B.4 | 60-case bucket-aggregate trace | distribution doc |
| C.B.5 | C.B verdict + C.C hand-off | C.B verdict doc |

## 5. Cross-references

- [docs/v3_18_c1_aec_state_design.md](v3_18_c1_aec_state_design.md) §2.C.B
- [docs/v3_18_c_a_verdict.md](v3_18_c_a_verdict.md) — C.A verdict (predecessor)
- [docs/aec3_extracts/src/aec3/aec_state.h](aec3_extracts/src/aec3/aec_state.h) — `UsableLinearEstimate` reference
