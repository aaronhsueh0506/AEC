# v3.18 Phase C.C.1 — AecState ADT sub-design (2026-05-16)

**Status**: sub-design under [docs/v3_18_c1_aec_state_design.md](v3_18_c1_aec_state_design.md) §2 C.C.
Read-only facade: no behaviour change; consumer migration deferred to C.E.

## 1. Scope and non-scope

**Scope** (this sub-phase):
- New `AecState` class wrapping existing AEC fields + C.A/C.B analyzers
- AEC3-aligned public method names (`UsableLinearEstimate`,
  `ActiveRender`, `SaturatedCapture`, `TransparentModeActive`, etc.)
- Lazy-init in `AEC.__init__`; per-frame `aec_state.update()` is a
  no-op-on-flag-OFF; flag-ON populates internal cache

**Non-scope** (deferred to C.E or beyond):
- Consumer migration (~120 sites still read `_filter_converged` directly)
- Any AecState method actually changing AEC behaviour
- ErleEstimator / ErlEstimator / SaturationDetector subsumption (C.7 in
  original plan — deferred to v3.19+)

## 2. Public API

New file `python/aec_state.py`:

```python
class AecState:
    """AEC3-aligned read-only ADT facade over AEC pipeline state.

    Wraps:
      - C.A FilterAnalyzer (consistent_estimate)
      - C.B FilteringQualityAnalyzer (multi-gate usable)
      - Legacy fields (_filter_converged, _prev_filter_state, epc_active,
        _saturation_level, _erl_estimate, far_activity)

    Initial C.C wave: read-only facade only. Returns AEC3-shaped query
    interface but doesn't mutate. Consumers migrate to this facade in C.E.
    """

    def __init__(self, aec):
        self._aec = aec   # back-reference; read-only access

    # ── AEC3-aligned public API ──────────────────────────────────────

    def usable_linear_estimate(self) -> bool:
        """Can the linear filter output be trusted for downstream use?"""
        fq = self._aec._filter_quality
        if fq is not None:
            return fq.usable
        return bool(self._aec._filter_converged)   # legacy fallback

    def use_linear_filter_output(self) -> bool:
        """AEC3 equivalent — for now equal to usable_linear_estimate."""
        return self.usable_linear_estimate()

    def active_render(self) -> bool:
        """Is the far-end signal currently active?"""
        fq = self._aec._filter_quality
        if fq is not None:
            return fq.far_active_recent
        return bool(getattr(self._aec, 'far_activity', 0.0) > 0.1)

    def transparent_mode_active(self) -> bool:
        """Pass mic through unchanged (no echo to cancel)?"""
        fq = self._aec._filter_quality
        if fq is not None:
            # AEC3 proxy: no far activity recently → no echo to cancel
            return not fq.far_active_recent
        return False

    def saturated_capture(self) -> bool:
        return float(self._aec._saturation_level) > 0.3

    def epc_active(self) -> bool:
        return bool(self._aec.epc_active)

    # ── State-mirror accessors (transition aid for C.E) ──────────────

    def filter_converged(self) -> bool:
        return bool(self._aec._filter_converged)

    def filter_state(self) -> str:
        return str(getattr(self._aec, '_prev_filter_state', 'idle'))

    def consistent_estimate(self) -> bool:
        fa = self._aec._filter_analyzer
        return bool(fa.consistent_estimate) if fa is not None else False

    def erl_estimate(self) -> float:
        return float(self._aec._erl_estimate)

    def saturation_level(self) -> float:
        return float(self._aec._saturation_level)

    # ── Per-frame snapshot (for diag tracing) ─────────────────────────

    def snapshot(self) -> dict:
        return {
            'usable_linear': self.usable_linear_estimate(),
            'active_render': self.active_render(),
            'transparent_mode': self.transparent_mode_active(),
            'saturated_capture': self.saturated_capture(),
            'epc_active': self.epc_active(),
            'filter_converged': self.filter_converged(),
            'filter_state': self.filter_state(),
            'consistent_estimate': self.consistent_estimate(),
            'erl_estimate': self.erl_estimate(),
            'saturation_level': self.saturation_level(),
        }
```

## 3. Wiring

```python
# AecConfig
aec_state_enabled: bool = False    # C.C audit-only ADT facade

# AEC.__init__ (lazy-init)
self._aec_state = None
if self.config.aec_state_enabled:
    from aec_state import AecState
    self._aec_state = AecState(self)

# AEC.process(), at the C.B FQA update block (after C.A + C.B both run):
if self._aec_state is not None:
    self._diag['aec_state_snapshot'] = self._aec_state.snapshot()
```

`AecState` doesn't have its own state — every method reads from AEC
fields live. So `reset()` is a no-op.

## 4. Verification gate (C.C.2 — replaces pre-bench)

Per C.1 §6 verification gate (NOT pre-bench): "AecState public-method
outputs match legacy scattered fields on byte-equal trace."

Test method: 5-case run with C.A + C.B + C.C all ON. For each frame:
- `aec_state.filter_converged() == aec._filter_converged`
- `aec_state.consistent_estimate() == aec._filter_analyzer.consistent_estimate`
- `aec_state.usable_linear_estimate() == aec._filter_quality.usable`
- `aec_state.epc_active() == aec.epc_active`
- `aec_state.saturated_capture() == (aec._saturation_level > 0.3)`

PASS if all 5 mirror methods match all 5 legacy/analyzer fields on all
frames of all 5 cases. FAIL if any mismatch (catches wiring bugs).

Also: byte-equal flag-OFF md5 = baseline (audit-only invariant).

## 5. C.C sprint sequence

| Sprint | Action | Output |
|---|---|---|
| C.C.1 | This design doc | doc only |
| C.C.2 | `aec_state.py` + AecConfig flag + AEC wiring + 5-case byte-equal flag-OFF + verification trace | byte-equal flag-OFF md5 PASS + mirror verification |
| C.C.3 | C.C verdict + C.D/C.E hand-off | C.C verdict doc |

(No C.C.4 60-case — C.C is read-only facade; behaviour can't change.)

## 6. Cross-references

- [docs/v3_18_c1_aec_state_design.md](v3_18_c1_aec_state_design.md) — parent design
- [docs/v3_18_c_a_verdict.md](v3_18_c_a_verdict.md) — C.A verdict (analyzer source)
- [docs/v3_18_c_b_verdict.md](v3_18_c_b_verdict.md) — C.B verdict (FQA source)
- [docs/aec3_extracts/src/aec3/aec_state.h](aec3_extracts/src/aec3/aec_state.h) — reference API
