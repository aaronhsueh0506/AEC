"""AecState — WebRTC-AEC3-aligned read-only seam over the 5 detectors.

Extracted from ``aec.py`` during refactor R.8. Hosts the v3.18 Phase
C.C ADT facade (fq_usable, post_reset_age_frames, etc.) consumed by
ResidualEchoEstimator + PathChangeRegimeHandler + Phase-3 retry logic.

Back-ref pattern: ``AEC.__init__`` injects ``self._aec_state._aec_ref =
self`` after construction, so AecState methods can read live config
without import-time circularity.

Self-contained: depends only on numpy.
"""
import numpy as np


class AecState:
    """WebRTC AEC3-style read-only aggregator over the per-frame detector outputs.

    Holds references to the 5 detectors (RenderActivityDetector, DoubleTalkAnalyzer,
    FilterConvergenceAnalyzer, EchoPathChangeDetector, PathChangeRegimeHandler) and
    DtdEstimator-coherence; exposes derived flags via @property so consumers don't
    rebuild aggregation logic in multiple places.

    Phase B consumes this as the gate for two-path residual echo attribution:
    `usable_linear_estimate` decides linear vs nonlinear R2 model.

    All properties are read-only and reflect the current frame's detector state
    (no explicit per-frame update needed — properties delegate live).
    """

    def __init__(self, *, render_activity, convergence, dt_analyzer, epc_det,
                 regime_handler, dtd_coherence_getter):
        self._render = render_activity
        self._conv = convergence
        self._dt = dt_analyzer
        self._epc = epc_det
        self._regime = regime_handler
        self._dtd_coh = dtd_coherence_getter
        # v3.18 Phase C.C — optional back-ref to AEC for C.A/C.B access.
        # Set by AEC.__init__ when aec_state_enabled=True. When None,
        # AEC3-aligned methods below fall back to legacy semantics.
        self._aec_ref = None

    # ── Render activity ──────────────────────────────────────────────────────
    @property
    def render_active(self) -> bool: return self._render.is_active
    @property
    def render_stationary(self) -> bool: return self._render.is_stationary

    # ── Filter convergence ───────────────────────────────────────────────────
    @property
    def filter_converged(self) -> bool: return self._conv.converged
    @property
    def filter_once_converged(self) -> bool: return self._conv.once_converged
    @property
    def divergence(self) -> float: return self._conv.divergence

    # ── Double-talk signals ──────────────────────────────────────────────────
    @property
    def dt_from_energy(self) -> float: return self._dt.dt_from_energy
    @property
    def dt_from_shadow(self) -> float: return self._dt.dt_from_shadow
    @property
    def dt_from_coherence(self) -> float: return self._dtd_coh()
    @property
    def dt_combined(self) -> float:
        """Max of the three DT signals — same aggregation as inline code today."""
        return max(self._dt.dt_from_energy, self._dt.dt_from_shadow, self._dtd_coh())

    # ── Echo path change ─────────────────────────────────────────────────────
    @property
    def epc_active(self) -> bool: return self._epc.active
    @property
    def epc_hangover_count(self) -> int: return self._epc.hangover_count

    # ── Shadow filter management ─────────────────────────────────────────────
    @property
    def main_paused(self) -> bool: return self._regime.main_paused
    @property
    def shadow_advantage(self) -> float: return self._dt.shadow_advantage

    # ── v3.18 Phase C.C — AEC3-aligned extensions ───────────────────────
    # These read from C.A/C.B substrate when AEC back-ref is wired and
    # the flags are enabled; else fall back to legacy semantics.

    def consistent_estimate(self) -> bool:
        """C.A FilterAnalyzer: filter impulse-response shape stable?

        Returns False when C.A is unavailable (legacy AEC).
        """
        a = self._aec_ref
        if a is None or getattr(a, '_filter_analyzer', None) is None:
            return False
        return bool(a._filter_analyzer.consistent_estimate)

    def fq_usable(self) -> bool:
        """C.B multi-gate usable. Falls back to legacy converged when C.B off."""
        a = self._aec_ref
        if a is None or getattr(a, '_filter_quality', None) is None:
            return bool(self._conv.converged)
        return bool(a._filter_quality.usable)

    def active_render(self) -> bool:
        """AEC3-aligned: far signal currently active.

        When C.B is on, uses far_active_recent (windowed); else live render flag.
        """
        a = self._aec_ref
        if a is not None and getattr(a, '_filter_quality', None) is not None:
            return bool(a._filter_quality.far_active_recent)
        return bool(self._render.is_active)

    def transparent_mode_active(self) -> bool:
        """AEC3-aligned: no echo to cancel right now (pass mic through)."""
        a = self._aec_ref
        if a is not None and getattr(a, '_filter_quality', None) is not None:
            return not bool(a._filter_quality.far_active_recent)
        return False

    def saturated_capture(self) -> bool:
        a = self._aec_ref
        if a is None:
            return False
        return float(a._saturation_level) > 0.3

    def erl_estimate(self) -> float:
        a = self._aec_ref
        return float(a._erl_estimate) if a is not None else 0.0

    def aec3_snapshot(self) -> dict:
        """Per-frame snapshot of AEC3-aligned methods (for trace + audit)."""
        return {
            'usable_linear': bool(self.usable_linear_estimate),
            'fq_usable': self.fq_usable(),
            'consistent_estimate': self.consistent_estimate(),
            'active_render': self.active_render(),
            'transparent_mode': self.transparent_mode_active(),
            'saturated_capture': self.saturated_capture(),
            'epc_active': bool(self.epc_active),
            'filter_converged': bool(self.filter_converged),
            'erl_estimate': self.erl_estimate(),
        }

    # ── Aggregated decisions (Phase B consumers) ─────────────────────────────
    @property
    def usable_linear_estimate(self) -> bool:
        """Can the main filter's echo estimate be trusted as residual-echo source?

        Legacy semantics (preserved for byte-equal): once-converged +
        currently-converged + not in EPC recovery. v3.18 Phase C.E migrates
        consumers to `fq_usable()` instead (multi-gate AEC3 semantic) once
        the flag promotion path is approved.
        """
        return (self._conv.once_converged
                and self._conv.converged
                and not self._epc.active)
