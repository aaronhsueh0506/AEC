"""ErleEstimator — wraps Fullband + Subband ERLE (single-channel).

Mirrors docs/aec3_extracts/src/aec3/erle_estimator.{cc,h}.

Startup-phase gating: ``startup_phase_length_blocks`` is supplied at
construction time in OUR hop units (caller computes via _constants).
AEC3 passes ``2 * kNumBlocksPerSecond`` (~2 s) -> our 200 hops.
"""
from typing import Optional

import numpy as np

from .fullband_erle import FullBandErleEstimator
from .subband_erle import SubbandErleEstimator


class ErleEstimator:
    def __init__(
        self,
        *,
        startup_phase_length_hops: int = 200,
        n_bins: int = 257,
        min_erle: float = 1.0,
        max_erle_l: float = 4.0,
        max_erle_h: float = 1.5,
        use_onset_detection: bool = True,
        subband_wallclock_smoothing: bool = False,
        fullband_wallclock_smoothing: bool = False,
        startup_follows_convergence: bool = False,
        hop_size: int = 160,
        sample_rate: int = 16000,
        e2y2_gate_enabled: bool = False,
        e2y2_gate_threshold: float = 0.5,
    ) -> None:
        self._startup_hops = int(startup_phase_length_hops)
        self._startup_follows_convergence = bool(startup_follows_convergence)
        self._fullband = FullBandErleEstimator(
            min_erle=min_erle, max_erle_l=max_erle_l,
            wallclock_smoothing=fullband_wallclock_smoothing,
            hop_size=hop_size, sample_rate=sample_rate,
        )
        self._subband = SubbandErleEstimator(
            n_bins=n_bins,
            min_erle=min_erle,
            max_erle_l=max_erle_l,
            max_erle_h=max_erle_h,
            use_onset_detection=use_onset_detection,
            wallclock_smoothing=subband_wallclock_smoothing,
            hop_size=hop_size,
            sample_rate=sample_rate,
            e2y2_gate_enabled=e2y2_gate_enabled,
            e2y2_gate_threshold=e2y2_gate_threshold,
        )
        self._blocks_since_reset = 0

    def reset(self, delay_change: bool) -> None:
        self._fullband.reset()
        self._subband.reset()
        if delay_change:
            self._blocks_since_reset = 0

    def update(
        self,
        *,
        x2: np.ndarray,                # render PSD with reverb (per-bin)
        y2: np.ndarray,                # capture PSD (per-bin)
        e2: np.ndarray,                # subtractor / filter error PSD (per-bin)
        converged_filter: bool,
        # Legacy kwargs (signal-dependent ERLE retired); accepted as no-op for
        # caller-side API compatibility.
        filter_freq_response: Optional[np.ndarray] = None,
        x2_history: Optional[np.ndarray] = None,
        # C': pre-computed coherence gate (bool array). None = disabled.
        coh_gate_mask: Optional[np.ndarray] = None,
    ) -> None:
        self._blocks_since_reset += 1
        # W1' (startup_follows_convergence): when ON, skip the fixed session
        # startup gate and update as soon as the filter is converged — the
        # sub-estimators already early-return when converged_filter=False, so
        # ERLE-readiness tracks the same convergence signal usable_linear uses
        # instead of an independent 200-hop clock that desyncs from it.
        if (not self._startup_follows_convergence
                and self._blocks_since_reset < self._startup_hops):
            return
        self._subband.update(x2=x2, y2=y2, e2=e2, converged_filter=converged_filter,
                             coh_gate_mask=coh_gate_mask)
        self._fullband.update(x2=x2, y2=y2, e2=e2, converged_filter=converged_filter)

    def erle(self, onset_compensated: bool) -> np.ndarray:
        return self._subband.erle(onset_compensated)

    def erle_unbounded(self) -> np.ndarray:
        return self._subband.erle_unbounded()

    def fullband_erle_log2(self) -> float:
        return self._fullband.fullband_erle_log2()

    def get_inst_linear_quality_estimate(self) -> Optional[float]:
        return self._fullband.get_inst_linear_quality_estimate()
