"""ErleEstimator — wraps Fullband + Subband ERLE (single-channel).

Mirrors docs/aec3_extracts/src/aec3/erle_estimator.{cc,h}.

v3.21.17 (2026-05-23): SignalDependentErleEstimator port added behind
``num_sections`` parameter. When ``num_sections == 0`` (default OFF),
behaviour is byte-equal to v3.21.6 (SDE not instantiated). When
``num_sections == 1`` (degenerate), SDE runs but produces ERLE clamped
to per-subband max_erle (Gate 3 degeneracy proof — should ≈ baseline).
When ``num_sections >= 2``, SDE refines per-section per-subband.

Startup-phase gating: ``startup_phase_length_blocks`` is supplied at
construction time in OUR hop units (caller computes via _constants).
AEC3 passes ``2 * kNumBlocksPerSecond`` (~2 s) -> our 200 hops.
"""
from typing import Optional

import numpy as np

from .fullband_erle import FullBandErleEstimator
from .subband_erle import SubbandErleEstimator
from .signal_dependent_erle import SignalDependentErleEstimator


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
        # v3.21.17 SDE flag (default 0 = OFF; byte-equal preserved)
        signal_dependent_erle_sections: int = 0,
        sde_num_blocks: int = 13,
        sde_delay_headroom_blocks: int = 0,
    ) -> None:
        self._startup_hops = int(startup_phase_length_hops)
        self._fullband = FullBandErleEstimator(min_erle=min_erle, max_erle_l=max_erle_l)
        self._subband = SubbandErleEstimator(
            n_bins=n_bins,
            min_erle=min_erle,
            max_erle_l=max_erle_l,
            max_erle_h=max_erle_h,
            use_onset_detection=use_onset_detection,
        )
        self._blocks_since_reset = 0
        # v3.21.17 SDE — only instantiate when sections > 0 (preserves byte-equal)
        self._sde: Optional[SignalDependentErleEstimator] = None
        if signal_dependent_erle_sections > 0:
            self._sde = SignalDependentErleEstimator(
                num_sections=signal_dependent_erle_sections,
                num_blocks=sde_num_blocks,
                delay_headroom_blocks=sde_delay_headroom_blocks,
                n_freqs=n_bins,
                min_erle=min_erle,
                max_erle_l=max_erle_l,
                max_erle_h=max_erle_h,
                use_onset_detection=use_onset_detection,
            )

    def reset(self, delay_change: bool) -> None:
        self._fullband.reset()
        self._subband.reset()
        if self._sde is not None:
            self._sde.reset()
        if delay_change:
            self._blocks_since_reset = 0

    def update(
        self,
        *,
        x2: np.ndarray,                # render PSD with reverb (per-bin)
        y2: np.ndarray,                # capture PSD (per-bin)
        e2: np.ndarray,                # subtractor / filter error PSD (per-bin)
        converged_filter: bool,
        # v3.21.17 SDE inputs (optional; None when SDE disabled)
        filter_freq_response: Optional[np.ndarray] = None,  # [num_blocks, n_freqs]
        x2_history: Optional[np.ndarray] = None,            # [num_blocks, n_freqs]
    ) -> None:
        self._blocks_since_reset += 1
        if self._blocks_since_reset < self._startup_hops:
            return
        self._subband.update(x2=x2, y2=y2, e2=e2, converged_filter=converged_filter)
        self._fullband.update(x2=x2, y2=y2, e2=e2, converged_filter=converged_filter)
        # SDE consumes the just-updated subband ERLE as average_erle
        if self._sde is not None and filter_freq_response is not None and x2_history is not None:
            avg_erle = self._subband.erle(onset_compensated=False)
            avg_erle_oc = self._subband.erle(onset_compensated=True)
            self._sde.update(
                x2=x2, y2=y2, e2=e2,
                average_erle=avg_erle,
                average_erle_onset_compensated=avg_erle_oc,
                filter_freq_response=filter_freq_response,
                x2_history=x2_history,
                converged_filter=converged_filter,
            )

    def erle(self, onset_compensated: bool) -> np.ndarray:
        # v3.21.17: when SDE active, return refined ERLE; else baseline subband
        if self._sde is not None:
            return self._sde.erle(onset_compensated)
        return self._subband.erle(onset_compensated)

    def erle_unbounded(self) -> np.ndarray:
        return self._subband.erle_unbounded()

    def fullband_erle_log2(self) -> float:
        return self._fullband.fullband_erle_log2()

    def get_inst_linear_quality_estimate(self) -> Optional[float]:
        return self._fullband.get_inst_linear_quality_estimate()
