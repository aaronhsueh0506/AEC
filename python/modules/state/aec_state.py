"""AecState — top-level AEC3 state machine.

Mirrors docs/aec3_extracts/src/aec3/aec_state.{cc,h}.

The state machine wraps 5 small helpers (InitialState, FilterDelay,
FilteringQualityAnalyzer, SaturationDetector, TransparentMode) plus
the heavyweight ERLE / ERL / Reverb estimators. Per-frame
``update()`` runs the helpers in STRICT order matching
aec_state.cc:189-291 (order matters because FilteringQualityAnalyzer
reads convergence flags computed earlier in the same call).

Phase 3.1 status: helpers in this file are live; ERLE / ERL /
TransparentMode / ReverbModelEstimator are STUBBED (return
sensible defaults). Subsequent Phase 3.2-3.4 commits fill them in.

``handle_echo_path_change(EchoPathVariability)`` is called BEFORE
``update()`` whenever the delay subsystem (Phase 1) reports a
delay change. Routes through to AEC3-defined full-reset on
``kBufferFlush`` / ``kNewDetectedDelay``, ERLE-only reset on
``gain_change``.
"""
from dataclasses import dataclass
from typing import Optional

import numpy as np

from ._constants import HOPS_PER_SECOND
from .erl_estimator import ErlEstimator
from .erle_estimator import ErleEstimator
from .filter_analyzer import FilterAnalyzer
from .filter_delay import FilterDelay
from .filter_quality import FilteringQualityAnalyzer
from .initial_state import InitialState
from .saturation_detector import SaturationDetector
from ..delay.delay_types import (
    DelayAdjustment,
    DelayEstimate,
    EchoPathVariability,
)
from ..filter.filter_state_bridge import FilterStateBridge


_CONSERVATIVE_INITIAL_HOPS = int(1.5 * HOPS_PER_SECOND)  # AEC3 1.5*250 -> 150 hops
_FAST_INITIAL_HOPS = int(0.8 * HOPS_PER_SECOND)           # AEC3 0.8*250 -> 80 hops
_ACTIVE_RENDER_BLOCKS = 80   # AEC3 200 blocks × 4 ms = 800 ms wall-clock
                             # = blocks_to_hops(200, 160, 16000) = 80 hops.
                             # Legacy 200-hops-verbatim was 2 s (2.5× too slow).


@dataclass
class AecStateConfig:
    """Knobs sourced from AEC3 EchoCanceller3Config slices we actually use."""

    use_linear_filter: bool = True
    conservative_initial_phase: bool = False
    initial_state_seconds: float = 2.5
    delay_headroom_samples: int = 32
    num_capture_channels: int = 1
    echo_can_saturate: bool = True
    n_bins: int = 257
    erle_startup_hops: int = 200    # AEC3 2*kNumBlocksPerSecond -> our 2*100 hops
    erle_min: float = 1.0           # AEC3 default
    erle_max_l: float = 4.0
    erle_max_h: float = 1.5
    erl_startup_hops: int = 200
    # hop_size/sample_rate feed the per_block→per_hop conversion.
    hop_size: int = 160
    sample_rate: int = 16000
    # TransparentMode is permanently disabled in production (legacy
    # 10-frame ERLE latch was retired); kwarg preserved as no-op for
    # AEC3-spec API compatibility.
    enable_transparent_mode: bool = False
    transparent_linear_and_stable: bool = False
    # AEC3 FilterAnalyzer port is shipped on; legacy off-path retired.
    enable_filter_analyzer: bool = True


class AecState:
    """Per-frame AEC3 state machine. Single source of truth for
    ``usable_linear_estimate()``."""

    def __init__(self, config: Optional[AecStateConfig] = None) -> None:
        self._config = config or AecStateConfig()
        self._initial_state = InitialState(
            conservative_initial_phase=self._config.conservative_initial_phase,
            initial_state_seconds=self._config.initial_state_seconds,
        )
        self._delay_state = FilterDelay(
            delay_headroom_samples=self._config.delay_headroom_samples,
            num_capture_channels=self._config.num_capture_channels,
        )
        self._filter_quality = FilteringQualityAnalyzer(
            use_linear_filter=self._config.use_linear_filter,
        )
        self._saturation_detector = SaturationDetector()
        self._erle_estimator = ErleEstimator(
            startup_phase_length_hops=self._config.erle_startup_hops,
            n_bins=self._config.n_bins,
            min_erle=self._config.erle_min,
            max_erle_l=self._config.erle_max_l,
            max_erle_h=self._config.erle_max_h,
            hop_size=self._config.hop_size,
        )
        self._erl_estimator = ErlEstimator(
            startup_phase_length_hops=self._config.erl_startup_hops,
            n_bins=self._config.n_bins,
            hop_size=self._config.hop_size,
        )
        # TransparentMode permanently disabled in production.
        self._transparent_mode = None
        self._filter_analyzer: Optional[FilterAnalyzer] = (
            FilterAnalyzer() if self._config.enable_filter_analyzer else None
        )
        # Counters tracked at this level (not in helpers).
        self._strong_not_saturated_render_blocks = 0
        self._blocks_with_active_render = 0
        self._capture_signal_saturation = False

    # ---------------------------------------------------------- public queries

    def usable_linear_estimate(self) -> bool:
        return (
            self._filter_quality.linear_filter_usable()
            and self._config.use_linear_filter
        )

    def active_render(self) -> bool:
        return self._blocks_with_active_render > _ACTIVE_RENDER_BLOCKS

    def saturated_echo(self) -> bool:
        return self._saturation_detector.saturated_echo()

    def saturated_capture(self) -> bool:
        return self._capture_signal_saturation

    def transparent_mode_active(self) -> bool:
        return self._transparent_mode is not None and self._transparent_mode.active()

    def transition_triggered(self) -> bool:
        return self._initial_state.transition_triggered()

    def initial_state_active(self) -> bool:
        return self._initial_state.initial_state_active()

    def min_direct_path_filter_delay(self) -> int:
        return self._delay_state.min_direct_path_filter_delay()

    def erle(self, onset_compensated: bool = False) -> np.ndarray:
        return self._erle_estimator.erle(onset_compensated)

    def erle_unbounded(self) -> np.ndarray:
        return self._erle_estimator.erle_unbounded()

    def fullband_erle_log2(self) -> float:
        return self._erle_estimator.fullband_erle_log2()

    def get_inst_linear_quality_estimate(self) -> Optional[float]:
        """AEC3-strict ``erle_estimator_.GetInstLinearQualityEstimates()``
        (fullband_erle_estimator.h:55). Continuous [0, 1] from the rolling
        ERLE max/min normalisation, with a ~400 ms hold counter after the
        last convergence-driven update. Returns None until the accumulator
        has accumulated 6 converged points OR after the hold expires.

        Consumed by ``ResidualEchoEstimator`` reverb update as the
        ``linear_filter_quality`` smoothing factor (cc:286-289). The hold
        counter keeps the reverb tail refresh window alive for ~400 ms
        beyond per-frame convergence — load-bearing on cases where the
        filter converges briefly then NE dominates, otherwise the tail
        gets frozen at the last update's value and inflates R²."""
        return self._erle_estimator.get_inst_linear_quality_estimate()

    def erl(self) -> np.ndarray:
        return self._erl_estimator.erl()

    def erl_time_domain(self) -> float:
        return self._erl_estimator.erl_time_domain()

    def external_delay_blocks(self) -> Optional[DelayEstimate]:
        return self._delay_state.external_delay_blocks()

    def filter_analyzer_consistent(self) -> bool:
        return (self._filter_analyzer is not None
                and self._filter_analyzer.any_filter_consistent())

    def filter_analyzer_peak_index(self) -> int:
        if self._filter_analyzer is None:
            return -1
        return self._filter_analyzer.peak_index()

    def filter_analyzer_max_echo_path_gain(self) -> float:
        if self._filter_analyzer is None:
            return 0.0
        return self._filter_analyzer.max_echo_path_gain()

    # ------------------------------------------------------------- mutators

    def update_capture_saturation(self, saturated: bool) -> None:
        self._capture_signal_saturation = bool(saturated)

    def handle_echo_path_change(self, variability: EchoPathVariability) -> None:
        if variability.delay_change is not DelayAdjustment.NONE:
            self._full_reset()
        elif variability.gain_change:
            # ERLE-only reset on gain change (mirrors aec_state.cc:165-167).
            self._erle_estimator.reset(delay_change=False)

    def update(
        self,
        *,
        bridge: FilterStateBridge,
        external_delay: Optional[DelayEstimate],
        render_psd: np.ndarray,
        capture_psd: np.ndarray,
        error_psd: np.ndarray,
        echo_psd: np.ndarray,
        active_render: bool,
        subtractor_s_refined_max_abs: float = 0.0,
        subtractor_s_coarse_max_abs: float = 0.0,
        echo_path_gain: float = 1.0,
        render_block: Optional[np.ndarray] = None,
        filter_taps_full: Optional[np.ndarray] = None,
        # SDE inputs (per-partition |W|² and X²). None when SDE OFF.
        sde_filter_freq_response: Optional[np.ndarray] = None,
        sde_x2_history: Optional[np.ndarray] = None,
        # AEC3-strict X²_reverb for ErleEstimator only (aec_state.cc:229-250).
        # When provided, the SubbandErleEstimator consumes X²+reverb_tail
        # instead of bare render_psd. ErlEstimator + SaturationDetector
        # keep using render_psd.
        x2_reverb_for_erle: Optional[np.ndarray] = None,
        # E1: windowed Y2 (capture PSD) for the ErleEstimator ONLY, so ERLE's
        # Y2/E2 share the sqrt-Hann coordinate (E2/error_psd is already windowed).
        # When None (default), ERLE uses the rectangular capture_psd. ErlEstimator
        # + saturation keep capture_psd.
        capture_psd_erle: Optional[np.ndarray] = None,
        # C': pre-computed per-bin coherence gate mask (bool array, len=n_bins).
        # True = ERLE update allowed; False = freeze (nearend detected via Γ²_ŶY).
        # None when gate is disabled (byte-equal path).
        erle_coh_gate_mask: Optional[np.ndarray] = None,
    ) -> None:
        """Per-frame state update. Strict order matches aec_state.cc:189-291.

        Inputs:
          - ``bridge``: FilterStateBridge snapshot from Phase 2.
          - ``external_delay``: from RenderDelayController (Phase 1); None
            if no estimate has crossed the COARSE threshold yet.
          - ``render_psd`` / ``capture_psd`` / ``error_psd`` / ``echo_psd``:
            single-channel kFftLengthBy2Plus1 spectra in linear power units.
            (Phase 3.2 ERLE / ERL will consume these.)
          - ``active_render``: precomputed (orchestrator-side) far-end
            active-frame flag.
          - subtractor max-abs values + echo_path_gain + render_block:
            inputs to SaturationDetector.
          - ``filter_taps_full``: full time-domain impulse response of the
            adaptive filter (concatenated partitions). Required when
            ``enable_filter_analyzer`` is True; ignored otherwise.
        """
        # 1. Filter quality + convergence pass (reads any_filter_converged
        #    from bridge; this is the AEC3 SubtractorOutputAnalyzer surface).
        any_filter_converged = bridge.filter_converged

        # 1b. FilterAnalyzer (aec_state.cc:199-200). Runs BEFORE FilterDelay
        # so the analyzer's per-channel delays are fresh on the same frame.
        analyzer_delays: Optional[list[int]] = None
        if (self._filter_analyzer is not None
                and filter_taps_full is not None
                and render_block is not None):
            self._filter_analyzer.update(filter_taps_full, render_block)
            analyzer_delays = self._filter_analyzer.filter_delays_blocks()

        # 2. FilterDelay update (aec_state.cc:203-206).
        self._delay_state.update(
            analyzer_filter_delay_estimates_blocks=analyzer_delays,
            external_delay=external_delay,
            blocks_with_proper_filter_adaptation=self._strong_not_saturated_render_blocks,
        )

        # 3. active_render + saturation block counters.
        if active_render:
            self._blocks_with_active_render += 1
        if active_render and not self._capture_signal_saturation:
            self._strong_not_saturated_render_blocks += 1

        # 4a. ERLE (transition-triggered soft reset BEFORE update — matches
        #     aec_state.cc:244-246; clears non-delay-change state).
        if self._initial_state.transition_triggered():
            self._erle_estimator.reset(delay_change=False)
        erle_converged = any_filter_converged
        self._erle_estimator.update(
            x2=x2_reverb_for_erle if x2_reverb_for_erle is not None else render_psd,
            y2=capture_psd_erle if capture_psd_erle is not None else capture_psd,
            e2=error_psd,
            converged_filter=erle_converged,
            filter_freq_response=sde_filter_freq_response,
            x2_history=sde_x2_history,
            coh_gate_mask=erle_coh_gate_mask,
        )
        # 4b. ERL update.
        self._erl_estimator.update(
            render_psd=render_psd,
            capture_psd=capture_psd,
            converged_filter=any_filter_converged,
        )
        # 4c. STUB: reverb / echo_audibility. Phase 3.4 fills.

        # 5. Saturation detector. AEC3 strict: echo_path_gain from FilterAnalyzer
        # (aec_state.cc:258-260) — when analyzer is present and ran this hop,
        # prefer its current MaxEchoPathGain() over the caller-supplied default.
        _epg = echo_path_gain
        if self._filter_analyzer is not None:
            _epg = float(self._filter_analyzer.max_echo_path_gain())
        if self._config.echo_can_saturate and render_block is not None:
            self._saturation_detector.update(
                render_block=render_block,
                saturated_capture=self._capture_signal_saturation,
                usable_linear_estimate=self.usable_linear_estimate(),
                subtractor_s_refined_max_abs=subtractor_s_refined_max_abs,
                subtractor_s_coarse_max_abs=subtractor_s_coarse_max_abs,
                echo_path_gain=_epg,
            )

        # 6. InitialState (uses post-saturation flag).
        self._initial_state.update(active_render, self._capture_signal_saturation)

        # 7. TransparentMode update (Legacy variant; HMM not ported).
        if self._transparent_mode is not None:
            # AEC3 derives any_filter_consistent from FilterAnalyzer
            # (aec_state.cc:271); fall back to the legacy proxy when the
            # analyzer is disabled.
            if self._filter_analyzer is not None:
                any_filter_consistent = self._filter_analyzer.any_filter_consistent()
            else:
                any_filter_consistent = (
                    any_filter_converged and external_delay is not None
                )
            # AEC3-strict SubtractorOutputAnalyzer divergence signal from
            # bridge. Falls back to the legacy heuristic when the orchestrator
            # has not populated it (back-compat for external callers that
            # build a minimal bridge).
            all_filters_diverged = bridge.all_filters_diverged or (
                (not any_filter_converged) and bridge.divergence_indicator > 1.0
            )
            self._transparent_mode.update(
                filter_delay_blocks=self.min_direct_path_filter_delay(),
                any_filter_consistent=any_filter_consistent,
                any_filter_converged=any_filter_converged,
                all_filters_diverged=all_filters_diverged,
                active_render=active_render,
                saturated_capture=self._capture_signal_saturation,
            )

        # 8. FilteringQualityAnalyzer (last; reads TM + convergence flags
        #    set above). filter_analyzer_consistent is consumed by the
        #    optional gate-3 AND knob.
        self._filter_quality.update(
            active_render=active_render,
            transparent_mode=self.transparent_mode_active(),
            saturated_capture=self._capture_signal_saturation,
            external_delay=external_delay,
            any_filter_converged=any_filter_converged,
            filter_analyzer_consistent=self.filter_analyzer_consistent(),
        )

    # -------------------------------------------------------------- helpers

    def _full_reset(self) -> None:
        """Handle a path-change event — reset everything except external
        delay (which is the trigger source itself). Mirrors
        ``aec_state.cc:145-157``."""
        self._strong_not_saturated_render_blocks = 0
        self._blocks_with_active_render = 0
        self._initial_state.reset()
        self._filter_quality.reset()
        self._capture_signal_saturation = False
        self._erle_estimator.reset(delay_change=True)
        self._erl_estimator.reset()
        if self._transparent_mode is not None:
            self._transparent_mode.reset()
        if self._filter_analyzer is not None:
            self._filter_analyzer.reset()
