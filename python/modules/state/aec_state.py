"""AecState: per-frame AEC3-style state machine; single source of truth for ``usable_linear_estimate()``.

It wraps InitialState, FilterDelay, FilteringQualityAnalyzer, SaturationDetector and
the ERLE / ERL estimators. ``update()`` runs them in a fixed order (the quality analyzer
reads flags computed earlier in the same call). Call ``handle_echo_path_change()``
before ``update()`` on a delay or gain change: a delay change resets everything, a gain
change resets ERLE only.
"""
from dataclasses import dataclass
from typing import Optional

import numpy as np

from .. import aec3_scale as _aec3_scale
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


# Marks AecStateConfig.erle_startup_hops / erl_startup_hops as unset; any explicit integer is honoured as given.
_AUTO_STARTUP_HOPS = None


@dataclass
class AecStateConfig:
    """AecState knobs; ``hop_size`` / ``sample_rate`` drive the per-block to per-hop conversion."""

    use_linear_filter: bool = True
    conservative_initial_phase: bool = False
    initial_state_seconds: float = 2.5
    delay_headroom_samples: int = 32
    num_capture_channels: int = 1
    echo_can_saturate: bool = True
    n_bins: int = 257
    # None: 2 s computed for the live grid; an integer overrides with a raw hop count.
    erle_startup_hops: Optional[int] = _AUTO_STARTUP_HOPS
    erle_min: float = 1.0
    erle_max_l: float = 4.0
    erle_max_h: float = 1.5
    erl_startup_hops: Optional[int] = _AUTO_STARTUP_HOPS
    hop_size: int = 160
    sample_rate: int = 16000
    # Transparent mode is permanently off; the fields are accepted and ignored.
    enable_transparent_mode: bool = False
    transparent_linear_and_stable: bool = False
    enable_filter_analyzer: bool = True


class AecState:
    """Per-frame AEC3-style state machine."""

    def _resolve_startup_hops(self, configured_hops: Optional[int]) -> int:
        """None -> 2 s at the live grid; otherwise the configured hop count."""
        if configured_hops is not None:
            return int(configured_hops)
        return _aec3_scale.ms_to_hops(
            2000.0, self._config.hop_size, self._config.sample_rate
        )

    def __init__(self, config: Optional[AecStateConfig] = None) -> None:
        self._config = config or AecStateConfig()
        self._initial_state = InitialState(
            conservative_initial_phase=self._config.conservative_initial_phase,
            initial_state_seconds=self._config.initial_state_seconds,
            hop_size=self._config.hop_size,
            sample_rate=self._config.sample_rate,
        )
        self._delay_state = FilterDelay(
            delay_headroom_samples=self._config.delay_headroom_samples,
            num_capture_channels=self._config.num_capture_channels,
            hop_size=self._config.hop_size,
            sample_rate=self._config.sample_rate,
        )
        self._filter_quality = FilteringQualityAnalyzer(
            use_linear_filter=self._config.use_linear_filter,
            hop_size=self._config.hop_size,
            sample_rate=self._config.sample_rate,
        )
        self._saturation_detector = SaturationDetector()
        erle_hops = self._resolve_startup_hops(self._config.erle_startup_hops)
        erl_hops = self._resolve_startup_hops(self._config.erl_startup_hops)
        self._erle_estimator = ErleEstimator(
            startup_phase_length_hops=erle_hops,
            n_bins=self._config.n_bins,
            min_erle=self._config.erle_min,
            max_erle_l=self._config.erle_max_l,
            max_erle_h=self._config.erle_max_h,
            hop_size=self._config.hop_size,
            sample_rate=self._config.sample_rate,
        )
        self._erl_estimator = ErlEstimator(
            startup_phase_length_hops=erl_hops,
            n_bins=self._config.n_bins,
            hop_size=self._config.hop_size,
            sample_rate=self._config.sample_rate,
        )
        self._transparent_mode = None
        self._filter_analyzer: Optional[FilterAnalyzer] = (
            FilterAnalyzer(
                hop_size=self._config.hop_size,
                sample_rate=self._config.sample_rate,
            ) if self._config.enable_filter_analyzer else None
        )
        self._strong_not_saturated_render_blocks = 0
        self._blocks_with_active_render = 0
        self._capture_signal_saturation = False
        # 800 ms of active render at the live grid.
        self._active_render_blocks = _aec3_scale.ms_to_hops(
            800.0, self._config.hop_size, self._config.sample_rate
        )


    def usable_linear_estimate(self) -> bool:
        return (
            self._filter_quality.linear_filter_usable()
            and self._config.use_linear_filter
        )

    def active_render(self) -> bool:
        return self._blocks_with_active_render > self._active_render_blocks

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
        """Fullband linear-filter quality in [0, 1] from the rolling ERLE max/min, held ~400 ms
        after the last converged update; None until 6 converged points accumulate or after the
        hold expires. Used by ``ResidualEchoEstimator`` as the reverb-update smoothing factor."""
        return self._erle_estimator.get_inst_linear_quality_estimate()

    def erl(self) -> np.ndarray:
        return self._erl_estimator.erl()

    def erl_time_domain(self) -> float:
        return self._erl_estimator.erl_time_domain()

    def external_delay_samples(self) -> Optional[DelayEstimate]:
        return self._delay_state.external_delay_samples()

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


    def update_capture_saturation(self, saturated: bool) -> None:
        self._capture_signal_saturation = bool(saturated)

    def handle_echo_path_change(self, variability: EchoPathVariability) -> None:
        if variability.delay_change is not DelayAdjustment.NONE:
            self._full_reset()
        elif variability.gain_change:
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
        sde_filter_freq_response: Optional[np.ndarray] = None,
        sde_x2_history: Optional[np.ndarray] = None,
        # X^2 plus reverb tail for the ERLE estimator only; other consumers use render_psd.
        x2_reverb_for_erle: Optional[np.ndarray] = None,
        # Windowed Y^2 for the ERLE estimator only (shares the sqrt-Hann coordinate with error_psd).
        capture_psd_erle: Optional[np.ndarray] = None,
        # Per-bin coherence gate (bool, len=n_bins): True allows the ERLE update, False freezes it. None disables.
        erle_coh_gate_mask: Optional[np.ndarray] = None,
    ) -> None:
        """Per-frame state update (fixed order).

        ``bridge`` is the FilterStateBridge snapshot; ``external_delay`` is None until a
        COARSE estimate exists; the four PSDs are single-channel linear-power spectra;
        ``active_render`` is the far-end active-frame flag. The subtractor max-abs values,
        ``echo_path_gain`` and ``render_block`` feed SaturationDetector. ``filter_taps_full``
        (full time-domain filter) is required when ``enable_filter_analyzer`` is True.
        """
        # Convergence flag from the filter bridge.
        any_filter_converged = bridge.filter_converged

        # FilterAnalyzer, before FilterDelay so its delays are current.
        analyzer_delays: Optional[list[int]] = None
        if (self._filter_analyzer is not None
                and filter_taps_full is not None
                and render_block is not None):
            self._filter_analyzer.update(filter_taps_full, render_block)
            analyzer_delays = self._filter_analyzer.filter_delays_blocks()

        self._delay_state.update(
            analyzer_filter_delay_estimates_blocks=analyzer_delays,
            external_delay=external_delay,
            blocks_with_proper_filter_adaptation=self._strong_not_saturated_render_blocks,
        )

        if active_render:
            self._blocks_with_active_render += 1
        if active_render and not self._capture_signal_saturation:
            self._strong_not_saturated_render_blocks += 1

        # ERLE: transition-triggered soft reset before the update.
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
        self._erl_estimator.update(
            render_psd=render_psd,
            capture_psd=capture_psd,
            converged_filter=any_filter_converged,
        )

        # Saturation detector; echo_path_gain comes from the FilterAnalyzer when present.
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

        self._initial_state.update(active_render, self._capture_signal_saturation)

        # Transparent mode (disabled in production).
        if self._transparent_mode is not None:
            # Consistency from the FilterAnalyzer, else a converged-with-delay proxy.
            if self._filter_analyzer is not None:
                any_filter_consistent = self._filter_analyzer.any_filter_consistent()
            else:
                any_filter_consistent = (
                    any_filter_converged and external_delay is not None
                )
            # Divergence signal from the bridge, with a heuristic fallback for minimal bridges.
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

        # FilteringQualityAnalyzer last; it reads the flags set above.
        self._filter_quality.update(
            active_render=active_render,
            transparent_mode=self.transparent_mode_active(),
            saturated_capture=self._capture_signal_saturation,
            external_delay=external_delay,
            any_filter_converged=any_filter_converged,
            filter_analyzer_consistent=self.filter_analyzer_consistent(),
        )


    def _full_reset(self) -> None:
        """Path-change reset of everything except the external delay (the trigger source)."""
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
