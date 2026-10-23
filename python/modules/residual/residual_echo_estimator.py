"""ResidualEchoEstimator: per-bin residual echo power R^2 (single channel).

* usable linear estimate: R^2 = S^2_linear / ERLE, plus the reverb tail
* otherwise: R^2 = X^2 * echo_path_gain^2 with a noise gate (skipped when ``use_stationarity_properties``), plus the tail
* saturated echo: R^2 = Y^2 in either path

The reverb decay estimator is bound lazily (``attach_reverb_decay_estimator``); the orchestrator zeroes stationary bands.
"""
from collections import deque
from dataclasses import dataclass
from typing import Optional

import numpy as np

from ..state.aec_state import AecState
from .. import aec3_scale as _aec3_scale
from .reverb_model import ReverbModel
from .reverb_decay_estimator import ReverbDecayEstimator
from .reverb_frequency_response import ReverbFrequencyResponse

# Depth of the delay-centred render buffer; covers max(filter_delay_blocks) + post blocks + 1.
_DELAY_BUF_SIZE = 16


@dataclass(frozen=True)
class EchoModelConfig:
    """Subset of the echo-model configuration."""

    # Hold time before the minimum-statistics noise floor may creep upward (wall-clock ms,
    # converted to hops at construction).
    noise_floor_hold_ms: int = 500
    min_noise_floor_power: float = 1638400.0  # AEC3 default (= 16²·6400)
    noise_gate_power: float = 27509562.0       # AEC3 default
    noise_gate_slope: float = 0.3
    stationary_gate_slope: float = 10.0
    model_reverb_in_nonlinear_mode: bool = True


@dataclass(frozen=True)
class EpStrengthConfig:
    """Echo-path strength: ``default_gain`` squares into R^2 = X^2 * gain^2 in the nonlinear path."""

    # Dimensionless ratio of PSDs; 1.0 means R^2 = X^2 in nonlinear mode (startup convergence).
    default_gain: float = 1.0
    bounded_erl: bool = False
    erle_onset_compensation_in_dominant_nearend: bool = False


@dataclass(frozen=True)
class ReverbConfig:
    """Reverb tail: ``decay`` is the per-block (4 ms) multiplier on reverb power (0.83), converted
    per hop as ``decay ** (hop_seconds / 0.004)`` (0.628 at a 10 ms hop). ``mild_decay_scale``
    scales the decay while near-end is dominant; 1.0 keeps it equal to ``decay``.
    """

    decay: float = 0.83
    mild_decay_scale: float = 1.0
    enabled: bool = True
    # ``use_adaptive_decay`` enables the adaptive decay estimator (off: static decay);
    # ``use_freq_response`` uses ReverbFrequencyResponse's tail response instead of the S^2/X^2 coupling.
    use_adaptive_decay: bool = False
    use_freq_response: bool = True
    # Conservative tail variant: point-wise max with the raw last partition, then neighbour-max smoothing.
    conservative_tail_freq_response: bool = True


_TRANSPARENT_MODE_GAIN = 0.01  # transparent-mode gain


class ResidualEchoEstimator:
    def __init__(
        self,
        *,
        n_bins: int = 257,
        echo_model: EchoModelConfig = EchoModelConfig(),
        ep_strength: EpStrengthConfig = EpStrengthConfig(),
        reverb: ReverbConfig = ReverbConfig(),
        sr: int = 16000,
        hop_size: int = 160,
        use_aec3_residual_noise_gate: bool = True,
        # Delay-centred EchoGeneratingPower window (pre=1, post=1) instead of the plain recent-N ring buffer.
        use_aec3_echo_gen_window: bool = True,
        # Skip the nonlinear-path noise gate (the orchestrator zeroes stationary bands downstream).
        use_stationarity_properties: bool = False,
        # Per-hop reverb-frequency-response smoothing converted from the per-block rate.
        use_aec3_wallclock_reverb_smoothing: bool = False,
        # Second-order nonlinear residual term (see AecConfig.nl_r2_*).
        nl_r2_enabled: bool = False,
        nl_r2_alpha: float = 0.1,
        # Cap on the nonlinear term as a multiple of the direct term; 0.0 = uncapped.
        nl_r2_cap_multiple: float = 0.0,
    ) -> None:
        self._n_bins = int(n_bins)
        self._echo_model = echo_model
        self._ep_strength = ep_strength
        self._reverb_cfg = reverb
        # Reverb-tail R^2 scale, set by the orchestrator from AecConfig.reverb_tail_strength.
        self._reverb_tail_strength = 1.0
        self._sr = int(sr)
        self._hop_size = int(hop_size)
        self._use_aec3_residual_noise_gate = bool(use_aec3_residual_noise_gate)
        self._use_aec3_echo_gen_window = bool(use_aec3_echo_gen_window)
        self._use_stationarity_properties = bool(use_stationarity_properties)
        self._noise_floor_hold_hops = _aec3_scale.ms_to_hops(
            echo_model.noise_floor_hold_ms, self._hop_size, self._sr
        )
        # Per-block growth of the noise floor (x1.1), converted to per hop.
        self._noise_floor_growth_per_hop = _aec3_scale.per_block_growth_to_per_hop(
            1.1, self._hop_size, self._sr
        )
        self._tm_gain_early = _TRANSPARENT_MODE_GAIN
        self._tm_gain_late = _TRANSPARENT_MODE_GAIN
        self._default_gain_early = float(ep_strength.default_gain)
        self._default_gain_late = float(ep_strength.default_gain)
        self._erle_onset_compensation_in_dominant = bool(
            ep_strength.erle_onset_compensation_in_dominant_nearend
        )
        self._x2_noise_floor = np.full(
            self._n_bins, echo_model.min_noise_floor_power, dtype=np.float32
        )
        self._x2_noise_floor_counter = np.full(
            self._n_bins, self._noise_floor_hold_hops, dtype=np.int32
        )
        self._reverb_model = ReverbModel(n_bins=self._n_bins)
        # EchoGeneratingPower walks the render history [delay - pre, delay + post] taking the per-bin max.
        self._render_pre_window_size = 1 if self._use_aec3_echo_gen_window else 0
        self._render_post_window_size = 1
        self._render_history_size = (
            self._render_pre_window_size + self._render_post_window_size + 1
        )
        self._render_history = np.zeros(
            (self._render_history_size, self._n_bins), dtype=np.float32
        )
        self._render_history_idx = 0
        self._render_history_initialised = False
        # Delay-centred render buffer: index 0 is the newest frame.
        self._delay_render_buf: deque = deque(maxlen=_DELAY_BUF_SIZE)
        # Reverb render history (index 0 = current frame); the reverb model is fed render from
        # filter_length_blocks + 1 (linear) or min filter delay + 1 (nonlinear) blocks ago.
        self._reverb_render_history: deque = deque(maxlen=_DELAY_BUF_SIZE)
        self._last_echo_gen_delay_blocks: int = 0
        self._last_echo_gen_idx_start: int = 0
        self._last_echo_gen_idx_stop: int = 0
        self._last_r2_path: str = 'unset'
        self._last_r2_direct_component = np.zeros(self._n_bins, dtype=np.float32)
        self._last_r2_reverb_component = np.zeros(self._n_bins, dtype=np.float32)
        self._last_s2_linear = np.zeros(self._n_bins, dtype=np.float32)
        self._last_erle_per_bin = np.zeros(self._n_bins, dtype=np.float32)
        self._last_tail_response = np.zeros(self._n_bins, dtype=np.float32)
        self._last_average_decay: float = 0.0
        # Bound lazily by the orchestrator once n_partitions and hop_size are known.
        self._reverb_decay_est: Optional[ReverbDecayEstimator] = None
        self._reverb_freq_resp: Optional[ReverbFrequencyResponse] = None
        # Second-order nonlinear residual: R^2_nl = alpha * x2^2 / norm, normalised by frame size (nonlinear path only).
        self._nl_r2_enabled = bool(nl_r2_enabled)
        self._nl_r2_alpha = float(nl_r2_alpha)
        self._nl_norm_power = float(_aec3_scale.nl_r2_norm_power(self._hop_size))
        # Cap on r2_nl as a multiple of the direct term (the raw term is unbounded on loud cold-start render).
        self._nl_r2_cap_multiple = float(nl_r2_cap_multiple)
        # per-bin residual noise gate scaled to the frame size
        self._noise_gate_power = float(
            _aec3_scale.per_bin_psd_threshold(27509.42, self._hop_size)
        )
        if self._reverb_cfg.use_freq_response:
            self._reverb_freq_resp = ReverbFrequencyResponse(
                n_freqs=self._n_bins,
                use_conservative_tail_frequency_response=(
                    self._reverb_cfg.conservative_tail_freq_response
                ),
                sr=self._sr,
                hop_size=self._hop_size,
                use_wallclock_smoothing=bool(use_aec3_wallclock_reverb_smoothing),
            )

    def attach_reverb_decay_estimator(self, n_partitions: int,
                                      hop_size: int,
                                      use_aec3_block_energy: bool = False) -> None:
        """Bind the adaptive decay estimator once; no-op unless ``use_adaptive_decay`` or if already bound.

        ``use_aec3_block_energy`` selects the time-domain impulse-response estimator instead of
        the partition-energy regression.
        """
        if not self._reverb_cfg.use_adaptive_decay:
            return
        if self._reverb_decay_est is not None:
            return
        self._reverb_decay_est = ReverbDecayEstimator(
            n_partitions=int(n_partitions),
            hop_size=int(hop_size),
            sample_rate=int(self._sr),
            default_decay=float(self._reverb_cfg.decay),
            mild_decay=(float(self._reverb_cfg.decay)
                        * float(self._reverb_cfg.mild_decay_scale)),
            use_adaptive=True,
            use_aec3_block_energy=bool(use_aec3_block_energy),
        )

    def update_reverb_models(self,
                             *,
                             frequency_response: np.ndarray,
                             filter_delay_blocks: int,
                             filter_quality: Optional[float],
                             usable_linear_filter: bool,
                             stationary_block: bool,
                             time_domain_filter: Optional[np.ndarray] = None) -> None:
        """Per-hop refresh of the adaptive decay and the tail frequency response.

        ``frequency_response`` is (n_partitions, n_freqs) float32 |W|^2 per partition;
        ``time_domain_filter`` (concatenated impulse response) is required by the
        time-domain decay estimator and ignored otherwise.
        """
        if self._reverb_decay_est is not None:
            partition_energies = frequency_response.sum(axis=1).astype(np.float32)
            self._reverb_decay_est.update(
                partition_energies=partition_energies,
                time_domain_filter=time_domain_filter,
                filter_quality=filter_quality,
                filter_delay_blocks=int(filter_delay_blocks),
                usable_linear_filter=bool(usable_linear_filter),
                stationary_signal=bool(stationary_block),
            )
        if self._reverb_freq_resp is not None:
            self._reverb_freq_resp.update(
                frequency_response=frequency_response,
                filter_delay_blocks=int(filter_delay_blocks),
                linear_filter_quality=filter_quality,
                stationary_block=bool(stationary_block),
            )

    def reset(self) -> None:
        self._x2_noise_floor.fill(self._echo_model.min_noise_floor_power)
        self._x2_noise_floor_counter.fill(self._noise_floor_hold_hops)
        self._reverb_model.reset()
        if self._reverb_decay_est is not None:
            self._reverb_decay_est.reset()
        if self._reverb_freq_resp is not None:
            self._reverb_freq_resp.reset()

    def estimate(
        self,
        *,
        aec_state: AecState,
        render_psd: np.ndarray,    # X² current frame (single-channel)
        capture_psd: np.ndarray,   # Y²
        s2_linear: np.ndarray,     # selected (coarse-or-refined via URO) |echo_spec|^2
        dominant_nearend: bool,
        filter_delay_blocks: int = 0,
        filter_length_blocks: int = 0,
        force_nonlinear_path: bool = False,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Returns ``(R2, R2_unbounded)``."""
        self._update_render_noise_power(render_psd)

        _rp_current = np.asarray(render_psd, dtype=np.float32).copy()
        self._reverb_render_history.appendleft(_rp_current)
        # Updated every hop so linear-to-nonlinear transitions see the current render PSD.
        if self._use_aec3_echo_gen_window:
            self._delay_render_buf.appendleft(_rp_current)

        r2 = np.empty(self._n_bins, dtype=np.float32)
        r2_unbounded = np.empty(self._n_bins, dtype=np.float32)

        usable = aec_state.usable_linear_estimate()
        # ``force_nonlinear_path``: the linear residual is untrustworthy this frame (e.g. coarse-rescue hangover).
        if force_nonlinear_path:
            usable = False
        saturated = aec_state.saturated_echo()

        if saturated:
            self._last_r2_path = 'saturated'
        else:
            self._last_r2_path = 'linear' if usable else 'nonlinear'
        self._last_r2_reverb_component = np.zeros(self._n_bins, dtype=np.float32)
        self._last_r2_direct_component = np.zeros(self._n_bins, dtype=np.float32)
        self._last_s2_linear = np.asarray(s2_linear, dtype=np.float32).copy()
        self._last_erle_per_bin = np.zeros(self._n_bins, dtype=np.float32)

        if usable:
            if saturated:
                r2[:] = capture_psd
                r2_unbounded[:] = capture_psd
            else:
                onset_compensated = (
                    self._erle_onset_compensation_in_dominant or not dominant_nearend
                )
                erle = aec_state.erle(onset_compensated)
                erle_unb = aec_state.erle_unbounded()
                self._last_erle_per_bin = np.asarray(
                    erle, dtype=np.float32
                ).copy()
                r2[:] = s2_linear / np.maximum(erle, 1e-30)
                r2_unbounded[:] = s2_linear / np.maximum(erle_unb, 1e-30)
            self._update_reverb_linear(
                render_psd, s2_linear, dominant_nearend,
                filter_length_blocks,
            )
            reverb = self._reverb_model.reverb * self._reverb_tail_strength
            self._last_r2_direct_component = r2.copy()
            self._last_r2_reverb_component = np.asarray(
                reverb, dtype=np.float32
            ).copy()
            r2 += reverb
            r2_unbounded += reverb
        else:
            echo_path_gain = self._get_echo_path_gain(
                aec_state, gain_for_early_reflections=True
            )
            if saturated:
                r2[:] = capture_psd
                r2_unbounded[:] = capture_psd
            else:
                _rp = np.asarray(render_psd, dtype=np.float32)
                if self._use_aec3_echo_gen_window:
                    # Delay-centred window: idx_start = max(0, delay - pre), idx_stop = delay + post.
                    _delay = max(0, int(filter_delay_blocks))
                    _pre  = self._render_pre_window_size   # 1
                    _post = self._render_post_window_size  # 1
                    _idx_start = max(0, _delay - _pre)
                    _idx_stop  = min(len(self._delay_render_buf) - 1,
                                     _delay + _post)
                    self._last_echo_gen_delay_blocks = _delay
                    self._last_echo_gen_idx_start    = _idx_start
                    self._last_echo_gen_idx_stop     = _idx_stop
                    _slices = [self._delay_render_buf[i]
                               for i in range(_idx_start, _idx_stop + 1)]
                    x2 = (np.maximum.reduce(_slices).copy()
                          if len(_slices) > 1 else _slices[0].copy())
                else:
                    if not self._render_history_initialised:
                        self._render_history[:] = _rp
                        self._render_history_initialised = True
                    else:
                        self._render_history[self._render_history_idx] = _rp
                        self._render_history_idx = (
                            self._render_history_idx + 1
                        ) % self._render_history_size
                    x2 = np.max(self._render_history, axis=0).copy()
                if not self._use_stationarity_properties:
                    # Nonlinear-path noise gate (skipped when stationarity properties are used).
                    _ng = (self._noise_gate_power
                           if self._use_aec3_residual_noise_gate
                           else self._echo_model.noise_gate_power)
                    mask = _ng > x2
                    x2[mask] = np.maximum(
                        0.0,
                        x2[mask]
                        - self._echo_model.noise_gate_slope
                        * (_ng - x2[mask]),
                    )
                x2 -= self._echo_model.stationary_gate_slope * self._x2_noise_floor
                np.maximum(x2, 0.0, out=x2)
                r2[:] = x2 * echo_path_gain
                r2_unbounded[:] = x2 * echo_path_gain
                # Second-order nonlinear residual term (nonlinear path only).
                if self._nl_r2_enabled and self._nl_r2_alpha > 0.0:
                    r2_nl = (self._nl_r2_alpha
                             * x2 ** 2 / self._nl_norm_power).astype(np.float32)
                    if self._nl_r2_cap_multiple > 0.0:
                        np.minimum(
                            r2_nl, r2 * self._nl_r2_cap_multiple, out=r2_nl
                        )
                    r2 += r2_nl
                    r2_unbounded += r2_nl
            if (
                self._echo_model.model_reverb_in_nonlinear_mode
                and not aec_state.transparent_mode_active()
            ):
                ep_late = self._get_echo_path_gain(
                    aec_state, gain_for_early_reflections=False
                )
                decay = self._reverb_decay(dominant_nearend)
                # Nonlinear reverb update reads render from min filter delay + 1 blocks ago.
                _nl_offset = max(0, int(filter_delay_blocks)) + 1
                if _nl_offset < len(self._reverb_render_history):
                    delayed_render = self._reverb_render_history[_nl_offset]
                else:
                    delayed_render = None
                if delayed_render is not None:
                    self._reverb_model.update_no_freq_shaping(
                        delayed_render, scaling=ep_late, decay=decay
                    )
                reverb = self._reverb_model.reverb * self._reverb_tail_strength
                self._last_r2_direct_component = r2.copy()
                self._last_r2_reverb_component = np.asarray(
                    reverb, dtype=np.float32
                ).copy()
                r2 += reverb
                r2_unbounded += reverb

        if self._reverb_freq_resp is not None:
            _tail = getattr(self._reverb_freq_resp, 'tail_response', None)
            if _tail is not None:
                self._last_tail_response = np.asarray(
                    _tail, dtype=np.float32
                ).copy()
            self._last_average_decay = float(
                getattr(self._reverb_freq_resp, 'average_decay', 0.0)
            )

        return r2, r2_unbounded

    # ------------------------------------------------------------- helpers

    def _update_render_noise_power(self, render_psd: np.ndarray) -> None:
        """Per-bin minimum-statistics render noise floor: rapid decrease, delayed 10 % increase."""
        mask_down = render_psd < self._x2_noise_floor
        self._x2_noise_floor[mask_down] = render_psd[mask_down].astype(np.float32)
        self._x2_noise_floor_counter[mask_down] = 0
        not_down = ~mask_down
        hold = self._noise_floor_hold_hops
        ramp_mask = not_down & (self._x2_noise_floor_counter >= hold)
        if ramp_mask.any():
            self._x2_noise_floor[ramp_mask] = np.maximum(
                self._x2_noise_floor[ramp_mask] * self._noise_floor_growth_per_hop,
                self._echo_model.min_noise_floor_power,
            )
        not_down_hold = not_down & (self._x2_noise_floor_counter < hold)
        self._x2_noise_floor_counter[not_down_hold] += 1

    def _get_echo_path_gain(
        self, aec_state: AecState, *, gain_for_early_reflections: bool
    ) -> float:
        if aec_state.transparent_mode_active():
            g = self._tm_gain_early if gain_for_early_reflections else self._tm_gain_late
        else:
            g = (
                self._default_gain_early
                if gain_for_early_reflections
                else self._default_gain_late
            )
        return float(g * g)

    def _reverb_decay(self, dominant_nearend: bool) -> float:
        """Reverb decay at the hop rate: the static ``decay`` (times ``mild_decay_scale`` when
        near-end is dominant) or the adaptive estimate, raised to ``hop_seconds / 4 ms``."""
        if not self._reverb_cfg.enabled:
            return 0.0
        if self._reverb_decay_est is not None:
            d = self._reverb_decay_est.decay(mild=dominant_nearend)
        else:
            d = float(self._reverb_cfg.decay)
            if dominant_nearend:
                d *= float(self._reverb_cfg.mild_decay_scale)
        d = _aec3_scale.per_block_growth_to_per_hop(float(d), self._hop_size, self._sr)
        return float(d)

    def _update_reverb_linear(
        self, render_psd: np.ndarray, s2_linear: np.ndarray,
        dominant_nearend: bool, filter_length_blocks: int,
    ) -> None:
        """Linear-mode reverb update: the reverb model is fed render from filter_length_blocks + 1
        blocks ago (energy the linear filter does not cover), scaled by the tail frequency response
        when bound, else by the clipped current-frame coupling S^2/X^2."""
        decay = self._reverb_decay(dominant_nearend)
        if decay <= 0.0:
            return
        _offset = max(0, int(filter_length_blocks)) + 1
        if _offset >= len(self._reverb_render_history):
            return  # buffer not warm yet — AEC3 RenderBuffer returns zeros
        delayed_render = self._reverb_render_history[_offset]
        if self._reverb_freq_resp is not None:
            scaling = self._reverb_freq_resp.tail_response.astype(np.float32)
        else:
            gain_cap = (self._default_gain_late * self._default_gain_late) * 4.0
            scaling = s2_linear / np.maximum(render_psd, 1e-10)
            np.minimum(scaling, gain_cap, out=scaling)
        self._reverb_model.update(delayed_render, scaling, decay)
