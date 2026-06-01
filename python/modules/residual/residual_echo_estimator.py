"""ResidualEchoEstimator — port of AEC3 residual_echo_estimator.cc.

Mirrors docs/aec3_extracts/src/aec3/residual_echo_estimator.{cc,h}.

Single-channel port. Computes R² (residual echo power²) per-bin from:
  - Linear path  (UsableLinearEstimate=True):  R² = S²_linear / ERLE
  - NonLinear   (UsableLinearEstimate=False): R² = X² * echo_path_gain²
                with noise gate + stationary subtraction.
  - Saturated echo override: R² = Y² (capture power) in either path.

Reverb addition WIRED — ReverbDecayEstimator + ReverbFrequencyResponse
are owned by this class (bound lazily via attach_reverb_decay_estimator
at first hop). estimate() reads decay/tail directly from the bound
instances. aec_state.reverb_decay() / get_reverb_frequency_response()
are now dead code (no production caller); kept on AecState surface for
future architectural refactor where ownership moves to AecState per
AEC3 layout.

NeuralResidualEchoEstimator NOT ported (off by default in AEC3 config).

UseStationarityProperties WIRED — stationary-band R² zeroing happens in
the orchestrator post-estimate (orchestrator.py ~line 4636); gated by
``aec3_post_stationarity_zero_enabled`` (default True via the legacy
alias which is now the canonical control point). EchoAudibility class
wrapper not yet ported; functional behaviour is in place inline.
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

# Delay-centered render buffer depth.  Must cover max(filter_delay_blocks) +
# post_blocks + 1.  filter_delay_blocks is bounded by n_partitions (default 5
# for 832-sample filter at 160-sample hop), so 16 gives 3× headroom.
_DELAY_BUF_SIZE = 16


@dataclass(frozen=True)
class EchoModelConfig:
    """Subset of AEC3 ``EchoCanceller3Config::EchoModel`` we use."""

    # noise_floor_hold_ms — delay before allowing minimum-statistics noise
    # floor to creep upward by 1.1x. PURE WALL-CLOCK semantics (room noise
    # drift). Stored as ms so the derived hop count auto-scales with
    # hop_size at ResidualEchoEstimator construction
    # (ms_to_hops(500, 160, 16000) = 50 hops at default 16k/10ms).
    # Quiet-room cohort favours slower adapt-up (=less false positive
    # on speech transients).
    noise_floor_hold_ms: int = 500
    min_noise_floor_power: float = 1638400.0  # AEC3 default (= 16²·6400)
    noise_gate_power: float = 27509562.0       # AEC3 default
    noise_gate_slope: float = 0.3
    stationary_gate_slope: float = 10.0
    model_reverb_in_nonlinear_mode: bool = True


@dataclass(frozen=True)
class EpStrengthConfig:
    """Subset of AEC3 ``EchoCanceller3Config::EpStrength``."""

    # AEC3 echo_canceller3_config.h:EpStrength.default_gain = 1.0f.
    # Squared in GetEchoPathGain → R² = X² × default_gain² = X² in nonlinear
    # mode (UsableLinearEstimate=False, i.e. during startup convergence).
    # Dimensionless ratio (PSD × PSD), no hop/int16 scaling needed.
    # Prior Python defaults (0.014 / 0.020) were cohort-tuned WORKAROUNDS,
    # not AEC3 port values — they massively under-estimated R² (2500×) in
    # nonlinear mode → opening-farend leaks through during convergence
    # window (~400 ms usable_linear gate + ~2 s ERLE warmup). Restored to
    # AEC3 strict default 2026-05-27.
    default_gain: float = 1.0
    bounded_erl: bool = False
    erle_onset_compensation_in_dominant_nearend: bool = False


@dataclass(frozen=True)
class ReverbConfig:
    """AEC3-strict reverb tail. Mirrors ``EchoCanceller3Config.ep_strength``:

      AEC3 default: default_len = nearend_len = 0.83  (h:134-135)
      ReverbDecayEstimator uses default_len for steady decay, nearend_len for
      mild decay. With default_len > 0, AEC3 disables adaptive estimation and
      returns the static value (use_adaptive_echo_decay_ flag in
      reverb_decay_estimator.cc:92 = (default_len < 0)).

    ``decay = 0.83`` is the AEC3 per-block (4 ms) multiplier. The
    physical wall-clock T_60 at AEC3 = ``log(0.001)/log(0.83)*4ms``
    ≈ 148 ms — typical echo-path (mic→speaker→mic) reverberation.
    Per-hop conversion at our 10 ms hop is applied at call site in
    ``ResidualEchoEstimator._reverb_decay()``:
        d_per_hop = 0.83 ** (hop_size / 64)  ≈ 0.624 at hop=160
    Using 0.83 verbatim per-hop would give T_60 ≈ 371 ms (2.5× too
    long), inflating R²_reverb steady-state by 2.2× when scaling is
    held stale during sustained NE without filter convergence.

    ``mild_decay_scale = 1.0`` keeps mild_decay == decay (AEC3 strict —
    default_len == nearend_len). The legacy 0.5 Python value was a
    Python-only acceleration during dominant_nearend.
    """

    decay: float = 0.83
    mild_decay_scale: float = 1.0
    enabled: bool = True
    # ``use_adaptive_decay = False`` is AEC3-strict default
    # (default_len = 0.83 > 0 → AEC3 disables the estimator). The
    # estimator is retained as a default-OFF candidate.
    # ``use_freq_response`` swaps the S²/X² coupling approximation for
    # ReverbFrequencyResponse-produced tail_response (AEC3-strict
    # linear path).
    use_adaptive_decay: bool = False
    use_freq_response: bool = True
    # AEC3 echo_canceller3_config.h:139 default = true. AEC3 strict semantic:
    # `tail_response[k] = max(tail, raw_tail_partition)` per bin, then
    # neighbour-max smoothing. Restored 2026-05-27 after ReverbFrequencyResponse
    # got fft-resolution-aware windows (raw_tail pre-smoothed over ±125 Hz
    # before the max; neighbour-max window also widened to ±125 Hz).
    conservative_tail_freq_response: bool = True


_TRANSPARENT_MODE_GAIN = 0.01  # AEC3 kDefaultTransparentModeGain verbatim


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
        # R0.2: use corrected residual noise gate constant (27509.42 int16²)
        # vs buggy 27509562 (1000× too large — see aec3_scale.py:134-149).
        # Default flipped to True 2026-05-27 for AEC3 alignment.
        use_aec3_residual_noise_gate: bool = True,
        # R0.3: AEC3 EchoGeneratingPower delay-centered window (pre=1, post=1)
        # vs legacy ring buffer (pre=0). Default flipped to True 2026-05-27.
        use_aec3_echo_gen_window: bool = True,
        # ReverbFrequencyResponse EMA wall-clock alignment.
        # AEC3 applies α=0.2·quality per 4 ms block; our verbatim port
        # applied per 10 ms hop (2.5× too slow). Default OFF for byte-equal.
        use_aec3_wallclock_reverb_smoothing: bool = False,
        # L1: Kuech-Kellermann second-order nonlinear residual (default OFF).
        # See AecConfig.nl_r2_* for full spec.
        nl_r2_enabled: bool = False,
        nl_r2_alpha: float = 0.1,
    ) -> None:
        self._n_bins = int(n_bins)
        self._echo_model = echo_model
        self._ep_strength = ep_strength
        self._reverb_cfg = reverb
        # v3.22: reverb-tail R² conservativeness scale (set by the orchestrator
        # from AecConfig.reverb_tail_strength). 1.0 = unchanged. See config.
        self._reverb_tail_strength = 1.0
        self._sr = int(sr)
        self._hop_size = int(hop_size)
        # R0.2: corrected residual noise gate (27509.42 int16² vs buggy 27509562).
        self._use_aec3_residual_noise_gate = bool(use_aec3_residual_noise_gate)
        # R0.3: corrected EchoGeneratingPower render pre-window (pre=1 vs default pre=0).
        self._use_aec3_echo_gen_window = bool(use_aec3_echo_gen_window)
        # Derive wall-clock hops from cfg.noise_floor_hold_ms once at init
        # (echo_model is frozen so the value is stable).
        self._noise_floor_hold_hops = _aec3_scale.ms_to_hops(
            echo_model.noise_floor_hold_ms, self._hop_size, self._sr
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
        # EchoGeneratingPower window walk. AEC3 walks the render history
        # `[delay - pre, delay + post + 1)` and takes the element-wise
        # max for each bin. AEC3 default pre=1, post=1 (3 blocks).
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
        # R0.3 strict AEC3 EchoGeneratingPower: delay-centered render buffer.
        # Index 0 = most recent frame, index k = k hops ago.
        # Used only when _use_aec3_echo_gen_window=True.
        self._delay_render_buf: deque = deque(maxlen=_DELAY_BUF_SIZE)
        # AEC3 strict reverb render history (residual_echo_estimator.cc:367-376).
        # Reverb model is fed render from `FilterLengthBlocks() + 1` blocks ago
        # (linear path) or `MinDirectPathFilterDelay() + 1` blocks ago (nonlinear).
        # Index 0 = current frame (after push), index k = k hops ago. Sized to
        # cover the largest plausible filter length + headroom.
        self._reverb_render_history: deque = deque(maxlen=_DELAY_BUF_SIZE)
        # Diagnostics for last estimate() call (readable by orchestrator trace).
        self._last_echo_gen_delay_blocks: int = 0
        self._last_echo_gen_idx_start: int = 0
        self._last_echo_gen_idx_stop: int = 0
        # HF paint-black diag — which R² path executed + components.
        self._last_r2_path: str = 'unset'
        self._last_r2_direct_component = np.zeros(self._n_bins, dtype=np.float32)
        self._last_r2_reverb_component = np.zeros(self._n_bins, dtype=np.float32)
        # Kill-stage R² decomposition: capture the inputs to R²_direct so
        # the tracer can attribute inflation to S²_linear vs ERLE divisor
        # vs reverb tail. All read-only (no audio path effect).
        self._last_s2_linear = np.zeros(self._n_bins, dtype=np.float32)
        self._last_erle_per_bin = np.zeros(self._n_bins, dtype=np.float32)
        self._last_tail_response = np.zeros(self._n_bins, dtype=np.float32)
        self._last_average_decay: float = 0.0
        # Adaptive reverb decay + tail freq response. Both are LAZY-bound;
        # orchestrator calls `attach_reverb_estimators` at the first hop
        # where it knows `n_partitions` and `hop_size`.
        self._reverb_decay_est: Optional[ReverbDecayEstimator] = None
        self._reverb_freq_resp: Optional[ReverbFrequencyResponse] = None
        # L1: Kuech-Kellermann second-order nonlinear residual (default OFF).
        # R²_nl = nl_alpha × x2² / _nl_norm_power, added to nonlinear path only.
        # _nl_norm_power scales with frame_size (2×hop_size) to preserve the
        # r2_nl/r2_linear ratio as hop_size changes — calibrated at hop=160
        # (empirically gives +0.079/+0.106 FS echo improvement).
        self._nl_r2_enabled = bool(nl_r2_enabled)
        self._nl_r2_alpha = float(nl_r2_alpha)
        self._nl_norm_power = float(_aec3_scale.nl_r2_norm_power(self._hop_size))
        # Per-bin residual noise gate — AEC3 27509.42 int16² scaled to our frame.
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
        """One-time bind of the adaptive decay estimator. No-op if
        ``use_adaptive_decay`` is False or estimator already bound.

        ``use_aec3_block_energy``: when True, the estimator runs the full
        AEC3-strict `reverb_decay_estimator.cc` algorithm on the
        time-domain impulse response (BlockEnergyAverage / Peak +
        EarlyReverbLengthEstimator + LateReverbLinearRegressor). When
        False, the legacy partition-energy linear regression path is used.
        """
        if not self._reverb_cfg.use_adaptive_decay:
            return
        if self._reverb_decay_est is not None:
            return
        self._reverb_decay_est = ReverbDecayEstimator(
            n_partitions=int(n_partitions),
            hop_size=int(hop_size),
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
        """Per-hop refresh of adaptive decay + tail-freq response.

        ``frequency_response`` shape ``(n_partitions, n_freqs)`` float32 with
        per-partition |W|² entries. The orchestrator computes this once and
        feeds it to both sub-estimators (avoids a duplicate FFT walk).

        ``time_domain_filter``: concatenated TD impulse response. Required
        when the decay estimator was constructed with
        ``use_aec3_block_energy=True``; ignored otherwise.
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
        s2_linear: np.ndarray,     # |H·X|² from PBFDKF
        dominant_nearend: bool,
        filter_freq_response: Optional[np.ndarray] = None,
        filter_delay_blocks: int = 0,
        filter_length_blocks: int = 0,
        force_nonlinear_path: bool = False,
        # B: Emura 2017 cross-PSD R² (nearend-robust per-bin estimate, int16² units).
        # When provided, blended with ERLE-based R² in the linear path.
        r2_emura: Optional[np.ndarray] = None,
        emura_r2_blend: float = 0.5,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Returns ``(R2, R2_unbounded)``.

        ``filter_freq_response``: per-bin filter response magnitude² (sum of
        |W[p]|² over partitions, then PSD-scaled to match render_psd scale).
        Used as ``power_spectrum_scaling`` for the linear-mode reverb update
        (AEC3 cc:391-392 ``GetReverbFrequencyResponse``). Falls back to
        per-frame S²/X² coupling when None — that fallback misses bins where
        the filter learned coupling but current frame has no render energy.
        """
        # Step 1: update stationary render noise floor.
        self._update_render_noise_power(render_psd)

        # AEC3 strict reverb history: push current render to ring buffer so
        # reverb update can read N+1 blocks ago (mirrors AEC3
        # render_buffer.Spectrum(first_reverb_partition), cc:367-376).
        # Index 0 = current frame after appendleft.
        self._reverb_render_history.appendleft(
            np.asarray(render_psd, dtype=np.float32).copy()
        )

        r2 = np.empty(self._n_bins, dtype=np.float32)
        r2_unbounded = np.empty(self._n_bins, dtype=np.float32)

        usable = aec_state.usable_linear_estimate()
        # AEC3 JustResetEchoPath override (use_aec3_just_reset_gate_on_linear_path):
        # caller signals that the linear residual is untrustworthy this
        # frame (e.g. coarse-rescue hangover active) and the nonlinear
        # estimator should run instead.
        if force_nonlinear_path:
            usable = False
        saturated = aec_state.saturated_echo()

        # Diagnostic — records which R² path executed this frame so
        # downstream paint-black trace can attribute HF gain drops.
        # No audio effect.
        if saturated:
            self._last_r2_path = 'saturated'
        else:
            self._last_r2_path = 'linear' if usable else 'nonlinear'
        self._last_r2_reverb_component = np.zeros(self._n_bins, dtype=np.float32)
        self._last_r2_direct_component = np.zeros(self._n_bins, dtype=np.float32)
        # Kill-stage diag — capture S²_linear input + ERLE divisor (linear
        # path only; reset to zeros at entry so nonlinear hops report 0).
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
                # Kill-stage diag — stash the ERLE that became the divisor.
                self._last_erle_per_bin = np.asarray(
                    erle, dtype=np.float32
                ).copy()
                # AEC3 cc:91-105 — R² = S²_linear / ERLE per-bin.
                r2[:] = s2_linear / np.maximum(erle, 1e-30)
                r2_unbounded[:] = s2_linear / np.maximum(erle_unb, 1e-30)
                # B: Emura 2017 blend — replace fraction of ERLE-based R² with
                # cross-PSD estimate (nearend-robust: E[nearend×X*]→0 in EMA).
                # In DT: R2_emura << R2_erle (nearend excluded) → gain higher
                # → nearend preserved. In FS: comparable → no echo regression.
                if r2_emura is not None and emura_r2_blend > 0.0:
                    _b = float(emura_r2_blend)
                    r2[:] = (1.0 - _b) * r2 + _b * r2_emura
                    r2_unbounded[:] = (1.0 - _b) * r2_unbounded + _b * r2_emura
            # AEC3 cc:257-260 — UpdateReverb(kLinear) + AddReverb.
            # Linear scaling uses per-bin filter freq-response. Without
            # the full ReverbFrequencyResponse port, approximate by the
            # current-frame per-bin coupling S²/X² (clipped to avoid
            # noise blow-up where X² is tiny).
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
                # EchoGeneratingPower window walk
                # (residual_echo_estimator.cc:133-165).
                _rp = np.asarray(render_psd, dtype=np.float32)
                if self._use_aec3_echo_gen_window:
                    # R0.3 strict AEC3: delay-centered window.
                    # idx_start = max(0, delay-pre), idx_stop = delay+post.
                    # Deque index 0 = current frame, k = k hops ago.
                    self._delay_render_buf.appendleft(_rp)
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
                    # Legacy: recent-N ring buffer (default-OFF path).
                    if not self._render_history_initialised:
                        self._render_history[:] = _rp
                        self._render_history_initialised = True
                    else:
                        self._render_history[self._render_history_idx] = _rp
                        self._render_history_idx = (
                            self._render_history_idx + 1
                        ) % self._render_history_size
                    x2 = np.max(self._render_history, axis=0).copy()
                if not aec_state.transparent_mode_active():
                    # AEC3 cc:121-129 noise gate.
                    # R0.2: use corrected 27509.42 (int16²) instead of buggy 27509562.
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
                # Subtract stationary noise (AEC3 cc:284-288).
                x2 -= self._echo_model.stationary_gate_slope * self._x2_noise_floor
                np.maximum(x2, 0.0, out=x2)
                r2[:] = x2 * echo_path_gain
                r2_unbounded[:] = x2 * echo_path_gain
                # L1: Kuech-Kellermann second-order nonlinear residual.
                # Adds quadratic render PSD term to capture loudspeaker
                # harmonic / intermodulation distortion echo.
                # Only in nonlinear path (filter not converged): linear path
                # already captures the true residual via S²/ERLE.
                if self._nl_r2_enabled and self._nl_r2_alpha > 0.0:
                    r2_nl = (self._nl_r2_alpha
                             * x2 ** 2 / self._nl_norm_power).astype(np.float32)
                    r2 += r2_nl
                    r2_unbounded += r2_nl
            # AEC3 cc:294-300 — UpdateReverb(kNonLinear) + AddReverb.
            if (
                self._echo_model.model_reverb_in_nonlinear_mode
                and not aec_state.transparent_mode_active()
            ):
                # Nonlinear flat scaling = echo_path_gain (post-square).
                ep_late = self._get_echo_path_gain(
                    aec_state, gain_for_early_reflections=False
                )
                decay = self._reverb_decay(dominant_nearend)
                # AEC3 nonlinear reverb update reads render from
                # MinDirectPathFilterDelay() + 1 blocks ago (cc:370).
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

        # Kill-stage diag — snapshot reverb-freq-resp state at end of hop
        # so the tracer can attribute reverb inflation (tail_response is
        # the per-bin tail shape; average_decay is the scalar damping).
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
        """Mirrors UpdateRenderNoisePower (cc:325-359). Per-bin
        min-statistics with rapid decrease + delayed leaky increase."""
        # Decrease rapidly: where X² < current noise floor, snap down.
        mask_down = render_psd < self._x2_noise_floor
        self._x2_noise_floor[mask_down] = render_psd[mask_down].astype(np.float32)
        self._x2_noise_floor_counter[mask_down] = 0
        # Increase: bins past the hold counter ramp up 10% / step (clamped to min).
        not_down = ~mask_down
        hold = self._noise_floor_hold_hops
        ramp_mask = not_down & (self._x2_noise_floor_counter >= hold)
        if ramp_mask.any():
            self._x2_noise_floor[ramp_mask] = np.maximum(
                self._x2_noise_floor[ramp_mask] * 1.1,
                self._echo_model.min_noise_floor_power,
            )
        # Counter increment for non-down bins not yet past hold.
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
        """``aec_state.ReverbDecay(mild=dominant_nearend)`` with wall-clock
        rate alignment to AEC3.

        When the adaptive estimator is bound + active, query it. Otherwise
        fall back to the static config decay × mild_decay_scale.

        AEC3 spec uses ``decay`` as the per-block (4 ms / 64-sample)
        multiplier in ``reverb_[k] = (reverb_[k] + injection) * decay``
        applied per ``ReverbModel::UpdateReverb`` call. Our pipeline calls
        the same update once per hop (10 ms / 160-sample), so the AEC3
        constant 0.83 applied verbatim would decay 2.5× slower in
        wall-clock — inflating R²_reverb steady-state when the filter is
        unconverged and the scaling input is held stale.

        Wall-clock alignment: ``d_per_hop = d_aec3 ** (hop / 64)``.
        At hop=160 this is ``0.83 ** 2.5 ≈ 0.624``. The conversion is
        applied to BOTH static-config decay and the adaptive estimator
        output, since both feed the same per-hop ReverbModel.update call.
        """
        if not self._reverb_cfg.enabled:
            return 0.0
        if self._reverb_decay_est is not None:
            d = self._reverb_decay_est.decay(mild=dominant_nearend)
        else:
            d = float(self._reverb_cfg.decay)
            if dominant_nearend:
                d *= float(self._reverb_cfg.mild_decay_scale)
        # Wall-clock alignment vs AEC3 kBlockSize=64.
        _AEC3_BLOCK_SAMPLES = 64
        if self._hop_size != _AEC3_BLOCK_SAMPLES:
            d = float(d) ** (self._hop_size / _AEC3_BLOCK_SAMPLES)
        return float(d)

    def _update_reverb_linear(
        self, render_psd: np.ndarray, s2_linear: np.ndarray,
        dominant_nearend: bool, filter_length_blocks: int,
    ) -> None:
        """Linear-mode reverb update (AEC3 cc:390-392).

        AEC3 strict (cc:367-376): the render power fed to the reverb model
        is from ``FilterLengthBlocks() + 1`` blocks ago — i.e. render whose
        echo arrives AFTER the linear filter's tap coverage and therefore
        constitutes the late reverberant tail the linear filter cannot model.

        Using the current frame's render here (the prior Python behaviour)
        double-counts: the linear filter is already cancelling that energy's
        direct echo via the S²/ERLE path, then reverb_model re-injects it
        as "tail" → R² grossly over-estimated, especially at HF where the
        filter's direct-path coupling is sparse. The HF "painted black"
        artifact during DT is a direct consequence.

        When ``ReverbFrequencyResponse`` is bound, use its
        ``tail_response`` (canonical AEC3 mechanism) as the per-bin
        scaling. Otherwise fall back to current-frame coupling ``S²/X²``.
        """
        decay = self._reverb_decay(dominant_nearend)
        if decay <= 0.0:
            return
        # AEC3 cc:367-368: linear path uses FilterLengthBlocks() + 1.
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
