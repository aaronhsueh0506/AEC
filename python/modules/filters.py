"""Adaptive filter primitives: ``PBFDAF`` (partitioned-block frequency-domain NLMS) and ``PBFDKF`` (Kalman extension)."""
from typing import Optional, Tuple

import numpy as np

from . import aec3_scale


class PBFDAF:
    """Partitioned-block frequency-domain adaptive filter (NLMS, overlap-save); ``PBFDKF`` adds Kalman adaptation."""

    def __init__(self, block_size: int, n_partitions: int,
                 mu: float = 0.3, delta: float = 1e-8,
                 hop_size: int = 0, sample_rate: int = 16000):
        # Overlap-save buffer is 2 x hop; the FFT size is the next power of two (a no-op on the product grids).
        self.hop_size = hop_size if hop_size > 0 else block_size // 2
        # Sample rate converts the per-4 ms-block timing constants to wall-clock at the live grid.
        self.sample_rate = sample_rate
        self.block_size = 2 * self.hop_size  # overlap-save buffer (exactly 2× hop)
        self.fft_size = 1 << (self.block_size - 1).bit_length()  # next pow2
        self.n_partitions = n_partitions
        self.n_freqs = self.fft_size // 2 + 1
        self.mu = mu
        self.delta = delta
        self.enable_td_constraint = True  # can be disabled for diagnosis
        # Round-robin constraint: constrain one partition per hop (cycling) instead of all of them.
        # The orchestrator sets this from config; the filter-level default is off.
        self._constraint_round_robin = False
        self._partition_to_constrain = 0

        # Constrain keeps the causal half of each overlap-save partition and clears the circular half.
        self._td_window = np.zeros(self.fft_size, dtype=np.float32)
        self._td_window[:self.hop_size] = 1.0

        self.W = np.zeros((n_partitions, self.n_freqs), dtype=np.complex64)

        self.X_buf = np.zeros((n_partitions, self.n_freqs), dtype=np.complex64)
        self.partition_idx = 0

        self.near_buffer = np.zeros(self.block_size, dtype=np.float32)
        self.far_buffer = np.zeros(self.block_size, dtype=np.float32)

        self.near_spec = np.zeros(self.n_freqs, dtype=np.complex64)
        self.echo_spec = np.zeros(self.n_freqs, dtype=np.complex64)
        self.error_spec = np.zeros(self.n_freqs, dtype=np.complex64)
        self.far_spec = np.zeros(self.n_freqs, dtype=np.complex64)

        # Windowed error spectrum for RES analysis: periodic sqrt-Hann analysis window (denominator N)
        # matching the synthesis window, so analysis x synthesis sums to 1 across 50 % overlapped hops.
        _idx = np.arange(self.block_size, dtype=np.float64)
        self._sqrt_hann_analysis = np.sqrt(
            0.5 * (1.0 - np.cos(2.0 * np.pi * _idx / float(self.block_size)))
        ).astype(np.float32)
        self.error_spec_windowed = np.zeros(self.n_freqs, dtype=np.complex64)

        # Coarse-filter update protections (applied in _update_weights): partition-summed X^2 mu
        # denominator and noise gate, poor-excitation / startup gate, narrow-band mask, saturation gate.
        # State shared with the shadow gate; PBFDKF re-assigns its own defaults.
        self._call_counter = 0
        self._poor_excitation_counter = 0  # safe init; orchestrator sets to hop-scaled 1000-block default
        self._render_signal_analyzer = None
        self._saturated_capture = False
        # Per-frame gate-fire trace for diagnostics; read-only, never affects the audio output.
        self._c1c5_trace: dict = {}
        # Deferred update: with process(..., defer_update=True) the W update and partition_idx advance
        # wait for complete_update(), so the orchestrator can inject a different error source.
        self._deferred_update_pending: bool = False
        self._deferred_curr_p: int = 0
        self._deferred_mu_scale = 1.0
        self._deferred_far_hop_energy: float = 0.0

        # Initial-state tracking, counted in process(); only PBFDKF._h_error_refresh reads it.
        from . import aec3_scale as _aec3_scale
        self._initial_state_active: bool = True
        self._initial_state_active_render_hops: int = 0
        # Initial-state window of 2.5 s at the live grid.
        self._initial_state_threshold_hops: int = _aec3_scale.ms_to_hops(
            2500.0, self.hop_size, self.sample_rate
        )
        self._initial_state_far_energy_floor: float = 1e-4
        self._last_initial_state_active: bool = True

    @property
    def tap_span(self) -> int:
        """Echo reach in samples (``n_partitions * hop_size``)."""
        return self.n_partitions * self.hop_size

    def reset_taps(self):
        """Reset the adaptation state only: the taps and the per-partition far spectra."""
        self.W.fill(0)
        self.X_buf.fill(0)

    def reset(self):
        self.reset_taps()
        self.near_buffer.fill(0)
        self.far_buffer.fill(0)
        self.partition_idx = 0
        self.error_spec_windowed.fill(0)

    def reset_carried_state(self):
        """Restore the construction-time value of every scalar that ``reset()`` carries forward:
        startup / saturation gates, initial-state tracking, constraint cursor and last-hop peak.
        ``_poor_excitation_counter`` is not reset here; the orchestrator seeds it."""
        self._partition_to_constrain = 0
        self._call_counter = 0
        self._saturated_capture = False
        self._initial_state_active = True
        self._initial_state_active_render_hops = 0
        self._last_initial_state_active = True
        # Created lazily; their readers treat absence as the fresh value.
        for attr in ('_last_s_max_abs', '_block_stationary'):
            try:
                delattr(self, attr)
            except AttributeError:
                pass

    def warm_shift_ir(self, shift_samples: int) -> None:
        """Warm delay realign: shift the learned impulse response by ``shift_samples`` taps
        (positive: far stream advanced, response moves toward tap 0, IR shifts left), zero-filling
        the freed end. Shifts beyond the tap span are clamped (callers gate on the span). ``X_buf``
        is kept on an advance and cleared on a retard, where the held history is ahead of the new stream.
        """
        s = int(shift_samples)
        if s == 0:
            return
        hop, nF, nP = self.hop_size, self.fft_size, self.n_partitions
        ir = self.get_time_domain_filter()
        total = self.tap_span
        s = max(-total, min(s, total))
        ir_new = np.zeros_like(ir)
        if s > 0:
            ir_new[:total - s] = ir[s:]
        else:
            ir_new[-s:] = ir[:total + s]
        for p in range(nP):
            blk = np.zeros(nF, dtype=np.float32)
            blk[:hop] = ir_new[p * hop:(p + 1) * hop]
            self.W[p] = np.fft.rfft(blk, nF).astype(np.complex64)
        # Advance keeps X_buf (clearing it loses the immediate cancellation); retard clears it.
        if s < 0 or getattr(self, '_warm_clear_xbuf', False):
            self.X_buf.fill(0)

    def zero_filter_partitions(self, old_size: int, new_size: int) -> None:
        """Zero the filter partitions in [old_size, new_size); a no-op for the fixed-size filter."""
        old = max(0, int(old_size))
        new = min(int(new_size), self.W.shape[0])
        if old >= new:
            return
        self.W[old:new].fill(0)

    def handle_echo_path_change(self, delay_change: bool = True,
                                  gain_change: bool = False,
                                  zero_filter: bool = False) -> None:
        """Echo-path-change handling: unless ``gain_change``, re-arm the poor-excitation counter and
        reset the call counter. ``zero_filter`` additionally clears all of W (ablation only; off at every
        orchestrator call site)."""
        from . import aec3_scale as _aec3_scale
        if delay_change and zero_filter:
            self.W.fill(0)
        if not gain_change:
            # Poor-excitation initial value, 4 s at the live grid.
            self._poor_excitation_counter = _aec3_scale.blocks_to_hops(
                1000, self.hop_size, self.sample_rate
            )
            self._call_counter = 0

    def process(self, near_end: np.ndarray, far_end: np.ndarray,
                mu_scale=1.0, defer_update: bool = False,
                precomputed_far_spec=None) -> np.ndarray:
        """Process hop_size samples; ``mu_scale`` is a scalar or a per-bin array [n_freqs].

        ``defer_update=True`` skips the W update and partition advance and stashes the inputs for
        ``complete_update()``, which may supply an error override.
        """
        hop = self.hop_size

        self.near_buffer[:-hop] = self.near_buffer[hop:]
        self.near_buffer[-hop:] = near_end

        self.far_buffer[:-hop] = self.far_buffer[hop:]
        self.far_buffer[-hop:] = far_end

        # FFT of the zero-padded buffer; near_spec is only needed by the main filter, so lightweight (shadow) mode skips it.
        if not getattr(self, '_lightweight', False):
            self.near_spec = np.fft.rfft(self.near_buffer, self.fft_size).astype(np.complex64)
        # far_spec is identical for main and shadow; reuse the caller's when provided.
        if precomputed_far_spec is not None:
            far_spec = precomputed_far_spec
        else:
            far_spec = np.fft.rfft(self.far_buffer, self.fft_size).astype(np.complex64)
        self.far_spec = far_spec  # expose for coherence DTD

        curr_p = self.partition_idx
        self.X_buf[curr_p] = far_spec

        self.echo_spec.fill(0)
        for p in range(self.n_partitions):
            p_idx = (curr_p - p) % self.n_partitions
            self.echo_spec += self.W[p] * self.X_buf[p_idx]

        echo_time = np.fft.irfft(self.echo_spec, self.fft_size).astype(np.float32)

        output = self.near_buffer[-hop:] - echo_time[self.hop_size:self.block_size]

        # Time-domain peak of the echo-predictor output on the valid hop (read by SaturationDetector via AecState).
        self._last_s_max_abs: float = float(
            np.max(np.abs(echo_time[self.hop_size:self.block_size]))
        )

        error_time = np.zeros(self.fft_size, dtype=np.float32)
        error_time[self.hop_size:self.block_size] = output
        self.error_spec = np.fft.rfft(error_time).astype(np.complex64)

        # Windowed error spectrum (near_buffer x sqrt-Hann minus echo_spec) for RES analysis; main filter only.
        if not getattr(self, '_lightweight', False):
            near_win = self.near_buffer[:self.block_size] * self._sqrt_hann_analysis
            near_spec_win = np.fft.rfft(near_win, self.fft_size).astype(np.complex64)
            self.error_spec_windowed = near_spec_win - self.echo_spec

        self._c1c5_trace = {}   # reset each hop; empty when far-inactive or flags OFF
        far_hop_energy = np.sum(far_end ** 2) / hop

        # Initial-state tracking: count active render hops and leave the initial state after the threshold.
        if (self._initial_state_active
                and far_hop_energy > self._initial_state_far_energy_floor):
            self._initial_state_active_render_hops += 1
            if (self._initial_state_active_render_hops
                    >= self._initial_state_threshold_hops):
                self._initial_state_active = False
        self._last_initial_state_active = self._initial_state_active

        if defer_update:
            # Stash the inputs for complete_update(); partition_idx is advanced there.
            self._deferred_update_pending = True
            self._deferred_curr_p = curr_p
            self._deferred_mu_scale = mu_scale
            self._deferred_far_hop_energy = float(far_hop_energy)
            return output.astype(np.float32)

        if far_hop_energy > 1e-4:  # ~ -40 dBFS, unified with far_active threshold
            self._update_weights(curr_p, mu_scale)

        self.partition_idx = (self.partition_idx + 1) % self.n_partitions
        return output.astype(np.float32)

    def complete_update(self, error_override: Optional[np.ndarray] = None) -> None:
        """Apply the W update deferred by ``process(defer_update=True)``; ``error_override`` (complex
        per-bin spec) replaces ``self.error_spec`` as the gradient source."""
        if not self._deferred_update_pending:
            return
        if self._deferred_far_hop_energy > 1e-4:
            self._update_weights(
                self._deferred_curr_p,
                self._deferred_mu_scale,
                error_override=error_override,
            )
        self.partition_idx = (self.partition_idx + 1) % self.n_partitions
        self._deferred_update_pending = False

    def _update_weights(self, curr_p: int, mu_scale,
                        error_override: Optional[np.ndarray] = None):
        """NLMS weight update; ``error_override`` replaces ``self.error_spec`` as the gradient source."""
        mu_scale_arr = np.asarray(mu_scale, dtype=np.float32)
        if mu_scale_arr.ndim == 0:
            mu_scale_arr = np.full(self.n_freqs, float(mu_scale_arr), dtype=np.float32)
        # The call counter advances before all gates, so skipped hops still count toward the startup window.
        self._call_counter += 1
        if not np.any(mu_scale_arr > 0):
            return
        if getattr(self, '_saturated_capture', False):
            self._c1c5_trace['A5_sat_skip'] = True
            return
        # Startup gate: skip while call_counter <= n_partitions or the poor-excitation counter < n_partitions.
        if (self._call_counter <= self.n_partitions
                or self._poor_excitation_counter < self.n_partitions):
            self._c1c5_trace['A3_poor_exc_skip'] = True
            self._c1c5_trace['A3_call_ctr'] = self._call_counter
            self._c1c5_trace['A3_exc_ctr'] = self._poor_excitation_counter
            return
        # Stationary-far gate: skip the update when the block is flagged stationary.
        if getattr(self, '_block_stationary', False):
            return
        # Narrow-band mask applied as a pre-multiplier on mu_scale.
        if self._render_signal_analyzer is not None:
            rsa_mask = np.ones(self.n_freqs, dtype=np.float32)
            self._render_signal_analyzer.mask_regions_around_narrow_bands(rsa_mask)
            mu_scale_arr = (mu_scale_arr * rsa_mask).astype(np.float32)
            self._c1c5_trace['A4_mask_frac'] = float(np.mean(rsa_mask < 0.5))
        # mu denominator: partition-summed X^2 of the current frame (no smoothing).
        x2_partition_sum = (np.abs(self.X_buf) ** 2).sum(axis=0).astype(np.float32)
        denom = x2_partition_sum + self.delta
        self._c1c5_trace['A1_x2_active'] = True
        # The coarse filter learns 0.95/0.7 faster during the initial state.
        mu_initial_boost = (np.float32(0.95 / 0.7)
                             if self._initial_state_active
                             else np.float32(1.0))
        mu_eff = (self.mu * mu_initial_boost * mu_scale_arr) / denom
        # Noise gate: mu[k] = 0 where partition-summed X^2[k] < FILTER_NOISE_GATE_POWER_FLOAT.
        from . import aec3_scale as _aec3_scale
        x2_for_gate = (np.abs(self.X_buf) ** 2).sum(axis=0).astype(np.float32)
        _ng_thr = np.float32(_aec3_scale.FILTER_NOISE_GATE_POWER_FLOAT)
        mu_eff = np.where(
            x2_for_gate >= _ng_thr,
            mu_eff,
            np.float32(0.0),
        ).astype(np.float32)
        self._c1c5_trace['A2_noise_gate_zero_frac'] = float(
            np.mean(x2_for_gate < _ng_thr))
        self._c1c5_trace['mu_eff_mean'] = float(np.mean(mu_eff))
        self._c1c5_trace['mu_eff_max'] = float(np.max(mu_eff))
        # error_override, when given, replaces self.error_spec as the gradient source.
        _err_grad = (
            error_override if error_override is not None else self.error_spec
        )
        for p in range(self.n_partitions):
            p_idx = (curr_p - p) % self.n_partitions
            grad = _err_grad * np.conj(self.X_buf[p_idx])
            self.W[p] += mu_eff * grad
            # Overlap-save constraint: clear the non-causal half (round-robin: only the scheduled partition).
            if self.enable_td_constraint and (
                    not self._constraint_round_robin
                    or p == self._partition_to_constrain):
                w_time = np.fft.irfft(self.W[p], self.fft_size).astype(np.float32)
                w_time *= self._td_window
                self.W[p] = np.fft.rfft(w_time).astype(np.complex64)
        if self.enable_td_constraint and self._constraint_round_robin:
            self._partition_to_constrain = (
                self._partition_to_constrain + 1) % self.n_partitions

    def get_error_energy(self) -> float:
        return float(np.sum(np.abs(self.error_spec) ** 2))

    def get_time_domain_filter(self) -> np.ndarray:
        """Concatenate the partitions into one time-domain impulse response of
        ``n_partitions * hop_size`` samples (first ``hop_size`` samples of each partition's IFFT)."""
        full = np.zeros(self.tap_span, dtype=np.float32)
        for p in range(self.n_partitions):
            w_time = np.fft.irfft(self.W[p], self.fft_size).astype(np.float32)
            full[p * self.hop_size:(p + 1) * self.hop_size] = w_time[:self.hop_size]
        return full

    def copy_weights_from(self, src: 'PBFDAF'):
        self.W[:] = src.W

    def scale_filter(self, scale: float) -> None:
        """Multiply every partition's W by ``scale`` in place."""
        self.W *= np.complex64(scale)


class PBFDKF(PBFDAF):
    """Partitioned-block frequency-domain Kalman filter: PBFDAF with per-bin Kalman gain."""

    def __init__(self, block_size: int, n_partitions: int,
                 mu: float = 0.3, delta: float = 1e-8,
                 hop_size: int = 0, sample_rate: int = 16000):
        super().__init__(block_size, n_partitions, mu, delta, hop_size, sample_rate)

        self.P = np.ones((n_partitions, self.n_freqs), dtype=np.float32) * 0.01
        self.Q_high = np.ones(self.n_freqs, dtype=np.float32) * 1e-4
        self.Q_low  = np.ones(self.n_freqs, dtype=np.float32) * 1e-5
        self.Q = self.Q_high.copy()
        self.R = np.ones(self.n_freqs, dtype=np.float32) * 1e-2
        self._error_psd = np.ones(self.n_freqs, dtype=np.float32) * 1e-2
        # Measurement-noise PSD EMA per hop (reference 0.95 at the 10 ms hop, retimed).
        self._alpha_r = aec3_scale.growth_rehop(
            0.95, 160, 16000, self.hop_size, self.sample_rate)

        # Startup / poor-excitation / saturation gates: while any fires the weight update is skipped.
        # The orchestrator sets ``_saturated_capture`` and ``_poor_excitation_counter``.
        self._call_counter = 0
        # Poor-excitation initial value; the orchestrator overrides it per config.
        self._poor_excitation_counter = 400
        self._saturated_capture = False
        # RenderSignalAnalyzer for the narrow-band mask, set by the orchestrator (None disables masking).
        self._render_signal_analyzer = None

        # H_error per-bin state is the primary Kalman-like state for K; P stays as a parallel field.
        # The orchestrator feeds _e2_coarse_for_refresh, _disallow_leakage_diverged and _erl_per_bin each hop.
        from . import aec3_scale as _aec3_scale
        self.H_error_per_bin = np.full(
            self.n_freqs, _aec3_scale.H_ERROR_INIT_FLOAT, dtype=np.float32
        )
        self._h_error_floor = np.float32(_aec3_scale.H_ERROR_FLOOR_FLOAT)
        self._h_error_ceil = np.float32(_aec3_scale.H_ERROR_CEIL_FLOAT)
        # Steady-state refined leakage per hop (5e-5 converged, 5e-2 diverged per block).
        self._leakage_converged = np.float32(
            _aec3_scale.per_block_rate_to_per_hop(5e-5, self.hop_size, self.sample_rate)
        )
        self._leakage_diverged = np.float32(
            _aec3_scale.per_block_rate_to_per_hop(5e-2, self.hop_size, self.sample_rate)
        )
        # Per-bin instantaneous coarse error PSD published each hop; the scalar sum is the fallback before it exists.
        self._e2_coarse_for_refresh = 0.0
        self._e2_coarse_per_bin: Optional[np.ndarray] = None
        # Hops before switching from single-partition to partition-sum X^2 (0 = partition-sum from the start).
        self._partition_sum_x2_startup_hops: int = 0
        self._disallow_leakage_diverged = False
        # Per-bin ERL, overwritten by the orchestrator once its estimator has a value.
        self._erl_per_bin = np.full(self.n_freqs, 0.1, dtype=np.float32)
        # Cold-start deadlock breaker (0 = off): on hard echo paths W ~ 0 gives no H_error refresh, so mu dies;
        # flooring the refresh ERL keeps mu alive and fades once sum|W|^2 exceeds the floor.
        self._h_error_refresh_erl_floor = np.float32(0.0)

        # Initial profile: the first 2.5 s of active render use aggressive leakage (100x / 10x steady);
        # ``_h_error_refresh`` consults ``_initial_state_active``. The threshold from PBFDAF.__init__ is kept.
        self._initial_state_active: bool = True
        self._initial_state_active_render_hops: int = 0
        self._initial_state_far_energy_floor: float = 1e-4
        self._last_initial_state_active: bool = True

        # Diagnostic buffer read by the orchestrator via getattr; stays empty.
        self._p53_innovation_trace = []

    def handle_echo_path_change(self, delay_change: bool = True,
                                 gain_change: bool = False,
                                 zero_filter: bool = False) -> None:
        """Echo-path-change handling: on a delay change H_error resets to 10000 (high uncertainty, so
        mu starts large and the filter re-tracks); the counters reset via the base class."""
        from . import aec3_scale as _aec3_scale
        super().handle_echo_path_change(
            delay_change=delay_change,
            gain_change=gain_change,
            zero_filter=zero_filter,
        )
        if delay_change:
            self.H_error_per_bin.fill(np.float32(_aec3_scale.H_ERROR_INIT_FLOAT))

    def _rearm_kalman(self):
        """Restart of the Kalman side shared by ``reset()`` and ``reset_taps()``: the measurement-noise
        and error-PSD trackers and the process noise are statistics of the taps; the
        ``_p_max_override`` / ``_p_floor_beta`` countdowns are cleared so a restart does not inherit them."""
        self.P.fill(0.01)
        self.R.fill(1e-2)
        self._error_psd.fill(1e-2)
        self.Q[:] = self.Q_high
        for attr in ('_p_max_override', '_p_max_override_frames',
                     '_p_floor_beta', '_p_floor_beta_frames'):
            try:
                delattr(self, attr)
            except AttributeError:
                pass

    def reset(self):
        super().reset()
        self._rearm_kalman()

    def reset_taps(self):
        super().reset_taps()
        self._rearm_kalman()

    def reset_carried_state(self):
        """``PBFDAF.reset_carried_state`` plus the orchestrator-fed refresh inputs and per-bin H_error."""
        super().reset_carried_state()
        self._e2_coarse_for_refresh = 0.0
        self._e2_coarse_per_bin = None
        self._disallow_leakage_diverged = False
        self.H_error_per_bin.fill(np.float32(aec3_scale.H_ERROR_INIT_FLOAT))
        try:
            delattr(self, '_last_leakage_div_frac')
        except AttributeError:
            pass

    def _update_weights(self, curr_p: int, mu_scale,
                        error_override: Optional[np.ndarray] = None):
        """Frequency-domain Kalman weight update; ``error_override`` replaces ``self.error_spec`` as the gradient source."""
        mu_scale_arr = np.asarray(mu_scale, dtype=np.float32)
        if mu_scale_arr.ndim == 0:
            mu_scale_arr = np.full(self.n_freqs, float(mu_scale_arr), dtype=np.float32)

        # Startup / poor-excitation / saturation gates: tick the call counter and skip the W update,
        # but still run the H_error refresh.
        self._call_counter += 1
        if (self._call_counter <= self.n_partitions
                or self._poor_excitation_counter < self.n_partitions
                or self._saturated_capture):
            self._h_error_refresh()
            return

        # Stationary-far gate: adapting against stationary broadband noise produces spurious
        # mic-as-echo coupling (the narrow-band analyzer does not catch it).
        if getattr(self, '_block_stationary', False):
            self._h_error_refresh()
            return

        # Narrow-band mask: pre-multiplier on mu_scale_arr zeroing mu around sustained tonal peaks.
        if self._render_signal_analyzer is not None:
            rsa_mask = np.ones(self.n_freqs, dtype=np.float32)
            self._render_signal_analyzer.mask_regions_around_narrow_bands(rsa_mask)
            mu_scale_arr = (mu_scale_arr * rsa_mask).astype(np.float32)

        error_psd = np.abs(self.error_spec) ** 2
        self._error_psd = self._alpha_r * self._error_psd + (1 - self._alpha_r) * error_psd
        self.R = np.maximum(self._error_psd, self.delta)

        self._update_weights_aec3(curr_p, mu_scale_arr, error_psd,
                                  error_override=error_override)

    def _update_weights_aec3(self, curr_p: int, mu_scale_arr: np.ndarray,
                              error_psd: np.ndarray,
                              error_override: Optional[np.ndarray] = None) -> None:
        """Per-bin Kalman gain via H_error: ``mu[k] = H_error[k] / (0.5 H_error[k] X^2[k] + n_partitions E^2_refined[k])``
        applied to all partitions with their conj(X). Then ``H_error[k] -= 0.5 mu[k] X^2[k] H_error[k]``,
        and the always-on refresh (``_h_error_refresh``)."""
        # X^2 is the partition-summed render power.
        X_latest = self.X_buf[curr_p]
        # Hybrid startup: single-partition X^2 for the first ``_partition_sum_x2_startup_hops`` calls, then partition-sum.
        if self._call_counter > self._partition_sum_x2_startup_hops:
            X2 = (np.abs(self.X_buf) ** 2).sum(axis=0).astype(np.float32)
        else:
            X2 = (np.abs(X_latest) ** 2).astype(np.float32)
        delta32 = np.float32(self.delta)
        n_part = np.float32(self.n_partitions)
        # E^2 is the current-block |error_spec|^2.
        e2_refined_current = (np.abs(self.error_spec) ** 2).astype(np.float32)
        denom_aec3 = (
            np.float32(0.5) * self.H_error_per_bin * X2
            + n_part * e2_refined_current
            + delta32
        )
        mu_aec3 = (self.H_error_per_bin / denom_aec3).astype(np.float32)

        # Noise gate: mu = 0 where X^2 < FILTER_NOISE_GATE_POWER_FLOAT.
        from . import aec3_scale as _aec3_scale
        _noise_gate = np.float32(_aec3_scale.FILTER_NOISE_GATE_POWER_FLOAT)
        mu_aec3 = np.where(X2 >= _noise_gate, mu_aec3, np.float32(0.0))

        # Per-partition W update with the shared gain; error_override replaces self.error_spec when given.
        _err_grad = (
            error_override if error_override is not None else self.error_spec
        )
        for p in range(self.n_partitions):
            p_idx = (curr_p - p) % self.n_partitions
            X = self.X_buf[p_idx]
            K = mu_aec3 * np.conj(X)             # per-bin AEC3 K
            K_scaled = K * mu_scale_arr          # apply DT scale
            self.W[p] += K_scaled * _err_grad
            # Overlap-save constraint: clear the non-causal half (round-robin: only the scheduled partition).
            if self.enable_td_constraint and (
                    not self._constraint_round_robin
                    or p == self._partition_to_constrain):
                w_time = np.fft.irfft(self.W[p], self.fft_size).astype(np.float32)
                w_time *= self._td_window
                self.W[p] = np.fft.rfft(w_time).astype(np.complex64)
        if self.enable_td_constraint and self._constraint_round_robin:
            self._partition_to_constrain = (
                self._partition_to_constrain + 1) % self.n_partitions

        # H_error decay only when W was adapted, scaled by mu_scale_arr so masked bins do not decay while W is frozen.
        mu_aec3 *= mu_scale_arr
        self.H_error_per_bin -= (
            np.float32(0.5) * mu_aec3 * X2 * self.H_error_per_bin
        )

        self._h_error_refresh()

    def _h_error_refresh(self) -> None:
        """H_error leakage refresh and clamp; runs every hop, including gated hops.

        Uses the per-bin instantaneous refined vs coarse error comparison once the orchestrator
        publishes ``_e2_coarse_per_bin``, else the scalar sums.
        """
        # Initial state: refined_initial leakage (5e-3 converged, 5e-1 diverged per block), else steady-state values.
        if self._initial_state_active:
            from . import aec3_scale as _aec3_scale
            _lc_eff = np.float32(
                _aec3_scale.per_block_rate_to_per_hop(5e-3, self.hop_size, self.sample_rate)
            )
            _ld_eff = np.float32(
                _aec3_scale.per_block_rate_to_per_hop(5e-1, self.hop_size, self.sample_rate)
            )
        else:
            _lc_eff = self._leakage_converged
            _ld_eff = self._leakage_diverged

        if self._e2_coarse_per_bin is not None:
            # Per-bin path: fresh |error_spec|^2 against the coarse per-bin error.
            e2_refined_per_bin = (np.abs(self.error_spec) ** 2).astype(np.float32)
            use_converged_mask = (
                e2_refined_per_bin <= self._e2_coarse_per_bin
            ) | self._disallow_leakage_diverged
            # Diagnostic: fraction of bins on the diverged-leakage branch.
            self._last_leakage_div_frac = float(np.mean(~use_converged_mask))
            leakage_arr = np.where(
                use_converged_mask,
                _lc_eff,
                _ld_eff,
            ).astype(np.float32)
            _erl_eff = (
                np.maximum(self._erl_per_bin, self._h_error_refresh_erl_floor)
                if self._h_error_refresh_erl_floor > 0.0
                else self._erl_per_bin
            )
            self.H_error_per_bin = (
                self.H_error_per_bin + leakage_arr * _erl_eff
            )
        else:
            e2_ref_sum = float(np.sum(self._error_psd))
            e2_coa_sum = float(self._e2_coarse_for_refresh)
            use_converged = (
                e2_ref_sum <= e2_coa_sum or self._disallow_leakage_diverged
            )
            # Diagnostic: all-or-nothing on the scalar path.
            self._last_leakage_div_frac = 0.0 if use_converged else 1.0
            leakage = _lc_eff if use_converged else _ld_eff
            self.H_error_per_bin = (
                self.H_error_per_bin + leakage * self._erl_per_bin
            )
        np.clip(self.H_error_per_bin, self._h_error_floor, self._h_error_ceil,
                out=self.H_error_per_bin)

    def copy_weights_from(self, src: 'PBFDAF'):
        self.W[:] = src.W
        # Copy W only: P/Q/R are confidence states of each filter's own history.
        # within a few frames.

    def scale_filter(self, scale: float) -> None:
        """Multiplicative W rescale (the Kalman gain self-corrects within a few frames)."""
        super().scale_filter(scale)
