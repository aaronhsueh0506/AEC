"""Adaptive filter primitives.

Extracted from ``aec.py`` during refactor R.5. Two filter classes:

* ``PBFDAF`` — Partitioned Block FDAF (NLMS adaptation in freq domain)
* ``PBFDKF`` — Partitioned Block FDKF (Kalman extension of PBFDAF)

Self-contained: depends only on numpy + collections.deque + typing.
"""
from typing import Optional, Tuple

import numpy as np


class PBFDAF:
    """
    Partitioned Block Frequency-Domain Adaptive Filter (PBFDAF)

    NLMS-based adaptive filter using overlap-save for linear convolution.
    For Kalman-based adaptation, use PBFDKF subclass.
    """

    def __init__(self, block_size: int, n_partitions: int,
                 mu: float = 0.3, delta: float = 1e-8,
                 hop_size: int = 0):
        # Overlap-save: block_size = 2 × hop (proper 50% ratio for TD constraint)
        # FFT zero-pads to next power of 2 if block_size isn't one
        self.hop_size = hop_size if hop_size > 0 else block_size // 2
        self.block_size = 2 * self.hop_size  # overlap-save buffer (exactly 2× hop)
        self.fft_size = 1 << (self.block_size - 1).bit_length()  # next pow2
        self.n_partitions = n_partitions
        self.n_freqs = self.fft_size // 2 + 1
        self.mu = mu
        self.delta = delta
        self.alpha_power = 0.9
        self.enable_td_constraint = True  # can be disabled for diagnosis

        # Time-domain constraint window: clean 50% truncation
        # block_size = 2×hop → truncation at 50%, minimal Gibbs ringing
        self._td_window = np.ones(self.fft_size, dtype=np.float32)
        fade_len = self.hop_size // 4  # 40 samples for hop=160
        fade = 0.5 * (1.0 - np.cos(np.pi * np.arange(fade_len) / fade_len))
        self._td_window[self.hop_size - fade_len:self.hop_size] = fade[::-1].astype(np.float32)
        self._td_window[self.hop_size:] = 0.0

        # Filter weights [n_partitions, n_freqs]
        self.W = np.zeros((n_partitions, self.n_freqs), dtype=np.complex64)

        # Reference spectrum history [n_partitions, n_freqs]
        self.X_buf = np.zeros((n_partitions, self.n_freqs), dtype=np.complex64)
        self.partition_idx = 0

        # Input buffers (block_size = 2 × hop for overlap-save)
        self.near_buffer = np.zeros(self.block_size, dtype=np.float32)
        self.far_buffer = np.zeros(self.block_size, dtype=np.float32)

        # Power estimation
        self.power = np.zeros(self.n_freqs, dtype=np.float32)

        # Output spectra (for RES / coherence DTD)
        self.near_spec = np.zeros(self.n_freqs, dtype=np.complex64)
        self.echo_spec = np.zeros(self.n_freqs, dtype=np.complex64)
        self.error_spec = np.zeros(self.n_freqs, dtype=np.complex64)
        self.far_spec = np.zeros(self.n_freqs, dtype=np.complex64)

        # Windowed error spectrum for RES analysis (sqrt-Hann, same variance
        # as OLA spec but time-aligned with far_spec/echo_spec for coherence).
        # P0.4: canonical periodic sqrt-Hann (denom = N), MATCHING the synthesis
        # window (orchestrator `_aec3_synth_window`). analysis × synthesis =
        # periodic Hann, which sums to 1 across 50%-overlap hops → true perfect
        # reconstruction (verified max|OLA-gain−1|=4e-16). numpy.hanning uses
        # denom N−1, which left ~0.25% OLA gain drift at frame boundaries (an
        # analysis/synthesis window mismatch — the synthesis side was already
        # canonical; this aligns the analysis side).
        _idx = np.arange(self.block_size, dtype=np.float64)
        self._sqrt_hann_analysis = np.sqrt(
            0.5 * (1.0 - np.cos(2.0 * np.pi * _idx / float(self.block_size)))
        ).astype(np.float32)
        self.error_spec_windowed = np.zeros(self.n_freqs, dtype=np.complex64)

        # AEC3 CoarseFilterUpdateGain protection (coarse_filter_update_gain.cc
        # :34-82) is applied unconditionally in _update_weights: partition-summed
        # X² mu denom + noise gate (cc:64-72), poor-excitation/startup gate
        # (cc:51-61), narrowband mask (cc:75), saturation gate (cc:56-57).
        # Shared shadow-gate supporting state (orchestrator overrides on
        # shadow construction). Defined on PBFDAF so subclass PBFDKF
        # inherits the same state attributes; PBFDKF __init__ re-assigns
        # these to its own AEC3-derived defaults.
        self._call_counter = 0
        self._poor_excitation_counter = 0  # safe init; orchestrator sets to hop-scaled 1000-block default
        self._render_signal_analyzer = None
        self._saturated_capture = False
        # Variant H Gate 0 — per-frame C1-C5 gate-fire trace. Empty dict when all
        # C1-C5 flags are OFF (byte-equal preserved). Reset each hop in process();
        # populated in _update_weights only when a C1-C5 flag is ON. Read-only by
        # orchestrator _hf_chain_trace; never affects audio output.
        self._c1c5_trace: dict = {}
        # Gap C (poor_coarse rescue copy E_refined override) — deferred-update
        # support. When process(..., defer_update=True) is called, the FFT /
        # echo / error_spec computation runs as usual but the W update +
        # partition_idx advance are deferred until complete_update() is called.
        # This lets the orchestrator inject E_refined as the gradient source on
        # the rescue fire hop (AEC3 subtractor.cc:302-304). Default-OFF preserves
        # byte-equal: process() runs the W update inline and these flags stay
        # False / unread.
        self._deferred_update_pending: bool = False
        self._deferred_curr_p: int = 0
        self._deferred_mu_scale = 1.0
        self._deferred_far_hop_energy: float = 0.0

        # AEC3 InitialState (aec_state.cc:336-353) — counted in process() so
        # both PBFDAF (shadow) and PBFDKF (refined, subclass) share the
        # counter. Only PBFDKF._h_error_refresh consumes _initial_state_active
        # to switch leakage source. PBFDAF carries the attrs as benign state.
        self._initial_state_active: bool = True
        self._initial_state_active_render_hops: int = 0
        self._initial_state_threshold_hops: int = 250
        self._initial_state_far_energy_floor: float = 1e-4
        self._last_initial_state_active: bool = True

    def reset(self):
        self.W.fill(0)
        self.X_buf.fill(0)
        self.near_buffer.fill(0)
        self.far_buffer.fill(0)
        self.power.fill(0)
        self.partition_idx = 0
        self.error_spec_windowed.fill(0)

    def zero_filter_partitions(self, old_size: int, new_size: int) -> None:
        """AEC3-strict port of ``AdaptiveFirFilter::ZeroFilter(old, new, &H_)``
        (adaptive_fir_filter.cc:460-472).

        Zeroes filter partitions in the half-open range [old_size, new_size).
        In AEC3 production where ``current_size == max_size == 13`` (steady
        state), the only call site is ``HandleEchoPathChange()`` which passes
        ``(current_size, max_size)`` — i.e. (13, 13) — making this a no-op.
        The non-trivial behaviour only fires during initial partition-size
        growth (12→13) or after ``SetSizePartitions`` shrinks ``current_size``.

        We never grow partitions in v3.21.x (fixed 13-partition filter), so
        this is a documented strict-port surface that is never expected to
        zero any partition during a single-session run. Provided for AEC3
        contract alignment and for future architecture work where partition
        count may become dynamic.
        """
        old = max(0, int(old_size))
        new = min(int(new_size), self.W.shape[0])
        if old >= new:
            return
        self.W[old:new].fill(0)

    def handle_echo_path_change(self, delay_change: bool = True,
                                  gain_change: bool = False,
                                  zero_filter: bool = False) -> None:
        """Port AEC3 CoarseFilterUpdateGain::HandleEchoPathChange counter reset
        (M3: not-gain_change → poor_excitation_counter = INITIAL + call_counter = 0).
        AEC3 Subtractor::HandleEchoPathChange dispatches to both refined + coarse.

        NOTE on zero_filter (non-AEC3 ablation path):
        AEC3 ``AdaptiveFirFilter::HandleEchoPathChange`` calls
        ``ZeroFilter(current_size_partitions_, max_size_partitions_, &H_)``,
        which in steady state (current=max=13) is a NO-OP — W is fully
        preserved across delay events. ``W.fill(0)`` here clears ALL
        partitions, which is strictly more aggressive than AEC3.
        ``zero_filter`` defaults to False at every orchestrator call site,
        so this branch is dormant in production. The AEC3-strict semantics
        live in ``zero_filter_partitions()`` above; the legacy aggressive
        path is retained behind ``zero_filter=True`` for ablation only.
        """
        from . import aec3_scale as _aec3_scale
        if delay_change and zero_filter:
            self.W.fill(0)
        if not gain_change:
            self._poor_excitation_counter = int(
                _aec3_scale.POOR_EXCITATION_COUNTER_INITIAL_HOPS_DEFAULT
            )
            self._call_counter = 0

    def process(self, near_end: np.ndarray, far_end: np.ndarray,
                mu_scale=1.0, defer_update: bool = False) -> np.ndarray:
        """Process hop_size samples. mu_scale: scalar or per-bin array [n_freqs].

        defer_update: when True, skip the W update + partition_idx advance and
            stash the inputs so the caller can drive both via complete_update()
            with an optional error_override. Used by the orchestrator's Gap C
            wiring (AEC3 poor_coarse rescue copy uses E_refined for the same-hop
            coarse update; subtractor.cc:302-304). Default-OFF preserves
            byte-equal vs the inline path.
        """
        hop = self.hop_size

        # Shift buffers (overlap-save)
        self.near_buffer[:-hop] = self.near_buffer[hop:]
        self.near_buffer[-hop:] = near_end

        self.far_buffer[:-hop] = self.far_buffer[hop:]
        self.far_buffer[-hop:] = far_end

        # FFT (zero-pad block_size buffer to fft_size, cast to float32 precision)
        near_spec = np.fft.rfft(self.near_buffer, self.fft_size).astype(np.complex64)
        far_spec = np.fft.rfft(self.far_buffer, self.fft_size).astype(np.complex64)
        self.near_spec = near_spec  # expose for RES overlap-save
        self.far_spec = far_spec  # expose for coherence DTD

        # Store far-end spectrum
        curr_p = self.partition_idx
        self.X_buf[curr_p] = far_spec

        # Update power estimate (cold start: initialize directly on first active frame)
        far_psd = np.abs(far_spec) ** 2
        if np.sum(self.power) < 1e-10 and np.sum(far_psd) > 1e-10:
            self.power = far_psd.astype(np.float32)
        else:
            self.power = (self.alpha_power * self.power +
                         (1 - self.alpha_power) * far_psd)

        # Compute echo estimate
        self.echo_spec.fill(0)
        for p in range(self.n_partitions):
            p_idx = (curr_p - p) % self.n_partitions
            self.echo_spec += self.W[p] * self.X_buf[p_idx]

        # IFFT (fft_size → take block_size valid samples)
        echo_time = np.fft.irfft(self.echo_spec, self.fft_size).astype(np.float32)

        # Error (take valid region for output)
        output = self.near_buffer[-hop:] - echo_time[self.hop_size:self.block_size]

        # AEC3 SubtractorOutput.s_*_max_abs (subtractor_output.cc).
        # Time-domain peak of the echo-predictor output on the valid hop —
        # consumed by SaturationDetector via AecState.update. Computed per
        # hop regardless of flags; the cost is one np.max on hop_size floats.
        self._last_s_max_abs: float = float(
            np.max(np.abs(echo_time[self.hop_size:self.block_size]))
        )

        # Error spectrum (zero-pad to fft_size, valid region at [hop, block_size))
        error_time = np.zeros(self.fft_size, dtype=np.float32)
        error_time[self.hop_size:self.block_size] = output
        self.error_spec = np.fft.rfft(error_time).astype(np.complex64)

        # Windowed error spec for RES analysis: near_buffer × sqrt-Hann − echo_spec.
        # Same time alignment as far_spec/echo_spec, but with sqrt-Hann variance
        # (low inter-frame noise). Used for coherence/PSD/ENR in ResFilter.
        near_win = self.near_buffer[:self.block_size] * self._sqrt_hann_analysis
        near_spec_win = np.fft.rfft(near_win, self.fft_size).astype(np.complex64)
        self.error_spec_windowed = near_spec_win - self.echo_spec

        # Update weights — gate on far-end activity
        self._c1c5_trace = {}   # reset each hop; empty when far-inactive or flags OFF
        far_hop_energy = np.sum(far_end ** 2) / hop

        # AEC3 InitialState tracking: count active render hops, exit initial
        # state after 250 active hops (2.5 s @ hop=160/sr=16000).
        if (self._initial_state_active
                and far_hop_energy > self._initial_state_far_energy_floor):
            self._initial_state_active_render_hops += 1
            if (self._initial_state_active_render_hops
                    >= self._initial_state_threshold_hops):
                self._initial_state_active = False
        self._last_initial_state_active = self._initial_state_active

        if defer_update:
            # Gap C wiring: stash the inputs so complete_update() can apply the
            # W update (optionally with an error_override) and then advance
            # partition_idx. partition_idx is NOT advanced here — keep the
            # update window aligned with curr_p across both phases.
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
        """Drive the W update that was deferred by process(defer_update=True).

        error_override: optional complex per-bin spec (shape [n_freqs]) to use
            as the gradient source instead of self.error_spec. Wires AEC3
            poor_coarse rescue copy's same-hop E_refined coarse update path
            (subtractor.cc:302-304). When None, behaves identically to the
            inline update in process().
        """
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
        """NLMS weight update.

        error_override: optional complex per-bin spec (shape [n_freqs]) that
            replaces self.error_spec as the gradient source.
        """
        mu_scale_arr = np.asarray(mu_scale, dtype=np.float32)
        if mu_scale_arr.ndim == 0:
            mu_scale_arr = np.full(self.n_freqs, float(mu_scale_arr), dtype=np.float32)
        if not np.any(mu_scale_arr > 0):
            return
        # AEC3 coarse_filter_update_gain.cc:56-57 saturation gate.
        # `saturated_capture_signal → return`.
        if getattr(self, '_saturated_capture', False):
            self._c1c5_trace['A5_sat_skip'] = True
            return
        # AEC3 coarse_filter_update_gain.cc:51-61 call_counter +
        # poor_excitation startup gate. `if poor_signal_excitation_counter_ <
        # size_partitions OR call_counter_ <= size_partitions → return`.
        self._call_counter += 1
        if (self._call_counter <= self.n_partitions
                or self._poor_excitation_counter < self.n_partitions):
            self._c1c5_trace['A3_poor_exc_skip'] = True
            self._c1c5_trace['A3_call_ctr'] = self._call_counter
            self._c1c5_trace['A3_exc_ctr'] = self._poor_excitation_counter
            return
        # Stationary-far gate (see PBFDKF version for full rationale). Shadow
        # NLMS also skips W update when the StationarityEstimator flags the
        # block as stationary; spurious mic-as-echo coupling from broadband
        # stationary noise damages nearend equally in NLMS path.
        if getattr(self, '_block_stationary', False):
            return
        # AEC3 coarse_filter_update_gain.cc:75
        # `render_signal_analyzer.MaskRegionsAroundNarrowBands(&mu)`. Apply
        # the narrowband mask as a pre-multiplier on mu_scale_arr so any
        # subsequent mu_eff = mu * mu_scale_arr inherits the mask.
        if self._render_signal_analyzer is not None:
            rsa_mask = np.ones(self.n_freqs, dtype=np.float32)
            self._render_signal_analyzer.mask_regions_around_narrow_bands(rsa_mask)
            mu_scale_arr = (mu_scale_arr * rsa_mask).astype(np.float32)
            self._c1c5_trace['A4_mask_frac'] = float(np.mean(rsa_mask < 0.5))
        # AEC3 coarse_filter_update_gain.cc:64-72 mu denominator uses
        # partition-summed X² = Σ_p |X_buf[p]|² from the current frame
        # (no smoothing). Matches AEC3 SpectralSum source semantic.
        x2_partition_sum = (np.abs(self.X_buf) ** 2).sum(axis=0).astype(np.float32)
        denom = x2_partition_sum + self.delta
        self._c1c5_trace['A1_x2_active'] = True
        # AEC3 coarse_initial.rate = 0.95 vs coarse.rate = 0.7 — 35% faster
        # learning during the first 2.5 s of active render. Boost self.mu by
        # the same ratio while _initial_state_active is True so URO has a
        # converged coarse path to route to at startup.
        # Source: echo_canceller3_config.h:108 (coarse_initial) +
        # echo_canceller3_config.cc:295 (coarse_initial.rate = 0.95f).
        mu_initial_boost = (np.float32(0.95 / 0.7)
                             if self._initial_state_active
                             else np.float32(1.0))
        mu_eff = (self.mu * mu_initial_boost * mu_scale_arr) / denom
        # AEC3 coarse_filter_update_gain.cc:67-71 noise_gate hard zero.
        # AEC3 sets `mu[k] = 0` where `X²[k] < noise_gate`. X² source =
        # SpectralSum. Uses FILTER_NOISE_GATE_POWER_FLOAT (20075344 int16²
        # = 0.01870) — the AEC3 coarse filter gate constant from
        # echo_canceller3_config.cc:99 (the suppression-path
        # NOISE_GATE_POWER_FLOAT 27509562 = 0.02562 is a different constant).
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
        # Record effective mu distribution for sub-ladder attribution trace.
        self._c1c5_trace['mu_eff_mean'] = float(np.mean(mu_eff))
        self._c1c5_trace['mu_eff_max'] = float(np.max(mu_eff))
        # Gap C: error_override (when provided) substitutes for self.error_spec
        # as the gradient source. AEC3 subtractor.cc:302-304 — on poor_coarse
        # rescue fire, coarse update consumes E_refined for the fire hop.
        _err_grad = (
            error_override if error_override is not None else self.error_spec
        )
        for p in range(self.n_partitions):
            p_idx = (curr_p - p) % self.n_partitions
            grad = _err_grad * np.conj(self.X_buf[p_idx])
            self.W[p] += mu_eff * grad
            # Time-domain constraint: fade out non-causal part (raised cosine)
            if self.enable_td_constraint:
                w_time = np.fft.irfft(self.W[p], self.fft_size).astype(np.float32)
                w_time *= self._td_window
                self.W[p] = np.fft.rfft(w_time).astype(np.complex64)

    def get_error_energy(self) -> float:
        return float(np.sum(np.abs(self.error_spec) ** 2))

    def get_time_domain_filter(self) -> np.ndarray:
        """Concatenate partitions to a single time-domain impulse response.

        Each PBFDAF partition stores the IR contribution at frame-lag ``p``;
        IFFT'ing ``W[p]`` and taking the first ``hop_size`` samples gives
        that partition's TD slice. Concatenated length is
        ``n_partitions * hop_size`` samples (covers the full adaptive
        filter span).

        Consumed by AEC3-aligned FilterAnalyzer for peak/consistency
        analysis.
        """
        full = np.zeros(self.n_partitions * self.hop_size, dtype=np.float32)
        for p in range(self.n_partitions):
            w_time = np.fft.irfft(self.W[p], self.fft_size).astype(np.float32)
            full[p * self.hop_size:(p + 1) * self.hop_size] = w_time[:self.hop_size]
        return full

    def copy_weights_from(self, src: 'PBFDAF'):
        self.W[:] = src.W

    def scale_filter(self, scale: float) -> None:
        """AEC3-aligned multiplicative W rescale.

        Multiplies every partition's W by `scale` in-place. Mirrors AEC3
        subtractor.cc ScaleFilter action used by FilterMisadjustmentEstimator
        to correct long-term W magnitude drift. PBFDKF overrides to
        optionally rescale Kalman P as well.
        """
        self.W *= np.complex64(scale)


class PBFDKF(PBFDAF):
    """
    Partitioned Block Frequency-Domain Kalman Filter (PBFDKF)

    Extends PBFDAF with per-bin Kalman gain for faster convergence
    and automatic step-size adaptation.
    """

    def __init__(self, block_size: int, n_partitions: int,
                 mu: float = 0.3, delta: float = 1e-8,
                 hop_size: int = 0):
        super().__init__(block_size, n_partitions, mu, delta, hop_size)

        # P: error covariance (real, per-partition per-bin)
        self.P = np.ones((n_partitions, self.n_freqs), dtype=np.float32) * 0.01
        # Q: process noise — controls adaptation speed (two-stage)
        self.Q_high = np.ones(self.n_freqs, dtype=np.float32) * 1e-4
        self.Q_low  = np.ones(self.n_freqs, dtype=np.float32) * 1e-5
        self.Q = self.Q_high.copy()
        # R: measurement noise PSD (estimated from error)
        self.R = np.ones(self.n_freqs, dtype=np.float32) * 1e-2
        self._error_psd = np.ones(self.n_freqs, dtype=np.float32) * 1e-2
        self._alpha_r = 0.95   # faster R tracking for DT protection

        # AEC3 startup / poor-excitation / saturation gates
        # (refined_filter_update_gain.cc:96-99). Orchestrator sets
        # `_saturated_capture` from its saturation detector and
        # `_poor_excitation_counter` from RenderSignalAnalyzer; we maintain
        # `_call_counter` here. While ANY of the gates fires, the weight
        # update is skipped (no G applied; H_error / P state still evolves
        # via the existing decay path).
        self._call_counter = 0
        # Initialised to POOR_EXCITATION_COUNTER_INITIAL_HOPS via aec3_scale;
        # orchestrator overrides per-config hop_size at construction time.
        self._poor_excitation_counter = 400
        self._saturated_capture = False
        # RenderSignalAnalyzer for per-bin narrow-band mask.
        # Set externally by orchestrator; None disables masking.
        self._render_signal_analyzer = None

        # AEC3 H_error per-bin state + leakage refresh. H_error replaces our
        # P matrix as the primary Kalman-like state in the K compute
        # (refined_filter_update_gain.cc:104-138). P stays as a parallel
        # field for backwards-compat (PathChangeRegimeHandler overrides /
        # diagnostic), but the K applied to W comes from H_error.
        # Orchestrator sets _e2_coarse_for_refresh, _disallow_leakage_diverged,
        # _erl_per_bin per hop to feed the always-on refresh formula.
        from . import aec3_scale as _aec3_scale
        self.H_error_per_bin = np.full(
            self.n_freqs, _aec3_scale.H_ERROR_INIT_FLOAT, dtype=np.float32
        )
        self._h_error_floor = np.float32(_aec3_scale.H_ERROR_FLOOR_FLOAT)
        self._h_error_ceil = np.float32(_aec3_scale.H_ERROR_CEIL_FLOAT)
        self._leakage_converged = np.float32(
            _aec3_scale.LEAKAGE_CONVERGED_PER_HOP_DEFAULT
        )
        self._leakage_diverged = np.float32(
            _aec3_scale.LEAKAGE_DIVERGED_PER_HOP_DEFAULT
        )
        # Updated per hop by orchestrator. Per-bin `_e2_coarse_per_bin` is
        # the instantaneous coarse error PSD published per hop, consumed by
        # the AEC3 cc:128-138 per-bin path. Scalar `_e2_coarse_for_refresh`
        # is the legacy sum kept for backwards-compat.
        self._e2_coarse_for_refresh = 0.0
        self._e2_coarse_per_bin: Optional[np.ndarray] = None
        # Hybrid-startup threshold: hops before switching single-partition →
        # partition-sum X² in _update_weights_aec3 (0 = partition-sum from the
        # first active-update hop).
        self._partition_sum_x2_startup_hops: int = 0
        self._disallow_leakage_diverged = False
        # ERL per bin (lazy init to 0.1 = -10 dB nominal; orchestrator
        # overwrites once its ERL estimator has a real value).
        self._erl_per_bin = np.full(self.n_freqs, 0.1, dtype=np.float32)
        # Cold-start DEADLOCK breaker (default 0 = OFF, byte-equal preserved).
        # The H_error leakage refresh is `H_error += leakage × erl`, where
        # erl = Σ_p|W_p|² (the filter's own weight energy). On hard echo paths
        # the filter can't bootstrap: W≈0 → refresh≈0 → H_error decays to floor
        # → mu dies → W stays ≈0 (self-reinforcing). Flooring the refresh erl
        # keeps a minimum refresh so mu survives cold-start; self-fades once
        # Σ|W|² grows past the floor (no effect on already-adapted bins).
        self._h_error_refresh_erl_floor = np.float32(0.0)

        # AEC3 refined_initial profile — first 2.5 s of active render uses
        # aggressive leakage (100×/10× steady) so the filter converges fast
        # at session start. Source: AecState::InitialState (aec_state.cc:336-353)
        # + FilterConfig refined_initial (echo_canceller3_config.h:102-113).
        # AEC3 threshold: 2.5 s × kNumBlocksPerSecond (250) = 625 active blocks.
        # hop=160/sr=16000 equivalent: 2.5 s × 100 hops/s = 250 active hops.
        # _h_error_refresh() consults _initial_state_active to pick the leakage
        # source. Counter only increments on active far render (energy gate
        # = 1e-4, same threshold the W update uses).
        self._initial_state_active: bool = True
        self._initial_state_active_render_hops: int = 0
        self._initial_state_threshold_hops: int = 250
        self._initial_state_far_energy_floor: float = 1e-4
        # Diag — last frame's initial_state_active value (for trace).
        self._last_initial_state_active: bool = True

        # P53 innovation-audit buffer — read by the orchestrator P53 diag via
        # getattr; its only populator was the retired legacy Kalman path, so it
        # now stays empty (kept for the diag's getattr contract).
        self._p53_innovation_trace = []

    def handle_echo_path_change(self, delay_change: bool = True,
                                 gain_change: bool = False,
                                 zero_filter: bool = False) -> None:
        """Port AEC3 RefinedFilterUpdateGain::HandleEchoPathChange
        (refined_filter_update_gain.cc:53-68): H_error reset to kHErrorInitial=10000
        on delay_change, counter reset via super() (CoarseFilterUpdateGain M3).

        NOTE: super() calls W.fill(0) only when zero_filter=True. That path is a
        default-OFF ablation, NOT AEC3 parity — AEC3 ZeroFilter(current=13, max=13)
        is a steady-state no-op (zeroes partitions in [current..max) = empty set).

        - delay_change: H_error = kHErrorInitial (10000) — high uncertainty
          so mu starts large; filter aggressively re-tracks new path.

        Without H_error reset, post-EPC the filter retains stale H_error
        (small, post-convergence) and full partition-sum X² mu denominator
        → mu stays small → can't re-track movement → DT damage accumulates.
        """
        from . import aec3_scale as _aec3_scale
        super().handle_echo_path_change(
            delay_change=delay_change,
            gain_change=gain_change,
            zero_filter=zero_filter,
        )
        if delay_change:
            self.H_error_per_bin.fill(np.float32(_aec3_scale.H_ERROR_INIT_FLOAT))

    def reset(self):
        super().reset()
        self.P.fill(0.01)
        self.R.fill(1e-2)
        self._error_psd.fill(1e-2)
        self.Q[:] = self.Q_high
        # B1 fix: unconditional cleanup of dynamic P-override attrs.
        # Using try/except is safer than hasattr+delattr: if reset() is called
        # mid-countdown the _frames attr exists and is deleted; if called when
        # only the base attr remains (countdown just expired) it is also deleted;
        # if called on a freshly-constructed filter (no attr) it silently skips.
        # This prevents any case where a second reset() re-inherits a stale
        # countdown that the first reset() partially cleared.
        for attr in ('_p_max_override', '_p_max_override_frames',
                     '_p_floor_beta', '_p_floor_beta_frames'):
            try:
                delattr(self, attr)
            except AttributeError:
                pass

    def _update_weights(self, curr_p: int, mu_scale,
                        error_override: Optional[np.ndarray] = None):
        """Frequency-Domain Kalman Filter weight update.

        error_override: optional per-bin complex spec replacing self.error_spec
            as the gradient source. Default None preserves byte-equal. Used by
            Gap C wiring when PBFDKF happens to be the shadow class (rare;
            production shadow is PBFDAF).
        """
        mu_scale_arr = np.asarray(mu_scale, dtype=np.float32)
        if mu_scale_arr.ndim == 0:
            mu_scale_arr = np.full(self.n_freqs, float(mu_scale_arr), dtype=np.float32)

        # AEC3 startup / poor-excitation / saturation gates
        # (refined_filter_update_gain.cc:96-99). Tick the call counter and
        # short-circuit out of the W update when any gate fires. R / Q / P
        # accounting still happens via the cold-init values; the filter just
        # doesn't accumulate gradient updates during the protected window.
        # NOTE: when gates fire, AEC3 still runs the H_error REFRESH at
        # the bottom of Compute() (lines 128-138). We mirror that by calling
        # `_h_error_refresh()` before any early return so H_error stays
        # in steady state during the gated window.
        self._call_counter += 1
        if (self._call_counter <= self.n_partitions
                or self._poor_excitation_counter < self.n_partitions
                or self._saturated_capture):
            self._h_error_refresh()
            return

        # Stationary-far gate. When the render-path StationarityEstimator
        # flags the current block as stationary (constant background hum /
        # fan / line noise), any filter adaptation against it produces
        # spurious mic-as-echo coupling because the noise has no causal
        # correlation with the nearend signal. RSA's `poor_signal_excitation`
        # covers tonal peaks but not broadband stationary noise (RSA
        # narrow-band counter 0% on E0l0 hum case); StationarityEstimator
        # catches that gap. When set, the orchestrator pushes
        # `_block_stationary = True` before `_update_weights` runs.
        if getattr(self, '_block_stationary', False):
            self._h_error_refresh()
            return

        # RenderSignalAnalyzer narrow-band mask. Zeros mu for ±2 bins
        # around any frequency that has sustained > 5 frames of tonal
        # X²[k] > 3 × max(neighbors) condition. Mask is applied as a
        # pre-multiplier on mu_scale_arr so the existing
        # K_scaled = K_optimal × mu_scale_arr path naturally zeroes
        # those bins' W update.
        if self._render_signal_analyzer is not None:
            rsa_mask = np.ones(self.n_freqs, dtype=np.float32)
            self._render_signal_analyzer.mask_regions_around_narrow_bands(rsa_mask)
            mu_scale_arr = (mu_scale_arr * rsa_mask).astype(np.float32)

        # Update measurement noise estimate from error PSD
        error_psd = np.abs(self.error_spec) ** 2
        self._error_psd = self._alpha_r * self._error_psd + (1 - self._alpha_r) * error_psd
        self.R = np.maximum(self._error_psd, self.delta)

        # AEC3 H_error path: per-bin Kalman gain via H_error (not the legacy
        # partition-summed P denominator). Mirrors refined_filter_update_gain.cc
        # :104-138. (The legacy P-denominator Kalman body it replaced was a
        # default-OFF path — retired with the v3.21 close.)
        self._update_weights_aec3(curr_p, mu_scale_arr, error_psd,
                                  error_override=error_override)

    def _update_weights_aec3(self, curr_p: int, mu_scale_arr: np.ndarray,
                              error_psd: np.ndarray,
                              error_override: Optional[np.ndarray] = None) -> None:
        """AEC3-aligned per-bin Kalman gain via H_error.

        Mirrors RefinedFilterUpdateGain::Compute in
        docs/aec3_extracts/src/aec3/refined_filter_update_gain.cc:104-138.

        Per-bin gain:
          mu[k] = H_error[k] / (0.5 × H_error[k] × X²[k]
                              + n_partitions × E²_refined[k])
        Applied to all partitions with their respective conj(X_buf[p]).

        H_error decay:
          H_error[k] -= 0.5 × mu[k] × X²[k] × H_error[k]

        Always-on refresh:
          if E²_refined_sum <= E²_coarse_sum OR disallow_leakage_diverged:
              H_error[k] += leakage_converged × erl[k]
          else:
              H_error[k] += leakage_diverged   × erl[k]
          H_error[k] = clamp(H_error[k], floor, ceil)
        """
        # X² for AEC3 formula. AEC3 uses the partition-summed render power
        # `X²[k] = Σ_p |X_buf[p][k]|²` (render_buffer.cc::SpectralSum)
        # inside RefinedFilterUpdateGain::Compute.
        X_latest = self.X_buf[curr_p]
        # Hybrid startup: partition-sum X² is "stable but slow" (mu effectively
        # divided by n_partitions in the early H_error-dominated phase), so use
        # single-partition X² for the first ~5 s (500 hops @ hop=160/sr=16000)
        # for fast initial convergence, then partition-sum for steady state.
        if self._call_counter > self._partition_sum_x2_startup_hops:
            X2 = (np.abs(self.X_buf) ** 2).sum(axis=0).astype(np.float32)
        else:
            X2 = (np.abs(X_latest) ** 2).astype(np.float32)
        delta32 = np.float32(self.delta)
        n_part = np.float32(self.n_partitions)
        # mu[k] (AEC3 formula `mu = H_error / (0.5·H_error·X² + n·E²)`).
        # E² source: current-block `|error_spec|²` per-bin matching AEC3
        # `SubtractorOutput.E2_refined` (refined_filter_update_gain.cc:106).
        e2_refined_current = (np.abs(self.error_spec) ** 2).astype(np.float32)
        denom_aec3 = (
            np.float32(0.5) * self.H_error_per_bin * X2
            + n_part * e2_refined_current
            + delta32
        )
        mu_aec3 = (self.H_error_per_bin / denom_aec3).astype(np.float32)

        # Per-bin noise_gate (refined_filter_update_gain.cc:104-111).
        # AEC3 zeros mu on bins where X² < `noise_gate`. The gate consumes
        # the same X² the denominator does. AEC3 filter-gate constant is
        # FILTER_NOISE_GATE_POWER_FLOAT (0.01870), distinct from the
        # suppression-path NOISE_GATE_POWER_FLOAT (0.02562).
        from . import aec3_scale as _aec3_scale
        _noise_gate = np.float32(_aec3_scale.FILTER_NOISE_GATE_POWER_FLOAT)
        mu_aec3 = np.where(X2 >= _noise_gate, mu_aec3, np.float32(0.0))

        # W update — per partition, use the per-bin mu × conj(X[p]).
        # Direction is per-partition irrespective of the X² source flag
        # (matches AEC3: gain is computed once on summed X², then applied
        # to each partition with its own conj(X)).
        # Gap C: error_override (when provided) substitutes for self.error_spec
        # as the gradient source.
        _err_grad = (
            error_override if error_override is not None else self.error_spec
        )
        for p in range(self.n_partitions):
            p_idx = (curr_p - p) % self.n_partitions
            X = self.X_buf[p_idx]
            K = mu_aec3 * np.conj(X)             # per-bin AEC3 K
            K_scaled = K * mu_scale_arr          # apply DT scale
            self.W[p] += K_scaled * _err_grad
            # Time-domain constraint (raised cosine fade).
            if self.enable_td_constraint:
                w_time = np.fft.irfft(self.W[p], self.fft_size).astype(np.float32)
                w_time *= self._td_window
                self.W[p] = np.fft.rfft(w_time).astype(np.complex64)

        # H_error decay (per-bin, refined_filter_update_gain.cc:116-119).
        # AEC3 places this decay INSIDE the active-update `else` block — it
        # only runs when W was actually adapted. The refresh below runs
        # ALWAYS, including the gated/stationary path (callers route to
        # `_h_error_refresh()` and return). Decay uses the same X² as the
        # denominator above (partition-summed when the flag is ON).
        self.H_error_per_bin -= (
            np.float32(0.5) * mu_aec3 * X2 * self.H_error_per_bin
        )

        # Always-on H_error refresh + clamp (cc:128-138). Identical to the
        # standalone `_h_error_refresh` path; inlined here for the active
        # branch so we don't re-iterate the array.
        self._h_error_refresh()

    def _h_error_refresh(self) -> None:
        """AEC3-aligned H_error leakage refresh + clamp.

        Mirrors lines 128-138 of refined_filter_update_gain.cc. Runs on
        every hop — both when the W-update gates fire (gated/stationary
        path returns early after calling this) and after the active
        decay step. Keeps H_error trending toward steady state and away
        from the init value during the startup window.

        Per-bin path: AEC3 cc:128-138 uses per-bin instantaneous E²_refined
        vs E²_coarse compare. Uses fresh ``|self.error_spec|²`` for refined
        (smoothed ``self._error_psd`` would be stale on early-return paths).
        The scalar fallback runs only before the orchestrator publishes the
        first per-bin ``_e2_coarse_per_bin``.
        """
        # AEC3 InitialState (aec_state.cc:336-353) — during the first 2.5 s of
        # active render, switch to refined_initial leakage (100×/10× steady)
        # so the filter converges fast. Source: FilterConfig refined_initial
        # (echo_canceller3_config.h:102-107).
        if self._initial_state_active:
            from . import aec3_scale as _aec3_scale
            _lc_eff = np.float32(_aec3_scale.LEAKAGE_CONVERGED_TRANSIENT_PER_HOP)
            _ld_eff = np.float32(_aec3_scale.LEAKAGE_DIVERGED_TRANSIENT_PER_HOP)
        else:
            _lc_eff = self._leakage_converged
            _ld_eff = self._leakage_diverged

        if self._e2_coarse_per_bin is not None:
            # AEC3 cc:128-138 per-bin path. Compute instantaneous
            # per-bin E²_refined from the current frame's filter error.
            e2_refined_per_bin = (np.abs(self.error_spec) ** 2).astype(np.float32)
            use_converged_mask = (
                e2_refined_per_bin <= self._e2_coarse_per_bin
            ) | self._disallow_leakage_diverged
            # Diag (2026-05-27): fraction of bins taking the diverged-leakage
            # branch this hop. Stashed for the orchestrator _hf_chain_trace.
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
            # Legacy scalar path (default → byte-equal preserved).
            e2_ref_sum = float(np.sum(self._error_psd))
            e2_coa_sum = float(self._e2_coarse_for_refresh)
            use_converged = (
                e2_ref_sum <= e2_coa_sum or self._disallow_leakage_diverged
            )
            # Diag (2026-05-27): scalar path is all-or-nothing, so frac is 0 or 1.
            self._last_leakage_div_frac = 0.0 if use_converged else 1.0
            leakage = _lc_eff if use_converged else _ld_eff
            self.H_error_per_bin = (
                self.H_error_per_bin + leakage * self._erl_per_bin
            )
        np.clip(self.H_error_per_bin, self._h_error_floor, self._h_error_ceil,
                out=self.H_error_per_bin)

    def copy_weights_from(self, src: 'PBFDAF'):
        self.W[:] = src.W
        # Only copy filter coefficients W, not Kalman internal state (P/Q/R).
        # P/Q/R are confidence states accumulated from each filter's own
        # learning history — copying them across filters contaminates the
        # destination's Kalman gain. AEC3's shadow uses NLMS (no state to
        # copy); our PBFDKF shadow needs this protection.
        # After W copy, P may temporarily mismatch W but Kalman self-corrects
        # within a few frames.

    def scale_filter(self, scale: float) -> None:
        """Multiplicative W rescale.

        AEC3 default: scale W only; Kalman gain
        K = P·X*/(X·P·X* + R) self-corrects within a few frames.
        """
        super().scale_filter(scale)
