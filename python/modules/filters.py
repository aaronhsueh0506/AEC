"""Adaptive filter primitives.

Extracted from ``aec.py`` during refactor R.5. Three filter classes:

* ``NlmsFilter`` — time-domain NLMS, sample-by-sample (legacy LMS line)
* ``PBFDAF`` — Partitioned Block FDAF (NLMS adaptation in freq domain)
* ``PBFDKF`` — Partitioned Block FDKF (Kalman extension of PBFDAF)

Self-contained: depends only on numpy + collections.deque + typing.
"""
from typing import Optional, Tuple

import numpy as np


class NlmsFilter:
    """Time-domain NLMS Adaptive Filter"""

    def __init__(self, filter_length: int, mu: float = 0.3,
                 delta: float = 1e-8,
                 normalize: bool = True):
        self.filter_length = filter_length
        self.mu = mu
        self.delta = delta
        self.normalize = normalize
        self.weights = np.zeros(filter_length, dtype=np.float32)
        self.ref_buffer = np.zeros(filter_length, dtype=np.float32)
        self.power_sum = 0.0
        self.clear_history = False
        self.max_w_norm = 1.5  # Weight norm constraint (prevents explosion during double-talk)

    def reset(self):
        self.weights.fill(0)
        self.ref_buffer.fill(0)
        self.power_sum = 0.0

    def process_sample(self, near_end: float, far_end: float,
                       mu_scale: float = 1.0) -> Tuple[float, float]:
        oldest = self.ref_buffer[-1]
        self.power_sum = max(0, self.power_sum - oldest * oldest + far_end * far_end)
        self.ref_buffer[1:] = self.ref_buffer[:-1]
        self.ref_buffer[0] = far_end
        echo_est = np.dot(self.weights, self.ref_buffer)
        error = near_end - echo_est

        if mu_scale > 0 and self.power_sum > self.delta * self.filter_length:
            if self.normalize:
                mu_eff = (self.mu * mu_scale) / (self.power_sum + self.delta)
            else:
                mu_eff = self.mu * mu_scale
            self.weights += mu_eff * error * self.ref_buffer

        return error, echo_est

    def process_block(self, near_end: np.ndarray, far_end: np.ndarray,
                      mu_scale: float = 1.0) -> Tuple[np.ndarray, np.ndarray]:
        # Optionally clear history (no carry-over between blocks)
        if self.clear_history:
            self.ref_buffer.fill(0)
            self.power_sum = 0.0

        n = len(near_end)
        output = np.zeros(n, dtype=np.float32)
        echo_est = np.zeros(n, dtype=np.float32)
        for i in range(n):
            output[i], echo_est[i] = self.process_sample(
                near_end[i], far_end[i], mu_scale)

        # Weight norm constraint: prevent explosion during double-talk
        w_norm = np.linalg.norm(self.weights)
        if w_norm > self.max_w_norm:
            self.weights *= self.max_w_norm / w_norm

        return output, echo_est


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
        # as OLA spec but time-aligned with far_spec/echo_spec for coherence)
        self._sqrt_hann_analysis = np.sqrt(
            np.hanning(self.block_size)).astype(np.float32)
        self.error_spec_windowed = np.zeros(self.n_freqs, dtype=np.complex64)

        # v3.21.14 — PBFDAF shadow NLMS AEC3 protection alignment flags
        # (default OFF preserves v3.21.6 baseline byte-equal). Set externally
        # by orchestrator from AecConfig. AEC3 reference:
        # docs/aec3_extracts/src/aec3/coarse_filter_update_gain.cc:34-82.
        # A.1: mu denom uses partition-summed X² = Σ_p |X_buf[p]|² (cc:64-72).
        self._use_partition_summed_x2_for_shadow_mu: bool = False
        # A.2: noise_gate hard zero — mu=0 where X² < NOISE_GATE_POWER_FLOAT (cc:67-71).
        self._use_aec3_noise_gate_for_shadow: bool = False
        # A.3: poor_excitation + startup gate — return when poor_excitation_counter
        # < n_partitions OR call_counter <= n_partitions (cc:51-61).
        self._use_poor_excitation_gate_for_shadow: bool = False
        # A.4: narrowband mask — RenderSignalAnalyzer.mask_regions_around_narrow_bands (cc:75).
        self._use_narrowband_mask_for_shadow: bool = False
        # A.5: saturation gate — early return when _saturated_capture (cc:56-57).
        self._use_saturation_gate_for_shadow: bool = False
        # v3.21.14 A.3 / A.4 supporting state (default values harmless under
        # flag-OFF; orchestrator overrides on shadow construction). Defined on
        # PBFDAF so subclass PBFDKF inherits the same state attributes; PBFDKF
        # __init__ already re-assigns these to its own AEC3-derived defaults.
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

    def reset(self):
        self.W.fill(0)
        self.X_buf.fill(0)
        self.near_buffer.fill(0)
        self.far_buffer.fill(0)
        self.power.fill(0)
        self.partition_idx = 0
        self.error_spec_windowed.fill(0)

    def handle_echo_path_change(self, delay_change: bool = True,
                                  gain_change: bool = False,
                                  zero_filter: bool = False) -> None:
        """Port AEC3 CoarseFilterUpdateGain::HandleEchoPathChange counter reset
        (M3: not-gain_change → poor_excitation_counter = INITIAL + call_counter = 0).
        AEC3 Subtractor::HandleEchoPathChange dispatches to both refined + coarse.

        NOTE on zero_filter: W.fill(0) is NOT AEC3 ZeroFilter parity.
        AEC3 AdaptiveFirFilter::ZeroFilter(current_size, max_size) zeroes only
        partitions in [current..max). In steady state (current=max=13) this is a
        NO-OP — W is fully preserved. W.fill(0) is a default-OFF ablation flag
        (non-AEC3 behaviour); it is NOT listed as a v3.21 strict alignment candidate.
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
            byte-equal vs the v3.21.6 inline path.
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
            replaces self.error_spec as the gradient source. Default None
            preserves byte-equal vs v3.21.6 baseline.
        """
        mu_scale_arr = np.asarray(mu_scale, dtype=np.float32)
        if mu_scale_arr.ndim == 0:
            mu_scale_arr = np.full(self.n_freqs, float(mu_scale_arr), dtype=np.float32)
        if not np.any(mu_scale_arr > 0):
            return
        # v3.21.14 A.5 — AEC3 coarse_filter_update_gain.cc:56-57 saturation gate.
        # `saturated_capture_signal → return`. Default-OFF preserves v3.21.6
        # behaviour (shadow updates regardless of mic saturation).
        if (self._use_saturation_gate_for_shadow
                and getattr(self, '_saturated_capture', False)):
            self._c1c5_trace['A5_sat_skip'] = True
            return
        # v3.21.14 A.3 — AEC3 coarse_filter_update_gain.cc:51-61 call_counter +
        # poor_excitation startup gate. `if poor_signal_excitation_counter_ <
        # size_partitions OR call_counter_ <= size_partitions → return`.
        # Counter increment is gated on the flag so default-OFF leaves the
        # counter untouched and produces byte-equal output.
        if self._use_poor_excitation_gate_for_shadow:
            self._call_counter += 1
            if (self._call_counter <= self.n_partitions
                    or self._poor_excitation_counter < self.n_partitions):
                self._c1c5_trace['A3_poor_exc_skip'] = True
                self._c1c5_trace['A3_call_ctr'] = self._call_counter
                self._c1c5_trace['A3_exc_ctr'] = self._poor_excitation_counter
                return
        # v3.21 Phase C.3+B/D extension — stationary-far gate (see PBFDKF
        # version for full rationale). Shadow NLMS also skips W update when
        # the StationarityEstimator flags the block as stationary; spurious
        # mic-as-echo coupling from broadband stationary noise damages
        # nearend equally in NLMS path.
        if getattr(self, '_block_stationary', False):
            return
        # v3.21.14 A.4 — AEC3 coarse_filter_update_gain.cc:75
        # `render_signal_analyzer.MaskRegionsAroundNarrowBands(&mu)`. Apply the
        # narrowband mask as a pre-multiplier on mu_scale_arr so any subsequent
        # mu_eff = mu * mu_scale_arr inherits the mask. Default-OFF preserves
        # baseline behaviour (no mask).
        if (self._use_narrowband_mask_for_shadow
                and self._render_signal_analyzer is not None):
            rsa_mask = np.ones(self.n_freqs, dtype=np.float32)
            self._render_signal_analyzer.mask_regions_around_narrow_bands(rsa_mask)
            mu_scale_arr = (mu_scale_arr * rsa_mask).astype(np.float32)
            self._c1c5_trace['A4_mask_frac'] = float(np.mean(rsa_mask < 0.5))
        # Per-bin local floor: allows low-energy mid-freq bins higher effective mu
        local_floor = self.power * 0.01 + self.delta        # per-bin 1% floor
        global_floor = np.mean(self.power) * 0.001 + self.delta  # global extreme floor
        power_floor = np.maximum(self.power, np.maximum(local_floor, global_floor))
        # v3.21.14 A.1 — AEC3 coarse_filter_update_gain.cc:64-72 mu denominator
        # uses partition-summed X² = Σ_p |X_buf[p]|² from the current frame
        # (no smoothing). Default-OFF preserves v3.21.6 EMA-smoothed
        # `power_floor × n_partitions` denominator. ON matches AEC3 SpectralSum
        # source semantic so shadow transient response aligns with AEC3 coarse.
        if self._use_partition_summed_x2_for_shadow_mu:
            x2_partition_sum = (np.abs(self.X_buf) ** 2).sum(axis=0).astype(np.float32)
            denom = x2_partition_sum + self.delta
            self._c1c5_trace['A1_x2_active'] = True
        else:
            denom = power_floor * self.n_partitions + self.delta
        mu_eff = (self.mu * mu_scale_arr) / denom
        # v3.21.14 A.2 — AEC3 coarse_filter_update_gain.cc:67-71 noise_gate
        # hard zero. AEC3 sets `mu[k] = 0` where `X²[k] < noise_gate`. X²
        # source = SpectralSum (matches A.1) regardless of A.1 flag, since
        # this is the AEC3-semantic X² for gate purposes. Default-OFF
        # preserves v3.21.6 floor-only behaviour (no hard zero).
        # T1.2 (2026-05-26): uses FILTER_NOISE_GATE_POWER_FLOAT (20075344 int16²
        # = 0.01870) — the AEC3 coarse filter gate constant confirmed from
        # echo_canceller3_config.cc:99. The prior NOISE_GATE_POWER_FLOAT (27509562
        # = 0.02562) was incorrectly ported from the suppression path.
        if self._use_aec3_noise_gate_for_shadow:
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
        """v3.18 Phase B.3 — AEC3-aligned multiplicative W rescale.

        Multiplies every partition's W by `scale` in-place. Mirrors
        AEC3 subtractor.cc ScaleFilter action used by
        FilterMisadjustmentEstimator to correct long-term W magnitude
        drift. PBFDKF overrides to optionally rescale Kalman P as well.
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

        # v3.21 Phase B.4 — AEC3 startup / poor-excitation / saturation gates
        # (refined_filter_update_gain.cc:96-99). The orchestrator sets
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
        # v3.21 Phase B.3 — RenderSignalAnalyzer for per-bin narrow-band mask.
        # Set externally by orchestrator; None disables masking.
        self._render_signal_analyzer = None

        # v3.21 Phase B.2 — AEC3 H_error per-bin state + leakage refresh.
        # H_error replaces our P matrix as the primary Kalman-like state in
        # the K compute (refined_filter_update_gain.cc:104-138). P stays as
        # a parallel field for backwards-compat (PathChangeRegimeHandler
        # overrides / diagnostic), but the K applied to W comes from
        # H_error when use_aec3_h_error=True (default). orchestrator sets
        # _e2_coarse_for_refresh, _disallow_leakage_diverged, _erl_per_bin
        # per hop to feed the always-on refresh formula.
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
        # Updated per hop by orchestrator. v3.21.1: per-bin support added.
        # Scalar `_e2_coarse_for_refresh` is the legacy sum used by the
        # OFF-default scalar path. Per-bin `_e2_coarse_per_bin` is the
        # instantaneous coarse error PSD published per hop, consumed by
        # the AEC3 cc:128-138 per-bin path when
        # `_use_per_bin_h_error_refresh = True`.
        self._e2_coarse_for_refresh = 0.0
        self._e2_coarse_per_bin: Optional[np.ndarray] = None
        self._use_per_bin_h_error_refresh: bool = True
        # v3.21.6 nores LF artifact debug 2026-05-22 — AEC3
        # `RefinedFilterUpdateGain::Compute` partition-summed X² parity.
        # OFF (default): X²_latest = |X_buf[curr_p]|² in denom / noise_gate
        # / H_error decay (byte-equal preserved).
        # ON: Σ_p |X_buf[p]|² (matches AEC3 render_buffer.SpectralSum).
        # W update partition direction unchanged either way.
        self._use_partition_summed_x2_for_h_error_gain: bool = True
        # v3.21.20 Phase C Fix B — startup hops before switching from single-
        # partition to partition-sum X². 500 hops = 5 s @ hop=160/sr=16000.
        # Orchestrator overrides from config field.
        self._partition_sum_x2_startup_hops: int = 0
        # v3.21.12 RefinedFilterUpdateGain input-parity audit 2026-05-22 — AEC3
        # `RefinedFilterUpdateGain::Compute` (refined_filter_update_gain.cc:103-107)
        # uses current-block `E2_refined[k]` = `SubtractorOutput.E2_refined` =
        # per-bin spectrum of THIS block's e_refined (instantaneous, no
        # smoothing). Our default uses `self._error_psd` (0.95 EMA of
        # |error_spec|², ~200 ms time constant).
        # OFF (default): smoothed `_error_psd` (byte-equal preserved).
        # ON: current-block `|self.error_spec|²` in the `n_part × E²` term of
        # the mu denominator (per-bin, no smoothing).
        # W update direction unchanged. See
        # docs/v3_21_12_refined_filter_update_gain_input_parity_plan.md.
        self._use_current_e2_refined_in_h_error_denominator: bool = True
        self._disallow_leakage_diverged = False
        # ERL per bin (lazy init to 0.1 = -10 dB nominal; orchestrator
        # overwrites once its ERL estimator has a real value).
        self._erl_per_bin = np.full(self.n_freqs, 0.1, dtype=np.float32)
        # Master switch: use AEC3 H_error path (True, default) vs legacy
        # P-based denominator (False, diagnostic only).
        self._use_aec3_h_error = True
        # R0.1 — refined filter noise gate constant. Default False → byte-equal
        # (NOISE_GATE_POWER_FLOAT=0.02562). True → FILTER_NOISE_GATE_POWER_FLOAT=0.01870.
        # Wired by orchestrator from config.use_aec3_filter_noise_gate_power.
        self._use_aec3_filter_noise_gate_power: bool = True

        # GPT Phase 1 debug trace (off by default, zero overhead).
        # When enabled, accumulates per-frame stats to verify hypothesis:
        # "DT 期間 mu_scale 壓低但 P 仍因 K_optimal 快下降 → DT 結束後 P 偏低 → recovery 慢"
        self._enable_kx_trace = False
        self._kx_trace = []  # list of dicts, one per call to _update_weights

        # P53 Step 0 innovation-audit hook (default OFF, zero overhead).
        # When enabled, _update_weights appends per-frame per-bin arrays
        # capturing the Kalman innovation orthogonality components.
        self._enable_p53_trace = False
        self._p53_innovation_trace = []  # list of dicts of np.ndarray

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
        → mu stays small → can't re-track movement → DT damage accumulates
        (wVYS movement-W-shift bug, evidence in v3.21.20 Phase A+C trace).
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

        # v3.21 Phase B.4 — AEC3 startup / poor-excitation / saturation gates
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
            if self._use_aec3_h_error:
                self._h_error_refresh()
            return

        # v3.21 Phase C.3+B/D extension — stationary-far gate. When the
        # render-path StationarityEstimator flags the current block as
        # stationary (constant background hum / fan / line noise), any
        # filter adaptation against it produces spurious mic-as-echo
        # coupling because the noise has no causal correlation with the
        # nearend signal. RSA's `poor_signal_excitation` covers tonal
        # peaks but not broadband stationary noise (RSA narrow-band counter
        # 0% on E0l0 hum case); StationarityEstimator (B.3 / Phase C.3 noise
        # tracker) catches that gap. When set, the orchestrator pushes
        # `_block_stationary = True` before `_update_weights` runs.
        if getattr(self, '_block_stationary', False):
            if self._use_aec3_h_error:
                self._h_error_refresh()
            return

        # v3.21 Phase B.3 — RenderSignalAnalyzer narrow-band mask. Zeros mu
        # for ±2 bins around any frequency that has sustained > 5 frames of
        # tonal X²[k] > 3 × max(neighbors) condition. Mask is applied as a
        # pre-multiplier on mu_scale_arr so the existing K_scaled = K_optimal
        # × mu_scale_arr path naturally zeroes those bins' W update.
        if self._render_signal_analyzer is not None:
            rsa_mask = np.ones(self.n_freqs, dtype=np.float32)
            self._render_signal_analyzer.mask_regions_around_narrow_bands(rsa_mask)
            mu_scale_arr = (mu_scale_arr * rsa_mask).astype(np.float32)

        # Update measurement noise estimate from error PSD
        error_psd = np.abs(self.error_spec) ** 2
        self._error_psd = self._alpha_r * self._error_psd + (1 - self._alpha_r) * error_psd
        self.R = np.maximum(self._error_psd, self.delta)

        # v3.21 Phase B.2 — AEC3 H_error path. Computes per-bin Kalman gain
        # via H_error rather than the legacy partition-summed P denominator.
        # Mirrors refined_filter_update_gain.cc:104-138.
        if self._use_aec3_h_error:
            self._update_weights_aec3(curr_p, mu_scale_arr, error_psd,
                                      error_override=error_override)
            return

        # Adaptive R: scale by mu_scale to break R-deadlock
        mu_mean = float(np.mean(mu_scale_arr))
        R_scale = 0.1 + 0.9 * (1.0 - mu_mean)
        self.R = self.R * R_scale

        # Q modulation: reduce Q during DT (Kalman-specific, not in NLMS)
        q_scale = 0.1 + 0.9 * mu_mean  # FS: 1.0, strong DT: 0.1
        Q_modulated = self.Q * q_scale

        # Per-bin far-end activity mask
        far_power_smooth = self.power
        far_activity_mask = far_power_smooth > (np.mean(far_power_smooth) * 0.01 + 1e-6)
        Q_floor = Q_modulated * 0.05
        Q_gated = np.where(far_activity_mask, Q_modulated, Q_floor)

        # P_MAX: overridable during EPC for faster re-convergence
        p_max = getattr(self, '_p_max_override', 0.5)
        if hasattr(self, '_p_max_override_frames'):
            self._p_max_override_frames -= 1
            if self._p_max_override_frames <= 0:
                self._p_max_override = 0.5
                del self._p_max_override_frames

        # P-floor: prevent P from collapsing to delta after long convergence
        # (which kills Kalman gain on movement). EPC sets beta=1.0 transiently
        # to force full re-tracking; steady state uses beta=0.1.
        beta = getattr(self, '_p_floor_beta', 0.1)
        if hasattr(self, '_p_floor_beta_frames'):
            self._p_floor_beta_frames -= 1
            if self._p_floor_beta_frames <= 0:
                self._p_floor_beta = 0.1
                del self._p_floor_beta_frames
        P_floor = self.Q_high * beta

        # Global denominator: sum over ALL partitions (correct Kalman theory)
        # C-parity fix: cast delta to float32 to prevent denominator from
        # promoting to float64, which cascades K_optimal to complex128 and
        # causes divergence from C's float32-only arithmetic.
        total_echo_var = np.zeros(self.n_freqs, dtype=np.float32)
        for p in range(self.n_partitions):
            p_idx = (curr_p - p) % self.n_partitions
            X = self.X_buf[p_idx]
            total_echo_var += self.P[p] * (np.abs(X) ** 2)
        denominator = total_echo_var + self.R + np.float32(self.delta)

        # P53 Step 0: per-frame innovation-audit capture (default OFF).
        # Stores per-bin arrays needed to compute r = C_obs / C_exp offline.
        if self._enable_p53_trace:
            P_diag = np.zeros(self.n_freqs, dtype=np.float32)
            far_psd_sum = np.zeros(self.n_freqs, dtype=np.float32)
            for _p in range(self.n_partitions):
                _pi = (curr_p - _p) % self.n_partitions
                P_diag += self.P[_p]
                far_psd_sum += np.abs(self.X_buf[_pi]) ** 2
            self._p53_innovation_trace.append({
                'innovation_power': error_psd.astype(np.float32).copy(),
                'R': self.R.astype(np.float32).copy(),
                'total_echo_var': total_echo_var.astype(np.float32).copy(),
                'denominator': denominator.astype(np.float32).copy(),
                'P_diag': P_diag,
                'far_psd': far_psd_sum,
            })

        # GPT Phase 1 trace: per-partition KX_optimal vs KX_scaled accumulator.
        if self._enable_kx_trace:
            kx_opt_acc = []
            kx_scaled_acc = []
            p_before_acc = []
            p_after_acc = []

        # Gap C: error_override (when provided) substitutes for self.error_spec
        # as the gradient source in the W += K_scaled * err line below.
        _err_grad = (
            error_override if error_override is not None else self.error_spec
        )
        for p in range(self.n_partitions):
            p_idx = (curr_p - p) % self.n_partitions
            X = self.X_buf[p_idx]

            K_optimal = (self.P[p] * np.conj(X)) / denominator

            # Bug 2 fix: separate K for weights (scaled) and P update (unscaled)
            K_scaled = K_optimal * mu_scale_arr

            self.W[p] += K_scaled * _err_grad

            # PR-G1 (v3.7 candidate, GPT Phase 1): blended KX for P update.
            # KX trace 2026-04-30 confirmed hypothesis on DT_st bucket: when
            # mu_scale=0 (full DT freeze), W stays put but P drops 72% per
            # cycle via K_optimal — P 與 W 不一致，DT 結束後 K 偏低、recovery 慢.
            # Blend KX_optimal (current) with KX_scaled (W-consistent) by
            # mu_mean: FS (mu=1) → 100% optimal, DT (mu=0) → 100% scaled,
            # smooth transition between. Avoids both extremes' regression
            # risk (full KX_optimal: P over-confident in DT; full KX_scaled:
            # P over-cautious in steady state with weak far / per-bin gating).
            KX_optimal = np.real(K_optimal * X).astype(np.float32)
            KX_scaled = np.real(K_scaled * X).astype(np.float32)
            mu_mean_f32 = np.float32(mu_mean)
            KX = mu_mean_f32 * KX_optimal + (np.float32(1.0) - mu_mean_f32) * KX_scaled
            if self._enable_kx_trace:
                kx_opt_acc.append(float(np.mean(KX_optimal)))
                kx_scaled_acc.append(float(np.mean(KX_scaled)))
                p_before_acc.append(float(np.mean(self.P[p])))
            self.P[p] = np.minimum(
                np.maximum((np.float32(1.0) - KX) * self.P[p] + Q_gated, P_floor),
                p_max
            )
            if self._enable_kx_trace:
                p_after_acc.append(float(np.mean(self.P[p])))

            # Time-domain constraint (raised cosine fade)
            if self.enable_td_constraint:
                w_time = np.fft.irfft(self.W[p], self.fft_size).astype(np.float32)
                w_time *= self._td_window
                self.W[p] = np.fft.rfft(w_time).astype(np.complex64)

        if self._enable_kx_trace:
            self._kx_trace.append({
                'mu_mean': float(np.mean(mu_scale_arr)),
                'kx_opt_mean': float(np.mean(kx_opt_acc)),
                'kx_scaled_mean': float(np.mean(kx_scaled_acc)),
                'p_before_mean': float(np.mean(p_before_acc)),
                'p_after_mean': float(np.mean(p_after_acc)),
                'p_p10': float(np.percentile([np.percentile(self.P[p], 10) for p in range(self.n_partitions)], 50)),
                'p_p50': float(np.median([np.median(self.P[p]) for p in range(self.n_partitions)])),
                'p_p90': float(np.percentile([np.percentile(self.P[p], 90) for p in range(self.n_partitions)], 50)),
                'q_gated_mean': float(np.mean(Q_gated)),
                'far_power_mean': float(np.mean(self.power)),
                'error_power_mean': float(np.mean(self._error_psd)),
            })

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
        # `X²[k] = Σ_p |X_buf[p][k]|²` (render_buffer.cc::SpectralSum) inside
        # RefinedFilterUpdateGain::Compute. Our default port substitutes
        # `X²_latest = |X_buf[curr_p]|²` (newest hop only), which under-states
        # the render energy when the impulse response spans multiple
        # partitions. The summed branch ships behind
        # `_use_partition_summed_x2_for_h_error_gain` (default False →
        # byte-equal vs v3.21.6 baseline).
        X_latest = self.X_buf[curr_p]
        if self._use_partition_summed_x2_for_h_error_gain:
            # v3.21.20 Phase C Fix B — hybrid startup window.
            # AEC3 partition-sum X² gives "stable but slow" convergence
            # (mu effectively divided by n_partitions in early H_error-
            # dominated phase). On our pipeline that was tuned around fast
            # single-partition convergence, this starves the refined filter
            # for the first ~5s (filter_w_norm ratio 0.032-0.288 vs single-
            # partition baseline; trace evidence Phase A nVUnxqHLr).
            # Hybrid: use single-partition X² during startup (fast initial
            # convergence matches v3.21.6 behaviour), switch to partition-
            # sum after `_partition_sum_x2_startup_hops` (AEC3 steady-state
            # stability). 500 hops = 5 s @ hop=160/sr=16000.
            if self._call_counter > self._partition_sum_x2_startup_hops:
                X2 = (np.abs(self.X_buf) ** 2).sum(axis=0).astype(np.float32)
            else:
                X2 = (np.abs(X_latest) ** 2).astype(np.float32)
        else:
            X2 = (np.abs(X_latest) ** 2).astype(np.float32)
        delta32 = np.float32(self.delta)
        n_part = np.float32(self.n_partitions)
        # mu[k] (AEC3 formula `mu = H_error / (0.5·H_error·X² + n·E²)`).
        # E² source is controlled by `_use_current_e2_refined_in_h_error_denominator`
        # (v3.21.12). OFF: smoothed `_error_psd` (legacy). ON: current-block
        # `|error_spec|²` per-bin matching AEC3 `SubtractorOutput.E2_refined`.
        # Comment correction: the previous "AEC3 also uses a smoothed
        # estimate" assertion was WRONG — AEC3 uses the current SubtractorOutput
        # E²_refined directly (refined_filter_update_gain.cc:106).
        if self._use_current_e2_refined_in_h_error_denominator:
            e2_refined_current = (np.abs(self.error_spec) ** 2).astype(np.float32)
        else:
            e2_refined_current = self._error_psd
        denom_aec3 = (
            np.float32(0.5) * self.H_error_per_bin * X2
            + n_part * e2_refined_current
            + delta32
        )
        mu_aec3 = (self.H_error_per_bin / denom_aec3).astype(np.float32)

        # v3.21 NE-outlier fix — per-bin noise_gate
        # (refined_filter_update_gain.cc:104-111). AEC3 zeros mu on bins
        # where X² < `noise_gate`. The gate consumes the same X² the
        # denominator does — partition-summed when the flag is ON.
        # R0.1: use_aec3_filter_noise_gate_power selects the correct AEC3
        # filter gate constant (0.01870 from 20075344 int16²) vs the legacy
        # default (0.02562 from 27509562, ported from suppression path).
        from . import aec3_scale as _aec3_scale
        _noise_gate = np.float32(
            _aec3_scale.FILTER_NOISE_GATE_POWER_FLOAT
            if self._use_aec3_filter_noise_gate_power
            else _aec3_scale.NOISE_GATE_POWER_FLOAT
        )
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

        v3.21.1: per-bin path added behind ``_use_per_bin_h_error_refresh``
        flag. AEC3 cc:128-138 uses per-bin instantaneous E²_refined vs
        E²_coarse compare. Legacy scalar path stays as default OFF for
        byte-equal preservation. Per-bin path uses fresh ``|self.error_spec|²``
        for refined (addresses Codex F2 staleness on early-return paths
        where the smoothed ``self._error_psd`` is from a prior frame).
        """
        if (self._use_per_bin_h_error_refresh
                and self._e2_coarse_per_bin is not None):
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
                self._leakage_converged,
                self._leakage_diverged,
            ).astype(np.float32)
            self.H_error_per_bin = (
                self.H_error_per_bin + leakage_arr * self._erl_per_bin
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
            leakage = (self._leakage_converged if use_converged
                       else self._leakage_diverged)
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

    def scale_filter(self, scale: float, scale_p: bool = False) -> None:
        """v3.18 Phase B.3 — multiplicative W rescale with optional P scale.

        AEC3 default (scale_p=False): scale W only; Kalman gain
        K = P·X*/(X·P·X* + R) self-corrects within a few frames.
        Option A (scale_p=True): also scale P by scale² (Kalman-canonical
        variance scaling). Set via config.filter_misadjustment_scale_p.
        """
        super().scale_filter(scale)
        if scale_p:
            self.P *= float(scale) ** 2
