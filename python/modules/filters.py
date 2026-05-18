"""Adaptive filter primitives.

Extracted from ``aec.py`` during refactor R.5. Three filter classes:

* ``NlmsFilter`` — time-domain NLMS, sample-by-sample (legacy LMS line)
* ``PBFDAF`` — Partitioned Block FDAF (NLMS adaptation in freq domain)
* ``PBFDKF`` — Partitioned Block FDKF (Kalman extension of PBFDAF)

Self-contained: depends only on numpy + collections.deque + typing.
"""
from typing import Tuple

import numpy as np
from collections import deque


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

    def reset(self):
        self.W.fill(0)
        self.X_buf.fill(0)
        self.near_buffer.fill(0)
        self.far_buffer.fill(0)
        self.power.fill(0)
        self.partition_idx = 0
        self.error_spec_windowed.fill(0)

    def process(self, near_end: np.ndarray, far_end: np.ndarray,
                mu_scale=1.0) -> np.ndarray:
        """Process hop_size samples. mu_scale: scalar or per-bin array [n_freqs]."""
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
        far_hop_energy = np.sum(far_end ** 2) / hop
        if far_hop_energy > 1e-4:  # ~ -40 dBFS, unified with far_active threshold
            self._update_weights(curr_p, mu_scale)

        self.partition_idx = (self.partition_idx + 1) % self.n_partitions
        return output.astype(np.float32)

    def _update_weights(self, curr_p: int, mu_scale):
        """NLMS weight update."""
        mu_scale_arr = np.asarray(mu_scale, dtype=np.float32)
        if mu_scale_arr.ndim == 0:
            mu_scale_arr = np.full(self.n_freqs, float(mu_scale_arr), dtype=np.float32)
        if not np.any(mu_scale_arr > 0):
            return
        # Per-bin local floor: allows low-energy mid-freq bins higher effective mu
        local_floor = self.power * 0.01 + self.delta        # per-bin 1% floor
        global_floor = np.mean(self.power) * 0.001 + self.delta  # global extreme floor
        power_floor = np.maximum(self.power, np.maximum(local_floor, global_floor))
        mu_eff = (self.mu * mu_scale_arr) / (power_floor * self.n_partitions + self.delta)
        for p in range(self.n_partitions):
            p_idx = (curr_p - p) % self.n_partitions
            grad = self.error_spec * np.conj(self.X_buf[p_idx])
            self.W[p] += mu_eff * grad
            # Time-domain constraint: fade out non-causal part (raised cosine)
            if self.enable_td_constraint:
                w_time = np.fft.irfft(self.W[p], self.fft_size).astype(np.float32)
                w_time *= self._td_window
                self.W[p] = np.fft.rfft(w_time).astype(np.complex64)

    def get_error_energy(self) -> float:
        return float(np.sum(np.abs(self.error_spec) ** 2))

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

    def _update_weights(self, curr_p: int, mu_scale):
        """Frequency-Domain Kalman Filter weight update."""
        mu_scale_arr = np.asarray(mu_scale, dtype=np.float32)
        if mu_scale_arr.ndim == 0:
            mu_scale_arr = np.full(self.n_freqs, float(mu_scale_arr), dtype=np.float32)

        # v3.21 Phase B.4 — AEC3 startup / poor-excitation / saturation gates
        # (refined_filter_update_gain.cc:96-99). Tick the call counter and
        # short-circuit out of the W update when any gate fires. R / Q / P
        # accounting still happens via the cold-init values; the filter just
        # doesn't accumulate gradient updates during the protected window.
        self._call_counter += 1
        if (self._call_counter <= self.n_partitions
                or self._poor_excitation_counter < self.n_partitions
                or self._saturated_capture):
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

        for p in range(self.n_partitions):
            p_idx = (curr_p - p) % self.n_partitions
            X = self.X_buf[p_idx]

            K_optimal = (self.P[p] * np.conj(X)) / denominator

            # Bug 2 fix: separate K for weights (scaled) and P update (unscaled)
            K_scaled = K_optimal * mu_scale_arr

            self.W[p] += K_scaled * self.error_spec

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


# Backward compatibility alias
SubbandNlms = PBFDKF
