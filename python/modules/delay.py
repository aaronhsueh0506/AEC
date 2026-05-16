"""DelayEstimator — GCC-PHAT delay estimator for AEC reference alignment.

Extracted from ``aec.py`` during refactor R.4. Self-contained: depends
only on numpy + collections.deque.
"""
import numpy as np
from collections import deque


class DelayEstimator:
    """GCC-PHAT delay estimator for AEC reference alignment.

    Uses short overlapping segments for fast initial estimation.
    Cross-spectrum is accumulated over segments and smoothed with EMA.
    """

    def __init__(self, sample_rate: int, max_delay_ms: float = 1024.0,
                 init_seconds: float = 0.5, period_seconds: float = 2.0,
                 par_low_threshold: float = 5.0,
                 par_solid_threshold: float = 8.0,
                 trace: bool = False,
                 fast_path_enabled: bool = False,
                 fast_par_threshold: float = 40.0):
        """v3.10.4: max_delay_ms default 250 → 512 → 1024 (matches WebRTC's
        Old AEC ~1 s far-end history; AEC3's 512 ms misses BT/mobile skew
        cases). seg_size auto-scales to 2× max_delay.

        Confidence reporting:
          peak_to_avg_ratio (PAR) is the GCC-PHAT peak height divided by
          the average magnitude over the search window. PAR > solid → trust
          the estimate (drive aggressive RES + filter mu_scale). PAR < low
          → don't trust (hold mu_scale at higher floor, RES stays
          conservative). Between: progressive blend.
        """
        self.sample_rate = sample_rate
        self.max_delay_samples = int(max_delay_ms * sample_rate / 1000)
        self.init_seconds = init_seconds
        self.period_seconds = period_seconds
        self.par_low_threshold = par_low_threshold
        self.par_solid_threshold = par_solid_threshold
        # P3c Phase 1a: high-PAR fast-path knobs.
        self.fast_path_enabled = bool(fast_path_enabled)
        self.fast_par_threshold = float(fast_par_threshold)
        self._prev_estimated_delay = -1  # set in _estimate before overwrite

        # Analysis window: 2x max_delay, but at least 2048
        self.seg_size = 1
        min_seg = max(2048, 2 * self.max_delay_samples)
        while self.seg_size < min_seg:
            self.seg_size *= 2
        self.seg_hop = self.seg_size // 2  # 50% overlap
        self.n_freqs = self.seg_size // 2 + 1

        # Smoothed cross-spectrum
        self._cross_spec = np.zeros(self.n_freqs, dtype=np.complex128)
        self._alpha = 0.6
        self._n_updates = 0

        # Sliding buffers (accumulate hop-by-hop)
        self._mic_buf = np.zeros(self.seg_size, dtype=np.float32)
        self._ref_buf = np.zeros(self.seg_size, dtype=np.float32)
        self._buf_pos = 0  # how many samples in current segment

        # State
        self.estimated_delay = -1
        self._samples_accumulated = 0
        self._samples_since_est = 0
        self._init_done = False
        self._init_samples = int(init_seconds * sample_rate)
        self._period_samples = int(period_seconds * sample_rate)
        self._n_estimates = 0
        self._last_par = 0.0  # A5: Peak-to-Average Ratio confidence

        # P3b instrumentation (opt-in; default off → zero overhead).
        self._trace = bool(trace)
        self._trace_rows: List[dict] = [] if self._trace else None
        self._trace_call_idx = 0
        self._trace_total_samples = 0

    def reset(self):
        self._cross_spec.fill(0)
        self._mic_buf.fill(0)
        self._ref_buf.fill(0)
        self._buf_pos = 0
        self._n_updates = 0
        self.estimated_delay = -1
        self._samples_accumulated = 0
        self._samples_since_est = 0
        self._init_done = False
        self._n_estimates = 0

    def accumulate(self, mic: np.ndarray, ref: np.ndarray) -> bool:
        """Feed mic/ref samples. Returns True if a new delay estimate was made."""
        n = len(mic)
        # P3b: capture pre-call state so we can emit a trace row even on
        # branches that don't run _estimate.
        if self._trace:
            mic_pwr_pre = float(np.mean(np.asarray(mic, dtype=np.float64) ** 2))
            ref_pwr_pre = float(np.mean(np.asarray(ref, dtype=np.float64) ** 2))
            n_updates_pre = self._n_updates
            samples_pre = self._samples_accumulated
        self._samples_accumulated += n
        self._samples_since_est += n

        # Accumulate into segment buffer
        remaining = n
        src_pos = 0
        while remaining > 0:
            space = self.seg_size - self._buf_pos
            chunk = min(remaining, space)
            self._mic_buf[self._buf_pos:self._buf_pos + chunk] = mic[src_pos:src_pos + chunk]
            self._ref_buf[self._buf_pos:self._buf_pos + chunk] = ref[src_pos:src_pos + chunk]
            self._buf_pos += chunk
            src_pos += chunk
            remaining -= chunk

            if self._buf_pos >= self.seg_size:
                # Segment full — update cross-spectrum
                self._update_cross_spectrum()
                # Shift by seg_hop (50% overlap)
                self._mic_buf[:self.seg_hop] = self._mic_buf[self.seg_hop:]
                self._ref_buf[:self.seg_hop] = self._ref_buf[self.seg_hop:]
                self._buf_pos = self.seg_hop

        # Estimate when enough data accumulated
        did_estimate = False
        if self._n_updates < 2:
            ret = False
        elif not self._init_done:
            if self._samples_accumulated >= self._init_samples:
                self._estimate()
                self._init_done = True
                did_estimate = True
                ret = True
            else:
                ret = False
        else:
            if self._samples_since_est >= self._period_samples:
                self._estimate()
                did_estimate = True
                ret = True
            else:
                ret = False

        # P3b: emit one trace row per accumulate() call.
        if self._trace:
            sr = self.sample_rate
            top3 = []
            par_top1 = 0.0
            if self._n_updates >= 1:
                # Recompute GCC-PHAT over the running EMA cross-spectrum
                # (same expression as _estimate; uses fp64 throughout).
                mag = np.abs(self._cross_spec) + 1e-12
                phat = self._cross_spec / mag
                gcc = np.fft.irfft(phat, n=self.seg_size).astype(np.float64)
                max_d = min(self.max_delay_samples, self.seg_size // 2)
                region = np.abs(gcc[: max_d + 1])
                # Top-3 with greedy 16-sample lobe suppression.
                tmp = region.copy()
                for _ in range(3):
                    idx = int(np.argmax(tmp))
                    height = float(tmp[idx])
                    if height <= 0.0:
                        break
                    # PAR vs full-region mean excluding this sample.
                    peak_v = float(region[idx])
                    mean_excl = (float(region.sum()) - peak_v) / (
                        len(region) - 1 + 1e-10
                    )
                    par = float(peak_v / (mean_excl + 1e-10))
                    top3.append((idx, height, par))
                    lo = max(0, idx - 16)
                    hi = min(len(tmp), idx + 17)
                    tmp[lo:hi] = -1.0
                if top3:
                    par_top1 = top3[0][2]
            row = {
                "call_idx": self._trace_call_idx,
                "frame_samples": n,
                "time_s": self._trace_total_samples / sr,
                "mic_pwr_pre": mic_pwr_pre,
                "ref_pwr_pre": ref_pwr_pre,
                "n_updates_pre": n_updates_pre,
                "n_updates_post": self._n_updates,
                "samples_accumulated": self._samples_accumulated,
                "in_init_window": self._samples_accumulated < self._init_samples,
                "init_done": self._init_done,
                "did_estimate": did_estimate,
                "estimated_delay": int(self.estimated_delay),
                "last_par": float(self._last_par),
                "confidence": float(self.confidence),
                "is_solid": bool(self.is_solid),
                "top1_lag": top3[0][0] if len(top3) >= 1 else -1,
                "top1_height": top3[0][1] if len(top3) >= 1 else 0.0,
                "top1_par": top3[0][2] if len(top3) >= 1 else 0.0,
                "top2_lag": top3[1][0] if len(top3) >= 2 else -1,
                "top2_height": top3[1][1] if len(top3) >= 2 else 0.0,
                "top2_par": top3[1][2] if len(top3) >= 2 else 0.0,
                "top3_lag": top3[2][0] if len(top3) >= 3 else -1,
                "top3_height": top3[2][1] if len(top3) >= 3 else 0.0,
                "top3_par": top3[2][2] if len(top3) >= 3 else 0.0,
            }
            self._trace_rows.append(row)
            self._trace_call_idx += 1
            self._trace_total_samples += n

        return ret

    def _update_cross_spectrum(self):
        """Update smoothed cross-spectrum from current segment."""
        mic_spec = np.fft.rfft(self._mic_buf)
        ref_spec = np.fft.rfft(self._ref_buf)
        cross = mic_spec * np.conj(ref_spec)
        self._n_updates += 1
        if self._n_updates == 1:
            self._cross_spec = cross.copy()
        else:
            self._cross_spec = self._alpha * self._cross_spec + (1 - self._alpha) * cross

    def _estimate(self):
        """Estimate delay from accumulated cross-spectrum using GCC-PHAT."""
        magnitude = np.abs(self._cross_spec) + 1e-10
        phat = self._cross_spec / magnitude
        gcc = np.fft.irfft(phat, n=self.seg_size)

        max_d = min(self.max_delay_samples, self.seg_size // 2)

        # Search positive delays (mic lags ref — normal case)
        pos_range = gcc[:max_d + 1]
        best_pos = np.argmax(np.abs(pos_range))

        # A5: PAR (Peak-to-Average Ratio) confidence
        peak = float(np.abs(gcc[best_pos]))
        mean_excl = (np.sum(np.abs(pos_range)) - peak) / (len(pos_range) - 1 + 1e-10)
        self._last_par = float(peak / (mean_excl + 1e-10))

        # P3c Phase 1a: remember the previous estimate's lag before
        # overwriting, so the fast-path can require lag stability across
        # two consecutive estimates.
        self._prev_estimated_delay = self.estimated_delay
        self.estimated_delay = best_pos
        self._samples_since_est = 0
        self._n_estimates += 1

    @property
    def confidence(self) -> float:
        """[0, 1] — 0 = no evidence, 1 = solid PAR ≥ par_solid_threshold.

        Used by AEC mu_scale floor and RES conservative-mode gating. v3.10.0:
        promoted from internal _last_par to first-class API.

        P3c Phase 1a: when `fast_path_enabled`, allow the gate to clear at
        n_updates >= 2 if the PAR is overwhelmingly above the solid
        threshold AND the same lag was reported by the previous estimate.
        Both guards are required; a single high-PAR sample alone does not
        promote (rules out spurious peaks). Default off — opt-in.
        """
        if self.estimated_delay < 0:
            return 0.0
        par = self._last_par
        if (self.fast_path_enabled
                and self._n_updates >= 2
                and par >= self.fast_par_threshold
                and self._prev_estimated_delay == self.estimated_delay
                and self._prev_estimated_delay >= 0):
            return 1.0
        if self._n_updates < 3:
            return 0.0
        lo = self.par_low_threshold
        hi = self.par_solid_threshold
        if par <= lo:
            return 0.0
        if par >= hi:
            return 1.0
        return float((par - lo) / (hi - lo))

    @property
    def is_solid(self) -> bool:
        """Convenience: True when confidence is at the strong threshold."""
        return self.confidence >= 1.0
