"""SubtractiveNLP — voice-band autocorrelation pitch tracker + NL substrate.

Extracted from ``aec.py`` during refactor R.10. Default-OFF substrate
from the v3.13 E4 NLP arc (amplitude-mask family closed CANNOT SHIP
2026-05-14; detector preserved for future Volterra integration).

Self-contained: numpy.
"""
import numpy as np


class SubtractiveNLP:
    """E4 NLP detector — voice-band autocorrelation pitch tracker.

    Audit-only in v3.13 E4.S3 (no output mutation). Per
    docs/v3_13_e4_s2_design_lock.md: detects mic-side non-linear
    residual (loudspeaker NL + transmission codec NL) via FFT-based
    autocorrelation peak in 80-500 Hz fundamental band, gated on
    filter_state == 'refined_usable' and far_active.

    Outputs per-hop scalar `nl_confidence` ∈ [0, 1]; downstream
    suppressor (S5+) will consume.
    """

    def __init__(self, sample_rate: int, hop_size: int,
                 window_ms: float = 32.0,
                 pitch_threshold: float = 0.45,
                 continuity_frames: int = 3,
                 history_len: int = 4,
                 min_residual_rms: float = 0.05,
                 cancellation_ratio_threshold: float = 2.0,
                 cancellation_ema_alpha: float = 0.99):
        self._sr = int(sample_rate)
        self._hop = int(hop_size)
        win = int(window_ms * sample_rate / 1000)
        n = 1
        while n < win:
            n *= 2
        self._win_samples = n
        self._qmin = int(0.002 * sample_rate)
        self._qmax = int(0.0125 * sample_rate)
        self._pitch_threshold = float(pitch_threshold)
        self._continuity_frames = int(continuity_frames)
        self._history_len = int(history_len)
        self._min_residual_rms = float(min_residual_rms)
        # S4.1 cancellation gate — "filter is canceling something" signal.
        # NE bucket: ratio ≈ 1 (filter doesn't cancel because no echo).
        # FS converged: ratio > threshold (mic energy reduced by filter).
        self._cancel_ratio_thr = float(cancellation_ratio_threshold)
        self._cancel_ema_alpha = float(cancellation_ema_alpha)
        self._mic_rms_ema = 0.0
        self._raw_rms_ema = 0.0
        self._cancel_ratio_last = 0.0
        self._window = np.hanning(self._win_samples).astype(np.float32)
        self._buf = np.zeros(self._win_samples, dtype=np.float32)
        self._buf_pos = 0
        self._buf_filled = 0
        self._pitch_lag_hist = []
        self._nl_confidence_last = 0.0
        self._pitch_strength_last = 0.0
        self._pitch_lag_last = 0
        self._fire_count = 0
        self._call_count = 0

    def _append_hop(self, hop_samples: np.ndarray) -> None:
        n = len(hop_samples)
        end = self._buf_pos + n
        if end <= self._win_samples:
            self._buf[self._buf_pos:end] = hop_samples
        else:
            tail = self._win_samples - self._buf_pos
            self._buf[self._buf_pos:] = hop_samples[:tail]
            self._buf[:n - tail] = hop_samples[tail:]
        self._buf_pos = (self._buf_pos + n) % self._win_samples
        self._buf_filled = min(self._buf_filled + n, self._win_samples)

    def _compute_pitch_strength(self) -> tuple:
        if self._buf_pos == 0:
            frame = self._buf
        else:
            frame = np.concatenate(
                (self._buf[self._buf_pos:], self._buf[:self._buf_pos]))
        x = frame * self._window
        n2 = 2 * self._win_samples
        spec = np.fft.rfft(x, n=n2)
        ac = np.fft.irfft(np.abs(spec) ** 2, n=n2)[:self._win_samples]
        if ac[0] < 1e-15:
            return 0.0, 0
        ac = ac / ac[0]
        slab = ac[self._qmin: self._qmax + 1]
        peak_idx = int(np.argmax(slab))
        return float(slab[peak_idx]), self._qmin + peak_idx

    def process(self, hop_samples: np.ndarray,
                filter_state: str, far_active: bool,
                mic_hop_samples: 'Optional[np.ndarray]' = None) -> float:
        self._call_count += 1
        if len(hop_samples) != self._hop:
            self._nl_confidence_last = 0.0
            return 0.0
        self._append_hop(hop_samples.astype(np.float32))
        # S4.1 cancellation-ratio EMA: track mic_rms / raw_output_rms.
        # Updated every hop regardless of gating so EMA reflects sustained
        # filter activity. When NE bucket (no echo to cancel), filter
        # doesn't reduce mic energy → ratio ≈ 1. When FS converged,
        # ratio ≫ 1.
        hop_rms = float(np.sqrt(
            np.mean(hop_samples.astype(np.float64) ** 2) + 1e-15))
        if mic_hop_samples is not None:
            mic_rms = float(np.sqrt(
                np.mean(mic_hop_samples.astype(np.float64) ** 2) + 1e-15))
            a = self._cancel_ema_alpha
            self._mic_rms_ema = a * self._mic_rms_ema + (1.0 - a) * mic_rms
            self._raw_rms_ema = a * self._raw_rms_ema + (1.0 - a) * hop_rms
            if self._raw_rms_ema > 1e-9:
                self._cancel_ratio_last = self._mic_rms_ema / self._raw_rms_ema
            else:
                self._cancel_ratio_last = 0.0
        # Gate: allow refined_usable OR coarse_learning — heavy-NL cases
        # never reach refined_usable because the NL itself prevents linear
        # filter convergence. Excluded: idle / startup / diverged /
        # suspicious_dt (DT protection — NE voice could mimic pitched NL).
        if (filter_state not in ('refined_usable', 'coarse_learning')
                or not far_active
                or self._buf_filled < self._win_samples):
            self._nl_confidence_last = 0.0
            self._pitch_lag_hist.clear()
            return 0.0
        # S3.1 secondary gate: minimum residual RMS on the current hop.
        # Filters out pitched-but-low-residual frames in clean cases.
        if hop_rms < self._min_residual_rms:
            self._nl_confidence_last = 0.0
            return 0.0
        # S4.1 cancellation-ratio gate: filter must be canceling something
        # (mic_rms_ema / raw_rms_ema > threshold). NE bucket fails this
        # because filter has nothing to cancel.
        if (mic_hop_samples is not None
                and self._cancel_ratio_last < self._cancel_ratio_thr):
            self._nl_confidence_last = 0.0
            return 0.0
        pitch_strength, pitch_lag = self._compute_pitch_strength()
        self._pitch_strength_last = pitch_strength
        self._pitch_lag_last = pitch_lag
        is_pitched = pitch_strength >= self._pitch_threshold
        continuity_passed = False
        if is_pitched and self._pitch_lag_hist:
            tol = 0.05 * pitch_lag
            in_tol = sum(1 for prev in self._pitch_lag_hist
                         if abs(prev - pitch_lag) <= tol)
            required = min(self._continuity_frames, len(self._pitch_lag_hist))
            continuity_passed = in_tol >= required
        if is_pitched and continuity_passed:
            x_log = (pitch_strength - self._pitch_threshold) * 4.0
            nl_confidence = float(1.0 / (1.0 + np.exp(-x_log)))
            self._fire_count += 1
        else:
            nl_confidence = 0.0
        self._pitch_lag_hist.append(pitch_lag)
        if len(self._pitch_lag_hist) > self._history_len:
            self._pitch_lag_hist.pop(0)
        self._nl_confidence_last = nl_confidence
        return nl_confidence
