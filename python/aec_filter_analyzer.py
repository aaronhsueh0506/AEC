"""v3.18 Phase C.A — FilterAnalyzer (audit-only port of AEC3 filter_analyzer.h).

Operates on the main PBFDKF filter's time-domain impulse response
(IFFT of W summed across partitions, HP-filtered at 600 Hz).
Outputs `consistent_estimate` (bool) + `max_echo_path_gain` (float) +
`peak_index` (int) for downstream FilteringQualityAnalyzer (C.B).

Audit-only in C.A: no consumer reads outputs; diagnostic surface only.

Design lock: docs/v3_18_c_a1_filter_analyzer_design.md.
"""
from collections import deque
import numpy as np


class FilterAnalyzer:
    """AEC3-aligned filter-shape consistency + peak/gain analyzer."""

    def __init__(self, sample_rate: int, hp_cutoff_hz: float = 600.0,
                 peak_window: int = 5, consistent_threshold: int = 30):
        self.sample_rate = sample_rate
        self._hp_alpha = float(np.exp(-2.0 * np.pi * hp_cutoff_hz / sample_rate))
        self._peak_window = peak_window
        self._consistent_threshold = consistent_threshold
        self._peak_history = deque(maxlen=consistent_threshold)
        # IIR state for HP across frames (not reset per frame so the filter
        # converges; reset only on `reset()`).
        self._hp_prev_x = 0.0
        self._hp_prev_y = 0.0
        self.consistent_estimate = False
        self.max_echo_path_gain = 0.0
        self.peak_index = -1

    def update(self, w_time: np.ndarray) -> None:
        h_hp = self._apply_hp(w_time)
        abs_h = np.abs(h_hp)
        self.peak_index = int(np.argmax(abs_h))
        self.max_echo_path_gain = float(abs_h[self.peak_index])
        self._peak_history.append(self.peak_index)
        if len(self._peak_history) >= self._consistent_threshold:
            recent = list(self._peak_history)
            ref = recent[0]
            stable = all(abs(p - ref) <= self._peak_window for p in recent)
            self.consistent_estimate = stable

    def _apply_hp(self, x: np.ndarray) -> np.ndarray:
        # 1st-order HP IIR: y[n] = α·(y[n-1] + x[n] - x[n-1])
        # Vectorised via scipy.signal.lfilter for performance.
        from scipy.signal import lfilter
        alpha = self._hp_alpha
        b = [alpha, -alpha]
        a = [1.0, -alpha]
        zi = np.array([alpha * (self._hp_prev_y - self._hp_prev_x)],
                      dtype=np.float64)
        y, zf = lfilter(b, a, x.astype(np.float64), zi=zi)
        # Carry IIR state across frames so HP cutoff is consistent.
        self._hp_prev_x = float(x[-1])
        self._hp_prev_y = float(y[-1])
        return y.astype(np.float32)

    def reset(self) -> None:
        self._peak_history.clear()
        self._hp_prev_x = 0.0
        self._hp_prev_y = 0.0
        self.consistent_estimate = False
        self.max_echo_path_gain = 0.0
        self.peak_index = -1
