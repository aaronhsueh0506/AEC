"""Render signal analyzer: tracks narrow-band (tonal) render regions and strong narrow peaks.

``mask_regions_around_narrow_bands(mu)`` zeroes mu bins around sustained peaks;
``poor_signal_excitation()`` is true when any narrow-band counter exceeds the
poor-excitation threshold (used to freeze the filter update). Peak detection is
ratio-based; the strong-peak amplitude floor is 100 in int16 units (3.05e-3 float).
"""
from typing import Optional

import numpy as np


_COUNTER_THRESHOLD = 5  # mask fires when a counter exceeds this many frames
_POOR_EXCITATION_THRESHOLD = 10  # poor excitation when a counter exceeds this
_MAX_ABS_THRESHOLD_INT16 = 100.0
_MAX_ABS_THRESHOLD_FLOAT = _MAX_ABS_THRESHOLD_INT16 / 32768.0
_PEAK_RATIO_NARROW = 3.0  # X^2[k] > 3 x max(neighbours)
_PEAK_RATIO_STRONG = 100.0  # strong peak: peak level > 100 x non-peak


class RenderSignalAnalyzer:
    """Per-hop render-signal analyser.

    ``update(render_psd, render_block)``: ``render_psd`` is the delay-aligned |X|^2
    (length n_freqs = fft_size/2 + 1), or None for "no delay estimate" (counters reset);
    ``render_block`` is the time-domain render hop for the strong-peak amplitude check,
    or None to skip it.
    """

    def __init__(self, n_freqs: int, strong_peak_freeze_duration: int = 6,
                 counter_threshold_hops: int = _COUNTER_THRESHOLD,
                 poor_excitation_threshold_hops: int = _POOR_EXCITATION_THRESHOLD
                 ) -> None:
        self._n_freqs = int(n_freqs)
        self._n_freqs_by2 = self._n_freqs - 1
        if self._n_freqs_by2 < 3:
            raise ValueError(f"n_freqs={n_freqs} too small for narrow-band detect")
        self._counters = np.zeros(self._n_freqs_by2 - 1, dtype=np.int64)
        self._strong_peak_freeze_duration = int(strong_peak_freeze_duration)
        self._narrow_peak_band: Optional[int] = None
        self._narrow_peak_counter = 0
        # Persistence thresholds in hops; the caller may pass wall-clock-retimed counts.
        self._counter_threshold = int(counter_threshold_hops)
        self._poor_excitation_threshold = int(poor_excitation_threshold_hops)

    def _identify_small_narrow_band_regions(
        self, render_psd: Optional[np.ndarray]
    ) -> None:
        """Per-bin counters of consecutive frames where X2[k] > 3 x max(X2[k-1], X2[k+1])."""
        if render_psd is None:
            self._counters.fill(0)
            return
        X2 = render_psd
        k_lo, k_hi = 1, self._n_freqs_by2  # exclusive upper
        x_center = X2[k_lo:k_hi]
        x_left = X2[k_lo - 1:k_hi - 1]
        x_right = X2[k_lo + 1:k_hi + 1]
        is_peak = x_center > _PEAK_RATIO_NARROW * np.maximum(x_left, x_right)
        self._counters = np.where(is_peak, self._counters + 1, 0).astype(np.int64)

    def _identify_strong_narrow_band_component(
        self,
        render_psd: Optional[np.ndarray],
        render_block: Optional[np.ndarray],
    ) -> None:
        """Track the strongest narrow peak with a freeze duration."""
        if self._narrow_peak_band is not None:
            self._narrow_peak_counter += 1
            if self._narrow_peak_counter > self._strong_peak_freeze_duration:
                self._narrow_peak_band = None
                self._narrow_peak_counter = 0
        if render_psd is None or render_block is None:
            return
        X2 = render_psd
        peak_bin = int(np.argmax(X2))
        lo_lo = max(0, peak_bin - 14)
        lo_hi = max(lo_lo, peak_bin - 4)
        hi_lo = min(self._n_freqs, peak_bin + 5)
        hi_hi = min(self._n_freqs, peak_bin + 15)
        non_peak_power = 0.0
        if lo_hi > lo_lo:
            non_peak_power = max(non_peak_power, float(np.max(X2[lo_lo:lo_hi])))
        if hi_hi > hi_lo:
            non_peak_power = max(non_peak_power, float(np.max(X2[hi_lo:hi_hi])))
        max_abs = float(np.max(np.abs(render_block)))
        peak_level = float(X2[peak_bin])
        if (peak_bin > 0
                and max_abs > _MAX_ABS_THRESHOLD_FLOAT
                and peak_level > _PEAK_RATIO_STRONG * non_peak_power):
            self._narrow_peak_band = peak_bin
            self._narrow_peak_counter = 0

    def update(
        self,
        render_psd: Optional[np.ndarray],
        render_block: Optional[np.ndarray] = None,
    ) -> None:
        """Per-hop update; ``render_psd`` / ``render_block`` may be None during cold start."""
        self._identify_small_narrow_band_regions(render_psd)
        self._identify_strong_narrow_band_component(render_psd, render_block)

    def poor_signal_excitation(self) -> bool:
        """True if any narrow-band counter exceeds the poor-excitation threshold."""
        if self._counters.size == 0:
            return False
        return bool(np.any(self._counters > self._poor_excitation_threshold))

    def narrow_peak_band(self) -> Optional[int]:
        return self._narrow_peak_band

    def mask_regions_around_narrow_bands(self, mu: np.ndarray) -> np.ndarray:
        """Zero mu in place for +-2 bins around any peak whose counter exceeds the threshold
        (edge bins mask 2 bins inward); returns ``mu``."""
        if mu.shape[0] != self._n_freqs:
            raise ValueError(
                f"mu has {mu.shape[0]} bins, expected n_freqs={self._n_freqs}"
            )
        N = self._n_freqs_by2  # = n_freqs - 1
        thr = self._counter_threshold
        if self._counters[0] > thr:
            mu[0] = 0.0
            mu[1] = 0.0
        for k in range(2, N - 1):
            if self._counters[k - 1] > thr:
                mu[k - 2:k + 3] = 0.0
        if self._counters[N - 2] > thr:
            mu[N - 1] = 0.0
            mu[N] = 0.0
        return mu

    def reset(self) -> None:
        self._counters.fill(0)
        self._narrow_peak_band = None
        self._narrow_peak_counter = 0
