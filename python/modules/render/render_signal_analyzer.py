"""Render signal analyzer — verbatim port of AEC3
docs/aec3_extracts/src/aec3/render_signal_analyzer.{cc,h}.

Used by the refined / coarse filter gain compute to:

1. Detect narrow-band (tonal) render regions and zero the corresponding mu
   bins via ``mask_regions_around_narrow_bands(mu)`` (AEC3 cc:143-159).
2. Report ``poor_signal_excitation()`` when any narrow-band counter has been
   sustained for > 10 frames — used to freeze the W update entirely
   (AEC3 cc:96-99 → refined_filter_update_gain.cc startup gate).
3. Report ``narrow_peak_band()`` for diagnostic / further use.

Float-scale conversion: peak detection uses ratios (unitless) so no constant
scaling. The "max_abs > 100" threshold in the strong-narrow-band path is
int16 amplitude; converted to float via /32768 = 3.05e-3.

Configurable for any FFT size; the AEC3 hard-coded kFftLengthBy2 = 64 becomes
runtime ``n_freqs - 1`` (where ``n_freqs = fft_size/2 + 1``).
"""
from typing import Optional

import numpy as np


_COUNTER_THRESHOLD = 5  # AEC3 kCounterThreshold (mask fires when > 5 frames)
_POOR_EXCITATION_THRESHOLD = 10  # AEC3 narrow_band_counter > 10 → poor_excitation
_MAX_ABS_THRESHOLD_INT16 = 100.0  # AEC3 strong-narrow-band amplitude floor
_MAX_ABS_THRESHOLD_FLOAT = _MAX_ABS_THRESHOLD_INT16 / 32768.0   # 3.05e-3
_PEAK_RATIO_NARROW = 3.0           # AEC3 X²[k] > 3 × max(neighbors)
_PEAK_RATIO_STRONG = 100.0         # AEC3 strong-peak: peak_level > 100 × non_peak


class RenderSignalAnalyzer:
    """Per-hop render-signal analyser. Track narrow-band tonal regions in the
    delay-aligned render spectrum + detect dominant narrow peaks.

    Inputs (per call to ``update``):
        render_psd     : per-bin power spectrum |X|² of the (delay-aligned)
                         render block. Length = n_freqs = fft_size/2 + 1.
                         May be None to signal "no delay estimate yet" →
                         counters reset.
        render_block   : optional time-domain render samples for the strong-
                         narrow-band detector's max_abs check. May be None
                         to skip strong-peak detection.

    Outputs (read via methods):
        poor_signal_excitation() -> bool
        narrow_peak_band()       -> Optional[int]
        mask_regions_around_narrow_bands(mu)  -> in-place mu zeroing
    """

    def __init__(self, n_freqs: int, strong_peak_freeze_duration: int = 6) -> None:
        # AEC3 array has size kFftLengthBy2 - 1 (= n_freqs - 2). Indexed by k-1
        # for bins k in [1, kFftLengthBy2 - 1).
        self._n_freqs = int(n_freqs)
        self._n_freqs_by2 = self._n_freqs - 1  # kFftLengthBy2 in AEC3
        if self._n_freqs_by2 < 3:
            raise ValueError(f"n_freqs={n_freqs} too small for narrow-band detect")
        # Per-bin counter, length kFftLengthBy2 - 1 (bins 1..n_freqs-2).
        self._counters = np.zeros(self._n_freqs_by2 - 1, dtype=np.int64)
        # Strong-peak state.
        self._strong_peak_freeze_duration = int(strong_peak_freeze_duration)
        self._narrow_peak_band: Optional[int] = None
        self._narrow_peak_counter = 0

    # -------------------------------------------------------- AEC3 detectors
    def _identify_small_narrow_band_regions(
        self, render_psd: Optional[np.ndarray]
    ) -> None:
        """Mirrors IdentifySmallNarrowBandRegions (cc:33-59).
        Single-channel: channel_counters trivially = 0 or 1 per bin."""
        if render_psd is None:
            # AEC3: no delay → zero all counters.
            self._counters.fill(0)
            return
        X2 = render_psd
        # For k in [1, kFftLengthBy2): check X2[k] > 3 * max(X2[k-1], X2[k+1])
        # We iterate bins 1..n_freqs_by2-1 (inclusive), array index k-1.
        k_lo, k_hi = 1, self._n_freqs_by2  # exclusive upper
        # Vectorise: build view of [k_lo..k_hi), check neighbour condition.
        x_center = X2[k_lo:k_hi]
        x_left = X2[k_lo - 1:k_hi - 1]
        x_right = X2[k_lo + 1:k_hi + 1]
        is_peak = x_center > _PEAK_RATIO_NARROW * np.maximum(x_left, x_right)
        # Update counters: if peak this frame → +1, else reset to 0.
        # (channel_counters > 0 ? counter+1 : 0 — for single channel, equivalent.)
        self._counters = np.where(is_peak, self._counters + 1, 0).astype(np.int64)

    def _identify_strong_narrow_band_component(
        self,
        render_psd: Optional[np.ndarray],
        render_block: Optional[np.ndarray],
    ) -> None:
        """Mirrors IdentifyStrongNarrowBandComponent (cc:62-121)."""
        # Tick down freeze counter — if expired, clear narrow_peak_band.
        if self._narrow_peak_band is not None:
            self._narrow_peak_counter += 1
            if self._narrow_peak_counter > self._strong_peak_freeze_duration:
                self._narrow_peak_band = None
                self._narrow_peak_counter = 0
        if render_psd is None or render_block is None:
            return
        X2 = render_psd
        peak_bin = int(np.argmax(X2))
        # Non-peak power in two windows around peak (avoiding ±5 bins).
        lo_lo = max(0, peak_bin - 14)
        lo_hi = max(lo_lo, peak_bin - 4)
        hi_lo = min(self._n_freqs, peak_bin + 5)
        hi_hi = min(self._n_freqs, peak_bin + 15)
        non_peak_power = 0.0
        if lo_hi > lo_lo:
            non_peak_power = max(non_peak_power, float(np.max(X2[lo_lo:lo_hi])))
        if hi_hi > hi_lo:
            non_peak_power = max(non_peak_power, float(np.max(X2[hi_lo:hi_hi])))
        # max_abs of render time block (float scale).
        max_abs = float(np.max(np.abs(render_block)))
        peak_level = float(X2[peak_bin])
        if (peak_bin > 0
                and max_abs > _MAX_ABS_THRESHOLD_FLOAT
                and peak_level > _PEAK_RATIO_STRONG * non_peak_power):
            self._narrow_peak_band = peak_bin
            self._narrow_peak_counter = 0

    # --------------------------------------------------------------- public
    def update(
        self,
        render_psd: Optional[np.ndarray],
        render_block: Optional[np.ndarray] = None,
    ) -> None:
        """Per-hop update.

        ``render_psd`` should be the delay-aligned partition's |X|² (passing
        the current-frame X² when no delay is locked is acceptable, mirrors
        the AEC3 ``delay_partitions=None`` case → counters reset).
        ``render_block`` is the time-domain render hop (for strong-peak's
        max_abs gate). Both may be None during cold-start.
        """
        self._identify_small_narrow_band_regions(render_psd)
        self._identify_strong_narrow_band_component(render_psd, render_block)

    def poor_signal_excitation(self) -> bool:
        """True iff ANY narrow-band counter > 10 (AEC3 cc:40-45)."""
        if self._counters.size == 0:
            return False
        return bool(np.any(self._counters > _POOR_EXCITATION_THRESHOLD))

    def narrow_peak_band(self) -> Optional[int]:
        return self._narrow_peak_band

    def mask_regions_around_narrow_bands(self, mu: np.ndarray) -> np.ndarray:
        """In-place mask: zero mu for ±2 bins around any peak with counter > 5
        (AEC3 cc:143-159). Returns the same array for chaining convenience.

        Bin edges are special-cased to mirror AEC3 (cc:148-149, 156-157):
          - counter[0]      > threshold  →  zero mu[0], mu[1]
          - counter[k-1]    > threshold  →  zero mu[k-2..k+2] (k in [2, N-1))
          - counter[N-2]    > threshold  →  zero mu[N-1], mu[N]   where N = n_freqs_by2
        """
        if mu.shape[0] != self._n_freqs:
            raise ValueError(
                f"mu has {mu.shape[0]} bins, expected n_freqs={self._n_freqs}"
            )
        N = self._n_freqs_by2  # = n_freqs - 1
        # Edge: counter[0] (bin k=1) → mask bins 0, 1.
        if self._counters[0] > _COUNTER_THRESHOLD:
            mu[0] = 0.0
            mu[1] = 0.0
        # Interior: counter[k-1] (bin k, k in [2, N-1)) → mask bins k-2..k+2.
        for k in range(2, N - 1):
            if self._counters[k - 1] > _COUNTER_THRESHOLD:
                mu[k - 2:k + 3] = 0.0
        # Edge: counter[N-2] (bin k=N-1) → mask bins N-1, N (=n_freqs-1).
        if self._counters[N - 2] > _COUNTER_THRESHOLD:
            mu[N - 1] = 0.0
            mu[N] = 0.0
        return mu

    # ----------------------------------------------------------------- reset
    def reset(self) -> None:
        self._counters.fill(0)
        self._narrow_peak_band = None
        self._narrow_peak_counter = 0
