"""Delay estimator adapter exposing the orchestrator's ``DelayEstimator`` interface
(``confidence``, ``is_solid``, ``estimated_delay``, ``accumulate``, ``reset``,
``_n_updates``, ``_period_samples``, ``_alpha``) on top of ``EchoPathDelayEstimator``.

The delay is the raw-sample value from ``estimate_delay()``, not quantised to
hops. Confidence: no estimate 0.0, COARSE 0.5, REFINED 1.0.
``_period_samples`` and ``_alpha`` are accepted and ignored.
"""
from typing import Optional

import numpy as np

from .delay_types import DelayAdjustment, DelayEstimate, DelayQuality, EchoPathVariability
from .echo_path_delay_estimator import EchoPathDelayEstimator


class LegacyDelayShim:
    def __init__(self, sample_rate: int, hop_size: int, **legacy_kwargs) -> None:
        self._sample_rate = int(sample_rate)
        self._hop_size = int(hop_size)
        # Extra kwargs are accepted and ignored; ``num_filters`` (matched-filter bank size) is forwarded.
        self._legacy_kwargs = legacy_kwargs

        num_filters = int(legacy_kwargs.get('num_filters', 5))
        if not 1 <= num_filters <= 5:
            # Same bank-size limit as AecConfig and the C path.
            raise ValueError(
                f"num_filters must be in [1, 5], got {num_filters}"
            )
        self._estimator = EchoPathDelayEstimator(
            sample_rate=self._sample_rate,
            num_filters=num_filters,
        )
        self._latest_estimate: Optional[DelayEstimate] = None  # raw (input-rate) samples
        self._estimate_count = 0
        self._latest_variability: Optional[EchoPathVariability] = None
        self._pending_flush = False

        # No-op attributes kept for caller compatibility.
        self._period_samples = 0
        self._alpha = 0.6

    # ------------------------------------------------------------------- public
    def reset(self) -> None:
        self._estimator.reset(reset_delay_confidence=True)
        self._latest_estimate = None
        self._estimate_count = 0
        self._latest_variability = None
        self._pending_flush = True

    def accumulate(self, near: np.ndarray, far: np.ndarray) -> bool:
        """Process one hop; True iff a new delay value was emitted."""
        estimate = self._estimator.estimate_delay(far, near)
        prev_delay = self._latest_estimate.delay if self._latest_estimate else -1
        emitted_new = False
        if estimate is not None:
            self._latest_estimate = estimate
            # Counts hops where any estimate is available (consumers gate on >= 3).
            self._estimate_count += 1
            if estimate.delay != prev_delay:
                emitted_new = True
        if self._pending_flush:
            delay_change = DelayAdjustment.BUFFER_FLUSH
            self._pending_flush = False
        elif emitted_new:
            delay_change = DelayAdjustment.NEW_DETECTED_DELAY
        else:
            delay_change = DelayAdjustment.NONE
        self._latest_variability = EchoPathVariability(
            gain_change=False,
            delay_change=delay_change,
            clock_drift=self.has_clockdrift(),
        )
        return emitted_new

    @property
    def estimated_delay(self) -> int:
        if self._latest_estimate is None:
            return -1
        return int(self._latest_estimate.delay)

    @property
    def confidence(self) -> float:
        if self._latest_estimate is None:
            return 0.0
        if self._latest_estimate.quality is DelayQuality.REFINED:
            return 1.0
        if self._latest_estimate.quality is DelayQuality.COARSE:
            return 0.5
        return 0.0

    @property
    def is_solid(self) -> bool:
        return self.confidence >= 1.0

    @property
    def _n_updates(self) -> int:
        return self._estimate_count

    @property
    def latest_variability(self) -> Optional[EchoPathVariability]:
        return self._latest_variability

    def has_clockdrift(self) -> bool:
        from .clockdrift_detector import ClockdriftLevel
        return self._estimator.clockdrift_level() is not ClockdriftLevel.NONE
