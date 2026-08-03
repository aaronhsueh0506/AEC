"""Compatibility shim: exposes legacy DelayEstimator API on top of the
AEC3-aligned EchoPathDelayEstimator.

Drop-in replacement for ``legacy_delay.DelayEstimator`` at orchestrator.py:374.
The orchestrator has ~30 call sites referencing ``self.delay_est`` via the
legacy API (``.confidence``, ``.is_solid``, ``.estimated_delay``,
``.accumulate``, ``.reset``, ``._n_updates``, plus mutable ``._period_samples``
and ``._alpha``). Rather than rewrite every site, this shim adapts the AEC3
estimator to that surface.

Design notes:
  * We bypass ``RenderDelayController`` because it quantises the delay to
    AEC3 4 ms blocks (or our 10 ms hops via the port), losing the sub-hop
    sample precision the orchestrator's ring buffer needs (the filter taps
    align to raw-sample resolution, not hop boundaries). Calling
    ``EchoPathDelayEstimator.estimate_delay()`` directly returns the raw
    16 kHz-sample delay (post ``_DOWN_SAMPLING_FACTOR=4`` multiplication
    inside the estimator).
  * Confidence mapping: None→0.0, COARSE→0.5, REFINED→1.0 — matches the
    orchestrator's ≥0.3 / ≥0.5 / is_solid gates.
  * The legacy EPC fast-cadence path writes ``._period_samples`` and
    ``._alpha``; AEC3 matched filter is always-on per 4 ms inner block,
    so these are accepted as no-op compat attributes.
  * ``latest_variability`` synthesises ``EchoPathVariability`` from the
    delay change for future migrations that want the AEC3-native event.
"""
from typing import Optional

import numpy as np

from .delay_types import DelayAdjustment, DelayEstimate, DelayQuality, EchoPathVariability
from .echo_path_delay_estimator import EchoPathDelayEstimator


class LegacyDelayShim:
    def __init__(self, sample_rate: int, hop_size: int, **legacy_kwargs) -> None:
        self._sample_rate = int(sample_rate)
        self._hop_size = int(hop_size)
        # Legacy kwargs accepted for call-site compat; ignored by the AEC3
        # estimator (which carries its own internal smoothing + cadence).
        self._legacy_kwargs = legacy_kwargs

        self._estimator = EchoPathDelayEstimator(sample_rate=self._sample_rate)
        self._latest_estimate: Optional[DelayEstimate] = None  # raw 16 kHz samples
        self._estimate_count = 0
        self._latest_variability: Optional[EchoPathVariability] = None
        self._pending_flush = False

        # Legacy EPC fast-cadence writes these. No-op compat attributes.
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
        """Process one hop. Returns True iff a NEW delay value was emitted
        this hop (mirrors legacy ``DelayEstimator.accumulate`` return)."""
        estimate = self._estimator.estimate_delay(far, near)
        prev_delay = self._latest_estimate.delay if self._latest_estimate else -1
        emitted_new = False
        if estimate is not None:
            self._latest_estimate = estimate
            # _estimate_count counts hops where ANY estimate is available,
            # matching the legacy `_n_updates` semantic (`>= 3` gate at
            # orchestrator.py:1644). Legacy incremented per segment-update;
            # AEC3 matched filter updates every inner block (>=2 per hop),
            # so a per-hop increment is the equivalent guard.
            self._estimate_count += 1
            if estimate.delay != prev_delay:
                emitted_new = True
        # Synthesise EchoPathVariability for downstream consumers.
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
