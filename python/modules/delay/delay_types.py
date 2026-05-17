"""Delay-subsystem POD types — DelayEstimate + EchoPathVariability.

Mirrors:
  - docs/aec3_extracts/src/aec3/delay_estimate.h
  - docs/aec3_extracts/src/aec3/echo_path_variability.{cc,h}

Both are dataclass-frozen; structural-equality comparable so downstream
state machinery can do change detection cheaply.
"""
from dataclasses import dataclass
from enum import Enum


class DelayQuality(Enum):
    COARSE = 0
    REFINED = 1


@dataclass(frozen=True)
class DelayEstimate:
    quality: DelayQuality
    delay: int  # samples


class DelayAdjustment(Enum):
    NONE = 0
    BUFFER_FLUSH = 1
    NEW_DETECTED_DELAY = 2


@dataclass(frozen=True)
class EchoPathVariability:
    gain_change: bool
    delay_change: DelayAdjustment
    clock_drift: bool

    def audio_path_changed(self) -> bool:
        return self.gain_change or self.delay_change is not DelayAdjustment.NONE
