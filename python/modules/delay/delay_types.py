"""Delay-subsystem value types (frozen dataclasses): DelayEstimate, EchoPathVariability."""
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
