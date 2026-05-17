"""Time-constant constants for AEC3 AecState ports.

AEC3 counter thresholds expressed as ``X * kNumBlocksPerSecond`` where
``kNumBlocksPerSecond = 250`` (250 blocks of 4 ms = 1 s). In our port
``AecState.update`` ticks per OUR 10 ms hop, so the same X seconds maps
to ``X * HOPS_PER_SECOND`` updates.

Keep the AEC3 source comment alongside the rescaled value:
  ``# AEC3 kNumBlocksPerSecond * X = N blocks (~Y ms) -> our M hops``
"""

HOPS_PER_SECOND = 100  # 10 ms hop @ 16 kHz; helper for AEC3-second thresholds
