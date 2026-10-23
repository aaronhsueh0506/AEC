"""Block / FFT rate constants and block-to-hop conversion helpers.

Fixed reference constants: 16 kHz, 160-sample (10 ms) hop, FFT 512, 64-sample (4 ms)
AEC3 block; they are not the live grid. Prefer the helpers in ``aec3_scale.py``.
"""
import math

HOP_SAMPLES = 160
FFT_SIZE = 512
SR_HZ = 16000

HOP_MS = HOP_SAMPLES * 1000 / SR_HZ  # 10.0
AEC3_BLOCK_SAMPLES = 64
AEC3_BLOCK_MS = AEC3_BLOCK_SAMPLES * 1000 / SR_HZ  # 4.0


def aec3_blocks_to_our_hops(n_blocks: int) -> int:
    """Convert an AEC3 per-block count to OUR per-hop count (ceil)."""
    return -(-n_blocks * int(AEC3_BLOCK_MS) // int(HOP_MS))


def ms_to_hops(ms: float) -> int:
    """Convert a duration in milliseconds to OUR hop count (ceil)."""
    return math.ceil(ms / HOP_MS)


def ms_to_aec3_blocks(ms: float) -> int:
    """Convert ms to AEC3 block count (ceil) for cross-checking spec."""
    return -(-int(ms) // int(AEC3_BLOCK_MS))
