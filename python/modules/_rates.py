"""Block / FFT rate constants and AEC3 rescale helpers.

LEGACY -- the only importer is ``delay/render_delay_controller.py``, itself
a reference/legacy port with no production caller (see that module's
docstring). Everywhere else in this package that needs a rate/hop
conversion takes ``hop_size``/``sample_rate`` as real parameters and uses
``aec3_scale.py``'s helpers instead (see e.g. ``state/filter_analyzer.py``,
which used to hardcode against this same module before being fixed that
way). Do not add a new import of this module -- use ``aec3_scale.py``.

Project-locked framing for AEC: SR=16 kHz, hop=160 samples (10 ms), FFT=512
(the legacy 16 kHz default before the 2026-08-01 grid change -- NOT any
current default). AEC3 reference uses SR=16 kHz, kBlockSize=64 (4 ms),
kFftLength=128.

Every AEC3 per-block constant must be converted to milliseconds first, then
re-discretised to OUR hops via ``aec3_blocks_to_our_hops`` (or ``ms_to_hops``
when the AEC3 source already documents the constant in time units). Inline
the rescaled value with a comment of the form:

    # AEC3 N blocks (~Xms) -> our M hops

Lag-range / coverage knobs (matched_filter window length, max_delay_ms,
etc.) MUST be specified in ms — not in "N blocks" — so we don't inherit
AEC3's 4-ms-block budgeting at our 10-ms hop and end up with 2.5x the
intended coverage.
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
