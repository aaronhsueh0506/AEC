"""Central conversion helpers for AEC3 constants → our scale.

AEC3 source operates on:
  * int16 PSD scale (samples in [-32768, 32767], PSD ~ int16² up to 32768² ≈ 1.07e9)
  * 4 ms blocks (kBlockSize = 64 samples @ 16 kHz)

Our pipeline operates on:
  * float[-1, 1] PSD scale (PSD ~ float² up to 1.0)
  * Config-driven hop_size (default 160 samples @ 16 kHz = 10 ms)
  * 50% overlap invariant: frame_size = 2 * hop_size

Use these helpers when porting AEC3 constants. NEVER paste a raw int16 PSD
threshold or 4 ms-block count into our code — always pass it through here so
the conversion is explicit, reviewable, and uniform across modules.
"""

PSD_SCALE = 32768.0 ** 2          # int16 max squared (1.0737e9)
PSD_SCALE_INV = 1.0 / PSD_SCALE   # 9.31e-10

AEC3_BLOCK_SAMPLES_16K = 64       # AEC3 kBlockSize at 16 kHz
AEC3_BLOCK_MS = 4.0               # 64 / 16000 * 1000


def psd_int16_to_float(value: float) -> float:
    """Convert an AEC3 PSD-scale constant (int16²) to our float[-1,1] PSD scale."""
    return value * PSD_SCALE_INV


def blocks_to_hops(blocks: int, hop_samples: int, sample_rate: int) -> int:
    """Convert an AEC3 block count (per AEC3_BLOCK_SAMPLES @ sample_rate)
    to an integer hop count that covers the equivalent wall-clock time.

    Rounds to nearest; clamps minimum to 1 hop if blocks > 0.
    """
    if blocks <= 0:
        return 0
    seconds = blocks * AEC3_BLOCK_SAMPLES_16K / 16000.0
    hops = int(round(seconds * sample_rate / hop_samples))
    return max(1, hops)


def ms_to_hops(ms: float, hop_samples: int, sample_rate: int) -> int:
    """Convert milliseconds to integer hop count.

    Rounds to nearest; clamps minimum to 1 if ms > 0.
    """
    if ms <= 0:
        return 0
    hops = int(round(ms * 1e-3 * sample_rate / hop_samples))
    return max(1, hops)


def per_block_rate_to_per_hop(per_block_rate: float,
                              hop_samples: int,
                              sample_rate: int) -> float:
    """Convert an AEC3 per-block injection rate (e.g. leakage_converged) to a
    per-hop rate that injects the same amount per second.

    AEC3 fires per 4 ms; we fire per hop. To match per-second injection:
        per_hop = per_block * (hop_seconds / block_seconds)
    """
    block_seconds = AEC3_BLOCK_SAMPLES_16K / 16000.0
    hop_seconds = hop_samples / float(sample_rate)
    return per_block_rate * (hop_seconds / block_seconds)


# ── Pre-converted AEC3 constants (16 kHz, hop_size 160 reference) ──
# These are the canonical float-scale values used across Phase B/C ports.
# At non-default hop_size, derive at runtime via blocks_to_hops() etc.

# H_error refresh (refined_filter_update_gain.cc:128-138)
H_ERROR_INIT_FLOAT = psd_int16_to_float(10000.0)       # 9.31e-6
H_ERROR_FLOOR_FLOAT = psd_int16_to_float(1e-3)         # 9.31e-13
H_ERROR_CEIL_FLOAT = psd_int16_to_float(1e2)           # 9.31e-8

# Per-block rates (default 10 ms hop equivalent)
LEAKAGE_CONVERGED_PER_HOP_DEFAULT = per_block_rate_to_per_hop(1e-3, 160, 16000)  # 2.5e-3
LEAKAGE_DIVERGED_PER_HOP_DEFAULT = per_block_rate_to_per_hop(1e-1, 160, 16000)   # 2.5e-1

# Suppression / residual thresholds
NOISE_GATE_POWER_FLOAT = psd_int16_to_float(27509562.0)  # 0.0256
NORMAL_RENDER_LIMIT_FLOAT = psd_int16_to_float(64.0)     # 5.96e-8
LOW_RENDER_LIMIT_FLOAT = psd_int16_to_float(256.0)       # 2.38e-7
FLOOR_POWER_FLOAT = psd_int16_to_float(128.0)            # 1.19e-7

# Matched filter operates directly on time-domain samples. AEC3's
# excitation_limit and saturation_limit are AMPLITUDE thresholds in int16
# amplitude units (NOT PSD values), so the conversion is /32768, not
# /32768². excitation_limit is squared inside MatchedFilterCore() to
# produce the per-sample power threshold: x2_sum > filter_size *
# excitation_limit². If we accidentally used the PSD conversion here, the
# threshold becomes ~10^14 too high and the filter never adapts.
MATCHED_FILTER_EXCITATION_LIMIT_FLOAT = 150.0 / 32768.0            # 4.578e-3
MATCHED_FILTER_SATURATION_LIMIT_FLOAT = 32000.0 / 32768.0          # 0.9766

# Default hop-count counters (10 ms hop reference)
COARSE_RESET_HANGOVER_HOPS_DEFAULT = blocks_to_hops(40, 160, 16000)    # 16
POOR_EXCITATION_COUNTER_INITIAL_HOPS_DEFAULT = blocks_to_hops(1000, 160, 16000)  # 400
CLOCKDRIFT_STABILITY_HOPS_DEFAULT = blocks_to_hops(7500, 160, 16000)   # 3000
NEAREND_SMOOTHER_HOPS_DEFAULT = blocks_to_hops(4, 160, 16000)          # 2
NOISE_ESTIMATOR_HOPS_DEFAULT = blocks_to_hops(50, 160, 16000)          # 20

# Unitless (no conversion needed — same constants on both sides)
MAX_ERLE_LF = 4.0
MAX_ERLE_HF = 1.5
MIN_ERLE = 1.0
LAG_AGGREGATOR_INITIAL_THRESHOLD = 5
LAG_AGGREGATOR_CONVERGED_THRESHOLD = 20
LAG_HISTOGRAM_WINDOW = 250
MATCHED_FILTER_NUM_FILTERS = 5
MATCHED_FILTER_ALIGNMENT_SHIFT_SUB_BLOCKS = 24
MATCHED_FILTER_WINDOW_SUB_BLOCKS = 32
