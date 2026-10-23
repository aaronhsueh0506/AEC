"""Conversion helpers from AEC3 constants (int16 PSD scale, 4 ms / 64-sample blocks) to this
pipeline (float [-1, 1] PSD scale, config-driven hop, 50% overlap, frame = 2 * hop).

Pass AEC3 PSD thresholds and block counts through these helpers instead of pasting raw values.
"""

PSD_SCALE = 32768.0 ** 2  # int16 max squared
PSD_SCALE_INV = 1.0 / PSD_SCALE

AEC3_BLOCK_SAMPLES_16K = 64
AEC3_BLOCK_MS = 4.0

# AEC3 per-bin PSD floors assume kFftLengthBy2 = 64; floors that scale with FFT length are
# multiplied by (fft_size / 2) / 64.
AEC3_FFT_LENGTH_BY_2 = 64


import numpy as np

def psd_int16_to_float(value: float) -> float:
    """AEC3 PSD-scale constant (int16^2) -> float [-1, 1] PSD scale."""
    return value * PSD_SCALE_INV


def blocks_to_hops(blocks: int, hop_samples: int, sample_rate: int) -> int:
    """AEC3 block count -> integer hop count of equal wall-clock time (nearest; at least 1 if blocks > 0)."""
    if blocks <= 0:
        return 0
    seconds = blocks * AEC3_BLOCK_SAMPLES_16K / 16000.0
    hops = int(round(seconds * sample_rate / hop_samples))
    return max(1, hops)


def ms_to_hops(ms: float, hop_samples: int, sample_rate: int) -> int:
    """Milliseconds -> integer hop count (nearest; at least 1 if ms > 0)."""
    if ms <= 0:
        return 0
    hops = int(round(ms * 1e-3 * sample_rate / hop_samples))
    return max(1, hops)


def ms_to_hops_f32(ms: float, hop_samples: int, sample_rate: int) -> int:
    """``ms_to_hops`` evaluated in float32 in the same operand order as the C helper
    ``aec3_ms_to_hops``, so the hop count agrees with C exactly (e.g. 20 ms at 16 kHz / hop
    128 is 2.5 in double but 2.5000002 in float32)."""
    if ms <= 0:
        return 0
    f32 = np.float32
    hops = f32(f32(ms) * f32(1e-3)) * f32(sample_rate) / f32(hop_samples)
    return max(1, int(np.rint(hops)))


def per_block_rate_to_per_hop(per_block_rate: float,
                              hop_samples: int,
                              sample_rate: int) -> float:
    """AEC3 per-block injection rate -> per-hop rate with the same per-second injection:
    ``per_block * hop_seconds / block_seconds``. Linear; valid for small rates only (use
    ``per_block_ema_alpha_to_per_hop`` for EMA alphas)."""
    block_seconds = AEC3_BLOCK_SAMPLES_16K / 16000.0
    hop_seconds = hop_samples / float(sample_rate)
    return per_block_rate * (hop_seconds / block_seconds)


def per_block_ema_alpha_to_per_hop(per_block_alpha: float,
                                   hop_samples: int,
                                   sample_rate: int) -> float:
    """AEC3 per-block EMA alpha (new-sample weight) -> per-hop alpha with the same
    wall-clock decay: ``1 - (1 - alpha) ** (hop_seconds / block_seconds)``."""
    if per_block_alpha <= 0.0:
        return 0.0
    if per_block_alpha >= 1.0:
        return 1.0
    block_seconds = AEC3_BLOCK_SAMPLES_16K / 16000.0
    hop_seconds = hop_samples / float(sample_rate)
    return 1.0 - (1.0 - per_block_alpha) ** (hop_seconds / block_seconds)


def fft_density_scale(value_int16sq: float, fft_size: int) -> float:
    """Scale an AEC3 per-bin PSD constant from kFftLengthBy2 = 64 to ``fft_size / 2``
    (e.g. 64 -> 256 at fft 512)."""
    if fft_size <= 0:
        raise ValueError(f"fft_size must be positive, got {fft_size}")
    return value_int16sq * ((fft_size // 2) / AEC3_FFT_LENGTH_BY_2)


def per_bin_psd_threshold(calibrated_value: float, hop_size: int,
                          ref_hop: int = 160) -> float:
    """Scale a per-bin X^2 threshold calibrated at ``ref_hop`` linearly with hop_size
    (per-bin power is proportional to the frame size)."""
    if hop_size <= 0:
        raise ValueError(f"hop_size must be positive, got {hop_size}")
    return calibrated_value * hop_size / ref_hop


def nl_r2_norm_power(hop_size: int, _ref_hop: int = 160) -> float:
    """Normalisation power of the nonlinear R^2 term: 1.07e7 at hop 160, linear in hop_size."""
    _calibrated_at_ref = 1.07e7  # calibrated at hop 160
    return _calibrated_at_ref * hop_size / _ref_hop


def block_energy_scale(value_int16sq: float, hop_samples: int) -> float:
    """Scale an AEC3 per-block time-domain energy threshold from 64 samples to ``hop_samples``."""
    if hop_samples <= 0:
        raise ValueError(f"hop_samples must be positive, got {hop_samples}")
    return value_int16sq * (hop_samples / AEC3_BLOCK_SAMPLES_16K)


def per_block_growth_to_per_hop(per_block_multiplier: float,
                                hop_samples: int,
                                sample_rate: int) -> float:
    """AEC3 per-block multiplicative growth factor -> per-hop: ``g ** (hop_seconds / block_seconds)``."""
    block_seconds = AEC3_BLOCK_SAMPLES_16K / 16000.0
    hop_seconds = hop_samples / float(sample_rate)
    return per_block_multiplier ** (hop_seconds / block_seconds)


def ema_alpha_from_tau_ms(tau_ms: float, hop_ms: float) -> float:
    """Per-hop EMA retention for a time constant in ms: ``exp(-hop_ms / tau_ms)``."""
    return float(np.exp(-hop_ms / tau_ms))


def db_to_linear(db: float) -> float:
    """dB -> linear power ratio."""
    return float(10.0 ** (db / 10.0))


def growth_rehop(retention_at_ref: float, ref_hop_samples: int, ref_sample_rate: int,
                 hop_samples: int, sample_rate: int) -> float:
    """Retime a retention-convention EMA constant (``x <- r * x + (1 - r) * new``) from a
    reference grid to the live grid: ``r ** (hop_seconds / ref_seconds)``."""
    ref_seconds = ref_hop_samples / float(ref_sample_rate)
    hop_seconds = hop_samples / float(sample_rate)
    return retention_at_ref ** (hop_seconds / ref_seconds)


# Pre-converted AEC3 constants (16 kHz, hop 160 reference).

# H_error is kept in AEC3's scale: the update K * E is invariant to the PSD representation
# and rescaling collapses learning.
H_ERROR_INIT_FLOAT = 10000.0
H_ERROR_FLOOR_FLOAT = 1e-3
# Error ceiling; a larger value lets H_error dominate the mu denominator.
H_ERROR_CEIL_FLOAT = 2.0
# Alias of H_ERROR_CEIL_FLOAT.
H_ERROR_CEIL_AEC3_FLOAT = 2.0

# Steady-state leakage per AEC3 block (converged 5e-5, diverged 5e-2), converted to per hop.
LEAKAGE_CONVERGED_PER_HOP_DEFAULT = per_block_rate_to_per_hop(5e-5, 160, 16000)  # 1.25e-4
LEAKAGE_DIVERGED_PER_HOP_DEFAULT = per_block_rate_to_per_hop(5e-2, 160, 16000)   # 0.125

# Transient-profile leakage rates per hop (applied after a path-change trigger).
LEAKAGE_CONVERGED_TRANSIENT_PER_HOP = per_block_rate_to_per_hop(5e-3, 160, 16000)  # 1.25e-2
LEAKAGE_DIVERGED_TRANSIENT_PER_HOP  = per_block_rate_to_per_hop(5e-1, 160, 16000)  # 1.25

# Initial-profile leakage per hop (converged 5e-3, diverged 5e-1 per block).
LEAKAGE_CONVERGED_SETCONFIG_INITIAL = LEAKAGE_CONVERGED_TRANSIENT_PER_HOP  # 0.0125
LEAKAGE_DIVERGED_SETCONFIG_INITIAL  = LEAKAGE_DIVERGED_TRANSIENT_PER_HOP   # 1.25

# Filter weight-update noise gate (refined and coarse): mu[k] = 0 where X^2[k] < gate.
FILTER_NOISE_GATE_POWER_FLOAT = psd_int16_to_float(20075344.0)  # 0.01870 — AEC3 filter gate (both refined+coarse)

# Residual-path noise gate in float scale; not for the filter gate.
NOISE_GATE_POWER_FLOAT = psd_int16_to_float(27509562.0)  # 0.02562 — DO NOT use for filter gate

# Residual echo model noise gate in int16^2 scale (applied without conversion).
RESIDUAL_NOISE_GATE_POWER = 27509.42  # int16² scale — AEC3 echo_model.noise_gate_power verbatim
NORMAL_RENDER_LIMIT_FLOAT = psd_int16_to_float(64.0)     # 5.96e-8
LOW_RENDER_LIMIT_FLOAT = psd_int16_to_float(256.0)       # 2.38e-7
FLOOR_POWER_FLOAT = psd_int16_to_float(128.0)            # 1.19e-7

# Matched-filter limits are amplitude thresholds (int16 / 32768), not PSD values.
MATCHED_FILTER_EXCITATION_LIMIT_FLOAT = 150.0 / 32768.0            # 4.578e-3
MATCHED_FILTER_SATURATION_LIMIT_FLOAT = 32000.0 / 32768.0          # 0.9766

# default hop-count counters at the 10 ms hop
COARSE_RESET_HANGOVER_HOPS_DEFAULT = blocks_to_hops(40, 160, 16000)    # 16
POOR_EXCITATION_COUNTER_INITIAL_HOPS_DEFAULT = blocks_to_hops(1000, 160, 16000)  # 400
CLOCKDRIFT_STABILITY_HOPS_DEFAULT = blocks_to_hops(7500, 160, 16000)   # 3000
NEAREND_SMOOTHER_HOPS_DEFAULT = blocks_to_hops(4, 160, 16000)          # 2
NOISE_ESTIMATOR_HOPS_DEFAULT = blocks_to_hops(50, 160, 16000)          # 20

MAX_ERLE_LF = 4.0
MAX_ERLE_HF = 1.5
MIN_ERLE = 1.0
LAG_AGGREGATOR_INITIAL_THRESHOLD = 5
LAG_AGGREGATOR_CONVERGED_THRESHOLD = 20
LAG_HISTOGRAM_WINDOW = 250
MATCHED_FILTER_NUM_FILTERS = 5
MATCHED_FILTER_ALIGNMENT_SHIFT_SUB_BLOCKS = 24
MATCHED_FILTER_WINDOW_SUB_BLOCKS = 32

# Stationarity estimator: numerical floor on the per-bin noise estimate.
STATIONARITY_MIN_NOISE_POWER_FLOAT = psd_int16_to_float(10.0)  # 9.31e-9
STATIONARITY_THR_RATIO = 10.0           # unitless; acum_power < 10 × noise → stationary
STATIONARITY_BLOCK_FRACTION = 0.75      # unitless; >75% bands stationary → block stationary
STATIONARITY_HANGOVER_HOPS_DEFAULT = blocks_to_hops(12, 160, 16000)        # 5
STATIONARITY_AVG_INIT_HOPS_DEFAULT = blocks_to_hops(20, 160, 16000)        # 8
STATIONARITY_INITIAL_PHASE_HOPS_DEFAULT = blocks_to_hops(500, 160, 16000)  # 200
STATIONARITY_WINDOW_HOPS_DEFAULT = blocks_to_hops(13, 160, 16000)          # 5
STATIONARITY_ALPHA = 0.004              # unitless (long-term smoothing)
STATIONARITY_ALPHA_INIT = 0.04          # unitless (warmup smoothing)
