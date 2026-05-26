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

# H_error refresh (refined_filter_update_gain.cc:128-138).
# AEC3 H_error is a Kalman-like internal scalar that is dimensionally
# self-consistent with X²/E² in AEC3's own scale. Empirically verified
# (v3.21 2026-05-18 63-case bench): rescaling H by 1/PSD_SCALE collapses
# learning (FS 3.804 → 2.311, ERLE drops from ~10 dB to ~2 dB) because mu
# loses 6 orders of magnitude and K·E per call drops below numerical
# noise. The 0.5·H·X² + n·E² denominator IS dimensionally invariant: in
# int16 0.5·10000·1e6 = 5e9 dominates and mu ≈ 2e-6; in float[-1,1]
# 0.5·10000·1e-4 = 0.5 dominates and mu ≈ 20000 — but K = mu·X scales
# inversely with X, so K·E = (H·X·E)/(0.5·H·X² + n·E²) is identical in
# both representations. Keep AEC3's constants UN-SCALED in float audio.
H_ERROR_INIT_FLOAT = 10000.0
H_ERROR_FLOOR_FLOAT = 1e-3
H_ERROR_CEIL_FLOAT = 1e2
# AEC3 production ceiling (refined_filter_update_gain.cc kHErrorCeiling = 2.0).
# Python default uses 100.0 (H_ERROR_CEIL_FLOAT) from the int16-scaled
# constant derivation; AEC3's float-domain target is 2.0.
# Consumed only when config.use_aec3_h_error_ceil = True (default OFF).
H_ERROR_CEIL_AEC3_FLOAT = 2.0

# Per-block rates (default 10 ms hop equivalent)
LEAKAGE_CONVERGED_PER_HOP_DEFAULT = per_block_rate_to_per_hop(1e-3, 160, 16000)  # 2.5e-3
LEAKAGE_DIVERGED_PER_HOP_DEFAULT = per_block_rate_to_per_hop(1e-1, 160, 16000)   # 2.5e-1

# AEC3 transient-profile (refined_initial config) leakage rates — per 10 ms hop.
# Applied by AEC3 SetConfig(refined_initial) after any EPC trigger for 2.5 s.
# Source: AecState::InitialState (aec_state.cc) + FilterConfig refined_initial defaults.
LEAKAGE_CONVERGED_TRANSIENT_PER_HOP = per_block_rate_to_per_hop(5e-3, 160, 16000)  # 1.25e-2
LEAKAGE_DIVERGED_TRANSIENT_PER_HOP  = per_block_rate_to_per_hop(5e-1, 160, 16000)  # 1.25

# Gate 0 Variant D1_corrected: AEC3 SetConfig(refined_initial) leakage rates (corrected 2026-05-26).
# Source: echo_canceller3_config.h FilterConfig refined_initial (lines 102-107):
#   refined_initial.leakage_converged = 0.005f  (vs 0.00005 normal refined → 100×)
#   refined_initial.leakage_diverged  = 0.5f    (vs 0.05 normal refined → 10×)
#   refined_initial.error_floor       = 0.001f  (SAME as normal refined — no change)
# Per-hop equivalent: identical to TRANSIENT_PER_HOP (both derived from refined_initial).
# Prior values (0.025/0.25) were wrong (sourced from 0.01/0.1 not 0.005/0.5 — WITHDRAWN).
LEAKAGE_CONVERGED_SETCONFIG_INITIAL = LEAKAGE_CONVERGED_TRANSIENT_PER_HOP  # 0.0125
LEAKAGE_DIVERGED_SETCONFIG_INITIAL  = LEAKAGE_DIVERGED_TRANSIENT_PER_HOP   # 1.25

# Filter noise gate — weight-update path only (filter_update gate, NOT suppression)
# AEC3 echo_canceller3_config.cc:97-100:
#   RefinedConfiguration refined  = {..., .noise_gate = 20075344.f};
#   CoarseConfiguration  coarse   = {..., .noise_gate = 20075344.f};
# Both refined AND coarse use 20075344 (int16²). Gate: mu[k]=0 where X²[k] < noise_gate.
# FFT-size note: AEC3 uses FFT=128 (64-bin SpectralSum); Python uses FFT=512 (257 bins).
# The int16²→float conversion is applied as-is; hop/FFT/window differences may affect
# the effective gate sensitivity — verify via trace (A2_zero_frac_LF/MF/HF) rather
# than assuming strict parity. The constant origin (20075344) is authoritative.
FILTER_NOISE_GATE_POWER_FLOAT = psd_int16_to_float(20075344.0)  # 0.01870 — AEC3 filter gate (both refined+coarse)

# Legacy constant: 27509562 was incorrectly ported from AEC3 echo_model (suppression
# path) into the filter weight-update gate. Retained as-is for default-OFF behaviour;
# flag-gated paths use FILTER_NOISE_GATE_POWER_FLOAT above.
# AEC3 echo_model.noise_gate_power = 27509.42f is in int16² scale (same as SpectrumBuffer /
# far_psd which is scaled by _PSD_SCALE=32768² before entering ResidualEchoEstimator).
# Python default 27509562.0 is 1000× too large: fires gate for amplitude < 5245 int16
# (0.16 float), while AEC3 only fires for amplitude < 166 int16 (0.005 float).
NOISE_GATE_POWER_FLOAT = psd_int16_to_float(27509562.0)  # 0.02562 — DO NOT use for filter gate

# Corrected residual echo model noise gate (echo_model.noise_gate_power path).
# Source: echo_canceller3_config.h EchoModel::noise_gate_power = 27509.42f.
# Unit: int16² scale (same as far_psd = float_spec² × 32768²). No psd_int16_to_float.
# Consumed only when config.use_aec3_residual_noise_gate = True (default OFF).
# R0.2 gap: Python default (27509562) is 1000× too large — non-linear path X2 gated
# for nearly all render signals, leaving R2 = reverb-only (direct echo path zeroed).
RESIDUAL_NOISE_GATE_POWER = 27509.42  # int16² scale — AEC3 echo_model.noise_gate_power verbatim
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

# StationarityEstimator (stationarity_estimator.cc).
# kMinNoisePower (int16²) is a numerical floor on the per-bin noise estimate
# so the long-ratio test (acum_power / noise) is well-defined when the actual
# noise spectrum is near zero.
STATIONARITY_MIN_NOISE_POWER_FLOAT = psd_int16_to_float(10.0)  # 9.31e-9
STATIONARITY_THR_RATIO = 10.0           # unitless; acum_power < 10 × noise → stationary
STATIONARITY_BLOCK_FRACTION = 0.75      # unitless; >75% bands stationary → block stationary
STATIONARITY_HANGOVER_HOPS_DEFAULT = blocks_to_hops(12, 160, 16000)        # 5
STATIONARITY_AVG_INIT_HOPS_DEFAULT = blocks_to_hops(20, 160, 16000)        # 8
STATIONARITY_INITIAL_PHASE_HOPS_DEFAULT = blocks_to_hops(500, 160, 16000)  # 200
STATIONARITY_WINDOW_HOPS_DEFAULT = blocks_to_hops(13, 160, 16000)          # 5
STATIONARITY_ALPHA = 0.004              # unitless (long-term smoothing)
STATIONARITY_ALPHA_INIT = 0.04          # unitless (warmup smoothing)
