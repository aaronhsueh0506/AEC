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

# AEC3 single-side FFT bin count for per-bin PSD constants. AEC3's
# aec3_common.h fixes ``kFftLengthBy2 = 64`` (and ``kFftLength = 128``).
# Many of AEC3's per-bin PSD floor constants encode this 64 implicitly
# (e.g. comfort_noise_generator.cc:46 ``return 64.f * powf(...)`` for the
# WGN-matching CN floor). At our hop=160 / fft=512 we run at
# ``kFftLengthBy2_ours = 256`` → 4× more bins per Hz → per-bin |X[k]|² for
# the same physical signal is intrinsically 4× larger. Floor constants
# that scale with FFT length must be scaled by ``fft/2 / 64 = 4`` to
# preserve per-bin physical meaning.
AEC3_FFT_LENGTH_BY_2 = 64


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

    LINEAR approximation — valid for *small* rates (≪ 1) where the per-second
    sum is the relevant invariant (leakage decay, noise injection). For EMA
    alphas (esp. ≥ 0.05) use ``per_block_ema_alpha_to_per_hop`` instead so the
    lag-decay envelope matches in wall-clock time.
    """
    block_seconds = AEC3_BLOCK_SAMPLES_16K / 16000.0
    hop_seconds = hop_samples / float(sample_rate)
    return per_block_rate * (hop_seconds / block_seconds)


def per_block_ema_alpha_to_per_hop(per_block_alpha: float,
                                   hop_samples: int,
                                   sample_rate: int) -> float:
    """Convert an AEC3 per-block EMA alpha (new-sample weight) to a per-hop
    alpha that gives the *same wall-clock lag-decay envelope*.

    AEC3 EMA: ``x ← (1-α)·x + α·new`` so the lag retention per AEC3 block is
    ``(1-α)``. Wall-clock equivalence over T seconds requires::

        (1 - α_per_block)^(T·n_blocks_per_sec) = (1 - α_per_hop)^(T·n_hops_per_sec)

    Solving for our per-hop alpha::

        α_per_hop = 1 − (1 − α_per_block)^(hop_seconds / block_seconds)

    Use for any AEC3 constant whose semantic is "fraction of old state forgotten
    per frame". The exact (exponential) form is required when α is non-small —
    the linear ``per_block_rate_to_per_hop`` approximation breaks down because
    α ≥ 1 has no EMA meaning.
    """
    if per_block_alpha <= 0.0:
        return 0.0
    if per_block_alpha >= 1.0:
        return 1.0
    block_seconds = AEC3_BLOCK_SAMPLES_16K / 16000.0
    hop_seconds = hop_samples / float(sample_rate)
    return 1.0 - (1.0 - per_block_alpha) ** (hop_seconds / block_seconds)


def fft_density_scale(value_int16sq: float, fft_size: int) -> float:
    """Scale an AEC3 per-bin PSD constant from AEC3's kFftLengthBy2=64 to
    our fft_size/2.

    AEC3's per-bin |X[k]|² PSD floor constants (CN noise floor,
    EchoAudibilityConfig floor_power / *_render_limit, EchoModel
    min_noise_floor_power, ...) are sized for AEC3's FFT length 128 →
    65 bins. At our fft=512 → 257 bins, per-bin PSD for the same
    physical signal is intrinsically (fft_ours / 2) / 64 = 4× larger.
    Floors that were calibrated against AEC3's per-bin scale must be
    scaled by the same ratio to keep equivalent per-bin protection.

    Example:
        fft_density_scale(64.0, fft_size=512) → 256.0     (CN floor base)
        fft_density_scale(128.0, fft_size=512) → 512.0    (floor_power)
        fft_density_scale(1638400.0, fft_size=512) → 6553600.0
    """
    if fft_size <= 0:
        raise ValueError(f"fft_size must be positive, got {fft_size}")
    return value_int16sq * ((fft_size // 2) / AEC3_FFT_LENGTH_BY_2)


def per_bin_psd_threshold(calibrated_value: float, hop_size: int,
                          ref_hop: int = 160) -> float:
    """Scale a per-bin X² threshold proportionally with frame size.

    Our frame = 2 × hop_size actual samples (50% OLA). Per-bin FFT power
    ∝ frame_size for broadband signals, so thresholds should scale linearly
    with hop_size to preserve the same signal-level gate semantics.

    ``calibrated_value`` is the desired threshold at ``ref_hop`` (default 160).
    Callers pass the current empirically-calibrated constant; the function
    returns the equivalent value for any other hop_size.

    Use for per-bin X² energy floors: kX2BandEnergyThreshold (44015068 at hop=160),
    ErlEstimator _X2_MIN (same), and residual noise_gate_power (27509.42 at hop=160).

    NOTE: AEC3 uses 64-sample blocks (FFT=128). The calibrated values at hop=160
    are 5× smaller than AEC3's raw constants due to our larger frame (320 vs 64).
    This is a known discrepancy, documented, and preserved here to maintain
    existing bench parity. A future correction pass can pass the AEC3 raw value
    with ref_hop=32 to get AEC3-aligned behaviour.

    Example:
        per_bin_psd_threshold(44015068.0, hop_size=160)  → 44015068.0  (no-op)
        per_bin_psd_threshold(44015068.0, hop_size=320)  → 88030136.0
        per_bin_psd_threshold(44015068.0, hop_size=80)   → 22007534.0
    """
    if hop_size <= 0:
        raise ValueError(f"hop_size must be positive, got {hop_size}")
    return calibrated_value * hop_size / ref_hop


def nl_r2_norm_power(hop_size: int, _ref_hop: int = 160) -> float:
    """Normalisation power for the L1 Kuech-Kellermann nonlinear R² term.

    The formula ``r2_nl = alpha × x2² / norm`` is calibrated at hop=160
    where norm=1.07e7 gives +0.079/+0.106 FS echo improvement (empirical).
    Scales linearly with hop_size to preserve the r2_nl/r2_linear ratio
    when hop changes (per-bin x2 ∝ frame_size = 2×hop for same signal).

    Example:
        nl_r2_norm_power(160) → 1.07e7   (exact — existing calibration preserved)
        nl_r2_norm_power(320) → 2.14e7   (halves L1 aggressiveness at hop=320)
        nl_r2_norm_power(80)  → 5.35e6   (doubles L1 aggressiveness at hop=80)
    """
    _calibrated_at_ref = 1.07e7  # empirically tuned at hop=160; do not change
    return _calibrated_at_ref * hop_size / _ref_hop


def block_energy_scale(value_int16sq: float, hop_samples: int) -> float:
    """Scale an AEC3 per-block time-domain energy threshold from
    kBlockSize=64 samples to our hop_samples.

    Used for ``sum(x[n]² for n in 0..N-1)`` style thresholds such as
    ``_LowNoiseRenderDetector``'s `50² × kBlockSize = 160000`. At our
    hop=160 the same physical RMS produces 2.5× more total energy per
    frame; the threshold must scale by the same wall-clock ratio.

    Example:
        block_energy_scale(160000.0, hop_samples=160) → 400000.0
    """
    if hop_samples <= 0:
        raise ValueError(f"hop_samples must be positive, got {hop_samples}")
    return value_int16sq * (hop_samples / AEC3_BLOCK_SAMPLES_16K)


def per_block_growth_to_per_hop(per_block_multiplier: float,
                                hop_samples: int,
                                sample_rate: int) -> float:
    """Convert an AEC3 per-block *multiplicative* growth factor to per-hop.

    AEC3 update: ``x ← x · g`` per block, so after T seconds the total growth
    is ``g^(T·n_blocks_per_sec)``. Wall-clock equivalence::

        g_per_hop = g_per_block ^ (hop_seconds / block_seconds)

    Use for per-frame multiplicative growth such as the CNG N2 slow-up
    factor 1.0002 (cc:172-174). For small δ where g = 1+δ, this is numerically
    equivalent to ``1 + per_block_rate_to_per_hop(δ, ...)``.
    """
    block_seconds = AEC3_BLOCK_SAMPLES_16K / 16000.0
    hop_seconds = hop_samples / float(sample_rate)
    return per_block_multiplier ** (hop_seconds / block_seconds)


def growth_rehop(retention_at_ref: float, ref_hop_samples: int, ref_sample_rate: int,
                 hop_samples: int, sample_rate: int) -> float:
    """Generalizes ``per_block_growth_to_per_hop``'s fixed AEC3-4ms-block
    reference to an ARBITRARY reference grid (``ref_hop_samples`` @
    ``ref_sample_rate``).

    For PROJECT-NATIVE (non-AEC3) per-hop RETENTION-convention EMA constants
    -- i.e. ones whose update is literally ``x <- retention * x + (1 -
    retention) * new`` (the constant multiplies the OLD state directly; this
    is the OPPOSITE convention from ``per_block_ema_alpha_to_per_hop``'s AEC3
    ``x <- (1-a)*x + a*new``, where ``a`` multiplies the NEW sample) --
    authored against this repo's own legacy hop=160/sample_rate=16000
    (10 ms) default, the implicit single-grid assumption that predates
    per-rate multi-grid support, with zero rate conversion (e.g.
    ``AecConfig.shadow_err_alpha``, ``EchoPathChangeDetector.EPV_FAST_TC``/
    ``EPV_SLOW_TC``).

    Same power-law derivation as ``per_block_growth_to_per_hop``::

        retention_new = retention_at_ref ^ (hop_seconds / ref_seconds)
    """
    ref_seconds = ref_hop_samples / float(ref_sample_rate)
    hop_seconds = hop_samples / float(sample_rate)
    return retention_at_ref ** (hop_seconds / ref_seconds)


# ── Pre-converted AEC3 constants (16 kHz, hop_size 160 reference) ──
# These are the canonical float-scale values used across Phase B/C ports.
# At non-default hop_size, derive at runtime via blocks_to_hops() etc.

# H_error refresh (refined_filter_update_gain.cc:128-138).
# AEC3 H_error is a Kalman-like internal scalar that is dimensionally
# self-consistent with X²/E² in AEC3's own scale. Empirically verified
# on 63-case bench: rescaling H by 1/PSD_SCALE collapses
# learning (FS 3.804 → 2.311, ERLE drops from ~10 dB to ~2 dB) because mu
# loses 6 orders of magnitude and K·E per call drops below numerical
# noise. The 0.5·H·X² + n·E² denominator IS dimensionally invariant: in
# int16 0.5·10000·1e6 = 5e9 dominates and mu ≈ 2e-6; in float[-1,1]
# 0.5·10000·1e-4 = 0.5 dominates and mu ≈ 20000 — but K = mu·X scales
# inversely with X, so K·E = (H·X·E)/(0.5·H·X² + n·E²) is identical in
# both representations. Keep AEC3's constants UN-SCALED in float audio.
H_ERROR_INIT_FLOAT = 10000.0
H_ERROR_FLOOR_FLOAT = 1e-3
# AEC3 ceiling (refined_filter_update_gain.cc kHErrorCeiling = 2.0 in
# echo_canceller3_config.h:96 refined.error_ceil). Aligned to AEC3 2026-05-27;
# prior Python 100.0 was inflated and let H_error dominate mu denom, blocking
# recovery from over-estimation states.
H_ERROR_CEIL_FLOAT = 2.0
# AEC3 production ceiling (refined_filter_update_gain.cc kHErrorCeiling = 2.0).
# Retained for back-compat with any consumer referencing this name; now
# numerically identical to H_ERROR_CEIL_FLOAT after the 2026-05-27 alignment.
H_ERROR_CEIL_AEC3_FLOAT = 2.0

# AEC3 steady refined config (echo_canceller3_config.h:92-97):
#   refined.leakage_converged = 0.00005 per AEC3 block
#   refined.leakage_diverged  = 0.05    per AEC3 block
# Per-hop at hop=160 / sr=16000: multiply by 160/64 = 2.5
#   converged = 0.00005 × 2.5 = 1.25e-4
#   diverged  = 0.05    × 2.5 = 0.125
# Earlier defaults (1e-3 / 1e-1 base) produced 2.5e-3 / 2.5e-1 — 20× / 2× too soft
# (closer to AEC3 INITIAL profile 0.005 / 0.5). Soft leakage keeps H_error
# inflated → mu stays large → filter over-adapts → linear-filter over-estimation
# artifact in weak-echo cases (FS quiet-mic). Aligned to AEC3 steady 2026-05-27.
LEAKAGE_CONVERGED_PER_HOP_DEFAULT = per_block_rate_to_per_hop(5e-5, 160, 16000)  # 1.25e-4
LEAKAGE_DIVERGED_PER_HOP_DEFAULT = per_block_rate_to_per_hop(5e-2, 160, 16000)   # 0.125

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
# Always active: hardcoded use_aec3_residual_noise_gate=True at all call sites.
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
