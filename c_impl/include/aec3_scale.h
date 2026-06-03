/* aec3_scale.h — AEC3 constant -> our-scale conversion helpers (C port of
 * python/modules/aec3_scale.py).
 *
 * AEC3 source operates on int16 PSD scale (PSD ~ int16^2 up to 32768^2) and
 * 4 ms blocks (kBlockSize = 64 @ 16 kHz). Our pipeline runs on float[-1,1] PSD
 * scale and a config hop_size (default 160 @ 16 kHz, 50% overlap frame=2*hop).
 * NEVER paste a raw int16 PSD threshold or 4 ms-block count into ported code —
 * route it through here so the conversion is explicit and uniform.
 *
 * Parity note: Python's int(round(x)) is round-half-to-EVEN; the *_to_hops
 * helpers below use lrint() (default FE_TONEAREST = round-half-to-even), NOT
 * round() (half-away-from-zero), so bin/hop counts match the reference exactly.
 */
#ifndef AEC3_SCALE_H
#define AEC3_SCALE_H

/* ── Core scale constants ──────────────────────────────────────────────── */
#define AEC3_PSD_SCALE        (32768.0 * 32768.0)   /* 1.0737e9  */
#define AEC3_PSD_SCALE_INV    (1.0 / AEC3_PSD_SCALE) /* 9.313e-10 */
#define AEC3_BLOCK_SAMPLES_16K 64                    /* kBlockSize @ 16 kHz */
#define AEC3_BLOCK_MS          4.0
#define AEC3_FFT_LENGTH_BY_2   64                    /* AEC3 kFftLengthBy2 */

/* ── Conversion functions (pure; match aec3_scale.py semantics) ─────────── */
double aec3_psd_int16_to_float(double value);
int    aec3_blocks_to_hops(int blocks, int hop_samples, int sample_rate);
int    aec3_ms_to_hops(double ms, int hop_samples, int sample_rate);
double aec3_per_block_rate_to_per_hop(double per_block_rate, int hop_samples, int sample_rate);
double aec3_per_block_ema_alpha_to_per_hop(double per_block_alpha, int hop_samples, int sample_rate);
double aec3_fft_density_scale(double value_int16sq, int fft_size);
double aec3_per_bin_psd_threshold(double calibrated_value, int hop_size, int ref_hop);
double aec3_nl_r2_norm_power(int hop_size, int ref_hop);
double aec3_block_energy_scale(double value_int16sq, int hop_samples);
double aec3_per_block_growth_to_per_hop(double per_block_multiplier, int hop_samples, int sample_rate);

/* ── Pre-converted constants (16 kHz, hop=160 reference) ───────────────────
 * Defined by their computing expression so they fold bit-identically to the
 * Python module-level constants. */
#define AEC3_H_ERROR_INIT_FLOAT   10000.0
#define AEC3_H_ERROR_FLOOR_FLOAT  1.0e-3
#define AEC3_H_ERROR_CEIL_FLOAT   2.0

/* AEC3 refined leakage rates, per-hop (per_block_rate_to_per_hop(rate,160,16k)
 * = rate × (160/16000)/(64/16000) = rate × 2.5). Steady + transient (initial-
 * state) profiles. Defined by their computing expression so they fold bit-
 * identically to the python aec3_scale.py module-level constants. */
#define AEC3_LEAKAGE_CONVERGED_PER_HOP            (5.0e-5 * 2.5)   /* 1.25e-4 */
#define AEC3_LEAKAGE_DIVERGED_PER_HOP             (5.0e-2 * 2.5)   /* 0.125   */
#define AEC3_LEAKAGE_CONVERGED_TRANSIENT_PER_HOP  (5.0e-3 * 2.5)   /* 0.0125  */
#define AEC3_LEAKAGE_DIVERGED_TRANSIENT_PER_HOP   (5.0e-1 * 2.5)   /* 1.25    */

/* POOR_EXCITATION_COUNTER_INITIAL_HOPS_DEFAULT = blocks_to_hops(1000,160,16k) = 400 */
#define AEC3_POOR_EXC_COUNTER_INITIAL_HOPS  400

/* Unitless (same on both sides) */
#define AEC3_MAX_ERLE_LF  4.0
#define AEC3_MAX_ERLE_HF  1.5
#define AEC3_MIN_ERLE     1.0

#define AEC3_STATIONARITY_THR_RATIO       10.0
#define AEC3_STATIONARITY_BLOCK_FRACTION  0.75
#define AEC3_STATIONARITY_ALPHA           0.004
#define AEC3_STATIONARITY_ALPHA_INIT      0.04

/* Matched-filter amplitude limits (/32768, NOT /32768^2 — amplitude not PSD) */
#define AEC3_MATCHED_FILTER_EXCITATION_LIMIT_FLOAT (150.0 / 32768.0)
#define AEC3_MATCHED_FILTER_SATURATION_LIMIT_FLOAT (32000.0 / 32768.0)

/* PSD-scale render/floor constants (echo_canceller3_config.h) */
#define AEC3_NORMAL_RENDER_LIMIT_FLOAT (64.0  * AEC3_PSD_SCALE_INV)  /* 5.96e-8 */
#define AEC3_LOW_RENDER_LIMIT_FLOAT    (256.0 * AEC3_PSD_SCALE_INV)  /* 2.38e-7 */
#define AEC3_FLOOR_POWER_FLOAT         (128.0 * AEC3_PSD_SCALE_INV)  /* 1.19e-7 */
#define AEC3_FILTER_NOISE_GATE_POWER_FLOAT (20075344.0 * AEC3_PSD_SCALE_INV) /* 0.018697 */
#define AEC3_STATIONARITY_MIN_NOISE_POWER_FLOAT (10.0 * AEC3_PSD_SCALE_INV)  /* 9.31e-9 */
/* echo_model.noise_gate_power — int16^2 scale, used verbatim (no conversion) */
#define AEC3_RESIDUAL_NOISE_GATE_POWER 27509.42

#endif /* AEC3_SCALE_H */
