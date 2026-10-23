/* aec_scale.h -- conversion helpers from reference constants to this library's scale.
 * Reference constants are authored on an int16^2 PSD scale with 4 ms blocks (64 samples @ 16 kHz);
 * the pipeline runs on a float [-1, 1] PSD scale with the live hop_size (50% overlap).
 * Route every raw PSD threshold or block count through these helpers.
 * The *_to_hops helpers round half to even (lrintf). Functions are init-only and float32 unless
 * noted. */
#ifndef AEC_SCALE_H
#define AEC_SCALE_H

/* Core scale constants */
#define AEC_PSD_SCALE        (32768.0 * 32768.0)
#define AEC_PSD_SCALE_INV    (1.0 / AEC_PSD_SCALE)
#define AEC_BLOCK_SAMPLES_16K 64                    /* samples per block @ 16 kHz */
#define AEC_BLOCK_MS          4.0

/* Conversion functions */
int    aec_blocks_to_hops(int blocks, int hop_samples, int sample_rate);
int    aec_ms_to_hops(float ms, int hop_samples, int sample_rate);
float  aec_per_block_rate_to_per_hop(float per_block_rate, int hop_samples, int sample_rate);
float  aec_per_block_ema_alpha_to_per_hop(float per_block_alpha, int hop_samples, int sample_rate);
float  aec_per_bin_psd_threshold(float calibrated_value, int hop_size, int ref_hop);
float  aec_block_energy_scale(float value_int16sq, int hop_samples);
/* per_block_multiplier is double: the power is taken in double and narrowed once, so the
 * result keeps full precision (pass the full-precision config constant, not a float32 value). */
float  aec_per_block_growth_to_per_hop(double per_block_multiplier, int hop_samples, int sample_rate);
/* Retention-form EMA constants (x <- r * x + (1 - r) * new) authored on an arbitrary reference
 * grid (ref_hop_samples, ref_sample_rate): r_per_hop = r_ref ^ (hop_seconds / ref_seconds).
 * Returns retention_at_ref unchanged when any rate or hop is non-positive. */
float  aec_growth_rehop(float retention_at_ref, int ref_hop_samples, int ref_sample_rate,
                        int hop_samples, int sample_rate);

/* Double-precision exceptions: they derive the observer corroboration gate's EMA alpha and margin,
 * which feed a discrete per-hop decision and must stay double. */
double aec_ema_alpha_from_tau_ms(double tau_ms, double hop_ms);
double aec_db_to_linear(double db);

/* Pre-converted constants (16 kHz, hop=160 reference), defined by their computing expression. */
#define AEC_H_ERROR_INIT_FLOAT   10000.0
#define AEC_H_ERROR_FLOOR_FLOAT  1.0e-3
#define AEC_H_ERROR_CEIL_FLOAT   2.0

/* Refined-filter leakage rates per hop (per-block rate x 2.5), steady and transient profiles. */
#define AEC_LEAKAGE_CONVERGED_PER_HOP            (5.0e-5 * 2.5)
#define AEC_LEAKAGE_DIVERGED_PER_HOP             (5.0e-2 * 2.5)
#define AEC_LEAKAGE_CONVERGED_TRANSIENT_PER_HOP  (5.0e-3 * 2.5)
#define AEC_LEAKAGE_DIVERGED_TRANSIENT_PER_HOP   (5.0e-1 * 2.5)

/* 1000 blocks at the reference grid */
#define AEC_POOR_EXC_COUNTER_INITIAL_HOPS  400

/* Unitless */
#define AEC_MAX_ERLE_LF  4.0
#define AEC_MAX_ERLE_HF  1.5
#define AEC_MIN_ERLE     1.0

#define AEC_STATIONARITY_THR_RATIO       10.0
#define AEC_STATIONARITY_BLOCK_FRACTION  0.75
#define AEC_STATIONARITY_ALPHA           0.004
#define AEC_STATIONARITY_ALPHA_INIT      0.04

/* Matched-filter amplitude limits (/32768, amplitude not PSD) */
#define AEC_MATCHED_FILTER_EXCITATION_LIMIT_FLOAT (150.0 / 32768.0)
#define AEC_MATCHED_FILTER_SATURATION_LIMIT_FLOAT (32000.0 / 32768.0)

/* PSD-scale render and floor constants */
#define AEC_NORMAL_RENDER_LIMIT_FLOAT (64.0  * AEC_PSD_SCALE_INV)
#define AEC_LOW_RENDER_LIMIT_FLOAT    (256.0 * AEC_PSD_SCALE_INV)
#define AEC_FLOOR_POWER_FLOAT         (128.0 * AEC_PSD_SCALE_INV)
#define AEC_FILTER_NOISE_GATE_POWER_FLOAT (20075344.0 * AEC_PSD_SCALE_INV)
#define AEC_STATIONARITY_MIN_NOISE_POWER_FLOAT (10.0 * AEC_PSD_SCALE_INV)
/* int16^2 scale, used verbatim */
#define AEC_RESIDUAL_NOISE_GATE_POWER 27509.42

#endif
