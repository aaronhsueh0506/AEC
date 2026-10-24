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

/* 1000 blocks at the reference grid */
#define AEC_POOR_EXC_COUNTER_INITIAL_HOPS  400

/* PSD-scale floor constant */
#define AEC_FILTER_NOISE_GATE_POWER_FLOAT (20075344.0 * AEC_PSD_SCALE_INV)

#endif
