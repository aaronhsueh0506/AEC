/**
 * pbfdkf.h - Partitioned Block Frequency-Domain Kalman Filter
 *
 * PBFDAF with FDKF Kalman adaptation (always on).
 * Independent hop_size (160) from block_size (512).
 * Overlap-save convolution.
 */

#ifndef PBFDKF_H
#define PBFDKF_H

#include "fft_wrapper.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct Pbfdkf Pbfdkf;

/**
 * Create PBFDKF filter
 *
 * @param block_size  FFT size (512)
 * @param hop_size    Processing hop (160, must be <= block_size/2)
 * @param n_partitions Number of filter partitions (10)
 * @param delta       Regularization (1e-8)
 * @param q_high      Initial Kalman Q (process noise, 1e-4)
 * @param q_low       Stable tracking Q (1e-7)
 * @return Filter handle, or NULL on error
 */
Pbfdkf* pbfdkf_create(int block_size, int hop_size, int n_partitions,
                       float delta, float q_high, float q_low);

/**
 * Destroy filter
 */
void pbfdkf_destroy(Pbfdkf* f);

/**
 * Reset all state (weights + buffers + Kalman state)
 */
void pbfdkf_reset(Pbfdkf* f);

/**
 * Reset weights and Kalman P only (keep buffers for reconvergence)
 */
void pbfdkf_reset_weights(Pbfdkf* f);

/**
 * Process hop_size samples
 *
 * @param f         Filter handle
 * @param near_end  Microphone input [hop_size]
 * @param far_end   Reference input [hop_size]
 * @param output    Echo-cancelled output [hop_size]
 * @param mu_scale  Adaptation scale [0,1] (DT protection)
 * @return 0 on success
 */
int pbfdkf_process(Pbfdkf* f,
                   const float* near_end,
                   const float* far_end,
                   float* output,
                   float mu_scale);

/**
 * Switch Q to low (stable tracking mode after convergence)
 */
void pbfdkf_switch_q_low(Pbfdkf* f);

/**
 * Set Q_high value (and reset Q to Q_high)
 */
void pbfdkf_set_q_high(Pbfdkf* f, float q_high);

/**
 * Set Q_high with ratio (shadow filter: Q = main_Q × ratio)
 */
void pbfdkf_set_q_ratio(Pbfdkf* f, float q_high, float q_low, float ratio);

/**
 * Copy weights + Kalman P from src to dst
 * Both must have same dimensions.
 */
int pbfdkf_copy_weights(Pbfdkf* dst, const Pbfdkf* src);

/**
 * Get total error energy (sum |error_spec[k]|^2)
 */
float pbfdkf_get_error_energy(const Pbfdkf* f);

/**
 * Get echo spectrum pointer (read-only, n_freqs elements)
 */
const Complex* pbfdkf_get_echo_spec(const Pbfdkf* f);

/**
 * Get far-end spectrum pointer (read-only)
 */
const Complex* pbfdkf_get_far_spec(const Pbfdkf* f);

/**
 * Get near-end (mic) spectrum pointer (read-only)
 */
const Complex* pbfdkf_get_near_spec(const Pbfdkf* f);

/**
 * Get far-end power (smoothed, per-bin mean)
 */
float pbfdkf_get_far_power(const Pbfdkf* f);

/* Getters */
int pbfdkf_get_block_size(const Pbfdkf* f);
int pbfdkf_get_hop_size(const Pbfdkf* f);
int pbfdkf_get_n_freqs(const Pbfdkf* f);
int pbfdkf_get_n_partitions(const Pbfdkf* f);

#ifdef __cplusplus
}
#endif

#endif /* PBFDKF_H */
