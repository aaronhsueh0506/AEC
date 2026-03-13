/**
 * subband_nlms.h - Frequency-domain NLMS adaptive filter
 *
 * Partitioned Block Frequency-Domain Adaptive Filter (PBFDAF)
 * More efficient than time-domain NLMS for long echo paths.
 *
 * Advantages:
 * - O(N log N) vs O(N^2) complexity
 * - Faster convergence due to frequency-domain whitening
 * - Per-bin step size adaptation
 */

#ifndef SUBBAND_NLMS_H
#define SUBBAND_NLMS_H

#include <stdbool.h>
#include "fft_wrapper.h"

#ifdef __cplusplus
extern "C" {
#endif

// Opaque subband NLMS handle
typedef struct SubbandNlms SubbandNlms;

/**
 * Create subband NLMS filter
 *
 * @param block_size FFT block size (e.g., 512)
 * @param n_partitions Number of filter partitions (filter_length / block_size)
 * @param mu Step size (0.1-0.5)
 * @param delta Regularization (1e-8)
 * @return Filter handle, or NULL on error
 */
SubbandNlms* subband_nlms_create(int block_size, int n_partitions,
                                  float mu, float delta);

/**
 * Destroy subband NLMS filter
 */
void subband_nlms_destroy(SubbandNlms* filter);

/**
 * Reset filter state (weights + buffers + power)
 */
void subband_nlms_reset(SubbandNlms* filter);

/**
 * Reset filter weights only (keep buffers and power estimates)
 * Used for retrain: faster reconvergence since X_buf history is preserved
 */
void subband_nlms_reset_weights(SubbandNlms* filter);

/**
 * Process one block of samples
 *
 * Uses overlap-save method for linear convolution
 *
 * @param filter Filter handle
 * @param near_end Microphone input [block_size/2] (hop_size samples)
 * @param far_end Reference input [block_size/2]
 * @param output Echo-cancelled output [block_size/2]
 * @param mu_scale Step-size scale factor [0.0, 1.0]. 0 = no update, 1 = full update
 * @return 0 on success
 */
int subband_nlms_process(SubbandNlms* filter,
                         const float* near_end,
                         const float* far_end,
                         float* output,
                         float mu_scale);

/**
 * Get echo estimate spectrum (for RES post-filter)
 *
 * @param filter Filter handle
 * @param echo_spec Output echo spectrum [n_freqs]
 */
void subband_nlms_get_echo_spectrum(const SubbandNlms* filter,
                                     Complex* echo_spec);

/**
 * Get error spectrum (for RES post-filter)
 *
 * @param filter Filter handle
 * @param error_spec Output error spectrum [n_freqs]
 */
void subband_nlms_get_error_spectrum(const SubbandNlms* filter,
                                      Complex* error_spec);

/**
 * Get block size
 */
int subband_nlms_get_block_size(const SubbandNlms* filter);

/**
 * Get hop size (block_size / 2)
 */
int subband_nlms_get_hop_size(const SubbandNlms* filter);

/**
 * Get number of frequency bins
 */
int subband_nlms_get_n_freqs(const SubbandNlms* filter);

/**
 * Get filter length in samples
 */
int subband_nlms_get_filter_length(const SubbandNlms* filter);

/**
 * Copy filter weights from src to dst (W only)
 * Both filters must have same block_size and n_partitions.
 *
 * @return 0 on success, -1 on mismatch
 */
int subband_nlms_copy_weights(SubbandNlms* dst, const SubbandNlms* src);

/**
 * Copy echo spectrum from src to dst (for RES consistency after weight copy)
 */
void subband_nlms_copy_echo_spec(SubbandNlms* dst, const SubbandNlms* src);

/**
 * Get near-end spectrum (for RES overlap-save)
 */
void subband_nlms_get_near_spectrum(const SubbandNlms* filter, Complex* near_spec);

/**
 * Get far-end spectrum (for coherence DTD)
 */
void subband_nlms_get_far_spectrum(const SubbandNlms* filter, Complex* far_spec);

/**
 * Get total error energy from last process() call
 * Returns sum of |error_spec[k]|^2 for all frequency bins.
 */
float subband_nlms_get_error_energy(const SubbandNlms* filter);

/**
 * Get number of partitions
 */
int subband_nlms_get_n_partitions(const SubbandNlms* filter);

#ifdef __cplusplus
}
#endif

#endif // SUBBAND_NLMS_H
