/**
 * nlms_filter.h - Time-domain NLMS adaptive filter
 *
 * Normalized Least Mean Squares algorithm for echo cancellation
 * Reference: Haykin, "Adaptive Filter Theory"
 */

#ifndef NLMS_FILTER_H
#define NLMS_FILTER_H

#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

// Opaque NLMS filter handle
typedef struct NlmsFilter NlmsFilter;

/**
 * Create NLMS/LMS filter
 *
 * @param filter_length Filter length in samples
 * @param mu Step size (NLMS: 0.1-0.8, LMS: 0.001-0.05)
 * @param delta Regularization constant (1e-8)
 * @param leak Weight leakage factor (0.9999 for NLMS, 1.0 for LMS)
 * @param normalize true=NLMS (power normalization), false=LMS (fixed step)
 * @return Filter handle, or NULL on error
 */
NlmsFilter* nlms_create(int filter_length, float mu, float delta, float leak,
                         bool normalize);

/**
 * Destroy NLMS filter
 */
void nlms_destroy(NlmsFilter* filter);

/**
 * Reset NLMS filter state
 */
void nlms_reset(NlmsFilter* filter);

/**
 * Process single sample through NLMS filter
 *
 * @param filter NLMS filter handle
 * @param near_end Microphone input sample (d[n])
 * @param far_end Reference/loudspeaker sample (x[n])
 * @param update_weights If false, only compute output without updating
 * @return Echo-cancelled output sample (e[n])
 */
float nlms_process_sample(NlmsFilter* filter,
                          float near_end,
                          float far_end,
                          bool update_weights);

/**
 * Process block of samples through NLMS filter
 *
 * @param filter NLMS filter handle
 * @param near_end Microphone input [num_samples]
 * @param far_end Reference input [num_samples]
 * @param output Echo-cancelled output [num_samples]
 * @param echo_est Estimated echo output [num_samples] (can be NULL)
 * @param num_samples Number of samples
 * @param update_weights If false, only compute output without updating
 */
void nlms_process_block(NlmsFilter* filter,
                        const float* near_end,
                        const float* far_end,
                        float* output,
                        float* echo_est,
                        int num_samples,
                        bool update_weights);

/**
 * Get current estimated echo for a sample (without processing)
 *
 * @param filter NLMS filter handle
 * @return Estimated echo based on current weights and reference buffer
 */
float nlms_get_echo_estimate(const NlmsFilter* filter);

/**
 * Get filter length
 */
int nlms_get_filter_length(const NlmsFilter* filter);

/**
 * Get current step size
 */
float nlms_get_mu(const NlmsFilter* filter);

/**
 * Set step size (for adaptive control)
 */
void nlms_set_mu(NlmsFilter* filter, float mu);

/**
 * Set clear-history mode: clear ref_buffer at start of each block
 *
 * When enabled, the filter has no carry-over between blocks.
 * Default: false (keep 1 hop_size of history in circular buffer)
 */
void nlms_set_clear_history(NlmsFilter* filter, bool clear);

/**
 * Get current reference signal power (for debugging/monitoring)
 */
float nlms_get_ref_power(const NlmsFilter* filter);

#ifdef __cplusplus
}
#endif

#endif // NLMS_FILTER_H
