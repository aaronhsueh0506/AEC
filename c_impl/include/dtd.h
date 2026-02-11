/**
 * dtd.h - Double-Talk Detector
 *
 * Detects when both near-end and far-end speakers are active simultaneously.
 * During double-talk, adaptive filter updates should be disabled to prevent
 * divergence.
 *
 * Methods implemented:
 * 1. Geigel DTD: Compares near-end energy to max far-end energy
 * 2. Energy ratio: Compares error energy to echo estimate energy
 */

#ifndef DTD_H
#define DTD_H

#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

// Opaque DTD handle
typedef struct DtdEstimator DtdEstimator;

/**
 * Create DTD estimator
 *
 * @param window_length Window length for max tracking (samples)
 * @param threshold Geigel threshold (0.4-0.8, typical 0.6)
 * @param hangover_frames Hangover duration in frames
 * @param energy_ratio_threshold Energy ratio threshold (0.3-0.5)
 * @return DTD handle, or NULL on error
 */
DtdEstimator* dtd_create(int window_length,
                         float threshold,
                         int hangover_frames,
                         float energy_ratio_threshold);

/**
 * Destroy DTD estimator
 */
void dtd_destroy(DtdEstimator* dtd);

/**
 * Reset DTD state
 */
void dtd_reset(DtdEstimator* dtd);

/**
 * Detect double-talk for a single sample
 *
 * @param dtd DTD handle
 * @param near_end Microphone input sample
 * @param far_end Reference/loudspeaker sample
 * @param error Error signal sample (after adaptive filter)
 * @param echo_est Echo estimate sample
 * @return true if double-talk detected
 */
bool dtd_detect_sample(DtdEstimator* dtd,
                       float near_end,
                       float far_end,
                       float error,
                       float echo_est);

/**
 * Detect double-talk for a block of samples
 *
 * @param dtd DTD handle
 * @param near_end Microphone input [num_samples]
 * @param far_end Reference input [num_samples]
 * @param error Error signal [num_samples]
 * @param echo_est Echo estimate [num_samples]
 * @param num_samples Number of samples
 * @return true if double-talk detected in this block
 */
bool dtd_detect_block(DtdEstimator* dtd,
                      const float* near_end,
                      const float* far_end,
                      const float* error,
                      const float* echo_est,
                      int num_samples);

/**
 * Get current DTD state
 */
bool dtd_is_active(const DtdEstimator* dtd);

/**
 * Get DTD confidence (0.0 = no double-talk, 1.0 = certain double-talk)
 */
float dtd_get_confidence(const DtdEstimator* dtd);

/**
 * Set threshold (for runtime tuning)
 */
void dtd_set_threshold(DtdEstimator* dtd, float threshold);

/**
 * Get current far-end activity level (for debugging)
 */
float dtd_get_far_end_level(const DtdEstimator* dtd);

#ifdef __cplusplus
}
#endif

#endif // DTD_H
