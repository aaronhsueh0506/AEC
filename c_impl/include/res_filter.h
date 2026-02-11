/**
 * res_filter.h - Residual Echo Suppressor (Post-Filter)
 *
 * Suppresses residual echo that the adaptive filter cannot remove.
 * Uses echo-to-error ratio (EER) based spectral suppression.
 *
 * Typical improvement: additional 10-20 dB suppression
 */

#ifndef RES_FILTER_H
#define RES_FILTER_H

#include "fft_wrapper.h"

#ifdef __cplusplus
extern "C" {
#endif

// Opaque RES handle
typedef struct ResFilter ResFilter;

/**
 * Create RES filter
 *
 * @param n_freqs Number of frequency bins
 * @param g_min_db Minimum gain in dB (e.g., -20)
 * @param over_sub Over-subtraction factor (1.0-3.0)
 * @param alpha Gain smoothing factor (0.8)
 * @return RES handle, or NULL on error
 */
ResFilter* res_create(int n_freqs, float g_min_db, float over_sub, float alpha);

/**
 * Destroy RES filter
 */
void res_destroy(ResFilter* res);

/**
 * Reset RES state
 */
void res_reset(ResFilter* res);

/**
 * Process spectrum through RES
 *
 * @param res RES handle
 * @param error_spec Error spectrum from adaptive filter [n_freqs]
 * @param echo_spec Estimated echo spectrum [n_freqs]
 * @param far_power Far-end power (for activity detection)
 * @param output_spec Enhanced output spectrum [n_freqs]
 */
void res_process(ResFilter* res,
                 const Complex* error_spec,
                 const Complex* echo_spec,
                 float far_power,
                 Complex* output_spec);

/**
 * Get current suppression gains (for visualization/debugging)
 *
 * @param res RES handle
 * @param gains Output gain array [n_freqs]
 */
void res_get_gains(const ResFilter* res, float* gains);

/**
 * Set minimum gain
 */
void res_set_g_min(ResFilter* res, float g_min_db);

/**
 * Set over-subtraction factor
 */
void res_set_over_sub(ResFilter* res, float over_sub);

#ifdef __cplusplus
}
#endif

#endif // RES_FILTER_H
