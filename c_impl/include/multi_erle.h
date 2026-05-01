/**
 * multi_erle.h - Multi-ERLE estimators (Phase 2 port from Python aec.py)
 *
 * - FilterErleEstimator: per-bin ERLE = |echo|²/|error|² with asymmetric EMA
 * - FullbandErleEstimator: broadband ERLE = near_pwr/error_pwr scalar
 * - compute_erle_confidence: L1/L2 cross-validation
 *
 * Matches Python aec.py lines 916-986 (v3.8.1).
 */

#ifndef MULTI_ERLE_H
#define MULTI_ERLE_H

#include "fft_wrapper.h"

#ifdef __cplusplus
extern "C" {
#endif

/* ========== FilterErleEstimator ========== */
typedef struct FilterErleEstimator FilterErleEstimator;

FilterErleEstimator* fe_create(int n_freqs);
void fe_destroy(FilterErleEstimator* fe);
void fe_reset(FilterErleEstimator* fe);

/**
 * Update per-bin ERLE.
 * @param echo_spec  [n_freqs] complex echo estimate from filter
 * @param error_spec [n_freqs] complex error from filter
 * @param far_active 1 if far-end active (caller-computed)
 * @param dt_indicator [0,1] DT confidence
 *
 * Skips update if far_active=0 (matches Python behavior).
 */
void fe_update(FilterErleEstimator* fe,
               const Complex* echo_spec,
               const Complex* error_spec,
               int far_active,
               float dt_indicator);

/* Read-only access to per-bin erle array (length n_freqs). */
const float* fe_get_erle(const FilterErleEstimator* fe);
int          fe_get_n_freqs(const FilterErleEstimator* fe);


/* ========== FullbandErleEstimator ========== */
typedef struct FullbandErleEstimator FullbandErleEstimator;

FullbandErleEstimator* fbe_create(void);
void fbe_destroy(FullbandErleEstimator* fbe);
void fbe_reset(FullbandErleEstimator* fbe);

/**
 * Update broadband ERLE (FS-only: gated by far_active AND dt_indicator <= 0.3
 * AND near_power >= 1e-8).
 */
void fbe_update(FullbandErleEstimator* fbe,
                float near_power, float error_power,
                int far_active, float dt_indicator);

float fbe_get_fb_erle(const FullbandErleEstimator* fbe);


/* ========== compute_erle_confidence (function, no state) ========== */

/**
 * Cross-validation between per-bin (L1) and broadband (L2) ERLE.
 * Returns [0, 1]: 1 = consistent, 0 = divergent.
 */
float compute_erle_confidence(const float* erle_l1, int n_freqs, float fb_erle);


#ifdef __cplusplus
}
#endif

#endif /* MULTI_ERLE_H */
