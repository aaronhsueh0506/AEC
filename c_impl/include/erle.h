/* erle.h — 1:1 port of FilterErleEstimator + FullbandErleEstimator +
 * compute_erle_confidence (python/aec.py 916-986).
 *
 * Numerical notes:
 *  - FilterErleEstimator inputs (echo_spec, error_spec) are complex64;
 *    intermediate alpha/EMA in Python become fp64 due to Python-scalar
 *    `dt_factor` mixing into np.where → final ".astype(float32)" pulls
 *    back to fp32. C mirrors with `double` intermediates + `float` storage.
 *  - FullbandErleEstimator is fp64 throughout (`self.fb_erle` is Python float).
 */
#ifndef AEC_ERLE_H
#define AEC_ERLE_H

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/* ── FilterErleEstimator ──────────────────────────────────── */
typedef struct FilterErleEst {
    int    n_freqs;
    float* erle;          /* [n_freqs], stored as float32 (matches Python) */
    double alpha_rise;    /* 0.95 */
    double alpha_drop;    /* 0.7 */
} FilterErleEst;

void filter_erle_init(FilterErleEst* f, int n_freqs);
void filter_erle_free(FilterErleEst* f);
void filter_erle_reset(FilterErleEst* f);
/* echo_spec / error_spec packed (re,im) per bin (n_freqs entries each) */
void filter_erle_update(FilterErleEst* f,
                           const float* echo_re, const float* echo_im,
                           const float* err_re,  const float* err_im,
                           int far_active, double dt_indicator);

/* ── FullbandErleEstimator ────────────────────────────────── */
typedef struct FullbandErleEst {
    double fb_erle;       /* fp64, matches Python float */
    double alpha;         /* 0.97 */
} FullbandErleEst;

void fullband_erle_init(FullbandErleEst* f);
void fullband_erle_reset(FullbandErleEst* f);
void fullband_erle_update(FullbandErleEst* f,
                             double near_power, double error_power,
                             int far_active, double dt_indicator);

/* ── compute_erle_confidence (pure) ───────────────────────── */
double erle_compute_confidence(const float* erle_l1, int n_freqs,
                                  double fb_erle);

#ifdef __cplusplus
}
#endif

#endif /* AEC_ERLE_H */
