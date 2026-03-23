/**
 * res_filter.h - Residual Echo Suppressor with WOLA
 *
 * WOLA (Weighted Overlap-Add) with sqrt-Hann window:
 * - Analysis: frame_size=320 × sqrt-Hann → zero-pad → FFT(512) → 257 bins
 * - Synthesis: IFFT → truncate → sqrt-Hann → OLA
 *
 * Features: coherence-based NL echo PSD, per-bin near-end gate,
 * dynamic g_min, spectral floor, rate limiting, CNG.
 */

#ifndef RES_FILTER_H
#define RES_FILTER_H

#include "fft_wrapper.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct ResFilter ResFilter;

/**
 * RES configuration
 */
typedef struct {
    int block_size;             /* FFT size (512) */
    int frame_size;             /* WOLA window length (320) */
    int hop_size;               /* Processing hop (160) */
    int n_freqs;                /* block_size/2 + 1 = 257 */
    float g_min_db;             /* Minimum gain in dB */
    float max_drop_db_per_frame; /* 6.0 */
    float max_rise_db_per_frame; /* 6.0 */
    float spectral_floor_db;    /* -25.0 */
    float ne_protect_db;        /* -8.0 */
    int enable_cng;             /* 1 */
} ResConfig;

/**
 * Create RES filter
 */
ResFilter* res_create(const ResConfig* cfg);

/**
 * Destroy RES filter
 */
void res_destroy(ResFilter* res);

/**
 * Reset RES state
 */
void res_reset(ResFilter* res);

/**
 * Process hop_size error samples → hop_size enhanced output
 *
 * @param res          RES handle
 * @param error_hop    Error signal from adaptive filter [hop_size]
 * @param echo_spec    Echo estimate spectrum [n_freqs] (from PBFDKF)
 * @param far_spec     Far-end spectrum [n_freqs] (for coherence)
 * @param far_power    Far-end mean power (activity detection)
 * @param converged    Filter convergence flag
 * @param erle_factor  [0,1] convergence factor for EER blending
 * @param dt_indicator [0,1] double-talk indicator
 * @param over_sub     Over-subtraction factor (dynamic, set by coordinator)
 * @param output_hop   Enhanced output [hop_size]
 */
void res_process(ResFilter* res,
                 const float* error_hop,
                 const Complex* echo_spec,
                 const Complex* far_spec,
                 float far_power,
                 int converged,
                 float erle_factor,
                 float dt_indicator,
                 float over_sub,
                 float* output_hop);

/**
 * Get echo_psd and error_psd arrays (for per-bin mu_scale)
 */
const float* res_get_echo_psd(const ResFilter* res);
const float* res_get_error_psd(const ResFilter* res);

#ifdef __cplusplus
}
#endif

#endif /* RES_FILTER_H */
