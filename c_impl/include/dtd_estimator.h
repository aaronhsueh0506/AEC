/**
 * dtd_estimator.h - Double-Talk Detector (coherence + divergence modes)
 * Port of Python aec.py DtdEstimator (lines 2080-2280, coherence + divergence).
 * Geigel mode (NLMS-only) NOT ported.
 */

#ifndef DTD_ESTIMATOR_H
#define DTD_ESTIMATOR_H

#include "fft_wrapper.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef enum {
    DTD_MODE_COHERENCE = 0,
    DTD_MODE_DIVERGENCE = 1
} DtdMode;

typedef struct DtdEstimator DtdEstimator;

/**
 * Create coherence-mode DTD. Needs spectral inputs.
 * Python defaults: coh_alpha=0.85, coh_high=0.6, coh_low=0.3,
 *                  energy_floor=0.01, abs_floor=1e-6,
 *                  attack=0.3, release=0.05, hangover_max=15, warmup=50.
 * Voice band weight: 300-4000 Hz at 3.0; <100 or >6000 at 0.3; else 1.0.
 */
DtdEstimator* dtd_create_coherence(int n_freqs, int sample_rate, int block_size);

/**
 * Create divergence-mode DTD. Operates on time-domain energies.
 * Python: divergence_factor=1.5, attack=0.3, release=0.05.
 */
DtdEstimator* dtd_create_divergence(void);

void dtd_destroy(DtdEstimator* d);
void dtd_reset(DtdEstimator* d);

/**
 * Detect for coherence mode: pass error_spec + far_spec [n_freqs] complex.
 * Returns post-update confidence [0,1].
 */
float dtd_detect_coherence(DtdEstimator* d, const Complex* error_spec, const Complex* far_spec);

/**
 * Detect for divergence mode: pass near_end + output time-domain hops.
 * Returns post-update confidence [0,1].
 */
float dtd_detect_divergence(DtdEstimator* d, const float* near_end, const float* output, int hop);

float dtd_get_confidence(const DtdEstimator* d);
void  dtd_scale_confidence(DtdEstimator* d, float factor); /* used by EPC shadow_rise */

#ifdef __cplusplus
}
#endif
#endif
