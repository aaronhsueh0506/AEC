/* dtd_v2.h — port of aec.py DtdEstimator (2080-2281).
 * BALANCED preset has enable_dtd=False, but DTD ported for completeness.
 */
#ifndef AEC_V2_DTD_H
#define AEC_V2_DTD_H

#include "fft_wrapper.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef enum {
    DTD_GEIGEL = 0,
    DTD_DIVERGENCE,
    DTD_COHERENCE,
} DtdMode;

typedef struct DtdEstimatorV2 {
    DtdMode mode;
    double  confidence;
    double  attack, release;
    int     warmup_frames;
    int     frame_count;

    /* Geigel */
    double* far_abs_buffer;
    int     buf_len;
    int     buf_idx;
    double  geigel_threshold;
    int     hangover_max;
    int     hangover_count;

    /* Divergence */
    double  divergence_factor;

    /* Coherence */
    int      n_freqs;
    Complex* S_ex;            /* NULL if mode != coherence */
    float*   S_ee;            /* fp32 (matches Python np.float32) */
    float*   S_xx;
    float*   voice_weight;
    double   coh_alpha;
    double   coh_high;
    double   coh_low;
    double   coh_energy_floor;
    double   coh_abs_floor;
} DtdEstimatorV2;

void dtd_v2_init_geigel(DtdEstimatorV2* d, int window_blocks,
                        double geigel_threshold, int hangover_max,
                        double attack, double release, int warmup_frames);
void dtd_v2_init_divergence(DtdEstimatorV2* d, double divergence_factor,
                            double attack, double release, int warmup_frames);
void dtd_v2_init_coherence(DtdEstimatorV2* d, int n_freqs,
                           int sample_rate, int block_size,
                           double coh_alpha, double coh_high, double coh_low,
                           double coh_energy_floor, double coh_abs_floor,
                           double attack, double release, int warmup_frames);
void dtd_v2_free(DtdEstimatorV2* d);
void dtd_v2_reset(DtdEstimatorV2* d);

/* For DTD_GEIGEL: pass near_end/far_end (output/error_spec/far_spec NULL).
 * For DTD_DIVERGENCE: pass near_end + output.
 * For DTD_COHERENCE: pass error_spec + far_spec. */
double dtd_v2_detect_block(DtdEstimatorV2* d,
                           const float*   near_end, int near_len,
                           const float*   far_end,  int far_len,
                           const float*   output,
                           const Complex* error_spec,
                           const Complex* far_spec);

#ifdef __cplusplus
}
#endif

#endif /* AEC_V2_DTD_H */
