/* Fullband ERLE accumulator in log2 units, float32. Reads:
 *  - fb_erle_log2(): EMA in the log2 domain.
 *  - fb_erle_get_inst_linear_quality_estimate(): rolling [0, 1] linear-filter quality, with *valid
 *  == 0 until the first 6-point accumulation completes.
 * Sums over the per-bin inputs are float32 pairwise sums (sk_pairwise_sum_tailfold_b_f32).
 * Python reference: modules/state/fullband_erle.py. */
#ifndef FULLBAND_ERLE_H
#define FULLBAND_ERLE_H

#include <stddef.h>

#define FBERLE_EPSILON              1.0e-3f
#define FBERLE_POINTS_TO_ACCUMULATE 6
/* Reference value, scaled to the live hop by aec_per_bin_psd_threshold() in fb_erle_init(). */
#define FBERLE_X2_BAND_ENERGY_THRESHOLD 44015068.0f
/* Max/min envelope forget step; rescaled to the live grid in fb_erle_init(). */
#define FBERLE_MAXMIN_FORGET        0.0004f

typedef struct {
    int    clamp_zero;
    int    clamp_one;
    float  quality_alpha;
    float  maxmin_forget;
    int    has_erle_log2;
    float  erle_log2;
    float  inst_quality_estimate;
    float  max_erle_log2;
    float  min_erle_log2;
    float  y2_acum;
    float  e2_acum;
    int    num_points;
} ErleInstantaneous;

typedef struct {
    int    n_freqs;
    float  x2_band_energy_threshold; /* per-bin X^2 gate (already hop-scaled,
                                       * result of the float32 threshold helper) */
    float  min_erle_log2;
    float  td_alpha;
    float  quality_alpha;
    int    blocks_to_hold_erle;
    int    hold_counter_inst_erle;
    float  erle_time_domain_log2;
    ErleInstantaneous inst;
    int    has_linear_filter_quality;
    float  linear_filter_quality;
} FullBandErleEstimator;

/* hop_size scales x2_band_energy_threshold; hop_size/sample_rate set blocks_to_hold_erle,
 * quality_alpha and td_alpha. */
void fb_erle_init(FullBandErleEstimator *s, int n_freqs,
                        float min_erle, int hop_size, int sample_rate);

void fb_erle_reset(FullBandErleEstimator *s);

/* x2/y2/e2 are float32 arrays of length n_freqs. */
void fb_erle_update(FullBandErleEstimator *s,
                          const float *x2, const float *y2, const float *e2,
                          int converged_filter);

float fb_erle_log2(const FullBandErleEstimator *s);

/* *valid = 0 when there is no estimate yet. */
float fb_erle_get_inst_linear_quality_estimate(const FullBandErleEstimator *s,
                                                      int *valid);

#endif
