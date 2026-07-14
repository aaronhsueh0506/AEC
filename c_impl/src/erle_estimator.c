/* erle_estimator.c — C port of python/modules/state/erle_estimator.py.
 * Thin startup-gated wrapper composing FullBandErleEstimator +
 * SubbandErleEstimator. See erle_estimator.h for the spec. */
#include "erle_estimator.h"

void erle_estimator_init(ErleEstimator *e,
                         int startup_phase_length_hops,
                         int n_bins,
                         float min_erle, float max_erle_l, float max_erle_h,
                         int use_onset_detection,
                         int hop_size,
                         float *max_erle_st, float *erle_st, float *erle_oc_st,
                         float *erle_unb_st, float *erle_during_st,
                         unsigned char *coming_onset_st, int32_t *hold_st,
                         float *y2_acc_st, float *e2_acc_st,
                         unsigned char *low_render_st) {
    e->startup_hops       = startup_phase_length_hops;
    e->blocks_since_reset = 0;
    /* Python ctor: FullBandErleEstimator(min_erle, max_erle_l, hop_size). */
    fb_erle_init(&e->fullband, n_bins, min_erle, max_erle_l, hop_size);
    /* Python ctor: SubbandErleEstimator(n_bins, min_erle, max_erle_l,
     *   max_erle_h, use_onset_detection, hop_size). use_min_erle_during_onsets
     *   defaults True in SubbandErleEstimator.__init__. */
    subband_erle_init(&e->subband, n_bins,
                      min_erle, max_erle_l, max_erle_h,
                      use_onset_detection, /*use_min_erle_during_onsets=*/1,
                      hop_size,
                      max_erle_st, erle_st, erle_oc_st, erle_unb_st,
                      erle_during_st, coming_onset_st, hold_st,
                      y2_acc_st, e2_acc_st, low_render_st);
}

void erle_estimator_reset(ErleEstimator *e, int delay_change) {
    fb_erle_reset(&e->fullband);
    subband_erle_reset(&e->subband);
    if (delay_change) {
        e->blocks_since_reset = 0;
    }
}

void erle_estimator_update(ErleEstimator *e,
                           const float *x2, const float *y2, const float *e2,
                           int converged_filter,
                           const unsigned char *coh_gate_mask) {
    e->blocks_since_reset += 1;
    if (e->blocks_since_reset < e->startup_hops) {
        return;
    }
    subband_erle_update(&e->subband, x2, y2, e2, converged_filter,
                        coh_gate_mask);
    fb_erle_update(&e->fullband, x2, y2, e2, converged_filter);
}

const float *erle_estimator_erle(const ErleEstimator *e, int onset_compensated) {
    return subband_erle_erle(&e->subband, onset_compensated);
}

const float *erle_estimator_erle_unbounded(const ErleEstimator *e) {
    return subband_erle_erle_unbounded(&e->subband);
}

float erle_estimator_fullband_erle_log2(const ErleEstimator *e) {
    return fb_erle_log2(&e->fullband);
}

float erle_estimator_get_inst_linear_quality_estimate(const ErleEstimator *e,
                                                       int *valid) {
    return fb_erle_get_inst_linear_quality_estimate(&e->fullband, valid);
}
