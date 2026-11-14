/* Startup-gated wrapper over FullBandErleEstimator and SubbandErle: updates are ignored until
 * startup_hops blocks have elapsed since a reset with delay_change=1.
 * Per-bin storage is caller-owned (see init). Python reference: modules/state/erle_estimator.py. */
#ifndef ERLE_ESTIMATOR_H
#define ERLE_ESTIMATOR_H

#include <stdint.h>

#include "subband_erle.h"
#include "fullband_erle.h"

typedef struct {
    int               startup_hops;
    int               blocks_since_reset;
    FullBandErleEstimator fullband;
    SubbandErle           subband;
} ErleEstimator;

/* Per-bin backing arrays, each of length n_bins: max_erle, erle, erle_oc, erle_unb, erle_during,
 * y2_acc, e2_acc (float); coming_onset, low_render (uchar); hold (int32_t).
 * The subband estimator is created with use_min_erle_during_onsets = 1. */
void erle_estimator_init(ErleEstimator *e,
                         int startup_phase_length_hops,
                         int n_bins,
                         float min_erle, float max_erle_l, float max_erle_h,
                         int use_onset_detection,
                         int hop_size, int sample_rate,
                         float *max_erle_st, float *erle_st, float *erle_oc_st,
                         float *erle_unb_st, float *erle_during_st,
                         unsigned char *coming_onset_st, int32_t *hold_st,
                         float *y2_acc_st, float *e2_acc_st,
                         unsigned char *low_render_st);

void erle_estimator_reset(ErleEstimator *e, int delay_change);

/* coh_gate_mask may be NULL. */
void erle_estimator_update(ErleEstimator *e,
                           const float *x2, const float *y2, const float *e2,
                           int converged_filter,
                           const unsigned char *coh_gate_mask);

/* Accessors: subband ERLE (onset_compensated selects the compensated array), unbounded ERLE,
 * fullband log2 ERLE. */
const float *erle_estimator_erle(const ErleEstimator *e, int onset_compensated);
const float *erle_estimator_erle_unbounded(const ErleEstimator *e);
float erle_estimator_fullband_erle_log2(const ErleEstimator *e);
/* Instantaneous linear quality estimate; *valid == 0 when there is no estimate. */
float erle_estimator_get_inst_linear_quality_estimate(const ErleEstimator *e,
                                                       int *valid);

#endif /* ERLE_ESTIMATOR_H */
