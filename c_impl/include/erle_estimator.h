/* erle_estimator.h — C port of python/modules/state/erle_estimator.py.
 *
 * Thin wrapper over FullBandErleEstimator + SubbandErleEstimator with a
 * startup gate. NO float math of its own: it owns the two sub-estimators plus
 * an integer _blocks_since_reset counter and does the startup gating.
 *
 *   update(x2,y2,e2,converged_filter,coh_gate_mask):
 *     blocks_since_reset += 1
 *     if blocks_since_reset < startup_hops: return       (gate; sub-modules
 *                                                          untouched)
 *     subband.update(x2,y2,e2,converged_filter,coh_gate_mask)
 *     fullband.update(x2,y2,e2,converged_filter)
 *
 *   reset(delay_change):
 *     fullband.reset(); subband.reset()
 *     if delay_change: blocks_since_reset = 0
 *
 * No new parity surface here — it composes the two already-bit-exact ports;
 * the ordering (subband BEFORE fullband) is preserved verbatim from the .py
 * but neither reads the other, so order is observationally inert. Storage for
 * the two sub-estimators' per-bin arrays is caller-owned (see *_init).
 */
#ifndef ERLE_ESTIMATOR_H
#define ERLE_ESTIMATOR_H

#include <stdint.h>

#include "subband_erle.h"
#include "fullband_erle.h"

typedef struct {
    int               startup_hops;       /* _startup_hops              */
    int               blocks_since_reset;  /* _blocks_since_reset        */
    FullBandErleEstimator fullband;        /* _fullband                  */
    SubbandErle           subband;         /* _subband                   */
} ErleEstimator;

/* Mirrors ErleEstimator.__init__. The caller supplies the SubbandErle per-bin
 * backing arrays (all length n_bins, same set subband_erle_init wants):
 *   max_erle, erle, erle_oc, erle_unb, erle_during, y2_acc, e2_acc : float
 *   coming_onset, low_render                                       : uchar
 *   hold                                                           : int32_t
 *
 * Sub-estimator kwargs match the Python ErleEstimator ctor defaults:
 *   use_onset_detection=1, use_min_erle_during_onsets=1. */
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
                         unsigned char *low_render_st);

void erle_estimator_reset(ErleEstimator *e, int delay_change);

/* update(x2,y2,e2,converged_filter,coh_gate_mask).
 * coh_gate_mask may be NULL (matches Python coh_gate_mask=None). */
void erle_estimator_update(ErleEstimator *e,
                           const float *x2, const float *y2, const float *e2,
                           int converged_filter,
                           const unsigned char *coh_gate_mask);

/* erle(onset_compensated) -> subband.erle(onset_compensated). */
const float *erle_estimator_erle(const ErleEstimator *e, int onset_compensated);
const float *erle_estimator_erle_unbounded(const ErleEstimator *e);
double erle_estimator_fullband_erle_log2(const ErleEstimator *e);
/* get_inst_linear_quality_estimate(): *valid==0 mirrors a Python None. */
double erle_estimator_get_inst_linear_quality_estimate(const ErleEstimator *e,
                                                       int *valid);

#endif /* ERLE_ESTIMATOR_H */
