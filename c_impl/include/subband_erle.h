/* Per-bin ERLE estimator with onset compensation, float32. Python reference:
 * modules/state/subband_erle.py.
 * Every 6th converged point a per-bin ERLE (y2_acc / max(e2_acc, 1e-30)) is formed when e2_acc > 0
 * and the optional gates pass; onset detection holds the estimate. Three EMAs: erle and
 * erle_onset_compensated (capped at max_erle[k]) and erle_unbounded (capped at 1e5); bins 0 and
 * n-1 mirror their neighbour. Storage is caller-supplied (see subband_erle_init()). */
#ifndef SUBBAND_ERLE_H
#define SUBBAND_ERLE_H

#include <stdint.h>

#define SE_POINTS_TO_ACCUMULATE   6
#define SE_UNBOUNDED_ERLE_MAX     100000.0f
/* Reference X2 band-energy threshold, scaled to the live hop in init. */
#define SE_X2_BAND_ENERGY_THRESHOLD 44015068.0f

typedef struct {
    int    n_bins;
    float  min_erle;
    int    use_onset_detection;
    int    use_min_erle_during_onsets;
    float  alpha_up;
    float  alpha_down;
    float  onset_release_decay;
    float  x2_band_energy_threshold;
    int    blocks_to_hold_erle;
    int    blocks_for_onset_detection;

    float   *max_erle;
    float   *erle;
    float   *erle_onset_compensated;
    float   *erle_unbounded;
    float   *erle_during_onsets;
    unsigned char *coming_onset;
    int32_t *hold_counters;
    float   *y2_acc;
    float   *e2_acc;
    unsigned char *low_render_energy;
    int      num_points;
} SubbandErle;

/* Storage, each of length n_bins: max_erle, erle, erle_onset_compensated, erle_unbounded,
 * erle_during_onsets, y2_acc, e2_acc (float); coming_onset, low_render_energy (unsigned char);
 * hold_counters (int32_t). hop_size only scales the x2 band-energy threshold (reference hop 160). */
void subband_erle_init(SubbandErle *s, int n_bins,
                       float min_erle, float max_erle_l, float max_erle_h,
                       int use_onset_detection, int use_min_erle_during_onsets,
                       int hop_size, int sample_rate,
                       float *max_erle_st, float *erle_st, float *erle_oc_st,
                       float *erle_unb_st, float *erle_during_st,
                       unsigned char *coming_onset_st, int32_t *hold_st,
                       float *y2_acc_st, float *e2_acc_st,
                       unsigned char *low_render_st);

void subband_erle_reset(SubbandErle *s);

/* coh_gate_mask may be NULL. */
void subband_erle_update(SubbandErle *s,
                         const float *x2, const float *y2, const float *e2,
                         int converged_filter,
                         const unsigned char *coh_gate_mask);

/* Returns erle_onset_compensated when onset_compensated && use_onset_detection, else erle. */
const float *subband_erle_erle(const SubbandErle *s, int onset_compensated);
const float *subband_erle_erle_unbounded(const SubbandErle *s);

#endif
