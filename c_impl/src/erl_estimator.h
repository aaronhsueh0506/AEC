/* Per-bin echo return loss (ERL = Y2/X2) with minimum-statistics tracking, float32.
 *  erl[k] for k = 1..n_bins-2 (endpoints mirror their neighbours) plus a fullband
 *  erl_time_domain. A downward jump moves the estimate 10% toward the new ratio and arms a
 *  hold counter; bins whose hold has expired double toward ERL_MAX_ERL each update.
 *  Python reference: modules/state/erl_estimator.py. */
#ifndef ERL_ESTIMATOR_H
#define ERL_ESTIMATOR_H

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

#define ERL_MIN_ERL  0.01f
#define ERL_MAX_ERL  1000.0f
/* Reference X2 minimum, scaled by aec_per_bin_psd_threshold() in erl_estimator_init(). */
#define ERL_X2_MIN  44015068.0f

/* Caller-provided storage: erl_storage (n_bins floats), hold_counters_storage (n_bins - 2 int32).
 * */
typedef struct {
    int     startup_hops;
    int     n_bins;
    int     hold_hops;
    float   x2_min;
    float   x2_min_total;
    float  *erl;
    int    *hold_counters;
    float   erl_time_domain;
    int     hold_counter_time_domain;
    int     blocks_since_reset;
} ErlEstimator;

/* hop_size scales x2_min; hop_size and sample_rate set hold_hops. */
void erl_estimator_init(ErlEstimator *e, int startup_phase_length_hops,
                        int n_bins, int hop_size, int sample_rate,
                        float *erl_storage, int *hold_counters_storage);

void erl_estimator_reset(ErlEstimator *e);

/* render_psd = X2, capture_psd = Y2 (n_bins floats each). */
void erl_estimator_update(ErlEstimator *e,
                          const float *render_psd,
                          const float *capture_psd,
                          int converged_filter);

#ifdef __cplusplus
}
#endif

#endif
