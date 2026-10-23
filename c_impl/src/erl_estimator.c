/* erl_estimator.c -- see erl_estimator.h. */
#include "erl_estimator.h"

#include "aec_scale.h"
#include "utility/simd_aec.h"

#include <limits.h>
#include <string.h>

void erl_estimator_init(ErlEstimator *e, int startup_phase_length_hops,
                        int n_bins, int hop_size, int sample_rate,
                        float *erl_storage, int *hold_counters_storage) {
    int k;
    e->startup_hops = startup_phase_length_hops;
    e->n_bins = n_bins;
    /* 4 s hold-after-drop, computed from the live grid. */
    e->hold_hops = aec_ms_to_hops(4000.0f, hop_size, sample_rate);
    /* x2_min is scaled for the live hop (reference hop 160). */
    e->x2_min = aec_per_bin_psd_threshold(ERL_X2_MIN, hop_size, 160);
    e->x2_min_total = e->x2_min * (float)n_bins;
    e->erl = erl_storage;
    e->hold_counters = hold_counters_storage;
    for (k = 0; k < n_bins; ++k) {
        e->erl[k] = ERL_MAX_ERL;
    }
    for (k = 0; k < n_bins - 2; ++k) {
        e->hold_counters[k] = 0;
    }
    e->erl_time_domain = ERL_MAX_ERL;
    e->hold_counter_time_domain = 0;
    e->blocks_since_reset = 0;
}

void erl_estimator_reset(ErlEstimator *e) {
    e->blocks_since_reset = 0;
}

void erl_estimator_update(ErlEstimator *e,
                          const float *render_psd,
                          const float *capture_psd,
                          int converged_filter) {
    const float *x2 = render_psd;
    const float *y2 = capture_psd;

    /* The counter stops at startup_hops so it cannot overflow. */
    if (e->blocks_since_reset < e->startup_hops) e->blocks_since_reset += 1;
    if (e->blocks_since_reset < e->startup_hops || !converged_filter) {
        return;
    }

    /* Skip DC/Nyquist; hold[j] maps to bin j + 1. */
    sk_erl_bin_update_f32(e->erl + 1, e->hold_counters, x2 + 1, y2 + 1,
                          e->x2_min, e->hold_hops, ERL_MIN_ERL,
                          e->n_bins - 2);

    /* Decrement every hold counter after the masked update, saturating at INT_MIN. */
    sk_dec1_floorintmin_s32(e->hold_counters, e->n_bins - 2);

    /* Bins with hold <= 0 double toward ERL_MAX_ERL. */
    sk_erl_hold_expire_f32(e->erl + 1, e->hold_counters, ERL_MAX_ERL,
                           e->n_bins - 2);

    /* Mirror endpoints. */
    e->erl[0] = e->erl[1];
    e->erl[e->n_bins - 1] = e->erl[e->n_bins - 2];

    /* Fullband ERL from pairwise float32 sums. */
    {
        float x2_sum = sk_pairwise_sum_tailfold_b_f32(x2, (size_t)e->n_bins);
        if (x2_sum > e->x2_min_total) {
            float y2_sum = sk_pairwise_sum_tailfold_b_f32(y2, (size_t)e->n_bins);
            float new_erl = y2_sum / x2_sum;
            if (new_erl < e->erl_time_domain) {
                e->hold_counter_time_domain = e->hold_hops;
                e->erl_time_domain += 0.1f * (new_erl - e->erl_time_domain);
                if (e->erl_time_domain < ERL_MIN_ERL) {
                    e->erl_time_domain = ERL_MIN_ERL;
                }
            }
        }
    }
    /* Decrement saturates at INT_MIN to avoid signed overflow. */
    if (e->hold_counter_time_domain > INT_MIN) e->hold_counter_time_domain -= 1;
    if (e->hold_counter_time_domain <= 0) {
        float doubled = 2.0f * e->erl_time_domain;
        e->erl_time_domain = doubled < ERL_MAX_ERL ? doubled : ERL_MAX_ERL;
    }
}

