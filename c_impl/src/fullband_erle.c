/* fullband_erle.c -- see fullband_erle.h. Float32 throughout. */
#include "fullband_erle.h"

#include "aec_scale.h"
#include "utility/simd_aec.h"

#include <limits.h>
#include <math.h>

static float fast_log2(float x) {
    return log2f(x);
}

/* Instantaneous ERLE */

static void inst_init(ErleInstantaneous *e, float quality_alpha,
                       float maxmin_forget) {
    e->clamp_zero = 1;
    e->clamp_one = 1;
    e->quality_alpha = quality_alpha;
    e->maxmin_forget = maxmin_forget;
    e->has_erle_log2 = 0;
    e->erle_log2 = 0.0f;
    e->inst_quality_estimate = 0.0f;
    e->max_erle_log2 = -10.0f;
    e->min_erle_log2 = 33.0f;
    e->y2_acum = 0.0f;
    e->e2_acum = 0.0f;
    e->num_points = 0;
}

static void inst_reset_accumulators(ErleInstantaneous *e) {
    e->has_erle_log2 = 0;
    e->erle_log2 = 0.0f;
    e->inst_quality_estimate = 0.0f;
    e->num_points = 0;
    e->y2_acum = 0.0f;
    e->e2_acum = 0.0f;
}

static void inst_reset(ErleInstantaneous *e) {
    inst_reset_accumulators(e);
    e->max_erle_log2 = -10.0f;
    e->min_erle_log2 = 33.0f;
    e->inst_quality_estimate = 0.0f;
}

static void inst_update_max_min(ErleInstantaneous *e) {
    e->max_erle_log2 -= e->maxmin_forget;
    if (e->erle_log2 > e->max_erle_log2) {
        e->max_erle_log2 = e->erle_log2;
    }
    e->min_erle_log2 += e->maxmin_forget;
    if (e->erle_log2 < e->min_erle_log2) {
        e->min_erle_log2 = e->erle_log2;
    }
}

static void inst_update_quality_estimate(ErleInstantaneous *e) {
    float alpha = e->quality_alpha;
    float quality = 0.0f;
    if (e->max_erle_log2 > e->min_erle_log2) {
        quality = (e->erle_log2 - e->min_erle_log2) /
                  (e->max_erle_log2 - e->min_erle_log2);
    }
    if (quality > e->inst_quality_estimate) {
        e->inst_quality_estimate = quality;
    } else {
        e->inst_quality_estimate += alpha * (quality - e->inst_quality_estimate);
    }
}

/* Returns whether the estimates were updated. */
static int inst_update(ErleInstantaneous *e, float y2_sum, float e2_sum) {
    int update_estimates = 0;
    e->e2_acum += e2_sum;
    e->y2_acum += y2_sum;
    e->num_points += 1;
    if (e->num_points == FBERLE_POINTS_TO_ACCUMULATE) {
        if (e->e2_acum > 0.0f) {
            update_estimates = 1;
            e->erle_log2 = fast_log2(e->y2_acum / e->e2_acum + FBERLE_EPSILON);
            e->has_erle_log2 = 1;
        }
        e->num_points = 0;
        e->e2_acum = 0.0f;
        e->y2_acum = 0.0f;
    }
    if (update_estimates) {
        inst_update_max_min(e);
        inst_update_quality_estimate(e);
    }
    return update_estimates;
}

/* *valid = 0 until the first accumulation completes. */
static float inst_get_quality_estimate(const ErleInstantaneous *e, int *valid) {
    float value;
    if (!e->has_erle_log2) {
        *valid = 0;
        return 0.0f;
    }
    *valid = 1;
    value = e->inst_quality_estimate;
    if (e->clamp_zero && value < 0.0f) {
        value = 0.0f;
    }
    if (e->clamp_one && value > 1.0f) {
        value = 1.0f;
    }
    return value;
}

/* Fullband estimator */

static void update_quality_estimate(FullBandErleEstimator *s) {
    int valid = 0;
    float q = inst_get_quality_estimate(&s->inst, &valid);
    s->has_linear_filter_quality = valid;
    s->linear_filter_quality = q;
}

void fb_erle_init(FullBandErleEstimator *s, int n_freqs,
                        float min_erle, float max_erle_l, int hop_size,
                        int sample_rate) {
    s->n_freqs = n_freqs;
    /* The EMAs are per 4 ms block and fire once per 6-hop accumulation window. The factor of 6
     * cancels, so plain hop_size is used against the single-block denominator. */
    s->td_alpha = aec_per_block_ema_alpha_to_per_hop(
        0.05f, hop_size, sample_rate);
    s->quality_alpha = aec_per_block_ema_alpha_to_per_hop(
        0.07f, hop_size, sample_rate);
    /* Additive per-event forget step on the same 6-point cadence: linear rescale, not the EMA
     * form. */
    float maxmin_forget = aec_per_block_rate_to_per_hop(
        FBERLE_MAXMIN_FORGET, hop_size, sample_rate);
    /* 400 ms ERLE hold, scaled to the live grid. */
    s->blocks_to_hold_erle = aec_ms_to_hops(400.0f, hop_size, sample_rate);
    s->x2_band_energy_threshold =
        aec_per_bin_psd_threshold(FBERLE_X2_BAND_ENERGY_THRESHOLD, hop_size, 160);
    s->min_erle_log2 = fast_log2(min_erle + FBERLE_EPSILON);
    s->max_erle_lf_log2 = fast_log2(max_erle_l + FBERLE_EPSILON);
    s->hold_counter_inst_erle = 0;
    s->erle_time_domain_log2 = s->min_erle_log2;
    inst_init(&s->inst, s->quality_alpha, maxmin_forget);
    s->has_linear_filter_quality = 0;
    s->linear_filter_quality = 0.0f;
    fb_erle_reset(s);
}

void fb_erle_reset(FullBandErleEstimator *s) {
    inst_reset(&s->inst);
    update_quality_estimate(s);
    s->erle_time_domain_log2 = s->min_erle_log2;
    s->hold_counter_inst_erle = 0;
}

void fb_erle_update(FullBandErleEstimator *s,
                          const float *x2, const float *y2, const float *e2,
                          int converged_filter) {
    if (converged_filter) {
        float x2_sum = sk_pairwise_sum_tailfold_b_f32(x2, (size_t)s->n_freqs);
        if (x2_sum > s->x2_band_energy_threshold * (float)s->n_freqs) {
            float y2_sum = sk_pairwise_sum_tailfold_b_f32(y2, (size_t)s->n_freqs);
            float e2_sum = sk_pairwise_sum_tailfold_b_f32(e2, (size_t)s->n_freqs);
            if (inst_update(&s->inst, y2_sum, e2_sum)) {
                s->hold_counter_inst_erle = s->blocks_to_hold_erle;
                s->erle_time_domain_log2 +=
                    s->td_alpha * (s->inst.erle_log2 - s->erle_time_domain_log2);
                if (s->erle_time_domain_log2 < s->min_erle_log2) {
                    s->erle_time_domain_log2 = s->min_erle_log2;
                }
            }
        }
    }
    /* Saturates at INT_MIN, not 0: the `== 0` check below is an edge trigger that must fire once
     * per re-arm cycle, and flooring at 0 would re-fire it every hop. */
    if (s->hold_counter_inst_erle > INT_MIN) s->hold_counter_inst_erle -= 1;
    if (s->hold_counter_inst_erle == 0) {
        inst_reset_accumulators(&s->inst);
    }
    update_quality_estimate(s);
}

float fb_erle_log2(const FullBandErleEstimator *s) {
    return s->erle_time_domain_log2;
}

float fb_erle_get_inst_linear_quality_estimate(const FullBandErleEstimator *s,
                                                      int *valid) {
    *valid = s->has_linear_filter_quality;
    return s->linear_filter_quality;
}
