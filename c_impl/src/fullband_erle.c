/* fullband_erle.c — C port of python/modules/state/fullband_erle.py.
 *
 * The per-bin inputs x2/y2/e2 are float32 arrays; np.sum over them is numpy
 * pairwise summation in float32, then float(..) widens to double. Every scalar
 * accumulator / ratio / log2 / EMA / max-min / quality op is in double, exactly
 * as the Python module computes them (the _fast_log2 helper returns a Python
 * float from float(np.log2(double))). Built with -ffp-contract=off so each
 * double op rounds separately, matching numpy's per-op rounding.
 */
#include "fullband_erle.h"

#include "aec3_scale.h"   /* aec3_per_bin_psd_threshold */

#include <math.h>

/* numpy 1.26 pairwise_sum over float32 (verbatim copy of the verified routine
 * in reverb_frequency_response.c):
 *   n < 8     -> sequential
 *   n <= 128  -> 8 partial accumulators (stride 8), balanced-tree combine, then
 *                the (n % 8) tail sequentially
 *   n > 128   -> split at n2 = (n/2) rounded DOWN to a multiple of 8, recurse,
 *                add the two halves.
 * Every op is a float32 add (no float64 widening). */
float fb_erle_pairwise_sum(const float *a, size_t n) {
    if (n == 0) {
        return 0.0f;
    }
    if (n < 8) {
        float res = a[0];
        size_t i;
        for (i = 1; i < n; ++i) {
            res = res + a[i];
        }
        return res;
    }
    if (n <= 128) {
        float r0 = a[0], r1 = a[1], r2 = a[2], r3 = a[3];
        float r4 = a[4], r5 = a[5], r6 = a[6], r7 = a[7];
        float res;
        size_t i;
        for (i = 8; i + 8 <= n; i += 8) {
            r0 = r0 + a[i + 0];
            r1 = r1 + a[i + 1];
            r2 = r2 + a[i + 2];
            r3 = r3 + a[i + 3];
            r4 = r4 + a[i + 4];
            r5 = r5 + a[i + 5];
            r6 = r6 + a[i + 6];
            r7 = r7 + a[i + 7];
        }
        res = ((r0 + r1) + (r2 + r3)) + ((r4 + r5) + (r6 + r7));
        for (; i < n; ++i) {
            res = res + a[i];
        }
        return res;
    } else {
        size_t n2 = n / 2;
        n2 -= n2 % 8;
        return fb_erle_pairwise_sum(a, n2) +
               fb_erle_pairwise_sum(a + n2, n - n2);
    }
}

/* _fast_log2(x) = float(np.log2(x)). np.log2 on a Python float returns a
 * float64; double log2() matches it bit-for-bit under -ffp-contract=off. */
static double fast_log2(double x) {
    return log2(x);
}

/* ── _ErleInstantaneous ─────────────────────────────────────────────────────── */

static void inst_init(ErleInstantaneous *e, double quality_alpha) {
    e->clamp_zero = 1;            /* clamp_quality_to_zero default True */
    e->clamp_one = 1;            /* clamp_quality_to_one  default True */
    e->quality_alpha = quality_alpha;
    e->has_erle_log2 = 0;
    e->erle_log2 = 0.0;
    e->inst_quality_estimate = 0.0;
    e->max_erle_log2 = -10.0;
    e->min_erle_log2 = 33.0;
    e->y2_acum = 0.0;
    e->e2_acum = 0.0;
    e->num_points = 0;
}

static void inst_reset_accumulators(ErleInstantaneous *e) {
    e->has_erle_log2 = 0;
    e->erle_log2 = 0.0;          /* mirror _erle_log2 = None (value unused) */
    e->inst_quality_estimate = 0.0;
    e->num_points = 0;
    e->y2_acum = 0.0;
    e->e2_acum = 0.0;
}

static void inst_reset(ErleInstantaneous *e) {
    inst_reset_accumulators(e);
    e->max_erle_log2 = -10.0;
    e->min_erle_log2 = 33.0;
    e->inst_quality_estimate = 0.0;
}

static void inst_update_max_min(ErleInstantaneous *e) {
    e->max_erle_log2 -= FBERLE_MAXMIN_FORGET;
    if (e->erle_log2 > e->max_erle_log2) {
        e->max_erle_log2 = e->erle_log2;   /* max(_max, _erle) */
    }
    e->min_erle_log2 += FBERLE_MAXMIN_FORGET;
    if (e->erle_log2 < e->min_erle_log2) {
        e->min_erle_log2 = e->erle_log2;   /* min(_min, _erle) */
    }
}

static void inst_update_quality_estimate(ErleInstantaneous *e) {
    double alpha = e->quality_alpha;
    double quality = 0.0;
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

/* ErleInstantaneous::update -> returns whether estimates updated. */
static int inst_update(ErleInstantaneous *e, double y2_sum, double e2_sum) {
    int update_estimates = 0;
    e->e2_acum += e2_sum;
    e->y2_acum += y2_sum;
    e->num_points += 1;
    if (e->num_points == FBERLE_POINTS_TO_ACCUMULATE) {
        if (e->e2_acum > 0.0) {
            update_estimates = 1;
            e->erle_log2 = fast_log2(e->y2_acum / e->e2_acum + FBERLE_EPSILON);
            e->has_erle_log2 = 1;
        }
        e->num_points = 0;
        e->e2_acum = 0.0;
        e->y2_acum = 0.0;
    }
    if (update_estimates) {
        inst_update_max_min(e);
        inst_update_quality_estimate(e);
    }
    return update_estimates;
}

/* get_quality_estimate() — *valid = 0 mirrors a None return (erle_log2 None). */
static double inst_get_quality_estimate(const ErleInstantaneous *e, int *valid) {
    double value;
    if (!e->has_erle_log2) {
        *valid = 0;
        return 0.0;
    }
    *valid = 1;
    value = e->inst_quality_estimate;
    if (e->clamp_zero && value < 0.0) {
        value = 0.0;
    }
    if (e->clamp_one && value > 1.0) {
        value = 1.0;
    }
    return value;
}

/* ── FullBandErleEstimator ──────────────────────────────────────────────────── */

static void update_quality_estimate(FullBandErleEstimator *s) {
    int valid = 0;
    double q = inst_get_quality_estimate(&s->inst, &valid);
    s->has_linear_filter_quality = valid;
    s->linear_filter_quality = q;
}

void fb_erle_init(FullBandErleEstimator *s, int n_freqs,
                        double min_erle, double max_erle_l, int hop_size) {
    s->n_freqs = n_freqs;
    s->td_alpha = FBERLE_TD_ALPHA;
    s->x2_band_energy_threshold =
        aec3_per_bin_psd_threshold(FBERLE_X2_BAND_ENERGY_THRESHOLD, hop_size, 160);
    s->min_erle_log2 = fast_log2(min_erle + FBERLE_EPSILON);
    s->max_erle_lf_log2 = fast_log2(max_erle_l + FBERLE_EPSILON);
    s->hold_counter_inst_erle = 0;
    s->erle_time_domain_log2 = s->min_erle_log2;
    inst_init(&s->inst, FBERLE_QUALITY_ALPHA);
    s->has_linear_filter_quality = 0;
    s->linear_filter_quality = 0.0;
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
        /* x2_sum = float(np.sum(x2)) — pairwise f32 then widen to double. */
        double x2_sum = (double)fb_erle_pairwise_sum(x2, (size_t)s->n_freqs);
        if (x2_sum > s->x2_band_energy_threshold * (double)s->n_freqs) {
            double y2_sum = (double)fb_erle_pairwise_sum(y2, (size_t)s->n_freqs);
            double e2_sum = (double)fb_erle_pairwise_sum(e2, (size_t)s->n_freqs);
            if (inst_update(&s->inst, y2_sum, e2_sum)) {
                s->hold_counter_inst_erle = FBERLE_BLOCKS_TO_HOLD_ERLE;
                s->erle_time_domain_log2 +=
                    s->td_alpha * (s->inst.erle_log2 - s->erle_time_domain_log2);
                if (s->erle_time_domain_log2 < s->min_erle_log2) {
                    s->erle_time_domain_log2 = s->min_erle_log2;
                }
            }
        }
    }
    s->hold_counter_inst_erle -= 1;
    if (s->hold_counter_inst_erle == 0) {
        inst_reset_accumulators(&s->inst);
    }
    update_quality_estimate(s);
}

double fb_erle_log2(const FullBandErleEstimator *s) {
    return s->erle_time_domain_log2;
}

double fb_erle_get_inst_linear_quality_estimate(const FullBandErleEstimator *s,
                                                      int *valid) {
    *valid = s->has_linear_filter_quality;
    return s->linear_filter_quality;
}
