/* delay_aec3.c — BIT-EXACT C port of python/modules/delay/.
 * See delay_aec3.h for the chain overview and parity notes.
 *
 * Build (standalone, do NOT link aec.c):
 *   gcc -Wall -Wextra -O2 -ffp-contract=off -std=gnu99 -Iinclude \
 *       src/delay_aec3.c test/parity_delay.c -lm -o /tmp/p_delay
 */
#include "delay_aec3.h"

#include <math.h>
#include <string.h>

/* float-domain equivalents of the AEC3 int16 thresholds (matched_filter.py) */
#define DA_EXCITATION_LIMIT   (150.0f   / 32768.0f)   /* 0.00457763671875 */
#define DA_SATURATION_LIMIT   (32000.0f / 32768.0f)   /* 0.9765625 */
#define DA_SMOOTHING_FAST     0.7f
#define DA_SMOOTHING_SLOW     0.1f
#define DA_MATCHING_THRESHOLD 0.3   /* used as f64 multiplier of anchor */

/* ------------------------------------------------------------------ helpers */

/* np.dot over float32 inputs.  numpy routes f32 dot through OpenBLAS SDOT
 * (not portably bit-reproducible); we accumulate the float32 products in a
 * double running sum -- the most faithful scalar value.  The downstream
 * integer lag / DelayQuality are robust to the sub-ULP difference (verified
 * bit-exact on the golden). */
static double delay_aec3_dot(const float *a, const float *b, int n) {
    double s = 0.0;
    int i;
    for (i = 0; i < n; ++i) s += (double)(a[i] * b[i]);
    return s;
}

/* numpy argmax over h*h (float32): FIRST index of the strongest squared tap.
 * Mirrors max_square_peak_index (returns 0 on size<2). */
static int da_max_square_peak_index(const float *h, int n) {
    int best = 0, i;
    float best_v;
    if (n < 2) return 0;
    best_v = h[0] * h[0];
    for (i = 1; i < n; ++i) {
        float v = h[i] * h[i];
        if (v > best_v) { best_v = v; best = i; }
    }
    return best;
}

/* numpy argmax over an int histogram: FIRST index of the maximum. */
static int da_argmax_i(const int *a, int n) {
    int best = 0, i;
    int best_v = a[0];
    for (i = 1; i < n; ++i) {
        if (a[i] > best_v) { best_v = a[i]; best = i; }
    }
    return best;
}

/* ------------------------------------------------------------------ biquad */

static const double DA_LP_B[3][3] = {
    {0.0180919877, 0.00320961363, 0.0180919877},
    {1.0,         -1.24550459,    1.0},
    {1.0,         -1.4221681,     1.0},
};
static const double DA_LP_A[3][2] = {
    {-1.5183195,   0.633165865},
    {-1.49784254,  0.853586692},
    {-1.49791282,  0.969572384},
};
static const double DA_HP_B[1][3] = {
    {0.757076375, -1.51415275, 0.757076375},
};
static const double DA_HP_A[1][2] = {
    {-1.45424359, 0.574061915},
};

static void da_biquad_reset(DaBiquad *bq) {
    memset(bq->z, 0, sizeof(bq->z));
}

static void da_biquad_init_lp(DaBiquad *bq) {
    int s;
    bq->n_sections = 3;
    for (s = 0; s < 3; ++s) {
        bq->b[s][0] = DA_LP_B[s][0]; bq->b[s][1] = DA_LP_B[s][1]; bq->b[s][2] = DA_LP_B[s][2];
        bq->a[s][0] = DA_LP_A[s][0]; bq->a[s][1] = DA_LP_A[s][1];
    }
    da_biquad_reset(bq);
}

static void da_biquad_init_hp(DaBiquad *bq) {
    bq->n_sections = 1;
    bq->b[0][0] = DA_HP_B[0][0]; bq->b[0][1] = DA_HP_B[0][1]; bq->b[0][2] = DA_HP_B[0][2];
    bq->a[0][0] = DA_HP_A[0][0]; bq->a[0][1] = DA_HP_A[0][1];
    da_biquad_reset(bq);
}

/* Process one sample through the cascade, returning the double output.
 * Direct-form-II transposed, all-float64 (mirrors _CascadedBiquad.process).
 *   out = b0*x + z0
 *   z0  = b1*x - a1*out + z1
 *   z1  = b2*x - a2*out
 */
static double da_biquad_process1(DaBiquad *bq, double xn) {
    int s;
    double yn = xn;
    for (s = 0; s < bq->n_sections; ++s) {
        double inp = yn;
        double out = bq->b[s][0] * inp + bq->z[s][0];
        bq->z[s][0] = bq->b[s][1] * inp - bq->a[s][0] * out + bq->z[s][1];
        bq->z[s][1] = bq->b[s][2] * inp - bq->a[s][1] * out;
        yn = out;
    }
    return yn;
}

/* ---------------------------------------------------------------- decimator */

static void da_decimator_init(DaDecimator *dec) {
    da_biquad_init_lp(&dec->anti_alias);
    da_biquad_init_hp(&dec->noise_reduction);
}

/* Decimate a 64-sample block to a 16-sample sub-block.
 * Python: x = lp.process(block) [f64], x = hp.process(x) [f64], then
 * x[::4].astype(f32).  The biquad chains run on ALL 64 samples (state
 * carries), but only every 4th post-HP sample is kept & cast to f32. */
static void da_decimator_decimate(DaDecimator *dec, const float *in_block, float *out_sub) {
    int i, o = 0;
    for (i = 0; i < DA_AEC3_BLOCK_SIZE; ++i) {
        double lp = da_biquad_process1(&dec->anti_alias, (double)in_block[i]);
        double hp = da_biquad_process1(&dec->noise_reduction, lp);
        if ((i % DA_DOWN_SAMPLING_FACTOR) == 0) {
            out_sub[o++] = (float)hp;
        }
    }
}

/* ------------------------------------------------------------------- ring */

static void da_ring_reset(DaRing *r) {
    memset(r->buffer, 0, sizeof(r->buffer));
    r->write = 0;
}

static void da_ring_push(DaRing *r, const float *sub, int n) {
    int i;
    for (i = 0; i < n; ++i) {
        r->buffer[r->write] = sub[i];
        r->write = (r->write + 1) % DA_RING_CAPACITY;
    }
}

/* gather_back: return `length` samples in time-reversed order starting at
 * newest_index() - start_offset.  out[0] = sample start_offset ago. */
static void da_ring_gather_back(const DaRing *r, int start_offset, int length, float *out) {
    /* Values identical to the per-sample-modulo walk, but copied as at most two
     * contiguous reversed segments — no integer division in the inner loop.
     * This gather runs 80x per 64-sample block (5 filters x 16 sub-samples,
     * 512 taps each), so the old per-sample `% DA_RING_CAPACITY` was ~10M
     * integer divisions/sec at 16 kHz and dominated the matched-filter cost
     * (measured 2.8x speedup of the delay-est chain from this change alone). */
    int newest = (r->write - 1 + DA_RING_CAPACITY) % DA_RING_CAPACITY;
    int end = (newest - start_offset + DA_RING_CAPACITY) % DA_RING_CAPACITY;
    int k = 0;
    int first = (end + 1 < length) ? end + 1 : length;
    const float *buf = r->buffer;
    for (; k < first; ++k) out[k] = buf[end - k];
    if (k < length) {
        int idx2 = DA_RING_CAPACITY - 1;
        int base = k;
        for (; k < length; ++k) out[k] = buf[idx2 - (k - base)];
    }
}

/* -------------------------------------------------------- matched filter core */

/* Mirrors matched_filter_core(). Returns filters_updated; writes *error_sum. */
static int da_matched_filter_core(const DaRing *ring, int alignment_shift_back,
                                  double x2_sum_threshold, float smoothing,
                                  const float *y, float *h, double *error_sum_out,
                                  float *instantaneous_error_out) {
    double error_sum = 0.0;
    int filters_updated = 0;
    int i;
    float x_window[DA_FILTER_SIZE];
    if (instantaneous_error_out) {
        int k0;
        for (k0 = 0; k0 < DA_ACC_ERR_SIZE; ++k0) instantaneous_error_out[k0] = 0.0f;
    }
    for (i = 0; i < DA_SUB_BLOCK_SIZE; ++i) {
        int start_back = alignment_shift_back + (DA_SUB_BLOCK_SIZE - 1 - i);
        double x2_sum, s, e;
        int saturation, t;
        da_ring_gather_back(ring, start_back, DA_FILTER_SIZE, x_window);
        x2_sum = delay_aec3_dot(x_window, x_window, DA_FILTER_SIZE);
        s = delay_aec3_dot(h, x_window, DA_FILTER_SIZE);
        e = (double)y[i] - s;
        saturation = (y[i] >= DA_SATURATION_LIMIT || y[i] <= -DA_SATURATION_LIMIT);
        error_sum += e * e;
        if (instantaneous_error_out) {
            /* AEC3 MatchedFilterCoreWithAccumulatedError (matched_filter.cc:106-139):
             * accumulate, per filter-tap GROUP of 4, the squared error of the filter
             * PREFIX up to that group: instE[j] += (Σ h[:4(j+1)]·x[:4(j+1)] − y[i])².
             * ComputePreEchoLag then walks back for the earliest group whose prefix
             * already explains the echo (= onset). Mirrors numpy exactly: f32 product,
             * f32 sequential group-sum (.sum over 4), f32 cumsum prefix, then the
             * (prefix − y[i]) subtraction and the += accumulation in float64.
             * Uses pre-NLMS-update h (the update below runs after this block). */
            float prefix = 0.0f;
            int j;
            const float *hp = h;
            const float *xp = x_window;
            for (j = 0; j < DA_ACC_ERR_SIZE; ++j) {
                float gs = hp[0] * xp[0];
                gs = gs + hp[1] * xp[1];
                gs = gs + hp[2] * xp[2];
                gs = gs + hp[3] * xp[3];
                prefix = prefix + gs;
                {
                    double d = (double)prefix - (double)y[i];
                    instantaneous_error_out[j] =
                        (float)((double)instantaneous_error_out[j] + d * d);
                }
                hp += DA_ACCUMULATED_ERROR_SUBSAMPLE_RATE;
                xp += DA_ACCUMULATED_ERROR_SUBSAMPLE_RATE;
            }
        }
        if (x2_sum > x2_sum_threshold && !saturation) {
            /* alpha = smoothing * e / x2_sum  (Python double); then the array
             * op casts alpha to float32 and computes h += f32(alpha)*x in f32. */
            double alpha_d = (double)smoothing * e / x2_sum;
            float  alpha_f = (float)alpha_d;
            for (t = 0; t < DA_FILTER_SIZE; ++t) {
                h[t] = h[t] + (alpha_f * x_window[t]);
            }
            filters_updated = 1;
        }
    }
    *error_sum_out = error_sum;
    return filters_updated;
}

/* Mirrors _update_accumulated_error.  `instantaneous` carries the per-tap-prefix
 * squared error filled by da_matched_filter_core for the last filter in the bank
 * (matches Python's single shared _instantaneous_error buffer). */
static void da_update_accumulated_error(const float *instantaneous, float *accumulated,
                                        double one_over_anchor) {
    int k;
    float ooa_f = (float)one_over_anchor;
    for (k = 0; k < DA_ACC_ERR_SIZE; ++k) {
        float norm = instantaneous[k] * ooa_f;   /* array scalar rule: f32 */
        if (norm < accumulated[k]) {
            accumulated[k] = norm;
        } else {
            float diff = norm - accumulated[k];
            accumulated[k] = accumulated[k] + (0.015f * diff);
        }
    }
}

/* Mirrors _compute_pre_echo_lag. */
static int da_compute_pre_echo_lag(const float *accumulated_error, int lag,
                                   int alignment_shift_winner) {
    int pre_echo_lag, maximum, k;
    if (lag < alignment_shift_winner) return lag;
    pre_echo_lag = lag - alignment_shift_winner;
    maximum = pre_echo_lag / DA_ACCUMULATED_ERROR_SUBSAMPLE_RATE;
    if (DA_ACC_ERR_SIZE < maximum) maximum = DA_ACC_ERR_SIZE;
    for (k = maximum - 1; k >= 0; --k) {
        if (accumulated_error[k] > 0.5f) break;
        pre_echo_lag = (k + 1) * DA_ACCUMULATED_ERROR_SUBSAMPLE_RATE - 1;
    }
    return pre_echo_lag + alignment_shift_winner;
}

static void da_matched_filter_init(DaMatchedFilter *mf) {
    int n, k;
    memset(mf->filters, 0, sizeof(mf->filters));
    for (n = 0; n < DA_NUM_FILTERS; ++n)
        for (k = 0; k < DA_ACC_ERR_SIZE; ++k) mf->accumulated_error[n][k] = 1.0f;
    memset(mf->instantaneous_error, 0, sizeof(mf->instantaneous_error));
    mf->last_detected_best_lag_filter = -1;
    mf->number_pre_echo_updates = 0;
    mf->reported_valid = 0;
    mf->reported_lag = 0;
    mf->reported_pre_echo_lag = 0;
    mf->winner_lag_valid = 0;
    mf->winner_lag = 0;
}

static void da_matched_filter_reset(DaMatchedFilter *mf, int full_reset) {
    memset(mf->filters, 0, sizeof(mf->filters));
    mf->winner_lag_valid = 0;
    mf->reported_valid = 0;
    if (full_reset) {
        int n, k;
        for (n = 0; n < DA_NUM_FILTERS; ++n)
            for (k = 0; k < DA_ACC_ERR_SIZE; ++k) mf->accumulated_error[n][k] = 1.0f;
        mf->number_pre_echo_updates = 0;
    }
}

/* Mirrors MatchedFilter.update (detect_pre_echo=True). */
static void da_matched_filter_update(DaMatchedFilter *mf, const DaRing *ring,
                                     const float *capture, int use_slow_smoothing) {
    float smoothing = use_slow_smoothing ? DA_SMOOTHING_SLOW : DA_SMOOTHING_FAST;
    double x2_sum_threshold = (double)DA_FILTER_SIZE
                              * (double)DA_EXCITATION_LIMIT * (double)DA_EXCITATION_LIMIT;
    double error_sum_anchor = delay_aec3_dot(capture, capture, DA_SUB_BLOCK_SIZE);
    double winner_error_sum = error_sum_anchor;
    int winner_index = -1;
    int alignment_shift = 0;
    int prev_lag_valid = 0, prev_lag = 0;
    int n;

    mf->winner_lag_valid = 0;
    mf->reported_valid = 0;

    for (n = 0; n < DA_NUM_FILTERS; ++n) {
        double error_sum;
        int filters_updated, lag_estimate, reliable, lag;
        filters_updated = da_matched_filter_core(ring, alignment_shift, x2_sum_threshold,
                                                 smoothing, capture, mf->filters[n], &error_sum,
                                                 mf->instantaneous_error);
        lag_estimate = da_max_square_peak_index(mf->filters[n], DA_FILTER_SIZE);
        reliable = (lag_estimate > 2
                    && lag_estimate < (DA_FILTER_SIZE - 10)
                    && error_sum < DA_MATCHING_THRESHOLD * error_sum_anchor);
        lag = lag_estimate + alignment_shift;
        if (filters_updated && reliable && error_sum < winner_error_sum) {
            winner_error_sum = error_sum;
            winner_index = n;
            if (prev_lag_valid && prev_lag == lag) {
                mf->winner_lag = prev_lag;
                mf->winner_lag_valid = 1;
                winner_index = n - 1;
            } else {
                mf->winner_lag = lag;
                mf->winner_lag_valid = 1;
            }
        }
        prev_lag = lag;
        prev_lag_valid = 1;
        alignment_shift += DA_FILTER_INTRA_SHIFT;
    }

    if (winner_index != -1 && mf->winner_lag_valid) {
        mf->reported_valid = 1;
        mf->reported_lag = mf->winner_lag;
        mf->reported_pre_echo_lag = mf->winner_lag;
        /* detect_pre_echo path (always enabled in our config). */
        if (mf->last_detected_best_lag_filter == winner_index) {
            /* AEC3 gate is int16-scale (error_sum > 1.0 in Q15²); the float[-1,1]
             * equivalent is 1.0 / 32768² ≈ 9.3e-10 (matches matched_filter.py). */
            if (error_sum_anchor > 1.0 / (32768.0 * 32768.0)) {
                da_update_accumulated_error(mf->instantaneous_error,
                                            mf->accumulated_error[winner_index],
                                            1.0 / error_sum_anchor);
                mf->number_pre_echo_updates += 1;
            }
            if (mf->number_pre_echo_updates >= DA_PRE_ECHO_UPDATES_TO_REPORT) {
                int pre_echo = da_compute_pre_echo_lag(
                    mf->accumulated_error[winner_index], mf->winner_lag,
                    winner_index * DA_FILTER_INTRA_SHIFT);
                mf->reported_pre_echo_lag = pre_echo;
            }
        }
        mf->last_detected_best_lag_filter = winner_index;
    }
}

/* ---------------------------------------------------- highest-peak aggregator */

static void da_highest_peak_reset(DaHighestPeak *hp) {
    memset(hp->histogram, 0, sizeof(hp->histogram));
    memset(hp->ring, 0, sizeof(hp->ring));
    hp->ring_index = 0;
    /* candidate is NOT reset in Python HighestPeakAggregator.reset; preserve. */
}

static void da_highest_peak_init(DaHighestPeak *hp) {
    da_highest_peak_reset(hp);
    hp->candidate = -1;
}

static void da_highest_peak_aggregate(DaHighestPeak *hp, int lag) {
    int old_lag = hp->ring[hp->ring_index];
    hp->histogram[old_lag] -= 1;
    if (lag < 0) lag = 0;
    else if (lag >= DA_HP_HIST_SIZE) lag = DA_HP_HIST_SIZE - 1;
    hp->ring[hp->ring_index] = lag;
    hp->histogram[lag] += 1;
    hp->ring_index = (hp->ring_index + 1) % DA_HIST_WINDOW;
    hp->candidate = da_argmax_i(hp->histogram, DA_HP_HIST_SIZE);
}

/* ------------------------------------------------------- pre-echo aggregator */

static int da_get_ds_block_size_log2(int down_sampling_factor) {
    int ds_log2 = 0;
    int f = down_sampling_factor >> 1;
    int v;
    while (f > 0) { ds_log2 += 1; f >>= 1; }
    v = DA_BLOCK_SIZE_LOG2 - ds_log2;
    return v < 0 ? 0 : v;
}

static void da_pre_echo_reset(DaPreEcho *pe) {
    int i;
    memset(pe->histogram, 0, sizeof(pe->histogram));
    for (i = 0; i < DA_HIST_WINDOW; ++i) pe->ring[i] = -1;
    pe->ring_index = 0;
    pe->number_updates = 0;
    pe->pre_echo_candidate = 0;
}

static void da_pre_echo_init(DaPreEcho *pe) {
    pe->block_size_log2 = da_get_ds_block_size_log2(DA_DOWN_SAMPLING_FACTOR);
    da_pre_echo_reset(pe);
}

static void da_pre_echo_aggregate(DaPreEcho *pe, int pre_echo_lag) {
    int pbs = pre_echo_lag >> pe->block_size_log2;
    int old;
    if (pbs < 0) pbs = 0;
    else if (pbs > DA_PE_HIST_SIZE - 1) pbs = DA_PE_HIST_SIZE - 1;
    old = pe->ring[pe->ring_index];
    if (old != -1) pe->histogram[old] -= 1;
    pe->ring[pe->ring_index] = pbs;
    pe->histogram[pbs] += 1;
    pe->ring_index = (pe->ring_index + 1) % DA_HIST_WINDOW;

    if (pe->number_updates < DA_K_NUM_BLOCKS_PER_SEC * 2) {
        int window = DA_K_MFW_SUB_BLOCKS;
        double penalty = 1.0, best_value = -1.0;
        int best_idx = 0, n = DA_PE_HIST_SIZE, i = 0;
        pe->number_updates += 1;
        while (n - i >= window) {
            int end = i + window;
            int local_max_offset = 0, j;
            int seg_best = pe->histogram[i];
            for (j = 1; j < window; ++j) {
                if (pe->histogram[i + j] > seg_best) { seg_best = pe->histogram[i + j]; local_max_offset = j; }
            }
            {
                double local_max_value = (double)seg_best * penalty;
                if (local_max_value > best_value) {
                    best_value = local_max_value;
                    best_idx = i + local_max_offset;
                }
            }
            penalty *= 0.7;
            i = end;
        }
        pe->pre_echo_candidate = best_idx << pe->block_size_log2;
    } else {
        int best_idx = da_argmax_i(pe->histogram, DA_PE_HIST_SIZE);
        pe->pre_echo_candidate = best_idx << pe->block_size_log2;
    }
}

/* ---------------------------------------------------------- lag aggregator */

static void da_aggregator_init(DaLagAggregator *agg) {
    da_highest_peak_init(&agg->highest_peak);
    da_pre_echo_init(&agg->pre_echo);
    agg->significant_candidate_found = 0;
}

static void da_aggregator_reset(DaLagAggregator *agg, int hard_reset) {
    da_highest_peak_reset(&agg->highest_peak);
    da_pre_echo_reset(&agg->pre_echo);
    if (hard_reset) agg->significant_candidate_found = 0;
}

static int da_aggregator_reliable_delay_found(const DaLagAggregator *agg) {
    return agg->significant_candidate_found;
}

static int da_aggregator_delay_at_highest_peak(const DaLagAggregator *agg) {
    return agg->highest_peak.candidate;
}

/* Mirrors MatchedFilterLagAggregator.aggregate. Returns 1 if an estimate was
 * produced (writes *quality, *delay in DOWNSAMPLED samples). */
static int da_aggregator_aggregate(DaLagAggregator *agg, int lag_estimate_valid,
                                   int lag_estimate_lag, int lag_estimate_pre_echo_lag,
                                   DelayQuality *quality_out, int *delay_out) {
    int pre_lag, lag, candidate, hist_val;
    if (!lag_estimate_valid) return 0;
    /* feed pre-echo BEFORE primary */
    pre_lag = lag_estimate_pre_echo_lag - DA_HEADROOM;
    if (pre_lag < 0) pre_lag = 0;
    da_pre_echo_aggregate(&agg->pre_echo, pre_lag);
    lag = lag_estimate_lag - DA_HEADROOM;
    if (lag < 0) lag = 0;
    da_highest_peak_aggregate(&agg->highest_peak, lag);
    candidate = agg->highest_peak.candidate;
    hist_val = agg->highest_peak.histogram[candidate];
    if (agg->significant_candidate_found || hist_val > DA_THRESH_CONVERGED)
        agg->significant_candidate_found = 1;
    if (hist_val > DA_THRESH_CONVERGED ||
        (hist_val > DA_THRESH_INITIAL && !agg->significant_candidate_found)) {
        *quality_out = agg->significant_candidate_found ? DELAY_QUALITY_REFINED
                                                        : DELAY_QUALITY_COARSE;
        *delay_out = agg->pre_echo.pre_echo_candidate;  /* pre-echo active */
        return 1;
    }
    return 0;
}

/* ------------------------------------------------------------ clockdrift */

static void da_clockdrift_init(DaClockdrift *cd) {
    cd->delay_history[0] = cd->delay_history[1] = cd->delay_history[2] = 0;
    cd->level = CLOCKDRIFT_NONE;
    cd->stability_counter = 0;
}

static void da_clockdrift_update(DaClockdrift *cd, int delay_estimate) {
    int d1, d2, d3, probable_up, verified_up, probable_down, verified_down;
    if (delay_estimate == cd->delay_history[0]) {
        cd->stability_counter += 1;
        if (cd->stability_counter > DA_STABILITY_RESET_HOPS) cd->level = CLOCKDRIFT_NONE;
        return;
    }
    cd->stability_counter = 0;
    d1 = cd->delay_history[0] - delay_estimate;
    d2 = cd->delay_history[1] - delay_estimate;
    d3 = cd->delay_history[2] - delay_estimate;
    probable_up   = (d1 == -1 && d2 == -2) || (d1 == -2 && d2 == -1);
    verified_up   = probable_up && d3 == -3;
    probable_down = (d1 == 1 && d2 == 2) || (d1 == 2 && d2 == 1);
    verified_down = probable_down && d3 == 3;
    if (verified_up || verified_down) cd->level = CLOCKDRIFT_VERIFIED;
    else if ((probable_up || probable_down) && cd->level == CLOCKDRIFT_NONE)
        cd->level = CLOCKDRIFT_PROBABLE;
    cd->delay_history[2] = cd->delay_history[1];
    cd->delay_history[1] = cd->delay_history[0];
    cd->delay_history[0] = delay_estimate;
}

/* --------------------------------------------------- EchoPathDelayEstimator */

static void da_estimator_init(DaEstimator *e) {
    da_decimator_init(&e->capture_decimator);
    da_decimator_init(&e->render_decimator);
    da_ring_reset(&e->render_ring);
    da_matched_filter_init(&e->matched_filter);
    da_aggregator_init(&e->aggregator);
    da_clockdrift_init(&e->clockdrift);
    e->old_agg_valid = 0;
    e->old_agg_delay = 0;
    e->old_agg_quality = DELAY_QUALITY_COARSE;
    e->consistent_estimate_counter = 0;
    e->pending_count = 0;
}

/* _reset (reset_lag_aggregator, reset_delay_confidence). */
static void da_estimator_reset_internal(DaEstimator *e, int reset_lag_aggregator,
                                        int reset_delay_confidence) {
    if (reset_lag_aggregator)
        da_aggregator_reset(&e->aggregator, reset_delay_confidence);
    da_matched_filter_reset(&e->matched_filter, reset_lag_aggregator);
    e->old_agg_valid = 0;
    e->consistent_estimate_counter = 0;
}

/* public reset(reset_delay_confidence) -> full reset of lag aggregator. */
static void da_estimator_reset(DaEstimator *e, int reset_delay_confidence) {
    da_estimator_reset_internal(e, 1, reset_delay_confidence);
    /* decimators / ring / clockdrift / pending are NOT cleared by Python
     * EchoPathDelayEstimator.reset (it only touches aggregator + matched
     * filter + old_agg + counter), so leave them intact for parity. */
}

/* _process_inner_block. Returns 1 if an estimate was produced this block
 * (writes *quality / *delay in RAW samples). */
static int da_estimator_process_inner_block(DaEstimator *e, const float *render_block,
                                            const float *capture_block,
                                            DelayQuality *quality_out, int *delay_out) {
    float render_sub[DA_SUB_BLOCK_SIZE];
    float capture_sub[DA_SUB_BLOCK_SIZE];
    int produced;
    DelayQuality q = DELAY_QUALITY_COARSE;
    int delay_ds = 0;
    int agg_valid;
    int agg_delay = 0;
    DelayQuality agg_quality = DELAY_QUALITY_COARSE;

    da_decimator_decimate(&e->render_decimator, render_block, render_sub);
    da_decimator_decimate(&e->capture_decimator, capture_block, capture_sub);
    da_ring_push(&e->render_ring, render_sub, DA_SUB_BLOCK_SIZE);
    da_matched_filter_update(&e->matched_filter, &e->render_ring, capture_sub,
                             da_aggregator_reliable_delay_found(&e->aggregator));
    produced = da_aggregator_aggregate(
        &e->aggregator,
        e->matched_filter.reported_valid,
        e->matched_filter.reported_lag,
        e->matched_filter.reported_pre_echo_lag,
        &q, &delay_ds);

    if (produced && q == DELAY_QUALITY_REFINED)
        da_clockdrift_update(&e->clockdrift, da_aggregator_delay_at_highest_peak(&e->aggregator));

    agg_valid = produced;
    if (produced) {
        agg_quality = q;
        agg_delay = delay_ds * DA_DOWN_SAMPLING_FACTOR;
    }

    /* stability counter */
    if (e->old_agg_valid && agg_valid && e->old_agg_delay == agg_delay)
        e->consistent_estimate_counter += 1;
    else
        e->consistent_estimate_counter = 0;
    e->old_agg_valid = agg_valid;
    e->old_agg_delay = agg_delay;
    e->old_agg_quality = agg_quality;

    if (e->consistent_estimate_counter > DA_CONSISTENT_EST_THR)
        da_estimator_reset_internal(e, 0, 0);  /* partial reset */

    if (agg_valid) {
        *quality_out = agg_quality;
        *delay_out = agg_delay;
        return 1;
    }
    return 0;
}

/* estimate_delay: buffer raw hop into the edge-chunker, drain 64-sample
 * blocks.  Returns 1 if ANY estimate was produced during the drain (writes
 * the LATEST such estimate into *quality / *delay). */
static int da_estimator_estimate_delay(DaEstimator *e, const float *render_hop,
                                       const float *capture_hop, int hop,
                                       DelayQuality *quality_out, int *delay_out) {
    int produced_any = 0;
    int i;
    DelayQuality q = DELAY_QUALITY_COARSE;
    int delay = 0;
    for (i = 0; i < hop; ++i) {
        e->render_pending[e->pending_count] = render_hop[i];
        e->capture_pending[e->pending_count] = capture_hop[i];
        e->pending_count += 1;
        if (e->pending_count == DA_AEC3_BLOCK_SIZE) {
            DelayQuality bq;
            int bd;
            if (da_estimator_process_inner_block(e, e->render_pending, e->capture_pending,
                                                 &bq, &bd)) {
                produced_any = 1;
                q = bq;
                delay = bd;
            }
            e->pending_count = 0;
        }
    }
    if (produced_any) {
        *quality_out = q;
        *delay_out = delay;
        return 1;
    }
    return 0;
}

/* ====================== LegacyDelayShim (public API) ====================== */

void delay_aec3_init(DelayAec3 *d) {
    da_estimator_init(&d->est);
    d->latest_valid = 0;
    d->latest_delay = 0;
    d->latest_quality = DELAY_QUALITY_COARSE;
    d->estimate_count = 0;
}

void delay_aec3_reset(DelayAec3 *d) {
    da_estimator_reset(&d->est, 1);
    d->latest_valid = 0;
    d->estimate_count = 0;
}

int delay_aec3_accumulate(DelayAec3 *d, const float *near, const float *far, int hop) {
    DelayQuality q = DELAY_QUALITY_COARSE;
    int delay = 0;
    int produced;
    int prev_delay = d->latest_valid ? d->latest_delay : -1;
    int emitted_new = 0;
    /* NOTE: estimate_delay takes (render, capture) = (far, near). */
    produced = da_estimator_estimate_delay(&d->est, far, near, hop, &q, &delay);
    if (produced) {
        d->latest_valid = 1;
        d->latest_delay = delay;
        d->latest_quality = q;
        d->estimate_count += 1;
        if (delay != prev_delay) emitted_new = 1;
    }
    return emitted_new;
}

int delay_aec3_estimated_delay(const DelayAec3 *d) {
    if (!d->latest_valid) return -1;
    return d->latest_delay;
}

double delay_aec3_confidence(const DelayAec3 *d) {
    if (!d->latest_valid) return 0.0;
    if (d->latest_quality == DELAY_QUALITY_REFINED) return 1.0;
    if (d->latest_quality == DELAY_QUALITY_COARSE) return 0.5;
    return 0.0;
}

int delay_aec3_is_solid(const DelayAec3 *d) {
    return delay_aec3_confidence(d) >= 1.0;
}

int delay_aec3_n_updates(const DelayAec3 *d) {
    return d->estimate_count;
}

int delay_aec3_has_clockdrift(const DelayAec3 *d) {
    return d->est.clockdrift.level != CLOCKDRIFT_NONE;
}
