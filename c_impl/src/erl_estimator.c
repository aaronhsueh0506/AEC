/* erl_estimator.c — C port of python/modules/state/erl_estimator.py.
 * See the header for the full parity contract. Build with -ffp-contract=off.
 *
 * Dtype map (all VERIFIED against numpy<2 scalar weak promotion this cycle):
 *   - Per-bin arrays _erl / inputs are float32.
 *   - new_erl = y2[k]/x2[k] is a float32 scalar divide.
 *   - delta = 0.1*(new_erl - erl[k]) is FLOAT64 (py-float scalar × f32 → f64),
 *     so erl[k] += delta is an f64 add stored back to f32.
 *   - max(erl[k], 0.01) / min(1000.0, 2.0*erl[k]) are Python builtins: the
 *     compare promotes the f32 element to double, so we branch in double.
 *   - Fullband path is all f64; the two np.sum() calls are pairwise f32 sums
 *     widened to f64 by float().
 */
#include "erl_estimator.h"

#include "aec3_scale.h"   /* aec3_per_bin_psd_threshold */

#include <string.h>

void erl_estimator_init(ErlEstimator *e, int startup_phase_length_hops,
                        int n_bins, int hop_size,
                        float *erl_storage, int *hold_counters_storage) {
    int k;
    e->startup_hops = startup_phase_length_hops;
    e->n_bins = n_bins;
    /* self._x2_min = aec3_scale.per_bin_psd_threshold(_AEC3_X2_MIN, hop_size)
     * (ref_hop default 160; no-op at hop=160). */
    e->x2_min = aec3_per_bin_psd_threshold(ERL_AEC3_X2_MIN, hop_size, 160);
    e->erl = erl_storage;
    e->hold_counters = hold_counters_storage;
    /* self._erl = np.full(n_bins, _MAX_ERL, dtype=f32) */
    for (k = 0; k < n_bins; ++k) {
        e->erl[k] = (float)ERL_MAX_ERL;
    }
    /* self._hold_counters = np.zeros(n_bins-2, int32) */
    for (k = 0; k < n_bins - 2; ++k) {
        e->hold_counters[k] = 0;
    }
    e->erl_time_domain = ERL_MAX_ERL;       /* Python float */
    e->hold_counter_time_domain = 0;
    e->blocks_since_reset = 0;
}

void erl_estimator_reset(ErlEstimator *e) {
    /* Python reset() only zeroes blocks_since_reset. */
    e->blocks_since_reset = 0;
}

void erl_estimator_update(ErlEstimator *e,
                          const float *render_psd,
                          const float *capture_psd,
                          int converged_filter) {
    const float *x2 = render_psd;
    const float *y2 = capture_psd;
    int k;

    e->blocks_since_reset += 1;
    if (e->blocks_since_reset < e->startup_hops || !converged_filter) {
        return;
    }

    /* Per-bin minimum-statistics update for k=1..n_bins-2. */
    for (k = 1; k < e->n_bins - 1; ++k) {
        /* x2[k] > self._x2_min : numpy promotes the f32 element to f64 to
         * compare against the python-float threshold, so compare in double. */
        if ((double)x2[k] > e->x2_min) {
            float new_erl = y2[k] / x2[k];          /* float32 divide */
            if (new_erl < e->erl[k]) {              /* f32 < f32 */
                e->hold_counters[k - 1] = ERL_HOLD_HOPS;
                /* delta = 0.1 * (new_erl - erl[k]).
                 * (new_erl - erl[k]) is an f32 scalar subtraction (both
                 * operands float32); only the 0.1× widens to float64. */
                float  diff_f = new_erl - e->erl[k];          /* float32 sub */
                double delta = 0.1 * (double)diff_f;          /* → float64   */
                /* erl[k] += delta : f64 add, store to f32 */
                e->erl[k] = (float)((double)e->erl[k] + delta);
                /* erl[k] = max(erl[k], _MIN_ERL): keep erl[k] unless
                 * (double)erl[k] < 0.01, then store f32(0.01). */
                if ((double)e->erl[k] < ERL_MIN_ERL) {
                    e->erl[k] = (float)ERL_MIN_ERL;
                }
            }
        }
    }

    /* self._hold_counters -= 1  (all n_bins-2 elements). */
    for (k = 0; k < e->n_bins - 2; ++k) {
        e->hold_counters[k] -= 1;
    }

    /* Bins with hold counter <= 0 double toward _MAX_ERL. */
    for (k = 1; k < e->n_bins - 1; ++k) {
        if (e->hold_counters[k - 1] <= 0) {
            /* erl[k] = min(_MAX_ERL, 2.0 * erl[k]) : 2.0*erl[k] in f64,
             * keep that unless it reaches 1000.0, store to f32. */
            double doubled = 2.0 * (double)e->erl[k];
            if (doubled < ERL_MAX_ERL) {
                e->erl[k] = (float)doubled;
            } else {
                e->erl[k] = (float)ERL_MAX_ERL;
            }
        }
    }

    /* Mirror endpoints. */
    e->erl[0] = e->erl[1];
    e->erl[e->n_bins - 1] = e->erl[e->n_bins - 2];

    /* Fullband ERL. x2_sum = float(np.sum(x2)) — pairwise f32 sum → f64. */
    {
        double x2_sum = (double)f32_pairwise_sum(x2, (size_t)e->n_bins);
        if (x2_sum > e->x2_min * (double)e->n_bins) {
            double y2_sum = (double)f32_pairwise_sum(y2, (size_t)e->n_bins);
            double new_erl = y2_sum / x2_sum;          /* f64 */
            if (new_erl < e->erl_time_domain) {
                e->hold_counter_time_domain = ERL_HOLD_HOPS;
                e->erl_time_domain += 0.1 * (new_erl - e->erl_time_domain);
                if (e->erl_time_domain < ERL_MIN_ERL) {
                    e->erl_time_domain = ERL_MIN_ERL;
                }
            }
        }
    }
    e->hold_counter_time_domain -= 1;
    if (e->hold_counter_time_domain <= 0) {
        double doubled = 2.0 * e->erl_time_domain;
        e->erl_time_domain = doubled < ERL_MAX_ERL ? doubled : ERL_MAX_ERL;
    }
}

const float *erl_estimator_erl(const ErlEstimator *e) {
    return e->erl;
}

double erl_estimator_erl_time_domain(const ErlEstimator *e) {
    return e->erl_time_domain;
}
