/* erl_estimator.c — C port of python/modules/state/erl_estimator.py.
 * See the header for the full parity contract. Build with -ffp-contract=off.
 *
 * Dtype map (float32-by-design; f32 campaign, Python bit-exact parity
 * retired for this module — drift accepted):
 *   - Per-bin arrays _erl / inputs are float32.
 *   - new_erl = y2[k]/x2[k] is a float32 scalar divide.
 *   - delta = 0.1f*(new_erl - erl[k]) is float32; erl[k] += delta stays f32
 *     end-to-end (single rounding, no widen-then-narrow round trip).
 *   - max(erl[k], 0.01f) / min(1000.0f, 2.0f*erl[k]) compare directly in
 *     float32 (the old "numpy builtin compares promote to double" concern
 *     doesn't apply to native C float arithmetic).
 *   - Fullband path is all float32; the two np.sum()-equivalent calls are
 *     pairwise f32 sums (f32_pairwise_sum), consumed directly as float32.
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
     * (ref_hop default 160; no-op at hop=160); float32 end-to-end. */
    e->x2_min = aec3_per_bin_psd_threshold(ERL_AEC3_X2_MIN, hop_size, 160);
    e->erl = erl_storage;
    e->hold_counters = hold_counters_storage;
    /* self._erl = np.full(n_bins, _MAX_ERL, dtype=f32) */
    for (k = 0; k < n_bins; ++k) {
        e->erl[k] = ERL_MAX_ERL;
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
        /* x2[k] > self._x2_min : both float32 now, compare directly. */
        if (x2[k] > e->x2_min) {
            float new_erl = y2[k] / x2[k];          /* float32 divide */
            if (new_erl < e->erl[k]) {              /* f32 < f32 */
                e->hold_counters[k - 1] = ERL_HOLD_HOPS;
                /* delta = 0.1f * (new_erl - erl[k]); float32 end-to-end. */
                float diff_f = new_erl - e->erl[k];   /* float32 sub */
                float delta = 0.1f * diff_f;          /* float32 mul */
                e->erl[k] = e->erl[k] + delta;        /* float32 add */
                /* erl[k] = max(erl[k], _MIN_ERL), compared in float32. */
                if (e->erl[k] < ERL_MIN_ERL) {
                    e->erl[k] = ERL_MIN_ERL;
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
            /* erl[k] = min(_MAX_ERL, 2.0f * erl[k]), all float32. */
            float doubled = 2.0f * e->erl[k];
            if (doubled < ERL_MAX_ERL) {
                e->erl[k] = doubled;
            } else {
                e->erl[k] = ERL_MAX_ERL;
            }
        }
    }

    /* Mirror endpoints. */
    e->erl[0] = e->erl[1];
    e->erl[e->n_bins - 1] = e->erl[e->n_bins - 2];

    /* Fullband ERL. x2_sum = pairwise f32 sum, consumed directly as float32
     * (no widen to double). */
    {
        float x2_sum = f32_pairwise_sum(x2, (size_t)e->n_bins);
        if (x2_sum > e->x2_min * (float)e->n_bins) {
            float y2_sum = f32_pairwise_sum(y2, (size_t)e->n_bins);
            float new_erl = y2_sum / x2_sum;          /* f32 */
            if (new_erl < e->erl_time_domain) {
                e->hold_counter_time_domain = ERL_HOLD_HOPS;
                e->erl_time_domain += 0.1f * (new_erl - e->erl_time_domain);
                if (e->erl_time_domain < ERL_MIN_ERL) {
                    e->erl_time_domain = ERL_MIN_ERL;
                }
            }
        }
    }
    e->hold_counter_time_domain -= 1;
    if (e->hold_counter_time_domain <= 0) {
        float doubled = 2.0f * e->erl_time_domain;
        e->erl_time_domain = doubled < ERL_MAX_ERL ? doubled : ERL_MAX_ERL;
    }
}

const float *erl_estimator_erl(const ErlEstimator *e) {
    return e->erl;
}

float erl_estimator_erl_time_domain(const ErlEstimator *e) {
    return e->erl_time_domain;
}
