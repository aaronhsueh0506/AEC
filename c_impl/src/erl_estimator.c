/* erl_estimator.c — C port of python/modules/state/erl_estimator.py.
 * See the header for the full parity contract. Build with -ffp-contract=off.
 *
 * Dtype map (float32; drift versus Python accepted):
 *   - Per-bin arrays _erl / inputs are float32.
 *   - new_erl = y2[k]/x2[k] is a float32 scalar divide.
 *   - delta = 0.1f*(new_erl - erl[k]) is float32; erl[k] += delta stays f32
 *     end-to-end (single rounding, no widen-then-narrow round trip).
 *   - max(erl[k], 0.01f) / min(1000.0f, 2.0f*erl[k]) compare directly in
 *     float32.
 *   - Fullband path is all float32; the two np.sum()-equivalent calls are
 *     pairwise f32 sums (sk_pairwise_sum_tailfold_b_f32), consumed directly
 *     as float32.
 */
#include "erl_estimator.h"

#include "aec_scale.h"   /* aec_per_bin_psd_threshold */
#include "utility/simd_aec.h"   /* sk_erl_bin_update_f32/sk_dec1_floorintmin_s32/sk_erl_hold_expire_f32 */

#include <limits.h>   /* INT_MIN */
#include <string.h>

void erl_estimator_init(ErlEstimator *e, int startup_phase_length_hops,
                        int n_bins, int hop_size, int sample_rate,
                        float *erl_storage, int *hold_counters_storage) {
    int k;
    e->startup_hops = startup_phase_length_hops;
    e->n_bins = n_bins;
    /* 1000 blocks (~4 s) hold-after-drop, computed live from hop and
     * sample rate. */
    e->hold_hops = aec_ms_to_hops(4000.0f, hop_size, sample_rate);
    /* self._x2_min = per_bin_psd_threshold(<reference X2 minimum>, hop_size)
     * (ref_hop default 160; no-op at hop=160); float32 end-to-end. */
    e->x2_min = aec_per_bin_psd_threshold(ERL_X2_MIN, hop_size, 160);
    e->x2_min_total = e->x2_min * (float)n_bins;
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

    /* Only the startup comparison reads this counter; saturation preserves it
     * and prevents signed overflow during long-running streams. */
    if (e->blocks_since_reset < e->startup_hops) e->blocks_since_reset += 1;
    if (e->blocks_since_reset < e->startup_hops || !converged_filter) {
        return;
    }

    /* Skip DC/Nyquist: erl/x2/y2 start at k=1; hold[j] already maps to
     * k-1. See utility/simd_aec.h for the masked-update contract. */
    sk_erl_bin_update_f32(e->erl + 1, e->hold_counters, x2 + 1, y2 + 1,
                          e->x2_min, e->hold_hops, ERL_MIN_ERL,
                          e->n_bins - 2);

    /* Decrement every hold counter after the masked update, before expiry.
     * Saturate at INT_MIN to avoid C signed overflow while retaining negative
     * counter state; flooring at zero would change observable raw values. */
    sk_dec1_floorintmin_s32(e->hold_counters, e->n_bins - 2);

    /* Bins with hold counter <= 0 double toward _MAX_ERL, kernel 26. Same
     * erl/hold_counters offset convention as kernel 24 above. */
    sk_erl_hold_expire_f32(e->erl + 1, e->hold_counters, ERL_MAX_ERL,
                           e->n_bins - 2);

    /* Mirror endpoints. */
    e->erl[0] = e->erl[1];
    e->erl[e->n_bins - 1] = e->erl[e->n_bins - 2];

    /* Fullband ERL. x2_sum = pairwise f32 sum, consumed directly as float32. */
    {
        float x2_sum = sk_pairwise_sum_tailfold_b_f32(x2, (size_t)e->n_bins);
        if (x2_sum > e->x2_min_total) {
            float y2_sum = sk_pairwise_sum_tailfold_b_f32(y2, (size_t)e->n_bins);
            float new_erl = y2_sum / x2_sum;          /* f32 */
            if (new_erl < e->erl_time_domain) {
                e->hold_counter_time_domain = e->hold_hops;
                e->erl_time_domain += 0.1f * (new_erl - e->erl_time_domain);
                if (e->erl_time_domain < ERL_MIN_ERL) {
                    e->erl_time_domain = ERL_MIN_ERL;
                }
            }
        }
    }
    /* Signed-overflow guard (same shape as hold_counters' kernel-25 floor above
     * and blocks_since_reset's ceiling at the top of this function), but floored
     * at INT_MIN rather than at 0 -- deliberately NOT the "floor at the gate's
     * own threshold" shape used for those two: the production consumer of this
     * field is the `<= 0` check immediately below (a level check -- floor-at-0
     * would leave that check's boolean outcome identical to an unbounded
     * decrement for every hop, same argument as hold_counters), BUT this field's
     * raw integer value is ALSO observable (test_counter_saturation.c reads it)
     * and must track python/modules/state/erl_estimator.py's own
     * `_hold_counter_time_domain -= 1` (an ordinary Python int -- no floor, no
     * wraparound). Floor-at-0 pins the value the hop after it first goes
     * non-positive, while Python's keeps counting down (-1, -2, -3, ...), so the
     * raw values would diverge. Flooring at INT_MIN instead reproduces the
     * unbounded decrement's exact trajectory for every reachable state (at 10 ms
     * hops INT_MIN is ~248 days of continuous uptime away). So: decrement only
     * while strictly greater than INT_MIN, which removes the UB at the true
     * overflow boundary while staying bit-exact-identical to an unconditional
     * `-= 1` (and to Python's unbounded int) for every practically reachable
     * state. */
    if (e->hold_counter_time_domain > INT_MIN) e->hold_counter_time_domain -= 1;
    if (e->hold_counter_time_domain <= 0) {
        float doubled = 2.0f * e->erl_time_domain;
        e->erl_time_domain = doubled < ERL_MAX_ERL ? doubled : ERL_MAX_ERL;
    }
}

