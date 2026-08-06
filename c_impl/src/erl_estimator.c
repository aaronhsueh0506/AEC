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
#include "aec_simd_kernels.h"   /* sk_erl_bin_update_f32/sk_dec1_floorintmin_s32/sk_erl_hold_expire_f32 */

#include <limits.h>   /* INT_MIN */
#include <string.h>

void erl_estimator_init(ErlEstimator *e, int startup_phase_length_hops,
                        int n_bins, int hop_size, int sample_rate,
                        float *erl_storage, int *hold_counters_storage) {
    int k;
    e->startup_hops = startup_phase_length_hops;
    e->n_bins = n_bins;
    /* AEC3 1000 blocks (~4 s) hold-after-drop. Was a frozen #define (400,
     * correct only at hop=160/sr=16000); computed live here. */
    e->hold_hops = aec3_ms_to_hops(4000.0f, hop_size, sample_rate);
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

    /* Ceilinged at startup_hops (UBSan-confirmed signed-overflow fix): the
     * ONLY consumer of blocks_since_reset anywhere in this file/header is the
     * `< startup_hops` comparison right below. Once the counter reaches
     * startup_hops, further increments can never change that comparison's
     * outcome (it's already false and stays false), so gating the increment
     * on `< startup_hops` is observationally identical to the old
     * unconditional `+= 1` for every reachable state, while eliminating the
     * eventual overflow (at 10 ms hops, unconditional would wrap ~248 days
     * into continuous uptime). */
    if (e->blocks_since_reset < e->startup_hops) e->blocks_since_reset += 1;
    if (e->blocks_since_reset < e->startup_hops || !converged_filter) {
        return;
    }

    /* Per-bin minimum-statistics update for k=1..n_bins-2, fused into one
     * masked NEON pass (kernel 24): erl/x2/y2 advanced by +1 so lane j lines
     * up with source index k=j+1; hold_counters passed unshifted since
     * hold_counters[j] already IS hold_counters[k-1] once erl/x2/y2 are
     * pre-offset (k-1 = (j+1)-1 = j). See aec_simd_kernels.h kernel 24's
     * header comment for the fusion-safety/masking argument. */
    sk_erl_bin_update_f32(e->erl + 1, e->hold_counters, x2 + 1, y2 + 1,
                          e->x2_min, e->hold_hops, ERL_MIN_ERL,
                          e->n_bins - 2);

    /* self._hold_counters -= 1, floored at INT_MIN (all n_bins-2 elements),
     * kernel 25. Must run strictly after the loop above (reads/writes ALL of
     * hold_counters, including bins the masked update above did not touch
     * this hop) and strictly before the loop below (which reads the
     * just-decremented values) -- three separate sequential kernel calls,
     * not fused. Floored at INT_MIN, NOT at 0 (UBSan-confirmed
     * signed-overflow fix -- same bug class/fix shape as this function's
     * hold_counter_time_domain below and fullband_erle.c's
     * hold_counter_inst_erle): floor-at-0 would satisfy kernel 26's `<= 0`
     * check just below identically to the old unbounded-negative decrement,
     * but hold_counters' raw values are ALSO read bit-exact by
     * test/historical/parity_erl_estimator.c's golden, which floor-at-0 desyncs from
     * almost immediately (confirmed: this field alone accounts for 2142 of
     * the golden's 3043 total broken-before-fix mismatches, on top of a
     * 901-mismatch true baseline unrelated to this bug -- NOT "baseline of
     * 3043, +5 more"; see kernel 25's header comment in aec_simd_kernels.h
     * for the full 901/2142/3043 breakdown and the numpy-wraps-vs-C-UB-vs-
     * C-saturates distinction) -- see that same header comment for the
     * flooring's NEON-wraparound safety at the INT_MIN boundary. */
    sk_dec1_floorintmin_s32(e->hold_counters, e->n_bins - 2);

    /* Bins with hold counter <= 0 double toward _MAX_ERL, kernel 26. Same
     * erl/hold_counters offset convention as kernel 24 above. */
    sk_erl_hold_expire_f32(e->erl + 1, e->hold_counters, ERL_MAX_ERL,
                           e->n_bins - 2);

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
                e->hold_counter_time_domain = e->hold_hops;
                e->erl_time_domain += 0.1f * (new_erl - e->erl_time_domain);
                if (e->erl_time_domain < ERL_MIN_ERL) {
                    e->erl_time_domain = ERL_MIN_ERL;
                }
            }
        }
    }
    /* UBSan-confirmed signed-overflow fix (same bug class as hold_counters'
     * kernel-25 fix above and blocks_since_reset's fix at the top of this
     * function), but floored at INT_MIN rather than at 0 -- deliberately
     * NOT the same "floor at the gate's own threshold" shape used for those
     * two: the production-only consumer of this field is the `<= 0` check
     * immediately below (a level check -- floor-at-0 would leave that
     * check's boolean outcome identical to the unbounded decrement for
     * every hop, same argument as hold_counters), BUT this field's raw
     * integer value is ALSO read directly (not just the boolean) by
     * test/historical/parity_erl_estimator.c's bit-exact golden comparison, which
     * mirrors python/modules/state/erl_estimator.py's own
     * `_hold_counter_time_domain -= 1` (an ordinary Python int -- no floor,
     * no wraparound). Floor-at-0 pins the value the hop after it first
     * goes non-positive, while Python's/the old C's kept counting down
     * (-1, -2, -3, ...); empirically (isolated golden replay,
     * 1100-frame regression fixture) that mismatch is real: floor-at-0
     * adds 5 new int mismatches beyond the module's already-accepted
     * float drift, while flooring at INT_MIN instead reproduces the
     * unbounded decrement's exact trajectory for every frame the fixture
     * (or any realistic run -- this is 10 ms hops, so INT_MIN is ~248 days
     * of continuous uptime away) ever reaches -- zero incremental
     * mismatches, confirmed by rerunning the same golden. So: decrement
     * only while strictly greater than INT_MIN, which eliminates the UB at
     * the true overflow boundary while being bit-exact-identical to the
     * old unconditional `-= 1` (and to Python's unbounded int) for every
     * practically-reachable state, including the ones this repo's own
     * regression golden happens to exercise. */
    if (e->hold_counter_time_domain > INT_MIN) e->hold_counter_time_domain -= 1;
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
