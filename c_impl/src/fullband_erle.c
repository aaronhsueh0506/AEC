/* fullband_erle.c — C port of python/modules/state/fullband_erle.py.
 *
 * PRECISION: float32-by-design (f32 campaign — Python bit-exact parity
 * retired, drift accepted). The per-bin inputs x2/y2/e2 are float32 arrays;
 * the pairwise sum over them stays float32 (fb_erle_pairwise_sum), no longer
 * widened to double afterward. Every scalar accumulator / ratio / log2 / EMA
 * / max-min / quality op is float32 end-to-end, matching the arrays'
 * precision (the _fast_log2 helper now wraps log2f). Built with
 * -ffp-contract=off so each float op rounds separately.
 */
#include "fullband_erle.h"

#include "aec3_scale.h"   /* aec3_per_bin_psd_threshold */
#include "aec_simd_kernels.h"

#include <limits.h>   /* INT_MIN */
#include <math.h>

/* numpy 1.26 pairwise_sum over float32 (verbatim copy of the verified routine
 * in reverb_frequency_response.c):
 *   n < 8     -> sequential
 *   n <= 128  -> 8 partial accumulators (stride 8), balanced-tree combine, then
 *                the (n % 8) tail sequentially
 *   n > 128   -> split at n2 = (n/2) rounded DOWN to a multiple of 8, recurse,
 *                add the two halves.
 * Every op is a float32 add (no float64 widening).
 * Body delegates to simd_kernels.h's sk_pairwise_sum_tailfold_b_f32 (this
 * tree is byte-identical, modulo whitespace, to filter_analyzer.c's
 * fa_f32_pairwise_sum — that kernel's `_scalar` twin's verbatim source);
 * local symbol/signature kept so call sites are unchanged. */
float fb_erle_pairwise_sum(const float *a, size_t n) {
    return sk_pairwise_sum_tailfold_b_f32(a, n);
}

/* _fast_log2(x) = log2f(x) — float32 in, float32 out (was float(np.log2(x))
 * i.e. a float64 log2; now a single-precision log2 to match the rest of the
 * float32 pipeline). */
static float fast_log2(float x) {
    return log2f(x);
}

/* ── _ErleInstantaneous ─────────────────────────────────────────────────────── */

static void inst_init(ErleInstantaneous *e, float quality_alpha,
                       float maxmin_forget) {
    e->clamp_zero = 1;            /* clamp_quality_to_zero default True */
    e->clamp_one = 1;            /* clamp_quality_to_one  default True */
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
    e->erle_log2 = 0.0f;         /* mirror _erle_log2 = None (value unused) */
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
        e->max_erle_log2 = e->erle_log2;   /* max(_max, _erle) */
    }
    e->min_erle_log2 += e->maxmin_forget;
    if (e->erle_log2 < e->min_erle_log2) {
        e->min_erle_log2 = e->erle_log2;   /* min(_min, _erle) */
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

/* ErleInstantaneous::update -> returns whether estimates updated. */
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

/* get_quality_estimate() — *valid = 0 mirrors a None return (erle_log2 None). */
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

/* ── FullBandErleEstimator ──────────────────────────────────────────────────── */

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
    /* AEC3 fullband EMAs (quality 0.07, erle_time_domain 0.05) are
     * per-4ms-block, and fire once per FBERLE_POINTS_TO_ACCUMULATE(=6)-hop
     * accumulation window (see fb_erle_update()), not every hop. Both our
     * accumulation window (6 hops) and AEC3's reference update period (6
     * native 4ms blocks = 24ms) carry the same factor of 6, so it cancels:
     * pass plain hop_size against aec3_per_block_ema_alpha_to_per_hop's
     * fixed single-native-block (4ms) denominator, NOT
     * FBERLE_POINTS_TO_ACCUMULATE*hop_size (an earlier fix on 2026-08-01
     * passed the latter, which compares 6 real hops against only 1 native
     * block, inflating the exponent 6x too far; caught in review). */
    s->td_alpha = aec3_per_block_ema_alpha_to_per_hop(
        0.05f, hop_size, sample_rate);
    s->quality_alpha = aec3_per_block_ema_alpha_to_per_hop(
        0.07f, hop_size, sample_rate);
    /* FBERLE_MAXMIN_FORGET is AEC3's own literal per-event forget *step*
     * (additive, not a retention alpha) on the same 6-point-gated cadence
     * as quality_alpha/td_alpha above -- the wall-clock-preserving rescale
     * is the linear aec3_per_block_rate_to_per_hop (same hop_size-cancels-
     * the-6x reasoning), not the EMA formula. */
    float maxmin_forget = aec3_per_block_rate_to_per_hop(
        FBERLE_MAXMIN_FORGET, hop_size, sample_rate);
    /* AEC3 kBlocksToHoldErle=100 blocks (~400 ms). Was a frozen #define (40
     * hops, correct only at hop=160/sr=16000); rescaled live here. */
    s->blocks_to_hold_erle = aec3_ms_to_hops(400.0f, hop_size, sample_rate);
    /* The helper (aec3_scale.h/.c) is float32-typed and out of scope for this
     * conversion; its result is stored directly here. */
    s->x2_band_energy_threshold =
        aec3_per_bin_psd_threshold(FBERLE_X2_BAND_ENERGY_THRESHOLD, hop_size, 160);
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
        /* x2_sum = pairwise f32 sum, consumed directly as float32. */
        float x2_sum = fb_erle_pairwise_sum(x2, (size_t)s->n_freqs);
        if (x2_sum > s->x2_band_energy_threshold * (float)s->n_freqs) {
            float y2_sum = fb_erle_pairwise_sum(y2, (size_t)s->n_freqs);
            float e2_sum = fb_erle_pairwise_sum(e2, (size_t)s->n_freqs);
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
    /* UBSan-confirmed signed-overflow fix, floored at INT_MIN -- NOT at 0
     * (⚠ do not "fix" this the way erle_estimator.c/erl_estimator.c's
     * LEVEL-check counters were fixed: the gate below is `== 0`, an
     * EDGE-triggered one-shot check, not a level check. The old unbounded
     * decrement crosses exactly 0 once per re-arm cycle then keeps
     * counting down (-1, -2, ...), so `== 0` fires exactly once per cycle.
     * Flooring at 0 would PIN the value at 0 once reached, making `== 0`
     * fire on every subsequent hop too -- inst_reset_accumulators() would
     * re-fire every hop instead of once, discarding any partial
     * accumulation inst_update() makes on those hops and changing
     * production behaviour (this field's raw value is also read bit-exact
     * by test/historical/parity_fullband_erle.c: `s.hold_counter_inst_erle !=
     * exp_hold`, which mirrors the module's Python reference's own
     * unbounded `_hold_counter_inst_erle`). Flooring at INT_MIN instead
     * preserves BOTH the one-shot edge-trigger semantics and the bit-exact
     * golden for every practically-reachable hop count (10 ms hops ->
     * INT_MIN is ~248 days of continuous uptime away), while eliminating
     * the UB at the true overflow boundary. */
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
