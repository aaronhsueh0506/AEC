/* erl_estimator.h — C port of python/modules/state/erl_estimator.py
 * (which mirrors AEC3 erl_estimator.{cc,h}).
 *
 * Per-bin echo return loss (ERL = Y²/X²) with minimum-statistics tracking:
 *   - per-bin _erl[k] for k=1..n_bins-2 (endpoints mirror neighbours)
 *   - a fullband _erl_time_domain scalar.
 * On each downward jump the estimate is nudged 10% toward the new ratio and a
 * hold counter is armed for _HOLD_HOPS (=400) hops; bins whose hold counter has
 * expired double (slow recovery) toward _MAX_ERL each update.
 *
 * PRECISION: float32-by-design (f32 campaign — Python bit-exact parity is
 * retired for this module; drift accepted). Real-pipeline shapes (single DT
 * case, preset=balanced, hop=160):
 *   render_psd / capture_psd : float32 (257,)        — inputs X², Y²
 *   self._erl                : float32 (257,)        — per-bin estimate
 *   self._hold_counters      : int32   (255,)        — sized n_bins-2
 *   self._x2_min             : float32 scalar (was 44015068.0 f64)
 *   self._erl_time_domain    : float32 scalar (was f64)
 *
 *   Per-bin update, all float32, single rounding per op (no more widen-to-
 *   double-then-narrow):
 *     new_erl   = y2[k] / x2[k]                  -> float32
 *     delta     = 0.1f * (new_erl - erl[k])      -> float32
 *     erl[k]    = erl[k] + delta                 -> float32
 *     erl[k]    = max(erl[k], 0.01f)             -> float32 compare
 *     erl[k]    = min(1000.0f, 2.0f*erl[k])      -> float32 compare (doubling)
 *   The old "must promote to double because numpy's max()/min() builtins
 *   compare in double" concern no longer applies — this is native C float32
 *   arithmetic, so the compares run directly in float.
 *
 *   Fullband: x2_sum/y2_sum are numpy-pairwise float32 sums
 *   (f32_pairwise_sum); the ratio/EMA/doubling that follow are float32
 *   end-to-end (previously widened to float64 to mirror Python's
 *   `float(np.sum(...))`; that widening is no longer needed or done).
 *
 * Built with -ffp-contract=off so the per-op roundings are not fused.
 */
#ifndef ERL_ESTIMATOR_H
#define ERL_ESTIMATOR_H

#include <stddef.h>

/* ── Constants (erl_estimator.py module level) ───────────────────────────── */
#define ERL_MIN_ERL  0.01f    /* _MIN_ERL */
#define ERL_MAX_ERL  1000.0f  /* _MAX_ERL */
/* _AEC3_X2_MIN source constant — deliberately left as a double literal (NOT
 * `f`-suffixed): its only use is as the `calibrated_value` argument to the
 * shared aec3_per_bin_psd_threshold() helper (aec3_scale.h/.c), which is a
 * double-typed API outside this conversion's scope. The double RESULT of
 * that call is narrowed to float32 exactly once, at the `x2_min` store in
 * erl_estimator_init(). _HOLD_HOPS = int(4.0 * HOPS_PER_SECOND) = 400. */
#define ERL_AEC3_X2_MIN  44015068.0f
#define ERL_HOLD_HOPS    400

/* numpy 1.26 pairwise float32 sum (declared in reverb_frequency_response.c).
 * Reused here for float(np.sum(x2)) / float(np.sum(y2)). */
float f32_pairwise_sum(const float *a, size_t n);

/* ── ErlEstimator ─────────────────────────────────────────────────────────
 * Caller provides backing storage:
 *   erl_storage          : n_bins              (float)
 *   hold_counters_storage: n_bins - 2          (int32_t) */
typedef struct {
    int     startup_hops;          /* _startup_hops               */
    int     n_bins;                /* _n_bins                     */
    float   x2_min;                /* _x2_min (float32; narrowed once from
                                     * the double threshold helper's result) */
    float  *erl;                   /* _erl,           length n_bins     */
    int    *hold_counters;         /* _hold_counters, length n_bins-2   */
    float   erl_time_domain;       /* _erl_time_domain            */
    int     hold_counter_time_domain;
    int     blocks_since_reset;    /* _blocks_since_reset         */
} ErlEstimator;

/* hop_size feeds aec3_per_bin_psd_threshold(44015068.0, hop_size) for x2_min. */
void erl_estimator_init(ErlEstimator *e, int startup_phase_length_hops,
                        int n_bins, int hop_size,
                        float *erl_storage, int *hold_counters_storage);

void erl_estimator_reset(ErlEstimator *e);

/* update(render_psd=X², capture_psd=Y², converged_filter) */
void erl_estimator_update(ErlEstimator *e,
                          const float *render_psd,
                          const float *capture_psd,
                          int converged_filter);

/* Accessors. erl() returns the internal _erl pointer (length n_bins). */
const float *erl_estimator_erl(const ErlEstimator *e);
float erl_estimator_erl_time_domain(const ErlEstimator *e);

#endif /* ERL_ESTIMATOR_H */
