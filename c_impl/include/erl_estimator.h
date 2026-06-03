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
 * PARITY (numpy 1.26 → C). Real-pipeline captured dtypes (single DT case,
 * preset=balanced, hop=160):
 *   render_psd / capture_psd : float32 (257,)        — inputs X², Y²
 *   self._erl                : float32 (257,)        — per-bin estimate
 *   self._hold_counters      : int32   (255,)        — sized n_bins-2
 *   self._x2_min             : Python float 44015068.0 (f64)
 *   self._erl_time_domain    : Python float          (f64 scalar)
 *
 *   Per-bin update dtype flow (VERIFIED against numpy<2 scalar weak promotion):
 *     new_erl   = y2[k] / x2[k]                  -> float32 (f32 scalar / f32)
 *     delta     = 0.1 * (new_erl - erl[k])       -> float64 (py-float × f32 = f64)
 *     erl[k]    = (float)((double)erl[k] + delta)            (f64 add, f32 store)
 *     erl[k]    = max(erl[k], 0.01)              -> (double)erl[k] compared
 *     erl[k]    = (float)min(1000.0, 2.0*(double)erl[k])     (doubling branch)
 *   max()/min() here are Python builtins: max(a,b) returns a unless b>a;
 *   the compare promotes the f32 element to double — so we compare in double,
 *   NOT fmaxf/fminf (which compare in float).
 *
 *   Fullband: x2_sum = float(np.sum(x2)) where np.sum over a float32 array is
 *   numpy PAIRWISE summation in float32 (NOT naive left-to-right), then widened
 *   to f64 by float(). Everything else in the fullband path is f64.
 *
 * Built with -ffp-contract=off so the per-op roundings are not fused.
 */
#ifndef ERL_ESTIMATOR_H
#define ERL_ESTIMATOR_H

#include <stddef.h>

/* ── Constants (erl_estimator.py module level) ───────────────────────────── */
#define ERL_MIN_ERL  0.01    /* _MIN_ERL */
#define ERL_MAX_ERL  1000.0  /* _MAX_ERL */
/* _AEC3_X2_MIN = 44015068.0, scaled per hop via aec3_per_bin_psd_threshold
 * (no-op at hop=160 ref). _HOLD_HOPS = int(4.0 * HOPS_PER_SECOND) = 400. */
#define ERL_AEC3_X2_MIN  44015068.0
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
    double  x2_min;                /* _x2_min  (Python float)     */
    float  *erl;                   /* _erl,           length n_bins     */
    int    *hold_counters;         /* _hold_counters, length n_bins-2   */
    double  erl_time_domain;       /* _erl_time_domain            */
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
double erl_estimator_erl_time_domain(const ErlEstimator *e);

#endif /* ERL_ESTIMATOR_H */
