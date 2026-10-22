/* erl_estimator.h — C port of python/modules/state/erl_estimator.py
 * (the reference per-bin ERL estimator).
 *
 * Per-bin echo return loss (ERL = Y²/X²) with minimum-statistics tracking:
 *   - per-bin _erl[k] for k=1..n_bins-2 (endpoints mirror neighbours)
 *   - a fullband _erl_time_domain scalar.
 * On each downward jump the estimate is nudged 10% toward the new ratio and a
 * hold counter is armed for _HOLD_HOPS (=400) hops; bins whose hold counter has
 * expired double (slow recovery) toward _MAX_ERL each update.
 *
 * PRECISION: float32 (drift versus Python accepted). Shapes (preset=balanced,
 * hop=160):
 *   render_psd / capture_psd : float32 (257,)        — inputs X², Y²
 *   self._erl                : float32 (257,)        — per-bin estimate
 *   self._hold_counters      : int32   (255,)        — sized n_bins-2
 *   self._x2_min             : float32 scalar
 *   self._erl_time_domain    : float32 scalar
 *
 *   Per-bin update, all float32, single rounding per op:
 *     new_erl   = y2[k] / x2[k]                  -> float32
 *     delta     = 0.1f * (new_erl - erl[k])      -> float32
 *     erl[k]    = erl[k] + delta                 -> float32
 *     erl[k]    = max(erl[k], 0.01f)             -> float32 compare
 *     erl[k]    = min(1000.0f, 2.0f*erl[k])      -> float32 compare (doubling)
 *
 *   Fullband: x2_sum/y2_sum are numpy-pairwise float32 sums
 *   (sk_pairwise_sum_tailfold_b_f32); the ratio/EMA/doubling that follow are
 *   float32 end-to-end.
 *
 * Built with -ffp-contract=off so the per-op roundings are not fused.
 */
#ifndef ERL_ESTIMATOR_H
#define ERL_ESTIMATOR_H

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/* ── Constants (erl_estimator.py module level) ───────────────────────────── */
#define ERL_MIN_ERL  0.01f    /* _MIN_ERL */
#define ERL_MAX_ERL  1000.0f  /* _MAX_ERL */
/* The reference X2-minimum constant, passed as the `calibrated_value` argument to
 * the shared aec_per_bin_psd_threshold() helper (aec_scale.h/.c, itself
 * float32-typed and outside this conversion's file set). The float32 result
 * of that call is stored directly at the `x2_min` store in
 * erl_estimator_init(). _HOLD_HOPS = int(4.0 * HOPS_PER_SECOND) = 400. */
#define ERL_X2_MIN  44015068.0f
/* hold_hops (1000 blocks, ~4 s; 400 at hop=160/sr=16000) is computed
 * live in erl_estimator_init() from hop_size/sample_rate -- see the
 * ErlEstimator.hold_hops field below. */

/* ── ErlEstimator ─────────────────────────────────────────────────────────
 * Caller provides backing storage:
 *   erl_storage          : n_bins              (float)
 *   hold_counters_storage: n_bins - 2          (int32_t) */
typedef struct {
    int     startup_hops;          /* _startup_hops               */
    int     n_bins;                /* _n_bins                     */
    int     hold_hops;             /* derived from the live grid (hop/sample rate) */
    float   x2_min;                /* _x2_min (float32; result of
                                     * aec_per_bin_psd_threshold(), itself float32) */
    float   x2_min_total;          /* = x2_min * (float)n_bins, cached at init */
    float  *erl;                   /* _erl,           length n_bins     */
    int    *hold_counters;         /* _hold_counters, length n_bins-2   */
    float   erl_time_domain;       /* _erl_time_domain            */
    int     hold_counter_time_domain;
    int     blocks_since_reset;    /* _blocks_since_reset         */
} ErlEstimator;

/* hop_size feeds aec_per_bin_psd_threshold(44015068.0, hop_size) for x2_min.
 * sample_rate (with hop_size) drives the live hold_hops computation. */
void erl_estimator_init(ErlEstimator *e, int startup_phase_length_hops,
                        int n_bins, int hop_size, int sample_rate,
                        float *erl_storage, int *hold_counters_storage);

void erl_estimator_reset(ErlEstimator *e);

/* update(render_psd=X², capture_psd=Y², converged_filter) */
void erl_estimator_update(ErlEstimator *e,
                          const float *render_psd,
                          const float *capture_psd,
                          int converged_filter);

#ifdef __cplusplus
}
#endif

#endif /* ERL_ESTIMATOR_H */
