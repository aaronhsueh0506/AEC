/* fullband_erle.h — C port of python/modules/state/fullband_erle.py
 * (which mirrors AEC3 fullband_erle_estimator.{cc,h}, single-channel).
 *
 * Fullband ERLE accumulator in log2 units. Two public reads:
 *   - fb_erle_log2()              : EMA in log2 domain (erle_time_domain_log2)
 *   - get_inst_linear_quality_estimate(): rolling [0,1] linear-filter quality
 *
 * PRECISION: float32-by-design (f32 campaign — Python bit-exact parity is
 * retired for this module; drift accepted). Real-pipeline shapes (captured
 * from a real BALANCED doubletalk case):
 *   - The per-bin inputs x2 / y2 / e2 are float32 arrays (length n_freqs=257).
 *   - np.sum-equivalent over a float32 array is numpy PAIRWISE summation in
 *     float32 (NOT naive left-to-right). fb_erle_pairwise_sum() below
 *     replicates that tree shape.
 *   - Every downstream scalar op (acum +=, ratio, log2, EMA, max/min,
 *     quality) is float32 end-to-end now (previously widened to float64 to
 *     mirror Python's `float(np.sum(...))`; that widen is retired). log2()
 *     is now log2f() (fast_log2 takes/returns float).
 *   - converged_filter is a bool.
 *   - _erle_log2 is Optional[float]: a NULL state before the first 6-point
 *     accumulation fires. We carry it as (has_erle_log2, erle_log2).
 *   - get_quality_estimate() returns None while erle_log2 is NULL; the C side
 *     exposes a *valid flag so the caller can mirror the None.
 *   - The one float64 holdout is x2_band_energy_threshold's SOURCE constant
 *     (FBERLE_X2_BAND_ENERGY_THRESHOLD): it is only ever an argument to the
 *     shared, unconverted aec3_per_bin_psd_threshold() double API; the
 *     double result is narrowed to float32 once, at the store in
 *     fb_erle_init().
 *
 * Built with -ffp-contract=off so the per-op float rounding matches across
 * builds.
 *
 * Time constants (see fullband_erle.py docstring):
 *   POINTS_TO_ACCUMULATE = 6 (a count, NOT rescaled).
 *   BLOCKS_TO_HOLD_ERLE  = int(0.4 * HOPS_PER_SECOND) = 40 hops.
 *   0.0004 max/min envelope per update — kept verbatim (NOT rescaled).
 */
#ifndef FULLBAND_ERLE_H
#define FULLBAND_ERLE_H

#include <stddef.h>

/* numpy 1.26 pairwise sum over float32, accumulated in float32. Shared copy of
 * the verified implementation in reverb_frequency_response.c (declared here so
 * parity replay links a single definition from fullband_erle.c). */
float fb_erle_pairwise_sum(const float *a, size_t n);

/* ── Constants mirrored from fullband_erle.py ───────────────────────────────── */
#define FBERLE_EPSILON              1.0e-3f
#define FBERLE_POINTS_TO_ACCUMULATE 6
/* int(0.4 * HOPS_PER_SECOND), HOPS_PER_SECOND = 100 */
#define FBERLE_BLOCKS_TO_HOLD_ERLE  40
/* AEC3 source value; per_bin_psd_threshold(.., hop=160, ref=160) is a no-op.
 * Kept as a double literal (NOT `f`-suffixed): it is only ever passed to the
 * shared, unconverted aec3_per_bin_psd_threshold() double API. */
#define FBERLE_X2_BAND_ENERGY_THRESHOLD 44015068.0f
#define FBERLE_MAXMIN_FORGET        0.0004f
#define FBERLE_QUALITY_ALPHA        0.07f
#define FBERLE_TD_ALPHA             0.05f

/* ── _ErleInstantaneous (FullBandErleEstimator::ErleInstantaneous) ──────────── */
typedef struct {
    int    clamp_zero;          /* _clamp_zero */
    int    clamp_one;           /* _clamp_one  */
    float  quality_alpha;       /* _quality_alpha */
    int    has_erle_log2;       /* _erle_log2 is not None */
    float  erle_log2;           /* _erle_log2 (valid only when has_erle_log2) */
    float  inst_quality_estimate; /* _inst_quality_estimate */
    float  max_erle_log2;       /* _max_erle_log2 */
    float  min_erle_log2;       /* _min_erle_log2 */
    float  y2_acum;             /* _y2_acum */
    float  e2_acum;             /* _e2_acum */
    int    num_points;          /* _num_points */
} ErleInstantaneous;

/* ── FullBandErleEstimator ──────────────────────────────────────────────────── */
typedef struct {
    int    n_freqs;
    float  x2_band_energy_threshold; /* per-bin X^2 gate (already hop-scaled,
                                       * narrowed once from the double helper) */
    float  min_erle_log2;            /* log2(min_erle + EPSILON) */
    float  max_erle_lf_log2;         /* log2(max_erle_l + EPSILON) */
    float  td_alpha;                 /* _td_alpha (0.05f) */
    int    hold_counter_inst_erle;   /* _hold_counter_inst_erle */
    float  erle_time_domain_log2;    /* _erle_time_domain_log2 */
    ErleInstantaneous inst;          /* _instantaneous_erle */
    int    has_linear_filter_quality;/* _linear_filter_quality is not None */
    float  linear_filter_quality;    /* _linear_filter_quality */
} FullBandErleEstimator;

/* Initialise. hop_size scales x2_band_energy_threshold via the per-bin PSD rule
 * (no-op at hop_size=160). min_erle / max_erle_l mirror the Python kwargs. */
void fb_erle_init(FullBandErleEstimator *s, int n_freqs,
                        float min_erle, float max_erle_l, int hop_size);

void fb_erle_reset(FullBandErleEstimator *s);

/* update(x2, y2, e2, converged_filter). x2/y2/e2 are float32 arrays of length
 * n_freqs (the real pipeline dtype). */
void fb_erle_update(FullBandErleEstimator *s,
                          const float *x2, const float *y2, const float *e2,
                          int converged_filter);

/* fb_erle_log2() (= _erle_time_domain_log2). */
float fb_erle_log2(const FullBandErleEstimator *s);

/* get_inst_linear_quality_estimate(). *valid = 0 mirrors a None return. */
float fb_erle_get_inst_linear_quality_estimate(const FullBandErleEstimator *s,
                                                      int *valid);

#endif /* FULLBAND_ERLE_H */
