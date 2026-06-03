/* fullband_erle.h — C port of python/modules/state/fullband_erle.py
 * (which mirrors AEC3 fullband_erle_estimator.{cc,h}, single-channel).
 *
 * Fullband ERLE accumulator in log2 units. Two public reads:
 *   - fb_erle_log2()              : EMA in log2 domain (erle_time_domain_log2)
 *   - get_inst_linear_quality_estimate(): rolling [0,1] linear-filter quality
 *
 * PARITY (numpy 1.26 -> C), captured from a real BALANCED doubletalk case:
 *   - The per-bin inputs x2 / y2 / e2 are float32 arrays (length n_freqs=257).
 *   - np.sum over a float32 array == numpy PAIRWISE summation in float32 (NOT
 *     naive left-to-right). f32_pairwise_sum() below replicates numpy 1.26.
 *   - float(np.sum(..)) then widens the float32 result to a C double, after
 *     which EVERY downstream scalar op (acum +=, ratio, log2, EMA, max/min,
 *     quality) is in double. _fast_log2 = float(np.log2(double)) is a double.
 *   - converged_filter is a bool.
 *   - _erle_log2 is Optional[float]: a NULL state before the first 6-point
 *     accumulation fires. We carry it as (has_erle_log2, erle_log2).
 *   - get_quality_estimate() returns None while erle_log2 is NULL; the C side
 *     exposes a *valid flag so the caller can mirror the None.
 *
 * Built with -ffp-contract=off so the per-op double rounding matches numpy.
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
#define FBERLE_EPSILON              1.0e-3
#define FBERLE_POINTS_TO_ACCUMULATE 6
/* int(0.4 * HOPS_PER_SECOND), HOPS_PER_SECOND = 100 */
#define FBERLE_BLOCKS_TO_HOLD_ERLE  40
/* AEC3 source value; per_bin_psd_threshold(.., hop=160, ref=160) is a no-op */
#define FBERLE_X2_BAND_ENERGY_THRESHOLD 44015068.0
#define FBERLE_MAXMIN_FORGET        0.0004
#define FBERLE_QUALITY_ALPHA        0.07
#define FBERLE_TD_ALPHA             0.05

/* ── _ErleInstantaneous (FullBandErleEstimator::ErleInstantaneous) ──────────── */
typedef struct {
    int    clamp_zero;          /* _clamp_zero */
    int    clamp_one;           /* _clamp_one  */
    double quality_alpha;       /* _quality_alpha */
    int    has_erle_log2;       /* _erle_log2 is not None */
    double erle_log2;           /* _erle_log2 (valid only when has_erle_log2) */
    double inst_quality_estimate; /* _inst_quality_estimate */
    double max_erle_log2;       /* _max_erle_log2 */
    double min_erle_log2;       /* _min_erle_log2 */
    double y2_acum;             /* _y2_acum */
    double e2_acum;             /* _e2_acum */
    int    num_points;          /* _num_points */
} ErleInstantaneous;

/* ── FullBandErleEstimator ──────────────────────────────────────────────────── */
typedef struct {
    int    n_freqs;
    double x2_band_energy_threshold; /* per-bin X^2 gate (already hop-scaled) */
    double min_erle_log2;            /* log2(min_erle + EPSILON) */
    double max_erle_lf_log2;         /* log2(max_erle_l + EPSILON) */
    double td_alpha;                 /* _td_alpha (0.05) */
    int    hold_counter_inst_erle;   /* _hold_counter_inst_erle */
    double erle_time_domain_log2;    /* _erle_time_domain_log2 */
    ErleInstantaneous inst;          /* _instantaneous_erle */
    int    has_linear_filter_quality;/* _linear_filter_quality is not None */
    double linear_filter_quality;    /* _linear_filter_quality */
} FullBandErleEstimator;

/* Initialise. hop_size scales x2_band_energy_threshold via the per-bin PSD rule
 * (no-op at hop_size=160). min_erle / max_erle_l mirror the Python kwargs. */
void fb_erle_init(FullBandErleEstimator *s, int n_freqs,
                        double min_erle, double max_erle_l, int hop_size);

void fb_erle_reset(FullBandErleEstimator *s);

/* update(x2, y2, e2, converged_filter). x2/y2/e2 are float32 arrays of length
 * n_freqs (the real pipeline dtype). */
void fb_erle_update(FullBandErleEstimator *s,
                          const float *x2, const float *y2, const float *e2,
                          int converged_filter);

/* fb_erle_log2() (= _erle_time_domain_log2). */
double fb_erle_log2(const FullBandErleEstimator *s);

/* get_inst_linear_quality_estimate(). *valid = 0 mirrors a None return. */
double fb_erle_get_inst_linear_quality_estimate(const FullBandErleEstimator *s,
                                                      int *valid);

#endif /* FULLBAND_ERLE_H */
