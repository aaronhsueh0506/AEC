/* reverb_decay_estimator.h — C port of
 * python/modules/residual/reverb_decay_estimator.py
 * (which mirrors AEC3 reverb_decay_estimator.{cc,h}).
 *
 * Adaptive late-reverb decay estimator. DEFAULT-OFF substrate in production
 * (use_adaptive_decay=False → never constructed in the balanced preset), but
 * ported faithfully so call sites exist and the load-bearing accumulate /
 * _analyze_filter / regression math is byte-equal to the numpy 1.26 reference.
 *
 * Two mutually-exclusive operating paths, selected per-instance:
 *   - Legacy partition path: update(partition_energies=...).  Linear regression
 *     over log2(Σ_k |W_p[k]|²) per partition.  Inputs are float32 at the call
 *     site; the regression math casts to double (matches Python .astype(f64)).
 *   - AEC3-strict TD path:   update(time_domain_filter=...).  BlockEnergy +
 *     EarlyReverbLengthEstimator + LateReverbLinearRegressor on the 64-sample
 *     sub-blocked impulse response.  TD taps are float32 at the call site, cast
 *     to double internally.
 *
 * PARITY NOTES (numpy 1.26 → C):
 *   - Python's np.sum over a 1-D contiguous array uses 8-way-unrolled pairwise
 *     summation for n<=128.  rde_pairwise_sum() reproduces that exactly; naive
 *     left-fold would NOT be bit-equal (see parity harness).  Used for the
 *     BlockEnergyAverage 64-tap mean and every legacy-regression Σ.
 *   - np.log2 (vectorised) and math.log2 (scalar) are bit-identical to libm
 *     log2() on this platform; math.pow(2.0,x) == pow(2.0,x).
 *   - _symmetric_arithmetic_sum(N) = N*(N²-1)/12 in double; N=0 yields -0.0
 *     (0.0*-1.0/12.0), reproduced verbatim.
 *
 * Build with -ffp-contract=off (mandatory for byte-equal).
 */
#ifndef REVERB_DECAY_ESTIMATOR_H
#define REVERB_DECAY_ESTIMATOR_H

#include <stddef.h>

#define RDE_K_EARLY_REVERB_MIN_SIZE_BLOCKS 3
#define RDE_K_BLOCKS_PER_SECTION           6
#define RDE_K_MIN_DECAY                    0.02
#define RDE_K_MAX_DECAY                    0.95
#define RDE_K_FFT_LENGTH_BY_2              64   /* AEC3 kFftLengthBy2 */
#define RDE_K_FFT_LENGTH_BY_2_LOG2         6    /* log2(64) */

/* log2(1.1) and log2(0.8) — AEC3 cc:372-374 */
#define RDE_LOG2_1_1   0.13750352374993502
#define RDE_LOG2_0_8  (-0.32192809488736229)

/* --- _LateReverbLinearRegressor (cc:277-304) --- */
typedef struct {
    int    N;        /* target number of data points */
    int    n;        /* accumulated count */
    double nz;       /* Σ z·(i - mid) */
    double nn;       /* Σ (i - mid)²  (closed form) */
    double count;    /* running (i - mid) value */
} RdeLateRegressor;

/* --- _EarlyReverbLengthEstimator (cc:306-404) --- */
typedef struct {
    double *numerators;        /* owned; length n_storage */
    double *numerators_smooth; /* owned; length n_storage */
    int     n_storage;         /* max(0, max_blocks - kBlocksPerSection) */
    int     n_sections;
    int     coefficients_counter;
    int     block_counter;
    double  first_point;
} RdeEarlyEstimator;

/* --- ReverbDecayEstimator --- */
typedef struct {
    int    n_partitions;
    int    hop_size;
    double default_decay;
    double mild_decay;
    int    use_adaptive;          /* bool */
    int    use_aec3_block_energy; /* bool */
    double decay;
    double smoothing_constant;

    /* AEC3-strict state */
    int    filter_length_blocks;
    int    has_early;             /* whether early estimator is allocated */
    RdeEarlyEstimator early;
    RdeLateRegressor  late;
    int    late_reverb_start;
    int    late_reverb_end;
    int    block_to_analyze;
    int    estimation_region_identified;       /* bool */
    int    estimation_region_candidate_size;
    double *previous_gains;       /* owned; length filter_length_blocks */
    int     n_previous_gains;
    double  tail_gain;
} ReverbDecayEstimator;

/* ── pairwise-sum helper (numpy 1.26 parity) ───────────────────────────── */
double rde_pairwise_sum(const double *a, int n);

/* ── construction / lifecycle ──────────────────────────────────────────── */
void   rde_init(ReverbDecayEstimator *r, int n_partitions, int hop_size,
                double default_decay, double mild_decay,
                int use_adaptive, int use_aec3_block_energy);
void   rde_free(ReverbDecayEstimator *r);   /* releases owned buffers */
void   rde_reset(ReverbDecayEstimator *r);

double rde_decay_value(const ReverbDecayEstimator *r);
double rde_decay(const ReverbDecayEstimator *r, int mild);

/* ── update entry points ───────────────────────────────────────────────── */
/* Legacy partition path. partition_energies length = n_partitions (float32 at
 * the call site → pass as const float*). filter_quality<0 means "None" (Python
 * treats None as 0.0); pass filter_quality_is_none=1 to signal None. */
void   rde_update_legacy(ReverbDecayEstimator *r,
                         const float *partition_energies,
                         double filter_quality, int filter_quality_is_none,
                         int filter_delay_blocks,
                         int usable_linear_filter, int stationary_signal);

/* AEC3-strict TD path. td_filter length = td_size (float32 at the call site). */
void   rde_update_aec3_strict(ReverbDecayEstimator *r,
                              const float *td_filter, int td_size,
                              double filter_quality, int filter_quality_is_none,
                              int filter_delay_blocks,
                              int usable_linear_filter, int stationary_signal);

#endif /* REVERB_DECAY_ESTIMATOR_H */
