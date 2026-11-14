/* Matched-filter delay estimation. Chain: Decimator x4 (biquad LP + HP) -> render ring ->
 * MatchedFilter bank (num_filters NLMS cross-correlators) -> lag aggregator (histogram vote) ->
 * ClockdriftDetector; the estimator drains the hop-sized input in 64-sample inner blocks.
 * The chain runs in float32 (-ffp-contract=off). Python reference: modules/delay/. */
#ifndef AEC_DELAY_AEC_H
#define AEC_DELAY_AEC_H

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Memory contract: every array sized by the bank (filter bank, render ring, histogram and its
 * estimate ring) is carved from a caller-owned, 16-byte-aligned block by delay_aec_init(), sized by
 * delay_aec_get_mem_size() for the same (sample_rate, hop_size, num_filters). No heap allocation;
 * the triple is init-time immutable. DA_NUM_FILTERS is only the upper bound used by validation. */

/* DelayQuality */
typedef enum {
    DELAY_QUALITY_COARSE  = 0,
    DELAY_QUALITY_REFINED = 1
} DelayQuality;

/* ClockdriftLevel */
typedef enum {
    CLOCKDRIFT_NONE     = 0,
    CLOCKDRIFT_PROBABLE = 1,
    CLOCKDRIFT_VERIFIED = 2
} ClockdriftLevel;

/* Defaults: num_filters=5, window 32 and alignment shift 24 sub-blocks, excitation limit
 * 150/32768, smoothing 0.7 (fast) / 0.1 (slow), matching threshold 0.3, headroom 32 samples,
 * thresholds (5, 20), consistent-estimate threshold 125. The reported delay is the dominant
 * highest-peak candidate. */
#define DA_DOWN_SAMPLING_FACTOR 4
#define DA_BLOCK_SIZE           64
#define DA_SUB_BLOCK_SIZE       (DA_BLOCK_SIZE / DA_DOWN_SAMPLING_FACTOR)
/* Upper bound of the bank size (delay_num_filters is in [1, DA_NUM_FILTERS], default 5). */
#define DA_NUM_FILTERS          5
#define DA_WINDOW_SIZE_SB       32
#define DA_ALIGNMENT_SHIFT_SB   24
#define DA_FILTER_SIZE          (DA_WINDOW_SIZE_SB * DA_SUB_BLOCK_SIZE)
#define DA_FILTER_INTRA_SHIFT   (DA_ALIGNMENT_SHIFT_SB * DA_SUB_BLOCK_SIZE)
/* Capacity of both aggregator rings (about 1 s of estimates at 64-sample cadence); the window in
 * use is round(feed rate / 64): 250 at 16 kHz (48 kHz is decimated to a 16 kHz feed), 125 at 8 kHz. */
#define DA_HIST_WINDOW          250
#define DA_HEADROOM             (32 / DA_DOWN_SAMPLING_FACTOR)

/* Per-bank-size geometry shared by get_mem_size and init. DA_MAX_FILTER_LAG_FOR(n) sizes the
 * histograms and keeps one filter of headroom over the reachable lag. */
#define DA_RING_CAPACITY_FOR(n) \
    ((DA_WINDOW_SIZE_SB + DA_ALIGNMENT_SHIFT_SB * ((n) - 1) + 1) * DA_SUB_BLOCK_SIZE)
#define DA_MAX_FILTER_LAG_FOR(n) ((n) * DA_FILTER_INTRA_SHIFT + DA_FILTER_SIZE)
#define DA_HP_HIST_SIZE_FOR(n)   (DA_MAX_FILTER_LAG_FOR(n) + 1)
#define DA_THRESH_INITIAL       5
#define DA_THRESH_CONVERGED     20

/* Biquad cascade, at most 4 sections. */
#define DA_BQ_MAX_SECTIONS 4
typedef struct {
    int    n_sections;
    float  b[DA_BQ_MAX_SECTIONS][3];
    float  a[DA_BQ_MAX_SECTIONS][2];
    float  z[DA_BQ_MAX_SECTIONS][2];
} DaBiquad;

/* 48 kHz -> 16 kHz pre-decimation (anti-alias filter, decimate by 3) ahead of the 16 kHz-native
 * estimator; its scratch buffers exist only at 48 kHz. */
#define DA_RESAMPLE48_FACTOR      3
/* Output samples per call are at most ceil(hop / 3). */
#define DA_RESAMPLE48_CAP_FOR(hop) ((hop) / DA_RESAMPLE48_FACTOR + 1)
typedef struct {
    DaBiquad near_lp;
    DaBiquad far_lp;
    int      phase;
} DaResample48;

typedef struct {
    DaBiquad anti_alias;
    DaBiquad noise_reduction;
} DaDecimator;

typedef struct {
    float *buffer;
    int    capacity;
    int    write;
} DaRing;

typedef struct {
    /* Runtime bank size, [1, DA_NUM_FILTERS]. */
    int   num_filters;
    /* Pool-carved, row-major: filters + i * DA_FILTER_SIZE. */
    float *filters;
    int   reported_valid;
    int   reported_lag;
    int   winner_lag_valid;
    int   winner_lag;
} DaMatchedFilter;

typedef struct {
    int *histogram;
    int *ring;
    int hist_size;
    int window;
    int ring_index;
    int candidate;
    int candidate_valid;   /* internal: 0 forces an argmax rescan on the next aggregate() */
} DaHighestPeak;

typedef struct {
    DaHighestPeak highest_peak;
    int           significant_candidate_found;
} DaLagAggregator;

typedef struct {
    int delay_history[3];
    ClockdriftLevel level;
    int stability_counter;
    int stability_reset_hops;
} DaClockdrift;

typedef struct {
    DaDecimator     capture_decimator;
    DaDecimator     render_decimator;
    /* Cached da_decimator_same_coefficients(capture, render); used by the NEON paired decimator. */
    int             decimators_same_coefficients;
    DaRing          render_ring;
    DaMatchedFilter matched_filter;
    DaLagAggregator aggregator;
    DaClockdrift    clockdrift;
    int old_agg_valid;
    int old_agg_delay;
    int consistent_estimate_counter;
    int consistent_estimate_threshold;
    /* Edge buffers (raw samples awaiting a 64-sample block). */
    float capture_pending[DA_BLOCK_SIZE];
    float render_pending[DA_BLOCK_SIZE];
    int   pending_count;
} DaEstimator;

/* Delay shim: the adapter the orchestrator drives */
typedef struct {
    DaEstimator est;
    /* 48 kHz sidechain; rate_factor is 3 at 48000, else 1. */
    int          rate_factor;
    DaResample48 resample48;
    /* Pool-carved [resample_cap]; NULL at rates other than 48 kHz. */
    float       *near16_scratch;
    float       *far16_scratch;
    int          resample_cap;
    /* Latest estimate in the estimator's 16 kHz-equivalent domain. */
    int          latest_valid;
    int          latest_delay;
    DelayQuality latest_quality;
    int          estimate_count;
} DelayAec;

/* Lifecycle: get_mem_size and init take the same (sample_rate, hop_size, num_filters).
 * sample_rate 48000 selects the sidechain; hop_size sizes only its scratch; num_filters outside
 * [1, DA_NUM_FILTERS] is rejected (size 0 / init -1). Reliable reach is
 * (n-1) * DA_FILTER_INTRA_SHIFT + (DA_FILTER_SIZE - 11) samples at 0.25 ms: n=1 -> 125 ms,
 * n=3 -> 317 ms, n=5 -> 509 ms. */

/* Bytes needed; 0 for invalid input or size overflow. Always a multiple of 16. */
size_t delay_aec_get_mem_size(int sample_rate, int hop_size, int num_filters);

/* Reach in samples at sample_rate: an upper bound of delay_aec_estimated_delay() for this bank,
 * 4 * ((n-1) * DA_FILTER_INTRA_SHIFT + DA_FILTER_SIZE - 11), times 3 at 48 kHz. 0 for a
 * non-positive rate or an out-of-range num_filters. */
int delay_aec_max_delay_samples(int sample_rate, int num_filters);

/* Place the estimator in `mem` (>= delay_aec_get_mem_size() bytes, 16-byte aligned). Returns 0,
 * or -1 on NULL, misaligned, undersized or invalid input. */
int delay_aec_init(DelayAec *d, void *mem, size_t bytes,
                   int sample_rate, int hop_size, int num_filters);

/* Clears the signal chain; keeps geometry, pool pointers and bank size. */
void delay_aec_reset(DelayAec *d);

/* Per-hop drive: near (HPF mic) and far (raw reference) are float[hop] at the caller's rate.
 * With run_matched_filter = 0 the hop is fed (decimators and ring) but not analysed, so latest_* is unchanged. */
void delay_aec_accumulate_ex(DelayAec *d, const float *near, const float *far,
                             int hop, int run_matched_filter);

/* accessors */
int    delay_aec_estimated_delay(const DelayAec *d);   /* -1 until the first estimate, else samples at the caller's rate */
float  delay_aec_confidence(const DelayAec *d);        /* 0.0 / 0.5 / 1.0 */
int    delay_aec_is_solid(const DelayAec *d);
int    delay_aec_n_updates(const DelayAec *d);
int    delay_aec_has_clockdrift(const DelayAec *d);

#ifdef __cplusplus
}
#endif

#endif
